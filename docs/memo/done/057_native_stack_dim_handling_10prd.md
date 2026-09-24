# Stack 次元のネイティブ対応(Reshape ハックを廃する根本対応)

## ゴール

フレームスタック(stack_count>1)の入力を、ブランチ先頭に `Flatten > Reshape` の
小細工を挟まずに扱えるようにする。具体的には **Stack(時間)次元を第一級の軸として
ダミー forward と実データの両方で一致させ**、時間軸 Conv1D を自然な構造定義で書けるようにする。

> 暫定対応として追加済みの Reshape モジュール(`010_reshape_module_*`)は、本対応が入れば不要になる。

## TL;DR

不整合の原因は1箇所だけ。**stacked obs の spec 表現が「stack を特徴次元 dim0 に畳む」**
(`shape[0] *= stack_count`、[8]→[32])であるのに対し、**実データ(行動時・学習時の両経路)は
stack を独立軸として持つ**((N, 4, 8))。ダミーだけが (1, 32) でランクが違う。

→ spec 表現を **`[stack, *orig_shape]`**([4, 8])に変えてダミーを (1, 4, 8) にすれば、
ダミー = 実データになり、Flatten/Reshape のブリッジが一切不要になる。
波及は `network_obs_spec`(ダミー生成専用のローカルコピー)に閉じる。

## 現状の規約と不整合

| 経路 | 生成箇所 | Stack の扱い | 形状(LunarLander) |
|---|---|---|---|
| spec/ダミー | `default_dqn_agent.cpp:117-134` → `nn_impl.cpp:1078-1092` | **dim0 に畳む**(`shape[0]*=stack`) | ダミー **(1, 32)** 2D |
| 実データ(行動時) | `stacker.cpp:55-58` DictFrameStacker | **dim1 に軸を挿入** | **(N, 4, 8)** 3D |
| 実データ(学習時) | `replay_buffer_impl.cpp:425-445` RingSlice | **時間軸を保持**(stack_count==1 のときだけ squeeze) | **(B, 4, 8)** 3D |

両ランタイム経路は (·, stack, *orig) で一致しているのに、**ダミーだけ畳まれている**。
現状はこの差を **ブランチ先頭ブロックがブリッジ**して吸収している:

- ベクトル + MLP: 先頭 `Flatten` が実データ (N,4,8)→(N,32) に畳んでダミーに合わせる
  (`nn_modules.cpp:255-262`、`flatten(1)`)。
- 画像 + CNN: 先頭 `StackMerge` が実データ 5D (N,S,C,H,W)→(N,S*C,H,W) に畳む
  (`nn_modules.cpp:281-291`)。ダミーは畳んだ spec から直接 4D で作られる。

## なぜ時間軸 Conv を阻むか

時間軸 Conv/Permute は **stack 軸を独立に保ったまま**演算する必要があるが、
ダミーにその軸が存在しない。Lazy 初期化の初回 forward が (1,32) の 2D に対して
3次元 permute を当てて落ちる(`permute: input.dim()=2 != len(dims)=3`)。
Reshape ハックは (N,32)→(N,4,8) と無理やり軸を復元して凌いでいるだけ。

## 維持すべき不変条件

本対応後も、以下を必ず満たすこと:

1. **ダミー forward と実データ(行動時・学習時)のランク・各軸の意味が完全一致**する。
2. 既存の MLP 構成(`Flatten` 先頭)が挙動不変で動く。
3. 既存の画像構成(`StackMerge` 先頭)が挙動不変で動く。

## 提案する正しい設計

### 1. spec 表現を「stack 独立軸」に統一

`default_dqn_agent.cpp:119-134` の畳み込みを、**stack 軸の挿入**に変更する:

```cpp
// 現状: stack を特徴次元に畳む
//   kv.second.shape[0] *= config_.stucker.stack_count;   // [8] -> [32]

// 変更案: stack を先頭に独立軸として挿入
kv.second.shape.insert(kv.second.shape.begin(), config_.stucker.stack_count);  // [8] -> [4, 8]
```

`nn_impl.cpp:1078-1092` のダミー生成はバッチ次元を先頭に足すだけなので、
spec [4,8] からダミー **(1, 4, 8)** が自然に出る(ビルダー側は原則無改修)。
→ 実データ (N,4,8) と一致。

### 2. ブランチ先頭ブロックが「意図」を宣言する(統一ルール)

stacked obs は常に **(N, stack, \*orig_shape)** でブランチに入る、という不変を確立し、
先頭ブロックで畳み方を明示する:

| 先頭ブロック | 意味 | 用途 |
|---|---|---|
| `Flatten` | 全軸を平坦化 (N, stack\*feat) | MLP |
| `StackMerge` | stack をチャネルに畳む (N, stack\*C, H, W) | 画像 CNN |
| `Permute`(+`Conv1d`) | stack を時間軸として演算 | 時間軸 Conv |

これで Reshape は不要。時間軸 Conv はこう書ける:

```
net.branch.[main_feature].structure = Conv1D_Permute > TConv64 > SiLU > Flatten > MLP_FC1 > ReLU > MLP_FC2 > ReLU
# (N,4,8) -Permute(0 2 1)-> (N,8,4) -Conv1d k2-> (N,64,3) -Flatten-> (N,192) -> MLP
```

## 実装ポイント(局所)

1. **spec 表現変更**: `default_dqn_agent.cpp:119-134` を「dim0 倍」→「先頭軸挿入」に
   (上記コード)。`RainbowAgent` 等、同じ畳み込みをしている箇所があれば横展開。
2. **ダミー生成**: `nn_impl.cpp` NetworkBuilder は原則無改修で追従するはず。要確認。
3. **画像経路の整合**: ダミーが stack 軸込み 5D (1,S,C,H,W) になるので、`StackMerge` の
   `if (x.dim()==5)` 分岐がダミー pass でも発火し、実データと同一になる(むしろ一貫性向上)。
   ただし既存挙動が変わるため画像構成は要再検証(下記)。
4. **config 移行**: ベクトル MLP は先頭 `Flatten` のままで不変。画像は先頭 `StackMerge` を
   担保。時間軸 Conv は Reshape を外して `Permute` 始まりに。

## 波及範囲

### 検証済み(影響なし)

- **`network_obs_spec` はローカルコピー**(`default_dqn_agent.cpp:117`)で、用途は
  NetworkModel 構築(`:140`)のみ。env_spec 本体・replay・reward_scaler に波及しない。
- **obs_norm は raw spec から生成**(`:81`、`env_spec.state_spec`)。stack 前の [8] を扱うため
  本変更と無関係。
- **replay buffer は実行時に時間軸を保持済み**(`replay_buffer_impl.cpp:441-445`)。
  目標表現と既に一致。

### 要再検証

- **画像/StackMerge 経路**: ダミーが 4D→5D に変わるため、StackMerge を使う既存構成
  (DropMerge ViT 等)で出力形状・パラメータ数が一致するか確認。
  ※ stack_count==1 の構成は squeeze で軸が消えるため影響軽微。実質 stack_count>1 を
    使う構成(現状ほぼ LunarLander のみ)が主対象。
- **nn_viz / metrics の input_shape ログ**: 表示が [32]→[4,8] に変わる(表示のみ)。

## 設計判断: 軸規約 vs spec メタデータ

- **案A(推奨): 暗黙の軸規約**。「batch の次が stack 軸」という規約だけで運用。実装最小。
  欠点は、ブロックが stack 軸を構造的に区別できない(Flatten/StackMerge/Permute の
  使い分けは構成者責任)。現状の設計思想(構造は config 宣言)と整合。
- **案B: spec に `stack_count` メタデータ**を持たせ、ブロック/ビルダーが introspect 可能に。
  堅牢だが TensorSpec 改修が広く、今回の局所性を損なう。将来 stack 依存ブロックが
  増えたら検討。

→ まず案Aで実装し、不変条件をテストで固定する。

## 移行計画

1. spec 表現変更(案A)を実装。MLP ベースライン(run_20260612-015116 相当)で
   **学習結果がビット単位で不変**なことを確認(Flatten 先頭なので一致するはず)。
2. 画像構成(StackMerge)を1つ選び、変更前後で net.body.json と reward 曲線を比較。
3. 時間軸 Conv 構成から Reshape を除去し、`Permute` 始まりで起動・形状確認。
4. Reshape モジュールを deprecated 扱いに(当面は残置可)。

## 関連

- 二重レイアウト規約の発端: `nn_impl.cpp` NetworkBuilder / `stacker.cpp` /
  `default_dqn_agent.cpp` / `replay_buffer_impl.cpp`
- 収束改善の本命は別軸(eval_policy のリスク回避 `uqe_use_tail_mean=false`+低 tau)
