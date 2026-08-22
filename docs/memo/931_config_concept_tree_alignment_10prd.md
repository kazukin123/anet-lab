# 概念ツリーと設定ツリーの一致（namespace 横断の合成単位）暫定 PRD

> 番号 999。backlog / 検討草案。
> 状態: **解決案を選定しない**。問題の一般化・同型事例の棚卸し・解決方向の候補併記までを行い、選定は次段階の grill に送る。本書は実装着手を意味しない。
> 起点: QR/IQN の切替が `net.$` と `R.quantile_mode` の 2 行セット操作になる件（`apps/runner/config/DropMerge.txt:126-130`）。
> 「別々の概念として動く NN と Agent が実際には一体で動くという責務が、そのまま設定階層に表れている」（LightZero で Model と MCTSTree をセットで Policy として扱うことに相当）という見立てを、機構レベルまで掘って記録する。
> 関連: [adr/0018-iqn-via-bind-product-dag.md](../adr/0018-iqn-via-bind-product-dag.md)、[adr/0027-eval-definition-schedule-separation.md](../adr/0027-eval-definition-schedule-separation.md)、[adr/0021-run-classification-by-workspace-folder.md](../adr/0021-run-classification-by-workspace-folder.md)、[adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、[029_config_profile_param_interp_10prd.md](done/029_config_profile_param_interp_10prd.md)、[033_imagecls_bf16_head_10prd.md](done/033_imagecls_bf16_head_10prd.md)、[052_eval_schedule_separation_10prd.md](done/052_eval_schedule_separation_10prd.md)、`CONTEXT.md`「Module Config」「Property」「configured eval tag」「eval schedule」。
> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。本書は self-contained。実装時は行番号ではなく、近傍のシンボル名・キー名で再検索する。

---

## 1. Problem Statement

### 1.1 症状

QR と IQN を切り替えるには、離れた 2 行を必ずセットで操作する必要がある。

```
#net.$ = net.qr
#R.quantile_mode = qr

net.$ = net.iqn
R.quantile_mode = iqn
```

同じイディオムが 3 つの env config で反復されている。

| ファイル | 状況 |
|---|---|
| `DropMerge.txt:126-130` | 2 行ペアが隣接。最も素直な形 |
| `LunarLander.txt:37-41` | qr 側（`net.$` → `quantile_mode`）と iqn 側（`quantile_mode` → `net.$`）で**2 行の順序すら揃っていない** |
| `Atari.txt:34-35` / `:187-188` / `:232` | 切替点が「基本設定」「実験」「Env依存 Agent設定」の**3 章・約 200 行に分散**し、しかも上書き層が `A` と `R` で不揃い |

`Atari.txt:33` のコメントが、機構で表現できていないことをそのまま自白している。

```
# quantile 系の配線選択。agent 側の quantile_mode(A/R 層)と揃えること。
net.$ = net.qr
```

「揃えること」と人間へ指示している時点で、それは設定の構造で表現できていない。

### 1.2 一般化 — 機構側の原因

`.$` によるオーバーレイ合成は、**マージ先が LHS の prefix に固定される**。

```
DefaultDQNAgent.$ = DefaultDQNAgent.baseline > AS.fast > A > R > X
```

は「`DefaultDQNAgent.*` という 1 つの namespace へ、右のグループ群を左から順に重ねる」という意味しか持てない。したがって**オーバーレイは常に単一 namespace の中で閉じる**。

「IQN」という概念は `net.*`（配線）と `DefaultDQNAgent.*`（Head / Learner / policy の選択）の 2 つの namespace にまたがる。よって**その概念を束ねる設定ノードが原理的に作れない**。2 行必要なのは記述の怠慢ではなく、合成機構の表現力の限界である。

これは QR/IQN 固有の問題ではない。**概念が複数 namespace にまたがるたびに同じことが起きる**（§2 で 12 件）。

### 1.3 現状の事実（コード確認済み）

合成機構は `ConfigManager::AutoMerge`（[config.cpp](../../core/anet-core/src/config.cpp) の `MERGE_KEYWORD = ".$"` 以下、約 60 行）が全て。

| 事実 | 根拠 |
|---|---|
| マージ先 = LHS prefix 固定 | `RemoveSuffix(merge_key, MERGE_KEYWORD)` で `net.$` → `net` |
| merge キーの判定は `.$` サフィックスのみ | `EndsWith(key, MERGE_KEYWORD)`。**裸の `$` は merge キーとして認識されない** |
| `>` 区切り、左から右へ適用、右が勝つ | `Split(merge_val, { ">" }, true)` の順次適用 |
| 子孫のみ対象（`backend.deterministic` が `backend.deterministic_algorithms` を巻き込まない） | `merge_target_key + "."` の `StartsWith` 判定 |
| **単一パス。`.$` の連鎖は解決されない** | 内側ループが展開前の `map` を参照する（`for (const auto& kv2 : map)`）。マージ元がさらに `.$` を持っていても展開されない |
| **マージ結果は行位置と無関係に、素の同名キーを常に上書きする** | 非マージキーを先に `new_map` へコピーし、後からマージ結果を `Set` する。よって `DefaultDQNAgent.quantile_mode = qr` を直書きしても、`R.quantile_mode` があれば必ず負ける |
| マージ元キーは最終 ConfigData に残る | `//new_map.Erase(key2); // 2回目のマージで困るので消さない` |
| `R.` / `A.` / `X.` / `E.` / `M.` / `P.` はスキーマも登録もない**自由記述の名前空間**。C++ 側に該当文字列は存在しない | `AutoMerge` は文字列 prefix 一致しか見ない |
| 未定義レイヤ・誤字レイヤは**無警告 no-op** | 該当キーが 0 件なら黙って何も起きない。実際 `P.` は Atari / LunarLander / GridMaze で定義 0 件のまま `.$` チェーンに書かれている |

`quantile_mode` と `net.*` を結ぶ制約はコード上に存在しない。両者は**同じ関数で同じフラットマップを別 prefix で切り出しているだけ**である。

```cpp
// DefaultDQNAgentFactory::CreateAgent
DefaultDQNAgentConfig config(config_data);        // prefix "DefaultDQNAgent"
anet::nn::NetworkConfig net_config(config_data);  // prefix "net"
```

組み合わせ検証は `DefaultDQNAgentConfig` のコンストラクタ末尾（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) の `ANET_READ_CONFIG(config_data, quantile_mode);` 以降、約 90 行）にほぼ全て集約されているが、**`net.*` ツリーへは一切届いていない**。

### 1.4 不整合時の実挙動

| 組み合わせ | 起きること |
|---|---|
| `quantile_mode=iqn` × `net.$=net.qr` | `taus` は input_specs へ能動追加されるが誰も bind しない → `NetworkBodyBuilder: input key 'taus' is present in input_specs but not bound by any branch...` の **WARN 1 行**。その後 `IQNDuelingHead expected rank-3 value and advantage inputs (B,K,D), but value_shape=...` で落ちる |
| `quantile_mode=qr` × `net.$=net.iqn` | `taus` が input_specs に無いのに `net.branch.[tau_embedding].bind = taus` → `NetworkBodyBuilder: Branch 'tau_embedding' requires unknown input key 'taus'. Check your 'bind' configuration.` |
| `net.$` 行の書き忘れ | `net.branch.[value_stream].bind` が存在せず、**静かに別グラフが構築される** |

**3 つとも `quantile_mode` にも `net.$` にも言及しない**。設定境界での検証ではなく、NetworkModel 構築まで進んでから形状 / キー欠落で落ちる。

### 1.5 これは意図された設計である

ADR 0018 が明記している。

> 検証は AGENTS.md「汎用機構と利用側の責任境界」に従い、フレームワークは局所契約のみを見る（略）DAG の推移的到達性検証は行わず、taus が最終出力へ意図どおり寄与することは NN 設定者の責任とする。

さらに同 ADR の Consequences は、**2 行での切替を利点として記述している**。

> Head は「最終的な Q 分位の出力層のみ」という既存の Body/Head 役割分担が IQN でも維持され、QR との切替が quantile_mode（none/qr/iqn）と NN 設定の差し替えだけで完結する。

つまり本件はバグではなく、**ADR レベルで確定した責任境界の帰結**である。動かすなら ADR 0018 の再検討が入口になる。

---

## 2. 同時変更セットの棚卸し（12 件）

「2 箇所以上を同時に変えないと壊れる / 意味を成さない」設定の組。

| # | 概念 | またがる namespace | 検証 |
|---|---|---|---|
| 1 | 分布表現（QR/IQN） | `net.*` × `DefaultDQNAgent.*` × `metrics.*` | なし（形状エラーで間接検出） |
| 2 | dueling V/A ストリーム | `net.body.output.*` × `net.branch.*` × `DefaultDQNAgent.use_dueling_net` | なし |
| 3 | frame stack 段数 | `DefaultDQNAgent.stucker.*` × `net.block.*.reshape` × ReplayBuffer history margin | 一部（ADR 0024） |
| 4 | env グリッド寸法 | `DropMergeEnv.*` × `net.branch.*` × `net.block.*` | なし |
| 5 | eval タグ | `train.eval.*` × `train.eval_schedule.*` × `app.*.eval_panel.*` × `metrics.*` | **あり（ADR 0027 で分離済み）** |
| 6 | BF16 / AMP | `BF16.agent.*` × `*.use_amp`（4 箇所） × `net.block.*.force_fp32` | なし |
| 7 | TBO | `learner.use_tbo` × `reward_scaler.*` × `target_policy.uqe_use_tail_mean` | WARN のみ（後者は無し） |
| 8 | optimistic target | `use_optimistic_target` × `target_policy.*` ブロック全行 | 該当せず（構造で解決済み） |
| 9 | 更新頻度 | `update_interval` × `replay_ratio` × `update_warmup_steps` | なし（コメントで導出式を指示） |
| 10 | 学習予算 | `app.*.exp_pause_step` × `eps_decay_steps` × `per_beta_step` × `update_warmup_steps` | なし（章コメントで指示） |
| 11 | 並列度 | `train.num_envs` × `train_policy.use_spatial_exploration` | なし（閾値 32 がコメントのみ） |
| 12 | 分位数 | `qr.num_quantiles` × `QValuePanel.smooth_radius` | なし（導出式が MEMO コメント） |

### 2.1 分布表現（QR/IQN）— 本件の起点

§1 のとおり `net.$` × `quantile_mode` の 2 行。さらに**隠れた第 3・第 4 の依存**がある。

- **tau 射影の次元**: `LunarLander.txt:205` `net.block.[TauProj].linear.out_features = 128 # main_feature 最終次元 128 と一致させる`、`Atari.txt:393` `net.block.[AtariIQNTauProj512].linear.out_features = 512 # main_feature 最終次元と一致必須`。backbone（`net.branch.[main_feature].$`）を差し替えると追随が要る
- **metrics 合成行**: `LunarLander.txt:350` `metrics.scalar.$ = metrics.scalar.baseline > metrics.scalar.iqn_search_p0 > M`

つまり実質 3〜4 行同時変更である。

### 2.2 dueling V/A ストリーム

`LunarLander.txt:194` に、config 自身が手順を書いている。

```
# --- Dueling V/A: [代替] 明示的な V/A ストリーム
#   切替時は上の output.[features] をコメントアウトし、下の6行を有効化する
```

`Atari.txt:407` は、この選択が §2.1 に従属することも明記している。

```
# dueling は明示 V/A ストリーム方式(IQN 切替を net.$ で成立させるため)
```

`DefaultDQNAgent.use_dueling_net`（Head factory の 4 択）と `net.body.output.[value_feature]/[adv_feature]`（配線）は独立に書けてしまう。

### 2.3 frame stack 段数

`DefaultDQNAgent.baseline.stucker.stack_count = 4` に対し、NN 側で手書きの対応が必要。

```
net.block.[ReS4].reshape.dims = 4 -1      # (N,32) -> (N,4,8) ※4=stack_count、残り8次元は-1で追従
```

`LunarLander.txt:163-164` と `:262-263` は「4 ↔ 8」を**離れた 2 箇所のコメント行ペアで同時に切り替える**運用になっている。第 3 の消費者が ReplayBuffer の history margin（`stack_count - 1`）で、こちらは ADR 0024 で決定済み。

`CONTEXT.md`「history margin」の `_Avoid_: stack margin（NN構成の語と紛れる）` は、**同じ stack 語が NN と replay で別概念**という認識が既に文書化されていることを示す。

### 2.4 env グリッド寸法

盤面サイズ 58×46 が最低 4 箇所に散在する。

| 箇所 | 内容 |
|---|---|
| `DropMerge.txt:32` | `DropMergeEnv.$ = DropMergeEnv.baseline > DropMergeEnv.G5846 > ...` |
| `DropMerge.txt:61` | `net.branch.[vector_feature].structure = Embed5846_v2` |
| `DropMerge.txt:899-900` | `net.block.[Embed5846_v2].embed.grid_width = 58` / `.grid_height = 46` |
| `DropMerge.txt:1447-1448` | `DropMergeEnv.G5846.grid_cols = 58` / `.grid_rows = 46` |

GridMaze でも同型で、可視化解像度（`image.phm.*.heatmap.width/height`）が env 寸法に従属している。

### 2.5 eval タグ — 唯一の解決済み事例

定義 × schedule × EvalPanel × metrics 参照の 4 namespace が同一タグ名で結合する。

```
train.eval_schedule.[eval1].interval = 100
train.eval.[eval1].run_mode = eval1
train.eval_schedule.[eval1].use_background = true
train.eval.[eval1].clone_model = true
app.online.eval_panel.eval_config_tag = eval1
```

**ここは ADR 0027 / PRD 052 で一度分離手術が入った領域**であり、本件の直接のテンプレートになる（§3）。

なお ADR 0027 が `interval=0`（明示 OFF）を残した理由 —「オーバーレイ構造がある限り、無効化を値で表現する手段は必須」— は、本件のどの案を採っても効いてくる制約である。

### 2.6 BF16 / AMP

3 ツリーにまたがる。

```
DefaultDQNAgent.$ = DefaultDQNAgent.baseline > AS.heavy > A > BF16.agent > R
BF16.agent.learner.use_amp = true
BF16.agent.learner.use_amp_bf16 = true
BF16.agent.learner.adam_eps = 1e-4   # AMP有効時は大きめにする
#net.block.[LN512].force_fp32 = false
```

`use_amp` / `use_amp_bf16` は train_policy / eval_policy / target_policy / learner の**4 箇所に別々に存在**する。ImageCls は別語彙（`ImageClsAgent.bf16.{enabled,learner,actor}` + `force_fp32` 22 行）を使っており、同一概念に 2 つの設定語彙がある。

### 2.7 TBO

```
R.reward_scaler.constant_scale = 0.01
R.learner.use_tbo = true
R.target_policy.uqe_use_tail_mean = false   # TBO整合: h単調は単一分位点でargmax不変、tail_mean(上側平均)は対象外
```

「TBO を有効にしたら tail_mean は使えない」という制約が `learner.*` と `target_policy.*` に分かれている。前者（TBO × dynamic_scaling）は WARN 実装あり、後者は**検証なし**。

### 2.8 optimistic target — 参考事例（構造で解決済み）

`use_optimistic_target = true` のとき `target_policy = train_policy` の丸ごとコピーになる。つまり**1 つの bool が別ブロック全行の既定値を差し替える**。

これは「2 行問題」ではなく、**片方から他方を導出することで 1 行に畳んだ既存例**である。`agent.txt` に 12 行の解説コメントが付いているのは、機構が暗黙すぎることの裏返しでもある。§4 の「導出」方向を評価するときの実例になる。

### 2.9 更新頻度

- `update_interval`（環境ステップ毎）と `replay_ratio` は排他（`# 負数ではupdate_interval側を使う`）だが**両方常に書かれている**
- `update_warmup_steps` は「`replay_batch_size * 32` もしくは `num_envs * エピソードの平均長`、どちらか大きい方」とコメントで導出式を指示

`common.txt:42-51` の `AS.fast|balance|heavy` は `replay_batch_size` + `replay_ratio` の 2 値セットを束ねる**既存の解決策**である（ただし単一 namespace 内。§3）。

### 2.10 学習予算

全 env config が専用章を持つ。

```
# 予算 2.5M steps(10M frames)に紐づく値をここに集約。予算変更時はセットで見直す。
A.train_policy.eps_decay_steps = 250,000
A.learner.per_beta_step = 2,500,000
A.learner.update_warmup_steps = 20,000
```

さらに env プリセットとのセット変更も明記されている。

```
# Atari-100k 予算(400k frames = 100k transitions)。AtariEnv.100k とセットで有効化し、
# ステップ数依存 章(eps_decay/per_beta/warmup)も 100k 相応へ見直すこと。
```

LunarLander は同じ予算値を `app.online.exp_pause_step` として 3 回書いている。

### 2.11 並列度

`# ApeXのVectorized Explorationを有効にするか。 num_envs<32の場合は不安定化するのでfalse推奨。` — `train.num_envs` と `DefaultDQNAgent.train_policy.use_spatial_exploration` の関係が**コメントでしか表現されていない**。

### 2.12 分位数と可視化

```
## MEMO: smooth_radius
# 計算式: Bins (510) ÷ Quantile数 (N) 程度が目安。
```

`qr.num_quantiles` を変えると `QValuePanel.smooth_radius` が意味を失う。導出式が MEMO コメントに留まる。

---

## 3. この repo が既に持つ解法（確定済み前提。ここは再議論しない）

本件を新規発明として扱わない。同種の問題を潰した先例が 4 つある。

| 先例 | 何を潰したか | 手口 |
|---|---|---|
| **ADR 0027 / PRD 052** eval 定義 / schedule 分離 | 「宣言＝起動」のもつれ | **namespace 分離 + 名前参照**。定義は純粋（書いても何もインスタンス化されない）、駆動は別 namespace が名前で参照する |
| **PRD 029** `net.config_profile` | 「18 ブロックに droppath を手書き」 | **スカラー 1 個の補間ポリシー + 構築時の自動展開**。設定の冗長性を機構で潰した最も近い先例 |
| **PRD 033** ImageCls Head 化 | BF16 と FP32 保護の組み合わせ責務 | **構造で保証**（Head=FP32 を型で担保）し、設定を「有効無効＋適用箇所」だけに絞った |
| **`AS.fast\|balance\|heavy`** | `replay_batch_size` + `replay_ratio` の 2 値セット | **束ねる名前を作る**。ただし単一 namespace 内に閉じている |

「カタログ（純粋な定義）+ 束ね（名前参照）」という分離は、この repo に既に 2 例ある。

- `DatasetKey` → `ImageDataset`（共有される定義実体） vs `ImageDataSource`（Env が専有する束ね）
- `configured eval tag`（`train.eval.[tag]`、純粋な定義） vs `eval schedule`（`train.eval_schedule.[tag]`、名前参照して駆動）

**`AS.fast` の「束ねる名前」を namespace 横断へ一般化したものが §4.1、「カタログ + 束ね」を net × agent へ適用したものが §4.2** にあたる。

---

## 4. 解決方向の候補（併記。本書では選定しない）

### 4.1 案1: concept overlay（root スコープ合成）

```
$ = IQN

IQN.net.$ = net.iqn
IQN.DefaultDQNAgent.quantile_mode = iqn
```

`.$` の意味を「マージ先 = LHS prefix。**prefix が空なら root**」へ一般化する。concept overlay の中身は root から見た絶対キーになるので、1 つのオーバーレイが複数 namespace へ同時に展開できる。

| 観点 | 内容 |
|---|---|
| 何を変えるか | `AutoMerge` に root スコープ展開と連鎖解決を追加。設定ファイルは追加のみ |
| 解決すること | 切替が 1 行になる。**概念ノードを設定ツリーに新設できる**ので概念ツリーとの一致が literal に達成できる |
| 解決しないこと | concept overlay を使わず手書きすれば不整合は依然作れる。検証は別途必要（案3 と組み合わせる余地） |
| コスト | 既存キーの rename ゼロ。既存 config は 1 文字も変えずに動く |

**前提となる機構改修（2 点。どちらも必須）**

この案は現行の `AutoMerge` では成立しない。壁が 2 つある。

- **壁1: root スコープのキーが merge キーとして認識されない。** merge キーの判定は `EndsWith(key, MERGE_KEYWORD)`（`".$"`）のみ。裸の `$` は `".$"` で終わらないため、ただの普通のキーとして無視される。加えて `target_key = base_key + key_suffix` の `key_suffix` は先頭ドット付き（`.yyy`）なので、`base_key` が空文字のとき先頭ドットの処理が要る
- **壁2: 単一パスなので二重に動かない。** root 合成が展開するのは `IQN.net.$ = net.iqn` → `net.$ = net.iqn` という**行を書き出すところまで**。その `net.$` をさらに展開するには 2 周目が要るが、内側ループは展開前の `map` を読む（`for (const auto& kv2 : map)`）ため 1 周で終わる。**不動点まで反復する（深さ上限 + 循環検出付き）改修が要る**

壁2 の解消は単独でも価値がある。現状は `net.branch.[main_feature].$` のマージ元がさらに `.$` を持っていても解決されないため、**レイヤ設計を 1 段でも深くすると即座に破綻する**。

### 4.2 案2: 所有の移動（カタログ / 束ね分離）

```
DefaultDQNAgent.baseline.net.$ = net.iqn
```

概念ツリー＝設定ツリーを literal に達成する正攻法。「NN は Agent の持ち物である」という実態を設定階層で表現する。

`net.*` は**純粋な定義カタログ**としてトップレベルに残し（複数 Agent / 可視化から参照されるため）、Agent 側が名前参照で束ねる。これは §3 の「カタログ + 束ね」パターンそのもの。

| 観点 | 内容 |
|---|---|
| 何を変えるか | `NetworkConfig` の config_prefix を Agent 配下へ移し、設定キーを大規模に rename |
| 解決すること | 切替が 1 行になり、かつ**所有関係が設定ツリーに現れる**。Agent が「自分に適合する net か」を検証する自然な位置も同時に手に入る |
| 解決しないこと | net 以外の同時変更セット（§2 の 3・4・9〜12 など、env / train / metrics にまたがるもの）はこの手では解けない |
| コスト | **大規模 rename**。Run 成果物の `config/config_data.txt` 比較、optuna の trial override、`apps/11_batch_run.bat` の override path（`DefaultDQNAgent.learner.*` 等を直接指定している）が全て影響を受ける。過去 Run との config 比較は断絶する |
| 前提 | `net.block.*` カタログは現状すでにグローバル `net.` からも拾われローカルが上書きする二重解決になっている。この非対称性をどう扱うかの判断が要る |

**本書ではこの案の実施可否を決めない**。rename コストの実測は §6-2 の未決論点へ送る。

### 4.3 案3: 契約検証（fail-fast）

Agent が `quantile_mode` から要求する net 契約（`taus` 入力の有無、出力 rank）を、**設定境界で**照合する。

| 観点 | 内容 |
|---|---|
| 何を変えるか | 既存の検証集約点（`DefaultDQNAgentConfig` コンストラクタ末尾）の自然な拡張 |
| 解決すること | 不整合が**原因の config キーを名指しして**起動時に落ちる。§1.4 の「形状エラーで間接検出」が消える |
| 解決しないこと | **2 行のままである**。概念ツリー不一致そのものは残る |
| コスト | 最小。設定ファイルもキー名も一切変わらない |
| 前提 | ADR 0018 の責任境界に触れる。「Agent が自分の要求する net 契約を検証する」が**局所契約**なのか、禁じられた**推移的到達性検証**なのかの線引きが要る |

`CONTEXT.md`「Actor Env contract」（Agent が対象 Env の EnvSpec を受理できるか判断する契約）と同型の概念を net 側にも置く、という整理になる。

### 4.4 評価軸の対比

| 軸 | 案1 concept overlay | 案2 所有の移動 | 案3 契約検証 |
|---|---|---|---|
| 切替行数 | 1 行 | 1 行 | 2 行のまま |
| 既存キー rename | なし | **大規模** | なし |
| 不整合の検出時期 | 変わらず（NN 構築時） | 変わらず（＋検証位置は得る） | **設定境界** |
| 必要な機構改修 | `AutoMerge` に root スコープ + 連鎖解決 | config_prefix 再設計 + 全 config 書換 | 検証追加のみ |
| 概念ツリー一致度 | 概念ノードを新設できる | 所有関係が階層に出る | 変わらない |
| §2 の 12 件のうち効く範囲 | 全件に適用可能 | net × agent の 1・2・6 中心 | 検証を書いた組だけ |

3 案は排他ではない。組み合わせと順序は §6-1 の未決論点。

---

## 5. Out of Scope

- 実装、コード変更、config ファイルの書き換え
- ADR の新設、`CONTEXT.md` の用語追加（用語の空白は §6-7 に未決として記録するに留める）
- 解決案の選定（次段階の grill で行う）
- 調査中に判明した個別の設定 drift の修正。以下は本件と別件として扱う
  - `net.body.$` が事実上デッド（`net.body.structure` の読み手が存在せず、実際の定義は `net.branch.[main_feature].*` 側にある）
  - `net.branch.AtariNature`（角括弧なし）は正規表現にマッチせず、`net.branch.[main_feature].$` 経由でのみ有効化される。**同じ `$` 記号に「prefix 丸ごと差し替え」と「テンプレート昇格」の 2 用法が同居している**
  - `taus` の K は `train_policy.tau_rule.num_taus` だけから input_spec が作られ、`eval_policy` / `learner.iqn.*` の値は spec に反映されない（現状は手で全て 32 に揃えてある）
  - `$include` の解決失敗が `LOG::warn()` + `continue` で継続する（タイポが黙って通る）

---

## 6. 未決論点（本書の本体。次段階の grill で確定させる）

1. **案 1/2/3 の選定、および併用する場合の順序**。3 案は排他ではない。「案3 で痛みを消してから案1 で機構を入れ、案2 は新規設定の置き方の規約としてだけ採る」といった段階案も候補
2. **案2 の rename コストの実測** — 影響キー数・ファイル数、過去 Run との `config/config_data.txt` 比較が断絶することの許容度、optuna trial override と `11_batch_run.bat` の書換量。これが出るまで案2 の可否は判断できない
3. **`AutoMerge` の適用順序契約** — root 展開と per-namespace `$` の適用順序、連鎖の深さ上限、循環検出。現状は単一パスかつ `merge_keys` の走査順が map 順に依存しており、階層を深くすると即座に破綻する（§4.1 壁2）
4. **マージ結果が素の同名キーを常に上書きする現行挙動を維持するか**。維持しないなら「行位置」と「オーバーレイ順序」のどちらを優先度の正本にするかを決める必要がある
5. **誤字レイヤ・未定義レイヤの無警告 no-op を fail-fast にするか**。`P.` のように意図的に空のレイヤを `.$` チェーンへ書いておく既存運用と、`R.qauntile_mode` のようなタイポを区別できるか
6. **ADR 0018 の責任境界をどこまで動かすか**。「Agent が自分の要求する net 契約を検証する」は局所契約か、禁じられた推移的到達性検証か。ADR 0018 は 2 行切替を**利点として**記述しているため、案1/案2 を採るなら ADR の Consequences も改訂対象になる
7. **設定ツリー自体の語彙が `CONTEXT.md` に 1 語もない**（合成 / オーバーレイ / 上書き層 / `$` / カタログ）。整理の前に既存機構の命名が要る。ただし語の選定に制約がある — 「プロファイル」は「ワークスペース」項で、「eval preset」は「configured eval tag」項で既に `_Avoid_` 指定済み
8. **§2 の 12 件のうち、どれが同一機構で解け、どれが個別対処になるかの切り分け**。分類軸の候補は「所有者が一意に決まるか」「カタログとして再利用したいか」「片方から導出できるか」の 3 つ。§2.8（optimistic target）は 3 軸目で既に畳まれた実例
9. **概念ノードを config dump にどう記録するか**。`config/config_data.txt` は展開後のフラットマップであり、そのままでは「どの概念を選んだか」が失われる。Run 設定の検証はこの dump を ground truth にしている運用のため、概念の選択が読めなくなるのは後退になる。`CONTEXT.md`「Module Config」（include・継承・override を解決した後の値を保持し、元の記述箇所や override 経路は追跡しない）との整合も要る
10. **optuna の trial override が概念ノードを跨ぐ場合の扱い**。ADR 0021 の「workspace config は env 選択を持つため trial override より後に読まれると探索パラメータを潰す。合成順の責任は config 生成側が `$include` の並びで持つ」という決定と、root スコープ合成をどう両立させるか

---

## 7. Further Notes

- 本件の見立て（NN と Agent は別概念だが実際は一体で動く）は、LightZero が Model と MCTSTree をセットで Policy として扱う構成と同じ問題意識にある。anet-lab では `Agent` がその役割に相当するため、新たな上位概念（Policy 相当）を導入する必要はなく、**Agent と net の関係をどう表現するか**に問題を絞れる
- MuZeroAgent は既に案2 寄りの形をしている。`MuZeroAgent.baseline.model.structure.{rep,dyn,pred,...}` として **Agent 配下に構造記述を持ち**、`net.block.[MuZero_Linear256]` などのグローバル block カタログを名前で参照する。つまり「カタログ + 束ね」が DQN 系とは別の流儀で既に同居している。全体整理の際はこの 2 流儀の統一が論点になる
- `agent.class_id = DefaultDQNAgent`（AgentRepository のキー）と `DefaultDQNAgentConfig` の `default_prefix "DefaultDQNAgent"`（ハードコード文字列）が一致しているのは**規約であってコード上のリンクではない**。概念ノードを設計する際、class_id と config prefix の関係を明示するか否かも判断対象になる

---

本PRDは `ready-for-agent` ではない。
解決案の選定 grill、案2 の rename コスト実測、設定ツリー語彙の確定を経た後に、実装用 PRD を別途作成する。
