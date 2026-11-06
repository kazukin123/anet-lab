# 直前行動観測の共通部品（`prev_action` フレーム段）

> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 前提: [999 観測パイプライン](944_observation_pipeline_10prd.md) の単位 A（強制行動の `MakeAction` 入力）と単位 B（`ObservationPipeline`）。
> 本 PRD はその上に最初のフレーム段を足す。
> 改訂: 2026-10-03。配置を Agent 内部に確定し、観測パイプラインのフレーム段として全面的に書き直した。BatchEnv デコレータ案（旧 案 B）と
> その wrap 点（旧 FLAG1）、ActionContext 前提の記述を削除した。NN 統合は現行の `bind` 記法と `bind_concat_dim` に合わせた。

## Context（背景・目的）

風など**非観測の外乱**を持つ POMDP では、エージェントは frame stack の速度差分から外乱を推定する必要がある。しかしその差分には
自機エンジンの推力が混入するため、観測だけからの推定は「まず行動を物理署名から間接デコードし、残差を外乱に帰属する」2 段階の暗黙学習を
要求する。直前の行動 a_{t-1} を観測に含めれば `Δv − 既知のエンジン効果 = 外乱` が well-posed になり、外乱推定の学習が速く安定する
（R2D2 / Agent57 が prev action を入力に含め、RMA が state-action 履歴から環境パラメータを推定するのと同型）。

「自分の方策だから行動は観測から復元できる」という仮定は成立しない。①学習時の行動は確率的（探索）、②replay は過去方策のデータ、
③a_{t-1} は stack 窓の外の入力 o_{t-4} から選ばれており窓内から復元できない。

env ローカルの先行実装が 2 つあり、どちらも現用である。

- [032](done/032_lunarlander_obs_include_action_10prd.md): LunarLanderEnv が 8 次元の `vector` に one-hot(4) を連結する（`E1.obs_include_action = true`）。
- [040](done/040_dropmerge_prev_action_obs_10prd.md): DropMergeEnv の prev-action trio と DROP 列マーカー。040 の Context は、LunarLander で成績・安定性の両面の効果を確認済みと記している。

本 PRD は env に依存しない恒久版である。env ごとに同じ機構を書き足さずに、任意の離散行動 env（Atari など）で使えるようにする。

配置は Agent 内部に確定した（2026-09-26）。理由は次の 3 つ。

- 直前行動は Agent の産物で、Observation へ入れるかどうかはアルゴリズム側の都合である。Env と EnvSpec は変わらないので、
  `CheckSameStateActionSpec` や他の Agent（MuZero / ImageCls）へ影響しない。
- BatchEnv デコレータ案は wrap 点が 2 箇所（factory と、`RunManager` が直接構築する評価用 env）あり、取りこぼすと評価だけキーを欠く。
  Agent 内部なら Actor を作った時点で必ず付く。
- Agent 内部案の難点だった「2 つの注入点と 2 つの spec 追従」は、観測パイプラインがフレーム spec を単一の源にし、行動選択側と
  受入れ側が同じクラスを使うことで消える。

## 確定した設計判断

1. **配置は観測パイプラインのフレーム段。** frame stack の前、ReplayBuffer へ保存する前に、lane の 1 frame へキーを足す。
2. **表現は別の観測キー。** `ObsKeys::kPrevAction = "prev_action"` を新設し、one-hot float32 `[action_count]` を独立キーとして足す。
   `vector` への連結はしない（env 由来の観測キーを書き換えない）。
3. **意味論は「その obs に至らせた行動」。** `obs_t` には a_{t-1} を入れる。`episode_start = true` の lane は全ゼロ（未行動）。
   experience の `next_state`（終端観測を含む）にはその experience の行動 a_t を入れる。
4. **記録するのは Actor の出力。** 強制行動は `MakeAction` の入力として Actor の出力に反映されるので、出力 = 実行された行動である。
   sticky actions のような env 内の執行差は問わない（CONTEXT.md「直前行動観測」: 記録するのは選択した命令）。
5. **行動選択側と受入れ側は同じクラスを 1 つずつ持つ。** lane ごとの「直前に実行された行動」を覚えてキーを足すクラスを、Actor（私有）と
   Agent の Experience 受入れ（私有）がそれぞれ 1 インスタンス持つ。規則が 1 つのコードにあるので、2 経路でずれない。
6. **spec はフレーム spec に 1 エントリ足すだけ。** Network の入力 spec、可視化の入力は観測パイプラインが導出する。
   ReplayBuffer は obs の storage を最初の `Push` の dict から確保するので、変更が要らない。
7. **パイプラインが足したキーは正規化しない。** normalizer はフレーム spec のうち env 由来のキーだけを対象に作る。one-hot は
   既に値域が揃っており、robust update の外れ値判定で稀な行動が毎回ログへ出る問題も起きない。
8. **stack の既定は全キー。** `stack_keys` が空なら `prev_action` も stack され、直近 S 個の行動履歴になる（R2D2 風）。
9. **離散行動のみ。** 連続行動は構築時エラー。
10. **既定は OFF。** OFF ではキーを足さず、現行と同一の挙動になる。
11. **interface 階層を作らない。** フレーム段は具体クラス 1 つで実装する。2 種類目のフレーム段（直前報酬など）が現れた時点で抽象化を検討する。
12. **可視化の合成 frame は未行動で埋める。** 状態スイープは env の観測キーだけを組み立てるので、`prev_action` が無ければ全ゼロを足す。
13. **032 / 040 と併用しない。** env 側の flag と同時に有効にすると二重の条件付けになる。エラーにはしない（意味の判断は利用側）。
14. **対象は DefaultDQN。** 設定を持つ Agent だけが使う。Rainbow / MuZero / ImageCls は設定を持たず、挙動も変わらない。

## 仕様

### 設定

```
DefaultDQNAgent.@baseline : prev_action_obs.enabled ?= false
```

env 別の設定で有効にする（例: `A1.prev_action_obs.enabled = true`）。有効にしただけでは Network は `prev_action` を使わない。
`bind` へ書く必要がある（§NN 統合）。書き忘れは、どの branch にも bind されていない入力に対する既存の WARN で気付ける。

構築時の検証（`ANET_SYSTEM_ERROR`）:

- 有効かつ `ActionSpec::is_discrete == false`
- 有効かつ env の `obs_spec` が既に `prev_action` キーを持つ

### フレーム spec へ足す `TensorSpec`

| field | 値 | 備考 |
|---|---|---|
| `type` | `SpaceType::Vector` | Vector は 1 次元 shape（`AssertSanity`） |
| `shape` | `{ action_count }` | `ActionSpec::GetNumActions()` |
| `dtype` | `torch::kFloat32` | |
| `num_classes` | `0` | float の離散指定は `AssertSanity` が弾く |
| `labels` | `action_spec.value_labels` | 例 LunarLander: do nothing / fire left / … |
| `min_values` / `max_values` | `{0.0}` / `{1.0}` | |

stack 有効時の Network 入力 spec は、stack 対象なら `[S, action_count]`、対象外（`stack_keys = vector` など）なら `[action_count]` になる。
どちらも `ObservationPipeline::NetworkInputSpec()` が導出する。

### キーの規則（全経路の不変条件）

| 対象の frame | lane i の `prev_action` |
|---|---|
| 行動選択に使う `state`、および experience の `state` | `episode_start[i]` なら全ゼロ。そうでなければ、その lane で直前に実行された行動の one-hot |
| experience の `next_state`（終端観測を含む） | その experience の `action[i]` の one-hot。マスクしない |
| 可視化の合成 frame | 全ゼロ |

`next_state.episode_start` は契約上 true にならない。非終端の `next_state` と、次の experience の `state` は同じ値になる
（次の `state` は `episode_start = false` で、直前に実行された行動が a_t のため）。

### フレーム段のクラス

```cpp
// anet/observation_pipeline.hpp
namespace anet::rl {

    /// 直前行動観測のフレーム段。lane ごとの「直前に実行された行動」を覚え、frame へ prev_action キーを足す。
    /// 行動選択側（Actor 私有）と受入れ側（Agent 私有）が 1 つずつ持つ。
    class PrevActionFrameStage {
    public:
        PrevActionFrameStage(int64_t num_envs, int64_t num_actions);

        /// 現在の state 用。episode_start の lane は全ゼロ、他は直前に実行された行動の one-hot。
        /// 入力 dict は変更せず、キーを足した shallow copy を返す。
        anet::TensorDict ApplyToState(const anet::TensorDict& frame_obs, const torch::Tensor& episode_start) const;

        /// experience の next_state 用。action の one-hot を足す（マスクしない）。
        anet::TensorDict ApplyToNextState(const anet::TensorDict& frame_obs, const torch::Tensor& action) const;

        /// 実行された行動を記録する（行動選択側: MakeAction の出力、受入れ側: experience の action）。
        void Advance(const torch::Tensor& action);
    };
}
```

- 生成は `ObservationPipeline` が行う（無効時は nullptr）。可視化用の全ゼロは `ObservationPipeline::ExpandFrame` の前段で足す。
- lane の最初の `ApplyToState` は、全 lane が `episode_start = true` でなければ `ANET_SYSTEM_ERROR` にする
  （ReplayBuffer の `Push` が lane の初回に履歴開始を要求するのと同じ契約。覚えた行動が無い状態で one-hot を作らない）。
- 入力 dict が既に `prev_action` を持つ場合は `ANET_SYSTEM_ERROR`。
- 明示の `Reset()` は持たない。lane の初期化は `episode_start` のマスクが担う（stacker と同じ）。
- 覚えた行動は checkpoint に保存しない（Actor の RNG や snapshot 進行と同じ、Run をまたがない State）。

### 経路

- **行動選択**（`dqn::Actor::MakeAction`）: `ApplyToState(state.obs, state.episode_start)` → stack → 正規化 → 行動選択（現行）→
  強制行動を反映した最終の action で `Advance`。aux の `raw_obs` / `norm_obs` は `prev_action` を含む。
- **Experience 受入れ**（`DefaultDQNAgent::UpdateFromBatch`）: 統計更新と `Push` の前に、`state.obs` へ `ApplyToState`、`next_state.obs` へ
  `ApplyToNextState` を適用した experience を作り、`Advance(action)` を呼ぶ。Runner から受け取った experience は変更しない
  （Event が運ぶ experience は env の観測キーのまま）。
- **ReplayBuffer**: 変更なし。履歴開始の padding は開始 frame の複製なので、episode 先頭では stack 全体がゼロになる。終端観測は
  `next_state.obs` が dummy slot へ保存されるので a_t の one-hot になる。
- **Sample**: 変更なし（保存された frame から stack を再構成し、正規化は `prev_action` を素通しする）。
- **可視化**（`GetTensorDictFunction`）: 合成 frame に `prev_action` が無ければ全ゼロを足してから `ExpandFrame` する。
- **spec**: `ObservationPipeline` のコンストラクタが、有効時にフレーム spec へ `prev_action` を足す。

行動選択側と受入れ側の一致は構造から出る。Runner は Actor の出力した `action_info` をそのまま env へ渡し、同じ `action_info` を
experience に入れる（単位 A）。両側が同じクラスへ同じ行動列と同じ `episode_start` を与えるので、同じキーが出る。

### NN 統合

branch の `bind` は複数の term を `bind_concat_dim`（既定 1）で連結する（[130 §2.4](../design/130_neural_networks.jp.md)）。stack 有効時の入力は
`vector [B, S, F]` と `prev_action [B, S, A]` で、既定の dim 1 では連結できない（後続の次元 F と A が違う）。dummy 入力も同じ rank なので、
この誤りは Network の構築時に検出される。

frame ごとに連結する（時間軸 Conv1d と MLP のどちらにも使える）:

```
DefaultDQNAgent.net.branch.[main_feature].bind = vector, prev_action
DefaultDQNAgent.net.branch.[main_feature].bind_concat_dim = -1        # [B,S,F] と [B,S,A] を連結して [B,S,F+A]
DefaultDQNAgent.net.branch.[main_feature].structure = Conv1D_Permute > TConv64 > SiLU > Flatten > Linear256 > LN256 > SiLU > Linear128 > LN128 > SiLU
```

LunarLander（F=8、A=4、S=4）では各 frame が「obs 8 次元、行動 4 次元」の順に並んだ `[B, 4, 12]` になり、032（env が 12 次元の
`vector` を出す）と同じテンソルになる。MLP なら同じ `bind` のまま `structure` を `Flatten > …` にする。

`stack_keys = vector` のように stack 設定がキーで違うときは rank が揃わないので、キーごとに `Flatten` してから連結する:

```
DefaultDQNAgent.net.branch.[vec_flat].bind = vector
DefaultDQNAgent.net.branch.[vec_flat].structure = Flatten
DefaultDQNAgent.net.branch.[pa_flat].bind = prev_action
DefaultDQNAgent.net.branch.[pa_flat].structure = Flatten
DefaultDQNAgent.net.branch.[main_feature].bind = vec_flat, pa_flat      # [B,S*F] と [B,A] を連結
```

temporal conv の後で合流させたい場合（late fusion）も、同じく `prev_action` を `Flatten` した branch を後段の `bind` へ足す。

### viewer 互換

`ToUnifiedObservation` は vector 扱いのキーを辞書順に連結する。stack されない `prev_action`（`[B, A]`）は vector 扱いになり、
`"prev_action" < "vector"` なので先頭に来る。これは `raw_obs` / `norm_obs` と、無接尾の obs probe が前提にする
「`vector` が先頭」（`probe.cpp` の `GetObservationSpecKeyForBase`）を壊す。

随伴変更として、vector 扱いの連結で `kVector` を先頭に置く。vector 扱いのキーを 2 つ以上持つ env は現状無いので、既存の挙動は変わらない。
stack された `prev_action`（`[B, S, A]`）は grid 扱いで `vector` の後ろに連結されるので、この変更が効くのは stack されない場合だけである。

### stacking

- `stack_keys` が空（既定）: 全キーを stack する。`prev_action` は直近 S 個の行動履歴になる。
- `stack_keys = vector`: `prev_action` は stack 対象外で、最新の 1 個（a_{t-1}）だけになる。行動選択側は `DictFrameStacker` が素通しし、
  ReplayBuffer は最新 frame だけを返す。

## 非対象（Non-goals）

- 連続行動（構築時エラー。将来は生の行動ベクトルを同じキーで入れる変種）。
- 直前報酬など、2 種類目のフレーム段。
- Rainbow / MuZero / ImageCls への展開、PPO。
- dynamic 正規化の成立（[999 観測パイプライン](944_observation_pipeline_10prd.md) §10）。
- 032 と 040 の整理（下記「確認したい点」）。

## 受け入れ基準

1. 無効時はキーが現れず、Actor の入力・ReplayBuffer の dict・Network の入力 spec が現行と一致する（同 seed の Run で metrics checksum 一致）。
2. 有効時、Network の入力 spec に `prev_action` が §仕様の `TensorSpec`（stack 対象なら `[S, A]`）で現れ、nn_viz の入力に出る。config dump に設定が出る。
3. §キーの規則が、行動選択・Experience 受入れ・Sample・可視化の全経路で成り立つ。
4. 強制行動で動かした lane は、次の step の `prev_action` が強制行動になる。
5. 行動選択側の Network 入力と、同じ時刻の ReplayBuffer 保存 frame の `prev_action` が一致する。
6. `ToUnifiedObservation` が `vector` を先頭に保つ。
7. `prev_action` は正規化されない（統計の対象に入らない）。
8. 等価性アッセイ（LunarLander）: 032 を有効にした Run と、032 を無効にして本機能と frame ごとの連結を有効にした Run は、
   Network の入力テンソルが同じになる。同 seed で主要な学習 scalar（loss、q_max 系、評価の episode return）の checksum が一致することを確認する。
   一致しない場合は、差の原因（初期化順、乱数消費）を特定して記録する。
9. 連続行動、env に既存の `prev_action` キー、lane 初回の `episode_start = false` は `ANET_SYSTEM_ERROR` になる。
10. `anet-core-test` 全緑、`git diff --check` 空。設計文書と CONTEXT.md が同期している。

## テスト項目

**Unit**（`observation_pipeline_test.cpp`、タグ `[prev_action]`）

1. spec: 有効時にフレーム spec へ `prev_action` が `{A}` / float32 / `num_classes = 0` / labels = `value_labels` / min 0 / max 1 で足され、
   他のキーは不変。stack 有効時の Network 入力 spec は `[S, A]`、`stack_keys = vector` では `[A]`。無効時はキーが無い。
2. 構築時エラー: 連続行動、env の `obs_spec` に既存の `prev_action`。
3. `PrevActionFrameStage`: 初回（全 lane `episode_start`）は全ゼロ。`Advance(a)` の後、`episode_start = false` の lane は one_hot(a)、
   true の lane はゼロ。`ApplyToNextState` はマスクせず one_hot(a)。入力 dict を変更しない。
4. 契約違反: 初回に `episode_start = false` の lane がある、入力 dict が既に `prev_action` を持つ。
5. normalizer: dynamic 設定でも `prev_action` に normalizer が作られない（stack なしの構成で確認する）。
6. `ToUnifiedObservation`: `{vector, prev_action}` の非 stack dict で `vector` が先頭。stack された dict でも `vector` が先頭。

**Actor / Agent**（`dqn_based_agent_test.cpp`、タグ `[dqn][prev_action]`）

7. `MakeAction` を数 step 回し、Network 入口の `prev_action`（trace）が直前の出力 action の one-hot で、stack 内が時系列順になっている。
8. 強制行動つきの `MakeAction` の次の step で、その lane の `prev_action` が強制行動になっている。
9. `episode_start` の lane は stack 全体がゼロになる。
10. 境界（done / truncated）を含む遷移列を `UpdateFromBatch` へ流し、`Sample` の `obs[prev_action]` が `[B, S, A]`、境界直後はゼロの padding、
    `next_obs` の最新 frame が当該遷移の行動になっている。
11. 行動選択側と受入れ側の一致: scripted な fake env を Runner で回し、各 step の Network 入口の `prev_action` と、同じ時刻の
    ReplayBuffer 保存 frame が一致する。
12. `stack_keys = vector`: 行動選択 `[N, A]`、Sample `[B, A]` で forward が通る（キーごとに `Flatten` する NN 構成）。
13. 可視化: `GetTensorDictFunction` へ env のキーだけの合成 frame を渡して forward が通り、`prev_action` の入力が全ゼロ。
14. 無効時: Actor 入力、ReplayBuffer の dict、Network の入力 spec にキーが無い。

**統合（ユーザー実施）**

15. 受け入れ基準 1 の等価 Run（無効時）。
16. 受け入れ基準 8 の等価性アッセイ（LunarLander、032 対 本機能）。

## 実装対象

- `core/anet-core/include/anet/rl.hpp` — `ObsKeys::kPrevAction`。
- `core/anet-core/src/rl.cpp` — `ToUnifiedObservation` で `kVector` を先頭に置く。
- `core/anet-core/include/anet/observation_pipeline.hpp` / `src/observation_pipeline.cpp` — 設定、フレーム spec への追加、`PrevActionFrameStage`、
  normalizer の対象を env 由来キーに限定、可視化の全ゼロ。
- `core/anet-core/src/dqn_based_agent.hpp` / `dqn_based_agent.cpp` — `dqn::Actor` がフレーム段を私有し、`MakeAction` で適用と `Advance`。
- `core/anet-core/include/anet/default_dqn_agent.hpp` / `src/default_dqn_agent.cpp` — 設定の読み込み、受入れ側のフレーム段、`UpdateFromBatch`、
  `GetTensorDictFunction`、`CreateActor`。
- `apps/runner/config/agent.txt` — `DefaultDQNAgent.@baseline` の既定葉。`apps/runner/config/LunarLander.txt` — コメントの設定例。
- テスト: `observation_pipeline_test.cpp`、`dqn_based_agent_test.cpp`。

## 設計文書の同期

| 文書 | 変更 |
|---|---|
| `docs/design/020_user_guide_run.jp.md` | 設定の使い方（有効化と `bind` の書き方） |
| `docs/design/130_neural_networks.jp.md` §2.1 | Network の入力に、観測パイプラインのフレーム段が足すキーがあること |
| `docs/design/150_replay_buffer.jp.md` §2.1 | ReplayBuffer へ渡る experience の obs は、フレーム段を通った後の frame であること |
| `docs/design/200_dqn_agents.jp.md` §2.3、§6.1、§6.2、§7.2 | フレーム段の説明、行動選択と Experience 受入れの sequence 図、設定グループ |
| `CONTEXT.md` | 「直前行動観測」を「観測と可視化」へ移し、観測キー `prev_action` を定義する。prev-action trio の `_Avoid_` にある PRD 番号での言及を、番号を使わない表現へ直す |

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[prev_action],[obs_pipeline],[native_stack],[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

等価 Run は [083 の実装メモ](done/083_action_context_retire_20impl.md) と同じ全点 checksum の手順で比較する。

## 確認したい点（再グリルで裁定）

1. 設定キーの名前（`prev_action_obs.enabled`）。
2. 受入れ側の置き場。外側の `DefaultDQNAgent::UpdateFromBatch`（提案）か、inner の `dqn::Learner`（Rainbow へ広げる場合）か。
3. Actor がフレーム段と stacker を別々の部品として持つか、1 つの私有オブジェクトにまとめるか（[999 観測パイプライン](944_observation_pipeline_10prd.md) §9 Q3）。
4. 受け入れ基準 8（032 との等価性アッセイ）を合否に含めるか。
5. 本機能の採用後に LunarLander を本機能へ移し、032 の env flag を削除するか。040 の trio は表現が違う（DROP 列の連続値）ので残る。

## 関連

- [944_observation_pipeline_10prd.md](944_observation_pipeline_10prd.md) — 前提。フレーム段の置き場、フレーム spec、強制行動の `MakeAction` 入力。
- [032_lunarlander_obs_include_action_10prd.md](done/032_lunarlander_obs_include_action_10prd.md) — env ローカルの先行実装（LunarLander）。
- [040_dropmerge_prev_action_obs_10prd.md](done/040_dropmerge_prev_action_obs_10prd.md) — env ローカルの先行実装（DropMerge）。
- [057_native_stack_dim_handling_10prd.md](done/057_native_stack_dim_handling_10prd.md) — stack 軸を Network の入力 spec に持つ現行契約。
- [083_action_context_retire_10prd.md](done/083_action_context_retire_10prd.md) — ActionContext の廃止。
- [ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md) — 履歴開始と padding の規則。
