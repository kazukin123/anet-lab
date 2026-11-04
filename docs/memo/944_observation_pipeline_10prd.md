# 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 PRD

> 2026-11-03 番号を 999 から 944 へ振り直した（PRD114）。

> 暫定 PRD（グリル前のドラフト）。番号 999。§9 の未決を裁定してから正式採番する。
> 起点: 2026-09-26 の [921](921_agent_prev_action_obs_10prd.md) 再検討。直前行動を Observation へ足す位置を Agent 内部に決めた結果、
> 前提として次の 2 つが残った。(1) frame stack と正規化の定義が 4 つの経路と spec 生成に散っていて、観測キーを 1 つ足すたびに
> 全部を手で揃える必要がある。(2) 強制行動が Actor の外で事後に差し替えられるので、Actor は自分の出力が実行されたかを知らない。
> [083](done/083_action_context_retire_10prd.md) は ActionContext の廃止だけを先行し、容器・用語・ADR を本 PRD へ送った。
> 基準コード: 083 適用後の main（`c62b93b6`）。[084 NoisyNet](done/084_noisynet_10prd.md) の実装が `dqn::Actor::MakeAction`・`ActionPolicy`・
> `dqn::Learner` に触れるので、実装は 084 が main に入った後に行い、§2 の事実は 084 適用後のコードで取り直す。
> 行番号ではなくシンボル名で再検索する。
> 関連: [921](921_agent_prev_action_obs_10prd.md)（フレーム段の最初の利用者）、
> 999 操作入力の入力先切替（未同梱の検討メモ: `923_input_target_train_forced_action_10prd.md`） と 999 操作マッピング（未同梱の検討メモ: `922_input_mapping_joystick_10prd.md`）（強制行動の入力の利用者）、
> [910](910_network_lock_audit_10prd.md)（network の排他精査）、[930](930_serialize_10prd.md)（checkpoint）、
> [ADR 0013](../adr/0013-actor-network-resource-policy.md)、[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、
> [ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)、
> [130 §2.2](../design/130_neural_networks.jp.md)、[150 §2.3](../design/150_replay_buffer.jp.md)、[200](../design/200_dqn_agents.jp.md)、[CONTEXT.md](../../CONTEXT.md)。
> 改訂: 2026-10-23。098のReplay窓との接続条件を補足。強制したstep自体を判断として記録する。
> 改訂: 2026-10-26。PRD098から参照する既存ドラフトとして同梱対象を明示した。入力側の利用者に関する未同梱の検討メモは名称で区別し、その設計や本ドラフトの未決は確定事項として採用していない。§3のNG4・NG5・NG9へ今回扱わない理由を補った。

## 1. Context / Problem Statement

### 1.1 観測加工の定義が 4 経路と spec 生成に散っている

DefaultDQN が「env の出した Observation を Network の入力にする」処理は、次の 4 経路と 1 つの spec 生成に分かれて書かれている。

| 経路 | 場所（シンボル） | していること |
|---|---|---|
| 行動選択 | `dqn::Actor::MakeAction` | `stacker_->Stack(obs, episode_start)`（stacker が無ければ `obs.To(device_)`）→ `obs_norm_->Normalize` |
| Experience 受入れ | `DefaultDQNAgent::UpdateFromBatch` | `obs_norm_->NormalizeAndUpdateStats` を `state.obs` と `next_state.obs` に 1 回ずつ呼ぶ（統計更新。戻り値は使わない）。ReplayBuffer へは生の frame を渡す |
| Sample | `dqn::Learner::SetupReplayBuffer` / `NormalizeSampleObservations` | `StuckerConfig` を `ReplayBufferConfig::stack_count` / `stack_keys` へ写す。ReplayBuffer が stack を再構成し、Learner が `obs_norm_->Normalize` を適用する |
| 可視化 | `DefaultDQNAgent::GetTensorDictFunction` のラムダ | 状態スイープの合成 1 frame を stack 幅へ複製 → `obs_norm_->Normalize` |
| spec | `DefaultDQNAgent` のコンストラクタ | env の `obs_spec` をコピーし、stack 対象キーの shape 先頭へ `stack_count` を挿入する（`network_obs_spec`）。normalizer は env の `state_spec` から別に生成する |

「どのキーを stack するか」「形がどう変わるか」「何で正規化するか」は 1 つの定義なのに、経路ごとに書き直されている。
観測キーを 1 つ足す（921 の `prev_action`）には、この全部を揃えて直す必要がある。

### 1.2 `stack_keys` の判定が複製されている

「`stack_keys` が空なら全キー、そうでなければ列挙されたキーだけ」という判定が、Agent 層に 2 箇所（コンストラクタの spec 畳み、
`GetTensorDictFunction`）、部品に 2 箇所（`DictFrameStacker::Stack`、ReplayBuffer の `RingSliceDict`）ある。
型も `vector`（空 = 全キー）と `optional<vector>`（nullopt = 全キー）で揃っておらず、`CreateFrameStacker` が変換している。
存在しないキー名を `stack_keys` に書いても検出されない。

### 1.3 形の食い違いと、潜伏している不成立

| # | 事実 | 表に出ていない理由 |
|---|---|---|
| a | `use_stacker = true` かつ `stack_count = 1` のとき、Actor と可視化は `[B, 1, *frame]`、Network の spec と ReplayBuffer の Sample は `[B, *frame]` になる | GridMaze の現行設定がこの組み合わせ（Run `run_20260924-225648` の実効設定で確認）。離散 Grid は `NetworkBoundaryPreprocessor` が rank を見て S=1 を channel へ畳み、Vector は `Flatten` が吸収する |
| b | normalizer は frame の形（`[B, *frame]`）を検査する。`RunningStdSingleTensorNormalizer` は rank と各次元、`ConstantSingleTensorNormalizer` は rank を検査する。stack 後の `[B, S, *frame]` を渡す行動選択・Sample・可視化は、`pass_through = false` だとこの検査に掛かる。dynamic は停止する。constant は assert が有効なビルドで停止し、無効なビルドではエラーログを出しながら broadcast で計算を続ける。統計更新は stack 前の frame に対して呼ぶので形は合う | リポジトリ管理下の設定は全て `obs_norm.pass_through = true`（検査せず入力をそのまま返す） |
| c | Actor の `Normalize` は Agent mutex の外で統計を読む。統計更新は `UpdateFromBatch` の unique lock 内。Train Actor は `PipelineTrainRunner` が前回の学習完了を待ってから `MakeAction` を呼ぶので重ならないが、背景評価と EvalPanel の Actor は学習と並行に走る | 同上（統計を読まない） |
| d | normalizer の統計は checkpoint に入らない（`DefaultDQNAgent::Save` の `@todo`、[200 §8.2](../design/200_dqn_agents.jp.md)） | 同上 |

a は形の一貫性の問題で、本 PRD が直す。b〜d は「dynamic 正規化を成立させる」という 1 つの関心で、実行時の観測正規化を使う構成
（連続制御の PPO など）で実害になる。現在の env はすべて離散行動で、DQN も離散行動の PPO も観測正規化なしで成立する。
本 PRD は b を構築時エラーへ前倒しするところまでを扱い、成立させる作業は §10 へ持ち越す。

### 1.4 強制行動が Actor の外で事後に差し替えられる

- 経路は `RunnerFrame::OnKey` / `QValuePanel` → `EvalPanel::DoStep(action)` → `EvalRunner::DoStepInternal` の 1 本で、
  `MakeAction` の結果を `BatchActionInfo::WithAction(tensor{action})` で差し替える。`action < 0` が「強制なし」の印。
  Train Runner（`SerialTrainRunner` / `PipelineTrainRunner`）に強制行動の口は無い。
- `DQNActionInfo::WithAction` は Actor Q ヒントの行動 Q と Munchausen bonus を強制行動へ gather し直す。そのために Actor は全行動分の
  `munchausen_terms` を aux へ載せ（読むのは `WithAction` だけ）、`WithAction` は aux の存在・shape・device を 4 つ検査する。
  QR / IQN の tail 診断は「最終実行 action」を遅延 gather する設計にして、差し替え後の action へ追従させている（[200 §9.1](../design/200_dqn_agents.jp.md)）。
- 行動を決める点が 2 つある（Actor の選択と Runner の差し替え）ので、Actor は「自分の出力 = 実行された行動」を仮定できない。
  921 の lane メモリ（直前行動）、Actor Q ヒント、Munchausen 項、将来の log π(a) は、どれも実行された行動から作る値である。
- 所見（コード読み。未実行）: `per_initial_priority_mode = actor_approx` かつ GPU 推論の Actor では、aux の `q_values` が GPU、`EvalRunner` の作る
  強制行動が CPU になり、`WithAction` の device 検査（`cannot replace a hinted action across devices`）で停止する読みになる。
  ヒントは `CreateActor` が全 Actor に同じ条件（`use_per` かつ `actor_approx`）で有効にするので、評価用の Actor も対象になる。
  該当するのは `run.@aainit`（Atari）や `run.@iqn32_actor_approx`（DropMerge）で EvalPanel の手動行動を使う場合。

### 1.5 本 PRD を前提にする利用者

| 利用者 | 必要なもの |
|---|---|
| [921](921_agent_prev_action_obs_10prd.md)（`prev_action`） | フレーム段の置き場、フレーム spec からの一括導出（単位 B）。Actor の出力 = 実行された行動（単位 A） |
| 操作入力の入力先切替（未同梱の検討メモ: `923_input_target_train_forced_action_10prd.md`） | Train Runner から同じ入力で強制行動を渡せること（単位 A） |
| 操作マッピング（未同梱の検討メモ: `922_input_mapping_joystick_10prd.md`） | `WithAction` 経路の乗り換え先（単位 A） |
| [098](098_temporl_skip_policy_10prd.md)（TempoRL のスキップ） | 強制された lane のスキップ状態（続けている行動・残りの step）を捨て、強制したstep自体を新しい判断として `decision=true` で記録すること（単位 A）。trainで区間内の行動が変わると098のReplay窓がSample時にfail-fastするため、両方を同時に満たす。それまでは強制行動とスキップ状態のずれを許す（098 §9、2026-10-23 追記） |
| PPO | 必須の前提は無い。on-policy なので、Learner は Actor が Network へ入れたテンソルをそのまま学習に使え、Sample 側の経路が生まれない。単位 B があれば stack 軸の spec 導出を DefaultDQN と共有できる。観測正規化を使う構成（連続制御）は §10 が前提 |

## 2. 確認した事実（実装の下地）

| # | 事実 |
|---|---|
| 1 | `rl::Actor::MakeAction(const StepCounts&, const BatchState&) const` の実装は production 3（`dqn::Actor`、`MuZeroActor`、`ImageClsActor`）、テスト fake 4（`dqn_based_agent_test.cpp`、`episode_end_test.cpp`、`observers_test.cpp`、`trainer_test.cpp`） |
| 2 | `MakeAction` の呼び出しは `trainer.cpp` の 3 箇所（`EvalRunner::DoStepInternal`、`SerialTrainRunner::DoStep`、`PipelineTrainRunner::DoStep`） |
| 3 | `WithAction` は基底 `BatchActionInfo`、override 2（`DQNActionInfo`、`MuZeroActionInfo`）、テスト fake 2（`episode_end_test.cpp`、`observers_test.cpp`）。production の呼び出しは `EvalRunner::DoStepInternal` の 1 箇所。テストの呼び出しは `dqn_based_agent_test.cpp` 6、`dqn_munchausen_test.cpp` 2 |
| 4 | `MuZeroActionInfo::WithAction` は action だけを差し替え、探索木（`mcts_trees_`）と info を引き継ぐ。`ImageClsActor` は基底の `WithAction`（action の差し替えだけ）を使う |
| 5 | `EvalRunner` の強制行動は `torch::tensor({action})`（shape `[1]`、CPU）。鏡写しインスタンスは 1 lane |
| 6 | `DefaultDQNAgent` コンストラクタの `network_obs_spec` は、`use_stacker && stack_count > 1` のときだけ stack 軸を足す。IQN の `taus` の spec はその後に足す |
| 7 | `GetTensorDictFunction` のラムダと `CreateFrameStacker` は `use_stacker` だけを見る（`stack_count > 1` を見ない）。`DictFrameStacker(1)` は `[B, 1, *frame]` を返す（`stacker_test.cpp` が固定している） |
| 8 | `Learner::SetupReplayBuffer` は `stack_count = use_stacker ? stack_count : 1` を渡す。ReplayBuffer は `stack_count == 1` で stack 軸を落とす。obs の storage は最初の `Push` の dict から確保するので、env の obs spec を使わない |
| 9 | `ObservationNormalizerFactory::CreateObservationNormalizer` は float かつ `num_classes == 0` のキーに normalizer を作る。`DictObservationNormalizer` は normalizer の無いキーをそのまま通す |
| 10 | Rainbow は `NetworkModel` へ env の `obs_spec` をそのまま渡し、`dqn::Learner` へ `obs_norm = nullptr`・stack 定義なしを渡す。`GetTensorDictFunction` は device 転送だけを行う |
| 11 | `dqn::Actor` を直接構築するテストは 8 箇所（`dqn_based_agent_test.cpp` 7、`dqn_munchausen_test.cpp` 1） |
| 12 | リポジトリ管理下の設定: `DefaultDQNAgent.@baseline` が `stucker.use_stacker ?= true`、`obs_norm.pass_through ?= true`。`stack_keys` を指定する設定は無い。GridMaze は `stack_count = 1`、LunarLander と DropMerge は 4 |
| 13 | aux の `raw_obs` / `norm_obs` を読む現用コードと設定は、リポジトリ管理下には無い（テストだけ） |

## 3. ゴール / 非ゴール

**ゴール**

- **G1**: 観測加工の定義（どのキーを stack するか、形がどう変わるか、何で正規化するか）を 1 つのオブジェクトが持ち、
  spec・行動選択・Sample・可視化がそこから導出される。
- **G2**: 既存の設定で挙動が変わらない（metrics checksum 一致）。
- **G3**: Actor 入力・可視化入力・Sample・Network spec の形が、`stack_count = 1` を含む全設定で一致する。
- **G4**: 成立しない組み合わせ（存在しない `stack_keys`、stack ありの非 pass-through 正規化）は構築時に止まる。
- **G5**: 強制行動が `MakeAction` の入力になり、「Actor の出力 = 実行された行動」が不変条件になる。
- **G6**: 921 が「フレーム段を 1 つ足す」局所変更で済む置き場になっている。

**非ゴール**

- **NG1**: フレーム段の中身（`prev_action`）と、その lane メモリ。921。
- **NG2**: dynamic 正規化の成立（§10）。
- **NG3**: 設定から段を並べ替える機構、段の interface 階層（§4.1）。
- **NG4**: ReplayBuffer の stack 再構成（ADR 0024 / 0044）と `DictFrameStacker` の契約変更。これらは保存済みフレームの再構成と逐次観測のstackをそれぞれ担っており、変更自体は可能だが、今回の観測定義・specの共有とは別の処理方式の変更になるため扱わない。
- **NG5**: taus、`NetworkBoundaryPreprocessor`、Env 内の前処理（Atari の resize など）の統合。それぞれ方策・学習、Network境界、Envの入力処理を担うため、統合自体は可能だが、Agentの観測定義を揃える今回の目的とは別の責務の再編になる。
- **NG6**: Rainbow / MuZero / ImageCls の観測加工。Rainbow は stack も正規化も持たず、MuZero は device 転送だけ、ImageCls は対象外。
- **NG7**: 設定キーの改名（`stucker.*` の綴りを含む）。設定キーは 1 つも変えない。
- **NG8**: Train Runner からの強制行動の注入口（操作入力の入力先切替 PRD）。
- **NG9**: PPO。新しいAgentの実装は独立した機能追加であり、既存DQN経路の観測定義と強制行動を揃えるために必要ないため、今回追加しない。

## 4. 提案（暫定）

### 4.1 3 つの置き場

観測パイプラインは「env の Observation を Network の入力へ加工する定義」で、加工を 3 つの置き場に分けて持つ。
置き場ごとに、見るデータと状態の持ち方が違う。1 列の鎖にしないのは、frame stack だけが他と質的に違うためである。
行動選択側は ring buffer へ流し込むストリーム処理、学習側は ReplayBuffer が保存済みの frame から再構成する格納境界の処理で、
同じ「加工」を別の機構が実現している。

| 置き場 | 見るもの | 状態 | 形の変化 | 本 PRD での中身 |
|---|---|---|---|---|
| フレーム段 | lane の 1 frame と、その lane の直前遷移 | lane ごとの小さなメモリ | キーを足す | 空（env の frame をそのまま通す）。921 が `prev_action` を足す |
| スタック段 | frame の時系列 | 行動選択側は ring buffer（`DictFrameStacker`）、学習側は ReplayBuffer の保存 frame | stack 対象キーの shape 先頭へ `[S]` を足す | frame stack |
| バッチ段 | stack 済みの batch | lane に依らない共有統計 | 変えない | 正規化（`ObservationNormalizer`） |

- 順序は固定（フレーム → スタック → バッチ）。設定で並べ替える機構は作らない。順序に意味があり、並べ替えの自由に用途が無い。
- **段は置き場の規約であって、C++ の interface 階層ではない。** パイプラインは 3 つの置き場を持つ具体クラスで、各置き場の中身は
  具体メンバ（stack 定義、normalizer）にする。同じ置き場に 2 種類目が現れるまで抽象 interface を作らない。
- フレーム段の出口の spec を**フレーム spec**と呼ぶ。ReplayBuffer が保存する 1 frame と、正規化統計の形である。
  スタック段の出口が Network の入力 spec（`taus` を除く）になる。
- 段の外に置くもの: `taus`（Observation ではなく forward ごとに変わる Agent の入力。Policy と Learner が注入する現行のまま）、
  `NetworkBoundaryPreprocessor`（Network が所有する spec 駆動の整形）、Env 内の前処理。

```
env の obs_spec ─▶ フレーム段 ─▶ フレーム spec ─▶ スタック段 ─▶ Network 入力 spec ─(+taus)─▶ NetworkModel
                                      │
                                      ├─▶ ReplayBuffer が保存する frame
                                      └─▶ 正規化統計の形
```

### 4.2 `ObservationPipeline`

Agent が所有する共有 Resource。構築後に変わるのは normalizer の統計だけで、spec と stack 定義は不変。

```cpp
// anet/observation_pipeline.hpp
namespace anet::rl {

    /// frame stack の定義（現行の dqn::StuckerConfig を置き換える。設定キーは stucker.* のまま）
    struct FrameStackConfig {
        bool use_stacker = false;
        int stack_count = 4;
        std::vector<std::string> stack_keys;   ///< 空なら全キー
    };

    struct ObservationPipelineConfig {
        FrameStackConfig stack;                ///< スタック段
        ObservationNormalizerConfig obs_norm;  ///< バッチ段
    };

    class ObservationPipeline {
    public:
        ObservationPipeline(const ObservationPipelineConfig& config, const StateSpec& env_state_spec);

        // ---- 各段の出口の spec
        const anet::TensorSpecMap& FrameSpec() const;         ///< フレーム段の出口（本 PRD では env の obs_spec と同じ）
        const anet::TensorSpecMap& NetworkInputSpec() const;  ///< スタック段の出口（taus を含まない）

        // ---- スタック段
        bool HasStack() const;                                ///< use_stacker かつ stack_count >= 2
        int StackCount() const;                               ///< HasStack() でなければ 1
        const std::vector<std::string>& StackKeys() const;    ///< 設定値（空なら全キー）。検証済み
        bool IsStackedKey(const std::string& key) const;      ///< stack_keys の判定はここだけ
        std::unique_ptr<FrameStacker> CreateFrameStacker(int64_t num_envs, torch::Device device) const;  ///< HasStack() でなければ nullptr
        anet::TensorDict ExpandFrame(const anet::TensorDict& frame_obs) const;  ///< 合成 1 frame を stack 幅へ複製する（可視化用）

        // ---- バッチ段
        const std::shared_ptr<ObservationNormalizer>& Normalizer() const;       ///< フレーム spec から生成した共有 normalizer
    };
}
```

- 本 PRD でパイプラインを持つのは `DefaultDQNAgent` だけ。Rainbow は変更しない（stack も正規化も持たず、消す重複が無い）。
- `dqn::Actor` と inner `dqn::Learner` はパイプラインを直接参照しない。Agent が部品（stacker、normalizer、stack 定義）を取り出して
  現行のシグネチャへ渡す。行動選択の「stack → 正規化」と Sample の「正規化」は、それぞれ Actor と Learner の 1〜2 行のまま残る（§9 Q3）。
- `DefaultDQNAgentConfig` のメンバ名 `stucker` は設定キーの源なので変えない。型だけ `FrameStackConfig` に変える。

### 4.3 経路ごとの置換

| 対象 | 改修後 |
|---|---|
| Network の入力 spec | `pipeline_->NetworkInputSpec()` に `taus` を足して `NetworkModel` へ渡す。コンストラクタの手畳みを削除する |
| normalizer の生成 | パイプラインのコンストラクタが `FrameSpec()` から生成する。Agent の `obs_norm_` メンバは `pipeline_->Normalizer()` に置き換える |
| 行動選択 | `CreateActor` が `pipeline_->CreateFrameStacker(num_envs, device)` と `pipeline_->Normalizer()` を `dqn::Actor` へ渡す。`DefaultDQNAgent::CreateFrameStacker` を削除する。`dqn::Actor` は変更なし |
| Experience 受入れ | `pipeline_->Normalizer()->NormalizeAndUpdateStats` を現行どおり 2 回呼ぶ |
| Sample | `dqn::Learner` へ渡す stack 定義を `pipeline_->StackCount()` / `StackKeys()` から作る。`NormalizeSampleObservations` は変更なし |
| 可視化 | `GetTensorDictFunction` のラムダ内の判定と複製を `pipeline_->ExpandFrame(obs)` に置き換える |
| metrics | `obs_norm.*` の `GetScalar` / `GetTensor` は `pipeline_->Normalizer()` へ委譲する（key は不変） |

### 4.4 スタック段の規則と検証

- **`stack_count = 1` は stack なしと同じ形にする。** `HasStack()` は `use_stacker && stack_count >= 2`。偽のとき stacker を作らず、
  `ExpandFrame` は入力をそのまま返し、spec に軸を足さない。Actor・可視化・Sample・spec がすべて `[B, *frame]` になる（§1.3 a の解消）。
  `DictFrameStacker` 自体の `stack_count = 1` の挙動は部品の契約として変えない。
- **構築時の検証**（`ANET_SYSTEM_ERROR`。キー、指定値、期待を含める）:
  - `stack_count < 1`
  - `stack_keys` にフレーム spec に無いキーがある
- 部品（`DictFrameStacker`、ReplayBuffer）へは設定値の `stack_keys` をそのまま渡す。「空なら全キー」は部品それぞれの公開契約として残る。
  Agent 層の判定 2 箇所（spec 畳み、可視化）が `IsStackedKey` の 1 箇所になる（§9 Q5）。

### 4.5 バッチ段: 成立しない組み合わせを構築時に止める

`HasStack()` かつ `obs_norm.pass_through = false` のとき、パイプラインのコンストラクタで `ANET_SYSTEM_ERROR` にする。
normalizer が stack 後の形に対応していないため（§1.3 b）で、現状は最初の `MakeAction` まで潜伏する。
メッセージには `obs_norm.pass_through`、`stucker.stack_count` の指定値と、対応していない旨を含める。§10 の作業が入った時点でこの検証を外す。

### 4.6 強制行動を `MakeAction` の入力にする（単位 A）

行動を決める点を Actor の中の 1 つに畳む。Runner は「この lane はこの行動にせよ」を `MakeAction` へ入力として渡し、
Actor は方策の選択結果へそれを反映した最終の行動を出力する。出力がそのまま実行されるので、出力から派生する値は自動的に揃う。

```cpp
// rl.hpp
/// MakeAction へ渡す lane 単位の強制行動。mask が未定義なら強制なし。
struct ForcedAction {
    torch::Tensor mask;    ///< [N] kBool。true の lane だけ強制する
    torch::Tensor action;  ///< [N] kInt64。mask が false の lane の値は参照しない
};

class Actor : public ModuleBase {
public:
    virtual std::shared_ptr<BatchActionInfo> MakeAction(
        const StepCounts& step, const BatchState& state, const ForcedAction& forced = {}) const = 0;
    // ...
};
```

契約:

1. 強制なし（`mask` が未定義、または全て false）は現行と同一。action、aux、ヒント、乱数消費のどれも変えない。
2. 方策の処理（forward、探索の乱数抽選、診断の集計）は全 lane で現行どおり行い、その後に強制 lane の action を差し替える。
   非強制 lane の結果と乱数列は、強制の有無で変わらない。
3. action から派生する値は差し替え後の action から作る。DQN では Actor Q ヒントの行動 Q と Munchausen bonus、QR / IQN tail 診断の
   最終実行 action 依存値が該当する。状態価値など action に依らない値は変えない。結果は現行の `WithAction` と同じ値になる。
4. 入力の検証: `mask` は `[N]` kBool、`action` は `[N]` kInt64、強制 lane の値は `[0, n_actions)`。違反は `ANET_SYSTEM_ERROR`
   （lane index、指定値、行動数を含める）。device は呼び出し側が合わせなくてよい（Actor が自分の device へ移す）。
5. 強制行動に対応できない Actor は、強制 lane があるときに fail-fast する（黙って無視しない）。現行の 3 Actor はすべて対応する。
   MuZero と ImageCls は action の差し替えだけで、MuZero の探索木と info は探索結果のまま（現行の `WithAction` と同じ）。
6. `BatchActionInfo::WithAction` と override 2 つ、テスト fake 2 つ、aux の `munchausen_terms` を削除する。
7. `EvalRunner::DoStepInternal` は `action >= 0` のとき全 lane を強制する `ForcedAction` を作って渡す。Train Runner は強制行動を渡さない。
8. [098 TempoRL](098_temporl_skip_policy_10prd.md) のスキップが有効な Actor では、強制 lane のスキップ状態（続けている行動・残りの step）を捨て、強制したstep自体を新しい判断として `decision=true` で記録する。trainへ強制行動を接続する場合も、この旗でReplay窓を区切り、区間内の行動不一致によるSample時のfail-fastを防ぐ。強制した step の行動と派生する値は、3 のとおり差し替え後の行動から作る（2026-10-23 追記。098 §9 の未決）。

DQN での実現は、`policy_->SelectAction` の直後・ヒント計算の前に強制 lane を反映する形が素直である（ヒント計算と遅延 gather の診断が
そのまま最終 action を読む）。置き場所は実装で決める。

### 4.7 実装単位と順序

| 単位 | 内容 | 依存 | 等価性の確認 |
|---|---|---|---|
| A | 強制行動の `MakeAction` 入力、`WithAction` の削除（§4.6） | なし | 通常の Run は強制行動を使わないので metrics が一致する。手動行動の値は、現行 `WithAction` のテスト期待値を移して確認する |
| B | `ObservationPipeline` の導入、経路の置換、スタック段とバッチ段の検証（§4.2〜§4.5） | なし | LunarLander（stack 4）と GridMaze（`stack_count = 1`）の改修前後で metrics checksum が一致する |

A と B は互いに独立で、別々に実装・検証できる。921 は A と B の両方、操作入力の 2 本は A を前提にする。PPO に必須の前提は無い（§1.5）。

## 5. 用語（CONTEXT.md へ足す候補）

| 語 | 意味 | _Avoid_ 候補 |
|---|---|---|
| **観測パイプライン**（ObservationPipeline） | Agent が所有する、env の Observation を Network の入力へ加工する定義。フレーム段・スタック段・バッチ段の 3 つの置き場と、各段の spec 変換を持つ | ActionContext（083 で廃止した旧クラス）, パイプライン（無修飾。`PipelineTrainRunner` と紛れる）, preprocessor（`NetworkBoundaryPreprocessor` と紛れる）, wrapper（Env 側の語）, 前処理チェーン（AtariEnv 内の前処理を指す） |
| **フレーム段 / スタック段 / バッチ段** | 観測パイプラインの 3 つの置き場。順序は固定。フレーム段は lane の 1 frame と直前遷移、スタック段は frame の時系列、バッチ段は stack 済みの batch を扱う | stage chain, レイヤ（NN の層と紛れる）, フィルタ |
| **フレーム spec** | フレーム段の出口の spec。ReplayBuffer が保存する 1 frame と、正規化統計の形 | env spec（フレーム段が足すキーを含まない）, raw spec |
| **強制行動**（forced action） | Actor の外（人の操作など）から lane 単位で指定され、Actor がその lane の出力として採用する行動。`MakeAction` の入力 | 行動上書き, override（事後の差し替えを連想させる）, WithAction（廃止する旧 API） |

「強制行動」は 操作マッピング PRD（未同梱の検討メモ: `922_input_mapping_joystick_10prd.md`） §4 の候補と同じ語で、定義はそちらと合わせて確定する。

## 6. 受入条件（暫定）

単位 A

1. `rl::Actor::MakeAction` が強制行動を入力に取る。`WithAction` が現用コード・テストから消える（`core/` と `apps/` の検索で 0 件）。
   aux の `munchausen_terms` も消える。
2. 強制なしの `MakeAction` は、action・aux・ヒント・乱数消費が改修前と一致する（通常 Run の metrics checksum が一致）。
3. 強制 lane の出力 action は強制行動に等しい。Actor Q ヒントの行動 Q と Munchausen bonus、tail 診断の最終実行 action 依存値は
   強制行動から計算され、現行 `WithAction` のテスト期待値と一致する。同じ batch の非強制 lane の値は、強制の有無で変わらない。
4. EvalPanel と QValuePanel の手動行動が動く。GPU 推論かつ `actor_approx` の組み合わせでも停止しない。
5. 範囲外の行動番号、`mask` / `action` の shape・dtype 不正は `ANET_SYSTEM_ERROR` になる。

単位 B

6. `DefaultDQNAgent` コンストラクタの spec 手畳みと、`GetTensorDictFunction` 内の `stack_keys` 判定が消え、`ObservationPipeline` の
   1 箇所になる。`dqn::StuckerConfig` と `DefaultDQNAgent::CreateFrameStacker` が消える。
7. 等価性: LunarLander（stack 4）と GridMaze（`stack_count = 1`）で、改修前後の主要 scalar の checksum が一致する。
   checkpoint の raw checksum は合否に使わない（[930](930_serialize_10prd.md)）。
8. `use_stacker = true`・`stack_count = 1` で、Actor 入力・可視化入力・Sample・Network spec の形がすべて `[B, *frame]` になる。
9. フレーム spec に無い `stack_keys`、`stack_count < 1`、stack ありの `pass_through = false` は、構築時に `ANET_SYSTEM_ERROR` になる。
10. `obs_norm.*` の metrics key が従来どおり取れる。Rainbow は変更が無く、既存テストが緑。

共通

11. `anet-core-test` 全緑、`git diff --check` 空。
12. §8 の設計文書、CONTEXT.md、ADR が新契約に同期している。

## 7. テスト項目（暫定）

単位 A

1. 2 lane で lane 0 だけ強制: 出力 action は lane 0 が強制行動、lane 1 が方策の選択。同じ seed の Actor を強制なしで動かした結果と、
   lane 1 の action・Q 値・ヒントが一致する（非干渉と乱数消費の不変）。
2. 現行の `WithAction` テスト（`dqn_based_agent_test.cpp` 6、`dqn_munchausen_test.cpp` 2）を、強制行動つきの `MakeAction`
   （または `SelectAction` 後の同じ経路）へ移す。期待値は変えない。
3. `EvalRunner::DoStep(action)` が全 lane 強制の `ForcedAction` を Actor へ渡す（fake Actor が入力を記録する）。
4. fail-fast: 範囲外の行動番号、長さの違う `mask` / `action`、dtype 不正。
5. MuZero と ImageCls: 強制 lane の action が差し替わる。MuZero の探索木の描画が最終 action を選択済みとして扱う（現行と同じ）。

単位 B（新規 `observation_pipeline_test.cpp`、タグ `[obs_pipeline]`）

6. spec: `stack_keys` 空・一部指定のそれぞれで、`NetworkInputSpec()` が stack 対象キーだけ `[S, *frame]`、対象外は `[*frame]`。
   `FrameSpec()` は env の `obs_spec` と一致。`taus` を含まない。
7. `HasStack()`: `use_stacker = false`、`stack_count = 1`、`stack_count = 4` の 3 通り。偽のとき `CreateFrameStacker` は nullptr、
   `ExpandFrame` は恒等、spec に軸なし。
8. `ExpandFrame`: stack 対象キーが `[B, S, *frame]` へ複製され、対象外は不変。
9. fail-fast: 未知の `stack_keys`、`stack_count = 0`、stack ありの `pass_through = false`。
10. Actor 経由（`CreateActor`）: `stack_count = 1` の aux `raw_obs` が `[N, *frame]`。既存の `[native_stack]` 5 件は無変更で緑。
11. 可視化: stack 4 の Agent の `GetTensorDictFunction` へ 1 frame を渡し、Network の入口が `[B, 4, *frame]` になる。

統合（ユーザー実施）

12. 受入 2 と 7 の等価 Run。改修直前の HEAD で baseline を採取する（手順は [083 の実装メモ](done/083_action_context_retire_20impl.md) と同じ全点 checksum）。

## 8. 設計文書の同期

| 文書 | 箇所 | 変更 |
|---|---|---|
| `docs/design/110_agents_and_learning.jp.md` | §3、§6.1 | 部品表へ `ObservationPipeline` を追加。`MakeAction` の入力に強制行動を追加 |
| `docs/design/130_neural_networks.jp.md` | §2.1、§2.2 | Network の入力 spec の出所を「観測パイプラインのスタック段の出口」に書き換える。`stack_count = 1` は軸なし、を明記 |
| `docs/design/150_replay_buffer.jp.md` | §2.3 | frame stacking の 2 つの利用箇所を「スタック段の 2 つの実体」として書く。stack 定義の源がパイプラインであること |
| `docs/design/200_dqn_agents.jp.md` | §2.3、§2.4、§3、§6.1、§7.1、§9.1、§9.3 | 「spec 変換は DefaultDQN 固有」の段落、`WithAction` の記述 2 箇所、Actor の sequence 図、既知事項の更新 |
| `docs/design/100_runtime_and_configuration.jp.md` / `160_applications_and_tools.jp.md` | Runner、EvalPanel | 手動行動が `MakeAction` の入力として渡ること |
| `CONTEXT.md` | 「観測と可視化」 | §5 の用語 |
| `docs/adr/` | 新規 | §9 Q10 |

`.en.md` は翻訳成果物なので触らない。

## 9. 未決（グリル対象）

| # | 論点 | 選択肢 / 備考 |
|---|---|---|
| Q1 | 単位 A を別 PRD に切り出すか | (a) 本 PRD のまま、A と B を別コミットにする (b) A を独立 PRD にして先行する。A は段や spec と無関係で、921 と操作入力の 2 本が待っている。083 を切り出したのと同じ形で、起票者の推奨は (b) |
| Q2 | `ObservationPipeline` をクラスにするか | (a) クラス（提案）(b) 自由関数 3 つ（spec 変換、キー判定、frame 展開）だけ足して重複を消す。(b) は最小だが、921 がフレーム spec を normalizer・Network・可視化の単一の源にするときの置き場が無く、PPO が同じ定義を再利用する単位も無い |
| Q3 | Actor と inner Learner がパイプラインを直接参照するか | (a) しない（提案）。Agent が部品を取り出して渡す。シグネチャとテストの構築 8 箇所が変わらない (b) する。「stack → 正規化」の順序もパイプラインの関数にできるが、Actor 用の私有インスタンス（083 の再グリルで見送った形）が要る。921 で lane メモリを足すときに再判断する |
| Q4 | stack ありの非 pass-through 正規化を構築時に止めるか | (a) 止める（提案）(b) 現状のまま §10 を待つ。現状は dynamic が assert と例外、constant が assert 有効ビルドだけの assert で、最初の `MakeAction` まで潜伏する |
| Q5 | 部品へ渡す `stack_keys` | (a) 設定値をそのまま渡す（提案）(b) パイプラインがフレーム spec で解決した明示リストを渡す。(b) は判定が完全に 1 箇所になるが、spec に無いキーを env が出したときの挙動が変わる |
| Q6 | ファイル配置 | (a) `observation_pipeline.hpp` / `.cpp` を新設し、`stacker.*` はそのまま（提案）(b) `stacker.*` を `observation_pipeline.*` へ改名し、`FrameStacker` 群も同居させる（ファイル数を増やさない） |
| Q7 | `MakeAction` のシグネチャ | (a) 既定引数つきの第 3 引数（提案。`GetScalar(key, index = -1)` と同じ流儀で、override も既定値を書く）(b) 入力を 1 つの構造体にまとめる (c) 非 virtual の 2 引数版と virtual の 3 引数版に分ける |
| Q8 | `EvalRunner::DoStep(int64_t action)` の負数 | 強制なしを負数で表している。(a) 本 PRD では触らず、Runner 内部で `ForcedAction` へ変換する（提案）(b) `std::optional<int64_t>` にする。操作マッピング PRD が同じ口を使うので、そちらと合わせて決める |
| Q9 | 設定キー `stucker.*` の綴り | 型名は `FrameStackConfig` にするが、設定キーは据え置く（提案）。直すなら現用設定・サンプル・テスト・現行文書を同じ変更で移行する別件 |
| Q10 | ADR | (a) 2 本: 「観測加工の定義を観測パイプラインへ一元化し、段は置き場の規約とする」「強制行動は `MakeAction` の入力とし、Actor の出力 = 実行された行動を不変条件にする」(b) Q1 で分けるなら PRD ごとに 1 本 |
| Q11 | MuZero の強制 lane | 探索木と target は探索結果のまま、action だけ差し替える（現行と同じ）。Train 側で使う段になったら再検討する（操作入力の入力先切替 PRD） |

## 10. 持ち越し: dynamic 正規化の成立

本 PRD では扱わず、別 PRD とする。観測正規化を使う構成（連続制御の PPO など）の前提で、離散行動の env では要らない。
§4.5 の検証はこの作業が入るまでの歯止めである。

| 論点 | 事実 | 方向（未検討を含む） |
|---|---|---|
| stack 軸の契約 | normalizer は `[B, *frame]` を前提に形を検査する | frame の形を末尾一致で検査し、先頭の stack 軸へ統計を broadcast する |
| 共有統計の並行アクセス | §1.3 c。重なりうるのは背景評価と EvalPanel の Actor | shared Actor は既存の shared lock の内側で読む。clone Actor は lock を取らない設計なので、network の snapshot と同じ同期点で統計の写しを持つ |
| network と統計の対応 | 評価は Sync 時点の network を使うが、統計は常に最新を読む。CONTEXT.md の「Train Actor network snapshot」は ObservationNormalizer を含まないと明記している | 統計は「その network の入力分布の定義」の一部なので、写しは network と対で取るのが筋。用語の定義変更になる |
| checkpoint | 統計は保存されない | [930](930_serialize_10prd.md) の範囲 |
| 評価専用 Run | `learner.enabled = false` でも統計更新は動く（[200 §6.2](../design/200_dqn_agents.jp.md)） | 統計の凍結を設定で明示する |
| 対象キーの除外 | float かつ `num_classes == 0` の全キーが対象 | パイプラインが足したキーは対象外にする（921 が導入する） |
| 統計の device | 正規化のたびに CPU の double 統計を対象 device の float へ変換している | 写しを持つ形にするなら、同時に解消できる |

## 11. 実装に触れる見込みの箇所

単位 A

- [rl.hpp](../../core/anet-core/include/anet/rl.hpp) — `ForcedAction`、`Actor::MakeAction`、`BatchActionInfo::WithAction` の削除。
- [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) / [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) — `dqn::Actor::MakeAction`、`DQNActionInfo::WithAction` の削除、`munchausen_terms`。
- [muzero_based_agent.hpp](../../core/anet-core/src/muzero_based_agent.hpp) / [muzero_based_agent.cpp](../../core/anet-core/src/muzero_based_agent.cpp)、
  [image_cls_agent.hpp](../../core/anet-core/include/anet/image_cls_agent.hpp) / [image_cls_agent.cpp](../../core/anet-core/src/image_cls_agent.cpp) — `MakeAction`。
- [trainer.cpp](../../core/anet-core/src/trainer.cpp) — `EvalRunner::DoStepInternal`。
- テスト: `dqn_based_agent_test.cpp`、`dqn_munchausen_test.cpp`、`episode_end_test.cpp`、`observers_test.cpp`、`trainer_test.cpp`。

単位 B

- `core/anet-core/include/anet/observation_pipeline.hpp`、`core/anet-core/src/observation_pipeline.cpp`、`observation_pipeline_test.cpp`（新規。§9 Q6）。
- [agent.hpp](../../core/anet-core/include/anet/agent.hpp) — `dqn::StuckerConfig` の削除。
- [default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) / [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) — コンストラクタ、`GetTensorDictFunction`、`CreateActor`、`UpdateFromBatch`、`GetScalar` / `GetTensor`。
- [dqn_based_agent.hpp](../../core/anet-core/src/dqn_based_agent.hpp) / [dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) — `dqn::Learner` の stack 定義の型。

設定ファイルの変更は無い。設計文書は §8。

## 12. 関連

- [921_agent_prev_action_obs_10prd.md](921_agent_prev_action_obs_10prd.md) — フレーム段の最初の中身。本 PRD の A と B を前提にする。
- [083_action_context_retire_10prd.md](done/083_action_context_retire_10prd.md) — 前段。容器・用語・ADR を本 PRD へ送った経緯は同 §複雑さ監査。
- 923_input_target_train_forced_action_10prd.md（未同梱の検討メモ: `923_input_target_train_forced_action_10prd.md`）、922_input_mapping_joystick_10prd.md（未同梱の検討メモ: `922_input_mapping_joystick_10prd.md`） — 単位 A の利用者。
- [084_noisynet_10prd.md](done/084_noisynet_10prd.md) — 同じ関数に触れる。実装は 084 の後。
- [910_network_lock_audit_10prd.md](910_network_lock_audit_10prd.md)、[930_serialize_10prd.md](930_serialize_10prd.md) — §10 の排他と永続化の隣接 PRD。
- [done/057_native_stack_dim_handling_10prd.md](done/057_native_stack_dim_handling_10prd.md) — stack 軸を Network の入力 spec に持つ現行契約の出所。
- [ADR 0013](../adr/0013-actor-network-resource-policy.md)、[ADR 0024](../adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md)、[ADR 0044](../adr/0044-replay-frame-history-start-from-episode-start-at-push.md)。
