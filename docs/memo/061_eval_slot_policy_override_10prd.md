# PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定

- 起票日: 2026-08-24(draft)、グリル: 2026-09-06〜07(`/grill-with-docs`。裁定は §5・§9、決定の記録は ADR 0038)、改訂: 2026-09-08(Codex DropMerge レビュー反映: §2.7-2.8、§5.3-5.4、§7.3-7.6、§8-10)
- 状態: **グリル完了・実装待ち**。実装順は PRD 072(設定リゾルバの選択コピーが source の最終値を読むようにする、別起票)→ 本 PRD P1 → P2。P3 は着手時に別途グリル
- 対象: `core/anet-core`(`rl.hpp` / `agent.hpp` の Agent・Actor IF、4 Agent 実装、`trainer.*` の RunManager / RunnerBase / EvalRunner、`observers.*` の metrics 参照先)、`apps/runner`(EvalPanel / RunnerFrame / RunnerApp)、`apps/runner/config` 全 env config と `agent.txt` / `common.txt` / `metrics_scalar.txt`、`viewers/metrics-tools/inspect_run.py`、`apps/runner/tools/dropmerge_optuna.py`
- 関連: PRD 072(前提。設定リゾルバの選択コピーが source の最終値を読む)、`done/060_eval_batch_episodes_10prd.md`(本数。§「本 PRD では解けない隣接論点」で本件を名指し)、`done/052_eval_schedule_separation_10prd.md` / ADR 0027(定義とスケジュールの分離)、`done/059_config_concept_tree_alignment_10prd.md`(カタログ / プロファイル / 上書き層の用語)、`912_background_eval_snapshot_ordering_10prd.md`(同じ eval 経路の別論点)、ADR 0038(本 PRD の決定と却下案)
- 発見経緯: Atari Breakout の探索(`docs/experiments/default-dqn/atari/2026-08-17_baseline.md` 探索ブロック 14 / 16 / 17 / 18 / 19)で、eval の ε と評価対象 network が Run 単位でしか選べないために測れなかった項目が積み上がった。起票後、`Atari.txt` の `run.@greedy_eval`(2026-09-06 実測)と `run.@evalonly`(checkpoint 評価専用 Run)で実需が 2 件増えた

## 1. 背景と実害

configured eval tag(現 `train.eval.[tag]`)は 052 で定義とスケジュールが分離され、060 で lane 数と本数が分離されて、**タグ単位で env 設定・並列度・本数・clone を持てる**。しかし **行動方策(ε 等)は `DefaultDQNAgent.eval_policy` の 1 本で全タグ共通**であり、**評価対象 network(target / online)は `run_mode`(Eval1 か否か)に固定配線**されている。この 2 つの縛りにより「同じ Run の中で、違う ε・違う network の eval を並べる」ことができない。

実害(起きたこと):

- **A. 文献比較の土俵が 1 Run で組めない。** Nature DQN 2015 は ε=0.05 / 30 本、BTR は ε=0.01 / 100 本。ε が agent 共通なのでどちらか一方しか置けない
- **B. 決定論 eval と文献 eval が排他。** v5 の FIRE デッドロック回避で eval を ε=0.01 にした結果、「ある重みに対する eval は毎回同一の 1 軌道」という回帰追跡用の性質を失った
- **C. eval1 / eval2 の健全性指標(target / online 比)と ε 変更が両立しない。** 両者が同一 policy であることが暗黙の前提
- **D. online net を 2 通りの ε で評価できない。** `IsForTarget` が Eval1 にだけ true を返す
- **E / F. 分布測定と NN 構成 A/B の判定手段が無い。** 決定論の 1 軌道では 5% の差が判定不能
- **G. `run.@greedy_eval`(`Atari.txt:387-392`)。** 2026-09-06 実測で ε 0.01→0 は平均 640.0→715.1・満点率 13.3%→24.7% だが上限空転 0.1%→10.4%。「eval の eps はエージェント単位なので eval1/eval2 の両方へ効く(スロットごとの指定は PRD 061)」と注記されている
- **H. `run.@evalonly`(`learner.enabled=false`)。** 1 checkpoint を複数プロトコル(greedy / BTR / Nature)で同時採点したい局面が現実化した

## 2. 現行コードで確定している事実(2026-09-07 時点)

### 2.1 Agent / Actor 内部で RunMode が意味を持つ箇所は 7 つ

| 箇所 | 現状 |
|---|---|
| DefaultDQN の policy 選択 | `IsEval ? eval_policy_ : train_policy_`(`default_dqn_agent.cpp:522-527`) |
| DefaultDQN / Rainbow の network 選択 | `Eval1 → target`、それ以外 online(`default_dqn_agent.cpp:485-494`、`rainbow_agent.cpp:212-216`) |
| DefaultDQN の clone 既定 | train は `train_actor.clone_model`、eval は false(`:505-506`) |
| DefaultDQN の定期 snapshot | `is_train && clone` のみ(`:543-545`)。周期 profile は `step.exp_step`、age は `step.train_step`(`dqn_based_agent.cpp:1791-1810`) |
| DefaultDQN の Actor Q ヒント送出 | `!IsEval && use_per && ACTOR_APPROX`(`:540-542`) |
| MuZeroActor の温度 / Dirichlet noise | `!IsEval` で温度 anneal・noise 付与(`muzero_based_agent.cpp:822-835,856`) |
| AgentBase の RNG | `GetRandomGenerator(RunMode)` = RunMode 別に 1 stream を全 actor で共有(`agent.cpp:30-44`)。EvalPanel と eval1 は同じ stream を消費する |

ImageClsActor は run_mode を保持するだけで参照しない。Env 側の RunMode 依存は `IsTrain / IsEval` のみ(ImageCls の dataset 選択と episode scope、CartPole の初期化)。**Eval1 と Eval2 を区別する Env・metrics・GUI は存在しない。**

### 2.2 policy は Agent 所有の 3 本で、Actor は shared_ptr を共有する

- `train_policy_ / eval_policy_ / target_policy_` を agent 構築時に生成(`default_dqn_agent.cpp:199-201`)。eval1 / eval2 / EvalPanel の全 eval actor が同じ `eval_policy_` インスタンスを使う(`:552`)
- `OnLearn(counts)` は learner thread が毎 `UpdateFromBatch` の末尾で 3 policy へ配る(`:595-597`)。eval ε も train 側 counts で進み、eval actor はそれを無同期で読む
- eval 既定は train の終端値から**コード内で導出**している(`default_dqn_agent.hpp:102-116`: Greedy、decay 0、`eps_start = eps_end = train.eps_end`、uqe_tau も同様)
- Rainbow は `action_policy_` 1 本で train / eval を兼ね、eval も train ε で走る(`rainbow_agent.cpp:229`。`:208` の `policy` は未使用)
- `train_policy` / `eval_policy` の残りの消費者: NN 入力 spec(`:165-173`、`taus` の shape = `train_policy.tau_rule.num_taus`)、`target_policy` 妥当性検査(`:192-196`)、Learner への `target_policy_` 注入(`:215 / 222 / 228`)、`GetScalar` dispatch(`:399-406`)、`OnLearn`(`:595-597`)、config dump と Save の `ToString()`(`:60-61`、`:289`)。`dqn_based_agent.*` は参照しない
- config ctor 内の派生: eval 既定(`hpp:102-116`)、**`use_optimistic_target=true` なら `target_policy = train_policy` を丸ごとコピーして eps / uqe_eps を 0・EpsilonGreedy → Greedy に強制し、どちらの分岐でも `tau_rule = {fixed, 32}` と `full_distribution_query` を既定へ戻してから `target_policy.*` の明示設定を読む(`hpp:142-185`)**、`quantile_mode` 伝播(`:310-312`)。`use_optimistic_target` に実行時分岐は無い(参照は ctor と tests のみ)

### 2.3 Actor IF と counts

- `Actor` IF は `MakeAction(const StepCounts&, const BatchState&)` と `Sync()` のみ(`rl.hpp:702-713`)。Actor は Module ではなく `GetScalar` を持たない
- EvalRunner は**自分の** `step_counts_` を MakeAction へ渡す(`trainer.cpp:278`)。train 側 counts は `RunSession(event_counts)` でセッション単位に受け取り(`:342-377`)、`@episode_end / @session_end` の step 座標に使う。`event_counts` は EpisodeEvalObserver が LearnEvent の `counts` を値コピーしたもの(`observers.cpp:550-584`)
- `Actor::Sync()` の呼び出し元は EvalRunner::Sync / RunSession 冒頭 / EvalPanel の 3 箇所。clone の有無に関係なく呼ばれ、shared のときは `CopySourceNetwork` が即 return する

### 2.4 clone_model と device 検証

- 生産者: スロット `clone_model`(既定 true、`trainer.cpp:910-911`)、EvalPanel `model_sync.mode != shared`(`EvalPanel.cpp:43-46`)、train runner は `std::nullopt`
- 消費者: `ValidateSharedActorDevice`(`trainer.cpp:35-53`)と、各 Agent の `.value_or(...)` + device チェック(DefaultDQN / Rainbow / ImageCls で 3 重複)。MuZero は引数を無視する

### 2.5 設定機構

- `anet::Config(config_data, default_prefix, override_prefix)`(`config.hpp:167,179-196`)。`MakeSubConfigData(prefix)` で `prefix.[key]` を列挙(`config.cpp:454-484`)。`$` チェーンの RHS は任意 prefix(`config_impl.cpp:288-324`)
- **リゾルバは root の `$` 選択を宣言順、term のコピーで生じた nested `$` を即時解決する**(`config_impl.cpp:30-77`)。カタログ項目間の継承(`X.[b].$ = X.[a] > …`)は宣言時点のコピーになり、後段 overlay の `[a]` 変更が `[b]` へ伝播しない → PRD 072
- Agent は生 ConfigData を保持しない(factory `default_dqn_agent.cpp:650-663`)。`docs/ownership_guideline.md`: Config は Agent 所有の読み取り専用 Resource、epsilon 等のスケジュール状態は ActionPolicy の State、Actor 専用 snapshot は Actor 所有の private Resource
- 前例: `ImageClsEnv.train.dataset_key = food101_train` → カタログ `ImageDataset.[food101_train]`(CONTEXT「DatasetKey」)

### 2.6 移行範囲の実測

- `train.*` の literal: コード 78(`train.eval` 49 / `train.eval_schedule` 23 / `num_envs` 3 / `seed` 1 / `main_runner_type` 1 / `eval_device_type` 1)、テスト 7 ファイル、config 8 ファイル約 156 行、Python 2(`inspect_run.py`、`dropmerge_optuna.py`)、docs/design 7 + CONTEXT 3 項 + AGENTS.md
- `[eval1] / [eval2]` の出現: config 8 ファイル 194 箇所(metrics の `$eval.[…]` を含む)
- `CreateActor` の実装 4 + テストダブル 5(`trainer_test.cpp:310` / `observers_test.cpp:296` / `episode_end_test.cpp:295` / `dqn_based_agent_test.cpp:653,1211`)
- BTR の eval ε は 125M frames まで 0.01、以降 0(`reports/btr_hyperparams_survey_2026-08-26.md:138,678,964`)

### 2.7 IQN の τ 入力 shape(spec K)

- `taus` の TensorSpec shape に `train_policy.tau_rule.num_taus` を置く(`default_dqn_agent.cpp:165-173`)。実消費者は `NetworkBuilder` の dummy forward `[1, K]`(`nn_impl.cpp:2147-2171`、lazy Linear の初期化と Head 構築)と graphviz 表示だけで、モジュールは K を保持しない(`CosineEmbedding` は `num_basis` だけを持ち、bind 積は batch 次元しか照合しない `nn_impl.cpp:1125-1135`、IQN Head は rank 3 のみ検証)。`ValidateQuantileNetworkContract` は `taus` bind 因子の有無だけを見る
- 同一 network が異なる K で forward される実績: DropMerge eval は spec 32 に対し risk 32 + full 32 の K=64、Atari は K=8 に対し policy_churn probe 32 と target_policy 既定 32、既存テスト `dqn_based_agent_test.cpp:2909` は spec 3 の network を K∈{1..5} で forward する。runtime の K は batch 的次元であり、spec K は構築時の placeholder にすぎない。`Network::Clone` も同じ spec で再 build する

### 2.8 DropMerge の実効 eval 設定と診断メトリクス(2026-09-08)

- 実効 eval policy(`@baseline > @iqn > @fast > A1 > @bf16 > A2 > A3`): UQE(config)、`uqe_tau_start / end` 0.85(**コード導出** = `train_policy.uqe_tau_end`、`DropMerge.txt:615`)、`uqe_eps` 0 / 0(**コード固定**)、`eps` 0.05 / 0.05(コード導出。UQE では不使用)、decay 全 0、`uqe_use_tail_mean` true、`tau_rule` fixed / 32、`full_distribution_query` enabled・fixed / 32(config)、amp false、spatial false(**コード強制**)。baseline の train 終端は τ 0.5 なので、共通設定だけで `[eval]` を明示すると DropMerge の評価条件が変わる
- eval の `clone_model = true` は `common.txt` のスロットキー由来(Agent 側の eval 既定は false)。train_actor は clone true / sync constant 200
- target は `A2.use_optimistic_target = false`(`:193`)が A1 の true に勝ち Greedy。`true` が効いているのは GridMaze(A1)と Munchausen テスト
- 診断メトリクス `51_eval1/41-55`・`52_eval2/41-60` は全て eval actor の policy が出す `$action_info`(`$eval.[tag] @train`)であり、`$agent eval_policy.*` を読む metrics は存在しない。成立条件は 3 群で異なる:
  - (i) UQE margin / NOOP margin(`41-44`、`51-52`。キー `action_uqe_*` / `episode_start_action_*`。NOOP は action index 0): eval policy が UQE / Thompson なら quantile_mode に依らず数値(`dqn_based_agent.cpp:1065-1108`)
  - (ii) `iqn_*` と `action_full_q_margin`(`53-55`): `quantile_mode == "iqn"` かつ UQE かつ full query enabled のときだけ数値(`:1656-1695`)。**QR では NaN が正常**
  - (iii) PRD 048 crossing(`52_eval2/56-60`): QR では policy 種別に依らず無条件(`:1285-1287`)、IQN では full query fixed のとき(`:1650-1652`)数値
  - QR51 は `run.@qr51_control`、IQN32 は現行チェーンで起動できる

## 3. ゴールと非ゴール

- **ゴール**: 「Actor 設定」をフレームワーク横断の概念として定義し、Runner(train / configured eval / EvalPanel)が**名前参照**で Actor 設定を選び、eval スロットごとに独立に方策と network を指定できる。設定の階層はフレームワークの概念を直接表す
- **非ゴール**: 本数(060 済)、スケジュール(052 済)、background snapshot 順序(912)、best checkpoint(913)、eval1 / eval2 の運用判断、eval ε の段階スケジュールそのもの(BTR 相当の ε=0 評価は常時 ε=0 の別スロットで実現する)

## 4. 設計原則(グリルでユーザーが提示)

1. 評価環境は専用の env と actor を持つ(マルチエージェント対応時は複数 agent になる)
2. 評価に必要なのは Model でも Agent でもなく本質的に Actor。`CreateActor` のための「Actor 設定」が要る
3. Actor 設定は Agent 設定と分離し、DQN 固有でなくフレームワーク全体の枠組みとして用意する
4. 記法の考え方は ENV と同じ(ベースは Agent 側、必要なら継承で差分)。Config 継承(`$`)を使う
5. 実装形は ConfigData 前提を検討する → 結論: Agent は構築時に typed カタログへ変換し、生 ConfigData は保持しない
6. `clone_model_override` は Agent 固有事項が個別値として漏れたもの。Clone の概念が無いアルゴリズムもあるので、Agent 側の Actor 設定へ吸収して廃止する
7. actor は元の NetworkModel と関連付く(直接参照または同期)
8. EvalPolicy のスケジュール枠は必要。Runner は Train 側の StepCounts を渡す。Train 側 / Eval 側 StepCounts の両立を整理する

## 5. 確定契約

### 5.1 用語(CONTEXT.md に登録済み)

| 用語 | 意味 |
|---|---|
| **Actor 設定**(actor config) | `CreateActor` が消費する設定の総体(方策とそのスケジュール、network 選択、clone / 同期周期、推論精度、探索器)。スキーマは Agent が所有し、`<AgentPrefix>.actor.[key].*` のカタログ項目として宣言する。Learner 側の `target_policy` は含まない |
| **Actor キー**(ActorKey) | 利用者が Actor 設定カタログで明示する Actor 設定の identity。`run.train.actor_key` / `run.eval.[tag].actor_key` が参照し、省略時は Runner 名(`train` / eval タグ名) |
| **Actor 生成要求**(ActorRequest) | Runner が Agent へ渡す「どの env に、どの device と seed で、どの Actor キーの Actor を作るか」の宣言。用途ラベル(RunMode)を含まない |
| **学習側 counts**(source counts) | Actor のスケジュールと snapshot 判定に使う、直近の Sync 時点の train runner の StepCounts。train runner 自身は live。eval runner 自身の counts(eval 座標系)とは別 |
| **`$actor`** | metrics の参照先で、当該 Runner の Actor を指す。ε・温度など Actor 所有の実行時値を読む |

### 5.2 Agent インタフェース

```cpp
struct ActorRequest {
    BatchEnvSpec batch_env_spec;   // 既存引数そのまま
    EnvSpec env_spec;              // 既存引数そのまま
    torch::Device device;          // Runner が決める推論 device(既定 agent device、eval は run.eval_device_type)
    seed_t seed;                   // Runner が master seed から派生した Actor 専用 seed(domain "actor/<Runner 名>")
    std::string actor_key;         // Actor 設定カタログのキー
};
virtual std::shared_ptr<Actor> CreateActor(const ActorRequest& request) const = 0;
```

- `run_mode` / `clone_model_override` / `device` の引数は廃止する。Agent 実装は RunMode を参照しない(`IsForTarget`、`IsEval` 分岐、`GetRandomGenerator(RunMode)`、`ActionContext(RunMode)` を削除)
- `Actor` は `Module` を実装する(`GetScalar` / `GetTensor` / `GetTensorVector` / `GetConfigData`)。`Runner::GetActor()` を追加する
- RunMode の enum、`RunModeFromString`、Env 側の利用は本 PRD では変えない(Eval1 / Eval2 は Env 用途にだけ残り、P3 で再考する)

### 5.3 Actor 設定カタログと参照規則

```text
# Agent 側(定義側)。各 Agent が自分の prefix 配下にカタログを持つ
DefaultDQNAgent.actor.[train].policy.policy_type = UQE        # 現 train_policy.*
DefaultDQNAgent.actor.[train].clone_model = true              # 現 train_actor.clone_model
DefaultDQNAgent.actor.[train].sync_interval.value = 400       # 現 train_actor.sync_interval.*
DefaultDQNAgent.actor.[eval].policy.policy_type = Greedy      # 現 eval_policy.*(既定導出をやめて明示)
DefaultDQNAgent.actor.[eval].network = online
DefaultDQNAgent.actor.@target.network = target                # 差分プロファイル(未選択なら dormant)
DefaultDQNAgent.actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target

# Runner 側(利用側)。名前参照のみ
run.train.actor_key = train                 # 既定 train。推論専用 Run なら eval を指せる
run.eval.[eval_target].actor_key = eval_target   # 省略時はタグ名。この行は書かなくてよい
run.eval.[eval].actor_key = eval
run.eval.[nature].actor_key = nature        # DefaultDQNAgent.actor.[nature].$ = …[eval] > …@nature を定義して参照
```

- **参照は名前だけ**。スロット内に Actor 設定の上書き層は持たない。差分はカタログ側で `$` 継承と `@` プロファイルを使って書く(直書きはチェーン結果に負ける既存規則どおり)
- **既知キー `train` / `eval` は Runner の既定値**であり、Agent 実装は名前を解釈しない。推論専用 Run は `run.train.actor_key = eval` で `[train]` を定義せずに済む。**唯一の例外**は DefaultDQN の `use_optimistic_target=true` で、Learner の `target_policy` の既定コピー元として `actor.[train].policy` を参照する(Learner を回す構成でしか使わないので `[train]` は定義済みが前提。未定義なら fail-fast。§5.4)
- **eval タグ名を改名する**: eval1 → `eval_target`、eval2 → `eval`。省略時の Actor キーがタグ名なので、`common.txt` は `run.eval.[eval_target]` / `run.eval.[eval]` を宣言するだけで各 Agent の `actor.[eval_target]` / `actor.[eval]` へつながる。target net の無い ImageCls / MuZero は env ファイルで `run.eval_schedule.[eval_target].interval = 0`(dormant)にする
- **metrics の tag 名(LHS)は全て不変**。RHS の `$eval.[eval1]` → `$eval.[eval_target]`、`$eval.[eval2]` → `$eval.[eval]` だけを再指定し、`21_eval/01_target_reward` = target net、`02_policy_reward` = online net、`51_eval1/*` = target、`52_eval2/*` = online という過去 Run との意味の一致を保つ
- 上書き層(A1 / A2 / A3 等)は `A2.actor.[eval].policy.eps_start = 0.01` のように**カタログ項目へ**書く。`[eval_target]` へは PRD 072(選択コピーが source の最終値と最終キー集合を読む。書き込み優先順位は不変)によって伝播する。CLI で `actor.[eval].policy.*` を直接上書きした場合も同様に伝播する(PRD 072 §4-6)
- **fail-fast**: 参照先 `actor.[<key>]` が未定義 → `ANET_SYSTEM_ERROR`(参照元 Runner 名・キー・定義済みキー一覧・`actor_key` の指定方法を含める)。`network` の未知値、`clone_model=false` で actor device ≠ agent device、MuZero での `clone_model=true`(非対応)も fail-fast
- **dormant スロット**(定義済みだが有効 schedule 無し)は Actor を作らないので `actor_key` を解決しない(定義側の宣言検証だけ行う)。EvalPanel が参照する definition-only タグは Actor を作るので解決する

### 5.4 各 Agent のカタログスキーマ(`<Agent>.actor.[key].*`)

| Agent | キー | 備考 |
|---|---|---|
| DefaultDQNAgent | `policy.*`(現 ActionPolicyConfig の全キー: `policy_type` / `eps_*` / `use_spatial_exploration` / `spatial_scale_type` / `uqe_*` / `use_amp` / `use_amp_bf16` / `tau_rule.*` / `full_distribution_query.*`)、`network = online\|target`(既定 online)、`clone_model`(bool)、`sync_interval.*`(ProfiledValueConfig<step_t>。**宣言されたときだけ**定期 snapshot を持つ) | 構築時に `MakeSubConfigData("<prefix>.actor")` で全項目を typed に読み `std::map<std::string, DQNActorConfig>` で保持。既存検証(`validate_tau_rule` / `validate_distributional_policy`)を名前付き namespace の free 関数へ昇格して各項目へ適用。`target_policy.*` は Learner 側として据え置き。**`use_optimistic_target=true` のコピー元は `actor.[train].policy`**(コピー後の強制と `target_policy.*` の明示上書きは現行どおり。`[train]` 未定義なら fail-fast)。**IQN の `taus` spec shape は `learner.iqn.current_taus.num_taus`(N)から取る**。Actor の K はカタログ項目ごとに自由で、Learner の N / M とも一致を要求しない(spec K は dummy forward 専用の placeholder)。`[eval]` / `[eval_target]` の `clone_model` は明示する(現行はスロットキー由来で true) |
| RainbowAgent | `policy.eps_start / eps_end / eps_decay_steps`、`network`、`clone_model` | eval は `[eval]` の policy を使う(現状は train ε で eval している潜在バグの修正=**意図した挙動変更**)。`[eval]` / `[eval_target]` の `clone_model = true` を明示 |
| MuZeroAgent | `temp_start / temp_end / temp_decay_steps`、`add_exploration_noise`、`clone_model` | 現 `actor.temp_*` と RunMode 分岐(温度 0 / noise 無し)を `[eval]` の設定へ。`mcts.*` は Agent 側のまま(gate)。`clone_model` は false のみ(現行もスロットの true を無視している) |
| ImageClsAgent | `clone_model`、`bf16` | `bf16.actor` を移設(`bf16.enabled / learner` は Agent 側に残す)。現行の eval clone はスロットキー由来 true なので `[eval]` に `clone_model = true` を明示 |

### 5.5 policy の所有とスケジュール

- policy は **Actor が専有**する(`CreateActor` ごとに生成)。Agent は `train_policy_` / `eval_policy_` を持たない。`target_policy_` は Learner 側に残る
- スケジュールは Actor が `MakeAction(step, state)` の冒頭で `policy->UpdateSchedule(step)` を呼んで進める。Learner は actor policy に触れない(`OnLearn` の配布は廃止。`target_policy_` だけ Learner が同名 API で進める)。これにより learner thread の書込と eval actor の無同期読みが消える
- **MakeAction の step は学習側 counts**: train runner は自分の live counts、EvalRunner は `Sync(const StepCounts& source_counts)` で受け取った値を保持して渡す。configured eval は `RunSession(event_counts)` が `Sync(event_counts)` を呼び、EvalPanel は `train_runner->GetCounts()` を渡す。`Actor::Sync()` は無引数のまま。eval runner 自身の `step_counts_` は TrainEvent(eval 座標系の metrics)用に残す
- ProfiledValue で ε / τ を書けるようにする「枠」はこの `UpdateSchedule(step)` フックまでとし、ProfiledValue 自体は導入しない(`eps_start / eps_end / eps_decay_steps` は現状維持)
- 互換の範囲: Sync の回数・タイミング、イベント列、step 座標系は不変。train actor の ε は「前 step の UpdateFromBatch 時点」から「当 step の MakeAction 時点」へ 1 step 前倒しになる(同じ f(exp_step))。現用 config の eval ε は全て定数なので eval の挙動は同じ

### 5.6 Actor Q ヒント・seed・clone

- Actor Q ヒントの送出は役割を見ず Agent の PER 設定(`use_per && ACTOR_APPROX`)だけで決める。全 Actor が付け、eval runner は無視する
- seed は Runner が `master_seed_->GetGroupSeed("actor/<Runner 名>")` で派生して request に載せる。RunMode 別共有 RNG は廃止し、EvalPanel と configured eval が同じ stream を消費する現状の結合を解く
- `clone_model` は Actor 設定の事項。`ValidateSharedActorDevice`(trainer)と各 Agent の重複チェックは `AgentBase::ValidateActorDevice(bool clone_model, const torch::Device& actor_device)` に共通化する。`run.eval_device_type / index` は Runner 側に残り、request の `device` になる
- EvalPanel の `model_sync.mode = shared` は廃止する(clone は参照タグの Actor 設定が決める。`frame / time / episode` の同期スケジュールは GUI の責務のまま)。EvalPanel は `app.*.eval_panel.eval_config_tag` で参照するタグの `actor_key` と env prefix を鏡写しする

### 5.7 metrics

- `EventField::ACTOR` と `$actor` を追加し、`event.runner->GetActor()` で解決する。`@train` / `@learn` / `@episode_end` / `@session_end` のいずれも runner を持つので `$actor` を使える。`$eval.[tag] $actor epsilon` でスロット別の ε を観測できる
- `$agent epsilon` / `$agent uqe_tau` / `$agent train_policy.*` / `$agent eval_policy.*` は削除する(クリーンブレーク)。`$agent target_policy.*` は Learner 側なので維持
- 既存 tag の LHS は不変。`32_agent_base/05_epsilon_learn` 等は RHS を `$actor epsilon` へ移す

## 6. フェーズ

| Phase | 内容 | 位置付け |
|---|---|---|
| P0 | **PRD 072**: 設定リゾルバの選択コピーが source の最終値を読むようにする(書き込み優先順位は不変。別 PRD、先行) | §5.3 の `[eval_target].$ = …[eval] > …@target` が後段 overlay を取りこぼさないための前提 |
| P1 | **`train.` root → `run.` 改名**(機械的・別コミット) | `run.seed` / `run.train.num_envs` / `run.train.runner_type` / `run.eval_device_type` / `run.eval.[tag].*` / `run.eval_schedule.[tag].*`。単独で成立し、P2 の新キーは新 root の下に生まれる |
| P2 | **Actor 設定カタログ**(本体) | §5.2〜5.7 の全部と config 移行。単独でゴールを達成する |
| P3 | **env 側**(方向のみ) | `<Env>.[key].*` カタログ + `run.train.env_key` / `run.eval.[tag].env_key`(省略時タグ名)。スロット内 `env.*` 上書き層の存廃、`run_mode` キーと Eval1 / Eval2 の Env 側整理は着手時にグリル |

P0 / P1 は単独で成立し、P2 で止めても P3 無しで一貫した状態になる。

## 7. 実装ノート(Codex 向け)

### 7.1 P1: `train.` → `run.`

| 旧 | 新 |
|---|---|
| `train.seed` | `run.seed` |
| `train.num_envs` | `run.train.num_envs` |
| `train.main_runner_type` | `run.train.runner_type` |
| `train.eval_device_type` / `_index` | `run.eval_device_type` / `_index` |
| `train.eval.[tag].*` | `run.eval.[tag].*` |
| `train.eval_schedule.[tag].*` | `run.eval_schedule.[tag].*` |

- 範囲: `trainer.cpp`(literal とエラー文言)、`config.cpp:243-253` のコメント、`RunnerApp.cpp`、テスト 7 ファイル、`viewers/metrics-tools/inspect_run.py`(+ `inspect_run_test.py`)、`apps/runner/tools/dropmerge_optuna.py`、config 8 ファイル(`run.@x : train.…` の trunk 行を含む)、`docs/design` 7 ファイル、CONTEXT.md 3 項、AGENTS.md
- `run.` は trunk(`run.$` / `run.@<name>`)と同じ root だが、リゾルバが特別扱いするのはその 2 形だけ(`config_impl.cpp:81-82`)なので素の `run.*` は通常キーとして通る。trunk 内の `run.@x : run.seed = 2` は完全キーの追記として今と同じ意味
- 過去 Run artifact・実験記録・ADR は当時の記録として変更しない
- 受入: 全テスト緑、`inspect_run_test.py` 緑、smoke Run の `config_data.txt` がキー名置換を除いて一致

### 7.2 P2: フレームワーク

| 対象 | 変更 |
|---|---|
| `rl.hpp` | `ActorRequest`、`Agent::CreateActor(const ActorRequest&) const`、`Actor : public Module`、`Runner::GetActor()`、`EventField::ACTOR` |
| `agent.hpp` / `agent.cpp` | `ActionContext(seed)`(RunMode 除去)、`AgentBase` から `GetRandomGenerator(RunMode)` / `run_mode_rngs_` / `action_context_seed_` を除去、`ValidateActorDevice` helper |
| `trainer.hpp` / `trainer.cpp` | `run.train.actor_key`(既定 `train`)、`run.eval.[tag].actor_key`(既定 tag)、seed domain `actor/<name>`、`RunnerBase(env, agent, notifier, ActorRequest, name)`(run_mode / clone_model_override / device 引数は廃止。`run.eval.[tag].run_mode` は Env 用に読み続ける)、`EvalRunner::Sync(const StepCounts&)`、`RunSession(event_counts)` → `Sync(event_counts)`、`DoStepInternal` は `source_counts_` を MakeAction へ、`CreateEvalRunner(name, config_tag)`、`ValidateSharedActorDevice` 削除、dormant タグは actor_key を解決しない |
| `observers.cpp` | `$actor` のパース(`:1287` 付近)と `event.runner->GetActor()` による解決 |
| `apps/runner` | `EvalPanelModelSyncMode::Shared` と `UsesClonedModel()` を削除、`RunnerFrame` は `CreateEvalRunner("EvalPanel", eval_config_tag)`、Sync 時に train runner の counts を渡す |

### 7.3 P2: DefaultDQNAgent

- `DQNActorConfig { ActionPolicyConfig policy; std::string network = "online"; bool clone_model = false; std::optional<ProfiledValueConfig<step_t>> sync_interval; }`。`DefaultDQNAgentConfig` の構築時に `MakeSubConfigData("<prefix>.actor")` で全 `[key]` を読み、各項目に検証を掛けて保持する。生 ConfigData は保持しない
- `ActionPolicyConfig` の読み取りを field 単位の helper(`ReadActionPolicyConfig(const ConfigData& sub, const std::string& prefix, ActionPolicyConfig&)`)に一本化する(現状 train / eval / target で 3 重複。`ANET_READ_CONFIG` は `[key]` を含む式に使えない)。`train_policy.*` / `eval_policy.*` / `train_actor.*` の読み取りと eval 既定導出(`hpp:102-116`)を削除する
- カタログの読み取りは `ReadConfig(config_data, "actor.[" + key + "].policy.eps_start", value)` のように**キー文字列を組み立てて `Config::ReadConfig` で読む**。これで `my_config_data_` / `my_config_json_` に記録され、Module Config(`GetConfigData()`)・config dump(`config/DefaultDQNAgent.txt`)・Save の `ToString()` に `actor.[key].*` が**既定補完後の値**で載る(§7.5 の typed 比較と §8 ① の前提)
- NN 入力 spec: `taus` の shape を `config_.learner.iqn.current_taus.num_taus` から作る(`default_dqn_agent.cpp:165-173` の置き換え)。`Network::Clone` は同じ spec で再 build するので Actor の K には依存しない。`net.structure` / `net.detail` に出る `taus` shape が N になる
- `use_optimistic_target=true`: `target_policy` の既定を `actor.[train].policy` からコピーする(eps / uqe_eps の 0 強制、EpsilonGreedy → Greedy、`tau_rule = {fixed, 32}`・`full_distribution_query` 既定化、`target_policy.*` の明示上書き、の順序は現行どおり)。`[train]` 未定義なら `ANET_SYSTEM_ERROR`(`use_optimistic_target` と `actor.[train]` の両キーをメッセージに含める)
- `CreateActor(request)`: カタログ lookup(未定義は一覧付き fail-fast)→ `CreateActionPolicy(cfg.policy, cfg.policy.use_spatial_exploration, request.batch_env_spec.num_envs, request.device)`(const 化)→ `network` で online / target を選択 → `clone_model` なら Clone → `DefaultActionContext / StackerActionContext(request.seed, device)` → Actor 生成。Q ヒントは §5.6、定期 snapshot は `clone_model && sync_interval.has_value()`
- `dqn::Actor`: `policy_` を専有し、`MakeAction` 冒頭で `policy_->UpdateSchedule(step)`。`GetScalar("epsilon" / "uqe_tau")` を policy へ委譲、`GetConfigData()` で実効 Actor 設定を返す
- `IsForTarget`、`train_policy_` / `eval_policy_`、`GetScalar` の `train_policy.` / `eval_policy.` / bare `epsilon` / `uqe_tau` 経路を削除。`UpdateFromBatch` は `target_policy_->UpdateSchedule(counts)` だけ残す

### 7.4 P2: 他 Agent

- RainbowAgent: `actor.[key].{policy.eps_start, policy.eps_end, policy.eps_decay_steps, network, clone_model}`。`rainbow_agent.cpp:208` の未使用 `policy` を整理し、eval は `[eval]` の policy を使う
- MuZeroAgent: `actor.[key].{temp_start, temp_end, temp_decay_steps, add_exploration_noise, clone_model}`。`MuZeroActor` から run_mode_ を除去。`clone_model=true` は非対応として fail-fast
- ImageClsAgent: `actor.[key].{clone_model, bf16}`。`ImageClsActor` から run_mode_ を除去

### 7.5 P2: config 移行

**手順(キー移動ではなく、env ごとの typed 実効値の移植)**

1. P2 着手前に現行コードで、各 env config(Atari / DropMerge / LunarLander / GridMaze / GridMaze_muzero / CartPole / ImageCls)の代表 Run プロファイルについて Agent の Module Config dump(`config/DefaultDQNAgent.txt` 等)を採取する。これは `ANET_READ_CONFIG` が既定補完後の値を記録したものなので、コード導出(eval の τ = train 終端、uqe_eps 0、spatial false、`use_optimistic_target` のコピー)を含む**実効値**である
2. 新カタログ(`actor.[train]` / `[eval]` / `[eval_target]` と `target_policy.*`)を、その実効値と一致するように書き出す。旧キー → 新キーの対応表(`train_policy.X` → `actor.[train].policy.X`、`eval_policy.X` → `actor.[eval].policy.X` と `actor.[eval_target].policy.X`、`train_actor.X` → `actor.[train].X`、スロット `clone_model` → `actor.[eval*].clone_model`)を PRD 実装ノートに残す
3. 移行後の dump を対応表で照合し、**全フィールド一致**を確認する。UQE では不使用の `eps_*` など「効かないフィールド」も差があれば明示して一致させる(判定を policy 種別に依存させない)

DropMerge の期待値(現行 §2.8 と同値):

| 項目 | `[eval]` / `[eval_target]` の期待値 |
|---|---|
| `policy.policy_type` | UQE(両方) |
| `policy.uqe_tau_start / uqe_tau_end / uqe_tau_decay_steps` | 0.85 / 0.85 / 0 |
| `policy.uqe_eps_start / uqe_eps_end / uqe_eps_decay_steps` | 0 / 0 / 0 |
| `policy.eps_start / eps_end / eps_decay_steps` | 0.05 / 0.05 / 0(UQE では不使用だが一致させる) |
| `policy.uqe_use_tail_mean` | true |
| `policy.tau_rule` | fixed / 32 |
| `policy.full_distribution_query` | enabled、fixed / 32 |
| `policy.use_amp / use_amp_bf16` | false / false |
| `policy.use_spatial_exploration` | false(spatial は `[train]` だけ) |
| `network` | `[eval]` = online、`[eval_target]` = target(それ以外は同一) |
| `clone_model` / `sync_interval` | true / 無し(現行のスロット clone true・eval に定期 snapshot 無しを維持) |
| `[train]` | policy = 現 `train_policy`(UQE、spatial true 等)、clone true、sync constant 200 |
| `target_policy.*` | 現行と同値(`use_optimistic_target = false` なので Greedy) |

GridMaze(`A1.use_optimistic_target = true`)は、コピー元が `actor.[train].policy` に変わっても `target_policy` の実効値が不変であることを同じ dump 比較で確認する。

- `agent.txt` `@baseline`: `actor.[train].{policy.*, clone_model, sync_interval.*}`、`actor.[eval].{policy.*(明示), network = online}`、`actor.@target.network = target`、`actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target`
- env ファイル: `A?.train_policy.*` → `A?.actor.[train].policy.*`、`A?.eval_policy.*` → `A?.actor.[eval].policy.*`、`A?.train_actor.*` → `A?.actor.[train].*`、`run.@…` の同種行も同様
- eval タグ改名(eval1 → `eval_target`、eval2 → `eval`)を 8 ファイル 194 箇所へ(metrics RHS `$eval.[…]`、`run.eval_schedule`、`app.*.eval_panel.eval_config_tag` を含む)。`run_mode = eval1 / eval2` 行は削除(既定で IsEval)。スロットの `clone_model` 行は削除し、各 Agent の `[eval]` / `[eval_target]` に `clone_model = true` を明示する(ImageCls も true、MuZero は false)
- ImageCls.txt / GridMaze_muzero.txt: `run.eval_schedule.[eval_target].interval = 0`
- `run.@greedy_eval`: `A3.actor.[eval].policy.eps_start / eps_end = 0`(eval と eval_target の両方へ効く)、または専用項目 `actor.[greedy]` + `run.eval.[greedy]`。「スロットごとの指定は PRD 061」の注記を削除
- metrics: `$agent epsilon` / `$agent uqe_tau` → `$actor epsilon` / `$actor uqe_tau`(`metrics_scalar.txt:26-27`)。`$agent target_policy.uqe_tau` は据え置き
- Rainbow / MuZero / ImageCls の config にも `actor.[train]` / `actor.[eval]` を明示する

### 7.6 テスト

- 既存ダブル 5 箇所を新シグネチャへ
- 新規: `actor_key` 既定(タグ名)と明示、未定義キーの fail-fast(一覧付き)、`network` 選択、Actor 別 seed で 2 スロットの RNG が独立、`$actor epsilon`、Rainbow eval が `[eval]` policy を使う、MuZero `[eval]` が温度 0 / noise 無し、EvalPanel が参照タグの actor_key を使う、`clone_model=false` の device 不一致 fail-fast、dormant スロットの未定義 actor_key が無視される、`UpdateSchedule` が MakeAction の step で進む
- 新規(2026-09-08 追加): `[train]` 未定義の IQN 構築と `[eval]` Actor 生成が通る、K の異なる 2 Actor(shared / clone の両方)が同一 network で forward できる、Actor の K ≠ Learner の N / M(既存 `dqn_based_agent_test.cpp:2909` を維持)、`use_optimistic_target=true` で `[train]` 未定義 → fail-fast、Actor(EvalPanel 相当 / 別スロット)を追加しても既存 Actor の行動列が不変、カタログ項目が Module Config dump に既定補完後の値で載る

## 8. 受入条件

1. Atari で `run.eval.[eval_target]`(target × greedy)、`[eval]`(online × ε 0.01)、`[nature]`(online × ε 0.05。`actor.[nature].$ = …[eval] > …@nature`)、`[greedy]`(online × ε 0)を同時に持つ smoke Run 1 本(`_tmp_`)が完走し、各スロットの `$actor epsilon` が metrics で期待値になる
2. `run.@greedy_eval` の「PRD 061 待ち」注記が消えている
3. Agent 側の grep で `RunMode::Eval1` / `RunMode::Eval2` / `IsForTarget` / `clone_model_override` の参照が 0
4. `anet-core-test` 全緑(既知の既存失敗を除く)+ §7.6 の新規テスト
5. 等価性(段階化。同 seed の新旧バージョン間 bit 一致は Actor 別 seed と seed domain の変更で不成立なので、次の順で確認し、各段が通れば次へ進む):
    1. **設定**: §7.5 の Module Config dump 比較で全 env・全フィールド一致
    2. **固定入力推論**: P2 着手前に、DropMerge と Atari の checkpoint + 固定 obs バッチに対する eval actor の Q 値と行動を記録し、P2 後に同一入力・同一 checkpoint で一致させる(greedy / fixed τ の policy で RNG 非依存にする)
    3. **metrics**: Atari ε smoke(受入 1)と DropMerge smoke(受入 7)
    4. **短時間 throughput**: DropMerge で train step/s と eval セッションの wall-clock を修正前後で比較する。全 Actor への Q ヒント送出(§5.6)で eval に増える計算はここで測る。機体ドリフト(1 時間で最大 8%)があるため前後をラウンドロビンで取る
    5. **学習比較**: ④ までで差が疑われた場合だけ、同一構成の複数 seed Run の終盤平均をブレ幅基準で比較する(任意ステップ。予算はその時点で決める)
6. P1: smoke Run の `config_data.txt` がキー名置換を除いて一致
7. DropMerge smoke(IQN32 = 現行チェーン、QR51 = `run.@qr51_control` の 2 本、`_tmp_`・短い予算で各スロットが 2 セッション以上発火)。判定は §2.8 の 3 群を mode 別に見る:
    - IQN32: (i) UQE / NOOP margin(`41-44`、`51-52`)、(ii) `53-55`、(iii) PRD 048 crossing(`52_eval2/56-60`)の全てが数値
    - QR51: (i) と (iii) が数値。(ii) `53-55` は **NaN が正常**(数値が出たら逆に異常)
    - 両 mode 共通: `51_eval1/*` = `[eval_target]`(target net)、`52_eval2/*` = `[eval]`(online net)、`$actor` の τ / ε が期待値
8. `net.structure` / `net.detail` の `taus` shape が Learner の N に変わることを認知する(受入ではなく既知の差分)

## 9. 複雑さ監査(グリル 2026-09-07 最終)

| # | 機構 | 裁定 | 理由 / 切ったら戻る痛み |
|---|---|---|---|
| 1 | ActorRequest 構造体 | keep | 引数追加ごとに実装 4 + ダブル 5 の改修 |
| 2 | Actor 設定カタログ `<Agent>.actor.[key]` | keep | 中核(実害 A〜H) |
| 3 | `actor_key`(省略時タグ名) | keep | 参照記法。common.txt が Agent 非依存のまま |
| 4 | RunMode を Agent IF から撤去 | keep(設計原則) | Agent 内の train / eval 固定分岐 |
| 5 | `network = online\|target` | keep | 実害 D |
| 6 | Actor 別 seed | keep | 4 の帰結。EvalPanel と eval1 の RNG stream 共有 |
| 7 | Actor 所有 policy + `UpdateSchedule(step)`、OnLearn 廃止 | keep | learner thread の無同期書込、eval スケジュールの別時計 |
| 8 | `EvalRunner::Sync(source_counts)` | keep | 7 の前提 |
| 9 | `$actor` + Actor の Module 化 | keep | `$agent epsilon` の代替 |
| 10 | clone_model を Actor 設定へ、override 廃止 | keep(原則 6) | 実装で縛る個別値 |
| 11 | eval 既定導出の廃止 | keep | 隠れ規則 |
| 12 | ProfiledValue 化の枠 | shrink | `UpdateSchedule(step)` フックのみ。導入は gate |
| 13 | `model_sync.mode = shared` の廃止 | keep | 10 の帰結 |
| 14 | MuZero `mcts.*` の per-actor 化 | defer | 実需未確認 |
| 15 | スロット内 actor 上書き層 | cut | カタログ側 `$` で代替 |
| 16 | env カタログ化 | P3 | 対称性のみ |
| 17 | Rainbow の eval policy 分離 | keep | 潜在バグ修正 |
| 18 | `run.` 改名 | P1 | 命名負債 |
| 19 | policy カタログ + `policy_key` | cut | PRD 072 で不要 |
| 20 | リゾルバの選択コピーの最終値読み(不動点反復) | 別 PRD 072(先行) | カタログ項目間の継承が後段 overlay を取りこぼす |
| 21 | `use_optimistic_target`(コピー元を `actor.[train].policy` へ) | keep | optimistic 化のたびに `target_policy.*` を丸ごと書き換える保守コスト。D1 の唯一の例外として明記 |
| 22 | IQN の spec K を Learner の N から取る | keep | `[train]` 非依存の構築。runtime の K はカタログ項目ごとに自由 |

最小解(却下した暫定案)との差分 4 / 6 / 7 / 9 / 10 / 18 は設計原則(概念整合)で受容した。P2 は Agent 4・Runner・Observer・GUI・全 config に及ぶ大規模改修である。

ゲート(将来): ProfiledValue による ε / τ スケジュール(段階スケジュールを常時 ε=0 スロットで代替できなくなったら)、MuZero `mcts.*` の per-actor 化(train / eval で simulations を変えたくなったら)、`target_policy.*` の Actor 設定側への統合(Learner 側方策を揃えたくなったら)、env カタログ化と Eval1 / Eval2 の廃止(P3)。

## 10. 却下した案と経緯

- **暫定案: スロット内 `train.eval.[tag].agent.eval_policy.*` の上書き**(2026-09-06 深夜に「今夜中の暫定」として設計)。実害 A〜H は満たすが、Agent / Actor の定義まで遡らない場当たりであり撤回した。§9 の「最小解」に相当する
- **旧候補 A / B / C(起票時)**: A(スロットが policy 設定を直接持つ)は trainer が Agent スキーマを知る漏れ、B(named eval policy カタログ)は本設計の Actor 設定カタログに発展、C(run_mode を分解して network 軸をスロットへ)は `network` キー + RunMode 撤去で代替。起票時の「D は B+C が要る」は誤りで、run_mode はスロット単位で一意性を要求されないため B だけでも組めた
- **2 ベース案(`train_actor` / `eval_actor` + スロット `actor.*` 上書き)**: Agent 実装に train / eval の固定分岐が残る
- **override prefix の併設(ENV と完全対称)**: request が 2 フィールドになり、Agent 側に未消費キー検査が要る。差分はカタログ側 `$` で書けるので過剰
- **RunMode を request に残す**: 用途ラベルで分岐する構造が残る
- **policy カタログ + `policy_key`**(`DefaultDQNAgent.policy.[key]`): リゾルバの宣言順コピー問題の回避策として提案したが、似た概念の二重化になる。リゾルバ側を直す(PRD 072)ことで不要
- **`${}` 値参照で `[eval_target]` を同期**: policy 17 キー分の行が要り 1 段限定
- **eval タグ名の維持(eval1 / eval2)**: 名前が意味を持たず、Actor キーと一致させられない。改名して metrics RHS だけ再指定した
- **`use_optimistic_target` の廃止**(2026-09-08 検討): 実効値は明示指定で同じになるが、optimistic 化のたびに `target_policy.*` を丸ごと書き換える(戻しも含む)保守コストが戻る。コピー元を `actor.[train].policy` に変えて維持する
- **IQN spec K の専用キー(`net.taus_shape` 等)/ カタログ全項目と Learner の最大 K**: Learner の N で足りる。設定項目やカタログ走査を増やさない

旧 D1〜D10(起票時の未決)との対応: D1(policy 軸 / network 軸)→ §5.3 / §5.4、D2(置き場と記法)→ §5.3、D3(既定と後方互換)→ クリーンブレーク(§5.3、§7.5)、D4(CreateActor の変更形)→ §5.2、D5(未定義名)→ §5.3 fail-fast、D6(metrics タグ名)→ LHS 不変・RHS 再指定、D7(スロット数)→ 任意、D8(wall-clock)→ 060 P3 の範囲、D9(060 との順)→ 060 完了済み、D10(OnLearn)→ §5.5。

## 11. 文書同期

- CONTEXT.md(用語 5 件追加、6 項修正)と ADR 0038 は本 PRD と同時に更新済み
- `docs/design`(`100_runtime_and_configuration.jp.md` §7 キー表、`110_agents_and_learning.jp.md` §2.1-2.2 / §6.1 / §7.1、`200_dqn_agents.jp.md` §2.3 / §6.1 / §6.4 / §7.2、`010_framework_overview.jp.md`、`160_applications_and_tools.jp.md`、`020_user_guide_run.jp.md`)は P1 / P2 の code / config 変更と同じ変更で同期する
- `docs/experiments` と過去 Run artifact は当時の記録として保持する
