# EpisodeEvalObserver の Eval メトリクス強化 — 実装計画

- 関連ドキュメント `docs/specs/003_eval_10spec.md` は初期の構造診断（Eval 専用イベント案を含む初期検討）であり、**最終仕様は本書を正とする**。10spec の Eval 専用イベント案は採用せず、Train/Eval 共通 `EpisodeEndEvent` に統一している。
- 用語: `tag` は TensorBoard 由来の **メトリクス識別子**、`name` は `train.eval.[<name>]` の **EvalRunner インスタンス識別子**。混同しない。

## Context

Eval（評価エピソード）から既存メトリクスパイプラインに乗っているのはエピソード総報酬 1 つだけで、設定文法は
`metrics.scalar.[<tag>] = $runner eval.[<eval_name>].eps_total_reward @learn` という TrainRunner 特殊キー経由の側道になっている。
Eval エピソードの **ENV 統計**（例 `mean.ep_max_rank`）や **Agent 統計** も
`$env`/`$agent`/`$exp`/`$runner` の通常メトリクス文法で取り出したい。

ボトルネックは「`EpisodeEvalObserver` が env / actor / step ループを自前で抱え、Notifier にイベントを一切流さない」こと。
本タスクは **EpisodeEvalObserver を「`EvalRunner` を間欠的に駆動するトリガ」に痩せさせ、エピソード終端で
`EpisodeEndEvent`（Train/Eval 共通）を発火、`MetricsLogEpisodeEndObserver` を新設して
3 軸直交文法（`@<event>` × `$<runner_scope>` × `$<target_field>`）で接続する**、という構造変更で達成する。

## 設計上の決定

- **EpisodeEndEvent は Train/Eval 共通**。`/// @todo EPISODE_END（ENV由来）を追加`（`rl.hpp:748`）の宿題も解消する。
- 設定文法は **3 軸直交**: `@<event>` × `$<runner_scope>` × `$<target_field>`。
- 用語: `tag` は TensorBoard 由来の **メトリクス識別子** に限定。EvalRunner などの **インスタンス識別子** は `name` を使う。

## 全体像

```
旧:
  EpisodeEvalObserver (LearnObserver)
    ├ BatchEnv (自前) ─── env_->Step()
    ├ Actor (自前)
    └ report_function_(float total_reward)
         └ TrainRunner::SetEvalLastReward(name, val)
              └ TrainRunner::GetScalar("eval.[<name>].eps_total_reward")  ← 文字列マッチで横取り
                   └ MetricsLogLearnObserver ($runner 経由)

新:
  EpisodeEvalObserver (LearnObserver, 痩せ)
    ├ eval_runner_ (shared_ptr<EvalRunner>)
    ├ eval_pool_ (background)
    └ OnLearn: 間隔到達で eval_runner_->DoStep() を episode 終端まで駆動するだけ

  EvalRunner / TrainRunner (両方を拡張)
    └ DoStep(): per-env で done/truncated のときに Notify(EpisodeEndEvent)
         EpisodeEndEvent { runner, counts, agent, env, env_index, eps_total_reward }

  Notifier ── EpisodeEndObserver 列を追加
                  └ RunnerScopedEpisodeEndObserver(target=<train_runner | eval_runner_[name]>) でルーティング
                       └ MetricsLogEpisodeEndObserver
                            └ $env / $agent / $runner を既存 MetricsLogObserverBase で解決
```

## 1. 変更内容

### 1.1 Event / Observer の追加（`core/anet-core/include/anet/rl.hpp`, `src/rl.cpp`）

- `enum class EventType` に `EPISODE_END` を追加。
- 新規 struct（Train/Eval 共通）:
  ```cpp
  struct EpisodeEndEvent {
      const std::shared_ptr<const Runner> runner;        // TrainRunner or EvalRunner
      const StepCounts counts;
      const std::shared_ptr<const Agent> agent;
      const std::shared_ptr<const BatchEnv> env;         // 発火 Runner の env
      int env_index;                                     // 終端を踏んだ env のインデックス
      float eps_total_reward;                            // その env のエピソード総報酬
  };
  ```
  - `UpdateEvent` からは派生させない（`experience` / `update_result_list` は episode end 文脈で無意味）。
  - tag/name は持たせない（`runner` を辿れば識別可能、`RunnerScopedEpisodeEndObserver` でも runner ポインタ比較で十分）。
- 新規クラス `EpisodeEndObserver`（`OnEpisodeEnd(const EpisodeEndEvent&)` 純粋仮想）と
  `RunnerScopedEpisodeEndObserver`（`RunnerScopedLearnObserver` をひな型に複製、`event.runner != target_runner_` ならスキップ）。
- `Notifier` に以下を追加（既存 Train/Learn 対と同形）:
  - `Attach(std::shared_ptr<EpisodeEndObserver>)` / `Detach` 一式
  - `Notify(const EpisodeEndEvent&)`
  - `episode_end_observers_` メンバ
  - `LogObservers()` に EPISODE_END 行追加
- `Notifier::AttachScoped` テンプレにも `EpisodeEndObserver` 分岐を追加。

### 1.2 EvalRunner / TrainRunner の共通拡張（`include/anet/trainer.hpp`, `src/trainer.cpp`）

EpisodeEnd 発火を **`RunnerBase`** に共通の helper として実装し、`EvalRunner::DoStep()` と
`TrainRunner` 配下（`SerialTrainRunner::DoStep()` / `PipelineTrainRunner::DoStep()`）の両方から呼ぶ。

#### `Runner::DoStep()` の戻り値契約は変えない

- `Runner::DoStep()` は `StepCounts` 戻り値の純粋仮想（`rl.hpp:899`、`trainer.hpp:29, 106`）。`RunnerBase::DoUpdateFrame()` と `RunnerThread::ProcessStep()` がこの契約に依存している。
- したがって **`DoStep()` の戻り値は `StepCounts` のまま維持** し、「直近 step で env 終端を踏んだか」は別 API で公開する。

#### `RunnerBase` に追加

```cpp
protected:
    std::vector<float> eps_total_reward_per_env_;          // env 単位の総報酬累積
    float last_episode_total_reward_                       // 直近確定エピソードの代表値（複数 env なら mean）
        = std::numeric_limits<float>::quiet_NaN();
    bool last_step_had_episode_end_ = false;

    /// 直近 DoStep で env 終端を踏んだ env が 1 つ以上あれば true。EvalRunner を駆動するトリガ側のループ条件。
public:
    bool LastStepHadEpisodeEnd() const { return last_step_had_episode_end_; }

protected:
    /// step 後に呼ぶ helper。env 単位の総報酬を蓄積し、終端 env ごとに EpisodeEndEvent を Notify する。
    /// self は notifier->Notify(EpisodeEndEvent) の event.runner にセットする値。
    /// 戻り値: 当該 step で終端を踏んだ env が 1 つでもあれば true。同時に last_step_had_episode_end_ も更新。
    bool AccumulateAndNotifyEpisodeEnd(
        std::shared_ptr<const Runner> self,
        std::shared_ptr<const BatchStepResult> result);
```

- `AccumulateAndNotifyEpisodeEnd()` の処理:
  1. `result->reward`（shape: [num_envs]）を env 単位で `eps_total_reward_per_env_` に加算
  2. env 内 `IsDone(i) || IsTruncated(i)` が立っている env i ごとに、
     `EpisodeEndEvent{ self, step_counts_, agent_, env_, i, eps_total_reward_per_env_[i] }` を Notify
  3. その env i の累積を `last_episode_total_reward_` に反映してからゼロクリア（複数 env が同時終端した場合は mean）
  4. `last_step_had_episode_end_` を当該 step の終端有無で更新
- 注: `RunnerBase` 自体は `enable_shared_from_this` を継承していない（`EvalRunner` / `TrainRunner` がそれぞれ個別に継承）。よって `self` は呼出元（派生 Runner）で `shared_from_this()` を呼び、helper に渡す。

#### `RunnerBase::GetScalar` の拡張

- `RunnerBase::GetScalar("eps_total_reward")` を実装し、`last_episode_total_reward_` を返す。
- これにより `MetricsLogEpisodeEndObserver` 経由の `$runner` で Eval/Train 両方とも同じキーで引ける。
- **役割分担を明確化**:
  - `EpisodeEndEvent.eps_total_reward` — 終端を踏んだ env_index の **生値（per-env）**。並列 Eval 時のデバッグや per-env 集計に使う。
  - `RunnerBase::GetScalar("eps_total_reward")` — **直近確定エピソードの代表値（複数 env なら mean）**。`$runner @episode_end` 経由はこちらを参照。
- EvalRunner 固有の `GetScalar` オーバーライドは不要（`RunnerBase` に統合）。

#### EvalRunner 拡張

- `std::string name_;` を追加（`RunManager` の `eval_runners_` map のキー、`train.eval.[<name>]` の name と一致）。コンストラクタ末尾に `name` 引数追加。
- `DoStep()` 末尾で `AccumulateAndNotifyEpisodeEnd(shared_from_this(), result)` を呼ぶ。
- `DoStep()` の戻り値型は **`StepCounts` 維持**。

#### TrainRunner 拡張

- `SerialTrainRunner::DoStep()` / `PipelineTrainRunner::DoStep()` の env step 直後で同 helper を呼ぶ（`shared_from_this()` を渡す）。
- 既存の `episode_total_reward_cur_` / `episode_total_reward_comp_` 経路（`train_episode_reward` キー）は本タスクでは **並存** とする。最小変更を優先。
- `EvalRunner::DoStep()` 内の既存 `Notifier::Notify(TrainEvent)`（`trainer.cpp:202-204`）は今回は触らない。
  既存 `RunnerScopedTrainObserver`（target=train_runner）が train_runner 以外を黙殺するので無害。

#### 補足

- 重要: `BatchEnv::Step()` 内で自動 Reset が走っても、`BatchStepResult` の `done`/`truncated` には終端瞬間の値が記録されている。env の集計指標（`ep_max_rank` 等）は **次 Step まで保持される**（`DropMergeEnv.cpp:770, 947, 1251` で確認済み）。Notify は次ステップを踏む前の現位置で行うので有効値が取れる。

### 1.3 TrainRunner 側の Eval 特殊経路を削除

- `include/anet/trainer.hpp`:
  - `TrainRunner::eval_last_rewards_` / `eval_rewards_mutex_` を削除（`trainer.hpp:126-128`）。
  - `TrainRunner::SetEvalLastReward()` 宣言を削除（`trainer.hpp:109`）。
- `src/trainer.cpp`:
  - `TrainRunner::SetEvalLastReward()` 実装削除（`trainer.cpp:228-232`）。
  - `TrainRunner::GetScalar()` の `key.find("eval.[") == 0` ブロックを削除（`trainer.cpp:265-281`）。
- `include/anet/rl.hpp`:
  - `Runner::TARGET_EVAL_REWARD` / `Runner::POLICY_EVAL_REWARD` 定数を削除（`rl.hpp:916-917`）。
- `apps/runner/src/RunnerApp.cpp:374-381` の `event.runner->GetScalar(...EVAL_REWARD)` を使ったログを削除
  （TrainEvent コールバック内で train_runner からの取得を試みていただけのデバッグログ。本タスク範囲外として削除）。

### 1.4 RunManager の Eval 構築（`src/trainer.cpp`）

- `trainer.cpp:696-740` の `EpisodeEvalObserver` 生成ループを以下に書き換える:
  1. name ごとに per-name env を `VectorizedDiscreteBatchEnv` で生成（既存 `observers.cpp:487` のロジックを移植）。
     - `single_env_factory` / `config_data` / `env_device` / `eval_obs_seed` / `config_prefix = "train.eval.[<name>].env"`。
  2. `EvalRunner` を生成（`agent_` / `notifier_` / `run_mode` / `clone_model=true` / `actor_device` / `name`）。
  3. `eval_runners_[name] = eval_runner`（既存 map に登録）。
  4. `notifier_->AttachScoped<EpisodeEvalObserver>(eval_runner, eval_runner, interval, use_background)` で痩せたオブザーバを登録。
- `RunManager::CreateEvalRunner(name, ...)`（`trainer.cpp:752-762`、RunnerFrame 用）はそのまま残す。
  `EvalRunner` コンストラクタには `name` をそのまま渡す。
- `trainer.hpp:209` のコメントアウトされている `GetEvalRunner(name)` をリストア（外部から name で取れるように）。

### 1.5 EpisodeEvalObserver を痩せさせる（`include/anet/observers.hpp`, `src/observers.cpp`）

新 I/F:
```cpp
class EpisodeEvalObserver : public LearnObserver {
public:
    EpisodeEvalObserver(
        std::shared_ptr<EvalRunner> eval_runner,
        int eval_interval,
        bool use_background);
    void OnLearn(const LearnEvent& event) override;
    ~EpisodeEvalObserver() override;
private:
    std::shared_ptr<EvalRunner> eval_runner_;
    int eval_interval_;
    bool use_background_;
    std::unique_ptr<anet::PinnedThreadPool> eval_pool_;   // use_background_ のときのみ
    std::future<void> eval_future_;
};
```
- `OnLearn` の処理:
  - Actor 初回生成・Sync・env Reset・自前 step ループを **全部 EvalRunner 側に委譲**。
  - `EvalRunner::Sync()` を呼んでから `DoStep()` ループを回す（`DoStep()` 戻り値は `StepCounts` のままなので、終端判定は `LastStepHadEpisodeEnd()` で行う）:
    ```cpp
    eval_runner_->Sync();
    do {
        eval_runner_->DoStep();
    } while (!eval_runner_->LastStepHadEpisodeEnd());
    ```
    - batch_size=1: 旧 `RunEvaluationEpisode()` と等価（最初の env 終端で抜ける）。
    - 複数 ENV 並列: 「いずれかの env が終端を踏んだ最初の step で抜ける」既定動作。
      別ポリシーが必要になればこのループだけ書き換える（`EvalRunner` API は不変）。
- `RunEvaluationEpisode()` / `env_` / `actor_` / `report_function_` / `actor_device_` / `runmode_` / `log_interval_` メンバは削除。
- 旧コンストラクタはシグネチャ変更。`RunManager` の呼び出し側だけ追従。

### 1.6 MetricsLogObserverBase をイベント非依存にリファクタ（`src/observers.cpp`）

`MetricsLogObserverBase::GetMetricsData(const UpdateEvent& event, EventField field)`（`observers.cpp:722-759`）が
`UpdateEvent` 直依存なので、EpisodeEndEvent からも再利用できるよう小さく抽象化する。

1. `MetricsLogObserverBase` に protected helper を追加:
   ```cpp
   void OnGenericUpdate(
       const StepCounts& counts,
       std::shared_ptr<const Agent> agent,
       std::shared_ptr<const Runner> runner,
       std::shared_ptr<const BatchEnv> env,            // 任意（EpisodeEnd は発火 Runner の env、Train は runner 経由でも OK）
       const BatchExperience* experience,              // nullable (EpisodeEnd は nullptr)
       const BatchUpdateResultList* update_result_list // nullable
   );
   ```
   - 既存 `OnUpdate(UpdateEvent&)` はこの新 helper へ薄く委譲する。
   - `GetMetricsData` も `event_field` ごとに各ポインタへ振り分け。
   - `UPDATE_RESULT` / `EXPERIENCE` が EpisodeEnd 用に解決されたら `LOG::warn` で「@episode_end では未対応」と通知して空を返す。
2. `MetricsLogEpisodeEndObserver` 新規:
   ```cpp
   class MetricsLogEpisodeEndObserver : public MetricsLogObserverBase, public EpisodeEndObserver {
   public:
       MetricsLogEpisodeEndObserver(/* tag, key, step_axis, event_field, interval, is_ema, ema_alpha, clip */);
       void OnEpisodeEnd(const EpisodeEndEvent& e) override {
           OnGenericUpdate(e.counts, e.agent, e.runner, e.env, nullptr, nullptr);
       }
       std::string GetClassName() const override { return "MetricsLogEpisodeEndObserver"; }
       std::string ToString() const override { return ToStringInternal(); }
   };
   ```

### 1.7 ObserverFactory に `@episode_end` と Runner スコープを追加（`src/observers.cpp`）

#### 文法の 3 軸直交化

```
metrics.scalar.[<output_tag>] = <key>
                                 [$<target_field>]   ← $env / $agent / $runner / $exp / $update_result
                                 @<event>            ← @train / @learn / @episode_end
                                 [$<runner_scope>]   ← $train / $eval.[<eval_name>]
                                 [$<step_axis>]
                                 [$ema] [interval:N] [ema_alpha:A] [clip:V]
```

- `@<event>` は **発火タイミング**を示す軸
- `$<runner_scope>` は **どの Runner 発火か**を示す軸（Train/Eval 横断）
- `$<target_field>` は **どのオブジェクトから取るか**の軸（既存）

#### パーサ変更

- 既存 `@train` / `@learn` 分岐に並べて `@episode_end` を追加。
- 新規 token:
  - `$train` → `runner_scope_opt = RunnerScope::TRAIN`
  - `anet::ExtractBetween(v, "$eval.[", "]")` で eval_name を抽出 → `runner_scope_opt = RunnerScope::EVAL`、`eval_name_opt = <name>`
- 列挙追加:
  ```cpp
  enum class EventType { TRAIN, LEARN, EPISODE_END };
  enum class RunnerScope { TRAIN, EVAL };
  ```
- デフォルト値:
  - `runner_scope_opt` 未指定時 → `RunnerScope::TRAIN`（後方互換）
  - `@episode_end` 指定時の `step_axis_opt` 未指定 → `StepAxis::TRAIN`。

#### `$eval.[..]` × `@<event>` の組合せ制約（重要）

3 軸直交を素直に許すと、`@train $eval.[eval1]`（= EvalRunner からの毎 step TrainEvent）も attach できてしまうが、
**EvalRunner からの `@train` / `@learn` 配信は本タスク範囲外**（step 単位 Eval イベントは別タスクで開放）。
よってパーサ段階で以下を **起動時エラー（`ANET_SYSTEM_ERROR`）** として弾く:

- `@train $eval.[<name>]`  → 不可（step 単位 Eval イベントは別タスク）
- `@learn $eval.[<name>]`  → 不可（LearnEvent は Learner 側で Eval Runner からは飛ばない）
- `$eval.[<name>]` は **`@episode_end` とのみ組合せ可**

`@train` / `@learn` のみ指定時（`$<runner_scope>` 未指定）は既存どおり `RunnerScope::TRAIN` で動く。

#### 出力 observer 生成パート

共通の `Observer` 基底は無いため、`ParsedObserver` は **3 種に分割**:

```cpp
struct ParsedTrainObserver       { RunnerScope scope; std::string eval_name; std::shared_ptr<TrainObserver>       obs; };
struct ParsedLearnObserver       { RunnerScope scope; std::string eval_name; std::shared_ptr<LearnObserver>       obs; };
struct ParsedEpisodeEndObserver  { RunnerScope scope; std::string eval_name; std::shared_ptr<EpisodeEndObserver>  obs; };
```

- `train_observers_` / `learn_observers_` / `episode_end_observers_` の要素型を上記 3 種にそれぞれ変更。
- アクセサ:
  - `ObserverFactory::GetUpdateObservers()` → `std::vector<ParsedTrainObserver>`
  - `ObserverFactory::GetLearnObservers()` → `std::vector<ParsedLearnObserver>`
  - `ObserverFactory::GetEpisodeEndObservers()` → `std::vector<ParsedEpisodeEndObserver>`（新規）
- 不正な eval_name は **起動時エラー**（AGENT.md「設定値の扱い」方針）。`RunManager` 側で `eval_runners_` map と突き合わせる。

#### 例

```
# Train per-step（既存と完全互換）
metrics.scalar.[10_train/01_step_reward] = train_reward $runner @train

# Train エピソード終端
metrics.scalar.[10_train/02_eps_reward] = eps_total_reward $runner @episode_end $train

# Eval[eval1] エピソード終端（旧 $runner eval.[eval1].eps_total_reward の置き換え）
metrics.scalar.[42_eval/01_eps_reward] = eps_total_reward $runner @episode_end $eval.[eval1]

# Eval[eval1] エピソード終端の env 集計
metrics.scalar.[42_eval/02_mean_rank] = mean.ep_max_rank $env @episode_end $eval.[eval1] $ema

# Eval[eval1] エピソード終端での agent eval_policy 統計
metrics.scalar.[42_eval/03_eval_eps] = eval_policy.epsilon $agent @episode_end $eval.[eval1] $ema
```

### 1.8 RunManager から observer を取り付け（Runner スコープ別ルーティング）

`trainer.cpp:684-694` の observer 取り付けループを以下に拡張:

```cpp
auto resolve_runner = [&](RunnerScope scope, const std::string& eval_name)
    -> std::shared_ptr<const Runner>
{
    if (scope == RunnerScope::TRAIN) return train_runner_;
    auto it = eval_runners_.find(eval_name);
    if (it == eval_runners_.end())
        ANET_SYSTEM_ERROR("Unknown eval name '" << eval_name << "' in metrics.scalar config.");
    return it->second;
};

// それぞれ ParsedTrainObserver / ParsedLearnObserver / ParsedEpisodeEndObserver で型が異なるため
// p.obs の静的型も対応する Observer 派生に固定される
for (const ParsedTrainObserver& p : factory.GetUpdateObservers())
    notifier_->Attach(std::make_shared<RunnerScopedTrainObserver>(p.obs, resolve_runner(p.scope, p.eval_name)));
for (const ParsedLearnObserver& p : factory.GetLearnObservers())
    notifier_->Attach(std::make_shared<RunnerScopedLearnObserver>(p.obs, resolve_runner(p.scope, p.eval_name)));
for (const ParsedEpisodeEndObserver& p : factory.GetEpisodeEndObservers())
    notifier_->Attach(std::make_shared<RunnerScopedEpisodeEndObserver>(p.obs, resolve_runner(p.scope, p.eval_name)));
```

- このループは **EpisodeEvalObserver 生成ループの後**に置く（eval_runner が先に必要）。
- `@train $eval.[..]` / `@learn $eval.[..]` の禁止組合せはここに来る前にパーサ側で起動時エラー化されている前提（1.7 参照）。

### 1.9 config 書き換え

#### 書き換え方針

**固定リストでは漏れるため、`rg` で全件検索して潰す**:

```powershell
rg "eval\.\[.*\.eps_total_reward" apps/runner/config
```

ヒットした旧文法 `metrics.scalar.[..] = $runner eval.[<name>].eps_total_reward @learn ...`
をすべて新文法 `metrics.scalar.[..] = eps_total_reward $runner @episode_end $eval.[<name>] ...` に置換。
コメントアウト行（`#metrics.scalar...`）も同等に置換しておく（後で復活したときに新文法であるべき）。

#### 既知の対象（2026-05-21 時点、`rg` 結果ベース）

- `apps/runner/config/metrics_scalar.txt:18-21`（baseline。最重要）
- `apps/runner/config/metrics_scalar.txt:92-95`（full。追加分）
- `apps/runner/config/LunarLander.txt:56-59`
- `apps/runner/config/ImageCls.txt:178-179`
- `apps/runner/config/GridMaze.txt:151-152, 158-161`（コメントアウト中）
- `apps/runner/config/GridMaze_muzero.txt:154-155`（追加分）
- `apps/runner/config/DropMerge.txt:784-785, 790-793`

置換例（metrics_scalar.txt baseline）:
```
metrics.scalar.baseline.[21_eval/01_target_reward]      = eps_total_reward $runner @episode_end $eval.[eval1]
metrics.scalar.baseline.[21_eval/02_policy_reward]      = eps_total_reward $runner @episode_end $eval.[eval2]
metrics.scalar.baseline.[21_eval/03_target_reward_ema]  = eps_total_reward $runner @episode_end $eval.[eval1] $ema ema_alpha:0.001
metrics.scalar.baseline.[21_eval/04_policy_reward_ema]  = eps_total_reward $runner @episode_end $eval.[eval2] $ema ema_alpha:0.001
```

置換後に再度 `rg "eval\.\[.*\.eps_total_reward" apps/runner/config` でヒット 0 件を確認すること。

## 2. 修正対象ファイル一覧

| ファイル | 主要変更 |
| --- | --- |
| `core/anet-core/include/anet/rl.hpp` | EventType::EPISODE_END / EpisodeEndEvent / EpisodeEndObserver / RunnerScopedEpisodeEndObserver / Notifier 拡張 / TARGET_EVAL_REWARD・POLICY_EVAL_REWARD 削除 |
| `core/anet-core/src/rl.cpp` | RunnerScopedEpisodeEndObserver / Notifier::Attach・Detach・Notify(EpisodeEndObserver 系) 実装 |
| `core/anet-core/include/anet/trainer.hpp` | RunnerBase に eps_total_reward_per_env_ / last_episode_total_reward_ / last_step_had_episode_end_ / LastStepHadEpisodeEnd() / AccumulateAndNotifyEpisodeEnd(self, result) helper / GetScalar("eps_total_reward") 追加。EvalRunner に name_ 追加（DoStep の戻り値型 StepCounts は維持）。TrainRunner から eval_last_rewards_ 系削除。GetEvalRunner 復活 |
| `core/anet-core/src/trainer.cpp` | helper 実装、EvalRunner::DoStep と Serial/Pipeline TrainRunner::DoStep の各派生で `shared_from_this()` を渡して helper を呼ぶ。TrainRunner::GetScalar の eval 分岐削除、RunManager の Eval 構築ループ書き換え |
| `core/anet-core/include/anet/observers.hpp` | EpisodeEvalObserver の I/F 痩せ。MetricsLogEpisodeEndObserver 追加。ObserverFactory に GetEpisodeEndObservers 追加。ParsedTrainObserver / ParsedLearnObserver / ParsedEpisodeEndObserver の 3 種を導入 |
| `core/anet-core/src/observers.cpp` | EpisodeEvalObserver 実装簡素化（`LastStepHadEpisodeEnd()` ループ）、MetricsLogObserverBase の OnGenericUpdate 抽出、ObserverFactory に `@episode_end` と `$train` / `$eval.[name]` のパース、`@train $eval.[..]` / `@learn $eval.[..]` の起動時エラー化、MetricsLogEpisodeEndObserver 実装 |
| `apps/runner/src/RunnerApp.cpp` | TARGET_EVAL_REWARD / POLICY_EVAL_REWARD を使った debug log 削除 |
| `apps/runner/config/metrics_scalar.txt` | baseline / full の eval 8 行を新文法に書き換え |
| `apps/runner/config/LunarLander.txt` | 22_eval 4 行を新文法に書き換え |
| `apps/runner/config/ImageCls.txt` | 21_eval 2 行を新文法に書き換え |
| `apps/runner/config/GridMaze.txt` | コメント中の eval 行を新文法に書き換え |
| `apps/runner/config/GridMaze_muzero.txt` | 21_eval 2 行を新文法に書き換え |
| `apps/runner/config/DropMerge.txt` | 同上（コメント中含む） |

## 3. 再利用する既存部品

- `EvalRunner` の env/actor/step ループ基盤（`trainer.hpp:76-92`, `trainer.cpp:133-213`）
- `RunnerScopedTrainObserver` / `RunnerScopedLearnObserver` の実装パターン（`rl.cpp:688-724`）— EpisodeEnd 版のひな型
- `MetricsLogObserverBase::OnUpdate` の EMA/clip/interval ロジック（`observers.cpp:865-909`）— `OnGenericUpdate` に抽出して共有
- `BatchEnv::GetScalar` の `mean.` / `max.` / `min.` 集約（`env.cpp:201-275`）— `$env` 経由で透過利用
- `VectorizedDiscreteBatchEnv` の per-name env 生成（`observers.cpp:487`）— RunManager 側に移設
- `PinnedThreadPool`（`thread.hpp`）— 痩せた EpisodeEvalObserver のバックグラウンド実行に流用

## 4. 検証

実機での学習回しはユーザー側で実施するため、コード側で確実に通せる範囲のみを担当する。

1. **Debug ビルド**
   ```powershell
   cmake --build --preset x64-Debug --target anet-core
   cmake --build --preset x64-Debug
   ```
2. **anet-core-test**
   ```powershell
   core\anet-core\bin\Debug\anet-core-test.exe
   ```
   既存テスト全パスを確認（Observer・Replay 系などに破壊的変更が漏れていないこと）。
3. **設定パーサ静的確認**
   - 新文法 `@episode_end $eval.[eval1]` / `@episode_end $train` を含む既存 config ファイルが起動時にエラーなくパースされること。
   - 不正系（`$eval.[nonexistent]`、`@train $eval.[eval1]`、`@learn $eval.[eval1]` など）が起動時エラーで弾かれること。
4. **旧文法残存ゼロの確認**
   - `rg "eval\.\[.*\.eps_total_reward" apps/runner/config` のヒットが 0 件であること。
   - `rg "TARGET_EVAL_REWARD|POLICY_EVAL_REWARD" core apps` のヒットが 0 件であること（定数撤去確認）。
5. **実機学習での確認はユーザー側で実施**
   - 新指標（`mean.ep_max_rank` 等）の値妥当性、回帰確認はユーザー側で実施。
   - 本実装プランでは「回帰しないこと」「新文法でメトリクスが record される経路が通ること」までを担保する。

## 5. 留意点

- `RunnerThread` で `EvalRunner` を回す現行使い方（`RunnerFrame.cpp:200`、`EvalPanel` 用）の挙動は壊さない。`EvalRunner` コンストラクタの `name` 引数には既存 `name` を流用すれば対応可能。
- `EvalRunner::DoStep()` 内の既存 `TrainEvent` Notify は今回触らない。step 単位 Eval イベントの追加は別タスク。
  - 本タスク範囲では **`$eval.[<name>]` は `@episode_end` とのみ組合せ可**（`@train` / `@learn` × `$eval.[..]` は起動時エラー）。step 単位 Eval イベントを開放する際に同時にこの制約を緩める。
- `MetricsLogObserverBase` の field 自動推定（`observers.cpp:828-859`）は `$update_result` / `$exp` を含むが、EpisodeEnd 用には対応不可。`OnGenericUpdate` 側で nullable を考慮して順序を変える（agent → runner → env のみ試す）。
- **複数 ENV 並列 EvalRunner への前方互換**: イベント発火を `DoStep()` 内 per-env にしたので、
  batch_size>1 へ拡張する際は env / agent / actor 側を VectorizedDiscreteBatchEnv の batch_size に合わせるだけで、
  `EvalRunner` API も `EpisodeEndObserver` 形状も変更不要。トリガ側ループポリシーだけ書き換える。
- **Train Runner の既存 episode 集計と並存**: `train_episode_reward` / `episode_total_reward_comp_` 経路は本タスクでは触らない。
  EpisodeEndEvent 経路と並存し、新文法に移行するタイミングで段階的に既存パスを削れる。
