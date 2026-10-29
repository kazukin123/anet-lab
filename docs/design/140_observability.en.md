<!-- translated-from: 140_observability.jp.md blob:9fc46fddc9534f7fcf4dd01b06acbb1c9b8d6b04 date:2026-10-23 progress:done -->
# Observability

> Primary perspective: function (Events, Observers, metrics, logs, profiling, and artifacts)

## 1. Introduction

### 1.1 Purpose

This document explains how ANET transforms runtime events into scalars, traces, images, videos, GraphViz, text logs, and profiling data. It aims to separate measurement from learning itself while preserving traceability of output meaning and step axes.

### 1.2 Audience

- Developers adding or modifying metrics, Observers, or visualization
- Developers tracing creation of `metrics.jsonl` and Run artifacts
- Reviewers of performance and lifetime boundaries for logging, flushing, and profiling

### 1.3 Scope

This document covers current `Notifier`, Observers, `ObserverFactory`, `MetricsLogger`, runner text logging, and profiling macros. See the [Analysis User Guide](030_user_guide_analysis.en.md) for Metrics Viewer UI operations and [Applications and Tools](160_applications_and_tools.en.md) for application structure.

## 2. Component Definitions

| Component | Definition |
|---|---|
| `TrainEvent` | Carries Experience, ActionInfo, Env, Agent, Runner, and counts after an Env step |
| `LearnEvent` | Carries Experience, UpdateResult, Agent, Runner, and counts after a Learner update |
| `EpisodeEndEvent` | Carries the terminating episode group, Agent, Env, Runner, counts, and step_result. Return values come from Runner scalars |
| `Notifier` | Run-local hub registering four Observer types and synchronously delivering corresponding events. Shared between Train/Eval by `RunManager` |
| Runner-scoped Observer | Wrapper forwarding only Train or a particular Eval Runner's events to the actual Observer |
| `SessionEndEvent` | Reports finalized aggregation of N adopted configured-Eval episodes, once per session |
| `MetricsLogTraceObserver` | Reads individual values within episode-completion callbacks and records one trace row |
| `ObserverFactory` | Constructs Observers from ConfigData such as `metrics.scalar.*`, `metrics.trace.*`, and `metrics.graph.*` |
| `MetricsLog*Observer` | Retrieves scalars from event-specified sources and applies step selection, interval, EMA, and clipping |
| `EpisodeEvalObserver` | Drives configured Eval sessions synchronously or in the background on Learn events, rethrowing worker exceptions to callers |
| Image/Graph Observer | Generates HeatMap, TimeHistogram, Conv2d, and GraphViz output from Probes or NN outputs |
| `MetricsLogger` | Singleton owning Run name/directory and saving scalars, JSON, images, videos, and DOT in shared formats |
| `IBackend` / `JsonlBackend` | Metric-record persistence boundary; the current backend appends to `metrics.jsonl` |
| `anet::log::Logger` | Lightweight logger creating `WxLogStream` with a prefix fixed at construction |
| `FileLogger` | Copies wxLog to UTF-8 `<run_name>.log`, immediately flushing warnings and above |
| `StandardStreamLogger` | Captures GUI-process stdout/stderr in the Run directory |
| `ProfileRange` and macros | Instrumentation boundary recording functions/phases with stable names in CPU/GPU profilers |

`Notifier` only delivers events; it does not collect values. Observers decide what to record and at which step; `MetricsLogger` and backends decide where to save it.

## 3. Code Map

| Area | Main files |
|---|---|
| Events, Observer interfaces, Notifier | [rl.hpp](../../core/anet-core/include/anet/rl.hpp), [rl.cpp](../../core/anet-core/src/rl.cpp) |
| Concrete Observers, Config parser | [observers.hpp](../../core/anet-core/include/anet/observers.hpp), [observers.cpp](../../core/anet-core/src/observers.cpp) |
| MetricsLogger contracts | [metrics_logger.hpp](../../core/anet-core/include/anet/metrics_logger.hpp) |
| JSONL, image, video, DOT output | [metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) |
| Text logging | [log.hpp](../../core/anet-core/include/anet/log.hpp) |
| stdout/stderr capture | [app_util.hpp](../../core/anet-core/include/anet/app_util.hpp), [app_util.cpp](../../core/anet-core/src/app_util.cpp) |
| Profiling | [profile.hpp](../../core/anet-core/include/anet/profile.hpp), [profile.cpp](../../core/anet-core/src/profile.cpp) |
| Scalar metric configuration examples | [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) |
| Image/graph metric configuration examples | [metrics_image.txt](../../apps/runner/config/metrics_image.txt) |
| Runner initialization/flushing | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) |

## 4. Static Structure

```mermaid
classDiagram
direction LR

class Runner
class Notifier {
  +Attach(observer)
  +Detach(observer)
  +Notify(event)
}
class TrainObserver
class LearnObserver
class EpisodeEndObserver
class ObserverFactory
class MetricsLogObserverBase
class MetricsLogTrainObserver
class MetricsLogLearnObserver
class MetricsLogEpisodeEndObserver
class ImageObserver
class GraphVizObserver
class MetricsLogger {
  +LogScalar(tag, step, value)
  +Log(tag, data)
  +Flush()
}
class IBackend
class JsonlBackend
class RunArtifacts

Runner --> Notifier : eventを通知
Notifier o-- TrainObserver
Notifier o-- LearnObserver
Notifier o-- EpisodeEndObserver
ObserverFactory ..> TrainObserver : Configから生成
ObserverFactory ..> LearnObserver : Configから生成
ObserverFactory ..> EpisodeEndObserver : Configから生成
MetricsLogObserverBase <|-- MetricsLogTrainObserver
MetricsLogObserverBase <|-- MetricsLogLearnObserver
MetricsLogObserverBase <|-- MetricsLogEpisodeEndObserver
TrainObserver <|.. MetricsLogTrainObserver
LearnObserver <|.. MetricsLogLearnObserver
EpisodeEndObserver <|.. MetricsLogEpisodeEndObserver
TrainObserver <|.. ImageObserver
TrainObserver <|.. GraphVizObserver
MetricsLogObserverBase --> MetricsLogger
ImageObserver --> MetricsLogger
GraphVizObserver --> MetricsLogger
MetricsLogger *-- IBackend
IBackend <|.. JsonlBackend
JsonlBackend --> RunArtifacts : metrics.jsonl
MetricsLogger --> RunArtifacts : json/video/image/dot
```

Observers do not own Runner's domain state; they obtain snapshots through events or explicit Probe/APIs. Each Observer does own state needed for observation/aggregation, such as EMA in `MetricsLogObserverBase` or episode capture in `GraphVizObserver`. Observers requiring runner scope use wrappers to restrict delivery to their target Runner.

## 5. Processing Flows

### 5.1 Recording Scalar Metrics

```mermaid
sequenceDiagram
    participant R as Train/Eval Runner
    participant N as Notifier
    participant O as MetricsLog Observer
    participant S as Agent/Runner/Env/Experience/Result
    participant M as MetricsLogger
    participant J as JsonlBackend
    participant F as metrics.jsonl

    R->>N: Notify(Train/Learn/EpisodeEnd event)
    loop 登録順の対応Observer
        N->>O: OnTrain / OnLearn / OnEpisodeEnd
        O->>O: runner scopeとstep軸を確認
        O->>S: GetScalarまたはevent fieldを取得
        S-->>O: optional scalar
        opt 値があり記録対象
            O->>O: EMAを更新
            O->>O: interval・finite値を確認し、clipを適用
            O->>M: LogScalar(tag, step, value)
            M->>J: WriteJsonl(record)
            J->>F: 1 JSON recordを追記
        end
    end
```

Observer callbacks execute on the thread calling `Notify()`. Profile expensive rendering, device synchronization, or I/O assuming it enters the Train/Learn critical path. Background evaluation in `EpisodeEvalObserver` is an exception using a dedicated pool; worker exceptions are rethrown to the caller at the next Learn boundary or on an explicit `Shutdown()`. Only the destructor safety net catches exceptions and records FATAL.

Runner aggregates episode returns and episode_steps completed in the latest Step, exposing `mean.episode_return`, `max.episode_return`, `min.episode_return`, `std.episode_return`, and `episode_steps` with the same prefixes. episode_steps counts Env Steps including the terminal one, without multiplying by lane count even for SHARED. The existing train value is `max.episode_return`; configured Eval aggregates the N adopted session episodes. `EvalSessionEnv` subscribes only to the target Eval's `@session_end $env` source keys from resolved metric definitions and snapshots values immediately after episode-completion Steps. Any `nullopt` makes the aggregate `nullopt`; NaNs are excluded. Aggregates with no valid values, and std with one valid value, are NaN. With at least two values, std is population standard deviation.

### 5.2 Finalizing Output at Run Shutdown

```mermaid
sequenceDiagram
    participant UI as RunnerFrame
    participant A as RunnerApp
    participant T as RunnerThread
    participant N as Notifier
    participant G as Agent
    participant M as MetricsLogger
    participant L as wxLog/FileLogger

    UI->>A: StopTraining()
    A->>T: Stop()
    opt app.save_agent_on_close=true
        UI->>A: SaveAgent(agent_close.anet)
        A->>G: Save(archive)
    end
    UI->>A: DrainBackgroundObservers(mode)
    A->>N: Shutdown(shared deadline, mode)
    N-->>A: background Observer停止
    UI->>A: ShutdownRunLogging()
    A->>A: periodic text-log timerを停止
    A->>M: Flush()
    A->>L: FlushActive()
    A->>L: chainをdetachしてclose
```

The periodic timer flushes only `RunName.log`. Metrics, stdout, and stderr are flushed together at explicit `FlushRunOutputs()` boundaries such as pause, save, and shutdown.

## 6. Metric Configuration Contracts

The basic scalar definition is:

```text
metrics.scalar.[tag] = key [$step_axis] [@event] [$target] [$runner_scope] [$ema] [interval:N] [ema_alpha:A] [clip:C]
```

| Element | Main values | Meaning |
|---|---|---|
| `@event` | `@train`, `@learn`, `@episode_end`, `@session_end` (eval only) | Event invoking the Observer; defaults to `@train` |
| `$step_axis` | `$train_step`, `$learn_step`, `$episode_step`, `$exp_step`, `$update_step`, `$sim_step` | Counter used for JSONL `step` |
| `$target` | `$runner`, `$agent`, `$actor`, `$env`, `$env_step`, `$exp`, `$update_result`, `$action_info` | Value source |
| `$runner_scope` | `$train`, `$eval.[name]` | Restricts the originating Runner; also affects whose counter supplies the step |
| `$ema` | - | Computes EMA inside the Observer |
| `ema_alpha:A` | Finite value greater than 0 and at most 1 | Coefficient controlling EMA movement toward new values |
| `interval:N` | Integer at least 1 | Thins events |
| `clip:C` | Finite value at least 0 | Clips values to `[-C, C]` before recording |

Without an explicit step axis, `@train` uses `train_step`; `@learn`, `@episode_end`, and `@session_end` use `exp_step`. Scalar JSON records contain only `type`, `tag`, `step`, and `value`, not the axis name. Do not reuse a tag for a different step axis when changing configuration.

### 6.x Step Coordinate Systems

`StepCounts` belongs to individual Runners; axis names do not identify globally unique coordinates. **A step is identified by the pair of its owning Runner and its axis.** This document calls that pair a [step coordinate system](../../CONTEXT.md).

In Eval scope, the counts depend on the event. `EvalRunner` attaches its own `step_counts_` to `@train` events, but attaches caller-supplied (train runner) `event_counts` to `@episode_end` and `@session_end`. Thus these two definitions use different coordinate systems despite both specifying `$eval.[eval1]` and `$exp_step`.

```text
metrics.scalar.[51_eval1/13_double_suika_created_mean] = $eval.[eval1] @session_end $env $exp_step ...
metrics.scalar.[51_eval1/41_noop_uqe_win_rate]         = $eval.[eval1] @train $exp_step ... $action_info
```

Measured maximum steps were 19,993,856 for the former and 151,185 for the latter, with their ratio drifting monotonically from 0.000039 to 0.0075 during the Run. Constant-factor conversion is invalid. Configuration has no token specifying whose Runner counts are used, so this distinction can only be derived from the combination of `@event` and `$runner_scope`.

To avoid reimplementing that derivation in analysis tools, Runner outputs resolved definitions from constructed Observers as `metrics.scalar.defs` ([ADR 0029](../adr/0029-analysis-metadata-emitted-by-runner.md)). Each tag includes `step_axis`, `runner`, `scope`, `eval_name`, `eval_episodes`, `num_envs`, `event`, `target`, `source_key`, `ema_alpha`, `interval`, and `clip`, written once to the Metrics master as an existing `type: "json"` record. These scalar definitions do not change the Metrics Viewer SQLite schema. `runner` is the eval name only when runner scope is EVAL and event is `train`; otherwise it is `train`.

Each scalar/trace definition stores subscription `scope` (`train` / `eval`), `eval_name`, `eval_episodes`, and `num_envs` separately from coordinate-owner `runner`. `eval_episodes` is the planned adopted count per session; `num_envs` is the constructed eval Env's lane count (`GetBatchSpec().num_envs`). SHARED lanes share one episode, so lane count need not equal parallel episode count; the planned count does not guarantee session completion either. Train scope uses `null` for `eval_name`, `eval_episodes`, and `num_envs`. Eval information is repeated in each metric definition rather than inferred from tag names.

Scalar definitions also store `clip`: `null` if absent, otherwise the symmetric clipping width applied on output. Processing order is value retrieval → EMA update → interval decision → clip → output. Factory supplies resolved subscriptions and clip; RunManager adds actual eval conditions to attached definitions.

`$ema` uses bias-corrected EMA: a zero-initialized internal value and observed-weight sum are updated with the same `ema_alpha`, and the value is normalized by the weight sum. It produces values from the first sample without gaps and continues correcting from accumulated weights even if `ema_alpha` changes.

`interval:N` controls output after value retrieval and EMA state updates. Sparse values such as first Learner priority updates can therefore update EMA from every finite event with `$ema interval:100`, while recording only every 100 steps. Source `NaN` indicates a non-event or unavailable statistic and is not fed into EMA as zero.

`interval:N` fires on bucket crossings: it fires once when the quotient `step / N` exceeds that at the previous firing, rather than waiting for zero remainder. Step increments differ by axis (`train_step`: +1 per round; Observer-visible `learn_step`: `num_envs × replay_ratio / batch`; `exp_step`: `num_envs`). Remainder-based checks stretch the effective interval to `LCM(increment, N)` and can omit firings when the increment does not divide N. Bucket crossing rounds the effective interval to `max(N, increment)` with phase jitter within one event. Shared `IntervalGate` implements this; the first event always fires regardless of `step`. Crossing several buckets in one call still fires only once, without catch-up. See [ADR 0028](../adr/0028-interval-fires-on-bucket-crossing.md).

EMA state updates on every event independently of `interval`. Changing `interval` changes raw recording resolution, not `$ema` values.

Unknown events, step axes, and targets have paths that warn and use defaults; unsupported Eval scope/field combinations fail fast. Scalar Eval scope is limited to `@session_end` or Eval `@train $action_info`. Eval scalar `@episode_end` fails fast with its replacement indicated; train scalar `@session_end` is also rejected.

Keys available through `$agent`, `$actor`, `$action_info`, and `$update_result` depend on the shared interface and concrete Agent's metrics. Unsupported keys are not made to appear equivalent across Agents; Observers handle `std::optional` and `NaN` according to metric definitions. See [DQN Agents](200_dqn_agents.en.md) for Agent-specific keys such as DefaultDQN Train Actor snapshot diagnostics.

`GetScalar()` returning `std::nullopt` means the key is unknown or cannot be handled by a delegate. A known key whose value is unavailable because of current state, timing, configuration, or missing input returns `NaN`. Observers, wrappers, and aggregators distinguish unknown keys from unavailable values, without replacing uninitialized EMA, unfinished episodes, disabled PER, or insufficient batches with zero, previous values, or defaults.

IQN diagnostics do not synchronize devices per metric key. Policy diagnostics pack multiple scalars into a detached Tensor; `DQNActionInfo` materializes them on CPU only at the first key access. Existing Learner IQN diagnostics share priority readback when PER is enabled. With PER disabled, a fixed-length diagnostic pack still uses the existing asynchronous readback path. `metrics.scalar.iqn_search_p0` composes only P0 selections for PER health and throughput without enabling all of `metrics.scalar.full`.

QR/IQN quantile-tail diagnostics use the same synchronization boundary. Five Policy scalars share per-action upper/lower widths, a detached full-quantile alias, and global disagreement/crossing values. On first access only, they gather the final action, select lane-wise nearest-rank p90 of positive crossing depth on device, and cache all five values together on CPU. Action creation and repeated cache access do not sort percentiles; `WithAction()` invalidates only the cache. Learner per-sample upper-tail width joins existing priority readback only with PER enabled, then aggregates into Spearman correlation with clipped raw priority on CPU. With PER disabled, it adds neither a pack nor a wait and returns `NaN`. Tail inputs are detached to float32 and do not feed loss, priority, action, sampling, or RNG.

The current parser converts `interval`, `ema_alpha`, and `clip` with `stoi`/`stof`. After conversion, `EmaFilter` validates finite `0 < ema_alpha <= 1`. Observer construction validates `interval >= 1`, failing fast for zero or negative values. `clip` has no range/finiteness validation, so do not specify a negative value. Invalid numeric strings throw during construction.

### 6.x Trace Channel

```text
metrics.trace.[51_eval1/episode] = $eval.[eval1] @episode_end $env_step game_score game_len game_frames hns57
```

Trace saves each completed adopted episode as one row without aggregation. Without declarations, neither Observers nor rows are created. Event and target are mandatory: only `@episode_end` / `event:episode_end` is supported; targets are `$env_step`, `$env`, `$runner`, `$agent`, or `$actor` (attribute forms are also accepted). Scope defaults to `$train`, and the step axis to `exp_step`.

Bare tokens specify one or more keys. Retrieval and the definition's `keys` array follow declaration order. Duplicate keys, aggregate prefixes, EMA, clip, interval, `key:`, and unknown/invalid controls are rejected at load time. Duplicate event, target, scope, or step-axis controls are rejected regardless of equal/different values or notation; later assignments cannot conceal invalid earlier ones. Existing scalar later-wins, default, and WARN behavior remains unchanged.

```json
{"type":"trace","tag":"51_eval1/episode","step":456,"lane":3,"data":{"game_score":422,"game_len":1242,"game_frames":4968,"hns57":31.2}}
```

Series are identified by `(type, tag)`, allowing scalar and trace to share a tag. `lane` is the event's `env_index` (lane for PER_LANE, -1 for SHARED). `step` uses the same integer coordinates as scalars; multiple episodes may share a step and lane. Records have `timestamp` but no top-level `value`, and `data` key order is unspecified. Unknown-key `nullopt` fails fast with tag/key/lane/target; NaN and ±Inf retain the key with a `null` value.

`EvalSessionEnv::LastAdoptedGroups()` returns only adopted groups completed in the immediately preceding Step. EvalRunner notifies immediately after that Step and before the next. SHARED group 0 becomes -1 at notification. Finalized values are read for the specified lane from event.step_result, so individual-value sequences are not accumulated in the decorator. Return aggregation and SessionEnd happen only once at session end. Train uses the same trace Observer.

Only attached definitions are recorded, separately as `metrics.scalar.defs` and `metrics.trace.defs`, excluding dormant eval. Empty definitions are not output. Trace definitions use `{tag: {step_axis, runner, scope, eval_name, eval_episodes, num_envs, event, target, keys}}`; both definition records are mirrored unchanged to `json/<definition tag>.json`. Only scalar definitions are passed to Agent as subscription hints.

The `inspect_run` master/cache paths prefer new `metrics.scalar.defs`, reading old `metrics.defs` only if absent. Both report `def_source=metrics_defs` without a WARN solely for renaming. Old-name compatibility is an exception lasting until the active Run working set uses only the new name; historical artifacts remain unchanged. Configuration derivation when definitions are absent is retained, including `session_end` derivation for selector expansion without caches and for `tags --no-observed`.

Metrics Viewer stores traces in existing `json_lines`, separately from scalars. `inspect_run.py trace-csv` reads and exports rows directly to CSV ([030 Section 6.9](030_user_guide_analysis.en.md#69-extracting-individual-trace-rows-as-csv)). Quantile/threshold aggregation, Metrics Viewer visualization, additional events, `episode_id`, and `model_version` are outside this feature's scope. See [ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md) for rationale.

### 6.1 Evaluation Trace Columns and Value Reading

`MetricValueContext`, `MetricValueReader`, `StepTraceColumnSpec`, `StepTrace`, and `StepTraceBuilder` belong to the shared metrics functional group (include/anet/metrics.hpp and src/metrics.cpp). `MetricValueReader` retrieves values using a target, key, lane, and borrowed notification. Shared DSL token classification is declared internally in src/metrics_impl.hpp and implemented in metrics.cpp. Metrics does not include observers or gui. Scalar lookup/aggregation/EMA and trace unknown-key diagnostics remain in their respective Observers. `RunManager::MetricsWiring` connects subscriptions, while `StepTraceBuilder` generates a `StepTrace` per lane from a mirror EvalRunner's `TrainEvent`. Columns are chosen independently of metrics definitions through `app.eval_trace_panel.columns.[number]`. Numbers are at least 1, may have gaps, and appear in ascending order. `StepTrace.step` uses the same step_t coordinate as counts.exp_step.

| target | Timing and keys |
|---|---|
| `$action_info` | Decision-time values. `action` is the executed action, including forced actions; continuous actions use `action.[dimension]` |
| `$actor` | Actor scalars at that decision time |
| `$env_step` | Transition results: `reward`, `done`, `truncated`, and result-type scalars, before Reset even at termination |
| `$runner` | Counts including this step, rewards, and completion aggregates. `step_in_episode` is 1-based; a terminal row shows the completed length |
| `$agent_step` | Scalars from the Module owned by Agent diagnostics. DQN provides `q_sa`, `state_value`, and `approx_priority`. Agents without diagnostics make the entire target NaN |

Columns cannot use `$agent`, `$env`, `$exp`, or `$update_result`. Aggregation prefixes are supported only for `$runner` episode_return/episode_steps, returning NaN on non-completion rows. Unknown keys fail fast on first read; known unavailable and nonfinite values appear blank. This does not add `$agent_step` to the metrics DSL.

`AgentStepDiagnosticsResult.current` owns batch values through `unique_ptr<const Module>` and exposes them via `GetScalar(key, lane)`. The shared layer does not restrict keys. If a provider exists, an unknown key (nullopt) fails fast on first read with the column number and key. Without a provider, arbitrary `$agent_step` columns yield NaN, allowing default columns to work. Known NaN and pending status are independent; only lane-specific `pending_keys` corresponding to selected columns are awaited.

Agent creates an `AgentStepDiagnosticsProvider` through `CreateStepDiagnosticsProvider`. The Provider owns a sequence window per lane, and even pending keys return known NaN in current. Every pending key eventually resolves, including with NaN if a value could not be established. Delayed resolution uses `AgentStepDiagnosticsCompletion` with `lane`, `steps_back`, `resolved_keys`, and `values` owning the target row; read the value through `GetScalar(key, 0)`. `StepTrace.value_ready` marks resolution per column, becoming true even for a resolved NaN. All selected columns resolving makes the row `complete`. An update handler receives newly resolved zero-based column indices through a synchronous `span<const size_t>` valid only during notification. Multiple columns for the same key each resolve once.

`BatchActionInfo` derives from ModuleBase and exposes lane-specific `action` and `action.[dimension]` through shared GetScalar. The entire dimension string and range are validated. Scalar queries with lane=-1 do not recognize these two keys; derived types provide Agent-specific scalars. DQNActionInfo delegates keys it does not own to the base. Unknown keys, including for non-DQN Agents, return nullopt, causing scalar Observers' ordinary value not found WARN. Requests whose notification lacks ActionInfo remain errors.

The concrete DQN provider and recognition of its three keys stay inside DQN. DQN step diagnostics use Q-hint generation shared with Actor, n-step assembly shared with replay, and the existing initial-priority estimator. Computation is independent of PER and adds no forward or RNG consumption. The reward scaler applies its current transform without updating statistics or clipping rates. RunningStdRewardScaler publishes the std computed by Scale and the initial std at Reset through an atomic; diagnostics read only this value, without accessing the Agent mutex or mutable Tensor statistics. Learning-side operation order is unchanged, and no restoration mechanism is added for reward statistics absent from current checkpoints. Windows containing truncation leave priority unavailable. Nonterminal values fill their original row once bootstrap Q is available; terminal steps resolve shorter windows.

The builder reports row additions separately from delayed column resolution. Images are not retained for rows beyond the GUI's retention limit. Verify public paths for column parsing, timing, diagnostics, playback, and numerical equivalence with the [PRD094 validation resources](../../core/anet-core/testdata/prd094/README.md).

## 7. Output and Lifetimes

| Output | Producer | Update/flush boundary |
|---|---|---|
| `metrics.jsonl` | `JsonlBackend` | Appended per scalar/metadata record; finalized by explicit flush |
| `config/*.txt` | `MetricsLogger` | Written per tag when Config is logged; Env uses `env.<Env name>.txt` |
| `json/*.json` | `MetricsLogger` | Overwritten or created per step when JSON metadata is logged |
| `json/eval.[<tag>].session_cancelled.json` | `MetricsLogger` | Records the coordinates, elapsed time, and number of completed episodes when an evaluation session is cooperatively cancelled. Also appended to the Metrics master with `type=json` |
| `videos/<tag>.mkv` | `VideoLogger` | Logger created at the first frame and closed at Run end. The receiver of ffmpeg termination notification deletes wxProcess; if its owner releases it first, it deletes itself upon notification |
| `images/<tag>/*.png` | `MetricsLogger` | Created per frame when `use_png_dump=true` |
| `dot/**/*.dot` | `MetricsLogger` | Created per GraphViz event |
| `<run_name>.log` | `FileLogger` | Periodic timer, warnings and above, explicit flush |
| `stdout.log` / `stderr.log` | `StandardStreamLogger` | Captured process standard streams; explicit flush/stop |

`MetricsLogger` is a process singleton owning the Run directory under the assumption of one active Run per process. `Reset()` releases it at Run end. Preserve application shutdown order so video loggers and wxLog chains are not destroyed while files are in use.

Concrete Env text logs begin with `env.[<Env name>]`, or for a single Env with the lane name that further appends `.[<lane index>]` (for example, `env.[train].[37]: `, `env.[eval_target].[0]: `, `env.[EvalPanel].[0]: `), allowing people to distinguish Train, configured Eval, EvalPanel, and batch lanes. This is the same configuration notation as `eval.[<tag>]: ` on evaluation-session lines, and for configured Eval the string inside `[ ]` matches the tag. `SingleDiscreteEnvBase` and `BatchEnvBase` hold protected `anet::log::Logger log`; concrete Envs use `log.info()`, `log.verbose()`, `log.warn()`, and `log.error()`. Do not use `LOG::` directly inside Envs or scatter prefix formatting/`GetName()` concatenation across log lines. Factories, free functions, Runners, Agents, and Views outside Env continue using `LOG::`.

Debug logs use `ANET_LOG_DEBUG_PREFIXED(expr)`, delegating to `ANET_LOG_DEBUG(log.prefix() << expr)` and preserving debugger/level guards, source information, and non-evaluation with `ANET_ENABLE_DEBUG_LOG=0`. Env names are opaque display strings and neither replace nor change MetricsLogger tags, JSONL fields, artifact paths, or runner scopes. Views may display them through common Env accessors but must not use them to branch Env behavior or metric identity.

### 7.1 Sparse Scalars and Subscription Information

Sparse known keys without current values return `NaN`; unknown keys return `nullopt`. Observers exclude nonfinite values before EMA updates and before averaging multiple UpdateResults. Later finite values resume normally from the last finite EMA state.

Source key, event, target, interval, runner scope, and eval name are passed to Agent as typed subscriptions from actually attached scalar definitions. Where a metrics row's `interval` is authoritative for expensive measurement cadence, commenting out the definition must stop the computation itself.

### 7.2 Evaluation Session Logs

Configured Eval writes one info-level `eval.[<tag>]: session start` and one `session end` line.
The start line is written from the train thread when the evaluation fires, that is, before any wait for the previous background session to complete; the end line is written from the thread that ran the session.
Both lines' learn_step/exp_step are session-start coordinates supplied by Train and correspond to session_end scalars.
The end line's elapsed time runs from before Sync through after SessionEnd notification, in seconds with two decimals; it is not Train's actual background wait time. Nor does it match the time elapsed since the start line.
In background mode, only when blocked waiting for the previous session, one info-level `waiting for previous session` line is written when the wait begins and one `waited for previous session` line after collection; both include the current trigger's learn_step/exp_step.
The completion line's elapsed is actual Train-thread wait time in seconds with two decimals. Neither line appears for foreground execution or shutdown waits; for an exception during the wait, only the wait-start line remains.
When a wait occurs, the lines appear in the order `session start`, `waiting for previous session`, the previous session's `session end`, and `waited for previous session`.
The end line includes mean.episode_return, max.episode_return, mean.episode_steps, and max.episode_steps matching finalized scalars.
When the exit drain waits, `draining in-flight session on exit` is written; when it switches to cooperative cancellation, `cancelling in-flight session on exit reason=close|config|timeout` is written. When a cancellation completes, the normal `session end` is not written; `session cancelled learn_step=... exp_step=... elapsed=... completed=...` is written instead. The corresponding session scalars are not emitted, and traces of completed episodes are retained.
Abnormal termination does not emit a normal end line. Startup scheduled lines include interval, background, wait_on_exit, episodes, and batch_size.
Generic evaluation episode lengths are available in baseline/full through `21_eval/05_target_ep_steps`–`08_policy_ep_steps_max`.

## 8. Profiling and Performance Considerations

- Use `ANET_PROFILE_FUNC()` for whole functions and `ANET_PROFILE_SCOPE(phase)` for ordinary phases.
- Switch consecutive phases with `ANET_PROFILE_SCOPE_NEXT(...)` to compare within the same visible lifetime.
- Use `ANET_PROFILE_SCOPE_FULL` only when automatic names do not represent the logical operation, such as callbacks or asynchronous workers.
- Expensive Probes, image rendering, GraphViz, and CPU transfers with `interval=1` directly affect training throughput.
- EMA, clipping, and thinning control display/storage volume but do not have the same meaning as original data. Retain configuration for analysis.
- Separate periodic text-log flushing from GUI FPS. Do not increase I/O frequency by flushing metrics/stdout on the same timer.

## 9. Tests and Extension Checks

### TempoRL Scalars and Session Evaluation

Read a variable Actor's limit through `$actor skip_max_length @learn $exp_step` and opt into `50_agent_skip/07_max_length` in the experiment's leaves. The default metric chain does not include it. Full/single/disabled skip return NaN for the recognized key. Single has decision rate and mean length 1, while trace skip_q.[j] still accepts 1..J. Existing 13_sample_k_mean, 14_kmax_mean and 11_loss describe auxiliary windows. See [PRD104 validation](../../core/anet-core/testdata/prd104/README.md).

Skip metrics use the `50_agent_skip` group. Training length tags use two-digit indices such as `03_skip_length_rate.[01]`, while source keys remain `skip_length_rate.[1]`. Actor step statistics aggregate decision lanes; unavailable values are NaN. Learner results retain seven skip values, including `17_skip_primary_grad_norm` and `18_skip_primary_clip_ratio` for the split primary-skip gradient before clipping; these two are NaN outside split.

`skip_session_decision_rate`, `skip_session_length_mean` and `skip_session_length_rate.[j]` accumulate between Actor Sync calls and are read through `$eval.[greedy] $actor @session_end`. Their tags are `21_greedy_decision_rate`, `22_greedy_length_mean` and `23_greedy_length_rate.[01]` onward, recorded with `runner=train` and the training coordinate. This is an approximate decision-count measure for sessions with multiple lanes and early lane completion. GridMaze, LunarLander and DropMerge add `21_eval/09_greedy_reward` and EMA/episode-step tags 10–12. Atari uses session name `greedy`, retaining the `53_evalg` group and its evaluation conditions. Counters stay on device and transfer once on first scalar access. See [PRD103 validation assets](../../core/anet-core/testdata/prd103/README.md) for numerical, Run and metadata checks.

- [PRD094 validation resources](../../core/anet-core/testdata/prd094/README.md): columns, timing, diagnostics, playback, snapshots, and numerical equivalence across five configurations.

When changing Observers or output formats, check the following:

1. Event, runner scope, step axis, and target combinations are explicit.
2. Callbacks add no unnecessary clones, device synchronization, or I/O.
3. A tag's type and step axis do not change during a Run.
4. Background exceptions and shutdown waits are preserved.
5. Existing `metrics.jsonl` readers and Metrics Viewer can safely ignore or interpret new records.
6. No file handles, timers, or ffmpeg processes remain after Run close.

Main regression tests belong in [observers_test.cpp](../../core/anet-core/src/observers_test.cpp), [metrics_logger_test.cpp](../../core/anet-core/src/metrics_logger_test.cpp), [log_test.cpp](../../core/anet-core/src/log_test.cpp), and [episode_end_test.cpp](../../core/anet-core/src/episode_end_test.cpp).

## 10. Related Documents

- [Run Execution User Guide](020_user_guide_run.en.md)
- [Run Analysis User Guide](030_user_guide_analysis.en.md)
- [Runtime Infrastructure and Configuration](100_runtime_and_configuration.en.md)
- [Agents and Learning](110_agents_and_learning.en.md)
- [Environments](120_environments.en.md)
- [ReplayBuffer](150_replay_buffer.en.md)
- [Applications and Tools](160_applications_and_tools.en.md)
- [DQN Agents](200_dqn_agents.en.md)
- [DropMerge Optuna Guide](optuna.md)

### Transition-Value Target `env_step`

`$env_step` reads `TrainEvent.step_result` or `EpisodeEndEvent.step_result`. Scalar train reads all lanes; episode_end reads only done or truncated lanes in ascending order, excluding NaN through the existing mean/std/min/max aggregation. Even when the same Step emits several EpisodeEnd notifications, scalar reading and EMA update occur only once. SHARED reads the lane=-1 batch value once. Trace reads only each notification's env_index and records one row per completed group.

`$env_step` is limited to train/episode_end. Declarations for learn/session_end, which have no result, are rejected at construction. Unknown or nonscalar keys fail fast on first read with tag/key/lane; known unavailable keys yield NaN. Built-in reward/done/truncated and derived or parameterized keys supplied by concrete results use the same target. `$env` reads current Env state, or at session_end the aggregates accumulated by EvalSessionEnv from adopted results.

When moving completion-statistic definitions from train to episode_end, preserve tags, EMA, and explicitly selected step axes. Definition records use target `env_step`. inspect_run also recognizes configuration-derived definitions, and Java Metrics Viewer stores definition JSON without restricting target values. Atari lives is configured as provisional values from all lanes at train, while game statistics are final values at episode_end.
