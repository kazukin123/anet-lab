<!-- translated-from: 120_environments.jp.md blob:a869ad75ffbbbda25c2d667e4328e27449ffa40c date:2026-10-23 progress:done -->
# Environments

> Primary perspective: function (Env, with internal processing stages in chronological order)

## 1. Introduction

### 1.1 Purpose

This document explains ANET's shared Env contracts, the mechanism for executing single Envs in batches, and the physical layout of concrete Envs.
It clarifies how Runner and Agent can use the same State, Action, Reset, and Step flow without depending on environment-specific implementations.

### 1.2 Audience

- Developers adding or modifying Envs
- Developers checking Observation, Action, Reward, and episode-termination contracts
- Reviewers checking parallel Env execution and seed/device configuration

### 1.3 Scope

This document covers current `SingleDiscreteEnv`, `BatchEnv`, batch wrappers, native batch Envs, factories/repositories, and registered Envs.

## 2. Basic Concepts and External Contracts

### 2.1 EnvSpec

`EnvSpec` is the shared specification connecting Env and Agent.

- `StateSpec`: keys, shapes, dtypes, and ranges of the TensorDict forming an Observation
- `ActionSpec`: discrete Action labels or continuous Action dimensions
- `reward_range`: expected range of Rewards returned by Env
- `info`: additional Env-specific metadata

Standard Observation keys are `vector`, `grid`, and `action_mask`. `action_mask` is legal-action metadata and is treated separately from ordinary Network inputs.

`BatchEnvSpec` contains `num_envs`, `num_threads`, and `episode_scope`. With `episode_scope=PER_LANE`, each lane is one episode group; with `SHARED`, the entire batch is one episode group. JSON uses `per_lane` / `shared`.

### 2.2 Reset, Step, and Result Ownership

- Env fixes its `RunMode` at construction and exposes it through `GetRunMode()`.
- `Reset()` returns the initial State. `Step(Action)` advances the transition and automatically Resets only terminal lanes within the same call. The ThreadPool implementation performs Step and Reset on the same worker.
- `BatchStepResult::next_state` holds the transition before Reset; `continue_state` is the state for the next action. Continuing lanes share the same observation, while terminal lanes use the post-Reset observation.
- A full Reset sets `episode_start` to true for every group. Step keeps done/truncated consistent within each group and satisfies `n_episode_end == number of completed groups` and `continue_state.episode_start == (done || truncated)`. `done && truncated` is allowed.
- Step results are newly created on every call. A single-Env result owns that Step's aux and statistics, which remain unchanged after automatic Reset or later Steps. Aux arrays retain one entry per lane, including empty lanes. Wrapper full Reset still reuses some result buffers.
- The default `SingleEnvResult::GetScalar(key)` implementation reads scalar values from its owned aux. `BatchStepResult::GetScalar(key, lane)` handles reward/done/truncated; wrappers delegate to the lane's single result. Native SHARED Envs answer batch values at lane=-1. ImageCls reward is the sum over valid lanes, and done/truncated represents the group's shared termination.
- Known but unavailable values return NaN; unknown or nonscalar keys return nullopt. Results also answer derived and parameterized keys themselves without copying all aux for every scalar.
- Queries to the Env itself return its current state, not transition-result values. Train/EpisodeEnd consumers read the Step result owned by the notification.
- Train/Eval-specific randomness, augmentation, and termination are separated by the `RunMode` fixed at construction.

### 2.3 Single and Batch

`SingleDiscreteEnv` represents one environment with one discrete Action.
`BatchEnv` groups multiple environments for Runner.

Ordinary concrete Envs inherit `SingleDiscreteEnvBase`; `BatchEnvBuilder` batches them using one of these:

- `VectorizedDiscreteBatchEnv`: executes multiple Envs sequentially on the calling thread.
- `ThreadPoolDiscreteEnv`: executes multiple Envs in parallel through a thread pool.

ImageCls is a native batch Env directly inheriting `BatchEnvBase`. `ImageDataSource` constructs fixed-B Tensors from a Dataset without creating N single Envs or using wrapper collation. Train uses `PER_LANE`; Eval uses `SHARED`. At an eval-window boundary, it sets every lane's `done` and returns `n_episode_end=1`, setting every lane's `episode_start` in the continue_state returned by the same Step. Data prefetching also occurs inside Step, providing prefetched observations for the next action.

ImageCls configuration requires standard Train/Eval Sources as a pair. Train uses `ImageClsEnv.train.dataset_key` and `ImageClsEnv.train.augment.*`; Eval uses `ImageClsEnv.eval.dataset_key` and `ImageClsEnv.eval.eval_window.mode` / `eval_window.rotating_size`. Untagged Eval uses standard Eval configuration; configured Eval overlays only required fields through `run.eval.[tag].env.eval.*`. Factory validates both manifests during Env construction, but image decoding and cache preparation are deferred until the selected Source is used.

### 2.4 Env Names

`SingleDiscreteEnv` and `BatchEnv` are interfaces without constructors or state, exposing pure virtual name accessors. `SingleDiscreteEnvBase` and `BatchEnvBase` hold immutable, human-facing `name` values and implement the accessors as `final override`. `BatchEnvBase::GetName()` returns the batch name; `GetEnvName(lane_index)` returns `env.[<name>].[0..N-1]` strings generated once at construction. Concrete Envs inherit the corresponding Base rather than implementing name accessors themselves.

A name is an opaque display string, not a substitute for Env class ID, RunMode, config prefix, seed, RNG, DatasetKey, or metrics tag. Env does not parse names or change Reset, Step, Reward, or termination behavior based on them. Empty names, nonpositive lane counts, and out-of-range lane indices always fail fast through `ANET_CHECK_MSG`.

Both Bases also hold protected `anet::log::Logger log`, constructing the prefix once when the name is established. A single Env uses the lane name as is, giving `<name>: ` (for example, `env.[train].[37]: `), and a BatchEnv uses `env.[<name>]: `. Active text logging in concrete Envs uses `log.info()`, `log.verbose()`, `log.warn()`, and `log.error()` instead of concatenating `GetName()` on every line. Debug logging uses `ANET_LOG_DEBUG_PREFIXED`, preserving the guards and build-time elimination behavior of ordinary `ANET_LOG_DEBUG`.

## 3. Component Definitions

| Component | Definition |
|---|---|
| `SingleDiscreteEnv` | Shared interface for one discrete-Action Env |
| `SingleDiscreteEnvBase` | Base holding a single Env's name and implementing shared name accessors |
| `SingleDiscreteEnvFactory` | Creates concrete single Envs from config, device, a completed lane name, and seed |
| `BatchEnv` | Exposes batched Reset/Step, specs, device, and shutdown |
| `BatchEnvBase` | Base holding BatchEnv and all lane names and implementing shared name accessors |
| `DiscreteBatchEnvBase` | Base providing spec validation, aggregation, and common metrics for multiple single Envs |
| `VectorizedDiscreteBatchEnv` | Batch implementation driving single Envs on one thread |
| `ThreadPoolDiscreteEnv` | Batch implementation driving single Envs through a thread pool |
| `BatchEnvFactory` | Per-Env-class factory interface creating native batch Envs |
| `EnvRepository` | Process registry holding either a single factory or a batch factory for each class ID |
| `BatchEnvBuilder` | Resolves factories through the repository and either creates native batches directly or batches single Envs with wrappers |
| `ImageDatasetManager` | ImageCls-specific singleton sharing DatasetKey, manifest, and pre-augmentation cache within the process |
| Env View | Optional View implementation displaying Env-specific State in Runner GUI |

## 4. Code Map

### 4.1 Shared Infrastructure

| Area | Main files |
|---|---|
| State/Action/Env interfaces | [rl.hpp](../../core/anet-core/include/anet/rl.hpp), [rl.cpp](../../core/anet-core/src/rl.cpp) |
| Batch wrappers/repository | [env.hpp](../../core/anet-core/include/anet/env.hpp), [env.cpp](../../core/anet-core/src/env.cpp) |
| View interfaces/repository and AUI layout base Frame (`AuiLayoutFrame`) | [gui.hpp](../../core/anet-core/include/anet/gui.hpp), [gui.cpp](../../core/anet-core/src/gui.cpp) |

### 4.2 Concrete Envs

| Env | Implementation | Main Observation characteristics |
|---|---|---|
| CartPole | [core/envs/cartpole2](../../core/envs/cartpole2) | Vector |
| LunarLander | [core/envs/lunarlander1](../../core/envs/lunarlander1) | Vector, Box2D physical state |
| DropMerge | [core/envs/dropmerge1](../../core/envs/dropmerge1) | Grid and vector, Box2D physical state |
| GridMaze | [core/envs/gridmaze1](../../core/envs/gridmaze1) | Vector-based maze state with guaranteed reachability; unsolvable seeds are deterministically resampled |
| ImageCls | [core/envs/imagecls1](../../core/envs/imagecls1) | Image grid and classification target |
| Atari | [core/envs/atari1](../../core/envs/atari1) | Single-frame uint8 grid from ALE |

Each Env directory groups the Env implementation, factory, and optional View and tests by function.

## 5. Static Structure

```mermaid
classDiagram
direction LR

class EnvRepository {
  +Regist(factory)
  +GetSingleDiscreteEnvFactory(class_id)
  +GetBatchEnvFactory(class_id)
}
class SingleDiscreteEnvFactory
class BatchEnvFactory
class SingleDiscreteEnv {
  +GetName()
  +GetRunMode()
  +GetSpec()
  +Reset()
  +Step(action)
}
class SingleDiscreteEnvBase
class BatchEnvBuilder
class BatchEnv {
  +GetName()
  +GetEnvName(lane_index)
  +GetRunMode()
  +GetSpec()
  +GetBatchSpec()
  +Reset()
  +Step(action_info)
  +Shutdown()
}
class BatchEnvBase
class DiscreteBatchEnvBase
class VectorizedDiscreteBatchEnv
class ThreadPoolDiscreteEnv
class ImageClsEnv

EnvRepository o-- SingleDiscreteEnvFactory
EnvRepository o-- BatchEnvFactory
BatchEnvBuilder --> EnvRepository : class_idを解決
BatchEnvFactory ..> ImageClsEnv : native batchを生成
SingleDiscreteEnvFactory ..> SingleDiscreteEnv : N個生成
SingleDiscreteEnv <|-- SingleDiscreteEnvBase
BatchEnv <|-- BatchEnvBase
BatchEnvBase <|-- DiscreteBatchEnvBase
DiscreteBatchEnvBase <|-- VectorizedDiscreteBatchEnv
DiscreteBatchEnvBase <|-- ThreadPoolDiscreteEnv
BatchEnvBase <|-- ImageClsEnv
VectorizedDiscreteBatchEnv *-- SingleDiscreteEnv
ThreadPoolDiscreteEnv *-- SingleDiscreteEnv
```

Shared infrastructure handles batch execution of single Envs. ImageCls, where batch generation itself is domain processing, is implemented as a native `BatchEnv`.

## 6. Processing Flows

### 6.1 Reset and Step

```mermaid
sequenceDiagram
    participant R as Train/Eval Runner
    participant B as BatchEnv
    participant E as Single Env群
    participant A as Actor

    R->>B: Reset()
    B->>E: 各EnvをReset
    E-->>B: SingleResetResult
    B-->>R: 集約したBatchResetResult

    R->>A: MakeAction(BatchState)
    A-->>R: BatchActionInfo
    R->>B: Step(action_info)
    B->>E: laneごとのActionでStep
    Note over B,E: ThreadPool実装では並列実行
    E-->>B: SingleStepResult（遷移値を所有）
    opt 終端lane
        B->>E: Reset()
        E-->>B: SingleResetResult
    end
    B-->>R: Reward・next_state・continue_state
    Note over R,E: 通知はStep結果を読み、次行動はcontinue_stateを使う
```

Batch-wrapper and Runner contracts determine when completed episode groups are reset and how `episode_start` is handled. `EpisodeStatsAccumulator` finalizes episode return and episode_steps together as `CompletedEpisodeResult`. Return is per lane for `PER_LANE`, or the sum of rewards across all lanes and steps for `SHARED`. episode_steps counts `Step()` calls including the terminal one; it is not multiplied by lane count even for `SHARED`. Configured Eval's `EvalSessionEnv` uses dynamic grants to adopt exactly N episodes, accumulates values through GetScalar on adopted Step results, and reuses the saved continue_state with empty aux as the next session's Reset result only when all groups are fresh. Otherwise it performs a full Reset. Concrete Envs focus on state transitions and Reward calculation within one episode.

### 6.2 Construction

1. `EnvRepository` resolves a concrete single or batch factory from `env.class_id`.
2. The caller supplies BatchEnv name, RunMode, and config prefix; `BatchEnvBuilder` establishes num_envs, device, seed, and worker settings.
3. A batch factory directly creates a native `BatchEnv`. For a single factory, the wrapper forms `env.[<name>].[lane_index]` and creates one single Env per lane.
4. The single path stores Envs in a vectorized or thread-pool wrapper according to worker mode. Native ImageCls applies the same worker settings to Source sample processing, performing decode/cache lookup through augmentation in one work item.
5. Runner construction passes EnvSpec and BatchEnvSpec to Agent.

## 7. Configuration, Lifetimes, Errors, and Performance

### 7.1 Shared Settings

| Key | Meaning |
|---|---|
| `env.class_id` | Concrete Env factory class ID |
| `env.worker_type` | Selects `AUTO`, single-thread, or thread-pool |
| `env.worker_threads` | Thread-pool worker count. Negative values select predefined automatic resolution modes |
| `env.device` | Env device: `auto`, `cpu`, `cuda`, or `cuda:N`; defaults to `cpu`, with the adopted value in `json/env.json` |
| `run.train.num_envs` | Main Train Env batch size |

Each factory reads Env-specific settings from the same ConfigData. Unknown class IDs, invalid worker settings, and inconsistent specs fail rather than being silently corrected.

### 7.2 Lifetimes and Shutdown

- Wrapper BatchEnvs own their single Envs and pools. Native ImageCls owns an Env-local Source and sample worker pool.
- Runner shutdown stops workers through `BatchEnv::Shutdown()`.
- Mutable Env state and RNGs are isolated per Env instance. Only ImageCls immutable Dataset/manifest/cache data is shared process-wide by DatasetKey.
- Treat EnvSpec as the post-construction connection contract; do not change shapes or Action counts during a Run.

### 7.3 Performance

- With small `num_envs` or lightweight Env Steps, thread-pool synchronization costs may outweigh benefits.
- Measure parallelism benefits when each lane performs expensive work such as Box2D or image decoding.
- Env Reset/Step are major profiling boundaries; compare worker counts together with `exp_step_per_sec`.
- Different Env and Agent devices incur transfer costs. Check supported devices in both configuration and each Env's implementation.

### 7.4 GridMaze Maze Generation

A present `GridMazeEnv.seed` fixes the maze using that value, regardless of mode. Without a seed, `GridMazeEnv.seed_mode = process | episode` selects automatic seeding (`process` by default). Zero is an ordinary fixed seed, not absence, and shared defaults do not define a seed. `episode` samples on each Reset. `process` samples once when first needed and shares only the input seed across training, evaluation, GUI instances and Runs in the same process, including after instance destruction and recreation. Each Env owns its map and RNG. The shared input is logged once as `GridMaze process seed initialized: seed=...`. Automatic seeding is independent of `run.seed`. Initialization failures are cached and rethrown on subsequent requests. Unknown modes are rejected even when a seed is present.

Every Reset adopts a maze whose goal is reachable from the start without crossing walls or holes. Solvable seeds retain the existing cell generation and RNG consumption order. Unsolvable candidates are regenerated using `splitmix64(seed + k)` for k=1..1000. If the initial candidate and all 1000 retries fail, the error identifies the seed, dimensions, densities, coordinates and retry limit.

Dimensions must be positive; each coordinate must be `-1` or inside its axis. Identical fixed start/goal coordinates and an empty random-start candidate set are errors. A randomly chosen goal coinciding with a fixed start triggers a retry. The aux `seed` is the actual accepted seed and reproduces the maze when configured again. The displayed seed uses unsigned decimal notation throughout the configurable range. Fixed-seed retries emit an INFO notice with the input seed, retry count and accepted seed. A synchronized process-wide record, explicitly allowed only within GridMaze, suppresses repeated notices for the same configured seed across instances. Automatic seeds emit no such notice.

## 8. Tests and Extension Checks

When adding or modifying an Env, check the following:

1. `EnvSpec` shapes, dtypes, Actions, and Reward ranges match actual data.
2. The reproducibility contract for identical seeds and settings is explicit.
3. Validate next_state/continue_state and result immutability immediately after Reset, on ordinary Steps, and for terminated/truncated states.
4. Batch wrappers aggregate correctly for both B=1 and multiple lanes.
5. Semantics remain identical between `VectorizedDiscreteBatchEnv` and `ThreadPoolDiscreteEnv`.
6. An added View's Env class ID matches its `ViewRepository` registration.
7. Frequently called Reset/Step operations have meaningful profiling ranges.

See the [PRD097 validation resources](../../core/anet-core/testdata/prd097/README.md) for acceptance procedures covering scalar/tensor values during notifications, aux lane alignment, result immutability, notification order by Runner type, reproducibility, and performance.

These Envs have dedicated tests:

- [LunarLanderEnv_test.cpp](../../core/envs/lunarlander1/src/LunarLanderEnv_test.cpp)
- [DropMergeEnv_test.cpp](../../core/envs/dropmerge1/src/DropMergeEnv_test.cpp)
- [GridMazeEnv_test.cpp](../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp)
- [ImageClsEnv_test.cpp](../../core/envs/imagecls1/src/ImageClsEnv_test.cpp)
- [AtariEnv_test.cpp](../../core/envs/atari1/src/AtariEnv_test.cpp)

Concrete single-Env tests also include paths through `VectorizedDiscreteBatchEnv`. ImageCls tests directly validate native batches, Dataset catalogs/caches, eval windows, and worker modes. When changing an Env or wrapper, add regression tests explicitly selecting the required worker modes.

See the [PRD100 validation assets](../../core/anet-core/testdata/prd100/README.md) for GridMaze golden maps, reachability, coordinate errors, log counts and exact short-Run scalar comparisons.

See the [PRD102 validation assets](../../core/anet-core/testdata/prd102/README.md) for seed precedence, automatic scope, concurrent initialization, recreation, RunManager integration, optional configuration and resolved-value differences.

## 9. Related Documents

- [Framework Overview](010_framework_overview.en.md)
- [Runtime Infrastructure and Configuration](100_runtime_and_configuration.en.md)
- [Agents and Learning](110_agents_and_learning.en.md)
- [ReplayBuffer](150_replay_buffer.en.md)
- [Applications and Tools](160_applications_and_tools.en.md)
- [Glossary](../../CONTEXT.md)
- [ImageCls Batch Input PRD](../memo/done/034_imagecls_batch_input_10prd.md)
- [ImageCls Batch Env Seam ADR](../adr/0009-imagecls-batch-env-seam.md)

### Transition Statistics in Concrete Envs

| Env | Step-result keys | Ordinary Step | Terminal Step | Env itself |
|---|---|---|---|---|
| GridMaze | reward_sum, wall_collisions | Cumulative provisional values | Final values | Unknown |
| GridMaze | is_success, is_hole | NaN | 0/1 derived from terminal reward | Unknown |
| DropMerge | Cumulative statistics including ep_step, ep_max_rank, ep_mean_settle_steps, ep_max_settle_steps, ep_mean_blocked_frames, ep_max_blocked_frames, ep_same_drop_col_ratio, ep_double_suika_created, ep_double_suika_achieved, ep_blocked_run_count | Current cumulative values; NaN when unavailable | Final values | Unknown |
| DropMerge | ep_end_fruit_count, term_reason_*, blocked_drop_on_candidate, no_drop_timeout_on_candidate, ep_terminal_blocked_frames | NaN | Final values | Unknown |
| ImageCls | accuracy, epoch_count | Snapshot of the latest scoring window; accuracy is NaN before the first window completes | Final window values | Current overall values are also exposed |

DropMerge retains the same statistical formulas and floating-point operation order as before values moved into results. View aux last_score/last_step/last_term_reason and reward history are preserved. ImageCls prefetches once inside Step, while the result retains values for the scored window. DropMerge and LunarLander results own transition-time native rendering values and convert them to Tensors only when display is requested. They retain no borrowed reference to the original Env.
