<!-- translated-from: 020_user_guide_run.jp.md blob:1ffc0d1074a50e095a8d59bf7135f1c6216a6c73 date:2026-09-19 progress:done -->
# Run Execution User Guide

> Primary perspective: workflow (configuration, startup, operation, shutdown, and artifact inspection)

## 1. Introduction

### 1.1 Purpose

This guide explains the basic steps from configuring a Run in ANET RL Runner through operating the training and evaluation screens to inspecting artifacts after shutdown.

### 1.2 Audience

- Users configuring existing Envs and Agents to execute Runs
- Users learning basic operations of Train, Eval, and visualization panels
- Users inspecting logs, metrics, and checkpoints saved in a Run directory

### 1.3 Scope

This guide covers the current `AnetRLRunner`, `apps/runner/config` and its configuration syntax, standard GUI operations, and Run artifacts.
Implementing new Envs, Agents, or Observers is outside its scope; see the respective design documents.

> [!NOTE]
> The execution paths in this guide have been verified with Windows x64 and NVIDIA CUDA. CPU-only, Linux, macOS, and other GPU backends are unverified and are not guaranteed to produce the same results.

## 2. Preparation

### 2.1 Requirements

This guide assumes that `apps/runner/bin/Release/AnetRLRunner.exe` has been built and that the DLLs and configuration required by the runner are installed. The verified configuration uses Windows x64, an NVIDIA driver/CUDA runtime, and CUDA-enabled libtorch.

See [Development Environment](040_development_environment.en.md) for development setup, dependencies, CMake presets, and build instructions.

### 2.2 Choosing Configuration Files

When started without arguments, the runner displays a workspace selection dialog and selects the Env from the chosen workspace's `config/_main.txt`. The shared `apps/runner/config/_main.txt` loads only Agent, Network, metrics, and similar settings; Env selection is kept in the workspace. For a new workspace, or missing workspace configuration when first selecting an existing directory, `apps/runner/config/_workspace_template.txt` is copied to `config/_main.txt`.

```text
# apps/runner/workspaces/<workspace>/config/_main.txt
#$include <LunarLander.txt>
$include <DropMerge.txt>
```

Enable only the intended Env configuration for a Run. See [3. Writing Configuration Files](#3-writing-configuration-files) for selection chains such as `app.$`, `DefaultDQNAgent.$`, and `metrics.scalar.$` within each Env configuration, the distinction between `=` and `?=`, and the effects of command-line `key=value` assignments.

### 2.3 Initial Settings to Check

| Key | Role |
|---|---|
| `app.run_name` | Run name. `{t}` expands to the startup time |
| `app.runs_dir` | In workspace mode, Runner derives this as `<workspace>/runs`. Configuration and CLI changes are prohibited |
| `app.train_auto_start` | Starts training after GUI initialization when `true` |
| `app.show_error_dialog` | Whether to display a modal dialog in addition to an error log. Defaults to `true` |
| `app.save_agent_on_close` | Whether to save `agent_close.anet` automatically on shutdown. Defaults to `true`. Manual Save Checkpoint remains available when `false` |
| `app.eval_panel.auto_start` | Whether to run the manual EvalPanel immediately after startup |
| `run.seed` | Base seed for the Run |
| `run.train.num_envs` | Number of lanes in the training BatchEnv |
| `run.train.actor` | Catalog name of the training Actor. Defaults to `train` |
| `run.eval.[tag].actor` | Catalog name of the evaluation Actor. Defaults to the evaluation tag name |
| `agent.class_id` | Agent implementation to use |
| `agent.device` | Agent device, default `auto` |
| `env.worker_type` / `env.worker_threads` | Env batch execution mode and worker count |
| `env.device` | Env device, default `cpu` |
| `run.eval_device` | Device for configured eval, default `auto` |
| `backend.deterministic_algorithms` | Whether to require deterministic algorithms |

Device values are `auto`, `cpu`, `cuda`, and `cuda:N`. `auto` selects the current CUDA device when CUDA is available and CPU otherwise. The requested values remain in `config/config_data.txt`; adopted values appear in `json/agent.json`, `json/env.json`, and `json/run.json` as `effective_device` or `effective_eval_device`. When Env runs on the CPU and Agent and Eval on CUDA, include device transfer costs when assessing performance.

### 2.4 Choosing an Evaluation Slot's Policy

Evaluation slots refer to `<Agent>.actor.[key]` by name. Policy type, epsilon, network, and cloning are specified in the Agent catalog. For example, to evaluate DefaultDQN's online network with epsilon 0.1:

```properties
DefaultDQNAgent.actor.[explore].$ = DefaultDQNAgent.actor.[eval]
DefaultDQNAgent.actor.[explore].policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.[explore].policy.eps_start = 0.1
DefaultDQNAgent.actor.[explore].policy.eps_end = 0.1
DefaultDQNAgent.actor.[explore].policy.eps_decay_steps = 0
DefaultDQNAgent.actor.[explore].network = online
DefaultDQNAgent.actor.[explore].clone_model = true
run.eval.[explore].actor = explore
run.eval.[explore].eval_batch_size = 2
run.eval.[explore].eval_episodes = 2
run.eval_schedule.[explore].interval = 100
run.eval_schedule.[explore].use_background = false
metrics.scalar.[explore/epsilon] = epsilon $actor @session_end $eval.[explore]
```

The shared slots are `eval_target` and `eval`, referring to target and online respectively. Existing metrics output tags are retained. Undefined Actor references fail at startup, but definitions without schedules and slots with `interval=0` do not create Actors. Shared Actors use the same device as the Agent. MuZero does not support cloning; specify `clone_model=false`.

## 3. Writing Configuration Files

This chapter explains only the syntax and precedence needed to read and write files in `apps/runner/config`. [CONTEXT.md](../../CONTEXT.md) is the canonical terminology reference. Section 2.1 of [Runtime Infrastructure and Configuration](100_runtime_and_configuration.en.md) covers detailed contracts and resolver internals.

### 3.1 Basic Policy

1. Build configuration as a base plus differences. Shared files and profiles provide the base, selection chains joined with `>` provide differences, and values written directly in Env-specific files provide the final explicit assignments.
2. `X.$ = …` specifies the base for configuration X. `X.key = v` is an explicit assignment (explicit leaf) that overrides the base; `X.key ?= v` is a default value (default leaf) overridden by the base.
3. The order of lines with different keys does not affect the result. If the same key is written twice, the later value wins; if `=` and `?=` are mixed, `=` wins regardless of order. Reassigning the same `X.$` replaces its entire chain.
4. Run profiles and command-line assignments override ordinary configuration. Their precedence applies only to the specified key, not to every leaf inherited from that key.
5. Choose operators according to the file's role. Use `?=` for base definitions in shared files (`common.txt`, `agent.txt`, `nn*.txt`, `metrics_*.txt`). In Env-specific files (`<Env>.txt`), use `?=` only in the default-settings block and `=` elsewhere. Selection declarations `.$`, Run profiles, and CLI assignments always use `=`.

### 3.2 Terminology

| Term | Meaning | Example |
|---|---|---|
| Profile | A named configuration component containing a segment beginning with `@`. It does not contribute values until selected | `backend.@deterministic`, `DefaultDQNAgent.@baseline` |
| Selection chain | A declaration such as `X.$ = A > B` that merges each term's final values from left to right as differences to build X's base | `DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2` |
| Final selection values | The set of values and keys after applying the selection source's own base, partial selections, explicit assignments, Run, and CLI settings. Each chain term contributes this set | The final values of `@iqn` include its leaves and the result of its own `.$` |
| Catalog | A set of component definitions identified by `[key]` and referenced by name | `net.block.[Linear_120]`, `run.eval.[test1]`, `metrics.scalar.@baseline.[21_eval/01_target_reward]` |
| Override layer | An ordinary prefix for differences placed at the end of a chain in an Env-specific file. A1/A2/A3 target Agent, E1 targets Env, M1/M2 target Metrics, and P1 targets app; larger numbers indicate more temporary settings | `A2.learner.per_alpha = 0.2` |
| Run profile | A named set of selections and values characterizing a Run, declared as `run.@<name>` and selected with `run.$` | `run.@repro` |
| Explicit assignment (explicit leaf) | A value assigned directly to a configuration leaf with `=`, overriding the base | `E1.obs_include_action = true` |
| Default leaf | A default written with `?=`, overridden only when that configuration's base supplies the same key | `AtariEnv.game ?= pong` |

### 3.3 Syntax Reference

| Syntax | Example | Meaning |
|---|---|---|
| `key = value` | `E1.obs_include_action = true` | Explicit leaf. Overrides the base and default leaf |
| `key ?= value` | `AtariEnv.game ?= pong` | Default leaf. Remains unless the configuration's `.$` supplies the same key |
| `Owner.@name.key = v` | `LunarLanderEnv.@trunk.wind_power = 3.0` | Profile definition (dot form) |
| `Owner.@name : key = v` | `DefaultDQNAgent.@qr  : quantile_mode = qr` | Same as above (colon form). `:` means the same as `.` and is used only at a profile boundary |
| `@vars : key = v` | `@vars : max_exp_step  = 100,000,000` | Value slot, referenced with `${@vars.max_exp_step}` |
| `Owner.$ = A > B > C` | `DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2` | Selection chain. Short `@name` refers to `Owner.@name` |
| Fully qualified term | `LunarLanderEnv.$ = LunarLanderEnv.@trunk > E1` | Refers to the prefix exactly as written. Definitions owned by another owner can also be selected |
| `Owner.sub.$ = …` | `run.eval.[test1].env.$ = LunarLanderEnv.@test1` | Partial selection: a base for only part of the subtree |
| `[key].leaf`, `[key].$` | `net.block.[Linear_120].type ?= Linear`, `net.block.[MLP_FC1].$ = net.block.[Linear_120] > net.block.FC1` | Definition of a catalog item and its base |
| Metrics definition and selection | `metrics.scalar.@baseline.[21_eval/01_target_reward] ?= mean.episode_return $runner @session_end $eval.[eval_target]`, `metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@iqn_search_p0 > M1` | Catalog whose `[key]` is a tag; selection uses a chain |
| Override layer | `A2.learner.per_alpha = 0.2`, `app.$ = app.online > P1` | Differences in an Env-specific file, placed at the end of the chain |
| `run.@name : key = v` | `run.@iqn32_stratified : A2.learner.per_alpha = 0.2` | Run profile leaf |
| `run.@name : Owner.$ = …` | `run.@repro : backend.$ = backend.@deterministic` | Chain replacement by a Run profile |
| `run.$ = …` | `#run.$ = run.@repro` (file), `run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base` (bat/CLI) | Run selection |
| `$include <file>` | `$include <common.txt>` | Include. The `"path"` form is also supported |
| CLI `key=value` | `E1.game=breakout`, `A3.auto_load_file=<path>` | Startup argument. The specified key has highest precedence |
| `${full.key}` | `app.run_name = run_{t}_atari_${E1.game}` | One-level reference to a resolved final value |
| `#`, `//` | `DropMergeEnv.grid_cols ?= 40              # Grid columns (horizontal)` | The rest of the line is a comment |
| `{t}` | `app.run_name ?= run_{t}` | Expands to the startup time |

### 3.4 Syntax Details

#### 3.4.1 Leaf Assignment with `=` and `?=`

`key = value` declares an explicit leaf; `key ?= value` declares a default leaf.

- An explicit leaf overrides values supplied by the same configuration's `.$`. This also applies within a profile: combining `X.@fast : $ = @baseline` with `X.@fast : lr = 0.01` makes `@fast` equal to `@baseline` with only `lr` changed.
- A default leaf is overridden if that configuration's `.$` (whole or partial) supplies the same key; otherwise it remains. Replacing a chain does not remove default leaves.
- For repeated assignments to a key with the same operator, later values win, including across `$include` and additional files. If `=` and `?=` are mixed, `=` wins regardless of order; a later default leaf never cancels an explicit leaf.
- `?=` is allowed only on leaves. Using it on `.$`, a Run profile (`run.@x : …`), or the CLI causes a load error.
- Pitfall: separating `?` and `=` as in `key ? = v` is an error. A `?=` within a profile yields only to that profile's own `.$`; it reaches the outer configuration as an ordinary base value.

```properties
backend.cudnn_benchmark ?= false                   # 既定葉
backend.torch_num_threads ?= 1                     # 既定葉
backend.@non-deterministic.cudnn_benchmark = true
backend.$ = backend.@non-deterministic             # ベースが cudnn_benchmark を供給する → true
                                                   # torch_num_threads はベースに無い → 1 のまま
```

#### 3.4.2 Profiles `@name`

`Owner.@name.key = v` and `Owner.@name : key = v` mean the same thing; `:` is used only at a profile boundary.

- A definition alone does not contribute values. Only the portion selected by a chain or partial selection enters the final values.
- Short `@name` resolves in the namespace where it is written (the definition site). `Env.$ = @a` refers to `Env.@a`, and `Env.@a : $ = @b` refers to `Env.@b`. Even when used elsewhere as `Other.$ = Env.@a`, the `@b` inside `@a` still refers to `Env.@b`.
- In a Run profile, `@a` in `run.@x : Env.$ = @a` refers to `Env.@a` (the expansion destination's namespace).
- Profiles can have their own `.$` (`X.@fast : $ = @baseline`). Normally use only one `@` on the left-hand side.
- Names such as `@vars` serve as value slots, referenced with `${@vars.key}`.

```properties
DefaultDQNAgent.@qr  : quantile_mode = qr
DefaultDQNAgent.@qr  : net.$ = net.@qr
DefaultDQNAgent.@iqn : quantile_mode = iqn
DefaultDQNAgent.@iqn : net.$ = net.@iqn
```

#### 3.4.3 Selection Chains `.$` and `>`

Write `Owner.$ = A > B > C`. A term can be a profile (`@name`, `Owner.@name`), catalog item (`net.block.[X]`), or ordinary prefix (`A2`, `app.online`).

- Each term contributes its own final values, overlaid from left to right. Keys present on the right override earlier values; keys absent on the right remain from the left (differential composition).
- A term's internal `.$` builds that term's base and is not rerun at the copy destination.
- Reassigning the same `Owner.$` replaces the whole chain. Writing `Env.$ = A3` after `Env.$ = A2` selects only A3. To combine them, write a single chain.
- Ordinary prefixes without `.$` are allowed, even when nothing is defined under `A3`. Selecting an undefined `@` profile or catalog item is an error.
- Selections can nest up to depth 10.

```properties
DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
```

The differences from `@iqn`, `@heavy`, A1 (permanent Env-specific adjustments), `@bf16`, and A2 (experimental values) are overlaid on `@baseline` in that order. Keys absent from A2 remain from the left.

#### 3.4.4 Partial Selection

`Owner.sub.$ = …` specifies a base for only part of the subtree. A nested partial `.$` overrides the whole `.$`, regardless of line order. A partial `.$` inside a profile reaches the outer configuration as part of that profile's final values; a partial `.$` written at the same location at the root takes precedence over it.

```properties
run.eval.[test1].env.$ = LunarLanderEnv.@test1
DefaultDQNAgent.@qr : net.$ = net.@qr
```

#### 3.4.5 Catalogs `[key]`

`[key]` is the identity of a catalog item. Current examples include NN blocks (`net.block.[Linear_120]`), metrics tags (`metrics.scalar.@baseline.[21_eval/01_target_reward]`), and eval tags (`run.eval.[test1]`). Define items with `[key].leaf = v` and their bases with `[key].$ = …`. Inheritance between items also uses chains. Referencing an undefined item is an error.

```properties
net.block.[Linear_120].type ?= Linear
net.block.[Linear_120].linear.out_features ?= 120
net.block.[MLP_FC1].$ = net.block.[Linear_120] > net.block.FC1
```

#### 3.4.6 Override Layers

Override layers hold experimental values and Env-specific differences in Env-specific files. A1/A2/A3 target Agent, E1 targets Env, M1/M2 target Metrics, and P1 targets app. Larger numbers indicate more temporary settings by convention; to the resolver these are ordinary prefixes, without special handling based on names. Putting them at the end of a chain makes them the last differences applied.

```properties
A2.learner.iqn.current_taus.num_taus = 32
DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
```

Pitfall: an override layer's values still yield to explicit leaves assigned with `=` at the root of that configuration. If the root specifies `DefaultDQNAgent.learner.iqn.current_taus.num_taus = 8`, it overrides A2's value of 32.

#### 3.4.7 Run Profiles `run.@name` and `run.$`

Use `run.@name : Owner.key = v` for leaves, `run.@name : Owner.$ = …` for chain replacements, and `run.$ = run.@a > run.@b` to select them.

- `run.$` expands before ordinary selections. Terms apply from left to right with later values winning; if multiple terms assign the same key, the rightmost value wins.
- Run profile leaves occupy a separate precedence level above ordinary `=` assignments and below only the CLI. Assigning `Owner.$` in a Run profile replaces the chain.
- `?=` is prohibited in Run profiles. Nesting that supplies another `run.$` is also prohibited.
- In files, use comments such as `#run.$ = run.@repro` to switch selections; in bat files or the CLI, use `run.$=run.@a>run.@b`. Keep seed outside Run profiles and specify it separately with `run.seed`, allowing multiple seeds with the same profile to be compared.

```properties
run.@repro : backend.$ = backend.@deterministic
run.@repro : DefaultDQNAgent.$ = @baseline > @iqn > @heavy > A1 > @bf16 > A2
run.@repro : LunarLanderEnv.$ = LunarLanderEnv.@trunk > E1
#run.$ = run.@repro
```

#### 3.4.8 Command-Line `key=value`

The syntax matches files. Write assignments without spaces, as in `E1.game=breakout`, or quote them. Their effect depends on the type of key specified.

| Assignment | Target and effect |
|---|---|
| `AtariEnv.game=breakout` | The effective leaf itself. Always overrides file, inherited, and Run profile values |
| `E1.game=breakout` | A leaf in override layer E1. The value breakout reaches the destination through `AtariEnv.$ = … > E1`, but an explicit root assignment `AtariEnv.game = pong` still wins |
| `AtariEnv.$=AtariEnv.@classic>E1` | Only the selection key. Replaces the entire chain; explicit and default leaves remain |
| `run.$=run.@a>run.@b` | Replaces the Run selection itself |

`?=` is prohibited on the CLI.

#### 3.4.9 `$include`, `${full.key}`, and Comments

- `$include <name>` searches the including file's directory first, then the config search dirs. The `"path"` form is also supported. Maximum depth is 10. Missing files produce a WARN and processing continues, so check `config/config_data.txt` to confirm that the intended configuration was loaded.
- See [4.1 Standard Startup](#41-standard-startup) for startup overlay order: shared main config, derived `app.runs_dir`, workspace `config/_main.txt`, then command line.
- `${full.key}` refers to a resolved final value at one level only. A missing target or a target containing another `${}` is an error.
- Text after `#` or `//` is a comment. Lines without `=` are skipped. Spaces in keys are removed, and only one `:` is allowed. `{t}` expands to the startup time.

### 3.5 Precedence Summary

```text
X.key ?= v              既定葉。X.$ に負ける
 < X.$ = A > B          全体のベース(チェーン。項同士は右勝ち)
 < X.sub.$ = C          部分のベース(全体より強い)
 < X.key = v            個別葉
 < run.@p : X.key = v   Run プロファイルの葉
 < CLI X.key=v          そのキーだけ最優先
```

- This precedence applies recursively per configuration (owner). Each chain term resolves its final values under its own precedence before merging; thus even a leaf assigned with `=` inside A3 is a base value from the outer configuration X's perspective.
- Reassigning the same key uses the later value; mixed `=` and `?=` use `=`; reassigning `X.$` replaces the chain (item 3 in Section 3.1).
- The highest precedence of Run profiles and CLI assignments applies only to the specified key (see the table in Section 3.4.8).

### 3.6 What to Write in Each File

| File | Role | Operator |
|---|---|---|
| `_main.txt` | List of `$include` directives for shared files | - |
| `common.txt` | Shared defaults for trainer, agent, env, gui, backend, and app | `?=`. Switching profiles such as `backend.@deterministic` use `=` |
| `agent.txt` | Each Agent's `@baseline`, switching profiles (`@qr`, `@iqn`, `@bf16`, etc.), and `<Agent>.$ = @baseline` | `?=` for `@baseline`; `=` for switching profiles and chains |
| `nn.txt`, `nn_cnx.txt` | Shared NN components (`net.block.[…]`, `net.body.@…`) | `?=` |
| `metrics_scalar.txt`, `metrics_image.txt` | Metrics catalogs such as `metrics.scalar.@baseline.[tag]` | `?=` |
| `<Env>.txt` | Env and algorithm configuration: chains, profiles, override layers, Run profiles, NN wiring, metrics differences | `=`. Only the Env default-settings block uses `?=`. Explain any leaf requiring `?=` outside that block in a comment |
| Workspace `config/_main.txt` | `$include` directives for the `<Env>.txt` files to enable | - |
| bat, CLI | `run.$` selection and leaf overrides | `=` |

The convention is **`?=` for base definitions in shared files; in Env-specific files, `?=` only for defaults and generally `=` everywhere else** (the canonical rule is Configuration File Assignment Operators in [AGENTS.md](../../AGENTS.md)). Do not use `?=` merely because a value might be overridden later. Group only the leaves that require `?=` to preserve existing effective values into defaults, with reasons. Block headings and comment styles are unrestricted (`DropMerge.txt` has a DropMergeEnv default-settings heading; `LunarLander.txt` has none). This is an editing and auditing convention; the resolver does not change behavior by file name or location. After adding or changing configuration, run the static check for `=` remaining in shared/default settings and `?=` mixed into Env-specific explicit assignments.

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

`rg` (ripgrep) must be on PATH. Defaults intentionally written with `=`, such as seeds in `DropMerge_optuna.txt`, must have their classification reasons registered in the checker.

### 3.7 Examples

**Backend default leaves and switching**

```properties
# common.txt
backend.cudnn_benchmark ?= false
backend.cudnn_deterministic ?= false
backend.deterministic_algorithms ?= true
backend.deterministic_warn_only ?= false
backend.@non-deterministic.cudnn_benchmark = true
backend.@non-deterministic.cudnn_deterministic = false
backend.@non-deterministic.deterministic_algorithms = false

# LunarLander.txt
backend.$ = backend.@non-deterministic
run.@repro : backend.$ = backend.@deterministic
```

In a normal Run, `cudnn_benchmark`, `cudnn_deterministic`, and `deterministic_algorithms` take the values from `@non-deterministic`; other default leaves such as `deterministic_warn_only` remain unchanged. Starting with `run.$=run.@repro` replaces the chain with `@deterministic`, without affecting the default leaves.

**DropMergeEnv defaults and selection chain**

```properties
# DropMerge.txt
DropMergeEnv.$ = DropMergeEnv.@baseline > DropMergeEnv.@G5846 > DropMergeEnv.@heavy > E1

# --- DropMergeEnv デフォルト設定
DropMergeEnv.grid_cols ?= 40              # グリッド列数（横方向）
DropMergeEnv.grid_rows ?= 64              # グリッド行数（縦方向）
DropMergeEnv.action_mode ?= move_fast     # move move_fast direct direct_noop
```

`grid_cols` takes the value from the rightmost term supplying it among `@baseline`, `@G5846`, `@heavy`, and E1 (currently, `@G5846` supplies 58). Keys supplied by none of the terms retain their default-leaf values.

**`?=` outside the default-settings block**

```properties
# Atari.txt
DefaultDQNAgent.net.branch.[value_stream].structure ?= AtariHeadFC512 > SiLU # NatureDQNのReLU選択とIQNの出力先を維持するために?=
DefaultDQNAgent.net.branch.[adv_stream].structure ?= AtariHeadFC512 > SiLU   # NatureDQNのReLU選択とIQNの出力先を維持するために?=
DefaultDQNAgent.net.body.output.[features] ?= main_feature # NatureDQNのReLU選択とIQNの出力先を維持するために?=
```

These leaves use `?=` with explanatory comments to retain defaults even in configurations selecting `@nature` (ReLU) in the chain or changing the `[features]` output destination through IQN wiring. Only leaves with such reasons use `?=` in Env-specific files.

**Changing a value for this Run only**

```powershell
apps\runner\bin\Release\AnetRLRunner.exe --workspace atari-03 E1.game=breakout
apps\runner\bin\Release\AnetRLRunner.exe --workspace atari-03 AtariEnv.game=breakout
```

`E1.game=breakout` changes a leaf in override layer E1 and reaches the destination through `AtariEnv.$ = … > E1`. This normally suffices. `AtariEnv.game=breakout` targets the effective leaf itself and wins even if the file explicitly specifies `AtariEnv.game = pong`. Group repeatedly used combinations into a Run profile and select it with `run.$=run.@breakout`.

```properties
run.@breakout : E1.game = breakout
run.@breakout : app.run_name = run_{t}_breakout
```

### 3.8 Verification and Common Errors

- `config/config_data.txt` contains the resolved effective configuration. It also includes definitions of override layers (`A1`–`A3`, `E1`, `M1`/`M2`, `P1`) and ordinary selection-source prefixes such as `app.online`, but excludes `@` profiles and `.$`. This file does not reveal which profiles were selected.
- `json/config_resolution.json` contains selection records (`selections`, with `run.$` first and `key` identifying the declaration site), value references (`references`), and leaves changed by Run profiles (`overrides`). See Sections 6.6 and 6.7 of the [Analysis User Guide](030_user_guide_analysis.en.md) for interpretation.
- Check for omitted operators with the static check in Section 3.6.

| Message prefix | Cause | Action |
|---|---|---|
| `Properties: invalid assignment operator` | `?` and `=` are separated as in `key ? = v`, or `?` remains in the key | Write `?=` together |
| `Properties: default assignment is only allowed for ordinary leaves` | `?=` was used on `.$` or in a Run profile | Use `=` |
| `ConfigManager: default assignment is not allowed on command line` | `?=` was used on the CLI | Use `=` |
| `ConfigResolver: material selection target not found` | An undefined `@` profile was selected (`material` means profile) | Check the definition namespace and spelling |
| `ConfigResolver: catalog selection target not found` | An undefined `[key]` was selected | Same as above |
| `ConfigResolver: selection self-supply detected`, `selection cycle detected` | A selection refers to itself or its descendants, or selections form a cycle | Review chain references |
| `ConfigResolver: selection depth limit exceeded` | Selection nesting exceeds 10 levels | Reduce intermediate profiles |
| `ConfigResolver: value reference target not found`, `chained value reference is not supported` | The `${}` target is missing or contains another `${}` | Refer to a key containing a final value |
| `ConfigResolver: named trunk must not select another trunk` | `run.$` was written inside a Run profile | Write `run.$` only once at the startup side |
| `Workspace config changed app.runs_dir` | Workspace configuration or CLI changed `app.runs_dir` | Do not change it in workspace mode |
| WARN `Properties: Failed to open include file` | An `$include` target is missing (processing continues without an error) | Check the path and config search dirs |

## 4. Starting a Run

### 4.1 Standard Startup

Run the following from `apps/runner`.

```powershell
10_run.bat
```

On first launch, the selection dialog opens with `_default` prefilled as a new name. Choose from history, all directories directly under `workspaces/`, an arbitrary path, or a new name. Existing directories without `config/_main.txt`, such as folders containing only relocated past Runs, are also listed; selecting one fills in only the missing configuration. New names are validated as entered, and OK remains disabled while a reason for invalidity appears below the input. Use `--workspace dm_long` to specify a relative workspace directly, or `--select-workspace` to show the dialog regardless of the skip preference. Relative paths are based on `apps/runner/workspaces/`; absolute paths are also supported. Leading and trailing whitespace is trimmed, and `#`, `//`, trailing `;`, and UNC paths are rejected.

Alternatively, specify the executable and main config from the repository root.

```powershell
apps\runner\bin\Release\AnetRLRunner.exe `
  --config apps\runner\config\_main.txt `
  app.run_name=run_{t}_trial `
  run.seed=12345
```

`--config` is a fully self-contained mode that never consults workspaces, history, or `last_workspace.txt`. Combining it with `--workspace` or `--select-workspace` is a startup error.

Startup initialization proceeds approximately in this order:

1. Resolve the workspace, then shared main config, derived `app.runs_dir`, workspace config, and command-line overrides in that order.
2. Prepare the Run directory, `metrics.jsonl`, and standard output logs.
3. Initialize the libtorch backend and registered Envs.
4. `RunManager` constructs Env, Agent, Train Runner, and configured Evals.
5. Connect the Train, Eval, QValue, and Log panels.
6. Start `RunnerThread`. With `app.train_auto_start=false`, it waits paused.

If startup fails, inspect the error log. Online configurations display an error dialog in addition to logging. Batchrun configurations do not display modal dialogs: errors go to the parent process's stderr before the Run directory exists, the Run's `stderr.log` after `StandardStreamLogger` starts, and `<run_name>.log` after the regular logger is constructed. A process that handles a fatal error exits with a nonzero status.

### 4.2 Automatic Exit and Pause

`app.train_exit_step` and `app.exp_exit_step` terminate the Run when their limits are reached. `app.train_pause_step` and `app.exp_pause_step` pause automatically once.
For batch experiments, `app.$=app.batchrun` selects a batchrun configuration combining low-FPS display, `exp_exit_step`, and `app.show_error_dialog=false`. For human-operated online runs, `app.$=app.online` selects `app.show_error_dialog=true`. These are separate concepts from Train/Eval `RunMode`.

`apps/11_batch_run.bat`, `apps/12_batch_run.bat`, and `apps/18_batch_run_atari5.bat` record each Run's exit code. A failed Run displays `[ERROR] RUN FAILED exit_code=<code> args=<args>`, then execution continues with subsequent Runs. After all Runs and the final `pause`, the script returns 1 if any failed, or 0 if all succeeded.

## 5. Application Screen

### 5.1 Basic Layout

The Runner screen consists of wxAUI panes.

Four task-specific toolbars appear at the top. Drag their grippers to dock, float, or redock them. `View > Reset Layout` restores the default single row at the top. Floating toolbar windows display their names (`Run Control`, `Steps`, `Run Operations`, `Panels`) as window titles.

- `Run Control`: provides Train pause/resume and, after a separator, Eval pause/resume and single-step execution. A running tool appears pressed, and its icon changes to a pause symbol representing the next action; while stopped, it returns to a play symbol. Resuming or stepping a hidden Eval also shows `Evaluation View`. Right-click resume behaves the same way; only the keyboard paths (`Space` / `Ctrl`) operate independently of visibility.
- `Steps`: displays Train `exp_step` and `train_step`.
- `Run Operations`: provides checkpoint saving to an arbitrary path and opening the Run folder.
- `Panels`: toggles `Logs`, `Eval View`, and `Q-Values`, synchronized with the corresponding `View` menu entries and pane state.

- `Train View`: displays the Env-specific View received from Train Runner.
- `Evaluation View`: advances an Eval Env separate from Train, manually or by timer. Initially hidden.
- `Evaluation Q-Values`: displays Eval Actor output and allows manual Action selection.
- `Logs`: displays runtime logs with Error, Warn, Info, and Verbose levels.
- `HeatMap` / `Conv2d`: additional panes opened from the `View` menu.

![DropMerge Train View and Evaluation View](assets/020_runner_dropmerge_train_eval.png)

![LunarLander runtime screen and visualization panes](assets/020_runner_lunarlander_visualization.png)

Closing panes or displaying a View before initialization may leave few items to render, as below. Use `View > Reset Layout` to restore the default layout.

![Runner screen with no rendering targets](assets/020_runner_empty_view.png)

## 6. Operations

### 6.1 Pausing and Resuming Training

Select `Train` on the Run Control toolbar to pause/resume Train. Left-clicking the Train/Eval View or pressing `Shift` performs the same operation. These also start training when launched with `app.train_auto_start=false`. Pausing explicitly flushes metrics, stdout/stderr, and text logs. The tool is disabled once Train has stopped and cannot resume.

### 6.2 Evaluation and Screen Operations

| Operation | Behavior |
|---|---|
| `Eval` on the Run Control toolbar | Pauses/resumes EvalPanel. Shows the pane on resume if hidden |
| `Step` on the Run Control toolbar | Automatically executes one Eval step. Shows the pane if hidden |
| Right-click | Pauses/resumes EvalPanel. Shows the pane on resume if hidden |
| `Space` | Pauses/resumes EvalPanel without changing pane visibility |
| `Ctrl` | Automatically executes one Eval step |
| Arrow keys | Select Actions `0` through `3` for LunarLander and advance Eval one step |
| Numpad `0` through `9` | Select the corresponding Action and advance Eval one step |
| `View > Evaluation View` | Toggles the Eval pane |
| `View > Evaluation QValue View` | Toggles the Q-value pane |
| `View > Log Level` | Changes the log level displayed in the GUI |
| `View > Reset Layout` | Restores default pane layout and frame size |
| `Save Checkpoint` on the Run Operations toolbar | Pauses Train first if running, then saves the Agent to an arbitrary path, defaulting to the Run directory and `agent_<exp_step>.anet` |
| `Open Run Folder` on the Run Operations toolbar | Opens the current Run directory in Explorer |

The number of Actions varies by Env. Check the QValue pane or the Env's ActionSpec rather than assuming out-of-range Actions are valid.

EvalPanel's shared default is `app.eval_panel.eval_config_tag = eval_panel`, referencing a dedicated Actor through `run.eval.[eval_panel].actor = eval_panel`. DQN Agents use Greedy on the target net (Rainbow uses ε=0); MuZero uses temperature 0 without search noise; ImageCls selects the highest-scoring Action. Periodic evaluation's ε and UQE settings remain available. Override `<Agent>.actor.[eval_panel].*` to change the display policy, or set `app.eval_panel.eval_config_tag` to display a different evaluation configuration.

`app.eval_panel.model_sync.mode` synchronizes periodically by `frame`, `time`, or `episode`. The referenced `<Agent>.actor.[key].clone_model` determines whether to clone the model. Even shared models update the training-side counts at synchronization, and synchronization also occurs on resume. The displayed Eval may not always match Train's latest parameters; record the Actor name, clone setting, mode, and interval when comparing.

### 6.3 Display FPS and Progress

`View > Train View FPS` changes only Train View's rendering frequency. `0 (Off)` stops its rendering timer while training continues. `View > Eval View FPS` changes EvalPanel's timer period, affecting Eval progress speed as well as rendering. Both `Config (N)` entries restore their startup config values. These are runtime UI operations; selections are not written back to the Run's config dump.

The Steps toolbar displays `exp` and `train` counts in separate read-only text fields. Values can be selected and copied, and a standard separator divides the fields. The right side of the status bar shows `exp <N> steps/s    train <N> steps/s` and elapsed time. Before the SPS EMA initializes, its value is `-`; before the first Train snapshot, both step fields show `-` and elapsed time shows `--:--:--`. Elapsed time continues as wall-clock time during pauses.

### 6.4 Stopping, Saving, and Resuming from a Checkpoint

Close the window or select `File > Exit` to stop the Run. Shutdown stops Train, saves `agent_close.anet`, flushes Run output, and destroys the GUI in that order. With `app.save_agent_on_close=false`, only the save is skipped; the rest of the sequence remains unchanged. Forcing the process to terminate during saving may leave the checkpoint, metrics, or video tails incomplete, so wait for the window to close.

`Save Checkpoint` first pauses Train if it is running. This prevents steps from advancing during the dialog and causing the default filename to diverge from the saved contents. Train does not resume automatically after either saving or canceling; use `Train` on the Run Control toolbar or `Shift`. Saving itself is safe even while Train runs. `DefaultDQNAgent` protects the entire serialization with the Agent's shared lock, excluding Learner updates. Failures caused by permissions, disk space, file locks, or similar issues log the target path and reason and display a dialog in online configurations. This is non-fatal and does not affect the Run or process exit status. The failed file may be incomplete and is not deleted automatically; inspect its contents before handling it. Saving can be retried with a valid path.

If saving `agent_close.anet` on close fails, the same notification policy applies, and log shutdown and GUI cleanup continue until the window closes. In that case, `agent_close.anet` may not be a valid checkpoint. If the Agent does not implement Save, a zero-byte file remains and a WARN includes the target path. Even successfully saved checkpoints should be used for resumption only after checking that Agent, Network, and archive contracts match.

To resume from a checkpoint, specify the Agent-specific `auto_load_file` in a compatible configuration for a new Run. Current examples are `R.auto_load_file`, an alias for `DefaultDQNAgent.auto_load_file`, and `ImageClsAgent.auto_load_file`. Checkpoints with different Network structures or archive contracts cannot be loaded. Saved contents vary by Agent; current DQN Agents do not restore ReplayBuffer contents or sampling state. Resumption creates a new Run directory and step series; it does not append to the old Run's `metrics.jsonl`. See [DQN Agents](200_dqn_agents.en.md) for what DQN saves.

## 7. Artifacts

With workspace `dm_long` and `app.run_name=run_{t}`, artifacts are saved under `apps/runner/workspaces/dm_long/runs/<run_name>/`. An absolute-path workspace likewise stores them under its own `runs/` directory.

| Artifact | Contents |
|---|---|
| `metrics.jsonl` | Primary metrics file, appending scalars, JSON metadata, and video metadata |
| `config/config_data.txt` | Resolved effective configuration, retaining requested device values such as `auto` and excluding `@` profiles and `.$` ([3.8](#38-verification-and-common-errors)) |
| `config/*.txt`, `json/*.json` | Per-component injected configuration and metadata dumps. Env uses `config/env.<Env name>.txt` |
| `json/run.json`, `json/env.json`, `json/agent.json` | Adopted evaluation, Env, and Agent devices in `effective_eval_device` or `effective_device` |
| `<run_name>.log` | Runner text log with timestamps and levels |
| `stdout.log` / `stderr.log` | Process standard output and standard error |
| `agent_close.anet` | Agent checkpoint saved on normal window close. Not created with `app.save_agent_on_close=false` |
| `videos/*.mkv` | Videos generated by image Observers |
| `images/<tag>/*.png` | Individual frames when `app.metrics_logger.use_png_dump=true` |
| `dot/**/*.dot` | GraphViz Observer output |

For comparison or reproduction, treat `config/config_data.txt` in the Run directory as authoritative, rather than the local configuration before edits. See the [Analysis User Guide](030_user_guide_analysis.en.md) for graph analysis.

## 8. Common Checks

- Train does not advance immediately after startup: check `app.train_auto_start` and resume with a left-click or `Shift`.
- Eval does not advance: show `Evaluation View` and check `app.eval_panel.auto_start` or `Space`.
- Toolbar checked state does not match the operation: wait up to 200 ms for synchronization with actual state. Use `View > Reset Layout` if the layout is disrupted.
- Only Train View fails to update: check that `View > Train View FPS` is not `0 (Off)`.
- Save fails: check the target path and failure stage in the error log (also shown in a dialog in online configurations). Resolve permissions, disk space, or file locks, or choose another path and retry. The Run continues, but the failed output file may be incomplete.
- Save produces zero bytes: check the WARN with the target path in `<run_name>.log`. The Agent may not implement Save.
- Run folder does not open: check the target path and OS folder association in the error log (also shown in a dialog in online configurations). The Run continues after the failure.
- CUDA initialization fails: check the libtorch/CUDA/driver combination, `agent.device`, and `run.eval_device`.
- The Env differs from the intended one: check enabled Env includes in the selected workspace's `config/_main.txt` and the Run's `config/config_data.txt`.
- To choose a workspace again: launch with `--select-workspace`. Delete `GetAppDataDir()/history.txt` to reset history or `prefs.txt` to reset dialog preferences independently.
- View is empty: check the Env class ID, View factory, and initialization errors in the Log pane; also try `Reset Layout`.

## 9. Related Documents

- [CONTEXT.md](../../CONTEXT.md) (canonical configuration terminology)
- [ADR 0042](../adr/0042-config-inheritance-as-differential-base.md) (rationale for configuration contracts)
- [AGENTS.md](../../AGENTS.md), Configuration File Assignment Operators (operator usage rules)
- [Analysis User Guide](030_user_guide_analysis.en.md)
- [Development Environment](040_development_environment.en.md)
- [Runtime Infrastructure and Configuration](100_runtime_and_configuration.en.md)
- [Agents and Learning](110_agents_and_learning.en.md)
- [Environments](120_environments.en.md)
- [Observability](140_observability.en.md)
- [ReplayBuffer](150_replay_buffer.en.md)
- [DQN Agents](200_dqn_agents.en.md)
- [Applications and Tools](160_applications_and_tools.en.md)
