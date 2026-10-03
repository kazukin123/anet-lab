<!-- translated-from: 030_user_guide_analysis.jp.md blob:1b117ec47d6db6188550e08a78e6af821a5b49c1 date:2026-09-19 progress:done -->
# Run Analysis User Guide

> Primary perspective: workflow (artifact inspection, visualization, comparison, and interpretation)

## 1. Introduction

### 1.1 Purpose

This guide explains how to visualize a Run's `metrics.jsonl` in Metrics Viewer, compare multiple Runs under the same conditions, and inspect Optuna multi-seed aggregates when needed.

### 1.2 Audience

- Users comparing learning curves, evaluation results, losses, and performance metrics
- Users assessing Runs over matching step ranges without overstating differences
- Users distinguishing DropMerge Optuna seed runs from summary studies

### 1.3 Scope

This guide covers the current Java/Spring Metrics Viewer, its Plotly interface, Run artifacts, and browsing DropMerge Optuna results.
For implementing new metrics or modifying Optuna search spaces, see the design documents and the [DropMerge Optuna Guide](optuna.md).

## 2. Starting Metrics Viewer

This guide assumes that the Metrics Viewer jar has been built. See [Development Environment](040_development_environment.en.md) for Java, Maven, and jar build instructions.

To view ordinary Runs, launch the following from `apps/runner`.

```powershell
22_metrics_viewer_java.bat
```

- URL: `http://localhost:8082`
- Workspace directory: `apps/runner/workspaces`

Use the `Workspace` selector at the top to switch between workspaces containing `runs/` or `config/`.
The browser saves the last selection and restores it on the next startup if that workspace still exists.
If the saved workspace is missing, it falls back to the server's current workspace.

To read a custom collection of workspaces, pass `--metricsviewer.workspaces-dir=<path>` to the jar and optionally specify the startup workspace with `--metricsviewer.initial-workspace=<name>`. The Viewer recognizes only directories directly under the selected workspace's `runs/` containing `metrics.jsonl` or `metrics.jsonl.gz` as Runs. If both exist, it uses `metrics.jsonl`.

### 2.1 Compressing Metrics from Completed Runs

`apps/70_compress_workspace_metrics.bat` migrates `metrics.jsonl` to `metrics.jsonl.gz` for every Run directly under the selected workspace's `runs/`. With no arguments, it lists workspaces containing `runs/` in name order; enter a number to select one. Select `[0] EXIT` to close the launcher. A workspace name or absolute path can also be supplied as the first argument.

After preflight, enter one of the following at `Execute compression? [YES/NO/DRY-RUN]:`.

- `YES`: create and verify gzip, then delete the original `metrics.jsonl` after finalization.
- `NO`: cancel processing for that workspace without changing files.
- `DRY-RUN`: display targets, skip reasons, and required capacity without changing files.

Runs whose nonempty files lack a terminating newline, and Runs currently being written by Runner, are skipped unchanged.
After compression, Metrics Viewer, TensorBoard bridge, MLflow bridge, and Optuna aggregation still prefer raw files and transparently read gzip when raw is absent.
When launched without a workspace argument, the launcher pauses after displaying the result of `YES`, `NO`, or `DRY-RUN`, then returns to workspace selection after a keypress. `--no-pause` skips the pause. Multiple workspaces can be processed before selecting `[0] EXIT`. With a workspace argument, it processes once only.

## 3. Basic Interface Operations

![Metrics Viewer with multiple Runs selected](assets/030_metrics_viewer_run_comparison.png)

| UI | Behavior |
|---|---|
| `Workspace` | Switches to a workspace listed by the server and resets Run selection, colors, viewport, and legend state |
| `Runs` | Clicking a row toggles it immediately. Clicking the same row again within 350 ms selects only that Run. Empty selection is allowed |
| `Select All` / `Select Latest` | Selects all Runs or the latest Run |
| `Recolor` | Redistributes maximally distinguishable colors among selected Runs. Colors follow a fixed sequence in selection order, so the same Run count produces the same color set |
| `Auto Recolor` | Automatically recolors only Runs whose colors become hard to distinguish after selection changes. ON by default |
| `Tags` | Selects metric tags to display |
| `Selected Only` (Runs / Tags) | Header toggle that keeps only selected Runs or tags in the list. Bulk operations such as `Select All` automatically turn it OFF |
| Run row background and `%` | Per-Run progress importing the selected Metrics master into SQLite |
| `Reload` | Refetches the Run list, tags, and metric ranges for the current viewport, then redraws |
| `Auto Reload` | Refreshes metadata every 30 seconds and updates only graphs currently displaying the latest data |
| `LOD: MinMax / Mean / Band` | Global LOD display mode immediately before Scroll Lock at the upper right. Changing it does not refetch data |
| Graph `Log` | Toggles signed-log display supporting positive, negative, and zero values |
| Graph `p5–p95` | Displays only finite values within each Run's p5–p95 range among currently displayed values |
| `Scroll Lock` | Suppresses graph interaction and uses drag/swipe for vertical scrolling |
| Screenshot button | Hides side panels for comparison screenshots. While active, drag/swipe over graphs scrolls vertically as with Scroll Lock |

The Plotly modebar provides zoom, pan, image saving, and `Reset axes`. The redundant `Autoscale` button is hidden. Double-clicking the graph preserves Plotly's axis reset while also triggering the Viewer's Reload.

Only the initial display automatically selects the latest Run. Thereafter, manual empty selections and selections emptied by disappearing Runs are preserved.
Reload preserves known OFF tags and automatically enables only newly discovered visible tags.
The selected workspace, tags, LOD mode, Scroll Lock, `Auto Recolor`, and per-tag Log and p5–p95 settings persist in browser `localStorage`. Run colors themselves do not persist; they are reassigned on reload and workspace changes. Log and p5–p95 apply to same-named tags across workspaces.

p5–p95 is computed independently for each Run from LOD rendering values in the current X range, excluding Runs hidden in the legend. Points below p5 or above p95 are removed from display traces and therefore have no hover information. Plotly autorange applies to the remaining points. There is no minimum point count; the tooltip shows displayed and input counts. When combined with Log, percentiles are computed from raw values before transformation to signed-log coordinates. Manual Y zoom is preserved; `Reset View` or Plotly's axis reset restores autorange after filtering. Graph-header statistics and raw client-cache values remain unchanged.

LOD `MinMax` connects each bucket's min, max, and last in their original step order. `Mean` displays bucket averages; `Band` overlays a mean line on a min/max band. When there are few enough points for L0, every mode displays the raw line. The graph header's `Min / Max / Avg / Std` combines `TagStats` over all committed raw points from selected Runs, rather than the viewport.

Run lists and committed graphs remain interactive during ingestion. Run-level errors and tags quarantined for backward steps display `⚠` with a tooltip; their committed portions remain visible.

## 4. Comparing Runs

### 4.1 Check Comparison Conditions

Before overlaying graphs, check the following for each Run.

- `config/config_data.txt`: complete configuration after resolving includes and CLI overrides
- `config/*.txt`: per-component injected configuration; Env uses `config/env.<Env name>.txt`
- Seed, Agent/Env, Network structure, batch size, and replay conditions
- CPU/GPU, device index, and determinism settings
- Configured eval interval, RunMode, and clone settings
- Why the Run stopped and the step it reached

Similar names do not imply identical settings. Treat Run artifacts, rather than Run names, as authoritative.

### 4.2 Select the Same Tag and Step Axis

A scalar record in `metrics.jsonl` contains `type`, `tag`, `step`, and `value`, but does not itself store whether `step` means `train_step`, `learn_step`, `episode_step`, `exp_step`, or another axis. The Run's metrics configuration determines the axis.

When comparing, verify both matching tag names and matching step-axis selection in `metrics.scalar.[tag]`. In current configuration, `@learn`, `@episode_end`, and `@session_end` default to `exp_step`, while `@train` defaults to `train_step`. Explicit selections such as `$exp_step` take precedence.

### 4.3 Compare Matched `exp_step` Ranges

Do not directly compare the final points of Runs that reached different steps. Choose a shared `exp_step` range containing data from both, and evaluate the same window.

Example:

```text
Run A: 0 - 100M exp_step
Run B: 0 - 70M exp_step

比較window: 50M - 70M exp_step
```

Within the same window, distinguish:

- Level: window mean/median or evaluation EMA
- Stability: fluctuations, sudden drops, and range/std across seeds
- Trend: differences between the first and second halves of the window, and slope
- Speed: differences in `90_perf/90_elapse_hour` under identical hardware and parallelism conditions; use `exp_step_per_sec` as supporting information

Do not conclude that one configuration is faster or better from a single long Run. If stopping points differ, first align matched windows.

### 4.4 Avoid Misreading Display Conditions

- Curves with different EMA `ema_alpha` values have different smoothness; do not directly compare their variance.
- Different `interval` or evaluation frequencies change point density. Do not interpret smoother lines as better performance.
- Signed-log expands the region around zero. Do not interpret visual distances between lines as the same ratios as on a linear scale.
- Configured eval and EvalPanel are separate Runners. Check each tag's runner scope.
- `exp_step_per_sec` is affected by other processes, video output, profiling, and parallel Optuna jobs.
- `90_perf/12_exp_step_per_sec` is a time-weighted EMA with τ=10 seconds. Weighting by window duration includes stalls such as evaluation, but it is not instantaneous and should not define interval boundaries. Compute true throughput from differences in `90_perf/90_elapse_hour` and `exp_step`. `90_perf/22_exp_step_per_sec_ema` is a longer-term curve further smoothed by `ema_alpha`.
- In Runs predating the time-weighted EMA, the same tag underestimates stalls (one measurement displayed 1,830 steps/s versus a true 478 steps/s). Do not directly compare its values with past Runs.
- A difference from one seed includes seed variability. Reevaluate candidates across multiple seeds after selection.

### 4.5 Reading IQN Exploration P0 Diagnostics

For IQN exploration, first verify that `metrics.scalar.iqn_search_p0` is merged into the resolved `config/config_data.txt`, and check Policy/Learner tau placement modes and counts `K/N/M`. Do not decide adoption from diagnostics alone: assess DropMerge Double Suika counts and achievement rates, rewards, PER health, and throughput separately over the same matched `exp_step` window.

- `iqn_policy_margin_mc_ratio` normalizes the gap between the top two UQE actions by the finite risk-quantile sample scale. With `random`, interpret it as Monte Carlo mean stability; with `fixed` or `stratified`, as a proxy for integration resolution rather than random variation between forwards.
- Read `iqn_current_mc_scale` and `iqn_target_mc_scale` separately for `N` and `M`. `iqn_priority_mc_ratio` measures the size of the current mean-TD priority signal relative to the finite-tau scales on both sides.
- `iqn_first_pair_abs_td` and `iqn_first_cancellation_ratio` cover only rows receiving their first Learner priority update. When `per_sample_initial_count=0`, `iqn_first_*` is `NaN`; do not interpret this as improvement or deterioration to zero.
- With TBO enabled, Learner diagnostics are in h-space, matching current priorities, rather than real space. Do not directly compare absolute values between TBO-enabled and disabled Runs.
- `iqn_uqe_full_q_argmax_disagreement` and `action_full_q_margin.[i]` are defined only for Policies with full-distribution queries. Do not interpret missing-data `NaN` as agreement or zero margin.
- Compare P0 group OFF/ON overhead serially with the same binary, seed, and execution conditions, using throughput computed from matched-window differences in `90_perf/90_elapse_hour`. Reject measurements affected by other processes or parallel Optuna jobs.

### 4.6 Reading Quantile-Tail Exploration Diagnostics

Quantile-tail diagnostics are six scalars observing existing QR/IQN return distributions, rather than signals changing Policy or priority. First check the resolved `config/config_data.txt` to confirm that five Policy metrics are registered under `eval2` and one Learner metric under `@learn`. Record the Policy's fixed full-distribution count `K` and the Learner's PER and TBO status.

- `policy_upper_truncated_std` and `policy_lower_truncated_std` measure spread above and below the median for the final executed action, in the same units as Q values. Their difference reveals tail asymmetry, but a single network's spread does not establish parametric uncertainty or the effectiveness of an exploration bonus.
- `lower_risk_full_q_argmax_disagreement` measures how often full-Q argmax changes under a hypothetical lower-tail penalty with coefficient 1. Its purpose differs from `iqn_uqe_full_q_argmax_disagreement`; it does not show that a risk-averse Policy is effective in practice.
- `quantile_crossing_ratio` is the fraction of adjacent quantiles, ordered by tau, that decrease. Where it is high, interpreting upper/lower tails as regions of a quantile function is less reliable, so check ordering before comparing tail widths. Equal values do not count as crossings.
- `policy_selected_crossing_depth_p90_ratio` is dimensionless: normalize positive crossing depths within the final executed action by the distribution range, compute lane-wise nearest-rank p90 within the action event, then average over the batch. Combine it with `quantile_crossing_ratio`, which measures frequency over all actions. Flat frequency with declining p90 may indicate shallower local inversions; declining frequency with rising p90 may indicate a few deep inversions remaining. It is not p90 pooled across crossing samples from the entire Run.
- `upper_tail_priority_spearman` is the rank correlation between upper-tail width and raw priority after clipping, restricted to a minibatch already biased by PER sampling. High positive correlation may mean both emphasize similar experiences, but does not prove redundancy over the whole ReplayBuffer. Low or negative correlation does not prove a new signal's usefulness either.
- Corresponding values are `NaN` when PER is disabled, the batch is insufficient, rank sequences are constant, the Policy full distribution is missing, or `K < 2`. Do not reinterpret this as agreement with zero or zero correlation. Crossing-depth p90 is normally `0` when valid inputs have no positive crossings or have zero range. With TBO, values are in the same h-space as current Policy scores/priorities rather than real space; do not directly compare absolute values between TBO-enabled and disabled Runs.

### 4.7 Reading Plasticity Metrics

The `34_agent_plasticity` group directly measures the health of NN representations in three ways: **activation distributions** (how many units fire: `dormant_ratio` / `dead_ratio`; feature-vector magnitude: `feature_norm`), **direction distributions** (the effective number of directions used by features: `srank` / `srank_ratio`), and **parameter magnitude** (`weight_norm_feature` / `weight_norm_readout`). See the plasticity and representation statistics section in `CONTEXT.md` for definitions.

The tens digit identifies the channel; the units digit identifies the statistic.

| Decade | Channel | Default |
|---|---|---|
| `0x` | actual (features produced by the training forward; measured on the current update batch selected by PER) | ON |
| `2x` | target (features produced by the target network for TD computation in the same update) | OFF |
| `4x` | probe (uniform sampling without replacement from the entire ReplayBuffer, followed by a partial forward in NoGrad and eval mode) | ON |
| `6x` | weight norm / spectral sigma (parameter-side, independent of data and therefore without a channel) | Only 61/62 ON |

The units digit is shared across the three channels: `x1` dormant, `x2` dead, `x3` feature_norm, `x4` srank, `x5` srank_ratio, and `x6`–`x9` srank with different δ values (OFF by default). Comparing `02` / `22` / `42`, for example, compares the same statistic across channels. Channels are enabled independently by the presence of subscription rows, so first check `learner.plasticity.feature_key` and those rows in resolved `config/config_data.txt`. All rows use the `$learn_step` axis.

**Use probe as the reference.** Uniform sampling avoids PER bias, and srank avoids its ceiling when `probe.batch_size` exceeds the feature dimension. Use probe for comparisons between Runs or `replay_ratio` settings. Its difference from actual directly reflects the distribution bias presented by PER: actual dormant values above probe suggest PER is concentrating learning on states where the representation is struggling.

**The trough in `42_probe_dead_ratio` marks a turning point.** It falls during early learning before turning upward, with that turn occurring in the same window as peak performance. `61_weight_norm_feature` also stops falling and begins rising in that window; display both to check their agreement.

**`feature_norm` cannot be interpreted alone.** Since the Head has the form Q = w·φ, flat `q_max` with rising `43_probe_feature_norm` cannot distinguish shrinking w from growth of φ in directions that do not contribute to Q. Attribution becomes possible only when viewed with `61` / `62`. Declining `62` indicates scale transfer with readout shrinkage; flat `62` with rising `61` indicates backbone scale growth; both rising indicates scale growth across the Network.

With SN enabled, `61` / `62` are L2 norms of raw parameters held by the optimizer, while `63_weight_norm_feature_effective` / `64_weight_norm_readout_effective` are L2 norms after applying SN to the weights actually used in forward. Biases, normalization affine parameters, and non-SN parameters contribute equally to both, so differences come only from SN weights. `65_spectral_sigma_feature` / `66_spectral_sigma_readout` are the maximum sigma in each group; a group without corresponding SN layers reports `NaN`. Metrics 63–66 default to OFF; enable their subscription rows for experiments.

With `spectral`, weights are always normalized according to sigma, so do not equate growth in 61/62 with growth in 63/64. With `spectral_cap`, weights with sigma at most 1 remain unchanged, and the cap applies only in groups whose 65/66 exceeds 1. Sigma is a group maximum rather than a per-layer series; use it to diagnose cap activation and scale trends, not to identify an anomalous layer.

Three constraints apply to comparisons.

- **Do not compare absolute `srank_ratio` across channels.** The ratio is srank / min(N, D), but N is the Learner batch size for actual and `probe.batch_size` for probe. Compare only temporal shape: onset and rate of decline, and recovery.
- **Do not compare `dead_ratio` levels across Runs with different `probe.batch_size`.** Smaller samples make infrequently firing units appear dead, so the value itself depends on batch size. Interpret changes over time within the same configuration.
- **Weight norm depends on parameter count.** Compare only time series within one structure or Runs with identical structures.

Additional interpretation notes:

- `dead_ratio` (τ=0) is a subset of `dormant_ratio` (τ=0.025) and normally follows it. Divergence provides independent information: irreversible death rather than shallow dormancy is increasing. Dead typically has larger amplitude.
- Srank measures directional distribution, while dormant/dead measures lost units, so they are nearly orthogonal. When damage mainly consists of unit death, srank may remain flat at a fraction of `min(N, D)`. Unchanged srank does not imply health.
- Different δ variants (`x6`–`x9`) use the same singular-value vector and do not add SVD calls. Enable them only to inspect energy concentration in leading directions.
- The target channel observes the online network with a `soft_update_tau` lag and is usually redundant, hence OFF by default. Enable it when examining collapse mechanisms; the online-to-target propagation lag helps distinguish a still-healthy target pulling the network back from self-sustaining collapse involving both.
- To align with exp-axis tags, convert using exp_step = learn_step × batch size / `replay_ratio` (with batch 256: RR8=×32, RR4=×64, RR1=×256). See Sections 4.2 and 6.3 for general step-axis cautions, and Section 9.4 of [DQN Agents](200_dqn_agents.en.md) for metric definitions and measurement contracts.

### 4.8 Reading Munchausen Diagnostics

`metrics.scalar.@munchausen` adds up to seven tags under `36_agent_munchausen`. **By default, `07_soft_gap` is commented out, leaving six enabled tags.** First check enabled, mode, and Double DQN OFF in the resolved configuration, and the score source in initialization logs. `target_policy=UQE` uses risk scores from empirical quantiles; interpret these separately from configurations using mean Q.

| Tag | Interpretation |
|---|---|
| `01_scaled_logp_mean` / `02_scaled_logp_mean_ema` | Mean and EMA of the executed action's scaled log-policy before clipping. At most 0 |
| `03_clip_ratio` | Frequency of bonus lower-bound clipping. Between 0 and 1 |
| `04_bonus_mean` / `05_bonus_mean_ema` | Mean and EMA of the bonus added once to the target. Between `alpha * clip_value_min` and 0 |
| `06_next_entropy` | Next-policy entropy. Between 0 and `ln(number of actions)` |
| `07_soft_gap` (OFF by default) | Difference between soft state value and maximum mean Q. Between 0 and `entropy_tau * ln(number of actions)` for mean scores; may be negative for risk scores |

`07_soft_gap` is another functional of the same Q distribution as `06_next_entropy`. Measured correlations over Breakout 50M were r=0.979 / 0.978 in two replicates. Since the diagnostic group accounts for 15.5% of scalar rows, only `06` is output by default. **However, for D15 (`use_optimistic_target=true`) risk-biased scores, `07` is the only metric that can become negative, so uncomment it in `metrics_scalar.txt` for the optimistic-target arm.**

`04_bonus_mean` is not simply `alpha` times `01_scaled_logp_mean`. Lower-bound clipping reduces its magnitude, so **the deviation of their ratio from `alpha` measures clipping's effective impact**. In Breakout 50M, `03_clip_ratio` was only 0.6%, yet the ratio was 0.78, 13% below `alpha=0.9`. Rare but deeply negative outliers explain this; a small `03` does not mean `clip_value_min` is ineffective.

The five raw diagnostics (`01` / `03` / `04` / `06` / `07`) are computed in FP32 real space even with TBO and are collected with PER OFF too. **Only metric output for `07` is disabled by default; computation and readback still run for all five.** Known keys for disabled or unavailable values return `NaN`; do not reinterpret it as zero. Readback transfers priority and clip counts, IQN diagnostics, Munchausen diagnostics, and upper-tail statistics together in that order. Actor `actor_approx` uses existing action scores and should be treated as a different approximation from Learner's empirical-quantile approximation.

Compare overhead between modes using `forward_target`, `forward_munchausen_online`, `munchausen_target`, and elapsed-time differences over the same exp-step interval. Diagnostics or one seed's score alone do not establish improvement.

## 5. Analyzing Optuna Results

### 5.1 Metrics Viewer and Dashboard Roles

| Target | Interface | Authoritative source |
|---|---|---|
| Seed-run time series | Metrics Viewer | `<study>_<trial>_s<seed>/metrics.jsonl` |
| Multi-seed results for one trial | Optuna Dashboard / artifacts | `multiseed_summary.json` and `seed_runs.json` in the representative folder |
| Reevaluation of the same parameter group | Summary study | `group_summary.json` and mean/range/std objective |
| Runner/configuration failure investigation | Run artifacts | `manifest.json`, `process.json`, `stdout.log`, `stderr.log` |

The representative folder `<study_name>_<trial_name>` has no `metrics.jsonl` and does not appear in Metrics Viewer. Compare time series only for seed runs suffixed `_s<seed>`. A summary study is for Dashboard browsing, not a Metrics Viewer Run.

Launch Optuna Dashboard from the repository root as follows.

```powershell
apps\23_optuna_dashboard.bat dm_opt
```

Specify a workspace name or absolute path as the argument. The URL is `http://127.0.0.1:8088`, storage is `<workspace>/optuna/optuna.db`, and the artifact store is `<workspace>/optuna/artifacts`. The launcher creates none of these; it fails fast if the workspace, DB, or artifact store is missing.

### 5.2 Reading Scores

A DropMerge trial value aggregates per-seed scores. Check `score_aggregate`; do not confuse `mean`, `median`, `mean-minus-std`, and `min`.

The current primary score averages the means of these two tags within the specified window.

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

Late-window `score_60_80`, `score_80_100`, and `late_slope` are supporting indicators of growth or saturation, not the trial value itself. For final candidates, inspect per-seed scores, range/std, and matched `exp_step` time series alongside the aggregate score.

`run-study --n-jobs > 1` can launch parallel runners on the same GPU, but duration and step/sec are affected by interference. Do not directly compare that throughput with a standalone Run.

## 6. Inspecting and Extracting Runs with `inspect_run.py`

Metrics Viewer is a visualization interface for people. Use `inspect_run.py` to extract structured results from the shell, especially when requesting Run analysis from an AI agent. This read-only CLI never modifies Run artifacts and is safe to use on running Runs.

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py <subcommand> [RUN ...] [options]
```

| Subcommand | Role |
|---|---|
| `runs` | Run discovery and artifact, Metrics master, and Metrics cache status |
| `tags` | Metric tag list, definitions (step coordinate system and source key), and reached steps |
| `config` | Effective configuration extraction and differences between Runs |
| `metrics` | Scalar extraction, range aggregation, and Run comparison |
| `trace-csv` | Extracts individual trace-channel rows as CSV |

All subcommands except `trace-csv` support `--format json|md` and `--output PATH`. JSON is the default; `--output` replaces the destination atomically through a temporary file. `trace-csv` always outputs CSV and has no `--format`.

### 6.1 Finding Runs

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py runs --workspace dm-iqn
```

`runs` without arguments lists Runs across all workspaces. Passing `RUN` returns details only for that Run. `RUN` accepts a Run name or an existing relative/absolute directory path. Run-name discovery is restricted to directories directly under `apps/runner/workspaces/*/runs/`. If the same name exists in multiple workspaces, the tool lists candidate paths and exits with status 2, without implicitly selecting a workspace. Legacy layouts under `apps/runner/runs_*` remain readable when explicitly specified by directory path.

`runs` does not open the Metrics master. It returns artifact paths, sizes, and modification times; SHA-256 of `config/config_data.txt`; selected Metrics master; Metrics cache status and reasons; and lists of `*.log` and `agent_close.anet`. Presence and modification time of `agent_close.anet` offer clues about whether a Run completed or stopped early. However, Runs with `app.save_agent_on_close=false` do not create it even on normal shutdown, so absence alone does not prove abnormal termination.

### 6.2 Seeing What Is Available

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py tags run_A --format md
```

For each tag, `tags` returns resolved definitions (`step_axis`, `runner`, `scope`, `eval_name`, `eval_episodes`, `num_envs`, `event`, `target`, `source_key`, `ema_alpha`, `interval`, `clip`) and observed ranges (`count`, `min_step`, `max_step`) in JSON or Markdown. These show valid `--metric` selectors and how far each tag has progressed; choose ranges after inspecting them.

`runner` identifies the step-counter owner; `scope` / `eval_name` identify the subscription destination. Session aggregates from eval1 and eval2 can both have `runner: "train"` while remaining distinguishable by `eval_name`. `eval_episodes` is the planned number of adopted episodes per session, and `num_envs` is the lane count of the constructed eval Env; neither guarantees session completion or the number of parallel episodes. Eval information in train scope and unspecified `clip` are `null`.

Both master and cache paths retain additional definition metadata. Fields absent from historical Run definitions are unknown (`null`) and are not inferred from tags or configuration. The existing configuration-derived path used when definitions are absent, including `tags --no-observed` without a cache, restores the subscription destination and clip but leaves `eval_episodes` / `num_envs` as `null` rather than guessing post-construction conditions. Historical artifacts are not rewritten.

`source_key` is the metric key selected on the right-hand side of `metrics.scalar.[tag]`. Search the code for this string to understand the value's meaning.

`--no-observed` returns only declared definitions without observed ranges. It never opens the Metrics master, even when a Run has no usable Metrics cache, so it always returns immediately.

### 6.3 Beware of Step Coordinate Systems

The `runner` column in `tags` identifies which Runner's counter provides the tag's steps. **Even the same `exp_step` belongs to a different coordinate system when `runner` differs.**

```text
51_eval1/13_double_suika_created_mean   runner=train   max_step 19,993,856
51_eval1/41_noop_uqe_win_rate           runner=eval1   max_step    151,185
```

Both definitions in `config/config_data.txt` contain `$eval.[eval1] ... $exp_step`, but `@session_end` uses train-runner steps while `@train $action_info` uses the eval runner's own steps. Applying a training-side range to an evaluation-side tag returns an empty result without error. Their ratio changes as learning advances, so conversion is not possible either.

`inspect_run.py` resolves relative ranges (percentages, tail-relative ranges, and `common`) independently for each coordinate system, so both tags can be specified in one call and still receive appropriate ranges.

### 6.4 Extracting Metrics

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py metrics run_A run_B --metric "42_env/*" --range -4M:
```

`--metric` accepts an exact tag or glob. Only `*` and `?` are glob metacharacters; `[` and `]` are literal, so keys using `[tag]` notation, such as `51_eval1/41_noop_uqe_win_rate`, can be written directly.

**Bundle metrics into one call.** Scanning the Metrics master takes one pass per Run, independent of tag count. Separate calls for each tag multiply scan time.

`--range` specifies two endpoints. Both are inclusive, so boundary points in overlapping ranges count in both.

| Form | Meaning |
|---|---|
| `10M:20M` | Absolute steps; supports `K`/`M`/`G` suffixes |
| `10%:20%` | Percentages of the maximum observed step in that coordinate system |
| `:20M` | Omitted lower endpoint (from 0) |
| `10M:` | Omitted upper endpoint (through the maximum observed step) |
| `-4M:` | Last 4M steps |
| `-10%:` | Last 10 percent (equivalent to `90%:100%`) |

`--range-mode common` uses the intersection of observed ranges across all Runs with the same coordinate system. This is the standard way to compare Runs with different reached steps on equal terms, automating the matched window in [4.3](#43-compare-matched-exp_step-ranges).

```text
run_A: 0 - 20,000,000 exp_step
run_B: 0 - 16,200,000 exp_step

--range-mode common  ->  両方とも 0:16,200,000
--range -4M:         ->  A は 16.0M:20.0M、B は 12.2M:16.2M（幅は同じ4M）
```

`common` means examining the same location; `-4M:` means examining the present over the same width. Choose according to the purpose.

Each Run × tag × range returns `count`, `mean`, `population_std`, `min`/`max`, `first`/`last`, and step range. The tag's full observed range before filtering is also reported, making coordinate-system mismatches apparent.

### 6.5 Comparing Runs

For each range, `metrics` returns a comparison table with tags as rows and Runs as columns. Cell statistics default to `mean`; select another with options such as `--stat last`.

- Two Runs add `delta` and `delta_ratio`.
- Three or more Runs add `mean`, `population_std`, and `range`. Passing repeated Runs with identical settings provides an immediate measure of variability.

Markdown includes both comparison and detail tables. Add `--series` only when curve shape is needed; it adds a `step:value` series thinned to at most 128 points. Thinning is deterministic and identical between Metrics cache and Metrics master paths.

### 6.6 Checking Effective Configuration

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py config run_A run_B --diff
```

`config` reads `config/config_data.txt`. This file contains effective values alongside definitions of override layers (`A1`–`A3`, `E1`, `M1`/`M2`, `P1`) and ordinary selection-source prefixes (`app.online`, `app.batchrun`, etc.). It excludes `@` profiles and `.$` selection lines, so **the selected profiles cannot be reconstructed from this file alone.** Inspect selection records with `resolution` in Section 6.7. See Chapter 3 of the [Run Execution User Guide](020_user_guide_run.en.md) for syntax and precedence.

Each key therefore includes `effective`: `true` if the same key occurs in `config/<module>.txt`, or `null` (unknown) otherwise. Some areas, such as `net.*`, have no module dump, so unconfirmed keys are not marked `false`. `--effective-only` selects only keys marked `true`.

```text
--config-key "*replay_batch_size*"                  9件（定義namespaceを含む）
--config-key "*replay_batch_size*" --effective-only  1件（DefaultDQNAgent.learner.replay_batch_size）
```

`--diff` returns only keys whose values or presence differ between Runs. Missing keys use `present: false`, distinct from a `null` value. Use it to confirm that only seeds differ or that repeated Runs truly have identical settings.

### 6.7 Checking Configuration Resolution

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py resolution run_A run_B --format md
```

`resolution` displays selection records and `${}` value references per Run. If the first selection is `run.$`, it summarizes the Run profile selection first, then displays each selection's `key` / `term` / `resolved` and each reference's `source` / `target` / `value`. Runs without Run profiles omit that summary.

Reading prefers the `type=json` / `tag=config_resolution` envelope in `json/config_resolution.json`; if absent, it reads the bare payload in `config/config_resolution.json` from transitional Runs. Pre-PH0 Runs with neither return `status: missing` successfully. Unknown resolution `schema_version` values are displayed best-effort with a warning. A corrupt preferred source returns `status: source_error` and exit status 1 rather than silently falling back to a lower-priority source.

This subcommand reads only mirrors and never opens `metrics.jsonl`, `metrics.jsonl.gz`, or `metrics_cache.db`. With multiple Runs, one Run's missing or corrupt resolution does not prevent displaying the others.

### 6.8 Relationship to the Metrics Cache

`metrics_cache.db` is used read-only only when fully current; otherwise the tool automatically falls back to the Metrics master. The decision and reason (`current` / `absent` / `invalid` / `partial` / `stale` / `error`) appear in `runs` and `metrics` results. The tool never creates, updates, repairs, or deletes caches.

After compressing `metrics.jsonl`, a Run's cache may remain `stale: source_kind_changed` or `partial` until its workspace is opened in Metrics Viewer. Results remain correct, but require scanning every Metrics master row. Opening it once in Viewer enables the fast path.

Running Runs are readable too. Raw files are read only up to their size at the start of the command, excluding an unterminated final line. If the master changes while reading, `provisional` and `source_changed_during_read` are set.

### 6.9 Extracting Individual Trace Rows as CSV

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py trace-csv run_A --tag "51_eval1/*"
```

`metrics` handles scalars: statistics after aggregation. One evaluation session is collapsed into one scalar point, so quantiles, threshold-exceedance rates, and the joint score × length distribution cannot be reconstructed. `trace-csv` exports trace-channel rows, one adopted episode per row, for further aggregation in Excel or pandas.

The output is one CSV with header `row_no,run,tag,step,lane,<key...>`. Each row is one episode, with no aggregation. Multiple rows sharing `(tag, step)` are normal and represent the lane-level breakdown of an evaluation session. `lane` is the event's `env_index`, or `-1` for SHARED configurations.

| Target | Rule |
|---|---|
| Tag selection | Without `--tag`, all declared trace tags. `--tag` accepts globs (`*` and `?` only) and can be repeated |
| `row_no` | One-based sequence per output. Restores original order after spreadsheet sorting; not a record identity. Changing `--tag` can assign the same row a different number |
| Column order | `keys` in `metrics.trace.defs` (configuration declaration order) is authoritative. Do not use JSONL `data` key order, which is alphabetical rather than declaration order |
| Multiple Runs/tags | Combined into one CSV and distinguished by `run` and `tag`. Columns form a union in first-seen order; undeclared keys are empty cells |
| Empty cells | Either a `null` value (NaN / ±Inf) or a column belonging to another tag. Numbers retain their JSON values without rounding |
| Destination | stdout by default. `--output PATH` writes one file; `--output-dir DIR` splits by tag, replacing `/` in tag names with `_` rather than creating subdirectories. Each split file has contiguous `row_no` starting at 1 |

`--output` and `--output-dir` are mutually exclusive. `--output` requires an existing parent directory; `--output-dir` creates its directory if absent. Both stream CSV into files and replace destinations atomically on completion.

The read path matches other subcommands: use `json_lines` when the Metrics cache is current; otherwise scan the Metrics master once and emit a warning. If a Run's definition record is unreadable, column order cannot be established, so the tool reads all rows before outputting a header in sorted observed-key order, with a warning.

### 6.10 Exit Status

| Value | Meaning |
|---|---|
| `0` | Success. Tags/keys missing from only some Runs appear as `missing` in results |
| `1` | Source read/query failure, or a metric/config selector matches nothing across all Runs (for `trace-csv`, no trace tags across all Runs) |
| `2` | Argument/range syntax error, missing or ambiguous Run, missing parent directory for `--output`, or combined `--output` and `--output-dir` / filename collisions in `trace-csv` |

## 7. Minimal Analysis Checklist

1. Saved `config/config_data.txt` for the compared Runs.
2. Confirmed matching tags, step axes, and eval definitions.
3. Selected a common `exp_step` window.
4. Interpreted scores and throughput separately.
5. Aligned display conditions such as EMA, interval, and signed-log.
6. Avoided a final decision based on only one seed.
7. For Optuna, checked aggregation method, per-seed distribution, and trial state.

## 8. Related Documents

- [Run Execution User Guide](020_user_guide_run.en.md)
- [Development Environment](040_development_environment.en.md)
- [Observability](140_observability.en.md)
- [Applications and Tools](160_applications_and_tools.en.md)
- [DropMerge Optuna Guide](optuna.md)
