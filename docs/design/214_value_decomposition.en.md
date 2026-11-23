<!-- translated-from: 214_value_decomposition.jp.md blob:1a7e94b88f52ef95c71314f14c60c39d28f82da3 date:2026-11-21 progress:done -->
# Extrinsic and Intrinsic Value Decomposition

> Main perspective: DefaultDQNAgent reward components, heads, action selection, learning, diagnostics, and persistence.

## 1. Introduction

### 1.1 Purpose and scope

DefaultDQNAgent supports `combined`, which learns one Q from the sum of extrinsic and intrinsic rewards, and `decomposed`, which learns them with separate heads. The default is `combined`. TD, QR, and IQN are supported. See [RND](212_rnd_lifelong_novelty.en.md) for the intrinsic reward estimator, [DQN Agents](200_dqn_agents.en.md) for shared action selection and learning, and [ReplayBuffer](150_replay_buffer.en.md) for component storage and sampling.

### 1.2 Responsibility boundaries

Replay handles component counts, rewards, discounts, and returns without knowing extrinsic or intrinsic semantics, β, or terminal masking. DefaultDQNAgent treats component 0 as extrinsic and component 1 as intrinsic. The network heads produce their respective values; DQN owns β combination and TBO handling.

## 2. Structure and reward flow

### 2.1 Heads

In `decomposed` mode, one existing TD, QR, or IQN head is built for each component. They share the body and forward executor but have separate head parameters. Dueling, initialization, and ordinary or Noisy projections can be selected per head. Both use the same distribution representation.

| Output | Extrinsic | Intrinsic |
|---|---|---|
| Action value | `extrinsic_q` | `intrinsic_q` |
| Quantiles | `extrinsic_q_dist` | `intrinsic_q_dist` |
| Dueling components | `extrinsic_v`, `extrinsic_a`, etc. | `intrinsic_v`, `intrinsic_a`, etc. |

The network itself does not return `q` or `q_dist` in decomposed mode. DQN's combination operation adds them using the β for that forward. Policies, existing Q metrics, and diagnostic consumers read the combined `q` and `q_dist`.

The intrinsic head's `input_mode=shared` reads the same body outputs as the extrinsic head: normally `features`, or `value_feature` and `adv_feature` when dedicated Dueling outputs are used. `dedicated` reads the corresponding `intrinsic_features`, or `intrinsic_value_feature` and `intrinsic_adv_feature`. Publish them through the body's `net.body.output`. IQN requires rank-3 features fused with τ. Missing keys and invalid shapes are rejected during construction.

### 2.2 Reward components

Extrinsic rewards pass through the existing extrinsic reward scaler. Intrinsic rewards subtract 1 from the RND multiplier, apply done handling through `intrinsic.use_done_mask`, and then pass through the intrinsic reward scaler.

| Mode | Rewards passed to Replay | Replay component count |
|---|---|---|
| `combined` | Extrinsic + β × intrinsic | 1 |
| `decomposed` | Extrinsic and intrinsic without β | 2 |

In decomposed mode, β is absent from stored Replay values. The learning β evaluated at the intake boundary is passed to the current Learner updates. Actors evaluate their action β at their own step. Zeroing a reward on done and cutting intrinsic value bootstrap at a terminal are independent settings.

### 2.3 Combination and TBO

In real space, `Q = Q_ext + β Q_int`. QR and IQN add quantiles at the same τ. With TBO, apply inverse h to each head output, combine in real space, and apply h again. Do not directly add transformed values.

## 3. Learning

### 3.1 Targets

For extrinsic and intrinsic n-step returns `R_ext` and `R_int`, and actual step count n, the targets are:

```text
y_ext = R_ext + gamma_ext^n (1 - done) Q_ext,target(s_next, a_ext)
y_int = R_int + gamma_int^n (1 - done_int) Q_int,target(s_next, a_combined)
```

Double DQN selects actions online and evaluates values with the target network. Otherwise, actions are also selected with the target network. Intrinsic actions use the combined Q. Extrinsic actions use the same action for `extrinsic_bootstrap_mode=combined`, or the extrinsic Q for `extrinsic`.

`discount_mode=shared` uses the extrinsic discount. `separate` uses an explicitly specified intrinsic discount, including schedules. Constant components store their returns at Push; only scheduled components recompute returns from stored step rewards after sampling.

With `terminal_mode=episodic`, done also cuts intrinsic bootstrap. `non_episodic` bootstraps only the intrinsic component from the first observation after termination. If done and truncated are both true, both components cut bootstrap. Truncated alone allows both to bootstrap. TBO is applied to the completed target in real space.

### 3.2 Munchausen

Only the extrinsic head receives the Munchausen and entropy terms. The intrinsic head uses only the expected intrinsic Q at the next state.

```text
y_ext = R_ext + bonus + gamma_ext^n (1 - done) sum(pi_ext * (Q_ext,target - tau log pi_ext))
y_int = R_int + gamma_int^n (1 - done_int) sum(pi_combined * Q_int,target)
```

`pi_combined` comes from the combined Q. The extrinsic `pi_ext` and bonus come from the extrinsic Q when the extrinsic bootstrap mode is `extrinsic`, and from the combined Q otherwise. The usual DQN rules for target, online, and current-output reuse under `log_policy_mode` apply. Add the bonus once to the first transition without terminal masking.

With shared discounts, episodic masking, and combined bootstrap, `y_ext + β y_int` matches the target of a single Q trained on combined rewards. With TBO, this holds in real space before h.

### 3.3 Loss, priorities, and optimization

The loss is `L_ext + loss_weight * L_int`; the PER error is `abs(delta_ext) + priority_lambda * abs(delta_int)`. Compute the existing TD, QR, or IQN loss and error separately for each head, then apply PER bounds once after summation. Huber thresholds, quantile κ, and TD error clipping are shared. The loss weight must be positive; the priority weight must be nonnegative.

`feature_gradient=merge` sums the losses, runs backward once, clips the combined gradient, and updates one Adam state. `split` obtains extrinsic and weighted intrinsic gradients separately from the same forward graph, clips each loss's gradients, and adds updates from independent Adam states. Shared parameters receive weight decay once on the extrinsic side; each head receives it once in its own update. Since Adam normalizes step sizes, the loss weight has less effect in split mode.

## 4. Configuration

### 4.1 Decomposition and actions

| Key under DefaultDQNAgent | Values and default | Conditions |
|---|---|---|
| `intrinsic.value_mode` | `combined`, `decomposed`; default `combined` | Decomposition requires intrinsic enabled and skip disabled |
| `learner.intrinsic.discount_mode` | `shared`, `separate`; default `shared` | Separate requires explicit gamma |
| `learner.intrinsic.gamma` | Greater than 0 and less than 1; no default | Constant or schedule |
| `learner.intrinsic.terminal_mode` | `episodic`, `non_episodic`; default `episodic` | Intrinsic bootstrap |
| `learner.intrinsic.loss_weight` | Default 1, finite and positive | Loss weight |
| `learner.intrinsic.priority_lambda` | Default 1, finite and nonnegative | Priority weight |
| `learner.intrinsic.extrinsic_bootstrap_mode` | `combined`, `extrinsic`; default `combined` | Extrinsic bootstrap policy |
| `learner.intrinsic.feature_gradient` | `merge`, `split`; default `merge` | Optimization separation |
| `intrinsic.value.head.input_mode` | `shared`, `dedicated`; default `shared` | Body output selection |
| `intrinsic.value.head.use_dueling` | Default true | Intrinsic head Dueling |
| `intrinsic.value.head.init.*`, `projection.*` | Same rules as ordinary heads | Intrinsic head initialization and projection |
| `actor.[name].intrinsic.action_beta_mode` | `learner`, `fixed`; default `learner` | Action β |
| `actor.[name].intrinsic.action_beta` | Finite and nonnegative; no default | Fixed requires an explicit value |

Intrinsic gamma in shared mode, action_beta in learner mode, and decomposition-only values in combined mode are unused. Recognized values receive type and range validation; shared configuration policies govern inactivity and unused-value diagnostics. Invalid values or configurations that request decomposition but cannot construct the heads are not replaced with combined mode.

### 4.2 Reward scalers

`reward_scaler.dynamic_mode` and `intrinsic.reward_scaler.dynamic_mode` accept `reward_std` (default) or `return_std`. They take effect with `use_dynamic_scaling=true`. Intrinsic defaults are an identity transform: dynamic scaling, automatic scaling, and clipping disabled; constant and manual scales set to 1.

`return_std` maintains `R = gamma * R + reward` per lane and divides the current reward by the standard deviation of R. It does not zero R at episode boundaries; reset zeros every lane. The statistics component, epsilon, manual scaling, and clipping are shared with reward_std. Extrinsic scaling uses extrinsic gamma; intrinsic scaling uses the effective intrinsic gamma at step 0 in decomposed mode. A gamma schedule does not change R's discount during execution.

Combining return_std with `use_auto_post_scale=true` is rejected. Trace uses a transform that does not update statistics. Intrinsic scaling is also available in combined mode and is separate from error normalization inside the RND estimator.

## 5. Diagnostics

### 5.1 Per-head values and learning

| Source | Keys | Meaning |
|---|---|---|
| `$agent_step` | `extrinsic_q_sa`, `intrinsic_q_sa` | Selected-action Q per head |
| `$agent_step` | `extrinsic_state_value`, `intrinsic_state_value` | State value per head |
| `$agent` | `extrinsic_loss`, `intrinsic_loss` | Each loss before weighting |
| `$agent` | `extrinsic_q_sa_mean`, `intrinsic_q_sa_mean` | Mean selected-action Q per head |
| `$agent` | `extrinsic_head_grad_norm`, `intrinsic_head_grad_norm` | Gradient norm of each head's final projections before clipping |
| `$agent` | `extrinsic_grad_norm`, `intrinsic_grad_norm` | Per-loss gradient norms before clipping in split mode |
| `$agent` | `intrinsic_q_share` | Intrinsic share of absolute selected-action values |
| `$agent` | `intrinsic.reward_scaler.scale` | Intrinsic reward scaling factor |

`intrinsic_q_share` is the sample mean of `abs(β Q_int) / (abs(Q_ext) + abs(β Q_int))`, using 0 when both are zero. Split-only norms are NaN in merge mode; the other per-head metrics and four Trace keys are NaN in combined mode. The intrinsic scaler factor is NaN when intrinsic rewards are disabled. Values that have not yet become available also return NaN. Existing q_sa, state_value, and learning Q metrics use combined values.

### 5.2 Effect on action selection

Actors compute four values from each head's mean Q in real space and that Actor's β. Distributional outputs invert each quantile before averaging. These values do not depend on UQE or Thompson scores, ε exploration, or forced actions.

| Lane key | Lane-mean key | Evaluation session key | Value |
|---|---|---|---|
| `extrinsic_q_argmax_disagreement` | `extrinsic_q_argmax_disagreement_rate` | `extrinsic_q_session_argmax_disagreement_rate` | 1 if combined and extrinsic Q select different best actions |
| `extrinsic_q_regret` | `extrinsic_q_regret_mean` | `extrinsic_q_session_regret_mean` | Extrinsic Q relinquished by changing the action |
| `extrinsic_q_gap` | `extrinsic_q_gap_mean` | `extrinsic_q_session_gap_mean` | Difference between the best and second-best extrinsic Q |
| `intrinsic_q_gap` | `intrinsic_q_gap_mean` | `intrinsic_q_session_gap_mean` | Difference between the best and second-best β-weighted intrinsic Q |

When decomposition is enabled, compute them every step regardless of subscriptions. Sessions count lane × step and reset accumulation on synchronization. Empty sessions and combined mode return NaN. With one action, disagreement and regret are 0 and both gaps are NaN. With β=0, disagreement, regret, and intrinsic gap are 0 (the one-action gap remains NaN). Lane keys reject out-of-range lanes.

Read lane means through `@train $action_info`, and evaluation accumulation through `$eval.[eval] $actor @session_end` and `$eval.[eval_target] $actor @session_end`. Add the optional `metrics.scalar.@value_decomposition` group to the selection chain. It is not added to default Run profiles or default Evaluation Trace columns.

### 5.3 Actor Q hints

Hints always contain five float32 columns: extrinsic Q_sa, extrinsic V_s, the Munchausen term, intrinsic Q_sa, and intrinsic V_s. In combined mode, the first two retain the previous Q_sa and V_s semantics and the last two are zero. Other column counts are rejected. Replacing an action regathers extrinsic Q_sa, intrinsic Q_sa, and the Munchausen term while retaining state values. No extra forward is performed.

Initial priorities estimate each head's error from component returns, discounts, and hints, then apply bounds once to the weighted sum of absolute errors, matching post-learning PER.

Even at a non_episodic terminal, the intrinsic initial-priority estimate omits bootstrap because a post-terminal hint is not yet available.

## 6. Saving and restoring

Network persistence includes both heads. Split mode saves two Adam states. A value_mode mismatch between saved and current configurations, or a feature_gradient mismatch in decomposed mode, is rejected. Both extrinsic and intrinsic reward scalers save their type, dynamic statistics, and per-lane R for return_std.

Old constant-scale combined checkpoints remain readable. Old checkpoints without dynamic state are treated as constant scalers and rejected as a type mismatch when loaded into dynamic scalers. Other saved/current scaler type mismatches are also rejected. If return_std lane counts differ, retain the statistics, zero R, and WARN once with the target and saved/current lane counts. Observation-normalization persistence is outside this feature's scope.

## 7. Tests and extension checks

See [testdata/prd121](../../src/core/base/testdata/rfd0121/README.md) for acceptance procedures and rerun entry points, and the [implementation memo](../memo/0121_extrinsic_intrinsic_value_decomposition_20impl.md) for the plan and validation record.

When adding a policy or diagnostic forward, apply combination to head outputs before reading q/q_dist. When adding reward components to Replay, check single-component shapes and operation order, constant-return storage, scheduled-component-only recomputation, initial priorities, and discount lifetimes under prefetch. Window rules with multiple components are unsupported and rejected during construction.

Intrinsic head weights also belong to the network's readout side. When extending plasticity, Noisy, policy churn, or Replay fit diagnostics, distinguish paths that consume combined Q from paths that inspect each head's parameters.

## 8. Related Documents

- [ADR on Replay reward components and discount factors](../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md) (§2.2 and §3.1)
- [ADR on placement of the intrinsic value Head](../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md) (§2.1)
