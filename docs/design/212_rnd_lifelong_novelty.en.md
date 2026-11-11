<!-- translated-from: 212_rnd_lifelong_novelty.jp.md blob:09b8abdd865ab3cd46e72d8f43178ccace0b446d date:2026-11-11 progress:done -->
# Novelty Estimation and Intrinsic Reward with RND

> Primary perspective: connect basic concepts and algorithmic background to ANET's scoring, reward composition, auxiliary learning, and persistence.
>
> Scope: standalone RND intrinsic reward in `DefaultDQNAgent`.

RND **uses a small model that learns the answers to a fixed problem, giving an exploration bonus to observations whose answers it cannot yet reproduce well**. It adds a learning signal separate from the environment's score, encouraging visits to places not yet sufficiently experienced.

This document starts from the basics for readers unfamiliar with reinforcement learning or RND. See §2 for concepts, §3 for the connection between calculations and implementation, and §4 for configuration and metrics. The overall DQN specification is in [200 DQN Agents](200_dqn_agents.en.md).

## 1. Terminology

### 1.1 Reinforcement Learning Basics

| Term | Meaning |
| --- | --- |
| Reinforcement learning | Learning how to choose actions that earn more future reward, using actions and the rewards they produce. |
| Environment (Env) | The system in which actions execute, such as a game or maze. It produces observations and rewards in response to actions. |
| Observation / next observation | Information the Agent receives, such as an image or position. A next observation is obtained after executing an action. It need not expose the entire internal environment state. |
| Agent / policy | An Agent combines action selection and learning. A policy is a rule for selecting actions from observations. |
| Extrinsic reward | Scores or success/failure evaluations supplied by the environment, representing the task ultimately to be achieved. |
| Intrinsic reward / bonus | Evaluation computed by the Agent to encourage exploration and added to training reward. Nonnegative in the default clipped mode; unclipped mode can also produce negative values. |
| Training reward | Reward passed to the learner. Here it is scaled extrinsic reward plus the bonus. |
| Exploration / exploitation | Exploration investigates unknown outcomes; exploitation chooses actions already known to be good. |
| Episode | One continuous trial from environment start to end. It may also be truncated, for example by a time limit. |
| Q value / DQN | A Q value estimates future reward after choosing an action in a state. DQN (Deep Q-Network) learns this estimate with a neural network. |

### 1.2 RND and Computation

| Term | Meaning |
| --- | --- |
| Neural network / weights | A model mapping inputs to numerical outputs, and the numbers defining that mapping. Learning adjusts the weights. |
| Random / seed | Random values use a random-number generator. A seed is its initial value for reproducing a sequence. |
| Distillation | Training one model to reproduce another's outputs. In RND, the model being reproduced is randomly constructed. |
| RND | Random Network Distillation: predict a fixed random model's output and estimate observation novelty from the error. |
| RND target | The fixed teacher: a model mapping observations to numerical vectors that the predictor should reproduce. |
| RND predictor | A model trained to reproduce the RND target's output from the same input. |
| Feature vector | A numerical representation of an observation. Individual RND target outputs need not have human-readable meanings. |
| Prediction error / MSE | The difference between prediction and teacher output. MSE (mean squared error) squares the differences and averages over elements. |
| Loss / auxiliary learning | Loss is the quantity minimized by learning. Auxiliary learning here reduces predictor error separately from the main action-value learning. |
| Mean / variance / standard deviation | The mean describes the center. Variance is the mean of squared deviations from the mean, and standard deviation is its square root. The same difference is relatively larger when the usual spread is smaller. whiten divides by the standard deviation and variance_feedback by the variance, so their multipliers differ. |
| Observation normalization | Aligning input scales. The current default subtracts each component's mean and divides using its standard deviation. |
| Clip / multiplier α | Clipping limits values to lower and upper bounds. Multiplier α is a score with baseline 1: the default clipped mode keeps it in `[1, L]`, while unclipped mode leaves `1+z` unchanged. |
| β / L | β controls bonus strength. L is the upper bound of multiplier α. |
| Warm-up | An initial period without a bonus. Default whiten trains the predictor and collects observation statistics, excluding errors. |
| NGU | Never Give Up: a method combining episodic and lifelong novelty and learning policies with different exploration strengths. |

### 1.3 ANET Implementation Terms

| Term | Meaning |
| --- | --- |
| Actor | The component selecting actions from observations. It does not call RND directly here. |
| Facade | The outer interface coordinating internal components. Here `DefaultDQNAgent::UpdateFromBatch` preprocesses rewards. |
| DQN Learner | The inner learner for Q values. It uses rewards composed by the facade. |
| ReplayBuffer | Stores experience for later learning. Experience includes observations, actions, rewards, and next observations. |
| Batch / lane | A batch is a set of observations processed together; a lane is one environment position within it. B is the lane count. |
| Intake / N / k / credit | An intake is the experience set received by the facade in one call. N is its lane count; k is the number supplied to one auxiliary-learning step. Credit accumulates fractional learning steps, with each unit allowing one step. |
| Tensor / shape / TensorDict | A tensor is a multidimensional numerical array; shape gives its axis lengths. TensorDict maps names such as `vector` and `grid` to tensors. |
| Body / Head | The Network's feature-extraction stage and final output stage. RND's Body is configurable; its Head linearly maps features to the specified dimension. |
| Network / NetworkModel | ANET's `Network` is a computational model with a Body and Head. DQN's `NetworkModel` groups Q-learning models and synchronization; it does not own RND. |
| Conv2d / MLP / fully connected layer / ReLU / Flatten | Conv2d handles local image patterns; an MLP stacks fully connected layers. Fully connected layers combine input components, ReLU sets negative values to zero, and Flatten merges axes within an observation. |
| Frame stacking | Combining observations from consecutive times into one model input, exposing motion and other information difficult to infer from one frame. |
| Forward / backward / gradient | Forward computes outputs from inputs; backward computes per-weight gradients from loss. A gradient describes how changing a weight changes loss. |
| Optimizer / AdamW | An optimizer updates weights using gradients. AdamW adjusts update magnitudes using past gradient statistics and also handles weight decay. |
| Computation graph | The connected operations performed during forward, retained for gradient computation during backward. |
| CPU / GPU / CUDA / FP32 | CPUs perform general computation; GPUs process many operations in parallel. CUDA is the NVIDIA GPU execution platform; FP32 is 32-bit floating point. |
| libtorch | PyTorch's C++ library. ANET uses it for tensor computation, Network training, and persistence. PyTorch provides these machine-learning operations. |
| GridMaze / LunarLander / Atari / Kung-Fu Master | GridMaze is a grid maze; LunarLander is a landing-control environment. Atari denotes the console and its games; Kung-Fu Master is the game used in this validation. |
| Checkpoint | Saved models and learning state for resuming training, beyond weights alone. |
| NaN / nullopt | NaN represents a value that is currently unavailable. `nullopt` means the requested metric itself is not provided. |
| Run / metrics | A Run is one experiment execution; metrics are measurements recorded during it. |
| ProfiledValue / exp_step | ProfiledValue specifies a constant or time-varying setting. `exp_step` counts experience collected for learning and is used here to evaluate β. |
| done / truncated | `done` marks termination; `truncated` marks truncation, for example by a time limit. This document refers to the values reaching the facade. |

## 2. Background: Why Reward Novelty?

### 2.1 Providing Direction Before Earning Points

Consider a maze that rewards only reaching the goal. Before the first success, both revisiting a dead end and entering a new passage give extrinsic reward 0. Reward-only learning has little evidence to distinguish them.

A small bonus for rarely experienced observations can help. If entering a new passage gives higher training reward, Q-value learning may make actions toward that passage more likely. RND approximates “not yet sufficiently experienced” through prediction difficulty.

This extra reward does not change the environment's goal. However, **it changes the reward optimized by the learner**. Exploration can reduce extrinsic scores, so check bonus production and improvement on the intended task separately.

### 2.2 Learning Fixed Answers

RND initializes the teacher's weights randomly and never updates them. The predictor takes an observation and reproduces the teacher's output. Errors should fall on sufficiently trained similar inputs and remain larger on inputs outside that training, serving as a novelty signal. This structure comes from [the RND paper, §2.2](https://arxiv.org/pdf/1810.12894).

In the maze example, the teacher defines exercises mapping each image to a numerical vector. The predictor practices on visited images. Progress is measurable without the Agent supplying environment-specific ground-truth labels.

If every teacher answer were the same constant, the predictor could simply output it, making it hard to distinguish practiced observations. A randomly initialized Network provides input-dependent answers. The teacher need not know game strategy; the predictor needs to learn the observation-dependent mapping.

What remains fixed is **the teacher's transformation of the same numerical input**. It does not generate a new random answer each time. Changing answers would leave persistent errors despite repeated practice, making insufficient experience hard to identify.

The predictor also does not predict the next screen. It already receives the observation obtained after the action and predicts the teacher's output for that observation. This differs from predicting future environmental changes.

### 2.3 Novelty Is Not a Visit Count

The implementation measures current predictor error, not a table of visit counts. Interpret it accordingly:

- **A first visit can score low.** Training on similar observations may already enable prediction.
- **A revisit can score high.** Training may be insufficient, or learning other observations may have changed the prediction.
- **Visual change need not help the task.** Numerically novel information need not lead toward the goal.
- **The same raw observation's error can change even with a fixed teacher.** Input-normalization statistics change alongside predictor weights.

Thus, high error means “hard for the current RND to predict,” not necessarily “unvisited,” “important,” or “a place that improves scores.”

### 2.4 Lifelong and Episodic Novelty Forget Different Things

Here, lifelong novelty refers to **learning history over the lifetime of one estimator**. Predictor weights and cumulative statistics are not reset at episode boundaries. All lanes of the same Agent share one estimator, so learning in one lane affects novelty in others. Restoring a checkpoint continues this history.

Episodic novelty can instead treat a location as not yet visited when a new trial begins. This difference matters when familiar passages must be crossed to reach unexplored regions.

NGU multiplies episodic reward by a lifelong-novelty factor. ANET does not currently implement that complete method. It borrows the multiplier form but adds `β × (α − 1)` to enable standalone RND comparisons. [NGU paper, §2, Equation (1)](https://arxiv.org/pdf/2002.06038)

| Aspect | Current ANET choice |
| --- | --- |
| Core RND mechanism | Error between a fixed teacher and a trainable predictor |
| Making error scale manageable | Default uses the historical mean and standard deviation; `variance_feedback` is selectable for comparison with Taiga et al. |
| Multiplier | Default `clipped` follows NGU's `[1, L]`; `unclipped` returns `1+z` unchanged |
| Training-reward composition | Add `β × (α − 1)` to scaled extrinsic reward |
| Episodic visit memory / a family of policies with different exploration strengths | Not included in this implementation |

## 3. Mechanism: Connecting Formulas and Code

### 3.1 Feeding the Same Input to Two Networks

First normalize the observation selected by the facade specifically for RND, then pass the same input to target and predictor. The default `intrinsic.observation=next_state` selects the transition's next observation; `state` selects the observation at which the action was taken. Both are the latest single frame before the Actor's frame stacking.

```mermaid
graph TD
    O[選んだ生の観測] --> N[観測正規化]
    N --> T[RND target]
    N --> P[RND predictor]
    T --> E[予測誤差]
    P --> E
    E --> A[倍率 α]
```

For one lane, let x be its normalized observation, f the teacher, g the predictor, and d the output dimension. The code uses this error:

$$
e(x)=\frac{1}{d}\sum_{j=1}^{d}\left(g(x)_j-f(x)_j\right)^2
$$

`j` indexes output-vector components; d defaults to 128 in the implementation and is 512 in Atari's `run.@rnd`. Squaring and averaging component differences produces one nonnegative value per lane.

| Location | Shape | Meaning |
| --- | --- | --- |
| Observation | `[B, ...]` | B observations; trailing axes depend on images, vectors, and other inputs |
| Body output `features` | `[B, F]` | Each observation converted into F features |
| Target / predictor output | `[B, d]` | Answers and predictions for each observation |
| Error after `mean(-1)`, multiplier | `[B]` | One value per lane |

`RndHeadFactory` checks during construction's trial calculation that Body output has two axes. An extra axis such as `[B, T, F]` would leave `[B, T]` after averaging the last axis, which cannot be used as per-lane reward.

The entry point is [`RndNoveltyEstimator::Estimate`](../../core/anet-core/src/intrinsic_rnd.cpp). Target and predictor have separate weights. Target gradient computation is disabled, and it uses a mode without training-state updates. Neither shares weights with the Q-value Network.

Target and predictor are built separately from the required `intrinsic.rnd.target.net.*` and `intrinsic.rnd.predictor.net.*`. Their Body depth and feature counts may differ, but both project to the shared `output_dim` dimensions through a Head. Head initialization is selected by `intrinsic.rnd.head.init.*`, with xavier as the implementation default. Initialization uses different named seeds, and teacher weights are not copied into the predictor: identical initial mappings would match before practice and eliminate error as a novelty signal.

### 3.2 Two Distinct Normalizations

RND maintains two statistics with different purposes.

| Statistics | Subject | Purpose |
| --- | --- | --- |
| Observation statistics | Each input component | Prevent coordinate, velocity, pixel, and other numerical scales from directly dominating prediction difficulty |
| Error statistics | Per-lane MSE | Interpret model- and training-dependent errors relative to their usual values |

Default observation normalization approximately computes `(value − mean) / standard deviation` per component, clipping to `[-5, 5]`. This adjusts component scales without removing intercomponent correlations. Statistics belong to RND, so changing Q-input normalization does not modify shared RND statistics.

The actual divisor is `sqrt(var) + epsilon` in [`RunningMeanStd::GetStd`](../../core/anet-core/src/scaler_impl.hpp). RND observation normalization also defaults to the common `epsilon=1.0`. Do not substitute a textbook formula assuming a tiny epsilon. Initial mean and variance are 0 and 1.

Noncategorical `uint8` observations, such as pixels, are converted to floating point inside RND before normalization. Categorical observations follow existing observation-specification and normalization handling.

Error statistics, by contrast, aggregate all lanes into one. `error_norm.average_mode=cumulative` (default) accumulates every batch entered into the statistics, using the existing Welford computation with `epsilon=0`. `ema` reduces the weight of the past by a factor `d=1−1/ema_steps` for each scored observation, using the mean and population variance divided by the weight sum (default `ema_steps=1,000,000`). Updates follow lane order, so the same sequence of errors yields the same statistics regardless of how it is split into batches. `ema_steps=1` keeps only the last observation, so variance is zero. Zero standard deviation or empty statistics receive the special handling in the next section. The mode in §3.5 determines whether errors during warm-up are included.

Everything up to this point is the default `statistics_mode=whiten`. `variance_feedback`, which aligns conditions with the public implementation of Taiga et al., uses a separate recurrence. The observation mean `obs_m` starts at 0, the variance-like value `obs_v` at 1, the error mean `err_m` at 0, the error variance-like value `err_v` at 1, and the count `n` at 0. Observation mean and variance are kept per input component; error mean and variance are scalars.

```text
各laneについて、直前までの観測統計を使う:
  n += 1
  p = clip((x - obs_m) / obs_v, -clip_range, clip_range)
  e = mean((predictor(p) - target(p))^2)
  d = p - obs_m
  obs_m += d / n
  obs_v = (obs_v * n + d^2 * n / (n + 1)) / (n + 1)
  d = e - err_m
  err_m += d / n
  err_v = (err_v * n + d^2 * n / (n + 1)) / (n + 1)
  z = (e - err_m) / err_v
```

This method feeds the normalized `p` back into the observation statistics and also standardizes errors with the updated values. It takes no square root of the divisor, so do not read it as the same formula as whiten. Statistics are updated in lane order even with two or more lanes. Because observation-statistics and error-statistics updates are independent, Network forwards can be batched after preparing each lane's normalized input. `Learn` renormalizes with the updated observation statistics and does not update these statistics.

The only normalization key used by `variance_feedback` is `obs_norm.clip_range`. Other `obs_norm.*` values are still range-checked, and when a resolved value differs from RND's internal default, construction warns once per key that it is unused. Handling does not depend on whether the value came from defaults, a profile, or the CLI.

### 3.3 From Error to Multiplier to Bonus

In the default whiten mode, let e be current error, and μ_e and σ_e the error mean and standard deviation **before adding this batch**. After warm-up, when statistics are available, compute the default clipped multiplier as:

$$
\alpha=\operatorname{clip}\left(1+\frac{e-\mu_e}{\sigma_e},\ 1,\ L\right)
$$

This measures how many standard deviations the error exceeds its usual value, then adds 1.

- At or below the mean, the lower bound yields `α=1`.
- One standard deviation above the mean yields `α=2`.
- Even large errors cannot exceed L, which defaults to 5.
- During warm-up, without scoring history, or when `σ_e=0`, every lane gets `α=1`.

The facade converts the multiplier into a bonus.

`error_norm.use_centering=false` does not subtract the mean and uses `α=1+e/σ_e`. Clipped mode caps this at L; unclipped mode returns it unchanged. The denominator is the same error standard deviation, and during warm-up, with empty statistics, or with zero standard deviation, both give multiplier 1. Centering and clipping are selected independently. Without subtracting the mean, ordinary observations also receive positive reward, which can also favor living longer, so read comparison conditions for β together.

`multiplier_mode=unclipped` applies neither lower nor upper bound and returns `α=1+z` in both statistics modes. Errors below the mean yield negative bonuses, and `α` itself may become negative. During warm-up or with a zero divisor the multiplier is 1, and `clip_ratio` is always 0. `multiplier_max` is only validated, with one construction-time WARN only when its resolved value differs from the implementation default 5.

$$
b_i=\begin{cases}
0 & \text{if next\_state is scored, use\_done\_mask=true, and lane }i\text{ has done = true}
\\
\beta\,(\alpha_i-1) & \text{otherwise}
\end{cases}
$$

$$
r_i^{\mathrm{learn}}=\operatorname{RewardScale}(r_i^{\mathrm{ext}})+b_i
$$

`i` is the lane index, `r_ext` the environment's extrinsic reward, and `RewardScale` the existing reward-scaling operation. Subtracting 1 makes the familiar-observation baseline `α=1` produce zero extra reward. With current β=0.1 and L=5, bonuses lie in `[0, 0.4]`. Extrinsic reward is not multiplied by α, so even zero-extrinsic-reward transitions can receive bonuses.

When scoring next observations, lanes with `done` still participate in scoring, statistics updates, and auxiliary learning. With `intrinsic.use_done_mask=true` (default), only **the amount added to reward** is zeroed; false adds the terminal observation's score unchanged. `truncated=true` alone never zeros it. `observation=state` uses the observation at which the action was taken and never masks done, because that observation was valid at the time of the action leading to termination. In that case use_done_mask is unused, and specifying the nondefault false produces one construction-time WARN.

### 3.4 Score First, Become Familiar Later

Operation order changes the bonus from the same observation and is therefore part of the algorithm.

The following table shows the default whiten, clipped, and next_state. Selecting state changes the observation and whether masking applies; use_done_mask=false changes reward masking; variance_feedback changes the update order of observation and error statistics as described in the previous section.

| Order | Operation | State used or updated |
| ---: | --- | --- |
| 1 | Scale extrinsic reward | Existing reward scaler |
| 2 | Normalize next observations | RND observation statistics before including current observations |
| 3 | Compute errors from target and predictor outputs | Predictor weights before updating |
| 4 | Compute multiplier α | Error statistics before adding current errors |
| 5 | Update observation/error statistics and scored-observation count | All lanes participate; default warm-up excludes errors |
| 6 | Evaluate β, mask done, and add bonuses | `counts.exp_step` and current α |
| 7 | Facade adds credit, selects k lanes per available learning step, and calls `Learn` | Observation statistics updated in step 5; normalize and forward all k inputs on each call, learning with a fresh graph |
| 8 | Pass experience with composed rewards to the inner Learner | Existing Q-learning and ReplayBuffer path |

Training the predictor first and then measuring error would score observations after becoming familiar with them, so both statistics modes score before learning. When error statistics are updated differs by method: whiten uses the baseline before adding current values, and variance_feedback uses the baseline after adding them.

`Estimate` also performs step 5. It is not a read-only query, but retains no computation graph. `Learn` computes independently from supplied observations, reading current observation statistics without updating them. Consequently, the same raw observation can normalize differently during scoring and learning.

The facade calls `Estimate` once for all intake observations, then `Learn` zero or more times according to credit. The public API requires neither identical observations nor a fixed order or count. For example, two scoring calls can precede learning on other observations, and `Learn` alone can train even during the default warm-up. `Learn` returns true when the optimizer advances, and false during collect_errors or variance_feedback warm-up. The interface also supports separating action-time scoring from auxiliary learning on Replay observations.

### 3.5 Warm-up and Lane Sampling

Default `warmup_steps=12,800` counts **scored observations**. Scoring B observations adds B; through the facade, this uses the same unit as exp steps. N=128 takes 100 intakes and N=64 takes 200 to collect 12,800 observations. `Learn` cannot finish warm-up by itself because it does not update observation statistics. With zero warm-up, the first `Learn` call trains.

Whiten's α checks the count before scoring the batch and returns 1 for all lanes below the threshold. With `warmup_mode=learn_predictor` (default), `Learn` still returns true and trains during that period, and only observation statistics are updated. Error statistics are empty, so error_mean/error_std are NaN. The first batch after warm-up also returns multiplier 1 from the empty statistics and adds its errors to the statistics. From the next batch on, the pre-scoring statistics are used.

`warmup_mode=collect_errors` updates both observation and error statistics during warm-up, while `Learn` returns false if the post-scoring count is at or below the threshold. variance_feedback warm-up keeps this form and does not use warmup_mode. However, its scoring checks the per-lane pre-scoring count, so normal multipliers begin partway through a batch that crosses the threshold. `warmup_steps=0` trains from the first call and adds errors to the statistics from the first call in either warmup_mode. An unused warmup_mode produces one construction-time INFO only when it differs from the default.

The facade owns learning cadence. Let k be `min(learn_batch_size,N)`; add `N × learn_ratio / k` to credit per intake. Each full unit permits one step on k observations, retaining fractional credit. At default ratio 0.25 and batch size 32, N=128 gives one step per intake, while N=64 gives one every two intakes: both train on 32 observations once per 128 experiences. If N is below batch size, construction warns once and shrinks the batch to N; equal step counts are then not guaranteed.

Sampling creates one lane permutation per intake and takes chunks of k. No lane repeats across steps, which are capped at `floor(N/k)`. If N is not divisible by k and the cap is reached, unspent credit, including its integer part, carries forward. Each `Learn` call consumes credit even if it returns false, preventing catch-up training after warm-up. Only successful true results increment `learn_count`.

All lanes are scored and included in statistics. RND `Learn` does no sampling: it normalizes all k supplied observations, forwards them through target and predictor, and backpropagates mean loss. Intakes without learning retain `predictor_loss` from the latest successful step; before any learning it is NaN.

### 3.6 Connection to DQN and the Two Targets

```mermaid
sequenceDiagram
    participant F as facade / UpdateFromBatch
    participant S as RewardScaler
    participant R as RND estimator
    participant T as RND target
    participant P as RND predictor
    participant O as RND optimizer
    participant L as DQN Learner
    participant N as NetworkModel
    F->>S: Scale(外的報酬)
    S-->>F: スケール済み外的報酬
    F->>R: Estimate(次観測)
    R->>R: byte→float、更新前統計で正規化、H2D
    R->>T: forward(全lane、勾配なし)
    T-->>R: 教師の出力
    R->>P: forward(全lane、勾配なし)
    P-->>R: 予測
    R->>R: 誤差とtarget出力のD2H、有限検査
    R->>R: 更新前の誤差統計でα、統計更新
    R-->>F: α（入力と同じdevice）
    F->>F: βと設定されたdoneマスクで内的報酬を合成
    F->>F: credit += N × learn_ratio / k、lane順列を準備
    loop creditが1以上、最大floor(N/k)回
        F->>R: Learn(k laneずつ、重複なし)
        opt learn_predictor、またはwarm-up後
        R->>R: 全k件を今の統計で正規化、H2D
        R->>T: forward(選択lane、勾配なし)
        T-->>R: 教師の出力
        R->>P: forward(選択lane、グラフを作る)
        P-->>R: 予測
        R->>R: 誤差のD2H、有限検査、lossの診断
        R->>O: zero_grad
        R->>P: 平均lossをbackward
        R->>O: step（predictorの重みを更新）
        end
        R-->>F: optimizerを進めたか（bool）
        F->>F: creditを1消費、trueならlearn_countを加算
    end
    F->>L: 合成済み報酬を含む経験
    L->>N: Replayからの学習でNNの重みを更新
```

The diagram shows one intake with the default whiten, clipped, and next_state, using CPU observations and CUDA computation. H2D transfers CPU to GPU; D2H transfers GPU to CPU; CPU computation omits these transfers. CUDA inputs move to CPU before normalization, and multipliers return to the input device. The facade completes RND `Learn` calls before passing experience to the inner Learner. The Learner updates its NN only when existing Replay and update-interval conditions hold. RND auxiliary learning and DQN NN updates use separate weights and optimizers.

| Comparison | DQN target network | RND target |
| --- | --- | --- |
| Output | Action-value estimates | Feature vectors for the predictor to reproduce |
| Purpose | Construct Q-learning targets | Fix the exercises used to measure novelty |
| Weight changes | Updated from the Q-learning Network by the configured method | Fixed after initialization |
| Owner | DQN `NetworkModel` | `RndNoveltyEstimator` |

Despite the shared name, RND target must not enter DQN synchronization. Reducing predictor error and changing Q values using training rewards are separate learning processes and do not share an optimizer.

### 3.7 Bonuses Stored in Replay Are Not Rescored

Experience arriving at the facade receives extrinsic reward plus bonus, passed to ReplayBuffer as one reward. Later sampling does not recompute the bonus with a newer predictor.

For example, a transition receiving bonus 0.2 when collected retains that bonus in its stored training reward even after the location becomes familiar. It need not match the bonus for visiting now. Changing β also does not rewrite rewards already in Replay.

This avoids changing Replay's format but prevents later separate handling of extrinsic and intrinsic rewards. It is an implementation choice, not an inherent RND restriction.

### 3.8 Reading the Implementation

| Question | File or operation | Connection to the background |
| --- | --- | --- |
| What callers can access | `LifelongNoveltyEstimator` and Factory in [`intrinsic.hpp`](../../core/anet-core/include/anet/intrinsic.hpp) | Expose multiplier scoring, auxiliary learning, diagnostics, and persistence while hiding RND-specific composition |
| Where reward is added | `UpdateFromBatch` in [`default_dqn_agent.cpp`](../../core/anet-core/src/default_dqn_agent.cpp) | Add after existing reward scaling |
| Learning cadence and sampling | `UpdateFromBatch` in [`default_dqn_agent.cpp`](../../core/anet-core/src/default_dqn_agent.cpp) | Facade owns credit and RNG with named seed `intrinsic.learn`, supplying k observations at a time |
| What RND owns | [`intrinsic_rnd.hpp`](../../core/anet-core/src/intrinsic_rnd.hpp) | Target, predictor, optimizer, two statistics, and scored-observation count |
| Scoring and learning order | `Estimate` / `Learn` in [`intrinsic_rnd.cpp`](../../core/anet-core/src/intrinsic_rnd.cpp) | Update statistics after scoring; learning computes independently from supplied observations and current statistics |
| Exact normalization | [`scaler_impl.cpp`](../../core/anet-core/src/scaler_impl.cpp) and [`scaler_impl.hpp`](../../core/anet-core/src/scaler_impl.hpp) | Use shared normalization, cumulative statistics, and persistence |
| Contract verification | [`intrinsic_test.cpp`](../../core/anet-core/src/intrinsic_test.cpp) | Test fixed teacher, learning, update order, reward addition, and restoration through public paths |

DefaultDQN interprets `intrinsic.enabled`, `intrinsic.beta`, `intrinsic.observation`, `intrinsic.use_done_mask`, `intrinsic.learn_ratio`, and `intrinsic.learn_batch_size`. RND-specific settings go to the Factory and are interpreted by `RndNoveltyEstimatorConfig`. Reward composition and learning cadence are separated from novelty computation and auxiliary learning itself.

## 4. Examples: Reading Values, Settings, and Metrics

### 4.1 A Four-Lane Calculation

Assume warm-up has finished, historical error mean is 0.20, standard deviation is 0.10, L=5, and β=0.1. The following errors and rewards are illustrative, not Run measurements.

| Lane | Current error e | Unclipped `1+(e−0.20)/0.10` | α | done | Scaled extrinsic reward | Added bonus | Training reward |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0.10 | 0 | 1 | false | 0 | 0 | 0 |
| 1 | 0.30 | 2 | 2 | false | −0.1 | 0.1 | 0 |
| 2 | 0.80 | 7 | 5 | false | 1 | 0.4 | 1.4 |
| 3 | 0.60 | 5 | 5 | true | −1 | 0 | −1 |

Lane 1 receives a positive bonus but training reward is zero after the extrinsic penalty cancels it. Lane 3 has high error but no bonus because it is done. Statistics updates use errors from all four lanes.

Here `bonus_mean` is `(0+0.1+0.4+0)/4=0.125` and `bonus_max` is 0.4. `clip_ratio` counts lanes with α equal to L, including done lane 3, so it is `2/4=0.5`.

### 4.2 Which Computations Settings Change

Keys below omit `DefaultDQNAgent.`. Defaults correspond to [`agent.txt`](../../apps/runner/config/agent.txt) and the RND configuration type.

| Key | Default | Effect and notes |
| --- | ---: | --- |
| `intrinsic.enabled` | false | False creates no estimator and does not interpret β or other RND settings |
| `intrinsic.beta` | 0.1 | Coefficient multiplying `α−1`; nonnegative ProfiledValue. Zero does not stop scoring or auxiliary learning |
| `intrinsic.observation` | next_state | Score/learn the next observation, or state for the raw action-time observation |
| `intrinsic.use_done_mask` | true | Zero done bonuses when scoring next_state; unused for state |
| `intrinsic.rnd.warmup_mode` | learn_predictor | Train during whiten warm-up; collect_errors instead collects errors and suspends learning |
| `intrinsic.rnd.error_norm.average_mode` | cumulative | Whiten error statistics: cumulative or ema |
| `intrinsic.rnd.error_norm.ema_steps` | 1,000,000 | EMA time constant in scored observations; integer at least one |
| `intrinsic.rnd.error_norm.use_centering` | true | Subtract the error mean in whiten scoring |
| `intrinsic.rnd.multiplier_mode` | clipped | Unclipped returns the multiplier without either bound, including negative bonuses |
| `intrinsic.rnd.statistics_mode` | whiten | variance_feedback uses the separate sequential recurrence |
| `intrinsic.rnd.head.init.*` | xavier, manual_gain=0 | Existing WeightInitConfig applied to both Heads |
| `intrinsic.rnd.adam_eps` | 1e-8 | Private AdamW epsilon; finite and positive, current setting also used on load |
| `intrinsic.rnd.output_dim` | 128 | Target/predictor vector dimension, at least 1 |
| `intrinsic.rnd.multiplier_max` | 5 | Multiplier upper bound L; finite and greater than 1 |
| `intrinsic.learn_ratio` | 0.25 | Auxiliary-learning observations per collected experience; finite, greater than 0 and at most 1 |
| `intrinsic.learn_batch_size` | 32 | Observations per auxiliary-learning step; integer at least 1. See §3.5 for shrinking to N |
| `intrinsic.rnd.warmup_steps` | 12,800 | Observations without a bonus; nonnegative, with behavior selected by warmup_mode |
| `intrinsic.rnd.learning_rate` | 0.0001 | Predictor learning rate; finite and positive |
| `intrinsic.rnd.weight_decay` | 0 | Predictor AdamW weight decay; finite and nonnegative |
| `intrinsic.rnd.obs_norm.clip_range` | 5 | Clipping range after observation normalization; separate from α upper bound L |
| `intrinsic.rnd.target.net.*` | Supplied by each environment's `run.@rnd` | Required target input and Body |
| `intrinsic.rnd.predictor.net.*` | Supplied by each environment's `run.@rnd` | Required predictor input and Body; never implicitly copied from target |

`learn_ratio` and `learn_batch_size` change optimizer-step counts and observations per step, affecting how quickly error falls. They are not merely speed settings. β changes reward magnitude for the same α; L changes the multiplier cap for large errors. Distinguish where each acts.

All five new keys are type/range/enum checked when intrinsic reward is enabled, including unused values. Variance feedback ignores warmup_mode and error_norm.*; cumulative ignores ema_steps. A resolved unused value differing from its implementation default produces one construction-time WARN per key. Whiten with zero warm-up instead reports a nondefault warmup_mode once at INFO. Disabled intrinsic reward does not read, validate, or notify for these values.

### 4.3 Structure Selected by `run.@rnd`

Appending `run.@rnd` to an existing Run chain jointly selects RND enablement, independent target/predictor Bodies, and thirteen metrics. It does not specify a complete environment and Q-learning configuration by itself. See [100 Runtime and Configuration](100_runtime_and_configuration.en.md) for configuration-chain semantics.

| Environment configuration | RND input | Default Body |
| --- | --- | --- |
| [Atari.txt](../../apps/runner/config/Atari.txt) | `grid` | Dedicated three-layer CNN and Flatten for target; two additional 512-wide Linear/ReLU layers for predictor |
| [GridMaze.txt](../../apps/runner/config/GridMaze.txt) | `vector` | `Flatten > MLP_FC1 > ReLU > MLP_FC2 > ReLU` |
| [LunarLander.txt](../../apps/runner/config/LunarLander.txt) | `vector` | Same as above |

Atari's CNN passes through `AtariRndConv1` (32 channels, kernel 8, stride 4), `AtariRndConv2` (64, 4, 2), and `AtariRndConv3` (64, 3, 1) in order, with `AtariRndLeaky` (LeakyReLU, negative_slope 0.2) after each Conv. All padding is 0, so a 1×84×84 input becomes 3,136 dimensions after Flatten. The Linear added only to the predictor is `AtariRndFC`. Both Bodies are followed by a 512-dimensional Linear Head. Conv, FC, and Head use orthogonal initialization, gain √2, and bias 0, giving 1,677,984 parameters for target and 2,203,296 for predictor. This is a model separate from the Q network.

GridMaze/LunarLander reference `net.branch.@RndMlp` from both Bodies. It uses `MLP_FC1` / `MLP_FC2` from the common catalog, with a default 128-dimensional Head. The common catalog's `Conv1d_S/M` and `Conv2d_S/M` read the `conv.*` keys; a two-layer Conv2d gives 6,400 after Flatten and 854,240 parameters including the 128-dimensional Head.

Atari's `run.@rnd_taiga` selects the following eight values: β=0.00005, observation=state, learn_ratio=1, learn_batch_size=1, multiplier_mode=unclipped, statistics_mode=variance_feedback, warmup_steps=0, and adam_eps=1e-5. Add the Q-side `run.@rainbow_like` to the existing chain, then layer `run.@rnd > run.@rnd_taiga` on top. It does not select num_envs. Also confirm in the resolved Run configuration the Q-side noop_max=0, the Nature structure of main_feature, and the absence of Munchausen metrics.

RND receives the selected raw observations, rather than Actor-preprocessed Q inputs. For Atari as well, distinguish Q-input frame stacking from RND input.

### 4.4 Reading Metrics Together

The following maps `GetScalar` keys to configured recording destinations. Prefix keys with `intrinsic.` and destinations with `56_agent_in/`.

| Key | Destination | Measurement |
| --- | --- | --- |
| `bonus_mean` | `01_bonus_mean` | Batch mean of bonuses actually added; done is masked only for next_state with use_done_mask=true |
| `bonus_max` | `02_bonus_max` | Maximum of that bonus |
| `bonus_share` | `03_bonus_share` | Absolute bonus sum divided by absolute scaled-extrinsic plus absolute bonus sums |
| `bonus_min` | `04_bonus_min` | Minimum bonus actually added, possibly negative in unclipped mode |
| `bonus_pos_ratio` | `05_bonus_pos_ratio` | Fraction of lanes receiving a positive bonus |
| `error_mean` | `11_error_mean` | Error mean used for the next batch: cumulative or EMA in whiten, NaN while empty |
| `error_std` | `12_error_std` | Corresponding error standard deviation, NaN while empty |
| `error_step` | `13_error_step` | Current error mean over all lanes |
| `target_feature_var` | `14_target_feature_var` | Mean cumulative population variance over target-output dimensions; zero for the first observation |
| `target_feature_max` | `15_target_feature_max` | Maximum absolute target output in the current batch |
| `clip_ratio` | `21_clip_ratio` | Fraction of lanes whose current α equals L, before done masking |
| `predictor_loss` | `31_predictor_loss` | Pre-update mean error of the latest successful auxiliary-learning step; NaN before learning |
| `learn_count` | `32_learn_count` | Cumulative successful optimizer steps, including learn_predictor warm-up; false returns do not count |

`bonus_share` uses the following ratio, not a mean of individual lane ratios:

$$
\mathrm{bonus\_share}=\frac{\sum_i |b_i|}{\sum_i |r_i^{\mathrm{scaled}}|+\sum_i |b_i|}
$$

A zero denominator yields NaN. In §4.1, it is `0.5/(2.1+0.5) ≈ 0.1923`. Absolute extrinsic rewards reduce instability caused by cancellation of positive and negative rewards.

Example interpretations:

- Falling `predictor_loss` and `error_step` indicate progress in predicting inputs. Their values usually differ because the former uses selected lanes and the latter all lanes.
- Low `bonus_mean` can reflect below-usual error, warm-up, small β, many done lanes, or other factors. Do not infer the cause from one metric alone.
- Persistently high `clip_ratio` means many observations receive the same capped multiplier. Check current errors, cumulative statistics, and input normalization together.
- `error_mean` / `error_std` are post-scoring statistics. Whiten's α uses the values before the update, while variance_feedback uses the updated mean and `err_v`. Even for the latter, the published std is the square root of the variance, so do not use the logged std directly as the divisor.

Disabled RND keys return `nullopt`. When enabled, `learn_count` starts at 0, and intakes without learning retain the latest loss. Scores before first scoring and losses before learning are NaN. This intentional unavailability differs from abnormally nonfinite computed errors, which stop processing when detected during scoring or learning. With N at least batch size and no intake-cap constraint, the post-warm-up incremental ratio of `learn_count / exp_step` is `learn_ratio / learn_batch_size`.

### 4.5 Saving and Resuming

Predictor weights alone cannot restore RND. Changing the teacher changes the answers; changing statistics changes relative novelty; changing sampling state changes which lanes are practiced.

RND first writes `RndNoveltyEstimator-4` for whiten EMA, or `RndNoveltyEstimator-3` for cumulative and variance_feedback, followed by statistics_mode. Target, predictor, optimizer, selected statistics, cumulative target-output statistics, and scored-observation count follow in the existing order. Whiten saves observation normalization plus Welford state for cumulative, or weight sum, mean, and squared deviations for EMA. Variance feedback saves obs_m, obs_v, err_m, err_v, and n. A different identifier or statistics_mode is rejected before reading the payload, reporting both saved and expected values. Cumulative format and payload remain unchanged, allowing existing -3 checkpoints to load directly. warmup_mode, use_centering, ema_steps, and use_done_mask are not saved; the settings at resume apply.

The Agent then saves the facade's unconsumed credit, lane-selection RNG state, and cumulative success count. The RND body comes first, so a different format is rejected by the body's check. Except in the middle of a call, saving and restoring work before the first scoring, between Estimate and Learn, and immediately after Learn. Computation graphs are not carried over. The latest loss is not saved and remains NaN until the first successful step after loading.

Agent checkpoints containing RND append `;intrinsic` to the existing identifier. Loading fails if the saved checkpoint has RND but the reader disables it, or vice versa. Disabled mode uses the previous format.

Loading restores predictor optimizer training state, then applies the current learning rate, weight decay, and adam_eps, warning when they differ. Resuming with changed values therefore differs from continuing identical training.

## 5. Current Status and Limitations

### 5.1 Computation Placement and Cost

RND target, predictor, and facade lane sampling use separate named seeds. Ordinary weight initialization currently uses the default CPU generator, so construction temporarily saves its state, seeds it, and restores the state afterward. Models are constructed on CPU and moved to the required device.

Inputs may reside on CPU or CUDA. RND moves observations to CPU for normalization, then sends them to the Factory's computation device. Finite-error checks and statistics run on CPU; returned α uses the input device. The facade selects lanes first, and `Learn` normalizes/transfers all k supplied observations. CUDA-input selection indices are sent asynchronously from pinned memory once per intake; there are no synchronous per-lane transfers.

RND owns independent FP32 models. Scoring performs two forwards on all N observations; each auxiliary step performs two forwards and one backward on k observations. After warm-up, without hitting the cap, average steps per intake are `N × learn_ratio / k`. Default N=128 gives one step on 32 observations. Learning needs fresh forwards because scoring graphs are not reused. The speed difference measured on GridMaze with CPU and 256 lanes is given in the next section. The cost for Atari image inputs remains unmeasured, and observation-statistics updates still duplicate normalization through the existing API.

variance_feedback updates observation statistics on CPU in lane order, then forwards the results together. To accumulate target_feature_var, scoring returns the N×output_dim target outputs to CPU in addition to errors. The Taiga configuration uses k=1 and ratio=1, so it runs N backwards per intake, a cost very different from the default cadence. Transfer/synchronization estimates and the scope actually confirmed are recorded in [23impl](../memo/116_rnd_lifelong_novelty_23impl.md). Performance measurement is separate.

Scoring and auxiliary learning occur per batch independently of Q-learning step counts, so increasing the Q-update interval need not reduce RND cost proportionally. Profiling separates sampling, normalization, input transfer, target/predictor forward, error transfer, statistics, backward, and optimizer step.

With `learner.enabled=false`, scoring and auxiliary learning pause with one warning. If intrinsic reward is enabled, components still exist for persistence. Diagnostics use a value-transfer mechanism without adding a new learning-lock wait for display.

### 5.2 Verified Scope

Validation procedures for error statistics, warm-up, centering, and terminal masking are in the [PRD123 validation assets](../../core/anet-core/testdata/prd123/README.md). They distinguish before/after comparisons for collect_errors (which selects the old behavior), variance_feedback, and disabled RND from hand-calculated EMA and save/restore checks. KFM effectiveness checks are performed only after separate approval.

Validation of the 2026-11-07 changes that added the public-implementation form and options is in [23impl](../memo/116_rnd_lifelong_novelty_23impl.md), with rerun entry points in [PRD116 validation assets](../../core/anet-core/testdata/prd116/README.md). For earlier validation, see [20impl](../memo/116_rnd_lifelong_novelty_20impl.md), [21impl](../memo/116_rnd_lifelong_novelty_21impl.md), and [22impl](../memo/116_rnd_lifelong_novelty_22impl.md). Enabled-mode learning sequences differ from past Runs because of Network and configuration changes.

| Check | Result and scope |
| --- | --- |
| Debug build and related tests | Full build; 50 tests / 4,371 assertions passed. CNN CPU tests used OMP_NUM_THREADS=1 |
| Fixed teacher and predictor | Confirmed unchanged target weights/outputs and predictor learning after 100 updates on identical input |
| Statistics, inputs, and save/resume | Confirmed three hand-calculated variance_feedback steps with N=1, sequential updates with N=2, preservation of categorical IDs (alone and mixed with continuous values; int64/uint8/float), scoring and learning with state, negative-bonus ratios, CPU/CUDA inputs for both modes, and continued saving plus rejection of mode mismatches |
| Shipped configuration | Parameter counts of both Atari Networks, the approximately 33.2 MB checkpoint increase, Conv1d/2d dimensions, and the eight Taiga keys matched |
| Run configuration resolution | Full-key comparison with a saved Montezuma baseline Run. The only differences from rainbow to taiga were 18 RND-setting keys and 13 diagnostic keys. Confirmed noop_max=0, the Nature structure, and the absence of Munchausen metrics |
| Disabled-mode equivalence | 105,682 points and metric definitions matched the previous implementation across GridMaze, LunarLander, Atari, and DropMerge |
| GridMaze smoke | Completed 100k experiences, with 5,448 post-learning points, a maximum recorded learn_count of 2,724, and no invalid nonfinite metric values, meeting the revised criteria of completion, finiteness, and auxiliary-learning progress. Because the observation is a normalized (x, y) pair on a fixed board, the default bonus becomes 0 once the predictor becomes familiar with the limited inputs and novelty is exhausted; sign and decrease are not judged |
| KFM smoke | Completed 100k experiences with the new three-layer CNN; clip_ratio over 5,448 post-learning points peaked at 0. However, the saved r1 lacks bonus_share at 4,008 points and fails the rejudgment with a stricter NaN exception. The old aggregation is retained, and no new Run was started |
| Montezuma smoke | Completed 100k experiences with the Taiga configuration. bonus_min was negative at all 6,233 recorded points, with no invalid nonfinite metric values, meeting the acceptance criteria |
| Performance | Ran three ABBA rounds on GridMaze with CPU, 256 lanes, and Release. The median for 165,632 experiences was 23.261 s without RND and 31.148 s with it (1.339×). Tracy on a separate Run showed RND-section forward 56.47%, backward plus optimizer 35.10%, and statistics 4.72%. The [experiment record](../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md) keeps the configuration, variance, and breakdown of every Run. The three Atari B=64 groups have not been run |

Whole-file bytes of disabled-mode checkpoints differ even between reruns of the old executable, owing to parameter-address information saved by existing libtorch. Validation compared header, configuration, and Network bytes, plus optimizer contents excluding address-related metadata. It does not claim whole-file byte equality.

### 5.3 What Remains Unproven

Short smoke runs establish the connection between computation, reward addition, learning, and recording. They do not demonstrate improved extrinsic performance, resolution of difficult-game exploration problems, or acceptable performance cost.

Motivation to revisit familiar paths, episodic novelty, separate extrinsic/intrinsic learning, and families of policies with different exploration strengths remain separate from this standalone RND implementation. Related options and open questions are in [PRD924](../memo/924_intrinsic_reward_novelty_10prd.md).

run.@rnd_taiga is not a complete reproduction of Taiga et al. either. Q uses IQN rather than C51, the CNN uses VALID rather than SAME padding, and Q embeddings, initialization, PER, the units used to count statistics, and the ALE version also differ. During auxiliary learning, inputs are renormalized with the updated statistics. The list of remaining differences is in the [23impl table](../memo/116_rnd_lifelong_novelty_23impl.md#%E6%9C%AC%E5%AE%B6%E3%81%A8%E3%81%AE%E6%AE%8B%E3%82%8B%E5%B7%AE).

### 5.4 Sources and Related Documents

- [Burda et al., Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894), §2.2: source for fixed-teacher/predictor novelty estimation.
- [Badia et al., Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038), §2, Equation (1): source for lifelong novelty as a multiplier.
- [PRD116](../memo/116_rnd_lifelong_novelty_10prd.md), [initial implementation (20impl)](../memo/116_rnd_lifelong_novelty_20impl.md), [public-API revision (21impl)](../memo/116_rnd_lifelong_novelty_21impl.md), and [cadence/batch-size revision (22impl)](../memo/116_rnd_lifelong_novelty_22impl.md): adopted contracts, implementation decisions, and validation. See 21impl for independent Estimate/Learn and device/persistence contracts, and 22impl for credit, observation counts, and bool results.
- [Options and Network fixes aligned with the public implementation (23impl)](../memo/116_rnd_lifelong_novelty_23impl.md): implementation and validation of separate Networks, selection of statistics/observations/multipliers, Head initialization, Adam eps, diagnostics, and format 3.
- [200 DQN Agents](200_dqn_agents.en.md): Agent specification including the stages surrounding intrinsic-reward integration.
- [110 Agents and Learning](110_agents_and_learning.en.md), [130 Neural Networks](130_neural_networks.en.md), and [150 ReplayBuffer](150_replay_buffer.en.md): specifications of shared components used here.
