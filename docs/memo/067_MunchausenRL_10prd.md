# Munchausen RL（M-DQN / M-QR / M-IQN）PRD

> 起点: 2026-09のAtari BTR差分キャンペーン。BTRは `gamma=0.997` とMunchausenを同時に採用するが、本コードベースはgammaだけを採用している。
>
> 一次根拠: [Munchausen Reinforcement Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)、[Supplementary Material](https://papers.nips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Supplemental.pdf)
>
> 関連決定: [ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)、[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)、[done/059](done/059_config_concept_tree_alignment_10prd.md)（TARGET軸の配置と遅延ゲート）、[999_noisynet](999_noisynet_10prd.md)（BTR採用部品のうち別途扱う未実装機能）
>
> 履歴資料: [done/035](done/035_approx_actor_priority_per_10prd.md)は当時のK2契約を記録した資料として変更しない。
>
> 追補（2026-09-05）: 別枠検討（方策温度の定量解析、soft楽観ターゲット）をD15と数理契約の2小節として反映した。

## Context（背景・目的）

Munchausen RLは、Bellmanターゲットの報酬側へエージェント自身のscaled log-policyを加え、次状態のhard argmax bootstrapをsoft価値へ置き換える。NeurIPS論文はM-DQNを1-step、M-IQNを3-stepで評価しているため、本PRDでは論文の1-step式とanet-labのN-step target returnを一括して同一視しない。補遺の分位点ごとの方策混合を根拠にしつつ、bonusのN-step上の帰属はBTR互換の拡張として明示する。

本PRDの起点は、`gamma=0.997` がaction gapを犠牲にして地平を延ばしているという[baseline探索ブロック19](../experiments/default-dqn/atari/2026-08-17_baseline.md)と、補償器であるMunchausenが未実装だと整理した[可塑性保護screening](../experiments/default-dqn/atari/2026-08-30_protection-screening.md)である。[BTR survey Table 2](../../reports/btr_hyperparams_survey_2026-08-26.md)にはMunchausen除去時のAction GapとPolicy Churnの差が記録されている。一方、[Atari実験README](../experiments/default-dqn/atari/README.md)は定常 `q_gap` を成績の予測子として採用しないと裁定しているため、本PRDもaction gapやscoreを合否ゲートにしない。

本PRDの目的は、Munchausen RLをDQNBasedの3 Learnerに共通する既定OFFの契約として実装可能な状態へ確定することである。対象は `TDLearner`、`QRLearner`、`IQNLearner` と、近似Actor初期優先度を成立させるActor Qヒントである。性能改善やスコア改善の証明ではなく、数理・設定・transport・診断・検証の契約を固定する。

実装は1フェーズで行う。推奨する実装順は、① `MakeRiskBiasedScore` の抽出（単独で完結するリファクタ。OFF数値不変）→ ② Munchausen本体 → ③ `GetRiskScoreSpec` seamとsoft楽観ターゲット（D15）である。③に着手せず止まる場合は、D5を「`use_optimistic_target=true` との併用もfail-fast」へ戻して整合させる。今回の改訂作業は文書のみであり、production code、config、テスト、現行実装を説明する `docs/design` は変更しない。

## Goals

- TD / QR / IQNで同じMunchausen意味論を提供する。
- TBOやAMPの有無にかかわらず、方策温度を報酬スケール上で解釈できるようFP32実空間で計算する。
- `target`、`online`、`online_reuse` の3 modeを、forward mode・forward回数・IQN RNG消費まで含めて区別する。
- 明示したが効果を持たない `learner.use_double_dqn=true` との併用を構築時にfail-fastし、`use_optimistic_target` はsoft経路でも尊重する（D15）。
- `actor_approx` の初期優先度をMunchausen targetと同型にしつつ、ActorへLearnerの責務を漏らさない。
- OFF時の保証範囲を、実際に価値があり検証可能な範囲へ限定する。

## Non-Goals

- SAC / M-SAC向け共通moduleの先行抽出。
- RainbowAgentへのMunchausen設定公開。
- `action_mask` 対応。
- softmax行動など探索方策の変更。
- M-VI、q-Munchausenなどの派生方式。
- action gap、policy churn、score、throughputを用いた採否判定。
- soft楽観ターゲット（D15）の性能評価。`use_optimistic_target` on/offの再測定を含む。
- IQNで厳密な `Z_tau` を得るための追加tau列（Complexity AuditのDefer）。
- 過去文書 `docs/memo/done/035_approx_actor_priority_per_10prd.md` の改訂。

## 数理契約

記号を次のように置く。

- `B`: batch size
- `A`: 行動数
- `N`: current側の分位点数
- `M`: target側の分位点数（QRでは固定分位点数）
- `R_i`: ReplayBufferが保持するN-step割引return
- `n_i`: sampleごとの実N-step数
- `d_i`: 真の終端
- `h` / `h^-1`: TBO変換。`use_tbo=false` では恒等写像
- `tau`: Munchausenの方策温度 `entropy_tau`
- `l0`: bonus側clip下限 `clip_value_min`

以下のMunchausen計算はすべてNoGradのFP32実空間で行う。

### Scaled log-policy

実空間Q値 `q[B,A]` に対し、数値安定な式でscaled log-policyを求める。

```text
m_i = max_a q_ia
scaled_log_policy_ia
  = q_ia - m_i - tau * log(sum_b exp((q_ib - m_i) / tau))
  = tau * log(pi_ia)
pi_ia = exp(scaled_log_policy_ia / tau) = softmax(q_i / tau)_a
```

`scaled_log_policy <= 0` である。`tau` はIQNのtaus、target network更新率、gradient clipなどのtauとは別概念である。

### Munchausen項

実行行動 `a_i` に対して次を求める。

```text
bonus_i = alpha * clip(scaled_log_policy_current[i,a_i], l0, 0)
```

bonusは集約済みN-step return `R_i` の先頭遷移へ一度だけ加える。N-step区間内の各遷移へ繰り返し加えず、終端でも残す。

### Soft価値ブートストラップ

scalar TDでは次状態のtarget Qから求める。

```text
V_soft_i = sum_a pi_next[i,a]
                    * (q_next_real[i,a] - scaled_log_policy_next[i,a])
```

QR / IQNでは、全分位点に共通する方策をbootstrap方策のスコア `s_next`（既定は実空間分位点の平均。楽観ターゲット時は後述のrisk-biasedスコア）から作り、同じ方策で各分位点を混合する。

```text
s_next[i,a] = mean_j Z_next_real[i,a,j]        （既定。楽観ターゲット時は RiskScore(Z_next_real[i,a,:])）
pi_next = softmax(s_next / tau)、scaled_log_policy_next = tau * log(pi_next)
soft_dist[i,j] = sum_a pi_next[i,a]
                     * (Z_next_real[i,a,j] - scaled_log_policy_next[i,a])
```

分位点ごとに別の方策を作らない。soft価値側のscaled log-policyはclipしない。

### N-step target

```text
TD:
  y_i = R_i + bonus_i + (1 - d_i) * gamma^n_i * V_soft_i

QR / IQN:
  y_ij = R_i + bonus_i + (1 - d_i) * gamma^n_i * soft_dist[i,j]
```

終端maskはbootstrapだけへ適用する。TBO有効時はscalar Qまたは各分位点へ個別に `h^-1` を適用してから平均・方策・bonus・soft価値を求め、target完成後にだけ `h` を適用する。非線形なので `h^-1(mean(Z))` で代用しない。

### 方策温度の効き方

`scaled_log_policy = q - LSE_tau(q)`（`LSE_tau(q) = tau * log(sum_a exp(q_a / tau))`）から次が従う。`tau` は方策温度 `entropy_tau` である。

- 上位2行動 `a1`、`a2` がどちらもclip未到達なら、bonus差は `alpha * (q_a1 - q_a2)` で `tau` に依存しない。action gapの拡大率 `(1 + alpha)` は温度で決まらない。
- clipは拡大に上限を与える。bonus差の最大は `alpha * |l0|` で、元のgapが `|l0|` 以上の行動対は拡大しない。`tau` を上げるとLSEが増えて全行動がclipへ到達し、bonusが定数 `alpha * l0` になってgapへの効果が消える。
- soft価値の楽観分は `V_soft - max_a q = LSE_tau(q) - max_a q ∈ [0, tau * ln A]`。`tau=0.03`、`A=6` では最大でも0.054で、gapが `tau` より十分大きい状態では `≈ tau * sum_{a≠a*} exp(-(q_a* - q_a) / tau)` まで縮む。既定の `tau` は楽観化の設定としては実質機能しない。
- `LSE_tau(q) >= max_a q` は恒等式なので、`tau` では悲観側を表現できない。悲観・楽観の軸は分位（risk）側にあり、方策温度とは直交する。

### bootstrap方策のスコア（楽観ターゲット）

soft価値ブートストラップは行動選択を行わないが、方策 `pi` を作るスコア `s[B,A]` は `target_policy` の種類に従う。scalar TDでは `s = q_real` である。分位点表現では次のとおり。

| `target_policy.policy_type`（`use_optimistic_target=true` ならtrain_policyのコピー） | `s` |
|---|---|
| `Greedy`、`EpsilonGreedy`（構成時にGreedyへ強制） | `mean_j Z_real[.,.,j]`（既定） |
| `ThompsonSampling` | `mean_j Z_real[.,.,j]`（Thompsonのscoreは乱択tausの平均で、target tausの平均と同じ量） |
| `UQE` | `MakeRiskBiasedScore(tau_risk, uqe_use_tail_mean, Z_real)`: 分位点を値で昇順ソートし、index `floor(tau_risk * (M - 1))` からのtail平均（`true`）または1点（`false`） |

`tau_risk` は `target_policy` が保持する現在の `uqe_tau`（`uqe_tau_start` から `uqe_tau_end` へexp_step減衰）であり、`uqe_use_tail_mean` も同じpolicyの値を使う。`s_current`（bonus側、`s_t`）と `s_next`（soft価値側、`s_{t+n}`）は同じスコア関数・同じ `tau_risk` で作る。

```text
pi = softmax(s / tau)
scaled_log_policy = s - LSE_tau(s)
bonus_i = alpha * clip(scaled_log_policy_current[i,a_i], l0, 0)
soft_dist[i,j] = sum_a pi_next[i,a] * (Z_next_real[i,a,j] - scaled_log_policy_next[i,a])
```

risk-biasedスコアは方策にだけ入り、価値側は全分布 `Z_real` のままである（hard楽観ターゲットが「選択はrisk、評価は選んだ行動の全分布」であるのと同じ分離）。`tau -> 0` でsoft楽観ターゲットはhard楽観ターゲット（実空間でのrisk argmax）へ収束する。risk方策では `V_soft = E_pi[q_mean] + tau * H(pi)` であり、`LSE_tau(q_mean)` とは一致しない。

- スコアは実空間 `Z_real`（TBO時は分位点ごとに `h^-1` 済み）から作るため、`uqe_use_tail_mean=true` もTBO下で成立する。`apps/runner/config/DropMerge.txt` のhard経路向け注記（h空間では単一分位点だけがargmax不変）はsoft経路には及ばない。同じ `target_policy.uqe_use_tail_mean` が両経路を支配する。
- soft経路のrisk-biasedスコアはM本の経験分位であり、networkを `tau` で直接問うhard経路のIQNとは一致しない。`target_taus.sample_mode = fixed / stratified` なら決定的、`random` では揺れる。
- TD Learner（`quantile_mode=none`）では `target_policy=UQE` が既存検証でfail-fastするため、soft楽観ターゲットは分位点表現でだけ成立する。

## 決定事項

| # | 論点 | 決定 |
|---|---|---|
| D1 | 対象 | DQNBased共通の `TDLearner` / `QRLearner` / `IQNLearner`。既定OFF、実装は1フェーズ。 |
| D2 | log-policy mode | 閉じた文字列enum `target` / `online` / `online_reuse`。既定は `target`。 |
| D3 | N-step | bonusは集約済みreturnの先頭へ1回だけ加え、終端でも残す。bootstrapだけをmaskし `gamma^n` を掛けるBTR互換拡張。 |
| D4 | 数値空間 | TD / QR / IQNすべてFP32実空間。TBOは分位点ごとの `h^-1` とtarget完成後の `h`。 |
| D5 | 競合設定 | Munchausen ON + `learner.use_double_dqn=true` は構築時に `ANET_SYSTEM_ERROR`（soft doubleは未定義）。`use_optimistic_target=true` は競合ではなくD15のスコア源として尊重する。disabled時は両方とも従来どおり許可。 |
| D6 | Actor hint | 常時K3 `[q_sa, state_value, munchausen_term]`。旧K2はschema違反。 |
| D7 | Actor config | 狭い `ActorQHintConfig` だけを渡す。Learner config全体とmodeは渡さない。 |
| D8 | OFF保証 | Learner数値経路・RNGと標準Atari構成の完全不変。actor_approxは優先度数値同値のみ。 |
| D9 | 診断 | raw 5値とEMA 2行。固定index readbackへ専用count・offset・結果fieldを持つ。 |
| D10 | 計測 | `forward_target`、`forward_munchausen_online`、`munchausen_target` を区別する。 |
| D11 | Rainbow | MunchausenアルゴリズムはOFF。共通transportがK3になることは許容する。 |
| D12 | action mask | 既知の未対応事項として記録し、現行比の相対安全性は主張しない。 |
| D13 | M-SAC | SACのActor/Critic契約成立後、共通処理が実際に2利用箇所になった時点で抽出を再検討する。 |
| D14 | 成績・性能 | runtimeと性能値は記録するが、合否閾値や成績ゲートは置かない。 |
| D15 | soft楽観ターゲット | Munchausen ON時の方策スコアは `target_policy` の種類で決まる（UQEならrisk-biased、他は分位点平均）。新キーなし。`ActionPolicy::GetRiskScoreSpec()` と `MakeRiskBiasedScore` の2機構で実装し、スコアは経験分位、診断は平均基準、初期化ログへスコア源を追記。採用根拠は設計整合（`target_policy` = bootstrap方策をhard/soft両経路が尊重）と後付け実装を避ける判断であり、RL効果の実測ではない。 |

## 実装契約

### 1. Config

`LearnerConfig` に次を追加する。

```cpp
struct MunchausenConfig {
    bool enabled = false;
    std::string log_policy_mode = "target";
    float alpha = 0.9f;
    float entropy_tau = 0.03f;
    float clip_value_min = -1.0f;
} munchausen;
```

読み取るキーは次の5つである。

```text
learner.munchausen.enabled
learner.munchausen.log_policy_mode
learner.munchausen.alpha
learner.munchausen.entropy_tau
learner.munchausen.clip_value_min
```

`enabled` に関係なく次を検証し、違反時はキー、指定値、期待値または許容範囲を含む `ANET_SYSTEM_ERROR` とする。

- `log_policy_mode` は `target` / `online` / `online_reuse` のいずれか。
- `alpha` はfiniteかつ `[0,1]`。
- `entropy_tau` はfiniteかつ `> 0`。
- `clip_value_min` はfiniteかつ `<= 0`。

`log_policy_source` は前版PRDの草案だけに存在した未実装名であり、production codeや現用configからの削除作業は発生しない。新規実装は `log_policy_mode` だけを認識し、草案名のalias、変換、互換分岐、専用tripwireは持たない。

Munchausen ON時は次の組み合わせを構築時にfail-fastする。

- `learner.munchausen.enabled=true` と `learner.use_double_dqn=true`

エラーには競合する両キー、実際の指定値 `true`、期待値 `false` を含める。`use_optimistic_target=true` は競合ではなく、D15のとおり `target_policy` のスコアがsoft経路の方策に入る。Munchausen OFF時は両機能を従来どおり許可する。

### 2. 3つのlog-policy mode

#### `target`

- `NormalizedSampleObservations` 後の `obs` と `next_obs` を各keyについてbatch方向へ連結する。
- target networkを `[2B,...]` で1回forwardし、先頭Bをcurrent bonus、後半Bをnext soft価値へ使う。
- target networkはeval modeのまま使う。
- IQNはtarget規則のM tausを2B分生成し、連結batchへ注入する。
- target側plasticity captureが有効な場合、forward直後のcapture shapeを `[2B,F]` と検証し、後半B行を `narrow(0,B,B)` して `plasticity_target_features` へ渡す。先頭Bのcurrent bonus用特徴を混ぜず、PRD 062の「TD bootstrapに使ったnext-state target特徴、B行」という意味を維持する。
- `forward_target` はこの2B forward全体を計測する。

#### `online`

- 既存のcurrent online forwardとtarget forwardを従来の順で完了する。
- その後、正規化済み `obs` をonline networkへNoGrad・eval modeで追加forwardする。
- IQNはcurrent規則のN tausを新規生成する。
- 既存current/targetのtau draw順を変えず、追加RNG消費はfresh bonus forwardの分だけにする。
- 追加forwardを `forward_munchausen_online` で計測する。

#### `online_reuse`

- 既存のtrain-mode current出力をdetachして再利用する。
- 追加forwardも追加tau生成も行わない。
- `online` と同じnetwork familyを使うが、eval-mode fresh出力とtrain-mode既存出力は異なる契約として扱う。

### 3. Learner共通計算

Munchausen用の純粋helperは既存の `dqn_based_agent.hpp` / `.cpp` 機能グループに置く。現時点で別moduleは作らない。

```cpp
struct MunchausenTargetTerms {
    torch::Tensor bonus;                    // [B], fp32 real space
    torch::Tensor next_policy;              // [B,A], fp32
    torch::Tensor next_scaled_log_policy;   // [B,A], fp32
    torch::Tensor diagnostics;              // [5] {scaled_log_policy_mean, clip_ratio, bonus_mean, next_entropy, soft_gap}
};
```

helperはcurrent/nextの実空間Q、actions、`MunchausenConfig` を受け取る。AMP領域から呼ばれても入力をFP32へcastし、安定式でscaled log-policyを計算する。`munchausen_target` はbonus、方策、soft価値、target組立の範囲を計測する。

方策スコアの入力は次で決める。`target_policy_->GetRiskScoreSpec()` が `std::nullopt` なら `q_current_real` / `q_next_real` は分位点平均（scalar TDでは `q` そのもの）、specがあれば両方を `MakeRiskBiasedScore(spec.tau, spec.use_tail_mean, Z_real)` で置き換えてからhelperへ渡す。helperのsignatureは変えない。

- `ActionPolicy::GetRiskScoreSpec()` は仮想関数で既定 `std::nullopt`。`UQEActionPolicy` は `{現在のuqe_tau, config_.uqe_use_tail_mean}` を返し、`ThompsonSamplingActionPolicy` は `std::nullopt` を返す（Thompsonのscoreは乱択tausの平均）。target policyはspatial explorationを持たないためscalar tauで足りる。現在のtauは既存の `OnLearn` によるexp_step減衰をそのまま使う。
- `MakeRiskBiasedScore(float tau, bool use_tail_mean, const torch::Tensor& quantiles)` は `UQEActionPolicy::MakeUQEValues` の本体を `anet::rl::dqn` 名前空間のfree関数へ抽出したもので、policy側のprivateメソッドはそのforwarderにする。スコア定義はhard経路と1箇所で共有する。この抽出はOFF経路のコードを書き換えるが数値とRNGは不変で、D8の範囲内である。
- Learner構築時の既存初期化ログ（`Initialized IQNLearner (...)` 系）へ、Munchausen ON時は `log_policy_mode` とスコア源（`mean` / `risk_biased(tau=..., tail_mean=...)`）を追記する。新規のログ機構やWARNは足さない。

TDのON経路はsoft scalar targetを作り、以降のTD error、clip、Huber、PER処理は既存経路へ戻す。QR / IQNのON経路は全行動・全分位点から実空間の `soft_dist[B,M]` を作り、ON専用の `CalcMunchausenTargetQuantiles(samples, soft_dist, bonus)` でreturn、bonus、terminal mask、`gamma^n` を合成し、完成後にだけ `h` を適用する。既存 `CalcTargetQuantiles` は内部で `h^-1` を適用するためOFF専用のままとし、ON経路から呼ばない。これによりTBO時の `h^-1` 二重適用を構造的に防ぐ。target完成後は既存lossとPER優先度計算へ戻す。ON時はargmax用の `SelectTargetActions` / `target_policy_->SelectAction` を呼ばない（`target_policy_` はスコア仕様の問い合わせにだけ使う）。

### 4. Actor Qヒント

Actor Qヒントは常時K3とする。

```text
[q_sa, state_value, munchausen_term]
```

- `q_sa`: 実行行動のActor score。
- `state_value`: OFFではmax Q、ONではsoft価値。TBO有効時は推定器へ渡すQ空間に戻して保持する。
- `munchausen_term`: OFFでは0、ONでは実行行動のclip済みscaled log-policy bonus。

Actorへ渡す設定は次の狭いvalue objectに限定する。

```cpp
struct ActorQHintConfig {
    bool enabled;
    float alpha;
    float entropy_tau;
    float clip_value_min;
    bool use_tbo;
    float tbo_epsilon;
};
```

Learner config全体や `log_policy_mode` は渡さない。Actorはmodeに関係なくTrain Actor snapshotのonline scoreから近似し、追加forwardは行わない。通常Q/QRは同一forwardの平均Q、IQN+UQEは同一forwardのrisk-biased action scoreを使う。

Actorは行動ごとのMunchausen項を一時auxとして保持する。`DQNActionInfo::WithAction` は行動差し替え時に `q_sa` と `munchausen_term` を再gatherし、`state_value` を維持する。hintがあるのに必要なauxが欠けていれば契約違反とする。

`DqnInitialPriorityEstimator` は次を計算する。

```text
target = target_return + start.munchausen_term
if not terminal:
    target += discount * h^-1(boot.state_value)
if use_tbo:
    target = h(target)
```

codecとvalidationはK3だけを受理し、全列finiteを要求する。旧K2 payloadはschema違反としてfail-fastし、互換分岐は持たない。

### 5. OFF保証

- `enabled=false` のLearnerはMunchausen分岐へ入らず、数値経路とRNG消費を実装前から完全に変えない。
- 標準Atari構成はmax初期優先度であり、Actor hint経路を有効化しない。実装前後のmetrics系列一致で保証する。
- `actor_approx + OFF` は初期優先度の数値だけを従来と同値にする。K3 transport、ゼロの第3列、一時aux生成、命令列は変化してよい。
- RainbowはMunchausenアルゴリズムOFFを保証するが、共有codec/transportがK3になることは許容する。

### 6. 診断とreadback

raw 5値を `BatchUpdateResult` へ運び、2つのEMA購読を加えて7行とする。

| GetScalar key | index | 契約 |
|---|---:|---|
| `munchausen_scaled_log_policy_mean` | 0 | 実行行動のclip前scaled log-policy平均。`<= 0`。 |
| `munchausen_clip_ratio` | 1 | 下限clipが発生した割合。`[0,1]`。 |
| `munchausen_bonus_mean` | 2 | targetへ加えたbonus平均。`[alpha*l0,0]`。 |
| `munchausen_next_entropy` | 3 | next policy entropy平均。`[0,ln A]`。 |
| `munchausen_soft_gap` | 4 | `V_soft - max_a q_mean_real` の平均（常に分位点平均基準）。方策スコアが平均なら `[0,tau*ln A]`、risk-biased（D15）なら負になり得るためfiniteのみ。 |

浮動小数点誤差を考慮した許容差を設ける。OFF時も既知keyとして扱い、値未成立を `NaN` で返す。

固定index readbackへ次の専用契約を追加する。

- `PerPriorityUpdatePending::munchausen_diagnostics_count`
- `PerPriorityUpdateInfo::munchausen_diagnostics`
- `BatchUpdateResult::munchausen_diagnostics`

readbackの並びは、priority、IQN diagnostics、Munchausen diagnostics、upper-tail統計とする。`PreparePerPriorityUpdate` に `munchausen_diagnostics` 引数を追加し、早期returnは `!use_per && !iqn_diagnostics.defined() && !munchausen_diagnostics.defined()` の場合だけとする。Munchausen diagnosticsだけが定義済みでもpendingを有効化する。`PreparePerPriorityUpdate` でcountとoffsetを決め、`ApplyPerPriorityUpdate` と結果生成で同じ並びを使う。これによりPERがOFFでもMunchausen diagnosticsをreadbackできることを保証する。

metricsは次の7 tagとする。

```text
metrics.scalar.@munchausen.[36_agent_munchausen/01_scaled_logp_mean] = munchausen_scaled_log_policy_mean @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/02_scaled_logp_mean_ema] = munchausen_scaled_log_policy_mean @learn $update_result $ema
metrics.scalar.@munchausen.[36_agent_munchausen/03_clip_ratio] = munchausen_clip_ratio @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/04_bonus_mean] = munchausen_bonus_mean @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/05_bonus_mean_ema] = munchausen_bonus_mean @learn $update_result $ema
metrics.scalar.@munchausen.[36_agent_munchausen/06_next_entropy] = munchausen_next_entropy @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/07_soft_gap] = munchausen_soft_gap @learn $update_result
```

### 7. ProfileRange

- `forward_target`: 通常のtarget forward。`target` modeでは `obs ∥ next_obs` の2B forward全体。
- `forward_munchausen_online`: `online` modeだけのNoGrad・eval fresh online forward。
- `munchausen_target`: FP32実空間化、方策・bonus・soft価値、target組立。

`online_reuse` は追加forward phaseを持たない。既存の連続phase規約に従い、計測のためだけに不自然なscopeを作らない。

### 8. 設定プロファイル

`apps/runner/config/agent.txt` のbaselineへ5値を明示する。`@munchausen` は[done/059](done/059_config_concept_tree_alignment_10prd.md)が定義した、NN配線を持たずALGO軸と直交するTARGET軸であり、本PRDが同文書の遅延ゲートを発動する。

```text
DefaultDQNAgent.@baseline.learner.munchausen.enabled = false
DefaultDQNAgent.@baseline.learner.munchausen.log_policy_mode = target # target / online / online_reuse
DefaultDQNAgent.@baseline.learner.munchausen.alpha = 0.9
DefaultDQNAgent.@baseline.learner.munchausen.entropy_tau = 0.03
DefaultDQNAgent.@baseline.learner.munchausen.clip_value_min = -1.0
```

標準 `@munchausen` profileは競合する `use_double_dqn` を明示的に無効化し、単体で正常構成を作る。`use_optimistic_target` は競合ではないため書かず、`@baseline` の `false`（平均スコア）を既定とする。

```text
DefaultDQNAgent.@munchausen.learner.munchausen.enabled = true
DefaultDQNAgent.@munchausen.learner.munchausen.log_policy_mode = target
DefaultDQNAgent.@munchausen.learner.use_double_dqn = false
```

soft楽観ターゲット（D15）の腕は新しいRun profileを置かず、`run.@munchausen` に後段overlayで `A3.use_optimistic_target = true`（train_policyがUQEの構成）または `A3.target_policy.policy_type = UQE` を足して作る。

下記Atari Run chainはbaselineより後で `@munchausen` を適用し、その後のA1/A2/A3は `use_double_dqn` と `use_optimistic_target` を再定義しない。実行時はeffective configと `config_resolution.json` の両方で解決結果を確認する。

他envへ `@munchausen` を組み込む場合は、後段overlayを含む最終effective configで `learner.use_double_dqn=false` と、`use_optimistic_target` の最終値を確認する。`use_optimistic_target` はfail-fastしないため、最終leafが `true` なら黙ってsoft楽観ターゲットになる。たとえば現行DropMergeは `A1.use_optimistic_target=true` を後段の `A2.use_optimistic_target=false` が戻しており、A2側を外すとrisk-biasedスコアの構成へ変わる。既存A層を一括変更せず、利用するRunごとに最終解決値と初期化ログのスコア源を確認する。

`metrics.scalar.@munchausen` は上記7 tagを購読し、`run.@munchausen` がagent profileとmetrics profileを束ねる。mode差分Runは解決後leafの `log_policy_mode` を明示的に上書きする。

```text
run.@munchausen : AtariEnv.$ = AtariEnv.@v5_noop30 > E1
run.@munchausen : DefaultDQNAgent.$ = @baseline > @iqn > @munchausen > A1 > @bf16 > A2 > A3
run.@munchausen : DefaultDQNAgent.net.branch.[main_feature].$ = net.branch.@AtariImpalaX2
run.@munchausen : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@munchausen > M1 > M2
run.@munchausen : app.run_name = run_{t}_munchausen_${E1.game}
```

## テスト契約

productionへtest-only APIを追加せず、既存のforward-count probe、network mode観測、TauEcho fixtureを利用する。

1. TD / QR / IQN × `target` / `online` / `online_reuse` の9組について、既知値targetを検証する。
2. `target` が2B target forward 1回、`online` が既存forward後のNoGrad・eval online forward 1回、`online_reuse` が追加forwardなしであることを検証する。
3. IQNのtaus shape、current/target/fresh-onlineの生成規則、生成順、各modeの追加RNG有無を検証する。
4. N-step returnへのbonus 1回加算、`gamma^n`、terminalでbonusを残してbootstrapだけを消すことを検証する。
5. clip境界、`alpha=0`、TBO ON/OFF、TD/QR/IQNのFP32実空間計算を検証する。quantileのTBO ONではON専用target組立が `h^-1` を一度だけ適用し、既存 `CalcTargetQuantiles` を経由しないことを既知値で確認する。
6. `tau -> 0` のmax bootstrap極限は `alpha=0` と組み合わせ、残存bonusを比較へ混ぜない。
7. CUDAが利用可能な環境ではBF16 autocast下でもtargetとdiagnosticsがFP32計算になることを検証する。
8. mode不正、alpha不正、entropy tau不正、clip下限不正を `enabled` に関係なくfail-fastさせる。
9. Munchausen ON + `use_double_dqn=true` を構築エラーとして検証する。メッセージの両キー、指定値、期待値も確認する。
10. Munchausen OFFでは `use_double_dqn=true` と `use_optimistic_target=true` がそれぞれ許可されることを確認する。Munchausen ON + `use_optimistic_target=true`（train_policy=UQE）は許可され、方策スコアがrisk-biasedになることを確認する。
11. diagnostics readbackがPER ON/OFFの両方で正しいoffsetと5値を返すことを確認する。PER OFFかつMunchausen diagnosticsだけが定義された場合も早期returnせず、pendingが有効になることを確認する。
12. diagnosticsについてfiniteに加え、scaled log-policy `<= 0`、clip ratio `[0,1]`、bonus `[alpha*l0,0]`、entropy `[0,ln A]` を許容差付きで検証する。soft gap `[0,tau*ln A]` は方策スコアが平均のときだけ検証し、risk-biased時はfiniteのみとする。
13. K3 pack/decode round-trip、旧K2拒否、全列finite、`WithAction` の再gather、aux欠落時のfail-fastを検証する。
14. `DqnInitialPriorityEstimator` がMunchausen込みtargetをTBO ON/OFFで再現し、OFF時の初期優先度数値が従来と一致することを検証する。
15. 同じseedで各ON modeを2回実行し、各mode内のloss/TD error系列が再現することを確認する。
16. 既存DQNテストとRainbowのMunchausen OFFを確認する。transportのK3化に伴う期待値更新は許容する。
17. `target` modeでplasticity targetを購読し、2B forwardのcaptureがB行へnarrowされ、各行が後半の `next_obs` に対応することを確認する。通常target forwardと他modeのcapture shape・意味は変更しない。
18. `MakeRiskBiasedScore` が、同じquantile tensorに対して `UQEActionPolicy` の `uqe_values` aux（QRの固定分位とIQN）を再現することを確認する。抽出リファクタ前後でhard経路のスコアが同値であることの担保である。
19. soft楽観ターゲットの既知値を検証する。QR × IQN × `uqe_use_tail_mean` true/false × `target` / `online_reuse` について、oracleはtestコード側で値ソート→index→tail平均または1点のスコアから方策を作り、全分布を混合する。さらに `tau -> 0`（`alpha=0`）でsoft楽観ターゲットが実空間のhard楽観ターゲット（risk-biased argmaxの行動の全分布 + return + `gamma^n` mask）に許容差内で一致することを確認する。
20. `target_policy` がGreedy / EpsilonGreedy / ThompsonSamplingのとき、Munchausen ONのtargetが平均スコア経路と同値であることを確認する。

## 実装時の受入基準

### 1. OFF比較

実装前base commitと実装後OFFで各1本、次の固定条件を使う。

```text
Environment: Breakout
train.seed: 1
max_exp_step: 400k
warmup_exp_step: 200k
backend: backend.@deterministic
run base: run.@v5_iqn_impala_x2
```

`37_agent_qtd/*` と `38_agent_loss/*` を `{tag,step,value}` に正規化し、各系列のSHA-256が一致することを必須とする。checkpoint raw hashはserialize非決定性のため使わない。標準max初期化では定数0となる `39_agent_per/05_sample_actor_init_ratio` は比較対象へ含めない。

### 2. 単体テスト

テスト契約の全項目と既存DQN関連テストが通ること。productionへtest-only APIを追加しないこと。

### 3. ON smoke

通常backend、Breakout、seed 1、400k exp step、warmup 200kで次の5本を実行する。

- `target`
- `online`
- `online_reuse`
- `target + per_initial_priority_mode=actor_approx`
- `target + use_optimistic_target=true + train_policy.policy_type=UQE`（soft楽観ターゲット。D15）

各Runでeffective configとresolutionを確認し、`learner.use_double_dqn=false`、意図した `use_optimistic_target`（5本目だけ `true`、他は `false`）、意図した `log_policy_mode` が最終leafであることを確認する。5本目では `target_policy.policy_type=UQE` が解決され、初期化ログのスコア源が `risk_biased` で、`07_soft_gap` は負でもよい（finiteのみ）。短縮後の `01_scaled_logp_mean` / `02_scaled_logp_mean_ema` を含む7つの診断tagが `status=ok`、count > 0、finite、契約範囲内であること、lossがfiniteであることを確認する。actor_approx Runでは `39_agent_per/05_sample_actor_init_ratio` が非ゼロで、`52_actor_learner_pair_count` が有効であることも確認する。

5本は確認用の使い捨てRunなので、恒久的な `run.@munchausen` の `app.run_name` をそのまま使わず、CLIで順に `run_{t}_tmp_smoke_067_target_${E1.game}`、`run_{t}_tmp_smoke_067_online_${E1.game}`、`run_{t}_tmp_smoke_067_online_reuse_${E1.game}`、`run_{t}_tmp_smoke_067_target_actor_approx_${E1.game}`、`run_{t}_tmp_smoke_067_target_risk_${E1.game}` へ上書きする。

### 4. Throughput記録

ON smokeの各modeについて250k〜400k exp step区間を使い、`150k / elapsed time差` で求めた実throughputと `exp_step_per_sec` を記録する。さらに `forward_target`、`forward_munchausen_online`、`munchausen_target` のProfileRangeを記録する。これは比較記録であり、2%などの合否閾値、複数回平均、round-robin配置は要求しない。

### 5. 非ゲート項目

action gap、policy churn、score改善は本PRDの合否に含めない。`action_mask` 対応も別作業とする。

## docs同期契約

本PRDの実装時に、production code・config・テストと同じ変更内で次の現行設計文書を同期する。

- `docs/design/200_dqn_agents.jp.md`: 3 mode、3 Learner target、K3 Actor Qヒント、設定、診断。
- `docs/design/030_user_guide_analysis.jp.md`: 7つのMunchausen診断tagの読み方。

履歴資料 `docs/memo/done/035_approx_actor_priority_per_10prd.md` は当時の記録として保持し、改訂しない。

## 現行実装で確認済みの前提

- target networkはeval modeで使われる。
- `ForwardOnlineWithTrain` はonline networkを一時的にtrain modeでforwardし、`ForwardOnline` はeval modeでforwardする。
- 3 Learnerともcurrent online出力を既に持つため、`online_reuse` はdetach再利用できる。
- IQNのTauGeneratorは行ごとに独立してtausを生成するため、target modeの2B生成で前後batchを表現できる。
- `PreparePerPriorityUpdate` の固定index readbackは現状IQN diagnosticsとupper-tail統計を扱うが、Munchausen専用count/offsetはまだない。さらに `!use_per && !iqn_diagnostics.defined()` で早期returnするため、PER OFFのMunchausen diagnosticsを運ぶにはこのゲートも変更する必要がある。
- `ForwardTarget` はtarget側branch captureを常に同じforwardへ渡す。`target` modeの2B forwardをそのまま使うとcaptureも2Bになるため、PRD 062の既存意味を保つには後半Bへのnarrowが必要である。
- ReplayBuffer共通層はhint幅を動的に運び、K3は既存inline capacity内に収まる。
- BTRの非IQN経路はtarget current、IQN経路はfresh online currentでbonusを計算し、集約済みN-step returnへbonusを一度加え `gamma^n` bootstrapを使う。BTRに `online_reuse` の前例があるとは主張しない。
- 論文著者のGoogle Research参照実装はDopamineを基盤とするが、本PRDの一次根拠は論文と補遺である。
- hard経路の `use_optimistic_target=true` は、`target_policy_->SelectAction(next_obs, greedy_only=true, ...)` でrisk-biased argmaxを行う。`greedy_only=true` はεをゼロにするだけでUQEのrisk-biased選択は維持される（`dqn_based_agent.cpp:1482-1486`）。`:3038` 付近の「greedy_only=falseにすることで…」というコメントは実装と逆であり、本PRDの実装時に同じ変更でコメントだけ訂正する（数値・RNGに影響せずD8の範囲内）。
- target policyの `uqe_tau` は `OnLearn` でtrain policyと同じexp_stepスケジュールで減衰し（`default_dqn_agent.cpp:583`）、`ActionPolicy::GetScalar("uqe_tau")` が現在値を返す。`UQEActionPolicy::MakeUQEValues`（`dqn_based_agent.cpp:1371-1403`）はprivateで、分位点を値で昇順ソートしてから `floor(tau*(N-1))` を起点にtail平均または1点を取る。
- `target_policy` の既定はGreedyで、Atariの現行構成はtrain/evalともEpsilonGreedy・`use_optimistic_target=false` である。soft楽観ターゲットが実際に効くのはtrain_policy=UQEのDropMerge / LunarLander / GridMaze系構成である。

実装時にはline numberと現行APIを再確認する。

## リスクと制約

- `target` はtarget forwardを2Bへ増やし、`online` は追加online forwardとIQN tau RNGを増やす。Learner律速のAtariではwall-clockへ現れる可能性があるため、合否ではなくmode別に記録する。
- `online_reuse` は追加費用を避ける一方、train-mode出力を使う。`online` との意味差を設定名と計測で可視化する。
- Actor hintはLearner modeにかかわらずonline近似であり、target modeと厳密一致しない。初回sampling前の順位付けという責務に限定して許容する。
- IQN+UQEのActor hintはrisk-biased action scoreを使う近似である。
- `action_mask` を持つ環境では非合法行動をsoft価値へ含め得る。既知の未対応事項であり、現行実装より相対的に悪化しないとは主張しない。
- `entropy_tau=0.03` は報酬スケール前提であり、異なる報酬スケールでの妥当性は利用側が判断する。
- soft楽観ターゲット（D15）ではbonus差がrisk-biasedスコア差に比例し、暗黙KLの基準がrisk方策になる。論文のaction gap拡大やpolicy churn抑制の保証は失う。効果の有無は本PRDでは測らない。
- soft経路のrisk-biasedスコアはM本の経験分位で、Atariの `M=8` では粗い。`target_taus.sample_mode=random` では揺れ、`fixed` / `stratified` で決定的になる。
- 同じ `target_policy.uqe_use_tail_mean` がhard / soft両経路を支配する。hard経路のTBO都合で `false` にした設定はsoft経路でも1点UQEになる。
- `use_optimistic_target=true` はfail-fastしないため、env側A層に残った `true` が黙ってsoft楽観ターゲットになる。config dumpと初期化ログのスコア源で確認する。

## Complexity Audit

### Keep

- TD / QR / IQNの3 Learner対応。
- `target` / `online` / `online_reuse` の3 mode。`online_reuse` は、Atari RR1が計測上Learner-boundであり、source選択と追加forward費用を分離するために残す。
- 常時K3のActor Qヒント。
- raw 5値 + EMA 2行の診断。
- `use_double_dqn=true` との併用の構築時fail-fast。
- mode別ProfileRangeとthroughput記録。
- 1フェーズ実装。TD / QRも、現用のNature DQN / QR control profileが存在するため対象から外さない。
- soft楽観ターゲット（D15）。新キーなしで `target_policy` のスコアをsoft経路が尊重する。根拠は設計整合と後付け実装を避けるユーザー判断であり、RL効果の実測ではない（2026-09-05追補）。
- `ActionPolicy::GetRiskScoreSpec()` seam。切るとtau減衰をLearnerへ二重実装することになる。
- `MakeRiskBiasedScore` の抽出。切るとスコア定義が2箇所に複製され乖離する。

### Shrink

- OFF保証を、Learner数値経路・RNG、標準Atari、actor_approxの優先度数値同値へ限定する。
- `CONTEXT.md` を純粋なドメイン用語集へ戻し、config・shape・forward・TBO手順をPRD / ADRへ集約する。
- soft楽観ターゲットのスコア源表示は、新規ログではなく既存のLearner初期化ログへの追記に留める。

### Defer

- M-SAC向け共通module抽出。SACのActor/Critic契約が成立し、同じ処理が実際に2利用箇所になってから再検討する。
- `action_mask` 対応。
- action gap、churn、scoreによる成績評価。
- soft楽観ターゲットの効果実験。`use_optimistic_target` on/offをround-robin・複数seedで再測定して効果が出た時点で、soft版との比較へ進む。
- IQNで厳密な `Z_tau` を得る追加tau列。Mが小さい構成で経験分位が粗くて困った時点で検討する。

### Cut

- 履歴資料 `done/035` の編集。
- 2% throughput合否ゲート。
- 旧設定キーと互換処理。
- `use_double_dqn=true` との併用を許容して警告だけ出す契約。
- `munchausen.*` 配下の独立キー（`policy_score` / `risk_tau` / `risk_use_tail_mean`）。同じ概念の二重表現になり、tau減衰も持てない。
- risk-biased専用の追加メトリクス。
- `action_mask` についての「現行比で相対劣化しない」という主張。

## 影響ファイル（実装時）

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/agent.hpp` | `LearnerConfig::MunchausenConfig` |
| `core/anet-core/include/anet/default_dqn_agent.hpp` | 5キーの読取・検証、`use_double_dqn` 併用のfail-fast |
| `core/anet-core/src/default_dqn_agent.cpp` | 狭い `ActorQHintConfig` の組立、Learner初期化ログへmunchausen modeとスコア源を追記 |
| `core/anet-core/src/rainbow_agent.cpp` | `Actor` 構築時にMunchausen OFFの `ActorQHintConfig` を渡し、共通K3 transportへ追従 |
| `core/anet-core/src/dqn_based_agent.hpp` / `.cpp` | 3 Learner、3 mode、実空間helper、K3 hint、診断readback、ProfileRange、`ActionPolicy::GetRiskScoreSpec`、`MakeRiskBiasedScore` の抽出（policyはforwarder）、`:3038` 付近のコメント訂正 |
| `core/anet-core/src/dqn_based_agent_test.cpp` | テスト契約の実装 |
| `apps/runner/config/agent.txt` | baselineと `@munchausen` profile |
| `apps/runner/config/metrics_scalar.txt` | 診断7 tag |
| `apps/runner/config/Atari.txt` | `run.@munchausen` |
| `docs/design/200_dqn_agents.jp.md` | 実装後の現行設計へ同期 |
| `docs/design/030_user_guide_analysis.jp.md` | 診断の読み方へ同期 |

## 今回の文書改訂範囲

- `docs/memo/067_MunchausenRL_10prd.md`
- `docs/adr/0035-munchausen-target-learner-local-real-space.md`
- `docs/adr/0036-actor-q-hint-three-columns-munchausen.md`
- `CONTEXT.md`
