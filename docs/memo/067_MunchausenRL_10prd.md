# Munchausen RL（M-DQN / M-QR / M-IQN）PRD

> 起点: 2026-09のAtari BTR差分キャンペーン。BTRは `gamma=0.997` とMunchausenを同時に採用するが、本コードベースはgammaだけを採用している。
>
> 一次根拠: [Munchausen Reinforcement Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf)、[Supplementary Material](https://papers.nips.cc/paper_files/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Supplemental.pdf)
>
> 関連決定: [ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)、[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)、[done/059](done/059_config_concept_tree_alignment_10prd.md)（TARGET軸の配置と遅延ゲート）、[999_noisynet](999_noisynet_10prd.md)（BTR採用部品のうち別途扱う未実装機能）
>
> 履歴資料: [done/035](done/035_approx_actor_priority_per_10prd.md)は当時のK2契約を記録した資料として変更しない。

## Context（背景・目的）

Munchausen RLは、Bellmanターゲットの報酬側へエージェント自身のscaled log-policyを加え、次状態のhard argmax bootstrapをsoft価値へ置き換える。NeurIPS論文はM-DQNを1-step、M-IQNを3-stepで評価しているため、本PRDでは論文の1-step式とanet-labのN-step target returnを一括して同一視しない。補遺の分位点ごとの方策混合を根拠にしつつ、bonusのN-step上の帰属はBTR互換の拡張として明示する。

本PRDの起点は、`gamma=0.997` がaction gapを犠牲にして地平を延ばしているという[baseline探索ブロック19](../experiments/default-dqn/atari/2026-08-17_baseline.md)と、補償器であるMunchausenが未実装だと整理した[可塑性保護screening](../experiments/default-dqn/atari/2026-08-30_protection-screening.md)である。[BTR survey Table 2](../../reports/btr_hyperparams_survey_2026-08-26.md)にはMunchausen除去時のAction GapとPolicy Churnの差が記録されている。一方、[Atari実験README](../experiments/default-dqn/atari/README.md)は定常 `q_gap` を成績の予測子として採用しないと裁定しているため、本PRDもaction gapやscoreを合否ゲートにしない。

本PRDの目的は、Munchausen RLをDQNBasedの3 Learnerに共通する既定OFFの契約として実装可能な状態へ確定することである。対象は `TDLearner`、`QRLearner`、`IQNLearner` と、近似Actor初期優先度を成立させるActor Qヒントである。性能改善やスコア改善の証明ではなく、数理・設定・transport・診断・検証の契約を固定する。

実装は1フェーズで行う。今回の改訂作業は文書のみであり、production code、config、テスト、現行実装を説明する `docs/design` は変更しない。

## Goals

- TD / QR / IQNで同じMunchausen意味論を提供する。
- TBOやAMPの有無にかかわらず、方策温度を報酬スケール上で解釈できるようFP32実空間で計算する。
- `target`、`online`、`online_reuse` の3 modeを、forward mode・forward回数・IQN RNG消費まで含めて区別する。
- 明示したが効果を持たない競合設定を構築時にfail-fastする。
- `actor_approx` の初期優先度をMunchausen targetと同型にしつつ、ActorへLearnerの責務を漏らさない。
- OFF時の保証範囲を、実際に価値があり検証可能な範囲へ限定する。

## Non-Goals

- SAC / M-SAC向け共通moduleの先行抽出。
- RainbowAgentへのMunchausen設定公開。
- `action_mask` 対応。
- softmax行動など探索方策の変更。
- M-VI、q-Munchausenなどの派生方式。
- action gap、policy churn、score、throughputを用いた採否判定。
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

QR / IQNでは、全分位点に共通する方策を実空間分位点の平均から作り、同じ方策で各分位点を混合する。

```text
q_next_real[i,a] = mean_j Z_next_real[i,a,j]
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

## 決定事項

| # | 論点 | 決定 |
|---|---|---|
| D1 | 対象 | DQNBased共通の `TDLearner` / `QRLearner` / `IQNLearner`。既定OFF、実装は1フェーズ。 |
| D2 | log-policy mode | 閉じた文字列enum `target` / `online` / `online_reuse`。既定は `target`。 |
| D3 | N-step | bonusは集約済みreturnの先頭へ1回だけ加え、終端でも残す。bootstrapだけをmaskし `gamma^n` を掛けるBTR互換拡張。 |
| D4 | 数値空間 | TD / QR / IQNすべてFP32実空間。TBOは分位点ごとの `h^-1` とtarget完成後の `h`。 |
| D5 | 競合設定 | Munchausen ONでDouble DQNまたはoptimistic targetがONなら、構築時に `ANET_SYSTEM_ERROR`。disabled時は許可。 |
| D6 | Actor hint | 常時K3 `[q_sa, state_value, munchausen_term]`。旧K2はschema違反。 |
| D7 | Actor config | 狭い `ActorQHintConfig` だけを渡す。Learner config全体とmodeは渡さない。 |
| D8 | OFF保証 | Learner数値経路・RNGと標準Atari構成の完全不変。actor_approxは優先度数値同値のみ。 |
| D9 | 診断 | raw 5値とEMA 2行。固定index readbackへ専用count・offset・結果fieldを持つ。 |
| D10 | 計測 | `forward_target`、`forward_munchausen_online`、`munchausen_target` を区別する。 |
| D11 | Rainbow | MunchausenアルゴリズムはOFF。共通transportがK3になることは許容する。 |
| D12 | action mask | 既知の未対応事項として記録し、現行比の相対安全性は主張しない。 |
| D13 | M-SAC | SACのActor/Critic契約成立後、共通処理が実際に2利用箇所になった時点で抽出を再検討する。 |
| D14 | 成績・性能 | runtimeと性能値は記録するが、合否閾値や成績ゲートは置かない。 |

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

Munchausen ON時は次の組み合わせを構築時に別々にfail-fastする。

- `learner.munchausen.enabled=true` と `learner.use_double_dqn=true`
- `learner.munchausen.enabled=true` と `use_optimistic_target=true`

各エラーには競合する両キー、実際の指定値 `true`、期待値 `false` を含める。Munchausen OFF時は両機能を従来どおり許可する。

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

TDのON経路はsoft scalar targetを作り、以降のTD error、clip、Huber、PER処理は既存経路へ戻す。QR / IQNのON経路は全行動・全分位点から実空間の `soft_dist[B,M]` を作り、ON専用の `CalcMunchausenTargetQuantiles(samples, soft_dist, bonus)` でreturn、bonus、terminal mask、`gamma^n` を合成し、完成後にだけ `h` を適用する。既存 `CalcTargetQuantiles` は内部で `h^-1` を適用するためOFF専用のままとし、ON経路から呼ばない。これによりTBO時の `h^-1` 二重適用を構造的に防ぐ。target完成後は既存lossとPER優先度計算へ戻す。ON時はargmax用の `SelectTargetActions` / `target_policy` を呼ばない。

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
| `munchausen_soft_gap` | 4 | `V_soft - max(Q)` の平均。`[0,tau*ln A]`。 |

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

標準 `@munchausen` profileは競合設定を明示的に無効化し、単体で正常構成を作る。

```text
DefaultDQNAgent.@munchausen.learner.munchausen.enabled = true
DefaultDQNAgent.@munchausen.learner.munchausen.log_policy_mode = target
DefaultDQNAgent.@munchausen.learner.use_double_dqn = false
DefaultDQNAgent.@munchausen.use_optimistic_target = false
```

下記Atari Run chainはbaselineより後で `@munchausen` を適用し、その後のA1/A2/A3は両競合キーを再定義しない。実行時はeffective configと `config_resolution.json` の両方で解決結果を確認する。

他envへ `@munchausen` を組み込む場合は、後段overlayを含む最終effective configで `learner.use_double_dqn=false` と `use_optimistic_target=false` を確認する。たとえば現行DropMergeは `A1.use_optimistic_target=true` を後段の `A2.use_optimistic_target=false` が戻すことで成立しており、A2側を外すとMunchausen構成は意図どおりfail-fastする。既存A層を一括変更せず、利用するRunごとに最終解決値を確認する。

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
9. Munchausen ON + `use_double_dqn=true` と、Munchausen ON + `use_optimistic_target=true` を別々の構築エラーとして検証する。メッセージの両キー、指定値、期待値も確認する。
10. Munchausen OFFでは `use_double_dqn=true` と `use_optimistic_target=true` がそれぞれ許可されることを確認する。
11. diagnostics readbackがPER ON/OFFの両方で正しいoffsetと5値を返すことを確認する。PER OFFかつMunchausen diagnosticsだけが定義された場合も早期returnせず、pendingが有効になることを確認する。
12. diagnosticsについてfiniteに加え、scaled log-policy `<= 0`、clip ratio `[0,1]`、bonus `[alpha*l0,0]`、entropy `[0,ln A]`、soft gap `[0,tau*ln A]` を許容差付きで検証する。
13. K3 pack/decode round-trip、旧K2拒否、全列finite、`WithAction` の再gather、aux欠落時のfail-fastを検証する。
14. `DqnInitialPriorityEstimator` がMunchausen込みtargetをTBO ON/OFFで再現し、OFF時の初期優先度数値が従来と一致することを検証する。
15. 同じseedで各ON modeを2回実行し、各mode内のloss/TD error系列が再現することを確認する。
16. 既存DQNテストとRainbowのMunchausen OFFを確認する。transportのK3化に伴う期待値更新は許容する。
17. `target` modeでplasticity targetを購読し、2B forwardのcaptureがB行へnarrowされ、各行が後半の `next_obs` に対応することを確認する。通常target forwardと他modeのcapture shape・意味は変更しない。

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

通常backend、Breakout、seed 1、400k exp step、warmup 200kで次の4本を実行する。

- `target`
- `online`
- `online_reuse`
- `target + per_initial_priority_mode=actor_approx`

各Runでeffective configとresolutionを確認し、`learner.use_double_dqn=false`、`use_optimistic_target=false`、意図した `log_policy_mode` が最終leafであることを確認する。短縮後の `01_scaled_logp_mean` / `02_scaled_logp_mean_ema` を含む7つの診断tagが `status=ok`、count > 0、finite、契約範囲内であること、lossがfiniteであることを確認する。actor_approx Runでは `39_agent_per/05_sample_actor_init_ratio` が非ゼロで、`52_actor_learner_pair_count` が有効であることも確認する。

4本は確認用の使い捨てRunなので、恒久的な `run.@munchausen` の `app.run_name` をそのまま使わず、CLIで順に `run_{t}_tmp_smoke_067_target_${E1.game}`、`run_{t}_tmp_smoke_067_online_${E1.game}`、`run_{t}_tmp_smoke_067_online_reuse_${E1.game}`、`run_{t}_tmp_smoke_067_target_actor_approx_${E1.game}` へ上書きする。

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

実装時にはline numberと現行APIを再確認する。

## リスクと制約

- `target` はtarget forwardを2Bへ増やし、`online` は追加online forwardとIQN tau RNGを増やす。Learner律速のAtariではwall-clockへ現れる可能性があるため、合否ではなくmode別に記録する。
- `online_reuse` は追加費用を避ける一方、train-mode出力を使う。`online` との意味差を設定名と計測で可視化する。
- Actor hintはLearner modeにかかわらずonline近似であり、target modeと厳密一致しない。初回sampling前の順位付けという責務に限定して許容する。
- IQN+UQEのActor hintはrisk-biased action scoreを使う近似である。
- `action_mask` を持つ環境では非合法行動をsoft価値へ含め得る。既知の未対応事項であり、現行実装より相対的に悪化しないとは主張しない。
- `entropy_tau=0.03` は報酬スケール前提であり、異なる報酬スケールでの妥当性は利用側が判断する。

## Complexity Audit

### Keep

- TD / QR / IQNの3 Learner対応。
- `target` / `online` / `online_reuse` の3 mode。`online_reuse` は、Atari RR1が計測上Learner-boundであり、source選択と追加forward費用を分離するために残す。
- 常時K3のActor Qヒント。
- raw 5値 + EMA 2行の診断。
- 競合設定の構築時fail-fast。
- mode別ProfileRangeとthroughput記録。
- 1フェーズ実装。TD / QRも、現用のNature DQN / QR control profileが存在するため対象から外さない。

### Shrink

- OFF保証を、Learner数値経路・RNG、標準Atari、actor_approxの優先度数値同値へ限定する。
- `CONTEXT.md` を純粋なドメイン用語集へ戻し、config・shape・forward・TBO手順をPRD / ADRへ集約する。

### Defer

- M-SAC向け共通module抽出。SACのActor/Critic契約が成立し、同じ処理が実際に2利用箇所になってから再検討する。
- `action_mask` 対応。
- action gap、churn、scoreによる成績評価。

### Cut

- 履歴資料 `done/035` の編集。
- 2% throughput合否ゲート。
- 旧設定キーと互換処理。
- 競合設定を許容して警告だけ出す契約。
- `action_mask` についての「現行比で相対劣化しない」という主張。

## 影響ファイル（実装時）

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/agent.hpp` | `LearnerConfig::MunchausenConfig` |
| `core/anet-core/include/anet/default_dqn_agent.hpp` | 5キーの読取・検証、競合組み合わせのfail-fast |
| `core/anet-core/src/default_dqn_agent.cpp` | 狭い `ActorQHintConfig` の組立 |
| `core/anet-core/src/rainbow_agent.cpp` | `Actor` 構築時にMunchausen OFFの `ActorQHintConfig` を渡し、共通K3 transportへ追従 |
| `core/anet-core/src/dqn_based_agent.hpp` / `.cpp` | 3 Learner、3 mode、実空間helper、K3 hint、診断readback、ProfileRange |
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

production code、config、テスト、`docs/design`、履歴資料は変更しない。
