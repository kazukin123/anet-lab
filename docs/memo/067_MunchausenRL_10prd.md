# Munchausen RL（M-DQN / M-QR / M-IQN）PRD

> 起点: 2026-09 の Atari BTR 差分キャンペーン。BTR は γ 0.997 と Munchausen（action gap を広げる補償器）を同時に採るが、
> 本コードベースは γ だけを採っている（`apps/runner/config/Atari.txt` の `run.@g99` 注記、
> [2026-08-30_protection-screening.md](../experiments/default-dqn/atari/2026-08-30_protection-screening.md) 次の検証
> 「`18_q_gap_rel` が動かない理由 … 補償器である Munchausen が未実装」）。BTR の ablation では Munchausen 除去で
> Action Gap が 0.282 → 0.055（[btr_hyperparams_survey_2026-08-26.md](../../reports/btr_hyperparams_survey_2026-08-26.md) Table 2）。
> `docs/memo/done/008_Transformed Bellman Operator_10prd.md`（TBO）は自らを「Munchausen RL 導入への事前整備」と位置づけている。
> 裁定: 2026-09-04 グリル（Q0〜Q9 → D1〜D14）で全決定済み。
> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained（file:line は 2026-09-04 時点。実装時に再確認する）。
> 関連: [ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)（Munchausen ターゲットの契約）、
> [ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)（Actor Qヒント 3 列化）、
> [done/035](done/035_approx_actor_priority_per_10prd.md)（近似Actor初期優先度。hint 契約の改訂元）、
> [done/001](done/001_iqn_10prd.md) / [done/044](done/044_iqn_tau_stratified_sampling_10prd.md)（IQN と tau配置方式）、
> [done/059](done/059_config_concept_tree_alignment_10prd.md)（§3.2 TARGET 軸 = `DefaultDQNAgent.@munchausen` を予約済み、§10 遅延ゲート「M-IQN 等へ着手するとき」。**本 PRD がこのゲートを開く**）、
> [999_noisynet](999_noisynet_10prd.md)（BTR 採用・未実装で残る 2 部品のもう一方）、
> `CONTEXT.md`「Munchausen項」「soft価値ブートストラップ」「方策温度」「Actor Qヒント」。

## Context（背景・目的）

Munchausen RL（Vieillard, Pietquin, Geist, NeurIPS 2020, https://arxiv.org/abs/2007.14430）は、任意の TD 方式の即時報酬へ
「エージェント自身の scaled log-policy」を足すだけの拡張である。M-DQN の回帰ターゲット（論文 Eq. 3）:

```
q̂(r_t, s_{t+1}) = r_t + α·[τ·ln π_θ̄(a_t|s_t)]_{l0}^0 + γ·Σ_a' π_θ̄(a'|s_{t+1})·( q_θ̄(s_{t+1},a') − τ·ln π_θ̄(a'|s_{t+1}) )
π_θ̄ = softmax(q_θ̄ / τ)      θ̄ = target network      [x]_{l0}^0 = clip(x, l0, 0)
```

第 2 項が **Munchausen項**（報酬側のボーナス、常に ≤0）、第 3 項が **soft価値ブートストラップ**（argmax を全行動の
softmax 混合に置換）。理論上は前反復の方策への暗黙の KL 正則化になり、action gap を広げ、policy churn を抑える
（BTR: Munchausen 除去で churn 3.8% → 11.0%、Action Gap 0.282 → 0.055）。

実験記録側の診断はこれと整合している: 「γ 0.997 は action gap を犠牲にして地平を買っている。BTR が γ 0.997 と Munchausen を
同時に採っているのに対し、本記録は gap 拡大機構なしで γ だけを 0.997 にしていた」
（[2026-08-17_baseline.md](../experiments/default-dqn/atari/2026-08-17_baseline.md) 探索ブロック 19 考察）。BTR 部品のうち未実装で残るのは
Munchausen と NoisyNet の 2 つである（[2026-08-27_atari5.md](../experiments/default-dqn/atari/2026-08-27_atari5.md) の未実装行。SN は PRD 065 で実装済み）。

本 PRD の目的は **DQNBased 共通の Learner 機能として Munchausen を実装し、既定 OFF で配置し、Atari での最小動作確認
までを完了する**ことである。性能評価（BTR 差分の attribution A/B）は本 PRD の対象外で、実験記録側で別途行う。
`q_gap` / `action_churn_ratio` はこのリポジトリでは成績の予測子でないと確定している
（[atari/README.md](../experiments/default-dqn/atari/README.md) 現時点の判断）ため、**本 PRD は成績ゲートを置かない**。

## 数理契約

記号: B=batch、A=行動数、M=target 分位点数（QR は N）、`R_i`=target return（RB 保持の N-step 割引和、実空間）、
`n_i`=実 n-step 数、`d_i`=真の終端、`h`/`h⁻¹`=TBO 変換（`use_tbo=false` なら恒等）。以下すべて NoGrad・fp32。

**scaled log-policy（安定式）**: 任意の実空間 Q 行 `q[B,A]` に対し

```
slp(q) = q − max_a q − τ·ln Σ_a exp((q − max_a q)/τ)        （= τ·ln π、[B,A]、常に ≤0）
π = softmax(q/τ) = exp(slp(q)/τ)
```

**Munchausen項（bonus）**: `q_cur_real[B,A]` = `log_policy_source` の network が s_t で出す実空間Q値

```
bonus_i = α · clip( slp(q_cur_real)[i, a_i], l0, 0 )           （[B]、実空間、終端でも加える）
```

**soft価値**: `q_next_real[B,A]` = target network が s_{t+n} で出す実空間Q値（分位点表現では分位点平均）

```
TD:      V_soft_i     = Σ_a π_next[i,a] · ( q_next_real[i,a] − slp(q_next_real)[i,a] )          （= τ·logsumexp(q_next_real[i,·]/τ)）
QR/IQN:  soft_dist_ij = Σ_a π_next[i,a] · ( Z_next_real[i,a,j] − slp(q_next_real)[i,a] )        （[B,M]、分位点ごとに同じ π で混合）
```

**ターゲット**（実空間で組み立て、最後に h）:

```
TD:      y_i  = R_i + bonus_i + (1 − d_i)·γ^{n_i}·V_soft_i          → td_target_i  = h(y_i)
QR/IQN:  y_ij = R_i + bonus_i + (1 − d_i)·γ^{n_i}·soft_dist_ij      → target_dist_ij = h(y_ij)
```

適用範囲の契約（論文・Dopamine 公式実装・BTR の三者で一致。D3）:

- (a) bonus は先頭遷移 t にだけ加える。RB の target return（N-step 割引和）は触らない。bootstrap は `γ^n·(1−d)·V_soft(s_{t+n})`。
- (b) 終端マスクは soft価値側だけ。bonus は終端遷移でも加える。
- (c) clip `[l0, 0]` は bonus 側の `τ·ln π(a_t|s_t)` にだけ適用する。soft価値内の `τ·ln π` は clip しない。
- (d) 分位点表現では π を分位点平均から作り、soft価値は分位点ごとに同じ π で混合する（分位点ごとに別の π を作らない）。
- (e) TBO 有効時の「実空間Q値」は、分位点表現では **分位点ごとに h⁻¹ を掛けてから平均**する（`q` キーへ h⁻¹ を掛けない。
  h は非線形なので `h⁻¹(mean_j Z_j) ≠ mean_j h⁻¹(Z_j)`）。scalar 表現では `h⁻¹(q)`。TBO 無効時は `q` キーそのまま。

## 決定事項（2026-09-04 グリル）

| # | 論点 | 裁定 |
|---|---|---|
| D1 | 目的・範囲 | **DQNBased 共通**（`TDLearner` / `QRLearner` / `IQNLearner` の 3 Learner）。**既定 OFF**。スコープは **Atari での最小動作確認まで**。性能評価 A/B は実験記録側で別途。 |
| D2 | bonus の ln π の計算元 | **mode `learner.munchausen.log_policy_source = target \| online`、既定 `target`**（論文 / Dopamine 公式 M-IQN と一致）。`target` = obs も target network で forward する。追加 forward を 1 本増やす代わりに **obs∥next_obs を batch 連結して target forward を 1 回（2B）**にまとめる（IQN は taus を `GenerateTaus(2B, M, …)` 1 回で生成）。`online` = 既存 online forward の `q`（quantile 表現では `q_dist`）を detach 再利用（BTR の IQN 経路 `self.net.qvals(states)` と同じ。追加 forward ゼロ）。棄却: online 単一方式（論文の定義から外れる）、target 単一方式（コスト比較の余地を残す）。 |
| D3 | 数理契約 | 上記 (a)〜(e) をそのまま固定。 |
| D4 | TBO | **実空間で全て計算**。`h⁻¹` を `[B,A]` / `[B,A,M]` へ elementwise 適用し、π・ln π・soft価値・bonus を実空間で作ってから `h`。τ=0.03 は報酬スケール前提の値なので h 空間の softmax は意味が変わる。棄却: h 空間計算、`use_tbo` との併用 fail-fast（LunarLander / DropMerge は TBO ON で運用中）。 |
| D5 | argmax の迂回 | enabled 時は `SelectTargetActions` / `target_policy_->SelectAction` を呼ばない（soft価値に置換）。`learner.use_double_dqn=true` または `use_optimistic_target=true` との併用は**構築時に 1 回 WARN**（効果なし。AGENTS.md「意図しない構成の可能性」区分）。`@munchausen` プロファイル自身が `learner.use_double_dqn = false` を書く（BTR も double=0）。棄却: 黙って迂回（config の見た目と実態がずれる）、併用 fail-fast（チェーン合成で起動不能になりやすい）。 |
| D6 | 設定キー | `DefaultDQNAgent.learner.munchausen.{enabled=false, log_policy_source=target, alpha=0.9, entropy_tau=0.03, clip_value_min=-1.0}`。検証は enabled に関わらず常時 fail-fast。命名根拠: 既存実装 5 件の集計（`alpha`: Dopamine / BTR / BY571 / DI-engine=`m_alpha`、`entropy_tau`: BTR / BY571 / DI-engine、`clip_value_min`: Dopamine 公式 + Acme）。`tau` 単独は `uqe_tau_*` / `tau_rule` / `soft_update_tau` / `grad_clip_tau` と 5 重多義になるため使わない。プロファイル名と置き場所は [done/059](done/059_config_concept_tree_alignment_10prd.md) §3.2 の TARGET 軸で予約済み（`DefaultDQNAgent.@munchausen`、NN 配線を持たず ALGO と直交、M-IQN = `@iqn` + `@munchausen`）。本 PRD が §10 の遅延ゲートを開く。 |
| D7 | PER Learner 優先度 | 現行経路のまま**自動追従**（優先度 = `\|E[Z(s,a)] − E[target]\| + per_eps` で target が Munchausen 込みになる。論文 / BTR も同じ）。 |
| D8 | actor_approx 整合 | **Actor Qヒントを 2 列 → 3 列** `[q_sa, state_value, munchausen_term]` へ拡張し、近似Actor初期優先度を Munchausen ターゲットと同型にする（本 PRD 内で実装）。前例なし（Dopamine / BTR / Acme / DI-engine はいずれも max または uniform）。RB 共通層は hint 幅を `payload.size(1)` から動的に取るため無改修。棄却: 文書化のみ、WARN のみ、別 PRD へ defer。 |
| D9 | メトリクス | 新群 `36_agent_munchausen`、7 行（raw 5 + ema 2）。tag は 40 字前後、サブキーに `munchausen` を重ねない。GetScalar キー（フラット名前空間）は `munchausen_*`。IQN 診断と同じ固定 index pack で D2H に相乗り（同期追加なし）。OFF 時は既知キーとして NaN。配置は `metrics.scalar.@munchausen` プロファイル + `run.@munchausen`（agent / metrics の 2 チェーンを Run プロファイル 1 本で束ねる）。棄却: `@baseline` にコメントアウトで置く方式（2 箇所同期が残る）。 |
| D10 | 用語・ADR | CONTEXT.md へ新規 3 語（Munchausen項 / soft価値ブートストラップ / 方策温度）+ 「Actor Qヒント」改訂（3 列）。ADR 0035（ターゲット契約）+ ADR 0036（hint 3 列化、ADR 0010 / 0012 の「2 列」を改訂）。 |
| D11 | 受入・フェーズ | 受入 5 項（後述）。**フェーズは 1 本**（分割しない）。 |
| D12 | OFF 完全不変 | `enabled=false` では計算経路・RNG 消費とも現行と同一。ON 経路は別分岐として追加し、**OFF 経路の既存コードは触らない**（`+ 0` の加算すら入れない）。 |
| D13 | Rainbow | 対象外。`LearnerConfig` の既定 OFF に乗り、`RainbowAgentConfig` は `munchausen.*` を読まない（[ADR 0001](../adr/0001-default-dqn-tbo-scope.md) の TBO と同じ整理）。 |
| D14 | action_mask | Learner は現行も `action_mask` を見ないため、softmax も全行動で取る。既知制約として記録し、ActionMasking 基盤側で扱う。 |

スコープ外（Out of Scope）: 性能評価 A/B / softmax 行動（BTR `--stoch`）などの探索側変更 / M-VI・q-Munchausen 等の派生 /
RainbowAgent への公開 / action_mask 対応 / Actor 側 hint の target-network 化（ADR 0010「追加 forward なし」を維持）。

## 実装仕様（Codex 向け）

### 1. Config

`core/anet-core/include/anet/agent.hpp` の `LearnerConfig`（`:189-261`）へ、`IqnConfig`（`:232-235`）の隣に追加:

```cpp
struct MunchausenConfig {
    bool enabled = false;                      ///< Munchausen項と soft価値ブートストラップを有効化する
    std::string log_policy_source = "target";  ///< bonus の ln π を出す network: target / online
    float alpha = 0.9f;                        ///< α: Munchausen項のスケール（学習率 learner.alpha とは無関係）
    float entropy_tau = 0.03f;                 ///< τ_ent: 方策温度（IQN taus・soft_update_tau・grad_clip_tau とは別概念）
    float clip_value_min = -1.0f;              ///< l0: τ·ln π(a_t|s_t) の下限 clip（bonus 側のみ）
} munchausen;
```

`core/anet-core/include/anet/default_dqn_agent.hpp` の `DefaultDQNAgentConfig::ReadConfig` で `ANET_READ_CONFIG` 5 本
（`learner.munchausen.enabled` / `.log_policy_source` / `.alpha` / `.entropy_tau` / `.clip_value_min`、既存の
`learner.iqn.*` 読取 `:187-227` の並び）。検証は `quantile_mode` ブロック（`:286-350`）と同型で、**enabled に関わらず常時**:

- `log_policy_source ∉ {target, online}` → `ANET_SYSTEM_ERROR`（キー・指定値・許容値を含める）
- `alpha` が非 finite または `[0, 1]` 外 → エラー
- `entropy_tau` が非 finite または `≤ 0` → エラー
- `clip_value_min` が非 finite または `> 0` → エラー

WARN（enabled 時のみ、構築時に 1 回、英語）:

- `learner.use_double_dqn=true` → `... has no effect: learner.munchausen.enabled=true replaces the argmax bootstrap with the soft value`
- `use_optimistic_target=true` → 同様（`target_policy` は構築するが target 計算では使われない）

`RainbowAgentConfig` は読まない（D13）。

### 2. Learner 共通ヘルパ（`dqn_based_agent.*` 同居、名前付き namespace）

```cpp
struct MunchausenTargetTerms {
    torch::Tensor bonus;                    // [B]   fp32 実空間。α·clip(slp(q_cur)[a_t], l0, 0)
    torch::Tensor next_policy;              // [B,A] fp32 π_next
    torch::Tensor next_scaled_log_policy;   // [B,A] fp32 slp(q_next)
    torch::Tensor diagnostics;              // [5]   fp32 {log_pi_mean, clip_ratio, bonus_mean, next_entropy, soft_gap}
};
MunchausenTargetTerms ComputeMunchausenTargetTerms(
    const torch::Tensor& q_current_real,    // [B,A] 実空間（source network、s_t）
    const torch::Tensor& q_next_real,       // [B,A] 実空間（target network、s_{t+n}、分位点平均）
    const torch::Tensor& actions,           // [B] int64
    const LearnerConfig::MunchausenConfig& cfg);
torch::Tensor ScaledLogSoftmax(const torch::Tensor& q_real, float entropy_tau); // slp(q)、fp32、[B,A]
```

- 入力は関数先頭で `torch::kFloat32` へ cast する（AMP 領域 `anet::Autocast` 内から呼ばれる。IQN 診断 `:3370-3383` の前例）。
- `ScaledLogSoftmax` は安定式（`q − max` を先に引く）。`π = exp(slp/τ)`。
- diagnostics: `log_pi_mean = mean_i slp(q_cur)[i,a_i]`（clip 前）、`clip_ratio = mean_i 1[slp(q_cur)[i,a_i] < l0]`、
  `bonus_mean = mean_i bonus_i`、`next_entropy = mean_i (−Σ_a π_next·slp(q_next)/τ)`（nats）、
  `soft_gap = mean_i ( Σ_a π_next·(q_next_real − slp(q_next)) − max_a q_next_real )`（≥0）。
- Actor 側の hint 計算（§8）も同じ `ScaledLogSoftmax` を使う（数式の二重実装を避ける）。

### 3. TDLearner（`core/anet-core/src/dqn_based_agent.cpp:3026-3063`）

現行の `{ select action → ForwardTarget(next_obs) → gather → bootstrap }` ブロックの**外側**に `if (config_.munchausen.enabled)`
分岐を追加し、OFF 経路は既存コードのまま（D12）。ON 経路:

1. `q_next_all = ForwardTarget(...).At("q")`（[B,A]）。`log_policy_source=target` なら §5 の 2B forward から split。
2. `q_next_real = use_tbo ? TransformHInv(q_next_all) : q_next_all`、`q_cur_real` は §5。
3. `terms = ComputeMunchausenTargetTerms(q_cur_real, q_next_real, actions, cfg)`。
4. `V_soft = (terms.next_policy * (q_next_real − terms.next_scaled_log_policy)).sum(1)`（[B]）。
5. `raw = target_returns + terms.bonus + not_terminal * gamma_n * V_soft`、`td_target = use_tbo ? TransformH(raw) : raw`。
6. 以降（`td_error = q_sa − td_target`、TD clip、Huber、PER）は既存のまま。

### 4. QuantileLearnerBase / QRLearner / IQNLearner（`:2789-2815`、QR `:3179-3198`、IQN `:3315-3340`）

ON 経路では `SelectTargetActions` を呼ばず、`next_dist_all`（[B,A,M]、`q_dist`）から:

1. `Z_next_real = use_tbo ? TransformHInv(next_dist_all) : next_dist_all`（[B,A,M]）、`q_next_real = Z_next_real.mean(2)`（[B,A]、(e)）。
2. `terms = ComputeMunchausenTargetTerms(q_cur_real, q_next_real, actions, cfg)`。
3. `soft_dist = (terms.next_policy.unsqueeze(2) * (Z_next_real − terms.next_scaled_log_policy.unsqueeze(2))).sum(1)`（[B,M]）。
4. `raw_dist = returns + terms.bonus.view({B,1}) + gamma_n * not_terminal * soft_dist`、`target_dist = use_tbo ? TransformH(raw_dist) : raw_dist`。

`CalcTargetQuantiles` は OFF 経路専用のまま残す（内部で `h⁻¹` を掛けるため ON 経路と混ぜない）。ON 経路用に
実空間入力を取る別関数（例: `CalcMunchausenTargetQuantiles(samples, soft_dist, bonus)`）を追加する。
IQN の `target_taus` 生成・注入は現行どおり（§5 の 2B 化を除く）。loss（`ComputeQuantileHuberLoss` / `ComputeIqnQuantileHuberLoss`）、
PER 優先度（`td_error = q_sa − target_dist.mean(1)`）、IQN 診断は既存のまま。

### 5. `log_policy_source` の実装

- `target`: `obs` と `next_obs`（`NormalizedSampleObservations` の正規化後）を **key ごとに `torch::cat(dim=0)` した 2B の
  TensorDict** を 1 回 `ForwardTarget` し、出力を `narrow` で前半（s_t）/ 後半（s_{t+n}）へ分ける。IQN は
  `GenerateTaus(2 * B, M, config_.iqn.target_taus.sample_mode, 0, 1, device_, *GetRandomGenerator())` を 1 回呼び、
  `kKey_Taus` として注入する。target network は常時 eval mode（`ForwardTarget` はモードを変えない）なので batch 連結による
  統計混入はない。TensorDict の batch cat ユーティリティが無ければ Learner 内の小ヘルパで足りる。
  `q_cur_real` = scalar: `h⁻¹(q_front)`、quantile: `h⁻¹(q_dist_front).mean(2)`。
- `online`: 既存の online forward 出力を detach して使う。scalar: `q_out.At("q")`（`:2994-3002`）、quantile:
  `current_out.At("q_dist")`（[B,A,N]、QR `:3172` / IQN `:3311` の近傍）→ `h⁻¹` → `.mean(2)`。追加 forward・追加 RNG なし。

### 6. ProfileRange

QR / IQN の既存フェーズ列（`normalize → … → select_target_action → forward_target → loss → …`）に、ON 経路では
`select_target_action` の代わりに `forward_target`（2B）→ `ANET_PROFILE_SCOPE_NEXT(munchausen_target)` → `loss` と並べる。
TD も同じ名前 `munchausen_target` を使う。

### 7. 診断 pack と `BatchUpdateResult`

`terms.diagnostics`（[5] fp32）を IQN 診断と同じ経路で運ぶ: `PreparePerPriorityUpdate` の `readback_parts`
（`:2386-2405`）へ 1 要素追加し、`ApplyPerPriorityUpdate` の `narrow`（`:2464-2467`）と `MakeBatchUpdateResult`（`:2533`）で
`BatchUpdateResult::munchausen_diagnostics` へ載せる。`GetScalar`（`dqn_based_agent.hpp:176-379`）に固定 index で 5 キー:

| GetScalar key | index | 意味 |
|---|---|---|
| `munchausen_log_pi_mean` | 0 | `τ·ln π(a_t\|s_t)` の clip 前平均 |
| `munchausen_clip_ratio` | 1 | l0 で clip された行の割合 |
| `munchausen_bonus_mean` | 2 | bonus（α·clip 後、target に実際に乗る量）の平均 |
| `munchausen_next_entropy` | 3 | `H(π(·\|s_{t+n}))` の平均（nats）。→0 で実質 max bootstrap に退化 |
| `munchausen_soft_gap` | 4 | `V_soft(s') − max_a Q'(s',a)` の平均（実空間、≥0） |

OFF 時（pack 未定義）は既知キーとして `NaN` を返す（AGENTS.md GetScalar 実装ルール）。

### 8. Actor Qヒント 3 列化（[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)）

- 定数（`dqn_based_agent.hpp:77-79`）: `kActorQHintColumnCount = 3`、`kActorQSaColumn = 0`、`kActorStateValueColumn = 1`、
  `kActorMunchausenTermColumn = 2`。`ActorQHintBatch` / `ActorQHintRow` に `actor_munchausen_term` を追加。
  `PackActorQHint(q_sa, state_value, munchausen_term)` / `DecodeActorQHint` を 3 列へ（`:238-283`。エラーメッセージの `[B,2]` / `K=2` も更新）。
- **Actor への設定受け渡し**: `struct ActorQHintConfig { LearnerConfig::MunchausenConfig munchausen; bool use_tbo; float tbo_epsilon; }`
  を `dqn_based_agent.hpp` に置き、`Actor` ctor（`:745-753`）の `emit_actor_q_hint` の隣で受け取る。生成点は
  `DefaultDQNAgent::CreateActor`（`default_dqn_agent.cpp:531-538`）で `config_.learner` から組む。
- **`Actor::MakeAction`**（`dqn_based_agent.cpp:1786-1800`）: `emit_actor_q_hint_` 時、aux `q_values`（[B,A]、Q空間）を fp32 へ cast し
  - OFF: `state_value = max_a q_values`、`term_all = zeros[B,A]`（現行と同値）。
  - ON: `q_real = use_tbo ? h⁻¹(q_values) : q_values`、`slp = ScaledLogSoftmax(q_real, τ)`、
    `term_all = α·clamp(slp, l0, 0)`（[B,A]）、`V_soft = (softmax(q_real/τ) * (q_real − slp)).sum(1)`、
    `state_value = use_tbo ? h(V_soft) : V_soft`（**Q空間で格納**。推定器の `h⁻¹` 経路を無改修で通す）。
  - 共通: `aux["munchausen_term_all"] = term_all`（device tensor、RB へは永続化しない。`q_values` と同じ扱い）、
    `PackActorQHint(q_sa, state_value, term_all.gather(1, a))`。
- **`DQNActionInfo::WithAction`**（`:560-580`）: hint がある場合は aux `q_values` に加えて aux `munchausen_term_all` を必須とし
  （欠落は契約違反で `ANET_SYSTEM_ERROR`）、差し替え後の行動で `q_sa` と `munchausen_term` を再 gather、`state_value` は維持。
- **`DqnInitialPriorityEstimator`**（`:287-326`）: `ValidateHint` は 3 列すべて finite、`Estimate` は
  `target = input.target_return + start.actor_munchausen_term; if (!terminal) target += discount · h⁻¹(boot.actor_state_value); if (use_tbo) target = h(target)`。
- 近似の前提（ADR 0036 に記録）: Actor の network は Train Actor snapshot（online 系）なので `log_policy_source=target` でも
  online 近似になる。IQN+UQE では `q_values` が risk-biased action score（[ADR 0019](../adr/0019-iqn-uqe-score-without-extra-forward.md)）
  なので π もその近似。いずれも初期優先度の順位付け用途として許容（[ADR 0010](../adr/0010-actor-priority-mean-q-approx.md) の系統差の延長）。

### 9. 設定ファイル

`apps/runner/config/agent.txt`:

```
# @baseline（デフォルト直書き。use_tbo の隣）
DefaultDQNAgent.@baseline.learner.munchausen.enabled = false           # Munchausen RL(PRD 067)。ON は @munchausen プロファイルで
DefaultDQNAgent.@baseline.learner.munchausen.log_policy_source = target # target / online
DefaultDQNAgent.@baseline.learner.munchausen.alpha = 0.9               # α
DefaultDQNAgent.@baseline.learner.munchausen.entropy_tau = 0.03        # τ_ent(方策温度。IQN の taus とは別)
DefaultDQNAgent.@baseline.learner.munchausen.clip_value_min = -1.0     # l0

# --- Munchausen RL プロファイル(PRD 067。論文/BTR 値で ON にする。全 env 共通) ---
# soft 価値ブートストラップは argmax を使わないため Double DQN を切る(BTR も double=0。true のままだと構築時 WARN)。
DefaultDQNAgent.@munchausen.learner.munchausen.enabled = true
DefaultDQNAgent.@munchausen.learner.use_double_dqn = false
```

`apps/runner/config/metrics_scalar.txt`（`@iqn_search_p0` の隣に新プロファイル）:

```
# ==============================================================================
# Munchausen RL 診断(PRD 067)。agent 側 @munchausen と対で run.@munchausen が束ねる。OFF 構成で購読すると NaN(既知キー)。
# ==============================================================================
metrics.scalar.@munchausen.[36_agent_munchausen/01_log_pi_mean] = munchausen_log_pi_mean @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/02_log_pi_mean_ema] = munchausen_log_pi_mean @learn $update_result $ema
metrics.scalar.@munchausen.[36_agent_munchausen/03_clip_ratio] = munchausen_clip_ratio @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/04_bonus_mean] = munchausen_bonus_mean @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/05_bonus_mean_ema] = munchausen_bonus_mean @learn $update_result $ema
metrics.scalar.@munchausen.[36_agent_munchausen/06_next_entropy] = munchausen_next_entropy @learn $update_result
metrics.scalar.@munchausen.[36_agent_munchausen/07_soft_gap] = munchausen_soft_gap @learn $update_result
```

`apps/runner/config/Atari.txt`（Run プロファイル節。現行の live 腕 `run.@v5_iqn_impala_x2` を写し、`@munchausen` を ALGO 直後に挿す）:

```
# Munchausen RL(PRD 067)。agent の @munchausen と metrics の @munchausen を 1 行ずつで束ねる(2 箇所同期をここに閉じる)。
run.@munchausen : AtariEnv.$ = AtariEnv.@v5_noop30 > E1
run.@munchausen : DefaultDQNAgent.$ = @baseline > @iqn > @munchausen > A1 > @bf16 > A2 > A3
run.@munchausen : DefaultDQNAgent.net.branch.[main_feature].$ = net.branch.@AtariImpalaX2
run.@munchausen : metrics.scalar.$ = metrics.scalar.@baseline > metrics.scalar.@munchausen > M1 > M2
run.@munchausen : app.run_name = run_{t}_munchausen_${E1.game}
```

A1 / A2 が `learner.use_double_dqn = true` を後段で供給する構成では WARN が出る（意図した診断）。smoke では `app.run_name` を
CLI で `run_{t}_tmp_smoke_067_${E1.game}` へ上書きする（AGENTS.md Run 命名規約）。

`@munchausen` は 059 §3.2 の TARGET 軸そのもので、NN 配線を持たないため `@qr` / `@iqn` と違い `net.$` 行を持たない
（plain 形 `DefaultDQNAgent.@munchausen.learner.*`）。α / τ / l0 の掃引値は 059 §4.1 の昇格ライフサイクルどおり `A2` / `A3` から始め、
定着したら `A1`、恒久化したら素材の既定値へ上げる。素材宣言は `config_data.txt` に出ない（059 §4.1）ので、
「どの腕が走ったか」の確認は解決済み leaf（`learner.munchausen.*`）と `json/config_resolution.json`
（`inspect_run.py resolution`）で行う。

### 10. 単体テスト（`core/anet-core/src/dqn_based_agent_test.cpp`）

雛形は `"DQN initial priority estimator matches scalar learner TD priority"`（`:1927-2005`。学習を止めた Learner に
`ExperienceSamples` を手組みし、`ForwardTarget` を自前で呼んで oracle を作る）。タグは `[dqn][learner][munchausen]` を基本に
`[math]` / `[tbo]` / `[iqn]` / `[qr]` / `[config]` / `[actor_initial]` を併用する。

1. **ターゲット既知値**: TD / QR / IQN × `log_policy_source` 2 値。oracle は scalar が `R + bonus + (1−d)γ^n·τ·logsumexp(q_next_real/τ)`、
   分位点が明示混合（テストコード側で `Σ_a π·(Z_j − τlnπ)` を素朴に計算）。`Catch::Approx` で `td_error` / `target_dist` 一致。
2. **α = 0** で bonus が 0（soft価値だけの soft-DQN に一致）。
3. **τ → 小**（例 1e-3 と行動間 gap ≥ 0.1）で soft価値が `max_a q_next_real` に収束し、Munchausen OFF のターゲットへ許容差内で一致。
4. **clip**: `slp(a_t) < l0` の行だけ bonus が `α·l0` に張り付き、`munchausen_clip_ratio` が一致。
5. **terminal**: `d=1` の行で bootstrap が消え bonus は残る。
6. **TBO ON / OFF** の両方で 1 が成立（TBO ON では (e) の「分位点ごとに h⁻¹」の oracle）。
7. **AMP bf16**（cuda がある環境のみ）: 診断 pack と target が fp32 で計算される（dtype 検証）。
8. **config fail-fast 4 種** と **WARN 2 種**（`ScopedLogCapture` 等の既存手段で 1 回だけ出ることを確認）。
9. **Rainbow 不変**: `RainbowAgentConfig` 経由で `munchausen.enabled` が false のまま。
10. **hint 3 列**: `PackActorQHint` / `DecodeActorQHint` の round-trip、旧 2 列 payload の schema 違反エラー、
    `DqnInitialPriorityEstimator` が `target_return + term + discount·h⁻¹(state_value)` を再現（TBO 両方）、
    `WithAction` で行動差し替え後の `q_sa` と `term` が再 gather され `state_value` が維持される、`munchausen_term_all` 欠落で契約違反。
11. **Actor hint 値**: Munchausen ON の Actor が出す `state_value` が `h(τ·logsumexp(h⁻¹(q)/τ))`、`term` が `α·clip(slp[a], l0, 0)` に一致。
12. **同 seed 決定性**: ON（両 source）で同 seed 2 回の `loss` / `td_error` 系列一致（determinism 既定 ON）。
13. **OFF 経路不変**: 既存の Learner / hint テストが無修正（hint 幅の 3 列化に伴う期待値更新を除く）で緑。

### 11. docs/design 同期（同一変更内。AGENTS.md「AI エージェントの作業ルール」）

- `docs/design/200_dqn_agents.jp.md`: §2.2（target network 段落に soft価値ブートストラップと `log_policy_source`）、§2.4 / §6.2（Actor Qヒント 3 列）、
  §7.2（設定群表の Learner 行に Munchausen）、§9.1（群 36 の購読キー）。
- `docs/design/030_user_guide_analysis.jp.md`: 群 36 の読み方（`06_next_entropy` → 0 は max bootstrap への退化、`03_clip_ratio` の意味）。
- `docs/memo/done/035_approx_actor_priority_per_10prd.md`: hint 契約の 3 列化を追記（当時の記録は残し、改訂行を足す）。

## 現行コードで確定している事実（実装の下地、2026-09-04）

1. **ターゲットの挿入点**: TD `dqn_based_agent.cpp:3053-3063`（`raw_td_target = target_returns + not_terminal * gamma_n * bootstrap` → `h`）、
   QR / IQN は `CalcTargetQuantiles`（`:2797-2815`。`next_dist`[B,N] は action gather 済みで、`h⁻¹` を内部で掛ける）。
   呼び出し側（QR `:3179-3198`、IQN `:3315-3340`）には `next_dist_all`[B,A,M] が残っている。`gamma_n` は `γ^{n_i}` の per-sample tensor。
2. **target forward の出力**: `ForwardTarget`（`:432-435`）は `"q"`[B,A]（QR / IQN では分位点平均）と `"q_dist"`[B,A,N]（action 次元 1、分位点次元 2。
   `dqn_based_heads.cpp:217-218`, `:423-427`）を常に返す。training mode は変えず、target network は常時 eval。
3. **`target_policy_` は Double DQN の argmax 選択専用**（`:3035-3039`、`:2789-2795`。network は `use_double_dqn` で online / target を選ぶ）。
   `use_optimistic_target` は `target_policy` の既定を `train_policy`（UQE 等）から複製する（`default_dqn_agent.hpp:142-155`）。
4. **softmax / log_softmax / logsumexp は DQN 経路に存在しない**（ImageCls / MuZero のみ）。`tau` は `uqe_tau_*` / `tau_rule` / `soft_update_tau` /
   `grad_clip_tau` で 4 重多義。
5. **online forward の `q`[B,A] は 3 Learner とも s_t で計算済み**（TD `:2994-3002`、QR `:3172`、IQN `:3311`。勾配付き → detach して使う）。
   obs は現在 `ForwardTarget` を通っていない（`target` source の追加 forward はここが由来）。
6. **forward + loss は `anet::Autocast`（`:2989`, `:3154`, `:3290`）の内側**。IQN 診断は fp32 へ cast している（`:3370-3383`）。
7. **PER 優先度は `|E[Z(s,a)] − E[target]|`**（QR `:3206-3209`、IQN `:3348-3351`、TD は unclipped `td_error`）。target を変えれば自動で追従する。
8. **診断 pack の前例**: `iqn_diagnostics` を `torch::stack` で作り（`:3403-3411`）、`PreparePerPriorityUpdate` が readback に連結（`:2386-2405`）、
   `ApplyPerPriorityUpdate` が narrow（`:2464-2467`）、`MakeBatchUpdateResult` が結果へ載せる（`:2533`）。`GetScalar` は固定 index（`dqn_based_agent.hpp:263-275`）。
9. **Rainbow は `TDLearner` / `QRLearner` を共用**（`rainbow_agent.cpp:110-117`）し、`RainbowAgentConfig` は learner キーの部分集合しか読まず
   `use_tbo` 等を強制 OFF する（`rainbow_agent.hpp:70-73`）。`agent.class_id = RainbowAgent` は現用 config で未選択。
10. **Actor Qヒント**: `kActorQHintColumnCount = 2`（`dqn_based_agent.hpp:77-79`）、pack / decode（`.cpp:238-283`）、Actor 側の生成（`:1786-1800`、aux `q_values` から
    `q_sa` と `max`）、`WithAction` の再 gather（`:560-580`）、推定器（`:287-326`。`target_return + discount·h⁻¹(state_value)` → `h`）。
    生成ゲートは `!IsEval && use_per && mode == actor_approx`（`default_dqn_agent.cpp:531-533`）。
11. **RB 共通層は hint 幅を持たない**: 行は `replay_hint_payload.size(1)` で走査され（`replay_buffer_impl.cpp:1335-1337`）、
    `c10::SmallVector<float, 4>`（`replay_buffer_impl.hpp:43`）へコピーされる。K=3 は inline capacity 内。
12. **設定プロファイルの前例**: ALGO `@qr` / `@iqn`（`agent.txt:72-79`）、`@bf16` / `@random`（`:81-102`）、Atari のチェーン
    `@baseline > @iqn > A1 > @bf16 > A2 > A3`（`Atari.txt:18-22`）、Run プロファイル `run.@v5_iqn_impala_x2`（`:251-255`）、metrics チェーン
    `metrics.scalar.$ = metrics.scalar.@baseline > M1 > M2`（`:1356`）。群 36 は未使用。
13. **BTR の実装事実**（`C:\dev\BTR\Agent.py`）: 非 IQN 経路は `self.tgt_net(states)`（target）、IQN 経路は `self.net.qvals(states)`（online）で
    bonus を計算する。既定 `munch=1, munch_alpha=0.9, entropy_tau=0.03, lo=-1, double=0`。Dopamine 公式 M-IQN は両方 target。

## 受入基準

1. **OFF 完全不変**（手順で証明する）: 本改修直前の base commit で smoke 構成（config・`train.seed`・step 数を実装計画に 1 組固定）を実行し
   主要 tag（loss / q_max / td_mean / `39_agent_per/05`）の metrics checksum を記録 → 改修後ビルドで同一コマンド → checksum 一致。
   加えて `enabled=false` では Munchausen の数値経路（2B forward・softmax・pack）に**不到達**であること。
   checkpoint の raw SHA は既存の serialize 非決定性のため合否に使わない（[930](930_serialize_10prd.md)）。
2. **単体テスト**: §10 の全項目が緑。既存 `[dqn]` 群も緑（hint 3 列化に伴う期待値更新のみ許容）。
3. **smoke**: Atari で `run.$ = run.@munchausen` + `app.run_name` を tmp 名へ CLI 上書きした短時間 Run（例 200k exp step）を実行し、
   `inspect_run.py tags` で `36_agent_munchausen/01`〜`07` が status=ok・count>0、`38_agent_loss/01_loss` が finite。
   同構成に `A3.learner.per_initial_priority_mode = actor_approx` を足した 2 本目で `39_agent_per/05_sample_actor_init_ratio` が
   非ゼロになり、`52_actor_learner_pair_count` が点灯する。`log_policy_source = online` でも 1 本通す。
   各 Run で `inspect_run.py config --effective-only` の `learner.munchausen.*` と `inspect_run.py resolution` の `@munchausen` 選択を確認する
   （素材宣言は `config_data.txt` に残らないため）。
4. **throughput**（実測・許容差つき）: 受入 1 と同じ smoke 構成、**ラウンドロビン配置**（実験機は 1 時間で最大 8% ドリフト）で
   (a) OFF vs base commit 各 2 本: steps/s 平均差 **< 2%**（ゲート）。(b) ON `target` / ON `online` を同配置で各 2 本: 実測を記録
   （ゲートなし。`munchausen_target` と `forward_target` の ProfileRange 値も添える）。
5. **成績ゲートなし**（D1）。Atari の性能評価は実験記録側で別途行う。

## 正直なリスク

- `log_policy_source = target` は Learner の GPU compute を増やす（target forward の入力が 2B）。RR1 の Atari は Learner 律速なので
  wall-clock に出る可能性がある。受入 4(b) で実測し、性能評価の腕として `online` を並べる。
- Actor 側 hint は `target` 設定でも online snapshot からの近似（ADR 0010 の「追加 forward なし」を維持したため）。actor_approx を
  Munchausen と併用する場合の初期優先度の質は `39_agent_per` の Actor/Learner 順位相関で監視する。
- IQN+UQE では Actor の `q_values` が risk-biased score なので π もその近似になる（Learner 側は target network の分位点平均で正確）。
- `action_mask` を持つ env（DropMerge）では softmax が非合法行動を含む。Learner の argmax も現行同様に mask を見ないため
  相対的な劣化ではないが、mask 対応時には soft価値も同時に直す必要がある。
- `entropy_tau = 0.03` は clip 報酬（±1）前提の値。報酬スケーラや TBO を使う env では実空間の報酬スケールに合わせて再調整が要る
  （config の責任、フレームワークは検証しない）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/agent.hpp` | `LearnerConfig::MunchausenConfig` |
| `core/anet-core/include/anet/default_dqn_agent.hpp` | 5 キー読取・検証・WARN 2 種 |
| `core/anet-core/src/default_dqn_agent.cpp` | `CreateActor` で `ActorQHintConfig` を組んで渡す |
| `core/anet-core/src/dqn_based_agent.hpp` / `.cpp` | 共通ヘルパ、3 Learner の ON 分岐、2B target forward、診断 pack、GetScalar 5 キー、hint 3 列（定数 / pack / decode / Actor / WithAction / 推定器）、ProfileRange |
| `core/anet-core/src/dqn_based_agent_test.cpp` | §10 |
| `apps/runner/config/agent.txt` / `metrics_scalar.txt` / `Atari.txt` | §9 |
| `docs/design/200_dqn_agents.jp.md` / `030_user_guide_analysis.jp.md`、`docs/memo/done/035_*_10prd.md` | §11 |
| `CONTEXT.md`、`docs/adr/0035`、`0036`、`0010` / `0012` の後発注記 | 本 PRD 作成時に実施済み |
