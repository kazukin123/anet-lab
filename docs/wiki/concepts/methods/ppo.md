---
title: PPO の実装の選択と既定値
page_kind: 手法
aliases:
  - PPO
  - Proximal Policy Optimization
  - CleanRL
  - SB3
  - Stable-Baselines3
  - GAE
  - approx_kl
  - target_kl
  - value clip
description: CleanRL と SB3 の PPO は GAE・比・approx KL・方策の損失の式が同じで、既定値、打ち切りの扱い、value の損失の clip と 0.5 倍、KL で止める位置と倍率が違う。
updated: 2026-11-28
---
# PPO の実装の選択と既定値

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

- CleanRL の `ppo.py` と SB3 の PPO は、GAE・新旧の方策の比・approx KL・方策の損失・advantage の標準化の式が同じで、optimizer はどちらも Adam の eps 1e-5 である。
- 違うのは既定値（収集の長さ、ミニバッチ、epoch、学習率、entropy の係数、value の clip）と、打ち切りの扱い（CleanRL は終端と同じ、SB3 は γ·V を報酬に足す）、value の損失の 0.5 倍、KL で止める位置と倍率である。
- 方策と価値の Network は、単純な環境では別のほうがよいと報告され、画像では計算を節約するため共有が使われる。

## 分かっていること

### anet の知見

なし

### 外の事例

- CleanRL の `ppo.py` の既定は、env 4 本、収集 128 step、ミニバッチ 4 分割、4 epoch、学習率 2.5e-4（線形に 0 へ）、γ 0.99、λ 0.95、clip 0.2、value の clip あり、entropy の係数 0.01、value の係数 0.5、勾配の clip 0.5、target KL 無し、総 500,000 step である。SB3 の既定は、収集 2048 step、ミニバッチ 64 件、10 epoch、学習率 3e-4、γ 0.99、λ 0.95、clip 0.2、value の clip 無し、entropy の係数 0.0、value の係数 0.5、勾配の clip 0.5、target KL 無しである（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- CleanRL と SB3 は、GAE（`delta = r + γ·nextvalues·nextnonterminal − values[t]`、`lastgaelam = delta + γ·λ·nextnonterminal·lastgaelam`、`returns = advantages + values`）、比（`exp(newlogprob − oldlogprob)`）、approx KL（`mean((ratio − 1) − logratio)`）、方策の損失（clip した比との max。SB3 は `min` の符号反転）、合計の損失（`pg_loss − ent_coef·entropy + vf_coef·v_loss`）の式が同じで、optimizer はどちらも Adam の eps 1e-5 である（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- 打ち切りは、CleanRL が終端と同じに扱って bootstrap せず（`next_done = terminations or truncations`）、SB3 は打ち切りの step の報酬に `γ·V(終端の観測)` を足す（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- value の損失は、CleanRL が 0.5 倍の二乗誤差で、clip ありでは収集時の value からの変化を ε で clip した側との max を取る。SB3 は 0.5 倍の無い `mse_loss` である（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- advantage の標準化は、どちらもミニバッチごとの `(A − mean)/(std + 1e-8)` で、SB3 は件数が 1 のときは行わない（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- KL で止めるのは、CleanRL が epoch の終わりに最後のミニバッチの `approx_kl > target_kl` なら残りの epoch をやめる形、SB3 はミニバッチごとに `approx_kl > 1.5·target_kl` なら optimizer の step の前でやめる形で、どちらも target KL の既定を持たない。OpenAI Spinning Up の PPO（PyTorch 版）は `target_kl=0.01` で、`kl > 1.5 * target_kl` のとき方策の更新をやめる（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7・§15）。
- CleanRL の `ppo.py` は、方策と価値を別の MLP（64-64、Tanh）にする（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- CleanRL の `ppo.py` は直交初期化を使い、方策の出力を価値の出力より小さい gain で始める → 詳しくは [重みの初期値](weight_init.md)（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- CleanRL の explained variance は `1 − Var(R − v)/Var(R)` で、`Var(R)` が 0 なら NaN にする（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§2.7）。
- 方策と価値の Network は、単純な環境では別のほうがよいと報告されており、画像では計算を節約するため共有が使われる（出典: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)、§4 P17。P17 の元になった調査はリポジトリに無い）。

## 横断的知見

なし

## 見立て

なし

## 分かっていないこと

- 小さな Network と少ない lane の PPO では、CPU と CUDA のどちらが速いか。設計文書は、毎 step に行動選択の転送が 2 回入るので CPU のほうが速い見込みとし、測っていない（出典: [PPO系Agent](../../../design/230_ppo_agents.jp.md)、§11）。

## 撤回・更新

なし

## 関連

- 設計: [PPO系Agent](../../../design/230_ppo_agents.jp.md)、§3（Network）・§6.2（次の観測の value）・§7（更新）・§9（メトリクス）・§11（device と同期の転送）。
- 設計: [方策ベースの学習の共通部品](../../../design/115_on_policy_learning.jp.md)。
- 用語: [CONTEXT.md](../../../../CONTEXT.md) の「rollout」「PPO系エージェント」。
- RFD: [RFD0209](../../../memo/0209_ppo_family_and_on_policy_core_parts_10rfd.md)。
- トピック別まとめ: [326 Agent個別実装 PPO系列](../framework/326_ppo.md)。
- [重みの初期値](weight_init.md)。
