# 勾配ノイズとbatch

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機構](../index.md#機構)

## 今の結論

GNSと有効batchの対応はoptimizer・曲率・抽出方法・予算に依存する。DropMergeでは臨界batchが学習とともに増えたが、batch増加の最終target評価の差は学習側の差より小さかった。

## 分かっていること

- GNSと有効batchの対応はoptimizer・曲率・抽出・予算に依存するため、大きなGNSだけでB増加を決めない。（出典: [gradient_noise_observation_and_control_survey_2026-10-15.md](../../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md)、1. 調査の結論・8. 総合評価と調査の限界）。
- DropMergeで臨界batchは学習とともに増えたが、B増加の最終target評価の差は学習側の差より小さかった。（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ1・3〜5）。
- DropMerge の B256/RR1 では、臨界バッチ（Adam 前処理版）が 0–10M の 232 から 170–230M の約 3,900 へ増え、B で割った比は 0.9 から 15.2 になった。0〜50M は分散が 4 倍に増え、50M 以降は分散が横ばいのまま信号が 1/3 に細った。分散の約 95% は feature 側にある（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ1・探索ブロック01。Run: run_20261015-204705_dmfb_40_s2 / run_20261016-005647_dmfb_41a_40_b256 / run_20261017-185400_dmfb_42a_34a_ctl）。
- 同じ 170M の checkpoint から分岐した B256/RR1・B512/RR2・B1024/RR2 の最初の測定（学習ステップ 0）は臨界バッチ 856 / 823 / 972 でほぼ同じだったが、学習が進むと 1 サンプルごとの勾配の分散が B とともに増え（170–230M で 133M / 272M / 351M）、÷B の正味の低下は B1024/RR2 だけ（−22%）で B512/RR2 は戻りきった。前処理なしの版で見ると B1024/RR1 も、直後の ÷B 0.48 から 40–60M には対照と同じ水準（1.08）へ戻った（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック04 探索結果・考察、探索ブロック13 考察。Run: run_20261017-185400_dmfb_42a_34a_ctl / run_20261017-235549_dmfb_42d_34a_b512rr2 / run_20261018-074714_dmfb_42h_34a_b1024rr2 / run_20261018-225933_dmfb_42j_34a_b1024rr1）。
- B と RR の組み合わせを 3 つの性質に分けると、1 回の更新のノイズ（B）は成績・NoisyNet の感度・Q の水準（B1024 で −4%）・feature の大きさの維持に、経験あたりの更新回数は feature 側の重みの増え方（2 倍で +7.7%、1/4 で +0.1%）に、1 つの経験が使われる回数は取りこぼしと DS の無い局に沿って動いた（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ12・探索ブロック13。Run: 42 系列 6 本）。
- 停滞区間（170→230M）で B を上げずに学習率を 7.1e-5 に下げた腕は、最終の評価専用 Run で B1024/RR1 と同じ +1.0%（約 2.5σ）だった。学習側の上乗せ（+1.4〜1.7%）と学習中の評価の上乗せ（+1.2〜1.8%）がほぼ同じで、NoisyNet で行動が変わる割合はほとんど下がらなかった（10.0%、対照 10.5%）。伸びている区間（50→150M）の同じ腕では、上乗せは 30〜40M で消えていた（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ13・探索ブロック14。Run: run_20261019-184443_dmfb_42b_34a_lr7.1e-5）。
- B1024/RR1 の系列は 230M 以降も評価で伸び、評価専用 Run は 60M ごとに +2.3%（230M 2,051 → 290M 2,099 → 350M 2,148）だった。230–290M の学習中の評価では B256/RR1 の系列より約 1.6% 高く、伸びの傾きは同程度だった（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、現時点のまとめ14・探索ブロック15・16。Run: run_20261021-001853_dmfb_45a_42j_ctl / run_20261022-132905_dmfb_46a_45a_ctl / run_20261012-122028_dmfb_29_21_to300）。
- B1024/RR1 に加えて学習率も 7.1e-5 に下げる 2 段目のノイズ低下（230→290M）は、online と target の食い違い（4.6%）と 1 回の更新で行動が変わる割合（0.47%）をここまでの腕で最も小さくしたが、評価専用 Run は −0.4%（約 0.9σ）、学習中の評価は +0.3% で、上積みは無かった。学習側だけ +0.7〜1.0% 上がった（出典: [2026-10-15_gradient-noise-batch.md](../../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md)、探索ブロック15 探索結果・考察。Run: run_20261021-071918_dmfb_45b_42j_lr71e-6 / run_20261021-001853_dmfb_45a_42j_ctl）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
