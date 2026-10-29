# τ配置とIQN比較

## 今の結論

直接比較を確認できなかった初期調査に対し、DropMergeの2 seedではstratifiedとantitheticが有力だった。M削減の速度利得とM64による十分なlate改善は確認されなかった。qbertでは行動を選ぶときのτを固定からランダムにすると、スキップなしでは評価の8面突破率が上がり、TempoRL分割のfullでは下がった（seed1）。

## 分かっていること

- 2026-08の調査で層化τの分布RLでの直接比較は確認できず、一般の分散低減理論をそのまま学習性能の保証にしていない。（出典: [tau_sampling_modes_survey.md](../../../reports/tau_sampling_modes_survey.md)、8. 総合評価）。
- 調査対象の主要実装ではτ生成の層化・quasi-randomの工夫を確認できなかった。評価時配置・リスク歪曲・FQFは別の軸として整理された。（出典: [_survey_tau_implementations.md](../../../reports/_survey_tau_implementations.md)、まとめ（τ生成方式の一覧））。
- FQFのfractionがほぼ一様へ収束するという後続報告は一次ソース未確認であり、確立した事実として採らない。（出典: [_survey_tau_core_papers.md](../../../reports/_survey_tau_core_papers.md)、10. 調査の限界）。
- 一般のMC理論の一部は教科書などの二次資料を経由しており、τ配置への探索も悉皆調査ではない。（出典: [_survey_tau_mc_theory.md](../../../reports/_survey_tau_mc_theory.md)、5. 調査の限界）。
- τを確率変数としたThompson samplingの直接の一次ソースは確認できていない。リターン分布の幅と認識的不確実性を混同しない。（出典: [_survey_tau_risk_adaptive.md](../../../reports/_survey_tau_risk_adaptive.md)、8. 調査限界）。
- DropMergeの2-seedではstratifiedとantitheticが有力で、M削減の速度利得やM64の十分なlate改善は確認されずM32を維持した。（出典: [2026-08-09_iqn.md](../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ6〜9）。
- qbertの50M終了checkpoint（seed1、各200ゲーム）で、評価時に行動を選ぶτを固定からランダム（学習中のTrain Actorと同じ8個の抽選、NoisyNetはμ-only）に替えると、8面突破率はスキップなしで50.5%→66.0%に上がり、TempoRL分割のfullで66.0%→50.5%に下がった。分割のsingleは68.5%→70.0%だった。（出典: [qbertの行動固定を評価時だけ外す比較](../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02。Run: `run_20261029-112021_evnt_off_a` / `run_20261029-114255_evnt_off_t` / `run_20261029-111656_evnt_spl_a_c` / `run_20261029-113025_evnt_spl_t_c` / `run_20261029-120441_evnt_spl_a_s` / `run_20261029-120013_evnt_spl_t_s`）。

## 分かっていないこと

なし

## 撤回・更新

なし。

## 関連

- [Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル](../../../reports/btr_hyperparams_survey_2026-08-26.md)。
- [Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分](../../../reports/btr_code_reading_2026-08-31.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../mechanisms/evaluation.md)。
- [仕組みと契約](../../design/200_dqn_agents.jp.md)。
