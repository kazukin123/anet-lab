# τ配置とIQN比較

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

DropMerge の 2 seed では stratified と antithetic が有力だった。M を減らした速度上の利得と、M64 による終盤の十分な改善は確認できなかった。
評価時の τ の乱数は成績にも影響する。qbert と Montezuma では、同じ重みのまま τ の選び方を変えて評価した結果がある。

## 分かっていること

### anet の知見

- qbertの分割・合算の50M終了checkpoint（full、学習seed1、評価seed1/2の計400ゲーム、NoisyNetはμ-only）では、Q側だけτがランダムなTQの全体8面突破率が58.0・79.5%、Q側とskip側が同じランダム抽選を読むTBは50.5・69.25%だった。差の95%区間は+7.50 [0.75, 14.50]・+10.25 [4.50, 16.25]ポイント。集中面（分割6・合算5）を除いた通過率から作るk_Qは0.521 [0.353, 0.696]・0.445 [0.218, 0.701]で、どちらも0.5を含んだ。TBを独立抽選のTB_indepに替えたk'も、主因の読みは両方「決めきれない」だった。これは評価条件の比較で、学習条件の比較ではない（出典: [Q側・skip側のノイズとτを分けた評価](../../../experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md)、分割・合算の「τ（実装前）」「独立のτ抽選との比較」・「G4の判断に使う差」。Run: `run_20261029-155111_evnt_p106_spl_ta` / `run_20261029-180635_evnt_p106_spl_tb_e2` / `run_20261029-214530_t106_eval_old_mrg_TQ_s2` / `run_20261029-220252_t106_eval_old_mrg_TB_s2`ほか同記録のτ条件の計20本）。
- DropMergeの2-seedではstratifiedとantitheticが有力で、M削減の速度利得やM64の十分なlate改善は確認されずM32を維持した。（出典: [2026-08-09_iqn.md](../../../experiments/default-dqn/dropmerge/2026-08-09_iqn.md)、現時点のまとめ6〜9）。
- qbertの50M終了checkpoint（seed1、各200ゲーム）で、評価時に行動を選ぶτを固定からランダム（学習中のTrain Actorと同じ8個の抽選、NoisyNetはμ-only）に替えると、8面突破率はスキップなしで50.5%→66.0%に上がり、TempoRL分割のfullで66.0%→50.5%に下がった。分割のsingleは68.5%→70.0%だった。（出典: [qbertの行動固定を評価時だけ外す比較](../../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック02。Run: `run_20261029-112021_evnt_off_a` / `run_20261029-114255_evnt_off_t` / `run_20261029-111656_evnt_spl_a_c` / `run_20261029-113025_evnt_spl_t_c` / `run_20261029-120441_evnt_spl_a_s` / `run_20261029-120013_evnt_spl_t_s`）。
- Montezumaのサンプル効率優先＋RNDの30Mの重み（IQNのτ 32本、NoisyNetなし、seed1）を評価だけで各32ゲーム測ると、ε=0でτ固定は400点以上19/32（平均259.4）、τ乱数は29/32（362.5）だった。ε=0.01ではτ固定が32/32（403.1）、τ乱数が32/32（412.5）だった（出典: [2026-11-08_montezuma-rnd-se-30m.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md)、現時点のまとめ5・探索ブロック04。Run: run_20261108-110233_mz_se_ev30m_2x2）。
- Breakout では UQE の上端を0.70にすると eval1 が492.5、0.90だと519.2だった。後者と0.99との差は1.6点で、両者の優劣は判定できなかった（出典: [Atari Breakout: UQE 探索方策の掃引](../../../experiments/default-dqn/atari/2026-09-07_uqe-exploration.md)、現時点のまとめ。Run: `run_20260907-005108_munch_uqe_s099e05`・`run_20260907-033751_munch_uqe_s09e05`・`run_20260907-062305_munch_uqe_s07e05`）。
- LunarLander で IQN の current τ 数を64から32へ減らすと、raw loss・勾配ノルム・clip ratio が低下した。loss が current sample 方向に和を取るため、この指標だけで過学習軽減とは判定できない（出典: [IQN導入とQR比較の探索履歴](../../../experiments/default-dqn/lunarlander/2026-08-08_iqn.md)、現時点のまとめ7、current τ の比較、`run_20260809-010602_ll_iqn_ctau32`）。
- 調査時の英語文献・公開実装の検索では IQN 系の学習用 τ に層化や低差異列を直接適用した例は見つからなかった。一般の積分推定の保証と RL 学習での採用実績を区別した。（出典: [Survey: 分布強化学習における τ(quantile fraction)のサンプリング/配置方式](../../../../reports/tau_sampling_modes_survey.md)、5. 分布RL の τ への stratified/QMC 適用の探索結果(負の結果を含む)）。

### 外の事例

- 2026-08の調査で層化τの分布RLでの直接比較は確認できず、一般の分散低減理論をそのまま学習性能の保証にしていない。（出典: [tau_sampling_modes_survey.md](../../../../reports/tau_sampling_modes_survey.md)、8. 総合評価）。
- 調査対象の主要実装ではτ生成の層化・quasi-randomの工夫を確認できなかった。評価時配置・リスク歪曲・FQFは別の軸として整理された。（出典: [_survey_tau_implementations.md](../../../../reports/_survey_tau_implementations.md)、まとめ（τ生成方式の一覧））。
- FQFのfractionがほぼ一様へ収束するという後続報告は一次ソース未確認であり、確立した事実として採らない。（出典: [_survey_tau_core_papers.md](../../../../reports/_survey_tau_core_papers.md)、10. 調査の限界）。
- 一般のMC理論の一部は教科書などの二次資料を経由しており、τ配置への探索も悉皆調査ではない。（出典: [_survey_tau_mc_theory.md](../../../../reports/_survey_tau_mc_theory.md)、5. 調査の限界）。
- τを確率変数としたThompson samplingの直接の一次ソースは確認できていない。リターン分布の幅と認識的不確実性を混同しない。（出典: [_survey_tau_risk_adaptive.md](../../../../reports/_survey_tau_risk_adaptive.md)、8. 調査限界）。
- IQNでτを毎回引くために同じ遷移でも引くたびにPERの優先度が揺らぐ点を論じた原典・公式の記述は、調査では見つからなかった。IQN・QR-DQNの原典はPERを組み込まず、DopamineのIQNもPERを使わない。BTRの実装はτをforwardのたびに引き直し、その学習stepのτから作った値（τの組ごとの|TD|を片方の軸で和、もう片方で平均）で優先度を毎回上書きする。Rainbow-IQNはIQNの損失の分布が広く一部の遷移が引かれすぎるとしてαを0.2にし、BTRも従った（出典: [PER の優先度の弱点と代替のサーベイ](../../../../reports/per_priority_weaknesses_survey_2026-11-18.md)、§3・§4.1・§4.2）。
- FQF は分位値と区切り位置を別々に学ぶ。位置を提案するネットワークの更新では、1-Wasserstein 距離そのものを計算せず、近似した分位値から位置の勾配を求める。学習途中の位置選択がどの程度重要かは原論文の未決事項である（出典: [Survey: 分布強化学習における quantile fraction τ の選び方・配置方式の系譜(原典論文)](../../../../reports/_survey_tau_core_papers.md)、4. FQF: fraction proposal network による τ の学習）。
- d3rlpy の調査対象実装では、学習時の一様乱数を推論時に等間隔の分位点へ切り替えた。この切り替えは学習時の分位点配置を変更する操作とは別である（出典: [実装サーベイ: IQN / FQF / QR-DQN における τ(quantile fraction)生成](../../../../reports/_survey_tau_implementations.md)、5. d3rlpy(Takuma Seno 氏、オフライン RL ライブラリ)【公式フレームワーク(個人発・現在は広く利用)】）。
- 対蹠ペアでの分散はペア間相関に依存し、単調性などの条件を外すと増える場合がある。比例配分の層化と同じ無条件の分散低減保証は使えない（出典: [Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用](../../../../reports/_survey_tau_mc_theory.md)、1. 一般理論: 分散低減サンプリング技法「1.3 対蹠変量 (antithetic variates)」）。
- ART-IQN は下側テールの分散からリスクレベルをオンラインで調節し、その水準で CVaR の τ 区間を制限する。状態ごとに分位点の配置自体を学習する FQF とは目的が異なる（出典: [分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択](../../../../reports/_survey_tau_risk_adaptive.md)、4. 適応的リスク調整 — 学習中に τ 分布・distortion を自動調整「Adaptive Risk-Tendency: Nano Drone Navigation with Distributional Reinforcement Learning(ART-IQN)」）。
- NDQFN の論点では、固定した分位点の差を使う探索ボーナスは、各更新で τ を引き直す IQN・FQF と組み合わせると不安定になりうる。同じ分布でもどの量を探索指標に使うかが条件になる（出典: [Survey: 分布強化学習における τ(quantile fraction)のサンプリング/配置方式](../../../../reports/tau_sampling_modes_survey.md)、7. 批判・懸念(ソースが自己申告・相互批判したもの)）。
- QR-DQN は固定した等幅区間の中点に対応する分位値を学ぶ。IQN は online と target の τ を独立に一様分布から引いて組み合わせごとの TD 誤差を使う。（出典: [Survey: 分布強化学習における quantile fraction τ の選び方・配置方式の系譜(原典論文)](../../../../reports/_survey_tau_core_papers.md)、3. IQN: τ ∼ U[0,1] のランダムサンプリングと cosine embedding）。
- 比例配分の層化では固定した被積分関数の積分推定量の分散は独立な一様抽出より大きくならない。層内の変動まで消えるわけではなく、学習中に変化する方策の成績を保証する定理ではない。（出典: [Survey: モンテカルロ積分の分散低減サンプリング理論と quantile/τ サンプリングへの適用](../../../../reports/_survey_tau_mc_theory.md)、1. 一般理論: 分散低減サンプリング技法・1.1 層化抽出 (stratified sampling)）。
- FQF の非一様な区切りは分布近似を改善するために学ぶ。CVaR で下側の τ を選ぶ方策とは、同じ非一様な分位点でも最適化する量が異なる。（出典: [分布強化学習における τ(quantile fraction)の非一様分布・distortion・適応的選択](../../../../reports/_survey_tau_risk_adaptive.md)、6. τ 自体の学習(fraction proposal)・Fully Parameterized Quantile Function for Distributional Reinforcement Learning(FQF)）。

## 横断的知見

- 補完: IQN の原典の整理では、学習用に2系列の τ を一様分布から別々に引き、方策用のサンプル数 K は32に固定した。実装調査の pfrl は決定的な行動選択時に、d3rlpy は推論時に、乱数の代わりに等間隔の τ を使う。条件の違い: 前者は原論文の学習手順と方策用サンプル数、後者は調査時点の実装で評価時に選ぶ分岐を記述する（出典: [_survey_tau_implementations](../../../../reports/_survey_tau_implementations.md)、まとめ(τ 生成方式の一覧)。[_survey_tau_core_papers](../../../../reports/_survey_tau_core_papers.md)、3. IQN: τ ∼ U[0,1] のランダムサンプリングと cosine embedding）。
- 補完: 一様分布の層化抽出では、区間を分け、各層の幅に比例した本数の τ を引くと、同数の独立な一様乱数に比べ、同じ期待値の推定分散は増えない。リスクの調査では τ の区間を変えて評価するリターンの側を選ぶ場合と、FQF が分布の近似誤差を減らす場合を区別する。条件の違い: 層化の理論は固定した期待値の推定を扱い、リスクの調査は評価目的や分布の近似の変更も扱う（出典: [_survey_tau_risk_adaptive](../../../../reports/_survey_tau_risk_adaptive.md)、§2・§3（下位・上位分位の選択）、§6（FQF の分位点学習）。[_survey_tau_mc_theory](../../../../reports/_survey_tau_mc_theory.md)、1. 一般理論: 分散低減サンプリング技法「1.1 層化抽出 (stratified sampling)」）。

## 見立て

なし

## 分かっていないこと

- Train Actorのskip側のτだけ固定にすると、qbertの中盤の学習の遅れが改善するか（出典: [Q側・skip側のノイズとτを分けた評価](../../../experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md)、「再現と残る判断」）？
- 調査した11実装には、学習時に層化サンプリングや低差異列を使う例が見つからなかった。学習時の配置を変えた場合の成績差を示す比較実験は、この調査からは分からない。（出典: [実装サーベイ: IQN / FQF / QR-DQN における τ(quantile fraction)生成](../../../../reports/_survey_tau_implementations.md)、まとめ（τ 生成方式の一覧）・§10.6）。

## 撤回・更新

なし。

## 関連

- [Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル](../../../../reports/btr_hyperparams_survey_2026-08-26.md)。
- [Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分](../../../../reports/btr_code_reading_2026-08-31.md)。
- [atari](../subjects/atari.md)。
- [evaluation](../measurement/evaluation.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [仕組みと契約](../../../design/200_dqn_agents.jp.md)。
