# Survey: 分布強化学習における τ(quantile fraction)のサンプリング/配置方式

## 元の材料

[Survey: 分布強化学習における τ(quantile fraction)のサンプリング/配置方式](../../../../reports/tau_sampling_modes_survey.md) blob:1032ed1d34a3b19b38f6078577577b2375299999

## 要点

- FQF は τ をサンプルせず、状態ごとに softmax 出力の累積和として単調増加する τ 集合を生成する fraction proposal network を導入し、1-Wasserstein 距離最小化で学習する。「τ の配置を学習で非一様化する」研究として最も直接的である。
- 比例配分の層化抽出(1 次元で「K 等分ビンに 1 点ずつ」= jittered sampling と同型)は、iid モンテカルロより分散が大きくなり得ないという無条件の保証を持つ。
- 「有限サンプルで [0,1] 上の期待値を近似する」同型の問題に対し、隣接分野では層化・低差異列が実際に使われている。
- 分布RL(IQN/QR-DQN/FQF 系)の τ サンプリングに層化・jitter・low-discrepancy 列を明示的に適用した論文・実装・記事は、英語 Web 検索の範囲では見つからなかった(2026-08-12 時点。検索クエリ 4 種は `_survey_tau_mc_theory.md` §3.3 に記録)。この不在自体が本調査の主要な成果である [_survey_tau_mc_theory (2026/08), §3.3]。
- NDQFN による IQN/FQF 批判: 毎 iteration の τ 再サンプルは、固定分位位置を前提とする分布ベース探索ボーナス(DLTV 型)を極めて不安定にする。

## 内容

### 1. τ 生成方式の全体分類

収集したソース全体から、τ の生成・利用方式は次の系統に整理できる(各行の根拠は括弧内の節に引用)。

### 2. 原典論文における τ 配置の系譜

QR-DQN は分布を N 個の等重み Dirac で表し、各出力を固定 grid の中点 τ̂_i に対応させる。τ は学習中に変化しない。

IQN は online 側 τ と target 側 τ′ を U([0,1]) から独立にサンプルし、pairwise TD 誤差で quantile regression loss を構成する。

τ サンプル本数の ablation(N: online、N′: target、1〜64)では、N は初期性能に劇的な効果、N′ は 8 以降は長期性能への影響が最小、N = N′ = 8 で改善の大部分を達成、方策側 K=32 は感度なしと報告されている。

### 3. risk-sensitive・適応的な τ 分布の変形

IQN 原典は、行動選択に使う τ のサンプリング分布を distortion risk measure β で歪めることで risk-sensitive 方策を定式化した。評価された distortion は CPW・Wang・Pow・CVaR・Norm の 5 種である [IQN (2018/06)]。

CVaR(η,τ) = ητ は τ ∼ U([0,1]) を U([0,η]) へ線形に写すため、τ のサンプリングを下位区間に制限することと等価である(β の定義式からの直接の帰結)[IQN (2018/06)]。

固定リスクレベルを避け、α や楽観度 β を動的に決める研究群が確認できた。

### 4. モンテカルロ分散低減理論と隣接分野での層化・QMC 適用

Variational Diffusion Models(Kingma et al.; Google Research。NeurIPS 2021。被引用数 約1,570)は、拡散損失の timestep サンプリングに low-discrepancy sampler を使い分散を低減した。共有オフセット 1 個による等間隔配置で、各 1/k 幅のビンに 1 点が入る。

Improved DDPM(Nichol & Dhariwal; OpenAI。ICML 2021。被引用数 約5,728)は同じ問題に重点サンプリング(p_t ∝ √E[L_t²])で対処した [nichol-iddpm (2021/02), §3.3]。NeRF(Mildenhall et al.; UC Berkeley ほか。ECCV 2020)はレンダリング積分の求積点を「等間隔ビン内 1 点一様サンプル」で取ることを stratified sampling と呼んで採用している。

RL 一般では、方策勾配・actor-critic の行動サンプリングへ randomized QMC を適用した Arnold et al.(AISTATS 2022)がある [arnold-rqmc-rl (2022), Abstract]。

### 6. 主要フレームワーク実装の現状

評価/決定的行動時の等間隔 linspace 切替 — pfrl(`act_deterministically=True` 時)、d3rlpy(train/eval 分岐)。

### 7. 批判・懸念(ソースが自己申告・相互批判したもの)

FQF の自己申告: fraction proposal network の収束保証がなく、追加ネットワークにより IQN 比約 20% 低速である。

NC-QR-DQN による固定 grid 系批判: 非交差保証がないと方策探索の方向が歪み、最適行動の選択が学習エポック間で大きく変動する [NC-QR (2020/12), Section 1]。FQF 自身も「学習中の quantile fraction の選択はどの程度重要か」を open question として残している [FQF (2019/11), Discussion/Conclusion]。

### 8. 総合評価

τ の配置方式として文献上確立しているのは「固定一様 grid(QR-DQN 系)」「iid 一様サンプル(IQN 系)」「学習(FQF)」の 3 つであり、両者の中間である「被覆保証つきランダム化(層化/jitter/低差異列)」は、分布RL の τ に限っては論文・主要実装・実務者議論のいずれにも確認できなかった(第 5, 6 節)。

一方でその中間形は、モンテカルロ分散低減の一般理論では「比例配分の層化は分散を増やさない」という無条件保証を持つ標準技法であり(第 4.1 節)、拡散モデルの timestep・NeRF の求積点など「[0,1] 上の期待値を少数サンプルで近似する」同型問題では実際に採用されている(第 4.2 節)。

### 9. 調査の限界

「見つからなかった」系の結論(τ への層化/QMC 適用、τ の importance sampling、τ の Thompson sampling 解釈、stratified τ の GitHub issue)は英語 Web 検索+ヒットした論文フルテキスト確認に基づく。arXiv 全文検索・GitHub コード検索 API・中国語文献は未走査であり、悉皆調査ではない。

FQF の「学習された fraction が最終的にほぼ一様に退化する」という後続の再現報告は一次ソースを確認できなかった(citation not confirmed)。

## 主要な概念

- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
