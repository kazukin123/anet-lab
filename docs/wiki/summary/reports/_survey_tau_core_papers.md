# Survey: 分布強化学習における quantile fraction τ の選び方・配置方式の系譜(原典論文)

## 元の材料

[Survey: 分布強化学習における quantile fraction τ の選び方・配置方式の系譜(原典論文)](../../../../reports/_survey_tau_core_papers.md) blob:8186e66251071db35ab21513551088baca797e8d

## 要点

- τ の生成方式: 分布を N 個の等重み Dirac(各重み 1/N)で表し、各出力 θ_i は固定 grid の中点 τ̂_i = (τ_{i-1}+τ_i)/2 に対応する quantile を推定する。τ は学習中に一切変化しない。
- τ の生成方式: online 側の τ と target 側の τ′ をそれぞれ一様分布 U([0,1]) から独立にサンプルし、その間の pairwise TD 誤差で quantile regression loss を構成する。
- 学習信号: proposal network は 1-Wasserstein 距離 W1 の最小化を目標とし、W1 自体は直接計算できないため ∂W1/∂τ_i の解析勾配を optimizer に渡す。真の quantile 関数の代わりに quantile value network を用いる。
- 単調性の実現機構: softmax logit の累積和 ψ_{i,a} と、非負スロープ α(s,a)・切片 β(s,a) を出す Scale Factor Network の線形変換で、quantile 値の非交差を構成的に保証する。
- FQF は考察で「学習中の quantile fraction の選択はどの程度重要か」を未解決問題として挙げている。

## 内容

### 1. 系譜の概観

τ の生成方式は、原典論文ベースで次の 4 系統に整理できる(各主張の根拠は以降の各節の引用)。

### 3. IQN: τ ∼ U[0,1] のランダムサンプリングと cosine embedding

τ の埋め込み: τ を n 次元 cosine 基底で埋め込み、状態埋め込みと結合(Hadamard 積)する。

N / N′(τ サンプル本数)の ablation: 詳細は第 8 節に引用。方策側の τ サンプル数 K は 32 に固定された。

### 4. FQF: fraction proposal network による τ の学習

τ の生成方式: fraction proposal network が状態ごとに softmax 出力 q_j を生成し、その累積和として単調増加する τ_i を得る(τ_0 = 0, τ_N = 1 が構成的に保証される)。

安定化のための実装措置: proposal network の学習率は quantile value network より大幅に小さく設定される(sweep の結果 2.5e-9)。

### 5. Non-crossing 系: NC-QR-DQN と NDQFN

動機: 複数の quantile level を個別に回帰すると単調性の大域制約がなく、推定 quantile が交差(crossing)する。

τ の扱い: QR-DQN と同じく固定の非減少 quantile fraction 列を用い、各出力は中点 τ̂_i の quantile 推定である(τ 自体は変更しない)。

動機: QR-DQN / IQN / FQF の理論的正当性は quantile 曲線の非減少性を前提とするが、大域制約がないため保証されない。

構成: 固定の supporting points p 上で「基底値 + 非負増分の累積」として非減少 quantile 関数を構成する。p は端点をクリップした一様 grid。

### 6. TQC: 固定 τ + truncation(連続制御)

τ の扱い: QR-DQN と同じ固定 midpoint grid を使う(論文の準備節で QR-DQN の説明として τ_m = (2m−1)/2M, m ∈ [1..M] が示され、TQC の各 critic はこの QR-DQN 型パラメータ化をそのまま用いる)。各 critic は M 個の等重み Dirac atom で分布を表す。

truncation: N 個の critic の atom を全部プールしてソートし、大きい側を切り捨てて(小さい方 kN 個だけ残して)過大評価バイアスを制御する。

### 7. 連続制御系のその他: D4PG / DSAC

τ の扱い: D4PG の critic は quantile fraction を持たない。C51 系の categorical パラメータ化(固定 atom 上の離散分布)を採用しており、mixture of Gaussians も検討したが劣後した。

掲載誌の補足: OpenAlex には Journal of Artificial Intelligence Research, Vol. 83 (2025) の雑誌版レコードが存在する [OpenAlex-DSAC-Ma (2026/08)]

τ の生成方式: fixed(QR-DQN 式 τ_i = i/N)・random(IQN 式サンプリング)・net(FQF 式 proposal network)の 3 方式を比較し、random を採用した。

τ の埋め込みは IQN と同じ cosine embedding を用いる。

### 8. τ の本数・配置が性能に与える影響の知見

IQN は τ サンプル本数 N(online 側)と N′(target 側)を 1〜64 で振る ablation を行い、N は初期性能に劇的な影響を持ち、N′ は N′=8 以降は長期性能への影響がほぼないと報告した。

なお、N=1 で DQN に近づくという仮説(分布 RL の効果は補助損失によるという説の検証)も同じ ablation の動機として明記されている。

第 7.2 節に引用の通り、fixed(QR-DQN 式)・random(IQN 式)・net(FQF 式)を比較し「random sampling has better performance and fewer parameters」と結論した [DSAC-Ma (2020/04), Section 5.1]。

M-IQN は IQN の TD ターゲットにスケール付き log-policy 項を加えるのみで、τ のサンプリング方式自体への変更や知見は本文に見当たらなかった(調査範囲内)。

### 9. 批判・限界(各ソースが自己申告・他ソースを批判したもの)

第 8.5 節に引用(「distributional guarantees vanish ... collapses to its mean」)。

### 10. 調査の限界

IQN の N/N′ ablation の節番号は ar5iv 版の変換で "Section 3.1" と表示されたが、公式版 PDF の節番号と一致しない可能性があるため、位置表記は「N/N′ ablation の段落」とした。

FQF の「学習された fraction が最終的にほぼ固定/一様になる」という後続の再現報告は、本調査では一次ソースを確認できなかった(citation not confirmed)。FQF 自身が書いているのは第 9.1 節の引用(収束保証なし・速度低下)まで。

D4PG の発表会場は OpenReview がブラウザ認証で直接取得できなかったため、dblp レコード(ICLR (Poster) 2018)で代替確認した。

## 主要な概念

- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
