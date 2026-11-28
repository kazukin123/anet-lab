# 勾配ノイズの観測と対処に関するサーベイ

調査日: 2026-10-15  
範囲: 一般の確率的最適化、画像分類、言語モデル。RL固有の問題は、一般知見とPRD092の対応を考察する章で扱う。  
検証する仮説: 「学習後半の停滞に対し、勾配ノイズを観測してbatch size Bを増やすことは、一般事例から見ても妥当か」。  
資料の扱い: 論文と研究者・研究組織の一次資料を採用。原典の結果と、本調査による数学的説明・提案を区別する。  
成果物の範囲: 調査と検証案。PRD、コード、設定の変更や学習実験は行っていない。

## 目次

- [1. 調査の結論](#1-調査の結論)
- [2. 何をノイズと呼び、何を観測するか](#2-何をノイズと呼び何を観測するか)
  - [2.1 平均勾配と勾配推定のばらつき](#21-平均勾配と勾配推定のばらつき)
  - [2.2 勾配ノイズ、更新ノイズ、学習中の漂いを分ける](#22-勾配ノイズ更新ノイズ学習中の漂いを分ける)
- [3. 観測方法の一般事例](#3-観測方法の一般事例)
  - [3.1 McCandlish 2018: 二つのbatch sizeとGNS](#31-mccandlish-2018-二つのbatch-sizeとgns)
  - [3.2 Gray 2024: per-example勾配と低費用の観測](#32-gray-2024-per-example勾配と低費用の観測)
  - [3.3 Wen 2018: 共有された重みノイズの床](#33-wen-2018-共有された重みノイズの床)
  - [3.4 推奨する観測セット](#34-推奨する観測セット)
- [4. Bと学習率を変える一般事例](#4-bと学習率を変える一般事例)
  - [4.1 Smith 2017: 学習率減衰をB増加で置き換える](#41-smith-2017-学習率減衰をb増加で置き換える)
  - [4.2 Goyal 2017: 大きなBで学習軌道を保つ](#42-goyal-2017-大きなbで学習軌道を保つ)
  - [4.3 Shallue 2019: Bごとの調整と比較予算](#43-shallue-2019-bごとの調整と比較予算)
  - [4.4 Malladi 2022: Adamに対するscaling rule](#44-malladi-2022-adamに対するscaling-rule)
  - [4.5 GPT-3 2020: 実用規模でのGNSとbatch ramp](#45-gpt-3-2020-実用規模でのgnsとbatch-ramp)
  - [4.6 Merrill 2025: GNSの反証と分岐学習による検証](#46-merrill-2025-gnsの反証と分岐学習による検証)
  - [4.7 Naganuma 2026: optimizerに合うノイズ尺度](#47-naganuma-2026-optimizerに合うノイズ尺度)
- [5. B以外の対処と限界](#5-b以外の対処と限界)
  - [5.1 MomentumとAdam: 時間平均と座標ごとの調整](#51-momentumとadam-時間平均と座標ごとの調整)
  - [5.2 Importance sampling: 抽出の仕方を変える](#52-importance-sampling-抽出の仕方を変える)
  - [5.3 SVRG: 参照勾配で揺れを相殺する](#53-svrg-参照勾配で揺れを相殺する)
  - [5.4 Clipping: 外れ値による更新を抑える](#54-clipping-外れ値による更新を抑える)
  - [5.5 対処の選び方](#55-対処の選び方)
- [6. 批判と懸念](#6-批判と懸念)
  - [6.1 ノイズを全部消せばよいわけではない](#61-ノイズを全部消せばよいわけではない)
  - [6.2 Adamの優位をノイズだけで説明しない](#62-adamの優位をノイズだけで説明しない)
  - [6.3 実務指針もBと他の設定を一緒に評価する](#63-実務指針もbと他の設定を一緒に評価する)
- [7. PRD092への考察と検証案](#7-prd092への考察と検証案)
  - [7.1 GNSを測る方向は妥当だが、成績の因果判定には足りない](#71-gnsを測る方向は妥当だが成績の因果判定には足りない)
  - [7.2 現在のmicrobatch案は共有ノイズを固定している](#72-現在のmicrobatch案は共有ノイズを固定している)
  - [7.3 PERの重みもbatchに依存する](#73-perの重みもbatchに依存する)
  - [7.4 前処理つきGNSと実更新を区別する](#74-前処理つきgnsと実更新を区別する)
  - [7.5 B・学習率・更新回数を分けて比較する](#75-b学習率更新回数を分けて比較する)
  - [7.6 暫定PRDで再検討する記述](#76-暫定prdで再検討する記述)
- [8. 総合評価と調査の限界](#8-総合評価と調査の限界)
- [9. 出典一覧](#9-出典一覧)

## 1. 調査の結論

**Bを変える方向は妥当で、有力な比較対象である。ただし、GNSが大きいからBを増やせば成績が上がる、という決定則にはできない。** B増加の成功事例は画像分類から大規模言語モデルまである。一方、GNSと実際の有効batch sizeの対応にはoptimizer、曲率、抽出方法、比較予算への依存がある。[ICLR Smith (2017/11)](https://arxiv.org/abs/1711.00489)、[JMLR Shallue (2019/07)](https://jmlr.org/papers/v20/18-789.html)、[NeurIPS Merrill (2025/11), §2–4](https://arxiv.org/html/2505.23971v3)。

本調査の整理では、目的を三つに分けると判断しやすい。

| 目的 | 問うこと | 対処の候補 |
|---|---|---|
| 勾配推定を正確にする | 同じ状態でsampleを取り直すと、勾配がどれだけ変わるか | B増加、抽出方法、control variate |
| 重みの更新を安定させる | その勾配でoptimizerがどれだけ重みを動かすか | 学習率、momentum、前処理、clipping |
| 成績・時間・計算量を改善する | 評価性能を、どの予算で達成できるか | 上記の比較と学習スケジュール |

この表は文献を横断した本調査の分類であり、単一論文の提案ではない。根拠は§3–6に分けて示す。

## 2. 何をノイズと呼び、何を観測するか

### 2.1 平均勾配と勾配推定のばらつき

固定した重みθと学習分布に対し、sampleの損失勾配をg、平均をG、共分散をΣとする。独立同分布で抽出し、平均損失を使うなら、

$$
G=\mathbb E[g],\qquad
g_B=\frac1B\sum_{i=1}^{B}g_i,\qquad
\operatorname{Cov}(g_B)=\frac{\Sigma}{B}.
$$

測りやすいGNSは、

$$
\mathcal B_{\mathrm{simple}}
=\frac{\operatorname{tr}\Sigma}{\|G\|^2},
\qquad
r_B=\frac{\mathcal B_{\mathrm{simple}}}{B}.
$$

r_Bは「勾配推定のノイズ二乗／平均勾配の二乗」。Bを倍にするとノイズの**分散**は半分、標準偏差は約0.707倍になる。Gは定義した学習目的の平均勾配であり、評価性能へ必ず向かう方向という意味ではない。[arXiv McCandlish (2018/12), §2.2](https://arxiv.org/pdf/1812.06162)。

### 2.2 勾配ノイズ、更新ノイズ、学習中の漂いを分ける

単純なSGDの更新Δθ=−ηg_Bから、固定状態での一回の更新について、

$$
\operatorname{Cov}(\Delta\theta)=\eta^2\frac{\Sigma}{B}.
$$

**本調査の数学的説明:** ηを半分にすると信号の更新幅も半分になるため、GNSやr_B自体は変わらない。Bを倍にする場合は、期待勾配を保って勾配推定のばらつきを下げられる。学習率低下とB増加は、同じ操作ではない。長時間のSGDを連続時間近似したノイズ強度にη/Bが現れることと、一回の更新分散にη²/Bが現れることも区別する。SGDの定義と近似の出典は[ICLR Smith (2017/11), §2](https://arxiv.org/abs/1711.00489)。

また、連続したlearn stepの勾配はθも目的も変わる。その変動をそのまま「固定状態でのsample分散」と呼ぶことはできない。ノイズの測定と、学習中のドリフトの記録を分けるべき、というのが固定状態という定義からの帰結である。

## 3. 観測方法の一般事例

### 3.1 McCandlish 2018: 二つのbatch sizeとGNS

同じ状態で異なる大きさのbatchの勾配二乗ノルムを測り、信号と分散を分離する。比は不安定なので、信号と分散を別々に平滑化してから割る。単純GNSと、曲率Hを含む尺度は区別される。[arXiv McCandlish (2018/12), §2・Appendix A](https://arxiv.org/pdf/1812.06162)。

~~~text
This is strictly true only when training examples are sampled independently from the same data distribution.

これは、学習例が同じデータ分布から独立に抽出される場合にのみ厳密に成立する。
~~~
[arXiv McCandlish (2018/12), §2.2, footnote 6](https://arxiv.org/pdf/1812.06162)

サイズbのmicrobatchをk個とし、B=kb、各平均勾配g_jについて、q=(1/k)Σ_j||g_j||²、ḡ=(1/k)Σ_jg_jと置く。上記推定式の代数的な書き換えは、

$$
\widehat{\operatorname{tr}\Sigma}
=\frac{q-\|\bar g\|^2}{1/b-1/B},\qquad
\widehat{\|G\|^2}
=\frac{B\|\bar g\|^2-bq}{B-b}.
$$

一回の信号推定が負になることはあり得る。平均化後も分母が識別できない領域では、巨大な比だけを信頼しない。これは推定式の読み方である。曲率を含む尺度は、

$$
\mathcal B_{\mathrm{noise}}
=\frac{\operatorname{tr}(H\Sigma)}{G^\top HG}.
$$

単純GNSがこれに一致するのはHが単位行列に比例する場合。局所改善を予測するモデルであり、最終成績の保証ではない。[arXiv McCandlish (2018/12), §2.2–2.4](https://arxiv.org/pdf/1812.06162)。

### 3.2 Gray 2024: per-example勾配と低費用の観測

小batchの単位を小さくすると推定精度が改善する。検証したTransformerではLayerNormのGNSが全体GNSをよく予測し、専用kernelによる測定とbatch scheduleで学習時間を18%短縮した。DDPや勾配累積の単位で測る方法もあるが、推定精度が実行構成に依存する。[NeurIPS Gray (2024/11), §2・Appendix A](https://arxiv.org/html/2411.00999v1)。

~~~text
For the same number of samples processed, a smaller Bsmall always has a lower standard error

処理するsample数が同じなら、Bsmallが小さい方が標準誤差は小さい。
~~~
[NeurIPS Gray (2024/11), Figure 2](https://papers.nips.cc/paper_files/paper/2024/file/a9d419ef12fb34105424fa3166716139-Paper-Conference.pdf)

この結果は当該Transformer群での経験的結果であり、LayerNormだけ測れば任意のnetworkで十分という保証ではない。

### 3.3 Wen 2018: 共有された重みノイズの床

Flipoutは、batch内で同じ重み摂動を共有するとsample間の勾配が相関し、Bを増やしても分散が頭打ちになり得ることを示した。FC・CNN・LSTMで検証している。[ICLR Wen (2018/03), Theorem 2・§4.1](https://arxiv.org/pdf/1803.04386)。

~~~text
all the examples in a mini-batch typically share the same weight perturbation

ミニバッチ内のすべての例は通常、同じ重み摂動を共有する。
~~~
[ICLR Wen (2018/03), Abstract](https://arxiv.org/pdf/1803.04386)

**本調査の数学的説明:** 共有乱数Zのもとでsampleが条件付き独立なら、

$$
\operatorname{Cov}(g_B)
=
\frac{\mathbb E_Z[\Sigma_Z]}{B}
+\operatorname{Cov}_Z(G_Z).
$$

前者はBで減る。後者は、抽選された重みなどによって平均勾配そのものが変わる項で、Bを増やしても消えない。独立sampleの1/B則を、すべての確率性へ一律に当てはめられない。Flipoutは対称性・factorizationの条件を使う手法であり、NoisyNetでの効果を直接示した研究ではない。

### 3.4 推奨する観測セット

以下は本調査による実験設計の提案。原典の計測法を、診断する目的ごとに並べたもの。

| 観測 | 方法 | 判断できること／できないこと |
|---|---|---|
| 平均勾配の二乗と分散 | 同じθで複数のbatchまたはmicrobatch | 信号減少か、sample間の不一致増加かを区別する |
| GNSとr_B | 上記の信号・分散の比 | 相対的な推定ノイズ。成績や最適Bは直接決めない |
| Bに対する分散曲線 | 固定状態で複数Bを反復測定 | 1/Bに近いか、ノイズ床・相関があるか |
| 外れ値・clip発火 | gradient normの分位点、clip前後、発火率 | 少数の極端な更新が支配するか |
| 実際の更新幅 | Δθのnorm、必要なら重みnormとの比 | optimizerを経た歩幅。これ単独ではSNRではない |
| 方向の安定性 | 固定θで再抽出した勾配同士のcosine等 | 向きの不一致。単一cosineを成績閾値にしない |
| 評価成績と予算 | sample、更新数、実時間の各軸で比較 | 測った機序が、目的とする成果に結びつくか |

原典として、固定状態・再抽出の測定は§3.1・3.3および[NeurIPS Zhang (2019/12), §2](https://arxiv.org/pdf/1912.03194)、更新幅の定義は[ICLR Adam (2014/12), Algorithm 1](https://arxiv.org/pdf/1412.6980)、予算の比較は§4.3に依拠する。

## 4. Bと学習率を変える一般事例

### 4.1 Smith 2017: 学習率減衰をB増加で置き換える

学習率を減衰させる代わりにBを増やし、SGD、momentum、Nesterov、Adamで類似した学習曲線を得る実験がある。SGDの近似では、ノイズ強度に学習率、B、momentumが共同で関与する。ただし、Bを際限なく増やす処方ではない。[ICLR Smith (2017/11), §2–4](https://arxiv.org/abs/1711.00489)。

~~~text
one can usually obtain the same learning curve on both training and test sets

通常、訓練・テストの双方で同じ学習曲線を得られる。
~~~
[ICLR Smith (2017/11), Abstract](https://arxiv.org/abs/1711.00489)

同論文の学習率を含む動力学のnoise scaleと、McCandlishのGNSは別の量である。B固定でηを下げる対照は、B増加を解釈するために重要、というのが本調査の判断。

### 4.2 Goyal 2017: 大きなBで学習軌道を保つ

ResNet-50/ImageNetで、Bの倍率に合わせてSGDの学習率も比例増加し、初期warmupを組み合わせてB=8192まで精度を維持した。BatchNormの統計単位も管理している。[Facebook Goyal (2017/06), §2–3](https://arxiv.org/abs/1706.02677)。

~~~text
minibatch size cannot be scaled indefinitely

ミニバッチサイズは際限なく拡大できない。
~~~
[Facebook Goyal (2017/06), §2.3](https://arxiv.org/abs/1706.02677)

これは小batchの複数更新を大batchの一回で近似する並列化の事例。固定ηでノイズを下げる実験と同じ目的として扱わない。

### 4.3 Shallue 2019: Bごとの調整と比較予算

35 workloadsの大規模比較では、B増加の効果に比例改善・逓減・飽和の領域があり、境界はmodel・data・optimizerで変わる。Bごとに他の設定を調整し、目標成績に至る更新数を比較することが重要だった。[JMLR Shallue (2019/07), §4–5](https://jmlr.org/papers/v20/18-789.html)。

~~~text
We find no evidence that larger batch sizes degrade out-of-sample performance.

大きなバッチが標本外性能を悪化させるという証拠は得られなかった。
~~~
[JMLR Shallue (2019/07), Abstract](https://jmlr.org/papers/v20/18-789.html)

対象実験での結果であり、任意の大batchで汎化が維持される保証ではない。同じepochなら大Bの更新数が少なくなるため、比較予算によって解釈が変わる。

### 4.4 Malladi 2022: Adamに対するscaling rule

AdamについてSDE近似を保つ規則は、SGDの線形学習率則と異なる。B′=κBに対し、

$$
\eta'=\sqrt{\kappa}\eta,\quad
\beta_1'=1-\kappa(1-\beta_1),\quad
\beta_2'=1-\kappa(1-\beta_2),\quad
\epsilon'=\epsilon/\sqrt{\kappa}.
$$

画像分類・言語モデルで検証されたが、近似の成立条件と適用範囲がある。[NeurIPS Malladi (2022/05), §5, Definition 5.2](https://papers.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper-Conference.pdf)。

~~~text
a square root scaling rule to adjust the optimization hyperparameters of RMSprop and Adam when changing batch size

batch size変更時のRMSprop・Adamの最適化hyperparameterを調整する平方根則。
~~~
[NeurIPS Malladi (2022/05), Abstract](https://papers.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper-Conference.pdf)

これは近似的な動力学を保つ規則であって、ノイズを減らして停滞を抜ける処方そのものではない。βが許容範囲を外れる倍率や、近似が破れる大Bへ機械的に適用しない。

### 4.5 GPT-3 2020: 実用規模でのGNSとbatch ramp

GPT-3はGNSを測ってBの選択に使ったと明記する。Adamを使い、batchは32k tokensから最終値へ徐々に増加し、学習率warmup・cosine decay・clippingも併用した。[NeurIPS Brown (2020/05), §2.3・Appendix B](https://arxiv.org/pdf/2005.14165)。

~~~text
We measure the gradient noise scale during training and use it to guide our choice of batch size

学習中にgradient noise scaleを測り、batch sizeの選択の指針に使う。
~~~
[NeurIPS Brown (2020/05), §2.3](https://arxiv.org/pdf/2005.14165)

GNS利用の実例ではあるが、B変更だけの効果を分離した実験ではない。Bだけを主因と認定する根拠にはしない。

### 4.6 Merrill 2025: GNSの反証と分岐学習による検証

OLMo 1B・7Bでは、GNSが実測したcritical batch size（CBS）を数桁過小推定し、7Bでは推移も一致しなかった。同じcheckpointからBと学習率を変えて分岐し、固定token予算でlossの回復を確認する方法を採った。[NeurIPS Merrill (2025/11), §2–3](https://arxiv.org/html/2505.23971v3)。

~~~text
the gradient noise scale underestimates the CBS by several orders of magnitude

gradient noise scaleはCBSを数桁過小推定する。
~~~
[NeurIPS Merrill (2025/11), §3.3](https://arxiv.org/html/2505.23971v3)

測定をもとにBを1024→2048→4096へ増やすOLMo 1B実験では、固定小Bより更新数を43%減らし、最終lossはわずかに改善した。ただし短期のloss回復が続くという仮定を置き、全学習を再現した比較ではない。下流指標の全面的な優越も示していない。[同論文, §4, Tables 1–2](https://arxiv.org/html/2505.23971v3)。

**本調査の判断:** この反証はB変更を否定しない。GNSを診断として使うことと、Bの効果を実際に分岐実験で確かめることを組み合わせる根拠になる。

### 4.7 Naganuma 2026: optimizerに合うノイズ尺度

2026年2月のプレプリントは、signSGD・Signumやspectral descent・Muonに対し、その更新の幾何に合うGNSを導出した。160M Llama等で、一定Bの基準に近いvalidation lossを保ちながら、更新数を最大66%減らす実験を報告する。[arXiv Naganuma (2026/02), Abstract・§3–5](https://arxiv.org/html/2602.03001v1)。

~~~text
their assumption of SGD’s Euclidean geometry creates a fundamental mismatch

SGDのEuclidean geometryを仮定することが、根本的な不一致を生む。
~~~
[arXiv Naganuma (2026/02), Abstract](https://arxiv.org/html/2602.03001v1)

本調査ではプレプリントとして扱う。AdamWのPRD092への直接の正当化ではなく、「更新則に合う尺度か」が現役の研究課題であることを示す資料。

## 5. B以外の対処と限界

### 5.1 MomentumとAdam: 時間平均と座標ごとの調整

Adamは勾配のEMAと勾配二乗のEMAを使う。後者vは平均を引いた分散ではなくraw second momentであり、vだけからノイズ分散を読めない。[ICLR Adam (2014/12), §2・Algorithm 1](https://arxiv.org/pdf/1412.6980)。

~~~text
All operations on vectors are element-wise.

vectorに対するすべての演算は要素ごとに行う。
~~~
[ICLR Adam (2014/12), Algorithm 1](https://arxiv.org/pdf/1412.6980)

**本調査の導出:** θ固定・独立・定常なg_tを仮定した正規化EMA m_t=βm_{t−1}+(1−β)g_tの定常共分散は、

$$
\operatorname{Cov}(m_t)
=\frac{1-\beta}{1+\beta}\operatorname{Cov}(g_t).
$$

時間方向の平滑化には効果があるが、平均勾配が動くと遅れが生じる。通常の非正規化momentumは尺度が違うので、この式をそのまま使わない。Adamは時間平均と座標別の前処理を組み合わせるため、Bの代わりにβを上げれば同じという関係ではない。

### 5.2 Importance sampling: 抽出の仕方を変える

sample gradient normの上界を推定し、非一様抽出と逆確率の補正を使って分散を下げる研究がある。候補集合のscore計算が増えるため、分散低減が費用を上回る場合だけ有効化する。[ICML Katharopoulos (2018/07), §3–4](https://proceedings.mlr.press/v80/katharopoulos18a.html)。

~~~text
we only perform importance sampling when we know that the variance of the gradients can be reduced

勾配分散を減らせると分かった場合にだけimportance samplingを実施する。
~~~
[ICML Katharopoulos (2018/07), §3.3](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)

同論文ではhigh-loss sampleの選択が分散を増やす場合もある。「重要そうなsampleを多く取る」と「勾配分散を減らす」は同義ではない。原目的を保つ抽出補正と、目的自体を変えるhard miningを分ける。

### 5.3 SVRG: 参照勾配で揺れを相殺する

SVRGは参照点θ̃でのfull gradientを周期的に計算し、同じsampleの参照勾配を差し引く。

$$
\hat g_i(\theta)
=g_i(\theta)-g_i(\tilde\theta)+\nabla F(\tilde\theta).
$$

期待勾配を保って共通の揺れを相殺するcontrol variate（既知の平均を持つ補助量）の事例である。[NIPS Johnson (2013/12), §2](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf)。

~~~text
Unlike SGD, the learning rate ηt for SVRG does not have to decay

SGDと異なり、SVRGの学習率は減衰させる必要がない。
~~~
[NIPS Johnson (2013/12), §2](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf)

証明の中心はsmooth・strongly convexな有限和問題。full gradientと追加のsample gradientが必要なので、計算時間の面で常にSGDより得とは限らない。動く目的で参照がどれだけ有効かも別の問題になる。

### 5.4 Clipping: 外れ値による更新を抑える

BERTとImageNetの勾配ノイズ分布を比較した研究は、heavy-tailed noiseへの対処としてclippingを解析し、座標別のadaptive clippingをBERTで評価した。[NeurIPS Zhang (2019/12), §2–5](https://arxiv.org/pdf/1912.03194)。

~~~text
A simple strategy to circumvent this issue is to use a biased clipped stochastic gradient estimator.

この問題を回避する単純な方法は、バイアスを持つclipped stochastic gradient estimatorを使うことである。
~~~
[NeurIPS Zhang (2019/12), §3](https://arxiv.org/pdf/1912.03194)

clipは大きな更新を抑えるが、真の大きな信号も削り得る。平均勾配をより正確に推定するB増加と、外れ値に強くするclipは別の対処。heavy-tailが問題かどうかは観測する必要がある。

### 5.5 対処の選び方

以下は本調査の統合判断。前提を満たさない手法を、Bの代替として一律に推奨するものではない。

| 観測された問題 | 比較する対処 | 主な交換条件 |
|---|---|---|
| 独立sample由来の分散が大きい | B増加・勾配累積 | 計算量、メモリ、更新頻度。累積は同じθで平均して一度だけstepする |
| batchの抽出が分散を増やしている | importance sampling、構成を管理するsampling | score計算、抽出補正、目的の維持 |
| 同じ有限データで高精度まで追い込む | SVRG等のcontrol variate | full gradient費用、参照の有効性 |
| sample noiseより歩幅が大きすぎる | 学習率低下、前処理の見直し | 信号の進み方も遅くなる |
| 時間方向の揺れが大きい | momentum・EMA | 急な変化への追従遅れ |
| 少数の極端なgradientが支配する | clipping、lossのrobust化を検討 | バイアス、真の信号を削る可能性 |
| 共有乱数由来のノイズ床がある | 共有方法や乱数の平均化を検証 | forward費用、学習目的や探索との関係 |
| ノイズを下げても伸びない | 曲率、optimizer、データ、目的の変化を調べる | ノイズが主因という仮説を見直す |

表の根拠は§3.3・§4・§5.1–5.4。構成を管理するsamplingやrobustなlossの具体的採用は、この調査では実証していない。

## 6. 批判と懸念

### 6.1 ノイズを全部消せばよいわけではない

SGDのノイズには方向の偏りがあり、同じ大きさの等方的ノイズと同じ汎化効果にはならない。ノイズ共分散と曲率の関係を調べた研究がある。[ICML Zhu (2019/06), §3–4](https://proceedings.mlr.press/v97/zhu19e.html)。

~~~text
measuring the alignment of noise covariance and the curvature of loss function

ノイズ共分散と損失関数の曲率との整列を測定する。
~~~
[ICML Zhu (2019/06), §3](https://proceedings.mlr.press/v97/zhu19e/zhu19e.pdf)

一方、large-batchのgeneralization gapを更新回数・学習regimeの変更で解消した結果もある。[NIPS Hoffer (2017/05)](https://proceedings.neurips.cc/paper/2017/hash/a5e0ff62be0b08456fc7f1e88812af3d-Abstract.html)。

~~~text
the "generalization gap" stems from the relatively small number of updates rather than the batch size

generalization gapは、batch sizeよりも相対的に少ない更新回数から生じる。
~~~
[NIPS Hoffer (2017/05), Abstract](https://proceedings.neurips.cc/paper/2017/hash/a5e0ff62be0b08456fc7f1e88812af3d-Abstract.html)

本調査の判断は「ノイズを最小化する」を目的に置かず、評価成績と効率を目的に置くこと。上記の研究を「小Bは必ず良い」「大Bは必ず悪い」のどちらにも一般化しない。

### 6.2 Adamの優位をノイズだけで説明しない

Transformerでfull-batch化し、dropoutを無効化しても、SGDとAdamの差が消えない反証実験がある。[ICLR Kunstner (2023/04), §3・Appendix B.4](https://arxiv.org/abs/2304.13960)。

~~~text
Removing noise by running in full batch does not reduce the gap

full-batchで実行してノイズを除いても、性能差は縮まらない。
~~~
[ICLR Kunstner (2023/04), §1.1](https://openreview.net/pdf?id=a65YK0cqH8g)

これは対象Transformerでの原因切り分けであり、全モデルの説明ではない。ただし、ノイズを除く操作で解決しない問題があることは重要。Adamの効果を「ノイズを処理するから」の一言で説明しない。

### 6.3 実務指針もBと他の設定を一緒に評価する

Google Research等の研究者によるTuning Playbookは、Bを主に訓練速度・資源消費の選択として扱い、B変更時にはoptimizerとregularizationの設定を再調整するよう勧める。企業が公開した実務指針であり、査読論文やGoogleの公式サポート製品とは区別する。[Google Research Tuning Playbook (2023), Choosing the batch size](https://github.com/google-research/tuning_playbook#choosing-the-batch-size)。

~~~text
The optimal values of most hyperparameters are sensitive to the batch size.

大部分のhyperparameterの最適値はbatch sizeに敏感である。
~~~
[Google Research Tuning Playbook (2023), Changing the batch size requires re-tuning most hyperparameters](https://github.com/google-research/tuning_playbook#changing-the-batch-size-requires-re-tuning-most-hyperparameters)

本調査では、この教師あり学習中心の指針をRLへ無条件に適用しない。固定環境経験予算での性能を目的にする場合は、Bが更新数やreplayとの関係も変えるためである。

## 7. PRD092への考察と検証案

**この章は本調査による考察・提案であり、外部論文がanet-labへ適用を推奨しているという意味ではない。現行PRDの変更決定でもない。**

読んだローカル資料は[PRD092](../../docs/memo/done/0092_gradient_noise_scale_10rfd.md)、[ハイパラ自律調整の暫定PRD](../../docs/memo/0152_metric_targeted_hyperparam_control_10rfd.md)、[ADR0031](../../docs/adr/0031-plasticity-metrics-out-of-band-partial-forward.md)、[ADR0048](../../docs/adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md)。規約上、現在のコードと設定も必要箇所だけ読み取った。

### 7.1 GNSを測る方向は妥当だが、成績の因果判定には足りない

PRD092の「勾配normだけでは平均とばらつきを分けられない」「信号と分散を別々に記録する」「自動制御より先に測る」は、§3の一般事例と整合する。[PRD092, §1・4–5](../../docs/memo/done/0092_gradient_noise_scale_10rfd.md)。

ただし、simple GNSが256を超えたことだけで「B512の成績改善はノイズが原因」と言い切れない。逆に256を大きく下回っても、AdamでB増加が効く可能性を棄却できない。§4.6のOLMoは、まさにsimple GNSと実際の有効Bが一致しない事例である。GNSを機序の観測、分岐Runを介入効果の検証として使うのが、本調査の推奨。

「真の勾配」も、現時点のsampling・weighting・target・乱数で定義した目的の期待勾配として説明した方が正確。その目的の停留点にいることと、環境の成績が最適であることは別である。

### 7.2 現在のmicrobatch案は共有ノイズを固定している

PRD092の§5は一回のforwardからmicrobatch勾配を取る。§9の#11はNoisyNetの抽選をbatchで共有し、経験とτのばらつきを測ると明記している。基準設定もlearner.currentとlearner.targetにbatch共有のact_sampleを参照させる。[PRD092, §5・9](../../docs/memo/done/0092_gradient_noise_scale_10rfd.md)、[agent.txt, 130–139付近](../../apps/runner/config/agent.txt)、[nn_modules.cpp, 284–296付近](../../src/core/base/src/nn_modules.cpp)。

したがって現案を「NoisyNetも含む総更新ノイズ」と読むのは広すぎる。共有抽選Zを固定した条件付きのsample分散を主に測る案である。§3.3のノイズ床を別に観測しなければ、総ノイズがBでどれだけ減るかは分からない。

**例モデルによる独自導出:** microbatch勾配がg_j=μ+u(Z)+ε_j、共有成分の共分散がCなら、二B推定の期待値は、

$$
\mathbb E[\widehat{\operatorname{tr}\Sigma}]
=\operatorname{tr}\Sigma_\epsilon,\qquad
\mathbb E[\widehat{\|G\|^2}]
=\|\mu\|^2+\operatorname{tr}C.
$$

共有成分が信号側へ入るため、EMAを長くしても不足する分散項は復元できない。これはNoisyNetでの実測結論ではなく、全共分散と推定式からの導出。

比較用には、同じcheckpoint・replay分布で、共有ε固定のsample再抽出と、ε再抽選の反復を分ける。通常学習のRNGを変えない専用の診断として行う案であり、現PRDのmicrobatch方式を採用済みのまま拡張すると決めたものではない。

### 7.3 PERの重みもbatchに依存する

現在のPER実装は抽出確率から重みを作り、batch内最大値でweightsを割っている。[replay_buffer_impl.cpp:1062](../../src/core/base/src/replay_buffer_impl.cpp)。

本調査の考察では、この共通正規化係数はbatchの構成・大きさで変わり得るため、「B変更は分散だけ変え、期待勾配と有効歩幅は常に同じ」という前提に注意が要る。一回のbatch内で共通係数が固定ならその回の比への一様スケールは相殺されるが、係数が回ごとに変わるとEMAした信号と分散の比も別の重み付けになる。

診断では実際のPER・IS重みを保持する値と、固定した目的について1/B則を確認する較正を区別することを提案する。uniform samplingのprobeを実際の更新分布と同じものとして扱う提案ではない。

### 7.4 前処理つきGNSと実更新を区別する

PRD092は現在のAdamのv̂によるP=diag(1/(√v̂+ε))を勾配へ掛ける。[PRD092, §5.4](../../docs/memo/done/0092_gradient_noise_scale_10rfd.md)。

固定Pなら、PGとPΣPᵀに対するGNSを測れる。一方、Adamの実更新にはmの履歴と、今回の勾配によるvの更新も入る。clippingとAdamWのweight decayも、歩幅に影響する。したがって前処理つきGNSを「実際の歩幅のSNR」と同一視しないことを提案する。根拠となる更新式は§5.1。

最初の観測として生GNS・固定PによるGNS・実更新normを並べるのは有用。ただし実更新norm単独でもSNRにはならない。実更新のノイズまで比べる場合は、optimizer stateの同じsnapshotを使った候補batchごとの反実仮想更新が必要になる、というのが本調査の考察。

### 7.5 B・学習率・更新回数を分けて比較する

現コードはreplay_ratioが正の場合、更新creditをnum_envs×RR/Bで計算する。従ってRR固定でBを倍にすると、経験あたりの更新数が半減する。[dqn_based_agent.cpp:2037](../src/core/base/src/dqn_based_agent.cpp)。PRD092もこの交絡を記載している。[PRD092, §3](../../docs/memo/done/0092_gradient_noise_scale_10rfd.md)。

以下は、091によって継続Runの学習率変更が有効になった後の**提案**。実行していない。η₀は選んだ基準Runの実効学習率であり、新しい数値の指定ではない。

| 腕 | B | RR | 学習率 | 環境経験あたりの更新数（相対） | 主な問い |
|---|---:|---:|---|---:|---|
| A 基準 | 256 | 1 | η₀ | 1 | 基準のばらつき |
| B 学習率のみ低下 | 256 | 1 | η₀/2 | 1 | 小さい歩幅だけで改善するか |
| C B増加・更新数半減 | 512 | 1 | η₀ | 1/2 | 既存の有望腕を再現するか |
| D B増加・更新数維持 | 512 | 2 | η₀ | 1 | 高精度な勾配で同じ回数更新するとどうか |
| E B増加・更新数維持・歩幅低下 | 512 | 2 | η₀/2 | 1 | Cとの差が更新の粒度か |

この表は厳密な全因子実験ではない。D/Eはreplay使用量と取りこぼしも変える。まず固定checkpoint・固定replay分布でBに対する分散を較正し、その後にオンライン分岐を比較する。環境経験数、learn step、sampleされた経験総数、実時間、target同期・EMA・scheduleの単位を記録する。

一回の更新分散や、固定目的での局所効果を見る試験と、長期の環境成績を確認する試験を分ける。A–Eの一部が同じ集計歩幅を持っても、Adam、target、PER、更新順序のため厳密には等価ではない。複数の独立seed・checkpointから再現性を確認し、必要な反復数は実測した成績分散から決める。

Bとηの同時scalingで並列化を目的にする試験は、§4.4に沿う別の比較として扱う。固定ηでノイズを減らす試験と混ぜない。

### 7.6 暫定PRDで再検討する記述

以下は修正候補であり、本調査ではPRDを編集していない。

| 暫定記述 | 再検討する理由 | 候補となる扱い |
|---|---|---|
| §2.2: B≪GNSなら「よく効く」、B≫GNSなら「ほぼ効かない」 | 局所SGDモデルと独立性の条件が必要。§4.6ではAdamの反例 | 条件付きの理論的予測として説明する |
| §2.3: 一回の更新ノイズはおおむねη/B | 一回の分散はη²Σ/B。η/Bは連続時間近似の強度 | どの時間単位・観測量かを明記する |
| §2.5・§5: 実際の更新のノイズ | §9#11では共有NoisyNet抽選を固定する | conditional sample noiseと共有抽選の分散を区別する |
| §5: Adam前処理で歩幅のノイズを見る | 固定Pの勾配GNSとAdamの実更新は違う | 前処理つき診断として明記し、実更新幅を補助的に記録する |
| §7: GNSがBを超えればBの効き目を説明できる | 相関のみでは交絡を除けない | 分岐介入と合わせて機序を支持・反証する |
| §7: GNSが小さければノイズでは説明できない | 単純尺度の誤差、共有ノイズ、optimizer依存がある | 検討対象をsample分散の仮説に限定する |
| §7: 何回かの測定後に有限かつ正になる | G≈0や識別困難な領域でも比が有限とは限らない | 合成問題で正しさを検証し、実Runでは信号・分散・成立可否を読む |

自律調整の暫定PRDが要求する「観測量と成績の較正が先」という順序は維持する。GNS/B=1を、そのまま普遍的な制御目標へ置く根拠は今回の調査では得られなかった。[ハイパラ自律調整の暫定PRD, 前提として先に要る作業](../../docs/memo/0152_metric_targeted_hyperparam_control_10rfd.md)。

## 8. 総合評価と調査の限界

**本調査の総合評価:** PRD092の計測を進める方向は支持する。B増加は一般事例でも強いbaselineで、学習率低下との比較を優先する価値がある。ただし採用判断は「noiseを減らせたか」と「目的の成績・予算が改善したか」の両方で行う。GNSは原因を絞る観測量であり、成績改善や最適Bの決定値ではない。

推奨する順序は、固定状態での分散較正、共有ノイズとsample noiseの切り分け、Bと学習率の介入比較、その後に自律調整の較正である。曲率の高価な測定やoptimizer変更を、最初から必須にする提案ではない。

調査の限界:

- 一次論文・原著者の実務指針に対象を絞ったサーベイで、網羅的なsystematic reviewではない。SNSや第三者の解説は技術的結論の根拠に採用していない。
- 2026-10-15までの資料を検索した。Naganumaの2026年研究は確認したv1プレプリントとして扱い、査読採択を確認したとは主張しない。
- 各論文の被引用数は未取得。0として扱わず、数による順位付けは行わない。発行年月は初稿または参照版、会議年は別に記載する。
- PRD092に記載されたDropMerge成績は、PRDの記録として参照した。元Runのデータを再解析・再実行して確認した数値ではない。
- RLのsampling、PER、target、探索による非定常性へ、教師あり学習の結果を直接一般化できない。§7は本調査の仮説と検証案。
- 部分勾配による推定の費用、AMP、autogradによるON/OFF等価性は今回測定していない。PRDの実装受入検証として残る。
- レポート内の原文引用は各原典について短い抜粋に限る。数学的な説明やプロジェクトへの考察は引用ではなく、著者の原文と区別している。

## 9. 出典一覧

論文の著者は研究者として記載。所属は原典に記された当時の所属で、現在所属を保証しない。被引用数は全論文で未取得。

- [arXiv McCandlish, 2018/12] Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team. “An Empirical Model of Large-Batch Training.” arXiv技術報告。OpenAI、KaplanはJohns Hopkins University/OpenAI。 [原典](https://arxiv.org/abs/1812.06162)。
- [NeurIPS Gray, 2024/11] Gavia Gray, Aman Tiwari, Shane Bergsma, Joel Hestness. “Normalization Layer Per-Example Gradients are Sufficient to Predict Gradient Noise Scale in Transformers.” NeurIPS 2024。Cerebras Systems、TiwariはSubjective。 [会議掲載](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a9d419ef12fb34105424fa3166716139-Abstract-Conference.html)。
- [ICLR Wen, 2018/03] Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, Roger Grosse. “Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches.” ICLR 2018。University of Toronto/Vector Institute、TranはColumbia University/Google。 [原典](https://arxiv.org/abs/1803.04386)。
- [ICLR Smith, 2017/11] Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le. “Don't Decay the Learning Rate, Increase the Batch Size.” ICLR 2018。Google Brain。 [原典](https://arxiv.org/abs/1711.00489)。
- [Facebook Goyal, 2017/06] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.” arXiv技術報告。Facebook。 [原典](https://arxiv.org/abs/1706.02677)。
- [JMLR Shallue, 2019/07] Christopher J. Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, George E. Dahl. “Measuring the Effects of Data Parallelism on Neural Network Training.” JMLR 20(112); 初稿2018/11。Google Brain。 [掲載ページ](https://jmlr.org/papers/v20/18-789.html)。
- [NeurIPS Malladi, 2022/05] Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora. “On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.” NeurIPS 2022。Princeton University。 [原典](https://papers.neurips.cc/paper_files/paper/2022/file/32ac710102f0620d0f28d5d05a44fe08-Paper-Conference.pdf)。
- [NeurIPS Brown, 2020/05] Tom B. Brown et al. “Language Models are Few-Shot Learners.” NeurIPS 2020。OpenAIの研究チーム等。 [原典](https://arxiv.org/abs/2005.14165)。
- [NeurIPS Merrill, 2025/11] William Merrill, Shane Arora, Dirk Groeneveld, Hannaneh Hajishirzi. “Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training.” NeurIPS 2025; 初稿2025/05、参照版v3は2025/11。Allen Institute for AI。 [原典](https://arxiv.org/abs/2505.23971)、[会議掲載](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a99f732df9b668284b449da0214a3286-Abstract-Conference.html)。
- [arXiv Naganuma, 2026/02] Hiroki Naganuma, Shagun Gupta, Youssef Briki, Ioannis Mitliagkas, Irina Rish, Parameswaran Raman, Hao-Jun Michael Shi. “Adaptive Batch Sizes Using Non-Euclidean Gradient Noise Scales for Stochastic Sign and Spectral Descent.” v1プレプリント。Mila/Université de Montréal、Meta Platforms。 [原典](https://arxiv.org/abs/2602.03001)。
- [ICLR Adam, 2014/12] Diederik P. Kingma, Jimmy Lei Ba. “Adam: A Method for Stochastic Optimization.” ICLR 2015; 参照PDFは2017/01改訂版。University of Amsterdam/OpenAI、University of Toronto。 [原典](https://arxiv.org/abs/1412.6980)。
- [ICML Katharopoulos, 2018/07] Angelos Katharopoulos, François Fleuret. “Not All Samples Are Created Equal: Deep Learning with Importance Sampling.” ICML 2018。Idiap/EPFL。 [掲載ページ](https://proceedings.mlr.press/v80/katharopoulos18a.html)。
- [NIPS Johnson, 2013/12] Rie Johnson, Tong Zhang. “Accelerating Stochastic Gradient Descent using Predictive Variance Reduction.” NIPS 2013。RJ Research Consulting、Baidu/Rutgers。 [原典](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf)。
- [NeurIPS Zhang, 2019/12] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, Suvrit Sra. “Why are Adaptive Methods Good for Attention Models?” NeurIPS 2020; 参照PDFは2020/10改訂版。MIT、EPFL、Google Research。 [原典](https://arxiv.org/abs/1912.03194)。
- [ICML Zhu, 2019/06] Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, Jinwen Ma. “The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Sharp Minima and Regularization Effects.” ICML 2019。Peking University、ZhuはBeijing Institute of Big Data Researchも併記。 [掲載ページ](https://proceedings.mlr.press/v97/zhu19e.html)。
- [NIPS Hoffer, 2017/05] Elad Hoffer, Itay Hubara, Daniel Soudry. “Train longer, generalize better: closing the generalization gap in large batch training of neural networks.” NIPS 2017。Technion。 [会議掲載](https://proceedings.neurips.cc/paper/2017/hash/a5e0ff62be0b08456fc7f1e88812af3d-Abstract.html)。
- [ICLR Kunstner, 2023/04] Frederik Kunstner, Jacques Chen, J. Wilder Lavington, Mark Schmidt. “Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be.” ICLR 2023。University of British Columbia、SchmidtはCIFAR AI Chair/Amii。 [原典](https://arxiv.org/abs/2304.13960)。
- [Google Research Tuning Playbook, 2023] Varun Godbole, George E. Dahl, Justin Gilmer, Christopher J. Shallue, Zachary Nado. “Deep Learning Tuning Playbook.” 研究者・engineerによる実務指針。Google Research/Brain Team、ShallueはHarvard University。初版年はCITATION.bib、本文は2026-10-15閲覧。月は未確認。 [資料](https://github.com/google-research/tuning_playbook)、[書誌](https://raw.githubusercontent.com/google-research/tuning_playbook/main/CITATION.bib)。
