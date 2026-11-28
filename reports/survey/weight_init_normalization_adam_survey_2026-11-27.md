# 正規化付きの層と Adam での、重みの初期値の大きさの効き方（短い調査）

Date: 2026-11-27
Scope: スケール不変な重み（正規化で出力が重みの大きさに依らない重み）と Adam の実効学習率、weight decay との関係、深層強化学習（Atari の DQN 系・IMPALA の ResNet）で初期値を比べた報告、公開実装の初期値の選び方。一次の資料（論文の本文と付録、公開実装のコード、公式の文書）だけを読んだ。

## 目次

1. [対象の問い](#1-対象の問い)
2. [結論](#2-結論)
3. [調べた範囲](#3-調べた範囲)
4. [見つからなかったこと](#4-見つからなかったこと)
5. [スケール不変な重みと Adam の実効学習率](#5-スケール不変な重みと-adam-の実効学習率)
6. [深層強化学習で初期値や正規化を比べた報告](#6-深層強化学習で初期値や正規化を比べた報告)
7. [公開実装の初期値](#7-公開実装の初期値)
8. [留保と食い違い](#8-留保と食い違い)
9. [この報告での読み取り（推論）](#9-この報告での読み取り推論)
10. [調査の限界](#10-調査の限界)
11. [出典の一覧](#11-出典の一覧)

## 1. 対象の問い

- 正規化（spectral normalization、LayerNorm、weight normalization など）の付いた層では、出力が重みの大きさに依らない。このとき Adam（1 回の更新幅が勾配の大きさに依らない）で学ぶと、重みの初期値の大きさが、学習の速さ（1 回の更新で重みが変わる割合、実効的な学習率）にどう効くか。これを扱った文献（scale-invariant な重みと実効学習率、weight decay との関係など）は何と書いているか。
- 深層強化学習（とくに Atari の DQN 系、Impala の ResNet）で、重みの初期値（He、orthogonal、PyTorch の既定、Xavier、Dopamine の variance scaling）を比べた報告はあるか。結果と条件は。
- 公開実装の初期値の選び方: PyTorch の Conv2d・Linear の既定の初期値の式、Dopamine の既定、BTR（Beyond The Rainbow）の公開実装、Stable-Baselines3、CleanRL の DQN 系。公式の文書かコードで確かめる。

## 2. 結論

- **問い 1。** スケール不変な重みの実効学習率は、SGD では η/‖w‖²、Adam では η/‖w‖ とされる（van Laarhoven 2017、§5.3・Table 1。Lyle ら 2024 の Definition 1 も同じ形）。したがって初期値のノルムが大きいほど、1 回の更新で重みの向きが変わる割合は小さい。weight decay があると、ノルムは η と λ で決まる平衡へ向かい、平衡では初期値の影響が消える。初期値が効くのは平衡に着くまでの過渡期で、初期のノルムが平衡より小さければ回りすぎ、大きければ回らなすぎる（Kosson ら 2024、§5.2）。weight decay が無いとノルムは単調に増え、実効学習率は下がり続ける（Heo ら 2021、§2・Table 1）。SN を Adam で学ぶと、Adam の ε がスペクトル半径倍に効く形になる（Gogianu ら 2021、§5.4）。
- **問い 2。** Atari の DQN 系や IMPALA の ResNet で、初期値の方式そのものを比べた報告は見つからなかった。近いものは、正規化の有無を比べた報告である。DQN-Adam の出力の 1 つ手前の層に SN を掛けると、54 ゲームの人間正規化スコアの平均が 358.45 から 719.95 になった（Gogianu ら 2021）。LayerNorm 付きの Rainbow で実効学習率を一定に固定すると多くのゲームで悪化し、学習率の線形の減衰で埋まった（Lyle ら 2024）。
- **問い 3。** PyTorch の既定は U(±1/√fan_in)（bias も同じ）である。Dopamine の Rainbow と IQN は 1/√3 の variance scaling（fan_in、一様）で、DQN は Keras の既定（TF）か Xavier（JAX）である。BTR、Stable-Baselines3 の DQN、CleanRL の DQN 系は、conv と線形層に初期化を書かず PyTorch の既定を使う。全層を直交初期化（gain √2）にするのは、確かめた中では PPO 系の実装（SB3 の `ActorCriticPolicy`、CleanRL の `ppo_atari.py`）と CleanRL の PQN だけだった。

## 3. 調べた範囲

- 日付: 2026-11-27。
- 検索語（一般の技術の語だけ）: weight initialization comparison deep Q-network Atari orthogonal He Xavier、"initialization" deep reinforcement learning Atari "orthogonal initialization" ablation、"weight initialization" "deep reinforcement learning" Atari DQN effect of initialization scale、DQN Rainbow Atari "initialization scheme" ablation kaiming、IMPALA ResNet CNN reinforcement learning weight initialization comparison Procgen、論文の題（Obando-Ceron らの題、Schmidt と Schmied の題、Lyle らの題）。
- 読んだ論文（本文と付録の該当の節）: van Laarhoven 2017、Zhang ら（ICLR 2019 の投稿版）、Salimans と Kingma 2016、Heo ら 2021、Kosson ら 2024、Lyle ら 2024、Gogianu ら 2021、Schmidt と Schmied 2021、Obando-Ceron ら 2024、Clark ら（BTR、arXiv v2）。
- 読んだコード: PyTorch v2.5.1 の `linear.py`・`conv.py`、Dopamine の TF 版（e7d780d）の `atari_lib.py` と JAX 版（master）の `networks.py`、Flax（main）の `linear.py`、BTR（093ca77）の `networks.py`、Stable-Baselines3（master）の `dqn/policies.py`・`common/policies.py`・`common/torch_layers.py`、CleanRL（master）の `dqn_atari.py`・`dqn_atari_jax.py`・`qdagger_dqn_atari_impalacnn.py`・`ppo_atari.py`・`ppo_procgen.py`・`pqn_atari_envpool.py`、OpenAI の phasic-policy-gradient（master）の `impala_cnn.py`・`torch_util.py`。master と main は 2026-11-27 に取得した版である。
- 被引用数は OpenAlex（2026-11-27 取得）の値である。

## 4. 見つからなかったこと

- Atari の DQN 系（DQN・Rainbow・IQN）や IMPALA の ResNet で、He・直交・PyTorch の既定・Xavier・Dopamine の variance scaling を並べて比べた報告。上の検索語で探し、Obando-Ceron ら 2024 の 12 の hyper-parameter（§4 の一覧）、Schmidt と Schmied 2021 の改良の一覧、BTR の論文の「試して効かなかったもの」の一覧（Appendix H）も見たが、初期値は入っていなかった（§6）。
- weight decay の無い Adam で、初期値の大きさの差がどれだけの更新回数で消えるかを定量した文献。Kosson らの平衡の解析は weight decay を前提にする。
- LayerNorm の付いた構成で、初期値を小さくしても学習が速くならない理由を論じた文献。
- 内的報酬のネットの初期値の比較（RLeXplore）は、既存の報告 [intrinsic_reward_landscape_survey_2026-11-18.md](intrinsic_reward_landscape_survey_2026-11-18.md) にあるので読み直さなかった。

## 5. スケール不変な重みと Adam の実効学習率

### 5.1 実効学習率の式

正規化の層の直前の重みは、出力が重みの倍率に依らない（スケール不変）。このとき勾配は重みのノルムに反比例し、同じ更新でも重みが大きいほど向きは変わりにくい。van Laarhoven は、重みを α 倍した系の更新が、学習率と正則化を替えた単位ノルムの系の更新と同じになることから、実効学習率を求めた。SGD とモメンタムでは η/‖w‖₂²、RMSProp と Adam では η/‖w‖₂ である（§3.1、§5.1〜§5.3、Table 1）。Adam は、重みを α 倍した系が、学習率を η/α、ε を αε にした系と同じ更新になる（§5.3）。実験は BN 付きの 4 層の CNN を CIFAR-10 で学んだものである（§6）。

```
Again, the effective learning rate is η/‖w‖₂.

ここでも実効学習率は η/‖w‖₂ である。
```

[van Laarhoven (2017/06), §5.3]

Lyle らも同じ形を定義に置く。ρ = 1/‖θ‖ として、勾配そのものの更新では ηρ²、正規化した勾配の更新では ηρ である（Definition 1）。付録では、Adam と RMSProp を「各パラメータの更新の大きさが固定の最適化」として扱う（Appendix C.4 の式 (30) の前）。Heo らは、スケール不変な重みの単位球の上での実効的な歩幅が、重みのノルムに反比例すると導く（§2.3、式 (5)）。

```
the effective step sizes are automatically reduced over time

実効的な歩幅は時間とともに自動的に小さくなる
```

[Heo AdamP (2021), Abstract]

正規化の種類で、スケール不変になる単位が違う。BN と Weight Standardization はニューロンごと、LayerNorm は層全体である（Kosson ら 2024、§2）。Lyle らは、bias の無い線形・ReLU のネットの最後に正規化を置くと、各層の重みの倍率がほかの層の勾配の大きさに効かなくなり、各層の実効学習率はその層の倍率で決まると書く（Appendix A.6、式 (17)〜(19)）。

### 5.2 weight decay との関係

スケール不変な重みでは、weight decay は重みの大きさを決め、それを通じて実効学習率を決める。van Laarhoven は、正規化と組むと L2 正則化には正則化の効果が無く、重みの大きさを通じて実効学習率に効くと書く（Abstract、§3.1）。正則化が無ければ重みは際限なく大きくなり、実効学習率は 0 へ向かう（§3.1）。RMSProp と Adam では、重みの大きさは η/λ の 3 乗根で決まる（§5.2、§6）。交差検証の最適な学習率は、SGD とモメンタムで λ に反比例し、RMSProp と Adam で λ の平方根に反比例した（§6、Figure 4）。

Zhang らは、BN 付きのネットを SGD か Adam で学ぶとき、weight decay の汎化の効果は主に実効学習率の引き上げを通じたものだとする（§4.1）。weight decay なしのネットの各層のノルムを、weight decay ありのネットの同じ epoch の値へ毎 epoch 合わせると、差のほとんどが消えた（ResNet32・CIFAR-100、Figure 4）。この節は SGD を中心に書かれ、Adam でも同様だったと添えている（§4.1）。

Kosson らは AdamW の平衡を導いた。ノルムの 2 乗の期待値は、初期値の項が (1 − 2ηλ + η²λ²) の割合で毎回縮み、更新の項が足される形で、平衡のノルムは √(ηC/2λ) にほぼ等しい（C は重みのベクトルの次元。§3.2、式 (15)〜(18)）。平衡の 1 回の更新の角度は η と λ と β1 で決まり、勾配の大きさにも初期値にも依らない（Table 1）。Adam に L2 正則化を足した形では、平衡が勾配の大きさに依り、層やニューロンごとに回り方がそろわない（§3.3）。

### 5.3 weight decay が無いとき（ノルムの増加）

勾配がスケール不変な重みに直交するので、勾配降下の 1 回ごとにノルムの 2 乗が更新の 2 乗だけ増える（Heo ら 2021、Lemma 2.1。Arora らの Lemma 2.4 の引用）。モメンタムを使うと、さらに速く増える（Lemma 2.2、Corollary 2.3）。Salimans と Kingma も weight normalization で同じ単調な増加を示し、Adam ではこの性質は厳密には成り立たないが、定性的には同じだったと書く（§2.1）。

```
the norm of v grows monotonically with the number of weight updates

v のノルムは重みの更新の回数とともに単調に増える
```

[Salimans WeightNorm (2016/06), §2.1]

Heo らの Table 1 は実際のノルムの増え方を示す。ImageNet の ResNet-18 を weight decay 0 の AdamW（Adam と同じ）で学ぶと、ノルムは最初の epoch の 1.70 から最後の 10.41 になり、精度は 68.38 だった。ノルムが増えないように更新の半径方向の成分を除く AdamP では 1.29 から 3.02 で、70.55 だった。音楽のタグ付けの HCNN では AdamW（weight decay 0）が 7.35 から 11.56、AdamP が 7.30 から 7.62 だった（§3.2、Table 1）。Salimans と Kingma は、ノルムが増えられることで学習率に頑健になり、学習率が大きすぎてもノルムが速く増えて適当な実効学習率に落ち着くと書く（§2.1）。

### 5.4 初期値の大きさが効く時期（過渡期）

Kosson らによると、学習の初めの過渡期には、回り方が初期のノルムと勾配の大きさで決まり、そのあと平衡に入る（§1、Figure 1）。平衡より小さいノルムの重みは平衡より速く回り、大きいノルムの重みは遅く回る。そのため初期のノルムが平衡と比べてどうかで、過渡期に回りすぎるか回らなすぎるかが決まる（§5.2、Figure 4）。学習率の warmup を入れても、過渡期に速く回りすぎる例があった（ResNet-20・CIFAR-10、§5.2）。回り方を直接決める変種（RV）では過渡期が無くなり、warmup の必要が減った（§5.2、Figure 5）。

```
Weight norms below the equilibrium magnitude result in rotation that is faster

平衡の大きさより小さい重みのノルムは、より速い回転を生む
```

[Kosson RotationalEquilibrium (2024/06), §5.2]

van Laarhoven も、weight decay が無いときの実効学習率の下がり方は制御しにくく、学習の初めの数 step に大きく左右され、結果の再現を難しくすると書く（§7）。

### 5.5 SN と Adam（強化学習）

Gogianu らは、C51 と DQN の Q ネットの、層の重みの数が最も多い出力の 1 つ手前の層だけに SN を掛けた。全層に掛けると成績が下がった（§3.1）。Atari 54 ゲームの人間正規化スコア（平均・中央値）は次のとおりで、評価は Hessel らの手順（250K step ごとに 125K step）による（Table 1、Table 8、Appendix C.1）。1 ゲームあたりの seed の数は、読んだ範囲で確かめられなかった。

| 構成 | 平均 | 中央値 |
|---|---:|---:|
| DQN-Adam（著者らの実装） | 358.45 | 119.45 |
| DQN-Adam SN[-2] | 719.95 | 178.18 |
| C51（著者らの実装） | 778.68 | 182.26 |
| C51 SN[-2] | 1073.18 | 248.45 |
| DQN（RMSProp、著者らの実装） | 357.36 | 102.94 |
| DQN SN[-2]（RMSProp） | 375.37 | 105.19 |

DQN-Adam の学習率は 0.00025、Adam の ε は 0.0003125 である（Table 7）。MinAtar では、6 つのモデル・4 ゲーム・2 seed で学習率と ε を振った 14000 の DQN を学び、SN が Adam の使える hyper-parameter の幅を広げた（§5.1、Figure 3）。

著者らは、SN の効きは主に最適化を通じたものだとする（§5.4）。SN は、重みが学習で大きくなるにつれて、更新の大きさと向きを変えるスケジューラとして働くという。

```
a scheduler that adapts the size and the direction of the optimisation step

最適化の 1 歩の大きさと向きを変えるスケジューラ
```

[Gogianu SN-DRL (2021/05), §5.4]

具体的には、SN は勾配をスペクトル半径 ρ で割る。Adam では、この勾配の縮みが、ε を ρ 倍にした更新とほぼ同じになる（式 (14)〜(16)）。勾配を ρ で割るだけの形（DIVGRAD）と ε に ρ を掛けるだけの形（MULEPS）で、MinAtar で SN の成績の多くを再現した（§5.4、Figure 6・7）。

### 5.6 LayerNorm と実効学習率（強化学習）

Lyle らは、正規化の層は重みのノルムの増加と実効学習率の減衰を同じものにすると書く（Abstract、§3.2）。提案の NaP は、各非線形の前に LayerNorm を入れ、各層の重みを周期的に初期のノルムへ射影して、実効学習率を明示のスケジュールに従わせる（§3.3、Algorithm 1）。

DQN Zoo の Rainbow の各非線形の前（最後の softmax を除く）に LayerNorm を入れ、Atari 57 ゲームを 200M フレーム学んだ（Appendix B.2）。射影で実効学習率を一定にすると、射影しない LayerNorm の Rainbow より悪いことが多かった（§4.2、Figure 3）。

```
frequently results in worse performance compared to unconstrained parameters

制約しないパラメータに比べて、しばしば成績が悪くなる
```

[Lyle NaP (2024/07), §4.2]

学習率を 6.25e-5 から 1e-6 へ線形に下げる（パラメータのノルムの平均の増え方にほぼ比例する）と差が埋まった（§4.2。本文の表記は 6.25·10⁻⁵ と 10⁻⁶。Appendix B.2 は Rainbow の既定の学習率を 0.000625 と書き、本文と桁が違う）。57 ゲーム・4 seed の集計（IQM と平均、95% の信頼区間）では、この減衰を付けた NaP が、素の Rainbow、LayerNorm だけ、減衰の無い NaP を上回った（§5.3、Figure 5 右）。集計の数値は図だけで、本文に値は無い。

## 6. 深層強化学習で初期値や正規化を比べた報告

### 6.1 weight normalization の DQN（Salimans と Kingma 2016）

DQN（Adamax、モメンタム 0.5、minibatch 64）に weight normalization と、データに依る初期化を入れた（§5.4）。学習率は {0.0001, 0.0003, 0.001, 0.003} から探し、通常の形では 0.0001、weight normalization では 0.0003 が良かった（§5.4）。4 ゲームの最高の評価点（通常の形 対 weight normalization）は、Breakout 410 対 403、Enduro 1,250 対 1,448、Seaquest 7,188 対 7,375、Space Invaders 1,779 対 2,179 だった（Figure 6 の表）。seed の数の記載は見当たらない。初期値の方式を比べた実験ではない。

### 6.2 SN と IMPALA の CNN の Rainbow（Schmidt と Schmied 2021）

Rainbow に IMPALA の大きい CNN を使い、SN を掛ける範囲を 3 通り（なし、6 つの残差 block の conv すべて、最後の 2 block だけ）比べた（§4.3）。5 ゲーム、3 seed の中央値の 100 エピソードの移動平均で、学習は 10M フレームである（§4.1、§4.3、Figure 2）。SN は学習の初めに最も効き、最初の進歩までの時間を大きく縮めた。Breakout・Seaquest・Tennis で強く見えた（§4.3）。ノイズ付きの最後の線形層だけに掛ける形は成績が悪く、すぐ捨てた（§4.3）。

```
It significantly reduces the time until initial learning progress is made.

最初の学習の進歩が出るまでの時間を大きく縮める。
```

[Schmidt FastRainbow (2021/11), §4.3]

同じ報告は、学習率を 4 倍（DQN と同じ値）にすると学習が大きく速くなり、最終の成績も上がったと書く（§4.5）。初期化の比較は無い（本文を init で検索した）。

### 6.3 BTR の論文（Clark ら）

BTR は IMPALA の各段の残差 block の conv に SN を掛ける（Appendix E.1）。BattleZone だけで試したものとして、線形層に SN を掛けると成績が大きく下がったと書く（Appendix H）。初期化の記述は本文と付録に無かった（本文を init で検索した）。

### 6.4 hyper-parameter の大きな比較（Obando-Ceron ら 2024）

DER と DrQ(ε) で、26 ゲーム・5 seed・約 108k の Run を回した（§1）。調べた hyper-parameter は活性化関数、正規化（BN と LayerNorm）、ネットの幅と深さ、Adam の学習率・ε・weight decay、ε-greedy、報酬の clip、割引率、replay の容量と開始の量、batch、更新の地平、target の更新の間隔などで、初期化は入っていない（§4）。

## 7. 公開実装の初期値

### 7.1 PyTorch の既定

`nn.Linear` の重みと bias は U(−√k, √k)、k = 1/in_features から引く（`linear.py` の docstring）。`nn.Conv2d` は k = groups ÷（C_in × kernel の要素数）である（`conv.py` の docstring）。コードは `init.kaiming_uniform_(self.weight, a=math.sqrt(5))` で、bias は U(±1/√fan_in) である（`Linear.reset_parameters`、`_ConvNd.reset_parameters`）。

```
Setting a=sqrt(5) in kaiming_uniform is the same as initializing with

kaiming_uniform で a=sqrt(5) とするのは、次で初期化するのと同じである
```

[PyTorch v2.5.1 (2024/10), torch/nn/modules/linear.py, Linear.reset_parameters]

この式の由来は Issue #15314 にある。保守者（soumith、2019-03-28）は、乱数を直接引いていた既定を `kaiming_uniform_` を使う形へ書き換えたとき、結果を変えないために √5 を置いたと答え、元は Torch7・Torch5 から続く形だと書いた。

### 7.2 Dopamine

TF 版（e7d780d の `atari_lib.py`）では、`RainbowNetwork` と `ImplicitQuantileNetwork` が conv 3 層・全結合・出力（IQN では τ の射影 `dense_quantile` も）を `VarianceScaling(scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')` で初期化する。`NatureDQNNetwork` は初期化を指定せず、Keras の既定になる。選んだ理由のコメントは無い。

JAX 版（master の `networks.py`）では、Rainbow・QR・IQN の Network が同じ variance scaling を使い、`NatureDQNNetwork` は `xavier_uniform` を使う。`FullRainbowNetwork` は conv とノイズなしの全結合が `xavier_uniform` で、ノイズ付きの層は μ を ±1/√fan_in の一様、σ を 0.1/√fan_in にする。IMPALA 型の `ImpalaEncoder` は、各段の最初の conv だけ `xavier_uniform` を指定し、残差 block の conv は Flax の既定になる。Flax の既定は `lecun_normal`、bias は 0 である（Flax main の `linear.py` の `default_kernel_init`）。

### 7.3 BTR（093ca77 の `networks.py`）

conv、τ の cos の埋め込み（`nn.Linear(self.n_cos, self.conv_out_size)`）、ノイズなしの線形層と Dueling の層には初期化を書かず、PyTorch の既定のままである。`ImpalaCNNLargeIQN` は SN を `torch.nn.utils.parametrizations.spectral_norm` で、各段の残差 block（`ImpalaCNNResidual`）の 2 つの conv にだけ掛け、各段の最初の conv には掛けない。因子分解のノイズ付き線形層（`FactorizedNoisyLinear.reset_parameters`）は、μ を U(±1/√in_features)、σ を σ0/√in_features（σ0 の既定は 0.5）にする。この関数のコメントは「Kaiming uniform に似る」と書くが、μ の幅は PyTorch の既定の Linear と同じである（本報告の照合）。

### 7.4 Stable-Baselines3（master）

DQN の `QNetwork` と `DQNPolicy` は初期化を書かず、CnnPolicy の特徴抽出の `NatureCNN` も初期化を書かないので、PyTorch の既定のままである。PPO・A2C の `ActorCriticPolicy` は `ortho_init=True` が既定で、特徴抽出と隠れ層を gain √2、行動の出力を 0.01、価値の出力を 1 の直交初期化にし、bias を 0 にする。コメントは、この gain が openai/baselines から来たと書く。

### 7.5 CleanRL（master）

- DQN 系: `dqn_atari.py`（Nature 型）と `qdagger_dqn_atari_impalacnn.py`（IMPALA CNN、16・32・32 channel）は初期化を書かず、PyTorch の既定のままである。`dqn_atari.py` の学習率の既定は 1e-4 で、Adam の ε は指定しない。JAX 版の `dqn_atari_jax.py` も `kernel_init` を指定せず、Flax の既定になる。
- PPO 系: `ppo_atari.py` は `layer_init`（直交初期化、gain の既定 √2、bias 0）を全層に使い、方策の出力を 0.01、価値の出力を 1 にする。`ppo_procgen.py` の IMPALA CNN は、conv と 256 の線形層が PyTorch の既定のままで、方策と価値の出力の 2 つだけ `layer_init` を使う。
- PQN: `pqn_atari_envpool.py` は conv 3 層と 512 の線形層の後ろに LayerNorm を置き、出力を含む全層を gain √2 の直交初期化にする。最適化は RAdam（学習率の既定 2.5e-4）で、既定で学習率を線形に下げる。

### 7.6 OpenAI の phasic-policy-gradient（IMPALA CNN）

`NormedConv2d` と `NormedLinear` は、PyTorch の既定で作った重みを、出力のユニットごとの L2 ノルムが `scale` になるように縮め、bias を 0 にする（`torch_util.py`）。`ImpalaCNN` は段の数の平方根で、`CnnDownStack` は block の数の平方根で scale を割り、`CnnBasicBlock` はその平方根を 2 つの conv に使う。最後の線形層は scale 1.4 である（`impala_cnn.py`）。

## 8. 留保と食い違い

- 実効学習率の式は、正規化で厳密にスケール不変になった重みについてのものである。bias は SN でも正規化されず、Gogianu らは bias が重みと一緒に縮まないことを断っている（§5.4）。
- Adam の η/‖w‖ は ε を無視した近似である。van Laarhoven の対応は ε も α 倍に変わることを含み（§5.3）、Gogianu らは SN の効きの多くを ε の変化で説明する（§5.4）。
- Kosson らの解析は、勾配が雑音で占められるランダムウォークの近似で、weight decay を前提にする（§3）。
- Lyle らの学習率の既定の値は、本文（6.25·10⁻⁵）と付録（0.000625）で桁が違う（§4.2、Appendix B.2）。
- Gogianu らの Atari の表の seed の数と学習の長さは、読んだ範囲で確かめられなかった。Salimans と Kingma の DQN の表も seed の数が無い。
- Schmidt と Schmied は学生の著者の workshop の報告で、5 ゲーム・3 seed・10M フレームの範囲である。

## 9. この報告での読み取り（推論）

ここから下は原典が書いていない、本報告の推論である。

- anet の qbert の記録が置いた試算（SN 付きの層は出力が重みの大きさに依らず、Adam の更新幅も依らないので、初期値が大きいと 1 回の更新で重みが変わる割合が小さくなる）は、§5.1 の Adam の η/‖w‖ と同じ形である。ε を無視した近似である点も同じで、§5.5 の Gogianu らの議論からは、大きい初期値は SN 付きの層の Adam の ε を相対的に大きくする向きにも効くと読める。ε が更新に効くほど勾配が小さいかは、外の文献からは決まらない。
- weight decay の無い構成なら、§5.3 のとおりノルムは更新とともに増えるので、初期値の差は、更新で積み上がったノルムが初期のノルムを上回るまで残り、その後は薄まると読める。どれだけの更新回数で薄まるかを示した文献は見つからなかった（§4）。
- Lyle らの Appendix A.6 の形（最後の正規化で手前の bias の無い ReLU の層がスケール不変になる）を当てると、胴体の後ろに LayerNorm を置いた構成でも、手前の層の初期値には同じ向きの効きが予想される。したがって、その構成で初期値を戻して学習が速くならなかったことは、§5 の文献の筋だけでは説明できない。
- pre-LN の Transformer（各 sublayer の入力に LayerNorm、残差の流れは正規化しない）の手前の conv の重みは、残差の流れにそのまま足されるので、§5.1 の意味でスケール不変にはならないと読める。この形の初期値の効きを扱った文献は探していない。
- 確かめた公開実装では、Atari の価値ベースの実装の多くが PyTorch の既定（BTR、SB3、CleanRL）か Dopamine の variance scaling を使い、He（fan_out、正規分布）を明示する実装は無かった。

## 10. 調査の限界

- PDF は文字の抽出で読んだので、式の添字や平方根の記号が崩れた所がある。崩れた式は、本文の言葉で書かれた所だけを使った（van Laarhoven の Table 1 の平衡の列は写していない）。
- master・main の版のコードは 2026-11-27 の取得で、以後に変わりうる。行番号ではなくクラス名と関数名で指した。
- PPO の実装の細部や連続制御での初期値の報告は、問いの範囲の外として読まなかった。
- Lyle らの集計の値、Schmidt と Schmied の Figure 2 の値は図だけで、数値を写していない。

## 11. 出典の一覧

論文（一次）:

- [van Laarhoven, 2017/06] Twan van Laarhoven（Radboud University）. "L2 Regularization versus Batch and Weight Normalization." arXiv:1706.05350v1. 被引用 428. https://arxiv.org/abs/1706.05350
- [Zhang ICLR, 2018/10] Guodong Zhang, Chaoqi Wang, Bowen Xu, Roger Grosse（University of Toronto, Vector Institute）. "Three Mechanisms of Weight Decay Regularization." arXiv:1810.12281v1（ICLR 2019 の投稿版）. 被引用 320. https://arxiv.org/abs/1810.12281
- [Salimans WeightNorm, 2016/06] Tim Salimans, Diederik P. Kingma（OpenAI）. "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks." arXiv:1602.07868v3. 被引用 2357. https://arxiv.org/abs/1602.07868
- [Heo AdamP, 2021] Byeongho Heo, Sanghyuk Chun ほか（Naver AI Lab, Naver Clova, Yonsei University）. "AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights." ICLR 2021（arXiv:2006.08217v3）. 被引用 226. https://arxiv.org/abs/2006.08217
- [Kosson RotationalEquilibrium, 2024/06] Atli Kosson, Bettina Messmer, Martin Jaggi（EPFL）. "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks." ICML 2024（arXiv:2305.17212v4）. 被引用 75. https://arxiv.org/abs/2305.17212
- [Lyle NaP, 2024/07] Clare Lyle, Zeyu Zheng, Khimya Khetarpal, James Martens, Hado van Hasselt, Razvan Pascanu, Will Dabney. "Normalization and effective learning rates in reinforcement learning." NeurIPS 2024（会議の名は検索結果と OpenAlex の DOI 10.52202/079017-3378 による。本文は arXiv:2407.01800v1 を読んだ）. 被引用 73. https://arxiv.org/abs/2407.01800
- [Gogianu SN-DRL, 2021/05] Florin Gogianu, Tudor Berariu, Mihaela Rosca, Claudia Clopath, Lucian Busoniu, Razvan Pascanu（Bitdefender, Technical University of Cluj-Napoca, Imperial College London, DeepMind, UCL）. "Spectral Normalisation for Deep Reinforcement Learning: An Optimisation Perspective." ICML 2021（arXiv:2105.05246v1）. 被引用 92. https://arxiv.org/abs/2105.05246
- [Schmidt FastRainbow, 2021/11] Dominik Schmidt, Thomas Schmied（TU Wien）. "Fast and Data-Efficient Training of Rainbow: an Experimental Study on Atari." NeurIPS 2021 Deep RL Workshop（会議の名は検索結果による。本文は arXiv:2111.10247v1 を読んだ）. 被引用 20. https://arxiv.org/abs/2111.10247
- [Obando-Ceron HPConsistency, 2024/11] Johan Obando-Ceron, João G.M. Araújo, Aaron Courville, Pablo Samuel Castro（Mila ほか）. "On the consistency of hyper-parameter selection in value-based deep reinforcement learning." RLC 2024（arXiv:2406.17523v3）. 被引用 25. https://arxiv.org/abs/2406.17523
- [Clark BTR, 2025/05] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare（University of Southampton）. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." arXiv:2411.03820v2. 被引用 13. https://arxiv.org/abs/2411.03820

コードと公式の文書（一次）:

- [PyTorch v2.5.1, 2024/10] PyTorch. `torch/nn/modules/linear.py`（Linear の docstring と reset_parameters）. https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/linear.py
- [PyTorch v2.5.1 conv, 2024/10] PyTorch. `torch/nn/modules/conv.py`（Conv2d の docstring と _ConvNd.reset_parameters）. https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/conv.py
- [PyTorch Issue 15314, 2018/12] PyTorch. "Kaiming init of conv and linear layers, why gain = sqrt(5)." Issue #15314（保守者の回答は 2019-03-28）. https://github.com/pytorch/pytorch/issues/15314
- [Dopamine TF, e7d780d] Google. `dopamine/discrete_domains/atari_lib.py`. https://github.com/google/dopamine/blob/e7d780d7c80954b7c396d984325002d60557f7d1/dopamine/discrete_domains/atari_lib.py
- [Dopamine JAX, 2026/11] Google. `dopamine/jax/networks.py`（master、2026-11-27 取得）. https://github.com/google/dopamine/blob/master/dopamine/jax/networks.py
- [Flax linear, 2026/11] Google. `flax/linen/linear.py`（main、2026-11-27 取得）. https://github.com/google/flax/blob/main/flax/linen/linear.py
- [BTR code, 093ca77] VIPTankz. `networks.py`. https://github.com/VIPTankz/BTR/blob/093ca77/networks.py
- [SB3, 2026/11] DLR-RM. Stable-Baselines3 の 3 つのファイル（master、2026-11-27 取得）. https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/policies.py 、https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py 、https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
- [CleanRL, 2026/11] vwxyzjn. CleanRL の 6 つのファイル（master、2026-11-27 取得）. https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py 、https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py 、https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_impalacnn.py 、https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py 、https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_procgen.py 、https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py
- [OpenAI PPG, 2026/11] OpenAI. phasic-policy-gradient の 2 つのファイル（master、2026-11-27 取得）. https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/impala_cnn.py 、https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/torch_util.py
