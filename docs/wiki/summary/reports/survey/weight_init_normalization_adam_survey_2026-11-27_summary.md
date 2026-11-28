---
title: 正規化付きの層と Adam での、重みの初期値の大きさの効き方（短い調査）
page_kind: サマリ
material: "[[reports/survey/weight_init_normalization_adam_survey_2026-11-27.md|weight_init_normalization_adam_survey_2026-11-27]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# 正規化付きの層と Adam での、重みの初期値の大きさの効き方（短い調査）

## 元の材料

[正規化付きの層と Adam での、重みの初期値の大きさの効き方（短い調査）](../../../../../reports/survey/weight_init_normalization_adam_survey_2026-11-27.md) blob:49349f2af50d3ff54cca6e4a57b8b45d2311c7eb

## 要点

- 正規化で出力が重みの倍率に依らなくなった重み（スケール不変な重み）の実効学習率は、SGD とモメンタムで η/‖w‖₂²、RMSProp と Adam で η/‖w‖₂ である [van Laarhoven (2017/06), §5.3、Table 1]。Lyle らの Definition 1 も同じ形を置く。
- weight decay があると、重みのノルムは η と λ で決まる平衡へ向かい、AdamW の平衡の 1 回の更新の角度は初期値にも勾配の大きさにも依らない [Kosson RotationalEquilibrium (2024/06), §3.2、Table 1]。
- 初期値が効くのは平衡までの過渡期で、平衡より小さいノルムの重みは速く、大きいノルムの重みは遅く回る。warmup を入れても過渡期に速く回りすぎる例があった（ResNet-20・CIFAR-10）[Kosson RotationalEquilibrium (2024/06), §5.2]。
- weight decay が無いと、スケール不変な重みのノルムは単調に増え、実効的な歩幅は小さくなる。ImageNet の ResNet-18 を weight decay 0 の AdamW で学ぶと、ノルムは 1.70 から 10.41 になった [Heo AdamP (2021), §2、Table 1]。
- SN を Adam で学ぶと、SN の勾配の縮みは ε をスペクトル半径 ρ 倍にした更新とほぼ同じになる。DQN-Adam の出力の 1 つ手前の層に SN を掛けると、Atari 54 ゲームの人間正規化スコアの平均・中央値は 358.45・119.45 から 719.95・178.18 になった [Gogianu SN-DRL (2021/05), §5.4、Table 1]。
- LayerNorm 付きの Rainbow（Atari 57 ゲーム、200M フレーム）で重みを初期のノルムへ射影して実効学習率を一定にすると、射影しないものより悪いことが多く、学習率を 6.25e-5 から 1e-6 へ線形に下げると差が埋まった [Lyle NaP (2024/07), §4.2、§5.3]。
- Atari の DQN 系や IMPALA の ResNet で初期値の方式そのものを比べた報告は、2026-11-27 の調査では見つからなかった（§4）。
- 公開実装では、PyTorch の既定は U(±1/√fan_in)（bias も同じ）、Dopamine の Rainbow と IQN は 1/√3 の variance scaling（fan_in、一様）である。BTR、Stable-Baselines3 の DQN、CleanRL の DQN 系は初期化を書かず PyTorch の既定を使う（§7）。

## 内容

### 1〜4. 問い・結論・範囲・見つからなかったこと

問いは 3 つで、正規化付きの層と Adam での初期値の大きさの効き方、深層強化学習で初期値を比べた報告、公開実装の初期値の選び方である。論文 10 本と、公開実装のコード・公式の文書を読んだ。被引用数は OpenAlex の値である。見つからなかったのは、Atari の DQN 系・IMPALA の ResNet で He・直交・PyTorch の既定・Xavier・variance scaling を並べた比較、weight decay の無い Adam で初期値の差がどれだけの更新回数で消えるかの定量、LayerNorm の付いた構成で初期値を小さくしても学習が速くならない理由を論じた文献である。

### 5. スケール不変な重みと Adam の実効学習率

Adam は、重みを α 倍した系が学習率を η/α、ε を αε にした系と同じ更新になる。実験は BN 付きの 4 層の CNN の CIFAR-10 である [van Laarhoven (2017/06), §5.3、§6]。単位球の上での実効的な歩幅は重みのノルムに反比例する [Heo AdamP (2021), §2.3]。スケール不変になる単位は、BN と Weight Standardization ではニューロンごと、LayerNorm では層全体である [Kosson RotationalEquilibrium (2024/06), §2]。bias の無い線形・ReLU のネットの最後に正規化を置くと、各層の実効学習率はその層の倍率で決まる [Lyle NaP (2024/07), Appendix A.6]。

weight decay との関係: 正規化と組むと L2 正則化には正則化の効果が無く、重みの大きさを通じて実効学習率に効く。RMSProp と Adam では重みの大きさは η/λ の 3 乗根で決まり、交差検証の最適な学習率は SGD とモメンタムで λ に反比例、RMSProp と Adam で λ の平方根に反比例した [van Laarhoven (2017/06), §3.1、§5.2、§6、Figure 4]。BN 付きのネットの weight decay なしの各層のノルムを、weight decay ありの同じ epoch の値へ毎 epoch 合わせると、汎化の差のほとんどが消えた（ResNet32・CIFAR-100）[Zhang ICLR (2018/10), §4.1、Figure 4]。AdamW の平衡のノルムは √(ηC/2λ) にほぼ等しく、Adam に L2 正則化を足した形では平衡が勾配の大きさに依り、回り方がそろわない [Kosson RotationalEquilibrium (2024/06), §3.2、§3.3]。

weight decay が無いとき: 勾配降下の 1 回ごとにノルムの 2 乗が更新の 2 乗だけ増え、モメンタムではさらに速く増える [Heo AdamP (2021), Lemma 2.1、Lemma 2.2]。AdamP は ResNet-18 で 1.29 から 3.02、精度 70.55（AdamW は 68.38）、音楽のタグ付けの HCNN では AdamW が 7.35 から 11.56、AdamP が 7.30 から 7.62 だった [Heo AdamP (2021), Table 1]。weight normalization はノルムが増えられることで学習率に頑健になり、Adam でも定性的に同じだった [Salimans WeightNorm (2016/06), §2.1]。

過渡期: 回り方を直接決める変種（RV）では過渡期が無くなり、warmup の必要が減った [Kosson RotationalEquilibrium (2024/06), §5.2、Figure 5]。weight decay が無いときの実効学習率の下がり方は制御しにくく、学習の初めの数 step に大きく左右される [van Laarhoven (2017/06), §7]。

SN と Adam: Gogianu らは、層の重みの数が最も多い出力の 1 つ手前の層だけに SN を掛け、全層に掛けると成績が下がった。C51 は 778.68・182.26 から 1073.18・248.45、RMSProp の DQN は 357.36・102.94 から 375.37・105.19 だった。DQN-Adam の学習率は 0.00025、ε は 0.0003125 である。MinAtar の 14000 の DQN で、SN が Adam の使える hyper-parameter の幅を広げた。勾配を ρ で割るだけ（DIVGRAD）、ε に ρ を掛けるだけ（MULEPS）でも SN の成績の多くを再現した [Gogianu SN-DRL (2021/05), §3.1、§5.1、§5.4、Table 1、Table 7、Table 8]。

LayerNorm と実効学習率: NaP は各非線形の前に LayerNorm を入れ、重みを周期的に初期のノルムへ射影する。57 ゲーム・4 seed の集計では、学習率の減衰を付けた NaP が、素の Rainbow、LayerNorm だけ、減衰の無い NaP を上回った（値は図だけ）[Lyle NaP (2024/07), §3.3、§5.3、Figure 5]。

### 6. 深層強化学習で初期値や正規化を比べた報告

weight normalization の DQN（Adamax）では、良かった学習率が通常の形の 0.0001 に対し 0.0003 で、4 ゲームの最高の評価点は Breakout 410 対 403、Enduro 1,250 対 1,448、Seaquest 7,188 対 7,375、Space Invaders 1,779 対 2,179 だった [Salimans WeightNorm (2016/06), §5.4]。IMPALA の CNN の Rainbow で 6 つの残差 block の conv すべてに SN を掛けると、最初の進歩までの時間が大きく縮み、Breakout・Seaquest・Tennis で強く見えた（5 ゲーム、3 seed、10M フレーム）。学習率を 4 倍にすると学習が大きく速くなった [Schmidt FastRainbow (2021/11), §4.3、§4.5]。BTR の論文は線形層に SN を掛けると成績が大きく下がったと書き（BattleZone だけ）、初期化の記述は無い [Clark BTR (2025/05), Appendix E.1、Appendix H]。DER と DrQ(ε) の 12 の hyper-parameter の比較に初期化は入っていない [Obando-Ceron HPConsistency (2024/11), §1、§4]。

### 7. 公開実装の初期値

PyTorch は `kaiming_uniform_(weight, a=math.sqrt(5))` と書き、保守者は旧い既定と結果を変えないための書き方だと説明した [PyTorch Issue 15314 (2018/12)]。Dopamine の TF 版の `NatureDQNNetwork` は Keras の既定、JAX 版は Rainbow・QR・IQN が variance scaling、DQN が `xavier_uniform` で、`ImpalaEncoder` の残差 block の conv は Flax の既定（`lecun_normal`）である。BTR は SN を各段の残差 block の 2 つの conv にだけ掛ける。SB3 と CleanRL の PPO 系は直交初期化（gain √2、方策の出力 0.01、価値の出力 1）を使う。CleanRL の PQN は各層の後ろの LayerNorm と gain √2 の直交初期化を組む。OpenAI の phasic-policy-gradient の IMPALA CNN は、出力のユニットごとの L2 ノルムを指定の scale にそろえる。

### 8〜10. 留保・推論・限界

留保: 式は厳密にスケール不変な重みについてのもので、SN でも bias は正規化されない。Adam の η/‖w‖ は ε を無視した近似である。Kosson らの解析は weight decay を前提にしたランダムウォークの近似である。Lyle らの学習率の既定は本文と付録で桁が違う。Gogianu らの Atari の表と Salimans と Kingma の DQN の表は seed の数を確かめられなかった。§9 は報告の推論で、ここには写さない。限界: PDF は文字の抽出で読み、崩れた式は写していない。master・main のコードは 2026-11-27 の取得である。

## 主要な概念

- [可塑性指標と成績](../../../concepts/phenomena/plasticity.md)
- [Atari横断](../../../concepts/subjects/atari.md)
- [NoisyNetの観測](../../../concepts/methods/noisynet.md)
- [315 ニューラルネットワーク](../../../concepts/framework/315_nn.md)
