---
title: 勾配ノイズの観測と対処に関するサーベイ
page_kind: サマリ
material: "[[reports/gradient_noise_observation_and_control_survey_2026-10-15.md|gradient_noise_observation_and_control_survey_2026-10-15]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/phenomena/gradient_noise_batch.md|勾配ノイズとbatch]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# 勾配ノイズの観測と対処に関するサーベイ

## 元の材料

[勾配ノイズの観測と対処に関するサーベイ](../../../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md) blob:0d66146fbcd7feaf53873be1b204f5cb3bef58ac

## 要点

- Bを変える方向は妥当で、有力な比較対象である。ただし、GNSが大きいからBを増やせば成績が上がる、という決定則にはできない。
- r_Bは「勾配推定のノイズ二乗／平均勾配の二乗」。Bを倍にするとノイズの分散は半分、標準偏差は約0.707倍になる。Gは定義した学習目的の平均勾配であり、評価性能へ必ず向かう方向という意味ではない。arXiv McCandlish (2018/12), §2.2。
- 小batchの単位を小さくすると推定精度が改善する。検証したTransformerではLayerNormのGNSが全体GNSをよく予測し、専用kernelによる測定とbatch scheduleで学習時間を18%短縮した。DDPや勾配累積の単位で測る方法もあるが、推定精度が実行構成に依存する。NeurIPS Gray (2024/11), §2・Appendix A。
- Transformerでfull-batch化し、dropoutを無効化しても、SGDとAdamの差が消えない反証実験がある。ICLR Kunstner (2023/04), §3・Appendix B.4。
- 本調査の総合評価: PRD092の計測を進める方向は支持する。B増加は一般事例でも強いbaselineで、学習率低下との比較を優先する価値がある。ただし採用判断は「noiseを減らせたか」と「目的の成績・予算が改善したか」の両方で行う。GNSは原因を絞る観測量であり、成績改善や最適Bの決定値ではない。

## 内容

### 1. 調査の結論

この表は文献を横断した本調査の分類であり、単一論文の提案ではない。根拠は§3–6に分けて示す。

### 2. 何をノイズと呼び、何を観測するか

本調査の数学的説明: ηを半分にすると信号の更新幅も半分になるため、GNSやr_B自体は変わらない。Bを倍にする場合は、期待勾配を保って勾配推定のばらつきを下げられる。学習率低下とB増加は、同じ操作ではない。長時間のSGDを連続時間近似したノイズ強度にη/Bが現れることと、一回の更新分散にη²/Bが現れることも区別する。SGDの定義と近似の出典はICLR Smith (2017/11), §2。

また、連続したlearn stepの勾配はθも目的も変わる。その変動をそのまま「固定状態でのsample分散」と呼ぶことはできない。ノイズの測定と、学習中のドリフトの記録を分けるべき、というのが固定状態という定義からの帰結である。

### 3. 観測方法の一般事例

単純GNSがこれに一致するのはHが単位行列に比例する場合。局所改善を予測するモデルであり、最終成績の保証ではない。arXiv McCandlish (2018/12), §2.2–2.4。

Flipoutは、batch内で同じ重み摂動を共有するとsample間の勾配が相関し、Bを増やしても分散が頭打ちになり得ることを示した。FC・CNN・LSTMで検証している。ICLR Wen (2018/03), Theorem 2・§4.1。

以下は本調査による実験設計の提案。原典の計測法を、診断する目的ごとに並べたもの。

### 4. Bと学習率を変える一般事例

学習率を減衰させる代わりにBを増やし、SGD、momentum、Nesterov、Adamで類似した学習曲線を得る実験がある。SGDの近似では、ノイズ強度に学習率、B、momentumが共同で関与する。ただし、Bを際限なく増やす処方ではない。ICLR Smith (2017/11), §2–4。

同論文の学習率を含む動力学のnoise scaleと、McCandlishのGNSは別の量である。B固定でηを下げる対照は、B増加を解釈するために重要、というのが本調査の判断。

### 5. B以外の対処と限界

BERTとImageNetの勾配ノイズ分布を比較した研究は、heavy-tailed noiseへの対処としてclippingを解析し、座標別のadaptive clippingをBERTで評価した。NeurIPS Zhang (2019/12), §2–5。

clipは大きな更新を抑えるが、真の大きな信号も削り得る。平均勾配をより正確に推定するB増加と、外れ値に強くするclipは別の対処。heavy-tailが問題かどうかは観測する必要がある。

### 6. 批判と懸念

SGDのノイズには方向の偏りがあり、同じ大きさの等方的ノイズと同じ汎化効果にはならない。ノイズ共分散と曲率の関係を調べた研究がある。ICML Zhu (2019/06), §3–4。

本調査では、この教師あり学習中心の指針をRLへ無条件に適用しない。固定環境経験予算での性能を目的にする場合は、Bが更新数やreplayとの関係も変えるためである。

### 7. PRD092への考察と検証案

ただし、simple GNSが256を超えたことだけで「B512の成績改善はノイズが原因」と言い切れない。逆に256を大きく下回っても、AdamでB増加が効く可能性を棄却できない。§4.6のOLMoは、まさにsimple GNSと実際の有効Bが一致しない事例である。GNSを機序の観測、分岐Runを介入効果の検証として使うのが、本調査の推奨。

「真の勾配」も、現時点のsampling・weighting・target・乱数で定義した目的の期待勾配として説明した方が正確。その目的の停留点にいることと、環境の成績が最適であることは別である。

### 8. 総合評価と調査の限界

推奨する順序は、固定状態での分散較正、共有ノイズとsample noiseの切り分け、Bと学習率の介入比較、その後に自律調整の較正である。曲率の高価な測定やoptimizer変更を、最初から必須にする提案ではない。

2026-10-15までの資料を検索した。Naganumaの2026年研究は確認したv1プレプリントとして扱い、査読採択を確認したとは主張しない。

## 主要な概念

- [勾配ノイズとbatch](../../concepts/phenomena/gradient_noise_batch.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
