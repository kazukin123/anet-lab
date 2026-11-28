# 作業ノート: 自己模倣の記憶の持ち方の事例

自己模倣（SIL とその後続）で、過去の良い軌跡をどう選んで持ち、どう更新に混ぜるかの外の事例。RFD0201（上位エピソードの記憶と SIL の下限損失）の「上位 16 本を持つ」形の先行例と、PPO 系に SIL を載せるときの原典の組み方を確かめるために調べた。

## 1. GASIL は割引収益で上位 K 本の軌跡を持つ

- 確かめた日: 2026-11-28
- 確かめた手段: web 検索で論文を特定し、ar5iv の本文を web 取得で読んだ。
- 分かったこと:
  - 良い軌跡の記憶に、割引収益で上位 K 本のエピソードを持つ（§4.1）。入れ替えの規則は「上位 K 本を持つ」以上には書いていない。
  - 記憶の大きさは step 数で表し、1,000 と 10,000 から選ぶ（付録 A の Table 1）。Walker2d-v2 で 500 と 5,000 を比べ、大きくすると記憶の軌跡の収益の平均が下がると述べる（§5.5、図 7a）。
  - 記憶は GAIL 型の識別器の学習に使い、方策は識別器から作った報酬と環境の報酬を混ぜて PPO で学ぶ（§4.3、§5.1）。
  - 実験は 2D Point Mass、MuJoCo、報酬を 20 step ごとにまとめて渡す MuJoCo、観測に雑音を入れた Walker2d-v2。Atari の実験は無い（§5.2〜§5.4）。
  - 関連研究の節で、自己模倣の多くは方策の改善を保証するのに決定的な環境を要し、確率性の扱いは今後の課題とする（§2）。
  - SIL を比較の基準に使い（PPO+SIL）、GASIL のほうが良かったとする（§5.3）。
- 出典: Guo, Oh, Singh, Lee. "Generative Adversarial Self-Imitation Learning." arXiv 1812.00950. https://arxiv.org/abs/1812.00950

## 2. SIDP は割り引かない和の優先度つきキューで軌跡を持つ

- 確かめた日: 2026-11-28
- 確かめた手段: web 検索で論文と著者のスライドを特定し、ar5iv の本文を web 取得で読んだ。本文の後半（§5 以降と付録）は取得の道具が返した要約しか読めていない。
- 分かったこと:
  - 良い軌跡の記憶を、軌跡の報酬の合計（割り引かない）で並べた優先度つきキューで持つ。上位 k 本か、収益がしきい値を超えた軌跡を残す（§2.2）。新しい軌跡はキューのしきい値と比べて入れる（§2.2）。著者のスライドは、今の最小の収益を下回る軌跡は入れないと述べる。
  - 主な実験の大きさは 10 本（§3.1）。後半の要約によれば、1・5・10・25・50 本を比べている（§5.4）。
  - 記憶は、今の方策と記憶の軌跡の分布の Jensen-Shannon ダイバージェンスを小さくする自己模倣の項に使い、方策は PPO で学ぶ。複数の方策を Stein variational policy gradient で同時に学ぶ（§2.2、§2.3、§3）。
  - 実験は MuJoCo の移動課題（報酬をエピソードの終わりにまとめて渡す版など）と 2 次元の迷路。Atari は序論で動機として触れるだけで、実験は無い。
  - 自己模倣の限界として、環境が確率的だと、過去の上位 k 本を真似るだけでは最適方策に戻りにくいと、2 本腕のベルヌーイ・バンディットの例で述べる（§2.2）。
  - SIL を比較の基準に使う（§3.1 の表）。
- 出典: Gangwani, Liu, Peng. "Learning Self-Imitating Diverse Policies." ICLR 2019. https://arxiv.org/abs/1805.10309 。スライド: https://tgangwani.github.io/data/SID_slides.pdf

## 3. 上位の何本だけを持つ形は、方策ベースの自己模倣にある

- 確かめた日: 2026-11-28
- 確かめた手段: 項目 1・2 と、[SIL を価値ベースへ広げた手法のサーベイ](../survey/sil_value_based_lower_bound_survey_2026-11-24.md) の §4・§9 を突き合わせた。
- 分かったこと:
  - 価値ベースへ広げた手法（Optimality Tightening、EMDQN、Tang の下限 Q 学習、SAIL、GEM、DB-ADP）には、上位の何本だけを残す方式が見つからなかった（サーベイ §9）。
  - 方策ベースの自己模倣では、GASIL（割引収益で上位 K 本）と SIDP（割り引かない和で上位 10 本、最小より良いものを入れる）が、上位の軌跡だけを持つ記憶を使う。
  - RFD0201 の暫定の既定（外的報酬の和で上位 16 本、満ちたら最小を外す）は、物差しと本数の単位が SIDP に近い。上位の記憶と価値の下限の損失を組み合わせた事例は、調べた範囲では見つかっていない。
- 出典: 項目 1・2、[サーベイ](../survey/sil_value_based_lower_bound_survey_2026-11-24.md) §9。

## 4. 原典の SIL は PPO と組むとき、PPO の更新ごとに SIL の更新を 10 回足す

- 確かめた日: 2026-11-28
- 確かめた手段: ar5iv の論文の本文と付録、著者の公式実装の README を web 取得で読んだ。
- 分かったこと:
  - PPO+SIL の実験は MuJoCo の 6 課題、10 seed（§5.6、図 5）。Swimmer・Walker2d・Ant で PPO より良くなった。数は図だけで、本文に無い。
  - 報酬を 20 step ごとか終端でまとめて渡す版（遅延報酬）では、PPO も PPO+SIL も下がるが、PPO+SIL の PPO に対する差は大きくなった（§5.6、図 5 の下段）。
  - 設定（付録の Table 4）: PPO の 1 回の更新（batch）ごとに SIL の更新 10 回、SIL のバッチ 512、記憶の大きさ 50,000（単位は書いていない）、優先度の指数は {0.6, 1.0} から選ぶ、優先度の偏りの補正 0.1、SIL の損失の重み 0.1、SIL の価値の項の重み β は {0.01, 0.05} から選ぶ、エントロピーの重み 0。PPO 側は horizon 2048、10 epoch、ミニバッチ 64、γ 0.99、GAE λ 0.95。
  - SIL の損失（式 1・2）には新旧の方策の比が無い。SIL の更新を PPO の更新のあとに回すのかなど、順序は PPO 版では書いていない。A2C 版は、A2C の更新のあとに SIL の更新を M 回続ける（Algorithm 1）。
  - 組める理由として、SIL の目的は重要度の重みを持たず、学習者より良いふるまいならどれからでも学べる、と述べる（§5.5）。PPO は SIL の目的と理論の上で強いつながりは無いが、どちらも方策と価値を最適へ寄せるので補い合う、と述べ、実験で支える（§5.6）。
  - Atari の A2C+SIL で、序盤に活用しすぎて準最適に留まったゲーム（James Bond・Star Gunner）は、後半に SIL の更新の回数か重みを減らすと解消した。減らし方の予定は書いていない。自動で調整するのは今後の課題とする（§5.4）。
- 出典: Oh, Guo, Singh, Lee. "Self-Imitation Learning." ICML 2018. https://arxiv.org/abs/1806.05635 。公式実装: https://github.com/junhyukoh/self-imitation-learning

## 5. 2022 年以後の SIL の後続には、手続き生成の環境と PPO の上の手法がある

- 確かめた日: 2026-11-28
- 確かめた手段: web 検索で見つけ、arXiv の要旨のページを web 取得で読んだ。本文は読んでいない。
- 分かったこと:
  - Andres・Villar-Rodriguez・Del Ser（2022/11）: 自己模倣に内的動機づけ（好奇心型の探索）を組み合わせ、手続き生成の環境で、以前の自己模倣の手法より良く、内的動機づけだけと比べて標本効率が同等以上だったと主張する。学習器と環境の名は要旨に無い。
  - Andres・Zha・Del Ser（2023/11）: 自己模倣で再生する遷移の優先度の付け方と、優先度と汎化の要求で失われる多様性を補う変更を提案する。手続き生成のまれな報酬の環境 3 つ（MiniGrid と ProcGen を含む）で試し、MiniGrid-MultiRoom-N12-S10 で最良と主張する。
  - Chaudhary・Behera・Mondal（2026/03）: PPO の上の on-policy の自己模倣で、過去の報酬の高い状態と行動で方策の更新を導く。密な報酬では、最も報酬の高い軌跡の状態の訪れ方の分布へ最適輸送の距離で寄せ、まれな報酬では、成功した自分の軌跡を一様に再生する。MuJoCo、Animal-AI Olympics、PointMaze で試し、自己模倣の比較手法より速く収束したと主張する。要旨は Oh らの SIL を名指ししていない。
- 出典: Andres ほか "Towards Improving Exploration in Self-Imitation Learning using Intrinsic Motivation." arXiv 2211.16838. https://arxiv.org/abs/2211.16838 。Andres ほか "Enhanced Generalization through Prioritization and Diversity in Self-Imitation Reinforcement Learning over Procedural Environments with Sparse Rewards." arXiv 2311.00426. https://arxiv.org/abs/2311.00426 。Chaudhary ほか "Match or Replay: Self Imitating Proximal Policy Optimization." arXiv 2603.27515. https://arxiv.org/abs/2603.27515
