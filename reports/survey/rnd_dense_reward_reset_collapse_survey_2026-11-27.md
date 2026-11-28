# Survey: RND の効き方と崩れ（密な報酬のゲームでの結果、predictor の初期化し直し、内的報酬の崩れと可塑性）

Date: 2026-11-27
Scope: RND（Random Network Distillation）を外的報酬が密なゲームやパズル型の環境に足した公開の結果と、効かない・害になる条件の報告。予測誤差型の新しさの推定器（RND・ICM など）の predictor を学習の途中で初期化し直す手法と、可塑性を戻すリセットを探索の補助の Network に当てた例。内的報酬が消える・崩れることを、学習の停滞や可塑性の喪失と結び付けた研究。[内的報酬の手法の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md)と[エピソード内の新しさを価値ベースに載せる論点のサーベイ](episodic_bonus_value_based_survey_2026-11-18.md)にすでにある内容（Taïga らの 60 本の集計の結論、Craftax、NGU の「密な報酬のゲームで R2D2 に劣る」、SND・DeRL・Go-Explore の detachment、NGU のエピソード記憶）は調べ直さず、そちらを指す。

## 目次

1. [対象の問いと結論](#1-対象の問いと結論)
2. [Q5: 密な報酬のゲームとパズル型の環境に RND を足した結果](#2-q5-密な報酬のゲームとパズル型の環境に-rnd-を足した結果)
   - [2.1 Taïga らの Rainbow の 60 本の表](#21-taïga-らの-rainbow-の-60-本の表)
   - [2.2 EIPO の論文の PPO と RND の 61 本](#22-eipo-の論文の-ppo-と-rnd-の-61-本)
   - [2.3 NGU の論文の密な報酬のゲーム](#23-ngu-の論文の密な報酬のゲーム)
   - [2.4 FARCuriosity の論文の PPO と RND の表](#24-farcuriosity-の論文の-ppo-と-rnd-の表)
   - [2.5 パズル型と毎手に報酬が付く環境](#25-パズル型と毎手に報酬が付く環境)
   - [2.6 効かない・害になる条件として報告されたこと](#26-効かない害になる条件として報告されたこと)
3. [Q20: predictor の初期化し直しと、可塑性を戻すリセット](#3-q20-predictor-の初期化し直しと可塑性を戻すリセット)
   - [3.1 FARCuriosity: 驚きで新しい predictor を作り、古いものを呼び戻す](#31-farcuriosity-驚きで新しい-predictor-を作り古いものを呼び戻す)
   - [3.2 PreND: predictor の学習を遅くする](#32-prend-predictor-の学習を遅くする)
   - [3.3 PAME: RND の新しさを合図に、Q 側へ可塑性を足す](#33-pame-rnd-の新しさを合図にq-側へ可塑性を足す)
   - [3.4 Qiao ら: 方策ではなく world model をリセットする](#34-qiao-ら-方策ではなく-world-model-をリセットする)
   - [3.5 Juliani と Ash: RND の agent に可塑性の対策を足す](#35-juliani-と-ash-rnd-の-agent-に可塑性の対策を足す)
   - [3.6 探したが見つからなかったもの](#36-探したが見つからなかったもの)
4. [Q22: 内的報酬の崩れと、学習の停滞・可塑性の喪失](#4-q22-内的報酬の崩れと学習の停滞可塑性の喪失)
   - [4.1 Lyle ら: 報酬の無い間に特徴のランクが下がる](#41-lyle-ら-報酬の無い間に特徴のランクが下がる)
   - [4.2 PAME: 可塑性を失うと、探索で見つけたデータから学べない](#42-pame-可塑性を失うと探索で見つけたデータから学べない)
   - [4.3 DrM: 眠ったユニットの割合と、動かなくなること](#43-drm-眠ったユニットの割合と動かなくなること)
   - [4.4 Klein らのサーベイ: 探索の不足を可塑性の喪失の症状に挙げる](#44-klein-らのサーベイ-探索の不足を可塑性の喪失の症状に挙げる)
   - [4.5 内的報酬の大きさが落ちる・戻る報告](#45-内的報酬の大きさが落ちる戻る報告)
   - [4.6 何が崩れると書いているかの整理](#46-何が崩れると書いているかの整理)
5. [批判と懸念](#5-批判と懸念)
6. [総合評価](#6-総合評価)
7. [調査の限界](#7-調査の限界)
8. [出典の一覧](#8-出典の一覧)

## 1. 対象の問いと結論

### 対象の問い

- Q5 の外の部分: RND を、外的報酬が密な Atari のゲーム（Kung-Fu Master など）に足したときの、ゲーム別の結果はあるか。効かない・害になる条件は何と報告されているか。パズル型や毎手に報酬が付く環境に RND を足した報告はあるか。
- Q20 の外の部分: RND（や ICM など予測誤差型の新しさの推定器）の predictor を、学習の途中で周期的に初期化し直す公開の手法・実装はあるか。可塑性の回復のためのリセット（primacy bias、BBF など）を、探索の補助の Network に当てた例はあるか。
- Q22 の外の部分: 「進歩が止まると探索の報酬が崩れうる」という趣旨の指摘をした論文はどれか（内的報酬が消える・崩れることと、学習の停滞や可塑性の喪失を結び付けた研究）。何が崩れると書いているか（内的報酬の大きさ、方策、価値の学習）。条件（学習器、予算）は。

### 結論

- **Q5**: ゲーム別の表は 3 つ見つかった。Kung-Fu Master（KFM）は、Taïga らの Rainbow（200M フレーム）では RND 24087.6 が ε-greedy 28741.8 を下回り、EIPO の論文の PPO（49152000 フレーム）では RND 66954.2 が外的報酬だけの 47266.5 を上回り、NGU の論文では RND の倍率を含む NGU 203.2k が R2D2(Retrace) 220.7k を下回った（§2.1〜2.3）。NGU の論文の密な報酬のゲームでは、R2D2(Retrace) に RND を足すと Q*bert・Beam Rider・Breakout で下がり、Space Invaders で上がった（§2.3）。害の条件として、ボーナスが報酬の地形を歪める（Taïga ら）、探索の易しいゲームで内的報酬に気を取られる（EIPO）、内的報酬がゲームの目的とずれる（NGU）が書かれている（§2.6）。パズル型は Sokoban（MCTS に RND を足して解けた難問が 30 → 52）、毎手に報酬が付く形は NetHack の score の課題（既存の内的報酬はほとんど益が無い）があった（§2.5）。
- **Q20**: RND・ICM の predictor を時刻の周期で初期化し直す公開の手法・実装は見つからなかった。いちばん近いのは FARCuriosity で、予測誤差の急な上昇を合図に新しい RND の module を初期化し、古い module を長期記憶に残して似た観測で呼び戻す（§3.1）。可塑性を戻すリセットを探索の補助の Network に当てた例は見つからなかった。PAME は RND の新しさを合図に Q 側（QMIX の mixer）へ可塑性を足し、RND は初期化しない。Qiao らは方策ではなく world model（予測の Network）をリセットした（§3.3・§3.4）。
- **Q22**: RND の内的報酬の崩れを predictor の可塑性の喪失と結び付けた論文は見つからなかった。近い主張は 3 系統あった。報酬の無い間に価値の Network の特徴のランクが下がる（Lyle ら、価値の学習が崩れる）、可塑性を失うと探索で見つけたデータから方策を学べない（PAME、方策の学習が崩れる）、眠ったユニットの割合が高い時期に agent が動かない（DrM、探索の行動が崩れる）である（§4.1〜4.3）。内的報酬の大きさが崩れる話は、predictor の適応が速すぎて RND の報酬が早く落ちる（PreND）と、忘却で一度下がった報酬が戻る（FARCuriosity）だった（§4.5）。

### 調べた範囲と日付

2026-11-27 に web で調べた。検索語は手法の名前・論文の題・著者名・ゲームの名前だけで組んだ。読んだのは arXiv の要旨の頁、ar5iv と arXiv の HTML、arXiv の PDF を文字に起こしたもの、AAMAS 2025 の会議録の PDF である。数値は、Taïga ら・EIPO・NGU の表は PDF の文字起こしで、FARCuriosity・Sokoban・PAME はそれぞれ ar5iv の HTML と会議録の PDF の画像で確かめた。

### 見つからなかったこと

- RND を足した KFM の結果で、価値ベースの学習器のものは Taïga らの Rainbow の表だけだった。BTR・IQN の上に RND を足した KFM の値は見つからなかった。
- Tetris・2048・落ち物のパズルに RND を足した報告は、検索で見つからなかった。
- RND・ICM の predictor を一定の step ごとに初期化し直す論文・公開の実装は、検索で見つからなかった。公開の実装（OpenAI の RND、RLeXplore、CleanRL）のコードは読んでいない。
- RND の predictor の可塑性（眠ったユニット、ランク）を測った研究は見つからなかった。

## 2. Q5: 密な報酬のゲームとパズル型の環境に RND を足した結果

### 2.1 Taïga らの Rainbow の 60 本の表

Taïga らは Dopamine の Rainbow に RND・ICM・CTS・PixelCNN を足し、ε-greedy・NoisyNets と比べた。学習は sticky actions 0.25、ボーナスのハイパーパラメータは Montezuma's Revenge の 200M フレームの最終成績（5 回の平均）で決め、学習曲線は 5 seed の平均である [Taïga Bonus-ALE (2021/09), §3.1・付録 A]。付録 C は 60 本の生の得点を 10M・50M・100M・200M フレームの 4 時点で表にする（表 2〜5）。表の値が seed の平均かは、表の題に書いていない。

KFM の行は、表 5（200M フレーム）で ε-greedy 28741.8、RND 24087.6、CTS 26439.4、PixelCNN 27234.6、ICM 28168.3、NoisyNets 39382.1 である。ε-greedy と RND は、表 2（10M）で 19824.1 と 17676.9、表 3（50M）で 26929.2 と 22972.3、表 4（100M）で 26459.1 と 23429.2 だった [Taïga Bonus-ALE (2021/09), 付録 C 表 2〜5]。4 時点とも RND は ε-greedy を下回った（表の値の比較で、表に検定は無い）。

同じ表 5 で RND が ε-greedy を大きく下回ったゲームに、ChopperCommand（ε-greedy 12578.4、RND 2674.9）、Seaquest（16044.8、11821.2）、UpNDown（38975.8、22559.4）がある [Taïga Bonus-ALE (2021/09), 付録 C 表 5]。

本文は、ボーナスが報酬の地形を歪めて成績を下げうると書き、数本のゲームで上がる一方で、ほかのゲームでほぼ同じだけ下がり、難探索のゲームを選んで上がる形でもないとする。

```
exploration bonuses can negatively impact performance by skewing the reward landscape

探索のボーナスは、報酬の地形を歪めて成績を下げうる
```

[Taïga Bonus-ALE (2021/09), §3.3]

### 2.2 EIPO の論文の PPO と RND の 61 本

Chen らは、PPO に RND を足して外的と内的の和を最適化する形（RND）と、外的報酬だけの PPO（EO）を Atari 61 本で比べた。多くのゲームは 3000 iteration（49152000 フレーム）、Montezuma's Revenge は 6000 iteration、各 5 seed 以上で、最後の 100 エピソードの外的収益の中央値で比べる [Chen EIPO (2022/11), §4.2・付録 A.4.1・A.4.2]。

- P(RND > EO) は 0.49（95% 信頼区間 0.46〜0.52）で、論文は RND の上乗せが一貫しないと書く [Chen EIPO (2022/11), §4.2・Fig. 3a]。Fig. 5 の対の比較では、RND が EO 以上だったゲームの割合は 51.7%、EO が RND 以上は 48.3% である。
- 付録 A.5 の「RND が PPO より良いゲーム」の一覧は 30 本（この報告で数えた）で、KungFuMaster・Breakout・ChopperCommand・SpaceInvaders を含む。
- 表 7（各手法の平均）の値は、KungFuMaster が EO 47266.5・RND 66954.2、Jamesbond が 13415.9・3971.6、Seaquest が 1453.9・1128.6、Gravitar が 1682.2・1089.8、ChopperCommand が 8225.1・9465.9、MontezumaRevenge が 0.2・2280.0 である [Chen EIPO (2022/11), 付録 A.6 表 7]。
- 内的報酬の重みをゲームごとに振っても、どの重みも RND を一貫して良くしなかった。探索の易しいゲームとされる Jamesbond で、内的報酬を大きくした形のほうが良かった例を挙げ、重みと成績の関係は直観に合わないと書く [Chen EIPO (2022/11), §4.3・Fig. 4]。
- 2 次元の迷路の例（Pycolab）では、上の場所にいる間は毎 step +1 の外的報酬が付き、下の通路にランダムに置いた赤い丸が RND の内的報酬を出し続ける。RND は下の通路に気を取られ、5 seed で EO より悪かった [Chen EIPO (2022/11), §4.1・Fig. 2]。

論文は、探索の易しい課題で内的報酬に気を取られることを問題として掲げる。

```
on easy exploration tasks, the agent gets distracted by intrinsic rewards

探索の易しい課題では、agent が内的報酬に気を取られる
```

[Chen EIPO (2022/11), Abstract]

### 2.3 NGU の論文の密な報酬のゲーム

NGU の論文は、R2D2 を Retrace の損失にした R2D2(Retrace) と、それに RND の内的報酬を足した R2D2+RND を基準に置いた。どちらも 35B フレーム、3 seed の平均である。R2D2+RND の内的報酬の重み β は 0.1・0.3・0.5 を 8 本のゲームで振り、0.3 にした [Badia NGU (2020/02), §4・付録 F]。

密な報酬のゲームの表 2 では、R2D2(Retrace) と R2D2+RND は、Pong 20.9 と 20.7、QBert 415.6k と 353.5k、Breakout 838.3 と 815.8、Space Invaders 35.0k と 54.5k、Beam Rider 111.1k と 85.7k だった [Badia NGU (2020/02), 表 2]。NGU(N=32) から RND の倍率を外す試験（表 3）では、外した形が Pong 19.7（NGU 19.6）、Qbert 550.3k（465.8k）、Breakout 553.7（532.8）、Space Invaders 47.6k（44.6k）、Beam Rider 87.1k（68.7k）で、Montezuma's Revenge は 3.0k（10.4k）だった [Badia NGU (2020/02), 付録 C 表 3]。57 本の表 10 の kung fu master は、R2D2(Retrace) 220.7k±3.9k、NGU(32) の評価 β 0.0 が 212.1k±11.2k、評価 β 0.3 が 203.2k±10.8k である [Badia NGU (2020/02), 付録 G 表 10]。

論文は、密な報酬のゲームで劣る理由を、内的報酬がゲームの目的とずれうることに求める。

```
the intrinsic reward signal may be completely misaligned with the goal of the game

内的報酬の信号は、ゲームの目的と全くずれていることがある
```

[Badia NGU (2020/02), §4.2]

### 2.4 FARCuriosity の論文の PPO と RND の表

Hwang らの表 2 は、PPO・RND（PPO の上）・FARCuriosity を Atari 16 本で並べる。100M フレーム、3 回の試行の平均である [Hwang FARCuriosity (2023/10), §5.1・表 2]。KFM は 16 本に入っていない。

- RND が PPO を下回ったゲーム: Jamesbond（PPO 3060.7、RND 2946.2）、Wizard of Wor（7392.0、6123.0）、Freeway（29.0、21.3）、Private Eye（97.0、47.6）、Tennis（7.8、-5.1）。
- RND が PPO を上回ったゲーム: River Raid（9860.9、11659.3）、Frostbite（1444.7、2282.0）、Space Invaders（1088.7、1352.6）、Breakout（377.3、478.5）、Montezuma Revenge（0.0、210.3）など。

### 2.5 パズル型と毎手に報酬が付く環境

- **Sokoban**: Feng らは AlphaZero に似た MCTS の学習で、新しく展開した節の Q の初期値を 0 ではなく RND の内的報酬にした。225 問の難しい問題で、基準の形は 30 問、RND の好奇心の報酬を足した形は 52 問を解いた [Feng Sokoban-Curriculum (2021/10), §4.1・表 1]。好奇心の報酬は部屋が複数ある問題で特に効き、無いと最初の部屋の箱だけを押し続けると書く。課題の報酬は解けたかどうかの疎なもので、seed の数とばらつきは書いていない。
- **NetHack の score の課題**（ゲームの得点を報酬にする密な課題）: Klissarov らは、外的報酬だけ、外的報酬＋RND、Motif を比べ、score の課題では既存の内的報酬の手法はほとんど益が無いと書く [Klissarov Motif (2023/09), §4.1・Fig. 1]。RND の重みは 0.001・0.01・0.1・0.5・1.0・10.0 を振って 0.5 にした [付録 F]。学習器は CDGPT5（Sample Factory の非同期 PPO）、§4.1 の結果は 10 seed の平均である [§3.1・付録 G]。階段を降りる課題の 3 階・4 階では、外的報酬だけの形も RND の形も成功率 0 だった。
- **毎 step に外的報酬が付く迷路**: EIPO の論文の Pycolab の例（§2.2）がこれに当たる。

### 2.6 効かない・害になる条件として報告されたこと

原典が書いている条件は次のとおりである。

- ボーナスが報酬の地形を歪めて成績を下げうる。上がるゲームとほぼ同じだけ下がるゲームがあり、ハイパーパラメータは Montezuma's Revenge で決めていた [Taïga Bonus-ALE (2021/09), §3.3・§3.4]。
- 探索の易しい課題で内的報酬に気を取られる。内的報酬のほうが外的報酬より得やすい所があると、その所に留まる（迷路の例）。内的報酬の重みとゲームごとの成績の関係は直観に合わない [Chen EIPO (2022/11), Abstract・§4.1・§4.3]。
- 内的報酬の信号がゲームの目的とずれうる [Badia NGU (2020/02), §4.2]。
- 既存の報告にある条件: Craftax の密な報酬と気を散らす可能性、PixelCNN の論文の「探索の易しいゲームで報酬の地形を歪める」は[内的報酬の手法の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md)の §6.4・§9 にある。

## 3. Q20: predictor の初期化し直しと、可塑性を戻すリセット

### 3.1 FARCuriosity: 驚きで新しい predictor を作り、古いものを呼び戻す

Hwang らは、予測誤差型の内的報酬は、predictor が前に訪れた状態を忘れないことに頼っており、忘却があると訪れ直した状態でも内的報酬が上がると論じる [Hwang FARCuriosity (2023/10), §1・§3]。FARCuriosity は RND を基に、今の観測の驚き（予測誤差）とその局所の平均の比が ρ = 10 を超えたとき、今の module とその鍵を長期記憶（最大 800 個、古いものから入れ替え）へ移し、新しい module を初期化する。似た鍵（cosine 類似度 0.75 超）が記憶にあるときは分けない。観測が記憶の鍵と cosine 類似度 0.99 を超えて似ていれば、その module を呼び戻す [§4.2・§4.3・§5.1]。周期ではなく、驚きを合図にした初期化である。

学習器は PPO、Atari 16 本、100M フレーム、3 回の試行で、人間正規化の IQM は PPO 0.69、RND 0.68、FARCuriosity 0.91（ar5iv の表の並びから読んだ）だった。Jamesbond は RND 2946.2 から 13073.0、Frostbite は 2282.0 から 9392.1 に上がり、Montezuma Revenge は 210.3 から 0.0 に下がった。論文は、画面の変化が小さいゲームで誤って分けたためと推測している [§5.2・表 2・付録 C]。発表は NeurIPS 2023 の workshop である。

### 3.2 PreND: predictor の学習を遅くする

Davoodabadi らは、RND の内的報酬は最初の数回の更新ですぐ落ちて報酬の分散が小さくなると書き、その主な原因を predictor が固定の target に速く適応しすぎることに求めた。対策の 1 つに、predictor の学習率だけを 0.01 倍にする形を試した [Davoodabadi PreND (2024/10), §2・§3・Fig. 1]。初期化し直すのではなく、適応を遅くする向きである。試したのは Boxing と Riverraid の 2 本・各 2 seed で、本文に得点の数値は無い。査読先は書かれていない。

### 3.3 PAME: RND の新しさを合図に、Q 側へ可塑性を足す

Zang らは、多 agent の QMIX の上で、RND の予測誤差を内的報酬にし、同時に可塑性を戻す時期の合図にも使う。Replay の中の経験の予測誤差の和が、早い時点の値の κ 倍（κ = 1.2）を超え、かつ module の活性なユニットの割合（FAU）が早い時点の値の μ 倍（μ = 0.9）を下回ったとき、その module に Plasticity Injection（元の重みを凍結し、学習する新しい Network と凍結した同形の Network の差を足す）を当てる [Zang PAME (2025/05), §6.2〜6.4]。当てる先は可塑性の落ちが大きい mixer の Network で、RND の predictor ではない。

SMAC の 5m_vs_6m（2M step）の試験では、mixer だけに当てる形が、何もしない形・local の Q だけの形・全部に当てる形より良く、全部に当てる形が最も悪かった [§7.3・Fig. 10]。全部を初期化すると探索が乱れ、性能を戻すのに多くの更新が要ると書く [§5]。発表は AAMAS 2025 である。

### 3.4 Qiao ら: 方策ではなく world model をリセットする

Qiao らは、モデルベースの RL で agent を周期的にリセットすると MBPO の Hopper で成績が下がったと報告し、代わりに world model（遷移を予測する Network）をリセットした。MBPO では各 ensemble の最後の 2 つの隠れ層を 2×10^4 step ごとに、DreamerV2 では RSSM の遷移の predictor を 2×10^5 step ごとに戻す。DreamerV2 の DeepMind Control 20 本の IQM は 762 から 783 になった [Qiao Model-Primacy (2023/10), §4・表 1]。探索のボーナスは扱っていない。発表は ECAI 2024 である。

### 3.5 Juliani と Ash: RND の agent に可塑性の対策を足す

Juliani と Ash は、Montezuma's Revenge で RND の agent（ハイパーパラメータは RND の原典のまま）に、やわらかい shrink+perturb と regenerative regularization を足した形が、通常の形より高い報酬を得たと書く [Juliani On-Policy-Plasticity (2024/05), §4.7・Fig. 6・付録 B]。対策を RND の predictor にも当てたかは本文に書いていない。得点の数値は本文に無く、seed の数は付録 B（5 回）と Fig. 6 の題（20 回）で食い違う。

### 3.6 探したが見つからなかったもの

- RND・ICM の predictor を一定の step ごとに初期化し直す論文・公開の実装。
- primacy bias・BBF 型のリセットを、内的報酬の Network（predictor・順モデル）に当てた例。
- Klein らの可塑性のサーベイ（2026-04 の版）で読めた範囲は、リセット（hard・soft・部分・continual backprop・Plasticity Injection）を方策と価値の Network に当てる話で、RND・好奇心・探索の補助の Network の語は無かった [Klein Plasticity-Survey (2024/11), §5]。

## 4. Q22: 内的報酬の崩れと、学習の停滞・可塑性の喪失

### 4.1 Lyle ら: 報酬の無い間に特徴のランクが下がる

Lyle らは、DDQN・QR-DQN・補助の乱数の目標を足した DDQN を Montezuma's Revenge（疎な報酬）と Pong（密な報酬）で学ばせ、特徴のランクが一貫して下がるのは疎な報酬の環境だけだったと報告する。Fig. 4b の agent は、この崩れを抜けたあとに初めて課題を解いた。各組 3 seed である [Lyle Capacity-Loss (2022/04), §3.2・Fig. 4]。理論の側では、方策の下で報酬が 0 なら特徴は全状態で 0 のベクトルへ収束するという系を示す [付録 A.2]。一方、ランダムな目標へ当てはめる力は、密な報酬のゲームを含む多くの環境の checkpoint で少し落ちていた [§3.1]。崩れるのは価値の Network の表現で、内的報酬・探索のボーナスは扱っていない。対策の InFeR（補助の Head の出力を初期の値へ引き戻す正則化）で、Rainbow の Montezuma's Revenge が大きく上がったと書く（値は図だけにある）[Lyle Capacity-Loss (2022/04), Fig. 5c]。発表は ICLR 2022 である。

### 4.2 PAME: 可塑性を失うと、探索で見つけたデータから学べない

Zang らは、多 agent の疎な報酬の課題で、探索の手法は成功する経験を見つけるのに、学習で可塑性を失った Network はそれを方策に変えられないと主張する。2 体で重い箱を押す課題（60000 step）では、RND は成功したエピソードを 201 本見つけたが、学習後の勝率は 4.8% だった。成功した軌跡を集めたデータで学ばせると、ランダムに初期化した Network はすぐに解き、長く学んだ（活性なユニットが減った）Network は勝率が 0 近くのままだった [Zang PAME (2025/05), §4・表 1・Fig. 2・Fig. 3]。崩れるのは方策の学習で、内的報酬の大きさではない。

```
neural networks struggle to learn effective exploration strategies

ニューラルネットワークは有効な探索の戦略を学びにくい
```

[Zang PAME (2025/05), Abstract]

### 4.3 DrM: 眠ったユニットの割合と、動かなくなること

Xu らは、画像入力の連続制御で、学習の早い時期に agent が動かなくなり、その時期に方策の Network の眠ったユニットの割合が高いと報告する。割合が下がると、意味のある動きが始まる。眠ったユニットの割合で探索のノイズの大きさを決め、約 2×10^5 フレームごとのやわらかいリセットの強さも決める [Xu DrM (2023/10), §1・§3.1・§3.2]。基盤は DrQ-v2、DeepMind Control は 30M フレーム、4 seed である。付録 E で DrQ-v2 に RND を足した形と比べ、RND は DrQ-v2 を上回ったが DrM には届かなかった。崩れるのは探索の行動（動きの量）で、内的報酬ではない。発表は ICLR 2024 である。

### 4.4 Klein らのサーベイ: 探索の不足を可塑性の喪失の症状に挙げる

Klein らは要旨で、可塑性の喪失が性能の頭打ちや過大評価とともに探索の不足に寄与すると書く。§5.1 ではリセットに可塑性と別の探索の利点がありうるとし、§6.3 では探索の一部が勾配の学習による頻繁な行動の変化から来うると書く [Klein Plasticity-Survey (2024/11), Abstract・§5.1・§6.3]。ここでの探索は方策の行動の変化で、内的報酬は扱っていない。

### 4.5 内的報酬の大きさが落ちる・戻る報告

- **PreND**: RND の内的報酬は最初の数回の更新ですぐ落ち、報酬の分散が小さくなって探索の信号が弱まると書く。原因は predictor の速すぎる適応とする（§3.2）[Davoodabadi PreND (2024/10), §2]。
- **FARCuriosity**: Jamesbond の RND の内的報酬は、学習の中ほどで下がったあと、100M フレームの終わりに再び上がった。論文はこれを predictor の忘却によるとする [Hwang FARCuriosity (2023/10), §3.5・Fig. 4]。崩れる向きとは逆である。
- **学習の進み（learning progress）を報酬にする形**: Hou らの LPM は、モデルが新しく何も学ばないときに限り内的報酬が 0 になる形で、これを望ましい性質として扱う。Montezuma's Revenge では探索が飽和すると報酬が下がり、新しい区域に入ると跳ねた [Hou LPM (2025/09), §4.1・Fig. 5]。可塑性・学習の停滞の危うさは論じていない。
- **報酬の非定常**: Castanyer らは、回数・擬似回数・状態エントロピーのボーナスは非定常で agent が最適化しにくいとし、十分統計量を状態に足して定常にする（要旨だけ読んだ）[Castanyer SOFE (2023/10), Abstract]。

### 4.6 何が崩れると書いているかの整理

| 原典 | 崩れるもの | 結び付ける原因 | 学習器・予算 | 内的報酬を扱うか |
|---|---|---|---|---|
| Lyle ら | 価値の Network の特徴のランク | 報酬が 0 の間の学習 | DDQN・QR-DQN、Montezuma's Revenge と Pong、3 seed | 扱わない |
| PAME | 探索で見つけたデータからの方策の学習 | 可塑性の喪失（活性なユニットの減少） | QMIX、SMAC（2M step）・GRF | RND を内的報酬と合図に使う |
| DrM | 探索の行動（動かない） | 眠ったユニットの割合 | DrQ-v2、DeepMind Control 30M フレーム、4 seed | 比較に RND |
| Klein らのサーベイ | 探索の量 | 可塑性の喪失 | （サーベイ） | 扱わない |
| PreND | 内的報酬の大きさと分散 | predictor の速すぎる適応 | 学習器の記載なし、Boxing・Riverraid、2 seed | 扱う（RND） |
| FARCuriosity | 内的報酬が下がり切らない（戻る） | predictor の忘却 | PPO、Atari 16 本、100M フレーム、3 回 | 扱う（RND） |
| LPM | 内的報酬が 0 になる（望ましい性質） | モデルが新しく学ばない | A2C・PPO ほか | 扱う（学習の進み） |

## 5. 批判と懸念

- **比べる条件の違い**: KFM の 3 つの値は、学習器（Rainbow・PPO・R2D2）、予算（200M・49152000・35B フレーム）、統計量（表の得点・最後の 100 エピソードの中央値・評価の平均）、RND の重みが違う。ゲームをまたぐ RND の効き方をこの 3 つから決めることはできない。
- **査読の状況**: FARCuriosity は workshop、PreND・Juliani と Ash・Klein らのサーベイ・LPM は arXiv のままで、査読先を確かめられなかった。PreND は 2 本・2 seed で得点の数値が無い。
- **領域の違い**: PAME は多 agent（QMIX）、DrM は連続制御、Qiao らはモデルベースで、どれも単一 agent の DQN 系に RND を足した設定ではない。
- **見出しの割り当て**: Q20 で「近い例」とした 3 つ（FARCuriosity・PAME・Qiao ら）は、どれも「RND の predictor を周期的に初期化する」ことそのものではない。

## 6. 総合評価

- Q5 は、ゲーム別の公開の表が Rainbow・PPO・R2D2 の 3 つの学習器で揃い、KFM の値も 3 つ取れた。密な報酬のゲームで RND が下げる例と上げる例の両方があり、害の条件は「報酬の地形の歪み」「気を取られる」「目的とのずれ」として書かれている。パズル型は Sokoban の 1 本、得点の密な課題は NetHack の 1 本だけだった。
- この報告での読み取り（推論）: KFM で RND が対照を下回るか上回るかは、学習器と予算で割れている。Rainbow（200M フレーム）と NGU の比較では下回り、PPO（49152000 フレーム）では上回った。3 つの論文は RND の重み・正規化・評価の方法が違うので、この差を学習器だけに帰すことはできない。
- Q20 は、「周期的に predictor を初期化し直す」ことの公開の前例は見つからず、驚きを合図にした初期化（FARCuriosity）と、RND を合図にした Q 側の可塑性の回復（PAME）が近い。
- Q22 は、「内的報酬の崩れ」と「可塑性の喪失」を同じ Network の上で結び付けた研究は見つからなかった。可塑性の喪失と探索を結ぶ研究は、価値の表現（Lyle ら）、方策の学習（PAME）、行動の量（DrM）のどれかが崩れると書き、内的報酬の大きさを崩れるものとして扱うのは PreND だけだった。

## 7. 調査の限界

- **被引用数**: Semantic Scholar の API が 429 を返し、被引用数は取れていない。Google Scholar は使っていない。
- **抜き出しの方法**: ar5iv の HTML と PDF の文字起こしから要約ツールで抜き出した。Taïga らの表 5 と NGU の表 3・表 10 は、PDF の文字起こしで列がずれていたので、ar5iv の行と照らして並びを合わせた。FARCuriosity の IQM の行は列見出しとの対応が表に明示されていない。
- **読めなかったもの**: FARCuriosity の PDF は大きさの上限で取れず、ar5iv だけで読んだ。Motif の RL の step 数と、PAME・Qiao らの seed の数は本文から確かめていない。Juliani と Ash の Fig. 6 の値は図の中だけにある。
- **公開の実装**: RND の公式の実装、RLeXplore、CleanRL、EIPO のコードは読んでいない。predictor の初期化し直しが実装の選択肢にあるかは確かめていない。
- **版**: Taïga らは arXiv の拡張版（ICLR 2020 の体裁）、Klein らは v3（2026-04）、LPM は v2（2026-04）を読んだ。Klein らの HTML は末尾の約 5,000 字（文献表の終わり）を読んでいない。

## 8. 出典の一覧

[Taïga Bonus-ALE, 2021/09] Adrien Ali Taïga, William Fedus, Marlos C. Machado, Aaron Courville, Marc G. Bellemare（MILA・Université de Montréal、Google Brain）. "On Bonus-Based Exploration Methods in the Arcade Learning Environment." ICLR 2020（読んだのは arXiv の版の PDF と ar5iv の HTML）. https://arxiv.org/pdf/2109.11052

[Chen EIPO, 2022/11] Eric Chen, Zhang-Wei Hong, Joni Pajarinen, Pulkit Agrawal（MIT Improbable AI Lab、MIT-IBM Watson AI Lab、Aalto University ほか）. "Redeeming Intrinsic Rewards via Constrained Optimization." NeurIPS 2022（読んだのは arXiv の PDF と ar5iv の HTML）. https://arxiv.org/pdf/2211.07627

[Badia NGU, 2020/02] Adrià Puigdomènech Badia ほか（DeepMind）. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020（読んだのは arXiv の PDF と ar5iv の HTML）. https://arxiv.org/pdf/2002.06038

[Hwang FARCuriosity, 2023/10] Jaedong Hwang, Zhang-Wei Hong, Eric Chen, Akhilan Boopathy, Pulkit Agrawal, Ila Fiete（MIT）. "Neuro-Inspired Fragmentation and Recall to Overcome Catastrophic Forgetting in Curiosity." NeurIPS 2023 Workshop on Intrinsically Motivated Open-ended Learning. https://arxiv.org/abs/2310.17537

[Feng Sokoban-Curriculum, 2021/10] Dieqiao Feng, Carla P. Gomes, Bart Selman（Cornell University）. "A Novel Automated Curriculum Strategy to Solve Hard Sokoban Planning Instances." arXiv（会議名は頁に無い）. https://arxiv.org/abs/2110.00898

[Klissarov Motif, 2023/09] Martin Klissarov, Pierluca D'Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal Vincent, Amy Zhang, Mikael Henaff. "Motif: Intrinsic Motivation from Artificial Intelligence Feedback." arXiv（会議名は頁に無い）. https://arxiv.org/abs/2310.00166

[Davoodabadi PreND, 2024/10] Mohammadamin Davoodabadi, Negin Hashemi Dijujin, Mahdieh Soleymani Baghshah（Sharif University of Technology）. "PreND: Enhancing Intrinsic Motivation in Reinforcement Learning through Pre-trained Network Distillation." arXiv（v1 の HTML を読んだ）. https://arxiv.org/html/2410.01745v1

[Zang PAME, 2025/05] Zehua Zang, Chuxiong Sun, Lixiang Liu, Fuchun Sun, Changwen Zheng（Institute of Software, Chinese Academy of Sciences、Tsinghua University ほか）. "Loss of Plasticity: A New Perspective on Solving Multi-Agent Exploration for Sparse Reward Tasks." AAMAS 2025, pp. 2299–2308. https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2299.pdf

[Qiao Model-Primacy, 2023/10] Zhongjian Qiao, Jiafei Lyu, Xiu Li（Tsinghua Shenzhen International Graduate School）. "Mind the Model, Not the Agent: The Primacy Bias in Model-based RL." ECAI 2024. https://arxiv.org/abs/2310.15017

[Juliani On-Policy-Plasticity, 2024/05] Arthur Juliani, Jordan T. Ash（Microsoft Research NYC）. "A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning." arXiv（v2 は 2024-11）. https://arxiv.org/abs/2405.19153

[Lyle Capacity-Loss, 2022/04] Clare Lyle, Mark Rowland, Will Dabney（University of Oxford、DeepMind）. "Understanding and Preventing Capacity Loss in Reinforcement Learning." ICLR 2022. https://arxiv.org/abs/2204.09560

[Xu DrM, 2023/10] Guowei Xu ほか（Tsinghua University、University of Maryland ほか）. "DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization." ICLR 2024. https://arxiv.org/abs/2310.19668

[Klein Plasticity-Survey, 2024/11] Timo Klein, Christoph Luther, Manus McAuliffe, Lukas Miklautz, Claudia Plant, Sebastian Tschiatschek. "Plasticity Loss in Deep Reinforcement Learning: A Survey." arXiv（v3 は 2026-04）. https://arxiv.org/abs/2411.04832

[Hou LPM, 2025/09] Zhibo Hou, Zhiyu An, Wan Du（University of California, Merced）. "Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring." arXiv（v2 は 2026-04。v2 の HTML を読んだ）. https://arxiv.org/html/2509.25438v2

[Castanyer SOFE, 2023/10] Roger Creus Castanyer, Joshua Romoff, Glen Berseth. "Improving Intrinsic Exploration by Creating Stationary Objectives." ICLR 2024. https://arxiv.org/abs/2310.18144
