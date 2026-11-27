# Survey: RND の効き方と崩れ（密な報酬のゲームでの結果、predictor の初期化し直し、内的報酬の崩れと可塑性）

## 元の材料

[Survey: RND の効き方と崩れ（密な報酬のゲームでの結果、predictor の初期化し直し、内的報酬の崩れと可塑性）](../../../../reports/rnd_dense_reward_reset_collapse_survey_2026-11-27.md) blob:cf6023701bdbdfa958f8da9d6aaa8c06d869ebc8

## 要点

- KFM の行は、Taïga らの Rainbow（sticky actions 0.25）の 200M フレームで ε-greedy 28741.8、RND 24087.6、NoisyNets 39382.1 で、10M・50M・100M でも RND が ε-greedy を下回った [Taïga Bonus-ALE (2021/09), 付録 C 表 2〜5]。
- EIPO の論文の PPO（49152000 フレーム、5 seed 以上）では、KungFuMaster が外的報酬だけ 47266.5、RND 66954.2 で、61 本全体の P(RND > EO) は 0.49（95% 信頼区間 0.46〜0.52）だった [Chen EIPO (2022/11), §4.2・付録 A.6 表 7]。
- NGU の論文（35B フレーム、3 seed）では、R2D2(Retrace) に RND（β 0.3）を足すと QBert 415.6k → 353.5k、Beam Rider 111.1k → 85.7k、Space Invaders 35.0k → 54.5k だった。kung fu master は R2D2(Retrace) 220.7k±3.9k、NGU（評価 β 0.3）203.2k±10.8k [Badia NGU (2020/02), 表 2・表 10]。
- 害の条件として、ボーナスが報酬の地形を歪める [Taïga Bonus-ALE (2021/09), §3.3]、探索の易しい課題で内的報酬に気を取られる [Chen EIPO (2022/11), Abstract]、内的報酬がゲームの目的とずれる [Badia NGU (2020/02), §4.2] が書かれている。
- パズル型では、Sokoban の MCTS に RND を足すと 225 問の難問で解けた数が 30 から 52 になった [Feng Sokoban-Curriculum (2021/10), §4.1・表 1]。NetHack の score の課題では、既存の内的報酬の手法はほとんど益が無いと書かれている [Klissarov Motif (2023/09), §4.1]。
- RND・ICM の predictor を一定の step ごとに初期化し直す公開の手法・実装は見つからなかった。近いのは、驚きの比が ρ = 10 を超えたときに新しい RND の module を初期化し、古い module を長期記憶から呼び戻す FARCuriosity（PPO、Atari 16 本、100M フレーム、IQM は RND 0.68 に対し 0.91）である [Hwang FARCuriosity (2023/10), §4.2・表 2]。
- 可塑性を戻すリセットを探索の補助の Network に当てた例は見つからなかった。PAME は RND の新しさを合図に QMIX の mixer へ Plasticity Injection を当て、RND は初期化しない [Zang PAME (2025/05), §6.3・§6.4]。Qiao らは world model をリセットした [Qiao Model-Primacy (2023/10), §4]。
- RND の内的報酬の崩れを predictor の可塑性の喪失と結び付けた論文は見つからなかった。近い主張は、報酬の無い間の特徴のランクの低下（Lyle ら）、可塑性を失った Network が探索で見つけたデータから学べないこと（PAME）、眠ったユニットの割合が高い時期に agent が動かないこと（DrM）である [Lyle Capacity-Loss (2022/04), §3.2；Zang PAME (2025/05), §4；Xu DrM (2023/10), §3.1]。

## 内容

### 1. 対象の問いと結論

外的報酬が密なゲームとパズル型の環境での RND の結果（Q5）、predictor の周期的な初期化し直しと可塑性のリセットの例（Q20）、内的報酬の崩れと学習の停滞・可塑性の喪失を結び付けた研究（Q22）を、2026-11-27 に web で調べた。BTR・IQN の上に RND を足した KFM の値、Tetris・2048・落ち物のパズルに RND を足した報告、RND の predictor の可塑性を測った研究は見つからなかった。

### 2. Q5: 密な報酬のゲームとパズル型の環境に RND を足した結果

Taïga らの表 5 では、ChopperCommand が ε-greedy 12578.4・RND 2674.9、Seaquest が 16044.8・11821.2 だった。ハイパーパラメータは Montezuma's Revenge で決め、上がるゲームとほぼ同じだけ下がるゲームがあると書く [Taïga Bonus-ALE (2021/09), §3.3・表 5]。

EIPO の論文の Fig. 5 では、RND が EO 以上だったゲームは 51.7%、付録 A.5 の「RND が PPO より良いゲーム」は 30 本だった。Jamesbond は EO 13415.9・RND 3971.6。内的報酬の重みをゲームごとに振っても一貫して良くする値は無く、毎 step に外的報酬が付く迷路では RND が下の通路に気を取られた [Chen EIPO (2022/11), §4.1・§4.3・表 7]。

NGU(N=32) から RND の倍率を外すと、Qbert 465.8k → 550.3k、Beam Rider 68.7k → 87.1k、Montezuma's Revenge 10.4k → 3.0k だった [Badia NGU (2020/02), 表 3]。FARCuriosity の表の PPO と RND（100M フレーム、3 回）では、Jamesbond・Wizard of Wor・Freeway・Private Eye・Tennis で RND が下回った [Hwang FARCuriosity (2023/10), 表 2]。NetHack の比較の RND の重みは 0.5、学習器は CDGPT5、10 seed である [Klissarov Motif (2023/09), 付録 F・G]。

### 3. Q20: predictor の初期化し直しと、可塑性を戻すリセット

FARCuriosity は周期ではなく驚きを合図に初期化し、Montezuma Revenge は 210.3 から 0.0 に下がった。PreND は初期化し直すのではなく predictor の学習率を 0.01 倍にした（2 本・各 2 seed、得点の数値なし）[Davoodabadi PreND (2024/10), §3]。PAME は、mixer だけに当てる形が良く、全部に当てる形が最も悪かったと書く [Zang PAME (2025/05), §7.3]。Qiao らの DreamerV2 の DeepMind Control 20 本の IQM は 762 から 783 になった [Qiao Model-Primacy (2023/10), 表 1]。Juliani と Ash は、Montezuma's Revenge の RND の agent に shrink+perturb と regenerative regularization を足すと報酬が上がったと書くが、predictor にも当てたかは書かない [Juliani On-Policy-Plasticity (2024/05), §4.7]。Klein らの可塑性のサーベイの読めた範囲に、RND・好奇心の Network のリセットは無かった。

### 4. Q22: 内的報酬の崩れと、学習の停滞・可塑性の喪失

Lyle らは、報酬が 0 なら特徴は 0 のベクトルへ収束するという系を示し、ランダムな目標への当てはめの力は密な報酬のゲームでも少し落ちたと書く [Lyle Capacity-Loss (2022/04), §3.1・付録 A.2]。PAME の箱を押す課題では、RND は成功したエピソードを 201 本見つけたが勝率は 4.8% だった [Zang PAME (2025/05), 表 1]。Klein らは、可塑性の喪失が探索の不足に寄与すると要旨に書く [Klein Plasticity-Survey (2024/11), Abstract]。内的報酬の大きさについては、PreND が RND の報酬はすぐ落ちて分散が小さくなると書き、FARCuriosity は Jamesbond の RND の内的報酬が 100M フレームの終わりに再び上がったと報告する。LPM は、モデルが新しく学ばないときに内的報酬が 0 になる形を望ましい性質として扱う [Hou LPM (2025/09), §4.1]。SOFE は要旨だけ読んだ [Castanyer SOFE (2023/10), Abstract]。

### 5. 批判と懸念

KFM の 3 つの値は、学習器、予算、統計量、RND の重みが違う。FARCuriosity は workshop、PreND・Juliani と Ash・Klein らのサーベイ・LPM は arXiv のままである。PAME は多 agent、DrM は連続制御、Qiao らはモデルベースで、単一 agent の DQN 系に RND を足した設定ではない。

### 6. 総合評価

ゲーム別の公開の表は Rainbow・PPO・R2D2 の 3 つの学習器で揃い、密な報酬のゲームで RND が下げる例と上げる例の両方があった。predictor の周期的な初期化し直しの公開の前例は見つからなかった。可塑性の喪失と探索を結ぶ研究は、価値の表現・方策の学習・行動の量のどれかが崩れると書き、内的報酬の大きさを崩れるものとして扱うのは PreND だけだった。

### 7. 調査の限界

被引用数は Semantic Scholar の 429 で取れていない。Taïga らと NGU の表は、PDF の文字起こしの列のずれを ar5iv の行と照らして合わせた。FARCuriosity の PDF は取れず ar5iv だけで読み、IQM の行は列見出しとの対応が明示されていない。公開の実装のコードは読んでいない。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [Model-based Atariの評価](../../concepts/methods/model_based_evidence.md)
- [LLMの外部知識と探索](../../concepts/methods/llm_prior_knowledge.md)
- [訪問回数と擬似カウントによる探索](../../concepts/methods/pseudo_counts.md)
