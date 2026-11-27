# Survey: Atari の Q*bert の公開の値は何面まで進んだ点か

Date: 2026-11-27
Scope: 実績のあるアルゴリズムの Q*bert の公表値と、その評価の条件、Atari 2600 版の得点の仕組み、同じ面で点を重ねる振る舞いと不具合の報告、面の進みを点と別に報告した研究。一次資料（論文の本文と付録の表、説明書、ALE の文書とコード）を先に読み、二次資料（報道）は一次に無い部分の補いにだけ使った。

## 対象の問い

- 実績のあるアルゴリズム（Rainbow、IQN、Agent57、MuZero、BTR など）の Q*bert の公表の点は、いくつか（学習器、フレーム数、評価の方策、統計量を添える）。
- Q*bert の得点の仕組み（1 面あたりの点、面とラウンドの進み方、ボーナス）から、その点が何面あたりに当たるかを読めるか。得点の仕組みは、公式の資料（マニュアルなど）か、ALE の文書で確かめる。
- Q*bert で、エージェントが面を進めずに同じ面で点を重ねる（稼ぐ）振る舞い、既知のバグ技や無限に点を取る抜け道（進化戦略の論文などで報告されたもの）は、どう報告されているか。高い公表値のうち、どれがそれに当たると書かれているか。
- 「何面まで進んだか」を、点とは別に報告した研究はあるか。

## 結論

- **公表の点。** 200M フレーム級では、Rainbow 33,817.5、IQN 25,750、BTR 42,927 で、同じ表の QR-DQN だけが 572,510 と桁が違う。数十億フレームの分散学習では、Ape-X 302,391.3、R2D2 408,850.00、Agent57 580,328.14 ± 151,251.66 で、MuZero は 72,276.00 にとどまる。統計量は手法ごとに違い、学習中の最高（Rainbow・BTR・Agent57）、最終評価（Ape-X）、1000 本の平均（MuZero）が混ざる（§1）。
- **点から面を読めるか。** 説明書では、目標の色にしたキューブが 25 点、ラウンド完了が 3,100 点で、5 レベル × 4 ラウンドである。第 3 レベル（通算の第 9 ラウンド）からは、目標の色のキューブに乗り直すと色が戻る。ALE の報酬は画面の得点の差分だけで、面の情報を含まない。キューブの数を 21 とすると 1 ラウンドは最低 3,625 点なので、点から読めるのは進んだラウンドの数の上限だけで、下限は読めない（本調査の読み、§2）。
- **稼ぎと不具合。** 進化戦略（Canonical ES）の論文は、Q*bert で 2 つの振る舞いを報告した。敵を道連れに落ちて残機を取り戻すくり返しと、第 1 面を終えても次のラウンドへ進まず、100 万点近くまで点が増える不具合である。高い公表値（Ape-X・R2D2・Agent57・QR-DQN）を、この不具合や稼ぎに当たると書いた論文は見つからなかった。各論文の本文は Q*bert の振る舞いに触れていない（§3）。
- **面を別に報告した研究。** Q*bert について、到達したラウンドやレベルを点と別に数えて報告した研究は見つからなかった。ES の論文の「次のラウンドへ進まない」という定性の記述が、見つけた範囲で唯一の面の記述である（§4）。

## 調べた範囲と日付

- 日付: 2026-11-27。
- 読んだ一次資料: Rainbow、IQN、Ape-X、Agent57、MuZero、BTR、Machado ら（Revisiting the ALE）、SABER（Toromanoff ら）、Canonical ES（Chrabaszcz ら）の論文の本文と付録の表。Atari 2600 版 Q*bert の説明書（ALE の文書がリンクする AtariAge の写し（1987 年の Atari Corporation 版）、1983 年の Parker Brothers の規則の写し、もう 1 つの 1987 年版の写し）。ALE の Q*bert の文書とコード（`QBert.cpp`、最終の変更 2024-09-04、GitHub の API で読んだ）。
- 検索語（一般の語だけ）: 「Q*bert Atari 2600 manual scoring」「Qbert bug exploit infinite points reinforcement learning」「Qbert bug high score Agent57 R2D2 Ape-X」「Qbert rounds levels reached agent」「Qbert level 3 cubes revert agent stuck」「Qbert reward hacking shortcut」。
- 被引用数は OpenAlex（2026-11-27 取得）の値で、ほかの集計より低く出ることがある。

## 見つからなかったこと

- 高い公表値（Ape-X・R2D2・Agent57・QR-DQN）が、不具合や同じ面での稼ぎによると書いた資料。上記の論文の本文の Q*bert の語を全部当たったが、どれも表と図の中だけだった。
- 到達したラウンドやレベルを点と別に報告した研究（検索語は上記）。
- R2D2 の原論文（OpenReview の PDF が取得の確認画面で読めなかった）。R2D2 の値は MuZero の Table S1 の写しを使った。
- Agent57 の Q*bert の値を出した学習のフレーム数（本文の検索で見つからなかった）。
- AtariAge の掲示板の「Qbert infinite points」の議論（HTTP 403 で読めなかった）。

## 目次

1. [公表の点](#1-公表の点)
2. [得点の仕組みと点から面を読めるか](#2-得点の仕組みと点から面を読めるか)
3. [同じ面で点を重ねる振る舞いと不具合の報告](#3-同じ面で点を重ねる振る舞いと不具合の報告)
4. [面の進みを点と別に報告した研究](#4-面の進みを点と別に報告した研究)
5. [anet の記録と並べた読み](#5-anet-の記録と並べた読み)
6. [批判・懸念](#6-批判懸念)
7. [総合評価](#7-総合評価)
8. [調査の限界](#8-調査の限界)
9. [出典](#9-出典)

## 1. 公表の点

### 1.1 200M フレーム級

**Rainbow の表の Q*bert は 33,817.5 で、同じ表の DQN 系の既存手法は 13,117.3〜19,220.3 だった。** Table 6 は no-op starts（開始時に最大 30 回の no-op）で、学習中に最高点を出したスナップショットを 200 本で評価した平均である。DQN・DDQN・Dueling・Prioritized は既存の論文の値を写し、Distributional DQN・Noisy DQN・Rainbow は著者の再評価である [Rainbow (2017/10), Table 6]。学習は 200M フレームで、1 ゲームは 108K フレーム（30 分）で打ち切る [Rainbow (2017/10), Evaluation Methodology]。Q*bert の行は DQN 13,117.3、DDQN 15,088.5、Prior. DDQN 18,802.8、Duel. DDQN 19,220.3、Distrib. DQN 16,956.0、Noisy DQN 15,276.3、Rainbow 33,817.5 である [Rainbow (2017/10), Table 6]。Rainbow の論文の読んだ範囲には seed の数が無く、IQN の Table 1 は Rainbow を 2 seed と書く [IQN (2018/06), Table 1]。

**IQN の表の Q*bert は IQN 25,750 で、同じ表の QR-DQN は 572,510 だった。** Figure 7 の表は 1 seed の値で、30 回の no-op で始める。Prior. Duel. は 18,760.3、DQN は 13,117.3 で、人の基準は 13,455.0、ランダムは 163.9 である [IQN (2018/06), Figure 7]。学習曲線は 200M フレームまで描かれている [IQN (2018/06), Figure 6]。

**BTR の Table A2 の Q*bert は 42,927 で、同じ表の Rainbow（Dopamine）は 18,503、Rainbow（Full）は 33,818、M-IQN は 14,739 だった。** Table A2 は 200M フレームの学習中の最高（100 本の平均）で、表題は 3 seed で行ったと書く。ライフ情報は使わない [BTR (2024/11), Table A2]。seed をどうまとめたか（max の平均か、平均の max か）は、既存の報告でも決まっていない（[few_seed_comparison_reporting_survey_2026-11-04.md](few_seed_comparison_reporting_survey_2026-11-04.md)、4.3）。ライフ情報を使って 200M フレームで学習した BTR は 130,348（3 seed）で、PQN（ライフ情報あり、400M フレーム）は 31,716 である [BTR (2024/11), Table A3]。BTR の評価の条件（sticky actions 0.25、最大 30 回の no-op、1 ゲームの上限 108K フレーム、評価は 1M フレームごとに 100 本）は既存の報告にある（[btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md)、5・7）。BTR の本文は Q*bert に触れず、名前は表にだけ現れる [BTR (2024/11)]。

### 1.2 分散・大規模

**Ape-X の最終評価の Q*bert は no-op starts で 302,391.3、human starts で 380,152.1 だった。** 表の注は、学習中の 1 ゲームの上限が評価より短いので、評価の値が学習曲線より高いゲームがあると書く [Ape-X (2018/03), Table 2]。学習は 22.8B フレームである [MuZero (2019/11), Table 1]。Figure 9 の Q*bert の Ape-X の曲線は、終盤に低い値から 30 万点台へ急に跳ね上がる（本調査の目視）[Ape-X (2018/03), Figure 9]。

**MuZero の表では、Q*bert は MuZero 72,276.00、R2D2 408,850.00、Ape-X 302,391.00、SimPLe 1,288.80 だった。** Table S1 は 30 回までの no-op で始め、1 ゲームを 30 分（108k フレーム）で打ち切る。MuZero の値は 1000 本の平均で、1 手 50 回の探索である [MuZero (2019/11), Table S1・Appendix の評価の節]。human starts の Table S2 では MuZero 39,302.10、Ape-X 380,152.00 である [MuZero (2019/11), Table S2]。学習のフレーム数は MuZero 20.0B、R2D2 37.5B、Ape-X 22.8B である [MuZero (2019/11), Table 1]。

**Agent57 の表の Q*bert は Agent57 580,328.14 ± 151,251.66、著者が回した R2D2 (Bandit) 777,071.30 ± 190,653.94 だった。** 同じ付録の別の表では、NGU 479,024.20 ± 98,094.39、R2D2 (Retrace) 434,753.72 ± 99,793.58、R2D2 771,069.21 ± 152,722.56 である [Agent57 (2020/03), 付録 H.4]。統計量は、評価専用のプロセスで 50 本の移動平均を取り、Agent57 は 6 seed、ほかは 3 seed で平均し、学習中の最大を最終の値にする [Agent57 (2020/03), §4]。1 ゲームの上限は 30 分、sticky actions なし、no-op は 30 まで、ライフ喪失で割引を 0 にしない [Agent57 (2020/03), Table 2]。± の意味は読んだ範囲に書かれていない。

### 1.3 sticky actions・時間の上限を変えた評価と、人の記録

**sticky actions 0.25 と 1 ゲーム 5 分の上限で評価した DQN は、200M フレームで 9,875.5（標準偏差 1,385.3）だった。** 10M・50M・100M フレームは 674.7・3,142.1・7,585.4 である。値は 5 回の試行の平均で、各試行は k フレームまでの最後の 100 本の平均である。行動は 18 個、frame skip は 5 である [Machado (2017/09), Table 9・Appendix B.1]。

**時間の上限を 100 時間にした SABER の評価では、Rainbow-IQN の Q*bert は 200M フレームで 18,736.25 だった。** 10M・50M・100M フレームは 4,343.75・14,809.5・16,812.5 である [SABER (2019/08), Table 6]。SABER は sticky actions 0.25、ライフ情報なし、行動 18 個、5 分動かなければ打ち切り、決めた時点の学習の 100 本の平均で報告する [SABER (2019/08), Table 1・§3.2]。Q*bert は 1 seed である（[few_seed_comparison_reporting_survey_2026-11-04.md](few_seed_comparison_reporting_survey_2026-11-04.md)、5.1）。同じ論文の人の世界記録は 2,400,000（表では 2400000.0、2019 年 7 月の参照）で、論文の人の基準は 13,455.0 である [SABER (2019/08), Table 5]。

### 1.4 表のまとめ

| 学習器 | Q*bert | 学習のフレーム | 評価の開始・上限 | 統計量 | seed | 出典 |
|---|---:|---|---|---|---|---|
| Rainbow | 33,817.5 | 200M | no-op 30・108K フレーム | 最高のスナップショットの 200 本の平均 | 記載を見つけず | Rainbow Table 6 |
| IQN | 25,750 | 200M | no-op 30 | 記載を見つけず | 1 | IQN Figure 7 |
| QR-DQN | 572,510 | 200M | no-op 30 | 記載を見つけず | 1 | IQN Figure 7 |
| BTR | 42,927 | 200M | sticky 0.25・no-op 30・108K | 学習中の最高（100 本） | 3 | BTR Table A2 |
| BTR（ライフ情報あり） | 130,348 | 200M | 同上 | 同上 | 3 | BTR Table A3 |
| Ape-X | 302,391.3 | 22.8B | no-op | 最終評価 | 記載を見つけず | Ape-X Table 2 |
| R2D2 | 408,850.00 | 37.5B | no-op 30・30 分 | 原論文の値の写し | 未確認 | MuZero Table S1 |
| Agent57 | 580,328.14 ± 151,251.66 | 未確認 | no-op 30・30 分 | 50 本の移動平均の学習中の最大 | 6 | Agent57 H.4 |
| MuZero | 72,276.00 | 20.0B | no-op 30・30 分 | 1000 本の平均 | 記載を見つけず | MuZero Table S1 |
| DQN（sticky） | 9,875.5 | 200M | sticky 0.25・5 分 | 最後の 100 本 × 5 試行 | 5 試行 | Machado Table 9 |
| Rainbow-IQN（SABER） | 18,736.25 | 200M | sticky 0.25・100 時間 | 学習の 100 本 | 1 | SABER Table 6 |

## 2. 得点の仕組みと点から面を読めるか

### 2.1 説明書の得点

**Atari 2600 版の説明書の得点は、キューブを目標の色にすると 25 点、Sam を捕まえると 300 点、緑の球を捕まえると 100 点、Coily を落とすと 500 点、ラウンドを終えると 3,100 点である。** 中間の色にした点と、使わなかった円盤の点は書かれていない。Sam はキューブの色を元に戻す相手として書かれる [AtariAge 説明書 (1987)、Scoring]。1983 年の Parker Brothers の説明書の写しと、もう 1 つの 1987 年版の写しも同じ 5 つの値である [説明書の写し (1983)、Scoring]、[説明書の写し (1987)]。説明書はピラミッドのキューブの数を書いていない [AtariAge 説明書 (1987)]。

### 2.2 面（ラウンド）とレベルの進み方

**説明書の Q*bert は 5 レベル × 4 ラウンドで、レベルごとにキューブの色の変わり方が違う。** レベル 1 は 1 回乗ると目標の色になり、そのまま残る。レベル 2 は中間の色を経て目標の色になる。レベル 3 は 1 回で目標の色になるが、目標の色のキューブに乗り直すと元の色に戻る。レベル 4 は目標の色から中間の色へ、レベル 5 は目標の色から元の色へ戻る。レベル 5 を終えると、レベル 5 のラウンド 1 からくり返す。ラウンドが進むたびに Q*bert も敵も速くなる [AtariAge 説明書 (1987)、Game Play]。残機は 4 で、最初の 5 ラウンドのあとに 1 機、その後は 4 ラウンドごとに 1 機増える [AtariAge 説明書 (1987)]。左の難易度スイッチの B（初心者）では赤い球が出ない [説明書の写し (1983)]。

各レベルが 4 ラウンドなので、通算の第 9 ラウンドがレベル 3 のラウンド 1 にあたり、目標の色のキューブが戻り始める最初のラウンドである（本調査の読み）。

### 2.3 ALE の報酬

**ALE の Q*bert の報酬は、RAM の 3 バイトから読む得点の前の step との差だけで、面の番号は報酬にも終了の判定にも使われない。** 残機は RAM 0x88 から読み、0xFE で終わりとする。終わりの step の報酬は 0 にする [ALE QBert.cpp (2024/09), step]。ALE の文書は、目標の色への変更、敵、レベルの完了で点が入ると書くが、値は書かず、AtariAge の説明書へリンクする。最小の行動は 6 個、難易度は 0 と 1 である [ALE-Qbert 文書 (2026/11)]。

### 2.4 点から面の数を読む（本調査の読み）

ここは本調査の読みである。キューブの数を、手元の記録の 21 個（[2026-09-21_realtime-profile-atari5.md](../docs/experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック 04）とすると、1 ラウンドを終えるには少なくとも 21 × 25 + 3,100 = 3,625 点が入る。8 ラウンドを終えた最低の点は 29,000 である。

- **上限は読める。** どのレベルでも、ラウンドを終えるたびに 3,625 点以上が入るので、終えたラウンドの数は「点 ÷ 3,625」を超えない。1 本ごとに成り立つので、本数の平均にも成り立つ。
- **下限は読めない。** レベル 3 以降は目標の色を戻して付け直すたびに 25 点が入り、Coily（500 点）や Sam（300 点）でも点が増えるので、同じラウンドの中で点を重ねられる。
- 公表値に当てはめた上限（本調査の計算）: Rainbow 33,817.5 は 9 ラウンド、IQN 25,750 は 7 ラウンド、BTR 42,927 は 11 ラウンド、MuZero 72,276.00 は 19 ラウンド、Ape-X 302,391.3 は 83 ラウンド、R2D2 408,850.00 は 112 ラウンド、QR-DQN 572,510 は 157 ラウンド、Agent57 580,328.14 は 160 ラウンドまでである。人の基準の 13,455.0 は 3 ラウンドまでである。
- 200M フレーム級の Rainbow と BTR の値は 29,000 を超えるが、上限が 9〜11 ラウンドなので、第 9 ラウンド（レベル 3）を大きく越えた値ではない。30 万点を超える値は、上限としては 80 ラウンドを超えるが、その分を面を進めて取ったか、同じ面で重ねたかは点からは分からない。

## 3. 同じ面で点を重ねる振る舞いと不具合の報告

### 3.1 進化戦略の論文の 2 つの振る舞い

**Chrabaszcz らは、Canonical ES が Q*bert で 2 つの特異な解を見つけたと報告した。** 1 つ目は、敵が隣に来たときに台から飛び降り、ついて来た敵を落とす解である。残機を 1 つ失うが、敵を倒した点で残機を取り戻せるので、これをくり返す [Chrabaszcz (2018/02), §5・Figure 7]。2 つ目は不具合で、第 1 面を終えたあと、でたらめに見える跳び方を続けると、次のラウンドへ進まずに台が点滅し、短い間に大量の点が入る。

```
the game does not advance to the second round

ゲームが 2 番目のラウンドへ進まない
```

[Chrabaszcz (2018/02), §5]

点は 1 ゲームの上限の中で 100 万点近くに達したが、同じ重みでも 30 回の評価のうち 22 回は低い点だった。著者は不具合の原因は分からないとした [Chrabaszcz (2018/02), §5]。1 ゲームの上限は 25k step で、ライフを失っても終わらせず、最大 30 回の no-op で始める [Chrabaszcz (2018/02), §4]。

**同じ論文の表の Q*bert は、Canonical ES の 5 時間で 263,242 ± 433,050、16,673.3 ± 6.2、5,136.7 ± 4,093.9 だった。** 3 回の学習を平均の順に並べ、各回を 30 回の評価（最大 30 回の no-op）で平均した値である。同じ条件の OpenAI ES（著者の再実装）は 12,775・5,075・4,300 だった [Chrabaszcz (2018/02), Table 1]。表の 263,242 の行が不具合の解かは、論文に書かれていない。30 回のうち 8 回が 100 万点近く、残りが低い点だと、平均は 25 万点を超えて 263,242 と同じ桁になり、標準偏差の大きさとも合う（本調査の読み）。

**説明書と ES の論文は、残機の増え方の書き方が違う。** 説明書は残機をラウンドの数で増やすと書き、ES の論文は敵を倒した点で残機が増えると書く [AtariAge 説明書 (1987)]、[Chrabaszcz (2018/02), §5]。どちらが ALE の挙動かは確かめていない。

報道では、著者が、不具合の出る正確な条件は分からず、最適でなく見える動きのあとにだけ出るのかもしれないと述べ、不具合が出たのは 30 回のうち 8 回だったと伝えられた。これは二次資料で、論文に同じ記述は無い [The Register (2018/03)]。

### 3.2 高い公表値について書かれていること

**30 万点を超える Q*bert の公表値（Ape-X 302,391.3、R2D2 408,850.00、Agent57 580,328.14、QR-DQN 572,510）について、不具合や同じ面での稼ぎに当たると書いた箇所は見つからなかった。** Rainbow・IQN・Ape-X・Agent57・MuZero の本文の Q*bert の語を全部当たると、表と図のほかに Q*bert が出るのは IQN の危険回避の方策の節（CVaR(0.1) が Q*bert と Space Invaders で少し下がる）だけで、振る舞いの記述ではない [IQN (2018/06), §4]。BTR の本文も Q*bert に触れない [BTR (2024/11)]。

### 3.3 ほかのゲームの得点の不具合

**SABER は、得点が 100 万に達すると巻き戻る不具合を Asterix・Video Pinball・Defender で見つけ、Ape-X と Rainbow の Asterix の公表値にもその形が出ていると書いた。** エミュレータが 100 万点で −1M の報酬を返し、点が 0 へ戻るので、弱いエージェントが高い点で報告されうる。著者は −1000 より小さい報酬を記録して、巻き戻りを見分けることを勧める [SABER (2019/08), 付録 C]。同じ付録は、乱数の行動でも永久に止まるゲームがあることを、5 分動かなければ打ち切る理由に挙げる [SABER (2019/08), 付録 C]。Q*bert はここに挙がっていない。

## 4. 面の進みを点と別に報告した研究

**Q*bert で、到達したラウンドやレベルを点と別に数えて報告した研究は見つからなかった。** 検索語は「調べた範囲と日付」に挙げた。見つけた範囲で面に触れたのは、ES の論文の「第 1 面を終えたあと次のラウンドへ進まない」という定性の記述だけである [Chrabaszcz (2018/02), §5]。Rainbow・IQN・Ape-X・Agent57・MuZero・BTR・Machado・SABER は、Q*bert を点だけで報告している（各論文の表）。

## 5. anet の記録と並べた読み

この節は、外の資料と anet の記録を並べた本調査の読みである。

- **第 9 面で止まる位置。** anet の記録は、BTR の公開実装を回した 2 本も anet も第 9 面（L3R1）で止まり、そこで点を重ねたと書く（[2026-10-23_btr-qbert-repro-init.md](../docs/experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)、現時点のまとめ 2・8・9）。Evaluation Trace では、第 9 面で真ん中付近のキューブの色を付けたり戻したりして、色が変わるたびに 25 点が入った（[2026-10-29_temporl-auxiliary-only-stage0.md](../docs/experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、現時点のまとめ 8）。説明書では、目標の色のキューブが戻り始めるのがレベル 3 で、通算の第 9 ラウンドから当たる（§2.2）。止まる位置と、色が戻るルールの始まる位置は同じである。ラウンドごとに速くなることも説明書にあるので、どちらが壁の原因かは、これだけでは分けられない。
- **第 1 面のあとに進まない形。** 64 env・1M の seed 4 は、34M 以降に面が進まないまま点だけが増えるゲームに入った。ε=0 の断面の 35.6M・40.6M・45.7M は、`round_clear` = 1 が 87 / 99 / 95 本で、その平均は 67,029〜67,050 点、どれも 26,993 step 以上（上限まで）続いた。train の最高は 996,450 点（`round_clear` = 3、25,798 step）だった（[2026-09-27_btr-gap-qbert-bz.md](../docs/experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、探索ブロック 06）。ES の論文の不具合も、第 1 面のあとに次のラウンドへ進まず、上限の中で 100 万点近くまで増えた（§3.1）。同じ不具合かは確かめていない。
- **1 ラウンドの点。** anet の Trace では、第 8 面の終わりに最後のキューブの +25 のあと +100 ずつ増え、計 3,125 点が入った（[2026-10-29_temporl-auxiliary-only-stage0.md](../docs/experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)、探索ブロック 04）。説明書の 25 点と 3,100 点の和と合う。

## 6. 批判・懸念

- **説明書は写しである。** AtariAge の写しは 1987 年の Atari Corporation 版で、ALE の文書がリンクする先である。1983 年の Parker Brothers 版の写しと得点・面の記述は一致した。ALE の ROM がどちらの版かは確かめていない。
- **公表値の統計量がそろわない。** 学習中の最高（Rainbow・BTR・Agent57）、最終評価（Ape-X）、1000 本の平均（MuZero）、学習中の最後の 100 本（Machado・SABER）が混ざる。Agent57 は 50 本の移動平均の最大で、ほかより上に出やすい [Agent57 (2020/03), §4]。
- **1 ゲームの上限が違う。** 30 分（108K フレーム）が多いが、Machado は 5 分、SABER は 100 時間、ES は 25k step である。同じ面で点を重ねる形では、上限が長いほど点が増える（本調査の読み）。
- **R2D2 の値は写しである。** 原論文を読めず、MuZero の表の値を使った。Agent57 の表の R2D2 は著者が回した値で、原論文の値と約 2 倍違う。
- **Ape-X の曲線の跳ね上がりは目視である。** 図の解像度が低く、値は読んでいない。

## 7. 総合評価

- 200M フレーム級の公表値（Rainbow 33,817.5、BTR 42,927、IQN 25,750）は、説明書の得点から読む上限で 7〜11 ラウンドに収まり、第 9 ラウンドの前後の点である。29,000 を超える分が第 9 ラウンドで重ねた点かは、点からは分からない。
- 30 万点を超える公表値は、上限としては 80 ラウンドを超えるが、面を進めた点か、不具合や稼ぎの点かを示す記述は、論文に無かった。ES の論文の不具合は、第 1 面のあとに 100 万点近くまで点が増える形で報告されている。
- 面の進みを点と別に報告した Q*bert の研究は見つからず、面で比べるには、手元で `round_clear` のような面の値を記録するしかない。

## 8. 調査の限界

- R2D2 の原論文、AtariAge の掲示板は読めなかった。報道は 1 本だけ二次資料として使った。
- Agent57 の Q*bert の学習のフレーム数と、± の意味は確かめていない。
- ES の不具合が ALE の今の版でも出るか、ALE の得点が 100 万点で巻き戻るかは確かめていない。
- キューブの数 21 は手元の記録の値で、説明書には無い。
- 被引用数は OpenAlex の値で、少なく出ることがある。

## 9. 出典

一次資料（論文）:

- [Chrabaszcz, 2018/02] Patryk Chrabaszcz, Ilya Loshchilov, Frank Hutter（University of Freiburg）. "Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari." arXiv:1802.08842（IJCAI 2018）。被引用数 130（OpenAlex）. https://arxiv.org/pdf/1802.08842
- [Rainbow, 2017/10] Matteo Hessel ほか（DeepMind）. "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv:1710.02298（AAAI 2018）。被引用数 2617（OpenAlex）. https://arxiv.org/pdf/1710.02298
- [IQN, 2018/06] Will Dabney, Georg Ostrovski, David Silver, Rémi Munos（DeepMind）. "Implicit Quantile Networks for Distributional Reinforcement Learning." arXiv:1806.06923（ICML 2018）。被引用数 857（OpenAlex）. https://arxiv.org/pdf/1806.06923
- [Ape-X, 2018/03] Dan Horgan ほか（DeepMind）. "Distributed Prioritized Experience Replay." arXiv:1803.00933（ICLR 2018）。被引用数 998（OpenAlex）. https://arxiv.org/pdf/1803.00933
- [Agent57, 2020/03] Adrià Puigdomènech Badia ほか（DeepMind）. "Agent57: Outperforming the Atari Human Benchmark." arXiv:2003.13350（ICML 2020）。被引用数 730（OpenAlex）. https://arxiv.org/pdf/2003.13350
- [MuZero, 2019/11] Julian Schrittwieser ほか（DeepMind、University College London）. "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." arXiv:1911.08265v2（Nature 2020）。被引用数 3468（OpenAlex）. https://arxiv.org/pdf/1911.08265
- [BTR, 2024/11] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." arXiv:2411.03820v2（ICML 2025）。被引用数 13（OpenAlex）. https://arxiv.org/html/2411.03820v2
- [Machado, 2017/09] Marlos C. Machado, Marc G. Bellemare ほか（University of Alberta ほか）. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." arXiv:1709.06009v2（JAIR 2018）。被引用数 685（OpenAlex）. https://arxiv.org/pdf/1709.06009
- [SABER, 2019/08] Marin Toromanoff, Emilie Wirbel, Fabien Moutarde（MINES ParisTech、Valeo）. "Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the playing field." arXiv:1908.04683v5。被引用数 12（OpenAlex）. https://arxiv.org/pdf/1908.04683

一次資料（説明書・ALE）:

- [AtariAge 説明書, 1987] Atari Corporation（2600 版は JVW Electronics, 1983）. "Q*bert" 説明書の HTML の写し. AtariAge. https://atariage.com/manual_html_page.php?SoftwareID=1224
- [説明書の写し, 1983] Parker Brothers. "Q*bert" 説明書（Rules (c) 1983）のテキストの写し. https://discmaster.textfiles.com/view/1886/1000gamemanuals.iso/Atari%202600/Q-Bert.txt
- [説明書の写し, 1987] Atari Corporation. "Q*bert" 説明書の HTML の写し（History of Home Video Games の旧サイトからの転載）. https://ftp.pigwa.net/stuff/collections/RK-DVD/Spiele%20fuer%20Konsolen/2600%20Games%20and%20Manuals/Manuals524/qbert.htm
- [ALE-Qbert 文書, 2026/11] Farama Foundation. "Qbert - ALE Documentation."（2026-11-27 参照）. https://ale.farama.org/environments/qbert/
- [ALE QBert.cpp, 2024/09] Farama Foundation. "Arcade-Learning-Environment / src/ale/games/supported/QBert.cpp"（最終の変更 2024-09-04、commit ca9b753a）. https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/QBert.cpp

二次資料（報道）:

- [The Register, 2018/03] Katyanna Quach（The Register の記者）. "AI racks up insane high scores after finding bug in ancient video game."（2018-03-02）. https://www.theregister.com/2018/03/02/ai_qbert_bug/

手元の資料:

- [btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md)
- [few_seed_comparison_reporting_survey_2026-11-04.md](few_seed_comparison_reporting_survey_2026-11-04.md)
- [2026-09-21_realtime-profile-atari5.md](../docs/experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)
- [2026-09-27_btr-gap-qbert-bz.md](../docs/experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)
- [2026-10-23_btr-qbert-repro-init.md](../docs/experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)
- [2026-10-29_temporl-auxiliary-only-stage0.md](../docs/experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md)
