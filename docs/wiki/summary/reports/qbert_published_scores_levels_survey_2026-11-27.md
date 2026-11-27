---
title: "Survey: Atari の Q*bert の公開の値は何面まで進んだ点か"
page_kind: サマリ
material: "[[reports/qbert_published_scores_levels_survey_2026-11-27.md|qbert_published_scores_levels_survey_2026-11-27]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-27
---
# Survey: Atari の Q*bert の公開の値は何面まで進んだ点か

## 元の材料

[Survey: Atari の Q*bert の公開の値は何面まで進んだ点か](../../../../reports/qbert_published_scores_levels_survey_2026-11-27.md) blob:76a5907bef609f5bd88bc6fd9573ff4b26779dde

## 要点

- 200M フレーム級の Q*bert の公表値は、Rainbow 33,817.5（学習中に最高だったスナップショットの 200 本の平均）[Rainbow (2017/10), Table 6]、IQN 25,750 と QR-DQN 572,510（1 seed）[IQN (2018/06), Figure 7]、BTR 42,927（学習中の最高、3 seed）[BTR (2024/11), Table A2] である。
- 数十億フレームの分散学習では、Ape-X 302,391.3（22.8B、最終評価）[Ape-X (2018/03), Table 2]、R2D2 408,850.00（37.5B）と MuZero 72,276.00（20.0B、1000 本の平均）[MuZero (2019/11), Table S1・Table 1]、Agent57 580,328.14 ± 151,251.66（6 seed、50 本の移動平均の学習中の最大）[Agent57 (2020/03), 付録 H.4・§4] である。
- sticky actions 0.25 の評価では、1 ゲーム 5 分の上限の DQN が 200M で 9,875.5 [Machado (2017/09), Table 9]、上限 100 時間の SABER の Rainbow-IQN が 200M で 18,736.25 だった [SABER (2019/08), Table 6]。
- Atari 2600 版の説明書の得点は、目標の色のキューブ 25 点、ラウンドの完了 3,100 点、Sam 300 点、緑の球 100 点、Coily 500 点で、5 レベル × 4 ラウンドである。レベル 3 からは目標の色のキューブに乗り直すと色が戻る [AtariAge 説明書 (1987)]。
- ALE の Q*bert の報酬は得点の差分だけで、面の番号を使わない [ALE QBert.cpp (2024/09), step]。
- 報告の読みでは、キューブ 21 個なら 1 ラウンドは最低 3,625 点なので、点から読めるのは終えたラウンドの数の上限だけで、下限は読めない（§2.4）。
- 進化戦略の論文は、敵を道連れに落ちて残機を取り戻すくり返しと、第 1 面のあとに次のラウンドへ進まず 100 万点近くまで点が入る不具合を報告した [Chrabaszcz (2018/02), §5]。
- 30 万点を超える公表値を、不具合や稼ぎに当たると書いた資料と、到達したラウンドを点と別に報告した Q*bert の研究は、見つからなかった（§3.2・§4）。

## 内容

### 1. 公表の点

Rainbow の Table 6（no-op starts、最大 30 回の no-op、1 ゲーム 108K フレーム）の Q*bert は、DQN 13,117.3、DDQN 15,088.5、Prior. DDQN 18,802.8、Duel. DDQN 19,220.3、Distrib. DQN 16,956.0、Noisy DQN 15,276.3、Rainbow 33,817.5 である。Rainbow の seed の数は読んだ範囲に無く、IQN の Table 1 は 2 seed と書く [Rainbow (2017/10), Table 6]、[IQN (2018/06), Table 1]。IQN の Figure 7 は 1 seed・30 回の no-op で、IQN 25,750、QR-DQN 572,510、Prior. Duel. 18,760.3、人の基準 13,455.0、ランダム 163.9 である [IQN (2018/06), Figure 7]。

BTR の Table A2 は Rainbow（Dopamine）18,503、Rainbow（Full）33,818、M-IQN 14,739、BTR 42,927 で、ライフ情報を使う BTR は 130,348、PQN（400M フレーム）は 31,716 である [BTR (2024/11), Table A2・Table A3]。BTR の本文は Q*bert に触れない。

Ape-X の human starts は 380,152.1 で、学習中の 1 ゲームの上限は評価より短い [Ape-X (2018/03), Table 2]。Figure 9 の Q*bert の曲線は、終盤に 30 万点台へ急に跳ね上がる（報告の目視）。MuZero の表は 30 分（108k フレーム）の上限、1 手 50 回の探索で、human starts では MuZero 39,302.10 である [MuZero (2019/11), Table S1・Table S2]。Agent57 の付録は、NGU 479,024.20 ± 98,094.39、R2D2 (Retrace) 434,753.72 ± 99,793.58、著者が回した R2D2 771,069.21 ± 152,722.56、R2D2 (Bandit) 777,071.30 ± 190,653.94 も載せる。sticky actions なし、30 分の上限で、± の意味は書かれていない [Agent57 (2020/03), 付録 H.4・Table 2]。

Machado らの DQN（5 回の試行、最後の 100 本、frame skip 5）は 10M・50M・100M で 674.7・3,142.1・7,585.4 である [Machado (2017/09), Table 9]。SABER の Rainbow-IQN（1 seed、5 分動かなければ打ち切り）は 10M・50M・100M で 4,343.75・14,809.5・16,812.5 で、人の世界記録は 2,400,000 である [SABER (2019/08), Table 5・Table 6]。

### 2. 得点の仕組みと点から面を読めるか

説明書では、レベル 1 は 1 回で目標の色、レベル 2 は中間の色を経る。レベル 3 は乗り直すと元の色、レベル 4 は中間の色、レベル 5 は元の色へ戻る。レベル 5 のあとはレベル 5 をくり返し、ラウンドごとに Q*bert も敵も速くなる。残機は 4 で、最初の 5 ラウンドのあとに 1 機、その後は 4 ラウンドごとに 1 機増える。中間の色と円盤の点、キューブの数は書かれていない [AtariAge 説明書 (1987)]。通算の第 9 ラウンドがレベル 3 のラウンド 1 にあたる（報告の読み）。

ALE は残機を RAM 0x88 から読み、0xFE で終わりとする。文書は点の値を書かず、説明書へリンクする [ALE QBert.cpp (2024/09)]、[ALE-Qbert 文書 (2026/11)]。報告の計算では、終えたラウンドの数の上限は Rainbow 9、IQN 7、BTR 11、MuZero 19、Ape-X 83、R2D2 112、QR-DQN 157、Agent57 160 である（§2.4）。

### 3. 同じ面で点を重ねる振る舞いと不具合の報告

不具合では、同じ重みでも 30 回の評価のうち 22 回は低い点で、原因は分からないとされた。1 ゲームの上限は 25k step で、ライフを失っても終わらせない [Chrabaszcz (2018/02), §4・§5]。同じ論文の Canonical ES（5 時間）の Q*bert は 263,242 ± 433,050 などで、この行が不具合の解かは書かれていない [Chrabaszcz (2018/02), Table 1]。説明書は残機をラウンドの数で増やすと書き、ES の論文の残機の書き方と違う。

SABER は、得点が 100 万で巻き戻る不具合を Asterix・Video Pinball・Defender で見つけ、−1000 より小さい報酬を記録して見分けることを勧める。Q*bert は挙がっていない [SABER (2019/08), 付録 C]。

### 4. 面の進みを点と別に報告した研究

Q*bert の到達ラウンドを点と別に報告した研究は見つからず、面に触れたのは ES の論文の定性の記述だけだった。

### 5〜8. 手元の記録との照合、懸念、総合評価、限界

§5 は手元の実験記録と並べた報告の読みで、比較は概念のページに置く。§6 は、統計量（学習中の最高・最終評価・1000 本の平均・最後の 100 本）と 1 ゲームの上限（30 分・5 分・100 時間・25k step）がそろわないことを懸念に挙げる。R2D2 の原論文と AtariAge の掲示板は読めず、Agent57 の学習のフレーム数は確かめていない（§8）。

## 主要な概念

- [Qbert](../../concepts/subjects/qbert.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
