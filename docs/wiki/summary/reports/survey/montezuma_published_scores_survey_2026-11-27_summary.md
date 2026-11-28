---
title: "Survey: Montezuma's Revenge の公開の成績と、その測り方"
page_kind: サマリ
material: "[[reports/survey/montezuma_published_scores_survey_2026-11-27.md|montezuma_published_scores_survey_2026-11-27]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/self_imitation.md|自己模倣による価値学習]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
updated: 2026-11-28
---
# Survey: Montezuma's Revenge の公開の成績と、その測り方

## 元の材料

[Survey: Montezuma's Revenge の公開の成績と、その測り方](../../../../../reports/survey/montezuma_published_scores_survey_2026-11-27.md) blob:ee7bdebdeaed7cda93dd29a1a64e30bcd45983e5

## 要点

- Taïga らの付録の表 2〜5 の表題は、10M・50M・100M・200M フレームの学習のあとの値で、学習中の最良とは書いていない。Montezuma の RND は 100M フレーム 5188.8、200M フレーム 5199.2 である [Taïga Bonus-ALE (2021/09), 表 2〜5]。
- Taïga らの本文は、評価の局の ε、1 回の評価のゲーム数、値が学習のゲームか評価のゲームかを書いていない。公開の設定は、1 iteration が学習 250,000 agent step と評価 125,000 agent step、200 iteration、評価の ε 0.001、1 ゲームの上限 27,000 agent step である [Taïga code (2026/11), `rainbow_rnd.gin`]。
- Taïga らの NoisyNets の版は ε を使わず、雑音の全結合層は評価のときに雑音を切る引数を持たない [Taïga code (2026/11), `noisy_rainbow_agent.py`・`noisy_dqn_agent.py`]。
- RND 原典の表 1（Montezuma の RND 8,152）は最終の平均で、学習器は PPO、多くの実験は 128 env・1.97B フレーム、同じ比較の図 7 は 3 seed である。1024 env の RNN の方策は学習の終わりの平均が 10,070 だった [Burda RND (2018/10), §3、§3.4、表 1、表 6]。
- Taïga らの ε-greedy で点を取ったのは 5 本のうち 2 本で、20M フレーム前後から点が出た。NoisyNets の線はどれも 10〜20M フレームで 0 を離れた（図を読んだ値）[Taïga Bonus-ALE (2021/09), 図 7]。
- Dopamine の公開の基準値では、Rainbow と IQN の 5 本のうち各 1 本だけが 2500.0 に達し、DQN の 5 本は 198 回目の iteration がどれも 0.0 だった [Dopamine baselines JSON (2026/11)]。
- Dopamine の既定の Rainbow と IQN は、n-step 3・ε・更新の間隔が同じで、PER の有無・分布の形・学習率が違う。SAIL の論文は、表で 0 の IQN を n-step を使わない設定と書く [Dopamine repo (2026/11)；Ferret SAIL (2020/12), §4.3]。
- 素の Rainbow に点が出て IQN・DQN が 0 に近い理由を説明した文は、読んだ範囲に無かった。
- BTR の表 A1 の Montezuma は BTR 0、Rainbow 500 で（学習中の最大、100 ゲームの平均、200M フレーム）、公開の CSV の 200 回の評価はすべて 0.0 だった [Clark BTR (2024/11), 表 A1；BTR results.csv (2026/11)]。

## 内容

### 1. 対象の問いと結論

Q16 は Taïga らと RND 原典の Montezuma の値の測り方（最後か最良か、評価の方策、sticky actions、ゲーム数、seed）、Q17 は内的報酬を足さない学習器の点（点を取り始めた時点、Dopamine の Rainbow と IQN の設定の差と説明、BTR の値）を問う。見つからなかったことは、Taïga らの値が学習の局と評価の局のどちらか、NoisyNets の版の評価の ε、RND 原典の表 1 の env・フレーム・seed、素の Rainbow に点が出る理由の説明、BTR の表 A1 の Rainbow 500 の出どころ、Dopamine の基準値が学習の局と評価の局のどちらかである。

### 2. Taïga らの比較の値と測り方

arXiv 2109.11052 は v1 だけで、注記は 1908.02388 の完全版、雑誌の欄と PDF の頁頭は ICLR 2020 の会議論文とする。全手法を Dopamine の Rainbow（n-step、PER、分布型）の上で、学習器の設定を固定して 200M フレーム学習し、sticky actions（0.25）を学習と評価に使い、mixed Monte-Carlo return は使わない [Taïga Bonus-ALE (2021/09), §3.1]。付録 A.1 は、割引率 0.99、学習の開始 80K フレーム、target の更新 32K フレーム、Adam の学習率 6.25×10^-5・ε 1.5×10^-4、n-step 3、atom 51、値の範囲 [−10, 10] とし、NoisyNets 以外は ε を 1M フレームで 1 から 0.01 へ下げる。ボーナスは Montezuma の 200M フレームの最終性能（5 本の平均）で調整し、NoisyNets は σ0 0.5 のまま調整しない [Taïga Bonus-ALE (2021/09), 付録 A]。

表 2〜5 の Montezuma の行は、RND が 1.5・2789.6・5188.8・5199.2、CTS が 231.8・2550.4・5118.8・5028.2、PixelCNN が 17.8・2528.5・4364.6・4652.3、ICM が 0.0・1160.7・2449.8・2795.9（10M・50M・100M・200M フレームの順）である [Taïga Bonus-ALE (2021/09), 表 2〜5]。図 1 は 5 seed の平均で、付録 C は報酬が階段状で分散が大きいとしてすべての曲線を図 7 に載せる（ar5iv の HTML では図 9）。RND の線は 90M フレームのあと平らで、約 5,700 の帯と約 4,550 の帯に分かれた（図を読んだ値）。

公開の設定は、学習の ε 0.01（250,000 agent step で下げる）、PER、n-step 3、replay 1,000,000、内的報酬の倍率 0.00005、ライフ区切りなしで、実行の既定は学習と評価を交互に行う Dopamine の Runner である [Taïga code (2026/11)]。今の Dopamine の Runner は局の値をその局で終わったゲームの平均にし、Dopamine の論文は学習と評価の点の報告にほとんど差が無いと書く [Dopamine repo (2026/11)；Castro Dopamine (2018/12), §4.2]。

### 3. RND 原典の値と測り方

Atari の設定は sticky actions 0.25、1 ゲーム 18K フレームまで、ライフ区切りなし。表 1 は PPO 2,497、ランダム特徴の順モデル 400 も並べる。表 1 が何 env・何フレーム・何 seed の値かは書いていない。32 env の RNN の方策は 1.6B フレームで平均 7,570 に届き、1 本は 24 部屋すべてに入って最良 17,500、1024 env の 1 本は平均 14,415 だった。別の評価の手順は本文に出てこない [Burda RND (2018/10), 表 2、§3.4、§3.6]。

測り方の並びの表は、Taïga ら（各時点の値、5 seed、公開の設定の評価の ε 0.001）、RND 原典（最終の平均）、SAIL（最終の値で期間の平均を別に書く。RND-IQN は 161、期間の平均 45。指標は 1M step ごとの 200 回の値で、seed は部分集合の比較が 6、全ゲームが 3）、BTR（学習中の最大、64 env、図 1 は 4 seed）を並べる [Ferret SAIL (2020/12), §4.1・§4.4；Clark BTR (2024/11)]。

### 4. 内的報酬を足さない学習器の点

Taïga らの ε-greedy の 2 本は、1 本が 20M フレーム前後に約 400、45M フレーム前後に約 2,500、もう 1 本が 30M フレーム前後に約 400、70M フレーム前後に約 1,400 になり、ほかの 3 本は 200M フレームまで 0 に見える。平均は 10M フレーム 0.5、50M フレーム 553.3、100M・200M フレーム 780.0。NoisyNets は 30〜40M フレームまでに約 2,500 に達し、平均は 10M フレーム 13.2、50M フレーム 2492.9、200M フレーム 2435.4 である（図の値は図を読んだ値）[Taïga Bonus-ALE (2021/09), 図 7、表 2〜5]。

Dopamine の基準値の Rainbow の 1 本は 20 回目の iteration に 133.3、34 回目に 2218.2、40 回目と 198 回目に 2500.0、IQN の 1 本は 100 回目に 100.0、123 回目に 2325.0、124 回目と 198 回目に 2500.0 だった。値が学習の局か評価の局かはデータに書いていない [Dopamine baselines JSON (2026/11)]。

Dopamine の既定では、Rainbow が C51・n-step 3・PER・学習率 0.0000625、IQN が n-step 3・一様抽出・学習率 0.00005、DQN が n-step 1・一様抽出で、学習の ε 0.01、評価の ε 0.001、更新の間隔 4、target の更新 8,000 agent step、replay 1,000,000 は同じである。Dopamine の Rainbow は Double DQN・Dueling・NoisyNets を含まない [Dopamine repo (2026/11)；Castro Dopamine (2018/12), §3.1、付録 D]。

近い記述として、Taïga らは Rainbow が古い DQN 系より成績を上げると書き、要旨で Montezuma の伸びは Network の形の変化によるかもしれないと書き、結論で PER がボーナスを余分にする可能性を推測する。SAIL は RND の設定が Rainbow に合わせて調整されたものだと書く [Taïga Bonus-ALE (2021/09), 要旨・§3.2・§4；Ferret SAIL (2020/12), §4.4]。

### 5. BTR の Montezuma の値と条件

表 A1 の Montezuma は、DQN (Nature) 0、M-IQN 0、FE-Rainbow 0（10M フレーム、ライフの情報を使う）。DQN (Nature) と Rainbow の列の出どころは表 A1 に書いていない。評価は 1M フレームごとに 100 ゲーム、評価の ε は 125M フレームまで 0.01 でそのあと 0、IQM は学習中の最良の 1 回の評価を使う（脚注 1）。学習の ε は 8M フレームで 1.0 から 0.01 へ下げて 100M フレームで切り、NoisyNets の σ 0.5、n-step 3、割引率 0.997、sticky actions あり（確率は表に無い）、1 ゲームの上限 108K フレーム、ライフ区切りなし。結論は、明示的な探索の部品が無いので難探索で苦戦すると書く [Clark BTR (2024/11), §4.1、§7、脚注 1、表 A1、表 D4、表 D6]。

### 6〜8. 批判と懸念、総合評価、調査の限界

統計量が報告ごとに違い、5 本の平均は Run ごとの有無を隠し、評価の局の手順が書かれていないことを挙げる。Taïga らの図の値は PDF の画像を拡大して目で読み、0 に留まった本数は平均と合うことからの読みである。Dopamine の JSON は 100K 字ずつ読み、Rainbow の本数は 2 回目の問い合わせで 5 本とし、C51 は食い違ったので載せていない。Dopamine の設定は 2026-11 の master を読んだ。

## 主要な概念

- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
- [Montezuma's Revenge](../../../concepts/subjects/montezuma_revenge.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [NoisyNetの観測](../../../concepts/methods/noisynet.md)
- [自己模倣による価値学習](../../../concepts/methods/self_imitation.md)
- [Atari横断](../../../concepts/subjects/atari.md)
