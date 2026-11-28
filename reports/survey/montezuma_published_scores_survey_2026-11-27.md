# Survey: Montezuma's Revenge の公開の成績と、その測り方

- 日付: 2026-11-27
- 範囲: Taïga らの Rainbow 上の探索ボーナスの比較（ICLR 2020）の値と測り方、RND 原典（Burda ら）の値と測り方、Dopamine の既定の設定と公開の基準値、SAIL の IQN の設定、BTR の Montezuma's Revenge の値と条件。一次の資料（論文の本文・表・図・付録、著者の公開の設定とコード、公式の基準値のデータ）だけを読んだ。二次の資料は使っていない。
- 読み方: 逐語の引用は 1 か所（15 語未満）にとどめ、ほかは要約と位置（節・表・図・頁・設定のキー）で指す。図から目で読んだ値には「図を読んだ値」と書く。「（読み）」と書いた文は、この報告の推論で、原典の記述ではない。すでに [内的報酬の手法の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md)（§2.3・§6.1・§6.3）と [自己模倣を価値ベースへ広げた手法のサーベイ](sil_value_based_lower_bound_survey_2026-11-24.md)（§3）が書いた値は、調べ直さずに指す。

## 目次

1. [対象の問いと結論](#1-対象の問いと結論)
2. [Taïga らの比較の値と測り方](#2-taïga-らの比較の値と測り方)
3. [RND 原典の値と測り方](#3-rnd-原典の値と測り方)
4. [内的報酬を足さない学習器の点](#4-内的報酬を足さない学習器の点)
5. [BTR の Montezuma の値と条件](#5-btr-の-montezuma-の値と条件)
6. [批判と懸念](#6-批判と懸念)
7. [総合評価](#7-総合評価)
8. [調査の限界](#8-調査の限界)
9. [出典一覧](#9-出典一覧)

## 1. 対象の問いと結論

### 1.1 対象の問い

- Q16 の外の部分: Taïga らの「On Bonus Based Exploration Methods in the Arcade Learning Environment」（Dopamine の Rainbow に RND などを足した比較）の、Montezuma's Revenge の値（表の 200M フレームの値、学習曲線）は、学習の最後の評価か、学習中の最良か。評価の方策（ε、NoisyNets の扱い）、sticky actions、1 回の評価のゲーム数・フレーム数、seed の数は。RND 原典（Burda ら）の Montezuma の値の測り方（学習器、フレーム数、並列の環境の数、統計量）も、同じ欄で並べる。
- Q17 の外の部分: 内的報酬を足さない学習器の Montezuma's Revenge の点。Taïga らの ε-greedy の Rainbow と NoisyNets の Rainbow は、学習のどの時点で点を取り始めたか（曲線から読めること）。Dopamine の Rainbow と IQN の設定の差（n-step、PER、ε、更新の間隔など）で、Rainbow に点が出て IQN や DQN が 0 に近い理由は、何と説明されているか（説明が無ければ、無いと書く）。BTR（Beyond The Rainbow）の論文の Montezuma's Revenge の値と条件は。

### 1.2 結論

Q16:
- Taïga らの表 2〜5 の表題は「N フレームの学習のあと」の値（10M・50M・100M・200M）で、学習中の最良とは書いていない。図 1 は 5 seed の平均の曲線で、付録の図 7 は Run ごとの曲線と読める。本文は、評価の方策、1 回の評価のゲーム数、曲線と表が学習のゲームか評価のゲームかを書いていない（§2.2）。
- 公開の設定は、1 iteration が学習 250,000 agent step と評価 125,000 agent step、200 iteration、評価の ε 0.001、1 ゲームの上限 27,000 agent step、sticky actions あり。NoisyNets の版は ε を使わず、雑音の層に評価のときに雑音を切る引数は無い（§2.3）。
- RND 原典の表 1（Montezuma の RND 8,152）は最終の平均で、学習器は PPO。多くの実験は 128 env・1.97B フレーム、同じ比較の図 7 は 3 seed。別の評価の手順は書いていない。並列の env を増やした RNN の方策は、1024 env で学習の終わりの平均 10,070 だった（§3）。

Q17:
- Taïga らの付録の図 7（図を読んだ値）では、ε-greedy で点を取ったのは 5 本のうち 2 本で、20M フレーム前後から点が出た。NoisyNets の線はどれも 10〜20M フレームで 0 を離れ、30〜40M フレームまでに約 2,500 に達した（表の平均から 5 本とも達したと読める）（§4.1）。
- Dopamine の公開の基準値では、Rainbow と IQN は 5 本のうち各 1 本だけが 2500.0 に達し、DQN の 5 本は 198 回目の iteration がどれも 0.0 だった。既定の設定の差は、分布の形（C51 と IQN）・PER の有無・学習率で、n-step 3・ε・更新の間隔は同じである。SAIL の表で 0 の IQN は n-step を使わない設定である（§4.2・§4.3）。
- Rainbow に点が出て IQN・DQN が 0 に近い理由を説明した文は、読んだ範囲に無かった（§4.4）。
- BTR の表 A1 の Montezuma は BTR 0（学習中の最大、100 ゲームの平均、200M フレーム）で、公開の CSV の 200 回の評価はすべて 0.0 だった（§5）。

### 1.3 見つからなかったこと

- Taïga らの曲線と表が、学習の局と評価の局のどちらの値か。表の値が 1 iteration の値か、複数の iteration の平均か。本文・付録 A〜C、公開の設定（`rainbow_rnd.gin`・`noisy_rainbow.gin`）、実行のコード（`run_experiment.py`）を読んだが、書いていない。
- Taïga らの NoisyNets の版の評価の ε の値。`noisy_rainbow.gin` は指定せず、`NoisyRainbowAgent` の評価の ε の既定は読んでいない。
- RND 原典の表 1 が、何 env・何フレーム・何 seed の値か。本文 §3・§3.4・§3.6、表 1・表 2・表 6 を読んだが、表 1 には無い。
- 素の Rainbow に点が出て、IQN・DQN が 0 に近い理由の説明。Taïga らの本文と結論、SAIL の §4.3・§4.4、Dopamine の論文と基準値の README を読んだが、無い。
- BTR の表 A1 の Rainbow 500 の出どころ。表題と図 1 の説明を読んだが、表 A1 には書いていない。
- Dopamine の基準値のデータが、学習の局と評価の局のどちらの値か。データのファイルと基準値の README を読んだが、書いていない。

## 2. Taïga らの比較の値と測り方

### 2.1 版と実験の条件

arXiv 2109.11052 は v1（2021-09-22）だけで、注記は 1908.02388 の完全版とし、雑誌の欄は ICLR 2020 の会議論文とする。PDF の各頁にも ICLR 2020 の会議論文と印字されている [Taïga Bonus-ALE (2021/09), arXiv の abstract の頁・PDF の頁頭]。

学習器は Dopamine の Rainbow（n-step、PER、分布型）で、学習器のハイパーパラメータは手法の間で固定した。学習は 200M フレームで、sticky actions（0.25）を学習と評価の両方に使い、mixed Monte-Carlo return は使わない [Taïga Bonus-ALE (2021/09), §3.1]。

付録 A.1 の Rainbow の値は、割引率 0.99、学習の開始 80K フレーム、target の更新 32K フレーム、Adam の学習率 6.25×10^-5・ε 1.5×10^-4、n-step 3、分布の atom 51、値の範囲 [−10, 10] である。ライフ区切りは切ってある。NoisyNets 以外の手法は ε-greedy で、ε を 1M フレームで 1 から 0.01 へ下げる [Taïga Bonus-ALE (2021/09), 付録 A.1]。

ボーナスの手法は、Montezuma の 200M フレームの最終性能（5 本の平均）で調整し、同じくらいの設定なら早く届いたほうを選んだ [Taïga Bonus-ALE (2021/09), 付録 A]。NoisyNets は調整せず、σ0 0.5 のままである [Taïga Bonus-ALE (2021/09), 付録 A.2.1]。

### 2.2 表と図の統計量

付録の表 2〜5 は、表題で、各時点の学習のあとの生の点だとしている。200M フレームの表題は次のとおりである。

```
Raw scores for all games after 200M training frames

200M フレームの学習のあとの、全ゲームの生の点
```

[Taïga Bonus-ALE (2021/09), 表 5（PDF p.24）]

表 2〜5 の表題はそれぞれ 10M・50M・100M・200M フレームで、どの表題にも、学習中の最良や評価の局の値とは書いていない。Montezuma の行は次のとおりである（PDF の表の画像で値を確かめた）[Taïga Bonus-ALE (2021/09), 表 2〜5（PDF p.21〜24）]。

| 表（時点） | ε-greedy | RND | CTS | PixelCNN | ICM | NoisyNets |
|---|---|---|---|---|---|---|
| 表 2（10M フレーム） | 0.5 | 1.5 | 231.8 | 17.8 | 0.0 | 13.2 |
| 表 3（50M フレーム） | 553.3 | 2789.6 | 2550.4 | 2528.5 | 1160.7 | 2492.9 |
| 表 4（100M フレーム） | 780.0 | 5188.8 | 5118.8 | 4364.6 | 2449.8 | 2486.2 |
| 表 5（200M フレーム） | 780.0 | 5199.2 | 5028.2 | 4652.3 | 2795.9 | 2435.4 |

図 1 は、縦軸が Average Score、横軸が百万フレームの曲線で、本文は 5 seed の平均だとする [Taïga Bonus-ALE (2021/09), §3.2、図 1（PDF p.4）]。付録 C は、Montezuma の報酬が階段状で分散が大きいので、すべての曲線を図 7 に載せると書く [Taïga Bonus-ALE (2021/09), 付録 C]。図 7（PDF p.13）は手法ごとに 1 枚で、薄い線と濃い線を描く。薄い線は Run ごと、濃い線は 5 本の平均と読める（読み。図の説明は「Training curves」だけで、線の意味を書いていない）。ar5iv の HTML 版では、この図に「図 9」の番号が付く。

RND の図 7 の線は、90M フレームのあとはほぼ平らで、約 5,700 の帯と約 4,550 の帯に分かれた（図を読んだ値。線が重なり、帯ごとの本数は数えにくい）。表 4 と表 5 の RND は 5188.8 と 5199.2 で、この手法の表では、最後の値と学習中の最良の値はほとんど違わないと読める（読み）。

本文が書いていないこと: 評価の局の ε、1 回の評価のゲーム数とフレーム数、曲線と表が学習のゲームか評価のゲームか、表の値が 1 iteration の値か複数の iteration の平均か。本文に「evaluation phase」や「iteration」の定義は出てこない（PDF の本文と付録を検索した）。

### 2.3 公開の設定とコード

著者の公開の設定 `rainbow_rnd.gin` は、Montezuma の RND の Rainbow を次の値で走らせる [Taïga code (2026/11), `bonus_based_exploration/configs/rainbow_rnd.gin`]。

- `Runner.num_iterations = 200`、`Runner.training_steps = 250000`（agent step）、`Runner.evaluation_steps = 125000`（agent step）、`Runner.max_steps_per_episode = 27000`（agent step）。
- 学習の ε 0.01（`epsilon_decay_period = 250000` agent step で下げる）、評価の ε 0.001（`epsilon_eval`）。
- `update_horizon = 3`、`replay_scheme = 'prioritized'`、`update_period = 4`、`target_update_period = 8000`、`min_replay_history = 20000`、replay 1,000,000、batch 32、学習率 0.0000625。
- 内的報酬の倍率 `RNDIntrinsicReward.reward_scale = 0.00005`、`clip_reward = False`。
- `sticky_actions = True`（コメントは確率 0.25）、`terminal_on_life_loss = False`。

実行のコード `run_experiment.py` の既定の schedule は `continuous_train_and_eval` で、Dopamine の `Runner`（学習と評価を交互に行う）を作る。schedule を `continuous_train` にすると学習だけの `TrainRunner` になる。Dopamine の Runner を上書きするコードは無い [Taïga code (2026/11), `bonus_based_exploration/run_experiment.py`]。

今の Dopamine の Runner は、局（phase）を、step の数が下限に届いた最初のゲームの終わりで閉じ、局の値を、その局で終わったゲームの点の和をゲームの数で割った平均にする。学習の局は `train_average_return`、評価の局は `eval_average_return` に記録する [Dopamine repo (2026/11), `dopamine/discrete_domains/run_experiment.py`]。Dopamine の論文は、`train` と `train_and_eval` の 2 つの schedule を挙げ、Machado らが学習中の曲線の報告を勧めること、学習と評価の点の報告にほとんど差が無いことを書く [Castro Dopamine (2018/12), §4.2]。

ここから、Taïga らの既定の実行では、評価の局は 1 iteration あたり 125,000 agent step 以上のゲーム（上限が 27,000 agent step なので 5 ゲーム以上）を ε 0.001 で流し、その平均を記録すると読める（読み）。論文の図と表が、この評価の局の値か学習の局の値かは分からない（§1.3）。

NoisyNets の版 `noisy_rainbow.gin` は、ε の設定を持たない。`NoisyRainbowAgent` の ε の関数の既定は常に 0 を返し、雑音の全結合層（`noisy_dqn_agent.py` の `fully_connected`）は、平均の重みに雑音を掛けて足す形で、評価のときに雑音を切る引数を持たない [Taïga code (2026/11), `configs/noisy_rainbow.gin`・`noisy_networks/noisy_rainbow_agent.py`・`noisy_networks/noisy_dqn_agent.py`]。Dopamine の TF 版の DQNAgent は、評価のときは `epsilon_eval`（既定 0.001）、学習のときは ε の関数で ε を選ぶ [Dopamine repo (2026/11), `dopamine/tf/agents/dqn/dqn_agent.py`]。公開のコードでは、NoisyNets の版は評価のときも雑音の層をそのまま使うと読める（読み）。論文は NoisyNets の評価の扱いを書いていない。

## 3. RND 原典の値と測り方

### 3.1 値と条件

RND 原典の学習器は PPO である。多くの実験は、128 env で 1 env あたり長さ 128 の rollout を 30K 回、合計 1.97B フレームで走らせた [Burda RND (2018/10), §3]。Atari の設定は sticky actions 0.25、1 ゲーム 18K フレームまで、ライフ区切りなしである [Burda RND (2018/10), 表 2]。

難探索の 6 本の比較（表 1）は、表題で最終の平均の成績だとしていて、Montezuma は RND 8,152、PPO 2,497、ランダム特徴の順モデル 400 である（値は [内的報酬の手法の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md) §2.3 と同じ）[Burda RND (2018/10), §3.6、表 1]。同じ比較の曲線の図 7 は、図の説明で RNN の方策の平均のエピソードの収益とし、表 6 は図 7 の seed を 3 とする（図 2 と図 5 は 10、ほかは 5）[Burda RND (2018/10), 図 7、表 6]。表 1 が何 env・何フレーム・何 seed の値かは、表 1 にも §3.6 にも書いていない。

並列の env の数を変えた実験（§3.4）では、32 env の RNN の方策を長く走らせて 1.6B フレームで平均 7,570 に届き、そのうち 1 本は 24 部屋すべてに入って最良 17,500 だった。1024 env の RNN の方策は学習の終わりの平均が 10,070 で、1 本は平均 14,415 だった [Burda RND (2018/10), §3.4]。平均は学習の終わりの値、17,500 は 1 本の Run の最良のゲームの値である。

学習の rollout と別の評価の手順は、本文に出てこない。Algorithm 1 は rollout で方策から行動を抽出するので、報告の収益は確率的な方策の学習の rollout の値と読める（読み）。

### 3.2 測り方の並び

| 報告 | 学習器 | 長さ | 並列の env | seed | 統計量 | 評価の方策 | sticky actions |
|---|---|---|---|---|---|---|---|
| Taïga ら 表 5（RND 5199.2） | Dopamine の Rainbow＋RND | 200M フレーム | 1（Dopamine の Runner） | 5 | 200M フレームの学習のあとの値（5 本の平均と読める） | 本文に記述なし（公開の設定の評価の局は ε 0.001） | 0.25 |
| RND 原典 表 1（RND 8,152） | PPO＋RND | 多くの実験は 1.97B フレーム（表 1 の値には記述なし） | 多くの実験は 128（表 1 には記述なし） | 表 1 には記述なし（図 7 は 3） | 最終の平均 | 別の評価の記述なし（確率的な方策の rollout と読める） | 0.25 |
| RND 原典 §3.4（10,070） | PPO＋RND、RNN | 記述なし | 1024 | 記述なし | 学習の終わりの平均 | 同上 | 0.25 |
| SAIL §4.4（RND-IQN 161） | IQN（n-step なし）＋RND | 総量の記述なし（指標は 1M step ごとの 200 回の値） | 記述なし | 記述なし（部分集合の比較は 6、全ゲームは 3） | 最終の値（期間の平均は 45） | 記述なし | 0.25 |
| BTR 表 A1（BTR 0） | BTR（IQN・Munchausen・NoisyNets ほか） | 200M フレーム | 64 | 記述なし（図 1 の BTR は 4） | 学習中の最大（100 ゲームの平均） | ε 0.01、125M フレームから 0 | あり（確率は記述なし） |

出典: Taïga の行は §2、RND 原典の行は §3.1、SAIL の行は [Ferret SAIL (2020/12), §4.1・§4.3・§4.4] と [自己模倣を価値ベースへ広げた手法のサーベイ](sil_value_based_lower_bound_survey_2026-11-24.md) §3、BTR の行は §5。Taïga の行の並列の env の「1」は、Dopamine の Runner が 1 つの環境を回す作りからの読みである（読み）。

## 4. 内的報酬を足さない学習器の点

### 4.1 Taïga らの ε-greedy と NoisyNets の曲線

付録の図 7（PDF p.13、手法ごとに Run ごとの曲線と読める）を、PDF の画像を拡大して読んだ（図を読んだ値）[Taïga Bonus-ALE (2021/09), 図 7（PDF p.13）]。

- ε-greedy: 点を取ったのは 2 本だった。1 本は 20M フレーム前後に約 400、45M フレーム前後に約 2,500 になり、200M フレームまで約 2,500 に留まった。もう 1 本は 30M フレーム前後に約 400、70M フレーム前後に約 1,400 になり、そのまま留まった。ほかの 3 本は 200M フレームまで 0 に見える。濃い線（平均と読める）は 45M フレーム前後に約 600、70M フレーム前後から約 780 である。
- NoisyNets: 線はどれも 10〜20M フレームで 0 を離れ、30〜40M フレームまでに約 2,500 に達し、200M フレームまで約 2,500 の近くに留まった（ところどころ下へ振れる）。線は重なっていて本数を数えられないが、表の平均（50M〜200M フレームで 2435.4〜2492.9）から、5 本とも約 2,500 に達したと読める（読み）。

表の値（§2.2）では、ε-greedy の平均が 10M フレーム 0.5、50M フレーム 553.3、100M フレームと 200M フレーム 780.0、NoisyNets の平均が 10M フレーム 13.2、50M フレーム 2492.9、100M フレーム 2486.2、200M フレーム 2435.4 である [Taïga Bonus-ALE (2021/09), 表 2〜5]。2 本の約 2,500 と約 1,400 と 3 本の 0 を平均すると、表の 780.0 に合う（読み）。

本文は、ε-greedy が Montezuma で弱く苦戦すると書き、どの時点から点を取り始めたかは書いていない [Taïga Bonus-ALE (2021/09), §3.2]。

### 4.2 Dopamine の基準値の Rainbow・IQN・DQN

Dopamine は公式の基準値の曲線のデータを、ゲームごとの JSON で公開している。Montezuma の `montezumarevenge.json` は、記録ごとに `Iteration`・`Value`・`Agent` を持ち、Run の番号の欄は無い（`Iteration` が 0 へ戻るところで Run を分けて読んだ）。`Agent` は DQN、C51、Rainbow、IQN、Quantile (JAX)、DQN (Adam + MSE in JAX) で、`Iteration` は 0〜198 である [Dopamine baselines JSON (2026/11)]。Dopamine の論文は、各手法を 60 本のゲームで 5 本ずつ独立に走らせたとする [Castro Dopamine (2018/12), §3.3]。1 iteration は、Dopamine の既定の設定で学習 250,000 agent step である（§4.3）。

- Rainbow（5 本）: 1 本が、20 回目の iteration に 133.3、30 回目に 833.3、34 回目に 2218.2、40 回目と 198 回目に 2500.0 だった。ほかの 4 本は、10・20・30・34・40・50・100・198 回目がどれも 0.0 だった。
- IQN（5 本）: 1 本が、100 回目に 100.0、113 回目に 116.7、123 回目に 2325.0、124 回目と 198 回目に 2500.0 だった（150 回目は 2388.9）。ほかの 4 本は 198 回目が 0.0 だった。
- DQN（5 本）: 198 回目の iteration がどれも 0.0 だった。

データの値が、学習の局と評価の局のどちらの平均かは、データにも基準値の README にも書いていない [Dopamine baselines JSON (2026/11)；Dopamine repo (2026/11), `baselines/atari/README.md`]。値は、読み取りの道具で 100K 字ずつ読んだもので、Run の数は 2 回の問い合わせで確かめた（§8）。

### 4.3 Dopamine の Rainbow・IQN・DQN の設定の差と、SAIL の IQN

Dopamine の今の既定の設定（JAX 版の gin）を並べる [Dopamine repo (2026/11), `dopamine/jax/agents/rainbow/configs/rainbow.gin`・`implicit_quantile/configs/implicit_quantile.gin`・`dqn/configs/dqn.gin`]。

| 設定 | Rainbow | IQN | DQN |
|---|---|---|---|
| 価値の形 | 分布型（C51、atom 51、vmax 10） | 分位点（τ 64・64、選ぶときは 32） | 期待値 |
| n-step（`update_horizon`） | 3 | 3 | 1 |
| Replay の抽出 | PER（`prioritized`） | 一様（PER の指定なし） | 一様 |
| 学習率（Adam） | 0.0000625（eps 0.00015） | 0.00005（eps 0.0003125） | 6.25e-5（eps 1.5e-4） |
| 学習の ε・下げる期間 | 0.01・250,000 agent step | 同じ | 同じ |
| 評価の ε | 0.001 | 0.001 | 0.001 |
| 更新の間隔・target の更新 | 4・8,000 agent step | 同じ | 同じ |
| 学習の開始・replay・batch | 20,000 agent step・1,000,000・32 | 同じ | 同じ |
| 割引率・sticky actions・1 ゲームの上限 | 0.99・あり・27,000 agent step | 同じ | 同じ |

Dopamine の論文は、TF 版の付録の設定で、Rainbow は n-step 3・PER、IQN は n-step 3・一様抽出、DQN と C51 は n-step 1・一様抽出とし、IQN の設定には PER に対応していないという注がある。Dopamine の Rainbow は Double DQN・Dueling・NoisyNets を含まないとも書く [Castro Dopamine (2018/12), §3.1、付録 D]。Taïga らの `rainbow_rnd.gin` の Rainbow の値（§2.3）は、上の表の Dopamine の Rainbow の既定と同じで、冒頭のコメントは、Hessel らの値に従い sticky actions だけが違うとする [Taïga code (2026/11), `rainbow_rnd.gin`]。

SAIL は Dopamine の参照の設定を使い、IQN だけは IQN の原典（Dabney ら）の設定を使う。その設定は、Dopamine と違って n-step を使わないと書く [Ferret SAIL (2020/12), §4.1・§4.3]。SAIL の表で Montezuma が 0 の IQN（[自己模倣を価値ベースへ広げた手法のサーベイ](sil_value_based_lower_bound_survey_2026-11-24.md) §3）は、この n-step を使わない IQN である。SAIL の IQN が PER を使うかは、本文に書いていない。

### 4.4 理由の説明の有無

Rainbow に点が出て IQN・DQN が 0 に近い理由を、設定の差で説明した文は、読んだ範囲に無かった。近い記述は次のとおりである。

- Taïga らは、Rainbow は文献で使われてきた古い DQN 系より成績を上げると書く [Taïga Bonus-ALE (2021/09), §3.2]。要旨では、Montezuma の最近の伸びは探索の改良より Network の形の変化によるかもしれないと書く [Taïga Bonus-ALE (2021/09), 要旨]。
- Taïga らは結論で、ボーナスが効かない理由の 1 つとして、Rainbow の PER が、まれにしか起きない大事な遷移を多く引くので、ボーナスを余分にしている可能性を挙げる [Taïga Bonus-ALE (2021/09), §4]。これはボーナスの効き方の推測で、素の Rainbow の点の理由の説明ではない。
- SAIL は Montezuma を悪名高い難探索のゲームと書くだけで、IQN・DQN が 0 の理由を書かない。RND の設定は Rainbow に合わせて調整されたものだとは書く [Ferret SAIL (2020/12), §4.4]。
- Dopamine の論文と基準値の README に、Montezuma の点の説明は無い [Castro Dopamine (2018/12)；Dopamine repo (2026/11), `baselines/atari/README.md`]。

## 5. BTR の Montezuma の値と条件

BTR の論文の Montezuma の値は付録の表 A1 だけにあり、表 A2（Atari-5）と表 A3 には無い。表 A1 の表題は、学習中の最大の点（100 ゲームの平均）で、200M フレームの Atari-60 の値だとする [Clark BTR (2024/11), 表 A1]。Montezuma の行は次のとおりである。

| Random | Human | DQN (Nature) | Rainbow | M-IQN | FE-Rainbow | BTR |
|---|---|---|---|---|---|---|
| 0 | 4753 | 0 | 500 | 0 | 0 | 0 |

表題は、FE-Rainbow がライフの情報を使い、10M フレームだけ走らせたと注記する。DQN (Nature) と Rainbow の列の出どころは表 A1 に書いていない。図 1 の説明は、図 1 の DQN と Rainbow の結果を RLiable（Agarwal ら）から取ったとする [Clark BTR (2024/11), 表 A1、図 1]。

測り方の条件は次のとおりである [Clark BTR (2024/11), §4.1、脚注 1、表 D4、表 D6、図 1]。

- 評価は 1M フレームごとに 100 ゲームで、評価の ε は 125M フレームまで 0.01、そのあと 0。IQM の値は、学習中の最良の 1 回の評価を使うと脚注 1 に書く。
- 学習の ε は 1.0 から 0.01 へ 8M フレームで下げ、100M フレームで切る。NoisyNets の σ 0.5、n-step 3、割引率 0.997、64 env。
- sticky actions はあり（確率は表に書いていない）、1 ゲームの上限 108K フレーム、ライフ区切りなし。
- seed は、表 A1 の表題がランダムな seed とだけ書き、図 1 の BTR は 4 seed である。

BTR の公開の `results.csv` の MontezumaRevenge の行は、1M〜200M フレームの 200 回の評価がすべて 0.0 だった [BTR results.csv (2026/11)]。同じ CSV を集計した既存の報告は、Atari-5 以外のゲームの値を、値の刻みから 100 ゲームの 1 系列と推定している（[少数 seed での比較と報告のサーベイ](few_seed_comparison_reporting_survey_2026-11-04.md) §4.2。その報告の推定）。評価の前に NoisyNets の雑音を切ることは、既存の報告が BTR のコードで確かめている（[BTR のハイパーパラメータのサーベイ](btr_hyperparams_survey_2026-08-26.md)、評価対象ネットワークの節）。

BTR は結論で、明示的な探索の部品を足していないので Montezuma など難探索で苦戦すると書く（[内的報酬の手法の全体像のサーベイ](intrinsic_reward_landscape_survey_2026-11-18.md) §6.3 と同じ）[Clark BTR (2024/11), §7]。

## 6. 批判と懸念

- **報告ごとに統計量が違う。** Taïga らは各時点の学習のあとの値、RND 原典は最終の平均、SAIL は最終の値、BTR は学習中の最大である（§3.2）。BTR の公開の CSV を集計した既存の報告は、学習中の最大が最後の 10 回の平均をゲームの中央値で 1.15 倍上回るとする（[少数 seed での比較と報告のサーベイ](few_seed_comparison_reporting_survey_2026-11-04.md) §4）。
- **5 本の平均は、Run ごとの有無を隠す。** Taïga らの付録 C は Montezuma の報酬が階段状で分散が大きいとして 5 本ずつの曲線を載せた。ε-greedy の 780.0 は、点を取った 2 本と 0 の 3 本の平均と読める（§4.1、読み）。
- **評価の局の手順が書かれていない。** Taïga らの本文と、Dopamine の基準値のデータは、値が学習の局か評価の局かを書いていない。Dopamine の論文は両者にほとんど差が無いとするが、Montezuma について確かめた記述ではない（§2.3）。
- **ハイパーパラメータの調整の先。** Taïga らはボーナスの手法を Montezuma で調整していて、ε-greedy と NoisyNets の学習器の設定は Rainbow の既定のままである（§2.1）。

## 7. 総合評価

Q16 について、Taïga らの Montezuma の値は、表題では各時点の学習のあとの値で、学習中の最良とは書いていない。RND の曲線は 90M フレームのあと平らなので、表の 200M フレームの値と学習中の最良の値の差は小さいと読める。一方で、評価の方策と 1 回の評価のゲーム数は本文に無く、公開の設定（評価の局は ε 0.001・125,000 agent step 以上）から読めるだけである。RND 原典の 8,152 は PPO の最終の平均で、学習器も、フレーム数（1.97B フレーム）も、並列の env（128）も Taïga らと違う。

Q17 について、内的報酬を足さない学習器の Montezuma の点は、Run ごとに点が出るかどうかが分かれている。Taïga らの ε-greedy は 5 本中 2 本、Dopamine の基準値の Rainbow と IQN は 5 本中各 1 本が点を取り、DQN は 5 本とも 0.0 で、NoisyNets は 5 本とも約 2,500 に達したと読める。Dopamine の既定の Rainbow と IQN は n-step と ε と更新の間隔が同じで、PER と分布の形と学習率が違う。SAIL の論文は、0 の IQN を n-step を使わない設定と書く（公開の gin は n = 3 を指定していて、実行時の値は確かめていない）。この差で点の有無を説明した文献は見つからなかった。BTR は自分の論文で Montezuma が 0 である。

## 8. 調査の限界

- **図の値。** Taïga らの図 7 の値は、PDF の頁を画像にして拡大し、目で読んだ。線の重なりで、0 に留まった本数（3 本）は直接は数えられず、平均と合うことからの読みである。
- **Dopamine の基準値のデータ。** JSON（約 40 万字）は、web の読み取りの道具で 100K 字ずつ読んだ。Rainbow の本数は、1 回目の問い合わせで 4 本、`Iteration` 0 の記録を数えた 2 回目の問い合わせで 5 本と出て、5 本を採った（Dopamine の論文の 5 本と合う）。IQN の本数は 2 回とも 5 本だった。C51 の値は 2 回の問い合わせで食い違ったので、この報告に載せていない。手元に写して集計してはいない。
- **版。** Dopamine の設定とコードは、2026-11 の時点の master を読んだ。Taïga らが使った当時の Dopamine と同じかは確かめていない。Dopamine の論文の付録 D（TF 版の設定）は、今の JAX 版の gin と、IQN の n-step・一様抽出の点で同じだった。BTR は arXiv の v2（2025/05）の HTML を読んだ。
- **抽出の照合。** Taïga らの表 2〜5 の Montezuma の行は、PDF の表の画像で確かめた（PDF の文字の抽出は列がずれていた）。ほかの論文の本文と表は HTML 版を読み取りの道具で読み、PDF と照合していない。
- **書誌。** 被引用数は取っていない。

## 9. 出典一覧

[Taïga Bonus-ALE, 2021/09] Adrien Ali Taïga, William Fedus, Marlos C. Machado, Aaron Courville, Marc G. Bellemare. "On Bonus-Based Exploration Methods in the Arcade Learning Environment." ICLR 2020（読んだのは arXiv 2109.11052 v1 の PDF と ar5iv の HTML。注記は 1908.02388 の完全版）. https://arxiv.org/abs/2109.11052 （PDF: https://arxiv.org/pdf/2109.11052 、HTML: https://ar5iv.labs.arxiv.org/html/2109.11052 ）

[Taïga code, 2026/11] google-research. "bonus_based_exploration"（Taïga らの論文の公開実装。読んだのは 2026-11 の master の `configs/rainbow_rnd.gin`・`configs/noisy_rainbow.gin`・`run_experiment.py`・`noisy_networks/noisy_rainbow_agent.py`・`noisy_networks/noisy_dqn_agent.py`）. GitHub. https://github.com/google-research/google-research/tree/master/bonus_based_exploration （ファイルは https://raw.githubusercontent.com/google-research/google-research/master/bonus_based_exploration/ の下の同じパスで読んだ）

[Burda RND, 2018/10] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. "Exploration by Random Network Distillation." arXiv 1810.12894 v1（2018-10-30。読んだのは ar5iv の HTML）. https://ar5iv.labs.arxiv.org/html/1810.12894

[Castro Dopamine, 2018/12] Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, Marc G. Bellemare. "Dopamine: A Research Framework for Deep Reinforcement Learning." arXiv 1812.06110 v1（2018-12-14。読んだのは ar5iv の HTML）. https://ar5iv.labs.arxiv.org/html/1812.06110

[Dopamine repo, 2026/11] google. "Dopamine"（読んだのは 2026-11 の master の `dopamine/jax/agents/rainbow/configs/rainbow.gin`・`dopamine/jax/agents/implicit_quantile/configs/implicit_quantile.gin`・`dopamine/jax/agents/dqn/configs/dqn.gin`・`dopamine/discrete_domains/run_experiment.py`・`dopamine/tf/agents/dqn/dqn_agent.py`・`baselines/atari/README.md`）. GitHub. https://github.com/google/dopamine （gin とコードは https://raw.githubusercontent.com/google/dopamine/master/ の下の同じパスで読んだ。README は https://github.com/google/dopamine/tree/master/baselines/atari ）

[Dopamine baselines JSON, 2026/11] google. "Dopamine baselines: montezumarevenge.json"（Atari の基準値の曲線のデータ。読んだのは 2026-11 の master）. GitHub. https://raw.githubusercontent.com/google/dopamine/master/baselines/atari/data/montezumarevenge.json

[Ferret SAIL, 2020/12] Johan Ferret, Olivier Pietquin, Matthieu Geist. "Self-Imitation Advantage Learning." AAMAS 2021（arXiv の注記による。読んだのは arXiv 2012.11989 の ar5iv の HTML）. https://ar5iv.labs.arxiv.org/html/2012.11989

[Clark BTR, 2024/11] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC."（読んだのは arXiv v2（2025/05）の HTML）. https://arxiv.org/html/2411.03820

[BTR results.csv, 2026/11] Tyler Clark et al. "VIPTankz/BTR"（`results.csv`。読んだのは 2026-11 の main）. GitHub. https://raw.githubusercontent.com/VIPTankz/BTR/main/results.csv
