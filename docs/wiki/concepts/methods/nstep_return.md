---
title: n-step の収益
page_kind: 手法
aliases:
  - n-step
  - n_step
  - nstep
  - nstep3
  - n10
  - 多段の目標
  - 多段の更新
  - λ 収益
  - Retrace
  - Peng's Q(λ)
  - MMC
description: Montezuma で n-step を 3 から 10 にした 2 本は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M（n-step 3 の 2 本は 26.2M・18.2M）で、外的の Q の 1 位と 2 位の差は 4M 窓 8 つのうち 5 つで大きかった。2 組は実行体の build も違う。
updated: 2026-11-27
---
# n-step の収益

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [手法](../../index.md#手法)

## 今の結論

Montezuma で n-step を 3 から 10 にした 2 本は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M（n-step 3 の 2 本は 26.2M・18.2M）で、外的の Q の 1 位と 2 位の差は 4M 窓 8 つのうち 5 つで大きかった。2 組は実行体の build も違う。
DropMerge では、ε ラダー基準の上で n-step 3 にすると長考型 NEET に入り、評価専用 Run で −10〜−12% だった。
外の DQN 系では、Rainbow が多段の目標の利点を報酬を手前の状態へ速く伝えることとし、Mountain Car の DQN では n を大きくすると序盤の成績が上がった。n ごとに action gap を測った報告は、2026-11-27 の調査では見つからなかった。
補正なしの n-step は分散が小さい代わりに固定点が偏り、20 状態の鎖の MDP では、環境の step が増えると偏りの小さい手法に追い越された。

## 分かっていること

### anet の知見

- n-step 3について: ε ラダー基準の上で長考型NEETに入り、評価専用Runで−10〜−12%（[Atari からのフィードバック](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md) 探索ブロック07）（出典: [2026-10-01_atari-feedback.md](../../../experiments/default-dqn/dropmerge/2026-10-01_atari-feedback.md)、現時点のまとめ・対応する探索ブロック）。
- Montezuma の割引率の分離の形で replay ratio を 8 から 4 にした 2 本（g99ne_rr4）の、n-step を 3 から 10 にした 2 本（seed 1・2、2026-11-21 の build）は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M で、そのあと 8 本以上だった回は 18 回中 18 回・16 回中 15 回（n-step 3 は 4 回中 4 回・12 回中 9 回）、最後の 5 回は 80 本とも倒した。学習側の 1M 窓で倒すゲームが 1 割以上になった最初の窓は 12M 台・11M 台（n-step 3 は 14M 台・18M 台）で、剣を取るゲームが 1 割以上になった窓からの間は 4M・3M（同 6M・7M）だった（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ37・探索ブロック23。Run: run_20261121-211321_mz_g99ne_n10_rr4_s1、run_20261122-030405_mz_g99ne_n10_rr4_s2）。
- Greedy の外的の Q の 1 位と 2 位の差（`43_greedy_ext_gap`）の 4M 窓の平均は、8 つの窓のうち 5 つで n-step 10 の 2 本がどちらも n-step 3 の 2 本より大きく（24〜28M は 0.0065・0.0047 と 0.0009・0.0019）、0〜4M と 12〜20M の 3 つの窓は値の範囲が重なった。2 組は実行体の build も違う（出典: [Montezuma's Revenge の RND（Taiga ら寄せ）の記録](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ39・探索ブロック23。Run: run_20261121-211321_mz_g99ne_n10_rr4_s1、run_20261122-030405_mz_g99ne_n10_rr4_s2、run_20261119-144336_mz_tg_g99ne_rr4_s1、run_20261119-202718_mz_tg_g99ne_rr4_s2）。

### 外の事例

- DQN 系の制御で n ごとに action gap を測った実験、時間の刻み（frame skip）と n-step と action gap を結ぶ理論、Advantage Learning・DAU・Munchausen と n-step を組み合わせて action gap や Greedy の止まりを比べた実験は、2026-11-27 の調査では見つからなかった（探した範囲: Rainbow、Mountain Car の DQN、補正なしの n-step の理論、policy churn、Munchausen、Episodic Backward Update、PixelCNN の論文と web の検索）（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、対象の問いと結論・§3.5・§7）。
- 補正なしの n-step（重点重みを掛けない n-step の Q 学習）は、分散が小さい代わりに固定点が偏り、固定点は 1 step を目標の方策 π、続く n−1 step を行動の方策 μ で進むことを繰り返す非定常な方策の価値になる。20 状態の鎖の MDP では、環境の step が少ないうちは n の大きい補正なしの更新が最もよく、step が増えると偏りの小さい手法に追い越された。Atari-57 の DQN 系の比較では、n-step の対照は n = 5 が最もよかった（C-trace は n = 10、3 seed）（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§2.1）。
- policy churn の論文は、補正なしの多段の更新が意外によく効くことを churn で説明した。Greedy の方策が速く変わるので、補正のある方法では数回の更新の遅れでも軌跡の打ち切りが大きくなり、補正を省く損より得が大きくなりうる、という議論で、実験ではない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§2.2）。
- 軌跡を切らない Peng's Q(λ) は、行動の方策が Greedy の方策をゆっくり追うなら最適な方策に収束する（Kozuno ら。抄録で確かめた）（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§2.2）。
- n-step の収益の分散は n とともに増え、TD 誤差に相関が無ければ n に線形、完全に相関すれば 2 乗で増える。同じ縮小率の複合の収益（λ 収益など）は n-step より分散が大きくならず、MinAtar の 5 ゲーム（32 試行）の多段の DQN では、n = 5 の比較で 5 ゲーム中 3 ゲーム、n = 3 で 1 ゲームで複合の収益が n-step を上回った。action gap には触れていない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§2.3）。
- Rainbow は多段の目標の利点を、新しく観測した報酬を先に訪れた状態へ速く伝えることとし、n は 3 だった。調整では n = 1・3・5 を比べ、3 と 5 は序盤がよく、最後は 3 が最もよかった。多段の目標を外すと序盤も最終の成績も下がり、57 ゲーム中 53 ゲームで完全な Agent を下回った（約 200M フレーム）。action gap は測っていない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.1）。
- DQN に似た構成の Mountain Car（γ = 1、ε 0.1 の固定の方策、各 100 回の独立な Run）で n ∈ {1, 3, 5, 10, 20} を比べると、n = 20 が序盤と全体の成績で最もよく、最終の成績は n ≥ 3 で同じくらいで、成績を損なうほど大きい n は見つからなかった。著者は効果を偏りと分散の釣り合いで説明し、報酬の伝わる速さと action gap は測っていない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.2・§5）。
- 疎な報酬の 10×10 の MNIST の迷路（ゴール +1000、壁 −1、壁の密度ごとに 50 の迷路、100,000 step のあと）では、最短路に対する経路の長さの平均が、壁 20% で 1-step DQN 14.40・n-step DQN（n はエピソードの長さ）3.26、壁 50% で 22.36・11.32 だった（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.3）。
- Episodic Backward Update の論文は、一様に引く 1-step の更新が疎で遅れた報酬では非効率だとする。同じ論文の Atari の比較（10M フレーム）は n-step を含まない（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.3）。
- PixelCNN の論文は、Monte Carlo 収益を混ぜた更新（MMC）を、一過性のボーナスを速く伝える多段の方法と位置づけたが、MMC は Bank Heist と Time Pilot では改善の大半を説明し、Ms. Pac-Man と Breakout では成績を下げた。著者は、Monte Carlo 収益は方策に依存して分散を増やし、off-policy の学習で最適な方策への収束を妨げうるとする（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.4）。
- PixelCNN の論文の著者は、Retrace(λ) を使う Reactor では、方策が外れた軌跡を慎重に切る性質がボーナスの効きを抑えたと推測している（出典: [nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.4）。

## 横断的知見

- 補完: Montezuma で n-step を 3 から 10 にした 2 本は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M（n-step 3 の 2 本は 26.2M・18.2M）で、Greedy の外的の Q の 1 位と 2 位の差は 4M 窓 8 つのうち 5 つで大きかった。外の DQN 系の報告も、多段の目標の利点を報酬を手前の状態へ速く伝えることとし（Rainbow）、n を大きくすると序盤の成績が上がった（Mountain Car の DQN）が、n ごとに action gap を測った報告は見つからなかった。条件の違い: anet は価値の分解・割引率の分離・損失を合わせた形の Run で、n-step 10 と 3 は実行体の build も違う。Rainbow は 57 ゲームで多段の目標の有無を比べ、Mountain Car は γ = 1・各 100 回の Run である（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 37・39；[nstep_action_gap_survey_2026-11-27.md](../../../../reports/nstep_action_gap_survey_2026-11-27.md)、§3.1・§3.2・§3.5）。

## 見立て

- Montezuma の Greedy が部屋 2 とその梯子の下で止まり、部屋 2 の敵を剣で倒さずに終わるのは、選ぶ行動と進む行動の外的の Q の差が、1 step 遅れたときの差（(1−γ)·V）より小さいためではないか。n-step を 3 から 10 にすると外的の Q の行動間の差が広がり、その局面を抜けて剣で倒すまでが早まったのではないか。 状態: 未検証
  - 支える事実: (1) r123 の exp step 29,892,992 の重みの EvalPanel（Greedy、sticky 0）は、部屋 2 の梯子の一番下で LEFTFIRE を繰り返して止まり、Q は LEFTFIRE 2.027・DOWN 2.024（差 0.003。γ 0.997 で 1 step 遅れたときの差は約 0.006）だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 14・探索ブロック 11）。 (2) b001_s2 の重みは、500 点（扉と剣）になった ep_step 246〜254 に梯子の下で DOWN を押し続け、Q は 18 行動とも 1.654〜1.663（DOWN 1.663、UP 1.659。1 step 遅れたときの差は約 0.005）で、手で梯子の上へ動かすと ep_step 704 に敵を倒して 2500 点になった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 10）。 (3) 割引率の分離の形・replay ratio 4 で n-step を 10 にした 2 本は、Greedy で剣で敵を倒すゲームが 16 本中 8 本以上になった最初の回が 12.2M・14.2M（n-step 3 の 2 本は 26.2M・18.2M）、最後の 5 回の倒した本数が 80/80・80/80（同 55/80・61/80）だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 37・探索ブロック 23）。 (4) 同じ 4 本の Greedy の外的の Q の 1 位と 2 位の差（`43_greedy_ext_gap`）は、4M 窓 8 つのうち 5 つで n-step 10 の 2 本がどちらも大きく、24〜28M は 0.0065・0.0047 と 0.0009・0.0019 だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 39・探索ブロック 23）。 (5) 学習側の 1M 窓で、剣を取ったゲームが 1 割以上になった窓から剣で倒したゲームが 1 割以上になった窓までは、n-step 10 が 4M・3M、n-step 3 が 6M・7M だった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 37・探索ブロック 23）。 (6) 外の事例: n を下げていくスケジュールの理由として BBF が挙げる Kearns と Singh の誤差上界は、大きい k は収束が速いが漸近誤差が大きいとする。対象は方策評価の phased TD である（出典: [reward_scale_discount_survey_2026-09-23.md](../../../../reports/reward_scale_discount_survey_2026-09-23.md)、5.6 スケジュールの理論的な説明）。
  - 合わない事実: (1) 外的の差は、0〜4M と 12〜20M の 3 つの窓では n-step 10 と 3 の値の範囲が重なった（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、現時点のまとめ 39）。 (2) 価値の分解で割引率を共有した vd の 30M の重みの Greedy（β 1）は 16 本とも 400 点・部屋 2 で、選んだ行動が外的の Q の 1 位と違う割合は 0.982、外的の差 0.0009、内的の差 0.0013 だった。部屋 2 の止まりに内的の Q が効いた Run がある（出典: [2026-11-08_montezuma-rnd-taiga-aligned.md](../../../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、探索ブロック 19）。
  - 条件の違い: EvalPanel の 2 件は、価値の分解なし・n-step 3・sticky 0 の Run の 1 ゲームの局面である。n-step の比較は、価値の分解・割引率の分離・損失を合わせた形の Run の Greedy（sticky 0.25）と学習側の窓の値で、部屋 2 の局面に絞った値ではない。n-step 10 の 2 本は実行体の build が n-step 3 の 2 本と違い（2026-11-21 と 2026-11-19）、各 2 seed である。外の事例は深層 RL の制御ではない。
  - 確かめ方: n-step 3 と 10 の 4 本の重み（剣で倒すまでの差が開いた 12〜18M 付近）を Evaluation Trace つきの Greedy で再生し、剣を取ってから部屋 2 の敵を倒すまでの局面で、`extrinsic_q_gap` と (1−γ)·V（外的の V）を並べる。n-step 10 でその局面の差が 1 step 遅れの差を上回っていれば読みと合う（評価だけの新しい Run が要る）。
  - 出どころ: AI（開発者の質問を起点、2026-11-21）

## 分かっていないこと

なし

## 撤回・更新

なし

## 関連

- [DropMerge](../subjects/dropmerge.md)。
- [Montezuma's Revenge](../subjects/montezuma_revenge.md)。
- [報酬尺度と割引率](reward_scale_discount.md)（NEET と Greedy の止まりの見立て）。
- [訪問回数と擬似カウントによる探索](pseudo_counts.md)。
- [Munchausen RL](munchausen.md)（M-IQN と多段の収益の組み合わせ）。
- [行動の時間尺度](temporal_abstraction.md)（action gap を広げる作用素、UTE の n-step）。
