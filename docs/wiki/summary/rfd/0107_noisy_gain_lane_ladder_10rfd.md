# Train Actor の lane 別 NoisyNet 倍率

## 元の材料

[RFD0107　Train Actor の lane 別 NoisyNet 倍率](../../../memo/0107_noisy_gain_lane_ladder_10rfd.md) blob:38f4b56736df30aac2d34bebfcb08712950a7077

## 要点

- lane ごとに NoisyNet の倍率を変える方式と、その比較方法は、再グリルで見直す暫定案である。
- Qbert の greedy 評価では同じ得点で終わる経路を繰り返し、学習中の探索でその経路を十分に経験していない可能性がある。
- 暫定案では学習する σ に別の非負の倍率を掛け、行動を集める Actor の lane 間で強さを変える。
- 倍率0ではノイズのない出力になり、1では従来の出力になる。
- 共通の倍率または両端の倍率を ProfiledValue で指定し、負数や非有限値を設定時に退ける。
- スキップなしの条件で比較し、評価時に繰り返した失敗が減るかを見る。

## 内容

### 目的と問題

見立て: 学習のデータは、ノイズで揺れる Train Actor が集めるので、Greedy がたどる道の状態は replay に少ない。その道の上の誤った判断は、経験されず、直す勾配がほとんど来ない。学習の lane の一部を、ノイズの弱い（または無い）動き方にすれば、Greedy に近い道のデータが replay に入り、その道の上の誤りが経験されて直る見込みがある。

### 起草時点の現状

Learner の forward は `learner.current`・`learner.target` の NN 実行設定で動き、Actor の設定とは別である。
ε のラダー: qbert で ε のラダーを 1e-4〜0.05 へ下げて効いたのは、低 ε の lane の割合で、同じ ε での群前ではなかった（Qbert の「分かっていること」。出典は 2026-09-25 の実験記録 の現時点のまとめ 12）。
NoisyNet と ε のラダー: Atari の実時間優先 50M で ε のラダーを NoisyNet に置き換えると 7 ゲームとも基準を下回り、Breakout の 2×2 では NoisyNet の有無の差がラダーありで +15.3、なしで +0.8 だった（NoisyNet の観測。出典は 2026-10-11 の実験記録）。lane ごとに探索の強さを散らすことが効いている可能性を示すが、NoisyNet の強さのラダーを測った結果は無い。

### 候補を採る場合のゴール・非ゴール・仮説と成功の測り方

既定の変更（BTR に寄せた構成の既定を、ラダーありにするなど）。消える案はその変更で、本当の制約ではないが、ステージ 1 の結果が無いと決められない。
仮説: Greedy の決まった失敗は、Greedy の道の状態が replay に少ないために直らない。ノイズの弱い lane を混ぜれば、その道のデータが入り、決まった失敗が減る。
比べ方: ステージ 1 で、スキップなしの構成（`run_20261025-231207_a5_qb_aln_btreps_s1` と同じ）にノイズのラダーを足した qbert 50M を 1 本流し、スキップなしの seed 1（`run_20261025-231207_a5_qb_aln_btreps_s1`）・seed 2（`run_20261028-132638_a5_qb_aln_btreps_s2`）と比べる。

### 候補を採る場合の実装契約

`spatial.scale = log` の扱いは ε と同じ（`max(端, 1e-4)` の列を指定端へ伸縮し、0 の端は 0 のまま）。
NoisyNet の部品は、ノイズを引く準備のとき、呼び出し情報に倍率があれば、ノイズの項に行の倍率を掛ける。無ければ今と同じ（倍率 1）。IQN の τ の行など、1 つの lane が複数の行になる forward では、その lane の倍率を各行に当てる（行対応はエピソード開始のマスクと同じ扱い）。
Learner の forward（更新の forward と、`47_agent_noisy` の probe）には倍率を載せない（非ゴール）。
変えないもの: 過去の Run の記録、実験記録、Learner の forward、評価の Actor の動き方。

### 候補を採る場合の検証と受入

見込みありの読み: 同じ得点で終わる道の集中が、スキップなしの seed 1・seed 2 のどちらよりも小さく、全体の 8 面突破率が下がっていない。1 seed なので合否ではない。
ラダー: 両端 1・0、linear、4 lane で、lane の倍率が 1、2/3、1/3、0 になる。1 lane なら first。log の 0 の端は 0 のまま。
検証: 負・NaN・∞ は、使う側でも使わない側でも、定数でも段の値でも、設定を読む時点でエラー。使わない側に 1 以外の値を書いても WARN は出ず、使う側の値で動く。`mu_only` の Actor に 1 以外の倍率を書いても構築でき、WARN は出ず、forward の値は倍率 1 のときと一致する。
`skip.policy.*` に倍率のキーを書かない（たとえば `skip.policy.tau_rule.sample_mode` だけを書く）とき、skip 側の forward には、主の方策から写った倍率（`policy.spatial.noisy_gain.*` のラダー）が載る。

## 主要な概念

- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [Qbert](../../concepts/subjects/qbert.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
