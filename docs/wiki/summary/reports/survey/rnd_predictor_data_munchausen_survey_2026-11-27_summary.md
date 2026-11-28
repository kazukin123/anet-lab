---
title: "Survey: RND の倍率の統計・predictor が学ぶデータ・Munchausen と内的報酬の組"
page_kind: サマリ
material: "[[reports/survey/rnd_predictor_data_munchausen_survey_2026-11-27.md|rnd_predictor_data_munchausen_survey_2026-11-27]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/reward_scale_discount.md|報酬尺度と割引率]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Survey: RND の倍率の統計・predictor が学ぶデータ・Munchausen と内的報酬の組

## 元の材料

[Survey: RND の倍率の統計・predictor が学ぶデータ・Munchausen と内的報酬の組](../../../../../reports/survey/rnd_predictor_data_munchausen_survey_2026-11-27.md) blob:a84122eda8446bca07b37277d50b7f4c723b920a

## 要点

- NGU は RND の倍率の誤差の平均と標準偏差を running の値とだけ書き、累積か直近の重みか、窓、序盤の扱いを書かない。Agent57 は倍率の作り方を NGU に譲り、MEME も running の平均と標準偏差とだけ書く [Badia NGU (2020/02), §2；Badia Agent57 (2020/03), §2；Kapturowski MEME (2022/09), 付録 G.1・G.2]。
- 読んだ公開の再実装 2 つ（DI-engine の NGU、deep_rl_zoo の NGU・Agent57。どちらも著者以外）は、件数で重みを付けた累積で統計を取り、序盤の待ちを置かず、統計の初めの分散を 1 にして最初の採点から割る（§3.2）。
- RND の原典の「predictor に使う経験の割合」0.25 は今の rollout の batch に掛ける dropout で、残す確率は 128 env で 0.25、256 env で 0.125、1,024 env で 0.03125 である。内的報酬が時間とともに下がる速さを env の数によらずそろえるためと書く [Burda RND (2018/10), 付録 A.4・表 5、§3.4]。
- NGU・Agent57・MEME は、優先度つきの Replay から引いた系列で RND の predictor を学び（Agent57 は NGU と同じと書く）、predictor の損失への重要度の補正を 3 本とも書かない。Agent57 の重要度の補正の指数は 0.0、MEME は 0.4 で、TD 誤差の統計だけを重要度で補正する [Badia Agent57 (2020/03), 付録 G.3 表 3；Kapturowski MEME (2022/09), §4 B1・表 2]。
- 公開の実装は、重要度の重みを掛ける（deep_rl_zoo、指数 0.6）、Replay を使わない（DI-engine）、重みなしで Replay の sample から学ぶ（Acme の RND）に分かれる（§4.3）。
- Sovrano（2019）は、Replay が主に珍しい状態を引き直すので、そこで predictor を学ぶとまだ珍しい状態の内的報酬が低く出るとして、RND を Replay の段で学ばせなかった [Sovrano POER (2019/05), §IV-C]。
- Munchausen と内的報酬を組んだ公開の結果は見つからなかった。Munchausen の原典の表 3（学習中の最大点）では、Montezuma's Revenge は M-DQN 0・M-IQN 0 だった [Vieillard M-DQN (2020/07), 付録 B.6 表 3]。
- Yu らは、SAC のエントロピーの報酬は終端の前の報酬を期待値で膨らませ、エントロピーが正なら生き続ける向きに働くと書く [Yu Entropy-Reward (2022/01), §5.2]。

## 内容

### 1. 対象の問いと結論

問いは 3 つある。NGU 系が RND の倍率に使う誤差の統計の期間と序盤の扱い。PER で引いた観測で predictor を学ぶときの偏りと重要度の補正、あわせて RND の原典が predictor を学ぶデータ。Munchausen（とエントロピーで正則化した学習）と内的報酬の組の公開の結果と理由の説明。

### 2. 調べた範囲と見つからなかったこと

2026-11-27 に、一次の資料（論文 9 本の arXiv の HTML と、公開実装 5 件のファイル）だけを読んだ。見つからなかったものは、NGU・Agent57・MEME の著者の公開の実装、直近の重みで倍率の統計を取る NGU 系の再実装、TD 誤差の PER で predictor を学んだときの新しさの偏りを測った文献と補正の有無を比べた文献、Munchausen と内的報酬を組んだ論文・結果・実装である。

### 3. Q18 RND の倍率の誤差の統計の期間と序盤

NGU の表 6 で RND に関わる行は、倍率の上限 L 5 と学習率 0.0005 だけである。Agent57 は α を [1, 5] に切ってエピソード内の報酬に掛け、表 3 の 6250 は Replay を使い始める条件で統計の待ちではない。MEME は RND の報酬を誤差 ÷ running の標準偏差と書き、表 2 の RND Clip Threshold は 5.0 である [Badia NGU (2020/02), 付録 F.2 表 6；Badia Agent57 (2020/03), §2・表 3；Kapturowski MEME (2022/09), 付録 G.1・表 2]。

DI-engine の NGU は、倍率を 1 ＋ (e − 平均) ÷ (標準偏差 ＋ 1e-11) にし、統計は平均 0・分散 1・件数 1e-4 から始め、learner が Replay から引いた batch で採点するたびに更新してから割る。deep_rl_zoo は actor ごとの部品で、誤差を平均を引かずに √(分散 ＋ 1e-8) で割り、[1, 5] に切る。統計は 10 件ごとに入れ、learner は `min_replay_size` に届くまで predictor を学ばない。rllte の NGU は clip(1 ＋ RND の値, 1, 5) で、RND の値の正規化は読んでいない（§3.2）。

### 4. Q19 predictor が学ぶデータと PER の偏り

RND の原典は rollout を集めた batch で蒸留の損失を最適化し、表 5 は rollout の長さ 128、env 128、ミニバッチ 4、エポック 4 である。NGU は TD 誤差の最大と平均を混ぜた優先度で引いた系列の末尾 5 時点で predictor を学び、優先度の指数は付録 A が 1.0、表 6 が 0.9 と食い違う。Agent57 の表 3 は優先度の指数 0.9・補正の指数 0.0、MEME の表 2 は 0.6・0.4（Max Priority Weight 0.9）である [Burda RND (2018/10), Algorithm 1・表 5；Badia NGU (2020/02), 付録 A・表 6；Badia Agent57 (2020/03), 表 3；Kapturowski MEME (2022/09), 表 2]。

deep_rl_zoo は末尾 5 時点で RND、末尾 6 時点で埋め込みの逆モデルを学び、Agent57 の Atari の設定は優先度の指数 0.9・補正の指数 0.6（減衰なし）、新しい優先度は 0.8 × 外的 ＋ 0.2 × 内的である。DI-engine は集めたばかりのデータから収集 1 回ごとに 64 件を 100 回引く。Acme の RND は更新したあとの predictor で報酬を計算し直し、係数の既定は内的 1.0・外的 0.0 である（§4.3）。

Sovrano は、重要度の補正を別に置かず PPO の確率比がその役をすると書き、Replay の頻度を上げると探索が悪くなった（μ=2、図 3）。PER の原典は、優先抽出が推定の収束する解を変え、重要度の重みは β = 1 で非一様な抽出を打ち消すとし、β を学習の終わりに 1 へ線形に上げる [Sovrano POER (2019/05), §III-B・§V-B；Schaul PER (2015/11), §3.4]。

### 5. Q21 Munchausen・エントロピーの正則化と内的報酬の組

RLeXplore は SAC（エントロピーの係数 0.2、自動調整）に Disagreement・RND・ICM を足し、Ant-UMaze（100 万 step）で素の SAC より効率よく迷路を進んだと書く（値は図 31 だけ）[Yuan RLeXplore (2024/05), 付録 G]。

Munchausen の原典の既定値は α 0.9・τ 0.03・l0 −1 で、softmax の探索は Q の尺度に依存するとして ε-greedy で行動した。Enduro では Q が 0 に近い序盤に softmax がほぼ一様になり学べなかった。表 3 では DQN 2・IQN 500 で、主な実験は 200M フレーム・3 seed・sticky actions である。§4 は、命を失ったときと game over のどちらで終わらせるかで点が大きく変わると書く [Vieillard M-DQN (2020/07), §4、付録 B.2・B.6]。

Yu らの SimpleChain（5 ノード、報酬 −0.05、α 0.2 を固定）では 1 step のエントロピーの報酬は約 0.14 で agent は終端を避け、BipedalWalker では前へ進まずに 1600 step の上限まで続く失敗が約 1.2M step まで続いた。対処は SACLite（方策の評価から外す）と SACZero（移動平均を引く）である [Yu Entropy-Reward (2022/01), §5.1〜§5.4]。

### 6. 批判と懸念

NGU は §2 で「学習中に見たすべての観測」で predictor を学ぶと書き、付録 A で末尾 5 時点と書く。再実装は原典と違う所があり（deep_rl_zoo の倍率は平均を引かず、補正の指数 0.6 は Agent57 の 0.0 と違う。README は NGU と Agent57 を Pong か Breakout だけで試したと印を付ける）、原典の意図の代わりにならない。Sovrano は数値の表と seed の数が無く、Yu らは連続制御の SAC で、M-DQN の表 3 は学習中の最大点である。

### 7. 調査の限界

論文と実装は要約するツールを通して読み、PDF で 1 語ずつ照合していない。被引用数は取得できなかった。rllte の基底クラス、Acme の RND と組む learner の Replay の設定は読んでいない。報告の §3.3・§4.5・§5.4 は報告での推論の節である。

## 主要な概念

- [内的報酬による探索の比較](../../../concepts/methods/intrinsic_novelty.md)
- [優先度付き経験再生（PER）](../../../concepts/methods/prioritized_replay.md)
- [報酬尺度と割引率](../../../concepts/methods/reward_scale_discount.md)
- [Montezuma's Revenge](../../../concepts/subjects/montezuma_revenge.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
