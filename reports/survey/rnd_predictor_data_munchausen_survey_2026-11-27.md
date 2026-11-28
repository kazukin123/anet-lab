# Survey: RND の倍率の統計・predictor が学ぶデータ・Munchausen と内的報酬の組

Date: 2026-11-27
Scope: NGU 系（NGU・Agent57・MEME）が RND の倍率に使う誤差の統計の取り方と序盤の扱い、RND の predictor が学ぶデータ（RND の原典と、優先度つきの Replay から学ぶ NGU 系）と重要度の補正、Munchausen RL（とエントロピーで正則化した学習）と内的報酬の組み合わせを、原典の本文・付録と公開の実装で確かめる。内的報酬の手法の全体像は [内的報酬の手法の全体像](intrinsic_reward_landscape_survey_2026-11-18.md)、NGU 系を価値ベースに載せる論点は [エピソード内の新しさを価値ベースに載せる論点](episodic_bonus_value_based_survey_2026-11-18.md) にあり、本冊はそこに無い部分に絞る。

読み方: 本文は「主張（筆者の要約）→ 出典の位置」の順で書く。逐語の引用は 1 か所だけにし、ほかは要約と位置（節・付録・表・図・式、実装はファイル）で示す。論文は arXiv・ar5iv の HTML を、実装は GitHub の生のファイルを、要約するツールを通して読んだ（§7）。「本報告の読み取り」の節は原典に無い推論で、事実の節と分けて書く。

## 目次

1. [1. 対象の問いと結論](#1-対象の問いと結論)
   - [1.1 対象の問い](#11-対象の問い)
   - [1.2 結論](#12-結論)
2. [2. 調べた範囲と見つからなかったこと](#2-調べた範囲と見つからなかったこと)
3. [3. Q18 RND の倍率の誤差の統計の期間と序盤](#3-q18-rnd-の倍率の誤差の統計の期間と序盤)
   - [3.1 NGU・Agent57・MEME の原典](#31-nguagent57meme-の原典)
   - [3.2 公開の再実装](#32-公開の再実装)
   - [3.3 本報告の読み取り](#33-本報告の読み取り)
4. [4. Q19 predictor が学ぶデータと PER の偏り](#4-q19-predictor-が学ぶデータと-per-の偏り)
   - [4.1 RND の原典が predictor を学ぶデータ](#41-rnd-の原典が-predictor-を学ぶデータ)
   - [4.2 NGU・Agent57・MEME の原典](#42-nguagent57meme-の原典)
   - [4.3 公開の実装](#43-公開の実装)
   - [4.4 Replay で predictor を学ぶ偏りと重要度の補正を論じた文献](#44-replay-で-predictor-を学ぶ偏りと重要度の補正を論じた文献)
   - [4.5 本報告の読み取り](#45-本報告の読み取り)
5. [5. Q21 Munchausen・エントロピーの正則化と内的報酬の組](#5-q21-munchausenエントロピーの正則化と内的報酬の組)
   - [5.1 組んだ公開の結果](#51-組んだ公開の結果)
   - [5.2 Munchausen の原典に書かれていること](#52-munchausen-の原典に書かれていること)
   - [5.3 終端の前に足すエントロピーの報酬（Yu ら）](#53-終端の前に足すエントロピーの報酬yu-ら)
   - [5.4 本報告の読み取り](#54-本報告の読み取り)
6. [6. 批判と懸念](#6-批判と懸念)
7. [7. 調査の限界](#7-調査の限界)
8. [8. 出典一覧](#8-出典一覧)

## 1. 対象の問いと結論

### 1.1 対象の問い

- Q18 の外の部分: NGU（Never Give Up）と Agent57 は、RND の倍率（一生の新しさの倍率 α）に使う誤差の平均と標準偏差を、どの期間（累積か、直近の重みか）で取るか。統計が育つ前の序盤をどう扱うか。論文の本文・付録と、公開の再実装で確かめる。
- Q19 の外の部分: Replay から優先度付き（TD 誤差の PER）で引いた観測で RND の predictor（NGU では補助の損失）を学ぶとき、新しさの判定がどの状態へ偏るか、重要度の補正を掛けるか、を扱った文献か公開実装はあるか（NGU・Agent57・MEME とその再実装）。元の RND 論文（Burda ら）が predictor を学ぶデータ（rollout の経験、間引きの割合）も、原典で確かめる。
- Q21 の外の部分: Munchausen RL（や softmax・エントロピーで正則化した Q 学習）と、RND などの内的報酬を組んだ公開の結果はあるか。組むと悪くなる・良くなるという報告や、その理由の説明はあるか。

### 1.2 結論

- **Q18**: NGU は倍率の誤差の平均と標準偏差を running の値とだけ書き、累積か直近の重みか、窓、序盤の扱いを書かない。Agent57 は倍率の詳細を NGU に譲り、MEME も running の平均と標準偏差とだけ書く。3 本とも統計が育つ前の序盤の扱いを書いていない。読んだ公開の再実装 2 つ（DI-engine の NGU、deep_rl_zoo の NGU・Agent57）は、件数で重みを付けた累積で統計を取り、序盤の待ちを置かず、統計の初めの分散を 1 にして最初の採点から割る。著者の公開の実装は見つからず、原典の running の中身は確かめられなかった。
- **Q19**: RND の原典は、今の rollout の経験で predictor を学び、Replay も優先度も使わない。割合 0.25 は、env の数を変えても predictor の実効の batch と内的報酬の下がる速さをそろえるための dropout である。NGU・Agent57・MEME は優先度つきの Replay から引いた系列で predictor を学び、predictor の損失への重要度の補正を 3 本とも書かない（Agent57 は補正の指数 0.0、MEME は TD 誤差の統計だけを補正）。公開の実装は、重みを掛ける（deep_rl_zoo）、Replay を使わない（DI-engine）、重みなしで Replay の sample から学ぶ（Acme の RND）に分かれる。偏りの向きを書いた文献は Sovrano（2019）だけで、Replay が珍しい状態を引き直すので、そこで学ぶとまだ珍しい状態の内的報酬が下がるとして、RND を Replay の段で学ばせなかった。TD 誤差の PER で学んだときの偏りを測った文献は見つからなかった。
- **Q21**: Munchausen と内的報酬を組んだ公開の結果は見つからなかった。Munchausen の原典は内的報酬に触れず、Montezuma's Revenge の学習中の最大点は M-DQN・M-IQN とも 0 だった。近い指摘として、Yu ら（SAC）は、終端の前の step に足すエントロピーの報酬が報酬を膨らませ、終わらない・長く続ける方策を生むと報告した。RLeXplore では、エントロピーで正則化した SAC に RND などを足すと、Ant-UMaze で素の SAC より良かった（数値なし）。

## 2. 調べた範囲と見つからなかったこと

- **日付**: 2026-11-27 に読んだ。
- **一次の資料**: NGU・Agent57・MEME・RND・PER・Munchausen RL・Yu ら・Sovrano・RLeXplore の arXiv（ar5iv）の HTML。公開の実装は deep_rl_zoo（main の 19e3844、2024-02-21）、DI-engine（`ding/reward_model/ngu_reward_model.py` の最後のコミット a580019（2023-04-03）、`ding/utils/default_helper.py`・`ding/entry/serial_entry_ngu.py` は main）、Acme（`acme/agents/jax/rnd/` の最後のコミット 56d1bcf（2026-10-06））、rllte（`rllte/xplore/reward/ngu.py`・`rnd.py` の main）。deep_rl_zoo・DI-engine・rllte は著者以外の再実装、Acme は DeepMind の公開ライブラリの RND で、NGU の実装ではない。
- **二次の資料**: 使っていない。
- **検索語**: NGU・Agent57 の公式のコード、RND の predictor と prioritized replay と importance sampling、off-policy の RND と replay、Munchausen と RND・内的報酬・難探索、エントロピーで正則化した Q 学習と RND、最大エントロピーの RL と終端の偏り。
- **見つからなかったこと**:
  - NGU・Agent57・MEME の著者の公開の実装（倍率の統計の running の中身を確かめる手段）。
  - 直近の重み（EMA・窓）で倍率の統計を取る NGU 系の再実装（読んだ 2 つには無かった）。
  - TD 誤差の PER で引いた観測で predictor を学んだときの新しさの偏りを測った文献、重要度の補正の有無を比べた文献。
  - Munchausen（M-DQN・M-IQN）と内的報酬を組んだ論文・公開の結果・実装、組んで悪くなる・良くなる理由を論じた文献。離散行動のエントロピー正則化 Q 学習と内的報酬を Atari の難探索で組んだ報告。

## 3. Q18 RND の倍率の誤差の統計の期間と序盤

### 3.1 NGU・Agent57・MEME の原典

NGU は、倍率 α_t = 1 ＋ (err(x_t) − μ_e) ÷ σ_e の μ_e と σ_e を running の平均と標準偏差とだけ書く。累積か、指数の重みか、窓かは書かず、更新の周期も書かない。倍率の上限は L = 5 に固定する。

```
where σe and μe are running standard deviation and mean for err(xt)

σe と μe は err(xt) の running（走る）標準偏差と平均である
```

[Badia NGU (2020/02), §2]

NGU は統計が育つ前の序盤の扱い（待ち、初めの値）を書いていない。付録 F.2 の表 6 にも統計の取り方や待ちの行は無く、RND に関わる行は「RND clipping factor L 5」と「Learning rate (RND and Action prediction) 0.0005」である。付録 I の残りの部分にも、統計と待ちの記述は無い。[Badia NGU (2020/02), §2、付録 F.2 表 6、付録 I]

Agent57 は、内的報酬を r^i_t = r^episodic_t · min{max{α_t, 1}, L}（L = 5）と書き、α_t の作り方は NGU を参照させる。α_t は学習を通じてゆっくり消えると書く。誤差の正規化の統計と序盤の扱いの記述は無い。表 3 の「Minimum sequences to start replay 6250」は Replay を使い始める条件で、RND の統計の待ちではない。[Badia Agent57 (2020/03), §2、付録 G.3 表 3]

MEME は、RND の報酬を誤差 ÷ running の標準偏差（付録 G.1、式 11）、一生の新しさの信号を誤差の running の平均と標準偏差で正規化した値（付録 G.2、式 13）と書く。推定の方式（EMA・窓・累積）は書かない。表 2 の「RND Clip Threshold」は 5.0 である。表 2 の「Burn-in 0」は系列の burn-in で、統計の待ちではない。序盤の扱いの記述は無い。[Kapturowski MEME (2022/09), 付録 G.1・G.2、表 2]

### 3.2 公開の再実装

**DI-engine の NGU（著者以外、査読なしの OSS）**: RND の部品 `RndNGURewardModel` は、倍率を 1 ＋ (e − 平均) ÷ (標準偏差 ＋ 1e-11) にする。統計の `RunningMeanStd` は、batch の平均と分散を件数で重みを付けて合流させる累積で、減衰も窓も無い。初めは平均 0・分散 1・件数 1e-4 で、標準偏差は √(分散 ＋ 1e-8) である。採点（`estimate`）のたびに、その batch の誤差で統計を更新してから割る。学習の入口（`serial_entry_ngu.py`）は、learner が Replay から引いた batch に対して `estimate` を呼ぶ。待ちは置かない。[DI-engine NGU (2023/04), ngu_reward_model.py の RndNGURewardModel；DI-engine (2026/11), default_helper.py の RunningMeanStd、serial_entry_ngu.py]

**deep_rl_zoo の NGU・Agent57（著者以外）**: RND の部品 `RndLifeLongBonusModule`（`curiosity.py`）は、predictor と target の出力の差の二乗の平均を誤差にし、平均を引かずに √(分散 ＋ 1e-8) で割る。統計の `RunningMeanStd` は件数で重みを付けた累積で、初めは平均 0・分散 1・件数 0、`update_single` が 10 件ためるごとに統計へ入れる。actor が `min(max(倍率, 1.0), 5.0)` で [1, 5] に切り、エピソード内の報酬に掛ける。部品は actor ごとに持つ。倍率の待ちは無く、learner は Replay が `min_replay_size` に届くまで predictor を学ばない。[deep_rl_zoo (2024/02), curiosity.py、normalizer.py、agent57/agent.py、ngu/agent.py]

**rllte の NGU（著者以外）**: 倍率を clip(1 ＋ RND の値, 1, 5)（上限の既定 5.0）にする。RND の値の正規化（既定は `rms`）は基底クラスにあり、読んでいない。[rllte (2026/11), ngu.py・rnd.py]

### 3.3 本報告の読み取り

- 原典の running は、累積と直近の重みのどちらにも読める。読んだ再実装 2 つはどちらも累積を選び、どちらも序盤の待ちを置かない。原典の意図がどちらかは、この 2 つからは決められない。
- deep_rl_zoo の倍率は、誤差を平均を引かずに標準偏差で割る形で、NGU の式（平均を引く）と違う。DI-engine は NGU の式どおりだが、統計は learner が Replay から引いた batch で更新するので、優先度つきの Replay なら統計も抽出の分布を映す（コードからの推論で、確かめていない）。

## 4. Q19 predictor が学ぶデータと PER の偏り

### 4.1 RND の原典が predictor を学ぶデータ

RND の原典は、predictor をエージェントが集めたデータで学ぶと書き（§2.2）、Algorithm 1 では、rollout を集めた batch B_i で蒸留の損失を Adam で最適化する。Replay と優先度の記述は無い。[Burda RND (2018/10), §2.2、付録 A.2 Algorithm 1]

表 5 の「predictor に使う経験の割合」は 0.25 で、rollout の長さ 128、env 128、ミニバッチ 4、エポック 4 である。付録 A.4 は、env を 32 から 128 に増やしても predictor の実効の batch をそろえるためと理由を書き、dropout の残す確率を 128 env で 0.25、256 env で 0.125、1,024 env で 0.03125 とした。§3.4 は、env が多いと predictor が速く学ぶので、内的報酬が時間とともに下がる速さを env の数によらずそろえるため、と書く。[Burda RND (2018/10), 付録 A.4・表 5、§3.4]

### 4.2 NGU・Agent57・MEME の原典

- **NGU**: learner は、TD 誤差の最大と平均を混ぜた優先度で Replay から系列を引き、系列の末尾 5 時点で行動予測の Network と RND の predictor を学ぶ（付録 A）。優先度の指数は付録 A が 1.0、表 6 が 0.9 と食い違う。表 6 に重要度の補正の指数の行は無く、RND と埋め込みの損失への補正の記述も無い。§2 は、学習中に見たすべての観測で predictor を学ぶと概念の形で書く。[Badia NGU (2020/02), §2、付録 A、付録 F.2 表 6]
- **Agent57**: 行動予測と RND の損失は NGU と同じく Adam で最適化すると書き、使う時点は書かない（付録 E）。表 3 は「Replay priority exponent 0.9」「Importance sampling exponent 0.0」で、指数 0.0 は重要度の重みをすべて 1 にするので、Q の損失にも補正を掛けない。[Badia Agent57 (2020/03), 付録 E、付録 G.3 表 3]
- **MEME**: learner は引いた系列で行動予測の Network と RND の predictor を学ぶ（付録 D）。表 2 は「Priority Exponent 0.6」「Importance Sampling Exponent 0.4」「Max Priority Weight 0.9」である。§4 の B1 は、TD 誤差の標準偏差の running の統計を learner で取り、抽出の分布を重要度で補正すると書く。RND と行動予測の損失への補正には触れない。付録 G.1 は、predictor を学習中に見たすべての観測で学ぶと概念の形で書く。[Kapturowski MEME (2022/09), §4 B1、付録 D・G.1、表 2]

### 4.3 公開の実装

- **deep_rl_zoo の NGU・Agent57**: learner は優先度つきで引いた系列の末尾 5 時点で RND を、末尾 6 時点（5 遷移）で埋め込みの逆モデルを学び、2 つの損失の和に重要度の重みを掛けて平均する。Agent57 の Atari の設定（`agent57/run_atari.py`）は、優先度の指数 0.9、重要度の補正の指数 0.6（コードの注記どおり減衰なし）、重みの正規化あり。新しい優先度は 0.8 × 外的 ＋ 0.2 × 内的である。Agent57 の原典の表 3 の指数 0.0 とは違う。[deep_rl_zoo (2024/02), agent57/agent.py、agent57/run_atari.py、ngu/agent.py]
- **DI-engine の NGU**: RND の predictor は Replay ではなく、集めたばかりのデータ（`collect_data`）で学ぶ。収集 1 回ごとに 64 件を 100 回引き、`only_use_last_five_frames` を選ぶと各エピソードの末尾 5 フレームだけを使う。優先度と重要度の重みは predictor の学習に無い。倍率の採点は learner が Replay から引いた batch で行う（§3.2）。[DI-engine NGU (2023/04), ngu_reward_model.py；DI-engine (2026/11), serial_entry_ngu.py]
- **Acme の RND（DeepMind の公開ライブラリ）**: predictor は、Q の学習（直接の RL の learner）と同じ Replay の sample から作った遷移で学び、損失は重みを掛けない二乗誤差である。報酬は更新したあとの predictor で計算し直して sample の報酬を置き換える。係数の既定は内的 1.0・外的 0.0 である。読んだファイルに誤差と観測の正規化は無い。Replay が優先度つきかは、組み合わせる learner の設定により、確かめていない。[Acme RND (2026/10), learning.py・networks.py]

### 4.4 Replay で predictor を学ぶ偏りと重要度の補正を論じた文献

**Sovrano（2019）**: PPO/RND に、内的報酬の和で優先度を付けた Replay（POER）を足して方策と価値を学び直すが、RND は Replay の段では学ばせない（§IV-C の 1）。理由は、Replay が主に珍しい状態を引き直すので、そこで predictor を学ぶと、まだ珍しい状態の予測誤差が下がり、内的報酬が低く出るためと書く。Replay した batch の内的の収益は計算し直す（§IV-C の 2）。重要度の補正は別に置かず、PPO の確率比が重要度の重みの役をすると書く（§III-B・§IV）。Replay の頻度を上げると探索が悪くなった（μ=2、図 3）。結果は図だけで、数値の表と seed の数は無い。[Sovrano POER (2019/05), §I・§III-B・§IV-B・§IV-C・§V-B、図 3〜6]

**PER の原典**: 優先抽出は経験の分布を制御なしに変え、方策と状態の分布が固定でも、推定が収束する解を変える。重要度の重み w_i = (1/N · 1/P(i))^β は β = 1 で非一様な抽出を打ち消し、重みは 1/max w で正規化する。偏りのない更新が要るのは学習の終わりの収束の近くとして、β を学習の終わりに 1 へ届くよう線形に上げる。[Schaul PER (2015/11), §3.4]

### 4.5 本報告の読み取り

- Sovrano の指摘（多く引き直した状態ほど predictor が学び、新しさが低く出る）を TD 誤差の PER に当てると、TD 誤差の大きい遷移の観測ほど predictor が多く学び、訪問が少なくても新しさが低く出ることになる。RND の原典が、predictor の学ぶ量を env の数に合わせて変え、内的報酬の下がる速さをそろえたこと（§4.1）とも同じ向きである。どの出典も TD 誤差の PER でこれを測っていない。
- PER の原典に従えば、β = 1 の重要度の重みで勾配の期待値は一様な抽出に戻る。NGU と Agent57 はこの補正を predictor に掛けず、deep_rl_zoo は指数 0.6 で掛ける。補正の有無で新しさがどれほど変わるかの報告は無い。

## 5. Q21 Munchausen・エントロピーの正則化と内的報酬の組

### 5.1 組んだ公開の結果

Munchausen（M-DQN・M-IQN）と RND などの内的報酬を組んだ公開の結果は、§2 の検索語では見つからなかった。Agent57 と MEME の読めた範囲に Munchausen の記述は無く、Agent57 の actor は ε-greedy である。[Badia Agent57 (2020/03), 付録 E；Kapturowski MEME (2022/09)]

エントロピーで正則化した学習と内的報酬を組んだ近い例として、RLeXplore は SAC（エントロピーの係数 0.2、自動調整、表 19）に Disagreement・RND・ICM を足し、報酬の疎な Ant-UMaze（100 万 step、表 19 の 1·10⁶）で、内的報酬ありの agent が素の SAC より効率よく迷路を進み、目標を多く見つけたと書く。値は図 31 だけで、本文に数値は無く、seed の数も書かれていない。連続制御の actor-critic で、Q 学習ではない。[Yuan RLeXplore (2024/05), 付録 G・表 19・図 31]

### 5.2 Munchausen の原典に書かれていること

- **式と既定値**: M-DQN の目標は、報酬に α τ ln π(a_t|s_t) を足し（τ ln π は [l0, 0] に切る）、bootstrap に Σ π(a'|s') (q(s', a') − τ ln π(a'|s')) を使う。π は softmax(q/τ) である（式 3）。既定値は α 0.9・τ 0.03・l0 −1 で、少数の Atari のゲームで調整した。原典は、これは最大エントロピーの RL とは違うと断る。[Vieillard M-DQN (2020/07), §3 式 3、§4]
- **softmax の方策と Q の尺度**: M-DQN は確率的な方策を自然に作るが、softmax の探索は Q の尺度に依存し ε-greedy は依存しないので、主な結果では ε-greedy で行動した。Enduro では Q が 0 に近い序盤に τ 0.03 の softmax がほぼ一様になり、報酬をほとんど得られずに学べなかった。[Vieillard M-DQN (2020/07), 付録 B.2・図 5]
- **Montezuma's Revenge**: 表 3（学習中の最大点、100 エピソード・3 seed の平均）では、Montezuma's Revenge は M-DQN 0・M-IQN 0、同じ表の DQN 2・IQN 500 だった。主な実験は 200M フレーム・3 seed・sticky actions で、比べる基準は Dopamine の値である。[Vieillard M-DQN (2020/07), §4、付録 B.6 表 3]
- **内的報酬**: 原典の本文と付録に、内的報酬・探索のボーナス・難探索のゲームの議論は無い。§4 は、命を失ったときと game over のどちらで終わらせるかで点が大きく変わると書く。[Vieillard M-DQN (2020/07), §4、付録 B]

### 5.3 終端の前に足すエントロピーの報酬（Yu ら）

Yu らは、SAC のエントロピーの報酬は終端でない step にだけ足され、終端のあとの Q は TD の backup で 0 になるので、終端の前の報酬を期待値で膨らませ、エントロピーが正なら生き続ける向きに働くと書く（負なら逆）。この副作用を reward inflation と呼ぶ。[Yu Entropy-Reward (2022/01), §5.2]

- **SimpleChain**: 5 ノード、終端でない 0〜3 番のノードの報酬 −0.05 で、α 0.2 を固定し方策のエントロピーが log 2 ≈ 0.7 なら、1 step のエントロピーの報酬は約 0.14、差し引き約 0.09 で、agent は終端を避ける。α を調整する場合は、α·H(π) が 0.05 を下回るまで終端へ向かわない。α を固定した SAC はほとんど進まなかった。[Yu Entropy-Reward (2022/01), §5.1・§5.3、図 1]
- **BipedalWalker**: SAC は、地面で脚を開いたまま前へ進まず 1600 step の上限まで続く失敗を見せ、エントロピーの報酬が下がる約 1.2M step まで続いた。[Yu Entropy-Reward (2022/01), §5.4]
- **対処**: SACLite は方策の評価からエントロピーの報酬を外し、SACZero は移動平均を引いて平均 0 にする。大規模な比較では SACLite が SAC と同等か上回ることが多かった。[Yu Entropy-Reward (2022/01), §5.2・§6・§8]

Yu らは連続制御の SAC を扱い、Munchausen と内的報酬の組には触れない。arXiv の版で、会議の記載は無い。

### 5.4 本報告の読み取り

- M-DQN の bootstrap の −τ Σ π ln π は、終端でない遷移の bootstrap にだけ付く正の値で、Yu らが扱うエントロピーの報酬と同じ形をしている。Yu らの仕組みが M-DQN にも当たるかは、どの出典も確かめていない。
- 本報告の計算: 18 行動の softmax が一様（Q が平らで、付録 B.2 の Enduro の序盤と同じ状況）なら、既定値（α 0.9・τ 0.03）で、報酬に足す項 α τ ln π は約 −0.078、bootstrap のエントロピーの項 τ ln 18 は約 0.087 で、γ を 1 に近いとすると 1 step あたり差し引き約 0.0087（= (1 − α) τ ln 18）の正の値になる。外的報酬も内的報酬もほぼ 0 の間は、この正の値が終端の前の step に付き続ける。実際の方策は一様でなく、値は選んだ行動の確率と次の状態のエントロピーで変わる（選んだ行動の確率が低いと報酬に足す項はもっと負になる）。
- 内的報酬を足したときにこの項が新しさの向きとどう競うかは、どの出典も扱っていない。

## 6. 批判と懸念

- **原典の中の食い違い**: NGU は §2 で「学習中に見たすべての観測」で predictor を学ぶと書き、付録 A で系列の末尾 5 時点と書く。優先度の指数も付録 A（1.0）と表 6（0.9）で違う。[Badia NGU (2020/02), §2、付録 A、表 6]
- **再実装は原典の意図の代わりにならない**: deep_rl_zoo の倍率は平均を引かず、重要度の補正の指数は 0.6 で Agent57 の表 3 の 0.0 と違う。DI-engine は predictor を Replay から学ばない。README は、deep_rl_zoo の NGU と Agent57 を Pong か Breakout だけで試したと印を付けている。[deep_rl_zoo (2024/02), README・agent57/run_atari.py；Badia Agent57 (2020/03), 表 3]
- **Sovrano の証拠の薄さ**: 著者 1 名の論文で、seed の数と数値の表が無く、結果は図だけである。優先度は内的報酬の和で、TD 誤差の PER ではない。[Sovrano POER (2019/05), §V]
- **Yu らの範囲**: 連続制御の SAC で、離散行動の Q 学習と内的報酬は扱わない。査読の有無は確かめていない。[Yu Entropy-Reward (2022/01)]
- **M-DQN の表 3 の統計量**: 学習中の最大点で、最終の評価の点ではない。[Vieillard M-DQN (2020/07), 付録 B.6 表 3]

## 7. 調査の限界

- **照合**: 論文と実装は要約するツールを通して読み、PDF で 1 語ずつ照合していない。数値はツールが示した表と本文の値を写した。NGU の付録 A の「末尾 5 時点」は、HTML の表示が「55」と崩れており、ツールの読み（5）に従った。このリポジトリの [RND を Burda らの形へ寄せる RFD](../../docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) §3.3 も、PDF で 5 時点と読んでいる。
- **被引用数**: Semantic Scholar の API が 429（要求が多すぎる）を返し、取得できなかった。
- **読めなかったもの・読まなかったもの**: Sovrano の PDF 版（arXiv の HTML 版を読んだ）、rllte の基底クラス（RND の値の正規化）、Acme の RND と組む learner の Replay の設定、DI-engine の NGU の Replay の種類の既定。
- **版**: deep_rl_zoo は 19e3844、DI-engine の `ngu_reward_model.py` は a580019、Acme の `rnd/` は 56d1bcf を最後のコミットとして見た。main の今の内容を 2026-11-27 に読んだので、のちの変更は反映していない。
- **引用**: 逐語の引用は §3.1 の 1 か所（12 語）だけで、ほかは要約と位置で示した。

## 8. 出典一覧

書誌の会議名は arXiv の頁の Comments 欄による。arXiv の頁に記載が無いもの（Agent57・MEME・RND）は [エピソード内の新しさを価値ベースに載せる論点](episodic_bonus_value_based_survey_2026-11-18.md) の書誌に従い、そう注記する。所属は論文の頁か同じ報告の書誌による。URL は、読んだ頁（本文の HTML、実装の生のファイル、コミットの履歴の頁）を挙げる。

- [Badia NGU, 2020/02] Adrià Puigdomènech Badia, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Bilal Piot, Steven Kapturowski, Olivier Tieleman, Martín Arjovsky, Alexander Pritzel, Andew Bolt, Charles Blundell. "Never Give Up: Learning Directed Exploration Strategies." ICLR 2020. https://arxiv.org/abs/2002.06038 ・ https://ar5iv.labs.arxiv.org/html/2002.06038 — 所属 DeepMind。
- [Badia Agent57, 2020/03] Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020（arXiv の頁に記載なし、既存の報告の書誌による）. https://arxiv.org/abs/2003.13350 ・ https://ar5iv.labs.arxiv.org/html/2003.13350 — 所属 DeepMind。
- [Kapturowski MEME, 2022/09] Steven Kapturowski, Víctor Campos, Ray Jiang, Nemanja Rakićević, Hado van Hasselt, Charles Blundell, Adrià Puigdomènech Badia. "Human-level Atari 200x faster." ICLR 2023（arXiv の頁に記載なし、既存の報告の書誌による）. https://arxiv.org/abs/2209.07550 ・ https://ar5iv.labs.arxiv.org/html/2209.07550 — 所属 DeepMind。
- [Burda RND, 2018/10] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. "Exploration by Random Network Distillation." ICLR 2019（arXiv の頁に記載なし、既存の報告の書誌による）. https://arxiv.org/abs/1810.12894 ・ https://ar5iv.labs.arxiv.org/html/1810.12894 — 所属 OpenAI、University of Edinburgh。
- [Schaul PER, 2015/11] Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. "Prioritized Experience Replay." ICLR 2016. https://arxiv.org/abs/1511.05952 ・ https://ar5iv.labs.arxiv.org/html/1511.05952 — 所属 Google DeepMind。
- [Vieillard M-DQN, 2020/07] Nino Vieillard, Olivier Pietquin, Matthieu Geist. "Munchausen Reinforcement Learning." NeurIPS 2020. https://arxiv.org/abs/2007.14430 ・ https://ar5iv.labs.arxiv.org/html/2007.14430 — 所属 Google Research, Brain Team。
- [Yu Entropy-Reward, 2022/01] Haonan Yu, Haichao Zhang, Wei Xu. "Do You Need the Entropy Reward (in Practice)?" arXiv（Comments 欄なし、査読の有無は未確認）. https://arxiv.org/abs/2201.12434 ・ https://arxiv.org/html/2201.12434
- [Sovrano POER, 2019/05] Francesco Sovrano. "Combining Experience Replay with Exploration by Random Network Distillation." IEEE Conference on Games (CoG) 2019. https://arxiv.org/abs/1905.07579 ・ https://ar5iv.labs.arxiv.org/html/1905.07579 — 所属 University of Bologna。
- [Yuan RLeXplore, 2024/05] Mingqi Yuan, Roger Creus Castanyer, Bo Li, Xin Jin, Wenjun Zeng, Glen Berseth. "RLeXplore: Accelerating Research in Intrinsically-Motivated Reinforcement Learning." Transactions on Machine Learning Research, 2025. https://arxiv.org/abs/2405.19548 ・ https://arxiv.org/html/2405.19548
- [deep_rl_zoo, 2024/02] michaelnny. Deep RL Zoo（main 19e3844）. https://github.com/michaelnny/deep_rl_zoo ・ https://github.com/michaelnny/deep_rl_zoo/commits/main ・ https://raw.githubusercontent.com/michaelnny/deep_rl_zoo/main/deep_rl_zoo/curiosity.py ・ https://raw.githubusercontent.com/michaelnny/deep_rl_zoo/main/deep_rl_zoo/normalizer.py ・ https://raw.githubusercontent.com/michaelnny/deep_rl_zoo/main/deep_rl_zoo/agent57/agent.py ・ https://raw.githubusercontent.com/michaelnny/deep_rl_zoo/main/deep_rl_zoo/agent57/run_atari.py ・ https://raw.githubusercontent.com/michaelnny/deep_rl_zoo/main/deep_rl_zoo/ngu/agent.py — 著者以外の公開の再実装（Apache License 2.0）。
- [DI-engine NGU, 2023/04] OpenDILab. DI-engine（ding/reward_model/ngu_reward_model.py、最後のコミット a580019）. https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/reward_model/ngu_reward_model.py ・ https://github.com/opendilab/DI-engine/commits/main/ding/reward_model/ngu_reward_model.py — 著者以外の公開の再実装（査読なし）。
- [DI-engine, 2026/11] OpenDILab. DI-engine（main を 2026-11-27 に読んだ）. https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/utils/default_helper.py ・ https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/entry/serial_entry_ngu.py — 公開実装。
- [Acme RND, 2026/10] Google DeepMind. Acme（acme/agents/jax/rnd、最後のコミット 56d1bcf）. https://raw.githubusercontent.com/google-deepmind/acme/master/acme/agents/jax/rnd/learning.py ・ https://raw.githubusercontent.com/google-deepmind/acme/master/acme/agents/jax/rnd/networks.py ・ https://github.com/google-deepmind/acme/commits/master/acme/agents/jax/rnd — DeepMind の公開ライブラリ（NGU の実装ではない）。
- [rllte, 2026/11] RLE-Foundation. rllte（main を 2026-11-27 に読んだ）. https://raw.githubusercontent.com/RLE-Foundation/rllte/main/rllte/xplore/reward/ngu.py ・ https://raw.githubusercontent.com/RLE-Foundation/rllte/main/rllte/xplore/reward/rnd.py — 著者以外の公開の再実装。
