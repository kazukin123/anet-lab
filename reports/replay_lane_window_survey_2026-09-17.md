# Survey: replay バッファの「1 アクターあたりの履歴窓」とエピソード長の比

調査日: 2026-09-17
調査範囲: off-policy 価値学習（DQN / IQN / PER / n-step）において、`replay_capacity / num_envs`（1 アクターあたりの履歴窓）とエピソード長の比が成績に与える影響。および周辺 4 角度 — 容量の分解研究、分散アクター構成での容量設計、エピソード単位 replay、評価 ε の影響。

引用規約: 原文と日本語訳を同一コードブロック内に空行区切りで併記し、出典はフェンス直後の行に `[ラベル (YYYY/MM), 位置]` 形式で置く。

## 目次

1. [容量の効果を分解した研究](#1-容量の効果を分解した研究)
   - [Fedus et al. (ICML 2020): 容量と最古方策年齢の直交分解](#fedus-et-al-icml-2020-容量と最古方策年齢の直交分解)
   - [容量 × n-step の相互作用](#容量--n-step-の相互作用)
   - [非単調性と文献間の不一致](#非単調性と文献間の不一致)
   - [replay ratio / UTD スケーリングの系列](#replay-ratio--utd-スケーリングの系列)
2. [分散アクター構成における容量設計](#2-分散アクター構成における容量設計)
   - [システム別の実数値](#システム別の実数値)
   - [Ape-X: 比例させないことの明示と対照実験](#ape-x-比例させないことの明示と対照実験)
   - [Stable-Baselines3: 実装による比の固定](#stable-baselines3-実装による比の固定)
   - [PQL: 比例スケーリングは不要という結論](#pql-比例スケーリングは不要という結論)
3. [エピソード単位 replay とエピソード長](#3-エピソード単位-replay-とエピソード長)
   - [シーケンス replay とエピソード境界](#シーケンス-replay-とエピソード境界)
   - [エピソード長でバッファ容量を正規化した唯一の実例](#エピソード長でバッファ容量を正規化した唯一の実例)
   - [エピソード境界の扱いがバッファサイズ感度を左右する](#エピソード境界の扱いがバッファサイズ感度を左右する)
   - [軌跡沿いの価値伝播](#軌跡沿いの価値伝播)
4. [評価 ε が結論を変える現象](#4-評価-ε-が結論を変える現象)
   - [評価 ε の値の系譜](#評価-ε-の値の系譜)
   - [BTR: 評価 ε で腕の順位が入れ替わる唯一の表](#btr-評価-ε-で腕の順位が入れ替わる唯一の表)
   - [評価 ε が比較を交絡させるという指摘](#評価-ε-が比較を交絡させるという指摘)
5. [批判と留保](#5-批判と留保)
6. [全体評価](#6-全体評価)
7. [調査の限界](#7-調査の限界)
8. [出典一覧](#8-出典一覧)

---

## 1. 容量の効果を分解した研究

### Fedus et al. (ICML 2020): 容量と最古方策年齢の直交分解

replay バッファの効果を「容量」と「最古方策の年齢」の 2 因子へ分解し、格子状に掃引した研究はこの 1 本のみである。年齢の単位は環境ステップではなく**学習器の勾配ステップ数**である。

```
The age of a transition stored in replay is defined to be the number of gradient steps taken by the learner since the transition was generated.
The age of the oldest policy represented in a replay buffer is the age of the oldest transition in the buffer.

リプレイに格納された遷移の「年齢」は、その遷移が生成されてから学習器が踏んだ勾配ステップ数として定義される。
リプレイバッファに含まれる最古方策の年齢とは、バッファ内の最古の遷移の年齢である。
```

[Fedus2020 (2020/07), §3.1 Independent factors of control]

この 2 因子と replay ratio は独立ではなく、2 つを決めると残り 1 つが決まる。

```
In particular, when the oldest policy is held fixed, increasing the replay capacity requires more transitions per policy, which decreases the replay ratio. When the replay capacity is held fixed, decreasing the age of the oldest policy requires more transitions per policy, which also decreases the replay ratio.

とくに、最古方策を固定したまま容量を増やすには方策あたりの遷移数を増やす必要があり、replay ratio は下がる。容量を固定したまま最古方策の年齢を下げる場合も方策あたりの遷移数が増え、やはり replay ratio が下がる。
```

[Fedus2020 (2020/07), §3.1]

測定は Atari 14 ゲーム × 3 seed、Dopamine 実装の Rainbow、容量 5 水準（0.1M〜10M）× 最古方策 4 水準（25k〜25M）の直積（うち replay ratio が最低の 2 設定は計算量の都合で除外）。結果は、最古方策を固定すれば容量は大きいほど良く、容量を固定すれば年齢は若いほど概ね良い。

**アルゴリズム依存が極めて強い。** 容量 1M → 10M で Rainbow は +28.7%（replay ratio 固定）/ +18.3%（最古方策固定）改善する一方、DQN は +0.1% / −0.4% で**まったく改善しない**。

### 容量 × n-step の相互作用

容量増の恩恵を受けられるかどうかを決めるのは n-step リターンだけである、というのが同論文の第 2 の柱である。

```
The only additive variant that materially improves with larger replay capacity is the DQN agent with n-step returns.
As predicted, a Rainbow agent stripped of n-step returns does not benefit with larger replay capacity, while the Rainbow agents stripped of other components still improve.
These results suggest that n-step returns are uniquely important in determining whether a Q-learning algorithm can improve with a larger replay capacity.
Another surprising finding is that prioritized experience replay does not significantly affect the performance of agents with larger memories.

大きなリプレイ容量で実質的に改善する唯一の加算変種は、n-step リターンを持つ DQN エージェントである。
予測どおり、n-step リターンを取り除いた Rainbow は大きなリプレイ容量の恩恵を受けない一方、他の成分を取り除いた Rainbow は依然として改善する。
これらの結果は、Q 学習アルゴリズムが大きなリプレイ容量で改善できるか否かを決める上で、n-step リターンが唯一無二に重要であることを示唆する。
もう一つの意外な発見は、優先度付き経験再生が大きなメモリを持つエージェントの性能に大きく影響しないことである。
```

[Fedus2020 (2020/07), §4.1 Additive and ablative experiments]

n-step がなぜ効くのかについて、同論文は 2 つの仮説を検証している。「縮小係数が小さくなり deadly triad が緩和される」説は**否定**され、「分散低減」説は sticky actions の ON/OFF で**部分的にのみ支持**された（確率性を切ると容量増の利得は一貫して縮むが消えはしない）[Fedus2020 (2020/07), §5.1・§5.2]。

n-step とリプレイの関係は BBF でも再登場し、リセット後に n を 10 → 3 へ指数減衰させる設計が採られている。

```
One of the surprising components of BBF is the use of an update horizon (n-step) that decreases exponentially from 10 to 3 over the first 10K gradient steps following each network reset.

BBF の意外な構成要素の一つは、各ネットワークリセット後の最初の 10K 勾配ステップにわたって更新ホライズン(n-step)を 10 から 3 へ指数的に減少させることである。
```

[Schwarzer2023 (2023/05), §3 Bigger, Better, Faster]

### 非単調性と文献間の不一致

バッファサイズの効果が非単調であるという報告は複数ある。Zhang &amp; Sutton は非線形関数近似では中間サイズが最良だとし、その機構を「鮮度と相関のトレードオフ」に帰した。

```
We hypothesize that there is a trade-off between the data quality and data correlation. With a smaller replay buffer, data tends to be more fresh however they are highly temporal correlated, while training a neural network often needs i.i.d. data. With a larger replay buffer, the sampled data tends to be uncorrelated, however they are more outdated.

データの鮮度とデータ相関の間にトレードオフがあると仮説を立てる。バッファが小さいほどデータは新鮮になるが時間相関が強く、ニューラルネットの学習は往々にして i.i.d. データを必要とする。バッファが大きいほどサンプルは無相関になるが、より古くなる。
```

[ZhangSutton (2017/12), §5.3 Non-linear Function Approximation]

**ただしこの結論は Fedus らが再現できていない。** 両者は正面から食い違う。

```
We also note the fixed replay ratio result disagrees with the conclusion in Zhang & Sutton 2017 that larger replay capacity is detrimental -- we instead observe no material performance change.

また、この replay ratio 固定の結果は、より大きなリプレイ容量は有害であるという Zhang & Sutton 2017 の結論とは食い違う。我々はむしろ実質的な性能変化を観測しない。
```

[Fedus2020 (2020/07), §3.3 Generalizing to other agents]

Liu &amp; Zou は Q 学習 + 経験再生の ODE モデルから解析解を導き、**非単調性の有無自体がミニバッチサイズに依存して切り替わる**ことを示した。

```
The learning performance is affected non-monotonically by the memory size for m < 20, while a monotonic relation is observed for m > 20, as shown in Fig. 1d.

図1d に示すように、m < 20 では学習性能はメモリサイズによって非単調に影響されるが、m > 20 では単調な関係が観測される。
```

[LiuZou (2017/10), §3 Effects of memory size]

Fedus らの格子内にも非単調性の例外がある。容量 10M では最古方策を 2.5M → 250k と若くすると性能が落ち、原因は Montezuma's Revenge と PrivateEye の 2 ゲーム（疎報酬・hard exploration）である [Fedus2020 (2020/07), §3.2]。

「格納量」と「実効的な記憶ホライズン」を別物として扱う方向は 2026 年に現れた。

```
By curating these end-points in a smaller recency buffer, our method maintains an effective memory horizon comparable to a standard large buffer while requiring an order of magnitude less storage.

これらの端点を小さな recency バッファに整理して保持することで、本手法は標準的な大バッファに匹敵する実効的な記憶ホライズンを維持しつつ、必要な記憶容量を一桁削減する。
```

[Panahi2026 (2026/07), Abstract]

同論文は、DQN 由来の既定値が事実上無検証で固定されてきたと指摘している。

```
Nearly every off-policy deep reinforcement learning (DRL) algorithm uses the replay recipe established by DQN: a FIFO buffer of the last one million transitions, sampled uniformly. This paper asks whether that buffer can be compressed by an order of magnitude or more without sacrificing performance.

ほぼすべての off-policy 深層強化学習アルゴリズムは、DQN が確立したリプレイのレシピ、すなわち直近 100 万遷移の FIFO バッファを一様サンプリングする方式を用いている。本論文は、そのバッファを性能を犠牲にせず一桁以上圧縮できるかを問う。
```

[Panahi2026 (2026/07), Summary / Contribution(s)]

### replay ratio / UTD スケーリングの系列

この系列は**バッファ容量ではなく replay ratio を主語**にしている点に注意が必要である。

| 文献 | 主変数 | 主要結果 |
|---|---|---|
| Nikishin2022 | replay ratio | RR が高いほど primacy bias の害が大きく、リセットの効果が大きい。**リセット時にバッファを保持することが決定的**で、空にすると強く有害 |
| DOro2023 | replay ratio | shrink-and-perturb 併用で RR を 16 まで性能劣化なしに上げられる |
| Schwarzer2023 | replay ratio | BBF は RR 8。SR-SPR との差が全 RR 域で線形に一定（約 0.45 IQM） |
| Rybkin2025 | UTD 比 | データ量と計算量の Pareto 前線を UTD 比が支配。**最適バッチサイズと学習率は UTD 比に反比例** |

Rybkin らは「絶対量ではなく比が不変量」という定式化に最も近いが、扱う比は UTD であってバッファ容量ではない。

```
We show that it is possible to account for the training dynamics unique to value-based RL, and are able to find the best hyperparameters by setting the batch size and learning rate inversely proportional to the UTD ratio.

価値ベース RL に固有の学習ダイナミクスを考慮することが可能であり、バッチサイズと学習率を UTD 比に反比例させることで最良のハイパーパラメータを見つけられることを示す。
```

[Rybkin2025 (2025/02), §1 Introduction]

---

## 2. 分散アクター構成における容量設計

### システム別の実数値

`capacity/actor` は本報告で容量を actor 数で割った**導出値**であり、出典が述べている数値ではない。

| システム | actors / envs | replay 容量 | capacity/actor（導出） | 比の根拠 |
|---|---|---|---|---|
| DQN (Mnih 2013/2015) | 1 | 1,000,000 frames | 1,000,000 | 無 |
| Rainbow | 1 | 1M transitions | 1,000,000 | 無 |
| **Ape-X DQN** | **360** | **2,000,000（soft limit）** | **約 5,556** | **部分的に有**（容量 sweep あり。比の根拠は無） |
| Ape-X DPG | 64 | 1,000,000（固定） | 15,625 | 無 |
| R2D2 | 256 | 4×10⁶ observations = 10⁵ sequences | 約 15,625 obs | 無 |
| NGU / Agent57 | 256 | 5e6（単位は表に明記無し） | 約 19,531 | 無 |
| MEME | 64 actors × 2 threads | 2×10⁵ trajectories | 約 1,563 traj/env | 無 |
| DreamerV3 | 16（既定） | 5e6 | 312,500 | 無 |
| **BTR** | **64** | **1,048,576（2²⁰）** | **16,384** | **無**（README に RAM 注意のみ） |
| Fast &amp; Data-Efficient Rainbow | 64 | 1M | 15,625 | 無 |
| **PQL** | **4,096（既定。256〜16,384 を sweep）** | **5×10⁶** | **約 1,221** | **有**（専用 ablation 節） |
| CleanRL `dqn_atari.py` | 1（`assert num_envs == 1`） | 1,000,000 | 1,000,000 | 無 |
| **Stable-Baselines3 DQN** | 任意 `n_envs` | `buffer_size // n_envs` | **1,000,000 / n_envs** | **有**（実装 + maintainer 回答） |
| Dopamine JaxDQN | 1 | 1,000,000 | 1,000,000 | 無 |
| Acme R2D2 | 4 (local) / 80 (distributed) | 100,000 sequences | 1,250 seq/actor | 無 |
| RLlib Ape-X DQN (ray 2.9) | 32 | 2,000,000 | 62,500 | 無 |

**19 件中、容量と actor 数の比に明示的な根拠を述べているのは PQL と Stable-Baselines3 の 2 件のみ。** 他は数値がハイパーパラメータ表に載るだけで、選定理由の記述が無い。

### Ape-X: 比例させないことの明示と対照実験

Ape-X は付録で、容量を actor 数に比例させていないこと、その帰結として actor を増やすと replay の入れ替えが速くなることを明言している。

```
In our main experiments we do not change the size of the replay memory in proportion to the number of actors, so by changing the number of actors we also increased the rate at which the contents of the replay memory is replaced. This means that in the experiments with more actors, transitions in the replay memory are more recent: they are generated by following policies whose parameters are closer to version of the parameters being optimized by the learner, and in this sense they are more on-policy.

主実験では replay memory のサイズを actor 数に比例させて変えてはいない。したがって actor 数を変えることで replay memory の内容が置き換わる速度も同時に増加している。これは、actor が多い実験では replay memory 内の transition がより新しいことを意味する。すなわち、learner が最適化しているパラメータにより近いパラメータのポリシーによって生成されており、その意味でより on-policy である。
```

[ApeX (2018/03), Appendix]

彼らはさらに、「置換速度（recency）だけが効いているのか」を切り分ける対照実験を行っている。32 actor の各遷移を 8 回複製して 256 actor と同じ置換速度を再現したが、性能は回復しなかった。

```
We observe (see Figure 6) that this does not recover the same performance, and therefore conclude that the recency of the experience alone is not sufficient to explain the performance of our method.

(図 6 の通り)これでは同じ性能を回復しないことを観測した。したがって、経験の新しさだけでは本手法の性能を説明できないと結論する。
```

[ApeX (2018/03), Appendix]

R2D2 は同じ論点を「parameter lag」の座標で定量化している。actor 数がバッファ内データの鮮度を直接支配する、という実測である。

```
Specifically, in our experiments, as the number of actors is changed from 256 to 64, the mean parameter lag goes from 1500 to approximately 5500 parameter updates, which in turn impacts the magnitude of representation drift and recurrent state staleness.

具体的に我々の実験では、アクター数を 256 から 64 に変えると、平均パラメータ遅延は 1500 からおよそ 5500 パラメータ更新へと変化し、それが表現ドリフトと再帰状態陳腐化の大きさに影響する。
```

[R2D2 (2019/05), §4]

### Stable-Baselines3: 実装による比の固定

SB3 は「`buffer_size` は総 transition 数」という規約をコードで強制している。結果として **1 env あたりの履歴長は `n_envs` に反比例する**。

```python
self.buffer_size = max(buffer_size // n_envs, 1)

self.buffer_size = max(buffer_size // n_envs, 1)
```

[SB3-buffers (2024–), `stable_baselines3/common/buffers.py` ReplayBuffer.__init__]

maintainer の回答が、この設計についての数少ない明示的説明である。

```
yes ... so at the end, the number of transitions stored is the same, you should not need to do anything.
About the size of the replay buffer in general: There is no recommended answer for that.

そのとおり ... 最終的に格納される transition 数は同じなので、何もする必要はない。
replay バッファのサイズ一般について: それについて推奨される答えは無い。
```

[SB3-issue1885 (2024/04), araffin のコメント]

### PQL: 比例スケーリングは不要という結論

PQL は問題設定そのものを「容量 ÷ 並列環境数 = バッファが保持する env step 数」として提示した唯一の論文である。

```
For instance, if there are 10,000 parallel environments and we still use the typical replay buffer capacity (say 1M samples), the entire replay buffer is refreshed every 100 environment steps, making the data in the replay buffer more like the data collected from an on-policy method. Do off-policy methods still retain their data efficiency in this scenario?

例えば並列環境が 10,000 あり、なお典型的なリプレイバッファ容量(仮に 1M サンプル)を使うとすると、リプレイバッファ全体が 100 環境ステップごとに入れ替わり、バッファ内のデータは on-policy 手法で集めたデータに近くなる。このシナリオでも off-policy 手法はデータ効率を保つのか。
```

[PQL (2023/07), §1 Introduction]

結論は否定的、すなわち比例スケーリングは不要である。

```
Even though the number of environments is 1000x more, we did not find it necessary to use a replay buffer that is 1000x bigger. In fact, a replay buffer with a capacity of 5M transitions is sufficient for our experiments even with 16843 parallel environments.

環境数が 1000 倍になっても、1000 倍大きいリプレイバッファが必要だとは分からなかった。実際、16843 並列環境であっても、容量 5M 遷移のリプレイバッファで我々の実験には十分である。
```

[PQL (2023/07), §6 Conclusion]

ただし彼らの対象は Isaac Gym の連続制御であり、Atari ではない。また掃引した容量の下端は 1M で、「やや劣る」と報告されている [PQL (2023/07), §4.4.4]。

---

## 3. エピソード単位 replay とエピソード長

### シーケンス replay とエピソード境界

R2D2 は遷移タプルではなく固定長シーケンスを格納し、エピソード境界を跨がないと明記する。容量も遷移数ではなくシーケンス本数で表記される。

```
Instead of regular (s, a, r, s') transition tuples, we store fixed-length (m = 80) sequences of (s, a, r) in replay, with adjacent sequences overlapping each other by 40 time steps, and never crossing episode boundaries.

通常の (s, a, r, s') 遷移タプルの代わりに、我々は (s, a, r) の固定長 (m = 80) シーケンスを replay に格納する。隣接するシーケンスは互いに 40 タイムステップ重なり、エピソード境界を跨ぐことは決してない。
```

[R2D2 (2019/05), §2.3]

同論文は「エピソード軌跡をまるごと replay する」戦略を検討したうえで退けており、理由は可変長がもたらす実務・計算・アルゴリズム上の問題と、軌跡内の状態の強い相関による更新分散の増大である [R2D2 (2019/05), §3]。

R2D2 は 1 本の格納シーケンス内部での表現の陳腐化を、burn-in を除いた部分の先頭 (i = l) と末尾 (i = l + m − 1) の Q 値差 ΔQ で直接測っている。結果は、zero start state の悪影響がシーケンス末尾へ向かって減衰するが完全には消えない、というものである [R2D2 (2019/05), §3]。

**ただし R2D2 論文は、容量（10⁵ シーケンス）と最大エピソード長（108,000 フレーム）の両方を与えながら、両者を関係づける記述を持たない。** MEME も容量を「2×10⁵ trajectories」と書くが、同様にエピソード長との関係は論じていない。

エピソード記憶系（MFEC / NEC / EMDQN）では「エピソード」はモンテカルロ収益を確定させ逆順で書き戻す単位であり、**メモリ容量はエントリ数（状態-行動キー数）であってエピソード本数ではない**。3 論文とも容量とエピソード長の関係は述べていない。

### エピソード長でバッファ容量を正規化した唯一の実例

本調査で見つかった、バッファ容量をエピソード長で明示的に割っている実装は 1 件である。

```python
# number of episodes which can be stored until buffer size is reached
self.max_episode_stored = self.buffer_size // self.max_episode_length

# バッファサイズに達するまでに格納できるエピソード数
self.max_episode_stored = self.buffer_size // self.max_episode_length
```

[SB3-HER (2026/09 参照), `stable_baselines3.her.her_replay_buffer` ソース]

HER 本体（論文）では、エピソード境界が goal 再サンプリング戦略の分界線として機能する。**エピソード内に閉じた戦略（future / episode / final）は機能し、バッファ全体から引く random は失敗する**というのが ablation の核心である [HER (2017/07), §4.5]。

### エピソード境界の扱いがバッファサイズ感度を左右する

Pardo らは、タイムアウト終端を環境本来の終端と区別してブートストラップを継続すれば、Zhang &amp; Sutton の「大きいバッファは有害」がほぼ消えることを示した。

```
Finally, we demonstrate that the negative impact of large experience replay buffers shown by Zhang & Sutton (2017) can often be vastly reduced if timeout terminations are properly handled.

最後に、Zhang & Sutton (2017) が示した大きな経験 replay バッファの負の影響が、タイムアウト終端を適切に扱えばしばしば大幅に軽減できることを我々は実証する。
```

[Pardo (2017/12), §1 Introduction]

この実験は、**エピソード長 T = 200 に対してバッファサイズを 40 / 60 / 100 / 1M と振っている**。すなわちバッファが 1 エピソードより短い領域を含む、本調査で唯一の設定である [Pardo (2017/12), Figure 7]。

### 軌跡沿いの価値伝播

EBU は replay メモリからエピソードを 1 本まるごとサンプルし、その全遷移を逆順に伝播させる。EVA は replay バッファに後続タプルへのポインタを持たせ、軌跡中心の価値推定を作る。TER は軌跡を共通状態で縫い合わせてグラフを構成し、終端から逆向き幅優先で更新する。

EVA は動機づけとして、遷移タプルのみを保持することの問題を列挙している。

```
(ii) the replay buffer is of limited size and experience tuples are regularly removed (thus limiting the opportunity for gradient descent to learn from it), (iii) training from experience tuples neglects the trajectory nature of an agents experience

(ii) replay バッファのサイズは有限で経験タプルは定期的に削除される（したがって勾配降下がそこから学ぶ機会が制限される）、(iii) 経験タプルからの学習はエージェントの経験の軌跡的性質を無視する
```

[EVA (2018/10), §1 Introduction]

**これら 3 手法はいずれも「軌跡構造がバッファ内に保持されている」ことを前提として設計されており、その前提を崩す（軌跡が分断される・一部が追い出される）実験は報告していない。**

---

## 4. 評価 ε が結論を変える現象

### 評価 ε の値の系譜

| 値 | 出所 | 述べられた理由 |
|---|---|---|
| 0.05 | DQN Nature (2015) | **有**:「評価中の過学習の可能性を最小化するため」 |
| 0.001 | Double DQN tuned (2015) | 実質無:「これらの変更はそれぞれ性能を改善した」 |
| 0.01 | Machado et al. (2018) の DQN 実験 | **有**: 評価フェーズが存在しないため |
| 0.001（既定化） | Dopamine (2018) | 無（「最適でなく一貫したベースラインを提供する意図」） |

DQN Nature の理由づけは、調査範囲で最古かつ最も影響力のある記述である。

```
The trained agents were evaluated by playing each game 30 times for up to 5 min each time with different initial random conditions ('no-op'; see Extended Data Table 1) and an ε-greedy policy with ε = 0.05. This procedure is adopted to minimize the possibility of overfitting during evaluation.

学習済みエージェントは、各ゲームを 30 回、1 回あたり最大 5 分、異なる初期ランダム条件('no-op')と ε = 0.05 の ε-greedy 方策でプレイさせて評価した。この手続きは、評価中の過学習の可能性を最小化するために採用している。
```

[Mnih2015 (2015/02), Methods "Evaluation procedure"]

ここでの「過学習」は、決定的な ALE で開始状態が固定されると同一軌道を暗記できてしまうことを指す文脈で使われている。つまり ε は方策の質を測るためではなく**決定性を壊すため**に入っている。

**評価 ε が測定値を実質的に動かすことは、標準プロトコル論文自身が述べている。**

```
Importantly, ϵ-greedy policies tend to perform better than greedy policies in the ALE (Bellemare et al., 2013; Mnih et al., 2015). Therefore, this protocol does not necessarily benefit from turning off exploration during evaluation.

重要なことに、ALE では ϵ-greedy 方策は greedy 方策より良い性能を出す傾向がある。したがって、このプロトコルは評価時に探索を切ることで必ずしも得をするわけではない。
```

[Machado2018 (2017/12), §4.1 "Evaluation after learning"]

同論文は、一様ランダム行動ノイズ（= 評価 ε と同型）の害がゲーム依存であることも認めている。

```
- May significantly interfere with agent's policy, e.g., when navigating a narrow cliff such as in the game Q*bert.

- エージェントの方策を著しく妨げうる。例えば Q*bert のような狭い崖を進む場合。
```

[Machado2018 (2017/12), §5 選択肢比較]

なお PER 論文の比較表は、**同一表内でエージェントごとに評価 ε が 50 倍違う**（DQN baseline 0.05、DDQN tuned 系 0.001）状態で並んでおり、その影響は議論されていない [Schaul2016 (2015/11), Appendix B.2 Table 5]。

### BTR: 評価 ε で腕の順位が入れ替わる唯一の表

Clark らは ablation を ε = 0 / 0.01 / 0.03 の 3 通り + ColorJitter で評価し、action gap・action swaps・policy churn と同じ 1 枚の表に並べている。本調査で見つかった、**評価 ε を独立変数として系統的に振った唯一の例**である。

```
Furthermore, we find that maxpooling produces a more robust policy. To test this, we evaluate the performance of BTR's ablations when taking different quantities of ε-actions and with altered observations and find maxpooling alleviates some of the performance loss (Table 2).

さらに、maxpooling がよりロバストな方策を生むことを見出した。これを検証するため、BTR の ablation について異なる量の ε-action を取った場合と観測を変化させた場合の性能を評価し、maxpooling が性能低下の一部を緩和することを見出した(Table 2)。
```

[BTR2025 (2024/11), §5.2]

Table 2 の抜粋（Atari Phoenix、200M frames、3 seed 平均）:

| 評価条件 | BTR | w/o Maxpool |
|---|---|---|
| Score ε = 0 | 330k | **406k** |
| Score ε = 0.01 | **194k** | 171k |
| Score ε = 0.03 | **94k** | 86k |
| Score ColorJitter | **212k** | 187k |
| Action Gap | 0.282 | 0.264 |
| % Action Swaps | 36.6% | 39.3% |
| Policy Churn | 3.8% | 4.2% |

（値は [BTR2025 (2024/11), Table 2 / Appendix Table E7・E8] より抽出）

**ε = 0 では ablation 版が本体を上回り、ε > 0 では本体が上回る。** ただし 95% 信頼区間は重なっており（ε=0 で [332k,479k] 対 [282k,377k]）、著者自身は「順位反転」という表現を用いていない。

### 評価 ε が比較を交絡させるという指摘

Nagarajan らは、評価時の探索が比較の妥当性を損なうと明言し、代替として greedy + 100 個の開始状態を提案した。調査範囲で唯一の明示的批判である。

```
However, in these evaluations, we are unable to attribute performance differences between the agents solely to differences between their Q-networks, since exploration can confound results. Even if exploration is seeded in the evaluation stage, a single deviation between policies will desynchronize the exploration seeds.

しかしそのような評価では、探索が結果を交絡させうるため、エージェント間の性能差をその Q ネットワークの違いだけに帰属させることができない。評価段階で探索にシードを与えたとしても、方策間の 1 回の逸脱で探索シードは同期を失う。
```

[Nagarajan2018 (2018/09), Appendix]

評価 ε ではないが、評価プロトコルの差が手法の順位を覆した査読済み実例は存在する。

```
In other words, this gap in evaluation procedures resulted in CURL being assessed as achieving a greater true median than DER, where our experiment gives strong support to DER being superior.

言い換えれば、この評価手続きのギャップにより CURL が DER より高い真の中央値を達成していると評価されてしまったが、我々の実験は DER のほうが優れていることを強く支持している。
```

[Agarwal2021 (2021/08), §3]

また Schaul らは、greedy 方策そのものが安定した測定対象ではないと報告している。

```
As a coarse magnitude for the impatient reader: in a typical run of DQN on Atari, the greedy policy changes in ≈10% of all states after a single gradient update.

せっかちな読者のための大まかな大きさ:Atari 上の DQN の典型的な run では、1 回の勾配更新の後に全状態の約 10% で greedy 方策が変わる。
```

[Schaul2022 (2022/06), §1 脚注]

理論側では、評価 ε を訓練 ε より小さくすることに根拠を与えた命題が 1 件ある。

```
Proposition 3. Let γ ∈ [0,1), ε' ∈ [0,ε], π0 be a policy, and πε be the ε-optimal policy w.r.t π0. Then, v^{π0} ≤ v^{(πε,ε)} ≤ v^{(πε,ε')}, with equality iff v^{π0} = v*.

命題 3。γ ∈ [0,1)、ε' ∈ [0,ε]、π0 を方策、πε を π0 に関する ε-最適方策とする。このとき v^{π0} ≤ v^{(πε,ε)} ≤ v^{(πε,ε')} が成り立ち、等号は v^{π0} = v* のときに限る。
```

[Shani2019 (2018/12), Proposition 3]

---

## 5. 批判と留保

**文献間の直接的な不一致が 1 件ある。** Zhang &amp; Sutton の「大容量は有害」を Fedus らは Atari / Rainbow スケールで再現できなかった（§1）。Pardo らはこの不一致にタイムアウト終端の扱いという第 3 の説明を与えている（§3）。どれが正しいかは本調査では決着しない。

**BTR の順位反転は統計的に確立していない。** 95% 信頼区間が重なっており、著者自身も順位反転として主張していない（§4）。3 seed である。

**Ape-X の対照実験の解釈には注意が要る。** 「遷移を 8 回複製して置換速度を揃えたが性能は戻らなかった」から「recency 単独では説明できない」を導いているが、著者自身が「複製は容量を減らすことと同様の効果を持つ」とも注記しており、対照が完全に交絡を落としているとは言い切れない [ApeX (2018/03), Appendix]。

**PQL の結論の適用範囲は限定的である。** Isaac Gym の連続制御であり、Atari ではない。エピソード長との関係も論じていない。

**Fedus らの「年齢」は勾配ステップ単位であり、環境ステップ単位ではない。** この座標系では、actor 数を変えたときに per-actor 履歴がどう変わるかは表現できない。

---

## 6. 全体評価

**調査対象の問い — `replay_capacity / num_envs` とエピソード長の比が成績を決めるか — を正面から扱った研究は見つからなかった。** 最も近い 3 件はいずれも部分的である。

| 文献 | どこまで近いか | 何が欠けているか |
|---|---|---|
| PQL (2023) | 容量 ÷ 並列環境数を「何 env step 分か」として定式化し掃引した | 単位がバッファ全体。エピソード長との関係は論じない。Isaac Gym 連続制御 |
| SB3 HER（実装） | `buffer_size // max_episode_length` でエピソード長正規化を実装 | 論文化されておらず、根拠の記述も無い。HER 固有 |
| Ape-X (2018) | 容量を actor 数に比例させないと明記し、置換速度の対照実験を行った | 単位がバッファ全体。actor 単位の履歴という概念が無い |

**主流の座標系は 2 つあり、どちらも本論点を表現できない。** Fedus 系は（容量、最古方策年齢 [勾配ステップ]）、replay ratio 系は（UTD 比）である。前者は actor 数を変えても両軸とも動かず、後者は replay ratio が actor 数と独立なため動かない。

**業界の既定は「総容量を固定し、per-actor 履歴を actor 数に反比例させる」である。** SB3 はこれをコードで強制し、Ape-X は明示的にそう設計したと述べている。この既定が妥当かを、per-actor 履歴という軸で検証した文献は無い。

**評価 ε については、標準プロトコル論文自身が「ALE では ε-greedy のほうが greedy より良い傾向がある」と述べている。** つまり評価 ε は測定値を動かすと分かったうえで、値の選定理由はほぼ慣例に委ねられている。ε を独立変数として振った例は BTR の 1 表のみである。

---

## 7. 調査の限界

1. **原典 PDF を取得できなかった文献がある。** D'Oro et al. (SR-SPR) は OpenReview の CAPTCHA により本文未取得で、ICLR 公式 Abstract と後続論文 BBF の記述に依拠している。R2D2 は Wayback Machine 経由で本文を取得できたため引用は確認済みだが、OpenReview 直接アクセスは失敗している。
2. **被引用数を取得できなかった文献がある。** PQL は Semantic Scholar API のレート制限により未取得。
3. **venue の確認が未了の文献がある。** Toromanoff et al. (SABER) は Semantic Scholar が NeurIPS と記載するが、会議採択の一次確認をしていない。MEME は arXiv プレプリントで、Semantic Scholar は ICLR と記録している。
4. **日本語文献を系統的には探索していない。** 英語クエリを主とし、日本語は補助的にしか当たっていない。
5. **エンジニアリングブログ・社内技術記事の体系的な探索はしていない。** 本報告に含まれる低ティア出典（GitHub issue 2 件）は英語クエリで偶然浮上したものである。
6. **「per-actor 履歴」という語彙が確立していないため、検索で取りこぼしている可能性がある。** 本調査では per-actor trajectory / capacity per actor / replay buffer refresh rate などを試したが、別の呼称で同じ概念を扱った文献がある可能性は排除できない。

---

## 8. 出典一覧

[ApeX, 2018/03] Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel, M., van Hasselt, H., Silver, D. (DeepMind). "Distributed Prioritized Experience Replay." ICLR 2018. 被引用 845. https://arxiv.org/abs/1803.00933

[Agarwal2021, 2021/08] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. G. (Google Research Brain Team / Mila). "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021. 被引用 1,015. https://arxiv.org/abs/2108.13264

[BTR2025, 2024/11] Clark, T., Towers, M., Evers, C., Hare, J. (University of Southampton). "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML 2025. 被引用 10. https://arxiv.org/abs/2411.03820

[DOro2023, 2023/05] D'Oro, P., Schwarzer, M., Nikishin, E., Bacon, P.-L., Bellemare, M. G., Courville, A. (Mila). "Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier." ICLR 2023 (Oral). 被引用 167. https://openreview.net/forum?id=OpC-9aBBVJe

[EVA, 2018/10] Hansen, S. S., Sprechmann, P., Pritzel, A., Barreto, A., Blundell, C. (DeepMind). "Fast deep reinforcement learning using online adjustments from the past." NeurIPS 2018. 被引用 49. https://arxiv.org/abs/1810.08163

[Fedus2020, 2020/07] Fedus, W., Ramachandran, P., Agarwal, R., Bengio, Y., Larochelle, H., Rowland, M., Dabney, W. (MILA / Google Brain / DeepMind / CIFAR). "Revisiting Fundamentals of Experience Replay." ICML 2020. 被引用 329. https://proceedings.mlr.press/v119/fedus20a.html

[HER, 2017/07] Andrychowicz, M., Crow, D., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J. (OpenAI), Abbeel, P., Zaremba, W. "Hindsight Experience Replay." NeurIPS 2017. 被引用 2,879. https://proceedings.neurips.cc/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf

[LiuZou, 2017/10] Liu, R., Zou, J. (Stanford University). "The Effects of Memory Replay in Reinforcement Learning." Allerton 2018 (IEEE). 被引用 134. https://arxiv.org/abs/1710.06574

[Machado2018, 2017/12] Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M. (University of Alberta / Google / Franklin &amp; Marshall / DeepMind / Microsoft Research). "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." JAIR 61:523–562. 被引用 630. https://arxiv.org/abs/1709.06009

[Mnih2015, 2015/02] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (Google DeepMind). "Human-level control through deep reinforcement learning." Nature 518:529–533. 被引用 33,140. https://doi.org/10.1038/nature14236

[Nagarajan2018, 2018/09] Nagarajan, P. (Preferred Networks), Warnell, G. (U.S. Army Research Laboratory), Stone, P. (UT Austin). "Deterministic Implementations for Reproducibility in Deep Reinforcement Learning." arXiv:1809.05676. 被引用 63. https://arxiv.org/abs/1809.05676

[Nikishin2022, 2022/05] Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., Courville, A. (Mila, Université de Montréal). "The Primacy Bias in Deep Reinforcement Learning." ICML 2022. 被引用 324. https://proceedings.mlr.press/v162/nikishin22a.html

[Panahi2026, 2026/07] Panahi, P. M., Ashrafi, A., Du, H., Patterson, A., White, M., White, A. (University of Alberta / Amii / CIFAR AI Chair). "Endpoint Replay: Compressing the Recency Buffer in Deep Reinforcement Learning." RLC 2026 / Reinforcement Learning Journal 2026. 被引用 0. https://arxiv.org/abs/2607.25123

[Pardo, 2017/12] Pardo, F., Tavakoli, A., Levdik, V., Kormushev, P. (Robot Intelligence Lab, Imperial College London). "Time Limits in Reinforcement Learning." ICML 2018. 被引用 202. https://arxiv.org/abs/1712.00378

[PQL, 2023/07] Li, Z., Chen, T., Hong, Z.-W., Ajay, A., Agrawal, P. (Improbable AI Lab, MIT). "Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation." ICML 2023. 被引用 36. https://proceedings.mlr.press/v202/li23f.html

[R2D2, 2019/05] Kapturowski, S., Ostrovski, G., Quan, J., Munos, R., Dabney, W. (DeepMind, London). "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR 2019. 被引用 582. https://openreview.net/forum?id=r1lyTjAqYX

[Rybkin2025, 2025/02] Rybkin, O., Nauman, M., Fu, P., Snell, C., Abbeel, P., Levine, S. (UC Berkeley), Kumar, A. (Carnegie Mellon University). "Value-Based Deep RL Scales Predictably." ICML 2025. 被引用 13. https://arxiv.org/abs/2502.04327

[SB3-buffers, 2024–] DLR-RM/stable-baselines3. `stable_baselines3/common/buffers.py`. GitHub. https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

[SB3-HER, 2026/09 参照] Stable-Baselines3 documentation (v1.6.2). `stable_baselines3.her.her_replay_buffer`. https://stable-baselines3.readthedocs.io/en/v1.6.2/_modules/stable_baselines3/her/her_replay_buffer.html

[SB3-issue1885, 2024/04] araffin (SB3 maintainer). DLR-RM/stable-baselines3 Issue #1885. https://github.com/DLR-RM/stable-baselines3/issues/1885

[Schaul2016, 2015/11] Schaul, T., Quan, J., Antonoglou, I., Silver, D. (Google DeepMind). "Prioritized Experience Replay." ICLR 2016. 被引用 4,607. https://arxiv.org/abs/1511.05952

[Schaul2022, 2022/06] Schaul, T., Barreto, A., Quan, J., Ostrovski, G. (DeepMind). "The Phenomenon of Policy Churn." NeurIPS 2022. 被引用 37. https://arxiv.org/abs/2206.00730

[Schwarzer2023, 2023/05] Schwarzer, M., Obando-Ceron, J., Courville, A., Bellemare, M. G., Agarwal, R., Castro, P. S. (Mila / Google DeepMind). "Bigger, Better, Faster: Human-level Atari with human-level efficiency." ICML 2023. 被引用 176. https://proceedings.mlr.press/v202/schwarzer23a.html

[Shani2019, 2018/12] Shani, L., Efroni, Y., Mannor, S. (Technion). "Exploration Conscious Reinforcement Learning Revisited." ICML 2019. https://proceedings.mlr.press/v97/shani19a.html

[ZhangSutton, 2017/12] Zhang, S., Sutton, R. S. (University of Alberta). "A Deeper Look at Experience Replay." arXiv:1712.01275. 被引用 327. https://arxiv.org/abs/1712.01275

被引用数は Semantic Scholar Graph API で 2026-09-17 に取得した値（PQL を除く）。
