# Survey: Atari の実時間効率フロンティア — 単一 GPU / デスクトップ 1 台という制約下で

Date: 2026-09-09
Scope: 「Atari (ALE) を、単一 GPU / デスクトップ 1 台という制約下で学習するとき、**同じ品質へ到達するまでの壁時計時間 (wall-clock time)** はいま何が最良か」を文献横断で確定させる。アルゴリズム縛りは置かず、価値ベース / 方策勾配 / モデルベース / replay の有無を問わない。軸はサンプル効率ではなく、**ハードウェアを明記した実時間**である。調査対象は論文原典・公式リポジトリ・公式ドキュメント・書誌データベース。第三者記事とコミュニティ情報は信頼度の階層を分けて別節に置く。

本レポートは「ある手法が速い」ことよりも先に、**そもそもその数値が比較可能か**を確定させることに紙幅を割いている。この分野の公開数値は、単位・プロトコル・測定者バイアスの 3 点で高い頻度で比較不能だからである。

## 目次

1. [数値を読む前に必要な 4 つの前提](#1-数値を読む前に必要な-4-つの前提)
   - frames と agent step / FPS の単位が文献ごとに 4 倍ずれる / プロトコル差 / 200M frames はプロトコルであって必要計算量ではない
2. [壁時計時間の開示状況 — 何が例外で、何が慣行か](#2-壁時計時間の開示状況--何が例外で何が慣行か)
   - 抄録レベルの開示率 / 本文レベルの手検査 / Rainbow が明文化した開示回避の論拠 / 評価プロトコル原典の指標設計
3. [単一 GPU・1 台構成で壁時計時間が報告されている研究](#3-単一-gpu1-台構成で壁時計時間が報告されている研究)
   - Stooke & Abbeel 2018 / rlpyt / Schmidt & Schmied / Daley & Amato / BTR / 一覧表
4. [PQN — replay buffer を捨てて並列 env で回す系統](#4-pqn--replay-buffer-を捨てて並列-env-で回す系統)
   - Table 3 / ハードウェア / 評価プロトコル / 倍率主張の分母 / 第三者による批判と独立計測
5. [高速 env 実装のスループット](#5-高速-env-実装のスループット)
   - EnvPool / Sample Factory / ALE 本体の AtariVectorEnv / JAX 系が実 ALE を扱っているか / 数値が存在しない実装群
6. [ALE を GPU 側で回す試み](#6-ale-を-gpu-側で回す試み)
   - CuLE / 保守状況 / ALE 公式の「XLA GPU support」の実体 / 新規実装の区別 / 判定
7. [サンプル効率志向の手法群は実時間軸でどこに来るか](#7-サンプル効率志向の手法群は実時間軸でどこに来るか)
   - 対応表 / BBF / SR-SPR / SPR / EfficientZero / DreamerV3 / MEME / replay ratio と壁時計時間のトレードオフ
8. [分散前提の手法を比較対象から外してよいか](#8-分散前提の手法を比較対象から外してよいか)
   - 判定表 / Podracer Sebulba / 実装構成とアルゴリズムの区別
9. [2025〜2026 の動向](#9-20252026-の動向)
   - RISE / Squeezing More from the Stream / Greener Deep RL / Hyperbolic RL / TARL / Physical Atari
10. [コミュニティ signals](#10-コミュニティ-signals)
11. [批判・懸念](#11-批判懸念)
    - 出典間の数値の食い違い / 測定者バイアス / ハード世代差 / 「Atari」を名乗るが Atari でないもの
12. [総合評価](#12-総合評価)
    - 発注の 6 つの問いへの回答 / 参照点を BTR のままにしてよいか
13. [調査限界](#13-調査限界)
14. [付録A: 発注元プロジェクトの実測（調査対象外の自プロジェクト情報）](#14-付録a-発注元プロジェクトの実測調査対象外の自プロジェクト情報)
15. [出典](#15-出典)

---

## 1. 数値を読む前に必要な 4 つの前提

### 1.1 frames と agent step は 4 倍違う

Atari の慣行では action repeat (frame skip) が 4 であり、「200M frames」と「50M agent step」は同じ量を指す。TorchBeast がこの対応を最も端的に定義している。

```
For the agent model, we use a version of the "deep network" without an LSTM from the IMPALA paper and train for 200 million frames per environment and run (corresponding to 50 million "agent steps" due to action repetitions).

エージェントモデルには IMPALA 論文の LSTM なし版「deep network」を用い、環境・実行あたり 2 億フレーム学習する (action repetition により 5000 万「agent steps」に相当)。
```

[TorchBeast (2019/10), §4 Experiments]

Acme も同じ関係を明示している。

```
In the Atari environment, since we repeat actions 4 times, as is now standard, our measurement of actor steps is exactly four times smaller than the number of environment frames, which is a more common measure in the ALE literature.

Atari 環境では、現在標準となっているように行動を 4 回繰り返すため、我々の actor steps の計測値は、ALE 文献でより一般的な指標である environment frames の数のちょうど 4 分の 1 になる。
```

[Acme (2020/06), 実験設定節]

### 1.2 「FPS」の単位は文献ごとに 4 倍ずれる

本調査で最も実害の大きい落とし穴がこれである。同じ「100,000 FPS」が、文献によって 4 倍違う実仕事量を指す。

EnvPool は公式ドキュメントで、報告 FPS に frame_skip を掛けていることを明示している。

```
To align with other baseline results, FPS is multiplied with `frame_skip` (4 for Atari and 5 for Mujoco).

他のベースライン結果と揃えるため、FPS には `frame_skip` (Atari では 4、MuJoCo では 5) が掛けられている。
```

[EnvPool 公式ドキュメント Benchmark (2026/09 取得), Benchmark ページ冒頭]

CuLE も同様に生フレームで報告し、学習に使うフレームはその 1/4 だと脚注で述べている。

```
Raw frames are reported here and in the rest of the paper, unless otherwise specified. These are the frames that are actually emulated, but only 25% of them are rendered and used for training. Training frames are obtained by dividing the raw frames by 4—see also [6].

特記しない限り、本稿では以降 raw フレーム数を報告する。これは実際にエミュレートされたフレームであり、そのうち 25% だけが描画されて学習に使われる。学習フレーム数は raw フレーム数を 4 で割って得る ([6] も参照)。
```

[CuLE (2020/12), p.2 脚注2]

Cleanba は SPS と FPS を同一文で併記しており、対応関係が最も明快である。

```
When using 128 GPUs, the agent has an SPS of 403253, translating to over 1.6M FPS in Breakout.

128 個の GPU を使うと、エージェントの SPS は 403253 となり、Breakout において 160 万 FPS 超に相当する。
```

[Cleanba (2023/09), スケーリング節]

一方で、**単位の記載自体がない比較表**も存在する。TorchRL の Table 5 は Breakout-v5 で 97〜19,401 という 200 倍の幅を持つライブラリ横断比較でありながら、キャプションにも表ヘッダにも単位が書かれていない (同論文の Table 3 はヘッダに "fps" と明記しているのと対照的である)。

さらに、SEED RL は同一論文内の 2 つの図キャプションで frames と steps の関係を**互いに逆向きに**書いている。

```
Right: X-axis is environment frames (a frame is 1/4th of an environment step due to action repeat).

右: X 軸は environment frames (action repeat により、1 フレームは 1 environment step の 1/4 である)。
```

[SEED RL (2019/10), Figure 6 キャプション]

```
Top Row: X-axis is per frame (number of frames = 4x number of steps).

上段: X 軸はフレーム単位 (フレーム数 = ステップ数の 4 倍)。
```

[SEED RL (2019/10), Figure 5 キャプション]

**単位は出典を読めば必ず分かる、とは限らない。**

### 1.3 プロトコルが違うとスコアは比較できない

sticky actions / noop reset / episodic life / ライフ情報の扱いが違えば、同じゲームのスコアを並べても意味を持たない。BTR がこれを明示的に述べている。

```
Some prior works choose to pass life information to the agent (Schmidt & Schmied, 2021). To clarify, this is different to terminal on loss of life. Life information does not reset the episode upon losing a life, but does pass a terminal to the buffer, allowing the agent to experience further into episodes while also giving the agent a negative signal for losing a life. This setting is not recommended in Machado et al. (2018), and works which use it are not comparable to those which don't.

一部の先行研究はエージェントにライフ情報を渡すことを選ぶ (Schmidt & Schmied, 2021)。明確にしておくと、これはライフ喪失時の terminal とは異なる。ライフ情報はライフを失ってもエピソードをリセットしないが、terminal をバッファに渡すことで、エージェントがエピソードのより先まで経験できるようにしつつ、ライフ喪失に対する負のシグナルも与える。この設定は Machado et al. (2018) では推奨されておらず、これを用いる研究は用いない研究と比較可能ではない。
```

[BTR (2025/05), Appendix I Altered Atari Environment Settings]

この論点は §4.3 で PQN に対して具体的に効いてくる。

### 1.4 200M frames は「必要な計算量」ではなくプロトコルである

Machado et al. は、200M frames という予算が DQN との比較を容易にするための慣行であると述べている。すなわち、この本数を変えると別ベンチマークになる。

```
Currently, it is fairly standard to train agents for 200 million frames, in order to facilitate comparison with the DQN results. This is equivalent to approximately 38 days of real-time gameplay and even at fast frame rates represents a significant computational expense.

現在、DQN の結果との比較を容易にするため、エージェントを 2 億 frames 訓練するのがかなり標準的である。これは実時間ゲームプレイの約 38 日に相当し、高速なフレームレートであっても相当な計算コストを表す。
```

[Machado et al. (2017/09), Section 4]

したがって **Atari-100k (400k frames = 100k agent step) と Atari-200M (200M frames = 50M agent step) は別ベンチマーク**であり、両者の壁時計時間を直接並べてはならない。本レポートでは常にどちらかを明記する。

---

## 2. 壁時計時間の開示状況 — 何が例外で、何が慣行か

### 2.1 抄録レベルでは明確に例外的である

arXiv API による抄録全文検索 (2026-09-09 実行、全期間) の結果は以下のとおりである。

| クエリ | 件数 |
|---|---|
| `abs:Atari` | 1055 |
| `abs:Atari AND abs:"wall-clock"` | 7 |
| `abs:Atari AND abs:"training time"` | 26 |
| `abs:Atari AND abs:"GPU hours"` | 0 |
| `abs:Atari AND abs:"desktop"` | 4 |
| `abs:Atari AND abs:"RTX"` | 1 |
| `abs:Atari AND abs:"A100"` | 2 |
| `abs:Atari AND abs:"V100"` | 0 |

[arXiv API export.arxiv.org/api/query (2026/09), search_query による totalResults]

抄録で Atari に言及する 1055 本のうち、抄録に "wall-clock" を含むのは 7 本 (0.66%)、GPU 型番を含むのは合計 3 本 (0.28%) にすぎない。`abs:Atari AND abs:"desktop"` の 4 件は Physical Atari (2026)、BTR (2024)、Daley & Amato (2021)、Deep Neuroevolution (2017) であり、**「デスクトップ級ハードでの Atari」を主題に掲げる論文は 10 年弱で 4 本しか抄録に現れない**。

ただしこれは抄録・タイトルのみの検索であり、付録で開示している論文を捕捉できない。この数値は開示率の**下限指標**としてのみ解釈すべきである。

### 2.2 本文・付録レベルでは、探せば相当数見つかる

本調査で本文を実際に検査した Atari 関連の主要文献 21 本について、開示状況を判定した結果が以下である。

| 文献 | 年 | 壁時計時間 | ハードウェア | 対象 frames |
|---|---|---|---|---|
| Beyond The Rainbow (BTR) | 2024/2025 | 12h 未満、5 構成の一覧表 | RTX4090/i9-14900k ほか 5 構成 | 200M |
| Human-Level Control without Server-Grade Hardware | 2021 | 25h → 9h | GTX 1080 / i7-7700K | 200M |
| Fast and Data-Efficient Training of Rainbow | 2021 | 24h → 7.5h、2080Ti で 10h | RTX 3090 / 2080Ti | 10M |
| PQN (Simplifying Deep TD Learning) | 2024/2025 | 1h / 2h / 5h | NVIDIA A40 + AMD 7513 32-Core | 200M / 400M / 800M |
| BBF | 2023 | 約 6h と 10h の 2 記述 | A100 の半分 + CPU 1 個 | 100k steps |
| EfficientZero | 2021 | 7 時間 | 「4 GPUs」型番なし | 100k steps |
| SEED RL | 2019 | 1.8 日 (R2D2 は 5 日) | TPU v3 8 コア + 610 actor | Atari-57 |
| Ape-X | 2018 | 5 日 / 35h / 5-6h | 376 cores + Tesla P100 | 22,800M ほか |
| CuLE | 2019/2020 | 1 時間、21.2 分 | Titan V / V100 / V100×8 | 200M ほか |
| EnvPool | 2022 | 5 分、73 分 | i7-8750H ラップトップ / DGX-A100 | 記載により可変 |
| Deep Neuroevolution | 2017/2018 | 約 4 時間 / 約 1 時間 | 「single desktop」「720 CPUs」型番なし | 1B |
| Deep RL at the Edge of the Statistical Precipice | 2021 | 1 run 3-5 時間 | Tesla P100 | 100k steps |
| Revisiting Rainbow | 2020/2021 | 5 日 / 34,200 GPU 時間 | Tesla P100 | 200M (他者研究の見積) |
| Rainbow (Hessel et al.) | 2017/2018 | 約 10 日、10 時間未満 | 「a single GPU」型番なし | 200M / 7M |
| DreamerV2 | 2020/2021 | 「10 days per change」 | 「a single GPU」型番なし | 200M |
| SimPLe | 2019 | 「more than three weeks」 | 記載なし (確認範囲内) | 100k steps |
| DreamerV3 | 2023 | Atari の時間は確認できず (GPU 日は Table 2 に有) | 単一 Nvidia A100 | 200M |
| Dopamine | 2018 | fps のみ、時間換算なし | Tesla P100 | 記載なし |
| Machado et al. (Revisiting the ALE) | 2017/2018 | 記載なし (確認範囲内) | 記載なし (確認範囲内) | 10/50/100/200M |
| IMPALA | 2018 | Atari は確認できず (DMLab-30 は 10h 記載) | FPS とコア数の記述あり | — |
| MEME | 2022 | 記載なし (BTR が「Not Reported」と記録) | 同上 | 200M |

内訳は、壁時計時間を数値で述べているものが 13 本、うち GPU 型番まで明記しているものが 11 本である。ただし**この 21 本は「壁時計時間を書いていそうな論文」を意図的に狙って集めた標本であり、無作為抽出ではない**。この比率を Atari 研究全体の開示率と読むことはできない。

### 2.3 開示を避ける論拠は Rainbow 原論文が明文化している

Rainbow は壁時計時間を書いているが、同時に「ハードウェアが違えば比較は成立しない」という理由で、壁時計時間ではなくアルゴリズム変種のみに注目する方針を明記している。この一節が、以後の Atari 研究が壁時計時間の報告を避ける論拠の起点になっている。

```
As in the original DQN setup, we ran each agent on a single GPU. The 7M frames required to match DQN's final performance correspond to less than 10 hours of wall-clock time. A full run of 200M frames corresponds to approximately 10 days, and this varies by less than 20% between all of the discussed variants. The literature contains many alternative training setups that improve performance as a function of wall-clock time by exploiting parallelism. Properly relating the performance across such very different hardware/compute resources is non-trivial, so we focused exclusively on algorithmic variations, allowing apples-to-apples comparisons.

元の DQN のセットアップと同様に、各エージェントを単一の GPU で実行した。DQN の最終性能に一致するのに必要な 700 万 frames は、壁時計時間で 10 時間未満に相当する。2 億 frames のフルランは約 10 日に相当し、議論したすべての変種の間でこの値は 20% 未満しか変動しない。文献には、並列性を活用して壁時計時間に対する性能を改善する代替的な訓練設定が多数存在する。そうした大きく異なるハードウェア/計算資源をまたいで性能を適切に関係づけることは自明ではないため、我々はアルゴリズム上の変種のみに注目し、同条件比較を可能にした。
```

[Rainbow (2017/10), Analysis 節]

### 2.4 評価プロトコルの原典は frames 数を標準指標に据えている

Machado et al. 2018 は ALE の標準評価プロトコルを定めた原典だが、**壁時計時間もハードウェアも記載がない**。計算コストへの言及は「実時間ゲームプレイ換算」(38 日) であって壁時計時間ではない (§1.4 に引用)。標準指標は frames 数に固定されている。

なお、この指標設計と後続論文の非開示との間に因果関係を主張する記述は原典には存在しない。ここで述べられるのは、原典が壁時計時間を標準指標に含めていないという事実のみである。

### 2.5 「同一アルゴリズム × 複数ハードウェア」の表は BTR が唯一

本調査の範囲では、同じアルゴリズムを複数のハードウェアで測って壁時計時間を並べた表は BTR の Appendix G.2 だけである。

```
Desktops:
Original: RTX 4090, Intel i9-13900k (2023), 64GB RAM - 11.5 Hours
RTX 3070, Ryzen 9 3900X (2019), 64GB RAM - 52 Hours
RTX 2080 ti, Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz (2018), 128GB RAM - 32 Hours
Internal Clusters:
Nvidia H100, 48 Core Intel(R) Xeon(R) Platinum 8468 (2023), 2TB RAM - 15 Hours
Nvidia A100, 24 Core Intel(R) Xeon(R) Gold 6336Y (2021), 512GB RAM - 22 Hours

デスクトップ:
オリジナル: RTX 4090、Intel i9-13900k (2023)、64GB RAM - 11.5 時間
RTX 3070、Ryzen 9 3900X (2019)、64GB RAM - 52 時間
RTX 2080 ti、Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz (2018)、128GB RAM - 32 時間
内部クラスタ:
Nvidia H100、48 コア Intel(R) Xeon(R) Platinum 8468 (2023)、2TB RAM - 15 時間
Nvidia A100、24 コア Intel(R) Xeon(R) Gold 6336Y (2021)、512GB RAM - 22 時間
```

[BTR (2025/05), Appendix G.2]

この表で H100 が RTX 4090 より遅い理由を、BTR 自身が律速の所在として説明している。

```
We note that there is significant variability in hardware (processors, memory bus speeds, etc), but the results still show reasonable times compared to not using BTR. Overall, we found that training BTR was very capable of running on lower end machines, with the agent (excluding the environments) using around 15GB of RAM. The main performance bottleneck was running the environment in parallel, making the number of CPU cores and processor speed most important.

ハードウェア (プロセッサ、メモリバス速度など) には大きなばらつきがあることに注意するが、それでも BTR を使わない場合と比べて妥当な時間を示している。全体として、BTR の訓練は低スペックのマシンでも十分に動作し、エージェント (環境を除く) は約 15GB の RAM を使用した。主要な性能ボトルネックは環境の並列実行であり、CPU コア数とプロセッサ速度が最も重要となった。
```

[BTR (2025/05), Appendix G.2]

Ape-X の Table 1 と CuLE の Table 1 は「複数アルゴリズム × 時間 × 資源」の表であって、同一アルゴリズムを複数ハードで測った表ではない。

---

## 3. 単一 GPU・1 台構成で壁時計時間が報告されている研究

### 3.1 Stooke & Abbeel 2018 — GPU 台数別の完走時間を表で与える唯一の出典

本調査で見つかった中で、単一 GPU 構成の Atari 200M frames 完走時間を**表として直接与えている**のはこの論文だけである。対象は NVIDIA DGX-1 (P100 8 GPU、40 CPU コア) 1 台で、GPU 台数を 1 / 2 / 4 / 8 と振っている。

```
We investigated the learning speeds obtainable when running an 8-GPU, 40-core server (P100 DGX-1) to learn a single game, as an example large-scale implementation.

大規模実装の一例として、単一のゲームを学習するために 8-GPU・40 コアのサーバ (P100 DGX-1) を走らせた場合に得られる学習速度を調査した。
```

[Stooke & Abbeel (2018/03), §6.3 Learning Speed]

```
Table 2. Hours to complete 50 million steps (200M frames) by GPU and CPU count. A2C/A3C used 16 environments per GPU, PPO/APPO used 8 (DQN batch sizes shown).

              # GPU (# CPU)
ALGO           1 (5)   2 (10)   4 (20)     8 (40)
A2C              3.8      2.2      1.2       0.59
A3C                -      2.4      1.3       0.65
PPO              4.4      2.6      1.5        1.1
APPO               -      2.8      1.5       0.71

ALGO-B.S.      1 (5)   2 (10)   4 (20)     8 (40)
DQN-512          8.3      4.8   3.1/3.9 *    2.6
e-RNBW-512      14.1      8.6      6.6        6.4
CATDQN-2K       10.7      6.0      2.8        1.8

* Asynchronous

表 2. GPU 数・CPU 数別の、5000 万ステップ (2 億フレーム) 完了までの時間 (時間単位)。A2C/A3C は GPU あたり 16 環境、PPO/APPO は 8 環境を使用 (DQN はバッチサイズを表示)。
1 GPU (5 CPU) では A2C 3.8 時間、PPO 4.4 時間、DQN-512 8.3 時間、ε-Rainbow-512 14.1 時間、Categorical-DQN-2048 10.7 時間。8 GPU (40 CPU) では A2C 0.59 時間、Categorical-DQN-2048 1.8 時間。
* 非同期
```

[Stooke & Abbeel (2018/03), Table 2]

本文はこの 1 GPU 列を Ape-X と直接比較している。

```
Using 1 GPU and 5 CPU cores, DQN and e-Rainbow completed 50 million steps (200 million frames) in 8 and 14 hours, respectively--a significant gain over the reference times of 10 days. These learning speeds are comparable to those in Horgan et al. 2018, which used 1 GPU and 376 CPU cores (see e.g. Figure 2 therein for 10-hour learning curves).

1 GPU と 5 CPU コアを用いて、DQN と ε-Rainbow はそれぞれ 8 時間と 14 時間で 5000 万ステップ (2 億フレーム) を完了した — これは参照時間である 10 日に対する大きな改善である。これらの学習速度は、1 GPU と 376 CPU コアを用いた Horgan et al. 2018 のものと同程度である (例えば同論文の図 2 の 10 時間学習曲線を参照)。
```

[Stooke & Abbeel (2018/03), §6.3 Learning Speed]

方策勾配側では、Pong を分単位で解いている。

```
Several games exhibit a steep initial learning phase; all algorithms completed that phase in under 10 minutes. Notably, PPO mastered Pong in 4 minutes. A2C with 256 environments processed more than 25,000 samples per second, equating to over 90 million steps per hour (360 million frames).

いくつかのゲームは急峻な初期学習フェーズを示す。全アルゴリズムがそのフェーズを 10 分未満で完了した。特筆すべきは、PPO が Pong を 4 分で習得したことである。256 環境の A2C は毎秒 25,000 サンプル超を処理し、これは 1 時間あたり 9000 万ステップ超 (3 億 6000 万フレーム) に相当する。
```

[Stooke & Abbeel (2018/03), §6.3 Learning Speed]

到達スコアは Table 1 にあり、対象は 49 ゲーム、DQN 系が 50M steps (200M frames)、方策勾配系が 25M steps (100M frames) である。DQN-512 が中央値 1.4、CatDQN-2048 が 2.9、ε-Rainbow-512 が 2.5 (いずれも人間正規化スコア中央値) と報告されている [Stooke & Abbeel (2018/03), Table 1]。

### 3.2 rlpyt — R2D2 級ベンチを単一ワークステーションで再現

rlpyt は、分散前提とされていた R2D2 のベンチマークを単一ワークステーションで再現している。GPU は 3 台だがマシンは 1 台である。

```
rlpyt achieves over 16,000 SPS when using only 24 CPUs and 3 Titan-Xp GPUs in a single workstation (one GPU for training, two for action-serving in the alternating sampler). This may be enough to enable experimentation without access to distributed infrastructure.

rlpyt は単一ワークステーション内の 24 CPU と 3 台の Titan-Xp GPU のみを用いて 16,000 SPS 超を達成する (1 台は学習用、2 台は alternating sampler での行動サービング用)。これは分散インフラなしでの実験を可能にするのに十分かもしれない。
```

[rlpyt (2019/09), §3.2 R2D1 節]

```
This run reached 8 billion steps and 1 million updates in less than 138 hours.

この実行は 138 時間未満で 80 億ステップと 100 万回の更新に到達した。
```

[rlpyt (2019/09), §3.2 R2D1 節]

CPU は 2 基の Intel Xeon Gold 6126 (2017 年頃) である [rlpyt (2019/09), §3.2 脚注 6]。単位は SPS = agent step/s であり、生フレーム換算では 64,000 FPS 相当になる (この換算は本レポートによるもので、出典に記載はない)。

### 3.3 Schmidt & Schmied — 単一 GPU で 10M frames を 7.5 時間

BTR が vectorization の出典として引く論文である。**対象 frames は 10M であって 200M ではない**点に注意が必要である。

```
In aggregate, these modifications decrease the training time by a factor of 3.2, from 24 to approximately 7.5 hours, for training on 10M frames on a single Nvidia RTX 3090 GPU.

総合すると、これらの変更により、単一の Nvidia RTX 3090 GPU 上での 1000 万 frames の訓練にかかる時間が 24 時間から約 7.5 時間へと 3.2 倍削減される。
```

[Schmidt & Schmied (2021/11), Section 4]

```
Each of our training runs took approximately 7.5 hours on a single Nvidia RTX 3090 GPU or 10 hours on a single Nvidia 2080Ti GPU. These requirements make performing larger numbers of experiments feasible, even with a more modest compute budget.

我々の各訓練 run は、単一の Nvidia RTX 3090 GPU で約 7.5 時間、単一の Nvidia 2080Ti GPU で 10 時間を要した。この要求水準であれば、より控えめな計算予算でも多数の実験が実行可能になる。
```

[Schmidt & Schmied (2021/11), Section 5]

### 3.4 Daley & Amato — 2016 年発売の GPU で 200M frames を 9 時間

壁時計時間の短縮そのものを主題に据え、「最先端ではない GPU で足りる」ことを明示的に主張している数少ない例である。

```
With just one NVIDIA GeForce GTX 1080 GPU, our implementation reduces the training time of a 200-million-frame Atari experiment from 25 hours to just 9 hours.

たった 1 枚の NVIDIA GeForce GTX 1080 GPU で、我々の実装は 2 億 frames の Atari 実験の訓練時間を 25 時間からわずか 9 時間へ削減する。
```

[Daley & Amato (2021/11), Abstract]

```
Our test hardware consists of a four-core (eight-thread) Intel Core i7-7700K CPU and an NVIDIA GeForce GTX 1080 GPU. Notably, this particular GPU model is not considered state of the art even for desktop hardware, having been released in 2016. As such, comparable or better hardware should be affordable to a large number of researchers and practitioners.

我々のテストハードウェアは 4 コア (8 スレッド) の Intel Core i7-7700K CPU と NVIDIA GeForce GTX 1080 GPU から成る。特筆すべきは、この GPU は 2016 年発売であり、デスクトップハードウェアとしてすら最先端とは見なされない点である。したがって、同等以上のハードウェアは多数の研究者・実務者にとって手の届く範囲にあるはずである。
```

[Daley & Amato (2021/11), Section 5.1]

### 3.5 BTR — 壁時計時間を設計目標として掲げた例

BTR は導入部で、壁時計時間そのものをアクセシビリティの制約条件として問題設定している。

```
However, recent state-of-the-art approaches (Schrittwieser et al., 2020; Badia et al., 2020a; Hessel et al., 2021; Kapturowski et al., 2023) are increasingly out of reach for those with more limited compute resources, either in terms of the required hardware or the walltime necessary to train a single agent.

しかし最近の最先端手法 (Schrittwieser et al., 2020; Badia et al., 2020a; Hessel et al., 2021; Kapturowski et al., 2023) は、必要なハードウェアの面でも、単一エージェントの訓練に要する壁時計時間の面でも、計算資源の限られた者にはますます手の届かないものになっている。
```

[BTR (2025/05), §1 Introduction]

```
Computationally Accessible (Figure 6) - Using a high-end desktop PC, BTR trains Atari agents for 200 million frames in under 12 hours, significantly faster than Rainbow DQN's 35 hours. This increases RL research's accessibility for smaller research labs and hobbyists without the need for GPU clusters or excessive walltime.

計算的にアクセス可能 (図 6) - ハイエンドのデスクトップ PC を用いて、BTR は Atari エージェントを 2 億 frames まで 12 時間未満で訓練する。これは Rainbow DQN の 35 時間より大幅に速い。これにより、GPU クラスタや過大な壁時計時間を必要とせずに、小規模研究室やホビイストにとっての RL 研究のアクセシビリティが高まる。
```

[BTR (2025/05), §1 Introduction 貢献リスト]

replay ratio を 1/64 に抑えた理由も壁時計時間である。

```
This results in a replay ratio (ratio of gradient updates to environment steps) of 1/64. Higher replay ratios have been shown to improve performance (D'Oro et al., 2022), however we opt to keep this value low to reduce walltime.

これにより replay ratio (勾配更新と環境ステップの比) は 1/64 になる。より高い replay ratio が性能を改善することは示されている (D'Oro et al., 2022) が、我々は壁時計時間を減らすためこの値を低く保つことを選択する。
```

[BTR (2025/05), §3.1]

さらに、比較図の凡例に frames 数・壁時計時間・ハードウェアを併記する形式を採っている。

```
Brackets show the number of frames the algorithms use, the number of walltime hours and the hardware used respectively.

括弧はそれぞれ、各アルゴリズムが使用する frames 数、壁時計時間 (時間)、使用ハードウェアを示す。
```

[BTR (2025/05), Figure 2 caption]

### 3.6 一覧表: 単一 GPU / 1 台構成での Atari 壁時計時間

各行は各論文の自己申告であり、測定年・ハード世代・プロトコルが揃っていない。**表を横断で読むときは必ず「対象 frames」と「注」の列を見ること。**

| 手法 | 対象 frames | ハードウェア | 壁時計時間 | 注 | 出典 |
|---|---|---|---|---|---|
| DQN | 200M | 1 GPU (型番なし) | 9.5 日 | Ape-X による集成 | Ape-X Table 1 |
| Rainbow | 200M | 1 GPU (型番なし) | 約 10 日 | 原論文の自己申告 | Rainbow Analysis 節 |
| Rainbow | 200M | Tesla P100 | 約 5 日 | Revisiting Rainbow による見積 | Revisiting Rainbow §3 |
| A2C | 200M | P100 × 1 + 5 CPU コア | 3.8 時間 | 到達スコアは HNS 中央値 0.65 | Stooke & Abbeel Table 2 / Table 1 |
| PPO | 200M | 同上 | 4.4 時間 | 同 1.2 | 同上 |
| DQN (バッチ 512) | 200M | 同上 | 8.3 時間 | 同 1.4 | 同上 |
| Categorical-DQN (バッチ 2048) | 200M | 同上 | 10.7 時間 | 同 2.9 | 同上 |
| ε-Rainbow (バッチ 512) | 200M | 同上 | 14.1 時間 | 同 2.5 | 同上 |
| Rainbow 改良版 | **10M** | RTX 3090 × 1 | 7.5 時間 | 200M ではない | Schmidt & Schmied §4/§5 |
| Rainbow 改良版 | **10M** | RTX 2080Ti × 1 | 10 時間 | 同上 | Schmidt & Schmied §5 |
| DQN 系 (Daley & Amato) | 200M | GTX 1080 × 1 + i7-7700K | 25 時間 → 9 時間 | 改良前 → 改良後 | Daley & Amato Abstract |
| BTR | 200M | RTX 4090 + i9-13900k | 11.5 時間 | Machado 準拠プロトコル | BTR Appendix G.2 |
| BTR | 200M | H100 + Xeon Platinum 8468 | 15 時間 | 同上 | 同上 |
| BTR | 200M | A100 + Xeon Gold 6336Y | 22 時間 | 同上 | 同上 |
| BTR | 200M | RTX 2080Ti + Xeon Silver 4112 | 32 時間 | 同上 | 同上 |
| BTR | 200M | RTX 3070 + Ryzen 9 3900X | 52 時間 | 同上 | 同上 |
| PQN | 200M | A40 × 1 + EPYC 7513 32C | 1 時間 | **sticky なし + episodic life** | PQN Table 3 |
| PQN | 400M | 同上 | 2 時間 (README は約 4 時間) | 同上・出典間で不一致 | PQN Table 3 / purejaxql README |
| PQN (Dopamine 準拠) | 800M | 同上 | 約 5 時間 | sticky あり | PQN Figure 12 |
| DreamerV3 | 200M | A100 × 1 | 7.7 A100 GPU 日 | 時間表記ではなく GPU 日 | DreamerV3 Table 2 |
| Deep Neuroevolution (GA) | 1B | 「単一デスクトップ」 | 約 4 時間 | GPU 型番なし・GA であって RL ではない | Such et al. §5 |
| rlpyt R2D1 | 40B 規模 | Titan-Xp × 3 + 24 CPU コア | 138 時間未満 | 200M ではない | rlpyt §3.2 |

---

## 4. PQN — replay buffer を捨てて並列 env で回す系統

### 4.1 書誌と位置づけ

原典は "Simplifying Deep Temporal Difference Learning" (arXiv:2407.04811)、著者は Matteo Gallici、Mattie Fellows、Benjamin Ellis、Bartomeu Pou、Ivan Masmitja、Jakob Nicolaus Foerster、Mario Martin の 7 名。所属は Universitat Politècnica de Catalunya / University of Oxford / Barcelona Supercomputing Center / Institut de Ciències del Mar。ICLR 2025 採択 (公式リポジトリ README は Spotlight と記す)。被引用数 87 (Semantic Scholar, 2026/09 取得)。

中核主張は、LayerNorm 等の正則化により target network も replay buffer もなしに TD が収束することを理論的に示した点にある。

```
Our key theoretical result demonstrates for the first time that regularisation techniques such as LayerNorm can yield provably convergent TD algorithms without the need for a target network or replay buffer, even with off-policy data. Empirically, we find that online, parallelised sampling enabled by vectorised environments stabilises training without the need for a large replay buffer.

我々の主要な理論的結果は、LayerNorm のような正則化技法が、target network や replay buffer を必要とせず、off-policy データであっても、収束が証明可能な TD アルゴリズムを生み出しうることを初めて示すものである。経験的には、ベクトル化環境によって可能となるオンラインかつ並列的なサンプリングが、大規模な replay buffer を必要とせずに学習を安定化させることを見出した。
```

[PQN (2025/04), Abstract]

replay buffer を捨てる動機は GPU メモリである。

```
However, the replay buffer's large memory footprint makes pure-GPU training impractical with traditional DQN. With the goal of enabling Q-learning in pure-GPU setting, we propose replacing a large replay buffer with a synchronous update across a large number of parallel environments, reducing memory requirements.

しかし replay buffer の大きなメモリ占有量は、従来の DQN での pure-GPU 学習を非現実的にする。pure-GPU 環境での Q 学習を可能にすることを目的として、我々は大規模な replay buffer を、多数の並列環境にまたがる同期更新で置き換え、メモリ要求量を削減することを提案する。
```

[PQN (2025/04), §1 Introduction]

### 4.2 壁時計時間の実測 (Table 3)

壁時計時間の短縮は経験的貢献の 3 番目として明示的に列挙されている。

```
III) our extensive empirical study demonstrates PQN achieves competitive results in significantly less wall-clock time than existing state-of-the-art methods.

III) 我々の広範な経験的研究は、PQN が既存の最先端手法よりも大幅に少ない壁時計時間で競争力のある結果を達成することを実証する。
```

[PQN (2025/04), §1 Introduction 経験的貢献の列挙]

中核データは Table 3 である。時間は 1 ゲームあたりである。

```
Table 3: Scores in ALE.
Method (Frames) | Time (hours) | Gradient Steps | Atari-10 Score | Atari-57 Median | Atari-57 Mean | Atari-57 >Human
PPO (200M)      | 2.5  | 780k  | 165 |     |      |
PQN (200M)      | 1    | 780k  | 191 |     |      |
PQN (400M)      | 2    | 1.4M  | 243 | 245 | 1440 | 40
Rainbow (200M)  | 100  | 12.5M | 239 | 230 | 1461 | 43

表 3: ALE におけるスコア。
手法 (フレーム数) | 時間 (時) | 勾配ステップ数 | Atari-10 スコア | Atari-57 中央値 | Atari-57 平均 | Atari-57 人間超えゲーム数
PPO (200M)      | 2.5  | 780k  | 165 |     |      |
PQN (200M)      | 1    | 780k  | 191 |     |      |
PQN (400M)      | 2    | 1.4M  | 243 | 245 | 1440 | 40
Rainbow (200M)  | 100  | 12.5M | 239 | 230 | 1461 | 43
```

[PQN (2025/04), Appendix D Table 3]

ハードウェアは GPU 1 枚で、**Atari の env のみ CPU 上で実行される**。

```
All experimental results are shown as mean of 10 seeds, except in Atari Learning Environment (ALE) where we followed a common practice of reporting 3 seeds. They were performed on a single NVIDIA A40 by jit-compiling the entire pipeline with Jax in the GPU, except for the Atari experiments where the environments run on an AMD 7513 32-Core Processor.

すべての実験結果は 10 seed の平均として示すが、Atari Learning Environment (ALE) では 3 seed を報告する一般的慣行に従った。実験は単一の NVIDIA A40 上で、Jax によりパイプライン全体を GPU 上に jit-compile して実施した。ただし Atari 実験では環境が AMD 7513 32-Core Processor 上で動作する。
```

[PQN (2025/04), Appendix C 実験設定]

並列 env 数は 128、rollout 長 32 (1 更新あたり 4096 遷移) である [PQN (2025/04), Appendix E Table 5]。

**PQN 論文には SPS / FPS / throughput のいずれの語も出現しない。** 速度は常に所要時間 (時・分) の形でのみ報告されており、スループット値の比較には使えない。

### 4.3 評価プロトコル — スコア比較の可否に直結する

公式設定は **sticky actions なし (repeat_action_probability = 0)**、**episodic life あり**、reward clip あり、frame skip 4、noop_max 30 である。

```
ENV_KWARGS:
  episodic_life: True # lost life -> done, increases sample efficiency, may hurt in some games
  reward_clip: True # reward into -1, 1
  repeat_action_probability: 0. # sticky actions
  frame_skip: 4
  noop_max: 30

ENV_KWARGS:
  episodic_life: True # ライフ喪失 -> done、サンプル効率を上げるが一部ゲームでは害になりうる
  reward_clip: True # 報酬を -1, 1 に収める
  repeat_action_probability: 0. # sticky actions
  frame_skip: 4
  noop_max: 30
```

[purejaxql `purejaxql/config/alg/pqn_atari.yaml` (2026/09 取得), 環境ブロック]

**そして PQN 自身が、Dopamine 準拠 (sticky あり) に切り替えると必要 frames 数と時間が跳ね上がると認めている。** 本文の 1 時間 / 2 時間はこの設定での数値ではない。

```
Figure 12: IQM computed over 3 seeds when training PQN with the ALE configuration proposed by Dopamine Castro et al. (2018). This configuration incorporates sticky actions and doesn't set the done flag when an agent loses a life. With this setup, PQN can still outperform Rainbow, but it requires significantly more compute time (almost 5 hours), corresponding to 800 million frames, indicating a loss of sample efficiency.

図 12: Dopamine (Castro et al., 2018) が提案する ALE 設定で PQN を学習した際の 3 seed にわたる IQM。この設定は sticky actions を組み込み、エージェントがライフを失っても done フラグを立てない。この構成でも PQN は Rainbow を上回りうるが、著しく多くの計算時間 (ほぼ 5 時間) を要し、これは 8 億フレームに相当し、サンプル効率の低下を示す。
```

[PQN (2025/04), Appendix D Figure 12 キャプション]

BTR はさらに強く、PQN のライフ情報つき結果は比較に無効であると結論している。

```
Figure I11: Graph shows Atari-5 performance with and without life information using Inter-quartile mean and Atari-60 predicted median from Aitchison et al. (2023). Life Information uses 3 seeds, with shaded areas showing 95% confidence intervals. From this we conclude results using life information are invalid for comparison.

図 I11: Aitchison et al. (2023) の四分位間平均と Atari-60 予測中央値を用いた、ライフ情報あり/なしでの Atari-5 性能のグラフ。ライフ情報は 3 seed を用い、網掛け領域は 95% 信頼区間を示す。ここから我々は、ライフ情報を用いた結果は比較には無効であると結論する。
```

[BTR (2025/05), Appendix I Figure I11 キャプション]

そのうえで BTR は、条件を揃えた比較表を用意している。

```
Table A3: Comparison of performance and walltime against PQN (Gallici et al., 2024). PQN only reports results at 400M frames and includes life information, which greatly affects performance (see Appendix I). To provide a fairer comparison, we also report our results using life information but only use 200M frames.
Game               | BTR (with life info, 200M frames) | PQN (with life info, 400M frames)
Inter-Quartile Mean| 12.18                             | 3.86
Walltime (A100)    | 22 Hours                          | 2 Hours

表 A3: PQN (Gallici et al., 2024) に対する性能と壁時計時間の比較。PQN は 400M フレームでの結果のみを報告し、性能に大きく影響するライフ情報を含んでいる (付録 I 参照)。より公平な比較を提供するため、我々もライフ情報を用いた結果を報告するが、200M フレームのみを使用する。
ゲーム               | BTR (ライフ情報あり、200M フレーム) | PQN (ライフ情報あり、400M フレーム)
四分位間平均           | 12.18                             | 3.86
壁時計時間 (A100)     | 22 時間                            | 2 時間
```

[BTR (2025/05), Appendix A Table A3]

**すなわち、条件を揃えると PQN は 11 倍速いが IQM は 1/3 である。**「1 時間で Rainbow 相当」という主張と、この表は矛盾しない — 前者は PQN 有利のプロトコル、後者は BTR がライフ情報側へ歩み寄った比較である。

### 4.4 「50x faster」の分母

Abstract の 50 倍は Rainbow (約 3 日/ゲーム) に対する比であり、DQN 側の時間は自己申告で「楽観的見積り」である。

```
DQN training time was optimistically estimated using the JAX-based CleanRL DQN implementation. . With an additional budget of 100M frames (30 minutes of training), PQN achieves the median score of Rainbow (Hessel et al., 2018), which is still a SOTA method in ALE for sample efficiency but requires around 3 days of training per game, meaning that PQN can be considered 50x faster.

DQN の学習時間は JAX ベースの CleanRL DQN 実装を用いて楽観的に見積もられた。100M フレームの追加予算 (30 分の学習) により、PQN は Rainbow の中央値スコアを達成する。Rainbow は ALE においてサンプル効率の点で依然として SOTA 手法だが、ゲームあたり約 3 日の学習を要する。すなわち PQN は 50 倍高速とみなせる。
```

[PQN (2025/04), §5.2 Atari および脚注 1]

分母となる Rainbow / DQN 側が PQN と同一ハードで測られたという記述は論文中にない。また Table 3 の「100 時間」と本文の「約 3 日」(= 72 時間) は一致しない。

PQN は分散手法を比較対象から明示的に外している。

```
We did not compare with distributed methods like Ape-X and R2D2 because they use an enormous time-budget (5 days of training per game) and frames (almost 40 Bilions), which are outside our computational budget.

Ape-X や R2D2 のような分散手法とは比較しなかった。これらは膨大な時間予算 (ゲームあたり 5 日の学習) とフレーム数 (ほぼ 400 億) を用いており、我々の計算予算の範囲外だからである。
```

[PQN (2025/04), Appendix C ALE 実験設定]

### 4.5 第三者による独立計測と、論文自身が認める限界

Hadamax (NeurIPS 2025) は PQN の壁時計時間を A100 上で独立に報告している数少ない資料である。

```
We run all our experiments on a HPC cluster equipped with A100 GPUs. Each run of Hadamax-PQN needs around 45 minutes for 40 millions frames and PQN needs around 20 minutes.

我々はすべての実験を A100 GPU を備えた HPC クラスタ上で実行する。Hadamax-PQN の各 run は 4000 万フレームに約 45 分を要し、PQN は約 20 分を要する。
```

[Hadamax (2025/05), Appendix C.3 Baseline Implementations]

論文自身は、Atari-57 での人間超えが 40/57 に留まる主因を ε-greedy 探索の単純さに帰している。

```
In this chart, we show that PQN reaches human-level performance in 40 of the 57 games of the ALE, underperforming mainly in the hard-exploration games, suggesting that the ϵ-greedy exploration used by PQN is too simple to solve ALE, and indicating a clear research direction to improve the method.

このチャートで我々は、PQN が ALE の 57 ゲームのうち 40 で人間水準の性能に到達し、主に hard-exploration ゲームで性能が劣ることを示す。これは PQN が用いる ε-greedy 探索が ALE を解くには単純すぎることを示唆し、手法改善の明確な研究方向を示している。
```

[PQN (2025/04), §5.2 Atari 末尾]

---

## 5. 高速 env 実装のスループット

### 5.1 EnvPool — 1M FPS は「学習なし・生フレーム」である

EnvPool 論文の Table 1 はランダム行動を投げるだけの純環境ベンチマークであり、学習を含まない。論文がそう明示している。

```
In the experiments of pure environment simulation, we obtain a randomly sampled action based on the action space definition of the games and send the actions to the environment executors.

純粋な環境シミュレーションの実験では、ゲームの行動空間定義に基づいてランダムサンプリングした行動を取得し、それを環境実行器へ送る。
```

[EnvPool (2022/06), §4.1 Pure Environment Simulation]

Atari 列の値 (単位: frame skip 4 を掛けた生フレーム/秒) は以下である。

| Method | Laptop (12 コア i7-8750H) | Workstation (32 コア Ryzen 9 5950X) | TPU-VM (96 コア Xeon) | DGX-A100 (256 コア EPYC 7742) |
|---|---|---|---|---|
| For-loop | 4,893 | 7,914 | 3,993 | 4,640 |
| Subprocess | 15,863 | 47,699 | 46,910 | 71,943 |
| Sample-Factory (env 実行器としてのみ) | 28,216 | 138,847 | 222,327 | 707,494 |
| EnvPool (sync) | 37,396 | 133,824 | 170,380 | 427,851 |
| EnvPool (async) | 49,439 | 200,428 | 359,559 | 891,286 |
| EnvPool (numa+async) | — | — | 373,169 | 1,069,922 |

[EnvPool 公式ドキュメント Benchmark (2026/09 取得), Atari ベンチマーク表]

**agent step 換算では DGX-A100 の 1,069,922 FPS が約 267,000 agent step/s、ラップトップの 49,439 FPS が約 12,400 agent step/s になる** (この換算は本レポートによるもので、出典に記載はない)。

この表の "Sample-Factory" 行は Sample Factory を env 実行器としてのみ使った値である。

```
Sample Factory [24]: pure asynchronous step with a given number of worker threads; we pick the best performance over various num_envs per worker.

Sample Factory: 指定したワーカースレッド数での純粋な非同期ステップ実行。ワーカーあたりの num_envs を変えて最良性能を採用した。
```

[EnvPool (2022/06), §4.1 ベースライン定義]

学習込みの end-to-end 数値も別途あるが、論文自身が横断比較を戒めている。

```
As a result, the end-to-end training time decreased from 200 minutes (CleanRL's PPO + For-loop) to approximately 73 minutes (CleanRL's PPO + EnvPool (Sync)) while maintaining sample efficiency.

その結果、エンドツーエンドの学習時間は 200 分 (CleanRL の PPO + For-loop) から約 73 分 (CleanRL の PPO + EnvPool (Sync)) へ短縮され、その間サンプル効率は維持された。
```

[EnvPool (2022/06), §4.2]

```
Note that the hardware specifications of these experiments are different thus readers should not compare training speeds across different training libraries.

これらの実験のハードウェア仕様は異なるため、読者は異なる学習ライブラリ間で学習速度を比較すべきではない点に注意。
```

[EnvPool (2022/06), §4.2]

### 5.2 Sample Factory — こちらは「学習込み」

Sample Factory の Table 1 は学習を含むスループットであり、EnvPool Table 1 と並べてはならない。

| Method | Atari, FPS (生フレーム、学習込み) |
|---|---|
| Pure simulation (学習なし上限) | 181,740 |
| DeepMind IMPALA | 9,961 |
| RLlib IMPALA | 22,440 |
| SeedRL V-trace | 39,726 |
| rlpyt PPO | 68,880 |
| SampleFactory APPO | 135,893 |

[Sample Factory (2020/06), Table 1]

ハードウェアは System #2 = 2x Intel Xeon Gold 6154 (36 物理コア) + 1x NVIDIA RTX 2080Ti である [Sample Factory (2020/06), 補足資料]。**agent step 換算では 135,893 ÷ 4 ≈ 34,000 agent step/s** (本レポートによる換算)。

なお、Sample Factory の Pure simulation 181,740 FPS (36 コア) と EnvPool の Workstation 200,428 FPS (32 コア) は同オーダーであり、両者は整合的に読める。

### 5.3 ALE 本体の AtariVectorEnv — CPU マルチスレッドで、公開ベンチマークがない

2025〜2026 年のこの分野で最も重要な動きは、EnvPool のメンテナンス停止と、ALE 本体への C++ vectorisation の取り込みである。ALE 公式が経緯を明記している。

```
For faster implementations, EnvPool provide C++ vectorisation that significantly increase the sample speed but it no longer maintained. Inspired by the `EnvPool` implementation, we've implemented an asynchronous vectorisation environment in C++, in particular, the standard Atari preprocessing including frame skipping, frame stacking, observation resizing, etc.

より高速な実装としては EnvPool が C++ ベクトル化を提供しサンプル速度を大きく向上させたが、現在は保守されていない。EnvPool の実装に着想を得て、我々は C++ で非同期ベクトル化環境を実装した。とくにフレームスキップ、フレームスタック、観測のリサイズ等を含む標準的な Atari 前処理を実装している。
```

[ALE v0.11.0 リリースノート (2025/04), 本文]

公式ドキュメントは、この実装が CPU マルチスレッドであり、最適設定は物理コア数に合わせることだと述べている。

```
The Arcade Learning Environment (ALE) Vector Environment provides a high-performance implementation for running multiple Atari environments in parallel. This implementation utilizes native C++ code with multi-threading to achieve significant performance improvements, especially when running many environments simultaneously. ... For optimal performance, set `num_envs` close to the number of physical CPU cores.

Arcade Learning Environment (ALE) の Vector Environment は、複数の Atari 環境を並列実行するための高性能実装を提供する。この実装はネイティブ C++ コードとマルチスレッドを用いており、とくに多数の環境を同時実行する場合に大きな性能向上を達成する。……最適な性能のためには、`num_envs` を物理 CPU コア数に近い値に設定せよ。
```

[ALE Vector Environment Guide (2026/09 取得), Introduction / Performance Considerations]

**しかし公式ドキュメント・リリースノートのいずれにも、スループット数値・EnvPool との定量比較・ベンチマークハードウェアの記載が一切ない。** リポジトリにベンチマークスクリプトはあるが結果値は同梱されていない。2026 年時点で最も実用的な Atari 高速 env 実装であるはずのものについて、公開された定量ベンチマークが存在しないというのが本調査の結果である。

第三者による実測としては、Octax 論文が EnvPool の ALE Pong を測っている。

```
EnvPool running ALE Pong with all available CPU cores shows reduced scaling, plateauing around 25,000 steps per second due to CPU saturation.

利用可能な全 CPU コアで ALE Pong を実行する EnvPool は、CPU 飽和によりスケーリングが劣り、毎秒約 25,000 step で頭打ちになる。
```

[Octax (2025/10), §4.2]

これは 20 コア Intel i7 上の **agent step** 単位の値であり、生フレーム換算で 100,000 FPS 相当になる (換算は本レポートによる)。EnvPool 論文の Workstation (32 コア) 200,428 FPS = 50,107 agent step/s と、コア数差を考えれば整合的なオーダーである。

### 5.4 JAX 系の「N 倍速い」の多くは実 ALE に対する主張ではない

PureJaxRL / gymnax が扱っているのは MinAtar であって実 ALE ではない。

```
Gymnax is an awesome library that contains several well-known environments such as Classic Control tasks, Bsuite tasks, and Minatar (Atari-like) environments.

Gymnax は、古典制御タスク、Bsuite タスク、そして Minatar (Atari 風) 環境といった複数の著名な環境を含む素晴らしいライブラリである。
```

[PureJaxRL ブログ (2026/09 取得), 環境の説明部]

Octax は CHIP-8 であり、著者自身が ALE の差し替え可能な代替ではないと明言している。

```
In this paper, we propose an alternative approach for training RL agents in environments that share mechanisms with ALE, but which is not intended as a drop-in replacement and offers significantly reduced computational cost.

本稿では、ALE と機構を共有する環境で RL エージェントを学習させるための代替手法を提案する。ただしこれは差し替え可能な代替物として意図されたものではなく、計算コストを大幅に削減するものである。
```

[Octax (2025/10), §1]

JAXAtari はゲームロジックの JAX 再実装であり、オリジナル ROM を実行できない (§6.4 に引用)。

**実 ALE を JAX から回す経路は、本調査で確認できた範囲では (a) EnvPool の XLA インタフェース (Cleanba、CleanRL の xla_jax 版) と (b) Podracer Sebulba だけであり、いずれも env 本体は CPU 上の C++ ALE のままである。** Sebulba についてはその設計が明記されている。

```
As in Anakin, Sebulba co-locates acting and learning on a single TPU machine. However, it steps the environments on the host CPU, and splits the available 8 TPU cores (for each host) in two disjoint sets: A cores are used exclusively to act, and the remaining 8-A cores are used to learn.

Anakin と同様、Sebulba は acting と learning を単一の TPU マシン上に同居させる。ただし環境のステップはホスト CPU 上で行い、利用可能な 8 個の TPU コア (ホストごと) を互いに素な 2 つの集合に分割する: A 個のコアは acting 専用に、残りの 8-A 個のコアは learning に使われる。
```

[Podracer (2021/04), §3.2]

### 5.5 有名な実装ほど公式の絶対値がない

本調査で確認した実装のうち、Atari の絶対スループットが公式一次出典に**存在しない**ものを列挙する。これ自体が結果である。

| 実装 | 公式一次出典の Atari 絶対スループット |
|---|---|
| TorchBeast | なし (TF IMPALA と同等、とのみ) |
| moolib | なし (README の Atari 結果は画像 2 枚のみ) |
| Tianshou (JMLR 論文) | なし (ベンチマークは MuJoCo のみ。README には有) |
| Acme | なし (TPU-v2 が V100 比 1.7x / 5.6x の相対値のみ) |
| Sample Factory 2 | なし |
| Stoix | なし |
| ALE AtariVectorEnv (2025〜) | なし |
| CleanRL (1 run 実時間) | なし (W&B の対話的グラフのみ) |

したがって「主要フレームワークの Atari スループット一覧表」は、原理的に一次出典だけでは埋まらない。埋まっている値の多くは第三者 (Sample Factory 著者、EnvPool 著者、Cleanba 著者、TorchRL 著者、Octax 著者) が競合を測ったものである。

なお Acme は "walltime" を独自定義しており、他文献の壁時計時間と直接比較できない。

```
we accumulate time within the learner from immediately after the first learner step; we refer to this measure of time as the learner walltime.

最初の learner ステップの直後から learner 内部で時間を積算する。この時間の指標を learner walltime と呼ぶ。
```

[Acme (2020/06), 実験設定節]

---

## 6. ALE を GPU 側で回す試み

### 6.1 CuLE — 唯一の本格的な ALE 移植

CuLE (NVIDIA, NeurIPS 2020) は ALE を CUDA へ移植し、GPU 上で Atari をエミュレートする実装である。

```
CuLE generates up to 155M frames per hour on a single GPU, a finding previously achieved only through a cluster of CPUs.

CuLE は単一 GPU で 1 時間あたり最大 155M フレームを生成する。これは従来 CPU クラスタによってのみ達成されていた水準である。
```

[CuLE (2020/12), Abstract]

Table 1 の学習込み行は「1 GPU・1 時間・200M frames・26K-68K FPS」である [CuLE (2020/12), Table 1]。System I は 12 コア Core i7-5930K + Titan V である [CuLE (2020/12), Table 2]。

**ただし CPU ベースライン比の優位は 3.56 倍にとどまる。**

```
the ratio between the median FPS generated by CuLE with 4096 environment (64K) and the peak FPS for OpenAI Gym (18K) is 3.56x.

CuLE の 4096 環境における FPS 中央値 (64K) と OpenAI Gym のピーク FPS (18K) の比は 3.56 倍である。
```

[CuLE (2020/12), p.5 Section 4]

### 6.2 忠実度と実用上の制限

CuLE はリセット手順が ALE と異なる。

```
To address this issue, we generate and store a cache of random initial states (30 by default) when a set of environments are initialized in CuLE. At the end of an episode, each emulator randomly selects one of the cached states as a seed and copies it into the terminal emulator state.

この問題に対処するため、CuLE では環境集合の初期化時にランダム初期状態のキャッシュ (既定 30 個) を生成・保存する。エピソード終了時、各エミュレータはキャッシュ済み状態の 1 つをランダムに選んでシードとし、終端エミュレータ状態へコピーする。
```

[CuLE (2020/12), p.4 Section 3]

GPU DRAM の制約により、実際の学習は 5K 環境未満に制限されている。

```
Since we did not implement any data compression scheme as in [8], we constrain our training configuration to fewer than 5K environments, but peak performance in terms of FPS would be achieved for a higher number of environments - this is left as a possible future improvement.

[8] のようなデータ圧縮方式を実装していないため、学習構成を 5K 環境未満に制限しているが、FPS のピーク性能はより多くの環境数で達成されるはずである。これは将来の改善余地として残す。
```

[CuLE (2020/12), p.7 Section 4]

ビット単位一致は主張されていない。第三者の 2026 年論文が明示している。

```
CuLE ports ALE to CUDA for massive-batch rollouts (up to ~155M frames/h on a single GPU) but is neither bit-exact nor differentiable (Dalton et al. 2020)

CuLE は大規模バッチのロールアウトのために ALE を CUDA へ移植している (単一 GPU で最大 ~155M フレーム/時) が、ビット単位で厳密でもなければ微分可能でもない (Dalton ら 2020)。
```

[A Differentiable Atari VCS (2026/06), Section 2 Related Work]

ゲーム本数もソースツリー上は 63 本であり、ALE 公式の 104 本に届かない [GitHub Trees API 集計 (2026/09/09 取得), NVlabs/cule および Farama-Foundation/Arcade-Learning-Environment]。

### 6.3 上流の保守は事実上停止している

GitHub API による事実として、NVlabs/cule の最終コミットは **2022-11-17**、以降 3 年 10 か月にわたり変更がない。README が公称する対応 CUDA は 10.0 である。

```
CuLE performs best when compiled with the CUDA 10.0 Toolkit. It is currently incompatible with CUDA 10.1.

CuLE は CUDA 10.0 Toolkit でコンパイルしたときに最良の性能を発揮する。現在 CUDA 10.1 とは非互換である。
```

[NVlabs/cule README.md (2022/11 最終更新), 18-19行]

open issue は「新しいツールチェインでビルドできない」に集中し、未応答のまま残っている。2026 年に入って個人 fork 3 件 (chengscott / davidquarel / MehrdadMoghimi) が独立に CUDA 12/13 対応を当てているが、いずれも star 0〜1 でコミュニティの集約先になっていない [GitHub REST API (2026/09/09 取得), repos/NVlabs/cule および forks]。

### 6.4 ALE 公式の「XLA GPU support」はエミュレーションを GPU へ移していない

v0.12.0 のリリースノートの文言だけを読むと GPU 実行に見える。

```
The XLA `VectorEnv` integration, previously CPU-only, now also runs on GPU for accelerated batched environment execution.

XLA の `VectorEnv` 統合は、従来 CPU 専用だったが、バッチ環境実行の高速化のため GPU でも動作するようになった。
```

[ALE v0.12.0 リリースノート (2026/05), 本文]

しかしソースを読むと、GPU 経路は転送だけであり、エミュレーション自体はホスト (CPU) 側で走る。

```cpp
ffi::Error XLAResetGPUImpl(
    cudaStream_t stream, ...
) {
    ...
    err = cudaMemcpyAsync(host_reset_indices.data(), reset_indices_buffer.typed_data(),
                          reset_indices_buffer.element_count() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost, stream);
    ...
    // Reset the environments (returns BatchResult directly)
    auto result = vectorizer->reset(reset_indices, reset_seeds);
    ...
    // Copy data from BatchResult to GPU buffers via host memory
    err = cudaMemcpyAsync(observations_buffer->typed_data(), result.obs_data(),
                          batch_size * stacked_obs_size,
                          cudaMemcpyHostToDevice, stream);

(コード。GPU 版 XLA reset の実装。入力インデックスをデバイスからホストへコピーし、ホスト上の vectorizer->reset() で環境をリセットし、結果をホストからデバイスへ戻している。)
```

[Farama-Foundation/Arcade-Learning-Environment `src/ale/python/ale_vector_xla_interface.cpp` (2026/09 取得), 111-215行付近]

**つまり ALE 公式の「XLA GPU support」は、GPU 常駐の JAX プログラムから CPU エミュレータを呼ぶための FFI 橋渡しである。**

### 6.5 新規実装は ALE 互換でないか、未成熟である

- **JAXAtari** (TU Darmstadt): ゲームロジックの JAX 再実装であり、ROM を実行しない。第三者論文が明記している。

```
JAXAtari reimplements each game's logic in JAX and cannot execute the original ROMs at all, and both are RL environments rather than a gate-level differentiable port validated bit-for-bit against the emulator reference.

JAXAtari は各ゲームのロジックを JAX で再実装しており、オリジナル ROM をまったく実行できない。また両者はいずれも RL 環境であって、エミュレータ参照実装に対してビット単位で検証されたゲートレベルの微分可能移植ではない。
```

[A Differentiable Atari VCS (2026/06), Section 2 Related Work]

- **Octax**: CHIP-8 であって Atari 2600 ではない (§5.4 に引用)。
- **jaxtari / jutari** (arXiv:2606.22447, 2026/06): xitari に対し 64/64 ゲームで RAM・画面のビット単位一致を主張する唯一の新規 ALE 互換 GPU 実行系である。GTX 1080 Ti で N=4096 のとき約 2.95M env-steps/s を報告している。ただし**単位が ALE と換算不能**である。

```
We measure soft-mode throughput as steps per second (one step = one CPU instruction) on a fixed ROM after just-in-time warm-up, on a single Apple M1 Max core (JAX CPU backend, Julia 1.12), reporting the median of repeated runs.

我々は soft モードのスループットを、固定 ROM 上で JIT ウォームアップ後の毎秒 step 数 (1 step = CPU 命令 1 個) として、Apple M1 Max の単一コア (JAX CPU バックエンド、Julia 1.12) で測定し、反復実行の中央値を報告する。
```

[A Differentiable Atari VCS (2026/06), Appendix F Throughput]

さらに、コードは未公開であり、目的は RL スループットではなく XAI 基盤である。

```
All effort numbers are derived from the project's git history, which—like the rest of the artifact—will be made public on acceptance of the paper

すべての工数の数値はプロジェクトの git 履歴から導出されており、これは成果物の他の部分と同様、論文の採録後に公開される予定である。
```

[A Differentiable Atari VCS (2026/06), Appendix G]

- **Madrona / Isaac Gym**: Madrona 公式サイトの環境一覧に Atari / ALE は含まれていない。Isaac Gym の Atari 適用例も本調査では確認できなかった。

### 6.6 判定

2026 年 9 月時点で、**「ALE を GPU 側で回す」方式は主流たり得ていない**。根拠は 4 点である。

1. 唯一の本格移植 CuLE は上流保守が停止しており (最終コミット 2022-11-17、公称 CUDA 10.0)、CPU 比の優位も 3.56 倍にとどまる。
2. ALE 公式はベクトル化に CPU の C++ マルチスレッドを選択し、v0.12.0 の「XLA GPU support」もエミュレーション自体は CPU 側に残している。
3. 2025〜2026 年の「GPU Atari」を名乗る新規実装は、ROM を実行しない再実装 (JAXAtari)、別プラットフォーム (Octax = CHIP-8)、あるいは査読前・コード未公開 (jaxtari) のいずれかである。
4. 他ドメインの GPU バッチシミュレータ (Madrona / Isaac Gym) に Atari 適用例がない。

---

## 7. サンプル効率志向の手法群は実時間軸でどこに来るか

### 7.1 対応表

「時間」列は各論文が**自ら報告した**値である。第三者論文が報告した値は出典ラベルで区別した。

| 手法 | ベンチ | RR (論文の定義) | 報告された壁時計時間 | ハード | 出典 |
|---|---|---|---|---|---|
| BBF | Atari-100k | RR=8 (既定) | 「約 6 時間」と「10 時間」の 2 記述 | 「single GPU」/「A100 の半分 + CPU 1 個」 | BBF |
| SR-SPR | Atari-100k | 最大 RR=16 | 原論文未確認。第三者値: 約 24 時間 / 1 run 5 GPU 時間 | A100 の 25% + CPU 1 個 (BBF による記述) | BBF / ICLR Blogposts 2024 |
| SPR | Atari-100k | updates per step = 2 | 4.6 時間 (拡張あり)、3.0 時間 (拡張なし) | P100 × 1 | SPR |
| EfficientZero | Atari-100k | 1.2 (BBF による記述) | 7 時間 | 「4 GPUs」型番なし | EfficientZero |
| EfficientZero V2 | Atari-100k | UTD = 1 | **記載なし** | — | EfficientZero V2 |
| DreamerV3 | Atari-100k | replay ratio = 128 (独自定義) | **0.1 A100 GPU 日** | A100 × 1 | DreamerV3 |
| DreamerV3 | **Atari-200M** | replay ratio = 32 (独自定義) | **7.7 A100 GPU 日** | A100 × 1 | DreamerV3 |
| MEME | Atari-57 | SPI = 6 | **記載なし** | TPUv4 2×2×1 + 推論 TPUv4 + actor 64 | MEME |
| BTR | Atari-200M | 1/64 | 12 時間未満、機種別 11.5〜52 時間、0.9 A100 GPU 日 | RTX 4090 + i9-13900k ほか | BTR |
| STORM | Atari-100k | 未確認 | 4.3 時間 | RTX 3090 × 1 | STORM |
| DIAMOND | Atari-100k | 未確認 | 約 2.9 日 | RTX 4090 × 1 (VRAM 約 12GB) | DIAMOND |
| EMERALD | Atari-100k (Breakout) | 未確認 | 17 時間 | RTX 3090 | EMERALD |
| Δ-IRIS | Atari-100k (Breakout) | — | 27 時間 (EMERALD による第三者報告) | RTX 3090 | EMERALD |
| IRIS | Atari-100k | — | 1 run 1 週間 (BBF) / 4.1 日 (DIAMOND) / 3.5 A100 GPU 日 (DreamerV3) | A100 の半分 (BBF による記述) | BBF / DIAMOND / DreamerV3 |
| SimPLe | Atari-100k | — | 500 時間 (SPR) / 5.0 A100 GPU 日 (DreamerV3) | P100 相当 (SPR) | SPR / DreamerV3 |
| Rainbow (controlled) | Atari-100k 設定 | — | 2.1 時間 (拡張あり)、1.4 時間 (拡張なし) | P100 × 1 | SPR |

### 7.2 BBF — タイトルの "Faster" は論文中で定義されていない

BBF の PMLR 版本文で "faster/Faster" が出現するのは 4 箇所のみであり、そのうちエージェント命名文に直付けされた唯一の定量値は壁時計時間である。

```
The culmination of our investigation is the Bigger, Better, Faster agent, or BBF in short, which achieves super-human performance on Atari 100K with about 6 hours on single GPU.

我々の調査の集大成が Bigger, Better, Faster エージェント、略して BBF であり、これは単一 GPU で約 6 時間で Atari 100K において超人的性能を達成する。
```

[BBF (2023/07), §4 Method 冒頭]

一方、論文の見出し的主張 (Figure 1) はサンプル効率の軸である。**したがって「Faster = 壁時計時間」とも「= サンプル効率」とも断定できない。タイトル語の定義そのものが論文に書かれていない、というのが原文に忠実な答えである。**

BBF の §5 は、Atari-100k 系で本調査が見つけた中で最も濃い壁時計時間の記述である。

```
Computational efficiency. As machine learning methods become more sophisticated, an often overlooked metric is their computational efficiency. Although EfficientZero trains in around 8.5 hours, it requires about 512 CPU cores and 4 distributed GPUs. IRIS uses half of an A100 GPU for a week per run. SR-SPR, at its highest replay ratio of 16, uses 25% of an A100 GPU and a single CPU for roughly 24 hours. Our BBF agent at replay ratio 8 takes only 10 hours with a single CPU and half of an A100 GPU. Thus, measured by GPU-hours, BBF provides the best trade-off between performance and compute (see Figure 2).

計算効率。機械学習手法が高度化するにつれ、しばしば見落とされる指標がその計算効率である。EfficientZero は約 8.5 時間で学習するが、およそ 512 の CPU コアと 4 基の分散 GPU を必要とする。IRIS は 1 run あたり A100 GPU の半分を 1 週間使う。SR-SPR は、その最大の replay ratio である 16 において、A100 GPU の 25% と単一 CPU をおよそ 24 時間使う。我々の BBF エージェントは replay ratio 8 において、単一 CPU と A100 GPU の半分でわずか 10 時間しかかからない。したがって GPU 時間で測ると、BBF は性能と計算のトレードオフにおいて最良のものを提供する (図 2 参照)。
```

[BBF (2023/07), §5 Analysis "Computational efficiency"]

**同一論文内で BBF 自身の所要時間が「約 6 時間」(§4) と「10 時間」(§5) の 2 通りに書かれており、論文はこれを説明していない。**

BBF は RR の選択理由として計算量を明示的に挙げている。

```
As we expect that many users will not wish to pay the computational costs of running at replay ratio 8, we also present results for ablations at replay ratio 2 (matching SPR).

多くのユーザが replay ratio 8 で実行する計算コストを支払いたがらないと予想されるため、我々は replay ratio 2 (SPR に一致) でのアブレーション結果も提示する。
```

[BBF (2023/07), §4 "Base agent"]

### 7.3 SPR — Atari-100k 系で壁時計時間の付録節を最初に立てた例

SPR は付録に "WALL CLOCK TIMES" という専用節を持ち、他手法との対応表を掲げている。

```
Table 8: Wall-clock runtimes for various algorithms for a complete training and evaluation run on a single Atari game using a P100 GPU.

Model / Runtime in hours (100k env steps)
SPR 4.6
Rainbow (controlled) 2.1
SPR (No aug) 3.0
Rainbow (controlled, no aug) 1.4
SimPLe 500

表 8: P100 GPU を用いた単一 Atari ゲームでの完全な学習・評価 run に対する、各種アルゴリズムの壁時計ランタイム。
モデル / ランタイム (時間) (100k env steps)
SPR 4.6 / Rainbow (controlled) 2.1 / SPR (拡張なし) 3.0 / Rainbow (controlled, 拡張なし) 1.4 / SimPLe 500
```

[SPR (2021/05), Table 8, p.18]

### 7.4 DreamerV3 — GPU 日を表の列として持っている

DreamerV3 はベンチマーク別の GPU 日を表の 1 列として持っており、Atari-200M と Atari-100k が同じ表に並ぶため両者のコスト差がそのまま読める。

```
Benchmark Tasks EnvSteps ActionRepeat EnvInstances ReplayRatio GPUDays ModelSize
Atari 57 200M 4 16 32 7.7 200M
Atari100K 26 400K 4 1 128 0.1 200M

Table 2: Benchmark overview. All agents were trained on a single Nvidia A100 GPU each.

ベンチマーク タスク数 環境ステップ アクションリピート 環境インスタンス数 リプレイ比 GPU日 モデルサイズ
Atari 57 200M 4 16 32 7.7 200M
Atari100K 26 400K 4 1 128 0.1 200M

表 2: ベンチマーク概要。全エージェントはそれぞれ単一の Nvidia A100 GPU で学習された。
```

[DreamerV3 (2024/04), Table 2, p.19]

ただし DreamerV3 の replay ratio は「勾配ステップ / 環境ステップ」ではなく独自定義であり、他手法の RR と単位が違う。

```
We parameterize the amount of training via the replay ratio. This is the fraction of time steps trained on per time step collected from the environment, without action repeat. ... For example, a replay ratio of 32 on Atari with action repeat of 4 and batch shape 16 × 64 corresponds to 1 gradient step every 128 environment steps, or 1.5 million gradient steps over 200 million environment steps.

我々は学習量を replay ratio でパラメータ化する。これは、環境から収集した 1 タイムステップあたりに学習されるタイムステップの割合であり、アクションリピートを含まない。……例えば Atari でアクションリピート 4、バッチ形状 16 × 64 のときの replay ratio 32 は、128 環境ステップごとに 1 勾配ステップ、すなわち 2 億環境ステップにわたって 150 万勾配ステップに相当する。
```

[DreamerV3-Nature (2025/04), Methods "Replay ratio"]

### 7.5 記載がないことを確認できたもの

**MEME** は Appendix F に "Compute Resources" 節を持つが、記載されているのはハードウェア構成とスループット (learner 毎秒 3.8 更新、actor が毎秒約 12970 frames を書き込む) のみで、総壁時計時間の記述がない。第三者 (BTR) も "Not Reported" と記録している。

```
A100 GPU Days / BTR: 0.9 / MEME: Not Reported / Dreamer-v3: 7.7

A100 GPU 日数 / BTR: 0.9 / MEME: 未報告 / Dreamer-v3: 7.7
```

[BTR (2025/05), Table 3]

**EfficientZero V2** は Atari-100k について壁時計時間を報告していない。時間の記述があるのは DMControl "Walker Run" の付録 J.3 のみ (8×RTX 3090 で 100k 学習あたり 2.7 時間) である。

### 7.6 replay ratio と壁時計時間のトレードオフ

このトレードオフを明示的に論じている出典は複数ある。最も切れ味があるのは BRO の非対称性の指摘である (ただし BRO の実験は DMControl であって Atari ではない)。

```
We additionally note that the replay ratio has a bigger impact on wallclock time than the model size. This stems from the fact that scaling replay ratio leads to inherently sequential calculations, whereas scaling model size leads to calculations that can be parallelized.

さらに、replay ratio はモデルサイズよりも壁時計時間に大きな影響を与えることを指摘する。これは、replay ratio のスケーリングが本質的に逐次的な計算を招くのに対し、モデルサイズのスケーリングは並列化可能な計算を招くという事実に由来する。
```

[BRO (2024/05), §3 Analysis]

```
For example, a 5 million parameter BRO model with RR=1 outperforms a 1 million parameter BRO agent with RR=15 despite being five times faster in terms of wall-clock time. This observation challenges the notion that a sample-efficient RL algorithm must use high replay settings.

例えば、RR=1 の 500 万パラメータの BRO モデルは、RR=15 の 100 万パラメータの BRO エージェントを上回りつつ、壁時計時間では 5 倍速い。この観察は、サンプル効率的な RL アルゴリズムは高いリプレイ設定を用いなければならないという通念に異議を唱えるものである。
```

[BRO (2024/05), Appendix B.4]

Atari 側で同種の判断をしているのは MEME、BTR、BBF である。MEME は SPI を上げると壁時計時間の観点で割に合わなくなると述べている。

```
This implies that with an SPI of 10 we obtain a much worse return in terms of wall-clock time as we replay more frequently.

これは、SPI が 10 ではより頻繁にリプレイする分、壁時計時間の観点で得られるリターンがはるかに悪くなることを意味する。
```

[MEME (2022/09), Appendix K]

---

## 8. 分散前提の手法を比較対象から外してよいか

### 8.1 判定表

判定基準は「論文が報告している**最小構成**が、単一マシン 1 台かつ GPU 1〜数台に収まるか」である。actor を数百台の別マシンに置くことを前提にしている場合は「不適」とした。

| 手法 | actor / worker 数 | アクセラレータ | マシン台数 | 壁時計時間 | 対象 frames | 判定 |
|---|---|---|---|---|---|---|
| Ape-X | 360 actor マシン | Tesla P100 × 1 (learner) | 360 + 2 | 5 日 | 22,800M | 不適 |
| R2D2 | 256 actor | GPU × 1 (V100) | 257 | 5 日 | 37.5B | 不適 |
| Agent57 | 256 actor | GPU × 1 (型番なし) | 257 | **記載なし** | 5B で 51 ゲーム、78B で Skiing | 不適 |
| MuZero | 自己対戦 TPU × 32 | 第 3 世代 Cloud TPU 計 40 | TPU 40 台分 | 12 時間 | 20.0B | 不適 |
| SEED RL (Atari 最速) | 610 actor | TPU v3 × 8 コア | actor 群 + TPU ホスト | 1.8 日 | 記載なし | 不適 |
| IMPALA (分散版) | 500 CPU | P100 × 1 | 複数 | DMLab-30 で約 10 時間 | — | 不適 |
| IMPALA (Atari 実験) | **記載なし** | **記載なし** | **記載なし** | 1 時間未満 | 200M | **判定不能** |
| IMPALA (Single-Machine 行) | 48 actor | P100 × 1 | 1 台 | 時間の記載なし (DMLab で 21K/24K FPS) | — | 条件付きで妥当 (計測は DMLab) |
| Podracer Sebulba | 環境はホスト CPU | 8-core TPU (1 ホスト) | **1 台** | **約 1 時間 / 約 2.88 ドル** | 200M | 条件付きで妥当 (TPU) |
| Podracer Sebulba + MuZero | 同上 | 16-core TPU | 同上 | 9 時間 / 約 40 ドル | 200M | 条件付きで妥当 (TPU) |
| Stooke & Abbeel (1 GPU) | なし (5 CPU コア) | P100 × 1 | **1 台** | A2C 3.8h / DQN-512 8.3h / ε-Rainbow 14.1h | 200M | **妥当** |
| rlpyt R2D1 | なし (24 CPU コア) | Titan-Xp × 3 | **1 台** | 138 時間未満 | 40B 規模 | 妥当 |

### 8.2 Podracer Sebulba — 壁時計時間とドル建てコストを両方書いている

Podracer は単一アクセラレータでの Atari 実測を、コストとともに明記している数少ない出典である。

```
However, we found that training an agent for 200 million frames of an Atari game could be done in just ~1 hour, by running Sebulba on a 8-core TPU. This comes at a cost of approximately 2.88 dollars, on GCP's pre-emptible instances. This is similar in cost to training with the more complex SEED RL framework, and much cheaper than training an agent for 200 million Atari frames using either IMPALA or single-stream GPU-based system such as that traditionally used by DQN.

しかしながら、Sebulba を 8 コアの TPU 上で走らせることで、Atari ゲームの 2 億フレームのエージェント学習がわずか約 1 時間で行えることが分かった。これは GCP のプリエンプティブル・インスタンスで約 2.88 ドルのコストである。これは、より複雑な SEED RL フレームワークでの学習と同程度のコストであり、IMPALA や、DQN が伝統的に用いてきたような単一ストリームの GPU ベースのシステムを用いて 2 億 Atari フレームのエージェントを学習させるよりもはるかに安い。
```

[Podracer (2021/04), §4.2 Sebulba]

```
Training a MuZero agent with Sebulba for 200M Atari frames takes 9 hours on a 16-core TPU (at a cost of ~40 $ on GCP's preemptible instances).

Sebulba で MuZero エージェントを 200M Atari フレーム学習させるのは 16 コアの TPU で 9 時間かかる (GCP のプリエンプティブル・インスタンスで約 40 ドルのコスト)。
```

[Podracer (2021/04), §4.2 Sebulba]

なお、この約 1 時間がどのゲーム・どの到達スコアなのかは論文に記載がない。**壁時計時間だけがあり、品質軸が欠けている。**

Anakin は環境自体が JAX で書かれている必要があるため Atari には適用できない。

```
The Anakin framework supports environments that are themselves written in Jax, and hence can run efficiently on the TPU devices. The Sebulba framework supports arbitrary environments (such as Atari video games) that run on the CPU hosts.

Anakin フレームワークは、それ自体が JAX で書かれた環境をサポートし、それゆえ TPU デバイス上で効率的に実行できる。Sebulba フレームワークは、CPU ホスト上で実行される任意の環境 (Atari ビデオゲームなど) をサポートする。
```

[Podracer (2021/04), §3]

### 8.3 実装構成とアルゴリズムを分けて扱うべきもの

R2D2 と MuZero は「原論文の実装構成」が分散なだけであり、**同じアルゴリズムを単一マシンで再現した報告が別の一次論文に存在する**。

```
Notably, rlpyt reproduces record-setting results in the Atari domain from "Recurrent Experience Replay in Distributed Reinforcement Learning" (R2D2). This benchmark requires on the order of 30 billion frames of game play and 1 million network updates, which rlpyt achieves in reasonable time without the use of distributed compute infrastructure.

特筆すべきは、rlpyt が "Recurrent Experience Replay in Distributed Reinforcement Learning" (R2D2) における Atari ドメインの記録的な結果を再現していることである。このベンチマークは 300 億フレーム規模のゲームプレイと 100 万回のネットワーク更新を必要とするが、rlpyt は分散計算インフラを用いずに妥当な時間でこれを達成する。
```

[rlpyt (2019/09), §1 Introduction]

MuZero については §8.2 の Sebulba 引用が対応する。したがって「R2D2 / MuZero というアルゴリズムは単一マシンでは無理」という主張は、これらの出典に照らすと成立しない。分散なのは原論文の実装構成であって、アルゴリズムの必要条件ではない。

### 8.4 IMPALA の Atari「1 時間未満」は構成不明で使えない

IMPALA には Atari について 200M frames を 1 時間未満とする記述が 1 文だけある。

```
Note that the shallow IMPALA experiment completes training over 200 million frames in less than one hour.

浅い (shallow) IMPALA の実験は 2 億フレームにわたる学習を 1 時間未満で完了することに注意されたい。
```

[IMPALA (2018/02), §5.3.2 Atari]

**この 1 文には actor 数・GPU 本数・マシン台数の記載がなく、Appendix G のハイパーパラメータ表にも actor 数がない。** したがってこの数値を単一マシン構成に紐付ける根拠は本論文中にない。なお IMPALA の Table 1 に "Single-Machine" 区分は存在するが、計測は DeepMind Lab であって Atari ではない。

---

## 9. 2025〜2026 の動向

「Atari を単一 GPU / 消費者向けハードで実時間短縮する」ことを主題または重要指標とする研究は、2025〜2026 年に複数存在する。

### 9.1 RISE (2025/12) — 「デスクトップ 1 台・1 日以内」を達成条件に据えた

BTR と同じ第一著者による後続研究で、壁時計制約を明示的な達成条件にしている。

```
When our framework is applied to Beyond The Rainbow (BTR) (Clark et al., 2024), to our knowledge, this produces the highest performance algorithm capable of running on a single high-end desktop PC within a day of walltime.

我々のフレームワークを Beyond The Rainbow (BTR) (Clark et al., 2024) に適用すると、我々の知る限り、単一のハイエンドデスクトップ PC 上で 1 日以内の壁時計時間で実行可能なアルゴリズムとしては最高性能となる。
```

[RISE (2025/12), §1 Introduction]

```
Figure 3 shows that RISE can achieve a 22.9% performance improvement over that of a typical recurrent model and uses 84% less walltime. We found that Rainbow DQN was unable to benefit from both RISE and a typical LSTM; however, RISE still saved 40 hours of walltime in comparison.

図 3 は、RISE が典型的な再帰モデルに対して 22.9% の性能改善を達成し、壁時計時間を 84% 削減することを示す。Rainbow DQN は RISE と典型的な LSTM のどちらからも恩恵を受けられなかったが、それでも RISE は比較において 40 時間の壁時計時間を節約した。
```

[RISE (2025/12), Section 5]

実験は Atari-5 サブセット、RTX4090 搭載デスクトップである。比較図の軸として "A100 walltime hours" を採用している点も特徴的である。

### 9.2 Squeezing More from the Stream (2026/02) — CPU 4 コアでの壁時計開示

streaming RL の文脈で、GPU を使わない Atari 実験の壁時計時間を開示している珍しい例である。

```
To run these experiments (both with and without SPR), we used 4 CPU cores with 4GB RAM. The networks and the training code was written in JAX. As shown in Table 6, the streaming DQN and Stream Q(lambda) agents completed 40M frames of training in roughly 20 to 35 hours, while the QRC(lambda) variants took longer, taking about 35 to 50 hours.

これらの実験 (SPR あり・なしの双方) には、4 CPU コアと 4GB RAM を用いた。ネットワークと訓練コードは JAX で書かれている。表 6 に示すとおり、streaming DQN および Stream Q(lambda) エージェントは 4000 万 frames の訓練をおよそ 20〜35 時間で完了し、QRC(lambda) の変種はより長く、約 35〜50 時間を要した。
```

[Nilaksh et al. (2026/02), Appendix]

### 9.3 Greener Deep Reinforcement Learning (2025/09) — 統一ハードでの横断ベンチ

Atari 10 ゲーム × 7〜8 アルゴリズムを 1 台のワークステーションで統一測定した研究であり、ハードウェアを型番まで開示している。

```
All experiments were conducted on a workstation equipped with an Intel Xeon W-2245 CPU, 128 GB of system memory, a 512 GB NVMe system disk, and an NVIDIA RTX A5000 GPU with 24 GB of GDDR6 memory. No other training load was simultaneously run on the machine. This setup allows us to accurately and consistently compare the performance of the algorithms.

すべての実験は、Intel Xeon W-2245 CPU、128 GB のシステムメモリ、512 GB の NVMe システムディスク、24 GB GDDR6 メモリの NVIDIA RTX A5000 GPU を備えたワークステーション上で実施した。このマシン上で他の訓練負荷を同時に実行することはなかった。この構成により、アルゴリズムの性能を正確かつ一貫して比較できる。
```

[Gardner et al. (2025/09), Section III]

### 9.4 壁時計時間を副次指標として掲げる 2 本

Hyperbolic RL は壁時計削減率をアブストラクトに含めている。

```
On ProcGen, we show that Hyper++ guarantees stable learning, outperforms prior hyperbolic agents, and reduces wall-clock time by approximately 30%. On Atari-5 with Double DQN, Hyper++ strongly outperforms Euclidean and hyperbolic baselines.

ProcGen において、Hyper++ が安定した学習を保証し、既存の双曲エージェントを上回り、壁時計時間を約 30% 削減することを示す。Double DQN を用いた Atari-5 では、Hyper++ はユークリッドおよび双曲のベースラインを大きく上回る。
```

[Klein et al. (2025/12, v2), Abstract]

TARL は壁時計時間の**増分**を主張に含めている。

```
We empirically demonstrate consistent improvements within discrete and continuous control algorithms across various benchmark environments without any hyperparameter tuning, including a 38.18% peak score gain on Atari-10, while incurring less than a 4% increase in wall-clock time.

ハイパーパラメータ調整なしで、様々なベンチマーク環境における離散・連続制御アルゴリズムの一貫した改善を実証する。Atari-10 でのピークスコア 38.18% の向上を含み、壁時計時間の増加は 4% 未満に留まる。
```

[TARL (2026/03, v2), Abstract]

Atari 実験のハードウェアは AMD Ryzen 9 7950X (32 コア) + RTX 4090 である [TARL (2026/03, v2), Appendix]。

### 9.5 Physical Atari (2026/05) — 実時間学習なので経験時間 = 壁時計時間

実機ロボットで Atari を学習するプラットフォーム論文である。実時間学習であるため、経験時間と壁時計時間が一致する。

```
We ran our agent on the Physical Atari platform to learn to play six games - Pong, Seaquest, MsPacman, Assault, Asterix, and Kangaroo - for five and a half hours. For each game, we repeated the experiment at least 4 times and report the results in Figure 9. The agent learned on all games, and its performance was consistent across multiple runs. These experiments took nearly 145 hours and required no interventions.

Physical Atari プラットフォーム上でエージェントを実行し、6 つのゲーム (Pong, Seaquest, MsPacman, Assault, Asterix, Kangaroo) を 5 時間半かけてプレイできるよう学習させた。各ゲームについて実験を少なくとも 4 回繰り返し、結果を図 9 に報告する。エージェントはすべてのゲームで学習し、その性能は複数回の run にわたって一貫していた。これらの実験は 145 時間近くを要し、人手の介入は不要だった。
```

[Physical Atari (2026/05), Section 5]

計算機は AMD Ryzen AI Max+ 395 チップ搭載の Framework デスクトップである [Physical Atari (2026/05), Section 2]。

---

## 10. コミュニティ signals

**本節の情報は査読を経ておらず、実験条件も検証されていない。§1〜§9 の公式論文・公式リポジトリの情報とは信頼度の階層が異なる。**

### 10.1 公式リポジトリの README (著者による記載だが査読なし)

PQN の公式実装 README は、論文本文より踏み込んだ時間主張を掲げている。

```
Using PQN on a single NVIDIA A40 (which has performance comparable to an RTX 3090), you can:
- Train an Atari agent for 200M frames within an hour (with environments running on a single CPU using Envpool, tested on an AMD EPYC 7513 32-Core Processor).

単一の NVIDIA A40 (RTX 3090 に匹敵する性能を持つ) で PQN を使うと、次のことができる:
- Atari エージェントを 200M フレーム、1 時間以内で学習 (環境は Envpool を用いて単一 CPU 上で実行、AMD EPYC 7513 32-Core Processor で検証)。
```

[purejaxql README (2026/09 取得), "🔥 Quick Stats" 節]

**この README は 400M frames について「約 4 時間」と述べており、論文 Table 3 / Figure 13 の「2 時間」と食い違う。** どちらも著者らによる公式資料であり、不一致の理由は出典未確認である。

### 10.2 Hacker News — 「単一パーソナルコンピュータ」の定義が争点になっている

「Atari を 1 台のマシンで数時間」という主題のスレッドが 2018 年に立ち、155 点・38 コメントを得ている (item 16905121, 2018-04-23)。スレッド内では「単一パーソナルコンピュータ」の定義そのものが問題視されている。

```
> a run that takes 1 hour on 720 cores can be run on the CPUs of a 48-core personal computer in 16 hours
Is calling a 48-core machine a "personal computer" a bit of a stretch, or am I missing something?

> 720 コアで 1 時間かかる run は、48 コアのパーソナルコンピュータの CPU で 16 時間で実行できる
48 コアのマシンを「パーソナルコンピュータ」と呼ぶのはやや無理があるのではないか、それとも何か見落としているだろうか?
```

[Hacker News item 16905121 (2018/04), コメント (ユーザ ironrabbit)]

BTR や PQN に関する Hacker News のスレッドは、Algolia 検索では見つからなかった。

### 10.3 Reddit / X は取得できなかった

r/reinforcementlearning および X (Twitter) については、検索 API 経由・直接 HTTP の双方が拒否された (HTTP 403 / クローラのドメイン非対応)。したがって「自宅マシンで何時間で回した」系の実測報告が存在するかどうかは、本レポートでは**判定不能**である。

### 10.4 個人ブログ・個人リポジトリ

いくつかの個人による DQN 再現記事・リポジトリを確認したが、**壁時計時間とハードウェアを両方明記した実測報告は本調査の範囲内では見つからなかった**。

---

## 11. 批判・懸念

### 11.1 出典間の数値の食い違い

本調査で検出した不一致を列挙する。いずれも理由を説明する記述を見つけられなかったため、両論併記としている。

| 項目 | 出典 A | 出典 B |
|---|---|---|
| Rainbow の 200M frames 所要時間 | Rainbow 原論文「約 10 日」(単一 GPU) | BTR「Rainbow DQN's 35 hours」 |
| PQN の 400M frames 所要時間 | PQN 論文 Table 3 / Figure 13「2 時間」 | purejaxql README / 公式ブログ「約 4 時間」 |
| PQN の測定 GPU | PQN 論文「NVIDIA A40」 | BTR Table A3「Walltime (A100)」 |
| PQN の Rainbow 比較値 | Table 3「Rainbow 100 時間」 | 本文「約 3 日/ゲーム」(= 72 時間) |
| EfficientZero の Atari-100k 所要時間 | EfficientZero「4 GPU で 7 時間」 | BBF「約 8.5 時間、512 CPU コアと 4 分散 GPU」 |
| BBF 自身の所要時間 | BBF §4「single GPU で約 6 時間」 | BBF §5「A100 の半分 + CPU 1 個で 10 時間」 |
| Revisiting Rainbow の 34,200 GPU 時間の日数換算 | 原文「(or 1425 days)」 | BTR の引用「(equivalent to 1435 days)」 |
| EnvPool のラップトップ speedup | 論文 Abstract「2.8x」 | 公式ドキュメント「約 3x」(Table 1 からの計算では 3.12 倍) |
| EfficientZero の GPU 日 | 自己申告「4 GPU × 7 時間」= 28 GPU 時間 | DreamerV3 Table 10「0.6 A100 GPU 日」= 14.4 GPU 時間 |

さらに、BBF が主張する EfficientZero の「512 CPU コア」という数値の出所を、EfficientZero 論文内に発見できなかった。

### 11.2 測定者バイアス

一次出典に絶対値がない実装が多いため (§5.5)、公開されている比較値の多くは**競合の著者が測ったもの**である。同じ実装が複数の第三者に測られると値が食い違う。RLlib の Atari スループットは、Sample Factory 論文では IMPALA 22,440 FPS (生フレーム、学習込み、36 コア)、TorchRL 論文 Table 5 では Breakout-v5 で 97 (単位不明、データ収集のみ、96 コア)、RLlib 自身の論文では Ape-X 160k environment frames/s (学習込み、ハードウェア不明) である。**これらは互いに比較可能ではない。**

### 11.3 ハード世代差が補正されていない

本レポートに登場するハードウェアは P100 (2016)、V100 (2017)、Titan-Xp (2017)、TPU v3 (2018)、Titan V (2017)、GTX 1080 (2016)、RTX 3090 (2020)、A100 (2020)、RTX 4090 (2022)、H100 (2022) と 6 年以上に散らばっている。**世代差を補正した横断比較を行った出典は本調査では見つからなかった。**

しかも BTR の Appendix G.2 では、GPU 性能の順序と壁時計時間の順序が一致していない。H100 (15 時間) が RTX 4090 (11.5 時間) より遅いという並びは、**BTR の実装においては**律速が CPU 側の環境並列実行にあることの直接証拠である (§2.5 の引用)。したがって、少なくとも env 側が律速する実装については、GPU 世代の比較記事を根拠に使うことはできない。

**ただしこれを「Atari 一般の性質」と読むのは誤りである。** BTR の実装は Python + gymnasium async の 64 env であり (BTR Table A3 の Backend 欄)、env 側の供給能力が十分に高い実装では律速が learner 側へ移りうる。BTR のハードウェア表が示すのは「BTR という実装の律速がどこにあるか」であって、「Atari というベンチマークの律速がどこにあるか」ではない。

### 11.4 「Atari」を名乗るが Atari でないものの混在

- gymnax / PureJaxRL: MinAtar であり実 ALE ではない。
- Octax: CHIP-8。論文自身が「Atari ゲームの代替」と述べている。
- JAXAtari: 各ゲームのロジックを JAX で再実装しており、オリジナル ROM を実行できない。

Octax 論文が比較として EnvPool ALE Pong を 25,000 steps/s と実測している一方、Octax 自身の 350,000 steps/s は CHIP-8 ゲームに対する値である。**14 倍という比較は異なるゲーム・異なる計算量に対するものであり、「JAX にすれば ALE が 14 倍速くなる」ことを意味しない。**

### 11.5 品質軸が欠けている数値

Podracer Sebulba の「200M frames を約 1 時間」には、どのゲームでどのスコアに到達したかの記載がない。IMPALA の「200M frames を 1 時間未満」も同様に構成とスコアの記載を欠く。**壁時計時間だけがあってスコアがない数値は、フロンティアの点として置けない。**

---

## 12. 総合評価

### 12.1 発注の 6 つの問いへの回答

**(1) 壁時計時間とハードを明記している Atari 論文は他にあるか。**

ある。本調査で本文を検査した 21 本のうち 13 本が壁時計時間を数値で述べ、11 本が GPU 型番まで書いている (§2.2)。ただし抄録レベルでは 1055 本中 7 本 (0.66%) にすぎず、探しに行かないと見つからない。単一 GPU・200M frames の比較対象として最も直接使えるのは **Stooke & Abbeel 2018 の Table 2** であり、GPU 台数別・アルゴリズム別の完走時間を表で与えている唯一の出典である。次点が **Daley & Amato (GTX 1080 で 9 時間)** と **Schmidt & Schmied (RTX 3090 で 10M frames を 7.5 時間)** である。

**(2) PQN の位置づけ。**

PQN は「品質 / 時間」の点としては極めて左下に来るが、**プロトコルが Machado 準拠でないため BTR 系の数値と直接比較できない**。公式 config は sticky actions なし + episodic life であり、BTR はこれを「比較には無効」と明示的に批判している。PQN 自身も、sticky ありの Dopamine 準拠設定では 800M frames・約 5 時間を要すると認めている。条件を揃えた BTR Table A3 では、PQN は 11 倍速いが IQM は 1/3 (3.86 対 12.18) である。すなわち **PQN は「同じ品質へ到達するまでの時間」ではなく「与えられた時間での品質」の軸で読むべき手法**であり、品質パリティ比較の相手にはならない。

**(3) envpool / JAX 系のスループット。**

EnvPool の公称 1,069,922 FPS は 256 コア DGX-A100 での**生フレーム・学習なし**の値であり、agent step 換算で約 267,000/s。学習込みの単一マシン値としては Sample Factory の 135,893 FPS (36 コア + RTX 2080Ti、agent step 換算 約 34,000/s) が参照点になる。**JAX 系の「N 倍速い」の大半は実 ALE に対する主張ではない**。実 ALE を JAX から回す経路は EnvPool の XLA インタフェースか Podracer Sebulba のみで、いずれも env 本体は CPU 上の C++ ALE のままである。なお 2026 年時点で最も実用的なはずの ALE 本体 `AtariVectorEnv` には**公開ベンチマーク数値が存在しない**。

**(4) ALE を GPU 側で回す実装は存在するか。**

存在するが、主流たり得ていない (§6.6)。CuLE は保守停止 (最終コミット 2022-11-17、CUDA 10.0) で CPU 比 3.56 倍にとどまり、ALE 公式は CPU マルチスレッドを選択した。ALE 互換を主張する唯一の新規 GPU 実行系 (jaxtari, 2026/06) は査読前・コード未公開・単位換算不能である。**律速構造が変わる見込みは現時点でない。**

**(5) BBF / SR-SPR / EfficientZero 系は実時間軸でどこに来るか。**

これらは **Atari-100k のベンチマークであり、200M frames とは別の土俵**である。同じ土俵には乗らない。100k 内での順序は BBF 6〜10 時間 (A100 の半分)、EfficientZero 7 時間 (4 GPU)、SPR 4.6 時間 (P100)、IRIS 1 週間 (A100 の半分)、SimPLe 500 時間。200M frames を単一 A100 で回した値としては DreamerV3 の **7.7 A100 GPU 日 (= 約 185 時間)** があり、これは BTR の RTX 4090 11.5 時間とは 1 桁以上離れている。「RR が高いから実時間は遅い」という理解は方向としては正しく、その機序を BRO が「RR は逐次計算、モデルサイズは並列計算」と定式化している。

**(6) Ape-X / R2D2 / Agent57 / MuZero を外してよいか。**

**原論文の実装構成としては外してよい** (§8.1)。いずれも最小構成が単一マシンを超える (Ape-X 360 actor マシン、R2D2 / Agent57 256 actor、MuZero は Atari 1 ゲームあたり TPU 40 台)。**ただし「アルゴリズムが分散必須」という理由付けは成立しない** — R2D2 は rlpyt の R2D1 として単一ワークステーション (24 CPU + Titan-Xp 3 台) で、MuZero は Podracer Sebulba として 16-core TPU 1 台・9 時間で再現されている。外す理由は「原論文が報告した構成が 1 台に収まらないから」であって、それ以上ではない。

### 12.2 参照点を BTR のままにしてよいか

**単一 GPU・200M frames・Machado 準拠プロトコル・ハードウェア型番の 4 条件を同時に満たす公開値は、本調査の範囲では BTR しか存在しない。** 他の候補はいずれか 1 つ以上を欠く。

| 候補 | 単一 GPU / 1 台 | 200M frames | Machado 準拠 | ハード型番 | 判定 |
|---|---|---|---|---|---|
| BTR (11.5h, RTX 4090) | ○ | ○ | ○ | ○ | **参照点として維持** |
| PQN (1h, A40) | ○ | ○ | **×** (sticky なし + episodic life) | ○ | プロトコルが違う |
| Stooke & Abbeel (DQN 8.3h, P100) | ○ | ○ | **×** (2018 年、sticky 以前) | ○ | 下限の参照として補助的に有効 |
| Podracer Sebulba (1h, 8-core TPU) | ○ (ただし TPU) | ○ | 記載なし | ○ | GPU での実測がない・スコアがない |
| DreamerV3 (7.7 A100 GPU 日) | ○ | ○ | 未確認 | ○ | 時間表記が GPU 日のみ |
| SEED RL (1.8 日) | **×** (610 actor) | 記載なし | 未確認 | ○ | 分散前提 |
| Daley & Amato (9h, GTX 1080) | ○ | ○ | 未確認 | ○ | プロトコル未確認・スコア軸の記載が薄い |
| Schmidt & Schmied (7.5h, RTX 3090) | ○ | **×** (10M) | 未確認 | ○ | frames 数が 20 分の 1 |

したがって **BTR を主参照点として維持するのが妥当**である。ただし補助的な参照として、次の 3 点を併置すると地図が立体的になる。

- **下限側**: Stooke & Abbeel の 1 GPU 列 (A2C 3.8h / DQN-512 8.3h / ε-Rainbow-512 14.1h)。2018 年の P100 1 枚で既にこの水準に達していたという事実は、「速さ」だけでは価値にならないことを示す。差別化は品質側にある。
- **上限側**: DreamerV3 の Atari-200M 7.7 A100 GPU 日。単一 A100 で 1 週間強という値が、モデルベース系の実時間コストの目安になる。
- **異種アクセラレータ側**: Podracer Sebulba の 8-core TPU で約 1 時間・2.88 ドル。GPU での再現値がないため直接比較はできないが、「1 台に収まる構成で 200M frames を 1 時間」という到達点が 2021 年に存在していたことは記録に値する。

### 12.3 この分野の構造について言えること

本調査を通じて、出典に基づいて次の 3 点が言える。

1. **本調査で壁時計時間を開示している実装の多くは、env 側 (CPU) が律速である。** BTR は H100 (15 時間) が RTX 4090 (11.5 時間) より遅いことを実測し、律速が環境の並列実行 (CPU コア数とプロセッサ速度) にあると明記している。ALE 公式ドキュメントも「num_envs を物理 CPU コア数に近づけよ」と述べており、同じ構造を前提にしている。PQN も env だけを 32 コア CPU に置いている。**これは「Atari では常に env が律速する」という命題ではなく、「これらの実装では env が律速している」という観測である。** env 側の供給能力を十分に上げれば律速は learner 側へ移り、そこから先は GPU 性能と learner 側の設計 (replay ratio、ネットワーク規模、精度) が効いてくる。律速がどちらにあるかは実装ごとに測って決めるべき事柄であり、本調査には両者を切り分けて測定した出典が無い (§13-18)。
2. **少なくとも env 側が律速している間は、高速化のレバーは env 側にある。** EnvPool → ALE 本体への C++ ベクトル化の取り込みという 2025〜2026 年の動きも、CuLE の GPU 移植が主流化しなかったことも、この構造から一貫して説明できる。逆に言えば、env 側を C++ で十分に速くした実装にとっては、これらの動きはもうレバーではない。
3. **壁時計時間を評価軸に据える研究は 2025〜2026 年に増えている。** RISE、TARL、Hyperbolic RL、Greener Deep RL、Squeezing More from the Stream、Physical Atari がいずれも壁時計時間を主張・制約・分析軸のいずれかに置いている。Rainbow が 2017 年に定式化した「ハードが違えば比較できないから書かない」という論拠は、少なくとも一部で覆されつつある。

---

## 13. 調査限界

1. **本文を検査した 21 本 (§2.2) は無作為標本ではない。** 「壁時計時間を書いていそうな論文」を狙って集めたものであり、母集団の開示率の推定には使えない。
2. **arXiv API の検索は抄録・タイトル・コメントのみを対象とし、本文全文検索ではない。** §2.1 の件数は付録での開示を捕捉していない。
3. **R2D2 の原論文本文を取得できなかった。** OpenReview がブラウザ検証 (HTTP 403) を返した。本レポートの R2D2 の数値はすべて rlpyt / SEED RL / MuZero の 3 本の一次論文からの二次記述であり、3 本は互いに整合しているが原論文で直接確認したものではない。
4. **SR-SPR の原論文全文も同じ理由で取得できなかった。** 計算コストの数値 (V100 で 4 GPU 日、Atari-100k RR16 で 1 run 5 GPU 時間) は ICLR Blogposts 2024 経由の二次情報で、原文未照合である。
5. **DQN 原論文 (Mnih et al., Nature 2015) の本文を直接確認できなかった** (有料)。Ape-X Table 1 の「9.5 days / 1 GPU」は Ape-X 側の集成であり、DQN 原論文自身の開示かどうかは未確認である。
6. **PQN 論文の Figure 4(d)(速度比較の対数軸棒グラフ)の数値が取得できなかった。** 個々の棒の値は本文にもキャプションにも記載がない。
7. **BTR の Figure 2 の凡例と Figure 6 は画像であり、数値を読み取れなかった。** 本文が言及する「Rainbow DQN's 35 hours」の測定条件も特定できなかった。
8. **BBF Figure 2 右パネル(Runtime 対 IQM)の各手法の具体的数値が読めなかった。** 図中の点は画像であり、対応する数値表が論文中に存在しない。
9. **ALE `AtariVectorEnv` の公称スループットは存在しない。** 公式ドキュメント・リリースノート・ベンチマークスクリプトのいずれにも結果数値がない。本調査では実測も行っていない。
10. **CleanRL の 1 run 実時間を確定できなかった。** W&B レポートが JavaScript 描画のためテキスト抽出できず、Open RL Benchmark 論文にも記載がない。
11. **CuLE のアブストラクト「155M frames per hour」と Table 1 の「41K〜155K FPS」の関係が算術的に合わない。** 155K FPS × 3600 = 558M raw frames/hour であり、frame skip を考慮しても一致しない。出典に説明がなく不明である。
12. **jaxtari の frame/s 換算ができない。** 論文は「1 step = CPU 命令 1 個」でしか報告しておらず、換算に必要な「1 フレームあたりの命令数」が記載されていない。
13. **Reddit / X のデータを一切取得できなかった** (HTTP 403 / クローラ非対応)。§10.3 の項目は判定不能である。
14. **被引用数は Semantic Scholar の 2026-09-09 時点の値であり、Google Scholar 等とは異なる。** SEED RL は API のレート制限で取得できなかった。CuLE は arXiv 版と NeurIPS 版でレコードが分裂しており統一値を確定できなかった。
15. **各文献の Atari 評価プロトコル (sticky actions / noop / episodic life / 評価 ε / 評価エピソード数) を網羅的には確認していない。** 本調査は壁時計軸に絞ったため、PQN と BTR 以外についてはプロトコル差の確認が不完全である。
16. **各論文のハードウェア世代差を補正した比較は行っていない** (§11.3)。同一条件で複数手法の壁時計時間を測り直した第三者ベンチマークも本調査では見つかっていない。
17. **2025〜2026 の網羅性は保証できない。** arXiv 抄録検索と Web 検索の組み合わせによるものであり、抄録に壁時計時間関連語を含めない論文は取りこぼしている可能性が高い。
18. **env 側 (CPU) と learner 側 (GPU) のどちらが律速かを切り分けて測定した出典を見つけられなかった。** BTR は「主要ボトルネックは環境の並列実行」と述べているが、これは定性的な記述であり、learner 側の稼働率や critical path の測定は示されていない。他の文献も同様で、律速の所在をプロファイルで示したものは本調査の範囲では存在しない。したがって §12.3-1 の「多くの実装で env 側が律速」は、各論文の定性的記述とハードウェア表の並びからの読み取りであって、測定に基づく判定ではない。

---

## 14. 付録A: 発注元プロジェクトの実測（調査対象外の自プロジェクト情報）

**本節は本調査の対象外である。** §1〜§13 は外部文献の事実のみを扱っており、本節の数値は一切そこに反映していない。本節は、本レポートを発注した anet-lab プロジェクト側の実測値を、比較の便のためだけに並べたものである。

### A.1 実測値

| 構成 | 200M frames | 50M step・ε=0 の eval (Breakout) |
|---|---|---|
| BTR (公開値、RTX 4090) | 11.5h | 602.01 |
| RR1 + Munchausen (RTX 5090) | 2.80h | 602.5 / 583.8 |
| RR2 + Munchausen (RTX 5090) | 4.85h | 未測定 (ε=0.01 で eval1 554.9) |

50M step・ε=0・800 エピソードでの断面評価が BTR の公開 CSV 最終値と同水準であることは、キャンペーン記録で確認されている。

```
**学習を止めた断面評価（`learner.enabled=false`）で BTR と同じ土俵の数字が出た。** 50M・ε=0・800 エピソードで **602.5 / 583.8**（学習 Run 2 本）。BTR の公開 CSV 最終値 **602.01**（200M frames = 50M step、eval ε=0）とほぼ同値である。
```

[docs/experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md (2026/09), まとめ 46]

RR2 の 4.85h と、同じ時間で RR1 が到達する地点はキャンペーン記録にある。

```
同じ 4.85h で RR1 は約 86M まで進み、85-90M 窓は 18.25%
```

[docs/experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md (2026/09), 探索ブロック 03]

RR1 の 50M step 所要時間 2.80h は、上記 4.85h / 86M からの整合的な値だが、リポジトリ内の実験記録には直接の記載がない (**引き継ぎ資料の記載であり、リポジトリ未文書化**)。

### A.2 律速の所在 — 本プロジェクトは GPU Learn 律速であり、BTR と構造が違う

**本プロジェクトの現行構成では、律速は GPU 側の Learn にある** (2026-09-09 ユーザー確認)。**この判定はリポジトリ内に文書化されておらず、出典は本セッションでのユーザー確認のみである。** なおリポジトリ内で「Learn が GPU compute-bound / GPU 稼働率 ~87%」と記述されているのは ImageCls についてであり (`docs/memo/done/033_imagecls_bf16_head_10prd.md`)、Atari の記述ではない。

引き継ぎ資料には Tracy 実測として `AtariEnv::Step` が 1288% (= 12.88 コア)、ネットワーク forward が 9.55%、`env.worker_threads=-1` (AUTO) は `min(lane, 論理16−2)` = 14 本という数値がある。**これらもリポジトリ未文書化である。** ただし 1288% は 14 本のワーカースレッドにわたる CPU 占有の合計であって、**クリティカルパスがどちらにあるかを決める量ではない**。env スレッドが CPU を使い切っていることと、壁時計を律速しているかどうかは別の命題である。したがって本レポートでは、この Tracy の数値を律速の根拠としては扱わない。

### A.3 本調査の結果と照らした位置づけ

- BTR との品質パリティ (602.5 / 583.8 対 602.01) を 2.80h で達成しており、公開値に対して **4.1 倍**である。差の出どころとして引き継ぎ資料が挙げるのは、128 env の C++ 直結 ALE (BTR は 64)、RR1 (BTR は 1/64 = RR4 相当)、世代差の 3 点である。
- **BTR が報告する律速構造 (環境の並列実行が主要ボトルネックであり、CPU コア数とプロセッサ速度が最も重要。§2.5 の引用) は、本プロジェクトには当てはまらない。** A.2 のとおり本プロジェクトは GPU Learn 律速である。これは矛盾ではなく、**律速が既に env 側から外れている**ことを意味する。C++ 直結 ALE と 128 env によって env 側の供給能力が上がり、BTR が直面していたボトルネックを追い越した形になっている。
- したがって、§12.3 で述べた「Atari の壁時計は GPU 性能でほぼ決まらない」という構造は、**BTR の実装 (Python + gymnasium async、64 env) に対する記述であって、あらゆる Atari 実装に成り立つ命題ではない。** env 側を C++ で十分に速くすれば律速は GPU 側へ移り、そこから先は GPU 性能と learner 側の設計 (replay ratio、ネットワーク規模、精度) が効いてくる。**§2.5 の BTR ハードウェア表を「Atari 一般の性質」として読むことはできない。**
- §12.2 の判定表に本プロジェクトの行を加えるとすれば、4 条件 (単一 GPU / 200M frames / Machado 準拠 / ハード型番) をすべて満たす。**ただし公開された査読済み出典ではないため、外部の参照点としては使えない。**

---

## 15. 出典

### 公式論文 — 単一 GPU / 壁時計時間

[Stooke & Abbeel, 2018/03] Adam Stooke, Pieter Abbeel (University of California, Berkeley). "Accelerated Methods for Deep Reinforcement Learning." arXiv:1803.02811 (v2)。会議発表の記載なし。被引用 143 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/1803.02811

[rlpyt, 2019/09] Adam Stooke, Pieter Abbeel (University of California, Berkeley). "rlpyt: A Research Code Base for Deep Reinforcement Learning in PyTorch." arXiv:1909.01500 (v2)。被引用 101 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/1909.01500

[Daley & Amato, 2021/11] Brett Daley, Christopher Amato (Khoury College of Computer Sciences, Northeastern University). "Human-Level Control without Server-Grade Hardware." arXiv:2111.01264。被引用 0 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2111.01264

[Schmidt & Schmied, 2021/11] Dominik Schmidt, Thomas Schmied (TU Wien). "Fast and Data-Efficient Training of Rainbow: an Experimental Study on Atari." arXiv:2111.10247。被引用 16 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2111.10247

[BTR, 2025/05] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare (University of Southampton). "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." arXiv:2411.03820 v2、ICML 2025 (PMLR 267:11064-11091)。被引用 10 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2411.03820

[RISE, 2025/12] Tyler Clark, Christine Evers, Jonathon Hare (University of Southampton). "Recurrent Off-Policy Deep Reinforcement Learning Doesn't Have to be Slow." arXiv:2512.20513。会議採録は確認できず。被引用 0。 https://arxiv.org/abs/2512.20513

### 公式論文 — 並列 env / 価値ベース

[PQN, 2025/04] Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou, Ivan Masmitja, Jakob Nicolaus Foerster, Mario Martin (Universitat Politècnica de Catalunya / University of Oxford / Barcelona Supercomputing Center / Institut de Ciències del Mar). "Simplifying Deep Temporal Difference Learning." arXiv:2407.04811 v6、ICLR 2025。被引用 87 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2407.04811

[purejaxql README, 2026/09 取得] PQN 公式実装リポジトリ。 https://github.com/mttga/purejaxql

[purejaxql `pqn_atari.yaml`, 2026/09 取得] PQN の公式 Atari 設定ファイル。 https://raw.githubusercontent.com/mttga/purejaxql/main/purejaxql/config/alg/pqn_atari.yaml

[Hadamax, 2025/05] Jacob E. Kooi, Zhao Yang, Vincent François-Lavet (Vrije Universiteit Amsterdam). "Hadamax Encoding: Elevating Performance in Model-Free Atari." arXiv:2505.15345 v2、NeurIPS 2025。被引用 7 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2505.15345

[Rainbow, 2017/10] Matteo Hessel ほか (DeepMind). "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv:1710.02298、AAAI 2018。被引用 2703 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/1710.02298

### 公式論文 — env 実行エンジン / フレームワーク

[EnvPool, 2022/06] Jiayi Weng ほか. "EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine." arXiv:2206.10558、NeurIPS 2022 Datasets and Benchmarks。被引用 84 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2206.10558

[EnvPool 公式ドキュメント Benchmark, 2026/09 取得] https://envpool.readthedocs.io/en/latest/content/benchmark.html

[Sample Factory, 2020/06] Aleksei Petrenko, Zhehui Huang, Tushar Kumar, Gaurav Sukhatme, Vladlen Koltun. "Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning." arXiv:2006.11751、ICML 2020。 https://arxiv.org/abs/2006.11751

[Cleanba, 2023/09] Shengyi Huang ほか. "Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform." arXiv:2310.00036、ICLR 2024。 https://arxiv.org/abs/2310.00036

[TorchBeast, 2019/10] Heinrich Küttler ほか. "TorchBeast: A PyTorch Platform for Distributed RL." arXiv:1910.03552。 https://arxiv.org/abs/1910.03552

[Acme, 2020/06] Matt Hoffman ほか (DeepMind). "Acme: A Research Framework for Distributed Reinforcement Learning." arXiv:2006.00979。 https://arxiv.org/abs/2006.00979

[TorchRL, 2023/06] Albert Bou ほか. "TorchRL: A data-driven decision-making library for PyTorch." arXiv:2306.00577、ICLR 2024。 https://arxiv.org/abs/2306.00577

[ALE v0.11.0 リリースノート, 2025/04] https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.11.0

[ALE v0.12.0 リリースノート, 2026/05] https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.12.0

[ALE Vector Environment Guide, 2026/09 取得] https://ale.farama.org/vector-environment/

[Farama-Foundation/Arcade-Learning-Environment `ale_vector_xla_interface.cpp`, 2026/09 取得] https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/main/src/ale/python/ale_vector_xla_interface.cpp

### 公式論文 — GPU エミュレーション / 新規実装

[CuLE, 2020/12] Steven Dalton, Iuri Frosio (NVIDIA). "Accelerating Reinforcement Learning through GPU Atari Emulation." NeurIPS 2020。arXiv 版 (2019/07) は Michael Garland を含む 3 名。被引用 9 (Semantic Scholar arXiv レコード, 2026-09-09)。 https://papers.nips.cc/paper/2020/file/e4d78a6b4d93e1d79241f7b282fa3413-Paper.pdf / https://arxiv.org/abs/1907.08467

[NVlabs/cule README.md, 2022/11 最終更新] https://github.com/NVlabs/cule

[A Differentiable Atari VCS, 2026/06] Andreas Maier, Siming Bayer, Patrick Krauss. "A Differentiable Atari VCS: A Complex, Fully Known Ground Truth for Explainable AI." arXiv:2606.22447。査読前・コード未公開。 https://arxiv.org/abs/2606.22447

[Octax, 2025/10] Waris Radji, Thomas Michel, Hector Piteau (Univ. Lille, Inria, CNRS, Centrale Lille, UMR 9189-CRIStAL). "Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX." arXiv:2510.01764 (v3 2026/07)。 https://arxiv.org/abs/2510.01764

[k4ntz/JAXAtari README.md, 2026/09 取得] TU Darmstadt AI/ML Lab。 https://github.com/k4ntz/JAXAtari

[Madrona Engine 公式サイト, 2026/09 取得] https://madrona-engine.github.io/

[PureJaxRL ブログ, 2026/09 取得] Chris Lu. https://chrislu.page/blog/meta-disco/

### 公式論文 — サンプル効率志向

[BBF, 2023/07] Max Schwarzer, Johan Obando-Ceron, Aaron Courville, Marc G. Bellemare, Rishabh Agarwal, Pablo Samuel Castro (Google DeepMind / Mila, Université de Montréal). "Bigger, Better, Faster: Human-level Atari with human-level efficiency." ICML 2023 (PMLR 202)。被引用 175 (Semantic Scholar, 2026-09-09)。 https://proceedings.mlr.press/v202/schwarzer23a/schwarzer23a.pdf

[SPR, 2021/05] Max Schwarzer ほか (Mila / Microsoft Research). "Data-Efficient Reinforcement Learning with Self-Predictive Representations." ICLR 2021。被引用 439 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2007.05929

[SRSPR-ICLRvirtual, 2023/05] Pierluca D'Oro ほか. "Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier." ICLR 2023 (Oral)。**本文 PDF は 2026-09-09 時点でアクセス不可**。 https://iclr.cc/virtual/2023/poster/11457

[ICLRblog2024, 2024/05] "It's Time to Move On: Primacy Bias and Why It Helps to Forget." ICLR Blogposts 2024。著者名未確認。 https://iclr-blogposts.github.io/2024/blog/primacy-bias-and-why-it-helps-to-forget/

[EfficientZero, 2021/11] Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao. "Mastering Atari Games with Limited Data." NeurIPS 2021。被引用 340 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2111.00210

[EfficientZero V2, 2024/03] Shengjie Wang ほか. "EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data." ICML 2024。被引用 43。 https://arxiv.org/abs/2403.00564

[DreamerV3, 2024/04] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. "Mastering Diverse Domains through World Models." arXiv:2301.04104 v2。被引用 1373 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2301.04104

[DreamerV3-Nature, 2025/04] 同上. "Mastering diverse control tasks through world models." Nature, 2025. DOI: 10.1038/s41586-025-08744-2。 https://pmc.ncbi.nlm.nih.gov/articles/PMC12003158/

[MEME, 2022/09] Steven Kapturowski ほか (DeepMind). "Human-level Atari 200x faster." arXiv:2209.07550、ICLR 2023。被引用 44 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/2209.07550

[DIAMOND, 2024/05] Eloi Alonso ほか. "Diffusion for World Modeling: Visual Details Matter in Atari." NeurIPS 2024。被引用 317。 https://arxiv.org/abs/2405.12399

[STORM, 2023/10] Weipu Zhang ほか. "STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning." NeurIPS 2023。被引用 146。 https://arxiv.org/abs/2310.09615

[EMERALD, 2025/07] Maxime Burchi, Radu Timofte. "Accurate and Efficient World Modeling with Masked Latent Transformers." arXiv:2507.04075。被引用 4。 https://arxiv.org/abs/2507.04075

[BRO, 2024/05] Michal Nauman ほか. "Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control." NeurIPS 2024。被引用 115。**対象は DMControl であり Atari ではない**。 https://arxiv.org/abs/2405.16158

### 公式論文 — 分散前提

[Ape-X, 2018/03] Dan Horgan ほか (DeepMind). "Distributed Prioritized Experience Replay." ICLR 2018。被引用 841 (Semantic Scholar, 2026-09-09)。 https://arxiv.org/abs/1803.00933

[R2D2, 2019] Steven Kapturowski ほか (DeepMind). "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR 2019。**本文取得不可** (OpenReview がブラウザ検証を要求)。被引用 581。 https://openreview.net/forum?id=r1lyTjAqYX

[Agent57, 2020/03] Adrià Puigdomènech Badia ほか (DeepMind). "Agent57: Outperforming the Atari Human Benchmark." ICML 2020。被引用 598。 https://arxiv.org/abs/2003.13350

[MuZero, 2019/11] Julian Schrittwieser ほか (DeepMind). "Mastering Atari, Go, chess and shogi by planning with a learned model." Nature。被引用 2708。 https://arxiv.org/abs/1911.08265

[IMPALA, 2018/02] Lasse Espeholt ほか (DeepMind). "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures." ICML 2018。被引用 1920。 https://arxiv.org/abs/1802.01561

[SEED RL, 2019/10] Lasse Espeholt ほか (Google Research, Brain Team). "SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference." ICLR 2020。被引用数不明 (API レート制限)。 https://arxiv.org/abs/1910.06591

[Podracer, 2021/04] Matteo Hessel ほか (DeepMind). "Podracer architectures for scalable Reinforcement Learning." arXiv:2104.06272。テクニカルレポート。被引用 52。 https://arxiv.org/abs/2104.06272

### 公式論文 — 評価プロトコルと計算コストの議論

[Machado et al., 2017/09] Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." arXiv:1709.06009、JAIR 2018。被引用 630。 https://arxiv.org/abs/1709.06009

[Revisiting Rainbow, 2020/11] Johan S. Obando-Ceron, Pablo Samuel Castro (Google Research, Brain Team). "Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research." arXiv:2011.14826、ICML 2021。被引用 133。 https://arxiv.org/abs/2011.14826

[Agarwal et al., 2021/08] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021 Outstanding Paper。被引用 1005。 https://arxiv.org/abs/2108.13264

[Dopamine, 2018/12] Pablo Samuel Castro ほか (Google Brain). "Dopamine: A Research Framework for Deep Reinforcement Learning." arXiv:1812.06110。被引用 297。 https://arxiv.org/abs/1812.06110

[SimPLe, 2019/03] Łukasz Kaiser ほか. "Model-Based Reinforcement Learning for Atari." arXiv:1903.00374、ICLR 2020。被引用 1011。 https://arxiv.org/abs/1903.00374

[DreamerV2, 2020/10] Danijar Hafner ほか. "Mastering Atari with Discrete World Models." arXiv:2010.02193、ICLR 2021。被引用 1346。 https://arxiv.org/abs/2010.02193

[Such et al., 2017/12] Felipe Petroski Such ほか (Uber AI Labs). "Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning." arXiv:1712.06567。被引用 770。 https://arxiv.org/abs/1712.06567

### 2025〜2026 の文献

[Nilaksh et al., 2026/02] Nilaksh, Antoine Clavaud, Mathieu Reymond, François Rivest, Sarath Chandar (Mila - Quebec AI Institute ほか). "Squeezing More from the Stream: Learning Representation Online for Streaming Reinforcement Learning." arXiv:2602.09396。被引用 0。 https://arxiv.org/abs/2602.09396

[Gardner et al., 2025/09] Jason Gardner, Ayan Dutta, Swapnoneel Roy, O. Patrick Kreidl (University of North Florida), Ladislau Boloni (University of Central Florida). "Greener Deep Reinforcement Learning: Analysis of Energy and Carbon Efficiency Across Atari Benchmarks." arXiv:2509.05273、Artificial Intelligence Review。被引用 3。 https://arxiv.org/abs/2509.05273

[Klein et al., 2025/12, v2] Timo Klein ほか (University of Vienna ほか). "Understanding and Improving Hyperbolic Deep Reinforcement Learning." arXiv:2512.14202 (v2 2026/03)。被引用 0。 https://arxiv.org/abs/2512.14202

[TARL, 2026/03, v2] Leonard S. Pleiss (Technical University of Munich), James Harrison (Google DeepMind), Maximilian Schiffer (Technical University of Munich). "Target-Aligned Reinforcement Learning." arXiv:2603.29501 (v2 2026/05)。被引用 0。 https://arxiv.org/abs/2603.29501

[Physical Atari, 2026/05] Khurram Javed, Joseph Modayil, Gloria Kennickell, Richard S. Sutton, John Carmack. "Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots." 所属不明。arXiv:2606.19357。被引用 1。 https://arxiv.org/abs/2606.19357

### 第三者・コミュニティ情報（信頼度は公式情報より低い）

[Hacker News item 16905121, 2018/04] "Accelerating Deep Neuroevolution: Train Atari in Hours on a Single Computer." 155 points、38 comments。 https://news.ycombinator.com/item?id=16905121

[Uber Engineering Blog, 2018/04] Jeff Clune, Kenneth O. Stanley, Felipe Petroski Such. "Accelerating Deep Neuroevolution: Train Atari in Hours on a Single Personal Computer." https://www.uber.com/en-AT/blog/accelerated-neuroevolution/

[CleanRL docs (PQN), 2026/09 取得] CleanRL による PyTorch 再実装のドキュメント。**Runtime 表に単位の記載がないため本レポートの壁時計比較には使用していない**。 https://docs.cleanrl.dev/rl-algorithms/pqn/

### 検索基盤・書誌データベース

[arXiv API, 2026/09] export.arxiv.org/api/query。§2.1 の件数はすべて 2026-09-09 実行。 https://info.arxiv.org/help/api/index.html

[Semantic Scholar Graph API, 2026/09] 被引用数はすべて 2026-09-09 取得。 https://api.semanticscholar.org/

[GitHub REST API, 2026/09] NVlabs/cule および Farama-Foundation/Arcade-Learning-Environment のリポジトリ状態・コミット・issue・fork の取得元。 https://api.github.com/

### 本レポート内で参照した自プロジェクト文書（§14 のみ）

[docs/experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md, 2026/09] BTR 差分の腕別スクリーニング記録。

[docs/experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md, 2026/09] 高 replay_ratio で成績が落ちる機序の記録。

[reports/btr_hyperparams_survey_2026-08-26.md, 2026/08] BTR のハイパーパラメータと Appendix G の壁時計表の転記。本レポートの出発点。
