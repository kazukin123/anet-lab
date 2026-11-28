# Survey: BTR (Beyond The Rainbow) — ハイパーパラメータと Atari 評価プロトコル

Date: 2026-08-26
Scope: arXiv:2411.03820 (ICML 2025 poster) の学習ハイパーパラメータ、ネットワーク構造、Atari 前処理・評価プロトコル、報告スコアの出所、ablation を原典から確定させる。あわせて、v5 世代における NoOp reset の扱いが実装系統でどう分裂しているかを一次資料横断で確認する。調査は論文 HTML 版（v2）、公式実装リポジトリのソース、各 RL ライブラリの公式ドキュメントとソース、Machado et al. 2018 原典に基づく。

## 目次

1. [書誌情報と調査方法](#1-書誌情報と調査方法)
2. [アルゴリズム構成 — Rainbow DQN からの差分](#2-アルゴリズム構成--rainbow-dqn-からの差分)
3. [ハイパーパラメータ](#3-ハイパーパラメータ)
   - Table D6 全文 / optimizer / target network / 勾配クリップ / replay ratio / PER
4. [ネットワーク構造](#4-ネットワーク構造)
   - Impala 幅倍率 / Adaptive Maxpool / Spectral Normalization の適用範囲 / IQN の tau / パラメータ数
5. [Atari 環境設定と前処理](#5-atari-環境設定と前処理)
   - Table D4 / noop_max / sticky / life 系 / frame skip / wrapper 実装
6. [v5 世代における NoOp reset の系統分裂](#6-v5-世代における-noop-reset-の系統分裂)
   - ale-py raw env / Gymnasium AtariPreprocessing / Machado et al. 2018 の立場 / 実装横断対応表
7. [評価プロトコル](#7-評価プロトコル)
   - 評価エピソード数・epsilon・間隔 / 評価対象ネットワーク / seed 数 / best checkpoint
8. [報告スコアと出所](#8-報告スコアと出所)
   - Breakout 682 / 比較列の出所 / raw score と human-normalized の別
9. [Ablation](#9-ablation)
   - Impala +142% IQM / Table 2 (Phoenix) / vectorization と maxpool の位置づけ
10. [論文とコードの食い違い](#10-論文とコードの食い違い)
11. [批判・懸念](#11-批判懸念)
12. [総合評価](#12-総合評価)
13. [調査限界](#13-調査限界)
14. [付録A: anet-lab 現行設定との差分（調査対象外の自プロジェクト情報）](#14-付録a-anet-lab-現行設定との差分調査対象外の自プロジェクト情報)
15. [出典](#15-出典)

---

## 1. 書誌情報と調査方法

論文の正式タイトルは "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC"、著者は Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare の 4 名。arXiv には 2024/11/06 に v1、2025/05/21 に v2 が投稿されており、v2 が ICML 2025 (Poster) 採択版である。被引用数は Semantic Scholar 上で 10 件（influential citation 1 件）。**著者の所属機関は arXiv abs ページ上では確認できなかった。**

公式実装は論文アブストラクト内で明示されている。

```
Code is available at https://github.com/VIPTankz/BTR.

コードは https://github.com/VIPTankz/BTR で入手できる。
```

[BTR (2025/05), Abstract]

本調査では、論文に記載がない項目・論文と実装が食い違う項目を切り分けるため、**論文と公式実装の双方を突き合わせた**。以下の各節では、論文にしか無い情報・コードにしか無い情報・両者が一致する情報を区別して記す。

信頼度の階層として、§6 では公式ドキュメント（層1）、公式リポジトリのソース（層2）、原論文（層3）を分けて扱う。第三者の解説記事・ブログは根拠として使用していない。

---

## 2. アルゴリズム構成 — Rainbow DQN からの差分

BTR は Rainbow DQN を基点とし、6 要素を追加、4 要素を維持、2 要素を削除した構成である。Double DQN の削除理由は、Munchausen RL が次状態の argmax を使わないため機能的に不要になることである。

```
Added To Rainbow DQN            Same As Rainbow DQN            Removed From Rainbow DQN
Impala (Scale=2)                N-Step TD Learning             Double (N/A with Munchausen)
Adaptive Maxpooling (6x6)       Prioritized Experience Replay  C51 (Upgraded to IQN)
Spectral Normalization          Dueling
Implicit Quantile Networks      Noisy Networks
Munchausen
Vectorized Environments

Rainbow DQN への追加             Rainbow DQN と同一              Rainbow DQN から削除
Impala (スケール=2)              N ステップ TD 学習              Double (Munchausen と併用で該当なし)
Adaptive Maxpooling (6x6)        優先度付き経験再生              C51 (IQN へ更新)
スペクトル正規化                 Dueling
Implicit Quantile Networks       Noisy Networks
Munchausen
ベクトル化環境
```

[BTR (2025/05), Table 1]

```
As Munchausen does not use argmax over the next state, Double DQN is obsolete.

Munchausen は次状態に対する argmax を使用しないため、Double DQN は不要となる。
```

[BTR (2025/05), Section 3.1]

**Noisy Networks は維持されている。** Dueling・PER・Noisy Networks の除去実験の結果、PER と Noisy Networks は有益と判明して残され、Dueling は有意差が無かったが Rainbow DQN からの継続性のため残された。

```
Lastly we also tried removing some of the original components from Rainbow DQN on Atari BattleZone, including Dueling, Prioritized Experience Replay and Noisy Networks. Prioritized Experience Replay and Noisy Networks both proved beneficial, so were kept in the algorithm. Dueling did not seem to make any significant difference, however we did not choose to remove it for a clearer continuation of Rainbow DQN, in addition to potentially being useful in other Atari environments.

最後に、Atari BattleZone において Rainbow DQN の元の構成要素のいくつか、すなわち Dueling、優先度付き経験再生、Noisy Networks を除去することも試みた。優先度付き経験再生と Noisy Networks はいずれも有益であることが示されたため、アルゴリズム内に残された。Dueling は有意な差を生まないように見えたが、Rainbow DQN からのより明確な継続性のため、また他の Atari 環境で有用である可能性を考え、除去しないことを選んだ。
```

[BTR (2025/05), Appendix H]

---

## 3. ハイパーパラメータ

### 3.1 Table D6 全文

論文が与えるハイパーパラメータの一覧は以下が全文である。

```
Table D6: Table showing the hyperparameters used in the BTR algorithm.

Learning Rate                          1e-4
Discount Rate                          0.997
N-Step                                 3
IQN Taus                               8
IQN Number Cos'                        64
Huber Loss κ                           1.0
Gradient Clipping Max Norm             10
Parallel Environments                  64
Gradient Step Every                    64 Environment Steps (1 Vectorized Environment Step)
Replace Target Network Frequency (C)   500 Gradient Steps (32K Environment Steps)
Batch Size                             256
Total Replay Ratio                     1/64
Impala Width Scale                     2
Spectral Normalization                 All Convolutional Residual Layers
Adaptive Maxpooling Size               6x6
Linear Size (Per Dueling Layer)        512
Noisy Networks σ                       0.5
Activation Function                    ReLu
ε-greedy Start                         1.0
ε-greedy Decay                         8M Frames
ε-greedy End                           0.01
ε-greedy Disabled                      100M Frames
Replay Buffer Size                     1,048,576 Transitions (2^20)
Minimum Replay Size for Sampling       200K Transitions
PER Alpha                              0.2
Optimizer                              Adam
Adam Epsilon Parameter                 1.95e-5 (equal to 0.005/batchsize)
Adam β1                                0.9
Adam β2                                0.999
Munchausen Temperature τ               0.03
Munchausen Scaling Term α              0.9
Munchausen Clipping Value (l0)         -1.0
Evaluation Epsilon                     0.01 until 125M frames, then 0
Evaluation Episodes                    100
Evaluation Every                       1M Environment Frames (250K Environment Steps)

表 D6: BTR アルゴリズムで使用されたハイパーパラメータを示す表。

学習率                                  1e-4
割引率                                  0.997
N ステップ                              3
IQN Tau 数                              8
IQN cos の個数                          64
Huber 損失 κ                            1.0
勾配クリッピング最大ノルム              10
並列環境数                              64
勾配ステップの間隔                      64 環境ステップごと (1 ベクトル化環境ステップ)
ターゲットネットワーク置換頻度 (C)      500 勾配ステップ (32K 環境ステップ)
バッチサイズ                            256
総リプレイ比                            1/64
Impala 幅スケール                       2
スペクトル正規化                        すべての畳み込み残差層
Adaptive Maxpooling サイズ              6x6
線形層サイズ (Dueling の各層あたり)     512
Noisy Networks σ                        0.5
活性化関数                              ReLu
ε-greedy 開始値                         1.0
ε-greedy 減衰                           8M フレーム
ε-greedy 終了値                         0.01
ε-greedy 無効化                         100M フレーム
リプレイバッファサイズ                  1,048,576 遷移 (2^20)
サンプリング開始に必要な最小リプレイ量  200K 遷移
PER Alpha                               0.2
オプティマイザ                          Adam
Adam Epsilon パラメータ                 1.95e-5 (0.005/batchsize に等しい)
Adam β1                                 0.9
Adam β2                                 0.999
Munchausen 温度 τ                       0.03
Munchausen スケーリング項 α             0.9
Munchausen クリッピング値 (l0)          -1.0
評価時 Epsilon                          125M フレームまで 0.01、その後 0
評価エピソード数                        100
評価間隔                                100 万環境フレーム (25 万環境ステップ)
```

[BTR (2025/05), Table D6 (Appendix D.2)]

### 3.2 Optimizer と weight decay

本体の optimizer は素の Adam であり、**Table D6 に weight decay の行が存在しない**。学習率は Rainbow DQN の 6.25e-5 よりやや高い 1e-4 が最良だったと本文が述べている。

```
Given our high batch size, we additionally performed minor hyperparameter tests using different learning rates finding that a slightly higher learning rate of 1×10^-4 performed best, compared to 6.25×10^-5 in Rainbow DQN.

大きなバッチサイズを用いていることから、我々はさらに異なる学習率を用いた小規模なハイパーパラメータ試験を行い、Rainbow DQN の 6.25×10^-5 と比べてやや高い 1×10^-4 の学習率が最良であることを見出した。
```

[BTR (2025/05), Section 3.2]

実装の既定分岐も `weight_decay` 引数を渡しておらず 0 である。Adam epsilon は batch size に連動する定義になっている。

```python
self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015
```

[BTR-code (2026/08), `Agent.py`]

weight decay については、Appendix H に AdamW（decay 1e-4）の試行が「有意差なし」として記録されている。

```
Using the AdamW optimizer (Loshchilov & Hutter, 2019) which uses weight decay with the decay parameter 1e-4, however found this made no significant difference.

AdamW オプティマイザ (Loshchilov & Hutter, 2019) を用いた。これは decay パラメータ 1e-4 の weight decay を使用するものだが、有意な差は生じなかった。
```

[BTR (2025/05), Appendix H]

### 3.3 Target network の更新方式

**hard update（重みの丸ごとコピー）であり、500 gradient steps ごと**である。これは 32,000 環境ステップ（128,000 フレーム）に相当する。

```
Firstly, how frequently the target network is updated is closely intertwined with batch size and replay ratio. We found that updating the target network every 500 gradient steps [footnote: This equates to 32,000 environment steps (128,000 frames), compared to Rainbow DQN's 8,000 steps.] performed best.

第一に、ターゲットネットワークの更新頻度はバッチサイズおよびリプレイ比と密接に絡み合っている。我々は、ターゲットネットワークを 500 勾配ステップごとに更新すること[脚注: これは 32,000 環境ステップ (128,000 フレーム) に相当し、Rainbow DQN の 8,000 ステップと対比される]が最良の性能を示すことを見出した。
```

[BTR (2025/05), Section 3.2]

**指数移動平均（soft update）は試行済みで、不採用である。**

```
Using Exponential Moving Average networks rather than using fixed target networks (this was both computationally slower and performed worse).

Varying the frequency of updating the target network (we tested 250, 500 and 1000, finding 500 to perform best).

固定ターゲットネットワークの代わりに指数移動平均ネットワークを用いること (これは計算的により遅く、かつ性能も劣った)。

ターゲットネットワークの更新頻度を変化させること (250、500、1000 を試験し、500 が最良の性能を示すことを見出した)。
```

[BTR (2025/05), Appendix H]

### 3.4 勾配クリップ

最大ノルム 10 のクリップが有効である（Table D6 の "Gradient Clipping Max Norm 10"）。実装も同値。

```python
torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
```

[BTR-code (2026/08), `Agent.py`]

### 3.5 Replay ratio

論文は「64 並列環境で 1 ステップ進め、batch size 256 で 1 回の勾配更新」と記述し、その比 1/64 を replay ratio と呼んでいる。高い replay ratio が性能を改善することは認めたうえで、実時間を削るために低く保つ選択をしたと明記している。

```
We follow Schmidt & Schmied (2021), taking 1 step in 64 parallel environments with one gradient update with batch size 256 (Schmidt & Schmied (2021) took two gradient updates). This results in a replay ratio (ratio of gradient updates to environment steps) of 1/64. Higher replay ratios have been shown to improve performance (D'Oro et al., 2022), however we opt to keep this value low to reduce walltime.

我々は Schmidt & Schmied (2021) に従い、64 個の並列環境で 1 ステップを進め、バッチサイズ 256 で 1 回の勾配更新を行う (Schmidt & Schmied (2021) は 2 回の勾配更新を行っていた)。この結果、リプレイ比 (勾配更新数と環境ステップ数の比) は 1/64 となる。より高いリプレイ比が性能を改善することは示されているが (D'Oro et al., 2022)、我々は実時間を削減するためこの値を低く保つことを選択した。
```

[BTR (2025/05), Section 3.1]

200M フレーム学習における総勾配ステップ数は 781,000 であると本文が述べている。

```
Omitting vectorization increases walltime by 328% (Figure 6) by processing environment steps in parallel and taking fewer gradient steps (781,000 compared to Rainbow DQN's 12.5 million).

ベクトル化を省くと壁時計時間が 328% 増加する (図6)。これは環境ステップを並列処理し、より少ない勾配ステップ (Rainbow DQN の 1250 万に対して 78 万 1000) で済ませるためである。
```

[BTR (2025/05), Section 5.1]

### 3.6 PER の α・β・priority 計算

α = 0.2 は本文でも明示され、IQN 併用時に Toromanoff et al. (2019) が推奨する低めの値を採ったと説明されている。

```
For our Prioritized Experience Replay, we use the lower value of α=0.2, the parameter used to determine sample priority, recommended by Toromanoff et al. (2019) when using IQN.

我々の優先度付き経験再生では、サンプル優先度を決定するパラメータについて、IQN を用いる場合に Toromanoff et al. (2019) が推奨する低めの値 α=0.2 を用いる。
```

[BTR (2025/05), Section 3.2]

**β（重要度サンプリング指数）は論文に一切記述がない。** 実装には `per_beta = 0.45` が定義されアニールは既定 OFF だが、重要度サンプリング重みの計算式は β ではなく **α** を指数に使っており、コード内コメントがこれを手違い由来と明記している。

```python
weights = (self.capacity * probs) ** -self.alpha  # self.beta originally this was an accident but actually performed better
```

[BTR-code (2026/08), `PER.py`]

priority の計算式も論文には無い。実装では quantile Huber loss の TD 誤差の絶対値を tau 次元で総和し平均した値に 1e-6 を加え、α 乗して和木に格納している。

Appendix H には、α を持たない Dopamine 版 PER が中程度に劣ったという記録がある。

```
Using Dopamine's Prioritized Experience Replay buffer which doesn't include a α value (moderately worse performance).

α 値を含まない Dopamine の優先度付き経験再生バッファを用いること (中程度に性能が悪化した)。
```

[BTR (2025/05), Appendix H]

---

## 4. ネットワーク構造

### 4.1 Impala 幅倍率と Adaptive Maxpool

幅スケール 2、adaptive maxpool 出力 6×6。3 ブロックのチャネル数は [16×width, 32×width, 32×width]、畳み込み部の出力は 6×6×32×width となる。

```
Lastly, the sizes of many of the layers given in Figure E7 are dependent upon the Impala width scale, of which we use the value 2. For example, the Impala CNN blocks have [16×width, 32×width, 32×width] channels respectively. The output size of the convolutional layers (including the maxpooling layer) is 6×6×32×width, as a 6x6 maxpooling layer is used.

最後に、図 E7 に示された多くの層のサイズは Impala 幅スケールに依存しており、我々は値 2 を用いる。例えば、Impala CNN ブロックのチャネル数はそれぞれ [16×width, 32×width, 32×width] である。畳み込み層 (maxpooling 層を含む) の出力サイズは、6x6 の maxpooling 層を用いるため 6×6×32×width となる。
```

[BTR (2025/05), Appendix E.1]

adaptive maxpool の採用理由は、学習の高速化と入力解像度非依存性である。

```
We include an additional 6x6 adaptive max pooling layer after the convolutional layers (Schmidt & Schmied, 2021), which was found to speed up learning and support different input resolutions. The adaptive maxpooling is identical to a standard 2D maxpooling layer, but can be used with any input resolution as it automatically adjusts the stride and kernel size to fit a specified output size.

我々は畳み込み層の後に追加の 6x6 adaptive max pooling 層を含める (Schmidt & Schmied, 2021)。これは学習を高速化し、異なる入力解像度に対応できることが分かっている。adaptive maxpooling は標準的な 2D maxpooling 層と同一であるが、指定された出力サイズに合わせてストライドとカーネルサイズを自動的に調整するため、任意の入力解像度で使用できる。
```

[BTR (2025/05), Section 3.1]

### 4.2 Spectral Normalization の適用範囲

Table D6 は "All Convolutional Residual Layers" と記す。より具体的には、各 Impala CNN ブロック内の residual layer（Conv 3x3 + ReLU が 2 つ）に適用される。

```
Within each Impala CNN block, each residual layer (containing two Conv 3x3 + ReLu) has spectral normalization applied, as discussed in Section 3.1.

第 3.1 節で述べたとおり、各 Impala CNN ブロック内の各残差層 (Conv 3x3 + ReLu を 2 つ含む) にスペクトル正規化が適用される。
```

[BTR (2025/05), Appendix E.1]

**実装では block 先頭のダウンサンプル用 conv には適用されておらず、residual 内の 2 conv のみが対象である。** したがって対象は 3 ブロック × 2 residual × 2 conv = 12 層になる。

線形層への適用は試行済みで、大幅に劣化したと記録されている。

```
Applying spectral normalization to the linear layers (dramatically worse performance).

線形層にスペクトル正規化を適用すること (劇的に性能が悪化した)。
```

[BTR (2025/05), Appendix H]

### 4.3 IQN の tau

**原典が与える tau 数は "IQN Taus 8" の単一値のみであり、current / target / policy を区別した記載は無い。** 論文は Munchausen の τ との記号衝突を避けるため IQN 側を σ に置き換えると明記しており、N / N′ / K という記号も使っていない。

```
It is also worth noting here that due to the character conflict of both Munchausen and IQN using τ (Munchausen as a temperature parameter, and IQN for drawing samples), we replace IQN's τ with σ.

ここで注記すべきは、Munchausen と IQN がともに τ を用いること (Munchausen は温度パラメータとして、IQN はサンプルを引くために) による記号の衝突のため、我々は IQN の τ を σ で置き換えているという点である。
```

[BTR (2025/05), Appendix E.2]

実装では online / target / 行動選択のすべてが同一の `num_tau`（既定 8）を用いる。cos 埋め込みの個数は 64。

### 4.4 Munchausen のパラメータと損失

α = 0.9、τ = 0.03、l0 = −1.0。これらは Munchausen 原論文と同一の値を用いると明記されている。

```
l0, τ and α are hyperparameters set by Munchausen. We use the same values in BTR, also shown in our hyperparameter table in Appendix D.2.

l0、τ、α は Munchausen によって設定されるハイパーパラメータである。BTR でも同じ値を用いており、それらは Appendix D.2 のハイパーパラメータ表にも示されている。
```

[BTR (2025/05), Appendix E.2]

### 4.5 パラメータ数

総パラメータ数は 291 万で、そのうち 252 万が線形層にある。

```
Figure E7: Architectural diagram of the BTR algorithm's neural network. The model contains a total of 2.91 million parameters, 2.52 million of which are within linear layers.

図 E7: BTR アルゴリズムのニューラルネットワークの構成図。モデルは合計 291 万個のパラメータを含み、そのうち 252 万個は線形層内にある。
```

[BTR (2025/05), Figure E7 caption]

---

## 5. Atari 環境設定と前処理

### 5.1 Table D4 全文

論文が与える環境設定は 8 項目のみである。**noop も frame skip も max pooling も含まれない。**

```
Table D4: Environment Details for Atari Experiments.

Grey-Scaling                 True
Observation down-sampling    84x84
Frames Stacked               4
Reward Clipping              [-1, 1]
Terminal on loss of life     False
Life Information             False
Max frames per episode       108K
Sticky Actions               True

表 D4: Atari 実験の環境詳細。

グレースケール化              True
観測のダウンサンプリング      84x84
スタックするフレーム数        4
報酬クリッピング              [-1, 1]
ライフ喪失での終端            False
ライフ情報                    False
1 エピソードあたり最大フレーム数  108K
Sticky Actions                True
```

[BTR (2025/05), Table D4 (Appendix D.1)]

### 5.2 noop_max — 論文に記載なし、コードで 30

**論文本文・付録・全表のどこにも `no-op` / `noop` / `random start` の語が出現しない。** 一方、公式実装は Gymnasium の `AtariPreprocessing` を fork した自前ファイルを使い、その既定 `noop_max=30` を上書きしていない。

```python
def __init__(
    self,
    env: gym.Env,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int = 84,
    terminal_on_life_loss: bool = False,
    life_information: bool = True,
    grayscale_obs: bool = True,
    grayscale_newaxis: bool = False,
    scale_obs: bool = False,
):
```

[BTR-code (2026/08), `AtariPreprocessingCustom.py`]

環境生成側が渡すキーワード引数は `life_information` のみで、`noop_max` は既定のまま残る。

```python
def make_env(envs_create, game, life_info, framestack, repeat_probs):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        AtariPreprocessingCustom(gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs), life_information=life_info), framestack,
        lz4_compress=False) for _ in range(envs_create)], context="spawn")
```

[BTR-code (2026/08), `main.py`]

適用は `reset()` 内で、1 以上 `noop_max` 以下の一様整数ぶん NOOP を踏む。

```python
noops = (
    self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
    if self.noop_max > 0
    else 0
)
for _ in range(noops):
    _, _, terminated, truncated, step_info = self.env.step(0)
```

[BTR-code (2026/08), `AtariPreprocessingCustom.py`]

**したがって BTR は sticky actions 0.25 と NoOp reset (max 30) を併用している。** 論文が noop に言及しないため、この併用が論文中で説明されることはない。論文は評価が Machado et al. (2018) に従うと述べている。

```
We evaluate BTR on the Atari-60 benchmark following (Machado et al., 2018) and without life information (see Appendix I for the impact), evaluating every million frames on 100 episodes.

我々は BTR を Atari-60 ベンチマーク上で Machado et al. (2018) に従って評価し、ライフ情報は使用しない（影響については Appendix I を参照）。評価は 100 万フレームごとに 100 エピソードで行う。
```

[BTR (2025/05), Section 4.1]

### 5.3 sticky actions の確率値

論文は Table D4 で `True` と記すのみで、確率値を書いていない。実装で 0.25 が確定する（`--sticky` 既定 1 から `repeat_probs = 0.25`）。sticky が既定である旨は Appendix I にも明記される。

```
Figure I10: Graph shows Atari-5 performance with and without sticky actions (sticky actions is the default) using Inter-quartile mean and Atari-60 predicted median from Aitchison et al. (2023). No Sticky Actions uses a single seed, so this result should be used with caution.

図 I10: sticky actions の有無による Atari-5 の性能を、四分位平均および Aitchison et al. (2023) による Atari-60 予測中央値で示したグラフ（sticky actions が既定）。sticky actions 無しは単一 seed のため、この結果は慎重に扱うべきである。
```

[BTR (2025/05), Appendix I]

### 5.4 life 系の扱い

episodic life（terminal on loss of life）と life information はいずれも不採用。論文は life information を用いた研究とは比較不能であると論じている。

```
Some prior works choose to pass life information to the agent (Schmidt & Schmied, 2021). To clarify, this is different to terminal on loss of life. Life information does not reset the episode upon losing a life, but does pass a terminal to the buffer, allowing the agent to experience further into episodes while also giving the agent a negative signal for losing a life. This setting is not recommended in Machado et al. (2018), and works which use it are not comparable to those which don't.

一部の先行研究はライフ情報をエージェントに渡すことを選んでいる (Schmidt & Schmied, 2021)。明確にしておくと、これは「ライフ喪失時の終端」とは異なる。ライフ情報はライフを失ってもエピソードをリセットしないが、バッファには終端を渡す。これによりエージェントはエピソードのより先まで経験でき、同時にライフ喪失に対する負のシグナルも得る。この設定は Machado et al. (2018) では推奨されておらず、これを用いた研究はそうでない研究と比較可能ではない。
```

[BTR (2025/05), Appendix I]

リポジトリの README は、このオプションが性能を大きく変えることを述べている。

```
Also note that although we use a custom atari environment, this is exactly the same as the standard by default. We also however add a --life_info option, which passes a terminal to the agent on life loss, but does not reset the episode. Using this will drastically improve performance on games with lives.

なお、我々はカスタムの Atari 環境を使用しているが、既定では標準のものと完全に同一である。ただし --life_info オプションを追加しており、これはライフ喪失時にエージェントへ終端を渡すがエピソードはリセットしない。これを使うとライフのあるゲームで性能が劇的に向上する。
```

[BTR-code (2026/08), `README.md`]

### 5.5 frame skip・max pooling・その他

frame skip 4 は Table D4 に無く、本文の間接記述と実装で確定する。ALE 内部の frameskip は 1 に固定され、wrapper 側で 4 を適用する。max pooling は直近 2 フレームの要素ごと最大値。resize の補間は `cv2.INTER_AREA`（論文に記載なし、実装のみ）。

```
When using the standard Atari wrapper, training for 200M frames is equivalent to training for 50M steps.

標準的な Atari wrapper を使用する場合、2億フレームの学習は 5000万ステップの学習に相当する。
```

[BTR (2025/05), Appendix D.3]

**fire reset は論文・コードのいずれにも存在しない**（不採用）。

reward clipping は学習ループ内でのみ適用され、**スコア集計はクリップ前**に行われる。したがって報告スコアは raw score である。

### 5.6 wrapper 実装の系統

Gymnasium の `AtariPreprocessing` を丸ごと複製し `life_information` 引数のみ追加した自前ファイルである。docstring は Gymnasium 版のまま残っている。Stable-Baselines3 / CleanRL / Dopamine のいずれも使用していない。

```
"""Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018."""

「Machado et al., 2018 のガイドラインに従った Atari 2600 前処理の実装。」
```

[BTR-code (2026/08), `AtariPreprocessingCustom.py`]

---

## 6. v5 世代における NoOp reset の系統分裂

本節は BTR 固有ではなく、Atari RL 一般における noop の扱いを一次資料横断で確認したものである。

### 6.1 ale-py の v5 env に noop は含まれない

v5 環境の登録 kwargs は 4 つのみで、noop に相当するキーは無い。

```python
gymnasium.register(
    id=f"ALE/{name}-v5",
    entry_point="ale_py.env:AtariEnv",
    vector_entry_point="ale_py.vector_env:AtariVectorEnv",
    kwargs=dict(
        game=rom,
        repeat_action_probability=0.25,
        full_action_space=False,
        frameskip=4,
        max_num_frames_per_episode=108_000,
    ),
)
```

[ale-py-src (2026/08), `src/ale/python/registration.py`]

`AtariEnv.__init__` にも noop 系引数は存在せず、`reset()` は `self.ale.reset_game()` を呼ぶだけである。**raw env は noop start を含まない。** 公式ドキュメントも v5 の変更点として sticky の復活と stochastic frame-skipping の廃止のみを挙げ、noop には言及していない。

```
Stickiness was added back and stochastic frame-skipping was removed. The environments are now in the "ALE" namespace.

スティッキー性が復活し、確率的フレームスキップが削除された。環境は "ALE" 名前空間に移された。
```

[ALE-docs-Pong (2026/08), Version History]

### 6.2 Gymnasium `AtariPreprocessing` の既定は noop_max = 30

公式ドキュメントとソースの両方で `noop_max=30` が既定である。

```python
class gymnasium.wrappers.AtariPreprocessing(
    env: Env,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int | tuple[int, int] = 84,
    terminal_on_life_loss: bool = False,
    grayscale_obs: bool = True,
    grayscale_newaxis: bool = False,
    scale_obs: bool = False,
)
```

[Gymnasium-docs (2026/08), `AtariPreprocessing`]

### 6.3 Machado et al. 2018 における sticky と no-op の位置づけ

sticky actions は標準的な学習・評価プロトコルとして提案され、ALE の新バージョンに組み込まれると述べられている。

```
we propose sticky actions as a standard training and evaluation protocol, which will be incorporated in a new version of the Arcade Learning Environment

我々はスティッキーアクションを標準的な学習・評価プロトコルとして提案する。これは Arcade Learning Environment の新しいバージョンに組み込まれる予定である。
```

[Machado+ (2017/12), Section 5.2]

initial no-ops は §5.3 で**代替手法**として、欠点付きで列挙されている。

```
Impact varies across games. For example, initial no-ops have no effect in Freeway.
The environment remains deterministic beyond the choice of starting state.

影響はゲームによって異なる。例えば、初期 no-op は Freeway では効果がない。
開始状態の選択を超えると、環境は依然として決定的なままである。
```

[Machado+ (2017/12), Section 5.3]

総括では、sticky が他手法の利点を取り込みつつ欠点の大半を回避するものとして提示されている。**位置づけは併用ではなく代替である。**

```
Our proposed solution, sticky actions, leverages some of the main benefits of other approaches without most of their drawbacks.

我々が提案する解決策であるスティッキーアクションは、他のアプローチの主要な利点のいくつかを、それらの欠点の大部分を伴わずに活かすものである。
```

[Machado+ (2017/12), Section 5.3]

**ただし「sticky を使う場合に no-op を併用してはならない」という明示的な禁止文言は確認できなかった。**

### 6.4 実装系統 × noop_max × sticky の対応表

| 実装系統 | 使用する env / 経路 | noop_max | sticky | Machado 準拠か |
|---|---|---|---|---|
| ale-py raw env (v5) | `ALE/<Game>-v5` | **機構自体が存在しない** | 0.25 | sticky のみ準拠 |
| ale-py raw env (v4) | `<Game>-v4` | 機構自体が存在しない | 0.00 | 非準拠 |
| Gymnasium `AtariPreprocessing` | wrapper | **30** | 下位 env に従う | noop を Machado 準拠として記述 |
| Dopamine（既定） | `ALE/<Game>-v5`, `sticky_actions=True` | **無し** | 0.25 | **準拠** |
| Dopamine（`use_ppo_preprocessing=True`） | 同上 | 30 | 0.25 | sticky + noop 併用 |
| Stable-Baselines3 `AtariWrapper` | wrapper | 30 | **0.0** | 非準拠 |
| CleanRL `dqn_atari.py` | `BreakoutNoFrameskip-v4` | 30 | 0.00 | 非準拠 |
| CleanRL `ppo_atari_envpool.py` | EnvPool `Breakout-v5` | 30 | 0 | **名前は v5 だが非準拠** |
| EnvPool Atari | `<Game>-v5`（独自 task_id） | 30 | 0 | 非準拠 |
| rlpyt `AtariEnv` | ALE 直結 | 30（区間 `[0,30]`） | 0. | 非準拠 |
| SPR | ALE 直結（rlpyt 派生） | 30（区間 `[1,30]`） | 0. | 非準拠 |
| dqn_zoo | 独自 `<game>-xitari-v1` | 1..30 | 0.0 | 非準拠 |
| **BTR** | `ALE/<Game>-v5` + Gymnasium wrapper fork | **30** | **0.25** | sticky 準拠・noop 併用 |

**noop_max の値そのものは 30 でほぼ一致しており、分裂しているのは「noop を使うか否か」と「sticky と組み合わせるか」である。** 「noop 無し + sticky ON」（Machado 原典の形）は Dopamine の既定のみで確認できた。

EnvPool は `-v5` サフィックスを使いながら `repeat_action_probability=0`（決定的な結果を得るため、と明記）かつ `noop_max=30` である。

```
noop_max (int): the maximum number of no-op action being executed when calling a single env.reset, default to 30
repeat_action_probability (float): the action repeat probability in ALE configuration, default to 0 (no action repeat to perform deterministic result)

noop_max (int): 単一の env.reset を呼び出す際に実行される no-op アクションの最大数。既定は 30。
repeat_action_probability (float): ALE 設定におけるアクション繰り返し確率。既定は 0（決定的な結果を得るためアクション繰り返しなし）。
```

[EnvPool-docs (2026/08), Atari — Options]

### 6.5 v5 世代の論文における noop の記述

MEME / DreamerV3 / BTR の 3 件を確認した。**3 件とも sticky actions の設定は明示するが、no-op starts については記述が確認できなかった。** MEME の環境ハイパーパラメータ表（Table 3）にも noop / random starts の行は無い。DreamerV3 は sticky を使う旨のみ述べ、使用した ALE バージョン・env id も本文中に確認できなかった。

---

## 7. 評価プロトコル

| 項目 | 値 | 出所 |
|---|---|---|
| 評価エピソード数 | 100 | 論文 Table D6 |
| 評価 epsilon | 125M フレームまで 0.01、以降 0 | 論文 Table D6 |
| 評価間隔 | 1M フレーム（250K env steps） | 論文 Table D6 |
| 評価対象ネットワーク | **online net**（NoisyNet ノイズ無効化） | **コードのみ** |
| seed 数 | Atari-60: 4 seeds / Atari-5: 3 seeds | 論文 Figure 1 / Table A2 caption |
| 報告値の checkpoint | **best**（学習中の最良評価） | 論文 脚注1 / Table A1 caption |
| 総フレーム数 | 200M frames（= 50M steps） | 論文 Abstract / Appendix D.3 |
| 壁時計時間 | RTX 4090 で 11.5 時間 | 論文 Appendix G |

### 7.1 評価対象ネットワーク

**online net である。** `agent.net`（online）の state_dict のみを評価プロセスへ渡しており、`agent.tgt_net` は評価経路に現れない。評価前に NoisyNet のノイズが無効化される。

```python
agent.disable_noise(agent.net)
net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
network_creator = deepcopy(agent.network_creator_fn)
```

[BTR-code (2026/08), `main.py`]

**この項目は論文に記載がない。**

### 7.2 報告値は best checkpoint

200M フレーム時点の値ではなく、学習中の最良評価を報告している旨が脚注で明示される。

```
All reported IQM scores use the best single evaluation for each environment throughout training as is standard, rather than the agent's score at 200 million, hence the discrepancy between the overall score and Figure 1.

報告される全ての IQM スコアは、標準的な方法に従い、2億フレーム時点でのエージェントのスコアではなく、学習全体を通じた各環境での最良の単一評価を使用している。したがって総合スコアと図1の間には食い違いが生じる。
```

[BTR (2025/05), Section 1, 脚注1]

### 7.3 壁時計時間

```
Original: RTX 4090, Intel i9-13900k (2023), 64GB RAM - 11.5 Hours
RTX 3070, Ryzen 9 3900X (2019), 64GB RAM - 52 Hours
RTX 2080 ti, Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz (2018), 128GB RAM - 32 Hours
Nvidia H100, 48 Core Intel(R) Xeon(R) Platinum 8468 (2023), 2TB RAM - 15 Hours
Nvidia A100, 24 Core Intel(R) Xeon(R) Gold 6336Y (2021), 512GB RAM - 22 Hours

オリジナル: RTX 4090, Intel i9-13900k (2023), 64GB RAM - 11.5 時間
RTX 3070, Ryzen 9 3900X (2019), 64GB RAM - 52 時間
RTX 2080 ti, Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz (2018), 128GB RAM - 32 時間
Nvidia H100, 48 Core Intel(R) Xeon(R) Platinum 8468 (2023), 2TB RAM - 15 時間
Nvidia A100, 24 Core Intel(R) Xeon(R) Gold 6336Y (2021), 512GB RAM - 22 時間
```

[BTR (2025/05), Appendix G]

---

## 8. 報告スコアと出所

### 8.1 Breakout 682 の性質

**682 は Table A1 の BTR 列の値であり、raw score、200M フレームまでの学習中に得られた最大スコア（100 エピソード平均）である。**

```
Table A1: Maximum scores obtained during training (averaged over 100 episodes and all performed using random seeds) after 200M Frames on the Atari-60 benchmark. Fast & Efficient Rainbow DQN and Munchausen-IQN refer to Schmidt & Schmied (2021) and (Vieillard et al., 2020) respectively. FE-Rainbow uses Life Information (See Appendix I), only 10M frames, and has missing games, so metrics are based on existing games.

表 A1: Atari-60 ベンチマークにおける 200M フレーム後の、学習中に得られた最大スコア（100 エピソードで平均、いずれもランダム seed で実施）。Fast & Efficient Rainbow DQN と Munchausen-IQN はそれぞれ Schmidt & Schmied (2021) と Vieillard et al. (2020) を指す。FE-Rainbow はライフ情報を使用し（Appendix I 参照）、10M フレームのみで、欠損ゲームがあるため、指標は存在するゲームに基づく。
```

[BTR (2025/05), Table A1 caption (Appendix A)]

同表の Breakout 行:

| Game | Random | Human | DQN (Nature) | Rainbow | M-IQN | FE-Rainbow | BTR |
|---|---|---|---|---|---|---|---|
| Breakout | 1 | 30 | 92 | 109 | 241 | 537 | **682** |

[BTR (2025/05), Table A1]

**リポジトリ公開の実測 CSV とは一致しない。** `results.csv`（200 回の評価、各回 100 エピソード平均）の Breakout 行は、最大 676.58、200M フレーム時点（最終評価）602.01、末尾 10 回平均 605.54 であった。**論文値 682 との差の理由は原典に説明がない。**

### 8.2 比較列の出所

Figure 1 の DQN / Rainbow は RLiable からの引用であると明記されている。

```
The results for DQN and Rainbow DQN are those reported in RLiable (Agarwal et al., 2021), and Dreamer-v3 refers to Hafner et al. (2023). Shaded areas show 95% bootstrapped confidence intervals, with BTR using 4 seeds.

DQN および Rainbow DQN の結果は RLiable (Agarwal et al., 2021) で報告されたものであり、Dreamer-v3 は Hafner et al. (2023) を指す。網掛け領域は 95% ブートストラップ信頼区間を示し、BTR は 4 seeds を使用している。
```

[BTR (2025/05), Figure 1 caption]

**一方 Table A1 の DQN (Nature) / Rainbow 列については、キャプションに出所の明記が無い。** キャプションが出所を示すのは M-IQN と FE-Rainbow の 2 列のみである。BTR 著者が自前で再実験したという記述も論文中に存在しない。

### 8.3 raw score と human-normalized の別

Table A1 の**各ゲーム行は raw score**（Random 列・Human 列が併記されている）。同表末尾の**集計行は human-normalized** である。

| 指標 | Random | Human | DQN (Nature) | Rainbow | M-IQN | FE-Rainbow | BTR |
|---|---|---|---|---|---|---|---|
| IQM (↑) | 0.000 | 1.000 | 0.771 | 1.852 | 2.181 | ≈2.769 | **7.361** |
| Median (↑) | 0.000 | 1.000 | 0.731 | 1.506 | 1.559 | ≈1.906 | **4.690** |
| Mean (↑) | 0.000 | 1.000 | 2.261 | 4.152 | 5.260 | ≈7.700 | **21.574** |
| Optimality Gap (↓) | - | - | 0.407 | 0.200 | 0.224 | ≈0.180 | **0.098** |
| Best | - | - | 0 | 3 | 3 | 2 | **54** |
| >Human | - | - | 22 | 43 | 34 | 38 | **52** |

[BTR (2025/05), Table A1]

抄録の 7.4 は Atari-60 の IQM（human-normalized）であり、表の 7.361 に対応する。

---

## 9. Ablation

### 9.1 数値が与えられているのは Impala のみ

**本文中に数値として与えられているのは Impala の +142% IQM だけである。** 他成分の IQM 変化率は Figure 5 / B3 / C6 のグラフのみで提示され、数値表は存在しない。

```
We find that Impala had the largest effect on performance (+142% IQM), with the other components generally causing a less significant effect. Despite this, simply using Rainbow with Impala does not produce similar results (6.3 IQM compared to 7.7 on Atari-5). Munchausen and IQN have a strong impact on environments requiring fine-grained control such as Phoenix, as explored in Section 5.2.

我々は Impala が性能に最大の影響を与えることを見出した（IQM +142%）。他の構成要素は概してより小さい効果しか及ぼさない。にもかかわらず、単に Rainbow に Impala を組み合わせただけでは同様の結果は得られない（Atari-5 で 7.7 に対して 6.3 IQM）。Munchausen と IQN は Phoenix のような精密な制御を要する環境に強い影響を持つ。これは 5.2 節で検討する。
```

[BTR (2025/05), Section 5.1]

### 9.2 Table 2 (Phoenix) — componentwise の定量比較

200M フレーム学習済み最終エージェント、3 seeds。**原典が与えるコンポーネント別の定量比較はこの表のみである。**

| Category | BTR | w/o Munchausen | w/o IQN | w/o SN | w/o Impala | w/o Maxpool |
|---|---|---|---|---|---|---|
| Action Gap | 0.282 | 0.055 | 0.180 | 0.274 | 0.215 | 0.264 |
| % Action Swaps | 36.6% | 47.7% | 42.2% | 40.3% | 41.1% | 39.3% |
| Policy Churn | 3.8% | 11.0% | 0.5% | 3.3% | 4.5% | 4.2% |
| Score ColorJitter | 212k | 85k | 110k | 187k | 19k | 187k |
| Score ε=0.03 | 94k | 42k | 62k | 75k | 10k | 86k |
| Score ε=0.01 | 194k | 70k | 110k | 132k | 13k | 171k |
| Score ε=0 | 330k | 184k | 187k | 296k | 21k | 406k |

[BTR (2025/05), Table 2 (Section 5.2)]

Policy churn への寄与は本文でも数値で与えられている。

```
Lastly, we find Munchausen and IQN to have a significant impact on Policy Churn (Schaul et al., 2022), with Munchausen reducing it by 6.4% and IQN increasing it by 3.3%.

最後に、Munchausen と IQN が Policy Churn (Schaul et al., 2022) に有意な影響を持つことを見出した。Munchausen はこれを 6.4% 減少させ、IQN は 3.3% 増加させる。
```

[BTR (2025/05), Section 5.2]

### 9.3 vectorization と maxpool は性能を下げる

**この 2 要素は性能を下げることを論文自身が認めたうえで、計算アクセシビリティのために採用している。**

```
For vectorization and maxpooling, while their inclusion reduces performance, we find their secondary effects crucial to keep BTR computationally accessible.

ベクトル化とマックスプーリングについては、それらを含めることで性能は低下するが、その副次的効果が BTR を計算的にアクセス可能に保つ上で決定的であると我々は考える。
```

[BTR (2025/05), Section 5.1]

maxpool の副次的効果はパラメータ削減である。

```
We find that maxpooling decreases the model's parameters by 77%, and makes using wider convolutional layers possible without causing the total number of parameters to increase drastically.

マックスプーリングはモデルのパラメータを 77% 削減し、総パラメータ数を劇的に増やすことなく、より幅の広い畳み込み層の使用を可能にすることが分かった。
```

[BTR (2025/05), Section 5.1, 脚注7]

Table 2 の Score ε=0 も、`w/o Maxpool` の 406k が BTR 本体の 330k を上回っており、この記述と整合する。

### 9.4 投稿後に判明した Layer Normalization

Table D6 の BTR 設定には含まれないが、投稿後に Layer Normalization が有益と判明した旨が脚注に記されている。Appendix H の Table H9 では Layer Normalization を加えた変種が Atari-5 IQM 8.191（BTR 本体 7.739）を示す。

```
After the completion of our work, we additionally found Layer Normalization applied after the stem of each residual block and between dense layers to be beneficial (see Appendix H for a discussion)

我々の研究の完了後、各残差ブロックのステムの後および密結合層の間に適用された Layer Normalization が有益であることをさらに見出した (議論については Appendix H を参照)
```

[BTR (2025/05), Section 3, 脚注2]

---

## 10. 論文とコードの食い違い

| 項目 | 論文 | 公式実装 | 評価 |
|---|---|---|---|
| noop_max | **記載なし**（語自体が皆無） | 30（Gymnasium 既定を継承） | **記載欠落**。論文だけでは再現不能 |
| PER の IS 重み | β の記載なし（α = 0.2 のみ） | `weights = (capacity * probs) ** -alpha`。`per_beta=0.45` は定義されるが IS 重みに使われない。コードコメントが「元は事故」と明記 | **記載欠落 + 実装の意図せぬ挙動** |
| sticky の確率値 | `True` のみ | 0.25 | 記載欠落 |
| frame skip | Table D4 に項目なし | 4 | 記載欠落 |
| max pooling（前処理） | Table D4 に項目なし | 直近 2 フレームの max | 記載欠落 |
| resize の補間 | 記載なし | `cv2.INTER_AREA` | 記載欠落 |
| max episode frames | 108K | `gym.make` に明示指定なし（ALE v5 既定に依存） | 実質一致 |
| 評価対象ネット | 記載なし | online net | 記載欠落 |
| Adam の betas | β1 0.9 / β2 0.999 | 明示指定なし → PyTorch 既定 (0.9, 0.999) | 一致 |
| ε-greedy decay | 8M frames | `--eps_steps` 既定 2,000,000 steps = 8M frames | 一致 |
| ε-greedy disabled | 100M frames | `env_steps < total_frames / 2` = 50M steps = 100M frames | 一致 |
| Layer Normalization | Table D6 に項目なし（脚注で言及） | `--layer_norm` 既定 0 | 一致 |

---

## 11. 批判・懸念

以下は本調査で観察された、原典の記述に起因する再現性・比較可能性の問題点である。

**再現に必要な情報が論文に揃っていない。** noop_max、sticky の確率値、frame skip、max pooling、resize の補間方法、評価対象ネットワーク、PER の β と priority 計算式のいずれも、論文だけでは確定できずコードを読む必要がある。とくに noop_max は語自体が論文に出現しない。

**Machado 準拠の主張と実装が一致しない可能性がある。** 論文は Machado et al. (2018) に従って評価すると述べるが、実装は sticky と noop を併用している。Machado 原典は sticky を no-op の代替として提案しており（§6.3）、併用を推奨してはいない。ただし原典に併用の禁止文言も無いため、これを違反と断定することはできない。

**PER の重要度サンプリングが意図せぬ実装になっている。** IS 重みの指数が β ではなく α であることをコードコメント自身が「事故」と認めており、論文にはこの点の記述が無い。第三者が論文どおり（β を用いて）実装した場合、公式実装と異なる挙動になる。

**報告値と公開実測値が一致しない。** Table A1 の Breakout 682 に対し、公開 `results.csv` の最大は 676.58 である。差の理由は原典に説明がない。

**比較表の一部に出所の記載が無い。** Table A1 の DQN (Nature) / Rainbow 列の出所がキャプションに書かれていない（Figure 1 のみ RLiable 由来と明記）。

**ablation の定量値がほとんど本文に無い。** 数値で与えられるのは Impala の +142% IQM のみで、他成分はグラフのみである。componentwise の表（Table 2）は Phoenix 単一環境・3 seeds に限られる。

**Table A1 の seed 数と集約方法が不明である。** キャプションは `using random seeds` としか述べておらず、seed 数も集約方法も書かれていない。

---

## 12. 総合評価

BTR は Rainbow DQN を基点に 6 要素を追加し、200M フレームの Atari-60 学習をデスクトップ PC（RTX 4090）で 11.5 時間に収めた構成である。設計思想として明確なのは、**性能と計算コストのトレードオフを意識的に性能側へ全振りしていない**点である。replay ratio を 1/64 に抑えたのも、vectorization と maxpool を性能低下と引き換えに採用したのも、いずれも壁時計時間を優先した判断として論文中に明記されている。

ハイパーパラメータの記載密度は Table D6 が 33 行と比較的高いが、**環境側（Table D4、8 行）が薄く、前処理の再現に必要な情報がコードにしか無い**。論文単独での再現は困難で、公式実装が事実上の仕様書になっている。

ablation については、Impala が支配的（+142% IQM）である一方、Munchausen と IQN は Phoenix のような精密制御を要する環境で効くという構造が示されている。Table 2 の Action Gap（BTR 0.282 に対し w/o Munchausen 0.055）と Policy Churn（3.8% に対し 11.0%）は、Munchausen の寄与が「スコア」だけでなく「方策の決然さ・安定性」として測定されていることを示す。

比較値としての Breakout 682 を用いる際は、**(a) best checkpoint であること、(b) online net の評価であること、(c) 100 エピソード平均であること、(d) 公開実測 CSV の最大 676.58・最終 602.01 と一致しないこと**の 4 点を明示しないと、他手法との比較が成立しない。

---

## 13. 調査限界

- 本調査の一次資料取得は 2026-08-26 時点のものである。GitHub 上のソースは main ブランチであり、以降の変更は反映されていない。
- 論文引用は arXiv v2（HTML 版）に基づく。v1（2024/11）との差分は本調査の対象外である。なお v1 のタイトル表記は "On A Desktop PC"、v2 は "on a Desktop PC" である。
- **著者の所属機関は arXiv abs ページ上で確認できなかった。**
- **PER の β の初期値・終値・スケジュール長は原典で確認できなかった**（論文中で "beta" は Adam の β1/β2 としてのみ登場する）。
- **IQN の tau 数の current / target / policy 別の値（N, N′, K の区別）は原典で確認できなかった。** 原典が与えるのは単一値 8 のみである。
- **PER の priority 計算式は原典で確認できなかった。** 論文は Schaul et al. (2015) の一般的説明を引用するのみである。
- **Impala 以外の個別コンポーネントの IQM 変化率は数値として確認できなかった**（Figure 5 / B3 / C6 のグラフのみ）。
- **Table A1 の DQN (Nature) / Rainbow 列の出所は確認できなかった。**
- **論文値 682 と公開 CSV 実測 676.58 の差の理由は確認できなかった。**
- **BTR / MEME / DreamerV3 の 3 件とも、論文本文に no-op starts の設定は確認できなかった**（BTR はコードで 30 と確定）。
- ALE 公式 FAQ には sticky actions / noop / v5 バージョニング / 推奨評価プロトコルに関する記述が無かった。
- Machado et al. 2018 に「sticky を使う際に no-op を併用してはならない」旨の明示的な禁止文言は確認できなかった。
- 本報告の引用は 3 つの独立した調査経路で収集された。Table D4 の内容、評価 epsilon・エピソード数・評価間隔、noop_max = 30 の 3 点は複数経路で一致を確認した。それ以外の項目は単一経路の取得に基づく。

---

## 14. 付録A: anet-lab 現行設定との差分（調査対象外の自プロジェクト情報）

**本節は BTR 原典の記述ではない。** 左列は §3〜§7 で確定した BTR の値、右列は anet-lab の Run `run_20260825-183923_atari_breakout_apex_e04_100m` の実行時 config ダンプ（`config/config_data.txt` および `config/DefaultDQNAgent.txt`）から取得した実測値である。

### A.1 一致している項目

γ 0.997 / batch size 256 / n-step 3 / replay capacity 1,048,576 / weight decay 0 / Adam lr 1e-4 / Huber κ 1.0 / Impala 幅スケール 2 / dueling 採用・512 per stream / IQN cos 埋め込み 64 / sticky 0.25 / episodic life false / max episode frames 108,000 / grayscale・84x84・frame stack 4 / reward clip [-1,1] / replay ratio（BTR の 1/64 は batch 256 ÷ 64 env steps = 4 samples/env-step であり、anet-lab の `replay_ratio = 4` と同義）。

### A.2 差分がある項目

| 項目 | BTR | anet-lab 現行 |
|---|---|---|
| **Munchausen RL** | α 0.9 / τ 0.03 / l0 −1.0 | **未実装** |
| **Spectral Normalization** | residual 内 conv 12 層 | **未実装** |
| **Noisy Networks** | σ = 0.5（有益と確認して維持） | **未実装** |
| **Adaptive Maxpool** | 6×6（性能は下がるがパラメータ 77% 減） | **無し**（Flatten 11×11×64 = 7,744） |
| **Double DQN** | 不採用（Munchausen と併用で不要） | **採用**（`use_double_dqn = true`） |
| **Target network** | hard update / 500 grad steps | **soft update**（`soft_update_tau = 0.001`） |
| 勾配クリップ | max norm 10 | `grad_clip_tau = 30.0` |
| Adam epsilon | 1.95e-5（= 0.005 / batch） | 1e-4 |
| min replay | 200,000 transitions | `update_warmup_steps = 20,000` |
| PER α | 0.2 | 0.5 |
| PER IS 指数 | α = 0.2 固定（アニールなし） | β 0.4 → 1.0 を 50M でアニール |
| IQN taus | 8（online / target / policy 共通） | 32 / 32 / 32 |
| 探索 | NoisyNet + ε 1.0→0.01（8M frames）、100M frames で ε 無効化 | spatial ε ラダー 0.01–0.4（減衰なし・固定） |
| 並列環境数 | 64 | 128 |
| **noop_max** | **30** | **0**（`AtariEnv.@v5_noop0`） |
| fire reset | 実装なし | false |
| 評価 ε | 0.01（125M frames まで）→ 以降 0 | 0.01 固定 |
| 評価エピソード数 | 100 | 1（`eval_batch_size = 1`） |
| 評価対象ネット | online net | target net（`eval1`） |
| 報告値の性質 | best checkpoint | 窓プール平均 |
| seed 数 | Atari-60 で 4 | 1 |
| 総勾配ステップ | 781,000（200M frames） | 1,562,500（100M steps = 400M frames） |
| Linear 入力次元 | 6×6×64 = 2,304 → 512（1,179,648 params） | 7,744 → 512（3,965,440 params） |
| 総パラメータ数 | 2.91M（うち linear 2.52M） | 未計測（Linear 単体で BTR 総数を超える） |

---

## 15. 出典

**原論文**

[BTR, 2025/05] Tyler Clark, Mark Towers, Christine Evers, Jonathon Hare. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." arXiv:2411.03820v2 (ICML 2025, Poster). https://arxiv.org/abs/2411.03820

[Machado+, 2017/12] Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." arXiv:1709.06009 (v2; JAIR 2018). https://arxiv.org/abs/1709.06009

[MEME, 2022/09] Steven Kapturowski et al. "Human-level Atari 200x faster." arXiv:2209.07550. https://arxiv.org/abs/2209.07550

[DreamerV3, 2023/01] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. "Mastering Diverse Domains through World Models." arXiv:2301.04104. https://arxiv.org/abs/2301.04104

**公式リポジトリのソースコード**

[BTR-code, 2026/08] Tyler Clark et al. "VIPTankz/BTR." GitHub (main). 参照ファイル: `AtariPreprocessingCustom.py`, `main.py`, `Agent.py`, `PER.py`, `networks.py`, `README.md`, `results.csv`. https://github.com/VIPTankz/BTR

[ale-py-src, 2026/08] Farama Foundation. "registration.py / env.py." Arcade-Learning-Environment (master). https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/python/registration.py

[Gymnasium-src, 2026/08] Farama Foundation. "atari_preprocessing.py." Gymnasium (main). https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/atari_preprocessing.py

[Dopamine-src, 2026/08] Google. "atari_lib.py." Dopamine (master). https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py

[SB3-src, 2026/08] Antonin Raffin et al. "atari_wrappers.py." Stable-Baselines3 (master). https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

[CleanRL-dqn-src, 2026/08] Shengyi Huang et al. "dqn_atari.py." CleanRL (master). https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py

[CleanRL-envpool-src, 2026/08] Shengyi Huang et al. "ppo_atari_envpool.py." CleanRL (master). https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool.py

[rlpyt-src, 2026/08] Adam Stooke. "atari_env.py." rlpyt (master). https://github.com/astooke/rlpyt/blob/master/rlpyt/envs/atari/atari_env.py

[SPR-src, 2026/08] Max Schwarzer et al. "rlpyt_atari_env.py." SPR (release). https://github.com/mila-iqia/spr/blob/release/src/rlpyt_atari_env.py

[dqn_zoo-src, 2026/08] DeepMind. "gym_atari.py / dqn/run_atari.py." dqn_zoo (master). https://github.com/google-deepmind/dqn_zoo/blob/master/dqn_zoo/gym_atari.py

**公式ドキュメント**

[ALE-docs, 2026/08] Farama Foundation. "Environments — Version History and Naming Schemes." Arcade Learning Environment Documentation. https://ale.farama.org/environments/

[ALE-docs-Pong, 2026/08] Farama Foundation. "Pong — Variants / Version History." Arcade Learning Environment Documentation. https://ale.farama.org/environments/pong/

[Gymnasium-docs, 2026/08] Farama Foundation. "AtariPreprocessing — Misc Wrappers." Gymnasium Documentation. https://gymnasium.farama.org/api/wrappers/misc_wrappers/

[EnvPool-docs, 2026/08] Jiayi Weng et al. "Atari." EnvPool Documentation. https://github.com/sail-sg/envpool/blob/main/docs/env/atari.rst

**その他**

[Semantic Scholar, 2026/08] Semantic Scholar. "Beyond The Rainbow: High Performance Deep Reinforcement Learning On A Desktop PC." Semantic Scholar Academic Graph API. https://api.semanticscholar.org/graph/v1/paper/arXiv:2411.03820
