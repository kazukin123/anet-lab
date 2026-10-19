# Code Reading: BTR 公開実装 — PER・ネットワーク結線・評価系と anet-lab 現行差分

Date: 2026-08-31
Scope: `C:\dev\BTR`（`VIPTankz/BTR` main、commit 093ca77）のソースと同梱 `results.csv` を直接読み、[btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md)（論文由来）で確定済みの内容を再確認したうえで、**論文には無くコードにしか無い挙動**を確定させる。あわせて anet-lab 現行 config（Run `run_20260831-050214_i_rr4_ln512_breakout` の実行時ダンプ）との差分表を更新する。

> **位置づけ**: 2026-08-26 の survey は arXiv:2411.03820 の本文・付録・表を一次ソースとした。本書はその**実装側からの裏取りと増分**であり、論文の記述と矛盾する点は見つかっていない。survey で既に確定していた項目は §2 に列挙するだけに留め、§3 以降に新規分を書く。

## 目次

1. [調査方法](#1-調査方法)
2. [既存 survey の再確認（新規性なし）](#2-既存-survey-の再確認新規性なし)
3. [新規に判明した点](#3-新規に判明した点)
   - PER 初期優先度 / β の死にコード化 / 正味の優先度指数 / replay ratio の実装的意味 / ネットワーク結線 / LayerNorm の実装位置 / 評価の実装詳細 / 学習曲線の相転移点
4. [anet-lab 現行 config との差分（2026-08-31 版）](#4-anet-lab-現行-config-との差分2026-08-31-版)
5. [config だけで試せる未検証項目](#5-config-だけで試せる未検証項目)
6. [再現実験の実行可能性](#6-再現実験の実行可能性)
7. [調査の限界](#7-調査の限界)

---

## 1. 調査方法

`C:\dev\BTR` のローカル clone を直接読んだ。参照ファイルは `main.py`(420行) / `Agent.py`(1070行) / `PER.py`(483行) / `networks.py`(1021行) / `README.md` / `requirements.txt` / `results.csv`。引用は `ファイル:行` で示す。

`results.csv` は 60 ゲーム × 200 列（1M フレームごと、各回 100 エピソード平均）の実測値であり、Python で直接集計した。

---

## 2. 既存 survey の再確認（新規性なし）

以下はすべて 2026-08-26 survey で論文から確定済みで、コード側でも一致を確認した。**再掲のみで、新しい情報は無い。**

| 項目 | 値 | コード上の根拠 |
|---|---|---|
| discount | 0.997 | `main.py:151` |
| n-step | 3 | `main.py:134` |
| batch size | 256 | `main.py:121` |
| replay capacity | 1,048,576 | `Agent.py:111` |
| min replay | 200,000 | `Agent.py:181` |
| lr | 1e-4 | `main.py:137` |
| Adam eps | `0.005 / batch_size` = 1.953e-5 | `Agent.py:350` |
| grad clip | 10 | `main.py:142` |
| IQN taus / cos 基底 | 8 / 64 | `main.py:153,165` |
| PER α | 0.2 | `main.py:167` |
| PER eps | 1e-6 | `PER.py:119` |
| target 更新 | hard C=500 grad steps（EMA は既定 off） | `main.py:154,164` |
| Double DQN | **不採用** | `main.py:160`, `Agent.py:106` |
| Munchausen | 採用（α 0.9 / τ_ent 0.03） | `main.py:140-141`, `Agent.py:255-257` |
| NoisyNet | 採用（σ0 = 0.5） | `main.py:144`, `networks.py:59` |
| Spectral Norm | residual 内 conv のみ | `networks.py:337-343` の `norm_func` 適用範囲 |
| Adaptive Maxpool | 6×6 | `main.py:136,148`, `networks.py:772` |
| Impala 幅倍率 | 2（32/64/64 ch） | `main.py:157`, `networks.py:758-768` |
| optimizer | 素の Adam（AdamW は既定 off） | `main.py:161`, `Agent.py:350` |
| sticky | 0.25 | `main.py:131,203` |
| episodic life | **不使用**（`--life_info` 既定 0） | `main.py:127` |
| 評価 | 100 エピソード / 1M フレームごと / online net | `main.py:128,299,398` |
| 報告値 682 | Table A1 の学習中最大スコア。CSV とは不一致 | survey §8.1 |
| maxpool の効果 | 性能は下がるがパラメータを大きく削る | survey §11（Appendix H） |

**2026-08-26 survey §14 の差分表にあった「BTR の 1/64 は batch 256 ÷ 64 env steps = 4 samples/env-step であり、anet-lab の `replay_ratio = 4` と同義」も、コード側で確認した（§3.4）。**

---

## 3. 新規に判明した点

### 3.1 PER 初期優先度は「単調非減少の過去最高水位」

新規遷移は `max_prio` を初期優先度として和木へ積まれる。

```python
# PER.py:113
self.max_prio = 1
# PER.py:196, 226  (append_pointer / finalize_experiences)
self.st.append(self.max_prio ** self.alpha)
# PER.py:385  (update_priorities)
self.max_prio = max(self.max_prio, np.max(priorities))
```

`max_prio` は **`max()` でしか更新されず、減衰も再計算もされない**。したがってこれは「現在バッファ内にある最大優先度」ではなく、**学習開始以降に一度でも観測された TD 誤差の最大値**である。一度大きな値が出れば、以後すべての新規遷移が永久にその水位で入る。

論文にも survey にも初期優先度の記述は無い。実質的には「文献の max-priority 初期化」というより **early に決まる準固定値**に近い挙動になる。

### 3.2 `beta` は死にコード

`PER.__init__` は `beta=0.4` を受け取るが（`PER.py:86,118`）、`append_pointer` の while ループ末尾で毎回 0 に潰される。

```python
# PER.py:204  (append_pointer の while ループ内)
self.beta = 0
```

そして重要度サンプリング重みの計算式は `beta` を参照していない。

```python
# PER.py:324
weights = (self.capacity * probs) ** -self.alpha  # self.beta originally this was an accident but actually performed better
```

**著者コメントが「元は手違いだが、こちらの方が性能が良かった」と明記している。** survey §3.6 はこの式を引用済みだが、`self.beta = 0` の存在と、`beta` が完全に到達不能である事実は本書が新規に確認した。

`--per_beta_anneal` 既定 0（`main.py:168`）も含め、**BTR に β という自由度は存在しない**。

### 3.3 正味の優先度指数と anet-lab への含意

PER の期待勾配は「サンプリング確率 × IS 重み」で決まる。サンプリングは `P_i ∝ p_i^α`、IS 重みは `w_i ∝ (N·P_i)^{-β}` なので、正味の重みは

```
P_i · w_i  ∝  p_i^α · p_i^(-αβ)  =  p_i^(α(1-β))
```

| | α | IS 指数 β | 正味 |
|---|---|---|---|
| BTR | 0.2 | **0.2 固定** | **p^0.16** |
| anet-lab 現行 | 0.2 | 0.4 → **1.0**（25M exp step で到達） | **p^0** |

**β = 1 は PER の勾配バイアスを完全に打ち消す。** これは canonical PER の設計意図（完全な不偏化）そのものだが、結果として **anet-lab は 25M step 以降、期待勾配の上では一様サンプリングと等価**になっている。BTR は正味 `p^0.16` で優先度が生きたままである。

これは `per_beta_end` 1 キーで追試できる。**未検証。**

### 3.4 `--rr` の実装的意味

`--rr` は「環境ステップあたりの勾配更新数」ではなく、**`replay_period`（= 並列環境数）ステップあたりの勾配ステップ数**である。

```python
# Agent.py:194  (コメント)
# in this code, every {replay period} steps, we take {replay_ratio} grad steps
# main.py:318
replay_period=num_envs        # num_envs = 64 (main.py:120)
# main.py:342-345
steps += num_envs
agent.learn()                 # rr=1 → 1 grad step
```

したがって既定は **64 環境ステップごとに 1 勾配ステップ、batch 256**。1 データあたりのサンプル回数は 256/64 = **4** で、勾配ステップ数は `exp/64`。

anet-lab の `replay_ratio = 4`, `replay_batch_size = 256` は `exp × 4/256 = exp/64` で**完全に一致する**。

**すなわち BTR の比較相手は anet-lab の RR4 であり、RR1 は BTR の 1/4 の勾配量にあたる。**

この帰結として、2026-08-31 の探索で観測した「RR4 では `07_evicted_unsampled_ratio` が 0.0001 になり PER の除外機能が失われる」現象は **BTR でも同様に起きているはず**であり、除外の有無は BTR との成績差の主因ではない。

### 3.5 ネットワーク結線 — 512 共有トランクが無い

`ImpalaCNNLargeIQN`（`networks.py:703-816`）の実際の結線は次のとおり。

```
conv:  ImpalaBlock(4→32) → ImpalaBlock(32→64) → ImpalaBlock(64→64) → ReLU
       各 block = Conv3x3 → MaxPool(3,2,pad1) → Res → Res
       Res = pre-activation（conv_0(act(x)) → conv_1(act(·)) → x + ·）、SN は Res 内 conv のみ
pool:  AdaptiveMaxPool2d(6,6)          →  conv_out_size = 1152 × model_size = 2304
cos:   Linear(64 → 2304)               ←  IQN の cos 埋め込みは 2304 次元
dueling（V と A は完全に独立）:
   value: NoisyLinear(2304→512) → ReLU → NoisyLinear(512→1)
   adv:   NoisyLinear(2304→512) → ReLU → NoisyLinear(512→actions)
```

`networks.py:786` が `self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)`、`networks.py:789-798` が dueling 定義。

anet-lab 現行（`Atari.txt` の `@AtariImpalaX2` + `net.@iqn`）:

```
conv:  同一の 32/64/64 Impala → ReLU
Flatten (11×11×64 = 7744)
AtariLinear512 → ReLU                  ←  BTR に存在しない共有トランク
tau_embedding: Linear(64→512) → SiLU   ←  IQN 融合は 512 次元
V/A: Linear(512→512) → SiLU → out      ←  共有トランクの出力を分岐
```

**構造差は 3 点。**

1. **BTR には V/A 共通の中間 Linear が無い。** 各ストリームが conv 出力から独立に 512 次元へ落とす。anet-lab は 512 の共有トランクを経てから分岐する。
2. **IQN の cos 埋め込みとの要素積が起きる次元が 2304 対 512 で 4.5 倍違う。** BTR は pooling 直後の広い表現に τ を乗せ、anet-lab は 512 に圧縮した後に乗せる。
3. head 活性が ReLU 対 SiLU。

survey §14 の差分表は「Linear 入力次元 2304→512 対 7744→512」までは記載していたが、**IQN 融合位置と共有トランクの有無**は本書が新規に確認した。

### 3.6 LayerNorm は実装済みで、位置は anet-lab の LN512 と同一

`--layer_norm` 既定 0（`main.py:169`）だが、実装は存在する。

```python
# networks.py:800-807  (dueling, layer_norm=True 時)
nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
              nn.LayerNorm(self.linear_size),
              activation(),
              linear_layer(self.linear_size, 1))
# networks.py:359-360, 369-370  (ImpalaCNNBlock, 入口 conv の直後・maxpool の前)
self.norm_layer1 = nn.LayerNorm(layer_norm_shapes[0])
```

**「512 次元へ落とす Linear の直後、活性化の直前」は anet-lab の `AtariLN512` と同じ位置である。** survey §11 が引いた Appendix H の「研究完了後に、各残差ブロックのステムの後および密結合層の間に適用された Layer Normalization が有益であるとさらに見出した」に、コード上の実体が対応している。

2026-08-31 の探索で LN512 が有効だった（RR1 で `02_dead_ratio` 0.197 → 0.005、壁突破率 2.8 倍）ことは、この Appendix H の記述と独立に一致した。

### 3.7 評価の実装詳細

```python
# main.py:397-398
agent.disable_noise(agent.net)
net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
# Agent.py:52-62
qvals = eval_net.qvals(state, advantages_only=True)
x = T.argmax(qvals, dim=1).cpu()
if rng > 0.:
    x = randomise_action_batch(x, 0.01, n_actions)
```

- 評価対象は **online net**（`agent.net`）で、**NoisyNet のノイズを無効化**してから複製する。
- 行動選択は `advantages_only=True` の argmax。dueling の V 項を計算しない最適化で、argmax には影響しない。
- ε は `rng > 0` のときだけ 1%。
- 評価は 10 並列環境（`main.py:126`）で 100 エピソード（`main.py:128`）を集める。
- 間隔は 250,000 環境ステップ（`main.py:299`）= 1M フレーム。

anet-lab の `eval1` は **target net を 1 エピソード**であり、統計量としても対象ネットワークとしても別物である。

### 3.8 Breakout 学習曲線の相転移点

`results.csv` の Breakout 行（200 列、各列 100 エピソード平均）を集計した。

| 統計 | 値 |
|---|---|
| 最大 | **676.58**（第 161 列 = 161M フレーム） |
| 最終（200M フレーム） | 602.01 |
| 末尾 10 列平均 | 605.54 |

survey §8.1 はこの 3 値を既に記録している。**本書の新規分は曲線の形状である。**

```
 11M- 20M frames:  295.9  369.1  398.7  411.3  409.6  394.9  386.1  418.1  370.1  478.2
 21M- 30M frames:  497.7  453.0  450.8  492.3  436.6  459.2  381.2  500.9  550.8  430.4
   （中略：31M-120M フレームは 400〜580 の帯を横ばい）
121M-130M frames:  528.7  545.2  531.7  569.8  544.8  522.9  567.3  657.8  612.6  566.6
131M-140M frames:  670.4  589.0  587.2  566.3  556.4  662.2  553.6  563.2  575.8  593.4
```

**20M フレーム（5M エージェントステップ）で既に 478 に到達し、そこから 128M フレームまで約 100M フレーム（25M ステップ）を 450〜550 の帯で横ばいに過ごす。128M フレーム（32M ステップ）で 657.8 へ一段上がり、以降 600 前後で安定する。**

Breakout の 1 画面は 18 列 ×(1+1+4+4+7+7) = **432 点**である。長い踊り場は 432 をわずかに上回る水準にあり、**「全消しが時々起きる混合状態」が 25M ステップ持続していると読める**。段差は 2 画面目へ安定して進めるようになった時点に対応する。

エージェントステップ換算での対応点（BTR 列 N = N M フレーム = N/4 M ステップ）:

| agent steps | BTR |
|---|---|
| 5M | 478.2 |
| 12.5M | 474.8 |
| 25M | 499.1 |
| 50M | 602.0 |

---

## 4. anet-lab 現行 config との差分（2026-08-31 版）

survey §14 の差分表は Run `run_20260825-183923`（旧バンドル）を右列に取っていた。その後のベースライン修正で複数項目が解消しているため、Run `run_20260831-050214_i_rr4_ln512_breakout` の実行時ダンプで更新する。

### 4.1 解消済み（survey §14 時点では差分だった）

| 項目 | BTR | anet-lab 現行 |
|---|---|---|
| Adam epsilon | 1.953e-5 | **1.95e-5** |
| min replay | 200,000 | **200,000** |
| PER α | 0.2 | **0.2** |
| 勾配クリップ | 10 | **10** |
| IQN taus | 8 | **8 / 8 / 8** |
| noop_max | 30 | **30** |
| replay ratio | exp/64 | **RR4 = exp/64** |

### 4.2 残っている差分

| 項目 | BTR | anet-lab 現行 | 種別 |
|---|---|---|---|
| **PER IS 指数** | α = 0.2 固定 | β 0.4 → 1.0（25M step でアニール） | **config 1 キー** |
| **Double DQN** | 不採用 | `use_double_dqn = true` | **config 1 キー** |
| **target 更新** | hard C=500 grad steps | `soft_update_tau = 0.001` | **config**（腕あり） |
| **Adaptive Maxpool** | 6×6（2304 次元） | 無し（Flatten 7744） | **config**（構造） |
| **探索 ε** | NoisyNet + ε 減衰後に無効化 | spatial ε ラダー（`use_spatial_exploration = true`、`eps_start 0.4 → eps_end 0.01` / 250k step） | config |
| **並列環境数** | 64 | `train.num_envs = 128` | config |
| **512 共有トランク** | 無し | あり（`AtariLinear512`） | 構造 |
| **IQN 融合次元** | 2304 | 512 | 構造 |
| **head 活性** | ReLU | SiLU | config |
| **Munchausen** | 採用（α 0.9 / τ_ent 0.03 / l0 −1.0） | 未実装 | **実装** |
| **NoisyNet** | 採用（σ0 0.5） | 未実装 | **実装** |
| **Spectral Norm** | residual conv のみ | PRD 065 で実装済み・既定 off | config |
| 評価エピソード数 | 100 | 1 | 実装 |
| 評価対象ネット | online net | target net | 実装 |
| 報告値の性質 | 学習中最大（200 点の max） | 窓プール平均 | 集計 |
| seed 数 | Atari-60 で 4 | 1 | 運用 |

---

## 5. config だけで試せる未検証項目

§4.2 のうち、実装を伴わずに測れるものを影響の大きい順に置く。**いずれも 2026-08-31 時点で未検証。**

| # | 変更 | 根拠 |
|---|---|---|
| 1 | `per_beta_end` を 1.0 → 0.2 | §3.3。現行は 25M step 以降 PER の勾配バイアスが完全に打ち消されており、期待勾配は一様サンプリングと等価になっている |
| 2 | `use_double_dqn` を false | §2。BTR は Munchausen と併用で不要として不採用。anet-lab は Munchausen 不在だが、IQN との併用効果は未測定 |
| 3 | 6×6 AdaptiveMaxPool を Flatten 前に挿入 | §3.5。Linear 入力が 7744 → 2304、Linear パラメータは 3.97M → 1.18M。BTR 自身は「性能は下がるが計算量が決定的に下がる」と評価している |
| 4 | head 活性を SiLU → ReLU | §3.5。trunk が ReLU で head だけ SiLU という現行の不整合の解消 |
| 5 | `train.num_envs` を 128 → 64 | §3.4。BTR と揃えると 1 勾配ステップあたりの環境ステップ数が一致する |

構造側（512 共有トランクの除去、IQN 融合を 2304 次元へ）は config の `structure` 記述で表現できるが、変更が大きいため単独の検討を要する。

---

## 6. 再現実験の実行可能性

| 項目 | 状況 |
|---|---|
| Python | BTR は 3.11.0 指定（README）。ローカルは **3.14.5** |
| PyTorch | BTR は 2.1.2+cu121。ローカルは **2.13.0+cpu、`torch.cuda.is_available()` = False** |
| ale-py / gymnasium | **未導入**（`import ale_py` が失敗） |
| ROM | AutoROM によるライセンス同意つき取得が必要 |
| 学習済みモデル | `final_models/` は Atari-5 の 5 本のみ（BattleZone / DoubleDunk / NameThisGame / Phoenix / Qbert）。**Breakout は無い** |
| メモリ | 訓練 64 並列 + 評価 10 並列。README が Windows で RAM を消費すると明記 |

**別 venv（Python 3.11 + torch cu121）の構築が前提**で、GPU も占有する。`results.csv` が 60 ゲーム × 200 点の実測を含むため、単純な再現の情報価値は低い。価値があるのは Breakout 単独の ablation（`--munch 0` との対など、論文の ablation は Atari-5 集計のみ）である。

---

## 7. 調査の限界

- 本書はソースの静的読解であり、**BTR を実行して挙動を確認してはいない**。
- `results.csv` の Breakout 行は 1 系列であり、seed 分散が分からない。論文は Atari-60 で 4 seed としているが、CSV がその平均か単一 seed かは記載が無い。
- 論文値 682 と CSV 最大 676.58 の差の理由は、survey §8.1 の時点と同じく不明のままである。
- `Agent.py` の Munchausen 損失本体、`AtariPreprocessingCustom.py` の前処理詳細、`SumTree` の実装は本書では読んでいない。
- §5 の「config だけで試せる」は anet-lab 側のキーの存在を確認したものであり、**期待される効果の方向は測定されていない**。

---

## 参照

**ソース**（`C:\dev\BTR`、`VIPTankz/BTR` main、commit 093ca77）

`main.py` / `Agent.py` / `PER.py` / `networks.py` / `README.md` / `requirements.txt` / `results.csv`

**関連文書**

- [btr_hyperparams_survey_2026-08-26.md](btr_hyperparams_survey_2026-08-26.md) — 論文一次ソースからのハイパーパラメータと評価プロトコル確定。本書はその増分にあたる。
- [atari_env_survey_2026-08-13.md](atari_env_survey_2026-08-13.md) — ALE と v5 プロトコル。
- `apps/runner/workspaces/atari-2nd/runs/run_20260831-050214_i_rr4_ln512_breakout/config/config_data.txt` — §4 の右列の出所。
