# Action Query 関連論文サーベイ

調査日: 2026-08-13

## 目次

- [結論](#結論)
- [論文一覧](#論文一覧)
- [1. Action Q-Transformer](#1-action-q-transformer)
- [2. Query2Label](#2-query2label)
- [3. Perceiver IO](#3-perceiver-io)
- [4. Set Transformer](#4-set-transformer)
- [5. DETR](#5-detr)
- [6. ACT](#6-act)
- [7. 近縁だが別方式の論文](#7-近縁だが別方式の論文)
- [DropMerge への対応](#dropmerge-への対応)

## 結論

今回話していた ActionQuery と最も直接的に一致する先行研究は **Action Q-Transformer (AQT)** である。AQT は、離散 action ごとの one-hot を共有 FC に通して action query を作り、Transformer encoder の空間特徴を key/value として decoder cross-attention を行う。さらに Dueling 構成と Rainbow の categorical distributional RL（51 atoms）まで採用している。

したがって、次の骨格は既知の手法である。

```text
state spatial tokens [B,N,D]
action queries       [A,D]
        ↓ Transformer decoder cross-attention
action features      [B,A,D]
        ↓ shared distributional head
action distributions [B,A,51]
```

一方、AQT は C51 系であり、`action feature × tau feature → quantile value` という IQN 結合は扱わない。DropMerge で検討している `[B,59,D] × [B,K,D] → [B,59,K]` は、AQT の action-query decoder を IQN へ拡張する設計になる。

## 論文一覧

引用数は OpenAlex 等の索引を 2026-08-13 に取得した参考値である。プレプリントと出版版の重複、索引差があるため影響度の概算にのみ使う。

| 優先 | 論文 | 著者・主所属 | 発表 | 引用数参考 | 今回との関係 |
|---:|---|---|---|---:|---|
| 1 | [Action Q-Transformer](https://arxiv.org/abs/2306.13879) | Hidenori Itaya et al.; 中部大学、慶應義塾大学 | arXiv, 2023 | 0 | 離散 action query、Dueling、51 atoms。最も直接的 |
| 2 | [Query2Label](https://arxiv.org/abs/2107.10834) | Shilong Liu et al.; 清華大学、IDEA | arXiv, 2021 | 120 | 各出力意味に固定された query と adaptive pooling |
| 3 | [Perceiver IO](https://arxiv.org/abs/2107.14795) | Andrew Jaegle et al.; DeepMind | ICLR, 2022 | 205 | 出力座標・種類を query に符号化する一般論 |
| 4 | [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | Juho Lee et al.; Oxford、POSTECH 等 | ICML, 2019 | 151 / 411 | 1 個の pooling から複数 learned seed への一般化 |
| 5 | [DETR](https://arxiv.org/abs/2005.12872) | Nicolas Carion et al.; Facebook AI、Paris Dauphine | ECCV, 2020 | 831 | learned decoder query の代表的な祖先 |
| 6 | [ACT](https://arxiv.org/abs/2304.13705) | Tony Z. Zhao et al.; Stanford、UC Berkeley、Meta | RSS, 2023 | 約218 | action sequence の各時刻を query として並列生成 |

## 1. Action Q-Transformer

**Hidenori Itaya, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Komei Sugiura. “Action Q-Transformer: Visual Explanation in Deep Reinforcement Learning with Encoder-Decoder Model using Action Query.” arXiv:2306.13879, 2023.**

AQT は Transformer encoder-decoder を Q-learning に導入し、encoder で state value、decoder で action ごとの advantage を計算する。decoder の query は action one-hot を FC に通した learned action embedding であり、query 数は action 数と等しい。

```text
“The number of queries is the same as the number of actions”

「クエリ数は行動数と同じである」
```

[Itaya et al. (2023/06), Section 3.3]

構成は次のようになる。

```text
CNN feature map
  → spatial tokens
  → Transformer Encoder
        ├─ flatten → Value branch → V(s, atoms)
        └─ Action queries → Transformer Decoder
                             → shared Advantage head
                             → A(s, action, atoms)
  → Dueling combine → Q(s, action, atoms)
```

実験では Rainbow を土台に 51 atoms を用いる。decoder 出力に同じ小さな advantage head を各 action へ適用するため、巨大な `flattened_state → actions × atoms` 全結合とは異なる。ただし value branch は encoder 全体を flatten しており、AQT 自体も flatten を完全には排除していない。また Atari で AQT 単体が常に baseline を上回ったわけではなく、ゲームによっては改善が確認されなかった。

DropMerge へそのまま移せない点は、AQT が `hidden_dim = action_count × 32` としていること、C51 であって QR/IQN ではないこと、提案目的が主として attention の可視化であることである。

## 2. Query2Label

**Shilong Liu, Lei Zhang, Xiao Yang, Hang Su, Jun Zhu. “Query2Label: A Simple Transformer Way to Multi-Label Classification.” arXiv:2107.10834, 2021.**

Query2Label は各 label class に一つの query を固定対応させ、画像 feature map を key/value として label 固有の特徴を adaptive pooling する。

```text
“each query (or label embedding) uniquely corresponds to one label class”

「各 query（label embedding）は一つの label class に一意に対応する」
```

[Liu et al. (2021/07), Section 1]

```text
spatial features [B,N,D]
label queries    [C,D]
    ↓ decoder cross-attention
label features   [B,C,D]
    ↓ classifier
logits           [B,C]
```

DETR の object query が交換可能な prediction slot であるのに対し、Query2Label の query は特定 label という固定意味を持つ。59 個の action ID が固定される DropMerge にはこの意味論が近い。また同論文は、GAP が空間 feature map の情報を一律に平均して小さな識別特徴を希釈し得ることを動機に挙げる。これは DropMerge の GAP 劣化を直接実証するものではないが、「action ごとに必要な場所を選んで読む」設計の根拠になる。

## 3. Perceiver IO

**Andrew Jaegle et al. “Perceiver IO: A General Architecture for Structured Inputs & Outputs.” ICLR 2022.**

Perceiver IO は、望む出力要素ごとに query を一つ作り、query 自身にその出力の意味を持たせる一般形を提示する。

```text
“We define a query array with the same number of elements as the desired output.”

「望む出力と同じ要素数を持つ query array を定義する」
```

[Jaegle et al. (2022/03), Section 3]

query は learned embedding、手設計特徴、入力から作る特徴のいずれでもよい。位置を持つ出力には位置 encoding、複数 task/modalities には type embedding を含められる。

```text
input/state array [M,C]
  → encode/process
latent array      [N,D]
  → output queries [O,QD] との cross-attention
outputs           [O,E]
```

DropMerge における次の query 設計は論文の直接主張ではなく、この原則からの設計推論である。

```text
q_drop(x) = drop_type_embedding + position_encoding(x / 57)
q_noop    = noop_type_embedding
```

## 4. Set Transformer

**Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, Yee Whye Teh. “Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.” ICML 2019.**

Pooling by Multihead Attention（PMA）は、`k` 個の learned seed vectors から入力集合へ attention し、`k` 個の出力を作る。

```text
PMA_k(Z) = MAB(S, rFF(Z)),  S ∈ R^{k×d}
```

```text
“the output of PMA_k is a set of k items”

「PMA_k の出力は k 個の要素からなる集合である」
```

[Lee et al. (2019/06), Section 3.2]

これは単一 CLS/GAP を `k` 個の query-dependent summaries へ拡張する原型である。ただし seed は特定 action へ固定対応するとは限らない。action の固定意味には Query2Label / Perceiver IO の考えを加える必要がある。

## 5. DETR

**Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko. “End-to-End Object Detection with Transformers.” ECCV 2020.**

DETR は learned object queries が画像 encoder memory を cross-attention で読み、複数 object predictions を並列生成する query decoder の代表例である。

```text
image tokens   [B,HW,D]
object queries [N,D]
    ↓ Transformer decoder
object features [B,N,D]
    ↓ shared FFN
class + box predictions
```

ただし各 query は action ID のような固定意味ではなく、Hungarian matching で対象へ割り当てられる交換可能な slot である。そのため構造の祖先ではあるが、DropMerge の意味論には AQT / Query2Label の方が直接的である。

## 6. ACT

**Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn. “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.” RSS 2023.**

ACT の policy decoder は、画像・関節・style の encoder 出力を key/value とし、未来 `k` 時刻を表す固定 sinusoidal embeddings を query として `[k,512]` を生成し、MLP で `[k,14]` action sequence へ射影する。

```text
“The ‘queries’ are fixed sinusoidal embeddings for the first layer.”

「第1層の query は固定正弦波 embedding である」
```

[Zhao et al. (2023/04), Appendix C]

query が「action 候補」ではなく「未来 action の時刻」を表す点、模倣学習で連続 action を予測する点は異なる。しかし、状態 token から複数の意味付き action outputs を小さな shared head で並列生成する実装例として参考になる。

## 7. 近縁だが別方式の論文

- [MUTEX](https://proceedings.mlr.press/v229/shah23b.html)（CoRL 2023）は learned action-type query と action-index position を Perceiver-style decoder に入れ、未来の連続 action 列を生成する。query の構成法は近いが、59 個の排他的候補の Q 値ではない。
- [Perceiver-Actor / PerAct](https://proceedings.mlr.press/v205/shridhar23a.html)（CoRL 2022）は voxel 位置ごとの translation Q map を予測する。明示的 query bank ではないが、空間 action 構造を一つの global vector に潰さない先例である。
- [Q-Transformer](https://proceedings.mlr.press/v229/chebotar23a.html)（CoRL 2023）は action の各次元を離散 token として自己回帰的に Q 評価する。Q-learning は近いが、完成した action 候補を query として並列評価する方式ではない。
- [TadTR](https://arxiv.org/abs/2106.10271)（2021）は “action queries” という語を明示的に使うが、意味は動画中の action instance を検出する prediction slots であり、RL action ID ではない。

## DropMerge への対応

一次論文が直接支持する範囲は次である。

```text
state tokens を K/V にする
出力/action ごとに異なる query を Q にする
cross-attention で出力/action 固有 feature を得る
同じ小さな head を各 query output へ適用する
```

DropMerge で最小の構成へ落とすと、概念的には次になる。

```text
state tokens:    [B,180,128]
action queries:  [59,128]
        ↓ 1-layer decoder cross-attention
action features: [B,59,128]

QR:
shared Linear(128,51)
        ↓
quantiles:       [B,59,51]

IQN:
action features → U [B,59,R]
taus            → T [B,K,R]
einsum(U,T)       → [B,59,K]
```

この最終 IQN 結合は、今回確認した先行研究に完成形のまま存在するわけではない。最も近い表現は「AQT の C51 advantage head を tau-conditioned low-rank quantile head に置き換える」である。
