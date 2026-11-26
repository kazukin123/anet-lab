# TempoRL の著者実装との比較と評価条件

## 元の材料

[RFD0103　TempoRL の著者実装との比較と評価条件](../../../memo/done/0103_temporl_reference_alignment_10rfd.md) blob:c4c59889b16938eef9fb5486ee4c7d035b0197e3

## 要点

- 著者の Atari 実装は Q とスキップの損失ごとに Adam の状態を持ち、anet の従来設定は合算した損失で更新する。
- 共有特徴への勾配を残しながら二つの Adam を使う分割更新を選択可能にする。
- 行動とスキップ長の両方を greedy にした評価で、学習したスキップ価値と判断頻度を観測する。
- 探索規則を変える群は設定で構成し、環境と時期をそろえて比較する。
- 著者の特徴量環境では遷移の長さが一律1として記録されるため、論文の結果を比較するときにこの条件を添える。

## 内容

### 目的と問題

PRD098 は論文と論文解説から TempoRL を組み、著者のコードとは突き合わせていない。2026-10-27 に突き合わせたところ、合算のときの共有部の更新の仕方など、論文に書かれていない実装上の差がいくつか見つかった。著者のコードの側にも、特徴量の環境の版で skip の学習が壊れている所がある。
合算の比べ方の土台が違う。 著者の Atari の実装は、skip の loss と Q の loss をそれぞれの Adam で更新する。anet は 2 つの loss を足して 1 つの Adam で更新するので、共有部を動かす力の配分が勾配の大きさで決まる。qbert では合算で primary の勾配ノルムが無効の約 2 倍になり、クリップが毎回発火した。サンプル効率優先の qbert の合算は、10M までは無効を上回ったが、中盤で第 8 面の突破が遅れた。
論文の結果との比べ方に誤りの余地がある。 著者のコードの特徴量の環境の版は、skip の遷移を全部 j=1 として記録する。PRD098 は LunarLander の J やステージ 2 の見方を論文の LunarLander の結果に照らしているが、その結果がこの版で出たかは分からない。

### 起草時点の現状

対象外（差として挙げない）: anet がベースの Agent として持つアルゴリズムの差（IQN・分布型、PER、n-step、Munchausen、NoisyNet、Dueling、replay ratio）、ベースのハイパーパラメータ、env のラッパー。
著者: 要素ごとの clamp ±40 を loss ごとに掛ける（A:826、992-997、1015-1020）。Huber の平均の勾配では実質的に効かない見込み（推定）。特徴量の環境の版にはクリップが無い。
skip Network 自体の勾配ノルム（`50_agent_skip/15_grad_norm`。learn step の軸なので、learn step × 64 で exp step に換算した 5M 窓）は 5M 以降 3.2〜5.2 で、閾値に届いていない。
遮断と合算で Munchausen の clip 率が同じなので、上乗せは TempoRL 有効によるもので、合算によるものではない。

### ゴール・非ゴール・成功の測り方

学習したスキップ価値そのものを測れる評価の条件を用意し、その評価の判断率を記録する。行動もスキップ長も greedy の評価 session を各 env に置き、skip の指標の tag を env をまたいで並べられる形に揃える。
アルゴリズムの差（IQN・PER・n-step・Munchausen など）を著者に揃えること: anet のベースの Agent の選択で、TempoRL の比較の土台ではない。制約ではなく、ゴールに要らないことによる非ゴール。
著者の更新の順（skip を先に step し、更新後のパラメータで Q の勾配を取る）を真似ること: current の forward が 1 回増え、差は η の 1 歩分である。制約ではなく、効きが小さい見込みによる非ゴール。
行動をスカラーの添字で入れる形: 行動に順序を仮定する著者の癖で、真似る価値が無い。skip Network の入力の形は設定とクリーンブレークで移せるので制約ではなく、anet の one-hot と分離型のほうが素直で、実測でも分離型が包含型と同等以上だったことによる非ゴール。

### 確定事項

`split` の 2 つの勾配は、同じパラメータの値から取る（著者の「skip を先に step してから Q の勾配を取る」順は真似ない）。
`split` のクリップは、loss ごと・Network ごとに掛ける。Q の loss の勾配（primary のパラメータ）、skip の loss の勾配の primary のパラメータの分、skip の loss の勾配の skip Network のパラメータの分の 3 つに、それぞれ global norm で閾値 `grad_clip_tau` を掛ける。2 つ目のクリップのノルムと発火率は `17`・`18` で記録する。
新しい設定がすべて既定（`feature_gradient` は今の値、`sample_weight = uniform`）のとき、Learner の forward と backward の回数・順序・RNG の消費・loss・優先度・checkpoint の形は今と同一。Actor の累積は乱数・loss・出力に触れない。
合算で眠った・死んだユニットが増える件と、TempoRL 有効で Munchausen の log π の clip 率が上がる件は、観察だけにする。ステージ 1（遮断・重み 0.3・閾値 30）とステージ 3（分割）で値を並べ、判定には使わない。確かめる計測（判断のフラグで分けた clip 率）と手立ては本書に入れない。

### 実装契約

`sample_weight` はどの `feature_gradient` とも組み合わせられ、PER の有無で式は変えない。
current の forward は今と同じ（primary を forward し、その出力の `skip_context` をそのまま skip Network に渡す）。
`21`・`22`・`03` の `[01]`・`17`・`18` は metrics_scalar.txt の `@skip` に置く（J に依らない）。`23` と `03` の `[02]`〜`[J]`、`21_eval/09`〜`12` は各 env の `run.@skip` に置く（Atari は `53_evalg` の群があるので `23` と `03` だけ）。
Evaluation Trace は変えない（`skip_decision`・`skip_length` の列は PRD098 のまま）。

### 検証と受入

qbert の突破率: 学習側の、2.5M 窓のゲームのうち「29,000 点以上かつ `round_clear` ≥ 8」の割合（campaign の判定指標と同じ）。評価側（`53_evalg`）で同じ条件を数えたものは「評価のエピソードのうち突破したものの割合」と書き分け、突破率とは呼ばない。
2026-10-31 に測った（今のソースの Release）。比（split／merge）は 1.117（merge 0.0511・0.0513、split 0.0571・0.0572 秒／更新）。記録は検証資材の README の性能の節。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Qbert](../../concepts/subjects/qbert.md)
- [GridMaze](../../concepts/subjects/gridmaze.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
