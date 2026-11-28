---
title: "TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD"
page_kind: サマリ
material: "[[docs/memo/0098_temporl_skip_policy_10rfd.md|0098_temporl_skip_policy_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/subjects/gridmaze.md|GridMaze]]"
updated: 2026-11-28
---
# TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD

## 元の材料

[TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）RFD](../../../memo/0098_temporl_skip_policy_10rfd.md) blob:ec84ac85236e4822f41dd091adf7833abf4e1e5c

## 要点

- TempoRL は、どの行動を選ぶかと、選んだ行動を何 step 続けるかを分けて学習する。
- DefaultDQN の Actor にスキップ実行と長さの探索を加え、行動を条件とした skip Q を学ぶ。
- ReplayBuffer は可変長区間を取り出し、一つの継続区間の内部にある複数の開始・終了の組を学習に使う。
- 設定で無効化し、通常の1 step ごとの判断と比較できるようにする。
- 判断回数と同じ経験 step での学習速度を目標にし、DropMerge の長い NOOP の抑制は目的に含めない。

## 内容

### 行動の繰り返しを学習する理由

Env が行動と回数の組を一つの離散行動として出す方式では、回数の候補を増やすたびに行動空間が広がる。TempoRL は行動と継続時間に別の価値を持たせ、継続中の部分区間にも学習信号を与える。

### Actor と ReplayBuffer

Actor は直前の判断で選んだ行動を予定した回数だけ実行する。ReplayBuffer はエピソードの境界をまたがない可変長の部分区間を取り、元の行動、継続時間、報酬と終了情報を Learner に渡す。

### 特徴をどこで共有するか

primary network は従来の行動価値を計算し、公開した body の出力を skip network の特徴として渡せる。skip network は別の online と target を持つが、通常の行動選択で用いる network と、Learner が bootstrap に使う側を区別する。特徴を取り出すためだけに新しい経路で同じ forward を実行しない。

skip Head の既定は全行動のスキップ価値を出して選択行動の分を取り出す方式である。もう一方は行動を skip network の入力として渡し、その行動の各継続時間の価値を計算する。いずれも方策が読むのは「すでに選んだ行動をどれだけ続けるか」の値である。TD、QR、IQN の場合にそれぞれスカラーや分位を返し、分位を使う UQE・Thompson とも組み合わせられる。

### 区間を作るときの境界

ReplayBuffer は通常の1 step の記録を原本とし、スケール適用後の報酬、終了種別、Actor の判断フラグを保持する。抽選時は同じ判断で選んだ行動の間をたどり、次の判断、エピソードの終わり、または最大スキップ長で窓を切る。起点から選べる継続時間のうち一つを抽選し、その長さ、割引報酬、bootstrap に必要な観測を返す。元の区間が長ければ同じ経験から違う部分区間が候補になるが、一つのサンプルで全組を列挙するわけではない。

### 学習と探索

通常の行動価値に加え、行動に条件づけた skip Q を学ぶ。スキップ長の探索と通常の行動探索を識別し、勾配や乱数への影響が意図した範囲かを確かめる。

### 勾配と優先度

特徴を共有する場合、skip loss が primary にも勾配を流すかを設定で選ぶ。optimizer を一つに保ち、通例の Q loss と重み付き skip loss を足して一度だけ backward する。Q と skip は同じ replay サンプル、同じ importance-sampling 重みを使う。PER の優先度の既定は Q 側の TD 誤差だけで、skip 側も寄与させるときは係数を明示する。global norm のクリップが両経路に影響するため、勾配を止める設定と optimizer を分ける方式が常に等価とはみなさない。

### 長さの探索と診断

既定の長さ探索は、行動の探索方策の ε、UQE の τ、lane ごとの設定を使う。長さを決め直す時点と前の判断に従って行動を継続する step を分けて記録し、Evaluation Trace でも判断の有無と選んだ長さを読めるようにする。強制行動で事後に書き換わった場合の扱いも原本に定め、外から見えた行動だけで内部の残りスキップ長を推測しない。

### 検証

スキップ無効時は GridMaze、LunarLander、Atari、DropMerge の既定条件で学習メトリクスを変更前と照合する。有効時は判断の頻度、同じ経験数での評価 AUC、seed 間のぶれを用いて比較する。

### 実験を進める順序

6×10 の GridMaze は最大7 step、LunarLander は4、Qbert は10を最初の比較条件とする。その後、DropMerge は基準 checkpoint から分岐させる。評価 AUC が同 seed 数の対照のばらつきを超えて悪化しないことと、無効時の判断率1から実際に頻度が下がることをそれぞれ調べる。スキップ中の forward そのものを省略して得る速度改善は、この RFD の効果に数えない。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [GridMaze](../../concepts/subjects/gridmaze.md)
