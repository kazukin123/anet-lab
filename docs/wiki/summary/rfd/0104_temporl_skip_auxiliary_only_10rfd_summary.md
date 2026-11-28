---
title: TempoRL の補助学習と行動固定の切り分け
page_kind: サマリ
material: "[[docs/memo/done/0104_temporl_skip_auxiliary_only_10rfd.md|0104_temporl_skip_auxiliary_only_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# TempoRL の補助学習と行動固定の切り分け

## 元の材料

[RFD0104　TempoRL の補助学習と行動固定の切り分け](../../../memo/done/0104_temporl_skip_auxiliary_only_10rfd.md) blob:54f184bc88467c10c40c49d2d564131865866ae9

## 要点

- Qbert で序盤の成績と後半の遅れが異なる傾向を示し、行動の固定と補助損失を分けて調べる。
- Actor ごとに毎 step 判断する設定、最大長まで選ぶ設定、時間で上限を変える設定を用意する。
- 毎 step 判断しても、同じ行動の連続区間やエピソード終端までの区間からスキップ価値の教材を作れる。
- この教材を使うときは全 Actor の判断間隔を1にし、スキップ出力は補助学習だけに使う。
- LunarLander での結果は Qbert の比較群を削る根拠にせず、環境別に効果を判定する。

## 内容

### 目的と問題

序盤の得は skip の loss の重みの順に並ぶ。 最初の 5M の ε=0 の評価と最初の 10M の学習側の平均点は、合算 > 重み 0.3 > 遮断 ≈ 無効だった。遮断は行動の固定と探索が合算と同じで、skip の loss が共有部に届かない点だけが違う。
中盤の遅れは重みによらない。 遮断・合算・重み 0.3 の 3 本とも、学習側の第 8 面の突破率と ε=0 の評価が 30M 以降に無効より遅れ、50M まで戻らなかった。遮断では Q の学習が無効と違うのは「行動を固定して集めたデータ」だけである。
このことから、TempoRL の得は skip の loss が共有部を形づくる補助タスクの効果で、損は行動を固定すること（着地の瞬間の方向を、その前の状態で決めてしまうこと）から来る、という仮説が立つ。仮説が正しければ、行動は毎 step 選び直し（スキップ長を常に 1 にし）、skip の loss だけを補助タスクとして学ぶ形で、得を残して損を消せる。

### 起草時点の現状

checkpoint は Network 4 object と optimizer の役割を持ち、窓の規則や Actor の skip の状態は持たない。
序盤（5.05M の評価と最初の 10M の学習側）は、skip の loss が共有部に届く量の順（合算 1.0 > 重み 0.3 > 遮断 0）に並び、遮断は無効と同じだった。行動を固定することと、固定による探索の形の違いは、序盤の得を生んでいない。
中盤の遅れは重みによらず 3 本で同じなので、共有部での勾配の混ざり方では説明できない。残る経路は行動側（固定した行動が、着地の瞬間の状況を見ずに決まる）と、その行動で集めたデータの偏り（後半の面まで行くゲームが半分しかなく、第 7・8 面のデータが少ない）である。学習の Actor は NoisyNet の摂動で選んだ行動も固定するので、行動側の誤りは学習側でより大きく出る。
TempoRL の論文は重み共有を「少ない重みで学べる」構成として述べ、skip を補助の loss としては位置づけていない（論文解説 3.3）。

### ゴール・非ゴール・仮説と成功の測り方

解く問題: 今の実装では、行動の固定と skip の loss の補助学習が一体で、qbert の序盤の得と中盤の損がどちらから来るかを切り分けられない。
Actor ごとに、選べるスキップ長の幅を決められる。常に 1（毎 step 判断）、1〜J（今と同一）、1〜`max_length`（ProfiledValue。時間で変えられる）の 3 つ。学習済みの TempoRL を固定なしで評価する切り分け（ステージ 0）にも使う。
skip の loss の目標値を作る区間（窓）を、判断ではなく、同じ行動が続く連続区間か、エピソードの終わりだけで閉じる区間で切れる。すべての Actor がスキップ長を常に 1 にするときだけ使え、skip の出力は行動に使わない補助タスクになる（補助学習のみモード）。
途中まで固定し、途中から毎 step 判断にする切り替え（スケジュール）を段に入れること: `variable` の ProfiledValue で書けるようになるが、仮説の判定に要らない。制約ではなく、ゴールに要らないことによる非ゴール。

### 確定事項

`full` は今と同一。`single` は、すべての lane のすべての step を判断にし、行動は方策の提案をそのまま使い、スキップ長を 1 にする。skip の方策は j の選択だけを省き、スケジュールの更新と τ の問い合わせは今と同じにする（aux の `skip_q_values` の問い合わせが `full` と変わらない）。skip Network の forward と、aux の `skip_decision`（すべて真）・`skip_length`（すべて 1）・`skip_q_values`（J 本）は今と同じく出す。判断のフラグはすべての step で立つ。`variable` は、`max_length` を exp_step で評価した上限 c の範囲（j = 1〜c）だけから j を選び、aux は J 本のまま出す。判断で決めた残りの区間は c が下がっても続ける。
窓の規則を 2 つ足す。`action_change`（連続区間の窓）は、起点の行動と同じ行動が続く間だけ伸ばし、違う行動の slot の手前で閉じる。`episode_end`（終わりだけの窓）は、行動も判断のフラグも見ず、終わり方の slot を含めて閉じる。どちらも上限 J で閉じ、終わり方の slot は含めて閉じ、判断のフラグは見ない。k は今と同じく起点の k_max 以下から一様に引く。
補助学習のみモードは、新しい値を足すのではなく、`learner.skip.target_window` が `decision` 以外で、全 Actor のスキップ長が常に 1（`skip.max_length_mode = single`、または `variable` で常に 1）の組み合わせで表す。群のプロファイルは `single` で書く。群の設定は検証資材のプロファイル `run.@skip_aux`（`core/anet-core/testdata/prd104/arms.txt`）に置き、env の設定ファイルには置かない。

### 実装契約

`action_change`: k_max = max { k ≤ J : t+1..t+k−1 の行動が t の行動と同じで、t..t+k−2 の終わり方がすべて「なし」}。行動が違う slot の手前で閉じる（今の規則が行動の不一致をエラーにする箇所が、閉じる条件になる）。判断のフラグは見ない。
`episode_end`: k_max = max { k ≤ J : t..t+k−2 の終わり方がすべて「なし」}。行動も判断のフラグも見ない。
dummy の slot が窓の中に入ったときのエラーは今と同じにする（終わり方で閉じるので跨がない。ADR 0063 の Consequences）。
振る舞いが変わる既存の設定・テスト・サンプルは無い（新しいキーの既定が今の振る舞いである）。互換のための仕組みは足さない。過去の Run と記録は書き換えない。暫定版のキー名（`skip.execution`、`learner.skip.window`）は実装されていないので移行対象は無い。

### 検証と受入

対照（`priority_lambda` は既定の 0、`skip.head.use_dueling` は既定の false）は、無効と同じ帯に入るかを見る観察で、合否にはしない。同じ帯に入らなければ、`single` の経路か窓の抽出が Q の学習に効いている疑いとして記録し、ステージ 2 の前に原因を調べる。
`action_change` の窓: 起点が連続区間の途中でも、残りの長さが k_max になる。行動が変わる slot の手前で閉じる。判断のフラグが立っていても行動が同じなら閉じない。done・truncated の slot を含めて閉じる。上限 J で閉じる。窓の中の dummy はエラー。
`episode_end` の窓: 行動が変わっても判断のフラグが立っていても閉じない。done・truncated の slot を含めて閉じる。上限 J で閉じる。窓の中の dummy はエラー。
`single` の Actor: 2 lane 以上・数 step で、返す行動が方策の提案と一致し、判断のフラグと `skip_decision` がすべて真、`skip_length` がすべて 1、`skip_q_values` が J 本。

## 主要な概念

- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [Qbert](../../concepts/subjects/qbert.md)
- [LunarLander](../../concepts/subjects/lunarlander.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
