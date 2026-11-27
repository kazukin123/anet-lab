---
title: NN の名前付き出力と要求時の nn_trace 生成
page_kind: サマリ
material: "[[docs/memo/0170_nn_trace_capture_unification_10rfd.md|0170_nn_trace_capture_unification_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-26
---
# NN の名前付き出力と要求時の nn_trace 生成

## 元の材料

[RFD0170　NN の名前付き出力と要求時の nn_trace 生成](../../../memo/0170_nn_trace_capture_unification_10rfd.md) blob:15ddeca6accd78480f380596f5bf402a49c03ddc

## 要点

- forward の途中結果を押し出す callback と、呼び出し側がキーを要求する capture の二つがあった。
- 名前付き出力を capture へ集め、呼び出し側が forward の前に必要なキーを指定する。
- nn_trace を使う step だけ Actor が値を切り出し、FP32 化と複製を行う。
- Conv2d pane、画像メトリクスの記録、テストなど、読む経路から要求が届くようにする。
- DefaultDQN 系と ImageCls で、要求時には従来と同じキーと値が得られることを調べる。

## 内容

### 目的と問題

forward の途中で作った Tensor を外へ持ち出す口が 2 つある。
追加の計算が要る出力を、この口には乗せられない。要求が無いので、乗せると常に計算することになる。

### 起草時点の現状

`nn_impl.cpp` は並行する作業で行が動いているので、位置はシンボル名で示す。実装時はシンボル名で探し直す。
`SelectAction` は 1 回の呼び出しで forward を 1 回だけ行う。TempoRL の skip 側の forward と、Learner が target の方策を呼ぶ `SelectAction` は callback を渡していない。
このほか、Q-Values の paneに、行動の受け口が設定されていないときに `EvalRunner::DoStep(action)` を直接呼ぶ予備の経路がある。本番では RunnerFrame が受け口を設定するので通らない。
Learner の診断は、学習の前に 1 回だけ渡す購読ヒント（`Agent::ConfigureScalarMetricSubscriptions`、ADR 0031）で「何が要るか」を受ける。Conv2d の pane は Run の途中で開閉するので、この経路には乗らない。

### 確定事項

`rl::Actor::MakeAction` に、3 つ目の入力として `MakeActionOptions` を足す。Actor は、その欄 `nn_trace` が真の呼び出しだけ nn_trace を作る。Actor は、読む側の数や状態を持たない。
`MakeAction` の 3 つ目の引数に、何も求めない既定値（空の `MakeActionOptions`）を付ける。基底と override の全部に同じ既定値を書く。追加の入力を求めない呼び出しは、今のまま 2 つの引数で書ける。
`dqn::Actor`（Rainbow を含む）と `ImageClsActor` を同時に移す。`TraceCallback` 型、callback の引数、`MakeActionTraceCallback` は、別名や互換の口を残さずに一度に消す。
ADR 0055: 「選ばなかった形」が nn_trace の capture への統合を本 RFD へ先送りしているので、統合を決めた旨を冒頭に注記する。本文は書き換えない。

### 仕様

`dqn::Actor` が nn_trace を取り出すのは、行動を選ぶ forward（今 callback を渡している forward）だけである。TempoRL の skip 側の forward からは取り出さない（今と同じ）。
Learner が target の方策を呼ぶ `SelectAction` は capture を渡さない（今と同じ）。
Runner は、受けた値をそのまま、その step の `MakeAction` へ渡す。`DoUpdateFrame` は、そのフレームの全 step に同じ値を渡す。
要求の照合、要求していない鍵と二重の書き込みの即エラー、出口での満たされない要求の即エラーは、capture の今の決まりをそのまま使う。branch の入力は必ず書かれるので、branch が 1 つでも実行されれば `nn_trace/` の要求は満たされる。

### 実装契約

過去の記録（`docs/memo/done/` の RFD、ADR 0031・0037・0055 の本文、実験記録）は書き換えない。設計文書の図にある `MakeAction(state)`・`DoStep()` のような略記は、既定値つきの引数を省いた呼び方として正しいままなので直さない。
検索に使った語は `TraceCallback`、`MakeActionTraceCallback`、`AppendTraceAux`、`ExtractNnTrace`、`kNnTracePrefix`、`nn_trace`、`MakeAction`、`DoStep`、`DoUpdateFrame`、`SelectAction` で、追跡しているファイルの全体を引いた。型名を書かずに関数を渡す呼び出しは `TraceCallback` では出ないので、`std::string_view` と `torch::Tensor` を引数に取る関数の形でも `core/`・`apps/` を引いた。当たるのは、`TraceCallback` の定義、`NetworkBranch::Execute`、`MakeActionTraceCallback`、`nn_test.cpp` の 3 か所だけである。

### 受入

メニューの Conv2d のコマンドイベント（`ID_Conv2d`）を送って pane を開き、`ID_EvalStep` のコマンドイベントで 1 回進める。その通知にはある。
タイマーの経路は、pane を開いた状態と閉じた状態のそれぞれで、EvalPanel のタイマーのイベントを 1 回送って進め、開いているときだけあることを確かめる。タイマーは利用者の操作ではないので、イベントを直接送るのは準備に当たる。再生中の動きそのものは、人が確かめる項で見る。
Trace の pane を開いて記録し、Conv2d の pane を開いていた間の行を選ぶと、Conv2d の控えが表示される。

### ほかの RFD との関係

093（done/093）: capture の今の決まり（鍵と接頭辞の要求、`GetAll`、実行位置の付与、出口の検査）が前提である。本 RFD は、公開の口が capture を引数で受ける形と、093 の診断を変えない。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
