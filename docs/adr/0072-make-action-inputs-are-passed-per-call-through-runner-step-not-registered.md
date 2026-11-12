# 行動選択に添える入力は Runner の step と MakeAction の引数で都度渡し、core に要求の登録を持たない

Conv2d の可視化が読む nn_trace は、読む人がいなくても Actor が `MakeAction` のたびに作っており（[ADR 0031](0031-plasticity-metrics-out-of-band-partial-forward.md) の A''、[PRD938](../memo/938_nn_trace_capture_unification_10prd.md)）、読む側は Run の途中で開閉するので、学習の前に 1 回だけ渡す購読ヒントには乗らない。そこで「その回の行動選択に何を添えるか」を構造体 `MakeActionOptions` にし、step を進める側（EvalPanel、学習のスレッド）が毎回決めて `Runner::DoStep`・`DoUpdateFrame` の引数で渡し、Runner がそのまま `Actor::MakeAction` へ渡す。入力は引数で渡し、出力は Observer で受けるという分担を保つことと、step と行動選択の結果が引数だけで決まることを、読む側が自分だけで要求できる手軽さより優先した。

## Considered Options

- **Runner が要求の数を持ち、読む側が Runner に登録する**: 変更は小さく、読む側が自立する。しかし step の結果が引数に現れない数で決まり、入力の側に「事前の登録」という 2 つ目の渡し方ができる。却下。
- **Actor が要求の数を持つ**: `MakeAction` の出力が、別のスレッドから変わる値で決まる。却下。
- **Observer が「要る」と宣言し、Runner が `Notifier` に聞く**: 全イベント・全 Runner が共有する仕組みに特定の出力の話が入り、Learner の購読に近い形が Actor 側にも増える。却下。
- **購読ヒントと同じ経路、または Actor の設定キーで渡す**: Run の途中の開閉に追従できない。却下。
- **真偽の値をそのまま引数にする**: 入力が増えるたびに、基底と全実装の引数が変わる。却下。

## Consequences

- 読む側は自立しない。nn_trace を読むものや、step の進め方を新しく足すときは、step を進める側で値を渡す。
- 行動選択に添える入力が増えるときは、同じ構造体の欄として足す。
- 学習のスレッド（`RunnerThread`）は step を進める側であり、作るときに受けた値を step のたびに渡すだけで、読む側の登録先ではない。
