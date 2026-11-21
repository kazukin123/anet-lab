# Notifier の Detach は進行中の通知を待たず、Observer はロックの外で呼ぶ

`Notifier` の Observer の列は、学習・評価・GUI の 3 つのスレッドが排他なしに読み書きしていた（[PRD192](../memo/192_concurrency_and_boundary_defect_fixes_10prd.md)）。列を守るにあたり、`Detach` はすぐ戻り、その時点で始まっていた通知は外した Observer にも届きうる形にし、その通知が終わるまで Observer は `Notifier` が生かして、Observer を呼ぶあいだ `Notifier` はロックを持たない。外した Observer は通知のスレッドで壊れうるので、Observer は、触る相手（Panel など）の寿命を自分で確かめ、どのスレッドで壊れてもよいように書く。

## Considered Options

- **`Detach` が、その Observer への進行中の呼び出しが終わるまで待つ**: 戻ったあとは呼ばれないので、相手を安全に壊せる。しかし、待たれている Observer が GUI スレッドの持つロックや GUI の処理を待つと、互いに待って固まる。Observer の中から自分を外すときの特例も要る。却下。
