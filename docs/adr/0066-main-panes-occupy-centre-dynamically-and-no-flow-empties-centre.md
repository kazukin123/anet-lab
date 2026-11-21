# Runner の主 pane は Centre を動的に占有し、Centre を空にする操作を導線に持たない

wxAUI は Centre dock が無いと中央を背景で埋めて側面の dock を広げないので、Train View を隠せるようにするために、Train View と Evaluation View を 1 つの pane にまとめる形は採らず、Centre を占める pane は隠れるか float するまで占有者のままで、離れたら表示中で docked の相方が Centre を占めて戻れば取り戻し、主 pane を隠す・float できるのは相方の主 pane が表示中で docked のときだけ（最後の 1 つは ✕ を持たずトグルは無効で、動かせない）ことに決める。pane の表示切替の仕組みを基底 `AuiLayoutFrame` の 1 本に保ち、Eval の float・maximize・✕ を失わず、利用者が明示的に行ったアンドックや dock の配置を ✕ の副作用で覆さないためである。✕・pin・Floatable を状態から宣言する作り、Centre を占める pane を caption の pin ボタンで float すること（wxAUI は Centre の pane の caption ドラッグを始めないため。`wxAuiManager` の派生で wx の判定を迂回しない）、昇格前の位置の記憶、知らない経路のための保険は PRD110 の D2・D3・D5・D13・D14 による。

## Considered Options

- **Train と Eval を 1 つの Centre pane にまとめ、wxSplitterWindow で分ける**: Centre は構造上空かず比率もリサイズに追従するが、Eval の float・単独 maximize・✕ を失い、表示切替が pane 系と splitter 系の 2 系統になる。却下
- **Eval の maximize で足りるとする**: Logs・Q-Values・Evaluation Trace も消え、起動時に選べず、Train の描画 timer は回り続ける。却下
- **float 中の相方を ✕ で Centre へ引き込む／Centre が空いたら隠れている方を出すことを導線の意味にする**: 利用者が明示的に決めた配置を、別の操作の副作用で覆す。知らない経路のための保険としてだけ置く。却下
- **主 pane を常に float 不可にする**: Train 表示中の Eval の float・移動の自由を失う。相方の有無で決めれば Centre を守るのに足りる。却下
- **Centre を占める pane を常に float 不可にし、docked の Train を常に占有者にする（初版）**: 2 つが合わさると Train だけが表示中は常に float できない。最後の 1 つを離れられなくすれば Centre は守れる。却下（2026-11-01）
- **占有者を Train 優先のまま、相方がいれば float だけ許す**: float した Train を縁の帯へ落とすと Centre へ跳び、側面に置き続けられない。占有者を今いる方にすれば落とした場所に留まる。却下（2026-11-01）
- **両方を Centre dock に置く**: 別の row では横に並ぶが間に sash が無く（今の 50:50 も遷移で比率を戻すので損は小さい）、float から戻した Eval は側面の dock にしか落ちないので結局昇格が要り、幅の縮小・記憶・テストを組み替える割に単純にならない。却下
- **✕ と Floatable をポリシーの適用の時機だけで付け外す**: ドラッグの float・dock は wxAUI がイベントを出さず、Eval を float したあとも Train の ✕ が残って押せる。却下
- **`wxAuiManager` を派生させて Centre の pane の caption ドラッグを始める**: wx の protected の状態と内部の呼び順に依存する歪で、wx の更新で壊れうる。Centre の pane は caption の pin ボタンで外せば足りる。却下（2026-11-01）
