# Runner の主 pane は Centre を動的に占有し、Centre を空にする操作を導線に持たない

## 元の材料

[ADR 0066](../../../adr/0066-main-panes-occupy-centre-dynamically-and-no-flow-empties-centre.md) blob:14aec6a5a2d4b53c888b4bac6c1161bea19937c2

## 要点

- wxAUI は中央 dock が空くと側面を広げず、中央を背景で埋める。
- Train View と Evaluation View の表示状態に応じて Centre の占有者を切り替える。
- 相方が表示・dock 済みのときだけ主 pane を隠したり float したりできる。
- 最後の主 pane を閉じて Centre を空にする操作は提供しない。
- 2つを単一の splitter pane にせず、それぞれの浮動や最大化と利用者の配置を維持する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
