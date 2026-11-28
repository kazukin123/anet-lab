---
title: Runner の主 pane の配置と Train View の表示
page_kind: サマリ
material: "[[docs/memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md|0110_runner_main_pane_dynamic_centre_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Runner の主 pane の配置と Train View の表示

## 元の材料

[RFD0110　Runner の主 pane の配置と Train View の表示](../../../memo/done/0110_runner_main_pane_dynamic_centre_10rfd.md) blob:76e6cadf040b361d13421aa55e4eba1fd9860c2a

## 要点

- 学習を続けたまま Train View を隠し、Evaluation View のみを広く使えるようにする。
- Train と Evaluation の二つを主 pane とし、表示と dock の状態に応じて中央を占める pane を決める。
- 中央の pane が消えたとき wxAUI が空白を残す挙動を考慮し、残った主 pane を配置し直す。
- 表示・非表示と dock・float の組み合わせ、メニューやツールバーでの操作を扱う。
- pane の表示を変更したときにも、学習や評価の実行状態と画面の状態を独立に保つ。

## 内容

### 確定事項

Train View の描画 timer は、表示状態と FPS から 1 か所で決める。 ポリシーの入口（maximize のガードより前）で、Train pane の表示状態（`IsShown`）を `TrainPanel` へ渡す。`TrainPanel` は「表示中か」と「選ばれている FPS」の 2 つから timer を回すかを 1 か所で決め、`SetFps`・`Initialize`・表示状態の変化のどれからも同じ判定を通す。隠れている間は timer を止め、`SetFps` は値を覚えるだけにする。表示に戻ったら、その時点の FPS で始める（0 なら始めない）。`Initialize` 前は表示状態を覚えるだけで timer に触らず、`Initialize` のときにその状態で決める。Train 側の observer は外さない。
ログは保険の WARN だけ。 pane の開閉・昇格・降格でログを出さない。保険が発動したときだけ WARN を 1 行出す。

### 実装契約

設定: common.txt に `app.main_view ?= eval`。`[shipped_config]` の検査を通す。`run.@evalonly`（Atari.txt:425）には足さない（既定が `eval` のため）。
互換の扱い: Train を隠せない今の形へ戻す設定や切替は作らない。既定が `eval` に変わることへの案内や互換の既定は置かない。

### 受入基準

h. Train View FPS を `0 (Off)` にして Train View を OFF、ON: timer は止まったまま。Train を OFF にしたまま FPS を 10 にしても timer は止まったまま。Train View を ON にすると timer が 100 ms の周期で動く。
i. Reset Layout: a と同じ状態（Eval だけ）に戻る。g で覚えた Left の位置は消え、Reset のあと Train View を ON にすると Eval は Right、layer 0 に出る。
m. a の状態（Eval だけ）で Eval の pin を押したことにしても、Eval は docked のまま Centre にいる。両方を表示して Train の maximize を押したことにし、Train の pin を押したことにすると、maximize が解けて Train が float し、宣言が追いついたあと j と同じ状態になる。
n. 両方を表示した状態で、Logs と、表示した Q-Values・Evaluation Trace の pane の `HasPinButton()` が true。Logs の pin を押したことにすると Logs が float し、Train・Eval の方向・Floatable・✕・pin・トグルは b のまま。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
