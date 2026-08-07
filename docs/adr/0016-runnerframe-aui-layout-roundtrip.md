# RunnerFrame の dock サイズ制御は構造化シリアライザ往復と遷移時同期で行う

wxAUI (wxWidgets 3.3.1) には既存 dock のサイズを設定する公開 API が無い。dock サイズは wxAuiManager 内部の dock 構造体が持ち、`pane.dock_size` は dock が(再)生成される瞬間にしか読まれない (framemanager.cpp:2100,2110)。また pane を隠して dock が空になると dock ごと削除されてサイズはどこにも書き戻されず (:2135)、サッシドラッグ確定を通知するイベントも存在しない。RunnerFrame の主領域 50:50 ポリシーと「hide→show で幅を記憶する」挙動はこの 3 つの欠落と正面衝突する。

RunnerFrame は次の 2 機構だけで dock サイズを制御する。

- **構造化シリアライザ往復**: 生きている dock のサイズ変更は `SaveLayout`(実 dock サイズが `wxAuiPaneLayoutInfo::dock_size` に入る) → サイズ編集 → `LoadLayout`(末尾の dock 全再構築で pane 側サイズが反映される) の in-memory 往復で行う。従来の SavePerspective 文字列を直接手術する方式は、書式のバージョン依存とエスケープ処理が脆いため廃止した。
- **遷移時同期**: pane を隠す系の遷移(メニュー OFF・✕・maximize)の直前に、実 dock サイズを `pane.dock_size` へ書き戻す。再表示時は dock 再生成が公式にこの値を読むため、復元コードは不要。幅の記憶は frame 側のメンバではなく `pane.dock_size` 自体に持たせる(wx 3.3.2 の `MinimizePane` 実装も同じ idiom を採用している)。

これに伴い、描画毎の `wxEVT_AUI_RENDER` フックによるレイアウト監視(トポロジ文字列差分・常時サイズキャプチャ・CallAfter 多段遅延・再入ガード 3 フラグ)を全廃した。ポリシー適用は明示的な遷移(表示トグル・✕・restore・aux 追加・リサイズ coalesce・Reset)でのみ走る。

## Consequences

- 許容した挙動差: pane ドラッグ直後の自動補正が次の遷移まで遅延する / 補助 pane を別レイヤへ動かしたときの引き戻しは行わない / float 中に閉じた pane の再表示幅はドック中の最終同期値になる。改善方向の差分として、rect 内寸/外寸の混同による適用毎のサイズ縮みドリフト(Log 高さ ~24px/回)が消えた。
- `LoadLayout` は pane 配列を丸ごと差し替えるため、往復後は取得済み `wxAuiPaneInfo` 参照がすべて無効になる。live pane への書き込みは往復前に行い、読み出しは `GetPane` で再取得する。
- `SaveLayout` は dock に属さない pane(非表示・浮動)の dock_size を 0 で返すため、往復時は live の記憶値でバックフィルする(`TakeLayoutSnapshot`)。怠ると「隠す→リサイズ→出す」で記憶幅が消える。
- maximize 中の往復は wxAUI の復元情報(savedHiddenState)を壊し、restore で全 pane が消える。ポリシーは maximize 中は何もしない。
- **wxWidgets 3.3.2+ へ上げる際の注意**: 3.3.2 で pane minimize が実装され、`MinimizeButton(true)`(現状 no-op)が実ボタンになる。minimize 用の min-dock は pane を生ポインタで保持するため、minimize 中に LoadLayout 往復を行うと dangling pointer (UAF) になる。アップグレード時は maximize ガードと同様の minimize ガード(または MinimizeButton の見直し)を要検討。

## Follow-up: 実装位置

本機構は GUI 共通基底クラス `anet::rl::gui::AuiLayoutFrame`(gui.hpp/gui.cpp)として実装した。RunnerFrame はこれを継承し、pane 定義とレイアウトポリシー(主領域 50:50・幅不足時縮小・Eval 非表示時の frame 縮退)のみを持つ。
