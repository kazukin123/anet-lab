---
title: Evaluation Trace の行移動と強制行動の識別
page_kind: サマリ
material: "[[docs/memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md|0105_eval_trace_arrow_key_forced_action_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Evaluation Trace の行移動と強制行動の識別

## 元の材料

[RFD0105　Evaluation Trace の行移動と強制行動の識別](../../../memo/done/0105_eval_trace_arrow_key_forced_action_10rfd.md) blob:5a7bac1768db6f6a9d169af5dbda22cd4484e335

## 要点

- 表を調べているときに矢印キーが EvalPanel の強制行動になる問題を扱う。
- Trace モードで他のパネルから転送された上下キーは表示順に1行移動し、左右キーは行動を起こさない。
- 実行時に行動を差し替えたかを lane ごとに記録し、Trace の列から Agent の判断と見分ける。
- Live モードでの強制操作も、その場で利用者が気付けるようにする。
- 表の並べ替えと選んだ行のダンプを用意し、列に出ていない情報も調べられるようにする。

## 内容

### 問題

qbert の行動は 0 NOOP、1 FIRE、2 UP、3 RIGHT、4 LEFT、5 DOWN である（Trace の action 列の表示名）。そのため、↓ キーは UP、↑ キーは NOOP になる。
Trace を見るつもりで ↑↓ を押しても、フォーカスが表以外（Env の View、QValuePanel など）にあれば、行は動かず、EvalPanel が強制行動で 1 step 進む。強制すると Live モードへ戻るので、見ていた行の選択も外れる。キーを押し続けると、キーの自動の繰り返しの数だけ step が進む。
強制した行と Agent が選んだ行を、Trace の上で見分けられない。強制した行を Agent の挙動として読んでしまう。
行の列に出していない値（全行動の Q 値、行動の情報の info、観測）は、Trace の上で見られない。step trace を Run の記録に残す機構もまだ無い（999_eval_session_step_trace、081）。

### 確定事項

EvalRunner が答える `$runner` の鍵 `forced` を足す。EvalRunner は、最後の step で行動を指定された（強制行動に差し替えた）かを lane ごとに覚え、lane を指定した問い合わせに 1、Actor の選んだ行動のままなら 0 を返す。lane を省いた問い合わせには値を返さない。lane が範囲外なら fail-fast にする（`$action_info action` と同じ扱い）。EvalRunner 以外の Runner はこの鍵を持たない。`WithAction` と行動の情報の型は変えない。
強制した step のたびに、`EvalPanel` がステータス欄 0 に `Forced action: <行動の表示名>` を出す（例: `Forced action: 2:UP`）。表示名は Trace の action 列の既定と同じ「番号:名前」の形にする。強制行動の入口は離散の行動番号だけなので、連続の行動の書き方は持たない。対象はテンキー、Live モードの間の矢印、QValuePanel のクリックである。行動を指定しない `Ctrl` とツールバーの Step では出さない。ログには出さない。
並べ替えている間（Live モードの間）に来た新しい行と、あとから値が決まった行（近似優先度など）は、並びの位置へ入れ直す。表に残す行は、並べ替えによらず step の新しい順に `max_rows` 件で、上限を超えたら並びによらず step のいちばん古い行を消す。

### 実装契約

Trace モードでは必ず行を選んでいるので、転送された ↑↓ は選んでいる行を起点に数える。
EvalRunner 以外の Runner は `forced` を持たない。問い合わせると、今の未知の鍵の扱い（列では最初の読み出しで fail-fast、scalar では値が見つからない WARN）になる。
値は「最後の step」のものである。Evaluation Trace の列は通知の中で同期に読むので、その行の step の値になる。
直さないもの: testdata/prd094/compare_configs.py と `testdata/prd103/resolved-before/` は、それぞれの PRD の検証の記録で、当時の既定列を持つ。過去の記録なので書き換えない。

### 受入基準

単体テスト: `$runner forced` を列に持つ step trace を EvalRunner の通知から作ると、`DoStep(action)` の行が 1、`DoStep()` の行が 0 になる（通知の中で読んだ値が、その行の step の値である）。
単体テスト（既定列）: common.txt を読んで解いた設定の `app.eval_trace_panel.columns.[260]` が `$runner forced label:forced` で、ほかの既定列の値が今の common.txt と同じである。
テンキーの 2 を押すと 1 step 進み、新しい行の `forced` が 1、ステータス欄に `Forced action: 2:UP` が出る。Agent が選んだ行の `forced` は 0 である。
手動の確認（Runner、LunarLander の online）: Evaluation Trace を閉じた状態と、開いて Live モードの状態で、矢印キーの強制行動が今までどおり 0〜3 になる。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
