---
title: WARN の意味と読まれない設定の検出
page_kind: サマリ
material: "[[docs/memo/0117_warn_policy_and_unread_config_10rfd.md|0117_warn_policy_and_unread_config_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# WARN の意味と読まれない設定の検出

## 元の材料

[RFD0117　WARN の意味と読まれない設定の検出](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) blob:2d632d47f09867b763ab362f9ead0d36260d353a

## 要点

- WARN を、利用者が設定・コード・環境を確認すべき状態にそろえる。
- 明示的に機能を休ませた場合や、設定どおりの結果を知らせる場合は INFO にする。
- 実体化したカタログ項目に未読の設定があれば構築時のエラーにする。
- それ以外の未読キーは Run 構築の終わりに WARN でまとめ、書き誤りに気付けるようにする。
- 未知の値や明示要求された入力の欠落は失敗として扱い、通常の Run で WARN が出ないことを確かめる。

## 内容

### 起草時点の現状

Logs pane は表示 level 以上の WARN を数え、▾の `Mark all as viewed` で確認済みにする。表示しない level の記録と Clear した記録は数えない。
旧キー専用の WARN: RunnerApp.cpp の `WarnDeprecatedRunsDirConfig`。クリーンブレーク方針（82・92 行）に反する。
個別の葉と継承値の競合の WARN — 不採用。本 RFD は競合を見ないので開き直さない。
DQN 側で Noisy の有無を見て効かない倍率に WARN を出す形 — 不採用。本 RFD の一覧は「読まれたか」だけを見るので、読まれて効かない設定キーには出さず、開き直さない。

### 確定事項

「Run の入力」は Run の結果に効く入力（設定、データセット、checkpoint、ROM）を指す。workspace の履歴のような補助のファイルは含めない。
上書き層や Run プロファイルで、既にある block の `type` だけを別の種類に書き換える書き方は、元の種類の設定キーが残るので止まる（設定には設定キーを消す書き方が無い）。別の名前の block を定義し、structure の側で差し替える。出荷設定にこの書き方は 0 件。
遅れて作る読み口（View の menu で開く panel など）は、Run の構築時に設定を読んで検証する。
例（条件ではない）: RunnerApp は、どの選択の宣言にも現れない mode の prefix（Env の既定の online の Run での `app.batchrun.*`）を、読んで検証するか、休止を宣言する。

### 実装契約

「対象ごとに 1 回」は、Run ごとに数える。同じプロセスで Run を作り直したら、また 1 回出る。NN の複製（target・Actor 用・skip・RND）で同じ構築を繰り返しても、同じ対象は 1 回にする。
ログのレベルの名前、Run のログの `[W]`・`[I]` の書式、Logs pane の数え方は変えない。
Run の実効設定から読み口へ渡した設定は、どの読み方（`Config` の読み取り、`ConfigData` の直接の `Get`・`Read`、カタログの項目へ分けた設定、複製した設定）で読んでも、元の完全なキーで「読んだ」と記録される。
210 の 129・474・975 行: Metrics Viewer 自身のログで、Run のログではないので対象外。

### 検証と受入

番号付きの bat（`apps/11_batch_run.bat`・`12_batch_run.bat`・`13_batch_run.bat`）に、実装の時点で書かれている Run のうち、`run.$` の組み合わせが違うもの（`goto` で飛ばされている段落を含む。2026-11-10 時点で約 17 通り）。
学習する Run は、Run のフォルダの `metrics.jsonl` に、learn 側（`@learn`）で宣言した scalar のレコードが 1 件以上書き出されるまで流す。learn 側の scalar を宣言していない Run は、learn step が 1 以上になるまで流す。評価の完了は待たない。
アプリの側で読む設定キー（`metrics.image.*`、`app.eval_trace_panel.*` など）が一覧に出ないことも、この Run で確かめる。
全体の block の設定キーを Agent の下の同じ block の設定キーで上書きした設定で、上書きされた側の設定キーが一覧にも出ず、構築も止まらない。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
