---
title: Metrics Viewer の名前付きタグセット
page_kind: サマリ
material: "[[docs/memo/0111_metrics_viewer_tag_sets_10rfd.md|0111_metrics_viewer_tag_sets_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
updated: 2026-11-28
---
# Metrics Viewer の名前付きタグセット

## 元の材料

[RFD0111　Metrics Viewer の名前付きタグセット](../../../memo/0111_metrics_viewer_tag_sets_10rfd.md) blob:223762d4697720e7e05cfb40cd512a208d1a0f7c

## 要点

- 観点ごとにタグを選び直す手間を減らすため、選択を名前付きで保存する。
- サーバーに保存して Workspace とブラウザをまたいで使う。
- 呼び出すと一覧に見えていないタグを含む選択全体が保存時の内容に置き換わる。
- 新しいタグは自動選択し、保存内容との差を変更印で知らせる。元に戻す操作も用意する。
- 名前変更・削除・内容の編集を扱い、再起動時は選択中の名前と未保存の選択状態を復元する。

## 内容

### 確定事項

呼び出すと、タグの選択全体をタグセットの中身で置き換える。Tags 一覧に出ていない tag も選択に入り、それまでの他の選択は外れる。
保存は選択全体を書く。変更の印も選択全体で比べる。
タグセットを選んでいる間も、新しい tag は今どおり自動で選択に入れる。そのため `*` が付く。
入口はモックの案 B。Tags の下の 1 行を `[選択中の名前 ▾][Save][Revert]` とし、▾ で開く一覧の中で呼び出し・名前変更・削除・新規保存を行う。

### 契約

jar のリソース `initial_tag_sets.json`（ソースは `apps/metrics-viewer/src/main/resources/initial_tag_sets.json`）。設定キーは持たない。
tag がどの Run にも無いことは検証しない（Run を消してもタグセットの中身を失わない）。
名前の UTF-16 コード単位順、各タグセットの中は tag の同じ順で、2 字下げ・1 行 1 tag・UTF-8（BOM なし）・LF・末尾に改行 1 つ。サーバーは保存のたびにファイル全体をこの形で書き直す（手で入れた並びや空白は保存で整う）。
削除: 確認のあと `delete` を送る。成功の応答を受けたら、選択中のタグセットなら「なし」にし、そのあとで応答の一覧に E20 を当てる（自分で消したので Toast は出さない）。タグの選択は変えない。

### 移行対象

既存の契約を置き換えるものは無い（足すだけ）。今の振る舞いのうち変わるのは、localStorage の state の件数、サイドパネルの Tags の区画の高さ（1 行増える）、Viewer の起動で workspaces-dir（既定の起動では `apps/runner/workspaces`）が無ければ作られることである。

### 受入基準

呼び出すと、Tags 一覧に出ていない tag を含めて選択がタグセットの中身と一致する（localStorage の `anet.metricsviewer.activeTags` で確かめる）。
tag を 1 本切り替えると開閉ボタンに ` *` が付き、title に差の数が出て（Tags 一覧に出ていない差が 0 なら括弧が無い）、`Save`・`Revert` が押せるようになる。`Save` で印が消え、`GET` の中身が選択と一致する。印が無いとき `Save`・`Revert` は disabled である。
選択中のタグセットを行の中で名前変更すると、選択中の名前と localStorage の `anet.metricsviewer.tagSet` が新しい名前（前後の空白を除いた名前）に追随し、Toast は出ない。選択中のタグセットを削除の確認で消すと「なし」になり、Toast は出ない。
行の並びが UTF-16 コード単位順（例: `Atari スコア`、`PER`、`eval 成績`、`可塑性` の順）で、`<見える数>/<数>` が出る。選択する Run を変えると見える数が変わる。

### 成功の測り方

少し足し引きしたあと元へ戻すのが `Revert` の 1 操作で済む（受入 3）。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
