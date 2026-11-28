---
title: Metrics Viewer で Run の設定とグラフを比較する
page_kind: サマリ
material: "[[docs/memo/0196_metrics_viewer_run_comparison_10rfd.md|0196_metrics_viewer_run_comparison_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/120_metrics_viewer.md|Metrics Viewer]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# Metrics Viewer で Run の設定とグラフを比較する

## 元の材料

[RFD0196　Metrics Viewer で Run の設定とグラフを比較する](../../../memo/0196_metrics_viewer_run_comparison_10rfd.md) blob:52bc077e11be28796a1dad7a3ac278a1015f7517

## 要点

- 選択した Run の設定を表に並べ、値が異なる行とキー名で絞り込めるようにする。
- Run とタグの一覧を検索できるようにし、必要な組み合わせを選びやすくする。
- タグ情報をグラフの近くで確認し、異なる Run の水準や傾向を共通の経験数で読めるようにする。
- 設定表は Run が保存した設定記録を基にし、画面上の選択と表示条件をそろえる。
- Plotly を同梱し、ネットワークにつながらない環境でグラフを利用できるようにする。

## 内容

### 問題

2026-11-21 の会話で、MetricsViewer に足りない機能と問題点を洗い出した。ユーザーはその中から、設定の比較、Run・tag の検索、グラフごとの統計、オフラインでの動作の 4 つを選び、課題と RFD の数を増やさないよう 1 件の課題（issue-20261121-201669）にまとめた。

### ゴールと非ゴール

新しい設定キー、メトリクス、TensorDict キーは足さない。

### 確定事項

設定表に出すキー。 `config_data` のキーをすべて出す。上書き層のキーや、使っていない名前空間のキーも、ほかのキーと同じに並べ、区別も印もしない。
TXT の読み方。 フレームワークの設定ファイル様式として、C++ の読み込み側 `Properties::Load` の行の規則で読む。C++ と違うのは、R1 の読めないバイト列と R5 の `$include`の 2 つだけである。書き出し側（`ToPropertiesString`）の形には合わせない。
値の比べ方。 値は文字列のまま比べる（`1,048,576` の桁区切りもそのまま）。
違う行。 選んだ Run の間で、値か有無が違うキーの行を違う行とする。キーが無いセルは、値（空文字を含む）と見分けられる形で出す。

### 実装契約

部品の種類の表に「タブ」の行を足す。ラベルは表示の名前（`Graphs`・`Config`）で、押した 1 つを `.active` と `aria-selected="true"` で示す。
C++ の `Properties::Load` が持つ、選択元の接頭辞の記録（`selection_source_prefixes`）は、設定表に要らないので写さない。C++ の規則が変わったら、Java の読み込みも同じ変更で直す。
互換のための仕組み。 残さない。CDN から読む経路や、CDN に届かないときの切り替えは持たない。見出しの統計を出す設定は足さない。C++ と Java の読み込みの規則は、R1 の UTF-8 として読めないバイト列の扱いと、R5 の `$include` の扱いのほかは同じ規則の 2 つの実装として持つ。
browser の部品をどのファイルに置くか（`metrics-viewer.js` に足すか、ファイルを分けるか）は、実装計画で決める。消すクラスは無い。

## 主要な概念

- [Metrics Viewer](../../concepts/apps/120_metrics_viewer.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
