---
title: MetricsLogger シングルトン・ライフサイクル堅牢化 — null-safe 静的ログ API と Reset 順序 PRD
page_kind: サマリ
material: "[[docs/memo/frozen/0026_metricslog_lifecycle_10rfd.md|0026_metricslog_lifecycle_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-26
---
# MetricsLogger シングルトン・ライフサイクル堅牢化 — null-safe 静的ログ API と Reset 順序 PRD

## 元の材料

[MetricsLogger シングルトン・ライフサイクル堅牢化 — null-safe 静的ログ API と Reset 順序 PRD](../../../memo/frozen/0026_metricslog_lifecycle_10rfd.md) blob:da47c27a8639694e127771b5a48807df392c3056

## 要点

- 終了処理で MetricsLogger を Reset した後、背景の評価 Observer がログを出そうとして落ちる事例を扱う。
- 評価はバックグラウンドでエピソード終端まで動くため、Runner の終了とログ呼び出しの寿命がずれていた。
- 静的なログ API が logger の有無を確かめ、呼び出し中の寿命を確保する案を記録した。
- Reset 前に背景 Observer を待つ対応は、原本の追記では RFD0076 で実装済みとされる。
- 静的 API の案は凍結中で、待ち合わせの外から Reset 後のログが実害を生んだときに再開する。

## 内容

### 終了中の競合

背景で動く評価エピソードは Runner 終了後もしばらく動き得る。Logger を取得した時点でインスタンスがなければ、メンバー関数呼び出しは成立しない。原本の再調査では、本番の Reset はアプリ終了時にあり、Optuna が同じプロセスの Run を切り替えるという当初の前提は撤回された。

### 修正案と再開条件

RFD0076 は Logger を Reset する前に背景処理の完了を待つ。インスタンスがないときの静的 API の扱いは、この待ち合わせを通らない別経路で終了後のログが障害となった場合に再検討する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
