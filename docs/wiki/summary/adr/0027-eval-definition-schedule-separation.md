---
title: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する
page_kind: サマリ
material: "[[docs/adr/0027-eval-definition-schedule-separation.md|0027-eval-definition-schedule-separation]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する

## 元の材料

[ADR 0027](../../../adr/0027-eval-definition-schedule-separation.md) blob:f4c9b12f7e6636c3fa895d6c1dba8f0c398e11f0

## 要点

- 評価タグの定義だけでバックグラウンド評価が始まり、画面専用のタグにも余分な評価が起動していた。
- 評価の内容は train.eval、定期実行は同じタグ名の train.eval_schedule に分ける。
- 実行する評価を列挙するのはスケジュール側とし、画面は評価定義だけを利用できる。
- 同じタグに複数の周期を割り当てる間接参照は導入しない。
- 後の改訂では休止タグへの参照通知を INFO とし、旧位置の設定は汎用の未読キー検出で扱う。

## 内容

### 定義だけの評価

設定値が1つでもあるタグを定期実行していたため、画面専用の設定でも背景 Env が起動した。interval を定義に必須化するだけでは、画面が使わない設定を書かされる。

### 設定の移行

周期と background 指定をスケジュール側へ移し、残った定義は EvalPanel からそのまま利用できる。休止タグの通知と未読設定の検出は後の ADR 0071 に従う。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [316 設定管理](../../concepts/framework/316_config.md)
