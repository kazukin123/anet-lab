---
title: 鏡写し eval インスタンスは dormant タグの trace を後から購読し、trace 行は時刻を持つ
page_kind: サマリ
material: "[[docs/adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md|0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-28
---
# 鏡写し eval インスタンスは dormant タグの trace を後から購読し、trace 行は時刻を持つ

## 元の材料

[ADR 0047](../../../adr/0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) blob:6ac7e98136726a2af7ed031df459473c8c52406e

## 要点

- EvalPanel は設定タグを参照して後から作る別インスタンスで、起動時の購読だけでは trace を受け取れない。
- 定期実行しないタグでも trace 宣言を保持し、後から作った評価に結び付ける。
- 休止タグの scalar は保持せず、通知レベルは後に WARN から INFO へ改めた。
- trace に timestamp を加え、同じ step でも表示時点を識別できる。
- 設定上の on_demand 印や起動時の全インスタンス生成は要求しない。

## 内容

### 後から作る評価

鏡写しインスタンスの生成を起動時に固定すると既存の生成経路が大きく変わる。休止しているタグを指す trace 宣言を保持し、タグが一致する Runner を作ったときに結び付ければよい。

### 記録の意味

利用者が任意に始めて止める評価を通常の scalar 統計に混ぜない。trace は実際の1ゲームに対応する。timestamp は時刻に依存するので、同 seed の回帰比較ではその違いを区別する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
