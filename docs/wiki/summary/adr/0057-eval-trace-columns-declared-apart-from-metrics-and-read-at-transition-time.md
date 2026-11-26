# Evaluation Trace の列はメトリクスの定義と分けて番号付きで宣言し、遷移の時点に合う target だけを読む

## 元の材料

[ADR 0057](../../../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) blob:82b9281679cb0b43f6954b30db9732cad73d14c9

## 要点

- Evaluation Trace は1件を1遷移とする。
- パネル側の設定に番号付きで1行1列を宣言する。
- 記録用 trace のタグに依存せず、記録を設定していなくても表示できる。
- 行動や終了結果など、その遷移の時点が決まる target だけを許す。
- 学習中に独立に変化する Agent 最新値などは構築時に拒否する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
