# メトリクスから action_info の値を読む境界

## 元の材料

[RFD0178　メトリクスから action_info の値を読む境界](../../../memo/done/0178_metrics_action_info_dynamic_cast_10rfd.md) blob:3832c554585b6c790327e1a1f2437d6c69b83963

## 要点

- 課題として取り上げることは採用されたが、action_info の具体的な契約変更と実装方式は起草時点で未決だった。
- action_info のメトリクス取得が派生型への実行時変換に依存していた。
- 変換できないときは指定した step 軸の位置と NaN を返す挙動がある。
- scalar を返せるかを型の実行時判別に依存せず表せる境界を検討する。
- 公開のメトリクス宣言と通知から、未知キー、値未成立、不正イベントの扱いを確かめる。
- 他の Observer の整理や履歴の書き換えには広げず、使う interface は着手時の設計と照合する。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
