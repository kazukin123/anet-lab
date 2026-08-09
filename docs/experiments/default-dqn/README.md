# DefaultDQNAgent ハイパーパラメータ探索

DefaultDQNAgent 系の探索記録です。ReplayBuffer、PER、batch、replay ratio、target 更新など、Agent 側の共通機構を軸に整理します。

## Env 別記録

| Env | 記録 | 概要 |
|---|---|---|
| DropMerge | [DropMerge 探索記録](dropmerge/README.md) | 長期 Run における batch size と replay ratio を中心とした探索 |

## Env をまたぐ際の扱い

Agent 側の設定名や更新量の式は共有できますが、最適値は Env に依存します。
特に次の要素が異なる結果は、別 Env へそのまま一般化しません。

- 報酬密度と遅延
- エピソード長
- 行動の時間的意味
- ReplayBuffer 内の経験分布
- 局所解と探索方策

Env をまたぐ知見として昇格する場合は、複数 Env で再現した事実と、Env 固有の推測を分けて記録します。
