# Actor network方針の既定値解決をAgentへ戻す

## 元の材料

[ADR 0013](../../../adr/0013-actor-network-resource-policy.md) blob:7e85430388bf120234851804411ab3f92680f733

## 要点

- Actor を複製できるかと、どの network を使うかは Agent の能力に依存する。
- 当初は Runner が未指定を渡し、Agent に既定値の解決を任せた。
- 後の ADR 0038 では ActorRequest を受け取り、カタログの clone_model が複製の要否を決める。
- Learner が更新する共有元の network は Agent が所有する。
- 単一 Actor の推論と同期のために作る private clone は Actor が所有する。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [325 Agent個別実装 MuZero系列](../../concepts/framework/325_muzero.md)
