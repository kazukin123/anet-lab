# stack 軸の確保と、並びが成り立つことの確かめは、Forwarder の仕組みではなく使う側が受け持つ

## 元の材料

[ADR 0096](../../../adr/0096-stack-axis-and-chain-validity-are-ensured-by-the-caller-not-the-forwarder-mechanism.md) blob:44e58393f780807ebf070d20a0c218ce5bbb9a05

## 要点

- 並びは設定に書かれた順で Forwarder を組み立てる。
- FrameStack の位置や数と Normalize の個数を共通機構では制限しない。
- 重ねる対象に長さ1の軸が必要な場合は DefaultDQN が補う。
- 行動選択・Sample・可視化の出力 spec が一致することも DefaultDQN が確かめる。
- RL 側が必要とする不変条件を利用側に置き、汎用の変換の並べ方を縛らない。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
