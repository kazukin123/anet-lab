# Replay は報酬を成分ごとに記録し、成分ごとの割引率で収益を作る。成分の意味は持たない

## 元の材料

[ADR 0073](../../../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md) blob:291226406e3d19669cf040592de59c923087ba74

## 要点

- Replay は1 step の報酬を複数成分で受け取り、構築時に成分数を決める。
- 各成分の割引率で n-step 収益と初期優先度の見積もりの入力を作る。
- Replay は成分の意味、合成時の重み、終端マスクを決めない。
- 定数の割引率では Push 時、schedule では抽選時の計算を各成分に適用する。
- 複数成分と skip 用の可変長窓を同時に指定した構成はエラーにする。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [報酬尺度と割引率](../../concepts/methods/reward_scale_discount.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
