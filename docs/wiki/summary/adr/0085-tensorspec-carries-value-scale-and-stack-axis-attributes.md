# TensorSpec に、値の尺度と stack 軸の 2 つの属性を持たせる

## 元の材料

[ADR 0085](../../../adr/0085-tensorspec-carries-value-scale-and-stack-axis-attributes.md) blob:8bf46b2994aac204368e0ce0df27d7995ce0819e

## 要点

- 観測の値の尺度と stack 軸の有無を TensorSpec の属性で宣言する。
- dtype や次元数から推測すると、one-hot や重ね済みデータを区別できない。
- Env など入力を作る側が宣言し、変換した部品が必要な属性を更新する。
- 尺度の指定は上下限の有無を表すものではない。
- 重ねるかどうかは実行時データの状態と分け、設定で選ぶ。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
