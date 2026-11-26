# forward の持ち帰りは鍵要求型の capture 1 つに統一し、module の診断も同じ器で運ぶ

## 元の材料

[ADR 0055](../../../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) blob:ea94e81b3bafbb2aa9bced226636ce21f506b085

## 要点

- Forward の途中結果と module 診断は、呼び出し側が要求キーを渡す共通の capture で受け取る。
- 購読がなければ attention 診断などの不要な計算を避ける。
- Network の出口で、要求したキーが得られたことを確認する。
- nn_trace も後に同じ仕組みへ統合され、専用 TraceCallback を廃止した。
- 共通の ForwardCapture への改名・移設は RFD0161 が定め、ADR 0084 は Forwarder の並びを定めた。

## 内容

### 測定要求

SDPA が返さない注意重みからエントロピーを測るには、測りたい時にだけ途中値を計算する必要がある。要求のない forward で常時呼ぶコールバックや、module ごとの診断引数を公開面に増やす形は採らなかった。

### 出口の検証

分岐名と module のキーが衝突しないよう名前空間を分け、要求に届かなければエラーにする。部分 forward と実バッチの両方が同じ受け皿を使える。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
