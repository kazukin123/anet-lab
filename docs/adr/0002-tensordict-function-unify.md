# 観測 sweep / 可視化の NN 呼び出しを TensorDictFunction に一本化する

V2 で Observation を multi-key `TensorDict` 化したのに伴い、可視化の状態スイープが NN を呼ぶ経路に旧 `TensorFunction(torch::Tensor)` と新 `TensorDictFunction(TensorDict)` の 2 API が併存し、前者は全 Agent で nullopt スタブ化して機能停止していた。利用箇所が sweep 経路のみに限定されることを確認した上で、`TensorDictFunction` に一本化し、`TensorFunctionProvider` と各 `GetTensorFunction` を撤去する。Tensor 専用 API は multi-key observation を運べず将来の grid チャネル sweep の障害になるため、薄いアダプタで延命するより撤去して入口を一つに保つ方を選んだ。

## Considered Options

- **GetTensorFunction を adapter 実装**: `Tensor → {"vector": t} → TensorDictFunction` に委譲する薄い層で延命。sweep 経路は無改修で最小だが、Tensor 専用 API が残り multi-key を運べず、将来結局統一が必要になる（拡張の door が閉じる）。
- **TensorDictFunction に一本化（採用）**: sweep が選択された vector-type 観測キーの格子と、その他観測キーの固定値を含む `TensorDict` を組み立てて呼ぶ。触るファイルは多いが機械的で、API が一本化され multi-key sweep の自然な拡張口になる。

## Consequences

- `StateSweepProcessor` / `SweepedHeatMap(Observer)` / `HeatMapPanel` / `image.cpp` の sweep factory が `TensorDict` 入力へ変わる。
- `NetworkModel::GetTensorDictFunction` に `policy-net.forward.*` / `target-net.*` の head routing を実装する必要がある（従来 `*.conv2d` のみ）。
- 実装手順は `docs/memo/010_image_metrics_v2_10prd.md`。
