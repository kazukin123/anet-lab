# V2 残件リスト

この文書は、現在の V2 `DefaultDQNAgent` / `TensorDict` 対応をいったんコミットした後に残る作業を整理するためのメモです。
IQN 対応や MuZero 本格実装で作業範囲が広がる前に、V2 の既知の暫定箇所を記録しておきます。

## 優先度高

- `auto_load_file` の所有を Agent から Trainer へ移す。
  - run / checkpoint の lifecycle 判断は Trainer が持つべき。
  - checkpoint 互換性チェックを追加する。
    - load 前に action count、observation spec、network config、quantile count、scaler 設定などの不一致を検出できる metadata を保存する。

## 可視化

- `probe.cpp` を整理する。
  - 複数箇所で default vector observation key 前提の暫定処理が残っている。
  - multi-key `TensorDict` observation でも使えるよう、probe の入力 key / 次元選択を明示できる形にする。
- V2 の `TensorDict` observation モデルに合わせて HeatMap 可視化を復旧。
  - 旧来の `TensorFunction(torch::Tensor)` API を残すのか、`TensorDictFunction` ベースへ寄せるのかを決める。
- `DefaultDQNAgent::GetTensorFunction` と `NetworkModel::GetTensorFunction` を V2 向けに正しく実装する。
- ReplayBuffer の可視化対応
  - V2 の `TensorDict` observation、`stack_keys`、sample extraction、PER priority data を前提に再設計する。
  - storage レベルの表示と sampled batch の表示は、必要なら別 view として扱う。
- metrics / image observer 経路の HeatMap 系 Observer 対応を追加する。
  - `observers.cpp` に残っている HeatMap 対応 TODO を処理する。
- `DefaultDQNAgent::GetTensorDictFunction`を見直す。
- `SweepHeatMapDialog::SweepHeatMapDialog()` を更新。
  - 現状の次元選択は暫定で、`ObsKeys::kVector` 前提になっている。
  - `StateSpec::obs_spec` から observation key と有効な次元を選べるようにする。

## Cleanup

- `#if 0` のデバッグ断片は `v2_work` 中はいったん残すが、mainline へ戻す前に削除または正式化。
