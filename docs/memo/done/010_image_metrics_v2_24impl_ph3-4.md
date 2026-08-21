# image metrics v2 Phase3/4 実装計画

## Summary

`SweepStateHeatMap` と runner の対話 Sweep HeatMap を同時に `TensorDictFunction` 経路へ移行し、旧 `TensorFunction` API を撤去する。`sweep_obs.obs_key` と `sweep_obs.output_key` を正式追加し、sweep 入力 key と描画対象 tensor を明示できる仕様へ更新する。

## Interface Changes

- `sweep_obs.obs_key` を追加する。
  - 未指定時の既定は `vector`。
  - `SpaceType::Vector` の Observation key だけ許可する。
  - `vector` が無くても他の vector-type key を metrics 側で推定しない。明示指定が必要。
- `sweep_obs.output_key` を追加する。
  - 未指定時の既定は `q`。
  - `network_key` は `policy-net.` / `target-net.` + `Network::GetTensorDictFunction()` key を表す。
  - `output_key` は NN 戻り値 `TensorDict` から extractor に渡す tensor key を表す。
- 旧 `TensorFunction` / `TensorFunctionProvider` / `GetTensorFunction()` は撤去する。
- `StateSweepProcessor` / `ISweepInputGenerator` は `TensorDict` を生成する。

## Implementation Changes

- sweep 入力生成を拡張する。
  - 選択 `obs_key` の flatten tensor を grid 化し、X/Y index を上書きする。
  - `x_index == y_index` は許容し、現行順序に近い `Y` 値で上書きする。
  - 選択 key 以外の observation key は spec shape/dtype に合わせた zero tensor を入れる。
  - invalid `obs_key`、non-vector key、flatten dim `< 1`、index out of range は明示エラーにする。

- network routing を一本化する。
  - `NetworkModel::GetTensorDictFunction()` は `policy-net.` / `target-net.` prefix を外して対象 network へ委譲する。
  - `DefaultDQNAgent` と `RainbowAgent` は同じ方針で `GetTensorDictFunction()` を提供する。
  - DefaultDQN の lock / obs_norm / synthetic stack 展開は維持し、sweep 入力での意味をコメント化する。

- observer / extractor 経路を dict 化する。
  - `SweepedHeatMapObserver` は `TensorDictFunction` を呼ぶ。
  - 戻り `TensorDict` から `output_key` を取得し、既存 extractor に渡す。
  - missing `output_key` は Observer/Panel 作成または render 時に明示エラーにする。Dialog では Agent 実検証しない。

- UI を新仕様に合わせる。
  - Dialog は `obs_key`、network side (`policy-net` / `target-net`)、function key (`forward`, `forward.q`, `forward.dist`, `forward.v`, `forward.a`)、`output_key` を分離する。
  - 保存時に network side + function key を結合して `network_key` にする。
  - function key 変更時は `output_key` を suffix から推定する。
    - `forward` / `forward.q` -> `q`
    - `forward.dist` -> `q_dist`
    - `forward.v` -> `v`
    - `forward.a` -> `a`
  - vector-type obs key 候補が 0 件なら Dialog 内で `OK` を無効化する。
  - obs key 変更時に X/Y range と tag を更新する。
  - tracked `metrics_image.txt` は新設定で既存 qmax 動作を保つ形へ更新する。

## Test Plan

- Unit tests:
  - `StateSweepProcessor` が指定 `obs_key` を sweep し、他 key を zero-fill する。
  - `x_index == y_index` が許可され、Y 上書き順序になる。
  - invalid obs key、non-vector obs key、index out of range、missing output key が明示失敗する。
  - `NetworkModel::GetTensorDictFunction()` が `policy-net.forward` / `policy-net.forward.dist` / target 系を prefix strip して解決する。
  - `SweepedHeatMapObserver` が `output_key` の tensor を extractor に渡す。

- Verification:
  - `anet-core-test` Debug build と実行。
  - `rg "GetTensorFunction|TensorFunctionProvider"` で残骸ゼロ。
  - LunarLander で `image.shm.qmax` PNG 出力。
  - DropMerge で `obs_key=vector` + zero-filled grid により起動 crash なし。
  - runner GUI で vector-type obs key が無い env の Dialog `OK` 無効化を確認する。

## Assumptions

- grid / image sweep は今回やらない。
- 外部の旧 config 互換パースは入れない。tracked config は新仕様へ更新する。
- Dialog は `output_key` の実在性を Agent に問い合わせない。実際の不一致は Observer/Panel 作成または render 経路で明示失敗させる。
