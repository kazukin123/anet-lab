# image metrics v2 Phase2 実装計画

## Summary

Phase2 では agent source の ReplayBuffer 可視化を復旧し、V1 で生きていた `ReplayBuffer` key 群も戻す。既存 `metrics_image.txt` の `image.phm.per-prio` と `image.thg.per-prio` を無改修で動かす。

## Key Changes

- `DefaultReplayBuffer::GetTensorVector()` / `GetTensor()` / `GetScalar()` を実装する。
  - `STATE_OBS`: sampleable valid rows の現在 observation を `ToUnifiedObservation()` で `[rows, D]` 化。
  - `NEXT_STATE_OBS`: `actual_n_steps_` から n-step bootstrap observation を復元して `[rows, D]` 化。
  - `ACTION`: valid rows の action。離散 action は probe index 互換のため `[rows, 1]`。
  - `REWARD`: V1 互換で raw reward ではなく `target_returns_` を返す。
  - `NEXT_STATE_TERMINAL`: `terminals_`。
  - `N_STEP`: `actual_n_steps_`。
- PER key も V1 互換で復旧する。
  - `PER_TOTAL`: `SumTree` total。
  - `PER_VALUES`: priority value。`GetTensor(index)` で物理 index の値、`GetTensorVector()` で valid rows の列。
  - `PER_DIST`: 名前に反して正規化確率へ変えず、V1 同様 priority 値列を返す。
- 公開 API は増やさない。
  - `ReplayPriorityController` public interface は広げない。
  - `.cpp` 内で `PrioritizedSampler` に read-only helper を足し、`DefaultReplayBuffer` から concrete type を確認して読む。
- `ReplayExperienceStorage` は raw tensor 所有者として維持し、valid row 抽出・n-step next 復元・PER row alignment は `DefaultReplayBuffer` 側で行う。

## Test Plan

- `core/anet-core/src/replay_buffer_test.cpp` に回帰テストを追加する。
  - `STATE_OBS` / `NEXT_STATE_OBS` が sampleable valid rows と同じ件数で返る。
  - `NEXT_STATE_OBS` が n-step 先の observation を返す。
  - `ACTION` / `REWARD` / `NEXT_STATE_TERMINAL` / `N_STEP` が V1 互換 key で取得できる。
  - `REWARD` は `target_returns_` であることを明示的に検証する。
  - `PER_TOTAL` / `PER_VALUES` / `PER_DIST` が `UpdatePriorities()` 後の adjusted priority を返す。
  - uniform sampler では PER key が `std::nullopt` になる。
- 検証コマンド:
  ```powershell
  cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
  core\anet-core\bin\Debug\anet-core-test.exe
  ```
- 画像 smoke:
  - `image.phm.per-prio` の PNG が生成されること。
  - `image.thg.per-prio` の PNG が生成されること。

## Assumptions

- `REWARD` の名前は変更せず、V1 互換の `target_returns_` として扱う。
- `storage.*` key は sampled-batch view ではなく sampleable storage-level view とする。
- Phase3 の `TensorDictFunction` 統一、Phase4 の UI sweep 復旧はこの Phase2 では触らない。
