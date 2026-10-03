# image metrics v2 Phase2.5 実装メモ

## Summary

Phase2 で復旧した ReplayBuffer 可視化 accessor は動作したが、storage 系 key でも学習 minibatch 用の `ExperienceSampleExtractor::ExtractSamples` を全 valid row に対して呼んでいたため、DropMerge の大容量 ReplayBuffer で HeatMap tick が重くなっていた。

Phase2.5 では可視化用 storage view を学習 sample view から切り離し、ReplayBuffer の物理 storage を直接 gather する経路に置き換える。

## Key Changes

- `DefaultReplayBuffer::GetTensorVector()` の storage 系 key は `ExtractSamples()` を呼ばない。
  - valid row は `env * capacity_per_env + time` の flat index として扱う。
  - storage tensor `[num_envs, capacity_per_env, ...]` を `[num_envs * capacity_per_env, ...]` に reshape し、`index_select(0, valid_indices)` で直接 gather する。
  - `STATE_OBS` / `NEXT_STATE_OBS` は V1 互換の単一フレーム view とし、`stack_count > 1` でも stacked sample を作らない。
  - `REWARD` は Phase2 と同じく V1 互換名として `target_returns_` を返す。
- `NEXT_STATE_OBS` は `actual_n_steps_` を valid row で gather し、`(time + actual_n_step) % capacity_per_env` から次状態の物理 index を作る。
- observation subkey を追加する。
  - `replaybuffer.storage.state.vector`
  - `replaybuffer.storage.next_state.vector`
  - subkey なしは従来どおり unified observation を返す。
  - 存在しない subkey は `ANET_SYSTEM_ERROR` で失敗させる。
- `DefaultReplayBuffer` private state に小さな versioned cache を追加した。
  - cache key は `key` と `index`。
  - `Push()` で storage/priority version を進める。
  - `UpdatePriorities()` で priority version を進める。
  - cache hit 時は `GetValidIndices1D()` の再計算も避ける。
- `apps/runner/config/metrics_image.txt` の per-prio heatmap x/y probe key を `replaybuffer.storage.next_state.vector` に変更した。
  - vector-only 環境では意味を維持する。
  - DropMerge のような vector+grid 環境では grid gather/concat を避ける。

## Tests

- 既存 Phase2 の ReplayBuffer 可視化テストを維持。
- 追加テスト:
  - `stack_count = 4` でも visualization accessor は `[N, D]` の単一フレームを返す。
  - 通常 sample 経路は引き続き stacked observation を返す。
  - `.vector` subkey が指定 obs key のみを返す。
  - `.action_mask` subkey が unified observation に含まれない tensor を直接返す。
  - unknown subkey は例外になる。

## Verification

以下を確認済み。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- `anet-core-test.exe "[replay_buffer]"`: 16 test cases / 521 assertions passed.
- `anet-core-test.exe`: 59 test cases / 691 assertions passed.
- `git diff --check`: whitespace error なし。
