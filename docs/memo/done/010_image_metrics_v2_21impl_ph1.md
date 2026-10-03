# image metrics v2 Phase1 実装計画

## Summary

Phase1 では experience source の `HeatMap visit-maxq` を復旧する。既存 `metrics_image.txt` の `experience.next_state.obs` と `action.max_q` を無改修で動かし、`obs.<subkey>` 文法は `vector` を中心に将来拡張できる形で追加する。

## Key Changes

- `core/anet-core/src/rl.cpp` の `BatchExperience::GetTensor()` に observation key 解決 helper を追加する。
  - `experience.state.obs` / `experience.next_state.obs` は従来どおり `ToUnifiedObservation()` を返す。
  - `experience.state.obs.vector` / `experience.next_state.obs.vector` は該当 `TensorDict` キーを直接返す。
  - 未知の `obs.<subkey>` は `ANET_SYSTEM_ERROR` で key と subkey を含めて失敗させる。
  - `action.max_q` など既存 `action.` aux 経路は維持する。
- `core/anet-core/src/probe.cpp` の probe 側を整理する。
  - `BatchExperienceStateProbe` は `BatchExperience::GetTensor()` 経由の unified observation 前提としてコメントを更新する。
  - `BatchExperienceVectorProbe` は `GetTensorVector()` 委譲を維持し、constructor の min/max/label 推定を `obs.<subkey>` でも扱えるようにする。
  - `MakeExperienceProbe()` は `experience.state.obs.*` / `experience.next_state.obs.*` でも `StateSpec` を渡す。

## Public Interface

- C++ の公開クラスやヘッダ API は増やさない。
- data key 文法だけを拡張する。
  - 既存: `experience.next_state.obs`
  - 追加: `experience.next_state.obs.vector`
  - 既存 config の `index = 0/1` は unified observation の vector 先頭レイアウトとして維持する。

## Test Plan

- 新規または小さな既存テストに `BatchExperience` / `BatchExperienceVectorProbe` の回帰テストを追加する。
  - multi-key observation で無接尾 `experience.next_state.obs` が `vector` を先頭に unified する。
  - `experience.next_state.obs.vector` が vector tensor を返す。
  - `experience.next_state.obs.unknown` が throw する。
  - `action.max_q` aux tensor が取得できる。
  - `BatchExperienceVectorProbe` が `experience.next_state.obs` の index 0/1 と `action.max_q` を `UpdateEvent` 経由で取り出せる。

- 検証コマンド:
  ```powershell
  cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
  core\anet-core\bin\Debug\anet-core-test.exe
  ```

- 画像 smoke は tracked config を恒久変更せず、現在 `_main.txt` が include している DropMerge で `image.phm.visit-maxq` を CLI override で有効化して確認する。出力先 `C:\dev\anet-lab\apps\runner\runs\...` に visit-maxq の PNG が生成されれば Phase1 の実経路は通っている。

## Assumptions

- `obs.<subkey>` は Phase1 では `vector` を主対象にするが、実装は存在する TensorDict key を返せる汎用 helper にする。
- `grid` sweep や ReplayBuffer 可視化は Phase2 以降で扱う。
- `ProfileRange` は既存 `BatchExperienceVectorProbe::GetVector()` の計測を維持し、Phase1 では新しい重い処理境界を増やさない。
