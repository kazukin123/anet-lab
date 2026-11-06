# Replay ring stack margin 実装メモ

## 概要

ring 折り返し後、上書き済み frame を必要とする transition を sample 候補から除外する。`ValidIndexManager` の未来側確定区間を ready range とし、sample 列挙時だけ history margin を適用する。public `ReplayBuffer` API、設定キー、metric key、extractor の挙動は変更しない。

## 主な変更

- `ValidIndexManager` の未来側確定区間を `LogicalReadyRange` / `GetLogicalReadyRange()` へ改名し、`ForEachSampleableIndex()` で `retained_start > 0 ? stack_count - 1 : 0` の history margin を下限へ加える。
- `GetValidIndices1D()` と `GetSampleableCount()` は同じ sampleable range を使い、wrap 前、`stack_count == 1`、dummy 除外、物理 index 昇順の挙動を維持する。
- `InitialPriorityCompleter` は `IsLogicalReady()`、eviction 統計は `IsOverwritingReady()` を使う。後者から `stack_count` 引数を削除し、metric key は変更しない。
- 未使用の `GetValidCount()` を削除する。
- 構築時の lane 最小 capacity を `max(1, n_step) + 1 + (stack_count - 1)` とし、不足時は既存の `ANET_SYSTEM_ERROR` で fail-fast する。
- `docs/design/150_replay_buffer.jp.md` を ready/sampleable 分離、padding 境界、eviction 統計の近似に合わせる。既存の PRD、ADR 0024、`CONTEXT.md` は整合済みのため変更しない。

## テスト

- Public interface / surface: `ReplayBuffer::Size()`、`Sample()`、`GetTensorVector()`、`GetScalar()`、`CreateReplayBuffer()` の構築時 validation。
- 優先 behavior: wrap 後の unsafe stack 除外、PER と unroll の同一 sampleable 集合、lane 別 margin、dummy 除外、eviction 統計、capacity 境界、`stack_count == 1` と初期 padding の非退行。
- TDD 順序: 既存 RED `ReplayBuffer excludes wrapped samples whose frame stack would read overwritten frames` を tracer bullet として最小実装で GREEN にする。次に PER、unroll を確認し、capacity 境界を RED -> GREEN にする。保持契約のテストを個別に追加し、GREEN 後に internal rename と dead code 削除を行う。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]~[episode_start]" -r compact
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]" -r compact
git diff --check
```

## 前提

- extractor、generation assert、PER sampler、保存形式、設定契約は変更しない。
- unroll を含む完全な capacity 下限再設計と `episode_start without done` の契約裁定は範囲外とする。
- baseline は ReplayBuffer 73件中68件成功で、PRD対象3件と範囲外2件の計5件が失敗している。
- unrelated な未コミット変更を保持する。
