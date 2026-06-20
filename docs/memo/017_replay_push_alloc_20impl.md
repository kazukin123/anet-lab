# ReplayBuffer Push 割当削減 実装メモ

## 概要
`docs/memo/017_replay_push_alloc_10prd.md` に基づき、`DefaultReplayBuffer::Push` の挙動を変えずに CPU 割当 churn を削る。

public `ReplayBuffer` / `ReplayBufferConfig` / config key は変更しない。新規フラグ、スレッド化、`SumTree` 最適化、`Sample` / `ExtractSamples` 変更は行わない。

## 主な変更
- `DefaultReplayBuffer` に再利用用の pending priority vector を追加し、`Push` 冒頭で `clear()`、必要時だけ `reserve()` する。
- 通常 write と truncated dummy の `0.0f`、`ProcessQueue` の初期優先度 `-1.0f` を順序どおり append し、`Push` 末尾で `metadata_mutex_` 配下の `UpdatePriorities` 1回にまとめる。
- `ProcessQueue` は `MarkValid` を従来どおり `metadata_mutex_` 配下で行い、優先度更新は呼ばず pending vector へ積む形へリファクタする。`UpdatePriorities` の外部 public 経路は現状維持する。
- `Push` の env ループ外に `TensorDict` スクラッチを置き、`batch.state.obs[b]` / `action_info[b]` / `batch.next_state.obs[b]` の新規 dict 構築をやめる。
- 匿名 namespace の helper で `kv.second[b]` view を `Set` 上書きし、`storage_->Push` / `PushTerminalDummy` の既存シグネチャへ渡す。
- `batch.action->GetInfo()` はコピーせず `const anet::TensorDict&` で参照する。`GetAction()` もループ前に取得して同じ tensor から `[b]` を読む。

## テスト
- `replay_buffer_test.cpp` に characterization test を追加し、固定 seed・固定 push 系列で `Sample().indices` の列が golden と一致することを `UNIFORM` / `PRIORITIZED` の両方で確認する。
- PER 側は sample 後に `UpdatePriorities(indices, positive priorities)` を挟み、初期優先度 `-1.0f`、無効化 `0.0f`、通常更新 `p>0` が同じ系列内に入るようにする。
- golden 値は実装編集前の現行コードから一度採取して固定 vector としてテストに埋め込む。以後は最適化後も sampled index 列が bit 一致することを回帰ガードにする。
- 既存 `[replay_buffer]`、`[replay_buffer][prefetch]`、並行 push/update test は維持し、production API をテスト用に広げない。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提
- `CONTEXT.md` 更新は不要。これはドメイン用語追加ではなく ReplayBuffer 内部最適化。
- ADR 追加も不要。ADR 0005 の `PrefetchingReplayBuffer` 順序契約を変更しない。
- 実 FPS / Tracy 比較はコード緑化後の別実測で行い、実装受け入れの主ガードは deterministic UT と既存テスト緑にする。
