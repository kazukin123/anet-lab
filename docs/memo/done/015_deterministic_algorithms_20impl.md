# `backend.deterministic_algorithms` 実装メモ

## 概要

`docs/memo/015_deterministic_algorithms_10prd.md` と `docs/adr/0006-deterministic-algorithms.md` に従い、学習 Run の同 seed 再現性を確保するため、backend config から `torch::globalContext().setDeterministicAlgorithms(...)` を一箇所で適用する。

`setSDPUse*` や `CUBLAS_WORKSPACE_CONFIG` の実コード投入は行わず、必要な背景は `InitRL` のコメントとして残す。`CONTEXT.md` と ADR は更新しない。

## 主な変更

- `BackendConfig` に `deterministic_algorithms = true` と `deterministic_warn_only = false` を追加し、既存 backend bool と同じく `ANET_READ_CONFIG` で読む。
- `InitRL` で `ctx.setBenchmarkCuDNN(...)` の直後、`MetricsLogger::Instance()->Log(backend_config)` の前に `ctx.setDeterministicAlgorithms(...)` を追加する。
- 同じ挿入位置に、SDPA backward 非決定、`warn_only=false` の意味、SDPA backend 固定を採らない理由、将来の `CUBLAS_WORKSPACE_CONFIG` 対応方針を説明する日本語コメントを残す。
- `apps/runner/config/common.txt` の `backend.cudnn_*` 付近に `backend.deterministic_algorithms` と `backend.deterministic_warn_only` を追加する。

## 公開インターフェース

- `backend.deterministic_algorithms`: bool、既定 `true`。全 ATen op の決定論ポリシーを有効化する。
- `backend.deterministic_warn_only`: bool、既定 `false`。決定版がない op を throw ではなく警告にする退避用で、再現性は保証しない。
- `BackendConfig::ToJson()` / `GetConfigData()` / `MetricsLogger::Log(backend_config)` には、既存 `ANET_READ_CONFIG` 経路で自動的に新フィールドが載る。

## テスト

- 新規単体テストは追加しない。既存 backend フラグと同様、config 読み込み経路と起動時ログ出力は `ANET_READ_CONFIG` の既存機構に乗せる。
- 再現性確認はユーザー実施扱いにする: `backend.deterministic_algorithms=true` で同 seed 2 run の train 側 loss/weight が一致し、`false` で従来挙動へ戻ることを確認する。

## 検証

```powershell
git diff --check
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
```

## 前提

- `torch/torch.h` 経由で `at::Context::setDeterministicAlgorithms(bool, bool)` が利用できるという PRD の確認済み前提を採用する。
- 現在の未コミット変更はユーザー変更として保持し、実装対象以外には触れない。
