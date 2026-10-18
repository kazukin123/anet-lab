# net.config_profile パラメータ補間 実装メモ

## 概要

`net.config_profile.[name]` を `NetworkConfig` に読み込み、branch 内の block 出現順に沿って `@group` マーカー付き config 値を `CreateModule` 前に実数へ展開する。初期実装は `type=linear` のみ対応し、`start` 既定値は `0.0`、`end` は必須とする。マーカー無し branch は従来と同じ `block_cfg.config_data` をそのまま渡す経路を維持する。

## 主な変更

- `core/anet-core/include/anet/nn.hpp`
  - `ConfigProfileConfig` を追加し、`NetworkConfig` に `config_profiles` を持たせる。
- `core/anet-core/src/nn_impl.cpp`
  - `ReadConfigProfileConfig` を追加し、`net.config_profile.[name].type|start|end` を読む。
  - `ConfigProfile` interface と `LinearConfigProfile` を追加し、profile config 検証と値生成を型別実装へ分離する。
  - `NetworkConfig::ToJson()` に `config_profiles` を含める。
  - `NetworkStructBuilder::Build` を branch 内 2 パスにし、出現順と `(*N)` 展開後の instance 単位で marker を集計して補間値を上書きする。
  - 未定義 group、`end` 未指定、旧 `min/max`、未知 `type` は `ANET_SYSTEM_ERROR` で fail-fast する。
  - 定義済みだが未使用の profile は `LOG::warn()` で通知する。
  - `NetworkBodyBuilder::Build` で branch ごとの使用 group を集約し、同一 group の branch 跨ぎ使用を `ANET_SYSTEM_ERROR` にする。
- `core/anet-core/src/nn_test.cpp`
  - `[nn][config_profile]` テストを追加し、linear 補間、stage 跨ぎ出現順、`(*N)`、N=1、`ToJson()` の `start/end` 出力、未定義 group、`end` 未指定、旧 `min/max`、未知 `type`、branch 跨ぎ重複、マーカー無し branch の回帰を確認する。
- `apps/runner/config/nn.txt`
  - 既存 run に影響しないコメントアウト済みの `start/end` 版 `config_profile` 最小サンプルを追加する。

## テスト

- `Dropout` block の `dropout_rate` を `@group` にして `NetworkStructBuilder::Build` から `GetCurrentConfigData()` を確認する。
- 18 instance の `A(*3) > B(*3) > C(*9) > D(*3)` で `linspace(0, 0.1, 18)` と一致することを検査する。
- `net.config_profile.[unused]` は build を通し、未使用 warn はログ目視対象に留める。
- エラー系は `CHECK_THROWS` で、`end` 未指定、旧 `min/max`、未知 `type` などが silent fallback しないことを確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[config_profile]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check -- core/anet-core/include/anet/nn.hpp core/anet-core/src/nn_impl.cpp core/anet-core/src/nn_test.cpp apps/runner/config/nn.txt docs/memo/029_config_profile_param_interp_20impl.md
```

## 前提

- branch 跨ぎ補間は今回実装しない。同じ group が複数 branch に現れた場合は明示的に落とす。
- `@group` は `ConfigData` 値の先頭一致で検出し、空 group 名は設定ミスとして落とす。
- 補間対象フィールドの型判定は factory 側に任せ、展開機構は marker 文字列を実数文字列へ置換するだけにする。
- `CONTEXT.md` と ADR は更新しない。今回の内容は強化学習ドメイン用語の追加ではなく、既存 NN DSL の局所的な構成機能追加であり、戻しにくい広域アーキテクチャ判断でもない。
