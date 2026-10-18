# VideoLogger 堅牢化 実装メモ

## 概要

`VideoLogger` が ffmpeg の起動失敗・異常終了を stream の状態だけで判断しないようにし、異常時は cmd、path、exit code、stderr を含む `ANET_SYSTEM_ERROR` で停止する。`metrics_logger.video_codec` の既定値は `auto` に変更し、`auto` は 160x64 以上かつ偶数サイズなら `h264_nvenc`、それ以外なら `libx264` を選ぶ。明示 `h264_nvenc` は非対応サイズで fatal にし、勝手に fallback しない。

## 主な変更

- `core/anet-core/include/anet/metrics_logger.hpp` に codec 判定用の `detail` helper を宣言し、`MetricsLoggerConfig::video_codec` の既定値を `"auto"` に変更する。
- `core/anet-core/src/metrics_logger.cpp` で `IsNvencEligibleVideoSize` と codec 選択を実装し、`VideoLogger` の起動処理を `LaunchFfmpeg` / `DiedAtStartup` / `OnProcessTerminated` に分ける。
- `VideoLogger` に PID、ffmpeg 死亡フラグ、exit code、stderr、launch cmd を保持させ、`wxProcess::OnTerminate` と `wxProcess::Exists` による stream 非依存の生存確認を追加する。
- `WriteFrame` と `Close` は同じ `write_mutex_` 下で `stream_` と `process_` を扱い、pipe failure は握り潰さず `ANET_SYSTEM_ERROR` にする。
- Windows の `wxPipeOutputStream` は非ブロッキング pipe なので、`WriteFrame` は大きい frame を 4KB chunk で書き込み、ffmpeg が生存中の一時的な 0 byte write は短く retry する。
- `auto` が `h264_nvenc` を選んだ後に起動直後死亡した場合だけ `libx264` で再起動し、それも失敗したら fatal にする。

## テスト

- `core/anet-core/src/metrics_logger_test.cpp` に `IsNvencEligibleVideoSize` の境界テストを追加する。
- `ResolveVideoCodec` の `auto` 小サイズ・大サイズ、明示 `h264_nvenc` 小サイズ fatal、`libx264` passthrough、任意 codec passthrough を検証する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- `SweepHeatMapPanel` の 128x128 生成自体は変更しない。
- padding mode は追加しない。
- `CONTEXT.md` はドメイン用語集なので、ffmpeg/codec の実装詳細は追記しない。
- 明示 `h264_nvenc` の意図を尊重し、非対応サイズで `auto` や `libx264` へ暗黙 fallback しない。
