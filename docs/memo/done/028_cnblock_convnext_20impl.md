# CNBlock（ConvNeXt v1）実装メモ

## 概要

`028_cnblock_convnext_10prd.md` に従い、ConvNeXt-Tiny 相当の `CNBlock`、NCHW 用 `LayerNorm2d`、ImageCls 用 `ConvNeXtT` branch を追加する。

現 checkout では `WeightInitConfig.mode` の文字列化と `net.config_profile` は実装済みだが、`trunc_normal` は未実装のため今回追加する。`net.config_profile` は現行実装に合わせ、PRD 中の `min/max` ではなく `start/end` を使う。

## 主な変更

- `WeightInitConfig` に `trunc_std=0.02`、`trunc_a=-2.0`、`trunc_b=2.0` を追加し、`mode="trunc_normal"` を erfinv ベースの切断正規分布初期化として実装する。
- 既存 `LayerNorm` に `eps` を追加し、未指定時は従来どおり `1e-5` を維持する。
- 新規 `LayerNorm2d` と `CNBlock` を `NetworkModuleRepository` に登録する。
- `CNBlock` は depthwise を `torch::nn::Conv2d(groups=channels)` で自己完結させ、CPU/CUDA とも通常の `Conv2d` を使う（device 分岐なし）。
  - 実装中、Debug ビルドで CPU の grouped Conv2d が oneDNN の thread 検証 assert（`nthr_ == nthr`, `dnnl_thread.hpp`）で落ちたため、当初は CPU 専用の手動 depthwise（slice 加算）/ pointwise（matmul）実装を入れていた。
  - 検証の結果、この assert は **Debug 版 libtorch 固有のデバッグ assert**（PyTorch issue #104421 — Windows の Debug では forward で assert するが Release では正常）と判明。本プロジェクトは Debug ビルド時のみ `libtorch/debug`、RelWithDebInfo/Release 時は `libtorch/release` をリンクする（`CMakeLists.txt`）ため、**release 版 libtorch + GPU 実行の本番では発生しない**。grouped conv 単体は Debug でも落ちず、CNBlock の演算連鎖で oneDNN プリミティブの生成時/実行時 thread 数が食い違うと発火する。
  - そのため **本番コードから手動実装を削除**し、CPU テスト用に `EnsureNNInitialized` で `at::globalContext().setUserEnabledMkldnn(false)` を1回設定する方式へ変更した。本番 runner は無改造。影響が残るのは「x64-Debug ビルド + CPU 実行 + CNBlock 使用」の組み合わせのみで、その場合は runner 側の初期化にも同設定を足せば解消する。
- `ImageCls.txt` に ConvNeXt-Tiny 構成の `ConvNeXtT` branch を追加し、`main_feature` をその branch へ切り替える。
- `CONTEXT.md` と ADR は更新しない。今回の内容は強化学習ドメイン用語ではなく、既存 NN DSL と初期化設定の局所拡張である。

## テスト

- `[nn][init]`: `trunc_normal` の範囲、標準偏差の概算、bias 0、不正 `trunc_std` / `trunc_a,b` の fail-fast。
- `[nn][layernorm]`: 既存 `LayerNorm` の `eps` 未指定既定値と明示指定値。
- `[nn][layernorm2d]`: `LayerNorm2d` の shape 不変、channel 軸正規化、invalid config/input の fail-fast。
- `[nn][cnblock]`: shape 不変、config dump、LayerScale gamma、`layerscale_init<=0`、eval DropPath no-op、train 高 droppath shortcut、CPU backward（mkldnn 無効化下の通常 Conv2d 経路）、channel mismatch、`norm_type=none` / unknown norm。
- `EnsureNNInitialized` で mkldnn を無効化しているため、Debug / RelWithDebInfo 双方で全 `[nn]` テストが green（237 assertions / 39 cases、oneDNN assert なし）。
- `[nn][config_profile]`: `start/end` 前提の既存補間テストを再実行する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn][init]"
core\anet-core\bin\Debug\anet-core-test.exe "[nn][layernorm]"
core\anet-core\bin\Debug\anet-core-test.exe "[nn][layernorm2d]"
core\anet-core\bin\Debug\anet-core-test.exe "[nn][cnblock]"
core\anet-core\bin\Debug\anet-core-test.exe "[nn][config_profile]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check -- core/anet-core/include/anet/nn.hpp core/anet-core/include/anet/default_dqn_agent.hpp core/anet-core/include/anet/rainbow_agent.hpp core/anet-core/src/nn_impl.hpp core/anet-core/src/nn_modules.cpp core/anet-core/src/nn_test.cpp apps/runner/config/ImageCls.txt docs/memo/028_cnblock_convnext_20impl.md
```

## 前提

- DropMerge 用 ConvNeXt branch、ConvNeXt v2/GRN、汎用 `Conv2d.groups`、GELU approximate 公開、精度評価は今回の範囲外。
- ImageCls の既定 branch は `ConvNeXtT` に切り替える。既存 ResNet/Hybrid/ViT branch は切替候補として残す。
- 無関係な未コミット変更は保持する。
