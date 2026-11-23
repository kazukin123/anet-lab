# ImageCls GAP2D 実装メモ

## 概要
`docs/memo/023_imagecls_gap_head_10prd.md` に従い、ImageCls の比較用 branch として `FoodResNet13_GAP` を追加し、最終段を `Flatten > Drop > LinearOut` から `GAP2D > Drop > LinearOut` へ置き換える。

既定実行はユーザー指定どおり、022 の train augmentation を有効にしたまま `FoodResNet13_GAP` を使う。`LinearOut` の `in_features` は既存の lazy 推論に任せ、config では明示しない。

## 主な変更
- `core/anet-core/src/nn_modules.cpp` に `GlobalAveragePooling2DModule` と factory を追加し、`GAP2D` として登録する。
- `GAP2D` は `[B,C,H,W]` の空間次元 `dim={2,3}` を平均し、`[B,C]` を返す。profile と config dump は既存 `GAP1D` に合わせる。
- `apps/runner/config/ImageCls.txt` に `FoodResNet13_GAP` branch を追加し、`main_feature` をその branch へ切り替える。既存 `FoodResNet13` は切戻し用に残す。
- `CONTEXT.md` と ADR は更新しない。

## テスト
- `core/anet-core/src/nn_test.cpp` に factory 経由で `GAP2D` を作る単体テストを追加し、既知入力 `[2,3,2,4]` から `[2,3]` の空間平均が得られることを確認する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

runner 実データ確認では MODEL SHAPE DUMP で `LinearOut` が `[101,128]` になり、logits が `[B,101]` で forward できることを確認する。

## 前提
- 022 の augmentation 設定は変更しない。
- Food-101 の eval 改善や gap 縮小は、ユーザー側の seed 違い複数 run の終盤平均で評価する。
- under-capacity が出た場合の `ConvDown256` や GeM は後続で扱う。
