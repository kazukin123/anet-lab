# ImageCls 過学習対策②: 最終段 Flatten → Global Average Pooling 2D（GAP2D）

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 過学習対策の対になるデータ側施策は `022_imagecls_augmentation_10prd.md`（train 専用拡張）。
> 両者は独立に実装・A/B 可能。効果切り分けのため **1 個ずつ** 入れて評価する。

## Context（背景・目的）

ImageCls（Food-101 / 101 クラス / 224×224）は **train 高・eval 低の過学習**（ユーザー確認, 2026-06-25）。

構造的な主因が **最終段の Flatten → Linear**:

- 現 net `FoodResNet13`（`ImageCls.txt:153-154`）:
  `... ResBlock128(*2) [28×28×128] > Flatten [100,352] > Drop > LinearOut [101]`。
- 224 を 3 回 stride2 で落として **空間 28×28 のまま Flatten** → 100,352 次元。
- `LinearOut` は in 100,352 × out 101 ≈ **1014 万パラメータが最終層 1 枚に集中**。容量過多で過学習を強く後押し。

目的: **Flatten を GAP2D に置換**し、空間 28×28 を平均で 1 点に潰す。`[B,128,28,28]→[B,128]→Linear[101]`
（最終層 ≈ 128×101 ≈ **1.3 万パラメータ**）。容量を構造的に削って汎化を上げる。

## 確定した設計判断

1. **GAP2D を `NetworkModule` として新規追加**。`input.mean({2,3})` で `[B,C,H,W]→[B,C]`。
   既存 `GlobalAveragePooling1DModule`（`nn_modules.cpp:1563`）の隣に置き、命名も揃えて **`GAP2D`** で登録。
2. **`LinearOut` の in_features 変更は不要**。`LinearModule` は forward 時に `in_features = x.size(-1)` を
   推論して `Linear` を構築する LazyLinear 的実装（`nn_modules.cpp:56-100`）。よって structure の
   `Flatten` を `GAP2D` に差し替えるだけで Linear が 128 入力に自動追従する。**config の `LinearOut` は触らない**。
3. **比較のため新 branch `FoodResNet13_GAP` を追加し、既存 `FoodResNet13` は温存**。`main_feature` 切替で
   A/B（cf. 構成比較はブレ幅基準・1 個ずつ）。即切戻し可。
4. **Dropout は GAP の後**（`GAP2D > Drop > LinearOut`）。現 `[Drop] p=0.2` を流用。Flatten 時は 100,352 次元への
   dropout だったが、GAP 後は 128 次元への素直な dropout 位置になる。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `100e483`（`nn_modules.cpp` は未コミット変更を含む。行番号は現 working tree 基準）。

- `GlobalAveragePooling1DModule`（`nn_modules.cpp:1563-1579`）: `Forward` は `input.mean(/*dim=*/1)`。
  `GetCurrentConfigData` で `dim/op` をダンプ。Factory（`:1581-1587`）→ `repo.Register("GAP1D", ...)`（`:1947`）。
- `FlattenModule` は別在（`:299-`、登録 `:1930`）。GAP2D はこれを置換する位置に入る。
- `LinearModule`（`:56-107`）: コンストラクタは `out_features` のみ保持。`Forward` で
  `const int64_t in_features = x.size(-1)` → `LinearOptions(in_features, out_features_)` を構築（`:71-75`）。
  `GetCurrentConfigData` が確定後の `in_features` をダンプ（`:100`）。→ **in_features は自動推論**。
- `NetworkModule::Forward` は `torch::Tensor` 単入出力。`ResBlock128` 出力は `[B,128,28,28]`（224/2/2/2=28）。
- `ImageCls.txt`: net.block 群 / `net.branch.FoodResNet13`（`:153-154`）/ `main_feature` 切替（`:159-161`）/
  `[Drop] p=0.2`（`:86-88`）/ `[LinearOut] out_features=101`（`:129-131`、in_features 指定なし）。
- パラメータダンプ: `ImageClsAgent` 構築時に `named_parameters()` を全列挙（`image_cls_agent.cpp:162-166`）。
  GAP 化の効果（最終層 shape と総数の激減）はこのログで確認できる。

## 設計方針

### A. GAP2D モジュール追加（`nn_modules.cpp`、GAP1D の隣）
- `GlobalAveragePooling2DModule : NetworkModule`。`Forward`: `input.mean(/*dims=*/{2,3}, /*keepdim=*/false)` → `[B,C]`。
  `ANET_PROFILE_FUNC()` を入れる。`GetCurrentConfigData` に `dims={2,3}, op=mean` を出す。
- `GlobalAveragePooling2DFactory` を追加。

### B. 登録（`nn_modules.cpp` の Register 群、`:1947` 隣）
- `repo.Register("GAP2D", std::make_shared<GlobalAveragePooling2DFactory>());`

### C. config（`apps/runner/config/ImageCls.txt`）
- 新 branch を追加（既存 `FoodResNet13` は残す）:
  ```
  net.branch.FoodResNet13_GAP.bind = grid
  net.branch.FoodResNet13_GAP.structure = ConvInit32 > BN32 > SiLU > ResBlock32 > ConvDown64 > BN64 > SiLU > ResBlock64 > ConvDown128 > BN128 > SiLU > ResBlock128(*2) > GAP2D > Drop > LinearOut
  ```
- `main_feature` を `FoodResNet13_GAP` に切替（A/B 用にコメントで両方残す）。

## 非対象（Out of Scope）

- 他環境のネット、`GAP1D` の変更、`Linear` の明示 in_features 化。
- 任意出力サイズの `AdaptiveAvgPool`（1×1 固定で十分）、GeM / attention pooling（後続候補）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/src/nn_modules.cpp` | `GlobalAveragePooling2DModule` + Factory 追加、`repo.Register("GAP2D", ...)` |
| `core/anet-core/src/nn_test.cpp` | GAP2D の単体テスト追加（`[nn]`、`[B,C,H,W]→[B,C]` と平均値の検証） |
| `apps/runner/config/ImageCls.txt` | `FoodResNet13_GAP` branch 追加、`main_feature` 切替 |

## 受け入れ基準

1. **ビルド緑**（x64-Debug）、既存テスト緑。
2. **GAP2D 単体テスト**: 既知入力 `[B,C,H,W]` に対し出力が `[B,C]` で各値が空間平均に一致。
3. **パラメータ激減の確認**: 起動時 MODEL SHAPE DUMP（`image_cls_agent.cpp:162-166`）で `LinearOut` が
   `[101,128]` になり、総パラメータが Flatten 版から大幅減。forward が通り logits `[B,101]`。
4. **効果（ユーザー実測）**: 同設定・seed 違い複数 run で eval accuracy 終盤平均が **ブレ幅を超えて改善** または
   gap 縮小。train accuracy は下がりうる（過学習緩和の正常な兆候）が、eval が改善すれば成功。
5. **perf**: パラメータ減で Learn が軽くなる方向（`90_perf` で確認、悪化しないこと）。

## 正直なリスク

- GAP は空間情報を平均で捨てる。Food-101 は局所テクスチャ依存が強く、容量を削りすぎて **train も eval も下がる
  （under-capacity）** 可能性がある。→ 新 branch 比較なので即切戻し可。その場合の後続は「GAP 前に `ConvDown256` を
  1 段足して channel を増やしてから GAP」または GeM pooling。
- 28×28→1×1 の急な圧縮。phase1 は素の GAP で評価し、足りなければ上記後続へ。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認は runner を online モードで起動し、起動ログの MODEL SHAPE DUMP で `LinearOut [101,128]` を確認。
- 効果（eval / gap）はユーザーが seed 違い複数 run の終盤平均で評価。

## 後続

1. 実装メモ `023_..._20impl.md`（必要なら）→ Codex 実装 → 受け入れ緑 → ユーザー A/B 評価。
2. under-capacity なら channel 増（ConvDown256）or GeM。022（拡張）と組合せて最終 gap を評価。
