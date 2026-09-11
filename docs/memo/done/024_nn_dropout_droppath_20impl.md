# NN dropout / DropPath 実装メモ

## 概要
`024_nn_dropout_droppath_10prd.md` に従い、NN モジュール内の正則化レバーを追加する。既定値はすべて `0.0` で既存挙動を維持する。`ResBlock` と `DropoutModule` は `dropout_rate`、`TransformerEncoder` の hidden/residual 用要素 dropout は `hidden_dropout_rate` として扱う。`DropoutModule` は `p` から `dropout_rate` へ後方互換なしで改名する。

## 主な変更
- `anet::nn::DropPath` を internal helper として追加し、ResBlock と TransformerEncoder から共有する。mask は sample 単位の `[N,1,1,...]` とし、eval または `drop_prob<=0` では no-op にする。
- `ResBlock` に `res.droppath_rate` と `res.dropout_rate` を追加する。`dropout_rate` は conv1 活性化後から conv2 前に `Dropout2d` として適用し、DropPath は残差枝だけに適用して shortcut/downsample は落とさない。
- `TransformerEncoder` に `tf.hidden_dropout_rate`、`tf.attn_dropout_rate`、`tf.droppath_rate` を追加する。attention weights dropout は `MultiheadAttentionOptions.dropout`、hidden/residual 用の要素 dropout は attention/FFN 枝、DropPath は residual add/norm の直前に適用する。
- `DropoutModule` は `[Drop].dropout_rate` のみを読む。旧 `[Drop].p` は読まず、fallback や WARN は入れない。
- `ImageCls.txt` は既存の branch/run_name 変更を保持し、`net.block.[Drop].p` の表記だけ `dropout_rate` へ置換する。
- `docs/adr/0007-nn-dropout-config-semantics.md` に、モジュールごとの dropout 設定名と `p` の後方互換なし改名を記録する。

## テスト
- DropPath helper: no-op、shape 維持、sample 単位 mask、inverted scaling。
- DropoutModule: `dropout_rate` が効くこと、旧 `p` が読まれないこと、config dump が `dropout_rate` になること。
- ResBlock: config dump、範囲外 error、BatchNorm 併用 warning、eval no-op、train 時 DropPath/Dropout2d の差分。
- TransformerEncoder: `hidden_dropout_rate` の config dump、pre-LN/post-LN と SDPA/legacy の eval no-op、train 時 dropout 差分、既存 `self_attn.*` checkpoint 名の維持。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[resblock]"
core\anet-core\bin\Debug\anet-core-test.exe "[transformer]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check -- core/anet-core/src/nn_impl.hpp core/anet-core/src/nn_modules.cpp core/anet-core/src/nn_test.cpp apps/runner/config/ImageCls.txt
```

## 前提
- 旧 `Drop.p` は意図的に読まない。移行漏れは silent に dropout=0 になるため、repo 内 config は本実装で `dropout_rate` へ更新する。
- CUDA で `tf.attn_dropout_rate>0` と deterministic algorithms が衝突して throw する場合は、既定値は 0 のまま維持し、運用上は `deterministic_warn_only=true` への退避または当面 0 を選ぶ。
- `CONTEXT.md` は用語集であり、今回の実装詳細は追記しない。
