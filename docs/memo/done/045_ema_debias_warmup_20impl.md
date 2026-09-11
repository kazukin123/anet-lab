# EmaFilter バイアス補正ウォームアップ 実装メモ

## 概要

`EmaFilter<T>` を、ゼロ初期化した内部値と観測済み重み和によるバイアス補正EMAへ変更する。公開シグネチャと呼び出し側は変更せず、step 1 から欠損のない補正済み値を返す。

## 主な変更

- `EmaFilter<T>` に `weight_` を追加し、`value_ += decay_ * (x - value_)` と `weight_ += decay_ * (1 - weight_)` を同時更新する。`Value()` と暗黙変換は初期化済みなら `value_ / weight_` を返す。
- `Set(v)` と2引数コンストラクタは `value_=v`、`weight_=1`、初期化済みとして補正をスキップする。2引数コンストラクタは現行コードからの意図的な意味変更だが、現用呼び出しはない。
- `Restart()` は値を保持しつつ `weight_=0`、未初期化へ戻す。NaN/Inf入力は値・重みとも更新しない。
- `decay` は全コンストラクタと `SetDecay()` で finite かつ `0 < alpha <= 1` を検証し、違反時は指定値と期待範囲を含む英語の `ANET_SYSTEM_ERROR` で fail-fast する。失敗した `SetDecay()` は既存値を変更しない。
- arithmetic 型契約は維持する。浮動小数点型の既定 decay は `0.01`、整数型は `1` とし、整数型で明示可能な有効 decay も実質 `1` とする。
- PRD の `pow_` 契約を数値安定な `weight_` 契約へ更新し、`docs/design/140_observability.jp.md` もバイアス補正方式と `0 < ema_alpha <= 1` に同期する。既に追加済みの `CONTEXT.md` 用語は変更しない。
- metrics parser 固有の tag/key 付き検証、既存Runの再計算、互換フラグ、新規ADRは追加しない。

## テスト

- Public interface / surface: `EmaFilter<T>` の公開コンストラクタ、`Update()`、`Value()`、暗黙変換、`Set()`、`SetDecay()`、`Restart()`、`IsInitialized()`。
- 優先 behavior: 1件目・2件目の補正値、途中 decay 変更、長期収束、既知値開始、再始動、非finite入力skip、不正 decay と整数既定値をこの順で確認する。
- TDD 順序: `core/anet-core/src/util_test.cpp` に tracer bullet を1件追加して RED を確認し、最小実装で GREEN にした後、次の behavior を1件ずつ追加する。private state は検査せず、既知リテラルまたは独立した参照系列と公開APIだけで確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --preset x64-Debug && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[util]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- `MetricsLogObserverBase` と `Trainer` の現用 decay は正の `float` であり、呼び出し側の変更は不要。
- `EmaFilter` はシリアライズ対象外なので、保存形式やcheckpoint移行は発生しない。
- 無関係な未コミット・未追跡ファイルは保持し、Gitのstage・commit・pushは行わない。
