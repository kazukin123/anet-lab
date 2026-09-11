# ImageCls Train 専用データ拡張 実装メモ

## 概要
`docs/memo/022_imagecls_augmentation_10prd.md` に従い、ImageCls の train 経路だけに水平フリップと RandomResizedCrop 相当の拡張を追加する。eval 経路と `ImageDataSource` は変更しない。

既定は後方互換のため `augment.enabled = false` とし、`apps/runner/config/ImageCls.txt` の baseline だけ `true` にする。強度既定は `hflip_p = 0.5`、`rrc_scale = 0.7..1.0`、`rrc_ratio = 0.75..1.3333333`。

## 主な変更
- `ImageClsEnvConfig` に `augment.enabled`、`augment.hflip_p`、`augment.rrc_scale_min/max`、`augment.rrc_ratio_min/max` を追加し、設定値を `ANET_SYSTEM_ERROR` で明示検証する。
- `ImageClsEnv::FetchRandomImageState` で `!IsEval(mode) && config_.augment.enabled` の場合だけ、`obs.Set(kGrid, ...)` 直前に拡張を適用する。
- crop は true RandomResizedCrop 寄りにし、10 回まで scale/ratio から矩形を試行する。失敗時は中心 crop に fallback し、float bilinear resize 後に clamp/round して `[3,H,W] uint8` へ戻す。
- 乱数は env の `rnd_` のみを使い、torch global RNG は使わない。拡張 helper には `ANET_PROFILE_SCOPE(augment)` を置く。
- `ImageCls.txt` は既存の未コミット変更を保持し、baseline block に train ON 設定、default block に OFF 設定を追記する。

## テスト
- x64-Debug の全体ビルド。
- 既存 `anet-core-test`。
- `git diff --check`。
- 実データ環境では runner の ImageCls online 起動で、train 表示が crop/flip され、eval 表示が無加工のままかを確認する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提
- 新しい env 専用 test target は今回は追加しない。
- `CONTEXT.md` と ADR は更新しない。
- 効果判定は seed 違い複数 run の eval accuracy 終盤平均と train-eval gap でユーザーが評価する。
