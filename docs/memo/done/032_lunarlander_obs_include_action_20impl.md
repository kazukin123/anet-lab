# LunarLanderEnv obs_include_action 実装メモ

## 概要

`LunarLanderEnv.obs_include_action` を default false の bool config として追加する。
false 時は obs 8 次元、既存 index、乱数消費列を維持する。true 時だけ `ObsKeys::kVector` の末尾に直前 action の one-hot 4 次元を追加し、obs/spec を 12 次元にする。

## 主な変更

- `LunarLanderEnvConfig` に `obs_include_action = false` を追加し、`ANET_READ_CONFIG` で読む。
- `LunarLanderEnv` に `kActionCount = 4` と `last_action_ = -1` を追加し、`Reset()` で未行動に戻し、`Step(action)` で物理 step 前に直前 action を記録する。
- `GetSpec()` は flag true 時に vector obs の shape、labels、min/max を末尾 4 要素分だけ拡張する。
- `makeState()` は先頭 8 要素の順序と値を変えず、flag true 時だけ `a_noop, a_left, a_main, a_right` の one-hot を末尾へ追加する。dead-state の zero obs も flag に応じた次元にする。
- constructor の `MetricsLogger::Instance()` 呼び出しは null guard 付きにして、logger 未初期化の単体テスト構築を許可する。
- `core/envs/lunarlander1` に Catch2 の `LunarLanderEnv-test` target を追加し、test cpp は library source から除外する。
- `apps/runner/config/LunarLander.txt` の base scope 近辺に、実験用のコメントアウト行 `#LunarLanderEnv.obs_include_action = true` を追加する。

## テスト

- default false で spec shape `{8}`、Reset/Step obs numel 8、config dump に `obs_include_action=false` が出ること。
- true で spec shape `{12}`、labels/min/max の末尾 4 要素、Reset 直後 one-hot 全ゼロ、4 action それぞれの Step 後 one-hot を検証する。
- 同 seed・同 action 列で false/true を並走し、各 step の先頭 8 次元が一致すること。
- base `LunarLanderEnv.obs_include_action=true` と eval/test prefix を持つ `ConfigData` を factory/batch env 経由で使い、spec と実 obs 次元が一致すること。
- `limit_step=1` で truncated 到達時も obs 12 次元を保つこと。
- MetricsLogger 未 Init で直接構築してもクラッシュしないこと。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target LunarLanderEnv-test'
core\envs\lunarlander1\bin\Debug\LunarLanderEnv-test.exe
git diff --check
```

## 前提

- `CONTEXT.md` は既存の Observation 用語で足りるため更新しない。
- ADR は不要。今回の flag は暫定、default off、Env 局所変更で、戻しにくい設計決定ではない。
- variant override は仕組みとして禁止しないが、運用上は base `LunarLanderEnv.obs_include_action` に置く。
- runner smoke と実験 run/commit は PRD 記載どおりユーザー実施。
