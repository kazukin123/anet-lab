# ImageCls Cosine LR 実装メモ

## 概要
`ProfiledValue<T>` を共通部品として追加し、まず `ImageClsAgent.learning_rate.*` の構造化 config に適用する。`ImageClsLearner` は `step.exp_step` を渡して learning rate を更新し、AdamW の全 param group へ optimizer step 前に反映する。

旧 scalar 形式 `ImageClsAgent.learning_rate = 1e-3` への fallback は実装しない。ImageCls の現在値は Agent scalar として `learning_rate` で公開し、`$agent learning_rate @learn interval:100` で記録する。

## 主な変更
- `anet::Config` に `ConfigReader<T>` customization point、`ReadSubConfig(...)`、`MakeTaggedSubConfigKey(...)` を追加し、通常型の読み取りは既存の default prefix -> override prefix の挙動を保つ。`ConfigReader<T>` primary template は `config.hpp` に置く。
- `ConfigData::Set` に vector 用 overload を追加し、`learning_rate.phases` を `ToConfigString()` / `ToJson()` に展開できるようにする。
- カテゴリ方針に合わせて `ProfiledValueConfig<T>`、`ProfiledValuePhaseConfig<T>`、`ProfiledValue<T>` を `schedule.hpp` に実装し、`schedule.cpp` を schedule カテゴリの translation unit として追加する。
- `schedule.hpp` が `ANET_SYSTEM_ERROR` を使えるようにしつつ include 循環を避けるため、診断系 API とマクロを `diag.hpp` / `diag.cpp` に分離する。汎用 util も `common.hpp` ではなく `diag.hpp` 経由で診断系 API を使う。
- `ConfigReader<ProfiledValueConfig<T>>` は `schedule.hpp` に特化と inline `Read(...)` 本体を置く。`schedule.hpp` は `config.hpp` を include し、`config.hpp` は `schedule.hpp` を include しない。
- `ProfiledValue<T>` の step 型は `uint64_t` とし、`schedule.hpp` から `rl.hpp` へ依存させない。
- `ProfiledValue<T>` は `constant`、`linear`、`cosine`、`cosine_restart`、`phased` を扱い、不正 type、必須 `steps` 不足、`cycle_mult <= 0`、空 `phases`、phased 内 `steps` 不足は `ANET_SYSTEM_ERROR` で失敗させる。
- `cosine_restart` の `0 < cycle_mult < 1` は有効値とし、cycle 長を restart ごとに縮小して 1 step 下限に飽和させる。
- `EvaluateByIndex(index, count)` は `constant` / `linear` / `cosine` を type-aware に評価し、`cosine_restart` / `phased` は `ANET_SYSTEM_ERROR` で拒否する。
- `ImageClsAgentConfig::learning_rate` を `ProfiledValueConfig<double>` に変更し、`ImageClsAgent` が単一の `std::shared_ptr<ProfiledValue<double>>` を保持して Learner に渡す。
- Learner と Agent scalar metric は同じ `ProfiledValue<double>` から現在値を読み、既存 `shared_mutex` 境界で同期する。
- `apps/runner/config/ImageCls.txt` は既存 dirty な active branch や実験値を維持し、`ImageClsAgent.learning_rate.*` と LR metric 行だけを更新する。

## テスト
- `core/anet-core/src/schedule_test.cpp` で `ProfiledValue<double>` の `constant`、`linear`、`cosine`、overrun、`cosine_restart`、縮小 restart、`phased`、`EvaluateByIndex`、invalid config を検証する。
- `ConfigReader<ProfiledValueConfig<double>>` が root field、`phases`、`phase.[name].*` を読み、override で `type` だけを変えた場合に dormant field を利用できることを検証する。
- `Config::ToConfigString()` と `Config::ToJson()` に `learning_rate.type`、`learning_rate.phases`、`learning_rate.phase.[warmup].start` などが出ることを検証する。
- ImageCls の小型 trainable network で Learner 更新後に共有 `ProfiledValue<double>` の現在値が変化し、`ImageClsAgent::GetScalar("learning_rate")` で同じ値を読めることを検証する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[profiled_value],[config],[image_cls]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

`anet-core-test.exe` が 120 秒前後で timeout する場合は、stdout/stderr をリダイレクトして長めの timeout で再実行する。

## 前提
- `net.config_profile` の既存挙動は今回変更しない。`EvaluateByIndex` は後続移行用 API として追加する。
- `ProfiledValue<T>` 自体は thread-safe にしない。排他は ImageCls 側の既存 mutex 境界で確保する。
- 既存 worktree には unrelated dirty files があるため、無関係な差分は保持する。
