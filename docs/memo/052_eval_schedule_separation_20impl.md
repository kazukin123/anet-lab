# PRD 052: eval 定義・スケジュール分離 実装メモ

## 概要

`train.eval.[tag]` を純粋な Eval 定義、`train.eval_schedule.[tag]` を定期駆動として分離する。
LunarLander の意図しない `eval_panel` 背景評価を解消し、既存の有効 Eval は周期、background 設定、`"eval_env/" + tag` seed domain を維持する。

## 主な変更

- RunManager 構築時に定義と schedule を別々に列挙し、main Env の生成前に全 schedule の参照、必須 `interval`、型、範囲を検証する。`use_background` は既定 `true` とし、`interval=0` の schedule も型検証する。
- 全定義タグで name 予約、`run_mode` / `eval_batch_size` / `clone_model` の解決、Env 設定検証、設定記録を行う。有効 schedule がないタグは dormant とし、Env / Runner / Observer を生成しない。
- `interval>0` のタグだけ従来と同じ seed domain で Eval Env、EvalRunner、EpisodeEvalObserver を生成する。EpisodeEvalObserver 本体は変更しない。
- 起動時に `eval tag '<tag>': scheduled (interval=<N>, background=<true|false>)` または `eval tag '<tag>': definition-only` を出力する。
- dormant metrics の WARN を新契約の文言へ変更してタグごとに一度だけ出力する。未定義タグ参照は従来どおりエラーにする。
- `RunManager::Config::eval_interval` と `train.eval_interval` を削除する。旧位置キーの互換処理、専用検出、WARN は追加しない。
- `apps/runner/config/*.txt` の `interval` / `use_background` をコメントアウトされたトグル候補も含めて `train.eval_schedule` へ移す。Atari の `eval_panel` は `run_mode = eval1` で定義を維持し、LunarLander の `eval_panel` は定義だけの状態を維持する。
- `docs/design/100_runtime_and_configuration.jp.md` と `docs/design/010_framework_overview.jp.md` を schedule 駆動と導出 dormant の契約へ更新する。既存の PRD、ADR 0027、`CONTEXT.md` の変更は保持する。

## テスト

- Public interface / surface: `ConfigData` からの `RunManager` 構築、`GetEvalRunner()`、Env factory の生成結果、起動ログ、metrics WARN、設定ファイル。
- 優先 behavior:
  1. 定義のみのタグでは Eval Env / Runner が生成されず、definition-only ログと dormant metrics WARN が観測できる。
  2. 未定義タグの schedule、`interval` 欠落、負値が構築時に fail-fast する。
  3. `interval=0` は schedule 無しと同じ生成結果になる。
  4. 有効 schedule では EvalRunner が生成され、設定した interval / background が role log に現れ、既存 Env config prefix と seed 生成経路が維持される。
  5. 同じ dormant タグを複数 metrics が参照しても WARN は一度だけ出る。
- TDD 順序: 上記 1 を tracer bullet とし、1 behavior ごとにテスト追加 → RED 確認 → 最小実装 → GREEN 確認を完了してから次へ進む。既存 active Eval テストは明示 schedule へ移行し、production に test-only API は追加しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe "[eval_schedule]"
core\anet-core\bin\Debug\anet-core-test.exe
rg -n "train\.eval\.\[[^]]+\]\.(interval|use_background)|train\.eval_interval" apps/runner/config
git diff --check
```

残存検索は出力なしを合格条件とする。

## 前提

- C++ 公開 API は変更せず、設定契約のみクリーンブレークする。
- Eval の LEARN 軸判定、background worker、同位相停止、EvalPanel mirror runner は変更しない。
- 現行 focused baseline は `[run_manager]` 4 cases / 29 assertions、`[dormant_eval]` 1 case / 4 assertions が成功済み。
- 無関係な未コミット変更を保持し、stage / commit / push は行わない。
