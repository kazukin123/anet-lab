# PRD060 評価セッション集約 実装計画

- 承認日: 2026-09-03
- 対象 PRD: `docs/memo/060_eval_batch_episodes_10prd.md`
- 決定記録: `docs/adr/0034-eval-session-aggregation-in-batchenv-decorator.md`
- 状態: implemented and verified

## 概要

- P1 の Atari 評価境界補正と、P2 の N-episode 評価セッション機構を実装する。
- P3 の恒久的な N/L/interval 選定と性能測定は実装範囲外とする。

## 主な変更

- `EpisodeScope { PER_LANE, SHARED }` を追加し、`BatchEnvSpec` に既定 `PER_LANE` の `episode_scope` を加える。JSON は `per_lane` / `shared` を出力する。
- 汎用 scalar 集約部品を `util.hpp` / `util.cpp` の `anet` 名前空間へ追加する。
  - `mean/max/min/std` を扱う `ScalarAggregation`、scalar key parser、`ScalarSampleAccumulator`。`nullopt` は poison、NaN は除外、標準偏差は母集団標準偏差とする。
- Env 固有部品として以下を `env.hpp` / `env.cpp` の機能グループへ追加する。
  - Reset/Step の episode-group 構造検証。
  - `{group_index, episode_return}` を返す `EpisodeReturnAccumulator`。
- `EvalSessionEnv(inner, eval_episodes, subscribed_env_keys)` を追加する。
  - `Reset()` でセッションを開始し、全 group が fresh なら cached `continue_state` と空 AuxData を再利用、それ以外は inner を全 Reset する。
  - 開始時動的グラントで正確に N 本を採用し、完了 Step 直後に scalar と return を取得する。
  - `GetSessionResult()` は進行中 `nullopt`、完了後は次回 Reset まで同じ `EvalSessionResult` を返す。
  - 購読外 scalar、indexed scalar、tensor、config、shutdown は inner へ透過委譲する。
- configured eval 用 `EvalRunner` は `EvalSessionEnv` を型付きで受け取る constructor と `RunSession(event_counts)` を持つ。通常の `BatchEnv` constructor/`DoStep()` は EvalPanel 用として残し、名前解析や `dynamic_cast` で両経路を判定しない。
- `RunSession()` は `Sync → Reset → 完了まで DoStep → 最終 EpisodeEndEvent 1回` を実行する。中間 episode event は抑制し、最終 event は decorator、`env_index=-1`、呼出元 counts を使用する。
- Runner の直近完了 return 群を共通集約へ移し、`mean/max/min/std.episode_return` を公開する。Train の現行値は `max.episode_return`、configured eval はセッション N 本の集約になる。
- `EpisodeEndEvent::eps_total_reward`、`eps_total_reward`、`train_episode_reward` はクリーンブレークで削除する。現用 config・テスト・設計文書を `mean.episode_return` / `max.episode_return` へ同時移行し、alias や旧キー専用 WARN は残さない。
- `RunManager` は active な評価タグごとに、解決済み metric 定義から `EVAL + EPISODE_END + ENV` の source key を抽出・重複排除して decorator へ渡す。`eval_episodes<=0` と N>1 の無 prefix は起動時エラー、N<G はタグごと1回 WARN とする。
- ImageCls eval は `SHARED` とし、window 終端時に全 lane の `done` / `continue_state.episode_start` を立てて `n_episode_end=1` とする。indexed scalar や preload 特例は追加せず、eval reward 2タグを削除して accuracy を維持する。
- Atari の eval1/eval2/eval_panel に `env.episodic_life=false` を追加する。DropMerge の出荷既定 L=N=1 は維持する。
- ADR 0034 と既存 `CONTEXT.md` を再利用し、新規 ADR は作らない。現行実装を説明する framework overview、runtime、environment、observability、Runner、Atari の設計文書を同期する。

## TDD とテスト

- PER_LANE L=1,N=1 の非退行は golden trace と configured eval の構造的等価性で固定する。編集前baselineを採取しなかったため、Atari `@v5_noop30` と現行DropMergeのbase/branch比較は実施しない。
- 次の public seam を縦スライスで、各項目ごとに RED → 最小 GREEN の順で実装する。
  1. `BatchEnvSpec` JSON、scalar 集約、PER_LANE/SHARED 構造検証。PER_LANE の continuation 不整合では env name / group / lane / expected / actual を固定する。
  2. `EvalSessionEnv` の L=2/N=3、遅い採用 group を待つ L=4/N=2、lane 固有値と区別できる SHARED N>1、同時完了順、正確なN件取得。
  3. fresh-state 再利用、全Reset条件、空AuxData、N=1 golden trace、session result lifecycle。
  4. `EvalRunner::RunSession` の単一event、counts、runner/env scalar、背景例外伝播。
  5. RunManager の `eval_episodes` 検証、購読key抽出、prefix error、N<G WARNがタグごと1回、SHAREDでWARNなし、EvalPanel非適用。
  6. ImageCls の train PER_LANE、eval SHARED、全lane mask、`EvalSessionEnv` N>1でのwindow数集計、accuracy維持。
  7. 旧 source key の完全移行と metrics 定義整合。
- 構造契約の各違反、`done && truncated` 許容、全aggregationのnullopt poison / 0件NaN、std境界を独立した golden 値で検証する。
- configured eval は `Sync`、fresh state再利用、同じStep列、同じstep座標の単一eventという経路をコードとgolden traceで確認する。trainの`mean.*`は、同一Stepで複数laneが完了した場合にfloat逐次和からdouble Welfordへ移行したことによる最下位bit差を意図差分として許容する。
- LunarLander、GridMaze、CartPole、ImageCls の一時 config smoke を `app.train_exit_step=24` で実行し、exit code 0、構造契約エラーなし、最終 `episode_count >= 6` を確認する。
- 検証コマンド:

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe "[env]"
core\anet-core\bin\Debug\anet-core-test.exe "[episode_end]"
core\anet-core\bin\Debug\anet-core-test.exe "[trainer]"
core\anet-core\bin\Debug\anet-core-test.exe "[observers]"
core\envs\imagecls1\bin\Debug\ImageClsEnv-test.exe
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 検証結果

### 受入8: N=1非退行

- `EvalSessionEnv keeps the N=1 single-lane trace and scalar identity` により、PER_LANE L=1,N=1のstate / reward / reset回数 / scalar identityを固定した。
- `EvalRunner::RunSession()` はmodel sync後にsessionをResetし、完了まで同じStep列を進め、呼出元countsを持つ単一`EpisodeEndEvent`を通知する。fresh stateではinnerを再Resetしない。
- base/branch checksumは、編集前baselineを採取しておらず開始時のdirty stateを再現できないため未実施とした。`HEAD`からの後追い比較は真正なbaseにならないため代替に用いない。
- trainのmean系scalarはdouble Welfordへの移行により、同一Stepで複数laneが完了した場合に主として`42_env/*`で最下位bit差が生じうる。これは意図した数値変更である。
- 今後、等価性比較を受入条件に置くPRDでは、編集前baseline採取を実装手順の最初に置き、完了確認後に編集を開始する。

### 受入11: RunnerBase train smoke

- 実行日: 2026-09-04
- 共通条件: `train.num_envs=2`、episode上限4、`app.train_exit_step=24`、configured eval無効、一時`episode_count` metric有効。
- 一時configと新規Runは`out/test-tmp/prd060-smoke/exit24/`へ出力し、既存Run artifactは変更していない。

| Env | Exit code | 最終 episode_count | group mismatch | continuation mismatch | `[E]` |
|---|---:|---:|---:|---:|---:|
| LunarLander | 0 | 10 | 0 | 0 | 0 |
| GridMaze | 0 | 10 | 0 | 0 | 0 |
| CartPole | 0 | 10 | 0 | 0 | 0 |
| ImageCls | 0 | 10 | 0 | 0 | 0 |

- Debug buildは成功し、`ninja: no work to do.`で現ソースと実行バイナリが一致していた。
- Source codeは変更していないためunit testは再実行せず、直前に通過済みの`[util]`、`[env]`、`[episode_end]`、`[trainer]`、ImageCls全件、`anet-core-test`全件を根拠として再利用した。

## 前提

- `eval_episodes` の既定値は1で、common.txt の eval1/eval2にも明記する。
- configured eval だけを session decorator へ載せ、EvalPanel の逐次操作、強制Action、同期方式は維持する。
- P2 は atomic に完成させ、partial Reset、masked Step、非採用groupのfreeze、`se.` prefix、恒久 `game_score_std` は追加しない。
- 過去 Run artifact、完了済みPRD/ADR、実験記録は変更しない。
- 無関係な未追跡ファイルは保持し、staging・commit・pushは行わない。
