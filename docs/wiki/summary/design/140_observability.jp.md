---
title: 可観測性
page_kind: サマリ
material: "[[docs/design/140_observability.jp.md|140_observability.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/metrics_resolution.md|Metrics Viewerの表示粒度]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-26
---
# 可観測性

## 元の材料

[可観測性](../../../design/140_observability.jp.md) blob:debb42a1c062b3cab8b197a52e10aeeaadcc52e4

## 要点

- 実行中の event を Observer が受け、scalar、trace、画像、動画、ログ、プロファイルへ変換する。
- 各 scalar 行はタグ・step・値を持ち、軸名を持たない。タグごとの座標は `metrics.scalar.defs` の `runner` と `step_axis` で確認する。
- 個体単位の trace と集約した scalar は、観測対象と記録のタイミングが異なる。
- Evaluation Trace では、遷移の成立と後から届く採点結果の対応を保つ。
- 画像メモリの寿命と終了時の出力確定を管理し、計測にかかる費用は学習の性能と分けて読む。

## 内容

### 2. コンポーネント定義

- `EpisodeEndEvent` — episode終端group、Agent、Env、Runner、counts、step_resultを運ぶevent。return値はRunner scalarから取得する。
- `SessionEndEvent` — configured Eval の採用 N episode の集約が確定したことを運ぶ event。セッションごとに1回通知する。
- `ObserverFactory` — `metrics.scalar.*`、`metrics.trace.*`、`metrics.graph.*`などのConfigDataからObserverを組み立てるfactory。
- `Notifier`は値を収集せず、event配送だけを担当する。何をどのstepで記録するかはObserver、どこへ保存するかは`MetricsLogger`とbackendの責務である。

### 4. 静的構造

- ObserverはRunner本体のdomain stateを所有せず、eventまたは明示されたProbe/APIから断面を取得する。一方、`MetricsLogObserverBase`のEMAや`GraphVizObserver`のepisode captureなど、観測・集約に必要なstateは各Observerが所有する。runner scopeが必要なObserverはwrapperを介して対象Runnerだけへ絞る。

### 5. 処理フロー

- 重いrender、device同期、I/Oを追加する場合はTrain/Learnのcritical pathへ入ることを前提にprofileする。`EpisodeEvalObserver`のbackground evalは専用poolを使う。worker例外は次のLearn境界または明示`Shutdown()`で呼び出し側へ再送出する。デストラクタ安全網だけは例外を捕捉してFATALを記録する。
- RunnerFrameは学習を停止し、背景評価を排水してから、状態表示用のtrain observerをDetachする。Detach前に始まった通知は解除後もそのObserverを呼びうるため、先に通知を終えてからFrameを破棄する。
- nn_trace は、行動選択に使った forward の途中の出力を可視化へ渡す。EvalPanel は、自分の Runner に付いた Conv2d の pane が `IsPaneOpen` で1枚以上開いているときだけ、タイマー・Step・手動行動に `MakeActionOptions.nn_trace = true` を添える。Trace の記録やダンプだけでは要求しない。

### 6. Metric設定contract

- step軸を省略した場合、`@train`は`train_step`、`@learn`、`@episode_end`、`@session_end`は`exp_step`を使う。scalar JSON recordは`type`、`tag`、`step`、`value`だけを持ち、軸名を保存しない。このため設定変更時は同じtagへ別step軸を流用しない。
- `StepCounts`はRunnerごとのメンバであり、軸名はグローバルに一意な座標を指さない。stepの同一性は「どのRunnerのcounterか」と「どの軸か」の組で決まる。 本書ではこの組をstep座標系と呼ぶ。
- Evaluation Trace では `$agent`・`$env`・`$exp`・`$update_result`は列で使えない。集約prefixを使えるのは`$runner`のepisode_return/episode_stepsだけで、完了しない行ではNaNになる。未知キーは最初の読出しでfail-fast、既知未成立と非有限値は空欄にする。metrics DSLへ`$agent_step`を足すものではない。
- 生成器は行追加と遅延列確定を別々に知らせる。GUIの保持上限を超えた行の画像を保持し続けない。列解析・時点・診断・再生の公開経路と数値等価性はRFD0094検証資材で確かめる。

### 7. 出力とlifetime

- 既知keyだが現在値が成立しない疎なscalarは`NaN`、未知keyは`nullopt`とする。休止中や設定で無効にした機能も既知keyを保って`NaN`を返す。Observerは非有限値をEMA更新前、および複数UpdateResultの平均へ加える前に除外する。後続の有限値は直前までの有限なEMA stateから正常に再開する。
- 開始行は評価の発火時点、つまり background の前セッション完了待ちより前に train thread から出し、終了行は実行した thread から出す。
- TimeHeatMapのScale描画、Histgram、SweepedHeatMapは、wxImage自身が確保した画素領域へ描く。返した画像の寿命が画素領域の寿命を持ち、生成元や元の画像を破棄しても、画像のコピーが残る間は画素を保持する。空のTimeHeatMapは1×1の黒画像を返す。画像を確保できないときは画素へアクセスする前にANET_SYSTEM_ERRORで停止し、描画対象と要求した幅・高さを報告する。

### 8. Profilingと性能上の注意

- 関数全体は`ANET_PROFILE_FUNC()`、通常phaseは`ANET_PROFILE_SCOPE(phase)`を使う。
- 連続phaseは`ANET_PROFILE_SCOPE_NEXT(...)`で切り替え、同じ可視lifetimeで比較できるようにする。
- callbackやasync workerなど自動名が論理処理名にならない場合だけ`ANET_PROFILE_SCOPE_FULL`を使う。
- `interval=1`の重いProbe、画像render、GraphViz、CPU転送は学習throughputへ直接影響する。
- EMA、clip、間引きは表示・保存量を制御するが、元データと同じ意味ではない。分析側へ設定を残す。
- text logのperiodic flushはGUI FPSから分離する。metrics/stdoutまで同じtimerでflushしてI/O頻度を増やさない。

### 9. テストと拡張時の確認事項

- 画像の固定画素・所有権の回帰テストは`heat_map_test.cpp`で確認する。確保・解放の不一致と確保失敗は通常の描画成功だけで判定せず、画像メモリの検証資材で実際のwxWidgets境界を通して確認する。
- scalar tag内にも添字を含められる。たとえば設定key `metrics.scalar.[50_agent_skip/03_skip_length_rate.[01]]` のtagは、内側の閉じ括弧を保持した `50_agent_skip/03_skip_length_rate.[01]` になる。tagの添字は2桁、source keyの添字は`[1]`のままである。
- `metrics.jsonl`の既存readerとMetrics Viewerが新recordを安全に無視または解釈できる。
- 主な回帰testはobservers_test.cpp、metrics_logger_test.cpp、log_test.cpp、episode_end_test.cppに置く。

## 主要な概念

- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [Metrics Viewerの表示粒度](../../concepts/measurement/metrics_resolution.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
