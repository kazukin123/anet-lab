# 可観測性

> 主たる観点: 機能単位（Event、Observer、metrics、log、profile、artifact）

## 1. はじめに

### 1.1 目的

本書は、ANETの実行状態をeventからscalar、画像、動画、GraphViz、text log、profileへ変換する仕組みを説明する。計測処理を学習本体から分離し、出力の意味とstep軸を追跡できることを目的とする。

### 1.2 対象読者

- metric、Observer、可視化を追加・変更する開発者
- `metrics.jsonl`とRun内artifactの生成経路を確認する開発者
- logging、flush、profilingの性能・lifetime境界をレビューする担当者

### 1.3 記載範囲

現行の`Notifier`、Observer群、`ObserverFactory`、`MetricsLogger`、runner text log、profile macroを扱う。Metrics ViewerのUI操作は[分析ユーザーガイド](030_user_guide_analysis.jp.md)、application構造は[アプリケーションとツール](160_applications_and_tools.jp.md)を参照する。

## 2. コンポーネント定義

| コンポーネント | 定義 |
|---|---|
| `TrainEvent` | Env step後のExperience、ActionInfo、Env、Agent、Runner、countsを運ぶevent |
| `LearnEvent` | Learner更新後のExperience、UpdateResult、Agent、Runner、countsを運ぶevent |
| `EpisodeEndEvent` | episode終端lane、累積reward、Agent、Env、Runner、countsを運ぶevent |
| `Notifier` | 3種類のObserverを登録し、対応eventを同期的に配信するRun内hub。`RunManager`がTrain/Eval間で共有する |
| Runner-scoped Observer | Trainまたは特定Eval Runnerのeventだけを実Observerへ通すwrapper |
| `ObserverFactory` | `metrics.scalar.*`、`metrics.graph.*`などのConfigDataからObserverを組み立てるfactory |
| `MetricsLog*Observer` | event内の指定sourceからscalarを取得し、step選択、interval、EMA、clipを適用するObserver |
| `EpisodeEvalObserver` | Learn eventを契機にconfigured Evalを同期またはbackgroundで駆動するObserver |
| Image/Graph Observer | ProbeやNN出力からHeatMap、TimeHistogram、Conv2d、GraphVizを生成するObserver |
| `MetricsLogger` | Run名とRun directoryを所有し、scalar、JSON、画像、動画、DOTを共通形式で保存するsingleton |
| `IBackend` / `JsonlBackend` | metric recordの永続化境界。現行backendは`metrics.jsonl`へ追記する |
| `anet::log::Logger` | 構築時に確定したprefixを先頭へ書き込んだ`WxLogStream`を生成する軽量logger |
| `FileLogger` | wxLogをUTF-8の`<run_name>.log`へ複製し、warning以上を即時flushする |
| `StandardStreamLogger` | GUI processのstdout/stderrをRun directoryへ退避する |
| `ProfileRange`とmacro | 関数・処理phaseをCPU/GPU profilerへ安定名で記録する計測境界 |

`Notifier`は値を収集せず、event配送だけを担当する。何をどのstepで記録するかはObserver、どこへ保存するかは`MetricsLogger`とbackendの責務である。

## 3. コードマップ

| 領域 | 主なファイル |
|---|---|
| event、Observer interface、Notifier | [rl.hpp](../../core/anet-core/include/anet/rl.hpp)、[rl.cpp](../../core/anet-core/src/rl.cpp) |
| 具象Observer、Config parser | [observers.hpp](../../core/anet-core/include/anet/observers.hpp)、[observers.cpp](../../core/anet-core/src/observers.cpp) |
| MetricsLogger contract | [metrics_logger.hpp](../../core/anet-core/include/anet/metrics_logger.hpp) |
| JSONL、画像、動画、DOT出力 | [metrics_logger.cpp](../../core/anet-core/src/metrics_logger.cpp) |
| text logging | [log.hpp](../../core/anet-core/include/anet/log.hpp) |
| stdout/stderr capture | [app_util.hpp](../../core/anet-core/include/anet/app_util.hpp)、[app_util.cpp](../../core/anet-core/src/app_util.cpp) |
| profiling | [profile.hpp](../../core/anet-core/include/anet/profile.hpp)、[profile.cpp](../../core/anet-core/src/profile.cpp) |
| scalar metric設定例 | [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) |
| image/graph metric設定例 | [metrics_image.txt](../../apps/runner/config/metrics_image.txt) |
| runner側の初期化・flush | [RunnerApp.cpp](../../apps/runner/src/RunnerApp.cpp) |

## 4. 静的構造

```mermaid
classDiagram
direction LR

class Runner
class Notifier {
  +Attach(observer)
  +Detach(observer)
  +Notify(event)
}
class TrainObserver
class LearnObserver
class EpisodeEndObserver
class ObserverFactory
class MetricsLogObserverBase
class MetricsLogTrainObserver
class MetricsLogLearnObserver
class MetricsLogEpisodeEndObserver
class ImageObserver
class GraphVizObserver
class MetricsLogger {
  +LogScalar(tag, step, value)
  +Log(tag, data)
  +Flush()
}
class IBackend
class JsonlBackend
class RunArtifacts

Runner --> Notifier : eventを通知
Notifier o-- TrainObserver
Notifier o-- LearnObserver
Notifier o-- EpisodeEndObserver
ObserverFactory ..> TrainObserver : Configから生成
ObserverFactory ..> LearnObserver : Configから生成
ObserverFactory ..> EpisodeEndObserver : Configから生成
MetricsLogObserverBase <|-- MetricsLogTrainObserver
MetricsLogObserverBase <|-- MetricsLogLearnObserver
MetricsLogObserverBase <|-- MetricsLogEpisodeEndObserver
TrainObserver <|.. MetricsLogTrainObserver
LearnObserver <|.. MetricsLogLearnObserver
EpisodeEndObserver <|.. MetricsLogEpisodeEndObserver
TrainObserver <|.. ImageObserver
TrainObserver <|.. GraphVizObserver
MetricsLogObserverBase --> MetricsLogger
ImageObserver --> MetricsLogger
GraphVizObserver --> MetricsLogger
MetricsLogger *-- IBackend
IBackend <|.. JsonlBackend
JsonlBackend --> RunArtifacts : metrics.jsonl
MetricsLogger --> RunArtifacts : json/video/image/dot
```

ObserverはRunner本体のdomain stateを所有せず、eventまたは明示されたProbe/APIから断面を取得する。一方、`MetricsLogObserverBase`のEMAや`GraphVizObserver`のepisode captureなど、観測・集約に必要なstateは各Observerが所有する。runner scopeが必要なObserverはwrapperを介して対象Runnerだけへ絞る。

## 5. 処理フロー

### 5.1 scalar metricの記録

```mermaid
sequenceDiagram
    participant R as Train/Eval Runner
    participant N as Notifier
    participant O as MetricsLog Observer
    participant S as Agent/Runner/Env/Experience/Result
    participant M as MetricsLogger
    participant J as JsonlBackend
    participant F as metrics.jsonl

    R->>N: Notify(Train/Learn/EpisodeEnd event)
    loop 登録順の対応Observer
        N->>O: OnTrain / OnLearn / OnEpisodeEnd
        O->>O: runner scopeとstep軸を確認
        O->>S: GetScalarまたはevent fieldを取得
        S-->>O: optional scalar
        opt 値があり記録対象
            O->>O: EMAを更新
            O->>O: interval・finite値を確認し、clipを適用
            O->>M: LogScalar(tag, step, value)
            M->>J: WriteJsonl(record)
            J->>F: 1 JSON recordを追記
        end
    end
```

Observer callbackは`Notify()`を呼んだthread上で実行される。重いrender、device同期、I/Oを追加する場合はTrain/Learnのcritical pathへ入ることを前提にprofileする。`EpisodeEvalObserver`のbackground evalは専用poolを使う例外であり、完了時の例外は次の境界で呼び出し側へ再送出する。

### 5.2 Run終了時の出力確定

```mermaid
sequenceDiagram
    participant UI as RunnerFrame
    participant A as RunnerApp
    participant T as RunnerThread
    participant G as Agent
    participant M as MetricsLogger
    participant L as wxLog/FileLogger

    UI->>A: StopTraining()
    A->>T: Stop()
    UI->>A: SaveAgent(agent_close.anet)
    A->>G: Save(archive)
    UI->>A: ShutdownRunLogging()
    A->>A: periodic text-log timerを停止
    A->>M: Flush()
    A->>L: FlushActive()
    A->>L: chainをdetachしてclose
```

periodic timerは`RunName.log`だけをflushする。metrics、stdout、stderrはpause、save、shutdownなどの明示的な`FlushRunOutputs()`境界でまとめてflushする。

## 6. Metric設定contract

scalar定義の基本形は次である。

```text
metrics.scalar.[tag] = key [$step_axis] [@event] [$target] [$runner_scope] [$ema] [interval:N] [ema_alpha:A] [clip:C]
```

| 要素 | 主な値 | 意味 |
|---|---|---|
| `@event` | `@train`、`@learn`、`@episode_end` | Observerを呼ぶevent。省略時は`@train` |
| `$step_axis` | `$train_step`、`$learn_step`、`$episode_step`、`$exp_step`、`$update_step`、`$sim_step` | JSONLの`step`へ使うcounter |
| `$target` | `$runner`、`$agent`、`$env`、`$exp`、`$update_result`、`$action_info` | 値を取得するsource |
| `$runner_scope` | `$train`、`$eval.[name]` | eventを発生させたRunnerを限定する |
| `$ema` | - | Observer内でEMAを計算する |
| `ema_alpha:A` | 0より大きく1以下のfinite値を指定する | EMAが新しい値へ寄る係数を指定する |
| `interval:N` | 1以上の整数を指定する | eventを間引く |
| `clip:C` | 0以上のfinite値を指定する | 記録前に値を`[-C, C]`へclipする |

step軸を省略した場合、`@train`は`train_step`、`@learn`と`@episode_end`は`exp_step`を使う。scalar JSON recordは`type`、`tag`、`step`、`value`だけを持ち、軸名を保存しない。このため設定変更時は同じtagへ別step軸を流用しない。

`$ema`は、ゼロ初期化した内部値と観測済み重み和を同じ`ema_alpha`で更新し、内部値を重み和で正規化するバイアス補正EMAを使う。初回サンプルから欠損なく値を出力し、途中で`ema_alpha`が変わっても観測済み重み和に基づく補正を継続する。

`interval:N`は出力判定であり、値取得とEMA状態更新の後に適用する。このため初回Learner priority更新のような疎な値も、`$ema interval:100`では各eventのfinite値でEMAを更新し、記録だけを100 step間隔へ抑えられる。sourceが返す`NaN`は非イベントまたは統計未成立を表し、0としてEMAへ投入しない。

不明なevent、step軸、targetにはWARN後に既定値を使う経路があり、対応しないEval scope/fieldの組み合わせはfail-fastする。特にEval scopeは現行contractで`@episode_end`、またはEvalの`@train $action_info`に限定される。

`$agent`、`$action_info`、`$update_result`で取得できるkeyは、共通interfaceと具象Agentが公開するmetricの組合せで決まる。対応しないkeyを全Agentで同じ値に見せることはせず、Observer側は`std::optional`や`NaN`の意味をmetric定義ごとに扱う。DefaultDQNのTrain Actor snapshot診断など、Agent固有keyの意味は[DQN系Agent](200_dqn_agents.jp.md)を参照する。

IQN診断ではdevice同期をmetric keyごとに発生させない。Policy診断は複数scalarをdetached packed Tensorへまとめ、`DQNActionInfo`が最初のkey参照時だけCPUへmaterializeする。Learnerの既存IQN診断はPER有効時にpriority readbackへ同梱し、PER無効時も固定長の診断packだけを既存の非同期readback経路で回収する。`metrics.scalar.iqn_search_p0`は`metrics.scalar.full`全体を有効化せず、PER健全性とthroughputのP0選抜だけを合成するgroupである。

QR / IQNの分位tail診断も同じ同期境界を使う。Policy側5 scalarはper-action上下幅、detached full quantile alias、globalなdisagreement / crossingを共有する。最初の参照時だけ最終actionをgatherし、positive crossing深度のlane別nearest-rank p90をdevice上で選んで、全5値を1本のCPU cacheへまとめる。action生成時とcache再参照時にはpercentile sortを行わず、`WithAction()`後はcacheだけを破棄する。Learner側のsample単位upper-tail幅はPER有効時だけ既存priority readbackへ同梱し、CPU上でclip後raw priorityとのSpearman相関へ集約する。PER無効時は追加packも追加waitも作らず`NaN`を返す。tail入力はfloat32へdetachし、loss、priority、action、sampling、RNGへ接続しない。

現行parserは`interval`、`ema_alpha`、`clip`を`stoi`/`stof`で変換する。`ema_alpha`は変換後に`EmaFilter`がfiniteかつ`0 < ema_alpha <= 1`を検証する。`interval`と`clip`は範囲やfinite性を検証しないため、`interval=0`や負の`clip`は指定しない。数値文字列の変換失敗は構築中の例外になる。

## 7. 出力とlifetime

| 出力 | 生成主体 | 更新・flush境界 |
|---|---|---|
| `metrics.jsonl` | `JsonlBackend` | scalar/metadataごとに追記し、明示flushで確定 |
| `config/*.txt` | `MetricsLogger` | ConfigをLogした時点でtag別に書出し。Envは`env.<Env name>.txt` |
| `json/*.json` | `MetricsLogger` | JSON metadataをLogした時点で上書きまたはstep別生成 |
| `videos/<tag>.mkv` | `VideoLogger` | 最初のframeでloggerを作り、Run終了時にclose |
| `images/<tag>/*.png` | `MetricsLogger` | `use_png_dump=true`時にframeごとに生成 |
| `dot/**/*.dot` | `MetricsLogger` | GraphViz eventごとに生成 |
| `<run_name>.log` | `FileLogger` | periodic timer、warning以上、明示flush |
| `stdout.log` / `stderr.log` | `StandardStreamLogger` | process標準streamをcaptureし、明示flush/停止 |

`MetricsLogger`はprocess singletonだが、1 processで1 active Runを前提にRun directoryを所有する。`Reset()`はRun終了時にsingletonを解放する。動画loggerやwxLog chainをfile利用中に破棄しないよう、applicationのshutdown順序を維持する。

具象Env本体のtext logは`<Env name>: `を先頭へ付け、Train、configured Eval、EvalPanelとbatch laneの出力元を人間が識別できるようにする。`SingleDiscreteEnvBase`と`BatchEnvBase`がprotectedな`anet::log::Logger log`を保持し、具象Envは`log.info()`、`log.verbose()`、`log.warn()`、`log.error()`を使う。Env本体で`LOG::`を直接使わず、prefix書式や`GetName()`連結を各ログ行へ分散させない。Env外のfactory、free関数、Runner、Agent、Viewは従来どおり`LOG::`を使用する。

debug logは`ANET_LOG_DEBUG_PREFIXED(expr)`を使用する。このmacroは`ANET_LOG_DEBUG(log.prefix() << expr)`へ委譲し、デバッガ接続・level guard、source情報、`ANET_ENABLE_DEBUG_LOG=0`での式非評価を維持する。Env nameは表示専用の不透明な文字列であり、`MetricsLogger`のtag、JSONL field、artifact path、runner scopeを変更・代替しない。Viewは共通Env accessorから表示に利用できるが、nameをEnv挙動やmetric identityの分岐へ使用しない。

## 8. Profilingと性能上の注意

- 関数全体は`ANET_PROFILE_FUNC()`、通常phaseは`ANET_PROFILE_SCOPE(phase)`を使う。
- 連続phaseは`ANET_PROFILE_SCOPE_NEXT(...)`で切り替え、同じ可視lifetimeで比較できるようにする。
- callbackやasync workerなど自動名が論理処理名にならない場合だけ`ANET_PROFILE_SCOPE_FULL`を使う。
- `interval=1`の重いProbe、画像render、GraphViz、CPU転送は学習throughputへ直接影響する。
- EMA、clip、間引きは表示・保存量を制御するが、元データと同じ意味ではない。分析側へ設定を残す。
- text logのperiodic flushはGUI FPSから分離する。metrics/stdoutまで同じtimerでflushしてI/O頻度を増やさない。

## 9. テストと拡張時の確認事項

Observerまたは出力形式を変更する場合は次を確認する。

1. event、runner scope、step軸、targetの組み合わせが明示されている。
2. callbackが必要以上のclone、device同期、I/Oを追加していない。
3. 同じtagの型とstep軸をRun途中で変えない。
4. background処理の例外とshutdown待機が失われていない。
5. `metrics.jsonl`の既存readerとMetrics Viewerが新recordを安全に無視または解釈できる。
6. Run close後にfile handle、timer、ffmpeg processが残らない。

主な回帰testは[observers_test.cpp](../../core/anet-core/src/observers_test.cpp)、[metrics_logger_test.cpp](../../core/anet-core/src/metrics_logger_test.cpp)、[log_test.cpp](../../core/anet-core/src/log_test.cpp)、[episode_end_test.cpp](../../core/anet-core/src/episode_end_test.cpp)に置く。

## 10. 関連文書

- [Run実行ユーザーガイド](020_user_guide_run.jp.md)
- [Run分析ユーザーガイド](030_user_guide_analysis.jp.md)
- [実行基盤と設定](100_runtime_and_configuration.jp.md)
- [Agentと学習](110_agents_and_learning.jp.md)
- [環境](120_environments.jp.md)
- [ReplayBuffer](150_replay_buffer.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)
- [DQN系Agent](200_dqn_agents.jp.md)
- [DropMerge Optuna利用ガイド](../optuna.md)
