# 実行基盤と設定

## 元の材料

[実行基盤と設定](../../../design/100_runtime_and_configuration.jp.md) blob:bf257f72821ba97096f394f85e90b00161e663df

## 要点

- 設定の解決から Env・Agent・Runner・Observer の構築、Run の開始と終了までを扱う。
- 実効設定を config_data.txt に、解決の経緯を構造化された記録に残す。
- Serial と Pipeline は経験を渡すタイミングが異なる。Pipeline では直前の経験に対する学習と次の環境ステップが並行する。
- 初回は学習に渡せる経験がなく、Pipeline の学習結果の回収と通知は後続のステップになる。
- 評価セッションと学習の step 軸、Observer の寿命、停止時に確定する出力を区別する。

## 内容

### 2. 基本概念と外部contract

- `ConfigResolver`がCLIを解決入力へ反映し、`run.$`を通常selectionより先に展開する。Runの項は左から右へ後勝ちでrootへ供給し、同じキーのCLIをさらに優先する。Run内の`Env.$ = @base`は展開先rootを定義位置とする。
- `linear`・`cosine` は `start`・`end`・`steps`、`cosine_restart` はさらに `cycle_mult`、`exponential` は `start`・`end`・`time_constant` を使う。rootの指数は `end` に漸近し続ける。
- `ControlSignal`はframe内継続、frame打切り、Runner停止をpre/post callbackから返す。

### 3. コンポーネント定義

- `ConfigManager` — main file、注入値、後勝ちoverlay、CLI overrideを収集し、resolverの結果を公開する。
- `ConfigResolver` — source mapからselection、CLI leaf、値参照を順に解決し、実効ConfigDataとresolution JSONを作るprivate deep module。
- `Config` — default/override prefixを使い、1コンポーネントの型付き設定を読む基底。
- `RunManager` — seed、Env、Agent、Notifier、Runnerの構築とRun内共有objectを管理する。

### 5. 静的構造

- 主Train EnvはTrainRunnerが使用し、EvalRunnerごとに別のEnvとActorを作る。AgentとNotifierはRun内で共有される。
- `RunManager`はprivateな入れ子`MetricsWiring`を所有し、Run構築時にscope解決、dormant tagのRunごとに一度だけのINFO、event別のObserver登録、attach済みのscalar/trace定義の出力、scalar購読ヒントの通知を委譲する。宣言はtrainer.hppのRunManager内部、定義はtrainer.cppに置く。`ObserverFactory`の解析と、Observerによる値の取得・集約は別の責務である。構築後の鏡写しinstanceへのtrace配線は本機構には含めない。

### 6. 処理フロー

- 構築中に型変換、EnvSpec、device、class ID、Env name衝突、または各`Config`の不整合を検出した場合は、RunnerThread開始前に失敗する。固定名`train`、全configured Eval tag、予約名`EvalPanel`は最初のBatchEnv構築前に一括検証する。型変換失敗時の契約は設定の解決のとおりである。
- `Learner`は`UpdateFromBatch()`で学習を実行し、`BatchUpdateResultList`を戻す。`LearnEvent`はLearner自身が発火するEventではなく、戻り値を受けた`SerialTrainRunner`が構築して`Notifier`へ通知する。`Notifier`からObserverへのcallbackも同じRunnerThread上で同期実行される。

### 7. 設定・lifetime・エラー・性能特性

- 手動closeは新しい評価投入との競合を避けるため、Train threadをStop/joinしてから`WillBlockOnShutdown()`を照会する。Keep runningでは同じthreadを従前のpause状態で再開する。予算到達はveto不可のため質問しない。
- process singleton repositoryはfactoryを保持するが、Run固有のAgent、Env、Runnerを保持しない。
- エラー・WARN・INFO が設定キーを示すときは、値を与えた置き場の完全キーを使う。閉じたスキーマの未読キーも完全キーで示す。
- RunnerThread内の例外は握りつぶさずapplicationのexception callbackへ渡す。
- `train_step_per_sec`と`exp_step_per_sec`はbatch sizeの意味が異なる。比較時は同じ設定とstep軸を使う。

### 8. テストと拡張時の確認事項

- RFD0117検証資材: 元キーの追跡、閉じたカタログ、未読一覧、Run単位の通知、通常RunのWARN 0件の対象と停止条件。
- RFD0194検証資材に、診断が示す設定キーの集計と期待一覧との照合手順を置く。
- RFD0094検証資材: 列、時点、診断、再生、snapshot、5構成の数値等価性。
- 現行`trainer_test.cpp`はSerial/Pipeline全体、count、shutdownを広く覆うものではない。これらを変更する場合は、Serial/Pipelineのaction/snapshot境界が一致すること、B=1と複数laneのcount、Evalのscope、停止時のworker回収を対象とする回帰testを追加して確認する。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [RunnerApp](../../concepts/apps/110_runner_app.md)
