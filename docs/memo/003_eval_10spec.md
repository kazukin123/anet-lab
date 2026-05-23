# EpisodeEvalObserver の Eval メトリクス強化 — 構造分析と修正方針

## Context（背景）

現状、Eval（評価エピソード）から取れているのは「エピソード総報酬」（`eval.[<tag>].eps_total_reward`）だけ。
ユーザーは Eval エピソードの **ENV 統計**（例: `mean.ep_max_rank`）や **Agent 統計** も同じ `metrics.scalar` の枠組みで出したい。

設定仕様を決める前に、まず **「`EpisodeEvalObserver` が独立して動いている影響で既存メトリクスパイプラインに統合しづらい」** という構造的問題を整理し、修正方針を確定するのが本プランの目的。

具体的な実装ステップは別ドキュメント（次の `*_20impl.md` 相当）で起こす想定で、本ファイルは **「構造診断＋方針確定」** までを担う。

---

## 1. 現状クラス構成の事実関係

### 1.1 Observer / Event 系の幹

- `anet::rl::Notifier` が `TrainObserver` / `LearnObserver` を保持し、`TrainEvent` / `LearnEvent` を配る
  （`core/anet-core/include/anet/rl.hpp:745-789, 822-876`）
- `TrainEvent` の payload: `env / experience / step_result / agent / runner / counts / update_result_list / action_info`
- `LearnEvent` は `UpdateEvent` 派生で env 系情報なし
- `EventField` は `EXPERIENCE / AGENT / ENV / UPDATE_RESULT / RUNNER / ACTION_INFO`
  （`core/anet-core/include/anet/rl.hpp:751-758`）

### 1.2 メトリクス出力 Observer

- `MetricsLogObserverBase` が `$agent / $env / $exp / $runner / $update_result` の各 EventField から `GetScalar(key)` を引く統一機構
  （`core/anet-core/src/observers.cpp:722-863`）
- `$env` は **常に `event.runner->GetBatchEnv()` 経由**（`observers.cpp:741`）→ 「イベント発行元 Runner の Env」を見る
- `$agent` は `event.agent`（main Agent。eval / train 共通）
- `MetricsLogTrainObserver` / `MetricsLogLearnObserver` がそれぞれ TrainEvent / LearnEvent を受ける

### 1.3 設定パーサ `ObserverFactory`

`core/anet-core/src/observers.cpp:1007-1167`

```
metrics.scalar.[tag] = <key> [@event] [$target] [$step] [$ema] [interval:N] [ema_alpha:A] [clip:V]
```

- `@event` は **`@train` / `@learn` のみ**
- `$target` は EventField から 1 つ
- **タグでスコープする概念は無い**

### 1.4 Runner ヒエラルキ

- `RunnerBase`（`core/anet-core/include/anet/trainer.hpp:19-69`）
  - `EvalRunner`（`trainer.hpp:76-92`, `src/trainer.cpp:133-213`）
    - **既に存在する Eval 専用 Runner**
    - `DoStep()` 内で `notifier_->Notify(TrainEvent{...})` も既に発火（`trainer.cpp:202-204`）
  - `TrainRunner` → `SerialTrainRunner` / `PipelineTrainRunner`
- `RunnerScopedTrainObserver` / `RunnerScopedLearnObserver` が **イベント発行元 Runner で振り分け**（`rl.hpp:794-816`）
- `RunManager::CreateEvalRunner()`（`trainer.cpp:752-762`）も既存

### 1.5 EpisodeEvalObserver の実体

`core/anet-core/include/anet/observers.hpp:212-246`, `src/observers.cpp:474-572`

- `LearnObserver` 実装
- **自前で**保有・運用:
  - `BatchEnv env_`（`VectorizedDiscreteBatchEnv` を `eval_env_factory` から生成）
  - `Actor actor_`（`agent->CreateActor(..., runmode_, clone=true, actor_device)`）
  - step ループ `RunEvaluationEpisode()`（`observers.cpp:508-528`）
  - スレッドプール `PinnedThreadPool eval_pool_`
- 出力は **`ReportFunction(float total_reward)` 1 callback のみ**

### 1.6 Eval 報酬の現在の伝搬経路

```
EpisodeEvalObserver::RunEvaluationEpisode()                ← Eval 専用ループ（自前）
  └─ report_function_(eps_total_reward)                    ← 単一 float
       └─ TrainRunner::SetEvalLastReward(tag, val)         ← map 保存 (trainer.cpp:228-232)
            └─ TrainRunner::GetScalar("eval.[tag].eps_total_reward")
                 ← key の文字列マッチで分解 (trainer.cpp:265-281)
                    └─ MetricsLogLearnObserver が $runner 経由で取得
```

Eval 結果は **「TrainRunner の特殊キー」→「$runner」** という Eval 専用の側道を通っている。

---

## 2. 構造上の問題点

### 問題 A. EvalRunner と並行実装

`EvalRunner` は `RunnerBase` 派生で env / actor / state / Notify 発火 / GetScalar を備えた**真の Runner**。
にもかかわらず、`EpisodeEvalObserver` は同等責務を**ゼロから別実装**している。

結果:
- `env_` / `actor_` / `state_` は EpisodeEvalObserver の private で **外から覗けない**
- Eval ループから `Notifier::Notify(...)` が **一切飛ばない**（既存メトリクスパイプラインに接続できない）
- 取った `BatchStepResult` / `BatchExperience` がそのまま捨てられる

### 問題 B. Observer の責務二重化

`EpisodeEvalObserver` は「Learn イベント監視（Listener）」と「Eval エピソード実行ドライバ（Runner）」を 1 クラスに同居。
本来後者は Runner の仕事。

### 問題 C. Eval 用イベントが存在しない

- `EventType` は `TRAIN` / `LEARN` のみ
- Eval ループからイベントが飛ばないので `$env` / `$agent` が Eval 文脈を指せない

### 問題 D. Eval タグでルーティングできない

`eval1` / `eval2` / `test1` などが並走する設計だが、**どの Eval から取るかを指定する文法が無い**。

### 問題 E. `ReportFunction(float)` の I/F が単一スカラー固定

Eval 結果の運搬パイプが「float 1 値」しかなく、env / agent の他スカラーを取れない。

### 問題 F. `TrainRunner::GetScalar()` の文字列マッチ分岐

`trainer.cpp:266-281` の `key.find("eval.[")` 分岐は、Eval 結果を訓練 Runner にぶら下げているための **回避的実装**。

---

## 3. 修正方針（確定）

### 3.1 全体コンセプト

> **`EpisodeEvalObserver` を「EvalRunner を間欠的に駆動するトリガ」に痩せさせる。Eval は `EvalRunner` で回し、エピソード終端で専用イベントを発火して既存メトリクスパイプラインに乗せる。**

### 3.2 EpisodeEvalObserver の役割変更（決定: EvalRunner に寄せる）

- env / actor / ループの所有を**やめる**
- `EvalRunner`（または `RunManager` から取得した EvalRunner shared_ptr）を保持
- `OnLearn` でインターバル到達時に `eval_runner->RunOneEpisode()` 相当を呼ぶだけのトリガ
- バックグラウンドスレッド実行・`Sync()` の同期境界はこのクラスに残す

### 3.3 EvalRunner 拡張

- `tag_` フィールド追加（"eval1" 等）
- `RunOneEpisode(RunMode mode)`：エピソード終端まで `DoStep()` を回す API を追加
- エピソード終端で **`EvalEpisodeEndEvent` を発火**
- `GetScalar()`：`eps_total_reward` などエピソード集計値を自身で返せるようにする

### 3.4 Event 拡張（決定: episode end のみ）

- `EventType` に `EVAL` を追加
- `EvalEpisodeEndEvent` を新設
  - payload: `tag`、`env`、`agent`、`counts`、`eval_runner`、エピソード集計値（total_reward 等）
- 既存 `EvalRunner::DoStep()` から出ている `TrainEvent` は今回は触らない（無害なら残置、ノイズなら scope wrapper で抑止）
  - **step ごとの Eval イベントは今回ターゲット外**（StepAxis 解釈の影響が広いため）

### 3.5 Notifier / Observer 拡張

- `Notifier` に `EvalObserver` 系の Attach / Notify を追加
- `RunnerScopedEvalObserver` を新設（タグ別ルーティング）
- `MetricsLogEvalObserver` を新規追加（`OnEvalEpisodeEnd` から `MetricsLogObserverBase::OnUpdate` 相当へつなぐ）

### 3.6 設定文法（決定: 案 A）

```
metrics.scalar.[<output_tag>] = <key> [$target] @eval.[<eval_tag>] [$step] [$ema] [interval:N] [ema_alpha:A] [clip:V]
```

- `@eval.[<eval_tag>]` を新規予約語として `ObserverFactory` に追加
- 既存 `@train` / `@learn` と同列のイベントソース指定
- `$env` / `$agent` / `$runner` / `$exp` はそのまま利用可。指す対象は `@eval.[<tag>]` で指定された Eval Runner の文脈

利用例:

```
# Eval Env の統計（カスタム指標）
metrics.scalar.[42_eval/01_mean_rank]   = mean.ep_max_rank $env @eval.[eval1] $ema

# Eval エピソード総報酬（旧 $runner 経由を置き換え）
metrics.scalar.[42_eval/02_eps_reward]  = eps_total_reward $runner @eval.[eval1]

# Eval 時 Agent 方策統計
metrics.scalar.[42_eval/03_eval_eps]    = eval_policy.epsilon $agent @eval.[eval1] $ema
```

### 3.7 後方互換（決定: 完全移行）

- 既存 `eval.[tag].eps_total_reward` キー＆ `TrainRunner::GetScalar` の文字列分岐（`trainer.cpp:265-281`）は **削除**
- `TrainRunner::SetEvalLastReward` / `eval_last_rewards_` も **撤去**
- `apps/runner/config/*.txt` の旧 `metrics.scalar.[...] = $runner eval.[...].eps_total_reward @learn` 行は新文法に書き換え
- `ReportFunction` 経路も撤去（または最小化）

---

## 4. 修正対象ファイル

### 中心となる変更

- `core/anet-core/include/anet/rl.hpp`
  - `EventType::EVAL` 追加
  - `EvalEpisodeEndEvent` / `EvalObserver` 定義
  - `Notifier` に Attach/Notify(EvalObserver) 追加
  - `RunnerScopedEvalObserver` 追加
- `core/anet-core/include/anet/trainer.hpp` / `src/trainer.cpp`
  - `EvalRunner::tag_` 追加
  - `EvalRunner::RunOneEpisode()` 追加（既存 `EpisodeEvalObserver::RunEvaluationEpisode` のロジックを移植）
  - `EvalRunner::GetScalar()` に `eps_total_reward` 等の集計値を実装
  - `TrainRunner::eval_last_rewards_` / `SetEvalLastReward` / `GetScalar` の eval 分岐を **削除**
  - `RunManager` で Eval ごとの EvalRunner 生成・保持（既存 `CreateEvalRunner` を活用）
- `core/anet-core/include/anet/observers.hpp` / `src/observers.cpp`
  - `EpisodeEvalObserver` を EvalRunner ベースに痩せさせる（env / actor / pool は呼出元 or RunManager 管理へ）
  - `MetricsLogEvalObserver` 新規追加
  - `ObserverFactory` のパースに `@eval.[<tag>]` を追加し、`MetricsLogEvalObserver` を生成
  - 生成された `MetricsLogEvalObserver` は `RunnerScopedEvalObserver` で対象 EvalRunner にスコープ

### 設定書き換え

- `apps/runner/config/LunarLander.txt` `:56-59`
- `apps/runner/config/ImageCls.txt` `:178-179`
- `apps/runner/config/GridMaze.txt` `:158-161`（コメントアウト中）
- `apps/runner/config/DropMerge.txt` `:790-793`（コメントアウト中）

---

## 5. 既存利用可能な部品（再利用先）

- `EvalRunner` 自体（`trainer.hpp:76-92`）— 単一 env / actor / step ループの実装は流用
- `RunManager::CreateEvalRunner()`（`trainer.cpp:752-762`）— Eval Runner ファクトリ
- `RunnerScopedTrainObserver` / `RunnerScopedLearnObserver` の実装パターン — `RunnerScopedEvalObserver` のひな型
- `MetricsLogObserverBase::OnUpdate` / `GetMetricsDataList`（`observers.cpp:812-909`）— EMA / clip / interval ロジックは流用可
- `PinnedThreadPool`（`thread.hpp`）— バックグラウンド評価実行に利用

---

## 6. 検証方針

1. **Debug ビルド**
   ```powershell
   cmake --build --preset x64-Debug
   ```
   `EpisodeEvalObserver` / `EvalRunner` 周りのリンクが通ること。
2. **anet-core-test 実行**
   ```powershell
   core\anet-core\bin\Debug\anet-core-test.exe
   ```
   既存テスト（DQN Agent / Replay 系等）に retro なし。
4. **複数 Eval の独立性**
   - `eval1` / `eval2` 並走時、`@eval.[eval1]` と `@eval.[eval2]` のメトリクスが混ざらないこと
5. **エラー系**
   - 存在しないタグ（`@eval.[nonexistent]`）指定時に起動時エラーで弾く（AGENT.md「設定値の扱い」の方針に従う）

---

## 7. 残課題（次フェーズ）

本プランは構造方針までを担う。具体的な実装ステップは別途 `*_20impl.md` 相当として:

- EvalRunner 拡張の段取り（先に既存テスト追加 → API 拡張 → 旧ロジック移植）
- Notifier / EvalObserver の I/F 詳細（payload 構造、tag 引き回し）
- `MetricsLogEvalObserver` の OnUpdate 統合（既存 `MetricsLogObserverBase` を継承するか、`OnEvalEpisodeEnd` 専用 base を新設するか）
- 旧 `eval.[tag].eps_total_reward` の削除タイミングと config 一括書き換え

を詰める。

---

> **Note（2026-05-21）**
> 本書は初期の構造診断＋方針メモ。実装フェーズで議論を経て、
> - Eval 専用イベント案は不採用 → **Train/Eval 共通 `EpisodeEndEvent`** に統一
> - 設定文法は `@eval.[<tag>]` 案ではなく **3 軸直交（`@<event>` × `$<runner_scope>` × `$<target_field>`）** を採用
> - `EvalRunner` の識別子は `tag` ではなく `name` を使う（`tag` は TensorBoard メトリクス識別子として温存）
>
> へ方針更新済み。**最終仕様および実装計画は `docs/specs/003_eval_20impl.md` を参照すること。**
