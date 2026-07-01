# MetricsLogger シングルトン・ライフサイクル堅牢化 — null-safe 静的ログ API と Reset 順序 PRD

> 関連: `core/anet-core/src/observers.cpp`(observer ログ), `core/anet-core/include/anet/metrics_logger.hpp`(Instance/Reset), `core/anet-core/include/anet/rl.hpp`(Notifier/Observer), `apps/runner/src/RunnerApp.cpp`(OnExit)。
> クラッシュ診断(2026-07-01)の成果物。PRD `025`(VideoLogger/ffmpeg)とは別テーマ＝MetricsLogger シングルトンのライフサイクル。実装は別途(Codex 想定)、本書は self-contained に記述する。

## Context / Problem Statement

`this==nullptr` で `MetricsLogger::LogScalar` が読み取りアクセス違反でクラッシュする。スタック（eval スレッド）:

```
MetricsLogger::LogScalar                      metrics_logger.hpp:131 (backend_->WriteJsonl)
MetricsLogObserverBase::OnGenericUpdate       observers.cpp:966 (Instance()->LogScalar)
MetricsLogEpisodeEndObserver::OnEpisodeEnd
RunnerScopedEpisodeEndObserver::OnEpisodeEnd
Notifier::Notify(EpisodeEndEvent)
RunnerBase::AccumulateAndNotifyEpisodeEnd
EvalRunner::DoStep
EpisodeEvalObserver::RunEvaluationEpisode
EpisodeEvalObserver::OnLearn::<lambda>()      (背景プール上)
```

**確定した原因（実証済み）:**

1. `MetricsLogger::Instance()` は `shared_ptr<MetricsLogger>` を**値返し**する（[metrics_logger.cpp:467-470](../../core/anet-core/src/metrics_logger.cpp)）。本来はローカルに受けて使えば呼び出し中の生存が保証される。だが [observers.cpp:966](../../core/anet-core/src/observers.cpp) 等の呼び出し側は戻り値を受けず `Instance()->LogScalar(...)` と**直呼び**している。
2. `EpisodeEvalObserver::OnLearn` は評価エピソードを**背景プール `eval_pool_`** に投げる（[observers.cpp:537](../../core/anet-core/src/observers.cpp)）。エピソードは終端まで多数ステップ走るため、しばらく実行が続く。
3. その in-flight エピソードが終了時に episode-end 通知経由でログ出力する。ところが `MetricsLogger::Reset()`（[RunnerApp.cpp OnExit](../../apps/runner/src/RunnerApp.cpp)）が `instance_` を先に null 化していると、`Instance()` は**null shared_ptr** を返し、`->LogScalar` が nullptr に対して呼ばれて `this==nullptr` → クラッシュ（`LogScalar` 本体で `backend_->` を触った瞬間、[metrics_logger.hpp:131](../../core/anet-core/include/anet/metrics_logger.hpp)）。
4. `eval_pool_` の排水は `~EpisodeEvalObserver`（`eval_future_.wait()`+`eval_pool_->Stop()`、[observers.cpp:500-511](../../core/anet-core/src/observers.cpp)）**のみ**＝`run_manager_` 破棄時＝`OnExit` の `Reset()` **より後**。→ Reset と eval が競合する。

Optuna の run 切替（`Reset()`+`Init()`）でも同根で再発し得る。本質は「**Reset 可能なシングルトンを背景スレッドから `Instance()->X()` 直呼びする**」無防備さ（systemic）。

## ゴール / 非ゴール

**ゴール:**
- **G1**: `MetricsLogger::Instance()` が null（Reset 後 / 未 Init）でも、ログ呼び出しがクラッシュせず「**安全に no-op**」になる。
- **G2**: 呼び出し側を煩雑にしない **null チェック内包の入口**を `MetricsLogger` の **public static メソッド**として用意し、全ログ呼び出しをそれ経由に統一する。`MetricsLogger` を唯一の入口に保つ。
- **G3**: シャットダウン時、`Reset()` の**前**に背景 observer（eval プール）を排水し、「**logger は全ユーザより長生き**」というライフサイクル不変を回復する。

**非ゴール:**
- **NG1**: 値返しアクセサ（`GetRunDir` / `GetRunName` / `GetNotifier` / `GetRunManager`）は対象外。main/setup スレッドから Reset 前に呼ばれるため。no-op 化にもなじまない（値を返す）。必要なら別途。
- **NG2**: PRD `025`（VideoLogger/ffmpeg）とは独立。混ぜない。

## 要件

### R1: null-safe 静的ログ API（G1 / G2）

`MetricsLogger` を唯一の入口に保つため、**public static** なログメソッドを追加し、`Instance()` の null チェックを内包する。現在の public インスタンスログメソッドは **private 実装（`*Impl`）へ改名**（本体不変）し、static がそこへ転送する。

```cpp
// metrics_logger.hpp, class MetricsLogger
public:
    // null-safe 静的入口（Instance() が null=Reset 後/未 Init なら no-op）
    static void LogScalar(const std::string& tag, int64_t step, double value) {
        if (auto m = Instance()) m->LogScalarImpl(tag, step, value);
    }
    static void Log(const std::string& tag, anet::rl::step_t step,
                    const anet::ImageSource& src, int w = -1, int h = -1) {
        if (auto m = Instance()) m->LogImpl(tag, step, src, w, h);
    }
    static void Log(const std::string& tag, anet::rl::step_t step, const wxImage& img) {
        if (auto m = Instance()) m->LogImpl(tag, step, img);
    }
    // … 使う Log オーバーロード分（json / step+json / Config / ConfigData / GraphViz / step+GraphViz）を用意
private:
    // 旧 public インスタンスメソッドを private 実装へ改名（本体不変・.cpp 定義側も改名）
    void LogScalarImpl(const std::string& tag, int64_t step, double value);   // 旧 LogScalar(inline)
    void LogImpl(const std::string& tag, anet::rl::step_t step,
                 const anet::ImageSource& src, int w, int h);                  // 旧 Log(…ImageSource…)
    // … 各 Log オーバーロードの Impl。MetricsLogger 内部の相互呼び出し（例 Log(Config)→Log(tag,Config)）も Impl 名へ
```

- **static は private impl へ転送するだけ**（`Instance()` / `Reset()` は不変）。実際に使われている Log オーバーロードだけ static/impl を用意すればよい。
- **移行（systemic・機械的）**: 全 `MetricsLogger::Instance()->Log(...) / ->LogScalar(...)` を `MetricsLogger::Log(...) / MetricsLogger::LogScalar(...)` へ置換（＝`Instance()->` を除去）。代表パス: `core/anet-core/src/observers.cpp`(~8 箇所), `default_dqn_agent.cpp`, `image_cls_agent.cpp`, `muzero_proto_agent.cpp` / `muzero_based_agent.cpp`, `image.cpp`, `env.cpp`, `core/envs/*/src/*Env.cpp`, `apps/runner/src/RunnerApp.cpp`。
- インスタンスメソッドを private 化するため、**全ログ呼び出し（テスト含む）の static 形への移行が必須**（部分移行はコンパイルエラー＝漏れが即検出できる）。これにより「ログの入口は `MetricsLogger::` static のみ」が型で保証される。
- （任意）`Flush()` も同様に static null-safe 化してよい（`MetricsLogger::Flush()`。main スレッド呼び出しで低リスクだが一貫性のため）。

### R2: Reset 前の背景 observer 排水（G3）

- observer 基底 3 種（`TrainObserver` / `LearnObserver` / `EpisodeEndObserver`、[rl.hpp:825/832/839](../../core/anet-core/include/anet/rl.hpp)。共通基底なし）に `virtual void Shutdown() {}`（default no-op）を追加。
  - ※ `Runner::Shutdown()`([rl.hpp:988](../../core/anet-core/include/anet/rl.hpp)) / `BatchEnv::Shutdown()`([rl.hpp:639](../../core/anet-core/include/anet/rl.hpp)) とは**別クラスの別物**（同名だが無関係）。
- `EpisodeEvalObserver`（`LearnObserver` 派生）が override（`~EpisodeEvalObserver` の中身を抽出し冪等化）:
  ```cpp
  void Shutdown() override {
      if (eval_future_.valid()) eval_future_.wait();
      if (eval_pool_) { eval_pool_->Stop(); eval_pool_.reset(); }
  }
  ~EpisodeEvalObserver() override { Shutdown(); }   // 二重停止で安全
  ```
- `Notifier::Shutdown()` を追加（全 observer の `Shutdown()` を呼ぶ。[rl.hpp:890 Notifier](../../core/anet-core/include/anet/rl.hpp) / rl.cpp 実装）:
  ```cpp
  void Notifier::Shutdown() {
      for (auto& o : learn_observers_)       o->Shutdown();
      for (auto& o : train_observers_)       o->Shutdown();
      for (auto& o : episode_end_observers_) o->Shutdown();
  }
  ```
- `RunnerApp::OnExit()` を順序変更（`trainer_thread_->Stop()` の後、`MetricsLogger::Reset()` の**前**に挿入）:
  ```cpp
  trainer_thread_->Stop();
  run_manager_->GetNotifier()->Shutdown();   // ← eval プール排水（Reset より前）
  standard_stream_logger_.Stop();
  anet::MetricsLogger::Reset();
  ```
- 明示 `Reset()` する run 切替経路（Optuna）が別にあれば、同様に直前へ `Notifier::Shutdown()` を入れる。

## 設計判断（why。self-contained のため明記）

1. **両輪（R1 ガード + R2 順序）**: R2（順序）で正常シャットダウンの競合窓を消し、R1（null-safe static 入口）で残る全経路（Optuna run 切替・想定外の背景呼び出し）を安全網でカバー。順序＝「ライフサイクル整合」、ガード＝「null 耐性」で 1 関心ずつに分離（`feedback_design_separate_concerns` の方針）。
2. **`MetricsLogger::` static を唯一の入口に**: `if (auto m = Instance())` を各所に散らさず static 入口へ内包。呼び側は `Instance()->` を除去するだけで null チェックが自動で効く。入口が `MetricsLogger` に集約され（別名前空間 facade を設けない＝「MetricsLogger が入口」）、新規ログも static 経由が自然に強制される。
3. **instance メソッドは private impl 化**: static と同名インスタンスメソッドは C++ で共存不可のため、旧メソッドを `*Impl` へ private 改名し static が転送する。ログの直接インスタンス呼び出しを塞ぎ「入口は static のみ」を型で保証する（＝全呼び出しの移行を強制、漏れがコンパイルで露見）。
4. **null = no-op が妥当**: ログは学習の付随出力。teardown / run 切替の一時的な欠落は許容範囲。ここでエラー化すると shutdown をかえって阻害する。
5. **アクセサは対象外**: 値返しは no-op になじまず、呼び出しも main/setup スレッド・Reset 前。スコープを膨らませない（NG1）。
6. **Shutdown hook を observer 基底に**: eval のような背景スレッドを持つ observer 一般の teardown 口。現状 override は `EpisodeEvalObserver` のみ。`Notifier` が一括で呼ぶ。

## 影響・移行

- 全ログ呼び出し（~30 箇所）を `MetricsLogger::Log/LogScalar` static へ機械置換。instance メソッド private 化のため**テスト内 `Instance()->Log` も移行必須**（部分移行はコンパイルエラー＝漏れが即検出できる）。挙動は不変（null 時のみ no-op が増える）。
- テストは `MetricsLogger` の `Init`/`Reset` を制御しているため、static 化後も既存挙動を維持。
- `Notifier::Shutdown()` / observer `Shutdown()` は既定 no-op・冪等で、既存 observer に無影響。

## 触る主なファイル

- `core/anet-core/include/anet/metrics_logger.hpp` / `src/metrics_logger.cpp` — public static ログメソッド追加＋旧 instance メソッドを private `*Impl` へ改名（.cpp 定義・内部相互呼び出しも）。
- `core/anet-core/include/anet/rl.hpp` — observer 3 基底に `virtual void Shutdown() {}`、`Notifier::Shutdown()` 宣言。
- `core/anet-core/src/rl.cpp` — `Notifier::Shutdown()` 定義。
- `core/anet-core/include/anet/observers.hpp` / `core/anet-core/src/observers.cpp` — `EpisodeEvalObserver::Shutdown()` override＋dtor 経由化、observers.cpp 内ログ呼び出しの static 化。
- `apps/runner/src/RunnerApp.cpp` — `OnExit()` に `Notifier::Shutdown()` を Reset 前へ挿入、config ログの static 化。
- ログ呼び出しを持つ各所（`*_agent.cpp`, `image.cpp`, `env.cpp`, `core/envs/*/src/*Env.cpp`, テスト）— `Instance()->Log/LogScalar` → `MetricsLogger::Log/LogScalar` static へ機械置換。

## Testing Decisions

- **単体**:
  - static 入口が null 時 no-op（`MetricsLogger::Reset()` 後に `MetricsLogger::LogScalar` / `MetricsLogger::Log` を呼んでも例外なし）、非 null 時は private impl へ転送される（backend に書かれる）こと。新タグ `[metrics][static-log]`。
  - `EpisodeEvalObserver::Shutdown()` が in-flight `eval_future_` を wait し `eval_pool_` を stop すること。二重呼び出し（`Shutdown()` 後に dtor）で安全なこと。
- **統合 / 手動**: `use_background` eval 有効の config で学習開始 → eval 実行タイミングでアプリ終了し、`this==nullptr` クラッシュが出ないこと（`OnExit` の `Notifier::Shutdown()` → `Reset()` 順序）。
- **回帰**: 通常学習中のメトリクス出力（scalar / 画像）が従来通り。`git diff --check` 必須。

## 受け入れ基準

- **AC1**: eval 実行中のシャットダウンで `this==nullptr` クラッシュが発生しない。
- **AC2**: `MetricsLogger::Reset()` 後に `MetricsLogger::Log` / `MetricsLogger::LogScalar`（static）を呼んでも安全に no-op。
- **AC3**: `OnExit` で `Reset()` の前に eval プールが排水される（`Notifier::Shutdown()` 経由）。
- **AC4**: 既存のメトリクス出力・テストが回帰しない。
