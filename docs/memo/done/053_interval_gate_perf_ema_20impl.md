# PRD 053 実装メモ（interval の bucket-crossing 化 / perf の時間重み EMA 化）

- 対象 PRD: `docs/memo/053_interval_gate_perf_ema_10prd.md`
- 関連: ADR 0028、CONTEXT.md 用語「時間重みEMA」（いずれも作成済み）
- 作成日: 2026-08-15
- 実装担当: 本セッション（Opus 5）

## 概要

飛び飛びの step カウンタに対する `interval` の発火判定を bucket-crossing へ統一し、共通部品 `IntervalGate` として切り出す。あわせて `TrainRunner` の perf メトリクスを、`EmaFilter` の時定数モード（τ=10 秒）による時間重み EMA へ置き換える。

固定した既定値:

- τ = 10.0 秒（`TrainRunner::kPerfEmaTauSec`、ハードコード。設定化は PRD スコープ外）
- 繰り越し閾値 = 1000 usec（`TrainRunner::kPerfMinUsec`。200ms から変更）
- α の下限クランプ = `std::numeric_limits<T>::epsilon()`
- EXP 軸 metrics の `interval` = 5000（baseline 3 本 + iqn_search_p0 10 本）
- `37_agent_qtd` の grad 4 本は `interval:10` を撤去し毎 round 出力

## 主な変更

### 1. `core/anet-core/include/anet/util.hpp`

**`IntervalGate` 新設**（PRD §2.1）

```cpp
class IntervalGate {
public:
    explicit IntervalGate(uint64_t interval);   ///< interval == 0 は ANET_SYSTEM_ERROR
    bool ShouldFire(uint64_t step);             ///< バケットを跨いだ最初の呼び出しで true
    void Reset();
private:
    uint64_t interval_;
    bool fired_ = false;
    uint64_t last_bucket_ = 0;
};
```

- 初回呼び出しは step 値によらず必ず発火し、`last_bucket_ = step / interval_` を記録する（D4）。
- 以降は `step / interval_ > last_bucket_` のときだけ発火する。1 回の呼び出しで複数バケットを跨いでも発火は 1 回（catch-up しない）。
- step 減少時は発火せず、`last_bucket_` も更新しない。
- 実効周期は `max(interval, 刻み)` に丸まる（D5）。

**`EmaFilter` に時定数モード追加**（PRD §2.2）

- `static EmaFilter TimeWeighted(T tau_sec)`：非浮動小数点型は `static_assert`、`tau_sec` は正かつ有限を検証。
- `void Update(T x, T dt_sec)`：`α = 1 - exp(-dt/tau)` を `[epsilon, 1]` にクランプして `decay_` に入れ、既存の `value_`/`weight_` 漸化式へ委譲する（PRD 045 のバイアス補正は可変 α でも厳密に成立）。
- 既存 `Update(T x)` の本体は private `UpdateCore(T x)` へ切り出す。
- モード排他（D8、いずれも `ANET_SYSTEM_ERROR`）:
  - 時定数モードで `Update(x)` 単体 / `SetDecay()`
  - サンプル重みモードで `Update(x, dt)`
  - `dt_sec <= 0` または非有限
- `Set()` / `Restart()` / `Value()` / `IsInitialized()` の意味は両モードで不変。

### 2. `core/anet-core/include/anet/trainer.hpp` / `src/trainer.cpp`

- dead member `acc_train_steps_` / `acc_exp_steps_` を削除。
- `last_train_step_per_sec_` / `last_exp_step_per_sec_` を `EmaFilter<float>`（`TimeWeighted(kPerfEmaTauSec)` 初期化）へ置換。
- `CalcPerformanceMetrics` を PRD §2.3 の形へ。`usec_diff < kPerfMinUsec` なら `last_time_` を更新せず return（時間を捨てない繰り越し方式、D10）。
- `GetScalar` の `TRAIN_STEP_PER_SEC` / `EXP_STEP_PER_SEC` は EMA 読み出し。未初期化時は NaN。

### 3. `core/anet-core/src/observers.cpp` / `include/anet/observers.hpp`

PRD §1.2 の対象 5 箇所のみ差し替える。TRAIN 軸・`local_episode_count_` 軸・Agent 内部 interval は触らない。

| 箇所 | 変更 |
|---|---|
| `EpisodeEvalObserver::OnLearn` | `std::optional<IntervalGate> eval_gate_`（`eval_interval_ > 0` のとき生成） |
| `TimeHistogramObserver::OnLearn` frame | `std::optional<IntervalGate> frame_gate_` |
| `TimeHistogramObserver::OnLearn` log | `std::optional<IntervalGate> log_gate_`。`log_interval <= frame_interval` の毎フレーム出力分岐は据え置き |
| `SweepedHeatMapObserver::OnLearn` | `std::optional<IntervalGate> log_gate_` |
| `MetricsLogObserverBase::OnGenericUpdate` | `IntervalGate gate_` をメンバ化。ctor で `interval < 1` を `ANET_SYSTEM_ERROR`。判定位置は現行と同じ（EMA 更新の後、finite/clip の前） |

`<=0 は無効` の既存意味を保つため、該当箇所は `std::optional` で保持する。gate は既存 `val_ema_` と同じ per-instance 状態のため、追加ロックは不要。

### 4. `apps/runner/config/metrics_scalar.txt`（PRD §2.4）

- baseline `90_perf/12_exp_step_per_sec` / `22_exp_step_per_sec_ema` / `90_elapse_hour`：`interval:100` → `interval:5000`
- iqn_search_p0 の `$exp_step` 明示 10 本：`interval:100` → `interval:5000`
- baseline `37_agent_qtd/21,22,24,25`：`interval:10` を撤去
- min 層 / full 層 perf（train 軸）は変更なし

### 5. ドキュメント

- `docs/design/140_observability.jp.md`：`interval` の意味論（bucket-crossing、実効周期 `max(interval, 刻み)`、初回発火、EMA は interval と無関係に毎回更新、`interval >= 1` 必須）を追記。
- `docs/design/030_user_guide_analysis.jp.md`：`exp_step_per_sec` は τ=10 秒の時間重み EMA であり、真の throughput は `90_perf/90_elapse_hour` 差分から算出する旨へ改訂。

## テスト

- **Public interface / surface**: `anet::IntervalGate`（ctor 検証 / `ShouldFire` / `Reset`）、`anet::EmaFilter<T>`（`TimeWeighted` / `Update(x,dt)` / `SetDecay` / `Value`）、`MetricsLogObserverBase` の `interval` 検証。
- perf 系（`CalcPerformanceMetrics`）は時刻注入の仕組みが無いため値テストを書かない。テスト可能な核（α 算出・発火判定）を `util.hpp` 側へ寄せることで担保する（PRD §3.2）。
- 追加先は `core/anet-core/src/util_test.cpp`。

**優先 behavior / TDD 順序**（1 behavior ずつ RED → GREEN）:

1. tracer bullet：`IntervalGate(100)` に J=16 刻み（0,16,32,…）を流し、100 刻みバケットごとにちょうど 1 回発火する。
2. step=0 で発火する。
3. 同一バケット内の複数呼び出しで発火は 1 回。
4. 1 回の呼び出しで複数バケットを跨いでも発火は 1 回。
5. 刻みが `interval` より大きい場合は毎回発火する。
6. 非整数的な刻み（3,3,4,3,…）で発火が欠落しない。
7. `interval = 0` は `ANET_SYSTEM_ERROR`。`Reset()` 後は次回必ず発火。
8. `TimeWeighted(tau <= 0)` / 非有限 tau が `ANET_SYSTEM_ERROR`。
9. `Update(x, dt)` の `dt <= 0` / 非有限が `ANET_SYSTEM_ERROR`。
10. 極小 dt で α がクランプされ throw しない。
11. 巨大 dt で α → 1 となり最新値へ収束する。
12. 可変 dt 列で `Value()` が時間重み平均として正しい。
13. モード混在 3 パターンが `ANET_SYSTEM_ERROR`。

以降、Observer / TrainRunner の差し替えを行い、そのつど既存テストを再実行する。config と docs は最後にまとめて更新する。

非退行の確認: 既存 `[util][ema]` 7 ケースと `[observers]` 系が緑のまま。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
```

```powershell
core\anet-core\bin\Debug\anet-core-test.exe "[util]"
```

```powershell
core\anet-core\bin\Debug\anet-core-test.exe
```

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

## 前提

- 実 Run が必要な受け入れ基準（eval 発火の num_envs 非依存、perf 真値 478 steps/s 付近、metrics 密度の num_envs 非依存、学習の非退行）はユーザーが実施する。本作業はコード + 単体テスト + ビルドまで。
- git 操作（add/commit/push）は行わない。ワークツリー変更までとし、コミット文案のみ提示する。
- 既存の未コミット変更は保持し、PRD が指定する行だけを編集する。
- tag `90_perf/12` / `22` は据え置きで意味だけ変わる（過去 Run と数値比較不可）。
