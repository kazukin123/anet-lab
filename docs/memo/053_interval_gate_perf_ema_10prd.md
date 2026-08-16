# PRD 053: interval 発火の bucket-crossing 化 と perf メトリクスの時間重み EMA 化

- 起票日: 2026-08-15
- 状態: implementation ready
- 対象: `core/anet-core`（util / trainer / observers）、`apps/runner/config/metrics_scalar.txt`
- 関連: ADR 0028（本 PRD で新設）、CONTEXT.md 用語「時間重みEMA」（本 PRD 作成時に追加済み）、PRD 045（EmaFilter バイアス補正）、PRD 052（eval 定義とスケジュールの分離）
- 設計文書: `docs/design/140_observability.jp.md`、`docs/design/030_user_guide_analysis.jp.md`
- 発見経緯: Atari num_envs A/B（`run_20260815-031815_atari_pong_n512` / `run_20260815-035220_atari_pong_n64` / `run_20260815-052657_atari_pong`）

## Context / Problem Statement

Atari の num_envs A/B で、独立した 2 つの不具合が同時に露見した。どちらも **「刻みが 1 でないカウンタ」「長さが不揃いな窓」を、均一である前提の式で扱っている** ことが根本にある。

### 不具合 A: eval 発火頻度が num_envs 依存

`EpisodeEvalObserver::OnLearn` は `step % eval_interval_ == 0` で発火を判定する（[observers.cpp:550](../../core/anet-core/src/observers.cpp)）。しかし `LearnEvent` は round（1 回の `UpdateFromBatch`）につき 1 回しか飛ばず、そこに載る `learn_step` は round あたり

```
J = num_envs × replay_ratio / batch
```

だけ飛ぶ。剰余判定は 1 刻み前提なので、実効周期は `LCM(J, interval)` に伸びる。

確定した原因連鎖（実証済み）:

1. `learn_step` は勾配更新 1 回ごとに `++`（[dqn_based_agent.cpp:2374](../../core/anet-core/src/dqn_based_agent.cpp)）。round 内の J 回分は `step_counts_.learn_step += update_results.size()` で**一括加算**され、しかも加算は `LearnEvent` 通知の**後**（[trainer.cpp:527,544](../../core/anet-core/src/trainer.cpp)）。
2. したがって Observer が見る `learn_step` は round 開始時点の値で、J 刻みの離散列になる（n512 なら 0, 16, 32, …／n64 なら 0, 2, 4, …）。
3. `interval=100` に対し、n512 は `LCM(16,100)=400` 更新ごと、n64 は `LCM(2,100)=100` 更新ごとに発火。**同じ設定で eval 回数が 4 倍違う**。
4. eval は同期的にコストを発生させるため、この差がそのまま wall-clock 差になった。当初「n512 の方が 2.65 倍速い」と読めたが、eval 頻度を揃えると n64 1,198 steps/s ≒ n512 1,264 steps/s で差は消える。**num_envs の性能差と読めていたものは、実体はこのバグだった**。

さらに `update_credit` は float であるため、`num_envs=100`（J=3.125）のような構成では `learn_step` が 3, 3, 4, 3, … と進み、99→102 のようにバケット境界を跨いでも発火しない。**周期が伸びるのではなく、発火が丸ごと欠落する**。

同じ構造は EXP 軸の metrics にもある（[observers.cpp:977](../../core/anet-core/src/observers.cpp)）。`exp_step` は round あたり `num_envs` 刻みなので、`interval:100` の perf メトリクスは実際には `LCM(num_envs, 100)` ごとにしか記録されていなかった（n64 で 1,600 / n512 で 12,800、実測一致）。

### 不具合 B: perf メトリクスが真値を隠す

`TrainRunner::CalcPerformanceMetrics` は 200ms 以上たまった窓を閉じてレートを上書きする（[trainer.cpp:404-424](../../core/anet-core/src/trainer.cpp)）。

```cpp
if (usec_diff >= 200000) { // 200msec 積算
    last_exp_step_per_sec_ = static_cast<float>(exp_step_delta) * 1000000.0f / usec_diff;
    last_time_ = now;
    last_exp_step_ = exp_step;
}
```

窓は時間軸を隙間なく敷き詰めるので、stall の時間もどれかの窓には入る。問題はその先にある:

1. 窓の**長さが不揃い**である。健全時は 1 窓 ≒ 0.2 秒だが、eval ブロック時は 3.6 秒がまるごと 1 窓に入る。
2. ログが拾うのは「最後に閉じた窓の値」なので、**3.6 秒の窓も 0.2 秒の窓も 1 サンプルとして同格**に並ぶ。時間重みが失われる。
3. さらに stall 中は step が進まないためログ発火自体も止まる。サンプリングは実質 step 比例であり、**step の進む健全窓ばかりを拾う**。

結果、真値 478 steps/s の Run が 1,830 steps/s と表示されていた。真の throughput は `90_perf/90_elapse_hour` の差分からしか得られない。`90_perf/22_exp_step_per_sec_ema` も observer 側のサンプル重み EMA なので、同じ偏りをそのまま引き継いでいた。

### 根本問題

- **(A)** 離散イベント列に対する `interval` の意味が、刻みと interval の整除性という**設定間の偶然の関係**に支配されている。
- **(B)** 経過時間で重み付けすべき量を、サンプル単位で平滑・サンプリングしている。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | 両不具合を本 PRD 1 本に収める。発見経緯・根本の性質・受入検証 Run が同一であり、分けると文脈が二重記載になる |
| D2 | 発火判定は **bucket-crossing**（`step / interval` の商が増えたら発火）に統一し、共通部品 `IntervalGate` として切り出す。意味論の確定は ADR 0028 |
| D3 | 適用対象の基準は「**軸の刻みが 1 でない箇所**」。LEARN 軸を Observer が直参照する 3 箇所と、EXP 軸を使う metrics が該当（§1.2） |
| D4 | 初回発火は `step=0` で行う（従来の剰余判定と等価）。warmup 明け直後のベースライン点を残す |
| D5 | `interval` が刻みより小さい場合は**毎 round 発火に丸まる**ことを仕様とする。round より細かい発火は構造的に不可能 — round 内の中間モデル状態は event 到達時点で既に失われており、per-update に割っても評価対象は「J 回後のモデル」1 つだけになる |
| D6 | 時間重み化は専用クラスを新設せず、`EmaFilter` に**時定数モード**として追加する |
| D7 | API は `static EmaFilter TimeWeighted(T tau_sec)` と `void Update(T x, T dt_sec)`。α = `1 - exp(-dt/tau)` を都度算出して既存 `Update(x)` へ委譲する。`value_`/`weight_` の漸化式をそのまま使うため、PRD 045 のバイアス補正は可変 α 下でも厳密に成立する |
| D8 | モードは排他 + fail-fast。時定数モードでの `Update(x)` 単体呼び出しと `SetDecay()`、サンプル重みモードでの `Update(x, dt)`、`dt <= 0`、非浮動小数点型での時定数モードは、すべて `ANET_SYSTEM_ERROR` |
| D9 | τ = 10 秒。名前付き `constexpr` でハードコードする。`TrainRunner` のコンストラクタは `(env, agent, notifier)` のみで `ConfigData` を受け取らず、設定化は `RunnerFactory` まで波及するため（§7） |
| D10 | dt ガードは現行の**繰り越し方式**を維持する（閾値未満なら `last_time_` を更新せず次回にまとめる＝時間を捨てない）。閾値は 200ms → 1ms |
| D11 | tag `90_perf/12_exp_step_per_sec` と `90_perf/22_exp_step_per_sec_ema` は**据え置き**。タグを増やさず同じ tag の意味を変える。22 は「EMA の EMA」になるが、12 が既に時間重み済みなので値にバイアスは入らない（短期＝12／超長期＝22 の 2 本立てになる） |
| D12 | EXP 軸 interval の値を `interval:5000` に再設計する（baseline perf 3 本 + iqn_search_p0 10 本）。主流 num_envs=256 の現行実効 6,400 に近く、全 config でサンプル間隔が τ=10 秒より短く保たれる |
| D13 | grad 4 本の `interval:10` は撤去し、`37_agent_qtd` グループを毎 round に統一する。間引く計算コスト上の理由はなく（grad_norm は UpdateResult に既載）、EMA は interval と無関係に毎回更新されるため、生値 2 本の解像度だけが 1/5 に落ちていた |
| D14 | 成果物は本 PRD、ADR 0028、CONTEXT.md 用語「時間重みEMA」1 件。実装時に `docs/design/` の関連記述を同一変更内で更新（§3.3） |

## 1. 現状の事実（コード確認済み）

2026-08-15 時点、branch `main`。実測値は上記 3 Run の `metrics.jsonl` から算出。

### 1.1 軸ごとの刻み

| 軸 | round あたりの刻み | 判定 |
|---|---|---|
| TRAIN（`train_step`） | +1 | 健全 |
| LEARN、metrics 経由 | +1（UpdateResult 1 件ごとに展開済み。[observers.cpp:802-820](../../core/anet-core/src/observers.cpp)） | 健全 |
| LEARN、Observer が直参照 | +J（`J = num_envs × replay_ratio / batch`） | **バグ** |
| EXP（`exp_step`） | +num_envs | **バグ** |
| `local_episode_count_` | +1（エピソード終端ごと） | 健全 |
| Agent 内部の `learn_step` | +1（更新ごと） | 健全 |

RR と batch を固定している限り `learn_step = exp_step × RR / batch` であり、比例定数は num_envs によらない（本 Run では 1 更新 = 32 exp_step）。**軸の選択は正しく、壊れているのは発火規則だけ**である。

### 1.2 修正対象と対象外

**対象**

| 箇所 | 参照軸 |
|---|---|
| [observers.cpp:550](../../core/anet-core/src/observers.cpp) `EpisodeEvalObserver::OnLearn` | LEARN 直参照 |
| [observers.cpp:148](../../core/anet-core/src/observers.cpp) `TimeHistogramObserver` frame_interval | LEARN 直参照 |
| [observers.cpp:157](../../core/anet-core/src/observers.cpp) `TimeHistogramObserver` log_interval | LEARN 直参照 |
| [observers.cpp:387](../../core/anet-core/src/observers.cpp) `SweepedHeatMapObserver` log_interval | LEARN 直参照（`event.counts.learn_step`） |
| [observers.cpp:977](../../core/anet-core/src/observers.cpp) `MetricsLogObserverBase` | config 指定軸（EXP 軸のとき該当） |

**対象外（刻み 1 で健全。誤って触らないこと）**

- [observers.cpp:71](../../core/anet-core/src/observers.cpp) `HeatMapVectorObserver::OnTrain`、[observers.cpp:322](../../core/anet-core/src/observers.cpp) `MultiPairHeatMapObserver::OnTrain`、[observers.cpp:1057](../../core/anet-core/src/observers.cpp) `GraphVizObserver::OnTrain` — いずれも TRAIN 軸
- [observers.cpp:629](../../core/anet-core/src/observers.cpp)、[observers.cpp:1041](../../core/anet-core/src/observers.cpp) — `local_episode_count_`
- [dqn_based_agent.cpp:440](../../core/anet-core/src/dqn_based_agent.cpp) `hard_update_interval`、[image_cls_agent.cpp:430](../../core/anet-core/src/image_cls_agent.cpp) `learn_log_interval` — Agent 内部で 1 刻み
- `metrics_scalar.txt` の min 層 / full 層 perf — `$exp_step` 未指定のため `@train` 既定の train_step 軸で健全

### 1.3 再現例

`interval=100`、`replay_ratio=8`、`batch=256`（本 Run の設定）:

| num_envs | J | 実効周期（更新） | exp_step 換算 | 実測発火数（2.5M step） |
|---|---|---|---|---|
| 64 | 2 | 100 | 3,200 | 775 |
| 512 | 16 | **400** | **12,800** | 194 |
| 100 | 3.125 | 不定（欠落あり） | — | — |

`interval=500`、num_envs=64 では `LCM(2,500)=500` 更新 = 16,000 exp_step、実測 154 発火。指定どおりに効く構成と 4 倍ずれる構成が混在する。

perf メトリクス（不具合 B）の読み値と真値:

| Run | eval 発火間隔 | メトリクス表示（調和平均） | 真値（elapse_hour 差分） | 乖離 |
|---|---|---|---|---|
| n64 `interval=100` | 3,200 exp_step | 1,830 steps/s | **478 steps/s** | 74% が未計上 |
| n64 `interval=500` | 16,000 exp_step | 1,531 steps/s | 1,198 steps/s | 22% |
| n512 `interval=100`（実効 400） | 12,800 exp_step | 1,607 steps/s | 1,264 steps/s | 21% |

n64 の `interval=100` と `interval=500` は **train 曲線が完全一致**する（`42_env/11_episode_score_mean_ema` が全マイルストーンで小数以下まで同値、`episode_len` / `loss` も同様）。eval 頻度は学習に影響せず、差分は eval の実行コストのみであることが確認できる。

## 2. 契約

### 2.1 IntervalGate

```cpp
/**
 * @brief 飛び飛びの step カウンタに対する発火間隔ゲート。
 *
 * step / interval の商（バケット）が前回発火時より増えたときに 1 回だけ発火する。
 * 刻みが interval を割り切るかどうかに依存せず、各バケットで必ず 1 回発火する。
 */
class IntervalGate {
public:
    explicit IntervalGate(uint64_t interval);   ///< interval == 0 は ANET_SYSTEM_ERROR
    bool ShouldFire(uint64_t step);             ///< バケットを跨いだ最初の呼び出しで true
    void Reset();                               ///< 未発火状態へ戻す
private:
    uint64_t interval_;
    bool fired_ = false;
    uint64_t last_bucket_ = 0;
};
```

- 初回呼び出し（`fired_ == false`）は step の値によらず必ず発火し、`last_bucket_ = step / interval_` を記録する（D4）。
- 以降は `step / interval_ > last_bucket_` のときだけ発火する。1 回の呼び出しで複数バケットを跨いだ場合も発火は 1 回（catch-up しない。ADR 0028）。
- `step` が減少する呼び出しは想定しない（step 軸は非減少）。減少時は発火せず、`last_bucket_` も更新しない。
- 実効周期は `max(interval, 刻み)` に丸まり、位相ジッタは 1 イベント以内（D5）。

置き場所は `core/anet-core/include/anet/util.hpp`（`EmaFilter` と同じ汎用小物の層）。`rl.hpp` に依存させないため step 型は `uint64_t` を直接使う。

### 2.2 EmaFilter 時定数モード

```cpp
static EmaFilter TimeWeighted(T tau_sec);   ///< 時定数モードで生成（tau_sec > 0、有限）
void Update(T x, T dt_sec);                 ///< 時間重み更新
```

`Update(x, dt)` の内部:

```cpp
// 時定数モードでのみ有効
const T alpha = T(1) - std::exp(-dt_sec / tau_);
SetDecayInternal(std::clamp(alpha, kMinDecay, T(1)));   // ValidateDecay を通す前に下限クランプ
Update(x);                                              // 既存の value_/weight_ 漸化式へ委譲
```

- **バイアス補正はそのまま成立する**。`weight_` を `value_` と同じ漸化式で更新する PRD 045 の方式は一様 α を仮定しないため、α が呼び出しごとに変わっても `Value()` は「観測済みサンプルの重み付き平均」であり続ける。
- α の下限クランプが必要。極小 `dt` では `1 - exp(-dt/tau)` が浮動小数点で 0 に落ち、`ValidateDecay` の `(0, 1]` 検査に引っかかる（[util.hpp:139-145](../../core/anet-core/include/anet/util.hpp)）。
- 排他契約（D8）:

| 呼び出し | 結果 |
|---|---|
| 時定数モードで `Update(x)` 単体 | `ANET_SYSTEM_ERROR` |
| 時定数モードで `SetDecay()` | `ANET_SYSTEM_ERROR` |
| サンプル重みモードで `Update(x, dt)` | `ANET_SYSTEM_ERROR` |
| `dt_sec <= 0` または非有限 | `ANET_SYSTEM_ERROR`（繰り越しは呼び出し側の責務） |
| 非浮動小数点型 `T` で `TimeWeighted()` | `static_assert` |

`Set()` / `Restart()` / `Value()` / `IsInitialized()` の意味は両モードで変わらない。

### 2.3 perf メトリクスの意味

`TrainRunner::CalcPerformanceMetrics` を次の形に置き換える:

```cpp
constexpr float kPerfEmaTauSec = 10.0f;      // 名前付き定数（D9）
constexpr int64_t kPerfMinUsec = 1000;       // 繰り越し閾値 1ms（D10）

auto usec_diff = duration_cast<microseconds>(now - last_time_).count();
if (usec_diff < kPerfMinUsec) return;        // last_time_ を更新しない = 次回へ繰り越す

const float dt = static_cast<float>(usec_diff) / 1e6f;
train_step_per_sec_ema_.Update(static_cast<float>(train_step_delta) / dt, dt);
exp_step_per_sec_ema_.Update(static_cast<float>(exp_step_delta) / dt, dt);
last_time_ = now;
last_train_step_ = train_step;
last_exp_step_ = exp_step;
```

- `GetScalar(EXP_STEP_PER_SEC)` は `exp_step_per_sec_ema_.Value()` を返す。未初期化時は現行同様 NaN（`IsInitialized()` で判定）。
- **意味の変更**: tag `90_perf/12` は「直近 200ms 窓の瞬間値」から「τ=10 秒の時間重み EMA」になる。過去 Run の同 tag と数値を直接比較できなくなる（過去値は stall を過小評価している）。
- `90_perf/22` は「時間重み EMA をさらにサンプル EMA で平滑した値」になる（D11）。`ema_alpha:0.001` × サンプル間隔から、実効的には Run 全体平均に近い長期線として機能する。
- `train_step_per_sec` も同一関数内なので同じ扱いになる。
- `acc_train_steps_` / `acc_exp_steps_` は 0 代入のみで読まれない dead member であり、本変更で削除する（[trainer.hpp:140-141](../../core/anet-core/include/anet/trainer.hpp)）。
- `EvalRunner` は `TrainRunner` の兄弟クラスで perf キーを持たないため（`RunnerBase::GetScalar` に perf キーなし）、影響しない。

### 2.4 config interval の再設計

修正により `interval` は指定どおりの周期になるため、現行の実効値を踏まえて値を決め直す。

現行の実効間隔（exp_step 単位）:

| config | num_envs | perf（`interval:100`） | grad（`interval:10`） | Run 長 | perf 点数 |
|---|---|---|---|---|---|
| Atari | 64 | 1,600 | 320 | 2.5M | 1,562 |
| DropMerge / GridMaze / LunarLander | 256 | 6,400 | 1,280 | 700M（DropMerge） | 109,375 |
| ImageCls | 128 | 3,200 | 640 | 10M | 3,125 |

新しい値:

| 対象 | 現行 | 新 | 根拠 |
|---|---|---|---|
| baseline `90_perf/12,22,90`（3 本） | `interval:100` | `interval:5000` | 主流 num_envs=256 の現行実効 6,400 に近い。DropMerge 700M で 14 万点（現行比 +28%）、Atari 2.5M で 500 点。τ=10 秒に対しサンプル間隔が 1〜4 秒で情報欠落なし |
| iqn_search_p0 の `$exp_step` 明示 10 本 | `interval:100` | `interval:5000` | 同上 |
| baseline `37_agent_qtd/21,22,24,25`（grad 4 本） | `interval:10` | **撤去** | 同グループの他 12 本と揃える（D13） |
| min 層 / full 層 perf | `interval:100` | 変更なし | train_step 軸で既に健全 |

grad 4 本の撤去により metrics.jsonl は num_envs=64 の 2.5M Run で約 +15%（117MB → 約 135MB）と見積もる。perf 側は点数が減るため相殺方向。

## 3. 実装範囲

実装は Codex 担当。本書は self-contained に記述する。

### 3.1 コード

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/include/anet/util.hpp` | `EmaFilter` に時定数モード（`TimeWeighted` / `Update(x,dt)` / `tau_` / 排他検査）。`IntervalGate` を新設（§2.1） |
| `core/anet-core/include/anet/trainer.hpp` | `acc_train_steps_` / `acc_exp_steps_` を削除。`EmaFilter<float>` 2 本（train / exp）をメンバ化。`last_train_step_per_sec_` / `last_exp_step_per_sec_` は EMA へ置き換え |
| `core/anet-core/src/trainer.cpp` | `CalcPerformanceMetrics` を §2.3 の形へ。`GetScalar` の perf キーを EMA 読み出しに変更 |
| `core/anet-core/src/observers.cpp` / `observers.hpp` | 対象 5 箇所（§1.2）を `IntervalGate` に差し替え。`MetricsLogObserverBase` はタグごとインスタンスなのでメンバとして保持できる |

### 3.2 テスト

| ファイル | 追加内容 |
|---|---|
| `core/anet-core/src/util_test.cpp` | `EmaFilter` 時定数モード: ①`dt=0` / 負 / 非有限で `ANET_SYSTEM_ERROR` ②極小 dt で α がクランプされ throw しない ③巨大 dt で α→1（最新値へ収束） ④可変 dt 列でも `Value()` が重み付き平均として正しい ⑤モード混在の 3 パターンが `ANET_SYSTEM_ERROR`。`IntervalGate`: ⑥step=0 で発火 ⑦同一バケット内の複数呼び出しで 1 回だけ ⑧1 回で複数バケットを跨いでも 1 回 ⑨`interval` より大きい刻みで毎回発火 ⑩非整数的な刻み（3,3,4,3,…）で欠損が出ない ⑪`interval=0` で `ANET_SYSTEM_ERROR` |

perf 系は実時間依存で時刻注入の仕組みが無いため、`CalcPerformanceMetrics` 自体の値テストは書かない。テスト可能な核（α 算出と発火判定）を `util.hpp` 側に寄せることで担保する。

### 3.3 config・文書（実装と同一変更内で更新）

| ファイル | 変更内容 |
|---|---|
| `apps/runner/config/metrics_scalar.txt` | §2.4 の表のとおり |
| `docs/design/140_observability.jp.md` | `interval` の意味論（bucket-crossing、実効周期 = `max(interval, 刻み)`、初回発火、EMA は interval と無関係に毎回更新される）を追記 |
| `docs/design/030_user_guide_analysis.jp.md` | `exp_step_per_sec` の運用注意を改訂。「安定区間の値を比較する」→「τ=10 秒の時間重み EMA であり、真の throughput は `elapse_hour` 差分から算出する」 |

## 4. 受け入れ基準

- 既存テスト全緑（`anet-core-test.exe`。ビルドは AGENTS.md 記載の `VsDevCmd.bat` 経由）。
- **eval 発火の num_envs 非依存**: 同一 `interval` で num_envs=64 と 512 の Run を回し、`51_eval1/*` サンプルの exp_step 間隔が両者で一致する（`interval × batch / replay_ratio`）。修正前は 4 倍ずれる。
- **欠損なし**: J が `interval` を割り切らない num_envs（例: 100）でも発火が欠落しない。
- **perf の真値一致**: `90_perf/12_exp_step_per_sec` の定常値が、`90_perf/90_elapse_hour` の差分から算出した真の throughput と一致する。eval stall のある構成（n64 `interval=100` 相当）で、修正前 1,830 に対し **478 付近**を示すこと。
- **metrics 密度の num_envs 非依存**: 同一 config で num_envs だけ変えた 2 Run のサンプル間隔（exp_step）が一致する。
- `interval` が刻みより小さい構成でエラーにならず、毎 round 発火に丸まる。
- 学習の非退行: 同一 seed・同一設定で train 曲線が修正前と一致する（eval 頻度は学習に影響しないため、`42_env/*` は不変であるべき）。

## 5. スコープ外

- **eval コスト自体の設計**。エピソード長がスキル向上に伴って伸びる（Pong で 764 → 5,342 step）ため、固定 interval では終盤ほど eval が高くつく。当面は `interval` の調整で対応する。`eval_batch_size > 1` は現行の `RunEvaluationEpisode` が `LastStepHadEpisodeEnd()`（[trainer.cpp:166](../../core/anet-core/src/trainer.cpp) — **どれか 1 lane でも終端で true**）で停止するため、記録されるのは最初に終わった 1 本だけであり、短いエピソードほど先に終わる分だけスコアが下振れする。使うなら「全 lane 終端まで回して平均」への変更がセットで必要。
- **Observer 呼び出し構造の変更**（Agent 内部トリガへの回帰）。round より細かい eval は原理的に不可能（D5）。
- **`interval` 指定なしの agent 系メトリクス**（metrics.jsonl の約 96%）の間引き。今回のバグとは独立で、毎イベント記録は意図どおり。
- **perf メトリクスの τ の設定化**。§7 参照。

## 6. Further Notes

- τ を config 化しなかったのは、`TrainRunner` に config 経路が存在しないため（コンストラクタは `(env, agent, notifier)` のみ、`RunnerBase` も同様）。設定化すると `RunnerFactory::CreateMainRunner` まで signature 変更が波及する。必要になった時点で `RunManager::Config` からの setter 注入を検討する。
- 判定軸と出力 step 軸が同一である（`$step_axis` が「JSONL の step へ使う counter」と「interval 判定に使う値」を兼ねる）ことは、今回は問題にならない。bucket-crossing により、どの軸を選んでも `interval` は指定どおりに効くため。
- 本 PRD の発見経緯そのもの（num_envs A/B の結論）は実験記録として別途 `docs/experiments/` に残す想定。num_envs=512 は wall-clock 優位が幻で、学習効率では 64 に大きく劣る（2.0-2.5M 帯の eval スコアが -5.8 対 +7.1）ことが同 A/B で確定している。
