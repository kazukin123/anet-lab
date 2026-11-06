# スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） 暫定 PRD

> 2026-11-03 番号を 999 から 954 へ振り直した（PRD114）。

> 状態: 暫定メモ。案と未決 Q は未確定。詳細は別途グリルで詰める。本 PRD は実装着手を意味しない。
> 起点: 2026-09-05、Atari の壁突破率（Breakout 1 画面 = 432 点以上のゲームの割合）をメトリクス化した際、同じパターンが **DropMerge で既に手書きされていた**ことが判明したこと。閾値だけが違う同一構造が 2 env で独立に再発明されている。
> 関連: [947_q_statistics_config_screening_10prd.md](947_q_statistics_config_screening_10prd.md)（多 env スクリーニング。本 PRD の観測量を使う側）、[069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md)（同じ欠落に別解を出している。§6 参照）、[done/054_inspect_run_10prd.md](done/054_inspect_run_10prd.md) / [done/041_metrics_sqlite_cache_10prd.md](done/041_metrics_sqlite_cache_10prd.md)（Viewer 層の実装先）、[../design/220_atari_env.jp.md](../design/220_atari_env.jp.md) §4.7（Atari 版の契約）。

## Context / Problem Statement

「ある閾値を越えたエピソードの割合」は、**題材をまたいで繰り返し必要になる**。にもかかわらず、現状は env ごとに手書きしている。

### 同じパターンが 2 env で独立に書かれている

**DropMerge**（`core/envs/dropmerge1/src/DropMergeEnv.cpp` の `GetScalar`）は、閾値をコードへ埋め込んだ 0/1 キーを持つ。

```cpp
if (key == "ep_double_suika_achieved") {
    if (!episode_just_ended_) return nan;
    return ep_double_suika_created_ > 0 ? 1.0f : 0.0f;
}
```

これを `M1.[42_env/81_double_suika_achieved_ema] = $env mean.ep_double_suika_achieved @train $exp_step $ema ema_alpha:0.001`（`DropMerge.txt:1311`）で率にしている。**閾値 `> 0` が C++ 側に固定されており、config からは動かせない。**

**Atari**（2026-09-05 追加、未コミット）は、キー側にパラメータを置く形で同じことをしている。

```cpp
if (const auto threshold = ParseGameScoreThreshold(key)) {   // "game_score.ge.[432]"
    if (!completion_available_) return std::numeric_limits<float>::quiet_NaN();
    return completed_game_score_ >= *threshold ? 1.0f : 0.0f;
}
```

**どちらも `AtariEnv` / `DropMergeEnv` のローカル実装**であり、3 つ目の env が必要になれば 3 度目を書くことになる。

### 現状で取れないもの

- **DropMerge のスイカ到達率**。`ep_max_rank` は `max`（`42_env/01`）と `mean`（`61`）しか出しておらず、「Rank 11 以上に到達した割合」が無い。Atari の 432 と同じ性質の指標なのに、閾値がコードに無いので出せない。
- **短手クリア率**のような下側の閾値（`ep_step` が N 以下）。現行の `ge` 相当しか手書きされていない。
- **過去 Run への遡及**。メトリクスを事前宣言していない Run では、閾値越え率は `metrics.jsonl` を直接舐めるしかない（[911](done/069_metrics_trace_channel_10prd.md) が指摘済み）。閾値は分析中に決まることが多く、事前宣言と相性が悪い。

### なぜ平均や max では代われないか

Atari の実測（`../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md`）で、判定に耐えたのは閾値越え率だけだった。

- **平均は二山分布の両端を潰す。** 同一設定の複製対をノイズ床にすると S/N は 0.41 で、腕の差が埋もれる。閾値越え率なら 2.4。
- **max は飽和する。** 評価セッションの `eval_episodes` を 10 にすると `>= 432` の max ベース率が 96.9% に達し、判別が ≥600 へ逃げる。N を上げるほど飽和が進む。
- **率は非飽和で、N 本すべての情報を使う。**

## 案

3 層に分ける。**Observation 層が本体**、Viewer 層と inspect_run は同じ導出を後付けで提供する。

### 案 1: Observation 層 — `<base>.ge.[N]` を env 非依存の共通 seam へ

env の `GetScalar` は生値だけを返す責務に戻し、閾値化は**全 env が通る 1 箇所**で行う。

- 対象経路は 2 つで、どちらも `inner->GetScalar(base_key, index)` を呼んでいる。
  - train `$env`: `core/anet-core/src/env.cpp:592` `DiscreteBatchEnvBase::GetScalar`（lane 方向へ集約するループ）
  - eval: `core/anet-core/src/env.cpp:316-325` `EvalSessionEnv::CaptureScalars`（採用 episode ごとに 1 サンプル）
- 共通の自由関数（例 `GetScalarThresholded(env, key, index)`）を作り、この 2 箇所から呼ぶ。`ParseScalarAggregationKey`（`core/anet-core/src/util.cpp:10`）が集約プレフィックスを剥がした**後**の base_key に対して働く。
- **NaN 契約が要点。** `NaN >= N` は false なので素直に比較すると NaN が 0.0 になり、集約の分母が「完了した env 数」ではなく num_envs になって率が壊れる。必ず `std::isnan` を先に見て NaN を透過させる。
- Atari のローカル実装（`AtariEnv::ParseGameScoreThreshold`）は削除し、この seam へ寄せる。DropMerge の `ep_double_suika_achieved` は `ep_double_suika_created.ge.[1]` と等価なので、キーを残すか置き換えるかは移行方針の判断（未決 Q3）。

これにより、**env 側は無改修で全キーに閾値越えが使える**。`mean.` がそのまま率になる意味論も現状と同じ。

### 案 2: Viewer 層 — 生系列への後付け閾値

Viewer には現在、系列の導出・変換の機構が無い（`apps/metrics-viewer/.../service/` に該当クラス無し）。ここへ「生系列 + 閾値 → 率」の導出を足す。

- **事前宣言が不要**なので、**過去 Run にも遡及して効く**。これが案 1 に対する独立の価値。
- 閾値を対話的に動かせる。Breakout の 432 は自明だったが、DropMerge の Rank 閾値のように「いくつが妥当か」を探る題材では、決め打ちより探索が要る。
- 0/1 の点列をそのまま描いても読めないので、**率にするための窓**（移動平均か固定ビン）が要る。ここが設計論点（未決 Q4）。
- 実装先は `service` 層。SQLite キャッシュ（[041](done/041_metrics_sqlite_cache_10prd.md)）の上で走らせる形になる。

### 案 3: inspect_run.py — `--stat` の拡張

`--stat` は現在 `mean / last / first / min / max / count / population_std`（`viewers/metrics-tools/inspect_run.py:2291`）。ここへ閾値越え率を足す。

- 閾値越え率は**カウンタ 2 個で逐次計算できる**ので、既存の Welford 集計器（`inspect_run.py:1094` 付近）にそのまま乗る。
- 記法案: `--stat ge:432`。既存 choices は固定文字列なのでパラメータ付きを許す形へ変える必要がある。
- [911](done/069_metrics_trace_channel_10prd.md) が「`inspect_run.py` は分位点と閾値越え率を持たない」と書いている、その片側を埋める。

## 未決 Q

- **Q1: 比較子をどこまで用意するか。** `ge` は必須。`le` は「N 手以下でクリア」のような下側の指標に要る（DropMerge の `ep_step`）。`gt` / `lt` は境界の扱いだけの差なので後回しでよいか。
- **Q2: 記法。** Atari 版は `game_score.ge.[432]`。角括弧つきキーは `action_uqe_win_rate.[0]`（`DropMerge.txt:1322`）に前例があり、config パーサは通る。ただし `:` を 1 個含むトークンは属性として解釈される（`observers.cpp:1265`）ので `ratio:432.foo` 形は破綻する。ドット区切りを維持するか、別記法にするか。
- **Q3: DropMerge の既存キーの移行。** `ep_double_suika_achieved` は `ep_double_suika_created.ge.[1]` と等価。残す（互換）か、置き換える（過去 Run の系列が切れる）か。**タグ名を変えると `metrics.jsonl` の系列が分断される**ので、既存 Run と繋ぐなら残す側。
- **Q4: Viewer での窓の取り方。** 0/1 点列を率にするには移動平均か固定ビンが要る。EMA（既存の `$ema` と同じ alpha 体系）に寄せるか、ビューポート幅に連動した可変ビンにするか。後者は拡大縮小で値が変わるので読み手が混乱しうる。
- **Q5: 案 1 と案 2 の役割分担。** 両方あると同じ数字が 2 経路で出る。事前宣言（案 1、判定用の正本）と後付け（案 2、探索用）で用途を分けるのが素直だが、値が食い違ったときの正本をどちらにするか決めておく必要がある。分母（案 1 は「その step で完了した env 数」）が案 2 で再現できるかも要確認。
- **Q6: [911](done/069_metrics_trace_channel_10prd.md) との境界。** あちらは「1 件 = 1 レコードのチャネル」を作って分位点と閾値越え率を後処理で自由に出す方向。本 PRD はスカラーメトリクス側で閉じる最小解。**両方やる必要は無い可能性がある**ので、911 を進めるなら本 PRD の案 2/3 は不要になりうる。着手前に整合を取る。

## 着手条件

**現時点では着手しない。** Atari 側は 2026-09-05 のローカル実装で当面の用が足りている。

着手の引き金になるのは次のいずれか。

1. **DropMerge でスイカ到達率が実際に要るとき。** そこで 3 度目の手書きをするくらいなら案 1 を通す。
2. **3 つ目の env が同じものを要求したとき。**
3. **閾値を後から動かしたくなったとき**（案 2 が主目的になる）。

なお案 1 は既存の集約機構（`ScalarAggregation` / `ScalarSampleAccumulator`）に一切手を入れない見込みで、変更は `env.cpp` の 2 箇所と共通ヘルパ 1 本、および各 env のローカル実装の削除に留まる。案 2 と案 3 は独立に着手できる。
