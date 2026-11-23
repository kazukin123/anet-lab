# PRD 045: EmaFilter バイアス補正ウォームアップ

- 対象: `core/anet-core/include/anet/util.hpp` の `EmaFilter<T>`
- テスト: `core/anet-core/src/util_test.cpp`(新規)
- 関連: CONTEXT.md「バイアス補正EMA」

## 1. 背景と課題

`EmaFilter::Update()` は初回サンプルを一発目採用(`value_ = x`)する。初回が外れ値の場合、
その影響は重み `(1-α)^t` でしか減衰せず(α=0.01 なら数百 step)、メトリクス曲線の序盤が
長く乱れる。

制約: ウォームアップ期間中の出力を欠損させる解(溜めてから出力)は、複数 tag 間で
step 系列がずれて比較性が壊れるため不可。metrics SQLite キャッシュ(PK=序数、PRD 041)
とも非互換。step 1 から欠損なしで値を出しつつ、序盤のバイアスだけを消す。

## 2. 解決方式: ゼロ初期化 + バイアス補正(debiased EMA)

Adam / TensorBoard smoothing と同一の標準方式。

- 内部値: `v_t = (1-α)·v_{t-1} + α·x_t`(v_0 = 0)
- 観測済み重み和: `w_t = (1-α)·w_{t-1} + α`(w_0 = 0)
- 読み出し: `Value() = v_t / w_t`

性質:

- 出力は常に「観測済みサンプルの指数重み付き平均」(重み和=1 に正規化)
- t=1 で出力 = x₁、t=2 で `((1-α)x₁+x₂)/(2-α)` ≈ 単純平均、以降なめらかに純 EMA へ収束
- 初期値バイアスの減衰が幾何級数 (1-α)^t から O(1/t) に改善
- 切替点なし・調整パラメータなし。ウォームアップ長は実質 ~1/α で α から自動的に整合
- `weight_` を内部値と同じ漸化式で更新するため、`SetDecay()` で α が途中変更されても
  補正は正確(一様 α の仮定を置かない)
- `1 - pow_t` の差分を取らず重み和を直接蓄積するため、極小の float α でも桁落ちによる
  ゼロ除算を起こさない

## 3. API 契約

公開シグネチャは一切変更しない。内部状態に `weight_`(T、初期値 0)を追加。

| メンバ | 契約 |
|---|---|
| `Update(x)` | 初回(`!init_`): `value_ = decay_·x; weight_ = decay_; init_ = true`。以降: `value_ += decay_·(x-value_); weight_ += decay_·(1-weight_)` |
| `Value()` / `operator T()` | `init_ ? value_ / weight_ : value_` |
| `Set(v)` / `operator=` | `value_ = v; weight_ = 1; init_ = true`(補正なし=ウォームアップスキップ。「既知の値から明示スタート」の意味を保存) |
| `EmaFilter(decay, value)` | 同上(`weight_ = 1` で開始) |
| `Restart()` | 現在の公開値を `value_` へ保持してから `init_ = false; weight_ = 0`。値は次の Update まで読める(現行契約維持) |
| `IsInitialized()` | 初回 Update 後 true(現行同一) |
| NaN/Inf 入力 | 現行どおり skip。`weight_` も進めない(スキップしたサンプルは重みに数えない) |
| `SetDecay(d)` | finite かつ `0 < d <= 1` を検証し、以降の `weight_` 更新に新 α を反映する。失敗時は旧 decay を保持する |
| 既定 decay | 浮動小数点型は 0.01、整数型は 1。arithmetic 型の公開契約は維持する |

数値安全性: `weight_` は正の decay を最初に代入し、その後は正の範囲で 1 へ漸近する。
差分 `1 - pow_t` を計算しないため、float の丸めで分母が 0 になる経路を持たない。
全コンストラクタと `SetDecay()` は decay の nonfinite、0以下、1超を `ANET_SYSTEM_ERROR` で
fail-fast する。

## 4. 影響範囲

呼び出し側の変更は一切なし。挙動が変わるのは以下2箇所の「序盤の出力値」のみ:

- `MetricsLogObserverBase::val_ema_`(observers.cpp)— is_ema 指定の全メトリクス
- `Trainer::reward_ema_`(trainer.cpp、α=0.001)— `REWARD_EMA` / `TRAIN_REWARD_EMA`

いずれも改善方向(初回サンプルを引きずらなくなる)の変化であり、互換 flag は設けない。
既存 Run との曲線比較への影響は序盤のみで、終盤平均のブレ幅基準の比較には影響しない。
EmaFilter はシリアライズ対象外(確認済み)のため保存形式への影響なし。

## 5. 代替案と却下理由

- 案B(累積平均→固定α遷移、`α_t = max(α, 1/t)`): 学習メトリクスは序盤こそ激しく
  トレンドするため、ウォームアップ中の一様平均が古い値を引きずりラグが出る。
  t=1/α に切替点(軽い折れ目)。warmup_n を設定可能にしたくなる誘惑があり
  設定項目ゼロの案Aに劣る
- 案C(バッファに溜めてウォームアップ後に遡及出力): MetricsLogger に過去 step への
  遡及出力 API が必要になり責務が漏れる。SQLite キャッシュ(PK=序数)と非互換。
  グラフ冒頭が同一値のフラット線になる artifact。ライブビュー遅延
- 案E(分散ベースのウォームアップ終了自動判定): 平滑と収束判定の2関心を1機構に
  混ぜる。案Aでは判定自体が不要(1/t で自動減衰)

## 6. テスト(util_test.cpp 新規作成)

既存テストの Catch2 規約に従う。

1. t=1: 出力 = x₁
2. t=2: 出力 = `((1-α)x₁+x₂)/(2-α)`(手計算値と一致)
3. t ≫ 1/α: 無補正 EMA(現行数式の参照実装)と数値一致(漸近収束)
4. `Set()` / `EmaFilter(decay, value)` 後: 補正なし(従来挙動と一致)
5. `Restart()` 後: 値は次 Update まで保持、次 Update からウォームアップ再突入
6. NaN/Inf 入力 skip で `weight_` が進まない(skip 前後と次の有限入力で確認)
7. `SetDecay()` 途中変更: 「観測サンプルの重み付き平均・重み和=1」が参照実装と一致
8. 不正 decay が fail-fastし、失敗した `SetDecay()` が旧 decay を保持する。整数型の既定
   decay=1 が最新値フィルタとして動作する

## 7. 受け入れ基準

- 上記8観点を含む7テストケースが緑
- 既存テストスイートに退行なし
- 公開シグネチャ無変更(呼び出し側の diff がゼロ)
