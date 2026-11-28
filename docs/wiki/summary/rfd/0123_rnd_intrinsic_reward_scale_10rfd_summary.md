---
title: RND 予測誤差の尺度と warm-up
page_kind: サマリ
material: "[[docs/memo/0123_rnd_intrinsic_reward_scale_10rfd.md|0123_rnd_intrinsic_reward_scale_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/subjects/montezuma_revenge.md|Montezuma's Revenge]]"
  - "[[docs/wiki/concepts/subjects/kung_fu_master.md|kung_fu_master（KFM）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# RND 予測誤差の尺度と warm-up

## 元の材料

[RFD0123　RND 予測誤差の尺度と warm-up](../../../memo/0123_rnd_intrinsic_reward_scale_10rfd.md) blob:740af0d2f67a1ff3e95d86917a39cc08dbfa1bb8

## 要点

- 序盤の大きな予測誤差が標準偏差を膨らませ、学習後半で実効的な内的報酬の強さが変わっていた。
- warm-up 中の統計と採点開始後の統計の扱いを明示し、初期の過渡状態の影響を調べる。
- 誤差を累積統計で測る方式と、最近の値を重くする EMA を選べるようにする。
- 誤差の平均を引くか、尺度だけを合わせるかを区別し、内的報酬がゼロになる割合も読む。
- 終端のマスクを明示し、同じ β でも統計方式や warm-up が違う群を区別して比較する。

## 内容

### 目的と問題

DefaultDQNAgent の RND（RFD0116、既定の `statistics_mode = whiten`・`multiplier_mode = clipped`）は、1 step の内的報酬を β × clip((e − μ_e) ÷ σ_e, 0, L − 1) で作る（L は `multiplier_max`、既定 5）。e は predictor と target の出力の二乗誤差、μ_e・σ_e は誤差の平均と標準偏差（以下、合わせて「誤差の統計」）である。誤差の統計は、Run の最初から採点したすべての誤差の累積で、warm-up 中の誤差も入る。
内的報酬が、序盤を除いた誤差の統計で作った場合の 1/3〜1/17、1/7〜1/19（KFM の 10M 時点）に抑えられている。
抑え方は Run が進むと薄まり、実効の β（実際に効いている内的報酬の大きさ）が Run の中で 3〜5 倍に上がる。設定に書いていないスケジュールになっている。
平均を引いて 0 で切るので、内的報酬が付く状態は約 1 割で、残りは 0 になる。学習とともに誤差が下がると、今の誤差が累積の平均を下回り続ける（LunarLander・GridMaze で内的報酬が消えた現象と同じ形。wiki の内的報酬のページ）。

### 背景

内的報酬は 1 step あたり Munchausen の項の約 1/100 だった（外的報酬は clip され、雑魚 1 体で +1）。ボスに挑んでいた時期（s04 の 4〜6M、s09 の 4M・8M。RAM の `boss_hit` の平均で判定）にも、内的報酬の平均は前後と変わらなかった。外的報酬の少ない 0〜1M の区間だけは `03_bonus_share` が 0.23〜0.85、2M 台の区間は 0.037〜0.14 だった。
誤差の統計の逆算: `11_error_mean`（μ_e）と `12_error_std`（σ_e）は累積の値なので、最初の記録（200k exp step。Q の学習の開始と同じ）と 10M の値から、200k 以降の誤差だけの平均と標準偏差を Welford の合成の式で逆算した。
膨らみは 5 本で同じ形なので、β 0.01 と 0.1 の比較（記録の結論）は崩れない。ただし β の値は「この膨らみの下の値」であり、内的報酬は Run の中で 3〜5 倍に強まっていた。Montezuma は外的報酬がほぼ 0 なので、`03_bonus_share` は小さな内的報酬でも 0.93〜1.0 になる。
序盤の大きな誤差が、predictor が未学習なためか、観測の正規化が未熟なためかは分けていない（`56_agent_in` の tag は `@learn` で記録され、Q の学習が始まるより前の値が無い）。

### 確定事項

RND 単独の群は NGU への通過点とし、既定の式（NGU の倍率の形。β × (α − 1)、α = clip(1 ＋ (e − μ_e) ÷ σ_e, 1, L)）は変えない。 平均を引かない形と直近重みは、比べるためのモードとして足す。
whiten の誤差の統計の取り方を `intrinsic.rnd.error_norm.average_mode`（`cumulative`｜`ema`）で選ぶ。既定は `cumulative`（累積）。 `ema` は直近重みである。
直近重みは、採点した観測 1 件ごとに重みを 1 − 1 ÷ `ema_steps` 倍に減らし、重みの和で割って読み出す。 時定数は `intrinsic.rnd.error_norm.ema_steps`（採点した観測の件数、整数 ≥ 1、既定 1,000,000）。batch の N 件は、lane の順に 1 件ずつ入れたのと同じ値にする。
whiten の warm-up の中身を `intrinsic.rnd.warmup_mode`（`learn_predictor`｜`collect_errors`）で選ぶ。既定は `learn_predictor`。 `learn_predictor` は、warm-up 中に predictor を学習させ、誤差を誤差の統計に入れない。`collect_errors` は今の中身で、predictor を学習させず、誤差を誤差の統計に入れる。どちらも warm-up 中の倍率は 1 で、観測の統計は更新する。

### 仕様

warm-up は、採点した観測の件数が `warmup_steps` に達するまでの期間である（今と同じ）。その間にすることを `warmup_mode` が決める。
warm-up かどうかの判定は今と同じである。採点は batch の採点前の件数が `warmup_steps` 未満なら warm-up、`collect_errors` の `Learn` は採点後の件数が `warmup_steps` 以下なら進めない。
facade の credit の扱いは変えない。`Learn` を呼んだ分の credit は、結果によらず消費する。`learn_predictor` では warm-up 中も `learn_count` が増える。
`ema_steps = 1` は d = 0 で、統計は直前の 1 件だけになり、σ_e が 0 なので α は常に 1 になる。設定どおりの動きとして止めない。

### 実装契約

(1) の検証は、(2) の条件（`statistics_mode` など）によらず行う。ERROR の文には、設定キー・値・期待する値（列挙か範囲）を入れる。
`intrinsic.enabled = false` のときは、今と同じく `intrinsic.*` のほかの設定キーを読まず、検証もしない。
既にある検証（`warmup_steps`・`multiplier_mode`・`multiplier_max` など）は変えない。
識別子と誤差の統計のほかの中身と並び（`statistics_mode`、target、predictor、optimizer、観測の統計、target の出力の統計、採点した観測の件数）は、どの行でも今と同じである。

### 成功の測り方と受入

主の条件（ゴール 1・2）。 step の軸は exp step、区間の値は 1M ごとの区間の平均である。
新しい既定の 4 本すべてで、2M〜3M の区間の `12_error_std` を 9M〜10M の区間の値で割った比が 1.5 以下である（ゴール 2）。今の中身では、最初の誤差が二乗偏差の総量を占める間、標準偏差が件数の平方根に反比例して下がるので、この比は約 1.9（√(9.5 ÷ 2.5)）になる見込みである。1.5 は、その見込みと 1 の間に置いた閾値である。
新しい既定の 4 本の `01_bonus_mean`（3M〜10M の平均）を、今の中身の同じ seed の値で割った倍率（仮説の見込みは 10 倍以上）。今の中身の 2 本の、主の条件 4 と同じ比（見込みは約 1.9）。

### RFD0116・RFD0121・0173 との関係

173（NGU 型）は、エピソード内の報酬に一生の新しさの倍率を掛ける。倍率は本 RFD の誤差の統計を使うので、warm-up と統計の取り方の変更はそのまま効く。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [Montezuma's Revenge](../../concepts/subjects/montezuma_revenge.md)
- [kung_fu_master（KFM）](../../concepts/subjects/kung_fu_master.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
