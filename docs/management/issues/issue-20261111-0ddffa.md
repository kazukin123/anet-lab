---
id: issue-20261111-0ddffa
title: RndNoveltyEstimator に 2 つの統計方式が同居して分岐が 5 つの関数の 9 か所に散り、Estimate と Learn も前半の約 20 行を同じ形で書いている
status: 未着手
raised: 2026-11-11
phase: 監査
scope: core/anet-core/src/intrinsic_rnd.cpp・intrinsic_rnd.hpp の RndNoveltyEstimator と RndErrorStatistics（RND の内的報酬の毎 update の経路と checkpoint）
kind: 成果物
area:
  - 321_agent_core
type: リファクタリング
domain: アルゴリズム
ai_judge: 未確認
ai_judged:
docs:
  - docs/memo/924_intrinsic_reward_novelty_10prd.md
  - docs/memo/123_rnd_intrinsic_reward_scale_10prd.md
prd:
priority: 中
depends: []
parent:
value: 3
urgency: 5
risk: 2
size: 5
wsjf: 2
resolved:
resolution:
memo:
tags: []
---

## 背景

2026-11-10〜11 のリファクタリング候補の抽出（C++ の本体を 16 単位に分けて読み、観点は docs/quality/refactor.md による）で見つかった。開発者が候補の一覧から選び、起票を指示した。2026-11-11 に今のコードで場所を確かめ直した。

PRD123 は 2026-11-11 に warm-up と誤差の正規化の選択肢を足した。PRD924（内的報酬の新しさと方策の組）の再グリルが次に控えている。

## 事象

- 統計方式の同居。
  - `statistics_mode` が whiten なら `normalizer_` と `error_stats_` を使い、variance_feedback なら `feedback_stats_` を使う。切り替えは `if (feedback_stats_)` で、コンストラクタ、`Estimate`（core/anet-core/src/intrinsic_rnd.cpp:459、486）、`Learn`、`Save`（583）、`Load`（621、631、635）にある。
  - variance_feedback のとき、`normalizer_` は空のままで、`error_stats_` は構築されるが使われない。どちらの組が有効かは、`feedback_stats_` が値を持つかどうかだけで表されている（core/anet-core/src/intrinsic_rnd.hpp）。
  - `RndErrorStatistics` も同じ形で、`use_ema_` によって累積と EMA の 2 組のメンバを持ち、メンバ関数がすべて `use_ema_` で分かれる（intrinsic_rnd.cpp:69、104、113 ほか）。
  - `VarianceFeedbackStatistics::UpdateErrors` は、直前に `NormalizeAndUpdate` が同じ件数で件数を進めていることを前提に件数を逆算する。この呼ぶ順の前提は、型にも名前にも出ていない。
  - 診断値 error_mean・error_std の書き込みが、方式ごとに `Estimate` と `Load` の両方にある。
- `Estimate`（intrinsic_rnd.cpp:446 から）と `Learn`（519 から）の前半が同じ形である。CPU への転送と uint8 の観測の float 化の 3 行、入力の転送・教師と予測の forward・誤差の計算の並び、誤差が有限かの検査とそのエラー文（478 と 548 から）が同じで、違いは `Learn` が予測の前に勾配を有効にする 1 行だけである。

## 影響

- 統計方式を 1 つ足すと、5 つの関数の 9 か所に枝を足すことになる。どのメンバがどの方式で生きているかは、分岐を全部追わないと分からない。PRD924 で新しさの推定を広げると、この上に足すことになる。
- 前処理やエラー文を直すと、`Estimate` と `Learn` の 2 か所を同じに直す。

## 対応方針

方式ごとの統計（観測の正規化、誤差からの倍率、保存と復元、診断の平均と標準偏差）を 1 つの内部の型にまとめ、whiten 用と variance_feedback 用の 2 つの実装を構築時に 1 つ選ぶ。`RndErrorStatistics` も累積用と EMA 用に分ける。`Estimate` と `Learn` の共通の前半は private の小さな関数にする。checkpoint の並びと書式名、設定キー、診断キー、ログの文言、プロファイルの区間名は変えない。warm-up の条件が方式をまたいで書かれているので、分けたあとにどちらの側へ置くかを PRD123 で置いた判断と突き合わせて決める。

## 対応内容
