---
id: issue-20261111-0ddffa
title: RndNoveltyEstimator に 2 つの統計方式が同居して分岐が 5 つの関数の 9 か所に散り、Estimate と Learn も前半の約 20 行を同じ形で書いている
status: 解決
raised: 2026-11-11
phase: 監査
scope: core/anet-core/src/intrinsic_rnd.cpp・intrinsic_rnd.hpp の RndNoveltyEstimator と RndErrorStatistics（RND の内的報酬の毎 update の経路と checkpoint）
kind: コード
area:
  - 321_agent_core
type: リファクタリング
domain: 学習基盤
ai_judge: 残件無し
ai_judged: 2026-11-16
docs:
  - docs/memo/173_intrinsic_reward_novelty_10prd.md
  - docs/memo/123_rnd_intrinsic_reward_scale_10prd.md
prd: "[[130_rnd_estimate_statistics_perf_10prd]]"
priority: 高
depends: []
parent:
value: 3
urgency: 5
risk: 2
size: 5
wsjf: 2
resolved: 2026-11-16
resolution: 対応済み
memo:
tags:
  - RND
---

## 背景

2026-11-10〜11 のリファクタリング候補の抽出（C++ の本体を 16 単位に分けて読み、観点は docs/quality/refactor.md による）で見つかった。開発者が候補の一覧から選び、起票を指示した。2026-11-11 に今のコードで場所を確かめ直した。

PRD123 は 2026-11-11 に warm-up と誤差の正規化の選択肢を足した。PRD173（内的報酬の新しさと方策の組）の再グリルが次に控えている。

## 事象

- 統計方式の同居。
  - `statistics_mode` が whiten なら `normalizer_` と `error_stats_` を使い、variance_feedback なら `feedback_stats_` を使う。切り替えは `if (feedback_stats_)` で、コンストラクタ、`Estimate`（core/anet-core/src/intrinsic_rnd.cpp:459、486）、`Learn`、`Save`（583）、`Load`（621、631、635）にある。
  - variance_feedback のとき、`normalizer_` は空のままで、`error_stats_` は構築されるが使われない。どちらの組が有効かは、`feedback_stats_` が値を持つかどうかだけで表されている（core/anet-core/src/intrinsic_rnd.hpp）。
  - `RndErrorStatistics` も同じ形で、`use_ema_` によって累積と EMA の 2 組のメンバを持ち、メンバ関数がすべて `use_ema_` で分かれる（intrinsic_rnd.cpp:69、104、113 ほか）。
  - `VarianceFeedbackStatistics::UpdateErrors` は、直前に `NormalizeAndUpdate` が同じ件数で件数を進めていることを前提に件数を逆算する。この呼ぶ順の前提は、型にも名前にも出ていない。
  - 診断値 error_mean・error_std の書き込みが、方式ごとに `Estimate` と `Load` の両方にある。
- `Estimate`（intrinsic_rnd.cpp:446 から）と `Learn`（519 から）の前半が同じ形である。CPU への転送と uint8 の観測の float 化の 3 行、入力の転送・教師と予測の forward・誤差の計算の並び、誤差が有限かの検査とそのエラー文（478 と 548 から）が同じで、違いは `Learn` が予測の前に勾配を有効にする 1 行だけである。

## 影響

- 統計方式を 1 つ足すと、5 つの関数の 9 か所に枝を足すことになる。どのメンバがどの方式で生きているかは、分岐を全部追わないと分からない。PRD173 で新しさの推定を広げると、この上に足すことになる。
- 前処理やエラー文を直すと、`Estimate` と `Learn` の 2 か所を同じに直す。

## 対応方針

方式ごとの統計（観測の正規化、誤差からの倍率、保存と復元、診断の平均と標準偏差）を 1 つの内部の型にまとめ、whiten 用と variance_feedback 用の 2 つの実装を構築時に 1 つ選ぶ。`RndErrorStatistics` も累積用と EMA 用に分ける。`Estimate` と `Learn` の共通の前半は private の小さな関数にする。checkpoint の並びと書式名、設定キー、診断キー、ログの文言、プロファイルの区間名は変えない。warm-up の条件が方式をまたいで書かれているので、分けたあとにどちらの側へ置くかを PRD123 で置いた判断と突き合わせて決める。

## 対応内容

- 2026-11-16: PRD130 のステージ 2 で、累積・EMA の誤差統計と whiten・variance_feedback の統計方式を内部の型へ分けた。構築時の選択へ方式分岐を集め、feedback の観測更新前の件数を同じ採点内で保持し、forward と誤差計算を共通化した。Clone・Save/Load 後の混合観測の学習を保護するテストを追加。最終 Debug・Release 全ビルド、既定 196,027 assertions、関連 9,886 assertions、採取器の Python 7 tests が成功した。通常 5 構成と同じ checkpoint からの継続 3 構成の 100k Run で、時間・資源以外の全 scalar 計 343,067 点が一致し、受入 11・12・14・15 を確認した。既定 suite の初回の workspace 履歴の置換例外と、断定を変えない同 seed の再試験は別記録で保持。日英の設計文書も追随した。性能比較 13 は開発者の指定により未実施で、残件有りを維持する。[実装メモ](../../memo/130_rnd_estimate_statistics_perf_20impl.md)に経過と結果を記録する。

- 2026-11-16: 開発者の評価 Run の PAUSE 完了の連絡により、受入 13 の性能比較を再開。採取器を stage1 対 stage2 の主・副・Taiga の交互比較へ拡張し、機能検証済み Release 実行体の選択と実効設定を検査する。採取器の Python 9 tests が成功。測定結果は取得後に追記する。

- 2026-11-16: 主・副の性能比較は合格。Taiga は休止 CPU の全プロセス区間の検査で 2 回拒否され、不採用記録を保持した。区間別診断の実測付近の 2 サンプル間の CPU 増加は 0.375 秒だったが、開始の読み取りは測定始点を越えており、実測全体の値とは断定できない。休止 Run の記録は全区間で停止していた。CPU 上限 1 秒を維持し、起動前から CPU 読取完了までの単調時計の経過秒数が、測定始点の Runner 経過秒数より短いことも要求して、出力遅延に依存しない包含確認を加えた。全区間の記録停止も検査する。レビュー指摘を保護する Python 14 tests が成功した。性能判定の式・経験数・固定実行体は変えず、Taiga の全採取をやり直す。詳細と不採用記録は実装メモへ保存した。

- 2026-11-16: Taiga の 3 回目は、4 本目で 1 回だけの CPU 読み取りが保守的な時刻条件に届かず停止したため、一式を不採用のまま保持。時刻条件は変えず、同じプロセスの CPU 履歴から測定開始前と証明できる最後の時刻を選ぶ方式へ修正した。CPU 値の大小では選ばない。Python 16 tests と、Windows API の生存プロセス読取・終了プロセス拒否の実検証が成功。再レビュー後に全件を新たに採る。

- 2026-11-16: 受入 13 を完了。主・副・Taiga の固定実行体の交互 3 巡、計 36 本が正常終了し、中央値の差の絶対値が基準の幅以内だった。差と幅は主 0.109000 / 0.581997 秒、副 -0.110999 / 0.692996 秒、Taiga -0.119499 / 1.121995 秒。設定・実行体・経験区間・順序・休止検査を原記録と照合し、不採用 3 組と診断は除外した。機能受入 11・12・14・15 と合わせ、この課題の残件は無しと判定。[実装メモ](../../memo/130_rnd_estimate_statistics_perf_20impl.md)と `acceptance/stage2-performance-evidence.json` に全結果を保存した。

- 2026-11-16: 開発者が選んだ 2 モデルレビューの指摘 3 件を修正。212 日英の性能受入の到達点、`WhitenStatistics` の関数群の区切りと追加した型の前方宣言の位置、有限性検査と warm-up の意図コメントを揃えた。計算処理・データメンバは不変で、修正後の Debug・Release 全ビルドと関連 9,886 assertions が成功。文書対応と非機能差分の照合も成功し、同じ両モデルへ再レビューする。詳細は[実装メモ](../../memo/130_rnd_estimate_statistics_perf_20impl.md)の「2 モデルレビューと指摘対応」と `acceptance/cc-r2-*` に保存した。

## 備考

- 2026-11-15 状態を `未着手` から `調査中` へ変更。工程の出来事による（PRD130 の再グリルを始めた）。
- 2026-11-16 状態を `調査中` から `対応中` へ変更。工程の出来事による（[PRD130 の実装計画](../../memo/130_rnd_estimate_statistics_perf_20impl.md)を保存した。今回は段取り整理だけで、実装は未着手）。
- 2026-11-16 状態を `対応中` から `解決` へ変更。コミットの準備で開発者が解決の候補を選んだ。統計方式の整理、受入 11〜15、2 モデルの再レビューを完了し、AI の判定は残件無し。
