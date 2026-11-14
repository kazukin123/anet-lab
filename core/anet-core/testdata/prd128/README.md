# PRD128 検証資材

正本は [PRD128](../../../../docs/memo/128_dqn_agent_refactor_bundle_10prd.md)、結果は [実装メモ](../../../../docs/memo/128_dqn_agent_refactor_bundle_20impl.md) に記録する。

`run_validation.py` は PRD125 の実行・入力生成・scalar 正規化・suite・性能の検査を呼ぶ。原本の資材は変えない。保存先は `.scratch/prd128/acceptance/` で、既存 label の上書きは拒否する。

## 基準と入力

実装前の Debug・Release を AGENTS.md の MSVC 初期化経由でビルドし、`snapshot` で実行体・DLL・設定・core テストを保存する。PRD129 適用後を基準にする。`prepare` は保存した GUI テストから設定を解決し、PRD125 の FP32 31 組、LunarLander/Atari の RND、DropMerge の全診断購読を凍結する。6 群の有限点の不足があれば U3 の追加構成を確定して記録する。

条件は seed 1、8 lane、batch 16、warmup 256、replay ratio 1、8,704 experience、評価停止、決定的 backend、FP32。GridMaze/LunarLander は CPU、Atari qbert/DropMerge は CUDA。CPU は最大 4 組を並行し、CUDA は逐次実行する。

RND の追加構成は intrinsic.learn_batch_size=8、intrinsic.rnd.warmup_steps=256 を明示する。8 lane で成立し、短い Run 内で補助学習も通すためである。修正前の入力は保持し、`_b8` 付きの別名で採取する。

## 比較

`repeat` は保存した Debug 実行体で各構成を 2 回実行し、`compare` はその 1 回目と変更後を比較する。比較対象は全 scalar（時間・資源の 3 prefix 以外）、`config/DefaultDQNAgent.txt` の全行、Agent の json 設定記録、WARN/ERROR の文言と順。時刻・Run 名・Run パスだけをログから除く。process のログと終了値、正規化結果、設定ハッシュ、成果物の比較を label の下へ保存し、summary に成功・失敗を記録する。

RND の LunarLander が終了時に保存した checkpoint から、新旧をそれぞれ 8,704 experience 続けて比較する。

## 速さと破壊検査

ステージ 1・3・4 で PRD125 の performance を呼ぶ。Release、Atari qbert、IQN、128 lane、batch 256、捨て Run 後に新旧交互 3 巡以上。変更後の中央値が基準の最大値以下を合格にする。ステージ 3 は RND を重ねる。他の Runner・test・build の実行中は測定しない。

指標の並び替えの破壊検査は、共通型に載る群と載らない群を選び、隣接 2 指標の交換でビルドが止まることと、復元後にビルドが成功することを記録する。

## 実施記録

- U2（2026-11-14）: 本体変更前に actions・target_returns・next_state.terminals を 1 件ずつ未定義にして CPU 転送すると、すべて例外になった。ERROR のログは出なかった。記録は `.scratch/prd128/characterization-samples-before-r2.xml` と同名の `.log`。全フィールドを入れた CPU/CUDA 転送と合わせて 47 assertions が成功した。
- 基準の既定 core suite は失敗 0、592.44 秒で成功した。追加した特性テスト群は `.scratch/prd128/characterization-before-final.xml` で 172 assertions、失敗 0。
- U3: `diagnostics_DropMerge_explicit_sources` の 1 回目で、6 群すべてに有限点があった。可塑性 14 点、policy churn 12 点、replay fit 18 点、noisy 13 点、attention 8 点、gradient noise 18 点（群ごとの tag prefix に属する有限の scalar 点数）。2 回目も正規化した全 51,012 点が一致した。取得先の無い @full の旧 Agent 指標だけ `$agent` を明示している。最初に失敗した入力と Run は保持し、出荷設定は変更していない。
- 保存済み RND checkpoint の継続比較の入口も、本体変更前に成功した（`.scratch/prd128/acceptance/baseline-checkpoint-summary.json`）。新旧で同じ checkpoint から各 8,704 experience 続け、scalar・設定・ログが一致した。
- 基準 34 構成は反復・成果物の比較を完了した。現用の基準は `.scratch/prd128/acceptance/baseline-checked-summary.json`。この基準を使う比較には `--reference baseline-checked` を指定する。旧入力の失敗と重複した GPU 実行は原記録として保持し、基準には含めていない。
- ステージ 1 は 34/34 構成で全比較項目が一致した（`stage1-summary.json`）。関連テストは 205 assertions 成功。Release の速度は変更後の中央値 40.46 ms/更新が基準の最大値 42.56 ms/更新以下で合格（`stage1-performance-summary.json`）。
- ステージ 2 は Debug 全ビルドと関連 757 assertions が成功し、34/34 構成で全比較項目が一致した（`stage2-summary.json`）。入口のコマンド、後続ステージ、未実施は実装メモへ追記する。
- ステージ 3 は Debug・Release 全ビルド、全 intrinsic 4,707 assertions、34/34 構成の比較、旧 RND checkpoint の 31,156 scalar 点の継続比較が成功した。RND 有効の性能は変更後中央値 70.30 ms/更新 ≤ 基準最大 73.32 ms/更新で合格（`stage3-performance-summary.json`）。
- ステージ 4 は Debug・Release 全ビルド、DQN 関連142,325 assertions、34/34構成の比較が成功。noisyとweight normの隣接指標交換はコンパイルで検出され、復元後のobjectビルドは成功した。性能は変更後中央値39.63 ms/更新 ≤ 基準最大41.91 ms/更新で合格（`stage4-performance-summary.json`）。
- ステージ 5 の最終比較はCPU30・CUDA4の全34構成、973,936 scalar点と全成果物が一致（`stage5-summary.json`）。最終版のDebug・Releaseビルドとcore既定179,995 assertionsが成功した。skip1件はCUDA無し用の検査で、基準と同じ理由である。途中停止した `stage5-cpu`・`final-suite` は受入へ含めず、再実行した `stage5-cpu-r2`・`stage5-gpu-r2`・`final-suite-r2` を採用した。
- 独立レビューで不足した指定入力 `Greedyy` を既存の生成テストへ追加し、ConfigData・DTOの両経路でD4の指定文言による拒否を確認した。追加後の `[prd128]` は240 assertions成功（`stage5-review-fix-characterization.xml`）。core既定suiteの上記件数は、この入力1値を追加する前の結果である。本体への変更はない。
- 追加の2モデルレビュー後、同じ値の定数を使うメッセージ生成とテストseedの定数化、コメントの修正を行った。修正後のDebug・Release全ビルドと関連15,233 assertionsが成功（`cc-fix-related.xml`）。34構成・既定suite・checkpoint・性能の上記結果はこの追加修正前の実行体によるもので、再実行していない。正常入力の数値処理・条件・順序・seed・保存形式は変更していない。
