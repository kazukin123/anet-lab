# PRD084 時計の種類と残件の追加修正計画

## 概要
正本は .scratch/prd084/handoff_terms_and_residuals_2026-10-04.md。修正A、B1、B2、文書4章を実施する。ユーザー判断が必要なブロッカーは0。B3は任意の小さな最適化であり、今回は挙動不変の改名・統合を明確に検証するため入れない。

## 主な変更
- NetworkForwardClockはkNoneに加えてkTransientRowsとkContinuingRowsの2種類。kActorをkContinuingRows、kLearner/kProbeをkTransientRowsへ移す。型名、clock_kind、検証APIは維持し、旧名aliasは残さない。
- LinearProjectionModuleの2判定と対応エラーメッセージだけを置換。episode保持の局所lanesをrows、コメントを行にする。構造・演算・抽選・保持・設定値は変えない。
- 呼び出し側とテストのenum参照・照合文言を移行。[noisynet]テストは定数・メッセージ置換のみ。episode_start、noisy.hold=episode、episode_atは維持する。
- dqn_based_heads.cppとnn_impl.hppのBOMをHEADに合わせる。変更した全trackedファイルのBOMと改行をHEADのbytesに照合する。新規・HEAD未登録ファイルは既存の形式を保持する。
- PRD084の指定箇条書きを差し替え、設計130の日本語/英語に時計の2種類を追記。READMEとreview_verification.mdに現行受け入れ・B1のRun値と中央値を保存。PRDに完了状態は書かない。
- 性能補助スクリプトに比較組の明示選択と中央値集計を追加し、off/hold_episodeのみ3反復A/B交互で取り直す。他10組と旧Run/summaryは改名前の測定として保持する。

### レビュー追加修正（2026-10-05）
- PRD冒頭へ今回の改訂日・概要を追加し、実装計画20・21・22の必須5節を整理する。実施結果の履歴は保持する。
- agent.txt・metrics_scalar.txt の今回追加した26行だけ、プロファイル境界を : にする。公開 ConfigManager の解決済み全キー・文字列値と型付き設定を前後比較する。
- optimizerの3テストを実際のTDLearnerのUpdateFromBatch・Save/LoadとNetworkModelの保存読込へ移し、テスト専用のSetupOptimizer公開化を削除する。本体APIと演算は変更しない。
- OFF回帰3本は500k、旧checkpoint読込は256、結合2本は500k以上という予算契約を実行・集計側で揃える。予算延長の受け入れと不正な予算の拒否を1挙動ずつRED→GREENで確認する。

## テスト
- 今回の追加修正では、σの独立decayとμ-only凍結、σ込みclip、optimizerのgroup/step/moment復元を公開の更新結果とparameter/gradientから確認する。既存成功を確認後、3テストを1本ずつ移行してGREENを維持する。
- Public surface: Network::ValidateForwardConfig、Forward、DQN呼び出し・通常Runner。
- 挙動追加ではなくGREENからの改名refactor。既存[noisynet]を実行して成功を確認し、定数・文言のみ変更して再実行する。期待値・assertion・fixtureの挙動は変更しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
core/anet-core/bin/Debug/anet-core-test.exe
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

- MSVC初期化経由Debug全体ビルド、[noisynet]、全core。指定NN6ファイルのActor|Learner|Probe|lane一致0件を確認。
- 最終Release Runnerをビルドし、実行体SHA、依存DLL、設定、seedを新出力に保存する。無効時Atari既定/SN効率優先/CPU LunarLanderを各500kで旧基準へ比較し、checksum/点数/既存設定値を照合する。今回対象外の結合2本等は再実行しない。
- B1は同一GPUに別学習Runが無い時だけ開始。前後のGPU使用状況と実行プロセスを保存し、別学習Runが動いていれば測定だけ保留して報告する。
- off/hold_episodeは各3反復A/B、224k経験、既存warmup200k・スケジュール・比較軸を維持。各Runの時間/学習中exp/s/メモリ、側別中央値と非軸設定一致を保存する。
- 同じ参照構成14本で学習中exp/sが1005〜2081だった事実と、数%の差はこの測定で読めない旨を記録する。

## 前提
- 今回のレビュー修正はテスト・設定記法・検証補助・文書に限定する。設定の実効値と本体ソースのSHAを維持するため、既存の最終Release OFF回帰記録を保持し、学習・性能Runは追加しない。
- 既存dirty作業ツリーを編集前コピー・差分として記録し、今回変更分の一覧を管理する。
- episodeに代わる新語、全12組の再測定、LinearProjectionModule構造変更、B3、CONTEXT.md/ADR0048の変更は行わない。
- 旧artifact・summaryを削除/書換しない。git add/commit/pushは行わない。
- 保留/失敗は成功扱いせず、理由とログ位置を完了報告に残す。

## 実施結果

指定のA/B1/B2と文書更新を完了。Noisy47件/10497 assertions、全core664passed/1skipped/34680 assertions、Debug/Releaseはexit0。OFF3構成は各500kで旧checksum・点数・既存設定値が完全一致。性能2組は各3反復A/B・12 Runすべてexit0、同一GPUの前後記録に別学習プロセス無し。中央値と各Run値はcore/anet-core/testdata/prd084/review_verification.mdとREADME.mdへ保存。B3と文書5章の対象外は未実施、保留無し。初回ログ処理の停止と2か所の照合文字列移行漏れは記録を保持し、修正後の成功結果だけを採用した。PRDに状態は追加せず、git add/commit/pushは行っていない。


## レビュー追加修正の結果（2026-10-05）

前回の指摘5点を修正した。PRDの改訂記録、3計画の5節、設定26行のプロファイル境界、公開更新経路でのoptimizerテスト3件、結合Runだけの予算延長を反映した。Debug全体ビルドはexit0、Noisy49件/10,690 assertions、全core666成功/1既存skip/34,873 assertions。Python予算4件と既定葉監査（errors0）、公開ConfigManagerによる6構成の全キー・文字列値・型付き設定一致（44 assertions）、保存済み6caseの集計CLI再判定も成功した。

本体SHAと現行Release SHAを維持しており、学習・性能Runは追加していない。最終ReleaseのOFF3本のchecksum根拠はContext最終検証の記録を維持する。BOM/LF、NN6ファイルのRL語0件を再確認した。途中の旧manifest/fixture/includeの失敗は最終成功と区別し、旧artifactとともに保持した。保留無し、git add/commit/push無し。詳細と今回の変更12本はcore/anet-core/testdata/prd084/review_verification.mdの「再レビュー指摘の修正と検証」、ログと設定採取は.scratch/prd084/review-followup-20261005/を参照。
