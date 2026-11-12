# PRD125 DQN 学習器の更新と target の共通化 実装メモ

## 概要

[PRD125](125_dqn_learner_update_skeleton_10prd.md) の D1〜D19、A1〜A11 を実装・検証する。学習の値を変えず、更新の進行、Munchausen の評価、Bellmanターゲットの式をそれぞれ 1 か所に寄せる。

## 対象

- 基準コミット: `fd12182892deb25eb59e390f1c7163c1b0e3231c`。既存の staged・unstaged・untracked の変更は保持する。着手時点で `dqn_based_agent.cpp`・`.hpp` と既存の DQN テストに差分はない。
- 課題: [issue-20261103-ab98b7](../management/issues/issue-20261103-ab98b7.md)。実装計画の保存時に対応中とし、解決にはしない。

## 主な変更

1. ステージ 0: 現作業ツリーから Debug・Release をビルドし、実行体・DLL・設定・既存テストの写しを `.scratch/prd125/baseline/` へ保存する。commit・差分の有無・SHA-256 を記録する。`core/anet-core/testdata/prd125/` に比較資材と README を作る。PRD098 の短縮条件、PRD109 の隔離実行と保存、PRD084 の正規化を利用する。FP32 の既定 4 組と LunarLander 27 組を 2 回ずつ比較し、診断の点と設定 dump を検査する。BF16 の 4 組は CPU/CUDA の優先順と一致条件を D14 に従って記録する。既定 core suite の失敗ケース一覧を保存する。
2. ステージ 1: `Learner::UpdateFromSamples` に進行全体を移す。通常 Q の `ForwardCurrent`、分位点で共通の current forward、IQN の τ と診断を分け、`CurrentForwardResult` を介して引き渡す。分布名による進行の分岐は作らない。派生 3 クラスの上書きを削除し、継承した仮想関数を外から呼べる形に保つ。D6〜D9 の重み・検査・ログ・区間を統一する。設計文書 200 の日英の Learner の責務の行だけを追随する。前後比較・既定 suite・Release の性能測定を完了してから次へ進む。
3. ステージ 2: `ForwardMunchausenValues` と `MunchausenForwardResult` へ評価の進行を移す。出力キーと τ の要否は呼ぶ側が渡す。通常 Q と分位点の実空間の方策と soft価値の計算は残す。`MakeBellmanTarget` に実空間 bootstrap と任意の Munchausen項から作る式を寄せる。`CalcTargetQuantiles`・`CalcMunchausenTargetQuantiles` を除く。ADR0035 は冒頭へ注記だけを足す。単体テスト・破壊検査と全比較・性能測定を行う。
4. 自己点検後、課題の対応内容へ検証結果を記録し、独立したレビューエージェントへ `impl-plan` と `impl` を渡す。差し戻しを直し、OK 後に課題の AI 判定と派生一覧を更新する。commit・push は行わない。

## テスト

- Public interface / surface: Runner の凍結設定を入力とする学習メトリクス、既存の公開 `UpdateFromSamples` 呼び出し、テスト専用派生から公開する保護された target の計算。
- 優先 behavior: 同 seed の更新結果と診断の全点一致、PER 無効、skip、Munchausen 3 方式、TBO、BF16、終端でも残る Munchausen項。
- TDD 順序: 最初は既定 LunarLander の隔離 Run を 2 回通す比較を tracer bullet にする。比較資材の未実装による RED から最小実装の GREEN を確かめ、条件群を 1 群ずつ増やす。既存挙動の共通化は、変更前に固めた特性比較を GREEN のガードレールとして行う。ステージ 2 の式は手計算と比較する 1 テストを先に足して RED を確認し、最小実装で GREEN にしたあと、通常 Q/分位点 × Munchausen 有無 × TBO 有無 × 終端有無の 16 通りへ広げる。h の欠落・終端マスクの欠落・終端時 bonus の欠落を個別に入れ、落ちることと復元後の成功を記録する。
- 既存の 2 テストファイルの削除・変更を着手時点の写しと比較する。追加するテストは `[prd125]` に限る。

## 検証

- ビルドは AGENTS.md の MSVC 初期化を用い、`x64-Debug` と `x64-Release` を通す。Debug の core suite はステージ 0・1・2 で失敗ケース集合を比較する。
- FP32 の 31 組と、基準で再現した BF16 の組を各ステージで比較する。除外は `90_perf/`・`91_resource/`・`92_resource/` のみ。設定 dump の各条件、replay fit・IQN・Munchausen の点を検査する。IQN は 7 定義を確認し、実点の範囲は改訂した A1 の PER 有無による分岐に従う。
- A3・A5 の検索件数を着手時点から採取し、指定の差を確認する。
- 性能測定は Release の qbert/IQN/Munchausen/BF16/prefetch/skip 無効。捨て Run のあと、保存した実行体→変更後を 3 巡以上。learn step 64〜256 の実時間差/192 を測り、変更後の中央値が基準の最大値以下であることを確認する。他の学習・ビルドとは重ねない。
- 詳細コマンドと結果は検証資材の README と本書の経過へ追記する。未実施は成功と扱わない。

## 前提・決めたこと

- 開始時の未決事項監査ではユーザー判断が必要なブロッカーは 0 だった。ステージ 0 で見つかった A1 の矛盾は、ユーザーが「受入を修正して続ける」を選び解決した。IQN の全群で 7 タグの定義を確かめ、PER 無効群では 01〜03 の実点と、少なくとも 1 つの有限値を確認する。04〜07 の実点は求めない。D1〜D19 と学習・メトリクスの振る舞いは維持する。
- `CONTEXT.md` の Bellmanターゲット・target return・Munchausen項・soft価値ブートストラップの意味を保つ。公開ヘッダ・設定キー・メトリクス・checkpoint は変えない。
- 乱数消費、forward 順、AMP の境界、優先度の準備→最適化→反映の順を保つ。値の無い診断や指標は未定義 Tensor のまま渡す。
- PRD121・924・957、replay fit の進行の共通化、TD clip 修正は範囲外。
- FP32 の基準が決定的にならない、または共通化で値を一致させられない場合は、原因を記録して D13・D18 に従い停止する。BF16 の代替は D14 の範囲で行う。

## 経過

- 2026-11-12: PRD、現行の更新 3 関数、用語、ADR0035、既存の比較資材、課題を照合した。PRD 自体には暫定の印はなく、PRD121 等への言及にある暫定と区別した。比較資材を流用してステージ 0 から進める。
- 2026-11-12 ステージ 0: 課題を対応中へ変更し、`reports/tools/wiki_support.py maps` は INDEX と地図321を更新して成功。変更前の Debug ビルドは成功（no work）、Release ビルドも成功（48 ターゲット）。MSVC 初期化経由で実行し、ログは `.scratch/prd125/build-baseline-{debug,release}.log`。
- 2026-11-12 比較資材の最初の RED: `.venv/Scripts/python.exe core/anet-core/testdata/prd125/run_validation.py snapshot` は、入口がまだ無いため終了値 2。入口と、保存・LunarLander の隔離実行・既存正規化を用いる比較を追加した。実行体と DLL は実ファイルをコピーし、既存出力先の上書きを拒否する。
- 2026-11-12 最初の GREEN: `repeat --label tracer --case default_LunarLander` は成功。2 Run の 24,399 点が完全一致し、正規化 SHA-256 は `941fa28b97b6a5833504c7a7b314bbc9966e77d6f66ff5edb8e4487e2a5a5a6c`。保存 manifest は `.scratch/prd125/baseline/manifest.json`。Run と採取のログは `.scratch/prd125/tracer.log`。
- 2026-11-12 A3・A5 の着手時点の件数は PRD の全行と一致した。CPU の独立条件は最大 4 組を並行し、CUDA 条件と性能測定は逐次実行する。性能測定時には他の学習・ビルドを重ねない。
- 2026-11-12 変種の入口の RED: `prepare --case none_base` は未実装で終了値 2。31 組と BF16 の入力生成・dump と診断点の検査を追加し、prepare は成功。最初の TD Run は、net の部分選択だけでは親の IQN 選択から来た tau_embedding を除けず構築失敗。§5.4 が許す変種設定の調整として Agent 全体の選択も QR 配線へ切り替えた。初回入力とログは保持し、修正入力は `inputs-r2` へ凍結する。DQN 本体はまだ変更していない。
- 2026-11-12 変種の最初の GREEN: `repeat --label td-probe-r2 --case none_base` は成功。2 Run の 20,040 点が完全一致（SHA-256 `ec7a05f7b5e6ee63bc58057c3361505a661cbdc9460ea5d48e0991b5e71219c6`）。dump の指定値と、replay fit の有限値の点の検査も成功。`repeat --label baseline --workers 4` で FP32 31 組と BF16 4 組の採取を開始した。結果は `baseline-summary.json` と `baseline-repeat.log`、既定 suite は `baseline-suite/` へ保存する。
- 2026-11-12 保存した Debug の既定 core suite は終了値 0、失敗ケース 0（834.44 秒）。実行体 SHA-256 は `c860b9c36d7442cf258dea624967ca6a597de8b1a5dde4839278b8048c76cd13`。JUnit は `.scratch/prd125/baseline-suite/core.xml`、結果は同 `suite.json`、標準出力は `process/process.log`。
- 2026-11-12 比較資材の保持・実行条件: 完了済み `td-probe-r2` の label 再利用は拒否され、summary の SHA-256 が変わらないことを確認した。並行 Run が動く間の `performance --label guard-probe` は、測定開始前に拒否した（`performance-guard.log`）。性能測定のときは別 Runner も含めた停止を確認する。
- 2026-11-12 保存した core suite の JUnit は 1,194 ケース。DQN 本体・既存テスト 2 ファイルは、この時点でも基準コミットから差分なし。別の学習として Release の `AnetRLRunner_ab.exe`（PID 36784、workspace `AtariRnd-01`、Montezuma 30M、`mz_tg_r123_b001_s1`）が稼働している。性能測定の直前にはこの学習との重なりを解消する必要がある。停止・一時停止の許可はまだ求めておらず、操作していない。
- 2026-11-12 設計文書の追随では責務の行に加え、実装品質確認観点51が求める新規検証資材への参照をテスト章へ追記する。既存のクラス図と他の責務の行は変更しない。
- 2026-11-12 停止: PRD A1 の IQN 7 タグの実点の条件は、PER 無効群では現行契約のまま満たせない（課題 ab98b7）。`iqn_no_per` は正常終了し 21,715 点を保存したが、01〜03 と 08 だけが出力された。04〜07 は dqn_based_agent.cpp の 5244 行付近の `initial_mask` が全偽のため NaN になり、observers.cpp の 1119〜1120 行で出力から除かれる。定義 JSON の data には 7 タグすべてがある。最新確認時点で 22 組の判定中 21 組が全点一致、1 組がこの検査で停止。DQN 本体は未変更で、ステージ 1・2 は未着手。独立したほかの条件の採取だけは起動済みバッチが継続し、結果を `baseline-summary.json` に保存する。完了報告ではない。

- 2026-11-12 再開: ユーザーが A1 の修正と続行を選択した。PRD の改訂行と A1、比較資材の診断検査へ反映した。元の採取は 35 組中 34 組が全点一致し、IQN+PER 無効群だけが旧検査で停止している。元の結果を保持し、この 1 組の再採取と全記録の再検査から基準を確定する。PRD の変更は独立レビューへ出す。
- 2026-11-12 PRD 改訂レビューは 1 往復で OK、不足・範囲外の指摘・確認事項なし。原文は `.scratch/prd125/review-20261112-1800-prd.md`。A1 の条件分岐が現行コードと一致し、全 scalar 比較とも矛盾しないと確認された。
- 2026-11-12 ステージ 0 完了: 再採取した IQN+PER 無効群は 21,715 点が完全一致（SHA-256 `c339f2084b61a96ea4ef69f0c29ed6cb7bb6833bab146457819efc25e9a4b89d`）。承認済み A1 で保存済み全 70 Run を再検査し、FP32 31 組と BF16 4 組がすべて完全一致。LunarLander の BF16 は 3 分布とも CPU で一致し、Atari は CUDA で一致した。除外した BF16 は無い。元の失敗を含む `baseline-summary.json` は保持し、当該 1 組だけ再採取へ差し替えた新しい正本を `baseline-accepted-summary.json` とした。出典・ハッシュは `baseline-accepted-provenance.json`、再検査の結果は `accepted-baseline.log`。以後は `--reference baseline-accepted` を使う。
- 2026-11-12 ステージ 1: 3 つの進行を基底へ移し、`ForwardCurrent` と `MakeUpdateDiagnostics` に分布ごとの計算を分けた。QR/IQN の forward・gather・指標は非仮想の `ForwardCurrentQuantiles` に共有し、IQN はその手前で tau を生成する。未定義の診断・上側の広がり・Q の広がりは未定義のまま渡す。Debug ビルドを開始した。
- 2026-11-12 ステージ 1 の Debug ビルド成功（`build-stage1-debug.log`）。A3 の全 11 行の件数と、ヘッダの `UpdateFromSamples` が 1 行であることを検査して成功。`compare --label stage1 --reference baseline-accepted --workers 4` と `suite --label stage1-suite` を開始した。Release は別 preset でビルド中。設計文書 200 の日英の責務と検証資材の参照を追随した。編集前の JP の blob が EN の翻訳元に一致することを確認し、変更した 2 か所の対応とリンクを確認して翻訳元 blob を更新した。
- 2026-11-12 ステージ 1 の FP32 31 組・BF16 4 組がすべて全点一致（`stage1-summary.json`、`stage1-compare.log`）。既定 core suite は終了値 0、失敗 0（654.52 秒、`stage1-suite/suite.json`）。Release ビルドも成功（`build-stage1-release.log`）。LF と既存ファイルの BOM を確認し、既存 DQN テスト 2 ファイルは保存した写しとバイト単位で同一。
- 2026-11-12 性能測定の準備: computer-use のアプリとウィンドウ一覧を読み取りで確認したが、以前の Montezuma Run は対象に無く、プロセスの再確認でも稼働していないことが分かった。一時停止・再開は操作していない。比較・テスト・ビルドがすべて終わったあと、`performance --label stage1-perf` を開始した。
- 2026-11-12 ステージ 1 の初回性能測定（3 巡）は不合格。基準は 0.040651・0.040896・0.041047 秒/更新、変更後は 0.041339・0.041318・0.041229。変更後中央値 0.041318 が基準最大 0.041047 を 0.66% 超えた。元の結果は `stage1-perf-summary.json` に保持する。実装を照合し、Release では Debug 検査が無効であること、ProfileRange は NVTX/Tracy の範囲通知だけで GPU 同期を行わないこと、TensorDict のコピーは参照と辞書のコピーで Tensor データの複製・転送を行わないことを確認した。各巡の比の超過は 1.69%→1.03%→0.44% と縮小しており、実行体を変えず 6 巡の再測定で再現性を調べる。判定条件は A9 のまま。`--rounds` は開始前に巡数を固定し、3 未満を拒否する。
- 2026-11-12 ステージ 1 の性能再測定（6 巡、実行体・設定は初回と同じ）は A9 を満たした。基準中央値 0.041411、最大 0.042068 秒/更新、変更後中央値 0.041250（基準中央値比 0.9961）。結果と全 12 Run は `stage1-perf-r2-summary.json`。3 巡の初回失敗を削除せず、6 巡の全データで判定した。0.66% の超過は再測定では基準側の変動幅に収まり、学習値を変える修正は行っていない。A2・A3・A7・A8・A9 のステージ 1 の確認が済み、ステージ 2 へ進む。
- 2026-11-12 ステージ 1 の初回 3 巡と再測定 6 巡を合わせた全 9 巡でも A9 を満たす。基準中央値 0.04136458082939498、基準最大 0.0420677155489102、変更後中央値 0.04130208326387219 秒/更新。初回の失敗を含む全点で成立することも確かめた。

## ステージ 2 の TDD 記録

- `stage2-red-build.log`: `[prd125]` の最初の式のテストは、`MakeBellmanTarget` が無いためコンパイルに失敗。関数を足した `stage2-scalar-{build,test}.log` は成功（5 assertions）。n-step 数が 1 と 3 の 2 行を使い、終端行の bootstrap を除き、勾配が無いことを確かめた。以後も同じ 1 ケースへ組み合わせを足す。
- TBO を増やした `stage2-tbo-red-test.log` は 2 assertions が失敗し、h を加えた `stage2-tbo-green-test.log` は 10 assertions が成功。bonus を増やした `stage2-bonus-red-test.log` は、終端行を含む 4 assertions が失敗した。各 build も同じ接頭辞のログへ保存した。
- bonus を加えた `stage2-bonus-green-test.log` は 20 assertions が成功。分位点と終端の指定を増やした `stage2-quantile-red-test.log` は 8 組が shape の不一致で失敗し、サンプル軸を保って広げる修正後の `stage2-quantile-green-test.log` は 16 組・112 assertions が成功した。
- 16 組が成功したあと、通常と Munchausen の Q/分位点の 4 経路を共通の式へつないだ。Munchausen の評価の進行は出力キーと tau の有無を受ける共通関数へ移し、旧 2 関数を削除した。ADR0035 は冒頭の注記だけを追加した。Debug の全ビルド結果は `build-stage2-debug.log`。
- 全 Debug ビルドと、接続後の `[prd125]` が成功（`stage2-connected-test.log`）。基底の 2 つの補助関数を Learner の実装の末尾へ配置し、旧関数の return と mask の shape 検査も共通の式に残した。破壊検査の再実行用に `run_mutations.py` を追加した。最初の起動は共通実行部がコマンドの絶対パスを必要とし失敗、2 回目は cmd の引用符でビルド前に失敗した。どちらもソースを復元し、元のログを保持した。MSVC 初期化を CP932/CRLF の作業用 cmd から呼ぶ形に直し、`stage2-mutations-r3` で検査を開始した。
- `stage2-mutations-r3/summary.json`: h の欠落、終端マスクの欠落、終端時 bonus の欠落のすべてで `[prd125]` は終了値 42、同じテストケースが失敗した。各々の復元後の再ビルドと `[prd125]` は終了値 0、失敗なし。最後も正しい式へ復元済み。A6 の破壊・復元は 3 件とも成功した。
- `stage2-shape.log`: A3 の全行と A5 の全行が指定件数に一致。旧 `CalcTargetQuantiles`・`CalcMunchausenTargetQuantiles` は .cpp/.hpp とも 0 件。既存テスト 2 ファイルの元のバイト列を保持し、追加は `[prd125]` の 1 ケースだけ。ADR0035 の元の本文は同一、注記だけが増えた。変更対象の LF と既存 BOM も保持した。
- 最後の復元後の全 Debug・Release ビルドが成功（`build-stage2-final-{debug,release}.log`）。同じソースから作った Debug で `compare --label stage2 --reference baseline-accepted --workers 4` と `suite --label stage2-suite` を実行する。変更後の C++ の追加行に重なる識別子リテラルはなく、現用のコード・設定・設計文書・実装の地図に旧 target 関数の参照は無い。設計 200 の既存 Mermaid 図は日英とも同一で、EN の翻訳元 blob は JP の `9621359eb8dd0d0e2ebadc8e806feccebb086b48` と一致する。
- Stage 2 の既定 core suite は 1,195 ケース、終了値 0、失敗 0（654.31 秒）。実行体 SHA-256 は `f5754510aa12bfcecbc5a6d4d21ee24596c5acbcafdb424ba472a9aef83331d1`。JUnit と process log は `stage2-suite/`、結果は同 `suite.json`。基準の失敗ケース集合と一致した。
- `stage2-summary.json` と `stage2-compare.log`: FP32 31 組・BF16 4 組すべての全 scalar の点が基準と一致した。除外した組は無い。最後の BF16 Atari は 23,219 点、SHA-256 `326fe2e65e79755e4704b2ee661bafe507351d16d44feef9c3c6512e7debc9fc`。比較・core suite・両ビルドが終わったあと、ほかの Runner・テスト・ビルドが無いことをガードで確認し、`performance --label stage2-perf --rounds 6` を開始した。巡数は開始前に 6 と固定した。
- `stage2-perf-summary.json` と `stage2-perf.log`: 6 巡の全 12 Run と捨て Run が完了し、A9 に合格。基準中央値 `0.04569270531646907`、最大 `0.04637500096578151`、変更後中央値 `0.04567187497741543` 秒/更新。変更後中央値が基準最大以下である。Stage 1/2 とも、受入を満たす範囲で性能を確認できた。ゼロの overhead や他の構成での速さを証明する結果とはしない。

## 自己点検

`docs/quality/impl.md` の全項目を適用範囲で走査した。必要な検証はすべて完了し、未実施の検証と残る未決は無い。課題の対応内容へ結果を追記してからレビューへ渡す。

| 観点 | 確認したこと |
| --- | --- |
| 検索・差分・文字コード | 対象の差分を特定。hpp の using namespace、production の dynamic_cast と TODO、共有契約の固有語、追加した C++ の重なる識別子リテラルは 0 件。LF/BOM と既存テスト・ADR 本文を保持。既存 Mermaid 図を保持。旧 target 関数は現用範囲に無い |
| 実装計画・PRD | 未決と前提、確認する面、縦に進める TDD と既存挙動の比較、ステージの順を記録。A1 はユーザー判断で改訂し別レビュー済み。A2〜A9 は各ステージで必要な結果が揃った。A10・A11 の文書追随を確認 |
| 契約・失敗・名前 | 公開ヘッダ・設定・メトリクス・checkpoint を維持。内部の旧関数を削除し、alias は無い。分布固有と共通の shape/NaN 検査を保持。呼ぶ側で real-space 変換を行い、bonus の有無で足す形を分ける。用語は CONTEXT の既存語に対応 |
| 設定 | 新しい設定キーは無い。凍結入力はプロファイル境界を `:` で区切り、M1 を保って必要な診断群を追加。dump と実点を全組で検査。並び順の番号や現用の設定・bat は変更していない |
| C++・所有権 | 既存の機能グループに内部の結果型と差し替えの口を置き、集成体は指示付き初期化。新しい永続 State・Resource・借用メンバ、公開ヘッダ、namespace、include、UI ロックは無い。Policy への依存方向を維持 |
| コメント・計測・ログ | 処理の順と bonus・実空間・tau・PER の意図コメントを移動先へ保持。主な段を計測し、重みの DEBUG ログは PER 有効時だけ出す。GPU 同期・転送・NN の forward を増やしていない。GetScalar の契約と NaN の扱いは維持。WARN/INFO は追加していない |
| テスト・検証・文書 | 手計算 16 組と 3 種の破壊・復元、既定 core suite、同 seed 全点比較、Debug/Release のビルドを記録。Python 2 ファイルの構文を確認。検証資材を testdata/prd125 に置き設計 200 のテスト章から参照。日英の変更箇所・リンク・翻訳元を照合。skill の変更は無い |

## レビュー記録

- 1 往復目: 実装計画は OK、実装は追加テストのバッチ件数の定数化 1 件で差し戻し。原文は `.scratch/prd125/review-20261112-1945-impl.md`。同じ件数を設定・環境指定・行ループで使うため、テスト内の `kBatchSize` を共有する形に直した。学習コード・値・期待値の変更は無い。Debug・Release の全ビルドと `[prd125]` を再確認する。ほかの不足、範囲外の検出、開発者への確認事項は無い。
- 定数化後の Debug・Release の全ビルドと `[prd125]` は、両方で 16 組・112 assertions が成功（`review-fix-{debug,release}-{build,test}.log`）。全 35 組の比較と性能測定に使った Debug・Release の Runner の SHA-256 は修正後も同一で、学習コードと実行体は変わっていない。テストだけの同値の定数化なので、既定 core suite の 1,195 ケース・全点比較・性能は先の記録を採り、追加したテストだけを再実行した。`review-fix-shape.log` も成功。
- 2 往復目: 実装計画・実装ともに OK。原文は `.scratch/prd125/review-20261112-1949-impl.md`。バッチ件数の定数化は充足し、thread 件数は別の役割の単発値と確認された。未変更の比較・性能・既定 suite は前回の確認を引き継ぎ、定数化後は両ビルドと追加テストの成功を確認した。残る不足・範囲外の検出・開発者への確認事項は無い。根拠が無く見送った指摘も無い。
- 完了時の課題更新: ab98b7 の AI 判定を `残件無し`、判定日を `2026-11-12` とした。状態は `対応中` を保持。`reports/tools/wiki_support.py maps` は INDEX と地図321を更新して成功し、解決できない PRD の参照は 0 件（`final-maps.log`）。

## 決まらなかったこと

なし。A1 の扱いは上のユーザー判断で解決した。

## 完了報告

- 実施内容: PRD125 の全 3 ステージを完了。承認された A1 の修正、更新の進行・Munchausen の評価・Bellman target の式の共通化、検証資材とテスト、文書の追随を行った。
- 変更ファイル: `dqn_based_agent.cpp/.hpp`、既存の `dqn_munchausen_test.cpp` への追加、`testdata/prd125` の README と Python 2 本、設計200の日英、ADR0035の冒頭注記、PRD A1、本実装記録、課題 ab98b7 と派生一覧。無関係な作業の変更は保持した。
- レビュー結果: PRD 改訂は 1 往復で OK、実装計画と実装は 2 往復目でともに OK。テストの定数化 1 件を修正し、両ビルドと該当テストを再確認。比較・性能・既定 suite の実行体は同一で、先の結果を引き継いだ。既定 suite はステージ 2 の 1,195 ケースが成功し、両ステージの 35 組が全点一致。性能も両ステージで受入内。最初の性能不合格を含む記録を保持した。未実施の検証は無い。
- 決まらなかったこと: なし。新規・既存への追記・この場で直す・見送り推奨の起票候補もすべてなし。
- 判断してほしいこと: なし。
- 次にできること: 本工程の追加作業は無い。ステージ・commit・push は実施していない。
