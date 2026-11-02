# PRD097 Claudeレビュー指摘対応 実装メモ

## 概要

ユーザーが承認した①〜④・⑥を実施する。⑤は対象外。性能再測定・追加プロファイル採取は行わない。20impl・21implと既存の無関係な変更を保持し、commit/pushしない。

## 主な変更

- DropMergeのResultはdropper・fruits・grid・world情報・報酬・step・score・last_*・action_modeをnative値とTensorOptionsで所有する。Env・body・設定を保持せず、GetAuxDataの要求時だけTensor化する。既存キー・shape/dtype/device・空fruitsのキー省略・果物順序を維持。GetScalarはnative値から既存singleton（1要素gridを含む）を返し、StepResultのfallbackも移す。統計式・演算順序は維持し、不要なEnv::CreateAuxDataを撤去する。capture/変換のProfileRangeと日本語段階コメントを付ける。
- PRD外の持ち越し: 最終終了時のRunnerApp join/Shutdownと最後の学習futureの失敗再送出、EnvTaskBatchのworker例外伝播を維持する。RunnerAppの古い段コメントを現在の目的へ修正し、レビュー対象へ追加する。通常Stop/再開は変えない。
- READMEのM2履歴の見出し階層を下げ、不採用履歴と廃止スクリプトの境界を明記する。履歴数値は保持。GridMaze Resetのコメントを終端フラグ/開始状態の初期化へ修正する。
- BatchStepResultにonce_flagを置き、GetEpisodeEndMaskの遅延初期化をcall_onceで保護する。公開関数/返値と遅延CPU転送を維持し、生成時の転送や⑤のreward/done/truncatedキャッシュは追加しない。

## テスト

Public surfaceはDropMerge Reset/Stepと結果GetAuxData/GetScalar、BatchStepResult::GetEpisodeEndMask、Runner通知/Shutdown。
内部表現と同期のrefactorなので、まず公開契約テストのGREENを確認してから変更し、変更後に回帰する。未再現競合をREDと扱わない。

1. 既存Drop回帰を実行し、全auxキーの値/shape/dtype/device、native scalar対応、空fruits・1要素grid・View履歴、初回aux読出し前の後続Step/Reset/Env破棄に対する保持を1件ずつ補強しGREEN→refactor→GREENを確認する。
2. 終端maskは終端なし/混在/全終端をCPUと利用可能CUDAで反復し、複数threadの初回同時読出し・反復読出しを検証する。既存の逐次試験は改修前にGREENを確認し、並行試験は同期保護後に実行する。
3. Shutdown例外・worker例外・Runner通知の回帰、Debug全体ビルドと隠し以外全テスト、Releaseビルドを行う。VsDevCmd経由、Pythonは.venvを使う。
4. 新しいafter-d7-v3へ実行体/設定/hash/Runを分離する。既存beforeを再監査し全7構成のscalar/trace・実効設定/成果物監査を行う。性能反復とNsightは行わない。

## 検証・前提

実行中Runnerがないことを確認済み。新しい公開API/設定は追加しない。Gridのscalar singletonも既存契約を保持する。公開値は不変のまま、内部maskキャッシュの一度だけの初期化を同期する。
修正後の速度は未測定と記録し、v2性能値を今回の効果とは扱わない。既存999性能再測定課題は未解決のまま残す。
自己点検後、同じan-reviewにRunnerAppを含む全対象と最終検証を渡して修正/再レビューし、結果と残る未決を本書へ記録して停止する。

## 経過

- 承認計画をコード変更前に保存した。元レビューは.scratch/prd097/claude-impl-review-20261020.md。①〜④・⑥に対応、⑤と性能再測定はユーザー指定で対象外。

- 変更前既存Drop28tests/378assertions成功。追加aux試験のCPU部分は成功したが、直接CUDA EnvのResetは既存makeStateがCUDAのdata_ptrをCPUから書くためSIGSEGVとなった（drop-contract-before.log）。今回の通常CPU Env回帰へ絞り、既存の直接CUDA Env不具合は範囲外の候補へ持ち越す。maskのCUDA回帰は予定どおり行う。

- 追加公開契約は変更前にDrop CPU431assertions、mask CPU/CUDA288assertions成功。native captureと要求時Tensor化、singleton scalar fallback、call_onceを実装。並行mask試験は保護後に追加した。RunnerApp/EnvTaskBatchの例外伝播は保持し、README履歴階層とGridコメントも修正。Debug全体ビルド中。

- 修正後Drop全29tests/1081assertions、mask CPU/CUDA全2tests/384assertionsが成功。once_flagによる非コピー化で既存fixture1件がビルド失敗したため、一時結果をconst参照で受けて公開値から新しい結果を構築するよう移行し、Debug全体再ビルドが成功。before14Run/241ファイルの再監査も成功（audit-22-before.json）。現在Debug全体CTestとRelease全体ビルドを実施している。

- Debug全体CTest6/6が成功（378.07秒）、Release全体ビルドも成功。after-d7-v3へ新exe c8ebf740e731eea75e1f0437ccc4653e3b28236ce9077e2e3746985a28c6074dとDLL/設定・ソース31ファイルのhashを固定し、7構成の数値受入を開始。自動記録される時間/資源タグは系列比較から除外し、速度差の受入判断には使わない。性能反復とNsightは起動していない。

- 修正コードと計画の独立レビューはともにOK、修正要求0件（原文 .scratch/prd097/review-20261020-2317-22impl-code.md）。残るImageCls/serialの数値比較と全7構成の最終監査を完了してから、同じレビュー先へ結果を追加確認する。

## 最終検証結果

①はnative値所有と要求時Tensor化、②は最終終了時とworkerの例外伝播の保持・レビュー範囲明記、③はM2履歴境界、④はGridMaze Resetコメント、⑥はcall_onceによるmask初期化保護を完了した。⑤のreward/done/truncatedキャッシュは追加していない。統計式・浮動小数点演算順序はaux fallback以外の元ソース一致でも点検した。

- Debug全体ビルド・Release全体ビルドが成功。いずれもVsDevCmd経由。
- Debugの隠し以外の全テストはCTest6/6成功（378.07秒）。Core714件中713成功・条件付きskip1、Lunar7、Drop29、Grid1、Image17、Atari39件。Shutdown/worker例外伝播・Runner通知の回帰を含む。Atari必須Asterix受入も成功。
- Drop公開契約は改修前CPU431 assertionsのGREENを確認し、改修後全29件1081 assertionsが成功。全aux・singleton scalar・空fruits・1要素grid・View用履歴と、初回aux読出し前にStep/Reset/Env破棄した保持結果を確認。
- maskは改修前CPU/CUDA288 assertions、改修後2件384 assertions。終端なし/混在/全終端、16thread同時初回読出しと反復読出しをCPU/CUDAで確認した。未再現競合をREDとは扱っていない。

| 構成 | 比較点数 | 結果 |
|---|---:|---|
| atari1 | 93027 | 一致 |
| atari2 | 90028 | 一致 |
| dropmerge | 161610 | 一致 |
| lunarlander | 63648 | 一致 |
| gridmaze | 90851 | 一致 |
| imagecls | 67160 | 一致 |
| dropmerge_serial | 162210 | 一致 |

全7構成728534点のscalar/traceが規定の比較条件で一致。beforeは14Run/241ファイル、after-d7-v3は7Run/158ファイルの実効設定・成果物・DLL/実行体hash監査に成功した。全7構成のconfig_data全キー・値も監査し、差分は出力先と意図したObserver DSL移行だけ。凍結ソース31ファイルと現在ソースも全一致。実行体SHA-256はc8ebf740e731eea75e1f0437ccc4653e3b28236ce9077e2e3746985a28c6074d。

証跡は`.scratch/prd097/claude-fixes-20261020/`の`debug-build-retry.log`、`release-build.log`、`debug-ctest.log`、`ctest-summary.json`、`drop-after.log`、`mask-after.log`、`v3-equivalence.log`、`after-audit.log`、`resolved-config-audit.json`。Run/比較/監査は`implementation-20261020/protocol-fixed/after-d7-v3/`。Python補助は全てリポジトリ.venvで実行した。

## 残る事項と省略

- ユーザー指定により性能比較の反復Run・Nsight採取は実施していない。修正後の速度は未測定。v2の速度を今回の効果には使わない。既存の`999_prd097_performance_remeasurement_10prd.md`は未解決のまま、native値所有の実装済み状況だけ追記した。
- ⑤は承認された対象外。既存Observerのdynamic_cast課題も範囲外で保持した。
- 改修前に直接CUDAのDrop EnvでSIGSEGVを確認した。既存makeStateがCUDA bufferをCPUから書く不具合であり、今回の通常CPU Env検証と分離した。AGENTS.mdの未解決事項記録規約に従い、当該箇所へDoxygen TODOを残した。maskのCUDA検証は完了している。
- 今回の①〜④・⑥に必要な実装・検証の未実施項目はない。性能課題と直接CUDA Env不具合の解消は主張しない。commit/pushなし。

## 最終レビューと完了

同じan-review（review_prd097_d7）へ全53ファイル（RunnerAppを含む）と最終検証を再提出し、計画・実装ともOK、修正要求0件、確認事項なし。原文は`.scratch/prd097/review-20261020-2326-22impl-final.md`。コードレビュー後の変更はREADMEと本書への検証結果追記のみで、凍結ソース31/31の一致を確認した。

承認された①〜④・⑥の対応、必要な検証、自己点検、独立レビューを完了した。追加の起票候補はなし。既存の性能再測定課題と範囲外TODOは上記のとおり保持する。20impl・21implと無関係な変更を保持し、commit/pushせず工程を終了する。
