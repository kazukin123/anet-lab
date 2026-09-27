# PRD083：ActionContext 廃止の実装計画

## 概要

Actor が RNG と frame stacker を直接所有する等価リファクタを行う。演算順、乱数列、device 転送、aux、snapshot 同期の挙動を維持する。

本メモをコード変更前に保存し、実装の正本とする。

## 実装変更

- `dqn::Actor` は `RandomHolder` を継承し、旧 context 引数を `unique_ptr<FrameStacker>`、必須 device、`optional<seed_t>` に置換する。`RandomHolder(seed)` を初期化し、`MakeAction() const` から `rnd_` を使用する。
- 観測加工は Actor 内で stack → device 転送 → 正規化の順を維持する。stacker 有りでは既存 `Stack()` に委譲し、無しでは `state.obs.To(device_)` を呼ぶ。
- DefaultDQN は `CreateFrameStacker(request)` で専有 stacker を生成する。Rainbow は nullptr と request の device・seed を渡す。
- ActionContext 群、関連実装・TODO を削除する。内部 Actor ヘッダは `stacker.hpp` を include、公開 DefaultDQN ヘッダは `FrameStacker` を前方宣言し、stacker 側の `agent.hpp` 依存を外す。
- Actor 直接構築の既存テスト8箇所を移行する。PRD指定の日本語設計文書、CONTEXT注記、設定コメントを同期する。

## テストと検証

- 変更前に既存関連テストを実行し、ユーザーによる等価 Run の baseline 採取を完了する。
- 新規テストケースは作らず、既存 DefaultDQN テストに `raw_obs`・`norm_obs` の値と shape、既存 Rainbow テストに `raw_obs` と `norm_obs` 不在の assertion を追加する。旧実装で成立を確認してからリファクタする。
- 構造変更後に Actor・snapshot・Q hint・Munchausen・native stack・stacker の既存テストを実行する。挙動追加ではないため、人工的な RED は作らず、既存挙動を保持する回帰検証とする。
- MSVC 初期化経由で Debug ビルドを行い、`anet-core-test` 全件成功、`git diff --check` 空を要求する。
- Git管理下の `core/`・`apps/` の現用ソース・テスト・設定から、PRD指定の廃止識別子が消えたことを確認する。生成物と Run artifact は検索対象から除外する。

## 等価性の受入

- ユーザー実施で DefaultDQN／LunarLander／stack4 の改修前後 Run を比較する。seed・解決済みconfig・終了step・実行環境を固定する。
- ユーザー指定により Rainbow は改修前後 Run の等価性検証から除外する。コード移行と既存テストによる回帰確認は維持する。
- loss、q_max、評価 episode return、action 系の主要タグを baseline で確定し、metrics マスタから全点の `(tag, step, value)` 系列を抽出して checksum を比較する。欠損・空系列・点数差を成功扱いしない。
- `inspect_run.py` は確認表示に使用する。最大128点に間引かれる `--series` と checkpoint の raw checksum は合否判定に使わない。
- baseline 未採取ならコード変更を開始しない。Run比較が未完了なら、等価性受入も未完了として報告する。

## 前提と非対象

新クラス、新ファイル対、新ADR、設定値変更は追加しない。既存の計測範囲を保持する。normalizer の潜在競合、FrameStacker の Reset 削除、観測加工の一般化、英訳・履歴資料は対象外とする。無関係な未コミット変更を保持する。

## 着手時の確認（2026-09-27）

- 計画保存と内容・SHA-256・更新時刻の確認を完了。
- 既存 Debug 実行体で `[stacker],[dqn]` を実行し、170 test cases / 15,819 assertions が成功（exit 0）。今回の再ビルドは未実施。
- `git diff --check` は成功。
- 改修前の等価 Run 2本の採取状況・Run名は未確認。承認済みの baseline ゲートに従い、コード・テスト・設定・現行設計文書の変更は未着手。
- ユーザーから baseline の Run名を受け取った後、採取条件を照合して実装を再開する。

## baseline 確定と検証範囲変更（2026-09-27）

- ユーザーが `run_20260927-010102_ll_iqn` を改修前 baseline として指定。前節の確認待ちは解消。
- DefaultDQNAgent、LunarLander、IQN、run.seed=1、stack_count=4、num_envs=256。deterministic_algorithms=true、cudnn_deterministic=true、cudnn_benchmark=false。
- 実際の停止条件は online の app.exp_pause_step=2,000,000、app.exp_exit_step=-1。自動pause後にユーザーが閉じ、checkpoint保存を完了。学習scalarの最終exp_step=1,999,616。app.batchrun.exp_exit_step=2,000,000は今回の実行モードでは使われていない。
- config/config_data.txt SHA-256: acebc770ac8c273242d2e355ebe35187ab41412108e05cacd3b335b5ee0de035。
- 実際の評価タグは `21_eval/01_target_reward` と `21_eval/02_policy_reward`。PRDの61/62という例示ではなく、このRunの実タグで照合する。
- Rainbow の等価Run比較はユーザー判断で除外。既存 aux assertion とコード移行は実施する。

## baseline 全点 checksum

各タグの出現順を維持し、JSON配列 `[tag, step, value]` を `ensure_ascii=False, separators=(',', ':'), allow_nan=False` でUTF-8化し、各行末にLFを付けてSHA-256を計算する。時刻・他タグとの交錯順は含めない。非数値の記録値も省略しない。

| tag | count | first step | last step | SHA-256 |
|---|---:|---:|---:|---|
| 38_agent_loss/01_loss | 7779 | 8448 | 1999616 | 06a5aa0df44bb5cfc00577c8ffc38a85279ed03435dbd56695cf1d7c610b7238 |
| 38_agent_loss/02_loss_ema | 7779 | 8448 | 1999616 | 3ddeb229f6b44f9de6c50eca8bd47501171446dfe0aead69f7f31cc06806cb64 |
| 37_agent_qtd/11_q_max_mean | 7779 | 8448 | 1999616 | 12e7561961ce60e4966bffd5d7a25344e59fcab0ebfcecd087ec5012af975b55 |
| 37_agent_qtd/12_q_max_max | 7779 | 8448 | 1999616 | 2b77036ff1674618b8a2b9e8a0ec1a9d2dd9354531f77602a8602aeaf01fd536 |
| 37_agent_qtd/13_q_max_std | 7779 | 8448 | 1999616 | 0dba4280108059735294bb9c4b4d2c6f11491869e565fdcabfa551fa574674b3 |
| 21_eval/01_target_reward | 312 | 8448 | 1998848 | dc91e26361039e68dd44cfc5932724154c2079f6c7352c826c8cefd744566ced |
| 21_eval/02_policy_reward | 312 | 8448 | 1998848 | 70f7addb21891be52fe891a51d1e7e5981e159846dff17c07135d3c103831cb4 |
| 21_eval/03_target_reward_ema | 312 | 8448 | 1998848 | c0b3c9860379ec7a70e8f5fa3da337ddc9e17ff48712346252609ce6caa58fb0 |
| 21_eval/04_policy_reward_ema | 312 | 8448 | 1998848 | 0a4fcd15f966f0f4852c5a779fc077164ce9c5aa545b0dd8badb759202c69191 |
| 33_agent_action/21_iqn_policy_margin_mc_ratio_ema | 7811 | 256 | 1999616 | dfe5323c581c09e999e1172991890b6392cc42e0cc0b03bb54177eed707b8e92 |
| 35_agent_churn/01_action_churn_ratio | 124 | 0 | 61869 | 2a64c0f6dce529ca2bf6288b263b249ab9df799f3e11ceeb3da81b1df1ca1c62 |
| 35_agent_churn/02_action_churn_ratio_ema | 124 | 0 | 61869 | 941952a8bd0dec2056bdcb06a7311380386fbb1f9be4b65729f36d767c0f2d25 |


## 実装時の確認

- 改修前に既存DefaultDQN native-stackテストをauxの値・shape assertionで補強し、11 assertions / 1 test case成功。
- 既存Rainbow snapshot diagnosticsテストをraw_obs・norm_obs不在のassertionで補強し、8 assertions / 1 test case成功。
- Actorの直接構築8箇所を移行。stacker_test.cppとFrameStacker::Resetは無変更。
- Debug全体ビルドの初回は並列PCH読み込みのメモリ不足（C3859/C1076、Windows code 1455）で失敗。並列数1で再実行する。
- baseline比較用の停止条件は上記online pauseを正本にする。改修後も同じ解決済みconfigを用い、Run名など実行出力先だけ区別する。

## Actor改修後・thread不具合修正前の検証結果

- MSVC `x64-Debug --parallel 1` 全ターゲットビルド成功（exit 0）。初回の並列メモリ不足は解消。
- MSVC `x64-Release --target AnetRLRunner --parallel 1` 成功（exit 0）。通常launcherが参照するRelease Runnerを更新済み。
- `[actor],[native_stack],[stacker],[actor_munchausen]`: 28 test cases / 338 assertions 全成功（exit 0）。
- 全件テストは `PinnedThreadPool stops while workers return to waiting` でSIGSEGV。614 cases中612 passed / 1 failed / 1 skipped、22,246 assertions中22,245 passed / 1 failed。Windows exit -1073741819（0xC0000005）。
- 固定seed=83083で同テストを単体再実行して再現。workers=4、iteration=875、1,876 assertions中1,875 passed / 1 failed、同じexit。全件実行時はiteration=579。
- thread.cpp / thread.hpp / thread_test.cppは今回無変更。この単体テストはActorを使用しない。この時点では原因未確定、全件成功の受入条件は未達。その後、ユーザー指示により診断・修正を追加した（後述）。
- 他の回帰確認として、上記1テストだけを除外したスイートをseed=83083で実行。617 test cases中616 passed / 1 skipped、22,183 assertions全成功（exit 0）。全件成功の代替とは扱わない。
- UTF-8 / LFを変更17ファイルで確認。既存BOMは維持。agent.txtの非コメント代入行はHEADと一致。
- 現用のcore/appsから廃止識別子を検索し0件。stacker_test.cppは無変更。FrameStacker::Resetは既存dead codeとして残す。
- 改修後Runの全点checksum比較は未実施。ユーザー採取後のRun名を受け取ってbaselineと比較する。RainbowのRun比較はユーザー指定により対象外。

## 現在の到達点

実装・現行ドキュメント同期・thread pool SIGSEGVの原因修正・Debug全体ビルド・Release Runner更新・除外なし全件テストは完了。DefaultDQNの改修後Runとの全点checksum比較も一致し、合意済みの実装・検証・等価性受入は完了。未コミットのため、docs/memo/README.mdの規約に従いPRDと実装メモは直下に保持する。

2026-09-27、ユーザー指示によりthread pool不具合の診断・修正を今回の範囲へ追加した。再現済みの既存テストを回帰検証の入口とし、原因を特定して局所修正する。修正後は除外なしの全件テストとRelease Runnerビルドを再実行する。DefaultDQNのユーザー実施Run比較は下記の最終受入結果で完了した。

### thread不具合修正前に実行した検証

- MSVC初期化後の `cmake --build --preset x64-Debug --parallel 1`（成功）。
- MSVC初期化後の `cmake --build --preset x64-Release --target AnetRLRunner --parallel 1`（成功）。
- `anet-core-test.exe "[actor],[native_stack],[stacker],[actor_munchausen]"`（成功）。
- `anet-core-test.exe`（SIGSEGV、全件成功は未達）。
- `anet-core-test.exe "PinnedThreadPool stops while workers return to waiting" --rng-seed 83083`（同じSIGSEGVを単体再現）。
- `anet-core-test.exe "~PinnedThreadPool stops while workers return to waiting" --rng-seed 83083`（残り成功、1 skipped）。
- `git diff --check`（成功）、現用core/appsの廃止識別子検索（0件）。

ユーザーのLunarLander.txt、12_batch_run.bat、gui.cpp、実験記録、その他未追跡ファイルの変更は保持した。コミット・pushは行わない。

## 追加範囲：thread pool SIGSEGVの原因と修正

- 通常実行の全件テストへ一時的な例外ダンプprobeを入れ、2回再現。debugger接続時はログを抑えても再現しなかった。probeは原因確認後に削除。
- クラッシュ元はWorkerLoop起動時のProfileThreadName → Tracy SetThreadNameWithHint → memcpy。登録名shutdown-test_1は正しく、15 byteをコピーする先がnullだった。
- 最初のアクセス違反時のWindows error=1455、available commit=14,446,592 byte、total commit limit=219,821,625,344 byteを記録。allocator確保失敗後のnullコピーが直接原因。
- third_party/tracy/CMakeLists.txtはTracyをSTATICで組み込む一方、内蔵rpmallocのBUILD_DYNAMIC_LINK=1がWindows FLSのthread destructor登録を無効化していた。終了workerのheapが回収・再利用されず、短命threadを5000本作る既存停止テストで蓄積した。
- 外部依存の不具合修正として、このマクロを実際のstatic integrationに合わせて0へ変更。既存FLS destructorのrpmalloc_thread_finalize(1)に委譲する。計測、テストの1000反復×workers{1,4}、poolの停止処理は維持する。
- 新クラス・新設定・互換層は追加しない。既存shutdownテストを回帰検証に用い、除外なしの全件実行で確認する。

## 最終検証結果（thread不具合修正後）

- MSVC初期化経由のx64-Debug全ターゲット、並列数1：exit 0。
- MSVC初期化経由のx64-Release AnetRLRunner、並列数1：exit 0。Release executable更新日時2026-09-27 02:06:02、size=5,560,320 byte。
- 既存shutdownテストをseed=83083で3回実行：各2,000 assertions / 1 test case成功、exit 0。各実行は元の1000反復×workers{1,4}のまま（worker生成計5000本）。ピークprocess commitは993.8 / 990.3 / 990.3 MiB。
- 除外なしのanet-core-test全件、seed=83083：618 cases中617 passed / 1 skipped、24,183 assertions全成功、exit 0。既存skipは失敗扱い・成功件数への加算をしない。
- main_test.cppのダンプprobeは全て除去し、HEADとの差分0。thread.cpp / thread.hpp / thread_test.cppも差分0。外部依存の修正はtracy_rpmalloc.cppのstatic linkage設定1箇所と理由コメントだけ。
- 現用core/appsのActionContext / PushObservation / CreateActionContext参照は0件。
- DefaultDQNのbaseline全点checksumは前節へ固定済み。改修後Runとの比較は下記の最終受入結果で一致。RainbowのRun比較はユーザー指定により除外。

詳細ログ（ローカル一時資材、Git管理外）：

- `%TEMP%/prd083-final-debug-build.log`
- `%TEMP%/prd083-final-release-build.log`
- `%TEMP%/prd083-thread-fixed-1.log` ～ `prd083-thread-fixed-3.log`
- `%TEMP%/prd083-final-core-tests.log`

コードと文書の実装、全件テスト、ビルド、DefaultDQNの等価性受入は完了。コミット・pushは実施していない。

## DefaultDQN Run等価性の最終受入（2026-09-27）

- 改修前：`run_20260927-010102_ll_iqn`。改修後：ユーザー指定の`run_20260927-134909_ll_iqn`。同じLunarLander-01 workspace。
- 両Runのconfig/config_data.txtはbyte単位で一致（SHA-256 `acebc770ac8c273242d2e355ebe35187ab41412108e05cacd3b335b5ee0de035`）。seed=1、DefaultDQN/IQN、stack4、256 envs。
- backend/app/run/train-seed/env/DefaultDQNAgentのJSON dumpもdataを含め一致。deterministic_algorithms=true、cudnn_deterministic=true、cudnn_benchmark=false、torch_num_threads=1。ログ上もAgent/Eval device=cuda、BatchEnv device=cpuで一致。OS/GPU/driverの版番号はRun artifactに記録されていないため、版番号の同一性はartifactだけでは独立検証していない。
- onlineのexp_pause_step=2,000,000、exp_exit_step=-1が共通。両ログで自動pause → OnClose → agent_close保存完了を確認。主要学習scalarの最終exp_step=1,999,616が一致。
- metrics.jsonlマスタから全点を読み、baseline表と同じ正規化で各タグのSHA-256を計算。読み取り前後でsize/mtimeが変わらないことも確認。baselineを再計算し、保存済みchecksum表との一致も確認した。
- 12タグ・48,202点すべてでcount/first step/last step/checksum一致。欠損・空系列・点数差なし。loss、q_max、評価episode return、action系の合意済み主要タグすべてで等価性受入成功。
- 時刻・実行時間・Metricsファイルのraw bytesは判定対象外。両masterのサイズは36,167,836 / 36,168,533 byteで異なるが、下記の全点(tag,step,value)系列は一致。inspect_run.pyの間引きseries、cache、checkpoint raw checksumは合否に使用していない。
- RainbowのRun比較はユーザー指定により対象外。コード移行・既存テスト回帰は実施済み。

| tag | count（両Run） | first step | last step | 改修後 SHA-256（baselineと一致） |
|---|---:|---:|---:|---|
| 38_agent_loss/01_loss | 7779 | 8448 | 1999616 | 06a5aa0df44bb5cfc00577c8ffc38a85279ed03435dbd56695cf1d7c610b7238 |
| 38_agent_loss/02_loss_ema | 7779 | 8448 | 1999616 | 3ddeb229f6b44f9de6c50eca8bd47501171446dfe0aead69f7f31cc06806cb64 |
| 37_agent_qtd/11_q_max_mean | 7779 | 8448 | 1999616 | 12e7561961ce60e4966bffd5d7a25344e59fcab0ebfcecd087ec5012af975b55 |
| 37_agent_qtd/12_q_max_max | 7779 | 8448 | 1999616 | 2b77036ff1674618b8a2b9e8a0ec1a9d2dd9354531f77602a8602aeaf01fd536 |
| 37_agent_qtd/13_q_max_std | 7779 | 8448 | 1999616 | 0dba4280108059735294bb9c4b4d2c6f11491869e565fdcabfa551fa574674b3 |
| 21_eval/01_target_reward | 312 | 8448 | 1998848 | dc91e26361039e68dd44cfc5932724154c2079f6c7352c826c8cefd744566ced |
| 21_eval/02_policy_reward | 312 | 8448 | 1998848 | 70f7addb21891be52fe891a51d1e7e5981e159846dff17c07135d3c103831cb4 |
| 21_eval/03_target_reward_ema | 312 | 8448 | 1998848 | c0b3c9860379ec7a70e8f5fa3da337ddc9e17ff48712346252609ce6caa58fb0 |
| 21_eval/04_policy_reward_ema | 312 | 8448 | 1998848 | 0a4fcd15f966f0f4852c5a779fc077164ce9c5aa545b0dd8badb759202c69191 |
| 33_agent_action/21_iqn_policy_margin_mc_ratio_ema | 7811 | 256 | 1999616 | dfe5323c581c09e999e1172991890b6392cc42e0cc0b03bb54177eed707b8e92 |
| 35_agent_churn/01_action_churn_ratio | 124 | 0 | 61869 | 2a64c0f6dce529ca2bf6288b263b249ab9df799f3e11ceeb3da81b1df1ca1c62 |
| 35_agent_churn/02_action_churn_ratio_ema | 124 | 0 | 61869 | 941952a8bd0dec2056bdcb06a7311380386fbb1f9be4b65729f36d767c0f2d25 |

PRD083の合意済み受入は完了。レビュー・コミットは人間の工程とし、未コミットの番号セットはdocs/memo直下へ保持する。Run artifactは変更していない。
