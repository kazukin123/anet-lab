# PRD098 設定整理 実装メモ

## 概要

2026-10-24 のユーザー依頼 `.scratch/prd098/codex-request-20261024-config.md` を入力契約とし、Head設定の階層、skip網の構成enum、プロファイル配置、誤った選択宣言の検出を修正する。PRD098の学習・推論の数式とskip無効時の挙動を維持する。既存20implとは別の設定移行なので21implへ記録する。

## 主な変更

- DefaultDQNのHead初期化と射影は `head.init.*` / `head.projection.*`、Duelingは `head.dueling=none|standard`（既定standard）へ移す。skipの初期化・射影もhead配下へ移し、primaryからの継承を保つ。
- `skip.network_mode=action_out|action_in` を読み、SkipConfigとSkipHeadFactoryへenumで渡す。Agent構築時にactionのbind有無とmodeが一致することを検証する。
- skip.txtの共有profileと配線をagent.txt、部品をnn.txt、指標をmetrics_scalar.txt、Traceをcommon.txtへ移す。env所有のチェーンをenv側で明示する。`+=`は使わない。
- 現用のコード・設定・テスト・bat・設計文書を移行する。PRD072固定入力、過去Run・実験記録は保持する。PRD/ADR/CONTEXTはユーザーが別途更新するため編集しない。
- check_default_leaves.pyは `$` を含む不正な選択キーを抽出し、`.$`以外をerrorにする。

## テスト

Public interface / surface: ConfigData→DefaultDQNAgentConfig、Agent構築、SkipHeadFactory出力、ConfigManager解決値、静的設定検査のmain、Runner出力のmetricsとDOT。

優先behaviorとTDD順序:
1. 静的検査で `X.$+` を拒否する1テストをRED→GREEN。正常な `X.$` とmetric値内の `$action_info` は維持する。
2. Headの新キーとenumをConfig読み込み経由で確認するテストをRED→GREEN。初期化・射影継承、primary Duelingのnone/standard、未知列挙値の拒否を確認する。
3. action_inのbind欠落、action_outのaction bindをAgent構築で拒否するテストをRED→GREEN。既存TD/QR/IQNと包含型の正例も通す。
4. 設定解決と短いRunnerでprofileの実効値、actionのbindとDOTを検証する。

## 検証

- 編集前のソース・設定を `.scratch/prd098/config-20261024/source-before/`、hashをbefore-hashes.jsonへ保存。現用Debugを先にビルドし、実行体をbaseline-binへ凍結する。
- MSVCのVsDevCmd.bat経由で `cmake --build --preset x64-Debug`。ビルド・テストを性能測定や比較Runへ重ねない。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

- `[prd093-config-snapshot]` で7envの全キー・文字列値を前後採取。キー名の機械的写像とprofile移動で説明できる差分だけを許容する。旧キーは実装で読み替えず、比較器だけが過去側の名前を写像する。
- skip無効の4envは現時点の変更前バイナリ・設定で短い段0を採取し、変更後と正規化metricsを完全比較する。以前のPRD098採取結果へ上書きしない。
- coreの通常スイートと関連hiddenテスト、既定葉検査とそのPython回帰を実行する。
- 4envのskip/merge/shared/action_in、およびIQN envのskip_iqnを短いRunで確認する。action_inでは解決済みbind、network_mode、dot/skip.net.detail.dotのinput: actionを必須にする。
- 自己点検後、独立an-reviewへ計画と実装を依頼し、原文と結果をscratchに保存する。

## 前提

- ユーザー判断待ちは0。依頼書の命名・置き場を優先し、不明点は本節へ理由を記録する。
- Head初期化の外部読み口はDefaultDQNとRainbow、射影の読み口はDefaultDQNに実在する。MuZeroのhead_init_weightとImageClsのhead_init_configはコード内の初期値で、外部head_init/head_projection読取りを持たないため新しい設定機能は追加しない。現用文書中のキー例は新名へ移す。
- `use_dueling_net`の置換対象は依頼表どおりDefaultDQN。Rainbowの同名boolはこの依頼の指定対象ではないため保持する。
- skipのenumは既存dqn名前空間のSkipNetworkModeとしてagent.hppに置く。公開SkipConfigと内部HeadFactoryが同じ型を使うためであり、互換bool引数を残さない。
- skip網はenv非依存の小さな共有MLPなのでagent.txt所有とする。切替profileは=、共有配線の既定葉とnn部品は?=、選択宣言は=。
- LunarLander/Atariに加えて現用DropMergeもIQN。action_inの網チェーンはenvのrun profileから `@base > @iqn > @action_in` と明示し、Agentの汎用action_in profileがIQNのtaus配線を落とさないようにする。
- Atariのskip profileは現用のMunchausenを含むAgentチェーンを保つ。任意の別実験チェーンを自動追従する仕組みは足さず、env所有の明示選択として説明する。
- app.$は通常の選択チェーンなのでapp.@skip_traceを利用可能。既定Trace列には適用しない。
- 長時間学習と前回の性能改善は今回の設定修正範囲外。段0・smokeを実施する。

## 経過・結果

検証資材とログのroot: `.scratch/prd098/config-20261024/`。着手時のsource/config/doc 647ファイルをsource-beforeへ保存し、変更前Debug実行体・設定・ドライバを固定した。ログ番号04/06/18で各behaviorのREDを確認した。

- 静的検査: `check_default_leaves_test.py` 14テスト成功（25-final-checker-tests.log）。`check_default_leaves.py` は28,183代入、3,564監査、errors=0（24-final-audit.log）。最終監査JSONもroot配下へ保存した。
- 新Headキーの読み込み・継承・未知列挙値の拒否はRED 11失敗→GREEN 11成功。action bindの2方向は、有効/無効の両方でRED 4失敗＋正例4成功を確認して構築検査を追加した。
- Debug全体: VsDevCmd.batをcallして `cmake --build --preset x64-Debug --parallel 2`、終了0（19-debug-build.log）。
- core通常全テスト: `core/anet-core/bin/Debug/anet-core-test.exe`、766件成功・1件skip、166,844アサーション成功（20-core-tests.log）。skipは「CUDA利用不能時のエラー検査」をCUDA利用可能な環境で実行しないための既存条件（init_test.cpp:145、CUDA is available）。
- 環境テスト: LunarLander・DropMerge・GridMaze・ImageCls・Atariの各Debugテストがすべて終了0（23-env-summary.json、23-env-*.log）。
- 7env全解決値: `[prd093-config-snapshot]` でresolved-afterを取得し、`verify_config_migration.py config` が成功。新旧キー写像と共通Skip部品6葉の配置変更以外の差分は0（config-comparison.json）。
- 機械点検: 変更C++のusing namespaceヘッダ、production dynamic_cast、TODO、共通protocolの固有語は0件。既存ファイルのHEAD基準BOM保持、LF、batのCP932/CRLFを確認した（mechanical-check.json、source-check.json）。
- 現用PRD084性能ツールの新規Run生成・集計も新キーへ移行した。性能測定自体は今回実施しない。品質観点の設定例2箇所も同じ現行キーへ追随させた。
- 長時間設定: 新キーで段1〜3の24設定を `long-config/` に再準備した。Runnerは起動していない。段4は選択済み50M checkpointが無いため未生成で、status.jsonへ理由を記録した。

4envの段0はすべて完全一致。実行は変更前の凍結ドライバと、変更後の `verify_config_migration.py after --label config_after1`。正規化するのは時間・資源以外の全scalar点で、比較条件は同じseed・8lane・batch16・FP32・8,704experienceである。26-after-runs.logと各 `config_after1_ENV/comparison.json` に記録した。

| 環境 | scalar点数 | SHA256 | 結果 |
| --- | ---: | --- | --- |
| GridMaze | 20,095 | `54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8` | 完全一致 |
| LunarLander | 16,494 | `9ea069dbb542d46971b111524778d10c29c6cf7df093b2401879312a207e7edf` | 完全一致 |
| Atari | 18,442 | `cfd00ecc860aa8098ae5fd0b8c96836cc4c55a98954e7b63f14555f771eb9b66` | 完全一致 |
| DropMerge | 23,247 | `08c721d17951a9263a7230c40f10deadb2d8fd36d69b4b725f96922a933d03b3` | 完全一致 |

19条件profile matrixはすべて成功。実行コマンドは `verify_config_migration.py profiles --label config_profiles1`。各Runは1,280experienceでwarmup後64更新を確保する。実行体を固定し、ビルド・単体テストと重ねず逐次実行した。各profileでskip有効、mode、merge/sharedの実効値、IQNのtaus、Trace列2本、9種類の指標定義と有限値、解決値に$+が無いことを確認した。action_inは4envすべて実効bindが `skip_context, action` で、DOTにも `input: action` がある。

| 環境 | 条件数 | 結果 |
| --- | ---: | --- |
| GridMaze | 4 | 全条件成功、action bindとDOT確認 |
| LunarLander | 5 | 全条件成功、action bindとDOT確認 |
| Atari | 5 | 全条件成功、action bindとDOT確認 |
| DropMerge | 5 | 全条件成功、action bindとDOT確認 |

27-profile-runs.logとconfig_profiles1_profiles-summary.json、各Runのexecution.jsonおよびprofile-check.jsonへ記録した。静的検査・7env比較・4env完全一致・19条件smokeの受入はすべて成功。

独立レビューの走査で、検証スクリプト2関数の処理段階コメントと計画内のビルド実行形の不足を指摘され、追記した。Pythonは追記前後のASTが同一である（29-review-comment-check.txt）。挙動変更はなく、既に成功したRunの再実行は不要。最終判定は計画・実装ともOK（差し戻し0件）。原文は `.scratch/prd098/review-20261024-0306-config.md`。

自己点検は品質観点の全項目を適用範囲に従って通した。未実施は長時間学習と今回の対象外である性能再測定、段4設定はcheckpoint未指定のため未生成。ユーザー判断待ち・今回の受入の未解決は0。


## 並行更新されたPRDとの差異

最終チェックで、当方の編集対象外である098_10prd本文が着手前snapshotから別途更新されていることを検出した。変更内容は `.scratch/prd098/config-20261024/concurrent-prd-change.diff` へ保存した。当方によるPRD/ADR/CONTEXTの編集は無い。

新本文の§5.2・§5.6・確定事項21は、今回ユーザーが直接実行指定した `codex-request-20261024-config.md` と次の点で異なる。

- 依頼書はDefaultDQNAgent.@skip系・skip.net.@base/@iqn/@action_inをagent.txtへ移し、envがチェーン全体を選ぶ。新PRDはこれらのprofileを作らず、baselineの既定とenv/CLIの葉へ分ける。
- 依頼書は既存IQNのtaus bind検査を維持し、action bindの検査を同じ境界へ足す。新PRDはprimaryの融合済み特徴を使い、skip側のtaus bind必須を外す。
- 依頼書はTraceをcommon.txtのapp.@skip_traceへ移す。新PRDは各envのrun.@skipへTraceと長さ別指標も持たせる。

今回は直接の修正指示である依頼書を正本として実装し、並行編集で追加された別方針へは変更しない。最新PRDとの上記差異の統一は今回の受入外として残す。PRD本文を戻したり、今回のprofile検証を成功扱いのまま別構成へ差し替えたりしない。

## 完了報告

直接指定された設定修正依頼の実装・自己点検・必要な検証・独立レビューを完了した。今回の受入の未解決は0、追加の判断待ちは無い。長時間学習と性能再測定は実施していない。

起票候補は、別途更新された最新PRDと今回の直接依頼の方針統一のみ。差異は上節へ記録した。新たなissueの作成や並行文書の書換えは行っていない。git add/commit/pushは実行していない。
