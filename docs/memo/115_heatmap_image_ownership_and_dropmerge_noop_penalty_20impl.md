# PRD115 画像メモリとNOOP判定の修正 実装メモ

## 概要

[PRD115](115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)のD1〜D5とA1〜A7を実装する。公開API・設定・既定値を保ち、画像の4経路とDropMergeのNOOPへの罰に変更を限る。ユーザー判断が必要なブロッカーは0。

## 主な変更

- 画像: heat_map.cpp内に状態を持たない画像確保helperを置く。wxImage自身に未初期化の画素領域を確保させ、IsOk/Createの失敗を画素アクセス前に描画対象・幅・高さ付きANET_SYSTEM_ERRORにする。TimeHeatMapの空画像だけは明示的に黒で埋める。通常画像・Histgram・SweepedHeatMapは既存の画素描画を使う。基底HeatMapの別の確保経路は変更しない。
- DropMerge: Stepで既に得ているis_noop_actionをNOOPへの罰の条件に使う。noop_overrideと実行されないDROPの意味、報酬の加算位置は保つ。
- 回帰テスト: heat_map_test.cppを新設し、固定した小画像の全画素・画像種別・寸法と、生成元・元画像を破棄した後のコピーを確認する。DropMergeEnv_test.cppに同seed・同行動列の2個体の比較を追加する。AuxDataのrewardsから現在の累積報酬を読み、物理状態の比較では報酬履歴だけを除く。
- 検証資材: core/anet-core/testdata/prd115/に画像メモリprobe・検証スクリプト・READMEを置く。既存Debugビルドのコンパイル条件とリンクライブラリを再利用して、実物のheat_map.cppを独立probeへリンクする。本体へtest-only APIを増やさない。
- ASan probeでは修正前に保存したheat_map.cppと修正後の各4経路を別プロセスで走らせる。確保失敗は同じDebug CRTの外部確保フックで要求画像サイズの確保を1回失敗させ、ANET_SYSTEM_ERRORの診断と停止を検証する。検出・注入が働かなければ未達として原因を追う。
- 設計文書の120/140の日英版に現行の挙動と検証資材の参照を追記する。実装完了時に索引issueを実装済み、子2件を直った見込みとする。statusは変えない。

## テスト

- Public interface / surface: ImageSource::RenderRaw/Render/GetImageSubType、wxImageのコピーと破棄、DropMergeEnvのReset/Step、返された結果のGetAuxData。
- 優先behaviorとTDD順序:
  1. 既存Debugビルドと既存画像・DropMergeテストのベースラインを確認する。
  2. tracer bulletとしてdirectの列0の罰の差を1テストで再現（RED）し、条件の1行修正でGREENにする。
  3. 残るモード・noop_override・実行されないDROP・複数NOOP・終了step・ゼロの罰を追加し、各テストを確認する。
  4. 画像の4経路は1経路ずつ、固定画像テストとASanによる修正前の不一致検出（RED）、その経路の最小修正、通常テストとASan完走（GREEN）の順に進める。
  5. 外部確保失敗注入で各4経路のfail-fastと診断を検証する。TimeHistogramのScale経由も含めて画像テストを確認する。
- 画像期待値は修正後の描画処理から作らず、修正前の小画像か独立した期待値と照合する。
- DropMergeはCPU・fixed seed 123、罰0対-0.25、許容絶対誤差1e-5。直接指定の列数3で先頭・中間・末尾を覆う。各ケースを独立した個体・Resetから始める。
- A2は経路ごとに独立したプロセスで100回。修正前は各経路の不一致を検出、修正後は4経路とも完走が条件。

## 検証

Windows/MSVCの初期化を同じcmd内で行う。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test DropMergeEnv-test'
```

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[image]"
```

```bash
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe "[dropmerge]"
```

画像probeのスクリプトと実際に実行したコマンド・結果は経過へ記録する。基準ソースと実行ログは.scratch/prd115/へ分け、既存Runには書かない。必要なDebugビルド・関連テスト・ASan・確保失敗注入を済ませて独立レビューへ出す。

## 前提

- repo evidenceで解決済み: wxImageが所有する画素領域はfreeで解放される。既存4経路のnew[]は不一致。4モードのNOOP判定はStep内に既にある。Directの全行動はDROP。
- 2ximplへ固定: 画像確保はwxImage自身、未初期化領域を既存の描画処理で埋める。検証専用のCRTフックは本体外に置き、画像の要求バイト数と検証中の区間を限定する。
- PRD範囲外: 基底HeatMapや画像全体の構造整理、報酬値・行動集合の変更、View用の前エピソード履歴の仕様変更、長期学習。
- 新しい公開名・設定名・ADR・用語は追加しない。
- 既存の未コミット変更は保持する。別作業が更新中のwikiはその差分を残して追記する。

## 経過と結果

- 2026-11-04: 未決事項監査を通過。承認済みPRDと実装起動の指示に従い、この計画を実装の正本として保存した。
- 変更前Debugビルドはno work。画像は43 assertions / 2 cases、DropMergeは1081 assertions / 29 casesで合格した。
- 最初のDirect列0のテストは報酬-0.25対0で失敗（RED）。罰の条件をis_noop_actionへ変え、1 assertionのGREENを確認した。その後の4モードと境界の追加テストは1725 assertions / 3 casesで合格した。
- 終了stepの追加テストは最初にログのダイアログ待ちとなったため、既存LogCaptureGuardをテスト側へ入れて再実行した。本体のログ仕様は変更していない。
- ASanは修正前の4経路をすべてalloc-dealloc-mismatchとして検出した。MSVCの既存ライブラリに合わせ、検証用ASanビルドだけSTLのstring/vector注釈を無効にした。ヒープの不一致検査は有効である。
- 空画像の最小修正と固定画素・破棄後コピーのテストは10 assertionsで合格した。
- probeでは画像の100回検査と画素採取が完了した後にDLL終了処理でtimeoutすることを、PASS行と採取画素の出力から確認した。対象画像の破棄後にWindowsの自己終了を使う方式へ改めた。画素採取専用モードは画像を破棄せず終了するため、A2の合格には数えない。
- 基準の実ソースは.scratch/prd115/baseline_heat_map.cpp（Git blob 1fff8bfb5629b1e23b4d732f3ddb3a7991286707）に保存した。
- Debugビルドは上記2対象で成功。初回レビュー前の画像テストは657 assertions / 7 cases、DropMergeは2850 assertions / 32 casesですべて合格した。TimeHistogramの独立期待値にあった色の計算誤りは、既存の値1=中間色・値2=上限色の規則に沿って直し、全画素の一致を確認した。
- [検証資材README](../../core/anet-core/testdata/prd115/README.md)の3コマンドを実行した。baseline ASanは4経路すべてでalloc-dealloc-mismatchを検出して終了値1、current ASanは各100回の生成・コピー・破棄を終了値0で完走した。current CRTは4経路の画素採取と4経路の実確保失敗注入がすべて終了値0だった。スクリプト自体の終了値はいずれも0。
- 実行結果とコマンドの原文は.scratch/prd115/memory-baseline-asan/、memory-current-asan/、memory-current-crt/のresult.jsonと各logに保存した。基準ソースのSHA-256は31d475ec625318444d035bb726d96dae158f1c6c668ac32ee44cee6d4ba453b3、修正後はcff7a887f31de7e990c6425f46984cff44dbbc0adc4b6c4e521b7910bbc321f2。

## 受入条件の確認

| 項番 | 結果と根拠 |
| --- | --- |
| A1 | 合格。heat_map_test.cppの5ケースで空・reset・データありの寸法、画像種別、RGB全画素、破棄後コピーを比較。TimeHistogramのScale経由も空・データありを確認した |
| A2 | 合格。修正前4経路のASan不一致検出と、修正後4経路それぞれ100回の生成・コピー・全所有者の破棄を確認した。alloc_dealloc_mismatch=1を明示した |
| A3 | 合格。実際の画像の確保をDebug CRTフックで1回失敗させ、全4経路で注入印とSystem errorの描画対象・要求幅・高さを確認した |
| A4 | 合格。CPU・seed 123の2個体で4モードの全行動、Directの先頭・中間・末尾列、正確な報酬差を確認した |
| A5 | 合格。noop_overrideの有無、busyで実行されないDROP、1 Step内の複数物理step、罰0の同行動列を確認した |
| A6 | 合格。複数NOOPと終了stepの報酬差・現在の累積報酬差、観測・全物理Aux・生スコア・終了判定の一致を確認した。加えてNOOPを持つ3モードで、DROP後のbusyを残して続くNOOP自体を複数物理step進め、同じ結果の物理step数>1、Step報酬差が罰1回分、累積報酬差の一致を確認した |
| A7 | 合格。Debugビルドと画像・DropMergeの対象テストを実行した。長期学習と対象外の画像経路は今回の検証範囲に含めない |

## 自己点検と未決

- docs/quality/impl.mdの全項目を適用範囲に従って自己点検した。計画・PRDの対応、fail-fast、局所的な変更、名前・include・定数・日本語コメント、計測、wxImageの所有権、public経路のテスト、検証記録と日英設計文書の参照を確認した。
- 検索では変更C++のTODO・dynamic_cast、共有プロセス契約のプロジェクト固有語は0件。変更ファイルはLF・UTF-8で、既存7ファイルのBOM有無を保持し、git diff --checkは問題なし。新しいヘッダ・設定・API・識別子の改名・互換層・skill・GetScalarの変更はなく、該当する項目は対象外。
- 未実施の必須検証: なし。DLL静的終了処理は画像所有者の破棄後に除外している。採取専用dumpモードはA2に数えない。
- ユーザー判断が必要な実装の未決: なし。

## レビュー対応

- 1往復目: impl-planとimplが差し戻し。A6の複数物理stepを選択したNOOPと同じStepで確認できていなかった点と、追加テストのモード名・画像種別・reload_min_stepsの重複値を指摘された。自分で直せる2件として両方に対応した。原文は.scratch/prd115/review-20261104-1446-impl.mdへ保存した。
- A6はuse_settle_after_drop=true、settle_max_steps=2、reload_min_steps=20でDROP後のbusyを残し、続くNOOPを2回選ぶテストを追加した。末尾の同じNOOP結果の物理step数>1と、比較helperによるStep報酬差・現在の累積報酬差を確認する。DirectにはNOOPがないため、direct_noop・move・move_fastの3モードで行う。
- 指摘の定数化は今回追加したテストの範囲だけに適用した。変更前からある直書きは変更していない。
- 修正後、同じDebugビルドと対象テストを再実行した。画像657 assertions / 7 cases、DropMerge3126 assertions / 32 casesですべて合格。追加したNOOPと複数物理stepの節も合格した。productionはレビュー前から変更しておらず、画像メモリ検査の対象SHA-256も同じである。
- 2往復目: impl-plan・implともOK。不足点・修正要求・範囲外の検出・開発者への確認事項はなし。再確認したA6と重複値の対応も確認済み。根拠がなく見送った指摘はなし。原文は.scratch/prd115/review-20261104-1453-impl.mdへ保存した。
- 別作業で設計120/140の英訳同期が加わったため、その差分は保持した。レビュー担当もPRD115の対象節は現在の内容で確認した。
- anet-update-issueの判定として、issue-20261104-cc2529を実装済み、issue-20261103-dd8058とissue-20261103-be4cd5を直った見込み、判定日を2026-11-04へ更新した。status・監理の項目・本文は変えていない。wiki_support.py mapsでINDEXと課題の派生区画を再生成した。
- wiki317/322の実装後の記述と設計文書の出典を更新した。PRDのblobと実装判定日、設計文書のblobはdocs/wiki/log.mdに記録した。今回対象のwiki2ページの相対リンク199件にリンク切れはなかった。
- mapsの再実行はupdated=[]・unresolved_prds=[]で終了値0。最終本文を再読し、文字コード・BOM・改行・差分を確認する。保存後のblob・SHA-256・mtimeは.scratch/prd115/final-artifacts.jsonに記録する。

## 工程末の候補

- 選択・修正済み1件: 2026-11-04、ユーザーが候補1を選択したため、docs/design/140_observability.jp.md末尾のOptunaガイドの既存リンクを../optuna.mdからoptuna.mdへ修正した。前回レビューOKのPRD115の追記は保持した。
- 新規・既存へ追記・見送り推奨の候補: なし。今回のPRDの受入条件には影響しない。

## 完了報告

- 実施内容: 画像メモリの4経路とDropMergeのNOOPへの罰を修正し、A1〜A7の受入条件を満たした。
- 変更ファイル: heat_map.cpp・heat_map_test.cpp、DropMergeEnv.cpp・DropMergeEnv_test.cpp、testdata/prd115の3ファイル、設計120/140の日英、関連wikiと課題の判定・派生一覧、およびこの実装記録。
- レビュー結果: 2往復目で実装計画・実装ともOK。NOOPの複数物理stepと定数化を再確認済み。Debugビルド、画像657 assertions、DropMerge3126 assertions、ASan4経路、確保失敗注入4経路は合格。見送った指摘はなし。
- 決まらなかったこと: 実装の未決・未実施の必須検証はなし。範囲外の既存リンク切れも選択を受けて修正済み。
- 判断してほしいこと: なし。
- 次にできること: コミット・stage・pushは行っていない。

## 追加選択の対応結果

- 2026-11-04: ユーザーが既存Optunaリンクの修正を選択。文書内の全相対リンク41件でリンク切れ0、UTF-8の厳密読み取り、LF、基準HEADとのBOM有無一致、git diff --checkを確認した。コード変更がないためビルド・実行テストは対象外。
- 品質確認観点の全項目を今回の文書変更の適用範囲に従って自己点検し、同じレビュー担当へ入力元なしのimplとして依頼。今回のリンク1行について1往復でOK。不足点・修正要求・範囲外の検出・開発者への確認事項はなし。原文は.scratch/prd115/review-20261104-2129-optuna-link.mdへ保存した。
- 保存後の本文・blob・SHA-256・mtimeは.scratch/prd115/optuna-link-final-20261104.jsonへ記録する。前回完了時のfinal-artifacts.jsonは当時の記録として保持する。
- 2026-11-04: 索引issueと題材issue 2件の対応内容に、PRD115の行を追記した。「レビュー対応」の節の「本文は変えていない」は、判定を更新したときのことである。
