# PRD084：NN公開ヘッダ・命名・コメントの整理

## 概要

引き継ぎ文書のH1〜H3を実装する。NN利用側の契約を公開ヘッダへ、Module／Head実装側の契約を内部ヘッダへ整理し、実行器の名前をExecutorに統一する。数値・設定値・RNG消費順・所有権は変えない。

基準はHEAD ed9311c5とし、着手時の差分も記録する。本計画をコード編集前にBOMなし・LFで保存し、正本とする。

## 主な変更

- nn.hppからHead・Head実行interface・HeadFactory・HeadGraphVizInfo・WeightNormConfigの定義をnn_impl.hppへ移す。公開側には引数・戻り値・private署名に必要な前方宣言だけを残す。NetworkForwardStateもprivateコンストラクタで必要なため前方宣言を維持する。
- 公開ヘッダを構造設定→実行設定・呼び出し情報→Network→Forward実行器→診断・可塑性→Builderへ並べる。診断型の後置に必要な前方宣言を追加し、新しいヘッダは作らない。
- observers.hppのanet/nn.hppを外し、不足する利用側へ必要なincludeを追加する。

| 対象 | 新名 |
|---|---|
| NetworkForwardExecution／NetworkHeadExecution／NetworkModuleExecution | それぞれ末尾をExecutorへ変更 |
| Network::PrepareForward | CreateForwardExecutor |
| Network::PrepareDefaultForward | CreateDefaultForwardExecutor |
| Module／HeadのPrepareForward | CreateExecutor |
| LinearProjectionModule::Execution | Executor |
| LinearProjectionModule::PrepareProjection | CreateProjectionExecutor |
| PreparedHeadExecution／MakePreparedHeadExecution | PreparedHeadExecutor／MakePreparedHeadExecutor |

現用コード・テスト・コメント・エラー文言・計測ラベルを移行する。

- 公開型すべてに日本語の概要、公開データメンバと列挙子に行末コメントを付ける。shape・単位・未成立条件は実装から確認する。SN対象がない群のsigma_maxはNaNのFP32 scalarと記載する。CreateExecutorのnullptrは本体を直接呼ぶ契約として説明する。
- 設計130／200の日英版、用語集、所有権ガイド、READMEの現行説明、外部拡張配置案のNN関連表を更新する。PRD084には時系列順で2026-10-11の改訂行を追加する。既存ADR・過去計画・過去検証節は保持する。

### 2026-10-11 レビュー修正

- 公開コメントの8統計順序と名前付き実行器の前提を戻し、関数合成・重み同期の説明とDoxygenの行末記法を正す。
- 配置PRDのHead公開面と到達本数を同期し、PRD084の改訂引用を連続させる。英訳のtranslated-fromは現在の日本語文書blobへ更新する。
- Module／Headの内部契約を隣接させ、Run資材にUTF-8 BOMを付けてWindows PowerShell 5.1の構文を検証する。
- コメント・文書・宣言配置・BOMのみを変更する。Debugビルドと関連テスト、機械監査を行い、数値・RNG・設定に変更がないため学習Runは再実行しない。

## テスト

- 編集前に既存Debugビルド・関連テストのGREENを確認する。
- tracer bulletは既存の「既定実行が部品の準備を呼ばない」テストとする。新API名へ移してコンパイルREDを確認し、最小の改名でGREENにする。その後、内部改名とヘッダ移動を進め、各段階で関連テストを通す。
- 出力・勾配・保持・同一時計・同時Forward・BindによるState共有・Network寿命・保存読込・AMP・診断非干渉を維持する。テストタグと期待数値は変えない。
- 公開型の概要、公開メンバ・列挙子のコメント、移動対象の定義と旧識別子の残存を機械検査し、ログへ保存する。

## 検証

資材・ログ・変更後出力は.scratch/prd084/forward-refactor/api-20261011へ分離する。基準の実行体・DLL・設定・seed・checkpoint、HEAD・差分・Git indexを保存する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 1'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --parallel 1'
```

- Debugの関連テストと全coreをCPU／CUDAで実施する。CPUはプロセス限定CUDA_VISIBLE_DEVICES=-1とする。
- 現用コードの旧識別子・Executerが0件、公開ヘッダの移動対象定義が0件であることを確認する。必要な前方宣言は許容する。BOMはHEADと一致、LF、対象git diff --checkを確認する。
- 最終ReleaseでCPU LunarLander OFF500kを実行し、88184点とb76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5の一致を確認する。既存資材の旧OFF checkpoint読込も実施する。
- Release実行体が使用中なら既存プロセスを操作せず、同じ最終オブジェクト・リンク条件で別出力先へリンクし、その実行体を指定する。
- SHA・結果・変更ファイル・未実施理由をforward_refactor_verification.mdの「公開面と命名 2026-10-11」節へ追記する。ON Runと性能再測定は行わない。

## 前提

- ユーザー判断が必要なブロッカーは0。公開ヘッダのState前方宣言とsigma_maxのNaNはrepo evidenceで解決済み。
- BindNetwork削除、既定生成API統合、実行の木・Head基底・Learner用途ペア整理は後続へ回す。
- CallInfoの不変クラスとWithRowMapping、再利用規則、PreparedModuleChildren、各部品のPreparedを維持する。互換alias・overload、新用語、新ADRは追加しない。
- 許可されたヘッダの空白・宣言配置整理は取り込み、行末空白を除く。無関係な変更・現在のGit index・ブランチ・過去artifactは保持する。
- add・commit・pushは行わない。計画承認後、Defaultモードで本計画から実装する。
