# PRD093 attention エントロピー診断の実装計画

## 概要
TransformerEncoder の各層の attention 集中度を、学習更新前の専用 probe で測り、その更新の `$update_result` へ公開する。学習 forward の SDPA 経路と数値系列は維持し、未購読時は追加計算を行わない。

2026-10-16 ユーザー承認済み。本ファイルを実装の正本とする。未決事項監査のブロッカーは 0 件。

## 主な変更
### 1. 共用設定キーの改名
- DQN の `learner.plasticity.feature_key` を `learner.final_feature_branch_key`、ImageCls の `plasticity.feature_key` を `final_feature_branch_key` へ移す。baseline設定の既定値は `main_feature`。DTOは空文字で未指定を保持し、関連診断の購読時に必須検証する。
- DTO、読み込み、利用側、テスト、現用設定・文書を同時に移行する。旧キーの alias・検出・互換処理は追加しない。
- 改名前後の現用設定を解決し、全キー・文字列値を比較する。この段階では差分をキー名だけに限定する。
- PRD072 の固定入力・golden・manifest、過去 Run の dump、履歴資料は保持する。

### 2. capture の一般化
- `NetworkBranchCapture` を `NetworkForwardCapture` へ置き換え、通常 `Forward`、実行器 `Forward`、`ForwardUpTo` で共用する。既存の可塑性 capture 利用側も移行する。
- 要求は完全一致または末尾 `/` の接頭辞。`Wants`・`Add`・`Get`・`GetAll`・`Unfulfilled`・`Reset` を実装し、鍵ごとに Tensor を1個保持する。`GetAll` は書き込み順、`Reset` は要求を保って結果を消去する。
- 未要求・重複書き込みは fail-fast。forward 入口でリセットし、出口で完全一致要求の欠落と接頭辞要求の一致ゼロを検出する。エラーには要求、到達 branch、書かれた鍵、利用可能 branch を含める。
- Context 経由で capture を届け、NetworkStruct が module の素鍵へ `<branch>/<block インスタンス名>` を付ける。module 自身に位置を保持させない。直接 module を呼ぶ場合は素鍵を使う。
- branch 出力は detach して書き込む。branch 名の `/` は構築時に拒否する。capture なしでは位置文字列を生成しない。TraceCallback は変更しない。

### 3. 計算・probe・公開
- `nn_util` に、副作用のないエントロピー計算関数を追加する。attention 入力、Q/K 射影重み・bias、head 数を受け、autocast を無効化して射影から集約まで FP32・NoGrad で計算する。
- head・query・sample のエントロピーを平均し、`ln n` で正規化する。系列長 `n < 2` は NaN。pre-LN は正規化後、post-LN は層入力を使い、SN 有効時は実効重みを使う。
- TransformerEncoder は要求された場合だけ計算し、層数長の FP32 Tensor を公開定数 `TransformerEncoder/attention_entropy` の鍵へ1回書く。確率行列は保持しない。
- Agent が専用 RNG を所有し、named seed `attention_entropy_probe` で初期化して Learner へ貸す。Learner が購読・interval・測定状態を所有する。
- 発火更新のパラメータ更新前に、一様・非復元 sample を正規化し、eval・NoGrad・Learner と同じ autocast で final feature branch まで実行する。既定 Context を使い、IQN の `taus` は生成しない。
- 接頭辞で得た Tensor を実行順に連結し、全層まとめて1回 CPU 転送する。値は当該 `BatchUpdateResult` にだけ格納する。
- `GetScalar("attention_entropy.[n]")` を追加する。未測定・sample 不足は NaN。非整数・負の番号は購読時、層数以上は最初の測定で fail-fast。Transformer のない閉包は capture 未達で停止する。
- `learner.attention_entropy.probe.batch_size` は既定256、1未満は読み込み時エラー。測定境界に `ANET_PROFILE_SCOPE` を置く。

### 4. 設定・文書
- `metrics.scalar.@attention` に4層分を追加する。群は `48_agent_attention`、interval503、`@learn $learn_step $update_result`。baseline 外とする。
- DropMerge・Atari に `run.@attention` を追加する。Atari は既定チェーンの munchausen・replay_fit を維持する。
- NN設計130、DQN設計200、分析ガイド030、CHANGELOG を同期する。ガイドは層平均の推移として説明し、0.8 は比較に使えない旨の補足枠へ置く。
- 用語集・ADR は既存裁定を維持する。PRD093 用 README に再現手順と検証結果を残す。

## TDD と検証
各 behavior を **1テスト → RED確認 → 最小実装 → GREEN** の順で進める。GREEN 後だけ整理する。

1. **改名**：設定読み込みと既存診断の public 経路を確認し、現用 config dump の等価性を検証する。
2. **capture**：branch 出力取得を最初の tracer bullet とする。その後、接頭辞、順序、再利用、detach、重複・未要求・未達、`/` 禁止、実行器、部分 forward、`(*2)` の位置識別を確認する。
3. **数値**：一様で1、強い集中で0付近、系列長1でNaN。libtorch MHA の head 別確率を独立参照とし、bias、pre/post-LN、SN、CPU FP32・CUDA autocast を確認する。
4. **Learner**：購読 interval の更新前だけ測定すること、未測定時の値失効、不足時NaN、不正番号、複数 module の通し番号、未購読時の抽出・追加 forward ゼロを確認する。
5. **非干渉性**：TD/QR/IQN で、probe 前後の学習用 RNG、PER・sampled-once 統計、正規化統計、network buffer、実行モードを比較する。例外時のモード復元も確認する。
6. **結合**：DropMerge 本線、Atari 既定チェーン＋`@vitva > @attention` を短期実行する。learn_step503・1006の両測定を含め、4タグの実効定義と有限な0〜1の値、Atari の既存診断行を確認する。
7. **等価性・費用**：同 seed・決定論設定で OFF/ON の tag・step・点数・数値 checksum を厳密比較する。除外は性能・資源群と追加 attention 群だけとし、既存 Noisy 診断も比較する。checkpoint SHA は判定に使わない。Release で測定時間と Run 実時間比を報告する。1%超過だけでは不合格にしない。

実装前に変更前の実行体・設定・比較基準を保存する。MSVC 初期化経由の Debug 全体ビルド、関連テスト、既定スイート、PRD072 比較・既定葉検査を実行する。検証ログは `.scratch/prd093/` へ保存する。

## 前提
- 完成単位は「設定改名」と「capture＋attention診断」の2つ。後者はテスト・両 smoke・等価性・費用報告・文書同期まで含む。
- 長期 Run、正則化、head 別公開、TraceCallback 統合、DefaultDQN 以外への診断追加は対象外。
- 既存の未コミット変更を保持する。既存 Run の停止、コミット・push は行わない。

## 実行検証の分離（2026-10-16 ユーザー合意）
既存Runの実行中は編集・静的確認だけを進める。ビルド、CPU/GPUテスト、両smoke、性能測定はRun終了後に実施する。RED確認済みの改名を除き、各behaviorのテストと最小実装を順に編集し、RED/GREENの実行確認は保留として記録する。実装済みと検証済みを区別し、GPU受入・性能報告まで機能完了とはしない。

## 編集・静的確認の到達点（2026-10-17）
共用キーの改名、鍵要求型captureと既存利用側の移行、FP32エントロピー計算、Transformerの位置付き層数長Tensor、Agent専用RNGと更新前probe、購読と更新結果、4行のmetrics/両Run profile、日本語設計・分析ガイド・CHANGELOGを編集した。joint targetの後半B行という既存可塑性挙動は、captureへ再書込みせず利用側の結果で切り出す。

public/config経路のテストと、改名比較・決定論CPU比較・両smoke・Release時間/メモリ記録の資材を `core/anet-core/testdata/prd093/` に揃えた。実行の順序と未確認項目は同READMEに記載する。性能レポートのNVTX/Tracy range合計は未採取であり、OFF/ONの実時間差で代用しない。

静的確認は `git diff --check`、追加Python 3ファイルのAST parse、UTF-8/LF、テスト名重複、現用の旧設定キー・旧capture型の検索、PRD072固定資料41ファイルのhash不変、既存gui.cpp変更の保持。記録は `.scratch/prd093/static-review.json`。実行中Run確認後のビルド・テスト・Runはすべて保留しており、コンパイル・数値・非干渉・性能は未検証。コミット・pushは行っていない。

## 軽量実行検証の再開（2026-10-17 ユーザー許可）
「GPUメモリを大量に使わない、性能検証以外であれば実行問題ありません」という追加指示により、Debugビルド、GPUを不可視にしたCPUテスト・小型CPU Run・設定比較と、小型CUDAテストを再開する。Release実行体、実行中Run、大きなGPU smoke、性能測定には触れない。CPU RunだけはCUDA_VISIBLE_DEVICES=-1を固定し、既存Runnerとの並走を許可する。性能・本線smokeは従来の排他ガードを維持する。

## 軽量検証の結果と残項目（2026-10-17）

- 共用設定キーの移行は現用7入力の全キー・文字列値比較でキー名以外の差分0。PRD072固定17入力・156アサーション成功、固定資料41ファイルのhash不変、既定葉検査25,981代入・3,463監査・エラー0、検査器回帰12件成功。
- MSVC初期化経由のDebug全体ビルド成功（`24-debug-build.log`）。性能・hiddenテストを除くCPU全体スイートは697件中688件成功・9件skip、91,715アサーション成功（`25-default-cpu-tests.log`）。
- 変更前・変更後OFF・変更後ONのCPU Runは既存Noisy診断を含む61タグ・31,694点のtag/step/点数/数値が完全一致。4層すべてでlearn_step503・1006を含む有限な0〜1を確認した。費用の比較には使用しない。
- 小型CUDA検証でTD/QR/IQN、各Munchausen方式、RNG・PER・正規化・SN buffer等の非干渉を確認。独立MHA参照テストのdevice比較を `cuda` 対 `cuda:0` から実入力deviceとの比較へ修正後、NNのCPU/CUDA BF16テスト3件・225アサーション成功。初回127.7秒、再確認15.8秒。GPU全体使用量の観測増分最大618MiB、子プロセスの停止なし。既存Runの変動を含む観測であり、process別allocationや本線の性能保証ではない。
- 通常forwardのcapture未達エラーで到達branchが空になる点は、追加テストのREDを確認してから修正し、CPU全体スイートでGREENを確認した。改名前のエラー文言を期待していた既存テストも新キーへ移行した。
- 初期の改名REDは、後続検証でgetterの観測先の誤りが判明したため、有効なbehavior REDとは扱わない。DTO公開メンバを検証するテストへ直し、現在の読み込みを確認した。編集先行時に保留したTDDの実行順序が後から満たされたとは報告しない。

変更内容・再現手順・ログの対応は `core/anet-core/testdata/prd093/README.md` に集約する。通常forwardのエラー表示修正以降、学習数値を変えるproduction変更はない。英語の現行設計文書も旧設定キー・旧capture型への参照だけを移行し、attention章の翻訳追加は行っていない。

設定改名は検証済み。capture＋attention診断の実装と許可範囲の軽量検証は済んだが、DropMerge/Atariの大容量GPU本線smokeとRelease性能・NVTX/Tracy費用報告は保留であり、PRD全体は受入未完了。既存RunとRelease実行体、ユーザーの未コミット変更を保持し、commit/pushは行っていない。

## GPU本線検証の再開（2026-10-17）

ユーザーの「GPU空きました。再開」により、保留していた両本線smoke、ReleaseビルドとOFF/ON等価性・費用測定、NVTX/Tracy範囲採取を再開する。起動済みRunnerがないこととGPU空きを確認した。変更前のRelease実行体は退避済みのまま保持し、現用Releaseは今回の実装で更新する。既存Run停止・commit/pushは行わない。

GPU再開後の検証資材修正: Atari既定チェーンがapp名・batch終了予算・評価scheduleを上書きするため、採取器の検証条件を末尾の `run.@prd093_validation` へ移した。実入力を読むConfigManagerのhiddenテストでRED→GREEN（8アサーション）を確認した。初回の誤設定による検証Runは、今回の採取器の子であることをPID・親・コマンドラインで確認して停止し、出力を保持した。既存ユーザーRunは停止していない。Nsight 2026.3の反復採取の接続エラーを避け、2026.1の通常NVTX採取・SQLite exportを事前確認し、費用集計器3テストをRED/GREENで追加した。

## GPU検証の到達点（2026-10-17）

DropMerge本線・Atari既定チェーン＋`@vitva > @attention`のDebug smokeが正常終了し、4タグの503・1006測定とAtari既存診断19行の実測を確認した。Release OFF/ON4本とNVTX2本も正常終了。OFF/ON実効設定はRun名・追加attention4行以外の全キー/文字列値が一致した。最終Release全体ビルドとCUDAを含むPRD093関連12ケース・91,970アサーションが成功した。

Releaseの503→1006実時間はDropMerge 32.924秒→32.861秒（−0.1913%）、Atari 22.946秒→22.972秒（+0.1133%）。各1対の観測であり、統計的な性能保証とはしない。NVTXの同区間probeは22.782ms/33.271秒（0.0685%）、12.840ms/22.628秒（0.0567%）。初回は317.513ms・1,225.349ms、3回合計363.316ms・1,252.002ms。同じRunの最終通常時計点までを母数にすると0.4394%・1.5036%。最終時計点とprocess終了を区別し、Nsight CLIのreport生成時間をRun時計に混ぜない。初回を含むAtariの1%超過だけを不合格としない。

CPU等価性・改名比較・PRD072・既定葉検査の結果は引き続き有効で、GPU再開後のproduction変更はない。全体結果・資源表・初回費用・TDD順序未達・失敗試行の説明は `core/anet-core/testdata/prd093/README.md` に集約した。保留していたGPU機能・費用検証は完了し、設定改名とcapture＋attention診断の機能検証を終えた。過去のTDD順序の未達は解消済みとは扱わない。commit/pushは行っていない。

## レビュー指摘の修正（2026-10-17）

`final_feature_branch_key`のDTO既定値を空文字へ戻した。`main_feature`はbaseline設定が指定する値とし、baseline未適用の設定・dumpでは未指定を保持する。これにより、関連診断の購読時に既存の必須検証が働く。旧キーの検出・alias・読み替えは追加しない。DQN/ImageClsの設定不在と、現用DropMerge/Atariのbaseline適用後の値を回帰テストで区別した。修正前は未指定を確認する2アサーションが失敗した（`55-review-default-red.log`）。

`dqn_based_agent.cpp`と`nn_impl.cpp`のUTF-8 BOMを復元し、`<charconv>`は自前ヘッダの後で標準ヘッダと並べた。設定txt 5本は全行LFの状態から、`core.autocrlf=true`に合わせて作業ツリーをCRLFへ統一した。BOMと改行を正規化した文字列内容は変更していない（`review-formatting.json`）。C++・MarkdownのLFは維持する。

既存のGPU smoke・等価性・費用記録は修正前の実行体による結果として保持する。これらのRunは共通境界キーをbaselineから明示的に解決しており、今回の修正はその値や診断の数式を変えない。今回のレビュー対応では性能Runを再測定しない。

レビュー修正後のDebug全体ビルドは成功（`56-review-debug-build.log`）。CPU既定スイートは697ケース中688成功・9スキップ、91,719アサーション成功（`57-review-cpu-suite.log`）。未指定値の回帰、baseline適用後の値、DQN/ImageClsの購読時検証を含む。

レビュー修正後のRelease全体ビルドも成功（`58-review-release-build.log`）。CPU指定のPRD093関連テストは12ケース中10成功・CUDA対象2スキップ、46,238アサーション成功（`59-review-release-tests.log`）。BOM・include順・改行、現用の旧契約参照なし、PRD072固定資料41本と既存gui.cppの保持を最終監査した（`review-audit.json`）。
