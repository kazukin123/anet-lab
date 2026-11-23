# Evaluation Trace の内的報酬列 実装メモ

## 概要
PRD127 §4 D1〜D20 と §5〜7 に従い、学習の採点を動かさず、Evaluation Trace に RND の誤差・標準化した値・ボーナスを出す。

## 主な変更
- intrinsic.hpp / intrinsic_rnd.hpp・cpp / intrinsic.cpp: 読むだけの採点、lane 診断の鍵、独立した推定器の Clone を追加する。whiten の累積・EMA と variance_feedback を扱う。checkpoint の契約は変えない。
- rl.hpp: AgentStepDiagnosticsRequest、ActorSyncFollower、AttachSyncFollower、Suspend / Resume を追加する。現用の呼び出し・override を同じ変更で移行する。
- default_dqn_agent.hpp・cpp / dqn_based_agent.hpp・cpp / rainbow_agent.hpp・cpp: β を学習の入口で記憶し、DefaultDQN の診断だけに採点を接続する。写しは診断部品が所有し、Actor は weak_ptr で同期に従う相手を保持する。列要求が無ければ採点・複製・接続・追加のロックを行わない。
- metrics.hpp・cpp / EvalTracePanel.cpp / EvalPanel.hpp・cpp: 宣言から鍵を重複なく集め、Runner の Actor を渡す。pane を閉じたら休ませ、開いて写しが無ければ通常の Actor 同期を呼ぶ。
- Atari.txt・GridMaze.txt・LunarLander.txt の run.@rnd に235 rnd_bonus・600 rnd_err・610 rnd_normの列を追加する（工程後の列配置の改訂を反映）。
- PRD §6.5 の現行設計文書 020・100・140・160・200・212 の日英を追随し、140 にシーケンス図、212 に制約と費用の確認状況を記す。wiki の既存の並行変更は保持する。
- core/anet-core/testdata/prd127/ に比較・測定の手順を保存する。課題 issue-20261107-1226e5 の対応内容と判定を実測に合わせて更新する。

## テスト
- Public interface / surface: LifelongNoveltyEstimator の Factory・EstimateWithoutUpdate・Clone、Agent::CreateStepDiagnosticsProvider、Actor::Sync / MakeAction、StepTraceBuilder、EvalTracePanel::SetActive、出荷設定。
- 優先 behavior / TDD 順序: 最初に Factory から読むだけの採点と既存の学習を通す 1 テストを RED→GREEN にする。次に式・統計未成立・warm-up・Clone、step 診断の値と β・done mask・観測選択、同期・寿命・Suspend/Resume、ロックの待ち方、評価専用 checkpoint、無効・未知鍵・Actor 欠け・非有限値、Trace を挟む学習不変性、出荷列・pane の順に、挙動ごとに 1 テスト→失敗確認→最小実装→成功確認を繰り返す。タグ [prd127] を使う。
- PRD §7 の 1〜12・15 の条件を個別に対応付ける。§7 の13（画面）・14（GUI の性能測定）は実施できた範囲を記録し、未実施は残件として明記する。

## 検証
- 変更前の Debug 全体ビルドを VsDevCmd.bat 経由で行い、実行体・設定を .scratch/prd127/ に保存する。
- PRD123 の比較ドライバを再利用し、今回の before/after に対して PRD127 §7-11 の 6 条件（LunarLander の通常 RND 100k・Taiga 8値の RND 100k・RND 無効 4環境各8,704）を流し、時刻・資源以外の全 scalar checksum と metrics.scalar.defs を比較する。設定は変更前の写しで固定する。
- Debug 全体ビルド、[prd127]、既存 [prd116]・[prd123]・step 診断・metrics・GUI の関連テスト、[shipped_config]、check_default_leaves.py と設定の全キー・文字列値の比較を行う。
- 長時間のビルド・テストの stdout/stderr は .scratch/prd127/ に保存し、画面には終了コードと短い要約だけ返す。
- 自己点検で docs/quality/impl.md 全項目を適用範囲に従って確認し、計画と実装を anet-review-agent にまとめて依頼する。

## 前提
- 未決事項監査: ユーザー判断が必要なブロッカーは0。PRDの決定点は現行コード・PRDの契約から解決できる。読むだけの採点結果は LifelongNoveltyEstimateResult（CPU float の multipliers と鍵別の lane Tensor）で運ぶ。入力の転送を終えた同じ CPU 上で式とボーナスを計算し、表示のための再転送を避ける。
- PRD D4 は UI スレッドの学習ロック禁止規約の、開発者が決めた例外である。network・推定器・β は同じ読み取りロックの中で写す。別のロックは作らない。
- 新たな device 転送は lane のループ外でまとめる。Trace の各 step に RND の target/predictor forward と往復転送が各1回ずつ入り、Atari 1 lane では計1〜3ms の見込み（PRD §5.4）。Actor 同期では推定器の複製が数ms/回、通常は10秒に1回である。性能に合否の閾値は置かない。
- scope外: 仮説の実験、DropMerge の RND プロファイル、評価 session の保存、Actor 同期全体の刷新。U1 の既定列の採否は測定後の判断として残す。GUI を動かせない場合は課題を残件有りにする。
- 無関係な index・worktree の変更を保持し、stage・commit・push はしない。

## 決めたこと
- PRDの D1〜D20 を維持する。β はcheckpointに保存しない。未知鍵の検出は既存の最初の読み出しのままにする。

## 経過
- 2026-11-12: PRD・既存コード・品質確認観点を照合し、ブロッカー0で計画を保存した。使用量確認時の週枠は残り22%。消費率の予測値は算出できないため提示していない。


## 検証と中断の記録
- 変更前 Debug 全体ビルド: 成功（.scratch/prd127/build-before.log）。Runner SHA256: 1d76c6e266ea097785eb0113b43e2431072e7ac57a423623714451ec9f89c5c1。実行体・DLL・設定を .scratch/prd127/baseline-20261112/ に保存した。
- 変更前の設定 snapshot: 成功（.scratch/prd127/config-before.log）。
- 最初の tracer bullet: Factory→EstimateWithoutUpdate→Estimate/Learn の学習不変性。RED は未実装APIによるコンパイル失敗（.scratch/prd127/red-estimator.log）。最小実装後の Debug anet-core-test ビルドと [prd127] は成功、1 case / 66 assertions（.scratch/prd127/green-estimator-build.log・green-estimator-test.log）。whiten の通常構成だけを確認したもので、PRD全体の受入ではない。
- 基準Run: GridMaze（RND無効、8,704件）は正常終了。LunarLanderの実行途中で下記のPRDの食い違いを確認し、この会話が起動した比較ドライバと子Runnerだけを停止した。残る条件・after比較は未実施。初回ドライバは引数beforeの重複で失敗し、修正して runs-before-r1.log へ再実行した。最初の失敗ログも保持した。LunarLanderの途中のexecution.jsonは完了記録として使わず、再開時は新しい保存先または明示的な再開手順を使う。
- 2026-11-12: PRD D11 の「写しを持つActorでは乱数の書き換えも写しの中だけなので学習不変」という前提と、実装の共有乱数の使用が食い違うため中断した。nn_modules.cpp の DropPath は generator を渡さない bernoulli_ を呼び、DropoutModule は torch::nn::Dropout を呼ぶ。nn_impl.cpp の Network::Forward は RNG の隔離や eval 切り替えをせず RunForward を呼ぶ。RND の predictor は学習モードなので、推定器の重み・統計の Clone だけでは評価の乱数消費から学習を隔離できない。
- 直し方の選択: (A) G2/D11 の保証を乱数を使う層を持たない構成に限定する（出荷設定はこの条件を満たす）。(B) 乱数の分離も設計する。学習と評価が並行するため、共有RNGの状態を単純に保存復元する方法は採らず、専用generator等の設計が必要で、作業範囲が広がる。別の案として写しだけevalへ替えるのも学習モードの採点から意味が変わるため、無断では選ばない。
- 未実施: Clone、Actor同期、step診断の3鍵、pane開閉、出荷列、残りの単体・結合・GUIテスト、受入比較、設計文書追随、自己点検の完了、独立レビュー。現在の変更は途中の実装である。PRDを直して方針が決まったら本計画の続きから再開する。


## 再開
- 2026-11-12: 開発者が「保証を乱数層なしの構成に限定し、PRDを直して実装を続ける」を選んだ。G2・D11と§5.1の制約を改訂し、同じ計画から再開する。共有RNGの隔離やevalへの切替は追加しない。写しを持たないActorでは、内部統計を更新する層も引き続き保証外である。
- 未決事項監査: この選択によりブロッカーは0。出荷設定・受入テストは乱数層を持たず、既存のTDD順序とAPIの計画は変わらない。
- 改訂PRDの独立レビュー: OK（1往復）。確認事項・範囲外の検出なし。原文: C:\dev\anet-lab\.scratch\prd127\review-20261112-2340-prd.md。

- step診断はDefaultDQNの非公開IntrinsicStepDiagnosticsProviderで既存DQN部品を包み、即時の値だけを合成する。共通DqnAgentStepDiagnosticsProviderに内的報酬の責務を混ぜず、既存の系列窓とpending/completedをそのまま委譲するため、PRD §6.4の許容する構成の調整として固定した。IntrinsicSnapshotはActorへ弱い参照で接続し、診断部品が所有する。
- Cloneはdevice内でNetworkを複製し、Tensorを含む統計とoptimizerは所有状態を深く複製する。構築時のCPU既定乱数の消費を持ち越さないこともテストする。

- Traceの要求・Actor・Suspend/Resumeを通すテストはGREEN（4 cases / 138 assertions）。その後、未要求・未成立の診断で毎step NaN Tensorを3つ確保していた処理を、既知の鍵とNaNの読み出しへ置き換える。Cloneが繰り返し呼ぶNormalizerのshape初期化通知は開発用診断なのでscaler_impl.cppの該当1行をDEBUGへ移し、同期ごとのINFO反復を避ける。方式・中心化・clip・warm-upの手計算と次の1件のEstimateとの一致も追加で確認する。
- 2026-11-13: 改訂後の実装を継続。Trace生成器までの公開経路、3方式の学習不変性、Clone、採点式（中心化・clip・warm-up・次の1件との倍率一致）、同期・周期同期・休止・再開・寿命のテストがGREEN。直近は6 cases / 516 assertions（.scratch/prd127/lifetime-r2-test.log）。lifetime初回の失敗はテスト入力の2回目のepisode_startを誤ってtrueにしたためで、falseへ修正した。
- 比較ドライバの再実行対応: 中断前と再採取Runの接尾辞が重なり選択が曖昧になったため、試行ごとに一意のRun名へ変更した。途中の記録は削除せず--retry-incompleteで保存先内の別名へ退避する。通常RNDの出荷learn_batch_size=32を維持するため、有効2条件は保存済みLunarLanderの既定lane数256を使う（従来の短い検証用8 lane上書きだと現行契約違反になる）。RND無効4条件は従来どおり8 laneである。
- 最新の基準採取は .scratch/prd127/runs-before-r3.log。GridMazeとDropMergeは先行の正常終了分を保持し、LunarLanderとAtariは一意名で正常終了を確認した。通常RNDとTaiga式を採取中。失敗・中断した先行試行は受入に数えない。
- 文書は020・100・140・160・200・212の日英の該当節を更新中。英訳の既存translated-fromは全文の再確認をしていないためハッシュだけ更新せず保持し、今回の確認範囲は変更した節に限る。全体を再翻訳したとは報告しない。
- 文字コードの点検で、今回編集した既存C++8ファイルのBOMが消えていたことを検出した。コード編集を終えた時点でHEADのBOM有無へ戻してから最終Debugビルドを行う。並行変更のあるファイルのBOMは対象外にする。

- 2026-11-13: ロックの待ち方を含む7 cases / 540 assertionsがGREEN。さらに3方式・state/next_state・done/maskの24組み合わせで、Traceのボーナスと実際のLearnerの1step Replay収益を照合し、8 cases / 639 assertionsがGREEN（.scratch/prd127/bonus-r3-test.log）。最初の報酬テストはReplayの最新行まで本数に数えたため失敗し、受入に要るprobe行の存在とその報酬だけの確認へ絞った。
- 受入5・10の観測手段の調整: DefaultDQNAgentの内部Learnerは非公開で注入の口が無い。テスト専用の差し替えAPIを作らず、n-step=falseの公開ReplayBuffer::TARGET_RETURNをAgent::GetTensorから読む。これは各遷移でLearnerが受け取った報酬そのものであり、非終端probeは後続の終端を保存してから読む。受入の値と比較範囲は維持し、PRDの方法の記述も明確化して別にレビューする。
- 2026-11-13: 受入5・10の観測方法を明記したPRDの再レビューはOK（通算2回、差し戻し0）。原文: C:\dev\anet-lab\.scratch\prd127\review-20261113-0020-prd.md。

- 2026-11-13: Traceを各step挟む12構成（統計3方式×Actor複製有無×state/next_state）の学習不変性までGREEN、9 cases / 2,262 assertions（.scratch/prd127/invariance-r2-test.log）。報酬と全13集約診断に加え、公開checkpointから復元した推定器の継続Estimate/Learn、補助学習credit・RNG・回数を照合した。最初の不一致はTorch archive全体のバイト比較が内容の同一性を表さなかったためで、値と学習継続の比較へ直した。CPU/CUDAのCloneがCPUと実行deviceの既定RNG状態を変えないことも確認した。
- GUIの初回ビルドは並列コンパイルでメモリが逼迫したため対象ビルドのみ停止し、--parallel 2で再開した。比較Runと別の実験は停止していない。
- 日英6対は、今回の変更部分と既存translated-from以後の日本語差分を照合した。020の固定列、160の固定列とINDEX位置番号の説明に英訳漏れがあり、追随した。見出し階層・表行・コードブロック数・リンク先の存在を確認して、翻訳元blobを現在の日本語へ更新した。上の「既存blobを保持」は作業途中の記録で、最終状態ではない。

- 2026-11-13: 基準Run6条件がすべて正常終了。通常RNDは927.2秒、Taiga式は1,599.4秒。最新の実行記録はbaseline-20261112/before_<case>/execution.jsonで、失敗した先行試行は別名で保持している。
- 追加受入の準備エラーを修正中。DQN診断に渡すActionInfoはActor::MakeActionの結果へ、出荷列のParseにはdiscrete ActionSpecを渡し、ゼロ観測分散のcheckpointもscore_countと統計件数を揃えた。GUIのOnInitは明示tagのホスト初期化判定に追加し、Envの二重登録を避ける。これらの初回失敗は実装のREDとは数えない。

- Cloneが毎回呼ぶNetworkBuilderのinput shape通知も、Normalizerと同じ理由でnn_impl.cppの該当1行をINFOからDEBUGへ移す。学習・設定の契約は変えず、定期同期でshapeログを繰り返さない。
- pane初期表示のREDを確認（gui-red-r3の転送stdout、resume/syncが0であること）。同じ試行の終了時にはMetricsLoggerが空の状態でFlushされるSIGSEGVも発生し、cdbでGUI fixtureとログ終了の経路を調査した。最終ビルドではfixtureが実アプリのMetricsLoggerをResetしない変更を含めて再確認する。

- 自己点検でoverflow時に倍率の差があると誤認し、追加テストを作った。実コードは既にisnanを使ってNaNだけを1に戻しており、正当なEMA checkpointで∞を作るテストは修正なしに4 assertions成功した（overflow-red-test.log）。これは不具合でもREDでもない。ユーザーへの不具合の説明を撤回し、追加のコード変更は行わない。
- 費用測定の環境: Nsight Systems 2026.5.1 CLIを特定し、スキルが検出した不足依存duckdb・pyarrow・pandas等を指定Pythonへ導入した（.scratch/prd127/nsys-dependencies.log）。ただし別のAtari実験が25.4GBのprivate memoryを使い、空き物理メモリは約2GBで、今回のDebugコンパイルも強いメモリ圧力を受けている。Atari実時間優先の追加Runによる安定した費用測定は行わず、§7-14とU1を残件とする。ほかの実験は停止しない。画面はCPU・Replay容量8192のLunarLanderで別に確かめる。

- 公開ヘッダを変える理由: 推定器の利用側が読むだけの採点・複製を選べるようintrinsic.hppへ公開し、Agent・Actor・metrics・GUIの間の要求と同期はrl.hppの汎用契約で運ぶ。RND型へのdowncastや取り出し口は足さない。既存動詞の参照はnn::Network::Clone（nn_impl.cpp）、RewardScaler::ScaleWithoutUpdate（scaler_impl.cpp）、既存DQNのCreateStepDiagnosticsProvider / Observe（dqn_based_agent.cpp）である。
- 2026-11-13: 追加受入は14 cases / 2,303 assertionsがすべて成功（.scratch/prd127/green-acceptance-before-overflow.log）。この実行体には後から追加したoverflowテストがまだ無いので、境界値の受入とは数えない。テスト名一覧の--list-test-names-onlyはこのホストで非対応だったため、完了ログの件数を使う。
- 設定の既定キー検査は成功（Assignments 33,341、audited 4,039、errors 0。config-default-leaves.log）。既定7環境のafter snapshotは1 case / 15 assertions成功（config-after.log）。3列だけの差分比較は検証中。
- 機械自己点検: 既存ファイルのBOM・LF、追加行のヘッダusing namespace・production dynamic_cast・TODO形式は成功。重複リテラルの残りは設定キーだけ。旧API検索は一度Run出力まで範囲を広げたため停止し、core/anet-core・apps/runner/src・docs/design・CONTEXT.mdのcpp/hpp/mdだけへ絞り直し、旧CreateStepDiagnosticsProviderシグネチャは0件を確認した。

- 2026-11-13: Debug全体とGUIターゲットのビルド成功（green-full-build.log）。pane開閉の再試験はexit0、JUnitで13 assertions・failures0・errors0を確認（gui-green.xml）。最初のSIGSEGVはfixtureが実アプリのMetricsLoggerをResetしていたためで、fixtureの所有範囲を分けた変更を含む最終ビルドでは再現しない。
- 設定の全キー比較も成功。既定7環境は着手前と完全一致し、Atari・GridMaze・LunarLanderのrun.@rnd選択で増えるのは600・610・620のgolden3列だけ（config-comparison.log）。

- 2026-11-13: 最終の回帰は108 cases / 5,658 assertions成功（core-regression.log）。PRD127全15 cases（overflowを含む）と、PRD116・123、step診断・replay・no-wait、metricsをOR指定した。出荷設定スイートも16 cases / 2,648 assertions成功（shipped-config.log）。
- 変更後Runも6条件すべて正常終了し、時刻・資源の群だけを除いた全scalar checksumとmetrics.scalar.defsが変更前と一致（compare-r1.log、6件PASS）。通常RNDは984.2秒、Taiga式は2,072.8秒。これらの実時間は並行実験・ビルド等の影響があるため、Traceの費用測定には使わない。
- 画面操作は一度Computer Useのアプリ承認でタイムアウトしたが、開発者が承認して続行した。専用CPU LunarLanderのrun_20261113-011349_prd127_gui_rndでPanelツールバーから開き、prio_pctの右のrnd_err・rnd_norm・rnd_bonusを確認。約78.2秒閉じて学習・評価を続け、Evalを一時停止してView > Evaluation Traceから再表示した。空の表でStepを1回押した最初の行step67780に0.0005216 / -0.2795 / 0が入り、後続行step67781を作ってからstep67780をTraceで選んでも同じ3値だった。RND非選択のrun_20261113-082246_prd127_gui_plainは右端がprio_pctで、3列が無いことを確認した。確認後は専用Runを通常終了した。これは§7-13の実UI操作の確認であり、Atariの費用測定ではない。


## 最終自己点検
- docs/quality/impl.mdの全項目を適用範囲に従って確認した。計画・公開APIの理由・既存動詞・所有権・初期化順・局所的な検証の責任は本計画とPRDへ対応する。公開ヘッダの変更と実装の置き場は計画内、ResourceはAgent、診断の写しはProviderの所有、Actorからは弱い参照で循環を作らない。
- 契約の移行は現用の全呼び出しとoverrideへ反映し、旧シグネチャ・互換overloadは無い。汎用層へRNDの鍵・型・分岐は持ち込まず、個別の鍵は推定器FactoryとDefaultDQNから供給する。
- 読むだけの採点・Clone・Observe・Actor同期にprofiler範囲がある。同期の転送はlaneループの外。頻度と1回の見込みは前提に記載した。UIから学習ロックを取る例外はPRD D4と212に明記済みである。
- GetScalarの未知鍵nullopt・既知未成立NaNを確認。新しい非有限誤差WARNはFail-Fast表の「構築後に表示の一部だけ欠ける異常」で、学習側の既存ERRORは変えない。正常な専用LunarLander Runではread-only prediction_errorのWARNは出ていない。shapeは開発用DEBUGにした。
- 機械検査のBOM・LF・追加行のusing namespace / production dynamic_cast / TODO・旧API・重複リテラル・翻訳元blobを確認。新規関数はstatic、補助型は無名namespace。設定・テスト・文書・検証資材・課題の対応内容を含む今回の変更だけをレビューへ渡す。
- 日英6対の見出し・表・コードブロック・リンクと翻訳元を照合済み。ユーザーガイドは操作とNOTEへ分けた。固有の検証資材はtestdata/prd127に置き140のテスト章から参照する。skill・AGENTS・共有プロセスの変更は無い。
- 受入§7-1〜12・15は自動テスト・設定golden・6条件Run比較に対応し、§7-13は実UI操作で確認した。§7-14とU1だけは未実施・未決として残す。既存Trace GUIの回帰も17 cases / 184 assertions、failures0・errors0（gui-regression-all.xml）で成功した。


## 実行コマンドの記録

作業ディレクトリは `C:\dev\anet-lab`。以下は実行済みのコマンドと結果であり、既存の保存先へ再実行する手順ではない。標準出力と標準エラーは表のログへ保存し、終わった時点で終了値と末尾だけを返した。長いビルド・Runはexecのsessionをwrite_stdinで追い、無関係な実験を停止していない。

| 検証 | 実行したコマンド | 結果 |
|---|---|---|
| 最終Debug全体とGUIターゲット | `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2 && cmake --build --preset x64-Debug --target AnetRLRunner-gui-test --parallel 2' *> .scratch/prd127/green-full-build.log` | exit 0 |
| 追加したoverflowテストのビルド | `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 2' *> .scratch/prd127/overflow-red-build.log` | exit 0。ログ名は試行時のもの |
| overflowの既存挙動 | `& core/anet-core/bin/Debug/anet-core-test.exe '[prd127_overflow]' *> .scratch/prd127/overflow-red-test.log` | 1 case / 4 assertions、exit 0。実装変更なしに成功 |
| PRD127全ケースと関連回帰 | `& core/anet-core/bin/Debug/anet-core-test.exe '[prd127],[prd116],[prd123],[step_diagnostics],[diagnostic_replay],[diagnostic_no_wait],[metrics]' *> .scratch/prd127/core-regression.log` | 108 cases / 5,658 assertions、exit 0 |
| 出荷設定 | `& core/anet-core/bin/Debug/anet-core-test.exe '[shipped_config]' *> .scratch/prd127/shipped-config.log` | 16 cases / 2,648 assertions、exit 0 |
| pane開閉 | `& apps/runner/bin/Debug/AnetRLRunner-gui-test.exe '[prd127]' --reporter junit --out C:/dev/anet-lab/.scratch/prd127/gui-green.xml *> .scratch/prd127/gui-green-junit.log` | 1 case / 13 assertions、exit 0 |
| 設定の既定キー | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd072/check_default_leaves.py *> .scratch/prd127/config-default-leaves.log` | audited 4,039、errors 0、exit 0 |
| 全7環境のafter採取 | `$env:ANET_PRD093_CONFIG_ROOT=(Resolve-Path apps/runner/config).Path; $env:ANET_PRD093_OUTPUT=Join-Path (Get-Location).Path '.scratch/prd127/baseline-20261112/resolved-after'; & core/anet-core/bin/Debug/anet-core-test.exe '[prd093-config-snapshot]' *> .scratch/prd127/config-after.log` | 1 case / 15 assertions、exit 0 |
| 全キーと3列goldenの照合 | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd127/verify_configuration.py *> .scratch/prd127/config-comparison.log` | 既定7環境は完全一致、選択時だけgolden3列の追加、exit 0 |
| 変更前の再採取4条件 | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd127/run_validation.py before --case LunarLander Atari collect feedback --retry-incomplete *> .scratch/prd127/runs-before-r3.log` | 4条件とも正常終了。先行のGridMaze・DropMergeの正常終了を合わせて6条件 |
| 変更後Taiga式 | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd127/run_validation.py after --label after-r1 --case feedback *> .scratch/prd127/runs-after-feedback.log` | 正常終了、exit 0 |
| 変更後の残る5条件 | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd127/run_validation.py after --label after-r1 --case GridMaze LunarLander Atari DropMerge collect *> .scratch/prd127/runs-after-other.log` | 5条件とも正常終了、exit 0 |
| before/after比較 | `& ./.venv/Scripts/python.exe core/anet-core/testdata/prd127/run_validation.py compare --label after-r1 *> .scratch/prd127/compare-r1.log` | 6条件の全scalar checksumとmetrics.scalar.defsが一致、exit 0 |

既存Trace GUI回帰は、testファイルから隠しタグ以外の15種類のTraceタグを列挙し、Python `subprocess.run` で次のargvを実行した。単なる `[trace_ui]` は接頭辞ではなく完全一致なので、最初の2ケースだけの試行で済ませていない。標準出力・標準エラーは `gui-regression-all.log`、JUnitは `gui-regression-all.xml` に保存した。結果は17 cases / 184 assertions、exit 0である。

```json
[
  "apps/runner/bin/Debug/AnetRLRunner-gui-test.exe",
  "[trace_ui_close],[trace_ui_controls],[trace_ui_dump],[trace_ui_font],[trace_ui_frozen],[trace_ui_header],[trace_ui_late],[trace_ui_manual],[trace_ui_minimum],[trace_ui_palette],[trace_ui_sort],[trace_ui_sort_icon],[trace_ui_step_pct],[trace_ui_toggle],[trace_ui_width]",
  "--reporter",
  "junit",
  "--out",
  "C:\\dev\\anet-lab\\.scratch\\prd127\\gui-regression-all.xml"
]
```

全6条件のafter execution.jsonに保存したbinary_sha256が、最終 `apps/runner/bin/Debug/AnetRLRunner.exe` のSHA256 `7c8b36a0c1d402a03f620e09d838f3cb4449f46e69b488f60bbc4ac2c0ac7186` と一致することも確認した。


## 完了報告
- 2026-11-13: 実装計画・実装の独立レビューはOK 2件、差し戻し0件。レビュー中に実行コマンドの記録を補い、計画全体を読み直して確認された。新たな範囲外の検出と開発者への確認事項は無い。
- 原文: C:/dev/anet-lab/.scratch/prd127/review-20261113-0836-impl.md
- PRD改訂のレビューは2回ともOK。実装・自動テスト・全キー比較・6条件Run比較・実UI確認・自己点検を終えた。学習不変性の保証範囲は改訂G2/D11に従い、乱数を使う層を持たない構成に限る。
- 残件は§7-14の費用測定とU1の判断。課題issue-20261107-1226e5のai_judgeを残件有り、ai_judgedを2026-11-13へ更新し、statusは対応中のままにした。解決・監理の項目は変更していない。
- 工程末の新規起票候補はPRD §8の3件（DropMerge用RNDプロファイル、未知Trace鍵で途中開き時にRun停止、EvalPanelのUIから学習ロックを取る同期）。重複する既存の同一課題は確認されず、候補として示すだけで起票しない。既存へ追記・この場で直す・見送り推奨の新たな候補は無い。
- stage・commit・pushは行っていない。並行作業と既存indexを保持した。

- 最終のmapsはexit0、INDEXとRunner/DQNの派生区画だけを更新し、unresolved_prdsは空。専用GUIの2プロセスは通常終了を確認した。
- 完了時に使用枠を再取得し、週枠の使用85%・残り15%を確認した。着手時は残り22%だったが、これは他のチャットも含む共有枠で、本作業だけの消費率を表さない。無料リセットは使用していない。


## 工程後の起票
- 2026-11-13: 開発者が候補3だけの起票を選択した。[issue-20261113-d38960](../management/issues/issue-20261113-d38960.md) に、EvalPanelのUIスレッドがAgentの学習ロックを待つ同期全体の見直しを記録した。候補1・2は起票していない。PRD127の費用測定とU1は従来どおり既存課題に残す。

## 工程後の列配置の改訂（2026-11-13）

- 開発者が`rnd_bonus`を`reward`の直後へ移す配置を選んだ。最初の240番はTempoRLの`decision`と重複するため、問い合わせへの回答で235番を選択した。240 `decision`・250 `skip`は維持し、誤差2列も600 `rnd_err`・610 `rnd_norm`のままとする。PRD D12に、一の位を空ける番号の原則の例外として理由を記録した。
- Atari・GridMaze・LunarLanderの設定の該当1行、PRD D12・§5.5・受入12/13、020/212の日英ガイド、golden、設定比較、合成Traceテストの並びを合わせた。LunarLanderの`run.$ = @rnd`の選択や開発者によるブロックの移動は保持する。実装コードと採点の式は変更していない。
- 設定比較は3環境それぞれの空の選択・RND単独・TempoRL単独・両方の順序での併用、計15通りに広げた。同じ選択の着手前設定と全キー・文字列値を比較し、RNDを選んだ場合の235・600・610の3列の追加だけを許す。235が230の直後に並ぶことも確認する。採取済みの既定7環境の比較記録は再検査するが、現在のenvファイルの既定選択を同じと主張する検査には使わない。
- 過去の実画面確認と全6条件Run比較は、その時点の記録として保持する。新しい列位置の実画面確認と長い学習Runの再実行は今回行わず、列位置・上書きは設定解決と既存の行生成のテストで確かめる。費用測定とU1は引き続き残件である。
- Debugビルド・関連テスト・設定比較・既定キーの検査のログは`.scratch/prd127/columns-235-*.log`へ保存する。標準出力と標準エラーをまとめて保存し、応答には終了コードと末尾の要約を返す。既存のステージは変更せず、今回の修正は未ステージにする。

実行コマンド（リポジトリルート、PowerShell）:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test --parallel 2' *> .scratch/prd127/columns-235-build.log
```

```bash
& core/anet-core/bin/Debug/anet-core-test.exe '[prd127],[shipped_config]' *> .scratch/prd127/columns-235-tests.log
```

```bash
& .\.venv\Scripts\python.exe core/anet-core/testdata/prd127/verify_configuration.py *> .scratch/prd127/columns-235-configuration.log
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py *> .scratch/prd127/columns-235-default-leaves.log
```

結果:

- Debugの`anet-core-test`ビルドはexit0。`[prd127],[shipped_config]`は30 cases / 4,950 assertionsが成功（exit0）。
- `verify_configuration.py`は15通りすべて成功（exit0）。RND選択時の差分はgoldenの235・600・610の追加だけで、既存列の上書き、他キーの変更・削除は無い。`rnd_bonus`は`reward`の直後である。
- `check_default_leaves.py`はAssignments 33,344 / audited 4,041 / errors 0（exit0）。
- PRD品質確認観点と実装品質確認観点の全項目を適用判定して自己点検した。該当する改訂記録・決定の理由・受入との対応・移行・検証記録を確認。旧620番は現用設定・テスト・設計文書・PRDに残らず、過去の検証記録だけに保持した。BOM/LF、`git diff --check`、日英の見出し階層・表の行数・コードフェンス・リンク・翻訳元情報も成功した。
