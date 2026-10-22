# 指摘台帳(派生ビュー)

生成 2026-10-22。正本は findings.jsonl。この文書は毎回再生成される。

## 概況

| 状態 | 件数 |
|---|---|
| fixed | 1 |
| open | 26 |

open の優先度別: P0 1, P1 4, P2 16, P3 5

open の観点別: architecture 0, backlog 0, defect 10, drift 4, process 0, refactor 10, test-gap 2

## 最近の変化(14 日)

| id | 更新 | 状態 | 優先度 | 観点 | 場所 | タイトル |
|---|---|---|---|---|---|---|
| F-0006 | 2026-10-22 | open | P0 | defect | `core/anet-core/src/rl.cpp`::Notifier | Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る |
| F-0007 | 2026-10-22 | open | P1 | defect | `core/envs/dropmerge1/src/DropMergeEnv.cpp`::DropMergeEnv::Step | Directモードで列0への落下にもNOOPペナルティが加算される |
| F-0008 | 2026-10-22 | open | P1 | defect | `core/anet-core/src/metrics_logger.cpp`::VideoLogger::WriteFrame | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む |
| F-0009 | 2026-10-22 | open | P2 | defect | `core/anet-core/src/trainer.cpp`::PipelineTrainRunner::DoStep | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| F-0010 | 2026-10-22 | open | P3 | defect | `core/anet-core/include/anet/rl.hpp`::ActionSpec::GetShape | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| F-0011 | 2026-10-22 | open | P2 | defect | `core/anet-core/src/heat_map.cpp`::TimeHeatMap::RenderRaw | wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する） |
| F-0012 | 2026-10-22 | open | P1 | refactor | `core/anet-core/include/anet/rl.hpp`::TrainObserver/LearnObserver/EpisodeEndObserver/SessionEndObserver | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| F-0013 | 2026-10-22 | open | P2 | refactor | `core/anet-core/include/anet/agent.hpp`::AgentRepository | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う |
| F-0014 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/dqn_based_agent.cpp`::TDLearner/QRLearner/IQNLearner::UpdateFromSamples | TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている |
| F-0015 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/trainer.cpp`::SerialTrainRunner::DoStep/PipelineTrainRunner::DoStep | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| F-0016 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/heat_map.cpp`::HeatMap::RenderRaw/TimeHeatMap::RenderRaw | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| F-0017 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/observers.cpp`::ObserverFactory::ObserverFactory | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| F-0018 | 2026-10-22 | open | P2 | refactor | `apps/runner/src/QValuePanel.cpp`::QValuePanel::Update | QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う |
| F-0019 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/dqn_based_agent.hpp`::BatchUpdateResult::GetScalar | BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖 |
| F-0020 | 2026-10-22 | open | P2 | refactor | `core/anet-core/src/gui.cpp`::Panel::OnKey | Panel::OnKeyがproductionコードでdynamic_castを3回使う |
| F-0021 | 2026-10-22 | open | P3 | refactor | `core/anet-core/src/dqn_based_agent.hpp` | dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い |
| F-0022 | 2026-10-22 | open | P3 | drift | `apps/runner/config/metrics_scalar.txt`::metrics.scalar 記法のコメント | metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落） |
| F-0023 | 2026-10-22 | open | P3 | drift | `apps/runner/config/nn.txt`::Supported Module Types のコメント | nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている |
| F-0024 | 2026-10-22 | open | P2 | drift | `apps/runner/config/nn.txt`::net.body.@ResNet/net.body.@MLP_RES | nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる |
| F-0025 | 2026-10-22 | open | P2 | drift | `core/anet-core/src/trainer.cpp`::RunnerFactory::CreateMainRunner | train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う |
| F-0026 | 2026-10-22 | open | P2 | test-gap | `core/anet-core/src/scaler_impl.cpp`::RewardScalerFactory::CreateRewardScaler/RunningStdSingleTensorNormalizer | 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効） |
| F-0027 | 2026-10-22 | open | P3 | test-gap | `core/anet-core/src/muzero_rb.cpp`::MuZeroReplayBuffer | MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い |

## open(優先度順)

| id | 優先度 | 確度 | 観点 | 分類 | 場所 | タイトル | 初出 | 最終確認 |
|---|---|---|---|---|---|---|---|---|
| F-0006 | P0 | high | defect | 並行性 | `core/anet-core/src/rl.cpp`::Notifier | Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る | 2026-07-04 | 2026-10-22 |
| F-0003 | P1 | high | defect | 境界 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::PlotlyController.renderBySelection | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない | 2026-09-23 | 2026-09-23 |
| F-0007 | P1 | high | defect | 境界 | `core/envs/dropmerge1/src/DropMergeEnv.cpp`::DropMergeEnv::Step | Directモードで列0への落下にもNOOPペナルティが加算される | 2026-07-04 | 2026-10-22 |
| F-0008 | P1 | high | defect | 境界 | `core/anet-core/src/metrics_logger.cpp`::VideoLogger::WriteFrame | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む | 2026-07-04 | 2026-10-22 |
| F-0012 | P1 | high | refactor | 重複の生成源（抽象の欠落） | `core/anet-core/include/anet/rl.hpp`::TrainObserver/LearnObserver/EpisodeEndObserver/SessionEndObserver | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している | 2026-07-04 | 2026-10-22 |
| F-0002 | P2 | high | defect | 境界 | `apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java`::LodPageCache.find | 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる | 2026-09-23 | 2026-09-23 |
| F-0004 | P2 | high | defect | 並行 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.onReload / onWorkspaceChanged | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる | 2026-09-23 | 2026-09-23 |
| F-0005 | P2 | high | defect | 並行 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.init / onToggleScreenshot | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない | 2026-09-23 | 2026-09-23 |
| F-0011 | P2 | high | defect | 未定義動作 | `core/anet-core/src/heat_map.cpp`::TimeHeatMap::RenderRaw | wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する） | 2026-07-04 | 2026-10-22 |
| F-0013 | P2 | high | refactor | 重複の生成源（抽象の欠落） | `core/anet-core/include/anet/agent.hpp`::AgentRepository | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う | 2026-07-04 | 2026-10-22 |
| F-0014 | P2 | high | refactor | 既存の重複・不要コード | `core/anet-core/src/dqn_based_agent.cpp`::TDLearner/QRLearner/IQNLearner::UpdateFromSamples | TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている | 2026-07-04 | 2026-10-22 |
| F-0015 | P2 | high | refactor | 既存の重複・不要コード | `core/anet-core/src/trainer.cpp`::SerialTrainRunner::DoStep/PipelineTrainRunner::DoStep | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている | 2026-07-04 | 2026-10-22 |
| F-0016 | P2 | high | refactor | 既存の重複・不要コード | `core/anet-core/src/heat_map.cpp`::HeatMap::RenderRaw/TimeHeatMap::RenderRaw | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている | 2026-07-04 | 2026-10-22 |
| F-0017 | P2 | high | refactor | 肥大化 | `core/anet-core/src/observers.cpp`::ObserverFactory::ObserverFactory | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む | 2026-07-04 | 2026-10-22 |
| F-0018 | P2 | high | refactor | 肥大化 | `apps/runner/src/QValuePanel.cpp`::QValuePanel::Update | QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う | 2026-07-04 | 2026-10-22 |
| F-0019 | P2 | high | refactor | 肥大化 | `core/anet-core/src/dqn_based_agent.hpp`::BatchUpdateResult::GetScalar | BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖 | 2026-07-04 | 2026-10-22 |
| F-0020 | P2 | high | refactor | 規約違反・不統一 | `core/anet-core/src/gui.cpp`::Panel::OnKey | Panel::OnKeyがproductionコードでdynamic_castを3回使う | 2026-07-04 | 2026-10-22 |
| F-0024 | P2 | high | drift | サンプル設定 | `apps/runner/config/nn.txt`::net.body.@ResNet/net.body.@MLP_RES | nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる | 2026-07-04 | 2026-10-22 |
| F-0025 | P2 | high | drift | fail-fastの方針 | `core/anet-core/src/trainer.cpp`::RunnerFactory::CreateMainRunner | train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う | 2026-07-04 | 2026-10-22 |
| F-0026 | P2 | high | test-gap | テストの空白 | `core/anet-core/src/scaler_impl.cpp`::RewardScalerFactory::CreateRewardScaler/RunningStdSingleTensorNormalizer | 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効） | 2026-07-04 | 2026-10-22 |
| F-0009 | P2 | med | defect | 整合性 | `core/anet-core/src/trainer.cpp`::PipelineTrainRunner::DoStep | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる | 2026-07-04 | 2026-10-22 |
| F-0021 | P3 | high | refactor | 規約違反・不統一 | `core/anet-core/src/dqn_based_agent.hpp` | dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い | 2026-07-04 | 2026-10-22 |
| F-0022 | P3 | high | drift | 設定のコメント | `apps/runner/config/metrics_scalar.txt`::metrics.scalar 記法のコメント | metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落） | 2026-07-04 | 2026-10-22 |
| F-0023 | P3 | high | drift | 設定のコメント | `apps/runner/config/nn.txt`::Supported Module Types のコメント | nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている | 2026-07-04 | 2026-10-22 |
| F-0027 | P3 | high | test-gap | テストの空白 | `core/anet-core/src/muzero_rb.cpp`::MuZeroReplayBuffer | MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い | 2026-07-04 | 2026-10-22 |
| F-0010 | P3 | med | defect | 境界 | `core/anet-core/include/anet/rl.hpp`::ActionSpec::GetShape | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す | 2026-07-04 | 2026-10-22 |

### 詳細

#### F-0006 P0 Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る

- 観点/分類: defect / 並行性
- 場所: `core/anet-core/src/rl.cpp`::Notifier
- 要約: Notifierの4種のobserver配列（train・learn・episode_end・session_end）は排他を持たず、Attach・Detach・Notifyが同じvectorを直接読み書きする。学習スレッドがNotifyで走査している間に、GUIスレッドがパネルの開閉でAttach・Detachを呼ぶ。
- シナリオ: 学習の実行中にConv2dPanelやQValuePanelを開く → GUIスレッドのAttachがtrain_observers_へpush_backし、容量を超えると再確保する → 学習スレッドのNotifyが持つイテレータは解放済みの領域を指し、observerの呼び出しで落ちるか、壊れたshared_ptrをコピーする。パネルを閉じたときのDetachのeraseでも同じことが起きる。
- 対処: Attach・Detach・Notifyを同じmutexで守り、Notifyはロック下でobserver配列のsnapshotを取ってロック外で呼ぶ。
- 根拠: core/anet-core/src/rl.cpp:979 Attachはロックを取らずにtrain_observers_へpush_backする; core/anet-core/src/rl.cpp:1003 Detachもロックを取らずにeraseする; core/anet-core/src/rl.cpp:1142 Notifyはロックを取らずにtrain_observers_を走査し、各observerを呼ぶ; apps/runner/src/Conv2dPanel.cpp:85 パネルを開いた時点で、学習中のNotifierへAttachScopedする; apps/runner/src/Conv2dPanel.cpp:110 パネルを閉じるとNotifierからDetachする; apps/runner/src/HeatMapPanel.cpp:484 パネルの初期化でNotifierへAttachする

#### F-0003 P1 window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない

- 観点/分類: defect / 境界
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::PlotlyController.renderBySelection
- 要約: serverは要求範囲内に点が無い系列をempty/pendingで返す。clientはok以外の系列を描かず、全Runがok以外のtagではgraph block自体を出さない。explicit viewportは残り、空windowが範囲を覆うので再取得も起きないため、そのgraphは操作手段ごと消えたままになる。元の優先度P2を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: LOD表示中のtagで、点の無い区間（長いepisodeによる間隔など）の中へ、その幅の1/3未満でbox zoomする。または末尾付近を拡大して右へ2〜3画面panし、window[from-w,to+w]が最終点を越える → 全Runがempty → そのtagのgraphが消える。double clickする対象も無く、tagの再選択やReloadでも同じviewportで再びemptyになる。floatingのReset Viewで全tagのzoomを捨てるか、ページを再読込するしか戻す手段がない。
- 対処: explicit viewportを持つtagは、traceが0本でも空の軸だけのgraph blockを描く。またはserverが範囲外の直近点を両端に1点ずつ含める
- 根拠: apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java:241 ordinal範囲は閉区間[fromStep,toStep]内の点だけで、範囲外の隣接点は含めない; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsQueryPlanner.java:86 範囲内が0点ならempty（取り込み中はpending）; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2082 empty/pendingの結果もwindowとして保存する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:951 availabilityがok以外の系列はtraceを作らない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:954 traceが0本のtagは、headerとLog/percentileボタンを含むgraph block自体を作らない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:412 空windowが要求範囲を覆うので、以後needsFetchはfalseのまま; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1976 tag単位のautorange復帰はgraph上のrelayout経由だけで、graphが無いと発火しない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2284 graph無しでviewportをautorangeへ戻す経路は、全tag一括のonResetViewだけ

#### F-0007 P1 Directモードで列0への落下にもNOOPペナルティが加算される

- 観点/分類: defect / 境界
- 場所: `core/envs/dropmerge1/src/DropMergeEnv.cpp`::DropMergeEnv::Step
- 要約: Stepは行動モードごとにis_noop_actionを決めているのに、末尾のNOOPペナルティだけは生のaction値を0と比べる。Directモードではaction 0は列0への落下なので、NOOPではない行動に罰が付く。
- シナリオ: action_mode=directで、noop_penaltyを0以外にする（コードの既定値は-0.001。DropMerge.txtは0.0で上書きしているので、今の既定の実行では休眠している）→ 列0に落とすたびにnoop_penaltyが報酬へ加算され、列0だけが不利な報酬で学習する。
- 対処: ペナルティの判定をis_noop_actionに揃える。
- 根拠: core/envs/dropmerge1/src/DropMergeEnv.cpp:1022 Directモードでは、どのactionもis_noop_action=falseとしている; core/envs/dropmerge1/src/DropMergeEnv.cpp:1262 NOOPペナルティはis_noop_actionではなく、生の値action==kActionNoop(0)で判定する; core/envs/dropmerge1/src/DropMergeEnv.hpp:17 kActionNoopは0。Directモードでは0は列0への落下; core/envs/dropmerge1/src/DropMergeEnv.hpp:102 noop_penaltyのコードの既定値は-0.001; apps/runner/config/DropMerge.txt:1620 既定の設定はnoop_penalty ?= 0.0で上書きしている

#### F-0008 P1 VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む

- 観点/分類: defect / 境界
- 場所: `core/anet-core/src/metrics_logger.cpp`::VideoLogger::WriteFrame
- 要約: WriteFrameは構築時の幅・高さからバイト数を決め、渡されたwxImageのGetData()からその長さを読む。画像の幅・高さが構築時と一致するかを検証していない。
- シナリオ: 同じtagの動画で、2枚目以降の画像が最初の画像より小さい（描画サイズが変わったなど）→ 小さいバッファの範囲外を読んでffmpegへ書き込み、異常終了するかゴミのフレームになる。大きい場合は画像の先頭部分だけが書かれ、ffmpegは別の行の画素を1フレームとして並べる。
- 対処: WriteFrameで画像の幅・高さを構築時の値と比べ、違えばANET_SYSTEM_ERRORで止める。
- 根拠: core/anet-core/src/metrics_logger.cpp:773 tagの最初の画像の幅と高さでVideoLoggerを作る; core/anet-core/src/metrics_logger.cpp:801 以後は同じtagの画像を、サイズを確かめずにWriteFrameへ渡す; core/anet-core/src/metrics_logger.cpp:494 読み出し元は渡された画像のGetData(); core/anet-core/src/metrics_logger.cpp:495 書き込むバイト数は構築時のwidth_*height_*3で、渡された画像の大きさを見ない

#### F-0012 P1 イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している

- 観点/分類: refactor / 重複の生成源（抽象の欠落）
- 場所: `core/anet-core/include/anet/rl.hpp`::TrainObserver/LearnObserver/EpisodeEndObserver/SessionEndObserver
- 要約: イベントの種類を1つ足すたびに、Event構造体、Observer interface、RunnerScopedラッパ、NotifierのAttach・Detach・Notify、AttachScopedの分岐、MetricsLogの派生を同じ形で書き足す構造になっている。2026-07時点の3種から、SessionEndの追加で4種に増えた。
- シナリオ: 5種目のイベントを足すとき、AttachScopedテンプレートのif constexprの分岐を書き漏らす → その種類だけを実装したobserverはAttachScoped<T>で何も登録されず、エラーも出ない。Detachのoverloadや、RunnerScopedのShutdown転送の書き漏らしも同じく黙って効かない。
- 対処: イベント型をテンプレート引数にした単一のObserver・Notifier登録へ寄せる。PRDで設計を決めてから行い、hot pathのNotifyの性能を確かめる。
- 根拠: core/anet-core/include/anet/rl.hpp:975 UpdateEventを基底にするのはTrain・Learnだけで、EpisodeEndEvent（992）とSessionEndEvent（1001）は同じrunner・counts・agentを別に持つ; core/anet-core/include/anet/rl.hpp:1008 4つのObserver interface（1008・1017・1026・1035）がOnXxx以外同じ（Shutdown・WillBlockOnShutdown・ToString）; core/anet-core/include/anet/rl.hpp:1047 RunnerScopedのラッパを4種、同じ形で持つ（1047〜1101）; core/anet-core/include/anet/rl.hpp:1110 NotifierがAttach・Detach2種・Notifyを4組、16メソッド持つ; core/anet-core/include/anet/rl.hpp:1148 AttachScopedテンプレートがif constexprで4種に分岐し、1166〜1189に非テンプレート版も4つある; core/anet-core/src/rl.cpp:979 Attach・Detachの実装12本が同じ形（979〜1105）; core/anet-core/include/anet/observers.hpp:369 MetricsLogObserverもイベント別に4つの派生（369・380・391・404）

#### F-0002 P2 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる

- 観点/分類: defect / 境界
- 場所: `apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java`::LodPageCache.find
- 要約: LodPageCache.findは未完成page（各tagの末尾page）を保持せず、呼ばれるたびにpage全体を読み直す。一方でMetricsRangeProjector.loadBucketsは表示範囲のbucketごとにfindを呼ぶ。このため末尾pageに入るB個のbucketについて、B回×最大B行を読み、12本の配列もB回確保することになり、query時間がO(B^2)になる。結果は正しいが、cacheを無効にした場合（capacity=0、bucket単位SELECT）より遅い。
- シナリオ: 10万点のtagを全域表示する（maxPoints=8000）→ selectLevelはlevel 2の391 bucketを選ぶ → 永続化済みの390 bucketはすべて未完成のpage 0にあり、391回のfindがそれぞれ390行を読む。合計約15万行で、生の点数10万より多い。10 Run×20 tagを同時表示すると1 requestで約3,000万行になり、permit（既定2）を長く占有して他のタブが503 query_busyになる。Auto Reloadのforce再取得のたびに同じことが起きる。
- 対処: 未完成pageをquery単位でmemoするか、loadBucketsで範囲を一括SELECTしてからbucketを引く
- 根拠: apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:50 cache missのたびに、1 bucketのためにpage全体（最大1024行）をSELECTする; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:51 未完成pageは保持せず捨てるので、同じpageの次のbucketで再びloadPageが走る; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/LodPageCache.java:252 1024 bucket未満のpageは常に未完成扱い。つまり各tag・各levelの末尾pageは一度もcacheされない; apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRangeProjector.java:170 loadBucketsは選択したlevelの全bucketについて1件ずつpageCache.findを呼ぶ

#### F-0004 P2 DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる

- 観点/分類: defect / 並行
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.onReload / onWorkspaceChanged
- 要約: runs.jsonとmetrics.jsonのAbortControllerはDataFetcherにそれぞれ1枠しかなく、Reload・Auto Reload・workspace切替・viewport変更・Run/tagのclickが互いの要求をabortする。abortされたonReload/onWorkspaceChangedは、abortやsupersededを失敗として通知してしまう。さらにworkspace切替のinitial=true要求がAuto Reloadのinitial=false要求にabortされると、初期Run選択が行われない。元の優先度P3を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: (a) Reloadを押した後、metrics.jsonの応答前にRunやtagをclickする → _bumpQueryRevisionがReloadの要求をabort → 『Reload failed.』が出る。workspace切替後の描画待ちにRunをclickした場合も『Workspace switched, but data refresh failed.』が出る。(b) Auto Reload ONでworkspaceを切り替え、切替側のruns.jsonの応答待ちにtickが来る → 切替側のrefreshMetadata（initial=true）がabortされてToastが出る。tick側（initial=false）が新workspaceを反映するが、選択Runは空のまま『No selection.』になる。
- 対処: AbortError/supersededを失敗Toastの対象から外す。workspace切替の後処理中はAuto Reloadのmetadata取得を止めるか、initial要求を後発の要求へ引き継ぐ
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:243 fetchRunsは呼び出し元を問わず直前のruns.jsonをabortする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:280 fetchMetricsも直前のmetrics.jsonをabortする。_bumpQueryRevisionも同じ枠をabortする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2087 requestVisibleDataはabortも含めて例外を再送出する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2441 onReloadはAbortErrorや409 supersededでも無条件に『Reload failed.』を出す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1812 onWorkspaceChangedも無条件に『Workspace switched, but data refresh failed.』を出す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2519 _handleQueryErrorとinit（1702行）はAbortError/supersededを除外しており、扱いが一致しない; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1861 初期Run選択はinitial=trueのrefreshMetadataだけが行う; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2414 Auto Reloadのtickはinitial=falseでrefreshMetadataを呼ぶ

#### F-0005 P2 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない

- 観点/分類: defect / 並行
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.init / onToggleScreenshot
- 要約: META_LOADING中にScreenshotへ切り替えるとbodyはscreenshot-modeになるが、init完了時にmodeが無条件にNORMALかERRORへ上書きされ、画面とmodeが食い違う。scroll lockと取り込み進捗pollの停止はmode==SCREENSHOTで判定するため、画面はscreenshotのまま通常時の挙動になる。元の優先度P3を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: (a) runs.jsonが遅い起動中に×/⬅を押す → init完了でmode=NORMALだが画面はscreenshotのまま → scroll lockが外れてgraphのdragがzoom/panになり、止まるはずの進捗pollも続く。(b) 同じ操作の後にinitが失敗する → mode=ERROR、side panelは非表示、graphも無い → Reloadボタンもgraphのdouble click Reloadも使えず、onToggleScreenshotはERRORで何もせず戻る。F5以外で復帰できない。
- 対処: META_LOADING中はscreenshot切替を無効にする。またはinit完了時にscreenshot状態を保ったままmodeを決め、ERROR中も解除だけは許可する
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1695 initはMETA_LOADINGにしてからworkspaceとmetadataをawaitする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2446 onToggleScreenshotはERROR以外（META_LOADINGを含む）で切替を許し、ERROR中は解除も拒否する; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1699 init成功時は現在のmodeを見ずにNORMALへ上書きする; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:1704 init失敗時はERRORへ上書きし、screenshot-mode classは残る; apps/metrics-viewer/src/main/resources/static/metrics-viewer.css:645 screenshot中はReloadを含むside panelが非表示

#### F-0011 P2 wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する）

- 観点/分類: defect / 未定義動作
- 場所: `core/anet-core/src/heat_map.cpp`::TimeHeatMap::RenderRaw
- 要約: wxImage::SetDataは受け取った配列の所有権を取り、破棄時にfree()で解放する。heat_map.cppの4か所はnew[]で確保して渡すので、new[]とfreeの組み合わせになる。
- シナリオ: TimeHistogram・Histgram・SweepedHeatMapの画像を描画して破棄する → new[]で確保した配列をwxImageがfreeで解放する未定義動作になる。MSVCの既定のCRTでは表面化しにくいが、デバッグヒープやAddressSanitizerではalloc-dealloc mismatchとして検出され、異常終了する。
- 対処: HeatMap::RenderRawと同じくmallocで確保するか、wxImage(W, H)が確保したバッファへGetData()で書く。
- 根拠: core/anet-core/src/heat_map.cpp:416 空画像にnew unsigned char[3]をSetDataする; core/anet-core/src/heat_map.cpp:502 描画画像にnew unsigned char[W*H*3]をSetDataする; core/anet-core/src/heat_map.cpp:561 Histgram::RenderRawも同じくnew[]を渡す; core/anet-core/src/heat_map.cpp:819 SweepedHeatMap::RenderRawも同じくnew[]を渡す; core/anet-core/src/heat_map.cpp:195 HeatMap::RenderRawはmallocで確保して渡している（wxImageの要求どおり）

#### F-0013 P2 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う

- 観点/分類: refactor / 重複の生成源（抽象の欠落）
- 場所: `core/anet-core/include/anet/agent.hpp`::AgentRepository
- 要約: Agent・Env・View・NNモジュールの4つのRepositoryが、singleton・mutex・名前からfactoryへのmap・登録関数をそれぞれ持つ。名前（Register／Regist、GetAgentFactory／GetViewCreator／GetFactory）も、同じclass_idの重複登録の扱いも揃っていない。
- シナリオ: 同じclass_idのAgentFactoryを2つの.cppが登録する → Envなら起動時にfail-fastするが、Agent・View・NNモジュールは後に初期化された方で黙って上書きされ、静的初期化の順でどちらが使われるかが変わる。未登録名のエラー文や一覧の出し方を変えるにも、4か所を別々に直す。
- 対処: 名前付きfactoryのRepositoryをテンプレート1つにまとめ、重複登録はfail-fastに揃える。
- 根拠: core/anet-core/include/anet/agent.hpp:281 AgentRepository: singleton・mutex・unordered_map・Register・GetAgentFactory; core/anet-core/include/anet/env.hpp:285 EnvRepository: 同じ形で、登録の名前はRegist; core/anet-core/include/anet/gui.hpp:372 ViewRepository: 同じ形で、取り出しはGetViewCreator; core/anet-core/src/nn_impl.hpp:483 NetworkModuleRepository: 同じ形で、取り出しはGetFactory; core/anet-core/src/env.cpp:1105 Envだけは重複登録をANET_SYSTEM_ERRORにする; core/anet-core/src/agent.cpp:53 Agentは重複登録を黙って上書きする（gui.cpp:387、nn_impl.cpp:2566も同じ）; core/anet-core/include/anet/agent.hpp:349 自動登録マクロもANET_REGISTER_AGENT_FACTORYとANET_REGISTER_VIEW_CREATOR（gui.hpp:448）で同じ形を複製している

#### F-0014 P2 TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている

- 観点/分類: refactor / 既存の重複・不要コード
- 場所: `core/anet-core/src/dqn_based_agent.cpp`::TDLearner/QRLearner/IQNLearner::UpdateFromSamples
- 要約: 3つのLearnerのUpdateFromSamplesが、入力のshape検査、観測の正規化、Autocastでのforwardと損失、PERのIS重みでの加重平均、PER優先度の準備と反映、最適化、結果の組み立てを同じ順で書いている。2026-07時点はTD・QRの2本で、IQNの追加で3本になった。
- シナリオ: PERの重みの扱い（IS重みの正規化やNaNの扱いなど）を変える → 3つの関数の同じ箇所を別々に直す必要があり、1つ漏らすとアルゴリズムによってPERの補正が違ったまま学習する。
- 対処: 分布ごとに違う部分（forward・MakeTarget・損失）だけを仮想関数にし、骨格を基底の1関数に寄せる。性能と同seedの再現性（metrics checksum）を確かめる。
- 根拠: core/anet-core/src/dqn_based_agent.cpp:4021 TDLearner::UpdateFromSamples; core/anet-core/src/dqn_based_agent.cpp:4156 QRLearner::UpdateFromSamples: shape検査・正規化・Autocastでのforward・MakeTarget・損失の順; core/anet-core/src/dqn_based_agent.cpp:4280 IQNLearner::UpdateFromSamples: tauの生成以外はQRと同じ順; core/anet-core/src/dqn_based_agent.cpp:4236 QRのPER重み（use_per ? is_weights : ones）と加重平均; core/anet-core/src/dqn_based_agent.cpp:4416 IQNも同じ行を同じ形で持つ; core/anet-core/src/dqn_based_agent.cpp:4102 TDは同じ処理をif (use_per)の分岐で書いており、書き方が既に食い違っている; core/anet-core/src/dqn_based_agent.cpp:4247 PreparePerPriorityUpdate→Optimize→ApplyPerPriorityUpdate→MakeBatchUpdateResultの末尾を、QR・IQN（4427）・TD（4113）が繰り返す

#### F-0015 P2 SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている

- 観点/分類: refactor / 既存の重複・不要コード
- 場所: `core/anet-core/src/trainer.cpp`::SerialTrainRunner::DoStep/PipelineTrainRunner::DoStep
- 要約: 2つのランナーのDoStepが、環境の初期化、BatchExperienceの組み立て、LearnEvent・TrainEventの通知、カウンタの更新をそれぞれ書いている。違いは学習を同期で回すか非同期で回すかだけだが、共通部分も別々に保守されている。
- シナリオ: Step結果の契約を変える → Serialの検査だけを直し、Pipelineは検査なしで進む。既に、検査の有無と、イベントに入れる経験と学習結果の組（Pipelineでは1 stepずれる）が2つのランナーで食い違っている。
- 対処: 初期化・経験の組み立て・通知・カウンタ更新を基底TrainRunnerの関数にし、2つのDoStepは学習の同期・非同期の違いだけを持つ。
- 根拠: core/anet-core/src/trainer.cpp:497 Serialの初期化ブロック（Reset・ValidateEpisodeStructure・時間計測の開始）; core/anet-core/src/trainer.cpp:635 Pipelineにも同じ初期化ブロック; core/anet-core/src/trainer.cpp:563 SerialのBatchExperienceの組み立て; core/anet-core/src/trainer.cpp:712 Pipelineにも同じ組み立て; core/anet-core/src/trainer.cpp:540 Step結果のdevice・shapeの検査（540〜551）はSerialだけにあり、Pipelineには無い

#### F-0016 P2 HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている

- 観点/分類: refactor / 既存の重複・不要コード
- 場所: `core/anet-core/src/heat_map.cpp`::HeatMap::RenderRaw/TimeHeatMap::RenderRaw
- 要約: 2つのRenderRawが、値レンジの決定、サンプルのピクセルへの配置、平均化、色付けと画像の生成を別々に書いている。TimeHeatMapのScaleモードはX座標の変換だけが違う。
- シナリオ: 色付けやゼロ値の扱いを変える → 2つのRenderRawを別々に直す。既にフラグの効き方（MeanMode）と画像の確保方法が食い違い、TimeHeatMapだけがwxImageとの確保方法の不一致を抱えている。
- 対処: サンプル列を受け取って画像にする共通関数を作り、TimeHeatMapはX座標の変換だけを差し替える。
- 根拠: core/anet-core/src/heat_map.cpp:247 HeatMapの値レンジ決定（AutoNormValue）; core/anet-core/src/heat_map.cpp:461 TimeHeatMapにも同じ値レンジ決定; core/anet-core/src/heat_map.cpp:271 HeatMapのサンプル配置（Xのサブピクセル補間）; core/anet-core/src/heat_map.cpp:476 TimeHeatMapにも同じサンプル配置; core/anet-core/src/heat_map.cpp:309 HeatMapは平均化をHM_MeanModeで切り替える; core/anet-core/src/heat_map.cpp:497 TimeHeatMapは常に平均化する; core/anet-core/src/heat_map.cpp:317 画像の確保もHeatMapはmalloc、TimeHeatMap（501）はnew[]で食い違う

#### F-0017 P2 ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む

- 観点/分類: refactor / 肥大化
- 場所: `core/anet-core/src/observers.cpp`::ObserverFactory::ObserverFactory
- 要約: 2026-07に指摘したenumの変換表の二重は、metric_tokensへの集約で解消した。しかしコンストラクタは277行から331行に伸び、graphの節だけが共通の字句解析を通らずに$action・target:・interval:を独自に読む。
- シナリオ: graphで@eventや$stepを指定できるようにする → scalar・traceと同じ字句解析が使えず、graphの節のif連鎖を別に伸ばす。今もgraphだけが未知のtargetをWARNで読み飛ばし、scalarに無い別名（step_interval・eps_interval）を受理している。
- 対処: 設定の種類ごとの読み取りを関数に分け、graphもParseMetricTokensを通す。
- 根拠: core/anet-core/src/observers.cpp:1333 コンストラクタは1333〜1664の331行で、設定の種類（scalar・trace・image・graph）ごとの読み取りを1関数で抱える; core/anet-core/src/observers.cpp:1349 scalarはmetric_tokens::ParseMetricTokensで字句解析する; core/anet-core/src/observers.cpp:1615 graphの節は$action・$action_infoを独自に判定する; core/anet-core/src/observers.cpp:1625 target:・interval:（step_interval）・episode_interval:も独自のif連鎖で読む; core/anet-core/src/metrics.cpp:40 共通の字句解析ParseMetricTokens。enumの変換表はここに集約済み

#### F-0018 P2 QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う

- 観点/分類: refactor / 肥大化
- 場所: `apps/runner/src/QValuePanel.cpp`::QValuePanel::Update
- 要約: 毎stepの描画更新で、統計量の計算、Gridの差分更新、HeatMap用のデータ加工を1つの関数が続けて行う。表示モード（Histogram・Advantage・Log）の分岐も関数全体に散っている。
- シナリオ: HeatMapの加工に表示モードを1つ足す → Grid更新と同じ関数の中で、Advantage・Histogramの分岐を読み分けて差し込むことになり、Gridの差分更新（毎step全行を塗り直さないための最適化）を壊しても気づきにくい。
- 対処: 統計・Grid・HeatMapの3段を関数に分ける。Gridの差分更新の性能を確かめる。
- 根拠: apps/runner/src/QValuePanel.cpp:418 Updateは418〜773の356行; apps/runner/src/QValuePanel.cpp:436 統計量の計算; apps/runner/src/QValuePanel.cpp:447 Gridの更新（行構成・数値セル・選択行のハイライト・列幅の再計算）; apps/runner/src/QValuePanel.cpp:568 HeatMapデータの加工（Advantage変換・ヒストグラム化）

#### F-0019 P2 BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖

- 観点/分類: refactor / 肥大化
- 場所: `core/anet-core/src/dqn_based_agent.hpp`::BatchUpdateResult::GetScalar
- 要約: メトリクスのkeyごとのif連鎖がヘッダのクラス定義の中にあり、2026-07の145行から234行に伸びた。
- シナリオ: Learnerに診断値を1つ足す → ヘッダのif連鎖に分岐を足し、このヘッダを含む6つの.cpp（テストを含む）がすべて再コンパイルされる。keyの綴りの重複や到達しない分岐も、連鎖が長くて見落とす。
- 対処: .cppへ移し、keyから値への表か、群ごとの関数に分ける。
- 根拠: core/anet-core/src/dqn_based_agent.hpp:276 GetScalarは276〜509の234行をクラス定義の中に持ち、keyの比較が43個ある

#### F-0020 P2 Panel::OnKeyがproductionコードでdynamic_castを3回使う

- 観点/分類: refactor / 規約違反・不統一
- 場所: `core/anet-core/src/gui.cpp`::Panel::OnKey
- 要約: キーイベントの多重発火を防ぐため、発生元とその親にPanelがあるかをdynamic_castで調べている。規約はproductionでのdynamic_castを禁じる。2026-07に挙げたobservers.cppの1か所は無くなった。
- シナリオ: 具体的なsmell: 品質確認の機械検査（impl.mdの項番2）がgui.cppを変更するたびに3件を出す。Panelかどうかの判定を型の照会に頼るので、Panelを継承しない部品がキーを転送する形に変えると、ここの判定を型ごとに書き足すことになる。
- 対処: 転送済みであることをイベント側の印で表すか、Panelの登録と親子関係で判定する。
- 根拠: core/anet-core/src/gui.cpp:92 イベント元をdynamic_cast<wxWindow*>で取り出す; core/anet-core/src/gui.cpp:97 イベント元がPanelかをdynamic_cast<Panel*>で判定する; core/anet-core/src/gui.cpp:105 親をたどりながらdynamic_cast<Panel*>で判定する; docs/quality/impl.md:166 dynamic_castはテストコードに限るという規約

#### F-0024 P2 nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる

- 観点/分類: drift / サンプル設定
- 場所: `apps/runner/config/nn.txt`::net.body.@ResNet/net.body.@MLP_RES
- 要約: nn.txtのサンプルのbodyのうち、@ResNetは未定義のブロックを、@MLP_RESは未登録のtype（Composite）を使っており、どちらも選ぶと起動できない。
- シナリオ: サンプルを試すためにbodyとして@MLP_RESを選ぶ → [ResBlock]のtype=Compositeの解決でUnknown module typeになり、起動できない。@ResNetを選ぶと、未定義のブロック（Linear_128、MyReLU）の参照でエラーになる。
- 対処: サンプルを今の記法と登録済みのtypeで書き直すか、消す。
- 根拠: apps/runner/config/nn.txt:241 @ResNetは定義の無いブロックLinear_128とMyReLUを参照する; apps/runner/config/nn.txt:301 @MLP_RESが使う[ResBlock]のtypeが、登録の無いComposite; core/anet-core/src/nn_impl.cpp:2575 未登録のtypeはUnknown module typeでANET_SYSTEM_ERRORになる

#### F-0025 P2 train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う

- 観点/分類: drift / fail-fastの方針
- 場所: `core/anet-core/src/trainer.cpp`::RunnerFactory::CreateMainRunner
- 要約: CreateMainRunnerはrunner_typeがpipeline（大文字小文字を無視）かどうかだけを見て、それ以外は全てSerialにする。未知の値をfail-fastにする方針に反する。
- シナリオ: train.runner_type = pipline（綴り誤り）と書く → エラーにならずにSerialで走り、Pipelineのつもりの性能比較がSerialの結果になる。
- 対処: serial・pipeline以外はANET_SYSTEM_ERRORにする。
- 根拠: core/anet-core/src/trainer.cpp:745 pipelineと一致しなければ、値を確かめずにSerialTrainRunnerを作る（745〜748）; core/anet-core/src/trainer.cpp:772 train.runner_typeを読む; AGENTS.md:183 現行契約の設定の未知列挙値はANET_SYSTEM_ERRORでfail-fastする方針

#### F-0026 P2 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効）

- 観点/分類: test-gap / テストの空白
- 場所: `core/anet-core/src/scaler_impl.cpp`::RewardScalerFactory::CreateRewardScaler/RunningStdSingleTensorNormalizer
- 要約: scalerのテストは定数スケールとクリップ・統計の公開だけを確かめ、既定で有効な3つの経路（gammaからのpost_scaleの自動設定、外れ値を統計から外すrobust update、SymLogの後処理）を通らない。
- シナリオ: gammaの式を(1-gamma)から別の形に変える、robustの判定の不等号を逆にする、SymLogのscaleの掛け方を変える → どのテストも落ちず、既定の設定で動くDQN系の報酬・観測のスケールが黙って変わる。
- 対処: gammaとreference_q_stdから期待値を手計算するテスト、外れ値を含むバッチで統計が更新されないことのテスト、SymLogの数値のテストを足す。
- 根拠: core/anet-core/src/scaler_impl.cpp:756 use_auto_post_scaleではpost_scale = reference_q_std * (1 - gamma)を使う（756〜758）; core/anet-core/src/scaler_impl.cpp:336 robust updateは、偏差がrobust_std_threshold*stdを超える値を統計から外す; core/anet-core/src/scaler_impl.cpp:279 後処理SymLogはscale*symlog(x/scale); core/anet-core/src/scaler_test.cpp:12 scalerのテスト3本はすべてuse_auto_post_scale=falseで、robust・SymLogを使うテストも無い; apps/runner/config/agent.txt:240 use_auto_post_scale ?= true。robust（251）とSymLog（254）も既定で有効

#### F-0009 P2 PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる

- 観点/分類: defect / 整合性
- 場所: `core/anet-core/src/trainer.cpp`::PipelineTrainRunner::DoStep
- 要約: step Nの終わりにprev_exp_は遷移Nになるが、step Nで投入した学習は遷移N-1で行う。step N+1の待ち合わせで、遷移Nと、遷移N-1を入れたあとの学習結果を同じLearnEvent・TrainEventに入れる。Serialでは遷移と学習結果が同じstepのものなので、ランナーによってイベントの組の意味が変わる。
- シナリオ: train.runner_type=pipelineで、LearnEventのexperienceとupdate_result_listを同じstepのものとして突き合わせる処理を使う → 遷移Nの経験に、遷移N-1を入れたあとの学習結果が付き、Serialで取った値と1 stepずれる。countsも学習に使ったものではなく、1つあとのprev_counts_になる。
- 対処: 学習を投入したときのexp・countsを結果と一緒に保持し、イベントにはその組を渡す。
- 根拠: core/anet-core/src/trainer.cpp:692 学習の投入は、この時点のprev_exp_（1つ前のstepの遷移）とprev_counts_で行う; core/anet-core/src/trainer.cpp:712 そのあとprev_exp_をこのstepの遷移で上書きする; core/anet-core/src/trainer.cpp:665 次のstepのLearnEventは、上書き後のprev_exp_と、上書き前の遷移で投入した学習の結果を組にする; core/anet-core/src/trainer.cpp:673 TrainEventも同じ組で通知する; core/anet-core/src/trainer.cpp:569 SerialTrainRunnerは同じexpで学習した結果を、そのexpと組にする

#### F-0021 P3 dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い

- 観点/分類: refactor / 規約違反・不統一
- 場所: `core/anet-core/src/dqn_based_agent.hpp`
- 要約: core/anet-coreのヘッダで、このファイルだけがインクルードガードを持たない。先頭のファイル名のコメントも旧名のまま。
- シナリオ: このヘッダを含む別のヘッダを作り、同じ.cppから両方を含める → BatchUpdateResultなどのクラス定義が二重になり、コンパイルエラーになる。今は6つの.cppが直接1回ずつ含めるだけなので表面化していない。
- 対処: #pragma onceを足し、先頭コメントを今のファイル名に直す。
- 根拠: core/anet-core/src/dqn_based_agent.hpp:1 先頭コメントがrainbow_agent_impl.hppのままで、#pragma onceが無い（coreの他のヘッダは全てガードを持つ）

#### F-0022 P3 metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落）

- 観点/分類: drift / 設定のコメント
- 場所: `apps/runner/config/metrics_scalar.txt`::metrics.scalar 記法のコメント
- 要約: metrics.scalarの記法を説明する冒頭のコメントが、@episode_end・@session_end、$actor・$env、$train・$eval.[..]、$ema、interval:・clip:・ema_alpha:を含まず、$stepを&stepと書く。2026-07に正本としていたobservers.cppのコメントも古くなり、今の受理トークンを書いているのはParseMetricTokens自身だけになっている。
- シナリオ: コメントを見て軸を&episode_stepと書く → ParseMetricTokensは未知の語をkeyとして扱い、後勝ちで本来のkeyを上書きする → 存在しないkey「&episode_step」を読みにいき、本来の値は記録されない。
- 対処: 冒頭のコメントをParseMetricTokensの受理トークンに合わせ、observers.cppの古いコメントは消すか参照に置き換える。
- 根拠: apps/runner/config/metrics_scalar.txt:11 11〜14行目: &step表記で、@eventは@learn・@trainだけ、$targetに$actor・$env・$env_stepが無く、$train・$eval.[..]・$ema・interval:・clip:・ema_alpha:も無い; apps/runner/config/metrics_scalar.txt:21 同じファイルの定義は@session_end・$eval.[eval_target]・$runnerを使っている; core/anet-core/src/observers.cpp:1405 ObserverFactoryの中の記法コメント（1405〜1409）も、@learn・@trainだけで、旧形式の例を載せている; core/anet-core/src/metrics.cpp:45 実際に受理するトークンはParseMetricTokensが45〜60で分類する; core/anet-core/src/metrics_impl.hpp:8 どれにも当たらない語は既定のKind::KEYになる

#### F-0023 P3 nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている

- 観点/分類: drift / 設定のコメント
- 場所: `apps/runner/config/nn.txt`::Supported Module Types のコメント
- 要約: nn.txtの冒頭の対応typeの一覧が、登録済みの7種を欠き、typeではないAddを含む。init.modeのコメントもconstant・trunc_normal・defaultを欠く。
- シナリオ: nn.txtの一覧を見てブロックを組む → MaxPool2dやDropoutは使えないと判断し、逆にnet.block.[X].typeにAddと書くとUnknown module typeで落ちる。trunc_normalの初期化も一覧に無いので選ばれない。
- 対処: 一覧を登録済みのtypeに合わせ、Addは記法の節へ移す。init.modeのコメントにconstant・trunc_normal・defaultを足す。
- 根拠: apps/runner/config/nn.txt:16 対応typeの一覧（16〜41）にReshape・StackMerge・Dropout・LayerNorm2d・GAP2D・MaxPool2d・HybridSpatialEmbedderが無い; apps/runner/config/nn.txt:19 Addをtypeとして挙げるが、Addは登録されたtypeではなくstructureの記法; core/anet-core/src/nn_modules.cpp:3272 Reshape・StackMerge・Dropout（3274）・LayerNorm2d（3286）・GAP2D（3289）・MaxPool2d（3290）・HybridSpatialEmbedder（3293）を登録している; apps/runner/config/nn.txt:311 init.modeのコメントがxavier・he・orthogonalだけ（5〜10行目も同じ）; core/anet-core/include/anet/nn.hpp:69 init.modeはdefault・xavier・he・orthogonal・constant・trunc_normal

#### F-0027 P3 MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い

- 観点/分類: test-gap / テストの空白
- 場所: `core/anet-core/src/muzero_rb.cpp`::MuZeroReplayBuffer
- 要約: MuZeroReplayBuffer・MuZeroExperienceQueue・MuZeroUniformSamplerを直接使うテストが、coreのテストに無い。n-stepの価値ターゲット、エピソード終端での打ち切り、unroll窓のmaskは、MuZeroの学習ターゲットそのものを決める。
- シナリオ: 終端でのフラッシュの範囲（steps_to_done）を1つずらす、終端のあとのmaskを1に戻す → 終端の向こうの報酬や価値が学習ターゲットに混ざるが、どのテストも落ちない。
- 対処: 短いエピソードを手で作り、価値ターゲットとmaskの期待値を並べるテストを足す。
- 根拠: core/anet-core/src/muzero_rb.cpp:98 最新のstepがdone・truncatedのとき、残りのstepを終端までの報酬で打ち切ってフラッシュする（98〜119）; core/anet-core/src/muzero_rb.cpp:222 サンプラーはunroll_steps+1の窓で行動・報酬・価値・maskを詰め、終端（248）以降をmaskで切る; core/anet-core/src/muzero_rb.cpp:287 Pushはbatchの経験をenvごとのキューへ渡す

#### F-0010 P3 ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す

- 観点/分類: defect / 境界
- 場所: `core/anet-core/include/anet/rl.hpp`::ActionSpec::GetShape
- 要約: 連続行動のshapeは、コメントでは(dims.size(),)だが、実装は次元数ぶんの1を並べる。今は連続行動のEnvが無いので表面化していない。
- シナリオ: 3次元の連続行動のEnvを足す → GetShapeが{1,1,1}を返し、replayのactions_が(num_envs, capacity, 1, 1, 1)で確保される → (N, 3)の行動を書き込むとshapeが合わずに落ちる。
- 対処: 連続行動では{dims.size()}を返す。連続行動を足すときに、replayへの書き込みと合わせて確かめる。
- 根拠: core/anet-core/include/anet/rl.hpp:255 連続行動では次元ごとに1をpushし、{1,1,...}を返す（コメントは(dims.size(),)と言う）; core/anet-core/src/replay_buffer_impl.cpp:492 replayのactions_をこのshapeで確保する


## クローズ済み・保留

| id | 状態 | 優先度 | 観点 | 場所 | タイトル | 更新 |
|---|---|---|---|---|---|---|
| F-0001 | fixed | P1 | refactor | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js` | 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している | 2026-10-03 |
