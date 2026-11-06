# 課題の一覧

道具が作り直す。手で直さない。既定は本版（次版・解決を除く）。

| # | ID | 状態 | WSJF | 優先度 | 待ち | 種別 | 工程 | 領域 | 性質 | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261105-a88e01](issue-20261105-a88e01.md) | 未着手 | 4.7 | 中 | — | ハーネス | ハーネス保守 | 311_dev_tools | 改善 | 直った見込み | コミットメッセージ案をそのまま使えるよう文言のルールを改善する |
| 2 | [issue-20261103-c1c608](issue-20261103-c1c608.md) | 未着手 | 4.5 | 中 | — | 成果物 | 実装 | 311_dev_tools | ずれ | 残っている | 現行ガイドのoptuna参照5か所が実在しない相対パスを指す |
| 3 | [issue-20261103-80c355](issue-20261103-80c355.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 318_rl_core | 不具合 | 残っている | Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る |
| 4 | [issue-20261103-ba635a](issue-20261103-ba635a.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 317_metrics | 不具合 | 残っている | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む |
| 5 | [issue-20261103-be4cd5](issue-20261103-be4cd5.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 322_env_individual | 不具合 | 直った見込み | Directモードで列0への落下にもNOOPペナルティが加算される |
| 6 | [issue-20261103-cfe994](issue-20261103-cfe994.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| 7 | [issue-20261103-73a449](issue-20261103-73a449.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 残っている | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない |
| 8 | [issue-20261103-93f8a6](issue-20261103-93f8a6.md) | 未着手 | 3.6 | 高 | — | 成果物 | 監査 | 318_rl_core | 不具合 | 残っている | actor_approxのGPU推論のEval Actorに手動の行動を打つと、CPUの強制行動とGPUのq_valuesのdevice不一致でWithActionが止まる |
| 9 | [issue-20261104-cc2529](issue-20261104-cc2529.md) | 未着手 | 3.6 | 高 | — | 成果物 | PRD | 317_metrics, 322_env_individual | 索引 | 実装済み | 画像メモリの所有権とDropMergeのNOOP判定の修正 PRD |
| 10 | [issue-20261103-07629c](issue-20261103-07629c.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 110_runner_app, 318_rl_core, 319_gui | 索引 | 未確認 | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ 暫定 PRD |
| 11 | [issue-20261103-124da5](issue-20261103-124da5.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 322_env_individual | 索引 | 未確認 | SMDP 対応（遷移ごとの経過時間を割引に通す）暫定 PRD |
| 12 | [issue-20261103-13bc20](issue-20261103-13bc20.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 322_env_individual | 索引 | 未確認 | DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化）暫定 PRD |
| 13 | [issue-20261103-182ca1](issue-20261103-182ca1.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 索引 | 未確認 | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 暫定 PRD |
| 14 | [issue-20261103-1edf84](issue-20261103-1edf84.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | Bootstrapped DQN 型の複数の Head（1 本の Run の中に別々の価値関数を持つ）暫定 PRD |
| 15 | [issue-20261103-206933](issue-20261103-206933.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 110_runner_app | 索引 | 未確認 | 多数の seed で打席を稼ぐ（条件付きの打ち切りで、火が付かなかった Run を早く止める）暫定 PRD |
| 16 | [issue-20261103-23d185](issue-20261103-23d185.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 311_dev_tools | 索引 | 未確認 | issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表）暫定 PRD |
| 17 | [issue-20261103-322a83](issue-20261103-322a83.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | 標準 Spectral Normalization と residual identity 開始を両立する α ゲート暫定 PRD |
| 18 | [issue-20261103-34d35d](issue-20261103-34d35d.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core | 索引 | 未確認 | `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる 暫定 PRD |
| 19 | [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 322_env_individual | 索引 | 未確認 | DropMerge の評価スコア（本家互換スコアほか複数の軸）暫定 PRD |
| 20 | [issue-20261103-4a2b5e](issue-20261103-4a2b5e.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 315_nn, 321_agent_core | 索引 | 未確認 | NN Head の最終射影と出力変換の責務分離 |
| 21 | [issue-20261103-52bda0](issue-20261103-52bda0.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | 索引 | 未確認 | メトリクスの横軸と間隔を数える単位を分ける PRD |
| 22 | [issue-20261103-55b5f1](issue-20261103-55b5f1.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 120_metrics_viewer | 索引 | 未確認 | Metrics ViewerのLOD集約数の記述整合 暫定PRD |
| 23 | [issue-20261103-5a783c](issue-20261103-5a783c.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 321_agent_core, 322_env_individual | 索引 | 未確認 | 次の判断まで遡る n-step（判断単位の可変長 n-step）暫定 PRD |
| 24 | [issue-20261103-61d771](issue-20261103-61d771.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 311_dev_tools | 索引 | 未確認 | CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助）暫定 PRD |
| 25 | [issue-20261103-6661d2](issue-20261103-6661d2.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 110_runner_app, 317_metrics, 312_util, 322_env_individual | 索引 | 未確認 | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| 26 | [issue-20261103-66f453](issue-20261103-66f453.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 316_config, 110_runner_app | 索引 | 未確認 | Run キュー（計画スナップショット + exe 固定）暫定 PRD |
| 27 | [issue-20261103-6a32b4](issue-20261103-6a32b4.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 312_util, 315_nn, 321_agent_core | 索引 | 未確認 | forward の名前付き出力を NetworkForwardCapture に一本化する 暫定 PRD |
| 28 | [issue-20261103-6c40b8](issue-20261103-6c40b8.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 315_nn | 索引 | 未確認 | NoisyNet のノイズ込み重みに対する Spectral Normalization |
| 29 | [issue-20261103-71bb9a](issue-20261103-71bb9a.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 311_dev_tools, 316_config | 索引 | 未確認 | 行カバレッジ計測（OpenCppCoverage）暫定 PRD |
| 30 | [issue-20261103-71c4f4](issue-20261103-71c4f4.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 315_nn, 317_metrics, 321_agent_core | 索引 | 未確認 | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| 31 | [issue-20261103-7227bf](issue-20261103-7227bf.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | Q 値統計による多環境ハイパラ健全性スクリーニング 暫定 PRD |
| 32 | [issue-20261103-75b8da](issue-20261103-75b8da.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 321_agent_core | 索引 | 未確認 | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正）暫定 PRD |
| 33 | [issue-20261103-a86dcc](issue-20261103-a86dcc.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 318_rl_core, 312_util, 321_agent_core, 324_dqn | 索引 | 未確認 | eval スケジュールの初回発火の抑止 暫定 PRD |
| 34 | [issue-20261103-c16aab](issue-20261103-c16aab.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools, 318_rl_core | 索引 | 未確認 | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） 暫定 PRD |
| 35 | [issue-20261103-c5c791](issue-20261103-c5c791.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | PRD097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| 36 | [issue-20261103-c8fb06](issue-20261103-c8fb06.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 312_util, 110_runner_app | 索引 | 未確認 | PRD 071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| 37 | [issue-20261103-cdac7c](issue-20261103-cdac7c.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 321_agent_core | 索引 | 未確認 | target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`）暫定 PRD |
| 38 | [issue-20261103-dbe666](issue-20261103-dbe666.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | DropMerge の NEET 診断（trace で足りるか確かめてから指標化）暫定 PRD |
| 39 | [issue-20261103-f993bc](issue-20261103-f993bc.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | NN 実行設定への AMP の集約（全 Agent 対象） |
| 40 | [issue-20261104-52ffba](issue-20261104-52ffba.md) | 未着手 | 2.6 | 中 | — | ハーネス | ハーネス保守 | 311_dev_tools | 改善 | 未確認 | 課題の仕組みに残る改善点をまとめて管理する（段階に分けた PRD の「実装済み」の時機ほか） |
| 41 | [issue-20261106-16b7af](issue-20261106-16b7af.md) | 未着手 | 2.6 | 中 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 315_nn, 317_metrics | 索引 | 未確認 | RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD |
| 42 | [issue-20261103-72f807](issue-20261103-72f807.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 321_agent_core | 不具合 | 残っている | use_td_clipが損失の前にTDをclampするため、clipを超えたサンプルの勾配が0になる |
| 43 | [issue-20261103-1beaf9](issue-20261103-1beaf9.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 318_rl_core | 不具合 | 残っている | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| 44 | [issue-20261103-2751c8](issue-20261103-2751c8.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | — | ずれ | 残っている | nn.txtのbodyのサンプル2つ（@ResNet・@MLP_RES）が、選ぶとエラーになる |
| 45 | [issue-20261103-582880](issue-20261103-582880.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 318_rl_core | ずれ | 残っている | train.runner_typeがpipeline以外の任意の文字列を黙ってserialとして扱う |
| 46 | [issue-20261103-58bf49](issue-20261103-58bf49.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 110_runner_app | 改善 | 残っている | QValuePanel::Updateが356行で、統計・Grid更新・HeatMap加工を1関数で行う |
| 47 | [issue-20261103-741251](issue-20261103-741251.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 321_agent_core | 改善 | 残っている | BatchUpdateResult::GetScalarがヘッダ内のinlineで234行、keyの比較43個のif連鎖 |
| 48 | [issue-20261103-78c643](issue-20261103-78c643.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| 49 | [issue-20261103-ab98b7](issue-20261103-ab98b7.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 321_agent_core | 改善 | 残っている | TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている |
| 50 | [issue-20261103-ca1a85](issue-20261103-ca1a85.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| 51 | [issue-20261103-d337bd](issue-20261103-d337bd.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 314_tensor | 改善 | 残っている | 報酬スケーラのauto_post_scaleと、観測正規化のrobust update・SymLogにテストが無い（どれも既定で有効） |
| 52 | [issue-20261103-dd8058](issue-20261103-dd8058.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 317_metrics | 不具合 | 直った見込み | wxImage::SetDataにnew[]の配列を渡している（wxImageはfreeで解放する） |
| 53 | [issue-20261103-e088eb](issue-20261103-e088eb.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 319_gui | 改善 | 残っている | Panel::OnKeyがproductionコードでdynamic_castを3回使う |
| 54 | [issue-20261103-e29c92](issue-20261103-e29c92.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 321_agent_core | 改善 | 残っている | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う |
| 55 | [issue-20261103-e6dbb4](issue-20261103-e6dbb4.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 317_metrics | 改善 | 残っている | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| 56 | [issue-20261103-df24e5](issue-20261103-df24e5.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 312_util | 不具合 | 判断できない | AnetExceptionのstack_trace()がwxStackWalkerに頼っていて、Runnerのプロセスでは空になる |
| 57 | [issue-20261103-39ddb2](issue-20261103-39ddb2.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 残っている | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない |
| 58 | [issue-20261103-863fe2](issue-20261103-863fe2.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 直った見込み | 未完成のLOD pageをbucketごとに全行読み直すため、LOD射影で読む行数が表示bucket数の2乗になる |
| 59 | [issue-20261103-a3574b](issue-20261103-a3574b.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 120_metrics_viewer | 不具合 | 残っている | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる |
| 60 | [issue-20261103-2c9a07](issue-20261103-2c9a07.md) | 未着手 | 2.2 | 中 | — | 成果物 | 監査 | 110_runner_app | 改善 | 残っている | 旧キーapp.metrics_logger.runs_dir/root_dir専用のWARNが残り、クリーンブレーク方針に反する |
| 61 | [issue-20261103-ee32bd](issue-20261103-ee32bd.md) | 未着手 | 1.7 | 中 | — | 成果物 | 実装 | 311_dev_tools | ずれ | 残っている | PRD069の検証出力2か所が消失しており実装記録のリンクを開けない |
| 62 | [issue-20261103-03a1b6](issue-20261103-03a1b6.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 321_agent_core | 改善 | 残っている | dqn_based_agent.hppにインクルードガードが無く、先頭のファイル名のコメントが古い |
| 63 | [issue-20261103-60f11c](issue-20261103-60f11c.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | MuZeroReplayBufferの価値ターゲットの打ち切りと、unroll窓のmaskにテストが無い |
| 64 | [issue-20261103-879a70](issue-20261103-879a70.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | ずれ | 残っている | metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落） |
| 65 | [issue-20261103-b2a0b1](issue-20261103-b2a0b1.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 318_rl_core | 不具合 | 残っている | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| 66 | [issue-20261103-ef0c2b](issue-20261103-ef0c2b.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | ずれ | 残っている | nn.txtの対応typeの一覧とinit.modeのコメントが実装とずれている |
| 67 | [issue-20261103-5a5373](issue-20261103-5a5373.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | InitialPriorityCompleterのロックの前提の@noteがGetStatsに触れていない |
| 68 | [issue-20261103-d4aa77](issue-20261103-d4aa77.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 318_rl_core | 改善 | 残っている | DefaultReplayBufferのコンストラクタの、priority storeとcompleterを対で持つ不変条件にテストが無い |
| 69 | [issue-20261103-78df9c](issue-20261103-78df9c.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 312_util | 不具合 | 残っている | device指定の文法誤りで、ParseDeviceとInitRLが同じ誤りのfatalを2回ログに出す |
| 70 | [issue-20261103-9b314a](issue-20261103-9b314a.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 312_util | ずれ | 残っている | 明示cudaが使えないときのエラーがキー名だけを並べ、指定値を書かない |
| 71 | [issue-20261103-c1e48b](issue-20261103-c1e48b.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 312_util | 改善 | 残っている | InitRLがagent.device・env.device・run.eval_deviceの既定値を、所有するConfigとは別に持っている |
| 72 | [issue-20261103-d4258c](issue-20261103-d4258c.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | ずれ | 残っている | 実効deviceを設計文書は「採用値」と呼び、PRD070とJSONの欄はeffective（実効）と呼ぶ |
| 73 | [issue-20261103-554849](issue-20261103-554849.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | ずれ | 残っている | apps/README.mdがDoxygen zipをanet-lab-docs-<version>.zipと書くが、実際はanet-lab-doxygen-<版>.zip |
| 74 | [issue-20261103-66100f](issue-20261103-66100f.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | ずれ | 残っている | Conv2dPanel.conv2d.mergin_x/yとimage.@conv2d.conv2d.mergin_x/yが綴りの誤りで読まれない |
| 75 | [issue-20261103-7c9a70](issue-20261103-7c9a70.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | — | 改善 | 残っている | Agentの切替プロファイル@fast/@balance/@heavyがcommon.txtにあり、020 §3.6の表（agent.txtの役割）と食い違う |
| 76 | [issue-20261103-b754ff](issue-20261103-b754ff.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 324_dqn | ずれ | 残っている | stucker.stack_keysに存在しないキー名を書いても検出されない |
| 77 | [issue-20261103-eeb2c3](issue-20261103-eeb2c3.md) | 未着手 | 1.7 | 低 | — | 成果物 | 監査 | 317_metrics | ずれ | 残っている | video_codecのエラー文が実キーapp.metrics_logger.video_codecではなくrootの無い名前を出す |
| 78 | [issue-20261103-0ef7b8](issue-20261103-0ef7b8.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | 索引 | 未確認 | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号）PRD |
| 79 | [issue-20261103-145ee2](issue-20261103-145ee2.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core | 索引 | 未確認 | メトリクス目標値によるハイパラ自律調整 暫定 PRD |
| 80 | [issue-20261103-15f115](issue-20261103-15f115.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 110_runner_app, 319_gui | 索引 | 未確認 | Runner を Per-Monitor V2 の DPI 対応で動かす PRD（manifest の gdiScaling を外す） |
| 81 | [issue-20261103-2349d0](issue-20261103-2349d0.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 315_nn, 324_dqn, 314_tensor, 317_metrics, 110_runner_app, 322_env_individual | 索引 | 実装済み | TempoRL 対応（スキップ: 選んだ行動を続ける長さの学習）PRD |
| 82 | [issue-20261103-2d66d2](issue-20261103-2d66d2.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 索引 | 未確認 | 999: train/eval mode 整理後の排他精査 PRD（草案） |
| 83 | [issue-20261103-2dd820](issue-20261103-2dd820.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools | 索引 | 未確認 | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける）暫定 PRD |
| 84 | [issue-20261103-4c0f04](issue-20261103-4c0f04.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 120_metrics_viewer, 312_util | 索引 | 未確認 | MetricsViewer のタグセット（名前付きのタグ選択の保存と呼び出し）PRD |
| 85 | [issue-20261103-50644a](issue-20261103-50644a.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 320_env_core, 317_metrics, 321_agent_core, 318_rl_core, 110_runner_app | 索引 | 未確認 | 評価 session の step trace を Run の記録に残す PRD |
| 86 | [issue-20261103-51220d](issue-20261103-51220d.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査）暫定 PRD |
| 87 | [issue-20261103-53ed7a](issue-20261103-53ed7a.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 110_runner_app, 319_gui | 索引 | 未確認 | 起動中の Runner を OS のテーマ（ライト／ダーク）の切替に追従させる PRD |
| 88 | [issue-20261103-578986](issue-20261103-578986.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 319_gui, 110_runner_app | 索引 | 未確認 | 操作入力の入力先切替と Train 側の強制行動（学習への反映）暫定 PRD |
| 89 | [issue-20261103-58696c](issue-20261103-58696c.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 311_dev_tools | 索引 | 未確認 | Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`）PRD |
| 90 | [issue-20261103-5addea](issue-20261103-5addea.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 索引 | 未確認 | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 91 | [issue-20261103-6f6c73](issue-20261103-6f6c73.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core | 索引 | 未確認 | TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習）暫定 PRD |
| 92 | [issue-20261103-736e39](issue-20261103-736e39.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 324_dqn, 322_env_individual, 110_runner_app | 索引 | 未確認 | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の家族 暫定 PRD |
| 93 | [issue-20261103-785e68](issue-20261103-785e68.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 索引 | 未確認 | anet-core の外で実装する拡張点とコード配置の規約 暫定 PRD |
| 94 | [issue-20261103-7c8cd7](issue-20261103-7c8cd7.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 316_config | 索引 | 未確認 | 設定ファイルの配置（Agent 全体・具象 Agent・機能 include・フォルダ）暫定 PRD |
| 95 | [issue-20261103-7f1789](issue-20261103-7f1789.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 322_env_individual, 315_nn | 索引 | 未確認 | 出力の大きさで指定する max pooling（AdaptiveMaxPool2d）暫定 PRD |
| 96 | [issue-20261103-8c3c6d](issue-20261103-8c3c6d.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 索引 | 未確認 | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る 暫定 PRD |
| 97 | [issue-20261103-955e50](issue-20261103-955e50.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | 索引 | 未確認 | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る PRD |
| 98 | [issue-20261103-969184](issue-20261103-969184.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 322_env_individual | 索引 | 未確認 | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド）暫定 PRD |
| 99 | [issue-20261103-9bd71e](issue-20261103-9bd71e.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 索引 | 未確認 | 直前行動観測の共通部品（`prev_action` フレーム段） |
| 100 | [issue-20261103-9df5c5](issue-20261103-9df5c5.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 110_runner_app, 314_tensor | 索引 | 未確認 | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す PRD |
| 101 | [issue-20261103-a39330](issue-20261103-a39330.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | PRD 999: IQN初回Learner priorityのquantile-pair方式 |
| 102 | [issue-20261103-aa5758](issue-20261103-aa5758.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor | 索引 | 未確認 | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 PRD |
| 103 | [issue-20261103-aa8701](issue-20261103-aa8701.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | Run Serialization（Run 全体 Save / Load）暫定 PRD |
| 104 | [issue-20261103-afe841](issue-20261103-afe841.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | — | 索引 | 未確認 | DefaultDQN Staged Batch / Replay Schedule 暫定 PRD |
| 105 | [issue-20261103-b13ed2](issue-20261103-b13ed2.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 120_metrics_viewer, 311_dev_tools, 312_util, 316_config, 320_env_core, 313_profiling | 索引 | 未確認 | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする PRD |
| 106 | [issue-20261103-b4e5c9](issue-20261103-b4e5c9.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 110_runner_app, 317_metrics, 318_rl_core, 321_agent_core, 312_util, 320_env_core, 314_tensor, 316_config, 311_dev_tools, 322_env_individual | 索引 | 未確認 | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) 暫定 PRD |
| 107 | [issue-20261103-b68e49](issue-20261103-b68e49.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 110_runner_app, 318_rl_core, 319_gui, 321_agent_core, 322_env_individual | 索引 | 未確認 | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel）暫定 PRD |
| 108 | [issue-20261103-b713c2](issue-20261103-b713c2.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core | 索引 | 実装済み | 課題の記録を issue（1 課題 1 ファイル）に替え、仕事は issue・指示は PRD で持つ（台帳の退役、課題の仕組み、PRD の採番とフォルダの改め）PRD |
| 109 | [issue-20261103-f33167](issue-20261103-f33167.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 315_nn | 索引 | 実装済み | 学習の Actor の lane ごとに NoisyNet のノイズの強さを変える（ノイズのラダー）PRD |
| 110 | [issue-20261103-f39d10](issue-20261103-f39d10.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 318_rl_core, 324_dqn | 索引 | 未確認 | Background eval の network snapshot 順序保証 |
| 111 | [issue-20261103-f40ae8](issue-20261103-f40ae8.md) | 未着手 | 1.6 | 中 | — | 成果物 | PRD | 317_metrics, 314_tensor, 318_rl_core | 索引 | 未確認 | TimeHistogram fine-bin保持・描画時集約 PRD |
| 112 | [issue-20261103-fb8a45](issue-20261103-fb8a45.md) | 未着手 | 1.6 | 中 | — | 成果物 | 実装 | 311_dev_tools | ずれ | 残っている | 過去PRDのコード参照16か所が文書からの相対パスとして解決できない |
| 113 | [issue-20261105-f4b870](issue-20261105-f4b870.md) | 未着手 | 1.0 | 低 | — | 成果物 | その他 | 311_dev_tools, 315_nn | 案 | 未確認 | LLM が NN の重みを直接読み書きする試み（Weight-Space Programming）を anet で試す |

着手できる 113 / 待ち 0 / 保留 0 / 次版 0 / 解決 0
