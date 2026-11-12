# 課題の一覧

道具が作り直す。手で直さない。既定は本版（次版・解決を除く）。

| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261107-949e8a](issue-20261107-949e8a.md) | 対応中 | 5.0 | — | 成果物 | その他 | 322_env_individual | Env | 機能追加 | [PRD126](../../memo/126_atari_ram_metric_distinct_and_bit_reducers_10prd.md) | 残件有り | AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない |
| [issue-20261111-77ac3c](issue-20261111-77ac3c.md) | 対応中 | 3.0 | — | 成果物 | PRD | 324_dqn, 321_agent_core | アルゴリズム | 機能改善 | [PRD123](../../memo/123_rnd_intrinsic_reward_scale_10prd.md) | 残件有り | RNDの内的報酬が序盤の大きな誤差で膨らんだ累積の誤差統計に抑えられ、βが名目どおりに効かずRunの中で強まる |
| [issue-20261111-61ad60](issue-20261111-61ad60.md) | 調査中 | 2.7 | — | 成果物 | 実装 | 110_runner_app | 開発 | 調査 | — | 未確認 | GUI既定スイートのLogs折り返し整列テストが変更前から失敗する |
| [issue-20261106-16b7af](issue-20261106-16b7af.md) | 対応中 | 2.6 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 315_nn, 317_metrics | アルゴリズム | 機能追加 | [PRD116](../../memo/116_rnd_lifelong_novelty_10prd.md) | 残件無し | RND による一生の新しさだけの群（内的報酬の第一の群） |
| [issue-20261103-ab98b7](issue-20261103-ab98b7.md) | 調査中 | 1.6 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | リファクタリング | [PRD125](../../memo/125_dqn_learner_update_skeleton_10prd.md) | 残件有り | TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている |
| [issue-20261104-52ffba](issue-20261104-52ffba.md) | 対応中 | 1.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | 課題の仕組みに残る改善点をまとめて管理する（段階に分けた PRD の「実装済み」の時機ほか） |
| [issue-20261103-955e50](issue-20261103-955e50.md) | 調査中 | 1.2 | — | 成果物 | PRD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [PRD081](../../memo/081_evalpanel_episode_trace_10prd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261103-6a32b4](issue-20261103-6a32b4.md) | 調査中 | 1.0 | — | 成果物 | PRD | 315_nn, 321_agent_core, 318_rl_core, 312_util | アルゴリズム | リファクタリング | [PRD938](../../memo/938_nn_trace_capture_unification_10prd.md) | 残件有り | forward の名前付き出力を NetworkForwardCapture に一本化する |
| [issue-20261103-23d185](issue-20261103-23d185.md) | 対応中 | 0.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD925](../../memo/925_issue_list_views_10prd.md) | 残件有り | issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表） |
| [issue-20261103-ba635a](issue-20261103-ba635a.md) | 未着手 | 6.5 | — | 成果物 | 監査 | 317_metrics | UI | 不具合 | — | 残件有り | VideoLoggerが最初の画像の大きさで固定され、以後の画像のサイズを確かめずに読む |
| [issue-20261103-b754ff](issue-20261103-b754ff.md) | 未着手 | 6.5 | — | 成果物 | 監査 | 324_dqn | 学習基盤 | 不整合 | [PRD944](../../memo/944_observation_pipeline_10prd.md) | 残件有り | stucker.stack_keysに存在しないキー名を書いても検出されない |
| [issue-20261103-80c355](issue-20261103-80c355.md) | 未着手 | 6.0 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | Notifierが無ロックのvectorを、GUIスレッドのAttach/Detachと学習スレッドのNotifyで同時に触る |
| [issue-20261103-93f8a6](issue-20261103-93f8a6.md) | 未着手 | 6.0 | — | 成果物 | 監査 | 318_rl_core, 321_agent_core | 学習基盤 | 不具合 | — | 残件有り | actor_approxのGPU推論のEval Actorに手動の行動を打つと、CPUの強制行動とGPUのq_valuesのdevice不一致でWithActionが止まる |
| [issue-20261109-1499d0](issue-20261109-1499d0.md) | 未着手 | 5.0 | — | 成果物 | 監査 | 321_agent_core, 319_gui, 315_nn | 基盤 | 不整合 | — | 未確認 | Agent・View・NNモジュールのRepositoryは同じ名前の重複登録を黙って上書きし、Envだけがエラーで止まる |
| [issue-20261110-3f4e63](issue-20261110-3f4e63.md) | 未着手 | 4.5 | — | 成果物 | PRD | 110_runner_app | UI | 不具合 | — | 未確認 | QValuePanelで平滑化が有効なままsmooth_radiusが0以下だと、空の重みの範囲外を読む |
| [issue-20261107-1226e5](issue-20261107-1226e5.md) | 未着手 | 4.3 | — | 成果物 | その他 | 324_dqn, 110_runner_app | UI | 機能追加 | — | 未確認 | Evaluation Traceで内的報酬（RNDの新しさ）を見られない |
| [issue-20261111-288879](issue-20261111-288879.md) | 未着手 | 3.7 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | ExperienceSamples のフィールドの並びが To の 2 経路・ForEachTensor・ToString に分かれ、To の 2 経路で未定義の tensor の扱いが食い違っている |
| [issue-20261111-9e6e27](issue-20261111-9e6e27.md) | 未着手 | 3.6 | — | 成果物 | PRD | 110_runner_app | UI | 機能改善 | — | 未確認 | 無人で走るRun（番号付きbat）でWARNに気づけない |
| [issue-20261110-546bdb](issue-20261110-546bdb.md) | 未着手 | 3.5 | — | 成果物 | 実装 | 316_config | 開発 | 不整合 | — | 未確認 | PRD103 の hidden 設定検査が現用定義と異なる指標名を要求し4環境で失敗する |
| [issue-20261103-1beaf9](issue-20261103-1beaf9.md) | 未着手 | 3.3 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| [issue-20261111-7cc67b](issue-20261111-7cc67b.md) | 未着手 | 3.3 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | EvalRunner の 1 step が、多重定義 4 つ・-1 の特別な値・bool の引数で、パネルからの 1 step と評価 session の 1 step を切り替えている |
| [issue-20261111-a5370c](issue-20261111-a5370c.md) | 未着手 | 3.3 | — | 成果物 | 監査 | 315_nn | アルゴリズム | リファクタリング | — | 未確認 | NetworkBody の走査（branch→struct→block→module）が 6 か所に写され、null の検査の有無が場所ごとに違う |
| [issue-20261111-c5c9a7](issue-20261111-c5c9a7.md) | 未着手 | 3.3 | — | 成果物 | 監査 | 315_nn | アルゴリズム | リファクタリング | — | 未確認 | spectral norm の式（行列化・実効の分母・無効の判定・単位化）が 3〜5 か所ずつ写され、無効の数を返す口も 2 つある |
| [issue-20261103-a3574b](issue-20261103-a3574b.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 120_metrics_viewer | UI | 不具合 | — | 残件有り | DataFetcherの単一のabort枠を独立した呼び出し元が奪い合い、abortが失敗Toastや初期Run選択の喪失になる |
| [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | 未着手 | 3.0 | — | 成果物 | PRD | 322_env_individual | Env | 機能追加 | [PRD085](../../memo/085_dropmerge_eval_score_axes_10prd.md) | 残件有り | DropMerge の評価スコア（本家互換スコアほか複数の軸） |
| [issue-20261103-c8fb06](issue-20261103-c8fb06.md) | 未着手 | 3.0 | — | 成果物 | PRD | 312_util, 110_runner_app | 基盤 | 機能改善 | [PRD071](../../memo/071_uncaught_fatal_visibility_10prd.md) | 残件有り | PRD 071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| [issue-20261111-006d9a](issue-20261111-006d9a.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 320_env_core | 学習基盤 | リファクタリング | — | 未確認 | env.cpp で lane 1 本分の Reset・Step の結果を書き込む処理が 4 か所に写され、GetAuxDataList も 3 つあり、数え方と範囲の外の扱いが食い違っている |
| [issue-20261111-3986b7](issue-20261111-3986b7.md) | 未着手 | 3.0 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 未確認 | テストを回す範囲・回数と、テストの控え・ログの置き場の決まりが無く、実装の時間と .scratch が膨らむ |
| [issue-20261111-4a6b6f](issue-20261111-4a6b6f.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | RunManager のコンストラクタが 277 行あり、評価の設定の解釈・名前の予約・構築・ログ・配線が 1 つの関数に並んでいる |
| [issue-20261111-585e60](issue-20261111-585e60.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 未確認 | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261111-7c926a](issue-20261111-7c926a.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 315_nn | アルゴリズム | リファクタリング | — | 未確認 | Network の CopyTo と SoftCopyTo が parameter と buffer を名前で対にする処理を別々に持ち、SoftCopyTo は learn step ごとに名前つきの一覧を 4 つ作る |
| [issue-20261111-be150a](issue-20261111-be150a.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 未確認 | StepTraceColumnSpec の target と display を文字列で持ち、metrics.cpp・gui.cpp・EvalTracePanel.cpp の 3 ファイル 17 か所で文字列を比べている |
| [issue-20261111-c68178](issue-20261111-c68178.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 322_env_individual | Env | リファクタリング | — | 未確認 | DropMerge のエピソード統計を 1 つ足すと、Env のメンバ・Reset・StepResult の初期化子とメンバ・GetScalar の 5 か所へ同じ形を足すことになる |
| [issue-20261111-e33797](issue-20261111-e33797.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 324_dqn | アルゴリズム | リファクタリング | — | 未確認 | DefaultDQNAgentConfig のコンストラクタが 534 行あり、読み取り・検証・既定値の導出・記録が 1 つの関数に並び、接頭辞の下を写して記録する処理も 4 か所に手書きされている |
| [issue-20261111-ea3483](issue-20261111-ea3483.md) | 未着手 | 3.0 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | scalar メトリクスの Observer が、イベントの中身を 6〜9 個の引数にばらして 3 段持ち回り、同じ 8 引数のコンストラクタを 4 回書いている |
| [issue-20261107-16ffd8](issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | 成果物 | PRD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 未確認 | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-15ba39](issue-20261111-15ba39.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | EvalPanel のモデル同期が mode を文字列のまま step ごとに解釈し直し、EvalTracePanel は pane の開閉から導ける active_ を 4 通りの問い合わせで合わせている |
| [issue-20261111-17fea4](issue-20261111-17fea4.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 324_dqn, 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | 方策の種類（policy_type）と分布の表し方（quantile_mode）を文字列のまま持ち回り、数字の別名の受け付け方が関数ごとに違う |
| [issue-20261111-52ed74](issue-20261111-52ed74.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 315_nn, 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | 可塑性指標の一覧と重みノルムのキーの並びを、nn・DQN・ImageCls の 5〜6 か所で手で合わせている |
| [issue-20261111-5f695d](issue-20261111-5f695d.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 314_tensor | 基盤 | リファクタリング | — | 未確認 | 報酬の scaler と観測の normalizer で、clip と変換の同じ数行が 4〜6 か所にあり、clip が無効のときの扱いが既に食い違っている |
| [issue-20261111-645ecc](issue-20261111-645ecc.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | メトリクス宣言の語（event・step・target）と enum の対応が 5 か所以上に分かれ、解決済みの定義も同じ値を文字列と enum の 2 形で持っている |
| [issue-20261111-6c0bf1](issue-20261111-6c0bf1.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 322_env_individual | Env | リファクタリング | — | 未確認 | GridMazeEnv が毎 step 変わらない盤面を 1 マスずつ tensor へ書き直し、成功と穴落ちを報酬の値と設定の値が等しいかで判定している |
| [issue-20261111-c90df5](issue-20261111-c90df5.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 319_gui | UI | リファクタリング | — | 未確認 | ForwardedMouseEvent と ForwardedKeyEvent、Panel::OnMouse と OnKey が、型の名前だけ違う写しになっている |
| [issue-20261111-cdc102](issue-20261111-cdc102.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 316_config, 315_nn, 324_dqn | 基盤 | リファクタリング | — | 未確認 | WeightInitConfig などの設定の部品に読み取りの口が無く、使う側が 1 キーずつ読むので、同じ 6〜8 行が十数か所にあり、読み方も食い違っている |
| [issue-20261111-e80c7b](issue-20261111-e80c7b.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | DQNActionInfo::GetScalar が 6 種の指標群を 1 つの関数（276 行）で処理し、「指定した行動を除いた最大値」の計算を 2 回写している |
| [issue-20261111-fe23e0](issue-20261111-fe23e0.md) | 未着手 | 2.7 | — | 成果物 | 監査 | 318_rl_core, 320_env_core | 学習基盤 | リファクタリング | — | 未確認 | BatchEnvSpec に「group の数」「group と lane の対応」を返す関数が無く、使う側が三項演算を 19 か所に書き、終端の mask を CPU へ移す処理も関数ごとに書いている |
| [issue-20261108-06a3f0](issue-20261108-06a3f0.md) | 未着手 | 2.6 | — | 成果物 | その他 | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 未確認 | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない |
| [issue-20261108-f0f243](issue-20261108-f0f243.md) | 未着手 | 2.6 | — | 成果物 | PRD | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 残件有り | Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする） |
| [issue-20261111-08d557](issue-20261111-08d557.md) | 未着手 | 2.6 | — | 成果物 | 監査 | 324_dqn | アルゴリズム | リファクタリング | — | 未確認 | 内的報酬の扱いが DefaultDQNAgent の 5 つの関数と 6 つのメンバに散り、UpdateFromBatch の 95 行のうち 56 行を占めている |
| [issue-20261111-24d885](issue-20261111-24d885.md) | 未着手 | 2.6 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | 診断指標の群を 1 つ足すたびに、キー配列・添字の解析・購読の間隔・要求の旗・読み出しの同じ形を 12 か所前後へ足し、添字も数値で直書きしている |
| [issue-20261103-b2a0b1](issue-20261103-b2a0b1.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| [issue-20261103-78df9c](issue-20261103-78df9c.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util | 基盤 | 不具合 | — | 残件有り | device指定の文法誤りで、ParseDeviceとInitRLが同じ誤りのfatalを2回ログに出す |
| [issue-20261103-9b314a](issue-20261103-9b314a.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util | 基盤 | 不整合 | — | 残件有り | 明示cudaが使えないときのエラーがキー名だけを並べ、指定値を書かない |
| [issue-20261103-d4258c](issue-20261103-d4258c.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 312_util, 316_config | 基盤 | 不整合 | — | 残件有り | 実効deviceを設計文書は「採用値」と呼び、PRD070とJSONの欄はeffective（実効）と呼ぶ |
| [issue-20261103-7c9a70](issue-20261103-7c9a70.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 316_config | 基盤 | 不整合 | [PRD907](../../memo/907_config_file_layout_10prd.md) | 残件有り | Agentの切替プロファイル@fast/@balance/@heavyがcommon.txtにあり、020 §3.6の表（agent.txtの役割）と食い違う |
| [issue-20261103-eeb2c3](issue-20261103-eeb2c3.md) | 未着手 | 2.5 | — | 成果物 | 監査 | 317_metrics | UI | 不整合 | — | 残件有り | video_codecのエラー文が実キーapp.metrics_logger.video_codecではなくrootの無い名前を出す |
| [issue-20261103-e6dbb4](issue-20261103-e6dbb4.md) | 未着手 | 2.3 | — | 成果物 | 監査 | 317_metrics | UI | リファクタリング | — | 残件有り | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| [issue-20261103-34d35d](issue-20261103-34d35d.md) | 未着手 | 2.3 | — | 成果物 | PRD | 318_rl_core | 学習基盤 | 不具合 | [PRD916](../../memo/916_episode_end_read_semantics_10prd.md) | 残件有り | `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる |
| [issue-20261110-0de9e7](issue-20261110-0de9e7.md) | 未着手 | 2.3 | — | 成果物 | PRD | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | [PRD121](../../memo/121_extrinsic_intrinsic_value_decomposition_10prd.md) | 未確認 | 外的と内的の価値の分解（DQN 系の Q を外的報酬と内的報酬の 2 つの Head で学ぶ） |
| [issue-20261111-2c251b](issue-20261111-2c251b.md) | 未着手 | 2.3 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | Replay の窓の規則 3 クラスが同じ骨格の写しで、PER のスカラーのキーも一覧・条件式・分岐の 3〜4 か所に並んでいる |
| [issue-20261111-360278](issue-20261111-360278.md) | 未着手 | 2.3 | — | 成果物 | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | Runner のパネルに描画と表の写しがある（GL キャンバスへ画像を貼る一式が 2 パネルに同文、EvalTracePanel の行の探索 8 か所と再生ツールの並び 3 か所） |
| [issue-20261103-2d66d2](issue-20261103-2d66d2.md) | 未着手 | 2.2 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 学習基盤 | 調査 | [PRD910](../../memo/910_network_lock_audit_10prd.md) | 残件有り | 999: train/eval mode 整理後の排他精査 PRD（草案） |
| [issue-20261103-5addea](issue-20261103-5addea.md) | 未着手 | 2.2 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-cdac7c](issue-20261103-cdac7c.md) | 未着手 | 2.2 | — | 成果物 | PRD | 321_agent_core | アルゴリズム | 機能改善 | [PRD940](../../memo/940_target_update_step_axis_10prd.md) | 残件有り | target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`） |
| [issue-20261111-4351bc](issue-20261111-4351bc.md) | 未着手 | 2.2 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | DQN の Actor が const の MakeAction の中で 11 個の mutable メンバを書き換えて状態を進め、一時的な出力を結果の public メンバで運んでいる |
| [issue-20261111-72d640](issue-20261111-72d640.md) | 未着手 | 2.2 | — | 成果物 | 監査 | 324_dqn | アルゴリズム | リファクタリング | — | 未確認 | DefaultDQNAgent のコンストラクタが 242 行あり、RainbowAgent が Head の選択・Actor の生成・値の振り分け・更新を写して既に 5 点で食い違っている |
| [issue-20261103-72f807](issue-20261103-72f807.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | 不具合 | — | 残件有り | use_td_clipが損失の前にTDをclampするため、clipを超えたサンプルの勾配が0になる |
| [issue-20261103-78c643](issue-20261103-78c643.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| [issue-20261103-39ddb2](issue-20261103-39ddb2.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 120_metrics_viewer | UI | 不具合 | — | 残件有り | 初期化中にScreenshotへ切り替えるとinit完了時にmodeが上書きされ、init失敗時はScreenshotから抜けられない |
| [issue-20261103-73a449](issue-20261103-73a449.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 120_metrics_viewer | UI | 不具合 | — | 判定不能 | window内に点が無いtagはgraph blockごと消え、全tag一括のReset View以外では戻せない |
| [issue-20261103-785e68](issue-20261103-785e68.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD945](../../memo/945_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-7c8cd7](issue-20261103-7c8cd7.md) | 未着手 | 2.0 | — | 成果物 | PRD | 316_config | 基盤 | リファクタリング | [PRD907](../../memo/907_config_file_layout_10prd.md) | 残件有り | 設定ファイルの配置（Agent 全体・具象 Agent・機能 include・フォルダ） |
| [issue-20261103-7f1789](issue-20261103-7f1789.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 322_env_individual | アルゴリズム | 機能追加 | [PRD088](../../memo/088_nn_adaptive_max_pool_10prd.md) | 残件有り | 出力の大きさで指定する max pooling（AdaptiveMaxPool2d） |
| [issue-20261103-a39330](issue-20261103-a39330.md) | 未着手 | 2.0 | — | 成果物 | PRD | 321_agent_core, 324_dqn | アルゴリズム | 機能追加 | [PRD941](../../memo/941_iqn_first_priority_quantile_pair_10prd.md) | 残件有り | PRD 999: IQN初回Learner priorityのquantile-pair方式 |
| [issue-20261103-aa5758](issue-20261103-aa5758.md) | 未着手 | 2.0（引継ぎ） | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 314_tensor, 325_muzero | 学習基盤 | リファクタリング | [PRD944](../../memo/944_observation_pipeline_10prd.md) | 残件有り | 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 |
| [issue-20261111-07f4ed](issue-20261111-07f4ed.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 未確認 | 「N エピソードに 1 回だけ記録する」状態が Conv2dVisualizationObserver と GraphVizObserver に少し違う形で書かれ、記録を始める条件と終える条件が食い違っている |
| [issue-20261111-0ddffa](issue-20261111-0ddffa.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | RndNoveltyEstimator に 2 つの統計方式が同居して分岐が 5 つの関数の 9 か所に散り、Estimate と Learn も前半の約 20 行を同じ形で書いている |
| [issue-20261111-0ef722](issue-20261111-0ef722.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 未確認 | probe.hpp・probe.cpp の約 3 分の 1 がどこからも使われず、残る 2 つの Probe クラスと 5 つの IdxExtractor も写しになっている |
| [issue-20261111-269a9f](issue-20261111-269a9f.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 322_env_individual | Env | リファクタリング | — | 未確認 | DropMergeEnv::Step が 267 行で行動の解釈・物理・裁定・統計・報酬を並べ、行動の解釈・DROP の位置・果物の走査・盤面の受け渡しにも写しがある |
| [issue-20261111-54c7e1](issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 未確認 | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261111-90fa5c](issue-20261111-90fa5c.md) | 未着手 | 2.0 | — | 成果物 | 監査 | 110_runner_app | UI | リファクタリング | — | 未確認 | QValuePanel の ComputeQValueHeatMap が 218 行で値の変換・範囲の更新・投票・色の正規化を並べ、データの受け渡しにスレッド越し用の部品と保護の無いメンバが混ざっている |
| [issue-20261103-58696c](issue-20261103-58696c.md) | 未着手 | 1.9 | — | ハーネス | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD112](../../memo/112_claude_memory_tidy_10prd.md) | 残件有り | Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`） |
| [issue-20261103-0ef7b8](issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | 成果物 | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-61d771](issue-20261103-61d771.md) | 未着手 | 1.6 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD906](../../memo/906_cicd_improvement_10prd.md) | 残件有り | CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助） |
| [issue-20261110-adf548](issue-20261110-adf548.md) | 未着手 | 1.6 | — | 成果物 | その他 | 322_env_individual | Env | 機能追加 | — | 未確認 | GridMaze に MiniGrid 互換の課題が無く、難しい探索の検証と DI-engine との比較に使えない |
| [issue-20261103-75b8da](issue-20261103-75b8da.md) | 未着手 | 1.5 | — | 成果物 | PRD | 321_agent_core, 318_rl_core | アルゴリズム | 機能改善 | [PRD943](../../memo/943_nstep_target_correction_10prd.md) | 残件有り | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正） |
| [issue-20261103-e29c92](issue-20261103-e29c92.md) | 未着手 | 1.4 | — | 成果物 | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 残件有り | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う |
| [issue-20261103-df24e5](issue-20261103-df24e5.md) | 未着手 | 1.4 | — | 成果物 | 監査 | 312_util | 基盤 | 不具合 | — | 判定不能 | AnetExceptionのstack_trace()がwxStackWalkerに頼っていて、Runnerのプロセスでは空になる |
| [issue-20261103-182ca1](issue-20261103-182ca1.md) | 未着手 | 1.4 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 学習基盤 | 性能改善 | [PRD909](../../memo/909_core_perf_candidates_202607_10prd.md) | 残件有り | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 |
| [issue-20261103-50644a](issue-20261103-50644a.md) | 未着手 | 1.4 | — | 成果物 | PRD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [PRD919](../../memo/919_eval_session_step_trace_10prd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-52bda0](issue-20261103-52bda0.md) | 未着手 | 1.4 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [PRD096](../../memo/096_metrics_step_axis_interval_10prd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-66f453](issue-20261103-66f453.md) | 未着手 | 1.4 | — | 成果物 | PRD | 110_runner_app, 316_config | 学習基盤 | 機能追加 | [PRD952](../../memo/952_run_queue_snapshot_10prd.md) | 残件有り | Run キュー（計画スナップショット + exe 固定） |
| [issue-20261103-aa8701](issue-20261103-aa8701.md) | 未着手 | 1.4 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 324_dqn, 320_env_core | 学習基盤 | 機能追加 | [PRD930](../../memo/930_serialize_10prd.md) | 残件有り | Run Serialization（Run 全体 Save / Load） |
| [issue-20261107-2d4932](issue-20261107-2d4932.md) | 未着手 | 1.4 | — | 成果物 | その他 | 110_runner_app, 324_dqn | UI | 機能追加 | — | 未確認 | EvalPanelで行動選択のτをランダムと固定で切り替えられず、学んだ行動と学習した価値の判断を見比べられない |
| [issue-20261110-9d4e5a](issue-20261110-9d4e5a.md) | 未着手 | 1.4 | — | 成果物 | その他 | 311_dev_tools | 開発 | 機能追加 | — | 未確認 | C++ の単純な書き方の問題を機械で検出する仕組み（lint・静的解析）が無い |
| [issue-20261103-71c4f4](issue-20261103-71c4f4.md) | 未着手 | 1.3 | — | 成果物 | PRD | 315_nn, 317_metrics, 321_agent_core | アルゴリズム | 機能追加 | [PRD920](../../memo/920_nn_block_metrics_10prd.md) | 残件有り | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| [issue-20261103-f39d10](issue-20261103-f39d10.md) | 未着手 | 1.3 | — | 成果物 | PRD | 318_rl_core, 324_dqn | 学習基盤 | 不具合 | [PRD912](../../memo/912_background_eval_snapshot_ordering_10prd.md) | 残件有り | Background eval の network snapshot 順序保証 |
| [issue-20261103-07629c](issue-20261103-07629c.md) | 未着手 | 1.2 | — | 成果物 | PRD | 110_runner_app, 318_rl_core, 319_gui | UI | 不具合 | [PRD948](../../memo/948_qvaluepanel_manual_action_state_alignment_10prd.md) | 残件有り | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ |
| [issue-20261103-71bb9a](issue-20261103-71bb9a.md) | 未着手 | 1.2 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能追加 | [PRD959](../../memo/959_test_line_coverage_10prd.md) | 残件有り | 行カバレッジ計測（OpenCppCoverage） |
| [issue-20261103-7227bf](issue-20261103-7227bf.md) | 未着手 | 1.2 | — | 成果物 | PRD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [PRD947](../../memo/947_q_statistics_config_screening_10prd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| [issue-20261103-8c3c6d](issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [PRD950](../../memo/950_replay_eviction_sample_count_10prd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261103-dbe666](issue-20261103-dbe666.md) | 未着手 | 1.2 | — | 成果物 | PRD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [PRD915](../../memo/915_dropmerge_neet_diagnostics_10prd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261110-d21e5f](issue-20261110-d21e5f.md) | 未着手 | 1.2 | — | 成果物 | PRD | 318_rl_core, 325_muzero | 学習基盤 | 不具合 | — | 未確認 | MuZeroのReplayで、truncatedで終わるときに補う価値が、打ち切った次の状態ではなく最後のstep自身のroot_valueになっている |
| [issue-20261111-e90762](issue-20261111-e90762.md) | 未着手 | 1.2 | — | 成果物 | PRD | 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | PERの抽選が埋まらないときに残りを一様の抽選で埋める代替が、明示した要求を黙って代替しない規約と食い違う |
| [issue-20261103-15f115](issue-20261103-15f115.md) | 未着手 | 1.1 | — | 成果物 | PRD | 110_runner_app, 319_gui | UI | 機能改善 | [PRD113](../../memo/113_runner_per_monitor_dpi_10prd.md) | 残件有り | Runner を Per-Monitor V2 の DPI 対応で動かす PRD（manifest の gdiScaling を外す） |
| [issue-20261103-1edf84](issue-20261103-1edf84.md) | 未着手 | 1.1 | — | 成果物 | PRD | 321_agent_core, 324_dqn, 315_nn | アルゴリズム | 機能追加 | [PRD905](../../memo/905_bootstrapped_multihead_q_10prd.md) | 残件有り | Bootstrapped DQN 型の複数の Head（1 本の Run の中に別々の価値関数を持つ） |
| [issue-20261103-206933](issue-20261103-206933.md) | 未着手 | 1.1 | — | 成果物 | PRD | 110_runner_app | 学習基盤 | 機能追加 | [PRD955](../../memo/955_seed_population_early_stop_10prd.md) | 残件有り | 多数の seed で打席を稼ぐ（条件付きの打ち切りで、火が付かなかった Run を早く止める） |
| [issue-20261103-51220d](issue-20261103-51220d.md) | 未着手 | 1.1 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [PRD932](../../memo/932_episode_forensics_10prd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261103-6f6c73](issue-20261103-6f6c73.md) | 未着手 | 1.0 | — | 成果物 | PRD | 321_agent_core | アルゴリズム | 機能追加 | [PRD958](../../memo/958_temporl_skip_followups_10prd.md) | 残件有り | TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習） |
| [issue-20261103-b4e5c9](issue-20261103-b4e5c9.md) | 未着手 | 1.0 | — | 成果物 | PRD | 316_config, 110_runner_app | 基盤 | リファクタリング | [PRD908](../../memo/908_config_root_ownership_alignment_10prd.md) | 残件有り | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) |
| [issue-20261103-c16aab](issue-20261103-c16aab.md) | 未着手 | 1.0 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [PRD954](../../memo/954_scalar_threshold_rate_10prd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261105-f4b870](issue-20261105-f4b870.md) | 未着手 | 1.0 | — | 成果物 | その他 | 311_dev_tools, 315_nn | アルゴリズム | 調査 | — | 未確認 | LLM が NN の重みを直接読み書きする試み（Weight-Space Programming）を anet で試す |
| [issue-20261111-d1d12b](issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | 成果物 | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | アルゴリズム | リファクタリング | — | 未確認 | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-cfe994](issue-20261103-cfe994.md) | 未着手 | 0.9 | — | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| [issue-20261103-c5c791](issue-20261103-c5c791.md) | 未着手 | 0.9 | — | 成果物 | PRD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [PRD946](../../memo/946_prd097_performance_remeasurement_10prd.md) | 残件有り | PRD097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| [issue-20261103-f33167](issue-20261103-f33167.md) | 未着手 | 0.9 | — | 成果物 | PRD | 324_dqn, 321_agent_core, 315_nn | アルゴリズム | 機能追加 | [PRD107](../../memo/107_noisy_gain_lane_ladder_10prd.md) | 残件有り | 学習の Actor の lane ごとに NoisyNet のノイズの強さを変える（ノイズのラダー） |
| [issue-20261103-f40ae8](issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | 成果物 | PRD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [PRD056](../../memo/056_timehistogram_fine_grid_10prd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261103-322a83](issue-20261103-322a83.md) | 未着手 | 0.8 | — | 成果物 | PRD | 315_nn | アルゴリズム | 機能追加 | [PRD951](../../memo/951_residual_alpha_gate_10prd.md) | 残件有り | 標準 Spectral Normalization と residual identity 開始を両立する α ゲート |
| [issue-20261103-53ed7a](issue-20261103-53ed7a.md) | 未着手 | 0.8 | — | 成果物 | PRD | 110_runner_app, 319_gui | UI | 機能追加 | [PRD953](../../memo/953_runner_theme_switch_10prd.md) | 残件有り | 起動中の Runner を OS のテーマ（ライト／ダーク）の切替に追従させる |
| [issue-20261103-afe841](issue-20261103-afe841.md) | 未着手 | 0.8 | — | 成果物 | PRD | 321_agent_core, 324_dqn, 318_rl_core | 学習基盤 | 機能追加 | [PRD933](../../memo/933_staged_batch_replay_schedule_10prd.md) | 残件有り | DefaultDQN Staged Batch / Replay Schedule |
| [issue-20261103-b13ed2](issue-20261103-b13ed2.md) | 未着手 | 0.8 | — | 成果物 | PRD | 311_dev_tools | 開発 | 機能改善 | [PRD949](../../memo/949_release_packaging_cmake_10prd.md) | 残件有り | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする |
| [issue-20261103-969184](issue-20261103-969184.md) | 未着手 | 0.6 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD926](../../memo/926_llm_actor_10prd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261103-145ee2](issue-20261103-145ee2.md) | 未着手 | 0.5 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [PRD927](../../memo/927_metric_targeted_hyperparam_control_10prd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261103-4a2b5e](issue-20261103-4a2b5e.md) | 未着手 | 0.5 | — | 成果物 | PRD | 315_nn, 321_agent_core | アルゴリズム | リファクタリング | [PRD937](../../memo/937_nn_head_projection_separation_10prd.md) | 残件有り | NN Head の最終射影と出力変換の責務分離 |
| [issue-20261103-6c40b8](issue-20261103-6c40b8.md) | 未着手 | 0.5 | — | 成果物 | PRD | 315_nn | アルゴリズム | 機能追加 | [PRD939](../../memo/939_noisynet_effective_weight_spectral_norm_10prd.md) | 残件有り | NoisyNet のノイズ込み重みに対する Spectral Normalization |
| [issue-20261103-f993bc](issue-20261103-f993bc.md) | 未着手 | 0.5 | — | 成果物 | PRD | 321_agent_core, 324_dqn, 325_muzero, 315_nn | 学習基盤 | リファクタリング | [PRD936](../../memo/936_nn_forward_amp_consolidation_10prd.md) | 残件有り | NN 実行設定への AMP の集約（全 Agent 対象） |
| [issue-20261103-9df5c5](issue-20261103-9df5c5.md) | 未着手 | 2.7 | [issue-20261103-80c355](issue-20261103-80c355.md) | 成果物 | PRD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [PRD942](../../memo/942_notifier_scoped_detach_10prd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261103-9bd71e](issue-20261103-9bd71e.md) | 未着手 | 2.0 | [issue-20261103-aa5758](issue-20261103-aa5758.md) | 成果物 | PRD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [PRD921](../../memo/921_agent_prev_action_obs_10prd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261103-ca1a85](issue-20261103-ca1a85.md) | 未着手 | 1.6 | [issue-20261103-1beaf9](issue-20261103-1beaf9.md) | 成果物 | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| [issue-20261103-124da5](issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | 成果物 | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD956](../../memo/956_smdp_transition_duration_10prd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-b754ff](issue-20261103-b754ff.md), [issue-20261103-aa5758](issue-20261103-aa5758.md), [issue-20261103-9bd71e](issue-20261103-9bd71e.md), [issue-20261103-ab98b7](issue-20261103-ab98b7.md), [issue-20261103-75b8da](issue-20261103-75b8da.md), [issue-20261107-1226e5](issue-20261107-1226e5.md), [issue-20261107-949e8a](issue-20261107-949e8a.md), [issue-20261110-0de9e7](issue-20261110-0de9e7.md) | 成果物 | PRD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [PRD924](../../memo/924_intrinsic_reward_novelty_10prd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261109-adc5c4](issue-20261109-adc5c4.md) | 未着手 | 1.4 | [issue-20261109-1499d0](issue-20261109-1499d0.md) | 成果物 | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 未確認 | Agent・Env・View・NNモジュールの4つのRepositoryが同じ形を別々に実装し、登録と取り出しの関数名もそろっていない |
| [issue-20261112-4efa57](issue-20261112-4efa57.md) | 未着手 | 1.4 | [issue-20261103-ab98b7](issue-20261103-ab98b7.md) | 成果物 | PRD | 321_agent_core | アルゴリズム | リファクタリング | — | 未確認 | replay fitの診断（EvaluateReplayFit）が学習の更新の前半を別に書き、分布の表し方での分岐を持っている |
| [issue-20261103-6661d2](issue-20261103-6661d2.md) | 未着手 | 1.3 | [issue-20261103-f39d10](issue-20261103-f39d10.md), [issue-20261103-2d66d2](issue-20261103-2d66d2.md) | 成果物 | PRD | 318_rl_core, 324_dqn, 110_runner_app | 学習基盤 | 機能追加 | [PRD913](../../memo/913_best_checkpoint_10prd.md) | 残件有り | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| [issue-20261103-b68e49](issue-20261103-b68e49.md) | 未着手 | 1.3 | [issue-20261103-aa5758](issue-20261103-aa5758.md) | 成果物 | PRD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [PRD922](../../memo/922_input_mapping_joystick_10prd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261103-2dd820](issue-20261103-2dd820.md) | 未着手 | 1.0 | [issue-20261103-785e68](issue-20261103-785e68.md) | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools, 322_env_individual | 開発 | リファクタリング | [PRD957](../../memo/957_source_layout_10prd.md) | 残件有り | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける） |
| [issue-20261103-13bc20](issue-20261103-13bc20.md) | 未着手 | 0.8 | [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | 成果物 | PRD | 322_env_individual | Env | 機能追加 | [PRD086](../../memo/086_dropmerge_macro_actions_10prd.md) | 残件有り | DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化） |
| [issue-20261108-227b05](issue-20261108-227b05.md) | 未着手 | 0.8 | [issue-20261103-736e39](issue-20261103-736e39.md) | 成果物 | PRD | 324_dqn, 321_agent_core | アルゴリズム | 調査 | — | 未確認 | RNDのpredictorを周期的に初期化してbonusを続かせる案は、GridMazeとLunarLanderの試行で利得が無く、実装を戻した |
| [issue-20261103-578986](issue-20261103-578986.md) | 未着手 | 0.6 | [issue-20261103-aa5758](issue-20261103-aa5758.md), [issue-20261103-b68e49](issue-20261103-b68e49.md) | 成果物 | PRD | 318_rl_core, 319_gui, 110_runner_app | 学習基盤 | 機能追加 | [PRD923](../../memo/923_input_target_train_forced_action_10prd.md) | 残件有り | 操作入力の入力先切替と Train 側の強制行動（学習への反映） |
| [issue-20261103-5a783c](issue-20261103-5a783c.md) | 未着手 | 0.6 | [issue-20261103-124da5](issue-20261103-124da5.md), [issue-20261103-75b8da](issue-20261103-75b8da.md) | 成果物 | PRD | 321_agent_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD914](../../memo/914_decision_aligned_nstep_10prd.md) | 残件有り | 次の判断まで遡る n-step（判断単位の可変長 n-step） |

着手できる 126 / 待ち 14 / 保留 0 / 次版 0 / 解決 0

## 解決待ち

AI の判定が「残件無し」で、解決でない課題。解決にするかは開発者が決める（コミットの準備で持ちかける）。

| ID | 状態 | AIの判定 | 判定日 | 課題 |
|---|---|---|---|---|
| [issue-20261106-16b7af](issue-20261106-16b7af.md) | 対応中 | 残件無し | 2026-11-07 | RND による一生の新しさだけの群（内的報酬の第一の群） |

## 再開候補

解決の日付以後に（解決の日付が無ければ日付によらず）AI の判定が「残件有り」になった解決の課題。状態を戻すかは開発者が決める。

該当する課題はありません。

## 課題の無い PRD

docs/memo 直下の PRD のうち、どの課題の prd からも指されないもの。その PRD の仕事の課題を作るか、題材の課題の prd に書く。

該当する PRD はありません。
