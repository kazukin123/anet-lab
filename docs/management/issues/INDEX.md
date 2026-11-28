# 課題の一覧

道具が作り直す。手で直さない。既定は本版（次版・解決を除く）。

| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261107-949e8a](issue-20261107-949e8a.md) | 対応中 | 5.0 | — | コード | その他 | 322_env_individual | Env | 機能追加 | [RFD0126](../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md) | 残件無し | AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない |
| [issue-20261111-61ad60](issue-20261111-61ad60.md) | 調査中 | 2.7 | — | コード | 実装 | 110_runner_app | 開発 | 調査 | — | 残件有り | GUI既定スイートのLogs折り返し整列テストが変更前から失敗する |
| [issue-20261117-eedd0c](issue-20261117-eedd0c.md) | 調査中 | 2.6 | — | コード | 実装 | 110_runner_app | UI | 不具合 | — | 残件無し | Evaluation Trace の単一行選択テストが合成ドラッグ後に失敗する |
| [issue-20261122-c1744a](issue-20261122-c1744a.md) | 対応中 | 2.6 | — | コード | RFD | 321_agent_core, 324_dqn | 学習基盤 | 機能改善 | [RFD0191](../../memo/0191_rnd_raw_error_and_recent_learning_10rfd.md) | 残件有り | RND と補助学習の設定キーが段階とモードの関係を表しておらず、warm-up の 1 つの件数が 2 つの待ちを束ねていて、設計文書 212 から設定と動きを引きにくい |
| [issue-20261125-a19ab8](issue-20261125-a19ab8.md) | 対応中 | 2.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | AI の応答が今の決まりと成果物に沿う案に寄り、根本策や既存から外れる別案を、本筋と分けて示す場所が無い |
| [issue-20261125-822a3e](issue-20261125-822a3e.md) | 対応中 | 1.6（引継ぎ） | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | [RFD0202](../../memo/0202_wiki_accumulate_and_lookup_10rfd.md) | 残件有り | wiki に知見が積み上がりにくく、引きにくい（質疑の答えを残す経路、題材をまたぐ知見、サーベイと公開実装の事例、「今の結論」と目録の長さ） |
| [issue-20261104-52ffba](issue-20261104-52ffba.md) | 対応中 | 1.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | 課題の仕組みに残る改善点をまとめて管理する（段階に分けた RFD の「実装済み」の時機ほか） |
| [issue-20261126-56483f](issue-20261126-56483f.md) | 対応中 | 1.3 | — | ハーネス | その他 | 311_dev_tools | 開発 | 機能追加 | [RFD0204](../../memo/0204_branch_merge_conflict_assist_10rfd.md) | 残件有り | ブランチをまたぐマージの衝突（番号の重なり・wiki の追記・生成物）を、両側の内容と文脈を読んで解く AI の支援が無い |
| [issue-20261103-955e50](issue-20261103-955e50.md) | 調査中 | 1.2 | — | コード | RFD | 318_rl_core, 317_metrics, 110_runner_app, 311_dev_tools, 120_metrics_viewer | UI | 機能追加 | [RFD0081](../../memo/0081_evalpanel_episode_trace_10rfd.md) | 残件有り | EvalPanel のゲームを trace へ出す — dormant タグの trace を鏡写しインスタンスへ結び付け、trace 行の時刻で振り返る |
| [issue-20261103-23d185](issue-20261103-23d185.md) | 対応中 | 0.9 | — | ハーネス | RFD | 311_dev_tools | 開発 | 機能追加 | [RFD0187](../../memo/0187_issue_list_views_10rfd.md) | 残件有り | issue の一覧の見せ方（依存のツリー、スプレッドシート、見やすい表） |
| [issue-20261123-abb245](issue-20261123-abb245.md) | 未着手 | 5.0 | — | ハーネス | 監査 | — | 開発 | 機能改善 | — | 未確認 | docs/StampFly.md（約 100 KB）が 2026-08-23 から 3 か月、Git に追跡されないまま作業ツリーで書き足されている |
| [issue-20261125-8eff2e](issue-20261125-8eff2e.md) | 未着手 | 5.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | DQN の Learner が、forward のあとに「価値の分解なら合わせた Q を作る」処理を 6 か所に条件つきで書き、同じ目的の DqnForwardExecutor と 2 つの道になっている |
| [issue-20261125-fa1be2](issue-20261125-fa1be2.md) | 未着手 | 5.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | ActorQHintConfig が、ヒントではなく行動の選択（価値の分解・β・TBO）と診断の割引率を決めていて、名前と役割が合わない |
| [issue-20261128-f0f3b8](issue-20261128-f0f3b8.md) | 未着手 | 4.3 | — | コード | 監査 | 311_dev_tools | 開発 | リファクタリング | — | 未確認 | Optuna の複数 seed の途中失敗を trial の状態と集計結果まで確かめるテストが無い |
| [issue-20261123-6b991d](issue-20261123-6b991d.md) | 未着手 | 4.0 | — | 文書 | 監査 | 311_dev_tools, 321_agent_core | 開発 | 調査 | — | 残件有り | 状態が中身と合わない課題が 3 件ある（対応中なのに実装計画も最近の作業も無い 23d185・52ffba、RFD が着手しないと書くのに未着手の 6f6c73） |
| [issue-20261125-fe014b](issue-20261125-fe014b.md) | 未着手 | 4.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | 不具合 | — | 未確認 | 価値の分解の行動選択の診断（extrinsic_q_gap など）が、IQN の UQE では平均の Q ではなく risk の τ（全分布の問い合わせがあれば risk と全範囲を混ぜた τ）の平均で作られる |
| [issue-20261123-b73a41](issue-20261123-b73a41.md) | 未着手 | 3.7 | — | ハーネス | 監査 | — | 開発 | 機能改善 | — | 未確認 | 実験記録の本文に名前の出る学習 Run のうち、実効 config が複製されていないものがある（KFM の RND 誤差の記録で 13 本、LunarLander の TempoRL の記録で 12 本ほか） |
| [issue-20261125-fd5a4a](issue-20261125-fd5a4a.md) | 未着手 | 3.7 | — | コード | 監査 | 120_metrics_viewer | UI | リファクタリング | — | 未確認 | Viewer のブラウザで、詳細キャッシュが 64 MiB を超えたときの解放と先読みの停止、Copy Name の代替のコピー、分割した数列の復号を、通すテストが無い |
| [issue-20261111-9e6e27](issue-20261111-9e6e27.md) | 未着手 | 3.6 | — | コード | RFD | 110_runner_app | UI | 機能改善 | — | 残件有り | 無人で走るRun（番号付きbat）でWARNに気づけない |
| [issue-20261123-3ebab4](issue-20261123-3ebab4.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Env の設定のファイル名の衝突検査（RunManager）が、MetricsLogger の private な置換規則と書き出し先の作り方を写して予測している |
| [issue-20261123-7bc755](issue-20261123-7bc755.md) | 未着手 | 3.5 | — | 文書 | 監査 | 110_runner_app, 319_gui | 開発 | 調査 | — | 未確認 | Logs の折り返し整列テストの失敗を 2 件の課題が扱い、調査中の 61ad60 は原因未特定のまま、RFD0113 の 15f115 が同じ失敗の原因と直す場所を特定している |
| [issue-20261123-ad7829](issue-20261123-ad7829.md) | 未着手 | 3.5 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | Runner の counts と scalar の鍵に、一度も進まない sim_step と、reward・reward_ema と同じ値を返す別名 train_reward・train_reward_ema が残る |
| [issue-20261125-29c3b2](issue-20261125-29c3b2.md) | 未着手 | 3.5 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | metrics.graph の宣言の文法が設計140 に無く、実装は未知の @ や $ の語を黙って捨て、interval を両方 0 以下にすると黙って 1000 step ごとになる |
| [issue-20261125-3cb74a](issue-20261125-3cb74a.md) | 未着手 | 3.5 | — | コード | 監査 | 321_agent_core | 学習基盤 | 不具合 | — | 未確認 | 価値の分解の split（feature_gradient=split）で勾配ノイズの指標を購読すると、内的の Head の state が外的の optimizer に無いため Adam 版が常に未成立になり、ERROR も WARN も無く tag が欠ける |
| [issue-20261115-e5a0eb](issue-20261115-e5a0eb.md) | 未着手 | 3.4 | — | コード | RFD | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 性能改善 | — | 残件有り | 可塑性メトリクスの計算が、500 learn step ごとにメインスレッドで約 2 秒走り、その間は学習が止まる |
| [issue-20261103-1beaf9](issue-20261103-1beaf9.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | PipelineTrainRunnerのイベントで、経験と学習結果が1 stepずれた組になる |
| [issue-20261111-7cc67b](issue-20261111-7cc67b.md) | 未着手 | 3.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | EvalRunner の 1 step が、多重定義 4 つ・-1 の特別な値・bool の引数で、パネルからの 1 step と評価 session の 1 step を切り替えている |
| [issue-20261111-a5370c](issue-20261111-a5370c.md) | 未着手 | 3.3 | — | コード | 監査 | 315_nn | 基盤 | リファクタリング | — | 残件有り | NetworkBody の走査（branch→struct→block→module）が 6 か所に写され、null の検査の有無が場所ごとに違う |
| [issue-20261111-c5c9a7](issue-20261111-c5c9a7.md) | 未着手 | 3.3 | — | コード | 監査 | 315_nn | 基盤 | リファクタリング | — | 残件有り | spectral norm の式（行列化・実効の分母・無効の判定・単位化）が 3〜5 か所ずつ写され、無効の数を返す口も 2 つある |
| [issue-20261123-c455b5](issue-20261123-c455b5.md) | 未着手 | 3.3 | — | コード | 実装 | 311_dev_tools, 110_runner_app | 開発 | 不具合 | — | 残件有り | 既定のスイートの batchrun_fatal_error_handling_test が、前の Run の重みを読む段を持つキューを既定にした bat で、重みが無い段を失敗に数えて落ちる |
| [issue-20261125-16374e](issue-20261125-16374e.md) | 未着手 | 3.3 | — | コード | 監査 | 321_agent_core, 318_rl_core | 学習基盤 | 不具合 | — | 未確認 | checkpoint の読み込みが parameter の device と形を今の Network と突き合わせず、CUDA で保存したものを agent.device=cpu で読むと Network が CUDA に載ったままになる（最初の forward で落ちる見込み） |
| [issue-20261128-c7d223](issue-20261128-c7d223.md) | 未着手 | 3.3 | — | コード | 監査 | 311_dev_tools | 開発 | リファクタリング | — | 未確認 | BTR の結果を Run 形式へ変換するときの評価時点と出力形式を通しで確かめるテストが無い |
| [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | 未着手 | 3.0 | — | コード | RFD | 322_env_individual | Env | 機能追加 | [RFD0085](../../memo/0085_dropmerge_eval_score_axes_10rfd.md) | 残件有り | DropMerge の評価スコア（本家互換スコアほか複数の軸） |
| [issue-20261103-c8fb06](issue-20261103-c8fb06.md) | 未着手 | 3.0 | — | コード | RFD | 312_util, 110_runner_app | 基盤 | 機能改善 | [RFD0071](../../memo/0071_uncaught_fatal_visibility_10rfd.md) | 残件有り | RFD0071: 捕まらない fatal の可視化（worker catch・set_terminate・エラーログ write-through） |
| [issue-20261111-006d9a](issue-20261111-006d9a.md) | 未着手 | 3.0 | — | コード | 監査 | 320_env_core | 学習基盤 | リファクタリング | — | 残件有り | env.cpp で lane 1 本分の Reset・Step の結果を書き込む処理が 4 か所に写され、GetAuxDataList も 3 つあり、数え方と範囲の外の扱いが食い違っている |
| [issue-20261111-3986b7](issue-20261111-3986b7.md) | 未着手 | 3.0 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | テストを回す範囲・回数と、テストの控え・ログの置き場の決まりが無く、実装の時間と .scratch が膨らむ |
| [issue-20261111-4a6b6f](issue-20261111-4a6b6f.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | RunManager のコンストラクタが 277 行あり、評価の設定の解釈・名前の予約・構築・ログ・配線が 1 つの関数に並んでいる |
| [issue-20261111-585e60](issue-20261111-585e60.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 名前やコメントが実際の動きと合わない関数・クラス・設定項目がある（BatchState の IsDone、DefaultAgentFactory、NetworkGraphVizConfig の 2 項目、Runner の 4 関数） |
| [issue-20261111-7c926a](issue-20261111-7c926a.md) | 未着手 | 3.0 | — | コード | 監査 | 315_nn | 基盤 | リファクタリング | — | 残件有り | Network の CopyTo と SoftCopyTo が parameter と buffer を名前で対にする処理を別々に持ち、SoftCopyTo は learn step ごとに名前つきの一覧を 4 つ作る |
| [issue-20261111-be150a](issue-20261111-be150a.md) | 未着手 | 3.0 | — | コード | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | StepTraceColumnSpec の target と display を文字列で持ち、metrics.cpp・gui.cpp・EvalTracePanel.cpp の 3 ファイル 17 か所で文字列を比べている |
| [issue-20261111-c68178](issue-20261111-c68178.md) | 未着手 | 3.0 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | DropMerge のエピソード統計を 1 つ足すと、Env のメンバ・Reset・StepResult の初期化子とメンバ・GetScalar の 5 か所へ同じ形を足すことになる |
| [issue-20261111-ea3483](issue-20261111-ea3483.md) | 未着手 | 3.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | scalar メトリクスの Observer が、イベントの中身を 6〜9 個の引数にばらして 3 段持ち回り、同じ 8 引数のコンストラクタを 4 回書いている |
| [issue-20261123-55c396](issue-20261123-55c396.md) | 未着手 | 3.0 | — | コード | 監査 | 324_dqn, 316_config | 基盤 | リファクタリング | — | 未確認 | DefaultDQNAgentConfig が「任意の ProfiledValue の設定キーが書かれているか」を 3 か所で別々の書き方で調べ、うち 2 か所は常に空の override 接頭辞のための分岐を持つ |
| [issue-20261123-6ffe02](issue-20261123-6ffe02.md) | 未着手 | 3.0 | — | 文書 | 監査 | 318_rl_core, 317_metrics | 学習基盤 | 調査 | — | 残件有り | scalar メトリクスの Observer の 8 引数のコンストラクタと Parsed の struct を、cfe994 と ea3483 の 2 件が別々の直し方で扱っている |
| [issue-20261125-0e98cb](issue-20261125-0e98cb.md) | 未着手 | 3.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | UQE の τ の問い合わせと score の作り方が、行動選択の MakeUQEActionInfo と、skip・価値の分解のための関数に 2 回ずつ書かれ、探索の適用も 4 か所にある |
| [issue-20261125-7c9e47](issue-20261125-7c9e47.md) | 未着手 | 3.0 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | RndNoveltyEstimator の外側に統計と誤差の方式の分岐が 4 か所戻り、倍率を 1 にする条件が、更新する採点と読むだけの採点の 2 つに別々に書かれている |
| [issue-20261125-eb6135](issue-20261125-eb6135.md) | 未着手 | 3.0 | — | 文書 | 監査 | 317_metrics, 318_rl_core | 学習基盤 | 不整合 | — | 未確認 | scalar の宣言の正本である設計140 §6 に、target を省いたときの探索順（出荷設定の 57 件が使う）・属性の既定・ema_alpha だけで EMA になること・$action_info などの event の制限が無く、§7 に動画の設定キーが無い |
| [issue-20261125-d1dfc6](issue-20261125-d1dfc6.md) | 未着手 | 2.8 | — | コード | 監査 | 321_agent_core, 324_dqn | 学習基盤 | リファクタリング | — | 未確認 | DQN の観測の加工（Perception）の組み立ての決まりが Agent・Learner・Actor の 3 ファイルに分かれ、依頼ごとに「作る → 補う → 照合する」を別々に書いている |
| [issue-20261103-9df5c5](issue-20261103-9df5c5.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 110_runner_app, 314_tensor | 学習基盤 | 不具合 | [RFD0146](../../memo/0146_notifier_scoped_detach_10rfd.md) | 残件有り | Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す |
| [issue-20261107-16ffd8](issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-15ba39](issue-20261111-15ba39.md) | 未着手 | 2.7 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 残件有り | EvalPanel のモデル同期が mode を文字列のまま step ごとに解釈し直し、EvalTracePanel は pane の開閉から導ける active_ を 4 通りの問い合わせで合わせている |
| [issue-20261111-17fea4](issue-20261111-17fea4.md) | 未着手 | 2.7 | — | コード | 監査 | 324_dqn, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 方策の種類（policy_type）と分布の表し方（quantile_mode）を文字列のまま持ち回り、数字の別名の受け付け方が関数ごとに違う |
| [issue-20261111-52ed74](issue-20261111-52ed74.md) | 未着手 | 2.7 | — | コード | 監査 | 315_nn, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 可塑性指標の一覧と重みノルムのキーの並びを、nn・DQN・ImageCls の 5〜6 か所で手で合わせている |
| [issue-20261111-5f695d](issue-20261111-5f695d.md) | 未着手 | 2.7 | — | コード | 監査 | 314_tensor | 基盤 | リファクタリング | — | 残件有り | 報酬の scaler と観測の normalizer で、clip と変換の同じ数行が 4〜6 か所にあり、clip が無効のときの扱いが既に食い違っている |
| [issue-20261111-645ecc](issue-20261111-645ecc.md) | 未着手 | 2.7 | — | コード | 監査 | 317_metrics, 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | メトリクス宣言の語（event・step・target）と enum の対応が 5 か所以上に分かれ、解決済みの定義も同じ値を文字列と enum の 2 形で持っている |
| [issue-20261111-6c0bf1](issue-20261111-6c0bf1.md) | 未着手 | 2.7 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | GridMazeEnv が毎 step 変わらない盤面を 1 マスずつ tensor へ書き直し、成功と穴落ちを報酬の値と設定の値が等しいかで判定している |
| [issue-20261111-c90df5](issue-20261111-c90df5.md) | 未着手 | 2.7 | — | コード | 監査 | 319_gui | UI | リファクタリング | — | 残件有り | ForwardedMouseEvent と ForwardedKeyEvent、Panel::OnMouse と OnKey が、型の名前だけ違う写しになっている |
| [issue-20261111-cdc102](issue-20261111-cdc102.md) | 未着手 | 2.7 | — | コード | 監査 | 316_config, 315_nn, 324_dqn | 基盤 | リファクタリング | — | 残件有り | WeightInitConfig などの設定の部品に読み取りの口が無く、使う側が 1 キーずつ読むので、同じ 6〜8 行が十数か所にあり、読み方も食い違っている |
| [issue-20261111-e80c7b](issue-20261111-e80c7b.md) | 未着手 | 2.7 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | DQNActionInfo::GetScalar が 6 種の指標群を 1 つの関数（276 行）で処理し、「指定した行動を除いた最大値」の計算を 2 回写している |
| [issue-20261111-fe23e0](issue-20261111-fe23e0.md) | 未着手 | 2.7 | — | コード | 監査 | 318_rl_core, 320_env_core | 学習基盤 | リファクタリング | — | 残件有り | BatchEnvSpec に「group の数」「group と lane の対応」を返す関数が無く、使う側が三項演算を 19 か所に書き、終端の mask を CPU へ移す処理も関数ごとに書いている |
| [issue-20261108-06a3f0](issue-20261108-06a3f0.md) | 未着手 | 2.6 | — | コード | その他 | 316_config, 322_env_individual | 基盤 | リファクタリング | [RFD0118](../../memo/0118_atari_run_profile_flatten_10rfd.md) | 残件有り | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない |
| [issue-20261108-f0f243](issue-20261108-f0f243.md) | 未着手 | 2.6 | — | コード | RFD | 316_config, 322_env_individual | 基盤 | リファクタリング | [RFD0118](../../memo/0118_atari_run_profile_flatten_10rfd.md) | 残件有り | Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする） |
| [issue-20261103-b2a0b1](issue-20261103-b2a0b1.md) | 未着手 | 2.5 | — | コード | 監査 | 318_rl_core | 学習基盤 | 不具合 | — | 残件有り | ActionSpec::GetShapeが連続行動で次元数ぶんの1を並べた形を返す |
| [issue-20261103-7c9a70](issue-20261103-7c9a70.md) | 未着手 | 2.5 | — | コード | 監査 | 316_config | 基盤 | 不整合 | [RFD0181](../../memo/0181_config_file_layout_10rfd.md) | 残件有り | Agentの切替プロファイル@fast/@balance/@heavyがcommon.txtにあり、020 §3.6の表（agent.txtの役割）と食い違う |
| [issue-20261123-6bb71d](issue-20261123-6bb71d.md) | 未着手 | 2.5 | — | コード | 監査 | 315_nn, 321_agent_core, 317_metrics | 学習基盤 | 不整合 | — | 残件有り | コードの名前・エラー文・メトリクスの tag に CONTEXT.md の避ける語（feature key・mean Q・action churn）が残り、エラー文は実在しない設定キー名 feature_key を示す |
| [issue-20261123-a45289](issue-20261123-a45289.md) | 未着手 | 2.5 | — | 文書 | 監査 | 316_config, 322_env_individual | 基盤 | 調査 | — | 未確認 | RFD0118（Atari の Run プロファイルの整理）を指す課題が、題材の 06a3f0 と RFD の仕事の f0f243 の 2 件あり、範囲が同じで重なっている |
| [issue-20261123-b25c2b](issue-20261123-b25c2b.md) | 未着手 | 2.5 | — | 文書 | 監査 | — | アルゴリズム | 調査 | — | 未確認 | docs/memo.txt にアルゴリズムの思いつき 2 件が、課題にも wiki にも載らないまま 2026-05 から置かれている |
| [issue-20261118-dc5de3](issue-20261118-dc5de3.md) | 未着手 | 2.4 | — | コード | その他 | 317_metrics, 110_runner_app | 基盤 | 機能改善 | — | 残件有り | Atari の Run の列ごとにメトリクスの組と tag がずれ、本線・Taiga 寄せ・易しいゲームの Run どうしを Viewer で重ねて比べられない |
| [issue-20261120-7ac5fa](issue-20261120-7ac5fa.md) | 未着手 | 2.4 | — | コード | 実装 | 110_runner_app, 324_dqn | UI | 不具合 | — | 残件有り | IQN の学習中に HeatMap を既定の選択で開くと taus の不足で Run が停止する |
| [issue-20261103-e6dbb4](issue-20261103-e6dbb4.md) | 未着手 | 2.3 | — | コード | 監査 | 317_metrics | UI | リファクタリング | — | 残件有り | HeatMapとTimeHeatMapのRenderRawが値レンジ・サンプル配置・画像生成を重複して持ち、既に食い違っている |
| [issue-20261103-34d35d](issue-20261103-34d35d.md) | 未着手 | 2.3 | — | コード | RFD | 318_rl_core | 学習基盤 | 不具合 | [RFD0177](../../memo/0177_episode_end_read_semantics_10rfd.md) | 残件有り | `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる |
| [issue-20261111-2c251b](issue-20261111-2c251b.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | Replay の窓の規則 3 クラスが同じ骨格の写しで、PER のスカラーのキーも一覧・条件式・分岐の 3〜4 か所に並んでいる |
| [issue-20261111-360278](issue-20261111-360278.md) | 未着手 | 2.3 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 残件有り | Runner のパネルに描画と表の写しがある（GL キャンバスへ画像を貼る一式が 2 パネルに同文、EvalTracePanel の行の探索 8 か所と再生ツールの並び 3 か所） |
| [issue-20261113-b96b2c](issue-20261113-b96b2c.md) | 未着手 | 2.3 | — | コード | RFD | 324_dqn, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | 観測の正規化の統計が checkpoint に保存されず、統計を使う構成では checkpoint 引き継ぎのたびに 0 から始まる |
| [issue-20261123-79bf13](issue-20261123-79bf13.md) | 未着手 | 2.3 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 観測のキー「base か base.subkey」の解き方が rl.cpp・Replay・probe.cpp・image.cpp の 4 か所にあり、subkey が空のときの扱いと dtype が既に食い違う |
| [issue-20261103-2d66d2](issue-20261103-2d66d2.md) | 未着手 | 2.2 | — | コード | RFD | 324_dqn, 321_agent_core, 325_muzero, 315_nn | 学習基盤 | 調査 | [RFD0134](../../memo/0134_network_lock_audit_10rfd.md) | 残件有り | 999: train/eval mode 整理後の排他精査 RFD（草案） |
| [issue-20261103-5addea](issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-cdac7c](issue-20261103-cdac7c.md) | 未着手 | 2.2 | — | コード | RFD | 321_agent_core | アルゴリズム | 機能改善 | [RFD0148](../../memo/0148_target_update_step_axis_10rfd.md) | 残件有り | target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`） |
| [issue-20261111-4351bc](issue-20261111-4351bc.md) | 未着手 | 2.2 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | DQN の Actor が const の MakeAction の中で 11 個の mutable メンバを書き換えて状態を進め、一時的な出力を結果の public メンバで運んでいる |
| [issue-20261123-c3b5ef](issue-20261123-c3b5ef.md) | 未着手 | 2.2 | — | コード | 監査 | 318_rl_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | Actor 近似の初期優先度で「n 先の hint を待って Estimate に渡す」手順が、Replay の内部と DQN の step 診断の 2 か所に写され、割引の計算はテストにも 3 つ目がある |
| [issue-20261125-5cef87](issue-20261125-5cef87.md) | 未着手 | 2.2 | — | コード | 監査 | 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | 価値の分解の目標値（MakeDecomposedTargets）が、RFD0125 で 1 か所にした Munchausen の評価と hard target の行動選択を写し、分布ごとの違いを quantile_mode の分岐で書いている |
| [issue-20261103-72f807](issue-20261103-72f807.md) | 未着手 | 2.0 | — | コード | 監査 | 321_agent_core | アルゴリズム | 不具合 | — | 残件有り | use_td_clipが損失の前にTDをclampするため、clipを超えたサンプルの勾配が0になる |
| [issue-20261103-78c643](issue-20261103-78c643.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | ObserverFactoryのコンストラクタが331行で、graphの節だけが共通の字句解析を使わずに独自に読む |
| [issue-20261103-785e68](issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-7c8cd7](issue-20261103-7c8cd7.md) | 未着手 | 2.0 | — | コード | RFD | 316_config | 基盤 | リファクタリング | [RFD0181](../../memo/0181_config_file_layout_10rfd.md) | 残件有り | 設定ファイルの配置（Agent 全体・具象 Agent・機能 include・フォルダ） |
| [issue-20261103-7f1789](issue-20261103-7f1789.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 322_env_individual | 基盤 | 機能追加 | [RFD0088](../../memo/0088_nn_adaptive_max_pool_10rfd.md) | 残件有り | 出力の大きさで指定する max pooling（AdaptiveMaxPool2d） |
| [issue-20261103-9bd71e](issue-20261103-9bd71e.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 318_rl_core, 324_dqn, 317_metrics | 学習基盤 | 機能追加 | [RFD0142](../../memo/0142_agent_prev_action_obs_10rfd.md) | 残件有り | 直前行動観測の共通部品（`prev_action` フレーム段） |
| [issue-20261103-a39330](issue-20261103-a39330.md) | 未着手 | 2.0 | — | コード | RFD | 321_agent_core, 324_dqn | アルゴリズム | 機能追加 | [RFD0147](../../memo/0147_iqn_first_priority_quantile_pair_10rfd.md) | 残件有り | RFD0147: IQN初回Learner priorityのquantile-pair方式 |
| [issue-20261111-07f4ed](issue-20261111-07f4ed.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | 「N エピソードに 1 回だけ記録する」状態が Conv2dVisualizationObserver と GraphVizObserver に少し違う形で書かれ、記録を始める条件と終える条件が食い違っている |
| [issue-20261111-0ef722](issue-20261111-0ef722.md) | 未着手 | 2.0 | — | コード | 監査 | 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | probe.hpp・probe.cpp の約 3 分の 1 がどこからも使われず、残る 2 つの Probe クラスと 5 つの IdxExtractor も写しになっている |
| [issue-20261111-269a9f](issue-20261111-269a9f.md) | 未着手 | 2.0 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | DropMergeEnv::Step が 267 行で行動の解釈・物理・裁定・統計・報酬を並べ、行動の解釈・DROP の位置・果物の走査・盤面の受け渡しにも写しがある |
| [issue-20261111-54c7e1](issue-20261111-54c7e1.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 321_agent_core, 315_nn, 319_gui, 110_runner_app | 基盤 | リファクタリング | — | 残件有り | 呼ばれない関数・型・マクロと読まれないメンバ・届かない分岐が、core と Runner に合わせて 1,100 行以上残っている |
| [issue-20261111-90fa5c](issue-20261111-90fa5c.md) | 未着手 | 2.0 | — | コード | 監査 | 110_runner_app | UI | リファクタリング | — | 残件有り | QValuePanel の ComputeQValueHeatMap が 218 行で値の変換・範囲の更新・投票・色の正規化を並べ、データの受け渡しにスレッド越し用の部品と保護の無いメンバが混ざっている |
| [issue-20261112-983afe](issue-20261112-983afe.md) | 未着手 | 2.0 | — | コード | その他 | 324_dqn, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | — | 残件有り | KFMの学習がボスへ行き着くかが偶然に頼る（1階の廊下の位置が観測に無く、Qも内的報酬も左へ進む向きを持てない） |
| [issue-20261114-9468cb](issue-20261114-9468cb.md) | 未着手 | 2.0 | — | コード | 実装 | 324_dqn, 110_runner_app | 学習基盤 | 性能改善 | — | 残件有り | Evaluation Traceを開いている間のActor同期で、RNDの推定器の写しが1回約170 msかかる |
| [issue-20261119-25e4d0](issue-20261119-25e4d0.md) | 未着手 | 2.0 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | 会話が要約されたあと、続ける工程の手順を読み直さずに進めてしまう（読み直しを促す仕掛けが無い） |
| [issue-20261123-22d9bf](issue-20261123-22d9bf.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 110_runner_app, 322_env_individual | 学習基盤 | リファクタリング | — | 未確認 | TrainEvent の action_info が experience.action と同じ実体を重ねて持ち、読む側（Observer・パネル・View）が 2 通りに分かれている |
| [issue-20261123-68723d](issue-20261123-68723d.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | 共通の Replay が Actor の info をすべて slot ごとに保存し、抽選のたびに切り出して学習 device へ送るが、本番で読む箇所が無い（入るのは step の旗にも保存する TempoRL の decision だけ） |
| [issue-20261123-c7c3ae](issue-20261123-c7c3ae.md) | 未着手 | 2.0 | — | 文書 | 実装 | — | 開発 | 不整合 | — | 残件有り | RFD0195 で名前だけを置き換えた既存の 118 行に、AGENTS.md の避ける語（既定葉・腕・旗・地図など）が残る |
| [issue-20261123-cbe34a](issue-20261123-cbe34a.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 317_metrics, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | 共通の Runner が 1 フレームごとに MetricsLogger を flush し（学習スレッドでは毎 step）、設計 140 とアプリが書く「境界でだけ flush」と食い違う |
| [issue-20261127-692b3e](issue-20261127-692b3e.md) | 未着手 | 2.0 | — | ハーネス | RFD | 311_dev_tools | 開発 | 機能改善 | — | 残件有り | RFD・ADR の採番が、ほかのワークツリーと main に無いブランチの番号を見ないので、並行の作業で同じ番号が重なる |
| [issue-20261103-58696c](issue-20261103-58696c.md) | 未着手 | 1.9 | — | ハーネス | RFD | 311_dev_tools | 開発 | 機能追加 | [RFD0112](../../memo/0112_claude_memory_tidy_10rfd.md) | 残件有り | Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`） |
| [issue-20261123-98baff](issue-20261123-98baff.md) | 未着手 | 1.8 | — | 文書 | 実装 | 311_dev_tools | 開発 | 不整合 | — | 残件有り | docs のリンク切れが実装の前から 258 件あり、memo README のリンク切れ検査が空にならないので、新しく切れたリンクを見分けられない |
| [issue-20261124-11428b](issue-20261124-11428b.md) | 未着手 | 1.8 | — | コード | RFD | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | [RFD0201](../../memo/0201_top_episode_memory_sil_lower_bound_10rfd.md) | 未確認 | まれに出た上位のエピソードを Replay と別に持ち続け、外的の Q を実際の収益で下から引き上げる仕組み（上位エピソードの記憶と SIL の下限損失）を DQN 系に足す |
| [issue-20261119-9afcf4](issue-20261119-9afcf4.md) | 未着手 | 1.7 | — | コード | RFD | 311_dev_tools, 321_agent_core | 開発 | リファクタリング | — | 残件有り | テストごとに書かれている代役の Agent 8 個を、基盤のテスト用の口の代役へ寄せる |
| [issue-20261123-8cf911](issue-20261123-8cf911.md) | 未着手 | 1.7 | — | コード | 監査 | 318_rl_core, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | Agent の拡張点 AgentFactory::CreateAgent が Notifier を受け取るが、本番の Agent 4 つはどれも使わず、使うのはテストの代役 1 つ（Observer の登録順を前にするため）だけ |
| [issue-20261103-0ef7b8](issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | RFD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [RFD0108](../../memo/0108_metrics_tag_numbering_10rfd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-61d771](issue-20261103-61d771.md) | 未着手 | 1.6 | — | コード | RFD | 311_dev_tools | 開発 | 機能追加 | [RFD0156](../../memo/0156_cicd_improvement_10rfd.md) | 残件有り | CI/CD 改善（ビルド・テスト自動化、定点ジョブ、LLM 補助） |
| [issue-20261110-adf548](issue-20261110-adf548.md) | 未着手 | 1.6 | — | コード | その他 | 322_env_individual | Env | 機能追加 | — | 残件有り | GridMaze に MiniGrid 互換の課題が無く、難しい探索の検証と DI-engine との比較に使えない |
| [issue-20261113-d38960](issue-20261113-d38960.md) | 未着手 | 1.6 | — | コード | 実装 | 110_runner_app, 324_dqn, 318_rl_core | UI | 性能改善 | — | 残件有り | EvalPanel のモデル同期が UI スレッドで学習ロックを待ち、画面の更新や操作応答を止め得る |
| [issue-20261114-fbcbca](issue-20261114-fbcbca.md) | 未着手 | 1.6 | — | ハーネス | ハーネス保守 | — | 開発 | 機能改善 | — | 残件有り | ADR へのたどり方を改善する（wiki への取り込み遅れと漏れ、入口の方針、サブエージェントの経路） |
| [issue-20261115-3e5b77](issue-20261115-3e5b77.md) | 未着手 | 1.6 | — | 文書 | その他 | 311_dev_tools | 開発 | リファクタリング | — | 残件有り | 設計160（アプリケーションとツール）に開発環境の道具の記述が混ざっているので、開発環境の文書へ分けて整える |
| [issue-20261103-75b8da](issue-20261103-75b8da.md) | 未着手 | 1.5 | — | コード | RFD | 321_agent_core, 318_rl_core | アルゴリズム | 機能改善 | [RFD0168](../../memo/0168_nstep_target_correction_10rfd.md) | 残件有り | n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正） |
| [issue-20261123-2cda10](issue-20261123-2cda10.md) | 未着手 | 1.5 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能追加 | — | 判定不能 | レビューの出し方（2 モデルのクロスレビューを使うか、Claude 側を API とサブスクのどちらで動かすか、どのモデルを使うか）を切り替える設定が無く、そのたびに個人の CLAUDE.md・AGENTS.md・スキル・スクリプトの文面を書き換えている |
| [issue-20261103-e29c92](issue-20261103-e29c92.md) | 未着手 | 1.4 | — | コード | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 残件有り | 名前→factoryのRepositoryと自動登録マクロを4か所で別々に実装し、重複登録の扱いが食い違う |
| [issue-20261103-df24e5](issue-20261103-df24e5.md) | 未着手 | 1.4 | — | コード | 監査 | 312_util | 基盤 | 不具合 | — | 残件有り | AnetExceptionのstack_trace()がwxStackWalkerに頼っていて、Runnerのプロセスでは空になる |
| [issue-20261103-182ca1](issue-20261103-182ca1.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 324_dqn, 315_nn, 320_env_core | 学習基盤 | 性能改善 | [RFD0180](../../memo/0180_core_perf_candidates_202607_10rfd.md) | 残件有り | 2026-07 のコアの詳細レビューで挙がった性能の改善候補 5 件 |
| [issue-20261103-50644a](issue-20261103-50644a.md) | 未着手 | 1.4 | — | コード | RFD | 317_metrics, 320_env_core, 321_agent_core, 318_rl_core, 110_runner_app | UI | 機能追加 | [RFD0184](../../memo/0184_eval_session_step_trace_10rfd.md) | 残件有り | 評価 session の step trace を Run の記録に残す |
| [issue-20261103-52bda0](issue-20261103-52bda0.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 120_metrics_viewer, 311_dev_tools | UI | 機能改善 | [RFD0096](../../memo/0096_metrics_step_axis_interval_10rfd.md) | 残件有り | メトリクスの横軸と間隔を数える単位を分ける |
| [issue-20261103-66f453](issue-20261103-66f453.md) | 未着手 | 1.4 | — | コード | RFD | 110_runner_app, 316_config | 学習基盤 | 機能追加 | [RFD0151](../../memo/0151_run_queue_snapshot_10rfd.md) | 残件有り | Run キュー（計画スナップショット + exe 固定） |
| [issue-20261103-aa8701](issue-20261103-aa8701.md) | 未着手 | 1.4 | — | コード | RFD | 318_rl_core, 321_agent_core, 324_dqn, 320_env_core | 学習基盤 | 機能追加 | [RFD0143](../../memo/0143_serialize_10rfd.md) | 残件有り | Run Serialization（Run 全体 Save / Load） |
| [issue-20261107-2d4932](issue-20261107-2d4932.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 324_dqn | UI | 機能追加 | — | 残件有り | EvalPanelで行動選択のτをランダムと固定で切り替えられず、学んだ行動と学習した価値の判断を見比べられない |
| [issue-20261109-adc5c4](issue-20261109-adc5c4.md) | 未着手 | 1.4 | — | コード | 監査 | 321_agent_core, 320_env_core, 319_gui, 315_nn | 基盤 | リファクタリング | — | 残件有り | Agent・Env・View・NNモジュールの4つのRepositoryが同じ形を別々に実装し、登録と取り出しの関数名もそろっていない |
| [issue-20261110-9d4e5a](issue-20261110-9d4e5a.md) | 未着手 | 1.4 | — | コード | その他 | 311_dev_tools | 開発 | 機能追加 | — | 残件有り | C++ の単純な書き方の問題を機械で検出する仕組み（lint・静的解析）が無い |
| [issue-20261112-4efa57](issue-20261112-4efa57.md) | 未着手 | 1.4 | — | コード | RFD | 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | replay fitの診断（EvaluateReplayFit）が学習の更新の前半を別に書き、分布の表し方での分岐を持っている |
| [issue-20261115-c89660](issue-20261115-c89660.md) | 未着手 | 1.4 | — | コード | その他 | 110_runner_app, 321_agent_core, 318_rl_core | UI | 機能追加 | — | 残件有り | Runner の Conv2dPanel と metrics.image の conv2d に、RND の target と predictor の畳み込みの出力が出ない |
| [issue-20261123-eac719](issue-20261123-eac719.md) | 未着手 | 1.4 | — | コード | 監査 | 318_rl_core, 317_metrics | 学習基盤 | リファクタリング | — | 残件有り | 画像の Observer 4 種は observers（318）にあり、それを作って Notifier へ付けるのは image（317）の ImageProviderFactory で、observers と image が互いを読んでいる |
| [issue-20261124-707776](issue-20261124-707776.md) | 未着手 | 1.4 | — | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | リファクタリング | [RFD0199](../../memo/0199_agents_md_split_by_situation_10rfd.md) | 残件有り | AGENTS.md の全文を毎回読み、ビルド・Run・コミットメッセージなど状況ごとの決まりまで読み込む |
| [issue-20261103-71c4f4](issue-20261103-71c4f4.md) | 未着手 | 1.3 | — | コード | RFD | 315_nn, 317_metrics, 321_agent_core | 学習基盤 | 機能追加 | [RFD0135](../../memo/0135_nn_block_metrics_10rfd.md) | 残件有り | NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定） |
| [issue-20261103-b4e5c9](issue-20261103-b4e5c9.md) | 未着手 | 1.3（引継ぎ） | — | コード | RFD | 316_config, 110_runner_app | 基盤 | リファクタリング | [RFD0155](../../memo/0155_config_root_ownership_alignment_10rfd.md) | 残件有り | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) |
| [issue-20261103-b68e49](issue-20261103-b68e49.md) | 未着手 | 1.3 | — | コード | RFD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [RFD0159](../../memo/0159_input_mapping_joystick_10rfd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261103-f39d10](issue-20261103-f39d10.md) | 未着手 | 1.3 | — | コード | RFD | 318_rl_core, 324_dqn | 学習基盤 | 不具合 | [RFD0141](../../memo/0141_background_eval_snapshot_ordering_10rfd.md) | 残件有り | Background eval の network snapshot 順序保証 |
| [issue-20261119-265a2e](issue-20261119-265a2e.md) | 未着手 | 1.3 | — | コード | RFD | 315_nn | 基盤 | 不整合 | — | 残件有り | 公開の Network のコンストラクタが内部の型 NetworkBody を引数に取り、基盤の外からは呼べない |
| [issue-20261123-b1205d](issue-20261123-b1205d.md) | 未着手 | 1.3 | — | コード | 実装 | 110_runner_app, 322_env_individual | 開発 | リファクタリング | — | 残件有り | Runner と Atari のテストが同じ文字列を重ねて書いている（環境変数の名前 RFD0094_UI_DARK を 3 か所、検証資材のパス rfd0099/frames_210x160.u8 を 2 か所） |
| [issue-20261103-07629c](issue-20261103-07629c.md) | 未着手 | 1.2 | — | コード | RFD | 110_runner_app, 318_rl_core, 319_gui | UI | 不具合 | [RFD0176](../../memo/0176_qvaluepanel_manual_action_state_alignment_10rfd.md) | 残件有り | QValuePanel の Q の表示と、手動で選んだ行動が当たる状態のずれ |
| [issue-20261103-71bb9a](issue-20261103-71bb9a.md) | 未着手 | 1.2 | — | コード | RFD | 311_dev_tools | 開発 | 機能追加 | [RFD0157](../../memo/0157_test_line_coverage_10rfd.md) | 残件有り | 行カバレッジ計測（OpenCppCoverage） |
| [issue-20261103-7227bf](issue-20261103-7227bf.md) | 未着手 | 1.2 | — | コード | RFD | 311_dev_tools, 317_metrics | 開発 | 機能追加 | [RFD0153](../../memo/0153_q_statistics_config_screening_10rfd.md) | 残件有り | Q 値統計による多環境ハイパラ健全性スクリーニング |
| [issue-20261103-8c3c6d](issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [RFD0171](../../memo/0171_replay_eviction_sample_count_10rfd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261103-dbe666](issue-20261103-dbe666.md) | 未着手 | 1.2 | — | コード | RFD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [RFD0163](../../memo/0163_dropmerge_neet_diagnostics_10rfd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261110-d21e5f](issue-20261110-d21e5f.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 325_muzero | 学習基盤 | 不具合 | — | 残件有り | MuZeroのReplayで、truncatedで終わるときに補う価値が、打ち切った次の状態ではなく最後のstep自身のroot_valueになっている |
| [issue-20261111-e90762](issue-20261111-e90762.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core | 学習基盤 | 不整合 | — | 残件有り | PERの抽選が埋まらないときに残りを一様の抽選で埋める代替が、明示した要求を黙って代替しない規約と食い違う |
| [issue-20261115-657248](issue-20261115-657248.md) | 未着手 | 1.2 | — | コード | ハーネス保守 | 311_dev_tools | 開発 | リファクタリング | — | 残件有り | 既存のソースと設定を、最新の規約（include の順・クラス宣言の区切り・関数名の形など）に合わせてまとめて点検し直す |
| [issue-20261122-258ef8](issue-20261122-258ef8.md) | 未着手 | 1.2 | — | コード | RFD | 321_agent_core, 324_dqn, 314_tensor | 学習基盤 | 不具合 | — | 残件有り | 統計つきの観測の正規化で、Actor が Agent の排他の外で統計を読み、学習と並んで走る評価の Actor で読み書きが競合する |
| [issue-20261103-15f115](issue-20261103-15f115.md) | 未着手 | 1.1 | — | コード | RFD | 110_runner_app, 319_gui | UI | 機能改善 | [RFD0113](../../memo/0113_runner_per_monitor_dpi_10rfd.md) | 残件有り | Runner を Per-Monitor V2 の DPI 対応で動かす RFD（manifest の gdiScaling を外す） |
| [issue-20261103-1edf84](issue-20261103-1edf84.md) | 未着手 | 1.1 | — | コード | RFD | 321_agent_core, 324_dqn, 315_nn | アルゴリズム | 機能追加 | [RFD0172](../../memo/0172_bootstrapped_multihead_q_10rfd.md) | 残件有り | Bootstrapped DQN 型の複数の Head（1 本の Run の中に別々の価値関数を持つ） |
| [issue-20261103-206933](issue-20261103-206933.md) | 未着手 | 1.1 | — | コード | RFD | 110_runner_app | 学習基盤 | 機能追加 | [RFD0175](../../memo/0175_seed_population_early_stop_10rfd.md) | 残件有り | 多数の seed で打席を稼ぐ（条件付きの打ち切りで、火が付かなかった Run を早く止める） |
| [issue-20261103-51220d](issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | RFD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [RFD0144](../../memo/0144_episode_forensics_10rfd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261116-cc5942](issue-20261116-cc5942.md) | 未着手 | 1.1 | — | コード | その他 | 318_rl_core, 321_agent_core, 314_tensor, 324_dqn | 学習基盤 | 性能改善 | — | 残件有り | 同じデータを何度もdevice間で転送している箇所を洗い出して減らす（Replayのobsとnext_obsの重なるフレーム、RNDの採点とActorが別々に送る観測など） |
| [issue-20261128-29cd97](issue-20261128-29cd97.md) | 未着手 | 1.1 | — | コード | RFD | 314_tensor, 318_rl_core, 321_agent_core | 学習基盤 | リファクタリング | — | 未確認 | 直近の系列（観測・行動・報酬・開始の印）を並びの外のバッファに置き、Forwarder の部品がいつでも見る形になっておらず、フレーム積層と直前の行動の観測が系列を別々に持つ |
| [issue-20261103-6f6c73](issue-20261103-6f6c73.md) | 未着手 | 1.0 | — | コード | RFD | 321_agent_core | アルゴリズム | 機能追加 | [RFD0183](../../memo/0183_temporl_skip_followups_10rfd.md) | 残件有り | TempoRL の後続（長い j の不確かさ・閉ループの割り込み・全 k の一括学習） |
| [issue-20261103-c16aab](issue-20261103-c16aab.md) | 未着手 | 1.0 | — | コード | RFD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [RFD0154](../../memo/0154_scalar_threshold_rate_10rfd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261105-f4b870](issue-20261105-f4b870.md) | 未着手 | 1.0 | — | コード | その他 | 311_dev_tools, 315_nn | アルゴリズム | 調査 | — | 判定不能 | LLM が NN の重みを直接読み書きする試み（Weight-Space Programming）を anet で試す |
| [issue-20261111-d1d12b](issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261119-cb9a0f](issue-20261119-cb9a0f.md) | 未着手 | 1.0 | — | コード | その他 | 324_dqn, 321_agent_core | アルゴリズム | 機能追加 | — | 残件有り | RNDの観測の正規化に、学習の前に一様乱数の行動で統計を作る前段（Burdaら）が無い |
| [issue-20261122-4b9e2d](issue-20261122-4b9e2d.md) | 未着手 | 1.0 | — | コード | RFD | 315_nn | 基盤 | 不整合 | — | 残件有り | NN の入口が、離散の Grid の入力で、stack 軸とクラスの次元を、定義に書かれないままチャネルへ畳む |
| [issue-20261122-d9499f](issue-20261122-d9499f.md) | 未着手 | 1.0 | — | コード | RFD | 316_config, 324_dqn, 314_tensor, 315_nn | 基盤 | 機能改善 | — | 残件有り | 見直したあとの設定の形と通知の決まりに沿わない設定キーと通知が、RND の外に残る（全体は調べていない） |
| [issue-20261103-cfe994](issue-20261103-cfe994.md) | 未着手 | 0.9 | — | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | イベント種別ごとにObserver interface・RunnerScopedラッパ・Notifierメソッド・MetricsLog派生を複製している |
| [issue-20261103-c5c791](issue-20261103-c5c791.md) | 未着手 | 0.9 | — | コード | RFD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [RFD0179](../../memo/0179_rfd0097_performance_remeasurement_10rfd.md) | 残件有り | RFD0097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| [issue-20261103-f33167](issue-20261103-f33167.md) | 未着手 | 0.9 | — | コード | RFD | 324_dqn, 321_agent_core, 315_nn | アルゴリズム | 機能追加 | [RFD0107](../../memo/0107_noisy_gain_lane_ladder_10rfd.md) | 残件有り | 学習の Actor の lane ごとに NoisyNet のノイズの強さを変える（ノイズのラダー） |
| [issue-20261103-f40ae8](issue-20261103-f40ae8.md) | 未着手 | 0.9 | — | コード | RFD | 317_metrics, 314_tensor, 318_rl_core | UI | 性能改善 | [RFD0056](../../memo/0056_timehistogram_fine_grid_10rfd.md) | 残件有り | TimeHistogram fine-bin保持・描画時集約 |
| [issue-20261103-322a83](issue-20261103-322a83.md) | 未着手 | 0.8 | — | コード | RFD | 315_nn | アルゴリズム | 機能追加 | [RFD0150](../../memo/0150_residual_alpha_gate_10rfd.md) | 残件有り | 標準 Spectral Normalization と residual identity 開始を両立する α ゲート |
| [issue-20261103-53ed7a](issue-20261103-53ed7a.md) | 未着手 | 0.8 | — | コード | RFD | 110_runner_app, 319_gui | UI | 機能追加 | [RFD0186](../../memo/0186_runner_theme_switch_10rfd.md) | 残件有り | 起動中の Runner を OS のテーマ（ライト／ダーク）の切替に追従させる |
| [issue-20261103-afe841](issue-20261103-afe841.md) | 未着手 | 0.8 | — | コード | RFD | 321_agent_core, 324_dqn, 318_rl_core | 学習基盤 | 機能追加 | [RFD0145](../../memo/0145_staged_batch_replay_schedule_10rfd.md) | 残件有り | DefaultDQN Staged Batch / Replay Schedule |
| [issue-20261103-b13ed2](issue-20261103-b13ed2.md) | 未着手 | 0.8 | — | コード | RFD | 311_dev_tools | 開発 | 機能改善 | [RFD0185](../../memo/0185_release_packaging_cmake_10rfd.md) | 残件有り | リリース物の組み立てを CMake（install・CPack・ワークフロー preset）へ移し、手元でも同じコマンドで作れるようにする |
| [issue-20261122-4c0af5](issue-20261122-4c0af5.md) | 未着手 | 0.8 | — | コード | RFD | 318_rl_core, 314_tensor, 321_agent_core | 学習基盤 | リファクタリング | — | 残件有り | 観測をためる仕組み（フレーム積層の履歴、Replay の観測の置き場、補助学習の rollout の置き場）が別々に実装されていて、共通の置き場の部品へ載せ替えていない |
| [issue-20261103-969184](issue-20261103-969184.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0174](../../memo/0174_llm_actor_10rfd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261113-6f954f](issue-20261113-6f954f.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 324_dqn | 学習基盤 | リファクタリング | — | 残件有り | Learner の診断の購読が、Run で使う診断の宣言と、呼び出しごとの要求を 1 つの口で兼ねている |
| [issue-20261116-948957](issue-20261116-948957.md) | 未着手 | 0.6 | — | コード | その他 | 315_nn, 324_dqn | 基盤 | 機能追加 | — | 残件有り | ImageClsで外部の学習済みモデルの重みを読み込んで転移学習できるようにする（今はanetが保存した同じ構造の重みしか読めない） |
| [issue-20261116-cc631d](issue-20261116-cc631d.md) | 未着手 | 0.6 | — | コード | その他 | 322_env_individual | Env | 機能追加 | — | 残件有り | ImageClsでFood-101以外のデータセット（CIFAR-100・ImageNetなど）を学習・評価に使えるようにする |
| [issue-20261119-497fce](issue-20261119-497fce.md) | 未着手 | 0.6 | — | コード | その他 | 324_dqn, 318_rl_core | アルゴリズム | 機能追加 | — | 残件有り | 内的の価値を非エピソードにしても、n-stepの和がエピソードの終わりで切れ、Burdaらのように内的の収益がエピソードをまたいで続かない |
| [issue-20261119-628657](issue-20261119-628657.md) | 未着手 | 0.6 | — | コード | その他 | 324_dqn | アルゴリズム | 機能追加 | — | 残件有り | 価値の分解で、合わせた Q に 1 本の損失を掛ける MEME 型を feature_gradient の第 3 のモードとして試す |
| [issue-20261122-ba42c9](issue-20261122-ba42c9.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 314_tensor | 学習基盤 | 機能改善 | — | 残件有り | RND の観測の加工（byte を float に直す、自前の統計で正規化する）が、Q の側の Forwarder の並びと別に書かれている |
| [issue-20261103-145ee2](issue-20261103-145ee2.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [RFD0152](../../memo/0152_metric_targeted_hyperparam_control_10rfd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261103-4a2b5e](issue-20261103-4a2b5e.md) | 未着手 | 0.5 | — | コード | RFD | 315_nn, 321_agent_core | 基盤 | リファクタリング | [RFD0165](../../memo/0165_nn_head_projection_separation_10rfd.md) | 残件有り | NN Head の最終射影と出力変換の責務分離 |
| [issue-20261103-6c40b8](issue-20261103-6c40b8.md) | 未着手 | 0.5 | — | コード | RFD | 315_nn | アルゴリズム | 機能追加 | [RFD0166](../../memo/0166_noisynet_effective_weight_spectral_norm_10rfd.md) | 残件有り | NoisyNet のノイズ込み重みに対する Spectral Normalization |
| [issue-20261103-f993bc](issue-20261103-f993bc.md) | 未着手 | 0.5 | — | コード | RFD | 321_agent_core, 324_dqn, 325_muzero, 315_nn | 学習基盤 | リファクタリング | [RFD0164](../../memo/0164_nn_forward_amp_consolidation_10rfd.md) | 残件有り | NN 実行設定への AMP の集約（全 Agent 対象） |
| [issue-20261115-d3e094](issue-20261115-d3e094.md) | 未着手 | 0.5 | — | コード | その他 | 324_dqn, 321_agent_core | アルゴリズム | 調査 | — | 残件有り | RNDの改良版のSND（targetも自己教師で学ぶ蒸留）を、RNDの評価が落ち着いたあとに試すかを検討する |
| [issue-20261124-f9f964](issue-20261124-f9f964.md) | 未着手 | 0.5 | — | コード | RFD | 315_nn | 基盤 | リファクタリング | — | 未確認 | nn.hpp（RL に依らない層）の NN 呼び出し情報に、行の開始の印 episode_start の名前が残り、Forwarder の口と渡し方がそろわない |
| [issue-20261126-4c7b4e](issue-20261126-4c7b4e.md) | 対応中 | 1.6 | [issue-20261125-822a3e](issue-20261125-822a3e.md) | ハーネス | ハーネス保守 | 311_dev_tools | 開発 | 機能改善 | [RFD0203](../../memo/0203_wiki_page_yaml_header_10rfd.md) | 残件有り | サマリのページに YAML ヘッダが無く、Obsidian で題名や材料の日付を列にした一覧を作れない |
| [issue-20261103-ca1a85](issue-20261103-ca1a85.md) | 未着手 | 1.6 | [issue-20261103-1beaf9](issue-20261103-1beaf9.md) | コード | 監査 | 318_rl_core | 学習基盤 | リファクタリング | — | 残件有り | SerialとPipelineのDoStepが初期化・経験の組み立て・通知・カウンタ更新を別々に持ち、既に食い違っている |
| [issue-20261103-124da5](issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0167](../../memo/0167_smdp_transition_duration_10rfd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-9bd71e](issue-20261103-9bd71e.md), [issue-20261103-75b8da](issue-20261103-75b8da.md), [issue-20261107-949e8a](issue-20261107-949e8a.md) | コード | RFD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [RFD0173](../../memo/0173_intrinsic_reward_novelty_10rfd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261103-6661d2](issue-20261103-6661d2.md) | 未着手 | 1.3 | [issue-20261103-f39d10](issue-20261103-f39d10.md), [issue-20261103-2d66d2](issue-20261103-2d66d2.md) | コード | RFD | 318_rl_core, 324_dqn, 110_runner_app | 学習基盤 | 機能追加 | [RFD0136](../../memo/0136_best_checkpoint_10rfd.md) | 残件有り | best checkpoint の自動保存（eval 自己ベスト更新時の重み保持） |
| [issue-20261115-b02db6](issue-20261115-b02db6.md) | 未着手 | 1.3 | [issue-20261103-b4e5c9](issue-20261103-b4e5c9.md) | コード | その他 | 316_config, 324_dqn, 315_nn | 基盤 | リファクタリング | — | 残件有り | RND の Head の出力次元 intrinsic.rnd.output_dim が rnd の直下にあり、Head の初期値 intrinsic.rnd.head.init.* と置き場が分かれている |
| [issue-20261103-13bc20](issue-20261103-13bc20.md) | 未着手 | 0.8 | [issue-20261103-3e1cf6](issue-20261103-3e1cf6.md) | コード | RFD | 322_env_individual | Env | 機能追加 | [RFD0086](../../memo/0086_dropmerge_macro_actions_10rfd.md) | 残件有り | DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化） |
| [issue-20261108-227b05](issue-20261108-227b05.md) | 未着手 | 0.8 | [issue-20261103-736e39](issue-20261103-736e39.md) | コード | RFD | 324_dqn, 321_agent_core | アルゴリズム | 調査 | — | 残件有り | RNDのpredictorを周期的に初期化してbonusを続かせる案は、GridMazeとLunarLanderの試行で利得が無く、実装を戻した |
| [issue-20261103-578986](issue-20261103-578986.md) | 未着手 | 0.6 | [issue-20261103-b68e49](issue-20261103-b68e49.md) | コード | RFD | 318_rl_core, 319_gui, 110_runner_app | 学習基盤 | 機能追加 | [RFD0160](../../memo/0160_input_target_train_forced_action_10rfd.md) | 残件有り | 操作入力の入力先切替と Train 側の強制行動（学習への反映） |
| [issue-20261103-5a783c](issue-20261103-5a783c.md) | 未着手 | 0.6 | [issue-20261103-124da5](issue-20261103-124da5.md), [issue-20261103-75b8da](issue-20261103-75b8da.md) | コード | RFD | 321_agent_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0162](../../memo/0162_decision_aligned_nstep_10rfd.md) | 残件有り | 次の判断まで遡る n-step（判断単位の可変長 n-step） |

着手できる 178 / 待ち 10 / 保留 0 / 次版 0 / 解決 0

## 解決待ち

AI の判定が「残件無し」で、解決でない課題。解決にするかは開発者が決める（コミットの準備で持ちかける）。

| ID | 状態 | AIの判定 | 判定日 | 課題 |
|---|---|---|---|---|
| [issue-20261107-949e8a](issue-20261107-949e8a.md) | 対応中 | 残件無し | 2026-11-23 | AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない |
| [issue-20261117-eedd0c](issue-20261117-eedd0c.md) | 調査中 | 残件無し | 2026-11-23 | Evaluation Trace の単一行選択テストが合成ドラッグ後に失敗する |
| [issue-20261103-785e68](issue-20261103-785e68.md) | 未着手 | 残件無し | 2026-11-25 | anet-core の外で実装する拡張点とコード配置の規約 |

## 再開候補

解決の日付以後に（解決の日付が無ければ日付によらず）AI の判定が「残件有り」になった解決の課題。状態を戻すかは開発者が決める。

該当する課題はありません。

## 課題の無い RFD

docs/memo 直下の RFD のうち、どの課題の rfd からも指されないもの。その RFD の仕事の課題を作るか、題材の課題の rfd に書く。

該当する RFD はありません。

## 対応内容のある未着手の課題

状態が未着手なのに、対応内容に行がある課題。行の置き場（対応でない行は anet-create-issue の「起票のあとの追記」の節へ）と状態を確かめる。開発者が未着手へ戻した課題のように、どちらも正しければそのままでよい（状態は開発者が決める）。

| ID | 起票日 | 課題 |
|---|---|---|
| [issue-20261115-b02db6](issue-20261115-b02db6.md) | 2026-11-15 | RND の Head の出力次元 intrinsic.rnd.output_dim が rnd の直下にあり、Head の初期値 intrinsic.rnd.head.init.* と置き場が分かれている |
