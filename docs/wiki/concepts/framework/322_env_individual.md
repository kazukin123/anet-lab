---
title: 322 Env個別実装
page_kind: フレームワーク
aliases:
  - 322_env_individual
  - GridMaze
  - seed_mode
  - DropMerge
  - NOOP
  - noop_penalty
  - RAMメトリクス
  - distinct_seen
description: 共通Env contract上のCartPole、GridMaze、LunarLander、DropMerge、ImageCls、Atariなどの具象環境の所在をまとめる。
updated: 2026-11-26
---
# 322 Env個別実装

[anet-lab](../../../../README.md) › [wiki の目録](../../index.md) › [フレームワーク](../../index.md#フレームワーク)

## 概要・責務

- 共通Env contract上のCartPole、GridMaze、LunarLander、DropMerge、ImageCls、Atariなどの具象環境の所在をまとめる。（出典: [環境](../../../design/120_environments.jp.md)、§1.3・§2・§3）。
- GridMazeは到達できる迷路の生成とseed採番の単位を所有し、LunarLanderViewは推力・風・運動の表示切替を所有する。具象Env・Viewの条件は共通基盤と分けて引く（出典: [環境](../../../design/120_environments.jp.md)、§7.4・§7.5）。
- DropMergeは行動モードで選択命令を判定し、NOOPへの罰をEnvのStepあたり1回加算する。Directの列0や実行されないDROPには加算せず、NOOPが移動へ変換される場合は加算する（出典: [環境](../../../design/120_environments.jp.md)、§7.6）。
- 環境ごとの実装と初期化関数は`src/env/`に置き、基盤の公開ヘッダを使う（出典: [ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md)、「ライブラリとアプリ」）。
- Envの公開初期化関数は、何度呼ばれてもEnvのfactoryとViewのcreatorを1回だけ登録し、途中で失敗したら以後の呼び出しも失敗する（出典: [環境](../../../design/120_environments.jp.md)、§3）。
- AtariEnvは観測を1枚ずつ返す。重ねる枚数と対象のキーはDefaultDQNの`frame_stack.*`が持ち、行動選択用のForwarderの`FrameStack`がActorの履歴を重ね、ReplayBufferが同じ枚数とキーでSampleを組み直す（出典: [Atari環境](../../../design/220_atari_env.jp.md)、§2）。

## 成立の経緯

- 2026-06-25: ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）（出典: [RFD0022](../../../memo/done/0022_imagecls_augmentation_10rfd.md)、題・目的）。
- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../../adr/0009-imagecls-batch-env-seam.md)、本文）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、題・目的）。
- 2026-07-22: DropMerge NoLegal 裁定の blocked-persistence 化（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、題・目的）。
- 2026-07-25: DropMergeのNoDropTimeoutをtruncationではなく敗着(done+罰)とする（出典: [ADR 0014](../../../adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md)、本文）。
- 2026-08-13: RFD0051: ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理（出典: [RFD0051](../../../memo/done/0051_atari_ale_env_10rfd.md)、題・目的）。
- 2026-08-14: ALE は外部ビルド木を ALE_ROOT で参照し、Atari env はオプショナルビルドにする（GPL 隔離）（出典: [ADR 0025](../../../adr/0025-ale-external-build-reference-and-license.md)、本文）。
- 2026-08-14: AtariEnv は SingleDiscreteEnv として登録し、前処理は env 内自前・frame stack は既存 stucker/RB に委ねる（後に ADR 0086 で、DefaultDQN の重ねは Perception の FrameStack の定義を使う形になった。出典: [ADR 0026](../../../adr/0026-atari-env-single-seam-own-preprocessing.md)、本文）。
- 2026-08-21: Codex 引継ぎ: DropMergeEnv の NoLegalDrop 終端追加（出典: [RFD0005](../../../memo/done/0005_no_legal_drop_10rfd.md)、題・目的）。
- 2026-08-21: DropMergeEnv 直前行動観測(prev-action trio + DROP 列マーカー・暫定)（出典: [RFD0040](../../../memo/done/0040_dropmerge_prev_action_obs_10rfd.md)、題・目的）。
- 2026-08-21: Envインスタンスname RFD（出典: [RFD0037](../../../memo/done/0037_env_instance_name_10rfd.md)、題・目的）。
- 2026-08-21: Envログ name prefix 自動付与 RFD（出典: [RFD0038](../../../memo/done/0038_env_log_auto_prefix_10rfd.md)、題・目的）。
- 2026-08-21: LunarLanderEnv obs_include_action（直前 action の観測化・暫定）（出典: [RFD0032](../../../memo/done/0032_lunarlander_obs_include_action_10rfd.md)、題・目的）。
- 2026-09-03: 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う（出典: [ADR 0034](../../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、本文）。
- 2026-09-04: ImageClsを含む評価episode集約を確認した（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、F4. env の確定値は「終端した Step が返った直後」にしか読めない）。
- 2026-09-17: GridMazeのmax_stepsとepisode長の扱いを整理した（出典: [RFD0075](../../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md)、エピソード長が framework の一級市民になっていない）。
- 2026-09-19: Actorカタログ移行後の各Envとの接続を回帰確認した（出典: [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md)、題・目的）。
- 2026-09-24: AtariEnv のゲーム固有 RAM メトリクスは番号付き汎用キーで参照し、意味はゲーム別の RAM 定義が持つ（出典: [ADR 0046](../../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、本文）。
- 2026-09-24: Atariのゲーム固有イベントを番号付きRAMメトリクスとして記録した（出典: [RFD0082](../../../memo/done/0082_atari_ram_progress_metrics_10rfd.md)、題・ゴールG1〜G4）。
- 2026-10-20: 各Envの遷移の値をStepの結果へ移し終端の読み取りをそろえた（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§4.3.5〜4.3.7・§4.4 クラス一覧）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。
- 2026-10-23: Atari観測の縮小方式を設定で選ぶ仕様を精査した。実装と数値・性能の受入は後続工程である（出典: [RFD0099](../../../memo/done/0099_atari_resize_mode_10rfd.md)、§1・§8）。
- 2026-10-24: GridMaze の迷路生成に到達可能性の保証を足し、解けない seed は派生 seed で決定的に引き直す。解ける seed の迷路は変えない（出典: [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、題・§4）。
- 2026-10-24: GridMazeの迷路seedを指定値またはエピソード・プロセス単位で選ぶよう整理した（出典: [GridMaze seed_mode RFD](../../../memo/done/0102_gridmaze_seed_mode_10rfd.md)、§1・§4）。
- 2026-11-04: DropMergeのNOOPへの罰を既存の行動モードの判定へそろえ、Directの列0のDROPへ罰が付く不具合を修正した（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§1.2、[環境](../../../design/120_environments.jp.md)、§7.6）。
- 2026-11-12: 各Envの設定の検証を、構築時に設定の型で行うようにし、AtariEnvの`display_screen`とworker threadの併用のWARNをRunで1回にした（出典: [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md)、§4 D5・§5.5）。
- 2026-11-12: AtariのRAMメトリクスに、1ゲームで観測した値の種類数（`distinct_seen`）と、マスクの中のビットが立った数・落ちた数（`bit_rise_count`・`bit_fall_count`）を数える畳み方を足した。RAM定義の読み取りを1か所にまとめて畳み方の名前と種別を1つの表にし、評価の上書きを拒否する専用の検査を消し、縮小方式は設定を読むときに1回だけ検証するようにした（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、題・§1・§4 D1〜D5・D9・D10・D16）。
- 2026-11-19: Envのフォルダをsrc/env/<名前>（英小文字、末尾の数字なし）にし、ターゲットをanet-env-<名前>にそろえた。EnvのテストがRunを組むときは、基盤のテスト用の口の代役のAgentを使う（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D3・D4・D14）。
- 2026-11-20: 各Envの初期化関数を、何度呼んでも登録を1回だけ行う形にした。Envのテストは、Envのfactoryを直接登録せずに初期化関数を呼ぶ（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5 D11・§6.5・§6.7）。
- 2026-11-21: 各Envのエラー・WARN・INFOが示す設定キーを完全キーにし、ImageClsのデータセットの項目を`ImageDataset.[<名前>]`の完全キーで示すようにした（出典: [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md)、§5 D2・D6・D21）。
- 2026-11-23: 各Envの観測のspecに値の尺度を宣言し、LunarLander・CartPoleのfloatのvectorを`Unscaled`、GridMaze・DropMerge・Atariを`Scaled`とした（出典: [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md)、§6.3 D18）。

## 選ばなかった形

- NOOPの判定修正と同時にDropMergeの罰の値・行動集合・報酬設計を変える形 — 範囲外: 行動モードに意味をそろえる修正とは別の関心。再開条件: 学習上の報酬設計を見直す実験を選んだとき（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§1.3）。
- OpenCV依存の追加・84専用制限・OpenCV全分岐の厳密再現 — 不採用: 自前前処理の境界と縮小全範囲の設定を保ち、anetの数値規則と固定画像の許容差を定める（出典: [RFD0099](../../../memo/done/0099_atari_resize_mode_10rfd.md)、§7）。
- 詰み判定のために内部の盤面を進めて裁定する方式 — 不採用: 問題設定が複雑化し、人間のプレイ感覚から乖離する（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、決定経緯の要約）。
- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 0176_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、§5・§4.1 D1/D2）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- ゲーム別RAM番地・畳み方をコードの固定表に持つ形 — 不採用: 番地を試すたびに再ビルドが要り、解析上の選択を固定する（出典: [ADR 0046](../../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- RAM番地またはラベルをmetricsキーへ直接出す形 — 不採用: ゲームやイベントの変更でmetrics定義を書き換えることになる（出典: [ADR 0046](../../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- frame skip窓のあとだけRAMを観測する形 — 不採用: 窓途中とreset中の遷移を回数系で取りこぼす（出典: [ADR 0046](../../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- ImageCls の decode cache を bounded LRU にする形 — 不採用: epoch シャッフルでは再利用距離が平均約 0.75N で、容量 K=N/2 でも LRU の hit 率は約 15% にとどまり、同じ容量の固定ピン留め（hit 率 50%）に常に劣る（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、12.11 D9解決）。
- ImageCls の評価を固定の部分集合で行う形 — 不採用: 毎回同じ画像だけを見ることになる（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、12.2 維持された主要判断・12.8 D5解決）。
- 評価の終わりを driver 側（Runner）で sample 数を数えて決める形 — 不採用: 端数の padding は物理的に Source 側で必須なので処理が二重になり、汎用の EvalRunner が valid 件数という dataset の語彙を持つことになる（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、12.8 D5解決）。
- EvalPanel の Source の乱数列を同じタグの configured eval にそろえる形（option 2） — 不採用（棄却）: GUI の操作で cursor がすぐずれて整列は保てず、同じ seed だから同じ対象を見ているという偽の手がかりだけが残る（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、12.12 D10解決）。
- NoDropTimeout を truncated にする形 — 不採用（却下）: ショットクロックは観測にあり、クロックを進めるのは agent 自身の NOOP なので、終端は agent の支配下の Markov な帰結で done が意味論的に正しい。flag の切り替えで行うと観測の次元まで変わり、一軸の A/B が壊れる（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、決定経緯の要約 8）。
- missed_resignation（persistence 条件付きの投了失敗）メトリクス — 不採用（却下）: use_instant_drop では次の行動選択の時点の persistence は高々 1 frame で、Phase 2 の後は該当する状態が構造的に空集合になり、厳密な投了率には反実仮想の物理シミュレーションが要る（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、決定経緯の要約 2）。
- timeout 時の再分類・裁定の上限・no_legal_adjudication_timeout — 不採用（導入しない）: blocked-persistence の裁定では到達できない dead 機構になり、瞬間の candidate による再分類は振動して時々 legal が開く盤面を NoLegal へ誤分類する（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、決定経緯の要約 5）。
- GridMaze で解けない迷路を止めるだけにする、通路を掘る、生成アルゴリズムを変える — 不採用: 自動 seed の Run が途中で止まる／迷路の分布が偏る／解ける seed の迷路が変わる（出典: [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、§8）。
- GridMaze の迷路の難しさ（最短路の長さ、穴の近さ）の制御 — 範囲外: 別の関心。再開条件: 難しさの違いで学習の差を見たくなったとき（出典: [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、§8）。
- GridMazeの個体別・BatchEnv別の一度だけの採番 — 不採用: 学習・評価・GUIを同じプロセスの入力seedへそろえる目的を満たさない（出典: [GridMaze seed_mode RFD](../../../memo/done/0102_gridmaze_seed_mode_10rfd.md)、§4 D2・§8）。
- フォルダとターゲットの名前をenvs（複数）にする形 — 不採用: include・名前空間・設定キーが単数で、フォルダだけが食い違う（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- Envの公開ヘッダの名前を、フォルダ名と同じ英小文字にそろえる形 — 不採用: ソース配置のゴールに効かず、参照を書き換えるだけになる。大文字小文字だけを変える改名は、WindowsとGitで事故になりやすい（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- Envのテストと系統のテストが、互いのライブラリを使ってよいとする形 — 不採用: 置き場が「どちらが主題か」で割れ、相手が壊れると使う側のテスト実行体が作れない（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.3）。
- 初期化関数を1回しか呼べない形にして、テストを直す形 — 不採用: `InitRL`が中でNNの初期化関数を呼ぶので2回呼ぶだけで止まり、直接の呼び出し46か所の書き換えが要る（出典: [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md)、§5.1）。
- RAMメトリクスで、Run全体で見た値の和集合を作る形（envの個体の一生の種類数、metrics側の累積、ログと後処理） — 不採用: 作らない判断で、1ゲームごとの値のmeanとmaxで後から集計できる。lane別の値は和集合より小さく出る、汎用のmetricsの記法に時間方向の新しい演算が入る、人が読むログを道具が読む、という難点もある（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§1 のN4・§4 D7・§4.1 A〜C）。
- ビットを数える語を、マスクの中が全部0からどれかが1になった回数や、マスクを掛けた値が指定の値に入った回数にする形 — 不採用: すでに同じ種類を持っているときの取得を数えない（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4.1 D・E）。
- ビットの語を足さず、番地の側にマスクを付けて既存の語を使う形 — 不採用: 複数ビットのマスクでは値の大小しか見ず、鍵4本分を1つの定義で数えられない（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4.1 I）。
- 立つ側の語だけを足す形 — 不採用: 敵を倒した回数と扉の回数を、得点から推し量るままになる（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4.1 G）。
- マスクを16進だけで書く形 — 不採用: 文書と設定の例のビットの値（鍵2、剣32）をそのまま写せなくなる。10進と16進の両方を開発者が選んだ（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4.1 K）。
- 評価の上書きを拒否する専用の検査を残す形（判定だけ縮める、各envの設定が自分の上書きだけを探す） — 不採用: 同じ誤りを、タグごとの読まれない設定キーの検査と2つの仕組みで止めることになる（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4 D10・§4.1 F・J）。
- レベルをまたいだときに部屋を区別する語（番地2つの組を数える語） — 先送り: 実験の結果が要る。再開条件: 2レベル目に入るRunが出たとき（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4 D14）。
- RAMメトリクスの番号の枠を広げ、5番以降のscalarのtagを作る形 — 範囲外: tagの番号を全体で振り直すRFD0108の関心（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§1 のN1）。
- ALE をリポジトリに取り込んで常時サブプロジェクトとしてビルドする — 不採用: ALE のビルドに必要な vcpkg toolchain が本体全体に入り、外部コードの同梱範囲も曖昧になる（出典: [ADR 0025](../../../adr/0025-ale-external-build-reference-and-license.md)、Considered Options）。
- ALE 内蔵の frame skip だけで前処理を済ませる — 不採用: 連続2フレームの最大値合成を省くことになり、スプライトの点滅対策を満たせない（出典: [ADR 0026](../../../adr/0026-atari-env-single-seam-own-preprocessing.md)、Considered Options）。
- ImageDataSource に拡張用フラグと乱数器を持たせる形 — 不採用。環境側にも乱数器があり、学習画像の拡張に使う乱数源が二つになる（出典: [RFD0022](../../../memo/done/0022_imagecls_augmentation_10rfd.md)、確定した設計判断）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../../reports/stats/trend.md)の「テスト充足」を見る。

- 縮小方式追加では既定の学習系列のchecksumを保持し、84×84の各参照画像で全画素差1以下・99.9%以上一致を確認する。64 envの速度比中央値0.98以上を要求し、対照測定の揺れが1%を超える場合は判定保留にする（出典: [RFD0099](../../../memo/done/0099_atari_resize_mode_10rfd.md)、§6）。
- 直前action観測のON/OFFを同seed・同action列で並走し、先頭8次元を全stepで一致させる（出典: [RFD0032](../../../memo/done/0032_lunarlander_obs_include_action_10rfd.md)、テスト項目）。
- 直前行動観測のON/OFFで、base scalarとmarker以外のgridが一致することを確認する（出典: [RFD0040](../../../memo/done/0040_dropmerge_prev_action_obs_10rfd.md)、テスト項目）。
- batch入力の移行でも既存Envのnameとrunner/observer/GUI接続を維持する（出典: [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md)、1.3 維持する外部contract）。
- RAM計測で学習の信号を変えず、番号を追加しても既存番号の意味を変えない（出典: [RFD0082](../../../memo/done/0082_atari_ram_progress_metrics_10rfd.md)、非ゴールNG1・契約8）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md)、ゴールG3・§7 1/4）。
- GridMaze の解ける seed の迷路は変えない。既定 seed（5×5）の 25 セルの golden と、既定 seed の短い Run の metrics checksum の一致で確かめる（出典: [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、§7 の 2・6）。
- GridMazeの変更では固定seedの反復・採用seedからの復元、表示seedを設定へ戻す再現経路、ランダム座標の到達可能性、不正な寸法・座標の構築エラー、複数Envでの通知件数も確認する（出典: [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md)、§5.3・§7の5・7〜9）。
- GridMazeのseed_mode追加では、全Env共有と破棄・再作成後の保持、並行8個体、episodeの50 Reset、独立プロセスでのINFO1回、採用seedからの復元を確認する。旧固定条件とのscalar比較はseedだけを指定する（出典: [GridMaze seed_mode RFD](../../../memo/done/0102_gridmaze_seed_mode_10rfd.md)、§7）。
- NOOPへの罰を直すときは4行動モードで同seed・同行動列を比較し、Step報酬と累積報酬の差、NOOPの変換、実行されないDROP、報酬以外の状態の一致を確認する（出典: [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md)、§5 A4〜A7）。
- ファイルの置き場やライブラリを移すときは、着手前に保存した実行体と、RFD0128 §6.4の前後比較の組（seed 1本、決定的なbackend）で、全scalarの点、設定の記録、ログのWARNとERRORの行の一致を確かめ、テストのケースの数を前後でそろえる（出典: [RFD0182](../../../memo/0182_source_layout_10rfd.md)、§4.1 D24・§6 C3・S3-6）。
- RAMメトリクスの既存の5語の意味、NaNの規則、集計の単位（実ゲーム1回）を変えない。過去のRunのtraceは書き換えないので、同じ名前のまま意味が変わると比べられない（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§1 のN2）。
- RAM定義と縮小方式の読み取りを整理するときは、既存の5語の結果、設定ファイルから届くERRORの文言と時機、観測の画素、実効設定とログに記録する`resize_mode`の値を変えない（出典: [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md)、§4 D11・§7.2 の受入7〜9）。
- 学習時の画像拡張を変更しても、画像観測は uint8 の3×224×224に保つ（出典: [RFD0022](../../../memo/done/0022_imagecls_augmentation_10rfd.md)、確定した設計判断）。
- 合法な DROP が残っている盤面で NOOP を続けた場合は、従来どおり NoDropTimeout で終わり、詰み候補には数えない（出典: [RFD0039](../../../memo/done/0039_dropmerge_nolegal_adjudication_10rfd.md)、受入条件）。
- Atari の前処理やバッチ化を変えるときは、sticky action 0.25 でも同じ seed・行動列で観測・報酬・終端列が bit 一致することを確認する（出典: [RFD0051](../../../memo/done/0051_atari_ale_env_10rfd.md)、§8）。
- ImageCls の共有エピソードでは、評価本数は window 数を表す。従来の global accuracy を保ち、epoch_count とともに完了時に読み出せることを確かめる（出典: [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md)、§6）。

## 外の事例

記述なし

## 関係

- 設計: [環境](../../../design/120_environments.jp.md)、§1.3・§2・§3。
- 初期化関数の登録: [環境](../../../design/120_environments.jp.md)、§3。
- 具象Envの条件と確認手順: [環境](../../../design/120_environments.jp.md)、§7.4・§7.5・§7.6・§8、[Atari環境](../../../design/220_atari_env.jp.md)、§4・§5・§6.1。
- 置き場とテストの参照範囲: [ソース配置とライブラリの境界](../../../design/011_source_layout.jp.md)、「ライブラリとアプリ」「テスト」。
- ユーザーガイド: [Run実行ガイド](../../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [Atari.hpp](../../../../src/env/atari/include/anet/env/Atari.hpp)
- [Atari.cpp](../../../../src/env/atari/src/Atari.cpp)
- [AtariEnv.cpp](../../../../src/env/atari/src/AtariEnv.cpp) / [AtariEnv.hpp](../../../../src/env/atari/src/AtariEnv.hpp) / [AtariEnv_test.cpp](../../../../src/env/atari/src/AtariEnv_test.cpp)
- [AtariPreprocess.cpp](../../../../src/env/atari/src/AtariPreprocess.cpp) / [AtariPreprocess.hpp](../../../../src/env/atari/src/AtariPreprocess.hpp)
- [pch.hpp](../../../../src/env/atari/src/pch.hpp)
- [CartPole.hpp](../../../../src/env/cartpole/include/anet/env/CartPole.hpp)
- [CartPole.cpp](../../../../src/env/cartpole/src/CartPole.cpp)
- [CartPoleEnv.cpp](../../../../src/env/cartpole/src/CartPoleEnv.cpp) / [CartPoleEnv.hpp](../../../../src/env/cartpole/src/CartPoleEnv.hpp) / [CartPoleEnv_test.cpp](../../../../src/env/cartpole/src/CartPoleEnv_test.cpp)
- [pch.hpp](../../../../src/env/cartpole/src/pch.hpp)
- [DropMerge.hpp](../../../../src/env/dropmerge/include/anet/env/DropMerge.hpp)
- [DropMerge.cpp](../../../../src/env/dropmerge/src/DropMerge.cpp)
- [DropMergeEnv.cpp](../../../../src/env/dropmerge/src/DropMergeEnv.cpp) / [DropMergeEnv.hpp](../../../../src/env/dropmerge/src/DropMergeEnv.hpp) / [DropMergeEnv_test.cpp](../../../../src/env/dropmerge/src/DropMergeEnv_test.cpp)
- [pch.hpp](../../../../src/env/dropmerge/src/pch.hpp)
- [GridMaze.hpp](../../../../src/env/gridmaze/include/anet/env/GridMaze.hpp)
- [GridMaze.cpp](../../../../src/env/gridmaze/src/GridMaze.cpp)
- [GridMazeEnv.cpp](../../../../src/env/gridmaze/src/GridMazeEnv.cpp) / [GridMazeEnv.hpp](../../../../src/env/gridmaze/src/GridMazeEnv.hpp) / [GridMazeEnv_test.cpp](../../../../src/env/gridmaze/src/GridMazeEnv_test.cpp)
- [pch.hpp](../../../../src/env/gridmaze/src/pch.hpp)
- [ImageCls.hpp](../../../../src/env/imagecls/include/anet/env/ImageCls.hpp)
- [ImageCls.cpp](../../../../src/env/imagecls/src/ImageCls.cpp)
- [ImageClsEnv.cpp](../../../../src/env/imagecls/src/ImageClsEnv.cpp) / [ImageClsEnv.hpp](../../../../src/env/imagecls/src/ImageClsEnv.hpp) / [ImageClsEnv_test.cpp](../../../../src/env/imagecls/src/ImageClsEnv_test.cpp)
- [ImageData.cpp](../../../../src/env/imagecls/src/ImageData.cpp) / [ImageData.hpp](../../../../src/env/imagecls/src/ImageData.hpp)
- [pch.hpp](../../../../src/env/imagecls/src/pch.hpp)
- [LunarLander.hpp](../../../../src/env/lunarlander/include/anet/env/LunarLander.hpp)
- [LunarLander.cpp](../../../../src/env/lunarlander/src/LunarLander.cpp)
- [LunarLanderEnv.cpp](../../../../src/env/lunarlander/src/LunarLanderEnv.cpp) / [LunarLanderEnv.hpp](../../../../src/env/lunarlander/src/LunarLanderEnv.hpp) / [LunarLanderEnv_test.cpp](../../../../src/env/lunarlander/src/LunarLanderEnv_test.cpp)
- [pch.hpp](../../../../src/env/lunarlander/src/pch.hpp)

## RFD の一覧

点検が作り直す。手で直さない。

- [RFD0034](../../../memo/done/0034_imagecls_batch_input_10rfd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [RFD0060](../../../memo/done/0060_eval_batch_episodes_10rfd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [RFD0061](../../../memo/done/0061_eval_slot_policy_override_10rfd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [RFD0075](../../../memo/done/0075_episode_steps_and_eval_session_log_10rfd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [RFD0082](../../../memo/done/0082_atari_ram_progress_metrics_10rfd.md) — Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
- [RFD0097](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [RFD0099](../../../memo/done/0099_atari_resize_mode_10rfd.md) — Atari の観測の縮小方式を設定で切り替える PRD
- [RFD0100](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md) — GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD
- [RFD0115](../../../memo/0115_heatmap_image_ownership_and_dropmerge_noop_penalty_10rfd.md) — RFD0115: 画像メモリの所有権とDropMergeのNOOP判定の修正
- [RFD0117](../../../memo/0117_warn_policy_and_unread_config_10rfd.md) — RFD0117: WARN の意味の線引きと、読まれない設定の検出 RFD
- [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md) — Atari の RAM メトリクスに値の種類数とビットの変化を数える畳み方を足し、RAM 定義と縮小方式の読み取りを整理する RFD
- [RFD0161](../../../memo/0161_observation_pipeline_10rfd.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化 暫定 RFD
- [RFD0182](../../../memo/0182_source_layout_10rfd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）RFD
- [RFD0192](../../../memo/0192_concurrency_and_boundary_defect_fixes_10rfd.md) — RFD0192: 監査で見つかった不具合 5 件をまとめて直す（Notifier の同時実行、動画の画像の大きさ、強制行動の device、重複登録、QValuePanel の平滑化）
- [RFD0194](../../../memo/0194_config_key_full_names_in_messages_10rfd.md) — RFD0194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）
- [RFD0195](../../../memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md) — RFD0195: 文書の呼び名を RFD に改め、番号を 0 埋めの 4 桁にし、今使っている場所をすべて移す PRD

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | RFD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) | 対応中 | 5.0 | — | コード | その他 | 322_env_individual | Env | 機能追加 | [RFD0126](../../../memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md) | 残件無し | AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない |
| [issue-20261123-9a0e78](../../../management/issues/issue-20261123-9a0e78.md) | 未着手 | 4.0 | — | コード | 実装 | 322_env_individual | 開発 | 不具合 | [RFD0206](../../../memo/0206_core_and_app_small_fixes_10rfd.md) | 残件有り | imagecls の Env のテスト「manifest construction failures sticky」が画像の口を初期化せず、乱数の並びで先に回ると wx のエラーのダイアログで止まる |
| [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) | 未着手 | 3.0 | — | コード | RFD | 322_env_individual | Env | 機能追加 | [RFD0085](../../../memo/0085_dropmerge_eval_score_axes_10rfd.md) | 残件有り | DropMerge の評価スコア（本家互換スコアほか複数の軸） |
| [issue-20261111-c68178](../../../management/issues/issue-20261111-c68178.md) | 未着手 | 3.0 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | DropMerge のエピソード統計を 1 つ足すと、Env のメンバ・Reset・StepResult の初期化子とメンバ・GetScalar の 5 か所へ同じ形を足すことになる |
| [issue-20261107-16ffd8](../../../management/issues/issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | コード | RFD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 残件有り | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| [issue-20261111-6c0bf1](../../../management/issues/issue-20261111-6c0bf1.md) | 未着手 | 2.7 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | GridMazeEnv が毎 step 変わらない盤面を 1 マスずつ tensor へ書き直し、成功と穴落ちを報酬の値と設定の値が等しいかで判定している |
| [issue-20261108-06a3f0](../../../management/issues/issue-20261108-06a3f0.md) | 未着手 | 2.6 | — | コード | その他 | 316_config, 322_env_individual | 基盤 | リファクタリング | [RFD0118](../../../memo/0118_atari_run_profile_flatten_10rfd.md) | 残件有り | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない |
| [issue-20261108-f0f243](../../../management/issues/issue-20261108-f0f243.md) | 未着手 | 2.6 | — | コード | RFD | 316_config, 322_env_individual | 基盤 | リファクタリング | [RFD0118](../../../memo/0118_atari_run_profile_flatten_10rfd.md) | 残件有り | Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする） |
| [issue-20261123-a45289](../../../management/issues/issue-20261123-a45289.md) | 未着手 | 2.5 | — | 文書 | 監査 | 316_config, 322_env_individual | 基盤 | 調査 | — | 未確認 | RFD0118（Atari の Run プロファイルの整理）を指す課題が、題材の 06a3f0 と RFD の仕事の f0f243 の 2 件あり、範囲が同じで重なっている |
| [issue-20261103-5addea](../../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | RFD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [RFD0079](../../../memo/0079_cpp_internal_linkage_convention_10rfd.md) | 残件有り | RFD0079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-785e68](../../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [RFD0169](../../../memo/0169_outside_core_extension_layout_10rfd.md) | 残件無し | anet-core の外で実装する拡張点とコード配置の規約 |
| [issue-20261103-7f1789](../../../management/issues/issue-20261103-7f1789.md) | 未着手 | 2.0 | — | コード | RFD | 315_nn, 322_env_individual | 基盤 | 機能追加 | [RFD0088](../../../memo/0088_nn_adaptive_max_pool_10rfd.md) | 残件有り | 出力の大きさで指定する max pooling（AdaptiveMaxPool2d） |
| [issue-20261111-269a9f](../../../management/issues/issue-20261111-269a9f.md) | 未着手 | 2.0 | — | コード | 監査 | 322_env_individual | Env | リファクタリング | — | 残件有り | DropMergeEnv::Step が 267 行で行動の解釈・物理・裁定・統計・報酬を並べ、行動の解釈・DROP の位置・果物の走査・盤面の受け渡しにも写しがある |
| [issue-20261112-983afe](../../../management/issues/issue-20261112-983afe.md) | 未着手 | 2.0 | — | コード | その他 | 324_dqn, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | — | 残件有り | KFMの学習がボスへ行き着くかが偶然に頼る（1階の廊下の位置が観測に無く、Qも内的報酬も左へ進む向きを持てない） |
| [issue-20261123-22d9bf](../../../management/issues/issue-20261123-22d9bf.md) | 未着手 | 2.0 | — | コード | 監査 | 318_rl_core, 110_runner_app, 322_env_individual | 学習基盤 | リファクタリング | — | 未確認 | TrainEvent の action_info が experience.action と同じ実体を重ねて持ち、読む側（Observer・パネル・View）が 2 通りに分かれている |
| [issue-20261110-adf548](../../../management/issues/issue-20261110-adf548.md) | 未着手 | 1.6 | — | コード | その他 | 322_env_individual | Env | 機能追加 | — | 残件有り | GridMaze に MiniGrid 互換の課題が無く、難しい探索の検証と DI-engine との比較に使えない |
| [issue-20261103-b68e49](../../../management/issues/issue-20261103-b68e49.md) | 未着手 | 1.3 | — | コード | RFD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [RFD0159](../../../memo/0159_input_mapping_joystick_10rfd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| [issue-20261123-b1205d](../../../management/issues/issue-20261123-b1205d.md) | 未着手 | 1.3 | — | コード | 実装 | 110_runner_app, 322_env_individual | 開発 | リファクタリング | — | 残件有り | Runner と Atari のテストが同じ文字列を重ねて書いている（環境変数の名前 RFD0094_UI_DARK を 3 か所、検証資材のパス rfd0099/frames_210x160.u8 を 2 か所） |
| [issue-20261103-8c3c6d](../../../management/issues/issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | コード | RFD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [RFD0171](../../../memo/0171_replay_eviction_sample_count_10rfd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| [issue-20261103-dbe666](../../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | コード | RFD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [RFD0163](../../../memo/0163_dropmerge_neet_diagnostics_10rfd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| [issue-20261103-51220d](../../../management/issues/issue-20261103-51220d.md) | 未着手 | 1.1 | — | コード | RFD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [RFD0144](../../../memo/0144_episode_forensics_10rfd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| [issue-20261103-c16aab](../../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | コード | RFD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [RFD0154](../../../memo/0154_scalar_threshold_rate_10rfd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| [issue-20261111-d1d12b](../../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261103-c5c791](../../../management/issues/issue-20261103-c5c791.md) | 未着手 | 0.9 | — | コード | RFD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [RFD0179](../../../memo/0179_rfd0097_performance_remeasurement_10rfd.md) | 残件有り | RFD0097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| [issue-20261103-969184](../../../management/issues/issue-20261103-969184.md) | 未着手 | 0.6 | — | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0174](../../../memo/0174_llm_actor_10rfd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| [issue-20261116-cc631d](../../../management/issues/issue-20261116-cc631d.md) | 未着手 | 0.6 | — | コード | その他 | 322_env_individual | Env | 機能追加 | — | 残件有り | ImageClsでFood-101以外のデータセット（CIFAR-100・ImageNetなど）を学習・評価に使えるようにする |
| [issue-20261103-124da5](../../../management/issues/issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) | コード | RFD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0167](../../../memo/0167_smdp_transition_duration_10rfd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| [issue-20261103-736e39](../../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md), [issue-20261107-949e8a](../../../management/issues/issue-20261107-949e8a.md) | コード | RFD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [RFD0173](../../../memo/0173_intrinsic_reward_novelty_10rfd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| [issue-20261103-13bc20](../../../management/issues/issue-20261103-13bc20.md) | 未着手 | 0.8 | [issue-20261103-3e1cf6](../../../management/issues/issue-20261103-3e1cf6.md) | コード | RFD | 322_env_individual | Env | 機能追加 | [RFD0086](../../../memo/0086_dropmerge_macro_actions_10rfd.md) | 残件有り | DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化） |
| [issue-20261103-5a783c](../../../management/issues/issue-20261103-5a783c.md) | 未着手 | 0.6 | [issue-20261103-124da5](../../../management/issues/issue-20261103-124da5.md), [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md) | コード | RFD | 321_agent_core, 322_env_individual | アルゴリズム | 機能追加 | [RFD0162](../../../memo/0162_decision_aligned_nstep_10rfd.md) | 残件有り | 次の判断まで遡る n-step（判断単位の可変長 n-step） |

着手できる 26 / 待ち 4 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-9bd71e](../../../management/issues/issue-20261103-9bd71e.md) — 直前行動観測の共通部品（`prev_action` フレーム段）
- 関連する課題: [issue-20261103-75b8da](../../../management/issues/issue-20261103-75b8da.md) — n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正）
<!-- issues:end -->
