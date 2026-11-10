# 322 Env個別実装

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- 共通Env contract上のCartPole、GridMaze、LunarLander、DropMerge、ImageCls、Atariなどの具象環境の所在をまとめる。（出典: [環境](../../design/120_environments.jp.md)、§1.3・§2・§3）。
- GridMazeは到達できる迷路の生成とseed採番の単位を所有し、LunarLanderViewは推力・風・運動の表示切替を所有する。具象Env・Viewの条件は共通基盤と分けて引く（出典: [環境](../../design/120_environments.jp.md)、§7.4・§7.5）。
- DropMergeは行動モードで選択命令を判定し、NOOPへの罰をEnvのStepあたり1回加算する。Directの列0や実行されないDROPには加算せず、NOOPが移動へ変換される場合は加算する（出典: [環境](../../design/120_environments.jp.md)、§7.6）。

## 成立の経緯

- 2026-06-25: ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）（出典: [PRD022](../../memo/done/022_imagecls_augmentation_10prd.md)、題・目的）。
- 2026-07-18: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ（出典: [ADR 0009](../../adr/0009-imagecls-batch-env-seam.md)、本文）。
- 2026-07-20: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、題・目的）。
- 2026-07-22: DropMerge NoLegal 裁定の blocked-persistence 化（出典: [PRD039](../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、題・目的）。
- 2026-07-25: DropMergeのNoDropTimeoutをtruncationではなく敗着(done+罰)とする（出典: [ADR 0014](../../adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md)、本文）。
- 2026-08-13: PRD 051: ALE (Atari) Env 統合 — C++ 直結・Single seam・自前前処理（出典: [PRD051](../../memo/done/051_atari_ale_env_10prd.md)、題・目的）。
- 2026-08-14: ALE は外部ビルド木を ALE_ROOT で参照し、Atari env はオプショナルビルドにする（GPL 隔離）（出典: [ADR 0025](../../adr/0025-ale-external-build-reference-and-license.md)、本文）。
- 2026-08-14: AtariEnv は SingleDiscreteEnv として登録し、前処理は env 内自前・frame stack は既存 stucker/RB に委ねる（出典: [ADR 0026](../../adr/0026-atari-env-single-seam-own-preprocessing.md)、本文）。
- 2026-08-21: Codex 引継ぎ: DropMergeEnv の NoLegalDrop 終端追加（出典: [PRD005](../../memo/done/005_no_legal_drop_10prd.md)、題・目的）。
- 2026-08-21: DropMergeEnv 直前行動観測(prev-action trio + DROP 列マーカー・暫定)（出典: [PRD040](../../memo/done/040_dropmerge_prev_action_obs_10prd.md)、題・目的）。
- 2026-08-21: Envインスタンスname PRD（出典: [PRD037](../../memo/done/037_env_instance_name_10prd.md)、題・目的）。
- 2026-08-21: Envログ name prefix 自動付与 PRD（出典: [PRD038](../../memo/done/038_env_log_auto_prefix_10prd.md)、題・目的）。
- 2026-08-21: LunarLanderEnv obs_include_action（直前 action の観測化・暫定）（出典: [PRD032](../../memo/done/032_lunarlander_obs_include_action_10prd.md)、題・目的）。
- 2026-09-03: 評価セッションの N 本集約は BatchEnv デコレータと開始時動的グラントで行う（出典: [ADR 0034](../../adr/0034-eval-session-aggregation-in-batchenv-decorator.md)、本文）。
- 2026-09-04: ImageClsを含む評価episode集約を確認した（出典: [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md)、F4. env の確定値は「終端した Step が返った直後」にしか読めない）。
- 2026-09-17: GridMazeのmax_stepsとepisode長の扱いを整理した（出典: [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md)、エピソード長が framework の一級市民になっていない）。
- 2026-09-19: Actorカタログ移行後の各Envとの接続を回帰確認した（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-24: AtariEnv のゲーム固有 RAM メトリクスは番号付き汎用キーで参照し、意味はゲーム別の RAM 定義が持つ（出典: [ADR 0046](../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、本文）。
- 2026-09-24: Atariのゲーム固有イベントを番号付きRAMメトリクスとして記録した（出典: [PRD082](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、題・ゴールG1〜G4）。
- 2026-10-20: 各Envの遷移の値をStepの結果へ移し終端の読み取りをそろえた（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§4.3.5〜4.3.7・§4.4 クラス一覧）。
- 2026-10-20: 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、本文）。

- 2026-10-23: Atari観測の縮小方式を設定で選ぶ仕様を精査した。実装と数値・性能の受入は後続工程である（出典: [PRD099](../../memo/done/099_atari_resize_mode_10prd.md)、§1・§8）。
- 2026-10-24: GridMaze の迷路生成に到達可能性の保証を足し、解けない seed は派生 seed で決定的に引き直す。解ける seed の迷路は変えない（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、題・§4）。


- 2026-10-24: GridMazeの迷路seedを指定値またはエピソード・プロセス単位で選ぶよう整理した（出典: [GridMaze seed_mode PRD](../../memo/done/102_gridmaze_seed_mode_10prd.md)、§1・§4）。

- 2026-11-04: DropMergeのNOOPへの罰を既存の行動モードの判定へそろえ、Directの列0のDROPへ罰が付く不具合を修正した（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§1.2、[環境](../../design/120_environments.jp.md)、§7.6）。

## 選ばなかった形

- NOOPの判定修正と同時にDropMergeの罰の値・行動集合・報酬設計を変える形 — 範囲外: 行動モードに意味をそろえる修正とは別の関心。再開条件: 学習上の報酬設計を見直す実験を選んだとき（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§1.3）。

- OpenCV依存の追加・84専用制限・OpenCV全分岐の厳密再現 — 不採用: 自前前処理の境界と縮小全範囲の設定を保ち、anetの数値規則と固定画像の許容差を定める（出典: [PRD099](../../memo/done/099_atari_resize_mode_10prd.md)、§7）。
- 詰み判定のために内部の盤面を進めて裁定する方式 — 不採用: 問題設定が複雑化し、人間のプレイ感覚から乖離する（出典: [PRD039](../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、決定経緯の要約）。
- episode開始通知とViewの開始表示 — 範囲外: 今回のゴールに効かず手動操作の時点ずれと同じ関心。再開条件: 948_qvaluepanel_manual_action_state_alignmentで手動の時点ずれを決めるとき（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5）。
- ResetEndedEpisodesの段・外から読む値の写し・予備ALE — 不採用: 段の案は非同期学習を5〜8%遅くし、写しは埋め合わせを残し、予備ALEはAtari固有になる（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、§5・§4.1 D1/D2）。
- 速報値と確定値を別の鍵名に分ける形 — 不採用: Step結果が2組の値を持つ作りになり、現用の設定キー名も移行する必要がある（出典: [ADR 0060](../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、Considered Options）。
- ゲーム別RAM番地・畳み方をコードの固定表に持つ形 — 不採用: 番地を試すたびに再ビルドが要り、解析上の選択を固定する（出典: [ADR 0046](../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- RAM番地またはラベルをmetricsキーへ直接出す形 — 不採用: ゲームやイベントの変更でmetrics定義を書き換えることになる（出典: [ADR 0046](../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- frame skip窓のあとだけRAMを観測する形 — 不採用: 窓途中とreset中の遷移を回数系で取りこぼす（出典: [ADR 0046](../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Considered Options）。
- ImageCls の decode cache を bounded LRU にする形 — 不採用: epoch シャッフルでは再利用距離が平均約 0.75N で、容量 K=N/2 でも LRU の hit 率は約 15% にとどまり、同じ容量の固定ピン留め（hit 率 50%）に常に劣る（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.11 D9解決）。
- ImageCls の評価を固定の部分集合で行う形 — 不採用: 毎回同じ画像だけを見ることになる（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.2 維持された主要判断・12.8 D5解決）。
- 評価の終わりを driver 側（Runner）で sample 数を数えて決める形 — 不採用: 端数の padding は物理的に Source 側で必須なので処理が二重になり、汎用の EvalRunner が valid 件数という dataset の語彙を持つことになる（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.8 D5解決）。
- EvalPanel の Source の乱数列を同じタグの configured eval にそろえる形（option 2） — 不採用（棄却）: GUI の操作で cursor がすぐずれて整列は保てず、同じ seed だから同じ対象を見ているという偽の手がかりだけが残る（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、12.12 D10解決）。
- NoDropTimeout を truncated にする形 — 不採用（却下）: ショットクロックは観測にあり、クロックを進めるのは agent 自身の NOOP なので、終端は agent の支配下の Markov な帰結で done が意味論的に正しい。flag の切り替えで行うと観測の次元まで変わり、一軸の A/B が壊れる（出典: [PRD039](../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、決定経緯の要約 8）。
- missed_resignation（persistence 条件付きの投了失敗）メトリクス — 不採用（却下）: use_instant_drop では次の行動選択の時点の persistence は高々 1 frame で、Phase 2 の後は該当する状態が構造的に空集合になり、厳密な投了率には反実仮想の物理シミュレーションが要る（出典: [PRD039](../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、決定経緯の要約 2）。
- timeout 時の再分類・裁定の上限・no_legal_adjudication_timeout — 不採用（導入しない）: blocked-persistence の裁定では到達できない dead 機構になり、瞬間の candidate による再分類は振動して時々 legal が開く盤面を NoLegal へ誤分類する（出典: [PRD039](../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、決定経緯の要約 5）。
- GridMaze で解けない迷路を止めるだけにする、通路を掘る、生成アルゴリズムを変える — 不採用: 自動 seed の Run が途中で止まる／迷路の分布が偏る／解ける seed の迷路が変わる（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、§8）。
- GridMaze の迷路の難しさ（最短路の長さ、穴の近さ）の制御 — 範囲外: 別の関心。再開条件: 難しさの違いで学習の差を見たくなったとき（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、§8）。


- GridMazeの個体別・BatchEnv別の一度だけの採番 — 不採用: 学習・評価・GUIを同じプロセスの入力seedへそろえる目的を満たさない（出典: [GridMaze seed_mode PRD](../../memo/done/102_gridmaze_seed_mode_10prd.md)、§4 D2・§8）。

## 変えるときの条件

- 縮小方式追加では既定の学習系列のchecksumを保持し、84×84の各参照画像で全画素差1以下・99.9%以上一致を確認する。64 envの速度比中央値0.98以上を要求し、対照測定の揺れが1%を超える場合は判定保留にする（出典: [PRD099](../../memo/done/099_atari_resize_mode_10prd.md)、§6）。
一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 直前action観測のON/OFFを同seed・同action列で並走し、先頭8次元を全stepで一致させる（出典: [PRD032](../../memo/done/032_lunarlander_obs_include_action_10prd.md)、テスト項目）。
- 直前行動観測のON/OFFで、base scalarとmarker以外のgridが一致することを確認する（出典: [PRD040](../../memo/done/040_dropmerge_prev_action_obs_10prd.md)、テスト項目）。
- batch入力の移行でも既存Envのnameとrunner/observer/GUI接続を維持する（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、1.3 維持する外部contract）。
- RAM計測で学習の信号を変えず、番号を追加しても既存番号の意味を変えない（出典: [PRD082](../../memo/done/082_atari_ram_progress_metrics_10prd.md)、非ゴールNG1・契約8）。
- Agentとreplayへ入る値、loss・Q、episode return・長さを保持する。Atari2構成と他Env・逐次/非同期の7構成で全点を比較し、実時間は交互3組、ばらつきが大きければ5組で性能低下を確認する（出典: [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md)、ゴールG3・§7 1/4）。
- GridMaze の解ける seed の迷路は変えない。既定 seed（5×5）の 25 セルの golden と、既定 seed の短い Run の metrics checksum の一致で確かめる（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、§7 の 2・6）。
- GridMazeの変更では固定seedの反復・採用seedからの復元、表示seedを設定へ戻す再現経路、ランダム座標の到達可能性、不正な寸法・座標の構築エラー、複数Envでの通知件数も確認する（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、§5.3・§7の5・7〜9）。


- GridMazeのseed_mode追加では、全Env共有と破棄・再作成後の保持、並行8個体、episodeの50 Reset、独立プロセスでのINFO1回、採用seedからの復元を確認する。旧固定条件とのscalar比較はseedだけを指定する（出典: [GridMaze seed_mode PRD](../../memo/done/102_gridmaze_seed_mode_10prd.md)、§7）。

- NOOPへの罰を直すときは4行動モードで同seed・同行動列を比較し、Step報酬と累積報酬の差、NOOPの変換、実行されないDROP、報酬以外の状態の一致を確認する（出典: [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md)、§5 A4〜A7）。

## 関係

- 設計: [環境](../../design/120_environments.jp.md)、§1.3・§2・§3。
- 具象Envの条件と確認手順: [環境](../../design/120_environments.jp.md)、§7.4・§7.5・§7.6・§8、[Atari環境](../../design/220_atari_env.jp.md)、§4・§5・§6.1。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [Atari.hpp](../../../core/envs/atari1/include/anet/env/Atari.hpp)
- [Atari.cpp](../../../core/envs/atari1/src/Atari.cpp)
- [AtariEnv.cpp](../../../core/envs/atari1/src/AtariEnv.cpp) / [AtariEnv.hpp](../../../core/envs/atari1/src/AtariEnv.hpp) / [AtariEnv_test.cpp](../../../core/envs/atari1/src/AtariEnv_test.cpp)
- [AtariPreprocess.cpp](../../../core/envs/atari1/src/AtariPreprocess.cpp) / [AtariPreprocess.hpp](../../../core/envs/atari1/src/AtariPreprocess.hpp)
- [pch.hpp](../../../core/envs/atari1/src/pch.hpp)
- [CartPole.hpp](../../../core/envs/cartpole2/include/anet/env/CartPole.hpp)
- [CartPole.cpp](../../../core/envs/cartpole2/src/CartPole.cpp)
- [CartPoleEnv.cpp](../../../core/envs/cartpole2/src/CartPoleEnv.cpp) / [CartPoleEnv.hpp](../../../core/envs/cartpole2/src/CartPoleEnv.hpp)
- [pch.hpp](../../../core/envs/cartpole2/src/pch.hpp)
- [DropMerge.hpp](../../../core/envs/dropmerge1/include/anet/env/DropMerge.hpp)
- [DropMerge.cpp](../../../core/envs/dropmerge1/src/DropMerge.cpp)
- [DropMergeEnv.cpp](../../../core/envs/dropmerge1/src/DropMergeEnv.cpp) / [DropMergeEnv.hpp](../../../core/envs/dropmerge1/src/DropMergeEnv.hpp) / [DropMergeEnv_test.cpp](../../../core/envs/dropmerge1/src/DropMergeEnv_test.cpp)
- [pch.hpp](../../../core/envs/dropmerge1/src/pch.hpp)
- [GridMaze.hpp](../../../core/envs/gridmaze1/include/anet/env/GridMaze.hpp)
- [GridMaze.cpp](../../../core/envs/gridmaze1/src/GridMaze.cpp)
- [GridMazeEnv.cpp](../../../core/envs/gridmaze1/src/GridMazeEnv.cpp) / [GridMazeEnv.hpp](../../../core/envs/gridmaze1/src/GridMazeEnv.hpp) / [GridMazeEnv_test.cpp](../../../core/envs/gridmaze1/src/GridMazeEnv_test.cpp)
- [pch.hpp](../../../core/envs/gridmaze1/src/pch.hpp)
- [ImageCls.hpp](../../../core/envs/imagecls1/include/anet/env/ImageCls.hpp)
- [ImageCls.cpp](../../../core/envs/imagecls1/src/ImageCls.cpp)
- [ImageClsEnv.cpp](../../../core/envs/imagecls1/src/ImageClsEnv.cpp) / [ImageClsEnv.hpp](../../../core/envs/imagecls1/src/ImageClsEnv.hpp) / [ImageClsEnv_test.cpp](../../../core/envs/imagecls1/src/ImageClsEnv_test.cpp)
- [ImageData.cpp](../../../core/envs/imagecls1/src/ImageData.cpp) / [ImageData.hpp](../../../core/envs/imagecls1/src/ImageData.hpp)
- [pch.hpp](../../../core/envs/imagecls1/src/pch.hpp)
- [LunarLander.hpp](../../../core/envs/lunarlander1/include/anet/env/LunarLander.hpp)
- [LunarLander.cpp](../../../core/envs/lunarlander1/src/LunarLander.cpp)
- [LunarLanderEnv.cpp](../../../core/envs/lunarlander1/src/LunarLanderEnv.cpp) / [LunarLanderEnv.hpp](../../../core/envs/lunarlander1/src/LunarLanderEnv.hpp) / [LunarLanderEnv_test.cpp](../../../core/envs/lunarlander1/src/LunarLanderEnv_test.cpp)
- [pch.hpp](../../../core/envs/lunarlander1/src/pch.hpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD060](../../memo/done/060_eval_batch_episodes_10prd.md) — 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD075](../../memo/done/075_episode_steps_and_eval_session_log_10prd.md) — PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ
- [PRD082](../../memo/done/082_atari_ram_progress_metrics_10prd.md) — Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD
- [PRD097](../../memo/done/097_env_auto_reset_after_notification_10prd.md) — 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
- [PRD099](../../memo/done/099_atari_resize_mode_10prd.md) — Atari の観測の縮小方式を設定で切り替える PRD
- [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md) — GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD
- [PRD115](../../memo/115_heatmap_image_ownership_and_dropmerge_noop_penalty_10prd.md) — PRD115: 画像メモリの所有権とDropMergeのNOOP判定の修正

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| # | ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) | 未着手 | 5.0 | — | 成果物 | その他 | 322_env_individual | Env | 機能追加 | — | 未確認 | AtariのRAMメトリクスに値の種類数とビットを数える畳み方が無く、Montezumaの訪れた部屋の数や持ち物を記録できない |
| 2 | [issue-20261103-3e1cf6](../../management/issues/issue-20261103-3e1cf6.md) | 未着手 | 3.0 | — | 成果物 | PRD | 322_env_individual | Env | 機能追加 | [PRD085](../../memo/085_dropmerge_eval_score_axes_10prd.md) | 残件有り | DropMerge の評価スコア（本家互換スコアほか複数の軸） |
| 3 | [issue-20261107-16ffd8](../../management/issues/issue-20261107-16ffd8.md) | 未着手 | 2.7 | — | 成果物 | PRD | 318_rl_core, 322_env_individual, 110_runner_app | 学習基盤 | 不整合 | — | 未確認 | ERROR のログを出したあとも Run を止めずに続ける箇所（Env・Agent の作成失敗、DropMerge の果物の範囲外）が Fail-Fast 原則と合わない |
| 4 | [issue-20261108-06a3f0](../../management/issues/issue-20261108-06a3f0.md) | 未着手 | 2.6 | — | 成果物 | その他 | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 未確認 | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない |
| 5 | [issue-20261108-f0f243](../../management/issues/issue-20261108-f0f243.md) | 未着手 | 2.6 | — | 成果物 | PRD | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 残件有り | Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする） |
| 6 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 318_rl_core, 321_agent_core, 324_dqn, 319_gui, 320_env_core, 314_tensor, 312_util, 110_runner_app, 322_env_individual, 325_muzero | 開発 | リファクタリング | [PRD945](../../memo/945_outside_core_extension_layout_10prd.md) | 残件有り | anet-core の外で実装する拡張点とコード配置の規約 |
| 7 | [issue-20261103-7f1789](../../management/issues/issue-20261103-7f1789.md) | 未着手 | 2.0 | — | 成果物 | PRD | 315_nn, 322_env_individual | アルゴリズム | 機能追加 | [PRD088](../../memo/088_nn_adaptive_max_pool_10prd.md) | 残件有り | 出力の大きさで指定する max pooling（AdaptiveMaxPool2d） |
| 8 | [issue-20261110-adf548](../../management/issues/issue-20261110-adf548.md) | 未着手 | 1.6 | — | 成果物 | その他 | 322_env_individual | Env | 機能追加 | — | 未確認 | GridMaze に MiniGrid 互換の課題が無く、難しい探索の検証と DI-engine との比較に使えない |
| 9 | [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 1.4 | — | 成果物 | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| 10 | [issue-20261103-8c3c6d](../../management/issues/issue-20261103-8c3c6d.md) | 未着手 | 1.2 | — | 成果物 | PRD | 318_rl_core, 324_dqn, 321_agent_core, 322_env_individual | 学習基盤 | 機能追加 | [PRD950](../../memo/950_replay_eviction_sample_count_10prd.md) | 残件有り | replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る |
| 11 | [issue-20261103-dbe666](../../management/issues/issue-20261103-dbe666.md) | 未着手 | 1.2 | — | 成果物 | PRD | 322_env_individual, 318_rl_core, 311_dev_tools | Env | 調査 | [PRD915](../../memo/915_dropmerge_neet_diagnostics_10prd.md) | 残件有り | DropMerge の NEET 診断（trace で足りるか確かめてから指標化） |
| 12 | [issue-20261103-c16aab](../../management/issues/issue-20261103-c16aab.md) | 未着手 | 1.0 | — | 成果物 | PRD | 320_env_core, 322_env_individual, 312_util, 120_metrics_viewer, 311_dev_tools | UI | 機能追加 | [PRD954](../../memo/954_scalar_threshold_rate_10prd.md) | 残件有り | スカラー閾値越え率の汎用化（Observation 層 / Viewer 層） |
| 13 | [issue-20261103-c5c791](../../management/issues/issue-20261103-c5c791.md) | 未着手 | 0.9 | — | 成果物 | PRD | 322_env_individual, 318_rl_core | 学習基盤 | 調査 | [PRD946](../../memo/946_prd097_performance_remeasurement_10prd.md) | 残件有り | PRD097: 判定不能4構成の性能再測定とDropMerge serial追加費用の調査 |
| 14 | [issue-20261103-51220d](../../management/issues/issue-20261103-51220d.md) | 未着手 | 0.7 | — | 成果物 | PRD | 318_rl_core, 321_agent_core, 322_env_individual, 317_metrics | 学習基盤 | 機能追加 | [PRD932](../../memo/932_episode_forensics_10prd.md) | 残件有り | Episode Forensics（特殊挙動エピソード自動捕捉・事後調査） |
| 15 | [issue-20261103-969184](../../management/issues/issue-20261103-969184.md) | 未着手 | 0.6 | — | 成果物 | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD926](../../memo/926_llm_actor_10prd.md) | 残件有り | LLM Actor（LLM の事前知識で行動する Actor。単独 Agent と DQN ハイブリッド） |
| 16 | [issue-20261103-124da5](../../management/issues/issue-20261103-124da5.md) | 未着手 | 1.5 | [issue-20261103-3e1cf6](../../management/issues/issue-20261103-3e1cf6.md) | 成果物 | PRD | 321_agent_core, 318_rl_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD956](../../memo/956_smdp_transition_duration_10prd.md) | 残件有り | SMDP 対応（遷移ごとの経過時間を割引に通す） |
| 17 | [issue-20261103-736e39](../../management/issues/issue-20261103-736e39.md) | 未着手 | 1.4 | [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md), [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md), [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md), [issue-20261103-ab98b7](../../management/issues/issue-20261103-ab98b7.md), [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md), [issue-20261107-1226e5](../../management/issues/issue-20261107-1226e5.md), [issue-20261107-949e8a](../../management/issues/issue-20261107-949e8a.md) | 成果物 | PRD | 324_dqn, 321_agent_core, 318_rl_core, 322_env_individual, 110_runner_app | アルゴリズム | 機能追加 | [PRD924](../../memo/924_intrinsic_reward_novelty_10prd.md) | 残件有り | 内的報酬（NGU・E3B のエピソード内の新しさと RND）と方策の組 |
| 18 | [issue-20261103-b68e49](../../management/issues/issue-20261103-b68e49.md) | 未着手 | 1.3 | [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) | 成果物 | PRD | 110_runner_app, 318_rl_core, 322_env_individual | UI | 機能追加 | [PRD922](../../memo/922_input_mapping_joystick_10prd.md) | 残件有り | 操作入力の Joystick 対応と操作マッピングの設定化（EvalPanel） |
| 19 | [issue-20261103-13bc20](../../management/issues/issue-20261103-13bc20.md) | 未着手 | 0.8 | [issue-20261103-3e1cf6](../../management/issues/issue-20261103-3e1cf6.md) | 成果物 | PRD | 322_env_individual | Env | 機能追加 | [PRD086](../../memo/086_dropmerge_macro_actions_10prd.md) | 残件有り | DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化） |
| 20 | [issue-20261103-2dd820](../../management/issues/issue-20261103-2dd820.md) | 未着手 | 0.6 | [issue-20261103-785e68](../../management/issues/issue-20261103-785e68.md) | 成果物 | PRD | 321_agent_core, 315_nn, 324_dqn, 312_util, 318_rl_core, 325_muzero, 110_runner_app, 311_dev_tools, 322_env_individual | 開発 | リファクタリング | [PRD957](../../memo/957_source_layout_10prd.md) | 残件有り | ソース配置の見直し（トップ階層を `src/` に、Env と同じ具象別のライブラリへ Agent を分ける） |
| 21 | [issue-20261103-5a783c](../../management/issues/issue-20261103-5a783c.md) | 未着手 | 0.6 | [issue-20261103-124da5](../../management/issues/issue-20261103-124da5.md), [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md) | 成果物 | PRD | 321_agent_core, 322_env_individual | アルゴリズム | 機能追加 | [PRD914](../../memo/914_decision_aligned_nstep_10prd.md) | 残件有り | 次の判断まで遡る n-step（判断単位の可変長 n-step） |

着手できる 15 / 待ち 6 / 保留 0 / 次版 0 / 解決 0
- 関連する課題: [issue-20261103-b754ff](../../management/issues/issue-20261103-b754ff.md) — stucker.stack_keysに存在しないキー名を書いても検出されない
- 関連する課題: [issue-20261107-1226e5](../../management/issues/issue-20261107-1226e5.md) — Evaluation Traceで内的報酬（RNDの新しさ）を見られない
- 関連する課題: [issue-20261103-aa5758](../../management/issues/issue-20261103-aa5758.md) — 観測パイプライン（観測加工の定義と spec の一元化）と強制行動の `MakeAction` 入力化
- 関連する課題: [issue-20261103-ab98b7](../../management/issues/issue-20261103-ab98b7.md) — TD・QR・IQNのUpdateFromSamplesが同じ骨格（正規化→損失→PER重み→PER更新→最適化）を丸写ししている
- 関連する課題: [issue-20261103-75b8da](../../management/issues/issue-20261103-75b8da.md) — n-step の目標値の補正（窓の途中の Munchausen の項・方策オフの補正）
- 関連する課題: [issue-20261103-9bd71e](../../management/issues/issue-20261103-9bd71e.md) — 直前行動観測の共通部品（`prev_action` フレーム段）
<!-- issues:end -->
