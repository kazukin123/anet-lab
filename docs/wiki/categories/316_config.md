# 316 設定管理

[anet-lab](../../../README.md) › [wiki の目録](../index.md) › [機能カテゴリ](../index.md#機能カテゴリ)

## 概要・責務

- ConfigDataに設定を集約し、ConfigResolverがinclude・selection・Run・CLI・継承とprofileを解決する。（出典: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2.1）。
- 設定DTOの`std::optional<T>`は既存の`ConfigData`が読む設定キーの値の型に対応し、未指定ならDTOの初期値を保持して、初期値も無ければ設定記録へ設定キーを足さない。指定があれば通常値の読み取りへ委譲する。複合`ConfigReader`をoptionalで包む形は型の段階で拒む（出典: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§7.3）。

## 成立の経緯

- 2026-07-15: Actor network方針の既定値解決をAgentへ戻す（出典: [ADR 0013](../../adr/0013-actor-network-resource-policy.md)、本文）。
- 2026-07-16: Train Actorのsnapshot設定を追加・検証した（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、題・目的）。
- 2026-07-20: batch Env構築に必要な設定とfactoryの接続を更新した（出典: [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md)、seam / factory / config）。
- 2026-08-14: PRD 052: eval 定義とスケジュールの namespace 分離（出典: [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md)、題・目的）。
- 2026-08-14: configured eval の定義と定期駆動を別 namespace に分離し、生成は schedule エントリが駆動する（出典: [ADR 0027](../../adr/0027-eval-definition-schedule-separation.md)、本文）。
- 2026-08-21: ProfiledValue による ImageCls learning_rate 制御（出典: [PRD031](../../memo/done/031_imagecls_cosine_lr_10prd.md)、題・目的）。
- 2026-08-23: NN 設定ツリーは Agent config prefix 配下に置く（出典: [ADR 0030](../../adr/0030-nn-config-tree-under-agent-prefix.md)、本文）。
- 2026-08-23: 設定体系再設計: 概念ツリー整合・素材宣言・ConfigResolver PRD（出典: [PRD059](../../memo/done/059_config_concept_tree_alignment_10prd.md)、題・目的）。
- 2026-09-07: PRD 072: 設定継承の差分適用と最終値参照（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、題・目的）。
- 2026-09-08: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す（出典: [ADR 0038](../../adr/0038-actor-config-catalog-without-runmode.md)、本文）。
- 2026-09-12: 設定の選択は参照先の最終値を読み、Run プロファイルの設定キーは CLI と同じ第 2 相で選択の後に勝つ（出典: [ADR 0040](../../adr/0040-config-selection-final-value-and-run-profile-tier.md)、本文）。
- 2026-09-13: 設定継承は最終値の差分合成とし、個別指定をベースより優先する（出典: [ADR 0042](../../adr/0042-config-inheritance-as-differential-base.md)、本文）。
- 2026-09-19: PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-24: deviceキーの文字列化に合わせ現用設定とbaseline検証を移行した（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD6・受入基準）。
- 2026-10-18: 特徴branchキー改名の設定比較を更新しPRD072の固定資料を保持した（出典: [PRD093](../../memo/done/093_attention_entropy_diagnostics_10prd.md)、§6 受入基準3・§10 文書同期）。

- 2026-10-24: GridMazeのseed指定と未指定を区別するため、設定DTOのoptional設定キーの値を共通読み取りへ追加した（出典: [PRD102](../../memo/done/102_gridmaze_seed_mode_10prd.md)、§5.1）。
- 2026-10-24: ProfiledValue の定数は裸のスカラー 1 行で書き、root の `.value` 記法は置かない（出典: [ADR 0064](../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、本文）。
- 2026-10-24: 探索と学習のパラメータのキーを概念ごとの入れ子（`spatial.*`・`uqe.*`・`learner.learning_rate` など）へ移し、定数は bare、スケジュールは sub-key で書く記法を加えた（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§4.1 D3・D4・§4.2 D5）。
- 2026-10-25: 迷路の寸法・開始とゴールの座標を構築時に検証する条件を整理した（出典: [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md)、§5.4・§5.6）。
- 2026-10-28: split・skipのsample_weightなどの設定値と既定値を整理した（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§5.1・§5.8）。
- 2026-10-29: Actorのfull・single・variableと補助学習用の窓の組み合わせを検証する設定へ広げた（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§5.1・§5.2）。
- 2026-10-30: Evaluation Traceの強制行動・列・ダンプに関わる設定を同期した（出典: [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md)、§5.7）。
- 2026-11-21: エラー・WARN・INFOが示す設定キーを完全キーで書くため、ConfigDataの項目と読み口（anet::Config）が値を与えた完全キーを記録し、設定の型と一緒に部品へ運ぶ仕様と、上書きの接頭辞が空の読み口がroot直下の相対キーを読まない仕様をグリルで決めた（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§5 D4・D18・§6.2；[ADR 0087](../../adr/0087-messages-name-config-keys-by-full-key-recorded-at-read-time.md)、本文）。

## 選ばなかった形

- PRD072固定入力・goldenの再採取 — 不採用: resolver契約は変わらず過去入力の固定資料を保持し、現用設定は別に前後比較する（出典: [PRD093](../../memo/done/093_attention_entropy_diagnostics_10prd.md)、§9 複雑さ監査）。
- 上書き層の予約名・ドット数による判定・明示登録 — 不採用（cut）: 通常の prefix の差分合成で扱え、名前だけで意味を変えない（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 弱い既定値を @defaults プロファイルへ移す形（全面移行とチェーン先頭での明示、先頭プロファイルが継承する慣用句、暗黙の補完） — 不採用（cut・取り下げ）: 既定値の脱落が実際に起き、C++ の既定値と一致して気づけない。空プロファイルの定義と静的検査、分類規約や予約名が増え、現行設定での利得は 0 件だった（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- = を下書き、:= を上書きとする記法 — 不採用（cut）: プロファイル内の = が再びベースに負け、上書きの記号に既知の意味がない（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 個別の設定キーと継承値の競合の WARN — 不採用（cut）: P2 が認める意図的な上書きにも警告してしまう。既定値は ?= で区別する（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 素材の表記案（キー先頭のドット、連続ドット、bracket の流用、セクションヘッダ、::） — 不採用（棄却）: 素材が root の別領域に集まって所有が切れる、見づらい、名前付きインスタンスの角括弧と意味が衝突する、1 行完結・位置独立を壊す、仰々しい（出典: [PRD059](../../memo/done/059_config_concept_tree_alignment_10prd.md)、付録 D. 検討経緯）。
- use_optimistic_target の廃止 — 不採用（却下）: 実効値は明示の指定で同じになるが、optimistic 化のたびに target_policy の全キーを書き換える（戻しも含む）保守コストが戻る（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、10. 却下した案と経緯）。
- IQN の spec K の専用キー（net.taus_shape など）と、カタログ全項目と Learner の最大の K — 不採用（却下）: Learner の N で足り、設定項目やカタログの走査を増やさない（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、10. 却下した案と経緯）。
- 機能ごとの設定ファイル（PRD098 の skip.txt のように 1 機能の Agent プロファイル・NN 部品・metrics・run.@・app の既定を 1 ファイルに置く形） — 不採用: 関心で分けた 020 §3.6 の表と二軸になり、置き場を表で答えられない。機能は具象 Agent のファイル（`<Agent>.txt`）から `<Agent>_<feature>.txt` を include する形にし、metrics はカタログ、`run.@` は env ファイルに残す（出典: [PRD181 設定ファイル配置](../../memo/181_config_file_layout_10prd.md)、§4・§8 D3）。
- 設定ファイルの Agent 別フォルダ化 — 先送り: include は含むファイルの隣から解決するので仕組みは足りるが、平らな名前で層を表せる間は増やさない。再開条件: 具象 Agent が 6 以上、または 1 Agent の機能ファイルが 4 本を超えたとき（出典: [PRD181 設定ファイル配置](../../memo/181_config_file_layout_10prd.md)、§7）。
- ProfiledValue の定数を常に `.value` で書く形 — 不採用: 定数の行が全部 1 段増え、既存の定数の行を書き換える（出典: [ADR 0064](../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、Considered Options）。
- bare を書いたら type を constant に強制する形 — 不採用: baseline が bare で書いた定数を Run プロファイルが `.type = linear` で差し替える用法が fail-fast になる（出典: [ADR 0064](../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、Considered Options）。
- bare と `.value` の両記法を許し両方書いたら fail-fast にする形 — 不採用: 読み取りが default と override の 2 つの prefix を見るので、ベース＋部分上書きの普通の層構成まで fail-fast になる（出典: [ADR 0064](../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、Considered Options）。
- 眠る bare が`=` で書いた設定キーなら fail-fast にする実行時検査と、`check_default_leaves.py` の静的検査 — 先送り: 既定キーの情報を読み取りまで運ぶ配線が要り、env の`=` で書いた設定キーの定数を Run プロファイルで schedule にする用法も止まる。再開条件: 眠った値に気づかずに Run を回した事故が起きたとき（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§7、[ADR 0064](../../adr/0064-profiledvalue-bare-scalar-reads-as-value-field.md)、Considered Options）。
- 読まれない設定キーの一覧を最初から ERROR にして Run を止める形 — 先送り: 読まれなくても誤りではない設定キー（在庫）の区別を実際の Run で確かめる前に止めると、正しい設定の Run が起動できなくなる。再開条件: 実装をコミットした日から 1 か月のあいだ、受入の Run と番号付き bat の Run のログで、一覧に出た設定キーがすべて設定の誤りだったとき（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D3・§8）。
- 既定キー（`?=`）を読まれない設定キーの一覧の対象から外す形 — 不採用: 共通の設定の綴り違いと旧キーが黙って残り、`?=` かどうかを構築の終わりまで運ぶ配線も増える（出典: [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md)、D4）。
- 利用側がrootまでの接頭辞を渡し、部品が欄の名前を足して完全キーを組む形 — 不採用: 利用側が読まない設定でも完全キーを組み、書いても効かないキーを指す（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§5.1；[ADR 0087](../../adr/0087-messages-name-config-keys-by-full-key-recorded-at-read-time.md)、Considered Options）。
- 利用側が、通知で名指す設定ごとに完全キーを手で渡す形 — 不採用: 上書きの置き場から読んだかを利用側が判定することになり、合成されたblockの設定はどちらの置き場の値かを示せず、渡し忘れを検出できない（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§5.1）。
- 設定だけで決まる検査を具象の読み口へ移して完全キーで書く形 — 不採用: 型の検査を型の共通処理に置く決まりに反し、blockの検査は移せない（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§5.1）。
- 上書きの接頭辞が空の読み口がroot直下の相対キーを読むことを、意図した読み方として残す形 — 不採用: 1行が2つのAgentに同時に効くキーが、仕様として残る（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§5 D18・§5.1）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 通常型は既存のConfigData::Readによる読み取りを維持する（出典: [PRD031](../../memo/done/031_imagecls_cosine_lr_10prd.md)、Config 読み取り仕様）。
- shared networkを既定とする既存Runのaction forward回数、保存形式、Train動作を維持する（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、目的）。
- 設定移行後も同じevalが同じ周期・seed系で走ることを、意図した寄生evalの除去と区別して確認する（出典: [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md)、Context（背景・目的））。
- deviceをauto解決してもconfig_dataとconfig/*.txtの指定値は保持する（出典: [PRD070](../../memo/done/070_device_spec_auto_10prd.md)、確定したD9）。
- 特徴branchキーの改名前後で現用config dumpの差分がキー名だけであることを確認する。PRD072の固定入力・golden・manifestと実験記録のdumpは保持する（出典: [PRD093](../../memo/done/093_attention_entropy_diagnostics_10prd.md)、§6 3）。
- 設定ファイルを分けたり移したりするときは、7 env の解決済み設定（全キーと文字列値）の前後一致と check_default_leaves.py の errors=0 で、解決値と演算子の規約が変わっていないことを確かめる（出典: [PRD181 設定ファイル配置](../../memo/181_config_file_layout_10prd.md)、§6）。

- optional設定キーの値の拡張時は、未指定/DTO初期値の設定記録、prefix優先、空の明示値の復元、不正値の早期失敗を共通Configの公開経路で確認する。指定があるときのdefault prefix→override prefixの順序と型検証・記録を保ち、空文字や不正な数値を未指定へ変換しない（出典: [PRD102](../../memo/done/102_gridmaze_seed_mode_10prd.md)、A15；[実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§7.3）。
- ProfiledValue を読む設定キーでは、root の定数は bare、段の定数は `.phase.[name].value` で書き、`.type` が constant でない層の下の bare は `value` 欄として眠る。config dump は実効 type が constant なら bare の 1 行、それ以外は type とその欄を記録する（出典: [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md)、§4.1 D3・D4）。
- 新しい設定が既定のときは、着手前に保存した実行体・設定を使い、スキップ無効と既存の有効構成のmetrics checksumを一致させる。古い設定と新しい実行体を組み合わせる比較を保つ（出典: [PRD103](../../memo/done/103_temporl_reference_alignment_10prd.md)、§4 D14・§5.7・§7の受入1）。
- 既定のfull・decisionの組み合わせではActorの行動・乱数消費とReplayの抽選列を変えず、PRD103の検証資材でmetrics checksumを一致させる（出典: [PRD104](../../memo/done/104_temporl_skip_auxiliary_only_10prd.md)、§5.7・§7の受入1）。
- 完全キーの記録を足しても、完全キーで書かれた設定の解決の結果（config_data.txtとModuleごとのdump）と、読まれない設定キーの検出の結果を変えない。完全キーを引くことは、項目を読んだことにしない（出典: [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md)、§6.4・§7 A7・A11）。

## 関係

- 設計: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2.1・§7.3。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [config.hpp](../../../src/core/base/include/anet/config.hpp)
- [config.cpp](../../../src/core/base/src/config.cpp) / [config_test.cpp](../../../src/core/base/src/config_test.cpp)
- [config_baseline_test.cpp](../../../src/core/base/src/config_baseline_test.cpp)
- [config_impl.cpp](../../../src/core/base/src/config_impl.cpp) / [config_impl.hpp](../../../src/core/base/src/config_impl.hpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD100](../../memo/done/100_gridmaze_solvable_maze_10prd.md) — GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD
- [PRD101](../../memo/done/101_exploration_schedule_profiled_value_10prd.md) — 探索と学習のパラメータを ProfiledValue で時間変化させる PRD
- [PRD105](../../memo/done/105_eval_trace_arrow_key_forced_action_10prd.md) — Evaluation Trace の矢印キーと強制した step の見分け、表の並べ替えと行のダンプ PRD
- [PRD117](../../memo/117_warn_policy_and_unread_config_10prd.md) — PRD117: WARN の意味の線引きと、読まれない設定の検出 PRD
- [PRD182](../../memo/182_source_layout_10prd.md) — ソース配置の見直し（置き場の規則、Agent の系統ごとのライブラリ、DQN のファイル分割、テストの参照範囲）PRD
- [PRD194](../../memo/194_config_key_full_names_in_messages_10prd.md) — PRD194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）

## 課題

issueの道具（`wiki_support.py maps`）が作り直す。手で直さない。

<!-- issues:begin -->
| ID | 状態 | WSJF | 待ち | 対象 | 工程 | 領域 | 分類 | 種類 | PRD | AIの判定 | 課題 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [issue-20261122-4adabb](../../management/issues/issue-20261122-4adabb.md) | 未着手 | 4.0 | — | コード | 実装 | 311_dev_tools, 316_config | 開発 | 不具合 | — | 未確認 | PRD072 の既定葉検査が移転前の core を探索して停止する |
| [issue-20261111-cdc102](../../management/issues/issue-20261111-cdc102.md) | 未着手 | 2.7 | — | コード | 監査 | 316_config, 315_nn, 324_dqn | 基盤 | リファクタリング | — | 残件有り | WeightInitConfig などの設定の部品に読み取りの口が無く、使う側が 1 キーずつ読むので、同じ 6〜8 行が十数か所にあり、読み方も食い違っている |
| [issue-20261108-06a3f0](../../management/issues/issue-20261108-06a3f0.md) | 未着手 | 2.6 | — | コード | その他 | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 未確認 | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない |
| [issue-20261108-f0f243](../../management/issues/issue-20261108-f0f243.md) | 未着手 | 2.6 | — | コード | PRD | 316_config, 322_env_individual | 基盤 | リファクタリング | [PRD118](../../memo/118_atari_run_profile_flatten_10prd.md) | 残件有り | Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする） |
| [issue-20261103-7c9a70](../../management/issues/issue-20261103-7c9a70.md) | 未着手 | 2.5 | — | コード | 監査 | 316_config | 基盤 | 不整合 | [PRD181](../../memo/181_config_file_layout_10prd.md) | 残件有り | Agentの切替プロファイル@fast/@balance/@heavyがcommon.txtにあり、020 §3.6の表（agent.txtの役割）と食い違う |
| [issue-20261103-5addea](../../management/issues/issue-20261103-5addea.md) | 未着手 | 2.2 | — | コード | PRD | 311_dev_tools, 318_rl_core, 315_nn, 312_util, 317_metrics, 321_agent_core, 319_gui, 316_config, 322_env_individual, 324_dqn, 110_runner_app, 325_muzero, 313_profiling | 開発 | リファクタリング | [PRD079](../../memo/079_cpp_internal_linkage_convention_10prd.md) | 残件有り | PRD 079: .cpp の内部リンケージと namespace インデント規約 |
| [issue-20261103-7c8cd7](../../management/issues/issue-20261103-7c8cd7.md) | 未着手 | 2.0 | — | コード | PRD | 316_config | 基盤 | リファクタリング | [PRD181](../../memo/181_config_file_layout_10prd.md) | 残件有り | 設定ファイルの配置（Agent 全体・具象 Agent・機能 include・フォルダ） |
| [issue-20261103-0ef7b8](../../management/issues/issue-20261103-0ef7b8.md) | 未着手 | 1.6 | — | コード | PRD | 317_metrics, 311_dev_tools, 316_config, 318_rl_core, 321_agent_core, 120_metrics_viewer, 110_runner_app | UI | 機能改善 | [PRD108](../../memo/108_metrics_tag_numbering_10prd.md) | 残件有り | メトリクスの tag の番号を全体で振り直す（群の番号と群の中の番号） |
| [issue-20261103-66f453](../../management/issues/issue-20261103-66f453.md) | 未着手 | 1.4 | — | コード | PRD | 110_runner_app, 316_config | 学習基盤 | 機能追加 | [PRD151](../../memo/151_run_queue_snapshot_10prd.md) | 残件有り | Run キュー（計画スナップショット + exe 固定） |
| [issue-20261103-b4e5c9](../../management/issues/issue-20261103-b4e5c9.md) | 未着手 | 1.3（引継ぎ） | — | コード | PRD | 316_config, 110_runner_app | 基盤 | リファクタリング | [PRD155](../../memo/155_config_root_ownership_alignment_10prd.md) | 残件有り | 設定 root の所有整合(`train.` / `app.run_name` 等の階層ずれ棚卸し) |
| [issue-20261111-d1d12b](../../management/issues/issue-20261111-d1d12b.md) | 未着手 | 1.0 | — | コード | 監査 | 321_agent_core, 315_nn, 318_rl_core, 317_metrics, 316_config, 312_util, 313_profiling, 314_tensor, 322_env_individual, 110_runner_app | 学習基盤 | リファクタリング | — | 残件有り | 今は起こさないリファクタリング候補 16 群を暫定で 1 件にまとめた（手を付けるときに群ごとに分けて起票し直す） |
| [issue-20261122-d9499f](../../management/issues/issue-20261122-d9499f.md) | 未着手 | 1.0 | — | コード | PRD | 316_config, 324_dqn, 314_tensor, 315_nn | 基盤 | 機能改善 | — | 未確認 | 見直したあとの設定の形と通知の決まりに沿わない設定キーと通知が、RND の外に残る（全体は調べていない） |
| [issue-20261103-145ee2](../../management/issues/issue-20261103-145ee2.md) | 未着手 | 0.5 | — | コード | PRD | 321_agent_core, 318_rl_core, 316_config | 学習基盤 | 機能追加 | [PRD152](../../memo/152_metric_targeted_hyperparam_control_10prd.md) | 残件有り | メトリクス目標値によるハイパラ自律調整 |
| [issue-20261115-b02db6](../../management/issues/issue-20261115-b02db6.md) | 未着手 | 1.3 | [issue-20261103-b4e5c9](../../management/issues/issue-20261103-b4e5c9.md) | コード | その他 | 316_config, 324_dqn, 315_nn | 基盤 | リファクタリング | — | 残件有り | RND の Head の出力次元 intrinsic.rnd.output_dim が rnd の直下にあり、Head の初期値 intrinsic.rnd.head.init.* と置き場が分かれている |

着手できる 13 / 待ち 1 / 保留 0 / 次版 0 / 解決 0
<!-- issues:end -->
