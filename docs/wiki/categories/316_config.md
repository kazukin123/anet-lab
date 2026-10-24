# 316 設定管理

## 概要・責務

- ConfigDataに設定を集約し、ConfigResolverがinclude・selection・Run・CLI・継承とprofileを解決する。（出典: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2.1）。

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
- 2026-09-12: 設定の選択は参照先の最終値を読み、Run プロファイルの葉は CLI と同じ第 2 相で選択の後に勝つ（出典: [ADR 0040](../../adr/0040-config-selection-final-value-and-run-profile-tier.md)、本文）。
- 2026-09-13: 設定継承は最終値の差分合成とし、個別指定をベースより優先する（出典: [ADR 0042](../../adr/0042-config-inheritance-as-differential-base.md)、本文）。
- 2026-09-19: PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、題・目的）。
- 2026-09-24: deviceキーの文字列化に合わせ現用設定とbaseline検証を移行した（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD6・受入基準）。
- 2026-10-18: 特徴branchキー改名の設定比較を更新しPRD072の固定資料を保持した（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§6 受入基準3・§10 文書同期）。

## 選ばなかった形

- PRD072固定入力・goldenの再採取 — 不採用: resolver契約は変わらず過去入力の固定資料を保持し、現用設定は別に前後比較する（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§9 複雑さ監査）。
- 上書き層の予約名・ドット数による判定・明示登録 — 不採用（cut）: 通常の prefix の差分合成で扱え、名前だけで意味を変えない（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 弱い既定値を @defaults プロファイルへ移す形（全面移行とチェーン先頭での明示、先頭プロファイルが継承する慣用句、暗黙の補完） — 不採用（cut・取り下げ）: 既定値の脱落が実際に起き、C++ の既定値と一致して気づけない。空プロファイルの定義と静的検査、分類規約や予約名が増え、現行設定での利得は 0 件だった（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- = を下書き、:= を上書きとする記法 — 不採用（cut）: プロファイル内の = が再びベースに負け、上書きの記号に既知の意味がない（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 個別の葉と継承値の競合の WARN — 不採用（cut）: P2 が認める意図的な上書きにも警告してしまう。既定値は ?= で区別する（出典: [PRD072](../../memo/done/072_config_selection_final_value_10prd.md)、8. 検討経緯と簡素化の判断）。
- 素材の表記案（キー先頭のドット、連続ドット、bracket の流用、セクションヘッダ、::） — 不採用（棄却）: 素材が root の別領域に集まって所有が切れる、見づらい、名前付きインスタンスの角括弧と意味が衝突する、1 行完結・位置独立を壊す、仰々しい（出典: [PRD059](../../memo/done/059_config_concept_tree_alignment_10prd.md)、付録 D. 検討経緯）。
- use_optimistic_target の廃止 — 不採用（却下）: 実効値は明示の指定で同じになるが、optimistic 化のたびに target_policy の全キーを書き換える（戻しも含む）保守コストが戻る（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、10. 却下した案と経緯）。
- IQN の spec K の専用キー（net.taus_shape など）と、カタログ全項目と Learner の最大の K — 不採用（却下）: Learner の N で足り、設定項目やカタログの走査を増やさない（出典: [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md)、10. 却下した案と経緯）。

## 変えるときの条件

一般の条件は[リファクタリング観点](../../quality/refactor.md)の「ガードレール」、テストの充足は[定点統計](../../../reports/stats/trend.md)の「テスト充足」を見る。

- 通常型は既存のConfigData::Readによる読み取りを維持する（出典: [PRD031](../../memo/done/031_imagecls_cosine_lr_10prd.md)、Config 読み取り仕様）。
- shared networkを既定とする既存Runのaction forward回数、保存形式、Train動作を維持する（出典: [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md)、目的）。
- 設定移行後も同じevalが同じ周期・seed系で走ることを、意図した寄生evalの除去と区別して確認する（出典: [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md)、Context（背景・目的））。
- deviceをauto解決してもconfig_dataとconfig/*.txtの指定値は保持する（出典: [PRD070](../../memo/070_device_spec_auto_10prd.md)、確定したD9）。
- 特徴branchキーの改名前後で現用config dumpの差分がキー名だけであることを確認する。PRD072の固定入力・golden・manifestと実験記録のdumpは保持する（出典: [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md)、§6 3）。

## 関係

- 設計: [実行基盤と設定](../../design/100_runtime_and_configuration.jp.md)、§2.1。
- ユーザーガイド: [Run実行ガイド](../../design/020_user_guide_run.jp.md)。
- 含まれるアプリケーション: なし。

## ファイル

点検が作り直す。手で直さない。

- [config.hpp](../../../core/anet-core/include/anet/config.hpp)
- [config.cpp](../../../core/anet-core/src/config.cpp) / [config_test.cpp](../../../core/anet-core/src/config_test.cpp)
- [config_baseline_test.cpp](../../../core/anet-core/src/config_baseline_test.cpp)
- [config_impl.cpp](../../../core/anet-core/src/config_impl.cpp) / [config_impl.hpp](../../../core/anet-core/src/config_impl.hpp)

## PRD の一覧

点検が作り直す。手で直さない。

- [PRD034](../../memo/done/034_imagecls_batch_input_10prd.md) — ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
- [PRD036](../../memo/done/036_train_actor_periodic_snapshot_10prd.md) — DefaultDQN Train Actor定期network snapshot PRD
- [PRD052](../../memo/done/052_eval_schedule_separation_10prd.md) — PRD 052: eval 定義とスケジュールの namespace 分離
- [PRD061](../../memo/done/061_eval_slot_policy_override_10prd.md) — PRD 061: Actor 設定カタログと eval スロットごとの方策 / network 指定
- [PRD070](../../memo/070_device_spec_auto_10prd.md) — PRD 070: device 指定の auto 対応とキー体系統一
- [PRD093](../../memo/093_attention_entropy_diagnostics_10prd.md) — TransformerEncoder の attention エントロピー診断 PRD

## 台帳の指摘

台帳のツールが作り直す。手で直さない。

<!-- registry:begin -->
なし

[台帳の一覧](../../../reports/registry/findings.md)
<!-- registry:end -->
