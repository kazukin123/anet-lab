# AI ハーネスの動線マップ

## 凡例と図の索引

AIハーネスはエージェントの振る舞いを決める指示・規約・文書・skill・memory・CIの総体。入口は毎セッション読むファイル、条件読みはそれ以外の読み、動線は入口から末端までの参照の連なり。shadowは別スコープの同名skillが起動しうる状態、実体コピーは同じ内容を両側へ意図して置く構造である（根拠: anet-harness-map contract §用語）。

生成根拠のGit revision: `5409cd1e9a8bf070c95d18fc606ce9b8694be6d9` と作業ツリー。日本語の図を作成し、`meta.locale` は省略。固定Viewer UIとHTMLのlangは英語になる。図のrepository evidenceはこのrevisionへ固定し、新設wikiの根拠はコミット済みPRD095を指す。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
| --- | --- | --- | --- | --- |
| 1 | 俯瞰 | 入口から領域の契約と検証へ進む読む道 | [開く](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [HTML](harness_map_architecture.html) / [JSON](harness_map_architecture.archify.json) |
| 2 | 工程 | PRDづくり・実装・検証と人間の確認 | [開く](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

## 表1 ノード在庫

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
| --- | --- | --- | --- | --- | --- |
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 毎セッション | 人間・Claude・Codex | AGENTS.md:23, .agents/skills/anet-harness-map/SKILL.md:28, .agents/skills/anet-harness-map/SKILL.md:31 ほか | CLAUDE.md:1 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 毎セッション | 人間・Claude・Codex | CLAUDE.md:4, .agents/skills/README.md:36, docs/design/010_framework_overview.jp.md:349 ほか | AGENTS.md  Triage labels |
| 規約・用語 | [CONTEXT.md](../../CONTEXT.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:102, AGENTS.md:104, AGENTS.md:756 ほか | CONTEXT.md  方策更新統計 |
| 規約・用語 | [docs/ownership_guideline.md](../ownership_guideline.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:301, README.md:44, docs/design/010_framework_overview.jp.md:402 ほか | docs/ownership_guideline.md  ◎ Private Resource（モジュール専用資源） |
| 規約・用語 | [docs/memo/README.md](../memo/README.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:746, .agents/skills/anet-create-prd/SKILL.md:91, docs/agents/issue-tracker.md:18 | docs/memo/README.md 工程と担当 |
| 規約・用語 | [docs/experiments/README.md](../experiments/README.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29 | docs/experiments/README.md campaign 文書の基本形 |
| wiki | [docs/wiki/README.md](../wiki/README.md) | PRD・Run分析・契約変更の前（AGENTS.md §wikiを引く条件） | 人間・Claude・Codex | AGENTS.md:100, .agents/skills/anet-create-prd/SKILL.md:95 | docs/wiki/README.md 書いたあとの機械的な確かめ |
| wiki | [docs/wiki/index.md](../wiki/index.md) | PRD・Run分析・契約変更の前（AGENTS.md §wikiを引く条件） | 人間・Claude・Codex | AGENTS.md:97, .agents/skills/anet-create-prd/SKILL.md:16 | docs/wiki/index.md アプリケーション |
| wiki | [docs/wiki/log.md](../wiki/log.md) | PRD・Run分析・契約変更の前（AGENTS.md §wikiを引く条件） | 人間・Claude・Codex | AGENTS.md:99, .agents/skills/anet-create-prd/SKILL.md:148 | docs/wiki/log.md:1 |
| 規約・用語 | [docs/agents/domain.md](domain.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:757, .agents/skills/setup-matt-pocock-skills/SKILL.md:75, .agents/skills/setup-matt-pocock-skills/SKILL.md:106 ほか | docs/agents/domain.md Use the glossary's vocabulary |
| 規約・用語 | [docs/agents/issue-tracker.md](issue-tracker.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:747, docs/memo/README.md:6, .agents/skills/setup-matt-pocock-skills/SKILL.md:75 ほか | docs/agents/issue-tracker.md Skill Routing |
| 規約・用語 | [docs/agents/triage-labels.md](triage-labels.md) | 入口またはskillの指定条件 | 人間・Claude・Codex | AGENTS.md:752, .agents/skills/setup-matt-pocock-skills/SKILL.md:75, .agents/skills/setup-matt-pocock-skills/SKILL.md:102 ほか | docs/agents/triage-labels.md:1 |
| ガイド・設計 | [docs/design/010_framework_overview.jp.md](../design/010_framework_overview.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | .agents/skills/README.md:16, docs/design/100_runtime_and_configuration.jp.md:357, docs/design/110_agents_and_learning.jp.md:229 ほか | docs/design/010_framework_overview.jp.md 8. 基本的な設計原則 |
| ガイド・設計 | [docs/design/020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | AGENTS.md:193, README.md:27, docs/design/010_framework_overview.jp.md:210 ほか | docs/design/020_user_guide_run.jp.md 8. よくある確認事項 |
| ガイド・設計 | [docs/design/030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | AGENTS.md:640, README.md:28, docs/design/010_framework_overview.jp.md:367 ほか | docs/design/030_user_guide_analysis.jp.md 7. 最小分析チェックリスト |
| ガイド・設計 | [docs/design/040_development_environment.jp.md](../design/040_development_environment.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | README.md:26, docs/design/010_framework_overview.jp.md:40, docs/design/010_framework_overview.jp.md:355 ほか | docs/design/040_development_environment.jp.md  8.6 CUDA または DLL の読み込みに失敗する |
| ガイド・設計 | [docs/design/100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:210, docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:252 ほか | docs/design/100_runtime_and_configuration.jp.md 8. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:228, docs/design/010_framework_overview.jp.md:361 ほか | docs/design/110_agents_and_learning.jp.md 8. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/120_environments.jp.md](../design/120_environments.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:222, docs/design/010_framework_overview.jp.md:361 ほか | docs/design/120_environments.jp.md 8. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:240, docs/design/110_agents_and_learning.jp.md:21, docs/design/110_agents_and_learning.jp.md:232 ほか | docs/design/130_neural_networks.jp.md 8. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/140_observability.jp.md](../design/140_observability.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89, docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:337 ほか | docs/design/140_observability.jp.md 9. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:234, docs/design/020_user_guide_run.jp.md:537, docs/design/100_runtime_and_configuration.jp.md:362 ほか | docs/design/150_replay_buffer.jp.md 8. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:343, docs/design/010_framework_overview.jp.md:367, docs/design/020_user_guide_run.jp.md:539 ほか | docs/design/160_applications_and_tools.jp.md 8. テストと変更時の確認事項 |
| ガイド・設計 | [docs/design/200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:228, docs/design/010_framework_overview.jp.md:252, docs/design/010_framework_overview.jp.md:361 ほか | docs/design/200_dqn_agents.jp.md 10. テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/210_metrics_viewer.jp.md](../design/210_metrics_viewer.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20, docs/design/160_applications_and_tools.jp.md:354, docs/design/README.jp.md:74 | docs/design/210_metrics_viewer.jp.md  12.2 変更時の確認事項 |
| ガイド・設計 | [docs/design/220_atari_env.jp.md](../design/220_atari_env.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190, docs/design/README.jp.md:75 | docs/design/220_atari_env.jp.md  6.1 テストと拡張時の確認事項 |
| ガイド・設計 | [docs/design/README.jp.md](../design/README.jp.md) | 当該領域を扱う前（AGENTS.md §作業ルール、設計索引） | 人間・Claude・Codex | README.md:35, docs/design/010_framework_overview.jp.md:397, .agents/skills/anet-harness-map/SKILL.md:29 | docs/design/README.jp.md  3.5 具象実装仕様 |
| 検証資材 | [core/anet-core/testdata/prd061/README.md](../../core/anet-core/testdata/prd061/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:214 | core/anet-core/testdata/prd061/README.md 移行比較 |
| 検証資材 | [core/anet-core/testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | AGENTS.md:193, docs/design/100_runtime_and_configuration.jp.md:351 | core/anet-core/testdata/prd072/README.md golden の更新方針 |
| 検証資材 | [core/anet-core/testdata/prd078/README.md](../../core/anet-core/testdata/prd078/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:348 | core/anet-core/testdata/prd078/README.md 実行が必要になるとき |
| 検証資材 | [core/anet-core/testdata/prd082/README.md](../../core/anet-core/testdata/prd082/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:324 | core/anet-core/testdata/prd082/README.md:1 |
| 検証資材 | [core/anet-core/testdata/prd084/README.md](../../core/anet-core/testdata/prd084/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:368, docs/design/200_dqn_agents.jp.md:335 | core/anet-core/testdata/prd084/README.md 公開面と命名 2026-10-11 |
| 検証資材 | [core/anet-core/testdata/prd089/README.md](../../core/anet-core/testdata/prd089/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1055 | core/anet-core/testdata/prd089/README.md 概観/詳細の描画検証 |
| 検証資材 | [core/anet-core/testdata/prd090/README.md](../../core/anet-core/testdata/prd090/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1048 | core/anet-core/testdata/prd090/README.md  5.1 再構築時間と資源 |
| 検証資材 | [core/anet-core/testdata/prd092/README.md](../../core/anet-core/testdata/prd092/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:511 | core/anet-core/testdata/prd092/README.md  証拠と再現 |
| 検証資材 | [core/anet-core/testdata/prd093/README.md](../../core/anet-core/testdata/prd093/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:494, docs/design/200_dqn_agents.jp.md:513 | core/anet-core/testdata/prd093/README.md  NVTX host inclusive範囲 |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | 各workflowのon条件 | CI |  | .github/workflows/gemini_ai_review.yml:1 |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | 各workflowのon条件 | CI | .agents/skills/anet-translate-docs/SKILL.md:8 | .github/workflows/pages.yml:1 |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | 各workflowのon条件 | CI | docs/adr/0017-release-zip-mirrors-repo-layout.md:14 | .github/workflows/release.yml:1 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | 各workflowのon条件 | CI | README.md:3, docs/design/040_development_environment.jp.md:45 | .github/workflows/windows-ci.yml:1 |
| ツール | [apps/runner/tools/compress_workspace_metrics_test.py](../../apps/runner/tools/compress_workspace_metrics_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:369 | apps/runner/tools/compress_workspace_metrics_test.py:1 |
| ツール | [apps/runner/tools/optuna_metrics_gzip_test.py](../../apps/runner/tools/optuna_metrics_gzip_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:375 | apps/runner/tools/optuna_metrics_gzip_test.py:1 |
| ツール | [apps/runner/tools/optuna_workspace_test.py](../../apps/runner/tools/optuna_workspace_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:374 | apps/runner/tools/optuna_workspace_test.py:1 |
| ツール | [viewers/metrics-tools/mlflow_bridge.py](../../viewers/metrics-tools/mlflow_bridge.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:581 | viewers/metrics-tools/mlflow_bridge.py:1 |
| ツール | [viewers/metrics-tools/atari5.py](../../viewers/metrics-tools/atari5.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:664, AGENTS.md:667, AGENTS.md:668 ほか | viewers/metrics-tools/atari5.py:1 |
| ツール | [viewers/metrics-tools/inspect_run.py](../../viewers/metrics-tools/inspect_run.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:632, AGENTS.md:643, AGENTS.md:648 ほか | viewers/metrics-tools/inspect_run.py:1 |
| ツール | [viewers/metrics-tools/inspect_run_test.py](../../viewers/metrics-tools/inspect_run_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:372 | viewers/metrics-tools/inspect_run_test.py:1 |
| ツール | [viewers/metrics-tools/metrics_source_test.py](../../viewers/metrics-tools/metrics_source_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:371 | viewers/metrics-tools/metrics_source_test.py:1 |
| ツール | [viewers/metrics-tools/mlflow_bridge_test.py](../../viewers/metrics-tools/mlflow_bridge_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:370 | viewers/metrics-tools/mlflow_bridge_test.py:1 |
| ツール | [viewers/metrics-tools/tb_bridge_test.py](../../viewers/metrics-tools/tb_bridge_test.py) | AGENTS.mdの該当する検証・分析手順 | 人間・Claude・Codex | AGENTS.md:373 | viewers/metrics-tools/tb_bridge_test.py:1 |
| 検証資材 | [core/anet-core/testdata/prd072/check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) | 設計文書のテスト章から | 人間・Claude・Codex | AGENTS.md:193, docs/design/020_user_guide_run.jp.md:290, docs/design/100_runtime_and_configuration.jp.md:42 | core/anet-core/testdata/prd072/check_default_leaves.py:1 |
| フォルダ | [docs/adr](../adr)（56本） | ファイル名を領域・番号で絞る | Claude・Codex | AGENTS.md、anet-create-prd、実装skill | AGENTS.md §ADRを読む条件／各skill §対象の決定 |
| フォルダ | [docs/memo](../memo)（138本） | ファイル名を領域・番号で絞る | Claude・Codex | AGENTS.md、anet-create-prd、実装skill | AGENTS.md §ADRを読む条件／各skill §対象の決定 |

## 表2 作業 → 読む文書

| これから何をするか | 読むもの | 読まなくてよいとき | 根拠 |
| --- | --- | --- | --- |
| グリルしてPRDを起こす | [docs/wiki/index.md](../wiki/index.md)、関係する知見・実装の地図、設計文書、memo README | 関係するwikiが無ければ「なし」 | anet-create-prd §1・§3・§7、AGENTS.md §wikiを引く条件 |
| 既存PRDを再グリルする | 上記＋対象PRD、引用ADR、CONTEXT.md | 確定事項を開き直す条件が無ければ再決定しない | anet-create-prd §既存PRDの要点提示・再グリルで開き直す条件 |
| 実装計画を作り実装する | 対象PRD、保存済み2ximpl、関連コード・設計文書・テスト | 資料で解けることを質問しない | implement-prd-with-docs §ワークフロー、PRD-IMPLEMENTATION.md |
| 契約・構造を変更する | [docs/wiki/index.md](../wiki/index.md)の実装の地図からPRD・ADR、CONTEXT.md | 日常の修正で構造・名前を変えなければADR・用語の条件読みは省略 | AGENTS.md §wikiを引く条件・ADRを読む条件・用語を引く条件 |
| 設定を書く・変更する | [docs/design/020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) §3.6・§3.7、prd072 README | 設定を変更しないとき | AGENTS.md §設定値の扱い |
| Runを分析する | [docs/design/030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md)、inspect_run.py、wikiの関係ページ | 分析しないとき | AGENTS.md §AIエージェントのRun結果分析ルール・wikiを引く条件 |
| 実験記録を書く | [docs/experiments/README.md](../experiments/README.md)、config複製の規約 | 実験を記録しないとき | AGENTS.md §実験記録と実効config、PRD095 §正本 |
| 検証・ビルドをする | [docs/design/040_development_environment.jp.md](../design/040_development_environment.jp.md) §6・§7、設計文書のテスト章からtestdata README | 文書だけの変更ではC++ build/testを行わない | AGENTS.md §検証・MSVC注意事項、実装計画 |
| 定常整備をする | anet-housekeepingから対象skill・固定出力先 | dry-runは選定まで。実行案の了承前には開始しない | anet-housekeeping §手順5・禁止 |
| レビューする | 対象成果物、プロセス契約、registry、該当品質確認観点 | 対象外のwiki本文は書いた側の検査で確認 | anet-review §2・§3、PRD095 D4 |
| コミットする | prepare-commit、AGENTS.mdのTopic表 | 文案の整理ではstage・commit・pushしない | prepare-commit、AGENTS.md §Git操作 |

## 表3 AGENTS.mdの節の配置

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
| --- | --- | --- | --- | --- |
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:3、本文の適用条件 |
| プロジェクト概要 | 16 | 常時必要(入口に残す) | — | AGENTS.md:26、本文の適用条件 |
| 基本方針 | 9 | 常時必要(入口に残す) | — | AGENTS.md:42、本文の適用条件 |
| クリーンブレーク方針 | 23 | 常時必要(入口に残す) | — | AGENTS.md:51、本文の適用条件 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:74、本文の適用条件 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:95、本文の適用条件 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:102、本文の適用条件 |
| 汎用機構と利用側の責任境界 | 37 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:114、本文の適用条件 |
| Fail-Fast 原則 | 25 | 常時必要(入口に残す) | — | AGENTS.md:151、本文の適用条件 |
| 設定値の扱い | 30 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:176、本文の適用条件 |
| コーディング規約 | 30 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:206、本文の適用条件 |
| コメント・TODO ルール | 10 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:236、本文の適用条件 |
| 性能測定・ProfileRange ルール | 42 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:246、本文の適用条件 |
| GetScalar 実装ルール | 9 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:288、本文の適用条件 |
| Agent 系実装の所有権ルール | 18 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:297、本文の適用条件 |
| ビルド | 32 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:315、本文の適用条件 |
| 検証 | 34 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:347、本文の適用条件 |
| Python 補助ツールの実行 | 22 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:381、本文の適用条件 |
| 人間へ提示するコマンドの書き方 | 27 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:403、本文の適用条件 |
| 編集しない・慎重に扱う領域 | 12 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:430、本文の適用条件 |
| AI エージェントの応答言語ルール | 11 | 常時必要(入口に残す) | — | AGENTS.md:442、本文の適用条件 |
| AI エージェントの概念説明ルール | 11 | 常時必要(入口に残す) | — | AGENTS.md:453、本文の適用条件 |
| ログ出力ルール | 9 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:464、本文の適用条件 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 常時必要(入口に残す) | — | AGENTS.md:473、本文の適用条件 |
| AI エージェントの作業ルール | 26 | 常時必要(入口に残す) | — | AGENTS.md:484、本文の適用条件 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:510、本文の適用条件 |
| Git 操作・コミットメッセージルール | 48 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:522、本文の適用条件 |
| Run 命名・後片付けルール | 36 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:570、本文の適用条件 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:606、本文の適用条件 |
| AI エージェントのRun結果分析ルール | 51 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:638、本文の適用条件 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 50 | 条件付き | 当該領域の文書またはskill（候補） | AGENTS.md:689、本文の適用条件 |
| Agent skills | 19 | 常時必要(入口に残す) | — | AGENTS.md:739、本文の適用条件 |

集計: 条件付き 23節・604行。分類は移設候補を示すだけで、入口は変更しない。

## 表4 skill在庫

| 名前 | 所在 | 実体コピー | shadow | Codex表示名 | 根拠 |
| --- | --- | --- | --- | --- | --- |
| anet-archify-atlas | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-archify-atlas](../../.agents/skills/anet-archify-atlas/SKILL.md)、skills README §追加と更新 |
| anet-audit | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-audit](../../.agents/skills/anet-audit/SKILL.md)、skills README §追加と更新 |
| anet-create-prd | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md)、skills README §追加と更新 |
| anet-harness-map | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-harness-map](../../.agents/skills/anet-harness-map/SKILL.md)、skills README §追加と更新 |
| anet-housekeeping | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md)、skills README §追加と更新 |
| anet-review | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-review](../../.agents/skills/anet-review/SKILL.md)、skills README §追加と更新 |
| anet-stats | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-stats](../../.agents/skills/anet-stats/SKILL.md)、skills README §追加と更新 |
| anet-survey-queue | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-survey-queue](../../.agents/skills/anet-survey-queue/SKILL.md)、skills README §追加と更新 |
| anet-translate-docs | project | 一致 | 機構は両側。実体は個人版 | あり | [anet-translate-docs](../../.agents/skills/anet-translate-docs/SKILL.md)、skills README §追加と更新 |
| grill-me | project | 一致 | 機構は両側。実体は個人版 | なし | [grill-me](../../.agents/skills/grill-me/SKILL.md)、skills README §追加と更新 |
| handoff | project | 一致 | 機構は両側。実体は個人版 | なし | [handoff](../../.agents/skills/handoff/SKILL.md)、skills README §追加と更新 |
| implement-prd-with-docs | project | 一致 | 機構は両側。実体は個人版 | なし | [implement-prd-with-docs](../../.agents/skills/implement-prd-with-docs/SKILL.md)、skills README §追加と更新 |
| improve-codebase-architecture | project | 一致 | 機構は両側。実体は個人版 | なし | [improve-codebase-architecture](../../.agents/skills/improve-codebase-architecture/SKILL.md)、skills README §追加と更新 |
| karpathy-guidelines | project | 一致 | 機構は両側。実体は個人版 | なし | [karpathy-guidelines](../../.agents/skills/karpathy-guidelines/SKILL.md)、skills README §追加と更新 |
| prepare-commit | project | 一致 | 機構は両側。実体は個人版 | あり | [prepare-commit](../../.agents/skills/prepare-commit/SKILL.md)、skills README §追加と更新 |
| setup-matt-pocock-skills | project | 一致 | 機構は両側。実体は個人版 | なし | [setup-matt-pocock-skills](../../.agents/skills/setup-matt-pocock-skills/SKILL.md)、skills README §追加と更新 |
| tdd | project | 一致 | 機構は両側。実体は個人版 | なし | [tdd](../../.agents/skills/tdd/SKILL.md)、skills README §追加と更新 |
| to-prd | project | 一致 | 機構は両側。実体は個人版 | なし | [to-prd](../../.agents/skills/to-prd/SKILL.md)、skills README §追加と更新 |

user-levelだけのskillは集合の差の本数を示し、名前は個人版へ置く。Claude: 33本、Codex: 37本、Codex-system: 1本（根拠: user-levelの各SKILL.mdの存在とprojectとの差集合）。

## 発見事項

- shadow: Claude 6本／中身違い6本、Codex 7本／中身違い6本、Codex-system 0本／中身違い0本。実体は個人版にのみ記録する（根拠: 各user-levelとprojectのファイルhash集合の比較）。
- 実体コピーのドリフト: なし（根拠: 全ファイルの相対パス・SHA-256比較）。
- 欠落: 今回調べた1xx・2xxの設計文書はテストの節を持つ（根拠: 下位見出しを含む各設計文書の見出し）。
- 死んだ動線: 1件。[.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml)。表1の参照元が空の行を数え、入口は除く（根拠: 現行文書・skillのリンクと名指し。Windows形式のパス、コマンドブロック、ファイル名も含む）。
- 機械化点: 実体コピーの同期検査とCIのパス条件化は候補として残す（根拠: skills README §追加と更新、workflowのon条件）。
- evidence gap: 暗黙起動や実行履歴の網羅は検証していない。本文の動線は現行規約を根拠にし、PRD094によるwikiの試走は別セッション待ち（根拠: PRD095 §9）。
- 契約と現行資料の差: harness-map contractの工程例は独立した「再グリル」を含むが、anet-create-prdは7段で独立レビューを持つ。再グリルを別の枠でも行える道として残し、本文の必須手順は現行skillに合わせた。設定変更は実装に畳み、実装計画と編集・検証は同じ実装側ノード内の順序で示す（根拠: contract §workflow、anet-create-prd §1〜7）。
