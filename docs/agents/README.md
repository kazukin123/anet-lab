# AI ハーネスの動線マップ

## 凡例と図の索引

AIハーネスはエージェントの振る舞いを決める指示・規約・文書・skill・memory・CIの総体。入口は毎セッション読むファイル、条件読みは作業に応じた参照、動線は入口から末端までの参照の連なり。shadowは別スコープの同名skillが起動しうる状態、実体コピーは両側に意図して置く同内容の構造である（根拠: anet-harness-map contract「用語」）。

生成根拠のGit revision: `c6425461a3f9635fffc887fce005912afb8fcae6` と作業ツリー。日本語の図ではmeta.localeを省略し、固定Viewer UIとHTMLのlangは英語になる。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
| --- | --- | --- | --- | --- |
| 1 | 俯瞰 | 入口から領域の契約と検証へ進む読む道 | [開く](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [HTML](harness_map_architecture.html) / [JSON](harness_map_architecture.archify.json) |
| 2 | 工程 | PRDづくり・実装・検証と人間の確認 | [開く](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

## 表1 ノード在庫

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
| --- | --- | --- | --- | --- | --- |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・検証・配信の変更時 | 人間・Claude・Codex |  | .github/workflows/gemini_ai_review.yml :1 |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・検証・配信の変更時 | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8 | .github/workflows/pages.yml docs/ 配下のコミット済みファイルだけを GitHub Pages へ配信する。 |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・検証・配信の変更時 | 人間・Claude・Codex | docs/adr/0017-release-zip-mirrors-repo-layout.md:14 | .github/workflows/release.yml :1 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・検証・配信の変更時 | 人間・Claude・Codex | README.md:3, docs/design/040_development_environment.jp.md:45 | .github/workflows/windows-ci.yml :1 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 毎セッション | 人間・Claude・Codex | CLAUDE.md:4, .agents/skills/README.md:40, docs/wiki/README.md:74 | AGENTS.md 長時間のビルド・テスト・バッチ検証 |
| 補助ツール | [apps/runner/tools/compress_workspace_metrics_test.py](../../apps/runner/tools/compress_workspace_metrics_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:270 | apps/runner/tools/compress_workspace_metrics_test.py :1 |
| 補助ツール | [apps/runner/tools/optuna_metrics_gzip_test.py](../../apps/runner/tools/optuna_metrics_gzip_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:276 | apps/runner/tools/optuna_metrics_gzip_test.py :1 |
| 補助ツール | [apps/runner/tools/optuna_workspace_test.py](../../apps/runner/tools/optuna_workspace_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:275 | apps/runner/tools/optuna_workspace_test.py :1 |
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 毎セッション | Claudeのみ | AGENTS.md:23, .agents/skills/anet-harness-map/SKILL.md:28, .agents/skills/anet-harness-map/SKILL.md:31 | CLAUDE.md Claude Code Guide |
| 規約・設計 | [CONTEXT.md](../../CONTEXT.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:119, AGENTS.md:121, AGENTS.md:650 | CONTEXT.md anet-lab |
| 検証資材 | [core/anet-core/testdata/prd061/README.md](../../core/anet-core/testdata/prd061/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:214 | core/anet-core/testdata/prd061/README.md PRD061 検証 |
| 検証資材 | [core/anet-core/testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:364, docs/quality/impl.md:120 | core/anet-core/testdata/prd072/README.md PRD072 検証 |
| 検証資材 | [core/anet-core/testdata/prd078/README.md](../../core/anet-core/testdata/prd078/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:348 | core/anet-core/testdata/prd078/README.md 可視テスト |
| 検証資材 | [core/anet-core/testdata/prd082/README.md](../../core/anet-core/testdata/prd082/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:323 | core/anet-core/testdata/prd082/README.md PRD082 RAM メトリクスの検証 |
| 検証資材 | [core/anet-core/testdata/prd084/README.md](../../core/anet-core/testdata/prd084/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:368, docs/design/200_dqn_agents.jp.md:335 | core/anet-core/testdata/prd084/README.md PRD 084 の検証手順 |
| 検証資材 | [core/anet-core/testdata/prd089/README.md](../../core/anet-core/testdata/prd089/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1055 | core/anet-core/testdata/prd089/README.md PRD089 MetricsViewer比較粒度の検証 |
| 検証資材 | [core/anet-core/testdata/prd090/README.md](../../core/anet-core/testdata/prd090/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1048 | core/anet-core/testdata/prd090/README.md PRD090 キャッシュ保守の検証 |
| 検証資材 | [core/anet-core/testdata/prd092/README.md](../../core/anet-core/testdata/prd092/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:511 | core/anet-core/testdata/prd092/README.md 勾配ノイズスケールの検証資材 |
| 検証資材 | [core/anet-core/testdata/prd093/README.md](../../core/anet-core/testdata/prd093/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:494, docs/design/200_dqn_agents.jp.md:513 | core/anet-core/testdata/prd093/README.md テストの観測面 |
| 検証資材 | [core/anet-core/testdata/prd094/README.md](../../core/anet-core/testdata/prd094/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:162, docs/design/100_runtime_and_configuration.jp.md:356, docs/design/140_observability.jp.md:287 | core/anet-core/testdata/prd094/README.md PRD094 検証資材 |
| 検証資材 | [core/anet-core/testdata/prd095/README.md](../../core/anet-core/testdata/prd095/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:366 | core/anet-core/testdata/prd095/README.md PRD095 段階2の検証 |
| 検証資材 | [core/anet-core/testdata/prd097/README.md](../../core/anet-core/testdata/prd097/README.md) | 設計文書のテスト章から | 人間・Claude・Codex | docs/design/120_environments.jp.md:247, docs/design/220_atari_env.jp.md:333 | core/anet-core/testdata/prd097/README.md PRD097 検証資材（結果所有契約） |
| 規約・設計 | [docs/agents/domain.md](domain.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:651, .agents/skills/setup-matt-pocock-skills/SKILL.md:75, .agents/skills/setup-matt-pocock-skills/SKILL.md:106 | docs/agents/domain.md Domain Docs |
| 規約・設計 | [docs/agents/issue-tracker.md](issue-tracker.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:641, docs/memo/README.md:6, .agents/skills/setup-matt-pocock-skills/SKILL.md:75 | docs/agents/issue-tracker.md Issue Tracker: Local Markdown |
| 規約・設計 | [docs/agents/README.md](README.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/memo/README.md:60, .agents/skills/anet-harness-map/SKILL.md:3, .agents/skills/anet-housekeeping/SKILL.md:37 | docs/agents/README.md AI ハーネスの動線マップ |
| 規約・設計 | [docs/agents/triage-labels.md](triage-labels.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:646, .agents/skills/setup-matt-pocock-skills/SKILL.md:75, .agents/skills/setup-matt-pocock-skills/SKILL.md:102 | docs/agents/triage-labels.md Triage Labels |
| 規約・設計 | [docs/design/010_framework_overview.jp.md](../design/010_framework_overview.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | .agents/skills/README.md:18, .agents/skills/anet-translate-docs/SKILL.md:12, .agents/skills/anet-translate-docs/SKILL.md:17 | docs/design/010_framework_overview.jp.md 6.11 テスト |
| 規約・設計 | [docs/design/020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | README.md:27, docs/design/010_framework_overview.jp.md:210, docs/design/010_framework_overview.jp.md:344 | docs/design/020_user_guide_run.jp.md Run実行ユーザーガイド |
| 規約・設計 | [docs/design/030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:534, README.md:28, docs/design/010_framework_overview.jp.md:368 | docs/design/030_user_guide_analysis.jp.md Run分析ユーザーガイド |
| 規約・設計 | [docs/design/040_development_environment.jp.md](../design/040_development_environment.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | README.md:26, docs/design/010_framework_overview.jp.md:40, docs/design/010_framework_overview.jp.md:356 | docs/design/040_development_environment.jp.md 4.4 テストを実行する |
| 規約・設計 | [docs/design/100_runtime_and_configuration.jp.md](../design/100_runtime_and_configuration.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:210, docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:252 | docs/design/100_runtime_and_configuration.jp.md 8. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/110_agents_and_learning.jp.md](../design/110_agents_and_learning.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:228, docs/design/010_framework_overview.jp.md:362 | docs/design/110_agents_and_learning.jp.md 8. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/120_environments.jp.md](../design/120_environments.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:222, docs/design/010_framework_overview.jp.md:362 | docs/design/120_environments.jp.md 8. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/130_neural_networks.jp.md](../design/130_neural_networks.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:240, docs/design/110_agents_and_learning.jp.md:21, docs/design/110_agents_and_learning.jp.md:232 | docs/design/130_neural_networks.jp.md 8. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/140_observability.jp.md](../design/140_observability.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89, docs/design/010_framework_overview.jp.md:216, docs/design/010_framework_overview.jp.md:338 | docs/design/140_observability.jp.md 9. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:86, docs/design/010_framework_overview.jp.md:234, docs/design/020_user_guide_run.jp.md:570 | docs/design/150_replay_buffer.jp.md 8. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/160_applications_and_tools.jp.md](../design/160_applications_and_tools.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:344, docs/design/010_framework_overview.jp.md:368, docs/design/020_user_guide_run.jp.md:524 | docs/design/160_applications_and_tools.jp.md 5.2 Metrics Viewerの取り込みとviewport range更新 |
| 規約・設計 | [docs/design/200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:228, docs/design/010_framework_overview.jp.md:252, docs/design/010_framework_overview.jp.md:362 | docs/design/200_dqn_agents.jp.md 10. テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/210_metrics_viewer.jp.md](../design/210_metrics_viewer.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20, docs/design/160_applications_and_tools.jp.md:386, docs/design/README.jp.md:74 | docs/design/210_metrics_viewer.jp.md 2.10 取り込みの進行state |
| 規約・設計 | [docs/design/220_atari_env.jp.md](../design/220_atari_env.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190, docs/design/README.jp.md:75 | docs/design/220_atari_env.jp.md 6.1 テストと拡張時の確認事項 |
| 規約・設計 | [docs/design/README.jp.md](../design/README.jp.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | README.md:35, .agents/skills/anet-harness-map/SKILL.md:29, docs/design/010_framework_overview.jp.md:398 | docs/design/README.jp.md ANET ドキュメント |
| 規約・設計 | [docs/experiments/README.md](../experiments/README.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29 | docs/experiments/README.md ハイパーパラメータ探索記録 |
| 規約・設計 | [docs/memo/README.md](../memo/README.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:640, .agents/skills/anet-create-prd/SKILL.md:114, docs/quality/prd.md:29 | docs/memo/README.md docs/memo — PRD と実装計画の置き場 |
| 規約・設計 | [docs/ownership_guideline.md](../ownership_guideline.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | README.md:44, .agents/skills/anet-harness-map/SKILL.md:29, docs/design/010_framework_overview.jp.md:403 | docs/ownership_guideline.md Agent 実装規約：変数・オブジェクトの所有者決定ガイドライン |
| 規約・設計 | [docs/quality/impl.md](../quality/impl.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:59, AGENTS.md:60, AGENTS.md:61 | docs/quality/impl.md テスト |
| 規約・設計 | [docs/quality/prd.md](../quality/prd.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | AGENTS.md:62, docs/memo/README.md:64, .agents/skills/anet-create-prd/SKILL.md:8 | docs/quality/prd.md PRD 品質確認観点 |
| 規約・設計 | [docs/quality/refactor.md](../quality/refactor.md) | 当該領域・作業を扱う前（AGENTS.md・skill） | 人間・Claude・Codex | docs/wiki/README.md:53, .agents/skills/anet-audit/SKILL.md:18 | docs/quality/refactor.md リファクタリング観点 |
| wiki | [docs/wiki/index.md](../wiki/index.md) | PRD・Run分析・契約変更、取り込み・点検 | 人間・Claude・Codex | AGENTS.md:114, docs/wiki/README.md:3, docs/wiki/README.md:19 | docs/wiki/index.md wiki の目録 |
| wiki | [docs/wiki/log.md](../wiki/log.md) | PRD・Run分析・契約変更、取り込み・点検 | 人間・Claude・Codex | AGENTS.md:116, docs/wiki/README.md:3, docs/wiki/README.md:20 | docs/wiki/log.md :1 |
| wiki | [docs/wiki/README.md](../wiki/README.md) | PRD・Run分析・契約変更、取り込み・点検 | 人間・Claude・Codex | AGENTS.md:117, .agents/skills/anet-create-prd/SKILL.md:118, .agents/skills/anet-ingest-wiki/SKILL.md:8 | docs/wiki/README.md 取り込み・点検 |
| 補助ツール | [viewers/metrics-tools/atari5.py](../../viewers/metrics-tools/atari5.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:558, AGENTS.md:561, AGENTS.md:562 | viewers/metrics-tools/atari5.py Atari-5 回帰の係数。Aitchison et al. 2023, Appendix E。 |
| 補助ツール | [viewers/metrics-tools/inspect_run.py](../../viewers/metrics-tools/inspect_run.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:524, AGENTS.md:537, AGENTS.md:542 | viewers/metrics-tools/inspect_run.py trace CSV の固定属性列。data 由来の列はこの後ろへ宣言順で続く。 |
| 補助ツール | [viewers/metrics-tools/inspect_run_test.py](../../viewers/metrics-tools/inspect_run_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:273 | viewers/metrics-tools/inspect_run_test.py :1 |
| 補助ツール | [viewers/metrics-tools/metrics_source_test.py](../../viewers/metrics-tools/metrics_source_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:272 | viewers/metrics-tools/metrics_source_test.py :1 |
| 補助ツール | [viewers/metrics-tools/mlflow_bridge.py](../../viewers/metrics-tools/mlflow_bridge.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:473 | viewers/metrics-tools/mlflow_bridge.py -*- coding: utf-8 -*- |
| 補助ツール | [viewers/metrics-tools/mlflow_bridge_test.py](../../viewers/metrics-tools/mlflow_bridge_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:271 | viewers/metrics-tools/mlflow_bridge_test.py -*- coding: utf-8 -*- |
| 補助ツール | [viewers/metrics-tools/tb_bridge_test.py](../../viewers/metrics-tools/tb_bridge_test.py) | AGENTS.mdが定める検証・Run分析時 | 人間・Claude・Codex | AGENTS.md:274 | viewers/metrics-tools/tb_bridge_test.py :1 |
| 正本の集合 | [docs/adr/](../adr/)（61本） | 状態フォルダ・ファイル名と領域で絞る | 人間・Claude・Codex | AGENTS.mdの読む条件 | AGENTS.md「設計判断の記録」「wikiを引く条件」、docs/memo/README.md |
| 正本の集合 | [docs/memo/](../memo/)（144本） | 状態フォルダ・ファイル名と領域で絞る | 人間・Claude・Codex | AGENTS.mdの読む条件 | AGENTS.md「設計判断の記録」「wikiを引く条件」、docs/memo/README.md |

## 表2 作業から読む文書

| これから何をするか | 読むもの | 読まなくてよいとき | 根拠 |
| --- | --- | --- | --- |
| C++の編集 | 品質確認観点のC++・コメント・計測・GetScalar・所有権・ログ・失敗 | C++を変更しない | AGENTS.md「作業ごとに読むもの」 |
| Python・Java・PowerShellの編集 | 品質確認観点のコメント・ログ | 該当コードを変更しない | AGENTS.md「作業ごとに読むもの」 |
| 設定の変更 | 品質確認観点の設定ファイル、020の設定説明と検査 | 設定契約を変更しない | AGENTS.md「設定値の扱い」、docs/quality/impl.md |
| PRD起票・グリル | wiki目録と関係する地図、PRD品質確認観点、正本、anet-create-prd | 該当ページは無ければ読まない | anet-create-prd 段1・goal anchor |
| 既存PRDの再グリル | 既存本文・ADRとwikiの不採用案・変更条件、前提と後続判断 | 確定事項を開き直す条件が無ければ再決定しない | anet-create-prd「要点提示」「再グリルで開き直す条件」 |
| 実装計画と実装 | PRD・CONTEXT・関連ADR・実装計画の品質観点、保存済み2ximpl | 日常修正でのCONTEXT・ADRは条件による | implement-prd-with-docs、AGENTS.md |
| 構造・契約の変更 | 実装の地図からPRD・ADRと設計文書へ | 単純な修正ではADR全体を読まない | AGENTS.md「wikiを引く条件」「ADRを読む条件」 |
| 名称を付ける | CONTEXT.mdの既存語とAvoid | 既存の名前をそのまま使う | AGENTS.md「用語を引く条件」 |
| Run分析 | 030、inspect_run.py、wiki目録と題材ページ、実験記録の規約 | Run treeやmetrics本文は直接探索しない | AGENTS.md「AI エージェントのRun結果分析ルール」 |
| 分析の書き戻し | PRD背景か実験記録への保存を確かめ、anet-ingest-wiki | 保存を依頼していない材料を推測で書かない | AGENTS.md Run分析、anet-ingest-wiki |
| wiki取り込み・点検 | wiki README、ingest/lint skill、wiki_support、台帳CLIのdocstring | 点検は開始revision以後の材料だけ、取り込み上限あり | anet-ingest-wiki、anet-lint-wiki |
| refactor・architecture監査 | 地図の経緯・不採用案・変更条件、品質観点、台帳list --category | 地図が無ければ省略し、ほかの観点へ強制しない | anet-audit 手順2・4・ガードレール |
| 検証 | 設計文書のテスト章からtestdata READMEへ、MSVC初期化の規約 | 変更に関係しない検証は増やさない | AGENTS.md「検証」、docs/design/*のテスト章 |
| レビューと引き渡し | anet-review、共有プロセス契約、品質観点 | wiki本文は書いた側で検査する | implement-prd-with-docs、PRD095 D4 |
| 定常整備 | anet-housekeepingの候補・予算・了承済み範囲 | dry-runでは成果物を書かない | anet-housekeeping 手順4・5 |
| コミット準備 | prepare-commitとAGENTSのGit規約 | stage・commit・pushは人間 | AGENTS.md「Git操作」、prepare-commit |

## 表3 AGENTS.mdの節の配置

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
| --- | --- | --- | --- | --- |
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:3 |
| プロジェクト概要 | 16 | 常時必要(入口に残す) | — | AGENTS.md:26 |
| 基本方針 | 13 | 常時必要(入口に残す) | — | AGENTS.md:42 |
| 作業ごとに読むもの | 13 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:55 |
| クリーンブレーク方針 | 23 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:68 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:91 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:112 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:119 |
| 汎用機構と利用側の責任境界 | 39 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:131 |
| Fail-Fast 原則 | 25 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:170 |
| 設定値の扱い | 13 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:195 |
| 改行と文字コード | 8 | 常時必要(入口に残す) | — | AGENTS.md:208 |
| ビルド | 32 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:216 |
| 検証 | 34 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:248 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:282 |
| 人間へ提示するコマンドの書き方 | 27 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:304 |
| 編集しない・慎重に扱う領域 | 12 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:331 |
| AI エージェントの応答言語ルール | 11 | 常時必要(入口に残す) | — | AGENTS.md:343 |
| AI エージェントの概念説明ルール | 11 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:354 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:365 |
| AI エージェントの作業ルール | 26 | 常時必要(入口に残す) | — | AGENTS.md:376 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:402 |
| Git 操作・コミットメッセージルール | 48 | 常時必要(入口に残す) | — | AGENTS.md:414 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:462 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:498 |
| AI エージェントのRun結果分析ルール | 53 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:530 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 50 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:583 |
| Agent skills | 19 | 条件付き | docs/agentsの作業別規約（候補） | AGENTS.md:633 |
| 集計 | 527 | 条件付き22節 | 移設は実施しない | 全##節の行数集計 |

## 表4 skill在庫

| 名前 | 所在 | 実体コピーの同期状態 | shadow | Codex表示名 | 根拠 |
| --- | --- | --- | --- | --- | --- |
| [anet-archify-atlas](../../.agents/skills/anet-archify-atlas/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-audit](../../.agents/skills/anet-audit/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-harness-map](../../.agents/skills/anet-harness-map/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-ingest-wiki](../../.agents/skills/anet-ingest-wiki/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-lint-wiki](../../.agents/skills/anet-lint-wiki/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-review](../../.agents/skills/anet-review/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-stats](../../.agents/skills/anet-stats/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-survey-queue](../../.agents/skills/anet-survey-queue/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [anet-translate-docs](../../.agents/skills/anet-translate-docs/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [grill-me](../../.agents/skills/grill-me/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [handoff](../../.agents/skills/handoff/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [implement-prd-with-docs](../../.agents/skills/implement-prd-with-docs/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [improve-codebase-architecture](../../.agents/skills/improve-codebase-architecture/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [karpathy-guidelines](../../.agents/skills/karpathy-guidelines/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [prepare-commit](../../.agents/skills/prepare-commit/SKILL.md) | project | 一致 | 個人版に記録 | あり | SKILL.md・両コピーの全ファイル比較 |
| [setup-matt-pocock-skills](../../.agents/skills/setup-matt-pocock-skills/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [tdd](../../.agents/skills/tdd/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |
| [to-prd](../../.agents/skills/to-prd/SKILL.md) | project | 一致 | 個人版に記録 | なし | SKILL.md・両コピーの全ファイル比較 |

user-levelだけのskill: Claude 33本、Codex 37本、Codex system 1本。名前は個人版へ置く（根拠: SKILL.mdの存在とprojectとの差集合）。

## 発見事項

- shadow: Claude 6本（内容違い6本）、Codex 7本（内容違い6本）、Codex system 0本（内容違い0本）。実体は個人版だけに載せる（根拠: user-levelとprojectの同名skill比較）。
- 実体コピーのドリフト: 0本（根拠: 両ディレクトリのファイル名・SHA-256比較）。
- 欠落: 調べた1xx・2xx設計文書はテストの節を持つ（根拠: 設計文書の見出し）。
- 死んだ動線: 1件。.github/workflows/gemini_ai_review.yml（根拠: 表1の参照元が空の行。入口を除く）。
- 機械化点: skillのコピー比較と参照検査は反復候補。今回、実行の自動化は追加しない（根拠: skills README、anet-harness-map）。
- evidence gap: memoryは件数と索引行数のみを確認し、本文を読む移行は別作業。生成時点の未コミット変更を含むため、固定revisionだけでは同じ内容にならない（根拠: Git status）。
- 契約と現行資料の差: 動線マップの工程例には独立した再グリルがあるが、anet-create-prdは7段の手順である。図は別枠でも再グリルできる道を示し、必須の手順は現行skillによる。設定変更は実装側へ畳む（根拠: harness-map contract、anet-create-prd）。
