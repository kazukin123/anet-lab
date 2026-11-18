# AI ハーネスの動線マップ

生成根拠: HEAD `a9a788643fe3a35eccbe2c8978dc9176bc6650b2` と、その上の現在の作業ツリー（未コミットの変更を含む）。参照元は作業ツリーを `rg` で実測した。
図は日本語で書いており、Viewer UI の対応 locale ではないため `meta.locale` を省いた。固定の Viewer UI と `<html lang>` は英語になる。

## 凡例と図の索引

| 語 | 意味 |
|---|---|
| AI ハーネス | エージェント（Claude Code / Codex）の振る舞いを決めるファイル群の総体。指示・規約・設計文書・PRD/ADR/用語集・検証資材・skill・memory・CI。コード本体は含まない |
| 入口 | 条件なしに毎セッション読まれるファイル。`CLAUDE.md`、`AGENTS.md`、global の `CLAUDE.md`、memory の index |
| 条件読み | 作業の種類や領域で読むかどうかが決まる読み。入口以外の読みはすべてこれ |
| 動線 | 入口から末端（文書の章・skill・ツール）まで、エージェントがたどる参照の連なり。参照元の無い末端は「死んだ動線」 |
| shadow | 別スコープ（user-level）に同名の skill があり、project 版の代わりに起動しうる状態。Claude Code は `~/.claude/skills`、Codex は `~/.agents/skills` |
| 実体コピー | 同じスコープ内で意図して同内容を 2 実体として置いている状態（`.agents/skills` と `.claude/skills`）。同期は手作業 |

shadow は消すべき事故、実体コピーは意図した構造で同期だけが問題なので、両方を「重複」とは呼ばない。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---|---|---|---|---|
| 1 | 俯瞰 | 入口 → 規約 → 領域の設計文書 → テスト章 → testdata README の主経路と、周りの文書・skill・CI・ユーザー環境 | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [HTML](harness_map_architecture.html) / [JSON](harness_map_architecture.archify.json) |
| 2 | 工程 | PRD の起票・再グリル・調整・実装・レビュー・人間のレビューと、側枝（Run 分析・実験記録・ビルド・定常整備） | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

## 表 1 ノード在庫

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時 | Claude | AGENTS.md:23、.agents/skills/anet-harness-map/SKILL.md:28、.agents/skills/anet-harness-map/references/harness-map-contract.md:12 ほか 7 ファイル | CLAUDE.md（実在と参照元を rg で確認） |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時 | 人間・Claude・Codex | CLAUDE.md:4、.agents/skills/anet-audit/SKILL.md:24、.agents/skills/anet-create-issue/SKILL.md:155 ほか 185 ファイル | AGENTS.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/agents/README.md](README.md) | 工程ごとの読む条件を引くとき（docs/memo/README.md「工程と担当」） | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:3、.agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-harness-map/SKILL.md:3 ほか 12 ファイル | docs/agents/README.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/agents/domain.md](domain.md) | ドメイン文書の置き場を引くとき（AGENTS.md「Agent skills」） | 人間・Claude・Codex | AGENTS.md:765、.agents/skills/anet-harness-map/references/harness-map-contract.md:22、.claude/skills/anet-harness-map/references/harness-map-contract.md:22 ほか 1 ファイル | docs/agents/domain.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/agents/issue-tracker.md](issue-tracker.md) | 課題・PRD・実装計画の置き場を引くとき | 人間・Claude・Codex | AGENTS.md:755、.agents/skills/anet-harness-map/references/harness-map-contract.md:22、.claude/skills/anet-harness-map/references/harness-map-contract.md:22 ほか 4 ファイル | docs/agents/issue-tracker.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/agents/triage-labels.md](triage-labels.md) | 課題の状態の語を引くとき | 人間・Claude・Codex | AGENTS.md:760、.agents/skills/anet-harness-map/references/harness-map-contract.md:22、.claude/skills/anet-harness-map/references/harness-map-contract.md:22 ほか 1 ファイル | docs/agents/triage-labels.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/010_framework_overview.jp.md](../../docs/design/010_framework_overview.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | .agents/skills/README.md:18、.agents/skills/anet-translate-docs/SKILL.md:12、.claude/skills/README.md:18 ほか 22 ファイル | docs/design/010_framework_overview.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/020_user_guide_run.jp.md](../../docs/design/020_user_guide_run.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | AGENTS.md:66、docs/quality/impl.md:168、docs/design/010_framework_overview.jp.md:210 ほか 58 ファイル | docs/design/020_user_guide_run.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/030_user_guide_analysis.jp.md](../../docs/design/030_user_guide_analysis.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | AGENTS.md:646、docs/design/010_framework_overview.jp.md:368、docs/design/020_user_guide_run.jp.md:375 ほか 36 ファイル | docs/design/030_user_guide_analysis.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/040_development_environment.jp.md](../../docs/design/040_development_environment.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:40、docs/design/020_user_guide_run.jp.md:31、docs/design/030_user_guide_analysis.jp.md:24 ほか 12 ファイル | docs/design/040_development_environment.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/100_runtime_and_configuration.jp.md](../../docs/design/100_runtime_and_configuration.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:210、docs/design/020_user_guide_run.jp.md:111、docs/design/110_agents_and_learning.jp.md:155 ほか 42 ファイル | docs/design/100_runtime_and_configuration.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/110_agents_and_learning.jp.md](../../docs/design/110_agents_and_learning.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216、docs/design/020_user_guide_run.jp.md:652、docs/design/100_runtime_and_configuration.jp.md:401 ほか 30 ファイル | docs/design/110_agents_and_learning.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/120_environments.jp.md](../../docs/design/120_environments.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:216、docs/design/020_user_guide_run.jp.md:653、docs/design/100_runtime_and_configuration.jp.md:402 ほか 25 ファイル | docs/design/120_environments.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/130_neural_networks.jp.md](../../docs/design/130_neural_networks.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:240、docs/design/110_agents_and_learning.jp.md:21、docs/design/150_replay_buffer.jp.md:393 ほか 35 ファイル | docs/design/130_neural_networks.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/140_observability.jp.md](../../docs/design/140_observability.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89、docs/design/020_user_guide_run.jp.md:654、docs/design/030_user_guide_analysis.jp.md:546 ほか 58 ファイル | docs/design/140_observability.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/150_replay_buffer.jp.md](../../docs/design/150_replay_buffer.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:65、.claude/skills/anet-audit/SKILL.md:65、docs/design/010_framework_overview.jp.md:234 ほか 36 ファイル | docs/design/150_replay_buffer.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/160_applications_and_tools.jp.md](../../docs/design/160_applications_and_tools.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:344、docs/design/020_user_guide_run.jp.md:431、docs/design/030_user_guide_analysis.jp.md:547 ほか 40 ファイル | docs/design/160_applications_and_tools.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/200_dqn_agents.jp.md](../../docs/design/200_dqn_agents.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:228、docs/design/020_user_guide_run.jp.md:534、docs/design/030_user_guide_analysis.jp.md:236 ほか 67 ファイル | docs/design/200_dqn_agents.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/210_metrics_viewer.jp.md](../../docs/design/210_metrics_viewer.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20、docs/design/README.jp.md:74、docs/wiki/apps/120_metrics_viewer.md:7 ほか 16 ファイル | docs/design/210_metrics_viewer.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/212_rnd_lifelong_novelty.jp.md](../../docs/design/212_rnd_lifelong_novelty.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/140_observability.jp.md:186、docs/design/200_dqn_agents.jp.md:48、docs/design/214_value_decomposition.jp.md:9 ほか 29 ファイル | docs/design/212_rnd_lifelong_novelty.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/214_value_decomposition.jp.md](../../docs/design/214_value_decomposition.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:104、docs/design/140_observability.jp.md:295、docs/design/200_dqn_agents.jp.md:797 ほか 4 ファイル | docs/design/214_value_decomposition.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/220_atari_env.jp.md](../../docs/design/220_atari_env.jp.md) | 領域に触れる作業の前（AGENTS.md「AI エージェントの作業ルール」） | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190、docs/design/README.jp.md:77、docs/wiki/categories/322_env_individual.md:89 ほか 26 ファイル | docs/design/220_atari_env.jp.md（実在と参照元を rg で確認） |
| 設計文書 | [docs/design/README.jp.md](../../docs/design/README.jp.md) | 設計文書の索引を引くとき | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29、.agents/skills/anet-harness-map/references/harness-map-contract.md:41、.claude/skills/anet-harness-map/SKILL.md:29 ほか 7 ファイル | docs/design/README.jp.md（実在と参照元を rg で確認） |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) | 新しい名前を付けるとき・別の語で呼びたくなったとき | 人間・Claude・Codex | AGENTS.md:134、.agents/skills/README.md:23、.agents/skills/anet-audit/SKILL.md:20 ほか 252 ファイル | CONTEXT.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/ownership_guideline.md](../../docs/ownership_guideline.md) | Agent 実装の所有権を変えるとき | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29、.agents/skills/anet-harness-map/references/harness-map-contract.md:41、.claude/skills/anet-harness-map/SKILL.md:29 ほか 22 ファイル | docs/ownership_guideline.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/memo/README.md](../../docs/memo/README.md) | PRD の採番・工程と担当を引くとき | 人間・Claude・Codex | AGENTS.md:754、.agents/skills/anet-create-prd/SKILL.md:171、.agents/skills/anet-harness-map/references/harness-map-contract.md:41 ほか 22 ファイル | docs/memo/README.md（実在と参照元を rg で確認） |
| 規約・索引 | [docs/experiments/README.md](../../docs/experiments/README.md) | 実験記録を書く・読むとき | 人間・Claude・Codex | .agents/skills/anet-harness-map/SKILL.md:29、.agents/skills/anet-harness-map/references/harness-map-contract.md:41、.claude/skills/anet-harness-map/SKILL.md:29 ほか 4 ファイル | docs/experiments/README.md（実在と参照元を rg で確認） |
| 品質確認観点 | [docs/quality/impl.md](../../docs/quality/impl.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | AGENTS.md:59、.agents/skills/README.md:27、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:15 ほか 233 ファイル | docs/quality/impl.md（実在と参照元を rg で確認） |
| 品質確認観点 | [docs/quality/prd.md](../../docs/quality/prd.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | AGENTS.md:62、.agents/skills/README.md:23、.agents/skills/anet-create-prd/SKILL.md:8 ほか 543 ファイル | docs/quality/prd.md（実在と参照元を rg で確認） |
| 品質確認観点 | [docs/quality/refactor.md](../../docs/quality/refactor.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:18、.claude/skills/anet-audit/SKILL.md:18、docs/process/registry.md:23 ほか 62 ファイル | docs/quality/refactor.md（実在と参照元を rg で確認） |
| プロセス契約 | [docs/process/numbering.md](../../docs/process/numbering.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:21、.claude/skills/anet-create-issue/SKILL.md:21、docs/management/issues/README.md:11 ほか 1 ファイル | docs/process/numbering.md（実在と参照元を rg で確認） |
| プロセス契約 | [docs/process/protocol.md](../../docs/process/protocol.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:236、.agents/skills/anet-create-prd/SKILL.md:8、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:198 ほか 26 ファイル | docs/process/protocol.md（実在と参照元を rg で確認） |
| プロセス契約 | [docs/process/registry.md](../../docs/process/registry.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | AGENTS.md:64、.agents/skills/anet-create-issue/SKILL.md:15、.agents/skills/anet-create-prd/SKILL.md:8 ほか 27 ファイル | docs/process/registry.md（実在と参照元を rg で確認） |
| wiki | [docs/wiki/README.md](../../docs/wiki/README.md) | PRD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:127、.agents/skills/anet-create-prd/SKILL.md:181、.agents/skills/anet-ingest-wiki/SKILL.md:8 ほか 18 ファイル | docs/wiki/README.md（実在と参照元を rg で確認） |
| wiki | [docs/wiki/index.md](../../docs/wiki/index.md) | PRD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:129、.agents/skills/anet-create-prd/SKILL.md:18、.claude/skills/anet-create-prd/SKILL.md:18 ほか 53 ファイル | docs/wiki/index.md（実在と参照元を rg で確認） |
| エージェント定義 | [.claude/agents/anet-other-model.md](../../.claude/agents/anet-other-model.md) | レビューや別モデルの検討を頼むとき | Claude | .agents/skills/anet-create-prd/OTHER-MODEL.md:3、.agents/skills/anet-create-prd/SKILL.md:207、.claude/skills/anet-create-prd/OTHER-MODEL.md:3 ほか 3 ファイル | .claude/agents/anet-other-model.md（実在と参照元を rg で確認） |
| エージェント定義 | [.claude/agents/anet-review-agent.md](../../.claude/agents/anet-review-agent.md) | レビューや別モデルの検討を頼むとき | Claude | .agents/skills/README.md:28、.agents/skills/anet-create-prd/SKILL.md:235、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:218 ほか 27 ファイル | .claude/agents/anet-review-agent.md（実在と参照元を rg で確認） |
| エージェント定義 | [.codex/agents/anet-review-agent.toml](../../.codex/agents/anet-review-agent.toml) | レビューや別モデルの検討を頼むとき | Codex | .agents/skills/README.md:28、.agents/skills/anet-create-prd/SKILL.md:235、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:218 ほか 27 ファイル | .codex/agents/anet-review-agent.toml（実在と参照元を rg で確認） |
| hook | [.agents/hooks/commit_cue.py](../../.agents/hooks/commit_cue.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:546、.codex/hooks.json:8、docs/design/160_applications_and_tools.jp.md:415 ほか 3 ファイル | .agents/hooks/commit_cue.py（実在と参照元を rg で確認） |
| hook | [.agents/hooks/search_guard.py](../../.agents/hooks/search_guard.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:432、.codex/hooks.json:20、docs/design/160_applications_and_tools.jp.md:416 ほか 3 ファイル | .agents/hooks/search_guard.py（実在と参照元を rg で確認） |
| hook | [.codex/hooks.json](../../.codex/hooks.json) | エージェントの実行時に自動で効く（読む文書ではない） | Codex | docs/design/160_applications_and_tools.jp.md:415、docs/design/160_applications_and_tools.en.md:416 | .codex/hooks.json（実在と参照元を rg で確認） |
| hook | [.claude/settings.json](../../.claude/settings.json) | エージェントの実行時に自動で効く（読む文書ではない） | Claude | docs/design/160_applications_and_tools.jp.md:415、docs/design/160_applications_and_tools.en.md:416、core/anet-core/testdata/prd092/README.md:120 | .claude/settings.json（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd061/README.md](../../core/anet-core/testdata/prd061/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:221、docs/design/110_agents_and_learning.en.md:222 | core/anet-core/testdata/prd061/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/quality/impl.md:168、docs/design/020_user_guide_run.jp.md:307、docs/design/100_runtime_and_configuration.jp.md:393 ほか 36 ファイル | core/anet-core/testdata/prd072/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd078/README.md](../../core/anet-core/testdata/prd078/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:384、docs/design/150_replay_buffer.en.md:385、docs/memo/done/078_replay_frame_history_integrity_10prd.md:17 | core/anet-core/testdata/prd078/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd082/README.md](../../core/anet-core/testdata/prd082/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:342、docs/design/220_atari_env.en.md:343、docs/memo/126_atari_ram_metric_distinct_and_bit_reducers_10prd.md:219 ほか 1 ファイル | core/anet-core/testdata/prd082/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd084/README.md](../../core/anet-core/testdata/prd084/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:383、docs/design/200_dqn_agents.jp.md:609、docs/design/130_neural_networks.en.md:386 ほか 15 ファイル | core/anet-core/testdata/prd084/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd089/README.md](../../core/anet-core/testdata/prd089/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1086、docs/design/210_metrics_viewer.en.md:1088、docs/memo/done/089_metricsviewer_comparison_resolution_20impl.md:31 ほか 1 ファイル | core/anet-core/testdata/prd089/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd090/README.md](../../core/anet-core/testdata/prd090/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1079、docs/design/210_metrics_viewer.en.md:1081、docs/memo/done/090_metricsviewer_cache_rebuild_20impl.md:50 | core/anet-core/testdata/prd090/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd092/README.md](../../core/anet-core/testdata/prd092/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:832、docs/design/200_dqn_agents.en.md:837、docs/memo/116_rnd_lifelong_novelty_10prd.md:60 ほか 6 ファイル | core/anet-core/testdata/prd092/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd093/README.md](../../core/anet-core/testdata/prd093/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:793、docs/design/200_dqn_agents.en.md:797、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 2 ファイル | core/anet-core/testdata/prd093/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd094/README.md](../../core/anet-core/testdata/prd094/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:180、docs/design/140_observability.jp.md:357、docs/design/160_applications_and_tools.jp.md:139 ほか 9 ファイル | core/anet-core/testdata/prd094/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd095/README.md](../../core/anet-core/testdata/prd095/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:432、docs/design/160_applications_and_tools.en.md:433、core/anet-core/testdata/prd114/README.md:7 ほか 9 ファイル | core/anet-core/testdata/prd095/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd097/README.md](../../core/anet-core/testdata/prd097/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:274、docs/design/220_atari_env.jp.md:352、docs/design/120_environments.en.md:275 ほか 5 ファイル | core/anet-core/testdata/prd097/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:433、docs/design/150_replay_buffer.jp.md:367、docs/design/200_dqn_agents.jp.md:817 ほか 15 ファイル | core/anet-core/testdata/prd098/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd099/README.md](../../core/anet-core/testdata/prd099/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:340、docs/design/220_atari_env.en.md:341、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 3 ファイル | core/anet-core/testdata/prd099/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd100/README.md](../../core/anet-core/testdata/prd100/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:276、docs/design/120_environments.en.md:287、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 2 ファイル | core/anet-core/testdata/prd100/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd101/README.md](../../core/anet-core/testdata/prd101/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:365、docs/design/200_dqn_agents.jp.md:815、docs/design/150_replay_buffer.en.md:366 ほか 2 ファイル | core/anet-core/testdata/prd101/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd102/README.md](../../core/anet-core/testdata/prd102/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:288、docs/design/120_environments.en.md:289、docs/memo/158_eval_schedule_first_fire_10prd.md:228 ほか 2 ファイル | core/anet-core/testdata/prd102/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd103/README.md](../../core/anet-core/testdata/prd103/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:431、docs/design/200_dqn_agents.jp.md:555、docs/design/140_observability.en.md:432 ほか 15 ファイル | core/anet-core/testdata/prd103/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd104/README.md](../../core/anet-core/testdata/prd104/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:161、docs/design/140_observability.jp.md:423、docs/design/150_replay_buffer.jp.md:220 ほか 10 ファイル | core/anet-core/testdata/prd104/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd105/README.md](../../core/anet-core/testdata/prd105/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:139、docs/design/160_applications_and_tools.en.md:141、docs/memo/122_small_refactor_and_test_gap_fixes_10prd.md:366 ほか 2 ファイル | core/anet-core/testdata/prd105/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd106/README.md](../../core/anet-core/testdata/prd106/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:809、docs/design/200_dqn_agents.en.md:813、docs/experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md:294 ほか 2 ファイル | core/anet-core/testdata/prd106/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd109/README.md](../../core/anet-core/testdata/prd109/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:428、docs/design/160_applications_and_tools.en.md:429、core/anet-core/testdata/prd122/README.md:10 ほか 3 ファイル | core/anet-core/testdata/prd109/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd110/README.md](../../core/anet-core/testdata/prd110/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:426、docs/design/160_applications_and_tools.en.md:427、docs/memo/done/110_runner_main_pane_dynamic_centre_10prd.md:110 ほか 5 ファイル | core/anet-core/testdata/prd110/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd111/README.md](../../core/anet-core/testdata/prd111/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1059、docs/design/210_metrics_viewer.en.md:1061 | core/anet-core/testdata/prd111/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd114/README.md](../../core/anet-core/testdata/prd114/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:424、docs/design/160_applications_and_tools.en.md:425、core/anet-core/testdata/prd095/README.md:3 ほか 7 ファイル | core/anet-core/testdata/prd114/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd115/README.md](../../core/anet-core/testdata/prd115/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:262、docs/design/140_observability.jp.md:417、docs/design/120_environments.en.md:263 ほか 3 ファイル | core/anet-core/testdata/prd115/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd116/README.md](../../core/anet-core/testdata/prd116/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:807、docs/design/212_rnd_lifelong_novelty.jp.md:592、docs/design/200_dqn_agents.en.md:811 ほか 13 ファイル | core/anet-core/testdata/prd116/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd117/README.md](../../core/anet-core/testdata/prd117/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:382、docs/design/130_neural_networks.jp.md:370、docs/design/100_runtime_and_configuration.en.md:382 ほか 10 ファイル | core/anet-core/testdata/prd117/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd121/README.md](../../core/anet-core/testdata/prd121/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:360、docs/design/200_dqn_agents.jp.md:801、docs/design/212_rnd_lifelong_novelty.jp.md:586 ほか 8 ファイル | core/anet-core/testdata/prd121/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd122/README.md](../../core/anet-core/testdata/prd122/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:422、docs/design/160_applications_and_tools.en.md:423、docs/memo/122_small_refactor_and_test_gap_fixes_20impl.md:126 ほか 1 ファイル | core/anet-core/testdata/prd122/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd123/README.md](../../core/anet-core/testdata/prd123/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:590、docs/design/212_rnd_lifelong_novelty.en.md:591、docs/memo/123_rnd_intrinsic_reward_scale_10prd.md:423 ほか 2 ファイル | core/anet-core/testdata/prd123/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd125/README.md](../../core/anet-core/testdata/prd125/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:805、docs/design/200_dqn_agents.en.md:809、docs/management/issues/issue-20261103-ab98b7.md:67 ほか 3 ファイル | core/anet-core/testdata/prd125/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd127/README.md](../../core/anet-core/testdata/prd127/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:415、docs/design/212_rnd_lifelong_novelty.jp.md:557、docs/design/140_observability.en.md:416 ほか 5 ファイル | core/anet-core/testdata/prd127/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd128/README.md](../../core/anet-core/testdata/prd128/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:362、docs/design/200_dqn_agents.jp.md:803、docs/design/212_rnd_lifelong_novelty.jp.md:588 ほか 10 ファイル | core/anet-core/testdata/prd128/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd130/README.md](../../core/anet-core/testdata/prd130/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:584、docs/design/212_rnd_lifelong_novelty.en.md:585、docs/memo/130_rnd_estimate_statistics_perf_10prd.md:770 ほか 2 ファイル | core/anet-core/testdata/prd130/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd158/README.md](../../core/anet-core/testdata/prd158/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:448、docs/design/140_observability.en.md:449、docs/memo/131_prd_9xx_renumber_by_creation_date_10prd.md:69 ほか 2 ファイル | core/anet-core/testdata/prd158/README.md（実在と参照元を rg で確認） |
| 検証資材 | [core/anet-core/testdata/prd170/README.md](../../core/anet-core/testdata/prd170/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:387、docs/design/140_observability.jp.md:419、docs/design/130_neural_networks.en.md:390 ほか 2 ファイル | core/anet-core/testdata/prd170/README.md（実在と参照元を rg で確認） |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/wiki/categories/311_dev_tools.md:128、docs/memo/156_cicd_improvement_10prd.md:7 | .github/workflows/gemini_ai_review.yml（実在と参照元を rg で確認） |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8、.claude/skills/anet-translate-docs/SKILL.md:8、docs/wiki/categories/311_dev_tools.md:129 ほか 3 ファイル | .github/workflows/pages.yml（実在と参照元を rg で確認） |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/wiki/categories/311_dev_tools.md:31、docs/adr/0017-release-zip-mirrors-repo-layout.md:14、docs/management/issues/issue-20261103-554849.md:41 ほか 9 ファイル | .github/workflows/release.yml（実在と参照元を rg で確認） |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:45、docs/wiki/categories/311_dev_tools.md:55、docs/design/040_development_environment.en.md:46 ほか 8 ファイル | .github/workflows/windows-ci.yml（実在と参照元を rg で確認） |
| Python tool | [apps/runner/tools/compress_workspace_metrics_test.py](../../apps/runner/tools/compress_workspace_metrics_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:292、docs/wiki/apps/110_runner_app.md:177、docs/wiki/categories/311_dev_tools.md:133 | apps/runner/tools/compress_workspace_metrics_test.py（実在と参照元を rg で確認） |
| Python tool | [apps/runner/tools/optuna_metrics_gzip_test.py](../../apps/runner/tools/optuna_metrics_gzip_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:298、docs/wiki/apps/110_runner_app.md:180、docs/wiki/categories/311_dev_tools.md:136 ほか 2 ファイル | apps/runner/tools/optuna_metrics_gzip_test.py（実在と参照元を rg で確認） |
| Python tool | [apps/runner/tools/optuna_workspace_test.py](../../apps/runner/tools/optuna_workspace_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:297、docs/wiki/apps/110_runner_app.md:181、docs/wiki/categories/311_dev_tools.md:137 ほか 2 ファイル | apps/runner/tools/optuna_workspace_test.py（実在と参照元を rg で確認） |
| Python tool | [reports/tools/stage_issue_memos.py](../../reports/tools/stage_issue_memos.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:469、.agents/skills/README.md:22、.agents/skills/anet-create-issue/SKILL.md:219 ほか 9 ファイル | reports/tools/stage_issue_memos.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/atari5.py](../../viewers/metrics-tools/atari5.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:671、docs/wiki/categories/311_dev_tools.md:146、docs/experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md:17 ほか 3 ファイル | viewers/metrics-tools/atari5.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/inspect_run.py](../../viewers/metrics-tools/inspect_run.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:636、.agents/skills/anet-harness-map/references/harness-map-contract.md:77、.claude/skills/anet-harness-map/references/harness-map-contract.md:77 ほか 54 ファイル | viewers/metrics-tools/inspect_run.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/inspect_run_test.py](../../viewers/metrics-tools/inspect_run_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:295、docs/wiki/categories/311_dev_tools.md:148、core/anet-core/testdata/prd089/README.md:22 ほか 14 ファイル | viewers/metrics-tools/inspect_run_test.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/metrics_source_test.py](../../viewers/metrics-tools/metrics_source_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:294、docs/wiki/categories/311_dev_tools.md:149、docs/memo/done/054_inspect_run_20impl.md:183 ほか 1 ファイル | viewers/metrics-tools/metrics_source_test.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/mlflow_bridge.py](../../viewers/metrics-tools/mlflow_bridge.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:585、docs/wiki/categories/311_dev_tools.md:151、docs/memo/done/041_metrics_sqlite_cache_10prd.md:986 ほか 2 ファイル | viewers/metrics-tools/mlflow_bridge.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/mlflow_bridge_test.py](../../viewers/metrics-tools/mlflow_bridge_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:293、docs/wiki/categories/311_dev_tools.md:151、docs/memo/done/046_workspace_20impl.md:30 | viewers/metrics-tools/mlflow_bridge_test.py（実在と参照元を rg で確認） |
| Python tool | [viewers/metrics-tools/tb_bridge_test.py](../../viewers/metrics-tools/tb_bridge_test.py) | AGENTS.md の該当節の作業（検証・Run 分析） | 人間・Claude・Codex | AGENTS.md:296、docs/wiki/categories/311_dev_tools.md:155 | viewers/metrics-tools/tb_bridge_test.py（実在と参照元を rg で確認） |
| 設計の記録 | [docs/adr/](../../docs/adr/)（77 ファイル） | 契約・構造を変えるとき。ファイル名の決定文を領域の語で絞る | 人間・Claude・Codex | AGENTS.md:96、docs/design/README.jp.md:84 | AGENTS.md「設計判断の記録(ADR)を読む条件」 |
| 指示文書 | [docs/memo/](../../docs/memo/)（直下 109・done 241・frozen 6・dropped 2 ファイル） | PRD は指示、状態と順序は課題。過去の保管場所も検索する | 人間・Claude・Codex | AGENTS.md:751、docs/memo/README.md | docs/agents/issue-tracker.md「Artifact Responsibilities」 |
| 課題記録 | [docs/management/issues/](../../docs/management/issues/)（207 ファイル） | 課題の一覧・起票・判定、PRD の起票と実装の完了 | 人間・Claude・Codex | AGENTS.md:481、docs/agents/issue-tracker.md:13 | docs/agents/issue-tracker.md「Artifact Responsibilities」 |

## 表 2 作業から読む文書

| これから何をするか | 読むもの（文書と章） | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| C++ を書く・直す | docs/quality/impl.md の「C++ コード」「名前」「コメントと TODO」「計測」「GetScalar」「所有権」「ログ」「失敗の扱い」 | C++ を変えないとき | AGENTS.md:59「作業ごとに読むもの」 |
| Python・Java・PowerShell を書く・直す | docs/quality/impl.md の「コメントと TODO」「ログ」 | 該当の言語を変えないとき | AGENTS.md:60 |
| 設定ファイルを書く・直す（番号付きの bat の `run.$` を含む） | docs/quality/impl.md の「設定ファイル」「名前」、020 の §3.6〜3.7、`core/anet-core/testdata/prd072/check_default_leaves.py` | 設定を変えないとき。設定の変更は実装の工程の中で同時に直す | AGENTS.md:61、docs/design/020_user_guide_run.jp.md:291 |
| PRD を起票する（グリル） | `anet-create-prd`、docs/quality/prd.md、impl.md の「名前」、wiki の目録と題材のトピック別まとめ | 実装だけの工程 | AGENTS.md:62・129、docs/memo/README.md:33 |
| PRD を再グリルする（精査） | `anet-create-prd` に既存の PRD を渡す。docs/quality/prd.md、PRD 冒頭の改訂履歴 | 起票と同じ枠では再グリルを置き換えない | docs/memo/README.md:34 |
| 実装計画を作る | `anet-impl-prd`、PRD、`anet-impl-prd/PRD-IMPLEMENTATION.md`、impl.md の変更の種類に当たる節 | skill の実行そのものを計画に代える PRD | docs/memo/README.md:36 |
| 編集前の条件読み（契約・構造を変える） | 関連する設計文書、ADR（ファイル名で絞る）、wiki のトピック別まとめ | バグ修正・テスト追加・ログ追加・既存 PRD に沿う実装 | AGENTS.md:96〜115・130 |
| 新しい名前を付ける | CONTEXT.md（`_Avoid_` を含む）、impl.md の「名前」 | 日常の実装・修正 | AGENTS.md:134〜144 |
| 検証する | 領域の設計文書のテスト章 → `core/anet-core/testdata/prdNNN/README.md`、AGENTS.md「検証」の標準手順 | 対象外の検証は理由を明記する | AGENTS.md:270〜302 |
| ビルドが詰まる | docs/design/040_development_environment.jp.md、AGENTS.md「AI エージェントでのビルド注意事項 (Windows/MSVC)」 | ビルドしない作業 | AGENTS.md:238・696 |
| レビューする | `anet-review` → `anet-review-agent`、docs/process/protocol.md、docs/process/registry.md、prd.md / impl.md | 対象が無いとき | docs/memo/README.md:46〜51 |
| 設計文書・ユーザーガイドを書く | impl.md の「検証と報告」 | - | AGENTS.md:63 |
| AGENTS.md・品質確認観点・プロセス契約・エージェント定義を直す | impl.md の「検証と報告」、直す観点の「位置づけ」、registry.md の「読み替え表」の前文 | - | AGENTS.md:64 |
| skill を書く・直す | impl.md の「skill」、`.agents/skills/README.md` | - | AGENTS.md:65 |
| Runner を起動する（smoke・Run・bat） | 020 の「最初に確認する設定」「確認方法とよくあるエラー」「標準起動」 | この行による自己点検は無い | AGENTS.md:66・70 |
| Run を分析する | 030、`inspect_run.py`、Atari-5 なら `atari5.py`、wiki の目録 | Run 名だけの提示も分析の依頼として扱う | AGENTS.md:642〜686 |
| 実験記録を書く | docs/experiments/README.md、AGENTS.md「実験記録と実効 config の保存ルール」 | tmp 付き・評価専用の Run の config は複製しない | AGENTS.md:610〜641 |
| 課題を扱う | `anet-create-issue`・`anet-list-issues`・`anet-update-issue`、docs/process/numbering.md | 工程の途中の未決はまず作業記録へ貯める | .agents/skills/anet-create-issue/SKILL.md:21 |
| コミットの準備をする | `anet-prepare-commit`、AGENTS.md「Git 操作・コミットメッセージルール」 | サブエージェントのとき、工程の途中 | AGENTS.md:465〜572 |
| 定常整備をする | `anet-housekeeping` → 統計・監査・翻訳・調査・動線マップ・atlas の各 skill | - | .agents/skills/anet-housekeeping/SKILL.md |

## 表 3 AGENTS.md の節の配置

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
|---|---|---|---|---|
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 常時必要 | 入口に残す | AGENTS.md:3 |
| プロジェクト概要 | 16 | 常時必要 | 入口に残す | AGENTS.md:26 |
| 基本方針 | 13 | 常時必要 | 入口に残す | AGENTS.md:42 |
| 作業ごとに読むもの | 18 | 常時必要 | 入口に残す | AGENTS.md:55 |
| クリーンブレーク方針 | 23 | 条件付き | docs/quality/impl.md（入口に読む条件を 1 行残す） | AGENTS.md:73 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | docs/quality/impl.md（入口に読む条件を 1 行残す） | AGENTS.md:96 |
| 設計判断の記録(ADR)を起こす条件 | 10 | 条件付き | .agents/skills/anet-create-prd/ADR-FORMAT.md・docs/process/registry.md | AGENTS.md:117 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | docs/wiki/README.md（読む条件は入口に残す） | AGENTS.md:127 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | docs/quality/impl.md の「名前」 | AGENTS.md:134 |
| 汎用機構と利用側の責任境界 | 41 | 条件付き | docs/quality/impl.md | AGENTS.md:146 |
| Fail-Fast 原則 | 29 | 条件付き | docs/quality/impl.md の「失敗の扱い」 | AGENTS.md:187 |
| 設定値の扱い | 13 | 条件付き | docs/quality/impl.md の「設定ファイル」 | AGENTS.md:216 |
| 改行と文字コード | 9 | 条件付き | docs/quality/impl.md | AGENTS.md:229 |
| ビルド | 32 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:238 |
| 検証 | 34 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:270 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/design/040_development_environment.jp.md の §7 | AGENTS.md:304 |
| 人間へ提示するコマンドの書き方 | 30 | 条件付き | docs/quality/impl.md | AGENTS.md:326 |
| 編集しない・慎重に扱う領域 | 12 | 常時必要 | 入口に残す | AGENTS.md:356 |
| AI エージェントの応答言語ルール | 30 | 常時必要 | 入口に残す | AGENTS.md:368 |
| AI エージェントの概念説明ルール | 11 | 常時必要 | 入口に残す | AGENTS.md:398 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | .agents/skills/karpathy-guidelines/SKILL.md | AGENTS.md:409 |
| AI エージェントの作業ルール | 33 | 常時必要 | 入口に残す | AGENTS.md:420 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:453 |
| Git 操作・コミットメッセージルール | 109 | 条件付き | .agents/skills/anet-prepare-commit/SKILL.md（git の禁止は入口に残す） | AGENTS.md:465 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/design/020_user_guide_run.jp.md | AGENTS.md:574 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/experiments/README.md | AGENTS.md:610 |
| AI エージェントのRun結果分析ルール | 54 | 条件付き | docs/design/030_user_guide_analysis.jp.md | AGENTS.md:642 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 50 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:696 |
| Agent skills | 20 | 条件付き | docs/agents/issue-tracker.md・triage-labels.md・domain.md | AGENTS.md:746 |
| 集計 | 全 29 節・763 行 | 条件付き 21 節・607 行 | 移設はこの作業の対象外 | AGENTS.md の全 `##` 見出し |

## 表 4 skill 在庫

| 名前 | ファイル | 所在 | 実体コピーの同期 | shadow（Claude 側・Codex 側） | Codex 表示名 | 根拠 |
|---|---|---|---|---|---|---|
| （README） | README.md | project | 一致 | 個人版に記録 | - | .agents/skills/README.md・.claude/skills/README.md |
| anet-archify-atlas | anet-archify-atlas/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-archify-atlas/SKILL.md・.claude/skills/anet-archify-atlas/SKILL.md |
| anet-archify-atlas | anet-archify-atlas/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-archify-atlas/agents/openai.yaml・.claude/skills/anet-archify-atlas/agents/openai.yaml |
| anet-archify-atlas | anet-archify-atlas/references/atlas-contract.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-archify-atlas/references/atlas-contract.md・.claude/skills/anet-archify-atlas/references/atlas-contract.md |
| anet-audit | anet-audit/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-audit/SKILL.md・.claude/skills/anet-audit/SKILL.md |
| anet-audit | anet-audit/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-audit/agents/openai.yaml・.claude/skills/anet-audit/agents/openai.yaml |
| anet-create-issue | anet-create-issue/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-issue/SKILL.md・.claude/skills/anet-create-issue/SKILL.md |
| anet-create-issue | anet-create-issue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-issue/agents/openai.yaml・.claude/skills/anet-create-issue/agents/openai.yaml |
| anet-create-prd | anet-create-prd/ADR-FORMAT.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-prd/ADR-FORMAT.md・.claude/skills/anet-create-prd/ADR-FORMAT.md |
| anet-create-prd | anet-create-prd/CONTEXT-FORMAT.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-prd/CONTEXT-FORMAT.md・.claude/skills/anet-create-prd/CONTEXT-FORMAT.md |
| anet-create-prd | anet-create-prd/OTHER-MODEL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-prd/OTHER-MODEL.md・.claude/skills/anet-create-prd/OTHER-MODEL.md |
| anet-create-prd | anet-create-prd/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-prd/SKILL.md・.claude/skills/anet-create-prd/SKILL.md |
| anet-create-prd | anet-create-prd/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-prd/agents/openai.yaml・.claude/skills/anet-create-prd/agents/openai.yaml |
| anet-harness-map | anet-harness-map/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/SKILL.md・.claude/skills/anet-harness-map/SKILL.md |
| anet-harness-map | anet-harness-map/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/agents/openai.yaml・.claude/skills/anet-harness-map/agents/openai.yaml |
| anet-harness-map | anet-harness-map/references/harness-map-contract.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/references/harness-map-contract.md・.claude/skills/anet-harness-map/references/harness-map-contract.md |
| anet-housekeeping | anet-housekeeping/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-housekeeping/SKILL.md・.claude/skills/anet-housekeeping/SKILL.md |
| anet-housekeeping | anet-housekeeping/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-housekeeping/agents/openai.yaml・.claude/skills/anet-housekeeping/agents/openai.yaml |
| anet-impl-prd | anet-impl-prd/PRD-IMPLEMENTATION.md | project | 一致 | 個人版に記録 | なし | .agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md・.claude/skills/anet-impl-prd/PRD-IMPLEMENTATION.md |
| anet-impl-prd | anet-impl-prd/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/anet-impl-prd/SKILL.md・.claude/skills/anet-impl-prd/SKILL.md |
| anet-improve | anet-improve/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-improve/SKILL.md・.claude/skills/anet-improve/SKILL.md |
| anet-improve | anet-improve/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-improve/agents/openai.yaml・.claude/skills/anet-improve/agents/openai.yaml |
| anet-ingest-wiki | anet-ingest-wiki/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-ingest-wiki/SKILL.md・.claude/skills/anet-ingest-wiki/SKILL.md |
| anet-ingest-wiki | anet-ingest-wiki/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-ingest-wiki/agents/openai.yaml・.claude/skills/anet-ingest-wiki/agents/openai.yaml |
| anet-lint-wiki | anet-lint-wiki/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-lint-wiki/SKILL.md・.claude/skills/anet-lint-wiki/SKILL.md |
| anet-lint-wiki | anet-lint-wiki/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-lint-wiki/agents/openai.yaml・.claude/skills/anet-lint-wiki/agents/openai.yaml |
| anet-list-issues | anet-list-issues/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-list-issues/SKILL.md・.claude/skills/anet-list-issues/SKILL.md |
| anet-list-issues | anet-list-issues/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-list-issues/agents/openai.yaml・.claude/skills/anet-list-issues/agents/openai.yaml |
| anet-prepare-commit | anet-prepare-commit/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-prepare-commit/SKILL.md・.claude/skills/anet-prepare-commit/SKILL.md |
| anet-prepare-commit | anet-prepare-commit/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-prepare-commit/agents/openai.yaml・.claude/skills/anet-prepare-commit/agents/openai.yaml |
| anet-review | anet-review/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-review/SKILL.md・.claude/skills/anet-review/SKILL.md |
| anet-review | anet-review/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-review/agents/openai.yaml・.claude/skills/anet-review/agents/openai.yaml |
| anet-stats | anet-stats/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-stats/SKILL.md・.claude/skills/anet-stats/SKILL.md |
| anet-stats | anet-stats/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-stats/agents/openai.yaml・.claude/skills/anet-stats/agents/openai.yaml |
| anet-survey-queue | anet-survey-queue/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-survey-queue/SKILL.md・.claude/skills/anet-survey-queue/SKILL.md |
| anet-survey-queue | anet-survey-queue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-survey-queue/agents/openai.yaml・.claude/skills/anet-survey-queue/agents/openai.yaml |
| anet-translate-docs | anet-translate-docs/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-translate-docs/SKILL.md・.claude/skills/anet-translate-docs/SKILL.md |
| anet-translate-docs | anet-translate-docs/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-translate-docs/agents/openai.yaml・.claude/skills/anet-translate-docs/agents/openai.yaml |
| anet-update-issue | anet-update-issue/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-update-issue/SKILL.md・.claude/skills/anet-update-issue/SKILL.md |
| anet-update-issue | anet-update-issue/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-update-issue/agents/openai.yaml・.claude/skills/anet-update-issue/agents/openai.yaml |
| grill-me | grill-me/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/grill-me/SKILL.md・.claude/skills/grill-me/SKILL.md |
| handoff | handoff/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/handoff/SKILL.md・.claude/skills/handoff/SKILL.md |
| improve-codebase-architecture | improve-codebase-architecture/DEEPENING.md | project | 一致 | 個人版に記録 | なし | .agents/skills/improve-codebase-architecture/DEEPENING.md・.claude/skills/improve-codebase-architecture/DEEPENING.md |
| improve-codebase-architecture | improve-codebase-architecture/HTML-REPORT.md | project | 一致 | 個人版に記録 | なし | .agents/skills/improve-codebase-architecture/HTML-REPORT.md・.claude/skills/improve-codebase-architecture/HTML-REPORT.md |
| improve-codebase-architecture | improve-codebase-architecture/INTERFACE-DESIGN.md | project | 一致 | 個人版に記録 | なし | .agents/skills/improve-codebase-architecture/INTERFACE-DESIGN.md・.claude/skills/improve-codebase-architecture/INTERFACE-DESIGN.md |
| improve-codebase-architecture | improve-codebase-architecture/LANGUAGE.md | project | 一致 | 個人版に記録 | なし | .agents/skills/improve-codebase-architecture/LANGUAGE.md・.claude/skills/improve-codebase-architecture/LANGUAGE.md |
| improve-codebase-architecture | improve-codebase-architecture/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/improve-codebase-architecture/SKILL.md・.claude/skills/improve-codebase-architecture/SKILL.md |
| karpathy-guidelines | karpathy-guidelines/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/karpathy-guidelines/SKILL.md・.claude/skills/karpathy-guidelines/SKILL.md |
| tdd | tdd/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/SKILL.md・.claude/skills/tdd/SKILL.md |
| tdd | tdd/deep-modules.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/deep-modules.md・.claude/skills/tdd/deep-modules.md |
| tdd | tdd/interface-design.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/interface-design.md・.claude/skills/tdd/interface-design.md |
| tdd | tdd/mocking.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/mocking.md・.claude/skills/tdd/mocking.md |
| tdd | tdd/refactoring.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/refactoring.md・.claude/skills/tdd/refactoring.md |
| tdd | tdd/tests.md | project | 一致 | 個人版に記録 | なし | .agents/skills/tdd/tests.md・.claude/skills/tdd/tests.md |

user-level にしかない skill: `~/.claude/skills` に 38 件、`~/.agents/skills` に 42 件、`~/.codex/skills` に 2 件。名前は個人版に置く。

## 発見事項

修正はしない。名指しだけを行う。

### shadow

- Claude Code 側 4 本（中身違い 4 本）、Codex 側 4 本（中身違い 4 本）。Codex 同梱の `~/.codex/skills` との同名は 0 本。どの skill かと差分は個人版にだけ書く。

### 実体コピーのドリフト

- 0 本。`diff -rq .agents/skills .claude/skills` の出力は空（54 ファイルすべて一致）。

### 欠落

- テスト章の無い設計文書: `212_rnd_lifelong_novelty.jp.md`（最終 2 章は「4. 具体例」「5. 現在の状態と限界」。testdata README への参照は §5 にある）、`220_atari_env.jp.md`（最終 2 章は「6. ビルド統合」「7. 関連文書」）。
- 「関連文書」章の無い設計文書: `212_rnd_lifelong_novelty.jp.md`、`214_value_decomposition.jp.md`（最終章が「7. テストと拡張時の確認事項」）。
- 章題のゆれ: `160_applications_and_tools.jp.md` だけ「テストと変更時の確認事項」（ほかは「テストと拡張時の確認事項」）。テスト章の章番号は 7〜12 に分かれる。
- Codex 表示名（`agents/openai.yaml`）の無い project skill: anet-impl-prd, grill-me, handoff, improve-codebase-architecture, karpathy-guidelines, tdd。汎用 skill にも必須かは決まっていない。
- 資料と実測の不一致: `docs/agents/issue-tracker.md` と AGENTS.md「Agent skills」は実装の課題を `.scratch/<feature>/` と書くが、課題の単票の正本は `docs/management/issues/`（207 件）にある。両方の置き場が並んで書かれており、どちらを先に引くかは文書から決まらない。
- 資料と実測の不一致: 契約は工程図のレーンを「人間 / 起票側 / 精査側 / 実装側 / 共通文書 / CI」の 6 本と定めるが、今回の図は 4 本（共通文書 / 起票・実装・レビュー / 精査・検証・整備 / 人間）に畳んでいる。Archify の workflow は列 6 本・同じレーンで隣の列を使えない制約があり、時間内に 6 レーンでの配置を試していない。CI の読む条件は表 1・表 2 が持つ。

### 死んだ動線

- 0 件（表 1 で参照元が空の行は無い）。
- 参照元が PRD・課題・wiki・ADR・実験記録だけで、エージェントが読む契約（AGENTS.md・skill・設計文書・品質確認観点）から名指しされないもの: .github/workflows/gemini_ai_review.yml、.github/workflows/release.yml。

### 機械化点

- 実体コピーの同期検査: `diff -rq .agents/skills .claude/skills` を hook か CI で回す（今は手作業。skills README:3）。
- CI のパス条件化: ハーネスの文書だけの変更で `windows-ci.yml` のビルドを回さない条件。
- 参照元の実測: 表 1 の参照元の `rg` を定常整備で回し、死んだ動線の増減を見る。
- 設計文書のテスト章の有無と章題を検査する。

### evidence gap

- ユーザー環境（global の指示・user-level skill・memory）の実体と読めたかどうかは個人版にだけ書く。
- shadow が実際にどちらを起動するか（解決順）は実行して確かめていない。名前の一致と中身の差だけを測った。
- `.claude/settings.local.json` はローカルの設定なので読んでいない。
- 参照元は文字列の一致で数えた。リンクの文脈（読む指示か、記録か）は表 1 の「読む条件」で手で補った。
- C++ の build と test は、この作業に含めず実行していない。

