# AI ハーネスの動線マップ

生成根拠: HEAD `ad6ce58fdb675083b047487419478c6758f7f055` と、その上の現在の作業ツリー（未コミットの変更を含む）。参照元は作業ツリーを `rg` で実測した。生成日 2026-11-19。
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
| 2 | 工程 | PRD の起票・再グリル・調整・実装・検証・レビュー・人間のレビューと、側枝（定常整備・Run 分析・実験記録・ビルド） | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

工程の図のレーンは、人間・精査側・起票側・実装側・共通文書の 5 本にした。設計の 3 工程（起票 → 再グリル → 調整）を 1 列に縦に積むため、精査側を起票側の上に置いた。CI のレーンを外した理由は発見事項の「欠落」に書く。

## 表 1 ノード在庫

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時 | Claude | AGENTS.md:23、.agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37 ほか 3 ファイル | CLAUDE.md:4 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時 | 人間・Claude・Codex | CLAUDE.md:4、.agents/skills/anet-audit/SKILL.md:24、.agents/skills/anet-create-issue/SKILL.md:156 ほか 216 ファイル | AGENTS.md:22 |
| 規約・索引 | [docs/agents/README.md](README.md) | 工程ごとの読む条件を引くとき | 人間・Claude・Codex | .agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37、docs/memo/README.md:45 ほか 12 ファイル | docs/memo/README.md:45 |
| 規約・索引 | [docs/agents/domain.md](domain.md) | ドメイン文書の置き場を引くとき | 人間・Claude・Codex | AGENTS.md:770、docs/memo/done/087_review_agent_quality_viewpoints_10prd.md:449、reports/stats/stats.json:4859 | AGENTS.md:770「Agent skills」 |
| 規約・索引 | [docs/agents/issue-tracker.md](issue-tracker.md) | 課題・PRD・実装計画の置き場を引くとき | 人間・Claude・Codex | AGENTS.md:760、docs/memo/114_findings_intake_and_fix_prd_10prd.md:37、docs/memo/120_issue_work_type_and_prd_link_10prd.md:213 ほか 3 ファイル | AGENTS.md:760「Agent skills」 |
| 規約・索引 | [docs/agents/triage-labels.md](triage-labels.md) | 課題の状態の語を引くとき | 人間・Claude・Codex | AGENTS.md:765、reports/stats/stats.json:8072 | AGENTS.md:765「Agent skills」 |
| 設計文書 | [docs/design/010_framework_overview.jp.md](../../docs/design/010_framework_overview.jp.md) | 拡張点を基盤の外で書けるようにするとき・全体像を引くとき | 人間・Claude・Codex | AGENTS.md:50、.agents/skills/README.md:18、.agents/skills/anet-translate-docs/SKILL.md:12 ほか 26 ファイル | AGENTS.md:50「基本方針」 |
| 設計文書 | [docs/design/011_source_layout.jp.md](../../docs/design/011_source_layout.jp.md) | ソース配置・公開ヘッダ・テストの置き場を変えるとき | 人間・Claude・Codex | docs/design/README.jp.md:37、docs/design/011_source_layout.en.md:1 | docs/design/README.jp.md:37「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/020_user_guide_run.jp.md](../../docs/design/020_user_guide_run.jp.md) | Runner を起動するとき・設定ファイルを書くとき | 人間・Claude・Codex | AGENTS.md:72、docs/design/010_framework_overview.jp.md:223、docs/design/030_user_guide_analysis.jp.md:325 ほか 60 ファイル | AGENTS.md:72「作業ごとに読むもの」 |
| 設計文書 | [docs/design/030_user_guide_analysis.jp.md](../../docs/design/030_user_guide_analysis.jp.md) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:652、docs/design/010_framework_overview.jp.md:381、docs/design/020_user_guide_run.jp.md:375 ほか 38 ファイル | AGENTS.md:652「AI エージェントのRun結果分析ルール」 |
| 設計文書 | [docs/design/040_development_environment.jp.md](../../docs/design/040_development_environment.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:40、docs/design/020_user_guide_run.jp.md:31、docs/design/030_user_guide_analysis.jp.md:24 ほか 15 ファイル | docs/design/README.jp.md:52「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/100_runtime_and_configuration.jp.md](../../docs/design/100_runtime_and_configuration.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:223、docs/design/020_user_guide_run.jp.md:111、docs/design/110_agents_and_learning.jp.md:155 ほか 45 ファイル | docs/design/README.jp.md:60「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/110_agents_and_learning.jp.md](../../docs/design/110_agents_and_learning.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:229、docs/design/020_user_guide_run.jp.md:654、docs/design/100_runtime_and_configuration.jp.md:401 ほか 32 ファイル | docs/design/README.jp.md:61「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/120_environments.jp.md](../../docs/design/120_environments.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:229、docs/design/020_user_guide_run.jp.md:655、docs/design/100_runtime_and_configuration.jp.md:402 ほか 27 ファイル | docs/design/README.jp.md:62「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/130_neural_networks.jp.md](../../docs/design/130_neural_networks.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:253、docs/design/110_agents_and_learning.jp.md:21、docs/design/150_replay_buffer.jp.md:393 ほか 40 ファイル | docs/design/README.jp.md:63「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/140_observability.jp.md](../../docs/design/140_observability.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89、docs/design/020_user_guide_run.jp.md:656、docs/design/030_user_guide_analysis.jp.md:546 ほか 65 ファイル | docs/design/README.jp.md:64「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/150_replay_buffer.jp.md](../../docs/design/150_replay_buffer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:65、.claude/skills/anet-audit/SKILL.md:65、docs/design/010_framework_overview.jp.md:247 ほか 39 ファイル | docs/design/README.jp.md:65「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/160_applications_and_tools.jp.md](../../docs/design/160_applications_and_tools.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:357、docs/design/020_user_guide_run.jp.md:431、docs/design/030_user_guide_analysis.jp.md:547 ほか 43 ファイル | docs/design/README.jp.md:66「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/200_dqn_agents.jp.md](../../docs/design/200_dqn_agents.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:241、docs/design/020_user_guide_run.jp.md:534、docs/design/030_user_guide_analysis.jp.md:236 ほか 78 ファイル | docs/design/README.jp.md:74「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/210_metrics_viewer.jp.md](../../docs/design/210_metrics_viewer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20、docs/design/README.jp.md:75、docs/design/210_metrics_viewer.en.md:1 ほか 20 ファイル | docs/design/README.jp.md:75「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/212_rnd_lifelong_novelty.jp.md](../../docs/design/212_rnd_lifelong_novelty.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/140_observability.jp.md:186、docs/design/200_dqn_agents.jp.md:48、docs/design/214_value_decomposition.jp.md:9 ほか 34 ファイル | docs/design/README.jp.md:76「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/214_value_decomposition.jp.md](../../docs/design/214_value_decomposition.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:104、docs/design/140_observability.jp.md:295、docs/design/200_dqn_agents.jp.md:797 ほか 10 ファイル | docs/design/README.jp.md:77「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/220_atari_env.jp.md](../../docs/design/220_atari_env.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190、docs/design/README.jp.md:78、src/core/base/testdata/prd099/README.md:5 ほか 31 ファイル | docs/design/README.jp.md:78「ドキュメント一覧」・AGENTS.md:431 |
| 設計文書 | [docs/design/README.jp.md](../../docs/design/README.jp.md) | 設計文書の索引を引くとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:411、README.md:37、docs/management/issues/issue-20261115-3e5b77.md:19 ほか 5 ファイル | docs/design/README.jp.md「3. ドキュメント一覧」 |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) | 新しい名前を付けるとき・別の語で呼びたくなったとき | 人間・Claude・Codex | AGENTS.md:140、.agents/skills/README.md:23、.agents/skills/anet-audit/SKILL.md:20 ほか 260 ファイル | AGENTS.md:140 |
| 規約・索引 | [docs/ownership_guideline.md](../../docs/ownership_guideline.md) | Agent 実装の所有権を変えるとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:416、docs/design/110_agents_and_learning.jp.md:55、docs/design/200_dqn_agents.jp.md:849 ほか 19 ファイル | docs/design/README.jp.md:84「関連文書」 |
| 規約・索引 | [docs/memo/README.md](../../docs/memo/README.md) | PRD の採番・工程と担当を引くとき | 人間・Claude・Codex | AGENTS.md:759、.agents/skills/anet-create-prd/SKILL.md:171、.claude/skills/anet-create-prd/SKILL.md:171 ほか 20 ファイル | AGENTS.md:759「Agent skills」 |
| 規約・索引 | [docs/experiments/README.md](../../docs/experiments/README.md) | 実験記録を書く・読むとき | 人間・Claude・Codex | docs/memo/done/095_llm_wiki_10prd.md:15、docs/memo/done/095_llm_wiki_22impl.md:13、docs/wiki/log.md:202 ほか 1 ファイル | AGENTS.md:616（節の対象） |
| 品質確認観点 | [docs/quality/impl.md](../../docs/quality/impl.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | AGENTS.md:65、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:78、.agents/skills/anet-impl-prd/SKILL.md:43 ほか 89 ファイル | AGENTS.md:65 |
| 品質確認観点 | [docs/quality/prd.md](../quality/rfd.md) | PRD を書く・直すとき | 人間・Claude・Codex | AGENTS.md:68、.agents/skills/anet-create-prd/SKILL.md:8、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:192 ほか 31 ファイル | AGENTS.md:68 |
| 品質確認観点 | [docs/quality/refactor.md](../../docs/quality/refactor.md) | リファクタリングの監査・レビューのとき | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:18、.claude/skills/anet-audit/SKILL.md:18、docs/process/registry.md:23 ほか 63 ファイル | docs/process/registry.md:23 |
| プロセス契約 | [docs/process/numbering.md](../../docs/process/numbering.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:21、.claude/skills/anet-create-issue/SKILL.md:21、docs/management/issues/README.md:11 ほか 2 ファイル | skill の手順（参照元の列） |
| プロセス契約 | [docs/process/protocol.md](../../docs/process/protocol.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:237、.agents/skills/anet-create-prd/SKILL.md:8、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:198 ほか 27 ファイル | skill の手順（参照元の列） |
| プロセス契約 | [docs/process/registry.md](../../docs/process/registry.md) | レビュー・課題・プロセス契約を扱うとき | 人間・Claude・Codex | AGENTS.md:70、.agents/skills/anet-create-issue/SKILL.md:15、.agents/skills/anet-create-prd/SKILL.md:8 ほか 29 ファイル | AGENTS.md:70 |
| wiki | [docs/wiki/README.md](../../docs/wiki/README.md) | PRD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:138、.agents/skills/anet-create-prd/SKILL.md:181、.agents/skills/anet-ingest-wiki/SKILL.md:8 ほか 20 ファイル | AGENTS.md:138 |
| wiki | [docs/wiki/index.md](../../docs/wiki/index.md) | PRD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:135、.agents/skills/anet-create-prd/SKILL.md:18、.claude/skills/anet-create-prd/SKILL.md:18 ほか 10 ファイル | AGENTS.md:135 |
| エージェント定義 | [.claude/agents/anet-other-model.md](../../.claude/agents/anet-other-model.md) | 別モデルの検討を頼むとき | Claude | .agents/skills/anet-create-prd/OTHER-MODEL.md:3、.agents/skills/anet-create-prd/SKILL.md:207、.claude/skills/anet-create-prd/OTHER-MODEL.md:3 ほか 4 ファイル | .agents/skills/anet-create-prd/OTHER-MODEL.md:3 |
| エージェント定義 | [.claude/agents/anet-review-agent.md](../../.claude/agents/anet-review-agent.md) | レビューを頼むとき | Claude | .agents/skills/README.md:28、.agents/skills/anet-create-prd/SKILL.md:237、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:218 ほか 28 ファイル | docs/memo/README.md:47 |
| エージェント定義 | [.codex/agents/anet-review-agent.toml](../../.codex/agents/anet-review-agent.toml) | レビューを頼むとき | Codex | .agents/skills/README.md:28、.agents/skills/anet-create-prd/SKILL.md:237、.agents/skills/anet-impl-prd/PRD-IMPLEMENTATION.md:218 ほか 28 ファイル | docs/memo/README.md:47 |
| hook | [.agents/hooks/commit_cue.py](../../.agents/hooks/commit_cue.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:552、docs/design/160_applications_and_tools.jp.md:415、.claude/settings.json:8 ほか 6 ファイル | AGENTS.md:552・.claude/settings.json:8 |
| hook | [.agents/hooks/search_guard.py](../../.agents/hooks/search_guard.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:438、docs/design/160_applications_and_tools.jp.md:416、.claude/settings.json:20 ほか 6 ファイル | AGENTS.md:438・.claude/settings.json:19 |
| hook | [.codex/hooks.json](../../.codex/hooks.json) | エージェントの実行時に自動で効く（読む文書ではない） | Codex | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:6、.agents/hooks/search_guard.py:4 ほか 3 ファイル | .codex/hooks.json:2 |
| hook | [.claude/settings.json](../../.claude/settings.json) | エージェントの実行時に自動で効く（読む文書ではない） | Claude | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:5、.agents/hooks/search_guard.py:4 ほか 3 ファイル | .claude/settings.json:2 |
| 検証資材 | [src/core/base/testdata/prd061/README.md](../../src/core/base/testdata/rfd0061/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:223、docs/design/110_agents_and_learning.en.md:224、reports/stats/stats.json:4522 | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd072/README.md](../../src/core/base/testdata/rfd0072/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:307、docs/design/100_runtime_and_configuration.jp.md:393、docs/quality/impl.md:168 ほか 43 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd078/README.md](../../src/core/base/testdata/rfd0078/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:384、docs/design/150_replay_buffer.en.md:385、docs/memo/done/078_replay_frame_history_integrity_10prd.md:17 ほか 2 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd082/README.md](../../src/core/base/testdata/rfd0082/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:342、docs/design/220_atari_env.en.md:343、docs/memo/126_atari_ram_metric_distinct_and_bit_reducers_10prd.md:219 ほか 2 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd084/README.md](../../src/core/base/testdata/rfd0084/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:385、docs/design/200_dqn_agents.jp.md:609、docs/design/130_neural_networks.en.md:388 ほか 27 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd089/README.md](../../src/core/base/testdata/rfd0089/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1089、docs/design/210_metrics_viewer.en.md:1091、docs/memo/done/089_metricsviewer_comparison_resolution_20impl.md:31 ほか 2 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd090/README.md](../../src/core/base/testdata/rfd0090/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1082、docs/design/210_metrics_viewer.en.md:1084、docs/memo/done/090_metricsviewer_cache_rebuild_20impl.md:50 ほか 1 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd092/README.md](../../src/core/base/testdata/rfd0092/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:834、docs/design/200_dqn_agents.en.md:839、docs/memo/116_rnd_lifelong_novelty_10prd.md:60 ほか 7 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd093/README.md](../../src/core/base/testdata/rfd0093/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:793、docs/design/200_dqn_agents.en.md:797、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd094/README.md](../../src/core/base/testdata/rfd0094/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:180、docs/design/140_observability.jp.md:357、docs/design/160_applications_and_tools.jp.md:139 ほか 10 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd095/README.md](../../src/core/base/testdata/rfd0095/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:432、src/core/base/testdata/prd114/README.md:9、docs/design/160_applications_and_tools.en.md:433 ほか 11 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd097/README.md](../../src/core/base/testdata/rfd0097/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:276、docs/design/220_atari_env.jp.md:352、docs/design/120_environments.en.md:277 ほか 6 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd098/README.md](../../src/core/base/testdata/rfd0098/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:433、docs/design/150_replay_buffer.jp.md:367、docs/design/200_dqn_agents.jp.md:819 ほか 26 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd099/README.md](../../src/core/base/testdata/rfd0099/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:340、docs/design/220_atari_env.en.md:341、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 5 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd100/README.md](../../src/core/base/testdata/rfd0100/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:278、docs/design/120_environments.en.md:289、docs/memo/158_eval_schedule_first_fire_10prd.md:230 ほか 4 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd101/README.md](../../src/core/base/testdata/rfd0101/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:365、docs/design/200_dqn_agents.jp.md:817、docs/design/150_replay_buffer.en.md:366 ほか 4 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd102/README.md](../../src/core/base/testdata/rfd0102/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:290、docs/design/120_environments.en.md:291、docs/memo/158_eval_schedule_first_fire_10prd.md:228 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd103/README.md](../../src/core/base/testdata/rfd0103/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:431、docs/design/200_dqn_agents.jp.md:555、docs/design/140_observability.en.md:432 ほか 15 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd104/README.md](../../src/core/base/testdata/rfd0104/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:161、docs/design/140_observability.jp.md:423、docs/design/150_replay_buffer.jp.md:220 ほか 12 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd105/README.md](../../src/core/base/testdata/rfd0105/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:139、docs/design/160_applications_and_tools.en.md:141、docs/memo/122_small_refactor_and_test_gap_fixes_10prd.md:366 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd106/README.md](../../src/core/base/testdata/rfd0106/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:811、docs/design/200_dqn_agents.en.md:815、docs/experiments/default-dqn/atari/2026-10-29_temporl-skip-forward-stage0.md:294 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd109/README.md](../../src/core/base/testdata/rfd0109/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:428、docs/design/160_applications_and_tools.en.md:429、docs/memo/122_small_refactor_and_test_gap_fixes_10prd.md:296 ほか 8 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd110/README.md](../../src/core/base/testdata/rfd0110/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:426、docs/design/160_applications_and_tools.en.md:427、docs/memo/done/110_runner_main_pane_dynamic_centre_10prd.md:110 ほか 6 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd111/README.md](../../src/core/base/testdata/rfd0111/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1062、docs/design/210_metrics_viewer.en.md:1064、reports/stats/stats.json:5414 | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd114/README.md](../../src/core/base/testdata/rfd0114/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:424、docs/design/160_applications_and_tools.en.md:425、docs/management/issues/issue-20261104-52ffba.md:98 ほか 7 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd115/README.md](../../src/core/base/testdata/rfd0115/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:264、docs/design/140_observability.jp.md:417、docs/design/120_environments.en.md:265 ほか 4 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd116/README.md](../../src/core/base/testdata/rfd0116/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:809、docs/design/212_rnd_lifelong_novelty.jp.md:592、docs/design/200_dqn_agents.en.md:813 ほか 17 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd117/README.md](../../src/core/base/testdata/rfd0117/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:382、docs/design/130_neural_networks.jp.md:372、docs/design/100_runtime_and_configuration.en.md:382 ほか 11 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd121/README.md](../../src/core/base/testdata/rfd0121/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:360、docs/design/200_dqn_agents.jp.md:803、docs/design/212_rnd_lifelong_novelty.jp.md:586 ほか 9 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd122/README.md](../../src/core/base/testdata/rfd0122/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:422、docs/design/160_applications_and_tools.en.md:423、docs/memo/122_small_refactor_and_test_gap_fixes_20impl.md:126 ほか 2 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd123/README.md](../../src/core/base/testdata/rfd0123/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:590、docs/design/212_rnd_lifelong_novelty.en.md:591、docs/memo/123_rnd_intrinsic_reward_scale_10prd.md:423 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd125/README.md](../../src/core/base/testdata/rfd0125/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:807、docs/design/200_dqn_agents.en.md:811、docs/management/issues/issue-20261103-ab98b7.md:67 ほか 5 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd127/README.md](../../src/core/base/testdata/rfd0127/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:415、docs/design/212_rnd_lifelong_novelty.jp.md:557、docs/design/140_observability.en.md:416 ほか 6 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd128/README.md](../../src/core/base/testdata/rfd0128/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:362、docs/design/200_dqn_agents.jp.md:805、docs/design/212_rnd_lifelong_novelty.jp.md:588 ほか 11 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd130/README.md](../../src/core/base/testdata/rfd0130/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:584、docs/design/212_rnd_lifelong_novelty.en.md:585、docs/memo/130_rnd_estimate_statistics_perf_10prd.md:770 ほか 4 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd158/README.md](../../src/core/base/testdata/rfd0158/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:448、docs/design/140_observability.en.md:449、docs/memo/131_prd_9xx_renumber_by_creation_date_10prd.md:69 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd170/README.md](../../src/core/base/testdata/rfd0170/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:389、docs/design/140_observability.jp.md:419、docs/design/130_neural_networks.en.md:392 ほか 3 ファイル | AGENTS.md:308「検証」 |
| 検証資材 | [src/core/base/testdata/prd182/README.md](../../src/core/base/testdata/rfd0182/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/011_source_layout.jp.md:22、docs/design/200_dqn_agents.jp.md:801、docs/design/011_source_layout.en.md:23 ほか 1 ファイル | AGENTS.md:308「検証」 |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/memo/156_cicd_improvement_10prd.md:7、docs/wiki/categories/311_dev_tools.md:153、reports/stats/stats.json:7929 | 実在と参照元を rg で確認 |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8、.claude/skills/anet-translate-docs/SKILL.md:8、docs/memo/156_cicd_improvement_10prd.md:7 ほか 4 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/adr/0017-release-zip-mirrors-repo-layout.md:16、docs/management/issues/issue-20261103-554849.md:41、docs/management/issues/issue-20261103-b13ed2.md:33 ほか 10 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:45、.github/workflows/release.yml:56、README.md:3 ほか 9 ファイル | 実在と参照元を rg で確認 |
| Python tool | [apps/tools/atari5.py](../../apps/tools/atari5.py) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:677、docs/experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md:17、docs/memo/108_metrics_tag_numbering_10prd.md:21 ほか 5 ファイル | AGENTS.md:677 |
| Python tool | [apps/tools/compress_workspace_metrics_test.py](../../apps/tools/compress_workspace_metrics_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:298、docs/wiki/categories/311_dev_tools.md:161、reports/stats/stats.json:7211 | AGENTS.md:298 |
| Python tool | [apps/tools/inspect_run.py](../../apps/tools/inspect_run.py) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:642、docs/design/030_user_guide_analysis.jp.md:363、docs/design/140_observability.jp.md:331 ほか 58 ファイル | AGENTS.md:642 |
| Python tool | [apps/tools/inspect_run_test.py](../../apps/tools/inspect_run_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:301、src/core/base/testdata/prd089/README.md:24、docs/memo/081_evalpanel_episode_trace_10prd.md:150 ほか 16 ファイル | AGENTS.md:301 |
| Python tool | [apps/tools/metrics_source_test.py](../../apps/tools/metrics_source_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:300、docs/memo/done/054_inspect_run_20impl.md:183、docs/memo/done/054_inspect_run_21impl.md:123 ほか 2 ファイル | AGENTS.md:300 |
| Python tool | [apps/tools/mlflow_bridge.py](../../apps/tools/mlflow_bridge.py) | Run フォルダ名の規約を変えるとき | 人間・Claude・Codex | AGENTS.md:591、apps/41_mlflow_bridge.bat:38、apps/tools/resolve_workspace_test.ps1:137 ほか 5 ファイル | AGENTS.md:591 |
| Python tool | [apps/tools/mlflow_bridge_test.py](../../apps/tools/mlflow_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:299、docs/memo/done/046_workspace_20impl.md:30、docs/wiki/categories/311_dev_tools.md:166 ほか 1 ファイル | AGENTS.md:299 |
| Python tool | [apps/tools/optuna_metrics_gzip_test.py](../../apps/tools/optuna_metrics_gzip_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:304、docs/memo/182_source_layout_20impl.md:190、docs/memo/done/046_workspace_25impl.md:28 ほか 3 ファイル | AGENTS.md:304 |
| Python tool | [apps/tools/optuna_workspace_test.py](../../apps/tools/optuna_workspace_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:303、docs/memo/182_source_layout_20impl.md:190、docs/memo/done/046_workspace_25impl.md:22 ほか 3 ファイル | AGENTS.md:303 |
| Python tool | [apps/tools/tb_bridge_test.py](../../apps/tools/tb_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:302、docs/wiki/categories/311_dev_tools.md:175、reports/stats/stats.json:7837 | AGENTS.md:302 |
| Python tool | [reports/tools/stage_issue_memos.py](../../reports/tools/stage_issue_memos.py) | コミットの準備で課題のメモを部分ステージするとき | 人間・Claude・Codex | AGENTS.md:475、.agents/skills/README.md:22、.agents/skills/anet-create-issue/SKILL.md:220 ほか 10 ファイル | AGENTS.md:475 |
| 設計の記録 | [docs/adr/](../../docs/adr/)（81 ファイル） | 契約・構造を変えるとき。ファイル名の決定文を領域の語で絞る | 人間・Claude・Codex | AGENTS.md:102、docs/design/README.jp.md:85 | AGENTS.md「設計判断の記録(ADR)を読む条件」 |
| 指示文書 | [docs/memo/](../../docs/memo/)（直下 112・done 241・frozen 6・dropped 2 ファイル） | PRD は指示、状態と順序は課題。過去の保管場所も検索する | 人間・Claude・Codex | AGENTS.md:756、docs/memo/README.md | docs/agents/issue-tracker.md「Artifact Responsibilities」 |
| 課題記録 | [docs/management/issues/](../../docs/management/issues/)（単票 221 ファイル・INDEX.md・README.md） | 課題の一覧・起票・判定、PRD の起票と実装の完了 | 人間・Claude・Codex | AGENTS.md:487、docs/agents/issue-tracker.md:14 | docs/agents/issue-tracker.md「Artifact Responsibilities」 |

## 表 2 作業から読む文書

| これから何をするか | 読むもの（文書と章） | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| C++ を書く・直す | docs/quality/impl.md の「C++ コード」「名前」「コメントと TODO」「計測」「GetScalar」「所有権」「ログ」「失敗の扱い」 | C++ を変えないとき | AGENTS.md:65「作業ごとに読むもの」 |
| Python・Java・PowerShell を書く・直す | docs/quality/impl.md の「コメントと TODO」「ログ」 | 該当の言語を変えないとき | AGENTS.md:66 |
| 公開ヘッダ・ライブラリの境界・テストの置き場を変える | AGENTS.md「基本方針」の公開ヘッダと置き場の 6 行、docs/design/011_source_layout.jp.md、010 の「4.3 拡張点」 | 既存の境界の中で書くとき | AGENTS.md:47〜52、docs/design/README.jp.md:37、docs/design/010_framework_overview.jp.md:157 |
| 設定ファイルを書く・直す（番号付きの bat の `run.$` を含む） | docs/quality/impl.md の「設定ファイル」「名前」、020 の「3.6 どのファイルに何を書くか」「3.7 例」、`src/core/base/testdata/prd072/check_default_leaves.py` | 設定を変えないとき。設定の変更は実装の工程の中で同時に直す | AGENTS.md:67、docs/design/020_user_guide_run.jp.md:291・307 |
| PRD を起票する（グリル） | `anet-create-prd`、docs/quality/prd.md、impl.md の「名前」、wiki の目録と題材のトピック別まとめ | 実装だけの工程 | AGENTS.md:68・135、docs/memo/README.md:33 |
| PRD を再グリルする（精査） | `anet-create-prd` に既存の PRD を渡す。docs/quality/prd.md、PRD 冒頭の改訂履歴 | 起票と同じ枠では再グリルを置き換えない | docs/memo/README.md:34 |
| 調整する | 人間が裁定し、必要ならエージェント間を仲介する。PRD は self-contained に保つ | - | docs/memo/README.md:35・44 |
| 実装計画を作る | `anet-impl-prd`、PRD、`anet-impl-prd/PRD-IMPLEMENTATION.md`、impl.md の「実装計画」 | skill の実行そのものを計画に代える PRD | docs/memo/README.md:36、.agents/skills/anet-impl-prd/SKILL.md:43 |
| 編集前の条件読み（契約・構造を変える） | 関連する設計文書、ADR（ファイル名で絞る）、wiki のトピック別まとめ | バグ修正・テスト追加・ログ追加・既存 PRD に沿う実装 | AGENTS.md:102〜121・136、AGENTS.md:431 |
| 新しい名前を付ける | CONTEXT.md（`_Avoid_` を含む）、impl.md の「名前」 | 日常の実装・修正 | AGENTS.md:140〜150 |
| 検証する | 領域の設計文書のテスト章 → `src/core/base/testdata/prdNNN/README.md`、AGENTS.md「検証」の標準手順 | 対象外の検証は理由を明記する | AGENTS.md:276〜308 |
| ビルドが詰まる | docs/design/040_development_environment.jp.md の「6. CLI によるビルド・テスト」、AGENTS.md「AI エージェントでのビルド注意事項 (Windows/MSVC)」 | ビルドしない作業 | AGENTS.md:244・702、docs/design/040_development_environment.jp.md:364 |
| レビューする | `anet-review` → `anet-review-agent`、docs/process/protocol.md、docs/process/registry.md、prd.md / impl.md | 対象が無いとき | docs/memo/README.md:38・46〜52 |
| 人間のレビュー・コミットの準備 | `anet-prepare-commit`、AGENTS.md「Git 操作・コミットメッセージルール」 | サブエージェントのとき、工程の途中 | docs/memo/README.md:39、AGENTS.md:471〜578 |
| 設計文書・ユーザーガイドを書く | impl.md の「検証と報告」 | - | AGENTS.md:69 |
| AGENTS.md・品質確認観点・プロセス契約・エージェント定義を直す | impl.md の「検証と報告」、直す観点の「位置づけ」、registry.md の「読み替え表」の前文 | - | AGENTS.md:70 |
| skill を書く・直す | impl.md の「skill」、`.agents/skills/README.md` | - | AGENTS.md:71 |
| Runner を起動する（smoke・Run・bat） | 020 の「2.3 最初に確認する設定」「3.8 確認方法とよくあるエラー」「4.1 標準起動」 | この行による自己点検は無い | AGENTS.md:72・76、docs/design/020_user_guide_run.jp.md:45・372・398 |
| Run を分析する | 030（「6. `inspect_run.py`でRunを検査・抽出する」）、`inspect_run.py`、Atari-5 なら `atari5.py`、wiki の目録 | Run 名だけの提示も分析の依頼として扱う | AGENTS.md:648〜700、docs/design/030_user_guide_analysis.jp.md:363 |
| 実験記録を書く | AGENTS.md「実験記録と実効 config の保存ルール」、docs/experiments/README.md | tmp 付き・評価専用の Run の config は複製しない | AGENTS.md:616〜646・650 |
| 課題を扱う | `anet-create-issue`・`anet-list-issues`・`anet-update-issue`、docs/process/numbering.md | 工程の途中の未決はまず作業記録へ貯める | .agents/skills/anet-create-issue/SKILL.md:21 |
| 定常整備をする | `anet-housekeeping` → 統計・wiki 検査・監査・翻訳・調査・動線マップ・atlas の各 skill | - | .agents/skills/anet-housekeeping/SKILL.md:32〜40、.agents/skills/README.md:31 |

工程の図で畳んだ工程: 「計画・実装」は実装計画・編集前の条件読み・編集・設定の同時更新、「検証」は設計文書のテスト章と testdata の README、「人のレビュー」は人間のレビューとコミット。

## 表 3 AGENTS.md の節の配置

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
|---|---|---|---|---|
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 常時必要 | 入口に残す（文字化けしたまま入口を読まないため） | AGENTS.md:3 |
| プロジェクト概要 | 17 | 常時必要 | 入口に残す | AGENTS.md:26 |
| 基本方針 | 18 | 常時必要 | 入口に残す（公開ヘッダ・テストの置き場の 6 行は C++ を書くときだけ要る。docs/design/011_source_layout.jp.md と重なる） | AGENTS.md:43 |
| 作業ごとに読むもの | 18 | 常時必要 | 入口に残す | AGENTS.md:61 |
| クリーンブレーク方針 | 23 | 条件付き | docs/quality/impl.md（契約を変えるときに読む条件を入口に 1 行残す） | AGENTS.md:79 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | docs/quality/impl.md（読む条件は入口に 1 行残す） | AGENTS.md:102 |
| 設計判断の記録(ADR)を起こす条件 | 10 | 条件付き | .agents/skills/anet-create-prd/ADR-FORMAT.md・docs/process/registry.md | AGENTS.md:123 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | docs/wiki/README.md（読む条件は入口に残す） | AGENTS.md:133 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | docs/quality/impl.md の「名前」 | AGENTS.md:140 |
| 汎用機構と利用側の責任境界 | 41 | 条件付き | docs/quality/impl.md | AGENTS.md:152 |
| Fail-Fast 原則 | 29 | 条件付き | docs/quality/impl.md の「失敗の扱い」 | AGENTS.md:193 |
| 設定値の扱い | 13 | 条件付き | docs/quality/impl.md の「設定ファイル」 | AGENTS.md:222 |
| 改行と文字コード | 9 | 条件付き | docs/quality/impl.md | AGENTS.md:235 |
| ビルド | 32 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:244 |
| 検証 | 34 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:276 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/design/040_development_environment.jp.md の「7. Python 補助ツール」 | AGENTS.md:310 |
| 人間へ提示するコマンドの書き方 | 30 | 条件付き | docs/quality/impl.md | AGENTS.md:332 |
| 編集しない・慎重に扱う領域 | 12 | 常時必要 | 入口に残す | AGENTS.md:362 |
| AI エージェントの応答言語ルール | 30 | 常時必要 | 入口に残す | AGENTS.md:374 |
| AI エージェントの概念説明ルール | 11 | 常時必要 | 入口に残す | AGENTS.md:404 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | .agents/skills/karpathy-guidelines/SKILL.md | AGENTS.md:415 |
| AI エージェントの作業ルール | 33 | 常時必要 | 入口に残す | AGENTS.md:426 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:459 |
| Git 操作・コミットメッセージルール | 109 | 条件付き | .agents/skills/anet-prepare-commit/SKILL.md（git の禁止は入口に残す） | AGENTS.md:471 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/design/020_user_guide_run.jp.md | AGENTS.md:580 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/experiments/README.md | AGENTS.md:616 |
| AI エージェントのRun結果分析ルール | 54 | 条件付き | docs/design/030_user_guide_analysis.jp.md | AGENTS.md:648 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 49 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:702 |
| Agent skills | 20 | 条件付き | docs/agents/issue-tracker.md・triage-labels.md・domain.md | AGENTS.md:751 |
| 集計 | 全 29 節・768 行 | 条件付き 21 節・606 行（到達しない節 0） | 移設はこの作業の対象外 | AGENTS.md の全 `##` 見出し |

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

user-level にしかない skill: `~/.claude/skills` に 36 件、`~/.agents/skills` に 42 件、`~/.codex/skills` に 6 件（同梱の `.system` を含む）。名前は個人版に置く。

## 発見事項

修正はしない。名指しだけを行う。

### shadow

- Claude Code 側 4 本（中身違い 4 本）、Codex 側 4 本（中身違い 4 本）。Codex 同梱の `~/.codex/skills` との同名は 0 本。どの skill かと差分は個人版にだけ書く。

### 実体コピーのドリフト

- 0 本。`diff -rq .agents/skills .claude/skills` の出力は空（54 ファイルすべて一致）。作業ツリーで書き換え中の `anet-create-prd/SKILL.md` も両側で一致している。

### 欠落

- テスト章の無い設計文書: `212_rnd_lifelong_novelty.jp.md`、`220_atari_env.jp.md`（212 の最終 2 章は「4. 具体例」「5. 現在の状態と限界」で testdata README への参照は §5 にある。220 の最終 2 章は「6. ビルド統合」「7. 関連文書」）。
- 「関連文書」章の無い設計文書: `011_source_layout.jp.md`、`212_rnd_lifelong_novelty.jp.md`、`214_value_decomposition.jp.md`。
- 章題・章番号のゆれ: `160_applications_and_tools.jp.md` だけ「テストと変更時の確認事項」（ほかは「テストと拡張時の確認事項」）。テスト章の章番号は 7〜12 に分かれる。PRD182 で増えた `011_source_layout.jp.md` は章番号が無く（「ライブラリとアプリ」「テスト」「出力の置き場」）、索引の 1 行だけから到達する。同じ規則を AGENTS.md「基本方針」の 6 行（AGENTS.md:47〜52）が持つが、011 へのリンクは無い。
- 索引に無い設計文書: `docs/design/optuna.md`（`NNN_*.jp.md` の命名でもない）。030・140・160 の本文と関連文書からは引かれている。
- Codex 表示名（`agents/openai.yaml`）の無い project skill: anet-impl-prd, grill-me, handoff, improve-codebase-architecture, karpathy-guidelines, tdd。汎用 skill にも必須かは決まっていない。
- 資料と実測の不一致: 検証資材の README 31 本（計 129 行）の実行手順に、PRD182 で移す前のパスと実行体名が残る。PRD182 の D23 の決定どおり、先頭に「AGENTS.md「検証」に合わせて読み替える」の注記が 1 行ある。ただし AGENTS.md「検証」は標準手順だけを持ち、旧い名前から新しい名前への対応表は持たない（対応は PRD182 の §5 にある）。動線の末端で、読み替えの材料が入口から引けない。
- 資料と実測の不一致: `docs/agents/issue-tracker.md` と AGENTS.md「Agent skills」（AGENTS.md:755）は実装の課題を `.scratch/` の下（issue-tracker.md では `.scratch/<feature-slug>/issues/`）に置くと書くが、その置き場は 0 件。課題の単票はすべて `docs/management/issues/`（221 件）にある。
- 資料と実測の不一致: 契約は工程図のレーンを「人間 / 起票側 / 精査側 / 実装側 / 共通文書 / CI」の 6 本と定めるが、図は CI を外した 5 本にした。Archify の workflow はレーンの高さが 104px 固定で、6 本では図の高さが 838px 以上になる。showcase の可読性（1440×900 で文字 6px 以上）を満たす幅は 1240px 以下で、画面内に収める横長の比（1.55 以上）と両立しない（6 本では幅 1310px で文字 5.7px になり検証が落ちた）。CI の読む条件と参照元は表 1、push 後の CI は俯瞰図の CI ノードが持つ。
- 資料と実測の不一致: 工程図の「ユーザー環境」は主経路に置いたが、列が足りず「設計」の列に入る。Archify の workflow は列の位置が固定（88・220・300・430・500・625）で、1・2 列目と 3・4 列目は同じレーンに並べられない。

### 死んだ動線

- 0 件（表 1 で参照元が空の行は無い）。
- 参照元が PRD・課題・wiki・ADR・実験記録・レポートだけで、エージェントが読む契約（AGENTS.md・skill・設計文書・品質確認観点・プロセス契約）から名指しされないもの: docs/experiments/README.md、.github/workflows/gemini_ai_review.yml、.github/workflows/release.yml。`docs/experiments/README.md` は AGENTS.md「実験記録と実効 config の保存ルール」が置き場を定めるのに、README 自体へのリンクが無い。

### 機械化点

- 実体コピーの同期検査: `diff -rq .agents/skills .claude/skills` を hook か CI で回す（今は手作業。skills README:3）。
- CI のパス条件化: ハーネスの文書だけの変更で `windows-ci.yml` のビルドを回さない条件。
- 配置替えの後の旧いパスの検索: PRD182 の受入 C5 と同じ `git grep -F` を、ハーネスの文書（検証資材の README を含む）にも定常整備で回す。
- 参照元の実測: 表 1 の参照元の `rg` を定常整備で回し、死んだ動線と、契約から名指しされない末端の増減を見る。
- 設計文書のテスト章と関連文書章の有無、章題を検査する。

### evidence gap

- ユーザー環境（global の指示・user-level skill・memory）の実体と読めたかどうかは個人版にだけ書く。
- shadow が実際にどちらを起動するか（解決順）は実行して確かめていない。名前の一致と中身の差だけを測った。
- `.claude/settings.local.json` はローカルの設定なので読んでいない。
- 参照元は文字列の一致で数えた。リンクの文脈（読む指示か、記録か）は表 1 の「読む条件」で手で補った。生成物（`docs/agents/README.md`、`docs/agents/harness_map_*`、`docs/archify/`、`*.html`）と、全ノードを列挙する `anet-harness-map` skill 自身は参照元から除いた。
- 調査の間、別の作業が `docs/archify/` と `docs/design/*.en.md` を書き換えていた。`.en.md` の行番号は書いた時点のもの。
- C++ の build と test は、この作業に含めず実行していない。
