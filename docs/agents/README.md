# AI ハーネスの動線マップ

生成根拠: HEAD `9310b087270668127f996cbb04eb959fc756f1de` と、その上の作業ツリー。参照元は作業ツリーの追跡ファイルと、ignore していない新しいファイルで実測した。生成日 2026-11-25。
図は日本語で書いており、Viewer UI の対応 locale ではないため `meta.locale` を省いた。固定の Viewer UI と `<html lang>` は英語になる。

## 凡例と図の索引

| 語 | 意味 |
|---|---|
| AI ハーネス | エージェント（Claude Code / Codex）の振る舞いを決めるファイル群の総体。指示・規約・設計文書・RFD/ADR/用語集・検証資材・skill・memory・CI。コード本体は含まない |
| 入口 | 条件なしに毎セッション読まれるファイル。`CLAUDE.md`、`AGENTS.md`、global の `CLAUDE.md`、memory の index |
| 条件読み | 作業の種類や領域で読むかどうかが決まる読み。入口以外の読みはすべてこれ |
| 動線 | 入口から末端（文書の章・skill・ツール）まで、エージェントがたどる参照の連なり。参照元の無い末端は「死んだ動線」 |
| shadow | 別スコープ（user-level）に同名の skill があり、project 版の代わりに起動しうる状態。Claude Code は `~/.claude/skills`、Codex は `~/.agents/skills` |
| 実体コピー | 同じスコープ内で意図して同内容を 2 実体として置いている状態（`.agents/skills` と `.claude/skills`）。同期は手作業 |

shadow は消すべき事故、実体コピーは意図した構造で同期だけが問題なので、両方を「重複」とは呼ばない。

| 順序 | 図 | 何が分かるか | 表示 | ファイル |
|---|---|---|---|---|
| 1 | 俯瞰 | 入口 → 規約 → 領域の設計文書 → テスト章 → testdata README の主経路と、周りの文書・skill・CI・ユーザー環境 | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_architecture.html) | [HTML](harness_map_architecture.html) / [JSON](harness_map_architecture.archify.json) |
| 2 | 工程 | RFD の起票・再グリル・調整・実装・検証・レビュー・人間のレビューと、側枝（定常整備・Run 分析・実験記録・ビルド） | [表示](https://kazukin123.github.io/anet-lab/docs/agents/harness_map_workflow.html) | [HTML](harness_map_workflow.html) / [JSON](harness_map_workflow.archify.json) |

工程の図のレーンは、人間・精査側・起票側・実装側・共通文書の 5 本にした。設計の 3 工程（起票 → 再グリル → 調整）を 1 列に縦に積むため、精査側を起票側の上に置いた。CI のレーンを外した理由は発見事項の「欠落」に書く。

## 表 1 ノード在庫

参照元は、パスかファイル名の文字列を引いた結果の先頭 3 ファイル（行は各ファイルで最初に当たった行）と、残りのファイル数。並びは入口・skill・設計文書と規約・hook と CI・検証資材とソース・英訳・RFD と課題・wiki・その他の順。

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時 | Claude | AGENTS.md:23、.agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37 ほか 12 ファイル | CLAUDE.md:4 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時 | 人間・Claude・Codex | CLAUDE.md:4、.agents/skills/anet-audit/SKILL.md:24、.agents/skills/anet-create-issue/SKILL.md:156 ほか 254 ファイル | AGENTS.md:22 |
| 規約・索引 | [README.md](../../README.md) | プロジェクトの概要と文書の置き場を引くとき（人間の入口） | 人間・Claude・Codex | docs/design/README.jp.md:83、docs/wiki/README.md:3、docs/design/README.en.md:84 ほか 48 ファイル | README.md:35・docs/design/README.jp.md:83 |
| 規約・索引 | [docs/agents/README.md](README.md) | 工程ごとの読む条件を引くとき | 人間・Claude・Codex | .agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37、docs/memo/README.md:47 ほか 16 ファイル | docs/memo/README.md:47 |
| 設計文書 | [docs/design/010_framework_overview.jp.md](../../docs/design/010_framework_overview.jp.md) | 拡張点を基盤の外で書けるようにするとき・全体像を引くとき | 人間・Claude・Codex | AGENTS.md:348、.agents/skills/README.md:18、.agents/skills/anet-translate-docs/SKILL.md:12 ほか 36 ファイル | AGENTS.md:348「基本方針」 |
| 設計文書 | [docs/design/011_source_layout.jp.md](../../docs/design/011_source_layout.jp.md) | ソース配置・公開ヘッダ・テストの置き場を変えるとき | 人間・Claude・Codex | docs/design/README.jp.md:37、docs/design/011_source_layout.en.md:1、docs/wiki/apps/110_runner_app.md:13 ほか 7 ファイル | docs/design/README.jp.md:37「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/020_user_guide_run.jp.md](../../docs/design/020_user_guide_run.jp.md) | Runner を起動するとき・設定ファイルを書くとき | 人間・Claude・Codex | AGENTS.md:56、docs/design/010_framework_overview.jp.md:224、docs/design/030_user_guide_analysis.jp.md:331 ほか 62 ファイル | AGENTS.md:56「作業ごとに読むもの」 |
| 設計文書 | [docs/design/030_user_guide_analysis.jp.md](../../docs/design/030_user_guide_analysis.jp.md) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:731、docs/design/010_framework_overview.jp.md:382、docs/design/020_user_guide_run.jp.md:375 ほか 43 ファイル | AGENTS.md:731「AI エージェントのRun結果分析ルール」 |
| 設計文書 | [docs/design/040_development_environment.jp.md](../../docs/design/040_development_environment.jp.md) | ビルド・テストの環境を扱うとき・ビルドが詰まるとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:40、docs/design/020_user_guide_run.jp.md:31、docs/design/030_user_guide_analysis.jp.md:24 ほか 16 ファイル | docs/design/README.jp.md:52「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/100_runtime_and_configuration.jp.md](../../docs/design/100_runtime_and_configuration.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:224、docs/design/020_user_guide_run.jp.md:111、docs/design/110_agents_and_learning.jp.md:160 ほか 46 ファイル | docs/design/README.jp.md:60「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/110_agents_and_learning.jp.md](../../docs/design/110_agents_and_learning.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:230、docs/design/020_user_guide_run.jp.md:654、docs/design/100_runtime_and_configuration.jp.md:407 ほか 35 ファイル | docs/design/README.jp.md:61「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/120_environments.jp.md](../../docs/design/120_environments.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:230、docs/design/020_user_guide_run.jp.md:655、docs/design/100_runtime_and_configuration.jp.md:408 ほか 30 ファイル | docs/design/README.jp.md:62「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/130_neural_networks.jp.md](../../docs/design/130_neural_networks.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:254、docs/design/110_agents_and_learning.jp.md:21、docs/design/135_forwarder_and_perception.jp.md:5 ほか 48 ファイル | docs/design/README.jp.md:63「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/135_forwarder_and_perception.jp.md](../../docs/design/135_forwarder_and_perception.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:47、docs/design/130_neural_networks.jp.md:41、docs/design/150_replay_buffer.jp.md:76 ほか 15 ファイル | docs/design/README.jp.md:64「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/140_observability.jp.md](../../docs/design/140_observability.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89、docs/design/020_user_guide_run.jp.md:656、docs/design/030_user_guide_analysis.jp.md:552 ほか 75 ファイル | docs/design/README.jp.md:65「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/150_replay_buffer.jp.md](../../docs/design/150_replay_buffer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:65、.claude/skills/anet-audit/SKILL.md:65、docs/design/010_framework_overview.jp.md:248 ほか 50 ファイル | docs/design/README.jp.md:66「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/160_applications_and_tools.jp.md](../../docs/design/160_applications_and_tools.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:358、docs/design/020_user_guide_run.jp.md:431、docs/design/030_user_guide_analysis.jp.md:553 ほか 51 ファイル | docs/design/README.jp.md:67「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/200_dqn_agents.jp.md](../../docs/design/200_dqn_agents.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:242、docs/design/020_user_guide_run.jp.md:534、docs/design/030_user_guide_analysis.jp.md:242 ほか 89 ファイル | docs/design/README.jp.md:75「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/210_metrics_viewer.jp.md](../../docs/design/210_metrics_viewer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20、docs/design/README.jp.md:76、docs/design/210_metrics_viewer.en.md:1 ほか 22 ファイル | docs/design/README.jp.md:76「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/212_rnd_lifelong_novelty.jp.md](../../docs/design/212_rnd_lifelong_novelty.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/140_observability.jp.md:188、docs/design/150_replay_buffer.jp.md:116、docs/design/200_dqn_agents.jp.md:48 ほか 46 ファイル | docs/design/README.jp.md:77「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/214_value_decomposition.jp.md](../../docs/design/214_value_decomposition.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:104、docs/design/140_observability.jp.md:297、docs/design/200_dqn_agents.jp.md:809 ほか 19 ファイル | docs/design/README.jp.md:78「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/220_atari_env.jp.md](../../docs/design/220_atari_env.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190、docs/design/README.jp.md:79、apps/runner/config/Atari.txt:1358 ほか 32 ファイル | docs/design/README.jp.md:79「ドキュメント一覧」・AGENTS.md:152 |
| 設計文書 | [docs/design/README.jp.md](../../docs/design/README.jp.md) | 設計文書の索引を引くとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:412、README.md:37、docs/design/README.en.md:1 ほか 7 ファイル | docs/design/README.jp.md「3. ドキュメント一覧」 |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) | 新しい名前を付けるとき・別の語で呼びたくなったとき | 人間・Claude・Codex | AGENTS.md:506、.agents/skills/README.md:23、.agents/skills/anet-audit/SKILL.md:20 ほか 291 ファイル | AGENTS.md:506 |
| 規約・索引 | [docs/ownership_guideline.md](../../docs/ownership_guideline.md) | Agent 実装の所有権を変えるとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:417、docs/design/110_agents_and_learning.jp.md:59、docs/design/200_dqn_agents.jp.md:861 ほか 19 ファイル | docs/design/README.jp.md:85「関連文書」 |
| 規約・索引 | [docs/memo/README.md](../../docs/memo/README.md) | RFD の採番・工程と担当・リンク規約を引くとき | 人間・Claude・Codex | .agents/skills/anet-create-rfd/SKILL.md:212、.claude/skills/anet-create-rfd/SKILL.md:212、docs/process/registry.md:62 ほか 24 ファイル | .agents/skills/anet-create-rfd/SKILL.md:212 |
| 規約・索引 | [docs/experiments/README.md](../../docs/experiments/README.md) | 実験記録を書く・読むとき | 人間・Claude・Codex | README.md:65、docs/memo/done/0095_llm_wiki_10rfd.md:15、docs/memo/done/0095_llm_wiki_22impl.md:13 ほか 2 ファイル | AGENTS.md:695（節の対象） |
| 規約・索引 | [reports/INDEX.md](../../reports/INDEX.md) | 調査のキューを回すとき・RFD の問いで事例を確かめるとき | 人間・Claude・Codex | .agents/skills/README.md:19、.agents/skills/anet-create-rfd/SKILL.md:144、.agents/skills/anet-housekeeping/SKILL.md:36 ほか 29 ファイル | .agents/skills/anet-create-rfd/SKILL.md:144・.agents/skills/anet-survey-queue/SKILL.md:3 |
| レビュー観点 | [docs/quality/impl.md](../../docs/quality/impl.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | AGENTS.md:49、.agents/skills/anet-create-rfd/SKILL.md:146、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:78 ほか 102 ファイル | AGENTS.md:49 |
| レビュー観点 | [docs/quality/rfd.md](../../docs/quality/rfd.md) | RFD を書く・直すとき | 人間・Claude・Codex | AGENTS.md:52、.agents/skills/anet-create-rfd/SKILL.md:8、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:192 ほか 23 ファイル | AGENTS.md:52 |
| レビュー観点 | [docs/quality/refactor.md](../../docs/quality/refactor.md) | リファクタリングの監査のとき | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:18、.claude/skills/anet-audit/SKILL.md:18、docs/process/registry.md:23 ほか 67 ファイル | docs/process/registry.md:23 |
| プロセス契約 | [docs/process/numbering.md](../../docs/process/numbering.md) | 課題などの ID を採番するとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:21、.claude/skills/anet-create-issue/SKILL.md:21、docs/management/issues/README.md:11 ほか 3 ファイル | .agents/skills/anet-create-issue/SKILL.md:21 |
| プロセス契約 | [docs/process/protocol.md](../../docs/process/protocol.md) | レビューを依頼する・課題の候補を挙げるとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:237、.agents/skills/anet-create-rfd/SKILL.md:8、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:198 ほか 33 ファイル | .agents/skills/anet-review/SKILL.md:3 |
| プロセス契約 | [docs/process/registry.md](../../docs/process/registry.md) | 成果物ごとの観点を引くとき・プロセス契約とレビューエージェントの定義を直すとき | 人間・Claude・Codex | AGENTS.md:54、.agents/skills/anet-create-issue/SKILL.md:15、.agents/skills/anet-create-rfd/SKILL.md:8 ほか 34 ファイル | AGENTS.md:54 |
| プロセス契約 | [docs/process/summary-page/README.md](../../docs/process/summary-page/README.md) | 会話のまとめページを作る・更新するとき | 人間・Claude・Codex | AGENTS.md:103、.agents/skills/anet-summary-page/SKILL.md:8、.claude/skills/anet-summary-page/SKILL.md:8 ほか 4 ファイル | AGENTS.md:103 |
| プロセス契約 | [docs/process/summary-page/grill.md](../../docs/process/summary-page/grill.md) | グリルの会話のまとめページを作るとき | 人間・Claude・Codex | docs/process/summary-page/README.md:9、docs/management/issues/issue-20261122-717309.md:59、docs/memo/0197_conversation_options_summary_page_10rfd.md:281 ほか 1 ファイル | docs/process/summary-page/README.md:9「場面の一覧」 |
| プロセス契約 | [docs/process/summary-page/exp.md](../../docs/process/summary-page/exp.md) | 実験の会話のまとめページを作るとき | 人間・Claude・Codex | docs/process/summary-page/README.md:10、reports/stats/stats.json:6742 | docs/process/summary-page/README.md:10「場面の一覧」 |
| プロセス契約 | [docs/process/summary-page/general.md](../../docs/process/summary-page/general.md) | ほかの場面のまとめページを作るとき | 人間・Claude・Codex | docs/process/summary-page/README.md:11、reports/stats/stats.json:6743 | docs/process/summary-page/README.md:11「場面の一覧」 |
| wiki | [docs/wiki/README.md](../../docs/wiki/README.md) | wiki を書く・直すとき | 人間・Claude・Codex | AGENTS.md:504、.agents/skills/anet-create-rfd/SKILL.md:224、.agents/skills/anet-ingest-wiki/SKILL.md:8 ほか 24 ファイル | AGENTS.md:504 |
| wiki | [docs/wiki/index.md](../../docs/wiki/index.md) | RFD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:501、.agents/skills/anet-create-rfd/SKILL.md:18、.claude/skills/anet-create-rfd/SKILL.md:18 ほか 12 ファイル | AGENTS.md:501 |
| エージェント定義 | [.claude/agents/anet-other-model.md](../../.claude/agents/anet-other-model.md) | 別モデルの検討を頼むとき | Claude | .agents/skills/anet-create-rfd/OTHER-MODEL.md:3、.agents/skills/anet-create-rfd/SKILL.md:263、.claude/skills/anet-create-rfd/OTHER-MODEL.md:3 ほか 5 ファイル | .agents/skills/anet-create-rfd/SKILL.md:263 |
| エージェント定義 | [.claude/agents/anet-review-agent.md](../../.claude/agents/anet-review-agent.md) | レビューを頼むとき | Claude | .agents/skills/README.md:29、.agents/skills/anet-create-rfd/SKILL.md:296、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:218 ほか 30 ファイル | docs/memo/README.md:49 |
| エージェント定義 | [.codex/agents/anet-review-agent.toml](../../.codex/agents/anet-review-agent.toml) | レビューを頼むとき | Codex | .agents/skills/README.md:29、.agents/skills/anet-create-rfd/SKILL.md:296、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:218 ほか 30 ファイル | docs/memo/README.md:49 |
| hook | [.agents/hooks/commit_cue.py](../../.agents/hooks/commit_cue.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:311、.claude/settings.json:8、.codex/hooks.json:8 ほか 8 ファイル | AGENTS.md:311・.claude/settings.json:8 |
| hook | [.agents/hooks/search_guard.py](../../.agents/hooks/search_guard.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:159、.claude/settings.json:20、.codex/hooks.json:20 ほか 8 ファイル | AGENTS.md:159・.claude/settings.json:20 |
| hook | [.codex/hooks.json](../../.codex/hooks.json) | エージェントの実行時に自動で効く（読む文書ではない） | Codex | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:6、.agents/hooks/search_guard.py:4 ほか 4 ファイル | .codex/hooks.json:2 |
| hook | [.claude/settings.json](../../.claude/settings.json) | エージェントの実行時に自動で効く（読む文書ではない） | Claude | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:5、.agents/hooks/search_guard.py:4 ほか 4 ファイル | .claude/settings.json:2 |
| 検証資材 | [src/core/base/testdata/rfd0061/README.md](../../src/core/base/testdata/rfd0061/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:228、docs/design/110_agents_and_learning.en.md:229、docs/memo/0195_document_term_rfd_and_4digit_numbers_20impl.md:237 ほか 1 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0072/README.md](../../src/core/base/testdata/rfd0072/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:307、docs/design/100_runtime_and_configuration.jp.md:399、docs/quality/impl.md:173 ほか 29 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0078/README.md](../../src/core/base/testdata/rfd0078/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:387、docs/design/150_replay_buffer.en.md:388、reports/stats/stats.json:6869 | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0082/README.md](../../src/core/base/testdata/rfd0082/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:342、docs/design/220_atari_env.en.md:343、docs/memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md:219 ほか 1 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0084/README.md](../../src/core/base/testdata/rfd0084/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:394、docs/design/200_dqn_agents.jp.md:621、src/core/base/testdata/rfd0084/run_forward_refactor.ps1:135 ほか 23 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0089/README.md](../../src/core/base/testdata/rfd0089/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1141、docs/design/210_metrics_viewer.en.md:1143、docs/memo/0196_metrics_viewer_run_comparison_10rfd.md:385 ほか 2 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0090/README.md](../../src/core/base/testdata/rfd0090/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1134、docs/design/210_metrics_viewer.en.md:1136、docs/memo/0195_document_term_rfd_and_4digit_numbers_20impl.md:299 ほか 1 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0092/README.md](../../src/core/base/testdata/rfd0092/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:846、docs/design/200_dqn_agents.en.md:851、docs/memo/0116_rnd_lifelong_novelty_10rfd.md:60 ほか 9 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0093/README.md](../../src/core/base/testdata/rfd0093/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:805、docs/design/200_dqn_agents.en.md:809、docs/memo/0158_eval_schedule_first_fire_10rfd.md:230 ほか 2 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0094/README.md](../../src/core/base/testdata/rfd0094/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:184、docs/design/140_observability.jp.md:359、docs/design/160_applications_and_tools.jp.md:139 ほか 9 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0095/README.md](../../src/core/base/testdata/rfd0095/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:432、src/core/base/testdata/rfd0114/README.md:9、docs/design/160_applications_and_tools.en.md:433 ほか 8 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0097/README.md](../../src/core/base/testdata/rfd0097/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:278、docs/design/220_atari_env.jp.md:352、docs/design/120_environments.en.md:279 ほか 9 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0098/README.md](../../src/core/base/testdata/rfd0098/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:440、docs/design/150_replay_buffer.jp.md:370、docs/design/200_dqn_agents.jp.md:831 ほか 25 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0099/README.md](../../src/core/base/testdata/rfd0099/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:340、src/env/atari/src/AtariEnv_test.cpp:1776、docs/design/220_atari_env.en.md:341 ほか 5 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0100/README.md](../../src/core/base/testdata/rfd0100/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:280、src/core/base/testdata/rfd0102/run_validation.py:62、docs/design/120_environments.en.md:291 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0101/README.md](../../src/core/base/testdata/rfd0101/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:368、docs/design/200_dqn_agents.jp.md:829、src/core/base/testdata/rfd0101/check_contract.py:15 ほか 4 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0102/README.md](../../src/core/base/testdata/rfd0102/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:292、src/env/gridmaze/src/GridMazeEnv_test.cpp:583、docs/design/120_environments.en.md:293 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0103/README.md](../../src/core/base/testdata/rfd0103/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:438、docs/design/200_dqn_agents.jp.md:565、src/algo/dqn/src/dqn_concrete_test.cpp:1369 ほか 17 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0104/README.md](../../src/core/base/testdata/rfd0104/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:166、docs/design/140_observability.jp.md:430、docs/design/150_replay_buffer.jp.md:223 ほか 10 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0105/README.md](../../src/core/base/testdata/rfd0105/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:139、docs/design/160_applications_and_tools.en.md:141、docs/memo/0122_small_refactor_and_test_gap_fixes_10rfd.md:366 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0106/README.md](../../src/core/base/testdata/rfd0106/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:823、docs/design/200_dqn_agents.en.md:827、docs/memo/0158_eval_schedule_first_fire_10rfd.md:230 ほか 2 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0109/README.md](../../src/core/base/testdata/rfd0109/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:428、src/core/base/testdata/rfd0110/run_validation.py:15、src/core/base/testdata/rfd0121/run_validation.py:31 ほか 8 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0110/README.md](../../src/core/base/testdata/rfd0110/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:426、docs/design/160_applications_and_tools.en.md:427、docs/memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md:329 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0111/README.md](../../src/core/base/testdata/rfd0111/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1113、docs/design/210_metrics_viewer.en.md:1115、reports/stats/stats.json:6891 | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0114/README.md](../../src/core/base/testdata/rfd0114/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:424、docs/design/160_applications_and_tools.en.md:425、docs/memo/0114_findings_intake_and_fix_rfd_20impl.md:195 ほか 6 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0115/README.md](../../src/core/base/testdata/rfd0115/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:266、docs/design/140_observability.jp.md:424、docs/design/120_environments.en.md:267 ほか 5 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0116/README.md](../../src/core/base/testdata/rfd0116/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:821、docs/design/212_rnd_lifelong_novelty.jp.md:379、src/core/base/testdata/rfd0116/run_performance_atari_test.py:10 ほか 14 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0117/README.md](../../src/core/base/testdata/rfd0117/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:387、docs/design/130_neural_networks.jp.md:381、docs/design/100_runtime_and_configuration.en.md:387 ほか 10 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0121/README.md](../../src/core/base/testdata/rfd0121/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:363、docs/design/200_dqn_agents.jp.md:815、docs/design/212_rnd_lifelong_novelty.jp.md:373 ほか 10 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0122/README.md](../../src/core/base/testdata/rfd0122/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:422、docs/design/160_applications_and_tools.en.md:423、docs/memo/0125_dqn_learner_update_skeleton_10rfd.md:117 ほか 2 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0123/README.md](../../src/core/base/testdata/rfd0123/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:377、docs/design/212_rnd_lifelong_novelty.en.md:378、docs/memo/0123_rnd_intrinsic_reward_scale_10rfd.md:424 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0125/README.md](../../src/core/base/testdata/rfd0125/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:819、src/core/base/testdata/rfd0128/run_validation.py:15、docs/design/200_dqn_agents.en.md:823 ほか 4 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0127/README.md](../../src/core/base/testdata/rfd0127/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:422、docs/design/212_rnd_lifelong_novelty.jp.md:418、docs/design/140_observability.en.md:423 ほか 6 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0128/README.md](../../src/core/base/testdata/rfd0128/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:365、docs/design/200_dqn_agents.jp.md:817、docs/design/212_rnd_lifelong_novelty.jp.md:375 ほか 10 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0130/README.md](../../src/core/base/testdata/rfd0130/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:371、src/core/base/testdata/rfd0130/run_performance.py:49、docs/design/212_rnd_lifelong_novelty.en.md:372 ほか 4 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0158/README.md](../../src/core/base/testdata/rfd0158/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:455、docs/design/140_observability.en.md:456、docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md:69 ほか 1 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0170/README.md](../../src/core/base/testdata/rfd0170/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:398、docs/design/140_observability.jp.md:426、docs/design/130_neural_networks.en.md:401 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0182/README.md](../../src/core/base/testdata/rfd0182/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/011_source_layout.jp.md:23、docs/design/200_dqn_agents.jp.md:813、docs/design/011_source_layout.en.md:24 ほか 2 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0191/README.md](../../src/core/base/testdata/rfd0191/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:369、docs/design/212_rnd_lifelong_novelty.en.md:370、docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md:787 ほか 3 ファイル | AGENTS.md:584「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0194/README.md](../../src/core/base/testdata/rfd0194/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:388、src/core/base/testdata/rfd0191/README.md:28、docs/design/100_runtime_and_configuration.en.md:388 ほか 3 ファイル | AGENTS.md:584「検証」 |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/memo/0156_cicd_improvement_10rfd.md:7、docs/wiki/categories/311_dev_tools.md:202、reports/stats/stats.json:9507 | 実在と参照元を rg で確認 |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8、.claude/skills/anet-translate-docs/SKILL.md:8、docs/memo/0156_cicd_improvement_10rfd.md:7 ほか 4 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/adr/0017-release-zip-mirrors-repo-layout.md:16、docs/management/issues/issue-20261103-554849.md:41、docs/management/issues/issue-20261103-b13ed2.md:33 ほか 12 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:45、README.md:3、.github/workflows/release.yml:56 ほか 9 ファイル | 実在と参照元を rg で確認 |
| Python tool | [apps/tools/atari5.py](../../apps/tools/atari5.py) | Atari-5 を 1 つのベンチマークとして読むとき | 人間・Claude・Codex | AGENTS.md:756、docs/memo/0108_metrics_tag_numbering_10rfd.md:21、docs/memo/done/0103_temporl_reference_alignment_10rfd.md:322 ほか 5 ファイル | AGENTS.md:756 |
| Python tool | [apps/tools/compress_workspace_metrics_test.py](../../apps/tools/compress_workspace_metrics_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:574、docs/wiki/categories/311_dev_tools.md:210、reports/stats/stats.json:5882 | AGENTS.md:574 |
| Python tool | [apps/tools/inspect_run.py](../../apps/tools/inspect_run.py) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:721、docs/design/030_user_guide_analysis.jp.md:369、docs/design/140_observability.jp.md:333 ほか 62 ファイル | AGENTS.md:734 |
| Python tool | [apps/tools/inspect_run_test.py](../../apps/tools/inspect_run_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:577、src/core/base/testdata/rfd0089/README.md:24、docs/memo/0081_evalpanel_episode_trace_10rfd.md:150 ほか 16 ファイル | AGENTS.md:577 |
| Python tool | [apps/tools/metrics_source_test.py](../../apps/tools/metrics_source_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:576、docs/memo/done/0054_inspect_run_20impl.md:183、docs/memo/done/0054_inspect_run_21impl.md:123 ほか 2 ファイル | AGENTS.md:576 |
| Python tool | [apps/tools/mlflow_bridge.py](../../apps/tools/mlflow_bridge.py) | Run フォルダ名の規約を変えるとき | 人間・Claude・Codex | AGENTS.md:670、apps/41_mlflow_bridge.bat:38、apps/tools/resolve_workspace_test.ps1:137 ほか 5 ファイル | AGENTS.md:670 |
| Python tool | [apps/tools/mlflow_bridge_test.py](../../apps/tools/mlflow_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:575、docs/memo/done/0046_workspace_20impl.md:30、docs/wiki/categories/311_dev_tools.md:215 ほか 1 ファイル | AGENTS.md:575 |
| Python tool | [apps/tools/optuna_metrics_gzip_test.py](../../apps/tools/optuna_metrics_gzip_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:580、docs/memo/0182_source_layout_20impl.md:190、docs/memo/done/0046_workspace_25impl.md:28 ほか 3 ファイル | AGENTS.md:580 |
| Python tool | [apps/tools/optuna_workspace_test.py](../../apps/tools/optuna_workspace_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:579、docs/memo/0182_source_layout_20impl.md:190、docs/memo/done/0046_workspace_25impl.md:22 ほか 3 ファイル | AGENTS.md:579 |
| Python tool | [apps/tools/tb_bridge_test.py](../../apps/tools/tb_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:578、docs/wiki/categories/311_dev_tools.md:224、reports/stats/stats.json:5901 | AGENTS.md:578 |
| Python tool | [reports/tools/stage_issue_memos.py](../../reports/tools/stage_issue_memos.py) | コミットの準備で課題のメモを部分ステージするとき | 人間・Claude・Codex | AGENTS.md:234、.agents/skills/README.md:22、.agents/skills/anet-create-issue/SKILL.md:220 ほか 11 ファイル | AGENTS.md:234 |
| 設計の記録 | [docs/adr/](../../docs/adr/)（96 ファイル） | 契約・構造を変えるとき。ファイル名の決定文を領域の語で絞る | 人間・Claude・Codex | AGENTS.md:470、.agents/skills/README.md:23、.agents/skills/anet-create-rfd/ADR-FORMAT.md:3 ほか 208 ファイル | AGENTS.md:468「設計判断の記録(ADR)を読む条件」 |
| 指示文書 | [docs/memo/](../../docs/memo/)（直下 134・done 240・frozen 6・dropped 2 ファイル） | RFD は指示、状態と順序は課題。過去の保管場所も検索する | 人間・Claude・Codex | .agents/skills/README.md:23、.agents/skills/anet-create-issue/SKILL.md:105、.agents/skills/anet-create-rfd/SKILL.md:12 ほか 429 ファイル | docs/memo/README.md:5「状態と作業順はissueで表す」 |
| 課題記録 | [docs/management/issues/](../../docs/management/issues/)（単票 289 ファイル・INDEX.md・README.md） | 課題の一覧・起票・判定、RFD の起票と実装の完了 | 人間・Claude・Codex | AGENTS.md:246、.agents/skills/README.md:17、.agents/skills/anet-audit/SKILL.md:8 ほか 102 ファイル | docs/management/issues/README.md:5 |

## 表 2 作業から読む文書

| これから何をするか | 読むもの（文書と章） | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| C++ を書く・直す | docs/quality/impl.md の「C++ コード」「名前」「コメントと TODO」「計測」「GetScalar」「所有権」「ログ」「失敗の扱い」 | C++ を変えないとき | AGENTS.md:49「作業ごとに読むもの」 |
| Python・Java・PowerShell を書く・直す | docs/quality/impl.md の「コメントと TODO」「ログ」 | 該当の言語を変えないとき | AGENTS.md:50 |
| 公開ヘッダ・ライブラリの境界・テストの置き場を変える | AGENTS.md「基本方針」の公開ヘッダと置き場の 6 行、docs/design/011_source_layout.jp.md、010 の「4.3 拡張点」 | 既存の境界の中で書くとき | AGENTS.md:345〜350、docs/design/README.jp.md:37、docs/design/010_framework_overview.jp.md:157 |
| 設定ファイルを書く・直す（番号付きの bat の `run.$` を含む） | docs/quality/impl.md の「設定ファイル」「名前」、020 の「3.6 どのファイルに何を書くか」「3.7 例」、`src/core/base/testdata/rfd0072/check_default_leaves.py` | 設定を変えないとき。設定の変更は実装の工程の中で同時に直す | AGENTS.md:51・61、docs/design/020_user_guide_run.jp.md:291・307 |
| 設計・直し方・決まりの案を出す | AGENTS.md「別案の書き方」「AI エージェントの概念説明ルール」（入口の中）、直す対象の行が指すレビュー観点 | 今の決まりや成果物から外れる案が無いとき。成果物の文書と、サブエージェントとして返す応答には書かない | AGENTS.md:107〜115・58 |
| RFD を起票する（グリル） | `anet-create-rfd`、docs/quality/rfd.md、impl.md の「名前」、wiki の目録と題材のトピック別まとめ、会話のまとめページ（`anet-summary-page`、docs/process/summary-page/grill.md）、事例を確かめる問いではサーベイの索引（reports/INDEX.md） | 実装だけの工程。RFD が要りそうなだけのときは始めず、課題と暫定 RFD の起票を提案して了承を待つ | AGENTS.md:52・501・103・180、.agents/skills/anet-create-rfd/SKILL.md:144、docs/memo/README.md:35、docs/process/summary-page/README.md「場面の一覧」 |
| RFD を再グリルする（精査） | `anet-create-rfd` に既存の RFD を渡す。docs/quality/rfd.md、RFD 冒頭の改訂履歴 | 起票と同じ枠では再グリルを置き換えない | docs/memo/README.md:36 |
| 調整する | 人間が裁定し、必要ならエージェント間を仲介する。RFD は self-contained に保つ | - | docs/memo/README.md:37・46 |
| 実装計画を作る | `anet-impl-rfd`、RFD、`anet-impl-rfd/RFD-IMPLEMENTATION.md`、impl.md の「実装計画」 | skill の実行そのものを計画に代える RFD | docs/memo/README.md:38、.agents/skills/anet-impl-rfd/SKILL.md:30 |
| 編集前の条件読み（契約・構造を変える） | AGENTS.md「# 設計・契約を変えるとき」の群、関連する設計文書、ADR（ファイル名で絞る）、wiki のトピック別まとめ | バグ修正・テスト追加・ログ追加・既存 RFD に沿う実装 | AGENTS.md:468〜487・502・152 |
| ユーザーの判断を RFD の工程の外で規約・コード・設定・プロセスに入れる | AGENTS.md「設計判断の記録(ADR)を起こす条件」、`anet-create-rfd/ADR-FORMAT.md` の「When to offer an ADR」 | RFD の工程の中（`anet-create-rfd`・`anet-impl-rfd` が扱う）、設計の判断が要らない「この場で直す」 | AGENTS.md:489〜497、.agents/skills/anet-create-rfd/ADR-FORMAT.md:29 |
| 新しい名前を付ける | CONTEXT.md（`_Avoid_` を含む）、impl.md の「名前」、AGENTS.md「AI エージェントの応答言語ルール」の避ける語の表 | 日常の実装・修正 | AGENTS.md:506〜516・69 |
| 検証する | 領域の設計文書のテスト章 → `src/core/base/testdata/rfdNNNN/README.md`、AGENTS.md「検証」の標準手順 | 対象外の検証は理由を明記する | AGENTS.md:552〜584 |
| ビルドが詰まる | docs/design/040_development_environment.jp.md の「6. CLI によるビルド・テスト」「8. トラブルシューティング」、AGENTS.md「AI エージェントでのビルド注意事項 (Windows/MSVC)」 | ビルドしない作業 | AGENTS.md:520・608、docs/design/040_development_environment.jp.md:364・498 |
| レビューする | `anet-review` → `anet-review-agent`、docs/process/protocol.md、docs/process/registry.md、rfd.md / impl.md | 対象が無いとき | docs/memo/README.md:40・48〜54 |
| 人間のレビュー・コミットの準備 | `anet-prepare-commit`、AGENTS.md「Git 操作・コミットメッセージルール」 | サブエージェントのとき、工程の途中 | docs/memo/README.md:41、AGENTS.md:230〜337 |
| 設計文書・ユーザーガイドを書く | impl.md の「検証と報告」 | - | AGENTS.md:53 |
| AGENTS.md・レビュー観点・プロセス契約・エージェント定義を直す（直し方の案を出す） | impl.md の「検証と報告」、直す観点の「位置づけ」、registry.md の「読み替え表」の前文 | - | AGENTS.md:54 |
| skill を書く・直す | impl.md の「skill」、`.agents/skills/README.md` | - | AGENTS.md:55 |
| 手戻りからハーネスを直す | `anet-improve` | 明示起動か、`anet-create-rfd`・`anet-impl-rfd` の完了報告の候補で選ばれたとき以外 | .agents/skills/README.md:30 |
| 会話のまとめページを作る・更新する | `anet-summary-page`、docs/process/summary-page/README.md と場面ごとの指針（grill.md・exp.md・general.md） | 場面の一覧の始める時機に当たらず、頼まれてもいないとき | AGENTS.md:103・104、docs/process/summary-page/README.md「場面の一覧」 |
| Runner を起動する（smoke・Run・bat） | 020 の「2.3 最初に確認する設定」「3.8 確認方法とよくあるエラー」「4.1 標準起動」 | この行による自己点検は無い | AGENTS.md:56・60、docs/design/020_user_guide_run.jp.md:45・372・398 |
| Run を分析する | 030（「6. `inspect_run.py`でRunを検査・抽出する」）、`inspect_run.py`、Atari-5 なら `atari5.py`、wiki の目録 | Run 名だけの提示も分析の依頼として扱う | AGENTS.md:727〜779、docs/design/030_user_guide_analysis.jp.md:369 |
| 実験記録を書く | AGENTS.md「実験記録と実効 config の保存ルール」、docs/experiments/README.md | tmp 付き・評価専用の Run の config は複製しない | AGENTS.md:695〜725・729 |
| 課題を扱う | `anet-create-issue`・`anet-list-issues`・`anet-update-issue`、docs/process/numbering.md | 工程の途中の未決はまず作業記録へ貯める | .agents/skills/anet-create-issue/SKILL.md:21 |
| 調査（サーベイ・論文解説）をする | `anet-survey-queue`、reports/INDEX.md。`survey` か `paper-details` の skill が要る | 済みのサーベイで答えが足りるとき | .agents/skills/anet-survey-queue/SKILL.md:10、.agents/skills/README.md:19 |
| 定常整備をする | `anet-housekeeping` → 統計・wiki 検査・監査・翻訳・調査・動線マップ・atlas の各 skill | - | .agents/skills/anet-housekeeping/SKILL.md:30〜38、.agents/skills/README.md:32 |

工程の図で畳んだ工程: 「計画・実装」は実装計画・編集前の条件読み・編集・設定の同時更新、「検証」は設計文書のテスト章と testdata の README、「人のレビュー」は人間のレビューとコミット。

## 表 3 AGENTS.md の節の配置

分類は、AGENTS.md の群の見出し（`#` の行）による。「毎回守ること」の群と冒頭を常時必要、ほかの 3 群を条件付きとした。移設先候補は、暫定の RFD0199 の既定を引く（決まった移設先ではない）。

| 節名 | 行数 | 分類 | 移設先候補 | 根拠 |
|---|---|---|---|---|
| Viewing UTF-8 Japanese Text in AI Agent Terminals on Windows | 23 | 常時必要 | 入口に残す（文字化けしたまま入口を読まないため） | AGENTS.md:3・冒頭（群の見出しの前） |
| プロジェクト概要 | 17 | 常時必要 | 入口に残す | AGENTS.md:28・「# 毎回守ること」 |
| 作業ごとに読むもの | 18 | 常時必要 | 入口に残す | AGENTS.md:45・「# 毎回守ること」 |
| AI エージェントの応答言語ルール | 30 | 常時必要 | 入口に残す（避ける語の表は RFD と実装の観点の検索の正本を兼ねる） | AGENTS.md:63・「# 毎回守ること」 |
| AI エージェントの概念説明ルール | 14 | 常時必要 | 入口に残す（会話のまとめページの始める時機の 2 行を含む。どの会話にも当てる） | AGENTS.md:93・「# 毎回守ること」 |
| 別案の書き方 | 10 | 常時必要 | 入口に残す（案を出すどの応答にも当てる） | AGENTS.md:107・「# 毎回守ること」 |
| 人間へ提示するコマンドの書き方 | 30 | 常時必要 | 入口に残す（群の見出しによる。要るのはコマンドを示す応答だけ） | AGENTS.md:117・「# 毎回守ること」 |
| AI エージェントの作業ルール | 39 | 常時必要 | 入口に残す（「RFD が要りそうなとき」の 5 行を含む） | AGENTS.md:147・「# 毎回守ること」 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 常時必要 | 入口に残す（群の見出しによる。全文は karpathy-guidelines の skill にある） | AGENTS.md:186・「# 毎回守ること」 |
| 編集しない・慎重に扱う領域 | 12 | 常時必要 | 入口に残す | AGENTS.md:197・「# 毎回守ること」 |
| 改行と文字コード | 9 | 常時必要 | 入口に残す（群の見出しによる。要るのはファイルを書くときだけ） | AGENTS.md:209・「# 毎回守ること」 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 常時必要 | 入口に残す（群の見出しによる。要るのは長時間の処理を流すときだけ） | AGENTS.md:218・「# 毎回守ること」 |
| Git 操作・コミットメッセージルール | 109 | 常時必要 | 入口に残す。うちコミットメッセージの書き方と Topic の表の 69 行（AGENTS.md:236〜304）は、RFD0199 の U4 が docs/rules/commit-message.md へ移す候補にする（暫定） | AGENTS.md:230・「# 毎回守ること」 |
| 基本方針 | 18 | 条件付き | RFD0199 の暫定の既定は移さない（U3）。公開ヘッダ・テストの置き場の行は docs/design/011_source_layout.jp.md と重なる | AGENTS.md:341・「# 設計・契約を変えるとき」 |
| クリーンブレーク方針 | 23 | 条件付き | RFD0199 の暫定の既定は移さない（U3。ADR 0061 が原則の移設を退けた） | AGENTS.md:359・「# 設計・契約を変えるとき」 |
| 汎用機構と利用側の責任境界 | 42 | 条件付き | RFD0199 の暫定の既定は移さない（U3） | AGENTS.md:382・「# 設計・契約を変えるとき」 |
| Fail-Fast 原則 | 30 | 条件付き | RFD0199 の暫定の既定は移さない（U3。ADR 0061 が原則の移設を退けた）。移すなら docs/quality/impl.md の「失敗の扱い」 | AGENTS.md:424・「# 設計・契約を変えるとき」 |
| 設定値の扱い | 14 | 条件付き | RFD0199 の暫定の既定は移さない（U3。ADR 0061 が原則の移設を退けた）。移すなら docs/quality/impl.md の「設定ファイル」 | AGENTS.md:454・「# 設計・契約を変えるとき」 |
| 設計判断の記録(ADR)を読む条件 | 21 | 条件付き | RFD0199 の暫定の既定は移さない（U3） | AGENTS.md:468・「# 設計・契約を変えるとき」 |
| 設計判断の記録(ADR)を起こす条件 | 10 | 条件付き | RFD0199 の暫定の既定は移さない（U3）。移すなら .agents/skills/anet-create-rfd/ADR-FORMAT.md | AGENTS.md:489・「# 設計・契約を変えるとき」 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | RFD0199 の暫定の既定は移さない（U3）。移すなら docs/wiki/README.md（読む条件は入口に残す） | AGENTS.md:499・「# 設計・契約を変えるとき」 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | RFD0199 の暫定の既定は移さない（U3）。移すなら docs/quality/impl.md の「名前」 | AGENTS.md:506・「# 設計・契約を変えるとき」 |
| ビルド | 32 | 条件付き | docs/rules/build.md（RFD0199 の U1。暫定） | AGENTS.md:520・「# ビルドと検証」 |
| 検証 | 34 | 条件付き | docs/rules/build.md（RFD0199 の U1。暫定。docs/quality/impl.md と比べる案がある） | AGENTS.md:552・「# ビルドと検証」 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/rules/build.md（RFD0199 の U1。暫定） | AGENTS.md:586・「# ビルドと検証」 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 49 | 条件付き | docs/rules/build.md（RFD0199 の U1。暫定） | AGENTS.md:608・「# ビルドと検証」 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/rules/run.md（RFD0199 の U2。暫定） | AGENTS.md:659・「# Run と実験」 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/rules/run.md（RFD0199 の U2。暫定） | AGENTS.md:695・「# Run と実験」 |
| AI エージェントのRun結果分析ルール | 53 | 条件付き | docs/rules/run.md（RFD0199 の U2。暫定） | AGENTS.md:727・「# Run と実験」 |
| 集計 | 全 29 節・769 行 | 条件付き 16 節・435 行（到達しない節 0） | 移設はこの作業の対象外 | AGENTS.md の全 `##` 見出し。群の見出し 4 つ（計 8 行）は数えない |

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
| anet-create-rfd | anet-create-rfd/ADR-FORMAT.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-rfd/ADR-FORMAT.md・.claude/skills/anet-create-rfd/ADR-FORMAT.md |
| anet-create-rfd | anet-create-rfd/CONTEXT-FORMAT.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-rfd/CONTEXT-FORMAT.md・.claude/skills/anet-create-rfd/CONTEXT-FORMAT.md |
| anet-create-rfd | anet-create-rfd/OTHER-MODEL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-rfd/OTHER-MODEL.md・.claude/skills/anet-create-rfd/OTHER-MODEL.md |
| anet-create-rfd | anet-create-rfd/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-rfd/SKILL.md・.claude/skills/anet-create-rfd/SKILL.md |
| anet-create-rfd | anet-create-rfd/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-create-rfd/agents/openai.yaml・.claude/skills/anet-create-rfd/agents/openai.yaml |
| anet-harness-map | anet-harness-map/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/SKILL.md・.claude/skills/anet-harness-map/SKILL.md |
| anet-harness-map | anet-harness-map/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/agents/openai.yaml・.claude/skills/anet-harness-map/agents/openai.yaml |
| anet-harness-map | anet-harness-map/references/harness-map-contract.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-harness-map/references/harness-map-contract.md・.claude/skills/anet-harness-map/references/harness-map-contract.md |
| anet-housekeeping | anet-housekeeping/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-housekeeping/SKILL.md・.claude/skills/anet-housekeeping/SKILL.md |
| anet-housekeeping | anet-housekeeping/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-housekeeping/agents/openai.yaml・.claude/skills/anet-housekeeping/agents/openai.yaml |
| anet-impl-rfd | anet-impl-rfd/RFD-IMPLEMENTATION.md | project | 一致 | 個人版に記録 | なし | .agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md・.claude/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md |
| anet-impl-rfd | anet-impl-rfd/SKILL.md | project | 一致 | 個人版に記録 | なし | .agents/skills/anet-impl-rfd/SKILL.md・.claude/skills/anet-impl-rfd/SKILL.md |
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
| anet-summary-page | anet-summary-page/SKILL.md | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-summary-page/SKILL.md・.claude/skills/anet-summary-page/SKILL.md |
| anet-summary-page | anet-summary-page/agents/openai.yaml | project | 一致 | 個人版に記録 | あり | .agents/skills/anet-summary-page/agents/openai.yaml・.claude/skills/anet-summary-page/agents/openai.yaml |
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

user-level にしかない skill: `~/.claude/skills` に 37 件、`~/.agents/skills` に 42 件、`~/.codex/skills` に 7 件（同梱の `.system` の 6 件を含む）。`~/.claude/skills` と `~/.agents/skills` の集合の差は、前者にだけあるものが 1 件、後者にだけあるものが 6 件。名前は個人版に置く。

## 発見事項

修正はしない。名指しだけを行う。

### shadow

- Claude Code 側 4 本（中身違い 4 本）、Codex 側 4 本（中身違い 4 本）。Codex 同梱の `~/.codex/skills` との同名は 0 本。どの skill かと差分は個人版にだけ書く。

### 実体コピーのドリフト

- 0 本。`diff -rq .agents/skills .claude/skills` の出力は空（56 ファイルすべて一致）。「品質確認観点」から「レビュー観点」への改名（0411d5eb。skill は 5 本・7 ファイル）と、`anet-create-rfd` のサーベイの索引を引く手順（作業ツリー）も、両側で一致している。

### 欠落

- テスト章の無い設計文書: `135_forwarder_and_perception.jp.md`（前回のあとに新設）、`212_rnd_lifelong_novelty.jp.md`、`220_atari_env.jp.md`。135 の最終 2 章は「4. 標準の部品と設定」「5. 図と拡張」で、§5 の末尾にテストのタグ（`[forwarder]`・`[perception]`）だけがあり、testdata README への参照が無い。この領域では主経路（設計文書 → テスト章 → testdata README）が設計文書で止まる。212 の最終 2 章は「8. Network の構造図」「9. 背景、実装と検証」で、testdata README への参照は §9.6「検証の記録」にある。220 の最終 2 章は「6. ビルド統合」「7. 関連文書」で、testdata README への参照は §6・§7 の本文にある。
- 「関連文書」章の無い文書: 1xx・2xx では `135_forwarder_and_perception.jp.md` と `212_rnd_lifelong_novelty.jp.md`（§9.9「出典と関連文書」を節として持つ）。0xx では `011_source_layout.jp.md` と `040_development_environment.jp.md`。
- 章題・章番号のゆれ: `160_applications_and_tools.jp.md` だけ「テストと変更時の確認事項」（ほかは「テストと拡張時の確認事項」）。テスト章の章番号は 7〜12 に分かれる（214 が 7、140 が 9、200 が 10、210 が 12、ほかは 8）。`011_source_layout.jp.md` は章番号が無く（「ライブラリとアプリ」「テスト」「出力の置き場」）、索引の 1 行だけから到達する。同じ規則を AGENTS.md「基本方針」の 6 行（AGENTS.md:345〜350）が持つが、011 へのリンクは無い。
- 索引に無い設計文書: `docs/design/optuna.md`（`NNN_*.jp.md` の命名でもない）。030・140・160 の本文と関連文書からは引かれている。
- Codex 表示名（`agents/openai.yaml`）の無い project skill: anet-impl-rfd, grill-me, handoff, improve-codebase-architecture, karpathy-guidelines, tdd。汎用 skill にも必須かは決まっていない。
- 資料と実測の不一致: skills README の「追加と更新の手順」1（`.agents/skills/README.md:50`）は `agents/openai.yaml` を添えると書くが、プロジェクト固有の `anet-impl-rfd` だけが持たない（改名前の `anet-impl-prd` から無い）。
- 資料と実測の不一致: CLAUDE.md は AGENTS.md を読んで従うよう書くが、Claude Code の Read は 1 回 25,000 トークンまでで、AGENTS.md（77,414 バイト・779 行。この作業の Read は 28,104 トークンと数えた）は 1 回では 589 行目で切れた。2 回目の Read をしないと、4 節（「AI エージェントでのビルド注意事項 (Windows/MSVC)」、「Run 命名・後片付けルール」、「実験記録と実効 config の保存ルール」、「AI エージェントのRun結果分析ルール」）が読まれない。CLAUDE.md と AGENTS.md に、読み切る手順は無い。
- 資料と実測の不一致: AGENTS.md の群の見出し（「毎回守ること」ほか 3 つ。ae2ac9c1 で置いた）と、前回の表 3 の分類が食い違う。前回は節ごとの見立てで、「毎回守ること」の群にある 5 節（コマンドの書き方・LLM コーディング規律・改行と文字コード・長時間の検証・Git 操作）を条件付き、「設計・契約を変えるとき」の群の「基本方針」を常時必要としていた。この版は AGENTS.md の群に合わせた（条件付き 16 節・435 行）。前回の分け方のまま数えると 20 節・588 行で、前回と同じ。
- 資料と実測の不一致: RFD0199（AGENTS.md を状況別に分ける暫定の RFD）の §5.1 は 28 節・770 行・75,769 バイトと数えるが、その後に「別案の書き方」の節（10 行。5cd3ec3b）が入り、今は 29 節・779 行・77,414 バイト。RFD が挙げる群の行番号は、「設計・契約を変えるとき」の群から後ろが 10 行ずれている。
- 資料と実測の不一致: 検証資材の README 29 本（計 98 行。`core\anet-core` の形も含む）の実行手順に、RFD0182 で移す前のパス（`core/anet-core`）と実行体名（`anet-core-test`）が残る（前回は 30 本・96 行。`rfd0072` の README は RFD0200 の実装で直った）。29 本とも「読み替える」の注記がある。ただし AGENTS.md「検証」は標準手順だけを持ち、旧い名前から新しい名前への対応表は持たない（対応は RFD0182 の「5. 実装契約」にある）。動線の末端で、読み替えの材料が入口から引けない。
- 資料と実測の不一致: memo README の「リンクの注意」は、移動・改名した RFD を指すリンクの直し方と、`docs/` 全体のリンク切れ検査を定めるが、同じ検査（コードの中を除き、`%20` などを戻して判定）を回すとリンク切れは 209 本ある（docs/memo の直下 122・過去の保管場所 79・課題 7・ほか 1）。うち 177 本は、RFD0182 などで移動・改名・分割したソースを指す。ソースの移動で切れたリンクの扱いは定めが無く、新しく切れたリンクが埋もれる。
- 資料と実測の不一致: 186eb184 で削除した `docs/agents/issue-tracker.md` を指す Markdown のリンクが、直下の RFD `docs/memo/0114_findings_intake_and_fix_rfd_10rfd.md`（37・417 行）と `docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md`（76・98・180・343 行）に残り、リンク切れになる（計 6 本）。「リンクの注意」は、削除した文書を指すリンクの扱いも定めていない。
- 資料と実測の不一致: 旧い語「品質確認観点」が、改名（0411d5eb）のあとも `docs/memo/0198_reread_after_compaction_10rfd.md` に 5 か所残る（改名の前に起草した RFD）。ADR 0061 の本文 1 行目にもあるが、ADR は書き換えない決まりなので当時の記録。ほかの契約・skill・設計文書には残っていない。
- 資料と実測の不一致: この skill の契約（references/harness-map-contract.md の「architecture 図」）は、ユーザー環境のカードに「Claude Code だけが global `CLAUDE.md` と memory 層を持つ」と書くよう定めるが、Codex も memory の機能を持ち、設定で有効にできる。カードは「global CLAUDE.md は Claude Code だけが読む。memory はエージェントごと」と書いた。
- 資料と実測の不一致: 契約の表 1 の対象の一覧に、レビュー観点（docs/quality）・プロセス契約（docs/process）・wiki・エージェント定義・hook・サーベイの索引（reports/INDEX.md）・root の README.md が無い。どれもエージェントか人間が条件つきで読むので、この版では行に足した。
- 資料と実測の不一致: 契約は工程図のレーンを「人間 / 起票側 / 精査側 / 実装側 / 共通文書 / CI」の 6 本と定めるが、図は CI を外した 5 本にした。2026-11-25 に CI のレーンを足した候補を 2 通り試し、1110×838 は検証を通るが 1440×900 で縦 1294px になって画面に収まらず、1310×845 は文字が 5.7px になって検証が落ちた（必要は 6px 以上）。CI の読む条件と参照元は表 1、push 後の CI は俯瞰図の CI ノードが持つ。
- 資料と実測の不一致: 工程図の「ユーザー環境」は主経路に置いたが、列が足りず「設計」の列に入る。Archify の workflow は列の位置が固定（88・220・300・430・500・625）で、1・2 列目と 3・4 列目は同じレーンに並べられない。

### 死んだ動線

- 0 件（表 1 で参照元が空の行は無い）。
- 参照元が RFD・課題・wiki・ADR・実験記録・レポートと root の README.md だけで、エージェントが読む契約（AGENTS.md・skill・設計文書・レビュー観点・プロセス契約）から名指しされないもの: docs/experiments/README.md、.github/workflows/gemini_ai_review.yml、.github/workflows/release.yml。`docs/experiments/README.md` は、e5033822 で root の README.md から引けるようになった（人間の入口）。AGENTS.md「実験記録と実効 config の保存ルール」は置き場を定めるのに、README 自体へのリンクが無い。

### 機械化点

- 実体コピーの同期検査: `diff -rq .agents/skills .claude/skills` を hook か CI で回す（今は手作業。skills README:3）。
- CI のパス条件化: ハーネスの文書だけの変更で `windows-ci.yml` のビルドを回さない条件。
- 改名・配置替えの後の旧い名前の検索: RFD0182・RFD0195 の受入で使った `git grep` を、ハーネスの文書（検証資材の README を含む）と、追跡していないファイル（`--untracked`）にも定常整備で回す。
- リンク切れの検査: memo README の「リンクの注意」のループを `docs/` 全体へ定常整備で回し、前回からの増減だけを見る。削除した文書と、移動したソースを指すリンクも拾う。
- AGENTS.md の大きさの検査: バイト数と、Read の 1 回の上限で切れる行を定常整備で測る。
- 参照元の実測: 表 1 の参照元の検索を定常整備で回し、死んだ動線と、契約から名指しされない末端の増減を見る。
- 設計文書のテスト章と関連文書章の有無、章題を検査する（新しい文書を足した変更で見る）。
- skill ごとの `agents/openai.yaml` の有無を skills README の手順と突き合わせる。

### evidence gap

- ユーザー環境（global の指示・user-level skill・memory）の実体と読めたかどうかは個人版にだけ書く。
- shadow が実際にどちらを起動するか（解決順）は、起動して確かめていない。この作業の会話（Claude Code）のスキル一覧に載った説明文で見分けられた 2 本は、1 本が user-level 版、1 本が project 版だった。残りは説明文が同じで見分けられない。Codex 側は見ていない。
- `.claude/settings.local.json` はローカルの設定なので読んでいない。
- 参照元は文字列の一致で数えた。リンクの文脈（読む指示か、記録か）は表 1 の「読む条件」で手で補った。生成物（`docs/agents/README.md`、`docs/agents/harness_map_*`、`docs/archify/`、`*.html`）と、全ノードを列挙する `anet-harness-map` skill 自身は参照元から除いた。root の `README.md` の参照元だけは、Markdown のリンクの行き先を解いて数えた。
- AGENTS.md のトークン数と切れる行は、Claude Code の Read が返した値で、Codex の読み方は確かめていない。
- 工程の正本（docs/memo/README.md「工程と担当」）と、RFD の改訂履歴の行は突き合わせていない。改訂の行はどの枠で再グリルしたかを持たないので、「少なくとも起票と精査は分ける」が守られたかを実測できない。
- 調査の間、別の定常整備の作業が `docs/wiki/`・`docs/management/issues/`・`INDEX.md`・`docs/design/*.en.md`・`docs/archify/` を書き換えていた。これらを参照元に含む行の件数と行番号は、測った時点のもの。
- C++ の build と test は、この作業に含めず実行していない。
