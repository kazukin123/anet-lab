# AI ハーネスの動線マップ

生成根拠: HEAD `0f4fd0f9718bddfd0e8f0ea5a9248920b0da209b` と、その上の現在の作業ツリー（未コミットの変更を含む）。参照元は作業ツリーを `rg --hidden` で実測した。生成日 2026-11-23。
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

参照元は、パスかファイル名の文字列を `rg --hidden -F` で引いた結果の先頭 3 ファイル（行は各ファイルで最初に当たった行）と、残りのファイル数。並びは入口・skill・設計文書と規約・hook と CI・検証資材・英訳・RFD と課題・wiki・その他の順。

| 種別 | パス | 読む条件 | 誰が読む | 参照元 | 根拠 |
|---|---|---|---|---|---|
| 入口 | [CLAUDE.md](../../CLAUDE.md) | 常時 | Claude | AGENTS.md:23、.agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37 ほか 7 ファイル | CLAUDE.md:4 |
| 入口 | [AGENTS.md](../../AGENTS.md) | 常時 | 人間・Claude・Codex | CLAUDE.md:4、.agents/skills/anet-audit/SKILL.md:24、.agents/skills/anet-create-issue/SKILL.md:156 ほか 240 ファイル | AGENTS.md:22 |
| 規約・索引 | [docs/agents/README.md](README.md) | 工程ごとの読む条件を引くとき | 人間・Claude・Codex | .agents/skills/anet-housekeeping/SKILL.md:37、.claude/skills/anet-housekeeping/SKILL.md:37、docs/memo/README.md:47 ほか 14 ファイル | docs/memo/README.md:47 |
| 設計文書 | [docs/design/010_framework_overview.jp.md](../../docs/design/010_framework_overview.jp.md) | 拡張点を基盤の外で書けるようにするとき・全体像を引くとき | 人間・Claude・Codex | AGENTS.md:50、.agents/skills/README.md:18、.agents/skills/anet-translate-docs/SKILL.md:12 ほか 32 ファイル | AGENTS.md:50「基本方針」 |
| 設計文書 | [docs/design/011_source_layout.jp.md](../../docs/design/011_source_layout.jp.md) | ソース配置・公開ヘッダ・テストの置き場を変えるとき | 人間・Claude・Codex | docs/design/README.jp.md:37、docs/design/011_source_layout.en.md:1、docs/wiki/apps/110_runner_app.md:15 ほか 6 ファイル | docs/design/README.jp.md:37「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/020_user_guide_run.jp.md](../../docs/design/020_user_guide_run.jp.md) | Runner を起動するとき・設定ファイルを書くとき | 人間・Claude・Codex | AGENTS.md:72、docs/design/010_framework_overview.jp.md:223、docs/design/030_user_guide_analysis.jp.md:331 ほか 61 ファイル | AGENTS.md:72「作業ごとに読むもの」 |
| 設計文書 | [docs/design/030_user_guide_analysis.jp.md](../../docs/design/030_user_guide_analysis.jp.md) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:664、docs/design/010_framework_overview.jp.md:381、docs/design/020_user_guide_run.jp.md:375 ほか 42 ファイル | AGENTS.md:664「AI エージェントのRun結果分析ルール」 |
| 設計文書 | [docs/design/040_development_environment.jp.md](../../docs/design/040_development_environment.jp.md) | ビルド・テストの環境を扱うとき・ビルドが詰まるとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:40、docs/design/020_user_guide_run.jp.md:31、docs/design/030_user_guide_analysis.jp.md:24 ほか 15 ファイル | docs/design/README.jp.md:52「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/100_runtime_and_configuration.jp.md](../../docs/design/100_runtime_and_configuration.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:223、docs/design/020_user_guide_run.jp.md:111、docs/design/110_agents_and_learning.jp.md:156 ほか 46 ファイル | docs/design/README.jp.md:60「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/110_agents_and_learning.jp.md](../../docs/design/110_agents_and_learning.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:229、docs/design/020_user_guide_run.jp.md:654、docs/design/100_runtime_and_configuration.jp.md:407 ほか 34 ファイル | docs/design/README.jp.md:61「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/120_environments.jp.md](../../docs/design/120_environments.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:229、docs/design/020_user_guide_run.jp.md:655、docs/design/100_runtime_and_configuration.jp.md:408 ほか 30 ファイル | docs/design/README.jp.md:62「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/130_neural_networks.jp.md](../../docs/design/130_neural_networks.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:253、docs/design/110_agents_and_learning.jp.md:21、docs/design/150_replay_buffer.jp.md:394 ほか 46 ファイル | docs/design/README.jp.md:63「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/140_observability.jp.md](../../docs/design/140_observability.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:89、docs/design/020_user_guide_run.jp.md:656、docs/design/030_user_guide_analysis.jp.md:552 ほか 69 ファイル | docs/design/README.jp.md:64「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/150_replay_buffer.jp.md](../../docs/design/150_replay_buffer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:65、.claude/skills/anet-audit/SKILL.md:65、docs/design/010_framework_overview.jp.md:247 ほか 44 ファイル | docs/design/README.jp.md:65「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/160_applications_and_tools.jp.md](../../docs/design/160_applications_and_tools.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:357、docs/design/020_user_guide_run.jp.md:431、docs/design/030_user_guide_analysis.jp.md:553 ほか 47 ファイル | docs/design/README.jp.md:66「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/200_dqn_agents.jp.md](../../docs/design/200_dqn_agents.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:241、docs/design/020_user_guide_run.jp.md:534、docs/design/030_user_guide_analysis.jp.md:242 ほか 84 ファイル | docs/design/README.jp.md:74「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/210_metrics_viewer.jp.md](../../docs/design/210_metrics_viewer.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:20、docs/design/README.jp.md:75、docs/design/210_metrics_viewer.en.md:1 ほか 22 ファイル | docs/design/README.jp.md:75「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/212_rnd_lifelong_novelty.jp.md](../../docs/design/212_rnd_lifelong_novelty.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/140_observability.jp.md:188、docs/design/150_replay_buffer.jp.md:114、docs/design/200_dqn_agents.jp.md:48 ほか 45 ファイル | docs/design/README.jp.md:76「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/214_value_decomposition.jp.md](../../docs/design/214_value_decomposition.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:104、docs/design/140_observability.jp.md:297、docs/design/200_dqn_agents.jp.md:804 ほか 17 ファイル | docs/design/README.jp.md:77「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/220_atari_env.jp.md](../../docs/design/220_atari_env.jp.md) | 領域に触れる作業の前 | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:190、docs/design/README.jp.md:78、apps/runner/config/Atari.txt:1358 ほか 32 ファイル | docs/design/README.jp.md:78「ドキュメント一覧」・AGENTS.md:437 |
| 設計文書 | [docs/design/README.jp.md](../../docs/design/README.jp.md) | 設計文書の索引を引くとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:411、README.md:37、docs/design/README.en.md:1 ほか 7 ファイル | docs/design/README.jp.md「3. ドキュメント一覧」 |
| 用語集 | [CONTEXT.md](../../CONTEXT.md) | 新しい名前を付けるとき・別の語で呼びたくなったとき | 人間・Claude・Codex | AGENTS.md:140、.agents/skills/README.md:23、.agents/skills/anet-audit/SKILL.md:20 ほか 272 ファイル | AGENTS.md:140 |
| 規約・索引 | [docs/ownership_guideline.md](../../docs/ownership_guideline.md) | Agent 実装の所有権を変えるとき | 人間・Claude・Codex | docs/design/010_framework_overview.jp.md:416、docs/design/110_agents_and_learning.jp.md:55、docs/design/200_dqn_agents.jp.md:856 ほか 20 ファイル | docs/design/README.jp.md:84「関連文書」 |
| 規約・索引 | [docs/memo/README.md](../../docs/memo/README.md) | RFD の採番・工程と担当・リンク規約を引くとき | 人間・Claude・Codex | .agents/skills/anet-create-rfd/SKILL.md:212、.claude/skills/anet-create-rfd/SKILL.md:212、docs/process/registry.md:62 ほか 23 ファイル | .agents/skills/anet-create-rfd/SKILL.md:212 |
| 規約・索引 | [docs/experiments/README.md](../../docs/experiments/README.md) | 実験記録を書く・読むとき | 人間・Claude・Codex | docs/memo/done/0095_llm_wiki_10rfd.md:15、docs/memo/done/0095_llm_wiki_22impl.md:13、docs/wiki/log.md:202 ほか 1 ファイル | AGENTS.md:628（節の対象） |
| レビュー観点 | [docs/quality/impl.md](../../docs/quality/impl.md) | AGENTS.md「作業ごとに読むもの」の行に当たるとき | 人間・Claude・Codex | AGENTS.md:65、.agents/skills/anet-create-rfd/SKILL.md:146、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:78 ほか 99 ファイル | AGENTS.md:65 |
| レビュー観点 | [docs/quality/rfd.md](../../docs/quality/rfd.md) | RFD を書く・直すとき | 人間・Claude・Codex | AGENTS.md:68、.agents/skills/anet-create-rfd/SKILL.md:8、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:192 ほか 21 ファイル | AGENTS.md:68 |
| レビュー観点 | [docs/quality/refactor.md](../../docs/quality/refactor.md) | リファクタリングの監査のとき | 人間・Claude・Codex | .agents/skills/anet-audit/SKILL.md:18、.claude/skills/anet-audit/SKILL.md:18、docs/process/registry.md:23 ほか 64 ファイル | docs/process/registry.md:23 |
| プロセス契約 | [docs/process/numbering.md](../../docs/process/numbering.md) | 課題などの ID を採番するとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:21、.claude/skills/anet-create-issue/SKILL.md:21、docs/management/issues/README.md:11 ほか 3 ファイル | .agents/skills/anet-create-issue/SKILL.md:21 |
| プロセス契約 | [docs/process/protocol.md](../../docs/process/protocol.md) | レビューを依頼する・課題の候補を挙げるとき | 人間・Claude・Codex | .agents/skills/anet-create-issue/SKILL.md:237、.agents/skills/anet-create-rfd/SKILL.md:8、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:198 ほか 30 ファイル | .agents/skills/anet-review/SKILL.md:9 |
| プロセス契約 | [docs/process/registry.md](../../docs/process/registry.md) | 成果物ごとの観点を引くとき・プロセス契約とレビューエージェントの定義を直すとき | 人間・Claude・Codex | AGENTS.md:70、.agents/skills/anet-create-issue/SKILL.md:15、.agents/skills/anet-create-rfd/SKILL.md:8 ほか 33 ファイル | AGENTS.md:70 |
| プロセス契約 | [docs/process/summary-page/README.md](../../docs/process/summary-page/README.md) | 会話のまとめページを作る・更新するとき | 人間・Claude・Codex | AGENTS.md:417、.agents/skills/anet-summary-page/SKILL.md:8、.claude/skills/anet-summary-page/SKILL.md:8 ほか 2 ファイル | AGENTS.md:417 |
| wiki | [docs/wiki/README.md](../../docs/wiki/README.md) | wiki を書く・直すとき | 人間・Claude・Codex | AGENTS.md:138、.agents/skills/anet-create-rfd/SKILL.md:224、.agents/skills/anet-ingest-wiki/SKILL.md:8 ほか 22 ファイル | AGENTS.md:138 |
| wiki | [docs/wiki/index.md](../../docs/wiki/index.md) | RFD を起こす前・Run 分析の結論の前・構造を変える前 | 人間・Claude・Codex | AGENTS.md:135、.agents/skills/anet-create-rfd/SKILL.md:18、.claude/skills/anet-create-rfd/SKILL.md:18 ほか 10 ファイル | AGENTS.md:135 |
| エージェント定義 | [.claude/agents/anet-other-model.md](../../.claude/agents/anet-other-model.md) | 別モデルの検討を頼むとき | Claude | .agents/skills/anet-create-rfd/OTHER-MODEL.md:3、.agents/skills/anet-create-rfd/SKILL.md:263、.claude/skills/anet-create-rfd/OTHER-MODEL.md:3 ほか 4 ファイル | .agents/skills/anet-create-rfd/SKILL.md:263 |
| エージェント定義 | [.claude/agents/anet-review-agent.md](../../.claude/agents/anet-review-agent.md) | レビューを頼むとき | Claude | .agents/skills/README.md:29、.agents/skills/anet-create-rfd/SKILL.md:296、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:218 ほか 28 ファイル | docs/memo/README.md:49 |
| エージェント定義 | [.codex/agents/anet-review-agent.toml](../../.codex/agents/anet-review-agent.toml) | レビューを頼むとき | Codex | .agents/skills/README.md:29、.agents/skills/anet-create-rfd/SKILL.md:296、.agents/skills/anet-impl-rfd/RFD-IMPLEMENTATION.md:218 ほか 28 ファイル | docs/memo/README.md:49 |
| hook | [.agents/hooks/commit_cue.py](../../.agents/hooks/commit_cue.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:564、docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue_test.py:12 ほか 7 ファイル | AGENTS.md:564・.claude/settings.json:8 |
| hook | [.agents/hooks/search_guard.py](../../.agents/hooks/search_guard.py) | エージェントの実行時に自動で効く（読む文書ではない） | Claude・Codex（hook から） | AGENTS.md:444、docs/design/160_applications_and_tools.jp.md:416、.agents/hooks/search_guard_test.py:11 ほか 7 ファイル | AGENTS.md:444・.claude/settings.json:20 |
| hook | [.codex/hooks.json](../../.codex/hooks.json) | エージェントの実行時に自動で効く（読む文書ではない） | Codex | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:6、.agents/hooks/search_guard.py:4 ほか 4 ファイル | .codex/hooks.json:2 |
| hook | [.claude/settings.json](../../.claude/settings.json) | エージェントの実行時に自動で効く（読む文書ではない） | Claude | docs/design/160_applications_and_tools.jp.md:415、.agents/hooks/commit_cue.py:5、.agents/hooks/search_guard.py:4 ほか 4 ファイル | .claude/settings.json:2 |
| 検証資材 | [src/core/base/testdata/rfd0061/README.md](../../src/core/base/testdata/rfd0061/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:224、docs/design/110_agents_and_learning.en.md:225、docs/memo/0195_document_term_rfd_and_4digit_numbers_20impl.md:237 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0072/README.md](../../src/core/base/testdata/rfd0072/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/020_user_guide_run.jp.md:307、docs/design/100_runtime_and_configuration.jp.md:399、docs/quality/impl.md:173 ほか 24 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0078/README.md](../../src/core/base/testdata/rfd0078/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:385、docs/design/150_replay_buffer.en.md:386 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0082/README.md](../../src/core/base/testdata/rfd0082/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:342、docs/design/220_atari_env.en.md:343、docs/memo/0126_atari_ram_metric_distinct_and_bit_reducers_10rfd.md:219 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0084/README.md](../../src/core/base/testdata/rfd0084/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:388、docs/design/200_dqn_agents.jp.md:616、src/core/base/testdata/rfd0084/run_forward_refactor.ps1:135 ほか 22 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0089/README.md](../../src/core/base/testdata/rfd0089/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1141、docs/design/210_metrics_viewer.en.md:1143、docs/memo/0196_metrics_viewer_run_comparison_10rfd.md:385 ほか 1 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0090/README.md](../../src/core/base/testdata/rfd0090/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1134、docs/design/210_metrics_viewer.en.md:1136、docs/memo/0195_document_term_rfd_and_4digit_numbers_20impl.md:299 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0092/README.md](../../src/core/base/testdata/rfd0092/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:841、docs/design/200_dqn_agents.en.md:846、docs/memo/0116_rnd_lifelong_novelty_10rfd.md:60 ほか 8 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0093/README.md](../../src/core/base/testdata/rfd0093/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:800、docs/design/200_dqn_agents.en.md:804、docs/memo/0158_eval_schedule_first_fire_10rfd.md:230 ほか 1 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0094/README.md](../../src/core/base/testdata/rfd0094/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:184、docs/design/140_observability.jp.md:359、docs/design/160_applications_and_tools.jp.md:139 ほか 8 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0095/README.md](../../src/core/base/testdata/rfd0095/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:432、src/core/base/testdata/rfd0114/README.md:9、docs/design/160_applications_and_tools.en.md:433 ほか 6 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0097/README.md](../../src/core/base/testdata/rfd0097/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:278、docs/design/220_atari_env.jp.md:352、docs/design/120_environments.en.md:279 ほか 5 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0098/README.md](../../src/core/base/testdata/rfd0098/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:439、docs/design/150_replay_buffer.jp.md:368、docs/design/200_dqn_agents.jp.md:826 ほか 24 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0099/README.md](../../src/core/base/testdata/rfd0099/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/220_atari_env.jp.md:340、src/env/atari/src/AtariEnv_test.cpp:1775、docs/design/220_atari_env.en.md:341 ほか 4 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0100/README.md](../../src/core/base/testdata/rfd0100/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:280、src/core/base/testdata/rfd0102/run_validation.py:62、docs/design/120_environments.en.md:291 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0101/README.md](../../src/core/base/testdata/rfd0101/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:366、docs/design/200_dqn_agents.jp.md:824、src/core/base/testdata/rfd0101/check_contract.py:15 ほか 3 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0102/README.md](../../src/core/base/testdata/rfd0102/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:292、src/env/gridmaze/src/GridMazeEnv_test.cpp:577、docs/design/120_environments.en.md:293 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0103/README.md](../../src/core/base/testdata/rfd0103/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:437、docs/design/200_dqn_agents.jp.md:560、src/algo/dqn/src/dqn_concrete_test.cpp:1108 ほか 16 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0104/README.md](../../src/core/base/testdata/rfd0104/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/110_agents_and_learning.jp.md:162、docs/design/140_observability.jp.md:429、docs/design/150_replay_buffer.jp.md:221 ほか 9 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0105/README.md](../../src/core/base/testdata/rfd0105/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:139、docs/design/160_applications_and_tools.en.md:141、docs/memo/0122_small_refactor_and_test_gap_fixes_10rfd.md:366 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0106/README.md](../../src/core/base/testdata/rfd0106/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:818、docs/design/200_dqn_agents.en.md:822、docs/memo/0158_eval_schedule_first_fire_10rfd.md:230 ほか 1 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0109/README.md](../../src/core/base/testdata/rfd0109/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:428、src/core/base/testdata/rfd0110/run_validation.py:15、src/core/base/testdata/rfd0121/run_validation.py:31 ほか 7 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0110/README.md](../../src/core/base/testdata/rfd0110/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:426、docs/design/160_applications_and_tools.en.md:427、docs/memo/0195_document_term_rfd_and_4digit_numbers_10rfd.md:329 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0111/README.md](../../src/core/base/testdata/rfd0111/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/210_metrics_viewer.jp.md:1113、docs/design/210_metrics_viewer.en.md:1115 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0114/README.md](../../src/core/base/testdata/rfd0114/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:424、docs/design/160_applications_and_tools.en.md:425、docs/memo/0114_findings_intake_and_fix_rfd_20impl.md:195 ほか 5 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0115/README.md](../../src/core/base/testdata/rfd0115/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/120_environments.jp.md:266、docs/design/140_observability.jp.md:423、docs/design/120_environments.en.md:267 ほか 4 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0116/README.md](../../src/core/base/testdata/rfd0116/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:816、docs/design/212_rnd_lifelong_novelty.jp.md:379、src/core/base/testdata/rfd0116/run_performance_atari_test.py:10 ほか 13 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0117/README.md](../../src/core/base/testdata/rfd0117/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:387、docs/design/130_neural_networks.jp.md:375、docs/design/100_runtime_and_configuration.en.md:387 ほか 9 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0121/README.md](../../src/core/base/testdata/rfd0121/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:361、docs/design/200_dqn_agents.jp.md:810、docs/design/212_rnd_lifelong_novelty.jp.md:373 ほか 9 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0122/README.md](../../src/core/base/testdata/rfd0122/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/160_applications_and_tools.jp.md:422、docs/design/160_applications_and_tools.en.md:423、docs/memo/0125_dqn_learner_update_skeleton_10rfd.md:117 ほか 1 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0123/README.md](../../src/core/base/testdata/rfd0123/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:377、docs/design/212_rnd_lifelong_novelty.en.md:378、docs/memo/0123_rnd_intrinsic_reward_scale_10rfd.md:424 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0125/README.md](../../src/core/base/testdata/rfd0125/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/200_dqn_agents.jp.md:814、src/core/base/testdata/rfd0128/run_validation.py:15、docs/design/200_dqn_agents.en.md:818 ほか 3 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0127/README.md](../../src/core/base/testdata/rfd0127/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:421、docs/design/212_rnd_lifelong_novelty.jp.md:418、docs/design/140_observability.en.md:422 ほか 5 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0128/README.md](../../src/core/base/testdata/rfd0128/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/150_replay_buffer.jp.md:363、docs/design/200_dqn_agents.jp.md:812、docs/design/212_rnd_lifelong_novelty.jp.md:375 ほか 9 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0130/README.md](../../src/core/base/testdata/rfd0130/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:371、src/core/base/testdata/rfd0130/run_performance.py:49、docs/design/212_rnd_lifelong_novelty.en.md:372 ほか 3 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0158/README.md](../../src/core/base/testdata/rfd0158/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/140_observability.jp.md:454、docs/design/140_observability.en.md:455、docs/memo/0131_rfd_9xx_renumber_by_creation_date_10rfd.md:69 | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0170/README.md](../../src/core/base/testdata/rfd0170/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/130_neural_networks.jp.md:392、docs/design/140_observability.jp.md:425、docs/design/130_neural_networks.en.md:395 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0182/README.md](../../src/core/base/testdata/rfd0182/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/011_source_layout.jp.md:23、docs/design/200_dqn_agents.jp.md:808、docs/design/011_source_layout.en.md:24 ほか 1 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0191/README.md](../../src/core/base/testdata/rfd0191/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/212_rnd_lifelong_novelty.jp.md:369、docs/design/212_rnd_lifelong_novelty.en.md:370、docs/memo/0191_rnd_raw_error_and_recent_learning_10rfd.md:787 ほか 2 ファイル | AGENTS.md:311「検証」 |
| 検証資材 | [src/core/base/testdata/rfd0194/README.md](../../src/core/base/testdata/rfd0194/README.md) | 該当領域の検証。設計文書のテスト章から引く | 人間・Claude・Codex | docs/design/100_runtime_and_configuration.jp.md:388、src/core/base/testdata/rfd0191/README.md:28、docs/design/100_runtime_and_configuration.en.md:388 ほか 2 ファイル | AGENTS.md:311「検証」 |
| CI | [.github/workflows/gemini_ai_review.yml](../../.github/workflows/gemini_ai_review.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/memo/0156_cicd_improvement_10rfd.md:7、docs/wiki/categories/311_dev_tools.md:198、reports/stats/stats.json:7929 | 実在と参照元を rg で確認 |
| CI | [.github/workflows/pages.yml](../../.github/workflows/pages.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | .agents/skills/anet-translate-docs/SKILL.md:8、.claude/skills/anet-translate-docs/SKILL.md:8、docs/memo/0156_cicd_improvement_10rfd.md:7 ほか 4 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/release.yml](../../.github/workflows/release.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/adr/0017-release-zip-mirrors-repo-layout.md:16、docs/management/issues/issue-20261103-554849.md:41、docs/management/issues/issue-20261103-b13ed2.md:33 ほか 12 ファイル | 実在と参照元を rg で確認 |
| CI | [.github/workflows/windows-ci.yml](../../.github/workflows/windows-ci.yml) | ビルド・配信・CI を直すとき | 人間・Claude・Codex | docs/design/040_development_environment.jp.md:45、.github/workflows/release.yml:56、README.md:3 ほか 9 ファイル | 実在と参照元を rg で確認 |
| Python tool | [apps/tools/atari5.py](../../apps/tools/atari5.py) | Atari-5 を 1 つのベンチマークとして読むとき | 人間・Claude・Codex | AGENTS.md:689、docs/memo/0108_metrics_tag_numbering_10rfd.md:21、docs/memo/done/0103_temporl_reference_alignment_10rfd.md:322 ほか 5 ファイル | AGENTS.md:689 |
| Python tool | [apps/tools/compress_workspace_metrics_test.py](../../apps/tools/compress_workspace_metrics_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:301、docs/wiki/categories/311_dev_tools.md:206、reports/stats/stats.json:7211 | AGENTS.md:301 |
| Python tool | [apps/tools/inspect_run.py](../../apps/tools/inspect_run.py) | Run を分析するとき | 人間・Claude・Codex | AGENTS.md:654、docs/design/030_user_guide_analysis.jp.md:369、docs/design/140_observability.jp.md:333 ほか 62 ファイル | AGENTS.md:667 |
| Python tool | [apps/tools/inspect_run_test.py](../../apps/tools/inspect_run_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:304、src/core/base/testdata/rfd0089/README.md:24、docs/memo/0081_evalpanel_episode_trace_10rfd.md:150 ほか 16 ファイル | AGENTS.md:304 |
| Python tool | [apps/tools/metrics_source_test.py](../../apps/tools/metrics_source_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:303、docs/memo/done/0054_inspect_run_20impl.md:183、docs/memo/done/0054_inspect_run_21impl.md:123 ほか 2 ファイル | AGENTS.md:303 |
| Python tool | [apps/tools/mlflow_bridge.py](../../apps/tools/mlflow_bridge.py) | Run フォルダ名の規約を変えるとき | 人間・Claude・Codex | AGENTS.md:603、apps/41_mlflow_bridge.bat:38、apps/tools/resolve_workspace_test.ps1:137 ほか 5 ファイル | AGENTS.md:603 |
| Python tool | [apps/tools/mlflow_bridge_test.py](../../apps/tools/mlflow_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:302、docs/memo/done/0046_workspace_20impl.md:30、docs/wiki/categories/311_dev_tools.md:211 ほか 1 ファイル | AGENTS.md:302 |
| Python tool | [apps/tools/optuna_metrics_gzip_test.py](../../apps/tools/optuna_metrics_gzip_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:307、docs/memo/0182_source_layout_20impl.md:190、docs/memo/done/0046_workspace_25impl.md:28 ほか 3 ファイル | AGENTS.md:307 |
| Python tool | [apps/tools/optuna_workspace_test.py](../../apps/tools/optuna_workspace_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:306、docs/memo/0182_source_layout_20impl.md:190、docs/memo/done/0046_workspace_25impl.md:22 ほか 3 ファイル | AGENTS.md:306 |
| Python tool | [apps/tools/tb_bridge_test.py](../../apps/tools/tb_bridge_test.py) | 標準手順で検証するとき | 人間・Claude・Codex | AGENTS.md:305、docs/wiki/categories/311_dev_tools.md:220、reports/stats/stats.json:7837 | AGENTS.md:305 |
| Python tool | [reports/tools/stage_issue_memos.py](../../reports/tools/stage_issue_memos.py) | コミットの準備で課題のメモを部分ステージするとき | 人間・Claude・Codex | AGENTS.md:487、.agents/skills/README.md:22、.agents/skills/anet-create-issue/SKILL.md:220 ほか 11 ファイル | AGENTS.md:487 |
| 設計の記録 | [docs/adr/](../../docs/adr/)（93 ファイル） | 契約・構造を変えるとき。ファイル名の決定文を領域の語で絞る | 人間・Claude・Codex | AGENTS.md:104、.agents/skills/README.md:23、.agents/skills/anet-create-rfd/ADR-FORMAT.md:3 ほか 200 ファイル | AGENTS.md:102「設計判断の記録(ADR)を読む条件」 |
| 指示文書 | [docs/memo/](../../docs/memo/)（直下 127・done 241・frozen 6・dropped 2 ファイル） | RFD は指示、状態と順序は課題。過去の保管場所も検索する | 人間・Claude・Codex | .agents/skills/README.md:23、.agents/skills/anet-create-issue/SKILL.md:105、.agents/skills/anet-create-rfd/SKILL.md:12 ほか 193 ファイル | docs/memo/README.md:5「状態と作業順はissueで表す」 |
| 課題記録 | [docs/management/issues/](../../docs/management/issues/)（単票 245 ファイル・INDEX.md・README.md） | 課題の一覧・起票・判定、RFD の起票と実装の完了 | 人間・Claude・Codex | AGENTS.md:499、.agents/skills/README.md:17、.agents/skills/anet-audit/SKILL.md:8 ほか 97 ファイル | docs/management/issues/README.md:3 |

## 表 2 作業から読む文書

| これから何をするか | 読むもの（文書と章） | 読まなくてよいとき | 根拠 |
|---|---|---|---|
| C++ を書く・直す | docs/quality/impl.md の「C++ コード」「名前」「コメントと TODO」「計測」「GetScalar」「所有権」「ログ」「失敗の扱い」 | C++ を変えないとき | AGENTS.md:65「作業ごとに読むもの」 |
| Python・Java・PowerShell を書く・直す | docs/quality/impl.md の「コメントと TODO」「ログ」 | 該当の言語を変えないとき | AGENTS.md:66 |
| 公開ヘッダ・ライブラリの境界・テストの置き場を変える | AGENTS.md「基本方針」の公開ヘッダと置き場の 6 行、docs/design/011_source_layout.jp.md、010 の「4.3 拡張点」 | 既存の境界の中で書くとき | AGENTS.md:47〜52、docs/design/README.jp.md:37、docs/design/010_framework_overview.jp.md:157 |
| 設定ファイルを書く・直す（番号付きの bat の `run.$` を含む） | docs/quality/impl.md の「設定ファイル」「名前」、020 の「3.6 どのファイルに何を書くか」「3.7 例」、`src/core/base/testdata/rfd0072/check_default_leaves.py` | 設定を変えないとき。設定の変更は実装の工程の中で同時に直す | AGENTS.md:67・77、docs/design/020_user_guide_run.jp.md:291・307 |
| RFD を起票する（グリル） | `anet-create-rfd`、docs/quality/rfd.md、impl.md の「名前」、wiki の目録と題材のトピック別まとめ、会話のまとめページ（`anet-summary-page`、docs/process/summary-page/grill.md） | 実装だけの工程。RFD が要りそうなだけのときは始めず、課題と暫定 RFD の起票を提案して了承を待つ | AGENTS.md:68・135・417・467、docs/memo/README.md:35、docs/process/summary-page/README.md「場面の一覧」 |
| RFD を再グリルする（精査） | `anet-create-rfd` に既存の RFD を渡す。docs/quality/rfd.md、RFD 冒頭の改訂履歴 | 起票と同じ枠では再グリルを置き換えない | docs/memo/README.md:36 |
| 調整する | 人間が裁定し、必要ならエージェント間を仲介する。RFD は self-contained に保つ | - | docs/memo/README.md:37・46 |
| 実装計画を作る | `anet-impl-rfd`、RFD、`anet-impl-rfd/RFD-IMPLEMENTATION.md`、impl.md の「実装計画」 | skill の実行そのものを計画に代える RFD | docs/memo/README.md:38、.agents/skills/anet-impl-rfd/SKILL.md:30 |
| 編集前の条件読み（契約・構造を変える） | 関連する設計文書、ADR（ファイル名で絞る）、wiki のトピック別まとめ | バグ修正・テスト追加・ログ追加・既存 RFD に沿う実装 | AGENTS.md:102〜121・136・437 |
| ユーザーの判断を RFD の工程の外で規約・コード・設定・プロセスに入れる | AGENTS.md「設計判断の記録(ADR)を起こす条件」、`anet-create-rfd/ADR-FORMAT.md` の「When to offer an ADR」 | RFD の工程の中（`anet-create-rfd`・`anet-impl-rfd` が扱う）、設計の判断が要らない「この場で直す」 | AGENTS.md:123〜131、.agents/skills/anet-create-rfd/ADR-FORMAT.md:29 |
| 新しい名前を付ける | CONTEXT.md（`_Avoid_` を含む）、impl.md の「名前」、AGENTS.md「AI エージェントの応答言語ルール」の避ける語の表 | 日常の実装・修正 | AGENTS.md:140〜150・383 |
| 検証する | 領域の設計文書のテスト章 → `src/core/base/testdata/rfdNNNN/README.md`、AGENTS.md「検証」の標準手順 | 対象外の検証は理由を明記する | AGENTS.md:279〜311 |
| ビルドが詰まる | docs/design/040_development_environment.jp.md の「6. CLI によるビルド・テスト」、AGENTS.md「AI エージェントでのビルド注意事項 (Windows/MSVC)」 | ビルドしない作業 | AGENTS.md:247・714、docs/design/040_development_environment.jp.md:364 |
| レビューする | `anet-review` → `anet-review-agent`、docs/process/protocol.md、docs/process/registry.md、rfd.md / impl.md | 対象が無いとき | docs/memo/README.md:40・48〜54 |
| 人間のレビュー・コミットの準備 | `anet-prepare-commit`、AGENTS.md「Git 操作・コミットメッセージルール」 | サブエージェントのとき、工程の途中 | docs/memo/README.md:41、AGENTS.md:483〜590 |
| 設計文書・ユーザーガイドを書く | impl.md の「検証と報告」 | - | AGENTS.md:69 |
| AGENTS.md・レビュー観点・プロセス契約・エージェント定義を直す | impl.md の「検証と報告」、直す観点の「位置づけ」、registry.md の「読み替え表」の前文 | - | AGENTS.md:70 |
| skill を書く・直す | impl.md の「skill」、`.agents/skills/README.md` | - | AGENTS.md:71 |
| 手戻りからハーネスを直す | `anet-improve` | 明示起動か、`anet-create-rfd`・`anet-impl-rfd` の完了報告の候補で選ばれたとき以外 | .agents/skills/README.md:30 |
| 会話のまとめページを作る・更新する | `anet-summary-page`、docs/process/summary-page/README.md と場面ごとの指針（grill.md・exp.md・general.md） | 場面の一覧の始める時機に当たらず、頼まれてもいないとき | AGENTS.md:417・418、docs/process/summary-page/README.md「場面の一覧」 |
| Runner を起動する（smoke・Run・bat） | 020 の「2.3 最初に確認する設定」「3.8 確認方法とよくあるエラー」「4.1 標準起動」 | この行による自己点検は無い | AGENTS.md:72・76、docs/design/020_user_guide_run.jp.md:45・372・398 |
| Run を分析する | 030（「6. `inspect_run.py`でRunを検査・抽出する」）、`inspect_run.py`、Atari-5 なら `atari5.py`、wiki の目録 | Run 名だけの提示も分析の依頼として扱う | AGENTS.md:660〜712、docs/design/030_user_guide_analysis.jp.md:369 |
| 実験記録を書く | AGENTS.md「実験記録と実効 config の保存ルール」、docs/experiments/README.md | tmp 付き・評価専用の Run の config は複製しない | AGENTS.md:628〜658・662 |
| 課題を扱う | `anet-create-issue`・`anet-list-issues`・`anet-update-issue`、docs/process/numbering.md | 工程の途中の未決はまず作業記録へ貯める | .agents/skills/anet-create-issue/SKILL.md:21 |
| 定常整備をする | `anet-housekeeping` → 統計・wiki 検査・監査・翻訳・調査・動線マップ・atlas の各 skill | - | .agents/skills/anet-housekeeping/SKILL.md:30〜38、.agents/skills/README.md:32 |

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
| 設計判断の記録(ADR)を起こす条件 | 10 | 条件付き | .agents/skills/anet-create-rfd/ADR-FORMAT.md・docs/process/registry.md | AGENTS.md:123 |
| wiki(docs/wiki)を引く条件 | 7 | 条件付き | docs/wiki/README.md（読む条件は入口に残す） | AGENTS.md:133 |
| 用語(CONTEXT.md)を引く条件 | 12 | 条件付き | docs/quality/impl.md の「名前」 | AGENTS.md:140 |
| 汎用機構と利用側の責任境界 | 42 | 条件付き | docs/quality/impl.md | AGENTS.md:152 |
| Fail-Fast 原則 | 30 | 条件付き | docs/quality/impl.md の「失敗の扱い」 | AGENTS.md:194 |
| 設定値の扱い | 14 | 条件付き | docs/quality/impl.md の「設定ファイル」 | AGENTS.md:224 |
| 改行と文字コード | 9 | 条件付き | docs/quality/impl.md | AGENTS.md:238 |
| ビルド | 32 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:247 |
| 検証 | 34 | 条件付き | docs/design/040_development_environment.jp.md（標準コマンドは入口に残す） | AGENTS.md:279 |
| Python 補助ツールの実行 | 22 | 条件付き | docs/design/040_development_environment.jp.md の「7. Python 補助ツール」 | AGENTS.md:313 |
| 人間へ提示するコマンドの書き方 | 30 | 条件付き | docs/quality/impl.md | AGENTS.md:335 |
| 編集しない・慎重に扱う領域 | 12 | 常時必要 | 入口に残す | AGENTS.md:365 |
| AI エージェントの応答言語ルール | 30 | 常時必要 | 入口に残す（避ける語の表は RFD と実装の観点の検索の正本を兼ねる） | AGENTS.md:377 |
| AI エージェントの概念説明ルール | 14 | 常時必要 | 入口に残す（会話のまとめページの始める時機の 2 行を含む。どの会話にも当てる） | AGENTS.md:407 |
| LLM コーディング規律 (karpathy-guidelines) | 11 | 条件付き | .agents/skills/karpathy-guidelines/SKILL.md | AGENTS.md:421 |
| AI エージェントの作業ルール | 39 | 常時必要 | 入口に残す（「RFD が要りそうなとき」の 5 行を含む） | AGENTS.md:432 |
| 長時間のビルド・テスト・バッチ検証 | 12 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:471 |
| Git 操作・コミットメッセージルール | 109 | 条件付き | .agents/skills/anet-prepare-commit/SKILL.md（git の禁止は入口に残す） | AGENTS.md:483 |
| Run 命名・後片付けルール | 36 | 条件付き | docs/design/020_user_guide_run.jp.md | AGENTS.md:592 |
| 実験記録と実効 config の保存ルール | 32 | 条件付き | docs/experiments/README.md | AGENTS.md:628 |
| AI エージェントのRun結果分析ルール | 54 | 条件付き | docs/design/030_user_guide_analysis.jp.md | AGENTS.md:660 |
| AI エージェントでのビルド注意事項 (Windows/MSVC) | 48 | 条件付き | docs/design/040_development_environment.jp.md | AGENTS.md:714 |
| 集計 | 全 28 節・761 行 | 条件付き 20 節・588 行（到達しない節 0） | 移設はこの作業の対象外 | AGENTS.md の全 `##` 見出し |

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

- 0 本。`diff -rq .agents/skills .claude/skills` の出力は空（56 ファイルすべて一致）。RFD0195 の改名（`anet-create-prd` → `anet-create-rfd`、`anet-impl-prd` → `anet-impl-rfd`、`PRD-IMPLEMENTATION.md` → `RFD-IMPLEMENTATION.md`）と、新設の `anet-summary-page` も両側で一致している。

### 欠落

- テスト章の無い設計文書: `212_rnd_lifelong_novelty.jp.md`、`220_atari_env.jp.md`。212 の最終 2 章は「8. Network の構造図」「9. 背景、実装と検証」で、testdata README への参照は §9.6「検証の記録」にある。220 の最終 2 章は「6. ビルド統合」「7. 関連文書」で、testdata README への参照は §6・§7 の本文にある。
- 「関連文書」章の無い文書: 1xx・2xx では `212_rnd_lifelong_novelty.jp.md`（§9.9「出典と関連文書」を節として持つ）。0xx では `011_source_layout.jp.md` と `040_development_environment.jp.md`。
- 章題・章番号のゆれ: `160_applications_and_tools.jp.md` だけ「テストと変更時の確認事項」（ほかは「テストと拡張時の確認事項」）。テスト章の章番号は 7〜12 に分かれる（214 が 7、140 が 9、200 が 10、210 が 12、ほかは 8）。`011_source_layout.jp.md` は章番号が無く（「ライブラリとアプリ」「テスト」「出力の置き場」）、索引の 1 行だけから到達する。同じ規則を AGENTS.md「基本方針」の 6 行（AGENTS.md:47〜52）が持つが、011 へのリンクは無い。
- 索引に無い設計文書: `docs/design/optuna.md`（`NNN_*.jp.md` の命名でもない）。030・140・160 の本文と関連文書からは引かれている。
- Codex 表示名（`agents/openai.yaml`）の無い project skill: anet-impl-rfd, grill-me, handoff, improve-codebase-architecture, karpathy-guidelines, tdd。汎用 skill にも必須かは決まっていない。
- 資料と実測の不一致: skills README の「追加と更新の手順」1（`.agents/skills/README.md:50`）は `agents/openai.yaml` を添えると書くが、プロジェクト固有の `anet-impl-rfd` だけが持たない（改名前の `anet-impl-prd` から無い）。
- 資料と実測の不一致: 検証資材の README 30 本（計 96 行）の実行手順に、RFD0182 で移す前のパス（`core/anet-core`）と実行体名（`anet-core-test`）が残る。30 本とも先頭に「AGENTS.md「検証」に合わせて読み替える」の注記がある（注記は全 40 本のうち 31 本）。ただし AGENTS.md「検証」は標準手順だけを持ち、旧い名前から新しい名前への対応表は持たない（対応は RFD0182 の「5. 実装契約」にある）。動線の末端で、読み替えの材料が入口から引けない。
- 資料と実測の不一致: 186eb184 で削除した `docs/agents/issue-tracker.md` を指す Markdown のリンクが、直下の RFD `docs/memo/0114_findings_intake_and_fix_rfd_10rfd.md`（37・417 行）に残り、リンク切れになる。memo README の「リンクの注意」は移動・改名した文書のリンクの直し方だけを定め、削除した文書を指すリンクの扱いは定めていない。
- 資料と実測の不一致: 契約は工程図のレーンを「人間 / 起票側 / 精査側 / 実装側 / 共通文書 / CI」の 6 本と定めるが、図は CI を外した 5 本にした。Archify の workflow はレーンの高さが 104px 固定で、6 本では図の高さが 838px 以上になる。showcase の可読性（1440×900 で文字 6px 以上）を満たす幅は 1240px 以下で、画面内に収める横長の比（1.55 以上）と両立しない（2026-11-19 の生成で、6 本では幅 1310px で文字 5.7px になり検証が落ちた）。CI の読む条件と参照元は表 1、push 後の CI は俯瞰図の CI ノードが持つ。
- 資料と実測の不一致: 工程図の「ユーザー環境」は主経路に置いたが、列が足りず「設計」の列に入る。Archify の workflow は列の位置が固定（88・220・300・430・500・625）で、1・2 列目と 3・4 列目は同じレーンに並べられない。

### 死んだ動線

- 0 件（表 1 で参照元が空の行は無い）。
- 参照元が RFD・課題・wiki・ADR・実験記録・レポートだけで、エージェントが読む契約（AGENTS.md・skill・設計文書・レビュー観点・プロセス契約）から名指しされないもの: docs/experiments/README.md、.github/workflows/gemini_ai_review.yml、.github/workflows/release.yml。`docs/experiments/README.md` は AGENTS.md「実験記録と実効 config の保存ルール」が置き場を定めるのに、README 自体へのリンクが無い。

### 機械化点

- 実体コピーの同期検査: `diff -rq .agents/skills .claude/skills` を hook か CI で回す（今は手作業。skills README:3）。
- CI のパス条件化: ハーネスの文書だけの変更で `windows-ci.yml` のビルドを回さない条件。
- 改名・配置替えの後の旧い名前の検索: RFD0182・RFD0195 の受入で使った `git grep` を、ハーネスの文書（検証資材の README を含む）にも定常整備で回す。
- リンク切れの検査: memo README の「リンクの注意」のループを `docs/` 全体へ定常整備で回し、削除した文書を指すリンクも拾う。
- 参照元の実測: 表 1 の参照元の `rg` を定常整備で回し、死んだ動線と、契約から名指しされない末端の増減を見る。
- 設計文書のテスト章と関連文書章の有無、章題を検査する。
- skill ごとの `agents/openai.yaml` の有無を skills README の手順と突き合わせる。

### evidence gap

- ユーザー環境（global の指示・user-level skill・memory）の実体と読めたかどうかは個人版にだけ書く。
- shadow が実際にどちらを起動するか（解決順）は実行して確かめていない。名前の一致と中身の差だけを測った。
- `.claude/settings.local.json` はローカルの設定なので読んでいない。
- 参照元は文字列の一致で数えた。リンクの文脈（読む指示か、記録か）は表 1 の「読む条件」で手で補った。生成物（`docs/agents/README.md`、`docs/agents/harness_map_*`、`docs/archify/`、`*.html`）と、全ノードを列挙する `anet-harness-map` skill 自身は参照元から除いた。
- 工程の正本（docs/memo/README.md「工程と担当」）と、RFD の改訂履歴の行は突き合わせていない。改訂の行はどの枠で再グリルしたかを持たないので、「少なくとも起票と精査は分ける」が守られたかを実測できない。
- 調査の間、別の定常整備の作業が `docs/management/issues/`・`INDEX.md`・wiki の課題の区画・`docs/design/*.en.md` を書き換えていた。これらを参照元に含む行の件数と `.en.md` の行番号は、測った時点のもの。
- C++ の build と test は、この作業に含めず実行していない。
