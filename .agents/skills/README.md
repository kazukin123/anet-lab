# スキル一覧

この README は `.agents/skills/`(Codex が読む)と `.claude/skills/`(Claude Code が読む)の両方に同じ内容で置く。2 つのディレクトリは実体コピーで、ジャンクションではない。SKILL.md やこの表を直したら両方へ反映し、`diff -rq .agents/skills .claude/skills` で差が無いことを確認する。

起動は Codex が `$name`、Claude Code が `/name`。引数はその後ろに空白区切りで続ける。

## プロジェクト固有

このリポジトリのために書いたもの。`anet-` 接頭辞は、定常整備系に限らない(`anet-review` など)。Codex 向けの表示名は各 `agents/openai.yaml` にある。

| スキル | 用途 | 起動例 | 主な出力先 |
|---|---|---|---|
| anet-housekeeping | 残トークンと残時間の予算内で、下の定常整備を自動選定して順に回す。予算は `auto` / `S` / `M` / `L` / `NN%`。Codex は残量 API が無いので予算引数が必須 | `/anet-housekeeping S dry-run`、`$anet-housekeeping M` | `reports/housekeeping/runs.jsonl` |
| anet-stats | git 履歴とローカル Run フォルダから定点統計を再生成し、digest を追記する。分類の正本は `reports/tools/categories.json` | `/anet-stats` | `reports/stats/` |
| anet-ingest-wiki | 正本をwikiへ取り込む。地図新設時は課題区画も生成する | `$anet-ingest-wiki` | `docs/wiki/`、INDEX |
| anet-lint-wiki | 取り込み漏れ・撤回・リンク・実装済みPRDを点検し、派生一覧を再生成する | `$anet-lint-wiki max=3` | `docs/wiki/`、INDEX |
| anet-audit | 観点1つ×範囲1つで定点監査し、指摘をissueへ記録する。既存課題はAIの判定だけを更新する | `$anet-audit lens=refactor scope=318_rl_core` | `docs/management/issues/`、INDEX、wikiの課題区画 |
| anet-translate-docs | `docs/design/*.jp.md` を 1 本ずつ英訳して `.en.md` を隣に置く。訳元の blob hash を 1 行目に持つ | `/anet-translate-docs 010_framework_overview.jp.md` | `docs/design/*.en.md`、`reports/translate/glossary.json` |
| anet-survey-queue | `reports/INDEX.md` のキューで状態 `ready` のサーベイを 1 件回し、済みへ移す。`survey` か `paper-details` スキルが必要 | `/anet-survey-queue` | `reports/*.md`、`reports/INDEX.md` |
| anet-archify-atlas | Archify スキルで 5 図の architecture atlas を生成する。明示起動のみ | `$anet-archify-atlas` | `docs/archify/` |
| anet-harness-map | AI ハーネス(指示・規約・設計文書・skill・memory・CI)の動線マップを再生成する。公開版は `docs/agents/`(README + Archify 2 図)、個人版は `.scratch/harness-map/`。Archify スキルが必要。明示起動と anet-housekeeping の変更検知から | `$anet-harness-map` | `docs/agents/`、`.scratch/harness-map/` |
| anet-prepare-commit | 会話と Git 差分からコミット文案と対象ファイル一覧を整理し、`コミット対象` と `コミット対象(一部)` の対象の hunk をステージするかをボタン 1 つで聞く。選ばれたときだけ `git add` と部分 stage（`git apply --cached`）。commit も push もしない | `/anet-prepare-commit`、`$anet-prepare-commit`、作業の終わりの提案で「コミットの準備をする」を選ぶ | 報告。選ばれたときは Git index |
| anet-create-prd | 題材または既存 PRD から、根拠を調べた質疑、起草、任意の別モデル検討（Claude Code だけ）、自己点検、レビューと差し戻し対応まで進め、未決を示して停止する。実装へ進まない。既存 PRD は要点を示してから改訂する。グリルの規律は mattpocock/skills の grill-with-docs から派生 | `/anet-create-prd 091_checkpoint_load_optimizer_options_10prd.md` | `docs/memo/*_10prd.md`、`CONTEXT.md`、`docs/adr/`、`docs/wiki/` |
| anet-create-issue | 選ばれた課題を1件1ファイルで記録し、一覧を作り直して書式を検証する | `$anet-create-issue` | `docs/management/issues/`、INDEX、wikiの課題区画 |
| anet-list-issues | 本版の課題を依存とWSJFで並べ、棚卸しと束ねる提案を示す。記録は変えない | `$anet-list-issues`、`$anet-list-issues 束ね max=5` | 報告 |
| anet-update-issue | 指示された状態・監理項目、またはAIの判定を更新し、一覧を作り直して書式を検証する | `$anet-update-issue` | `docs/management/issues/`、INDEX、wikiの課題区画 |
| implement-prd-with-docs | PRD と現行実装を照合し、実装計画を保存して実装・検証する。自己点検、レビューと差し戻し対応を行い、未決を示して停止する | `/implement-prd-with-docs 013_sample_prefetch_10prd.md` | `docs/memo/*_2ximpl.md`、計画に従うコード・テスト・設定・文書 |
| anet-review | 成果物(PRD、実装計画、PRD の無い変更を含む実装)をレビューエージェント `anet-review-agent` に出し、応答を平易に言い直して示す。指摘があれば修正・再レビューの選択肢を出し、選ばれたときだけ対応する。引数が無ければ、その会話で変更したファイルが対象。明示起動のみ | `/anet-review`、`$anet-review 087_review_agent_quality_viewpoints_10prd.md` | 報告(応答の原文は毎回 `.scratch/` に保存する。実装のレビューでは差分も `.scratch/` に書き出す)。選ばれたときは対象の修正と再レビュー |
| anet-improve | この会話の手戻りからハーネスの改善策を選んで反映し、PRD送りと採らなかった問題をissueへ記録する | `$anet-improve` | ハーネスの変更先、`.scratch/anet-improve/`、`docs/management/issues/`、派生一覧 |

anet-housekeeping が呼ぶ候補と対応: stats → anet-stats、wiki-lint → anet-lint-wiki、audit → anet-audit、translate → anet-translate-docs、survey → anet-survey-queue、harness-map → anet-harness-map、atlas → anet-archify-atlas。

anet- スキル共通の約束(対象は定常整備系。anet-housekeeping と、それが呼ぶ上の候補): git の書き込み操作をしない(人が行う)、実行前の対象・順序の了承確認は anet-housekeeping が行い、了承後の各作業は質問して止まらない、出力先は各スキルが定める固定パスだけ、1 単位ごとに保存して途中終了しても成果が残る。出力は `reports/` 配下に置く(`docs/` は GitHub Pages に配信されるため)。公開が目的のもの(英訳 → `docs/design`、atlas → `docs/archify`、動線マップ → `docs/agents`、wiki → `docs/wiki`、課題 → `docs/management/issues`)だけは `docs/` 配下に置く。

`wiki_support.py maps` はINDEXと地図の課題区画、ファイルとPRD一覧を作り直す。全入力と目印を検証してから書き、区画外とwikiログを保つ。

## 汎用(導入済み)

| スキル | 用途 | 出所 |
|---|---|---|
| grill-me | 計画や設計を決定木の枝ごとに問い詰め、共通理解に到達させる | mattpocock/skills(skills-lock.json) |
| to-prd | 会話の文脈から PRD を起こす。この repo では `docs/memo/*_10prd.md` に置く | mattpocock/skills(skills-lock.json) |
| handoff | 会話を別エージェントへ引き継ぐための要約文書を作る | mattpocock/skills(skills-lock.json) |
| setup-matt-pocock-skills | AGENTS.md と `docs/agents/` に issue tracker と triage label の規約を用意する。初回導入用 | mattpocock/skills(skills-lock.json) |
| improve-codebase-architecture | CONTEXT.md と ADR を手掛かりに deep module 化の機会を探す。anet-audit の architecture 観点が語彙を借りる | 手動コピー(lock 未登録) |
| tdd | red-green-refactor でのテスト先行開発 | 手動コピー(lock 未登録) |
| karpathy-guidelines | LLM がやりがちな過剰実装・想定の隠蔽を避ける行動指針 | 手動コピー(lock 未登録) |

## 追加と更新の手順

1. `.agents/skills/<name>/SKILL.md` を書く。frontmatter の `name` と `description` は必須で、`description` に起動条件(`$name` / `/name`)を含める。Codex の表示用に `agents/openai.yaml` を添える(`interface.display_name` / `short_description` / `default_prompt`、暗黙起動を避けるなら `policy.allow_implicit_invocation: false`)。
2. `.claude/skills/<name>/` へ丸ごとコピーする。
3. この README の表を両方で更新する。
4. 外部スキルは skills CLI で導入し、`skills-lock.json` に記録される。手動でコピーしたものは表の出所に「lock 未登録」と書く。
