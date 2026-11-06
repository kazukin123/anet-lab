---
name: anet-harness-map
description: Regenerate anet-lab's AI harness reading-path map (harness map). Investigate the current checkout and the user-level agent environment, then rebuild the public map (docs/agents/README.md plus two Archify diagrams) and the personal map (.scratch/harness-map/). Requires the archify skill. Use when the user invokes $anet-harness-map or /anet-harness-map, or from anet-housekeeping when the harness-map candidate is selected.
---

# ANET Harness Map

anet-lab の AI ハーネス(エージェントの振る舞いを決めるファイル群: 指示、規約、設計文書、PRD/ADR/用語集、検証資材、skill、memory、CI)を現行 checkout とユーザー環境から調査し、**動線マップ**を再生成する。公開版は `docs/agents/` へ、個人版は `.scratch/harness-map/` へ出力する。一度きりの成果物ではなく、ハーネスが変わるたびに作り直す。

## 起動条件

- `$anet-harness-map` / `/anet-harness-map` で明示的に指定された場合、または anet-housekeeping が候補 `harness-map` を選んだ場合に実行する。引数は無い。
- 最初に、利用可能なスキルから名前が正確に `archify` であるものを探し、その `SKILL.md` を最後まで読む。この確認が完了するまで、候補を含む成果物を作成・変更しない。
- `archify` を発見できない、または全文を読めない場合は、次のエラーを返して終了する。

  ```text
  ERROR: anet-harness-map requires the "archify" skill, but it is unavailable or unreadable. No artifacts were created.
  ```

- Archify が定める実行経路と fallback だけを利用する。Mermaid、draw.io、独自レンダラーなど、Archify 外の代替経路を追加しない。
- Plan Mode では調査と計画だけを行い、成果物を生成しない。実行可能なモードでのみ、以下のワークフローを進める。

## ワークフロー

1. `git rev-parse --show-toplevel` などのリポジトリ情報から root を特定する。root から適用範囲にある作業規約を読み、Git・ビルド宣言・プロジェクト資料の複数の手掛かりから対象が anet-lab であることを確認する。単一の固定パスだけを識別条件にしない。
2. [Harness map contract](references/harness-map-contract.md) を全文読み、用語、公開マニフェスト、README の 6 節、2 図の契約、品質基準を確定する。
3. 現在の Git revision を取得し、ハーネスを調査する。名称を決め打ちせず、役割と内容から発見する。
   - **入口**: `CLAUDE.md`、`AGENTS.md`(全 `##` 節の見出しと行数を取る)。
   - **リポジトリ内の文書**: `docs/agents/*.md`、`docs/design/README.jp.md` の索引と各設計文書の最終 2 章、`CONTEXT.md`、`docs/adr/`(件数)、`docs/memo/`(直下 / `done` / `frozen` / `dropped` の件数)、`core/anet-core/testdata/*/README.md`、`docs/ownership_guideline.md`、`docs/experiments/README.md`、`.github/workflows/`、`AGENTS.md` が名指しする Python tool。
   - **skill**: project の `.agents/skills` と `.claude/skills` を `diff -rq` で比較する。user-level の `~/.claude/skills`、`~/.agents/skills`、`~/.codex/skills` を一覧し、project と同名のものは `diff -rq` で中身を比較する。各 skill の `agents/openai.yaml` の有無を取る。
   - **ユーザー環境**: `~/.claude/CLAUDE.md`、`~/.codex/AGENTS.md`、memory `~/.claude/projects/<slug>/memory/`(件数と `MEMORY.md` の index 行数)。`<slug>` はリポジトリ絶対パスの `:` `\` `/` を `-` に置換した名前。見つからなければ evidence gap として記録して続行する。
   - **参照元**: 各ノードがどこからリンク・名指しされているかを `rg` で実測する。参照元が無い末端が「死んだ動線」になる。
   - 現行の読む条件は `AGENTS.md` の本文を根拠にし、設計文書の索引と skills README を補助にする。過去資料や生成済み成果物だけを根拠にしない。
4. 公開 `README.md` の 6 節(凡例と図の索引、表 1〜4、発見事項)を契約どおりに組む。表の各行に根拠(ファイルと行、または節名)を付ける。発見事項は名指しするだけで、対象を修正しない。
5. OS の一時領域へ `architecture` と `workflow` の Archify JSON 候補を作る。各図について、読み込んだ Archify スキルの type router に従い、該当 schema、common schema、対応例だけを読む。候補作成後の検証・修正・delivery・visual-check も Archify の契約に従う。
   - `meta.quality_profile` は `showcase`。`meta.locale` は設定しない(日本語 authored、固定 Viewer UI は英語)。
   - `validate --quality showcase --json` で 9 件すべての artifact check が成功し、composition errors と warnings がともに 0 になることを必須とする。validation 成功後の JSON は編集しない。`deliver` が非 0 で終了した図は成功扱いしない。
   - 画像を確認できる場合は `visual-check --json` の出力を目視する。1 図につき最大 2 回まで原因を限定して修正する。目視手段が無ければ visual review を `skipped` として扱い、reviewed と偽らない。
6. 2 図と README が契約を満たした場合だけ、`docs/agents/` へ公開する。
   - 契約の固定マニフェスト(README と 2 図の JSON・HTML の計 5 ファイル)だけを置換対象にする。`docs/agents/` にある他のファイルには触れない。
   - 既存の公開版がある場合は一時バックアップを取り、全ファイルの置換が完了するまで保持する。失敗したら置換対象だけを前回版へ戻し、今回新設した対象は除去する。
   - 2 組の JSON と HTML を置いた後、完了マーカーとして `README.md` を最後に置換する。
7. 個人版を `.scratch/harness-map/` に生成する(公開版と同じ basename の README と 2 図)。
   - 公開版の JSON を複製し、ユーザー環境境界の中身(global `CLAUDE.md`、user-level skills と shadow の一覧、memory、Codex の user-level 入口)を実ノード化する。workflow への追加ノードは契約の条件付き(Archify の交差制約を満たせるときだけ)。満たせなければ公開版と同じノード構成にし、理由を個人版 README の差分節に書く。
   - 公開版のノード id が個人版に全て含まれることを `diff` で確認する。含まれなければ公開版と同じ手順で修正する。
   - 個人版も Archify の validate を通す。`.scratch/` は gitignored なので `git status` に出ないことを確認する。
8. 今回作成した一時ファイルとバックアップだけを片付ける。

## 停止条件

- いずれかの図で、実証された主経路と 3 要素を確保できなければ、空の図や推測図を作らず公開版を更新しない。
- Archify の validation または delivery が失敗し、2 回連続の限定修正でも最良のエラー数を改善できなければ停止し、前回の公開版を維持する。
- ユーザー環境の一部が読めない場合は、読めた範囲で個人版を作り、読めなかった対象を個人版 README の evidence gap に書く。公開版はユーザー環境の実体に依存しないので止めない。
- 失敗時は、どの段階で止まったか、前回版を維持できたか、未解決の診断を正確に報告する。

## 禁止

- git の書き込み操作(add、commit、push、checkout、stash、branch、reset)。人間が行う。
- `AGENTS.md`、`CLAUDE.md`、skill、設計文書、user-level のファイルの修正。発見事項(shadow、実体コピーのドリフト、欠落、死んだ動線)は名指しするだけで直さない。直すのは別の PRD の仕事。
- 固定マニフェスト以外への書き込み。`docs/` は GitHub Pages に配信されるので、個人環境の実体(どの skill が shadow か、memory の内容)を公開版に書かない。

## 完了報告

- 公開 5 ファイルと個人版 3 ファイルのリポジトリ相対パス。
- **件数**: 死んだ動線(表 1 で参照元が空の行数)、実体コピーのドリフト本数(表 4)、shadow の本数と中身違いの本数(エージェント別)、移設候補の節数と行数合計(表 3 の集計行)。anet-housekeeping から呼ばれた場合、この件数は `runs.jsonl` の `note` に書かれる。
- 2 図が内部検証を通過したことだけを短く示す。個別の検証結果、SHA-256、visual review status は公開コンテンツへ追加しない。
- evidence gap と、資料の記述と実測の不一致を要約する。
- C++ の build/test はこのスキルの作業に含めず、実行していないことを明記する。
