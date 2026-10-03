# ANET Harness Map Contract

動線マップの出力仕様。起票時の判断は `docs/memo/080_harness_map_10prd.md` にあり、本契約が出力の正本になる。

## 用語

成果物の凡例にこの 6 語を載せ、避け語を使わない。`CONTEXT.md` は RL ドメインの用語集なので、プロセス語彙であるこれらは載せない。

| 語 | 定義 | 避け語 |
|---|---|---|
| **AI ハーネス** | エージェント(Claude Code / Codex)の振る舞いを決めるファイル群の総体。指示・規約・設計文書・PRD/ADR/用語集・検証資材・skill・memory・CI。コード本体は含まない | 環境、設定 |
| **入口** | 条件なしに毎セッション読まれるファイル。`CLAUDE.md`、`AGENTS.md`、global `CLAUDE.md`、memory の index | 常時読み、エントリ |
| **条件読み** | 作業種別や領域で読むかどうかが決まる読み。入口以外の読みはすべてこれ | 参照、オンデマンド |
| **動線** | 入口から末端(文書の章・skill・ツール)まで、エージェントが辿る参照の連なり。参照元の無い末端は「死んだ動線」 | 流路、経路、フロー |
| **shadow** | 別スコープ(user-level)に同名の skill があり、project 版の代わりに起動しうる状態。Claude Code は `~/.claude/skills`、Codex は `~/.agents/skills` が user-level | 上書き、重複 |
| **実体コピー** | 同じスコープ内で意図して同内容を 2 実体として置いている状態(`.agents/skills` と `.claude/skills`)。同期は手作業 | 重複、ミラー |

shadow は消すべき事故、実体コピーは意図した構造で同期だけが問題、と対処が正反対なので、両方を「重複」と呼ばない。

## 公開マニフェスト

`docs/agents/` には次の固定名だけを置く。更新時は同じ名前を置換し、世代別コピーを作らない。`docs/agents/` にある他のファイル(`domain.md`、`issue-tracker.md`、`triage-labels.md` など)には触れない。

| Archify type | basename | 役割 |
|---|---|---|
| — | `README.md` | 動線マップ本文。図の索引と `docs/agents/` の索引を兼ねる |
| `architecture` | `harness_map_architecture` | ハーネスの俯瞰: ノード、境界、主経路 |
| `workflow` | `harness_map_workflow` | 動線: 主体レーン × 作業段階 |

図は `.archify.json` と `.html` を保存する。visual-check の sidecar、スクリーンショット、contact sheet は公開しない。

## 個人版マニフェスト

`.scratch/harness-map/`(gitignored)に同じ basename で `README.md`、`harness_map_architecture.archify.json` / `.html`、`harness_map_workflow.archify.json` / `.html` を置く。個人版は公開版の上位集合で、公開版のノード id をすべて含む。

## README の構成

公開版 `README.md` は次の 6 節を持つ。**表の各行に根拠(確認したファイルと行、または節名)を必須**とし、陳腐化を `rg` で検出できるようにする。

1. **凡例と図の索引**。用語 6 語。図の索引表は `docs/archify/README.md` の「推奨閲覧順」と同じ形式で、列 = 順序 / 図 / 何が分かるか / 表示 / ファイル。「表示」は GitHub Pages の URL `https://kazukin123.github.io/anet-lab/docs/agents/<basename>.html`、「ファイル」は `.html` と `.archify.json` への相対リンク。生成根拠の Git revision を書く。日本語は Viewer UI の対応 locale ではないため `meta.locale` を省略しており、固定 Viewer UI と `<html lang>` が英語であることを明記する。
2. **表 1 ノード在庫**。ファイル単位で全列挙する。対象は `CLAUDE.md`、`AGENTS.md`、`docs/agents/*.md`、設計文書(`docs/design/*.jp.md` と `README.jp.md`)、`CONTEXT.md`、`docs/ownership_guideline.md`、`docs/memo/README.md`、`docs/experiments/README.md`、`core/anet-core/testdata/*/README.md`、`.github/workflows/*.yml`、`AGENTS.md` が名指しする Python tool。ADR(`docs/adr/`)と PRD(`docs/memo/` の直下・`done`・`frozen`・`dropped`)は読む条件が一律(ファイル名で絞る / 状態フォルダで表す)なので**フォルダ単位 1 行**に件数を添える。列 = 種別 / パス / 読む条件 / 誰が読む(人間・Claude・Codex)/ 参照元(どこからリンク・名指しされているか。**空 = 死んだ動線**)/ 根拠。
3. **表 2 作業 → 読む文書**。列 = これから何をするか / 読むもの(文書と章)/ 読まなくてよいとき / 根拠。workflow 図はラベルしか持てないので、条件文はこの表が持つ。行は `AGENTS.md` が定める読む条件と skill の手順から起こし、PRD ライフサイクルの工程(起票・再グリル・実装計画・検証)も行にする。
4. **表 3 AGENTS.md の節の配置**。`AGENTS.md` の全 `##` 節について、列 = 節名 / 行数 / 分類 / 移設先候補 / 根拠。分類は「常時必要(入口に残す)」「条件付き(特定の作業でしか要らない。移設先候補を書く)」「どの作業からも到達しない」の 3 値。末尾に集計行(条件付きの節数と行数合計)を置く。これが入口の削減候補の測定器になる。
5. **表 4 skill 在庫**。project の skill をファイル単位で全列挙し、列 = 名前 / 所在(project・user-level・両方)/ 実体コピーの同期状態(`.agents` と `.claude` が一致 / どちらが新しいか)/ shadow(user-level に同名があるか。Claude 側・Codex 側)/ Codex 表示名(`agents/openai.yaml`)の有無 / 根拠。user-level にしかない skill は本数と集合の差だけ書き、名前の列挙は個人版に置く。
6. **発見事項**。次の分類で名指しする。修正はしない。
   - shadow(エージェント別に本数と中身違いの本数)
   - 実体コピーのドリフト(skill 名と、どちらの側が新しいか)
   - 欠落(例: テスト章の無い設計文書、章題や章番号のゆれ、実運用されているが規約文書に定義の無い工程)
   - 死んだ動線(参照元の無いノード)
   - 機械化点(実体コピーの同期検査、CI のパス条件化など。将来の候補として書くだけ)
   - evidence gap(読めなかったもの、実証できなかった関係)

## architecture 図

- `meta.quality_profile: "showcase"`。主要ノードは 3 個以上 12 個以下。
- ノード候補: `CLAUDE.md`、`AGENTS.md`、`docs/agents`、設計文書(1xx・2xx)、ユーザーガイド(0xx)、`CONTEXT.md`、ADR、PRD(`docs/memo`)、testdata README、skill(実体コピー 2 系統を 1 ノードにしカードで説明)、CI workflows、ユーザー環境(抽象境界ノード)。実在を確認できたものだけを置く。
- 境界は 2 種: リポジトリ内 / ユーザー環境、両エージェントが読むもの / 片方だけが読むもの。
- 主経路は 1 本: 入口 → 規約 → 領域の設計文書 → テスト章 → testdata README。
- 各ノードの `sources` は判定力の高いリポジトリ相対参照を最大 3 件。ユーザー環境ノードの根拠は、リポジトリ内で観測できる事実(skills README の実体コピー記述、`CLAUDE.md` の global 指示への言及)に限る。
- ユーザー環境ノードのカードには機構だけを書く: 同名 skill の shadow が両エージェントで起きうること、Claude Code だけが global `CLAUDE.md` と memory 層を持つこと。実体(どの skill が shadow か、memory の内容)は書かない。
- 機械化点(実体コピーの同期検査、CI のパス条件化)は該当ノードのカードに書く。実装はしない。

## workflow 図

- `meta.quality_profile: "showcase"`。lanes = 人間 / 起票側 / 精査側 / 実装側 / 共通文書 / CI。
  **レーンはエージェント名でなく役割名にする**(担当はトークン予算で入れ替わるため)。phases = 入口 → 設計 → 実装 → 検証 → 引き渡し。
- Archify の `workflow` は列を 0〜5 の 6 本しか持たず、同じレーンで隣り合う列も使えない(列ピッチ約 80px に対しノード幅 92px で、辺の最短 28px を満たせない)。
  主経路の工程が 6 を超えるときは**隣接する工程を 1 ノードへ畳んでよい**。畳んだ内訳は sublabel と README の表 2 に書き、工程そのものは落とさない。
- 主経路 = **PRD ライフサイクル**。正本は [docs/memo/README.md](../../../../docs/memo/README.md) の §工程と担当 で、図はそれを描く:
  入口読み → グリルして起票(起票側。`grill-with-docs` と `to-prd`)→ **再グリル**(起票側とは別の枠が精査する)→
  **調整**(人間。必要なら人間が仲介してエージェント間を往復させる)→ 実装計画(`implement-prd-with-docs`。**実装する枠**が作る)→
  編集前の条件読み(設計文書・ADR・`CONTEXT.md`)→ 編集 → 検証(設計文書のテスト章 → testdata README)→
  レビュー(起票側)→ **人間のレビュー**(コミット前)→ コミット(人間のみ。`prepare-commit` は文案まで)。
- **担当は固定しない。** どの枠がどの工程を持つかはトークン予算で変わる。図が描くのは「起票・精査・実装は別の枠に置ける」
  「実装計画は実装する枠が作る」「コミット前に人間がレビューする」という**工程の非対称**だけで、特定のエージェント名に結び付けない。
  正本が更新されたら図もそれに従う。正本と実測(PRD の改訂履歴行)が食い違う場合は、両方を発見事項に記録する。
- 側枝: Run 分析(030、`inspect_run.py`)/ 実験記録(`docs/experiments/README.md`)/ ビルド詰まり(040)/ 定常整備(`anet-housekeeping` → 各 skill)。
  側枝は主経路の最寄りノードから出す。引けない場合(Archify の交差制約)は意味が最も近い工程から出し、理由を記録する。
- **設定変更を側枝にしない。** 設定ファイルと 020 §3 のドキュメントは実装時に同時更新されるので、実装ノードへ畳む
  (sublabel に含める)。独立ノードにすると工程に見え、そこへ辺を引いた主体が「設定を変える人」と誤読される。
  2026-09-21 に実際にそうなった: `設定変更` を共通文書レーンの独立ノードにしたところ、`実装` からは col の逆行で辺を引けず
  (`return-left` も `channelX` も交差が解消しない)、`調整`(人間レーン)から引いた結果、人が設定を変更する図になった。
  読む条件(020 §3.6-3.7 と `check_default_leaves.py`)は README の表 2 が持つ。
- ユーザー環境の読み込み(global 指示・user-level skill・memory)はエージェント固有なので、抽象ステップとして 1 つ置く。どちらのエージェントがどのレーンに入るかは固定しないので、レーン名では区別しない。
- 条件文はラベルに詰めず README の表 2 に置く。

## 個人版

- 公開版の JSON を複製し、ユーザー環境境界の中身を実ノード化する: global `CLAUDE.md`、user-level skills(`~/.claude/skills`、`~/.agents/skills`、`~/.codex/skills`)と project 同名の shadow の一覧、memory 層(件数と index 行数)、Codex の user-level 入口(`~/.codex/AGENTS.md`)。
- workflow 図は、追加ノードが Archify の制約を満たす場合だけ足す。2026-09-21 の実測では、レーンを増やして「memory の読み込み」「user-level skill の解決」を置くと、そのレーンを縦断する主経路の辺と必ず交差し、showcase の交差 0 を満たせなかった(レーン順序・列位置・`route` / `channelX` / `fromSide` の組み合わせを一通り試した)。満たせない場合は公開版と同じノード構成にし、実体は `ユーザー環境` ノードの sublabel とカードへ畳んだうえで、理由を個人版 README の差分節に書く。
- README は公開版の 6 節に加え、冒頭に「公開版との差分」節(追加したノードとステップの一覧)、shadow 対象の diff 要約、memory の件数と種別、global `CLAUDE.md` の内容、Codex 側の user-level 状態、読めなかった対象の evidence gap を書く。
- 公開版のノード id をすべて含む。含まれない id があれば契約違反。

## 品質と公開判定

- 全図で `meta.quality_profile: "showcase"` を使用し、Archify の 9 artifact checks、composition errors 0、warnings 0 を満たす。
- `deliver` 成功後の SHA-256 は公開可否の一時的な作業記録として確認し、公開コンテンツには追加しない。
- 公開版の README は各行に根拠があり、表 3 に集計行があり、発見事項が 6 分類で書かれていること。
- README からの相対リンクに切れが無いこと(`docs/memo/README.md` のリンク検査ループを使う。Pages の表示 URL は検査対象外)。

## Evidence と不一致

根拠の優先順位は次のとおりとする。

1. 現行の `AGENTS.md`・`CLAUDE.md`・skill の `SKILL.md`・設定ファイル・CI 定義など、エージェントが実際に読む契約
2. 設計文書の索引と最終章、`CONTEXT.md`、ADR、ownership 資料
3. skills README、`docs/memo/README.md`、`docs/archify/README.md` などの運用資料
4. 過去の生成済み動線マップと実験記録

優先順位の異なる根拠が矛盾する場合、動線は上位の根拠に合わせる。資料の記述と実測(例: 「`diff -rq` で同期する」と書かれているが実際は不一致)は消さず、発見事項に双方を記録する。現行 checkout で確認できないノードや動線を、以前の生成物や既知の名称だけから補完しない。
