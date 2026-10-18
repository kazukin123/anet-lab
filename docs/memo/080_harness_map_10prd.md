# PRD 080: AI ハーネス動線マップ(harness_map)— 入口・条件読み・動線の可視化

> 起点: 2026-09-20。前セッションで AGENTS.md から PRD 固有の検証手順を testdata README へ追い出した際、
> 「AI エージェントがどの条件でどの文書を読むか」の地図が無いことが残課題になった。
> 本セッションで調べると、読む条件の記述が 4 箇所に散在し、skill の実体コピーが双方向にドリフトし、
> user-level の同名 skill が project 版を shadow していた(本セッションの `grill-with-docs` 自体が shadow 版で起動)。
> いずれも地図が無いため今日まで気づかれなかった。
> 本 PRD は **AI ハーネス全体の最適化の初期段階**として動線を可視化し、重複・欠落・死んだ動線を名指しするまでを範囲とする。
> 成果物は一度きりの図ではなく、**動線マップを再生成する skill `anet-harness-map`** と、その実行で生まれる動線マップである。
> 削減・移設・修正は次フェーズ。
> 改訂: 2026-09-21(skill 化。動線マップ本体を skill の出力に変え、実装計画ファイルを持たない形にした。
> 主経路へ「再グリル」「調整の往復」「実装計画は実装する枠が作る」「コミット前の人間レビュー」を追加。
> この運用は PRD の改訂履歴行にしか痕跡が無く規約文書に定義が無かったので、
> [docs/memo/README.md](README.md) に §工程と担当 を新設して正本にした)。
> 関連: [AGENTS.md](../../AGENTS.md)、[skills README](../../.agents/skills/README.md)、
> [anet-harness-map](../../.agents/skills/anet-harness-map/SKILL.md) と [その契約](../../.agents/skills/anet-harness-map/references/harness-map-contract.md)、
> [anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md)、
> [docs/design/README.jp.md](../design/README.jp.md)、[docs/archify/README.md](../archify/README.md)、
> [atlas 契約](../../.agents/skills/anet-archify-atlas/references/atlas-contract.md)、
> [pages.yml](../../.github/workflows/pages.yml)、[999_cicd_improvement](999_cicd_improvement_10prd.md)

## Context(背景・目的)

### 読む条件が 4 箇所に散在している

「いつ何を読む・呼ぶか」を定めている場所は現在 4 つあり、相互に参照していない。

| 場所 | 定めている内容 |
|---|---|
| `AGENTS.md` の 8 節 | ADR を読む条件 / CONTEXT.md を引く条件 / 作業ルールの編集前項目(設計文書) / 検証末尾の一般則(testdata README) / 設定値の扱い(020 §3) / Agent 系の所有権(`docs/ownership_guideline.md`) / Run 結果分析(030、`inspect_run.py`) / Agent skills(docs/agents 3 本) |
| `.agents/skills/README.md` | skill 一覧、起動記法(Codex `$name` / Claude `/name`)、出力先、`anet-housekeeping` → 各 skill の呼び出し対応、実体コピーの同期手順 |
| `docs/design/README.jp.md` §3 | カテゴリ → 設計文書の索引。各文書の §4 コードマップが文書 → ファイル、最終章「テストと拡張時の確認事項」が検証の着地点 |
| `docs/archify/README.md` | atlas 5 図の索引と doc/code drift の記録 |

人間がハーネスの全体像を把握するには 4 箇所を頭の中で合成するしかなく、合成した結果を置く場所も無い。

### 実測した実害(2026-09-20〜21)

- `AGENTS.md` は 720 行 / 56.5KB(約 2 万トークン)で、`CLAUDE.md` の指示により毎セッション全文が読まれる。
  PRD ごとに節が増える構造で、直前の整理(764 → 720 行)は PRD 固有ブロックの追い出しだけを行った。
- `.agents/skills`(Codex 用)と `.claude/skills`(Claude 用)は実体コピーで、README が `diff -rq` による同期を定めているが、
  3 本が**双方向に**ドリフトしていた: `anet-audit` と `grill-with-docs` は `.claude` 側が新、`anet-housekeeping` は `.agents` 側が新で
  「実行前にユーザーの了承を得る」への方針変更が `.claude` 側に届いていない。
- **shadow は両エージェントで起きている。** user-level の skill 層は Claude Code が `~/.claude/skills`(38 本)、Codex が `~/.agents/skills`(43 本。
  前者の上位集合で共通分は同一内容)。どちらにも project と同名の skill が 8 本あり、中身違いは Claude 側 6 本、Codex 側 5 本。
  本セッションの `/grill-with-docs` は user-level 版で起動し、project 版にしか無い `ADR-FORMAT.md` / `CONTEXT-FORMAT.md` が読まれなかった。
- 入口の非対称は user-level 側にある。Claude Code は global `CLAUDE.md` と memory 層(`~/.claude/projects/<slug>/memory/`)を持ち、
  Codex の user-level 入口 `~/.codex/AGENTS.md` は空ファイルで、`~/.codex/skills` は無関係な 1 本だけ。
  `archify` を含む汎用 skill は両エージェントの user-level にあり、2026-09-19 の atlas 更新は Codex が実行している。
- 設計文書の「テストと拡張時の確認事項」章は 1xx / 2xx にしか無く(0xx は利用者向け)、220 は欠落、160 は章題が「変更時」、
  章番号は 8 / 9 / 10 / 12 とばらつく。前セッションで合意した道筋「設計文書テスト章 → testdata README」は 1xx / 2xx でしか成立しない。
- docs 全体のリンク切れは 171 件(前セッション実測、行番号付きリンク等)。手書きの文書参照は放置すると腐る。

### 一度きりの図では足りない

ハーネスは PRD ごとに変わる(AGENTS.md の節、skill、testdata README、設計文書の章)。一度描いた図を手で保守すると、
上のリンク切れ 171 件と同じ経路で腐る。地図は「描く」ものではなく、ハーネスが変わるたびに**現行 checkout から再生成する**ものにする。
再生成のたびに shadow・ドリフト・死んだ動線の件数が出るので、ハーネス最適化の進み具合はその件数の推移で読める。

### 前セッションで合意済みの原則(本 PRD の前提)

- 道筋は「領域の設計文書のテスト章 → `testdata/prdNNN/README.md`」と「編集地点のコメント」で残す。
- AI が確実に守るのは編集地点コメント・失敗するテスト・CI のパス条件であり、**表や図は人間の理解と AGENTS.md 縮小のため**に作る。
  したがって動線マップは AI に読ませない。

## ゴール / 非ゴール

- **G1**: 人間がハーネスの全ノードと動線を辿れ、重複(shadow・実体コピーのドリフト)・欠落・死んだ動線を**名指しできる**。
- **G2**: 入口(常時読み)の削減候補が図と表から導ける。具体的には `AGENTS.md` 全節の動線上の位置が分かる。
- **G3**: 人が思い出さなくても地図が現行に追随する。ハーネスの構成ファイルが変わったら `anet-housekeeping` が再生成候補に挙げる。
- **NG1**: 図を「どの条件でどれを読むか」の正本にしない。正本は従来どおり `AGENTS.md` と各文書側に残す。
- **NG2**: AI に図を読ませない。`AGENTS.md` に参照行を足さない。
- **NG3**: 削減・移設・ドリフト修正・shadow 解消は次フェーズ。skill は発見と名指しまでで、対象を直さない。
- **NG4**: 図と表を吐くスクリプト生成器、実体コピーの同期検査スクリプト、CI パス条件化は作らない。再生成は skill(LLM)が現行 checkout から行い、機械化点は図上で名指しするだけ。
- **NG5**: 生成履歴の専用ファイルは持たない。README は最新状態だけを持ち、件数は housekeeping の `runs.jsonl` の `note` に残す。

## 用語

本 PRD と成果物で使う語。`CONTEXT.md` は RL ドメインの用語集なのでプロセス語彙は載せず、契約(`harness-map-contract.md`)と生成物の凡例に置く。

| 語 | 定義 | 避け語 |
|---|---|---|
| **AI ハーネス** | エージェント(Claude Code / Codex)の振る舞いを決めるファイル群の総体。指示・規約・設計文書・PRD/ADR/用語集・検証資材・skill・memory・CI。コード本体は含まない | 環境、設定 |
| **入口** | 条件なしに毎セッション読まれるファイル。`CLAUDE.md`、`AGENTS.md`、global `CLAUDE.md`、memory の index | 常時読み、エントリ |
| **条件読み** | 作業種別や領域で読むかどうかが決まる読み。入口以外の読みはすべてこれ | 参照、オンデマンド |
| **動線** | 入口から末端(文書の章・skill・ツール)まで、エージェントが辿る参照の連なり。参照元の無い末端は「死んだ動線」 | 流路、経路、フロー |
| **shadow** | 別スコープ(user-level)に同名の skill があり、project 版の代わりに起動しうる状態。Claude Code は `~/.claude/skills`、Codex は `~/.agents/skills` が user-level | 上書き、重複 |
| **実体コピー** | 同じスコープ内で意図して同内容を 2 実体として置いている状態(`.agents/skills` と `.claude/skills`)。同期は手作業 | 重複、ミラー |

「重複」を避け語にするのは、shadow は消すべき事故、実体コピーは意図した構造で同期だけが問題、と対処が正反対だからである。
成果物の総称は「動線マップ」、図の basename は `harness_map_*`。

## 確定事項

| 論点 | 決定 |
|---|---|
| 成果物 | skill `anet-harness-map`(`.agents/skills` と `.claude/skills` の実体コピー、`SKILL.md` + `references/harness-map-contract.md` + `agents/openai.yaml`)、`anet-housekeeping` の候補登録、skills README の表更新。動線マップ本体(公開 5 ファイル + 個人 3 ファイル)は skill の実行結果 |
| 出力仕様の正本 | `harness-map-contract.md`。本 PRD は判断の記録で、6 節・2 図・個人版の細則は契約に置く |
| 公開 / 個人の切り方 | **機構 = 公開、実体 = 個人**。`docs/` は GitHub Pages に配信されるため、個人環境の実体(どの skill が shadow か、memory の内容)は置かない。公開図には「ユーザー環境」抽象境界ノードを 1 個置き、カードに機構だけ書く(同名 skill の shadow が両エージェントで起きうる / Claude だけ global CLAUDE.md と memory 層を持つ) |
| 図の型 | archify `architecture`(俯瞰: ノード・境界・主経路)と `workflow`(動線: 主体レーン × 作業段階)。俯瞰と動線は読み方が違うので 2 型 |
| 公開版の置き場 | `docs/agents/` に **`README.md`**(動線マップ本文。`docs/archify/README.md` と同じ形式で図の索引を兼ね、`docs/agents/` 自体の索引にもなる)、`harness_map_architecture.archify.json` / `.html`、`harness_map_workflow.archify.json` / `.html`。Pages の index は `docs/archify/*.html` しか自動掲載しないので、README の索引表に表示 URL(`https://kazukin123.github.io/anet-lab/docs/agents/<basename>.html`)を持つ |
| 個人版の置き場 | `.scratch/harness-map/`(gitignored、既存の作業場)に同じ basename。**Claude Code でも Codex でも同一 run で公開版と個人版の両方を生成する**(user-level のファイルはパスで読める) |
| 整合 | **再生成**。公開版と個人版は同じ run で現行 checkout から作り直す。個人版は公開版のノード id をすべて含み、skill が `diff` で確認する。手保守はしない |
| 履歴 | README は最新状態 + 生成根拠の revision だけ(atlas と同じ)。件数(死んだ動線 / ドリフト / shadow / 移設候補)は skill の完了報告に出し、housekeeping 経由なら `runs.jsonl` の `note` に残る |
| housekeeping | 候補名 `harness-map`。前提 = `archify` があり、`docs/agents/README.md` が無いか、記録された revision 以降にハーネス構成ファイル(`AGENTS.md`、`CLAUDE.md`、`.agents/skills`、`.claude/skills`、`docs/agents`、`docs/design`、`core/anet-core/testdata`、`.github/workflows`)に変更がある。優先順は survey の後、atlas の前。初期見積 15% / 20 分 |
| 触らないもの | `AGENTS.md`(PRD 079 が別途編集予定)、atlas 契約(`docs/archify/` は固定 5 名のみ)、`pages.yml`、`CONTEXT.md`、`docs/design/`、`anet-housekeeping` 両コピーの既存差分(追加行だけ入れる) |
| 担当 | skill と契約の作成は Claude。動線マップの生成は skill を実行するエージェント(Claude Code / Codex どちらでも。archify は両方の user-level にある) |
| ADR | 起票しない。公開 / 個人分離も skill 化も可逆で、ADR の条件(戻しにくい)を満たさない |

## 実装契約

### skill `anet-harness-map`

`anet-archify-atlas` を雛形にする。

- 起動: `$anet-harness-map` / `/anet-harness-map` の明示起動、または `anet-housekeeping` の候補 `harness-map`。引数なし。`archify` skill が無ければ atlas と同文のエラーで終了。Plan Mode では生成しない。
- ワークフロー: root 特定 → 契約を全文読む → revision 取得と調査(入口、リポジトリ内文書、skill の 2 系統と user-level 3 箇所、ユーザー環境、参照元の `rg` 実測)→ 公開 README の 6 節 → archify 2 図の候補・validate(showcase 9/9、errors 0、warnings 0)・deliver・visual-check → 公開(固定名 5 ファイルだけ置換、README を最後に)→ 個人版 3 ファイル(id 包含を `diff` で確認)→ 片付け。
- 停止条件・禁止・完了報告は `SKILL.md` に書く。完了報告の件数の定義は契約に置く。
- `agents/openai.yaml` で Codex の表示名を与え、暗黙起動を禁止する。

### 契約 `harness-map-contract.md`

用語 6 語、公開・個人のマニフェスト、README の 6 節(凡例と図の索引 / 表 1 ノード在庫 = ファイル単位全列挙、ADR と PRD はフォルダ 1 行 / 表 2 作業 → 読む文書 / 表 3 AGENTS.md の節の配置と集計行 / 表 4 skill 在庫 / 発見事項 6 分類、各行に根拠必須)、architecture 図と workflow 図の契約(主要ノード ≤12、境界 2 種、主経路、lanes と phases、主経路 = PRD ライフサイクル(別モデルでの再グリルと、実装計画を実装する側が作る工程の非対称を含む。担当そのものは固定しない)、側枝)、個人版の契約(workflow への追加ノードは Archify の交差制約を満たすときだけ)、品質と公開判定、evidence の優先順位。

### `docs/memo/README.md` の §工程と担当

PRD ライフサイクルの工程(起票 → 再グリル → 調整 → 実装計画 → 実装・検証 → レビュー → コミット)と、
「担当は固定しないが工程の非対称は固定する」という原則の**正本**を置く。動線図と README の表 2 はこの節を根拠にする。
`AGENTS.md` へは書かない(入口を太らせない。NG2)。

### `anet-housekeeping` と skills README

- 両コピーの候補表に `harness-map` 行を足し、既定の優先順に `harness-map` を atlas の前に入れる。`runs.jsonl` の `note` に件数を書く旨を 1 行足す。両コピーの既存差分には触れない。
- skills README(両コピー)のプロジェクト固有の表に `anet-harness-map` を足し、housekeeping の対応に `harness-map → anet-harness-map` を足す。出力先の約束を「公開が目的のもの(英訳・atlas・動線マップ)は `docs/` 配下」に改める(atlas が例外として書かれていなかった)。

### 初回生成と以後の再生成

**初回の動線マップは `anet-harness-map` の初回実行で作る。別途の実装計画(`*_2ximpl.md`)は持たない。**
skill 化の前に手で生成した版があったが、初回実行で置き換えた。
以後の再生成は `$anet-harness-map` の明示起動、または `anet-housekeeping` の変更検知から skill で行う。
生成物と契約の差(用語の細部、表の列)は、その回の実行で契約側に揃える。

## テスト / 検証

- skill: `diff -rq .agents/skills/anet-harness-map .claude/skills/anet-harness-map` が空。frontmatter の `name` / `description` があり、`description` に起動条件を含む。
- housekeeping: 両コピーに同じ `harness-map` 行と優先順があり、`$anet-housekeeping S dry-run` で `harness-map` が候補に出る(README 未生成なので前提が通る)。
- 生成物(初回実行後): 各図の `validate --quality showcase --json` が 9/9・errors 0・warnings 0、`deliver` 成功。公開 README の 6 節、表 3 の集計行、発見事項 6 分類、各行の根拠。`docs/memo/README.md` のリンク検査ループを `docs/agents/README.md` に当てて切れ 0(Pages の表示 URL は対象外)。個人版 3 ファイルが `.scratch/harness-map/` にあり、公開版のノード id をすべて含み、`git status` に出ない。

## Complexity audit

グリルの最終簡素化パスの裁定(2026-09-20)と skill 化での改訂(2026-09-21)。

| # | 機構 | 裁定 | 理由 / 切ったら戻る痛み |
|---|---|---|---|
| 1 | 公開 architecture 図 | keep | 4 箇所散在・全体俯瞰なし |
| 2 | 公開 workflow 図 | keep | 俯瞰と動線は読み方が違う(用途 pin。痛みではない) |
| 3 | 個人 architecture 図 | keep | shadow・memory 層の実体が図に出ない。可逆(不要なら統合) |
| 4 | 個人 workflow 図 | keep、**統合候補 No.1** | 差分 2 ノード。公開図 + 個人 README の差分節で読めるが、構成差があるうちは分ける(仮説として実施)。判断は 2 回目以降の生成時 |
| 5 | 公開 README 6 節 + 根拠パス | keep | G1/G2 が測れない。リンク切れ 171 件が示すとおり根拠無しの参照は腐る |
| 6 | 個人 README | keep | 公開できない実体の置き場が無い |
| 7 | 「ユーザー環境」抽象境界ノード | keep | 両エージェントの user-level 層と shadow 機構が公開図から消える |
| 8 | 公開版と個人版の id 包含 | keep | skill が `diff` で検証する。手保守の整合規約は skill 化で不要になった |
| 9 | PRD 起票 | keep | 本表の置き場 |
| 10 | 凡例 6 語 | keep | shadow と実体コピーが「重複」に混ざる |
| 11 | 図と表を吐くスクリプト生成器 | cut(NG4) | 再生成は skill が現行 checkout から行う。archify JSON は座標を持つのでスクリプト生成しても手調整が残る |
| 12 | 実体コピーの同期検査スクリプト | defer(NG3) | 表 4 が再生成のたびにドリフトを検出する。修正は次フェーズ、図で機械化点として名指し |
| 13 | ファイル → 設計文書の逆引き生成 | cut | `anet-audit` の drift レンズが同じ守備範囲 |
| 14 | CI パス条件化 | defer | `999_cicd_improvement` の範囲 |
| 15 | atlas 契約改訂(6 枚目) | cut | 主題が違う(atlas はコード構造、本件はハーネス構造)。単体の skill で足りる |
| 16 | `AGENTS.md` への参照行 | cut(NG2) | AI に読ませない |
| 17 | ADR / `CONTEXT.md` 用語 | cut | 可逆な分離に ADR は不要。RL 用語集にプロセス語彙を混ぜない |
| 18 | skill 化 | keep | ハーネスは PRD ごとに変わり、手保守の図は腐る(実害: リンク切れ 171 件と同じ経路) |
| 19 | housekeeping 候補登録(変更検知) | keep | 人が思い出さないと再生成されない。変更が無いときに走らせない前提で無駄も出ない |
| 20 | 生成履歴の専用ファイル | cut(NG5) | `runs.jsonl` の `note` で件数の推移は読める |
| 21 | 個人版を Codex で skip する分岐 | cut | user-level のファイルはパスで読めるので分岐が要らない |

- 要件の実在性: 実在 5(公開 / 個人分離 = Pages 配信、根拠パス = リンク切れ 171 件、表 3 = AGENTS.md の増加履歴、発見事項 = shadow・ドリフトが今日まで不可視、skill 化 = 手保守の図は腐る)、
  用途 1(2 型の図)、仮説 1(個人版にも図)。降格なし。
- 決定の残滓: `AGENTS.md` の読む条件 4 節統合、how-to の全面移設、Git 規約の別文書化、220 テスト章追加と章番号統一は**次フェーズ**。220・160 は発見事項に載る。
  起票時の「archify は Claude 側にしか無い」「Codex にレビュー動線が無い」「Codex は project だけを読む」は user-level `~/.agents/skills` の見落としで、skill 化の調査で訂正した。
- 最小解との差分: 最小解は公開 architecture 図 + 公開 README 3 節(凡例・表 3・発見事項)+ 個人 README を一度だけ作ること。差分(workflow 図、個人図、表 1・2・4、skill 化、housekeeping 登録)は上表で再正当化済み。
  表 1・4 のファイル単位全列挙はユーザー裁定。

## 受入基準

1. `anet-harness-map` が `.agents/skills` と `.claude/skills` に同一内容で存在し、`SKILL.md`・`references/harness-map-contract.md`・`agents/openai.yaml` を持つ。
2. `anet-housekeeping` の両コピーに同じ `harness-map` 行があり、既定の優先順で atlas の前にある。`$anet-housekeeping S dry-run` で `harness-map` が候補に出る。
3. skills README の両コピーに `anet-harness-map` の行と housekeeping の対応がある。
4. 公開版 5 ファイル(`README.md`、2 図の `.archify.json` と `.html`)が `docs/agents/` にあり、archify showcase 検証を通っている。skill による再生成でも同じ 5 ファイルだけが置換される。
5. `README.md` が 6 節を持ち、図の索引表(表示 / html / json)と生成根拠の revision があり、表の各行に根拠(ファイルと行または節名)がある。
6. 発見事項の件数が実測の下限以上: 実体コピーのドリフト 3、shadow 8(中身違い Claude 6 / Codex 5)、テスト章欠落 1(220)、章題ゆれ 1(160)。
7. 表 3 の集計行に「条件付き」の節数と行数合計がある。
8. `README.md` からのリンク切れが 0 件(Pages の表示 URL は検査対象外)。
9. 個人版 3 ファイルが `.scratch/harness-map/` にあり、`README.md` 冒頭に「公開版との差分」節があり、図 JSON が公開版のノード id を全て含む。`git status` に出ない。
10. `AGENTS.md`、`CONTEXT.md`、`docs/design/`、`docs/archify/`、`pages.yml` に差分が無い。`anet-housekeeping` 両コピーの差分は追加行だけ。
11. `docs/memo/README.md` に §工程と担当 があり、動線図の主経路と README の表 2 がその節を根拠にしている。

## 影響・移行

- **他文書への影響**: skill の追加と、`anet-housekeeping`・skills README への追加行のみ。既存の規約・契約は変えない。skill と README の変更は同じコミットに載せる。
- **更新頻度**: housekeeping の変更検知に従う。ハーネス構成ファイルに変更が無ければ走らない。明示起動はいつでも可。
- **次フェーズの入力**: 次フェーズ(`AGENTS.md` 4 節統合と入口 → 条件読みの移設 / 実体コピーの同期検査 / shadow の解消 / 220・160 の修正 / CI)の PRD は、
  最新の公開 README の表 3 と発見事項の行を引用して起票する。効果は次回再生成の件数で確認する。
- **統合の判断点**: 個人 workflow 図の差分が 2 ノードのまま増えなければ、2 回目以降の生成で公開 workflow 図へ統合し、個人版は README の差分節だけにする(契約の改訂で行う)。
- **陳腐化**: 手保守しない。古くなった地図は再生成で置き換える。根拠パスは再生成のたびに実測から書き直される。
