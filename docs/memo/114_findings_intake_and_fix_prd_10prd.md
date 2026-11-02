# 直す課題の受け皿を台帳にし、F 番号を fix PRD で片付ける（工程末の起票の区分と `anet-create-fix-prd`）PRD

> 2026-11-02 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した。番号はユーザーの指示で連番）。数値と行番号は起票時点のもので、実装時は取り直す。同日、暫定 PRD には確定した決定を置かないという anet-create-prd の決まりに合わせ、起点の行と §4 の書き方を改めた（会話で出た方向も暫定の既定として扱う）。
> 起点: 2026-11-01〜11-02 の会話。ユーザーの問題提起は「anet-create-prd や implement-prd-with-docs の最後に起票対象を聞かれると、小さな修正も PRD として起票される。しかし、小さな修正では PRD を起票しない運用である」。案を比べる中で、会話では次の 5 つの方向が出た。(1) 入口の受け皿は台帳（F 番号）。(2) 出口は、台帳の F 番号を最大 5 件まとめて 1 本の PRD として起票する skill `anet-create-fix-prd`。(3) F 番号の選定以外は anet-create-prd の手順と定義を流用する。(4) F 番号の対応は基本この PRD に統一し、大きい件も外さない。(5) 入口と出口の両方を整える全体の PRD を連番で起こす。本書は暫定の起票なので、これらも決定ではなく、再グリルですべて見直す。§4 で「起点の方向（n）」と書くのは、この番号を指す。
> 関連: [プロセスレジストリ](../process/registry.md)、[プロセス契約](../process/protocol.md) §7、[ADR 0049](../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)、[anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md)、[implement-prd-with-docs の手順](../../.agents/skills/implement-prd-with-docs/PRD-IMPLEMENTATION.md)、[anet-audit](../../.agents/skills/anet-audit/SKILL.md)、[anet-improve](../../.agents/skills/anet-improve/SKILL.md)、[台帳の CLI](../../reports/tools/registry.py)、[skills README](../../.agents/skills/README.md)、[memo README](README.md)、[PRD095](done/095_llm_wiki_10prd.md)（台帳の派生区画）、[311 開発ツール（wiki）](../wiki/categories/311_dev_tools.md)。

## 1. 目的と問題

工程の最後に出す起票候補で、直す課題を台帳（`reports/registry/findings.jsonl`）へ積めるようにする。台帳に溜まった F 番号は、新しい skill `anet-create-fix-prd` で最大 5 件ずつまとめて 1 本の PRD にし、通常の PRD と同じ流れ（実装・レビュー・`done/`）で片付ける。

解く問題は次の 3 つ（事実は §2）。

1. 工程の最後の起票候補で「新規」を選ぶと、anet の置き場は 999 の暫定 PRD しかない。小さな修正も 1 件 1 本の PRD になり、「小さな修正では PRD を起票しない」運用と合わない。
2. 台帳は anet-audit と anet-improve が書いているが、工程の最後の起票候補からは使われていない。片付ける入口も無い。open は 34 件で、PRD に上げた（`promoted`）件は 0 である。
3. 同じ種類の課題が、見つけた工程によって 999 PRD と台帳に分かれる。production の `dynamic_cast` を例にとると、PRD097 の工程末に見つかった件は 999 PRD になり、監査で見つかった件は台帳の F-0020 になった。

### 1.1 用語

本書の語はハーネス（エージェントの作業環境）の語で、フレームワークの語ではないので、[CONTEXT.md](../../CONTEXT.md) には足さない。既存の語で言い換えられないものだけを次のとおり使う。

| 語 | 意味 |
|---|---|
| 台帳 | `reports/registry/findings.jsonl`。1 行 1 課題で、F 番号・状態・履歴を持つ。CLI は [registry.py](../../reports/tools/registry.py)（anet-audit・anet-improve の skill が使っている語） |
| F 番号 | 台帳の課題の番号（`F-0001` の形）。`registry.py add` が振る |
| 直す課題 | 今ある契約・規約・設計文書・テストに合わせて直す課題。不具合、規約違反、文書とコードのずれ、テストの不足、構造の悪さ、工程の不備。台帳の観点（lens）の `defect`・`refactor`・`architecture`・`drift`・`test-gap`・`backlog`・`process` で表せる |
| 新しい振る舞いの案 | 今の契約を足したり変えたりする案。新しい機能・機構・設定キー・指標 |
| fix PRD | `anet-create-fix-prd` で起こす PRD。台帳の F 番号 1〜5 件を題材にする。成果物としては通常の PRD（成果物キー `prd`）と同じである |
| 工程の最後の起票候補 | [プロセス契約](../process/protocol.md) §7 の、工程の完了判定で未解決のものを候補に挙げて開発者が選ぶ手順。anet では anet-create-prd の段 7 と implement-prd-with-docs の工程末で行う |

## 2. 現状（2026-11-02 に確認したこと）

1. **置き場の読み替え。** [レジストリ](../process/registry.md)の成果物 `issue-record`（未決の受け皿）の出力先は「`docs/memo/999_{slug}_10prd.md`、PRD の未決表、`/// @todo`」である（15 行目）。読み替え表の「課題記録への起票」の行も同じ 3 つを挙げる（81 行目）。台帳は入っていない。
2. **共有コアの名前。** [プロセス契約](../process/protocol.md) §7 は、起票に `an-create-issue` SKILL を、新規か既存への追記かの判定にその「分け方」を、起票済みの一覧に `an-list-issues` SKILL を使うと書く（497・521・567 行目）。どちらの skill もこの repo（`.agents/skills`、`.claude/skills`）にもユーザーの設定にも無い。読み替え表にあるのは起票の置き場の行だけで、「分け方」と一覧の読み替えは無い。同節は「1 課題 1 ファイルとし、ブランチ間でのコンフリクトを避ける」とも書く（487 行目）。
3. **共有コアは書き換えない。** [ADR 0049](../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md) は、プロセス契約をプロジェクトに依存しない文面に保ち、anet 固有のことはレジストリと観点に置くと決めている。同じ契約をほかのプロジェクトでも使っているためである。
4. **999 PRD。** `docs/memo/` 直下の `999_*_10prd.md` は 40 本ある。短いものは 29〜43 行で、1 か所の直しを題材にしている（例: [999_metrics_action_info_dynamic_cast](done/999_metrics_action_info_dynamic_cast_10prd.md) は 33 行で、production の `dynamic_cast` 1 か所を直す題材。PRD097 の工程末の候補②から起票した）。
5. **台帳。** 37 件ある。状態は open 34・wontfix 2・fixed 1・promoted 0。観点は refactor 13・defect 13・drift 6・test-gap 3・process 2。open のうち 30 件が機能カテゴリ（`reports/tools/categories.json`）に当たる。多いのは 318_rl_core 10・321_agent_core 5・312_util 4・120_metrics_viewer 4・317_metrics 3 である。残る 4 件はどの機能カテゴリにも当たらない。台帳を書くのは anet-audit（観点ごとの監査）と anet-improve（会話の手戻りのうち採らなかった問題と PRD 送り）である。`registry.py build` は、派生一覧（`findings.md`・`findings.html`）と wiki の地図の「台帳の指摘」の区画を作り直す。
6. **F 番号はすでに参照されている。** 実装記録は、見つけた既存の課題を台帳の F 番号で参照し、新しく起票していない（[110_20impl](done/110_runner_main_pane_dynamic_centre_20impl.md) の 140 行目・187 行目、[110_22impl](done/110_runner_main_pane_dynamic_centre_22impl.md) の 87 行目。どれも F-0020）。
7. **台帳の CLI。** [registry.py](../../reports/tools/registry.py) の `add` は、F 番号を台帳の最大の番号＋1 で振る（119 行目）。同じ課題かどうかは `fingerprint`（観点・ファイル・シンボルの組）で判定し、同じなら新しく積まずに最終確認日を更新する。`validate` は fingerprint の重複を検査するが、F 番号の重複は検査しない（129 行目からの `cmd_validate`）。台帳の CLI のテストは [PRD095 の検証資材](../../core/anet-core/testdata/prd095/README.md)の `registry_test.py` にあり、一時ディレクトリの台帳で確かめる。状態は `open`・`fixed`・`wontfix`・`stale`・`promoted` の 5 つである。
8. **PRD に上げたときの閉じ方。** [anet-audit](../../.agents/skills/anet-audit/SKILL.md) の禁止の節は「人が PRD に昇格させたら `set --status promoted --note "<PRD 番号>"` で閉じる」と書く（94 行目）。PRD に上げる手順を持つ skill は無い。
9. **範囲外の送り先。** [anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md) の論点の仕分けは、⑥ 範囲外を「999 の暫定 PRD か未決表へ送る」とする（83 行目）。[implement-prd-with-docs の手順](../../.agents/skills/implement-prd-with-docs/PRD-IMPLEMENTATION.md)は、工程末の起票候補を「プロセス契約 7 節に従って提示する」とだけ書き（203 行目）、置き場を名指さない。
10. **skill の置き場。** [skills README](../../.agents/skills/README.md) は、`.agents/skills`（Codex）と `.claude/skills`（Claude Code）に同じ中身を実体コピーで置き、`diff -rq` で差が無いことを確かめると決めている。
11. **採番。** [memo README](README.md) は、新しい下書きを 999 に置き、実装に着手するときに正式な番号へ振り直すと決めている。本書はユーザーの指示で起票時から連番にした。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1（入口）: 工程の最後の起票候補で、直す課題を台帳へ積める。新しい振る舞いの案だけが 999 PRD になる。
- G2（出口）: 台帳の F 番号を最大 5 件まとめて、1 本の PRD として起票できる。F 番号の選定以外は anet-create-prd の手順と定義をそのまま使う。起票したあとは、通常の PRD と同じ流れ（implement-prd-with-docs での実装、レビュー、`done/` への移動）で片付く。
- G3（一本化）: 台帳の F 番号を PRD にするのは、基本 fix PRD だけにする。設計の判断が要る大きい件も外さず、その判断はグリルで詰める。

### 3.2 非ゴール

- プロセス契約（`docs/process/protocol.md`）を書き換えること。退ける案は「§7 に台帳の区分を書き足す」案である。ADR 0049 の判断（ほかのプロジェクトの版と直しを持ち寄る）による本当の制約で、anet 固有の決まりはレジストリの読み替えで足りる（D3）。
- 台帳の項目と状態の種類を変えること。退ける案は「fix PRD 用の状態（例: `in-prd`）を足す」案である。本当の制約ではない。今ある `promoted` と注記で足り（D12）、変えると anet-audit・anet-improve・`build` の派生一覧まで波及する。
- 今ある 999 PRD のうち直す課題に当たるものを台帳へ移すこと。退ける案は「本書の実装で 999 PRD を棚卸しして台帳へ移す」案である。本当の制約ではない。入口と出口が動いてから決めればよく、先送りする（§8、§9 の 7）。
- anet-housekeeping の候補にすること。退ける案は「定常整備で fix PRD を自動で起票する」案である。本当の制約である。housekeeping は了承のあと質問せずに回るが、fix PRD の起票はグリルの質疑を挟む。
- GitHub Issue を課題の置き場にすること。退ける案は「Topic の Issue（例: 040_不具合対応）にチェックリストで積む」案である。本当の制約である。エージェントが repo の中を検索して読む流れの外にあり、投稿のたびに外部への書き込みになる。

### 3.3 足す機構とゴールへの寄与

| 機構 | 寄与するゴール |
|---|---|
| レジストリの読み替え（`issue-record` の出力先、読み替え表の 4 行、D1〜D3） | G1 |
| レジストリの新しい節「台帳への記録」（D4） | G1（工程を問わず同じ手順で積む） |
| skill `anet-create-fix-prd`（D5〜D14） | G2・G3 |
| `registry.py validate` の F 番号の重複検査（D15） | G2（fix PRD は F 番号で台帳を指すので、番号の取り違えを防ぐ） |
| anet-create-prd の ⑥、anet-audit の閉じ方、anet-improve の報告の書き換え（§5.5） | G1・G3 |

### 3.4 成功の測り方

- 本書の実装のあと、最初の 5 回の工程の最後の起票候補（anet-create-prd と implement-prd-with-docs を合わせて数える）で、次を数える。
  - 候補のうち台帳へ積んだ件数と、999 PRD にした件数。
  - 999 PRD にした件のうち、直す課題に当たるもの。0 件であれば G1 を満たす。
- fix PRD を 1 本起票した時点で、次を満たせば G2 を満たす。
  - レビューが OK になる。
  - 題材にした F 番号がすべて `promoted`（注記に PRD のファイル名）になる。
  - `registry.py validate` が問題 0 件で終わる。
- 台帳の推移を `findings.md` の状態の件数で見る。起票時の基準値は open 34・promoted 0 である。

## 4. 確定事項

本節の各行はすべて暫定の既定で、決定ではない（暫定の起票）。再グリルでは、各行を anet-create-prd の「論点の仕分け」に通して見直し、§9 とあわせて決める。理由に「起点の方向（n）」とあるのは、冒頭の引用行の n 番の方向に沿って置いた既定であることを示す。

| # | 決定 | 理由 |
|---|---|---|
| D1 | anet の課題記録（`issue-record`）の出力先に台帳を足す。出力先は、台帳、999 の暫定 PRD、PRD の未決表、`/// @todo` の 4 つになる | 台帳を受け皿にするのは起点の方向（1）に沿う。台帳は F 番号・状態・履歴・同じ課題の判定（fingerprint）・派生一覧・wiki の区画をすでに持つ（§2 の 5・7）ので、新しい置き場を作らずに済む |
| D2 | 起票候補の区分は「台帳へ新規」「999 PRD を新規」「既存へ追記（F 番号か 999 PRD）」「見送り推奨」の 4 つにする。直す課題は台帳へ、新しい振る舞いの案は 999 PRD へ入れる。大きさでは分けない | 大きさの境目は人と場面で揺れる。直す課題かどうかは台帳の観点（lens）で決まる（§1.1）。大きい直しも fix PRD のグリルで扱う（G3）ので、大きさで 999 PRD へ逃がす必要が無い |
| D3 | 入口の決まりは、プロセス契約を変えず、レジストリの `issue-record` の行と読み替え表に置く。読み替え表では 4 つの記述を読み替える。起票の置き場、`an-create-issue` の「分け方」、`an-list-issues`、1 課題 1 ファイル（§5.1） | ADR 0049 による（§2 の 3）。共有コアが名指す 2 つの skill は anet に無く、読み替えが無いと、分け方と一覧の見方が決まらない（§2 の 2） |
| D4 | 台帳へ積む手順は、レジストリの新しい節「台帳への記録」を正本にする。anet-create-prd と implement-prd-with-docs は、レジストリの読み替えを通してこれに従う。入口専用の skill（`an-create-issue` に当たるもの）は作らない | 積む操作は `registry.py` の `add`・`touch`・`set` だけで、anet-audit と anet-improve も CLI を直接使っている（§2 の 5）。手順を 1 か所に置けば、2 つの工程 skill に写さずに済む |
| D5 | 出口として skill `anet-create-fix-prd` を足す。`.agents/skills` と `.claude/skills` の両方に置く | skill を足すのは起点の方向（2）に沿う。両方に置くのは skills README の約束による（§2 の 10） |
| D6 | 1 本の fix PRD に入れる F 番号は 1〜5 件とする。6 件以上を指定されたら、PRD を書かずに止まり、選び直しを求める | 最大 5 件は起点の方向（2）に沿う。6 件以上で止まるのは、上限を黙って切り詰めると、どの件が外れたかが残らないためである |
| D7 | 題材にできるのは、状態が `open` か `stale` の F 番号だけとする。`fixed`・`wontfix`・`promoted` を指定されたら、PRD を書かずに止まり、その件の状態と履歴の最後の注記を示す。引数が無いか絞り込み（機能カテゴリの key かパス）だけのときは、台帳から `open`・`stale` の件を引き、同じ機能カテゴリ（当たらない件はファイルの親ディレクトリ）の中で 5 件ずつ区切った組の候補を示して、ユーザーが選ぶ（§5.3 の手順 1）。引数の扱いは §5.3 の表を上から当てて決め、外れた引数では止まる | `promoted` は別の PRD に入っている（D12）。同じ機能カテゴリの件は、実装で読むコードと回すテストが重なる。引数を黙って読み替えると、意図と違う件で PRD を起こしうる |
| D8 | 選んだ件の根拠（`evidence` の位置）を今のコードと文書で読み直す。根拠が消えて直っていれば、`set --status fixed --note "<確かめた根拠>"` で閉じ、組から外して報告する | 存在しない課題の PRD を作らない。[anet-audit](../../.agents/skills/anet-audit/SKILL.md) の手順 4（既存レコードの再検証）と同じ扱いである |
| D9 | 根拠を確かめたあとは、anet-create-prd の段 1〜7 に従う。読み替えるのは 3 つだけにする。(a) 題材は選んだ F 番号の組とし、段 1 の要点提示は件ごとに台帳の要点（題・`summary`・`scenario`・`evidence`・`fix_hint`）と今のコードの確認を示す。(b) goal anchor のゴールに「選んだ F 番号がすべて直る」を含める。(c) 段 7 で台帳を更新する（D12）。「暫定」の指示で質疑を省く扱いも、anet-create-prd と同じにする | anet-create-prd に従うのは起点の方向（3）に沿う。読み替えを 3 つに絞るのは、読み替えが少ないほど、anet-create-prd を直したときに fix PRD にもそのまま効くためである |
| D10 | 大きい件も組から外さない。契約に効く論点は、通常の PRD と同じく 1 問ずつ詰める。組から外すのは、グリルでユーザーが選んだときだけとし、外した件は台帳で `open` のまま残す | 大きい件を外さないのは起点の方向（4）に沿う。外す道を AI の判断で作ると、外した件のはけ口が無くなる |
| D11 | fix PRD は成果物としては通常の PRD と同じに扱う。レジストリの行（`prd`）、品質確認観点（`docs/quality/prd.md`）、採番、フォルダでの状態は共通にする。fix PRD 専用の観点とテンプレートは作らない。`anet-create-fix-prd` の SKILL.md は、選定（D6〜D8）、読み替え（D9）、台帳の更新（D12）、ファイル名（D13）、本文の形（D14）だけを書き、anet-create-prd の手順と規律は写さずに参照する | 定義を流用するのは起点の方向（3）に沿う。その読み方（成果物の定義と skill の定義の 2 つ）は §9 の 9 に残した。写すと 2 つの skill の手順が別々に古くなる |
| D12 | 台帳の更新は段 7 で行う。レビューが OK になり、起票候補を記録したあと、題材の各件を `set --status promoted --note "<fix PRD のファイル名>"` にし、`build` と `validate` を回す。fix PRD を implement-prd-with-docs で実装するときは、題材の件の状態を変えない（`promoted` のまま）。fix PRD が終わったことは、PRD の `done/` への移動が表す。その実装の工程末で新しく見つかった課題を台帳へ積むのは D4 のとおりで、これとは別である | anet-audit の既存の決まり（PRD に上げたら `promoted`、§2 の 8）に揃える。実装の完了を台帳にも書くと、状態をフォルダと台帳の 2 か所で持つことになる（memo README の「状態の二重管理はしない」） |
| D13 | fix PRD のファイル名は `{NNN}_fix_{範囲}_{f番号}_10prd.md` とする。`{NNN}` は memo README の採番に従う（anet-create-prd と同じ）。`{範囲}` は、組の件の選び方（候補から選んだか、F 番号の指定か）によらず、組の件だけから次の順で決める。全件が同じ機能カテゴリなら、その名前（key の番号を除いた部分。例 `rl_core`）。そうでなく、全件がどの機能カテゴリにも当たらず、ファイルの親ディレクトリが同じなら、そのディレクトリの最後の名前。ただし、その親ディレクトリがリポジトリの根（全件が `AGENTS.md` のような直下のファイル）なら `root`。どちらでもなければ `mixed`。`{f番号}` は組の中で最も小さい F 番号を小文字で書く（例 `f0020`） | 一覧で fix PRD と分かる。F 番号は 1 本の fix PRD にしか入らない（D7・D12）ので、最も小さい F 番号を付ければ名前が重ならない |
| D14 | fix PRD の本文には、件ごとの節を置く。各節には、F 番号、現象、根拠（台帳の `evidence` と、今のコードで確かめた位置）、直し方、件ごとの受入を書く。受入は件ごとに、外から観測できる形で書く（PRD 品質確認観点の項番 14） | 件ごとに直ったかを確かめられるようにする。組の一部が未決で残っても、ほかの件の受入は決まる |
| D15 | `registry.py validate` に F 番号の重複の検査を足す。重複があれば、その F 番号の組を問題として出し、終了コード 1 で終える | fix PRD と実装記録は F 番号で台帳を指す（§2 の 6）。2 つのブランチで同時に `add` すると同じ番号が振られうるが、今の `validate` は気づかない（§2 の 7） |
| D16 | 名前は、skill を `anet-create-fix-prd`、成果物の呼び名を「fix PRD」とする | skill の名前は起点の方向（2）に沿い、「fix PRD」は起点の会話で使われた呼び名に揃えた。skill は `anet-create-prd` に揃えた動詞＋目的語の形である |
| D17 | ADR は起こさない | skill とレジストリの読み替えは後から変えやすく、ADR を起こす 3 条件（[anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md) の Offer ADRs sparingly）のうち、戻しにくさを満たさない。共有コアを変えない判断は ADR 0049 がすでに記録している |

## 5. 実装契約

### 5.1 レジストリの変更（D1・D3・D4・D11）

[プロセスレジストリ](../process/registry.md)を次のとおり変える。

- 成果物レジストリの `prd` の行の担当を「`anet-create-prd`、`anet-create-fix-prd`」にする。
- `issue-record` の行の出力先を「`reports/registry/findings.jsonl`（台帳。直す課題）、`docs/memo/999_{slug}_10prd.md`（新しい振る舞いの案）、PRD の未決表、`/// @todo`」にする。担当の列は変えない。
- 工程 Agent の切り替え先の表に行を足す。工程は「台帳の課題の PRD 化」、入口は `anet-create-fix-prd`、扱いは「台帳の F 番号を 1〜5 件まとめて PRD を起票する。F 番号の選定のあとは PRD の作成／改訂と同じで、起草・自己点検・レビュー後に停止する」とする。
- 読み替え表の 4 行を次のとおりにする（1 行目は改める。2〜4 行目は足す）。

| 共有コアの記述 | anet での読み替え |
|---|---|
| 課題記録への起票 | 直す課題は台帳、新しい振る舞いの案は `999` の暫定 PRD。どちらにも当たらない未決は、PRD の未決表か `/// @todo`。台帳へ積む手順は本書の「台帳への記録」による |
| `an-create-issue` の「分け方」（新規か既存への追記か。7 節） | 台帳の同じ課題の判定（fingerprint）と、台帳の `open`・`stale` の件・直下の 999 PRD の題との照合。同じ課題なら既存への追記にする |
| `an-list-issues` で起票済みの課題を見る（7 節） | 台帳は `registry.py list`（派生一覧は `reports/registry/findings.md`）。999 PRD は `docs/memo/` 直下の一覧 |
| 課題記録は 1 課題 1 ファイル（7 節） | 台帳は 1 課題 1 行。ブランチ間の衝突は行の単位で直す。F 番号が重なったら、後から入った側を振り直す（`validate` が重なりを出す） |

- 新しい節「台帳への記録」を足す（中身は §5.2）。

### 5.2 台帳への記録（レジストリの新しい節の中身。D4）

工程の最後の起票候補で、開発者が台帳への新規か既存の F 番号への追記を選んだ件を、工程を実行する側が次の手順で積む。台帳の CLI は `./.venv/Scripts/python.exe reports/tools/registry.py`、項目と CLI の正本はその冒頭の docstring である。

1. 新規の件は、新しく積む記録を `.scratch/<工程>/` の JSONL に書き、`add --file <JSONL> --run <run_id>` を実行する。`add` が同じ課題（fingerprint の一致）と判定したら新しくは積まれないので、報告ではその F 番号への追記として扱う。
2. 既存の F 番号への追記は、本文（`title`・`summary`・`scenario`）と `file`・`symbol`・`fix_hint` を書き換えない。`touch --ids <id> --run <run_id>` で最終確認日を更新し、`set --id <id> --status <今の状態のまま。stale なら open> --note "再発: <今回の工程と見つけた場面>" --run <run_id>` を実行する。
3. `build` と `validate` を実行する。
4. 成果物（PRD か実装計画）の「未解決」を F 番号へ置き換える（プロセス契約 §7 の段階 3）。

新しく積む記録の項目:

| 項目 | 値 |
|---|---|
| `lens` | 課題の種類。`defect`・`refactor`・`architecture`・`drift`・`test-gap`・`backlog`・`process` のどれか。複数に当たるときは最も根本のもの（anet-audit のガードレールと同じ） |
| `category` | 観点の中の分類タグ（anet-audit の観点の表の「典型的な分類タグ」） |
| `title` | 課題の 1 行 |
| `file`、`symbol` | 課題の主な置き場のファイルと、関数・クラス・節の名前（無ければ空） |
| `evidence` | 置き場の位置（`path`・`line`・`note`） |
| `summary` | 何が問題か（1〜3 文） |
| `scenario` | 具体的な失敗の筋書きか、具体的な smell |
| `fix_hint` | 工程の中で考えた直し方があれば、その一言 |
| `priority`、`confidence` | anet-audit の「レコードの品質基準」の基準による |
| `topic` | 関係する GitHub Topic の Issue 番号が分かれば書く |

`<run_id>` は `<YYYY-MM-DD>-prd<NNN>-<claude|codex>` とする（`<NNN>` はその工程の PRD の番号）。

### 5.3 skill `anet-create-fix-prd` の手順（D5〜D12）

起動は `/anet-create-fix-prd [F-xxxx,F-yyyy,... | <機能カテゴリの key> | <パス>]`（Codex は `$anet-create-fix-prd`）。引数はカンマで区切った要素の並びとして読み、次の表を上から当てて、最初に当たった行で扱う。

| # | 引数 | 扱い |
|---|---|---|
| 1 | 無し | 台帳の `open`・`stale` の全件から候補の組を示す |
| 2 | 要素のどれか 1 つでも `F-` で始まる | F 番号の指定として扱う。次の条件を順に確かめ、最初に外れた条件を理由にして止まる。(a) すべての要素が `F-` と 4 桁の数字の形である（key やパスが混ざったときは、ここで外れる）。(b) 要素が 1〜5 個である。(c) 同じ F 番号が重なっていない。(d) どの F 番号も台帳にある。(e) どの件も状態が `open` か `stale` である。すべて満たせば、その F 番号を題材にする |
| 3 | 要素が 1 つで、`reports/tools/categories.json` の key に一致する | その機能カテゴリの件から候補の組を示す |
| 4 | 要素が 1 つで、上に当たらない | 作業ツリーのパスとして扱い、その下の件から候補の組を示す。パスが実在しなければ止まる |
| 5 | 要素が 2 つ以上で、上に当たらない | 止まる |

止まるときは PRD を書かず、外れた条件か当たった行と、受け付ける形を示す。候補の組が 1 つも作れない（当たる `open`・`stale` の件が 0 件）ときも同じに止まる。

1. **対象を選ぶ。**
   - F 番号の指定（表の 2）は、条件をすべて満たせば、その組で次の手順 2 へ進む。
   - 候補の組を示すとき（表の 1・3・4）は、`registry.py list --status open` と `--status stale` を、表の 3 なら `--category`、表の 4 なら `--path` で絞って引く。件を機能カテゴリごとに分ける。どの機能カテゴリにも当たらない件は、ファイルの親ディレクトリごとに分ける。分けた中を `list` の並び（優先度の順、同じなら F 番号の順）で 5 件ずつ区切り、それぞれを 1 組とする。1 つの機能カテゴリに 6 件以上あれば、組は 2 つ以上になる。
   - 組を、組の中で最も高い優先度の順に、同じなら件数の多い順に、さらに同じなら組の中で最も小さい F 番号の順に並べ、先頭から最大 3 組を示す。ユーザーは 1 組を選ぶ。示した組から件を除いて選んでもよい。示していない組や件を使いたいときは、F 番号の指定で起動し直す。
2. **根拠を確かめる。** 各件の `evidence` を読み直す。直っていれば `fixed` で閉じ、組から外して報告する（D8）。全件が外れたら、PRD を書かずに止まる。
3. **anet-create-prd の段 1〜7 に従う。** 読み替えは D9 の 3 つだけである。段 1 の wiki の目録の引き方、goal anchor、scope screen、論点の仕分け、起草、別モデルの検討、自己点検、レビュー、未決の提示は、anet-create-prd のとおりに行う。
4. **段 7 で台帳を更新する。** 起票候補の記録のあと、完了報告の前に、題材の各件を `promoted` にして `build`・`validate` を回す（D12）。グリルで組から外した件は触らない（D10）。

`agents/openai.yaml` は anet-create-prd に揃える。表示名は `ANET Create Fix PRD`、短い説明は「台帳の F 番号を最大 5 件まとめて PRD を起こす。選定のあとは anet-create-prd と同じ」とする。

### 5.4 `registry.py validate` の変更（D15）

- F 番号が 2 つ以上の記録に現れたら、その F 番号と該当する記録の題を問題として出す。終了コードは、ほかの問題と同じく 1 にする。
- 冒頭の docstring の `validate` の説明に、F 番号の重複を検査することを書く。
- 振り直しは自動で行わない。人かエージェントが、後から入った側の記録の `id` を直す。

### 5.5 skill と文書の書き換え

- [anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md) の論点の仕分けの ⑥ を、「レジストリの課題記録の読み替えに従い、直す課題は台帳へ、新しい振る舞いの案は 999 の暫定 PRD か未決表へ送る。起票はプロセス契約7節に従い、ユーザーが選んだものだけ行う」にする。
- [anet-audit](../../.agents/skills/anet-audit/SKILL.md) の禁止の節の「人が PRD に昇格させたら `set --status promoted --note "<PRD 番号>"` で閉じる」を、「PRD に上げるのは `anet-create-fix-prd` で、閉じるのも同 skill が行う」にする。
- [anet-improve](../../.agents/skills/anet-improve/SKILL.md) の「報告して止まる」の節（176 行目）の「PRD 送りの対策と、台帳の id。`/anet-create-prd`（Codex は `$anet-create-prd`）の題材になる」を、「`/anet-create-fix-prd`（Codex は `$anet-create-fix-prd`）の題材になる」にする。台帳の F 番号を PRD にするのは基本 fix PRD だけにするためである（G3）。
- [implement-prd-with-docs の手順](../../.agents/skills/implement-prd-with-docs/PRD-IMPLEMENTATION.md) は変えない。工程末の起票候補はプロセス契約 7 節を指していて、置き場はレジストリの読み替えで決まる（§2 の 9）。
- [memo README](README.md) の「工程と担当」の「グリルして起票」の行に、「台帳の F 番号からは `anet-create-fix-prd`」を足す。

### 5.6 移行対象と文書の更新

| 対象 | 変更 |
|---|---|
| `docs/process/registry.md` | §5.1・§5.2 |
| `.agents/skills/anet-create-fix-prd/SKILL.md`、`agents/openai.yaml`（新規） | §5.3 |
| `.claude/skills/anet-create-fix-prd/`（新規） | 上の実体コピー |
| `.agents/skills/anet-create-prd/SKILL.md` と `.claude/` の同じファイル | §5.5 の ⑥ |
| `.agents/skills/anet-audit/SKILL.md` と `.claude/` の同じファイル | §5.5 の禁止の節 |
| `.agents/skills/anet-improve/SKILL.md` と `.claude/` の同じファイル | §5.5 の報告の節（176 行目） |
| `.agents/skills/README.md` と `.claude/skills/README.md` | プロジェクト固有の表に `anet-create-fix-prd` の行を足す |
| `docs/memo/README.md` | §5.5 の工程と担当 |
| `reports/tools/registry.py` | §5.4 |
| `core/anet-core/testdata/prd095/registry_test.py` | §5.4 のテストを足す（§7 の 3） |
| 動線マップ（`docs/agents/README.md` と図） | 手で直さない。実装のあとで `anet-harness-map` を回して作り直す |

台帳の既存の記録（37 件）は書き換えない。過去の PRD・実装記録・ADR も書き換えない。

### 5.7 クラス一覧

足す・変える・消すクラスは無い。

## 6. 細部（暫定で置いた既定）

1. 候補の組の数: 最大 3 組。同じ機能カテゴリの組が 2 つ以上入ってもよい（理由: 選ぶ手間と、ほかの組を見落とさない幅の折り合い。件数の多い機能カテゴリを 1 組に絞ると、6 件目以降が候補に出なくなる）。
2. 束ねる単位: 機能カテゴリ（`categories.json` の key）。当たらない件はファイルの親ディレクトリ。組の中の件は `list` の並び（優先度の順、同じなら F 番号の順）で 5 件ずつ区切る（理由: 実装で読むコードとテストが重なる。分類の定義と CLI の並びを変えずに使える）。
3. `run_id` の形: 入口は `<YYYY-MM-DD>-prd<NNN>-<claude|codex>`、fix PRD の段 7 は `<YYYY-MM-DD>-fixprd<NNN>-<claude|codex>`（理由: anet-improve の `<日付>-improve-<枠>` に揃え、どの工程が触ったかを履歴で引ける）。
4. `promoted` の注記: fix PRD のファイル名（パスを除く）（理由: 採番し直しでも名前の後ろが残り、検索で引ける）。
5. `{範囲}` の名前: key の番号を除いた部分（`318_rl_core` → `rl_core`）。親ディレクトリで決めるときは、その最後の名前。リポジトリの根なら `root`（理由: 番号は分類の並び順で、PRD の番号と紛れる。根には名前が無く、作業ツリーのフォルダ名は置き場によって変わる）。

## 7. 検証と受入

1. `diff -rq .agents/skills .claude/skills` が差を出さない。
2. `docs/process/registry.md` の `issue-record` の行に `reports/registry/findings.jsonl` があり、読み替え表に §5.1 の 4 行がある。
3. PRD095 の検証資材の `registry_test.py` に、F 番号を重ねた一時台帳で `validate` が重なった F 番号を出して終了コード 1 で終わるテストを足し、`registry_test.py` 全体が通る。正本の台帳への `validate` は問題 0 件で終わる。
4. 試走 A（出口）: `anet-create-fix-prd` を、同じ機能カテゴリの `open` の F 番号 2 件以上で回す。次を満たす。
   - fix PRD が 1 本でき、ファイル名が D13 の形である。
   - 本文に件ごとの節と受入がある（D14）。
   - an-review が OK になる。
   - 題材の各件が `promoted` になり、注記に fix PRD のファイル名がある。
   - `validate` が問題 0 件で終わり、`build` で派生一覧と地図の区画が作り直されている。
5. 試走 B（上限と状態）: どちらも PRD を書かずに止まり、外れた条件を示す。
   - `open` の F 番号を 6 件指定したときは、§5.3 の表の 2 の条件 (b)（件数）を示す。
   - `open` の件に `promoted` の件を 1 件混ぜて 5 件以内で指定したときは、条件 (e)（状態）を示す。
6. 試走 C（入口）: 本書のレジストリの変更（§5.1・§5.2）が作業ツリーに入ったあとの、anet-create-prd か implement-prd-with-docs の工程末のうち、次の 3 つをすべて満たす最初のものを試走 C とする。(a) 起票候補が 1 件以上ある。(b) そのうち 1 件以上に、区分「台帳へ新規」か、F 番号への「既存へ追記」が付いている。(c) 開発者がその件を 1 件以上選んだ。満たさない工程末は試走 C に数えず、合否も付けない。試走 C では、次をすべて満たせば合格とする。
   - 起票候補の表の各行に、「台帳へ新規」「999 PRD を新規」「既存へ追記」「見送り推奨」のどれか 1 つの区分が付いている。
   - 選ばれた「台帳へ新規」の件が台帳に F 番号つきで現れ、選ばれた F 番号への追記の件は、その記録の履歴の最後に今回の注記がある。
   - その工程の成果物（PRD か実装計画）で、選ばれた件の「未解決」が F 番号に置き換わっている。
   - `registry.py validate` が問題 0 件で終わる。
   本書の実装を終えるまでに試走 C に当たる工程末が無ければ、この項を未実施として実装記録に残し、当たる工程末で確かめる。

## 8. 作らないもの・先送り

- 入口専用の skill（`an-create-issue` に当たるもの） — 不採用: 積む操作は CLI の 3 つで、手順はレジストリの 1 節で足りる（D4）。
- 起票候補の区分「その場で直す」 — 先送り: 別の関心である。本書は見つけた課題の置き場を決めるもので、工程の中で直す範囲を広げることは扱わない。工程の最後に直すと、直したあとのレビューの扱いも決める必要がある。再開条件: 試走 C のあとの工程末で、台帳へ積んだ件のうち、その工程の中で直せる大きさの件が続けて出たとき。そのときは、直したあとのレビューの扱いとあわせて別の PRD で決める（§9 の 2）。
- 999 PRD のうち直す課題に当たるものの台帳への移し替え — 先送り: 入口と出口が動いてから決める。再開条件: 試走 A と試走 C を終えたとき。
- anet-housekeeping の候補 — 不採用: 起票にはグリルの質疑が要り、housekeeping の「了承後は質問しない」約束と合わない（§3.2）。
- 台帳の F 番号の自動の振り直し — 不採用: 衝突は稀で、`validate` が出せば人が直せる（D15）。
- 台帳の新しい状態 — 不採用: `promoted` と注記で足りる（§3.2、D12）。

## 9. 未決

| # | 事項 | 今の既定 | 決めること |
|---|---|---|---|
| 1 | 入口の分け方 | 直す課題は台帳、新しい振る舞いの案は 999 PRD。大きさでは分けない（D2） | 直す課題でも、設計の判断が大きいものを 999 PRD にする道を残すか |
| 2 | 「その場で直す」区分 | 足さない（§8） | 工程の最後に「この場で直す」を選べるようにするか。選べるなら、直したあとのレビューをどうするか |
| 3 | 1 課題 1 ファイルとの違い | 台帳は 1 課題 1 行。衝突は行の単位で直し、F 番号の重なりは `validate` で見つけて人が振り直す（§5.1・D15） | ブランチを並べて台帳に積む運用で足りるか。足りなければ、ブランチでは仮の番号にして main への取り込み時に振る形にするか |
| 4 | 束ね方 | 同じ機能カテゴリ、当たらなければディレクトリ（D7、§6 の 2） | 機能カテゴリをまたぐ組を、F 番号の指定以外でも作れるようにするか |
| 5 | fix PRD の番号 | memo README の採番（999 で起票し、着手時に正式番号）に従う（D13） | fix PRD は片付ける前提なので、起票時から連番にするか |
| 6 | `promoted` のあと | fix PRD の実装では題材の件の状態を変えない。完了は `done/` が表す（D12） | fix PRD が `dropped/` に移ったとき、題材の件を `open` に戻すか。誰がいつ戻すか |
| 7 | 999 PRD の移し替え | 本書では行わない（§8） | 直す課題に当たる 999 PRD（§2 の 4 の例など）を台帳へ移すか。移すなら、999 PRD を `dropped/` へ移すときの理由の書き方 |
| 8 | anet-create-prd に F 番号を渡されたとき | 何もしない（G3 は「基本」） | anet-create-prd の題材に F 番号があるとき、`anet-create-fix-prd` で起こすよう示すか |
| 9 | 「定義も流用」の読み | 成果物の定義（`prd` の行・観点・採番）を共通にし、skill の定義は参照で書く（D11） | ユーザーの言う「定義」がこの 2 つで合っているか |
| 10 | 別モデルの検討 | 暫定の起票なので行っていない | 再グリルで別モデルの検討を入れるか |

## 10. 依存と順序

- 先に要るものは無い。台帳と CLI（PRD095 で整備）はそのまま使う。
- 実装の順は、レジストリ（§5.1・§5.2）→ `registry.py`（§5.4）→ skill（§5.3・§5.5）→ README と memo README → 試走 A・B → 動線マップの作り直し、とする。試走 C は次の工程の最後で行う。

## 11. 参照

- [プロセスレジストリ](../process/registry.md)、[プロセス契約](../process/protocol.md) §7
- [ADR 0049](../adr/0049-process-contract-kept-project-neutral-anet-specifics-in-registry-and-viewpoints.md)
- [anet-create-prd](../../.agents/skills/anet-create-prd/SKILL.md)、[implement-prd-with-docs の手順](../../.agents/skills/implement-prd-with-docs/PRD-IMPLEMENTATION.md)、[anet-audit](../../.agents/skills/anet-audit/SKILL.md)、[anet-improve](../../.agents/skills/anet-improve/SKILL.md)、[skills README](../../.agents/skills/README.md)
- [台帳の CLI](../../reports/tools/registry.py)、[台帳の一覧](../../reports/registry/findings.md)
- [memo README](README.md)、[PRD 品質確認観点](../quality/prd.md)
- [999_metrics_action_info_dynamic_cast](done/999_metrics_action_info_dynamic_cast_10prd.md)、[110_20impl](done/110_runner_main_pane_dynamic_centre_20impl.md)、[110_22impl](done/110_runner_main_pane_dynamic_centre_22impl.md)
- [PRD095](done/095_llm_wiki_10prd.md)、[311 開発ツール（wiki）](../wiki/categories/311_dev_tools.md)
