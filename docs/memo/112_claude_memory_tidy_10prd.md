# Claude Code の memory の定期整備（`anet-tidy-memory` と housekeeping の候補 `memory-tidy`）PRD

> 2026-11-02 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した。番号はユーザーの指示で連番）。数値と行番号は起票時点のもので、実装時は取り直す。
> 起点: 2026-10-31〜11-02 の会話。LLM-Wiki（[PRD095](done/095_llm_wiki_10prd.md)）の目的と実績を整理したところ、Claude Code の memory に結果の数値と古い状態の記述が溜まり直していると分かった。ユーザーの望みは「memory の整合チェックとお掃除を、wiki の保守とは別に定期的に回す。まず Claude Code、そのあと Codex 側の対応を検討する」。
> 関連: [PRD095](done/095_llm_wiki_10prd.md)（§4.5 の memory の移し替え、D2・D38）、[ADR 0056](../adr/0056-wiki-is-derived-summary-and-readme-keeps-original-analysis.md)、[anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md)、[anet-improve](../../.agents/skills/anet-improve/SKILL.md)、[anet-lint-wiki](../../.agents/skills/anet-lint-wiki/SKILL.md)、[skills README](../../.agents/skills/README.md)、[wiki の規約](../wiki/README.md)、[311 開発ツール（wiki）](../wiki/categories/311_dev_tools.md)。

## 1. 目的と問題

Claude Code の auto memory（以下 memory）を、git・フォルダ・正本・規約と定期的に照合して直す skill `anet-tidy-memory` を足し、anet-housekeeping の候補 `memory-tidy` にする。直すのは次の 4 つである。古くなった状態の記述。正本や規約へ移った中身の二重持ち。重複と、終わった仕事の記述。読み込み上限を超えた索引。

解く問題は次の 4 つ（事実は §2）。

1. memory の索引（`MEMORY.md`）が公式の読み込み上限を超えている。超えた分は会話の始めに読み込まれない。新しい項目は末尾に足されるので、読まれなくなるのは新しい項目である。
2. 状態の記述（未コミット、実装済みなど）が古くなる。書いた時点では正しくても、コミットや PRD の `done/` への移動で事実が変わる。memory は書いたときにしか直されない。
3. 知見（Run の数値、調査の結論）が memory に溜まり直している。PRD095 の段階 3 で一度ポインタ化したが、その後も索引と topic file に結果の数値が書き足されている。memory は Claude Code だけが読めるので、正本と二重に持つと食い違いの元になる。
4. memory を定期的に見直す仕組みが無い。既存の仕組みは対象が狭いか、anet の規則（状態はフォルダと git、知見は正本から wiki へ）を知らない。

### 1.1 用語

本書の語はハーネス（エージェントの作業環境）の語で、フレームワークの語ではないので、[CONTEXT.md](../../CONTEXT.md) には足さない。既存の語で言い換えられないものだけを次のとおり使う。

| 語 | 意味 |
|---|---|
| memory | Claude Code の auto memory。Claude が会話をまたいで残す覚え書きで、索引と topic file から成る |
| 索引 | memory の `MEMORY.md`。1 行 1 項目で topic file を指す |
| topic file | memory の 1 件 1 ファイルの本文 |
| 正本 | [wiki の規約](../wiki/README.md)の語。実験記録、サーベイ、PRD、設計文書、ADR |
| 正本なし | memory にしか無く、正本に同じ中身が見つからない知見 |
| ポインタにする | topic file の中身を、正本や規約の所在へのリンクと、正本に無い経緯だけに縮めること |

## 2. 現状（2026-11-02 に確認したこと）

1. 公式の仕様（[How Claude remembers your project](https://code.claude.com/docs/en/memory) の auto memory の節）。
   - `MEMORY.md` は、先頭の 200 行か 25KB の早い方までが会話の始めに読み込まれる。それを超えた分は読み込まれない。
   - topic file は会話の始めには読み込まれず、必要なときに Claude が読む。
   - Claude が `MEMORY.md` を書くと、Claude Code が上限と比べる。上限に近ければ短くするよう知らせる。超えていても書き込みは通るが、索引を書き直すよう Claude にエラーを返す。
   - 置き場は `~/.claude/projects/<project>/memory/` である。同じ git リポジトリの worktree とサブフォルダは 1 つの置き場を共有する。memory はマシンの外へは共有されない。
   - 公式のページは、整理の機能（Auto-dream）の挙動を記述していない。
2. 今の memory。
   - 索引は 137 行・28,845 バイトで、25KB を超えている。topic file は 212 本・約 674KB ある。
   - 索引の行が指すファイルはすべて実在し、索引から指されない topic file は無い。
   - この会話の始めには索引の末尾の行まで見えていた。上限をバイトと文字のどちらで数えるかは確かめていない（§9 の 6）。
3. 状態の記述。
   - 索引に「未コミット」を含む行が 16 行ある。
   - そのうち 4 件を git で確かめたところ、どれもコミット済みだった。
     - PRD095 段階 3: `docs/memo/done/095_llm_wiki_22impl.md` が 0dc43645（10/23）。
     - 動線マップ: `docs/agents/README.md` が 36cc56eb（10/31）。
     - 空き時間スキル群: `.agents/skills/anet-housekeeping/SKILL.md` が 9f0f0cc8（10/22）。
     - anet-improve: `.agents/skills/anet-improve/SKILL.md` が d8ee7501（11/02）。
4. 知見。10/24〜11/02 に更新した project 型の topic file のうち、17 本が数値の知見（%・seed・中央値・倍）を持つ。そのうち 16 本は repo のパスも書いている。正本を指しながら、数値も memory に写している。
5. 段階 3 の移し替え（[PRD095](done/095_llm_wiki_10prd.md) §4.5 の 3、[記録](done/095_llm_wiki_22impl.md)の「2026-10-23 Claude Code の memory の移し替え」）。
   - 10/23 に 97 本をポインタ化した。
   - Run で確かめると、memory の値に誤りが多数見つかった（打ち切り扱いが完走、複製扱いが別設定、Run フォルダの改名など）。
   - このため、memory の数値は正本へ写す前に Run で確かめる。
6. 既存の仕組み。
   - `consolidate-memory`（Anthropic 提供の汎用 skill）。
     - 棚卸し、持続するもの（好み・作業の流儀）と日付のあるもの（終わった仕事）の分離、重複のまとめ、相対日付の絶対日付化、すぐ引き直せる記述の削除を行う。
     - 索引を 200 行・約 25KB 以内、1 行約 150 字に整える。
     - git・フォルダ・repo の正本・規約とは照合しない。
   - Auto-dream（`/dream`）。
     - 公式では「上級者向けのヒント」（[Claude Code power user tips](https://support.claude.com/en/articles/14554000-claude-code-power-user-tips)）に名前が出るだけである。
     - 第三者の解説は、裏での整理と実行前のバックアップを挙げている（例: [claudefa.st](https://claudefa.st/blog/guide/mechanics/auto-dream)）。挙動は確かめていない。
   - [anet-improve](../../.agents/skills/anet-improve/SKILL.md) の「memory（Claude Code だけ）」。対象は、その会話で書いた・直した feedback memory のうち規約へ移したものだけである。
   - [anet-lint-wiki](../../.agents/skills/anet-lint-wiki/SKILL.md)。材料は repo の中の文書だけである（[PRD095](done/095_llm_wiki_10prd.md) §4.7）。[wiki の規約](../wiki/README.md)は「memory は出典にしない」と定めている。
   - [anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md)。
     - memory の候補は無い。
     - 禁止事項に「各スキルの出力先と `reports/housekeeping/` 以外への書き込み」がある。
     - `runs.jsonl` は repo で管理されている（`git ls-files reports/housekeeping`）。
   - [skills README](../../.agents/skills/README.md)。
     - `.agents/skills`（Codex）と `.claude/skills`（Claude Code）に同じ中身の実体コピーを置き、`diff -rq` で差が無いことを確かめる。
     - 定常整備系の出力は `reports/` の下に置く。`docs/` に置くのは公開が目的のものだけである。
     - 定常整備系（anet-housekeeping と、それが呼ぶ候補）は、実行前の了承確認を anet-housekeeping が行い、了承後の各作業は質問して止まらない（「anet- スキル共通の約束」）。
   - `.scratch/` は git の管理外である（[.gitignore](../../.gitignore) の `/.scratch/`）。
7. 一般の事例（どれも会話の外で memory をまとめ直し、元を壊さない）。
   - [Claude Platform の Dreams](https://platform.claude.com/docs/en/managed-agents/dreams): 既存の memory store と過去の会話記録を読んで、新しい store を作る。入力の store は変えない。
   - Letta の sleep-time agent（[解説](https://learn.traeai.com/t/ai-engineering/phases/14-agent-engineering/08-memory-blocks-sleep-time-compute.html)）: 本体と memory を共有する別のエージェントが裏で重複をまとめ、古い事実を無効にする。
   - LangMem の background memory manager（[Background Quickstart](https://langchain-ai.github.io/langmem/background_quickstart)）: memory の抽出とまとめを会話から切り離して後で処理する。
8. Codex の memory。
   - 置き場は `~/.codex/memories/`（`MEMORY.md`、`memory_summary.md`、`raw_memories.md`、`rollout_summaries/`）で、Codex アプリが更新する。
   - PRD095 D38 で「読むだけで書き換えない」と決めた。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

- G1: memory の索引を公式の読み込み上限（200 行・25KB）の内に保ち、次の実行までに足す分の余裕を残す。
- G2: git とフォルダで答えが出る状態の記述が、事実と食い違わない。
- G3: 正本・規約に同じ中身がある知見と feedback は、memory の側をポインタにして中身を二重に持たない。正本の無い知見は、ユーザーが扱いを決められる一覧になる。
- G4: これを定期的に回せる（housekeeping の候補）。明示的に起動もできる。
- G5: 整備で memory を失わない。実行前の写しから戻せ、同じ時期にほかの会話が書いた中身を上書きしない。

### 3.2 非ゴール

- memory を出典にして wiki や正本を書くこと。退ける案は「整備のついでに正本・wiki へ書き出す」案である。PRD095 D2 の原則で、memory の値には誤りが多かった（§2 の 5）。正本へ起こすのはユーザーが選んだものだけで、Run で確かめる作業は実験記録や PRD の工程で行う。正本の質を保つための本当の制約で、移しても解けない。
- Codex の memory の整備。退ける案は「段階 1 で Codex の memory も整備する」案である。本当の制約ではなく、書き換えの可否を確かめるまでの先送りである（§8）。
- `CLAUDE.md`（ユーザーの全体の指示、プロジェクトの指示）を直すこと。退ける案は「`CLAUDE.md` も整備の対象にする」案である。人が書く指示で、Claude が書く memory とは書き手が違う。書き手の違いによる本当の制約である。
- Auto-dream を使うかどうかを決めること。退ける案は「Auto-dream を有効にして任せる」案である。挙動の記述が公式に出るまでの先送りで、本当の制約ではない（§8）。
- memory の置き場を変えること（`autoMemoryDirectory`）。退ける案は「置き場を repo の中へ移す」案である。公開 repo に私的な memory が入るという本当の制約による（§8）。
- memory の中身を `reports/` や `docs/` に書くこと。退ける案は「1 回分の記録を `reports/housekeeping/` に置く」案である。repo は公開しており、memory には私的な事柄がある。本当の制約である。

### 3.3 足す機構とゴールへの寄与

| 機構 | 寄与するゴール |
|---|---|
| skill `anet-tidy-memory`（照合と直し、D3・D4） | G1・G2・G3 |
| 補助スクリプト `reports/tools/memory_support.py`（`scan`・`backup`・`record`、D9） | G1・G2・G5（数え方を毎回同じにし、読む量を減らす） |
| 実行前の写しと、書く直前の確かめ（D5） | G5 |
| housekeeping の候補 `memory-tidy`（D10） | G4 |
| 1 回分の記録（`.scratch/anet-tidy-memory/`、D6） | G3（正本なしの一覧）・G5（戻し方） |

### 3.4 成功の測り方

- 実行のたびに、記録（D6）へ次の数を残す。
  - 索引の行数とバイト数（実行の前と後）。
  - 状態の食い違いを見つけた数と直した数。
  - ポインタにした数（知見と feedback）。
  - 正本なしとして挙げた数。
  - まとめた数と引退させた数。
  - パス切れの数（直した数、注記した数）。
  - 消費率と所要時間（housekeeping の `runs.jsonl`）。
- 4 回実行した時点で、次の 2 つを満たせば効果ありとする。
  - 実行の直前の索引が上限（200 行・25KB）を超えていた回数が、初回を除いて 0。
  - 2〜4 回目のそれぞれで、実行の直前に見つかる状態の食い違いの数が、初回より少ない。
  - 満たさなければ、§6 の閾値と間隔を見直す。
- 起票時の基準値: 索引 137 行・28,845 バイト、「未コミット」を含む索引の行 16、数値の知見を持つ最近の project 型 17 本。

## 4. 確定事項（暫定の既定。会話で合意した決定ではない。§9 とあわせてグリルで見直す）

| # | 決定 | 理由 |
|---|---|---|
| D1 | 対象は、Claude Code の、この repo の memory の置き場（索引 `MEMORY.md` と topic file）だけとする。置き場は、会話のシステムプロンプトの auto memory の節が名指すフォルダから実行時に決める | 問題はここで起きている（§2 の 2〜4）。パスはユーザー名を含むので、skill と PRD に書かない |
| D2 | 整備は skill `anet-tidy-memory` に置き、`.agents/skills` と `.claude/skills` の両方に同じ中身を置く。Codex で起動されたときは、何も書かずに「Claude Code の memory は Codex からは整備しない」と報告して終える | skills README の両方に置く約束（§2 の 6）に従う。段階 1 は Claude Code だけ（D12） |
| D3 | 見るものは 7 つとし、それぞれ確かめる先を決める。(1) 索引の大きさ → 公式の上限。(2) 索引と topic file の対応 → 置き場のファイル。(3) 状態の記述 → git（log・status）と `docs/memo/` のフォルダ。(4) repo のパスの実在 → 作業ツリー。(5) 知見 → 正本（実験記録、PRD、ADR、設計文書、`reports/` の報告）と wiki。(6) feedback → `AGENTS.md`、`docs/quality/`、`docs/process/`、skill。(7) 重複と終わった仕事 → topic file どうしと `docs/memo/done/` | 問題 2・3 の原因は、memory の外の事実が変わることである（§1）。外の事実と照合しないと直せない。(7) は `consolidate-memory` の考え方を借りる（§2 の 6） |
| D4 | 直し方は種類ごとに決める（§5.1 の表）。索引には git とフォルダで答えが出る状態の語を置かない | 状態の語は時間で古くなり（問題 2）、索引の長さも増やす（問題 1）。答えは git とフォルダで引き直せる。D3 に依存する |
| D5 | 書き込む前に、memory の置き場の全体を `.scratch/anet-tidy-memory/<run_id>/before/` へ写す。直しは、書く・消すファイルの組（まとまり）ごとに、書く直前に中身が写したときと同じかを確かめ、違えばまとまりごと触らずに報告する。同じ topic file への直しはすべて 1 つのまとまりに合わせ、まとまりどうしを重ねない（各 topic file は 1 回の実行で多くとも 1 回だけ書く）。索引は最後に読み直し、今の索引に行の単位で当て直す（§5.2） | memory は git の管理外で、ほかに戻す手段が無い。同じ repo のほかの会話（worktree を含む）が同じ置き場を同時に書きうる（§2 の 1）。一般の事例も元を壊さない（§2 の 7） |
| D6 | 1 回分の記録（直したもの、正本なしの一覧、パス切れ、戻し方）は `.scratch/anet-tidy-memory/<run_id>/report.md` に置く。前回の実行の記録は `.scratch/anet-tidy-memory/state.json` に置く。`.scratch/anet-tidy-memory/` は、どの worktree から起動しても主 worktree（`git rev-parse --git-common-dir` が指す共通の git ディレクトリの親）の下に置く。本書の `.scratch/anet-tidy-memory/` はすべてこの置き場を指す。`reports/` と `docs/` には memory の中身を書かない。housekeeping 経由のときは、`runs.jsonl` の `note` に件数だけを書く | repo は公開しており、memory には私的な事柄がある（§3.2）。`.scratch/` は git の管理外である（§2 の 6）。memory の置き場は同じ repo の worktree で共有される（§2 の 1）ので、前回の記録と写しも worktree によらず 1 か所にしないと、前回の実行と差分の基準が worktree ごとに割れる |
| D7 | 直す範囲を 2 つ置き、起動のしかたで使い分ける。「全部」は §5.1 の表の「直し方」の列の全部で、まとめと引退を含む。「正誤が決まるものだけ」は同じ表の右の列で、確かめた先で正誤が決まるものに限る。正本なしの知見は、どちらの範囲でも触らない。明示起動では、棚卸しのあと直す案を種類ごとの件数と一覧で示し（まとめは残す側を、引退は残す教訓の有無を案に書く）、1 問で了承を得てから直す。選択肢は「全部直す」「正誤が決まるものだけ直す」「やめる」。housekeeping 経由では、了承は housekeeping が取るので質問せず、「正誤が決まるものだけ」を直し、残り（まとめ、引退、正本なしの知見）は報告に挙げる | 定常整備系は、了承後の各作業で質問して止まらない約束である（§2 の 6、[skills README](../../.agents/skills/README.md) の「anet- スキル共通の約束」）。まとめと引退には、残す側と残す教訓の判断が要る。無人で行うと memory を失いうる（G5）。明示起動ではユーザーが案を見て決められる。正本なしの知見は、正本へ起こすかの判断と Run での確かめが要り、本書の範囲外である（§3.2） |
| D8 | 毎回読む範囲は差分にする。索引の全行と、前回から変わった topic file と、状態の語か repo のパスを含む topic file（`scan` の検索で絞る）を読む。引数 `full` を付けたときと、前回の記録が無いときは全部を読む | topic file は 212 本・約 674KB あり、毎回全部を読むと 1 回の費用が大きい（§2 の 2）。古くなる原因の状態とパスは検索で拾える |
| D9 | 機械的な部分は補助スクリプト `reports/tools/memory_support.py` が受け持つ（§5.2）。本文の照合と縮め方の判断は skill を実行する側が行う | 決まった検査を毎回同じに数え、読む量を減らす。`wiki_support.py` と同じ分け方である（[wiki の規約](../wiki/README.md)の「機械処理の補助」） |
| D10 | anet-housekeeping に候補 `memory-tidy` を足す（§5.3） | 定期的に回すため（G4）。索引が上限を超えると、超えた分は会話の始めに読まれない（§2 の 1）ので、早めに回す。D7 と D8 に依存する |
| D11 | 名前は、skill が `anet-tidy-memory`、housekeeping の候補が `memory-tidy`、補助スクリプトが `memory_support.py` とする | 既存の形に揃える（[実装 品質確認観点](../quality/impl.md#名前)「名前」の 3）。skill は動詞＋目的語（`anet-lint-wiki`・`anet-ingest-wiki`）、候補は目的語＋動作（`wiki-lint`）、補助は `wiki_support.py` に揃える |
| D12 | Codex の memory（段階 2）は本書では作らず、§8 に先送りとして残す | ユーザーの指示（まず Claude Code）。Codex の memory は Codex アプリが更新し、PRD095 D38 で読むだけと決めている（§2 の 8） |
| D13 | ADR は起こさない | skill と補助スクリプトは後から変えやすく、戻しにくさの条件を満たさない |

## 5. 実装契約

### 5.1 直し方（D4）

| 見るもの | 直し方（「全部」の範囲） | 「正誤が決まるものだけ」の範囲（housekeeping 経由） |
|---|---|---|
| 索引の大きさ | 1 行 1 項目、1 行の目安は 150 字にする。詳しい中身は topic file へ移す。目標は 160 行・20,000 バイト以下（§6 の 1） | 直す |
| 索引と topic file の対応 | 指す先が無い行は消す。索引に無い topic file は、索引に 1 行足す | 直す |
| 状態の記述 | git とフォルダで照合する。索引の行からは、git とフォルダで答えが出る状態の語（§6 の 4）を外す。topic file では事実に合わせ、確かめた日を添える。git とフォルダで答えの出ない状態（グリル前、レビュー OK、確認待ちなど）は topic file に残す | 直す |
| repo のパス | 移った先が 1 つに決まれば直す（`done/` への移動、改名）。決まらなければ、パスの後ろに「（パス切れ）」と注記する | 直す |
| 知見 | 正本に同じ中身があれば、topic file を「正本（と wiki のページ）へのポインタ」と「正本に無い経緯（なぜ・どう使うか）」だけに縮め、数値は写さない。正本に無ければ触らず、正本なしの一覧に挙げる | 正本にあるものだけ直す |
| feedback | `AGENTS.md`・`docs/quality/`・`docs/process/`・skill に同じ中身があれば、規約の所在へのポインタに縮める。全部が移っていれば topic file と索引の行を消す（anet-improve の memory の扱いと同じ考え） | 規約に同じ中身があるものだけ直す |
| 重複と終わった仕事 | 重複はまとめ、中身の多い側のファイル名を残す。相対日付は絶対日付にする。`done/` へ移った PRD だけを指し、残す教訓の無い project 型は引退させる（topic file と索引の行を消す） | 日付だけ直す。まとめと引退は報告に挙げる |

### 5.2 補助スクリプト `reports/tools/memory_support.py`

repo の `.venv` で、repo の根から実行する。memory の置き場は引数で受け取り、パスをスクリプトに書かない。

| サブコマンド | 引数 | 動き | 失敗 |
|---|---|---|---|
| `scan` | `--memory-dir <dir>`、`--state <file>`（省略可） | 読み取りだけを行い、JSON を標準出力へ出す。中身は次のとおり。索引の行数とバイト数、上限（200 行・25,000 バイト）と目標（§6 の 1）を超えたか。索引の行が指す先の無いファイル名と、索引に無い topic file。状態の語（§6 の 4）を含む行（ファイル、行番号、語）。repo のパス（`docs/`・`core/`・`apps/`・`reports/`・`viewers/`・`.agents/`・`.claude/`・`.github/` で始まるもの）と、その実在、無いときに同じファイル名を持つ repo のファイルの候補。`--state` があれば、前回の記録の有無、前回の時刻、前回から中身が変わった topic file。`--state` のファイルが無いときは前回の記録が無いことを示し、全部の topic file を変わったものとする | 置き場が無ければ、その旨を標準エラーへ出して終了コード 1。`--state` のファイルがあって読めない（壊れている）ときは終了コード 1 |
| `backup` | `--memory-dir <dir>`、`--out <dir>` | 置き場の全ファイルを `--out` へ写し、`hashes.json`（ファイル名と sha256）を書く | 置き場が無いときと、`--out` が既にあるときは、何も書かずに終了コード 1 |
| `record` | `--memory-dir <dir>`、`--state <file>`、`--run-id <id>` | 実行の最後に、各ファイルの sha256 と run_id と時刻を `--state` へ書く | 置き場が無ければ、何も書かずに終了コード 1 |

上限のバイトは、安全側に 25,000 バイトで数える（§9 の 6）。必須の引数が無いとき、知らないサブコマンドや引数のときは、argparse の既定どおり使い方を出して終了コード 2 で終える。`--run-id` が空の文字列のときも、何も書かずに使い方を出して終了コード 2 で終える。`--run-id` の形は確かめない。

skill の引数は `full` だけである。ほかの語が付いたときは、何も書かずに受け付ける引数（`full`）を示して止まる。

直しは「まとまり」ごとに当てる（D5）。まとまりは、書く・消す topic file の組と、それに伴う索引の行の直しである。

- 1 本の topic file に当たる直し（状態の記述、repo のパス、知見や feedback のポインタ化、相対日付など）は、すべて 1 つのまとまりに合わせ、1 回の書き込みにする。
- 重複のまとめは、残す側と消す側を 1 つのまとまりにする。残す側にほかの直しが当たるときは、それも同じまとまりに入れる。
- 引退と、移し終えた feedback は、消すファイルとそれを指す索引の行を 1 つのまとまりにする。
- ある topic file が 2 つの直しの組に入るときは、その組どうしを 1 つのまとまりに合わせる。こうして、まとまりどうしは重ならず、各 topic file は 1 回の実行で多くとも 1 回だけ書く。このため、書く直前の判定で比べる先は、つねに実行前の `hashes.json` である。
- まとまりは、その実行の範囲（D7）で選んだ直しだけから作る。範囲の外の直し（「正誤が決まるものだけ」の範囲でのまとめと引退）は、まとまりに入れず、当てなかった直しとして報告に挙げる。範囲の内の直しは、同じ topic file に範囲の外の直しがあっても当てる。

書く直前の判定は次のとおりで、どの場合も 1 つに決まる。

| 書く直前の状態 | 扱い |
|---|---|
| まとまりの topic file の sha256 がすべて `hashes.json` と同じで、消すファイルを指す索引の行が、読み直した今の索引にそのままある | まとまりの topic file を書く・消す |
| topic file のどれかの sha256 が違うか、ファイルが無い。または、消すファイルを指す索引の行が今の索引で書き換わっているか、無い | まとまり全体を当てず、索引の行の直しも捨てる。「scan のあとに変わった」「scan のあとに消えた」として報告に挙げる |

索引は topic file のあとに 1 回だけ書く。索引の直しは行の単位（消す行、置き換える行、足す行）で持ち、書く直前に索引を読み直して、今の索引に当て直す。ほかの会話が足した行はそのまま残る。当て直す先の行が今の索引に無い（ほかの会話が書き換えた・消した）ときは、その行の直しを捨てて報告に挙げる。

### 5.3 anet-housekeeping の候補

| 名前 | スキル | 前提条件 | 単位 | 初期見積（消費率 / 分） |
|---|---|---|---|---|
| memory-tidy | anet-tidy-memory | Claude Code で動いていて、次のどれかに当たる。`memory_support.py scan` で索引が目標（160 行か 20,000 バイト）を超えている。`.scratch/anet-tidy-memory/state.json` が無い。前回の実行から 7 日以上たっている | 1 回 | 10% / 20 |

- 前提条件の判定は次のとおりで、どの組み合わせでも 1 つに決まる。
  - Codex で動いているときは、ほかの条件によらず選ばない。
  - `scan` は D6 の置き場の `state.json` を `--state` に渡して回す。`state.json` が無いときは `scan` が終了コード 0 で前回の記録が無いことを示し、「`state.json` が無い」の条件に当たる。
  - `scan` が終了コード 0 以外で終わったとき（置き場が無い、`state.json` が壊れていて読めない）は選ばず、理由を完了報告に書く。
  - 目標の超過は、行数とバイト数のどちらか一方が超えれば当たる。
  - 7 日は `state.json` に書いた前回の時刻から数える。
  - 3 つの条件のどれか 1 つに当たれば選ぶ。
- 既定の優先順は stats、wiki-lint の次（audit の前）に置く。stats と wiki-lint の順は変えない。どちらも前提条件を満たすときだけ選ばれ、memory の整備とは読む先も書く先も重ならないので、前に置いたままでも memory の整備を妨げない。audit は前提条件が「常に可」で毎回選ばれうるので、その後ろに置くと予算を先に取られる。索引が上限を超えると超えた分が毎回の会話で読まれない（§2 の 1）ので、audit より先に直す。
- 禁止事項の「各スキルの出力先」に当たるものとして、この候補に限り、memory の置き場と `.scratch/anet-tidy-memory/` への書き込みを認める旨を書く。
- Codex では前提条件を満たさないので選ばれない。

### 5.4 skill の手順（`anet-tidy-memory [full]`）

1. Codex で起動されたら、D2 の報告をして終える。memory の置き場を D1 で決める。決められなければ止まる。
2. `run_id`（`<YYYY-MM-DD>-claude-<HHMM>`。housekeeping 経由ならその `run_id`）を決め、`backup` で写してから、D6 の置き場の `state.json` を渡して `scan` を回す。
3. D8 の範囲を読み、§5.1 の表で直す案をまとまりごとに作る。
4. 明示起動なら案を示して了承を得る（D7）。housekeeping 経由なら質問せず、「正誤が決まるものだけ」の範囲にする。
5. 直す。まとまりごとに §5.2 の判定で当て、索引は最後に 1 回、読み直してから当て直す（D5）。
6. `scan` を回し直し、次を確かめる。索引が目標以下であること。対応の漏れ（指す先の無い行、索引に無いファイル）が 0 であること。索引に状態の語が 0 であること。注記の無いパス切れが 0 であること。対応の漏れがあれば、索引を読み直して §5.1 の「索引と topic file の対応」の直しを 1 回だけ当て直し、もう一度 `scan` で確かめる。満たせない項目が残っても止まらずに 7・8 へ進み、満たせなかった項目とその原因（当てなかったまとまり、ほかの会話が足した行など）を `report.md` と報告に書く。housekeeping 経由なら、`runs.jsonl` の `outcome` を `partial` にする。
7. `record` を回し、`report.md` を書く。6 を満たせなかったときも回す。写しは `before/` に残り、索引が目標を超えていれば次の housekeeping で前提条件に当たって再び選ばれる。housekeeping 経由なら `runs.jsonl` に件数だけの `note` を書く（D6）。
8. 報告する。件数、正本なしの一覧（topic file の名前と 1 行）、当てなかった直し、パス切れ、戻し方（`before/` から写し戻す）。「当てなかった直し」は直しの単位で挙げる（範囲の外として残したまとめ・引退、§5.2 の判定で当てなかったまとまり）。同じ topic file に当てたほかの直しがあっても、当てなかった直しはその中身が残る。

### 5.5 移行対象と文書の更新

新しい契約への移行は無い。同じ変更で足す・直すものは次のとおり。

- `.agents/skills/anet-tidy-memory/`（`SKILL.md`、`agents/openai.yaml`。暗黙起動は無効）と `.claude/skills/anet-tidy-memory/`（同じ中身）。
- `.agents/skills/anet-housekeeping/SKILL.md` と `.claude/skills/anet-housekeeping/SKILL.md`。候補表、既定の優先順、禁止事項、`runs.jsonl` の `note` の説明を直す。あわせて、frontmatter の `description` と `agents/openai.yaml` の `short_description`（両方）を、`memory-tidy` を含む今の候補表に合わせる。この 2 か所は起票時点で候補表とずれている（`description` に wiki-lint が無い。`short_description` に wiki-lint・harness-map・atlas が無く、候補表に無い「コミット案」がある）ので、同じ変更で揃える。
- skills README（両方）。表の行、「anet-housekeeping が呼ぶ候補と対応」、共通の約束を直す。共通の約束には「`anet-tidy-memory` だけは memory の置き場と `.scratch/anet-tidy-memory/` へ書き、`reports/` と `docs/` に memory の中身を書かない」を足す。
- `reports/tools/memory_support.py`（新規）。
- 検証資材 `core/anet-core/testdata/prd112/`（`README.md`、`memory_support_test.py`）。
- [160 アプリケーションとツール](../design/160_applications_and_tools.jp.md)。§7.4 の隣に memory の整備の節を足し、§8 に検証資材への 1 行を足す。英訳（`.en.md`）は anet-translate-docs が追う。
- 動線マップ（`docs/agents/`）は、`.agents/skills` の変化を anet-harness-map の前提条件が拾って作り直すので、この変更では直さない。
- `reports/housekeeping/runs.jsonl` の既存の行は書き換えない。新しい候補の行を足すだけである。

### 5.6 性能と資源

- 1 回の読み込みは、索引（約 29KB）と、差分の topic file と、状態の語・パスを含む topic file である（D8）。初回と `full` は全部（約 674KB）を読む。
- 正本との照合は、topic file が書く repo のパスの節だけを読む。
- 実行前の写しは 1 回約 700KB である。`.scratch/anet-tidy-memory/` の古い写しは消さない（消すかは §9 の 8）。

### 5.7 クラス一覧

対象外（C++・Java のクラスを足さない、変えない、消さない。足すのは skill の文書と Python の補助スクリプトだけである）。

## 6. 細部（暫定で置いた既定）

1. 索引の目標: 160 行・20,000 バイト（上限の 8 割）。理由: 次の実行までに足す分の余地を残し、Claude Code の「上限に近い」知らせ（§2 の 1）を待たずに直す。
2. 1 行の目安: 150 字。理由: `consolidate-memory` の目安（§2 の 6）。
3. 定期の間隔: 7 日。理由: housekeeping の stats と同じ周期。10/24〜11/02 の 10 日に更新された project 型の topic file のうち、17 本が数値の知見を持っていた（§2 の 4）。
4. 状態の語（git とフォルダで答えが出るもの）: 未コミット、コミット済み、ステージ済み、未ステージ、未実装、実装済み、実装完了。理由: §2 の 3 で古くなっていた語と、その反対の語。
5. housekeeping の初期見積: 10% / 20 分。理由: wiki-lint（10% / 15）と同じ程度の読み込み量で、写しと照合の分を足した。`runs.jsonl` の実測で較正する。
6. 優先順: stats、wiki-lint の次。理由: §5.3。
7. 記録の置き場と `run_id` の形: §5.4 の 2 と D6。理由: housekeeping の `run_id` と揃える。

## 7. 検証と受入

1. `diff -rq .agents/skills .claude/skills` が差を出さない。skills README の表（両方）に `anet-tidy-memory` の行があり、「anet-housekeeping が呼ぶ候補と対応」に `memory-tidy → anet-tidy-memory` がある。`.agents/skills/anet-tidy-memory/agents/openai.yaml` があり、`policy.allow_implicit_invocation: false` である。
2. `memory_support_test.py`（一時フォルダに作った試験用の置き場で行う。本物の memory を使わない）が通る。確かめるのは次のとおり。
   - 索引の行数・バイト数と、上限・目標を超えたかの判定。
   - 指す先の無い行と、索引に無い topic file。
   - 状態の語の行番号。
   - repo のパスの実在と、同じファイル名の候補。
   - `--state` による変わったファイルの検出。`--state` のファイルが無いときは終了コード 0 で前回の記録が無いことを示し、全部の topic file を変わったものとする。壊れた `state.json` では終了コード 1。
   - `backup` の写しと `hashes.json`。`--out` が既にあるときは何も書かずに終了コード 1。
   - `scan` の前後で試験用の置き場の sha256 が変わらないこと。
   - 置き場が無いときは、`scan`・`backup`・`record` のどれも何も書かずに終了コード 1。
   - `record` に空の `--run-id` を渡すと、何も書かずに終了コード 2。
3. 本物の memory で `scan` を回すと、終了コード 0 で、索引の行数・バイト数がその時点の `wc -l`・`wc -c` と一致する。状態の語の行に、その時点で「未コミット」を含む索引の行がすべて含まれる。前後で置き場の sha256 が変わらない。
4. Claude Code で `/anet-tidy-memory full` を回し、「全部直す」を選んだ後に次を満たす。
   - `.scratch/anet-tidy-memory/<run_id>/before/` の `hashes.json` が、実行前の置き場と一致する。`before/` と `state.json` は主 worktree の下にある（D6。worktree から起動したときも同じであることを、人が確かめる項として実装計画に記録する）。
   - `scan` で、索引が 160 行以下・20,000 バイト以下、対応の漏れが 0、索引の状態の語が 0、注記の無いパス切れが 0。
   - `report.md` の「ポインタにした」の各行に正本のパスと節があり、パスが実在する。人がそのうち 3 件を `before/` の中身と見比べ、写していた数値が正本の節にあることを確かめる。
   - `git status --porcelain` の出力が実行の前後で同じである（`.scratch/` は管理外）。
5. Codex で `$anet-tidy-memory` を回すと、D2 の報告をして終わり、置き場の sha256 が前後で変わらない。
6. housekeeping の選定。
   - Claude Code で、`state.json` が無い状態で `/anet-housekeeping dry-run` を回すと、選定結果に `memory-tidy` が出る。
   - 受入 4 のあと、索引が目標以下で 7 日たたないうちに `dry-run` を回すと出ない。
   - Codex の `$anet-housekeeping S dry-run` では出ない。
7. housekeeping 経由の実行（`only=memory-tidy`）。
   - 準備: 受入 6 のあと、D6 の置き場の `state.json` を別名へ退避し、前回の記録が無い状態にする（前提条件の「`state.json` が無い」に当たり、D8 により全部を読む）。そのうえで `/anet-housekeeping only=memory-tidy` の実行案を了承して回す。終わったら退避したファイルは消す。
   - 報告の「当てなかった直し」に挙げた直しが当たっていない。まとめで消すはずだった側の topic file と、引退の候補の topic file は、実行後も置き場にあり、索引の行も残っている。ただし、同じファイルが範囲の内の直し（規約へ全部が移った feedback の削除）で消えたときは除く（§5.2）。報告の正本なしの一覧に挙げた知見の文は、実行後の topic file にそのまま残っている（同じ topic file にほかの直しが当たり、sha256 が変わっていてもよい）。
   - `runs.jsonl` に足された行の `note` が件数だけで、topic file の名前と本文を含まない。
8. 実行中にほかの会話が書いたものは上書きしない（D5、§5.2）。受入 4 で、直す案が示されて了承を待っている間（`backup` と `scan` のあと、どのまとまりも当てる前）に、人が別の会話で次の 2 つを行う。
   - (a) 直す案に入っている topic file のうち、そのまとまりに索引の行の直しが伴わないもの（まとめ・引退・移し終えた feedback の削除と、索引の行を短くする直しの先のどれにも当たらないもの）を 1 本選び、中身を書き換える。
   - (b) 索引に 1 行足す。その行は、実在する topic file を指し、状態の語（§6 の 4）を含まず、repo のパスを含まない。
   - そのあと「全部直す」を選ぶ。期待値は次のとおり。
     - (a) のファイルを含むまとまりは当たらず、報告の「当てなかった直し」に「scan のあとに変わった」として挙がる。(a) のファイルの中身は書き換えた後のままである。
     - (b) の行は実行後の索引に残る。
     - ほかのまとまりの直しは当たっている。
   - 受入 8 を行った受入 4 の実行では、受入 4 の `scan` の期待値を次のように数える。索引の行数とバイト数は (b) の行を除いて数える。状態の語、注記の無いパス切れ、対応の漏れは、(a) のファイルを含むまとまりに当たる行と (b) の行を除いて数える。
   - 人が確かめる項として実装計画に記録する。

## 8. 作らないもの・先送り

| 案 | 裁定 | 理由と再開条件 |
|---|---|---|
| Codex の memory の整備（段階 2） | 先送り | Codex の memory は Codex アプリが更新し、PRD095 D38 で読むだけと決めた。書き換えてよいか（アプリが作り直すか、どのファイルなら手で直せるか）を確かめていない。再開条件: 段階 1 を 2 回以上回して §3.4 の数を見たうえで、Codex の memory の書き換えの可否を確かめたとき。そのとき比べる形は 2 つある。読むだけで「正本へ移す候補」と「Claude の memory との食い違い」を挙げる形と、同じ整備を Codex の memory に行う形である |
| Auto-dream に任せる | 先送り | 公式に挙動と設定の記述が無く（§2 の 6）、anet の規則（D3・D4）を渡せるか分からない。再開条件: 公式に挙動・設定が文書化されたとき。併用は妨げず、同時の書き込みは D5 の確かめで上書きしない |
| `consolidate-memory` をそのまま定期に回す | 不採用 | git・フォルダ・正本・規約と照合しないので、古い状態の記述をそのまま整えて残し、知見を memory の中でまとめて二重持ちを固める（問題 2・3） |
| anet-lint-wiki に memory の点検を足す | 不採用 | wiki の点検の材料は repo の中の文書だけという契約で（PRD095 §4.7）、前提条件も材料と blob の記録だけで決まる。memory は repo の外にあり、Claude Code からしか見えない。関心が別である |
| anet-improve の memory の扱いを広げる | 不採用 | 対象がその会話の手戻りで、起動のきっかけ（明示起動か PRD 工程の完了報告）も定期の棚卸しと違う |
| memory を書くたびに hook で検査する | 不採用 | 古くなる原因はコミットと時間の経過で、書いた時点では正しい。大きさの知らせは Claude Code が書き込みのたびに出す（§2 の 1） |
| memory の置き場を repo の中へ移し、Codex と共有する（`autoMemoryDirectory`） | 範囲外 | 公開 repo に私的な memory が入る。共有の仕方は段階 2 と一緒に決める。再開条件: 段階 2 のグリル |
| 正本なしの知見を、整備の中で正本へ起こす | 範囲外 | 値を Run で確かめる作業で（§2 の 5）、実験記録・PRD の工程に属する。再開条件: 正本なしの一覧が 3 回続けて減らないとき、一覧の扱いを運用として見直す |

## 9. 未決

| # | 事項 | 置いた既定 |
|---|---|---|
| 1 | 索引に状態の語を置かない（D4）。進み具合を索引で一目で見たい用途があるか | 置かない。topic file に確かめた日つきで残す |
| 2 | 索引の目標の大きさ | 160 行・20,000 バイト（§6 の 1） |
| 3 | 定期の間隔 | 7 日（§6 の 3） |
| 4 | 明示起動の了承の選択肢 | 「全部直す」「正誤が決まるものだけ直す」「やめる」（D7） |
| 5 | 引退の基準と、無人で引退させるか | `done/` の PRD だけを指し、残す教訓の無い project 型を、明示起動でだけ引退させる（§5.1） |
| 6 | 上限の数え方（バイトか文字か）と、超えた分の実際の扱い。起票時の会話では、28,845 バイトの索引の末尾まで読めていた | 安全側にバイトで数える（25,000 バイト）。実装時に、索引の末尾に目印の行を置いて新しい会話で見えるかを確かめ、結果を実装計画に書く |
| 7 | 正本なしの一覧を毎回ユーザーに聞くか | 聞かずに報告に挙げるだけにする。扱いはユーザーが別に決める |
| 8 | `.scratch/anet-tidy-memory/` の古い写しを消すか | 消さない。大きくなったら人が消す |
| 9 | 補助スクリプトを置くか（skill の中のコマンドだけにするか） | 置く（D9） |
| 10 | 名前 | D11 |
| 11 | 前回の記録と写しの置き場を、主 worktree の下の 1 か所にするか、worktree ごとにするか | 主 worktree の下の 1 か所（D6） |
| 12 | 同時の書き込みで当てなかったまとまりを、次の実行で試し直す手当て | 特別な記録は持たない。次の実行の `scan` で同じ食い違いがまた見つかれば、そのとき直す |

## 10. 依存と順序

- 先に要る PRD は無い。PRD095 の wiki と anet-lint-wiki の上に乗る。
- anet-housekeeping の候補を足すので、anet-housekeeping の SKILL.md（両方）と同じ変更で入れる。
- 段階 1 を 2 回以上回してから、段階 2（Codex）をグリルする（§8）。

## 11. 参照

- [PRD095 LLM-Wiki](done/095_llm_wiki_10prd.md)、[段階 3 の記録](done/095_llm_wiki_22impl.md)、[ADR 0056](../adr/0056-wiki-is-derived-summary-and-readme-keeps-original-analysis.md)
- [anet-housekeeping](../../.agents/skills/anet-housekeeping/SKILL.md)、[anet-improve](../../.agents/skills/anet-improve/SKILL.md)、[anet-lint-wiki](../../.agents/skills/anet-lint-wiki/SKILL.md)、[skills README](../../.agents/skills/README.md)
- [wiki の規約](../wiki/README.md)、[311 開発ツール（wiki）](../wiki/categories/311_dev_tools.md)、[160 アプリケーションとツール](../design/160_applications_and_tools.jp.md)、[PRD095 の検証資材](../../core/anet-core/testdata/prd095/README.md)
- [How Claude remembers your project](https://code.claude.com/docs/en/memory)、[Claude Code power user tips](https://support.claude.com/en/articles/14554000-claude-code-power-user-tips)、[Dreams](https://platform.claude.com/docs/en/managed-agents/dreams)、[LangMem Background Quickstart](https://langchain-ai.github.io/langmem/background_quickstart)
