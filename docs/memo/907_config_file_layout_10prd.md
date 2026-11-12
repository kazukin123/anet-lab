# 設定ファイルの配置（Agent 全体・具象 Agent・機能 include・フォルダ）暫定 PRD

> 2026-11-03 番号を 999 から 907 へ振り直した（PRD114）。

> 2026-10-24 起票。暫定の既定を置き、決めることを §8 の表に残した。グリル前で、最終簡素化パスは経ていない。行番号・行数は起票時点のもので、実装時は近傍のキー名・ファイル名で探し直す。
> 起点: [PRD098](098_temporl_skip_policy_10prd.md) の実装で `apps/runner/config/skip.txt` が増え、[020 §3.6](../design/020_user_guide_run.jp.md#36-どのファイルに何を書くか) の表に無い種類のファイルが生まれた。2026-10-23 の会話で、むやみにファイルを増やさない方針はそろそろ限界で、「`agent.txt` は Agent 全体、具象 Agent 別のファイル、そこから機能別のファイルを include」が自然だと裁定した。
> 関連: [999 設定 root 所有](908_config_root_ownership_alignment_10prd.md)（root 名と所有者。本書はファイルの置き場だけを扱う）、[999 ソース配置](957_source_layout_10prd.md)（ソースも具象 Agent 別に分ける。設定の層と同じ単位）、[059](done/059_config_concept_tree_alignment_10prd.md)・[072](done/072_config_selection_final_value_10prd.md)（概念と解決規則。変えない）。

## 1. 目的と問題

新しいキーをどのファイルに書くかを、「Agent 全体 → 具象 Agent → 機能」の 3 層で 1 行で答えられるようにする。

今は次の状態である。

- 規約の正本は [020 §3.6](../design/020_user_guide_run.jp.md#36-どのファイルに何を書くか) の表（ファイルの役割と演算子）と [impl.md の「設定ファイル」](../quality/impl.md#設定ファイル)（演算子とプロファイル境界）だけで、Agent の層も機能の層も無い。
- `agent.txt`（413 行）に MuZeroAgent・DefaultDQNAgent・RainbowAgent・ImageClsAgent の 4 つが節で同居する。DefaultDQNAgent の節は 59〜301 行で、うち約 170 行は NN 実行設定の説明である。
- `common.txt` に `DefaultDQNAgent.@fast` / `@balance` / `@heavy`（39〜48 行）が紛れている。表では切替プロファイルは `agent.txt` の役割である。
- PRD098 の `skip.txt`（44 行）は、1 つの機能のために Agent のプロファイル・`net.block`・`metrics.scalar.@skip`・`run.@skip*`・`app.eval_trace_panel.columns` の 5 つの関心を 1 ファイルに置き、`_main.txt` が読む。`run.@` を共通ファイルで定義するのも表の外である。表では `run.@` は `<Env>.txt` の所有で、実際も Atari 161 本・DropMerge 84 本・LunarLander 14 本が env 側にあり、`agent.txt` には 1 本も無い。
- 同じ機能系の先例は関心ごとの分散である。Munchausen は `DefaultDQNAgent.@munchausen`（agent.txt 82〜84 行）、`metrics.scalar.@munchausen`（metrics_scalar.txt 303〜313 行）、`run.@munch`（Atari・LunarLander・DropMerge）。PRD098 §5.6 の設定例も `DefaultDQNAgent.@skip` を Agent 側に置き、env の `run.@skip` がチェーンを書き直して選ぶ形である（チェーンへ項を足す記法は無く、`X.$ += …` と書くと `X.$+` という葉になって黙って無視される。2026-10-24 に `skip.txt` の `run.@skip_concat` で判明。§8 D12）。

PRD098 の skip は、本書とは別に案 A（関心ごとに既存ファイルへ分散。[PRD098 §5.6](098_temporl_skip_policy_10prd.md) の設定例の形）で戻す。本書が通れば機能ファイルへ 2 度目の移動になるが、44 行である。

## 2. 現状（2026-10-24 の調査）

### 2.1 ファイルと行数

| 種別 | ファイル（行数） |
|---|---|
| 共通 | `_main.txt` 8（include 一覧）、`common.txt` 169、`agent.txt` 413、`nn.txt` 382、`nn_cnx.txt` 214、`metrics_scalar.txt` 366、`metrics_image.txt` 115、`skip.txt` 44（PRD098） |
| env | `Atari.txt` 1,429、`DropMerge.txt` 1,694、`ImageCls.txt` 699、`LunarLander.txt` 480、`GridMaze.txt` 247、`GridMaze_muzero.txt` 258、`CartPole.txt` 86、`DropMerge_optuna.txt` 11、`LunarLander_cpu.txt` 2 |
| workspace | `_workspace_template.txt` 11（workspace の `config/_main.txt` の雛形。env の include だけ） |

### 2.2 規約の正本

[020 §3.6](../design/020_user_guide_run.jp.md#36-どのファイルに何を書くか) の表は、`_main.txt`（include 一覧）、`common.txt`（trainer・agent・env・gui・backend・app の共有既定値）、`agent.txt`（各 Agent の `@baseline`、切替プロファイル、`<Agent>.$`）、`nn*.txt`（共有 NN 部品）、`metrics_*.txt`（カタログ）、`<Env>.txt`（チェーン、プロファイル、上書き層、Run プロファイル、NN 配線、metrics 差分）、workspace の `config/_main.txt`（env の include）、bat・CLI（`run.$` の選択と葉の上書き）の 8 行である。演算子は、共通ファイルのベース定義が `?=`、切替プロファイルとチェーンが `=`、env ファイルはデフォルト設定だけ `?=`。[impl.md の「設定ファイル」](../quality/impl.md#設定ファイル)は演算子とプロファイル境界（`:`）の規約で、ファイル別の一覧は 020 を正本にしている。

### 2.3 include の解決

[config.cpp:191〜207](../../core/anet-core/src/config.cpp) は、絶対パス → 含むファイルの隣（親ディレクトリ） → 設定の検索ディレクトリの順に解決する。深さは 10 まで（549 行）。同じフォルダなら `$include <name.txt>`、サブフォルダなら `$include <sub/name.txt>` で届くので、フォルダ分けに仕組みの変更は要らない。

### 2.4 `agent.txt` の節

MuZeroAgent 7〜58 行、DefaultDQNAgent 59〜301 行（ALGO プロファイル 73〜80 行、TARGET 81〜、BF16 87〜、Random Policy 102〜、NN 実行設定 130〜259、方策の設定ガイドライン 260〜）、RainbowAgent 302 行〜、ImageClsAgent。行数は `DefaultDQNAgent.@baseline` 127、`RainbowAgent.@baseline` 46、`MuZeroAgent` 44、`ImageClsAgent` 7。73〜74 行の注記「切替はチェーンの @qr / @iqn の 1 項だけ。配線実体 net.@qr / net.@iqn は各 env ファイルが所有する」が、Agent 側のプロファイルと env 側の配線の分担の先例である。

### 2.5 `metrics_scalar.txt` のプロファイル

`@full` 113 行、`@baseline` 90、`@gradnoise` 15、`@replay_fit` 13、`@noisy` 11、`@iqn_search_p0` 10、`@munchausen` 6、`@min` 5、`@attention` 4。機能ごとのプロファイルがカタログに並ぶ形が先例で、群番号（`50_agent_skip` など）の衝突は 1 ファイルに並んでいるから見つかる（PRD098 の起草で 53 の衝突を見つけた）。

### 2.6 道具

- [check_default_leaves.py](../../core/anet-core/testdata/prd072/): 演算子の規約検査。
- 7 env の解決済み設定（全キーと文字列値）の前後比較。PRD098 の実装で `config-equivalence.json` として使った（[prd098 の README](../../core/anet-core/testdata/prd098/README.md)）。
- Run の `json/config_data.json` と `config/config_data.txt`（Run 設定の正本）。

## 3. ゴールと非ゴール

ゴール:

1. 3 層の規約を 020 §3.6 の表に置き、新しいキーの置き場を 1 行で答えられる。
2. `agent.txt` は Agent 横断だけ。具象は `<Agent>.txt`、機能は `<Agent>_<feature>.txt` を具象から include する。
3. 移行で解決済み設定を変えない。

非ゴール: キー名と root 名（999 設定 root 所有）、解決規則（072）、env ファイルの中身の整理（[061 P3](done/061_eval_slot_policy_override_10prd.md)）、metrics カタログの分割（§7）。

## 4. 提案の骨子（暫定の既定）

1. 層。

   | 層 | ファイル | 中身 | 演算子 |
   |---|---|---|---|
   | L0 | `common.txt` | app・backend・run・agent スロット・env スロットの共有既定。Agent のプロファイルは置かない | `?=`（切替は `=`） |
   | L1 | `agent.txt` | Agent 横断の内容だけ。無ければ `_main.txt` から外す（§8 D7） | `?=` |
   | L2 | `<Agent>.txt`（`DefaultDQNAgent.txt`、`RainbowAgent.txt`、`MuZeroAgent.txt`、`ImageClsAgent.txt`） | その Agent の `@baseline`、切替プロファイル、`<Agent>.$ = @baseline`、NN 実行設定、説明。機能ファイルの `$include` | `@baseline` は `?=`、切替とチェーンは `=` |
   | L3 | `<Agent>_<feature>.txt`（例 `DefaultDQNAgent_skip.txt`） | その機能のプロファイルと、その機能専用の `<Agent>.<sub>.net.@*` の配線・`net.block.[…]` 部品 | 既定は `?=`、切替は `=` |

2. 置かないもの。`run.@` は `<Env>.txt`（J のような env ごとの値もそこ）。metrics のプロファイルは `metrics_*.txt`（群番号の衝突を 1 か所で見るため）。app の既定（trace 列など）は `common.txt`（切替は `app.@<name>` の形。§8 D8）。共有 NN 部品は `nn*.txt`。
3. 名前。ファイル名は設定の prefix と一致させる（`DefaultDQNAgent.txt`）。env ファイルが Env 名（`GridMaze.txt`）なのと同じ規則として表に書く。機能ファイルは `<Agent>_<feature>.txt`。
4. フォルダ。平ら（`apps/runner/config/` 直下）。フォルダ化は §7。
5. `_main.txt` は L0〜L2 を include する。L3 は L2 から include する。`_workspace_template.txt` は変えない。
6. 演算子の規約は変えない。L2・L3 は共通ファイル扱い。
7. 移行。`common.txt` の `DefaultDQNAgent.@fast/@balance/@heavy` を `DefaultDQNAgent.txt` へ。`agent.txt` を 4 つに分ける。PRD098 の skip（案 A の後の形）を `DefaultDQNAgent_skip.txt` へ。1 回で行い、§6 の検査で解決値の不変を確かめる。
8. 文書。020 §3.6 の表の `agent.txt` の行を L1・L2・L3 の 3 行にし、名前の規則を足す。impl.md の「設定ファイル」は演算子だけを持ち、配置は 020 を指す。CONTEXT.md に新語は足さない（§8 D9）。

設定例（コロン記法）:

```
# apps/runner/config/DefaultDQNAgent.txt
$include <DefaultDQNAgent_skip.txt>
DefaultDQNAgent.@baseline : learner.gamma ?= 0.99
DefaultDQNAgent.$ = @baseline

# apps/runner/config/DefaultDQNAgent_skip.txt
DefaultDQNAgent.@skip : skip.enabled = true
DefaultDQNAgent.@skip : net.body.output.[skip_context] = main_feature
DefaultDQNAgent.@skip : skip.net.$ = skip.net.@base
DefaultDQNAgent.skip.net.@base : branch.[skip_feature].bind ?= skip_context

# apps/runner/config/GridMaze.txt（チェーンは env が所有するので書き直す。追加記法は無い）
run.@skip : DefaultDQNAgent.$ = @baseline > @fast > A1 > A2 > A3 > @skip
run.@skip : DefaultDQNAgent.skip.max_length = 7
```

## 5. 機構ごとの根拠

| 機構 | 根拠（起きたこと・測った量） | 裁定（暫定） |
|---|---|---|
| 3 層の表 | skip.txt の置き場を表で答えられず、2 点（機能ファイル、共通の `run.@`）が表の外になった | 作る |
| 機能ファイルの境界（プロファイルと専用の配線・部品だけ） | metrics を同居させると群番号の衝突が見えない。`run.@` を同居させると env 所有の規則が崩れる。先例（Munchausen）は分散 | 作る |
| metrics をカタログに残す | PRD098 で群番号 53 の衝突を 1 ファイルで見つけた | 残す |
| 平らなフォルダ | include は隣解決で届く。`_workspace_template.txt`・検索・設計文書の相対リンクがそのまま | 平ら |
| `common.txt` の Agent プロファイルの移動 | 表の規約（切替は `agent.txt`）と食い違ったまま 3 本ある | 動かす |
| 1 回の移行 | 機械的な移動で、7 env の解決値一致で守れる | 1 回 |

## 6. 受入基準

1. 7 env（`_workspace_template.txt` が include 候補に挙げる env ファイル）の解決済み設定が、全キーと文字列値で移行前後一致する。
2. `check_default_leaves.py` が errors=0。
3. 既存の bat と CLI（`run.$` と葉の上書きだけ）が変更なしで同じ `config_data.txt` を出す。
4. 020 §3.6 の表で、PRD098 の全キー（`skip.*`・`learner.skip.*`・`actor.[key].skip.policy.*`・`metrics.scalar.@skip`・`run.@skip*`・trace 列）の置き場が 1 行ずつ答えられる。
5. `skip.txt` は残らない。`_main.txt` の include は L0〜L2 だけ。
6. 設計文書から設定ファイルへの相対リンクが切れていない。

## 7. 先送りした段階と、作る条件

- フォルダ化（`DefaultDQNAgent/` に L2・L3 を入れる、env を `envs/` へ）: 条件は、具象 Agent が 6 以上、または 1 Agent の機能ファイルが 4 本を超えたとき。include は隣解決で届く（§2.3）ので仕組みの変更は要らない。
- metrics カタログの Agent 別分割: 条件は、群番号の衝突を機械的に検査する道具ができたとき。
- `nn.txt`・`nn_cnx.txt` の分割: 条件は、共有 NN 部品が Agent 固有に偏ったとき。
- env ファイルの中身（Run プロファイルの整理）: [061 P3](done/061_eval_slot_policy_override_10prd.md) の別グリル。

## 8. 決めること（暫定の既定）

| # | 決めること | 暫定の既定 | 代案と退けた理由 |
|---|---|---|---|
| D1 | L2 の名前 | 設定の prefix と一致（`DefaultDQNAgent.txt`） | `agent_dqn.txt`: env ファイル名（Env 名）の規則と揃わない |
| D2 | 機能ファイルの名前 | `<Agent>_<feature>.txt` | `<feature>.txt`: 所属が名前で分からない |
| D3 | 機能ファイルの境界 | プロファイルと、その機能専用の配線・部品。metrics・`run.@`・app は含めない | metrics も同居: 群番号の衝突が見えない |
| D4 | 平らかフォルダか | 平ら | フォルダ: §7 の条件まで増やさない |
| D5 | Rainbow の置き場 | `RainbowAgent.txt`（独立の具象。`@baseline` 46 行） | `DefaultDQNAgent.txt` に同居: 具象 1 つに 1 ファイルの規則が崩れる |
| D6 | `common.txt` の Agent プロファイル 3 本 | `DefaultDQNAgent.txt` へ | 残す: 表の規約と食い違う |
| D7 | L1 `agent.txt` の中身 | 横断の説明と `net` 6 行の行き先を決め（`net` は `nn.txt` へ）、中身が無ければ `_main.txt` から外す | 空ファイルで残す: 役割の無いファイルになる |
| D8 | trace 列のような app の切替 | `app.@skip_trace` を `common.txt` に置き、`app.$ += app.@skip_trace` で選ぶ | env の `run.@` に `app.eval_trace_panel.columns.[30] = …` を書く: 4 env に同じ 2 行が要る。`app.$` のチェーンにプロファイルを足せるかは要確認（§10） |
| D9 | 用語 | CONTEXT.md に足さない。「機能ファイル」は表の中の一般語 | 定義を足す: 一般語で通じる |
| D10 | 移行の単位 | 1 回 | DQN から順に: 途中の状態が表と食い違う期間ができる |
| D11 | 文書の正本 | 020 §3.6 の表。impl.md は演算子だけ | impl.md に配置も書く: 二重管理 |
| D12 | 機能プロファイルを Run プロファイルから選ぶ書き方 | env の `run.@<feature>` がチェーンを書き直す（Atari の `run.@munch` の先例）。resolver に追加記法は足さない | `X.$ += item` を resolver に足す: 末尾に足す意味だけなら小さいが、order の規約（上書き層の後ろに来てよいか）を決める必要がある。`X.$+` のような読まれない葉を `check_default_leaves.py` で拒否する検査は別途足す |

## 9. 依存と順序

1. PRD098 の案 A（`skip.txt` の関心ごとの分散）が先。設定のウォークスルーの後に、設定関連の見直しとしてまとめて依頼する（2026-10-23 の裁定）。
2. 本書はグリル後に実装する。[999 ソース配置](957_source_layout_10prd.md)と同じ「具象 Agent」の単位を使う（設定の L2・L3 とソースの `src/agents/<agent>/`）。順序は独立。
3. [999 設定 root 所有](908_config_root_ownership_alignment_10prd.md)は root 名の話で、本書の前でも後でもよい。

## 10. 再確認ポイント

- 行番号・行数（2026-10-24）。`agent.txt` の節境界、`common.txt` の Agent プロファイル。
- 7 env 比較の道具の所在と作り方（PRD098 の `run_validation.py` が `resolved-before/<env>.json` を作った手順）。
- `app.$` のチェーンにプロファイルを足せるか（D8）。足せても書き直しであり、追加記法は無い（D12）。
- `check_default_leaves.py` が `X.$+` のような読まれないキーを検出するか（D12）。

## 11. 参照

- [020 §3.6 どのファイルに何を書くか](../design/020_user_guide_run.jp.md#36-どのファイルに何を書くか)
- [impl.md 設定ファイル](../quality/impl.md#設定ファイル)
- [config.cpp](../../core/anet-core/src/config.cpp)（include の解決）
- [PRD098 §5.6](098_temporl_skip_policy_10prd.md)、`apps/runner/config/skip.txt`、`agent.txt`、`common.txt`
- [999 設定 root 所有](908_config_root_ownership_alignment_10prd.md)、[999 ソース配置](957_source_layout_10prd.md)
- [059](done/059_config_concept_tree_alignment_10prd.md)、[072](done/072_config_selection_final_value_10prd.md)、[061](done/061_eval_slot_policy_override_10prd.md)
