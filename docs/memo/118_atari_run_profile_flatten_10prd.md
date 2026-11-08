# PRD118: Atari の Run プロファイルの整理（本流を 1 つの平らなプロファイルにする）

> 2026-11-08 起票（暫定。質疑を省き、既定を §8 の未決表に置いた）。行番号は起票時点の作業ツリーの位置で、`apps/runner/config/Atari.txt` と `apps/11_batch_run.bat` は未コミットの変更を含む。実装時は近傍の記述で探し直す。
> 起点: 2026-11-08 の会話。Montezuma の RND の Run を段取りする途中で、ユーザーが Atari.txt の Run プロファイルについて、「大量すぎてワケワカメ」「Run プロファイルを組み合わせて一つの設定にするのは無理があるのでは」「BTR 寄せのサンプル効率優先は本流のプロファイルとして用意しておきたいが、他プロファイルの組み合わせだと何が何だか分からない」「ベース部分は仕方ないかもしれないが、トップの Run プロファイルにフラットに設定を並べるなら人間でも分かる」「とにかく Atari.txt をきれいにしたい」と述べた。Atari を題材に整理の形を考え、その結果をほかの env へ広げるかは後で考えるとして、課題と暫定 PRD の起票を求めた。
> 題材: [issue-20261108-06a3f0](../management/issues/issue-20261108-06a3f0.md)。関連: [PRD907](907_config_file_layout_10prd.md)（設定ファイルの配置。§7 で「env ファイルの中身（Run プロファイルの整理）」を別のグリルへ送った）、[PRD096](096_metrics_step_axis_interval_10prd.md)（間隔を数える単位）、[PRD940](940_target_update_step_axis_10prd.md)（target 更新を数える step の軸）、[PRD059](done/059_config_concept_tree_alignment_10prd.md)・[PRD072](done/072_config_selection_final_value_10prd.md)（Run プロファイルと選択の最終値）、[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)・[ADR 0042](../adr/0042-config-inheritance-as-differential-base.md)。

## 1. 問題

### 1.1 本流の構成を 1 か所で読めない

Atari の Run は 2 つの本流の構成で回している。実時間優先は `Atari.txt` の既定の `run.$`（11 個の Run プロファイルの連結）、サンプル効率優先（BTR 寄せ）は 11 の bat の変数 `SE30`（14 個の連結）である。どちらにも名前の付いたプロファイルが無く、構成を知るには、連結した全部のプロファイルと、`Atari.txt` 本体の上書き層 A2 の行と、bat の引数を重ねて読む必要がある。連結の中では同じキーが何度も書き換わり、最後に効く値がどれかは並べ替えないと分からない（§2.3）。

### 1.2 並べる順の決まりがコメントにしかない

後ろのプロファイルが前のプロファイルの選択（`.$`）を丸ごと書き換えるため、連結の順を変えると結果が変わる。その決まりは `Atari.txt` のコメントにだけ書かれている（§2.4）。

### 1.3 値だけ違う腕と、使われないプロファイルが積もっている

72 個の Run プロファイルのうち 30 個は、1〜2 個の値だけを変える実験の腕である。どこからも参照されないものが 17 個ある（§2.2）。[実装 品質確認観点](../quality/impl.md) の「設定ファイル」の規約（174 行）は、機能ごとの変種のために `@` プロファイルを作らないとし、その理由を「プロファイルが増えるほど、Run で何が効いているかを追えなくなるため」としている。

### 1.4 learn step で数える値が、replay ratio ごとの別プロファイルを要する

評価の間隔（`run.eval_schedule.[tag].interval`）と target の同期（`model.hard_update_interval`）は learn step で数える。1 回の更新あたりの exp step は replay ratio（RR）で変わるので、RR ごとに別のプロファイル（`btreval_r1`／`r2`／`r4`、`eval2ch`／`eval2ch_r1`、`hard125`／`hard250`／`hard500`）が並んでいる（§2.5）。

### 1.5 同じキーを別名と正式名で書くと、片方が黙って効かない

Run プロファイルの葉は、上書き層の別名（`A2.learner.replay_ratio`）で書いたものと、正式名（`DefaultDQNAgent.learner.skip.target_window`）で書いたものが混ざっている。同じキーを両方の名前で書くと正式名の個別指定が勝つ（[CONTEXT.md](../../CONTEXT.md) 274 行の「個別指定」、ADR 0042、[020](../design/020_user_guide_run.jp.md) 249〜251 行の `AtariEnv.game` と `E1.game` の例）。bat で値を上書きするときは、プロファイルと同じ名前で書かないと黙って効かない。

## 2. 現状（2026-11-08 に確認したこと）

### 2.1 `Atari.txt` の構成

[Atari.txt](../../apps/runner/config/Atari.txt) は 1,658 行で、見出しは次の順である。

| 行 | 見出し | 中身 |
| --- | --- | --- |
| 2 | 基本設定 | RND・Rainbow 寄せなどの Run プロファイル（8〜120 行）、seed・lane 数、`DefaultDQNAgent.$`（135 行）、`AtariEnv.$`（141 行）、Network の選択、既定の `run.$`（156 行） |
| 162 | Run設定 | ゲームの選択（`E1.game`）など |
| 287 | ステップ数 依存設定 | 予算（`@vars`）と PER の β の段 |
| 322 | 実験 | 上書き層 A2 の行（RR 1・`adam_eps`・ε の梯子・評価の ε 0.01。337〜363 行） |
| 372 | Run プロファイル(確立済み) | Run プロファイルの大半（380〜706 行） |
| 708 | デバッグ用 | |
| 722 | Nature DQN 2015 再現構成 | |
| 754 | Env依存 Agent設定 | 上書き層 A1（Atari の既定） |
| 822 | NN | Network の部品 |
| 1318 | AtariEnv | env の約束（上書き層 E1、Greedy の評価の A3 など） |
| 1455 | scalar metrics | メトリクスの差分（M1） |
| 1622 | app | |

Run プロファイルは 72 個、葉は 267 行である。ほかの env の Run プロファイルの数は、DropMerge 8、LunarLander 7、GridMaze 2、ImageCls 2 である（各 `<Env>.txt` の `run.@` の名前を数えた）。

### 2.2 Run プロファイルの棚卸し

参照は `git grep` で数えた。bat は 11・12・13 の今の版、テストは `core`・`apps/runner`・`viewers`、文書は `docs/` である。12 の bat は DropMerge と LunarLander だけを使い、Atari のプロファイルを参照しない。

| 種類 | 数 | 名前 |
| --- | ---: | --- |
| 本流の部品（既定の `run.$` か、bat の本流の連結で使う） | 20 | v5_iqn_impala_x2、a5、a5_apex、va_base、hard125、hard500、rr4、munch、rfit、a5_metrics、cap2m、eval2ch、eval2ch_r1、eval2ch_r1_50m、btrnet、btrsn12、batch、to_10m、to_30m、to_100m |
| 機能を丸ごと入れる単位 | 22 | rnd、rnd_taiga、skip、skip_aux、no_munch、rainbow_like、rainbow_like_impala、evalonly、greedy_eval、eval2only、attention、plasticity、pl_check、pl_snmetrics、breakout_metrics、capall、nz_learning、nz_reference、nz_reference_off、pixel_area、vitva、vitbtr |
| 値だけ違う腕 | 30 | a5_rr4、a5_20m、to_400k、to_50、evaloff、seed2、evalN10、btreval_r1、btreval_r2、btreval_r4、gdist400、pool6、pool4、hard250、rr2、b512、aainit、alpha06、cap512k、cap4m、cap1536k、dp010、dp020、wd001、btrsn、btrpool4、btrtrunk、btrtrunknopool、btrtrunkpool4、envs64 |

bat・テスト・文書のどれにも名前が出ないものは 17 個ある: a5_rr4、btreval_r1、btreval_r2、btreval_r4、pl_snmetrics、breakout_metrics、pool6、hard250、rr2、b512、alpha06、dp010、dp020、wd001、btrpool4、btrtrunk、btrtrunkpool4。

`run.@rainbow_like`（75〜113 行、葉 37 行）は、Rainbow へ寄せた構成を 1 つのプロファイルに平らに並べた形で、ほかのプロファイルと連結せずに使う。

### 2.3 本流の組み立て

- 実時間優先: 既定の `run.$`（[Atari.txt](../../apps/runner/config/Atari.txt) 156 行）は `v5_iqn_impala_x2 > a5 > a5_apex > va_base > hard125 > munch > rfit > a5_metrics > cap2m > eval2ch_r1 > to_100m`。
- サンプル効率優先: [11 の bat](../../apps/11_batch_run.bat) 30〜35 行の変数を展開すると `v5_iqn_impala_x2 > a5 > a5_apex > va_base > hard500 > rr4 > munch > rfit > a5_metrics > btrnet > btrsn12 > cap2m > eval2ch > to_30m`。[13 の bat](../../apps/13_batch_run.bat) 28〜31 行も同じ変数を持つ。

サンプル効率優先の連結で、最後に効く値が 1 つなのに途中で書き換わるキー:

| キー | 書く場所（行） | 回数 |
| --- | --- | ---: |
| `DefaultDQNAgent.net.branch.[main_feature].$` | v5_iqn_impala_x2（382）、va_base（573）、btrnet（676）、btrsn12（681） | 4 |
| `DefaultDQNAgent.$` | v5_iqn_impala_x2（381）、munch（601）、btrnet（677） | 3 |
| `metrics.scalar.$` | munch（602）、rfit（606）、a5_metrics（609） | 3 |
| `A2.learner.replay_ratio` | 本体の「実験」（337）、a5（433）、rr4（595） | 3 |
| `@vars.max_exp_step` | 本体（304）、a5（427）、to_30m（453） | 3 |

このほか、bat の引数が τ の本数・Network の初期値（7 行）・評価の間隔を足す（11 の bat 40〜41 行と各ステージの行）。

### 2.4 並べる順の決まり

[Atari.txt](../../apps/runner/config/Atari.txt) のコメントにある決まり:

- 577 行「@va_base より後ろへ置く。main_feature の .$ を上書きするため。」（pool6・pool4）
- 604 行「metrics.scalar.$ は同じスロットなので @munch より後ろへ置くこと。」
- 608 行「@rfit より後ろへ置く。」（a5_metrics）
- 340 行「adam_eps は @bf16 … より後段でないと効かないので A1 ではなく A2 に置く。」（上書き層の順）

`metrics.scalar.$` を書く Run プロファイルが、重ねて使うプロファイルの群を落とさない決まりは、[実装 品質確認観点](../quality/impl.md) の「設定ファイル」の規約（178 行）と項番 68 にある。`metrics.scalar.$` を書く Atari の Run プロファイルは 13 個、`DefaultDQNAgent.$` を書くものは 9 個、`DefaultDQNAgent.net.branch.[main_feature].$` を書くものは 16 個ある。

### 2.5 learn step で数える値

- 評価の間隔: `Atari.txt` 488〜489 行のコメント「interval は learn_step 基準なので、1 更新あたり exp が変わる replay_ratio ごとに換算が要る」。`eval2ch`（510・513 行、RR 4 で 3,906・78,907）と `eval2ch_r1`（518・521 行、RR 1 で 1,954・39,453）が別にある。
- target の同期: 581〜586 行のコメント「env 時間 32,000 exp step で揃える」「RR と batch とセットで選ぶこと」。`hard125`／`hard250`／`hard500` が別にある。
- 単位の直しは [PRD096](096_metrics_step_axis_interval_10prd.md)（評価の間隔など）と [PRD940](940_target_update_step_axis_10prd.md)（target の更新）が扱う。どちらも未実装。

### 2.6 規約と過去の判断

- [CONTEXT.md](../../CONTEXT.md) 266 行: Run プロファイルは「Run を特徴づける環境・Agent・予算等の選択と、宣言済み値スロットの値の組を `run.@<name>` で命名したもの」。
- [CONTEXT.md](../../CONTEXT.md) 270 行: 上書き層 A1/A2/A3 は恒久度（大番号ほど一時的）を示す運用上の名前。
- [ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md): Run プロファイルの葉は CLI と同じ第 2 相で選択の後に勝つ。
- [実装 品質確認観点](../quality/impl.md) の「設定ファイル」の規約: 選択宣言・Run プロファイル・CLI は常に `=`（153 行）。機能の有効化は葉 1 つで済むようにし、機能ごとの変種のために `@` プロファイルを作らない（174 行）。
- [PRD907](907_config_file_layout_10prd.md) §7: `run.@` は `<Env>.txt` の所有。「env ファイルの中身（Run プロファイルの整理）」を別のグリルへ送った。
- [020 §3.6](../design/020_user_guide_run.jp.md) の表: `<Env>.txt` にチェーン・プロファイル・上書き層・Run プロファイルを書き、bat・CLI が `run.$` の選択と葉の上書きを持つ。137 行の例は `run.$=run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base`。
- [316 設定管理の地図](../wiki/categories/316_config.md) の「変えるときの条件」: 設定ファイルを分けたり移したりするときは、env の解決済み設定（全キーと文字列値）の前後一致と `check_default_leaves.py` の errors=0 を確かめる（出典は PRD907 §6）。

### 2.7 プロファイルの名前を参照するもの

| 種類 | 場所 |
| --- | --- |
| bat | [11](../../apps/11_batch_run.bat)・[13](../../apps/13_batch_run.bat)（本流の部品・機能・予算） |
| テスト（消える名前を書くもの） | `dqn_based_agent_test.cpp`（4370・9806〜9807 行）、`dqn_munchausen_test.cpp`（227・244 行）、`core/envs/atari1/src/AtariEnv_test.cpp`（1631 行） |
| テスト（残る名前と既定の `run.$` を読むもの） | [config_baseline_test.cpp](../../core/anet-core/src/config_baseline_test.cpp)（既定の `run.$` を読み、`run.@skip` などを足す）、`intrinsic_test.cpp`（`run.@rnd`・`run.@rnd_taiga`） |
| 固定入力 | `core/anet-core/testdata/prd072`（manifest の固定 commit を読む。[README](../../core/anet-core/testdata/prd072/README.md)） |
| 過去の PRD の検証資材 | `core/anet-core/testdata/prd084`・`prd094`・`prd097`・`prd099`・`prd102`・`prd103`・`prd104`・`prd106` のスクリプト・README・manifest。prd102 の manifest は今の設定を解決して比べる |
| 文書 | [020](../design/020_user_guide_run.jp.md)（jp・en）137 行、ADR 0040、PRD117・905・907・918・944（`run.@aainit`、72 行）、実装済みの PRD098、wiki の breakout、issue-20261103-93f8a6・issue-20261107-a9a0e3、実験記録 |
| 同じ名前の別 env のプロファイル | LunarLander.txt に `run.@munch`・`run.@rfit` がある（DropMerge.txt には無い）。12 の bat、[実装 品質確認観点](../quality/impl.md) 178 行の例、wiki の lunarlander はそちらを指す |
| workspace の設定（追跡外） | `apps/runner/workspaces/AtariRnd-01/config/rnd_explore.txt` は、先頭（6・7 行）で 2026-11-07 の写し `frozen/_main.txt`・`frozen/Atari.txt` を読み、`run.@rnd_nn` を定義する。11 の bat はこのファイルを `CFG`（36 行）にして `--config` で渡す（130 行）。13 の bat は workspace の `config/_main.txt`（`$include <Atari.txt>`）を通して今の `Atari.txt` を読む |

bat はステージを始めるたびに設定を読み直す。そのため、今の `Atari.txt` を読む bat（13）の後続のステージは、その時点の `Atari.txt` を読む。写しを読む bat（11）は、今の `Atari.txt` を変えても影響を受けないが、bat の名前だけを新しくすると写しに無い名前になる。`run.$` に無い名前を書くと、Run は起動時に `material selection target not found` で止まる（[config_test.cpp](../../core/anet-core/src/config_test.cpp) 965〜971 行）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

1. Atari の本流（サンプル効率優先・実時間優先）を、それぞれ 1 つの Run プロファイルを開けば構成が読める形にする。本流のプロファイルは、ほかのプロファイルとの連結の順に依存しない。
2. `Atari.txt` の Run プロファイルを、本流・機能・予算の 3 種類に減らし、どれも参照のあるものだけにする。
3. 整理の前後で、本流の Run の実効の設定（解決した全キーと文字列値）を変えない。
4. 題材の issue-20261108-06a3f0 を片付ける。

### 3.2 非ゴール

- ほかの env（DropMerge・LunarLander・GridMaze など）の設定ファイルの整理。Atari で形を確かめてから別に判断する（起点の会話の方針。§8 の 17）。本当の制約ではなく、同じ形を後から広げられる。
- 設定の文法や resolver の変更（チェーンへ項を足す記法など）。本当の制約ではなく、別の PRD で足すこともできるが、PRD907 の D12 で足さないとしており、本 PRD の形は今の文法で書けるので要らない。
- learn step で数える値の単位の変更。PRD096・PRD940 の関心である。本 PRD は今の単位のまま、本流ごとに値を書く（§8 の 12）。
- 実効の値の変更（性能や成績の改善）。整理は解決結果を保つ。
- 過去の実験記録・Run の設定の写し・PRD072 の固定資料・ADR の書き換え（[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針）。

### 3.3 成功の測り方

- 本流 1 本の構成を知るのに開く Run プロファイルが、その本流 1 つと予算 1 つだけになる（受入 A4・A6）。
- `Atari.txt` の Run プロファイルの数が 72 から、本流・機能・予算の合計に減り、値だけ違う腕と参照の無いものが 0 になる（受入 A4）。
- 並べる順の決まりのコメントが Run プロファイルについて 0 件になる（受入 A5）。
- ユーザーが本流 1 本を開き、ほかのプロファイルを見ずに構成を説明できる（受入 A6）。

### 3.4 用語（暫定）

- 本流の Run プロファイル: Atari の主な構成 1 つを、土台との差を全部並べて 1 つにした Run プロファイル。ほかの Run プロファイルと連結せずに選べる。
- 機能のプロファイル: 本流の後ろに足して、機能 1 つを丸ごと入れる Run プロファイル（RND、TempoRL、評価だけの Run など）。
- 土台: 本流が前提にする、Run プロファイルではない設定の全部。共通ファイル（`_main.txt`・`common.txt`・`agent.txt`・`nn*.txt`・`metrics_*.txt`）と、`Atari.txt` のうち Run プロファイルの定義（`run.@` で始まる行）以外の全部である。`Atari.txt` の節では、基本設定（lane 数・`DefaultDQNAgent.$`・`AtariEnv.$`・Network の選択・既定の `run.$` を含む）・Run設定・ステップ数 依存設定・A1・E1・M1・NN・AtariEnv・メトリクス・app に当たる。今の「実験」の節は §8 の 11 で無くす。

本流と機能は、CONTEXT.md の「Run プロファイル」の下位の呼び分けである。土台は Run プロファイルではない。既存の語では、この役割の違いを言い分けられない（「ベース」は [020](../design/020_user_guide_run.jp.md) 102 行で共通ファイルとプロファイルの両方を指し、「上書き層」はファイル本体の prefix を指す）。CONTEXT.md に足すかは §8 の 20 で決める。

## 4. 提案の形（暫定の既定）

この節は §8 の未決表の暫定の既定を前提に書く。行ごとに前提にした番号を添える。

### 4.1 Run プロファイルの 3 種類（§8 の 1・5・6・7・8）

| 種類 | 数（見込み） | 書き方 |
| --- | --- | --- |
| 本流 | 2（サンプル効率優先・実時間優先）。Rainbow 寄せは候補 | 土台との差を全部、話題ごとに並べる。ほかのプロファイルに依存しない |
| 機能 | 参照のある機能だけ。bat が実行の形を選ぶ `run.@batch` もここに入る | 本流の後ろに足す。葉を足すか、§4.3 の決まりで選択を書く |
| 予算 | `to_10m`・`to_30m`・`to_50m`・`to_100m` の 4 個。`to_50m` は `run.@a5` が持つ 50M を移して新しく作る | `@vars` の予算と、予算で終える `app.batchrun.exp_exit_step = ${@vars.max_exp_step}` だけを持つ |

値だけ違う腕は作らない。値の実験は bat の引数（正式名の葉）で書く。特別な長さの予算（`to_50`・`to_400k`・`a5_20m`）も値の腕として消し、要る Run は bat の引数で `@vars.max_exp_step` を書く（§8 の 5）。

### 4.2 本流の書き方（§8 の 2・3・4・10・11・12）

本流は、今の連結を解決した結果のうち、土台と違う葉と選択を、話題ごとの見出しのコメントの下に並べる。話題は Env、Agent のチェーン、Network、学習器、探索、Replay、target、評価、メトリクスの順にする。同じキーは 1 回だけ書く。予算・seed・Run 名は書かない。葉は正式名で書く。例（今のサンプル効率優先の連結で最後に効く値を、正式名で書いた行）:

```
run.@sample_efficient : DefaultDQNAgent.$ = @baseline > @iqn2304 > @munchausen > A1 > @bf16 > A2 > A3
run.@sample_efficient : DefaultDQNAgent.net.branch.[main_feature].$ = net.branch.@AtariImpalaX2SN12Pool6Flat
run.@sample_efficient : DefaultDQNAgent.learner.replay_ratio = 4
run.@sample_efficient : DefaultDQNAgent.model.hard_update_interval = 500
run.@sample_efficient : DefaultDQNAgent.learner.replay_capacity = 2,097,152
run.@sample_efficient : run.eval_schedule.[greedy].interval = 78907
```

learn step で数える値（評価の間隔・target の同期）は、その本流の RR に合わせた値を書き、行末のコメントに exp step を添える。実時間優先の Greedy の間隔は、今の既定（100M で 39,453 learn step）を本流に書く。予算で間隔を変える今の `run.@eval2ch_r1_50m`（50M 以下で 19,727）は、本流にも予算にも置かず、bat の引数で書く（§8 の 22）。

Run 名の既定 `app.run_name = run_{t}_atari_${E1.game}`（今は `run.@a5_apex` 443 行）は、本流ではなく `Atari.txt` 本体の app の節（土台）に置く（§8 の 23）。

選ぶときは `run.$ = run.@sample_efficient > run.@to_30m` のように、本流・機能・予算の順に並べる。既定の `run.$` は `run.@realtime > run.@to_100m` にする（§8 の 13）。

### 4.3 機能のプロファイルと選択（§8 の 9）

- 機能のプロファイルは、葉を足して有効化することを基本にする（[実装 品質確認観点](../quality/impl.md) 174 行）。
- メトリクスの群を足す機能は、今の決まり（178 行）のとおり `metrics.scalar.$` を書き直してよい。本流どうしのメトリクスのチェーンを同じにして、機能が書き写すチェーンを 1 つにする。
- 残すかどうかは §8 の 8 の参照の決まりで先に決め、この節は残った機能の書き方だけを決める。
- `DefaultDQNAgent.$` か `DefaultDQNAgent.net.branch.[main_feature].$` を書き直す機能は、本流の後ろに足す順だけで結果が決まるよう、同じキーを書き直すほかの機能と 1 つの Run で重ねない。2026-11-08 時点で §8 の 8 により残るのは `vitva`（13 の bat が参照）だけで、`no_munch`・`vitbtr`・`capall`・`rainbow_like_impala` は参照が無いので消える。

### 4.4 `Atari.txt` の節の並び（§8 の 11）

土台（§3.4。基本設定・Run設定・ステップ数 依存設定・A1・E1・M1・NN・AtariEnv・メトリクス・app の節）を先に、Run プロファイルを本流・機能・予算の順の節にまとめる。今の「実験」の節の A2 の行は、両方の本流が同じ値なら A1 へ、本流ごとに違うものは各本流へ移し、節を消す。RND・Rainbow 寄せなどが「基本設定」の節の先頭（8〜120 行）にある並びも、この並びへ移す。

### 4.5 性能

設定の解決は Run の起動時に 1 回だけ行う。本 PRD は Run の中の処理を変えない。

## 5. 実装契約

### 5.1 解決結果を保つ

整理の前の今の `Atari.txt` で次の構成を解決した結果を控え、整理のあとに新しい名前で同じ構成を解決して比べる（受入 A1）。

| 構成 | 整理の前 | 整理のあと |
| --- | --- | --- |
| 実時間優先 100M | 既定の `run.$`（156 行） | 既定の `run.$` |
| サンプル効率優先 30M | 11 の bat の `SE30`（30〜35 行を展開した連結） | `run.@sample_efficient > run.@to_30m` |
| 実時間優先 30M | 11 の bat の `RT30B`（34 行を展開した連結） | `run.@realtime > run.@to_30m` と、bat の引数 `run.eval_schedule.[greedy].interval=19727`（§8 の 22） |
| 13 の bat のステージ | 切り替えの時点でまだ流していないステージの連結と引数 | 新しい名前に替えた連結と引数 |

比べるキーは、解決結果の全キーから次を除いたものである。除いたキーの値は、Agent・Env・Runner が読む正式名のキーへ届いた値として比べる。

- 先頭の区切りが上書き層の名前（`A1`・`A2`・`A3`・`E1`・`M1`・`M2`・`P1`）のキー。§8 の 10・11 で行を移したり正式名へ書き換えたりするので、層の名前のキーは前後で変わる。
- `@vars.` で始まるキー（予算の素材。実効の予算は `app.*` のキーで比べる）。
- `run.@` で始まるキー（プロファイルの定義）と `run.$`。

比べるキーの集合と文字列値が一致することを求める。連結の途中で効いていなかった値（後ろに書き換えられた行）は、本流へ写さない。

### 5.2 移行対象

直すもの:

| 場所 | 直すこと | 前提の番号 |
| --- | --- | --- |
| [Atari.txt](../../apps/runner/config/Atari.txt) | 本流 2 本と予算 `to_50m` を足し、部品・値の腕・参照の無いプロファイルを消す。節を §4.4 の並びにする。既定の `run.$` を替える | 1〜13・21〜23 |
| [13 の bat](../../apps/13_batch_run.bat) | 本流の変数（`A5`・`RR1`・`RR4`・`RF` と、それを使う連結の変数）を、本流・機能・予算の名前に替える。bat の葉を正式名に揃える | 10・14 |
| [11 の bat](../../apps/11_batch_run.bat) の、`CFG` が今の設定を指すステージ | 13 の bat と同じ | 10・14 |
| `core/anet-core/src/dqn_based_agent_test.cpp`、`core/anet-core/src/dqn_munchausen_test.cpp`、`core/envs/atari1/src/AtariEnv_test.cpp` | 消えた名前を参照する行を、新しい名前か葉へ替える | 6・8 |
| `core/anet-core/testdata/prd103`・`prd104` | 消えた名前を新しい名前へ替える | 15 |
| [020](../design/020_user_guide_run.jp.md)（jp・en） | 137 行の例と §3.6・§3.7 の Atari の説明を、本流・機能・予算の形にする | 16・18 |
| `docs/memo` 直下の PRD のうち、索引の issue の AI の判定が実装済みでないもの（2026-11-08 時点で 117・905・907・918・944）、wiki の breakout | 消えた名前を今の状態として書いた箇所を、新しい置き場に替える | 16 |

直さないもの:

| 場所 | 理由 | 前提の番号 |
| --- | --- | --- |
| `docs/experiments/`、`docs/memo/done/`、`docs/memo` 直下の実装済みの PRD（索引の issue の AI の判定が実装済みのもの。2026-11-08 時点で 098）、`docs/adr/`、`docs/management/issues/`、`docs/wiki/log.md` | 過去の記録（[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針） | 16 |
| 本 PRD（`docs/memo/118_atari_run_profile_flatten_10prd.md`） | 整理の前の状態と、消す名前の一覧を記録する | 16 |
| `core/anet-core/testdata/prd072`・`prd084`・`prd094`・`prd097`・`prd099`・`prd102`・`prd106` | 固定入力と、再実行を求められていない過去の検証資材 | 15 |
| [11 の bat](../../apps/11_batch_run.bat) の、`CFG` が写し（`rnd_explore.txt`）を指すステージ | 写しは今の `Atari.txt` を読まず、旧い名前で起動する | 14 |
| `apps/runner/config/LunarLander.txt`、`apps/12_batch_run.bat`、`docs/quality/impl.md` 178 行の例、`docs/wiki/subjects/lunarlander.md` | 同じ名前の LunarLander のプロファイルを指す | 16 |
| 追跡外の workspace の設定 | リポジトリの外の運用。写しを使う bat の設定 | 14 |

`config_baseline_test.cpp` と `intrinsic_test.cpp` は直さない。読むのは既定の `run.$` と、`run.@skip`・`run.@skip_aux`・`run.@attention`・`run.@rnd`・`run.@rnd_taiga`・`run.@rainbow_like` で、これらはテストのソースから参照されるので §8 の 8 で残る。受入 A3 で通ることを確かめる。

消す名前の別名や、旧い名前を新しい名前へ読み替える仕組みは残さない（[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針）。旧い名前を書いた bat や CLI は、起動時に止まる（§2.7）。

### 5.3 クラス一覧

足す・変える・消すクラスは無い。変えるのは設定ファイル・bat・テスト・文書である。

### 5.4 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261108-06a3f0](../management/issues/issue-20261108-06a3f0.md) | Atari.txt の本流（サンプル効率優先・実時間優先）が Run プロファイル 11〜14 個の連結と本体の A2 の行と bat の引数の重ね合わせで定義され、人が読んで構成を理解できない | `316_config`, `322_env_individual` | [Atari.txt](../../apps/runner/config/Atari.txt) 156・337〜363・372〜706 行、[11 の bat](../../apps/11_batch_run.bat) 30〜35 行 | §8 の 1〜16・21〜23 | A1〜A8 |

## 6. 検証と受入

### A1. 解決結果の一致

§5.1 の表の各構成について、§5.1 の「比べるキー」の集合と文字列値が、整理の前後で一致する。比べた構成と、除いたキーの数を実装記録に残す。

### A2. 既定葉の検査

[core/anet-core/testdata/prd072/README.md](../../core/anet-core/testdata/prd072/README.md) の `check_default_leaves.py` が errors=0 で、`check_default_leaves_test.py` が通る。

### A3. 出荷設定の検査と既存のテスト

`anet-core-test.exe "[shipped_config]"` と既定のテストの集合が、整理の前と同じ結果になる（整理の前に落ちるテストがあれば、同じものだけが落ちる）。

### A4. Run プロファイルの数と種類

`Atari.txt` に残る `run.@` の名前が、それぞれ次のどれか 1 つに当たる。内訳を実装記録に残す。

- §8 の 1 の本流
- §8 の 5 の予算 4 個（`to_10m`・`to_30m`・`to_50m`・`to_100m`）
- 機能: §8 の 8 の参照元のどれかから、`run.@` を付けた名前を `git grep -w -F` で引けるもの（11 の bat は、`CFG` が今の設定を指すステージの行だけで引く）

§2.2 の「値だけ違う腕」30 個の名前と、参照の無い 17 個の名前は残っていない（名前の一覧で数える）。本流のプロファイルの中で、同じキーが 2 回書かれていない。

### A5. 並べる順の決まり

`Atari.txt` に、Run プロファイルを並べる順を指示するコメント（「より後ろへ置く」「より後ろへ置くこと」）が 0 件である。

### A6. 人が読めること（人が確かめる）

ユーザーが本流のプロファイル 1 つを開き、ほかの Run プロファイルを見ずに、その本流の Network・学習器・探索・Replay・評価の構成を説明できる。

### A7. bat の展開

13 の bat の、§5.1 の表で比べたステージの引数を展開し、A1 で比べた整理のあとの連結と CLI の葉と一致する。11 の bat は、`CFG` を今の設定へ替えたステージがあれば、そのステージだけを同じく確かめる。走っている bat を書き換えない（§8 の 14）。

### A8. 旧い名前の残り

消した Run プロファイルの名前ごとに、`run.@` を付けた文字列（例 `run.@a5`）をリポジトリ全体で `git grep -w -F` で引き、当たる場所が §5.2 の「直さないもの」の表の場所だけである。素の名前（`a5` など）では引かない。ほかのライブラリや文の中の同じ綴りに当たるためである。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| ほかの env の設定ファイルの整理 | 先送り（別の関心） | Atari で形を確かめてから判断する（起点の会話の方針）。再開条件: 本 PRD の実装が済み、ユーザーが Atari の形を確かめたとき（§8 の 17） |
| learn step で数える値の単位の変更 | 範囲外（別の関心） | PRD096・PRD940 で扱う。そちらが入ったら、本流の評価の間隔と target の同期を exp step の値に書き換えられる |
| チェーンへ項を足す記法 | 不採用 | PRD907 D12 で足さないとした。本 PRD の形は今の文法で書ける |

## 8. 未決

暫定の起票なので、確定事項を持たない。次の各行は、起点の会話の方向と、AI が置いた既定である。グリルですべて見直す。

| # | 事項 | 暫定の既定 | 理由と依存 | 決め方 |
| --- | --- | --- | --- | --- |
| 1 | 本流の本数と中身 | 2 本。サンプル効率優先は 11 の bat の `SE30` から予算を除いたもの、実時間優先は既定の `run.$` から予算を除いたもの。`run.@rainbow_like` は本流の候補として残す | bat と既定の `run.$` が使う構成がこの 2 つ（§2.3）。Rainbow 寄せはすでに平ら（§2.2） | 選択 |
| 2 | 本流の名前 | `run.@sample_efficient`・`run.@realtime`。Rainbow 寄せは `run.@rainbow_like` のまま | 略さずに読めることがゴール 1 に効く。Run 名の略号（`se`・`rt`）は bat の Run 名に残る。名前の基準は [実装 品質確認観点](../quality/impl.md) の「名前」 | 選択 |
| 3 | 本流どうしで同じ値の置き場 | 各本流に重ねて書く。土台へ下ろさない | 1 か所で読めることがゴール 1。本流 2〜3 本なら重なりは数十行。重なりを減らすなら土台（A1）へ下ろすが、本流を読むのに土台も要る。#1 に依存 | 選択 |
| 4 | 本流に書く範囲 | 土台との差を全部（Env・Agent のチェーン・Network・学習器・探索・Replay・target・評価・メトリクス）。予算・seed・Run 名は書かない | 予算は Run ごとに変わる別の関心。CONTEXT.md の Run プロファイルの定義は予算を含むが、予算のプロファイルを本流に足して選ぶので定義とは食い違わない | 選択 |
| 5 | 予算の持ち方 | 予算のプロファイルは `to_10m`・`to_30m`・`to_50m`・`to_100m` の 4 個。`to_50m` は `run.@a5` の 50M を移して作る。`app.batchrun.exp_exit_step = ${@vars.max_exp_step}` は 4 個に持たせる。特別な長さの `to_50`・`to_400k`・`a5_20m` は値の腕として消し（#6）、要る Run は bat の引数で `@vars.max_exp_step` を書く | 予算は本流と別の関心（#4）。bat は今も予算をプロファイルで足す。`to_50` を使うのは固定入力の prd072 と過去の検証資材の prd102 だけで、どちらも直さない（#15） | 選択 |
| 6 | 値だけ違う腕 30 個 | 消す。値の実験は bat の引数で書く | [実装 品質確認観点](../quality/impl.md) 174 行（機能ごとの変種のために `@` プロファイルを作らない）。テストが使う腕はテストを葉へ替える | 選択 |
| 7 | 参照の無い 17 個 | 消す | 実験記録は設定の写しを持つので、消しても再現できる | 選択 |
| 8 | 機能のプロファイルの残し方 | 次の参照元のどれかに `run.@<名前>` が出る機能は残し、どれにも出ないものは消す。参照元: テストのソース（追跡している `*_test.cpp`）、13 の bat、11 の bat のうち `CFG` が今の設定を指すステージの行、§5.2 の「直すもの」の文書（020・PRD117・905・907・918・944・wiki の breakout）と検証資材（prd103・prd104）。残った機能の書き方は #9 で決める | 参照元は、今の `Atari.txt` を読む場所に限る。写しを読む 11 の bat のステージと、直さない過去の記録は今の `Atari.txt` を読まない。参照の無い機能は管理の手間だけを残す（#7 と同じ扱い）。2026-11-08 時点では、テストが `run.@skip`・`run.@skip_aux`・`run.@attention`・`run.@rnd`・`run.@rnd_taiga`・`run.@rainbow_like` を参照する | 選択 |
| 9 | #8 で残った機能が選択（`.$`）を書き直すこと | `metrics.scalar.$` は今の決まり（impl.md 178 行）のまま書き直してよい。本流どうしのメトリクスのチェーンを同じにする。`DefaultDQNAgent.$` か `main_feature.$` を書き直す機能は、同じキーを書き直すほかの機能と 1 つの Run で重ねない（§4.3） | 書き直しの順の依存（§2.4）を「本流の後ろに足す」の 1 つに減らす。残すかどうかは #8 で先に決まる（2026-11-08 時点で該当は `vitva` だけ）。消える `no_munch` に当たる Run（Munchausen なし）が要るときは、bat の引数で書く。葉 `DefaultDQNAgent.learner.munchausen.enabled = false` だけで書けるかは、その Run を作るときに確かめる | 選択 |
| 10 | 葉の名前 | Run プロファイルと bat の葉は正式名（`DefaultDQNAgent.*`・`AtariEnv.*`）で書く。上書き層 A1/A2/A3 は設定ファイル本体の層としてだけ使う | 同じキーを別名と正式名で書くと正式名が勝ち、bat の別名の葉が黙って効かない（§1.5） | 選択 |
| 11 | 本体の「実験」の節（A2 の行）と節の並び | 両方の本流が同じ値なら A1 へ、本流ごとに違うものは各本流へ移し、「実験」の節を消す。Run プロファイルは本流・機能・予算の順の節にまとめる（§4.4） | 本流を読むのに本体の A2 を重ねて読まずに済む。値の実験は bat の引数（#6） | 選択 |
| 12 | learn step で数える値 | 本流ごとに、その本流の RR に合わせた値を書き、exp step をコメントで添える。bat で RR を変える Run は、評価の間隔と target の同期も bat で書く（今と同じ） | 単位の直しは PRD096・PRD940 の関心（§7） | 別の関心 |
| 13 | 既定の `run.$` | `run.@realtime > run.@to_100m` | 今の既定（実時間優先 100M）と同じ構成 | 選択 |
| 14 | 切り替えの時機 | 13 の bat が Atari のキューを流していないときに、`Atari.txt`・13 の bat・テストを同じ変更で切り替える。11 の bat は、写し（`rnd_explore.txt`）を読むステージを旧い名前のまま残し、今の設定を読むステージを足すときに新しい名前で書く。サブ機の作業ツリーの更新は人が行う | 今の `Atari.txt` を読む bat の後続のステージは、その時点の `Atari.txt` を読む。写しを読む bat は影響を受けない（§2.7）。消えた名前を読むと起動時に止まる。サブ機が読む `Atari.txt` の所在は ④（実装計画で確かめる） | 選択と④ |
| 15 | 過去の PRD の検証資材と固定入力 | `core/anet-core/testdata/prd072`（固定入力）と、`prd084`・`prd094`・`prd097`・`prd099`・`prd102`・`prd106`（検証資材）は直さない。[316 の地図](../wiki/categories/316_config.md)の「変えるときの条件」が再実行を求める `prd103`・`prd104` だけを新しい名前へ直す | prd072 は manifest の固定 commit を読む。ほかの資材は、その PRD の実装を確かめ終えた記録。直さない資材を今の設定で流すと、消えた名前で起動時に止まる。prd103・prd104 の各資材が今の `Atari.txt` を読むかは ④（実装計画で確かめる） | 選択と④ |
| 16 | 今の状態として名前を書いた文書 | 020（jp・en）と、`docs/memo` 直下の PRD のうち索引の issue の AI の判定が実装済みでないもの（2026-11-08 時点で 117・905・907・918・944）、wiki の breakout を直す。ADR・実験記録・done と実装済みの PRD（098）・issue の本文・wiki のログは直さない | クリーンブレーク方針（[AGENTS.md](../../AGENTS.md)）。ADR は書き換えない。実装前の PRD は、実装のときに今の状態として読まれる | 選択 |
| 17 | ほかの env へ広げるか | 本 PRD では扱わない。Atari の形を確かめてから別の PRD で判断する | 起点の会話の方針（§3.2） | 後で決める |
| 18 | 規約にするか | 020 の Atari の説明に書き、[実装 品質確認観点](../quality/impl.md) の「設定ファイル」の規約には #17 の判断のあとで足す | 規約にすると全 env に当たる。#17 に依存 | 選択 |
| 19 | ADR を起こすか | グリルで確定したら起こす | 戻すのが重く、文脈なしでは意外で、連結の形との実際の得失がある見込み | グリルで確定後 |
| 20 | 用語（§3.4）を CONTEXT.md に足すか | グリルで確定したら足す | 「Run プロファイル」の下位の呼び分けで足りるか、新しい語が要るか | グリルで確定後 |
| 21 | `run.@batch`（`app.$ = app.batchrun > P1`、432 行） | 機能のプロファイルとして残す | 本流の構成ではなく、bat が Run の実行の形（batch run）を選ぶ切り替え。11・13 の bat の連結が使う。§2.2 では本流の部品に数えたが、移す先は機能 | 選択 |
| 22 | 予算で変わる評価の間隔（`run.@eval2ch_r1_50m`、524 行。RR 1 で 50M 以下のとき Greedy の間隔 19,727） | プロファイルにせず、bat の引数 `run.eval_schedule.[greedy].interval=19727` で書く | 本流（予算を書かない。#4）にも予算（RR を知らない。#5）にも置けない。間隔の単位が exp step になれば（PRD096）、予算で変える必要自体が無くなる | 選択 |
| 23 | Run 名の既定（`app.run_name = run_{t}_atari_${E1.game}`、`run.@a5_apex` 443 行） | `Atari.txt` 本体の app の節（土台）に置く | どの本流でも同じ既定で、本流には Run 名を書かない（#4）。bat は今も Run 名を引数で書く | 選択 |

## 9. 依存と順序

- §8 の 14 のとおり、今の `Atari.txt` を読む bat（13）のキューが止まっているときに切り替える。2026-11-08 時点で 11 の bat が流している Montezuma の 100M は写しを読むので、切り替えの影響を受けない。
- 本 PRD を PRD096・PRD940 より先に入れても、あとから入れても、本流の値の書き換えで済む（§8 の 12）。

## 10. 参照

- 題材: [issue-20261108-06a3f0](../management/issues/issue-20261108-06a3f0.md)
- 設定: [Atari.txt](../../apps/runner/config/Atari.txt)、[11 の bat](../../apps/11_batch_run.bat)、[13 の bat](../../apps/13_batch_run.bat)
- 規約: [AGENTS.md](../../AGENTS.md) のクリーンブレーク方針、[実装 品質確認観点](../quality/impl.md) の「設定ファイル」、[CONTEXT.md](../../CONTEXT.md) の Run プロファイル・上書き層・個別指定
- 設計文書: [020](../design/020_user_guide_run.jp.md) §3.6・§3.7
- PRD: [PRD907](907_config_file_layout_10prd.md)、[PRD096](096_metrics_step_axis_interval_10prd.md)、[PRD940](940_target_update_step_axis_10prd.md)、[PRD059](done/059_config_concept_tree_alignment_10prd.md)、[PRD072](done/072_config_selection_final_value_10prd.md)
- ADR: [0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)、[0042](../adr/0042-config-inheritance-as-differential-base.md)
- 地図: [316 設定管理](../wiki/categories/316_config.md)、[322 個別 env](../wiki/categories/322_env_individual.md)
