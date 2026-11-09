# PRD119: 文書・コメント・設定の小さな食い違いをまとめて直す

> 2026-11-09 実装のレビューの指摘で直した。課題の記録の仕組みで AI の判定の値「直った見込み」が「残件無し」に改められた（コミット `180213e2`）ので、§2.2・D14・A10 の題材の課題の判定の値を `残件無し` にした。
> 2026-11-09 実装の入口で直した。D8 の英語版のヘッダの扱いを、実装 品質確認観点の「検証と報告」の規約に合わせ、「変えない」から「対応を確かめて blob と日付を更新する」に改めた。§5.1 の D8 の行、§5.2 の「触らないもの」、A5 を合わせた。
> 2026-11-09 再グリル（同じ日の別の枠）。暫定の印を外し、§8 の未決表の 11 行と §4 の暫定の既定を、§4 の確定事項 D1〜D15 へ移した。題材に [issue-20261103-fb8a45](../management/issues/issue-20261103-fb8a45.md) を足して 8 件にした（D13）。照合で分かった事実を §2.1 に足し、直した。structure は `(=Tag)`・`(@Source)`・`Add(...)` を解釈しない。nn.txt の 311〜312 行にも古い init.mode のコメントがある。既定葉検査はキーの名前で分類する。`anet-lab-docs-` は done/043 にもある。§1 に 1.5 を足し、§2.2 に規約 4 項目を足した。§5.1 に nn.txt の「記法」の節と設計文書 130 §2.4 を足し、§5.2・§5.4 を D 番号と fb8a45 に合わせた。§6 の A2・A5・A6 を書き直し、A10 を足した。§3.2 の非ゴールに記法の実装を足した。§7 に 4 行を足した。§8 には範囲外の 2 件（記法の実装、設計文書 220 の fps の記述）を残し、§9・§10 を整えた。レビューの指摘で、nn.txt の 376 行（密なスキップ結合をタグで書けるとする MEMO）を D7 の直す箇所に足し、§5.2 の移行対象の範囲と、タグを書く箇所の検索の書き方と、D13 の理由を直した。再グリルで足した行番号も、同じ日の作業ツリーの位置である。
> 2026-11-09 起票（暫定。質疑を省き、既定を §8 の未決表に置いた）。行番号は起票時点の作業ツリーの位置で、実装時は近傍の記述で探し直す。
> 起点: 2026-11-09 の課題の棚卸し。本版の課題の WSJF を同じ基準で採点し直したところ、手間の小さい課題（文書・コメント・設定の数行）が一覧の上位に並んだ。ユーザーが、それらを 1 本の暫定 PRD に束ねて起票するよう指示した。グリルと実装は別の枠で行う。
> 題材: [issue-20261103-554849](../management/issues/issue-20261103-554849.md)、[issue-20261103-c1c608](../management/issues/issue-20261103-c1c608.md)、[issue-20261103-5a5373](../management/issues/issue-20261103-5a5373.md)、[issue-20261103-2751c8](../management/issues/issue-20261103-2751c8.md)、[issue-20261103-879a70](../management/issues/issue-20261103-879a70.md)、[issue-20261106-ddd395](../management/issues/issue-20261106-ddd395.md)、[issue-20261103-ee32bd](../management/issues/issue-20261103-ee32bd.md)、[issue-20261103-fb8a45](../management/issues/issue-20261103-fb8a45.md)。関連: [PRD117](117_warn_policy_and_unread_config_10prd.md)（読まれない設定の検出）、[PRD118](118_atari_run_profile_flatten_10prd.md)（Atari の Run プロファイルの整理）、[PRD934](934_metrics_viewer_lod_documentation_consistency_10prd.md)（LOD の記述の整合）、[PRD949](949_release_packaging_cmake_10prd.md)（リリース物の組み立て）、[PRD114](114_findings_intake_and_fix_prd_10prd.md)（課題の仕組み。過去の PRD のパスの表記の直し）。

## 1. 問題

### 1.1 文書のリンクと名前が実物とずれている

README の Doxygen の zip の名前が、リリースの手順が作る名前と違う。設計文書の optuna の手引きへのリンクが切れている。利用者が README の名前でリリースの zip を探すと見つからず、文書から手引きを開けない。

### 1.2 コメントと文書の説明が実装とずれている

- ReplayBuffer の初回優先度の補完の `@note` が、ロックを持って呼ぶ関数に `GetStats` を挙げていない。
- `metrics_scalar.txt` の冒頭の記法の説明は、今は受け付けない `&step` の書き方と、一部の event・target だけを載せている。observers.cpp の中のコメントも同じく古い。
- `nn.txt` の冒頭の type 一覧は、登録済みの 7 種を欠き、type ではない `Add` を含む。init.mode のコメントは 6 種のうち 3 種しか載せていない。
- `nn.txt` の「記法」の節と設計文書 130 §2.4 は、structure が出力 tag と入力 tag 参照（`(=Tag)`・`(@Source)`・`Add(...)`）を表せると書くが、パーサーはこれを解釈しない。`nn.txt` の「表現出来ないパターン」の MEMO も、密なスキップ結合を今の記法でタグを使って書けるとしている。

説明どおりに書くと、データ競合に気づけない、値が記録されない、使える type を使えないと判断する、解釈されない記法で起動が止まる、といったことが起きる。

### 1.3 選ぶと起動できないサンプルが残っている

`nn.txt` の body のサンプル `@ResNet` と `@MLP_RES` は、定義の無い block、登録の無い type、解釈されない記法を使っていて、選ぶと起動エラーで止まる。

### 1.4 既定葉検査が既存の違反で失敗する

`Atari.txt` の fps の 2 行が、設定ファイルの代入演算子の規約に反していて、既定葉検査が errors 2 で失敗する。設定を変えるたびに、この既存の違反を切り分ける手間がかかる。

### 1.5 過去の記録の切れたリンクの扱いが決まっていない

PRD069 の実装記録の 2 か所と、done/009・done/010 の PRD の 16 か所のリンクが切れている。課題の対応方針は「直す」と「過去の本文は保持する」を両方書いていて、どちらで片づけるかが決まっていない。

## 2. 現状（2026-11-09 に確認したこと）

### 2.1 題材ごとの事実

| 題材 | 事実 |
| --- | --- |
| 554849 | [apps/README.md](../../apps/README.md) の 52 行は Doxygen の zip を `anet-lab-docs-<version>.zip` と書く。[release.yml](../../.github/workflows/release.yml) の 254 行と 317 行が作る名前は `anet-lab-doxygen-<版>.zip` である。現用の文書で `anet-lab-docs-` を書くのは、README の 1 行と、それを引く PRD949 §2 の 15 である。ほかに、過去の記録の [done/043](done/043_release_packaging_10prd.md) の 14 行と、課題の記録（単票の題・事象、課題の一覧、wiki の地図の課題の区画）にある |
| c1c608 | 切れているリンクは 4 か所で、[030 の日本語版](../design/030_user_guide_analysis.jp.md) の 20 行・525 行と、[160 の日本語版](../design/160_applications_and_tools.jp.md) の 109 行・444 行の `../optuna.md` である。手引きの実物は同じフォルダの [optuna.md](../design/optuna.md) にある。140 の日本語版の行は PRD115 の変更で直っていて、030・140・160 の英語版はどれも `optuna.md` を指していて切れていない |
| 5a5373 | [replay_buffer_impl.hpp](../../core/anet-core/src/replay_buffer_impl.hpp) の 325 行の `@note` は、`metadata_mutex_` を持って呼ぶ関数として `Enqueue` と `CompleteReady` だけを挙げる。372 行の `GetStats` は `CompleteReady` が更新する統計を写して返す。今の呼び出し元（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の 2128〜2130 行）はロックを取ってから呼んでいる |
| 2751c8（記法） | structure は、block 名を `>` でつなぎ、`(*N)` で繰り返す直列の記法である。[nn_impl.cpp](../../core/anet-core/src/nn_impl.cpp) の 541 行の `SplitPipelineString` が `>` で分けた語を、1680〜1700 行が `(*N)` を除いて block 名として引き、見つからなければ `Block definition not found` で止まる。`(=Tag)`・`(@Source)`・`Add(...)` を解釈する箇所は無く、書くと語ごと block 名として引かれて止まる。branch の合流は bind の `*`（積）と `,`（連結）だけで、足し算は無い（[130](../design/130_neural_networks.jp.md) の 84 行）。nn.txt の 362〜368 行の「記法」の節は `(=Tag)`・`(@Source)` を記法として載せ、`Add(@skip @main)` を例にしている。同じファイルの 371〜382 行の「表現出来ないパターン」の MEMO は、376 行で密なスキップ結合を「今の記法での表現: 可能だが、タグだらけになる」と書く。branch を分ければ、bind の連結（`,`）で前の branch の出力を後の branch へつなげる（130 の 84〜91 行）が、structure の中では書けない。130 の日本語版の 94 行と英語版の 95 行も、structure が出力 tag・入力 tag 参照を表せると書く |
| 2751c8（サンプル） | [nn.txt](../../apps/runner/config/nn.txt) の 241 行の `@ResNet` は、定義の無い `Linear_128`・`MyReLU` と、解釈されない `(=skip_val)`・`Add(@skip_val @proc_val)` を使う。304 行の `@MLP_RES` は、301 行で type を `Composite`（登録の無い type）とした block `[ResBlock]` を使い、その structure（302 行）も `Add(@input @proc)` を使う。登録済みの `ResBlock` は畳み込みの block（[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の 1336 行）で、MLP の残差を書ける type は無い。`@ResNet`・`@MLP_RES` と、`@MLP_RES` だけが使う block `Linear_Proj`・`Linear_Hid`・`Linear_Out`・`[ResBlock]` を、今の設定（`apps/runner/config/`）で参照するものは無い。残るのは、過去の Run の設定の写し（`docs/experiments/**/config/`、`apps/runner/workspaces/**/runs/`、`core/anet-core/testdata/prd092/config/`）、PRD072 の golden（`core/anet-core/testdata/prd072/baseline/`）、workspace の写し（`apps/runner/workspaces/AtariRnd-01/config/frozen/nn.txt`）、過去の記録（done/059 の実装記録）である |
| 2751c8（コメント） | nn.txt の 16〜41 行の type 一覧は、[nn_modules.cpp](../../core/anet-core/src/nn_modules.cpp) の 3279〜3317 行が登録する 29 種のうち `Reshape`・`StackMerge`・`Dropout`・`LayerNorm2d`・`GAP2D`・`MaxPool2d`・`HybridSpatialEmbedder` を欠き、type ではない `Add` を含む。5〜10 行と 311 行の init.mode のコメントは xavier・he・orthogonal だけで、[nn.hpp](../../core/anet-core/include/anet/nn.hpp) の 69 行が受け付ける default・constant・trunc_normal を欠く。312 行は mode を番号で呼ぶ「1=Xavier」と書くが、mode は名前で指定する |
| 879a70 | [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の 11〜14 行は、`&step` の書き方、event に `@learn`・`@train` だけ、target に `$actor`・`$env` などが無い説明である。[observers.cpp](../../core/anet-core/src/observers.cpp) の 1410〜1414 行も、同じく古い説明と旧形式の例を載せる。受け付ける語は [metrics.cpp](../../core/anet-core/src/metrics.cpp) の 40 行の `ParseMetricTokens` が分け、どれにも当たらない語は key として扱われる。記法の正本は [140](../design/140_observability.jp.md) §6（Metric設定contract）の 189 行（`metrics.scalar.[tag] = key [$step_axis] [@event] [$target] [$runner_scope] [$ema] [interval:N] [ema_alpha:A] [clip:C]`）と、192〜201 行の要素ごとの表である |
| ddd395 | [check_default_leaves.py](../../core/anet-core/testdata/prd072/check_default_leaves.py) の結果は `Assignments: 32011, audited: 4015, errors: 2` で、検出されるのは [Atari.txt](../../apps/runner/config/Atari.txt) の 1650 行（`app.online.eval_panel.fps ?= 14.5`）と 1654 行（`app.online.train_panel.fps ?= 10`）である。検査は節ではなくキーの名前で分類し、環境別のファイルで `<名前>Env.` で始まらず、Atari 用の既定の一覧にも無いキーを明示の割り当てとして扱い、`=` を求める（[migrate.py](../../core/anet-core/testdata/prd072/migrate.py) の 37〜44 行）。train_panel は、同じファイルの 328 行（実験の節）の `app.online.train_panel.fps = 0` が勝つので、今の実効値は 0 である。eval_panel の実効値は 14.5 で、[common.txt](../../apps/runner/config/common.txt) の 138 行の `?= 15` より後に読まれる。bat・Run プロファイル・ツールに、この 2 つのキーを上書きするものは無い。2026-11-08 の Run（`run_20261108-232006_mz_tg_rnd_s1`）の設定の写しでも、train_panel は 0、eval_panel は 14.5 だった |
| ee32bd | [PRD069 の実装記録](done/069_metrics_trace_channel_20impl.md) の 123 行と 187 行が、`out/test-tmp/prd069/` の比較結果とテストのログへリンクしている。このフォルダは実装時の一時出力で、リポジトリに入ったことが無い（`git log --all -- out/test-tmp/prd069` が空）。同じ文書の 52・60・111・155〜157 行は、同じフォルダを code の表記で書いている |
| fb8a45 | [done/009](done/009_dueling_dag_head_10prd.md) の 12〜15・45・76・81・82・86 行の 14 か所と、[done/010](done/010_image_metrics_v2_12prd.md) の 10・24 行の 2 か所のリンクは、`core/anet-core/src/...` のようにリポジトリの根からのパスで書かれていて、文書の位置からは辿れない。参照先のファイルは今もある（行番号は当時のもの） |
| 設定の解決結果 | 2026-11-08 の Run の設定の写し（解決結果）には、`@MLP_RES` だけが使う block のキーが `net.block.[Linear_Proj].`・`net.block.[Linear_Hid].`・`net.block.[Linear_Out].`・`net.block.[ResBlock].` で始まる 20 キーとして現れる。body のサンプル（`net.body.@ResNet.structure`・`net.body.@MLP_RES.structure`）はプロファイルの定義なので、解決結果には現れない。[PRD072 の検証](../../core/anet-core/testdata/prd072/README.md) の固定 17 入力比較は、manifest の固定 commit の `apps/runner/config` を入力にする（`prepare.py` の 36〜40 行、README の 15 行）ので、作業ツリーの設定を変えても入力も golden も変わらない |

### 2.2 規約と過去の判断

- 設定ファイルの代入演算子: 環境別のファイルで `?=` にするのはデフォルト設定の行だけで、それ以外は原則 `=` である。既存の実効値を保つために `?=` が要る葉だけ、理由を添えてデフォルト設定へまとめる（[実装 品質確認観点](../quality/impl.md) の「設定ファイル」の 154 行）。
- 同じキーの重なり: 同じ演算子どうしは後に書いたほうが勝ち、`=` と `?=` が混在したら順序によらず `=` が勝つ（[020](../design/020_user_guide_run.jp.md) の 104 行）。
- 過去の記録: 過去の PRD（done/・frozen/・dropped/）・実装記録・ADR の本文は書き換えない（[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針）。PRD114 の 999 の振り直しでは、番号の振り直しで動いたファイル名を指すパスの表記だけを直した（[wiki 311](../wiki/categories/311_dev_tools.md) の「変えるときの条件」、出典は PRD114 §5.9・§5.12）。
- 旧い記述の検索: 現用の範囲の検索には、設計文書の `.en.md` を含める（[実装 品質確認観点](../quality/impl.md) の項番 20）。
- Train panel の fps が 0 でも学習は続く（[160](../design/160_applications_and_tools.jp.md) §7.1）。
- 課題の状態: 状態（`解決` など）を変えるのは開発者で、AI は対応内容と AI の判定（`ai_judge`）を書く。題材の課題の判定は `残件無し` を使う（[anet-create-issue](../../.agents/skills/anet-create-issue/SKILL.md) の「各項目の書き方」、[anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)）。

## 3. ゴール・非ゴール・成功の測り方

### 3.1 ゴール

題材の 8 件の課題をすべて片づける。直した箇所について、リンク切れが 0 件、既定葉検査の errors が 0 件になり、設定の解決結果は、消すサンプルのキーのほかは変わらない。新しい仕組みや概念は足さず、文書・コメント・設定を直すだけにする。

### 3.2 非ゴール

| 退ける案 | 理由 | 本当の制約か |
| --- | --- | --- |
| PRD934（LOD の集約数の記述、issue-20261103-55b5f1）を本 PRD に吸収する | 専用の PRD が既にあり、そのまま実装すれば片づく | 制約ではない。PRD934 を退役させて吸収もできるが、PRD を 2 本並べないほうが記録が単純になる |
| PRD949 のリリース物の組み立てを一緒に直す | 本 PRD は README の名前の 1 行だけを直す。組み立ての移し替えは別の関心である | 制約ではない（別の関心） |
| nn.txt のほかの見本に残る読まれない葉（issue-20261107-0c34dd）を一緒に直す | 葉を検出する仕組みは PRD117 の範囲で、直し方も PRD117 の検出と一緒に決める | 制約ではない（別の関心） |
| structure に `(=Tag)`・`(@Source)`・`Add(...)` の記法を実装し、文書のほうを正にする | 新しい仕組みで、小さな食い違いの直しとは別の関心である。C++ の変更とテストが要る | 制約ではない（別の関心。§7、§8 の 1） |
| 過去の記録を今の事実へ書き換える（PRD069 の実装記録と done/009・done/010 の切れたリンクを直すことを含む） | 過去の Run の設定の写し、done/ の PRD と実装記録、workspace の写しは当時の記録である | 本当の制約（AGENTS.md のクリーンブレーク方針で過去の記録は書き換えない。§2.2）。リンクの記法だけを外す案も退けた（D12） |

### 3.3 成功の測り方

- 題材の 8 件を、実装のあとに解決にできる（§6 の受入がすべて通る）。
- 既定葉検査と、題材の 2 ファイル（030・160 の日本語版）のリンク検査が、既存の違反で落ちなくなる。以後の設定変更と文書の直しで、検査の失敗を自分の変更のものとして読める。

## 4. 確定事項

2026-11-09 の再グリルで決めた。D4・D9・D12・D13 はユーザーの選択、ほかは規約・コード・ほかの決定から導いた。

| # | 決定 | 理由と依存 |
| --- | --- | --- |
| D1 | README の zip の名前を `anet-lab-doxygen-<version>.zip` にする | release.yml が作る名前である（§2.1 の 554849） |
| D2 | 030・160 の日本語版の 4 か所のリンク先を `optuna.md` にする | 手引きの実物は同じフォルダにあり、英語版も同じ書き方である（§2.1 の c1c608） |
| D3 | `InitialPriorityCompleter` の `@note` の関数の並びに `GetStats` を足す | `GetStats` は `CompleteReady` が更新する統計を読むので、同じロックが要る（§2.1 の 5a5373） |
| D4 | nn.txt のサンプル `@ResNet`（240〜241 行の `[Pattern 4]` のコメントと定義）と、「MLP ResBlock」の節（275〜304 行の見出し、`Linear_Proj`・`Linear_Hid`・`Linear_Out`・`[ResBlock]` の定義、`@MLP_RES`）を消す。ほかの Pattern の番号は振り直さない（Pattern 4 は欠番になる）。消したあとに選ぶと、設定の解決で `material selection target not found` として止まる（[config_impl.cpp](../../core/anet-core/src/config_impl.cpp) の 328〜331 行。今の起動エラーと同じく fail-fast で、止まる時機が構築時から設定の解決時へ早まる） | 今の設定・テスト・bat に選ぶものが無い。書き直すには MLP の残差の type が要り、非ゴールの記法の実装か新しい type に当たる（§2.1 の 2751c8、§3.2）。当時の構成は過去の Run の設定の写しに残る。番号を振り直さないのは差分を最小にするため |
| D5 | nn.txt の type 一覧を、nn_modules.cpp が登録する 29 種に合わせ、`Add` を外す。今の見出しを保ち、足りない 7 種を当てはまる見出しへ入れる（`Reshape`・`StackMerge` は【Base / Routing】、`LayerNorm2d`・`GAP2D`・`MaxPool2d` は【Normalization & Pooling】、`HybridSpatialEmbedder` は【Embedding / Bridge】、`Dropout` は新しい見出し【Regularization】） | 一覧の正本は登録である（§2.1 の 2751c8）。`Add` は type でも記法でもない（D7）。見出しを保つのは読み方を崩さず差分を小さくするため |
| D6 | nn.txt の init.mode のコメント（5〜10 行と 311 行）に default・constant・trunc_normal を足して 6 種にし、312 行の「1=Xavier」を「xavier」に直す | nn.hpp が受け付けるのは 6 種の名前である（§2.1 の 2751c8） |
| D7 | nn.txt の「記法」の節（362〜368 行）から `(=Tag)`・`(@Source)` の 2 行を消し、`>` と `(*N)` の行を残す。`(*N)` の例はそのまま残す。合流の案内として「合流（積・連結）は branch の bind で書く（docs/design/130_neural_networks.jp.md §2.4）」の 1 行を置く。「表現出来ないパターン」の MEMO の 376 行は、「structure の中では書けない。branch を分けて bind の連結（`,`）でつなげば書けるが、branch だらけになる」の趣旨に直す | パーサーは `>` と `(*N)` しか解釈しない（§2.1 の 2751c8（記法））。記法の実装は非ゴールである（§3.2）。案内は、消す行の代わりに、今できる書き方の在りかを示すため |
| D8 | 設計文書 130 §2.4 の、structure が出力 tag・入力 tag 参照を表せるという記述（日本語版 94 行・英語版 95 行）を、block 反復（`(*N)`）だけを表せる記述に直す。英語版の冒頭の翻訳のヘッダ（`translated-from`）は、日英の対応を確かめてから、直したあとの日本語版の blob と日付に更新する | D7 と同じ事実による。旧い記述の検索には `.en.md` を含める（§2.2）。英語版を編集するときは、日本語版との対応を確かめた範囲に合う blob を記録する（[実装 品質確認観点](../quality/impl.md) の「検証と報告」の規約）。2026-11-09 の時点で英語版のヘッダの blob は日本語版の今の blob と一致しているので、同じ 1 文を日英で直せば全体が対応する |
| D9 | metrics_scalar.txt の 11〜14 行を、`# metrics.scalar.[tag] ?= key [$step_axis] [@event] [$target] [$runner_scope] [$ema] [interval:N] [ema_alpha:A] [clip:C]` と、「各要素に書ける語と既定は docs/design/140_observability.jp.md §6」の 2 行に置き換える。observers.cpp の 1410〜1414 行は、「記法の正本は docs/design/140_observability.jp.md §6、受け付ける語は metrics.cpp の ParseMetricTokens」の 1 行に置き換える。要素ごとの語の一覧は写さない | 語は足されやすく、写しは古くなる。知らない語は黙って key として扱われ、古い写しどおりに書くと値が記録されない（§2.1 の 879a70）。要素の並びはめったに変わらないので、1 行だけ写す |
| D10 | Atari.txt の 1650 行を `app.online.eval_panel.fps = 14.5` にする | 環境別のファイルは原則 `=` で（§2.2）、Atari.txt・bat・Run プロファイルにこのキーのほかの `=` は無いので、実効値は 14.5 のまま（§2.1 の ddd395） |
| D11 | Atari.txt の 1654 行の `app.online.train_panel.fps ?= 10` と、その説明の 1652〜1653 行を消す。328 行の `app.online.train_panel.fps = 0` の直前に、「描画は throughput に効くので 0。見るときは 10 で足りる（128 タイルのモザイク）」の趣旨のコメントを置く。実効値は 0 のまま | ゴールは解決結果を変えない（§3.1）。1654 行を `=` にすると、後に書いた `=` が勝って実効値が 10 に変わる（§2.2）。説明を値のそばへ移し、10 の目安も残す |
| D12 | PRD069 の実装記録は変えない。[issue-20261103-ee32bd](../management/issues/issue-20261103-ee32bd.md) の対応内容に、直さない理由（リンク先はリポジトリに入ったことが無く、過去の記録は書き換えない）を書く | 過去の記録は書き換えない（§2.2）。リンクの記法だけを外しても、読める中身は増えない（§2.1 の ee32bd）。PRD114 の先例は、番号の振り直しでリンク先が動いた場合に限る |
| D13 | [issue-20261103-fb8a45](../management/issues/issue-20261103-fb8a45.md) を題材に入れる。done/009・done/010 は変えず、対応内容に D12 と同じ理由を書く。fb8a45 の親に、本 PRD の索引の課題を書く | 題材に入れて done/ を変えないのは D12 と同じ判断で、同じ規則で閉じれば、方針の食い違った課題が残らない（§1.5）。親を書くのは、題材の課題は parent で索引の課題を指すという決まりによる（[memo README](README.md) の冒頭） |
| D14 | 実装の工程では、題材の各課題の対応内容に直した内容（D12・D13 では直さない理由）と場所を書き、AI の判定を `残件無し` にする。状態を `解決` にするのは開発者である | 課題の状態は開発者が変える（§2.2） |
| D15 | PRD949 §2 の 15（README の名前の食い違いの記述）は、本 PRD では書き換えない | PRD949 は暫定の起票で、まだグリルしていない。PRD949 の再グリルで §2 の事実を今のコードと照合し直す。本 PRD が別の PRD の本文を直すと、改訂の記録が 2 本に分かれる |

## 5. 実装契約

### 5.1 直す箇所

| 題材 | ファイルと位置 | 直し方 | 確定事項 |
| --- | --- | --- | --- |
| 554849 | apps/README.md の 52 行 | `anet-lab-docs-<version>.zip` を `anet-lab-doxygen-<version>.zip` にする | D1 |
| c1c608 | docs/design/030_user_guide_analysis.jp.md の 20 行・525 行、docs/design/160_applications_and_tools.jp.md の 109 行・444 行 | リンク先の `../optuna.md` を `optuna.md` にする | D2 |
| 5a5373 | core/anet-core/src/replay_buffer_impl.hpp の 325 行 | `@note` の関数の並びに `GetStats` を足す | D3 |
| 2751c8 | apps/runner/config/nn.txt の 240〜241 行、275〜304 行 | 消す | D4 |
| 2751c8 | apps/runner/config/nn.txt の 16〜41 行 | type 一覧を 29 種に合わせ、`Add` を外す | D5 |
| 2751c8 | apps/runner/config/nn.txt の 5〜10 行、311〜312 行 | init.mode のコメントを 6 種にし、「1=Xavier」を直す | D6 |
| 2751c8 | apps/runner/config/nn.txt の 362〜368 行、376 行 | `(=Tag)`・`(@Source)` の行を消し、bind への案内を置く。376 行の MEMO を直す | D7 |
| 2751c8 | docs/design/130_neural_networks.jp.md の 94 行、docs/design/130_neural_networks.en.md の 1 行・95 行 | 出力 tag・入力 tag 参照の記述を直し、英語版のヘッダの blob と日付を更新する | D8 |
| 879a70 | apps/runner/config/metrics_scalar.txt の 11〜14 行、core/anet-core/src/observers.cpp の 1410〜1414 行 | 記法の 1 行と参照、参照の 1 行に置き換える | D9 |
| ddd395 | apps/runner/config/Atari.txt の 1650 行 | `= 14.5` にする | D10 |
| ddd395 | apps/runner/config/Atari.txt の 1652〜1654 行・328 行 | 1652〜1654 行を消し、328 行の直前に説明を置く | D11 |
| ee32bd | docs/memo/done/069_metrics_trace_channel_20impl.md の 123 行・187 行 | 変えない。課題の対応内容に直さない理由を書く | D12・D14 |
| fb8a45 | docs/memo/done/009_dueling_dag_head_10prd.md の 14 か所、docs/memo/done/010_image_metrics_v2_12prd.md の 2 か所 | 変えない。課題の対応内容に直さない理由を書く | D13・D14 |

### 5.2 移行対象と、触らないもの

移行対象は、§5.1 の表のうち直し方が「変えない」でない行のファイルと、題材の 8 件の課題の記録（対応内容、AI の判定、fb8a45 の親）である。§5.1 の ee32bd・fb8a45 の行のファイルは、下の「触らないもの」に入る。振る舞いを変えるのは nn.txt のサンプルを消すことだけで、消すサンプルを選ぶ今の設定・テスト・bat は無い（§2.1 の 2751c8（サンプル））。設定の解決結果からは、`@MLP_RES` だけが使う block の 20 キーが消える（§2.1 の「設定の解決結果」）。`(=Tag)`・`(@Source)`・`Add(` を使う今の設定は、消すサンプルのほかに無い。`git grep -n -e "(=" -e "(@" -e "Add(" -- apps/runner/config` の当たりのうち、structure の値とその記法の説明は nn.txt の 241・302・366・367 行だけである。ほかの当たり（Atari.txt・DropMerge.txt・GridMaze.txt・LunarLander.txt・agent.txt の 10 行）は、コメントの中の丸括弧（`(@qr / @iqn)`・`(=640トークン)` など）で、structure と関係が無い。structure がタグを表せると文章で書くのは、nn.txt の 376 行の MEMO だけである。`git grep -n -i -e "タグ" -e "tag" -- apps/runner/config/nn.txt`（大文字と小文字を区別しない）の結果は 366・367・376 行で、366・367 行は D7 で消す記法の行である。ほかの設定ファイルは `git grep -n -e "タグ" -- apps/runner/config` で引き、当たりは nn.txt の 376 行だけだった。英語の `tag` をほかの設定ファイルまで引くと、metrics の tag や `eval_config_tag` などの structure と関係の無い語に当たるので、ほかの設定ファイルでは引かない。

触らないもの:

- PRD072 の固定 17 入力の golden（`core/anet-core/testdata/prd072/baseline/`）と manifest。固定 commit の設定を入力にするので、本 PRD の変更では変わらない（§2.1）。
- 過去の Run の設定の写し（`docs/experiments/**/config/` の `net.body.MLP_RES.structure` などの行、`apps/runner/workspaces/**/runs/`、`core/anet-core/testdata/prd092/config/`）と、workspace の写し `apps/runner/workspaces/AtariRnd-01/config/frozen/nn.txt`。当時の記録である（§3.2）。
- 過去の記録: PRD069 の実装記録（D12）、done/009・done/010（D13）、[done/043](done/043_release_packaging_10prd.md) の 14 行（当時の zip の名前）、[PRD001](done/001_iqn_10prd.md) の 271 行（type 一覧へ足したことの記録）、[PRD059 の実装記録](done/059_config_concept_tree_alignment_29impl_layer_biaxial.md) の 32 行（`@MLP_RES` の名前の記録）。
- PRD949 §2 の 15（D15）。
- [220](../design/220_atari_env.jp.md) の日本語版 166 行と英語版 167 行の `app.online.eval_panel.fps = 15`。Atari.txt の実効値 14.5 と前から食い違っていて、本 PRD は値を変えない（D10）。扱いは §8 の 2。
- 課題の単票の題・事象に書かれた当時の名前（`anet-lab-docs-`、`@ResNet` など）。課題の記録であり、課題の一覧と wiki の地図の課題の区画は単票から作り直される。

### 5.3 クラス一覧

足す・変える・消すクラスは無い。変えるのは文書・コメント・設定ファイルである。

### 5.4 issue の対応表

| issueのID | 題 | area | 現在のコードで確認した根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-554849](../management/issues/issue-20261103-554849.md) | apps/README.mdがDoxygen zipをanet-lab-docs-<version>.zipと書くが、実際はanet-lab-doxygen-<版>.zip | 311_dev_tools | apps/README.md:52、.github/workflows/release.yml:254 | D1 | A1 |
| [issue-20261103-c1c608](../management/issues/issue-20261103-c1c608.md) | 現行ガイドのoptuna参照5か所が実在しない相対パスを指す | 311_dev_tools | docs/design/030_user_guide_analysis.jp.md:20、docs/design/160_applications_and_tools.jp.md:109 | D2 | A2 |
| [issue-20261103-5a5373](../management/issues/issue-20261103-5a5373.md) | InitialPriorityCompleterのロックの前提の@noteがGetStatsに触れていない | 318_rl_core | core/anet-core/src/replay_buffer_impl.hpp:325 | D3 | A7 |
| [issue-20261103-2751c8](../management/issues/issue-20261103-2751c8.md) | nn.txtのサンプル（@ResNet・@MLP_RES）と、対応typeの一覧・init.modeのコメントが、登録済みのtypeと記法に追いついていない | 315_nn, 316_config | apps/runner/config/nn.txt:241、apps/runner/config/nn.txt:16、apps/runner/config/nn.txt:362、apps/runner/config/nn.txt:376、core/anet-core/src/nn_modules.cpp:3279、core/anet-core/src/nn_impl.cpp:1680 | D4〜D8 | A3・A5・A8 |
| [issue-20261103-879a70](../management/issues/issue-20261103-879a70.md) | metrics_scalar.txt冒頭のscalar記法のミニ仕様が古い（&step、@learn/@trainだけ、属性の欠落） | 317_metrics, 318_rl_core | apps/runner/config/metrics_scalar.txt:11、core/anet-core/src/observers.cpp:1410 | D9 | A6 |
| [issue-20261106-ddd395](../management/issues/issue-20261106-ddd395.md) | Atari のパネル fps 設定2箇所が既定葉検査に反する | 316_config, 110_runner_app | apps/runner/config/Atari.txt:1650、apps/runner/config/Atari.txt:328 | D10・D11 | A3・A4 |
| [issue-20261103-ee32bd](../management/issues/issue-20261103-ee32bd.md) | PRD069の検証出力2か所が消失しており実装記録のリンクを開けない | 311_dev_tools | docs/memo/done/069_metrics_trace_channel_20impl.md:123 | D12・D14 | A2 |
| [issue-20261103-fb8a45](../management/issues/issue-20261103-fb8a45.md) | 過去PRDのコード参照16か所が文書からの相対パスとして解決できない | 311_dev_tools | docs/memo/done/009_dueling_dag_head_10prd.md:12、docs/memo/done/010_image_metrics_v2_12prd.md:10 | D13・D14 | A2 |

## 6. 検証と受入

### A1. README の zip の名前

`git grep -n "anet-lab-docs-" -- apps/README.md` が 0 件で、`git grep -n "anet-lab-doxygen-<version>.zip" -- apps/README.md` が 1 件である。

### A2. リンク切れと過去の記録

- [memo README](README.md) の「リンクの注意」のリンク切れ検査を、`docs/design/030_user_guide_analysis.jp.md` と `docs/design/160_applications_and_tools.jp.md` の 2 ファイルに当て、`BROKEN` の行が 0 件である。
- `git diff HEAD -- docs/memo/done/069_metrics_trace_channel_20impl.md docs/memo/done/009_dueling_dag_head_10prd.md docs/memo/done/010_image_metrics_v2_12prd.md` が空である。
- [issue-20261103-ee32bd](../management/issues/issue-20261103-ee32bd.md) と [issue-20261103-fb8a45](../management/issues/issue-20261103-fb8a45.md) の対応内容に、直さない理由（D12）が書かれている。

### A3. 設定の解決結果

直す前と後の作業ツリーで、出荷設定の 8 つの env（`apps/runner/config/` の `CartPole.txt`・`GridMaze.txt`・`GridMaze_muzero.txt`・`ImageCls.txt`・`Atari.txt`・`LunarLander.txt`・`LunarLander_cpu.txt`・`DropMerge.txt`）を、それぞれ Run プロファイルも CLI の上書きも足さない既定の構成で解決し、解決結果の全キーと文字列値を比べる。

- どの env でも、差は `net.block.[Linear_Proj].`・`net.block.[Linear_Hid].`・`net.block.[Linear_Out].`・`net.block.[ResBlock].` で始まるキーが消えることだけである。ほかのキーの増減と値の変化は無い。
- Atari では、`app.online.train_panel.fps` が `0`、`app.online.eval_panel.fps` が `14.5` で、直す前と同じである。
- 解決の手段（hidden test の `[prd103-resolve]` か、短い Run の設定の写しか）は実装計画で決める。比べた env と、消えたキーの数を実装記録に残す。

PRD072 の固定 17 入力比較は、固定 commit の設定を入力にするので、この受入には使わない（§2.1）。

### A4. 既定葉の検査

`check_default_leaves.py` が errors=0 で、`check_default_leaves_test.py` が通る（手順は PRD072 の検証の「既定葉の `?=` 検査」）。

### A5. nn.txt のコメントと設計文書 130

- nn.txt の type 一覧の節（`Supported Module Types` を含む行の次の行から、`# 1. Block Definitions` を含む行の手前まで。直す前の作業ツリーでは 17〜45 行）に、nn_modules.cpp が `Register("<名前>"` で登録する 29 種の名前がすべてある。同じ節に、正規表現 `\bAdd\b` に当たる行が無い。
- nn.txt のファイルの先頭から type 一覧の節の前までのコメントと、`## init.mode` で始まる行に、`default`・`xavier`・`he`・`orthogonal`・`constant`・`trunc_normal` がある。`git grep -n "1=Xavier" -- apps/runner/config/nn.txt` が 0 件である。
- `git grep -n -e "(=" -e "(@" -e "タグ" -- apps/runner/config/nn.txt` が 0 件である（直す前は 241・302・366・367・376 行の 5 件）。nn.txt の `## MEMO:表現出来ないパターン` で始まる行から次の空行までに、`bind` を含む行がある。nn.txt の `## 記法` で始まる行から次の空行までに、`bind` と `130_neural_networks` を含む行がある。
- `git grep -n -e "出力tag" -e "入力tag参照" -- docs/design/130_neural_networks.jp.md` と `git grep -n -e "output tags" -e "input-tag references" -- docs/design/130_neural_networks.en.md` が、どちらも 0 件である（直す前は 1 件ずつ）。130 の英語版の 1 行目の `translated-from` の blob が、直したあとの `git hash-object docs/design/130_neural_networks.jp.md` と一致する。

### A6. scalar の記法の説明

- `git grep -n -e "&step" -e "&episode_step" -e "(default: @train) @learn || @train" -- apps/runner/config/metrics_scalar.txt core/anet-core/src/observers.cpp` が 0 件である（直す前は 4 件）。
- metrics_scalar.txt の「# ----- metrics」の行から、最初の `metrics.scalar.` の定義の行の手前までのコメント（直す前は 9〜14 行。ファイルの先頭の trace の説明は含めない）に、`key [$step_axis] [@event] [$target] [$runner_scope] [$ema] [interval:N] [ema_alpha:A] [clip:C]` と `140_observability` がある。
- `git grep -n -e "metrics.scalar.min.\[10_train" -- core/anet-core/src/observers.cpp` が 0 件で、`git grep -n -e "140_observability" -- core/anet-core/src/observers.cpp` が 1 件以上である（直す前は 0 件）。その行に `ParseMetricTokens` がある。

### A7. `@note`

replay_buffer_impl.hpp の `class InitialPriorityCompleter` の直前のコメントにある、`metadata_mutex_` を保持して呼ぶ関数を並べた `@note` の行に `GetStats` がある。

### A8. 消したサンプル

`git grep -n -e "@ResNet\." -e "@MLP_RES" -e "block.\[ResBlock\]" -e "Linear_Proj" -e "Linear_Hid" -e "Linear_Out" -e "MyReLU" -e "Linear_128" -- apps/runner/config` が 0 件である。workspace・`docs/experiments`・`core/anet-core/testdata/prd092` は過去の記録、`core/anet-core/testdata/prd072/baseline` は固定 commit の設定から採った golden なので、数えない（§5.2）。

### A9. ビルドと既存のテスト

コメントだけを変える .hpp・.cpp を含むので、Debug の全体ビルドが通る。`anet-core-test.exe "[shipped_config]"` と既定のテストの集合が、直す前と同じ結果になる（直す前に落ちるテストがあれば、同じものだけが落ちる）。

### A10. 課題の記録

題材の 8 件の対応内容に、直した内容（ee32bd・fb8a45 では直さない理由）と場所が書かれ、AI の判定が `残件無し` である（D14）。fb8a45 の `parent` が本 PRD の索引の課題である（D13）。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| リンク検査に除外の一覧を持たせ、過去の記録の切れたリンクを数えない形 | 不採用 | 除外の一覧という新しい保守対象が増える。過去の記録の切れたリンクは、D12・D13 で直さないと決めた |
| 過去の記録の切れたリンクの、リンクの記法だけを外す形 | 不採用 | 過去の記録に手を入れる例が増え、読める中身は増えない（D12） |
| nn.txt のサンプル `@ResNet`・`@MLP_RES` を今の記法で書き直す | 不採用 | MLP の残差を書ける type が無く、書き直すには新しい type か記法の実装が要る（§2.1、D4） |
| metrics_scalar.txt に要素ごとの語の一覧を写す | 不採用 | 語を足すたびに `ParseMetricTokens`・140 §6・設定ファイルの 3 か所を直すことになり、古い写しは値を黙って記録しない事故になる（D9） |
| structure の `(=Tag)`・`(@Source)`・`Add(...)` の記法の実装 | 範囲外（別の関心） | 小さな食い違いの直しとは別の関心で、C++ の変更とテストが要る。再開条件: 合流に足し算や名前付きの途中出力が要る NN の構成が出たとき（§8 の 1） |
| PRD934・PRD949・PRD117 の範囲 | 範囲外（別の関心） | §3.2 のとおり |

## 8. 未決

確定事項は §4 にある。残るのは、工程末に起票の候補として示す範囲外の事項 2 件だけである。

| # | 事項 | 区分 | 扱い |
| --- | --- | --- | --- |
| 1 | structure に `(=Tag)`・`(@Source)`・`Add(...)` の記法を実装するか | 範囲外（§3.2・§7） | 未解決。工程末に起票の候補として示す |
| 2 | 設計文書 220 の Atari の音声の説明が、EvalPanel の fps を `15` と書き、Atari.txt の実効値 14.5 と食い違う | 範囲外（題材の 8 件に入らない。§5.2） | 未解決。工程末に起票の候補として示す |

## 9. 依存と順序

- 先に要る PRD は無い。
- [PRD118](118_atari_run_profile_flatten_10prd.md) は Atari.txt を作り直す。どちらを先に入れてもよい。PRD118 が先なら、D10・D11 を移った先の行に当てる。
- [PRD117](117_warn_policy_and_unread_config_10prd.md) の読まれない葉の検出が先に入ると、`[ResBlock]` の `Composite` などが検出に当たりうる。本 PRD が先に消せば当たらない。順序はどちらでもよい。
- [PRD949](949_release_packaging_cmake_10prd.md) の再グリルは、本 PRD の実装の前後を問わず、§2 の 15 を今の README と照合し直す（D15）。

## 10. 参照

- 規約: [実装 品質確認観点](../quality/impl.md)「設定ファイル」と項番 20、[020](../design/020_user_guide_run.jp.md) §3、[AGENTS.md](../../AGENTS.md) のクリーンブレーク方針。
- 記法の正本: [140](../design/140_observability.jp.md) §6（Metric設定contract）、[130](../design/130_neural_networks.jp.md) §2.4（設定記法）。
- 検査: [PRD072 の検証](../../core/anet-core/testdata/prd072/README.md)、[memo README](README.md) の「リンクの注意」。
- 課題の記録: [anet-create-issue](../../.agents/skills/anet-create-issue/SKILL.md)、[anet-update-issue](../../.agents/skills/anet-update-issue/SKILL.md)。
