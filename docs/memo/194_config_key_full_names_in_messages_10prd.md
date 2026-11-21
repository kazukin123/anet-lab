# PRD194: エラー・WARN・INFO が示す設定キーを完全キーにする（device の二重の fatal、指標名、用語の不整合 6 件を含む）

> 2026-11-21 起票（anet-create-prd、Claude Code）。題材は課題 6 件（§10 の対応表）。グリルで、ゴールと非ゴール（§2）、共通部品が設定キーを名指す形（D1）、完全キーにする範囲（D2・D3）、完全キーを運ぶ仕組み（D4）、fatal の二重出力の直し方（D8）、device を解決した結果の語（D10）、指標名を合わせる向き（D11）、上書きの接頭辞が空の読み口の扱い（D18）をユーザーが選び、細部 12 件（§5.2）は推奨の既定で埋めた。起草のあと、別モデル（Opus）の検討を受け、コードで確かめて、仕組みの契約（§6.2）と一覧（§8）を直した。D18 は、その検討で見つかった事実（P24）から足した問いである。別モデルが結論に反対した決定は無い。起草と別モデルの検討で出た未決 8 件は、ユーザーが推奨の既定でまとめて埋めた（D19〜D25、§5.3）。

> 関連: 設計 [100](../design/100_runtime_and_configuration.jp.md)・[110](../design/110_agents_and_learning.jp.md)・[120](../design/120_environments.jp.md)・[130](../design/130_neural_networks.jp.md)・[140](../design/140_observability.jp.md)・[020](../design/020_user_guide_run.jp.md)、ADR [0030](../adr/0030-nn-config-tree-under-agent-prefix.md)・[0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)・[0087](../adr/0087-messages-name-config-keys-by-full-key-recorded-at-read-time.md)、[070_device_spec_auto_10prd.md](done/070_device_spec_auto_10prd.md)、[117_warn_policy_and_unread_config_10prd.md](117_warn_policy_and_unread_config_10prd.md)、[128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md)、[155_config_root_ownership_alignment_10prd.md](155_config_root_ownership_alignment_10prd.md)、[071_uncaught_fatal_visibility_10prd.md](071_uncaught_fatal_visibility_10prd.md)

> 2026-11-21 改訂（実装時）。§8.2 の設定欄名の採取範囲とログの始まりを補正し、§8.3 の基準集計 303 文・64 呼び出しを再現できる形にした。完全キーの照会・記録に渡す相対キーも集計から除く。RainbowAgent の購読処理を維持する開発者の選択に従い、RainbowAgentConfig で構築した共通 Learner で診断の文面を確かめる形に A4 と D24 の説明を補正した。

## 1. 問題

### 1.1 経緯

2026-09〜11 の監査と、PRD158・PRD182 の工程で、設定の誤りを知らせる文面と、用語・検査の期待が、利用者が実際に書く設定や出荷定義と食い違う小さな不整合が 6 件見つかった（§10）。うち 2 件（eeb2c3・4124b6）は、「共通の部品が、持ち主が別にいる設定キーを文面でどう名指すか」という同じ問いに帰着する。

グリルで、この問いの答え（D1）を題材の 6 か所に限らず、`src` のテスト以外のエラー・WARN・INFO の全部へ広げることをユーザーが選んだ（D2・D3）。そのため本書は、題材の 6 件の直しと、設定キーを運ぶ仕組み、基盤全体の文面の移行を扱う。

本書は、root から末尾までを書いた、設定ファイルにそのまま書ける設定キーを「完全キー」、部品や切り出した設定の中だけで通じる名前を「相対キー」と呼ぶ（§4）。グリルの会話では「全名」と呼んでいた。

課題の事象に書かれたパス（`core/anet-core/...`）と行番号の一部は、[182_source_layout_10prd.md](182_source_layout_10prd.md)（ソース配置の見直し）より前のものである。本書の根拠は、2026-11-21 に今のコードで確かめ直した。

### 1.2 今のコードで確かめたこと（題材の 6 件）

行番号は 2026-11-21 のもの。6 件とも今のコードに残っている。

| 課題 | 今のコード |
|---|---|
| 78df9c | `InitRL` は、`agent.device`・`env.device`・`run.eval_device` のうち設定に書かれたキーを `ParseDevice` に通し、例外を捕まえて、キーの名前を添えた `ANET_SYSTEM_ERROR` で投げ直す（[init.cpp](../../src/core/base/src/init.cpp) 101〜105 行）。`ParseDevice` は不正な指定を、キーの名前を持たない文面の `ANET_SYSTEM_ERROR` で投げる（[tensor_util.cpp](../../src/core/base/src/tensor_util.cpp) 29 行）。`ANET_SYSTEM_ERROR` の実体の `ThrowError` は、投げる前に必ず `LOG::fatal` を出す（[diag.cpp](../../src/core/base/src/diag.cpp) 22 行）。同じ誤りの fatal が 2 回出て、2 回目は 1 回目の全文を入れ子に含む |
| 78df9c | `ParseDevice` を呼ぶのは、テスト以外で 4 か所である（[init.cpp](../../src/core/base/src/init.cpp) 102 行、[agent.cpp](../../src/core/base/src/agent.cpp) 96 行、[env.cpp](../../src/core/base/src/env.cpp) 1026 行、[trainer.cpp](../../src/core/base/src/trainer.cpp) 829 行。`src` を `ParseDevice` で引いて 12 行。残りは宣言 1、定義 1、テスト 6） |
| 9b314a | 明示の cuda が使えないときの一括のエラーは、キーの名前だけをカンマで並べる（[init.cpp](../../src/core/base/src/init.cpp) 110〜118 行）。各キーに書いた値（`cuda`・`cuda:1` など）は出ない |
| eeb2c3 | `ResolveVideoCodec` のエラーは、`metrics_logger.video_codec` という名前を 2 か所に書く（[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 97・101 行）。利用者が書くキーは `app.metrics_logger.video_codec` である。読むのは RunnerApp の設定の読み口だけで（[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 116 行）、基盤の MetricsLogger は root を知らない型（`MetricsLoggerConfig`、[metrics_logger.hpp](../../src/core/base/include/anet/metrics_logger.hpp) 171 行）で値だけを受け取る。codec の可否は動画の大きさで決まる（[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 95 行）ので、設定を読む時点では判定できない。[metrics_logger_test.cpp](../../src/core/base/src/metrics_logger_test.cpp) 284 行がこの文面を期待している |
| d4258c | device 指定を解決した結果を「採用値」と書く行が 5 つある（[020](../design/020_user_guide_run.jp.md) 67 行、[100](../design/100_runtime_and_configuration.jp.md) 338・346 行、[110](../design/110_agents_and_learning.jp.md) 191 行、[120](../design/120_environments.jp.md) 218 行）。同じ意味で「採用したdevice」と書く行が 1 つある（[020](../design/020_user_guide_run.jp.md) 623 行）。`docs/design/*.jp.md` を「採用値」で引くと 6 行で、残る 1 行（[100](../design/100_runtime_and_configuration.jp.md) 326 行）は評価セッションの採用エピソードの値である。テストのコメントにも、同じ意味の「採用値」が 1 行ある（[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) 659 行）。追跡しているファイルの全体を「採用値」で引くと 15 ファイル 23 行で、device の意味はこの 6 行（設計文書 5、テスト 1）だけである。残る 17 行は、100 の 326 行、用語集の避ける語の行 1、探索で選んだ値を言う設定のコメント 1（[Atari.txt](../../apps/runner/config/Atari.txt) 798 行）、workspace の履歴の値を言うコードのコメント 1（[app_util.cpp](../../src/core/base/src/app_util.cpp) 485 行）、課題記録 8、課題の題を写す wiki 2、過去の PRD と実装記録 2、サーベイ 1 である。記録する JSON の欄は `effective_device`・`effective_eval_device` である（[trainer.cpp](../../src/core/base/src/trainer.cpp) 829・830 行） |
| 546bdb | hidden のテスト `[prd103-config]` は、`metrics.scalar.[50_agent_skip/19_primary_clip_ratio_ema]` が空でないことを断定する（[dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) 108 行）。出荷定義の tag は `50_agent_skip/19_skip_primary_clip_ratio_ema` である（[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 391 行）。設計文書もテストと同じ名前を書く（[140](../design/140_observability.jp.md) 431 行）。出荷定義の名前は、実験記録の設定の写し 36 本に入っている（`docs/experiments` を `19_skip_primary_clip_ratio_ema` で引いたファイルの数。テスト側の名前は 0 本）。指標を足したコミット 2 つ（5764ca10・7ac66c63）の件名は、出荷定義の名前を書く。追跡しているファイルの全体をテスト側の名前（`19_primary_clip_ratio_ema`）で引くと 5 ファイル 7 行で、直すのはテスト 1 行と設計文書（日本語版）1 行、残すのは英訳 1 行、課題記録 2 行、PRD158 の実装記録 2 行である |
| 546bdb | 出荷定義の tag は群名と `/` を含めて 44 文字、テスト側の名前は 39 文字である。兄弟の指標は `50_agent_skip/17_skip_primary_grad_norm`（39 文字）と `50_agent_skip/18_skip_primary_clip_ratio`（40 文字）である。出荷定義には、40 文字を超える tag が 65 個ある（[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の `[群名/名前]` を重複なく数えた 280 個のうち） |
| 4124b6 | DQN の共通部品は PRD182 でファイルが分かれた。`DefaultDQNAgent.` を固定で書くエラーは、[dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) 110 行（`skip.network_mode`）と、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363 行（`learner.noisy_probe`）・395 行・399 行（`learner.final_feature_branch_key`）の 4 か所である。`src/algo/dqn` のテストと具象 2 Agent のファイルを除いて `DefaultDQNAgent` を引くと 5 行で、残る 1 行は登録（[dqn.cpp](../../src/algo/dqn/src/dqn.cpp) 11 行）である |
| 4124b6 | RainbowAgent は skip 網を渡さず、`learner.final_feature_branch_key` と `learner.noisy_probe.*` を読まない（[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp)、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp)）。RainbowAgent は指標購読を共通 Learner へ転送しない。3 か所の診断は、RainbowAgentConfig で組み立てた共通 Learner の `ConfigureScalarMetricSubscriptions` に該当の購読を渡して検証する（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp)） |
| 4124b6 | 363 行は DefaultDQNAgent からは届かない。DefaultDQNAgent は、Noisy の層がある網には必ず probe の実行設定を作って渡す（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 489〜497 行） |

### 1.3 今のコードで確かめたこと（基盤全体の文面と、設定の読み口）

2026-11-21 に、`src` のテスト以外のエラー・WARN・INFO の文を、§8.2 の条件で全件拾い、1 文ずつ読んで仕分けた。

- 設定キーらしき名前を含む文は 303 文、44 ファイルである。うち、相対キーで設定を名指している文が 178、対象外が 125 である（§8.3）。
- 検証の関数などへ、相対キーを文字の引数で渡している行は 64 行、13 ファイルである。うち、渡した相対キーが文面に出るものが 16 行、対象外が 48 行である（§8.3）。
- 集計に現れないが直す文が 8 つある（§8.5）。直すものは、合わせて 202 である。
- 直す 178 文の種類は、エラー 134、WARN 19、INFO 25 である。
- 直す 178 文と 16 行のある場所と、完全キーの得かた（§8.1 の凡例）は次のとおり。

| 文のある場所 | 文の数 | 呼び出しの行 | 完全キーの得かた |
|---|---|---|---|
| 設定の読み口とその持ち主（具象の Agent・Env、Agent と Env の factory） | 37 | 2 | a: 自分の読み口が記録した完全キー |
| 設定の型を受け取る共通部品（DQN の共通部品、RND、Replay、Observer、scaler、ProfiledValue、MetricsLogger） | 57 | 8 | b: 設定の型と一緒に運ばれた完全キー |
| NN の block と forward の実行設定、`ConfigData` 自身のエラー | 31 | 6 | c: `ConfigData` の項目が覚えている完全キー |
| カタログの項目を指す識別子（ImageDataset、評価の定義、Actor のカタログ） | 35 | 0 | d: 項目の完全キー |
| 素の値や名前を引数で受ける関数（状態の sweep、ProfiledValue の検証、指標の間隔） | 18 | 0 | e: 呼ぶ側が完全キーを渡す |

設定の読み口について確かめたこと。

- 設定の値は、読み口（`anet::Config` の派生）が完全キーを組んで読み、型の欄に詰めて部品へ渡す。渡ったあとの欄は名前を持たない。読み口は、既定の置き場と上書きの置き場（評価用 Env の `run.eval.[tag].env.*` など）を順に読む（[config.hpp](../../src/core/base/include/anet/config.hpp) 224〜238 行）。
- 上書きの接頭辞が空の読み口では、上書きのキーが相対キーそのものになる（[config.hpp](../../src/core/base/include/anet/config.hpp) 285〜288 行）。その結果、root 直下に `learner.per_eps = -1` と書くと、DefaultDQNAgent と RainbowAgent の両方がその値を読む。読んだ扱いになるので、読まれない設定キーの一覧にも出ない。既定の接頭辞だけを渡して作る読み口は、`src` のテスト以外に 8 つある（`Config(<設定>, "<接頭辞>")` の形の初期化を引いた数）。上書きの接頭辞を引数で受ける読み口も 8 つあり、空の接頭辞を渡されると同じになる。
- 出荷設定は、root 直下の相対キーを使っていない。`apps/runner/config/*.txt` の、コメントと `$` の行を除いた行頭の語は、root（`net`・`run`・`metrics`・`app`・`image`・`backend`・`env`・`agent`）、型名（`DefaultDQNAgent`・`RainbowAgent`・`MuZeroAgent`・`ImageClsAgent`・`AtariEnv`・`DropMergeEnv`・`LunarLanderEnv`・`GridMazeEnv`・`ImageClsEnv`・`CartPoleEnv`・`ImageDataset`・`Conv2dPanel`・`QValuePanel`）、上書き層（`A1`〜`A3`・`E1`・`M1`）、`@vars` だけである。番号付きの bat（`apps/*.bat`）が引数で渡す設定にも、root 直下の相対キーは無い（`learner.`・`skip.`・`intrinsic.`・`actor.`・`head.`・`model.`・`target_policy.`・`metrics_logger.`・`reward_scaler.` で始まる代入を引いて 0 件）。
- `ConfigData` は、切り出しや合成を経ても、項目ごとに元の項目へのつながりを持つ。読まれない設定キーの検出（PRD117）が使う（[config.cpp](../../src/core/base/src/config.cpp) 601〜615 行の `MakeSubConfigData`、[config.hpp](../../src/core/base/include/anet/config.hpp) 40〜51 行）。既存の項目へ値を書き込んでも、つながりは残る（[config.cpp](../../src/core/base/src/config.cpp) 297〜301 行）。設定のプロファイルの補間は、既存の項目へ値を書き込む（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1694 行）。
- `ConfigData` 自身が出すエラーは、その `ConfigData` が持つキーをそのまま書く。型の読み取りの失敗（[config.cpp](../../src/core/base/src/config.cpp) 42〜48 行）、読まれないキー（353〜371 行）、合成の食い違い（505〜513 行）である。切り出した設定では、相対キーになる。
- NN の block の設定は、共有の `net.block.[X]` と Agent の下の `<Agent>.net.block.[X]` を合成し、相対キーだけで block に渡る（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 808〜829 行）。block の `type` は、block に渡す設定から除かれる（822〜825 行）。block を作る口は、block の設定と乱数の源だけを受け取る（[nn_impl.hpp](../../src/core/base/src/nn_impl.hpp) 246〜256 行）。block の設定を読む読み口には、既定の接頭辞が空のものがある（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 787・798 行、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 244 行）。
- dump に記録しない読み取りがある（ProfiledValue の各欄。[schedule.hpp](../../src/core/base/include/anet/schedule.hpp) 284〜289 行）。読み口が、読んだ値を別の欄へコードで写す箇所がある（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 639・704・742 行の `quantile_mode`、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 87 行の `num_quantiles`）。
- anet の例外を捕まえ、文脈を足して `ANET_SYSTEM_ERROR` で投げ直す箇所は 6 つある（`src` のテスト以外で `.what()` を含む 19 行を読んだ）。設定キーや置き場の名前を足すための投げ直しが 3 つ（[init.cpp](../../src/core/base/src/init.cpp) 104 行、[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 204・293 行）、内側からは分からない文脈を足す投げ直しが 3 つ（[trainer.cpp](../../src/core/base/src/trainer.cpp) 55 行、[GridMazeEnv.cpp](../../src/env/gridmaze/src/GridMazeEnv.cpp) 42 行、[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) 730 行）である。NN の 2 つの内側のエラーは、`ConfigData` の型の読み取りの失敗を通る。
- 変わる文面を期待しているテストは、少なくとも 9 ファイル 44 行ある（§6.6 の検索）。

## 2. ゴールと非ゴール

### 2.1 ゴール

- G1: 題材の課題 6 件がすべて片付く。
- G2: `src` のテスト以外のエラー・WARN・INFO が設定キーを示すときは、利用者が設定に書く完全キーで示す。device の文法の誤りのエラーは 1 回だけ出る。
- G3: device を解決した結果を指す語が設計文書で 1 つに決まり、評価の採用エピソードの語と紛れない。
- G4: `[prd103-config]` が 4 環境で通り、指標名がテスト・出荷定義・設計文書で一致する。
- G5: 設定に書いて読み口に効くのは、完全キーだけである。root 直下に書いた相対キーは、どの読み口にも効かない。

G1〜G4 は、グリルの初めに合意した（G2 は、範囲の問いの答えで、題材の 6 か所から基盤全体へ書き換えた）。G5 は、別モデルの検討で見つかった事実（P24）について、ユーザーが本書で直すと選んだこと（D18）を、ゴールとして書いたものである。

### 2.2 非ゴール（退けた案と、本当の制約か）

| 非ゴール | 退けた案 | 本当の制約か |
|---|---|---|
| 設定キーと root の改名 | `app.metrics_logger.*` を別の root へ移して文面と合わせる | 制約ではない。クリーンブレークで移せる。別の関心なので、[155_config_root_ownership_alignment_10prd.md](155_config_root_ownership_alignment_10prd.md) が扱う |
| 正しい設定の Run の挙動を変えること | 指標の tag を 40 文字に収まる名前へ変える（D11 の退けた案） | 制約ではない。変えられるが、10 月 29 日以後の Run の記録と tag が分かれる。本書は tag を変えない。D18 が変えるのは、root 直下に相対キーを書いた設定の読み取りだけで、出荷設定は当たらない（§1.3） |
| fatal の出し方全体の作り直し | 文脈を足して投げ直す 3 か所（§1.3）も、ログを重ねない口を作って直す。`ThrowError` がログを出さず、いちばん外で 1 回だけ出す形にする | 制約ではない。fatal の出し方は [071_uncaught_fatal_visibility_10prd.md](071_uncaught_fatal_visibility_10prd.md) の関心で、設定キーの名指しとは別である |
| RainbowAgent に設定キーを足して診断を使えるようにすること | `RainbowAgent.learner.final_feature_branch_key` を読むようにする | 制約ではない。機能の追加で、文面の直しとは別である |
| 40 文字を超えるほかの tag（65 個）の整理 | 546bdb の直しと一緒に、長い tag を縮める | 制約ではない。Run の記録との重なりを切る変更で、別に扱う |

### 2.3 足す仕組みとゴールへの寄与

| 足すもの | ゴールへの寄与 |
|---|---|
| `ConfigData` が、項目ごとの完全キーと、切り出した設定の置き場を覚える（D4） | G2。NN の block と forward の実行設定、`ConfigData` 自身のエラーが、合成のあとでも書かれた置き場を示せる |
| 読み口が、読んだ欄ごとに完全キーを記録し、設定の型と一緒に部品へ運ぶ（D4） | G1（eeb2c3・4124b6）と G2。共通部品が、利用側の名前を固定で持たずに完全キーを書ける |
| 上書きの接頭辞が空の読み口は、上書きの読み取りをしない（D18） | G5。G2 の文面に出る完全キーだけが、設定に書いて効くキーになる |
| `ParseDevice` がキーの名前を受け取る（D8） | G1（78df9c）と G2。投げ直しが要らなくなり、fatal が 1 回になる |
| 用語「実効 device」「完全キー」（D10・§4） | G3、G2 の規約の語 |
| 規約 1 つ（D12） | G2。直したあとに、相対キーの文面が増えない |

新しい設定キー、メトリクス、TensorDict キーは足さない。

### 2.4 成功の測り方

- 誤った設定で止まったときのログに、同じ誤りの fatal が 1 件だけ出て、文面の設定キーを設定ファイルに貼ると、そのまま効く（§7 の A1〜A4・A8）。
- §8.2 の集計を流し直して、相対キーで設定を名指す文と呼び出しが残らない。集計の出力が、§8.7 の期待の一覧と一致する（§7 の A9・A10）。
- `[prd103-config]` が通る（§7 の A6）。
- root 直下に書いた相対キーが、読まれない設定キーの一覧に出る（§7 の A14）。

## 3. 前提

グリルで示した前提と、起草（別モデルの検討を含む）で足した前提。確かめの列は、共通の前提（P1〜P7）が「済み」、1 つの論点だけの前提と、起草で足した前提（P26〜P29）が「対象外」である。

| 番号 | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実 | `ThrowError` は投げる前に必ず `LOG::fatal` を出す。anet の例外を捕まえて `ANET_SYSTEM_ERROR` で投げ直すと、同じ誤りの fatal が 2 回出る | [diag.cpp](../../src/core/base/src/diag.cpp) 22 行 | 済み |
| P2 | 守る線 | エラーには対象の識別子（キー）・指定値・期待値を入れる。完全キーで書くことまでは決めていなかった | [AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」の末尾の箇条、[impl.md](../quality/impl.md) の項番 27 | 済み |
| P3 | 過去の判断 | 共通の検証の関数が出すキーの文字は、呼ぶ側が渡す | [128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md) の §5 D6 | 済み |
| P4 | 守る線 | 汎用の層と共通部品は、特定の Agent の名前を持たない | [AGENTS.md](../../AGENTS.md) の「汎用機構と利用側の責任境界」、[128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md) の §5 D3 | 済み |
| P5 | 事実 | DQN の共通部品で設定キーを書くエラー文は 2 通りある。`DefaultDQNAgent.` を固定で書くものが 4 か所、相対キーで書くものがある。相対キーの数は、グリルの途中と起草で数え直した（P10） | §1.2・§8.3 | 済み |
| P6 | 事実 | RainbowAgent は skip 網を渡さず、`learner.final_feature_branch_key` と `learner.noisy_probe.*` を読まない。RainbowAgent は指標購読を共通 Learner へ転送しない。RainbowAgentConfig で組み立てた共通 Learner に該当の指標購読を要求すると、読み取り記録が無い欄の診断を検証できる | §1.2 | 済み |
| P7 | 事実 | `app.metrics_logger.*` を読むのは RunnerApp で、基盤の MetricsLogger は root を知らず値だけを受け取る。codec の可否は動画の大きさで決まるので、設定を読む時点では判定できない | §1.2 | 済み |
| P8 | 既存部品の働き | 設定を読む時点の検証は、呼ぶ側が完全キーを渡す関数で行っている。キーの名前を型に入れて後の時点まで運ぶ例もある | [rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 35 行、[nn.hpp](../../src/core/base/include/anet/nn.hpp) 58・109 行 | 対象外 |
| P9 | 事実 | 4 か所のうち 363 行は DefaultDQNAgent からは届かず、110 行は DefaultDQNAgent だけが通る | §1.2 | 対象外 |
| P10 | 事実 | 相対キーで設定を名指す文と呼び出しは、基盤全体で 202 ある。グリルの初めの見積もり（階層つきの名前 53 個）と、グリルの途中の集計（階層つき 106 個、1 語 102 個）は、どちらも数え方に漏れがあった | §1.3・§8 | 対象外 |
| P11 | 事実 | NN の部品の一部は、置き場の名前を別に示して、相対キーで書く形をとっている。block は自分の設定の置き場を受け取っていない | [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 204 行、§1.3 | 対象外 |
| P12 | 事実 | 1 語だけで出る設定の名前は、直すキーを指す書き方と、値の添え書きが混ざる。検索では分けられない | §8.3 | 対象外 |
| P13 | 既存部品の働き | 読み口は、既定の置き場と上書きの置き場を順に読む。どちらから値を得たかを知っているのは読み口だけである | §1.3 | 対象外 |
| P14 | 既存部品の働き | `ConfigData` は、切り出しや合成を経ても、項目ごとに元の項目へのつながりを持つ | §1.3 | 対象外 |
| P15 | 事実 | NN の block の設定は、共有の置き場と Agent の下の置き場を合成し、相対キーだけで block に渡る。どちらに書いた値かは block からは見えない | §1.3 | 対象外 |
| P16 | 事実 | anet の例外を捕まえ、文脈を足して投げ直す箇所は 6 つある | §1.3 | 対象外 |
| P17 | 事実 | 6 つのうち 3 つは設定キーや置き場の名前を足す投げ直しで、残る 3 つは内側からは分からない文脈を足す | §1.3 | 対象外 |
| P18 | 用語 | 「実効」は、設定を解決した結果（`config/config_data.txt`）を指す語として使われている。device の `auto` は、そこでは `auto` のまま残る | [AGENTS.md](../../AGENTS.md) の「AI エージェントのRun結果分析ルール」、[CONTEXT.md](../../CONTEXT.md) の「読まれない設定キー」、[100](../design/100_runtime_and_configuration.jp.md) 346 行 | 対象外 |
| P19 | 事実 | 記録する JSON の欄の名前は `effective_device` と `effective_eval_device` である。PRD070 は「実効 device」「実効値」の両方で書いている | [070_device_spec_auto_10prd.md](done/070_device_spec_auto_10prd.md) の D9 と受入 5 | 対象外 |
| P20 | 用語 | 「採用」は、評価セッションの採用エピソードの語である | [CONTEXT.md](../../CONTEXT.md) の「採用エピソード」 | 対象外 |
| P21 | 守る線 | metrics の tag は、群名と `/` を含めて 40 文字以内にする。今の出荷定義には、超える tag が 65 個ある | [impl.md](../quality/impl.md) の「設定ファイル」の規約、§1.2 | 対象外 |
| P22 | 事実 | 出荷定義の tag は 44 文字、テストと設計文書の名前は 39 文字である | §1.2 | 対象外 |
| P23 | 事実 | 出荷定義の名前は、実験記録の設定の写し 36 本とコミットの件名にある。テスト側の名前は 0 本である | §1.2 | 対象外 |
| P24 | 事実 | 上書きの接頭辞が空の読み口は、root 直下の相対キーを読む。root 直下に `learner.per_eps` と書くと、2 つの DQN Agent の両方に効き、読まれない設定キーの一覧にも出ない | §1.3 | 対象外 |
| P25 | 事実 | 出荷設定は、root 直下の相対キーを使っていない | §1.3 | 対象外 |
| P26 | 事実 | `ConfigData` 自身が出すエラーは、切り出した設定では相対キーを書く。NN の 2 つの投げ直しの内側のエラーは、これを通る | §1.3 | 対象外 |
| P27 | 事実 | block の `type` は block に渡す設定から除かれる。block の設定を読む読み口には、既定の接頭辞が空のものがある | §1.3 | 対象外 |
| P28 | 事実 | 既存の項目へ値を書き込んでも、元の項目へのつながりは残る。設定のプロファイルの補間は、既存の項目へ書き込む | §1.3 | 対象外 |
| P29 | 事実 | dump に記録しない読み取りがある。読み口が、読んだ値を別の欄へコードで写す箇所がある | §1.3 | 対象外 |

## 4. 用語

[CONTEXT.md](../../CONTEXT.md) に 2 語を足した。

- **完全キー**: root から末尾までを書いた、設定ファイルにそのまま書ける設定キー。設計文書 100 の §2.1 と `config.hpp` がすでに使っている語で、用語集に無かった。既存の語なので、言い換えは要らない。
- **実効 device**: `auto` を含む device 指定を解決して、実際に使う device（D10）。「実効値」は設定を解決した値と紛れ、「採用値」は採用エピソードと紛れるので、既存の語では言い換えられない。

「相対キー」は、設計文書 100 と `config.hpp` が使う語で、本書も同じ意味（部品や切り出した設定の中だけで通じる名前）で使う。

本書は、切り出した設定が元にあった場所の、root からの接頭辞を「置き場」と呼ぶ（たとえば `net.block.[CN1]`）。用語集には足さない。設計文書 100 が同じものを prefix と書いていて、本書の中の説明の語である。

## 5. 確定事項

2026-11-21 のグリルで決めた。D1〜D4・D8・D10・D11・D18 はユーザーが案を比べて選んだ。D5・D6・D15〜D17 は、ユーザーがまとめて採った推奨の既定である（§5.2）。D7・D9・D12〜D14 は、規約と決定から導いて一覧で示し、ユーザーが戻さなかった。D19〜D25 は、起草と別モデルの検討で出た未決について、ユーザーがまとめて採った推奨の既定である（§5.3）。

| 番号 | 決定 | 理由と前提 |
|---|---|---|
| D1 | **共通部品が設定キーを名指す形。** 共通部品は、利用側が読んだ設定の完全キーだけで設定キーを書く。利用側が読んでいない設定を、完全キーの形では名指さない | ユーザーの選択。P3 と同じ向きである。P6 のもとで、接頭辞を渡して共通部品が欄の名前を足す形は、RainbowAgent が読まないキーを指す。依存: P3・P4・P6 |
| D2 | **完全キーにする範囲。** `src` のテスト以外のエラー・WARN・INFO の全部を対象にする。題材の 6 か所に限らない | ユーザーの選択（推奨は題材の 6 か所だけだった）。形の不ぞろいを残さない。依存: P10・P11 |
| D3 | **1 語だけで出る設定の名前。** `quantile_mode=iqn` のように 1 語で出る設定の名前も、完全キーにする。対象かどうかは §8.1 の規則で 1 文ずつ決める | ユーザーの選択（推奨は階層つきの名前だけだった）。依存: P12 |
| D4 | **完全キーを運ぶ仕組み。** 読み口が自動で記録する。`ConfigData` は、項目ごとの完全キーと、切り出した設定の置き場を、切り出し・写し・合成・読み取り範囲の作成を経ても覚える。`anet::Config` は、読んだ欄ごとに、値を与えた完全キーを記録する。部品は、設定の型と一緒に運ばれた記録を、欄の相対キーで引く。契約は §6.2 | ユーザーの選択。P13 のとおり、上書きの置き場から読んだかを知っているのは読み口だけである。P15 のとおり、block がどちらの置き場に書かれた値かを示せるのは、`ConfigData` が覚える形だけである。P14 の、元の項目へのつながりを流用できる。依存: D1・D2・P13・P14・P15 |
| D5 | **記録を引いた結果。** §6.2 の 2 つの表による。記録に無い相対キーを引くと、相対キーに ` (not read from config)` を添えた文字を得る。既定と上書きの両方の置き場があるときは、値を与えた側の完全キーを得る。どちらにも書かれていなければ、既定の置き場の完全キーを得る。NN の block で、どこにも書かれていない設定は、その block の `type` の値を与えた置き場（両方に書かれていれば Agent の下）に相対キーを続けて示す | 推奨の既定（細部 1〜3）。形が 1 つで済み、部品ごとに文面を作らない。書かれていない設定は、利用者が書くならその置き場になる。P27 のとおり `type` は block の設定に残らないので、`ConfigData` が置き場を覚える形で示す。依存: D1・D4・P27 |
| D6 | **カタログの項目を指す識別子。** `dataset_key='food101'` や `eval.[greedy]` のように項目を指す書き方は、項目の完全キー（`ImageDataset.[food101]`、`run.eval.[greedy]`）で書く | 推奨の既定（細部 4）。設定に書く名前はこちらである。依存: D3 |
| D7 | **WARN と INFO。** エラーと同じに扱う | 同じ関数の中でエラーと WARN が同じキーを使うので、エラーだけ直すと形が混ざる（[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 244〜268 行）。P2 の守る線は、エラーと WARN を同列に扱う。依存: D2・P2 |
| D8 | **fatal の二重出力。** `ParseDevice` がキーの名前を必須で受け取り、文面にそのキーと指定値を書く。`InitRL` は例外を捕まえない。NN の投げ直し 2 つ（[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 204・293 行）は、`ConfigData` 自身のエラーが完全キーを書くようになったあとに消す。文脈を足す投げ直し 3 つは変えない（§2.2） | ユーザーの選択。P17 のとおり、名前を足すための投げ直しは、名前を渡せば要らなくなる。名前なしの呼び出しを残すと、キーの無いエラーが残るので、名前は必須にする（細部 5）。P26 のとおり、NN の投げ直しを先に消すと、置き場の名前が文面から落ちるので、順を守る。依存: D1・D4・P1・P16・P17・P26 |
| D9 | **明示 cuda のエラーの形。** `agent.device='cuda:1', run.eval_device='cuda'` のように、`キー='値'` をカンマと空白で並べる。値は設定に書かれたままの文字にする | 同じ関数の文法の誤りのエラーの形（`Invalid agent.device='…'`）と、PRD128 D6 の語の形に合わせる。P2 の、指定値を入れる守る線による。依存: P2・P3 |
| D10 | **device を解決した結果の語。** 「実効 device」と書く。直すのは §1.2 の 7 行（設計文書の「採用値」5 行と「採用したdevice」1 行、テストのコメント 1 行）である。評価の採用エピソードの値を言う「採用値」（100 の 326 行）は変えない | ユーザーの選択。P19 のとおり JSON の欄と PRD070 の語に合う。P18 のとおり「実効値」だけだと、設定の実効値（`auto`）と同じ語で別の値を指す。P20 のとおり「採用」は別の概念の語である。依存: P18・P19・P20 |
| D11 | **指標名を合わせる向き。** テスト（1 行）と設計文書（1 行）を、出荷定義の `19_skip_primary_clip_ratio_ema` に合わせる。出荷定義の tag は変えない。`[prd103-config]` は hidden のまま、置き場を変えない。PRD103 の前後比較の道具（[verify_config_migration.py](../../src/core/base/testdata/prd103/verify_config_migration.py)）は直さない | ユーザーの選択。P23 のとおり、Run の記録とコミットの件名は出荷定義の名前で、テストと設計文書のほうが書き違いである。過去の Run と同じグラフに重なるまま保てる。P21 の 40 文字の超過は、65 個をまとめて扱うときに一緒に扱う（§2.2）。出荷設定の値を固定するテストは hidden の採取テストにする決まりである（[impl.md](../quality/impl.md) の「テスト」の規約）。前後比較の道具は PRD103 の変更の前後を比べるもので、この指標はそのあとに足された（細部 9）。依存: P21・P22・P23 |
| D12 | **規約。** [impl.md](../quality/impl.md) の「失敗の扱い」に、規約の文と項目 1 つを足す。文は「エラー・WARN・INFO が設定キーを示すときは、利用者が設定に書く完全キーで書く。部品は、設定の型と一緒に運ばれた記録から引き、利用側が読んでいない設定を完全キーの形で名指さない」とする。機械の検査は足さず、読んで確かめる項目にする | 規約に足さないと、次の変更でまた相対キーの文面が増える。P12 のとおり、1 語だけの名前は検索で判定できないので、機械の検査にはしない（細部 7）。依存: D2・D3・P12 |
| D13 | **意図して変える点。** 変えるのは、§6.4 の「意図して変える点」の表の 10 行だけである。文面では、設定キーの書き方（相対キーから完全キーへ）、device の 2 文（D8・D9）、設定では直せない誤りの 3 文に足す 1 句（D24）が変わる。文面のほかでは、投げ直しを消すことによる fatal の件数（D8）、呼ぶ側に足す検査（D21）、root 直下の相対キーの読み取り（D18）、テストの期待（D11）が変わる。これら以外の、文の形（`Invalid <key>='<値>'; expected …` など）、止まる条件、止まる時機は変えない。変わる文面を期待する既存のテストは、新しい文面へ直す（§6.6） | 文の形と止まる条件を変えないのは制約ではなく、範囲の線である。変えられるが、本書の目的は設定キーの書き方で、ほかを変えると確かめる範囲が広がる。DQN 系の「挙動を変えない直し」は ERROR の文言を変えない決まりだが（[128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md) の §6.2）、本書は文面を変えることが目的である。クリーンブレークで、旧い文面を残さない。依存: D2 |
| D14 | **PRD155 との順序。** eeb2c3 は本書で直す。[155_config_root_ownership_alignment_10prd.md](155_config_root_ownership_alignment_10prd.md) の §3.4 の、video_codec の文面を直す行は、PRD155 を再グリルするときに外す。PRD155 が先に実装されたら、本書は確かめだけにする | PRD155 は暫定で未実装である。同じ直しを 2 本の PRD が持たないようにする。依存: なし |
| D15 | **英訳。** `*.en.md` は実装で直さず、`anet-translate-docs` で作り直す | 推奨の既定（細部 8）。英訳は日本語版の更新を追って作り直す。依存: D10・D11 |
| D16 | **実装のステージ。** §6.1 の 3 つに分ける | 推奨の既定（細部 10）。題材の 6 件をステージ 2 までで閉じられる。依存: D4 |
| D17 | **受入の確かめ方。** 誤った設定を流すテストを、部品の種類ごとに 1 つ以上置き、加えて §8.2 の集計の出力を、§8.3 から作る §8.7 の期待の一覧と比べて確かめる（§7） | 推奨の既定（細部 11）。202 か所すべてにテストを書くより、種類ごとに経路を確かめる。依存: D2・D4 |
| D18 | **上書きの接頭辞が空の読み口。** 上書きの読み取りをしない。root 直下に相対キーを書いても、どの読み口にも効かず、読まれない設定キーの一覧に出る | ユーザーの選択（推奨は、塞がずに別の課題にすることだった）。P24 のとおり、今は 1 行が 2 つの Agent に同時に効き、検出もされない。P25 のとおり、出荷設定は使っていない。塞ぐと、§6.2 の「上書きの置き場」が、接頭辞が空でないときだけになる。依存: D4・P24・P25 |
| D19 | **一覧で「直す」とした文と呼び出しの見直し。** 実装で、値が設定から来ていないと分かったものは、対象外（§8.1 の R2）へ移し、理由を実装計画に書く。質問しない | 推奨の既定。§8.3 の仕分けには、呼ぶ側まで読み切っていない文がある（§11）。依存: D3 |
| D20 | **集計が拾えない文。** 一覧の文を直すときに、同じ文の中の設定キーは全部直す。一覧の外で見つけた文（§8.2 の集計が拾えない文と、本書の起草のあとにほかの変更が足した文）は、同じ PRD の移行漏れとして直し、実装計画に足す | 推奨の既定。`=` を伴わない名前や、変数に組み立てる文面は、検索で拾えない（§8.2）。クリーンブレークでは、残った旧い書き方は同じ変更の移行漏れとして扱う。依存: D2・D3 |
| D21 | **関数の引数の事前条件とした文。** §8.1 の R3 に当てた 7 つ（§8.3 の 6 文と、直す文の中の名前 1 つ。§8.7 の、ImageData.cpp 619 行の `batch_size`）は、実装で呼ぶ側を読む。設定の誤りから届くなら、呼ぶ側（設定の型を持つ側）に検査を足して完全キーで書く。届かないなら変えない | 推奨の既定。届くかどうかは 7 つそれぞれの呼ぶ側しだいで、起草では読み切っていない。型の検査は、設定の型を持つ側に置く。依存: D3 |
| D22 | **読んだあとにコードで値を書き換える欄。** 記録は、読んだときの完全キーのままにする | 推奨の既定。利用者が直す先はそのキーである。依存: D4 |
| D23 | **固定の root を書く文。** すでに root を持つが、値を与えた置き場と違うかもしれない置き場を固定で書く文（§8.6）も、記録か `ConfigData` を引く形へ直す | 推奨の既定。上書きの置き場から読んだ値でも既定の置き場のキーが出て、G2 を満たさない。4124b6 と同じ種類の誤りである。依存: D1・D4 |
| D24 | **設定では直せない誤りの文面。** 記録が返す文字（D5）は変えない。題材の 3 文（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363・395・399 行）には、記録に無いときに限り、`This agent does not read this setting; remove the metric subscription that requires it.` の 1 句を足す | 共通 Learner への購読要求が設定に無い欄を必要とするときは、その購読を外す（P6）。` (not read from config)` だけでは手立てが伝わらない。RainbowAgent の購読時の挙動は従来どおりとする。依存: D5・P6 |
| D25 | **親の完全キーを示した文の中の、直下の欄。** 欄も完全キーで書く（`eval.[greedy]: scheduled (interval=50, …)` の `interval` など） | 推奨の既定。D2・D3 のとおり、設定キーはどれも完全キーで書く。親の完全キーがあるときだけ欄の名前でよいとする例外は置かない。INFO の行は長くなる。依存: D2・D3 |

### 5.1 選ばなかった形

| 形 | 扱い | 理由 |
|---|---|---|
| 利用側が root までの接頭辞を 1 つ渡し、共通部品が欄の名前を足して完全キーを組む | 不採用（D1） | RainbowAgent が読まない設定でも完全キーを組み、読まれないキーを指す（P6）。この弱点を消すには利用側が「読まない設定」も伝える必要があり、渡す量が D1 と同じになる |
| 共通部品が相対キーだけを書く（今の多くの箇所の形） | 不採用（D1） | block の設定のように、文面のキーを設定に貼っても効かないものがある。読み口の設定では、P24 の読み方のために root 直下に貼ると効いてしまい、2 つの Agent に同時に効く。G2 を満たさない |
| 共通部品が完全キーを固定で書く（今の 4 か所の形） | 不採用（D1） | 利用側が 2 つあると誤ったキーを指す（4124b6 の事象）。P4 に反する |
| 題材の 6 か所だけを直し、ほかは棚卸しの課題にする | 不採用（D2） | 推奨だったが、ユーザーが基盤全体を選んだ |
| 相対キーに、設定の置き場の名前を別に添える形（NN の一部の文の今の形。P11）を認める | 不採用（D2） | 文面のキーを、そのまま設定に貼れない。基盤全体を完全キーにそろえる選択と合わない |
| DQN の共通部品までを直す | 不採用（D2） | 同上 |
| 階層つきの名前だけを完全キーにし、1 語の名前は棚卸しの課題にする | 不採用（D3） | 推奨だったが、ユーザーが 1 語の名前も選んだ |
| 利用側が、名指す設定ごとに完全キーを手で渡す（今の検証の関数の形を全箇所へ広げる） | 不採用（D4） | 上書きの置き場から読んだかを利用側が自分で判定することになる。block がどちらの置き場の値かを示せない |
| 設定だけで決まる検査を、具象の読み口へ移して完全キーで書く | 不採用（D4） | 型の検査を型の共通処理に置く決まり（[AGENTS.md](../../AGENTS.md) の「汎用機構と利用側の責任境界」）に反する。block の検査は移せない |
| block の、書かれていない設定を、同じ設定の `type` の項目の完全キーから補う | 不採用（D5） | `type` は block に渡す設定から除かれていて、引けない（P27）。`ConfigData` が置き場を覚える形なら、forward の実行設定やカタログの項目にも同じ決まりが使える |
| 文脈を足して投げ直す口を作り、6 か所とも同じ口で直す | 不採用（D8） | device の誤りの fatal がキーの無い文面になり、キーが別の行に分かれる。G2 を 1 行では満たさない |
| `ParseDevice` を変えず、ログを出さずに失敗を返す関数を足して、呼ぶ 4 か所が文面を組む | 不採用（D8） | 別モデルの検討で挙がった作りで、グリルでは示していない。fatal が 1 回になる結果は同じで、差は小さい。呼ぶ 4 か所が同じ文面を別々に組むことになる |
| 文脈を足す投げ直し 3 つも、ログを重ねない口を作って直す | 先送り（D8） | 理由: 別の関心（fatal の出し方）。作る条件: [071_uncaught_fatal_visibility_10prd.md](071_uncaught_fatal_visibility_10prd.md) を再グリルするとき |
| device を解決した結果を「実効値」と書く | 不採用（D10） | 同じ段落で、設定の実効値（`auto`）と別の値を同じ語で指す（P18） |
| 「解決後の device」と書く | 不採用（D10） | JSON の欄の名前（`effective`）から引けない。新しい語が増える |
| 「採用値」のままにする | 不採用（D10） | 採用エピソードの値と同じ語のままになる（P20） |
| 出荷定義の tag を、テストと設計文書の名前（39 文字）に変える | 不採用（D11） | 10 月 29 日以後の Run と tag が分かれる。兄弟の指標と違って `skip_` が無くなる |
| 3 つとも 40 文字に収まる新しい名前にそろえる | 不採用（D11） | 過去の Run と tag が分かれる |
| 上書きの接頭辞が空の読み口の読み方を変えず、記録は実際に値を与えたキー（root 直下の相対キーを含む）を返す | 不採用（D18） | 推奨だったが、ユーザーが本書で塞ぐことを選んだ |
| root 直下の相対キーを読むことを、意図した読み方として設計文書に書く | 不採用（D18） | 1 行が 2 つの Agent に同時に効くキーが、仕様として残る |

### 5.2 細部（推奨の既定で埋めた 12 件）

2026-11-21 に一覧で示し、ユーザーが「推奨で埋める」を選んだ。

| 番号 | 事項 | 既定 | 書いた先 |
|---|---|---|---|
| 1 | 利用側が読んでいない設定を引いたときの文字 | 相対キーに ` (not read from config)` を添える | D5・§6.2 |
| 2 | 既定と上書きの両方の置き場があるとき | 値を与えた側の完全キー。どちらにも無ければ既定の置き場の完全キー | D5・§6.2 |
| 3 | NN の block で、どこにも書かれていない設定 | block の `type` を書いた置き場の完全キー（`type` が両方の置き場に書かれたときの決まりは、起草で D5 に足した） | D5・§6.2 |
| 4 | カタログの項目を指す識別子 | 項目の完全キー | D6 |
| 5 | `ParseDevice` に渡す名前 | 必須 | D8 |
| 6 | 手で型を組むとき | 記録は設定の型と一緒に運び、手で組めば空になる | §6.2 |
| 7 | 規約の文面と検査 | D12 の文。機械の検査は足さない | D12 |
| 8 | 英訳 | `anet-translate-docs` で作り直す | D15 |
| 9 | PRD103 の前後比較の道具 | 直さない | D11 |
| 10 | 実装のステージ | 3 つ | D16・§6.1 |
| 11 | 受入の確かめ方 | 種類ごとのテストと、集計・一覧との照合 | D17・§7 |
| 12 | PRD の課題 | 題材で覆えない仕事（仕組みと基盤全体の移行）の課題を 1 件作り、本書に結び付ける | §10 |

### 5.3 起草のあとの未決（推奨の既定で埋めた 8 件）

2026-11-21 に、自己点検のあとで一覧で示し、ユーザーが「推奨で埋める」を選んだ。

| 番号 | 事項 | 既定 | 書いた先 |
|---|---|---|---|
| 1 | 「直す」とした文の値が、実装で設定から来ていないと分かったとき | 対象外へ移し、理由を実装計画に書く | D19 |
| 2 | 集計が拾えない文と、起草のあとに足された文 | 同じ文の中の設定キーは全部直す。一覧の外の文は移行漏れとして直す | D20 |
| 3 | 関数の引数の事前条件とした 6 文（レビューの対応で、直す文の中の名前 1 つを足して 7 つにした。D21） | 呼ぶ側を読み、設定の誤りから届くなら呼ぶ側に検査を足す | D21 |
| 4 | 読んだあとにコードで値を書き換える欄 | 記録は読んだときの完全キーのまま | D22 |
| 5 | 固定の root を書く文 | 範囲に含める | D23・§8.6 |
| 6 | 規約の検査 | 足さない | D12 |
| 7 | 設定では直せない誤りの文面 | 題材の 3 文に 1 句足す | D24 |
| 8 | 親の完全キーを示した文の中の、直下の欄 | 欄も完全キーで書く | D25 |

6 について、別モデルは、機械で判定できる部分（階層つきの名前と、相対キーを渡す呼び出し）を、差し戻しの根拠にしない検査として足す案を挙げた。検証資材のスクリプト（§6.6）は残るので、要るときに流せる。

## 6. 実装契約

### 6.1 ステージ

| ステージ | 直すもの | 閉じる課題 |
|---|---|---|
| 1 | 仕組みを使わずに直せる題材。device の 2 文と `ParseDevice`（D8・D9）、設計文書とテストのコメントの語（D10）、テストと設計文書の指標名（D11） | 78df9c・9b314a・d4258c・546bdb |
| 2 | 完全キーを運ぶ仕組み（D4・D5）、上書きの接頭辞が空の読み口（D18）、`ConfigData` 自身のエラー 3 文（§8.3 の config.cpp の 2 文と、§8.5 の読まれないキーの 1 文）、規約（D12）、題材の残りの 5 文（MetricsLogger の 1 文、DQN の共通部品の 4 文）。設計文書 100 への追記（§6.6） | eeb2c3・4124b6 |
| 3 | §8.3 と §8.5 の残り。NN の投げ直し 2 つを消す（D8） | §10 の、題材で覆えない仕事の課題 |

ステージの切れ目では、既定のテストが通る状態にする。ステージ 3 は、§8.3 のファイルの単位で分けて進めてよい。NN の投げ直しを消すのは、ステージ 2 の `ConfigData` 自身のエラーを直したあとである（D8）。

### 6.2 完全キーを運ぶ仕組みの契約

**`ConfigData` が覚えるもの。**

- 項目ごとに、値を与えた完全キーを覚える。root の `ConfigData`（設定ファイルと CLI から作ったもの、手で組んだもの）では、項目のキーそのものである。
- 切り出し（`MakeSubConfigData`）、別の `ConfigData` からの写し（`CopyValueFrom`）、読み取り範囲の作成（`CreateReadScope`・`Config::MakeReadScope`）を経た項目は、元の項目の完全キーを保つ。
- 上書きの合成（`OverwriteFrom`）で値が置き換わった項目は、置き換えた側の完全キーになる。同じ値どうしを畳む合成（`MergeFromChecked`）で重なった項目は、あとから合成した側の完全キーになる。
- 既存の項目へコードが値を書き込んでも（`Set`）、完全キーは変わらない。設定のプロファイルの補間が書き込む項目は、補間を定義した項目の完全キーを保つ（P28）。
- 切り出した `ConfigData` は、自分の置き場を覚える。切り出しと、接頭辞を渡した読み取り範囲の作成が置き場を決める。NN の block の設定のように 2 つの置き場を合成したものは、block の `type` の値を与えた側の置き場を覚える（D5）。`type` が共有の置き場と Agent の下の両方に書かれていれば、値を与えるのは Agent の下である（上書きの合成の項目の決まりと同じ向き）。`type` がどちらにも書かれていない block の設定は、Agent の下に項目が 1 つでもあれば Agent の下の置き場を、無ければ共有の置き場を覚える。この block は、block を作る前に止まる（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 1678〜1680 行）。
- `ConfigData` は、相対キーを渡すと完全キーを返す口を持つ。結果は下の「項目の表」による。完全キーを引くことは、項目を読んだことにしない。
- `ConfigData` 自身が出すエラー（型の読み取りの失敗、読まれないキー、合成の食い違い）も、この口で完全キーを書く。読まれないキーのエラーの持ち主の名前は、呼ぶ側が固定の文字で渡さず、その `ConfigData` の置き場で書く。

**読み口の記録。**

- `anet::Config` は、欄を読むたびに、欄の相対キーと、値を与えた完全キーの対応を記録する。dump に記録するかどうかの指定に依らず、読んだ欄はすべて記録する（P29）。記録を持つのは、読んだ欄だけである。
- 上書きの接頭辞が空の読み口は、上書きの読み取りをしない（D18）。値があるかを確かめる口と、読み取り範囲を作る口も同じにする。
- 読み口が、読んだ値を別の欄へコードで写すとき（`quantile_mode` を Actor の方策の欄へ写すなど）は、記録の対応も写す（P29）。
- 読み口が、読んだ欄の値をあとからコードで書き換えても、記録は読んだときの完全キーのままにする（D22）。
- 読み口は、記録の全部か、ある接頭辞の下だけを取り出して、相対キーを付け替えた記録を作れる（たとえば `learner.` の下だけを取り出し、`per_eps` で引けるようにする）。
- 部品へ渡す設定の型は、記録を欄として持つ。読み口が詰める。手で組んだ型は、空の記録を持つ。
- 部品が別の部品の設定の型を組み立てるときは、値を写した欄ごとに、記録の対応も写す（たとえば Learner が Replay の設定の `capacity` に `replay_capacity` の値を写すなら、`capacity` で引くと `replay_capacity` の完全キーが返るようにする）。
- 素の値や名前を引数で受ける関数（`ParseDevice`、ProfiledValue の検証、drop の率の検証、状態の sweep の検査など）は、呼ぶ側が完全キーの文字を引数で渡す。

**引いた結果。** どちらの表も、上から順に当て、最初に条件を満たした行で扱う。

記録の表: 読み口の記録を、相対キーで引く。

| 順 | 条件 | 得る文字 |
|---|---|---|
| K1 | 相対キーが空である | 止める（`ANET_SYSTEM_ERROR`。呼ぶ側の誤り） |
| K2 | 記録に無い（読み口が読んでいない欄、手で組んだ型） | 相対キーに ` (not read from config)` を添えた文字 |
| K3 | 読み口の上書きの接頭辞が空でなく、読んだ `ConfigData` に上書きのキーの項目がある | 上書きのキーを、読んだ `ConfigData` で項目の表に通した結果 |
| K4 | 上のどれでもない（既定のキーに項目がある場合と、どちらにも無く型の既定値のままの場合） | 既定のキーを、読んだ `ConfigData` で項目の表に通した結果 |

項目の表: `ConfigData` を、その `ConfigData` の中のキーで引く。

| 順 | 条件 | 得る文字 |
|---|---|---|
| C1 | キーが空である | 止める（`ANET_SYSTEM_ERROR`。呼ぶ側の誤り） |
| C2 | 項目があり、完全キーを持つ | その完全キー |
| C3 | `ConfigData` が置き場を覚えている（項目が無い場合と、切り出した設定にコードが新しく作った項目の場合） | 置き場に、渡されたキーを続けた文字 |
| C4 | 上のどれでもない（置き場を持たない `ConfigData` の、無い項目。root の `ConfigData` と、手で組んだ `ConfigData` が当たる） | 渡されたキーそのもの |

記録の表の「既定のキー」「上書きのキー」は、読み口の接頭辞に相対キーを続けたものである。既定の接頭辞が空の読み口（block の設定を読む読み口）では、相対キーそのものになり、項目の表の C2 か C3 で完全キーになる。

**設定の例と文面。**

```properties
DefaultDQNAgent.learner : per_eps = -1
net.block.[CN1] : cn.channels = 0
net.block.[CN2] : cn.channels = abc
run.eval.[greedy].env : game = breakout
learner.per_alpha = 2
```

| 場面 | 今 | 直したあと |
|---|---|---|
| Learner の PER の検査（DefaultDQNAgent） | `Invalid learner.per_eps=-1. Expected a finite value >= 0.` | `Invalid DefaultDQNAgent.learner.per_eps=-1. Expected a finite value >= 0.` |
| 同じ検査（RainbowAgent） | 同じ文面 | `Invalid RainbowAgent.learner.per_eps=-1. …` |
| 可塑性の診断を購読して境界の branch が空（RainbowAgentConfig で組み立てた共通 Learner） | `DefaultDQNAgent.learner.final_feature_branch_key='' is required when feature-boundary diagnostics are subscribed; …` | `final_feature_branch_key (not read from config)='' is required when feature-boundary diagnostics are subscribed; … This agent does not read this setting; remove the metric subscription that requires it.`（末尾の 1 句は D24） |
| CNBlock の検査（共有の置き場に書いた値） | `CNBlock: 'cn.channels' must be strictly positive.` | `CNBlock: 'net.block.[CN1].cn.channels' must be strictly positive.` |
| block の設定の型の誤り | `ConfigData::Read failed. key=cn.channels value="abc" expected=…` | `ConfigData::Read failed. key=net.block.[CN2].cn.channels value="abc" expected=…` |
| 128x128 の動画で `h264_nvenc` を明示 | `metrics_logger.video_codec=h264_nvenc is not supported for video size 128x128. … Use metrics_logger.video_codec=auto or libx264. …` | `app.metrics_logger.video_codec=h264_nvenc is not supported for video size 128x128. … Use app.metrics_logger.video_codec=auto or libx264. …` |
| 評価用 Env の、上書きの置き場から読んだ値 | `AtariEnv.mode=… is unavailable for game='breakout'. …` | `… is unavailable for run.eval.[greedy].env.game='breakout'. …`（同じ文の `AtariEnv.mode` も、D23 で、値を与えた置き場の完全キーになる） |
| データセットの項目 | `… dataset_key='food101' path='…'` | `… ImageDataset.[food101] path='…'` |
| root 直下の相対キー（最後の行） | DefaultDQNAgent と RainbowAgent の両方が `per_alpha` に 2 を読む。通知は出ない | どの読み口も読まない。Run の構築の終わりの、読まれない設定キーの一覧に `learner.per_alpha` が出る |

### 6.3 device の 2 文（D8・D9）

| 場面 | 今 | 直したあと |
|---|---|---|
| `agent.device = cuda:x` | fatal が 2 件。1 件目は `Invalid device specification 'cuda:x'. Expected auto, cpu, cuda, or cuda:<non-negative integer>.`、2 件目は `Invalid agent.device='cuda:x'. ` に 1 件目の全文を続けたもの | fatal が 1 件。`Invalid agent.device='cuda:x'. Expected auto, cpu, cuda, or cuda:<non-negative integer>.` |
| CUDA が使えない機械で `agent.device = cuda:1`、`run.eval_device = cuda` | `CUDA is unavailable for explicit device settings: agent.device, run.eval_device. Change these keys to auto to select CPU when CUDA is unavailable.` | `CUDA is unavailable for explicit device settings: agent.device='cuda:1', run.eval_device='cuda'. Change these keys to auto to select CPU when CUDA is unavailable.` |

`ParseDevice` に渡すキーの名前は、空でない文字とする。空なら `ANET_SYSTEM_ERROR` で止める（呼ぶ側の誤り）。呼ぶ 4 か所は、`InitRL` が検査中のキーの名前、Agent が `agent.device`、Env が `env.device`、評価が `run.eval_device` を渡す。受け付ける device の文法と `auto` の選び方、`InitRL` が見るキーの集まり、明示の cuda を一括で報告することは、PRD070 の D5・D10 と [122_small_refactor_and_test_gap_fixes_10prd.md](122_small_refactor_and_test_gap_fixes_10prd.md) の D9 が決めたもので、本書は文面だけを直す。

### 6.4 変えないものと、意図して変える点

変えないもの。

- 設定キー、メトリクスの tag、checkpoint の形式。
- 完全キーで書かれた設定の、解決の結果（`config/config_data.txt` と Module ごとの dump）。出荷設定は、すべてこれに当たる（P25）。
- エラー・WARN・INFO の出る条件、時機、回数。ただし、次の 4 つは変わる。device の文法の誤りの fatal は 2 件から 1 件になる（D8）。NN の投げ直し 2 つを消すので、その 2 つを通っていた誤りの fatal も 2 件から 1 件になる（D8）。関数の引数の事前条件とした 7 つ（§8.1 の R3。6 文と、ImageData.cpp 619 行の `batch_size`）のうち、設定の誤りから届くものは、呼ぶ側に検査を足すので、止まる場所が呼ぶ側へ早まり、エラーの文が増える（D21）。root 直下に相対キーを書いた設定では、読まれない設定キーの一覧の WARN にそのキーが新しく出る（D18）。
- 文の形と、設定キーでないラベル（§8.1 の R1〜R3 の名前）。ただし、設定では直せない誤りの 3 文には、記録に無いときに限り 1 句を足す（D24）。NN の投げ直し 2 つを消すと、その 2 つが文面の前に足していた前置き（`Invalid projection config: owner=…; ` と `Invalid Linear execution setting '…': `）は無くなる（D8）。
- 完全キーで書かれた設定についての、読まれない設定キーの検出の結果（PRD117）。完全キーを引くことは、項目を読んだことにしない。これは本当の制約である。引くことを読んだことにすると、通知を出した部品が読まなかった設定キーが、読まれない設定キーの一覧から消える（[ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)）。
- 学習と評価の値。性能の受入は置かない。完全キーの記録は設定を読むときだけ行い、記録を引くのは通知を出すときだけで、step のループには触れない。

意図して変える点。

| 番号 | 対象 | 今 | 直したあと |
|---|---|---|---|
| 1 | §8.3 の「直す文」178 文と「直す呼び出し」16 行、§8.5 の 8 文、§8.6 の文の、設定キーの書き方 | 相対キー、または固定の root | 完全キー（§6.2 の表の結果） |
| 2 | device の 2 文 | §6.3 の「今」 | §6.3 の「直したあと」 |
| 3 | `ParseDevice` の口 | 指定の文字だけを受け取る | 指定の文字と、キーの名前を受け取る |
| 4 | `[prd103-config]` の期待 | `19_primary_clip_ratio_ema` | `19_skip_primary_clip_ratio_ema` |
| 5 | 設計文書の語と指標名、テストのコメントの語 | §1.2 の d4258c・546bdb の行 | D10・D11 |
| 6 | 変わる文面を期待する既存のテスト | 旧い文面 | 新しい文面 |
| 7 | 上書きの接頭辞が空の読み口の読み取り（D18） | root 直下の相対キーを読む。読んだ扱いになる | 読まない。root 直下の相対キーは、読まれない設定キーの一覧に出る |
| 8 | 設定では直せない誤りの 3 文（D24。[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363・395・399 行） | 設定キーと条件だけを書く | 記録に無いときに限り、末尾に `This agent does not read this setting; remove the metric subscription that requires it.` を足す |
| 9 | 関数の引数の事前条件とした 7 つ（D21。§8.1 の R3。§8.3 の 6 文と、§8.7 の ImageData.cpp 619 行の `batch_size`）のうち、設定の誤りから届くもの | 引数を受けた関数が、相対の名前で止める | 呼ぶ側（設定の型を持つ側）が、完全キーを書いたエラーで先に止める。届かない文は変えない |
| 10 | NN の投げ直し 2 つ（D8。[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) 204・293 行） | 内側のエラーを捕まえ、前置きを足して投げ直す。fatal が 2 件出る | 投げ直しを消す。fatal は内側の 1 件になり、前置きは無くなる。置き場は、内側のエラーの完全キーが示す |

### 6.5 クラス一覧

従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `ConfigFullKeyTable` | 新規 | 欄の相対キーから、値を与えた完全キーを引く。接頭辞の下だけを取り出した記録と、対応を写した記録を作る | 設定（`config.*`） |
| `ConfigData` | 変更 | 項目ごとの完全キーと、切り出した設定の置き場を保ち、相対キーから引けるようにする。自分が出すエラーを完全キーで書く | 設定（`config.*`） |
| `Config` | 変更 | 欄を読むたびに `ConfigFullKeyTable` へ記録し、記録を返す。上書きの接頭辞が空なら、上書きの読み取りをしない | 設定（`config.*`） |
| `ProfiledValue` | 変更 | 読んだ完全キーを保ち、評価のときのエラーに使う | schedule（`schedule.*`） |
| `MetricsLoggerConfig` | 変更 | 記録を欄として運ぶ | メトリクス（`metrics_logger.*`） |
| `NetworkModelConfig`・`ActionPolicyConfig`・`DQNActorConfig`・`SkipConfig`・`LearnerConfig` | 変更 | 記録を欄として運ぶ | DQN の設定（`dqn_config.*`） |
| `ReplayBufferConfig` | 変更 | 記録を欄として運ぶ | Replay（`replay_buffer.*`） |
| `RewardScalerConfig`・`SweepedHeatMapObserverConfig` | 変更 | 記録を欄として運ぶ | scaler（`scaler.*`）、Observer（`observers.*`） |
| RND の設定の型 | 変更 | 記録を欄として運ぶ | 内的報酬（`intrinsic_rnd.*`） |

消すクラスは無い。

### 6.6 移行対象

**コード。** §8.3 の「直す文」か「直す呼び出し」を持つ 30 ファイルと、§8.5 の [init.cpp](../../src/core/base/src/init.cpp)（§8.5 のほかのファイルは §8.3 にもある）。§8.6 の、固定の root を書く文を持つファイル（具象の Env の 6 つと [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp)。うち [LunarLanderEnv.cpp](../../src/env/lunarlander/src/LunarLanderEnv.cpp) と [CartPoleEnv.cpp](../../src/env/cartpole/src/CartPoleEnv.cpp) は §8.3 に無い）。仕組みを足す [config.hpp](../../src/core/base/include/anet/config.hpp)・[config.cpp](../../src/core/base/src/config.cpp)。`ParseDevice` の宣言と定義（[tensor_util.hpp](../../src/core/base/include/anet/tensor_util.hpp)・[tensor_util.cpp](../../src/core/base/src/tensor_util.cpp)）と、呼ぶ 4 か所（§1.2）。記録を詰める読み口（[RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp)、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) と、§6.5 の型を詰めるほかの読み口）。

**テスト。** 変わる文面を期待しているテストを、新しい文面へ直す。2026-11-21 に、`src` の `*_test.cpp`・`*_test.hpp` を、表の下の検索語 T1 で引いて 9 ファイル 44 行だった。

| ファイル | 行の数 |
|---|---|
| [dqn_actor_test.cpp](../../src/algo/dqn/src/dqn_actor_test.cpp) | 4 |
| [dqn_concrete_test.cpp](../../src/algo/dqn/src/dqn_concrete_test.cpp) | 11 |
| [dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) | 1 |
| [dqn_intrinsic_test.cpp](../../src/algo/dqn/src/dqn_intrinsic_test.cpp) | 6 |
| [dqn_learner_test.cpp](../../src/algo/dqn/src/dqn_learner_test.cpp) | 7 |
| [dqn_test.cpp](../../src/algo/dqn/src/dqn_test.cpp) | 3 |
| [metrics_logger_test.cpp](../../src/core/base/src/metrics_logger_test.cpp) | 1 |
| [nn_test.cpp](../../src/core/base/src/nn_test.cpp) | 9 |
| [DropMergeEnv_test.cpp](../../src/env/dropmerge/src/DropMergeEnv_test.cpp) | 2 |

T1 は、文面を比べる呼び出しの行に、相対キーの先頭の語がある行を拾う。語を並べた検索なので、文面を変数に入れてから比べるテストと、ここに無い語で比べるテストは拾えない。拾えなかったテストは、直した文を通るテストが落ちることで分かる。落ちたテストは、同じステージで新しい文面へ直す。`ParseDevice` を直接呼ぶテスト（[init_test.cpp](../../src/core/base/src/init_test.cpp) 206〜214 行）は、キーの名前を渡す形へ直す。[dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) 108 行の指標名を直す（D11）。root 直下の相対キーを置いて読み口に読ませているテストがあれば、完全キーへ直す（D18）。該当は検索では拾えず、ステージ 2 でテストが落ちることで分かる。

```text
T1（正規表現。ripgrep で src の *_test.cpp と *_test.hpp を引く）:
(ContainsSubstring|ContainsText|CHECK_THROWS_WITH|REQUIRE_THROWS_WITH|Message|Matches)\(.*"[^"]*(learner\.|skip\.|intrinsic\.|target_policy\.|actor\.\[|noisy\.|weight_norm\.|cn\.|pool\.|res\.|cos\.|tf\.|init2\.|metrics_logger\.|dataset_key|class_id=|per_initial_priority|quantile_mode=|video_codec|device specification|explicit device settings|final_feature_branch_key|replay_capacity=|no_legal_min_blocked_frames|post_process_type|statistics_mode|multiplier_mode)
```

**設計文書と規約。**

- [100](../design/100_runtime_and_configuration.jp.md): §2.1「設定の解決」に、完全キーの記録と引いた結果（§6.2）、上書きの接頭辞が空の読み口が上書きを読まないこと（D18）を足す。§7.3「エラー」に、通知が示す設定キーの書き方を足す。338・346 行の語を直す（D10）。新しい設計文書は作らない。仕組みは設定の読み口の一部で、100 の §2.1 に書く場所がある。
- [020](../design/020_user_guide_run.jp.md) 67・623 行、[110](../design/110_agents_and_learning.jp.md) 191 行、[120](../design/120_environments.jp.md) 218 行: 語を直す（D10）。テストのコメント（[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) 659 行）の語も直す。
- [140](../design/140_observability.jp.md) 431 行: 指標名を直す（D11）。
- [130](../design/130_neural_networks.jp.md): block の設定の誤りの文面が完全キーを示すことを、設定の節に 1 文足す。
- [impl.md](../quality/impl.md) の「失敗の扱い」: 規約の文と項目 1 つを足す（D12）。項番は、足す時点の最大の次を取る。
- 英訳（`100`・`020`・`110`・`120`・`130`・`140` の `*.en.md`）は実装で直さない（D15）。
- 用語集（[CONTEXT.md](../../CONTEXT.md)）と ADR 0087 は、本書の起草で書いた。

**検証資材。** §8.2 の集計のスクリプト、§8.7 の期待の一覧、集計の出力と期待の一覧を比べる手順を `src/core/base/testdata/prd194/` に置き、README を書く（[AGENTS.md](../../AGENTS.md) の「検証」の決まり）。

**直さないもの。** 過去の記録（実験記録、ADR、保管場所の PRD と実装記録、課題記録の事象）に書かれた旧い文面と、`19_primary_clip_ratio_ema` の名前は、当時の記録として残す。PRD103 の前後比較の道具は直さない（D11）。

**互換のための仕組み。** 残さない。`ParseDevice` の、キーの名前を取らない呼び出しは残さない。旧い文面を出す分岐は残さない。root 直下の相対キーを読む経路は残さず、それを知らせる専用の検出も足さない。読まれない設定キーの一覧が、ほかの読まれない設定キーと同じに知らせる。

### 6.7 ほかの PRD との順序

- [155_config_root_ownership_alignment_10prd.md](155_config_root_ownership_alignment_10prd.md)（暫定・未実装）: D14 による。本書が先なら、PRD155 が root を改名しても、文面は読み口の記録から完全キーを得るので追随する。PRD155 が先なら、本書の例に出る root が新しい名前になるだけである。
- [161_observation_pipeline_10prd.md](161_observation_pipeline_10prd.md) など、設定の読み口と §8.3 のファイルを触る未実装の PRD: どちらが先でもよい。あとから入る側は、新しく足す文を D12 の規約で書く。本書のステージ 3 の途中でほかの PRD が入ったときは、その PRD が足した文を、一覧の外の文として同じステージで直す（D20）。ステージ 2 の途中では入れない。仕組みの口が決まる前に、記録を運ぶ型へ別の欄が足されると、取り直しが要る。
- [071_uncaught_fatal_visibility_10prd.md](071_uncaught_fatal_visibility_10prd.md)（未実装）: 文脈を足す投げ直し 3 つを、その再グリルで扱う（§5.1）。
- [193_variance_scaling_weight_init_10prd.md](193_variance_scaling_weight_init_10prd.md)（未実装）: どちらが先でもよい。扱いは同書の §11 にもある。本書が先なら、PRD193 は新しく足す文を D12 の規約で書き、block の検査の設定キーを読み口の記録から引く。PRD193 が先なら、ステージ 3 で次の 2 つを行う。(1) PRD193 が足す初期化の設定の読み取りの口は、block の検査の文の設定キーを、実行時に接頭辞から組み立てて相対キーで出す（`init1.variance_scale` など）。§8.2 の集計は、先頭が `{}` の名前を root を持つとして除くので、この文を拾わない。一覧の外の文として、完全キーに直す。(2) PRD193 は、初期化の設定を 1 キーずつ読む `ANET_READ_CONFIG` の行を口に替えるので、§8.2 の「ANET_READ_CONFIG が読む欄」が変わる。集計を取り直す。ステージ 2 の途中には、PRD193 を入れない（上の PRD161 の行と同じ理由）。

## 7. 受入

テストのタグは `[prd194]` とする。A2 のテストは、CUDA が使える機械では環境変数 `CUDA_VISIBLE_DEVICES` を `-1` にして流し、skip を合格に数えない（[122_small_refactor_and_test_gap_fixes_10prd.md](122_small_refactor_and_test_gap_fixes_10prd.md) の §6 A6）。

| 番号 | 対象 | 確かめること |
|---|---|---|
| A1 | 78df9c | `agent.device` に `cuda:x` を置いた設定で `InitRL` を 1 回呼ぶと止まり、その呼び出しの間に出た fatal のログが 1 件で、文面に `agent.device='cuda:x'` を含むテストがある。`env.device`・`run.eval_device` でも同じである |
| A2 | 9b314a | CUDA が使えない状態で、`agent.device` に `cuda:1`、`run.eval_device` に `cuda` を置いた設定で `InitRL` が止まり、文面に `agent.device='cuda:1'` と `run.eval_device='cuda'` の両方を含むテストがある |
| A3 | eeb2c3 | RunnerApp の設定の読み口から作った設定で、128x128 の動画に `h264_nvenc` を明示すると止まり、文面に `app.metrics_logger.video_codec=h264_nvenc` と `Use app.metrics_logger.video_codec=auto` を含むテストがある。手で組んだ `MetricsLoggerConfig` では、文面に `video_codec (not read from config)` を含む |
| A4 | 4124b6 | 可塑性の診断を購読して境界の branch が空のとき、DefaultDQNAgent の文面は `DefaultDQNAgent.learner.final_feature_branch_key` を含み、`remove the metric subscription` を含まない。RainbowAgentConfig で組み立てた共通 Learner に同じ購読を直接渡したときは `final_feature_branch_key (not read from config)` と `remove the metric subscription` を含み、`DefaultDQNAgent` を含まない（D24）。RainbowAgent 自身に同じ購読を渡した場合は従来どおり何もしない。表の下の検索 S1 が、登録の 2 行だけに当たる |
| A5 | d4258c | `docs/design/*.jp.md` を固定文字列「採用値」で引くと 1 行（100 の、評価セッションの行）だけに当たる。「採用したdevice」で引くと 0 行である。表の下の検索 S2 で引くと 6 行以上に当たる。`src` を「採用値」で引くと 1 行（[app_util.cpp](../../src/core/base/src/app_util.cpp) の、workspace の履歴の値を言うコメント）だけに当たる。[CONTEXT.md](../../CONTEXT.md) に「実効 device」の項目がある。設計文書と `src` のほかの「採用値」（§1.2 の d4258c の行の 17 行のうち、課題記録・wiki・過去の PRD と実装記録・サーベイ・用語集・設定のコメント）は、device の意味でないか当時の記録なので、範囲から除く |
| A6 | 546bdb | `[prd103-config]` を単独で流すと、4 環境の断定がすべて通り、終了コードが 0 である。`src` と `docs/design/*.jp.md` を固定文字列 `19_primary_clip_ratio_ema` で引くと 0 行である。英訳は作り直すまで旧い名前が残るので、範囲から除く（D15）。過去の記録（課題記録、実装記録、実験記録）は当時の記録なので、範囲から除く |
| A7 | 仕組み（`ConfigData`） | 切り出し、写し、上書きの合成、同じ値を畳む合成、読み取り範囲の作成を経た項目から、§6.2 の項目の表のとおりに完全キーを引けるテストがある。合成で置き換わった項目は、置き換えた側の完全キーを返す。既存の項目へ値を書き込んだあとも、同じ完全キーを返す。切り出した設定の、無い項目は、置き場に相対キーを続けた文字を返す（C3）。root の `ConfigData` の無い項目は、キーそのものを返す（C4）。空のキーは止まる（C1）。完全キーを引いたあとも、その項目が読まれない設定キーの一覧に残る |
| A8 | 種類ごとの経路 | 誤った設定を流して文面の完全キーを確かめるテストが、次の 8 種類に 1 つ以上ある。(1) 具象 Agent の読み口の文、(2) DQN の共通部品の文を DefaultDQNAgent から、(3) 同じ文を RainbowAgent から、(4) NN の block の文を共有の置き場に書いた値で、(5) 同じ文を Agent の下の置き場に書いた値で、(6) 上書きの置き場から読んだ Env の値（相対キーで書いていた名前と、既定の置き場を固定で書いていた名前（§8.6）の両方が、上書きの置き場の完全キーになる）、(7) MetricsLogger（A3 と兼ねてよい）、(8) device（A1 と兼ねてよい）。(4) と (5) は、block の設定に数値でない値を書いたときの型の読み取りの失敗の文と、綴りを誤ったキーを書いたときの読まれないキーの文も、1 つずつ確かめる。block の、どこにも書かれていない設定の文（項目の表の C3）は、次の 3 つの block で確かめる。`type` を共有の置き場だけに書いた block は、共有の置き場に相対キーを続けた文字になる。`type` を共有の置き場に書き、ほかの欄だけを Agent の下に書いた block も、共有の置き場になる。`type` を Agent の下にも書いた block は、Agent の下の置き場になる |
| A9 | 階層つきの名前 | 実装を終えた `src` に §8.2 の集計の 1 段目を流すと、階層つきで root の無い名前が、§8.7 の期待の一覧にある名前だけである。起草時の期待は、§8.4 の除外表の名前である |
| A10 | 集計の出力の全体 | 実装を終えた `src` に §8.2 の集計を流した出力（1 段目の文と 2 段目の行）が、§8.7 の期待の一覧と一致する。比べ方と、期待の一覧を変えたときの扱いは §8.7 による。起草時の期待は、1 段目が 152 文（§8.3 の対象外の文 125 と、変えない名前で拾われ続ける直す文 27）、2 段目が 48 行（§8.3 の対象外の行）である |
| A11 | 変えないもの | テストの実行体 9 つ（`anet-core-base-test`、`anet-algo-dqn-test`・`anet-algo-imagecls-test`・`anet-algo-muzero-test`、`anet-env-atari-test`・`anet-env-dropmerge-test`・`anet-env-gridmaze-test`・`anet-env-imagecls-test`・`anet-env-lunarlander-test`）を引数なしで流して、どれも終了コードが 0 である。`anet-core-base-test` を `[shipped_config]` と `[prd072]` で流して通る。hidden の `[prd072-baseline]`（固定した入力の設定の解決の結果を、保存した基準と比べる）を流して通り、設定の解決の結果が変わっていない |
| A12 | 規約と文書 | [impl.md](../quality/impl.md) の「失敗の扱い」に D12 の文と項目がある。[100](../design/100_runtime_and_configuration.jp.md) の §2.1 に、完全キーの記録と引いた結果、上書きの接頭辞が空の読み口の読み方がある |
| A13 | NN の投げ直し（D8） | Linear の forward の実行設定に不正な値を置くと止まり、その構築の間に出た fatal のログが 1 件で、文面が完全キーを含むテストがある |
| A14 | 上書きの接頭辞が空の読み口（D18） | root 直下に `learner.per_alpha` を置き、ほかは既定の設定で DefaultDQNAgent と RainbowAgent の設定の型を作ると、どちらの `per_alpha` も既定値のままで、読まれない設定キーの一覧に `learner.per_alpha` が出るテストがある。上書きの接頭辞を渡した読み口（評価用 Env）は、上書きの置き場の値を今までどおり読む |

```text
S1（固定文字列を 2 つ。git grep で src/algo/dqn を引き、default_dqn_agent.* と rainbow_agent.* と *_test.cpp と *_test.hpp を除く）:
DefaultDQNAgent
RainbowAgent
当たってよいのは、src/algo/dqn/src/dqn.cpp の登録の 2 行だけ。

S2（正規表現。docs/design の *.jp.md を引く。設計文書は和文と英字のあいだに空白を入れない書き方なので、空白の有無の両方に当てる）:
実効 ?device
```

後の作業の中で確かめる受入は無い。Run を流す受入は無い。

## 8. 直す文の一覧

### 8.1 判定の規則と凡例

集計（§8.2）の 1 段目が拾った文を、次の規則で仕分ける。上から順に当て、最初に条件を満たした行で扱う。1 つの文に名前が複数あるときは、名前ごとに当て、1 つでも R4 か R5 に当たれば「直す文」とする。直す文の中の、R1〜R3・R6 に当たる名前は変えない。

| 規則 | 条件 | 扱い |
|---|---|---|
| R1 | 名前が、後ろの値として完全キーそのものを出すラベルである（`key=` の後ろに、root の `ConfigData` のキーが出る文） | 対象外。すでに完全キーを示している |
| R2 | 名前が示す値が、設定から来ていない（scalar の鍵、TensorDict の鍵、登録するときの class ID、件数、形、device、標本や lane の位置、GUI で選んだ値、導いた seed など）。ただし、checkpoint に保存された値を、同じ設定キーの今の値と並べて示す文の、checkpoint の側の名前は除く（R5 で扱う） | 対象外 |
| R3 | 名前が、起草で「呼ぶ側のコードの誤りを知らせる、引数の事前条件」と見立てた 7 つのどれかである。§8.3 に R3 と記した 6 文の名前と、§8.7 の表の ImageData.cpp 619 行の `batch_size` である | 対象外。設定の誤りから届くかは、D21 で確かめる |
| R4 | 名前が、カタログの項目を指す識別子である。項目が定義されていないことを言う文も含む | 直す。項目の完全キーで書く（D6） |
| R5 | 名前が設定の値を示していて、値がその設定から来ている。名前が設定キー（またはその末尾）と同じ場合と、設定キーを縮めたラベルの場合（`image_width` の値に付けた `width`、`adam_eps` の値に付けた `eps`、`weight_norm.mode` の値に付けた `mode` など）の両方を含む。R2 が除いた、checkpoint の側の名前も当たる。`key=` や `field=` の後ろに、呼ぶ側が渡した相対キーや、切り出した `ConfigData` のキーが値として出る文も含む（直したあとは、後ろの値が完全キーになり、ラベルは R1 に当たる） | 直す。完全キーで書く（D5） |
| R6 | 上のどれでもない（文の区切りを、集計が名前と拾ったものなど） | 対象外。設定キーの名前ではない |

R5 に当てた文のうち、実装で、値が設定から来ていないと分かったものは、R2 へ移す（D19）。R3 に当てた 7 つは、実装で呼ぶ側を読んで、設定の誤りから届くかを確かめる（D21）。

2 段目が拾った行（相対キーを文字の引数で渡している呼び出し）は、渡した文字が文面に設定キーとして出るなら「直す呼び出し」、出ないなら「対象外の行」とする。対象外の行は、乱数の種の名前、TensorDict の欄の名前、scalar の鍵、自分の読み口の中で完全キーに組み直してから使う名前である。

完全キーの得かた（一覧の「/」の後ろの字）。

| 字 | 得かた |
|---|---|
| a | 文が設定の読み口かその持ち主の中にあり、自分の読み口の記録を引く |
| b | 部品が、設定の型と一緒に運ばれた記録を引く |
| c | `ConfigData` を引く（NN の block と forward の実行設定、`ConfigData` 自身のエラー） |
| d | カタログの項目の完全キーを書く |
| e | 呼ぶ側が完全キーの文字を引数で渡す |

### 8.2 集計の条件

読み取りだけを行うスクリプトで、次のとおりに拾う。検証資材として `src/core/base/testdata/prd194/` に置く（§6.6）。

- 対象のファイル: `git ls-files src` のうち、拡張子が `.cpp`・`.hpp` で、パスに `_test.`・`/testdata/`・`/src_testing/`・`/include/anet/test/` を含まないもの。
- 1 段目の対象の文: 行頭がコメントでなく、行に「エラーの始まり」のどれかを含むか、「ログの始まり」の正規表現に当たる文。その行から、行末が `;` になる行まで（最大 15 行）を 1 つの文とする。「ログの始まり」の名前が `info` なら INFO、`warn` なら WARN、`error`・`fatal` ならエラーとして数える。
- 文面: 文の中の文字列を、順に `{}` でつないだもの。`GetFullKey("…")` の文字列引数は検索に使う相対キーなので、文面に含めない。
- 階層つきの名前: 文面に「階層つきの名前」の正規表現を当てたもの。末尾の語が拡張子（`txt`・`cpp`・`hpp`・`json`・`jsonl`・`db`・`gz`・`mkv`・`png`・`py`・`bat`・`exe`・`log`・`md`・`g`・`e`）のものと、`std.`・`torch.`・`e.g`・`i.e` で始まるものは除く。先頭の語が root（`app`・`run`・`env`・`agent`・`net`・`metrics`・`backend`・`image`）か、大文字で始まるか、`{}` であるものは、root を持つとして除く。
- 1 語の名前: 先頭の「対象のファイル」に該当するすべてのファイルから、`ANET_READ_CONFIG` が読む欄の末尾の語を集める。文面に「1 語の名前」の正規表現を当てた結果のうち、この共通の集合に含まれるものを拾う。
- 2 段目の対象の行: 1 段目の文に入らず、行頭がコメントでなく、「相対キーの引数」の正規表現に当たり、「読み取りの口」のどれも含まない行。拾った名前の先頭の語が root のもの、末尾の語が上の拡張子のものは除く。

```text
エラーの始まり（固定文字列）:
ANET_SYSTEM_ERROR(   ANET_CHECK_MSG(   ANET_ASSERT_MSG(

ログの始まり（正規表現。1 つ目の括弧の中が名前）:
\b(info|warn|error|fatal)(_once)?\((?:[^()]|\([^()]*\))*\)\s*(<<|$)

階層つきの名前（正規表現）:
(?<![A-Za-z0-9_.\]}])([A-Za-z_][A-Za-z0-9_]*(?:\.(?:\[[^\]]*\]|\{\}|[A-Za-z_][A-Za-z0-9_]*))+)

1 語の名前（正規表現）:
(?<![A-Za-z0-9_.\]}])([a-z][a-z0-9_]{2,})=

ANET_READ_CONFIG が読む欄（正規表現。2 つ目の引数の、最後の . より後ろを末尾の語とする）:
ANET_READ_CONFIG\(\s*[A-Za-z_]+\s*,\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)

相対キーの引数（正規表現）:
[(,]\s*"([a-z_][a-z0-9_]*(?:\.(?:\[[^\]"]*\]|[a-z_][a-z0-9_]*))+)"\s*[,)]

読み取りの口（固定文字列）:
ANET_READ_CONFIG   ReadConfig(   ReadSubConfig(   .Get(   .Get<   .Has(   .Set(   .Read(
MakeSubConfigData(   CreateReadScope(   MakeReadScope(   MakeDefaultConfigKey(   MarkInventory(
HasConfigValue(   RecordConfigValue(   GetSubConfigTags(   MakeTaggedSubConfigKey(   CopyValueFrom(
GetFullKey(   RecordFullKey(   ReadInheritedSubConfig(   .CopyFieldFrom(
```

この集計は、次の文を拾えない。`=` を伴わない 1 語の名前、実行のときに組み立てる名前、文面を変数に組み立ててから出す文、2 行以上にまたがる呼び出しの引数、すでに root を持つが誤った置き場を指す名前。拾えなかった文のうち分かっているものは §8.5 と §8.6 に挙げた。ほかの扱いは D20 による。

### 8.3 一覧

2026-11-21 の作業ツリーで流した結果である。

**1 段目（文）。** 303 文、44 ファイル。直す文が 178、対象外の文が 125 である。文ごとの規則の数は、R1 が 15、R2 が 104、R3 が 6、R4 が 35、R5 が 143 である（R5 の得かたは a が 37、b が 57、c が 31、e が 18。R4 は d が 35）。直す文は、文の中の直す名前の規則で数えた。対象外の文は、文の中の名前がどれも R1〜R3・R6 に当たる文で、表にはそのうち最初に当たった規則を書いた。R6 だけに当たった文は無い。直す文があるのは 30 ファイルである。

「直す文」は、行・種類・集計が拾った名前・規則/得かたの順に書く。規則/得かたは、文の中の直す名前のものである。名前の欄には、同じ文の中の、変えない名前（R1〜R3・R6 に当たるラベルと、文の区切りを名前と拾ったもの）も出る。直す文の中の変えない名前は、§8.7 の表の 27 文にあるものがすべてで、ほかの名前はどれも R4 か R5 に当たり、完全キーに直す。行番号は 2026-11-21 のもので、実装では名前と文面で探し直す。

| ファイル | 直す文（行・種類・名前・規則/得かた） | 対象外の文（規則: 行） |
|---|---|---|
| [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) | 92 INFO `learner.enabled` R5/a<br>311 エラー `skip.enabled` R5/a<br>330 INFO `gamma` R5/a<br>332 INFO `gamma` R5/a<br>600 エラー `intrinsic.value_mode` R5/a<br>603 エラー `learner.intrinsic.feature_gradient` R5/a<br>608 エラー `intrinsic.reward_scaler` R5/a<br>748 エラー `actor.[{}].nn_forward_key` R5/a | R2: 824・829・913 |
| [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) | 169 WARN `intrinsic.observation` R5/a<br>253 エラー `intrinsic.enabled`・`skip.enabled` R5/a<br>257 エラー `discount_mode` R5/a<br>278 WARN `intrinsic.value_mode`・`type`・`discount_mode` R5/a<br>304 WARN `intrinsic.value_mode` R5/a<br>392 エラー `actor.[train].policy` R5/a<br>403 WARN `actor.[train].policy.policy_type`・`target_policy.policy_type`・`use_optimistic_target` R5/a<br>524 WARN `learner.use_tbo`・`reward_scaler.use_dynamic_scaling` R5/a<br>580 エラー `skip.enabled` R5/a<br>595 エラー `learner.use_amp`・`learner.use_amp_bf16` R5/a<br>603 エラー `skip.enabled` R5/a<br>627 WARN `learner.enabled` R5/a<br>698 WARN `actor.[{}].skip.nn_forward_key`・`skip.nn_forward_mode` R5/a<br>735 INFO `target_window` R5/a<br>759 エラー `learner.munchausen.enabled`・`learner.use_double_dqn` R5/a<br>763 エラー `learner.munchausen.enabled`・`target_policy.policy_type`・`actor.[train].policy`・`use_optimistic_target` R5/a | なし |
| [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) | 1455 エラー `skip.max_length` R5/b<br>1773 エラー `skip.max_length`・`key` R5/b | R2: 404・485・579・594・600・661 |
| [dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) | 92 エラー `n_step` R5/b<br>306 エラー `quantile_mode` R5/b<br>314 エラー `quantile_mode` R5/b | R2: 64・112・120 |
| [dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) | 124 エラー `quantile_mode` R5/b<br>131 エラー `quantile_mode` R5/b | なし |
| [dqn_heads.cpp](../../src/algo/dqn/src/dqn_heads.cpp) | 149 エラー `intrinsic.value.head.input_mode`・`key` R5/b | R2: 44・255 |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) | 142 エラー `learner.{}.nn_forward_key` R5/b<br>163 エラー `use_amp`・`use_amp_bf16` R5/b<br>189 INFO `learner.enabled` R5/b<br>318 エラー `target_policy.policy_type` R5/b<br>328 エラー `learner.grad_noise.num_micro_batches`・`replay_batch_size` R5/b<br>332 エラー `learner.use_amp`・`learner.use_amp_bf16` R5/b<br>377 WARN `hard_update_interval` R5/b<br>1055 エラー `feature_gradient` R5/b<br>1532 エラー `final_feature_branch_key` R5/b<br>1540 エラー `final_feature_branch_key` R5/b<br>1586 INFO `update_warmup_steps` R5/b<br>1592 INFO `update_warmup_steps` R5/b<br>1837 エラー `skip.enabled` R5/b<br>1850 エラー `learner.skip.feature_gradient` R5/b<br>1913 WARN `learner.skip.feature_gradient` R5/b<br>1960 INFO `weight_decay`・`eps` R5/b | R2: 267・619・627・1335・1841・1866 |
| [dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) | 234 エラー `learner.per_initial_priority_mode` R5/b<br>244 エラー `learner.per_initial_priority`・`learner.per_initial_priority_mode` R5/b<br>248 エラー `learner.per_eps` R5/b<br>251 WARN `learner.per_eps`・`learner.use_per` R5/b<br>255 WARN `learner.per_initial_priority_mode`・`learner.per_alpha` R5/b<br>260 エラー `learner.per_prio_clip_value`・`learner.use_per_prio_clip` R5/b<br>264 WARN `learner.use_per_prio_clip`・`learner.per_prio_clip_value` R5/b<br>268 WARN `learner.per_prio_clip_value`・`learner.per_eps` R5/b | R3: 36・121・124・129<br>R2: 48・554 |
| [dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) | 110 エラー `skip.net` R5/b<br>298 エラー `skip.enabled` R5/b | R2: 143 |
| [image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp) | なし | R2: 249 |
| [muzero_based_agent.cpp](../../src/algo/muzero/src/muzero_based_agent.cpp) | 201 エラー `config.hidden_state_dim` R5/a | なし |
| [muzero_proto_agent.hpp](../../src/algo/muzero/src/muzero_proto_agent.hpp) | 135 エラー `start`・`end` R5/a | なし |
| [HeatMapPanel.cpp](../../src/apps/runner/src/HeatMapPanel.cpp) | なし | R2: 477・482 |
| [QValuePanel.cpp](../../src/apps/runner/src/QValuePanel.cpp) | なし | R2: 434 |
| [RunnerApp.cpp](../../src/apps/runner/src/RunnerApp.cpp) | なし | R2: 827 |
| [RunnerFrame.cpp](../../src/apps/runner/src/RunnerFrame.cpp) | なし | R2: 854 |
| [agent.hpp](../../src/core/base/include/anet/agent.hpp) | 48 エラー `key` R4/d | なし |
| [schedule.hpp](../../src/core/base/include/anet/schedule.hpp) | 122 エラー `key` R5/e<br>128 エラー `key` R5/e<br>133 エラー `key` R5/e<br>137 エラー `key` R5/e<br>158 エラー `key` R5/e<br>163 エラー `key` R5/e<br>170 エラー `key` R5/b<br>176 エラー `key` R5/e<br>183 エラー `key` R5/e<br>188 エラー `key` R5/e<br>192 エラー `key` R5/e<br>205 エラー `key` R5/e<br>210 エラー `key` R5/e<br>386 エラー `type` R5/b<br>389 エラー `type` R5/b<br>473 エラー `type` R5/b<br>527 エラー `type` R5/b | なし |
| [util.hpp](../../src/core/base/include/anet/util.hpp) | なし | R2: 170・230<br>R3: 279 |
| [agent.cpp](../../src/core/base/src/agent.cpp) | 71 エラー `class_id` R5/a | R2: 54 |
| [app_util.cpp](../../src/core/base/src/app_util.cpp) | なし | R2: 474 |
| [config.cpp](../../src/core/base/src/config.cpp) | 45 エラー `key` R5/c<br>511 エラー `key` R5/c | R1: 318・551・653・661・665・789 |
| [config_impl.cpp](../../src/core/base/src/config_impl.cpp) | なし | R1: 488・522・531・537・556 |
| [env.cpp](../../src/core/base/src/env.cpp) | 1180 エラー `class_id` R5/a | R2: 183・218・695・1142・1152 |
| [gui.cpp](../../src/core/base/src/gui.cpp) | なし | R2: 397 |
| [heat_map.cpp](../../src/core/base/src/heat_map.cpp) | なし | R2: 31 |
| [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) | 39 エラー `head.output_dim` R5/b<br>603 WARN `multiplier_mode` R5/b<br>654 WARN `statistics_mode` R5/b<br>792 INFO `warmup_mode`・`warmup_steps` R5/b<br>808 INFO `warmup_mode`・`warmup_steps` R5/b<br>921 エラー `statistics_mode` R5/b<br>932 INFO `learning_rate`・`weight_decay`・`adam_eps` R5/b | R2: 360・408・446・522・527・532・536 |
| [metrics.cpp](../../src/core/base/src/metrics.cpp) | なし | R2: 249・285 |
| [metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) | 96 エラー `metrics_logger.video_codec` R5/b | R2: 180・317・318 |
| [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) | 305 エラー `weight_norm.mode` R5/c<br>383 エラー `init2.mode`・`weight_norm.mode`・`mode` R5/c | R2: 55・57・131・236 |
| [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) | 21 エラー `key` R5/c<br>31 エラー `weight_norm.mode` R5/c<br>264 エラー `noisy.mode` R5/c<br>267 エラー `noisy.hold` R5/c<br>270 エラー `noisy.share` R5/c<br>273 エラー `noisy.count` R5/c<br>276 エラー `noisy.hold`・`noisy.share` R5/c<br>279 エラー `noisy.mode` R5/c<br>282 エラー `noisy.hold`・`noisy.share` R5/c<br>286 エラー `noisy.hold`・`noisy.count` R5/c<br>320 エラー `noisy.mode` R5/c<br>336 エラー `noisy.mode` R5/c<br>356 エラー `noisy.hold`・`noisy.share` R5/c<br>417 エラー `noisy.hold` R5/c<br>1712 WARN `res.dropout_rate`・`res.droppath_rate`・`res.norm_type` R5/c<br>1786 エラー `cn.channels` R5/c<br>1807 エラー `cn.norm_type` R5/c<br>1959 エラー `cn.channels` R5/c<br>1962 エラー `cn.kernel_size` R5/c<br>1965 エラー `cn.ffn_expand_ratio` R5/c<br>1968 エラー `cn.norm_type` R5/c<br>2754 エラー `weight_norm.mode`・`tf.use_sdpa` R5/c<br>2895 エラー `pool.kernel_size` R5/c<br>2898 エラー `pool.stride` R5/c<br>2901 エラー `pool.padding` R5/c<br>2904 エラー `pool.dilation` R5/c<br>3189 エラー `cos.num_basis`・`key` R5/c | R2: 346 |
| [observers.cpp](../../src/core/base/src/observers.cpp) | 456 エラー `output_key` R5/b<br>548 INFO `eval.[{}]` R4/d<br>553 INFO `eval.[{}]` R4/d<br>556 INFO `eval.[{}]` R4/d<br>606 INFO `eval.[{}]` R4/d<br>615 INFO `eval.[{}]` R4/d<br>702 INFO `channels_per_row` R5/b<br>809 エラー `interval` R5/e<br>1164 INFO `episode_interval` R5/b | R2: 92・874・1082・1086・1099・1130<br>R1: 1250・1465・1471 |
| [probe.cpp](../../src/core/base/src/probe.cpp) | 594 エラー `obs_key` R5/e<br>599 エラー `obs_key` R5/e<br>604 エラー `obs_key` R5/e<br>613 エラー `obs_key`・`index` R5/e<br>655 エラー `obs_key` R5/e | R2: 251 |
| [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) | 1242 エラー `unroll_steps` R5/b<br>2521 エラー `replay_capacity` R5/b<br>2526 エラー `per_initial_priority_mode` R5/b<br>2535 エラー `per_initial_priority_mode` R5/b<br>2538 INFO `num_envs.{}`・`config.capacity` R5/b | R2: 190・194・396・1239・1248・1486・1927・1933<br>R3: 1785 |
| [replay_experience.cpp](../../src/core/base/src/replay_experience.cpp) | なし | R2: 116 |
| [rl.cpp](../../src/core/base/src/rl.cpp) | なし | R2: 57・62・419・422・435・636・637・649・651・653・655・657・659・661・666・759・762 |
| [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) | 443 エラー `post_process_type` R5/b<br>956 エラー `gamma` R5/b | R2: 115・121 |
| [trainer.cpp](../../src/core/base/src/trainer.cpp) | 394 INFO `eval.[{}]` R4/d<br>1048 エラー `class_id` R5/a<br>1126 INFO `eval.[{}]` R4/d<br>1140 INFO `eval.[{}]` R4/d<br>1146 INFO `eval.[{}]`・`interval`・`batch_size` R4/d | R2: 244・569・1028 |
| [AtariEnv.cpp](../../src/env/atari/src/AtariEnv.cpp) | 651 エラー `game` R5/a<br>660 エラー `game` R5/a | R2: 957・962・968 |
| [AtariPreprocess.cpp](../../src/env/atari/src/AtariPreprocess.cpp) | なし | R2: 171 |
| [DropMergeEnv.cpp](../../src/env/dropmerge/src/DropMergeEnv.cpp) | 147 エラー `key` R5/a<br>155 エラー `key`・`no_drop_timeout_steps`・`use_no_legal_adjudication`・`use_no_drop_timeout_gameover` R5/a<br>221 エラー `action_mode` R5/a | なし |
| [GridMazeEnv.cpp](../../src/env/gridmaze/src/GridMazeEnv.cpp) | 261 エラー `width`・`height` R5/a<br>319 エラー `seed`・`width`・`height`・`wall_density`・`hole_density` R5/a | R2: 44・336 |
| [ImageClsEnv.cpp](../../src/env/imagecls/src/ImageClsEnv.cpp) | 270 エラー `dataset_key` R5/a | R2: 31・238・243 |
| [ImageData.cpp](../../src/env/imagecls/src/ImageData.cpp) | 98 エラー `dataset_key` R4/d<br>111 エラー `dataset_key` R4/d<br>118 エラー `dataset_key` R4/d<br>125 エラー `dataset_key` R4/d<br>136 エラー `dataset_key` R4/d<br>142 エラー `dataset_key` R4/d<br>149 エラー `dataset_key` R4/d<br>187 エラー `dataset_key` R4/d<br>192 エラー `dataset_key`・`width`・`height` R4/d<br>197 エラー `dataset_key` R4/d<br>305 エラー `dataset_key` R4/d<br>330 WARN `dataset_key` R4/d<br>340 エラー `dataset_key` R4/d<br>349 エラー `dataset_key` R4/d<br>353 WARN `dataset_key` R4/d<br>359 エラー `dataset_key` R4/d<br>363 WARN `dataset_key` R4/d<br>377 INFO `dataset_key` R4/d<br>402 エラー `dataset_key`・`index` R4/d<br>420 エラー `dataset_key`・`index` R4/d<br>482 エラー `dataset_key`・`field` R4/d<br>509 エラー `dataset_key` R4/d<br>525 INFO `dataset_key` R4/d<br>619 エラー `dataset_key`・`batch_size` R4/d<br>635 エラー `dataset_key` R4/d | R1: 272 |

**2 段目（呼び出し）。** 64 行、13 ファイル。直す呼び出しが 16 行（得かたは a が 2、b が 8、c が 6）、対象外の行が 48 行である。直す呼び出しがあるのは 6 ファイルで、どれも 1 段目の直す文があるファイルである。

| ファイル | 直す呼び出し（行・渡している相対キー・得かた） | 対象外の行 |
|---|---|---|
| [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) | 397 `target_policy.epsilon` a | なし |
| [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) | 469 `learner.munchausen.log_policy_mode` a | 159・184・195・265・267・269・271・273・275・282・284・428・430 |
| [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) | 736 `spatial.first` b<br>737 `spatial.last` b<br>740 `spatial.scale` b | なし |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) | 136 `learner.grad_noise` b<br>1939 `learner.learning_rate` b<br>1946 `learner.adam_eps` b | 131・160・166・168・171・367 |
| [dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) | 180 `model.soft_update_tau` b<br>191 `model.soft_update_tau` b | なし |
| [muzero_based_agent.cpp](../../src/algo/muzero/src/muzero_based_agent.cpp) | なし | 251・265・277 |
| [rl.hpp](../../src/core/base/include/anet/rl.hpp) | なし | 928・929・930・932・933・934・935・936・937 |
| [env.cpp](../../src/core/base/src/env.cpp) | なし | 174・177・200・207 |
| [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) | なし | 640・641・642・647 |
| [metrics.cpp](../../src/core/base/src/metrics.cpp) | なし | 115 |
| [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) | 1709 `res.droppath_rate` c<br>1710 `res.dropout_rate` c<br>1971 `cn.droppath_rate` c<br>2750 `tf.hidden_dropout_rate` c<br>2751 `tf.attn_dropout_rate` c<br>2752 `tf.droppath_rate` c | 220・221 |
| [trainer.cpp](../../src/core/base/src/trainer.cpp) | なし | 427・428 |
| [ImageData.cpp](../../src/env/imagecls/src/ImageData.cpp) | なし | 181・182・221・222 |

### 8.4 除外表（階層つきで、設定キーでない名前）

A9 で残ってよい名前。どれも R2 に当たるか、文の区切りを名前と拾ったものである。

| ファイル | 名前 |
|---|---|
| [util.hpp](../../src/core/base/include/anet/util.hpp) | `mode.{}` |
| [env.cpp](../../src/core/base/src/env.cpp) | `state.episode_start`・`continue_state.episode_start` |
| [observers.cpp](../../src/core/base/src/observers.cpp) | `y.size`・`v.size`・`eval.[name]` |
| [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) | `step.{}`・`num_envs.{}` |
| [rl.cpp](../../src/core/base/src/rl.cpp) | `action.[dimension]`・`action.[integer dimension]`・`state.obs`・`next_state.obs`・`next_states.obs`・`state.done`・`state.truncated`・`state.episode_start`・`next_state.done`・`next_state.truncated`・`next_state.episode_start`・`action.action` |
| [trainer.cpp](../../src/core/base/src/trainer.cpp) | `state_.obs` |
| [metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) | `recorded.{}`・`health.{}` |
| [AtariEnv.cpp](../../src/env/atari/src/AtariEnv.cpp) | `ram_metric.[n]` |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) | `attention_entropy.[{}]`・`samples.replay_item_keys`・`config.{}` |
| [dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) | `attention_entropy.[n]` |
| [dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) | `skip.{}` |

### 8.5 集計に現れない、直す文

§8.2 の集計には現れないが、直す文。8 つある。

| 位置 | 今 | 直すこと |
|---|---|---|
| [init.cpp](../../src/core/base/src/init.cpp) 104 行 | `ParseDevice` の例外を捕まえて投げ直す。すでに root を持つ | 投げ直しを消す（D8） |
| [init.cpp](../../src/core/base/src/init.cpp) 116 行 | キーの名前だけを並べる。すでに root を持つ | `キー='値'` で並べる（D9） |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363 行 | `DefaultDQNAgent.learner.noisy_probe` を固定で書く | 運ばれた記録を引く（D1・D5） |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 395 行 | `DefaultDQNAgent.learner.final_feature_branch_key` を固定で書く | 運ばれた記録を引く（D1・D5） |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 399 行 | 同上 | 運ばれた記録を引く（D1・D5） |
| [config.cpp](../../src/core/base/src/config.cpp) 353〜371 行 | 読まれないキーのエラー。文面を変数に組み立ててから出す。切り出した設定では相対キーを並べる。持ち主の名前は、呼ぶ側が固定の文字で渡す（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 864・1699 行は、Agent の下に書いた block でも共有の置き場の名前を渡す） | キーを完全キーで並べ、持ち主の名前を `ConfigData` の置き場で書く（§6.2） |
| [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 632〜634 行 | 使われない設定の通知。文面を変数に組み立ててから出す。先頭のキーは渡された接頭辞を持つが、続く `statistics_mode`・`error_norm.average_mode`・`warmup_steps` は相対キーである | 運ばれた記録を引く（D5） |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 1941〜1943 行 | checkpoint と違う optimizer の設定の通知。`learner.weight_decay` か `learner.noisy_weight_decay` を、2 行にまたがる呼び出しで渡す | 運ばれた記録を引く（D5） |

[dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) 110 行は、`skip.net` が相対キーなので §8.3 にある。同じ文の `DefaultDQNAgent.skip.network_mode` も、運ばれた記録を引く形に直す。

### 8.6 固定の root を書く文（D23）

すでに root を持つので §8.2 の集計には現れないが、値を与えた置き場と違うかもしれない置き場を固定で書く文。記録か `ConfigData` を引く形（§8.1 の a か c）へ直す。数は起草で数え切っていないので、実装計画で数え直す。

| 文のある場所 | 今 | 分かっている数 |
|---|---|---|
| 具象の Env（Atari・DropMerge・GridMaze・ImageCls・LunarLander・CartPole） | 上書きの置き場（`run.eval.[tag].env.*`）から読んだ値でも、既定の置き場の完全キー（`AtariEnv.mode=` など）を固定で書く | `.cpp` の 6 ファイルに少なくとも 27 行（2026-11-21 に、文字列の中の `<Env の型名>.<小文字>` を引いた。Atari 6、DropMerge 1、GridMaze 13、ImageCls 1、LunarLander 5、CartPole 1）。`.hpp` は数えていない |
| NN の設定のプロファイルと block（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp)） | Agent の下（`<Agent>.net.config_profile`・`<Agent>.net.block`）に書いた値でも、共有の置き場の名前（`net.config_profile.[X]`）を固定で書く。相対キーを `field '…'` の形で書く文もある | 5 行（866・869・1644 行と、1607・1643 行） |

### 8.7 直したあとの集計の期待

受入の A9・A10 が比べる先を決める。

**基準。** §8.3 は、コミット a6e920ed の `src` に §8.2 の集計を当てた結果である（集計の対象の `.cpp`・`.hpp` に、作業ツリーの変更は無かった）。

**直したあとに拾われてよいもの。** 実装を終えた `src` に §8.2 の集計を当てたとき、拾われてよいのは次の 3 つだけである。

- 1 段目の文のうち、§8.3 の「対象外の文」125 文。階層つきで root の無い名前は、§8.4 の除外表の名前だけが残る。
- 1 段目の文のうち、下の表の 27 文。§8.3 の「直す文」だが、変えない名前を持つので、直したあともその名前で拾われ続ける。直したあとに拾われる名前は、表の名前だけである。
- 2 段目の行のうち、§8.3 の「対象外の行」48 行。

下の表は、§8.3 の「直す文」178 文の名前を 1 つずつ §8.1 の規則に上から当て、R4・R5 に当たらない名前を持つ文と、直したあとにラベルが R1 として残る文を、すべて挙げたものである。

| ファイル | 変えない名前で拾われ続ける直す文（§8.3 の行、残る名前、当たる規則） |
|---|---|
| [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) | 1773（`key`。値は scalar の鍵。R2） |
| [dqn_heads.cpp](../../src/algo/dqn/src/dqn_heads.cpp) | 149（`key`。値は TensorDict の鍵。R2） |
| [schedule.hpp](../../src/core/base/include/anet/schedule.hpp) | 122・128・133・137・158・163・170・176・183・188・192・205・210（どれも `key`。今は R5 で、直したあとは後ろの値が完全キーになり、ラベルは R1 として残る） |
| [config.cpp](../../src/core/base/src/config.cpp) | 45・511（どちらも `key`。同上） |
| [nn_modules.cpp](../../src/core/base/src/nn_modules.cpp) | 21・3189（どちらも `key`。同上） |
| [replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) | 2538（`num_envs.{}`。文の終わりの `num_envs.` と次の文字列を、集計が 1 つの名前と拾ったもの。§8.4 の除外表にある。R6） |
| [DropMergeEnv.cpp](../../src/env/dropmerge/src/DropMergeEnv.cpp) | 147・155（どちらも `key`。今は R5 で、直したあとは後ろの値が完全キーになり、ラベルは R1 として残る） |
| [GridMazeEnv.cpp](../../src/env/gridmaze/src/GridMazeEnv.cpp) | 319（`seed`。設定に seed が無いときは導いた値になる。R2） |
| [ImageData.cpp](../../src/env/imagecls/src/ImageData.cpp) | 402・420（どちらも `index`。値は標本の位置。R2）、482（`field`。今は R5 で、直したあとは後ろの値が完全キーになり、ラベルは R1 として残る）、619（`batch_size`。R3） |

表に無い直す文の名前は、どれも R4 か R5 に当たり、完全キーに直す。ラベルに見える名前のうち、直すと決めたものは次のとおりである。

| 位置 | 名前 | 当たる規則と理由 |
|---|---|---|
| [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 278 行 | `type` | R5。値は、同じ文が完全キーで示す設定の、直下の欄（`type`）の値である（D25） |
| [nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 383 行 | `mode` | R5。設定キー `weight_norm.mode` の値に付けた、縮めたラベルである |
| [probe.cpp](../../src/core/base/src/probe.cpp) 613 行 | `index` | R5。値は sweep する軸の位置で、設定に欄がある。同じ文の `obs_key` と同じく、呼ぶ側が完全キーを渡す（得かた e） |
| [agent.hpp](../../src/core/base/include/anet/agent.hpp) 48 行 | `key` | R4。Actor のカタログの項目を指す識別子で、項目の完全キー（`<Agent の root>.actor.[<鍵>]`）で書く（D6） |
| [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 921 行 | 2 つの `statistics_mode` | R5。checkpoint の値と今の設定の値を並べた文で、checkpoint の側は R2 が除く名前である |
| [ImageData.cpp](../../src/env/imagecls/src/ImageData.cpp) 192 行 | `width`・`height` | R5。設定キー `image_width`・`image_height` の値に付けた、縮めたラベルである |
| [dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 1960 行 | `weight_decay`・`eps` | R5。設定から適用した optimizer の値に付けたラベルである |

**期待の一覧と比べ方。** 期待の一覧は、検証資材（§6.6）に置く。1 段目は 1 文を 1 行、2 段目は 1 行を 1 行とし、ファイル、種類（エラー・WARN・INFO・呼び出し）、拾われる名前の並びを持つ。行番号は持たない。実装で動くためである。起草時の期待の一覧は、基準のコミットに集計を当てた出力から、§8.3 の「直す文」のうち上の 27 文の表に無い 151 文と、「直す呼び出し」16 行を除き、27 文の名前を表の名前だけにしたものである。比べるときは、ファイルごとに、種類と名前の並びの組が同じ行の数を、集計の出力と期待の一覧で比べ、すべてのファイルとすべての組で数が一致することを確かめる。

**期待の一覧を変えたとき。** 受入が比べる先は、実装を終えた時点の期待の一覧である。起草時の一覧から変えてよいのは、次の 4 つの場合だけで、変えた行と、当てた規則（§8.1）を実装計画に書く。

- 直した文の、文字列のつなぎ方が変わって、変えない名前（R1〜R3・R6 に当たるラベル）の拾われ方が変わった。たとえば、`{}key=` と書かれていて拾われなかったラベルが、` key=` になって拾われるようになった場合である。設定キーが相対キーのまま残ったのでなければ、その文の行を今の出力に合わせる。
- 「直す」とした文や呼び出しを、値が設定から来ていないと分かって対象外へ移した（D19）。その文を足す。
- 一覧の外の文を見つけた（D20）。直したなら一覧は変わらない。§8.1 の規則で対象外と決めたなら、その文を足す。
- ほかの PRD が先に入って、集計の結果が変わった（§6.7）。取り直した出力を §8.1 の規則で仕分け直し、対象外の文と行を一覧に反映する。

## 9. 決めること（未決表）

決めずに残る行は無い。U1〜U3・U5・U9〜U12 は、2026-11-21 に推奨の既定で決め、確定事項（D19〜D25 と D12）に書いた。U4 は、今の作りで答えが出た。U6・U7 は範囲外である。U6 は既存の課題に追記した。U8 は PRD155 の側で扱い、その課題に追記した。

| 番号 | 事項 | 推奨の既定 | 扱い |
|---|---|---|---|
| U1 | §8.3 で「直す」とした文と呼び出しのうち、実装で、値が設定から来ていないと分かったもの | 対象外（R2）へ移し、理由を実装計画に書く。質問しない | 決めた（2026-11-21、推奨の既定。§5.3） |
| U2 | §8.2 の集計が拾えない文と、本書の起草のあとにほかの変更が足した文 | 一覧の文を直すときに、同じ文の中の設定キーは全部直す。一覧の外で見つけた文は、同じ PRD の移行漏れとして直し、実装計画に足す | 決めた（2026-11-21、推奨の既定。§5.3） |
| U3 | §8.1 の R3 に当てた 6 文（レビューの対応で、直す文の中の名前 1 つを足して 7 つにした）が、設定の誤りから届くか | 実装で呼ぶ側を読む。設定の誤りから届くなら、呼ぶ側（設定の型を持つ側）に検査を足し、完全キーで書く。届かないなら変えない | 決めた（2026-11-21、推奨の既定。§5.3） |
| U4 | NN の block の設定のうち、設定のプロファイルの補間でコードが書き込む項目の完全キー | 解決した。補間は既存の項目へ書き込むので、補間を定義した項目の完全キーが保たれる（§6.2、P28） | 起草のあと、別モデルの検討で、今の作りで答えが出ると分かった |
| U5 | 読み口が、読んだあとにコードで値を書き換える欄 | 記録は読んだときの完全キーのままにする。利用者が直す先はそのキーである | 決めた（2026-11-21、推奨の既定。§5.3） |
| U6 | 文脈を足す投げ直し 3 つ（Runner の Actor 作成、GridMaze の seed、保存ファイル）の fatal の二重出力 | 範囲外。[issue-20261103-c8fb06](../management/issues/issue-20261103-c8fb06.md)（PRD071 の課題）の対応方針に追記した（§5.1） | グリルで範囲外にした |
| U7 | 40 文字を超える tag 65 個の整理 | 範囲外。課題の候補にする（§2.2） | グリルで範囲外にした |
| U8 | PRD155 の §3.4 の、video_codec の行 | PRD155 を再グリルするときに外す（D14）。[issue-20261103-b4e5c9](../management/issues/issue-20261103-b4e5c9.md)（PRD155 の課題）の対応方針に追記した | PRD155 の側で扱う |
| U9 | すでに root を持つが、値を与えた置き場と違うかもしれない置き場を固定で書く文。具象の Env は、上書きの置き場から読んだ値でも、既定の置き場の完全キーを固定で書く（`AtariEnv.mode=` など。具象の Env の `.cpp` 6 ファイルに少なくとも 27 行。2026-11-21 に、文字列の中の `<Env の型名>.<小文字>` を引いた。`.hpp` は数えていない）。NN は、Agent の下に書いた block や設定のプロファイルでも、共有の置き場の名前を固定で書く（[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) の 866・869・1644 行と、相対キーを `field '…'` と書く 1607・1643 行） | 本書の範囲に含める。ステージ 3 で、これらの文を、記録か `ConfigData` を引く形（§8.1 の a か c）へ直す。数は実装計画で数え直す | 決めた（2026-11-21、推奨の既定。§5.3） |
| U10 | 規約（D12）の検査。§8.2 の集計のうち、階層つきの名前と、相対キーを渡す呼び出しは、機械で判定できる | 足さない（細部 7 のまま）。検証資材のスクリプトは残るので、要るときに流せる。別の案: [impl.md](../quality/impl.md) の機械の検査に、当たった行を報告に挙げるだけで差し戻しの根拠にしない項目として足す | 決めた（2026-11-21、推奨の既定。§5.3）。別モデルの検討で挙がった |
| U11 | 利用側が読んでいない設定の文面。`final_feature_branch_key (not read from config)='' is required …` は、RainbowAgent の利用者が設定では直せない誤りで、直す手立ては指標の購読を外すことである | 記録が返す文字（細部 1）は変えない。題材の 3 文（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363・395・399 行）には、記録に無いときに限り、設定では直せないことと、手立て（その診断指標の購読を外す）を 1 句足す | 決めた（2026-11-21、推奨の既定。§5.3）。別モデルの検討で挙がった |
| U12 | 項目や親の完全キーを示した文の中で、その直下の欄を並べる書き方（`eval.[greedy]: scheduled (interval=50, offset=3, …)`、`Invalid MuZeroAgent.… temperature schedule: start=… end=…` など） | 欄も完全キーで書く（D2・D3 のとおり）。別の案: 親の完全キーが同じ文にあるときに限り、直下の欄は欄の名前だけでよいとする（INFO の行が短く保てる） | 決めた（2026-11-21、推奨の既定。§5.3） |

## 10. issue の対応表

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261103-78df9c](../management/issues/issue-20261103-78df9c.md) | device指定の文法誤りで、ParseDeviceとInitRLが同じ誤りのfatalを2回ログに出す | 312_util | [init.cpp](../../src/core/base/src/init.cpp) 101〜105 行、[tensor_util.cpp](../../src/core/base/src/tensor_util.cpp) 29 行、[diag.cpp](../../src/core/base/src/diag.cpp) 22 行 | D8 | A1 |
| [issue-20261103-9b314a](../management/issues/issue-20261103-9b314a.md) | 明示cudaが使えないときのエラーがキー名だけを並べ、指定値を書かない | 312_util | [init.cpp](../../src/core/base/src/init.cpp) 110〜118 行 | D9 | A2 |
| [issue-20261103-eeb2c3](../management/issues/issue-20261103-eeb2c3.md) | video_codecのエラー文が実キーapp.metrics_logger.video_codecではなくrootの無い名前を出す | 317_metrics | [metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 97・101 行 | D1・D4・D5 | A3 |
| [issue-20261103-d4258c](../management/issues/issue-20261103-d4258c.md) | 実効deviceを設計文書は「採用値」と呼び、PRD070とJSONの欄はeffective（実効）と呼ぶ | 312_util、316_config | [100](../design/100_runtime_and_configuration.jp.md) 338・346 行、[020](../design/020_user_guide_run.jp.md) 67・623 行、[110](../design/110_agents_and_learning.jp.md) 191 行、[120](../design/120_environments.jp.md) 218 行、[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp) 659 行 | D10 | A5 |
| [issue-20261110-546bdb](../management/issues/issue-20261110-546bdb.md) | PRD103 の hidden 設定検査が現用定義と異なる指標名を要求し4環境で失敗する | 316_config | [dqn_config_baseline_test.cpp](../../src/algo/dqn/src/dqn_config_baseline_test.cpp) 108 行、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) 391 行、[140](../design/140_observability.jp.md) 431 行 | D11 | A6 |
| [issue-20261119-4124b6](../management/issues/issue-20261119-4124b6.md) | DQN の共通部品のエラー文 4 か所が DefaultDQNAgent の設定キーを固定で書く。欄を読まない利用側の設定を渡しても DefaultDQNAgent のキーが表示される | 321_agent_core、324_dqn | [dqn_network_model.cpp](../../src/algo/dqn/src/dqn_network_model.cpp) 110 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 363・395・399 行 | D1・D4・D5・D24 | A4 |

題材の 6 件で覆えない仕事（完全キーを運ぶ仕組みの残り、上書きの接頭辞が空の読み口の直し、§8.3・§8.5・§8.6 の一覧の移行。ステージ 2 の一部とステージ 3）は、次の課題が表す（細部 12）。本書の起票で作った。

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261121-9c0b1b](../management/issues/issue-20261121-9c0b1b.md) | エラー・WARN・INFO が設定キーを相対キーや固定の root で示す文が基盤全体に約 200 か所あり、完全キーを運ぶ仕組みが無い | 316_config、312_util、315_nn、317_metrics、318_rl_core、320_env_core、321_agent_core、322_env_individual、324_dqn、325_muzero | §8.3・§8.5・§8.6 の一覧の各ファイル、[config.hpp](../../src/core/base/include/anet/config.hpp) 224〜238・285〜288 行、[config.cpp](../../src/core/base/src/config.cpp) 42〜48・353〜371 行、[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp) 808〜829 行 | D2〜D7・D12・D18〜D25 | A7〜A14 |

## 11. グリルの材料

決定として扱わない。

- グリルの会話では、完全キーを「全名」と呼んでいた。設計文書 100 と `config.hpp` に「完全キー」という既存の語があったので、起草で置き換えた。
- 問い 2（範囲）で、AI は「題材の 6 か所だけ」を推奨し、ユーザーは「基盤全体」を選んだ。そのあと全数を数え直して、見積もり（階層つきの名前 53 個）が実際の半分以下だったことを示し、範囲を戻すこともできると添えて問い直した。ユーザーは「1 語の名前も含める」を選んだ。
- 起草で、§8.3 の仕分けは AI が 1 文ずつ読んで付けた。R3 と R5 の境目（関数の引数の事前条件か、設定の値か）と、値が設定から来ているかは、呼ぶ側を読み切っていない文がある（D19・D21）。
- 起草のあとの別モデル（Opus）の検討は、D1・D4 の方向に賛成したうえで、次を指摘した。`ConfigData` 自身のエラーが相対キーのまま一覧から漏れていること。集計が INFO の文と、相対キーを渡す呼び出しを拾っていないこと。block の `type` が block の設定に残らないこと。コードが書き込む項目の決まりが今の作りと食い違うこと。上書きの接頭辞が空の読み口が root 直下の相対キーを読むこと。記録の対象（dump に記録しない読み取り、読み口の中の値の写し）が契約に無いこと。一覧の仕分けの 2 件。固定の root を書く文が Env のほかに NN にもあること。AI はどれもコードで確かめて、本書へ反映した。
- 別モデルは、D4 の理由のうち「手で渡すと渡し忘れを検出できない」は、記録を引く形でも同じだと指摘した。部品が引き忘れて文字を固定で書けば、検出は集計頼みになる。理由から外した。
