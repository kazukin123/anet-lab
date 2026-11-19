# 設計文書200（DQN系Agent）の記述をコード・設定・用語集・ADRへそろえる PRD

> 2026-11-19 起票（anet-create-prd、Claude Code）。題材は、2026-11-18 の定常整備の監査（観点 drift、範囲は設計文書 200）で起票した課題 7 件。グリルでは、ゴールと範囲（§2）、ADR へのリンクの置き方（D14）をユーザーが選び、細部（D3・D15・D16・D17・D13 の注記の形）は推奨の既定で埋めた。用語集の「tau配置方式」の項目から実装の名前を消す直し（D8 の後半）は、PRD づくりの中で済ませた。

> 2026-11-19 工程末の起票の候補で、ユーザーの選択により、コードのエラー文に残る名前 `TauGenerator` の直しを課題 [issue-20261119-20603e](../management/issues/issue-20261119-20603e.md) として起票し、本 PRD の範囲に取り込んだ（D19、§7 の受入 10・11）。D19 のうち、同じ名前を持つ検証関数とテスト 14 本の名前の直しは、起草した側が足した既定で、ユーザーには確かめていない（D19 の理由）。取り込みに合わせて、§2・§3・D18・§5・§5.1・§6・§7 の受入 1・8・9・§8 の U1 も直した。未決表の U3（AGENTS.md 672 行）と U4（設計130 41 行）は、同じ選択で「この場で直す」で直した（§8）。U2 は見送った。

> 関連: 設計 [200](../design/200_dqn_agents.jp.md)・[140](../design/140_observability.jp.md)・[214](../design/214_value_decomposition.jp.md)・[030](../design/030_user_guide_analysis.jp.md)、[CONTEXT.md](../../CONTEXT.md)、ADR [0010](../adr/0010-actor-priority-mean-q-approx.md)・[0012](../adr/0012-replay-initial-priority-hint-completion.md)・[0013](../adr/0013-actor-network-resource-policy.md)

> 2026-11-19 改訂（着手前の再照合）。PRD182 のファイル名短縮（`e72bab34`）に合わせて参照先と受入手順を更新し、設計文書 200 のテスト一覧に足す対象を明記した。

## 1. 問題

### 1.1 経緯

2026-11-18 の監査が、設計文書 200 を今のコード・出荷設定・用語集・ADR と突き合わせ、ずれを課題 7 件に記録した（§3 の対応表）。監査のあとの 2026-11-19 に [PRD182](182_source_layout_10prd.md)（ソース配置の見直し）が入り（コミット `f25ff82b`）、課題の事象に書かれたコードのパス（`core/anet-core/...`）と行番号は古くなった。その後、コミット `e72bab34` で DQN 共通部品とテストのファイル名が短縮され、設計文書 200 のコードへのリンクとテスト一覧が追随した。本書の根拠は、2026-11-19 に改名後のコードと設計文書で確かめたものである。

PRD182 で、DQN の設定の型（`NetworkModelConfig`・`TauRuleConfig`・`ActionPolicyConfig`・`DQNActorConfig`・`MunchausenConfig`・`LearnerConfig`）は `agent.hpp` から [dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp)（17・31・44・145・206・226 行）へ移った。今の `agent.hpp` の「DQN」の見出し（[agent.hpp](../../src/core/base/include/anet/agent.hpp) 63 行）の下は空である。

### 1.2 今の記述と、今のコード・設定

行番号は 2026-11-19 のもの。「200:」は設計文書 200 の jp の行である。

| 課題 | 場所 | 今の記述 | 今のコード・設定・用語集 |
|---|---|---|---|
| 3aa088 | 200:147・152 | 「継続長j」「継続区間」 | 用語集の「スキップ長」は _Avoid_ に「継続長」を挙げる（[CONTEXT.md](../../CONTEXT.md) 92〜94 行）。「スキップ」は、選んだ行動をスキップ長の間続ける区間を指す（同 89 行） |
| 3aa088 | 200:57・293 | 「TD target」 | 「Bellmanターゲット」の _Avoid_ に「TD target」（同 30〜32 行） |
| 3aa088 | 200:239・485・576・619 | 「減衰後の実効tau」「UQEのtau減衰State」「UQE tau減衰」「UQE tau減衰schedule」 | 「時間の値」の _Avoid_ に「減衰（直線に限る印象）」（同 146〜148 行）。課題が挙げたのは 239 行だけで、ほかの 3 行は本 PRD のグリルで見つけた |
| 8933e7 | 200 §4（302〜308 行） | 設定の型・内的報酬・frame stack の置き場が無い。テストの行は `dqn_*_test.cpp` と `dqn_test.cpp` だけ | 設定の型は §1.1 のとおり。内的報酬は [intrinsic.hpp](../../src/core/base/include/anet/intrinsic.hpp)（20 行の `LifelongNoveltyEstimator`）と `src/core/base/src/intrinsic.cpp`・`intrinsic_rnd.hpp`・`intrinsic_rnd.cpp`、frame stack は [stacker.hpp](../../src/core/base/include/anet/stacker.hpp)（28 行の `DictFrameStacker`）と `stacker.cpp`。この一覧に未掲載で、今のパターンに当たらないのは `src/algo/dqn/src` の `prd101_ladder_test.cpp` と `src/core/base/src` の `intrinsic_test.cpp`・`stacker_test.cpp` |
| 8933e7 | 200:589 | 「方策とτの検証は、型と同じ`agent.hpp`の`ValidateActionPolicy`・`ValidateTauRule`が持つ」 | [dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) の 97 行に `ValidateTauRule`、110 行に `ValidateActionPolicy` がある。PRD182 の移動で古くなった（本 PRD のグリルで見つけた） |
| 8933e7 | 200:583 | 「Atari・GridMaze・LunarLanderの `run.@rnd`」 | [DropMerge.txt](../../apps/runner/config/DropMerge.txt) 532〜558 行にも `run.@rnd`（専用の盤面の CNN `DMRnd*`、β=0.01、`warmup_steps`=200,000、群 56 の 13 指標）があり、560〜566 行に vector も RND へ入れる `run.@rnd_vec` がある（2026-11-18 に、PRD を経ないコミット `513bb6e5` で追加。件名に PRD の番号が無い） |
| 95f463 | 200:487 | 「Atariの`run.@munchausen`」 | 実在するのは `run.@munch`（[Atari.txt](../../apps/runner/config/Atari.txt) 614・615 行、`LunarLander.txt` にもある）。`run.@munchausen` は `apps/runner/config` に 0 件 |
| 95f463 | 200:706、140:301 | `metrics.scalar.full` | 実在するのは `metrics.scalar.@full`（[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) に 117 行） |
| 95f463 | 140:301、030:179 | `metrics.scalar.iqn_search_p0` | 実在するのは `metrics.scalar.@iqn_search_p0`（[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 492 行ほか）。140:301 は課題の範囲（140 が書くプロファイル名）の同じ行、030:179 は課題の範囲の外で、どちらも本 PRD のグリルで見つけた |
| 95f463 | 200:245・393 | `use_stacker`・`stack_count`・`stack_keys`（接頭辞なし。文書全体で `stucker.` が 0 件） | コードは `stucker.use_stacker`・`stucker.stack_count`・`stucker.stack_keys` を読む（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 336〜338 行）。出荷設定は `agent.txt` 178〜180 行の `DefaultDQNAgent.@baseline.stucker.*`。綴りは `stucker` である |
| 95f463 | 200:785 | `num_micro_batches`・`ema_alpha`（接頭辞なし） | `learner.grad_noise.num_micro_batches`・`learner.grad_noise.ema_alpha`（同 493・494 行） |
| 95f463 | 200:733、§9.5 | probe の件数のキーを名指ししていない | `learner.plasticity.probe.batch_size`（コードの既定 512。[dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 348 行、`agent.txt` 290 行の `@baseline` は 4096）と `learner.policy_churn.probe.batch_size`（既定 1024。同 354 行、`agent.txt` 291 行）。読むのは [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 500・501 行 |
| 83eb00 | 200:237・281・576、§5 の図（354・355・367 行）、§6.1 の図（386 行） | Agent が Train・Eval・target の方策を分けて持つ、Agent が「複数Policy」を持つ、Agent が ActionPolicy を合成で持つ、と書く | Agent がメンバに持つ方策は `target_policy_` だけ（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 849 行、[rainbow_agent.hpp](../../src/algo/dqn/src/rainbow_agent.hpp) 137 行）。Actor の方策は `CreateActor` のたびにカタログ項目の `policy` から作り（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 751 行、[rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp) 124 行）、Actor が `policy_` として持つ（[dqn_actor.hpp](../../src/algo/dqn/src/dqn_actor.hpp) 292 行）。Learner は target の方策を参照する（[dqn_learner.hpp](../../src/algo/dqn/src/dqn_learner.hpp) 542 行） |
| 83eb00 | 200:239 | 「Trainの既定はrandom×32、Eval/targetはfixed×32」 | 方策の τ の既定は random×32（[dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 31〜34 行）。target は `ReadTargetPolicy` が fixed×32 に戻す（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 412〜415 行）。eval の fixed×32 は profile の `actor.@eval_base` が与える（`agent.txt` 223〜225 行） |
| 83eb00 | 200:257、§6.1 の図（405 行） | 「`actor_approx`の場合だけ、Train Actorは…hintを作る」「opt actor_approx」 | Actor を作る口は `use_per` かつ初期優先度の方式が `actor_approx` のときだけヒントを出させ、Actor の名前や用途を見ない（[dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 383・384 行）。ADR 0038 は「Actor Q ヒント送出」を RunMode への依存として外した |
| 83eb00 | 200:192 | 「`actor.[train].policy`をtarget方策の既定へコピーし、epsilonを0にする」だけ | `ReadTargetPolicy`（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 385〜421 行）は、分岐の前と `use_optimistic_target=false` の分岐で `policy_type` を Greedy、`epsilon` を 0 にする（387・388・409 行。楽観ターゲットでないときの既定）。`use_optimistic_target=true` では、コピーのあと `epsilon` と `uqe.epsilon` を 0 にし、`EpsilonGreedy` を `Greedy` にし、コピーした種類が UQE・ThompsonSampling でなければ一度 WARN を出す。`tau_rule` は fixed×32、`full_distribution_query` は既定に戻し、train から引き継がない。そのあと `target_policy.*` で上書きでき、上書きのあとで `spatial.enabled` を false にする |
| 9c6895 | 200:65・186・293 | 通常 Q の target・loss・TBO の変換を `TDLearner` が持つと書く | `TDLearner`（[dqn_learner.hpp](../../src/algo/dqn/src/dqn_learner.hpp) 709 行）はコンストラクタだけを持ち、コンストラクタ（[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 2285〜2295 行）は `SetupReplayBuffer` と `SetupOptimizer` を呼ぶだけ。計算は基底の `dqn::Learner` の `MakeTarget`（2460 行）・`ComputeElementError`（2407 行）・`ForwardCurrent`（2760 行）・`TransformH`/`TransformHInv`（1475・1481 行）にある。QR・IQN は `QuantileLearnerBase::MakeTarget`（2422 行）。200:292 は `dqn::Learner` が通常 Q の計算を持つと正しく書いており、文書の中で食い違う |
| 9c6895 | 200:295、[CONTEXT.md](../../CONTEXT.md) 83 行 | 表の `TauGenerator` の行（「…stateless component」）、「TauGenerator が担当し」 | `TauGenerator` という型は無い。実体は自由関数 `anet::rl::dqn::GenerateTaus` の 2 つの多重定義（[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 184 行の固定の範囲を受ける形、201 行の lane ごとの下限を受ける形）。名前が残るのは、同じファイルのエラー文 4 か所（121・124・129・210 行の「TauGenerator requires …」など）と、引数の検証関数 `ValidateTauGeneratorArguments`（118 行で定義、194・213 行で呼ぶ）、[dqn_test.cpp](../../src/algo/dqn/src/dqn_test.cpp) の τ の生成のテスト 14 本の名前（401〜672 行）である（2026-11-19 に `git grep -n TauGenerator -- src apps` で 21 行。`apps` は 0 件）。エラー文の中の「TauGenerator」を断定するテストは無い（657 行のテストは `sample_mode=invalid` と方式の一覧だけを見る） |
| 0fc0d7 | [ADR 0013](../adr/0013-actor-network-resource-policy.md) 3 行 | `CreateActor()` が `std::optional<bool> clone_model_override` を受け取ると決めたまま、後発の注記が無い | 今の口は `CreateActor(const ActorRequest& request)`（[rl.hpp](../../src/core/base/include/anet/rl.hpp) 814 行）で、`clone_model_override` は `src`・`apps` に 0 件。置き換えたのは [ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)（2026-09-08） |
| 0fc0d7 | [ADR 0010](../adr/0010-actor-priority-mean-q-approx.md) 3・4 行、[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md) 3 行 | 最新の注記が「0036 で 3 列」で止まる。0010 は「UQE再現見送りは不変」と注記する | ヒントは [ADR 0075](../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md)（2026-11-13）で常時 5 列。[ADR 0019](../adr/0019-iqn-uqe-score-without-extra-forward.md) は IQN+UQE のヒントを risk-biased action score から作ると決め、[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md) の Consequences はこれを ADR 0010 の近似の範囲として扱う |
| 7f7757 | 200 §11.1（838〜854 行）、214 | 200 の ADR へのリンクは 0010・0012・0013・0018・0019 の 5 本と、本文 767 行の 0039。214 には ADR へのリンクが無い | DQN 系の ADR 18 本（0001・0031・0032・0033・0035・0036・0038・0048・0053・0054・0058・0062・0063・0064・0065・0073・0074・0075）は、`docs/design/*.jp.md` のどこからもリンクされていない（2026-11-19 に `git grep -l "adr/<番号>-" -- 'docs/design/*.jp.md'` で 18 本とも 0 件） |

英訳（`docs/design/*.en.md`）にも、和文と同じ誤り（`run.@munchausen`、`metrics.scalar.full` など）がある。英訳は 1 行目に訳元の和文の blob を持ち、和文を直すと翻訳の skill（[anet-translate-docs](../../.agents/skills/anet-translate-docs/SKILL.md)）が「更新あり」として拾う。

### 1.3 前提

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | リポジトリ | PRD182 のファイル名短縮後のコードと設計文書 200 を基準にする | §1.1、開発者の発言（2026-11-19） | 済み |
| P2 | 用語・仕様 | DQN の用語は用語集、DQN の方策、学習、ヒント、設定の記述は現行コード・出荷設定・関連 ADR に合わせる | §1.2 | 済み |

## 2. ゴールと非ゴール

ゴール:

- 題材の課題 8 件（監査の 7 件と、工程末に起票して取り込んだ 20603e）がすべて片付く。
- 設計文書 200 と、140・214・030 の該当箇所、ADR 0010・0012・0013 の冒頭の注記、`CONTEXT.md` の 1 項目を、今のコード・出荷設定・用語集・ADR に一致させる。
- DQN 系の ADR に、設計文書から入れるようにする。
- コードの τ の生成の名前（エラー文・検証関数・テストの名前）を、文書と同じ `GenerateTaus` にそろえる（D19）。
- 直したことを、検索と読み合わせで確かめる（§7）。

非ゴール（退けた案と、それが本当の制約か）:

- D19 の名前の直しのほかに、コード・設定・テストを変える案 — 範囲外。本件は文書の記述と、文書から消す名前 `TauGenerator` のコードでの残りだけを扱う。制約ではなく別の関心であり、それで題材は片付く。
- 設計文書 200 を全面的に書き直す案、用語集の避ける語を文書の全体で全数点検する案 — 退ける。2026-11-19 に `CONTEXT.md` の _Avoid_ の語を機械的に 200 と照合したところ、課題の 3 語のほかの一致は「step」「tau」「統計」のような一般の語の誤検出だった。制約ではないが、直す対象が見つかっていない。
- ADR の本文を書き換える案 — 退ける。ADR は当時の判断の記録であり、後の決定で変わったときは冒頭に注記を足す（anet-create-prd の Offer ADRs sparingly、[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。本当の制約である。
- DQN 系以外の ADR（未決表 U2 の 42 本）への設計文書からの入口をつくる案 — 範囲外。題材の課題は DQN 系の ADR を扱う。制約ではなく別の範囲で、未決表 U2 に置く。
- ほかの文書にある同じ種類の誤り（[設計130](../design/130_neural_networks.jp.md) 41 行の接頭辞の無いキー、[AGENTS.md](../../AGENTS.md) 672 行の profile の名前）を本 PRD の実装の範囲に入れる案 — 範囲外。題材の課題の範囲（200・140）の外で、自己点検で見つけた。制約ではなく別の範囲で、工程末にユーザーの選択で「この場で直す」で直した（§8 の U3・U4）。
- 英訳の本文を本 PRD の実装で直す案 — 退ける（D17）。翻訳の skill が英訳の持ち主で、人の起動でだけ動く。制約ではないが、手で直しても、あとで skill が全体を訳し直す。
- 課題の事象の行（古いパスと行番号を書いたもの）を直す案 — 退ける。課題記録の事象は書いた時点の観測であり、過去の記録である（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針、PRD182 の D23）。本当の制約である。

## 3. issue の対応表

| ID | 題（要点） | area | 現在のコードで確かめた根拠の位置 | 確定事項 | 受入 |
|---|---|---|---|---|---|
| [issue-20261118-3aa088](../management/issues/issue-20261118-3aa088.md) | 200 が用語集で避ける語（継続長・TD target・減衰）を使う | 324_dqn | [CONTEXT.md](../../CONTEXT.md) 30〜32・89・92〜94・146〜148 行 | D1 | 2・9 |
| [issue-20261118-8933e7](../management/issues/issue-20261118-8933e7.md) | 200 のコードマップに設定型・内的報酬・stacker が無く、`run.@rnd` の列挙に DropMerge が無い | 324_dqn | [dqn_config.hpp](../../src/algo/dqn/src/dqn_config.hpp) 17〜226 行、[intrinsic.hpp](../../src/core/base/include/anet/intrinsic.hpp) 20 行、[stacker.hpp](../../src/core/base/include/anet/stacker.hpp) 28 行、[DropMerge.txt](../../apps/runner/config/DropMerge.txt) 532〜566 行 | D2・D3・D4 | 2・3・9 |
| [issue-20261118-95f463](../management/issues/issue-20261118-95f463.md) | 200・140 が実在しない名前と接頭辞の無いキーを書く | 324_dqn, 317_metrics | [Atari.txt](../../apps/runner/config/Atari.txt) 614 行、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt)、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 336〜338・493・494・500・501 行 | D5・D6 | 2・3・9 |
| [issue-20261118-83eb00](../management/issues/issue-20261118-83eb00.md) | 200 の方策の持ち主・ヒントの送出・楽観ターゲットが ADR 0038 と実装に合わない | 324_dqn | [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 385〜421・849 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 751 行、[dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 383・384 行、[dqn_actor.hpp](../../src/algo/dqn/src/dqn_actor.hpp) 292 行 | D9・D10・D11・D12 | 4・9 |
| [issue-20261118-9c6895](../management/issues/issue-20261118-9c6895.md) | 200 の部品表の TDLearner と TauGenerator が、計算を持たない型と無い名前を指す | 324_dqn | [dqn_learner.hpp](../../src/algo/dqn/src/dqn_learner.hpp) 709 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 2285〜2295・2460 行、[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 184・201 行 | D7・D8 | 2・4・9 |
| [issue-20261118-0fc0d7](../management/issues/issue-20261118-0fc0d7.md) | ADR 0013・0010・0012 に後発の注記が無い | 324_dqn | [rl.hpp](../../src/core/base/include/anet/rl.hpp) 814 行、[dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1617 行（ヒントを出す箇所） | D13 | 6・9 |
| [issue-20261118-7f7757](../management/issues/issue-20261118-7f7757.md) | DQN 系の ADR 約 20 本が設計文書から参照されない | 324_dqn | `docs/adr` の 18 本（§1.2 の最後の行） | D14・D15・D16 | 5・9 |
| [issue-20261119-20603e](../management/issues/issue-20261119-20603e.md) | DQN の τ の生成のエラー文・検証関数・テストの名前に、実在しない部品名 TauGenerator が残る | 324_dqn | [dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 118・121・124・129・194・210・213 行、[dqn_test.cpp](../../src/algo/dqn/src/dqn_test.cpp) 401〜672 行 | D19 | 9・10・11 |

## 4. 確定事項

| D | 決定 | 理由（依存するもの） |
|---|---|---|
| D1 | 200 の避ける語を用語集の語へ置き換える。147 行の「継続長j」は「スキップ長j」、「継続区間」は「スキップ」にする。152 行の図の「継続長jを選ぶ」は「スキップ長jを選ぶ」にする。57 行の「次のTD target」は「Bellmanターゲット」にする。239 行の「減衰後の実効tau」は「時間の値として評価した実効τ」にする。485 行の「UQEのtau減衰State」、576 行の「UQE tau減衰」、619 行の「UQE tau減衰schedule」は、「UQEのτの時間の値」を使う言い方にする（485 行は「UQEのτの時間の値のState」、619 行は「epsilonとUQEのτの時間の値」） | 用語集の canonical term（[CONTEXT.md](../../CONTEXT.md) 30〜32・89・92〜94・146〜148 行）。293 行の「TD target」は D7 で行ごと書き直す。485・576・619 行は課題が挙げていないが、同じ語・同じ文書であり、課題の題（「減衰」を使っている）を片付けるには全部を直す必要がある（2026-11-19 のゴールの合意） |
| D2 | 200 のコードマップ（§4）に置き場を足す。「DQN共通部品」の行に設定の型 `dqn_config.hpp` を足す。「内的報酬」の行を足し、`intrinsic.hpp`・`intrinsic.cpp`・`intrinsic_rnd.hpp`・`intrinsic_rnd.cpp` を挙げて詳細は 212 と書く。「frame stack」の行を足し、`stacker.hpp`・`stacker.cpp` を挙げる。589 行の「型と同じ`agent.hpp`」は「型と同じ`dqn_config.hpp`」にする | §1.1・§1.2 の置き場（PRD182 で設定の型が移った）。589 行は課題が挙げていないが、課題の「設定の型の置き場」と同じ事実で、PRD182 の移動で古くなった（2026-11-19 のゴールの合意） |
| D3 | 200 の §4 と §10 のテスト一覧に、`prd101_ladder_test.cpp` と、`src/core/base/src` の `intrinsic_test.cpp`・`stacker_test.cpp` を足す。両節にある `dqn_*_test.cpp` と `dqn_test.cpp` の案内も掲載する | `e72bab34` が両節のテスト一覧を現行名へ移行したため、今の一覧でまだ案内していない 3 本を追加する。§1.2 |
| D4 | 200:583 の `run.@rnd` の列挙に DropMerge を足す。DropMerge は専用の盤面の CNN（`DMRnd*`）、β=0.01、`warmup_steps`=200,000 を選び、`run.@rnd_vec` をその後ろへ足すと vector も RND へ入れる、と書く | [DropMerge.txt](../../apps/runner/config/DropMerge.txt) 532〜566 行。トピック別まとめの「選ばなかった形」にある、DropMerge 用の `run.@rnd` を範囲外とした [PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md) §1.3・§4.1 の再開条件（DropMerge で実験を起こすとき）は、2026-11-18 に PRD を経ないコミット `513bb6e5` で満たされた |
| D5 | プロファイルの名前を実在するものに直す。200:487 の `run.@munchausen` は `run.@munch` にする。200:706 と 140:301 の `metrics.scalar.full` は `metrics.scalar.@full` にする。140:301 と 030:179 の `metrics.scalar.iqn_search_p0` は `metrics.scalar.@iqn_search_p0` にする | §1.2 の出荷設定。140:301 の `iqn_search_p0` は課題 95f463 の範囲（140 が書くプロファイル名）の同じ行、030:179 は範囲の外だが同じ誤りで 1 語の直し（2026-11-19 のゴールの合意） |
| D6 | 設定キーに接頭辞を付ける。200:245 の `use_stacker`・`stack_count`（2 か所）・`stack_keys` の 4 か所すべてを `stucker.use_stacker`・`stucker.stack_count`・`stucker.stack_keys` にし、393 行の図の `opt use_stacker` を `opt stucker.use_stacker` にする。綴りは今のキーのまま `stucker` とする（キーの改名はコードと設定の変更で本 PRD の範囲外（§2）。本当の制約ではないが、[PRD161](161_observation_pipeline_10prd.md) の NG7 も改名しない）。785 行の `num_micro_batches`・`ema_alpha`（各 2 か所）を `learner.grad_noise.num_micro_batches`・`learner.grad_noise.ema_alpha` にする。733 行の「probe batch size」に、キー `learner.plasticity.probe.batch_size` と、コードの既定 512・`@baseline` の 4096 を添える。§9.5 の手順 1（743 行）に、probe の件数のキー `learner.policy_churn.probe.batch_size`（既定 1024）を添える | §1.2 のコードと出荷設定。`ema_alpha` は metrics の `$ema ema_alpha:` と同じ名前で取り違えやすい（課題 95f463 の影響）。同じ段落の中の 2 か所目も付けるのは、検索で確かめられる形（§7 の受入 2）にするため |
| D7 | `TDLearner` の書き方を直す。200:293 の行を「通常Qのとき選ばれる具象型。上書きを持たず、ReplayBufferとoptimizerを準備するだけで、target・誤差・forward・TBOの変換は基底の`dqn::Learner`が持つ」とする。65 行の「通常Qのlossを`TDLearner`」は、loss を基底の `dqn::Learner` が持ち、通常 Q の具象型が `TDLearner` である形にする。186 行のリンクの文字「TDLearner / QuantileLearnerBase」は「dqn::Learner / QuantileLearnerBase」にする。35・228 行の「通常Q → `TDLearner`」は具象型の対応として正しいので変えない | §1.2 のコード（[dqn_learner.hpp](../../src/algo/dqn/src/dqn_learner.hpp) 709 行、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) 2285〜2295・2422・2460 行）。200:292 の `dqn::Learner` の説明と食い違わなくなる |
| D8 | `TauGenerator` を実名に直す。200:295 の行を、自由関数 `GenerateTaus` が IQN の 5 種の tau 配置方式を指定の device 上で作り、固定の範囲を受ける形と lane ごとの下限を受ける形の 2 つがある、という内容にする。`CONTEXT.md` の「tau配置方式」の定義から「TauGenerator が担当し、」を消す（関数名に置き換えない）。`CONTEXT.md` の直しは PRD づくりの中で済ませた | [dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 184・201 行。用語集は実装の名前を持たない（anet-create-prd の Update CONTEXT.md inline、[CONTEXT-FORMAT.md](../../.agents/skills/anet-create-prd/CONTEXT-FORMAT.md) の Rules） |
| D9 | 方策の持ち主を実装に合わせて書く。200:237 は「Actor の方策は、カタログ項目ごとに `CreateActor` がその項目の `policy` から作り、Actor が持つ。Agent が持つのは target 用の方策だけ」という内容にする（Rainbow も同じ）。281 行の「複数Policy」は「target用Policy」にする。576 行の「Train/Eval/targetの選択」は「Actorカタログ項目とtargetの方策」にする。§5 の図は、`DefaultDQNAgent *-- ActionPolicy`・`RainbowAgent *-- ActionPolicy` に target 用と書き添え、`DqnActor --> ActionPolicy` を合成（`DqnActor *-- ActionPolicy`）にし、`DqnLearner --> ActionPolicy`（target）を足す。§6.1 の図の「Policyとsource Networkを選択」は「カタログ項目のpolicyからPolicyを作り、source Networkを選ぶ」にする | §1.2 のコード。ADR 0038 の「policy は Actor が専有」 |
| D10 | 200:239 の τ の既定の出どころを書き分ける。「方策の τ の既定（コード）は random×32、target の方策は fixed×32 に戻す、eval の fixed×32 は既定の設定（`agent.txt` の `actor.@eval_base`）が与える」という内容にする | §1.2 のコードと設定。コードの既定と profile の既定を分けないと、カタログ項目を新しく足した読者が eval 用でも random×32 になることに気づかない（課題 83eb00 の影響） |
| D11 | 200:257 と §6.1 の図（405 行）のヒントの条件を「PER 有効かつ初期優先度の方式が `actor_approx` のとき、Agent が作るすべての Actor（カタログ項目の名前や用途によらない）がヒントを作る」にする | [dqn_assembler.cpp](../../src/algo/dqn/src/dqn_assembler.cpp) 383・384 行。ADR 0038 が「Actor Q ヒント送出」を RunMode への依存として外した |
| D12 | 200:192 の楽観ターゲットに、コピーのあとに変える項目と戻す項目を足す。変える項目: `epsilon` と `uqe.epsilon` を 0、`EpsilonGreedy` を `Greedy`。戻す項目: `tau_rule` を fixed×32、`full_distribution_query` を既定（train から引き継がない）。そのあと `target_policy.*` で上書きできること、コピーした種類が UQE・ThompsonSampling でなければ一度 WARN を出すこと、上書きのあとで `spatial.enabled` を false にする（`target_policy.*` で指定しても lane ごとの配置は使わない）ことも書く。この最後の扱いは `use_optimistic_target` によらない。あわせて、楽観ターゲットでないときの既定（`policy_type` は Greedy、`epsilon` は 0。`tau_rule` と `full_distribution_query` と `spatial.enabled` の扱いは楽観ターゲットのときと同じ）も書く | [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 385〜421 行 |
| D13 | ADR 0013・0010・0012 の冒頭に後発の注記を足し、本文と既存の注記は変えない。注記の形は、0010・0012 の既存の注記と同じ `> **後発決定（<注記した日>）**: …` とし、既存の注記の後ろ（0013 は題と空行の次）に足す。中身は次のとおり。0013: `CreateActor` の `clone_model_override` 引数は ADR 0038 で無くなり、`CreateActor` は `ActorRequest` だけを受け取る。clone するかはカタログ項目の `clone_model` で決める。Learner が更新する source/shared network を Agent、private clone/snapshot を Actor が所有する分担は引き続き有効である。0010: Actor Q ヒントは ADR 0075 で常時 5 列（価値の分解の内的の Head の Q_sa と V_s を足す）になった。IQN+UQE では、ヒントの Q 系値は平均 Q ではなく、行動選択と同じ forward の risk-biased action score である（ADR 0019。ADR 0036 は本 ADR の近似の範囲として扱う）。追加 forward 禁止は不変である。0012: DQN payload の列数 `K` は ADR 0075 で 5 になった（価値の分解が無効なら後ろの 2 列は 0）。単一 carrier、completer、推定器の責務分離は不変である | ADR は本文を書き換えず、後の決定で変わったときは冒頭に注記を足し、何が置き換わり何が残るかを書く（anet-create-prd の Never rewrite an existing ADR、[PRD 品質確認観点](../quality/prd.md) 項番 23）。注記の形をそろえ、日付を注記した日にすれば注記の並びと日付が食い違わない（2026-11-19、細部の推奨の既定）。中身は §1.2 の ADR 0038・0075・0019・0036 と、200 §8.1 の所有の記述（618 行） |
| D14 | ADR へのリンクは、関連文書の ADR の一覧に足し、各行に関係する節の番号を添える。本文には足さない（本文に今ある 767 行の 0039 はそのまま残す） | 置き場が 1 か所で、130・150・100 の設計文書と同じ形になり、すでに長い 200 の本文を増やさない。節番号を添えれば節から ADR を引ける。監査の「入口なし」は設計文書からの参照の有無で判定する（[anet-audit](../../.claude/skills/anet-audit/SKILL.md) の drift）。2026-11-19 のグリルで、本文にも置く案・本文だけに置く案と比べてユーザーが選んだ |
| D15 | 200 §11 を「11.1 設計文書」「11.2 ADR」「11.3 参考論文」に分け、今の 11.1 にある ADR 5 本も 11.2 へ移す。11.2 は ADR の番号順に並べ、各行を `- [<短い題> ADR](../adr/NNNN-….md)（§x.y・§x.y）` の形にする（今ある行にも節を添える）。11.2 に載せるのは次の 22 本で、括弧は添える節。0001（§2.1.10・§7.1）、0010（§2.5）、0012（§2.5）、0013（§6.1・§8.1）、0018（§2.1.4）、0019（§2.4・§2.5）、0031（§9.4）、0032（§8.2・§9.3）、0033（§9.5）、0035（§2.1.7・§6.3）、0036（§2.5・§9.3）、0038（§2.4・§6.1・§6.4）、0039（§9.6）、0048（§7.3）、0053（§8.2）、0054（§9.8）、0058（§6.2）、0062（§6.5・§8.2）、0063（§6.5）、0064（§2.4）、0065（§2.5）、0075（§2.5・§9.3）。0013 は注記（D13）を足したうえで残す（所有の分担の判断が今も有効なため。D13） | 今の 11.1 は見出しが「設計文書」なのに ADR が混ざる。§11.1・§11.2 を名指すリンクはリポジトリに 0 件（2026-11-19 に、`200_dqn_agents.jp.md#11` と `200_dqn_agents.en.md#11` をそれぞれ固定文字列で検索した）。番号順は 130・150 の一覧と同じ。本文だけにあった 0039 も一覧に載せ、一覧で ADR を見渡せるようにする。添える節は、課題 7f7757 の事象が挙げた節（0001 の §7.1、0036・0075 の §9.3 など）に、200 で同じ判断を書くほかの節（0001 の §2.1.10、0036・0075 の §2.5 など）を足して決めた（2026-11-19、細部の推奨の既定） |
| D16 | 価値の分解の ADR 0073・0074 は 214 に置く。214 の末尾に「## 8. 関連文書」を新設し、D15 と同じ形の行で 0073（§2.2・§3.1）と 0074（§2.1）を載せる。0075 と 0036 は、5 列の表がある 200 に置く（D15） | 200 §9.10（797 行）が価値の分解の正本を 214 とする。ヒントの 5 列の表は 200 §2.5（257〜266 行）にある。214 は §7 で終わっていて関連文書の節が無いので、ほかの設計文書と同じく末尾に一覧を置く（2026-11-19、細部の推奨の既定） |
| D17 | 英訳（200・140・214・030 の `.en.md`）の本文は、本 PRD の実装では直さない。人が翻訳の skill を起動して追随させ、実装の完了報告でその起動を頼む | 翻訳の skill が英訳の持ち主で、人の起動でだけ動く。和文を直すと英訳の 1 行目の blob が古くなり、skill が「更新あり」として拾う。手で直しても、あとで skill が全体を訳し直す。[PRD182](182_source_layout_10prd.md) の D23 と同じ扱い（2026-11-19、細部の推奨の既定） |
| D18 | 次は変えない: D19 の 2 ファイルのほかのコード・設定・テスト（`src`・`apps`）、D19 のエラー文の名前のほかの文字とテストのタグ・本体、課題記録の事象の行、過去の PRD と実装計画（`docs/memo/done` ほか）、実験記録、ADR の本文と既存の注記、200:35・228 行の `TDLearner` の対応 | §2 の非ゴール。D7（35・228 行）。D19 |
| D19 | τ の生成のコードの名前を `GenerateTaus` にそろえる。[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) のエラー文 4 か所（121・124・129・210 行）の先頭の「TauGenerator」を「GenerateTaus」にし、引数の検証関数 `ValidateTauGeneratorArguments`（118 行、呼び出しは 194・213 行）を `ValidateGenerateTausArguments` に改名する。[dqn_test.cpp](../../src/algo/dqn/src/dqn_test.cpp) の τ の生成のテスト 14 本の名前の先頭の「TauGenerator」を「GenerateTaus」にする。別名や旧い名前の読み替えは残さない | 課題 20603e。D8 で文書から `TauGenerator` を消すので、エラー文とテストの一覧に出る名前を文書と同じ実名にそろえないと、表示の名前で文書もコードも引けない。検証関数は同じファイルの `static` で外から見えないが、エラー文と同じ部品の名前を持つので一緒に直す。エラー文の「TauGenerator」を断定するテストは無く（§1.2）、テストはタグ（`[dqn][iqn][tau]` など）で選ぶので、名前を変えてもテストの本体とタグで流す手順は変わらない。旧い名前を残さないのはクリーンブレーク方針（[AGENTS.md](../../AGENTS.md)）による。ユーザーが 2026-11-19 の工程末の起票の候補で選んだのは、エラー文の名前を直して本 PRD で対応することまでである。検証関数の改名とテスト 14 本の名前の直しは、`src` に `TauGenerator` を残さないために起草した側が足した既定で、ユーザーには確かめていない |

### 4.1 実装の順序

同じ設計文書を触る、まだ実装していない PRD がある。なかでも [PRD161](161_observation_pipeline_10prd.md)（観測パイプライン。課題は調査中）は、200 の §2.3・§2.4・§3・§6.1・§7.1・§9.1・§9.3 を移行対象に挙げ、本 PRD の D6（245・393 行）、D9（§3、§6.1 の図の 386 行）、D11（§6.1 の図の 405 行）と同じ段落を触る。PRD161 は §6.1 の「Actor の sequence 図」を移行対象に挙げている。[PRD183](183_temporl_skip_followups_10prd.md) は 200 の skip の節（§6.5）を触るが、本 PRD は §6.5 を触らない（§2.1.8 の語だけ）。

- 本 PRD には段階が無く、どちらが先でもよい。
- 本 PRD が先なら、あとの PRD は、本 PRD が直したあとの文（`stucker.*` の名前、Actor が方策を作る書き方）の上に自分の変更を書く。PRD161 の NG7 は `stucker.*` の設定キーを変えないので、D6 の名前はそのまま残る。
- PRD161 が先なら、本 PRD は §7 の受入 1 で 200 の §2.4・§3・§6.1 の今の文を読み直し、D6・D9・D11 が直す記述（接頭辞の無いキー、Agent が方策を持つ書き方、ヒントの条件）が残っていれば同じ直しを当て、無ければ当てない。読み直した結果（残っていた記述と、当てた直し）は実装の記録に書く。
- PRD161 は段階を持つが、本 PRD は 200 の言い回しと名前だけを変え、PRD161 の契約に触れないので、PRD161 の段階の途中で本 PRD を入れてもよい。そのときも上の読み直しを行う。

### 4.2 クラス

足す・変える・消すクラスは無い（文書の直しと、D19 の関数とテストの名前の直しだけ）。

## 5. 直す箇所と直さない箇所

| 場所 | 扱い | 決定 |
|---|---|---|
| 200:57・147・152・239・485・576・619 | 語を置き換える | D1 |
| 200 §4 のコードマップ、589 行 | 置き場を足す・直す | D2 |
| 200 §4 のテストの行、§10 の 821 行 | テストの書き方を広げる | D3 |
| 200:583 | DropMerge を足す | D4 |
| 200:487・706、140:301、030:179 | プロファイルの名前を直す | D5 |
| 200:245・393・733・743・785 | 設定キーに接頭辞を付ける・添える | D6 |
| 200:65・186・293 | `TDLearner` の書き方を直す | D7 |
| 200:295 | `GenerateTaus` に直す | D8 |
| `CONTEXT.md` の「tau配置方式」 | 「TauGenerator が担当し、」を消す（PRD づくりの中で済ませた） | D8 |
| 200:237・281・576、§5 の図、§6.1 の図（386 行） | 方策の持ち主を直す | D9 |
| 200:239 | τ の既定の出どころを書き分ける | D10 |
| 200:257、§6.1 の図（405 行） | ヒントの条件を直す | D11 |
| 200:192 | 楽観ターゲットの項目を足す | D12 |
| ADR 0013・0010・0012 の冒頭 | 注記を足す | D13 |
| 200 §11 | 3 つに分け、ADR 22 本の一覧にする | D14・D15 |
| 214 の末尾 | 「## 8. 関連文書」を新設し、0073・0074 を載せる | D16 |
| 200・140・214・030 の `.en.md` | 変えない（人が翻訳の skill で追随させる） | D17 |
| 200:35・228 | 変えない | D7・D18 |
| `dqn_math.cpp` の 118・121・124・129・194・210・213 行、`dqn_test.cpp` の τ の生成のテスト 14 本の名前 | 名前を `GenerateTaus` にそろえる | D19 |
| D19 の 2 ファイルのほかの `src`・`apps`、課題記録の事象の行、過去の PRD・実装計画・実験記録、ADR の本文と既存の注記 | 変えない | D18 |
| 題材の課題 8 件の対応内容と判定 | 実装する側が書き足す（`anet-impl-prd` の手順） | — |
| wiki の 324_dqn のトピック別まとめ「選ばなかった形」の DropMerge 用 `run.@rnd` の行 | 追加された旨を書き添える（PRD づくりの中で済ませた。§1.2 の 583 行の行を出典にする） | D4 |

### 5.1 受入 2 の語と名前に当たる、範囲の外のファイル

§7 の受入 2 の 3 つの検索の文字列を、範囲を限らずに追跡しているファイルの全体（`git grep -l`）と本書にかけると、2026-11-19 の時点（工程末に U3・U4 を直したあと）で 752 ファイルに当たる。そのうち直すのは 5 ファイル（200・140・030 の jp と、D19 の `src` の 2 ファイル）だけで、残りの 747 ファイルは次のとおり残す（U3・U4 で直した `AGENTS.md` と設計130 の jp は、直したあとは当たらない。直すファイルのうち、214 と ADR 0012・0013 は受入 2 の文字列に当たらない。`CONTEXT.md` と ADR 0010 は当たるが、当たる行は直す行ではないので、表の 4 行目と 5 行目で残すものとして数える）。行は上から順に当て、最初に当たった行で扱う。

| 行 | 当たるファイル | 件数 | 扱いと理由 |
|---|---|---|---|
| 1 | `docs/design` の 200・140・030 の `.jp.md` | 3 | 直す（D1・D3・D5・D6・D7・D8）。140 と 030 では、直す行のほかに metrics の EMA の係数 `ema_alpha` の記述（140 の 246・256・275・281・305 行、030 の 168・173・397 行）が受入 2-3 の文字列に当たる。設定キーではなく metrics の指定の記述なので残す（2-3 の範囲は 200 だけ） |
| 2 | `docs/design/*.en.md` | 6 | 残す。英訳は翻訳の skill で追随させる（D17） |
| 3 | ほかの `docs/design` の `.jp.md`（150・212・220） | 3 | 残す。150 と 220 の `stack_count` は ReplayBuffer の stack の数の記述、212 の「減衰」は重み減衰（weight decay）で、どれも設定キーや時間の値の意味ではない |
| 4 | `CONTEXT.md` | 1 | 残す。当たるのは、避ける語として挙げる _Avoid_ の行（32・84・94・144・148 行）、ReplayBuffer の history margin を `stack_count - 1` で書く 197 行（ReplayBuffer の stack の数の記述で、Agent の設定キーではない）、DropMerge の NoLegal の区間の長さを指す 319 行の「継続長」（スキップとは別の概念） |
| 5 | `docs/adr` | 6 | 残す。ADR の本文（D13・D18） |
| 6 | `docs/memo/done`・`frozen`・`dropped` | 56 | 残す。過去の PRD・実装計画・検証の記録（D18） |
| 7 | ほかの `docs/memo`（本書を含む） | 10 | 残す。ほかの PRD の本文はその PRD の工程で扱う。本書は古い語と名前を引用する |
| 8 | `docs/experiments` | 567 | 残す。実験記録と Run の設定の写し（D18）。設定の写しの `ema_alpha` は metrics の EMA の係数である |
| 9 | `docs/management` | 7 | 残す。課題記録と、道具が作る INDEX（D18） |
| 10 | `docs/wiki` | 4 | 残す。道具が作る課題の表（課題の題の引用）と、知見のページの Run の観測の記述 |
| 11 | `reports` | 17 | 残す。調査レポート（文献の語の「TD target」を含む） |
| 12 | `src/algo/dqn/src/dqn_math.cpp`・`src/algo/dqn/src/dqn_test.cpp` | 2 | 直す（D19）。当たるのは `TauGenerator` だけ |
| 13 | ほかの `src`・`apps` | 69 | 残す。コード・設定の注記（「減衰」「TD target」など）、検証資材の文書（D18）。設定の `ema_alpha` は metrics の EMA の係数である |
| 14 | `CHANGELOG.md` | 1 | 残す。変更の記録のすでにある項 |

受入 2 の 0 件を期待する文字列（`継続長`・`TD target`・`減衰`・`TauGenerator`・`dqn_based_*_test.cpp`・`run.@munchausen`・`metrics.scalar.full`・`metrics.scalar.iqn_search_p0`、接頭辞の無いキー）は、本書で書く新しい名前（`スキップ長`・`Bellmanターゲット`・`時間の値`・`GenerateTaus`・`dqn_*_test.cpp`・`run.@munch`・`metrics.scalar.@full`・`metrics.scalar.@iqn_search_p0`・`stucker.*`・`learner.grad_noise.*`）のどれの一部にもならない（2-3 は `.` の後ろに続く名前を数えない）。

## 6. 成功の測り方

- §7 の受入 2 の検索が 0 件になること。直す前（2026-11-19）は、2-1 が 9 行、2-2 が 5 行、2-3 が 3 行（9 か所）である。
- §7 の受入 10 の検索（`src`・`apps` の `TauGenerator`）が 0 件になること。直す前（2026-11-19）は 21 行である。
- DQN 系の ADR 18 本（§1.2 の最後の行）がすべて、設計文書（jp）から 1 本以上リンクされること（§7 の受入 5）。直す前は 18 本とも 0 件である。

## 7. 受入

受入 1 は実装に入る前に、ほかの項目は実装を終えたあとに、リポジトリの根で確かめる。受入 8 は、実装に入る前と終えたあとの両方で記録する。検索の範囲は、書いたパスのファイルだけである。受入 3・4 が挙げる 200 の行番号は実装前（2026-11-19）のもので、実装後は同じ記述の後継の行を指す。

1. 根拠の再確認（実装に入る前）。次がすべて成り立つ。どれかが違えば実装に入らず、本 PRD を再グリルする。
   - `git grep -n -F "class TDLearner final" -- src/algo/dqn/src/dqn_learner.hpp` が 1 件で、[dqn_learner.cpp](../../src/algo/dqn/src/dqn_learner.cpp) の `TDLearner::TDLearner` の本体が `SetupReplayBuffer` と `SetupOptimizer` の呼び出しだけである。
   - `git grep -n -E "(class|struct) TauGenerator" -- src` が 0 件で、`git grep -n -F "anet::rl::dqn::GenerateTaus(" -- src/algo/dqn/src/dqn_math.cpp` が 2 件である。
   - `git grep -n -F "ActionPolicy>" -- src/algo/dqn/src/default_dqn_agent.hpp src/algo/dqn/src/rainbow_agent.hpp` の結果が、`target_policy_` のメンバの行の 2 件だけである。
   - `git grep -n -A1 -F "const bool emit_hint" -- src/algo/dqn/src/dqn_assembler.cpp` の結果が、`use_per` と `ReplayInitialPriorityMode::ACTOR_APPROX` の 2 つだけで条件を作っている。
   - [default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) の `ReadTargetPolicy`（実装前の 385〜421 行）が、D12 が書く扱い（楽観ターゲットでないときの既定の Greedy・`epsilon` 0（387・388・409 行）、変える項目、戻す項目、`target_policy.*` による上書き、WARN、上書きのあとの `spatial.enabled`）のとおりで、そのほかに target の方策の値を変える行が無い。
   - `git grep -n -E "ANET_READ_CONFIG\(config_data, (stucker\.|learner\.grad_noise\.(num_micro_batches|ema_alpha)|learner\.(plasticity|policy_churn)\.probe\.batch_size)" -- src/algo/dqn/src/default_dqn_agent.hpp` が 7 件である。
   - `git grep -n -E "struct (NetworkModelConfig|TauRuleConfig|ActionPolicyConfig|DQNActorConfig|MunchausenConfig|LearnerConfig) " -- src/algo/dqn/src/dqn_config.hpp` が 6 件で、`git grep -n -E "inline void Validate(ActionPolicy|TauRule)\(" -- src/algo/dqn/src/dqn_config.hpp` が 2 件である。
   - 出荷設定: `git grep -n -F "run.@munch :" -- apps/runner/config/Atari.txt`、`git grep -n -F "metrics.scalar.@full" -- apps/runner/config/metrics_scalar.txt`、`git grep -n -F "metrics.scalar.@iqn_search_p0" -- apps/runner/config` がそれぞれ 1 件以上で、`git grep -n -F "run.@rnd : DefaultDQNAgent.intrinsic.beta = 0.01" -- apps/runner/config/DropMerge.txt` が 1 件である。
   - `git status --short -- src/algo/dqn/src/dqn_math.cpp src/algo/dqn/src/dqn_test.cpp` の出力が空である（受入 10 で `git diff HEAD` を比べる前提。2026-11-19 は空）。空でなければ、実装に入る前の差分を実装の記録に書き、受入 10 の比較はその差分を除いて行う。
   - `git grep -n -F TauGenerator -- src apps` の当たりが、`dqn_math.cpp` の 7 行と `dqn_test.cpp` の 14 行の 21 行だけで、`dqn_test.cpp` の当たりがどれも `TEST_CASE("TauGenerator ` で始まる（エラー文の「TauGenerator」を断定するテストが無い）。
   - 200 の §2.4・§3・§6.1 を読み直し、D6・D9・D11 が直す記述が残っているかを確かめ、結果（残っていた記述と、当てる直し）を実装の記録に書いた（§4.1。残っていなければ、その直しを当てない）。
2. 古い語と名前の不在。次の 3 つの検索が、それぞれ 0 件である。
   - 2-1: `git grep -n -F -e 継続長 -e "TD target" -e 減衰 -e TauGenerator -e "dqn_based_*_test.cpp" -- docs/design/200_dqn_agents.jp.md`
   - 2-2: `git grep -n -F -e "run.@munchausen" -e "metrics.scalar.full" -e "metrics.scalar.iqn_search_p0" -e TauGenerator -- 'docs/design/*.jp.md' CONTEXT.md`
   - 2-3: `git grep -n -o -E "(^|[^.a-z_])(use_stacker|stack_count|stack_keys|num_micro_batches|ema_alpha)" -- docs/design/200_dqn_agents.jp.md`（接頭辞の無いキー。`.` の後ろに続く名前は数えない）
3. 名前と置き場の記述がある。`dqn_*_test.cpp` と `dqn_test.cpp` は `e72bab34` で反映済みで、ほかの確認対象は 2026-11-19 の着手前の文書に無い。
   - `docs/design/200_dqn_agents.jp.md` を固定文字列で引いて、それぞれ 1 件以上: `metrics.scalar.@full`、`stucker.use_stacker`、`stucker.stack_count`、`stucker.stack_keys`、`learner.grad_noise.num_micro_batches`、`learner.grad_noise.ema_alpha`、`learner.plasticity.probe.batch_size`、`learner.policy_churn.probe.batch_size`、`GenerateTaus`、`prd101_ladder_test.cpp`、`intrinsic_test.cpp`、`stacker_test.cpp`。`dqn_*_test.cpp` は 2 件以上（§4 のテストの行と §10）。両節に `dqn_test.cpp` と、D3 で足す 3 本の案内がある。
   - `git grep -n -E "run\.@munch([^a-z]|$)" -- docs/design/200_dqn_agents.jp.md` が 1 件以上（`run.@munchausen` は数えない）。
   - `awk '/^## 4\. コードマップ/,/^## 5\./' docs/design/200_dqn_agents.jp.md` の出力（§4 のコードマップ）に、`dqn_config.hpp`・`intrinsic.hpp`・`intrinsic.cpp`・`intrinsic_rnd.hpp`・`intrinsic_rnd.cpp`・`stacker.hpp`・`stacker.cpp` がそれぞれある（実装前の §4 には、どれも無い）。あわせて、589 行の後継の行に `dqn_config.hpp` があり、`agent.hpp` が無い。
   - `git grep -n -F "run.@rnd" -- docs/design/200_dqn_agents.jp.md` の当たる行のうち、583 行の後継の行に `DropMerge` と `run.@rnd_vec` がある。
   - `git grep -c -F "metrics.scalar.@iqn_search_p0" -- docs/design/140_observability.jp.md docs/design/030_user_guide_analysis.jp.md` が各 1 以上、`git grep -c -F "metrics.scalar.@full" -- docs/design/140_observability.jp.md` が 1 以上。
4. 読み合わせ（実装する側が読み、結果を実装の記録に書く）。200 jp の次の箇所が、それぞれの決定の内容を書いている。
   - §2.4 の最初の段落（237 行）、§3 の表の `DefaultDQNAgent` の行（281 行）、§7.2 の表の ActionPolicy の行（576 行）、§5 の図、§6.1 の図が D9 のとおり。§5 の図に `DqnActor *-- ActionPolicy` があり、`DqnActor --> ActionPolicy` が無い。
   - §2.4 の IQN の段落（239 行）が D10 のとおり。§2.5 の 2 つ目の段落（257 行）と §6.1 の図が D11 のとおり。§2.1.11 の 2 つ目の段落（192 行）が D12 の変える項目・戻す項目・上書き・WARN・`spatial.enabled` の扱いと、楽観ターゲットでないときの既定を書いている。
   - §2.1.2 の最後の段落（65 行）、§2.1.10 の最後の段落（186 行）、§3 の表の `TDLearner` の行（293 行）が D7 のとおりで、§2.1.1 の表の DQN の行（35 行）と §2.3 の表の通常Qの行（228 行）は変わっていない。§3 の表の 295 行が D8 のとおり。
5. ADR への入口。
   - 200 の §11.2 が D15 の 22 本だけを番号順に載せ、各行の末尾に D15 の節の番号がある。§11.1 は設計文書（今の 11.1 にある設計文書の 7 本と `ownership_guideline.md`）だけ、§11.3 は参考論文である。
   - 214 の末尾に「## 8. 関連文書」があり、0073・0074 が D15 と同じ形の行で載っている。
   - 次のコマンドの出力の各行が 1 以上である（D15 と D16 の 24 本）。

     ```bash
     for n in 0001 0010 0012 0013 0018 0019 0031 0032 0033 0035 0036 0038 0039 0048 0053 0054 0058 0062 0063 0064 0065 0075; do echo "$n $(git grep -c "adr/$n-" -- docs/design/200_dqn_agents.jp.md | cut -d: -f2)"; done
     for n in 0073 0074; do echo "$n $(git grep -c "adr/$n-" -- docs/design/214_value_decomposition.jp.md | cut -d: -f2)"; done
     ```

6. ADR の注記。`git diff HEAD --no-renames --name-status -- docs/adr` に出る行のうち、本 PRD の実装で生じたものが 0010・0012・0013 の `M` だけである。3 本それぞれの `git diff HEAD -- <ファイル>` に、`-` で始まる本文の行（差分の見出しの `---` を除く）が無く、足した行が `> **後発決定（` で始まる注記 1 行と空行だけである。注記は、0013 が `0038-actor-config-catalog-without-runmode.md` へ、0010 が `0075-actor-q-hint-five-columns-carry-per-head-values.md` と `0019-iqn-uqe-score-without-extra-forward.md` へ、0012 が `0075-actor-q-hint-five-columns-carry-per-head-values.md` へリンクする（[PRD 品質確認観点](../quality/prd.md) 項番 28 と同じ見方）。
7. 用語集。`git grep -c -F TauGenerator -- CONTEXT.md` が 0 件である（D8。PRD づくりの中で直した分が残っている）。
8. 変えないもの。実装に入る前と終えたあとに、次の 2 つのコマンドの出力を実装の記録に書き、前後で同じである。1 つ目は、本 PRD が直してよいファイルを除いた追跡しているファイルの、作業ツリーと HEAD の差分の hash で、実装の前から変更のあるファイル（D19 の 2 ファイルのほかの `src`・`apps`、`docs/design/*.en.md` を含む）でも、実装で変えれば値が変わる。2 つ目は、除いた範囲の外の未追跡のファイルの一覧の hash である。前後で違うときは、違ったファイルの差分を示し、違いが本 PRD の実装によらないこと（別のセッションの変更など）を実装の記録に書く。

   ```bash
   git diff HEAD -- . ':!docs/design/200_dqn_agents.jp.md' ':!docs/design/140_observability.jp.md' ':!docs/design/214_value_decomposition.jp.md' ':!docs/design/030_user_guide_analysis.jp.md' ':!docs/adr/0010-*' ':!docs/adr/0012-*' ':!docs/adr/0013-*' ':!src/algo/dqn/src/dqn_math.cpp' ':!src/algo/dqn/src/dqn_test.cpp' ':!docs/management/issues' ':!docs/wiki' ':!docs/memo/190_*' | git hash-object --stdin
   git ls-files -o --exclude-standard -- . ':!docs/management/issues' ':!docs/wiki' ':!docs/memo/190_*' | git hash-object --stdin
   ```

   除くのは、本 PRD が直す 200・140・214・030 の `.jp.md` と ADR 3 本、D19 の `src` の 2 ファイル、題材の課題 8 件と道具が作る INDEX（`docs/management/issues`）、道具 `wiki_support.py maps` が作り直す「課題」の節と実装の手順が足す書き戻し・作業の記録（`docs/wiki`）、本 PRD と実装の記録（`docs/memo/190_*`）である。
9. リンク切れと課題。[memo README](README.md) §リンクの注意の検査を、直した `docs/design` の 4 本と ADR 3 本にかけて、`BROKEN` が 0 件である。題材の課題 8 件の「対応内容」に直した内容と変更の場所の行があり、AI の判定が「残件無し」である（`anet-update-issue` の判定）。
10. コードの名前（D19）。`git grep -n -F TauGenerator -- src apps` が 0 件である。`git grep -c -E "GenerateTaus (requires|received)" -- src/algo/dqn/src/dqn_math.cpp` が 4、`git grep -c -F ValidateGenerateTausArguments -- src/algo/dqn/src/dqn_math.cpp` が 3（定義 1 と呼び出し 2）、`git grep -c -F 'TEST_CASE("GenerateTaus ' -- src/algo/dqn/src/dqn_test.cpp` が 14 である。`git diff HEAD -- src/algo/dqn/src/dqn_math.cpp src/algo/dqn/src/dqn_test.cpp` の変わった行が、受入 1 で確かめた 21 行だけである。
11. ビルドとテスト。[AGENTS.md](../../AGENTS.md) の「検証」の節の手順でビルドが通り、DQN のテストのうちタグ `[tau]` を持つテスト（D19 で名前を変えた 14 本）がすべて通る。CUDA の無い環境で `[cuda]` のタグのテストが実行されずに終わるときは、そのことを実装の記録に書く。流したコマンドと結果を実装の記録に書く。

## 8. 未決表

| # | 事項 | 扱い |
|---|---|---|
| U1 | コードのエラー文（[dqn_math.cpp](../../src/algo/dqn/src/dqn_math.cpp) 118〜131・210 行）が、実在しない名前「TauGenerator」で始まる。D8 のあと、エラー文の名前で文書を引いても見つからない | 工程末に起票し（[issue-20261119-20603e](../management/issues/issue-20261119-20603e.md)）、ユーザーの選択で本 PRD の範囲に取り込んだ（D19） |
| U2 | DQN 系の 18 本のほかに、設計文書（jp）から 1 本もリンクされない ADR が 42 本ある（2026-11-19 に、`docs/design/*.jp.md` にある `adr/<番号>-` の形のリンクをすべて洗い出して確かめた。0003・0014・0016・0017・0020・0021・0022・0023・0024・0025・0026・0027・0030・0034・0040・0041・0043・0044・0045・0047・0049・0050・0052・0055・0056・0057・0059・0060・0061・0066・0067・0068・0069・0070・0071・0072・0076・0077・0078・0079・0080・0081）。工程や運用の ADR（0049・0068・0069・0077・0081 など）も混ざり、設計文書から入口を持つべきかは ADR ごとに違う | 範囲外（題材は DQN 系の ADR）。工程末に起票の候補として示す |
| U3 | [AGENTS.md](../../AGENTS.md) 672 行は、`config_data.txt` に `metrics.scalar.baseline.*`・`metrics.scalar.full.*` のような未選択の profile の定義が同居すると書く。2026-11-19 の Run（`run_20261119-144336_mz_tg_g99ne_rr4_s1`）の `config_data.txt` には、`metrics.scalar.@` で始まる行も `metrics.scalar.full.` で始まる行も 0 件だった（`M1.*` などのマージ元の行はある） | 工程末に、ユーザーの選択で「この場で直す」で直した。2026-11-19 に 4 つの workspace の新しい Run の `config_data.txt` で、`@` で始まる profile の定義の行が 0 件、マージ元（`A1.*`・`A2.*`・`M1.*`・`E1.*` など）の行があることを確かめ、672 行の例をマージ元の名前に替え、profile の定義は dump に出ないと書いた |
| U4 | [設計130](../design/130_neural_networks.jp.md) 41 行が、200:245 と同じく `use_stacker`・`stack_count`・`stack_keys` を接頭辞なしで書く | 工程末に、ユーザーの選択で「この場で直す」で直した（3 か所に `stucker.` を付けた。D6 と同じ直し） |
