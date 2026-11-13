# Evaluation Trace のボーナスの β を、学習の入口を通る前から用意する PRD

> 2026-11-13 起票。グリルを行った（問い 1 つと細部 6 件）。決定は §4。
> 起点: 課題 [issue-20261113-317d32](../management/issues/issue-20261113-317d32.md)。2026-11-13、Montezuma の RND の継続 Run（`run_20261113-113201_mz_tg_r123_s1_c50m`、β 0.01、倍率 `clipped`、上限 5）の 50M の `agent_close.anet` を `run.@evalonly` の Runner で読んだところ、Evaluation Trace の `rnd_err` と `rnd_norm` には値が入り、`rnd_bonus` は全行が空欄だった。
> 関連: [127_eval_trace_intrinsic_reward_columns_10prd.md](127_eval_trace_intrinsic_reward_columns_10prd.md)（Evaluation Trace の内的報酬の列。本 PRD は D2・D7 の一部と §5.2 の一部を置き換える。範囲は §9）、[128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md)・[121_extrinsic_intrinsic_value_decomposition_10prd.md](121_extrinsic_intrinsic_value_decomposition_10prd.md)（同じ場所を触る）、[930_serialize_10prd.md](930_serialize_10prd.md)（Run の step 数の保存）。ほかの PRD との関係は §9。

## 1. 目的と問題

### 1.1 問題

Evaluation Trace の `rnd_bonus`（鍵 `intrinsic.bonus`）は、学習の報酬に足す単位のボーナスで、β × (倍率 − 1) で作る。β は、学習の入口が評価した値を写して使う。学習の入口がまだ β を評価していない Run では、β が無いので列が空欄になる（[PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md) の D7）。

- 評価専用の Run（`learner.enabled=false`）では、学習の入口が β を評価しないので、最後まで空欄のままである（§2.3）。
- checkpoint を読んで学習を始めずに見る使い方でも、同じく最後まで空欄のままである（§2.3）。
- PRD127 の D7 は、空欄の不利益を「再開から最初の学習の入口までの間」だけと見て、β を保存しないと決めた。評価専用の Run で最後まで埋まらないことは、その比べ方に入っていなかった。

そのため、checkpoint を読んで EvalPanel で止まった局面を調べる使い方（Montezuma の Greedy の止まりの調査）で、内的報酬を学習の報酬と同じ単位で読めない。今は β × clamp(`rnd_norm`, 0, 上限 − 1) を手で計算している。

### 1.2 ゴール

- **G1**: 評価専用の Run で、学習を始めても始めなくても、`rnd_bonus` に学習の報酬に足す単位の値が出る。
- **G2**: 学習の挙動と、Evaluation Trace を使わない Run の結果を変えない（PRD127 の G2 を保つ）。
- **G3**: 題材の課題が片付く。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 扱いと理由 |
|---|---|
| checkpoint に Run の step 数を保存し、評価専用・継続の Run の予定（ε・学習率・β など全部）を、読み込んだ重みの step から続ける | 範囲外。β だけでなく全部の予定に効く別の関心で、永続化の本実装（[PRD930](930_serialize_10prd.md)）が持つ。本当の制約ではない（クリーンブレークで足せる）。入ったときの本 PRD の扱いは D1 の「変えるときの条件」 |
| β を checkpoint に保存する（課題の対応方針の案 b） | 不採用（D1）。本当の制約ではない（書式は変えられる）が、今ある RND の checkpoint を読めなくする |
| PRD127 のほかの決定（推定器と β を写す時機、ロックの取り方、列の配置）を変える | 範囲外。本 PRD のゴールに要らない。制約ではなく、変える理由が無い |
| 価値の分解が入ったときの列の意味の変化 | 範囲外。[PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md) が持つ（§9）。制約ではなく、別の PRD の線引き |

### 1.4 機構とゴールの対応

| | 機構 | すること | 効くゴール |
|---|---|---|---|
| A | 構築時の β | Agent を作るとき、β の予定を exp step 0 で評価して直近の β にする（D1） | G1・G2 |
| B | 学習を止めた入口での β | 学習を止めた Run でも、学習の入口が β を評価して直近の β にする（D2） | G1（β に予定を持たせた設定で、同じ行の Q・γ と時点をそろえる） |

先送りした機構は無い。

### 1.5 成功の測り方

- G1: 学習を一度も通らずに checkpoint を見る場合と、学習を止めた入口を通した場合と、新しい Run の起動直後に、ボーナスが §5.2 の表どおりの有限の値になる（§7 の 1〜3）。Montezuma の 50M の checkpoint を評価専用の Run で開くと、`rnd_bonus` が最初の行から入る（§7 の 6）。
- G2: Trace を挟んでも学習が変わらないことを確かめる PRD127 のテストが通り、学習を止めた入口が内的報酬の診断値を変えない（§7 の 4・5）。
- G3: 上の 2 つを満たし、課題の事象（評価専用の Run で `rnd_bonus` が全行空欄）が、人の確かめ（§7 の 6）で起きない。

## 2. 現状（2026-11-13 にコードと設定で確認）

行番号は 2026-11-13 の作業ツリーの位置である。[PRD128](128_dqn_agent_refactor_bundle_10prd.md) が同じファイルを触るので、実装時は関数の名前と近くの記述で探し直す。

### 2.1 β の評価と、Trace のボーナス

- β は設定 `intrinsic.beta` の予定（ProfiledValue。型は `constant`・`exponential`・`linear`・`cosine`・`cosine_restart`・`phased`。[schedule.hpp:52-60](../../core/anet-core/include/anet/schedule.hpp)）で、Agent が構築時に作る（[default_dqn_agent.cpp:122](../../core/anet-core/src/default_dqn_agent.cpp)）。予定は、作るときに設定を検証し、exp step 0 で一度評価する（[schedule.hpp:358-363](../../core/anet-core/include/anet/schedule.hpp)）。
- 学習の入口 `DefaultDQNAgent::UpdateFromBatch` は、RND が有効で `learner.enabled` が真のときだけ（[default_dqn_agent.cpp:851](../../core/anet-core/src/default_dqn_agent.cpp)）、入口の exp step で β を評価して直近の β として覚え（同 858）、評価したその値でボーナスを作る（同 859）。
- Trace の採点は、Actor が network を写し直すロックの中で推定器と直近の β を写し（同 692-718・797-799）、β が無ければボーナスを NaN にする（同 801-815）。写しを持たない Actor では、step ごとに読み取りロックの中で直近の β を読む（同 823-825）。
- 直近の β を読むのは、上の 2 か所（写しを作るときと、写しを持たない Actor の採点）だけである。`GetScalar` にもメトリクスにも出ない。学習のボーナスは、入口で評価し直した値で作る（同 858-859）。

### 2.2 Run の step と checkpoint

- checkpoint には、設定の文字列、Network、Optimizer、RND の本体と facade の状態（補助学習の credit、lane の抽選の乱数、成功回数）が入り、β も Run の step 数も入らない（[default_dqn_agent.cpp:334-383・385-448](../../core/anet-core/src/default_dqn_agent.cpp)）。設計文書 200 も、checkpoint に保存しないものに「Run の StepCounts と metric 系列」を挙げ、checkpoint の読み込みを「旧 Run の完全再開ではなく、新しい Run へ Network と Optimizer を引き継ぐ操作」と書いている（[200_dqn_agents.jp.md:620・623](../design/200_dqn_agents.jp.md)）。
- Run の step は 0 から数え、外から入れる口は無い（学習の Runner がカウンタを書くのは step を進めるときだけ。[trainer.cpp:152](../../core/anet-core/src/trainer.cpp)）。継続 Run（`run_20261113-113201_mz_tg_r123_s1_c50m`）も、前の Run の `agent_close.anet` を `auto_load_file` で読んで始めた新しい Run で（[実効設定の写し:816-817](../experiments/default-dqn/atari/config/run_20261113-113201_mz_tg_r123_s1_c50m.txt)）、step は 0 から数える。
- そのため、課題の対応方針の案 a（β の予定を、読み込んだ重みの exp step で評価する）は、その step が手に入らないので、今は作れない。
- Run の step 数と予定の進行位置を保存して続ける形は、暫定の [PRD930](930_serialize_10prd.md) が要求として持っている（同 PRD の「Run の `StepCounts` と schedule の進行位置」と、`StepCounts` を Run の継続の中核の状態として扱うという記述）。

### 2.3 学習を止めた Run と、checkpoint を見るだけの使い方

- `learner.enabled=false` でも、学習の Runner は train step ごとに学習の入口を呼ぶ（[trainer.cpp:613](../../core/anet-core/src/trainer.cpp)）。入口は、報酬のスケール、観測の正規化の統計の更新、目標の方策の予定の更新を今までどおり行い（[default_dqn_agent.cpp:848-849・909-911・927](../../core/anet-core/src/default_dqn_agent.cpp)）、内的報酬の部分（採点、ボーナス、補助学習、β の評価）だけを飛ばす（同 851）。設計文書 200 も「外側 Agent の前処理（Reward scale と ObservationNormalizer 統計、ActionPolicy の schedule）は無効時も従来どおり動く」と書いている（[200_dqn_agents.jp.md:450](../design/200_dqn_agents.jp.md)）。
- PRD127 の §2.6 は「学習の入口は呼ばれないので、β は評価されない」と書いている。「学習の入口は呼ばれない」の部分は誤りで、入口は呼ばれる。「β は評価されない」の部分は正しい。
- 出荷の評価専用の組 `run.@evalonly` は、学習の Runner を動かし、1 train step を 1 learn step として評価の間隔を数える（[Atari.txt:541-560](../../apps/runner/config/Atari.txt)、[200_dqn_agents.jp.md:450](../design/200_dqn_agents.jp.md)）。
- checkpoint を見るだけの組は、[Atari.txt:334-336](../../apps/runner/config/Atari.txt) にコメントで置いてある（`learner.enabled = false`・`auto_load_file`・`app.train_auto_start = false`）。学習を始めないので、学習の入口は一度も呼ばれない。`app.train_auto_start` の既定は true である（[common.txt:123](../../apps/runner/config/common.txt)）。

### 2.4 β の設定

- 使っている β は定数だけである。既定 0.1（[agent.txt:4](../../apps/runner/config/agent.txt)）、Taiga 式 0.00005（[Atari.txt:36](../../apps/runner/config/Atari.txt)）、Montezuma の継続 Run 0.01（[実効設定の写し:816](../experiments/default-dqn/atari/config/run_20261113-113201_mz_tg_r123_s1_c50m.txt)）。
- β に予定を持たせた設定は、追跡しているファイル（`git grep` で `intrinsic.beta.type`・`.start`・`.end`・`.steps`・`.phase` を引いて 0 件）にも、手元の Run の実効設定にも無い（2026-11-13）。

### 2.5 今のテストと検証資材

- [intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) の `DQN trace scores its Actor snapshot after synchronization`（1834 行）は、学習の入口を通る前に Actor を同期した行のボーナスが NaN であることを断定する（1856 行）。
- 同ファイルの `Evaluation-only checkpoint scores errors without a training beta`（2201 行）は、学習を止めた Agent が checkpoint を読んだときのボーナスが NaN であることを断定する（2237 行）。回すのは `learner.enabled` が偽の場合だけである（2225 行）。PRD127 の検証資材の受入の表が、このテストを名前で引く（[testdata/prd127/README.md:38](../../core/anet-core/testdata/prd127/README.md)）。

### 2.6 移行対象を洗った検索

2026-11-13 に `git grep -l -i -F`（追跡しているファイルの全体）で引いた。検索語と、当たったファイルの数は次のとおりである。

- コードの名前: `last_intrinsic_beta`（直近の β を持つ DefaultDQNAgent の非公開の状態）3（[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) と [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)、PRD128）、`intrinsic_beta_`（β の予定を持つ非公開の状態）4（同じコード 2、課題 08d557、PRD128）。ヘッダの 730 行には、直近の β の注記「学習の入口が直近に評価した値。checkpointには保存しない。」がある。
- テストの名前: `without a training beta` 2（テスト、PRD127 の検証資材）。
- 文書の言い回し: `βが無い` 2（設計文書 212、wiki の内的報酬のページ）、`β が無い` 2（実験記録、PRD127）、`βがまだ` 1（ユーザーガイド 020）、`ボーナスだけが未成立` 1（212）、`No beta because` 1（212 の英訳）、`until beta is established` 1（020 の英訳）、`have an unavailable bonus` 1（212 の英訳）、`It is not checkpointed` 1（212 の英訳）、`学習の入口を通らない` 1（実験記録）、`学習の入口は呼ばれない` 1（PRD127）、`学習の入口を通らず` 1（wiki の内的報酬のページ）、`再開直後`・`freshly resumed` 合わせて 4（020・212 の日英）。

## 3. 用語

- **直近の β**: ボーナスに使う β の値。PRD127 では「学習の入口が直近に評価した値」だった（PRD127 の D2・§5.2）。本 PRD で、Agent の構築も書き換える時機に加える（D1・§5.1）。PRD127 のこの定義の箇所は、§9 の置き換える範囲に入る。
- 誤差・倍率・ボーナス・標準化した値は、[PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md) §3 と設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md) の語のまま使う。

用語集（`CONTEXT.md`）は変えない。新しい名前を付けない。

## 4. 確定事項

2026-11-13 のグリルで決めた。D1 は問いへの開発者の答え、D2〜D8 は規約・現行コード・D1 から導いて示し、開発者の異議が無かったもの、D9〜D14 は推奨の既定を開発者が採った細部である。

### D1. 学習の入口を通る前の β は、Agent を作るときに exp step 0 で評価して用意する

RND が有効な DefaultDQN は、構築時に β の予定を exp step 0（この Run の step の始まり）で評価し、直近の β にする。学習の入口は、今までどおり評価した値で直近の β を上書きする。

理由:

- G1 のどの使い方（評価専用で学習を始める・始めない、起動直後、checkpoint からの再開直後）でも値が出る。学習の入口に頼る形は、checkpoint を見るだけの使い方で入口が一度も呼ばれない（§2.3）。
- checkpoint の書式を変えないので、今ある RND の checkpoint（課題の事象の 50M を含む）をそのまま読める。β を checkpoint に保存する形は、書式の識別子を上げて今ある checkpoint を読めなくし、課題の事象の checkpoint そのものを読めなくする（§4.1）。
- 今の設定は β がすべて定数なので（§2.4）、値は学習が足す値と一致する。
- 予定を持たせた β で「読み込んだ重みを学習していたときの β」を出すには、Run の step 数の保存が要る。それは全部の予定に効く別の関心（§1.3）である。
- Run の step は 0 から数える（§2.2）ので、exp step 0 は、この Run の予定の始まりの位置そのものである。黙った代替や特別な値ではない。

依存: §2.2・§2.3・§2.4。

変えるときの条件: Run の step 数を checkpoint から復元するようになったら（[PRD930](930_serialize_10prd.md)）、構築時の評価を、復元した step で評価する形に直す（開発者の答え「step 数の復元は永続化の本実装で検討する」による）。

### D2. 学習を止めた Run でも、学習の入口で β を評価する

`learner.enabled` が偽でも、学習の入口は、RND が有効なら入口の exp step で β を評価し、直近の β にする。採点、ボーナスの加算、補助学習、内的報酬の診断値の公開は、今までどおり飛ばす。

理由: 学習を止めた Run でも、外側の Agent の前処理（報酬のスケール、観測の正規化の統計、方策の予定）は Run の step で動き続ける（§2.3、[200_dqn_agents.jp.md:450](../design/200_dqn_agents.jp.md)）。β も同じ外側の Agent の予定なので、同じ step に従わせる。β に予定を持たせた設定で、同じ行の Q・γ（同期した時点の学習の step で評価される。PRD127 §2.4）とボーナスの β の時点がそろう。これは PRD127 の D4 が写す時機を network と同じにした理由（値の時点をそろえる）と同じである。

費用: 学習を止めた Run で、train step ごとに予定の評価が 1 回増える。評価は数回の算術で、入口がすでに持っている書き込みロックの中で行う。新しいロックも、device との転送も足さない。学習を止めた Run の train step は、env の step と行動選択の forward を伴うので、増える分は比べられないほど小さい（未測定。性能の受入は置かない）。

依存: D1、§2.3。

### D3. 帰結: RND が有効な DefaultDQN では、β が無い状態が無くなる

D1 により、RND が有効な DefaultDQN は、構築の時点から直近の β を持つ。PRD127 §5.2 の表の行 5（β が無い）は無くなる。ボーナスが NaN になるのは、RND が無効のとき、写しを持つ Actor で写しがまだ無いとき、誤差が有限でないときだけである（§5.2）。

根拠: D1。

### D4. 帰結: 新しい Run の起動直後は、ボーナスが 0 と出る

学習の入口を一度も通る前は、誤差の統計が空なので、標準化した値が NaN になり、倍率が 1 になる（PRD127 §5.1 の倍率の表の行 2）。ボーナスは β × 0 = 0 で、空欄ではなく 0 と出る。warm-up 中と同じ扱いである。

根拠: D1 と、PRD127 の D7 の 1 文目（学習が倍率を 1 にする条件では 0。0 は学習が実際に足す値と同じで、偽りではない）。この文は本 PRD のあとも変わらない（§9）。

### D5. 学習の挙動と checkpoint の書式は変えない

直近の β を読むのは Trace の採点だけで、学習のボーナスは入口で評価し直した値で作る（§2.1）。D1・D2 で直近の β を書く時機が増えても、学習へ渡る報酬、Replay、内的報酬の診断値、メトリクスは変わらない。checkpoint に足すものは無い。

根拠: §2.1、D1・D2。

### D6. ADR は起こさない

β を評価する時機を 2 つ足すだけで、あとから戻しにくい形ではない。ADR を起こす 3 条件のうち「戻しにくい」に当たらない。

根拠: anet-create-prd の「Offer ADRs sparingly」。

### D7. PRD127 の冒頭に、置き換えと訂正の注記を足す

PRD127 の冒頭に、次を 1 行で足す。§9 の置き換える範囲（PRD127 の D2 の β の出どころ、D7 の 2 文目、§5.2 の表の行 5 と表の下の 1 つ目の箇条）を本 PRD が置き換え、D7 の 1 文目と 3 文目は変わらないこと。PRD127 の §2.6 の「学習の入口は呼ばれない」は誤りで、入口は呼ばれ、学習を止めた分岐で内的報酬の部分を飛ばしていること（§2.3）。

根拠: [PRD094 の冒頭 30 行目](done/094_evalpanel_step_history_10prd.md)の前例（PRD105 が D11・D12 を置き換えた旨の行）と、[PRD 品質確認観点](../quality/prd.md)の項番 5（改訂の記録）。

### D8. PRD128 との順序

本 PRD と [PRD128](128_dqn_agent_refactor_bundle_10prd.md) は、どちらを先に実装してもよい。PRD128 が先に入れば、D1・D2 の評価は、PRD128 が作る内的報酬のクラスの中（学習の入口での採点と、Evaluation Trace の写しを持つ所）に置く。契約は変わらない。本 PRD が先に入れば、PRD128 は D1・D2 の評価をそのクラスへ移す。

根拠: PRD128 の D11（内的報酬の扱いを 1 つのクラスへ移す。保存の形式は変えない）と、D1・D5（本 PRD は保存の形式を変えない）。

### D9. 足すテスト

テストを 2 本足す。(a) 学習を止めた入口を通したとき、予定を持たせた β が入口の exp step の値になること（§7 の 2）。(b) 学習を止めた入口が内的報酬の診断値を変えないこと（§7 の 5）。§7 の 1 と 3 は、今あるテストを直して確かめる（D10）。

理由: 今あるテストで覆えない場合（予定を持たせた β、学習を止めた入口の副作用）と、D2・D5 を確かめるため。覆えるものは、今あるテストの断定を直して使う。

### D10. 直すテストと検証資材

ボーナスの NaN を断定している 2 本のテスト（§2.5）を、次のとおり直す。

- (a) 2201 行のテスト（評価専用の checkpoint）: `learner.enabled` の偽と真（再開直後）の両方で回し、ボーナスの NaN の断定を、§7 の 1 の断定に置き換える。名前から「without a training beta」を外し、PRD127 の検証資材の表（[testdata/prd127/README.md:38](../../core/anet-core/testdata/prd127/README.md)）の名前も同じに直す。
- (b) 1834 行のテスト（同期のあとの採点）: 学習の入口を通る前に同期した行の、ボーナスの NaN の断定（1856 行）を、§7 の 3 の断定（標準化した値が NaN、ボーナスが 0）に置き換える。

理由: D1 で、今の断定が成り立たなくなるため（D3）。どちらのテストも、§7 の 1・3 の場面をすでに組んでいる。

### D11. Trace を使わない Run の同一性は、テストで確かめる

実装の前と後の実行体を比べることはしない。PRD127 の「Trace を挟んでも学習が変わらない」テスト 2 本（§7 の 4）と、D9 の (b) で確かめる。

理由: 変える値を読むのは Trace の採点だけで（D5）、実行体を 2 つ作る手間に見合わない。

### D12. 人が確かめる受入

Montezuma の 50M の `agent_close.anet` を、`run.@evalonly` で学習を始める場合と、`app.train_auto_start = false` で学習を始めない場合の 2 通りで開いて確かめる（§7 の 6）。

理由: 課題の事象を、そのまま確かめるため。

### D13. 直す文書

設計文書 212 §4.6（表の「β が無い」行と、その下の β の段落）、ユーザーガイド 020 §6.5 の `rnd_bonus` の文、設計文書 200 §6.2 の `learner.enabled=false` の段落と、それぞれの英訳を直す（§6.5）。

理由: 「β が無ければ空欄」と書いている箇所と、学習を止めた入口の働きを書いている箇所を、全部合わせるため。

### D14. テストのタグ

足すテストと直すテストに `[prd129]` を付ける。直すテストは、今の `[prd127]` も残す。

理由: PRD127 と同じく、PRD の番号で受入のテストを引けるようにするため。

### 4.1 選ばなかった形

| 形 | 裁定 | 理由 |
|---|---|---|
| β を checkpoint に保存し、読んだら直近の β にする（課題の案 b） | 不採用 | 書式の識別子を上げ、今ある RND の checkpoint（課題の事象の 50M を含む）を読めなくする。新しい Run の起動直後は空欄のまま残る。β に予定を持たせた設定で「保存したときの β」が出る利点は、今は使う設定が無い（D1、§2.4）。PRD127 の D7 の 3 文目（β を checkpoint に保存しない）を、評価専用の Run の事象を入れて比べ直しても変わらない |
| 構築時に評価し、checkpoint があればその β で上書きする（両方） | 不採用 | 今ある checkpoint を読めなくする点が、上の行と同じ（D1） |
| 学習の入口でだけ β を評価する（学習を止めた Run でも評価するが、構築時には評価しない） | 不採用 | checkpoint を見るだけの使い方では入口が一度も呼ばれず、空欄のまま残る（§2.3）。G1 を満たさない |
| β の予定を、読み込んだ重みの exp step で評価する（課題の案 a） | 範囲外 | checkpoint に exp step が無く、今は作れない（§2.2）。Run の step 数の保存は PRD930 が持つ。再開条件: Run の step 数を checkpoint から復元するようになったとき（D1 の「変えるときの条件」） |
| 学習を止めた Run では、学習の入口で β を評価せず、構築時の値のままにする | 不採用 | β に予定を持たせた設定で、同じ行の Q・γ と β の時点がずれる（D2） |
| 実装の前と後の実行体で、Trace を使わない Run の metrics checksum を比べる | 不採用 | 変える値を読むのは Trace の採点だけで、実行体を 2 つ作る手間に見合わない（D11） |

## 5. 仕様

### 5.1 直近の β の決まり方

RND が有効な DefaultDQN の直近の β は、次の 2 つの時機にだけ書き換える。ほかの時機（checkpoint の読み込み、Actor の同期、Trace の採点）では変えない。RND が無効の DefaultDQN は、直近の β を持たない。

| 時機 | 直近の β にする値 |
|---|---|
| Agent を作るとき | β の予定を exp step 0 で評価した値（D1） |
| 学習の入口を通るたび（`learner.enabled` に依らない） | β の予定を、その入口の exp step で評価した値（D2） |

推定器と直近の β を Trace へ写す時機と、写しを持たない Actor で読む時機は、PRD127 の D4 のまま変えない。

### 5.2 Trace の内的報酬の値（PRD127 §5.2 の表の置き換え）

DefaultDQN の step 診断の 3 つの鍵の値は、次の表を上から順に当て、最初に当たった行で決める。PRD127 の表から行 5（β が無い）を除いたもので、ほかの行は変えない。

| # | 条件 | `intrinsic.error` | `intrinsic.normalized_error` | `intrinsic.bonus` |
|---|---|---|---|---|
| 1 | Agent が Rainbow | 知らない鍵（最初の読み出しで止まる） | 同左 | 同左 |
| 2 | `intrinsic.enabled=false` | NaN | NaN | NaN |
| 3 | 写しを持つ Actor で、部品に写しが無い（PRD127 の D20） | NaN | NaN | NaN |
| 4 | 誤差が有限でない lane | NaN | NaN | NaN |
| 5 | `next_state` を採点し、`intrinsic.use_done_mask` が真で、done の lane | 誤差 | 標準化した値 | 0 |
| 6 | それ以外 | 誤差 | 標準化した値 | 直近の β（§5.1）×（倍率 − 1） |

- 標準化した値と倍率は、PRD127 §5.1 の 2 つの表のとおりである。倍率は、warm-up 中、誤差の統計が空、除数が 0 のとき 1 になり、そのとき行 6 のボーナスは 0 である（D4）。
- 写しを持つ Actor では、行 6 の β は、Actor が最後に network を写し直した時点の直近の β である。写しを持たない Actor では、採点した時点の直近の β である（PRD127 の D4）。
- 行 3 は、RunnerApp では Evaluation Trace の pane を開いたときの同期で埋まる（PRD127 の D20）。学習を始めない Run でも、pane を開けば同期が入る。

## 6. 実装契約

### 6.1 公開の口と鍵

足す口も、変える口も無い。鍵（`intrinsic.error`・`intrinsic.normalized_error`・`intrinsic.bonus`）と列の宣言は、PRD127 のまま変えない。

### 6.2 設定と失敗の扱い

新しい設定キーは無い。新しい失敗も無い。β の予定は、今も設定の読み込みで検証し、Agent の構築時に exp step 0 で一度評価している（§2.1）。D1 は同じ評価の値を直近の β にするだけである。

### 6.3 checkpoint

書式も中身も変えない（D5）。

### 6.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `DefaultDQNAgent` | 変更 | RND が有効なら、構築時に直近の β を、β の予定の exp step 0 の値にする（D1）。学習の入口は、`learner.enabled` に依らず、入口の exp step で β を評価して直近の β にする（D2）。PRD128 が先に入っていれば、この 2 つは PRD128 の内的報酬のクラスが持つ（D8） | `default_dqn_agent`（`include/anet/default_dqn_agent.hpp`・`src/default_dqn_agent.cpp`） |

### 6.5 移行対象

検索の語と件数は、本 PRD の §2.6 にある。

- コード: [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) の構築（RND の推定器と β の予定を作る所。122 行付近）と、学習の入口の内的報酬の分岐（851・858 行付近）。[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp) の直近の β の注記（730 行。「学習の入口が直近に評価した値」）を、構築時の評価も含む言い方に直す。checkpoint に保存しない旨は残す。PRD128 が先に入っていれば、移した先のクラス（D8）。
- テスト（直すもの）: [intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) の 2201 行のテスト（D10 の (a)。§7 の 1。`learner.enabled` の偽と真の両方で回し、2237 行の NaN の断定を置き換え、名前を直す）と、1834 行のテスト（D10 の (b)。§7 の 3。1856 行の NaN の断定を置き換える）。
- テスト（足すもの）: §7 の 2 と 5 の 2 本（D9）。タグは、足すテストと直すテストのすべてに `[prd129]`（D14）。
- テスト（直さないもの）: §7 の 4 の 2 本は、断定を変えずに通す。学習を止めた Agent を使うほかのテスト（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) の `learner.enabled` に偽を入れるテスト）は、直近の β を読まないので断定は変わらない。
- 検証資材: [testdata/prd127/README.md](../../core/anet-core/testdata/prd127/README.md) の受入の表の行 8 のテストの名前（D10）。
- 設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md) §4.6: 表の「学習の入口をまだ通っておらずβが無い」行（486 行）を消し、その下の段落（490 行。「checkpointへ保存せず、評価専用・再開直後にはボーナスだけが未成立となる」）を、§5.1 の直近の β の決まり方に合わせて書き直す。β を checkpoint に保存しないことは書き残し、ボーナスが未成立になる部分だけを消す。英訳 [212 en](../design/212_rnd_lifelong_novelty.en.md) の同じ箇所（487 行の表の行と、491 行の段落。「It is not checkpointed」は残し、「so evaluation-only and freshly resumed Runs have an unavailable bonus」を消す）。
- ユーザーガイド [020](../design/020_user_guide_run.jp.md) §6.5: 「βがまだ決まっていない評価専用・再開直後は、ボーナスが空欄になります」（559 行）を、評価専用の Run でもボーナスが出ることと、起動直後と warm-up 中は 0 と出ることに改める。英訳 [020 en](../design/020_user_guide_run.en.md) の同じ文（560 行）。
- 設計文書 [200](../design/200_dqn_agents.jp.md) §6.2: `learner.enabled=false` の段落（450 行）の、無効時も動く外側の Agent の前処理の列挙に、β の評価を足す。英訳 [200 en](../design/200_dqn_agents.en.md) の同じ段落（452 行）。
- PRD127: 冒頭に注記を足す（D7）。本 PRD の起草で行った。
- wiki: トピック別まとめの [324 DQN 系列](../wiki/categories/324_dqn.md) と、[内的報酬のページ](../wiki/mechanisms/intrinsic_novelty.md)（「学習の入口を通らず」の言い回し）。本 PRD の起草の書き戻しで行った。
- 直さないもの: 実験記録（[2026-11-08_montezuma-rnd-taiga-aligned.md](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md) の「評価だけの Run では学習の入口を通らないので、β が無い」）、課題記録、PRD127 の本文（冒頭の注記のほか）、PRD128 の本文は、当時の記録か別の PRD なので書き換えない。PRD128 の §2.1 が数える、Agent が持つ内的報酬の状態の数は、本 PRD で変わらない（直近の β を書く時機を足すだけで、足す状態は無い）。

## 7. 成功の測り方と受入

1〜5 はテストで確かめる。6 は人が確かめる。数えるのは、テストの合否と、テストが断定する値である。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 学習を一度も通らずに checkpoint を読む | 学習した Agent の checkpoint を、学習の入口を一度も呼ばない Agent が読む。`learner.enabled` が偽の場合と真の場合（再開直後）の両方で、`clone_model = true` の Actor を同期したあと、`intrinsic.bonus` が有限で、同じ行の `intrinsic.normalized_error` を z として β × clamp(z, 0, `multiplier_max` − 1) で作った値と、浮動小数点の丸めの範囲で一致する（倍率 `clipped`、`warmup_steps = 0`、β は定数）。`intrinsic.error` と `intrinsic.normalized_error` は有限である |
| 2 | 学習を止めた入口で、予定を持たせた β が進む | `learner.enabled` が偽で、β を `linear`（start 0.2、end 0.6、steps 10）にした Agent が checkpoint を読む（1 と同じく倍率 `clipped`、`warmup_steps = 0`）。(a) 学習の入口を通る前に Actor を同期した行の `intrinsic.bonus` は、β 0.2 で 1 の式で作った値と一致する。(b) 学習の入口を exp step 5 で通したあと Actor を同期した行の `intrinsic.bonus` は、β 0.4 で 1 の式で作った値と一致する |
| 3 | 新しい Run の起動直後 | checkpoint を読まない新しい Agent（`learner.enabled` が真、統計の取り方は既定の whiten）で、学習の入口を通る前に `clone_model = true` の Actor を同期した行は、`intrinsic.error` が有限、`intrinsic.normalized_error` が NaN、`intrinsic.bonus` が 0 である |
| 4 | Trace を挟んでも学習が変わらない | PRD127 のテスト `Trace bonus equals the reward consumed by the real learner` と `Interleaved Trace leaves learner rewards and saved intrinsic state unchanged` が、断定を変えずに通る |
| 5 | 学習を止めた入口は、内的報酬の診断値を変えない | `learner.enabled` が偽の Agent が checkpoint を読み、学習の入口を 3 回通す。通す前と後で、`$agent` の `intrinsic.` で始まる鍵（[212](../design/212_rnd_lifelong_novelty.jp.md) の群 56 の鍵）の値がすべて同じである（NaN どうしは同じと数える） |
| 6 | 画面（人が確かめる） | 実装後の実行体で、Montezuma の継続 Run（`run_20261113-113201_mz_tg_r123_s1_c50m`）の `agent_close.anet` を、`run.@evalonly` で 2 通りに開く。(a) 学習を始める（既定）。(b) `app.train_auto_start = false` で学習を始めない。どちらでも、Evaluation Trace を開いた最初の行から `rnd_bonus` に値が入る。値の入った行を 3 行選び、0.01 × clamp(`rnd_norm`, 0, 4) と表示の 4 桁で一致する。GUI を動かせない枠で実装を終えるときは、測らずに終え、本 PRD の課題に残りの作業として書く（課題の判定は「残件有り」） |

文書の直し（§6.5）は、次の検索で確かめる。範囲は `docs/design/`、`core/anet-core/src/`、`core/anet-core/testdata/prd127/README.md` で、`git grep -n -F` で引き、0 件であること: `βが無い`、`ボーナスだけが未成立`、`βがまだ決まっていない`、`No beta because`、`have an unavailable bonus`、`until beta is established`、`without a training beta`。日本語と英訳の対応は、`βが無い` と `No beta because`（212 の表の行）、`ボーナスだけが未成立` と `have an unavailable bonus`（212 の段落）、`βがまだ決まっていない` と `until beta is established`（020）である。β を checkpoint に保存しない記述（212 の「checkpointへ保存せず」、英訳の「It is not checkpointed」）は、本 PRD のあとも正しいので、0 件の対象にしない。過去の記録（`docs/experiments/`、`docs/memo/`、`docs/management/`）は範囲に入れない（書き換えないため）。

## 8. 決めること（未決表）

なし。

## 9. ほかの PRD との関係

- [PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md): 本 PRD が置き換える範囲は次の 3 つで、ここを正本とする（ほかの節と PRD127 の冒頭の注記は、ここを指す）。
  1. D2 のうち β の出どころ: 題の「ボーナスの β は、学習の入口が直近に使った値」、本文の 2 文目の前半「β は、学習の入口が直近に評価した値を Agent が覚え」、本文の最後の文「学習の入口をまだ通っていない間は、ボーナスだけが空欄になる」。本 PRD の直近の β（§3・§5.1、D1・D2）に置き換える。D2 のほかの部分（出す値が 3 つであること、学習と同じ式で作ること、2 文目の後半の「推定器と同じ時機に写す」）は変わらない。
  2. D7 の 2 文目: 「学習の入口をまだ通っていない Run（起動直後、評価専用、checkpoint から再開した直後）では NaN にする」。本 PRD の D3・D4 に置き換える。D7 の 1 文目（ボーナスが 0 になる条件）と 3 文目（β は checkpoint に保存しない）は変わらない。
  3. §5.2: 表の行 5（β が無い）と、表の下の 1 つ目の箇条「β は、学習の入口が直近に評価した値である」。本 PRD の §5.2 の表と §5.1 に置き換える。

  あわせて、PRD127 の §2.6 の「学習の入口は呼ばれない」を訂正する（本 PRD §2.3）。冒頭に注記を足した（D7）。
- [PRD128](128_dqn_agent_refactor_bundle_10prd.md): 同じ箇所を内的報酬のクラスへ移す。順序はどちらでもよい（D8）。
- [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md): Trace のボーナスの列を、内的報酬用の報酬スケーラを通した値に β を掛けた値に変える。β は本 PRD の直近の β を使う。121 は報酬スケーラの状態を checkpoint に保存するので（121 の D16）、評価専用の Run でも、読み込んだ報酬スケーラと本 PRD の β で列が出る。
- [PRD930](930_serialize_10prd.md)（暫定）: Run の step 数と予定の進行位置を保存して続ける形を持つ。入ったら、D1 の構築時の評価を、復元した step で評価する形に直す（D1 の「変えるときの条件」）。
- [PRD919](919_eval_session_step_trace_10prd.md)（暫定・未実装）: 評価 session で同じ鍵を使う。本 PRD のあとは、session の開始の同期のあと、最初の step からボーナスが入る。

## 10. issue の対応表

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| issue-20261113-317d32 | 評価専用の Run では Evaluation Trace の rnd_bonus 列がずっと空欄になり、内的報酬を学習の報酬の単位で読めない | 324_dqn、110_runner_app | [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) の 851 行（β は `learner.enabled` が真の入口でだけ評価）・858 行（β の評価）・801-815 行（β が無ければボーナスを NaN） | D1・D2 | §7 の 1・2・6 |

## 11. 参照

- 設計文書 [212 RND による新しさの評価と内的報酬](../design/212_rnd_lifelong_novelty.jp.md) §4.6（Evaluation Trace で読む値）。
- 設計文書 [200 DQN 系の Agent](../design/200_dqn_agents.jp.md) §6.2（`learner.enabled=false`、checkpoint に保存しないもの）。
- ユーザーガイド [020 Run の実行](../design/020_user_guide_run.jp.md) §6.5（Evaluation Trace の列）。
