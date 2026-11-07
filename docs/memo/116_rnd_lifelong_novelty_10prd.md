# RND による一生の新しさだけの腕（内的報酬の第一の腕）暫定 PRD

> 2026-11-06 起票（暫定）。質疑を省き、決めきれない点は §8 の未決表に置いた。確定事項（§4）の各行は暫定の既定であり、会話で合意した決定ではない。
> 担当: 起票と精査を同じ枠でまとめる。理由は、実装の枠の予算が 2026-11-07 朝に切れる前に部品と対照腕を作り始めるためで、ユーザーの裁定（2026-11-06）で**暫定のまま実装へ出す**。実装は §4 の暫定の既定に従い、従えない事情があれば実装計画に理由を書いて変えてよい（質問しない）。設計の採否は [924](924_intrinsic_reward_novelty_10prd.md) の再グリルで見直す。
> 2026-11-06: 口の形（§5.1 の 4 操作と Builder、RND の本体を非公開に閉じること）はユーザーが確認した。同日夜に操作の名前・独立性・device の規約を見直した（2 行下）。
> 2026-11-06（実装中）: §9 の移行対象に設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md) の新設を追記した。起票時の書き漏れで、実装中にユーザーが要望した成果物を記録したもの。英訳は別途。
> 2026-11-06（実装中）: 口の形を見直した。`Multiplier` → `Estimate`、`Update` → `Learn` に改名し、`Learn` を渡された観測で自立して学習する操作にし、device の縛りを外した（§5.1）。改名の理由（impl.md §名前の 9）: Multiplier は名詞で処理が読めない。Update は統計の更新と紛れる。Score は anet で成績の語、Train は Train 側と torch の training mode の語なので避けた。独立化の理由: NGU では採点が行動時、補助学習が Learner 側の Replay 抽出になり、隣接した呼び出しで同じ観測を渡す前提の契約では分離できない。実装は直前の採点の計算グラフを持ち越していて、呼び順の誤りが黙って通っていた。
> 2026-11-07（実装中）: 補助学習の刻みと件数を lane 数（num_envs）に依らない形に改めた（D6・D7・D12・§5.1〜§5.7・§6.2・§8 の 4・5・12・13）。置き場は Push 側（facade の intake）のままとし、exp step 基準の credit と lane の抽選を facade に置き、`Learn` を「渡された観測すべてで 1 step」にした。葉は `intrinsic.rnd.train_fraction` を消して facade の `intrinsic.learn_ratio`・`intrinsic.learn_batch_size` にした（impl.md §名前の 9: Train 側の語を避けて `learn_` に揃え、Learner の `replay_ratio`・`replay_batch_size` と同じ形にする）。`warmup_steps` は採点した観測の件数にした。`Learn` は進めたかを bool で返す（実装中の確認で追加。warm-up 中に何もしなかったことを、呼ぶ側が数えられるようにするため）。経緯は §8.1。
> 起点: [924 §10](924_intrinsic_reward_novelty_10prd.md) は「RND 単独はそれだけでも検証できる独立の腕」とし、§9.1 の比較腕に「RND 単独」を置いている。924 は未決が 23 件あり、エピソード内の新しさ・Replay の契約・方策の家族を一度に決めないと実装へ出せない。RND は Replay も Actor も触らずに作れる唯一の部品なので、先に切り出す。
> 関連: [924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（内的報酬の全体。本 PRD はその第一の腕）、索引 issue [issue-20261106-16b7af](../management/issues/issue-20261106-16b7af.md)、[DQN 系 Agent 設計 §6.2・§8.2](../design/200_dqn_agents.jp.md)、[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)（補助 Network の置き場。§4 D3 で関係を書く）、[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)（checkpoint 読込時の optimizer options）、[CONTEXT.md](../../CONTEXT.md) の「報酬スケーラ」。

## 1. 問題とゴール

### 1.1 問題

anet の DQN 系 Agent には、観測の新しさに応じて Agent 自身が足す報酬（内的報酬）が無い。[924](924_intrinsic_reward_novelty_10prd.md) はエピソード内の新しさ（NGU の k 近傍、E3B）、RND による一生の新しさ、方策の家族を 1 本で扱うが、採点点・Replay の契約・方策の表現が未決で、実装の単位にできない。

RND（Random Network Distillation）は、固定の乱数 Network の出力を学習する Network が真似し、その予測誤差を新しさとする方式である。一生の履歴に対する新しさなので lane 別の記憶を持たず、採点は観測 1 つで閉じる。924 §2.5 の見立てでは、KFM のボス戦の壁には効かない（挑み始めると画面に慣れて消える）ので、この腕は KFM では**対照腕**であり、判断に効くのは GridMaze の到達率と、部品・費用・統計の確認である。

### 1.2 ゴール

1. RND による一生の新しさの内的報酬を、`DefaultDQNAgent` の学習報酬に足せる。既定は無効で、無効のときは今と同じ学習系列（metrics checksum の一致）になる。
2. 一生の新しさの口（§5.1）と RND の実装を、924 の最終形でもそのまま使える部品として作る。Replay・Actor・NetworkModel の契約は変えない。
3. 内的報酬の大きさ・予測誤差の統計・報酬全体に占める割合・費用を指標で見られる。
4. 腕 `run.@rnd` を Atari・GridMaze・LunarLander で回せる。効果の判定は 924 §9.1 に従う（§7）。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 扱いと理由 |
| --- | --- |
| エピソード内の新しさ（k 近傍・E3B）、Embedding と逆動力学 | 別の関心。924 の再グリルで採点点と Replay の契約を決めてから作る。技術の制約ではなく順序の選択 |
| 方策の家族（β_i・γ_i の組）、β=0 の評価 | 924 の問い 1・11 で決める。本 PRD の価値は 1 本で、評価も内的報酬込みで学んだ Q を使う |
| 内的報酬を外的報酬と別に Replay へ保存すること | Replay の契約の変更（924 問い 4）。本 PRD は Push の前に報酬へ畳む仮の運び方（§4 D2）を使い、924 の契約が入った時点で置き換える。制約ではなく移行の順序 |
| 2 本の価値 Head（RND 論文の V_E＋V_I）と非エピソード的な内的収益 | 価値の分解は 924 §6 の別の選択肢。本 PRD では終端で切る 1 本の価値のまま |
| Actor での採点（924 の準備が推す最終形） | 採点を Actor に置くと 1 step 遅れる採点を Replay の注釈で運ぶ契約が要る。facade で次観測を採点すれば終端観測も採点でき契約を増やさない。最終形への移行は 924 |
| 評価（configured eval・EvalPanel）での採点と診断 | eval の runner は `UpdateFromBatch` を呼ばず、採点点が facade なので構造上届かない。評価で内的報酬を見たくなったら 924 の採点点の問いで決める |

## 2. 現状（2026-11-06 にコードと文書で確認）

- 内的報酬・RND の実装は無い（`core`・`apps/runner` を `intrinsic`・`novelty`・`rnd` で検索して該当なし）。メトリクスの群番号 56 は設定・コードとも未使用（`56_` で当たる行はすべて群の中の項目番号）。
- `DefaultDQNAgent::UpdateFromBatch`（[default_dqn_agent.cpp:624](../../core/anet-core/src/default_dqn_agent.cpp)）は `unique_lock` の中（:632）で報酬を `reward_scaler_->Scale`（:635）に通し、`obs_norm` の統計を更新し（:638-639）、スケール済み報酬で `BatchExperience` を組み直して（:643-648）内側の Learner へ渡す（:651）。Replay の `Push` は内側の [dqn_based_agent.cpp:4030](../../core/anet-core/src/dqn_based_agent.cpp) にあり、同じロックの範囲内である。`learner.enabled=false` では Push しない（:4025-4027）。
- 遷移の終端は `BatchExperience.next_state` の `done`・`truncated` で届く（[rl.hpp:423-427](../../core/anet-core/include/anet/rl.hpp)）。facade が受ける観測は Actor の frame stack を通す前の生の観測で、done の lane では終端観測が入る（[924 §4.6](924_intrinsic_reward_novelty_10prd.md) の訂正候補のとおり、終端観測が届くのは facade だけ）。
- Network は `NetworkBuilder::BuildNetwork`（[nn.hpp:390](../../core/anet-core/include/anet/nn.hpp)）が `NetworkConfig` と観測の `TensorSpecMap` から組む。`Network::CopyTo`・`Clone`・`CreateForwardExecutor` は :214-220。観測の正規化部品は `ObservationNormalizer`（[scaler.hpp:19](../../core/anet-core/include/anet/scaler.hpp)、Factory は :64-67）。名前付き seed は `SeedMaker::MakeNamedSeed`（[random.hpp:100](../../core/anet-core/include/anet/random.hpp)）。
- checkpoint は header の識別子 `DefaultDQNAgent`（第 1 版）／`DefaultDQNAgent-2`（skip 有効）で形を区別し（[default_dqn_agent.cpp:29-30・:305](../../core/anet-core/src/default_dqn_agent.cpp)）、読込は `LoadNetwork`（:334）。識別子以外は ERROR（:355-358）、skip 有効の checkpoint を無効で読むと ERROR（:359）、skip 無効の checkpoint を有効で読むと skip を新規初期化して WARN 1 回（:368-370）。reward scaler と obs_norm の統計は保存していない（:314-315 の `@todo`）。
- 設定の接頭辞は `DefaultDQNAgent`（[default_dqn_agent.hpp:102](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。節は `learner.`・`actor.[k].`・`reward_scaler.`・`obs_norm.`・`skip.` など。schedule を書ける値は `ProfiledValue`（[schedule.hpp:354](../../core/anet-core/include/anet/schedule.hpp)）で、`ANET_READ_CONFIG` から自動で読む。
- facade の統計をメトリクスへ出す経路は `$agent <key> @learn`（例: [metrics_scalar.txt:152](../../apps/runner/config/metrics_scalar.txt) の `reward_scaler.scale`）。腕ごとの tag は `run.@skip : metrics.scalar.[50_agent_skip/...] = ...` の形で腕のプロファイルにだけ足す（[Atari.txt:280](../../apps/runner/config/Atari.txt)）。
- 無効時の同一性の受入は PRD098 の段 0 の手順（[testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md) の「段0・smoke」、`run_validation.py` の before／after／compare）。
- 毎 step の経路に別スレッドのロック待ち・同期の転送を足さない規約は [impl.md:321-326](../quality/impl.md)。本 PRD の採点はロックの中の学習スレッドで行い、新しいロックは足さない。転送は、観測を RND の Network の device へ送る H2D と誤差を戻す D2H が step ごとに加わる（§5.3）。

## 3. 用語（暫定）

[CONTEXT.md](../../CONTEXT.md) に無い概念で、既存の語（報酬スケーラ、spatial exploration）では言い換えられない。確定は 924 §8 の 20 の裁定に合わせ、本 PRD では次の語を使う。

| 語 | 意味 |
| --- | --- |
| 内的報酬 | Env の外的報酬とは別に、Agent が探索のために算出して学習報酬へ足す値 |
| 一生の新しさ | Run の全履歴に照らした観測の新しさ。エピソードごとに戻らない（NGU の lifelong novelty） |
| 新しさの倍率 α | 一生の新しさを [1, L] に収めた値。1 が「見慣れた」。NGU では エピソード内ボーナスに掛ける倍率、本 PRD では α−1 を内的報酬にする |

## 4. 確定事項（暫定の既定）

各行は暫定の既定である。理由が他の決定や事実に依存するときは名指す。

| # | 決定 | 理由 |
| --- | --- | --- |
| D1 | **採点点は facade。** `DefaultDQNAgent::UpdateFromBatch` が、reward scaler を通した後・内側の Learner へ渡す前に、`next_state` の観測を採点する | 遷移の 4 つ組と終端観測が揃う唯一の場所（§2）。Replay・Actor の契約を変えずに済む。ロックの中の経路に forward が乗る費用は §6.3 で測る |
| D2 | **運び方は報酬へ畳む（仮）。** 学習報酬 = スケール済み外的報酬 ＋ β × (α − 1)。done の lane の遷移は内的報酬 0（truncated は採点する）。β は `intrinsic.beta`（ProfiledValue、schedule 可） | 外的・内的を別に保存する契約は 924 問い 4。終端遷移に内的報酬を付けると終端そのものが報酬になるので 0 にする（924 §5.5 の 2 と同じ扱い）。D1 に依存 |
| D3 | **公開ヘッダ（`anet::rl::intrinsic`）に置くのは一生の新しさの口と生成の口（Builder）だけ。RND の本体は専用の非公開ヘッダと cpp に閉じ、NetworkModel の外で自分の Network・optimizer・統計を所有する。** DQN は RND の型も `intrinsic.rnd.*` の葉も見ず、`intrinsic.enabled`・`intrinsic.beta` を読んで Builder に設定の接頭辞を渡す | 924 の最終形でも同じ口を使う。RND の本体は DQN と独立に試験・差し替えでき、2 つ目の方式（RECODE 型など）を足すときも DQN 側は変わらない。[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) は「primary の出力を入力に取り、同じ target 同期を持つ補助 Network」を NetworkModel に置く判断で、RND は観測を入力に取り target 同期を持たないので適用範囲の外。矛盾ではないが、適用範囲の注記は 924 の再グリルで足す（§8 の 9） |
| D4 | **内的報酬は NGU の倍率の形で作る。** e_t = ‖predictor(x_t) − target(x_t)‖²（次元方向の平均）、α_t = clip(1 + (e_t − μ_e)/σ_e, 1, L)、内的報酬 = α_t − 1。μ_e・σ_e は一生の累積（Welford）で、採点の後に更新する | 924 の RND の倍率と同じ式・同じ統計なので、エピソード内ボーナスが入ったときに部品を変えない。RND 原典の e/σ（平均を引かない）は §8 の 1 の比較候補。L は `intrinsic.rnd.multiplier_max` |
| D5 | **入力は facade に届く生の観測（最新 1 フレーム）で、RND 専用の正規化を通す。** `ObservationNormalizer` を別インスタンスで作り、移動平均・標準偏差で白色化して clip する。採点は更新前の統計で行い、採点の後に更新する | RND 原典 §2.4（固定の target は尺度に合わせられないので正規化が必須、最後の 1 フレーム、Algorithm 1 の順）。既存部品の再利用（AGENTS.md の共通処理を先に探す）。Q 入力用の `obs_norm`（Atari は pass_through）とは独立 |
| D6 | **warm-up。** 採点した観測の件数が `intrinsic.rnd.warmup_steps` に達するまでは統計だけ更新し α=1（内的報酬 0）を返し、`Learn` は何もしない。件数は intake で呼ぶなら exp step と同じで、既定 12,800 は N=128 の採点 100 回と同じ長さ | RND 原典は乱数行動で観測統計を作ってから学習する。anet では方策をそのまま動かし、ボーナスだけ止める。採点の回数で数えると warm-up の長さが lane 数に比例する（§8.1） |
| D7 | **predictor の学習は on-policy で、刻みと件数は lane 数に依らない。** facade は intake ごとに credit を N × `intrinsic.learn_ratio` ÷ k だけ貯め（k = min(`intrinsic.learn_batch_size`, N)）、credit が 1 以上の回数だけ、その intake の観測から k lane を重複なく選んで `Learn` を呼ぶ。端数は次の intake へ繰り越し、1 intake の step 数は ⌊N ÷ k⌋ を上限にする。`Learn` は渡された観測すべてで MSE を 1 回 backward し optimizer を 1 回進め、進めたかを返す（warm-up 中は進めず false）。credit は結果に依らず消費する。母集団は intake の観測（Push 側。§8 の 12、924 問い 13）。optimizer は AdamW をモジュールが私有し、`intrinsic.rnd.learning_rate`・`.weight_decay` で設定する | exp step あたりの step 数（`learn_ratio` ÷ `learn_batch_size`）と 1 step の件数が、N・RR・replay batch のどれにも依らない（§8.1）。Learner の `replay_ratio`・`replay_batch_size`（U = N × RR ÷ B）と同じ形。RND 原典の間引き（§3.4、「内的報酬が消える速さ」を揃える）は N=128 で割合 0.25 の手動の補正で、`learn_ratio` 0.25 が同じ量。Replay からの抽出を要らなくし（924 問い 8 を先取りしない）、Learner の optimizer に group を足すと役割表・checkpoint・[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) へ波及する |
| D8 | **target と predictor は同じ構造で、観測仕様と `intrinsic.rnd.net.*` から組む。** 出力は `intrinsic.rnd.output_dim` 次元の Linear。target は乱数初期化のあと固定し、predictor から同期しない。重みは共有しない | NGU 付録 H.2（同形、出力 128）。predictor を深くする RND 原典の形は §8 の 2。構造は Body カタログから選ぶので、ベクトル観測（MLP）と画像観測（CNN）を同じ仕組みで扱う（924 のゴール） |
| D9 | **無効化は `intrinsic.enabled ?= false` の葉 1 つ。** 無効のときは Network・optimizer・統計・seed のどれも作らず、採点も forward も無い | 有効化の口を 1 つに保つ（924 問い 9 の推奨）。無効時の同一性（§6.1）。RND しか無い間は方式の列挙キーを作らない（impl.md「名前」5。924 で 2 つ目が入るときに足す） |
| D10 | **評価専用 Run（`learner.enabled=false`）では採点も更新も行わず、`intrinsic.enabled=true` なら 1 回 WARN** | 学習が無く報酬は使われない。意図された休止はエラーにしない（AGENTS.md Fail-Fast の表） |
| D11 | **メトリクスは群 56（`56_agent_in`）、`$agent intrinsic.<key> @learn` で読む。tag は腕のプロファイルにだけ足す** | 群 56 は未使用（§2）。無効時に `metrics.scalar.defs` を変えないため腕にだけ足す。tag は群名と `/` を含めて 40 文字以内 |
| D12 | **checkpoint は `intrinsic` 節を learner の後に足し、header の識別子を `<今の識別子>;intrinsic` にする。** 節には target・predictor・optimizer・観測の正規化の統計・誤差の統計・採点した観測の件数と、facade の credit の端数・lane 抽選の乱数状態を入れる。有効で節が無い checkpoint は ERROR、無効で節がある checkpoint は ERROR、無効で節が無い checkpoint は今のまま（バイト同一） | 固定 target を作り直すと predictor の教師が変わる（924 §4.9）。統計を保存しないと再開後の α の尺度が変わる。旧 checkpoint の有効読みは 924 問い 21 の推奨 (a)（互換分岐を残さない）。skip の先例（WARN 1 回）は採らない |
| D13 | **腕は `run.@rnd`。** Atari・GridMaze・LunarLander に `intrinsic.enabled = true`、`net`、tag の宣言を置く。既定値は `agent.txt` の `@baseline` に `?=` で置く | 機能ごとの @ プロファイルは作らず、既存の `run.@skip` と同じ形（§2）。GridMaze・LunarLander は害の確認と到達率の測定に使う（§7） |
| D14 | **SHARED scope との併用は制限しない** | RND は lane 別の状態を持たない。924 のエピソード内の記憶で初めて要る制約 |

## 5. 公開契約

### 5.1 一生の新しさの口（インタフェース）

名前空間 `anet::rl::intrinsic`、公開ヘッダ `include/anet/intrinsic.hpp`。anet-core の公開 API（`NetworkBuilder`・`Network`・`ObservationNormalizer`・`SeedMaker`・`Serializable`）だけに依存し、DQN の型を参照しない。

| 操作 | 受け取るもの | 返すもの | 副作用 |
| --- | --- | --- | --- |
| `Estimate` | lane ごとの観測（[B, …]。device は任意） | 新しさの倍率 α [B]（[1, L]、有限。入力と同じ device） | 誤差の統計と観測の正規化の統計を、採点の後に更新する。warm-up 中は統計だけ更新して 1 を返す。次の呼び出しへ状態を持ち越さない |
| `Learn` | 学習に使う観測（[B, …]。device は任意。`Estimate` に渡したものと同じでなくてよい） | 学習を進めたか（bool。warm-up 中は進めず false） | 渡された観測すべてで predictor の補助学習を 1 step 進める（抽選や刻みは持たない。統計は読むだけで更新しない）。warm-up 中は何もしない。target は変えない。`Estimate` とは独立で、順序も回数も縛らない |
| `Diagnostics` | なし | 直近の採点と更新の診断値（§5.5 の key） | なし |
| `Save`／`Load` | archive | — | `Serializable`。§5.6 の節の中身 |

生成の口（Builder）は同じ公開ヘッダに置く。

| 操作 | 受け取るもの | 返すもの | 副作用 |
| --- | --- | --- | --- |
| `LifelongNoveltyEstimatorFactory::Create` | 設定（`ConfigData`）と設定の接頭辞（例: `DefaultDQNAgent.intrinsic.rnd`）、観測の仕様（`TensorSpecMap`）、device、Agent の `SeedMaker` | 口の実体（所有権つき） | 接頭辞の下の葉を自分で読んで検証し（§5.4）、反していれば ERROR。Network の構築と seed の派生 |

DQN が見るのは口と Builder だけで、RND の型・設定 DTO・葉の名前は非公開ヘッダ（`src/intrinsic_rnd.hpp`）に閉じる。テストは非公開ヘッダを含めて RND を直接試験する。

本 PRD の呼び方は、1 intake につき facade が観測で `Estimate` を呼び、α から内的報酬を作って終端のマスクと β を掛け（D2）、credit（D7）が許す回数だけ、同じ intake の観測から選んだ lane で `Learn` を呼ぶ。口は α までを返し、β・終端・刻み・抽選を知らない。2 つの操作は独立で、どちらをいつ呼んでもよく、Save/Load は呼び出しの途中を除きどの時点でもよい。device は入力に従い、計算の device は Builder に渡す。CPU を前提にしない。NGU（924）でも `Estimate` は行動時、`Learn` は Push 側（intake）の刻みのままとし、口は変えない。母集団を Replay に変えるか、Push 側を消すかは §8 の 12・13（経緯は §8.1）。924 でエピソード内の新しさが入るときは、同じ α をエピソード内ボーナスの倍率として使う。日本語との対応は、採点＝`Estimate`、補助学習＝`Learn`。

RND 以外の実装（RECODE 型など）が入るときも、この 4 つの操作で足りる見込みである。足りなくなったときは 924 の再グリルで口を変える。

### 5.2 RND の実装の契約

- target と predictor は観測仕様と `intrinsic.rnd.net.*` から組んだ同じ構造の Network で、末尾に `output_dim` 次元の Linear を持つ。target は seed `intrinsic.target`、predictor は seed `intrinsic.predictor` で初期化する（`SeedMaker::MakeNamedSeed`）。seed が同じなら構造と初期値が再現する。
- 誤差 e_t は出力の二乗誤差の次元方向の平均。μ_e・σ_e は一生の累積（Welford）。σ_e が 0 のときは α=1。
- 観測の正規化は RND 専用の `ObservationNormalizer`（`intrinsic.rnd.obs_norm.*`）で、採点は更新前の統計、更新は採点の後。
- 補助学習は渡された観測すべてで MSE を 1 回 backward して optimizer を 1 回進め、進めたかを bool で返す（warm-up 中は進めず false）。lane の抽選と刻みは facade（§5.3）が持ち、RND は乱数を持たない。
- 乱数は Agent の `SeedMaker` から名前付きで派生し（lane の抽選は facade の `intrinsic.learn`）、既存の名前（`replaybuffer`・`learner`・`network`・`plasticity_probe` など）の派生と消費列を変えない。

### 5.3 facade の接続

`DefaultDQNAgent::UpdateFromBatch` の中で、`reward_scaler_->Scale` の後・内側の Learner へ渡す `BatchExperience` を組む前に、次を行う（有効のときだけ）。

1. `alpha = lifelong->Estimate(next_state.obs)`。
2. 内的報酬 = β × (alpha − 1)。`next_state.done` の lane は 0。
3. 学習報酬 = スケール済み外的報酬 ＋ 内的報酬。
4. credit += N × `learn_ratio` ÷ k（k = min(`learn_batch_size`, N)。N < `learn_batch_size` なら構築時に WARN 1 回）。credit ≥ 1 の間、`next_state.obs` から k lane を重複なく選んで `lifelong->Learn(選んだ観測)` を呼び、credit −= 1（warm-up 中で false が返っても消費し、繰り越さない）。true が返った回数を `learn_count` に数える。1 intake の step 数は ⌊N ÷ k⌋ を上限にし、超えた credit は次の intake へ繰り越す。
5. 診断値を `GetScalar` の `intrinsic.*` で読めるよう保持する。

追加する処理は 1 intake あたり、`Estimate` で forward 2 回（target・predictor、B 観測）と統計の更新、`Learn` で平均 B × `learn_ratio` ÷ k 回の forward 2 回と backward 1 回（各 k 観測。既定の 0.25・32 と N=128 なら intake ごとに 1 回・32 観測）、転送は観測の H2D（`Estimate` で B、`Learn` で k ずつ）と誤差の D2H、診断値の CPU 化。既存のロックの中で行い、新しいロック・スレッドを足さない。`Learn` の forward の重複は、`learn_ratio` が 0.5 以下なら backward が k 観測だけになる分で相殺される見積もり。単価は未測定で §6.3 で測る。

### 5.4 設定

接頭辞は `DefaultDQNAgent.`。既定値は `agent.txt` の `@baseline` に `?=` で置く。

```text
DefaultDQNAgent.@baseline : intrinsic.enabled ?= false
DefaultDQNAgent.@baseline : intrinsic.beta ?= 0.1                 # ProfiledValue。内的報酬の重み
DefaultDQNAgent.@baseline : intrinsic.learn_ratio ?= 0.25         # exp step あたりの補助学習の件数（Learner の replay_ratio と同じ形）
DefaultDQNAgent.@baseline : intrinsic.learn_batch_size ?= 32      # 補助学習 1 step の件数
DefaultDQNAgent.@baseline : intrinsic.rnd.output_dim ?= 128       # target・predictor の出力次元
DefaultDQNAgent.@baseline : intrinsic.rnd.multiplier_max ?= 5     # L
DefaultDQNAgent.@baseline : intrinsic.rnd.warmup_steps ?= 12,800  # 採点した観測の件数（intake なら exp step）。N=128 の 100 intake と同じ
DefaultDQNAgent.@baseline : intrinsic.rnd.learning_rate ?= 1e-4
DefaultDQNAgent.@baseline : intrinsic.rnd.weight_decay ?= 0
DefaultDQNAgent.@baseline : intrinsic.rnd.obs_norm.clip_range ?= 5   # ObservationNormalizer の既存の葉
run.@rnd : DefaultDQNAgent.intrinsic.enabled = true
run.@rnd : DefaultDQNAgent.intrinsic.rnd.net.body.$ = net.body.@MLP2      # 例（GridMaze）。Atari は画像用の Body
```

| キー | 型・値域 | 反したとき |
| --- | --- | --- |
| `intrinsic.enabled` | bool | 型違いは ERROR |
| `intrinsic.beta` | ProfiledValue、全端点が有限で ≥ 0 | ERROR（キー・値・期待を含む） |
| `intrinsic.learn_ratio` | 0 < 値 ≤ 1 | ERROR |
| `intrinsic.learn_batch_size` | 整数 ≥ 1。N より大きいときは N 件で学習し WARN 1 回 | ERROR（値域） |
| `intrinsic.rnd.net.*` | 既存の `NetworkConfig`（Body。Head は持たない） | 既存の NN の構築時検証（未知参照・rank・shape） |
| `intrinsic.rnd.output_dim` | 整数 ≥ 1 | ERROR |
| `intrinsic.rnd.multiplier_max` | 有限 > 1 | ERROR |
| `intrinsic.rnd.warmup_steps` | 整数 ≥ 0 | ERROR |
| `intrinsic.rnd.learning_rate` | 有限 > 0 | ERROR |
| `intrinsic.rnd.weight_decay` | 有限 ≥ 0 | ERROR |
| `intrinsic.rnd.obs_norm.*` | 既存の `ObservationNormalizerConfig` | 既存の検証 |

- `intrinsic.enabled`・`intrinsic.beta`・`intrinsic.learn_ratio`・`intrinsic.learn_batch_size` は DQN が読む。`intrinsic.rnd.*` は Builder（§5.1）が接頭辞を受けて読み、検証する。DQN 側に RND の葉の読み込みを書かない。
- `intrinsic.enabled=false` のときは `intrinsic.*` の他の葉を読まず、検証もしない（未指定のまま無効にできる）。
- `intrinsic.enabled=true` で `intrinsic.rnd.net` が無いときは ERROR。
- `learner.enabled=false` かつ `intrinsic.enabled=true` は ERROR にせず WARN 1 回（D10）。
- 値域の検証は構築時（設定読み込み）に行う。観測の形・device の不一致は NN の既存の初期化検証に任せる。
- 0 や負に別の意味を持たせない（0 は 0 として検証に掛かる）。

### 5.5 メトリクス

`$agent intrinsic.<key> @learn` で読む。群は `56_agent_in`。

| tag | key | 意味 |
| --- | --- | --- |
| `56_agent_in/01_bonus_mean` | `intrinsic.bonus_mean` | 内的報酬（β 倍後）の lane 平均 |
| `56_agent_in/02_bonus_max` | `intrinsic.bonus_max` | 同、lane 最大 |
| `56_agent_in/03_bonus_share` | `intrinsic.bonus_share` | Σ内的報酬 ÷ (Σ\|外的報酬\| ＋ Σ内的報酬)。両方 0 なら NaN |
| `56_agent_in/11_error_mean` | `intrinsic.error_mean` | μ_e |
| `56_agent_in/12_error_std` | `intrinsic.error_std` | σ_e |
| `56_agent_in/13_error_step` | `intrinsic.error_step` | この step の e_t の lane 平均 |
| `56_agent_in/21_clip_ratio` | `intrinsic.clip_ratio` | α が L に当たった lane の割合 |
| `56_agent_in/31_predictor_loss` | `intrinsic.predictor_loss` | 補助学習の直近の step の loss（まだ 1 回も進めていなければ NaN） |
| `56_agent_in/32_learn_count` | `intrinsic.learn_count` | 補助学習の累計 step 数（`Learn` が true を返した回数）。exp step に対する傾きが `learn_ratio` ÷ `learn_batch_size` になる（N 不変の確認用） |

warm-up 中は bonus 系が 0、error 系は値を持つ。無効のときは key が存在せず（`std::nullopt`）、腕に tag を足さないので defs は変わらない。

### 5.6 checkpoint

- header の識別子: 有効なら `DefaultDQNAgent;intrinsic` または `DefaultDQNAgent-2;intrinsic`。無効なら今のまま。
- `intrinsic` 節（learner の後）: RND の本体（書式の識別子、target の重み、predictor の重み、optimizer の状態、観測の正規化の統計、μ_e・σ_e と件数、採点した観測の件数）の後に、facade の credit の端数と lane 抽選の乱数状態。RND の本体を先に置くのは、形の違う checkpoint を RND の書式の検査で ERROR にするため。
- 読込の判定（上から順に当てる）: (1) 識別子が既知の形でない → ERROR。(2) 設定が有効で節が無い → ERROR。(3) 設定が無効で節がある → ERROR。(4) それ以外 → 節があれば復元し、無ければ今の読込のまま。
- optimizer の options は [ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md) と同じく現在の設定で置き換え、moment と step を引き継ぐ。

### 5.7 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
| --- | --- | --- | --- |
| `LifelongNoveltyEstimator` | 新規（公開の口） | §5.1 の 4 操作。方式に依らない一生の新しさの入口 | RL 共通基盤の内的報酬（`anet::rl::intrinsic`、`include/anet/intrinsic.hpp`） |
| `LifelongNoveltyEstimatorFactory` | 新規（公開の生成の口） | 設定の接頭辞・観測の仕様・device・seed から口の実体を作る。葉の読み込みと検証 | 同上（`include/anet/intrinsic.hpp`、実装は `src/intrinsic_rnd.cpp`） |
| `RndNoveltyEstimator` | 新規（非公開） | §5.2 の RND。target・predictor・optimizer・正規化・誤差統計を所有し、保存する | 同上（`src/intrinsic_rnd.hpp`・`src/intrinsic_rnd.cpp`。DQN からは見えない） |
| `RndNoveltyEstimatorConfig` | 新規（非公開） | `intrinsic.rnd.*` の DTO と値域の検証 | 同上（`src/intrinsic_rnd.hpp`） |
| `DefaultDQNAgent` | 変更 | 有効時の生成、採点・畳み込み・補助学習の credit と lane の抽選（§5.3）、`GetScalar` の `intrinsic.*`、checkpoint の節 | DefaultDQNAgent |
| `DefaultDQNAgentConfig` | 変更 | `intrinsic.*` の節（`enabled`・`beta`・`learn_ratio`・`learn_batch_size`・`rnd`） | DefaultDQNAgent |

## 6. 受入

### 6.1 無効時の同一性

- PRD098 の段 0 の手順（[testdata/prd098/README.md](../../core/anet-core/testdata/prd098/README.md)）で、着手前に保存した実行体と新しい実行体を、Atari・DropMerge・GridMaze・LunarLander の既定（`intrinsic.enabled=false`）で動かし、全 scalar 点の checksum と `metrics.scalar.defs` が一致する。
- 有効の checkpoint の読み書きが無効の checkpoint のバイト配置を変えない（無効で保存したファイルが着手前と同一）。

### 6.2 正しさ（anet-core-test、タグ `[prd116]`）

| 観点 | 観測する結果 |
| --- | --- |
| target の固定 | 同じ入力で `Learn` を 100 回進めても target の出力と重みが変わらない |
| predictor の学習 | 同じ入力で `Learn` を進めると predictor の loss が単調に近く下がり、e_t が下がる |
| α の値域 | 任意の入力で α が有限かつ [1, L]。σ_e=0 のとき α=1 |
| warm-up | 採点した観測の件数が `warmup_steps` に達するまでの `Estimate` が全 lane で 1 を返し（N を変えても同じ exp step 数で終わる）、μ_e・σ_e・正規化の統計は更新されている。その間の `Learn` は何もしない |
| 統計の順 | 採点が更新前の統計で行われる（同じ入力を 2 回渡した 1 回目と 2 回目の α が、更新順どおりの値になる） |
| 操作の独立 | `Learn` に `Estimate` と別の観測を渡すと、その観測の loss で学習する。`Estimate` を 2 回続けても、`Learn` を単独で呼んでも、状態の取り残しが無い |
| device | 入力を CUDA にすると α も CUDA で返り、CPU 入力の結果と一致する（CUDA が無い環境では skip） |
| Save の時点 | `Estimate` と `Learn` の間に Save/Load を挟んでも、以後の α と loss が挟まない場合と一致する |
| 刻みの不変 | 同じ exp step 数（N × intake 数）を N=2 と N=4 で流すと、`learn_count` と `Learn` に渡した観測の総件数が一致する。1 intake の step 数が ⌊N ÷ k⌋ を超えず、credit の端数が次の intake へ繰り越される |
| 抽選 | 同じ intake の中で同じ lane を 2 度渡さない。`learn_batch_size` > N のとき N 件で学習し、WARN が 1 回出る |
| 終端のマスク | `next_state.done` の lane の学習報酬がスケール済み外的報酬と一致し、truncated の lane には内的報酬が付く |
| 畳み込みの位置 | 内側の Learner が受ける報酬 = reward scaler の出力 ＋ β × (α − 1)（偽の Learner で受け取って比べる） |
| 保存・復元 | 保存→復元の前後で、同じ入力の α と、μ_e・σ_e・正規化の統計・採点した観測の件数・credit の端数・抽選の列が一致する |
| 読込の判定 | §5.6 の (1)〜(3) がそれぞれ ERROR になり、メッセージに識別子と設定の状態を含む |
| 設定の検証 | §5.4 の表の各行で、反した値が ERROR になり、キー・値・期待が含まれる。`enabled=false` では他の葉を読まない |
| seed の再現 | 同じ seed で 2 回構築した target の出力が一致し、既存の名前付き seed の派生と消費列が変わらない（無効時の同一性で確かめる） |
| メトリクス | 有効時に §5.5 の key が `GetScalar` で読め、無効時は `std::nullopt` |

### 6.3 性能

- 有効・無効を ABBA で交互に 3 巡以上回し、実時間の中央値と Tracy の内訳（target forward・predictor forward・backward・統計・転送）を記録する。環境は Atari（B=64）と GridMaze。
- 合否の閾値は置かない。数値は基準 Run の事前測定と本測定を見て 924 問い 20 で置く。本 PRD では内訳と実時間を記録に残すことを受入にする。

### 6.4 腕の smoke

- GridMaze で `run.@rnd` を短く回し、`56_agent_in/01_bonus_mean` が warm-up 後に正で、以後減少傾向を示す（同じ盤面なので慣れる）。`56_agent_in/03_bonus_share` が有限。
- Atari（KFM）で `run.@rnd` を 100k step 回し、落ちない・NaN が出ない・`56_agent_in/21_clip_ratio` が 1 に張り付かない。

## 7. 成功の測り方

効果の判定は [924 §9.1](924_intrinsic_reward_novelty_10prd.md) に従い、本 PRD では機構の指標（§6.4）までを受入にする。924 の腕として回す Run の候補は次のとおりで、合否の数値は 924 問い 20・21 で置く。

| Run | 見るもの | 本 PRD での位置づけ |
| --- | --- | --- |
| GridMaze: 通っていないセルからの到達率 | RND だけで収束後も被覆が続くか。924 §2.7 の見立ては「続かない」 | 判断に効く。続くなら方策の家族の優先度が下がる |
| LunarLander: 既定との比較 | 害の確認（易しい env で成績を落とさないか） | 924 §9.1 の害の確認 |
| KFM 10M × 数 seed、16 seed の基準と比較 | 挑む窓の本数、撃破÷命中、費用 | 対照腕。924 §2.5 の見立てでは壁は越えない |

## 8. 未決表（924 の再グリルで決める）

| # | 事項 | 暫定の既定と候補 |
| --- | --- | --- |
| 1 | 内的報酬の式 | 既定は α−1（NGU の倍率の形）。候補: RND 原典の e_t／σ_e（平均を引かない、上限なし）。差は尺度と下限 |
| 2 | predictor の深さ | 既定は target と同形。候補: RND 原典のように predictor に Linear を 2 層足す |
| 3 | 誤差統計の窓 | 既定は一生の累積。候補: EMA（係数をキーに） |
| 4 | 補助学習の刻みと件数の既定値 | 既定は `learn_ratio` 0.25・`learn_batch_size` 32（N=128 の旧既定と同じ量。exp step あたり 1/128 回・32 件）。候補: Taiga らと同じ毎 exp step 1 回・1 件（`learn_ratio` 1・`learn_batch_size` 1。費用は N 倍の backward）、件数 64 で回数を半分。比較対象との差は §8.1 の 6。指標は内的報酬が消える速さ（§6.4） |
| 5 | warm-up の既定値 | 既定は採点した観測 12,800 件（N=128 の旧既定と同じ長さ）。候補: RND 原典の正規化の初期化の長さに揃える |
| 6 | 終端遷移の内的報酬 | 既定は 0。候補: 採点して付ける（終端観測は facade に届く） |
| 7 | β を掛ける位置 | 既定は reward scaler の後。候補: scaler の前（スケールに乗せる） |
| 8 | 評価での採点 | 本 PRD では無し（構造上届かない）。924 の採点点の問いで決める |
| 9 | ADR 0062 の適用範囲の注記と、RND の Network の所有・補助学習の置き場と刻み（§8.1）を記す新 ADR | 本 PRD では起こさない。924 で起こす |
| 10 | 用語の CONTEXT.md への追加 | §3 の 3 語。924 §8 の 20 と一緒に確定する |
| 11 | 仮の運び方（D2）を 924 の注釈経路へ置き換える時期 | 924 の Replay 契約の段が入るとき。置き換えは同じ変更で行い、畳む経路を残さない |
| 12 | 補助学習の母集団と置き場 | 既定は Push 側（facade が採点した intake の観測を exp step 基準の credit で渡す）。候補: Learner の step で Replay から抽出した観測を渡す（NGU の置き場）。2 つは等価でない（§8.1）。924 問い 13 と一緒に決める |
| 13 | NGU 対応後に Push 側（intake での補助学習）を残すか | 既定は 924 の移行時にクリーンブレークで消す（1 関心 1 機構。RND 単独の腕は NGU の機構でエピソード内の β を 0 にして作る）。候補: 即時に反応する一生の新しさが要るなら残す。§8.1 |

### 8.1 補助学習の置き場と刻みの経緯（2026-11-07）

2026-11-06 夜に口を独立させた（冒頭の記録）あと、11/07 未明の問答で決めた。発言と判断を分けて残す。

1. 事実の整理（Claude）: Push 側（facade の intake）の補助学習は RR と replay batch に元から依らない。lane 数 N には 3 か所で依っていた。1 step の件数（割合 × N）、exp step あたりの step 数（1 ÷ N）、warm-up の長さ（採点回数 × N）。RND 原典（§3.4）の間引きは、環境数を増やしたときに predictor の学習に使う経験を 32 環境ぶんに固定する手動の補正で、128 環境のとき割合 0.25。anet はこの 0.25 を割合のまま持ち込んでいた。
2. ユーザーの発言の順: (a) N は env ごとにほぼ固定で RR は頻繁に変えるので、RR に引きずられない形がよい。(b) RND は NGU への通過点で、分散学習も将来の候補なので、NGU 方式（Learner の step で Replay から学習する Sample 側）を強く意識したい。部品としての RND ができていれば NGU 対応は問題ない。残る論点は NGU 対応後に Push 側を潰すか。(c) NGU 完成までは特性研究として RND の評価を進めるので、比較対象と実績のある Push 側で進める。無くす可能性が高くても、N に依存しない形で実装する。
3. Push 側と Sample 側は等価ではない（Claude の整理）。頻度の単位（exp step／learn step）、母集団（今の方策の観測を多くて 1 回／Replay の混合を RR 回ほど）、反応の速さ（採点の直後／Replay から引かれるまで遅れる）、忘れ方（方策が行かなくなった領域を忘れる／buffer にある間は覚える）。分散処理との相性は直感と逆で、actor-learner 型は勾配計算を learner に集めるため Sample 側になる。NGU の置き場は構成の都合で、アルゴリズムの必然ではない（原典の付録 A: actor が埋め込み網で r^i_t を計算して Replay へ入れ、learner が抽出した系列の末尾 5 フレームで行動予測網と RND の predictor を学習する）。RND 原典は同期型で Push 側。anet は同期型で facade が両方を持つ。
4. 不変性と置き場は別（Claude の整理）。exp step 基準の credit を facade に置けば、Sample 側でも RR に不変にできる。RR に依るのは学習が起きる時刻の粒度だけ。
5. 判断（本改訂の暫定の既定）: 置き場は Push 側、刻みは facade の credit（Learner の U と同じ形）、`Learn` は渡された観測で 1 step、母集団は intake の観測。924 で変えるとすれば母集団の 1 か所だけで、credit と `Learn` は残る。NGU 対応後に Push 側を消すかは §8 の 13。
6. 比較対象の確認（2026-11-07。原典のコード [bonus_based_exploration](https://github.com/google-research/google-research/tree/master/bonus_based_exploration) の `intrinsic_motivation/intrinsic_rewards.py`・`base_intrinsic_agent.py`・`configs/rainbow_rnd.gin`）: Taiga らの Rainbow＋RND は Push 側。agent の `step`・`end_episode` が、行動した観測 1 件で `compute_intrinsic_reward` を呼び、その中で eval でなければ predictor の学習 op を 1 回走らせ（batch 1）、観測と誤差の移動統計を更新し、(誤差 − 平均) ÷ 標準偏差 × `reward_scale`（0.00005）を外的報酬に足して [−1, 1] に clip した値を Replay に入れる。Rainbow の `update_period` 4・batch 32 は Q の学習だけに効き、predictor は毎 agent step 更新する。採点する観測は行動した観測 x_t（anet と RND 原典は次観測）。exp step あたりの predictor の step 数と 1 step の件数で並べると、Taiga 1 回・1 件、anet の既定 1/128 回・32 件、RND 原典 約 1/1,000 回・約 1,000 件（rollout 128 × 128、4 epoch × 4 minibatch、割合 0.25 から計算）、NGU 約 1/13,000 回・320 件（learner 5 更新/秒 × 64 系列 × 末尾 5 フレーム、actor 256 × 260 step/秒から計算）。Adam の進みは lr × step 数が主なので、Taiga の predictor は anet の既定より 2 桁速く慣れる。KFM・Montezuma の結果を Taiga らと並べるときはこの差を添える。

## 9. 移行対象

新しい契約へ移すもの（同じ変更で更新する）。旧契約の互換層は作らない。

- コード: `core/anet-core/include/anet/intrinsic.hpp`（新規、口と Builder）、`src/intrinsic_rnd.hpp`・`src/intrinsic_rnd.cpp`（新規、RND の本体）、`default_dqn_agent.hpp`／`.cpp`（`enabled`・`beta` の読み込み、採点・畳み込み、checkpoint の節、GetScalar）、`CMakeLists.txt`。
- テスト: `core/anet-core/src/intrinsic_test.cpp`（新規、`[prd116]`）、既存の DefaultDQN の checkpoint テストは無効時の形を変えないので断定は変わらない（`DefaultDQNAgent` の header の断定を検索して確かめる）。
- 設定: `apps/runner/config/agent.txt`（`@baseline` の既定。2026-11-07 の改訂で `learn_ratio`・`learn_batch_size` を足し、`train_fraction` を消し、`warmup_steps` を件数へ）、`Atari.txt`・`GridMaze.txt`・`LunarLander.txt`（`run.@rnd` の腕と tag。`32_learn_count` を足す）。
- 検証資材: `core/anet-core/testdata/prd116/README.md`（段 0 の比較手順の参照と、§6.3・§6.4 の手順）。
- 設計文書: [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) の §6.2（前処理に内的報酬の採点を足す）、§7.2（設定グループに「内的報酬」）、§8.2（`intrinsic` 節と識別子）、§9.1（群 56）。2026-11-07 の改訂では §6.2 の RND の段落（採点のグラフの再利用・独立 RNG の lane 抽選・採点回数の warm-up は旧い記述）と §7.2 の葉を日英とも現在形に直す。
- 設計文書の新設: [212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md)（RND の概念・計算・設定・指標の現在形。2026-11-06 にユーザーが実装中に要望し、起票時の本節に書き漏れていた項目として追記した。英訳は別途。2026-11-07 の改訂では §3.4〜§3.6・§3.8・§4.2・§4.4・§4.5・§5.1・§5.2 を追随させる）。
- wiki: [intrinsic_novelty](../wiki/mechanisms/intrinsic_novelty.md) に本 PRD を出典として足す（起票時に実施）。

## 10. 924 との関係と、後で消すもの

- 本 PRD は 924 の部品（一生の新しさの口と RND）を先に作り、924 の比較腕「RND 単独」を成立させる。924 §8 の 7・9・18 の一部（RND の統計・倍率・保存）に暫定の既定を与える。
- 924 の再グリルで採点点が Actor になれば、`Estimate` を呼ぶ側が facade から Actor に移り、口はそのまま残る。Actor 側で統計を更新しない採点が要るなら、口に「更新なしの採点」を足す（924 で決める）。
- D2 の畳む経路は仮で、924 の Replay の注釈経路が入る変更で削除する。残すと「外的・内的を別に保存する」契約と二重になる。
- 補助学習の置き場は Push 側のまま 924 へ持ち越す。Sample 側（Learner の step で Replay から）へ変えるか、Push 側を消すかは 924 で決める（§8 の 12・13、§8.1）。credit と `Learn` の形は置き場に依らず残る。

## 11. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.2（構造）、§2.4（観測と報酬の正規化、Algorithm 1）、§3.4（学習データの間引き）。
- Badia ら、[Never Give Up](https://arxiv.org/abs/2002.06038)、ICLR 2020。§2 式 1（α の clip）、付録 H.2（RND の構造、出力 128）、付録 B.3（L=5）、付録 A（actor が r^i_t を計算して Replay へ入れ、learner が抽出した系列の末尾 5 フレームで predictor を学習する）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。易しいゲームでの害の確認の条件。公開コードは [bonus_based_exploration](https://github.com/google-research/google-research/tree/master/bonus_based_exploration)（predictor の学習の置き場は §8.1 の 6）。
- [924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md) §2.5・§4.3・§7.1・§8・§9.1・§10。
