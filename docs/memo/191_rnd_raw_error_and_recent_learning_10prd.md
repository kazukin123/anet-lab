# PRD191: RND の誤差で割らない内的報酬の式と、直近の観測から引く predictor の補助学習（Burda らの形へ寄せる） 暫定 PRD

> 2026-11-19 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §6 の未決表に置いた。§7〜§9 は §6 の暫定の既定を前提に書き、前提にした行の番号を名指す。行番号は起票時点の作業ツリーの位置である。
> 起点: 2026-11-19 の会話。開発者が、今の RND の実装を Burda らの論文（[arXiv 1810.12894](https://arxiv.org/abs/1810.12894)）と比べるよう求めた。AI が、設定で変えられる差と Run の規模の差を除いた実装の差を挙げ、開発者がそのうち 4 件を課題に起票させた（[cb9a0f](../management/issues/issue-20261119-cb9a0f.md)・[4976bf](../management/issues/issue-20261119-4976bf.md)・[027698](../management/issues/issue-20261119-027698.md)・[497fce](../management/issues/issue-20261119-497fce.md)）。開発者は、027698 と 4976bf を先にする見込みを示し、2 件を 1 本の PRD にまとめて暫定で起こすよう指示した。あわせて「優先度が高くなくても、ついでに実装だけまとめて入れることも考えておきたい」と述べた。どれも決定ではない。
> 題材: [issue-20261119-027698](../management/issues/issue-20261119-027698.md)（RND に生の予測誤差を返す内的報酬の式が無い）、[issue-20261119-4976bf](../management/issues/issue-20261119-4976bf.md)（predictor の補助学習が今の intake の観測だけから引く）。
> 関連: [116](116_rnd_lifelong_novelty_10prd.md)（RND の本体）、[123](123_rnd_intrinsic_reward_scale_10prd.md)（誤差の統計の取り方・warm-up・平均を引くか）、[121](121_extrinsic_intrinsic_value_decomposition_10prd.md)（価値の分解と、内的報酬用の報酬スケーラの `return_std`）、[130](130_rnd_estimate_statistics_perf_10prd.md)（採点の統計の性能と統計方式の整理）、[117](117_warn_policy_and_unread_config_10prd.md)（読まない設定の扱い）、[173](173_intrinsic_reward_novelty_10prd.md)（NGU 型の内的報酬）、[190](190_dqn_design_doc_alignment_10prd.md)（設計文書 200 の直し）、設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md)（RND）・[214](../design/214_value_decomposition.jp.md)（価値の分解）。

## 1. 目的と問題

### 1.1 問題

1. **内的報酬の式を、論文の形にできない。** 論文の内的報酬は、target と predictor の出力の差の二乗ノルムで、それを内的の収益の標準偏差の移動推定で割る。clip はせず、平均も引かない（§3.1、前提 P1・P2）。anet の whiten は、どの設定でも誤差を誤差の統計の標準偏差で割る（§5.1、P4）。内的報酬用の報酬スケーラの `return_std`（PRD121）を重ねると、標準偏差で 2 回割る形になり、誤差の統計の標準偏差は Run の中で変わるので、値は論文の値に比例しない（[027698](../management/issues/issue-20261119-027698.md)）。
2. **predictor の補助学習を、論文の学び方にできない。** 論文は、rollout ごとに集めた観測の 25% を残し、4 エポック、約 1,000 件ずつのミニバッチで学ぶ（§3.1、P3）。anet は、今の intake の観測だけから、各観測を高々 1 回、`learn_batch_size` 件ずつ引く。1 step の件数は lane の数を超えられず、`learn_ratio` は 1 以下である（§5.2、P6・P7）。既定（`learn_ratio` 0.25、32 件）では 1 経験あたり 0.25 回の勾配で、論文の約 1 回の 4 分の 1 である。`learn_ratio` を 1 にすれば回数はそろうが、同じ観測の再利用と、lane の数を超えるバッチは作れない（[4976bf](../management/issues/issue-20261119-4976bf.md)）。

### 1.2 ゴール

- **G1**: whiten で、誤差を誤差の統計で割らない式を選べる。内的報酬用の報酬スケーラの `return_std` と組めば、論文の内的報酬の形（生の誤差 ÷ 内的の収益の標準偏差）になる。
- **G2**: predictor の補助学習を、直近の観測を貯める置き場から引けるようにする。lane の数を超えるバッチと、同じ観測の再利用（1 経験あたり 1 回を超える学習）を選べる。
- **G3**: 既定は今と同じにする。新しい設定キーが既定のとき、同 seed・決定的モードの短い Run の metrics checksum が実装前の実行体と一致する。
- **G4**（未決表の 11 の暫定の既定で範囲に入れたもの）: RND の target と predictor の Network の構造図を、primary の Network と同じく記録する。

### 1.3 仮説・比べ方・終わり方

| | 仮説 | 比べ方 |
|---|---|---|
| A（内的報酬の式） | 論文の形の内的報酬（誤差で割らない式、`return_std`、内的の価値を非エピソードにし割引率を別に持つ）にすると、内的報酬が正になる経験の割合が上がり、Montezuma の Greedy で 2500 点のあとに Q が平らになる局面（PRD121 §2.1）が変わる | 同じ実行体で、今の既定の RND と論文の形の RND を、価値の分解を有効にした同じ構成で比べる |
| B（predictor の学習） | predictor の学習の量と 1 step の件数を論文にそろえると、慣れた状態の誤差が早く下がり、新しい状態との誤差の差が広がる | 今の intake から引く形（`learn_ratio` 0.25 と 1.0）と、直近の観測から引く形（`learn_ratio` 1.0）を比べる。intake から引く形の `learn_ratio` 1.0 を入れると、量の効果と、再利用・大きいバッチの効果を分けて読める |

群の組み立て（構成、β の値、seed の数、長さ）は本 PRD で決めない。Atari の枠で実験の段取りと一緒に決める（未決表の 13）。

**終わり方**: 効果が出なくても仕組みは残す。既定は今のままである（G3）。効果が出たとき、既定を変えるかは Atari の枠で別に決める。

### 1.4 非ゴール（退けた案と、本当の制約か）

| 案 | 扱い | 本当の制約か |
|---|---|---|
| 学習の前に一様乱数の行動で観測の統計を作る手順（[cb9a0f](../management/issues/issue-20261119-cb9a0f.md)） | 範囲外。Train Actor の ε の予定を、最初の N step だけ 1.0 にする設定で近い形を試せる見込み（`spatial.epsilon.first`・`last` は時間の予定を書ける。[200](../design/200_dqn_agents.jp.md) 249 行。試していない）。PRD123 の U2 で観測の正規化の式を開き直さないと決めた（P10）。作る条件: 設定で試した結果から、乱数の行動で統計を作る手順そのものが要ると分かったとき（実験の結果が要る） | 制約ではない |
| 非エピソードの内的の収益を、エピソードをまたいで n-step の和に足すこと（[497fce](../management/issues/issue-20261119-497fce.md)） | 範囲外。Replay の窓がエピソードをまたぐ契約の変更になる別の関心（PRD121 §1.4 と同じ）。作る条件: `non_episodic` の効果が分かり、終端の近くの数 step の差が効いていると分かったとき | 制約ではない |
| PPO と再帰（GRU など）の Network | 範囲外。学習器そのものの作りという別の関心。作る条件: 学習器の作りを変える PRD を起こすとき | 制約ではない |
| 既定を論文の形に変えること | 先送り。実験の結果が要る。作る条件: §1.3 の仮説 A・B の群で効果が出て、Atari の枠で既定を変えると決めたとき | 制約ではない |
| `variance_feedback`（Taiga らの再現のモード）の中身 | 変えない。PRD123 の D8 による | 制約ではない（別の関心として PRD123 で決めた） |
| RND の Head の出力次元の設定キーの置き場（[b02db6](../management/issues/issue-20261115-b02db6.md)） | 範囲外（未決表の 12 の暫定の既定）。Head の設定を `net.head.*` へ移すかを、PRD155 の §3.8・Q8 が決める別の関心。作る条件: PRD155 がその置き場を決めたとき | 制約ではない |
| Atari の Run プロファイルと bat の群 | 範囲外。Atari の枠で実験の段取りと一緒に決める別の関心（未決表の 13）。作る条件: Atari の枠で群を組むとき | 制約ではない |

### 1.5 機構とゴールの対応

| 機構 | ゴール | 寄与 |
|---|---|---|
| 誤差で割らない式（`error_norm.mode = none`。未決表の 1） | G1 | 誤差の統計で割らずに倍率を作る。`return_std` と組んで論文の形になる |
| 直近の観測の置き場から引く補助学習（`learn_observation_mode = recent`。未決表の 5） | G2 | 置き場から引くので、1 step の件数が lane の数に縛られず、同じ観測を何度も学べる |
| 置き場の容量の設定キー（未決表の 6） | G2 | 置き場が覚えている観測の件数を決める。再利用の度合いは容量と `learn_ratio` で決まる |
| 構造図の記録（未決表の 11） | G4 | RND の Network の形を、Run の `dot/` で確かめられる |

## 2. 前提

暫定の起票なので、確かめは行っていない（確かめの列は「対象外」）。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実（文書） | 論文は内的報酬を、内的の収益の標準偏差の移動推定で割って正規化する | [Burda ら](https://arxiv.org/abs/1810.12894) §2.4 | 対象外 |
| P2 | 事実（文書） | 論文の内的報酬は target と predictor の出力の差の二乗ノルムで、内的報酬の clip は無い | 同 付録 A.2 の Algorithm 1、表 2 | 対象外 |
| P3 | 事実（文書） | 論文の predictor は rollout ごとの batch を 4 エポック・4 ミニバッチで学び、経験の 25% を使う。128 env で 32 env のときと実効バッチをそろえるため、要素を確率 0.25 で残す | 同 付録 A.4 の表 5 と本文 | 対象外 |
| P4 | 事実（コード） | whiten の倍率は α = 1 ＋（e − 平均）÷ 標準偏差（平均を引かない設定では 1 ＋ e ÷ 標準偏差）。warm-up 中と統計が空のときは 1 | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 293〜301 行 | 対象外 |
| P5 | 事実（コード・文書） | 内的報酬用の報酬スケーラの `return_std` は lane ごとに R = γR ＋ r を持ち、R の標準偏差で r を割る。分解して内的の割引率を別にしたときは、その割引率で作る | [214](../design/214_value_decomposition.jp.md) §4.2、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 74〜78 行・347〜349 行 | 対象外 |
| P6 | 事実（コード） | 補助学習は、intake ごとに credit を「lane の数 × `learn_ratio` ÷ k」だけ貯め、同じ intake の lane を重複なく k 件ずつ渡す | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 139〜166 行 | 対象外 |
| P7 | 事実（コード） | `learn_batch_size` が lane の数を超えると構築時に止まる。`learn_ratio` は 0 より大きく 1 以下 | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 79〜83 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 172〜175 行 | 対象外 |
| P8 | 過去の判断 | 平均を引かない形を `multiplier_mode` の 3 つ目の値にする形は不採用。clip の有無を表すキーに別の軸が混ざるため | [PRD123](123_rnd_intrinsic_reward_scale_10prd.md) D7・§4.1、[321 のトピック別まとめ](../wiki/categories/321_agent_core.md) の選ばなかった形 | 対象外 |
| P9 | 過去の判断 | `learn_batch_size` の既定を lane の数に結び付ける形は不採用。lane の数より大きい指定は構築時に止める | [PRD117](117_warn_policy_and_unread_config_10prd.md) D11、[324 のトピック別まとめ](../wiki/categories/324_dqn.md) の選ばなかった形 | 対象外 |
| P10 | 過去の判断 | PRD123 の U2 で、観測の正規化の式と 2 ステージの warm-up を開き直さないと決めた | [PRD123](123_rnd_intrinsic_reward_scale_10prd.md) 冒頭の 2026-11-15 の改訂行 | 対象外 |
| P11 | 事実（測定） | Atari の主な構成で、補助学習 1 回は 4.4〜4.9 ms、1 intake の `DefaultDQNAgent::UpdateFromBatch` は 118.5 ms（うち RND 23.2 ms） | [PRD130](130_rnd_estimate_statistics_perf_10prd.md) §2.9・§2.11 | 対象外 |
| P12 | 守る線 | RND の既定や統計の取り方を変えるときは、RND 無効を含む構成で、同 seed・決定的モードの短い Run の metrics checksum を着手前の実行体と一致させる | [321 のトピック別まとめ](../wiki/categories/321_agent_core.md) の変えるときの条件 | 対象外 |
| P13 | 事実（コード） | Evaluation Trace の「標準化した値」は、統計を変えない口（`StandardizeWithoutUpdate`）が倍率と同じ式で作る | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 233〜239 行 | 対象外 |
| P14 | 事実（コード） | Network の構造図を記録するのは primary の Network と、skip が有効なときの補助 Network だけ | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 460〜472 行 | 対象外 |
| P15 | 事実（文書） | 共通の ReplayBuffer は Agent の archive へ serialize されず、checkpoint を読んでも新規構築の状態から始まる | [150](../design/150_replay_buffer.jp.md) §7.6 | 対象外 |

## 3. 背景

### 3.1 論文の内的報酬と predictor の学習

論文（Burda ら、ICLR 2019）は、PPO の上に RND を載せる（付録 A.1）。

- **内的報酬。** 次観測について、target と predictor の出力の差の二乗ノルムを内的報酬にする（Algorithm 1）。内的報酬は、内的の収益の標準偏差の移動推定で割る（§2.4）。内的報酬の clip は無い（表 2）。
- **観測の正規化。** target と predictor の入力は 1 フレームで、成分ごとに平均を引いて標準偏差で割り、[−5, 5] で切る（§2.4、表 4）。正規化の統計は、学習の前に一様乱数の行動で回した観測で作る（Algorithm 1）。
- **predictor の学習。** rollout（128 step × env）ごとに、集めた観測で観測の統計を更新してから、同じ batch で PPO と蒸留の損失を N_opt 回最適化する（Algorithm 1）。表 5 は、ミニバッチ 4、エポック 4、predictor に使う経験の割合 0.25 とする。付録 A.4 によれば、128 env で 32 env のときと実効バッチをそろえるため、要素を確率 0.25 で残す。
- **2 つの価値。** 内的の収益は非エピソード、外的はエピソードで、価値の Head を 2 つ持つ（§2.3）。表 5 の割引率は γ_E 0.999・γ_I 0.99、係数は外的 2・内的 1 である。

表 5 から計算すると、predictor は 1 経験あたり約 1 回の勾配を受け（残した 25% を 4 エポック）、Adam の 1 step は約 1,000 件になる（128 env × 128 step ÷ 4 ミニバッチ × 0.25）。

### 3.2 ほかの実装の事例

| 実装 | 内的報酬の式 | predictor の学習 | 確かめ |
|---|---|---|---|
| Burda らの公開実装 | 内的の収益の標準偏差で割る（論文の §2.4 と同じ） | rollout の batch を複数エポック | 未確認（論文が公開実装に従うと書くことによる） |
| Taiga ら（Rainbow＋RND） | 誤差の z を分散に相当する値で割り、0.00005 倍する | 行動した観測を agent step ごとに 1 件で採点・学習する | anet の `run.@rnd_taiga` が寄せた形（[PRD116](116_rnd_lifelong_novelty_10prd.md) §2.1） |
| CleanRL の PPO＋RND | 内的の収益の標準偏差で割る | rollout の batch を、要素を割合で残して複数エポック | 記憶による |
| DI-engine の RND の報酬モデル | 正規化の有無を選べる | 集めたデータで、収集 1 回ごとに複数回の更新 | 記憶による |

### 3.3 これまでの anet の判断

- 平均を引くかは、`multiplier_mode` の値ではなく別のキー `error_norm.use_centering` にした（PRD123 D7、P8）。clip の有無を表すキーに別の軸を混ぜないためである。
- `learn_batch_size` の既定を lane の数に結び付けない。lane の数より大きい指定は構築時に止める（PRD117 D11、P9）。
- 観測の正規化の式と 2 ステージの warm-up は開き直さない（PRD123 の U2、P10）。
- 内的報酬用の報酬スケーラの `return_std` と、内的の価値を非エピソードにして割引率を別に持つ形は、PRD121 で入った（P5）。論文の形に足りないのは、誤差で割らない式だけである（§1.1 の 1）。

## 4. 用語

新しい語は、補助学習のための観測の置き場（未決表の 5 で置く「直近の観測の置き場」）である。[CONTEXT.md](../../CONTEXT.md) の「補助学習」（主学習とは別に predictor を学ぶこと）の範囲の中の、学習に使う観測の出どころで、既存の語（Replay、intake）では言い換えられない。Replay は Q の学習の経験を持つ部品で、intake は 1 回に取り込む経験のまとまりである。用語集に足すかは未決表の 14 で決める。暫定の起票なので、本 PRD では `CONTEXT.md` を変えない。

## 5. 現状（2026-11-19 にコードで確認）

### 5.1 倍率の式

- whiten の採点は、誤差の統計の平均と標準偏差で倍率を作る（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 293〜301 行）。warm-up 中と、統計が空か標準偏差が 0 のときは 1 を返す。誤差の統計は warm-up 明けから更新し、観測の統計はそのあとに更新する（同 302〜305 行）。
- `multiplier_mode = clipped` なら倍率を [1, L] に切る（同 805 行）。
- Evaluation Trace の「標準化した値」は、統計を変えない口が同じ式で作る（同 233〜239 行）。
- `error_norm.*`（`average_mode`・`ema_steps`・`use_centering`）は whiten だけが使う。`variance_feedback` で既定と違う値を書くと、読まない設定として通知する（同 638〜642 行）。

### 5.2 補助学習

- `IntrinsicRewardCoordinator::Learn`（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 139〜166 行）は、intake の観測を選び、credit を「lane の数 × `learn_ratio` ÷ k」だけ貯める。credit が 1 以上なら、lane の順列を作り、重複なく k 件ずつ RND の `Learn` に渡す。warm-up で学習しなかった分も credit を消費し、明けたときに追いつき学習をしない。
- RND の `Learn` は、渡された観測を最新の観測の統計で正規化してから、predictor を 1 回更新する（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 856 行から）。
- `learn_batch_size` は lane の数以下、`learn_ratio` は 0 より大きく 1 以下（P7）。既定は 0.25 と 32（[agent.txt](../../apps/runner/config/agent.txt) 48〜49 行）。

### 5.3 内的報酬用の報酬スケーラ

内的報酬は、RND の倍率から 1 を引いて β を掛け、内的報酬用の報酬スケーラを通す。`return_std` を選ぶと、内的の収益の標準偏差で割る（P5）。

### 5.4 構造図

`DefaultDQNAgent::RecordNetworkStructure` は、primary の Network と、skip が有効なときの補助 Network の `structure`・`detail` を記録する。RND の target と predictor は記録しない（P14、[issue-20261115-ce0101](../management/issues/issue-20261115-ce0101.md)）。

### 5.5 性能

Atari の主な構成（128 env）では、既定の補助学習（`learn_ratio` 0.25、32 件）は 1 intake あたり 1 回で、1 回 4.4〜4.9 ms だった。1 intake の `UpdateFromBatch` は 118.5 ms だった（P11）。

## 6. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§7〜§9 はこれを前提に書いた。再グリルですべて開き直す。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | 誤差で割らない式の設定キー | `intrinsic.rnd.error_norm.mode`（`standardize`｜`none`）を足す。既定は `standardize`（今と同じ）。`none` では α = 1 ＋ e とする | 誤差の正規化の軸は `error_norm.*` が持つ（P8 の、clip の軸に別の軸を混ぜない判断による）。観測の正規化は whiten のまま使う（論文も観測は正規化する。§3.1） | `statistics_mode` の 3 つ目の値（観測の正規化と誤差の正規化の 2 軸が 1 つのキーに混ざる）。`multiplier_mode` の値（P8 で不採用の形と同じ混ざり方） |
| 2 | `none` のときの誤差の統計 | 今と同じく更新し続ける。診断（`11_error_mean`・`12_error_std`）と checkpoint の形は変えない | 統計を止めると診断が NaN になり、checkpoint の形も分かれる。更新するのは lane の数だけの誤差のスカラーで、費用は小さい | 統計を止める（診断を NaN にし、checkpoint の識別子を分ける） |
| 3 | `none` のときの warm-up | 今と同じく、warm-up 中は α = 1（bonus 0）。論文に寄せるときは `warmup_steps = 0` を書く | `warmup_steps` の意味をモードで変えない（[AGENTS.md](../../AGENTS.md) の設定値の扱い）。論文には bonus を 0 にする期間が無い（§3.1） | `none` のときは warm-up を読まない |
| 4 | `none` のときの Trace の「標準化した値」 | e（倍率から 1 を引いた、clip の前の値）を出す | 「倍率を作った値」という今の列の意味を保つ（P13） | NaN を出す |
| 5 | 直近の観測の置き場から引く補助学習 | `intrinsic.learn_observation_mode`（`intake`｜`recent`）を足す。既定は `intake`（今と同じ）。`recent` では、intake の観測（RND に渡す前の、正規化していない観測）を、容量で古いものから捨てる置き場に貯め、1 step ごとに置き場から k 件を、1 step の中では重複なく、step をまたいでは重複を許して一様に引く | 置き場から引けば、lane の数を超えるバッチと再利用の両方ができる（G2）。正規化していない観測を貯めるので、学習の時点の最新の観測の統計で正規化する今の作り（§5.2）を保てる | 論文と同じく、R intake を貯めてから E エポック学ぶ形（計算が R intake ごとに偏る）。Replay から引く形（PER の分布で引くことになり、観測の形も Q 側に縛られる） |
| 6 | 置き場の容量 | `intrinsic.learn_recent_capacity`（観測の件数、1 以上の整数）を足す。既定は 16,384。`learn_observation_mode = recent` のときだけ読む | 128 lane なら 128 intake 分で、論文の rollout（128 step × env）と同じ長さになる。Atari の 1 フレーム（84 × 84 画素、1 画素 1 バイト）なら約 116 MB | 既定を lane の数から決める（P9 の、既定を lane の数に結び付けない判断と同じ理由で避ける） |
| 7 | `recent` のときの `learn_ratio` と `learn_batch_size` の範囲 | `learn_ratio` は有限で 0 より大きい（1 を超えてよい）。`learn_batch_size` は 1 以上で容量以下（lane の数の検査は当てない）。`intake` のときは今と同じ | 1 を超える `learn_ratio` が再利用を表す。lane の数の検査は、intake から重複なく引くための制約である（P7） | `recent` でも `learn_ratio` の上限を置く |
| 8 | 置き場が k 件に満たないとき | その intake では学習しない。credit は今と同じく消費し、追いつき学習をしない | warm-up で学習しなかった分の扱い（§5.2）と同じにする | 置き場にある件数だけで学ぶ |
| 9 | 置き場の保存 | checkpoint に入れない。読み込んだあとは空から貯め直す。credit と抽選の乱数の状態は今と同じく保存する | 置き場は直近の観測の写しで、Run を続ければ容量 ÷ lane の数の intake で埋まり直す。共通の ReplayBuffer も checkpoint に入らない（P15） | 置き場も保存する（checkpoint が容量の分だけ大きくなる） |
| 10 | 置き場の device | RND に渡す観測と同じ device に置く | intake の観測を移さずに貯められ、抽選のあとの転送が要らない | 常に CPU に置き、引いたあとに送る |
| 11 | 構造図の記録（[ce0101](../management/issues/issue-20261115-ce0101.md)） | 範囲に入れる。RND が有効なとき、target と predictor の Network の `structure`・`detail` を、primary の Network と同じ時機に記録する | 開発者の「ついでに実装だけまとめて入れる」の発言による。同じ RND の部品に触り、構造図の記録は今の口（P14）に足すだけである | 範囲に入れない |
| 12 | Head の出力次元の設定キーの置き場（[b02db6](../management/issues/issue-20261115-b02db6.md)） | 範囲に入れない | 残っているのは、Head の設定を `net.head.*` へ移すかの判断で、PRD155 の §3.8・Q8 が決める。本 PRD で決めると、DQN の Head の設定の置き場と食い違うおそれがある | 範囲に入れて、RND の分だけ先に決める |
| 13 | 効果の確かめの置き場 | Atari.txt の Run プロファイルと bat の群は本 PRD の範囲に入れず、Atari の枠で実験の段取りと一緒に決める。本 PRD が足す設定は `agent.txt` の既定キーだけにする | PRD121 の D19 と同じ。Atari.txt の変更は Atari の枠で行う | 本 PRD で Run プロファイル（例 `run.@rnd_burda`）を足す |
| 14 | 名前と用語 | 設定キーと値の名前は §8.1 の暫定の名前。「直近の観測の置き場」を用語集に足すかは再グリルで決める | 暫定の起票なので `CONTEXT.md` を変えない（§4） | — |
| 15 | 設計文書 | RND の設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md) に、誤差で割らない式と、直近の観測から引く補助学習の節を足す。設定の表も直す | 今ある RND の仕組みの選択肢で、新しい仕組みではない | 2xx の設計文書を新設する |
| 16 | ADR | 起こさない | どちらも既定を変えない選択肢で、戻しやすい | 置き場の形（R intake とエポックの形と比べた判断）を ADR にする |
| 17 | 性能 | 受入に閾値を置かない。実装後に、Atari の実時間を `intake`（`learn_ratio` 0.25）と `recent`（`learn_ratio` 1.0）で交互に測って記録する | §7.2 の見積もりで、`recent` の `learn_ratio` 1.0 は 1 intake で最大 15 ms ほど増える見込み（`UpdateFromBatch` の約 13%）。既定は変えない | 実時間の予算を置く |

## 7. 仕様（暫定の既定を前提）

### 7.1 誤差で割らない式

前提: 未決表の 1〜4。

- `intrinsic.rnd.error_norm.mode = none` では、whiten の採点の倍率を α = 1 ＋ e とする。e は今と同じ誤差（出力の次元で平均した二乗誤差）である。
- warm-up 中は今と同じく α = 1 とする（未決表の 3）。
- `multiplier_mode = clipped` なら [1, L] に切り、`unclipped` なら切らない（今と同じ）。
- 誤差の統計と観測の統計は今と同じく更新する（未決表の 2）。
- Trace の「標準化した値」は e を出す（未決表の 4）。
- `error_norm.use_centering` は `none` では使わない。`standardize` では今と同じ。

論文の形に寄せる設定の例（コロン記法。Atari の枠で群を組むときの材料で、本 PRD では足さない。未決表の 13）:

```text
run.@rnd_burda : DefaultDQNAgent.intrinsic.rnd.error_norm.mode = none
run.@rnd_burda : DefaultDQNAgent.intrinsic.rnd.multiplier_mode = unclipped
run.@rnd_burda : DefaultDQNAgent.intrinsic.rnd.warmup_steps = 0
run.@rnd_burda : DefaultDQNAgent.intrinsic.use_done_mask = false
run.@rnd_burda : DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling = true
run.@rnd_burda : DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode = return_std
run.@rnd_burda : DefaultDQNAgent.intrinsic.value_mode = decomposed
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.discount_mode = separate
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.gamma = 0.99
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.terminal_mode = non_episodic
```

### 7.2 直近の観測の置き場から引く補助学習

前提: 未決表の 5〜10。

- `intrinsic.learn_observation_mode = recent` では、intake ごとに、RND に渡す観測（`intrinsic.observation` で選んだ観測）の全 lane を置き場に入れる。置き場は `learn_recent_capacity` 件を超えたら古いものから捨てる。
- credit の貯め方は今と同じ（「lane の数 × `learn_ratio` ÷ k」）。credit が 1 以上で、置き場が k 件以上あれば、1 step ごとに置き場から k 件を引いて RND の `Learn` に渡し、credit を 1 減らす。1 step の中では重複なく引き、step をまたいでは重複を許す。
- 置き場が k 件に満たなければ、その intake では学習せず、credit は消費する（未決表の 8）。
- 抽選の乱数は今の名前付き seed（`intrinsic.learn`）を使う。

例（コロン記法）:

```text
run.@rnd_recent : DefaultDQNAgent.intrinsic.learn_observation_mode = recent
run.@rnd_recent : DefaultDQNAgent.intrinsic.learn_recent_capacity = 16,384
run.@rnd_recent : DefaultDQNAgent.intrinsic.learn_ratio = 1.0
run.@rnd_recent : DefaultDQNAgent.intrinsic.learn_batch_size = 256
```

**頻度×単価の見積もり（未決表の 17）。** Atari の主な構成（128 lane）で、既定の補助学習は 1 intake に 1 回（32 件）、1 回 4.4〜4.9 ms である（P11）。`learn_ratio` 1.0 で k 256 なら、1 intake に 0.5 回（256 件）になる。1 回の時間が件数に比例すると見ると、1 intake あたり約 17〜20 ms で、今より 13〜15 ms ほど増える（`UpdateFromBatch` の 118.5 ms の 11〜13%）。GPU では件数を増やしても時間はふつう比例より緩く増えるので、これは上限に近い見込みである。

### 7.3 構造図

前提: 未決表の 11。

- RND が有効なとき、target と predictor の Network の `structure`・`detail` を、primary の Network と同じ時機に記録する。tag の名前は §8.1 の暫定の名前とする。

## 8. 実装契約（暫定の既定を前提）

### 8.1 設定と失敗の扱い

名前はすべて暫定（未決表の 14）。

| 設定キー | 型・値 | 読む条件 | 前提にした未決表の行 |
|---|---|---|---|
| `DefaultDQNAgent.intrinsic.rnd.error_norm.mode ?= standardize` | `standardize`｜`none` | `statistics_mode = whiten` のとき | 1 |
| `DefaultDQNAgent.intrinsic.learn_observation_mode ?= intake` | `intake`｜`recent` | 常に（`intrinsic.enabled = true` のとき） | 5 |
| `DefaultDQNAgent.intrinsic.learn_recent_capacity ?= 16,384` | 1 以上の整数（観測の件数） | `learn_observation_mode = recent` のとき | 6 |
| `DefaultDQNAgent.intrinsic.learn_ratio`（今のキー） | `intake`: 有限で 0 より大きく 1 以下。`recent`: 有限で 0 より大きい | 今と同じ | 7 |
| `DefaultDQNAgent.intrinsic.learn_batch_size`（今のキー） | 1 以上の整数。`intake`: 学習が有効なら lane の数以下。`recent`: 容量以下 | 今と同じ | 7 |

構造図の tag は `intrinsic.rnd.target.net.structure`・`intrinsic.rnd.target.net.detail`・`intrinsic.rnd.predictor.net.structure`・`intrinsic.rnd.predictor.net.detail` とする（暫定。未決表の 11・14）。

判定は、次の表を上から順に当て、最初に当たった行で扱う。どの組み合わせも最後の行までに必ず当たる。

| # | 条件 | 扱い |
|---|---|---|
| 1 | `error_norm.mode` か `learn_observation_mode` が列挙の外 | ERROR（キーと値と列挙を示す） |
| 2 | `learn_batch_size` が 1 未満 | ERROR（今と同じ） |
| 3 | `learn_observation_mode = intake` で、`learn_ratio` が有限でないか、0 以下か、1 を超える | ERROR（今と同じ） |
| 4 | `learn_observation_mode = intake` で、学習が有効で、`learn_batch_size` が lane の数を超える | ERROR（今と同じ） |
| 5 | `learn_observation_mode = recent` で、`learn_recent_capacity` が 1 未満 | ERROR（キーと値を示す） |
| 6 | `learn_observation_mode = recent` で、`learn_ratio` が有限でないか、0 以下 | ERROR（キーと値を示す） |
| 7 | `learn_observation_mode = recent` で、`learn_batch_size` が容量を超える | ERROR（2 つのキーと値を示す） |
| 8 | それ以外 | 有効。読まない設定（`statistics_mode = variance_feedback` のときの `error_norm.mode`、`error_norm.mode = none` のときの `error_norm.use_centering`、`learn_observation_mode = intake` のときの `learn_recent_capacity`）は、[PRD117](117_warn_policy_and_unread_config_10prd.md) の読まない設定の決まりに従う |

黙った代替はしない。`recent` で容量を lane の数に合わせて変える、`learn_ratio` を丸める、といった扱いをしない。

### 8.2 変えないもの

- 新しい設定キーが既定のとき（`standardize`・`intake`）、採点・補助学習の forward と backward の回数と順、乱数の消費、診断、メトリクスの tag、checkpoint の形を今と同じに保つ（G3、P12）。
- `variance_feedback` の中身と warm-up（§1.4）。
- Evaluation Trace の鍵と、`standardize` のときの値（P13）。

### 8.3 checkpoint

- 置き場は保存しない（未決表の 9）。credit・抽選の乱数の状態・累計の学習回数は今と同じく保存する。
- `none` と `standardize` で、RND の checkpoint の形は同じにする（未決表の 2）。

### 8.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| RndNoveltyEstimatorConfig | 変更 | `error_norm.mode` を読み、検証する | 内的報酬（RND） |
| WhitenStatistics | 変更 | `none` のとき、誤差の統計で割らずに倍率と Trace の値を作る | 内的報酬（RND） |
| DefaultDQNAgent の内的報酬の設定 | 変更 | `learn_observation_mode`・`learn_recent_capacity` を読み、`learn_ratio`・`learn_batch_size` の範囲を出どころで分けて検証する | DefaultDQNAgent |
| IntrinsicRewardCoordinator | 変更 | `recent` のとき、intake の観測を置き場へ入れ、置き場から k 件を引いて補助学習に渡す | DefaultDQNAgent |
| IntrinsicLearnBuffer（候補） | 新規 | 直近の観測を容量まで貯め、乱数で k 件を引く | DefaultDQNAgent（内的報酬） |
| LifelongNoveltyEstimator | 変更 | 推定器が持つ Network の構造図を返す口を足す | 内的報酬の口 |
| RndNoveltyEstimator | 変更 | target と predictor の構造図を返す | 内的報酬（RND） |
| DefaultDQNAgent | 変更 | `RecordNetworkStructure` で、RND の構造図も記録する | DefaultDQNAgent |

### 8.5 移行対象

経路の検索（2026-11-19、`git grep -i -l -F`、追跡しているファイルの全体）: `learn_ratio` 99 件、`learn_batch_size` 109 件、`exceeds num_envs` 4 件、`error_norm` 64 件、`RecordNetworkStructure` 7 件、`StandardizeWithoutUpdate` 1 件。そのうち PRD・実装計画・実験記録・ADR・課題・レポート・wiki の作業の記録（過去の記録で、書き換えない）を除いた現用のものと、扱いは次のとおり。

- コード: [intrinsic_rnd.hpp](../../src/core/base/src/intrinsic_rnd.hpp)・[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)（式・設定・構造図）、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp)（設定・補助学習・構造図）、内的報酬の口 [intrinsic.hpp](../../src/core/base/include/anet/intrinsic.hpp)（`LifelongNoveltyEstimator`）。テスト用の推定器（[src_testing/intrinsic.cpp](../../src/core/base/src_testing/intrinsic.cpp)・[include/anet/test/intrinsic.hpp](../../src/core/base/include/anet/test/intrinsic.hpp)）は、口を足すなら追随する。
- テスト: [intrinsic_test.cpp](../../src/core/base/src/intrinsic_test.cpp)（式・Trace の値・checkpoint）、[dqn_intrinsic_test.cpp](../../src/algo/dqn/src/dqn_intrinsic_test.cpp)（設定の判定・置き場・構造図）。`learn_ratio` の 1.01 を拒否する今の断定は `intake` のときの断定として残る見込み（実装計画で確かめる）。
- 設定: [agent.txt](../../apps/runner/config/agent.txt)（新しい既定キー 3 件）。Atari.txt と bat は変えない（未決表の 13）。
- 設計文書: [212](../design/212_rnd_lifelong_novelty.jp.md) の jp と en（式・補助学習・設定の表）、[200](../design/200_dqn_agents.jp.md) の jp と en（内的報酬の設定の行と、構造図の記録の記述）。
- 検証資材: `src/core/base/testdata/prd116`・`prd121`・`prd123`・`prd127`・`prd128`・`prd130` の道具は `learn_ratio`・`learn_batch_size`・`error_norm` を今の値で書く。既定を変えないので直さない。
- 互換層: 旧い形の別名や互換の分岐は作らない（クリーンブレーク）。過去の Run・実験記録・ADR は書き換えない。

### 8.6 ほかの PRD との順序

- [PRD190](190_dqn_design_doc_alignment_10prd.md)（未実装）は、設計文書 200 のコードマップに内的報酬の行を足す。どちらが先でもよい。本 PRD が後なら、足した行に新しいファイル（置き場の部品を別のファイルに置くとき）を足す。本 PRD が先なら、PRD190 はその時点のファイルで行を書く。
- [PRD173](173_intrinsic_reward_novelty_10prd.md)（NGU 型。暫定）は、エピソード内の新しさを RND の隣に足す。本 PRD の補助学習の置き場は RND の補助学習の中に閉じ、PRD173 の採点の場所の論点に影響しない。

## 9. 成功の測り方と受入

### 9.1 効果（Atari の枠で確かめる）

効果は §1.3 の仮説 A・B で確かめる。群の組み立ては Atari の枠で決める（未決表の 13）。次は群を組むときの材料で、決定ではない。

- **読む指標。** 評価側（Greedy と ε 0.01）の点、Montezuma なら 2500 点の割合と訪れた部屋の種類数。学習側では `56_agent_in/05_bonus_pos_ratio`（内的報酬が正になる割合）、`01_bonus_mean` の時間の変わり方、`12_error_std`。
- **seed。** 群ごとに 2 本以上にし、同じ構成の seed のあいだの幅を超える差だけを読む。

### 9.2 正しさの受入

暫定の既定を前提にした候補で、再グリルで固める。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 既定のときの同一性 | 新しい設定キーが既定のとき、同 seed・決定的モードの短い Run で、実装前に保存した実行体と metrics checksum が一致する。対象は Atari の `run.@rnd`、LunarLander の RND あり・なし、GridMaze の RND あり。区間は PRD116 の無効時の同一性の検証と同じ短縮条件（PRD098 の検証の profile、exp step 8,704 まで）とする。比べるのは全 tag の全点の値 |
| 2 | 誤差で割らない式 | 既知の誤差を渡すテストで、`none` の倍率が 1 ＋ e（`unclipped`）、[1, L] に切った値（`clipped`）、warm-up 中は 1 になる。`11_error_mean`・`12_error_std` の診断値が `standardize` と同じに更新される |
| 3 | Trace の値 | `none` のとき、統計を変えない口が返す「標準化した値」が e に等しい |
| 4 | 置き場から引く補助学習 | テストで、置き場が容量を超えると古い観測から捨てる。1 step の中で同じ観測を 2 回引かない。lane の数を超える `learn_batch_size` で学習が進む。`learn_ratio` 2.0 で、1 intake あたりの step の数が 1.0 のときの 2 倍になる。置き場が k 件に満たない intake では `Learn` を呼ばず、credit を消費する。同じ seed で引く観測の並びが一致する |
| 5 | 設定の判定 | §8.1 の判定の表の各行が、表のとおりに扱われる。ERROR の文に対象のキーと値が入る |
| 6 | checkpoint | `intake`・`standardize` で、実装前の実行体が保存した checkpoint を読める。`recent` で保存して読み込むと、置き場は空から始まり、credit と抽選の乱数の状態は保存前と一致する |
| 7 | 構造図 | RND を有効にした短い Run の `dot/` に、`intrinsic.rnd.target.net.structure.dot`・`intrinsic.rnd.target.net.detail.dot`・`intrinsic.rnd.predictor.net.structure.dot`・`intrinsic.rnd.predictor.net.detail.dot` の 4 つがある（primary の Network の `net.structure.dot` と同じ付け方）。RND を無効にした Run の `dot/` には、この 4 つが無い |
| 8 | 性能の記録 | Atari で、`intake`（`learn_ratio` 0.25）と `recent`（`learn_ratio` 1.0）を同じ実行体で交互に 3 巡以上測り、群ごとの実時間の中央値を実装計画に記録する。閾値は置かない |

## 10. issue の対応表

| ID | 題 | area | 根拠の位置 | 直す（未決表） | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261119-027698](../management/issues/issue-20261119-027698.md) | RND に生の予測誤差を返す内的報酬の式が無い | 324_dqn, 321_agent_core | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 293〜301 行・233〜239 行 | 1〜4 | 2・3・5 |
| [issue-20261119-4976bf](../management/issues/issue-20261119-4976bf.md) | predictor の補助学習が今の intake の観測だけから引く | 324_dqn, 321_agent_core | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 79〜83 行・139〜166 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 172〜175 行 | 5〜10 | 4・5・6 |
| [issue-20261115-ce0101](../management/issues/issue-20261115-ce0101.md) | NN の構造図に RND の target と predictor の Network が出ない | — | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 460〜472 行 | 11 | 7 |

ce0101 は、未決表の 11 の暫定の既定で範囲に入れた。範囲に入れると再グリルで決めたら、その課題の `prd` に本 PRD を書く。

## 11. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.3（2 つの価値）、§2.4（報酬と観測の正規化）、§3.4（env の数と predictor の実効バッチ）、付録 A.2（Algorithm 1）、表 2・4・5。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
