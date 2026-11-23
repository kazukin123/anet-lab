# RFD0191: RND の設定を段階ごとに組み替え、誤差で割らない内的報酬の式と、ためてからエポックで学ぶ predictor の補助学習を足す（Burda らの形へ寄せる） RFD

> 2026-11-19 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §6 の未決表に置いた。§7〜§9 は §6 の暫定の既定を前提に書き、前提にした行の番号を名指す。行番号は起票時点の作業ツリーの位置である。
> 2026-11-21 改訂（再グリル、anet-create-prd、Claude Code）。グリルを行い、暫定の印を外した。起票時の未決表の 17 行をすべて振り分け（行き先は §10.1）、確定事項の節（§6）を新しく作った。開発者が選んだのは 10 件である。補助学習の形（D1）、置き場の共通化の範囲（D2）、割らない式の待ちの間の内的報酬スケーラの統計（D3）、誤差の式のモード（D4）、構造図を入れるか（D5）、ためてから学ぶ形の量のキー（D6）、RND の設定の組み替え（D7）、warm-up を段階ごとの待ちに分けること（D8）、clip を誤差のモードごとに持つこと（D9）、実装を、コードと設定のステージと、設計文書のステージに分けること（D10）。細部は 2 回、一覧で示し、どちらも推奨の既定で埋めた（§6.3）。起草のあとに別モデル（Opus）の検討を受け、論点 10 件を扱った（§12）。そのあと開発者が、設定を段階ごとのスコープとモード別の名前空間に分ける形を示し、D7〜D10 を決めた。起草のあとに確かめた 3 件（ためる形の引き方、ステージの分け方、clip の真偽のキーの名前）は、開発者の答えを D41・D10・D28 に反映した。起票時の暫定の既定から変わった主な点は 5 つある。(1) 補助学習は「直近の観測の置き場から引く形」をやめ、「rollout ぶんをためてからエポックで学ぶ形」にした。(2) 割らない式の待ちの間は、内的報酬スケーラの統計を実際の値で育てる。(3) RND と補助学習の今ある設定キーも、段階ごとのスコープとモード別の名前空間へ移す。(4) warm-up の束（`warmup_mode`）をやめ、段階ごとの待ちに分ける（PRD123 の D4 を改める）。(5) 選ばれていないモードのキーには WARN も INFO も出さず、`rnd` の下の今ある同種の通知も無くす。あわせて、題、目的（§1）、前提の表（P1〜P63）、背景（§3.2 の事例を公開実装で確かめ直し、§3.3 に NGU の学び方を足した）、用語（§4）、現状（§5.3・§5.5〜§5.8 を足し、行番号を 2026-11-21 の作業ツリーの位置に直した）、仕様（§7）、実装契約（§8）、受入（§9）、未決表（§10）、issue の対応表（§11）を書き直し、グリルの材料（§12）を足した。課題 [ce0101](../management/issues/issue-20261115-ce0101.md) を範囲へ取り込んだ。[ADR 0088](../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md) を起こし、[CONTEXT.md](../../CONTEXT.md) に 2 語を足した。ファイル名は起票時のままである（課題の `prd` 欄と wiki がこの名前で指す）。
> 2026-11-22 改訂（Claude Code）。置き場の部品の機能グループの名前と機能カテゴリを、開発者が選んだ（D56）。§8.4 に機能グループの見込みを、§8.8 に機能カテゴリの行の変更を足した。
> 起点: 2026-11-19 の会話。開発者が、今の RND の実装を Burda らの論文（[arXiv 1810.12894](https://arxiv.org/abs/1810.12894)）と比べるよう求めた。AI が、設定で変えられる差と Run の規模の差を除いた実装の差を挙げ、開発者がそのうち 4 件を課題に起票させた（[cb9a0f](../management/issues/issue-20261119-cb9a0f.md)・[4976bf](../management/issues/issue-20261119-4976bf.md)・[027698](../management/issues/issue-20261119-027698.md)・[497fce](../management/issues/issue-20261119-497fce.md)）。開発者は、027698 と 4976bf を先にする見込みを示し、2 件を 1 本の PRD にまとめて暫定で起こすよう指示した。あわせて「優先度が高くなくても、ついでに実装だけまとめて入れることも考えておきたい」と述べた。どれも決定ではない。
> 題材: [issue-20261119-027698](../management/issues/issue-20261119-027698.md)（RND に生の予測誤差を返す内的報酬の式が無い）、[issue-20261119-4976bf](../management/issues/issue-20261119-4976bf.md)（predictor の補助学習が今の intake の観測だけから引く）。範囲へ取り込んだ課題: [issue-20261115-ce0101](../management/issues/issue-20261115-ce0101.md)（NN の構造図に RND の Network が出ない）。グリルの中で起こした課題: [issue-20261122-c1744a](../management/issues/issue-20261122-c1744a.md)（RND と補助学習の設定キーが、段階とモードの関係を表していない）。
> 関連: [116](0116_rnd_lifelong_novelty_10rfd.md)（RND の本体）、[123](0123_rnd_intrinsic_reward_scale_10rfd.md)（誤差の統計の取り方・warm-up・平均を引くか。D4 を本 PRD が改める）、[121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md)（価値の分解と、内的報酬用の報酬スケーラの `return_std`）、[130](0130_rnd_estimate_statistics_perf_10rfd.md)（採点の統計の性能と統計方式の整理）、[117](0117_warn_policy_and_unread_config_10rfd.md)（読まない設定の扱い）、[173](0173_intrinsic_reward_novelty_10rfd.md)（NGU 型の内的報酬）、[190](0190_dqn_design_doc_alignment_10rfd.md)（設計文書 200 の直し）、[194](0194_config_key_full_names_in_messages_10rfd.md)（エラー文が示す設定キーの形）、ADR [0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md)（WARN の意味と読まれない設定キー）・[0088](../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md)（RND の設定の形）、設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md)（RND）・[214](../design/214_value_decomposition.jp.md)（価値の分解）。

## 1. 目的と問題

### 1.1 問題

1. **内的報酬の式を、論文の形にできない。** 論文の内的報酬は、target と predictor の出力の差の二乗ノルムで、それを内的の収益の標準偏差の移動推定で割る。clip はせず、平均も引かない（§3.1、前提 P1・P2）。anet の whiten は、どの設定でも誤差を誤差の統計の標準偏差で割る（§5.1、P4）。内的報酬用の報酬スケーラの `return_std`（RFD0121）を重ねると、標準偏差で 2 回割る形になり、誤差の統計の標準偏差は Run の中で変わるので、値は論文の値に比例しない（[027698](../management/issues/issue-20261119-027698.md)）。割らない式を足すだけでは、Run の出だしの内的報酬の大きさも論文と違う。論文は rollout を集め終えてから統計を作って割るが、anet は intake ごとに報酬を確定するので、統計が育つ前の数 intake で大きな値が出る（§5.3、P31・P33）。
2. **predictor の補助学習を、論文の学び方にできない。** 論文は、rollout ごとに集めた観測の 25% を残し、4 エポック、約 1,000 件ずつのミニバッチで学ぶ（§3.1、P3）。anet は、今の intake の観測だけから、各観測を高々 1 回、`learn_batch_size` 件ずつ引く。1 step の件数は lane の数を超えられず、`learn_ratio` は 1 以下である（§5.2、P6・P7）。既定（`learn_ratio` 0.25、32 件）では 1 経験あたり 0.25 回の勾配で、論文の約 1 回の 4 分の 1 である。`learn_ratio` を 1 にすれば回数はそろうが、同じ観測の再利用と、lane の数を超えるバッチは作れない（[4976bf](../management/issues/issue-20261119-4976bf.md)）。
3. **RND の Network の形を、Run の成果物で確かめられない。** Network の構造図は primary と skip の Network だけを記録し、RND の target と predictor は記録しない（§5.4、P14。[ce0101](../management/issues/issue-20261115-ce0101.md)）。
4. **設定の形が、段階とモードの関係を表していない。** RND の設定キーは `rnd` の直下に並び、あるモードでだけ使うキーと、どのモードでも使うキーを、名前から見分けられない（§5.8）。warm-up は、1 つの件数と `warmup_mode` が、誤差の統計の待ちと predictor の待ちを束ねている（P59）。倍率の上限は、誤差の式で、切る対象の単位が変わる（P56）。使われないキーが既定と違うと、WARN が出る（§5.7）。割らない式と、ためてから学ぶ形を足すと、この絡みが増える。再グリルの中で、開発者は設定の持ち方を問う選択肢を読み取れず、設定の形そのものを直す案を示した（§12 の発言 8〜10）。

### 1.2 ゴール

- **G1**: whiten で、誤差を誤差の統計で割らない式を選べる。内的報酬用の報酬スケーラの `return_std` と組めば、論文の内的報酬の形（誤差 ÷ 内的の収益の標準偏差）になる。待ちの間に内的報酬スケーラの統計を育て、出だしの内的報酬を落ち着いた大きさにできる。
- **G2**: predictor の補助学習で、lane の数を超えるバッチと、同じ観測の再利用（1 経験あたり 1 回を超える学習）を選べる。
- **G3**: 既定の動きは今と同じにする。既定の設定と、今ある形（`collect_errors`、直近重み、`variance_feedback`、RND 無効）を新しいキーで書き直した設定で、同 seed・決定的モードの短い Run の metrics が、実装前の実行体と一致する。
- **G4**: RND の target と predictor の Network の構造図を、primary の Network と同じく記録する。
- **G5**: RND と補助学習の設定キーを、段階ごとのスコープとモード別の名前空間にそろえる。モードを 1 つ切り替えると、そのモード向けのキーが使われる。設計文書 212 を、段階・モード・キーの表から読める形に書き直す。

G1〜G3 で題材の課題 2 件が片付き、G4 で取り込んだ課題 1 件が片付く。G5 に当たる課題は、工程の終わりに 1 件作る（§11）。

### 1.3 仮説・比べ方・終わり方

| | 仮説 | 比べ方 |
|---|---|---|
| A（内的報酬の式） | 論文の形の内的報酬（誤差で割らない式、`return_std`、内的の価値を非エピソードにし割引率を別に持つ）にすると、内的報酬が正になる経験の割合が上がり、Montezuma の Greedy で 2500 点のあとに Q が平らになる局面（RFD0121 §2.1）が変わる | 同じ実行体で、今の既定の RND と論文の形の RND を、価値の分解を有効にした同じ構成で比べる |
| B（predictor の学習） | predictor の学習の量と 1 step の件数を論文にそろえると、慣れた状態の誤差が早く下がり、新しい状態との誤差の差が広がる | 今の形（1 経験あたり 0.25 件・32 件と、1.0 件・128 件）と、ためてから学ぶ形（論文の表 5 の量）を比べる。今の形の 1.0 を入れると、量の効果と、再利用・大きいバッチ・rollout の間 predictor を変えないことの効果を分けて読める |

群の組み立て（構成、β の値、seed の数、長さ）は本 RFD で決めない。Atari の枠で実験の段取りと一緒に決める（D55）。学習器が PPO ではないので、論文の点とそのまま比べることはできない（§12 の発言 3）。

G5 は仮説を持たない。確かめるのは、動きが変わらないこと（G3）と、設計文書が読めること（受入 18）である。

**終わり方**: 効果が出なくても仕組みは残す。既定の動きは今のままである（G3）。効果が出たとき、既定を変えるかは Atari の枠で別に決める。

### 1.4 非ゴール（退けた案と、本当の制約か）

| 案 | 扱い | 本当の制約か |
|---|---|---|
| 学習の前に一様乱数の行動で観測の統計を作る手順（[cb9a0f](../management/issues/issue-20261119-cb9a0f.md)） | 範囲外。Train Actor の ε の予定を、最初の N step だけ 1.0 にする設定で近い形を試せる見込み（`spatial.epsilon.first`・`last` は時間の予定を書ける。[200](../design/200_dqn_agents.jp.md) 249 行。試していない）。RFD0123 の U2 で観測の正規化の式を開き直さないと決めた（P10）。作る条件: 設定で試した結果から、乱数の行動で統計を作る手順そのものが要ると分かったとき（実験の結果が要る） | 制約ではない |
| 非エピソードの内的の収益を、エピソードをまたいで n-step の和に足すこと（[497fce](../management/issues/issue-20261119-497fce.md)） | 範囲外。Replay の窓がエピソードをまたぐ契約の変更になる別の関心（RFD0121 §1.4 と同じ）。作る条件: `non_episodic` の効果が分かり、終端の近くの数 step の差が効いていると分かったとき | 制約ではない |
| PPO と再帰（GRU など）の Network | 範囲外。学習器そのものの作りという別の関心。作る条件: 学習器の作りを変える RFD を起こすとき | 制約ではない |
| 既定の動きを論文の形に変えること | 先送り。実験の結果が要る。作る条件: §1.3 の仮説 A・B の群で効果が出て、Atari の枠で既定を変えると決めたとき | 制約ではない |
| `variance_feedback`（Taiga らの再現のモード）の中身 | 変えない。RFD0123 の D8 による。キーの置き場は移す（D7） | 制約ではない（別の関心として RFD0123 で決めた） |
| predictor が Replay から引いた観測で学ぶ形（NGU の学び方） | 範囲外（D1・D19）。PER の偏りを受け、論文の形に寄せる目的に合わない（§3.3）。NGU の構成での predictor の学び方という別の関心で、[RFD0173](0173_intrinsic_reward_novelty_10rfd.md) の未決表の 13 が決める。作る条件: RFD0173 がその形を選んだとき | 制約ではない |
| フレーム積層の履歴と Replay の観測の置き場を、新しい置き場の部品へ載せ替えること | 範囲外（D2）。RND を使わない Run を含む全 Run の経路に触るので、RND の学び方とは別の関心である（P27・P28）。作る条件: 「ためる仕組みの統合整理」の課題（§10 の U1）に取り組むとき | 制約ではない |
| 報酬の確定を rollout の終わりまで遅らせ、論文と同じ時機で統計を作って割ること | 先送り。Replay への Push を rollout の長さだけ遅らせる変更になり、Q の学習の経路に触る。規模は数えていない。実験の結果が要る。作る条件: D3 の形で比べた結果から、最初の rollout の内的報酬が 0 であることが効いていると分かったとき | 制約ではない |
| `observation`・`use_done_mask` と `reward_scaler.*` を、同じ形（モード別の名前空間）にそろえること | 範囲外（D55）。`reward_scaler` は外的報酬用と共用で、書いている箇所が 662 行ある（P60）。RND の中の組み替えとは別の関心である。作る条件: §10 の U2 の課題に取り組むとき | 制約ではない |
| 使われない設定の通知の決まり（[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則の表、RFD0117 の D1、ADR 0071）と、`rnd` の外に残る通知（`intrinsic.use_done_mask` など）の見直し | 範囲外。本 RFD は、RND の設定を組み替える中で、`rnd` の下の通知だけを無くす（D12）。決まりそのものの見直しは別の関心である。作る条件: §10 の U2 の課題に取り組むとき | 制約ではない |
| RND の Head の出力次元の設定キーの置き場（[b02db6](../management/issues/issue-20261115-b02db6.md)） | 範囲外（D55）。Head の設定を `net.head.*` へ移すかを、RFD0155 の §3.8・Q8 が決める別の関心。作る条件: RFD0155 がその置き場を決めたとき | 制約ではない |
| Atari の Run プロファイルと bat の群を足すこと | 範囲外（D55）。Atari の枠で実験の段取りと一緒に決める別の関心。今ある行のキーの名前は、本 RFD で移す（D49）。作る条件: Atari の枠で群を組むとき | 制約ではない |

### 1.5 機構とゴールの対応

| 機構 | ゴール | 寄与 |
|---|---|---|
| 段階ごとのスコープとモード別の名前空間（D7） | G5 | あるモードでだけ使うキーが、名前から分かる。モードを切り替えると、そのモードのキーが使われる |
| 段階ごとの待ち（D8） | G5・G1 | warm-up の束をほどき、待ちをそれぞれ 1 つの段階のスコープに入れる。割らない式の待ちを、その式自身のキーで書ける |
| モードごとの clip（D9） | G5・G1 | 倍率の上限を、切る対象の単位が同じモードの中で持つ。割らない式に切り替えるだけで、論文の形（切らない）になる |
| 誤差で割らない式（`error_norm.mode = none`。D4） | G1 | 誤差の統計で割らずに倍率を作る。`return_std` と組んで論文の形になる |
| 割らない式の待ちの間に内的報酬スケーラの統計を育てる扱い（D3） | G1 | 待ちの間、実際の値で統計を育て、内的報酬は 0 にする。明けの最初から落ち着いた大きさで割れる |
| ためてからエポックで学ぶ補助学習（`learn.mode = rollout`。D1・D6） | G2 | rollout ぶんの観測をためてから学ぶので、1 step の件数が lane の数に縛られず、同じ観測を何度も学べる |
| lane × 時間の tensor の置き場の部品（D2） | G2 | rollout ぶんの観測をためる入れ物。RND と無関係な名前で共通の層に置き、あとでほかの置き場を載せ替えられる形にする |
| 構造図の記録（D5） | G4 | RND の Network の形を、Run の `dot/` で確かめられる |
| 設計文書 212 の書き直し（D10・D52） | G5 | コードのあとに、段階・モード・キー・待ちを、読む順に組み直す |

## 2. 前提

P1〜P20 は、再グリルの最初の応答で共通の前提として示し、説明が要る番号は挙がらなかった。P21 以降は、1 つの問いに添えた前提と、起草で足した前提である。行番号は 2026-11-21 の位置で、同じ日に入った [RFD0194](0194_config_key_full_names_in_messages_10rfd.md) の実装（b72d6e46・7d90a2a7）のあとのものである。

| # | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 事実（文書） | 論文は内的報酬を、内的の収益の標準偏差の移動推定で割って正規化する | [Burda ら](https://arxiv.org/abs/1810.12894) §2.4 | 済み |
| P2 | 事実（文書・公開実装） | 論文の内的報酬は target と predictor の出力の差の二乗ノルムで、内的報酬の clip も平均の差し引きも無い。公開実装は、差の二乗を特徴の次元で平均する | 同 付録 A.2 の Algorithm 1、表 2。[公開実装の cnn_policy_param_matched.py](https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py) の `define_self_prediction_rew`（2026-11-20・21 に確認） | 済み |
| P3 | 事実（文書） | 論文の predictor は rollout ごとの batch を 4 エポック・4 ミニバッチで学び、経験の 25% を使う。128 env で 32 env のときと実効バッチをそろえるため、要素を確率 0.25 で残す。1 経験あたり約 1 回、1 step 約 1,000 件になる | 同 付録 A.4 の表 5 と本文。§3.1 の計算 | 済み |
| P4 | 事実（コード） | whiten の倍率は α = 1 ＋（e − 平均）÷ 標準偏差（平均を引かない設定では 1 ＋ e ÷ 標準偏差）。warm-up 中と統計が空のときは 1 | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 297〜312 行 | 済み |
| P5 | 事実（コード・文書） | 内的報酬用の報酬スケーラの `return_std` は lane ごとに R = γR ＋ r を持ち、R の標準偏差で r を割る。分解して内的の割引率を別にしたときは、その割引率で作る | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 962〜971 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 351〜356 行、[212](../design/212_rnd_lifelong_novelty.jp.md) §3.9 | 済み |
| P6 | 事実（コード） | 補助学習は、intake ごとに credit を「lane の数 × `learn_ratio` ÷ k」だけ貯め、同じ intake の lane を重複なく k 件ずつ渡す | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 142〜176 行 | 済み |
| P7 | 事実（コード） | 学習が有効で `learn_batch_size` が lane の数を超えると、構築時に止まる。`learn_ratio` は 0 より大きく 1 以下 | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 80〜85 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 177〜182 行 | 済み |
| P8 | 過去の判断 | 平均を引かない形を `multiplier_mode` の 3 つ目の値にする形は不採用。clip の有無を表すキーに別の軸が混ざるため | [RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md) D7・§4.1、[321 のトピック別まとめ](../wiki/categories/321_agent_core.md) の選ばなかった形 | 済み |
| P9 | 過去の判断 | `learn_batch_size` の既定を lane の数に結び付ける形は不採用。lane の数より大きい指定は構築時に止める | [RFD0117](0117_warn_policy_and_unread_config_10rfd.md) D11、[324 のトピック別まとめ](../wiki/categories/324_dqn.md) の選ばなかった形 | 済み |
| P10 | 過去の判断 | RFD0123 の U2 で、観測の正規化の式と 2 ステージの warm-up を開き直さないと決めた | [RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md) 冒頭の 2026-11-15 の改訂行 | 済み |
| P11 | 事実（測定） | Atari の主な構成（128 lane）で、RND 全体は 1 intake あたり約 6.0 ms（採取した区間の約 5.6%）、補助学習 1 回（32 件）は約 3.6 ms だった。割り算から、1 intake は約 107 ms である | [RFD0130 の実装メモ](0130_rnd_estimate_statistics_perf_20impl.md) の「主の再採取と GridMaze 前後の 3 本の Tracy」の行。§5.5 | 済み |
| P12 | 守る線 | RND の既定や統計の取り方を変えるときは、RND 無効を含む構成で、同 seed・決定的モードの短い Run の metrics checksum を着手前の実行体と一致させる | [321 のトピック別まとめ](../wiki/categories/321_agent_core.md) の変えるときの条件 | 済み |
| P13 | 事実（コード） | Evaluation Trace の「標準化した値」は、統計を変えない口（`StandardizeWithoutUpdate`）が、倍率から 1 を引いた、clip の前の値として作る | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 237〜243 行・841〜849 行、[212](../design/212_rnd_lifelong_novelty.jp.md) §4.6 | 済み |
| P14 | 事実（コード） | Network の構造図を記録するのは primary の Network と、skip が有効なときの補助 Network だけ。記録する時点（同 399 行）で、内的報酬の部品（同 354〜356 行）はもう作られている | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 354〜356 行・399 行・468〜483 行 | 済み |
| P15 | 事実（文書） | 共通の ReplayBuffer は Agent の archive へ serialize されず、checkpoint を読んでも新規構築の状態から始まる | [150](../design/150_replay_buffer.jp.md) §7.6 | 済み |
| P16 | 過去の判断（暫定） | 補助学習の出どころは intake の観測である。RFD0116 は、Replay から引く形を候補にし、RFD0173 への移行のときに intake での補助学習を消すことを暫定の既定にしていた。開発者は 2026-11-16 に「NGU の対応のあとも残る仕組みだけを今入れる」という基準を示した | [RFD0116](0116_rnd_lifelong_novelty_10rfd.md) §8 の 12・13、[RFD0130](0130_rnd_estimate_statistics_perf_10rfd.md) D9・§5.7 | 済み |
| P17 | 過去の判断 | credit と抽選は Coordinator が持ち、推定器の `Learn` は渡された観測すべてで 1 step 進める。収益の標準偏差で割る正規化は、推定器ではなく報酬スケーラが持つ | [RFD0128](0128_dqn_agent_refactor_bundle_10rfd.md) D11、[RFD0121](0121_extrinsic_intrinsic_value_decomposition_10rfd.md) D9、[intrinsic.hpp](../../src/core/base/include/anet/intrinsic.hpp) 28〜31 行 | 済み |
| P18 | 事実（コード） | 内的報酬は warm-up 中（α = 1 で 0）も内的報酬スケーラを通り、動的なスケーラでは 0 が統計に入る | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 120〜125 行・253 行 | 済み |
| P19 | 事実（計算。Run では未確認） | P18 から、今の作りで誤差で割らない式と `return_std` を組むと、warm-up 明けの最初の batch の内的報酬が桁で大きくなる。値は P33 と §5.3 の表による | §5.3 | 済み |
| P20 | 事例 | CleanRL は 128 env × 128 step、4 ミニバッチ × 4 エポック、割合 0.25 で、エポックごとに並べ替えて重複なく切る。DI-engine は集めた観測を貯め、収集 1 回ごとに 64 件を 100 回引く。NGU は Replay から引いた系列で predictor を学ぶ | [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)、[DI-engine](https://github.com/opendilab/DI-engine/blob/main/ding/reward_model/rnd_reward_model.py)（2026-11-20 に確認）、§3.3 | 済み |
| P21 | 事実（文書） | 学習の入口（`UpdateFromBatch`）の間は学習のロックを持ち、EvalPanel などの Actor の同期はその終わりを待つ | [212](../design/212_rnd_lifelong_novelty.jp.md) §5.1 | 対象外 |
| P22 | 事実（文書） | RND に渡す観測は、フレーム積層の前の最新 1 フレームである。Atari なら 84 × 84 の 1 バイト画素で、16,384 件で約 116 MB | [212](../design/212_rnd_lifelong_novelty.jp.md) §3.1 | 対象外 |
| P23 | 事実（設定）・過去の判断 | Atari の主な構成の Q の学習は、Replay から 256 件を引く。Replay からの抽出を 2 本持つ形は、RFD0103 で先送りにした（PER・prefetch・世代つきの key が 2 系統になる） | [Atari.txt](../../apps/runner/config/Atari.txt) 809〜810 行、[324 のトピック別まとめ](../wiki/categories/324_dqn.md) の選ばなかった形 | 対象外 |
| P24 | 開発者の発言 | RND と NGU は考え方が違い、両立を目指す。今は NGU を引きずらず、RND 単体としてどうすべきかで決める。NGU の対応のあとは、設定で切り替える形になる見込み | 開発者の発言（2026-11-21）。§12 の発言 1・2 | 対象外 |
| P25 | 事実（文書） | NGU の predictor は、優先度つきで引いた系列の末尾 5 時点だけで学び、経験の約 2.4% に当たる。論文は理由も、補助損失への重要度の補正も書いていない | [NGU 論文](https://arxiv.org/abs/2002.06038) §2・§3・付録 A・表 6（2026-11-21 に PDF で確認）。§3.3 | 対象外 |
| P26 | 事実（コード） | 今ある「ためる仕組み」は 3 つで、tensor の置き場 2 つ（フレーム積層の履歴、Replay の観測の置き場）は「キーごとに lane × 時間の tensor を最初の書き込みで確保し、時間の軸に書き、番号で読む」ところが同じである。書き込み位置の持ち方と読み方は違う | §5.6 | 対象外 |
| P27 | 事実（コード） | フレーム積層は Actor の毎 step、Replay の置き場は毎 intake の Push と抽出で通る。どちらも全 Run の経路である | [dqn_actor.cpp](../../src/algo/dqn/src/dqn_actor.cpp) 1536〜1537 行、[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 525 行・1481 行 | 対象外 |
| P28 | 過去の判断 | Replay の置き場は、フレーム履歴の整合（RFD0077・RFD0078）と抽出の性能を合わせて作り込んだ箇所である | [RFD0077](done/0077_valid_index_scratch_reuse_10rfd.md)、[RFD0078](done/0078_replay_frame_history_integrity_10rfd.md) | 対象外 |
| P29 | 事実（コード） | `return_std` は、Run の最初からの R すべての標準偏差で r を割る。割引和 R は、エピソードで切らない。その intake の値で統計を更新してから割る。標準偏差には下限（既定 1e-8）を足す | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 855〜880 行・962〜971 行、[scaler_impl.hpp](../../src/core/base/src/scaler_impl.hpp) 44 行、[agent.txt](../../apps/runner/config/agent.txt) 33 行 | 対象外 |
| P30 | 事実（コード） | warm-up の間、推定器は倍率 1 を返す。内的報酬は β ×（倍率 − 1）なので、スケーラには 0 が流れ、その 0 も統計の件数に入る | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 120〜125 行・253 行、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 301〜304 行 | 対象外 |
| P31 | 事実（公開実装） | Burda らの公開実装は、rollout を集め終えてから、その rollout の全 step の割引和で統計を更新し、同じ rollout の内的報酬を標準偏差で割る。平均は引かず、clip もしない。内的報酬を 0 にする期間は無い | [公開実装の ppo_agent.py](https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py) の `update`（2026-11-21 に確認）。§3.2 | 対象外 |
| P32 | 事実（設定） | いまの設定ファイルと bat に、内的報酬スケーラを動的にした構成は無い | `apps` の下の txt と bat を `intrinsic.reward_scaler` で検索（2026-11-21、14 件。どれも定数倍） | 対象外 |
| P33 | 事実（計算。Run では未確認） | 今の作りで G1 の形を使うと、出だしの内的報酬（β を掛ける前）は、warm-up 200,000 件の明けで約 40、12,800 件の明けで約 10 になる。warm-up 0 では、最初の intake が「誤差 ÷ lane の間の誤差のばらつき」になる | §5.3 | 対象外 |
| P34 | 事実（文書） | warm-up のもともとの役目は、誤差の統計で割る式で、序盤の大きな誤差を誤差の統計に入れないことである。誤差で割らない式は誤差の統計を倍率に使わないので、この役目は無くなる | [212](../design/212_rnd_lifelong_novelty.jp.md) §3.5、[RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md) D4 | 対象外 |
| P35 | 事実（コード） | `statistics_mode`（`whiten`｜`variance_feedback`）は、観測の統計と誤差の統計の取り方をまとめて選ぶ。値を checkpoint に書き、読み込み時に設定と違えば止める | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 709 行・905 行・923〜927 行 | 対象外 |
| P36 | 事実（文書・コード） | `error_norm.*`（`average_mode`・`ema_steps`・`use_centering`）は、whiten の誤差の正規化だけを持つ。checkpoint には保存せず、再開時の設定に従う。ただし `average_mode` は、checkpoint の書式の識別子を選ぶ | [212](../design/212_rnd_lifelong_novelty.jp.md) §4.2・§4.5、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 220 行 | 対象外 |
| P37 | 事実（文書） | 誤差で割らない式でも、観測の正規化は whiten と同じものを使う。論文も観測は正規化する | §3.1 | 対象外 |
| P38 | 事実（設定） | この一帯の「する／しない」の書き方は 2 通りある。列挙の `*_mode`（`multiplier_mode`・`warmup_mode`・`average_mode`）と、真偽の `use_*`・`enabled`（`use_centering`・`use_done_mask`・`intrinsic.enabled`） | [agent.txt](../../apps/runner/config/agent.txt) 3〜55 行 | 対象外 |
| P39 | 事実（コード） | P14 と同じ（構造図は primary と skip だけ） | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 468〜483 行 | 対象外 |
| P40 | 事実（課題記録） | ce0101 は未着手で、手間 3、WSJF 5 である | [ce0101](../management/issues/issue-20261115-ce0101.md) の監理の項目（2026-11-21 に確認） | 対象外 |
| P41 | 開発者の発言 | 構造図について、2026-11-15 に「早めに入れたい」、2026-11-19 に「ついでに実装だけまとめて入れることも考えておきたい」と述べた。どちらも決定ではない | 開発者の発言（2026-11-15・2026-11-19）。[ce0101](../management/issues/issue-20261115-ce0101.md) と冒頭の起点の行 | 対象外 |
| P42 | 事実（コード） | D3 と構造図は、同じ口（`LifelongNoveltyEstimator`）に操作を足す。口を実装しているクラスは `RndNoveltyEstimator` の 1 つである | `src` を `public LifelongNoveltyEstimator` で検索（2026-11-21、1 件。[intrinsic_rnd.hpp](../../src/core/base/src/intrinsic_rnd.hpp) 52 行） | 対象外 |
| P43 | 事実（文書） | 今の形のキーは 2 つで、`intrinsic.learn_ratio` は 1 経験あたりの学習の件数、`intrinsic.learn_batch_size` は 1 step の件数である | [212](../design/212_rnd_lifelong_novelty.jp.md) §3.5・§4.2 | 対象外 |
| P44 | 事実（計算） | 128 lane で、長さ 128、割合 1.0、1 step 1,024 件、エポック 4 なら、1 rollout で 16 step になる。エポックあたり 4 step、残す割合は 0.25 で、論文の表 5 と同じになる | §7.5 の対応表 | 対象外 |
| P45 | 事例（公開実装） | Burda らの公開実装は、エポック数（`nepochs`）、ミニバッチ数（`nminibatches`）、predictor に使う割合（`proportion_of_exp_used_for_predictor_update`）の 3 つで書く。ミニバッチは env の番号の続きの区切りで、エポックごとに並べ替えない。割合は、要素ごとの乱数と比べて残すので、残る件数は毎回ゆらぐ | [公開実装の ppo_agent.py](https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py)・[cnn_policy_param_matched.py](https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py)（2026-11-21 に確認）。§3.2 | 対象外 |
| P46 | 開発者の発言 | 使わない設定が既定と違うときの WARN は要らない。モードを変えたいだけなのに、関係ない設定まで同時に変える形は想定していない。INFO も要らない | 開発者の発言（2026-11-21）。§12 の発言 5・6 | 対象外 |
| P47 | 守る線（規約） | Fail-Fast 原則の表は、有効フラグの偽などで休止した機能の通知を INFO、ほかの設定で値が使われない構成の通知を WARN と定める | [AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」の表、[RFD0117](0117_warn_policy_and_unread_config_10rfd.md) D1 | 対象外 |
| P48 | 開発者の発言 | 有効・無効やモードの値を変えても、ほかの設定の意味がつられて切り替わらないようにする。そうなる場合は、モードごとに別の設定キーにする。モードだけ切り替えれば、そのモード向けの設定が使われる | 開発者の発言（2026-11-21）。§12 の発言 7 | 対象外 |
| P49 | 守る線（規約の読み） | Fail-Fast 原則の表は、通知が知らせる状態からログのレベルを決める表で、通知を出すことは義務づけていない。使わないキーも読んで検証すれば、読まれない設定キーの一覧には出ない | [AGENTS.md](../../AGENTS.md) の「Fail-Fast 原則」の表の前文、[RFD0117](0117_warn_policy_and_unread_config_10rfd.md) D5。AI の読みで、規約の見直しは §10 の U2 | 対象外 |
| P50 | 事実（規約） | エラー・WARN・INFO が設定キーを示すときは、利用者が設定に書く完全キーで書く | [RFD0194](0194_config_key_full_names_in_messages_10rfd.md) D12（2026-11-21 に実装が入った） | 対象外 |
| P51 | 事実（コード） | 内的報酬スケーラは、統計の件数を持ち、checkpoint に保存する。統計を変えずに割る口もある | [scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp) 898〜951 行 | 対象外 |
| P52 | 事実（設定・コード） | 報酬スケーラの設定の型は、外的報酬用と内的報酬用で同じである。外的報酬用は既定で動的で、全 Run の外的報酬が通る | [agent.txt](../../apps/runner/config/agent.txt) 300 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 77 行・359 行 | 対象外 |
| P53 | 事実（コード） | 定数倍のスケーラは、件数も状態も持たない | [scaler_impl.hpp](../../src/core/base/src/scaler_impl.hpp) 191〜197 行 | 対象外 |
| P54 | 事実（文書） | `warmup_steps` は、今も whiten と `variance_feedback` で同じキーを使う。warm-up の中身は、方式ごとに違う | [212](../design/212_rnd_lifelong_novelty.jp.md) §3.5、§5.8 | 対象外 |
| P55 | 事実（設定） | 倍率の clip の既定は `clipped`、上限 5 である | [agent.txt](../../apps/runner/config/agent.txt) 40〜41 行 | 対象外 |
| P56 | 事実（コード） | 上限が切る対象は、誤差の式で違う。標準化する式では「標準化した誤差に 1 を足した値」、割らない式では「誤差に 1 を足した値」である。上限 5 は、前者では平均より標準偏差 4 つ分、後者では誤差 4 に当たる | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 304 行・765 行・809〜810 行 | 対象外 |
| P57 | 事実（測定） | 誤差は、序盤を除けば 4 よりずっと小さい。KFM の実測で、誤差の標準偏差は 200k exp step の時点で 1.40〜4.60、それ以降だけで作ると 0.026〜0.034 だった | [RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md) §2.6 | 対象外 |
| P58 | 開発者の発言 | 設定は、段階ごとにスコープを分け、モードごとに下位のキーを分けて名前空間を分離する。被る要素があっても割り切り、モード別に調整する。スコープをまたいで相関するものは、`rnd` 直下で扱う | 開発者の発言（2026-11-21）。§12 の発言 8 | 対象外 |
| P59 | 過去の判断・事実（コード） | `warmup_mode` は、「誤差を誤差の統計に入れるか」と「predictor を学ばせるか」を、束で選ぶキーである。別々のスイッチにする形は、RFD0123 で不採用にした（要求の無い組が 2 つ増え、うち 1 つは誤差の統計の膨らみが再発する） | [RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md) D4・§4.1、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 176〜180 行・303〜306 行 | 対象外 |
| P60 | 事実（集計） | 今の RND と補助学習の設定キーを書いている箇所は、追跡しているファイルの全体で 138 ファイル・1,329 行である。そのうち、実装で書き直す対象は 17 ファイル・306 行（設定 3、bat 3、コード 6、テスト 2、RFD0194 の検証資材の期待の一覧 1、設計文書の jp 2）である。ほかに、過去の検証の道具の 13 ファイル・47 行は、説明に 1 行足すだけにし、wiki の 5 ファイル・12 行は、取り込みと点検が直す。残りは、過去の記録と英訳である。`reward_scaler.` を書いている箇所は、`apps`・`src`・設計文書（jp）で 662 行ある | §5.8 | 対象外 |
| P61 | 守る線（規約） | 形が 1 つしか無いものに、列挙の値を持つキーを作らない。モードのキーは、何のモードかが分かる名にする | [impl.md](../quality/impl.md) の「名前」の規約の 4・5 | 対象外 |
| P62 | 事実（設定・用語集） | スコープの下に `mode` を置く形（`init.mode`、`weight_norm.mode`、`eval_window.mode`）は、今の設定にある。モードの名前の下位スコープに、そのモードのキーを置く形は無い。`eval_window.rotating_size` は、モードの名前を前に付けた平らなキーである | [Atari.txt](../../apps/runner/config/Atari.txt) の `rnd.head.init.mode` と `weight_norm.mode` の行、[ImageCls.txt](../../apps/runner/config/ImageCls.txt) の `eval_window.rotating_size` の行、[CONTEXT.md](../../CONTEXT.md) の「分類・データセット」の節 | 対象外 |
| P63 | 過去の判断（ADR） | 読まれない設定キーは、実体化したカタログ項目では構築時に止め、それ以外は Run の構築の終わりに WARN で一覧する | [ADR 0071](../adr/0071-warn-means-user-must-fix-and-unread-config-leaves-are-detected.md) | 対象外 |

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
| Burda らの公開実装 | 差の二乗を特徴の次元で平均する。rollout を集め終えてから、その rollout の全 step の割引和で統計を更新し、同じ rollout の内的報酬を標準偏差で割る。平均は引かず、clip もしない。内的報酬を 0 にする期間は無い。学習の前の乱数行動は、観測の正規化のためだけにある | エポック数・ミニバッチ数・割合の 3 つで書く。ミニバッチは env の番号の続きの区切りで、エポックごとに並べ替えない。割合は、要素（env × step）ごとの乱数と比べて残すので、残る件数は毎回ゆらぐ | [ppo_agent.py](https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py)・[cnn_policy_param_matched.py](https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py)（2026-11-21 に確認） |
| Taiga ら（Rainbow＋RND） | 誤差の z を分散に相当する値で割り、0.00005 倍する | 行動した観測を agent step ごとに 1 件で採点・学習する | anet の `run.@rnd_taiga` が寄せた形（[RFD0116](0116_rnd_lifelong_novelty_10rfd.md) §2.1） |
| CleanRL の PPO＋RND | 内的の収益の標準偏差で割る | 128 env × 128 step、4 ミニバッチ × 4 エポック、割合 0.25。env の番号をエポックごとに並べ替えて、重複なく切る | [ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)（2026-11-20 に確認） |
| DI-engine の RND の報酬モデル | 正規化の有無を選べる（記憶による） | 集めた観測を貯め、収集 1 回ごとに 64 件を 100 回引く。貯めた観測は、古い順ではなく、まとめて消す | [rnd_reward_model.py](https://github.com/opendilab/DI-engine/blob/main/ding/reward_model/rnd_reward_model.py)（学習の列は 2026-11-20 に確認） |
| NGU | 誤差を誤差の平均と標準偏差で正規化し、1〜L に切った倍率にする（anet の whiten の `clipped` と同じ式） | Replay から優先度つきで引いた系列の末尾 5 時点（§3.3） | [NGU 論文](https://arxiv.org/abs/2002.06038)（2026-11-21 に確認） |

### 3.3 NGU の predictor の学び方（RFD0173 が引くための材料）

2026-11-21 に、NGU 論文（[arXiv 2002.06038](https://arxiv.org/abs/2002.06038)）の §2〜§5 と付録 A〜H を PDF で読んで確かめた。

- NGU は、actor が行動のときに内的報酬を計算し、報酬に足して Replay へ入れる。learner は優先度つきで引いた系列で Q を学び、その系列の末尾 5 時点だけで、行動予測と RND の predictor を学ぶ（付録 A）。
- learner が predictor に使うのは、毎秒約 5 回 × 64 系列 × 5 時点で 1,600 時点である。actor は 256 台 × 約 260 step で、毎秒 66,560 step を出す。割ると約 2.4% になる（数字は付録 A、割り算は本 RFD）。
- 論文は、この間引きの理由を書いていない。構成から読み取れるのは、行動と学習を切り離す分散構成（learner が経験を見る口は Replay だけ）と、量が合わないこと（1 台の learner が毎秒 6.6 万 step ぶんを学べない）の 2 つである。本文の §2 は「学習中に見た観測すべてで predictor を学ぶ」と概念の形で書き、付録 A の手順との差には触れていない。
- 優先度は Q の TD 誤差である。predictor 用の別の抽出も、補助損失への重要度の補正も書かれていない。ハイパーパラメータの表 6 にも、重要度の補正の指数の行は無い。
- NGU は、RND を「あってもなくてもよい」倍率として使う。倍率は 1〜5 に切り、時間がたてば 1 に戻ると書く（§2）。RND を外すと、Montezuma は 10.4k から 3.0k に下がる（表 3）。

Burda らの論文は、PPO の rollout で集めた経験そのもので predictor を学び、Replay も PER も無い。25% への間引きは乱数によるので、経験の分布は変わらない。RND 単独を Burda らの形に寄せるなら、PER と無関係に、経験した観測そのもので学ぶ形が合う（D1 の理由）。NGU が間引いた事情（毎秒 6.6 万 step）は、同期型で 128 lane の anet には当たらない。

### 3.4 これまでの anet の判断

- 平均を引くかは、`multiplier_mode` の値ではなく別のキー `error_norm.use_centering` にした（RFD0123 D7、P8）。clip の有無を表すキーに別の軸を混ぜないためである。
- `learn_batch_size` の既定を lane の数に結び付けない。lane の数より大きい指定は構築時に止める（RFD0117 D11、P9）。
- 観測の正規化の式と 2 ステージの warm-up は開き直さない（RFD0123 の U2、P10）。
- 内的報酬用の報酬スケーラの `return_std` と、内的の価値を非エピソードにして割引率を別に持つ形は、RFD0121 で入った（P5）。
- RFD0116 は、NGU の対応のときに intake で学ぶ道を消すことを暫定の既定にしていた（P16）。本 RFD は、RND 単体として決める（P24）。intake を起点に predictor が学ぶ道は、今の形も、ためてから学ぶ形も、NGU の対応のあとに設定で選べる形として残す前提で作る。

## 4. 用語

- **intake**: 学習側の Agent が、全 lane ぶんの経験を 1 回受け取ること。Replay へ Push する前の受け取りで、128 lane なら 128 件である。設計文書 212 が使ってきた語で、用語集には無かった。本 RFD で設定キーの単位になる（ためる形の長さは intake の数）ので、[CONTEXT.md](../../CONTEXT.md) に足した（D53）。既存の語では言い換えられない。「batch」は Replay から引く batch も指し、「exp step」は経験 1 件の単位である。
- **rollout（補助学習）**: 補助学習で、学ぶ前にためる、続いた intake のまとまり。論文の rollout（128 step × env）に当たる。用語集に足した（D53）。anet には、Q の学習の側に rollout に当たる単位が無い（Replay から引くため）ので、既存の語では言い換えられない。
- **補助学習**: 主学習（Q の学習）とは別に predictor を学ぶこと。[212](../design/212_rnd_lifelong_novelty.jp.md) §1.2 の語である。用語集にあるのは TempoRL の「補助学習のみモード」だけで、RND の補助学習の項目は無い。本 RFD では足さない（212 の語のまま使う）。
- **エポック（補助学習）**: ためた rollout の全件を一巡する回。用語集の epoch（データセットを一巡する単位）と同じく、一巡を数える語として使う。ためる形では、一巡のうち、predictor の学習に残す観測は `ratio ÷ epochs` の割合である（論文の「predictor に使う割合」）。全件を学習に使うとは限らない点は、用語集の項目に無い中身なので、本書で定める（§7.5）。用語集には足さない（論文の表 5 の語で、設定キーの名前 `epochs` に使う）。
- **段階**: 内的報酬ができる流れの区切り。本 RFD では、統計・誤差・predictor・補助学習の 4 つを指し、設定のスコープの単位にする（§7.1）。本書の中の説明の語で、用語集には足さない。実装の順の区切りは「ステージ」と書く。
- **待ち**: Run の最初に、ある段階の働きを始めるまで待つ件数。今の warm-up を、段階ごとに分けたものである（§7.2）。設定キーの名前は、今の語の `warmup_steps` を使う。本書の中の説明の語で、用語集には足さない（warm-up の言い換えである）。

## 5. 現状（2026-11-21 にコードで確認）

### 5.1 倍率の式

- whiten の採点は、誤差の統計の平均と標準偏差で倍率を作る（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 297〜312 行）。warm-up 中と、統計が空か標準偏差が 0 のときは 1 を返す。誤差の統計は、既定の `warmup_mode = learn_predictor` では warm-up 明けから更新し、`collect_errors` では warm-up の間も更新する（同 306 行）。観測の統計はそのあとに更新する（同 309 行）。
- `multiplier_mode = clipped` なら倍率を [1, L] に切る（同 809〜810 行）。
- Evaluation Trace の「標準化した値」は、統計を変えない口が同じ式で作る（同 237〜243 行）。warm-up の間の倍率は 1 にする（同 846〜847 行）。
- `error_norm.*`（`average_mode`・`ema_steps`・`use_centering`）は whiten だけが使う。

### 5.2 補助学習

- `IntrinsicRewardCoordinator::Learn`（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 142〜176 行）は、intake の観測を選び、credit を「lane の数 × `learn_ratio` ÷ k」だけ貯める。credit が 1 以上なら、lane の順列を作り、重複なく k 件ずつ RND の `Learn` に渡す。warm-up で学習しなかった分も credit を消費し、明けたときに追いつき学習をしない。
- RND の `Learn` は、渡された観測を最新の観測の統計で正規化してから、predictor を 1 回更新する（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 861〜888 行）。`collect_errors` の warm-up の間と `variance_feedback` の warm-up の間は、何も進めずに偽を返す（同 867 行）。
- `learn_batch_size` は、学習が有効なら lane の数以下、`learn_ratio` は 0 より大きく 1 以下（P7）。既定は 0.25 と 32（[agent.txt](../../apps/runner/config/agent.txt) 50〜51 行）。Montezuma の主な Run は、bat で 1 と 128 を書いている（[11_batch_run.bat](../../apps/11_batch_run.bat) の `TGRND`）。

### 5.3 内的報酬用の報酬スケーラと warm-up

内的報酬は、intake ごとに次の順で作る（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 113〜126 行）。

```text
観測 → 推定器: 倍率 α → α − 1（done の lane は 0） → 内的報酬スケーラ → β を掛ける → Replay
```

- `return_std` は、lane ごとに割引和 R = γR ＋ r を持ち、Run の最初からの R すべての標準偏差 σ で r を割る。その intake の値で統計を更新してから割る（P29）。
- warm-up は、Run の最初の一定の件数だけ内的報酬を 0 にする期間である。件数は採点した観測の数で、既定は 12,800 件（128 lane で 100 intake）、Atari の `run.@rnd` は 200,000 件である（[agent.txt](../../apps/runner/config/agent.txt) 45 行、[Atari.txt](../../apps/runner/config/Atari.txt) 11 行）。warm-up の間、推定器は倍率 1 を返すので、スケーラには 0 が入り、その 0 も統計の件数に数える（P30）。
- いまの設定ファイルと bat に、内的報酬スケーラを動的にした構成は無い（P32）。

**出だしの大きさの計算（Run では確かめていない）。** 誤差で割らない式と `return_std` を今の作りのまま組んだときの、スケーラを通した値（β を掛ける前）を計算した。置いた前提は、γ = 0.99、128 lane、warm-up のあとの誤差 e がどの lane でもどの intake でも同じ、の 3 つである。warm-up 明けから t 回めの intake の割引和は、どの lane でも R_t = e ×（1 − γ^t）÷（1 − γ）になる。統計は、warm-up の間の 0（warm-up の件数ぶん）と、R_1〜R_t（各 128 個）から作る。値は e に依らない。

| 場合 | 最初の intake | 4 回め | 10 回め |
|---|---|---|---|
| warm-up 200,000 件の明け（今の作り） | 約 40 | 約 7 | 約 2 |
| warm-up 12,800 件の明け（今の作り） | 約 10 | 約 2 | 約 0.6 |
| warm-up 0 の Run の最初 | 誤差 ÷ lane の間の誤差のばらつき（未測定） | 約 0.9 | 約 0.4 |
| D3 の形で warm-up 16,384 件の明け | 約 0.05 | 約 0.05 | 約 0.05 |

- warm-up 明けが大きいのは、統計が 0 ばかりで、標準偏差が小さいからである。最初の intake では、統計は 0 が warm-up の件数ぶんと e が 128 個で、標準偏差は e × √(p(1 − p))、p = 128 ÷（warm-up の件数 ＋ 128）になる。
- warm-up 0 の最初の intake は、統計に 128 個の値しか無い。標準偏差は lane の間の誤差のばらつきになる。Run の最初は全 lane がほぼ同じ画面なので、ばらつきが小さく、非常に大きな値になりうる。2 回めは約 2 である。
- 最後の行は、warm-up の間も実際の e を統計に入れた場合（D3）である。128 intake ぶんの R_1〜R_128 の標準偏差は約 20 × e で、論文の公開実装が最初の rollout で作る統計（P31）と同じ大きさになる。
- 誤差が一定という前提なので、落ち着いたあとの値は目安である。大きな内的報酬が付いた遷移は Replay に残り、Q の学習でくり返し使われる。実害の大きさは測っていない。

### 5.4 構造図

`DefaultDQNAgent::RecordNetworkStructure` は、primary の Network と、skip が有効なときの補助 Network の `structure`・`detail` を記録する。tag は `net.structure`・`net.detail`・`skip.net.structure`・`skip.net.detail` で、Run の `dot/` に、tag をファイル名にした `.dot` を書く（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 468〜483 行、[metrics_logger.cpp](../../src/core/base/src/metrics_logger.cpp) 845〜852 行）。RND の target と predictor は記録しない（P14）。

### 5.5 性能

Atari の主な構成（128 lane、replay ratio 8）で、RFD0130 の実装のあとの実行体では、RND 全体が 1 intake あたり 5.999638 ms（採取した区間の 5.594605%）、うち採点が 2.167598 ms、補助学習が 3.624247 ms だった（既定の `learn_ratio` 0.25・32 件で、1 intake に 1 回）。CPU 側の区間の時間である（[RFD0130 の実装メモ](0130_rnd_estimate_statistics_perf_20impl.md) の「主の再採取と GridMaze 前後の 3 本の Tracy」の行）。割り算から、1 intake は約 107 ms である。

### 5.6 今ある「ためる仕組み」

| | フレーム積層の履歴 | Replay の観測の置き場 | Replay の n-step の待ち行列 |
|---|---|---|---|
| 場所 | `DictFrameStacker`（[stacker.cpp](../../src/core/base/src/stacker.cpp) 21 行） | `ReplayExperienceStorage`（[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp) 525 行） | `ExperienceQueue`（[replay_experience.hpp](../../src/core/base/include/anet/replay_experience.hpp) 28 行） |
| 入れるもの | 観測の tensor | 観測・info・行動の tensor | 報酬や終了の印などの小さな記録 |
| 形 | lane × 時間。時間は積層の数 | lane × 時間。時間は lane あたりの容量 | lane ごとの列 |
| 確保 | 最初の書き込みで形を知って確保 | 同じ | 都度 |
| 書き込み | 全 lane を 1 回で。位置は全 lane 共通 | lane を 1 つずつ。位置は lane ごと（終端のダミーでずれる） | lane ごとに末尾へ |
| 読み出し | 全体を古い順に並べて返す | 抽出の番号で拾う（積層と n-step の窓つき） | 先頭から数件を見て取り除く |
| 固有のもの | リセットした lane を 1 フレームで埋める | ダミーの slot、世代つきの key、抽出できる範囲の管理、PER | n-step の確定 |

- tensor の置き場 2 つは、「キーごとに lane × 時間の tensor を最初の書き込みで確保し、時間の軸に書き、番号で読む」ところが同じである。書き込み位置の持ち方（全 lane 共通か lane ごとか）と読み方は違う。
- n-step の待ち行列は、tensor ではなく小さな記録の列なので、同じ部品にはならない。
- 事例（記憶による）: Stable-Baselines3 は、ためてエポックで回す `RolloutBuffer` と `ReplayBuffer` を共通の基底に載せ、フレーム積層の履歴は別に持つ。TorchRL は Replay の置き場を `Storage` として切り出している。

### 5.7 使われない設定の通知

RND と内的報酬の設定には、今の組み合わせで使われないキーが既定と違うときに知らせる箇所がある（[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 600〜602 行・628〜668 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 172〜176 行）。

| 条件 | 対象のキー | レベル |
|---|---|---|
| `intrinsic.observation = state` | `intrinsic.use_done_mask` | WARN |
| `multiplier_mode = unclipped` | `multiplier_max` | WARN |
| `statistics_mode = variance_feedback` | `warmup_mode`・`error_norm.average_mode`・`error_norm.ema_steps`・`error_norm.use_centering`・`obs_norm.*` | WARN |
| `whiten` で `error_norm.average_mode = cumulative` | `error_norm.ema_steps` | WARN |
| `whiten` で `warmup_steps = 0` | `warmup_mode` | INFO |

出荷の設定と bat に、この WARN が出る組み合わせは無い（`apps` の下の txt と bat を `error_norm`・`warmup_mode`・`multiplier_m`・`statistics_mode` で検索、2026-11-21）。これらは、開発者の発言（P46・P48）と食い違う。本 RFD は、設定の組み替え（D7・D12）の中で、`rnd` の下の通知（表の下の 4 行）を無くす。`intrinsic.use_done_mask` の通知と、決まりそのものの見直しは、別の課題にする（§1.4、§10 の U2）。

### 5.8 今の設定キーの形と、書いている箇所

今のキー（`DefaultDQNAgent.intrinsic.` を省く。[agent.txt](../../apps/runner/config/agent.txt) 36〜55 行）:

| 段階 | 今のキー | 使うモード |
|---|---|---|
| 統計 | `rnd.statistics_mode`（`whiten`／`variance_feedback`） | — |
| | `rnd.obs_norm.*` | whiten は全部。`variance_feedback` は `clip_range` だけ |
| 誤差 | `rnd.error_norm.average_mode`・`ema_steps`・`use_centering` | whiten だけ |
| 倍率 | `rnd.multiplier_mode`（`clipped`／`unclipped`）・`multiplier_max` | どの方式でも |
| warm-up | `rnd.warmup_steps` | どの方式でも |
| | `rnd.warmup_mode`（`learn_predictor`／`collect_errors`） | whiten だけ |
| 補助学習 | `learn_ratio`・`learn_batch_size` | — |

**warm-up が束ねているもの。** 1 つの件数 N が、2 つの待ちを切り替える（P59、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 176〜180 行・303〜306 行、[212](../design/212_rnd_lifelong_novelty.jp.md) §3.5）。どの形でも、warm-up の間の倍率は 1 で、内的報酬は 0 である。

| 今の設定 | 誤差を誤差の統計に入れ始めるまで | predictor が学び始めるまで |
|---|---|---|
| whiten・`learn_predictor`（既定） | N 件待つ | 待たない |
| whiten・`collect_errors` | 待たない | N 件待つ |
| `variance_feedback` | 待たない | N 件待つ（lane ごとに判定する） |

**書いている箇所の数（2026-11-21、`git grep -n -E`、追跡しているファイルの全体）。** 検索語は、`statistics_mode`・`multiplier_mode`・`multiplier_max`・`warmup_mode`・`error_norm`・`intrinsic.rnd.warmup_steps`・`rnd.obs_norm`・`learn_ratio`・`learn_batch_size` である。

| 置き場 | ファイル | 行 | 扱い |
|---|---|---|---|
| 設定（agent.txt・Atari.txt・DropMerge.txt） | 3 | 18 | 直す |
| bat（11・12・13_batch_run.bat） | 3 | 18 | 直す |
| コード | 6 | 83 | 直す |
| テスト | 2 | 148 | 直す |
| RFD0194 の検証資材の期待の一覧（`src/core/base/testdata/rfd0194/expected.jsonl`） | 1 | 5 | 直す（D50） |
| 過去の検証の道具（`src/core/base/testdata` の rfd0116・rfd0121・rfd0123・rfd0127・rfd0128・rfd0130） | 13 | 47 | 説明に 1 行足す（D50） |
| 設計文書（jp） | 2 | 34 | 直す |
| wiki | 5 | 12 | 取り込みと点検が直す |
| 設計文書（en） | 2 | 34 | `anet-translate-docs` で作り直す |
| RFD・実装計画 | 15 | 228 | 過去の記録。書き換えない |
| 実験記録 | 78 | 690 | 過去の記録。書き換えない |
| 課題 | 8 | 12 | 過去の記録。書き換えない |
| 合計 | 138 | 1,329 | |

扱いが「直す」の 6 行（設定、bat、コード、テスト、RFD0194 の検証資材の期待の一覧、設計文書の jp）は、合わせて 17 ファイル・306 行で、実装で書き直す対象である。過去の検証の道具と wiki を足すと、35 ファイル・365 行になる。`docs/adr`・`reports`・`CONTEXT.md` には無い。`src` の中で、相対キーだけを書いた行（`warmup_steps` だけの形など）は、この検索語では数えていない。当たるファイルは変わらない（`src` で RND のキーを書くのは、コード 6 とテスト 2 の 8 ファイルである）。`reward_scaler.` を書いている箇所は、`apps`・`src`・設計文書（jp）で 662 行ある。

## 6. 確定事項

設定キーは `DefaultDQNAgent.intrinsic.` を省いて書く。キーの全体は §8.1 にある。

### 6.1 契約（グリルで開発者が選んだ 11 件）

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | **補助学習の形。** rollout ぶんの観測をためてから、エポックで学ぶ形を足す。学習を rollout の終わりにまとめて行う形を基本にし、次の rollout の間に散らす形も設定で選べる。既定は今の形（その intake の観測だけで学ぶ）のままである。Replay から引く形は作らない | 開発者の選択（推奨も同じ）。P3 の論文の量と学び方が、この形でそろう。性能の評価を論文に頼るので、学び方の差を残さない（§12 の発言 3）。P24 から、NGU の対応に引きずられずに決める。P25 から、Replay から引く形は PER の偏りを受け、論文の形に寄せる目的に合わない。P21 から、まとめて行う形は rollout ごとに学習が続けて入り、その間 Actor の同期が待つので、散らす形を選べるようにする。依存: P3・P21・P24・P25、G3 |
| D2 | **置き場の共通化の範囲。** lane × 時間の tensor をためる汎用の部品を新しく作り、RND と無関係な名前で共通の層に置く。使うのは、今回の rollout の置き場だけである。フレーム積層の履歴と Replay の観測の置き場は載せ替えない。載せ替えるときに足す操作と足りないものは、§8.5 の対応表に書く。「ためる仕組みの統合整理」は課題として残す（§10 の U1） | 開発者の選択（推奨も同じ）と指示。開発者は、ためる仕組みを汎用にして共通化したいと述べた（§12 の発言 4）。P26 から、共通にできる芯は薄い。P27 から、載せ替えは全 Run の経路に触り、RND と無関係な Run の同一性と性能の確かめが要る。P28 から、Replay の置き場は特に別の RFD にする。依存: P26・P27・P28 |
| D3 | **割らない式の待ちの間の、内的報酬スケーラの統計。** 割らない式では、待ち（`rnd.error_norm.none.warmup_steps`）の間も、推定器は式どおりの倍率（1 ＋ e）を返す。呼び出し側は、その値を内的報酬スケーラに通して統計を育て、スケーラを通したあとの値を 0 にする。論文の形は、待ちを rollout 1 回ぶん（128 lane で 16,384 件）にして使う。誤差を標準化する式と `variance_feedback` の動きは変えない | 開発者の選択（推奨も同じ）。P30・P33 から、今の作りでは待ちの間の 0 で統計の標準偏差が小さくなり、明けに約 10〜40 の値が出る。P31 から、論文は rollout をまとめて統計に入れるので、最初から落ち着いた大きさである。この形は、最初の rollout 1 回ぶんを統計に使うので、それに最も近い（§5.3 の表の最後の行）。P34 から、warm-up のもともとの役目は割らない式では無くなるので、役目はぶつからない。P32 から、動的な内的報酬スケーラを使う構成は今は無く、既存の Run に影響しない。不利益は、推定器の口に操作が 1 つ増えることと、最初の rollout 1 回ぶんの内的報酬が 0 になることである。依存: P30〜P34、D4・D8 |
| D4 | **誤差の式のモード。** `rnd.error_norm.mode`（`standardize`｜`none`）を足す。既定は `standardize`（今と同じ）。`none` では倍率を α = 1 ＋ e にする。統計のモードは `whiten` のまま使う。checkpoint には保存せず、再開時の設定に従う | 開発者の選択（推奨も同じ）。P8 と同じ理由で、観測の統計を選ぶキーに誤差の式を混ぜない。P36 から、誤差の正規化は `error_norm` のスコープが持つ。P37 から、観測の正規化は whiten のままなので、統計のモードを `whiten` に保つと名前と中身が合う。列挙にしたのは、式を足すときに値を足せるからである。依存: P8・P36・P37 |
| D5 | **構造図。** RND の target と predictor の Network の構造図の記録を、本 RFD の範囲に入れる（G4）。構造図の変更は、ほかの変更に依らない単位にし、コードと設定のステージの中で先に入れられるようにする（§8.6） | 開発者の選択（推奨も同じ）。P41 の「ついでにまとめて入れる」に沿い、「早めに入れたい」には、ほかに依らない単位にして応える。P42 から、D3 と同じ口に操作を足すので、口を触るのが 1 つの RFD で済む。依存: P41・P42 |
| D6 | **ためてから学ぶ形の量のキー。** ためる形の量（長さ、エポック数、1 経験あたりの学習の件数、1 step の件数）は、今の形のキーと共用せず、今の形と同じ語彙の別のキーで持つ。1 rollout の step 数は「lane の数 × 長さ × 1 経験あたりの件数 ÷ 1 step の件数」で決まる。キーの置き場と名前は D36 による | 開発者の選択（推奨は、論文の表の書き方で持つ形だった）。P48 から、形で意味や適した値が変わる量は、形ごとに別のキーで持つ。今の 2 キーを共用すると、適した値（今の形は 0.25・32 か 1・128、ためる形は 1.0・1,024）も、検査の範囲（lane の数以下か）も、形で変わる。同じ語彙なら、lane の数を変えても 1 step の件数が動かない。依存: P43・P44・P48、D1 |
| D7 | **設定の組み替え。** RND と補助学習の設定キーを、段階ごとのスコープに分ける。スコープは、統計（`rnd.statistics`）、誤差（`rnd.error_norm`）、predictor（`rnd.predictor`）、補助学習（`learn`）である。モードを持つ段階は、モードのキー（`mode`）と、モードの名前の名前空間（`<スコープ>.<モード>.<キー>`）で持つ。モードによって意味や適した値が変わるキーは、被ってもモードごとに持つ。今あるキーも、本 RFD で移す | 開発者が示した形と選択（§12 の発言 8。P58）。P48 の「モードだけ切り替えれば、そのモード向けの設定が使われる」を、形で保てる。本 RFD は、ちょうどこれらのスコープにキーを足すので、足したあとで組み替えると、設定と bat を 2 回直すことになる。規模は P60 のとおり、実装で書き直す対象が 17 ファイル・306 行である。P62 から、スコープの下に `mode` を置く形の前例がある。不利益は、RFD が大きくなることと、bat を実行体の切り替えと同時に直す必要があることである。依存: P48・P58・P60・P62 |
| D8 | **warm-up を、段階ごとの待ちに分ける。** `rnd.warmup_mode` と `rnd.warmup_steps` をやめ、3 つの待ちにする。誤差の統計の待ち（`rnd.error_norm.standardize.warmup_steps`）、predictor の待ち（`rnd.predictor.warmup_steps`）、割らない式の待ち（`rnd.error_norm.none.warmup_steps`）である。内的報酬は、選んだモードに当たる待ちが全部明けてから出る。件数 0 は「待たない」を表す。既定と、今ある 3 つの形（`learn_predictor`、`collect_errors`、`variance_feedback`）の動きは変えない。RFD0123 の D4（warm-up の中身を `warmup_mode` の 2 値で選ぶ）を改める | 開発者の選択（推奨も同じ。§12 の発言 9）。P59 から、1 つの件数が、別々の 2 つの待ちを束ねていた。段階ごとに分けると、段階をまたぐ設定が無くなる（D7）。割らない式の待ちも、束の 3 つ目の値ではなく、その式自身の待ちとして書ける。RFD0123 が心配した「膨らみが再発する組」は、今も件数 0 で書ける形と同じで、新しい危険は増えない。不利益は、`collect_errors` の再現に 2 キーを書くことである。依存: P54・P59、D7 |
| D9 | **clip を、誤差のモードごとに持つ。** 倍率の clip は、列挙のモードにせず、真偽のキー（`use_clipping`）と上限（`multiplier_max`）で持ち、誤差のモードごとの名前空間に置く。倍率だけのスコープは作らない | 開発者の選択（推奨は、倍率を独立の段階のままにする形だった。§12 の発言 10）。P56 から、上限が切る対象の単位が、誤差の式で変わる。clip は「するか、しないか」と上限の値だけなので、P61 のとおり真偽で足りる。モードの中にモードを置かない。別モデルも、割らない式で同じキーが効く形に反対した（§12）。依存: P56・P61、D7 |
| D10 | **ステージは、コードと設定、設計文書の 2 つに分ける。** 設計文書は、コードの実装が済んでから、コードと矛盾しない形で書き直す。コードも設計文書も、本 RFD に沿う。設定の組み替えだけを、別のステージにはしない。設計文書の書き直しの中も、ステージに分けない。書き直しの中身は §8.7 による | 開発者の指示（§12 の発言 11・12）。グリルで、設定の持ち方を問う選択肢を、開発者が続けて読み取れなかった（§1.1 の 4）。同じ絡みを、文書の読み手に残さないために、設計文書をまとめて書き直す。設定の互換はどのみち変わるので、組み替えだけを分けて入れる利点は小さい。依存: D7 |
| D56 | **置き場の部品の機能グループ。** 名前は `tensor_store` とし、機能カテゴリは 314 テンソル処理（Topic #10）に入れる。受け持つ関心、入る見込みのもの、入れないものは、§8.4 の「機能グループの見込み」による | 開発者の選択（推奨も同じ。2026-11-22）。新しい機能グループは、機能カテゴリ（統計・監査・wiki の分類と、commit の Topic）と設計文書の章立てにも波及するので、名前と機能カテゴリを開発者が確かめる決まりにした（[anet-create-rfd](../../.agents/skills/anet-create-rfd/SKILL.md) の「選択肢の示し方」）。部品は、特定の Agent・Env・アルゴリズムに依らない tensor の入れ物で（§8.5）、フレーム積層（`stacker`）と device 転送（`transfer`）と同じ 314 の関心である。依存: D2 |

### 6.2 聞かずに決めた事項（規約・現行コード・決定の組み合わせから導いた）

グリルの中で一覧で示したものと、起草で決定の組み合わせから導いたものである。

| # | 決定 | 理由と依存 |
|---|---|---|
| D11 | **統計と誤差のスコープ。** 2 つは横並びのスコープにする。統計が `variance_feedback` のとき、誤差のスコープ（`rnd.error_norm.*`）は丸ごと使わない。統計のモードの 1 キーで切り替わる | 開発者が示した形（統計と誤差を、別のスコープとして並べる。P58）と P46 による。今も、誤差の正規化のキーは whiten だけが使う（§5.8）。一覧で示し、開発者は番号を挙げなかった |
| D12 | **選ばれていないモードのキー。** 選ばれていないモードの名前空間のキーは、読んで検証するだけにする。WARN も INFO も出さない。`rnd` の下の今ある通知（§5.7 の表の下の 4 行）は、キーを移すのと一緒に無くなる | P46（WARN も INFO も要らない）と P49 による。読んで検証するので、読まれない設定キーの一覧には出ない。依存: D7 |
| D13 | **別名と互換。** 旧いキーの別名や、互換の読みは作らない。設定・bat・テスト・設計文書を、同じ変更で直す。旧いキーを書いた設定は、今ある仕組みによって、読まれない設定キーとして知らされる | [AGENTS.md](../../AGENTS.md) のクリーンブレーク方針と P63 |
| D14 | **不正値。** 列挙の外・値域の外は、選ばれていないモードのキーも含めて、構築時に ERROR にする | AGENTS.md の Fail-Fast 原則と設定値の扱い |
| D15 | **既定の置き場。** キーの既定は [agent.txt](../../apps/runner/config/agent.txt) の既定キー（`?=`）に置く | 今の RND の既定キーの置き場（同 36〜55 行） |
| D16 | **収益の標準偏差で割る処理の持ち主。** 推定器には作らず、内的報酬スケーラの `return_std` を使う | P17（RFD0121 D9） |
| D17 | **割らない式の e。** 今と同じ、出力の次元で平均した二乗誤差である | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 765 行。P2 から、公開実装も同じ定義である |
| D18 | **動きが同じであることの確かめ方。** 同 seed・決定的モードの短い Run の metrics を、着手前に保存した実行体と比べる。checkpoint のバイト列の一致は見ない | P12、[212](../design/212_rnd_lifelong_novelty.jp.md) §5.2 |
| D19 | **Replay から引く形。** 本 RFD では作らない。NGU の構成での predictor の学び方は RFD0173 に残し、確かめた NGU 論文の事実を §3.3 に書いて引けるようにする | D1 から導いた。P25 |
| D20 | **0 にする役。** 割らない式の待ちの間に 0 にするのは、内的報酬を合成する側である。推定器は、次に採点する batch の内的報酬を、呼び出し側が 0 にするかを返す | D3・D8 から導いた。待ちのキーは RND の中にあり、件数を数えているのは推定器である。内的報酬スケーラは、合成する側が持つ（P17）。スケーラの側に待ちを持たせる形は §6.4 |
| D21 | **0 にする範囲。** D3 で 0 にするのは、内的報酬スケーラを通したあとの値である。合わせた報酬に足す分と、価値の分解で内的の Head へ渡す分の両方が 0 になる。ボーナスの診断値（`bonus_mean` など）も 0 を記録する | D3 から導いた。今のコードは、スケーラを通した値を、β を掛ける分と内的の Head へ渡す分の両方に使う（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 124〜126 行） |
| D22 | **組み合わせの検査を当てる条件。** D39 の検査は、`learn.mode = rollout` で、学習が有効（`learner.enabled = true`）のときだけ当てる | D39 と、今の lane の数の検査が、学習の有効なときだけ当たること（P7）から導いた。評価だけの Run は、学習の Run と lane の数が違ってよい |
| D23 | **散らす形の等間隔。** 次の rollout の i 回めの intake（i は 1 から長さまで）を受け取ったあとまでに、1 つ前の rollout の step を、累計で「i × 1 rollout の step 数 ÷ 長さ」の整数部分だけ済ませる。長さの回数めで、全部が済む | D42 から導いた。128 lane の既定では、8 intake に 1 step になる |
| D24 | **ためる形の間の credit。** `learn.mode = rollout` の間、今の形の credit は進めない。checkpoint に保存する値は、読み込んだ値のままである | D45 から導いた。checkpoint の形を保つ |
| D25 | **エラーの文の設定キー。** 本 RFD が足す・直す ERROR の文は、設定キーを完全キーで書く | P50（[RFD0194](0194_config_key_full_names_in_messages_10rfd.md) D12）。順序は §8.9 |

### 6.3 細部（推奨の既定で埋めた）

一覧で 2 回示し、開発者はどちらも「推奨で埋める」を選んだ。1 回めは契約 D1〜D6 のあとの 20 件、2 回めは D7〜D9 のあとの 16 件である。2 回めで改めた 1 回めの件は、改めたあとの形で書く。

| # | 事項 | 決定 | 理由と依存 |
|---|---|---|---|
| D26 | 割らない式の、診断用の誤差の統計 | 診断（`error_mean`・`error_std`）のために、累積の統計を、割らない式の待ちが明けてから更新する。checkpoint は、累積の書式で持つ。`standardize` の名前空間のキーの値は、割らない式の動きに使わない（D12 のとおり、読んで検証はする） | 統計を止めると、診断が NaN になり、checkpoint の形も分かれる。あるモードの名前空間のキーの値を、別のモードの動きに使わない（D7） |
| D27 | 割らない式の Trace | 「標準化した値」は e を出す。待ちの間の内的報酬は 0 を出す | 「倍率から 1 を引いた、clip の前の値」という今の列の意味を保ち、学習側の報酬とそろえる。依存: P13、D3 |
| D28 | clip の名前 | `use_clipping`（真偽）と `multiplier_max`（上限） | 開発者が示した名前は `use_clip` だった。観測の正規化と報酬スケーラの今のキー（`use_clipping`）に、語をそろえた（[impl.md](../quality/impl.md) の「名前」の 3）。起草のあとに示して、開発者が推奨で埋めた。上限は、今の語を残す。依存: D9 |
| D29 | clip の既定 | `standardize` は切る（上限 5）。`none` は切らない。`variance_feedback` は切らない | `standardize` は今と同じ。`none` は論文の形（P2）、`variance_feedback` は Taiga らの再現が切らない。Taiga らのプロファイルは今も `unclipped` を書いているので、出荷の設定の動きは変わらない（§8.2 の意図して変える点の 3） |
| D30 | `variance_feedback` の clip | `rnd.statistics.variance_feedback.` の下に、同じ 2 キーを持つ | 誤差のスコープを使わないモードなので（D11）、自分で持つ |
| D31 | 観測の正規化のキー | 今の `rnd.obs_norm.*` を、`rnd.statistics.whiten.obs_norm.*` へ移す。`variance_feedback` は、`rnd.statistics.variance_feedback.obs_norm.clip_range` だけを持つ | `variance_feedback` が使うのは、今も `clip_range` だけである（[212](../design/212_rnd_lifelong_novelty.jp.md) §3.2） |
| D32 | 平均の取り方 | `average_mode`（`cumulative`｜`ema`）と `ema_steps` を、`rnd.error_norm.standardize.` の直下に並べる。`ema` 用の名前空間は作らない | 値の名前を変えない。モードの中に、モードの名前空間を重ねない |
| D33 | 待ちの名前と既定 | 3 つとも名前は `warmup_steps`。既定は、誤差の統計の待ちが 12,800、割らない式の待ちが 12,800、predictor の待ちが 0。選んだモードが使う待ちのうち、件数が 0 でないものは、始まりと終わりを INFO で 1 回ずつ知らせる | 今の語を残す。何の待ちかは、置き場で分かる。既定の動きが、今（`learn_predictor`、12,800 件）と同じになる。INFO は、今の warm-up の通知を待ちごとに分けたものである |
| D34 | 割らない式で、predictor の待ちも付けたとき | 2 つの待ちは、どちらも Run の最初から数える。predictor の待ちの間は倍率を 1 にし、スケーラには 0 が入る。内的報酬は、両方の待ちが明けてから出す | 学んでいない predictor の誤差で、スケーラの統計を育てない。依存: D8 |
| D35 | `variance_feedback` の待ち | 自分の待ちのキーは持たず、predictor の待ちを使う。待ちの間、倍率は 1、誤差は統計に入れ、predictor は学ばない。lane ごとの判定は、今のままである | 今の `variance_feedback` の warm-up の中身（§5.8）と同じになる。中身を変えない（§1.4） |
| D36 | 補助学習のキー | `learn.mode`（`intake`｜`rollout`）、`learn.intake.ratio`・`learn.intake.batch_size`、`learn.rollout.length`・`epochs`・`ratio`・`batch_size`・`use_spread` | D6・D7 の形。今の `learn_ratio`・`learn_batch_size` は、`learn.intake.` の下へ移す |
| D37 | ためる形の既定値 | `length` 128、`epochs` 4、`ratio` 1.0、`batch_size` 1,024、`use_spread` 偽 | `learn.mode = rollout` だけを書けば、128 lane で論文の表 5 と同じ量になる。依存: P44 |
| D38 | ためる形の値の範囲 | `length`・`epochs`・`batch_size` は 1 以上の整数、`ratio` は有限で、0 より大きく `epochs` 以下 | `ratio` が `epochs` を超えると、1 エポックで全件より多く引くことになる。依存: D14 |
| D39 | 組み合わせの検査 | 1 rollout の step 数（lane の数 × `length` × `ratio` ÷ `batch_size`）が整数で、`epochs` で割り切れること。外れたら、式と値を示して ERROR にする | 値を丸めない。当てる条件は D22 |
| D40 | ためる観測 | `observation` で選んだ、正規化する前の観測を、全 lane ぶんためる。学ぶ時点の最新の観測の統計で正規化する | 今の学習の作り（§5.2）と、論文の「統計を更新してから学ぶ」順（§3.1）に合う |
| D41 | 引き方 | 1 エポックは、rollout の全件を乱数で並べ替えて一巡する回である。そのうち、先頭から `batch_size` 件ずつ、エポックあたりの step 数だけ組を取って学ぶ。残りは、そのエポックでは学習に残さない。1 エポックの中で同じ観測を 2 回引かない。乱数は、今の補助学習の抽選と同じ名前付き seed（`intrinsic.learn`）を使う | 論文の「全体を回し、割合で残す」に当たる（Burda らの形で、1 エポックに 16,384 件のうち 4,096 件を、1,024 件ずつ 4 回）。全件を等分した区画を順に回り、区画ごとに決まった割合を残す書き方と、引かれ方の分布は同じである。公開実装との差は §7.5 に書く。predictor は再帰を持たないので、公開実装のように env ごとに区切る必要は無い。起草のあとに公開実装の区切り方を確かめて示し、開発者は、並べ替える形のままにする推奨で埋めた。依存: P3・P45 |
| D42 | 散らす形 | 真偽の `learn.rollout.use_spread` で選ぶ。真なら、1 つ前にため終わった rollout から、次の rollout をためる間に、等間隔で学ぶ。最初の rollout の間は学ばない。ため終わった rollout とため中の rollout を、同時に持つ（Atari で約 232 MB） | するか、しないかだけなので、列挙のモードにしない（P61）。D1。P22 |
| D43 | 学習が進まない間 | `Learn` が偽を返した step（predictor の待ちの間）は、済んだ step として数え、あとで取り戻さない。成功した回数（`learn_count`）には数えない | 今の形の credit の扱い（§5.2）と同じにする |
| D44 | 今の形のキー | `learn.intake.ratio`・`learn.intake.batch_size` の意味・範囲・lane の数の検査は、今の `learn_ratio`・`learn_batch_size` と同じである。lane の数の検査は、`learn.mode = intake` のときだけ当てる | P48。名前だけを移す |
| D45 | checkpoint | rollout の置き場と、rollout の進み（何回めの intake か、散らす形の学びかけの rollout）は保存しない。読み込んだあとは、空からため直す | 共通の ReplayBuffer も checkpoint に入らない（P15）。既定の長さなら 128 intake でため直る |
| D46 | device | 置き場は、RND に渡す観測と同じ device に置く | 移さずにためられる。Atari の 16,384 件で約 116 MB（P22） |
| D47 | 性能 | 閾値は置かない。Atari で、今の形・ためる形（まとめて）・ためる形（散らす）を同じ実行体で交互に測り、実時間の中央値と、まとめて学ぶ 1 回（1 rollout ぶんの step）の時間を記録する。学習が長いときに分割する仕組みは作らない | §7.5 の見積もり（まとめて学ぶ 1 回は最大で約 1.85 秒）は測っていない。長ければ、散らす形を使う |
| D48 | 構造図の tag と時機 | `intrinsic.rnd.target.net.structure`・`intrinsic.rnd.target.net.detail`・`intrinsic.rnd.predictor.net.structure`・`intrinsic.rnd.predictor.net.detail` を、primary の Network と同じ時機に記録する。`detail` は、primary と同じ可視化の設定で作る。RND の推定器が無い Run では出さない | 設定キーの接頭辞と同じ名前にし、skip の図（`skip.net.structure`）と同じ付け方にする。依存: D5、P14 |
| D49 | 設定と bat の移行 | 設定 3 ファイルと bat 3 本を、コードと同じ変更で新しいキーに直す。動きは変えない。`warmup_mode = collect_errors` を書いている bat の行は、2 つの待ちのキーに書き直す。旧いキーと新しいキーの対応を、[CHANGELOG.md](../../CHANGELOG.md) に書く | D13。対応の表は §8.1 |
| D50 | 検証資材 | 過去の検証の道具（`src/core/base/testdata` の rfd0116・rfd0121・rfd0123・rfd0127・rfd0128・rfd0130 の 13 ファイル）は直さない。置き場ごとの説明に、「設定キーは RFD0191 より前の名前」と 1 行足す。前後を比べる道具は、RFD0191 用に新しく作る。RFD0194 の検証資材の期待の一覧（`src/core/base/testdata/rfd0194/expected.jsonl`）は、コードと設定のステージで、変わった文と名前に合わせて直す | 過去の道具は、過去の検証の記録として残す。前後の比較は、前の実行体に旧いキー、後の実行体に新しいキーを渡す。RFD0194 の期待の一覧は、今のソースと突き合わせる入力で、ソースが変われば直す決まりである（§8.8） |
| D51 | ステージの中身 | コードと設定のステージは、コード・設定・bat・テスト・前後を比べる道具・RFD0194 の検証資材の期待の一覧・変更履歴を持つ。設計文書のステージは、212・200・150 の jp を持つ（§8.6） | D10。変更履歴は、設定の互換が変わる変更と一緒に入れる。動きが同じことは、コードと設定のステージの終わりに、前後の実行体の比較で確かめる（受入 1） |
| D52 | 設計文書 | 新設しない。[212](../design/212_rnd_lifelong_novelty.jp.md) を §8.7 の構成に書き直す。置き場の部品は共通の層に置くので、[150](../design/150_replay_buffer.jp.md) の、`DictFrameStacker` を挙げている部品の表に 1 行足し、212 を指す。[200](../design/200_dqn_agents.jp.md) の内的報酬の設定の行も直す | 今ある RND の仕組みの中の変更で、置き場の部品の使い手も今は RND だけである。使い手が増えるときに、置き場の設計文書を起こすかを決める（U1） |
| D53 | 用語 | [CONTEXT.md](../../CONTEXT.md) に「intake」と「rollout（補助学習）」を足す | 設計文書にだけあった語で、設定キーの単位になる（§4） |
| D54 | ADR | 1 本起こす。[ADR 0088](../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md)「RND の設定は、段階ごとのスコープとモード別の名前空間で持つ」 | 被るキーをなぜ分けたかが、あとで読む人に分からなくなる。1 回めの一覧では「起こさない」としていたが、D7 を決めたあとの 2 回めで改めた |
| D55 | 範囲外 | Atari の Run プロファイルと bat の群を足すこと、RND の Head の出力次元の設定キーの置き場、`observation`・`use_done_mask`・`reward_scaler.*` の組み替えは入れない（§1.4） | Atari.txt に群を足すのは、Atari の枠で実験の段取りと一緒に行う別の関心である（RFD0121 の D19 と同じ）。Head の置き場は RFD0155 が決める別の関心である。`reward_scaler` は外的報酬用と共用である（P60）。どれも本当の制約ではない |

### 6.4 選ばなかった形

| 選ばなかった形 | 裁定 | 理由 |
|---|---|---|
| 直近の観測を容量まで貯め、古い順に捨てながら、置き場から毎回引く形（起票時の暫定の既定） | 不採用 | 量・件数・対象は論文とそろうが、predictor が少しずつ進む点が論文と違う。事例も確かめられなかった。学び方の差を残さない形（D1）を選んだ |
| predictor が Replay から引いた観測で学ぶ形 | 範囲外 | §1.4。RFD0173 の未決表の 13 が決める |
| フレーム積層の履歴や Replay の観測の置き場も、同じ RFD で新しい置き場の部品に載せ替える形 | 範囲外 | §1.4。全 Run の Actor の毎 step、Push と抽出に触る。U1 の課題に残す |
| 汎用の部品を作らず、RND の中だけで intake ごとの観測を並べて持つ形 | 不採用 | 開発者の指示による（ためる仕組みを汎用にして共通化する。§12 の発言 4）。部品と、その契約と受入は要らなくなるが、あとで載せ替える先が無い |
| 割らない式の待ちの間も、0 を内的報酬スケーラの統計に入れる形 | 不採用 | 明けに約 10〜40 の値が出る（§5.3） |
| 割らない式の待ちの間は、内的報酬スケーラを通さない形 | 不採用 | 統計が待ちのあとから始まり、出だしの大きな値が残る |
| 内的報酬スケーラの側に待ちを持たせる形（別モデルが、比べるよう求めた形） | 不採用 | 報酬スケーラは外的報酬用と同じ型で、全 Run の外的報酬が通る（P52）。外的報酬を 0 にできるキーができる。定数倍のスケーラは件数を持たない（P53）ので、割らない式と定数倍の組で、出だしの大きな誤差を止められない。待ちを段階ごとに分けたので（D8）、割らない式の待ちは、その式の名前空間に置いた |
| warm-up を `warmup_mode` で束ねたままにする形 | 不採用 | 段階をまたぐ束が残る（D8）。割らない式を、束の 3 つ目の値として足すことになる |
| warm-up の件数を、`rnd` 直下に 1 つ持つ形 | 不採用 | モードを切り替えたときに、前のモード用に書いた件数が残る（D8） |
| 式を真偽のキー（`error_norm.enabled`）で選ぶ形 | 不採用 | 3 つ目の式を足すときに、キーを作り直すことになる |
| 式を、統計のモードの 3 つ目の値で選ぶ形 | 不採用 | 観測の統計と誤差の式が 1 つのキーに混ざる（P8 と同じ混ざり方）。checkpoint が値を持つので、`whiten` と行き来する再開が止まる |
| 誤差のスコープを、`rnd.statistics.whiten.` の下に入れ子にする形 | 不採用 | 誤差が whiten だけのものであることは形で表せるが、キーが深くなる。開発者が示した形は、段階を横並びにする（D11） |
| 選ばれていないキーが既定と違えば、WARN（または INFO）を出す形 | 不採用 | 開発者の指示（P46）。モードを変えたいだけなのに、関係ない設定を同時に直させることになる |
| 倍率を、どの誤差の式にも同じに効く独立の段階のままにする形（1 回めの細部の決定） | 不採用 | 上限が切る対象の単位が、式で変わる（P56）。割らない式に切り替えただけでは、既定の上限 5 が効き、論文の形にならない |
| 割らない式では、倍率を切らないだけにする形 | 不採用 | 誤差に上限を付けたくなっても、付けられない |
| clip を、列挙のモード（`clipped`｜`unclipped`）と、モードの名前空間で持つ形 | 不採用 | モードの中にモードが入る。するか、しないかだけなので、真偽で足りる（P61） |
| ためる形の量を、今の `learn_ratio`・`learn_batch_size` と共用する形 | 不採用 | 形によって、適した値も検査の範囲も変わる（P48）。1,024 を書いたまま今の形に戻すと、構築時に ERROR になる |
| ためる形の量を、論文の表の書き方（ミニバッチ数と、残す割合）で持つ形 | 不採用 | 開発者の選択。lane の数を変えると 1 step の件数が比例して変わり、割合を手で直すことになる。今の形と語彙が分かれる |
| ためる形を、有効フラグ（`learn_rollout.enabled`）のまとまりと、今の形の平らなキーで持つ形（組み替えの前の決定） | 不採用 | 組み替え（D7）で、補助学習も `learn.mode` と、形ごとの名前空間にそろえた |
| 散らす形を、列挙のモード（`step_mode = burst`｜`spread`）で選ぶ形（1 回めの細部の決定） | 不採用 | モードの中にモードが入る。真偽の `use_spread` にした（D42） |
| 全件を等分した区画から引き、区画が割り切れない組を ERROR にする形 | 不採用 | 並べ替えて先頭から取る書き方（D41）と、引かれ方の分布は同じで、割り切れの検査が要らない（別モデルの指摘） |
| RFD0191 が足すキーだけを新しい形にし、今あるキーは今の名前のまま残す形 | 不採用 | 2 つの形が並ぶ。足したキーを含めて、設定と bat をあとでもう一度直すことになる |
| 組み替えを別の RFD にして、先に行う形 | 不採用 | RFD が 2 本になり、本 RFD をもう一度起草し直すことになる |
| 構造図を本 RFD に入れず、別の小さな直しにする形 | 不採用 | 同じ口を、2 つの変更に分けて触ることになる（P42）。早く届ける点は、ほかの変更に依らない単位にして補う |

## 7. 仕様

設定キーは `DefaultDQNAgent.intrinsic.` を省いて書く。

### 7.1 設定の全体：段階とモード

内的報酬は、次の流れでできる。

```text
観測 → [統計] 観測の正規化 → target・predictor → 誤差 e → [誤差] 倍率 α → α − 1 → 内的報酬スケーラ → × β → 報酬
別の流れ: [predictor]・[補助学習] predictor を学ばせる
```

段階ごとに、設定のスコープを 1 つ持つ（D7）。

| 段階 | スコープ | モード | モードごとの名前空間が持つもの |
|---|---|---|---|
| 統計 | `rnd.statistics` | `whiten`（既定）／`variance_feedback` | `whiten`: 観測の正規化の設定。`variance_feedback`: 観測の正規化の上限と、clip |
| 誤差 | `rnd.error_norm` | `standardize`（既定）／`none` | `standardize`: 平均を引くか、平均の取り方、誤差の統計の待ち、clip。`none`: 割らない式の待ち、clip |
| predictor | `rnd.predictor` | なし | predictor の待ち（Network の設定は今のまま `rnd.predictor.net.*`） |
| 補助学習 | `learn` | `intake`（既定）／`rollout` | `intake`: 1 経験あたりの件数、1 step の件数。`rollout`: 長さ、エポック数、1 経験あたりの件数、1 step の件数、散らすか |

- 統計が `variance_feedback` のとき、誤差の式は Taiga らの式に決まり、誤差のスコープは丸ごと使わない（D11）。
- 選ばれていないモードの名前空間のキーは、読んで検証するだけで、動きに効かない。通知も出さない（D12）。
- 次のキーは、今の置き場のままである。`beta`、`observation`、`use_done_mask`、`reward_scaler.*`、`value_mode`、`value.head.*`、`rnd.head.*`、`rnd.target.net.*`、`rnd.predictor.net.*`、`rnd.learning_rate`、`rnd.weight_decay`、`rnd.adam_eps`（D55）。

誤差の倍率は、統計と誤差のモードの組で決まる。

| 統計 | 誤差 | 待ちが明けたあとの倍率 α |
|---|---|---|
| `whiten` | `standardize` | 1 ＋（e − 平均）÷ 標準偏差。`use_centering` が偽なら 1 ＋ e ÷ 標準偏差（今と同じ） |
| `whiten` | `none` | 1 ＋ e |
| `variance_feedback` | （使わない） | 1 ＋ z。z は Taiga らの再帰統計による（今と同じ） |

### 7.2 待ち

件数は、採点した観測の数である。whiten は batch を採点する前の件数で、`variance_feedback` は lane ごとの件数で判定する（今と同じ）。

| 待ち | キー | 待ちの間 | 使うモード |
|---|---|---|---|
| 誤差の統計の待ち | `rnd.error_norm.standardize.warmup_steps`（既定 12,800） | 誤差を誤差の統計に入れない。統計が空なので、倍率は 1 になる | 誤差が `standardize` |
| predictor の待ち | `rnd.predictor.warmup_steps`（既定 0） | predictor は学ばない（`Learn` が偽を返す）。倍率は 1 にする | すべて |
| 割らない式の待ち | `rnd.error_norm.none.warmup_steps`（既定 12,800） | 倍率は式どおり。内的報酬スケーラの統計を実際の値で育て、スケーラを通したあとの値を 0 にする（§7.4） | 誤差が `none` |

- 内的報酬は、選んだモードに当たる待ちが全部明けてから出る（D8）。どの待ちも、Run の最初から数える。
- 件数 0 は「待たない」である。
- 誤差の統計は、誤差の統計の待ちが明けていれば、predictor の待ちの間も更新する。
- 割らない式で predictor の待ちも付けたとき、predictor の待ちの間は倍率が 1 で、スケーラには 0 が入る（D34）。
- 境界の判定は今のままにする。倍率を 1 にするのは、件数が待ちの件数に満たない間である。`Learn` が偽を返すのは、predictor の待ちが 0 でなく、採点した件数が待ちの件数以下の間である。
- 選んだモードが使う待ちのうち、件数が 0 でないものは、始まりと終わりを INFO で 1 回ずつ知らせる（D33）。選ばれていないモードの待ちは、知らせない（D12）。

今の形との対応（N は今の `warmup_steps`）:

| 今の形 | 誤差の統計の待ち | predictor の待ち |
|---|---|---|
| whiten・`learn_predictor`（既定） | N | 0 |
| whiten・`collect_errors` | 0 | N |
| `variance_feedback` | （使わない） | N |

新しく書ける組が 1 つある。predictor の待ちを短く、誤差の統計の待ちを長くすると、「最初は観測の統計だけを育て、次に predictor を学ばせ、最後に誤差の統計と内的報酬を始める」形になる。RFD0123 で先送りにした、2 ステージの warm-up に当たる。本 RFD は、この組を書けるようにするだけで、効果は確かめない。

### 7.3 誤差の式と clip

- `rnd.error_norm.mode = standardize` は、今の式のままである。
- `rnd.error_norm.mode = none` は、倍率を α = 1 ＋ e とする（D4）。e は、今と同じ誤差である（D17）。観測の統計は、今と同じく更新する。診断用の誤差の統計は、累積で、割らない式の待ちが明けてから更新する（D26）。
- clip は、選んだモードの名前空間の `use_clipping` と `multiplier_max` で決まる（D9）。`use_clipping` が真なら、倍率を 1〜`multiplier_max` に切る。偽なら切らず、標準化する式では負の内的報酬も残る（今の `unclipped` と同じ）。
- 既定は、`standardize` が切る（上限 5）、`none` と `variance_feedback` が切らない（D29）。
- Trace の「標準化した値」は、`none` では e を出す（D27）。

今の形との対応:

| 今の形 | 新しい書き方 |
|---|---|
| `multiplier_mode = clipped`、`multiplier_max = L` | 選んだモードの名前空間で、`use_clipping = true`、`multiplier_max = L` |
| `multiplier_mode = unclipped` | 選んだモードの名前空間で、`use_clipping = false` |

### 7.4 割らない式の待ちと、内的報酬スケーラ

- 誤差が `none` で、batch を採点する前の件数が、割らない式の待ちに満たないとき、内的報酬を合成する側は、スケーラを通したあとの値を 0 にする（D3・D20・D21）。
- スケーラは、待ちの間も、実際の値（α − 1。done の lane は今と同じ扱い）で統計を更新する。スケーラの種類（定数倍、`reward_std`、`return_std`）に依らず、同じ順で通す。
- `standardize` と `variance_feedback` では、合成する側は 0 にしない。推定器が、今と同じく倍率 1 を返す。
- Trace の内的報酬は、待ちの間 0 を出す（D27）。Trace は、今と同じく、写した推定器の件数で判定する。
- 待ちを 0 にすると、0 にする期間は無い。最初の数 intake の値が大きくなりうる（§5.3）。検出も抑制もしない。

動き（128 lane、誤差 `none`、スケーラ `return_std`、割らない式の待ち 16,384 件、補助学習 `rollout` の既定）:

| intake | 推定器 | スケーラの統計 | 内的報酬 | predictor |
|---|---|---|---|---|
| 1〜128 | 1 ＋ e を返す | 実際の e で育つ | 0 | 128 回めの終わりに初めて学ぶ |
| 129〜 | 1 ＋ e を返す | 育ち続ける | e ÷ σ × β | rollout ごとに学ぶ |

### 7.5 ためてから学ぶ形

記号: N は lane の数、L は `learn.rollout.length`、E は `epochs`、r は `ratio`、k は `batch_size`。1 rollout の件数は N × L、1 rollout の step 数は S = N × L × r ÷ k、エポックあたりの step 数は S ÷ E である。

- `learn.mode = rollout` では、intake ごとに、RND に渡す観測（`observation` で選んだ、正規化する前の観測）の全 lane ぶんを置き場に足す（D40）。今の形の credit と lane の抽選は使わない（D24）。
- L 回めの intake で、rollout がため終わる。
- **まとめて学ぶ形**（`use_spread = false`、既定）: ため終わった intake の中で、S step を続けて学び、置き場を空にする（D1）。rollout の間、predictor は変わらない。
- **散らす形**（`use_spread = true`）: ため終わった rollout を学ぶ側に回し、次の rollout をためる間に、D23 の間隔で S step を学ぶ。最初の rollout の間は学ばない。ため終わった rollout とため中の rollout を、同時に持つ（D42）。
- **引き方**（D41）: 1 エポックは、rollout の全件（N × L）を乱数で並べ替えて一巡する回である。そのうち、先頭から k 件ずつ、S ÷ E 個の組を取る。各 step は、その 1 組で学ぶ。残りは、そのエポックでは学習に残さない。1 エポックの中で同じ観測を 2 回引かない。エポックをまたぐと、同じ観測を引きうる。1 エポックで学習に残すのは（S ÷ E）× k 件で、全件のうち r ÷ E の割合である。
- 1 step は、今と同じ RND の `Learn` を 1 回呼ぶ。学ぶ時点の最新の観測の統計で正規化する。`Learn` が偽を返した step は、済んだものとして数える（D43）。
- 学習が無効（`learner.enabled = false`）の Run では、今と同じく、ためることも学ぶこともしない。
- checkpoint を読み込んだあとは、空からため直す。読む前にためていた分と、散らす形の学びかけの分は学ばれない（D45）。

128 lane で既定値（D37）なら、1 rollout は 16,384 件、S は 16、エポックあたり 4 step で、1 エポックに 4,096 件（割合 0.25）を 1,024 件ずつ学ぶ。

**論文の表 5 との対応**

| 論文の表 5 | 値 | anet の設定 | 差 |
|---|---|---|---|
| rollout の長さ | 128 | `learn.rollout.length = 128` | なし |
| エポック数 | 4 | `learn.rollout.epochs = 4` | なし |
| ミニバッチ数 | 4 | 決まる（N × L × r ÷ k ÷ E。128 lane の既定で 4） | 公開実装は env の番号の続きで区切り、エポックごとに並べ替えない。anet は要素（lane × 時間）をエポックごとに並べ替えて区切る（P45。D41） |
| predictor に使う割合 | 0.25 | 決まる（r ÷ E。既定で 0.25） | 論文と公開実装は、要素ごとの乱数で残すので、件数が毎回ゆらぐ。anet は、決まった件数を引く（P3・P45） |
| 1 step の件数 | 約 1,000（計算） | `learn.rollout.batch_size = 1,024` | 上の行と同じ |

lane の数を変えると、1 step の件数は変わらず、ミニバッチ数が変わる（256 lane でエポックあたり 8）。

**頻度×単価の見積もり（D47。測っていない）。** Atari の主な構成（128 lane）で、補助学習 1 回（32 件）は約 3.62 ms である（§5.5）。1 回の時間が件数に比例すると見ると、1,024 件の 1 step は最大で約 116 ms になる。

- まとめて学ぶ形: 128 intake に 1 回、16 step で最大約 1.85 秒。その intake の間は学習のロックを持つので、EvalPanel などの Actor の同期が待つ（P21）。128 intake は約 14 秒である。平均では 1 intake あたり最大約 14.5 ms で、今の 3.62 ms から約 11 ms 増える（1 intake 約 107 ms の約 10%）。
- 散らす形: 8 intake に 1 step、最大約 116 ms。平均は、まとめて学ぶ形と同じである。
- 置き場へ足す費用は、1 intake につき 128 件（Atari で約 0.9 MB）の写しである。
- 32 件の実測は、件数に依らない固定の時間を含む。GPU では、件数を増やしても時間はふつう比例より緩く増える。上の値は上限に近い見込みである。

### 7.6 構造図

- RND の推定器があるとき、target と predictor の Network の `structure`・`detail` を、primary の Network と同じ時機に記録する。tag は D48 の 4 つである。記録する時点で、内的報酬の部品はもう作られている（P14）。
- 推定器は、自分が持つ Network の構造図を、役割の名前（`target`・`predictor`）とともに返す。tag の接頭辞（`intrinsic.rnd.`）は、呼び出し側が付ける。

### 7.7 名前のある組み合わせ

| 段階など | 既定の RND（`run.@rnd`） | Taiga らの再現（`run.@rnd_taiga`） | Burda らの形（Atari の枠で組む） |
|---|---|---|---|
| 統計 | `whiten` | `variance_feedback` | `whiten` |
| 誤差 | `standardize`（平均を引く） | （使わない） | `none` |
| clip | 切る（上限 5） | 切らない | 切らない |
| 待ち | 誤差の統計の待ち 12,800（Atari は 200,000） | なし | 割らない式の待ち 16,384 |
| 補助学習 | `intake`（0.25・32） | `intake`（1・1） | `rollout`（128・4・1.0・1,024） |
| 観測 | `next_state`。done の lane は 0 | `state` | `next_state`。done でも 0 にしない |
| スケーラ | 定数倍 | 定数倍 | `return_std` |
| 価値 | `combined` | `combined` | `decomposed`。内的の割引率 0.99、game over で切らない |

Burda らの形の設定の例（コロン記法。Atari の枠で群を組むときの材料で、本 RFD では足さない。D55）:

```text
run.@rnd_burda : DefaultDQNAgent.intrinsic.rnd.error_norm.mode = none
run.@rnd_burda : DefaultDQNAgent.intrinsic.rnd.error_norm.none.warmup_steps = 16,384
run.@rnd_burda : DefaultDQNAgent.intrinsic.learn.mode = rollout
run.@rnd_burda : DefaultDQNAgent.intrinsic.use_done_mask = false
run.@rnd_burda : DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling = true
run.@rnd_burda : DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode = return_std
run.@rnd_burda : DefaultDQNAgent.intrinsic.value_mode = decomposed
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.discount_mode = separate
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.gamma = 0.99
run.@rnd_burda : DefaultDQNAgent.learner.intrinsic.terminal_mode = non_episodic
```

今ある設定の行の書き直し（D49）:

| 今の行 | 新しい行 |
|---|---|
| `run.@rnd : …intrinsic.rnd.warmup_steps = 200,000`（Atari.txt・DropMerge.txt） | `run.@rnd : …intrinsic.rnd.error_norm.standardize.warmup_steps = 200,000` |
| `run.@rnd_taiga : …intrinsic.learn_ratio = 1`・`…intrinsic.learn_batch_size = 1` | `…intrinsic.learn.intake.ratio = 1`・`…intrinsic.learn.intake.batch_size = 1` |
| `run.@rnd_taiga : …intrinsic.rnd.multiplier_mode = unclipped` | `…intrinsic.rnd.statistics.variance_feedback.use_clipping = false` |
| `run.@rnd_taiga : …intrinsic.rnd.statistics_mode = variance_feedback` | `…intrinsic.rnd.statistics.mode = variance_feedback` |
| `run.@rnd_taiga : …intrinsic.rnd.warmup_steps = 0` | `…intrinsic.rnd.predictor.warmup_steps = 0` |
| bat の `…intrinsic.rnd.warmup_mode=collect_errors …intrinsic.rnd.warmup_steps=12800` | `…intrinsic.rnd.error_norm.standardize.warmup_steps=0 …intrinsic.rnd.predictor.warmup_steps=12800` |
| bat の `…intrinsic.rnd.error_norm.average_mode=ema`・`…error_norm.use_centering=false` | `…intrinsic.rnd.error_norm.standardize.average_mode=ema`・`…error_norm.standardize.use_centering=false` |

## 8. 実装契約

### 8.1 設定キーと判定

キーの全体（[agent.txt](../../apps/runner/config/agent.txt) の既定キー。`DefaultDQNAgent.@baseline : intrinsic.` に続けて書く）:

| キー | 型・値 | 既定 | 今のキー |
|---|---|---|---|
| `rnd.statistics.mode` | `whiten`｜`variance_feedback` | `whiten` | `rnd.statistics_mode` |
| `rnd.statistics.whiten.obs_norm.*` | 今の `rnd.obs_norm.*` と同じ | 今と同じ（`clip_range` 5 など） | `rnd.obs_norm.*` |
| `rnd.statistics.variance_feedback.obs_norm.clip_range` | 今の `clip_range` と同じ | 5 | `rnd.obs_norm.clip_range`（共用していた） |
| `rnd.statistics.variance_feedback.use_clipping` | 真偽 | 偽 | `rnd.multiplier_mode`（共用していた） |
| `rnd.statistics.variance_feedback.multiplier_max` | 有限で 1 より大きい | 5 | `rnd.multiplier_max`（共用していた） |
| `rnd.error_norm.mode` | `standardize`｜`none` | `standardize` | （新しい） |
| `rnd.error_norm.standardize.use_centering` | 真偽 | 真 | `rnd.error_norm.use_centering` |
| `rnd.error_norm.standardize.average_mode` | `cumulative`｜`ema` | `cumulative` | `rnd.error_norm.average_mode` |
| `rnd.error_norm.standardize.ema_steps` | 1 以上の整数 | 1,000,000 | `rnd.error_norm.ema_steps` |
| `rnd.error_norm.standardize.warmup_steps` | 0 以上の整数 | 12,800 | `rnd.warmup_steps`（`learn_predictor` のときの件数） |
| `rnd.error_norm.standardize.use_clipping` | 真偽 | 真 | `rnd.multiplier_mode` |
| `rnd.error_norm.standardize.multiplier_max` | 有限で 1 より大きい | 5 | `rnd.multiplier_max` |
| `rnd.error_norm.none.warmup_steps` | 0 以上の整数 | 12,800 | （新しい） |
| `rnd.error_norm.none.use_clipping` | 真偽 | 偽 | （新しい） |
| `rnd.error_norm.none.multiplier_max` | 有限で 1 より大きい | 5 | （新しい） |
| `rnd.predictor.warmup_steps` | 0 以上の整数 | 0 | `rnd.warmup_steps`（`collect_errors`・`variance_feedback` のときの件数） |
| `learn.mode` | `intake`｜`rollout` | `intake` | （新しい） |
| `learn.intake.ratio` | 有限の小数。0 より大きく 1 以下 | 0.25 | `learn_ratio` |
| `learn.intake.batch_size` | 1 以上の整数 | 32 | `learn_batch_size` |
| `learn.rollout.length` | 1 以上の整数（intake の数） | 128 | （新しい） |
| `learn.rollout.epochs` | 1 以上の整数 | 4 | （新しい） |
| `learn.rollout.ratio` | 有限の小数。0 より大きく `epochs` 以下 | 1.0 | （新しい） |
| `learn.rollout.batch_size` | 1 以上の整数（観測の件数） | 1,024 | （新しい） |
| `learn.rollout.use_spread` | 真偽 | 偽 | （新しい） |

無くなるキーは、`rnd.statistics_mode`、`rnd.obs_norm.*`、`rnd.multiplier_mode`、`rnd.multiplier_max`、`rnd.warmup_steps`、`rnd.warmup_mode`、`rnd.error_norm.use_centering`・`average_mode`・`ema_steps`、`learn_ratio`、`learn_batch_size` である。別名は作らない（D13）。

`intrinsic.enabled = true` のとき、どのキーも、選ばれていないモードの名前空間にあっても、読んで検証する（D12・D14）。`intrinsic.enabled = false` のときは、今と同じく、`intrinsic` の下の設定を休止した置き場として扱い、どのキーも読まない（[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 165〜189 行）。

判定は、`intrinsic.enabled = true` のときに、次の表を上から順に当て、最初に当たった行で扱う。どの組み合わせも最後の行までに必ず当たる。「学習が有効」は `learner.enabled = true` を指す。`obs_norm.*` と、学習率などのほかのキーの検証は、今のままである。

| # | 条件 | 扱い |
|---|---|---|
| 1 | `rnd.statistics.mode` が `whiten`・`variance_feedback` のどちらでもない | ERROR（キーと値と列挙を示す） |
| 2 | `rnd.error_norm.mode` が `standardize`・`none` のどちらでもない | ERROR（キーと値と列挙を示す） |
| 3 | `rnd.error_norm.standardize.average_mode` が `cumulative`・`ema` のどちらでもない | ERROR（キーと値と列挙を示す） |
| 4 | `learn.mode` が `intake`・`rollout` のどちらでもない | ERROR（キーと値と列挙を示す） |
| 5 | 待ちのキー（`rnd.error_norm.standardize.warmup_steps`・`rnd.error_norm.none.warmup_steps`・`rnd.predictor.warmup_steps`）のどれかが負 | ERROR（キーと値を示す） |
| 6 | `multiplier_max` のキー（`standardize`・`none`・`variance_feedback` の 3 つ）のどれかが、有限でないか、1 以下 | ERROR（キーと値を示す。今の範囲と同じ） |
| 7 | `rnd.error_norm.standardize.ema_steps` が 1 未満 | ERROR（今と同じ） |
| 8 | `learn.intake.ratio` が有限でない、0 以下、または 1 を超える | ERROR（今の `learn_ratio` と同じ） |
| 9 | `learn.intake.batch_size`・`learn.rollout.length`・`learn.rollout.epochs`・`learn.rollout.batch_size` のどれかが 1 未満 | ERROR（キーと値を示す） |
| 10 | `learn.rollout.ratio` が有限でない、0 以下、または `learn.rollout.epochs` を超える | ERROR（2 つのキーと値を示す） |
| 11 | `learn.mode = intake` で、学習が有効で、`learn.intake.batch_size` が lane の数を超える | ERROR（今の `learn_batch_size` と同じ） |
| 12 | `learn.mode = rollout` で、学習が有効で、1 rollout の step 数（lane の数 × `length` × `ratio` ÷ `batch_size`）が整数でない | ERROR（式と、4 つの値と、計算した step 数を示す。D39） |
| 13 | `learn.mode = rollout` で、学習が有効で、1 rollout の step 数が `learn.rollout.epochs` で割り切れない | ERROR（式と値を示す。D39） |
| 14 | それ以外 | 有効 |

- 行 1〜10 は、選んだモードに依らず当てる（D14）。行 11 と行 12・13 は、補助学習のモードのどちらか一方でだけ当たる（D22・D44）。
- 行 12 の step 数は倍精度で計算し、最も近い整数との差が相対で 1e-6 以下なら、その整数として扱う。小数の表現の誤差を吸収するためで、値を丸めて動かすものではない。
- 黙った代替はしない。step 数を丸める、`batch_size` を lane の数に合わせて変える、といった扱いをしない。
- ERROR の文の設定キーは、完全キーで書く（D25）。

### 8.2 変えないものと、意図して変える点

**意図して変える点**（これ以外の動きは変えない）

1. 設定キーの名前と置き場（§8.1 の表）。
2. `rnd` の下の、使われない設定の通知（§5.7 の表の下の 4 行）が無くなる（D12）。
3. `variance_feedback` で clip のキーを書かないときの既定が、切る（今）から切らない（新）に変わる（D29）。出荷の設定では、Taiga らのプロファイルが切らない形を明示しているので、動きは変わらない。
4. warm-up の始まりと終わりの INFO が、待ちごとの INFO になる（D33）。
5. RND と補助学習の設定の ERROR の文面が、新しいキーの名前になる（D25）。
6. Agent の checkpoint に入る設定の文字列が、新しいキーの名前になる（§8.3）。

**変えないもの**

確定事項と実装契約で「変えない」「今のまま」と書いた箇所について、消える案と、それが本当の制約かは、この表と §1.4 が正本である。

| 変えないもの | 書いた箇所 | 消える案 | 本当の制約か |
|---|---|---|---|
| 既定の設定と、今ある形を新しいキーで書き直した設定の、採点・補助学習の forward と backward の回数と順、乱数の消費、診断、メトリクスの tag と値（G3、P12）。上の意図して変える点と、構造図の `.dot` が RND 有効の Run で 4 つ増えることは除く | D1・D7・D8・D18・D49、§7.2・§7.3・§8.1 | 今の形を消して、ためる形 1 本にまとめる案。組み替えと一緒に、既定の値や、待ちの境界の判定（未満か、以下か）、ほかのキーの検証を見直す案 | 制約ではない。走っている Montezuma の比較の対照を保つための選択である |
| `variance_feedback` の中身（観測と誤差の再帰統計、lane ごとの判定、待ちの間に predictor を止めること） | D3・D35、§1.4 | `variance_feedback` にも割らない式や D3 の扱いを足す案 | 制約ではない。RFD0123 の D8 で別の関心と決めた |
| 誤差を標準化する式の、待ちの間の動き（推定器が倍率 1 を返し、スケーラに 0 が入る） | D3 | どの式でも、待ちの間にスケーラの統計を実際の値で育てる案 | 制約ではない。既存の Run の結果を変えないための選択である |
| 統計のモードを `whiten` のまま、割らない式を選ぶこと | D4 | 統計のモードに 3 つ目の値を足す案 | 制約ではない。観測の統計と誤差の式を 1 つのキーに混ぜないための選択である（§6.4） |
| RND の checkpoint の形（割らない式でも誤差の統計を持つ。rollout の置き場と進みは入れない。credit の値は入れたままにする） | D24・D26・D45、§8.3 | 割らない式のときに誤差の統計を止めて形を分ける案。置き場も保存する案 | 制約ではない。今ある checkpoint を読めるままにするための選択である。置き場は、既定の長さなら 128 intake でため直る |
| 今の形（`intake`）の補助学習の意味・範囲・lane の数の検査 | D44 | 今の形の範囲を広げる案（1 経験あたりの件数が 1 を超えてよい、など） | 制約ではない。今の形の範囲を広げる要求は無い |
| `rnd` の外の、使われない設定の通知（`intrinsic.use_done_mask`）と、通知の決まり | §1.4、§5.7 | 本 RFD の中で、決まりごと見直す案 | 制約ではない。規約（RFD0117 D1・ADR 0071）の見直しを伴う別の関心である（U2） |
| Evaluation Trace の鍵と、標準化する式のときの値 | D27 | Trace の列を式ごとに分ける案 | 制約ではない。列の意味（倍率から 1 を引いた clip の前の値）を保てば、同じ列で読める |
| フレーム積層の履歴と Replay の観測の置き場 | D2、§1.4 | 同じ RFD で載せ替える案 | 制約ではない（§1.4） |
| `beta`・`observation`・`use_done_mask`・`reward_scaler.*`・`value_mode` などの、§7.1 に挙げた今の置き場のままのキー | D55、§7.1 | 内的報酬の設定の全部を、同じ形にそろえる案 | 制約ではない（§1.4）。`reward_scaler` は外的報酬用と共用である |
| Atari の Run プロファイルの中身（キーの名前の移し替えを除く）と、RND の Head の設定 | D49・D55、§8.8、§8.9 | 本 RFD で Run プロファイルを足す案。RND の Head の置き場を先に決める案 | 制約ではない（§1.4） |
| 過去の検証の道具（`src/core/base/testdata` の rfd0116〜rfd0130 の 6 つの置き場）が書く設定。RFD0194 の検証資材の期待の一覧は、これに入らない（直す） | D50、§8.8 | 過去の道具を、新しいキーに書き直す案 | 制約ではない。過去の検証の記録として残す選択である。前後の比較は、新しい道具で行う |
| キーと値の名前のうち、今の語を使うもの（`multiplier_max`、`warmup_steps`、`average_mode` とその値、`ema_steps`、`use_centering`、`whiten`・`variance_feedback`） | D28・D32・D33、§8.1 | 移すついでに、名前も付け替える案（上限を別の名前にする、待ちごとに違う名前にする、平均の取り方を真偽にする、など） | 制約ではない。bat や過去の記録に出てくる語を、移した先でも同じ語で読めるようにする選択である |

### 8.3 checkpoint

- 統計のモードは、今と同じく、値（`whiten` など）を保存し、読み込み時に設定と違えば止める。キーの名前は保存していない。今ある checkpoint は、新しいキーで同じ値を書いた設定で読める。
- 誤差の式のモードは保存しない。再開時に切り替えられる（D4）。割らない式は、累積の書式で誤差の統計を持つ（D26）。書式の識別子が違う checkpoint（直近重みで保存したもの）は、今の書式違いと同じく、割らない式の設定では読めない。
- rollout の置き場と rollout の進みは保存しない（D45）。credit・抽選の乱数の状態・累計の学習回数は、今と同じ形で保存する。`learn.mode = rollout` の間、credit は進めない（D24）。エポックの並べ替えは、保存している抽選の乱数を使う（D41）。
- 内的報酬スケーラの状態は、今と同じく保存する。待ちの間かどうかは、保存済みの採点した件数から決まる。
- Agent の checkpoint に入る設定の文字列は、`intrinsic.enabled = true` のとき、新しいキーの名前になる。読み込みでは、この文字列を比べない（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 586 行。ログに出すだけ）。

### 8.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| RndNoveltyEstimatorConfig | 変更 | §8.1 の `rnd.*` のキーを、スコープとモードの名前空間ごとに読み、検証する。旧いキーの読み取りと、使われない設定の通知をやめる | 内的報酬（RND） |
| WhitenStatistics | 変更 | 誤差の式のモード（標準化する／割らない）、誤差の統計の待ち、モードごとの clip を扱う。割らない式のとき、誤差の統計で割らずに倍率と Trace の値を作る | 内的報酬（RND） |
| VarianceFeedbackStatistics | 変更 | 自分の名前空間の clip と観測の正規化の上限を使う。待ちは predictor の待ちを使う | 内的報酬（RND） |
| LifelongNoveltyEstimator | 変更 | 操作を 2 つ足す。次に採点する batch の内的報酬を、呼び出し側が 0 にするかを返す操作（`SuppressesBonus`）と、推定器が持つ Network の構造図を役割の名前とともに返す操作（`MakeNetworkGraphs`） | 内的報酬の口 |
| RndNoveltyEstimator | 変更 | 足した 2 つの操作を実装する。0 にするかは、誤差が割らない式で、採点した件数が割らない式の待ちに満たないときだけ真。predictor の待ちで `Learn` を止める | 内的報酬（RND） |
| DefaultDQNAgent の内的報酬の設定（IntrinsicConfig） | 変更 | `learn.*` を読み、§8.1 の表の行 4・8〜13 を検証する。旧い `learn_ratio`・`learn_batch_size` の読み取りをやめる | DefaultDQNAgent |
| IntrinsicRewardCoordinator | 変更 | 推定器が 0 にすると返した batch で、スケーラを通したあとの値を 0 にする（学習側と Trace 側）。`learn.mode = rollout` のとき、intake の観測を置き場へ足し、まとめて学ぶ形・散らす形の時機で補助学習を呼ぶ | DefaultDQNAgent（内的報酬） |
| IntrinsicLearnRolloutBatcher | 新規 | ため終わった rollout について、エポックごとの並べ替えと組を作り、step ごとに引く番号を返す。済んだ step を数える | DefaultDQNAgent（内的報酬） |
| LaneTimeTensorStore | 新規 | キーごとに lane × 時間の tensor を持つ置き場。全 lane ぶんを 1 回で足す、lane と時間の番号で読む、空にする、たまった時間の数を返す（契約は §8.5） | 共通の層（core/base）の新しい機能グループ。RND と無関係な名前で置く |
| DefaultDQNAgent | 変更 | `RecordNetworkStructure` で、推定器の構造図も記録する | DefaultDQNAgent |
| RND の設定を読むテスト用の補助（`ReadRndConfig`） | 変更 | 新しいキーの形で読む | 内的報酬（テスト用の補助） |

口に足す操作を「待ちの間か」ではなく「呼び出し側が 0 にするか」で返すのは、呼び出し側が式の種類を知らずに済むようにするためである。標準化する式と `variance_feedback` は、推定器が自分で倍率 1 を返しており、呼び出し側が重ねて 0 にする必要が無い。

機能グループの見込み（`LaneTimeTensorStore` の置き場。D56）:

| 項目 | 中身 |
|---|---|
| ファイル名と置くライブラリ | `tensor_store`。公開ヘッダ `src/core/base/include/anet/tensor_store.hpp` と、`src/core/base/src/tensor_store.cpp`。使い手の DQN Agent は別のライブラリ（`src/algo/dqn`）にあるので、公開ヘッダに置く |
| 受け持つ関心 | lane × 時間の tensor を、キーごとにためて、lane と時間の番号で読む置き場。特定の Agent・Env・アルゴリズムの名前や鍵に依らない（§8.5） |
| 今回入るもの | `LaneTimeTensorStore` と、読む位置を表す lane と時間の組（`LaneTimeIndex`） |
| あとで入る見込みのもの | 「ためる仕組み」の統合整理（U1、[issue-20261122-4c0af5](../management/issues/issue-20261122-4c0af5.md)）で載せ替えると決めたときに、§8.5 の対応表の「載せ替えるときに足す操作」が入る。上限に達したら古い時間から上書きする書き方、全体を古い順に並べて読む操作、選んだ lane の全時間を 1 つの中身で埋める操作、lane を 1 つずつ lane ごとの位置に書く操作である。載せ替えないと決めたら、今回の 2 つのままである |
| 入れないもの | §8.5 の対応表の「部品の外に残るもの」。積層するキーの選択とリセットの判定（フレーム積層の側）、ダミーの slot・世代つきの key・抽出できる範囲の管理・PER・積層と n-step の窓（Replay の側）と、n-step の待ち行列（`ExperienceQueue`） |
| 近い既存の機能グループと、入れない理由 | `tensor_util`（`TensorDict` と、tensor を扱う関数の集まり。状態を持つ置き場を入れると、関心が混ざる）。`stacker`（フレーム積層。U1 で Replay の観測の置き場がこの部品を使うとき、Replay がフレーム積層のヘッダを読む向きになる） |
| 機能カテゴリ | 314 テンソル処理（Topic #10）。[categories.json](../../reports/tools/categories.json) の 314 の行の `match` に `tensor_store` を足す。フレーム積層（`stacker`）と device 転送（`transfer`）と同じ行である |
| 設計文書 | D52 のとおり。説明は [212](../design/212_rnd_lifelong_novelty.jp.md) の補助学習の節に置き、[150](../design/150_replay_buffer.jp.md) の部品の表から案内する。使い手が増えるときに、置き場の設計文書を起こすかを決める（U1） |

### 8.5 置き場の部品の契約と、載せ替えの対応表

`LaneTimeTensorStore` の契約（本 RFD で作る分）:

- キーと形は、最初に足したときの中身で決まる。あとから足す中身は、同じキー・同じ lane の数・同じ形でなければならない。違えば ERROR にする。
- 時間の数の上限は、作るときに決める。上限を超えて足すと ERROR にする（黙って古いものを捨てない）。
- 番号で読むとき、たまっていない時間の番号や、範囲の外の lane の番号は ERROR にする。
- 空にすると、たまった時間の数が 0 に戻る。確保した領域は使い回す。
- 中身は、足したときの device と型のまま持つ。
- 特定の Agent・Env・アルゴリズムの名前や鍵に依らない。
- 上限に達したら古い時間から上書きする書き方は、載せ替えのときに、作るときに選ぶ形で足す。今回の契約（上限を超えたら ERROR）を選んだ使い手は、そのまま動く。

載せ替えの対応表（本 RFD では行わない。U1 の課題の材料）:

| 既存の置き場 | そのまま使える操作 | 載せ替えるときに足す操作 | 部品の外に残るもの |
|---|---|---|---|
| フレーム積層の履歴（`DictFrameStacker`） | 最初の書き込みで確保する。全 lane を 1 回で足す | 上限に達したら古い時間から上書きする書き方。全体を古い順に並べて読む操作。選んだ lane の全時間を 1 つの中身で埋める操作 | 積層するキーの選択、リセットの判定 |
| Replay の観測の置き場（`ReplayExperienceStorage`） | 最初の書き込みで確保する。lane と時間の番号で読む | lane を 1 つずつ、lane ごとの位置に書く操作。上限に達したら古い時間から上書きする書き方 | ダミーの slot、世代つきの key、抽出できる範囲の管理、PER、積層と n-step の窓 |
| Replay の n-step の待ち行列（`ExperienceQueue`） | なし | 載せ替えない（tensor ではなく小さな記録の列である） | 全部 |

### 8.6 ステージ

| ステージ | 中身 | 順序 |
|---|---|---|
| 1 コードと設定 | §7 と §8.1〜§8.5 の全部を実装する。設定の組み替え（待ちの分離と clip の移動を含む）、構造図、割らない式、ためてから学ぶ形と置き場の部品である。コード、設定（agent.txt・Atari.txt・DropMerge.txt）、bat、テスト、前後を比べる道具と RFD0194 の検証資材の期待の一覧（D50）、変更履歴（D49）を直す | 先に行う |
| 2 設計文書 | 212 を §8.7 の構成に書き直し、200 と 150 を直す（D52）。コードは変えない | ステージ 1 のあと |

- ステージ 1 の中の順は、実装計画が決める。構造図の変更は、ほかの変更に依らないので、先に入れられる（D5）。
- 設計文書は、本 RFD と、実装済みのコードの両方に合わせて書く。2 つが食い違う所を見つけたら、文書を合わせて済ませず、食い違いを先に直す。
- ステージ 1 の終わりからステージ 2 の終わりまでの間、設計文書（212・200）には旧いキーの名前が残る。
- CONTEXT.md と ADR 0088 は、本 RFD の起草で直した。

### 8.7 設計文書 212 の書き直しの中身

書き直した 212 は、次のものを、この順に持つ。今ある節のうち、用語、背景、Network の構造、費用と確認済みの範囲の記述は、中身を保って、この並びに合わせて置き直す。

1. **流れの図。** 観測から報酬までの流れと、predictor を学ばせる流れを 1 枚で示す。段階の名前を図に入れる（§7.1 の流れ）。
2. **段階とモードの表。** 段階ごとに、スコープ、モードのキーと値、モードごとの名前空間が持つキーを示す（§7.1 の表）。今の置き場のままのキーも、どの段階に効くかを示す。
3. **モードごとの動き。** 統計と誤差の組ごとの倍率の式（§7.1）と、clip（§7.3）。
4. **待ちの節。** 3 つの待ちの表、内的報酬が出る時機、境界の判定、待ちごとの INFO（§7.2）。割らない式の待ちと、内的報酬スケーラの統計の育て方（§7.4）。
5. **補助学習の節。** 今の形の刻み（credit）と抽選。ためてから学ぶ形、引き方、論文の表 5 との対応表、置き場の部品（§7.5・§8.5）。
6. **名前のある組み合わせ。** 既定の RND、Taiga らの再現、Burda らの形を、段階ごとの選び方の表と、設定の行で示す（§7.7）。
7. **設定の表。** §8.1 の全キーについて、型・値域・既定・変わるものを示す。選ばれていないモードのキーは読んで検証するだけであることを書く。
8. **構造図。** 記録する tag と時機（§7.6）。

- 旧いキーの名前と、旧いキーからの対応は、212 に書かない。対応は CHANGELOG に書く（D49）。
- 英訳（`212_rnd_lifelong_novelty.en.md`）は実装で直さず、`anet-translate-docs` で作り直す。

### 8.8 移行対象

経路の検索（2026-11-21、追跡しているファイルの全体）の結果は §5.8 の表のとおりで、138 ファイル・1,329 行である。口と関数の名前の検索（`git grep -i -l -F`）は、`RecordNetworkStructure` 8 件、`StandardizeWithoutUpdate` 2 件、`LifelongNoveltyEstimator` 21 件、`EstimateWithoutUpdate(` 5 件、`exceeds num_envs` 5 件である。扱いは次のとおり。

- コード（6 ファイル）: [intrinsic_rnd.hpp](../../src/core/base/src/intrinsic_rnd.hpp)・[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)（設定の読み取り、待ち、式、clip、口の 2 つの操作）、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp)（補助学習の設定、0 にする扱い、ためる形、構造図）、テスト用の補助 [src_testing/intrinsic.cpp](../../src/core/base/src_testing/intrinsic.cpp)・[include/anet/test/intrinsic.hpp](../../src/core/base/include/anet/test/intrinsic.hpp)（RND の設定を読む補助）。内的報酬の口 [intrinsic.hpp](../../src/core/base/include/anet/intrinsic.hpp) は、操作を足す。口を実装するクラスは `RndNoveltyEstimator` の 1 つである（P42）。置き場の部品は、新しいファイルで足す。
- テスト（2 ファイル）: [intrinsic_test.cpp](../../src/core/base/src/intrinsic_test.cpp)、[dqn_intrinsic_test.cpp](../../src/algo/dqn/src/dqn_intrinsic_test.cpp)。旧いキーを書いている断定は、新しいキーに書き直す。使われない設定の WARN・INFO を期待する断定は、通知が出ないことの断定に替える（D12）。`learn_ratio` の 1.01 を拒否する断定と、`exceeds num_envs` の断定は、`learn.intake.*` の断定として残る。置き場の部品のテストは、新しく足す。
- 設定（3 ファイル）: [agent.txt](../../apps/runner/config/agent.txt)（§8.1 の既定キー）、[Atari.txt](../../apps/runner/config/Atari.txt)・[DropMerge.txt](../../apps/runner/config/DropMerge.txt)（§7.7 の書き直しの表）。
- bat（3 ファイル）: [11_batch_run.bat](../../apps/11_batch_run.bat)・[12_batch_run.bat](../../apps/12_batch_run.bat)・[13_batch_run.bat](../../apps/13_batch_run.bat) の、キーを書いている行（§7.7 の書き直しの表）。bat は、実行体を替えるのと同時に替える。旧い実行体で走らせている Run の列は、列の区切りで実行体と bat を一緒に替える。
- 設計文書（jp、3 ファイル。設計文書のステージで直す）: [212](../design/212_rnd_lifelong_novelty.jp.md)（§8.7）、[200](../design/200_dqn_agents.jp.md)（内的報酬の設定の行と、構造図の記録の記述）、[150](../design/150_replay_buffer.jp.md)（部品の表に、置き場の部品の行。D52。検索には当たらない新しい行である）。英訳（`*.en.md`）は実装で直さず、`anet-translate-docs` で作り直す。
- 機能カテゴリ（1 ファイル）: [categories.json](../../reports/tools/categories.json) の 314 テンソル処理の行の `match` に `tensor_store` を足す（D56）。wiki の 314 のページのファイルの一覧は、点検（`anet-lint-wiki`）が作り直す。
- 変更履歴: [CHANGELOG.md](../../CHANGELOG.md) に、§8.1 の「今のキー」と新しいキーの対応を書く（D49）。
- 用語集と ADR: [CONTEXT.md](../../CONTEXT.md)（「intake」「rollout（補助学習）」）と [ADR 0088](../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md) は、本 RFD の起草で直した。
- 過去の検証の道具（13 ファイル）: `src/core/base/testdata/rfd0116`・`rfd0121`・`rfd0123`・`rfd0127`・`rfd0128`・`rfd0130` の道具と説明は直さない。置き場ごとの説明に、設定キーが RFD0191 より前の名前であることを 1 行足す（D50）。前後を比べる道具は、`src/core/base/testdata/rfd0191` に新しく作る。
- RFD0194 の検証資材（1 ファイル）: [expected.jsonl](../../src/core/base/testdata/rfd0194/expected.jsonl) は、エラー・WARN・INFO と診断へ設定キーを渡す呼び出しを、ファイル・種類・名前の並びで持つ期待の一覧である。[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) と `default_dqn_agent` を指す行が 31 行ある。本 RFD で、キーの名前が変わり、使われない設定の通知が無くなり、ERROR の文が増えるので、集計との突き合わせの結果が変わる。コードと設定のステージで、集計の道具（`audit_config_messages.py`）の結果に合わせて期待の一覧を直し、変えた規則と根拠を実装計画に書く（同じ置き場の [README](../../src/core/base/testdata/rfd0194/README.md) の決まり）。
- wiki（5 ファイル）: 取り込みと点検が直す。本 RFD の起草で、関係する 3 ページに書き戻した。
- 過去の記録（RFD・実装計画 15、実験記録 78、課題 8）: 書き換えない。過去の Run の設定の写しも、旧い名前のまま残る。
- 互換層: 旧い形の別名や互換の分岐は作らない（D13）。
- 置き場の変更: 追跡しているファイルやフォルダを移す・改名する・分ける・消すことはしない。

### 8.9 ほかの RFD との順序

- [RFD0194](0194_config_key_full_names_in_messages_10rfd.md) は、実装が入っている（2026-11-21 に git log で確認。b72d6e46・7d90a2a7）。エラー・WARN・INFO が示す設定キーを完全キーにし、[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) と [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の文面と、設定の読み口を直した。本 RFD が足す・直す文は、RFD0194 の決まり（D25）で書く。RFD0194 が直した文のうち、`intrinsic_rnd.cpp` の使われない設定の通知と、warm-up の INFO は、本 RFD で無くなるか、文面が変わる（§8.2 の意図して変える点の 2・4）。RFD0194 の検証資材の期待の一覧は、本 RFD のコードと設定のステージで直す（D50、§8.8、受入 20）。
- [RFD0123](0123_rnd_intrinsic_reward_scale_10rfd.md)（実装済み）の D4（warm-up の中身を `warmup_mode` で選ぶ）と、`error_norm.*`・`multiplier_mode` のキーの置き場は、本 RFD が改める（D7〜D9）。RFD0123 の本文は、過去の記録として書き換えない。
- [RFD0190](0190_dqn_design_doc_alignment_10rfd.md) は、設計文書 200 のコードマップに内的報酬の行を足した（[200](../design/200_dqn_agents.jp.md) §4 の「内的報酬」の行。2026-11-21 に確認）。本 RFD の実装で、内的報酬の側に新しいファイルを足すなら、その行に足す。
- [RFD0173](0173_intrinsic_reward_novelty_10rfd.md)（NGU 型。未実装）は、エピソード内の新しさを RND の隣に足す。本 RFD の補助学習は、intake を起点に predictor が学ぶ道の中に閉じ、RFD0173 の採点の場所の論点に影響しない。RFD0173 の未決表の 13（補助学習のデータ量と頻度）は、§3.3 の事実と、本 RFD の 2 つの形（`intake`・`rollout`）を材料にできる。RFD0173 が NGU の構成で Replay から学ばせると決めたら、補助学習のモードに値を足す形で入れられる。RFD0173 が設定キーを書くときは、本 RFD の形（§8.1）に合わせる。
- [RFD0155](0155_config_root_ownership_alignment_10rfd.md)（未実装）は、Head の設定の置き場を決める。本 RFD は RND の Head の設定に触らない（D55）。どちらが先でもよい。

## 9. 成功の測り方と受入

### 9.1 効果（Atari の枠で確かめる）

効果は §1.3 の仮説 A・B で確かめる。群の組み立ては Atari の枠で決める（D55）。理由は、別の関心（Atari の実験の段取り）だからである。次は群を組むときの材料で、決定ではない。

- **読む指標。** 評価側（Greedy と ε 0.01）の点、Montezuma なら 2500 点の割合と訪れた部屋の種類数。学習側では `56_agent_in/05_bonus_pos_ratio`（内的報酬が正になる割合）、`01_bonus_mean` の時間の変わり方、`12_error_std`、`31_predictor_loss`。
- **読み方の注意。** ためてから学ぶ形（まとめて学ぶ）では、`31_predictor_loss` は rollout に 1 回、その rollout の最後の step の値に変わる（診断値は、最後に学んだ step の値だけを持つ。[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 880 行）。`32_learn_count` も rollout ごとにまとめて増える。今の形と同じ細かさでは比べられない。
- **seed。** 群ごとに 2 本以上にし、同じ構成の seed のあいだの幅を超える差だけを読む。

G5（設定の組み替えと、設計文書の書き直し）の成功は、動きが変わらないこと（受入 1）、旧いキーが残らないこと（受入 2）、設計文書が §8.7 の構成を持ち、開発者が読んで分かること（受入 18）で測る。

### 9.2 正しさの受入

「テスト」と書いた項は、単体か結合のテストの合否で確かめる。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 動きが同じであること | 着手前に保存した実行体に旧いキーの設定を渡した Run と、実装後の実行体に、同じ内容を新しいキーで書いた設定を渡した Run で、metrics の全 scalar 点が一致する。構成は 5 つ。RND 無効、whiten の累積（既定）、whiten の直近重み、`variance_feedback`、`collect_errors` に当たる形（誤差の統計の待ち 0、predictor の待ちが今の既定の件数）。条件は [RFD0130 の検証資材](../../src/core/base/testdata/rfd0130/README.md) の機能の比較と同じにする。LunarLander の既定の連鎖、seed 1、決定的 backend、100,000 experience（exp step の軸）、torch thread 1、評価と描画・画像出力を止める。`variance_feedback` は、Taiga らのプロファイルの 8 値を明示する。比べるのは全 tag の全点の値で、時間と資源の 3 つの prefix だけを除く（実行ごとに変わるため）。コードと設定のステージの終わりに確かめる。新しいモード（割らない式、ためてから学ぶ形）は、どの構成でも選ばない |
| 2 | 旧いキーが残っていないこと | 下の「旧いキーの検索語」の 11 個を、固定文字列として、追跡しているファイルで検索し、0 行である。範囲は、コードと設定のステージの終わりには、`apps/` の下と、`src/` の下である。ただし、`src/core/base/testdata/` の、過去の検証の道具の 6 つの置き場（rfd0116・rfd0121・rfd0123・rfd0127・rfd0128・rfd0130）と、RFD0191 用の道具の置き場（rfd0191）は除く。設計文書のステージの終わりには、それに `docs/design/*.jp.md` を足す。過去の検証の道具は、旧いキーのまま残すので除く（D50）。RFD0191 用の道具は、前の実行体に渡すために旧いキーを書くので除く。RFD0194 の検証資材の期待の一覧は、範囲に入る（直したあとは、旧いキーの名前を持たない）。設定の型の欄の名前は、キーの名前と対になっているので、変数の名前にも検索語を残さない。この検索は、接頭辞を省いて書いた旧いキー（`rnd` の直下に書く `warmup_steps`・`multiplier_max`・`obs_norm.*`）には当たらない。新しいキーも同じ語で終わるので、検索では見分けられない。テストと設計文書の中のこの形は、受入では確かめない。実装のレビューで、テスト 2 ファイルとテスト用の補助、212・200 の本文を読んで確かめる |
| 3 | 旧いキーを書いた設定 | 既定の設定に `DefaultDQNAgent.intrinsic.rnd.warmup_steps = 200,000` を足して RND 有効の Run を起動すると、Run の構築の終わりの、読まれない設定キーの一覧に、そのキーが出る。待ちの件数は既定のままで、旧いキーは別名として読まれない |
| 4 | 待ち | テストで、次のとおりになる。(a) 誤差の統計の待ち N・predictor の待ち 0: 件数が N に満たない間、倍率は 1、誤差の統計は空、`Learn` は真を返す。(b) 誤差の統計の待ち 0・predictor の待ち N: 件数が N に満たない間、倍率は 1、誤差の統計は進む。`Learn` は、採点した件数が N 以下の間、偽を返す。(c) `variance_feedback`・predictor の待ち N: (b) と同じで、倍率は lane ごとの件数で判定する。(d) 誤差の統計の待ち N1・predictor の待ち N2（N2 < N1）: 件数が N2 に満たない間は `Learn` が偽で誤差の統計は空、N2 から N1 の間は `Learn` が真で誤差の統計は空で倍率は 1、N1 からは誤差の統計が進む。(e) どの待ちも 0: 最初の batch から誤差の統計が進み、`Learn` が真を返す |
| 5 | clip | テストで、選んだモードの名前空間の `use_clipping` が真のとき、倍率が 1〜`multiplier_max` に切られ、偽のとき切られない。`standardize`・`none`・`variance_feedback` のそれぞれで確かめる。選ばれていないモードの名前空間の `use_clipping`・`multiplier_max` を変えても、倍率は変わらない |
| 6 | 割らない式 | 既知の誤差を渡すテストで、`none` の倍率が 1 ＋ e になる（既定は切らない）。`none.use_clipping = true` なら 1〜`multiplier_max` に切った値になる。割らない式の待ちの間も、同じ値を返す。診断の `error_mean`・`error_std` は、待ちが明けてからの誤差の累積の平均と標準偏差に等しい |
| 7 | 0 にするかを返す操作 | テストで、`none` のとき、採点した件数が割らない式の待ちに満たない間は真、以後は偽を返す。待ちが 0 なら、最初から偽を返す。`standardize` と `variance_feedback` では、待ちの間も偽を返す |
| 8 | 割らない式の待ちの間の統計 | `none`・`return_std`・割らない式の待ちありのテストで、待ちの間、合わせた報酬に足す内的報酬と、価値の分解で内的の Head へ渡す値が、どの lane でも 0 になる。その間の内的報酬スケーラの標準偏差の診断値が、同じ観測の列を待ち 0 で流したときの値と一致する。明けの最初の batch の内的報酬が、β ×（e ÷ その時点の標準偏差）に一致する |
| 9 | Trace の値 | `none` のとき、統計を変えない口が返す「標準化した値」が e に等しい。割らない式の待ちの間の Trace の内的報酬が 0 で、明けたあとは、同じ観測を学習側で採点したときの値と一致する |
| 10 | ためてから学ぶ形（まとめて学ぶ） | テストで、lane の数 4、`length` 4、`epochs` 2、`ratio` 1.0、`batch_size` 8 のとき（1 rollout 16 件、2 step）、1〜3 回めの intake では `Learn` が呼ばれず、4 回めに 2 回呼ばれる。各回に渡る観測は 8 件で、lane の数を超える。1 エポックの中で同じ観測（lane と時間の組）が 2 回渡らない。4 回めのあと、置き場のたまった時間の数が 0 になる。同じ seed で、渡る観測の並びが一致する。`ratio` 2.0 にすると、4 回めに 4 回呼ばれる |
| 11 | ためてから学ぶ形（散らす） | テストで、受入 10 と同じ値に `use_spread = true` を足すと、最初の rollout（1〜4 回めの intake）では `Learn` が呼ばれない。5〜8 回めの intake までの `Learn` の累計の回数が 0・1・1・2 になる。渡る観測は、1〜4 回めの intake でためたものである |
| 12 | 設定の判定 | §8.1 の判定の表の行 1〜13 のそれぞれについて、その行に当たる設定で構築が ERROR になり、文に対象のキー（完全キー）と値が入る。行 12・13 は、式に入る値も入る。行 1〜10 は、選ばれていないモードの名前空間のキーでも ERROR になる。行 11 に当たる設定で `learn.mode = rollout` にすると、構築できる。行 12・13 に当たる設定で、`learn.mode = intake` にするか、`learner.enabled = false` にすると、構築できる。行 1〜10 に当たる値を書いても、`intrinsic.enabled = false` なら構築できる（今と同じ扱い） |
| 13 | 通知が出ないこと | 次の 5 つの設定で Run を構築したとき、ログに、対象のキーを名指す WARN と INFO が 1 行も無く、読まれない設定キーの一覧にも対象のキーが出ない。(a) `rnd.error_norm.mode = none` で、`rnd.error_norm.standardize.use_centering = false` と `rnd.error_norm.standardize.average_mode = ema`。(b) `rnd.statistics.mode = variance_feedback` で、`rnd.error_norm.mode = none` と `rnd.error_norm.none.warmup_steps = 0`。(c) `learn.mode = intake` で、`learn.rollout.length = 64`。(d) `learn.mode = rollout` で、`learn.intake.ratio = 0.5`。(e) `rnd.error_norm.standardize.use_clipping = false` で、`rnd.error_norm.standardize.multiplier_max = 3` |
| 14 | checkpoint | 着手前の実行体が、既定の設定の Run の終わりに保存した checkpoint を、実装後の実行体が、既定の設定で読める。`learn.mode = rollout` で、rollout の途中で保存して読み込むと、置き場のたまった時間の数が 0 から始まり、抽選の乱数の状態と累計の学習回数は保存前と一致する。`standardize`（累積）で保存した checkpoint を、`none` の設定で読める |
| 15 | 構造図 | RND を有効にした短い Run の `dot/` に、`intrinsic.rnd.target.net.structure`・`intrinsic.rnd.target.net.detail`・`intrinsic.rnd.predictor.net.structure`・`intrinsic.rnd.predictor.net.detail` の 4 つの tag をファイル名にした `.dot` がある（primary の Network の `net.structure.dot` と同じ付け方）。RND を無効にした Run の `dot/` には、この 4 つが無い |
| 16 | 置き場の部品 | テストで、足した中身を lane と時間の番号で読むと、足したときの値が返る。空にすると、たまった時間の数が 0 になる。上限を超えて足す、形の違う中身を足す、たまっていない時間の番号で読む、のそれぞれが ERROR になる |
| 17 | 性能の記録 | Atari の主な構成（128 lane）で、今の形（既定）、ためる形（まとめて学ぶ）、ためる形（散らす）の 3 群を、同じ実行体で順を入れ替えて 3 巡以上測り、群ごとの実時間の中央値を実装計画に記録する。ためる形のキーは既定値にする。あわせて、まとめて学ぶ 1 回（1 rollout ぶんの 16 step）の時間を、別の Run の計測で記録する。閾値は置かない（D47） |
| 18 | 設計文書 | [212](../design/212_rnd_lifelong_novelty.jp.md) の jp が、§8.7 の 1〜8 を、その順に持つ。設定の表のキーが、§8.1 の表のキーと、実装済みの [agent.txt](../../apps/runner/config/agent.txt) の既定キーに一致する。そのうえで、開発者が 212 を読み、段階・モード・待ちの説明で、設定から動きを引けることを確かめる（人が確かめる項。設計文書のステージの終わり。開発者の確かめが済むまで、この項は済みにしない。確かめを受けられないままステージを終えるときは、実装計画に、済んでいないことを書く） |
| 19 | 用語集・ADR・変更履歴 | [CONTEXT.md](../../CONTEXT.md) に「intake」と「rollout（補助学習）」がある。[ADR 0088](../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md) がある。[CHANGELOG.md](../../CHANGELOG.md) に、§8.1 の無くなるキー 11 個と、移した先がある |
| 20 | RFD0194 の検証資材 | リポジトリの根で、RFD0194 の集計の道具を、直した期待の一覧と突き合わせて実行し、終了コードが 0 である（`.\.venv\Scripts\python.exe src/core/base/testdata/rfd0194/audit_config_messages.py --expected src/core/base/testdata/rfd0194/expected.jsonl`）。コードと設定のステージの終わりに確かめる |

旧いキーの検索語（受入 2）:

```text
statistics_mode
multiplier_mode
warmup_mode
learn_ratio
learn_batch_size
intrinsic.rnd.warmup_steps
intrinsic.rnd.multiplier_max
intrinsic.rnd.obs_norm
error_norm.use_centering
error_norm.average_mode
error_norm.ema_steps
```

## 10. 決めること（未決表）

グリルで「後で決める」とした事項は無い。残るのは、範囲外とした 2 件で、どちらも課題に起票した。起草のあとに確かめた 3 件（ためる形の引き方、ステージの分け方、clip のキーの名前）は、開発者の答えを D41・D10・D28 に反映した。

| # | 事項 | 区分 | 扱い |
|---|---|---|---|
| U1 | 「ためる仕組み」の統合整理。フレーム積層の履歴と Replay の観測の置き場を、`LaneTimeTensorStore` へ載せ替えるか。置き場の設計文書を起こすか | 範囲外（課題にした） | 開発者が、課題を残すと指示した（D2）。[issue-20261122-4c0af5](../management/issues/issue-20261122-4c0af5.md) に起票した（2026-11-22）。材料は §5.6 と §8.5 の対応表 |
| U2 | 設定の形と、使われない設定の通知を、RND の外でもそろえること。`observation` と `use_done_mask`、`reward_scaler.*` を、モード別の名前空間にするか。段階ごとのスコープとモード別の名前空間の形を、名前の決まり（[impl.md](../quality/impl.md)）に書くか。[AGENTS.md](../../AGENTS.md) の Fail-Fast 原則の表の「ほかの設定で値が使われない」の行、RFD0117 の D1、ADR 0071 を、開発者の発言（P46・P48）に合わせて見直すか | 範囲外（課題にした） | 2 件に分けて起票した（2026-11-22）。決まりの見直しは [issue-20261122-2fc208](../management/issues/issue-20261122-2fc208.md)、決まりに沿わない今ある設定キーと通知の見直しは [issue-20261122-d9499f](../management/issues/issue-20261122-d9499f.md) である。後者は、開発者の指示で、RND の外の特定の設定に限らず、決まりに沿わない設定の全体を対象にした |

### 10.1 起票時の未決表の 17 行の行き先

| 起票時の # | 事項 | 行き先 |
|---|---|---|
| 1 | 誤差で割らない式の設定キー | D4 |
| 2 | `none` のときの誤差の統計 | D26 |
| 3 | `none` のときの warm-up | D3・D8（暫定の既定から変えた） |
| 4 | `none` のときの Trace | D27 |
| 5 | 直近の観測の置き場から引く補助学習 | D1・D6・D36（形を変えた。暫定の既定の形は §6.4 の 1 行め） |
| 6 | 置き場の容量 | D36・D37（容量のキーは作らず、rollout の長さで持つ） |
| 7 | `recent` のときの範囲 | D38・D39・D44 |
| 8 | 置き場が k 件に満たないとき | 当たる場面が無くなった（ため終わってから学ぶ）。学ばない期間は D42、進まなかった step は D43 |
| 9 | 置き場の保存 | D45 |
| 10 | 置き場の device | D46 |
| 11 | 構造図の記録 | D5・D48 |
| 12 | Head の出力次元の設定キーの置き場 | D55（§1.4） |
| 13 | 効果の確かめの置き場 | D55（§1.4） |
| 14 | 名前と用語 | D28・D33・D36・D53、§8.4 のクラス一覧 |
| 15 | 設計文書 | D10・D52 |
| 16 | ADR | D54（暫定の既定から変えた） |
| 17 | 性能 | D47 |

## 11. issue の対応表

| ID | 題 | area | 根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261119-027698](../management/issues/issue-20261119-027698.md) | RND に生の予測誤差を返す内的報酬の式が無い | 324_dqn, 321_agent_core | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 237〜243 行・297〜312 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 113〜126 行 | D3・D4・D20・D21・D26・D27 | 6・7・8・9・12・13・14 |
| [issue-20261119-4976bf](../management/issues/issue-20261119-4976bf.md) | predictor の補助学習が今の intake の観測だけから引く | 324_dqn, 321_agent_core | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 80〜85 行・142〜176 行、[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp) 177〜182 行 | D1・D2・D6・D22〜D24・D36〜D47 | 10・11・12・13・14・16・17 |
| [issue-20261115-ce0101](../management/issues/issue-20261115-ce0101.md) | NN の構造図に RND の target と predictor の Network が出ない | 324_dqn, 321_agent_core, 317_metrics | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 468〜483 行 | D5・D48 | 15 |
| [issue-20261122-c1744a](../management/issues/issue-20261122-c1744a.md) | RND と補助学習の設定キーが段階とモードの関係を表しておらず、warm-up の 1 つの件数が 2 つの待ちを束ねていて、設計文書 212 から設定と動きを引きにくい | 321_agent_core, 324_dqn | [intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp) 176〜180 行・303〜306 行・600〜602 行・628〜668 行、[agent.txt](../../apps/runner/config/agent.txt) 36〜55 行 | D7〜D13・D28〜D35・D49〜D52 | 1〜5・12・13・18〜20 |

4 行目は、G5（設定の組み替えと、設計文書の書き直し）の課題である。題材の 2 件と取り込んだ 1 件では覆えないので、グリルの中で起こした（2026-11-22）。

## 12. グリルの材料

決定ではない。開発者の発言と、AI が出して退いた案を、あとの再グリルで読めるように残す。

**開発者の発言（2026-11-21）**

1. 「NGU になっても残る形にはしたい。NGU 実装後も現状相当（RND のみ有効）にする設定構成自体は可能に維持するから。」
2. 「RND と NGU で考え方が違う、かつその両立を目指している、という状況を考えると、今は NGU を引きずらず、RND 単体としてどうすべきかを考えるべきですね。多分 NGU 対応後は設定で切り替える形になると思う。」
3. 「性能評価が論文だよりである事からすると C（ためてからエポック）になるけど、実際には PPO じゃないからどっちにしろ単純比較はできない事情も。まあベースは C で、設定で C′（散らす形）にも出来るように。」
4. 「貯める仕組みはできるだけ汎用・共通化してたい。理想は RB 内にあるキューとかスタッキングのキューとかとも共有に出来ると良いな。」そのあと、載せ替えを本 RFD に入れない案を選び、「『貯める仕組み』を統合整理する Issue を残す」と指示した。
5. 「使わない設定が既定と違うなら WARN なんてそんな余計な事は要らない。誤差モードを変えたいだけなのに関係ない設定まで同時に変えないといけないなんて事は想定してない、というかそうならないように気をつけてたはずなんだけどな。」
6. 「INFO ログも要らないと思う」
7. 「有効無効やモードの設定値を変更すると、他の設定の意味がつられて切替わることも想定してなくて、そういう場合はモード毎に別の設定項目にしてます。モードだけ切り替えればそれぞれのモード向けとして意図された設定が使われるように。」
8. 「まずは『段階』に対応して設定のスコープ自体を分ける形に統一したらどうかな？んでモード毎にサブキーを分けて名前空間を分離。多少被る要素あっても割り切り。モード別で調整する。」「もしスコープを跨って相関するものがあれば、仕方ないので rnd 直下とかでグローバルに扱うしかないかな。」例として、統計（`rnd.statistics.mode` と、モードの名前の下位のキー）、誤差（`rnd.error_norm.mode`）、倍率（`rnd.multiplier.mode`）の 3 つを挙げた。そのあと、今あるキーも本 RFD で組み替える案を選んだ。
9. 「2 つの warmup が混ざってる？誤差を統計に入れない warmup と predictor を止める warmup、分離できないんですか？」そのあと、段階ごとの待ちに分ける案を選んだ。
10. 「モード毎に clip の意味が変わるのだからモード毎に入れるべき。ただしモードの中にモードがあると面倒なので、」として、`rnd.error_norm.standardize.use_clip` と `rnd.error_norm.none.use_clip` の真偽のキーを示した。「クリップっていまのところするかしないかとする場合の値だけで、モード化需要はそんなだよね？」
11. 「ちょっとゴタゴタしたし難しいから、design 文書は解りやすく更新したいな。そのために RFD 上でフェーズを分けておきたい。」
12. 「ステージの定義がわからない。どれがコード実装？？コード実装終わってからそれに矛盾しない形で design 文書作るのが良いような。両方 RFD に沿う前提の上で。文書整備自体をステージ分割する必要はありません。コード＆設定と文書で分けておけば十分。どうせ設定互換性変わるし設定組み替えだけステージ分けしなくて良い。」

**AI が出して退いた案と、訂正**

- 補助学習の形は、最初に「直近の観測の置き場」を推した。NGU の対応のときに消すものが少ない、という理由だった。発言 1・2 でこの理由が無くなり、「ためてからエポック」へ推奨を変えた。
- 割らない式の待ちの間の統計の問いで、開発者は「ためる設定なら、本家と同じく warm-up は要らなくなるのでは」と尋ねた。ためるのは predictor の学習に使う観測だけで、報酬はためないこと、anet は intake ごとに報酬を確定することを説明した（§1.4 の「報酬の確定を rollout の終わりまで遅らせる」の行）。
- 式のキーの問いで、AI は「使われないキーが既定と違えば WARN を 1 回」と書いた。Fail-Fast 原則の表の当て方を誤っていた。INFO に直したあと、発言 6 で通知そのものをやめた。
- ためる形のキーの問いで、AI は最初に、量を今の `learn_ratio`・`learn_batch_size` と共用する形を推した。発言 7 を受けて取り下げ、別のキーを持つ 2 案（論文の表の書き方、今の形と同じ語彙）に組み直した。推奨は論文の表の書き方だったが、開発者は今の形と同じ語彙を選んだ。
- 別モデル（Opus）の検討は、起草のあと、D7〜D10 を決める前の草案に対して受けた。報告の中の D 番号は、その時点の草案の番号である。論点は 10 件で、扱いは次のとおり。(1) 割らない式の間の通知の食い違い、(2) 平均の取り方のキーの書き方、(5) 区画の割り切れの検査が要らないこと、(6) 整数かの判定の書き方、(7) 置き場の部品の上書きの書き方と設計文書の案内、(8) 用語集の定義、(9) 前提の抜け 2 件、(10) 診断値の読み方は、指摘のとおり直した。(3) 割らない式でも同じキーで clip する形への反対と、(4) 内的報酬スケーラの側に待ちを持たせる形は、開発者に問うた。(3) は D9 になり、(4) は選ばなかった（§6.4）。
- (4) を問う中で、AI は、待ちの件数の置き場を 3 案で 2 度示した。開発者は「よく分からない」と返し、RND のモードの全体の整理を求め、そのあと発言 8 の形を示した。AI は、発言 8 の形を今の全キーに当てた表と、直す対象の数（35 ファイル・365 行。このときは、説明に 1 行足すだけの検証資材と、取り込みが直す wiki を含めて数えた）を示し、開発者が本 RFD で組み替える案を選んだ（D7）。
- 待ちの件数の置き場（モードごとか、`rnd` 直下か）を問うたところ、開発者は発言 9 を返した。AI は、warm-up の 1 つの件数が 2 つの待ちを束ねていること（P59）を示し、段階ごとの待ちに分ける案を推した。「件数 0 を、待たないことの表し方にする」点は、RFD0123 のグリルでの開発者の発言（待たせるかどうかと、何件待つかは別の関心）に触れて確かめを求めた。開発者は、分ける案を選び、別のフラグは求めなかった。
- clip の問いで、AI は、倍率を独立の段階のままにする形を推した。開発者は、発言 10 のとおり、モードごとに真偽で持つ形を選んだ。発言 10 の例では、割らない式の clip の既定は真だった。AI は、2 回めの細部の一覧で、既定を偽にする推奨（D29）と、例と違うことを示し、開発者は推奨で埋めた。
- キーと値の名前は、仮の名前として示した。clip の真偽のキーは、開発者が示した名前（`use_clip`）を、起草のあとに、今ある語（`use_clipping`）にそろえた（D28）。
- 発言 11 を受けて、AI は、ステージを 5 つ（設定の組み替え、設計文書の書き直し、構造図、割らない式、ためる形）に分け、設計文書の書き直しを組み替えの直後に置いた。開発者は、発言 12 で、コードと設定、設計文書の 2 つに改めた（D10）。
- 起草のあとに、確かめを 3 件示した。ためる形の引き方を公開実装に合わせるか、設計文書のステージの位置、clip の真偽のキーの名前である。開発者は「基本、推奨で良い」と答え、ステージについては発言 12 を述べた。

## 13. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.3（2 つの価値）、§2.4（報酬と観測の正規化）、§3.4（env の数と predictor の実効バッチ）、付録 A.2（Algorithm 1）、表 2・4・5。公開実装は [openai/random-network-distillation](https://github.com/openai/random-network-distillation)。
- Badia ら、[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)、ICLR 2020。§2・§3、付録 A、表 1・3・6。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
- 別モデルの検討の報告の原文は、作業用の置き場（`.scratch/prd191/other-model-20261121-2052.md`）にある。追跡していないファイルである。
