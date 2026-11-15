# PRD130: RND の採点の統計の区間を軽くし、2 つの統計方式を型で分ける 暫定 PRD

> 2026-11-15 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §8 の未決表に置いた。§5〜§7 は §8 の暫定の既定を前提に書き、前提にした行の番号を名指す。行番号は起票時点の作業ツリーの位置で、実装時は関数の名前と近くの記述で探し直す。
> 起点: 2026-11-15 の会話。開発者が、RND の性能改善の課題を、PRD121（価値の分解）を実装したあとのコードを基に入れたいと述べ、題材を §9 の 2 件として暫定の PRD を起こすよう指示した。開発者は、起票のあとに Tracy と Nsight で現状を分析して対応方針を整理し、そのあとでグリルを行うと述べた。直し方の中身について、会話で出た方向は無い。
> 題材: §9 の課題 2 件。
> 関連: [116_rnd_lifelong_novelty_10prd.md](116_rnd_lifelong_novelty_10prd.md)（RND の本体と、一生の新しさの口）、[123_rnd_intrinsic_reward_scale_10prd.md](123_rnd_intrinsic_reward_scale_10prd.md)（誤差の統計の取り方と warm-up、checkpoint の書式）、[127_eval_trace_intrinsic_reward_columns_10prd.md](127_eval_trace_intrinsic_reward_columns_10prd.md)（読むだけの採点と推定器の写し）、[128_dqn_agent_refactor_bundle_10prd.md](128_dqn_agent_refactor_bundle_10prd.md)・[121_extrinsic_intrinsic_value_decomposition_10prd.md](121_extrinsic_intrinsic_value_decomposition_10prd.md)（採点を呼ぶ側の形を変えた PRD）、[924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（新しさの推定を広げる PRD）、[Atari の RND 性能測定](../experiments/default-dqn/atari/2026-11-12_rnd_performance.md)、[RND の設計文書](../design/212_rnd_lifelong_novelty.jp.md)、[リファクタリング観点](../quality/refactor.md)。

## 1. 目的と問題

### 1.1 問題

既定の RND（`statistics_mode = whiten`）を有効にした Atari の Run は、RND なしの 1.661 倍の実時間がかかる。RND の時間の 62.66% は、採点（`Estimate`）の中の統計の区間が占める（§2.1）。この区間は CPU の 1 スレッドで動く。放っておくと、RND を使う実験（Montezuma の β の比較、価値の分解、[PRD924](924_intrinsic_reward_novelty_10prd.md) の NGU）の 1 本ごとの実時間が、この分だけ長いままになる。

同じクラスに 2 つの統計方式（whiten と variance_feedback）が同居し、どちらの組が有効かを表す分岐が、採点・補助学習・読むだけの採点・写し・保存・読み込みに散っている（§2.5）。統計の区間を組み替えると、この分岐の全部に触ることになる。

統計の区間のどの処理に時間がかかっているかは、まだ測っていない（§2.6）。

### 1.2 ゴール

- **G1**: 既定の RND を有効にした Run の実時間が、RND なしに近づく（測り方は §1.5）。
- **G2**: 2 つの統計方式が型で分かれ、方式を 1 つ足すときに触る場所が 1 か所になる。採点と補助学習の共通の前半が 1 つになる。
- **G3**: 統計の区間の内訳を、あとからいつでも Tracy で測れる。
- **G4**: §9 の課題 2 件が片付く。

### 1.3 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| Evaluation Trace を開いている間の、推定器の写し（`Clone`）を軽くすること | 題材の外（[issue-20261114-9468cb](../management/issues/issue-20261114-9468cb.md)）。採点の統計とは別の経路で、開発者は今の遅れを許容とみている | 制約ではない（別の課題で扱える） |
| Taiga 設定（`variance_feedback`、補助学習を 1 件ずつ）の補助学習の費用を下げること | 費用の主因は設定そのもの（1 件ずつの backward と optimizer step で 64.63%）で、統計の区間は 0.34% である（§2.1） | 制約ではない |
| target と predictor の forward と、補助学習の backward を軽くすること | 既定の RND では合わせて 20.70% で、統計と正規化（74.58%）より小さい。Network の形と AMP の話になり、別の関心である | 制約ではない |
| 観測の正規化の部品を使う、RND 以外の呼び出し（DQN の観測の正規化）を変えること | Q の学習の値と診断値に関わる（§8 の 5） | 制約ではない |
| RND の設定キー、メトリクスの名前、checkpoint の書式を変えること | 今ある Run の設定と checkpoint を、そのまま使い続ける（§4） | 制約ではない（クリーンブレークで移せるが、移す理由が無い） |

### 1.4 機構とゴールの対応

新しい機構は足さない。足す部品と区間は次のとおりで、名前は §6.2 と §8 の 3 に置いた。

| 足すもの | ゴールへの寄与 |
|---|---|
| 統計の区間の中の計測の区間 | G3。ステージ 0 の現状分析（§5.2）の前提 |
| 方式ごとの統計をまとめる型と、その 2 つの実装 | G2、G4（0ddffa） |
| 観測の正規化の部品の、統計だけを更新する口 | G1、G4（f9790d）。正規化の結果を捨てる二重計算を無くす |
| 統計の集計と正規化を計算 device で行う経路 | G1、G4（f9790d）。入れるかは §5.2 の結果で決める |

### 1.5 成功の測り方

- 既定の RND の実時間 ÷ RND なしの実時間（同じ経験数、3 巡以上の群ごとの中央値）が、実装前の基準より下がる。暫定の閾値は 1.30 以下（§5.5、§8 の 12）。
- Tracy で、採点の統計と正規化の区間の 1 intake あたりの時間が、実装前より短い。
- 整理のあと、`if (feedback_stats_)` の形の分岐が `RndNoveltyEstimator` に残らない。

## 2. 現状（2026-11-15 にコードで確認）

### 2.1 測定の結果（2026-11-12、PRD121 より前の実行体）

出典は [Atari の RND 性能測定](../experiments/default-dqn/atari/2026-11-12_rnd_performance.md)（条件と比較方法、通常比較の結果、Tracy による内訳）。Montezuma、64 lane、CUDA、Release で、RND なし・既定 RND・Taiga 設定を各 6 本、順を入れ替えて 3 巡した。

| 群 | 89,984 経験の中央値（秒） | なしとの比 |
|---|---:|---:|
| RND なし | 19.019 | 1 |
| 既定 RND（`run.@rnd`） | 31.584 | 1.661 |
| Taiga 設定（`run.@rnd > run.@rnd_taiga`） | 280.616 | 14.754 |

既定 RND の別の Run で、Tracy を 8 秒つないだ。388 intake（1 intake は 64 観測）で、RND 全体は 1 intake あたり 13.097 ms だった。区間の時間は CPU 側の時間で、GPU の kernel の時間ではない。

| 区間 | 1 intake あたり ms | RND の中の割合 |
|---|---:|---:|
| 統計 | 8.206 | 62.66% |
| 正規化（採点と補助学習） | 1.561 | 11.92% |
| target forward（採点と補助学習） | 0.799 | 6.10% |
| predictor forward（採点と補助学習） | 0.770 | 5.88% |
| backward | 0.603 | 4.61% |
| optimizer step | 0.538 | 4.11% |
| 誤差の転送 | 0.246 | 1.88% |
| 入力の転送 | 0.245 | 1.87% |
| 観測の転送・結果の転送 | 0.003 | 0.02% |

- 採点（`Estimate`）は RND 全体の 82.68%、補助学習（`Learn`）は 16.58% だった。正規化は 582 回で、うち 194 回は補助学習の入力の正規化である。
- Taiga 設定では、統計は 0.34% で、optimizer step が 35.90%、backward が 28.73% だった。
- 測った Run は `backend.torch_num_threads = 1`（[common.txt:88](../../apps/runner/config/common.txt) の既定）である。
- 当時の `run.@rnd` は β 0.1、warm-up 12,800 件だった。今の `run.@rnd` は β 0.03、warm-up 200,000 件である（[Atari.txt:9-11](../../apps/runner/config/Atari.txt)）。

### 2.2 採点の処理の流れと、計算する場所

`RndNoveltyEstimator::Estimate`（[intrinsic_rnd.cpp:504](../../core/anet-core/src/intrinsic_rnd.cpp) から）は、1 intake の N 件の観測をまとめて、次の順に処理する。

1. 観測を CPU へ移す（515 行）。
2. CPU で、整数の画素を浮動小数点にし（518〜519 行）、正規化する（522〜528 行）。
3. 正規化した観測を計算 device へ送る（530 行）。
4. target と predictor の forward を行い、誤差を作る（532〜536 行）。
5. 誤差 N 件と、target の出力（N×出力次元）を CPU へ戻し、誤差が有限かを確かめる（538〜546 行）。
6. 統計の区間（547〜580 行。中身は §2.3）。
7. 倍率 α を、入力と同じ device へ返す（582 行）。

設計文書も、観測を CPU へ移して正規化し、誤差の検査と統計を CPU で扱うと書いている（[212](../design/212_rnd_lifelong_novelty.jp.md) の §5.1）。入力は、学習の入口に届く生の観測の最新 1 フレームである（[PRD116](116_rnd_lifelong_novelty_10prd.md) の D5）。

§2.1 の測定では、観測の転送と結果の転送が 1 intake あたり合わせて 0.003 ms だった。この値から、測った構成では観測と α が CPU にあったと読める。コードでは確かめていない（§5.2 の Q3）。

### 2.3 統計の区間の中身

既定（whiten）では、統計の区間で次を CPU で行う。課題 f9790d が起票時に挙げた中身と、今のコードは同じである。

- **α の計算と誤差の統計の更新**（556〜561 行）。誤差の統計の平均と標準偏差で α を作り、N 個の誤差で統計を更新する。
- **観測の正規化統計の更新**（563 行）。`NormalizeAndUpdateStats` を呼び、戻り値は使わない（562 行の `@todo`）。この関数は、統計を更新したあと（[scaler_impl.cpp:553](../../core/anet-core/src/scaler_impl.cpp)）、渡された観測の全体を正規化し直して、clip した割合を数える（同 556〜557 行、425〜434 行）。
- **target の出力の統計の更新**（575 行）。N×出力次元の値で、次元ごとの累積の統計を更新する。
- **診断値の読み出し**（576〜580 行）。target の出力の分散の平均と最大、誤差の平均、α が上限に当たった割合。

統計の更新（`RunningMeanStd::Update`、[scaler_impl.cpp:149](../../core/anet-core/src/scaler_impl.cpp) から）は、batch の平均と分散を、渡された入力の device と精度で計算し（171 行）、結果を CPU の倍精度へ移して（179〜180 行）、累積へ合流する（197〜221 行）。正規化（同 374〜388 行）は、呼ぶたびに、累積の標準偏差と平均を、入力の device の単精度に作り直す（377・383 行）。

variance_feedback では、観測の統計も誤差の統計も、lane の順に 1 件ずつ更新する（[intrinsic_rnd.cpp:177-225](../../core/anet-core/src/intrinsic_rnd.cpp)）。この順は方式の定義である（[PRD116](116_rnd_lifelong_novelty_10prd.md) の D5）。

### 2.4 採点を呼ぶ側（PRD128・PRD121 のあと）

- 採点を呼ぶのは `DefaultDQNAgent::IntrinsicRewardCoordinator::ScoreBonus`（[default_dqn_agent.cpp:110](../../core/anet-core/src/default_dqn_agent.cpp) から）である。α からボーナスを作り、内的報酬用の報酬スケーラを通し（117 行）、価値の分解のために、β を掛ける前の成分も返す（119 行）。
- 同じ関数が、ボーナスの診断値を 6 回の `.item<float>()` で読み出す（122〜127 行）。ボーナスが CUDA にあれば、1 回ごとに同期の転送になる。PRD128 は、これを 1 回の転送にまとめる案を範囲の外に置き、RND の時間の内訳と合わせて測るときに決め直すとした（[PRD128](128_dqn_agent_refactor_bundle_10prd.md) の §1.3、[324 DQN のトピック別まとめ](../wiki/categories/324_dqn.md) の「選ばなかった形」）。
- §2.1 の数値は、PRD121 より前の実行体のものである。価値の分解を入れたあとの基準は取り直す（§5.2 の Q6）。

### 2.5 2 つの統計方式の同居と、前半の重複

- whiten は `normalizer_` と `error_stats_` を使い、variance_feedback は `feedback_stats_` を使う。どちらが有効かは、`feedback_stats_` が値を持つかだけで表される（[intrinsic_rnd.hpp:116-119](../../core/anet-core/src/intrinsic_rnd.hpp)）。
- この分岐は、コンストラクタ（[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp) の 456 行）、写し（478・487・492 行）、`Estimate`（522・549 行）、`EstimateWithoutUpdate`（594・604 行）、`Learn`（633・642 行）、`Save`（691 行）、`Load`（729・739・743 行）にある。課題の起票のあと、PRD127 の読むだけの採点と写しにも同じ分岐が入った。
- `RndErrorStatistics` も、累積と直近重みの 2 組のメンバを持ち、メンバ関数が `use_ema_` で分かれる（同 72・91・96・101・107・116 行）。
- `VarianceFeedbackStatistics::UpdateErrors` は、直前に `NormalizeAndUpdate` が同じ件数で件数を進めていることを前提に、件数を逆算する（同 218 行）。この呼ぶ順の前提は、型にも名前にも出ていない。
- `Estimate`（514〜546 行）と `Learn`（637〜661 行）の前半は、CPU への転送、浮動小数点への変換、正規化、device への転送、2 つの forward、誤差の計算、有限性の検査が同じ並びである。違いは、`Learn` が predictor の forward の前に勾配を有効にする 1 行である。`EstimateWithoutUpdate`（590〜600 行）も同じ並びを持つ。

### 2.6 まだ測っていないこと

- 統計の区間の中の処理ごとの時間（§2.3 の 4 つ）と、正規化の区間の中の処理ごとの時間。
- GPU 側の時間（kernel、転送、同期の待ち）。
- 128 lane での 1 intake あたりの時間。
- PRD121 のあとの実行体での基準。
- ボーナスの診断値の読み出しが、同期の転送になっているか。

## 3. 用語

新しい語は足さない。[CONTEXT.md](../../CONTEXT.md) を変えない。「統計方式」は、設定キー `intrinsic.rnd.statistics_mode` の 2 つの値（`whiten`・`variance_feedback`）を指す。

## 4. 規約と過去の判断

- 毎回の学習の更新に通る経路に、device 間の同期の転送を足さない。足すときは、頻度×1 回あたりの時間の見積もりを書く。計測済みの処理を分けたら、分けたあとの区間にも計測を置く（[実装 品質確認観点](../quality/impl.md) の「計測」）。
- RND の統計の取り方を変えるときは、`warmup_mode = collect_errors` の組、`variance_feedback`、RND 無効の 3 構成で、同 seed・決定的モードの短い Run の metrics checksum を、着手前に保存した実行体と一致させる（[PRD123](123_rnd_intrinsic_reward_scale_10prd.md) の §4 D5・§7.2 の 1）。
- RND の累積と `variance_feedback` の checkpoint は、書式の識別子と中身を変えず、着手前に保存した実行体の checkpoint を同じ設定で読める（[PRD123](123_rnd_intrinsic_reward_scale_10prd.md) の §4 D11）。
- 読むだけの採点を挟んでも、学習へ渡る報酬の列と診断値が変わらない（[PRD127](127_eval_trace_intrinsic_reward_columns_10prd.md) の §7 の 10）。
- 挙動を変えない直しでは、hot path の速さが変わらないことを Run のブレ幅で確かめる。「挙動を変えない」には、ログの文言と WARN の回数を含める（[リファクタリング観点](../quality/refactor.md) の「ガードレール」「運用ルール」）。
- 汎用の部品の局所的な情報だけで結果が決まる処理は、個別の実装の中に書かず、その部品の共通処理として足して利用側から呼ぶ（[AGENTS.md](../../AGENTS.md) の「汎用機構と利用側の責任境界」）。
- 課題 f9790d には、2026-11-14 の課題の整理での開発者の判断が記録されている。入れる時期は、PRD121 の実装のあと、Montezuma で確かめの Run を回す前とする。0ddffa と束ねて 1 回で直す。Run は非決定論の設定で回しているので、RND を有効にした Run の学習の系列が変わることは問題にしない。確かめは、RND 無効の構成で学習の値が変わらないことと、速さの 3 巡とする。本 PRD の未決表は、この記録を理由に引くが、再グリルで開き直す。
- 速さの比較は、PRD116 の採取器（[run_performance_atari.py](../../core/anet-core/testdata/prd116/run_performance_atari.py)）で、群の順を入れ替えて 3 巡以上回し、群ごとの中央値で比べる（[PRD116 の検証資材](../../core/anet-core/testdata/prd116/README.md) の「性能測定」）。

性能の区分（リファクタリング観点の「ガードレール」）は次のとおり。

| 課題 | 区分 | 確かめ方 |
|---|---|---|
| f9790d | 改善（intake ごとの採点の経路） | 速さを 3 巡の中央値で確かめる |
| 0ddffa | 中立（同じ経路） | 速さをブレ幅で確かめる。同 seed の前後比較 |

## 5. 進め方（§8 の暫定の既定を前提）

### 5.1 ステージ

| ステージ | 中身 | 値 |
|---|---|---|
| 0 | 現状分析。計測の区間を足し、Tracy と Nsight で §5.2 の問いに答える。対応方針を整理して、再グリルに入る | 変えない |
| 1 | 統計方式の整理（0ddffa。§5.3） | 変えない |
| 2 | 採点の統計の区間を軽くする（f9790d。§5.4） | 候補による（§5.4 の表） |

順は §8 の 1、ステージ 0 の範囲は §8 の 2 による。ステージ 0 の計測の区間は、Q1・Q2 の前提なので、再グリルより前に入れる想定で書いた（§8 の 3）。

### 5.2 ステージ 0: 現状分析で確かめること

Tracy は、PRD116 の採取器の `profile` と同じ条件（Montezuma、64 lane、CUDA、Release、warm-up のあとに 8 秒）で採る。Nsight Systems は、同じ構成の Run を採る。計測の区間は NVTX にも出る設定で build される（[CMakeLists.txt:33](../../CMakeLists.txt) の `ANET_ENABLE_NVTX`、既定は ON）。

| # | 問い | 測り方 | 結果ごとの分かれ方 |
|---|---|---|---|
| Q1 | 統計の区間（8.206 ms）の内訳。誤差の統計、観測の統計の更新、捨てている正規化、target の出力の統計、診断値の読み出し | 計測の区間を足した実行体で Tracy を採る | 捨てている正規化が主なら、候補 C1 だけで大半が消える。観測の統計の更新が主なら C2。target の出力の統計が主なら C4 |
| Q2 | 正規化の区間（1 回 1.041 ms）の内訳。浮動小数点への変換、平均と標準偏差の作り直し、割り算と clip | 同上 | 作り直しが主なら C3。割り算と clip が主なら C2 |
| Q3 | 学習の入口に届く観測、α、ボーナス、外的報酬が置かれる device。観測が CUDA に置かれる構成があるか | 転送の区間の時間と、Nsight の転送の記録で確かめる | 観測が CPU なら、C2 は device への転送を浮動小数点から整数の画素に変える話になる。CUDA なら、今は intake ごとに CPU への転送が走っており、C2 で消える。ボーナスが CUDA にあれば C5 を入れる |
| Q4 | GPU 側の時間。採点と補助学習の kernel、転送の回数と大きさ、同期の待ち、Q の学習との重なり | Nsight Systems | GPU が Q の学習で埋まっていれば、device へ寄せた集計が Q の学習を待たせうる。その場合は、C2・C4 の利得を実時間で確かめてから入れる |
| Q5 | lane の数の依存（64 と 128） | 128 lane で Tracy を採る | 比例するなら、128 lane の Run ほど利得が大きい。受入の構成（§8 の 13）を決める材料にする |
| Q6 | PRD121 のあとの基準。RND なし、既定 RND、価値の分解を有効にした RND の実時間 | 採取器に群を足して 3 巡 | 受入の基準値（§7 の 5）を置き換える。価値の分解で採点を呼ぶ側の区間が増えていれば、Q1 と同じに分ける |
| Q7 | CPU のスレッド数を 2・4 にしたときの統計の区間と実時間 | `backend.torch_num_threads` を変えた短い Run | 実時間が下がるなら、C6 を候補に戻す。Env の lane と取り合って上がるなら入れない |
| Q8 | Taiga 設定の基準 | 採取器の Taiga の群 | ステージ 1 の速さの受入（§7 の 3）の基準値にする |

### 5.3 ステージ 1: 統計方式の整理（値を変えない）

- 方式ごとの統計（観測の正規化、誤差からの倍率、診断の平均と標準偏差、保存と復元）を 1 つの型にまとめ、whiten 用と variance_feedback 用の実装を、構築時に 1 つ選ぶ（§8 の 14）。
- 誤差の統計も、累積用と直近重み用に分ける（§8 の 14）。
- `Estimate`・`Learn`・`EstimateWithoutUpdate` の共通の前半を、private の関数 1 つにする（§8 の 16）。
- warm-up の条件は、件数と `warmup_steps` の比較を推定器に、warm-up 中の誤差を統計に入れるか（`warmup_mode`）を whiten の実装に置く（§8 の 15）。
- `UpdateErrors` が件数を逆算する前提は、観測の統計の更新と誤差の統計の更新を、variance_feedback の実装の中の 1 つの流れにして無くす。
- checkpoint の並びと書式の識別子、設定キー、診断の鍵、ログの文言、既存の計測の区間名は変えない（§6.1）。

### 5.4 ステージ 2: 採点の統計の区間を軽くする（候補）

どの候補を入れるかは、ステージ 0 の結果を見て、再グリルで決める。暫定の既定は §8 の 4〜10 に置いた。

| 候補 | 中身 | 効く区間 | RND 有効の学習の値 |
|---|---|---|---|
| C1 | 観測の正規化の部品に、統計だけを更新する口を足し、採点はそれを呼ぶ | 統計（捨てている正規化） | 変わらない |
| C2 | 観測を整数の画素のまま計算 device へ送り、浮動小数点への変換、正規化、batch の集計を device で行う。累積（CPU・倍精度）へは集計の結果だけを渡す | 統計（観測の統計の更新）、正規化、入力の転送 | 丸めの範囲で変わる |
| C3 | 正規化に使う平均と標準偏差の単精度の写しを持ち、統計を更新したときだけ作り直す | 正規化 | 変わらない |
| C4 | target の出力の統計と診断値を device で集計し、採点 1 回の CPU への読み戻しを 1 回にまとめる。α は、今と同じ式で CPU で作る | 統計（target の出力の統計、診断値）、誤差の転送 | 変わらない（診断値は丸めの範囲で変わる） |
| C5 | ボーナスの診断値の読み出し 6 回を、1 回の転送にまとめる | 採点を呼ぶ側 | 変わらない |
| C6 | 統計の CPU の演算のスレッド数を増やす | 統計、正規化 | 変わらない |
| C7 | 累積の統計も計算 device に置く | 統計 | 丸めの範囲で変わる |

variance_feedback には、C1〜C7 を当てない（§8 の 10）。

### 5.5 頻度と単価の見積もり

- 頻度: 採点は intake ごとに 1 回で、N 件をまとめて処理する。64 lane なら、64 exp step に 1 回である。補助学習は、既定（`learn_ratio` 0.25、`learn_batch_size` 32）なら、64 lane で 2 intake に 1 回である（§2.1 の 388 intake に対して 194 回）。
- 今の単価: Tracy をつないだ状態で、RND 全体が 1 intake あたり 13.097 ms（1 exp step あたり 0.205 ms）。うち統計と正規化が 9.767 ms（74.58%）である。
- 実時間の差から見た単価: (31.584 − 19.019) 秒 ÷ 1,406 intake ＝ 1 intake あたり 8.94 ms。Tracy をつながない状態の値で、報酬の違いによる学習の系列の違いを含む。
- 見積もり（未測定）: 実時間の差が Tracy の内訳と同じ割合で分かれると仮定する。統計と正規化の区間が 5 分の 1 になれば、既定 RND ÷ RND なしの比は 1 ＋ 0.661 ×（1 − 0.7458 × 0.8）＝ 1.267 になる。暫定の閾値 1.30 は、これに余裕を足した値である（§8 の 12）。
- 足す同期の転送: C2 と C4 を入れると、採点 1 回の device への転送は 1 回のまま（大きさは、浮動小数点から整数の画素になり 4 分の 1）、CPU への転送は 2 回から 1 回になる。観測が CPU にある場合の見積もりで、Q3・Q4 の結果で取り直す。

## 6. 実装契約（§8 の暫定の既定を前提）

### 6.1 変えないもの

設定キー、メトリクスの名前、診断の鍵、checkpoint の書式の識別子と中身の並び、ログと WARN と ERROR の文言・順・回数、seed の作り方、乱数の消費の順、forward の回数と順、一生の新しさの口（採点、読むだけの採点、補助学習、診断、保存と読み込み、写し）、既存の計測の区間名。

意図して変える点は次のとおり。

- ステージ 0: 統計の区間の中に、計測の区間が増える（§8 の 3）。既存の `statistics` の区間は、α の計算と誤差の統計の更新だけを指すようになる。
- ステージ 2 で C2 か C7 を入れた場合: RND を有効にした Run の、観測の統計と α が、丸めの範囲で変わる（§8 の 11）。RND 無効の Run は変わらない。
- ステージ 2 で C4 を入れた場合: 診断値 `target_feature_var`・`target_feature_max`・`error_step`・`clip_ratio` が、丸めの範囲で変わる。

### 6.2 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| RndNoveltyEstimator | 変更 | 採点・補助学習・読むだけの採点の共通の前半を 1 つにし、統計の扱いを方式の型へ委ねる | Agent 共通基盤（intrinsic_rnd） |
| RndStatistics | 新規 | 方式ごとの統計（観測の正規化、誤差からの倍率、診断の平均と標準偏差、保存と復元）の口 | Agent 共通基盤（intrinsic_rnd） |
| WhitenRndStatistics | 新規 | whiten の統計。観測の正規化の部品と、誤差の統計を持つ | Agent 共通基盤（intrinsic_rnd） |
| VarianceFeedbackStatistics | 変更 | `RndStatistics` の実装にする。観測と誤差の統計の更新を 1 つの流れにする | Agent 共通基盤（intrinsic_rnd） |
| RndErrorStatistics | 変更 | 誤差の統計の口にする | Agent 共通基盤（intrinsic_rnd） |
| CumulativeErrorStatistics | 新規 | 累積の誤差の統計 | Agent 共通基盤（intrinsic_rnd） |
| EmaErrorStatistics | 新規 | 直近重みの誤差の統計 | Agent 共通基盤（intrinsic_rnd） |
| ObservationNormalizer | 変更 | 統計だけを更新する口 `UpdateStats` を足す | テンソル処理（scaler） |
| SingleTensorNormalizer・ConstantSingleTensorNormalizer・RunningStdSingleTensorNormalizer・DictObservationNormalizer | 変更 | `UpdateStats` を実装する。`RunningStdSingleTensorNormalizer` は、C3 を入れるなら平均と標準偏差の写しを持つ | テンソル処理（scaler_impl） |
| DefaultDQNAgent::IntrinsicRewardCoordinator | 変更 | C5 を入れるなら、ボーナスの診断値を 1 回の転送で読む | DefaultDQNAgent（default_dqn_agent） |

### 6.3 移行対象

経路の検索（2026-11-15、`git grep -l -i -F`、範囲は追跡しているファイルの全体）で、直す関数と型を名指す箇所を確かめた。検索語と、当たったファイルの数は、`NormalizeAndUpdateStats` 14、`RndNoveltyEstimator` 33、`VarianceFeedbackStatistics` 6、`RndErrorStatistics` 5、`feedback_stats_` 3、`ScoreBonus` 7、`target_feature_stats_` 2、`EstimateWithoutUpdate` 12、`intrinsic_reward` 59 である。過去の PRD・実装計画・ADR・課題・wiki・実験記録（書き換えない）を除くと、現用のものは次のとおり。

- **変えるコード**: [intrinsic_rnd.hpp](../../core/anet-core/src/intrinsic_rnd.hpp)・[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp)・[scaler.hpp](../../core/anet-core/include/anet/scaler.hpp)・[scaler_impl.hpp](../../core/anet-core/src/scaler_impl.hpp)・[scaler_impl.cpp](../../core/anet-core/src/scaler_impl.cpp)。C5 を入れるなら [default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)。
- **検索に当たるが変えないコード**: [intrinsic.hpp](../../core/anet-core/include/anet/intrinsic.hpp)（一生の新しさの口は変えない）。[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) の 959〜960 行は、DQN の観測の正規化が `NormalizeAndUpdateStats` を呼んで結果を捨てる箇所で、変えない（§8 の 5）。
- **テスト**: [intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp) に、整理の前後で採点・補助学習・保存の結果が変わらないことの特性テストと、§7 の 6 のテストを足す。[scaler_test.cpp](../../core/anet-core/src/scaler_test.cpp) に、統計だけを更新する口のテストを足す。既存の断定の期待値は変えない。[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) は、検索に当たるが変えない。
- **設定**: 変えない。
- **検証資材**: [run_performance.py](../../core/anet-core/testdata/prd116/run_performance.py) は、Tracy の区間を `RndNoveltyEstimator` の名前で絞り（213 行）、集計する区間の名前の組を固定で持つ（272 行）。[run_performance_atari.py](../../core/anet-core/testdata/prd116/run_performance_atari.py) は、その集計と、群の定義（33 行）を使う。足す区間と、Q6 の群を読めるように追随させる。`core/anet-core/testdata/prd130/README.md` を新設する（実装前の実行体と設定の保存、前後比較、Tracy と Nsight の採り方）。[prd123 の README](../../core/anet-core/testdata/prd123/README.md) と [prd114 の migration.json](../../core/anet-core/testdata/prd114/migration.json) は、検索に当たるが、過去の記録なので変えない。
- **設計文書**: [212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md) の §3.2（2 種類の正規化）、§3.8（実装を読む順番）、§5.1（計算する場所と費用）、§5.2（確認済みの範囲）を追随させる。[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) は、推定器と `ScoreBonus` を名指す箇所を確かめ、変わる記述があれば追随させる。新しい仕組みではないので、設計文書は新設しない（§8 の 17）。en は `anet-translate-docs` で作る。
- **互換層**: 作らない。過去の Run・ADR・実験記録は書き換えない。

## 7. 受入の候補（§8 の暫定の既定を前提）

再グリルで契約を確定したあとに固める。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | ステージ 1 の同一性 | 実装前に保存した実行体と、seed 1・決定的モードの 100k exp step の Run で、metrics checksum が一致する。構成は、LunarLander の既定の連鎖に次を足した 4 つである。(a) 何も足さない（RND 無効）。(b) `run.@rnd`。(c) `run.@rnd` と `intrinsic.rnd.error_norm.average_mode = ema`。(d) `run.@rnd` と、Atari の `run.@rnd_taiga` の 8 つの設定値。比べるのは全 scalar の点で、時間と資源の tag は除く（実時間で変わるため） |
| 2 | ステージ 1 の checkpoint | 実装前の実行体が、1 の (b)(c)(d) で保存した checkpoint を、実装後の実行体が同じ設定で読める。続きの 100k exp step の metrics checksum が、実装前の実行体で続けた場合と一致する |
| 3 | ステージ 1 の速さ | 採取器で、既定 RND と Taiga 設定の実時間を、実装前と実装後の実行体で交互に 3 巡以上測る。群ごとの中央値の差が、同じ実行体どうしの最小〜最大の幅の内である |
| 4 | ステージ 2 の同一性（RND 無効） | 1 の (a) の構成で、ステージ 1 のあとの実行体と metrics checksum が一致する |
| 5 | ステージ 2 の速さ | 採取器（Montezuma、64 lane、CUDA、Release、同じ経験数）で、RND なしと既定 RND を、実装後の実行体で順を入れ替えて 3 巡以上測る。既定 RND の中央値 ÷ RND なしの中央値が 1.30 以下である。あわせて、ステージ 0 で取り直した実装前の比より小さい |
| 6 | ステージ 2 の値の近さ（RND 有効） | 単体テストで、同じ観測の列を、実装後の採点と、CPU で集計する参照の計算（テストの中に書く）に渡す。α、観測の統計の平均と標準偏差、誤差の統計の平均と標準偏差の差が、§8 の 11 の許す差の内である |
| 7 | 読むだけの採点 | PRD127 の §7 の 10 のテストが、断定を変えずに通る。whiten の累積・直近重みと variance_feedback の 3 構成で、読むだけの採点の倍率が、直後に同じ観測 1 件を `Estimate` へ渡した倍率と一致する |
| 8 | 計測の区間 | 既定 RND の Run の Tracy に、§8 の 3 の区間が出る。採取器の集計が、足した区間を読む |
| 9 | 転送の回数 | Nsight の記録で、既定 RND の採点 1 回あたりの、device から CPU への転送が 1 回である（C4 を入れた場合） |
| 10 | 既存のテスト | `anet-core-test` の既定のスイートと、`[prd116]`・`[scaler]`・`[shipped_config]` が通る |

## 8. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§5〜§7 はこれを前提に書いた。再グリルですべて開き直す。再グリルは、ステージ 0 の現状分析と対応方針の整理のあとに行う。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | 束ね方とステージの順 | 1 本の PRD で扱う。現状分析、整理、性能の順（§5.1） | 課題 f9790d の 2026-11-14 の記録は、0ddffa と束ねて 1 回で直すとしている（§4）。整理を先にすると、性能の直しが whiten の実装 1 か所で済み、前後の比較で値の差の原因を絞れる。依存: なし | 性能を先に入れ、整理をあとにする。整理と性能を同時に入れる。課題ごとに PRD を分ける |
| 2 | ステージ 0 で測る範囲 | §5.2 の Q1〜Q8 | 候補のどれが効くかは、内訳が無いと決まらない（課題 f9790d の対応方針）。Tracy の区間は CPU 側の時間なので、device へ寄せる候補の当否には GPU 側の記録が要る。依存: なし | Tracy の Q1・Q2 だけにする |
| 3 | 統計の区間の中に足す計測の区間 | `observation_statistics`（観測の統計の更新）、`target_feature_statistics`（target の出力の統計）、`diagnostics`（診断値の読み出し）の 3 つを足す。既存の `statistics` は α の計算と誤差の統計の更新に狭める。実行体に残す。入れる時機は、再グリルの前（現状分析の最初）とする | 計測済みの処理を分けたら、分けたあとの区間にも計測を置く（§4）。残せば、あとから同じ内訳を測れる（G3）。整理でコードがほかのクラスへ移っても、既存の区間名は保つ（§6.1）。計測の追加は学習の値を変えず、Q1・Q2 はこの区間が無いと測れない。依存: なし | 測るときだけ足して、あとで消す。再グリルのあとに入れる（Q1・Q2 は、作業用の実行体で測る） |
| 4 | 正規化の結果を捨てる二重計算 | C1 を入れる。統計だけを更新する口を、観測の正規化の部品に足す | PRD116 の実装が、費用を測ってから決めるとして残した件である（[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp) の 562 行の `@todo`）。値が変わらず、部品の局所的な情報だけで決まる処理なので、共通処理として足す（§4）。依存: なし | 足さずに、RND の側で統計の部品を直接持つ |
| 5 | 統計だけを更新する口を使う範囲 | RND だけが使う。DQN の観測の正規化の 2 つの呼び出しは変えない | DQN の側は、`NormalizeAndUpdateStats` が数える clip の割合を診断値として読める。口を替えると、その診断値が変わる。依存: 4 | DQN の側も替える（診断値の扱いを決める） |
| 6 | 観測の統計の集計と正規化を行う場所 | C2 を入れる。累積は CPU・倍精度のままにする | 統計と正規化で RND の 74.58% を占める（§2.1）。累積を CPU・倍精度のままにすれば、checkpoint の中身と書式は変わらない。入れるかは Q1〜Q4 の結果で決め直す。依存: 2、11 | C1 と C3 だけにして、値を変えない。C7（累積も device に置く） |
| 7 | 正規化に使う平均と標準偏差の写し | C3 を入れる | 正規化は、統計を更新しない補助学習と読むだけの採点でも走り、そのたびに作り直している（§2.3）。値は変わらない。Q2 の結果で決め直す。依存: 2 | 入れない |
| 8 | target の出力の統計と、採点 1 回の読み戻し | C4 を入れる。target の出力の統計は、今と同じく採点のたびに更新する | 今は、診断のためだけに N×出力次元の値を CPU へ戻している（§2.2 の 5）。α を CPU で作れば、学習の値は変わらない。Q1・Q4 の結果で決め直す。依存: 2 | target の出力の統計を、メトリクスの間隔だけ更新する（診断値の意味が、全採点の累積から変わる）。入れない |
| 9 | ボーナスの診断値の読み出し 6 回 | Q3 で、ボーナスが CUDA にある構成があると分かれば、C5 を入れる。無ければ変えない | PRD128 が、RND の時間の内訳と合わせて測るときに決め直すとした件である（§2.4）。CPU にあるなら、同期の転送にならない。依存: 2 | 構成に依らず入れる |
| 10 | variance_feedback の扱い | 値を変えない。ステージ 1 の整理だけを当て、C1〜C7 は当てない | lane の順の逐次の更新は、方式の定義である（§2.3）。統計の区間は 0.34% で、費用の主因は設定である（§1.3）。依存: なし | 観測の統計の更新だけを device へ寄せる |
| 11 | 値の同一性と、RND 有効で許す差 | ステージ 1 は、RND 有効を含む 4 構成で metrics checksum を一致させる。ステージ 2 は、RND 無効で一致させ、RND 有効は一致を求めない。許す差は、観測の統計の平均と標準偏差が相対 1e-5、誤差の統計の平均と標準偏差が相対 1e-5、α が絶対 1e-4 | 整理は値を変えない直しなので、PRD123 の条件（§4）をそのまま当てる。性能の直しは、課題 f9790d の 2026-11-14 の記録（§4）が、RND を有効にした Run の系列が変わることを問題にしないとしている。許す差の数値は見積もりで、ステージ 0 のあとに、単精度の集計の順を変えたときの実際の差を見て置き直す。依存: 6 | ステージ 2 も、値を変えない候補（C1・C3・C4・C5）だけにして、RND 有効でも一致を求める |
| 12 | 性能の受入の閾値 | 既定 RND ÷ RND なしの中央値の比が 1.30 以下 | §5.5 の見積もり（1.267）に余裕を足した。ステージ 0 で基準と内訳を取り直してから置き直す。依存: 2、6 | 比ではなく、Tracy の区間の時間で決める。閾値を置かず、前後の差だけを見る |
| 13 | 速さを測る構成 | Montezuma、64 lane、CUDA、Release（PRD116 の採取器の条件）を主にする。128 lane と、価値の分解を有効にした構成は、ステージ 0 で内訳と基準だけを採る | 前の測定（§2.1）と比べられる。次に積む Run は 128 lane と価値の分解を使うので、基準は採っておく。依存: 2 | 128 lane を主にする |
| 14 | 統計方式の型の分け方 | 方式ごとの統計を 1 つの型にまとめ、whiten 用と variance_feedback 用の実装を構築時に選ぶ。誤差の統計も、累積用と直近重み用に分ける（§5.3、§6.2） | 課題 0ddffa の対応方針のとおり。どのメンバがどの方式で生きているかが、型で分かる。依存: なし | 誤差の統計は分けず、今の 1 つの型のままにする |
| 15 | warm-up の条件の置き場 | 件数と `warmup_steps` の比較は推定器に、warm-up 中の誤差を統計に入れるか（`warmup_mode`）は whiten の実装に置く | 件数の比較は 2 つの方式で共通で、`warmup_mode` は whiten だけが読む（PRD123 が、variance_feedback では `warmup_mode` を使わない設定として知らせる）。依存: 14 | どちらも方式の実装に置く |
| 16 | 共通の前半にまとめる範囲 | `Estimate`・`Learn`・`EstimateWithoutUpdate` の 3 つ | 3 つとも同じ並びを持つ（§2.5）。エラー文と前処理を直す場所が 1 つになる。依存: なし | 課題の記述のとおり、`Estimate` と `Learn` の 2 つだけにする |
| 17 | 設計文書と ADR | 設計文書は新設せず、212 と 200 を追随させる（§6.3）。ADR は起こさない | 新しい仕組みではない。統計の集計を device へ寄せる判断を ADR にするかは、再グリルで、入れる候補が決まってから 3 条件に当てる。依存: 6 | — |
| 18 | 新しい名前（§6.2、3 の区間名、口の名前） | `RndStatistics`・`WhitenRndStatistics`・`CumulativeErrorStatistics`・`EmaErrorStatistics`・`UpdateStats` と、3 の 3 つの区間名 | 型の名前は、既存の `RndErrorStatistics`・`VarianceFeedbackStatistics` と同じ語尾にそろえる。口の名前は、既存の `NormalizeAndUpdateStats` にそろえる（[実装 品質確認観点](../quality/impl.md) の「名前」）。再グリルと実装で見直してよい。依存: 3、4、14 | — |
| 19 | CPU のスレッド数 | 変えない（C6 を入れない） | `backend.torch_num_threads` は Run 全体の設定で、Env の lane のスレッドと取り合う。Q7 の結果で決め直す。依存: 2 | RND の統計の演算だけ、スレッド数を増やす |

## 9. issue の対応表

| 課題 | 題 | area | 根拠の位置（2026-11-15） | 未決表 | 受入 |
|---|---|---|---|---|---|
| [issue-20261112-f9790d](../management/issues/issue-20261112-f9790d.md) | RND の既定設定で、採点ごとの統計の更新が CPU の 1 スレッドで走り、RND の時間の 63% を占める | 324_dqn、321_agent_core | [intrinsic_rnd.cpp:547-580](../../core/anet-core/src/intrinsic_rnd.cpp)、[scaler_impl.cpp:149](../../core/anet-core/src/scaler_impl.cpp)、[default_dqn_agent.cpp:122-127](../../core/anet-core/src/default_dqn_agent.cpp) | 2〜9、11〜13、19 | 4、5、6、8、9 |
| [issue-20261111-0ddffa](../management/issues/issue-20261111-0ddffa.md) | RndNoveltyEstimator に 2 つの統計方式が同居して分岐が散り、Estimate と Learn も前半を同じ形で書いている | 321_agent_core | [intrinsic_rnd.hpp:116-119](../../core/anet-core/src/intrinsic_rnd.hpp)、[intrinsic_rnd.cpp:504-670](../../core/anet-core/src/intrinsic_rnd.cpp) | 1、10、14〜16、18 | 1、2、3、7 |

## 10. ほかの PRD・課題との関係

- [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)（価値の分解）は実装済みで、本 PRD はそのあとのコードを基にする。採点を呼ぶ側（`ScoreBonus`）は、C5 を除いて変えない。
- [PRD924](924_intrinsic_reward_novelty_10prd.md)（内的報酬の新しさと方策の組）は、新しさの推定を広げる。本 PRD の整理のあとに足すと、統計方式を足す場所が 1 か所になる。
- [issue-20261114-9468cb](../management/issues/issue-20261114-9468cb.md)（推定器の写しが 1 回約 170 ms）は、題材の外である。写しは、統計を保存の形に書いて読み戻す。整理で統計の型が変わっても、写しの結果は今と同じに保つ。
- [PRD944](944_observation_pipeline_10prd.md)（観測パイプライン。暫定で、グリルの前）は、観測の正規化の部品の持ち方を変える。本 PRD が部品に足す口（`UpdateStats`）は、PRD944 を実装するときに、部品と一緒に移す。本 PRD を先に実装する。
- [PRD910](910_network_lock_audit_10prd.md)（排他の精査。草案）は、DQN の観測の正規化の `NormalizeAndUpdateStats` の呼び出しとロックの関係を扱う。本 PRD はその呼び出しを変えない（§8 の 5）ので、順は問わない。
- 2026-11-15 に、触るファイルとクラスの名前（`intrinsic_rnd`・`RndNoveltyEstimator`・`LifelongNoveltyEstimator`・`IntrinsicRewardCoordinator`・`scaler_impl`・`ObservationNormalizer`・`RunningMeanStd`・`NormalizeAndUpdateStats`）で `docs/memo/` の PRD を検索した。当たった PRD のうち、同じ箇所の形を変える予定で、まだ実装していないものは、上の PRD924・PRD944・PRD910 である。
