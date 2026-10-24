# replay の遷移が消えるまでの抽選回数と優先度を、PER の外の区分で測る 暫定 PRD

> 2026-10-16 起票。詳細設計は別途グリルで行う前提で、暫定の既定を置き、決めることを §6 に残した。
> 2026-10-18 改訂: 段階 1 に「区分ごとの raw priority（3 時点）」を追加し、TD 誤差に珍しさが現れているかも測るようにした（§1、§2、§3 の 5、§6 の 10〜12、§7）。指標の群番号を 48 から 50 に変更した（48 は PRD 093 の attention が使用）。
> 2026-10-23 改訂: 指標の群番号を 50 から 59 に変更した（50 は [098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md) のスキップの指標が使用。51・52 は DropMerge の評価の群）。
> 2026-10-23 改訂: §2 の KFM の Run 名を今のフォルダ名に直した（`…_kung_fu_master_…` から `…_kfm_…` への改名。値は変えていない）。
> 起点: Atari kung_fu_master でボス討伐が定着しない原因を調べたとき、PER が珍しい経験（ボス撃破）を消える前に何度も引けているかを、既存の指標では確かめられなかったこと。
> 関連: [done/073_replay_fit_metrics_10prd.md](done/073_replay_fit_metrics_10prd.md)（replay 当てはまり診断。抽選履歴の 2 群）、[ADR 0039](../adr/0039-replay-fit-sampling-history-groups-not-holdout.md)、[done/069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md)（trace チャネル）、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)、[CONTEXT.md](../../CONTEXT.md) の「Replay・PER」。

## 1. 目的と問題

PER が効いているかを、PER の内側の量（優先度、TD 誤差）ではなく、遷移の中身で分けた区分ごとに「消えるまでに何回、学習用抽選で引かれたか」で測れるようにする。知りたいのは、報酬の大きい遷移や終端の遷移のような珍しい経験が、ふつうの遷移より多く引かれているか、どれだけ引かれずに消えているか、である。

あわせて、同じ区分ごとに優先度（raw priority）も集計し、珍しい経験の貴重さがそもそも TD 誤差に現れているかを確かめる。TD 誤差は 1 step の予測の外れであって、その経験が方策をどれだけ変えうるかとは別の量である。たとえば報酬クリップの下では、ボス撃破の報酬は雑魚と同じ +1 で、撃破の先の価値（出口の換算）はまだ target に入っていないので、撃破の遷移の TD 誤差は小さく出うる。抽選回数だけでは、「TD に出ていないから引かれない」のか、「TD に出ているが `per_alpha` で潰れている」のかを分けられない。

### ゴール

- replay の遷移ごとに、学習用抽選で引かれた回数（抽選回数）を数える。
- 遷移が上書きで消えるとき、その抽選回数を区分ごとに集計し、スカラー指標として出す。区分は PER の優先度に依らない（学習報酬の大きさ、終端かどうか。段階 2 で env が出す値）。
- PER あり・なし（一様抽選）のどちらでも同じ指標を出し、一様の場合と比べられるようにする。
- PER ありのとき、同じ区分ごとに raw priority を 3 つの時点（入ったとき、初回 Learner priority 更新の直後、退去の直前）で集計し、優先度の持ち上げ率として出す。

### 非ゴール

- PER の改良そのもの（`per_alpha` の調整、優先度を手前の遷移へ流す方式、成功例の保持、SIL）。本 PRD はその効き目を測る計器だけを作る。
- 学習系列を変えること。抽選・優先度・学習の値は変えない。
- 引かれた遷移が学習に反映されたか（当てはまり）の判定。replay 当てはまり診断（PRD 073）の範囲である。
- 消える遷移 1 件ずつの記録（trace 行）。§4 の先送りに置く。

## 2. 現状（2026-10-16 に確認したこと）

- 実時間優先の Atari 構成（`run_20261016-042902_kfm_rr1_50m_noclip200_seed2` の config dump）は、`replay_capacity` 2,097,152（128 lane × 16,384）、`replay_batch_size` 256、50M 遷移で learn 194,500 回（257 遷移に 1 回）。遷移 1 件が寿命中に引かれる回数の期待値は約 1.0 で、寿命は約 8,160 回の更新（RTX 4060 で実時間約 18 分）。
- PER は `per_alpha` 0.2、β 0.2 固定、raw priority は |TD| + 1e-6 を上限 20 で clip、初期優先度は max。α = 0.2 では raw priority が 20 倍でも SumTree leaf priority は 20^0.2 ≈ 1.8 倍にしかならない。
- 同じ Run の `39_agent_per/07_evicted_unsampled_ratio`（退去時に一度も引かれていなかった割合）は、0〜50M の 10M 窓で 0.245〜0.263。一様抽選で期待値 1.0 なら e^−1 ≈ 0.37 で、PER と最大優先度の初期化で少し下がっているが、4 件に 1 件は一度も学習に使われずに消えている。
- この指標は全遷移をまとめた値で、どの遷移が引かれずに消えたかは分からない。値は直前の 1 回の Push で上書きされた分だけで（[replay_buffer_impl.cpp](../../core/anet-core/src/replay_buffer_impl.cpp) の `RecordEvictionIfSampleable` と `StoreLastEvictionStats`）、PER が無いと値を返さない（同ファイルの `GetScalar`、`PER_LAST_EVICTED_NEVER_SAMPLED_RATIO`）。
- 遷移ごとの抽選の記録は、`sampled_once_`（uint8 のフラグ、[replay_buffer_impl.hpp](../../core/anet-core/src/replay_buffer_impl.hpp)）だけである。`Sample` の `MarkSampledOnce` で 1 にし、`Push` で slot を再利用するときに 0 に戻す。PER の有無によらず保守されている。回数は持たない。
- 退去の数え方は、上書きされる slot が ready range のときだけ数える（`IsOverwritingReady`）。CONTEXT の「ready range」の定義どおり、eviction 統計の判定基準である。dummy の slot も同じ判定を通る。
- `Push` の時点で、その step の報酬（`QueueRecord.reward`）と終端（`done`）は分かる。replay に入る報酬は reward scaler を通した後の値（[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp) の `reward_scaler_->Scale`）。
- Atari の報酬クリップ（`reward_clip=true`）では、kung_fu_master のボス撃破（2,000 点）も雑魚（100〜500 点）も学習報酬は +1 で、学習報酬では見分けられない。クリップ前の 1 step の報酬は `AtariEnv::Step` のローカル変数（[AtariEnv.cpp](../../core/envs/atari1/src/AtariEnv.cpp) の `reward_raw`）で、外へは出ていない。env の step の出力（`BatchState`、[rl.hpp](../../core/anet-core/include/anet/rl.hpp)）は観測・done・truncated・episode_start だけで、step ごとの付帯値を replay まで運ぶ口は無い。
- 報酬クリップを外して 1/200 倍にした Run（上と同じ Run）では、撃破は学習報酬 10、雑魚は 0.5〜2.5 になる。この Run で撃破が出たゲームは学習中の 0.1〜0.3% だった。
- replay 当てはまり診断（PRD 073）は、`sampled_once_` で未抽選群と抽選済み群に分ける。[ADR 0039](../adr/0039-replay-fit-sampling-history-groups-not-holdout.md) は「per-entry の抽選回数カウンタ」を、二群の比較には一度以上か否かで足り、Sample の hot path に加算が乗るとして不採用にした。本 PRD はその判断と見かけ上ぶつかる（§3 の 1、§6 の 5）。
- trace チャネルの event は `@episode_end` だけで、追加の event は PRD 069 の範囲外とされている（[140_observability.jp.md](../design/140_observability.jp.md) の trace チャネルの節）。
- 優先度について（2026-10-18 に確認）:
  - 分位点系の Learner（IQN・QR）の優先度の元になる TD は、「現在の分位点の平均 − target の分位点の平均」である（[dqn_based_agent.cpp](../../core/anet-core/src/dqn_based_agent.cpp) の `QuantileLearnerBase::ComputeElementError`）。IQN では現在側・target 側とも τ を 8 本ずつランダムに引くので、収益の分布が広い状態ほど、学べる誤差とは別の揺らぎが優先度に乗る。
  - 優先度は、引かれて Learner が更新したときにだけ書き換わる（`ApplyLearnerPriorities`）。初期優先度は `per_initial_priority_mode`（実時間優先の構成では `max`）で決まり、各 slot の優先度 source（CONTEXT の「優先度source」）が `max_initial` などから `learner_updated` へ移る。CONTEXT の「初回Learner priority更新」がその最初の書き換えに当たる。
  - 優先度ストア（`ReplayPriorityStore`）は slot ごとの優先度の行を返せる（`GetPriorityTensor`、`GatherPriorityRows`）。退去の直前の raw priority をここから取れるかは §6 の 10 で確かめる。
  - 既存の優先度の指標は、学習バッチ単位の TD の平均と標準偏差（`37_agent_qtd/01_td_mean`、`04_td_std`）と、Actor 近似と Learner の比較（`39_agent_per/01`〜`06`、`50`〜`65`）だけで、遷移の区分ごとの値は無い。replay 当てはまり診断の「PER 選択比」も、実 PER バッチと replay 全体の平均 TD の比で、区分ごとではない。

## 3. 提案の骨子

### 段階 1: 抽選回数と、学習報酬・終端による区分

1. `sampled_once_` のフラグを、slot ごとの抽選回数（uint32）に置き換える。`MarkSampledOnce` の書き込みが加算に変わるだけで、hot path の仕事量は今と同じ。1 回の抽選の中で同じ slot が 2 回選ばれたら 2 と数える。未抽選群・抽選済み群は「回数が 0 か、1 以上か」で今と同じに決まる。
2. `Push` のとき、その slot の区分を決めて slot ごとに持つ（uint8）。区分は 2 つの軸で、PER の優先度を使わない。
   - 報酬の区分: その step の学習報酬の絶対値を、設定の境界で分ける。既定の境界は `1` で、区分は「0」「0 より大きく 1 以下」「1 より大きい」の 3 つ。報酬クリップの Run では、報酬のある遷移はすべて真ん中に入る。
   - 終端: `done` の遷移かどうか（報酬の区分とは別に数える）。
3. slot が上書きで消えるとき（今と同じ ready range の判定）、その抽選回数を区分ごとに集計する。
   - 区分ごとの累積の退去件数。
   - 区分ごとの直近 N 件（既定 1,024 件）の平均抽選回数と未抽選率。珍しい区分でも、件数が少ないまま平均が読めるように、時間の窓ではなく件数の窓にする。
   - 全区分の同じ値と、区分の平均抽選回数を全区分の平均で割った比（持ち上げ率）。一様抽選なら、どの区分も 1 付近になる。
4. 値は ReplayBuffer の `GetScalar` から出し、`metrics_scalar.txt` の購読で `@learn` に載せる。PER の有無によらず値を返す。まだ 1 件も消えていない区分は値なし（既存の PER 指標と同じ扱い）。
5. PER ありのとき、退去する slot について、raw priority を 3 つの時点で区分ごとに集計する（直近 N 件の平均。件数の窓は 3 と同じ）。
   - 入ったとき: 初期優先度。`max` の初期化では全件がその時点の最大値なので、区分の差は出ない（初期化の方式を変えた腕との比較用）。
   - 初回 Learner priority 更新の直後: 一度だけ記録する（slot ごとに float を 1 つ持つ）。「TD 誤差に珍しさが現れるか」の主な読み。一度も引かれずに消えた slot は値なしとして数えない。
   - 退去の直前: 最後に適用された raw priority。
   - 区分の平均を全区分の平均で割った比を「優先度の持ち上げ率」として出す。3 の「抽選回数の持ち上げ率」と並べて読む。優先度の持ち上げ率が 1 付近なら TD に出ていない。高いのに抽選回数の持ち上げ率が低ければ、`per_alpha` などで潰れている。

段階 1 だけで、報酬のある遷移・終端の遷移が、ほかより何倍引かれているか、TD 誤差の上で何倍目立っているかが分かる。報酬クリップを外した Run では、境界を 5 にすれば kung_fu_master の撃破だけを 1 つの区分に分けられる。

### 段階 2: env が出す 1 step の値による区分

1. env が step ごとの付帯値（Atari ではクリップ前の 1 step の報酬）を出し、replay の `Push` まで運ぶ口を作る。
2. 段階 1 の報酬の区分を、学習報酬の代わりにその値で決められるようにする（設定で値の名前を指定する）。
3. これで、報酬クリップの Run でも、kung_fu_master の撃破（2,000 点）を雑魚（100〜500 点）と分けて数えられる。

段階 2 は段階 1 の区分の決め方だけを足す。段階 1 で止めても、報酬クリップを外した Run と、報酬のある遷移全体の見え方は手に入る。

## 4. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| 消える遷移 1 件ずつの trace 行（lane、Push と退去の exp step、抽選回数、最後の優先度） | 先送り | trace に event を足すのは PRD 069 の範囲外で、段階 1 のスカラーで足りるかを先に見る。珍しい区分の件数が少なすぎて平均が読めないとき、または lane・年齢ごとに見たくなったときに起こす |
| 退去の時点で、その遷移に学習が当てはまっていたか | 対象外 | replay 当てはまり診断（PRD 073）の範囲。区分別に見たくなったら 073 側の拡張として起こす |
| エピソードの成績（収益の上位など）による区分 | 先送り | エピソード終了後にそのエピソードの slot へ遡って印を付ける仕組みが要る。段階 2 の 1 step の値で足りないと分かったときに起こす |
| Push から退去までの年齢、抽選の時期の分布 | 先送り | 平均の抽選回数と未抽選率で問いに答えられる。抽選が寿命の前半に偏るかを見たくなったときに起こす |

## 5. User Stories

1. 研究者として、ボス撃破のような珍しい報酬の遷移が、消えるまでに何回学習に使われたかを、Run の指標で確かめたい。
2. 研究者として、`per_alpha` を変えた腕どうしで、報酬のある遷移の持ち上げ率がどう変わったかを比べたい。
3. 研究者として、PER を切った Run で同じ指標が 1 付近になることを見て、計器が正しいことを確かめたい。
4. 研究者として、報酬クリップをかけたままの Run でも、クリップ前の得点の大きさで遷移を分けて数えたい（段階 2）。
5. 研究者として、珍しい経験の TD 誤差が本当に大きく出ているのかを確かめ、引かれない原因が TD 誤差の側か、抽選の側かを分けたい。

## 6. 決めること（詳細設計で詰める）

暫定の既定を置いた。グリルで確定する。

| # | 決めること | 暫定の既定 |
| --- | --- | --- |
| 1 | 設定キーの名前と置き場所 | `learner.replay_eviction_stats.reward_edges`（正の実数の昇順リスト、既定 `1`）、`learner.replay_eviction_stats.window`（1 以上の整数、既定 1,024）。replay の他の設定と同じく `learner.*` に置く |
| 2 | 不正値の扱い | `reward_edges` が空・非有限・0 以下・昇順でない、`window` が 1 未満は、Agent の構築時に設定キーと値を示して fail-fast。値を丸めたり既定へ置き換えたりしない |
| 3 | 指標の tag と GetScalar のキー | 群番号 59 を使う（46〜50 は `46_agent_replay_fit`・`47_agent_noisy`・`48_agent_attention`・`49_agent_gradnoise`・`50_agent_skip` が使用中、51・52 は DropMerge の評価の群）。例: `59_agent_rb_evict/11_r1_mean_samples`。キーは `replaybuffer.eviction.[<区分>].<量>`。tag は AGENTS.md の規約どおり 40 文字以内 |
| 4 | 区分の名前 | 報酬の区分は境界の番号で `r0`（0）、`r1`、`r2` …、終端は `terminal`、全体は `all` |
| 5 | ADR 0039 との関係 | 0039 が不採用にしたのは「二群の比較のための」回数カウンタで、理由の 1 つは hot path の加算。本 PRD は回数そのものが目的で、フラグの書き込みを加算に替えるだけなので仕事量は変わらない。0039 の先頭に注記を足し、新しい ADR を起こすかをグリルで決める |
| 6 | 用語 | 「抽選回数」（replay の遷移が、現在保持している遷移として学習用抽選で引かれた回数）を CONTEXT に足すか。段階 2 の「クリップ前の 1 step の報酬」の呼び名（生スコアは 1 ゲームの累計なので別の語が要る） |
| 7 | 段階 2 の口 | env の step の付帯値を replay の Push まで運ぶ経路（`BatchState` に足すか、別の経路か）。値の名前を設定で選ぶ形にするか |
| 8 | prefetch との関係 | `PrefetchingReplayBuffer` の先読み抽選も、抽選回数に数えるか。暫定は数える（`sampled_once_` が今そうであるのと同じ） |
| 9 | MuZero の replay | 同じ `DefaultReplayBuffer` を使う MuZero でも出すか。暫定は出す（区分の決め方は同じ） |
| 10 | 退去の直前の raw priority の取り方 | 優先度ストアの slot ごとの行から取る（SumTree leaf priority から α を戻すのではなく、raw の値を使う）。行に raw が無ければ、slot ごとに raw を float で 1 つ持つ |
| 11 | 初回 Learner priority 更新の直後の値の持ち方 | slot ごとに float を 1 つ（2M slot で 8 MB）と、記録済みかの印。Push で slot を再利用するときに消す。stale な更新（generation が変わった slot への更新）は記録しない |
| 12 | 優先度の指標の出し方 | PER なしでは値なし。分位点系の Learner では TD の定義が分位点平均の差であることを、指標の説明に書く |

## 7. 成功の測り方と受入の目安

成功の測り方:

- PER を切った Run（一様抽選）で、全区分の平均抽選回数が「1 回の更新で引く件数 × 寿命中の更新回数 ÷ sampleable range の件数」に近く、どの区分の持ち上げ率も 1 付近になる。
- PER ありの Run で、全区分の未抽選率が既存の `39_agent_per/07_evicted_unsampled_ratio` の長い窓の平均と合う。
- kung_fu_master の報酬クリップを外した構成で、`per_alpha` を 0.2 から 0.6 にすると、撃破を分けた区分（境界 5 の `r2`）の持ち上げ率が上がる。
- 同じ構成で、撃破の区分の「初回 Learner priority 更新の直後」の優先度の持ち上げ率が出る。おおよそ、抽選回数の持ち上げ率 ≈ 優先度の持ち上げ率の `per_alpha` 乗、の関係になるかで、計器どうしの整合を確かめる（初期優先度が max なので、1 回目の抽選の分だけずれる）。

受入の目安（詳細はグリルで確定する）:

1. 抽選の位置を固定した単体テストで、退去時の区分ごとの件数・平均抽選回数・未抽選率が、手で数えた値と一致する。優先度も、決めた値で Learner 更新を与えた単体テストで、3 時点の区分平均が手で計算した値と一致する。
2. 一様抽選の単体テストで、全区分の平均抽選回数が理論値の ±10% に入る。
3. smoke Run の `metrics.jsonl` と `metrics.scalar.defs` に、PER あり・なしの両方で新しい tag が出る。
4. §6 の 2 の不正値で、Agent の構築時にエラーで止まり、メッセージに設定キーと値が出る。
5. 学習の値が変わらない: 本機能の前後で、同 seed・決定的モードの短い Run の metrics checksum（抽選回数の指標を除く）が一致する。
6. learn の速度の差が、ラウンドロビンの A/B で測定のぶれの内側に収まる。

## 8. 依存と順序

段階 1 に依存は無い。段階 2 は §6 の 7 の口を決めてから始める。

## 9. 参照

- [CONTEXT.md](../../CONTEXT.md) の「ready range」「sampleable range」「replay 抽選履歴群」「未抽選群」「抽選済み群」「raw priority」「SumTree leaf priority」「生スコア」。
- [現行の scalar 定義](../../apps/runner/config/metrics_scalar.txt) の `37_agent_qtd`、`39_agent_per`。
- Mattar & Daw 2018「Prioritized memory access explains planning and hippocampal replay」（Nature Neuroscience）: 再生の価値を「その更新で方策がどれだけ良くなるか（gain）× その状態にこれから来る見込み（need）」とする理論。TD 誤差は gain の粗い近似で、need を含まない。
