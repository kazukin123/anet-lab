# RND の内的報酬の尺度（誤差の統計の窓・warm-up・平均を引かない形） 暫定 PRD

> 2026-11-11 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §7 の未決表に置いた。§4・§5・§6 は §7 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: 2026-11-11 の会話。サブ機の KFM×RND が基準を上回らなかったので、内的報酬の大きさを Run の cache で確かめた（§2）。会話で出た方向は次の 3 つで、どれも決定ではない。(1) 序盤の誤差を誤差の統計に入れない（以下 A）。(2) 誤差の統計を直近に重みを置いて取る（以下 B、指数移動平均）。(3) 平均を引かない形（e ÷ σ_e、以下 C）を比べられるようにする。Burda らの形（内的報酬の割引収益の標準偏差で割る形）は [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)・[924](924_intrinsic_reward_novelty_10prd.md) に残す。起票の途中で、開発者から「今の動きとの互換性を維持するモードは残さなくていいか」という問いが出た。これも決定ではなく、§7 の 18 に論点として置いた。
> 関連: [116_rnd_lifelong_novelty_10prd.md](116_rnd_lifelong_novelty_10prd.md)（RND の本体。§8 の 1・3・5 を本 PRD へ移した。§8）、[121_extrinsic_intrinsic_value_decomposition_10prd.md](121_extrinsic_intrinsic_value_decomposition_10prd.md)（外的と内的の価値の分解）、[924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（NGU 型の内的報酬）、設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md)（RND の解説）・[200 §6.2](../design/200_dqn_agents.jp.md)、課題 [issue-20261111-77ac3c](../management/issues/issue-20261111-77ac3c.md)（本 PRD の仕事）・[issue-20261108-227b05](../management/issues/issue-20261108-227b05.md)（predictor の周期リセットを試して戻した記録）。

## 1. 目的と問題

### 問題

DefaultDQNAgent の RND（PRD116、既定の `statistics_mode = whiten`・`multiplier_mode = clipped`）は、1 step の内的報酬を β × clip((e − μ_e) ÷ σ_e, 0, L − 1) で作る（L は `multiplier_max`、既定 5）。e は predictor と target の出力の二乗誤差、μ_e・σ_e は誤差の平均と標準偏差（以下、合わせて「誤差の統計」）である。誤差の統計は、Run の最初から採点したすべての誤差の累積（Welford）で、warm-up 中の誤差も入る（§2.3）。

序盤の大きな誤差が σ_e を膨らませ、次の 3 つが起きた（§2.1・§2.2）。

1. 内的報酬が、序盤を除いた誤差の統計で作った場合の 1/3〜1/17（Montezuma）、1/7〜1/19（KFM の 10M 時点）に抑えられている。
2. 抑え方は Run が進むと薄まり、実効の β（実際に効いている内的報酬の大きさ）が Run の中で 3〜5 倍に上がる。設定に書いていないスケジュールになっている。
3. 平均を引いて 0 で切るので、内的報酬が付く状態は約 1 割で、残りは 0 になる。学習とともに誤差が下がると、今の誤差が累積の平均を下回り続ける（LunarLander・GridMaze で内的報酬が消えた現象と同じ形。[wiki の内的報酬のページ](../wiki/mechanisms/intrinsic_novelty.md)）。

### ゴール

- whiten の内的報酬の大きさが、その時点の誤差の散らばりで決まり、序盤の一時的な大きな誤差に左右されない。
- 実効の β が Run の中で隠れて上がらず、β の比較が名目どおりの意味を持つ。
- 平均を引かない形（C）を、同じ部品のモードとして比べられる。

### 非ゴール

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| Burda らの形（e ÷ 内的報酬の割引収益の標準偏差） | lane ごとの割引収益の保持と、外的・内的の価値を分ける話（PRD121）と組になる。別の関心として PRD121・924 に残す | 制約ではない（別 PRD で入れられる） |
| `variance_feedback`（Taiga らの統計）の中身を変える | Taiga らの条件を再現するためのモードで、再現の意味が変わる | 再現のために変えない |
| 観測の正規化の式（累積の統計、除数の ε 1.0、clip 5）を変える | 序盤の未熟な正規化から来る大きな誤差は A で統計から外せる見込み。効果を見てから決める（§7 の 13） | 制約ではない（Run の結果で開き直す） |
| 内的報酬の運び方（スケール後の外的報酬に足す） | 外的・内的を別に持つのは PRD121 の題材 | 制約ではない |
| predictor の周期リセット | GridMaze・LunarLander で利得が無く戻した（issue-20261108-227b05） | 制約ではない（試した結果で退けた） |

## 2. 背景（実測と原典）

### 2.1 KFM×RND（サブ機、2026-11-10〜11）

Run は workspace AtariSub-01 の `run_20261110-*_kfm_10m_rnd_b01_s04`〜`s10`（7 本、10M）。構成は実時間優先（RT10）＋ `run.@rnd`、β 0.1、`whiten`・`clipped`、warm-up 12,800 件、Munchausen 有効。基準は同じ seed の `run_2026101[89]-*_kfm_10m_s04`〜`s10`（RND なし）。完走した Run の `metrics_cache.db` をメイン機へコピーし、読み取り専用で読んだ（どれも `source_meta` の state=ready、committed_offset と jsonl の大きさが一致）。

内的報酬の大きさ（3M 以降を 1M ごとの区間で平均した値の、7 本の範囲）:

| 指標 | 値 |
|---|---|
| `56_agent_in/01_bonus_mean`（1 step の lane 平均） | 1e-4〜8e-4 |
| `56_agent_in/02_bonus_max`（同じ step の最大の lane） | 0.008〜0.034 |
| `56_agent_in/03_bonus_share`（Σ\|内的\| ÷ (Σ\|外的\| ＋ Σ\|内的\|)） | 0.014〜0.057 |
| `56_agent_in/05_bonus_pos_ratio`（内的報酬が正の lane の割合） | 0.04〜0.15 |
| 参考: `36_agent_munchausen/04_bonus_mean`（Munchausen の項） | −0.05〜−0.18 |

内的報酬は 1 step あたり Munchausen の項の約 1/100 だった（外的報酬は clip され、雑魚 1 体で +1）。ボスに挑んでいた時期（s04 の 4〜6M、s09 の 4M・8M。RAM の `boss_hit` の平均で判定）にも、内的報酬の平均は前後と変わらなかった。外的報酬の少ない 0〜1M の区間だけは `03_bonus_share` が 0.23〜0.85、2M 台の区間は 0.037〜0.14 だった。

誤差の統計の逆算: `11_error_mean`（μ_e）と `12_error_std`（σ_e）は累積の値なので、最初の記録（200k exp step。Q の学習の開始と同じ）と 10M の値から、200k 以降の誤差だけの平均と標準偏差を Welford の合成の式で逆算した。

| seed | 200k の μ_e | 200k の σ_e | 10M の σ_e | 最初の 200k が二乗偏差の総量に占める割合 | 200k 以降だけの σ_e | その時点の誤差（`13_error_step` の直近 400 点の平均） |
|---|---|---|---|---|---|---|
| s04 | 0.236 | 2.61 | 0.372 | 99% | 0.029 | 0.016 |
| s05 | 0.173 | 1.40 | 0.201 | 97% | 0.027 | 0.012 |
| s06 | 0.299 | 4.60 | 0.653 | 99% | 0.034 | 0.014 |
| s07 | 0.219 | 2.20 | 0.314 | 98% | 0.028 | 0.015 |
| s08 | 0.300 | 1.95 | 0.280 | 97% | 0.026 | 0.013 |
| s09 | 0.305 | 1.84 | 0.265 | 97% | 0.028 | 0.020 |
| s10 | 0.247 | 2.92 | 0.415 | 99% | 0.027 | 0.015 |

10M 時点でも σ_e は 200k 以降だけの値の 7〜19 倍で、その時点の誤差の z は −0.03〜−0.06 だった。内的報酬の上限（β × 4 = 0.4）に届くには、誤差が μ_e ＋ 4σ_e（約 1.6、その時点の誤差の約 100 倍）に達する必要がある。

評価側（参考。exe が基準 10/18・RND 11/10 で違い、設定は RND とキーの改名・既定で無効の新機能のほかは同じ。RND の効果と exe の違いは分けていない）:

| seed | Greedy 5.25M（100 ゲーム）基準 → RND | ε 0.01 評価 8M 以降の 4 回の平均 基準 → RND |
|---|---|---|
| s04 | 14,834 → 12,896 | 17,468 → 18,440 |
| s05 | 15,062 → 11,436 | 20,352 → 11,795 |
| s06 | 35,640 → 21,177 | 24,602 → 15,872 |
| s07 | 21,445 → 16,922 | 21,040 → 24,290 |
| s08 | 13,090 → 10,104 | 25,272 → 15,612 |
| s09 | 21,314 → 14,315 | 19,262 → 17,810 |
| s10 | 19,260 → 23,669 | 23,058 → 27,402 |
| 平均 | 20,092 → 15,788 | 21,579 → 18,746 |

### 2.2 Montezuma の Taiga ら寄せの構成（メイン機、2026-11-08〜11）

Run は workspace AtariRnd-01 の `mz_tg_` 系のうち `whiten` の 5 本（`run_20261108-232006_mz_tg_rnd_s1`、`run_20261109-105448_mz_tg_rnd_tr40k_s1`、`run_20261110-003002_mz_tg_nmdd_b001_s1`、`run_20261110-120903_mz_tg_nmdd_b01_s1`、走行中の `run_20261110-233727_mz_tg_rnd_b001_s1`）。構成は [Montezuma の RND（Taiga ら寄せ）の記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md) による。どれも seed 1 で、200k までは同じ動きなので、200k の μ_e 0.150・σ_e 0.910 は 5 本とも一致した。最初の 200k が二乗偏差の総量に占める割合は、30M まで走った 2 本と 22.6M で止めた b01 が 88〜90%、tr40k（17.5M）が 90%、rnd_b001（5.9M 時点）が 97% だった。

σ_e の膨らみ（実際の σ_e ÷ 200k 以降の誤差だけで逆算した σ_e）:

| Run | 0.5M | 1M | 2M | 4M | 8M | 14M | 20M | 29M |
|---|---|---|---|---|---|---|---|---|
| s1（β 0.1、Munchausen 有効） | 13 倍 | 11 倍 | 10 倍 | 5 倍 | 4 倍 | 3 倍 | 3 倍 | 3 倍 |
| b001（β 0.01、Munchausen 無効） | 11 倍 | 10 倍 | 9 倍 | 7 倍 | 4 倍 | 4 倍 | 3 倍 | 3 倍 |
| b01（β 0.1、Munchausen 無効） | 11 倍 | 10 倍 | 9 倍 | 5 倍 | 4 倍 | 4 倍 | 3 倍 | — |
| rnd_b001（β 0.01、Munchausen 有効） | 17 倍 | 17 倍 | 16 倍 | 13 倍 | — | — | — | — |

膨らみは 5 本で同じ形なので、β 0.01 と 0.1 の比較（記録の結論）は崩れない。ただし β の値は「この膨らみの下の値」であり、内的報酬は Run の中で 3〜5 倍に強まっていた。Montezuma は外的報酬がほぼ 0 なので、`03_bonus_share` は小さな内的報酬でも 0.93〜1.0 になる。

### 2.3 今の仕組み（2026-11-11 にコードで確認）

- 採点（[intrinsic_rnd.cpp:336-403](../../core/anet-core/src/intrinsic_rnd.cpp)）: whiten では、今回の batch を入れる前の μ_e・σ_e で α を作り（[同 384-385](../../core/anet-core/src/intrinsic_rnd.cpp)）、そのあと `error_stats_` に今回の誤差を足す（[同 386](../../core/anet-core/src/intrinsic_rnd.cpp)）。`error_stats_` は `RunningMeanStd`（[scaler_impl.hpp:17-61](../../core/anet-core/src/scaler_impl.hpp)）で、窓を持たない累積である。
- warm-up: 採点した観測の件数が `warmup_steps` に達するまで α = 1 を返すが、観測の統計と誤差の統計は更新する（[同 384-388](../../core/anet-core/src/intrinsic_rnd.cpp)）。`Learn` は件数が `warmup_steps` 以下の間は predictor を学習させずに false を返す（[同 411](../../core/anet-core/src/intrinsic_rnd.cpp)）。既定は 12,800 件（[agent.txt:12](../../apps/runner/config/agent.txt)）で、128 lane なら 100 env step。
- RND の採点と predictor の学習は exp step 0 から始まり、Q の学習の開始（Atari は `learner.update_warmup_steps` 200,000、[Atari.txt:318](../../apps/runner/config/Atari.txt)）とは独立である。KFM の s04 では 200k の時点で `32_learn_count` が 1,465 だった。
- 観測の正規化（whiten）も累積の統計で、初期値は平均 0・分散 1、除数は √分散 ＋ 1.0（[212 §3.2](../design/212_rnd_lifelong_novelty.jp.md)）。
- 内的報酬の合成は facade（[default_dqn_agent.cpp:713-735](../../core/anet-core/src/default_dqn_agent.cpp)）で、β × (α − 1) をスケール後の外的報酬に足し、`next_state` を採点するときは done の lane を 0 にする。
- 序盤の大きな誤差が、predictor が未学習なためか、観測の正規化が未熟なためかは分けていない（`56_agent_in` の tag は `@learn` で記録され、Q の学習が始まる 200k より前の値が無い）。

### 2.4 原典と公開実装の形

| | Burda ら（RND 原典、PPO） | Taiga ら（Rainbow＋RND） | NGU・Agent57 | anet の既定（whiten・clipped） | anet の `run.@rnd_taiga` |
|---|---|---|---|---|---|
| 内的報酬の式 | e ÷ σ_R | 0.00005 × (e − m) ÷ v | エピソード内の報酬 × clip(1 ＋ (e − μ_e) ÷ σ_e, 1, 5) | β × clip((e − μ_e) ÷ σ_e, 0, 4) | β × (e − m) ÷ v |
| 平均を引くか | 引かない | 引く（負もある） | 引く（倍率なので 1 未満は 1） | 引く（平均以下は 0） | 引く（負もある） |
| 割る値 | 内的報酬の割引収益（γ 0.99、エピソードで切らない）の標準偏差 | 分散に当たる再帰の値（平方根を取らない） | 誤差の標準偏差 | 誤差の標準偏差 | Taiga らと同じ |
| 統計の期間 | 件数で重みを付けた累積 | 累積 | 本 PRD では確かめていない | 累積 | 累積 |
| 序盤の扱い | 学習の前に乱数行動で 819,200 件（128 env × 128 step × 50）を集め、観測の正規化だけを作る。predictor と内的報酬の統計には触れない | 置かない | 本 PRD では確かめていない | 方策で動きながら 12,800 件。内的報酬 0、観測と誤差の統計は更新 | 置かない |

出典: Burda らは公開の PyTorch 実装（jcwleo/random-network-distillation-pytorch の `train.py`・`agents.py`・`utils.py`・`config.conf`）。内的報酬は `(target − predictor)² の和 ÷ 2` を、`RewardForwardFilter`（γ 0.99 の割引和）の値の `RunningMeanStd` の √分散で割る。`RunningMeanStd` は件数で重みを付けた累積で、初期の件数は 1e-4。観測の正規化の前段は `ObsNormStep` 50 × `NumStep` 128 × `NumEnv` 128。Taiga らは PRD116 §2.1（原典のコードで確認）。NGU の式は 212 §5.4 が引く NGU 論文 式 (1)。

### 2.5 今の式の出どころ（2026-11-11 の会話で確かめた範囲）

- 今の既定の内的報酬 β × (α − 1)、α = clip(1 ＋ (e − μ_e) ÷ σ_e, 1, L) は、NGU の一生の新しさの倍率（エピソード内の報酬に掛ける 1 以上の数）を、エピソード内の報酬なしで外的報酬に足す形である。PRD116 は、924 で NGU の倍率として使うときに部品を変えずに済むようにこの形を選んだ（PRD116 D2・D4）。
- NGU では、平均以下の状態は倍率 1（増やさない）になるだけで、エピソード内の報酬は残る。単独で足すと、平均以下の状態は内的報酬 0 になる（§1 の 3）。
- §2.4 の 3 つ（Burda ら・Taiga ら・NGU）のどれとも式が違い、§2.4 の原典の中に、この単独の形で出した結果は無い。そのため、β の値や効き方を文献から持ち込めない。
- 実装は PRD116 の契約どおりに動いている（採点は更新前の統計、done の lane を 0、warm-up の扱い。§2.3 のコードで確かめた）。§1 の問題は、実装の誤りではなく、契約の側（単独で使う形と、累積の統計）から来ている。
- 開発者の問い（2026-11-11）: 今の動作は NGU とも Taiga らとも Burda らとも違う独自の形になっているか、その妥当性はどうか。§7 の 19 に論点として置いた。

## 3. 用語（暫定）

| 語 | 意味 |
|---|---|
| 誤差の統計 | RND の採点で、誤差 e を標準化するための平均 μ_e と標準偏差 σ_e |
| 誤差の統計の窓 | 誤差の統計に入れる誤差の重みの付け方。累積（すべて同じ重み）か、指数移動平均（直近ほど重い）か |
| 時定数 H | 指数移動平均の重みが 1/e に減るまでの、採点した観測の件数 |
| 実効の β | 実際に効いている内的報酬の大きさを、名目の β と比べて言うときの語 |

`CONTEXT.md` には足さない（§7 の 14）。

## 4. 提案（§7 の暫定の既定を前提）

### 4.1 A: warm-up 中は誤差の統計を更新しない（§7 の 1・2・12・18）

whiten の warm-up の中身を `intrinsic.rnd.warmup_mode`（`predictor`｜`statistics`）で選べるようにし、既定を `predictor` にする。`statistics` は今の中身である。`variance_feedback` は変えない（§7 の 12）。

| | 今（`statistics`） | 提案（`predictor`） |
|---|---|---|
| α | 1 | 1（変えない） |
| 観測の統計 | 更新する | 更新する（変えない） |
| 誤差の統計 | 更新する | 更新しない |
| predictor の学習（`Learn`） | 進めない（false） | 進める（true） |

predictor を warm-up 中に学習させるのは、warm-up 明けの誤差を、学習が進んだ predictor の誤差から始めるためである（学習させないと、warm-up 明けに未学習の大きな誤差がまた統計に入る）。Atari の `run.@rnd` の warm-up は 200,000 件（Q の学習の開始と同じ）にする（§7 の 2）。warm-up 明けの最初の batch は、誤差の統計がまだ無いので α = 1 で、そのあと統計に入る（今の「採点履歴が無いとき α = 1」の規則のまま）。

`error_stats.mode = lifelong` と `warmup_mode = statistics` を選ぶと、whiten の中身は今（PRD116）と同じになる。同じ実行体の中で新旧を比べる対照に使う（§7 の 18）。

### 4.2 B: 誤差の統計を指数移動平均で取る（§7 の 3・4・5）

`intrinsic.rnd.error_stats.mode` に `lifelong`（今の累積）と `ema`（指数移動平均）を置き、既定を `ema` にする（§7 の 3）。`ema` の時定数は `intrinsic.rnd.error_stats.horizon`（採点した観測の件数、既定 1,000,000。§7 の 4）。

`ema` の統計は、採点した観測 1 件ごとに重みを d = 1 − 1/H 倍に減らした、重み付きの平均と分散である（§7 の 5）。batch の N 件は lane の順に 1 件ずつ入れたのと同じ値にする。

```text
S0 ← d^N × S0 ＋ Σ_{i=0..N-1} d^(N-1-i)
S1 ← d^N × S1 ＋ Σ_{i=0..N-1} d^(N-1-i) × e_i
S2 ← d^N × S2 ＋ Σ_{i=0..N-1} d^(N-1-i) × e_i²
μ_e = S1 ÷ S0、σ_e² = max(S2 ÷ S0 − μ_e², 0)
```

S0 で割るので、記録が少ない間も平均が 0 側へ偏らない。σ_e が 0 なら α = 1（今の規則のまま）。採点は今どおり、今回の batch を入れる前の μ_e・σ_e で行う。lane 数を変えても、同じ観測を同じ順に採点すれば同じ統計になる（PRD116 D7 の、lane 数に依らない考え方と同じ）。

### 4.3 C: 平均を引かない形（§7 の 6・7）

`intrinsic.rnd.multiplier_mode` に `uncentered` を足す。α = clip(1 ＋ e ÷ σ_e, 1, L)、内的報酬は β × (α − 1) = β × min(e ÷ σ_e, L − 1)。どの状態にも正の内的報酬が付く。`variance_feedback` との組は止める（§7 の 7）。

帰結: 内的報酬が 1 step ごとに正で付くので、`next_state` で done の lane を 0 にする今の合成では、エピソードを長く続けるほど得になる向きの力も生む。§2.1 の 200k 以降の値（μ_e 約 0.022、σ_e 約 0.028）なら、普段の状態で 1 step 約 0.8β になり、β 0.1 では KFM の外的報酬（雑魚 1 体 +1 が数十 step に 1 回）を上回る。C を試す β は小さく取る（§7 の 10）。

### 4.4 処理の順（whiten、提案のあと）

| 順番 | 処理 | 変わるところ |
|---:|---|---|
| 1 | 外的報酬をスケールする | なし |
| 2 | 観測を正規化する（今回の観測を入れる前の観測の統計） | なし |
| 3 | 誤差を計算する（更新前の predictor） | なし |
| 4 | α を求める（今回の誤差を入れる前の誤差の統計。warm-up 中と統計が無いときは 1） | 統計が `ema` のときの μ_e・σ_e、`uncentered` の式 |
| 5 | 観測の統計を更新する。warm-up でなければ誤差の統計を更新する | `predictor` では warm-up 中は誤差の統計を更新しない |
| 6 | β を評価し、done を 0 にして内的報酬を足す | なし |
| 7 | facade が credit に応じて `Learn` を呼ぶ | `predictor` では warm-up 中も predictor を学習させる |
| 8 | 経験を内側の Learner へ渡す | なし |

## 5. 実装契約（暫定の既定を前提）

### 5.1 設定と失敗の扱い（§7 の 3・4・6・7・15）

| 設定キー | 値 | 既定（`agent.txt` の `@baseline` に `?=`） | 読む条件 |
|---|---|---|---|
| `DefaultDQNAgent.intrinsic.rnd.error_stats.mode` | `lifelong`｜`ema` | `ema` | `statistics_mode = whiten` |
| `DefaultDQNAgent.intrinsic.rnd.error_stats.horizon` | 整数 ≥ 1（採点した観測の件数） | 1,000,000 | `whiten` かつ `ema` |
| `DefaultDQNAgent.intrinsic.rnd.multiplier_mode` | `clipped`｜`unclipped`｜`uncentered` | `clipped`（変えない） | 常に |
| `DefaultDQNAgent.intrinsic.rnd.warmup_mode` | `predictor`｜`statistics` | `predictor` | `statistics_mode = whiten` |
| `DefaultDQNAgent.intrinsic.rnd.warmup_steps` | 整数 ≥ 0（変えない） | 12,800（変えない）。Atari の `run.@rnd` は 200,000 | 常に。whiten では warm-up の中身を `warmup_mode` が決める（§4.1） |

設定例（コロン記法）:

```text
DefaultDQNAgent.@baseline : intrinsic.rnd.error_stats.mode ?= ema        # lifelong｜ema
DefaultDQNAgent.@baseline : intrinsic.rnd.error_stats.horizon ?= 1,000,000
DefaultDQNAgent.@baseline : intrinsic.rnd.warmup_mode ?= predictor       # predictor｜statistics
run.@rnd : DefaultDQNAgent.intrinsic.rnd.warmup_steps = 200,000
```

判定（構築時。上から順に当て、最初に条件を満たした行で扱う）:

| # | 条件 | 扱い |
|---|---|---|
| 1 | `error_stats.mode` が列挙の外 | ERROR（キー・値・列挙を示す） |
| 2 | `warmup_mode` が列挙の外 | ERROR（キー・値・列挙を示す） |
| 3 | `error_stats.horizon` が 1 未満か整数でない | ERROR（キーと値を示す）。モードによらず検証する |
| 4 | `multiplier_mode` が列挙の外 | ERROR（列挙に `uncentered` を含めて示す） |
| 5 | `statistics_mode = variance_feedback` かつ `multiplier_mode = uncentered` | ERROR（組を示す） |
| 6 | `statistics_mode = variance_feedback` で、`error_stats.mode`・`error_stats.horizon`・`warmup_mode` のどれかの解決した値が実装の既定と違う | 違うキーごとに構築時に WARN を 1 回出し、値は使わない |
| 7 | `statistics_mode = whiten`・`error_stats.mode = lifelong` で、`error_stats.horizon` の解決した値が実装の既定と違う | WARN を 1 回出し、値は使わない |
| 8 | それ以外 | 有効 |

行 6・7 は PRD116 の「モードが読まない値」の扱い（検証のうえ、実装の既定と違えば WARN 1 回）と同じである。`@baseline` の `?=` のままなら WARN は出ない。`multiplier_max` は `uncentered` でも読む（α の上限）。

### 5.2 メトリクス（§7 の 8）

tag は増やさない。`56_agent_in/11_error_mean`・`12_error_std` は、次の batch の採点に使う誤差の統計（`ema` なら指数移動平均）を出す。whiten・`warmup_mode = predictor` の warm-up 中は誤差の統計が無いので NaN になる（`statistics` では今どおり有限）。

### 5.3 checkpoint（§7 の 9）

RND の書式を `RndNoveltyEstimator-4` にし、`statistics_mode` のあとに `error_stats.mode` を書く。whiten の誤差の統計は、`lifelong` なら今の累積、`ema` なら S0・S1・S2 を保存する。書式が `-4` でないもの（今の `-3` を含む）と、`error_stats.mode` が設定と違うものは ERROR（checkpoint の値と設定の値を示す）。旧書式を読む互換分岐は置かない（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。

### 5.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RndNoveltyEstimatorConfig` | 変更 | `error_stats.mode`・`error_stats.horizon`・`warmup_mode`・`uncentered` を読んで検証し、読まない値を WARN する | `intrinsic_rnd`（`src/intrinsic_rnd.hpp`・`.cpp`） |
| `RndNoveltyEstimator` | 変更 | whiten の warm-up の中身（§4.1）、誤差の統計のモードの切り替え、`uncentered` の倍率、書式 `-4` の保存と読み込み | `intrinsic_rnd` |
| `RndErrorStatistics` | 新規（非公開） | whiten の誤差の統計を `lifelong`（累積）と `ema`（§4.2 の式）で持ち、更新・平均と標準偏差の読み出し・保存と読み込みを行う | `intrinsic_rnd` |

### 5.5 移行対象

- コード: [intrinsic_rnd.hpp](../../core/anet-core/src/intrinsic_rnd.hpp)・[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp)。facade（[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)）は変えない見込み。
- 設定: [agent.txt](../../apps/runner/config/agent.txt) の `@baseline` に §5.1 の 3 キー（`error_stats.mode`・`error_stats.horizon`・`warmup_mode`）を `?=` で足す。[Atari.txt](../../apps/runner/config/Atari.txt) の `run.@rnd` に `warmup_steps = 200,000` を足す（§7 の 16 の時機に注意）。`run.@rnd_taiga` は変えない（行 6 の WARN が出ないこと）。[LunarLander.txt](../../apps/runner/config/LunarLander.txt)・[GridMaze.txt](../../apps/runner/config/GridMaze.txt) の `run.@rnd` は設定ファイルを直さず、既定の変更（`ema`、whiten の warm-up の中身）がそのまま効く。
- WARN の線引き: [PRD117](117_warn_policy_and_unread_config_10prd.md)（未実装）は、RND のモードが読まない値の WARN を扱う。§5.1 の行 6・7 の WARN は 117 の線引きに合わせ、117 が先に入っていれば 117 の形に従う。
- テスト（[intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp)、タグ `[prd116]` の 36 件のうち次の断定が変わる）:
  - whiten の warm-up 中に誤差の統計が有限、または `Learn` が false と断定するもの: `RND factory scores observations and warms up statistics`、`RND warmup counts observations and reports optimizer progress`、`DQN intrinsic consumes warmup credit and holds the last loss`、`RND checkpoint preserves normalization warmup and training continuation`。今の中身を確かめるテストは `warmup_mode = statistics` を明示して残し、`predictor` の中身の断定を足す。
  - whiten の α を累積の統計の手計算と比べるもの（`RND keeps its seeded target fixed and uses preceding error statistics` ほか）: `error_stats.mode = lifelong` を明示するか、`ema` の手計算に直す。
  - 書式 `RndNoveltyEstimator-3` を断定するもの（`RND rejects mismatched checkpoint formats before reading payload` ほか、文字列 `RndNoveltyEstimator-3` で `core/anet-core/src` を検索して挙げる）。
  - `variance_feedback` のテスト（`[rnd_reference_feedback]`）は変わらない。
- 出荷設定のテスト: `[shipped_config]` の RND の既定の一致の検査に、§5.1 の 3 キーを足す。
- 設計文書: [212](../design/212_rnd_lifelong_novelty.jp.md) の §3.2（誤差の統計は累積、の記述）・§3.3・§3.4（処理の順の表）・§3.5（warm-up）・§4.2（設定の表）・§4.4（`error_mean`・`error_std`）・§4.5（書式）と英訳 [212 en](../design/212_rnd_lifelong_novelty.en.md)。[200](../design/200_dqn_agents.jp.md) の RND の既定の行（設定の節）と英訳。
- 文書: PRD116 の §8 の 1・3・5 は本 PRD へ移した旨を書いた（§8）。wiki の内的報酬のページ。
- 過去の Run・実験記録・checkpoint は書き換えない。今の書式 `-3` の checkpoint は新しい実行体では読めない（ERROR）。

## 6. 成功の測り方と受入の候補

### 6.1 効果（§7 の 10・11）

- **内的報酬の大きさが戻ったか（主）。** KFM（サブ機）で、β 0.1・A＋B（`ema`、Atari の warm-up 200,000）・seed s04〜s07 の 4 本・10M を、同じ seed の旧 RND（§2.1）と並べる。
  - 2M 以降の `03_bonus_share` と `01_bonus_mean` が、旧 RND の 4 本の範囲の上端を 4 本とも超える。見込みは旧 RND の 10 倍前後。
  - 8〜10M の `12_error_std` が 0.01〜0.05 の桁に入る（旧 RND は 0.20〜0.65）。
  - 2〜4M と 8〜10M の `01_bonus_mean` の比が、旧 RND より 1 に近い（隠れたスケジュールが消えた）。
- **同じ実行体の対照。** 旧 RND の Run は実行体が違う（§2.1 の注記）。exe の違いを外すため、新しい実行体で `error_stats.mode = lifelong`・`warmup_mode = statistics`・warm-up 12,800（今の中身）を同じ seed で 2 本（s04・s05）回し、A＋B の 4 本と並べる（§7 の 18）。
- **成績（副）。** 評価側の Greedy と ε 0.01 評価を、基準（RND なし）・旧 RND と並べる。seed 4 本のぶれ（基準の Greedy 5.25M で 13,090〜35,640）を超える差だけを読む。ボスの命中と撃破（RAM）も見る。
- **Montezuma。** メイン機の 12 bat のキューが終わったあと、b001 の構成（β 0.01）に A＋B を足して比べる（§7 の 10）。
- **C。** KFM で `uncentered`・β 0.01 を 2 本。内的報酬の大きさと、ゲームの長さ（エピソードを長く続ける向きの力。§4.3）を見る。

### 6.2 正しさの受入候補

契約を再グリルで確定したあとに固める。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 今の挙動との同一性 | `error_stats.mode = lifelong`・`warmup_mode = statistics`・`warmup_steps = 12,800`・`clipped` の whiten（PRD116 の既定と同じ中身）で、同 seed・決定的モードの短い Run（LunarLander の `run.@rnd`、100k exp step）の metrics checksum が、実装前の実行体と一致する。`variance_feedback`（Atari の `run.@rnd_taiga` の RND の設定を LunarLander に当てた短い Run）と、`intrinsic.enabled = false` も同じく一致する |
| 2 | 指数移動平均の式 | 既知の誤差の列で、§4.2 の S0・S1・S2 と μ_e・σ_e が手計算と一致する。同じ観測を N=1 で 4 回と N=4 で 1 回採点したときの統計が一致する |
| 3 | warm-up | whiten・`predictor` で、warm-up 中は α = 1、`11_error_mean`・`12_error_std` が NaN、`Learn` が true で predictor の損失が有限。warm-up 明けの最初の batch は α = 1、次の batch から統計で α が作られる。`statistics` では今の断定（誤差の統計が有限、`Learn` が false）が通る |
| 4 | 平均を引かない形 | `uncentered` の α が clip(1 ＋ e ÷ σ_e, 1, L) と一致する。`variance_feedback` との組が ERROR になる |
| 5 | 設定の判定 | §5.1 の判定の表の各行が表のとおりに扱われる。ERROR の文にキーと値が入る。WARN は構築時に 1 回 |
| 6 | checkpoint | `ema` の保存と再開で、続きの α と統計が保存しなかった場合と一致する。書式 `-3` と、`error_stats.mode` の違う checkpoint が ERROR になる |
| 7 | 出荷設定 | `agent.txt` の `@baseline` の 3 キーの値が実装の既定と一致する（`[shipped_config]`） |

## 7. 決めること（未決表）

暫定の起票なので、どの行も決まっていない。「暫定の既定」は AI が置いた既定で、§4・§5・§6 はこれを前提に書いた。再グリルですべて開き直す。

| # | 事項 | 暫定の既定 | 理由と依存 | ほかの候補 |
|---|---|---|---|---|
| 1 | whiten の warm-up 中にすること | 既定の `warmup_mode = predictor` では、誤差の統計を更新しない。観測の統計は更新する。predictor は学習させる。α は 1。今の中身は `statistics` で選べる（18） | 膨らみの主因は最初の 200k の誤差（§2.1・§2.2）。predictor を学習させないと、warm-up 明けに未学習の大きな誤差が統計に入る | (a) predictor は今どおり学習させない。(b) 前半は観測の統計だけ、後半は predictor も学習の 2 ステージ（キーが増える）。(c) Burda らのように乱数行動で観測の統計を作る（Actor の契約に触れる） |
| 2 | warm-up の長さ | `agent.txt` の既定は 12,800 のまま。Atari の `run.@rnd` は 200,000 | Atari は Q の学習の開始（200,000）まで内的報酬を使わない。200k 時点の誤差はまだ下がっている途中（KFM s04 で 0.057、0.25M で 0.026）だが、残りは B で追える。LunarLander・GridMaze は Run が短く、序盤の内的報酬が要る | Atari も 12,800 のまま（B だけで追う）。Q の学習の開始に自動で合わせる（Q の設定を読む依存が増える） |
| 3 | 誤差の統計のモードと既定 | `error_stats.mode`（`lifelong`｜`ema`）、既定 `ema` | ゴールの「その時点の散らばり」に当たる。`lifelong` は過去の Run と比べるために残す | 既定を `lifelong` のままにして `ema` を明示で選ぶ |
| 4 | 時定数 H の既定と単位 | 採点した観測の件数で 1,000,000 | Atari の 10〜30M の Run で、誤差が数 M で 2 割ほど下がる動き（§2.1 の `13_error_step`）より十分短く、1 batch（128）の揺れより十分長い。件数で数えると lane 数に依らない | 100,000、10,000,000。exp step で数える（intake の件数と同じなので同値） |
| 5 | 指数移動平均の式 | §4.2 の重み付きの和（S0 で割って初期の偏りを除く。lane の順に 1 件ずつと同じ値） | 記録が少ない間に 0 側へ偏らない（PRD045 の EMA の偏りの補正と同じ考え）。lane 数に依らない | batch ごとに 1 回の更新（lane 数で時定数が変わる） |
| 6 | 平均を引かない形の名前と式 | `multiplier_mode = uncentered`、α = clip(1 ＋ e ÷ σ_e, 1, L) | 倍率の作り方の違いなので `multiplier_mode` の値にする。上限は今の `clipped` と同じ L | 平均を引くかを別のキー（例 `error_centering`）にして clip と組み合わせる |
| 7 | `uncentered` と `variance_feedback` の組 | ERROR | `variance_feedback` は Taiga らの再現のためのモードで、平均を引かない形の原典が無い | 許す（e ÷ v） |
| 8 | メトリクス | tag は増やさない。`11_error_mean`・`12_error_std` は使う統計を出す。warm-up 中は NaN | 今の tag で膨らみが読めた（§2）。統計の意味をモードに合わせるだけで足りる | 両方のモードの値を並べて出す tag を足す |
| 9 | checkpoint | 書式 `RndNoveltyEstimator-4`、旧書式は ERROR | 統計の中身が変わり、今の書式では復元できない。互換分岐を置かない（クリーンブレーク） | 旧書式を `lifelong` として読む（互換分岐になる） |
| 10 | 実験の組み立て | §6.1（KFM はサブ機で β 0.1・A＋B の 4 本と、同じ実行体で今の中身の対照 2 本、C は β 0.01 の 2 本。Montezuma はメイン機の 12 bat のあと） | サブ機が空くので、KFM で内的報酬の大きさが戻るかを先に確かめる。C は §4.3 の帰結で β を下げる | seed の数、Run の長さ、Montezuma を先にする |
| 11 | β の既定 | `agent.txt` の 0.1 のまま | 実効の β は 10 倍前後になる見込みだが、どの β がよいかは Run で決める（④） | 0.01 に下げる |
| 12 | `variance_feedback` の warm-up | 変えない | Taiga らの再現のためのモード（`run.@rnd_taiga` は warm-up 0） | whiten と同じにそろえる |
| 13 | 観測の正規化 | 変えない | 序盤の誤差が predictor の未学習と正規化の未熟のどちらから来るかを分けていない（§2.3）。A の効果を見てから決める（④: §6.1 の Run の `12_error_std` が 0.05 を超えて残れば開き直す） | 除数の ε を小さくする、Burda らのように乱数行動で前段を置く |
| 14 | 設計文書・ADR・用語集 | 212（jp・en）と 200 の該当行を直す。新しい設計文書・ADR・`CONTEXT.md` の語は足さない | 新しい仕組みではなく RND の統計の取り方の変更で、212 が仕組みの正本 | 「誤差の統計の窓」を ADR にする（戻しにくさは小さい） |
| 15 | 名前 | `error_stats.mode`・`error_stats.horizon`・`warmup_mode`（`predictor`｜`statistics`）・`uncentered`・`RndErrorStatistics` | 何の統計かを添える（「窓」を単独で使わない）。impl.md の名前の規約との照合は再グリルで行う | `error_window.*`、`error_average`、`ratio` |
| 16 | 走っている bat への影響 | `Atari.txt` の `run.@rnd` の warm-up の変更は、メイン機の 12 bat の Run（凍結した 11/08 の実行体、`run.@rnd` を使う stage 18〜20）が終わってから入れる。`agent.txt` の新しいキーは凍結した実行体が読まないので、いつ入れてもよい | 設定ファイルは Run の開始ごとに読み直すので、`warmup_steps` を変えると凍結した実行体の Run の条件が変わる | 新しい Run 用に別のプロファイルを足して、`run.@rnd` を変えない |
| 17 | PRD121 の実験との関係 | PRD121 の比較（b001・b01 が基準）は今の統計の Run なので、PRD121 を実装して比べるときは誤差の統計のモードをそろえる（`lifelong` を明示するか、基準を取り直す） | §2.2 の膨らみは β と同じ働きをするので、統計が違うと β の比較が崩れる | 本 PRD を先に入れて基準を取り直す |
| 18 | 今の挙動を再現するモード | 残す。`error_stats.mode = lifelong` と `warmup_mode = statistics` の組で、whiten の中身が PRD116 と同じになる。旧書式の checkpoint は読まない（9） | 同じ実行体の中で新旧を比べる対照に要る。KFM では基準と RND の実行体が違い、RND の効果と分けられなかった（§2.1）。今の中身も実験の条件として意味があるので、別名・旧キー・互換読みのような互換の層には当たらない。費用は今のコードの経路を残すだけ。1・3・9 に依存 | 残さない（新旧の比較は凍結した実行体で取り、実行体の違いが混ざる）。warm-up の中身を `error_stats.mode` に含める（キーは 1 つ減るが、統計の窓と warm-up の 2 つの関心が 1 つのキーに混ざる） |
| 19 | RND 単独の群の目的と、式の出どころ | 今の形（NGU の倍率を単独で足す形）を既定に残し、平均を引かない形（6）を比べる候補にする | 924 で NGU の倍率として使うときに部品を変えずに済む（PRD116 D4）。ただし単独の形は文献に結果が無く（§2.5）、β の効き方を文献と比べられない。6・10 に依存 | (a) 群の目的を「文献と比べられる RND 単独」とし、既定を Burda ら寄せにする（6 の形か、PRD121・924 の割引収益で割る形）。(b) 目的を「NGU への通過点」と割り切り、単独の群は比較用として式は変えない。(c) Taiga ら寄せ（`run.@rnd_taiga`）を既定にする |

## 8. PRD116・PRD121・924 との関係

- PRD116 の §8 の 1（内的報酬の式。Burda らの e ÷ σ は比較の候補）・3（誤差統計の窓）・5（warm-up の既定値）を本 PRD へ移した。本 PRD の C は誤差の標準偏差で割る形で、Burda らの割引収益の標準偏差で割る形は移していない（PRD121・924）。PRD116 の D5 が whiten を「移動平均・標準偏差」と書いていたのは実物（累積）と違うので、2026-11-11 に PRD116 側で直した。
- PRD121（外的と内的の価値の分解）は β の大きさの議論（スケール説）を持つ。§2.2 の膨らみは β と同じ働きをするので、PRD121 の実験は誤差の統計をそろえて比べる（§7 の 17）。
- 924（NGU 型）は、エピソード内の報酬に一生の新しさの倍率を掛ける。倍率は本 PRD の誤差の統計を使うので、`ema` と warm-up の変更はそのまま効く。

## 9. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.4（観測の正規化と内的報酬の正規化）。
- 公開の PyTorch 実装 [jcwleo/random-network-distillation-pytorch](https://github.com/jcwleo/random-network-distillation-pytorch)（`train.py`・`agents.py`・`utils.py`・`config.conf`）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
- Badia ら、[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)、ICLR 2020。式 (1)。
- [Montezuma の RND（Taiga ら寄せ）の記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。
