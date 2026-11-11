# RND の内的報酬の尺度（誤差の統計の取り方・warm-up・平均を引かない形・終端の mask） PRD

> 2026-11-11 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §7 の未決表に置いた。§4・§5・§6 は §7 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: 2026-11-11 の会話。サブ機の KFM×RND が基準を上回らなかったので、内的報酬の大きさを Run の cache で確かめた（§2）。会話で出た方向は次の 3 つで、どれも決定ではない。(1) 序盤の誤差を誤差の統計に入れない（以下 A）。(2) 誤差の統計を直近に重みを置いて取る（以下 B、指数移動平均）。(3) 平均を引かない形（e ÷ σ_e、以下 C）を比べられるようにする。Burda らの形（内的報酬の割引収益の標準偏差で割る形）は [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)・[924](924_intrinsic_reward_novelty_10prd.md) に残す。起票の途中で、開発者から「今の動きとの互換性を維持するモードは残さなくていいか」という問いが出た。これも決定ではなく、§7 の 18 に論点として置いた。
> 2026-11-11 改訂（再グリル）: グリルを行い、暫定の印を外した。起票時の未決表（旧 §7）の 19 行を、確定事項（§4 の D1〜D15）・残る未決（§8 の U1〜U3）・事実（§2.3）へ振り分けた（対応は §4.2）。暫定の既定から変えた点は次のとおり。誤差の統計の既定を累積のままにし、直近に重みを置く形は選べるモードにした（D2）。設定キーを `error_norm.average_mode`（`cumulative`｜`ema`）・`error_norm.ema_steps`・`warmup_mode`（`learn_predictor`｜`collect_errors`）に改めた（D12）。平均を引くかを `multiplier_mode` の値ではなく別のキー `error_norm.use_centering` にした（D7）。`variance_feedback` で平均を引かない指定は ERROR ではなく WARN にした（D8）。同じ実行体の対照を含む Run の組み立てを決めた（D14）。グリルの途中で開発者が出した問いを受けて、終端（done）の遷移の内的報酬を 0 にするかを選ぶ `intrinsic.use_done_mask` を足した（D16、§2.7、§5.5。題にも足した）。章を組み直し、確定事項を §4、仕様を §5、実装契約を §6、成功の測り方と受入を §7、未決表を §8、関係を §9、参照を §10 にした。§1 のゴールを決定に合わせて書き直し、RND 単独の群の位置づけと、仮説・比べ方・終わり方を足し、非ゴールに 2 行（乱数行動で観測の統計を作る前段、内的の価値を game over で切らない形）を足した。移行対象（§6.5）に、PRD116 の検証資材の smoke の判定と、検証資材の新設を足した。関連に PRD117 を足した。別モデル（Opus）の検討を受けて、次を直した。checkpoint の書式の識別子を誤差の統計の中身の形で分け、累積の書式と中身は変えない形にした（D11、§6.3。今ある checkpoint をそのまま読める）。設定の判定を設定キーごとの表に組み直し、warm-up 0 のときの `warmup_mode` の通知を INFO にした（§5.1、§6.1）。効果の条件に、誤差の標準偏差の 2M〜3M と 9M〜10M の比を足し、仮説を合否の言い方に合わせた（§1、§7.1）。ゴール 1 の対象を Atari の `run.@rnd` に限り、LunarLander・GridMaze は動きが変わるが確かめないと書いた（§1、D6）。直近重みの式は定義として書き、計算の手順は実装が選べるようにした（§5.2）。別モデルの案のうち、件数の整数キー 1 つにする形と、直近重みを外す形は採らなかった（§4.1）。§2.3 に Q の学習の開始が 20,000 の構成、モードが読まない値と checkpoint の今の扱い、facade 側の設定の読み口、走っている bat の事実を足し、§2.5 の開発者の問いに答えの所在を足し、§2.6（直し方の見立て）を新設した。§3 の用語を決定に合わせて直した。
> 関連: [116_rnd_lifelong_novelty_10prd.md](116_rnd_lifelong_novelty_10prd.md)（RND の本体。§8 の 1・3・5・6 を本 PRD へ移した。§9）、[121_extrinsic_intrinsic_value_decomposition_10prd.md](121_extrinsic_intrinsic_value_decomposition_10prd.md)（外的と内的の価値の分解）、[924_intrinsic_reward_novelty_10prd.md](924_intrinsic_reward_novelty_10prd.md)（NGU 型の内的報酬）、[117_warn_policy_and_unread_config_10prd.md](117_warn_policy_and_unread_config_10prd.md)（WARN の線引き）、設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md)（RND の解説）・[200 §6.2](../design/200_dqn_agents.jp.md)、課題 [issue-20261111-77ac3c](../management/issues/issue-20261111-77ac3c.md)（本 PRD の仕事）・[issue-20261108-227b05](../management/issues/issue-20261108-227b05.md)（predictor の周期リセットを試して戻した記録）。

## 1. 目的と問題

### 問題

DefaultDQNAgent の RND（PRD116、既定の `statistics_mode = whiten`・`multiplier_mode = clipped`）は、1 step の内的報酬を β × clip((e − μ_e) ÷ σ_e, 0, L − 1) で作る（L は `multiplier_max`、既定 5）。e は predictor と target の出力の二乗誤差、μ_e・σ_e は誤差の平均と標準偏差（以下、合わせて「誤差の統計」）である。誤差の統計は、Run の最初から採点したすべての誤差の累積（Welford）で、warm-up 中の誤差も入る（§2.3）。

序盤の大きな誤差が σ_e を膨らませ、次の 3 つが起きた（§2.1・§2.2）。

1. 内的報酬が、序盤を除いた誤差の統計で作った場合の 1/3〜1/17（Montezuma）、1/7〜1/19（KFM の 10M 時点）に抑えられている。
2. 抑え方は Run が進むと薄まり、実効の β（実際に効いている内的報酬の大きさ）が Run の中で 3〜5 倍に上がる。設定に書いていないスケジュールになっている。
3. 平均を引いて 0 で切るので、内的報酬が付く状態は約 1 割で、残りは 0 になる。学習とともに誤差が下がると、今の誤差が累積の平均を下回り続ける（LunarLander・GridMaze で内的報酬が消えた現象と同じ形。[wiki の内的報酬のページ](../wiki/mechanisms/intrinsic_novelty.md)）。

### ゴール

1. whiten の内的報酬の大きさが、序盤（warm-up 中）の一時的な大きな誤差に左右されない（問題の 1）。対象は、warm-up を誤差が落ち着くまで取った構成（Atari の `run.@rnd`。D6）である。LunarLander・GridMaze は warm-up の長さを変えないので、この改善を狙わない（D6）。
2. 序盤の誤差が原因の、実効の β の隠れた強まりが無い（問題の 2）。warm-up 明けのあとに残る緩やかな変化は、大きさを測って記録する（§7.1）。
3. 直近に重みを置く誤差の統計と、平均を引かない形を、同じ部品の設定で比べられる（問題の 3）。問題の 3 は、既定では「慣れた状態の内的報酬は消える」という一生の新しさの性質として残し（D1・D2）、既定を変えるかは Run の結果で決める（§8 の U3）。
4. 今の中身を同じ実行体で再現でき、変更の前後を比べられる。
5. 終端（done）の遷移の内的報酬を 0 にするかを、設定で選べる（§2.7）。
6. 課題 issue-20261111-77ac3c の事象のうち、内的報酬が抑えられることと、Run の中で強まることが片付く。

### RND 単独の群の位置づけ

RND 単独の群は、NGU（[924](924_intrinsic_reward_novelty_10prd.md)）への通過点である。本 PRD の目的は、924 が倍率として使う部品の尺度を直すことで、既定の式（NGU の倍率の形。§2.5）は変えない（D1）。文献と数値を並べる再現は本 PRD の範囲に入れず、Taiga らの条件は `run.@rnd_taiga`（PRD116）、Burda らの形は PRD121・924 が持つ。

### 仮説・比べ方・終わり方

- 仮説: warm-up 中の誤差を誤差の統計から外せば、内的報酬の大きさが今の中身を上回り、誤差の標準偏差が Run の中でほぼ一定になる。大きさの見込みは今の中身の 10 倍以上だが、倍率は合否にせず記録する（§7.1）。
- 比べ方: KFM の 10M で、新しい既定の群と、同じ実行体で今の中身を再現した群を並べる（§7.1）。
- 終わり方: §7.1 の主の条件を満たせば、ゴール 1・2 を満たしたとして終える。成績が上がるかは副の指標で、β の選び直しは Run の結果で決める（§8 の U1）。満たさなければ §8 の U2 を開き直す。

### 非ゴール

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| Burda らの形（e ÷ 内的報酬の割引収益の標準偏差） | lane ごとの割引収益の保持と、外的・内的の価値を分ける話（PRD121）と組になる。別の関心として PRD121・924 に残す | 制約ではない（別 PRD で入れられる） |
| `variance_feedback`（Taiga らの統計）の中身を変える | Taiga らの条件を再現するためのモードで、再現の意味が変わる | 再現のために変えない |
| 観測の正規化の式（累積の統計、除数の ε 1.0、clip 5）を変える | 序盤の未熟な正規化から来る大きな誤差は、warm-up 中の誤差を統計から外せば効かなくなる見込み。効果を見てから決める（§8 の U2） | 制約ではない（Run の結果で開き直す） |
| 内的報酬の運び方（スケール後の外的報酬に足す） | 外的・内的を別に持つのは PRD121 の題材 | 制約ではない |
| predictor の周期リセット | GridMaze・LunarLander で利得が無く戻した（issue-20261108-227b05） | 制約ではない（試した結果で退けた） |
| Burda らのように、学習の前に乱数行動で観測の統計を作る | Actor の行動の契約に触れる別の関心 | 制約ではない |
| 内的の価値（収益）を game over で切らない形 | Q を外的と内的に分ける必要がある別の関心で、PRD121 が持つ。本 PRD は終端の 1 step の内的報酬を選べるようにするところまで（D16） | 制約ではない（PRD121 で入れられる） |

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
- RND の採点と predictor の学習は exp step 0 から始まり、Q の学習の開始とは独立である。Q の学習の開始（`learner.update_warmup_steps`）は、KFM の実時間優先の構成が 200,000（[Atari.txt:318](../../apps/runner/config/Atari.txt)）、Montezuma の Taiga ら寄せが重ねる `run.@rainbow_like` が 20,000（[Atari.txt:94](../../apps/runner/config/Atari.txt)）で、構成によって違う。KFM の s04 では 200k の時点で `32_learn_count` が 1,465 だった。
- 観測の正規化（whiten）も累積の統計で、初期値は平均 0・分散 1、除数は √分散 ＋ 1.0（[212 §3.2](../design/212_rnd_lifelong_novelty.jp.md)）。
- 内的報酬の合成は facade（[default_dqn_agent.cpp:713-735](../../core/anet-core/src/default_dqn_agent.cpp)）で、β × (α − 1) をスケール後の外的報酬に足し、`next_state` を採点するときは done の lane を 0 にする（[同 717-721](../../core/anet-core/src/default_dqn_agent.cpp)）。`truncated` だけの lane は 0 にせず、`intrinsic.observation = state` では done の lane も 0 にしない。採点・統計の更新・補助学習は、done の lane も対象にする（212 §3.3）。facade 側の設定（`intrinsic.observation` など）は `DefaultDQNAgentConfig` が読む（[default_dqn_agent.hpp:117-131](../../core/anet-core/include/anet/default_dqn_agent.hpp)）。
- 序盤の大きな誤差が、predictor が未学習なためか、観測の正規化が未熟なためかは分けていない（`56_agent_in` の tag は `@learn` で記録され、Q の学習が始まるより前の値が無い）。
- モードが読まない値は、検証したうえで、解決した値が実装の既定と違えば構築時に WARN を 1 回出して使わない（[intrinsic_rnd.cpp:275-294](../../core/anet-core/src/intrinsic_rnd.cpp) の `variance_feedback` での `obs_norm.*`、[同 257-259](../../core/anet-core/src/intrinsic_rnd.cpp) の `unclipped` での `multiplier_max`）。
- checkpoint は、RND の書式の識別子と `statistics_mode` を先に読み、どちらかが設定と違えば ERROR にする（[intrinsic_rnd.cpp:482-490](../../core/anet-core/src/intrinsic_rnd.cpp)）。
- 走っている Run への影響: メイン機の [12_batch_run.bat](../../apps/12_batch_run.bat) の Montezuma のキュー（314〜355 行）は、凍結した 2026-11-08 の実行体で、リポジトリの `Atari.txt` を Run の開始ごとに読む。stage 18〜20 は `run.@rnd` を使う whiten の Run で、終わりは 2026-11-12 の 22 時ごろの見込みである。凍結した実行体は `intrinsic.rnd.warmup_steps` を読むので、`Atari.txt` の `run.@rnd` の warm-up を変えると、これらの Run の条件が変わる。`agent.txt` に新しい設定キーを `?=` で足しても、凍結した実行体は読まない。

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
- 開発者の問い（2026-11-11）: 今の動作は NGU とも Taiga らとも Burda らとも違う独自の形になっているか、その妥当性はどうか。答えは上の 4 項目で、位置づけは再グリルで §1 のとおりに決めた（D1）。

### 2.6 直し方の見立て（2026-11-11 の再グリルで整理）

§2.1 の表から計算した概算で、Run で確かめた値ではない。

- **膨らみの主因は warm-up の前後の誤差である。** KFM の 200k 時点の σ_e（1.40〜4.60）は、200k 以降だけの σ_e（0.026〜0.034）の 52〜135 倍（分散では約 2,700〜18,200 倍）である。最初の 200k が二乗偏差の総量の 97〜99% を占める（§2.1）。この区間を誤差の統計から外せば、膨らみの主因が無くなる。
- **直近に重みを置く統計だけでは直らない。** 最初の 20 万件の誤差を統計に入れたまま、重みが時定数 H ごとに 1/e に減る平均（§5.2）に替えた場合を見積もる。序盤の区間の重みは、区間の終わりから t 件後に (1 − e^(−20 万 ÷ H)) × e^(−t ÷ H)、そのあとの誤差の重みは 1 − e^(−t ÷ H) に比例する。H = 100 万件なら、標準偏差の膨らみが 2 倍を切るのは t が 510〜700 万件、1.05 倍を切るのは 850〜1,040 万件である（平均のずれの寄与は分散の 3% 以下なので入れていない）。KFM の 10M の Run では、ほぼ最後まで膨らみが残る。
- **統計を累積のままにして warm-up 中の誤差だけを外した場合。** KFM の 10M では、200k 以降だけの μ_e は約 0.022、σ_e は 0.026〜0.034、その時点の誤差は 0.012〜0.020 なので、普段の状態の z は約 −0.36〜−0.07 になる。一生の平均より誤差が大きい状態にだけ内的報酬が付き、誤差が下がるにつれて、付く lane の割合はゆっくり下がる。
- **直近に重みを置く統計は、内的報酬の意味を変える。** 累積なら、慣れた状態の内的報酬は 0 へ消える。直近に重みを置くと、直近の平均との比較になるので消えず、割合と大きさが Run を通してほぼ一定になる（誤差が正規分布なら、1 step の平均で β の約 0.4 倍）。覚えきった env（GridMaze は入力が 25 通り以下で誤差が 1e-9。wiki の内的報酬のページ）では、時定数の数倍が過ぎると統計も今の大きさに縮み、微小な差に内的報酬が付き続ける。

### 2.7 終端の遷移の内的報酬（2026-11-11 の再グリルで整理）

anet の既定は、`next_state` を採点するとき、done の lane の内的報酬を 0 にする（§2.3）。PRD116 の D2 が置いた暫定の既定で、理由は「終端の観測に内的報酬を付けると、終端そのものが報酬になる」だった。facade に届く done の lane の次観測は、終端の観測そのものである（PRD116 §2）。

- Taiga らは、終端の遷移も行動した観測 x_(T−1) で採点し、0 にしない（PRD116 §2.1。原典のコードで確認）。anet の `intrinsic.observation = state` は同じ動きである。
- Burda らは、内的の収益を game over で切らず、外的と内的で価値の Head を分ける（PRD121 §2.2。論文 §2.3）。終端の遷移の内的報酬を 0 にする処理は論文に書かれていない。公開実装のコードは、本 PRD では確かめていない。
- 違いは 2 つあり、大きさが違う。終端の 1 step の内的報酬を 0 にすることは、1 エピソードにつき 1 step の差である（KFM は 1 エピソードが数千 step）。内的の価値を game over で切ることは、内的報酬が正で付く限り、死ぬと先の内的報酬を失うので、長く生きる向きの力になる。後者を変えるには Q を外的と内的に分ける必要があり、PRD121 の題材である。
- 開発者の問い（2026-11-11）: 終端で内的報酬を 0 にしているのは、ほかの RND の実装と違うのではないか。変えるか。再グリルで、設定で選べるようにし、既定は 0 のままにすると決めた（D16）。

## 3. 用語

| 語 | 意味 |
|---|---|
| 誤差の統計 | RND の採点で、誤差 e を標準化するための平均 μ_e と標準偏差 σ_e |
| 累積 | 誤差の統計の取り方の 1 つ。統計に入れたすべての誤差を同じ重みで平均する |
| 直近重み | 誤差の統計の取り方の 1 つ。直近の誤差ほど重く、古い誤差の重みが時定数ごとに 1/e に減る。[CONTEXT.md](../../CONTEXT.md) の「バイアス補正EMA」の形で、重みの和で割って読み出す |
| 時定数 | 直近重みで、誤差の重みが 1/e に減るまでの、採点した観測の件数（設定キー `error_norm.ema_steps`） |
| 実効の β | 実際に効いている内的報酬の大きさを、名目の β と比べて言うときの語 |

`CONTEXT.md` には足さない（D13）。「内的報酬」「一生の新しさ」は PRD116 §3 の語で、用語集への追加は 924 の裁定に合わせる。直近重みは、用語集にある「バイアス補正EMA」で言い換えられる。

## 4. 確定事項

2026-11-11 のグリルで決めた。理由が他の決定や事実に依存するときは名指す。設定キーは `DefaultDQNAgent.` を省いて書く。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | **RND 単独の群は NGU への通過点とし、既定の式（NGU の倍率の形。β × (α − 1)、α = clip(1 ＋ (e − μ_e) ÷ σ_e, 1, L)）は変えない。** 平均を引かない形と直近重みは、比べるためのモードとして足す | 924 で NGU の倍率として使うときに部品を変えずに済む（PRD116 D4。§2.5）。単独の形は文献に結果が無いが、文献と数値を並べる再現は `run.@rnd_taiga` と PRD121・924 が持つ。2026-11-07 の開発者の発言「RND は NGU への通過点」（PRD116 §8.1 の 2）と同じ考えであることを、再グリルで確かめた。依存: なし |
| D2 | **whiten の誤差の統計の取り方を `intrinsic.rnd.error_norm.average_mode`（`cumulative`｜`ema`）で選ぶ。既定は `cumulative`（累積）。** `ema` は直近重みである | 膨らみの主因は warm-up の前後の誤差で、D4 で除ける（§2.6）。既定を累積にすると、「慣れたら消える」という一生の新しさの性質と、NGU の倍率の前提（見慣れた状態は倍率 1）を保てる。直近重みを既定にすると、時定数という env ごとの調整値と、覚えきった env で微小な差を拾う帰結（§2.6）が既定に入る。直近重みを既定にするかは、Run で 2 つを比べてから決め直す（§8 の U3）。依存: D1、D4 |
| D3 | **直近重みは、採点した観測 1 件ごとに重みを 1 − 1 ÷ `ema_steps` 倍に減らし、重みの和で割って読み出す（§5.2）。** 時定数は `intrinsic.rnd.error_norm.ema_steps`（採点した観測の件数、整数 ≥ 1、既定 1,000,000）。batch の N 件は、lane の順に 1 件ずつ入れたのと同じ値にする | 重みの和で割るので、記録が少ない間も平均が 0 側へ偏らない（用語集の「バイアス補正EMA」）。件数で数えるので、lane の数を変えても同じ観測を同じ順に採点すれば同じ統計になる（`warmup_steps` と同じ単位。212 §3.5）。既定の 100 万件は、Atari の 10〜30M の Run で誤差が数 M かけて 2 割ほど下がる動き（§2.1 の `13_error_step`）より短く、1 batch（128 件）の揺れより十分長い。依存: D2 |
| D4 | **whiten の warm-up の中身を `intrinsic.rnd.warmup_mode`（`learn_predictor`｜`collect_errors`）で選ぶ。既定は `learn_predictor`。** `learn_predictor` は、warm-up 中に predictor を学習させ、誤差を誤差の統計に入れない。`collect_errors` は今の中身で、predictor を学習させず、誤差を誤差の統計に入れる。どちらも warm-up 中の倍率は 1 で、観測の統計は更新する（§5.1） | 累積が既定（D2）なので、未学習の predictor の誤差を統計に入れないには、統計を取り始める前に predictor を学習させておくしかない。書ける組を 2 つに限るのは、要るのが新しい中身と今の中身だけだからで、「predictor を学習させるか」と「誤差を統計に入れるか」を別々のスイッチにすると、膨らみが再発する組（学習させず、統計にも入れない。warm-up 明けに未学習の誤差が入る）が黙って書ける。「誤差を統計に入れ始める件数」を整数 1 つで持ち、0 を今の中身とする形（別モデルの検討で出た案）を採らないのは、開発者の判断（2026-11-11）による。設定値と動作の切り替えは分ける。統計を取り始めるのを待たせるかどうか（動作の切り替え）と、何件待つか（値）は別の関心なので、切り替えはモードのキー、件数は値のキー（`warmup_steps`）で表す（[AGENTS.md](../../AGENTS.md) §設定値の扱いの「意図の分岐が必要な場合はモード値として明示させる」）。依存: D2、D5 |
| D5 | **今の中身を再現する組を残す。** `warmup_mode = collect_errors`・`warmup_steps = 12,800`・`error_norm.average_mode = cumulative`・`error_norm.use_centering = true`・`multiplier_mode = clipped`・`intrinsic.use_done_mask = true` の whiten は、本 PRD の実装前（PRD116）と同じ値になる | 開発者の選択（2026-11-11）。whiten でも「実装前に保存した実行体と metrics checksum が一致」を受入にでき（§7.2 の 1）、新旧の RND を同じ実行体で比べられる（§7.1）。§2.1 では基準と RND の実行体が違い、RND の効果と分けられなかった。`collect_errors` は現行契約のモード値で、旧キーの別名・互換読み・旧契約専用の検出ではない（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針の互換層に当たらない）。依存: D2、D4 |
| D6 | **warm-up の長さ（`intrinsic.rnd.warmup_steps`）は、`agent.txt` の既定を 12,800 件のままにし、Atari の `run.@rnd` を 200,000 件にする。** `Atari.txt` の行は、メイン機の 12_batch_run.bat の Montezuma のキューが終わってから入れる。それまでの新しい Run は、bat の引数で `warmup_steps` を渡す。`agent.txt` の新しい設定キーは、いつ入れてもよい | KFM の誤差は 200k で 0.057、250k で 0.026 と落ち着く途中にあり（s04）、それより前を統計に入れない。Q の学習の開始とは結び付けない（構成によって 200,000 と 20,000 がある。§2.3）。LunarLander と GridMaze は Run が短く、序盤の内的報酬が要るので、warm-up の長さを変えない。この 2 つは、warm-up の中身の既定の変更（D4）で動きが変わる（predictor が 0 件から学習し、最初の 12,800 件の誤差が統計に入らない）が、12,800 件では predictor が学習の途中で、膨らみは残る見込みである。動きが変わることを確かめる Run は本 PRD に入れない（開発者の選択、2026-11-11。2026-11-08 の LunarLander の確認は今の中身でのもの）。時機は、凍結した実行体が `Atari.txt` を Run の開始ごとに読む（§2.3）ため。依存: D4 |
| D7 | **誤差から平均を引くかを `intrinsic.rnd.error_norm.use_centering`（bool、既定 true）で選ぶ。** false は平均を引かない形で、α = 1 ＋ e ÷ σ_e（`clipped` なら L で切る）。割る値は、平均を引く形と同じ σ_e である。clip の有無（`multiplier_mode`）とは独立で、4 通りを書ける（§5.3） | clip の有無と、平均を引くかは別の軸である。`multiplier_mode` の 3 つ目の値にすると、clip の有無を表すキーに別の軸が混ざり、平均を引かず clip もしない形（Burda らの形に近い比較の候補）が書けない。語は観測側の `obs_norm.use_centering` に揃える（[実装 品質確認観点](../quality/impl.md)の「名前」の 3）。依存: D1、D12 |
| D8 | **`statistics_mode = variance_feedback` の中身と warm-up は変えない。** `warmup_mode`・`error_norm.average_mode`・`error_norm.ema_steps`・`error_norm.use_centering` は whiten だけが読む。`variance_feedback` では、値を検証したうえで、解決した値が実装の既定と違えば、設定キーごとに構築時に WARN を 1 回出し、値は使わない | Taiga らの再現のためのモードで、`run.@rnd_taiga` は warm-up 0 である。読まない値の扱いは、今の `obs_norm.*` と同じ（§2.3）で、[PRD117](117_warn_policy_and_unread_config_10prd.md) の D1 の 3 の (b)（書いた値が、ほかの設定のために使われない）とも合う。平均を引かない指定だけを ERROR にすると、同じ階層の設定キーの扱いが割れる。依存: D2、D4、D7 |
| D9 | **誤差の統計が空か、標準偏差が 0 の間は、倍率を 1 にする（今の規則のまま）。** warm-up 明けの最初の batch は統計が空なので倍率 1 で、その batch の誤差から統計を取り始める。統計の最小件数を決める設定キーは置かない | lane が 128 なら、2 batch 目から 128 件の統計で作れる。`unclipped` で lane が極端に少ない構成では、最初の数 batch の値が大きく振れうるが、今の warm-up 0 と同じ性質で、新しい設定キーを足すほどの要求が無い。依存: D4 |
| D10 | **メトリクスの tag は増やさない。** `56_agent_in/11_error_mean`・`12_error_std` は、次の batch の採点に使う誤差の統計（`average_mode` に応じて累積か直近重み）を出す。誤差の統計が空の間（`learn_predictor` の warm-up 中）は NaN にする | 今の tag で膨らみを読めた（§2）。warm-up 明けの落ち着きは `13_error_step` で読める。NaN は「値が今は成立していない」の表し方で、今の未採点・未学習の扱いと同じ（212 §4.4）。依存: D2、D4 |
| D11 | **RND の checkpoint の書式の識別子を、誤差の統計の中身の形で分ける。** `cumulative` と `variance_feedback` は、今の識別子 `RndNoveltyEstimator-3` と今の中身のまま変えない。`ema` のときだけ、識別子を `RndNoveltyEstimator-4` にし、誤差の統計を直近重みの中身で保存する。読み込みでは、識別子が設定の取り方に対応するものと違えば ERROR にする（§6.3）。`warmup_mode`・`error_norm.use_centering`・`error_norm.ema_steps` は保存せず、再開時は今の設定に従う | 開発者の選択（2026-11-11。別モデルの検討が、書式を一律に上げると今ある checkpoint が読めなくなることを挙げた）。累積の統計は保存する中身が変わらないので、識別子も変えなければ、今ある RND の checkpoint（Montezuma の 30M など）を新しい実行体でそのまま読める。直近重みは中身の形が違うので、別の識別子にして取り違えを ERROR にする（今の `statistics_mode` の検査と同じ考え方。§2.3）。旧い契約を読み替える互換分岐ではない（累積の中身が変わらないだけで、識別子ごとに読む中身は 1 つ）。保存しない 3 つは状態ではなく式の選択で、今の `multiplier_mode` も保存していない。依存: D2 |
| D12 | **名前。** 階層は `intrinsic.rnd.error_norm.*`。設定キーは `error_norm.average_mode`・`error_norm.ema_steps`・`error_norm.use_centering`・`warmup_mode`、値は `cumulative`｜`ema` と `learn_predictor`｜`collect_errors` | 暫定の名前から改めた理由（実装 品質確認観点の「名前」の 9）: `error_stats.mode` は隣の `statistics_mode` と紛れ、何のモードかを言わない（同 2・4）。階層は観測側の `obs_norm.*` と対にする（同 1・3）。値 `lifelong` は「一生の新しさ」の口の名前（`LifelongNoveltyEstimator`）と紛れるので、取り方を表す `cumulative` にする（同 2）。`horizon` は割引や n-step の語と紛れるので、同じ単位の `warmup_steps` に揃えて `ema_steps` にする（同 3）。`predictor`｜`statistics` は、`statistics` が `statistics_mode` と紛れ、warm-up で何をするかが読めないので、することを表す語にする（同 4）。依存: D2、D3、D4、D7 |
| D13 | **文書は、設計文書 212（jp・en）と 200（jp・en）の該当箇所を直す。新しい設計文書と ADR は起こさず、`CONTEXT.md` に語を足さない** | 新しい仕組みではなく、RND の統計の取り方と warm-up の変更で、212 が仕組みの正本である。ADR を起こす条件のうち「戻しにくい」に当たらない（設定のモードと既定値の変更）。用語は §3 のとおり、既存の語で言い換えられる。依存: なし |
| D14 | **効果は KFM の 10M で確かめる。** サブ機で、新しい既定 4 本・直近重み 2 本・今の中身 2 本・平均を引かない形 2 本の 10 本を回す（§7.1）。終端の mask を外した群は入れない。Run の段取りは、起こす前に別に示して承認を受ける | 新しい既定と、同じ実行体で再現した今の中身を並べれば、実行体の違いを外して内的報酬の大きさを比べられる（D5）。平均を引かない形は、§5.3 の帰結で β を 0.01 に下げる。終端の mask は 1 エピソードにつき 1 step の差で、内的報酬の大きさの判定に効かない（§2.7）。依存: D2、D4、D5、D7、D16 |
| D15 | **PRD121 を実装して比べるときは、誤差の統計と warm-up の条件をそろえるか、基準を取り直す** | PRD121 の比較の基準（Montezuma の b001・b01）は、今の中身の Run である。§2.2 の膨らみは β と同じ働きをするので、条件が違うと β の比較が崩れる。依存: D2、D4 |
| D16 | **`next_state` を採点するとき、done の lane の内的報酬を 0 にするかを `intrinsic.use_done_mask`（bool、既定 true）で選ぶ。** true は今と同じで 0 にする。false は、終端の観測を採点した値 β × (α − 1) をそのまま足す。`truncated` だけの lane の扱いと、採点・統計の更新・補助学習が done の lane も対象にすることは変えない。`intrinsic.observation = state` では今と同じく 0 にせず、この設定キーを読まない（解決した値が実装の既定と違えば、構築時に WARN を 1 回出す） | 開発者の選択（2026-11-11）。Taiga らは終端の遷移の内的報酬を 0 にせず、Burda らは内的の収益を game over で切らない（§2.7）ので、ほかの実装と同じ形を選べるようにする。既定を 0 のままにするのは、anet が終端の観測そのものを採点するので、終端が報酬になる懸念（PRD116 D2）が残ることと、今の中身の再現（D5）を変えないため。`intrinsic.observation` の値に足さず別の設定キーにするのは、どの観測を採点するかと、done の lane を 0 にするかが別の軸だからである（D7 と同じ考え方）。`state` で読まないのは、採点する観測が終端の観測ではなく、0 にする理由が無いため（PRD116 D5）。名前は、既存の bool の形（`use_centering`・`use_double_dqn`）と、設計文書の語（done の mask。212 §3.3）に揃える。設定キーは facade が読む階層（`intrinsic.*`）に置く（0 にするのは facade で、RND の口は終端を知らない。PRD116 D3）。依存: D5 |

### 4.1 選ばなかった形

| 選ばなかった形 | 裁定 | 理由 |
|---|---|---|
| 直近重みを既定にする | 先送り（実験の結果が要る） | D2。再開条件は §8 の U3 |
| 今の中身を再現する組を残さない（warm-up の中身を新しい 1 つだけにする） | 不採用 | D5。whiten の checksum の一致を受入にできず、新旧を同じ実行体で比べられない |
| warm-up の中身を、独立したスイッチ 2 つ（predictor を学習させるか、誤差を統計に入れるか）で選ぶ | 不採用 | D4。要求の無い組が 2 つ増え、うち 1 つは膨らみが再発する |
| 前半は観測の統計だけ、後半は predictor も学習させる 2 ステージの warm-up | 先送り（実験の結果が要る） | 効果が未測定で、設定キーが増える。再開条件は §8 の U2 |
| warm-up の長さを Q の学習の開始に自動で合わせる | 不採用 | D6。Q の学習の開始は構成によって違い、warm-up の長さの理由は Q の開始ではない |
| `multiplier_mode` に 3 つ目の値 `uncentered` を足す | 不採用 | D7 |
| `variance_feedback` と、平均を引かない指定の組を ERROR にする | 不採用 | D8。同じ階層の設定キーの扱いを WARN にそろえる |
| 統計の最小件数を決める設定キー | 不採用 | D9 |
| 累積と直近重みの両方の値を並べて出す tag | 不採用 | D10。今の tag で足りる |
| checkpoint の書式の識別子を一律に `RndNoveltyEstimator-4` へ上げ、取り方を中に書く | 不採用 | D11。識別子は 1 つで済むが、今ある RND の checkpoint が新しい実行体で読めなくなる |
| 直近重みのモードを本 PRD から外す（別モデルの検討で出た案） | 不採用 | D2・D11。累積の書式を変えなければ checkpoint の不利益は無く、直近重みを比べる手段を残せる |
| 誤差を統計に入れ始める件数を整数 1 つ（0 が今の中身）で持ち、`warmup_mode` を作らない（別モデルの検討で出た案） | 不採用 | D4。値に動作の切り替えを兼ねさせず、切り替えはモードのキーで表す |
| LunarLander・GridMaze の `run.@rnd` に `warmup_mode = collect_errors` を明示して動きを変えない | 不採用 | D6。新しい既定を使わない env を残さない |
| LunarLander の比較（3 seed × 2M）を受入に足す | 不採用 | D6。目的は Atari の尺度で、内的報酬が早く消える env では中身の差が成績に効く見込みが小さい |
| 「誤差の統計の取り方」を ADR にする | 不採用 | D13 |
| 終端の遷移の内的報酬を、既定で 0 にしない（ほかの実装と同じにする） | 不採用 | D16。終端が報酬になる懸念が残り、今の中身の再現に指定が 1 つ増える |
| 終端の遷移の内的報酬を変えずに、PRD116 の未決のままにする | 不採用 | D16。開発者の選択で、選べるようにする |
| `intrinsic.observation = state` でも done の lane を 0 にできるようにする | 不採用 | D16。0 にする理由の無い組が増える |

### 4.2 起票時の未決表との対応

起票時（暫定）の未決表の行と、再グリルでの行き先の対応である。

| 起票時の行 | 行き先 |
|---|---|
| 1（warm-up 中にすること） | D4 |
| 2（warm-up の長さ） | D6 |
| 3（統計のモードと既定） | D2。既定を直近重みにするかは §8 の U3 |
| 4（時定数の既定と単位）・5（式） | D3 |
| 6（平均を引かない形の名前と式） | D7 |
| 7（`variance_feedback` との組） | D8 |
| 8（メトリクス） | D10 |
| 9（checkpoint） | D11 |
| 10（実験の組み立て） | D14、§7.1 |
| 11（β の既定） | §8 の U1 |
| 12（`variance_feedback` の warm-up） | D8 |
| 13（観測の正規化） | §8 の U2 |
| 14（設計文書・ADR・用語集） | D13 |
| 15（名前） | D12。クラス名は §6.4 |
| 16（走っている bat への影響） | 事実は §2.3、時機は D6 |
| 17（PRD121 の実験との関係） | D15 |
| 18（今の挙動を再現するモード） | D5 |
| 19（RND 単独の群の目的と式の出どころ） | D1 |
| （起票時の表に無い。再グリルの途中で開発者が出した問い）終端の遷移の内的報酬 | D16 |

## 5. 仕様

### 5.1 whiten の warm-up（D4・D6・D9）

warm-up は、採点した観測の件数が `warmup_steps` に達するまでの期間である（今と同じ）。その間にすることを `warmup_mode` が決める。

| | `learn_predictor`（既定） | `collect_errors`（今の中身） |
|---|---|---|
| 倍率 α | 1 | 1 |
| 観測の統計 | 更新する | 更新する |
| 誤差の統計 | 更新しない | 更新する |
| predictor の学習（`Learn`） | 進める（true を返す） | 進めない（false を返す） |

- warm-up かどうかの判定は今と同じである。採点は batch の採点前の件数が `warmup_steps` 未満なら warm-up、`collect_errors` の `Learn` は採点後の件数が `warmup_steps` 以下なら進めない（212 §3.5）。
- `learn_predictor` では、warm-up 明けの最初の batch は誤差の統計が空なので α = 1 で、その batch の誤差を統計に入れる。次の batch から統計で α を作る（D9）。
- `warmup_steps = 0` は warm-up を置かない指定で、`warmup_mode` はどちらでも同じ動きになる（最初の batch から誤差を統計に入れ、最初の `Learn` から predictor を学習させる）。利用者が warm-up を止めたうえで、その中身の指定が効かないだけなので、止めた機能を参照する設定に当たる（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表の「意図された休止・無効化」）。`warmup_mode` の解決した値が実装の既定と違うときだけ、構築時に INFO を 1 回出す（§6.1）。ほかのモードのために値が使われない場合の WARN とは分ける。
- facade の credit の扱いは変えない。`Learn` を呼んだ分の credit は、結果によらず消費する（212 §3.5）。`learn_predictor` では warm-up 中も `learn_count` が増える。

### 5.2 誤差の統計の取り方（D2・D3）

`error_norm.average_mode = cumulative` は今の累積で、値を変えない。

`ema` は、採点した観測 1 件ごとに重みを d = 1 − 1 ÷ `ema_steps` 倍に減らした、重み付きの平均と母分散である。契約は、次の式で決まる値（lane の順に 1 件ずつ入れたのと同じ値）で、計算の手順は実装が選んでよい（平均に比べて散らばりが極端に小さいときに桁落ちしない、差分で更新する形など）。S0・S1・S2 は説明のための記号で、重みの和、誤差の重み付きの和、誤差の二乗の重み付きの和を表す。初期値はどれも 0 で、batch の N 件（lane の順に e_0 〜 e_(N−1)）を次のとおり入れる。

```text
S0 ← d^N × S0 ＋ Σ_{i=0..N-1} d^(N-1-i)
S1 ← d^N × S1 ＋ Σ_{i=0..N-1} d^(N-1-i) × e_i
S2 ← d^N × S2 ＋ Σ_{i=0..N-1} d^(N-1-i) × e_i²
μ_e = S1 ÷ S0、σ_e² = max(S2 ÷ S0 − μ_e², 0)
```

- S0 が 0（統計が空）か、σ_e が 0 なら α = 1（D9）。
- 採点は、どちらの取り方でも、今回の batch を入れる前の μ_e・σ_e で行う（今と同じ）。
- lane の数を変えても、同じ観測を同じ順に採点すれば同じ統計になる。
- `ema_steps = 1` は d = 0 で、統計は直前の 1 件だけになり、σ_e が 0 なので α は常に 1 になる。設定どおりの動きとして止めない。
- 費用は 1 batch あたり N 個の積和で、採点の forward に比べて無視できる。

### 5.3 倍率の式（D1・D7）

whiten の倍率 α と内的報酬（β × (α − 1)。done の lane の扱いは §5.5）は、次の 4 通りになる。

| `error_norm.use_centering` | `multiplier_mode` | α | 内的報酬 |
|---|---|---|---|
| true（既定） | `clipped`（既定） | clip(1 ＋ (e − μ_e) ÷ σ_e, 1, L) | β × clip((e − μ_e) ÷ σ_e, 0, L − 1) |
| true | `unclipped` | 1 ＋ (e − μ_e) ÷ σ_e | β × (e − μ_e) ÷ σ_e（負もある） |
| false | `clipped` | min(1 ＋ e ÷ σ_e, L) | β × min(e ÷ σ_e, L − 1) |
| false | `unclipped` | 1 ＋ e ÷ σ_e | β × e ÷ σ_e（上限なし） |

- `multiplier_max`（L）は、今と同じく `clipped` のときだけ読む。
- `clip_ratio` は、今と同じく α が L に等しい lane の割合で、`unclipped` では 0 である。

平均を引かない形の帰結: 内的報酬が 1 step ごとに正で付くので、`next_state` で done の lane を 0 にする今の合成では、エピソードを長く続けるほど得になる向きの力も生む。§2.1 の 200k 以降の値（μ_e 約 0.022、σ_e 約 0.028）なら、普段の状態で 1 step 約 0.8β になり、β 0.1 では KFM の外的報酬（雑魚 1 体 +1 が数十 step に 1 回）を上回る。平均を引かない形を試す β は小さく取る（D14）。

### 5.4 処理の順（whiten）

| 順番 | 処理 | 変わるところ |
|---:|---|---|
| 1 | 外的報酬をスケールする | なし |
| 2 | 観測を正規化する（今回の観測を入れる前の観測の統計） | なし |
| 3 | 誤差を計算する（更新前の predictor） | なし |
| 4 | α を求める（今回の誤差を入れる前の誤差の統計。warm-up 中と統計が空のときは 1） | `ema` のときの μ_e・σ_e、`use_centering = false` の式 |
| 5 | 観測の統計を更新する。誤差の統計を更新する | `learn_predictor` では、warm-up 中は誤差の統計を更新しない |
| 6 | β を評価し、done を 0 にして内的報酬を足す | `intrinsic.use_done_mask = false` では done を 0 にしない（§5.5） |
| 7 | facade が credit に応じて `Learn` を呼ぶ | `learn_predictor` では、warm-up 中も predictor を学習させる |
| 8 | 経験を内側の Learner へ渡す | なし |

### 5.5 終端の遷移の内的報酬（D16）

done の lane の内的報酬は、facade が次のとおりに決める。`statistics_mode`・`multiplier_mode`・`error_norm.*` によらない。

| `intrinsic.observation` | `intrinsic.use_done_mask` | done の lane の内的報酬 |
|---|---|---|
| `next_state`（既定） | true（既定） | 0（今と同じ） |
| `next_state` | false | β × (α − 1)（終端の観測を採点した値） |
| `state` | 読まない | β × (α − 1)（今と同じ。行動した観測を採点した値） |

- `truncated` だけの lane は、どの組でも内的報酬を足す（今と同じ）。
- 採点・観測と誤差の統計の更新・補助学習は、どの組でも done の lane を対象にする（今と同じ）。変わるのは、学習報酬へ足す値だけである。
- `bonus_mean` などの内的報酬の診断値は、今と同じく、実際に足した値から作る。

## 6. 実装契約

### 6.1 設定と失敗の扱い（D2〜D8・D12・D16）

接頭辞は `DefaultDQNAgent.`。既定は `agent.txt` の `@baseline` に `?=` で置く。`intrinsic.rnd.*` は RND の設定（Factory が読む）、`intrinsic.use_done_mask` は facade の設定（`DefaultDQNAgentConfig` が読む）である。

| 設定キー | 値 | 既定 | 読む条件 |
|---|---|---|---|
| `intrinsic.use_done_mask` | bool | true | `intrinsic.observation = next_state` |
| `intrinsic.rnd.warmup_mode` | `learn_predictor`｜`collect_errors` | `learn_predictor` | `statistics_mode = whiten` かつ `warmup_steps` が 1 以上 |
| `intrinsic.rnd.error_norm.average_mode` | `cumulative`｜`ema` | `cumulative` | `statistics_mode = whiten` |
| `intrinsic.rnd.error_norm.ema_steps` | 整数 ≥ 1（採点した観測の件数） | 1,000,000 | `whiten` かつ `average_mode = ema` |
| `intrinsic.rnd.error_norm.use_centering` | bool | true | `statistics_mode = whiten` |
| `intrinsic.rnd.warmup_steps` | 整数 ≥ 0（変えない） | 12,800（変えない）。Atari の `run.@rnd` は 200,000（D6） | 常に（今と同じ） |
| `intrinsic.rnd.multiplier_mode` | `clipped`｜`unclipped`（変えない） | `clipped`（変えない） | 常に（今と同じ） |

上の 5 つの新しい設定キーの「読む条件」は、下の判定の表の「読まれない条件」の裏である（読む条件に当たらないとき、読まれない条件のどれかに当たる）。

設定例（コロン記法）:

```text
DefaultDQNAgent.@baseline : intrinsic.use_done_mask ?= true
DefaultDQNAgent.@baseline : intrinsic.rnd.warmup_mode ?= learn_predictor          # learn_predictor｜collect_errors
DefaultDQNAgent.@baseline : intrinsic.rnd.error_norm.average_mode ?= cumulative   # cumulative｜ema
DefaultDQNAgent.@baseline : intrinsic.rnd.error_norm.use_centering ?= true
DefaultDQNAgent.@baseline : intrinsic.rnd.error_norm.ema_steps ?= 1,000,000
run.@rnd : DefaultDQNAgent.intrinsic.rnd.warmup_steps = 200,000                   # Atari.txt。入れる時機は D6
```

判定（構築時。`intrinsic.enabled = true` のときだけ行う）。判定は設定キーごとに行い、1 つの設定キーの値を、次の順に当てて最初に当たった扱いにする。(1) 値が不正なら ERROR。(2) 値は正しいが「読まれない条件」に当たるなら、値を使わない。解決した値が実装の既定と違えば通知を 1 回出し、同じなら何も出さない。(3) どちらにも当たらなければ有効で、値を使う。構築は、最初に見つけた ERROR で止まる。

| 設定キー | (1) 不正な値 | (2) 読まれない条件 | (2) の通知のレベル |
|---|---|---|---|
| `intrinsic.rnd.warmup_mode` | `learn_predictor`・`collect_errors` のどちらでもない | (a) `statistics_mode = variance_feedback`。(b) `statistics_mode = whiten` かつ `warmup_steps = 0` | (a) は WARN、(b) は INFO |
| `intrinsic.rnd.error_norm.average_mode` | `cumulative`・`ema` のどちらでもない | `statistics_mode = variance_feedback` | WARN |
| `intrinsic.rnd.error_norm.ema_steps` | 整数でないか、1 未満 | `statistics_mode = variance_feedback`、または `statistics_mode = whiten` かつ `error_norm.average_mode = cumulative` | WARN |
| `intrinsic.rnd.error_norm.use_centering` | bool でない | `statistics_mode = variance_feedback` | WARN |
| `intrinsic.use_done_mask` | bool でない | `intrinsic.observation = state` | WARN |

- (1) の検証は、(2) の条件（`statistics_mode` など）によらず行う。ERROR の文には、設定キー・値・期待する値（列挙か範囲）を入れる。
- `warmup_mode` の (a) と (b) は `statistics_mode` が違うので、同じ入力が両方に当たることは無い。
- (2) の WARN は、今の「モードが読まない値」の扱い（§2.3）と同じで、PRD117 の D1 の 3 の (b)（書いた値が、ほかの設定のために使われない）に当たる（D8・D16）。`warmup_mode` の (b) の INFO は、利用者が warm-up を止めたもとで、その中身の指定を飛ばした通知で、PRD117 の D1 の 1 の (a) に当たる（§5.1）。どちらも、解決した値が `?=`・個別指定・プロファイル・CLI のどれで与えられたかは問わない。`@baseline` の `?=` のままなら通知は出ない。PRD117 が先に実装されていれば、通知の形は PRD117 に従う。
- `intrinsic.enabled = false` のときは、今と同じく `intrinsic.*` のほかの設定キーを読まず、検証もしない。
- 既にある検証（`warmup_steps`・`multiplier_mode`・`multiplier_max` など）は変えない。

### 6.2 メトリクス（D10）

tag は増やさない。`56_agent_in/11_error_mean`・`12_error_std` は、次の batch の採点に使う誤差の統計を出す。誤差の統計が空の間は NaN になる（whiten・`learn_predictor` の warm-up 中。`collect_errors` では今どおり有限）。ほかの tag の意味は変えない。群 56 の tag は Q の学習が始まってから記録される（§2.3）ので、Q の学習の開始が warm-up より早い構成（`run.@rainbow_like` の 20,000 と warm-up 200,000 の組など）では、warm-up 中の NaN の点が記録に残る。§7.1 の区間は warm-up のあとなので、NaN の点を含まない。

### 6.3 checkpoint（D11）

RND の書式の識別子は、誤差の統計の中身の形で決まる。

| 設定 | 書式の識別子 | 誤差の統計の中身 |
|---|---|---|
| `statistics_mode = whiten`・`error_norm.average_mode = cumulative` | `RndNoveltyEstimator-3`（今と同じ） | 今の累積（変えない） |
| `statistics_mode = whiten`・`error_norm.average_mode = ema` | `RndNoveltyEstimator-4` | 直近重み（§5.2 の平均と母分散を復元できる値） |
| `statistics_mode = variance_feedback` | `RndNoveltyEstimator-3`（今と同じ） | 今の中身（変えない） |

識別子と誤差の統計のほかの中身と並び（`statistics_mode`、target、predictor、optimizer、観測の統計、target の出力の統計、採点した観測の件数）は、どの行でも今と同じである。

読込の判定（上から順に当て、最初に条件を満たした行で扱う）:

| # | 条件 | 扱い |
|---|---|---|
| 1 | 書式の識別子が、設定に対応する識別子（上の表）と違う | ERROR（checkpoint の識別子と、設定が求める識別子を示す） |
| 2 | `statistics_mode` が設定と違う | ERROR（checkpoint の値と設定の値を示す。今と同じ） |
| 3 | 上のどれにも当たらない | 復元する |

- 本 PRD の実装前に保存した書式 `-3` の checkpoint は、`cumulative` か `variance_feedback` の設定で、そのまま読める。再開後の動きは今の設定に従う。
- `warmup_mode`・`error_norm.use_centering`・`error_norm.ema_steps` と、facade の `intrinsic.use_done_mask` は保存せず、再開時は今の設定に従う。
- facade が保存する中身（credit の端数、lane 抽選の乱数状態、累計回数）は変えない。

### 6.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RndNoveltyEstimatorConfig` | 変更 | `warmup_mode`・`error_norm.average_mode`・`error_norm.ema_steps`・`error_norm.use_centering` を読んで検証し、読まない値を §6.1 の判定の表のレベル（WARN か INFO）で通知する | `intrinsic_rnd`（`src/intrinsic_rnd.hpp`・`.cpp`） |
| `RndNoveltyEstimator` | 変更 | whiten の warm-up の中身の切り替え（§5.1）、倍率の式（§5.3）、取り方ごとの書式の識別子での保存と読み込み（§6.3） | `intrinsic_rnd` |
| `RndErrorStatistics` | 新規（非公開） | whiten の誤差の統計を、累積と直近重み（§5.2）で持ち、更新・平均と標準偏差の読み出し・保存と読み込みを行う | `intrinsic_rnd` |
| `DefaultDQNAgentConfig` | 変更 | `intrinsic.use_done_mask` を読んで検証し、`state` で読まない値を WARN する | `default_dqn_agent`（`include/anet/default_dqn_agent.hpp`） |
| `DefaultDQNAgent` | 変更 | done の lane の内的報酬を 0 にするかの切り替え（§5.5） | `default_dqn_agent`（`src/default_dqn_agent.cpp`） |

### 6.5 移行対象

検索は 2026-11-11 に `git grep`（追跡しているファイルの全体）で行った。過去の記録（`docs/memo` の PRD116 と実装記録、`docs/experiments`、課題記録、Run の設定の写し）は書き換えない。

- コード: [intrinsic_rnd.hpp](../../core/anet-core/src/intrinsic_rnd.hpp)・[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp)（RND の設定と中身）、[default_dqn_agent.hpp](../../core/anet-core/include/anet/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../core/anet-core/src/default_dqn_agent.cpp)（`intrinsic.use_done_mask` の読み込みと、done の lane の扱い）。
- 設定: [agent.txt](../../apps/runner/config/agent.txt) の `@baseline` に、§6.1 の 5 つの設定キー（`intrinsic.use_done_mask` と、`intrinsic.rnd.` の `warmup_mode`・`error_norm.average_mode`・`error_norm.ema_steps`・`error_norm.use_centering`）を `?=` で足す。[Atari.txt](../../apps/runner/config/Atari.txt) の `run.@rnd` に `warmup_steps = 200,000` を足す（時機は D6。§7.2 の 8）。`run.@rnd_taiga` は変えない（§6.1 の判定の通知が 1 つも出ないこと）。[LunarLander.txt](../../apps/runner/config/LunarLander.txt)・[GridMaze.txt](../../apps/runner/config/GridMaze.txt) の `run.@rnd` は設定ファイルを直さず、既定の変更（warm-up の中身）がそのまま効く（動きは変わるが、確かめる Run は入れない。D6）。
- テスト（[intrinsic_test.cpp](../../core/anet-core/src/intrinsic_test.cpp)。タグ `[prd116]`）:
  - whiten で 0 以外の warm-up を使い、warm-up 中の誤差の統計か `Learn` の結果を断定するもの。文字列 `warmup_steps` で同ファイルを引き（25 行。うち 1 行は `learner.update_warmup_steps`）、補助 `RndTestConfig` の 4 件（59 行）を 0 に上書きしないテストと、372 行（4 件）・1168 行（8 件）のテストを対象にする。分かっているのは `RND factory scores observations and warms up statistics`、`RND warmup counts observations and reports optimizer progress`、`DQN intrinsic consumes warmup credit and holds the last loss`、`RND checkpoint preserves normalization warmup and training continuation` の 4 つである。今の中身を確かめる断定は `warmup_mode = collect_errors` を明示して残し、`learn_predictor` の断定を足す。883 行（`variance_feedback`、2 件）は変わらない。
  - 書式の識別子を断定するもの。文字列 `RndNoveltyEstimator-3` で `core/anet-core/src` を引くと 2 行（[intrinsic_rnd.cpp](../../core/anet-core/src/intrinsic_rnd.cpp) の 20 行と、intrinsic_test.cpp の 46 行の定数）で、テストは定数を使う `RND rejects mismatched checkpoint formats before reading payload` ほかである。`cumulative` と `variance_feedback` の識別子は変わらない（D11）ので、これらの断定は変わらない。`ema` の識別子と、識別子の取り違えの ERROR の断定を足す。
  - `variance_feedback` のテスト（`[rnd_reference_feedback]`）と、warm-up 0 の whiten のテストは、断定が変わらない。done の lane の扱いのテスト（`DQN intrinsic masks done but scores truncated lanes`、`DQN intrinsic state scoring ignores next observations and terminal masks`）は、既定（`use_done_mask = true`）のままなので断定が変わらない。
  - 新しいテストはタグ `[prd123]` を付ける。
- 出荷設定のテスト: `[shipped_config]` の RND の既定の一致の検査（intrinsic_test.cpp の `RND shipped Atari networks and Taiga settings match their contracts`）に、§6.1 の 5 つの設定キーを足す。
- 検証資材: `core/anet-core/testdata/prd123/README.md` を新設し、§7.2 の 1 の比較の手順（実装前の実行体の保存、前後の実行、比較）を書く。[PRD116 の検証資材](../../core/anet-core/testdata/prd116/README.md)の比較の道具を使えるなら使う。
- PRD116 の検証資材の smoke の判定: [run_validation.py](../../core/anet-core/testdata/prd116/run_validation.py) の smoke は、`32_learn_count` が 0 より大きい点を「warm-up 後」として選び、群 56 の全指標が有限であることを見る（92〜101 行。README の「smoke」の節）。新しい既定（`learn_predictor`）では warm-up 中も `learn_count` が増え、その間の `11_error_mean`・`12_error_std` は NaN になるので、この選び方が成り立たない。対象の点の選び方を、exp step が実効の `warmup_steps` を超えた点に改め、README の同じ節も直す。あわせて、Atari（KFM）の smoke だけ、`DefaultDQNAgent.intrinsic.rnd.warmup_steps = 12,800` を明示して渡す（同 72〜79 行の、env ごとの上書きに足す）。smoke は 100,000 exp step の Run で、`Atari.txt` に D6 の行（200,000）が入ると Run の全体が warm-up になり、対象の点が 0 件で止まる（同 103〜104 行）うえ、`clip_ratio` の確認も倍率が常に 1 で意味を失うためである。行が入る前は既定と同じ値なので、この上書きは実装のときに入れてよい。Montezuma の smoke は `run.@rnd` のあとに `run.@rnd_taiga`（warm-up 0）を重ねるので、D6 の行の影響を受けない。GridMaze の smoke は既定の 12,800 のままである。
- 設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md): §1.2 の warm-up の語の説明、§3.2（誤差の統計は累積、の記述に取り方の選択を足す）、§3.3（倍率の式に平均を引かない形を足し、done の lane の式の場合分けに `use_done_mask` を足す）、§3.4（処理の順の表と、その前文の mask の説明、`Learn` が warm-up 中に false を返す説明。237・246・254 行）、§3.5（warm-up の中身）、§3.6 の図（291 行の done の mask と、補助学習を warm-up 後に限る条件）、§4.2（設定の表。371 行の `intrinsic.observation` の説明、380 行の `warmup_steps` の説明、足す 5 つの設定キー）、§4.3（391 行。`run.@rnd` が選ぶものの列挙に、Atari の warm-up 200,000 を足す）、§4.4（`error_mean`・`error_std` の意味と NaN、413 行の `bonus_mean` の mask の説明、`learn_count` と warm-up の関係）、§4.5（448 行。直近重みの書式の識別子 `-4` と、識別子の取り違えの ERROR）。英訳 [212 en](../design/212_rnd_lifelong_novelty.en.md) の同じ章。括弧の行番号は 2026-11-11 時点の位置を示す例で、挙げた節の中の、当たる記述をすべて直す。
- 設計文書 [200](../design/200_dqn_agents.jp.md): §6.2 の RND の段落（400 行付近。done の lane の扱いを含む）、§7.2 の設定グループの「内的報酬」の行（560 行）と、その下の RND の既定と `run.@rnd` が選ぶものの段落（562 行。足す既定と、Atari の warm-up 200,000）、§8.2 の checkpoint の段落（600 行。直近重みの書式の識別子を足す）、§9.1 の群 56 の段落（633 行）。英訳 [200 en](../design/200_dqn_agents.en.md) の同じ箇所（402・562・564・602・635 行）。
- 設計文書のうち、Atari の `run.@rnd` の warm-up 200,000 を書く箇所（212 の §4.3、200 の 562 行と、それぞれの英訳）は、`Atari.txt` に行を入れるとき（§7.2 の 8）に同じ変更で直す。ほかの箇所は、実装と同じ変更で直す。
- wiki: [内的報酬のページ](../wiki/mechanisms/intrinsic_novelty.md)、[KFM のページ](../wiki/subjects/kung_fu_master.md)（出典のリンクの文字列）、地図の [321 Agent 共通基盤](../wiki/categories/321_agent_core.md)・[324 DQN 系列](../wiki/categories/324_dqn.md)（再グリルの書き戻しで直す）。
- PRD116: §8 の 1・3・5 が本 PRD を名指す箇所を、再グリル後の番号に直し、§8 の 6（終端遷移の内的報酬）に本 PRD へ移した旨を書く（再グリルで直す）。
- 今の書式 `-3` の checkpoint は、`cumulative` か `variance_feedback` の設定で、新しい実行体でもそのまま読める（D11）。過去の Run の checkpoint は書き換えない。

## 7. 成功の測り方と受入

### 7.1 効果（D14）

KFM（サブ機、実時間優先の構成＋ `run.@rnd`、10M exp step）で、次の 4 群を同じ実行体で回す。`s04` などは §2.1 と同じ seed である。

| 群 | `run.@rnd` に足す設定 | seed | 本数 |
|---|---|---|---|
| 新しい既定 | `intrinsic.rnd.warmup_steps = 200,000` | s04〜s07 | 4 |
| 直近重み | 上に加えて `intrinsic.rnd.error_norm.average_mode = ema`（`ema_steps` は既定の 1,000,000） | s04・s05 | 2 |
| 今の中身 | `intrinsic.rnd.warmup_mode = collect_errors`・`intrinsic.rnd.warmup_steps = 12,800` | s04・s05 | 2 |
| 平均を引かない形 | `intrinsic.rnd.warmup_steps = 200,000`・`intrinsic.rnd.error_norm.use_centering = false`・`intrinsic.beta = 0.01` | s04・s05 | 2 |

どの群も `warmup_steps` を明示して渡す。`Atari.txt` の `run.@rnd` に D6 の行が入る前でも後でも、同じ条件になる（今の中身の群は D5 の組になる）。β は、平均を引かない形のほかは既定の 0.1 である。1 本は約 1 時間 45 分で、10 本で約 17.5 時間かかる。

**主の条件（ゴール 1・2）。** step の軸は exp step、区間の値は 1M ごとの区間の平均である。

1. 新しい既定の 4 本すべてで、3M〜10M の 7 区間の `56_agent_in/01_bonus_mean` の最小が、今の中身の 2 本の 3M〜10M の全区間（2 本 × 7 区間）の値の最大を超える。`03_bonus_share` も同じ条件を満たす。
2. 新しい既定の 4 本すべてで、8M〜10M の 2 区間の `12_error_std` が 0.01 以上 0.05 以下である（旧 RND は 10M で 0.20〜0.65。§2.1）。帯は、predictor が 12,800 件から学習した旧 RND の Run の逆算（200k 以降だけの σ_e が 0.026〜0.034。§2.1）から置いた見込みである。新しい既定では predictor が 0 件から学習するので、200k 時点の誤差は旧い Run と同じにはならない。
3. 今の中身の 2 本の、9M〜10M の区間の `12_error_std` が 0.1 以上である（膨らみが同じ実行体で再現していること。旧 RND の同じ seed は 10M で s04 が 0.372、s05 が 0.201）。
4. 新しい既定の 4 本すべてで、2M〜3M の区間の `12_error_std` を 9M〜10M の区間の値で割った比が 1.5 以下である（ゴール 2）。今の中身では、最初の誤差が二乗偏差の総量を占める間、標準偏差が件数の平方根に反比例して下がるので、この比は約 1.9（√(9.5 ÷ 2.5)）になる見込みである。1.5 は、その見込みと 1 の間に置いた閾値である。

**記録する値（合否を置かない）。**

- 新しい既定の 4 本の `01_bonus_mean`（3M〜10M の平均）を、今の中身の同じ seed の値で割った倍率（仮説の見込みは 10 倍以上）。今の中身の 2 本の、主の条件 4 と同じ比（見込みは約 1.9）。
- 各群の、2M〜4M と 8M〜10M の `01_bonus_mean` の比（warm-up 明けのあとに残る変化の大きさ。ゴール 2、§8 の U3）と、`05_bonus_pos_ratio`。
- 評価側の Greedy（5.25M の 100 ゲーム）と ε 0.01 評価（8M 以降の 4 回の平均）。基準（RND なし）・旧 RND（§2.1）と並べ、seed のぶれ（基準の Greedy 5.25M で 13,090〜35,640）を超える差だけを読む。ボスの命中と撃破（RAM）も見る。
- 全群の、評価側のエピソードの長さ（評価セッションの `mean.episode_steps` を記録する tag。既定の群なら `21_eval/06_policy_ep_steps`）。内的報酬が大きくなると、長く生きる向きの力（§2.7、§5.3 の帰結）が出うるので、今の中身の 2 本と並べる。

終端の mask を外した群（`intrinsic.use_done_mask = false`）は、この Run に入れない（D14）。

Montezuma は本 PRD の合否に入れない。メイン機のキューのあとに b001 の構成（β 0.01）へ新しい既定を足した Run は、PRD121 の基準の取り直し（D15）として扱う。

### 7.2 正しさの受入

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 今の挙動との同一性 | 着手前に保存した実行体と、実装後の実行体に、同じ設定を渡して比べる。構成は次の 3 つで、どれも seed 1 本・決定的モードである。(a) LunarLander の既定の連鎖に `run.@rnd` と `intrinsic.rnd.warmup_mode = collect_errors` を足した 100k exp step の Run（実装前の実行体は `warmup_mode` を読まない）。(b) 同じ連鎖に `run.@rnd` と、Atari の `run.@rnd_taiga` の 8 つの設定値を足した 100k exp step の Run。(c) `intrinsic.enabled = false` の 4 env の既定（PRD116 の検証資材の比較）。比べるのは全 scalar 点の checksum と `metrics.scalar.defs` で、除くのは時刻・資源の群だけである（実時間に依るため）。3 つとも一致する |
| 2 | 直近重みの式 | 既知の誤差の列で、μ_e・σ_e が §5.2 の式の手計算と、浮動小数点の丸めの範囲で一致する。同じ誤差を N = 1 で 4 回と N = 4 で 1 回入れたときの μ_e・σ_e が、同じ範囲で一致する |
| 3 | warm-up | whiten・`learn_predictor` で、warm-up 中は α = 1、`intrinsic.error_mean`・`intrinsic.error_std` が NaN、`Learn` が true を返して `intrinsic.predictor_loss` が有限である。warm-up 明けの最初の batch は α = 1 で、次の batch から、前の batch までの誤差の統計で α が作られる。`collect_errors` では今の断定（誤差の統計が有限、`Learn` が false）が通る。`warmup_steps = 0` では、2 つの `warmup_mode` で同じ入力の α と `Learn` の結果が一致する |
| 4 | 倍率の式 | `error_norm.use_centering = false` で、`clipped` の α が min(1 ＋ e ÷ σ_e, L)、`unclipped` の α が 1 ＋ e ÷ σ_e と一致する。true では今の断定が通る |
| 5 | 設定の判定 | §6.1 の判定の表の 5 つの設定キーのそれぞれで、(1) の不正な値が ERROR になり、文に設定キーと値が入る。(2) の読まれない条件のそれぞれ（`warmup_mode` は (a)・(b) の両方）で、実装の既定と違う値を書くと、表のレベルの通知が構築時に 1 回だけ出て、値が使われない。`@baseline` の `?=` の値のままなら通知は出ない |
| 6 | checkpoint | `ema` の保存と再開で、続きの α と誤差の統計が、保存しなかった場合と一致する。`cumulative` も同じ。保存した識別子が §6.3 の表のとおりになる（`cumulative` と `variance_feedback` は `RndNoveltyEstimator-3`、`ema` は `RndNoveltyEstimator-4`）。§6.3 の読込の判定の行 1・2 が ERROR になり、文に checkpoint の値と設定が求める値が入る。着手前に保存した実行体で保存した whiten の checkpoint（§7.2 の 1 の (a) の構成）を、実装後の実行体が同じ設定で ERROR なく読み込む |
| 7 | 出荷設定 | `agent.txt` の `@baseline` の §6.1 の 5 つの設定キーの値が、実装の既定と一致する（`[shipped_config]`） |
| 8 | Atari の warm-up の行 | メイン機の 12_batch_run.bat の Montezuma のキューの stage 20 が終わったあとに、`Atari.txt` の `run.@rnd` へ `warmup_steps = 200,000` を足し、`run.@rnd` を解決した値が 200,000 になることを `[shipped_config]` のテストで確かめる。この場面が来ないまま実装を終えるときは、行とテストを入れずに終え、本 PRD の課題に残りの作業として書く（課題の判定は「残件有り」）。この項のほかの受入は、行を入れる前に確かめてよい |
| 9 | 終端の遷移の内的報酬 | `intrinsic.observation = next_state`・`intrinsic.use_done_mask = false` で、done の lane の学習報酬が、スケール済みの外的報酬 ＋ β × (α − 1) になる（偽の Learner で受け取って比べる）。true では、done の lane の学習報酬がスケール済みの外的報酬と一致する（今の断定）。`truncated` だけの lane には、どちらでも内的報酬が付く。`intrinsic.observation = state` で false を書くと WARN が 1 回出て、done の lane に内的報酬が付く（今と同じ動き） |

## 8. 決めること（未決表）

どれも Run の結果が要るので、確かめ方と、結果ごとの分かれ方を書く。

| # | 事項 | 確かめ方 | 分かれ方 |
|---|---|---|---|
| U1 | β の既定（`agent.txt` の `intrinsic.beta`） | §7.1 の新しい既定の 4 本の、2M 以降の各区間の `03_bonus_share` と、評価側（Greedy 5.25M と ε 0.01 評価） | 0.1 のまま回す。`03_bonus_share` が 0.5 を超える区間があるか、評価側が今の中身の 2 本の範囲を 4 本とも下回れば、β 0.03 の群を足して決め直す。どちらでもなければ 0.1 のままにする |
| U2 | 観測の正規化と、2 ステージの warm-up | §7.1 の主の条件の 2 と 4（新しい既定の 4 本の、8M〜10M の `12_error_std` と、2M〜3M と 9M〜10M の比） | 0.05 を超えて残る Run か、比が 1.5 を超える Run があれば、warm-up の長さ、観測の正規化の式（除数の ε、乱数行動の前段）、2 ステージの warm-up を開き直す。どちらも満たせば変えない |
| U3 | 直近重みを既定にするか | §7.1 の直近重みの 2 本と、新しい既定の同じ seed の 2 本を比べる。見るのは、2M〜4M と 8M〜10M の `01_bonus_mean` の比と、評価側 | 既定を変えるかは、結果を見て別に決める。変えるときは、LunarLander・GridMaze の時定数（Run が短い）も一緒に決める |

## 9. PRD116・PRD121・924 との関係

- PRD116 の §8 の 1（内的報酬の式。Burda らの e ÷ σ は比較の候補）・3（誤差統計の窓）・5（warm-up の既定値）を本 PRD へ移し、D7・D2・D3・D4・D6 で決めた。本 PRD の平均を引かない形は誤差の標準偏差で割る形で、Burda らの割引収益の標準偏差で割る形は移していない（PRD121・924）。PRD116 の D5 が whiten を「移動平均・標準偏差」と書いていたのは実物（累積）と違うので、2026-11-11 に PRD116 側で直した。PRD116 の §8 の 6（終端遷移の内的報酬）も本 PRD へ移し、D16 で決めた。PRD116 の D2・D4〜D6・D12 と §5.2〜§5.4・§5.6 のうち、warm-up の中身・誤差の統計・done の lane の扱いの選択・書式の識別子は、本 PRD の決定が上書きする。
- PRD121（外的と内的の価値の分解）は β の大きさの議論（スケール説）を持つ。§2.2 の膨らみは β と同じ働きをするので、PRD121 の実験は条件をそろえて比べる（D15）。
- 924（NGU 型）は、エピソード内の報酬に一生の新しさの倍率を掛ける。倍率は本 PRD の誤差の統計を使うので、warm-up と統計の取り方の変更はそのまま効く。
- PRD117（WARN の線引き。未実装）は、モードが読まない値の通知を扱う。§6.1 の判定の表の (2) の WARN は PRD117 の D1 の 3 の (b) と、`intrinsic.rnd.warmup_mode` の (b) の INFO は同じ D1 の 1 の (a) と、同じ線引きである。

## 10. 参照

- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.4（観測の正規化と内的報酬の正規化）。
- 公開の PyTorch 実装 [jcwleo/random-network-distillation-pytorch](https://github.com/jcwleo/random-network-distillation-pytorch)（`train.py`・`agents.py`・`utils.py`・`config.conf`）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。
- Badia ら、[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)、ICLR 2020。式 (1)。
- [Montezuma の RND（Taiga ら寄せ）の記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)。
