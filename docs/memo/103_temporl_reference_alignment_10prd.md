# TempoRL 第二弾（著者の実装との差の洗い出しと是正）PRD

> 2026-10-27 起票（暫定。質疑を省き、既定を置いて、決めることを §9 に残した）。行番号は起票時点のもので、実装時は近傍のシンボル名で探し直す。
> 起点: [PRD098](098_temporl_skip_policy_10prd.md)（TempoRL 対応）の段 3 の qbert で、合算（`feature_gradient = merge`）の Run の勾配クリップが 7.5〜20M で毎回発火していた件と、2026-10-27 の会話で「skip と Q の loss の扱いが著者の実装と違うのでは」と挙がったこと。著者の公開コード（[automl/TempoRL](https://github.com/automl/TempoRL)）と anet の実装を、経験と replay・学習と最適化・Network と行動選択の 3 観点で突き合わせ、差を T1〜T15 に整理した（§2.2）。本書はその全部を載せ、直すもの・腕で試すもの・文書に記すだけのもの・作らないものに分ける。
> 位置づけ: TempoRL の第二弾。第一弾は [PRD098](098_temporl_skip_policy_10prd.md)、第三弾は [999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)（長い j の不確かさ・閉ループの割り込み・全 k の一括学習）とする。
> 関連: [行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[評価と比較の条件（wiki）](../wiki/mechanisms/evaluation.md)、[TempoRL 論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)、[BTR と qbert の初期値の campaign](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)。
> 改訂: 2026-10-27（夕方）。§2.3 に、終わった合算の 2 本（サンプル効率優先 50M、実時間優先 100M）の勾配ノルム以外のメトリクス（眠った・死んだユニットの率、特徴量のノルム、Munchausen の log π の clip 率、TD）と、実時間優先 100M のクリップ率を足した。§7 の段 1 に並べる値を足し、§9 に 11・12 を足した。
> 改訂: 2026-10-28（グリル。`/anet-create-prd`、Claude Code）。goal anchor と scope screen に合意し、契約 6 件と細部 14 件を決めた。評価の条件は `[eval]` を変えず、行動もスキップ長も greedy の評価 session `[greedy]` を各 env に置く（Atari は `[greedy_dist]` を改名。D7）。その session に判断率・平均長・長さ別の割合と報酬の tag を置き、学習側の長さ別の割合の添字を 2 桁にする（D8）。T1 は `feature_gradient` の 3 値目 `split` にして、`optimizer_mode` キーと組み合わせのエラーは作らない（D2）。skip の方策の既定は PRD098 D9 のまま（D3）。`loss_weight` は据え置き（D12）。眠ったユニットと Munchausen の clip 率は観察だけ（D13）。照合で、`[eval_panel]` のスキップ長は今も Greedy であること、Munchausen の clip 率の上乗せは遮断でも同じであることを直した。§2.3 に段 1 の遮断の 10M までの値を足した。起草後の別モデル（Opus）の検討で、評価側の skip 指標は Actor の session 内の累積を `@session_end` で読む形に改め（D8。評価 runner の step 値では学習の進みと並ばない）、`21`・`22` の定義を `@skip` に集め、GridMaze のスキップ長の ε-greedy は 6×10 で確かめてから既定にすることにし（D3。同日に段 1' で確かめた）、本書の検証資材（D14）、移行一覧の漏れ、T10 の腕の spatial の葉、weight decay の文言、D12 の理由を直した。レビュー（`an-review`、3 往復）で、段 3 の窓の定義、GridMaze の既定を変える条件の統一（§7 の段 1' に 1 つ。届かない seed の扱いを含む）、突破率の語の書き分け、prd084 の README の移行、tag の綴り（`length`、`skip_primary`）、新しい ADR を起こさない理由、§3 の成功の測り方の段ごとの形、§6 の 5 の範囲の書き方を直した。同日に段 1'（GridMaze 6×10、3 seed）を流して条件を満たし、D3 を「`run.@skip` の既定を ε-greedy にする」に確定した（結果は実験記録 [2026-10-28_temporl-skip-policy-6x10](../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md)）。別ツールのレビュー 5 件で、判断回数の換算と `kmax` の重みを近似と明記し、§2.3 の倍率（2.2〜2.8 倍、1.5〜1.8 倍）、段 1' の基準が届かないときの扱い、受入 4 の記録先を直した。
> 改訂: 2026-10-28（段 2 の結果）。§7 の段 2 に LunarLander の 12 本の結果を記し（3 腕とも基準以上、上回る腕は無し）、§9 の 1 の LunarLander を「既定は据え置き」で閉じた。段 3 の腕に分割・`kmax` を足す条件を満たした。§5.7 の「振る舞いが変わるもの」に、`[greedy]` session を足すと同じ seed でも学習前から軌跡が変わることを足した。
> 改訂: 2026-10-31。§7 の段 1（遮断・重み 0.3、合算の seed 2）と段 3（分割）の結果を記録した（[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）。閾値 30 は実行中。
> 改訂: 2026-10-31（2 回目）。§7 の段 1 に閾値 30 の結果（3 腕とも無効の帯に戻らない）と、Atari の `run.@skip` の `grad_clip_tau` を 30 にしたことを記録した。qbert の調査を区切り、段 3 の分割・`kmax` と段 4 を見送った（§7・§8。再開条件つき。[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）。
> 改訂: 2026-10-31（3 回目）。§7 の受入 4（性能）を測り、split／merge の比 1.117 を記録した（§9 の 4 にも転記）。

## 1. 目的と問題

PRD098 は論文と論文解説から TempoRL を組み、著者のコードとは突き合わせていない。2026-10-27 に突き合わせたところ、合算のときの共有部の更新の仕方（T1）など、論文に書かれていない実装上の差がいくつか見つかった。著者のコードの側にも、特徴量の環境の版で skip の学習が壊れている所がある（T14）。

問題は 3 つある。

1. **合算の比べ方の土台が違う。** 著者の Atari の実装は、skip の loss と Q の loss をそれぞれの Adam で更新する。anet は 2 つの loss を足して 1 つの Adam で更新するので、共有部を動かす力の配分が勾配の大きさで決まる（T1）。qbert では合算で primary の勾配ノルムが無効の約 2 倍になり、クリップが毎回発火した（§2.3）。サンプル効率優先の qbert の合算（1 seed）は、10M までは無効を上回ったが、中盤で第 8 面の突破が遅れた（§2.3）。
2. **判定の指標が欠けている。** PRD098 はゴールの指標を評価 slot の判断率に置いたが、評価 slot の判断率を記録していない（T13）。評価 slot ではスキップ長にも探索や悲観の規則が掛かり、学習したスキップ価値そのものを測る条件が無い（T12）。PRD098 D9 の「評価 slot では skip も greedy」は、行動の方策が `Greedy` 型の slot でしか成り立たない。
3. **論文の結果との比べ方に誤りの余地がある。** 著者のコードの特徴量の環境の版は、skip の遷移を全部 j=1 として記録する（T14）。PRD098 は LunarLander の J や段 2 の見方を論文の LunarLander の結果に照らしているが、その結果がこの版で出たかは分からない。

## 2. 現状（2026-10-27 に確認したこと）

### 2.1 調べ方

- 著者のコード: [automl/TempoRL](https://github.com/automl/TempoRL) の master（commit `b8f4b0648489dbcc4895374df56a0d051379f2a8`、2022-03-02。この commit は LICENSE だけで、コードの最後の変更は 2021-06）。Atari は [run_atari_experiments.py](https://github.com/automl/TempoRL/blob/master/run_atari_experiments.py)（論文の Atari の結果は `--agent tdqn`＝重み共有の TDQN）、特徴量の環境（LunarLander・MountainCar など）は [run_featurized_experiments.py](https://github.com/automl/TempoRL/blob/master/run_featurized_experiments.py)。以下、行番号は `A:` が前者、`F:` が後者。
- anet のコード: 2026-10-27 の作業ツリー。
- 3 観点（経験と replay、学習と最適化、Network と行動選択）をサブエージェントで並行に調べ、主な主張はコードで裏を取った。
- 対象外（差として挙げない）: anet がベースの Agent として持つアルゴリズムの差（IQN・分布型、PER、n-step、Munchausen、NoisyNet、Dueling、replay ratio）、ベースのハイパーパラメータ、env のラッパー（sticky actions、frame skip）。

### 2.2 差の一覧

#### 学習と最適化

**T1. 合算のときの共有部の更新。**
- 著者: 重み共有では `_skip_q = _q` に Adam を 2 つ作る（A:811-812、836-837）。更新ごとに skip の loss で zero_grad → backward → clamp → step（A:990-998）、続けて更新後のパラメータで Q の loss を forward して同じ手順で step する（A:1013-1021）。他方の Head は勾配が 0 なので、2 回動くのは共有の畳み込みだけ。2 つの Adam のハイパーパラメータは同じ値（lr 1e-4、betas (0.9, 0.999)、eps 1e-8、weight decay 無し）をリテラルで書いたもので、skip 用の引数は無い。
- anet: primary の 2 組と skip Network の 2 組の param group を 1 つの AdamW に入れ（[dqn_based_agent.cpp:3223-3236](../../core/anet-core/src/dqn_based_agent.cpp)）、総 loss = L_Q + w·L_skip を 1 回 backward する（同 3375、4780・4919・5105。PRD098 D6）。
- 差: 共有部の 1 回の更新は、著者が −η(m̂_s/√v̂_s + m̂_Q/√v̂_Q)、anet が −η·m̂/√v̂（g = g_Q + w·g_s）。(i) 著者では skip と Q がそれぞれ約 η ずつ共有部を動かし、勾配の大きさに依らない。anet では勾配の大きい側が向きを決め、小さい側は √v で薄まる。(ii) 共有部の歩幅は、著者では向きが揃えば最大で約 2η。(iii) 著者の形では w は Adam の尺度不変性でほぼ効かない。
- PRD098 D6 は「optimizer を Network ごとに分ける形」と比べて同じ更新になるとした。遮断では 2 つの loss が触るパラメータが重ならないので loss ごとに分ける形とも一致するが、合算では一致しない。loss ごとに分ける形は検討していなかった。
- 更新の順（skip が先で、Q は更新後のパラメータで勾配を取る）による差は η の 1 歩分である。

**T2. 勾配のクリップ。**
- 著者: 要素ごとの clamp ±40 を loss ごとに掛ける（A:826、992-997、1015-1020）。Huber の平均の勾配では実質的に効かない見込み（推定）。特徴量の環境の版にはクリップが無い。
- anet: global norm を primary と skip Network で別々に取り、閾値は共通（[dqn_based_agent.cpp:3377-3393](../../core/anet-core/src/dqn_based_agent.cpp)。サンプル効率優先の Atari は 10）。
- 差: 合算では、Body に流れた skip の loss の勾配が primary 側のノルムに入る（§2.3 の実測）。

**T3. 共有の深さ。**
- 著者: Atari の重み共有は畳み込み 3 層だけを共有し、skip 側は専用の `Linear(7*7*64+10, 512)` → ReLU → `Linear(512, J)` を持つ（A:147-150、162-166）。特徴量の環境の重み共有は fc1・fc2（幅 50）を共有し、skip 側は `Linear(50+10, J)` の 1 層（F:204-221）。
- anet: skip Network は primary の公開出力 `skip_context` を読み、その後ろは `SkipFC1(128) > ReLU > SkipFC2(128) > ReLU` と射影（[agent.txt](../../apps/runner/config/agent.txt) の `@baseline`）。Atari の `skip_context` は `iqn_fusion`（[Atari.txt:620](../../apps/runner/config/Atari.txt)）。
  - サンプル効率優先（`run.@btrnet`）: `iqn_fusion` は（IMPALA > Pool6 > Flatten、2304 次元）と τ 埋め込みの積で、FC512 は Dueling の Head 側にある。共有は畳み込みと τ 埋め込みまでで、著者の形に近い（違いは τ 埋め込みの共有と、skip 側の幅）。
  - 実時間優先: `iqn_fusion` は（IMPALA > Linear512 > LN > ReLU）と τ 埋め込みの積で、FC512 と LN まで共有する。著者より深い。
- PRD098: D3・D4 で `skip_context` に何を出すかは設定に任せた。深さの比較はしていない。

**T4. 非共有の構成と遮断の違い。**
- 著者: Atari の `t-dqn` と特徴量の環境の `tqn` は、skip が自前の Body を持ち、skip の loss だけで学ぶ（A:813-817、F:719・950）。
- anet: 遮断（`stop`）は primary の出力を切り離して skip Network に渡す（[dqn_based_agent.cpp:2265-2274](../../core/anet-core/src/dqn_based_agent.cpp) の `MakeSkipInput`）。Q だけが学んだ特徴の上に skip の層を載せる形で、非共有とは別物である。非共有は「skip.net が obs の鍵を bind する」で表せる（PRD098 §5.3）が、段 1〜3 の腕に入れていない。

**T5. 一致している点。** skip の目標値（R + (1−d)·γ^L·Q_target(s', argmax Q_online(s'))、終点は行動価値。A:980-982 ／ anet は Q と同じ目標値の関数を窓に当てる。PRD098 §5.4）、skip と Q の更新回数の比（1:1）、学習開始の待ち、skip の target を学習に使わないこと（PRD098 D3）。

#### 経験と replay

**T6. sub-skip の重み。**
- 作る集合は同じ: 著者の Atari の版は、区間の各 step で、区間内の各起点からその step までの sub-skip を足す（A:962-973。区間全体で j(j+1)/2 本、判断の境界はまたがない）。anet は起点を任意の slot から引き、k を 1..k_max から一様に引く（[replay_buffer_impl.cpp:29-82・1379-1408](../../core/anet-core/src/replay_buffer_impl.cpp)、PRD098 D13）。
- 重みが違う: 著者は sub-skip を 1 本ずつ別の replay に入れて一様に引く（A:358-368）。長さ L の区間の重みは L(L+1)/2 に比例し、起点の分布は「区間の残りの長さ」に比例する（判断の時点の状態が最も多く引かれる）。anet は起点が区間内で一様、k が一様なので、区間の重みは L に比例し、1 本の sub-skip の確率は 1/(区間の残りの長さ) に比例する。
- 試算: 常に L=10 なら、k=1 は著者 18%・anet 29%、k=10 は 1.8%・1.0%。L が 1〜10 に一様なら、k=10 は 0.45%・0.18%。長い k のデータは著者の 1/1.8〜1/2.5。
- PRD098: D13 で「短い k が論文より多く入る」と意図し、§8 で「k_max に比例する重み」を先送りした。区間どうしの重みと起点の分布にも効くことは書いていない。

**T7. skip のデータの新しさ。** 著者の skip 用の replay（Atari は 5e4、特徴量の環境は 1e6。A:839-840、F:962-963）は、1 step あたり平均（区間長+1）/2 本ずつ埋まり、Q 用より速く入れ替わる（平均の区間長 4.4 なら Q の約 1/2.7 の範囲）。anet は slot を共有する（PRD098 D11）。目標値はサンプルのたびに今の Q で作るので、古くなるのは起点・行動・k_max の分布だけ。影響の大きさは分からない。

**T8. バッチの引き方。** 著者は skip と Q で別々に一様に引く（A:976-1024）。anet は同じ起点の B 行から両方の目標値を作る（PRD098 D14。独立に引くと forward と抽出が 2 倍になるため）。

**一致している点（経験と replay）:** 区間の実行（j と実際の長さ、途中の終端での打ち切り）、区間内の 1 step を全部 Q の replay に入れること、使えるまでの遅れ（PRD098 §6 の 11 で意図）。truncated は、anet が本当の最終観測から bootstrap する（[replay_buffer_impl.cpp:1580-1593](../../core/anet-core/src/replay_buffer_impl.cpp)）。

#### Network と行動選択

**T9. 行動の入れ方。** 著者の 3 構成は、行動の添字を float のスカラーのまま `Linear(1,10)` + ReLU に通して入れる（A:104・148・163、F:184-186）。特徴量の環境の文脈化と重み共有は出力層の直前で連結するので、状態と行動の交差を表せない（F:194-196・218-220）。anet の分離型（`action_out`）は全行動 × J を出して選んだ行動の行を取り、包含型（`action_in`）は one-hot を `skip_context` の末尾に連結する（PRD098 D5、§6 の 22）。

**T10. スキップ長を選ぶ規則。** 著者は行動と skip に同じ ε（乱数は別）を使い、greedy は平均の argmax（A:844-862、925-932）。anet は PRD098 D9 で行動の方策と同じ規則を使うので、UQE の env（LunarLander・GridMaze・DropMerge）ではスキップ長を楽観的な分位の risk score で選ぶ。GridMaze 5×5 の既定 seed でゴールに届かなかった原因はこれだった（PRD098 §9 の 2、[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §7 の段 0）。

**T11. IQN の τ の揺らぎ。** 学習の Actor は τ を毎回 8 本引き、その平均でスキップ長を選ぶ（[dqn_based_agent.cpp:2503-2513](../../core/anet-core/src/dqn_based_agent.cpp)）。greedy の lane でもスキップ長が揺らぐ。著者の Q はスカラーで、選択は決定的。

**T12. 評価のときのスキップ長。** 著者は評価で行動もスキップ長も ε=0（A:864-897）。anet は評価 slot の方策がスキップ長にも掛かり、Atari の `eval` は ε 0.01（[Atari.txt:244-247](../../apps/runner/config/Atari.txt)）、LunarLander の `eval` は UQE の τ 0.25（[LunarLander.txt:125-135](../../apps/runner/config/LunarLander.txt)）でスキップ長を選ぶ。PRD098 D9 の「評価 slot では skip も greedy になる」は、行動の方策が `Greedy` 型の slot でだけ成り立つ（行動の方策が `Greedy` 型なら、skip の葉が無くても skip は同じ Greedy になる。[default_dqn_agent.cpp:575](../../core/anet-core/src/default_dqn_agent.cpp)）。`[eval_panel]` はこれに当たり、今もスキップ長は Greedy である。Atari の `[greedy]`（ε-greedy の ε 0）は、行動と同じ ε 0 の ε-greedy でスキップ長を選ぶ。

**T13. 評価の判断率。** 著者は評価のエピソードごとに判断の回数を記録する（A:872-895）。anet の skip の指標は学習側（`@train`）だけで（[metrics_scalar.txt:382-390](../../apps/runner/config/metrics_scalar.txt)）、PRD098 §3・§7 がゴールの指標にした評価 slot の `01_decision_rate` が取れていない（PRD098 §9 の 3）。評価 slot の action_info を `@train` の event で scalar にしても、その値は評価 runner 自身の step 軸に載り、学習側の exp step とは換算できない（[140 §6.x](../design/140_observability.jp.md)）。

**一致している点（Network と行動選択）:** 判断の順（行動を選んでから、その行動を条件にスキップ長を選ぶ）、区間の途中での再判断が無いこと（割り込みは第三弾の (I)）。

#### 著者のコードと論文の不具合・食い違い

**T14. 特徴量の環境の版の skip の replay。** TDQN（F:1079-1087）と TQN（F:843-850）で、報酬の和が `skip_rewards[skip:]`（`skip` は選んだ番号）になり、`skip_id += 1` が起点のループの外にある。作る遷移は全部「ラベル 0（j=1）、長さ 1、次の観測は今の step、報酬は区間の最後の step 以外 0」になり、skip の出力の j≥2 は学習されない（初期値のまま）。Atari の版（A:964-973）は正しい。論文の特徴量の環境の結果がこの版で出たかは分からない（照合した commit は §2.1。コードの最後の変更は 2021-06 で、論文のあとの整理で壊れた可能性もある）。もしこの版なら、j の選択は学習された j=1 の値と初期値の比べ合いになり、論文の LunarLander が「ほぼ毎 step 判断する方策」に落ち着いたこととは整合する（推測）。PRD098 は §6 の 2（LunarLander の J=4 を論文の最良に合わせた）と §7 の段 2（論文の LunarLander のような「ほぼ毎 step 判断する方策」にはならなかった）で、この論文の結果を参照している。

**T15. 論文とコードの食い違い。** 付録 H は skip 側を「512 の隠れ層 2 層」と書くが、Atari のコードは隠れ層 1 層と出力層。特徴量の環境の ε は 0.2（F:1299）で、論文の本文は 0.1。

### 2.3 実測

いずれも qbert の Run。勾配ノルムは `37_agent_qtd/21_grad_norm`（primary のパラメータの、クリップ前の勾配ノルム）の窓ごとの中央値、クリップ率は `37_agent_qtd/24_grad_clip_ratio`（ノルムが閾値を超えた更新の割合）の窓平均。

サンプル効率優先（BTR に揃えた構成、閾値 10、seed 1）の、合算（`run_20261027-004945_a5_qb_aln_t98mrg_s1`）と無効（`run_20261025-231207_a5_qb_aln_btreps_s1`）:

| 窓 | 合算のノルム | 合算のクリップ率 | 無効のノルム | 無効のクリップ率 |
|---|---:|---:|---:|---:|
| 2.5〜5M | 17.4 | 90% | 6.4 | 9% |
| 7.5〜20M | 27〜30 | 100% | 12.8〜13.2 | 65〜78% |
| 25〜30M | 20〜22 | 97〜98% | 11.3〜12.3 | 57〜60% |
| 35〜37.5M | 16.6 | 94% | 9.0 | 44% |
| 42.5〜45M | 12.9 | 79% | 7.2 | 32% |

skip Network 自体の勾配ノルム（`50_agent_skip/15_grad_norm`。learn step の軸なので、learn step × 64 で exp step に換算した 5M 窓）は 5M 以降 3.2〜5.2 で、閾値に届いていない。

実時間優先の段 3 のふるい（PRD098 §7。7.5〜10M、seed 1／seed 2）: 無効のノルム 7.2／6.5・クリップ率 11%／5%、分離×遮断 9.3／8.5・38%／24%、分離×合算 16.0／15.0・97%／98%（`run_20261026-*_a5_qb_rt_t98{off,sep,mrg}_s{1,2}`）。遮断でも無効より高い理由は確かめていない。

ゲームによる違い（実時間優先 100M、無効、seed 2、10〜20M の中央値とクリップ率）: qbert 8.5・24%、NameThisGame 3.8・0.2%、Phoenix 2.4・0%、BattleZone 0.63・0%、DoubleDunk 0.46・0%（`run_20260923-163113_a5_qbert_rr1_100m_seed2` ほか同時期の 4 本）。閾値 10 で発火するのは qbert だけで、合算の影響が clip を通して出るのは qbert に限られる見込み。共有部の発言権の配分（T1 の (i)）は clip と関係なくどのゲームでも起きる。

サンプル効率優先の qbert の成績（同じ 2 本、seed 1。合算は 50M まで終わった）:

| 指標 | 窓・時点 | 合算 | 無効 |
|---|---|---:|---:|
| 評価（ε 0.01、`52_eval2`）の平均 | 2.5〜5M | 6,078 | 3,768 |
| 同 | 5〜7.5M | 11,424 | 9,177 |
| 同 | 7.5〜10M | 14,066 | 13,914 |
| 学習側の第 8 面の突破率 | 32.5〜35M | 15% | 63% |
| 同 | 40〜42.5M | 34% | 82% |
| ε=0 の評価（`53_evalg`）の平均 | 40.6M | 28,828 | 43,621 |

突破は「29,000 点以上かつ `round_clear` ≥ 8」（学習側は Run のログのゲーム終了の行から 2.5M 窓で数えた）。ε 0.01 の評価は 40〜45M でも合算 26,708・無効 25,388 とほぼ同じで、第 8 面の差が見えない。どれも 1 seed で、BTR 自身の 2 seed でも突破の時期は大きくずれた（40〜42.5M の突破率は BTR r0 2.3%、BTR r1 は 37.5〜40M で 79.6%。[campaign](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) の探索ブロック 08）。合算の 34% はその間に入るので、合算が悪いとはまだ言えない。

合算の Run の、勾配ノルム以外のメトリクス（2026-10-27 の夕方に、終わった 2 本の `metrics.jsonl` から集計）。比べる無効は、実時間優先 100M では TempoRL 無しの 100M（`run_20260923-163113_a5_qbert_rr1_100m_seed2`。seed 2、9 月のビルド）、サンプル効率優先 50M では上の無効（`btreps`）。眠ったユニットの率（`34_agent_plasticity/01_dormant_ratio`）、死んだユニットの率（`02_dead_ratio`）、特徴量のノルム（`03_feature_norm`）は、`learner.final_feature_branch_key = main_feature` の出力で測る（[ADR 0031](../adr/0031-plasticity-metrics-out-of-band-partial-forward.md)）。`main_feature` は skip の文脈（`skip_context = iqn_fusion`、`main_feature * tau_embedding`）の入力なので、合算では skip の loss の勾配がここを通る。実時間優先の `main_feature` は畳み込みから `AtariLinear512 > AtariLN512 > ReLU` まで、サンプル効率優先は畳み込みから `AtariPool6 > Flatten` まで。値は窓平均で、learn step の軸の値は exp step へ換算した。

| 指標 | 実時間優先・合算（`run_20261027-122647_a5_qb_rt100_t98_s1`、seed 1）90〜100M | 実時間優先・無効 90〜100M | サンプル効率優先・合算 45〜50M | サンプル効率優先・無効 45〜50M |
|---|---:|---:|---:|---:|
| 眠ったユニットの率 | 3.6% | 0.95% | 14.4% | 10.4% |
| 死んだユニットの率 | 0.69% | 0.15% | 7.3% | 3.8% |
| 特徴量のノルム | 16.0 | 17.0 | 2.8 | 6.3 |
| クリップ率 | 25% | 6.5% | 68% | 26% |
| Munchausen の log π の clip 率（`36_agent_munchausen/03_clip_ratio`） | 4.6% | 2.3% | 2.9% | 0.14% |
| TD の平均（`37_agent_qtd/01_td_mean`） | 1.28 | 1.89 | 0.92 | 0.44 |

- 眠ったユニットの率の推移: 実時間優先の合算は 10〜20M の 0.9% から 100M まで増え続け、無効は 30M から 0.8〜1.0% で横ばいだった。サンプル効率優先は両方とも増えるが、合算は 5〜10M の 1.8%（無効 4.1%）から 20M ごろに無効を追い越して差が開いた。死んだユニットの率も同じ向き。サンプル効率優先の合算は 45.7M の ε=0 の評価のエピソードのうち第 8 面を突破したものが 68%（評価側の数え方。§7 の「突破率」は学習側）なので、50M までは成績を壊すほどではない。
- 実時間優先 100M のクリップ率: 合算は 10〜20M 100%、20〜30M 96%、40〜50M 73%、90〜100M 25%。無効は 10〜20M 24%、40〜50M 10%、90〜100M 6.5%。
- Munchausen の log π の clip 率は、合算に限らず TempoRL 有効で上がる（下の段 1 の表。遮断と合算が同じ値）。この指標は、サンプルした行の取った行動 a_t について、今の online の方策（Q を `entropy_tau`（0.03）で softmax したもの）の `entropy_tau`·log π(a_t|s_t) が下限 −1 を下回った割合で、おおよそ「取った行動が今の最善の行動より Q で 1 以上（clip 後の報酬 1 回分）損な行」の割合に当たる（[dqn_based_agent.cpp:108-119](../../core/anet-core/src/dqn_based_agent.cpp)）。TempoRL では判断で選んだ行動をスキップ長の分だけ繰り返し、途中の step も replay に入るので、途中の状態で 1 step の Q の最善が別の行動に変わっている行がこれに数えられる。ε が 0 になったあとの終盤（無効 0.14% 対 合算 2.9%）は、判断率が約 0.2 なので replay の約 8 割が途中の行で、そのうち「明らかに損」は 4% 弱に当たる。これは T1（勾配の混ぜ方）とは別の、TempoRL と Munchausen の組み合わせの性質で、著者は Munchausen を使わない（D13、§9 の 12）。
- TD の平均は、実時間優先では合算が低く、サンプル効率優先では高い。向きがそろわないので、合算の性質とは読まない。
- 同じビルドの 10M（PRD098 §7 の段 3 のふるいの s1・s2、7.5〜10M）では、Munchausen の clip 率は合算 9.9%・8.6%、遮断 9.5%・9.2%、無効 2.9%・4.6% だった。眠ったユニットの率は合算 0.74%・0.33%、無効 0.29%・0.74% で、まだ差が無い。実時間優先 100M の比べる相手は seed もビルドも違うので、100M の差がすべて合算によるとは言えない。
- 壊れている兆候は無かった。実時間優先 100M の合算は、Q の最大値の平均が 40M から 44 前後で横ばい（最大 76）、TD と loss は 10M を山に下がり続け、行動の入れ替わり率（`35_agent_churn/01_action_churn_ratio`）は 1.0% から 0.3% へ、srank の比は 0.84 から 0.89 へ動いた。81M の ε=0 の評価の崩れ（平均 14,313、111 回のうち 80 回が `round_clear` 1〜4 で終了）の前後 0.5M 刻みでも、行動の入れ替わり率・Q の変化量・skip の loss・眠ったユニットの率に動きは無く、ε 0.01 の評価も前後と同じ幅（76〜86M で 16,922〜26,198）だった。

段 1 の遮断（`run_20261027-175541_a5_qb_aln_t98stp_s1`、サンプル効率優先、seed 1）の 10M までの値（2026-10-27 の 19:49 に集計。遮断は 9.2M まで。無効・合算は上と同じ Run。値は窓平均。眠った・死んだユニットの率と特徴量のノルムは learn step の窓を exp step に換算した）:

| 指標 | 窓 | 無効 | 遮断 | 合算 |
|---|---|---:|---:|---:|
| primary の勾配ノルム | 2.5〜5M ／ 5〜7.5M ／ 7.5〜10M | 6.6 ／ 11.6 ／ 14.2 | 9.1 ／ 14.0 ／ 16.7 | 18.7 ／ 25.1 ／ 33.0 |
| クリップ率 | 同 | 9% ／ 57% ／ 78% | 32% ／ 73% ／ 88% | 90% ／ 100% ／ 100% |
| Munchausen の log π の clip 率 | 同 | 4.8% ／ 6.4% ／ 6.2% | 11.7% ／ 10.5% ／ 10.5% | 11.4% ／ 9.4% ／ 7.9% |
| 眠ったユニットの率 | 同 | 4.2% ／ 4.1% ／ 4.0% | 4.5% ／ 4.9% ／ 5.1% | 1.8% ／ 1.7% ／ 1.8% |
| 死んだユニットの率 | 同 | 0.79% ／ 0.78% ／ 0.73% | 1.09% ／ 1.14% ／ 1.31% | 0.32% ／ 0.35% ／ 0.37% |
| 特徴量のノルム | 同 | 12.6 ／ 13.3 ／ 12.7 | 17.8 ／ 15.7 ／ 15.3 | 18.8 ／ 10.0 ／ 8.1 |
| 評価（ε 0.01、`52_eval2`）の平均 | 0〜2.5M ／ 2.5〜5M ／ 5〜7.5M ／ 7.5〜9.2M | 824 ／ 3,770 ／ 9,180 ／ 13,600 | 1,330 ／ 4,660 ／ 9,500 ／ 13,200 | 1,320 ／ 6,080 ／ 11,400 ／ 13,700 |
| 学習側の判断率・平均スキップ長 | 0〜2.5M | — | 0.20・5.1 | 0.22・4.7 |

- 遮断と合算で Munchausen の clip 率が同じなので、上乗せは TempoRL 有効によるもので、合算（T1）によるものではない。
- 遮断の勾配ノルムも無効より 1.2〜1.4 倍高い（実時間優先のふるいと同じ。理由は確かめていない。候補は replay に入るデータの違い）。合算は無効の 2.2〜2.8 倍（窓ごとに 18.7/6.6、25.1/11.6、33.0/14.2）で、遮断との差が skip 由来の勾配に当たる。遮断のノルムを Q の勾配、合算のノルムを両方の和とし、向きが直交すると仮定すると、skip 由来の勾配は Q の勾配の約 1.5〜1.8 倍（窓ごとに 1.8・1.5・1.7）と見積もれる。
- 眠った・死んだユニットの率は、遮断は無効よりやや高く増え続け、合算は 10M までは無効より低い（上の 20M ごろの逆転は合算だけ）。特徴量のノルムは、合算だけ 5M から速く落ちる。
- 評価（ε 0.01）は 2.5〜7.5M で合算 > 遮断 > 無効（1 seed）。序盤の有利は、TempoRL 自体の分（遮断 > 無効）と合算の分（合算 > 遮断）に分かれる。
- 眠ったユニットが合算で増える機序は確かめていない。候補は 3 つで、どれも skip の loss の勾配が共有部に入ることを前提にする（遮断が無効の線に留まることがその裏づけになる）。(1) 更新量: 共有部に入る勾配の総量と非定常さ（bootstrap の目標値が 2 本）が増え、既知の可塑性低下（更新の積み重ねで ReLU のユニットが負側に落ちて戻らない）が速く進む。実時間優先で 100M まで増え続けることと、サンプル効率優先で序盤は合算のほうが低く 20M で追い越すこと（補助の目標でユニットが使われ、そのあと蓄積する）に合う。(2) 表現の干渉: skip の目的に要る特徴が Q に要る特徴を押しのける。サンプル効率優先で特徴量のノルムが速く落ちることに合うが、実時間優先は LN でノルムが固定されても増えるので、ノルムの縮みだけでは説明できない。(3) クリップとの結合: クリップ後の向きを大きい skip の勾配が決め、Q の学習信号が薄まる。サンプル効率優先の 7.5〜20M のクリップ率 100% に合うが、実時間優先の 90〜100M はクリップ率 25% でも増え続ける。段 1 の重み 0.3 は (1) と (3) を同時に弱め、閾値 30 は (3) だけを外す。段 3 の分割（D2）は Q 側の歩幅を戻して (3) を外す一方で共有部の総歩幅を増やす（√2〜2 倍）ので、(1) が主因なら分割で悪化し、(3) が主因なら改善する。向きで (1) と (3) を分けられる（D13、§9 の 11）。

## 3. ゴール・非ゴール・成功の測り方

ゴール:

- 著者の実装との差のうち、学習の結果に効きうるもの（T1・T6）を、設定で著者の形に切り替えられるようにする。既定は今の形のままで、今と同一の結果になる。
- 学習したスキップ価値そのものを測れる評価の条件を用意し、その評価の判断率を記録する（T12・T13）。行動もスキップ長も greedy の評価 session を各 env に置き、skip の指標の tag を env をまたいで並べられる形に揃える。
- 設定だけで作れる腕（T2・T3・T4・T10）を用意し、段ごとに比べる（§7）。
- 著者のコードと論文の不具合・食い違い（T14・T15）を、PRD098 と wiki に記して、論文の結果との比べ方を正す。

非ゴール（消える案と、それが本当の制約かどうか）:

- アルゴリズムの差（IQN・PER・n-step・Munchausen など）を著者に揃えること: anet のベースの Agent の選択で、TempoRL の比較の土台ではない。制約ではなく、ゴールに要らないことによる非ゴール。
- 著者の更新の順（skip を先に step し、更新後のパラメータで Q の勾配を取る）を真似ること: current の forward が 1 回増え、差は η の 1 歩分である。制約ではなく、効きが小さい見込みによる非ゴール（§8）。
- skip 専用の replay とバッチ（T7・T8）: PRD098 D11・D14 の理由（PER・prefetch・世代 key の 2 系統化、forward と抽出が 2 倍）がそのまま残る。クリーンブレークで移せるので制約ではなく、費用に見合う証拠が無いことによる非ゴール（§8 に再開条件）。
- 行動をスカラーの添字で入れる形（T9）: 行動に順序を仮定する著者の癖で、真似る価値が無い（§8）。skip Network の入力の形は設定とクリーンブレークで移せるので制約ではなく、anet の one-hot と分離型のほうが素直で、実測でも分離型が包含型と同等以上だったことによる非ゴール。
- `[eval]`・`[eval_target]` の評価の条件を変えること（スキップ長だけを Greedy にする形、行動も greedy にして著者の評価を `[eval]` に写す形）: 評価の方策は設定なので移せる（制約ではない）。既存の評価の時系列（PRD098 の段 1〜3、2026-10-26 の LunarLander の既定の判断）と、TempoRL 以外の比較の条件（Atari の BTR 準拠の ε 0.01、悲観の UQE）を保つことによる非ゴール（§8）。
- 第三弾の拡張（不確かさ・割り込み・全 k）: 別 PRD（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)）。範囲の分割で、制約ではない。

成功の測り方（判定の定義は §7）:

- 新しい設定が既定のとき、本書の検証資材（D14）の比較で metrics checksum が今と一致する（受入 1）。
- skip が有効な Run の `metrics.jsonl` に、`[greedy]` session の判断率・平均長・長さ別の割合（`50_agent_skip/21`〜`23`）と、その session の報酬（LunarLander・GridMaze・DropMerge は `21_eval/09`〜`12`、Atari は `53_evalg`）が学習側の exp step の座標で出て、`config/config_data.txt` に `[greedy]` session がある（受入 3）。
- 段ごとの判定で、各腕を基準の腕と比べた答えが 1 つに決まる（段 2 は §7 の「以上」「上回る」の定義で、段 1' は seed ごとの届く速さと終盤の安定さで、段 3 は突破率の窓と最初の 10M の平均点で、段 4 は 2 seed の seed ごとの比較で。§7）。T1（分割）と T6（k_max に比例する重み）は、それぞれ既定にするかを段 2・段 3 の結果で決める（§9 の 1）。

## 4. 確定事項

各決定に理由を書き、ほかの決定や事実に依存するときはそれを名指す。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | 差 T1〜T15 を 4 つに分ける。**直す（設定で切り替え）**: T1・T6。**判定の条件を揃える（greedy の評価 session と指標）**: T12・T13。**設定だけの腕**: T2・T3・T4・T10。**文書に記す**: T14・T15。**作らない**: T7・T8・T9・T11（§8）。T5 と一致している点は記録だけ | T1 と T6 は学習の結果に効く見込みがあり、設定の葉では作れない。T12・T13 は判定に要る。T2・T3・T4・T10 は既存の設定で作れる。T7・T8 は PRD098 の理由が残り、T9 は真似る価値が無い。T11 は影響が小さい見込み（§8） |
| D2 | T1 は、`learner.skip.feature_gradient` に 3 値目 `split`（分割）を足す。`split` は合算（勾配を共有部へ流す）のまま、Adam のモーメントを loss ごとに分ける更新の方式で、primary のパラメータは Q の loss の勾配と skip の loss の勾配でそれぞれのモーメントを持ち、2 つの Adam の更新を足した分だけ動く。skip Network のパラメータは skip の loss のモーメントだけを持つ。既定は変えない（agent.txt は `stop`、Atari と LunarLander は env の設定ファイルの葉で `merge`） | 著者の重み共有の形（T1）を、共有部の発言権が勾配の大きさに依らない性質ごと再現する。設計の空間は遮断・合算・分割の 3 点（分割は遮断とは意味を持たない組み合わせ）なので、1 キー 3 値が最小で、`optimizer_mode` のような別キーだと無効な組み合わせ（`per_loss`×`stop`）とその検査・テストが要る（§8）。値の名前は `stop`・`merge` に揃えた 1 語で、`merge`（1 つに合わせる）の対にした。既定を変えないのは、段 2・段 3 の結果が要るため（§9 の 1） |
| D3 | スキップ長を選ぶ方策の既定は PRD098 D9（行動の方策と同じ規則・同じ値）のまま。GridMaze は、6×10 の 3 seed（§7 の段 1'）で段 1' の条件（ε-greedy の 3 seed の届く速さの最大が UQE の 3 seed の最大以下で、終盤の安定さの最小が UQE の最小以上。seed ごとに比べ、平均は使わない。届かない seed は、腕なら条件不成立、基準なら 500k として扱う）を 2026-10-28 に満たしたので、`run.@skip` の既定でスキップ長を平均に対する ε-greedy（`actor.[train].skip.policy.policy_type = EpsilonGreedy`、ε 0.60→0.05）にする（結果は [2026-10-28_temporl-skip-policy-6x10](../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md)）。別のプロファイルは作らない。LunarLander は段 2 の T10 の腕（§7）の結果で決める | 第三弾の段 0 で、GridMaze 5×5 の既定 seed は UQE でスキップ長を選ぶとゴールに届かず、ε-greedy なら届いた（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §7 の段 0。腕は `run_20261026-041740_t98_gm_eps_s1`）。ただし 6×10 の 3 seed と 5×5 の seed 3 は UQE のままでも届き、6×10 は最後まで +0.93 を保った（PRD098 §7 の段 1）ので、ε-greedy の結果が無い迷路のために既定は動かさない方針（D9 と同じ）で段 1' を先に流した。GridMaze は 5×5 で届かない原因が分かっているので、6×10 で悪くないこと（段 1' の条件）で足り、§9 の 1 の LunarLander・Atari 向けの一般則（同じビルドの 2 seed 以上で最小の seed が基準の最大を上回る）は掛けない。段 1' では ε-greedy が UQE と同じ速さで届き（282k／311k／313k 対 269k／323k／359k）、6 本とも最後まで +0.93 を保った。ε-greedy の方策は 6×10 では j=1 が 93〜95%（判断率 0.88〜0.89）で、ほぼスキップしない形に落ち着く（UQE は j=2 を 15〜41% 使う）。既定を変えるので、受入 1 の GridMaze の skip 有効の基準は取り直す（D14）。第三弾の §9 の 9（skip の方策の既定を改めるか）はこの裁定で閉じる（2026-10-28） |
| D4 | `split` の 2 つの勾配は、同じパラメータの値から取る（著者の「skip を先に step してから Q の勾配を取る」順は真似ない） | 著者の順にすると current の forward が 1 回増え、差は η の 1 歩分（T1）。D2 に依存 |
| D5 | `split` のクリップは、loss ごと・Network ごとに掛ける。Q の loss の勾配（primary のパラメータ）、skip の loss の勾配の primary のパラメータの分、skip の loss の勾配の skip Network のパラメータの分の 3 つに、それぞれ global norm で閾値 `grad_clip_tau` を掛ける。2 つ目のクリップのノルムと発火率は `17`・`18` で記録する（§5.4） | loss ごとに分けた更新と揃える（著者もクリップを loss ごとに掛ける。T2）。合算の skip 由来の勾配が Q のクリップを発火させる結合（§2.3）も無くなる。Network ごとにも分けるのは、[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) の「global norm クリップは Network 別に取る」を保つためで、ADR とは矛盾しない。skip Network の分のノルムは今と同じ範囲になる（§5.4 の `15` の意味が変わらない）。発火率を出すのは、§2.3 の候補 (3)（クリップとの結合）を分けるのに率が要り、ノルムの平均からは率を出せないため。名前の `skip_primary` は「skip の loss の、primary のパラメータへの勾配」を表す（Head の分は 0 なので共有部の分と同じ値。`body` は NN の Body と weight norm の二群分割の語と紛れるので使わない）。D2 に依存 |
| D6 | T6 は、skip の loss をサンプルごとに「k_max ÷ バッチの k_max の平均」で重み付けする方式（`learner.skip.sample_weight = kmax`）を足す。既定は今の `uniform` | 起点を一様に引くとき、この重みで 1 本の sub-skip の重みの期待値が著者の「sub-skip ごとに等しい」とほぼ一致する（§2.2 の T6 の確率から。分母がそのバッチの平均なので、有限のバッチでは期待値が 1/B の程度でずれる近似。B=256）。forward は増えない。PER では起点の分布が優先度に従うので一致しない（重みは同じ式で掛ける）。第三弾の (K)（全 k の一括）より安く、同じ偏りに当たる |
| D7 | T12 は、skip を有効にする各 env に、行動もスキップ長も greedy の評価 session `[greedy]` を置く。actor は、Atari は既存の `[greedy]`（`[eval]` の ε 0）、LunarLander・GridMaze は `[greedy].$ = [eval] > @greedy`（online）、DropMerge は `[greedy].$ = [eval_target] > @greedy`（target）。Atari は既存の `[greedy_dist]`（`53_evalg`）を `[greedy]` に改名して揃え、LunarLander・GridMaze・DropMerge は `run.@skip` に session を足す。`[eval]`・`[eval_target]`・`[eval_panel]` は変えない。session の頻度と大きさは §6 の 2 | ゴール 2 の「学習したスキップ価値そのもの」を測れるのは、行動もスキップ長も greedy の条件（著者の評価と同じ。T12）で、行動の方策が `Greedy` 型なら skip の葉が無くても skip は Greedy になる（[default_dqn_agent.cpp:575](../../core/anet-core/src/default_dqn_agent.cpp)。コードは変えない）。`[eval]` を変えないので、既存の評価の時系列（PRD098 の段 1〜3、2026-10-26 の LunarLander の既定の判断）が切れず、段 1 の 4 本（基準の合算を含む）の `53_evalg` が同じ条件のまま段 3 の判定に使える（§7）。network は各 env の判定の slot に揃える（LunarLander・GridMaze の判定は `[eval]` の policy 報酬、DropMerge は `eval_target`。DropMerge の `eval_target` でスキップを測るために skip の target を持つ、という ADR 0062 の理由とも合う）。Atari の改名は、session 名を 4 env で揃えて J に依らない `21`・`22` の定義を 1 か所に置くため（D8）。評価 env と Actor の seed は session 名から作る（[trainer.cpp:1098-1114](../../core/anet-core/src/trainer.cpp)）ので、改名で Atari の greedy の評価の乱数の系列は変わる（条件は同じ。100 エピソードの平均なので学習の seed 差より小さい見込み）。`[eval_panel]` は行動が `Greedy` 型なので今もスキップ長が Greedy（T12）。`[eval]` のスキップ長だけを Greedy にする形は条件が混ざり時系列が切れ、行動も greedy の `[eval]` は TempoRL 以外の比較の土台を動かす（§3 の非ゴール、§8）。費用は低頻度の評価 1 本（§5.6） |
| D8 | T13 は、Actor が評価 session の中の判断の回数・step 数・スキップ長の和・長さ別の回数を累積し、`[greedy]` session の終わりに `$eval.[greedy] $actor <鍵> @session_end` で、判断率・スキップ長の平均・長さ別の割合の tag（`50_agent_skip/21_greedy_decision_rate`・`22_greedy_length_mean`・`23_greedy_length_rate.[01]`〜`[J]`）として記録する。累積は session の頭の `Sync()` で消す。LunarLander・GridMaze・DropMerge では報酬とエピソード長の tag（`21_eval/09`〜`12`）も足す。`21`・`22` の定義は `metrics_scalar.txt` の `@skip` に、`23` と報酬の tag は各 env の `run.@skip` に置く。学習側の `03_skip_length_rate.[1]`〜`[J]` は添字を 2 桁（`[01]`〜`[J]`）に改名する（§5.4） | 判断率がゴールの指標（PRD098 §3）。評価 slot の action_info を `@train` の event で scalar にすると評価 runner 自身の step 軸に載り、学習の進みや腕どうしと並べられず、1 エピソードあたりの判断の回数（判断率 × エピソード長）も座標が違って作れない（T13、[140 §6.x](../design/140_observability.jp.md)）。`@session_end` は学習側の座標に載り、`21_eval/09` と同じ座標になる。`$actor` は既存の取得元で、session の頭の `Sync()` も既存（[trainer.cpp:233-237](../../core/anet-core/src/trainer.cpp)）。長さ別の割合を評価側にも置くのは、`kmax`（D6）で長い j が使われるようになったかを、探索の混ざらない評価で直接読むため。添字を 2 桁にするのは、Metrics Viewer の並びで `[10]` が `[1]` の上に来るためで、J が 10 未満の env も揃える（env をまたいで同じ tag で並べる）。過去の `[1]` の Run とは別の系列になる。別名・互換読みは置かず、過去の Run の metrics と実験記録は書き換えない（[AGENTS.md](../../AGENTS.md) §クリーンブレーク方針）。tag は群名と `/` を含めて 40 文字以内で、綴りは群の既存の `length`（`02_skip_length_mean`・`03_skip_length_rate`）に揃える（[impl.md](../quality/impl.md) 名前の規約）。D7 に依存。PRD098 §9 の 3 を閉じる |
| D9 | T2・T3・T4・T10 は、設定だけで作る腕として段に入れ、既定は変えない。既定を変えるかは各段の結果で決める（§7） | 結果が出る前に既定を動かさない。腕の設定は §5.5 |
| D10 | T14・T15 は、PRD098（D6・D9 の注記、§6 の 2 と §7 の段 2 の注記、§9 の 3 の行き先）と wiki に記す。論文の特徴量の環境の結果と比べるときは、著者の Atari の版を照合の正本にする。論文解説には注記しない（§6 の 14） | 論文の特徴量の環境の結果が壊れた版で出た可能性を、読み手が知らないまま比べないため |
| D11 | 新しい設定がすべて既定（`feature_gradient` は今の値、`sample_weight = uniform`）のとき、Learner の forward と backward の回数・順序・RNG の消費・loss・優先度・checkpoint の形は今と同一。Actor の累積（D8）は乱数・loss・出力に触れない | PRD098 の段 0 と同じ等価性の受入を保つ |
| D12 | `loss_weight` は、どのモードでも skip の loss に掛ける重みのまま。`split` では Adam の尺度不変性で効きが eps の分だけになり、クリップ（D5）が掛かる間だけ skip 側の閾値を τ/w にしたのと同じ効きを持つことを本文に書く（§5.2） | キーの意味を 1 つ（skip の loss の重み）に保つ。`split` で効かないのは Adam の性質で、文書で足りる。skip 側の歩幅を変えたくなったら、PRD098 §8 で先送り中の skip 専用 lr を別キーで起こす。`split` で歩幅の倍率に読み替える形（同じキーがモードで意味を変える）と、`split` で 1 以外をエラーにする形（プロファイルの使い回しで止まる）は採らない（§8）。D2・D5 に依存 |
| D13 | 合算で眠った・死んだユニットが増える件（§2.3）と、TempoRL 有効で Munchausen の log π の clip 率が上がる件（§2.3）は、観察だけにする。段 1（遮断・重み 0.3・閾値 30）と段 3（分割）で値を並べ、判定には使わない。確かめる計測（判断の旗で分けた clip 率）と手立ては本書に入れない | 眠ったユニットの機序は 3 候補あり、段 1・段 3 の腕の向きで分けられる（§2.3）。Munchausen の上乗せは遮断でも同じで、著者との差（T1）ではなく TempoRL と Munchausen の組み合わせの性質（著者は Munchausen を使わない）。どちらも本書の関心（著者との差）の外で、計測には replay のサンプルの出力に判断の旗を運ぶ契約の変更が要る。工程末に 999 の起票候補として挙げたが、2026-10-28 に起票しないと決め、本書の未決表に残す（§9 の 11・12） |
| D14 | 本書の検証資材を `core/anet-core/testdata/prd103/` に置く。着手前に、本書の前の実行体と設定一式を snapshot し、評価を止める session の一覧は今の解決済みの設定から取り、新しい実行体で前の設定を動かす形を持つ | PRD098 の検証資材は、凍結した解決済みの設定から評価 session を列挙して `interval = 0` を渡す（[run_validation.py:28・72](../../core/anet-core/testdata/prd098/run_validation.py)）ので、Atari の session を改名すると宣言の無い名前に schedule を渡して起動時に止まり（[trainer.cpp:916-920](../../core/anet-core/src/trainer.cpp)）、新しい `[greedy]` session は止まらずに learn step 0 で発火する。`before` は実行体と設定を同時に切り替えるので「新しい実行体＋前の設定」の組を作れない。受入 1 の skip 有効の構成にはこの組が要る。D7・D8 に依存 |

## 5. 実装契約

### 5.1 設定キー

すべて `DefaultDQNAgent.` の下。不正値（型・列挙値の外）は Agent の構築時に、キーと値と期待値を示して `ANET_SYSTEM_ERROR` で止める（値を丸めない、既定へ置き換えない）。`skip.enabled = false` のときは検証だけして使わない（PRD098 §5.6 と同じ）。

| キー | 型・値域 | 既定 | 意味 |
|---|---|---|---|
| `learner.skip.feature_gradient` | `stop` ／ `merge` ／ `split` | `stop`（agent.txt。Atari・LunarLander は env の設定ファイルの葉で `merge`） | `stop`（遮断）は skip の入力を切り離し、skip の loss の勾配を primary へ流さない（今のまま）。`merge`（合算）は流して、総 loss を 1 回 backward し 1 組のモーメントで更新する（今のまま）。`split`（分割）は流して、primary のパラメータのモーメントを Q と skip の loss ごとに分ける（D2）。今の 2 値に 1 つ足すだけで、無効な組み合わせは無い |
| `learner.skip.sample_weight` | `uniform` ／ `kmax` | `uniform` | skip の loss のサンプルごとの重み。`kmax` は k_max ÷ バッチの k_max の平均を、IS 重みと一緒に掛ける（D6） |

`sample_weight` はどの `feature_gradient` とも組み合わせられ、PER の有無で式は変えない。

設定例（コロン記法）:

```
# agent.txt: 既定（?=）。feature_gradient は値が 1 つ増えるだけ
DefaultDQNAgent.@baseline : learner.skip.feature_gradient ?= stop
DefaultDQNAgent.@baseline : learner.skip.sample_weight ?= uniform

# metrics_scalar.txt の @skip: J に依らない評価側の指標（D8。session の終わりに学習側の座標で出る）
metrics.scalar.@skip : [50_agent_skip/21_greedy_decision_rate] ?= $eval.[greedy] $actor skip_session_decision_rate @session_end
metrics.scalar.@skip : [50_agent_skip/22_greedy_length_mean] ?= $eval.[greedy] $actor skip_session_length_mean @session_end
metrics.scalar.@skip : [50_agent_skip/03_skip_length_rate.[01]] ?= skip_length_rate.[1] @train $action_info
metrics.scalar.@skip : [50_agent_skip/17_skip_primary_grad_norm] ?= skip_primary_grad_norm @learn $learn_step $update_result
metrics.scalar.@skip : [50_agent_skip/18_skip_primary_clip_ratio] ?= skip_primary_clip_ratio @learn $learn_step $update_result

# LunarLander.txt の run.@skip: greedy の評価 session（D7）と、J に依る指標・報酬（D8）
run.@skip : A2.actor.[greedy].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@greedy
run.@skip : run.eval.[greedy].actor = greedy
run.@skip : run.eval_schedule.[greedy].interval = 1009
run.@skip : run.eval_schedule.[greedy].use_background = true
run.@skip : metrics.scalar.[50_agent_skip/23_greedy_length_rate.[01]] = $eval.[greedy] $actor skip_session_length_rate.[1] @session_end
run.@skip : metrics.scalar.[21_eval/09_greedy_reward] = mean.episode_return $runner @session_end $eval.[greedy]

# GridMaze.txt の run.@skip: スキップ長を ε-greedy で選ぶ（D3。spatial は無効なので epsilon.start/end）
run.@skip : A1.actor.[train].skip.policy.policy_type = EpsilonGreedy
run.@skip : A1.actor.[train].skip.policy.epsilon.start = 0.60
run.@skip : A1.actor.[train].skip.policy.epsilon.end = 0.05

# CLI・bat: 腕は葉で
DefaultDQNAgent.learner.skip.feature_gradient=split
DefaultDQNAgent.learner.skip.sample_weight=kmax
```

### 5.2 Learner の 1 回の更新（`split`）

1. current の forward は今と同じ（primary を forward し、その出力の `skip_context` をそのまま skip Network に渡す）。
2. Q の loss の、primary のパラメータへの勾配と、skip の loss（`loss_weight` を掛けたもの）の、primary と skip Network のパラメータへの勾配を、同じパラメータの値から別々に取る（D4）。
3. Q の loss の勾配（primary）、skip の loss の勾配の primary の分、skip の loss の勾配の skip Network の分に、それぞれ global norm のクリップを掛ける（D5）。
4. primary のパラメータは、Q 側のモーメントによる Adam の更新と、skip 側のモーメントによる Adam の更新の和だけ動く。skip Network のパラメータは skip 側の更新だけで動く。lr・betas・eps は primary と同じ値。decoupled weight decay は各パラメータに 1 回だけ掛ける（primary は Q 側の更新で、skip Network は skip 側の更新で。§6 の 8）。
5. GNS は今までどおり Q の loss と primary のパラメータで測る。
6. `loss_weight` は skip の勾配に掛かる。クリップ（3）が掛からない間は Adam の尺度不変性で効きはほぼ eps の分だけになり、クリップが掛かる間は skip 側の閾値を τ/w にしたのと同じ効きになる（D12）。

`stop`・`merge` では今と同一（D11）。

### 5.3 checkpoint

- `split` の Run の Save は、optimizer の状態に「primary のパラメータの skip 側のモーメント」の役割を足す（PRD098 §5.8 の役割ごとの書き方に 1 つ足す）。この役割は primary の全パラメータ（Noisy の σ を含む）を覆う。
- Load: 設定が `split` で archive にその役割が無ければ（`stop`・`merge` の Run の checkpoint）、skip 側のモーメントを初期化して WARN を 1 回出す（PRD098 §5.8 の、skip の無い checkpoint から有効にして分岐するときと同じ扱い）。設定が `stop`・`merge` で archive にその役割があれば `ANET_SYSTEM_ERROR`（黙って読み飛ばさない）。
- optimizer の options の置換（[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)）は役割ごとに行う。
- [ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) の「optimizer の param group は Network ごと」は、`split` では「primary は Q 側と skip 側の 2 組のモーメント」になる。ADR の先頭に注記を足す（書き換えない。クリップが Network 別であること、skip の target を行動にだけ使うこと、checkpoint の 4 object は変わらない）。新しい ADR は起こさない: ADR 0062 の判断（補助 Network の持ち方）は変わらず、変わるのは optimizer の状態の持ち方で、設定の 1 値で戻せるので「戻しにくい」に当たらないため（ADR を起こす 3 条件のうち 1 つ目を満たさない）。

### 5.4 メトリクスと Evaluation Trace

群は `50_agent_skip`（PRD098 D16）と、評価の報酬は各 env の評価の群。tag は 40 文字以内。

| tag | 元の鍵（取得元） | 意味 |
|---|---|---|
| `50_agent_skip/21_greedy_decision_rate` | `skip_session_decision_rate`（`$eval.[greedy] $actor … @session_end`） | `[greedy]` session の中で判断した lane の割合（判断の回数 ÷ lane の step 数。エピソード開始を含む）。ゴールの指標。学習側の exp step の座標 |
| `50_agent_skip/22_greedy_length_mean` | `skip_session_length_mean`（同上） | `[greedy]` session の中で判断した lane のスキップ長の平均 |
| `50_agent_skip/23_greedy_length_rate.[01]`〜`[J]` | `skip_session_length_rate.[j]`（同上。鍵の添字は 1 始まりのまま） | `[greedy]` session の中で判断した lane のうちスキップ長 j の割合。tag の添字は 2 桁 |
| `50_agent_skip/03_skip_length_rate.[01]`〜`[J]` | `skip_length_rate.[j]`（`@train $action_info`） | 学習側の長さ別の割合。今の `[1]`〜`[J]` の改名 |
| `21_eval/09_greedy_reward`・`10_greedy_reward_ema`・`11_greedy_ep_steps`・`12_greedy_ep_steps_max` | `mean.episode_return`、同 EMA（ema_alpha 0.1）、`mean.episode_steps`、`max.episode_steps`（`$runner @session_end $eval.[greedy]`） | LunarLander・GridMaze・DropMerge の `[greedy]` session の報酬とエピソード長。Atari は既存の `53_evalg/10_game_score_mean`・`20_game_len_mean` が同じ役 |
| `50_agent_skip/17_skip_primary_grad_norm` | `skip_primary_grad_norm`（`@learn $learn_step $update_result`） | `split` のとき、skip の loss の勾配の primary のパラメータの分のノルム（クリップ前。D5 の 2 つ目のクリップに掛けたもの）。`stop`・`merge` では NaN（出力されない） |
| `50_agent_skip/18_skip_primary_clip_ratio` | `skip_primary_clip_ratio`（同上） | `split` のとき、D5 の 2 つ目のクリップが発火した更新の割合（`37_agent_qtd/24_grad_clip_ratio` と同じ作り）。`stop`・`merge` では NaN |

- Actor の累積: Actor は、最後の `Sync()` からの判断の回数・lane の step 数・判断したスキップ長の和・長さ別の判断の回数を持ち、`skip_session_*` の鍵で割合と平均を返す。評価 session の頭で `Sync()` が呼ばれる（[trainer.cpp:233-237・346](../../core/anet-core/src/trainer.cpp)）ので、`@session_end` の値はその session の中の値になる。学習の Actor も同じ累積を持つが定期同期ごとに消えるので、学習側ではこの鍵を使わない（学習側は今までどおり `01`〜`04`）。skip 無効のときは NaN。
- `21`・`22`・`03` の `[01]`・`17`・`18` は [metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の `@skip` に置く（J に依らない）。`23` と `03` の `[02]`〜`[J]`、`21_eval/09`〜`12` は各 env の `run.@skip` に置く（Atari は `53_evalg` の群があるので `23` と `03` だけ）。
- `17`・`18` を `@skip` に置くのは、`run.@skip` が `metrics.scalar.$` の連鎖を丸ごと上書きする（[LunarLander.txt:29](../../apps/runner/config/LunarLander.txt)）ので、`split` の腕だけ別のプロファイルを CLI で足すと連鎖を書き直す事故（PRD098 §7 の段 2 で Munchausen の指標が落ちた形）が起きるため。`stop`・`merge` では NaN を返し、NaN は出力されない（[observers.cpp:1126](../../core/anet-core/src/observers.cpp)、[140 §7.1](../design/140_observability.jp.md)）ので D11 は保たれる（`metrics.scalar.defs` には載る）。
- `37_agent_qtd/21_grad_norm` は「primary のクリップに掛けた勾配のノルム」として読む。`stop`・`merge` では今と同じ（合算なら skip 由来を含む）、`split` では Q の loss の勾配だけになる（D5 の 1 つ目）。`24_grad_clip_ratio` もこれに従う。
- `50_agent_skip/15_grad_norm` は、どのモードでも skip Network のパラメータの、skip の loss の勾配のノルム（クリップ前）で、今と同じ。`split` では D5 の 3 つ目のクリップに掛けたノルムに当たる。
- 1 エピソードあたりの判断の回数は、`21` × 同じ session の評価の平均エピソード長で読む（どちらも `@session_end` で同じ座標）。LunarLander・GridMaze・DropMerge は `21_eval/11_greedy_ep_steps`、Atari は `53_evalg/20_game_len_mean`。これは近似で、`21` は session の全 lane の step で取り、エピソード長は採用したエピソードだけで取る。batch 1・1 エピソードの session（LunarLander・GridMaze・DropMerge）では一致し、Atari（batch 16・100 エピソード）では採用されなかった途中のエピソードの step が `21` に入る。厳密な回数が要るときは trace で取る（§8 の先送り）。
- Evaluation Trace は変えない（`skip_decision`・`skip_length` の列は PRD098 のまま）。

### 5.5 設定だけで作る腕（D9）

| 差 | 腕 | 設定（葉） |
|---|---|---|
| T2 | 合算・クリップの閾値 30 | `DefaultDQNAgent.learner.grad_clip_tau=30` |
| T3 | 実時間優先で、共有を畳み込みまでにする | primary の `main_feature` を、畳み込みの branch（`trunk`）と `Linear512 > LN > ReLU` の branch に分け、`net.body.output.[skip_context] = trunk`。skip Network は `taus` を bind して自前の τ 埋め込みと融合する（PRD098 §5.3 の「taus を bind する」形）。primary の branch の名前が変わる（§9 の 5） |
| T4 | 非共有 | skip Network が obs の鍵（Atari は `grid`）を bind して、primary と同じ畳み込みの構造を自前で持ち、`taus` を bind して τ 埋め込みと融合する。`feature_gradient` は効かない（skip が primary の出力を読まない）ので `stop` のまま |
| T10 | LunarLander で、スキップ長を平均に対する ε-greedy で選ぶ | `A2.actor.[train].skip.policy.policy_type=EpsilonGreedy`、`A2.actor.[train].skip.policy.spatial.epsilon.first=0.3`、`A2.actor.[train].skip.policy.spatial.epsilon.last=0.0`（行動の UQE の ε のラダーと同じ値）。LunarLander の `[train]` は spatial が有効で、ε-greedy は lane のラダー `spatial.epsilon` を使い、時間のスケジュール `epsilon.start/end` は使わない（[dqn_based_agent.cpp:1756](../../core/anet-core/src/dqn_based_agent.cpp)）。skip の方策は行動の方策の設定を写してから葉を重ねる（[default_dqn_agent.hpp:435-436](../../core/anet-core/include/anet/default_dqn_agent.hpp)）ので、葉を書かないと agent.txt の 0.4→0.01 が継承される。GridMaze は spatial が無効なので `epsilon.start/end` で書く（D3、§5.1） |

T3・T4 が設定の記法だけで書けるか（skip.net から primary の block の定義を引けるか、`taus` の bind と融合の書き方）は未確認（§9 の 6）。

### 5.6 性能の見積もり（ホットパス）

「単位」は B 行の forward 1 回分の計算（PRD098 §5.10 と同じ数え方。今の学習 1 回の更新は current 1、target 1〜2、backward 2 の 4〜5 単位）。

| 経路 | 頻度 | 単価の増分 | 見込み |
|---|---|---|---|
| `split` の Learner の 1 回の更新 | 1 回／更新 | 勾配を loss ごとに取るため、primary の backward がもう 1 回分（約 2 単位）。Adam の更新が primary のパラメータで 2 回分 | 4〜5 単位が 6〜7 単位（学習 1 回の更新で約 1.3〜1.5 倍）。`split` の腕だけで、既定では増えない。受入 4 で測る |
| `split` のメモリ | — | primary のパラメータ数 × 2 の値（skip 側の m と v） | Atari の primary は数百万パラメータの規模で、小さい |
| `kmax` の重み | 1 回／更新 | バッチの k_max の平均と、行ごとの掛け算 | 無視できる |
| Actor の累積（D8） | step ごと | lane ごとの加算（判断の旗と長さ） | 無視できる |
| `[greedy]` session（LunarLander・GridMaze・DropMerge） | 1 回／interval（1009・53・503 learn step）、エピソード 1、background | 既存の `eval` と同じ 1 エピソード分の forward と env の step | 既存の評価の約 1/5 の頻度。Atari は今と同じ（改名だけ） |

### 5.7 移行対象と等価性

- 設定: [agent.txt](../../apps/runner/config/agent.txt) の `DefaultDQNAgent.@baseline` に `sample_weight` を `?=` で足す（`feature_gradient` は値が増えるだけ）。[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の `@skip` に `21`・`22`・`17`・`18` を足し、`03_skip_length_rate.[1]` を `[01]` にする。[Atari.txt](../../apps/runner/config/Atari.txt) は `greedy_dist` を `greedy` に改名（`run.eval.[greedy_dist]`・`run.eval_schedule.[greedy_dist]`・`$eval.[greedy_dist]`・`run.@eval2ch`・`run.@eval2ch_r1`・`run.@eval2ch_r1_50m`・`run.@gdist400` の 27 か所。プロファイル名と群名 `53_evalg` はそのまま）、`run.@skip` に `23` と `03` の `[02]`〜`[10]` の改名。[LunarLander.txt](../../apps/runner/config/LunarLander.txt)・[GridMaze.txt](../../apps/runner/config/GridMaze.txt)・[DropMerge.txt](../../apps/runner/config/DropMerge.txt) の `run.@skip` に `[greedy]` の actor と session（D7）、`23`・`21_eval/09`〜`12`、`03` の改名（D8）。GridMaze の `run.@skip` にスキップ長の ε-greedy の葉（D3）。
- ツール: [atari5.py:1085](../../viewers/metrics-tools/atari5.py)（Atari-5 の集計の `eval_name`）を `greedy` に。
- 検証資材: session 名 `greedy_dist` を列挙・指定する [prd084/summarize_performance.py:18](../../core/anet-core/testdata/prd084/summarize_performance.py)・[prd084/summarize_forward_refactor.py:84](../../core/anet-core/testdata/prd084/summarize_forward_refactor.py)・[prd084/run_performance.ps1:78](../../core/anet-core/testdata/prd084/run_performance.ps1)・[prd084/run_forward_refactor.ps1:56・135・164](../../core/anet-core/testdata/prd084/run_forward_refactor.ps1)・[prd084/run_review_validation.ps1:67](../../core/anet-core/testdata/prd084/run_review_validation.ps1)・[prd094/run_validation.py:72](../../core/anet-core/testdata/prd094/run_validation.py)・[prd097/run_validation.py:76](../../core/anet-core/testdata/prd097/run_validation.py)・[prd097/audit_validation.py:62](../../core/anet-core/testdata/prd097/audit_validation.py)・[prd099/run_validation.py:52](../../core/anet-core/testdata/prd099/run_validation.py) と、それらの使い方を記す [prd084/README.md:35・63・135・179](../../core/anet-core/testdata/prd084/README.md) を `greedy` に。prd084 の verification.md・review_verification.md・forward_refactor_verification.md は検証の記録なので書き換えない。PRD098 の検証資材（[testdata/prd098](../../core/anet-core/testdata/prd098/README.md)）の [run_validation.py:258](../../core/anet-core/testdata/prd098/run_validation.py)・[verify_config_migration.py:98](../../core/anet-core/testdata/prd098/verify_config_migration.py) の `03_skip_length_rate.[1]` を `[01]` に。PRD098 の検証資材の凍結した設定（`resolved-before`）は過去の記録として書き換えず、本書の検証資材（D14、`testdata/prd103/`）が受入 1 を担う。D3・D7・D8 の env の設定を入れたあとの skip 有効の値を新しい基準として `testdata/prd103/` の README に記す。
- 文書: [diagnostic_cost.md](../wiki/mechanisms/diagnostic_cost.md) と [999_eval_schedule_first_fire](999_eval_schedule_first_fire_10prd.md) の `greedy_dist` は、過去の測定の記述として残す（書き換えない）。
- テスト: 既存の skip のテスト（[dqn_based_agent_test.cpp](../../core/anet-core/src/dqn_based_agent_test.cpp) の `feature_gradient` を切り替えるもの）は `stop`・`merge` のまま通る。[observers_test.cpp:528](../../core/anet-core/src/observers_test.cpp) の tag の例を `[01]` に。
- 振る舞いが変わるもの: 既存の評価 slot・session の方策は変わらない。変わるのは、skip 有効の Run に `[greedy]` session と tag が増えること（LunarLander・GridMaze・DropMerge）、GridMaze の skip 有効の Run の学習側のスキップ長の選び方（D3）、学習側の `03_skip_length_rate` の tag 名、Atari の評価 session の名前（乱数の系列も。D7）。2026-10-26 から既定で有効の LunarLander と、2026-10-27 から実時間優先の既定で有効の Atari が当たる。過去の Run と実験記録は書き換えない。 `[greedy]` session を足した Run は、同じ seed でも学習前から軌跡が変わる（段 2 の合算の基準は、10/26 の同 seed の Run と `[eval]` の 2 点目（14848 step、学習前）から違い、学習側の報酬の EMA も 37 エピソード目から違った。間に `[greedy]` の初回のエピソードが入る。どの乱数の系列を共有しているかは未確認。[2026-10-28_temporl-split-kmax-epsilon](../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md) 探索ブロック 02）。`[greedy]` を足す前の Run とは比べず、同じビルド・同じ設定の腕どうしで比べる（§7 の段 2）。
- 設計文書: [200](../design/200_dqn_agents.jp.md)（skip の節に `feature_gradient` の 3 値と `sample_weight`、Actor の session の累積）、[140](../design/140_observability.jp.md)（`[greedy]` session の skip の指標と `$actor @session_end` の座標、添字 2 桁の例）。
- ADR: [ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md) の先頭に `split` の注記（§5.3）。
- PRD098 の改訂（2026-10-28 に済ませた）: D6 の理由に「3 値目 `split` を本書 D2 で足した」、D9 の理由に「行動が `Greedy` 型の slot でだけ成り立ち、学習したスキップ価値そのものは本書 D7 の `[greedy]` session で測る。GridMaze の ε-greedy はプロファイル」、§6 の 2 と §7 の段 2 に T14 の注記、§9 の 3 の行き先を本書 D8 に。第三弾の §9 の 9 は本書 D3 を指して閉じた。
- 等価性: 新しい設定が既定のとき、§7 の受入 1 の比較で metrics checksum が一致する。

### 5.8 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場 |
|---|---|---|---|
| `LearnerConfig` | 変更 | `skip.feature_gradient` の 3 値目 `split` と `skip.sample_weight` を持つ | Agent の設定（agent） |
| `DefaultDQNAgent` | 変更 | 2 つのキーの読み取りと列挙値の検証 | DefaultDQN（default_dqn_agent） |
| `dqn::Learner` | 変更 | `split` の勾配の取り方・クリップ・2 組のモーメントの更新、`kmax` の重み、`17`・`18` の値、optimizer の状態の役割の Save/Load | DQN 共通 component（dqn_based_agent） |
| `dqn::Actor` | 変更 | `Sync()` からの skip の判断の累積と `skip_session_*` の鍵 | DQN 共通 component（dqn_based_agent） |
| `dqn::BatchUpdateResult` | 変更 | `skip_primary_grad_norm`・`skip_primary_clip_ratio` を載せる（RL 共通の基底 `BatchUpdateResult` は変えない） | DQN 共通 component（dqn_based_agent） |

## 6. 細部（2026-10-28 のグリルで推奨の既定で埋め、別モデルの検討で直した）

1. `[greedy]` session の actor: LunarLander・GridMaze は `[greedy].$ = [eval] > @greedy`（online）、DropMerge は `[greedy].$ = [eval_target] > @greedy`（target）。clone_model は継承元に従う。各 env の判定の slot の network に揃える（D7）。Atari は既存の `[greedy]`（`[eval]` の ε 0。online）。
2. `[greedy]` session の頻度と大きさ: エピソード 1・batch 1（既存の `eval` と同じ）、interval は既存の評価（LunarLander 200、GridMaze 10、DropMerge 100 learn step）の約 5 倍で互いに素な値（1009・53・503）、`use_background = true`。費用を既存の評価の約 1/5 に抑え、既存の評価と同じ learn step に発火する重なりを減らす（初回の learn step 0 と最小公倍数では重なる。既存の `eval` と `eval_target` は同じ interval で重なっている）。背景の評価は、次の発火のときに前回が終わっていなければ学習が待つ（[observers.cpp:586](../../core/anet-core/src/observers.cpp) `WaitBackgroundEval`）ので、interval は 1 回の評価の実時間より長く取る。
3. `[greedy]` session の置き場: 各 env の `run.@skip` の中。skip 無効の Run の metrics checksum を変えない（D11）。Atari は既定の連鎖（`run.@eval2ch`・`run.@eval2ch_r1`）にすでにある。
4. `[greedy]` session の報酬・エピソード長の tag は `21_eval/09_greedy_reward`・`10_greedy_reward_ema`（ema_alpha 0.1）・`11_greedy_ep_steps`・`12_greedy_ep_steps_max`。既存の `21_eval/01`〜`08` の並びに続ける。
5. Atari の改名の範囲は §5.7 の一覧のとおり（Atari.txt の 27 か所、atari5.py、検証資材のスクリプト 9 本と prd084 の README）。プロファイル名 `@gdist400` と群名 `53_evalg` はそのまま。別名は置かない。
6. `17_skip_primary_grad_norm`・`18_skip_primary_clip_ratio` は `@skip` に置き、`split` 以外では NaN（出力されない）。
7. 1 エピソードあたりの判断の回数は `21_greedy_decision_rate` × `21_eval/11_greedy_ep_steps`（Atari は `53_evalg/20_game_len_mean`）で読む。どちらも `@session_end` で同じ座標。判断率は全 lane の step、エピソード長は採用したエピソードだけなので近似（§5.4）。
8. `split` の lr・betas・eps は primary と同じ値（著者も 2 つの Adam に同じ値を固定で書き、skip 用の引数は無い。T1）。skip 専用の lr は PRD098 §8 の先送りのまま。decoupled weight decay は各パラメータに 1 回だけ掛ける（primary は Q 側の更新で、skip Network は skip 側の更新で。今の group の decay の値のまま。LunarLander 1e-4・GridMaze 1e-2・DropMerge 1e-3 の env で、`split` と `merge` の比較に decay の有無の交絡を入れないため）。skip の loss の勾配が 0 のパラメータ（primary の Head など）は、skip 側のモーメントを持っても更新に効かない（著者の形と同じ）。モーメントを持たせるかは実装に任せる。
9. `kmax` の重みの分母は、そのバッチの k_max の平均（サンプルの数で割る）。
10. checkpoint は §5.3。`split` の Run は役割を 1 つ足し、`split` で役割の無い archive は初期化して WARN 1 回、`stop`・`merge` で役割のある archive は `ANET_SYSTEM_ERROR`。
11. `[greedy]` session の初回の発火（learn step 0 で必ず 1 回発火する。[util.hpp:271](../../core/anet-core/include/anet/util.hpp) `IntervalGate::ShouldFire`、[999_eval_schedule_first_fire](999_eval_schedule_first_fire_10prd.md)）の点は、未学習の方策の捨て点として扱い、抑止はしない。
12. 段 3 の判定は、学習側の値に加えて `53_evalg`（改名後も同じ条件）の点も使う。最初の 10M は 0・5.05・10.1M の 3 点なので、学習側の値と併用する（§7）。
13. GridMaze の `run.@skip` に置くスキップ長の葉は §5.1 の設定例（第三弾の段 0 の腕と同じ。spatial 無効なので `epsilon.start/end`）。既定が変わるので、受入 1 の GridMaze の skip 有効の基準は変更前の設定で比べ、新しい基準を README に記す（D14）。
14. 論文解説（[reports/biedenkapp2021_temporl_paper_2026-10-03.md](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)）に T14・T15 を注記しない。解説は論文の記述の正本で、コードの照合は本書と wiki が持ち、wiki の目録から辿れる。
15. `[greedy]` session のスキップ長の Greedy は、分布型では分位平均の argmax（PRD098 D9 の greedy と同じ。行動の方策と共有）。
16. Actor の累積（D8）の鍵は `skip_session_decision_rate`・`skip_session_length_mean`・`skip_session_length_rate.[j]`。学習の Actor でも同じ鍵を返すが、定期同期（`sync_interval`）ごとに消えるので、学習側の指標には使わない（§5.4）。

## 7. 検証と受入

評価の値は PRD098 §7 と同じ評価 slot から読み、`[greedy]` session の値を並べる。用語:

- **AUC**: 評価 slot の評価報酬の、Run 全体での平均。
- **A が B 以上**: 腕 A の 3 seed の AUC の平均が、腕 B の 3 seed の AUC の最小値を下回らない（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md) §7 と同じ）。
- **A が B を上回る**: 腕 A の 3 seed の AUC の最小値が、腕 B の 3 seed の AUC の最大値を上回る（ブレ幅を超えて上回る。PRD098 §7 の段 2 で分離型が無効を上回ったと判じたのと同じ取り方）。
- **qbert の突破率**: 学習側の、2.5M 窓（exp step）のゲームのうち「29,000 点以上かつ `round_clear` ≥ 8」の割合（[campaign](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) の判定指標と同じ）。評価側（`53_evalg`）で同じ条件を数えたものは「評価のエピソードのうち突破したものの割合」と書き分け、突破率とは呼ばない。

| 段 | env と構成 | 腕 | seed・長さ | 判定と分かれ方 |
|---|---|---|---|---|
| 0 | 本書の検証資材（D14）の 6 構成 | 新しい設定は既定 | 各 1 seed、短縮条件 | 受入 1。不一致なら実装の不備 |
| 1（コードを変えない。2026-10-27 に 11 の bat へ積んだ） | qbert、サンプル効率優先（`run_20261027-004945_a5_qb_aln_t98mrg_s1` と同じ構成） | 分離×遮断（`run_20261027-175541_a5_qb_aln_t98stp_s1`）、合算・`loss_weight` 0.3、合算・クリップの閾値 30。比べる相手は合算（同 Run）と無効（`run_20261025-231207_a5_qb_aln_btreps_s1`） | 1 seed、50M | 最初の 10M の評価（`52_eval2`、2.5M 窓）、突破率が初めて 50% を超える窓、ε=0 の評価（`53_evalg`。40.6M・45.7M）を並べる。「無効の帯に戻る」は、突破率が初めて 50% を超える窓が無効（32.5〜35M）と同じか早いこととする。遮断か重み 0.3 だけが無効の帯に戻れば、合算の共有部の混ざり方が中盤の遅れの候補で、段 3 で `split` を確かめる。閾値 30 だけが戻ればクリップとの結合が候補。両方が戻れば両方を候補とし、段 3 は `split`（D5 でクリップの結合も無くなる）を流す。どれも戻らなければ seed のばらつきの可能性が高く、段 3 に合算の seed 2 を足す。1 seed なので、どの場合も合否ではなく段 3 の腕の選び方に使う。あわせて §2.3 の勾配ノルム以外のメトリクス（眠った・死んだユニットの率、特徴量のノルム、Munchausen の log π の clip 率）を 5M 窓で並べる（D13。判定には使わない）。遮断の眠ったユニットの率が 50M まで無効の線に留まり、合算だけが追い越すなら、skip の loss の勾配が共有部に入ることが前提になる（§2.3 の 3 候補、§9 の 11）。結果（2026-10-31、[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）: 遮断・重み 0.3 とも 50M までに突破率 50% を超えず、無効の帯に戻らなかった（47.5〜50M の窓で 38%・35%）。分かれ方に従って合算の seed 2（サブ機、`run_20261029-165351_a5_qb_aln_t98mrg_s2`）を流し、これも 50M までに超えなかった（同 28%）。閾値 30（`run_20261030-215554_a5_qb_aln_t98c30_s1`）も 50M までに超えず（同 31%）、3 腕とも無効の帯に戻らなかった。閾値 30 の Q 側のクリップ率は 10〜15M の 36.4% から 40〜45M の 16.6% へ下がった（合算は閾値 10 で 99.9%・81.8%）が、`53_evalg` の 3 回平均は 24,201 で、同じ重みを毎 step 判断で評価した `54_evalgs` も 27,732 だった。これを受け、点は変わらなくても TempoRL 有効では勾配ノルムが約 2 倍になるので、Atari の `run.@skip` で `grad_clip_tau` を 30 にした（2026-10-31。閾値は状況に応じて調整する設定なので、張り付いたままにしない。ノルムの比だけから合わせるなら 20 前後だが、試した 30 を使う）。最初の 10M の評価（`52_eval2`）は、2.5〜5M で遮断 4,657・重み 0.3 6,050（合算 6,078、無効 3,768）。`53_evalg` の 35.55〜45.65M の 3 回平均は、遮断 26,985、重み 0.3 28,459、合算 27,058・26,523、無効 40,999・30,415。眠ったユニットの率（40〜50M）は遮断 8.65%、重み 0.3 6.56%、閾値 30 5.27%、合算 12.48〜13.66%、無効 9.00〜9.66% |
| 1'（コードを変えない。2026-10-28 に実施、結果は [2026-10-28_temporl-skip-policy-6x10](../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md)） | GridMaze 6×10（PRD098 の段 1 と同じ構成。J=7、分離×遮断） | `run.@skip`（UQE。基準）、スキップ長を ε-greedy にした腕（CLI の葉。D3） | 3 seed、500k | 届く速さ（評価の EMA が初めて 0.5 以上になる step。500k の中で届かない seed は、腕なら条件を満たさない、基準なら 500k として最大を取る）と終盤の安定さ（最後の EMA）を seed ごとに取る。条件は 1 つ: ε-greedy の 3 seed の届く速さの最大が基準（UQE）の 3 seed の最大以下で、かつ ε-greedy の終盤の安定さの最小が基準の最小以上（seed ごとに比べ、平均は使わない）。満たせば GridMaze の `run.@skip` の既定を ε-greedy にする（D3・§8・§9 の 1 が指すのはこの条件で、§9 の 1 の末尾の一般則は掛けない）。満たさなければ既定は UQE のまま、5×5 の既定 seed を使う Run だけ CLI の葉で ε-greedy にする。結果（2026-10-28）: 届く速さは ε-greedy 282k／311k／313k、UQE 269k／323k／359k、終盤は 6 本とも最後の EMA 0.93 で、条件を満たした。既定を ε-greedy にする（D3）。ε-greedy の学習側は j=1 が 93〜95%・判断率 0.88〜0.89 で、6×10 ではほぼスキップしない方策になった（UQE は j=2 が 15〜41%）。`[greedy]` session はまだ無いので評価側の判断率は取っていない |
| 2 | LunarLander（現行の LunarLander.txt の既定。IQN、J=4、分離×合算、`[greedy]` session あり） | 合算（基準）、分割（`split`）、合算・`kmax`、スキップ長を ε-greedy（T10） | 3 seed、2M（全部の腕を同じビルドで流す） | 各腕が基準以上か（評価は `[eval]` の policy 報酬の AUC）。基準を上回る腕（上の定義）は、その方式を LunarLander の既定にするかを決める材料にする（§9 の 1）。`[greedy]` session の判断率（`21`）・長さ別の割合（`23`）・報酬（`21_eval/09`）を並べ、`kmax` で長い j の割合が増えるか、T10 で判断率がどう動くかを読む（観察）。結果（2026-10-28、[2026-10-28_temporl-split-kmax-epsilon](../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md)）: AUC は合算 135.2／112.5／125.2（平均 124.3）、分割 147.8／133.3／98.1（126.4）、`kmax` 140.5／134.8／134.1（136.5）、ε-greedy 149.8／104.0／84.0（112.6）。3 腕とも基準以上（ε-greedy は平均 112.6 対 基準の最小 112.5）で、基準を上回る腕は無い（`kmax` は最小 134.1 対 基準の最大 135.2）。LunarLander の既定は据え置き（§9 の 1）。`kmax` は基準以上なので段 3 に分割・`kmax` の腕を足す。`[greedy]` の判断率（1.5M〜）は合算 0.53〜0.56、分割 0.47〜0.57、`kmax` 0.56〜0.64、ε-greedy 0.57〜0.58 で、`kmax` は長い j を増やさず j=1 の割合が増えた（0.53〜0.66 対 合算 0.48〜0.53）。学習側の判断率・長さ別の割合はどの腕も合算と同じ（判断率 0.46〜0.49）。基準の 3 本は 10/26 の同 seed の Run（131／143／138）と値が違い（§5.7）、基準の設定の 6 回分の AUC は 112〜143 に散るので、3 seed のブレ幅は分散を小さく見積もっている |
| 3 | qbert、サンプル効率優先（段 1 と同じ構成） | 分割（`split`）。段 2 で `kmax` が基準以上なら分割・`kmax` も（2026-10-28: 段 2 で `kmax` は基準以上だったので流す）。段 1 で seed のばらつきの可能性が高ければ合算の seed 2 も | 1 seed、50M | 基準は段 1 の合算の Run（`run_20261027-004945_a5_qb_aln_t98mrg_s1`。ビルドが違う）。評価の条件は D7 で変わらない（`53_evalg` は改名前と同じ条件。乱数の系列は違う）ので、`53_evalg` の点（0・5.05・10.1M…）と学習側の値（最初の 10M の学習側のゲームの平均点（2.5〜5M・5〜7.5M・7.5〜10M の 3 つの 2.5M 窓の平均。0〜2.5M は立ち上がりなので除く。§2.3 の評価の表と同じ窓）、突破率）の両方で見る。`split` が合算より早く突破率 50% を超え、最初の 10M の学習側の平均点が合算の 90% 以上なら、Atari の既定を `split` にするかを決める材料にする（§9 の 1）。1 seed で、基準とビルドも違うので、合否ではない。seed 2 を足すときは、合算の seed 2 も同じビルドで流し、既定を変える判断は同じビルドの 2 seed で行う（§9 の 1）。機序の問い（§2.3 の候補）には、s1・s2 で揃っていたクリップ率・勾配ノルム・眠ったユニットの率を使い、分割で合算より増えるか減るかを並べる（D13。候補 (1) と (3) を分ける）。結果（2026-10-31、[実験記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md)）: 分割（`run_20261028-215839_a5_qb_aln_t103spl_s1`）も合算も 50M までに突破率 50% を超えず（分割は 47.5〜50M の窓で 42%）、前半の条件を満たさない。最初の 10M の学習側の平均点は 9,909（合算 s1 の 109%）。`53_evalg` の 35.55〜45.65M の 3 回平均は 29,066（合算 27,058・26,523、無効 40,999・30,415）。機序の値: Q 側の勾配のクリップ率（閾値 10）は 10〜35M の 5M 窓で 47〜67%（合算 93〜100%、無効 51〜79%）、勾配ノルムは 11〜15（合算 17〜35、無効 12〜17）、眠ったユニットの率（40〜50M）は 14.16%（合算 12.48〜13.66%）。分割・`kmax` は流さず見送る（2026-10-31。qbert の TempoRL の調査は、合算の変種 8 本が `53_evalg` の 3 回平均 24,201〜29,066 に固まり無効に届かなかったので区切った。再開条件: Freeway・BeamRider など別のゲームで、合算が無効を Greedy の評価で上回ったとき） |
| 4 | qbert、実時間優先（Atari.txt の既定の連鎖、J=10） | 合算（基準。同じビルドで取り直す）、非共有（T4）、共有を畳み込みまで（T3） | 2 seed、10M | PRD098 の段 3 のふるいと同じ 3 つの値（評価（ε 0.01）の Run 全体の平均、同じ評価の 7.5〜10M、学習側のゲームの平均点の 7.5〜10M）を並べる。3 つのどれでも、2 seed とも同じ seed の基準を上回る腕があれば、実時間優先の構成を変えるかを決める材料にする。2026-10-31: 流さず見送る（段 3 と同じ理由と再開条件） |

T10 の GridMaze 5×5 の既定 seed の結果（ε-greedy でゴールに届く）は第三弾の段 0 で得ているので、ここでは取り直さない。

受入（外から観測できるもの）:

1. 新しい設定が既定で、次の 6 つの比較の metrics checksum が、本書の前のビルドと一致する。比べ方と短縮条件は PRD098 の検証資材（[README](../../core/anet-core/testdata/prd098/README.md)）の段 0 と同じ（`90_perf/`・`91_resource/`・`92_resource/` を除く全 scalar 点の完全一致）。実行は本書の検証資材（D14、`testdata/prd103/`）で行い、評価を止める session の一覧は今の解決済みの設定から取る。skip 有効の 2 構成は、新しい実行体で、D7・D8 の env の設定を足す前の設定（snapshot）を動かして比べる。
   - skip 無効: GridMaze、LunarLander、Atari BTR 基準、DropMerge nzrep 基準。
   - skip 有効: GridMaze（現行の既定。QR、分離×遮断）、LunarLander（現行の既定。IQN、分離×合算）。
2. 単体テスト: 小さな Network で、`split` の 1 回の更新が「Q の勾配の Adam の更新」と「skip の勾配の Adam の更新」の和に一致する（数値例）。`stop`・`merge` は今と同じ。`feature_gradient` の列挙の外の値が構築時のエラーになる。`kmax` の重みの計算の数値例（k_max の違う 3 行。重みの和がバッチの行数に等しいこと。期待値の一致は確かめない）。§5.1 の不正値のエラー。`split` の checkpoint の Save と Load の往復、`merge` の checkpoint を `split` で読むときの WARN、`split` の checkpoint を `merge` で読むときのエラー。Actor の累積が `Sync()` で消え、判断の旗と長さから割合と平均が合う数値例。
3. smoke Run（GridMaze と LunarLander の短縮条件、skip 有効）の `metrics.jsonl` と `metrics.scalar.defs` に `50_agent_skip/21_greedy_decision_rate`・`22_greedy_length_mean`・`23_greedy_length_rate.[01]`・`03_skip_length_rate.[01]`・`21_eval/09_greedy_reward` が出て、`21`〜`23` の `runner` が `train`（学習側の座標）であり、`config/config_data.txt` に `run.eval.[greedy]` と `[greedy]` の actor がある。GridMaze の skip 有効の短い Run の `config/config_data.txt` に `actor.[train].skip.policy.policy_type = EpsilonGreedy` がある（D3）。Atari の skip 有効の短い Run の `config/config_data.txt` に `run.eval.[greedy]` があり、`greedy_dist` が無い。
4. 性能: Atari BTR 基準（N=128、B=256、BF16、prefetch、Release）で、`merge` と `split` を逆順の 4 Run（PRD098 の検証資材の `performance` と同じ測り方）で比べ、学習 1 回の更新の実時間の比を `core/anet-core/testdata/prd103/README.md` の性能の節に記録する（測り方、4 Run の Run 名と値、比）。§9 の 4 には比だけを転記する（閾値は §9 の 4）。
   - 2026-10-31 に測った（今のソースの Release）。比（split／merge）は 1.117（merge 0.0511・0.0513、split 0.0571・0.0572 秒／更新）。記録は検証資材の README の性能の節。

## 8. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
|---|---|---|
| 著者の更新の順（skip を先に step し、更新後のパラメータで Q の勾配を取る） | 範囲外 | current の forward が 1 回増え、差は η の 1 歩分（D4）。再開条件: `split` が `merge` を上回り、順の違いで結果が変わる兆候（`split` の腕どうしの不安定さ）が出たとき |
| `optimizer_mode = joint／per_loss` の別キーと、`per_loss`×`stop` の構築時エラー（起票時の D2・D3） | 不採用 | 設計の空間が遮断・合算・分割の 3 点なので、`feature_gradient` の 3 値目で表せる（D2）。別キーだと直交に見えて 1 つの組み合わせが無効になり、検査とテストが増える |
| `[eval]`（と継承する slot）のスキップ長だけを Greedy にする形（起票時の D7） | 不採用 | 行動は ε 0.01／UQE、スキップ長だけ greedy の混ざった条件になり、skip 有効の Run の `52_eval2`・`21_eval` の時系列が切れる。学習したスキップ価値そのものは `[greedy]` session で測る（D7） |
| `[eval]` の行動も greedy にして著者の評価の条件を `[eval]` に写す形 | 不採用 | 評価の方策は設定なので移せるが、Atari の BTR 準拠の ε 0.01 や悲観の UQE は TempoRL 以外の比較の条件でもあり、本書の関心の外まで動かす（D7） |
| 評価側の skip 指標を、評価 slot の action_info の step 値（`$eval.[greedy] @train $exp_step … $action_info`）で取る形（起票時の D8） | 不採用 | 評価 runner 自身の step 軸に載り、学習の進みや腕どうしと並べられず、判断の回数も作れない（T13）。Actor の session の累積を `@session_end` で読む（D8） |
| エピソードごとの判断回数を trace 行で記録する形 | 先送り | 著者の T13 と同じ量だが、scalar としては並ばず集計が別工程になる。再開条件: 著者の評価のエピソードごとの判断回数と直接比べるとき、または Atari で近似（§5.4）ではない厳密な回数が要るとき（D8 の累積の上に足せる） |
| EvalRunner が `$action_info` の鍵を session で集約する汎用の形 | 範囲外 | 汎用層（EvalRunner）の変更で、本書の関心の外まで広がる。再開条件: Agent の累積で足りない鍵が複数出たとき |
| `split` で `loss_weight` を skip 側の歩幅の倍率に読み替える形、`split` で `loss_weight ≠ 1` を構築時エラーにする形 | 不採用 | 前者は同じキーがモードで意味を変える。後者は `merge` 用に 0.3 を置いた共有プロファイルを `split` に切り替えると止まる。`loss_weight` は skip の loss の重みのままにし、歩幅は skip 専用 lr（下）で扱う（D12） |
| 添字の改名の代わりに Metrics Viewer の Tags 一覧を数値を含む自然順にする形 | 不採用（本書） | 過去の `[1]` の Run も揃う利点があるが、本書は tag の改名を選んだ（D8）。Viewer の並びは別の小さな直しの候補として工程末に挙げたが、起票しないと決めた（2026-10-28） |
| GridMaze のスキップ長の ε-greedy を、段 1' の前に既定にする形 | 先送り（2026-10-28 に再開条件を満たし、D3 で既定にした） | 5×5 の既定 seed・1 seed の結果だけでは既定を動かさなかった。再開条件: §7 の段 1' の条件（ε-greedy の 3 seed の届く速さの最大が UQE の最大以下で、終盤の安定さの最小が UQE の最小以上。届かない seed は、腕なら条件不成立、基準なら 500k として扱う）を満たす。2026-10-28 に満たした |
| skip 専用の replay（T7） | 先送り | PRD098 D11 の理由（PER・prefetch・世代 key の 2 系統、obs の複製）。再開条件: `kmax` の重み（D6）を入れても長い j の学習が遅く、skip のデータの古さが効くと分かったとき（skip の起点だけ直近から引く腕で確かめる） |
| skip と Q で別々にバッチを引く（T8） | 先送り | PRD098 D14 の理由（forward と抽出が 2 倍）。再開条件: T7 と同じ |
| 行動をスカラーの添字で入れる形と、10 ユニットの文脈層（T9） | 範囲外 | 行動に順序を仮定する著者の癖で、anet の one-hot と分離型のほうが素直（PRD098 D5）。実測でも分離型が包含型と同等以上だった（PRD098 §7 の段 1〜3） |
| IQN の τ の揺らぎを skip の選択から除く（T11） | 先送り | 影響は小さい見込み。専用方策の `tau_rule` を fixed にすれば設定で消せる見込み（未確認）。再開条件: `[greedy]` session で、同じ状態のスキップ長が揺れることが Evaluation Trace で目立ったとき |
| skip 専用の lr とクリップの閾値 | PRD098 §8 のまま | `split` で共有部の配分は直るので、まず段 2・段 3 の結果を見る。再開条件: `split` で skip 側の歩幅を変えたくなったとき（D12） |
| 段 3 の分割・`kmax` と段 4（実時間優先の非共有・畳み込みまで共有） | 見送り（2026-10-31） | qbert で合算の変種 8 本が同じ帯に固まり、無効に届かなかったので、qbert の調査を区切った（§7 の段 1・段 3 の結果）。再開条件: Freeway・BeamRider など別のゲームで、合算が無効を Greedy の評価で上回ったとき。効くゲームで、共有の度合い（段 4）と長い j の学習（`kmax`）を詰める |
| 全 k の一括（第三弾の (K)） | 第三弾 | `kmax` の重み（D6）は同じ偏りに安く当たる。(K) を作るかは第三弾の段 0 の結果のまま（第三弾 §9 の 10） |
| Munchausen の log π の clip 率を判断の旗で分ける計測と、途中の行の Munchausen の項の扱い | 範囲外 | 著者との差ではなく、TempoRL と Munchausen の組み合わせの関心（D13）。replay のサンプルの出力に判断の旗を運ぶ契約の変更が要る。再開条件: 段 1・段 3 の観察で、途中の行の扱いが成績に効く兆候が出たとき（§9 の 12。2026-10-28 に起票しないと決めた） |
| 眠った・死んだユニットが増えることへの手立て | 範囲外 | 機序が未確認で、可塑性の関心（D13）。再開条件: 段 1・段 3 で機序の候補が絞れたとき（§9 の 11） |
| `[greedy]` session を skip 無効の Run の既定にすること | 先送り | skip 無効の Run の metrics checksum と実時間が変わる（D11）。再開条件: ε=0 の評価を skip 無効の比較でも常用すると決めたとき |

## 9. 未決

| # | 事項 | 扱い |
|---|---|---|
| 1 | `split`・`kmax`・スキップ長の ε-greedy（GridMaze・LunarLander）を、どの env の既定にするか | 既定は今のまま（`feature_gradient` は今の値、`uniform`、行動と同じ規則）。GridMaze は §7 の段 1' の条件を 2026-10-28 に満たしたので、実装で `run.@skip` の既定を ε-greedy にする（D3。決まった）。LunarLander は段 2（2026-10-28）で基準を上回る腕（§7 の定義）が無かったので、既定は据え置き（閉じた。`kmax` は基準以上だが最小の seed が基準の最大を下回る）。Atari は段 3 で `split` が合算より早く突破率 50% を超え、最初の 10M の学習側の平均点（2.5〜5M・5〜7.5M・7.5〜10M の 3 窓の平均）が合算の 90% 以上なら既定の候補に。LunarLander と Atari の既定を変える判断は、同じビルドの 2 seed 以上で最小の seed が基準の最大を上回るときに限る（比較はブレ幅基準）。この一般則は GridMaze には掛けない（5×5 で届かない原因が分かっており、6×10 で悪くないことで足りる）。agent.txt の既定にするかは、2 env 以上で同じ向きのときだけ決める |
| 2 | 段 1・段 3 の読み方の閾値 | 「無効の帯に戻る」（突破率が初めて 50% を超える窓）と、段 3 の「最初の 10M の学習側の平均点が 90% 以上」を既定に置いた（§7）。1 seed なので、段 1 の結果を見てから段 3 の腕を決めるときに確かめ直す |
| 3 | `split` での `loss_weight` の意味 | 2026-10-28 に閉じた（D12。skip の loss の重みのまま） |
| 4 | `split` の性能の閾値 | 既定は記録だけ（受入 4）。見積もりは 1.3〜1.5 倍（§5.6）。Atari の既定にするなら閾値を置くかを決める。2026-10-31 の実測は 1.117（受入 4） |
| 5 | T3 の腕で primary の branch を分けること | 実時間優先の primary の branch の名前が変わり、今の checkpoint を読めなくなる。腕を別のプロファイルに置き、既定の構造は変えない（既定）。既定にするなら移行を決める |
| 6 | T3・T4 が設定の記法だけで書けるか | skip.net から primary の block の定義を引けるか、`taus` の bind と τ 埋め込みの融合を skip.net の中で書けるかを、実装の最初に確かめる。書けなければ、要る最小の記法を足すかを決める |
| 7 | 第三弾（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)）との結び | 2026-10-28 に閉じた。同書 §9 の 9（skip の方策の既定）は本書 D3 で決め、同書の (K) は同書の段 0 の結果（作らない）のまま。再開を考えるときは本書の段 2・段 3 の `kmax` の結果も材料にする（同書 §9 の 10） |
| 8 | 論文解説に T14・T15 を注記するか | 2026-10-28 に閉じた（注記しない。§6 の 14） |
| 9 | 段 4 の非共有（T4）の Actor の費用 | skip Network が畳み込みを持つので、Actor の forward の費用が増える。段 4 で 10M の実時間を記録し、既定に入れるかの材料にする |
| 10 | 起票時の D7 の括弧書きの正確さ | 2026-10-28 に閉じた。D7 を `[greedy]` session に改め、`[eval_panel]` は今もスキップ長が Greedy であること（T12）を本文に書いた |
| 11 | 合算で眠った・死んだユニットが増えること（§2.3）の機序 | 観察だけ（D13）。候補は更新量・表現の干渉・クリップとの結合の 3 つ（§2.3）。段 1 の遮断が無効の線に留まるか、段 3 の分割で増えるか減るかで分ける。手立ては本書に入れない。2026-10-28 の工程末に起票しないと決め、この表に残す（再開条件は §8） |
| 12 | TempoRL 有効で Munchausen の log π の clip 率が上がること（§2.3）の扱い | 観察だけ（D13）。遮断でも合算でも同じだけ上がるので合算（T1）によるものではなく、skip の間に繰り返す行動が途中の状態で方策の選ぶ行動と違うことによる見込み。確かめる計測（判断の旗で分けた clip 率）と、途中の行の Munchausen の項の扱いは、2026-10-28 の工程末に起票しないと決め、この表に残す（再開条件は §8） |

## 10. 依存と順序

- [PRD098](098_temporl_skip_policy_10prd.md) の実装に乗る。
- 着手の最初に、本書の前の実行体と設定一式を `testdata/prd103/` に snapshot する（D14）。コードと設定を変える前に取る。
- 段 1 はコードを変えずに今のビルドで流れている（2026-10-27 から 11 の bat、`run_*_a5_qb_aln_t98{stp,lw03,c30}_s1`。遮断は 10M を越え、重み 0.3 と閾値 30 は待ち）。段 1 の結果は段 3 の腕の選び方にだけ使い、実装の着手は待たない。段 1'（GridMaze 6×10）は 2026-10-28 にコードを変えずに流し、結果は実験記録にある（§7）。
- D3・D7・D8 の env の設定は単独で先に入れられる。走行中・待ちの Atari の腕には効かない（Atari の改名は群名 `53_evalg` を変えないので比較に効かず、`03` の改名は待ちの腕から新しい tag になるだけ）。
- 第三弾（[999_temporl_skip_followups](999_temporl_skip_followups_10prd.md)）とは §9 の 7 で結んだ。

## 11. 参照

- [PRD098 TempoRL 対応](098_temporl_skip_policy_10prd.md)、[PRD098 の検証資材](../../core/anet-core/testdata/prd098/README.md)
- [第三弾（TempoRL の後続）](999_temporl_skip_followups_10prd.md)
- 著者の実装: [automl/TempoRL](https://github.com/automl/TempoRL)（[run_atari_experiments.py](https://github.com/automl/TempoRL/blob/master/run_atari_experiments.py)、[run_featurized_experiments.py](https://github.com/automl/TempoRL/blob/master/run_featurized_experiments.py)）
- [TempoRL 論文解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md)
- [BTR と qbert の初期値の campaign](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md)
- [eval スケジュールの初回発火の抑止 暫定 PRD](999_eval_schedule_first_fire_10prd.md)
- [ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)、[可観測性（140）§6.x step 座標系](../design/140_observability.jp.md)
- [行動の時間尺度（wiki）](../wiki/mechanisms/temporal_abstraction.md)、[評価と比較の条件（wiki）](../wiki/mechanisms/evaluation.md)
