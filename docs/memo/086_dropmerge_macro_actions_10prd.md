# DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化）暫定 PRD

> 2026-10-03 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。
> 2026-10-23 改訂: Run 名を今のフォルダ名に合わせた（2026-10-15〜17 の改名。対応は DropMerge の workspace の改名の対応表）。中身は変えていない。
> 起点: DropMerge の NEET（合法 DROP があるのに NOOP を選び続ける病理、[CONTEXT.md](../../CONTEXT.md)）の分析で、「1 step 待つ」ことの価値が「最善の DROP を 1 step 遅らせた価値」とほぼ同じで、両者の差が推定の揺れより小さいことが分かったこと。人間のプレイヤーは物性が落ち着くのを待ってから落とす一方、上級者は敢えて落ち着く前に滑り込ませる（時差置き）。待つこと自体は悪くないので、待ちを 1 回の判断で表せる行動を Env の設定で足せるようにする。
> 関連: [085_dropmerge_eval_score_axes_10prd.md](085_dropmerge_eval_score_axes_10prd.md)（比較の物差し）、[999_smdp_transition_duration_10prd.md](999_smdp_transition_duration_10prd.md)（マクロ 1 回が進める時間を割引に反映する）、[098_temporl_skip_policy_10prd.md](098_temporl_skip_policy_10prd.md)（続ける長さを学習する）、[done/039_dropmerge_nolegal_adjudication_10prd.md](done/039_dropmerge_nolegal_adjudication_10prd.md)、[ADR 0014](../adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md)。

## 1. 目的と問題

DropMerge のアクションモードを拡張し、設定しだいで次の 2 種類のマクロ行動を使えるようにする。Agent は行動の数を ActionSpec から受け取るので、Agent 側の改修なしで試せる形にする。

- 落ち着くまで待つ: 盤面が静止するか、上限のフレーム数に達するまで NOOP を続ける。
- 繰り返し回数の離散化: 指定した行動を k 回続ける行動を、設定した k の組（例: NOOP × {4, 16}）だけ足す。

1 フレームの NOOP と各列の DROP は残す。時差置きのような細かいタイミングは、従来どおり 1 step 単位で選べる。

背景にある問題は次のとおり。

- 1 step の NOOP は盤面をほとんど変えないので、その価値は「次の step で最善手を打つ価値」を γ で 1 回割り引いたものとほぼ同じになる。最善の DROP との本来の差は (1−γ)·V ≈ 0.3%（γ = 0.997）しかない。
- QR51 100M（`run_20261002-031314_dmfb_qr`）の NEET 期（8〜30M）では、評価セッションで記録している「NOOP の UQE スコア − 最善 DROP の UQE スコア」が、NEET の前後の −0.020〜−0.026 から −0.001〜−0.002 に潰れ、NOOP が評価 step の 74〜84% で選ばれた。評価のエピソード長は 140〜190 step から 580〜1,180 step に伸びた。NEET の出ない基準 IQN 100M（`run_20261001-191232_dmfb_base`）は同じ期間に −0.021〜−0.033 のままで、NOOP の選択率は 0〜19% だった（2026-10-03 集計）。
- 待ちを 1 step ずつ判断するので、待つたびに判断の回数が増え、実時間もかかる。n-step = 1 では、待った結果の情報が 1 回の更新で 1 step ずつしか遡らない。

## 2. 現状（コードと設定で確認したこと）

- アクションモードは Move / MoveFast / Direct / DirectNoop（`core/envs/dropmerge1/src/DropMergeEnv.cpp:177-181`）。現行は `action_mode = direct_noop` で、行動 0 が NOOP、1〜58 が各列の DROP（同 `:993-995`）。
- 1 回の Step の中で物理を回すループがある（同 `:1061-1137`）。
  - 既定は 1 フレームで抜ける。
  - `use_instant_drop = true`（現行）では、落とした果物が着地するか待ち時間が尽きるまで回し続ける（同 `:1122-1124`、着地の判定は `updateDropperStatus()`）。DROP の 1 step は着地までの複数フレーム、NOOP の 1 step は 1 フレームになる。
  - `use_settle_after_drop`（現行 false）では、DROP の後や盤面が動いた後に、静止が 10 フレーム続くまで強制的に回す（同 `:1113-1121`、上限 `settle_max_steps = 500`）。待つかどうかを Agent が選べない固定の待ちである。
- 盤面の静止は `DropMergeEnv::isWorldSettled()` で判定できる（投了の裁定でも使っている、同 `:932-938`）。
- ショットクロックは、DROP 以外の行動で `steps_since_last_drop_` を 1 増やし（同 `:1018-1022`）、100 に達すると NoDropTimeout で終端する（同 `:1192-1203`）。残り時間は観測のベクトル `[4]` に `no_drop_timeout_ratio` として入る（同 `:1336-1341`）。
- 物理の 1 フレームは 1/60 秒（同 `:1064`）。
- 2026-10-03 に、QR51 + Munchausen τ0.03 100M（`run_20261003-081944_dmfb_qr_munch`）の 14M 付近を EvalPanel（target ネット + Greedy の Actor）で録画した。NoDropTimeout にならない範囲で NOOP を続けてから DROP する、長考型の NEET が出ていた。録画から切り出した 3 場面は、どれも盤面に余裕があり、静止していた（Sim Steps 1）。長考は、静止した盤面での 1 フレームの NOOP の連続で起きている。
- その 3 場面では、NOOP の平均 Q が全 DROP を上回っていた（NOOP 3.206 対 最善の DROP 3.131、2.363 対 2.303、2.104 対 2.099。EvalPanel の Q 値表の Mean 列）。NOOP の分位点の広がり（同じ表の Std 列）は 0.005〜0.007 で、DROP の 0.017〜0.045 より細かった。
- 同じ Run の評価セッション（UQE）では、NOOP の UQE スコアから最善 DROP の UQE スコアを引いた差が、0〜10M の −0.096〜−0.162 から 12〜14M に +0.024 へ振れ、NOOP が評価 step の 86% で選ばれた（評価のエピソード長 1,458 step）。22〜24M に −0.109 へ戻り、NEET を抜けた。NoDropTimeout で終わった回は 10〜15M に約 1 割だけで、大半は締切の手前で DROP していた。

## 3. 提案の骨子

1. DirectNoop を拡張したアクションモード（または DirectNoop の追加設定）で、マクロ行動を ActionSpec に足す。行動のラベルも付ける。
2. 落ち着くまで待つ行動: `isWorldSettled()` が成り立つか、上限 N フレーム（設定）に達するまで NOOP を続ける。
3. 繰り返し回数の離散化: 対象の行動と k の組を設定で指定し、「その行動を k 回続ける」行動を足す。最初の対象は NOOP を想定する。
4. マクロの実行は Env の Step の中で完結させる。途中の報酬は合計し、終端したらその時点で止める。観測は最後のフレームの後のものを返す。実行したフレーム数を記録する。
5. Agent は改修しない。マクロ 1 回は Agent の 1 step として割り引かれる（何フレーム続いても γ は 1 回）。待つことが割引の上で安くなるという帰結を、比較のときの注意として残す。進めた時間を割引に正確に反映するのは [999 SMDP 対応](999_smdp_transition_duration_10prd.md) の範囲にする。
6. 指標: エピソードあたりの Agent の step 数とフレーム数、マクロ行動ごとの選択率を出す。比較は [085](085_dropmerge_eval_score_axes_10prd.md) のスコア軸で行う。

## 4. User Stories

1. 研究者として、Agent を改修せずに「落ち着くまで待つ」行動を足し、NEET が減るか、点数が上がるかを試したい。
2. 研究者として、待つ長さの選択肢（例: 1・4・16 フレーム）を設定だけで変え、どの粒度が学習に向くかを比べたい。
3. 研究者として、時差置きのような細かいタイミングの手を、マクロを足した後も選べるようにしておきたい。
4. 研究者として、マクロで判断の回数が減ったぶん、実時間あたりの学習が速くなるかを確かめたい。

## 5. 決めること（詳細設計で詰める）

1. ショットクロックと残り時間の観測を、物理のフレームで数えるか、Agent の step で数えるか。NEET の意味を保つならフレームが自然。
2. 投了の裁定（NoLegal candidate が 60 フレーム続いたら受理）との関係。マクロの途中で裁定の条件を満たしたときの扱い。
3. 直前行動の観測（`prev_valid` / `prev_noop` / `prev_drop_x`）で、マクロをどう表すか。
4. EvalPanel で、マクロの途中のフレームを描くか。
5. spatial exploration のランダムな行動は行動の一覧から一様に選ぶので、マクロを足すとランダムで待つ割合が変わる。これを許すか、選び方を変えるか。
6. 行動の番号に依存する指標（行動 0 の NOOP の margin と選択率、`51_eval1/41,42` など）を、マクロの行動にも広げるか。
7. `ep_step` と `max_step`（現行 20000）の単位。
8. 繰り返しの対象を DROP にも広げるか（DROP は 1 回で働くので、最初は NOOP に限る案を既定にする）。
9. 既存の `use_settle_after_drop`（固定の待ち）と同時に使えるか。
10. 他の env でも使える BatchEnv の decorator として一般化するか（「落ち着くまで待つ」は DropMerge の物理の知識が要るので、本 PRD は DropMerge に閉じる案を既定にする）。
11. 静止した盤面では「落ち着くまで待つ」はその場で終わるので、§2 の長考型 NEET（静止した盤面での 1 フレームの NOOP の連続）は、マクロを足しても 1 フレームの NOOP が残る限り起こりうる。時差置き用の 1 フレームの NOOP を、どの範囲で残すか。

## 6. 対象外

- 続ける長さを Agent が学習すること（TempoRL、[999](098_temporl_skip_policy_10prd.md)）。
- マクロが進めた時間を割引に反映すること（[999 SMDP 対応](999_smdp_transition_duration_10prd.md)）。
- 次の判断まで遡る n-step（[999](999_decision_aligned_nstep_10prd.md)）。

## 7. 依存と順序

実装は単独でできる。効果を比べる実験は [085](085_dropmerge_eval_score_axes_10prd.md) の物差しが入ってから行う。

## 8. 参照

- [Sharma et al., 2017] Sahil Sharma, Aravind S. Lakshminarayanan, Balaraman Ravindran. "Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning." ICLR 2017. https://arxiv.org/abs/1702.06054（詳細解説: `reports/sharma2017_figar_paper_2026-10-03.md`）
- [Biedenkapp et al., 2021] André Biedenkapp, Raghu Rajan, Frank Hutter, Marius Lindauer. "TempoRL: Learning When to Act." ICML 2021. https://arxiv.org/abs/2106.05262（詳細解説: `reports/biedenkapp2021_temporl_paper_2026-10-03.md`）
- [Pardo et al., 2018] Fabio Pardo, Arash Tavakoli, Vitaly Levdik, Petar Kormushev. "Time Limits in Reinforcement Learning." ICML 2018. https://proceedings.mlr.press/v80/pardo18a.html
