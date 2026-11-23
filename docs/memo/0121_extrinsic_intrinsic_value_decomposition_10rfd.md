# 外的と内的の価値の分解（Q を外的報酬と内的報酬の 2 つの Head で学ぶ） PRD

> 2026-11-10 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定と決めきれない点を §8 の未決表に置いた。§5・§7・§9 は §8 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 起点: Montezuma の RND の Run（§2.1）を見た 2026-11-10 の会話。会話で出た方向は次の 3 つで、どれも決定ではない。(1) 主題は 2500 点が評価側（Greedy）で安定するかで、途中の時期の差は Run ごとの揺れの内とみなす。(2) Body を分けるか共有するかは Network の設定で決まり、Agent57 型でも Burda 型でも Head の分離になるので、2 つの型の実装の違いは小さい見込み。(3) Q の計算を分けると、Learner だけでなく Replay にも影響が出る（§4）。
> 関連: [173_intrinsic_reward_novelty_10prd.md](0173_intrinsic_reward_novelty_10rfd.md)（NGU 型の内的報酬と方策の組。本 PRD との関係は §7）、[116_rnd_lifelong_novelty_10prd.md](0116_rnd_lifelong_novelty_10rfd.md)（RND による一生の新しさ。本 PRD の内的報酬の出どころ）、[123_rnd_intrinsic_reward_scale_10prd.md](0123_rnd_intrinsic_reward_scale_10rfd.md)（RND の内的報酬の尺度）、[125_dqn_learner_update_skeleton_10prd.md](0125_dqn_learner_update_skeleton_10rfd.md)（学習器の更新の進行を 1 本に寄せた）、[127_eval_trace_intrinsic_reward_columns_10prd.md](0127_eval_trace_intrinsic_reward_columns_10rfd.md)（Evaluation Trace の内的報酬の列）、[128_dqn_agent_refactor_bundle_10prd.md](0128_dqn_agent_refactor_bundle_10rfd.md)（本 PRD より先に片づけるリファクタリング 5 件）、[098_temporl_skip_policy_10prd.md](0098_temporl_skip_policy_10rfd.md)（TempoRL）、[103_temporl_reference_alignment_10prd.md](done/0103_temporl_reference_alignment_10rfd.md)（skip の損失ごとの最適化）、[168_nstep_target_correction_10prd.md](0168_nstep_target_correction_10rfd.md)（n-step の目標値の補正）、[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md)・[ADR 0035](../adr/0035-munchausen-target-learner-local-real-space.md)・[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)・[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)・[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)・[ADR 0062](../adr/0062-networkmodel-holds-auxiliary-network-fed-by-primary-outputs-with-target-used-only-for-acting.md)・[ADR 0063](../adr/0063-replay-slot-keeps-per-step-reward-and-end-and-samples-variable-windows.md)・[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)・[ADR 0073](../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)・[ADR 0074](../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md)・[ADR 0075](../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md)、[CONTEXT.md](../../CONTEXT.md) の「価値の分解」「報酬の成分」「報酬スケーラ」「価値ストリーム」。

> 2026-11-12 改訂: グリルを行って暫定の印を外した。起票時の未決表の 21 行をすべて決め直し、確定事項（§5）へ移した（対応は §5.2）。節の番号は付け直した（起票の行にある節の番号は当時のもの）。起票のあとに変わった前提を反映した。変わった前提は、β と Munchausen の切り分けが済んだこと（§2.1）、PRD123 で既定の RND の中身が変わったこと、PRD125 で 3 つの学習器の更新が 1 本になったこと、PRD127 が Evaluation Trace の内的報酬の列を持つこと、終端の遷移で Replay が返す次観測が次のエピソードの最初の観測であること（§4）である。起票時の既定から変えた主な点は、内的報酬を割引収益の標準偏差で割る正規化を範囲に入れたこと（D9）、Munchausen と併用できるようにしたこと（D10）、損失の重みと優先度の重みの設定キーを足したこと（D4・D12）、外的の Head の bootstrap のモードを足したこと（D5）、終端で切らない形を近似ではなく次のエピソードの最初の観測からの bootstrap にしたこと（D7）、効果を確かめる Run の組み立てを Atari の枠へ移したこと（D19）である。本文の「葉」を「設定キー」に改めた。
> 2026-11-13 改訂: 起草のあと、別モデル（Opus）の検討を受けた。結論への反対 1 件と、前提が変わった 2 件を開発者に聞いて決め直し、理由や前提の不足は本文を補った。変えた点は次のとおり。(1) 分解が有効のとき、Network は `q`・`q_dist` を出さず、DQN 側の 1 つの関数が合わせる形にした（D24。「`q` は外的のまま」を取り下げた）。(2) 損失ごとに勾配の clip と Adam の状態を分けるモードを足した（D25。skip の既存の経路を流用できると分かり、先送りをやめた）。(3) 報酬スケーラの状態を、外的報酬用も内的報酬用も checkpoint に保存する形にした（D16。内的報酬用だけ保存する形をやめた）。(4) 細部として、割引収益の種類と自動の倍率の同時指定を ERROR にし（D9）、窓の収益の成分ごとの対応を先送りにし（D8）、Evaluation Trace の鍵の名前を改め、分解が有効のときの既存のメトリクスの値を定めた（D20）。Actor の hint を、モードに依らない 5 列にし（D12、ADR 0075）、足すメトリクスの tag を、既定では選ばれない組に置いた（D19・D20）。(5) 事実と文言として、done と打ち切りが同時に真の遷移の扱いを足し（D7）、「game over」を「終端（done）」に改め、TBO のとき合わせた値を h 空間で渡すと定め（D14）、収益を Learner が作る形を退けた理由を書き直した（D8、ADR 0073）。
> 2026-11-13 改訂（2 回目）: 確定のあと、開発者の求めで開き直し、内的の Head が行動の選択をどれだけ変えたかを見る診断を足した。行動は Q_外 ＋ β·Q_内 で選ぶので、β の釣り合いは分解しても行動の選択に残るが、それまでの診断（選んだ行動の値に占める内的の割合）では、選択への効き方を読めなかったためである。起草のあと、別モデル（Opus）の検討を受け、指標の組を開発者に聞き直した（差に占める内的の割合をやめ、Head ごとの行動間の差を 2 つ足した）。lane ごとの値と lane の平均を別の鍵にし、値を実空間で取り、評価の名前と tag の名前の決まりを足した。足したのは、ゴール G8、非ゴール 2 行（§1.4）、機構の 1 行（§1.5）、事例（§2.5）、用語 1 行（§3）、現状 3 項目とトピック別まとめの 1 行（§4）、確定事項 D26〜D28、選ばなかった形 7 行（§5.1）、定義（§6.8）と §6.7・§8.3 の 1 項目ずつ、指標 6〜9（§8.4）、クラス一覧（dqn::Actor の行の責務と、足した DQNActionInfo の行）、移行対象（§8.7 の検索語、テスト、設定、設計文書）、読み方の材料（§9.1）、受入 22・23 と人が確かめる項（§9.2）である。D1〜D25 の決定は変えていない。D20 の「5 種」を「1〜5」に改め、D22 の費用の列挙に、足した診断の 2 項目（Actor の step ごとの診断、評価の Actor の session の足し込み）と、依存の D26・D27 を加えた。

> 2026-11-15 改訂: 開発者の選択により、§8.7 の設定比較を現用17構成の変更前後比較へ改めた。PRD072のgoldenは固定commitの履歴で、廃止済みのプロファイルを含むため保持する。既存キーの全文字列値とscalar定義を一致させ、許す差を本PRDの追加既定値だけに限定する。§4の基準の説明も履歴であると明記した。学習・保存・診断の契約は変えていない。

> 2026-11-15 改訂（2 回目）: §8.1の設定判定順と、使われない射影設定値の通知を共通処理の責務を保って満たすため、§8.7のコード対象にnn.hpp・nn_impl.cpp・nn_modules.cppを加えた。nn.hppは初期化方式の共通検査と使われない射影設定値の通知の宣言、nn_impl.cppは共通初期化検査、nn_modules.cppは射影設定値の通知を持つ。実際の所在への追随で、設定値・学習・保存・診断の契約は変えていない。

## 1. 目的と問題

### 1.1 問題

DQN 系の Agent は、内的報酬（RND の新しさ）を外的報酬に足した 1 つの報酬で、1 つの Q を学ぶ（§4）。Montezuma の RND の Run（§2.1）と今の作りから、次の 3 つが問題になる。

1. **外的と内的の大きさの釣り合いに、結果が大きく左右される。** β 0.01・Munchausen 無効の Run は、Greedy の 2500 点が 20.2〜29.2M に 99/160（seed 1）と 81/160（seed 2）だった。β 0.1 の Run は Munchausen の有無によらず 0 本だった。seed だけを変えた Run で、内的報酬の平均が 8M 以降に 1.6〜3.2 倍違った。1 本の Q では、外的と内的の大きさ・分散の違いを、NN が 1 つの出力で吸収するしかない。
2. **内的だけ、割引率や終端の扱いを変えられない。** Burda らは、内的の収益を game over で切らず、内的の割引率を外的より小さくして成績を上げた（§2.2）。1 本の Q では試せない。
3. **Head ごとの価値を見られず、内的を外した行動を評価できない。** 2500 点を取ったあと、外的報酬の残らない局面で Q が行動の間でほぼ同じ値になり、Greedy が動かなくなった（§2.1）。この局面で、外的の価値と内的の価値がそれぞれどうなっているかを分けて見る手段が無い。

### 1.2 ゴール

- **G1**: DQN 系 Agent が、Q を外的の Head と内的の Head の 2 つで学べる（Q = Q_外 ＋ β·Q_内）。既定は無効で、無効のときは今と同じ学習系列になる（同 seed・決定的モードの metrics checksum が一致する）。
- **G2**: 内的の Head の割引率と終端の扱いを、外的と同じにする（Agent57 型）か、別にする（Burda 型: 割引率を別に持ち、終端で切らない）かを、設定のモードで選べる。
- **G3**: 外的報酬と内的報酬を、取り込みから Replay・PER の優先度・Learner の目標値まで別々に運ぶ。Replay の契約は「報酬の成分」として汎用に書き、内的報酬という名前を Replay に持ち込まない。
- **G4**: 内的報酬を、その割引収益の標準偏差で割って正規化できる（Burda らの形）。checkpoint 引き継ぎのあとも正規化の統計が続く。
- **G5**: 2 つの Head の Q を、診断（Evaluation Trace の列とメトリクス）で別々に見られる。Actor ごとに、行動を選ぶときの β を変えられる（評価で β=0 にできる）。
- **G6**: [173](0173_intrinsic_reward_novelty_10rfd.md)（NGU 型と方策の組）が、この分解の上に方策の番号の入力（UVFA）を重ねられる形にする（§7）。
- **G7**: 効果を §1.3 の 2 つの仮説で確かめられる。
- **G8**: 価値の分解が有効のとき、内的の Head が行動の選択をどれだけ変えたかを、学習の Actor・評価の Actor・Evaluation Trace で読める。仮説 A の読みに使える（§9.1）。

### 1.3 仮説・比べ方・終わり方

| | 仮説 | 比べ方 |
|---|---|---|
| A（大きさ） | 分解すると、β を変えたときの成績の変わり方が小さくなる（Agent57 §4.2） | 同じ内的報酬の定義で、1 本の Q と分解した Q を、β 2 点以上で比べる。分解した側で、β による評価の差が小さければ支持 |
| B（内的の形） | 内的を終端で切らず、割引率を別にすると、探索する部屋の数と、2500 点のあとの平らな局面が変わる（Burda ら §3.2・§3.3） | 分解した Q の中で、内的を外的と同じ扱いにした群と、終端で切らず割引率を別にした群を比べる |

- **終わり方**: 効果が出なくても、仕組みは残す。既定は無効で、173 の土台になる（§7）。効果が出たとき、Atari の RND の構成の既定に入れるかは、Atari の枠で決める（D19）。
- 群の組み立て（構成、β の値、seed の数、長さ）は本 PRD で決めない。材料は §9.1 に書く。

### 1.4 非ゴール（退けた案と、本当の制約か）

| 退けた案 | 理由 | 本当の制約か |
|---|---|---|
| エピソード内の新しさ（NGU・E3B）、方策の組（方策ごとの β_j・γ_j と、方策の番号の入力） | 別の関心として 173 で扱う。本 PRD は方策 1 つ（β 1 つ）のまま扱う | 制約ではない（173 が本 PRD の上に重ねる。§7） |
| Retrace などの方策オフの補正 | 別の関心で、173 と [168](0168_nstep_target_correction_10rfd.md) の範囲 | 制約ではない |
| n-step の和に、次のエピソードの報酬まで足すこと | Replay の窓がエピソードをまたぐ契約の変更になる別の関心。終端で切らない形の bootstrap は、窓を変えずに作れる（D7） | 制約ではない |
| Agent57 の方策の選択（バンディット） | 別の関心で、173 の範囲 | 制約ではない |
| Atari のゲームの仕様を変えること、RAM の値を報酬にすること | 開発者の裁定（173 の非ゴールと同じ） | 本当の制約 |
| 鍵の直後・剣のあとで Greedy が止まる型（§2.1）を直すこと | 外的の価値が大きい局面で行動間の差が小さい、という別の関心（割引率、命の扱い）。価値の分解で直る根拠が無い。Head ごとの Trace で、止まる局面の向きがどちらの Head から来ているかは読める（G5） | 制約ではない |
| 価値の分解と TempoRL を同時に有効にすること | 先送り（実験の結果が要る）。§5.1 | 制約ではない |
| 観測の正規化の統計を checkpoint に保存すること | 報酬スケーラの状態の保存（D16）と同じ場所の未対応だが、本 PRD の機構が要るのは報酬スケーラだけである。別の関心 | 制約ではない |
| Atari.txt の Run プロファイルと bat の群の組み立て | Run の条件に効く値は、実験の段取りと一緒に Atari の枠で決める（D19） | 制約ではない |
| RainbowAgent への対応 | RainbowAgent は内的報酬を持たない（§4） | 制約ではない |
| 行動の選び方を変えること（行動間の差を広げる学習則、Head ごとに差の大きさをそろえる正規化） | 別の関心。足す診断（D26）の結果を見てから考える。作る条件は §5.1 | 制約ではない |
| QValuePanel に Head ごとの Q を出すこと | 別の関心（Runner の画面）。Head ごとの値と、足す診断は、Evaluation Trace の列で読める（D20、D27） | 制約ではない |

### 1.5 機構とゴールの対応

| 機構 | ゴールへの寄与 |
|---|---|
| 価値の分解（2 つの Head と、Head ごとの目標値・損失） | G1、問題 1、仮説 A |
| 合わせた Q を作る関数 | G1。行動の方策や診断などの読み手を変えずに、合わせた Q を届ける（D24） |
| 内的の割引率と終端のモード | G2、問題 2、仮説 B |
| Replay の報酬の成分 | G3。分解の前提で、173 が要る「外的と内的を分けて保存する契約」も満たす |
| 割引収益の標準偏差で割る報酬スケーラと、内的報酬用の報酬スケーラ | G4。内的の Head が学ぶ値の桁を、割引率に依らずそろえる（§6.6） |
| 報酬スケーラの状態の保存 | G4。checkpoint 引き継ぎのあとも正規化の統計が続く（D16） |
| 損失の重み、優先度の重み | G1。共有の Body に流れる勾配と、抽選の釣り合いを調整する（§6.4） |
| 損失ごとに勾配の clip と Adam の状態を分けるモード | G1、問題 1。外的と内的の大きさが違っても、共有の Body の更新が片方に埋もれない（D25） |
| 外的の Head の bootstrap のモード | G5・G6。内的報酬を使わない点数だけの方策を、173 を待たずに評価できる |
| Actor ごとの行動の β | G5、問題 3 |
| Head ごとの診断（Trace の列、メトリクス） | G5、問題 3 |
| 行動の選択への効き方の診断（不一致、譲った外的の価値、外的の差、内的の差） | G8、問題 1、仮説 A。β の釣り合いが、学習ではなく行動の選択にどう効いたかを読む（D26） |

## 2. 背景

### 2.1 Montezuma の RND の Run（2026-11-08〜12）

5 本は同じ構成（Run 名に `mz_tg_` を持つ群）から、β・Munchausen・Double DQN・seed だけを変えた。内的報酬は、PRD123 より前の実行体の既定の式（`clipped`・`whiten`。1 step の内的報酬は β × (α − 1) で、α − 1 は 0〜4）で、外的報酬に足して 1 本の Q で学ぶ。割引率は 0.997、n-step は 3、分布の表し方は IQN、Dueling は有効、勾配の clip の上限は 10 である。Greedy は `53_evalg`（16 ゲーム、約 1M ごと）の値である。

| Run | β | Munchausen | Double DQN | seed | Greedy の 2500 点（20.2〜29.2M の 160 ゲーム） | ε 0.01 の評価 |
|---|---|---|---|---|---|---|
| `run_20261108-232006_mz_tg_rnd_s1` | 0.1 | 有効 | 無効 | 1 | 0 本（29.2M まで）。おおむね 400 点 | 10.2〜28.21M に 400〜490 |
| `run_20261110-120903_mz_tg_nmdd_b01_s1` | 0.1 | 無効 | 有効 | 1 | 0 本（22.2M まで。22.63M で止めた）。16.2〜20.2M はおおむね 400 点 | 12.21〜22.21M に 400〜450 |
| `run_20261110-003002_mz_tg_nmdd_b001_s1`（以下 b001） | 0.01 | 無効 | 有効 | 1 | 99 本。14.2M に初めて出て、1 回あたり 0/16〜16/16 で揺れた | 14.21〜28.21M に 74/80 が 2500 点 |
| `run_20261111-123702_mz_tg_nmdd_b001_s2` | 0.01 | 無効 | 有効 | 2 | 81 本。18.2M に初めて出て、21.2M は 0/16、28.2M は 15/16 | 20.21〜28.21M に 32/50 が 2500 点 |
| `run_20261110-233727_mz_tg_rnd_b001_s1` | 0.01 | 有効 | 無効 | 1 | 5 本。500 点（剣を取って敵を倒さない）が 106 本 | 20.21〜28.21M に 7/50 が 2500 点 |

学習側では、b001 が 16〜30M の 2M 窓で、終わったゲームの 81〜93% が 2500 点だった。seed 2 の Run は 16〜24M が 28〜45%、24〜30M が 72〜87% だった。

出典は実験記録 [2026-11-08_montezuma-rnd-taiga-aligned.md](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md) の「現時点のまとめ」1〜3・8〜10 と、探索ブロック 02〜04・06・09・10 である。

**seed だけの違い。** seed 2 の Run は、b001 と同じ実行体・同じ設定で seed だけが違う。誤差の標準偏差は b001 の 0.55〜0.56 倍、内的報酬の平均は 8M 以降の 4M 窓で 1.6〜3.2 倍だった。

**b001 の 2500 点のあと。** EvalPanel で見ると、2500 点を取ったあと部屋 1 の開始の足場で UP を押し続け、27,000 step の打ち切りまで動かなかった。このときの Q は 18 行動とも 0.055〜0.059（差 0.004 で、τ の標準偏差 0.009〜0.010 より小さい）、V_s は 0.056〜0.063 だった。29.2M の Greedy の 16 ゲームは、すべて 2500 点・打ち切りで終わった。学習側は、16M 以降の 2M 窓ごとの打ち切りが、終わったゲームの 4〜11% だった。

**2500 点の前にも止まる。** seed 2 の Run の 29.94M の重みを EvalPanel（Greedy）で見ると、鍵を取った直後に Q が 18 行動とも 2.692〜2.698 で、命を失う行動を選んだ。剣を取ったあとは梯子の下で DOWN を押し続け、Q は 1.654〜1.663（DOWN 1.663、UP 1.659）だった。外的の価値が大きい局面で、行動間の差だけが小さい型である。

**b001 の内的報酬の大きさ。** 12M 以降の窓（4M ごと。最後は 28〜30M）の平均で、1 step の内的報酬は 2.3〜3.6e-4（β を掛ける前は 0.023〜0.036）、正の値の割合は 13〜17% だった（実験記録の探索ブロック 06 の表）。

**見立て（未検証）。** 2500 点のあとの局面では、外的報酬が近くに残らないので、Q は内的報酬（1 step 3e-4 を約 330 step 積んだ 0.1 前後）だけでほぼ一様になり、行動を選ぶ向きが無くなる。鍵の直後・剣のあとの型は、この見立てでは説明できない（§1.4）。

### 2.2 文献

- **Taiga ら（ICLR 2020、[arXiv 2109.11052](https://arxiv.org/abs/2109.11052)）。** Rainbow＋RND は Montezuma の 5 seed すべてが 90M フレームまでに 4,400 点を超えた（図 7 を目で読んだ値）。NoisyNets の 5 seed は 35〜40M フレームから 200M フレームまで 2500 点にとどまった。同じ RND でも 2500 点は天井ではない。
- **NGU（[Badia ら 2020a](https://arxiv.org/abs/2002.06038)）。** 方策の番号 j（β_j と γ_j の組）を one-hot で NN に入力し、j ごとに外的報酬と β_j 倍の内的報酬を足した報酬で、1 本の価値を学ぶ（[Agent57](https://arxiv.org/abs/2003.13350) §2 の要約による）。Agent57 §2 は、外的と内的の報酬の大きさや疎らさが違うと NGU が不安定になると書いている。
- **Agent57（Badia ら 2020b）§3.1。** Q(x, a, j) = Q(x, a, j; θ_e) ＋ β_j·Q(x, a, j; θ_i) に分ける。θ_e と θ_i は同じ形の別々の NN で、どちらも j を入力に取る。2 つは外的報酬と内的報酬で別々に学び、bootstrap に使う方策は、合わせた Q の argmax で共通にする。単純な勾配降下では、合わせた報酬で 1 本を学ぶのと目的が等価になる（付録 B）。分ける理由は、各 NN がそれぞれの報酬の大きさと分散に合わせられることと、optimizer の状態を分けられること。
- **Agent57 §4.2。** 分けると、内的報酬の重みの広い範囲で学習が安定した。外すと成績が 20% 以上落ちた。
- **Burda ら（[RND、ICLR 2019](https://arxiv.org/abs/1810.12894)）§2.3。** 内的の収益を game over で切らない（非エピソード）。外的は game over で切る。性質の違う 2 つを組むために、価値の Head を 2 つ（V_E と V_I）にし、和を価値にする。
- **Burda ら §3.2（図 4）。** 内的を非エピソードにすると、探索した部屋の数が増えた。両方エピソードのままなら、Head を 2 つにしても 1 つより良くならなかった。
- **Burda ら §3.3（図 5）。** 外的の割引率 0.999・内的 0.99 にすると成績が大きく上がった。内的を 0.999 にすると下がった。
- **Burda らの内的報酬の正規化。** 誤差を、内的報酬の割引和（γ 0.99、エピソードで切らない）の値の、累積の標準偏差で割る。平均は引かない。出典は公開の PyTorch 実装で、[123](0123_rnd_intrinsic_reward_scale_10rfd.md) §2.4 が確かめた。

### 2.3 2 つの型

| | Agent57 型 | Burda 型 |
|---|---|---|
| 割引率 | 外的と内的で同じ | 内的を別に持つ（Burda らは外的 0.999・内的 0.99） |
| 終端 | 両方とも終端で切る | 内的は切らない |
| 内的報酬の正規化 | 誤差の標準偏差で割って切った倍率（NGU の形） | 割引収益の標準偏差で割る |
| 目的 | 合わせた報酬で 1 本を学ぶのと同じ。変わるのは学び方（それぞれの大きさに合わせられる） | 目的そのものが変わる |
| 本 PRD での指定 | 割引率と終端のモードが既定（D6・D7） | 割引率と終端のモードを変え、内的報酬用の報酬スケーラを使う（D6・D7・D9） |

### 2.4 価値を分けた手法の優先度（2026-11-12 に確認）

| 事例 | 優先度の作り方 |
|---|---|
| NGU、今の anet（1 本の Q に内的報酬を足す） | 足した報酬での TD 誤差 |
| Agent57 の論文 | 系列の TD 誤差の max と mean の混合とだけ書き、価値を分けたあとにどの誤差を使うかは書いていない（付録 E） |
| Agent57 の公開の再実装（[deep_rl_zoo](https://github.com/michaelnny/deep_rl_zoo) の `agent57/agent.py`。DeepMind の実装ではない） | Head ごとに優先度を作り、0.8 × 外的 ＋ 0.2 × 内的 で足す。損失は単純な和 |
| anet の skip | 主の誤差の絶対値 ＋ λ × skip の誤差の絶対値。既定の λ は 0（`Learner::CombinePriorityErrors`） |

### 2.5 行動間の差と、選ぶ行動の不一致の事例（2026-11-13 に確認）

| 事例 | 中身 |
|---|---|
| Bellemare ら（AAAI 2016、[arXiv 1512.04860](https://arxiv.org/abs/1512.04860)） | 行動間の差（action gap）を、最善の行動と次善の行動の価値の差で定義する。差が大きいほど、近似の誤差で greedy の行動が変わりにくい |
| anet の `q_gap` | Learner が、学習のバッチで、1 位と 2 位の行動の Q の差の平均を出す |
| anet の `iqn_uqe_full_q_argmax_disagreement` | 方策が、行動を選ぶたびに、2 つのスコア（UQE のスコアと full Q）で 1 位の行動が違う割合を出す |
| 外的と内的に分けた Head で、選ぶ行動が違う割合を測った事例 | 検索 2 回では見つからなかった（無いと確かめたわけではない） |

## 3. 用語

| 語 | 意味 |
|---|---|
| 価値の分解 | Q を外的の Head と内的の Head の和で表し、それぞれを別の報酬で学ぶこと。[CONTEXT.md](../../CONTEXT.md) の「価値ストリーム」（Dueling の V の流れ）とは別のもので、本 PRD ではストリームの語を使わない |
| 外的の Head・内的の Head | 外的報酬で学ぶ Q の出力と、内的報酬で学ぶ Q の出力。Head・Body は訳さない |
| 合成 Head | 外的の Head と内的の Head を束ね、1 つの Network の Head として 2 組の出力を出すもの |
| 合わせた Q | Q_外 ＋ β·Q_内。行動の選択と bootstrap の行動の選択に使う。β は使う側（Actor・Learner）が持つ |
| 報酬の成分 | Replay が 1 step ごとに別々に記録する報酬の並び。Replay の汎用の語として使い、内的報酬の意味を持たせない |
| 終端（done） | Env が done を返した遷移。game over のときのほか、命を失うと done になる構成（`episodic_life`）では、命を失ったときも当たる。打ち切り（truncated）は含まない |
| 行動間の差 | ある局面での、1 位の行動と 2 位の行動の Q の差（§2.5） |

既存の語では言い換えられない。「価値ストリーム」は Dueling の V の流れに限った語で、報酬の種類で分けた出力を表さない。「報酬スケーラ」は報酬の大きさを変える部品で、報酬を複数の並びで持つことを表さない。「価値の分解」と「報酬の成分」を [CONTEXT.md](../../CONTEXT.md) に足す（D21）。「行動間の差」は指標の定義に使う語で、用語集には足さない。

## 4. 現状（2026-11-12〜13 にコードで確認）

同じファイルを触る作業が並行しており、本 PRD の実装は §7.3 の課題 5 件の直し（[128](0128_dqn_agent_refactor_bundle_10rfd.md)）のあとに行う（D18）。行番号がずれるファイル（default_dqn_agent・dqn_based_agent・rl）は関数の名前で示す。行番号を書いたものも、実装時は近くの記述で探し直す。

- **取り込み**: `DefaultDQNAgent::UpdateFromBatch` が、報酬スケーラを通した外的報酬に、内的報酬 β × (α − 1) を足して 1 本の報酬にし、内側の Learner へ渡す（[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp)）。`next_state` を採点するとき、`intrinsic.use_done_mask` が真なら done の lane の内的報酬を 0 にする。内的報酬を持つのは DefaultDQNAgent だけである。
- **受け渡しの形**: `BatchExperience` の報酬は 1 本（[rl.hpp](../../src/core/base/include/anet/rl.hpp)）。
- **Replay の記録**: Push は割引率を 1 つ受け取る（[replay_buffer.hpp:121](../../src/core/base/include/anet/replay_buffer.hpp)）。1 step ごとに報酬 1 つを待ち行列と step の記録へ入れる（[replay_buffer_impl.cpp:1636-1642](../../src/core/base/src/replay_buffer_impl.cpp)、[replay_buffer_impl.hpp:236](../../src/core/base/src/replay_buffer_impl.hpp)）。
- **n-step の収益**: Push の時点で、1 つの割引率で逆順に足して slot に 1 つ保存する（[replay_buffer_impl.cpp:327-353](../../src/core/base/src/replay_buffer_impl.cpp)）。収益の計算は終端で切れる。割引率が schedule のときは、抽選のあとに保存済みの step の報酬から作り直す（[ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)）。
- **次観測**: 抽選で返る次観測は、同じ lane の「実際に進んだ step 数」だけ先の slot の観測である（[replay_buffer_impl.cpp:1447](../../src/core/base/src/replay_buffer_impl.cpp)）。打ち切りのときだけ、打ち切り時点の観測を別の slot に保存する（[同 1656](../../src/core/base/src/replay_buffer_impl.cpp)）。終端のときは終端の観測を保存しないので、返る次観測は、同じ lane の次の slot の観測になる。game over で終端になる構成では**次のエピソードの最初の観測**で、命を失うと終端になる構成では同じゲームの続きの観測である。この slot は履歴の開始の印を持ち、frame stack は、その観測を標本として引くときと同じ形に組まれる。今は終端のマスクで使われていない。遷移が抽選できるようになるのは、次観測の slot が書かれたあとである（[同 408](../../src/core/base/src/replay_buffer_impl.cpp)）。
- **終端と打ち切りが同時に真の遷移**: 打ち切りの slot が挿入され、遷移は終端として保存される。返る次観測は打ち切り時点の観測で、この観測から始まる遷移は無い（学習されない）。
- **標本**: n-step の収益、窓の収益、抽選時の step の報酬は、どれも 1 本（[rl.hpp](../../src/core/base/include/anet/rl.hpp) の `ExperienceSamples`）。
- **PER の優先度**: 標本ごとの TD 誤差の絶対値。skip が有効なら、設定した重みで skip の誤差の絶対値を足す（`Learner::CombinePriorityErrors`）。初期優先度の見積もりの入力は、確定済みの n-step の収益 1 つと、bootstrap の割引と終端（[replay_buffer.hpp:41](../../src/core/base/include/anet/replay_buffer.hpp)）。Actor の hint は Q_sa・V_s・Munchausen項の 3 列を 1 つに詰めて運ぶ（[ADR 0012](../adr/0012-replay-initial-priority-hint-completion.md)）。列の数は Munchausen の有無に依らず 3 つで、無効のとき Munchausen項の列は 0 である（[ADR 0036](../adr/0036-actor-q-hint-three-columns-munchausen.md)、`kActorQHintColumnCount`）。hint を使うのは、初期優先度を Actor の値から見積もる構成（`learner.per_initial_priority_mode = actor_approx`）だけで、出荷の設定では、`Atari.txt` の `run.@aainit` と、`DropMerge.txt` の `run.@iqn32_actor_approx` が指定する。どちらも、既定の Run プロファイルの連鎖には入っていない。
- **Learner**: 1 回の更新の進行は、基底の `dqn::Learner::UpdateFromSamples` に 1 本である（[dqn_based_agent.hpp](../../src/core/base/src/dqn_based_agent.hpp)。PRD125）。分布の表し方ごとの違いは、目標値（`MakeTarget`）と要素誤差（`ComputeElementError`）の上書きである。目標値の式「収益 ＋ γ^n ×（終端でない）× bootstrap」は `MakeBellmanTarget` の 1 か所にある。
- **`q`・`q_dist` の読み手**: forward の結果の `q`・`q_dist` を直接読む箇所が、[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp) に 17 か所ある。行動の方策 3 種、bootstrap の行動の選択、Munchausen の目標値、NoisyNet の診断、policy churn の probe、学習のメトリクス、Actor の hint である。TBO が有効のとき、これらは h で圧縮した値の `q` を前提にしている。
- **勾配の clip と、損失ごとの最適化**: clip は Network ごとに取る（`Learner::Optimize`）。skip の `learner.skip.feature_gradient = split`（[103](done/0103_temporl_reference_alignment_10rfd.md)）は、同じ forward の結果に対して backward を損失ごとに取り、損失ごとに clip し、primary の重みに損失ごとの Adam の状態を持つ。checkpoint の optimizer の状態に、役割が 1 つ増える。
- **Head と Body**: Head は出力層だけを持つ。Dueling の Head が持つのは最終の Linear 2 本と合成で（[dqn_based_heads.cpp:229](../../src/algo/dqn/src/dqn_based_heads.cpp)）、V と A の隠れ層は Body の branch である（[Atari.txt:1281-1285](../../apps/runner/config/Atari.txt)）。Network は Head を 1 つだけ持つ（[nn.hpp:188-193](../../src/core/base/include/anet/nn.hpp)）。`NetworkModel` は、primary の Network のほかに、skip 用の補助 Network を名指しで持つ。
- **報酬スケーラ**: 定数倍の種類と、1 step の報酬の標準偏差で割る動的な種類がある（[scaler_impl.hpp:182-215](../../src/core/base/src/scaler_impl.hpp)）。lane ごとの割引和は持たない。統計を更新せずに変換する口がある（[scaler.hpp:89](../../src/core/base/include/anet/scaler.hpp)）。構築のとき、step 0 の割引率を受け取る。自動の倍率（`use_auto_post_scale`）は、目標の Q の標準偏差に (1 − γ) を掛けた値である（[scaler_impl.cpp:884-902](../../src/core/base/src/scaler_impl.cpp)）。使っているのは外的報酬だけである。
- **報酬スケーラの状態と checkpoint**: 報酬スケーラの状態と、観測の正規化の統計は、checkpoint に保存されていない（`DefaultDQNAgent::Save` に未対応の印がある）。動的な報酬スケーラを使う構成では、checkpoint 引き継ぎのたびに統計が 0 から始まる。PRD072の固定commitにおける解決済み設定の履歴（`core/anet-core/testdata/prd072/baseline/`）では、動的な種類を使うのは LunarLander・GridMaze・CartPole の既定で、Atari と DropMerge は定数の倍率である。これは現用設定の比較基準ではなく、本PRDの検証は§8.7の変更前後比較による。
- **Trace**: 評価の Trace の `q_sa`・`state_value` の列は、Agent の step の診断から読む（[common.txt:176-177](../../apps/runner/config/common.txt)、[ADR 0057](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md)・[ADR 0058](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md)）。1 step の内的報酬の列は [127](0127_eval_trace_intrinsic_reward_columns_10rfd.md) が足す。127 は、`intrinsic.` で始まる鍵を、RND の推定器の写しを使う経路で計算する（127 の D3〜D5）。
- **方策の診断と、行動の情報**: 方策は、行動を選ぶのに使った forward の結果から診断を作り、行動の情報に載せる。追加の forward はしない。値は Actor と同じ device に置き、読むときだけ CPU へ移す（[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp) の `UQEActionPolicy::SelectAction`）。購読の有無に依らず、毎 step 作る。行動の情報の鍵には、lane の平均を返すもの（`skip_decision_rate`・`iqn_uqe_full_q_argmax_disagreement` など。lane の指定を使わない）と、lane ごとの値を返すもの（`skip_decision`・`skip_length`。lane の指定が要り、範囲の外なら ERROR）がある。学習の Actor のメトリクスは、前者を `@train $action_info` で読む。Evaluation Trace は、行ごとに lane を指定して読むので（[metrics.cpp](../../src/core/base/src/metrics.cpp) の、Trace の行を作る処理）、列にするのは後者である（[Atari.txt](../../apps/runner/config/Atari.txt) の `run.@skip` の列）。評価の lane が 2 つ以上の構成がある（[Atari.txt](../../apps/runner/config/Atari.txt) の `eval_batch_size`）。
- **評価の Actor の指標**: 評価の Actor の値は、Actor が session のあいだ足し込み、session の終わりに `$eval.[<名前>] $actor @session_end` で読む（skip の `skip_session_decision_rate`。[103](done/0103_temporl_reference_alignment_10rfd.md) D8）。足し込みは Actor と同じ device で行い、session の始まりで 0 に戻す（`Actor::ResetSkipSession`）。数えた数が 0 の session で読むと、NaN を返す。評価の名前は、`[eval]`・`[eval_target]`・`[eval_panel]` がどの Env にもあり（[common.txt](../../apps/runner/config/common.txt)）、`[greedy]` などは Atari の設定だけにある。
- **行動間の差**: Learner は、行動が 2 つ以上のとき、学習のバッチで 1 位と 2 位の Q の差を `q_gap` として出す。Actor の側に、行動間の差を出す鍵は無い。

**触る既存部品と、トピック別まとめに書かれた選ばなかった形・変えるときの条件**

| トピック別まとめの項目 | 本 PRD との関係 |
|---|---|
| 「1 つの Network に skip の branch と射影を足す形 — 不採用」（[324 DQN](../wiki/categories/324_dqn.md)。出典 [098](0098_temporl_skip_policy_10rfd.md) §4 D3、ADR 0062） | 本 PRD は、内的の Head を同じ Network の合成 Head に置く（D2）。ADR 0062 が退けた形と同じ形なので、開き直す条件は「別の題材での採用による前提の変化」に当たる。退けた理由 3 つとの関係は D2 に書く |
| 「特徴空間・行動価値・スキップ価値を別 Network にする 3 分割 — 不採用」（324。出典 098 §4 D3、ADR 0062） | 別 Network にすると checkpoint の形と、Network 1 つを前提にする診断・可視化へ波及する、という理由。D2 の根拠の 1 つ。開き直さない |
| 「γ の schedule で常に抽選時に収益を作り直す形 — 不採用」（[318 RL 共通基盤](../wiki/categories/318_rl_core.md)。出典 ADR 0065） | 定数の割引率では Push 時の保存値の経路を保つ。D8 は、成分ごとにこの規則をそのまま当てる |
| 「初期優先度のヒントを TensorDict で任意の named tensor として運ぶ形 — 不採用」（318。出典 ADR 0012） | hint は 1 つに詰めて運ぶ。hint の値が増えても、この形を保つ（D12） |
| 成立の経緯: 「Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする」（324。出典 ADR 0036） | 列の数をモードで変えない判断を保ち、列を 5 つに広げる（D12）。ADR 0036 の「常時 3 列」が改まるので、ADR 0075 を起こし、ADR 0036 の先頭に注記を足す（D21） |
| 変えるときの条件: 「スキップが無効のとき、forward の回数・順序、RNG の消費、loss、優先度、checkpoint の形を今と同一に保ち、metrics checksum の一致で確かめる」（324。出典 098 §5.9・§7） | 価値の分解が無効のときも同じ形で確かめる（D17、§9.2 の 1） |
| 変えるときの条件: 「slot に 1 step の記録を足しても、sampling 分布・item key・PER の completion と eviction 統計の ready 基準・Push と Extract の lock・index の昇順を変えない」（318。出典 098） | 報酬の成分を足すときも守る（D8、§9.2 の 2） |
| 「評価側の skip 指標を評価 slot の action_info の step 値で取る形 — 不採用」（324。出典 103 §4 D8） | 評価の step の軸に載り、学習の進みと並べられない、という理由。評価の Actor の診断は、session の累積を終わりに読む形にする（D27） |

## 5. 確定事項

2026-11-12 のグリルと、2026-11-13 の別モデルの検討のあとの質疑で決めた。D2〜D5・D9〜D13・D16・D18・D19・D24・D25 は問いへの答え、D1・D6〜D8・D14・D17・D23 は規約・現行コード・ほかの決定から決まるもの、D15・D20〜D22 は推奨の既定を採った細部である。D12 のうち hint の列の数は、ADR 0036 の判断から決まるものである。D26〜D28 は、確定のあとの 2026-11-13 の改訂（2 回目）で足した。D26・D27 は問いへの答え、D28 は推奨の既定を採った細部である。D26 の指標の組と、D27 の鍵の分け方は、別モデルの検討のあとに決め直した。設定キーは、どれも `DefaultDQNAgent` の下である。

| # | 決定 | 理由と依存 |
|---|---|---|
| D1 | **対象と有効化。** 対象は DefaultDQNAgent の TD・QR・IQN のすべての学習器。有効化は `intrinsic.value_mode`（`combined`｜`decomposed`）で、既定は `combined`（今の 1 本の Q） | 更新の進行が基底の 1 本になり、分布ごとの違いは目標値と要素誤差の上書きだけなので（§4）、3 つとも同じ場所で対応できる。一部だけ未対応にすると、設定の組み合わせで止める行が増える。有効・無効はモードの値で書く（[AGENTS.md](../../AGENTS.md) §設定値の扱い）。依存: なし |
| D2 | **内的の Head は、合成 Head として primary と同じ Network に置く。** Body をどこまで共有するかは、内的の Head が読む Body の出力で決まり、設定で選ぶ（D15 の `input_mode` と Network の設定）。補助 Network に置く形は先送りにする（§5.1） | 開発者の選択。forward が 1 回のままで、target の同期・Actor の写し・checkpoint が今の Network の仕組みにそのまま乗る。新しい器も要らない。ADR 0062 が skip で退けた形だが、退けた理由のうち「行動を入力に取る構成で forward が 2 回になる」は、行動を入力に取らない内的の Head には当たらない。「無効の checkpoint から分岐できない」は当たるので、D16 で ERROR にする。「勾配を止める NN ブロックが要る」は、止める形を先送りにしたので今は要らない（§5.1）。不利益は、既定では勾配の clip が外的と内的で共有になること（§6.4）で、分けるには D25 のモードを使う。[ADR 0074](../adr/0074-intrinsic-value-head-lives-in-primary-network-as-composite-head.md)。依存: なし |
| D3 | **β は、Q を合わせるときに掛ける。** Replay の内的の成分は、β を掛けない値にする。内的の Head が学ぶ値は β に依らない。`intrinsic.beta` の設定キーは今のまま使う | 開発者の選択。Agent57 §3.1 の形。β を schedule で変えても過去の遷移を作り直さずに済み、Actor ごとに行動の β を変えられる（D13）。173 の「方策ごとに β が違う」形でも、保存した内的報酬を使い回せる。仮説 A は「内的の学習が β に依らないと、β への敏感さが減るか」を見るので、内的の目標値の大きさが β で変わらない形が合う。不利益は、内的の Head が学ぶ値が外的より大きくなり、共有の Body に流れる勾配が内的に寄ること（§6.4）で、D4・D9・D25 で調整する。`combined` のときは今のまま、β を掛けてから外的報酬に足す。依存: D2 |
| D4 | **損失は「外的 ＋ w × 内的」で足す。** w は `learner.intrinsic.loss_weight` で、既定 1.0、0 より大きい有限の値。0 以下は ERROR | 開発者の選択。D2・D3 の帰結で、共有の Body に流れる勾配は内的が主になる（§6.4）。共有したまま調整する手段が無いと、効果が出なかった理由が分解そのものか、内的の勾配の大きさかを分けられない。0 に「学ばない」の意味を持たせない（[AGENTS.md](../../AGENTS.md) §設定値の扱い）。同じ形のキーが skip にある（`learner.skip.loss_weight`）。依存: D2、D3 |
| D5 | **bootstrap の行動。** 内的の Head は、常に合わせた Q で選んだ行動を使う。外的の Head は、`learner.intrinsic.extrinsic_bootstrap_mode`（`combined`｜`extrinsic`）で選ぶ。`combined`（既定）は合わせた Q で選んだ行動、`extrinsic` は外的の Q で選んだ行動を使う。Double DQN が有効なら online 側の値で行動を選び、target 側の各 Head で評価する。無効なら target 側の値で選ぶ | 開発者の選択。`combined` は Agent57 §3.1 の形で、割引率と終端が外的と同じなら、2 つの目標値を β で合わせた値が今の 1 本の Q の目標値と一致する（§6.1）。仮説 A を「目的は同じで、分け方だけが違う」比較として読める。`extrinsic` は、外的の Head が内的報酬を使わない点数だけの方策の価値を学ぶ形で、173 の方策の組の番号 0（β=0）に当たる。173 を待たずに評価できる。`extrinsic` の値は、173 が方策の番号を入れるときに置き換えて消す（§7.1）。Double DQN の規則は今の規則をそのまま当てる。依存: D2、D3 |
| D6 | **内的の割引率。** `learner.intrinsic.discount_mode`（`shared`｜`separate`）で選び、既定は `shared`（外的と同じ割引率）。`separate` のときだけ `learner.intrinsic.gamma` を読む。`gamma` は必須で既定値を置かず、0 より大きく 1 より小さい。外的の割引率と同じく schedule も書ける | G2。`shared` は目的が今と等価（Agent57 付録 B）なので、効果を分け方だけに絞って測れる。Burda らの 0.99 は外的 0.999 との組の値で、anet の外的の値との組では決まらないので、既定値にしない。依存: D1 |
| D7 | **終端の扱い。** `learner.intrinsic.terminal_mode`（`episodic`｜`non_episodic`）で選び、既定は `episodic`（終端で切る）。`non_episodic` のとき、内的の Head は終端の遷移でもマスクせず、Replay が返す次観測から bootstrap する。この次観測は、終端のあとの最初の観測である（§4）。終端と打ち切りが同時に真の遷移だけは、`non_episodic` でも bootstrap しない。n-step の和は終端で切れたままにする。終端の扱いのために、Replay が返す次観測・終端・打ち切りは変えない。打ち切りの扱いは今のまま（両方の Head が bootstrap を続ける） | G2。Burda らの非エピソードの形は、game over のあとの価値を次のエピソードへつなぐ。今の Replay は、終端の遷移の次観測として終端のあとの最初の観測を返しており、遷移が抽選できる時点でその slot は書かれている（§4）。マスクを外すだけで作れ、Replay も抽選できる時機も変わらない。終端のあとの最初の観測は普通に学習される状態なので、bootstrap の値に裏づけがある。終端と打ち切りが同時に真の遷移は、返る次観測が学習されない観測なので（§4）、値を取らない。依存: D1。事実の依存: §4 の「次観測」「終端と打ち切りが同時に真の遷移」 |
| D8 | **Replay の報酬の成分。** Replay は、報酬を成分ごとに受け取り、成分ごとの割引率で n-step の収益を作って返す。成分の数は Replay を組むときに決まる。定数の割引率は Push 時に成分ごとに保存し、schedule の割引率は抽選のあとに成分ごとに作り直す。抽選時の step の報酬、初期優先度の見積もりへ渡す収益と割引も、成分ごとにする。成分が 1 つのときは、今と同じ記録・収益・標本になる。Replay は、成分の意味（外的・内的）、β、終端のマスクを持たない。窓の収益の成分ごとの対応は先送りにし、成分が 2 つ以上の Replay に窓の規則を渡したら ERROR にする | G3。汎用層に内的報酬の名前を持ち込まない（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。収益を Replay が作るのは、無効のときの同一性（D17）のために、定数の割引率では Push 時の保存値の経路を保つ必要があり（ADR 0065）、成分が増えても同じ規則を当てるのが、経路を 2 通りにしない形だからである。成分の数を 2 に固定しないのは、Replay が成分の意味を持たないためである。窓の収益を先送りにするのは、窓の規則を使うのが skip だけで、分解と skip の併用を ERROR にした（D11）ので、今は使う構成が無く、試せないため。[ADR 0073](../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md)。依存: D3、D6、D11、D17 |
| D9 | **内的報酬の正規化。** 報酬スケーラに、割引収益の標準偏差で割る種類を足す。種類は、報酬スケーラの設定の `dynamic_mode`（`reward_std`｜`return_std`）で選び、既定は `reward_std`（今の動的な種類）。今の `use_dynamic_scaling` はそのまま残し、真のときだけ `dynamic_mode` を読む。`return_std` と、自動の倍率（`use_auto_post_scale = true`）の同時指定は ERROR にする。内的報酬には専用の報酬スケーラ（`intrinsic.reward_scaler.*`）を当て、既定は何も変えない組にする。取り込みで、done の lane の扱い（`intrinsic.use_done_mask`）を当てたあとの RND の倍率 − 1 を、内的報酬用の報酬スケーラに通し、その値を内的の成分にする。分解が無効のときも通し、通した値に β を掛けて外的報酬に足す。RND の推定器は変えない。中身は §6.6 | 開発者の選択（範囲に入れることと、置き場）。内的の Head が学ぶ値の桁は、内的の割引率で変わる（§6.4）。割引収益の標準偏差で割ると、割引率に依らずそろう（Burda らの狙い）。報酬スケーラに置くのは、割引和の割引率を内的の価値の割引率と 1 か所で共有でき、RND の推定器の「lane ごとの記憶を持たない」契約を変えずに済み、内的報酬の出どころを問わずに使えるためである。用語集の報酬スケーラの定義（Replay に保存される報酬の大きさを調整する部品）にも合う。分解が無効でも使えるようにするのは、1 本の Q と分解した Q を同じ内的報酬の定義で比べるため（仮説 A）。種類のキーを、定数・`reward_std`・`return_std` の 1 つの 3 択にしないのは、今ある `use_dynamic_scaling` を置き換えると出荷の設定の書き換えが要り、メイン機の固定の実行体が Run の開始ごとに読む `Atari.txt` から古いキーが消えると、既定の動的な種類に黙って切り替わるためである。自動の倍率との同時指定を止めるのは、自動の倍率が「1 step の報酬の標準偏差で割る」前提の式（§4）で、割引収益の標準偏差で割った値に掛けると割引を二重に見込むため。done の扱いを先に当てるのは、報酬スケーラが正規化するのが、Head が学ぶ報酬そのものだからである。Burda らの式との違いは §6.6 に書く。依存: D3、D6 |
| D10 | **Munchausen と併用できる。** 方策は合わせた Q から作る。Munchausen項と、soft価値ブートストラップのエントロピーの項は、外的の Head の目標値に置く。内的の Head の目標値は、同じ方策での期待値だけを使う。`extrinsic_bootstrap_mode = extrinsic` のときは、外的の Head は外的の Q から作った方策で Munchausen項と期待値を作り、内的の Head は合わせた Q の方策のままにする。式は §6.2 | 開発者の選択。この 2 つの項は、合わせた Q の単位（係数 1）の量である。内的の側に置くと β で割ることになり、β=0 で定義できない。外的に置けば、割引率と終端が外的と同じとき、2 つの目標値を β で合わせた値が今の Munchausen の目標値と一致する。置き方が 1 つに決まるので、併用を止める理由が残らない。Atari の標準の構成（Munchausen 有効）のまま分解を使える。168 が扱う「n-step の窓の途中の Munchausen項」とは独立である。依存: D3、D5 |
| D11 | **TempoRL（skip）との併用は ERROR にする。** `decomposed` と `skip.enabled = true` を同時に指定すると、構築時に止める | 開発者の選択。先送りで、理由と作る条件は §5.1。依存: D1 |
| D12 | **PER の優先度。** 優先度は「外的の TD 誤差の絶対値 ＋ λ × 内的の TD 誤差の絶対値」にする。λ は `learner.intrinsic.priority_lambda` で、既定 1.0、0 以上の有限の値。今の優先度の下限と上限の扱いは、足したあとの値に 1 回だけ当てる。初期優先度の見積もりも、同じ決まりで Head ごとに作って足す。Actor の hint は、`intrinsic.value_mode` に依らず常に 5 列（Q_sa、V_s、Munchausen項、内的の Q_sa、内的の V_s）にする。`decomposed` のとき、前の 3 列は外的の Head の値である。`combined` のとき、前の 3 列は今と同じ 1 本の Q の値で、内的の 2 列は 0 にする。`non_episodic` の終端の遷移は、内的の側も bootstrap なしで見積もる | 開発者の選択。合わせた Q の誤差（符号つきで足してから絶対値）にすると、β が小さいとき内的の Head だけが外している遷移が抽選されず、内的の学習が β に左右される。2 つの誤差が逆向きだと打ち消し合う。損失の重み（D4）と同じ値に固定しないのは、損失の重みは共有の Body の勾配の釣り合い、優先度の重みは抽選の釣り合いで、効く先が違うためである。事例は §2.4。「書かなければ損失の重みと同じ」にしないのは、未指定に意味を持たせないため。hint の運び方は ADR 0012 のまま。列の数をモードで変えないのは、ADR 0036 が「Munchausen が有効のときだけ 3 列にする形」を、検証と詰め方の契約がモードに依るという理由で退けたのと同じ判断による。ADR 0036 の「常時 3 列」は「常時 5 列」に改まるので、[ADR 0075](../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md) を起こし、ADR 0036 の先頭に注記を足す（D21）。終端の遷移の時点では、終端のあとの hint がまだ無い。依存: D3、D4 |
| D13 | **Actor ごとの行動の β。** `actor.[<名前>].intrinsic.action_beta_mode`（`learner`｜`fixed`）で選び、既定は `learner`（学習と同じ β）。`fixed` のときだけ `actor.[<名前>].intrinsic.action_beta` を読む。値は必須で、0 以上の有限の値。`learner` のモードでは、Actor が β の schedule を自分の step で評価する。学習の Actor にも指定でき、その場合は行動だけが変わり、目標値は学習の β のままになる | 開発者の選択。G5。値 0 で、外的の Head だけで選んだ行動を評価できる。途中の値も書けるので、同じ重みで行動の β だけを変えた評価ができ、β への敏感さが学習から来るのか行動の選び方から来るのかを分けられる（仮説 A の材料）。β の読み方は、割引率や ε と同じ扱いである。学習の Actor への指定が意味のある組み合わせかは、利用側の責任とする（[AGENTS.md](../../AGENTS.md) §汎用機構と利用側の責任境界）。173 が方策の番号で評価する形に決めたら、方策の番号の指定に置き換えて消す。依存: D3 |
| D14 | **合わせる計算。** 分布型（QR・IQN）では、合わせた分位点を「同じ τ どうしを β で足した値」とする（近似）。`learner.use_tbo` が有効のときは、各 Head の出力を h の逆変換で戻してから β で合わせ、h を掛け直した値を読み手へ渡す | 分位点の平均は厳密に一致するので、平均を使う方策（Greedy、ε-greedy）の行動は近似の影響を受けない。分布を使う方策（UQE、Thompson）は、近似の影響を受ける。2 つの確率変数の和の分位点は、分位点の和にならないためである。TBO では Head の出力が h で圧縮した値で、h は和を保たない。h を掛け直すのは、読み手が h で圧縮した値を前提にしており（§4）、戻した値のまま渡すと逆変換が二重になるため。依存: D3、D24 |
| D15 | **名前。** 設定キーと、Network の出力・Trace の鍵は §8.1・§8.4 のとおり。学習側のキーは `learner.intrinsic.*`、内的の Head の構造は `intrinsic.value.head.*`、Actor ごとのキーは `actor.[<名前>].intrinsic.*` に置く。内的の Head が読む Body の出力は、`intrinsic.value.head.input_mode`（`shared`｜`dedicated`）で選ぶ。内的の Head の設定キーの既定は `agent.txt` に書いた値で、外的の Head の解決後の値を継がない | 階層は、値を読む持ち主に合わせる（[実装 品質確認観点](../quality/impl.md) の「名前」の 1）。skip の `learner.skip.*`・`skip.head.*`・`actor.[<名前>].skip.*` と同じ分け方になる（同 3）。`value_mode` は何のモードかが名前に出る（同 4）。`priority_lambda` は skip の同じ役のキーと語をそろえる（同 3）。`feature_gradient` も、skip の同じ役のキー（`learner.skip.feature_gradient`）と、キーの語も値の語もそろえる（同 3）。`extrinsic_bootstrap_mode` は、誰の bootstrap かを名前に入れる（同 4）。`input_mode` は、共用か専用かをモードで書き、未指定に意味を持たせない。既定を継がないのも同じ理由である。Network の出力の鍵は、`skip_q` と同じ付け方にする（同 3）。Trace の鍵を `intrinsic.` で始めないのは、127 がその接頭辞の鍵を RND の推定器の写しを使う経路で計算するので（§4）、Q の列を見るだけで写しの費用がかかるためである（同 2）。依存: D1〜D13、D24、D25 |
| D16 | **checkpoint。** 内的の Head は Network の一部として保存する。設定と checkpoint で、価値の分解の有無か、損失ごとの最適化のモード（D25）が違えば ERROR にする。状態を持つ報酬スケーラ（動的な種類）の状態を、外的報酬用も内的報酬用も保存する。報酬スケーラの種類が設定と checkpoint で違えば ERROR にする。`return_std` で lane の数が違う checkpoint は、統計を読み、割引和だけ 0 から始めて WARN を 1 回出す。互換の読み替えは作らない。状態を持つ報酬スケーラを使う構成の、今ある checkpoint は読めなくなる | 開発者の選択（外的報酬用も保存すること）。分解の有無が違う重みは Network の形が違うので、黙って読むと学習していない Head で動く。報酬スケーラの状態を保存しないと、checkpoint 引き継ぎの直後に標準偏差が小さく出て、割った報酬が大きく振れる。今の報酬スケーラも同じ未対応を持つので（§4）、同じ部品の扱いをそろえて片づける。読めなくなるのは LunarLander・GridMaze・CartPole の既定の構成の checkpoint で、Atari と DropMerge は定数の倍率なので、checkpoint の形が今と同じで影響を受けない（§4）。lane の数が違う場合の扱いは、checkpoint 引き継ぎで lane の数を変える使い方を止めないためで、[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則の表の「checkpoint に設定が求める部分が無く、初期化した」に当たる。依存: D2、D9、D25 |
| D17 | **無効のときの同一性。** `intrinsic.value_mode` と報酬スケーラの新しい設定キーが既定のとき、forward の回数と順、乱数の消費、loss、優先度を今と同じに保つ。checkpoint の形は、状態を持つ報酬スケーラを使わない構成では今と同じに保つ。記録されるメトリクスの tag も変えない（足す tag は、既定では選ばれない組に置く。D20）。Actor の hint を使う構成では、hint が 5 列になるが（D12）、足した 2 列は 0 で、初期優先度の値は今と同じである。実装前に保存した実行体と、同 seed・決定的モードの短い Run の metrics checksum を比べて確かめる | トピック別まとめの「変えるときの条件」（§4）。比較の相手は、実装前に保存した実行体にする（[324 DQN](../wiki/categories/324_dqn.md) の PRD084 の条件）。checkpoint の形の条件を狭めるのは、D16 で動的な報酬スケーラの状態を保存するため。hint の列が増えても初期優先度の値を同じに保つ形は、ADR 0036 が Munchausen 無効のときに置いた条件と同じである。依存: D1、D8、D9、D12、D16、D20 |
| D18 | **順序。** 本 PRD の実装は、§7.3 に挙げるリファクタリングの課題 5 件を片づけたあとに行う。5 件は本 PRD の範囲に取り込まず、[128](0128_dqn_agent_refactor_bundle_10rfd.md) が扱う | 開発者の選択。5 件は、本 PRD が変更を足す場所（取り込み、標本、Head の選択、設定の読み取り、診断の群）の重複や散らばりである。先に片づけると、本 PRD の変更が 1 か所ずつで済む。取り込まないのは、機能の追加と挙動を変えない直しを 1 本に混ぜると、前後の比較で差が出たときに原因を分けにくいため（PRD125 と同じ進め方）。依存: なし |
| D19 | **効果の確かめの置き場。** Atari.txt の Run プロファイルと bat の群の組み立ては、本 PRD の範囲に入れず、Atari の枠で実験の段取りと一緒に決める。本 PRD が足す設定は、`agent.txt` の既定の設定キーと、`metrics_scalar.txt` のメトリクスの組（既定では選ばれない。D20）だけにする。比較の基準は、PRD123 の新しい RND の条件で取り直す。実装の受入は正しさの項目だけにし（§9.2）、Run の結果は Atari の枠で実験記録に書く | 開発者の選択。Atari.txt は Run の開始ごとに読まれるので、Run の条件に効く値は、流す直前に、そのときの基準の Run と合わせて決める。基準の取り直しは [123](0123_rnd_intrinsic_reward_scale_10rfd.md) D15 による（b001 は PRD123 より前の中身の Run である）。依存: D18 |
| D20 | **診断。** Evaluation Trace の `$agent_step` に、`extrinsic_q_sa`・`intrinsic_q_sa`・`extrinsic_state_value`・`intrinsic_state_value` の鍵を足す。今の `q_sa`・`state_value` は、その Actor の β で合わせた値を出す。既定の列には足さない。1 step の内的報酬の列は 127 が持ち、本 PRD は足さない。分解が有効のとき、今ある学習のメトリクス（`q_sa`・`max_q` など）は合わせた Q の値を出し、RND の診断の `bonus_*` は「β × 内的報酬用の報酬スケーラを通した値」を出す。メトリクスは §8.4 の 1〜5 を足す。tag の定義は、`metrics_scalar.txt` の組 `metrics.scalar.@value_decomposition` に置き、どの Run プロファイルの連鎖にも足さない | G5。鍵は Network の出力の鍵（`extrinsic_q`・`intrinsic_q`）と同じ付け方にする（D15）。今ある値は、今の意味を保つ。学習のメトリクスは行動に使う Q、`bonus_*` は学習に入る内的報酬の大きさなので、分解の有無をまたいで比べられる。足すメトリクスは、損失の重み（D4）と clip の共有（D2）が効いているかを学習中に読むためのものである。tag を組に置くのは、skip・Munchausen・NoisyNet の診断と同じ置き方（`metrics.scalar.@skip` など）で、使う Run が `metrics.scalar.$` の連鎖に足すまで、記録される tag が変わらないためである。RND の tag のように Env ごとの Run プロファイルへ書かないのは、価値の分解を使う Run プロファイルを本 PRD が作らないため（D19）。依存: D2、D4、D9、D13、D19、D24 |
| D21 | **文書。** 設計文書 `docs/design/214_value_decomposition.jp.md` を新設し、今ある設計文書とユーザーガイドを追随させる（一覧は §8.7 が正本）。ADR は 0073・0074・0075 の 3 本を起こし、ADR 0036・ADR 0062・ADR 0065 の先頭に注記を足す。[CONTEXT.md](../../CONTEXT.md) に「価値の分解」と「報酬の成分」を足す。「Actor Qヒント」の項目の、運ぶ中身の説明（実行行動の価値、bootstrap 用の状態価値、Munchausen項）には、実装のときに、内的の Head の Q_sa と V_s を足す | 新しい仕組みなので、現在形を書く設計文書が要る。ADR は、戻しにくく、文脈が無いと驚き、本当のトレードオフがあった 3 つに限る（Replay の口、ADR 0062 が skip で退けた形を内的の Head で採ったこと、Actor の hint の列の数）。既存の ADR の判断を改めるときは、古い ADR の先頭に注記を足し、新しい ADR を起こす（[PRD 品質確認観点](../quality/rfd.md) の項番 23）。依存: D2、D8、D12 |
| D22 | **性能。** 受入に閾値を置かない。実装後に、Atari の実時間を無効と有効（Body を共有する構成）で交互に測り、実装計画に記録する | 既定のモードでは、forward と backward の回数と順は変えない（D17）。足されるのは、Actor の forward ごとに Head の射影 1 組と合わせる計算、Learner の更新ごとに online と target の射影 1 組ずつと損失 1 つ、Replay の 1 step あたり報酬 1 つと slot あたり収益 1 つ、取り込みの lane あたりの四則演算数回、Actor の hint の値 2 つ、Actor の step ごとの診断（D26。行動の数の並びへの argmax と、1 位・2 位の取り出し）、評価の Actor の session の足し込み（D27。Actor と同じ device での加算）である。hint の 2 列は、hint を使う構成なら分解が無効でも増える（値は 0。D12）。`feature_gradient = split`（D25）では backward が 2 回になり、Body を別にする構成では Body の forward が 2 倍になるが、それは構成を選んだ利用側の費用である。依存: D2、D8、D12、D25、D26、D27 |
| D23 | **互換。** 旧い Push の口の別名や互換の分岐は作らない。過去の Run・ADR・実験記録は書き換えない | [AGENTS.md](../../AGENTS.md) §クリーンブレーク方針。依存: D8 |
| D24 | **合わせた Q の届け方。** 分解が有効のとき、Network は `q`・`q_dist` を出さない。合成 Head の出力は、外的が `extrinsic_q`・`extrinsic_q_dist`、内的が `intrinsic_q`・`intrinsic_q_dist` である。DQN 側の 1 つの関数が、forward の結果に、β で合わせた `q`・`q_dist` を足す（合わせ方は D14）。行動の方策や診断などの読み手は、今のまま `q`・`q_dist` を読む。外的の Q だけが要る箇所（D5 の `extrinsic`）は、`extrinsic_q` を名指しで読む。分解が無効のときは、今と同じく Network が `q`・`q_dist` を出す | 開発者の選択。`q`・`q_dist` を直接読む箇所が 17 か所ある（§4）。`q` を外的の Head の出力のままにして読み手ごとに合わせる形は、直し漏れた読み手が ERROR にならず、外的の Q だけで動く（[AGENTS.md](../../AGENTS.md) §Fail-Fast 原則に反する）。Network が `q` を出さなければ、合わせる関数を通し忘れた経路は、鍵が無いので必ず止まる。合わせる計算と TBO の扱いが 1 か所にまとまり、読み手を変えずに済む。β を Network の入力にする形を採らないのは、Head が TBO の変換を知る必要が出て、Network の入力の契約が変わるためである。依存: D2、D3、D14 |
| D25 | **損失ごとの最適化のモード。** `learner.intrinsic.feature_gradient`（`merge`｜`split`）で選び、既定は `merge`（D4 の和で backward を 1 回）。`split` は、backward を外的の損失と内的の損失で 1 回ずつ取り、勾配の clip を損失ごとに掛け、共有の重みを損失ごとの Adam の状態で更新する | 開発者の選択。skip の `feature_gradient = split` の経路がすでにあり（§4）、同じ作りを流用できる。キーの語と値の語は、skip の `learner.skip.feature_gradient`（`stop`｜`merge`｜`split`）にそろえる。`stop` を値に持たないのは、共有の Body への勾配を止める形を先送りにしたため（§5.1）。Agent57 が NN を分ける理由の 1 つ（optimizer の状態を分ける）を、Body を共有したまま満たせる。損失ごとに歩幅がそろうので、内的の値の桁が大きくても外的の更新が埋もれず、問題 1 に直接効く。`split` では、損失の重み（D4）はほとんど効かなくなる（Adam が損失ごとに歩幅をそろえるため。skip の `split` と同じ性質）。依存: D2、D4 |
| D26 | **行動の選択への効き方の診断。** 価値の分解が有効のとき、Actor は、行動を選ぶたびに、lane ごとに次の 4 つを作る。(1) 不一致: 合わせた Q の 1 位の行動と、外的の Q の 1 位の行動が違えば 1、同じなら 0。(2) 譲った外的の価値: 外的の Q で見た、外的の 1 位の行動と、合わせた Q の 1 位の行動の差。(3) 外的の差: Q_外 の 1 位と 2 位の差。(4) 内的の差: β·Q_内 の 1 位と 2 位の差。定義は §6.8、鍵は §8.4 | 開発者の選択。行動は合わせた Q で選ぶので、β の釣り合いは分解しても行動の選択に残る。`intrinsic_q_share`（§8.4 の 4）は選んだ行動の値に占める割合で、値が行動間で一定なら選択に効かない。不一致で頻度、譲った外的の価値で変えたときに手放した量、2 つの差で行動間の差の大きさの釣り合いが読める。外的の差は、譲った外的の価値を比べる相手にもなる。差の定義は、今ある `q_gap` と同じである（§2.5）。合わせた Q の 1 位と 2 位の差に占める内的の割合を 1 つの値で出す形は、別モデルの検討で弱点が分かり、開発者が外すと決めた（§5.1）。依存: D3、D13、D14、D24 |
| D27 | **診断を読める場所。** Evaluation Trace は、lane ごとの値の鍵を列にする。学習の Actor は、step ごとの lane の平均の鍵を読む（`@train $action_info`）。評価の Actor は、session のあいだの累積の鍵を、session の終わりに読む（`$eval.[<名前>] $actor @session_end`）。3 種の鍵は、別の名前にする（§8.4）。Learner のバッチでは数えない | 開発者の選択（評価の Actor の累積を足すこと）。成績は評価側で読み、起点の Run でも Greedy が止まる局面が問題だった（§2.1）。学習の Actor の値は、探索しながら訪れた局面のものである。評価の Actor の値を step の値で取らないのは、103 の D8 が退けた形のためで（§4）、skip の session の指標と同じ作りにする。lane ごとの値と lane の平均を別の鍵にするのは、Trace が行ごとに lane を指定して読むためで（§4）、同じクラスの skip の鍵（`skip_decision` と `skip_decision_rate`）と同じ分け方にする。Learner のバッチで数えないのは、Actor ごとの行動の β（D13）を反映できず、局面が PER の抽選に偏るためである。依存: D13、D20、D26 |
| D28 | **診断の細部。** 比べる Q は、方策が使うスコアではなく、Head ごとの平均の Q にする。4 つとも実空間の値で作る。内的の差には、その Actor の行動の β を掛ける。行動が 1 つの Env では、不一致と譲った外的の価値は 0、2 つの差は NaN にする。ε の探索や強制行動で実際の行動が変わっても、値は変えない。分解が有効なら、購読の有無に依らず毎 step 作る。数えた数が 0 の session で読んだ値は NaN にする。tag の定義は `metrics.scalar.@value_decomposition` に置き、評価の session の累積の tag は、どの Env にもある評価（`[eval]`・`[eval_target]`）のぶんを書く。tag は、鍵と別の短い名前にする。Evaluation Trace の既定の列には足さない | 推奨の既定を採った細部。平均の Q で比べるのは、方策の種類に依らず同じ定義にでき、追加の計算が要らないため。診断が表すのは「平均の Q で選んだとしたときの、選び方への効き方」で、分布を使う方策（UQE、Thompson）では、方策が実際に選ぶ行動とずれうる。実空間にするのは、β が 0 のとき、合わせた Q と外的の Q が同じ値になって不一致が必ず 0 になり、差と譲った外的の価値が、報酬スケーラを通した外的報酬の単位になるためである（TBO の h は和を保たない。D14）。方策が argmax を取る値（h を掛けた値）とは、同点に近い行動で、丸めのぶんだけ 1 位がずれうる。内的の差に β を掛けるのは、外的の差と同じ単位で並べるためで、`intrinsic_q_share` も β を掛けた値で作る（§8.4）。行動が 1 つのときの扱いは、今ある `q_gap` が行動 2 つ以上でだけ計算するのに合わせる（§4）。実際の行動に依らないのは、探索を除いた選び方に内的が効いたかを見る指標にするため。毎 step 作るのは、既存の方策の診断と同じ作りで（§4）、分岐を増やさないため。数えていない session の扱いは、skip の session の指標と同じである（§4）。評価の名前を 2 つに絞るのは、`[greedy]` などが Atari の設定だけにあり（§4）、その評価が無い構成で組を選ぶと、評価の名前が無いという ERROR になるためである。そのぶんは Atari の枠で足す（D19）。tag を鍵と別の名前にするのは、tag が群名と `/` を含めて 40 文字以内の決まりで（[実装 品質確認観点](../quality/impl.md) の「設定ファイル」）、鍵のままでは超えるためである。Trace の扱いは D20 と同じ。依存: D14、D19、D20、D26、D27 |

### 5.1 選ばなかった形と先送り

| 形 | 裁定 | 理由と、作る条件 |
|---|---|---|
| 内的の Head を補助 Network（skip と同じ器）に置く形 | 先送り | forward が Actor と Learner で 1 回ずつ増え、skip 専用の箇所を内的用にも書くことになる。ADR 0062 の「補助の target は bootstrap に使わない」も改める必要がある。勾配の clip と Adam の状態を分けることは、D25 のモードで合成 Head のまま満たせる。先送りの理由は、実験の結果が要ること。作る条件: `split`（D25）でも共有が学習を乱し、Network ごとに分ける必要があると分かったとき |
| 共有の Body への内的の勾配を止める NN ブロック | 先送り | Montezuma のように外的報酬が疎らなゲームでは、外的だけでは特徴が育ちにくく、内的の Head が読む特徴も貧しくなるおそれがある。先送りの理由は、実験の結果が要ること。作る条件: 損失の重みを下げても、`split` にしても、共有の Body が内的に引っ張られると分かったとき |
| β を Replay に保存する前に掛ける形 | 不採用 | β を変えると内的の目標値の大きさも変わり、仮説 A の読みを濁す。173 の方策ごとの β に使い回せない（D3） |
| 損失を 1:1 に固定する形 | 不採用 | 共有の Body のまま調整する手段が無くなる（D4） |
| 外的の Head の bootstrap を、合わせた Q だけ、または外的の Q だけにする形 | 不採用 | 前者は点数だけの方策を 173 まで評価できず、後者は今の 1 本の Q との一致の受入を置けない（D5） |
| 内的の Head も自分の argmax で bootstrap する形 | 不採用 | 合わせた Q が、どの方策の価値でもなくなる。使う場面が無い |
| 終端の観測から bootstrap する近似（起票時の既定） | 不採用 | 今の Replay は終端の観測を保存していないので作れない。終端のあとの最初の観測からの bootstrap が、Replay を変えずに作れる（D7） |
| 収益を Learner が抽選のあとに作る形 | 不採用 | 無効のときの同一性のために定数の割引率の経路を保つ必要があり（ADR 0065、D17）、成分があるときだけ別の経路にすると、経路が 2 通りになる（D8） |
| 窓の収益を成分ごとに作る形 | 先送り | 窓の規則を使うのは skip だけで、分解と skip の併用は ERROR である（D11）。先送りの理由は、TempoRL との併用と同じ（実験の結果が要る）。作る条件: TempoRL との併用を作るとき |
| 割引収益の標準偏差で割る正規化を、RND の推定器の統計の取り方として作る形 | 不採用 | 割引率を推定器の設定に別に書くことになり、内的の価値の割引率と 2 か所になる。RND 専用で、推定器の口が lane の並びを覚えるものに変わる。勝つ点は Burda らの式との厳密な一致だけ（D9） |
| 報酬スケーラの種類を、1 つの 3 択の設定キーにする形 | 不採用 | 今ある `use_dynamic_scaling` を置き換えることになり、固定の実行体が読む設定から古いキーが消えると、黙って動的な種類に切り替わる（D9） |
| 割引収益の種類のとき、自動の倍率を (1 − γ) を掛けない意味に読み替える形 | 不採用 | 同じ設定キーが種類で意味を変える。同時指定を ERROR にし、倍率は手動で書く（D9） |
| 報酬スケーラの状態を、内的報酬用だけ保存する形、どちらも保存しない形 | 不採用 | 前者は同じ部品で扱いが分かれ、後者は checkpoint 引き継ぎの直後に正規化が乱れる（D16） |
| Munchausen との併用を ERROR にする形（起票時の既定） | 不採用 | 置き方が 1 つに決まったので、止める理由が残らない（D10） |
| TempoRL と併用できる形（skip は合わせた価値で学ぶ形、skip の Head も分ける形） | 先送り | 前者は、行動の β を変えた Actor で、行動は外的の Q で選ぶのに続ける長さは β 込みの価値で選ぶ食い違いが残る。後者は skip の Head・損失・設定・診断が 2 本ぶんになる。先送りの理由は、実験の結果が要ること。TempoRL が既定で有効なのは LunarLander で、LunarLander の RND は bonus が採点 5〜10 万件で消え、評価は基準と同じ帯だった（[内的報酬による探索の比較](../wiki/mechanisms/intrinsic_novelty.md)）。作る条件: 分解の効果が出て、TempoRL が既定で有効な構成でも使いたくなったとき。173 も TempoRL との併用を未決に持つ |
| 優先度を合わせた Q の誤差にする形（起票時の既定）、損失と同じ重みに固定する形 | 不採用 | D12 の理由 |
| Actor の hint を、価値の分解が有効のときだけ 5 列にする形 | 不採用 | 検証と詰め方の契約がモードに依る。ADR 0036 が Munchausen で退けた形と同じ（D12、ADR 0075） |
| Actor の hint は 3 列のままにし、初期優先度だけ合わせた Q の誤差で見積もる形 | 不採用 | 初期優先度と、学習のあとの優先度で決まりが分かれ、内的の Head だけが外している遷移の初期優先度が低く出る（D12、ADR 0075） |
| 行動の β を、β=0 だけ選べる 2 値のモードにする形 | 不採用 | 途中の値での評価ができない。実装の手間はほぼ変わらない（D13） |
| `q` を外的の Head の出力のままにし、読み手ごとに合わせる形 | 不採用 | 直し漏れが黙って通る（D24） |
| β を Network の入力にし、合成 Head が合わせた `q` も出す形 | 不採用 | Head が TBO の変換を知る必要が出て、Network の入力の契約が変わる（D24） |
| リファクタリングの課題 5 件を本 PRD に取り込む形、順序を決めない形 | 不採用 | D18 の理由 |
| Atari.txt の Run プロファイルの追加を本 PRD に入れる形 | 範囲外 | Atari の枠で決める（D19） |
| Head ごとの行動間の差の比（β × 内的の 1 位と 2 位の差 ÷ 外的の 1 位と 2 位の差）を、1 つの値で出す形 | 不採用 | 外的の差が 0 に近い局面で、比が跳ね上がる。割らずに、2 つの差をそのまま並べる形にした（D26） |
| 合わせた Q の 1 位と 2 位の差に占める内的の割合を出す形 | 不採用 | 向きを持たず、内的が同じ向きに押した標本と、逆らって及ばなかった標本が同じ値になる。内的が行動を変えた標本では 0.5 を超えやすく、平均が不一致率と混ざる。外的の差も内的の差も 0 に近い局面では、雑音で決まる（別モデルの検討で分かった。D26） |
| lane ごとの値と lane の平均を、同じ鍵で返す形（lane を指定すればその値、指定しなければ平均） | 不採用 | 鍵は少なくなるが、同じクラスに、その形の鍵が無い。skip の鍵と同じく、別の名前にする（D27） |
| 2 つの Head の 1 位の行動が違う割合を診断にする形 | 不採用 | 行動が多い Env では、2 つの Head の 1 位が偶然そろうことがまれで、ほぼ常に 1 に近くなる（D26） |
| 診断を Learner のバッチで数える形 | 不採用 | Actor ごとの行動の β を反映できず、局面が PER の抽選に偏る（D27） |
| 評価の Actor の診断を足さない形、評価の Actor の値を step の値で取る形 | 不採用 | 前者は、Greedy の評価で読めない。後者は、103 の D8 が退けた形である（D27） |
| 行動間の差を広げる学習則、Head ごとに差の大きさをそろえる正規化 | 範囲外 | 別の関心。作る条件: 足した診断で、内的が行動を動かせていない（不一致率が 0 に近く、内的の差が外的の差よりずっと小さい）か、外的の価値を大きく手放している（譲った外的の価値が、外的の差に比べて大きい）と分かったとき（§1.4） |

### 5.2 起票時の未決表との対応

| 起票時の行 | 移した先 |
|---|---|
| 1（入れる範囲） | D1。課題 ab98b7 を先に片づけるかは、PRD125 で片づいた（§7.3） |
| 2（Head の入れ方） | D2、D24、D25 |
| 3（割引率） | D6 |
| 4（終端） | D7（近似をやめ、終端のあとの最初の観測からの bootstrap にした） |
| 5（β を掛ける位置） | D3 |
| 6（Replay の口） | D8 |
| 7（bootstrap の行動） | D5 |
| 8（Actor ごとの行動の β） | D13 |
| 9（Munchausen との併用） | D10（併用できる形に変えた） |
| 10（TempoRL との併用） | D11 |
| 11（PER の優先度） | D12 |
| 12（損失の重み） | D4（設定キーを足す形に変えた） |
| 13（Trace の列）・14（メトリクス） | D20 |
| 15（checkpoint） | D16 |
| 16（実験の組み立て） | D19、§9.1（材料） |
| 17（173 の課題との依存） | §7.3（173 の課題の待ちに、本 PRD の課題が入っている） |
| 18（名前と用語） | D15、D21 |
| 19（設計文書） | D21（設計文書を新設する形に変えた） |
| 20（ADR） | D21 |
| 21（性能） | D22 |

## 6. 仕様

### 6.1 目標値

**今の 1 本の Q。** r = r_外 ＋ β·r_内 として、

y = Σ_{h=0}^{n−1} γ^h r_{t+h+1} ＋ γ^n (1 − d) Q_target(s_{t+n}, a*)

**分解した Q。** 合わせた Q は Q(s, a) = Q_外(s, a) ＋ β·Q_内(s, a) である。

y_外 = Σ_{h=0}^{n−1} γ_外^h r_外,{t+h+1} ＋ γ_外^n (1 − d) Q_外,target(s_{t+n}, a_外*)

y_内 = Σ_{h=0}^{n−1} γ_内^h r_内,{t+h+1} ＋ γ_内^n (1 − d_内) Q_内,target(s_{t+n}, a*)

- a* は、合わせた Q で選んだ行動である。a_外* は、`extrinsic_bootstrap_mode` が `combined` なら a*、`extrinsic` なら外的の Q で選んだ行動である（D5）。
- `discount_mode = shared` なら γ_内 = γ_外、`separate` なら γ_内 は `learner.intrinsic.gamma` の値である（D6）。
- `terminal_mode = episodic` なら d_内 = d である。`non_episodic` なら d_内 = 0 で、s_{t+n} は終端のあとの最初の観測である。ただし、終端と打ち切りが同時に真の遷移では d_内 = d にする（D7）。
- `discount_mode = shared`・`terminal_mode = episodic`・`extrinsic_bootstrap_mode = combined` のとき、y_外 ＋ β·y_内 は、今の 1 本の Q の y と同じ値になる。TBO が有効のときは、h を掛ける前の値で成り立つ。
- r_外 は外的報酬の報酬スケーラを通した値（今と同じ）、r_内 は内的報酬用の報酬スケーラを通した値で、β を掛けていない（D3、D9）。

### 6.2 Munchausen が有効のとき

π は Q から作る方策である。今の 1 本の Q の目標値は次のとおり。

y = 収益 ＋ Munchausen項 ＋ γ^n (1 − d) Σ_{a'} π(a') (Q_target(s', a') − τ·log π(a'))

分解したときは、π を合わせた Q から作り、次のように置く（D10）。

y_外 = 外的の収益 ＋ Munchausen項 ＋ γ_外^n (1 − d) Σ_{a'} π_外(a') (Q_外,target(s', a') − τ·log π_外(a'))

y_内 = 内的の収益 ＋ γ_内^n (1 − d_内) Σ_{a'} π(a') Q_内,target(s', a')

- π_外 は、`extrinsic_bootstrap_mode` が `combined` なら π、`extrinsic` なら外的の Q から作った方策である。Munchausen項も、同じ方策から作る。
- 方策を online の値から作るか target の値から作るか（`log_policy_mode`）は、今の決まりのままで、使う値を合わせた Q（`extrinsic` のときの外的の Head は外的の Q）に替える。
- `discount_mode = shared`・`terminal_mode = episodic`・`extrinsic_bootstrap_mode = combined` のとき、y_外 ＋ β·y_内 は、今の Munchausen の目標値と同じ値になる。TBO が有効のときは、h を掛ける前の値で成り立つ。
- Munchausen項を先頭の遷移に 1 度だけ足し、終端のマスクの対象にしないことは、今と同じである。

### 6.3 合わせる計算

分解が有効のとき、forward の結果は `extrinsic_q`・`intrinsic_q`（分布型では `extrinsic_q_dist`・`intrinsic_q_dist` も）を持ち、`q`・`q_dist` を持たない。DQN 側の 1 つの関数が、β を受け取って、合わせた `q`・`q_dist` を forward の結果に足す（D24）。

- 分布型（QR・IQN）では、合わせた分位点は、同じ τ どうしを β で足した値とする（D14）。IQN の τ は Body で特徴に混ぜるので、Body を共有する構成では、2 つの Head は同じ τ の標本を使う。
- `learner.use_tbo` が有効のときは、各 Head の出力を h の逆変換で戻し、β で合わせ、h を掛け直す。
- 渡す β は、Learner では学習の β、Actor では行動の β（§6.7）である。学習の β は、取り込みの入口が評価した値を、更新ごとに Learner へ渡す。
- この関数を通さない経路で `q`・`q_dist` を読むと、鍵が無いので止まる。

### 6.4 損失と優先度、共有の Body への影響

- 損失は L = L_外 ＋ w·L_内 である（D4）。L_外 と L_内 は、今の損失の計算を、それぞれの Head の現在値と目標値に当てたものである。各 Head の目標値は、今と同じく、TBO が有効なら h を掛けた値で組む。
- 優先度は |δ_外| ＋ λ·|δ_内| である（D12）。δ は、今の優先度に使っている標本ごとの TD 誤差である。今の優先度の下限と上限の扱いは、足したあとの値に 1 回だけ当てる。
- **内的の Head が学ぶ値の桁（目安）。** b001 では、β を掛ける前の 1 step の内的報酬が 0.023〜0.036 だった（§2.1）。割引率 0.997 なら、Q_内 は 0.03 ÷ (1 − 0.997) で 10 前後、0.99 なら 3 前後になる。外的の Q は 0.05〜2.7 だった。PRD123 の新しい RND では、内的報酬が数倍大きくなる見込みである（PRD123 の膨らみの直しからの推定で、Run では確かめていない）。
- **損失の効く範囲。** 損失の折れ目（Huber と分位点の κ）と TD 誤差の clip は、外的と内的で共用する。内的の Head は値の桁が大きいので、今の 1 本の Q とは違う範囲（誤差が折れ目より大きい側）で学ぶことが多くなる。
- **`feature_gradient = merge`（既定）。** 2 つの損失を足して 1 回の backward をするので、Body を共有する構成では、共有の Body に流れる勾配は内的が主になる。勾配の clip も共有である。調整の手段は、損失の重み（D4）、内的報酬用の報酬スケーラ（D9）、内的の割引率（D6）、Body の共有の度合い（§8.1 の `input_mode`）である。内的の Head だけが持つ重みは、Adam が歩幅をそろえるので、w を下げても学ぶ速さはあまり変わらない見込みである（Run では確かめていない）。
- **`feature_gradient = split`。** backward を w·L_内 と L_外 で 1 回ずつ取り、損失ごとに clip し、共有の重みを損失ごとの Adam の状態で更新する（D25）。損失ごとに歩幅がそろうので、w はほとんど効かなくなる。

### 6.5 報酬の流れ

| 流れの位置 | `combined`（今の 1 本の Q） | `decomposed` |
|---|---|---|
| 外的報酬 | 外的報酬の報酬スケーラを通す | 同じ |
| 内的報酬 | RND の倍率 − 1 に、done の lane の扱い（`intrinsic.use_done_mask`）を当てる。そのあと、内的報酬用の報酬スケーラに通す | 同じ |
| 取り込みから Replay へ | 外的 ＋ β × 内的 の 1 成分 | 外的と内的の 2 成分。β は掛けない |
| Replay | 1 成分。割引率は外的の割引率 | 2 成分。割引率は外的の割引率と、内的の割引率（D6） |
| Learner | 1 本の Q | 2 つの Head。β は合わせるときに掛ける |

- 内的報酬用の報酬スケーラが既定（何も変えない組）のとき、`combined` の流れは今と同じ値になる。
- `intrinsic.use_done_mask`（done の step の内的報酬を 0 にするか）と、`terminal_mode`（内的の価値を終端で切るか）は、別々に指定できる（[123](0123_rnd_intrinsic_reward_scale_10rfd.md) D16 が、done の step の扱いを独立の軸として決めた）。Burda らの形に寄せるなら、`use_done_mask` を偽にする。

### 6.6 割引収益の標準偏差で割る報酬スケーラ

`dynamic_mode = return_std` の報酬スケーラは、次のように動く（D9）。

1. lane ごとに、報酬の割引和 R を持つ。step ごとに R ← γ·R ＋ r と更新する。エピソードの終わりでも R を 0 に戻さない。
2. R の値の、累積の標準偏差 σ_R を取る。
3. 報酬 r を σ_R で割る。平均は引かない。

- γ は、その報酬の価値に使う割引率である。外的報酬の報酬スケーラは外的の割引率を使う。内的報酬用の報酬スケーラは、`decomposed` なら内的の割引率（D6）、`combined` なら外的の割引率を使う。割引率が schedule のときは、構築時（step 0）の値を使う。今の報酬スケーラが step 0 の割引率を使うのと同じ扱いである（§4）。
- 0 で割らないための扱い（`epsilon`）、切る範囲（`use_clipping`・`clip_range`）、手動の倍率（`manual_post_scale`）は、今の動的な種類と同じ設定キーで同じに効く。自動の倍率（`use_auto_post_scale = true`）との同時指定は ERROR である（D9）。
- 統計を更新せずに変換する口は、R と σ_R を変えずに、今の σ_R で割った値を返す。
- 状態（lane ごとの R と、σ_R の統計）は checkpoint に保存する（D16）。
- **Burda らの式との違い。** Burda らは誤差そのものを σ_R で割る。本 PRD では、RND の推定器が返す倍率 − 1 を割る。Burda らの形に寄せるには、RND 側を「平均を引かない・切らない」の組（[123](0123_rnd_intrinsic_reward_scale_10rfd.md) §5.3 の 4 行目）にする。割る値が、誤差の標準偏差と割引収益の標準偏差の 2 段になる。誤差の標準偏差がゆっくりしか動かなければ、Burda らの式とほぼ同じ値になる。

### 6.7 Actor

- `decomposed` のとき、Actor は forward の結果に合わせる関数（§6.3）を通し、合わせた Q で行動を選ぶ。合わせるときの β は、`action_beta_mode` が `learner` なら学習と同じ β（Actor が自分の step で評価する）、`fixed` なら `action_beta` の値である（D13）。
- ε・τ・NoisyNet の扱いは今と同じで、合わせた Q（分布型では合わせた分位点）に当てる。
- `decomposed` のとき、Actor は行動を選ぶたびに §6.8 の診断を作り、行動の情報に載せる。

### 6.8 行動の選択への効き方の診断

D26〜D28 による。`decomposed` のとき、Actor が行動を選ぶたびに、その forward の結果から作る。追加の forward はしない。

- Q_外(a) と Q_内(a) は、各 Head の、行動ごとの平均の Q である（分布型では、分位点の平均）。実空間の値を使う（TBO が有効なら、h の逆変換で戻した値）。方策が使うスコア（UQE のスコアなど）は使わない。NoisyNet が有効なら、その forward の値をそのまま使う。
- Q(a) = Q_外(a) ＋ β·Q_内(a) は、合わせた Q である。β は、その Actor の行動の β である（§6.7）。
- a* は Q の 1 位の行動、a_外 は Q_外 の 1 位の行動である。1 位は、同じ argmax の決まりで取る。

| 指標 | lane ごとの値 |
|---|---|
| 不一致 | a* と a_外 が違えば 1、同じなら 0 |
| 譲った外的の価値 | Q_外(a_外) − Q_外(a*)。a* と a_外 が同じなら 0 |
| 外的の差 | Q_外 の 1 位の値 − 2 位の値 |
| 内的の差 | β·Q_内 の 1 位の値 − 2 位の値 |

- lane の平均は、その step の全 lane の平均である。
- 評価の session の累積は、不一致が「1 だった数 ÷ 数えた数」、ほかの 3 つが平均である。数える単位は、Actor が行動を選んだ lane × step である。session の始まりで 0 に戻す。数えた数が 0 のときは NaN である。
- 行動の β が 0 の Actor では、不一致、譲った外的の価値、内的の差が 0 になる。外的の差は β に依らない。
- ε の探索や強制行動で、実際の行動が a* と違っても、値は変えない。
- 分布を使う方策（UQE、Thompson）では、a* が、方策が探索なしで選ぶ行動と違うことがある。診断は、平均の Q で選んだとしたときの値である。
- 行動が 1 つの Env では、不一致と譲った外的の価値は 0、2 つの差は NaN である。行動の β が 0 でも、内的の差は NaN である。
- `decomposed` でないときは、作らない。鍵は NaN を返す（§8.4）。

**例。** 行動が 3 つ、β = 0.01 で、Q_外 = (1.00, 1.02, 0.90)、β·Q_内 = (0.08, 0.05, 0.12) のとき、Q = (1.08, 1.07, 1.02) である。a* は 1 番目、a_外 は 2 番目になる。不一致 = 1。譲った外的の価値 = 1.02 − 1.00 = 0.02。外的の差 = 1.02 − 1.00 = 0.02。内的の差 = 0.12 − 0.08 = 0.04。

## 7. 173 ほかとの関係

### 7.1 アルゴリズム

| | NGU | Agent57 | 本 PRD |
|---|---|---|---|
| 方策の番号 j の入力（UVFA） | あり（one-hot） | あり（2 つの NN の両方に入力） | なし（方策 1 つ） |
| 価値の分解 | なし（j ごとに合わせた報酬で 1 本） | あり（外的・内的の 2 つの NN） | あり（外的・内的の 2 つの Head） |
| 割引率 | j ごとの γ_j | j ごとの γ_j を 2 つの NN で共通に使う | 外的と同じか、内的を別に持つ（D6） |
| 方策の選択 | 均等 | バンディット | なし |

入力側の分け方（j）と、価値の分解は、別の軸である。j は「どの β・γ の組の価値を求めるか」を指定し、価値の分解は「外的と内的を別の出力で学ぶか」を決める。Agent57 は NGU に価値の分解を重ねた。したがって本 PRD は 173 と食い違わず、173 が j を足すときは、2 つの Head の両方に j を入力する。

173 が入るときに置き換えて消すものが 2 つある。

- `extrinsic_bootstrap_mode = extrinsic`（D5）。173 の方策の組では、番号 0（β=0）の方策がこれに当たる。
- Actor ごとの行動の β（D13）。173 が方策の番号で評価する形に決めたら、方策の番号の指定に置き換える。

価値を分けるだけで、全方策の γ や bootstrap の方策の整合が解決するわけではない（173 §4.5）。本 PRD は方策 1 つなので、この問題を持ち込まない。

### 7.2 実装

| 部品 | 本 PRD | 173 で足すもの |
|---|---|---|
| Replay の報酬の成分 | 外的・内的の 2 成分を 1 step ごとに記録し、成分ごとの割引率で n-step の収益を作る（D8） | 方策の番号の保存と、方策ごとに違う γ を同じバッチで使う契約。本 PRD の割引率は成分ごとで、標本ごとではない。抽選時の step の報酬を成分ごとに返す口は、標本ごとの割引率で作り直すときに使える |
| 取り込み | 分解が有効なら、内的報酬を報酬へ足さず、別の成分で渡す | 採点の場所（Actor か取り込みか）が変わっても、成分で渡す形は同じ |
| Q の Head | 内的の Head を足す（D2） | j の入力を Body に足す。2 つの Head の両方が j を受ける |
| 合わせた Q | DQN 側の 1 つの関数が、β で合わせる（D24） | 方策ごとの β_j を、同じ関数に渡す |
| Learner の目標値 | Head ごとの目標値、Munchausen の置き方（D10） | 方策ごとの γ、複数の j の loss |
| 優先度 | Head ごとの誤差を重み付きで足す（D12） | 複数の j の loss を使う場合の集約 |
| 内的報酬の正規化 | 内的報酬用の報酬スケーラ（D9） | エピソード内の新しさにも、同じ報酬スケーラを当てられる |

[116](0116_rnd_lifelong_novelty_10rfd.md) の部品（一生の新しさの口、採点と補助学習）は変えない。

### 7.3 課題の関係

| 課題 | 関係 |
|---|---|
| [issue-20261110-0de9e7](../management/issues/issue-20261110-0de9e7.md) | 本 PRD の仕事 |
| [issue-20261103-736e39](../management/issues/issue-20261103-736e39.md)（173） | 待ちに本 PRD の課題が入っている。173 の再グリルは本 PRD の契約を前提にできる |
| [issue-20261106-16b7af](../management/issues/issue-20261106-16b7af.md)（116） | 内的報酬の出どころ。116 の部品は変えない |
| [issue-20261107-1226e5](../management/issues/issue-20261107-1226e5.md)（Trace で内的報酬を見られない） | 127 が扱う。本 PRD は Head ごとの Q の列だけを足す（D20） |
| [issue-20261103-ab98b7](../management/issues/issue-20261103-ab98b7.md)（TD・QR・IQN の更新の重複） | PRD125 で片づいた |
| [issue-20261103-75b8da](../management/issues/issue-20261103-75b8da.md)（168） | n-step の窓の途中の Munchausen項を扱う。本 PRD の Munchausen の置き方（D10）とは独立 |
| [issue-20261111-08d557](../management/issues/issue-20261111-08d557.md)（内的報酬の扱いが DefaultDQNAgent の 5 関数と 6 メンバに散っている） | 先に片づける（D18）。本 PRD は取り込みに、内的報酬用の報酬スケーラ、成分での受け渡し、保存する状態を足す |
| [issue-20261111-288879](../management/issues/issue-20261111-288879.md)（ExperienceSamples のフィールドの並びが分かれている） | 先に片づける（D18）。本 PRD は成分ごとの収益を足す |
| [issue-20261111-72d640](../management/issues/issue-20261111-72d640.md)（DefaultDQNAgent の構築と、Rainbow に写した Head の選択） | 先に片づける（D18）。本 PRD は Head の選択に合成 Head を足す |
| [issue-20261111-e33797](../management/issues/issue-20261111-e33797.md)（DefaultDQNAgentConfig のコンストラクタが 534 行） | 先に片づける（D18）。本 PRD は設定キーを足す |
| [issue-20261111-24d885](../management/issues/issue-20261111-24d885.md)（診断指標の群を足すたびに 12 か所前後を直す） | 先に片づける（D18）。本 PRD はメトリクスを足す |
| [issue-20261113-b96b2c](../management/issues/issue-20261113-b96b2c.md)（観測の正規化の統計が checkpoint に保存されない） | 本 PRD の範囲の外（§1.4）。本 PRD の報酬スケーラの状態の保存（D16）のあとに、同じ形で足せる |

### 7.4 ほかの PRD との順序

- [128](0128_dqn_agent_refactor_bundle_10rfd.md)（§7.3 のリファクタリングの課題 5 件）を先に実装する（D18）。本 PRD の課題は、その 5 件を待つ。
- 127（Trace の内的報酬の列）と本 PRD は、どちらが先に入ってもよい。あとから入る側が、Trace のボーナスの列を「内的報酬用の報酬スケーラを通した値に β を掛けた値」に合わせる（§8.7）。
- 173 の再グリルは、本 PRD の契約（報酬の成分、内的の Head、合わせる関数、内的報酬用の報酬スケーラ）を前提に、j の入力と方策の組を重ねる。

## 8. 実装契約

### 8.1 設定と失敗の扱い

足す設定キーは次のとおり。どれも `DefaultDQNAgent` の下で、既定は `agent.txt` に書く。

| 設定キー | 型・値 | 読む条件 | 決定 |
|---|---|---|---|
| `intrinsic.value_mode ?= combined` | `combined`｜`decomposed` | 常に | D1 |
| `intrinsic.value.head.use_dueling`・`intrinsic.value.head.init.*`・`intrinsic.value.head.projection.noisy.*` | `head.*` の同じ名前のキーと同じ型。既定は、`agent.txt` の `head.*` の既定と同じ値を書く | `decomposed` のとき | D15 |
| `intrinsic.value.head.input_mode ?= shared` | `shared`（外的の Head と同じ鍵の Body の出力を読む）｜`dedicated`（内的用の鍵の Body の出力を読む） | `decomposed` のとき | D15 |
| `learner.intrinsic.discount_mode ?= shared` | `shared`｜`separate` | `decomposed` のとき | D6 |
| `learner.intrinsic.gamma` | 実数、0 より大きく 1 より小さい。外的の割引率と同じ形で、schedule も書ける。既定値なし | `separate` のとき | D6 |
| `learner.intrinsic.terminal_mode ?= episodic` | `episodic`｜`non_episodic` | `decomposed` のとき | D7 |
| `learner.intrinsic.loss_weight ?= 1.0` | 実数、0 より大きい有限の値 | `decomposed` のとき | D4 |
| `learner.intrinsic.priority_lambda ?= 1.0` | 実数、0 以上の有限の値 | `decomposed` のとき | D12 |
| `learner.intrinsic.extrinsic_bootstrap_mode ?= combined` | `combined`｜`extrinsic` | `decomposed` のとき | D5 |
| `learner.intrinsic.feature_gradient ?= merge` | `merge`｜`split` | `decomposed` のとき | D25 |
| `actor.[<名前>].intrinsic.action_beta_mode ?= learner` | `learner`｜`fixed` | `decomposed` のとき | D13 |
| `actor.[<名前>].intrinsic.action_beta` | 実数、0 以上の有限の値。既定値なし | `fixed` のとき | D13 |
| `reward_scaler.dynamic_mode ?= reward_std` | `reward_std`｜`return_std` | `reward_scaler.use_dynamic_scaling` が真のとき | D9 |
| `intrinsic.reward_scaler.*` | `reward_scaler.*` と同じキーの一式（`dynamic_mode` を含む）。既定は、動的なスケールなし・定数 1・切らない・自動の倍率なし・手動の倍率 1 | `intrinsic.enabled` が真のとき | D9 |

- 分布の表し方（`quantile_mode`）は、外的と内的で共用する。
- 内的の Head が読む Body の出力の鍵は、`input_mode = shared` なら外的の Head と同じ鍵（`features`、Dueling が有効なら `value_feature` と `adv_feature`）、`dedicated` なら `intrinsic_features`（Dueling が有効なら `intrinsic_value_feature` と `intrinsic_adv_feature`）である。Dueling の有無は、内的の Head の設定（`intrinsic.value.head.use_dueling`）で決まる。鍵は Network の設定（`net.body.output.[<鍵>]`）で定める。
- 内的の Head は、読む Body の出力の形を、外的の Head と同じ決まりで検査する（IQN なら τ を混ぜた出力であること、など）。
- Network の出力の鍵は、`decomposed` のとき、外的が `extrinsic_q`・`extrinsic_q_dist`、内的が `intrinsic_q`・`intrinsic_q_dist` である。Dueling の内訳（今の `v`・`a` など）も、同じ接頭辞を付ける。`q`・`q_dist` は Network の出力に無く、合わせる関数が足す（D24）。`combined` のときは、今と同じ鍵である。

判定は、次の表を上から順に当て、最初に当たった行で扱う。どの組み合わせも、最後の行までに必ず当たる。

| # | 条件 | 扱い |
|---|---|---|
| 1 | 上の表の列挙のキーのどれかが、列挙の外の値 | ERROR（キーと値と列挙を示す） |
| 2 | 上の表の数値のキーのどれかが、書かれていて、値域の外か有限でない | ERROR（キーと値と値域を示す） |
| 3 | 使われる報酬スケーラ（外的報酬用。`intrinsic.enabled` が真なら内的報酬用も）のどれかで、`use_dynamic_scaling` が真、`dynamic_mode = return_std`、`use_auto_post_scale` が真の 3 つがそろう | ERROR（報酬スケーラの接頭辞と、3 つのキーを示す） |
| 4 | `intrinsic.value_mode = combined` | 1 本の Q で学ぶ。「読む条件」が `decomposed`・`separate`・`fixed` のキーは使わない。書いた値が実装の既定と違うか、既定値の無いキーが書かれていれば、[117](0117_warn_policy_and_unread_config_10rfd.md) の「書いた値が、ほかの設定のために使われない」の扱いに従う |
| 5 | `decomposed` かつ `intrinsic.enabled = false` | ERROR（内的報酬が無いので分解できない） |
| 6 | `decomposed` かつ `skip.enabled = true` | ERROR（D11） |
| 7 | `decomposed` かつ `discount_mode = separate` で、`learner.intrinsic.gamma` が書かれていない | ERROR（キーを示す） |
| 8 | `decomposed` かつ、どれかの Actor が `action_beta_mode = fixed` で、`action_beta` が書かれていない | ERROR（Actor の名前とキーを示す） |
| 9 | `decomposed` かつ、内的の Head が読む Body の出力の鍵が Network の設定に無いか、形が Head の検査に合わない | ERROR（構築時。足りない鍵か、合わない形を示す） |
| 10 | それ以外（`decomposed`） | 有効。`shared` での `gamma`、`learner` での `action_beta` は使わず、書かれていれば行 4 と同じ扱いに従う |

- 黙った代替はしない。`decomposed` を、内的の Head が組めない構成で今の 1 本の Q に戻さない。
- `reward_scaler.dynamic_mode` と `intrinsic.reward_scaler.*` は、`intrinsic.value_mode` によらず、表の「読む条件」のとおりに読む。`use_dynamic_scaling` が偽のときの `dynamic_mode` と、`intrinsic.enabled` が偽のときの `intrinsic.reward_scaler.*` は使わず、扱いは行 4 と同じである。

### 8.2 Replay の契約（汎用）

D8 による。

- Replay を組むときに、報酬の成分の数を渡す。既定は 1 成分である。
- Push は、1 step ごとに、成分の数だけの報酬と、成分の数だけの割引率を受け取る。
- slot に、成分ごとの n-step の収益と、成分ごとの 1 step の報酬を持つ。
- 抽選は、n-step の収益と、抽選時の step の報酬を、成分ごとに返す。
- 割引率が定数の成分は、Push の時点で収益を保存する。schedule の成分は、抽選のあとに保存済みの step の報酬から作り直す。
- 初期優先度の見積もりへ、成分ごとの収益と、成分ごとの bootstrap の割引を渡す。
- 成分が 2 つ以上の Replay に窓の規則を渡したら、構築時に ERROR にする（成分の数と、窓の規則が指定されたことを示す）。成分が 1 つのときの窓の収益は今と同じである。
- 終端、打ち切り、次観測、実際に進んだ step 数、抽選の分布、item key、PER の completion と eviction 統計の ready の基準、Push と Extract の lock、index の昇順は変えない。
- 成分が 1 つのとき、記録・収益・標本・乱数の消費は今と同じである。`BatchExperience` と `ExperienceSamples` の報酬と収益の形も今と同じで、報酬を 1 本で読む利用側（MuZero 系、ImageClsAgent、Trainer、Observer など）は変えない。
- 成分の意味（外的・内的）、β、終端のマスクは持たない。どの成分を何に使うかは DQN 側が決める。
- DefaultDQNAgent は、`combined` なら 1 成分、`decomposed` なら 2 成分で Replay を組む。

### 8.3 Learner と Actor

- Q の Network の Head を、`decomposed` のとき合成 Head にする（D2）。target の同期、Actor の写し、checkpoint は、今の Network の仕組みに乗る。
- forward の結果を読む経路は、`decomposed` のとき、合わせる関数（§6.3）を通す。読み手は変えない（D24）。
- 目標値は §6.1・§6.2 の式で、Head ごとに作る。
- 損失と優先度は §6.4 の式で作る。`learner.intrinsic.feature_gradient = split` の最適化は、skip の `learner.skip.feature_gradient = split` と同じ作り（損失ごとの backward、損失ごとの clip、共有の重みの損失ごとの Adam の状態）にする（D25）。
- forward の回数と順は、`combined` でも `decomposed` でも今と同じである。backward の回数は、`feature_gradient = split` のときだけ 2 回になる（D17、D22）。
- Actor は §6.7 のとおりに行動を選ぶ。
- 初期優先度の hint は、常に 5 列（Q_sa、V_s、Munchausen項、内的の Q_sa、内的の V_s）で運ぶ（D12）。`decomposed` のとき、前の 3 列は外的の Head の値である。V_s は、Learner の bootstrap（§6.1・§6.2）と同じ決まりを、Actor が行動に使った合わせた Q に当てて、Head ごとに作る。追加の forward はしない。`combined` のとき、前の 3 列は今と同じ値で、内的の 2 列は 0 にする。行動を差し替えるとき（行動の情報の `WithAction`）は、Q_sa と Munchausen項に加えて、内的の Q_sa も選び直す。RainbowAgent の hint も 5 列になり、内的の 2 列は 0 である。
- 初期優先度の見積もりは、`decomposed` のとき、Head ごとに目標値を作って D12 の決まりで足す。`combined` のとき、前の 3 列だけを今と同じに使う。hint の検証は、列の数が 5 で、すべてが有限の値であることを求める。
- Actor は、`decomposed` のとき、§6.8 の診断を作る。評価の session の累積は、skip の session の指標と同じ時機に 0 に戻す。

### 8.4 診断

- Evaluation Trace の `$agent_step` に、次の鍵を足す（D20）。`decomposed` でないときは NaN を返す。

| 鍵 | 値 |
|---|---|
| `extrinsic_q_sa`・`intrinsic_q_sa` | 選んだ行動の、外的の Head と内的の Head の Q |
| `extrinsic_state_value`・`intrinsic_state_value` | 外的の Head と内的の Head の状態の価値（今の `state_value` と同じ決め方を、Head ごとに当てる） |

- 今の `q_sa`・`state_value` は、`decomposed` のとき、その Actor の β で合わせた値を出す。
- 今ある学習のメトリクス（選んだ行動の Q、最大の Q、行動間の差など）は、`decomposed` のとき、学習の β で合わせた Q の値を出す。RND の診断の `bonus_mean`・`bonus_max`・`bonus_min`・`bonus_share`・`bonus_pos_ratio` は、`combined` でも `decomposed` でも、「β × 内的報酬用の報酬スケーラを通した値」から作る。
- メトリクスに次の 5 種を足す。鍵は `$agent` で読む名前である。`decomposed` でないとき、1〜4 は NaN にする。`extrinsic_grad_norm`・`intrinsic_grad_norm` は、`feature_gradient = split` でないとき NaN にする。5 は、今ある `intrinsic.` で始まる鍵と同じく、`intrinsic.enabled` が偽のとき NaN にする。tag の定義は、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt) の組 `metrics.scalar.@value_decomposition` に置く。この組は、どの Run プロファイルの連鎖にも足さない。使う Run が、`metrics.scalar.$` の連鎖に足す（D20）。tag の群の番号は [108](0108_metrics_tag_numbering_10rfd.md) の決まりで、実装のときに空きを取る。

| # | 鍵 | 値 |
|---|---|---|
| 1 | `extrinsic_loss`・`intrinsic_loss` | 外的の Head の損失と、内的の Head の損失（重みを掛ける前） |
| 2 | `extrinsic_q_sa_mean`・`intrinsic_q_sa_mean` | 選んだ行動の Q の平均（外的の Head、内的の Head） |
| 3 | `extrinsic_head_grad_norm`・`intrinsic_head_grad_norm`、`extrinsic_grad_norm`・`intrinsic_grad_norm` | Head ごとの最終の射影のパラメータの、勾配の大きさ（clip の前）。後ろの 2 つは、`feature_gradient = split` のときの、損失ごとの勾配の大きさ（clip の前） |
| 4 | `intrinsic_q_share` | β·Q_内 の絶対値が、「Q_外 の絶対値 ＋ β·Q_内 の絶対値」に占める割合（選んだ行動の値の、バッチの平均）。分母が 0 の標本は 0 とする |
| 5 | `intrinsic.reward_scaler.scale` | 内的報酬用の報酬スケーラの倍率（報酬に掛かっている係数。今の `reward_scaler.scale` と同じ決め方） |

- Actor の行動の情報と、評価の Actor の session の累積に、次の鍵を足す（D26〜D28。定義は §6.8）。`decomposed` でないときは NaN を返す。

| # | 指標 | lane ごとの値の鍵（`$action_info`） | lane の平均の鍵（`@train $action_info`） | 評価の session の累積の鍵（`$eval.[<名前>] $actor @session_end`） |
|---|---|---|---|---|
| 6 | 不一致 | `extrinsic_q_argmax_disagreement` | `extrinsic_q_argmax_disagreement_rate` | `extrinsic_q_session_argmax_disagreement_rate` |
| 7 | 譲った外的の価値 | `extrinsic_q_regret` | `extrinsic_q_regret_mean` | `extrinsic_q_session_regret_mean` |
| 8 | 外的の差 | `extrinsic_q_gap` | `extrinsic_q_gap_mean` | `extrinsic_q_session_gap_mean` |
| 9 | 内的の差 | `intrinsic_q_gap` | `intrinsic_q_gap_mean` | `intrinsic_q_session_gap_mean` |

- lane ごとの値の鍵は、lane を指定して読む。指定した lane が範囲の外なら ERROR にする（skip の lane ごとの鍵と同じ）。Evaluation Trace の列にできる（`app.eval_trace_panel.columns.[<番号>] = $action_info <鍵>`）。既定の列には足さない。
- tag の定義は、1〜5 と同じ組 `metrics.scalar.@value_decomposition` に置く。評価の session の累積の tag は、`$eval.[eval]` と `$eval.[eval_target]` のぶんを書く。ほかの評価（Atari の `[greedy]` など）のぶんは、Atari の枠で足す（D19）。tag は、鍵と別の短い名前にし、群名と `/` を含めて 40 文字以内にする。名前と群の番号は、1〜5 と同じく、実装のときに決める。

### 8.5 checkpoint

D16 による。次の表を上から順に当て、最初に当たった行で扱う。

| # | 状況 | 扱い |
|---|---|---|
| 1 | 設定と checkpoint で、価値の分解の有無が違う | ERROR（設定の値と、checkpoint の側の有無を示す） |
| 2 | 設定と checkpoint が `decomposed` で、`feature_gradient` が違う | ERROR（両方の値を示す） |
| 3 | 報酬スケーラ（外的報酬用か、内的報酬用）の種類が、設定と checkpoint で違う。種類は「状態を持たない（定数の倍率）」「`reward_std`」「`return_std`」の 3 つで、状態の保存の前に作られた checkpoint は「状態を持たない」として比べる | ERROR（報酬スケーラの接頭辞と、両方の種類を示す） |
| 4 | `return_std` の報酬スケーラで、lane の数が設定と checkpoint で違う | 統計を読み、lane ごとの割引和だけ 0 から始める。WARN を 1 回出す（両方の lane の数を示す） |
| 5 | それ以外 | 読む。内的の Head は Network の一部として、報酬スケーラの状態は種類ごとの中身として読む |

- 報酬スケーラが状態を持たない構成（定数の倍率）で、分解も使わない構成の checkpoint は、形が今と同じである。今ある checkpoint をそのまま読める。
- 状態を持つ報酬スケーラを使う構成（LunarLander・GridMaze・CartPole の既定など）の今ある checkpoint は、行 3 で ERROR になる。互換の読み替えは作らない。
- 観測の正規化の統計と Replay は、今と同じく保存の範囲の外である。

### 8.6 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

名前は 2026-11-13 のものである。§7.3 の課題の直しで名前や分け方が変わっていれば、同じ責務を持つ部品に読み替える。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| DefaultDQNAgent | 変更 | 内的報酬を内的報酬用の報酬スケーラに通す。`decomposed` のとき、内的報酬を別の成分で内側の Learner へ渡す。学習の β を更新ごとに Learner へ渡す。2 つの報酬スケーラの状態の保存と読み込み | DefaultDQNAgent（default_dqn_agent） |
| DefaultDQNAgentConfig | 変更 | §8.1 の設定キーの読み取りと判定 | DefaultDQNAgent（default_dqn_agent） |
| BatchExperience・ExperienceSamples | 変更 | 報酬・収益を成分ごとに運ぶ | RL 共通基盤（rl） |
| ReplayBufferConfig・PrefetchingReplayBuffer・ReplayExperienceStorage・DefaultExperienceBuilder・InitialPriorityEstimateInput | 変更 | 成分ごとの step の記録、n-step の収益、抽選時の報酬、見積もりの入力。成分が 2 つ以上のときの窓の規則の拒否。hint の行の型（`ReplayInitialPriorityHintRow`）を、5 列が追加の確保なしで載る要素数にする | ReplayBuffer（replay_buffer） |
| DqnInitialPriorityEstimator | 変更 | Head ごとの見積もりを、優先度の決まりで足す | DQN 系（dqn_based_agent） |
| dqn::Learner・QuantileLearnerBase | 変更 | Head ごとの目標値・損失・優先度。Munchausen の置き方。bootstrap のモード。損失ごとの最適化のモード | DQN 系（dqn_based_agent） |
| dqn::Actor | 変更 | forward の結果に合わせる関数を通し、行動の β で合わせた Q から行動を選ぶ。5 列の hint。行動の選択への効き方の診断（§6.8）を作り、評価の session の累積を持つ | DQN 系（dqn_based_agent） |
| DQNActionInfo | 変更 | 行動を差し替えるときの hint の選び直しに、内的の Q_sa を足す。§6.8 の診断の鍵の読み出し | DQN 系（dqn_based_agent） |
| DqnAgentStepDiagnosticsProvider・DqnAgentStepDiagnosticsValues | 変更 | Head ごとの Q_sa と状態の価値を出す | DQN 系（dqn_based_agent） |
| DecomposedValueCombiner | 新規 | forward の結果の外的と内的の出力から、β で合わせた `q`・`q_dist` を作って足す。TBO の変換を含む | DQN 系（dqn_based_agent） |
| DecomposedHeadFactory | 新規 | 外的の Head と内的の Head を束ねた合成 Head を組む。既存の Head の生成部品を使い、複製しない | DQN 系の Head（dqn_based_heads） |
| RewardScaler・RewardScalerConfig・RewardScalerFactory | 変更 | `dynamic_mode` を読み、種類を選ぶ。状態の保存と読み込みの口 | 報酬スケーラ（scaler） |
| RunningStdRewardScaler | 変更 | 統計の保存と読み込み | 報酬スケーラ（scaler） |
| ReturnStdRewardScaler | 新規 | lane ごとの割引和の標準偏差で報酬を割る。割引和と統計の保存と読み込み | 報酬スケーラ（scaler） |

行動の方策（`ActionPolicy` の派生）と、NoisyNet・policy churn の診断は、`q`・`q_dist` を今のまま読むので変えない（D24）。

### 8.7 移行対象

経路の検索（2026-11-12〜13、`git grep -il -F`、範囲は追跡している `core`・`apps`・`docs`・`viewers`・`reports`）: `target_returns`・`InitialPriorityEstimateInput`・`sampled_rewards`・`sampled_window_rewards`・`FinalizeReturns`・`GetStepReward`・`PackActorQHint`・`ActorQHint`・`CreateRewardScaler`・`use_dynamic_scaling`・`RewardScalerConfig`・`ScaleWithoutUpdate`・`報酬へ畳む`・`munchausen_terms`・`MakeBellmanTarget`・`CombinePriorityErrors`・`HeadFactory`・`RunningStdRewardScaler`・`kCheckpointFormat`・`kCheckpointIntrinsic`・`skip_primary_optimizer_`・`kFeatureGradientSplit`・`kQOutputKey`・`state_value`・`kActorQHintColumnCount`・`ReplayInitialPriorityHintRow`・`BatchExperience`・`iqn_uqe_full_q_argmax_disagreement`・`skip_session_decision_rate`・`ResetSkipSession` のどれかを含むファイルと、`ReplayBuffer` を継承するクラス（正規表現 `public (anet::)?(rl::)?ReplayBuffer\b` で引いた。本体の 2 つと、テスト用の 3 つ）のあるファイル、`Push(` を呼ぶファイル（正規表現 `(->|\.|::)Push\(`、範囲は `core`・`apps`。11 ファイルで、うち Atari の Env の 3 つは別の型の `Push` である）から、過去の PRD・実装計画・ADR・実験記録・課題・wiki・サーベイ（書き換えない）を除くと、現用のものは次のとおり。

- **コード**: [nn.hpp](../../src/core/base/include/anet/nn.hpp)・[nn_impl.cpp](../../src/core/base/src/nn_impl.cpp)・[nn_modules.cpp](../../src/core/base/src/nn_modules.cpp)・[rl.hpp](../../src/core/base/include/anet/rl.hpp)・[rl.cpp](../../src/core/base/src/rl.cpp)・[replay_buffer.hpp](../../src/core/base/include/anet/replay_buffer.hpp)・[replay_buffer_impl.hpp](../../src/core/base/src/replay_buffer_impl.hpp)・[replay_buffer_impl.cpp](../../src/core/base/src/replay_buffer_impl.cpp)・[dqn_based_agent.hpp](../../src/core/base/src/dqn_based_agent.hpp)・[dqn_based_agent.cpp](../../src/core/base/src/dqn_based_agent.cpp)・[dqn_based_heads.hpp](../../src/algo/dqn/src/dqn_based_heads.hpp)・[dqn_based_heads.cpp](../../src/algo/dqn/src/dqn_based_heads.cpp)・[default_dqn_agent.hpp](../../src/algo/dqn/src/default_dqn_agent.hpp)・[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp)・[scaler.hpp](../../src/core/base/include/anet/scaler.hpp)・[scaler_impl.hpp](../../src/core/base/src/scaler_impl.hpp)・[scaler_impl.cpp](../../src/core/base/src/scaler_impl.cpp)。dqn_based_agent.cpp では、forward の結果を読む経路に合わせる関数を通す。`q`・`q_dist` の読み手 17 か所は変えない。hint の列の数（`kActorQHintColumnCount`）と、hint を詰める・検証する・ほどく箇所（dqn_based_agent.hpp・dqn_based_agent.cpp）、hint の行の型（replay_buffer_impl.hpp の `ReplayInitialPriorityHintRow`）を、5 列に合わせる。
- **口が変われば呼び出しを合わせるコード（動きは変えない）**: [rainbow_agent.cpp](../../src/algo/dqn/src/rainbow_agent.cpp)（Head の生成部品と、Actor の hint の設定を使う側。RainbowAgent は内的報酬を持たない）、[agent.hpp](../../src/core/base/include/anet/agent.hpp)（checkpoint の書式の識別子を使う側）。
- **検索に当たるが変えないコード**: [nn_impl.hpp](../../src/core/base/src/nn_impl.hpp)・[nn_heads.hpp](../../src/core/base/include/anet/nn_head.hpp)・[nn_heads.cpp](../../src/core/base/src/nn_head.cpp)・[nn_test.cpp](../../src/core/base/src/nn_test.cpp)・[intrinsic_rnd.cpp](../../src/core/base/src/intrinsic_rnd.cpp)・[muzero_based_agent.cpp](../../src/algo/muzero/src/muzero_based_agent.cpp)・[image_cls_agent.cpp](../../src/algo/imagecls/src/image_cls_agent.cpp)・[optuna_common.py](../../apps/tools/optuna_common.py)・[config_test.cpp](../../src/core/base/src/config_test.cpp)。Network が Head を 1 つ持つ作り、Head の生成部品の口、RND の推定器、今ある Trace の鍵の名前は変えない（D2、D9、D20）。`BatchExperience` を含むファイルは 41 ある。上に挙げたもののほかは、報酬を 1 本で読むか、型を受け渡すだけの利用側（Trainer、Observer、probe、metrics、image、ImageClsAgent、Env の View、Runner のテストと、それぞれのテスト）で、1 成分のときの形が今と同じなので変えない（§8.2）。MuZero 系（[muzero_rb.hpp](../../src/algo/muzero/src/muzero_rb.hpp)・[muzero_rb.cpp](../../src/algo/muzero/src/muzero_rb.cpp)・[muzero_based_agent.hpp](../../src/algo/muzero/src/muzero_based_agent.hpp)・muzero_based_agent.cpp・[prd101_muzero_test.cpp](../../src/algo/muzero/src/rfd0101_muzero_test.cpp)）は、`MuZeroReplayBuffer` が `ReplayBuffer` を継承せず、自分の `Push` を持つので変えない。
- **テスト**: [replay_buffer_test.cpp](../../src/core/base/src/replay_buffer_test.cpp)・[dqn_based_agent_test.cpp](../../src/core/base/src/dqn_based_agent_test.cpp)・[dqn_based_test.cpp](../../src/algo/dqn/src/dqn_based_test.cpp)・[dqn_munchausen_test.cpp](../../src/algo/dqn/src/dqn_munchausen_test.cpp)・[intrinsic_test.cpp](../../src/core/base/src/intrinsic_test.cpp)・[scaler_test.cpp](../../src/core/base/src/scaler_test.cpp)・[trainer_test.cpp](../../src/core/base/src/trainer_test.cpp)・[config_baseline_test.cpp](../../src/core/base/src/config_baseline_test.cpp)（出荷の設定を読むテストを足す。§9.2 の 17・23）。Push の口が変わるので、Push を呼ぶテストと、`ReplayBuffer` を継承して `Push` を上書きするテスト用のクラス（dqn_based_agent_test.cpp の `RecordingReplayBuffer`、replay_buffer_test.cpp の `BlockingReplayBuffer`、trainer_test.cpp の `HintRecordingReplayBuffer`）は、新しい口に合わせる。hint を 3 列で断定するか、3 列で組んでいるテスト（dqn_based_agent_test.cpp の、`kActorQHintColumnCount` を使う断定と、hint の行を手で組む箇所）は、5 列に合わせる。1 成分のときの値は今と同じなので、既存の断定の期待値は変えない。checkpoint の書式と中身を断定するテスト（dqn_based_agent_test.cpp の `[dqn][serialize]` のタグのテスト、intrinsic_test.cpp の `[checkpoint]` のタグのテスト。`DefaultDQN checkpoint headers identify Agent serialization formats` など）は、動的な報酬スケーラを使う構成の checkpoint に報酬スケーラの状態が入ることに合わせて、断定を直す。設定の型の既定は動的な種類なので、既定の設定で Agent を作って保存するテストが当たる。§9.2 の受入のテストを足す。
- **設定**: [agent.txt](../../apps/runner/config/agent.txt)（§8.1 の設定キーの既定）、[metrics_scalar.txt](../../apps/runner/config/metrics_scalar.txt)（組 `metrics.scalar.@value_decomposition` を足す。どの Run プロファイルの連鎖にも足さない。§8.4）。[Atari.txt](../../apps/runner/config/Atari.txt)・[DropMerge.txt](../../apps/runner/config/DropMerge.txt)・[GridMaze.txt](../../apps/runner/config/GridMaze.txt)・[common.txt](../../apps/runner/config/common.txt) は、検索に当たるが変えない（今ある報酬スケーラのキーは、新しい既定で今と同じ値になる。Trace の既定の列は足さない。`actor_approx` を指定する 2 つの Run プロファイルも、hint の列の数が設定に現れないので変えない）。[12_batch_run.bat](../../apps/12_batch_run.bat) は、skip の session の指標を引数で指定しているので検索に当たるが、変えない。
- **設定の検証資材**: 現用17構成（Atari 5、DropMerge 3、LunarLander 3、GridMaze 2、CartPole、GridMaze MuZero、ImageCls 2）を、変更前後で全キー・文字列値を比較する。参照側は現用設定の写しから`agent.txt`と`metrics_scalar.txt`だけを本PRDの実装前へ戻し、並行するEnv設定の変更を両側で同じにする。既存キーの削除・値の変更を認めず、追加は本PRDが定めた既定キーとその既定値だけを許す。既存scalar定義も一致させる。足すメトリクスの組はどの構成の連鎖にも入らないので、解決値には現れない。手順と比較結果の形式は`core/anet-core/testdata/prd121/README.md`と`run_validation.py`に置く。`core/anet-core/testdata/prd072/baseline/`は固定commitの履歴で、現用で廃止されたプロファイルを含むため保持し、更新しない。`core/anet-core/testdata/prd084/`・`prd092/`・`prd094/`の資材も過去の検証の記録として保持する。
- **設計文書**: `docs/design/214_value_decomposition.jp.md` を新設する（価値の分解の仕組み、式、合わせる関数、設定、診断とメトリクスの組、checkpoint。可塑性の診断で、内的の Head の重みが readout の側に入ることも書く）。[140_observability.jp.md](../design/140_observability.jp.md)（`$agent_step` の鍵と、報酬スケーラの診断の鍵の説明に、足す鍵と種類。行動の情報の鍵と、評価の session の累積の鍵の説明に、§8.4 の 6〜9）、[150_replay_buffer.jp.md](../design/150_replay_buffer.jp.md)（§2.1・§2.2・§2.4・§6.1・§6.1.1・§7.6・§8 に報酬の成分）、[200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md)（§2.5 の Actor の hint の列を 5 つに。§6.2・§6.3・§7.2・§8.2・§9.1・§10 に価値の分解と、214 への参照。§8.2 の「報酬スケーラの統計を保存しない」の記述を、保存する形に直す）、[212_rnd_lifelong_novelty.jp.md](../design/212_rnd_lifelong_novelty.jp.md)（§3.6・§3.7 に、内的報酬用の報酬スケーラと成分での受け渡し）、[020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md)（Evaluation Trace の鍵を挙げている箇所に、足す鍵）、[030_user_guide_analysis.jp.md](../design/030_user_guide_analysis.jp.md)（方策の診断の読み方を書いている箇所に、§8.4 の 6〜9 の読み方）。[130_neural_networks.jp.md](../design/130_neural_networks.jp.md) は、検索に当たるが、Network と Head の契約を変えないので変えない。en は `anet-translate-docs` で作る。
- **検証資材**: `core/anet-core/testdata/prd121/README.md` を新設する（実装前の実行体の保存と、無効のときの metrics checksum の比較の手順。§9.2 の 1）。200 の §10 から参照する。
- **127 との調整**: 127 が先に入っていれば、Trace のボーナスの列（`intrinsic.bonus`）の計算を、内的報酬用の報酬スケーラを通した値（統計を更新しない口）に β を掛けた値に合わせる。127 があとから入るなら、127 の実装がこの形に合わせる。
- **用語集と ADR**: [CONTEXT.md](../../CONTEXT.md) の「価値の分解」「報酬の成分」、ADR 0073・0074・0075、ADR 0036・0062・0065 の先頭の注記は、PRD づくりで済ませた（D21）。[CONTEXT.md](../../CONTEXT.md) の「Actor Qヒント」の項目は、実装のときに、運ぶ中身の説明へ内的の Head の Q_sa と V_s を足す。実装が入るまでは、今の 3 つの中身の説明が正しいためである。
- **互換層**: 作らない（D23）。

## 9. 成功の測り方と受入

### 9.1 効果（Atari の枠で確かめる）

効果は §1.3 の仮説 A・B で確かめる。Run の組み立ては Atari の枠で決める（D19）。次は群を組むときの材料で、決定ではない。

- **そろえる条件。** PRD123 の新しい RND の条件にする。Montezuma では Munchausen 無効・Double DQN 有効にする（Munchausen 有効は Greedy の 2500 点が 5/160 だった。§2.1）。
- **seed。** 同じ構成でも seed 1 と 2 で Greedy の 2500 点が 99/160 と 81/160 に分かれたので、群ごとに 2 本以上にし、この幅を超える差だけを読む。
- **仮説 A の群。** 1 本の Q と分解した Q（`discount_mode = shared`・`terminal_mode = episodic`・`extrinsic_bootstrap_mode = combined`）を、β 2 点以上で。分解した側は、次のような群を並べると、共有の Body への影響を分けて読める。損失の重みが 1.0 の群、損失の重みを β に近い値にした群、`feature_gradient = split` の群、内的報酬用の報酬スケーラの定数の倍率で内的の目標値の桁を外的に合わせた群（損失の効く範囲をそろえる。§6.4）。
- **仮説 B の群。** 分解した Q で、内的を外的と同じ扱いにした群と、`separate`・`non_episodic` にした群。内的報酬用の報酬スケーラを `return_std` にし、`intrinsic.use_done_mask` を偽にした群を足すと、Burda らの形に寄る。
- **読む指標。** 評価側（Greedy と ε 0.01）の 2500 点の割合と平均点、訪れた部屋の種類数、Head ごとの Q（Trace とメトリクス）。行動の β を 0 にした評価を足すと、2500 点のあとの局面が外的の Head にもあるかを分けられる。
- **行動の選択への効き方（D26）。** 群どうしで、評価の Actor の不一致率、譲った外的の価値、2 つの差を比べる。点数が β で変わらず、不一致率がどの群でも 0 に近く、内的の差が外的の差よりずっと小さければ、内的が行動を動かせていない。不一致率が高いのに点数が同じなら、動かしているが効いていない。譲った外的の価値は、外的の差の平均に対する比で読む。比が大きい群は、β が大きい側にある。群どうしで値をそのまま比べられるのは、外的報酬の報酬スケーラが同じ（定数の倍率）ときだけである。2500 点のあとの平らな局面では、Trace の列で、その step の 4 つの値を読める。
- **途中の時期の差。** 鍵や扉に届いた時期の差は、Run ごとの揺れの内とみなし、判定に使わない（起点の方向 (1)）。

### 9.2 正しさの受入

特に書かない限り、単体テストで確かめる。値の一致は、float32 の計算で相対誤差 1e-5 以内とする。

| # | 観点 | 観測する結果 |
|---|---|---|
| 1 | 無効のときの同一性 | `intrinsic.value_mode` と報酬スケーラの新しい設定キーが既定のとき、実装前に保存した実行体と、seed 1・決定的モードの短い Run の metrics checksum が一致する。対象の構成は、Atari の `run.@rnd`、LunarLander の RND あり・なし、GridMaze の既定、DropMerge の既定の 5 つである。Run の長さは Env の step で数え、構成ごとに、学習の更新と評価がそれぞれ 1 回以上起き、target を一定の間隔で同期する構成では同期も 1 回以上起きる長さにする。構成ごとの step 数は、検証資材の README に書く。比べるのは、Run の最初から最後までの全 scalar の点で、時間と資源の tag は除く（実時間で変わるため）。足すメトリクスの組は既定では選ばれないので（§8.4）、比べる tag の集合は前後で同じである |
| 2 | 1 成分の Replay | 成分が 1 つの Replay で、[replay_buffer_test.cpp](../../src/core/base/src/replay_buffer_test.cpp) の既存の断定が、期待値を変えずにすべて通る |
| 3 | 成分ごとの収益 | 2 成分・割引率 2 つの Replay に既知の step の報酬を入れると、成分ごとの n-step の収益が手計算と一致する。定数と schedule の両方、終端と打ち切りで窓が短くなる場合を含む。成分が 2 つの Replay に窓の規則を渡すと、構築時に ERROR になる |
| 4 | 今の 1 本の Q との一致 | `discount_mode = shared`・`terminal_mode = episodic`・`extrinsic_bootstrap_mode = combined` で、既知の値を入れた同じ標本について、2 つの目標値を β で合わせた値が、`intrinsic.value_mode = combined` の 1 本の Q の目標値と一致する。TD・QR・IQN のそれぞれで、Munchausen 無効と有効の両方を確かめる。分布型では同じ τ どうしを比べる。TBO が有効の場合は、h を掛ける前の値で比べる |
| 5 | 終端の扱い | `non_episodic` のとき、終端の遷移で、内的の Head の目標値に次観測からの bootstrap が入り、外的の Head には入らない。終端と打ち切りが同時に真の遷移では、どちらの Head にも入らない。打ち切りだけの遷移では、どちらのモードでも両方に入る |
| 6 | bootstrap のモード | 合わせた Q の argmax と外的の Q の argmax が違う標本で、`extrinsic` のとき外的の目標値が外的の Q の argmax の行動を使い、内的の目標値は合わせた Q の argmax の行動のままである。Double DQN の有効と無効の両方を確かめる |
| 7 | 内的の割引率 | `separate` のとき、内的の目標値の収益と bootstrap が `learner.intrinsic.gamma` で作られ、外的は外的の割引率のままである |
| 8 | 損失の重み | `feature_gradient = merge` で、既知の値の損失が「外的 ＋ w × 内的」になる（w が 1.0 と 0.5） |
| 9 | 損失ごとの最適化 | `feature_gradient = split`・weight decay 0 で 1 回更新すると、共有の重みの変化が「外的の損失だけで clip と Adam の更新をした変化」と「内的の損失だけで clip と Adam の更新をした変化」の和と一致する。片方の損失の勾配だけが clip の上限を超える値で確かめ、もう片方の更新が縮まないことを確かめる |
| 10 | 優先度 | 既知の値で、優先度が「外的の誤差の絶対値 ＋ λ × 内的の誤差の絶対値」になる（λ が 1.0・0.2・0）。2 つの誤差が逆向きでも打ち消さない |
| 11 | 合わせた Q | `decomposed` の forward の結果に `q`・`q_dist` が無く、合わせる関数を通したあとの `q` が「外的 ＋ β × 内的」と一致する。分布型では同じ τ どうしの和と一致する。TBO が有効のとき、各 Head の出力を h の逆変換で戻して β で合わせ、h を掛け直した値と一致する |
| 12 | 行動の β | `fixed`・値 0 の Actor の greedy の行動が、外的の Head の argmax と一致する。`learner` の Actor は、学習の β で合わせた Q の argmax と一致する |
| 13 | 設定の判定 | §8.1 の判定の表の各行が、表のとおりに扱われる。ERROR の文に、対象のキーと値が入る |
| 14 | 割引収益の標準偏差で割る報酬スケーラ | 2 lane に既知の報酬の列を入れると、lane ごとの割引和、標準偏差、割った値が手計算と一致する。エピソードの終わりをまたいでも割引和が続く。統計を更新しない口は、割引和と統計を変えない |
| 15 | 内的報酬用の報酬スケーラ | 既定では、通した値が入力と一致する。`combined` で `return_std` のとき、学習へ渡る報酬が「外的 ＋ β × 通した値」になる。`decomposed` のとき、内的の成分が通した値（β を掛けない）になる。done の lane の扱いが、報酬スケーラに通す前に当たっている |
| 16 | Trace の鍵 | `decomposed` の Actor の step 診断が、§8.4 の 4 つの鍵に値を返し、`q_sa` が「外的 ＋ その Actor の β × 内的」と一致する。`combined` では 4 つの鍵が NaN になる |
| 17 | メトリクス | `decomposed` の小さな Agent を 1 回以上更新したあと、§8.4 の 1〜5 の鍵を読むと、`feature_gradient = merge` では `extrinsic_grad_norm`・`intrinsic_grad_norm` だけが NaN で、ほかはすべて有限の値である。`feature_gradient = split` では、すべて有限の値である。`intrinsic.value_mode = combined`・`intrinsic.enabled = true` では、1〜4 が NaN で、5 が有限の値である。今ある学習のメトリクスの選んだ行動の Q が、学習の β で合わせた Q の値と一致する。出荷の設定を読むテスト（`[shipped_config]` のタグ）で、GridMaze の既定に、`run.@rnd`、`intrinsic.value_mode = decomposed`、`metrics.scalar.$` の連鎖への `metrics.scalar.@value_decomposition` を足した構成が解決でき、その組の tag の鍵が、どれも §8.4 の鍵である（GridMaze の既定は skip が無効なので、§8.1 の判定の行 6 に当たらない） |
| 18 | checkpoint | §8.5 の表の各行が、表のとおりに扱われる。保存して読み直すと、内的の Head の出力と、外的報酬用・内的報酬用の報酬スケーラの変換の結果が、保存の前と一致する（`reward_std` と `return_std` の両方）。実装前の実行体が保存した、定数の倍率の構成の checkpoint を、新しい実行体が読める |
| 19 | 初期優先度 | 初期優先度を Actor の hint から見積もる構成で、hint が `combined` でも `decomposed` でも 5 列である。`combined` のとき、内的の 2 列が 0 で、見積もりの値が今と同じである（既存の断定の期待値が変わらない）。`decomposed` のとき、hint が Head ごとの値を運び、見積もりが優先度の決まり（#10）と同じ形になる。列の数が 5 でない hint は ERROR になる |
| 20 | 性能の記録 | 実装計画に、Atari の同じ構成の実時間を、`combined` と `decomposed`（Body を共有、`feature_gradient = merge`）で交互に 3 巡以上測った値が書かれている。閾値は置かない |
| 21 | 文書 | §8.7 の設計文書・検証資材・用語集・ADR がそろっている |
| 22 | 行動の選択への効き方の診断 | 行動が 3 つ以上の、既知の Q_外・Q_内・β で、§6.8 の 4 つの指標の lane ごとの値（lane を指定して読む）と、lane の平均が、手計算と一致する。lane が 2 つ以上で、lane ごとに値が違う入力で確かめる。TD・QR・IQN のそれぞれで確かめ、分布型では分位点の平均で比べる。TBO が有効のとき、h の逆変換で戻した値での手計算と一致する。行動の β が 0 の Actor では、不一致、譲った外的の価値、内的の差が 0 になる。`combined` では、§8.4 の 6〜9 の鍵がすべて NaN になる。行動を差し替えても、値が変わらない。lane ごとの値の鍵を、範囲の外の lane で読むと ERROR になる |
| 23 | 評価の session の累積 | 評価の Actor を既知の値で数 step 進めると、session の終わりに読む 4 つの鍵が、その session の lane × step の値から作った値（不一致は割合、ほかは平均）と一致する。1 step も進めていない session では NaN になる。次の session の始まりで 0 に戻る。出荷の設定を読むテスト（`[shipped_config]` のタグ）で、組 `metrics.scalar.@value_decomposition` の tag の鍵に、§8.4 の 6〜9 の lane の平均の鍵と、`[eval]`・`[eval_target]` の session の累積の鍵が入っていて、どの tag も 40 文字以内である |

人が確かめる項: EvalPanel の Evaluation Trace に、プロファイルで足した Head ごとの列と、診断の列（§8.4 の 6〜9 の lane ごとの値の鍵）が表示される。評価の lane が 2 つ以上のとき、行ごとに、その lane の値が出る。

## 10. 決めること（未決表）

なし。起票時の 21 行は §5.2 のとおりに移した。先送りにした形と作る条件は §5.1 にある。

## 11. 参照

- Badia ら、[Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/abs/2003.13350)、ICML 2020。§2（NGU の要約と不安定さ）、§3.1（価値の分解）、§4.2（分解の効果）、付録 B（等価）、付録 E（優先度）。
- Badia ら、[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038)、ICLR 2020。
- Burda ら、[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)、ICLR 2019。§2.3（内的と外的の収益の組み方）、§3.2（図 4）、§3.3（図 5）。
- Taiga ら、[On Bonus-Based Exploration Methods in the Arcade Learning Environment](https://arxiv.org/abs/2109.11052)、ICLR 2020。図 7。
- [deep_rl_zoo](https://github.com/michaelnny/deep_rl_zoo)（Agent57 の公開の再実装。優先度の作り方の事例）。
