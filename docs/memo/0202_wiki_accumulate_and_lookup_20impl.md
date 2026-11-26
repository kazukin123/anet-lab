# wiki に知見を積み上げ、引けるようにする 実装メモ（成功の測り方とステージ 4）

## 概要

[RFD0202](0202_wiki_accumulate_and_lookup_10rfd.md) と [issue-20261125-822a3e](../management/issues/issue-20261125-822a3e.md) に対応する。RFD の D18 と §9.10 が Claude Code の担当に置いた、成功の測り方（§1.5 の問いの指標と正しさの指標）とステージ 4（RFD の段 4）の実装メモである。ステージ 1〜3 は Codex の担当で、その計画は Codex が起こす実装計画に書く。

この版は、ステージ 1 が wiki を変える前に行う作業（RFD §9.10 の最後の箇条）と、ステージ 2 の終わり・ステージ 3 の終わりの確かめの記録である。問いの指標に使う問いを選び、実装の前の wiki に判定を付けた。ステージ 2 の終わりに、横断的知見の正しさを確かめた。ステージ 3 の終わりに、同じ問いに判定を付け直した。残りの作業は「これから行うこと」に並べる。

## 主な変更

- このファイルを足した。受入 A21 の記録のうち、選んだ問いの一覧と、実装の前の判定を持つ。
- RFD の問いの指標で比べる数を、「答えがある」の数に改めた（冒頭の改訂の行、§1.4・§1.5・S30・§13）。
- 正しさの指標に、横断的知見 20 件の判定と根拠、誤りの率（15%）を足した。
- 問いの指標に、ステージ 3 の終わりの判定と根拠を足した（答えがある 4、答えの種がある 12、無い 0。「答えがある」は実装の前から増えていない）。
- wiki のページ・wiki の規約・skill・道具は変えていない。

## 問いの指標

### 問いの出し方

- 出どころは、2026-10-19 以降の Claude Code の会話での、開発者の質問である（RFD §5.2）。質問らしい発言のうち、AI が道具を 5 回以上使ったものを、開発者の発言の行だけ引いて読んだ。
- 残さないと断りのある会話は、丸ごと外した。Run の段取り、状況の確認、画面や道具の使い方、ハーネスの質問は、材料をまたぐ問いでないので外した。
- 問いの文は要旨で、AI の過去の答えの数値は写していない。
- AI が候補を 29 個出し、2026-11-26 に開発者が 16 個を選んだ（RFD §8.9 の K3 の確認に当たる）。番号は候補のときのもので、抜けている番号は選ばれなかった候補である。

### 選んだ問い

| 番号 | 出どころ（開発者の質問の日付） | 題材 | 問い |
|---|---|---|---|
| Q1 | 2026-11-04 | TempoRL（LunarLander・qbert・DropMerge） | TempoRL が LunarLander では効き、Atari と DropMerge では効かなかったのはなぜか。得られた教訓は何か |
| Q2 | 2026-11-04 | 重みの初期化（Atari・DropMerge） | Atari で見られた「初期値を PyTorch 既定に戻すと立ち上がりと成績が上がる」は、DropMerge に反映済みか。機序と、反映の実験の状況は |
| Q3 | 2026-11-08・11-19 | batch size（DropMerge・Atari） | DropMerge では batch を大きくするほうがよいと分かった。Atari に当てる試しは済んだか、どうだったか |
| Q5 | 2026-11-10・11-13 | RND（KFM・Montezuma・DropMerge） | Montezuma で効いた RND が、KFM で効果が薄いのはなぜか。DropMerge に当てるとどうなりそうか |
| Q6 | 2026-11-19 | replay ratio（Atari） | RR4 で探索して絞った設定を RR8 で試す方針は成り立つか。RR4 と RR8 で、設定変更の効き方の傾向は似ているか |
| Q7 | 2026-10-29 | 評価の読み方（qbert） | Train では 8 面を抜けないのに Greedy では抜ける。この差は何から来るか（ε、NoisyNet、ほか） |
| Q10 | 2026-11-21 | 部屋 2 の停滞と n-step（Montezuma） | 部屋 2 でうろうろするのはなぜか。n-step の N を上げると変わるか |
| Q11 | 2026-11-11 | ボス行き（KFM） | ボスに行き着くことを、偶然に頼らず起こすには。エピソード内の新しさでも解決できないのではないか |
| Q14 | 2026-11-16 | BTR の再現の経過 | BTR の再現では、何が突破口で、何が悪さをしていたか |
| Q18 | 2026-11-11・11-19 | RND の実装 | anet の RND は、元の RND 論文・Taiga ら・NGU のどれとも違う独自の形か。足りない要素は何か |
| Q20 | 2026-11-07 | RND のリセット | RND のネットワークを定期的に初期化する事例はあるか。anet で試した結果は |
| Q21 | 2026-11-10 | Munchausen と RND | Munchausen は RND と組むと悪さをするのか。外すと BTR の改善の大きなところを失わないか |
| Q22 | 2026-11-10 | 探索の報酬の崩壊と可塑性 | 進歩が止まると探索の報酬が崩れうるという論文の指摘は、anet の Run で起きているか。可塑性と関係があるか |
| Q23 | 2026-11-12・11-15 | 価値の分解 | 外的と内的の価値を分ける Head は、一般の事例ではどう組むか（最終段だけ分けるのは普通か）。割引率を分けると n-step と Replay はどう変わるか |
| Q24 | 2026-11-17 | 内的報酬の全体 | RND のほかに ICM などがある。全体を見渡した整理は、今どこにあるか |
| Q27 | 2026-11-15 | Replay の n-step | Push のときに計算する方式と Sample のときに計算する方式を、1 つにまとめないのはなぜか |

### 判定の基準

RFD §1.5 は 3 つの語だけを定めるので、次の基準を置いた。ステージ 3 の終わりにも、同じ基準で付ける。

| 判定 | 基準 |
|---|---|
| 答えがある | 1 つのページの「今の結論」か、1〜2 個の項目が、問いの全部にそのまま答えている |
| 答えの種がある | 答えに要る事実は wiki にあるが、3 個以上の項目か 2 つ以上のページを、読む人が突き合わせる必要がある。または、問いの一部に当たる記述が欠けている |
| 無い | 答えの材料になる記述が wiki に無い |

### 実装の前の判定（2026-11-26）

当てたのは、ステージ 1 の前の `docs/wiki/` である。根拠にしたページの内容は、コミット d39d2f07 と同じである。リンクは、ステージ 1 で移したあとの置き場を指す（判定したのは、移す前の同じページである）。

答えがある 4、答えの種がある 12、無い 0。判定は AI が付け、開発者が見た。「ステージ 3 の終わり」の列の根拠は、下の「ステージ 3 の終わりの判定」にある。

| 番号 | 実装の前 | 根拠のページと節 | 欠けているもの | ステージ 3 の終わり |
|---|---|---|---|---|
| Q1 | 答えの種がある | [行動の時間尺度](../wiki/concepts/methods/temporal_abstraction.md)「今の結論」「分かっていないこと」 | 題材の違いを並べた読み取りと教訓。Atari と DropMerge の理由は、未確認と書かれている | 答えの種がある |
| Q2 | 答えの種がある | [Qbert](../wiki/concepts/subjects/qbert.md)「今の結論」「分かっていること」（初期値の比較）「分かっていないこと」、[Breakout](../wiki/concepts/subjects/breakout.md)「分かっていること」（Dopamine の IQN に寄せた構成の項目） | DropMerge に反映したかどうか（[DropMerge](../wiki/concepts/subjects/dropmerge.md) に初期値の記述が無い）と、機序 | 答えの種がある |
| Q3 | 答えの種がある | [勾配ノイズとbatch](../wiki/concepts/phenomena/gradient_noise_batch.md)「今の結論」「分かっていること」、[kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「分かっていること」（batch 1024 の 3 seed の項目） | DropMerge の知見を Atari に当てた結果としての並び。Atari の側は、KFM のボス行きの観点だけ | 答えの種がある |
| Q5 | 答えの種がある | [kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「今の結論」「分かっていること」（RND の 4 項目）、[内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「今の結論」 | Montezuma と並べた理由の読み取り。DropMerge と RND の記述は無い | 答えの種がある |
| Q6 | 答えの種がある | [Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)「今の結論」「分かっていること」、[Breakout](../wiki/concepts/subjects/breakout.md)「今の結論」、[可塑性指標と成績](../wiki/concepts/phenomena/plasticity.md)「今の結論」 | RR4 で選んだ設定が RR8 でも同じ向きに効くかを並べた記述 | 答えの種がある |
| Q7 | 答えがある | [評価と比較の条件](../wiki/concepts/measurement/evaluation.md)「分かっていること」（qbert の Greedy と、学習中と同じ選び方の比較の項目）、[Qbert](../wiki/concepts/subjects/qbert.md)「分かっていること」（Q 側・skip 側のノイズと τ を分けた 2 項目） | — | 答えがある |
| Q10 | 答えの種がある | [Montezuma's Revenge](../wiki/concepts/subjects/montezuma_revenge.md)「分かっていること」（部屋 2 の梯子の下で止まる項目、割引率の分離の形の 9 本の項目）、[Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)「分かっていること」（n-step を 10 にした項目） | n-step を上げると部屋 2 の停滞が変わるかを、直接書いた記述 | 答えの種がある |
| Q11 | 答えの種がある | [kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「分かっていること」（既定の設定の 16 seed の項目、挑み始めるきっかけの項目、方策の多様性の項目、文献上の留保）「分かっていないこと」 | 起こす手段を並べた記述。エピソード内の新しさで足りるかは、問いのまま | 答えの種がある |
| Q14 | 答えがある | [Qbert](../wiki/concepts/subjects/qbert.md)「今の結論」「分かっていること」（初期値の比較、BTR へ寄せた 50M の項目）。補い: [Atari横断](../wiki/concepts/subjects/atari.md)「分かっていること」（観測の縮小の項目） | — | 答えがある |
| Q18 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」の「比較対象の実装」の 7 項目 | 項目が分かれていて、独自かどうかと足りない要素をまとめた記述が無い | 答えの種がある |
| Q20 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「今の結論」「分かっていること」（LunarLander の RND の 2 項目。predictor の周期初期化を含む）「分かっていないこと」 | 外の事例があるかどうか | 答えの種がある |
| Q21 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「今の結論」「分かっていること」「撤回・更新」（2026-11-12）、[Breakout](../wiki/concepts/subjects/breakout.md)「分かっていること」（Munchausen RL の項目） | 外したときに失うものを、BTR の要素の内訳と並べた記述 | 答えの種がある |
| Q22 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「今の結論」、[可塑性指標と成績](../wiki/concepts/phenomena/plasticity.md)「今の結論」 | 論文の指摘そのものと、可塑性との関係 | 答えの種がある |
| Q23 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」（Burda ら・RLeXplore・MEME の項目）、[318 強化学習 共通基盤](../wiki/concepts/framework/318_rl_core.md)「成立の経緯」（2026-11-15）「選ばなかった形」 | Head の中をどこで分けるか（最終段だけを分ける形）の事例 | 答えの種がある |
| Q24 | 答えがある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」の文献の項目（出典が、内的報酬の手法の全体像のサーベイ） | — | 答えがある |
| Q27 | 答えがある | [318 強化学習 共通基盤](../wiki/concepts/framework/318_rl_core.md)「成立の経緯」（2026-10-24、ADR 0065）「選ばなかった形」（常に抽選時に収益を作り直す形） | — | 答えがある |

### ステージ 3 の終わりの判定（2026-11-26）

当てたのは、ステージ 3 の取り込みと点検が済んだあとの `docs/wiki/index.md` と、`docs/wiki/concepts/` の 51 ページである。判定のあいだ、51 ページの内容は、Codex が評価の時点として控えた版（手元の `.scratch/rfd0202/stage3-review-files/concept-blobs.tsv`）と同じだった。wiki のページは変えていない。

答えがある 4、答えの種がある 12、無い 0。**「答えがある」は 4 のままで、実装の前から増えていない。** 16 個とも、判定は実装の前と同じだった。RFD §1.4 の終わり方の線（「答えがある」の数が、実装の前より増えていない）に当たる。判定は AI が付けた。

#### 確かめ方

- 実装の前の判定を見たうえで、同じ基準を当てた。別の枠では確かめていない。
- 根拠にしたページとその周りの 17 ページを、実装の前の版（コミット d39d2f07 の同じページ）と行ごとに比べ、足された行を読んだ。実装の前に根拠にした項目が、今の版に残っていることも確かめた（Qbert のページにあった TempoRL の 5 項目は、行動の時間尺度のページへ移り、Qbert のページには要旨の行がある）。
- 実装の前に無かった 5 ページ（GridMaze、優先度付き経験再生、擬似カウント、訪問済み状態へ戻る探索、自己模倣）と、全ページの横断的知見 36 項目を読んだ。
- 欠けているものは、概念のページとサマリ（`docs/wiki/summary/`）を語で検索し、足されていないことを確かめた。原本は検索していない。

#### 判定と根拠

| 番号 | 判定 | 根拠のページと節（今の版） | 足されたもの | まだ欠けているもの |
|---|---|---|---|---|
| Q1 | 答えの種がある | [行動の時間尺度](../wiki/concepts/methods/temporal_abstraction.md)「今の結論」「横断的知見」（TempoRL の 5 項目）「分かっていないこと」 | 横断的知見に、題材を並べた項目が入った（qbert と BeamRider、DropMerge と BeamRider、LunarLander と GridMaze）。「今の結論」に、LunarLander の利得が Atari と DropMerge では再現していないことが入った | Atari と DropMerge で効かなかった理由と、教訓。LunarLander で効いた経路（序盤の学習での繰り返し）と、Atari・DropMerge の結果は別々の項目にあり、3 項目以上を突き合わせる |
| Q2 | 答えの種がある | [Qbert](../wiki/concepts/subjects/qbert.md)「今の結論」「分かっていること」（初期値の比較）「分かっていないこと」、[Breakout](../wiki/concepts/subjects/breakout.md)「分かっていること」（Dopamine IQN に寄せた構成の項目） | 無い | 実装の前と同じ。DropMerge と初期値を同じ行に持つ記述は、概念のページにもサマリにも見当たらない |
| Q3 | 答えの種がある | [勾配ノイズとbatch](../wiki/concepts/phenomena/gradient_noise_batch.md)「今の結論」「横断的知見」、[kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「分かっていること」（batch 1024 の 3 seed の項目）、[Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)「横断的知見」（旧ネットワークの Breakout の項目） | 横断的知見の 3 項目（DropMerge の batch の利得は評価の側では小さい、という 2 項目と、Breakout で batch を大きくして更新の頻度をそろえた比較） | 実装の前と同じ。DropMerge の知見を Atari に当てた結果としての並び |
| Q5 | 答えの種がある | [kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「今の結論」「分かっていること」（RND の 4 項目）、[内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「今の結論」 | 「今の結論」に、Montezuma と KFM で RND の尺度や warm-up に応じて探索の経過が変わったことが入った | 実装の前と同じ。Montezuma と並べた理由の読み取り。DropMerge と RND は、[RND の設計文書のサマリ](../wiki/summary/design/212_rnd_lifelong_novelty.jp.md)に β の値があるだけ |
| Q6 | 答えの種がある | [Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)「分かっていること」（Montezuma で replay ratio を 8 から 4 にした項目）「横断的知見」（旧ネットワークと BTR 型の項目）、[可塑性指標と成績](../wiki/concepts/phenomena/plasticity.md)「横断的知見」（LayerNorm の項目）、[Breakout](../wiki/concepts/subjects/breakout.md)「今の結論」 | 横断的知見の 2 項目（LayerNorm の効き方が RR8・5M と RR4・50M で違う。BTR 型の利得は RR4 に出て、RR1・RR2 では出ない） | 同じ設定変更を RR4 と RR8 の両方で、予算をそろえて比べた記述と、方針が成り立つかを述べた記述 |
| Q7 | 答えがある | [評価と比較の条件](../wiki/concepts/measurement/evaluation.md)「今の結論」「分かっていること」（qbert の Greedy と、学習中と同じ選び方の比較の項目）、[Qbert](../wiki/concepts/subjects/qbert.md)「今の結論」「分かっていること」（選び方を変えた 8 面突破率の項目、Q 側・skip 側のノイズと τ を分けた 2 項目） | 2 つのページの「今の結論」に、Greedy と学習中と同じ選び方で上下が逆になることが入った | — |
| Q10 | 答えの種がある | [Montezuma's Revenge](../wiki/concepts/subjects/montezuma_revenge.md)「今の結論」「分かっていること」（部屋 2 の梯子の下で止まる項目、n-step を 10 にした 2 本の項目、割引率の分離の形の 9 本の項目）、[Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)「分かっていること」（n-step を 10 にした項目） | 「今の結論」に、部屋 2 での停滞の読み方が 1 文入った。項目は実装の前と同じ | 実装の前と同じ |
| Q11 | 答えの種がある | [kung_fu_master](../wiki/concepts/subjects/kung_fu_master.md)「分かっていること」（既定の設定の 16 seed の項目、挑み始めるきっかけの項目、方策の多様性の項目、文献上の留保）「分かっていないこと」 | 外の手法のページが 2 つできた（[訪問済み状態へ戻る探索](../wiki/concepts/methods/return_and_explore.md)、[自己模倣による価値学習](../wiki/concepts/methods/self_imitation.md)）。KFM に当てた記述は無い | 実装の前と同じ |
| Q14 | 答えがある | [Qbert](../wiki/concepts/subjects/qbert.md)「今の結論」「分かっていること」（初期値の比較、BTR へ寄せた 50M の項目）「横断的知見」（ε の経過の項目）。補い: [Atari横断](../wiki/concepts/subjects/atari.md)「分かっていること」（観測の縮小の項目） | 横断的知見に、2 つの 50M の違い（ε の経過、BTR 寄せの構成、初期値、Munchausen の log π）を並べた項目が入った | — |
| Q18 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」の「比較対象の実装」の 7 項目 | 無い | 実装の前と同じ |
| Q20 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」（LunarLander の RND の 3 項目）「分かっていないこと」。補い: [324 Agent個別実装 DQN系列](../wiki/concepts/framework/324_dqn.md)「課題」（predictor を周期的に初期化する案の行） | anet の結果を 1 項目にまとめた行（周期リセットでボーナスは続いたが、評価の範囲はリセットなしと重なった） | 実装の前と同じ。外の事例は、概念のページにもサマリにも見当たらない |
| Q21 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「横断的知見」「分かっていること」（β 0.01・Munchausen 有効の項目）「撤回・更新」（2026-11-12・2026-11-26）、[Breakout](../wiki/concepts/subjects/breakout.md)「分かっていること」（Munchausen RL の項目） | 横断的知見に、β 0.01 で Munchausen の有無を並べた項目が入った（Double DQN も同時に変えたことを、条件の違いに挙げる） | 実装の前と同じ |
| Q22 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」（LunarLander と GridMaze でボーナスが消えた項目、Montezuma で内的報酬が下がった項目）、[可塑性指標と成績](../wiki/concepts/phenomena/plasticity.md)「分かっていること」（Montezuma の RND の Run の 5 項目） | 無い。2 つのページの「今の結論」は短くなり、根拠は「分かっていること」に残る | 実装の前と同じ |
| Q23 | 答えの種がある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」（Agent57・Burda ら・RLeXplore・MEME の項目）、[318 強化学習 共通基盤](../wiki/concepts/framework/318_rl_core.md)「成立の経緯」（2026-11-15）「選ばなかった形」 | 無い | 実装の前と同じ |
| Q24 | 答えがある | [内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)「分かっていること」の「外の事例」（出典が、内的報酬の手法の全体像のサーベイの項目）。補い: [訪問回数と擬似カウントによる探索](../wiki/concepts/methods/pseudo_counts.md)、[訪問済み状態へ戻る探索](../wiki/concepts/methods/return_and_explore.md) | 同じサーベイを出典にする項目が増え、手法のページが 2 つできた | — |
| Q27 | 答えがある | [318 強化学習 共通基盤](../wiki/concepts/framework/318_rl_core.md)「成立の経緯」（2026-10-24、ADR 0065）「選ばなかった形」（常に抽選時に収益を作り直す形） | 無い | — |

#### 判定のほかに分かったこと

事実と、AI の読みを分けて書く。結果の読み方と、このあとの扱いは、開発者が決める。

事実は、次のとおりである。

- 見立ては、51 ページとも「なし」である。見立てを起こすのはステージ 4 で、この判定は、RFD §1.5 の決まりどおり、その前に付けている。
- 横断的知見は 36 項目ある。12 個の「答えの種がある」のうち、問いに当たる横断的知見が入ったのは 4 個（Q1・Q3・Q6・Q21）である。どれも、1〜2 項目で問いの全部に答える形にはなっていない。
- 「今の結論」は、実装の前は、複数の Run の数値を並べた長い文章だった。今は 1〜2 文である。削られた文の事実は「分かっていること」に残っていて、判定が下がった問いは無い。
- 欠けているものは、2 つに分かれる。
  - 事実は項目にあるが、問いに答える読み取りが 1〜2 項目になっていないもの: Q1・Q3・Q6・Q10・Q11・Q18・Q23 の 7 個。
  - 問いの一部に当たる記述が、概念のページにもサマリにも見当たらないもの: Q2（DropMerge への反映）、Q5（DropMerge と RND）、Q20（外の事例）、Q21（BTR の要素の内訳）、Q22（論文の指摘）の 5 個。

AI の読みは、次のとおりである。

- 横断的知見は、出典の値と条件だけを並べ、推測を足さない決まりである（[wiki の規約](../wiki/README.md)）。理由・教訓・方針を聞く問いには、並べた事実だけでは、そのままの答えにならない。
- 16 個の問いは、過去の会話の質問から取った。会話の中で調べた答えが、材料（実験記録・報告・RFD）に書かれていなければ、材料の取り込み直しでは wiki に入らない。会話の記録を引くのは、ステージ 4 の作業である。原本は検索していないので、見当たらない 5 個が材料に無いとまでは言えない。

## 正しさの指標

横断的知見はステージ 2 の終わりに確かめた（下の節）。見立ては、ステージ 4 の最初の点検のあとに確かめる（RFD §1.5）。

### 横断的知見（ステージ 2 の終わり、2026-11-26）

- 確かめたもの: Codex がステージ 2 で書いた横断的知見 34 件から、無作為に取った 20 件。乱数の種と、取った時点の文は、手元の `.scratch/rfd0202/stage2-independent-sample.json` に固めてある。番号は、取った順である。
- 確かめた枠: 書いた枠（Codex）とは別の、Claude Code の会話。
- 確かめ方: 1 件ごとに、出典の原本 2 本の、引かれた節と、数値のある表を直接読んだ。抜粋は使っていない。下の表の行番号は、2026-11-26 の時点のものである。
- 取ったあとに Codex が文を直した 6 件（2・10・15・18・19・20 番）は、取った時点の文と、今の文の両方を確かめた。ほかの 14 件は、今のページの文が取った時点と同じであることを、文字列の一致で確かめた。

#### 判定の基準

| 判定 | 基準 |
|---|---|
| 正しい | 主張の事実・数値・条件が、原本の該当の節と合う |
| 正しい（注あり） | 事実は合うが、語の選び方か、引いた節の示し方が粗い。落とした条件を戻しても、主張は弱まらない |
| 誤り | RFD §1.5 の 4 つ（出典に無い、数値が違う、条件を落とした、推測が混ざっている）のどれかに当たり、wiki の文が、原本より強いこと・広いこと・原本に無いことを言っている |
| 保留 | 原本を読んでも決められない |

誤りの率は、「誤り」だけで数える。基準は AI が置いた。

#### 結果

取った時点の文は、正しい 14、正しい（注あり）3、誤り 3、保留 0 だった。**誤りの率は 3 ÷ 20 = 15%** である。RFD §1.4 の線（20 件なら 4 件以上）には当たらないが、線まで 1 件である。「注あり」も誤りに数えると、6 ÷ 20 = 30% になる。

誤りの分類は、条件を落とした 2（10・18 番）、推測が混ざっている 1（19 番）で、出典に無い・数値が違うは 0 だった。

直した 6 件の今の文は、5 件が正しい。18 番は、直した文に、原本と合わない箇所が 1 つ残る（この確かめの時点。そのあと Codex が直した。下の「今のページに残る箇所」）。

| 番号 | 主張（ページ） | 取った時点 | 今の文 | 根拠にした原本の節と行、理由 |
|---|---|---|---|---|
| 1 | SimPLe の経験の予算と実時間（[Model-based Atariの評価](../wiki/concepts/methods/model_based_evidence.md)） | 正しい（注あり） | 同じ文 | [SimPLe の報告](../../reports/simple_model_based_atari_survey_2026-09-24.md) §4.2（実データは 1〜2 時間分でも、world model の中で 15.2M 回の相互作用が要る。316 行）・§4.3（著者が、実時間が model-free の学習より長いと認めた。322 行）。[実時間の調査](../../reports/atari_walltime_frontier_survey_2026-09-09.md) §1.1（frames と agent step は 4 倍違う。43 行）。注: 「計測区間」の語は、引いた §1.1 に無い。当たる記述は §1.2（FPS の単位）と §5.1・§5.2（学習を含まない値と含む値の区別）にある |
| 2 | 勾配ノイズの尺度と batch の利得（[勾配ノイズとbatch](../wiki/concepts/phenomena/gradient_noise_batch.md)） | 正しい | 正しい | [勾配ノイズの報告](../../reports/gradient_noise_observation_and_control_survey_2026-10-15.md) §1（勾配ノイズの尺度が大きいから batch を増やせば成績が上がる、という決まりにはできない。50 行）。[DropMerge の記録](../experiments/default-dqn/dropmerge/2026-10-15_gradient-noise-batch.md) まとめ 3（学習側は +0.5〜3.9%。19 行）・まとめ 4（最終 checkpoint の評価は B1024/RR1 が +0.9%、B512/RR2 と B1024/RR2 が +0.2〜0.3%、各群約 10,000 局。20 行）・探索ブロック 05（NoisyNet を使わない評価方策。253 行）。今の文の数値は、まとめ 4 と合う |
| 3 | IGE の戻る前提と公開実装（[訪問済み状態へ戻る探索](../wiki/concepts/methods/return_and_explore.md)） | 正しい | 同じ文 | [内的報酬の報告](../../reports/intrinsic_reward_landscape_survey_2026-11-18.md) §5.4（戻る操作はシミュレータの状態の復元。226 行）。[探索情報の報告](../../reports/llm_rl_exploration_information_2026-11-07.md) §4（戻る操作は可能だと仮定する。99 行）・§5（同じ seed で reset し、保存した行動列を再実行して戻り、そのあと `actions_taken` を 0 に戻す。107 行） |
| 4 | τ の置き方の 3 つの目的（[τ配置とIQN比較](../wiki/concepts/methods/tau_sampling.md)） | 正しい（注あり） | 同じ文 | [分散低減の報告](../../reports/_survey_tau_mc_theory.md) §1.1（層化抽出。同じ積分の分散を下げる。35 行）。[リスクの報告](../../reports/_survey_tau_risk_adaptive.md) §6（FQF は、リスクの目的でなく分布の近似の精度の目的。282 行）・§7（CVaR は τ を下位の区間に限る。299 行）。注: 冒頭の「非一様配置」は、層化には当たらない（層化は、一様分布のまま層に分けて引く）。あとに続く文は原本と合う |
| 5 | lane 窓とエピソード長、HER の実装（[Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)） | 正しい | 同じ文 | [lane 窓の記録](../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md) まとめ 1・2（lane 窓と貪欲エピソード長の比。19〜26 行）。[lane 窓の調査](../../reports/replay_lane_window_survey_2026-09-17.md) §3「エピソード長でバッファ容量を正規化した唯一の実例」（SB3 の HER の実装と、HER の目標の引き直し。306〜318 行）・§6（この問いを正面から扱った研究は見つからなかった。485 行） |
| 6 | 評価のばらつきと学習側の突破率（[評価と比較の条件](../wiki/concepts/measurement/evaluation.md)） | 正しい | 同じ文 | [LN512 の記録](../experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md) 探索ブロック 05（突破は評価の窓平均に埋もれる。144〜148 行）・06（学習側の突破率は 1.32% 対 0.47%。156〜157 行）。[BTR の個別比較の記録](../experiments/default-dqn/atari/2026-09-01_btr-feedback-arms.md) まとめ 31・探索ブロック 26（評価を 10 エピソードにすると、評価点の標準偏差が平均 3.45 倍縮む。45、728〜736 行） |
| 7 | qbert と BeamRider の合算の Greedy（[行動の時間尺度](../wiki/concepts/methods/temporal_abstraction.md)） | 正しい | 同じ文 | [qbert の記録](../experiments/default-dqn/atari/2026-10-31_temporl-qbert-50m-learning.md) まとめ 1・探索ブロック 01（合算 27,058・26,523、スキップなし 40,999・30,415。7、44〜47 行）。[BeamRider の記録](../experiments/default-dqn/atari/2026-11-04_temporl-freeway-beamrider.md) 探索ブロック 01（合算 s1 は full 9,854・single 31,458、スキップなしは 22,914・85,872 と 23,626・84,636。合算 s2 は 15.35M まで。45〜59 行）・08（213 行） |
| 8 | Kung-Fu Master の 2 本の開きと少数 seed（[評価と比較の条件](../wiki/concepts/measurement/evaluation.md)） | 正しい | 同じ文 | [100M の記録](../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md) まとめ 2・3、探索ブロック 01 の表（seed 2 ÷ seed 1 は 10.3〜50.7M で 0.52〜0.70、91.1M で 0.99。21〜26、96〜107 行）。[少数 seed の報告](../../reports/few_seed_comparison_reporting_survey_2026-11-04.md) §1（52 行）・§7.2・§7.3（380〜395 行） |
| 9 | 複製の無い小差と 2 本で言えること（[評価と比較の条件](../wiki/concepts/measurement/evaluation.md)） | 正しい | 同じ文 | [Atari の探索索引](../experiments/default-dqn/atari/README.md)「追加探索と複製の運用方針」（46〜49 行）。[少数 seed の報告](../../reports/few_seed_comparison_reporting_survey_2026-11-04.md) §1（52 行）・§7.2（380〜381 行） |
| 10 | LunarLander の STOP×Single と無効（[行動の時間尺度](../wiki/concepts/methods/temporal_abstraction.md)） | 誤り（条件を落とした） | 正しい | [Single の比較の記録](../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md) まとめ 3・8、AUC の表（STOP 対照 102.58／50.95／18.20、無効 64.93／4.50／−6.83。37、42、48〜52 行）。[bit 比較の記録](../experiments/default-dqn/lunarlander/2026-11-05_temporl-stop-single-bit-compare.md) まとめ 2・4（seed 1、100 万 exp step まで、共通の 103 tag・239,213 点が一致。多数の seed で分布を測ったのではない。30、32 行）。理由: 一致を確かめた範囲（seed 1 の 1 本、最初の比較の期間 2M の前半）を書いていない。ほかに、「上回った」は記録の決まりでは「同じ帯」に当たり、「補助勾配の効果の根拠」は、記録の「skip の効果ではなく偶然」と指すものがずれる。今の文は、3 点とも原本に合わせてある |
| 11 | SGRL の速度と DERL の探索費用（[評価と比較の条件](../wiki/concepts/measurement/evaluation.md)） | 正しい | 同じ文 | [小目標の報告](../../reports/llm_rl_goals_and_reflection_2026-11-07.md) §3（SGRL 33.8±1.5 対 PPO 24.8±5.7、1,850 対 13,530 step/s。60〜62 行）。[小型モデルの報告](../../reports/llm_rl_small_models_and_costs_2026-11-07.md) §5（最後の方策学習の予算は揃えるが、その前の報酬探索の費用は追加である。88 行） |
| 12 | qbert の第 8 面突破の 2 つの基準（[Qbert](../wiki/concepts/subjects/qbert.md)） | 正しい | 同じ文 | [サンプル効率優先の記録](../experiments/default-dqn/atari/2026-09-25_sample-efficiency-profile-atari5.md) まとめ 13（第 8 面に入ってすぐ終わったゲームも `round_clear` 8 を返す。56〜58 行）・探索ブロック 03（385〜392 行）。[BTR との差の記録](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md) まとめ 13（64 env の seed 4 は、第 8 面に着いたまま 29,000 点を超えた。57〜60 行）・探索ブロック 06（349〜366 行） |
| 13 | iLLM と LLM Priors の行動の出し方（[LLMの外部知識と探索](../wiki/concepts/methods/llm_prior_knowledge.md)） | 正しい | 同じ文 | [探索情報の報告](../../reports/llm_rl_exploration_information_2026-11-07.md) §2（行動の Head は推奨行動を予測する。36 行）・§3（実行可能な行動の中から log probability が最大のものを選ぶ。70〜72 行）。[行動助言の報告](../../reports/llm_rl_action_advice_2026-11-07.md) §2（自由文を規則で行動へ写す。候補から行動を選び、次状態の価値更新も候補に限る。33、42 行） |
| 14 | TempoRL と UTE の長さの決め方（[行動の時間尺度](../wiki/concepts/methods/temporal_abstraction.md)） | 正しい | 同じ文 | [TempoRL の解説](../../reports/biedenkapp2021_temporl_paper_2026-10-03.md) §3.1・§3.2（スキップの価値で長さを選ぶ。式 (4)。146〜162 行）。[UTE の解説](../../reports/lee2024_uncertainty_aware_temporally_extended_actions_paper_2026-11-04.md)「アンサンブルによる不確かさの定量化」（平均と標準偏差で長さを選ぶ。135〜149 行） |
| 15 | qbert の ε の経過と初期値（[Qbert](../wiki/concepts/subjects/qbert.md)） | 正しい（注あり） | 正しい | [BTR との差の記録](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md) まとめ 1・探索ブロック 01（ε を 3 区間に分けた Run。45〜50M の突破率は 7.7% 対 5.1%。17〜18、106〜110、136 行）。[初期値の記録](../experiments/default-dqn/atari/2026-10-23_btr-qbert-repro-init.md) まとめ 8・対象 Run の表（`aln_btreps` は、ε の経過・初期値・Munchausen の log π を BTR に合わせた 50M。47.5〜50M は 86.8%。26、41〜44 行）。注: 2 本の ε の経過は同じ作りではない（3 区間の近似と、指数で下げて 25M から 0）。後の Run は、BTR 寄せの構成と log π も違う。取った時点の「条件の違い」は、初期値だけを挙げていた。どれも「ε の経過だけには帰せない」を強める向きなので、誤りには数えない。今の文は、違いを並べてある |
| 16 | 旧ネットワークと BTR 型の replay ratio（[Replay ratioとlane窓](../wiki/concepts/methods/replay_ratio_lane_window.md)） | 正しい | 同じ文 | [replay ratio の機序の記録](../experiments/default-dqn/atari/2026-09-08_replay-ratio-mechanism.md) まとめ 1（25〜50M の突破率は RR4／B512 が 8.84%、RR2 が 14.33%）。[BTR ネットワークの移植の記録](../experiments/default-dqn/atari/2026-09-12_btr-network-port.md) まとめ 13（旧ネットワークとの差は RR1 −54.1、RR2 −38.2、RR4 +133.9） |
| 17 | NoisyNet の行動変化と成績（[NoisyNetの観測](../wiki/concepts/methods/noisynet.md)） | 正しい | 同じ文 | [LunarLander の NoisyNet の記録](../experiments/default-dqn/lunarlander/2026-10-06_noisynet.md) メタデータ（QR、3 seed）・まとめ 2・6（成績の差は seed の幅の中。終盤の食い違いは 2〜4%）。[Atari の NoisyNet の記録](../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md) まとめ 1（7 ゲームとも、基準のどの seed も下回る）・5（40〜50M の平均は 0.0〜4.9%） |
| 18 | Montezuma の評価方策と学習時の設定（[内的報酬による探索の比較](../wiki/concepts/methods/intrinsic_novelty.md)） | 誤り（条件を落とした） | 直す箇所が 1 つ残る | [30M の記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-se-30m.md) まとめ 5（30M の重みは 400 点以上が 19/32〜32/32）・6（20.07M の重みは 4 条件とも全ゲーム 400 点。11〜12、140〜154 行）。[Taiga 寄せの記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md) まとめ 1・12・21・25 と群の表（66〜94 行）。理由（取った時点）: 評価の ε・τ で 400 点の到達が分かれたのは 30M の重みで、同じ記録の 20.07M の重みでは分かれなかった。保存の時点を落とすと、原本より広い主張になる。今の文: 前半は直っている。後半の「β=0.01・Munchausen 無効の 2 seed だけが」は、記録のまとめ 1 と同じ言い方だが、同じ記録のあとの Run（β 0.003 の `b0003` は 100/160、`g99ne_split` は 146/160、`r123` は 4 回で 42/64）も、Greedy で 2,500 点以上に何度も届いている。まとめ 1 が比べた 5 本の中でも、Munchausen 有効の `rnd_b001` が 20.2〜29.2M の 10 回のうち 4 回で届いている（計 5/160。回ごとの表は 154〜175 行）。範囲を添えても「だけ」は成り立たない |
| 19 | IQN の τ の引き方と評価時の扱い（[τ配置とIQN比較](../wiki/concepts/methods/tau_sampling.md)） | 誤り（推測が混ざっている） | 正しい | [原典の報告](../../reports/_survey_tau_core_papers.md) §3（τ と τ′ を一様分布から別々に引く。方策に使うサンプル数 K は 32 に固定。68〜91 行）。[実装の報告](../../reports/_survey_tau_implementations.md) §3（pfrl。102 行）・§5（d3rlpy。153 行）・まとめの表（328〜344 行）。理由: 「論文の学習手順から評価時の乱数の有無までは決まらない」に当たる記述は、どちらの原本にも無い。原典の報告は、方策に使うサンプル数を論文が決めていると書く。今の文は、この 1 文を外し、原本の事実に置き換えてある |
| 20 | Breakout の RR8 の崩れとピーク（[可塑性指標と成績](../wiki/concepts/phenomena/plasticity.md)） | 正しい | 正しい | [可塑性の較正の記録](../experiments/default-dqn/atari/2026-08-28_plasticity.md) まとめ 6（同じ設定の 5 本のうち 1 本は崩れない。23 行）・探索ブロック 01（70 行）。[設定修正後の記録](../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md) まとめ 2・3、探索ブロック 01 の表（2〜3M の窓は旧 165.8、新 323.7・239.7。新は 2 本とも崩れた。18〜19、60〜65 行） |

#### 抜粋で付いていた判定との違い

この確かめの前に、Codex の依頼で、原本の抜粋から付いた判定があった（正しい 9、誤り 3、資料不足 8。手元の `.scratch/rfd0202/stage2-claude-review.md`）。原本の節を読んだ結果との違いは、次のとおりである。

- 資料不足の 8 件（2・3・6・11・13・17・18・19 番）は、原本の節を読むと決まった。6 件は正しく、18・19 番は誤りだった。
- 誤りとされた 3 件のうち、10 番は誤りのままである。15 番は「注あり」にした。引いた Run が主張と合わないという指摘は、Run の取り違えだった（`aln_btreps` は、初期値も戻した 50M の Run である）。20 番は正しい。ピークの値は原本のまとめ 3 と表にあり、抜粋から落ちていた。

#### 今のページに残る箇所

wiki のページは、この確かめでは変えていない。確かめの時点で残っていた 3 か所は、そのあと Codex が直した。ステージ 3 の終わりに、今の文を原本と照らし、3 か所とも合うことを確かめた。

| 番号 | 確かめの時点で残っていた箇所 | 今の文（ステージ 3 の終わり） |
|---|---|---|
| 18 | 「β=0.01・Munchausen 無効の2 seed だけが」 | 直っている。「だけ」を外し、まとめ 1 が比べた 5 本の名前と、3 本の本数（99/160・81/160 と、Munchausen 有効の 5/160）を並べる文になった。「撤回・更新」に 2026-11-26 の行がある |
| 1 | 「計測区間」と、引いた節 | 直っている。語は「FPS が生フレーム数か agent step 数か、環境の実行だけか方策の学習も含むか」になり、出典の節に §1.2・§5.1・§5.2 が足された |
| 4 | 冒頭の「非一様配置」 | 直っている。「一様分布の層化抽出では、…各層の幅に比例した本数の τ を引くと、…分散は増えない」になった。原本 §1.1 の比例配分の記述と合う |

18 番について、Claude Code が Codex に渡した文の案（比べた範囲の 5 本を添えて「2 seed だけが」を残す形）は、誤っていた。5 本の中でも、Munchausen 有効の `rnd_b001` が複数の回で 2,500 点に届いている。Codex が原本の回ごとの表と照らして、この案を採らなかった。

実験記録のまとめ 1 の「届き続けたのは…2 本だけだった」は、あとから足した Run のまとめ（12・21・25）と合わなくなっている。実験記録は正本なので、扱いは開発者が決める。

## これから行うこと

| 時機 | 作業 |
|---|---|
| ステージ 2 の終わり | 済み（2026-11-26）。横断的知見の 20 件を原本で確かめ、誤りの率を「正しさの指標」に書いて、開発者に示した |
| ステージ 3 の終わり | 済み（2026-11-26）。上の 16 個に、同じ基準で判定を付け直し、表の「ステージ 3 の終わり」の列と、節「ステージ 3 の終わりの判定」に書いて、開発者に示した |
| ステージ 4 | 再グリルのあとの RFD §9.10 の順で行う。(1) wiki の規約と skill の直し、(2) 会話の記録を引く道具と検証資材、(3) 未決表の U23 の試しと、最初の点検、(4) 過去の質問 29 個を書く（16 個を先、13 個をあと。RFD の S51）、(5) 開発者の発言の見立て、(6) 16 個の付け直し（S37）と、足した横断的知見と見立ての正しさの確かめ（S43）。計画は、着手のときに、このファイルへ足す |
| ステージ 4 を終えた日の翌日から 4 週間 | これからの質問の指標（RFD の S35・S36）と、数の指標（RFD §1.5）の値を RFD に追記する |

## テスト

文書の記録だけで、コードと設定を変えないので、TDD の対象外である。ステージ 4 の道具のテストは、着手のときに計画へ足す。

## 検証

- このファイルのリンク先の実在を、[memo の README](README.md) の「リンクの注意」のリンク切れ検査と同じ手順で確かめる。
- 根拠に挙げた節の見出しを、`rg -n "^## " docs/wiki/concepts` で確かめる（実装の前の判定のときは、移す前の置き場で確かめた）。
- 正しさの指標の標本が今のページの文と同じかどうかを、標本の `claim` の文字列がページにあるかで確かめる。
- [AGENTS.md](../../AGENTS.md) の避ける語の表のうち、検索する語を、このファイルに対して固定文字列で引く。
- 公開できない値（絶対パス、ユーザー名、マシン名、認証情報、メールアドレス、IP アドレス、内部の URL）が無いことを、目で確かめる。

## 検証結果

- 2026-11-26 このファイルのリンク先 17 件（wiki のページ 13、RFD、課題、memo の README、AGENTS.md）は、すべて実在した。
- 根拠に挙げた節の見出しは、13 ページのどれにもあった。項目の数（「比較対象の実装」の 7 項目など）は、ページを検索して数えた。
- 避ける語は 0 件、公開できない値は 0 件だった。改行は LF である。
- このファイルは、レビューエージェントに出していない。開発者が、会話で問いと判定を見ている。
- 2026-11-26 正しさの指標の標本 20 件は、14 件が今のページの文と同じで、6 件（2・10・15・18・19・20 番）が変わっていた。足した節のリンク先 49 件（wiki のページ 12、実験記録 21、報告 16）は、すべて実在した。避ける語は 0 件、公開できない値は 0 件だった。この節も、レビューエージェントには出していない。
- 2026-11-26 直した RFD は、リンク切れ、避ける語、状態の行、番号の重なりの検査を通した。レビューエージェントに出し、1 往復目で OK だった（原文は `.scratch/rfd0202/review-20261126-1712.md`）。レビューは、RFD に書いた数（4・12・0）が、このファイルの表と合うことも確かめた。
- 2026-11-26 ステージ 3 の終わりの判定を足したあと、このファイルのリンク先 127 件（延べ）は、すべて実在した。足した節のリンクは 35 件（延べ。異なるリンク先は 18）である。根拠に挙げた節の見出し 32 個は、13 ページのどれにもあった。判定の表は 16 行で、答えがある 4、答えの種がある 12 である。避ける語は 0 件、公開できない値は 0 件、改行は LF だった。概念の 51 ページは、判定の前に、Codex が控えた版と `git hash-object` で比べて同じだった。この節も、レビューエージェントには出していない。

## 前提

- 問いの数は 16 個で固める。ステージ 3 の終わりにも、同じ問いと同じ基準で付ける。
- 判定の基準は、AI が置いた。RFD に基準の定めは無い。
- 問いの指標で比べるのは、「答えがある」の数（実装の前は 4）である。実装の前の「答えがある」と「答えの種がある」を合わせた数が 16 個中 16 で、これ以上増えないので、2026-11-26 に開発者の選択で、RFD の §1.4・§1.5・S30・§13 を改めた。
- 2026-11-26〜27 の再グリルで、16 個は、答えの項目を書く対象になった（RFD の D19）。仮説は、これからの質問で測る（D20）。16 個は、書いたあとに付け直して、書けたことの確かめに使う（S37）。
- RFD の「段」は、このファイルでは「ステージ」と書く。

## 経過

- 2026-11-26 候補 29 個を出し、開発者が 16 個を選んだ。今の wiki に判定を付け、開発者に示した。記録の置き場は、開発者の答えで、この実装メモにした。
- 2026-11-26 実装の前の判定で、合わせた数が上限だと分かった。比べる数を「答えがある」の数に改める文の草案を示し、開発者の選択で RFD を直した。RFD のレビューは OK だった。
- 2026-11-26 ステージ 2 の終わりに、横断的知見の 20 件を原本で確かめた。取った時点の文の誤りは 3 件（15%）だった。抜粋で付いていた判定（正しい 9、誤り 3、資料不足 8）は、原本を読んで付け直した。
- 2026-11-26 ステージ 3 の終わりに、16 個の問いに判定を付け直した。16 個とも実装の前と同じで、「答えがある」は 4 のままだった。Codex が直した 3 か所（1・4・18 番）の今の文は、原本と合っていた。18 番で Claude Code が渡した文の案は誤っていて、Codex が採らなかった。
- 2026-11-26〜27 判定が RFD §1.4 の終わり方の線に当たり、開発者の選択で、ステージ 4 の前に再グリルを行った。欠けた前提を 2 つの試しでそろえ（RFD §5.12）、D19〜D21、S32〜S51、未決表の U23・U24 が決まった。ステージ 3 の終わりの wiki の控えを、手元に取った（RFD の P77）。
