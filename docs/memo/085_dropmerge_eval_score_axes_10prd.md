# DropMerge の評価スコア（本家互換スコアほか複数の軸）暫定 PRD

> 2026-10-03 起票。詳細設計は別途グリルで行う前提で、決めることを §5 に残した。
> 起点: DropMerge フィードバック実験（2026-10-01〜03、Munchausen と評価方策の比較）で、NoDropTimeout の罰や待ち時間の罰を変える実験を控え、罰込みの return ではシリーズをまたいだ比較ができなくなることが分かったこと。あわせて、DropMerge の最終目標である「人間のチャンピオン超え」を測る物差しが無いこと。
> 関連: [086_dropmerge_macro_actions_10prd.md](086_dropmerge_macro_actions_10prd.md)（行動の単位が変わる実験。比較に本 PRD の物差しを使う）、[956_smdp_transition_duration_10prd.md](956_smdp_transition_duration_10prd.md)（待ち時間の罰で return が変わる）、[CONTEXT.md](../../CONTEXT.md) の「投了」「NEET」「NoDropTimeout」、[done/039_dropmerge_nolegal_adjudication_10prd.md](done/039_dropmerge_nolegal_adjudication_10prd.md)、[ADR 0014](../adr/0014-dropmerge-nodroptimeout-loss-not-truncation.md)。

## 1. 目的と問題

評価の物差しを、学習の報酬から切り離す。人間が画面で見る「ゲームの点数」を、本家スイカゲームと同じ得点体系で出せるようにし、評価に支障がある場合に備えて複数の軸を持てるようにする。

いまの評価値は `mean.episode_return`（`21_eval/03_target_reward_ema` など）で、学習の報酬そのものの合計である。報酬には合体の点のほかに罰が入っている。

- SpawnBlocked と Overflow の `game_over_penalty`（現行 −10）
- NoDropTimeout の `no_drop_timeout_gameover_penalty`（現行 −10）
- DROP 以外の step に掛かる `time_penalty`、NOOP に掛かる `noop_penalty`（現行どちらも 0）

このため、罰の値を変えると return が変わり、シリーズをまたいで比べられない。

合体の点も、本家とは違う独自の得点表を使っている。学習用に、1 つ上のランクの点が下のランクの 2 倍を超えるよう調整した表である。

```text
DropMergeEnv.fruit_scores ?= 0.1  0.3  0.7  1.5  3.1  6.3  12.7  25.5  51.1  102.3  205.0 820  # 本家ベースで「Rank N+1 のスコア ＞ Rank N のスコア × 2」になるよう調整
```

`apps/runner/config/DropMerge.txt:1624`（`@baseline` は最後のダブルスイカを 1000.0 にしている、同 `:1549`）。したがって今の点数は、人間のスイカゲームの点と比べられない。

## 2. 現状（コードと設定で確認したこと）

- 合体の点は `episode_score_` に積まれる。通常の合体は `s = fruit_scores[next_rank - 1]`（`core/envs/dropmerge1/src/DropMergeEnv.cpp:754-756`）、ダブルスイカは `fruit_scores[kFruitTypeCount]`（同 `:779-781`）。
- 同じ合体の点がそのまま報酬に入る（同 `:1461-1463`）。罰は同じ step の報酬に加算される（`time_penalty` は `:1152-1155`、`game_over_penalty` は `:1157-1160`、NoDropTimeout の罰は `:1195-1197`、`noop_penalty` は `:1245-1248`）。
- `episode_score_` は env の scalar として公開されていない。scalar のキーは `ep_step`、`ep_max_rank`、`ep_double_suika_created` / `achieved`、`ep_end_fruit_count`、`term_reason_*` など（同 `:1562-1668`）で、点数は aux の `score` / `last_score`（同 `:1546-1547`）とログの `episode_score=` にしか無い。
- `noop_penalty` は `action == kActionNoop` で判定しているため、全行動が DROP の Direct モードでは列 0 の DROP にも掛かる（同 `:1245-1248`）。DirectNoop（現行の `action_mode = direct_noop`）では問題にならない。
- 設定には「本家相当」の表が残っている（`apps/runner/config/DropMerge.txt:1545` のコメント行と、`:1579` の `@orig`）。どちらも `0.1 0.3 0.6 1.0 1.5 2.1 2.8 3.6 4.5 5.5` で始まり、`@orig` は要素が 11 個しかない。得点表は要素数 12（`kFruitTypeCount + 1`）でないと検証で落ちる（`core/envs/dropmerge1/src/DropMergeEnv.hpp:195-198`）。
- 二次情報（攻略サイト）では、本家は進化のたびに次の点が入るとされる: いちご 1、ぶどう 3、デコポン 6、かき 10、りんご 15、なし 21、もも 28、パイナップル 36、メロン 45、スイカ 55。この表と比べると、設定に残る「本家相当」の表は 1 ランクずれて見える。ダブルスイカの点の扱いは確認できていない。
- 出現する果物は小さい 5 種から等確率（`drop_probs = 20.0 20.0 20.0 20.0 20.0`、`core/envs/dropmerge1/src/DropMergeEnv.hpp:179`）。
- 人間の参考値として、元総合 1 位のランカーが 7,355 点（ダブルスイカ 4 回）という記事がある（§8）。

## 3. 提案の骨子

1. 学習の報酬（`fruit_scores` と罰）と、評価用のスコアを分ける。評価用のスコアは同じ合体イベントから計算し、学習には渡さない。
2. 評価用のスコアを、名前付きの軸として複数持てるようにする。最初の軸は次の 2 つ。
   - 本家互換スコア: 本家の得点表とダブルスイカの規則に合わせる。人間との比較用。
   - DropMerge スコア: 今の `fruit_scores` の合計（現行の `episode_score_` と同じ値）。過去の Run との連続性のため。
   - 罰込みの return は、今の評価値としてそのまま残す。
3. 各軸をエピソード単位の env scalar として公開し、評価セッション（`eval_target` / `eval` / `eval_target_greedy` など）と学習側の両方で指標に購読できるようにする。
4. 得点表は設定で持つ。本家互換の表は、一次情報で確かめた値を既定にする。

## 4. User Stories

1. 研究者として、最良の Run の点数を、本家スイカゲームの人間の記録と同じ物差しで比べたい。最終目標（人間のチャンピオン超え）の進み具合を測るため。
2. 研究者として、NoDropTimeout の罰や待ち時間の罰を変えた Run どうしを、罰の影響を受けない物差しで比べたい。
3. 研究者として、これまでの Run と、新しい物差しを入れた後の Run をつなげて見たい。DropMerge スコアの軸は今の値と一致してほしい。
4. 研究者として、物差しが評価に向かないと分かったときに、軸を足して使い分けたい。

## 5. 決めること（詳細設計で詰める）

1. 本家の得点表を一次情報で確定する。対象の版（Nintendo Switch 版とスマートフォン版で同じか）、「作った果物で点が付く」のか「合体した組で点が付く」のか、ダブルスイカの点（0 か 66 か、ほかか）。
2. DropMerge のランク番号（1 = さくらんぼ … 11 = スイカ、ダブルスイカはランク 12 相当）への写し方。
3. 盤面に残った果物を点に数えるか（本家は合体の点だけと理解しているが、確認する）。
4. scalar のキー名と指標の tag。軸の名前の付け方（`<thing>_key` 規約、設定階層の置き場所）。
5. 物理と出現規則の差（箱の大きさ、果物の半径、反発、出現する果物の分布、即時落下）の一覧を、比較の注意として残す範囲。
6. 過去の Run を本家互換スコアで計算し直せるか。合体のログは通常レベルではスイカとダブルスイカしか出ないので、計算し直せない前提で考える。
7. 他の env（Atari の生スコアとクリップ後の報酬など）へ一般化するか。本 PRD は DropMerge に閉じる案を既定にする。

## 6. 対象外

- 学習の報酬表（`fruit_scores`）や罰の値を変えること。
- DropMerge の物理を本家に合わせること。
- オンラインのランキングとの照合。

## 7. 依存と順序

依存は無い。NoDropTimeout の罰の変更、[086](086_dropmerge_macro_actions_10prd.md) のマクロ行動、[999 SMDP 対応](956_smdp_transition_duration_10prd.md) の待ち時間の罰など、報酬や行動の単位を変える実験より前に入れる。

## 8. 参照

- [みんなの楽々, 2023] 「スイカゲーム」攻略記事（得点表の二次情報）。https://minnano-rakuraku.com/contents/suika-game-tips-7583/
- [電ファミニコゲーマー, 2023/10] 「“ダブルスイカ” を4回達成し、元総合スコア1位(7355点)のランカーが『スイカゲーム』を完全解説!」。https://news.denfaminicogamer.jp/kikakuthetower/231031f/embed
