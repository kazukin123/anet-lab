---
title: 評価スケジュールの初回発火と位相
page_kind: サマリ
material: "[[docs/memo/0158_eval_schedule_first_fire_10rfd.md|0158_eval_schedule_first_fire_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
updated: 2026-11-26
---
# 評価スケジュールの初回発火と位相

## 元の材料

[RFD0158　評価スケジュールの初回発火と位相](../../../memo/0158_eval_schedule_first_fire_10rfd.md) blob:6b0c5989f68439e59e4e4e0278dcdf2a012d7587

## 要点

- 従来は最初の学習イベントで評価が必ず発火し、未学習の greedy 方策が打ち切りまで走ることがあった。
- 学習開始直後の評価を既定で抑え、必要な評価専用 Run などではタグごとに最初の評価を許す。
- 最初の評価を実施するかと、その後の定期評価の発火時点をずらす指定を区別する。
- 複数の評価が同じ時点に集中する条件も扱い、初回や区間境界での発火列を検証する。
- IntervalGate を使う他のメトリクスや DQN・ImageCls の診断には変更を及ぼさない。

## 内容

### 問題

`run.eval_schedule.[tag].interval` で組んだ評価は、step の値によらず最初に必ず 1 回発火する。最初の学習イベントは warmup が明けた最初の学習 step で、そのときの learn step は 0 である。そのため、どの評価の系列も、先頭に初期の重みを測った点が入る。
Atari-5 の 5 ゲームを同じ設定（10M / RR4 / `run.@eval2ch`）で回したときの、learn_step=0 の ε=0 の評価セッション。当時のタグ名は `greedy_dist` で、400 エピソード・`eval_batch_size` 16。
学習の停止は 2/5 ゲームで 692.6s と 885.5s、合計約 26 分だった。5 ゲーム × 50M のスイープ（約 47.5h）に対して 0.9% である。
この点は使われていない。実験記録 2026-09-15 探索ブロック 07 の ε=0 の 9 点の対の表は 5.2M から始まり、learn_step=0 の点を使っていない。

### ゴール・非ゴール・成功の測り方

eval schedule の既定で、学習直後（発火の列の先頭）の評価を走らせない。全 env・全タグに効く。要る Run（評価専用 Run など）は、タグごとに走らせるよう書ける。
interval を変えずに、評価ごとに発火の時点をずらせる。今ある、位相を避けるために間隔を曲げた設定を置き換える。
`IntervalGate` と、そのほかの利用者（メトリクスの observer、DQN の診断、ImageCls）の挙動は変えない。
既定の Run（どの env でも）で、各タグの最初の評価セッションの learn step が `offset + interval` 以上である（Run ログの最初の `session end` 行）。

### 確定事項

`run.eval_schedule.[tag].warmup_count` は、発火の列の先頭から評価を走らせない発火の回数。発火の回数で数え、評価を走らせなかった発火も「前回の発火」として扱う（後ろの発火は遅れない）。eval schedule の既定は 1（全 env・全タグ）。評価専用 Run など、学習直後の評価が要る Run は 0 を書く。
2 つのキーは `interval` と同じ読み口で読む。負の値は Env を作る前に `ANET_SYSTEM_ERROR` で止め、メッセージにキー・指定値・期待（0 以上）を入れる。`interval = 0` のタグ（休止）でも、書いてあれば読んで検証し、エラーにも WARN にもしない。`run.eval_schedule.[tag]` に 2 つのキーだけを書いて `interval` を書かないと、今と同じく `interval` が無いエラーで止まる。
`IntervalGate` は変えない。offset と warmup_count は `EpisodeEvalObserver` が受け持つ。learn step が offset に届くまではゲートに問い合わせず、届いたら learn step から offset を引いた値でゲートに問い合わせ、発火を数えて先頭の warmup_count 回では評価を走らせない。bucket-crossing と、catch-up しない規則はゲートのまま。
本体の設定で offset を持つタグの interval をプロファイルや bat で変える箇所では、その offset が新しい interval でも意図どおりかを確かめ、違えば offset も書く。

### 実装契約

発火のうち、最初から数えて `warmup_count` 回目までは評価を走らせない。それより後の発火では、今と同じく評価セッションを 1 回始める（背景で走らせ、前回が終わっていなければ待つ）。評価を走らせなかった発火も、3 の「前回の発火」になる。learn step は減らない。
`warmup_count` の warmup は、Learner の `update_warmup_steps`（学習を始める前に経験をためる期間）とは別のもので、評価の発火の列の先頭から数える回数である。設計文書とユーザーガイドのキーの説明にも、この 1 文を書く。
ImageCls・GridMaze_muzero・LunarLander の `[test1]`・`[test2]`: 評価のタグが 1 本だけか休止なので、変えない。
旧い挙動を探す検索語は、`learn_step 0`・`learn step 0`・`learn_step=0`・`初回`・`互いに素`・`同位相`・`ShouldFire`・`scheduled (`・`1009`・`53`・`503`・`11`・`12` である。`1009` などの数は、`run.eval_schedule` を含む行に限って引く。

### 受入

LunarLander の既定（`run.@skip` を含む）を `app.batchrun.exp_exit_step=300000` で 1 本流す。Run ログで次を確かめる。
各タグの最初の `session end` の learn_step が、そのタグの offset + interval 以上である（`[eval_target]`・`[eval]` は 200、`[greedy]` は 1,050）。
`[greedy]` の `session start` の learn_step が、`[eval_target]`・`[eval]` の `session start` のどれとも一致しない。
既定のテストスイートが通る。既知の失敗（RFD0093 の attention のテスト 1 件）は、本 RFD の前後で変わらないことを確かめる。

### 影響

書かない Run は、どの env でも、各タグの評価が 1 回（学習直後の 1 回）減る。評価のセッションの足し引きで同じ seed の学習の軌跡が変わるので（wiki「評価と比較の条件」）、本 RFD のあとの Run は、前の同じ seed の Run と、学習の初めから軌跡が変わりうる。新旧の実行ファイルを決定的モードで突き合わせる比較は、新しい設定で取り直す。
Atari の ε=0 の評価の系列は、5.05M（`run.@eval2ch`）か 10.1M（`run.@eval2ch_r1`）から始まる。旧い Run と k 番目の点で対にするときは、新しい Run の k 番目を旧い Run の k+1 番目と対応させる。実験記録の 9 点の対の表は learn_step=0 の点を使っていないので、作り方は変わらない。
ADR 0045 の Consequences にある「初回発火の抑止が入るまでの移行の依存」は、要らなくなる。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [Atari横断](../../concepts/subjects/atari.md)
