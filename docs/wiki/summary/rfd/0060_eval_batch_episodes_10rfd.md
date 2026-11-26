# 評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する

## 元の材料

[評価セッション契約: eval_batch_size（lane 数）と eval_episodes（本数）で N 本平均を 1 点として記録する](../../../memo/done/0060_eval_batch_episodes_10rfd.md) blob:a13e70100e9432cd1e7a4fa915d2eee7c8cd838b

## 要点

- 評価の並列度と、一つの評価点に集約するエピソード数を独立に指定する。
- eval_batch_size は物理 lane 数、eval_episodes は完走して採用する論理エピソード群の数を表す。
- 個体ごとに終わる環境と、バッチで一つのエピソードを共有する環境で group の数え方を変える。
- 複数エピソードの値を記録するときは、mean・max・min などの集約方法を明示する。
- 採用したエピソードだけで1点を作り、評価に参加しなかった lane が結果に入ることを防ぐ。

## 内容

### 背景

現状の実害は eval 値の分散である。Breakout で 32M step まで学習した Run の `51_eval1/10_game_score_mean`（旧名 `10_episode_score_mean`）直近 7 点は `[44, 57, 83, 56, 86, 97, 38]` で、38〜97 と 2.5 倍の幅がある。探索記録では後期 Breakout で 44〜227 の幅も観測されている。文献値（Nature DQN Breakout 401.2）は 30 エピソードの平均で報告されており、1 本評価のままでは学習曲線の判読にも文献比較にも精度が足りない。探索ブロック 14 では 40〜50M step の eval1 で 5% 差（n=25〜29）が判定不能だった（探索記録 :955）。
現在は 2.5M step の window 平均（3〜4 点）で代用しているが、これは「異なる時点の方策の平均」であり、単一時点の N 本平均とは意味が違う。

### 現行コードで確定している事実（2026-09-02 時点）

`train.eval_schedule.[eval1].interval` は現行 1000（Atari.txt:708、コメント「旧 2500」）。
帰結: 旧候補 B の「完走した lane に no-op を送り続ける」は不成立（no-op でも Step 冒頭で消える）。N 本の確定値は、完走した Step の直後に誰かが控えなければ集約できない。

### 問題

残り N-1 lane 分の env ステップと NN 推論が捨てられる。
現在の eval は batch=1 の CNN forward を 1 step ずつ回しており GPU がほぼ遊んでいる（`train.eval_device_type = cuda`）。lane 数を増やしても GPU 推論は 1 回のバッチ処理で済むため、N 本取るコストは N 倍にならない。
ゴール: `eval_batch_size`（並列度）と `eval_episodes`（統計サンプル数）を分離し、1 回の network snapshot に対する N episode の集約を `@episode_end` 1 点として記録する。Atari / DropMerge / ImageCls の契約を同じモデルで説明できる状態にする。

### 確定契約

`eval_episodes` の消費者は EvalSessionEnv、`eval_batch_size` の消費者は BatchEnvBuilder のままとする。
`BatchEnvSpec` に `EpisodeScope episode_scope` を追加する。既定は `PER_LANE` とし、`ToJson()` は `"per_lane"` / `"shared"` の文字列で出力する。
完了時にキャプチャする値はその Step の確定値であり、次 Step へ持ち越さない。

### 受入条件

`BatchEnvSpec`: `PER_LANE` / `SHARED` と JSON 文字列、既定 `PER_LANE` を固定する。
動的グラント: PER_LANE L=2,N=3 は両 group の初回 episode を採用し、先に完了した group の次 episode へ3本目をグラントする。3本目より遅い他 group の初回採用 episode も最後まで待ち、最初の3完了だけを選ばない。L=4,N=2 は先頭2 groupだけへ初回グラントし、他groupの結果を無視する。同一Stepで複数groupが完了し残りグラントが不足する場合はgroup index順に割り当てる。SHARED G=1,N>1は同じgroupへ直列にグラントする。発行、capture、結果は常に正確にN件で、同じseed・決定論backendではグラント列と結果順が再現する。

### 成功指標（P3 の一回限りの運用検証）

`σ30 ≈ std(B-C) / sqrt(2)`、`σ1 ≈ std(A-(B+C)/2)` とし、`σ1 / σ30 >= 4` を確認する（理想は `sqrt(30) ≈ 5.5`）。

### コスト見積もり（P3 で実測すること）

PER_LANEでgroup数G=Lなら、採用episodeを完了したgroupが残りの採用権を受け取るため、N>Gでは固定quotaよりlane間の仕事量を平準化できる。session長は概ね採用episodeの総step数をGで分担する時間に、最後まで残る採用episodeのtailを加えたものになる。N=Gでは各groupに1本だけをグラントするためL本の平均長ではなく最長episodeまで待つ。SHAREDはG=1なので、N windowを直列に評価する。
全N件の採用権を発行した後も、採用権を持たないgroupを含む全groupを通常どおりStepするため、最後の採用episodeを待つtailではCPU浪費が残る。実測で支配的になった場合だけfreeze / Step除外を別設計する。

### 却下した旧解決候補（経緯）

旧候補 A/B の停止条件だけの変更は F1 により N 本平均にならない。走行中の全episodeから終了後に最初のN完了を選ぶ終了順採用は、長いepisodeを打ち切って短いepisodeへ偏るため却下する。一方、開始時の動的グラントは次episodeの結果を見る前に採用を決め、グラントした固定N本をすべて待つため、このbiasを持たない。
Runner集中集約は`$env` metricsの全面移行を招くため、captureと集約はBatchEnv decoratorへ置く。全session無条件Reset案はauto-reset済みstateを二重Resetし、N=1のRNG系列とImageCls preloadを壊すためfresh判定へ置き換えた。ImageClsのlane別採用案はbatch-level accuracyと一致しないためSHARED groupへ置き換えた。

### Atari プロトコルとの関係

Classic / 100k は train で `episodic_life=true` を使い続けてよい。eval だけ false にする。v5 はもともと false。
lane 独立性は Classic の noop、v5 の sticky action で確保される。旧 `@v5_noop0` かつ Greedy ε=0 の Run は同一軌道複製なので N 本平均の比較対象にしない。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
