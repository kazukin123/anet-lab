# `@episode_end` で読む値を、どの target でもそのエピソードの値にそろえる 暫定 PRD

> 2026-10-20 起票（暫定。グリル前）。質問はせず、暫定の既定を置いて §5 に決めることを残した。
> 起点: [PRD 097](097_env_auto_reset_after_notification_10prd.md) の再グリル（2026-10-20）。`$env` を「今の env の状態」を返す target として残した。そのため、`@episode_end` で `$env` を読むと、終わった lane はリセット後の値になる（097 の §4.3.2）。あわせて、手戻りの分析の中で、`$runner` の `@episode_end` の scalar が集計の途中の値を読むことが分かった。ユーザーの指示で、この 2 つを 1 本にまとめた。
> 関連: [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md)、[ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)、[140_observability.jp.md](../design/140_observability.jp.md)、[CONTEXT.md](../../CONTEXT.md) の「trace」「速報値／確定値」。

## 1. 目的と問題

`EpisodeEndEvent` は、終わった lane（episode group）ごとに 1 本ずつ出る。その通知の中で読む target のうち、次の 2 つは「そのエピソードの値」にならない。

- **`$env`**: 097 の実装のあと、env への問い合わせは今の env の状態を返す。自動リセットは `Step` の中で済んでいるので、終わった lane はリセット後の値になる。例えば `$env lives @episode_end` は、次のゲームの残機を黙って読む。遷移の結果は `$env_step` で読むのが正しい（097 の D1）。
- **`$runner` の集計**（`mean.episode_return`・`max.episode_steps` など）: Runner は、終わったエピソードを集計に 1 本加えるたびに通知を出す（[trainer.cpp:90-103](../../core/anet-core/src/trainer.cpp)）。同じ step に 2 本が終わると、1 本目の通知の中では集計に 1 本分しか入っていない。`$runner mean.episode_return @episode_end` は、1 本目の通知で 1 本目だけの平均を、2 本目の通知で 2 本の平均を読む。EMA は両方を取り込む。`$runner` の集計は lane の指定を受け取らない（[trainer.cpp:171-188](../../core/anet-core/src/trainer.cpp)、ADR 0037 の「`$runner` は index を無視してカウンタ等を返せる」）。

今のリポジトリの設定には、どちらに当たる宣言も無い（`@episode_end` の scalar は 0 行。`@episode_end` の trace は Atari の 4 行で、097 が `$env_step` へ移す）。実害は今は無いが、宣言すると黙って違う値になる。

### ゴール

- `@episode_end` で読む値が、どの target でも、終わったエピソード（その Step）の値になる。そうならない読み方は、構築のときに止まる。

### 非ゴール

- `$env_step` の読み方。097 で決めた（1 step に 1 回読む、終わった lane だけを集約する）。
- `@session_end` の評価セッションの集約（ADR 0034・0037）。
- `@train` で読む値の意味。

## 2. 現状（2026-10-20 にコードで確かめたこと）

- `RunnerBase` は、Step の結果からエピソードの集計を作り、終わった group を昇順に 1 本ずつ集計へ加えては `EpisodeEndEvent` を出す（[trainer.cpp:90-103](../../core/anet-core/src/trainer.cpp)）。
- `RunnerBase::GetScalar` の `episode_return`・`episode_steps` の集計は、その集計を返す（[trainer.cpp:182-187](../../core/anet-core/src/trainer.cpp)）。
- `@episode_end` の scalar の読み口は、通知のたびに値を取って EMA を更新し、そのあとで間隔の判定を見る（[observers.cpp:1110-1127](../../core/anet-core/src/observers.cpp)）。
- 097 は、1 step に 1 回だけ読む規則を `$env_step` の scalar に限った（097 の §4.2 の 12）。ほかの target は今のまま、通知のたびに読む。

## 3. 提案の骨子（暫定）

| 対象 | 暫定の既定 | ほかの案 |
|---|---|---|
| `$env` の `@episode_end`（scalar・trace） | 構築のときに止める。案内の文に「遷移の結果は `$env_step` で読む」と書く | 止めずに、140 に「今の状態を読む」と注意を書くだけにする |
| `$runner` の集計の `@episode_end`（scalar） | Runner が、同じ step で終わったエピソードを全部集計に加えてから、通知を出す。scalar の読み口は、097 の `$env_step` と同じく 1 step に 1 回だけ読む | `$runner` の集計を `@episode_end` で止め、`@train` で読ませる。今のまま、140 に注意を書くだけにする |

設定の例（今は宣言が無いので、暫定の既定での振る舞いを示す）:

```
M1.[42_env/30_lives_at_end]  = $env mean.lives @episode_end $exp_step                 # 構築のときに止まる（$env_step で読む）
M1.[40_train/10_return_end]  = $runner mean.episode_return @episode_end $exp_step     # 同じ step の全エピソードの平均を 1 回だけ記録する
```

## 4. 作らないもの・先送り

なし（暫定の起票のため、グリルで決める）。

## 5. 決めること（グリルで詰める）

| # | 決めること | 暫定の既定 |
|---|---|---|
| 1 | `$env` を `@episode_end` で止めるか | 止める。黙って違う値になるのを防ぐため |
| 2 | `$runner` の集計の直し方 | 集計を先に済ませてから通知する。そのうえで 1 step に 1 回だけ読む |
| 3 | 1 step に 1 回だけ読む規則を、`$runner` のほかの target（`$agent`・`$actor`）の `@episode_end` の scalar にも広げるか | 広げる。同じ値を lane の数だけ EMA に入れないため |
| 4 | trace で `$runner` の集計を読む宣言の扱い | 止める。trace は個体の値なので、集計は合わない |
| 5 | 着手の時期 | 097 の実装のあと。097 が作る 1 step に 1 回の読み口を使う |

## 6. 成功の測り方の目安

- 単体テストで、同じ step に 2 lane が終わるとき、`$runner mean.episode_return @episode_end` が 2 本の平均を 1 回だけ記録し、EMA を 1 回だけ更新する。
- `$env ... @episode_end` を宣言すると、構築のときに止まる。
