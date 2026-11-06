# エピソード長は SHARED でも Step() 回数とし、episode return の総和とは非対称にする

`episode_return` は「一つの episode group が開始から完了までに得た reward の総和」で、`PER_LANE` では当該 lane の総和、
`SHARED` では全 lane・全 step の総和になる。lane をまたいで足す定義であり、`num_envs` を増やせば値も増える。

エピソード長を同じ継ぎ目（`EpisodeStatsAccumulator`）へ足すとき、同じ流儀に揃えるなら
`SHARED` の `episode_steps` は `Step()` 回数 × lane 数になる。定義文の対称性はそのほうが高い。

**`episode_steps` は `SHARED` でも `Step()` の呼び出し回数とし、lane 数を掛けない**ことを決定する。
`PER_LANE` は当該 lane の episode が開始から完了までに要した `Step()` 回数、
`SHARED` は batch 全体が 1 episode なので `Step()` 呼び出し回数そのもの、という定義にする。
`episode_return` との非対称は意図的なものであり、揃え忘れではない。

理由は、**長さは量ではなく時間だから**である。lane 数を掛けた値は「そのセッションで消費した遷移数」であって
「エピソードがどれだけ続いたか」ではない。掛けると `num_envs` を変えただけで「エピソード長」が変わり、
env の性質ではなく実行構成を測る値になる。また `PER_LANE` の値と同じ軸で比較できなくなり、
`episode scope` の違う env を横に並べた瞬間に意味が壊れる。

単位は Env の `Step()` 回数＝agent step である。frameskip の前の生フレーム数ではない。framework は frameskip を知らない。

## Considered Options

- **`SHARED` で lane 数を掛ける（`episode_return` と同じ流儀）**: 定義文が 1 つの規則で書ける。
  しかし得られる値は遷移数であり、`num_envs` 依存で env の性質を表さない。`PER_LANE` と同じ軸に乗らない。却下。
- **`SHARED` を未対応にして NaN を返す**: 非対称を持ち込まずに済む。
  `SHARED` を使う ImageCls（`core/envs/imagecls1/src/ImageClsEnv.cpp:31`）でエピソード長が永久に見えなくなる。
  1 episode = 1 epoch 相当で minibatch 数が出るのは十分に有用なので却下。
- **`episode_return` 側を lane 平均へ変更して両者を揃える**: 対称性は得られるが、既存 Run の eval reward の
  意味が変わり、過去の実験記録と比較できなくなる。既存の metrics 契約を壊す代償が大きすぎる。却下。
- **`episode_steps` を env 側の責務のままにする**: 決定そのものが不要になる。
  現に AtariEnv は `game_len`、GridMazeEnv は `episode_len` と名前すら揃わず、他 env では見えない。
  `episode_return` が全 env 共通で出ているのに長さだけ env 依存という非対称のほうが大きい。却下。

## Consequences

- **`episode_return` と `episode_steps` は同じ `SHARED` group に対して別の畳み方をする。**
  この非対称は CONTEXT.md の「エピソード長」の項目に明記する。
- `SHARED` を使うのは現状 ImageCls だけである。1 episode = 1 epoch 相当なので、値は 1 epoch あたりの minibatch 数になる。
- `PER_LANE` / `SHARED` のどちらでも、`mean.episode_steps` は env を横断して同じ意味で読める。
  `eval_batch_size` を変えても値の意味は変わらない。
- **frameskip のある env では、生フレーム数は依然として env 側の責務である。**
  Atari の `game_frames` は存置する。`game_len` も、`episodic_life = true` では framework の episode（= 1 ライフ）と
  境界が一致しないため別概念として存置する。
- GridMazeEnv の `episode_len` は汎用版と同一物なので、クリーンブレーク方針に従って削除し config を移行する。
- metrics 契約なので、後から定義を変えると既存 Run と比較できなくなる。変更するときは新しい ADR で行う。
- 詳細設計は `docs/memo/done/075_episode_steps_and_eval_session_log_10prd.md`。
