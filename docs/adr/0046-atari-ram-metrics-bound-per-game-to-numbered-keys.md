# AtariEnv のゲーム固有 RAM メトリクスは番号付き汎用キーで参照し、意味はゲーム別の RAM 定義が持つ

ALE のゲーム設定（`KungFuMasterSettings` 等）が読む RAM はスコアと残機だけで、面やボスの体力のような
ゲーム固有の進行情報は RAM を直接読むしかない。RAM の番地とその意味はゲームごとに違う。
一方、metrics の定義（`metrics.trace` / `metrics.scalar`）は Atari-5 の batch 起動が 1 本の腕チェーンを
全ゲームに使うため、ゲームを変えるたびに書き換えたくない。イベントの数だけ metrics を増やす運用も持たない。

**AtariEnv は番号付きの汎用キー `ram_metric.[n]`（n は 1 以上の整数）で RAM メトリクスを返し、
番地・畳み方・ラベルは設定のゲーム別 RAM 定義が持ち、番号付け行で結ぶ**ことを決定する。

```
AtariEnv.ram_metric.[kung_fu_master].[floor_clear] = 0x9F inc_count
AtariEnv.ram_metric.[kung_fu_master].[boss_kill]   = 0xCC reach_count:0
AtariEnv.ram_metric.[kung_fu_master].metrics       = 1:floor_clear 2:boss_kill
metrics.trace.@atari.[42_env/episode] = $train @episode_end $env game_score ... ram_metric.[1] ram_metric.[2]
```

- どこかのゲームのブロックが n を番号付けしていれば既知キー。今のゲームに n が無ければ NaN、
  どのゲームにも無ければ未知（`nullopt`）。確定タイミングと NaN の慣行は `game_score` と同一にする。
- 畳み方は `max_seen` / `min_seen`（v0 を含む到達値）と `inc_count` / `dec_count` / `reach_count:N`（遷移の回数）の
  閉じた 5 語にする。`min` / `max` の語は lane 方向の集約 prefix（`min.` / `max.`）と紛れるので使わない。
- 観測はゲーム中の全フレーム、集計単位は実ゲーム 1 回（soft reset をまたぐ）、`reset_game()` ごとに仕切り直す。
- 番号の無い定義は評価せず、RAM 知識の置き場として設定に残せる。上書き prefix（`run.eval.[tag].env`）からは読まない。
- 番号には全ゲーム共通の大まかな意味を持たせる（1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数）。当てはまらないゲームは番号を付けず NaN にする。

理由は 3 つある。**番号が metrics 側の固定の接点になる**ので、ゲームを変えても定義を変えずに済み、
当てはまらないゲームでは trace に `null` 列が残るだけで落ちない。**意味を設定側に置く**ので、
番地の当たり付けを再ビルド無しで回せ、フレームワークは構文・番地範囲・語彙・参照整合だけを検証し、
番地の意味は利用側が持つという責任境界にそのまま乗る。**定義と番号付けを 2 段に分ける**ので、
評価しない定義を知識として残せ、番号を付け替えても定義行は動かない。

## Considered Options

- **番地をキーに直書きする（`ram_min.[0xCC]` / `AtariEnv.ram_watch = 0xCC,0x9F`）**: 実装は最小だが、
  キーから意味が読めず、追跡対象の宣言と参照が 2 か所に分かれ、metrics 定義がゲーム別になる。却下。
- **コードのゲーム別表（HNS 表と同じやり方）**: 名前は意味を持つが、番地を試すたびに再ビルドが要る。
  HNS 表がコードにあるのは固定の式が使う物理定数だからで、どのバイトをどう畳むかは解析上の選択であり
  `game_score.ge.[N]` をキー側のパラメータにした先例に近い。却下。
- **ラベルをキーにする（`ram.[boss_hit]`）**: 意味は読めるが、イベントの数だけ metrics が増え、
  ゲームを変えるたびに metrics 定義を書き換える。却下。
- **購読された env キーから追跡対象を自動で決める**: `EvalSessionEnv` が受け取る購読キーは eval scalar の
  `@session_end $env` だけで内側の env へ渡らず、train env は metrics 解析より先に構築される。配線を足す価値が無い。却下。
- **畳み方の語に `min` / `max` を使う**: `max.ram_metric.[1]` が「max の max」になり、時間方向と lane 方向のどちらか読めない。却下。
- **番号に意味を持たせない（純粋な接点にする）**: 仕組みは同じだが、同じ番号の列がゲームごとに別の意味になり、
  trace をゲームをまたいで読むたびに config を引くことになる。却下。
- **Step の frame skip 窓後に 1 回だけ観測する**: 実装は 1 箇所で済むが、回数系が窓の途中と reset 中の遷移を
  取りこぼし、「見ない区間」の但し書きが契約に残る。毎フレームの比較は 1 バイトの参照で、コストは無視できる。却下。

## Consequences

- 番号の意味は Run の `config/config_data.txt`（`AtariEnv.ram_metric.*`）で引く。`metrics.trace.defs` はラベルを持たない。
  env のメタデータを定義レコードへ載せるかは別の判断にする。
- 番号付けの無いゲームでは trace が `null` 列、scalar は行が出ない。既定の Atari.txt は番号 1〜3 を
  4 本の trace 行と `42_env` の `mean.` 3 本に載せる。
- 5 語は閉じた列挙で、未知語は設定読み込み時に fail-fast する。語を足すのは既存契約を壊さない追加である。
- 進行中の値は AuxData `ram_metric.<label>` で AtariView のオーバーレイに出し、ゲーム完了ログにも同じ値を出す。
  観戦（EvalPanel）と記録（trace / scalar）で同じ定義を使う。
- 他の env がゲーム固有の指標を持つときは、「番号付き汎用キー + env 側の意味定義」の同型を採るかをこの ADR を先例に判断する。
- metrics 契約なので、後から畳み方の意味や NaN の規則を変えると既存 Run と比較できなくなる。変更するときは新しい ADR で行う。
- 詳細設計は `docs/memo/082_atari_ram_progress_metrics_10prd.md`。
