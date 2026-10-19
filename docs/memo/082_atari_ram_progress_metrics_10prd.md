# Atari のゲーム固有イベントを RAM メトリクスとして取る — 番号付き汎用キーへゲーム別に紐づける PRD

> 起点: 2026-09-23、kung_fu_master を 2 seed 回した結果、最終スコアの断絶は 75,000 点の 1UP だと分かった。一方で「ボスに攻撃を当てたか」「面を抜けたか」は今のメトリクスでは測れない。ボス討伐を直近の目標に置くため、まず測れるようにする。2026-09-24 のグリルで、原案の「番地をキーに直書きした min / max」を退け、番号付きの汎用キーへゲーム別に紐づける形へ改めた。同日、Atari-5 と Breakout の RAM も調べ、番号に全ゲーム共通の大まかな意味（1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数）を持たせた。
> 関連: [ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)（本 PRD の決定）、[081](081_evalpanel_episode_trace_10prd.md)（NG2 で本件を別 PRD とした）、[ADR 0037](../adr/0037-metrics-trace-channel-and-session-end-event.md)（trace チャネル）、[220 §4.7](../design/220_atari_env.jp.md)（AtariEnv の Env accessor）、[2026-09-23 の実験記録](../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)。

## Context / Problem Statement

### 今は測れないもの

AtariEnv がゲーム 1 回ぶんの値として出すのは `game_score` / `game_len` / `game_frames` / `hns57` / `hns49` / `game_score.ge.[N]` と、常時値の `lives` だけである。ALE の `KungFuMasterSettings` が読む RAM もスコア（0x98〜0x9A）と残機（0x9D）だけで、面やボスの情報は持たない。

2 seed の実測では、最高のゲームは 1 機あたり約 8,180 フレームのタイマーいっぱいまで粘っていた（4 機なら 32,745、1UP で 5 機なら 40,861 フレームで打ち切りがそろう）。このとき、ボスに当てたか・面を抜けたかは記録に残っていない。

### エミュレータで分かったこと（2026-09-23）

ALE v0.12.0 を直接動かす使い捨てのプログラムで調べた。sticky なし、決まった操作パターンを繰り返す。ボスまで進むために、自機の体力とタイマーを毎フレーム書き戻した。換算ボーナスを測るときは、ボスを倒した時点で書き戻しを止めた。

| アドレス | 内容 | 根拠 |
|---|---|---|
| 0x98 / 0x99 / 0x9A | スコア（BCD、上位 → 下位） | ALE の `KungFuMasterSettings` |
| 0x9B / 0x9C | タイマー（BCD、上位 / 下位。2000 から減る） | 6 つのフレームで画面の表示と一致 |
| 0x9D | 残機（ALE の `lives()` = (b & 7) + 1、0xFF でゲームオーバー） | ALE |
| 0x9F | 面の番号（1 始まり） | 面 1 → 2 の切り替わりで 1 → 2。切り替わりは 1 回しか見ていない |
| 0xCB | 自機の体力（満タン 39、0 で 1 機失う） | 画面の PLAYER バーの長さと一致 |
| 0xCC | ボスの体力（満タン 39） | ボス戦の 340 フレームすべてで ENEMY バーの長さと一致 |

ゲームの仕組みとして次を確かめた。

- **1UP は 75,000 点の 1 回だけ。** 25,000 / 50,000 / 60,000 / 100,000 / 125,000 / 150,000 / 200,000 / 225,000 では残機は増えない。
- **ライフを失うと面の最初に戻る。** タイマーは 2000 に戻り、スコアは残る。
- **ボスは 1 発で体力が 4 減る。** 4 は 1 フレームでまとめて減るので、`dec_count` は命中 1 回を 1 と数える。戦闘中に体力が 1 戻ることがあり（2026-09-24 の確認では 1 戦に 2 回、39 → 35 → 31 → 27 → 28 → 24 → … → 12 → 13 → 9 → 5 → 1 → 0）、倒すまでの命中は 10〜11 発になる。倒したフレームに +2,000 点がまとめて 1 回で入る。間合いを詰めながら攻撃しないと当たらない（その場での攻撃やボタンの押しっぱなしは 1 発も当たらない）。行動を 4 フレームずつ続ける粒度（例: `LEFTFIRE` と `LEFT` を 4 フレームずつ交互）でも倒せる。途中でやられると、ボスの体力も戻る。
- **倒したあと面の出口まで歩くと、残りを点数に換算する。** 体力は 1 あたり 100 点（4 フレームに 1 単位）、タイマーは 1 あたり 10 点（1 フレームに 16 単位）。体力の換算中はタイマーが止まる。換算が終わると次の面に移り、タイマー 2000・自機の体力 39・ボスの体力 39 に戻る。残機は減らず、画面下の帯の色が青から緑に変わる。

### 他のゲームの RAM（2026-09-24）

同じ使い捨てプログラムで、Atari-5 と Breakout の「面の区切り」を探した。qbert は面クリアの cheat（タイル 21 個を目標色に書き換えて 1 回跳ぶ）で 20 面まで進め、phoenix と breakout は残機を書き戻しながら決まった操作で遊ばせた。

| ゲーム | アドレス | 内容 | 根拠 |
|---|---|---|---|
| qbert | 0xE3 | 面番号（0 始まり。20 面を抜けると 16 へ戻る） | 20 面ぶんの面クリアで 1 ずつ増える。面の途中で死んでも変わらない |
| qbert | 0x82 | 今の面の立方体の側面色（ROM 0xE1A からの 20 面ぶんの表を面の開始時に写す） | 8 面だけ 0x00（黒）。ROM を 1 バイト変えた版で、変わる RAM はこのバイトだけ |
| phoenix | 0xCA | ウェーブ番号（0〜4 を繰り返す。0・1 = 小鳥、2・3 = 大鳥、4 = 母艦） | 20 万フレームで 0 → 1 → 2 → 3 → 4 → 0 → … |
| phoenix | 0xFC | 周回数（母艦を倒すたびに 1 増える） | 0xCA が 4 → 0 に戻るのと同じフレームで増える |
| breakout | 0x9E〜0xA3 | 最下段のブロック（1 バイトに 2 個ぶんのビット、満タン 192） | 壁の作り直しの時だけ 0 → 192 に戻る。作り直しは累計スコアがちょうど 432 になったフレーム |
| breakout | 0xCC / 0xCD | スコア（BCD。0xCC が百の位で 0〜8、0xCD が下 2 桁） | ALE の `BreakoutSettings`（`readRam` 76 / 77）。15 万フレームの記録で、復元したスコアが累計報酬と全フレームで一致 |

- **breakout に「何枚目の壁か」を持つバイトは無い。** 1 枚目を消すと同じフレームでブロックが満タンに戻る。2 枚目を消すと作り直しは起きない（スコア 864 で止まる）。
- **breakout のスコアの百の位は値を飛ばさない。** 1 フレームの得点は最大 7 点なので、0xCC は 0 → 1 → … → 8 と 1 ずつ増える。ゲームの中でスコアは減らないので、0xCC が N に入るのはスコアが N × 100 を初めて越えた 1 回だけである。
- **name_this_game に面の区切りは無い。** 1 万点ごとに増える 0xC4 はスコアの万の位（BCD）で、0xD6 はその 8 倍の難易度段階である。スコアから決まるので RAM メトリクスにする意味が無い。
- **battle_zone と double_dunk** は面の区切りが無いゲームで、今回は調べていない。

### なぜ要るか

- **ボス討伐を目標に置くため。** ゲームごとに「ボスに当てた回数」「倒した回数」「抜けた面」を数えたい。
- **スコアの平均では見えない、方策の質的な違いを行動で比べるため。** 例えば NN 構成の違いを、「先へ進むか」「ボスと戦うか」で比べられるようにしたい。
- **ゲームを変えるたびに metrics 定義を書き換えたくないため。** Atari-5 の batch 起動（bat で `E1.game=` だけを変える）は 1 本の腕チェーンを全ゲームに使う。ゲーム固有の指標もその形に乗らなければ、指標の数だけ metrics 定義が増え、ゲームごとに腕を切り替える運用になる。

### 設計を縛る事実

- RAM の情報源は ALE の `getRAM()`（128 バイト。`act()` / `reset_game()` のたびに更新される内部キャッシュへの参照）だけ。リポジトリ内に RAM を読む箇所はまだ無い。ALE のゲームコードはバス番地（0x80〜0xFF）で書き、`readRam` は `peek((offset & 0x7F) + 0x80)` である。
- AtariEnv の accessor には 2 系統の先例がある。ゲーム 1 回で確定して未確定 step は NaN の `game_score` 一族と、常時値の `lives`。パラメータ付きキー `game_score.ge.[N]` は呼び出しごとに角括弧をパースし、不正なら fail-fast（`ParseGameScoreThreshold`）。
- metrics 側は `$env <key>` を `@episode_end` で `GetScalar` するだけである。未知キー（`nullopt`）は trace で fail-fast、scalar observer は NaN を出力しないので全 lane が NaN の系列は行が出ない。購読キーが env に届く仕組みは無く、train env は metrics 解析より先に構築される（`trainer.cpp` の `CreateTrainEnv` が `ObserverFactory` より前）。
- 設定基盤は `Prefix.[tag].sub = 値` の列挙（`ConfigData::MakeSubConfigData`。actor スロットや `run.eval` で使用中）を持つ。サブキーは `[label]` / `metrics` の形で返る。16 進の読み取りは無く、数値はカンマを除去してから読む。`${key}` 展開は最終値だけなので `.$` 選択をゲーム名で切り替えることはできない。
- `MakeSubConfigData` はサブキーを最初の `]` で切るため、角括弧入りの上書き prefix（`run.eval.[tag].env...`）では正しく分解できない。
- `AtariEnv` は `final` で `ale_` は private。test-only subclass で `setRAM` を叩く seam は無い。
- AuxData は `std::unordered_map<std::string, torch::Tensor>`。AtariView は固定キー（`game_score` / `lives` / `game_len` / `game_frames`）を lane 0 から読み、1 行のラベルに整形する。
- Pong はスコアを RAM index 13（CPU）/ 14（player）に持つ（バス番地 0x8D / 0x8E）。ROM 依存テストの題材にできる。

## ゴール / 非ゴール

**ゴール:**

- **G1**: ゲーム固有のイベントの発生状況（回数と到達値）を、ゲーム 1 回ぶんの AtariEnv スカラーとして出す。確定のタイミングと NaN の扱いは `game_score` と同じにする。
- **G2**: 番地と意味はコードに持たず、設定で宣言する。番地の当たり付けは再ビルド無しで回せる。
- **G3**: metrics 側はゲームに依らない番号付きキー `ram_metric.[n]` で参照し、ゲームを変えても metrics 定義（trace / scalar）を書き換えない。
- **G4**: kung_fu_master・qbert・phoenix・breakout の定義を既定設定に入れる（§設定例）。番号は §8 の慣例に従う。
- **G5**: 進行中の値を AtariView のオーバーレイに出し、EvalPanel の観戦中に目で確かめられる。

**非ゴール:**

- **NG1**: RAM の値を報酬に使うこと（報酬の整形）。学習の信号は変えない。
- **NG2**: G4 の 4 ゲーム以外の RAM 表を整えること。仕組みは全ゲーム共通で、表は必要になったゲームから足す。name_this_game・battle_zone・double_dunk は定義を置かない（NaN）。
- **NG3**: EvalPanel の trace。081 で扱う。
- **NG4**: 進行中の値を scalar として毎 step 出す常時値キー（`lives` 型）。
- **NG5**: 初回発生の時刻（何 step 目で初めて当てたか）。
- **NG6**: `metrics.trace.defs` にラベルを載せること。env のメタデータを定義レコードへ出す形は別の判断にする。

## 契約

### 1. 設定: RAM 定義と番号付け

```
AtariEnv.ram_metric.[<game>].[<label>] = <番地> <畳み方>
AtariEnv.ram_metric.[<game>].metrics   = <n>:<label> [<n>:<label> ...]
```

| 要素 | 書式 | 意味 |
|---|---|---|
| `<game>` | `AtariEnv.game` と同じ ROM 名。小文字 snake_case（`[a-z0-9]+(_[a-z0-9]+)*`）だけを受け付ける | このブロックが効くゲーム。他のゲームでは評価しない。ROM 名になりえない綴りは fail-fast |
| `<label>` | `[A-Za-z0-9_]+` | 人が読む名前。オーバーレイとログに出る。metrics 側の参照には使わない |
| `<番地>` | `0x80`〜`0xFF` の 16 進（バス番地。ALE のゲームコードと同じ。内部で `& 0x7F`） | 読む RAM バイト |
| `<畳み方>` | §2 の 5 語のいずれか | 毎フレーム列を 1 値に畳む規則 |
| `metrics` 行 | 1 以上の整数 `n` とラベルを `:` で結んだトークンの空白区切り。1 つ以上 | 評価する定義と、metrics 側が参照する番号。番号付けされた定義だけを評価する。行が無いブロックは許すが、空の行は fail-fast |

- 番号の無い定義は RAM 知識の置き場として許し、評価しない。
- ブロックは今のゲーム以外も同じファイルに並べてよい。env は `AtariEnv.game` に一致するブロックだけを使い、他のブロックは検証だけする。
- 定義は既定 prefix `AtariEnv.` からだけ読む。`run.eval.[tag].env.ram_metric.*` の上書きは受け付けない（§7）。
- 受理した定義と `metrics` 行は、他のキーと同じく Module Config（`GetConfigData()`）に載せる。Run の `config/env.*.txt` と eval env ごとの設定ダンプで、番号の意味をそのまま引けるようにするためである。
- eval env のダンプ（`config/env.eval.txt` など）では、他のキーと同じく `run.eval.[tag].env.ram_metric.*` の形で出る。これは実効値の記録で、同じ形を入力に書くと §7 の 6 で fail-fast になる。

### 2. 畳み方

v0 を `reset_game()` 直後の値、v_t を t フレーム目の `act()` 後の値とする。

| 語 | 種類 | 値 |
|---|---|---|
| `max_seen` | 値 | v0 を含む最大値 |
| `min_seen` | 値 | v0 を含む最小値 |
| `inc_count` | 回数 | v_t > v_{t-1} となったフレーム数 |
| `dec_count` | 回数 | v_t < v_{t-1} となったフレーム数 |
| `reach_count:N` | 回数 | v_t == N かつ v_{t-1} != N となったフレーム数。N は 0〜255。レベル（N のまま続いたフレーム）ではなく遷移を数える |

バイトは生の値で比較する。BCD も生の値で単調なので `inc_count` / `dec_count` の向きは保たれる。語は閉じた列挙で、`min` / `max` は lane 方向の集約 prefix（`min.` / `max.`）と紛れるので使わない。

kung_fu_master での対応は次のとおり。

| 定義 | 意味 |
|---|---|
| `0xCC dec_count` | ボスに当てた回数（1 発で 4 減る） |
| `0xCC reach_count:0` | ボスを倒した回数 |
| `0x9F max_seen` | 到達した面（1 始まり） |
| `0xCC min_seen` | 番地の当たり確認。39 未満なら当てている |

死亡時や次の面でボス体力が 39 へ戻るのは増加なので、`dec_count` には入らない。

### 3. 観測とゲーム単位

- フレームを進める全箇所で観測する。Step の frame skip ループ、hard reset 直後の NOOP 群と FIRE 系列、soft reset の NOOP と FIRE 系列である。
- 集計単位は実ゲーム 1 回（hard reset から real game over / truncation まで）。`episodic_life` の soft reset をまたいで数え続ける。`game_score` と同じ単位である。
- `reset_game()` のたびに仕切り直し、直後の値を v0 にする。
- 番号付けの無いゲームでは何もしない（RAM の参照も比較もしない）。

### 4. accessor `ram_metric.[n]`

| 条件 | 返り値 |
|---|---|
| `[n]` が整数でない、または書式が壊れている | fail-fast（`game_score.ge.[N]` と同じ流儀） |
| どのゲームのブロックも n を番号付けしていない | `nullopt`（未知キー。trace では fail-fast） |
| 今のゲームに n が無い | NaN |
| 今のゲームに n はあるが、完了した step ではない | NaN |
| 完了した step | `RecordGameCompletion` で確定したスナップショット |

確定のタイミングと NaN の慣行は `game_score` と同一である。したがって trace は `@episode_end` で完了値を、scalar の `mean.` は完了 lane だけを分母に取る。番号の無いゲームでは trace が `null` 列、scalar は行が出ない。

### 5. AuxData とオーバーレイ

- 今のゲームで番号付けされた定義ごとに、AuxData へ `ram_metric.<label>`（int64、進行中の値）を載せる。
- AtariView は prefix `ram_metric.` を拾い、キー順に既存 1 行の末尾へ `RAM: boss_hit=3 boss_kill=0 floor_clear=1` を足す。無ければ何も足さない。番号のない `floor_max` は評価も表示もしない。

### 6. ゲーム完了ログ

既存の行（`Game over.` / `Game truncated by max_episode_frames.`）の末尾に、番号順で ` ram_metric: floor_clear=0 boss_kill=0 boss_hit=3` を足す。番号付けが無いときは足さない。220 §4.7 の趣旨（metrics / trace を持たない EvalPanel のゲームもログから追える）を保つ。

### 7. 検証（fail-fast）

| # | 検証 | 時点 |
|---|---|---|
| 1 | 全ゲームブロックを検証する。今のゲーム以外も対象 | `AtariEnvConfig` の読み込み時（ROM 不要） |
| 2 | 番地は `0x80`〜`0xFF` の 16 進だけ。10 進、`0x00`〜`0x7F`、範囲外は fail-fast | 同上 |
| 3 | 畳み方は 5 語の閉じた列挙。`reach_count:N` の N は 0〜255。未知語・範囲外・書式不正は fail-fast | 同上 |
| 4 | `metrics` 行: n が 1 未満・整数でない・重複、同じブロックに無いラベル、トークンが 1 つも無い空の行は fail-fast。`metrics` 行の無いブロックと番号の無い定義は黙って許す | 同上 |
| 5 | ブロック内の未知サブキー（`[label]` / `metrics` 以外）、ラベルの字種違い（`[A-Za-z0-9_]+` 以外）、ゲーム名の字種違い（小文字 snake_case 以外）は fail-fast | 同上 |
| 6 | `run.eval.[tag].env.ram_metric.` で始まるキーがあれば fail-fast。ROM の事実は Runner 別に変えない | 同上 |
| 7 | 参照キー `ram_metric.[n]` の規則は §4 のとおり | `GetScalar` 呼び出し時 |

### 8. 番号の慣例

番号は metrics 側の固定の接点だが、全ゲーム共通の大まかな意味を持たせる。trace の列をゲームをまたいで同じ読み方で読めるようにするためである。

| 番号 | 大まかな意味 | 畳み方の目安 |
|---|---|---|
| 1 | 面クリア回数（面・ラウンド・ウェーブ・壁を抜けた回数） | 面番号の `inc_count` |
| 2 | ボス撃破回数（面の最後の強敵・母艦など） | ボス体力の `reach_count:0`、撃破で増えるカウンタの `inc_count` |
| 3 | ボス命中回数（撃破の手前の進み具合） | ボス体力の `dec_count` |

- 当てはまらないゲームは番号を付けない（NaN）。値の尺度はゲームごとに違うので、ゲームをまたいで足したり平均したりはしない。比べるのは「0 か 1 以上か」と、同じゲームの中での増減である。
- 1UP（残機の増加）は入れない。kung_fu_master（0x9D）も qbert（0x88）も、ゲームオーバーで残機バイトが 0x00 → 0xFF に変わり、`inc_count` がこれを増加と数える。
- 番号を足すときは 4 以降に足し、既存の番号の意味は変えない。

## 設定例

```
# --- RAM 定義(220 §4.9 の RAM 表)。番号の慣例: 1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数 ---
AtariEnv.ram_metric.[kung_fu_master].[floor_clear] = 0x9F inc_count      # 抜けた面の数(面番号が増えた回数)
AtariEnv.ram_metric.[kung_fu_master].[boss_kill]   = 0xCC reach_count:0  # ボスを倒した回数
AtariEnv.ram_metric.[kung_fu_master].[boss_hit]    = 0xCC dec_count      # ボスに当てた回数(1 発で 4 減る。1 戻る回復は数えない)
AtariEnv.ram_metric.[kung_fu_master].[floor_max]   = 0x9F max_seen       # 到達した面(1 始まり)。番号を付けないので評価しない
AtariEnv.ram_metric.[kung_fu_master].[boss_hp_min] = 0xCC min_seen       # 番地の当たり確認用。番号を付けないので評価しない
AtariEnv.ram_metric.[kung_fu_master].metrics = 1:floor_clear 2:boss_kill 3:boss_hit

AtariEnv.ram_metric.[qbert].[round_clear] = 0xE3 inc_count   # 抜けた面の数(0xE3 は 0 始まりの面番号。20 面の後に 16 へ戻るのは減少なので数えない)
AtariEnv.ram_metric.[qbert].[side_color]  = 0x82 min_seen    # 立方体の側面色。8 面だけ 0x00。番号を付けないので評価しない
AtariEnv.ram_metric.[qbert].metrics = 1:round_clear

AtariEnv.ram_metric.[phoenix].[wave_clear] = 0xCA inc_count  # 抜けたウェーブ数。母艦ウェーブの突破は 4 → 0 の減少なので数えない
AtariEnv.ram_metric.[phoenix].[boss_kill]  = 0xFC inc_count  # 母艦を倒した回数(1 周ごとに 1 増える)
AtariEnv.ram_metric.[phoenix].metrics = 1:wave_clear 2:boss_kill

AtariEnv.ram_metric.[breakout].[wall_clear] = 0x9E inc_count # 1 枚目の壁を消した回数(最下段のブロックは壁の作り直しの時だけ 0 → 192 に戻る)
AtariEnv.ram_metric.[breakout].[score_ge600] = 0xCC reach_count:6 # スコアが 600 を越えた(game_score.ge.[600] と同じ)。番号を付けないので評価しない
AtariEnv.ram_metric.[breakout].metrics = 1:wall_clear

# --- metrics 側はゲームに依らない番号で参照する(既定の 4 本の trace 行の末尾へ足す) ---
metrics.trace.@atari.[51_eval1/episode] = $eval.[eval_target] @episode_end $env game_score game_len game_frames hns57 ram_metric.[1] ram_metric.[2] ram_metric.[3]
metrics.trace.@atari.[52_eval2/episode] = $eval.[eval] @episode_end $env game_score game_len game_frames hns57 ram_metric.[1] ram_metric.[2] ram_metric.[3]
metrics.trace.@atari.[53_evalg/episode] = $eval.[greedy_dist] @episode_end $env game_score game_len game_frames hns57 ram_metric.[1] ram_metric.[2] ram_metric.[3]
metrics.trace.@atari.[42_env/episode]  = $train @episode_end $env game_score game_len game_frames hns57 ram_metric.[1] ram_metric.[2] ram_metric.[3]

# RAM メトリクス。番号 1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数(220 §4.9)
M1.[42_env/50_ram_metric_1_mean]     = $env mean.ram_metric.[1] @train $exp_step
M1.[42_env/51_ram_metric_1_mean_ema] = $env mean.ram_metric.[1] @train $exp_step $ema ema_alpha:0.001
M1.[42_env/52_ram_metric_2_mean]     = $env mean.ram_metric.[2] @train $exp_step
M1.[42_env/53_ram_metric_2_mean_ema] = $env mean.ram_metric.[2] @train $exp_step $ema ema_alpha:0.001
M1.[42_env/54_ram_metric_3_mean]     = $env mean.ram_metric.[3] @train $exp_step
M1.[42_env/55_ram_metric_3_mean_ema] = $env mean.ram_metric.[3] @train $exp_step $ema ema_alpha:0.001
M1.[42_env/56_ram_metric_1_max]      = $env max.ram_metric.[1] @train $exp_step
M1.[42_env/57_ram_metric_2_max]      = $env max.ram_metric.[2] @train $exp_step
M1.[42_env/58_ram_metric_3_max]      = $env max.ram_metric.[3] @train $exp_step

M1.[51_eval1/50_ram_metric_1_mean]     = $eval.[eval_target] @session_end $env mean.ram_metric.[1]
M1.[51_eval1/51_ram_metric_1_mean_ema] = $eval.[eval_target] @session_end $env mean.ram_metric.[1] $ema ema_alpha:0.1
M1.[51_eval1/52_ram_metric_2_mean]     = $eval.[eval_target] @session_end $env mean.ram_metric.[2]
M1.[51_eval1/53_ram_metric_2_mean_ema] = $eval.[eval_target] @session_end $env mean.ram_metric.[2] $ema ema_alpha:0.1
M1.[51_eval1/54_ram_metric_3_mean]     = $eval.[eval_target] @session_end $env mean.ram_metric.[3]
M1.[51_eval1/55_ram_metric_3_mean_ema] = $eval.[eval_target] @session_end $env mean.ram_metric.[3] $ema ema_alpha:0.1
M1.[51_eval1/56_ram_metric_1_max]      = $eval.[eval_target] @session_end $env max.ram_metric.[1]
M1.[51_eval1/57_ram_metric_2_max]      = $eval.[eval_target] @session_end $env max.ram_metric.[2]
M1.[51_eval1/58_ram_metric_3_max]      = $eval.[eval_target] @session_end $env max.ram_metric.[3]

M1.[52_eval2/50_ram_metric_1_mean]     = $eval.[eval] @session_end $env mean.ram_metric.[1]
M1.[52_eval2/51_ram_metric_1_mean_ema] = $eval.[eval] @session_end $env mean.ram_metric.[1] $ema ema_alpha:0.1
M1.[52_eval2/52_ram_metric_2_mean]     = $eval.[eval] @session_end $env mean.ram_metric.[2]
M1.[52_eval2/53_ram_metric_2_mean_ema] = $eval.[eval] @session_end $env mean.ram_metric.[2] $ema ema_alpha:0.1
M1.[52_eval2/54_ram_metric_3_mean]     = $eval.[eval] @session_end $env mean.ram_metric.[3]
M1.[52_eval2/55_ram_metric_3_mean_ema] = $eval.[eval] @session_end $env mean.ram_metric.[3] $ema ema_alpha:0.1
M1.[52_eval2/56_ram_metric_1_max]      = $eval.[eval] @session_end $env max.ram_metric.[1]
M1.[52_eval2/57_ram_metric_2_max]      = $eval.[eval] @session_end $env max.ram_metric.[2]
M1.[52_eval2/58_ram_metric_3_max]      = $eval.[eval] @session_end $env max.ram_metric.[3]

M1.[53_evalg/50_ram_metric_1_mean] = $eval.[greedy_dist] @session_end $env mean.ram_metric.[1]
M1.[53_evalg/52_ram_metric_2_mean] = $eval.[greedy_dist] @session_end $env mean.ram_metric.[2]
M1.[53_evalg/54_ram_metric_3_mean] = $eval.[greedy_dist] @session_end $env mean.ram_metric.[3]
M1.[53_evalg/56_ram_metric_1_max]  = $eval.[greedy_dist] @session_end $env max.ram_metric.[1]
M1.[53_evalg/57_ram_metric_2_max]  = $eval.[greedy_dist] @session_end $env max.ram_metric.[2]
M1.[53_evalg/58_ram_metric_3_max]  = $eval.[greedy_dist] @session_end $env max.ram_metric.[3]
```

- kung_fu_master では `ram_metric.[1]` が 1 以上なら面 1 を抜けている、`ram_metric.[2]` が 1 以上なら少なくとも 1 回ボスを倒している、`ram_metric.[3]` が 1 以上ならボスに当てている。
- phoenix の `ram_metric.[1]` は母艦ウェーブの突破を数えない。抜けたウェーブの総数は `ram_metric.[1]` + `ram_metric.[2]` になる。
- breakout の `ram_metric.[1]` は 0 か 1 で、`game_score.ge.[432]`（1 枚目の壁を消した）と一致する。`@breakout` の ge432 系は `mean.ram_metric.[1]` で置き換えられる。ge600 もスコアの百の位 0xCC の `reach_count:6` で同じ値になる（`score_ge600`）。残すかは決めていないので番号を付けていない。付けるなら慣例の 1〜3 に当たらないので 4 以降になる。
- 番号の意味は Run の `config/config_data.txt` の `AtariEnv.ram_metric.*` で引く。`metrics.trace.defs` はラベルを持たない（NG6）。
- 番号付けの無いゲーム（例: name_this_game）や、番号の一部だけを持つゲーム（例: qbert は 1 だけ）では、無い列が trace で `null`、scalar は行が出ない。metrics 定義はそのままで batch 起動できる。
- scalar の tag は番号から作る（`ram_metric.[1]` → `ram_metric_1`）。`@breakout` の `game_score_ge432` と同じく、tag は何を測ったか（キー）を表し、意味は §8 の慣例とコメント行が持つ。慣例は大まかな意味なので、`stage_clear` のような名前を tag に入れると、phoenix の [1]（母艦ウェーブを数えない）や breakout の [1]（0 か 1）のようにずれるゲームで、実際より正確に見えてしまう。
- scalar は train・eval1・eval2・evalg の 4 群に置き、番号ごとに mean と EMA を組にする（train は `ema_alpha:0.001`、eval1・eval2 は `0.1`、evalg は既存の行と同じく mean だけ）。加えて 4 群とも番号ごとに `max.` を置き、EMA は付けない（既存の `game_score_max` と同じ）。EMA を付けるのは、各 step の `mean.` が事実上ゲーム 1 回ぶんの値の列になるためである（220 §4.7）。ボス撃破のように稀な出来事は大半が 0 になり、EMA にすると発生率として読める。eval にも置くのは、train には ε の探索が混ざり、ボス戦の振る舞いを確かめる場（EvalPanel）も eval だからである。
- `max.` を置くのは、1 回だけ面を抜けた・1 回だけボスを倒したという最高到達を見るためである。eval の `mean.` は 1 セッション（例えば eval は 10 本、greedy_dist は 100 本）の平均なので、1 本だけの成功が 0.1 や 0.01 に薄まる。`max.` ならそのまま 1 と出る。train の `mean.` は事実上ゲーム 1 回ぶんの値なので差は小さいが、`game_score_max` と同じく 4 群にそろえる。MetricsViewer は間引き表示でも区間の最小・最大を残すので（210 §2.6）、1 点だけの突出もズームアウトしたまま見える。
- tag 番号は 50 番台を使い、mean と EMA の並べ方は既存の 20/21・22/23 と同じにする。max は 56〜58 で、既存の 16・24 と同じく組の後ろに置く。40 番台は `@breakout` の `game_score.ge` 系が使っている。

## 実装指針

- `core/envs/atari1/src/AtariEnv.hpp` / `.cpp` に同居させる。新規ファイルは作らない。
  - `AtariEnvConfig` に RAM 定義表（ゲーム → {ラベル → 定義} と番号付け）を持たせ、コンストラクタで `MakeSubConfigData(MakeDefaultConfigKey("ram_metric"))` から読んで §7 を検証する。上書き prefix 側は `<override>.ram_metric.` で始まるキーの存在だけを見て fail-fast にする。
  - 畳み方は純粋な状態構造体（`Begin(v0)` / `Observe(v)` / `Value()`）にし、ROM 無しで単体テストできる形にする。
  - `ale_->act()` の 5 箇所（`ApplyFireReset` の 2 回、`ApplyResetActions` の NOOP、soft reset の NOOP、`Step` のループ）を 1 つの補助関数へ集約し、フレームごとに観測する。`reset_game()` の直後（`ApplyFireReset` 内 2 箇所、`ApplyResetActions` 内 1 箇所、`Reset` の hard / soft 経路）で仕切り直す。
  - `RecordGameCompletion` で番号ごとの確定値をスナップショットし、ログ行に番号順のラベル付きで足す。`GetScalar` に `ram_metric.[n]` の分岐を足す（パースは `ParseGameScoreThreshold` と同じ fail-fast 流儀）。`MakeAuxData` に `ram_metric.<label>` を足す。
  - `ANET_PROFILE_SCOPE` は既存の `Step` の範囲で足りる。フレームごとの比較は細粒度なので別スコープにしない。
  - RAM 定義は `ReadConfig` を通らない読み方になるので、受理したキーと値を `my_config_data_` / `my_config_json_` へ明示的に記録し、Module Config（§1）に載せる。
  - 定義 1 行のパースと `metrics` 行のパースは、`ParseGameScoreThreshold` と同じく純粋計算の static ヘルパに分け、コンストラクタに DSL を抱え込ませない。
  - 禁止キー（上書き prefix、ブロック内の不正キー）の検出で全設定キーを走査するときは、`starts_with` で候補を絞ってから regex を当てる。env は lane ごとに構築されるので、走査は lane 数だけ繰り返される。
- `core/envs/atari1/src/AtariView.cpp`: AuxData から prefix `ram_metric.` を拾い、キー順に整形して既存 1 行の末尾へ付ける。
- `docs/design/220_atari_env.jp.md`: §4.2 に `ram_metric` ブロック、§4.7 に `ram_metric.[n]` の行と AuxData / ログ行、新しい §4.9「RAM メトリクス」（契約、§8 の番号の慣例、4 ゲームの RAM 表）、§5 にオーバーレイ。
- `apps/runner/config/Atari.txt`: 設定例の 4 ゲームのブロックと trace / scalar の追加行。

## テストの方針

- **ROM 不要**
  - 畳み方の意味。v0 を含む `max_seen` / `min_seen`、39 → 0 → 39 の列で `dec_count` が戻りを数えないこと、`reach_count:0` がレベルではなく遷移を数えること。
  - §7 の fail-fast。番地の範囲と書式、未知語、N の範囲、番号の重複、未知ラベル、空の `metrics` 行、未知サブキー、ラベルとゲーム名の字種、上書き prefix。`AtariEnvConfig` を `ConfigData` から作るだけで再現できる。
  - Module Config。`AtariEnvConfig::GetConfigData()` に受理した `ram_metric.[game].[label]` と `ram_metric.[game].metrics` が入り、番号の無い定義も含む。
- **ROM あり**（無ければ SKIP。先行例は `AtariEnv_test.cpp` の Pong spec テスト）
  - pong: `0x8D inc_count` / `0x8D max_seen` を番号付けし、NOOP で truncation まで回す（sticky 無し・`noop_max = 0`）。完了 step で `inc_count` が 1 以上、未完了 step は NaN、番号の無い n は `nullopt`、kung_fu_master のブロックを同時に書いても pong では NaN。
  - kung_fu_master: 短い truncation で `floor_clear` = 0、`floor_max` = 1、`boss_hp_min` = 39、`boss_hit` = `boss_kill` = 0。`episodic_life = true` で `0x9D dec_count` が soft reset をまたいで数え続け、hard reset で戻る。
  - AuxData に `ram_metric.<label>` があり進行中の値を持つ。完了ログの行に `ram_metric:` が出る（先行例はゲーム完了ログのテスト）。
- **既存の trace / scalar が変わらないこと**: `ram_metric` キーを使わない構成で行が変わらない。ADR 0037 の受入方式（編集前 baseline との同 seed 比較）に従う。
- **既定の scalar 定義**: `Atari.txt` のプリセットテストで、4 群 33 本（mean 12・EMA 9・max 12）の tag・`source_key`・EMA の有無と係数を確かめ、残りの項目（event・eval 名など）が同じ群の `game_score` の定義と一致することを見る。
- **設定の検査**: `Atari.txt` を変えたら `check_default_leaves.py` と全キー・文字列値の比較で、RAM 追加分以外に意図しない変化が無いことを確認する（手順は `core/anet-core/testdata/prd072/README.md`）。
- 実行: `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target AtariEnv-test'` の後、`core\envs\atari1\bin\Debug` 配下の `AtariEnv-test.exe "[atari]"`。`ATARI_ROM_DIR` を設定して ROM テストも走らせる。

## 決定の記録

2026-09-24 のグリル（`/grill-with-docs`）で決めた。理由と却下案は [ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) に置く。

| # | 決定 |
|---|---|
| D1 | 概念: RAM バイト 1 本の毎フレーム列をゲーム 1 回に畳む。到達値（最大 / 最小）と遷移の回数を同じ枠で扱う |
| D2 | 定義は設定の名前付き宣言。キーはゲーム ID + ラベルの 2 段（番地の意味はゲームごとに違う） |
| D3〜D4 | metrics 側は具体的なイベント名ではなく番号付きの汎用キーを参照し、ゲームを変えても metrics 定義を変えない。宣言は「名前付き RAM 定義 + 番号付け行」の 2 段 |
| D5 | 番号付け行は番号を明示したトークン列 `1:boss_hit 2:boss_kill` |
| D6 | 畳み方は `max_seen` `min_seen` `inc_count` `dec_count` `reach_count:N` の 5 語。`min` / `max` は集約 prefix と語を分ける |
| D7 | 観測は毎フレーム・全箇所。`reset_game()` ごとに仕切り直し |
| D8 | 値は確定値（`game_score` と同型）。加えて進行中の値を AtariView のオーバーレイへ |
| D9 | 名前: 設定 `AtariEnv.ram_metric.[game].[label]`、番号付け `AtariEnv.ram_metric.[game].metrics`、参照 `ram_metric.[n]` |
| D10 | §7 の検証契約 |
| D11 | AuxData のキーは `ram_metric.<label>`。オーバーレイはキー順、無ければ出さない |
| D12 | 文書は PRD + ADR 0046 + CONTEXT.md の 3 語。220 と Atari.txt は実装と同じ変更で更新 |
| D13 | 既定: trace 4 行へ `ram_metric.[1..3]`、scalar は D19、ゲーム完了ログにもラベル付きで出す |
| D14 | 担当: PRD / ADR / CONTEXT.md は Claude、実装・テスト・220・Atari.txt は Codex |
| D15 | 番号に全ゲーム共通の大まかな意味を持たせる（§8。1 = 面クリア回数、2 = ボス撃破回数、3 = ボス命中回数）。当てはまらないゲームは番号を付けない |
| D16 | 既定設定に kung_fu_master・qbert・phoenix・breakout の定義を置く（2026-09-24 のエミュレータ調査） |
| D17 | 受理した RAM 定義と `metrics` 行は Module Config（`GetConfigData()`）に載せる。`config/env.*.txt` と eval env の設定ダンプから番号の意味を引けるようにする（2026-09-24 の実装レビュー） |
| D18 | 空の `metrics` 行は fail-fast、ゲーム名は小文字 snake_case、ラベルは `[A-Za-z0-9_]+` に限る。ROM 名になりえない綴りと空の紐づけは書き間違いとして扱う（同レビューで実装の挙動を契約に取り込んだ） |
| D19 | scalar は train・eval1・eval2・evalg の 4 群へ、番号ごとに mean と EMA の組で置く（evalg は mean だけ）。最高到達を見るため、4 群とも `max.` も置く（EMA なし）。tag は番号から作り（`50_ram_metric_1_mean`）、慣例の名前は入れない。当初の D13 は train の mean 3 本で、tag に慣例の名前を入れていた（2026-09-24 の再レビュー） |

却下した案の要点:

- **番地をキーに直書き（原案 A、`ram_min.[0xCC]`）**: キーから意味が読めず、metrics 定義がゲーム別になる。
- **コードのゲーム別表（HNS 流）**: 番地の当たり付けのたびに再ビルドが要る。HNS 表は固定の式が使う物理定数だが、どのバイトをどう畳むかは解析上の選択で、`game_score.ge.[N]` 側の先例に近い。
- **ラベルをキーにする（`ram.[boss_hit]`）**: イベントの数だけ metrics が増え、ゲームを変えるたびに定義を書き換える。
- **購読キーからの自動追跡**: env は metrics 解析より先に構築され、配線が無い。
- **Step の窓後 1 回だけ観測**: 回数系が skip 窓の途中と reset 中の遷移を取りこぼす。毎フレームの比較コストは無視できる。

## 既知の限界と後続

- **0x9F の確認範囲。** 面 1 → 2 しか見ていない。面 2 → 3 と、5 面を抜けて 1 面へ戻るときの値は未確認である。実 Run の `floor_max` の分布（2 以上の出方、6 以上が出ないこと）で確かめる。
- **0xCC の戻り。** 死亡時と次の面でボス体力は 39 へ戻る。増加なので `dec_count` には入らないが、`boss_hit` は「面をまたいだ累計」である。面ごとの命中数が要るなら NG5 と同じ後続になる。
- **phoenix の周回。** 0xCA と 0xFC の意味は 2 周ぶん（母艦 2 回）しか見ていない。
- **breakout の 2 枚目。** 2 枚目を消しても作り直しが起きないので、`wall_clear` は 0 か 1 にしかならない。
- **後続候補。** 初回発生の step（NG5）、常時値キー（NG4）、`metrics.trace.defs` へのラベル出力（NG6）、kung_fu_master 以外の RAM 表（NG2）、他 env が同型を採るときの共通化（ADR 0046 の Consequences）。
