# PRD 075: エピソード長の汎用メトリクス化と評価セッションの実行ログ

> 確定。D1〜D9、complexity audit、受入基準を実装契約とする。
> 起点: 2026-09-13。Atari の学習が進むとスコアがカンストし、eval のエピソードが長大化して train が目で見て分かる
> レベルで停止するようになった。eval schedule の interval 調整で軽減したが、評価セッションがいつ走り、
> どれだけかかったのかが実行ログに一切残らないため、停止区間との突き合わせができない。
> 関連: PRD 060（評価セッション、`eval_batch_size` と `eval_episodes` の分離）、ADR 0034、
> ADR 0043（本 PRD が決める `SHARED` 定義）、
> `core/anet-core/src/env.cpp`、`core/anet-core/src/trainer.cpp`、`apps/runner/config/metrics_scalar.txt`

## Context（背景・目的）

### 動機

Atari は `episodic_life = false` の eval で `max_episode_frames` を指定していない（`Atari.txt:1206-1207` はコメントアウト）。
学習が進んでスコアがカンストすると 1 エピソードが終わらなくなり、評価セッションの実行時間が膨らむ。
`use_background = true` でも `EpisodeEvalObserver::OnLearn` は次の発火時に `WaitBackgroundEval()` で前回の完了を待つため
（`observers.cpp:564`）、セッションが interval を超えて伸びると train が待たされる。

Atari の既定は `eval_batch_size = 1` / `eval_episodes = 1`（`common.txt:12-13` を継承し、`Atari.txt` は上書きしない）なので、
1 セッション = 1 レーン × 1 エピソードであり、セッション長 = エピソード長そのものになる。

実行ログに残るのは起動時の 1 行だけである。

| 種別 | 出力 | 位置 |
|---|---|---|
| 起動時 | `eval tag 'eval1': scheduled (interval=1000, background=true)` | `trainer.cpp:934` |
| 起動時 | `eval tag 'eval1': definition-only` | `trainer.cpp:917`、`trainer.cpp:928` |
| セッション実行中 | **無し**（`EvalRunner::RunSession` には `ANET_LOG_DEBUG` すら無い） | `trainer.cpp:342-377` |

### エピソード長が framework の一級市民になっていない

調査の過程で、より根の問題が見つかった。

| Env | キー | 実装 | 汎用版との関係 |
|---|---|---|---|
| AtariEnv | `game_len` / `game_frames` | env 固有 | **別概念**。`episodic_life = true` の train では framework の episode = 1 ライフ、`game_len` = ゲーム 1 回。`game_frames` は frameskip 前の生フレーム |
| GridMazeEnv | `episode_len` | `last_episode_len_ = step_count_` を返すだけ（`GridMazeEnv.cpp:236`、`GridMazeEnv.cpp:257`、`GridMazeEnv.hpp:90`） | **同一物** |
| LunarLander / DropMerge / ImageCls | なし | — | 見えない |

`episode return` は `RunnerBase` が全 env 共通で `$runner` から `mean./max./min./std.episode_return` として出している
（`trainer.cpp:209-212`）のに、対になるエピソード長だけが env 側の手書きへ落ち、名前すら揃っていない。
`DropMerge` と `Atari` で同じ 0/1 指標が独立に手書きされていた件（`docs/memo/999_scalar_threshold_rate_10prd.md`）と同型である。

### 継ぎ目は `EpisodeStatsAccumulator` ただ一つ

```
EpisodeReturnAccumulator (env.hpp:111 / env.cpp:178-226)
   └─ CompletedEpisodeReturn { group_index, episode_return }   (env.hpp:103)
        ├─ RunnerBase::episode_return_accumulator_ (trainer.hpp:77)
        │     → completed_episode_returns_ (trainer.hpp:78, ScalarSampleAccumulator)
        │     → $runner mean./max./min./std.episode_return   ... 全 env 共通
        └─ EvalSessionEnv::return_accumulator_ (env.hpp:180)
              → EvalSessionResult { episode_returns } (env.hpp:132)
              → RunnerBase::SetCompletedEpisodeReturns (trainer.cpp:143, 369)
```

`Add()` は毎 `Step()` で group ごとの生存・完了を既に判定している（`env.cpp:192-226`）。
ここへ step カウントを 1 本足せば、`RunnerBase` と `EvalSessionEnv` の両方へ同時に入る。
episode 境界の判定を二重に書く必要がない。

なお eval 経路では `RunnerBase` 側の accumulator は回らない。
`EvalRunner::RunSession` は `DoStepInternal(..., notify_episode_end = false)` で呼ぶため
`AccumulateAndNotifyEpisodeEnd` を通らず、セッション終了時に `SetCompletedEpisodeReturns` で値が入る（`trainer.cpp:355-369`）。
したがって `EvalSessionResult` 側にも steps を通す必要がある。

## 比較履歴

| 論点 | 案 | 長所 | 短所 | 採否 |
|---|---|---|---|---|
| キー名 | **`episode_steps`** | `episode_return` と同じ `episode_<何を>` 形。単位が名前に出る | わずかに長い | **採用** |
| キー名 | `episode_len` | GridMaze の旧 env キーと同名で config 移行が差し替えだけ | 移行期間中 `$env mean.episode_len` と `$runner mean.episode_len` が並存。単位が読めず Atari の `game_frames` と混同しうる | 却下 |
| キー名 | `episode_length` | 略語を使わない | リポジトリの既存キーは短縮形が優勢（`game_len` / `exp_step`）。単位が名前に出ない | 却下 |
| `SHARED` 定義 | **`Step()` 回数** | `PER_LANE` と同じ軸で比較できる。`num_envs` を変えても env の性質を表す | `episode_return`（総和）と非対称 | **採用**（ADR 0043） |
| `SHARED` 定義 | `Step()` 回数 × lane 数 | `episode_return` と定義が揃う | 「消費した遷移数」であって長さではない。`num_envs` を変えるだけでエピソード長が変わる | 却下 |
| 集計の置き場 | **既存 accumulator を拡張・改名** | episode 境界判定が 1 箇所。2 つの保持者へ同時に入る | リネーム差分が 5 ファイルに出る（実測 18 箇所） | **採用** |
| 集計の置き場 | 名前据え置きでフィールド追加 | 差分最小 | `CompletedEpisodeReturn` が return 以外を持ち、名前が嘘をつく | 却下 |
| 集計の置き場 | 別 accumulator を並べる | 既存へ触らない | 完了判定と reset が 2 系統に分かれ、片方だけ reset し忘れる種のバグを招く | 却下 |
| ログの値名 | **metrics scalar キーと完全同名** | ログで見た値をそのまま Viewer / `inspect_run` で引ける。概念を二重に命名しない | `mean.episode_return=` とドットを含む | **採用** |
| ログの値名 | `return_mean=` 等の短縮名 | 行がわずかに短い | 同じ値に metrics キーとログ名の 2 つの名前ができる | 却下 |
| ログの書式 | 完全 key=value（前置き無し） | grep / パースが楽 | 同じ eval tag の他の行（`eval tag 'eval1': scheduled ...`）と書式が分裂 | 却下 |
| ログレベル | **開始・終了とも info** | `app.log_level` の既定が info なので設定変更なしで見える。セッションが終わらないとき「開始行だけがある」状態がそのまま診断情報になる | 行数が eval schedule の interval に比例 | **採用** |
| ログレベル | 開始 verbose / 終了 info | 行数が半分 | 既定設定では「今 eval が走っている」が見えず、train が止まっている最中にログを見ても無言 | 却下 |

## 確定した D1〜D9

| # | 決定 | 契約 |
|---|---|---|
| D1 | キー名 | `episode_steps`。`$runner` スコープ。`mean.` / `max.` / `min.` / `std.` の集約 prefix 付きで参照する。prefix 無しの `episode_steps` は提供しない（`episode_return` と同じ） |
| D2 | `PER_LANE` の定義 | 当該 lane の episode が開始から完了までに要した `Step()` 回数。完了した `Step()` 自身を含む |
| D3 | `SHARED` の定義 | batch 全体で 1 episode なので `Step()` 呼び出し回数。**lane 数を掛けない**。`episode_return`（全 lane・全 step の総和）とは意図的に非対称にする。根拠は ADR 0043 |
| D4 | 単位 | Env の `Step()` 回数＝agent step。frameskip の前の生フレーム数ではない。framework は生フレームを知らない |
| D5 | 集計の置き場 | `EpisodeReturnAccumulator` を `EpisodeStatsAccumulator` へ改名し、`CompletedEpisodeReturn` を `CompletedEpisode` へ拡張する。クリーンブレーク方針に従い旧名の alias は残さない |
| D6 | eval 経路 | `EvalSessionResult` へ `episode_steps` を追加し、`SetCompletedEpisodeReturns` を `SetCompletedEpisodes(returns, steps)` へ置換する。旧シグネチャの overload は残さない |
| D7 | 既定メトリクス定義 | eval 側（`21_eval/`）を `@baseline` へ、train 側（`20_eps/`）を `@full` へ。eval 側に `mean.` と `max.` の両方を入れるのは、N > 1 で「1 本だけカンストしてセッション全体を引っ張る」が mean だけでは見えないため |
| D8 | env 固有実装の移行 | GridMazeEnv の `episode_len` は汎用版と同一物なので削除し、`GridMaze.txt:172-173` を `$runner mean.episode_steps` へ移行する。AtariEnv の `game_len` / `game_frames` は別概念なので存置する |
| D9 | ログ | `EvalRunner::RunSession` の冒頭と末尾に `LOG::info()` で各 1 行。値名は metrics scalar キーと同名。前置きは既存の `eval tag '<tag>': ` を踏襲する |

### 導出で決めた点（グリル中に裁定不要と判断したもの）

- **静的な設定値はセッション行に出さない。** `eval_episodes` / `eval_batch_size` はセッション間で不変なので、
  起動時の `scheduled` 行へ追記する。毎セッション出すのは重複。
- **セッション消費 step 総数は出さない。** `event_counts` はセッション中に変化しないので step 座標は開始行と同値であり、
  「なぜ長いか」は `mean.episode_steps` が説明する。
- **step 座標は開始行・終了行の両方へ載せる。** background 実行では 2 行の間に train の進捗行が挟まるため、
  終了行だけで完結して読めることを優先する。

## PH1 実装契約: エピソード長の汎用メトリクス化

PH1 単独で全 env に metrics が増えるので、ここで止めても悪化しない。

### `core/anet-core/include/anet/env.hpp` / `src/env.cpp`

1. `CompletedEpisodeReturn`（`env.hpp:103`）を `CompletedEpisode` へ改名し、`int64_t episode_steps` を追加する。
   `operator==` の `= default` はそのまま。
2. `EpisodeReturnAccumulator`（`env.hpp:111`）を `EpisodeStatsAccumulator` へ改名し、
   `current_returns_` と並べて `std::vector<int64_t> current_steps_` を持つ。コンストラクタで同じ長さへ 0 初期化する。
3. `Reset()`（`env.cpp:187`）は両方をゼロ埋めする。
4. `Add()`（`env.cpp:192`）は reward 累積と同じループの中で step を数える。
   `PER_LANE` は group ごとに +1、`SHARED` は group 0 へ 1 回だけ +1（D3）。
   完了 group は return と steps を同時に push し、両方をその場で 0 へ戻す。
5. `EvalSessionResult`（`env.hpp:132`）へ `std::vector<int64_t> episode_steps` を追加する。
6. `EvalSessionEnv` へ `captured_episode_steps_` を `captured_episode_returns_`（`env.hpp:187`）と並べて持ち、
   `BeginSession()`（`env.cpp:278`）で clear、`Step()` の採用 episode 処理（`env.cpp:345-361`）で
   `completed_returns[i].episode_steps` を push、セッション確定（`env.cpp:363-365`）で `EvalSessionResult` へ渡す。

### `core/anet-core/include/anet/trainer.hpp` / `src/trainer.cpp`

1. `episode_return_accumulator_`（`trainer.hpp:77`）を `episode_stats_accumulator_` へ改名し、
   `InitializeMetrics()`（`trainer.cpp:103`）の生成も合わせる。
2. `completed_episode_returns_`（`trainer.hpp:78`）の隣へ `anet::ScalarSampleAccumulator completed_episode_steps_` を足す。
   `InitializeMetrics()` と `AccumulateAndNotifyEpisodeEnd()`（`trainer.cpp:128`、`trainer.cpp:133`）で
   return とまったく同じ寿命・同じタイミングで `Reset()` / `Add()` する。
   `ScalarSampleAccumulator::Add` は `float` を取るので `static_cast<float>` する。
3. `RunnerBase::GetScalar`（`trainer.cpp:209-212`）へ `episode_steps` の base_key 分岐を追加する。`episode_return` と同形。
   値が未成立のとき `ScalarSampleAccumulator` が NaN を返す挙動は既存のまま（AGENTS.md「GetScalar 実装ルール」）。
4. `SetCompletedEpisodeReturns(const std::vector<float>&)`（`trainer.hpp:56`、`trainer.cpp:143`）を
   `SetCompletedEpisodes(const std::vector<float>& returns, const std::vector<int64_t>& steps)` へ置換する。
   両者の size 一致を `ANET_CHECK` する。呼び出しは `EvalRunner::RunSession`（`trainer.cpp:369`）の 1 箇所。

### config / env

1. `apps/runner/config/metrics_scalar.txt`
   - `@baseline` の `21_eval/`（既存は `01_target_reward`〜`04_policy_reward_ema`、21-24 行目）へ 4 行追加する。
     `05_target_ep_steps` / `06_policy_ep_steps` は `mean.episode_steps`、
     `07_target_ep_steps_max` / `08_policy_ep_steps_max` は `max.episode_steps`。
     いずれも `$runner @session_end $eval.[eval1]`（target）/ `$eval.[eval2]`（policy）。EMA と clip は付けない。
   - `@full` の `20_eps/`（既存は 178-179 行目）へ train 側を追加する。既存 `10_train_reward` と同じ軸・同じ流儀。
   - `@full` の `21_eval/`（既存は 180-183 行目）にも `@baseline` と同じ 4 行を入れる。
2. `apps/runner/config/GridMaze.txt:172-173` の `$env mean.episode_len @train` を
   `$runner mean.episode_steps @train` へ移行する。tag 名（`42_env/00_ep_step_mean`）は据え置いてよい。
3. `core/envs/gridmaze1/src/GridMazeEnv.hpp:90` の `last_episode_len_` と
   `GridMazeEnv.cpp:236` の代入、`GridMazeEnv.cpp:257` の `episode_len` 分岐を削除する。

## PH2 実装契約: 評価セッションの実行ログ

PH1 の値を使うので **PH1 → PH2 の順序は必須**。

### `EvalRunner::RunSession`（`trainer.cpp:342`）

`namespace LOG = anet::log;` は `trainer.cpp:14` に既にある。`RunnerBase` は prefix 付き logger を持たないので
`LOG::info()` を直接使う（`docs/design/140_observability.jp.md:276` の「Runner は従来どおり `LOG::` を使用する」に従う）。

- **開始行**: `ANET_CHECK_MSG`（`trainer.cpp:345-346`）の直後、`Sync()` の前。
  同じ位置で `std::chrono::high_resolution_clock::now()` を控える。
  `steady_clock` ではなく `high_resolution_clock` を使うのは `trainer.cpp` の既存の流儀に揃えるため
  （`trainer.cpp:409`、`431`、`482`、`620`）。
- **終了行**: `notifier_->Notify(event)`（`trainer.cpp:376`）の**後**。
  `elapsed` は RunSession 冒頭からの経過とし、**train が待たされていた全区間**を含める。
  Notify は metrics 記録を伴うのでセッションのコストの一部である。

```
eval tag 'eval1': session start learn_step=125000 exp_step=500000
eval tag 'eval1': session end learn_step=125000 exp_step=500000 elapsed=12.35s mean.episode_return=402.0 max.episode_return=402.0 mean.episode_steps=3184 max.episode_steps=3184
```

- step 座標は引数 `event_counts`（train 側 counts）。`@session_end` メトリクスと同じ座標系なので、
  ログの行と Viewer の点が突き合わせられる（CONTEXT.md「step座標系」）。
- `elapsed` は秒。小数 2 桁 + `s`。
- 値は `GetScalar("mean.episode_return")` 等ではなく、`SetCompletedEpisodes` へ渡した値から直接組み立ててよい。
  どちらでも同じ値になるが、ログの数値と metrics の数値が乖離しないことを優先する。

### 起動時 `scheduled` 行の拡張（`trainer.cpp:934`）

```
eval tag 'eval1': scheduled (interval=1000, background=true, episodes=1, batch_size=1)
```

`eval_episodes` と `eval_batch_size` はこの行へ集約する。`definition-only` 行（`trainer.cpp:917`、`928`）は変更しない。

### background 実行

`use_background = true` では `RunSession` は `PinnedThreadPool`（1 本、`observers.cpp:510-512`）の worker 上で走る。
wxLog は worker thread のメッセージをバッファして main thread で flush するが、timestamp は記録時刻なので順序は保たれる。
worker thread からの info ログが `FileLogger` へ到達することは `log_test.cpp:62` の既存テストが保証している。
追加の同期は要らない。

## 設計ドキュメントの更新（同じ変更内で行う）

| ページ | 更新内容 |
|---|---|
| `docs/design/140_observability.jp.md:149` | 「Runnerは直近Stepで完了したepisode return群を共通集約し…」の段落へ `episode_steps` を並記する。`EvalSessionEnv` が session で採用した N 本を集約する点も return と同じ |
| `docs/design/120_environments.jp.md:189` | `EpisodeReturnAccumulator` の名前を `EpisodeStatsAccumulator` へ改め、`SHARED` で reward は全 lane・全 step の合計、step は `Step()` 回数（lane 数を掛けない）と書き分ける |
| `docs/design/220_atari_env.jp.md:231` | `game_len` の行へ、汎用 `episode_steps` とは別概念である旨（`episodic_life` 下で episode 境界が一致しない）を 1 行足す |
| `docs/design/140_observability.jp.md:276` 付近 | Env の prefix 付き logger の規約は変更しない。評価セッションの開始 / 終了行を記載する場所があれば足す |

## テスト

| 対象 | 内容 |
|---|---|
| `env_test.cpp:349`（`EpisodeReturnAccumulator aggregates by episode group`） | テスト名を新クラス名へ改め、`PER_LANE` と `SHARED` 双方で `episode_steps` を検証する。**`SHARED` が lane 数を掛けないこと**を明示的に固定する（D3 の回帰防止） |
| `episode_end_test.cpp:466`（`EvalRunner RunSession emits adopted episodes then one session event`） | `runner->GetScalar("mean.episode_steps")` / `max.episode_steps` を検証する。`anet::test::LogCaptureGuard`（`test_util.hpp:189`）を `wxLOG_Message` で張り、`anet::test::HasRecordContaining`（`test_util.hpp:337`）で `session start` / `session end` 各 1 行を検証する。このテストは `GENERATE(false, true)` で foreground / background 両方を回すので、background 経路のログもそのまま担保される |
| `episode_end_test.cpp:412` 付近（train 側 runner scalar） | `mean./max.episode_steps` の検証を追加する |

`episode_end_test.cpp` は `anet/test_util.hpp` を include していないので追加が要る。
`LogCaptureGuard` の使用例は `trainer_test.cpp:783`。`LOG::info()` は `wxLOG_Message` なので
`HasRecordContaining` の level 引数も `wxLOG_Message`。

## Complexity audit

| 機構 | 切ったら戻る痛み | 判定 |
|---|---|---|
| accumulator への step カウント | 全 env でエピソード長が消え、動機が達成不能になる | Keep |
| `$runner *.episode_steps` の読み口 | metrics に出せない | Keep |
| `EvalSessionResult.episode_steps` | eval 経路は `RunnerBase` の accumulator を回さないので eval 側が空になる | Keep |
| `min.` / `std.episode_steps` | `ScalarSampleAccumulator` が全集約を既に持つので追加コストゼロ | Keep |
| `scheduled` 行への `episodes` / `batch_size` | 終了行の数値が解釈できない | Keep（1 行 1 回） |
| セッション消費 step 総数 | `mean.episode_steps` が説明する | **Cut** |
| 生カウンタ（`step_counts_` 差分）での近似平均 | PH1 で正確値が手に入り前提が消滅した（決定残渣） | **Cut** |
| エピソード長の per-episode trace | `@episode_end` の trace チャネルで後から足せる。実需が出るまで作らない | Defer |
| セッション所要時間の metric 化 | ログで足りるかを先に見る。必要になったら `$runner` へ足す | Defer |

Shrink: `EvalSessionResult` は `episode_returns` / `episode_steps` の 2 本の `std::vector` のままとし、
per-episode の構造体配列へは作り替えない。`SetCompletedEpisodes` の 2 引数で閉じる。

## 受入基準

1. `core\anet-core\bin\Debug\anet-core-test.exe "[episode_end]"` と `"[env]"` が緑。全体テストも緑。
2. `GridMazeEnv` から `episode_len` の実装が消え、`GridMaze.txt` が `$runner mean.episode_steps` で移行前と同じ値を出す
   （同 seed の Run で確認）。
3. Atari Run の `51_eval1/` と `52_eval2/` に `mean.episode_steps` / `max.episode_steps` が出る。
4. Atari Run の `RunName.log` に eval1 / eval2 それぞれ `session start` / `session end` が interval ごとに 1 対ずつ現れ、
   `elapsed` が train の停止時間と一致する。
5. `use_background = true` / `false` の両方で 4 が成り立つ。
6. 起動時の `scheduled` 行に `episodes` と `batch_size` が出る。
7. リポジトリ管理下の現用コード・config・テストに `EpisodeReturnAccumulator` / `CompletedEpisodeReturn` /
   `SetCompletedEpisodeReturns` / `episode_len` が残っていない。

## 非目標

- AtariEnv の `game_len` / `game_frames` の撤去。`episodic_life` 下で framework の episode と一致しない別概念である。
- 生フレーム数の汎用化。framework は frameskip を知らない。
- エピソード長の per-episode trace チャネル。`@episode_end` で後から足せる。
- セッション所要時間の metric 化。まずログで足りるかを見る。
- eval schedule / 採用エピソードの集計契約そのものの変更。PRD 060 と ADR 0034 の契約は動かさない。
- EvalPanel の手動 eval と train runner へのログ追加。
- eval が train を止めること自体の解消（背景実行の二重化、セッション打ち切り、`max_episode_frames` の既定変更など）。
  本 PRD は見えるようにするところまでで、対策は実測を見てから決める。
