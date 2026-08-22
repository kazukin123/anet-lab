# best checkpoint の自動保存（eval 自己ベスト更新時の重み保持）

- 起票日: 2026-08-18
- 状態: draft（バックログ。正式番号は着手時に採番）
- 対象: `core/anet-core`（observers.cpp の eval 経路、default_dqn_agent.cpp の Save）、`apps/runner`（保存の呼び出し元）
- 依存: `911_eval_batch_episodes_10prd.md`、`912_background_eval_snapshot_ordering_10prd.md`、`910_network_lock_audit_10prd.md`（§依存）
- 発見経緯: Atari Breakout の長期 Run でピークアウト劣化が 3 例揃ったこと（`docs/experiments/default-dqn/atari/2026-08-17_baseline.md` 探索ブロック 04 / 07 / 08）

## Context / Problem Statement

長期 Run では **ピークを打ったあと 10〜20% 劣化する**ことが 3 例で確認されている。

| Run | ピーク | 最終 | 変化 |
|---|---|---|---|
| Nature CNN 50M（ブロック 04） | 137.81 @43,991,872 | 121.27 | −12.0% |
| IMPALA fire_off 50M（ブロック 07） | 139.97 @49,395,200 | 137.67 | −1.6%（ピーク直後に終了） |
| IMPALA fire_on 102M（ブロック 08） | 188.46 @66,839,296 | 150.91 | **−19.9%** |

一方、比較先である Mnih et al. 2015 の Methods は評価値について "the highest average episode score is reported" と明記しており、**Breakout 401.2 も best checkpoint の値**である。したがって文献と同じ土俵で数字を出すには、best 時点の重みが残っている必要がある。

**現状は残らない。** 重みが書かれるのは Run 終了時（GUI の close ハンドラ）の 1 回だけで、定期 checkpoint の機構が存在しない。上記 3 Run のピーク時点の重みは既に失われている。

## 現状の事実（コード確認済み）

2026-08-18 時点、branch `main`。

| 事実 | 根拠 |
|---|---|
| 保存は GUI の close ハンドラ 1 箇所のみ。`save_interval` 相当の設定・コードは存在しない | `apps/runner/src/RunnerFrame.cpp:663-676` → `apps/runner/src/RunnerApp.cpp:511-545` |
| 保存内容は online net + target net + optimizer の 3 点。**RB / RNG / step counts / 正規化器は含まない** | `core/anet-core/src/default_dqn_agent.cpp:255-290`（`:270-272` に未対応の `@todo` が明記されている） |
| サイズは online params の約 4 倍（model = online+target、adam = exp_avg+exp_avg_sq）。IMPALA 50.1MB（model 25.1 + adam 25.0）、Nature CNN 43.7MB、QR51 37.6MB | Run ログの内訳出力（`default_dqn_agent.cpp:283-287`） |
| 所要時間 0.08〜0.58 秒。`WriteTorchObject` が `std::ostringstream` を経由するため、保存中は checkpoint サイズ相当のホストメモリを一時確保する | Run ログのタイムスタンプ差、`core/anet-core/include/anet/serialize.hpp:97-122` |
| **`Agent::Save` は network mutex を取らない**。現状は「学習停止後にしか呼ばない」ことで成立している | `default_dqn_agent.cpp:255-290`（lock 文なし）。`910_network_lock_audit_10prd.md` が未決事項として起票済み |
| Load は `auto_load_file` 経由で存在するが、復元されるのは重み + optimizer のみ。step counts / RB / RNG は戻らないため「学習再開」ではなく warm start | `default_dqn_agent.cpp:214-217, 292-320` |
| Observer から Agent へ到達できる（`Agent::Save` は const メソッド） | `core/anet-core/include/anet/rl.hpp:902`（`EpisodeEndEvent::agent`）、`rl.hpp:1075`（`Runner::GetAgent`） |
| Run dir は core からも取得できる（`RunnerApp` 依存なしに完結可能） | `core/anet-core/include/anet/metrics_logger.hpp:186` |
| 「N step ごとに 1 回」の発火判定は `IntervalGate`（bucket-crossing）が正本 | `core/anet-core/include/anet/util.hpp:236-277` |

## 標準実装の事例（一次ソース確認済み）

| 実装 | best 保存 | 判定 | 保存先 | ノイズ対策 |
|---|---|---|---|---|
| **原典 DQN（Lua）** | **あり** | eval 平均報酬 vs 履歴 `max()` | **メモリに clone** → 定期保存ファイルへ `best_model` キーで同梱 | 平均化のみ |
| SB3 `EvalCallback` | あり | `mean_reward`（既定 5 エピソード）を厳密 `>` | `best_model.zip` へ上書き・同期 | **なし** |
| nanoGPT | 条件式はあるが**既定で無効** | val loss（200 バッチ平均） | `ckpt.pt` へ上書き | — |
| Dopamine | なし | — | `ckpt.<iter>` + sentinel、直近 4 世代 | — |
| dqn_zoo / rlpyt | なし（dqn_zoo は checkpoint 自体が no-op） | — | — | — |

**原典 DQN が最も参考になる。** 判定と保存を分離し、best 時は network を clone してメモリに置くだけで、ディスク書き出しは定期保存に相乗りする:

```lua
if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
    agent.best_network = agent.network:clone()
end
```

**nanoGPT は注意が必要**。条件式は `if losses['val'] < best_val_loss or always_save_checkpoint:` だが `always_save_checkpoint` の既定が `True` のため常に真になり、直下の `best_val_loss = losses['val']` が悪化時も実行される。つまり**既定では `ckpt.pt` は best ではなく「最新 eval 時点」の上書き**であり、変数名に反して `best_val_loss` は走査中の最小値になっていない。真の best を得るには `always_save_checkpoint=False` の明示が要る。

Dopamine / dqn_zoo / rlpyt に best 機構が無いことから、**Atari 系の報告値の多くは「学習曲線から best iteration を選ぶ後処理」で成立している**と推測される（論文本文での確認は未了。§未調査）。ただし後処理方式は重みが残っていることが前提であり、我々の現状（Run 終了時 1 回のみ）では成立しない。

## 問題

### A. 判定に使えるスコアの質

現行 eval は `eval_batch_size = 1` の 1 エピソード評価で、Breakout の直近実測が `[44, 57, 83, 56, 86, 97, 38]` と 2.5 倍の幅を持つ（`911_eval_batch_episodes_10prd.md`）。この生値で best 判定すると、性能ではなくノイズで更新が決まる。

SB3 でさえ 5 エピソード平均であり、しかも移動平均・最小改善幅・連続改善要求のいずれも持たない。「ノイズを含む推定量の走査中最大値」は真の最大値を系統的に過大評価する（最大値選択バイアス）ため、**判定値の設計が本 PRD の中核**になる。

### B. 保存コスト

IMPALA で 1 回 50.1MB。序盤は best が頻繁に更新されるため、無条件にディスクへ書くと I/O が支配的になりうる。加えて `WriteTorchObject` が `ostringstream` を経由するため、保存のたびに同サイズのホストメモリを確保する。

### C. 排他

`Agent::Save` は network mutex を取らない。background eval（`use_background = true`、Atari の現行設定）の worker から呼ぶと、**learner の update と並行して `torch::save` が走る**。`910_network_lock_audit_10prd.md` が同じ論点を未決として挙げている。

### D. 採点した weight と保存する weight の同一性

background eval では eval が採点した network version が未確定（`912_background_eval_snapshot_ordering_10prd.md`）。best checkpoint は「そのスコアを出した重み」を保存する機構なので、**採点対象と保存対象がずれると機能自体が意味を失う**。

### E. eval 完走直後にスコアが読めない場合がある

`EpisodeEvalObserver::RunEvaluationEpisode` は `LastStepHadEpisodeEnd()` で停止する（`core/anet-core/src/observers.cpp:524-530`）が、この判定は `done | truncated` ベースであり、`episodic_life=true` では life-loss の疑似 done でも真になる。その step では `completion_available_` が false のため `game_score` は NaN（確定は実 game over / truncation のみ。`core/envs/atari1/src/AtariEnv.cpp:531, 575-579, 593-595`）。Atari の現行設定は `E.episodic_life = true` なので、**eval 完走直後にスコアが取れないケースが常態的に起きる**。

## 目標契約（案）

- eval スコアが自己ベストを更新したとき、その時点の重みを Run フォルダに残す
- 判定に使うスコアは単発 eval ではなく、ノイズに耐える集約値
- 保存が学習ループをブロックせず、learner と競合しない
- 保存した重みと、それを採点した eval の weight version が一致する
- best が一度も更新されない場合でも、現行の `agent_close.anet` の挙動は変わらない（非退行）

## 解決候補（最終選択は後続セッション）

### 保存方式

- **候補 A: 原典 DQN 方式**（best 時は online net をメモリへ clone し、ディスク書き出しは Run 終了時または低頻度）。ディスク I/O が最小で、判定と保存を分離できる。GPU/ホストメモリを params 分（IMPALA で約 12.5MB）追加で占有する。排他は clone の瞬間だけで済むため問題 C が小さくなる
- **候補 B: SB3 方式**（best 更新時に即ディスクへ上書き）。実装が最小。I/O とメモリ一時確保が更新のたびに走る
- **候補 C: 定期保存 + best 併用**（Dopamine 的な世代保存に best を足す）。再開機能の整備と抱き合わせになり範囲が広い

### 判定値

- eval1 の生値（最も単純だが問題 A に直撃）
- N 本平均（`911_eval_batch_episodes` の実装に依存）
- window 平均 / EMA（`51_eval1/11_game_score_mean_ema` が既に存在。ema_alpha 0.1）
- 最小改善幅（min_delta）や連続改善要求を足すか

### 保存内容

- online net のみ（推論・評価用途に限れば十分。サイズは現行の約 1/4）
- 現行と同じ 3 点セット（学習再開も視野に入れる場合）

## 依存

本 PRD は以下 3 件の未決事項に依存する。単独で着手すると設計が宙に浮く。

| PRD | 依存する理由 |
|---|---|
| `911_eval_batch_episodes_10prd.md` | 判定に使うスコアの質（問題 A）。N 本平均が取れないと best 判定が成立しにくい |
| `912_background_eval_snapshot_ordering_10prd.md` | 採点 weight と保存 weight の同一性（問題 D） |
| `910_network_lock_audit_10prd.md` | 学習中に `Save` を呼ぶ場合の排他設計（問題 C）。同 PRD が `Save()` / `torch::save` 経路の並行可能性を未決として起票済み |

## スコープ外

- 学習の完全再開機構（RB / RNG / step counts / 正規化器の直列化）。現行 Load は warm start であり、その拡張は別 PRD
- 定期 checkpoint の世代管理（Dopamine 的な `keep_every` / GC）
- best 選択を後処理で行う運用。学習曲線から best step を特定すること自体は現状でも可能だが、その時点の重みが残っていないため実行できない

## 未調査事項

- Dopamine / dqn_zoo の論文本文が「学習曲線から best iteration を選んで報告」と明記しているか（コード側に best 機構が無いことは確認済み。論文 PDF は未取得）
- 原典 DQN の `NeuralQLearner.lua` における `best_network` の初期化位置（更新が `train_agent.lua` 側であることは確認済み）
- best 更新頻度の実測。既存 Run の `51_eval1/10_game_score_mean` 系列から「履歴 max を更新した回数」を数えれば、候補 B のディスク I/O 量を見積もれる
