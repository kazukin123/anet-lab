# eval_batch_size > 1 の完走契約（N エピソード平均）

- 起票日: 2026-08-16
- 状態: draft（バックログ。正式番号は着手時に採番）
- 対象: `core/anet-core`（observers.cpp の `EpisodeEvalObserver::RunEvaluationEpisode`、trainer.cpp の `EvalRunner`）
- 関連: PRD 053（§スコープ外で本件に言及）、`912_background_eval_snapshot_ordering_10prd.md`（同じ eval 経路の別論点）
- 発見経緯: Atari Breakout の A/B（2026-08-16）で eval 値の分散が大きく、文献比較に耐えないことが判明

## Context（背景）

`eval_batch_size` は設定として存在し、`CreateBatchEnv(tag, seed, eval_batch_size, ...)`（trainer.cpp）で eval 用 BatchEnv の lane 数として渡されている。しかし評価ループの停止条件が「どれか 1 lane が終端したら止める」であるため、**2 以上を指定しても 1 本ぶんの結果しか得られず、しかもその 1 本は最短エピソードに偏る**。

現状の実害は eval 値の分散である。Breakout で 32M step まで学習した Run の `51_eval1/10_episode_score_mean` 直近 7 点は `[44, 57, 83, 56, 86, 97, 38]` で、38〜97 と 2.5 倍の幅がある。文献値（Nature DQN Breakout 401.2）は 30 エピソードの平均で報告されており、1 本評価のままでは学習曲線の判読にも文献比較にも精度が足りない。

現在は 2.5M step の window 平均（3〜4 点）で代用しているが、これは「異なる時点の方策の平均」であり、単一時点の N 本平均とは意味が違う。

## 現行コードで確定している事実

### 1. 停止条件は「最初の 1 lane」

```cpp
// observers.cpp: EpisodeEvalObserver::RunEvaluationEpisode
eval_runner_->Sync();
do {
    eval_runner_->DoStep(event_counts);
} while (!eval_runner_->LastStepHadEpisodeEnd());
```

`LastStepHadEpisodeEnd()` は `RunnerBase::AccumulateAndNotifyEpisodeEnd` の第 1 パスで **どれか 1 lane でも終端していれば true** になる（trainer.cpp、`last_step_had_episode_end_ = true;` を lane ループ内で立てる）。したがって残り lane のエピソードは中断され、完成しない。

### 2. スコアは実ゲーム境界でのみ確定し、次 Step で NaN へ戻る

env 側（AtariEnv）は `completion_available_` が真のときだけ `episode_score` / `episode_len` / `episode_frames` を返し、それ以外は NaN。`completion_available_` は `Step()` の冒頭で false に戻る。つまり**確定値が読めるのは終端した step のその瞬間だけ**である。

### 3. metrics 側は NaN 除外の平均に既に対応している

`mean.` 集約（env.cpp、`DiscreteBatchEnvBase::GetScalar`）は

```cpp
// NaN は集計から除外
if (std::isnan(val)) continue;
```

で NaN lane を飛ばし、有効値だけで平均を取る。よって**同一 step で複数 lane が終端すれば、その平均が自然に得られる**。集約側の改修は不要。

ただし @episode_end イベントは終端した lane ごとに 1 回ずつ飛ぶ（`AccumulateAndNotifyEpisodeEnd` 第 2 パス）ため、lane が別々の step で終端する通常のケースでは「1 イベント = 1 本」の記録が N 回並ぶ形になる。1 セッション 1 点として平均済みの値を出したい場合は別途集約器が要る（§解決候補 D）。

### 4. eval_batch_size の現行配線

- `train.eval.[tag].eval_batch_size`（既定 1、正数チェックあり）→ `CreateBatchEnv` の lane 数
- DropMerge.txt に `#train.eval.[eval1].eval_batch_size = 16` のコメントアウトが残っており、過去に試して機能しなかった形跡がある
- ImageCls.txt は 128 を現用しているが、こちらは env が epoch ベースで「1 エピソード = 1 バッチ」の意味論のため本件の影響を受けない

## 問題

### A. 最短エピソードバイアス

N lane のうち最初に終わった 1 本だけが採用される。Breakout / Pong では「早く終わる = 下手」なので、**評価値が系統的に下振れする**。lane 数を増やすほどバイアスが強くなる（最小値統計に近づく）。

### B. 計算の破棄

残り N-1 lane 分の env ステップと NN 推論が捨てられる。

### C. GPU 効率（本件の隠れた利点）

現在の eval は batch=1 の CNN forward を 1 step ずつ回しており、GPU がほぼ遊んでいる（`train.eval_device_type = cuda`）。lane 数を増やしても GPU 推論は 1 回のバッチ処理で済むため、**N 本取るコストは N 倍にならない**。増えるのは CPU 側の env ステップ（ALE は軽量）だけで、eval の分散低減が安価に手に入る可能性が高い。

### D. episodic_life との相互作用

`episodic_life=true` では life-loss でも done が立つため `LastStepHadEpisodeEnd()` が真になり、**eval が 1 ライフで停止する**。その時点では `episode_score` が未確定（NaN）なので記録されず、実ゲーム完走まで 5 セッションを要する。

実測（Breakout、ライフ 5）: eval1 の記録間隔が false 160,000 exp_step に対し true は 800,000 exp_step（ちょうど 5 倍）、記録時点の `lives` 平均が 3.04（= (4+3+2+1+5)/5 の理論値と一致）。値そのものは正しい（生スコアは soft-reset で切れない）が、**記録密度が 1/5 になり、1 ゲームが 5 時点の方策にまたがる**。

「1 エピソード完走」の判定を done ではなく `episode_score` の確定（`completion_available_`）に置く必要がある。

## 目標契約

- `eval_batch_size = N` は「**1 回の評価セッションで N 本のエピソードを完走させ、その平均を評価値とする**」を意味する。
- `eval_batch_size = 1` の挙動は現行と完全に一致する（非退行）。
- 完走の判定は「学習系の done」ではなく「**env が確定スコアを出したこと**」に基づく（episodic_life に依存しない）。
- 各 lane の採用は 1 本目のみとし、他 lane を待つ間に走った 2 本目以降は評価値に混ぜない（短いエピソードの重複カウントを防ぐ）。

## 解決候補（最終選択は後続セッション）

### 候補 A: 全 lane 完走まで回す（最小変更）

停止条件を「全 lane が 1 回以上終端したか」に変える。lane ごとの完走フラグを Observer か Runner が持つ。

- 利点: 変更が小さい
- 欠点: 早く終わった lane は自動 reset されて 2 本目を走り始め、その結果も @episode_end で記録されてしまう。短いエピソードが多重カウントされ、バイアスが完全には消えない

### 候補 B: 各 lane 1 本目で凍結（正確）

完走した lane を集計対象から外し、以降の記録を抑止する。lane マスクを EvalRunner が持ち、マスク済み lane の @episode_end 通知を止めるか、env に no-op を送り続ける。

- 利点: 契約どおりの「N 本の独立サンプル平均」になる
- 欠点: マスク機構が新規。通知を止める層（Runner か Observer か）の選択が要る

### 候補 C: N 本集まるまで回す（lane 数と独立）

`eval_episodes` を別キーとして持ち、lane 数とは独立に「N 本完走するまで」回す。lane 数は並列度、N はサンプル数として分離する。

- 利点: 概念が分離して分かりやすい。lane 数を GPU 効率で決め、N を統計精度で決められる
- 欠点: 設定キーが増える。lane ごとの本数が不均一になる点は候補 A と同じ

### 候補 D: EvalRunner が集約して 1 セッション 1 点で出す

EvalRunner が N 本のスコアを内部集約し、セッション終了時に `GetScalar` で平均値を返す。metrics は 1 セッション 1 点を記録する。

- 利点: グラフが「1 点 = N 本平均」になり読みやすい。window 平均との二重平均を避けられる
- 欠点: env の `episode_score` を Runner が横取りする形になり、@episode_end 由来の既存 metrics 定義（`$eval.[eval1] @episode_end $env mean.episode_score`）と経路が変わる。metrics 定義側の変更も伴う

## コスト見積もり（着手時に実測すること）

現行 Breakout の eval は 1 セッション約 1,000〜3,000 agent step（batch 1）。`eval_batch_size=16` にした場合:

- GPU 推論: batch 1 → 16 のバッチ化で、step 数あたりのコストはほぼ横ばいの見込み
- CPU env: 16 lane 分の ALE step が並列に走る（worker スレッド数に依存）
- wall-clock: 最長 lane に律速。候補 B なら「全 lane が 1 本終える」までなので、最長エピソード長 ≒ 現行の 2〜3 倍が上限

eval が wall-clock を大きく食う構成（PRD 053 の Pong 実測では interval=100 で全体の 74%）では、`interval` との合わせ込みが必要。

## 未調査事項

- BatchEnv の自動 reset が done 直後のどの時点で走るか（候補 B のマスク実装位置に影響）
- lane ごとの seed 分配。N 本のサンプルが互いに独立か（`noop_max=0` の決定的スタートでは、sticky と ε のみが多様性の源になる）
- ImageCls の `eval_batch_size = 128` 現用パスに影響が出ないこと（env の意味論が異なるため、変更を Atari 系に限定できるか要確認）
- eval 側 env の worker スレッド構成（lane を増やしたときの並列度）

## 追記（2026-08-23）: Atari プロトコル 2 プロファイル体制を受けての更新

Atari の運用を 2 プロファイルに分ける方針が決まったため、本 PRD の前提と受入条件に影響する点を記録する。

| プロファイル | 用途 | sticky | noop | episodic_life | eval 上限 | 想定 eval ε |
|---|---|---|---|---|---|---|
| Classic | デモ・長時間回し。Nature 2015 系との比較 | 0.0 | 30 | true | 18,000 frames | 0.05 |
| v5 | BTR との比較 | 0.25 | 0 | false | 108,000 frames | 0.01 |

### 未調査事項 1 件が解決: lane の独立性

> lane ごとの seed 分配。N 本のサンプルが互いに独立か（`noop_max=0` の決定的スタートでは、sticky と ε のみが多様性の源になる）

**2 プロファイルのどちらでも lane は独立になる。** Classic は `noop_max = 30` が hard reset ごとに 1〜30 回の NOOP を打つ（AtariEnv.cpp:436）ため初期状態が散り、v5 は `repeat_action_probability = 0.25` が全 step に確率性を入れる。

独立でなかったのは旧設定（`AtariEnv.v5 > E` で sticky 0 / noop 0、かつ `eval_policy.policy_type = Greedy` の ε=0）だけである。この構成では**同一重みに対する eval は毎回同一の 1 本の軌道**になり、lane を増やしても複製にしかならなかった（探索記録 探索ブロック 14）。

したがって本 PRD の実装は「lane が独立である」を前提にしてよい。ただし**旧設定の Run を再解析する際は N 本平均が成立しない**点に注意する。

### §D（episodic_life との相互作用）は Classic 固有の問題

`episodic_life = true` で eval が 1 ライフで停止する問題は **Classic プロファイルでのみ発生する**。v5 は `episodic_life = false` なので `LastStepHadEpisodeEnd()` が実ゲーム境界でしか立たず、§D の症状は出ない。

受入条件を書くときはこの非対称を明示すること。「完走の判定を done ではなく `completion_available_`（確定スコアの出現）に置く」という目標契約は **Classic を成立させるために必須**であり、v5 だけを見て検証すると素通りする。

### 文献側が要求する本数と ε（着手時の目標値）

| | エピソード数 | eval ε | 評価間隔 | 出典 |
|---|---|---|---|---|
| Nature DQN 2015 | 30 | 0.05 | — | Mnih 2015 |
| BTR | 100 | 0.01（125M frames まで、以降 0） | 1M environment frames | arXiv 2411.03820 Table D6 |

現行は 1 本・ε=0。本 PRD が解ければ Classic で 30、v5 で 100 が視野に入る。コスト見積もりの節にある `eval_batch_size=16` の試算は、この 30 / 100 を目標に置き直すこと。

なお BTR は「評価 1M environment frames ごと」であり、frame_skip 4 で 250,000 exp_step 相当。現行の `interval = 2500` とは基準が異なるため、本数を上げるときは interval とセットで決める必要がある（コスト見積もりの節の指摘のとおり）。

### 本 PRD では解けない隣接論点: eval スロットごとの policy override

eval1 に ε=0.05、eval2 に ε=0 のように**スロットごとに ε を変えることは現状の設定体系では表現できない**。

- `eval_policy_` は agent 構築時に 1 個だけ生成される（default_dqn_agent.cpp:194）
- eval 時は run_mode に関係なく無条件に `eval_policy_` が選ばれる（同 :496）
- eval スロットが持てるキーは `run_mode` と `eval_batch_size` のみで、config_prefix は `train.eval.[tag].env`（env 限定）

`train.eval.[eval1].run_mode = eval2` で eval1 を online net 側に寄せることは今でも可能だが（`IsForTarget` は `RunMode::Eval1` のときだけ true、同 :466）、ε が共通のため eval1 と eval2 が同一評価になり意味がない。

本数（本 PRD）と ε（別 PRD）は独立に解ける。着手順は本 PRD が先でよい。
