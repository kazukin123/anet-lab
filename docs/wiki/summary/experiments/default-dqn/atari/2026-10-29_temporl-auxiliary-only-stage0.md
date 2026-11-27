---
title: qbert の行動固定を評価時だけ外す比較（PRD104 段0）
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md|default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/subjects/qbert.md|Qbert]]"
updated: 2026-11-26
---
# qbert の行動固定を評価時だけ外す比較（PRD104 段0）

## 元の材料

[qbert の行動固定を評価時だけ外す比較（PRD104 段0）](../../../../../experiments/default-dqn/atari/2026-10-29_temporl-auxiliary-only-stage0.md) blob:4cde81a16598ea87a260f8305958fa206075fd0b

## 要点

- 元の50Mで学習済みの3 checkpointは、評価時だけ行動の固定を外すとすべて平均点が上がった。
- PRD104のgの点推定は3本とも「固定が主因」の行になった。
- この判定は評価時の固定の損を示す。学習側の遅れの原因全体はまだ分けていない。
- 追加指定のsplit checkpointも同条件で評価し、平均点は26,913.6→31,358.9、gは0.61だった。
- 評価時の行動の選び方を学習中のTrain Actorに寄せると、分割のfullでだけ8面突破率が下がった。
- 8面を抜けたゲームのほとんどは第9面で終わり、第9面を抜けたのはsingleでも200ゲーム中4〜15本、fullでは0〜2本だった。

## 内容

### 目的と比較条件

最終Release exeのSHA256は`2f74a21f48cf72349f79323c06b1e2bee1af3542943d20870905c101f82156e1`。実行体・DLL111ファイルと設定snapshotを固定した。

### 探索ブロック01: 同じ重みでfullとsingleを比べる

3本とも分母は正なので「評価では無効に遅れていない」の例外処理は不要だった。遮断と重み0.3のsingleは無効の平均点に近づき、合算のsingleは無効を超えた。これは今回の1 checkpointずつ・200ゲームの点推定であり、学習seed間の優劣を確定する検定ではない。gの暫定閾値0.5/0.2/−0.2は変更していない。

### 追加splitと別枠の評価

指定された`run_20261028-215839_a5_qb_aln_t103spl_s1`は、当初は実行中で代替候補の入力だけを準備していた。元の3 checkpointがすべて読めたため、代替としては実行していない。今回、ユーザー指定の追加行として同じ1Run内の2sessionを採取した。

### 未確定と今回の区切り

段0はPRDの判定規則では「固定が主因」を支える。学習側の30M以降の遅れがどれだけデータの偏りによるか、補助学習のみでどこまで回復するかは未確認。

### 探索ブロック02: 評価時のNoisyNetとτを、fullとsingleで比べる

評価設定はブロック01の評価専用設定から`[greedy_single]` sessionを除き、`[greedy]`のActorの葉だけを上書きした。1Run1条件、各200ゲーム。ゲーム・前処理・lane数・突破の定義はブロック01と同じ。

分割のfullの判断率（0.186〜0.191）と平均スキップ長（5.24〜5.39）は、4条件でほぼ同じだった。

### 探索ブロック03: 無効のseed2のcheckpointを評価する

ブロック02の無効（seed1）は、Train Actorと同じ選び方（両方）でも60.0%で、学習中の最後の400ゲーム（86.5%）より低かった。同じ構成のseed2の終了時checkpointで、同じ評価を行った。

### 探索ブロック04: 第9面で終わったゲームと、合算のsingleのEvaluation Trace

探索ブロック01の評価Run（各200ゲーム）を、ゲームごとの記録から数え直した。新しい評価は流していない。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [NoisyNetの観測](../../../../concepts/methods/noisynet.md)
- [τ配置とIQN比較](../../../../concepts/methods/tau_sampling.md)
- [Qbert](../../../../concepts/subjects/qbert.md)
