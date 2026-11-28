---
title: 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD
page_kind: サマリ
material: "[[docs/memo/done/0097_env_auto_reset_after_notification_10rfd.md|0097_env_auto_reset_after_notification_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# 遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD

## 元の材料

[遷移の結果を Step の結果で運ぶ — 通知の中で読む遷移の結果を、終端の lane でもリセット前の値にする PRD](../../../memo/done/0097_env_auto_reset_after_notification_10rfd.md) blob:db007ff5e2700618033d51761840a3f06cf1a66a

## 要点

- 終端の lane が Step 内で自動 Reset されると、Observer が後から Env へ問い合わせて次エピソードの値を読む場合があった。
- 通知で必要な得点・残機・完了値などを、遷移時点の Step の結果として運ぶ。
- 学習・エピソード終了の通知を受ける側は、動き続ける Env よりその遷移のデータを参照する。
- Reset を通知後まで遅らせる案は処理の重なりを失ったため、Step 内で学習と並行する流れを保つ。
- 終端・非終端の両方で遷移内容を照合し、値の正しさと性能を確認する。

## 内容

### 背景

ユーザーの説明（2026-10-19）: 自動リセットが要るようになったとき、`Step` の中でリセットまで行えば、既存の流れとも単一の env とも同じ形を保て、`Reset` と `Step` の 2 つだけという単純さも保てた。
ユーザーの方向（同日）: 欲しいのは step の結果なので、リセットの前に Observer を呼ぶのが正しい。この方向を受けて M2 を選んだ。

### 現状（2026-10-19〜20 にコードと計測で確かめたこと）

`next_state` にはリセット前、`continue_state` にはリセット後の状態を詰める。
評価セッション（`RunSession`）: 採用 episode の `EpisodeEndEvent` は、Step の処理を抜けたあとに `LastAdoptedGroups()` から出す（同 315-342）。
`Reset` の中でもゲームの完了を記録することがある。残機を失った直後の NOOP でゲームオーバーになった場合である（同 725-733）。
env の乱数は env ごとに持ち、torch の共有の乱数は使わない（`core/envs` に `torch::rand` などは無い）。リセットを Step の中で行っても、学習の乱数の系列は変わらない。
後半の実時間の約 53% は同期の評価で、exp/s の差はそのままの割合では実時間に出ない。

### 確定事項

env の `GetScalar`・`GetTensor` は、今の env の状態（リセット後もあり得る）を返す。遷移の時点との一致は保証しない。
ADR 0037 の「trace の読み手は完了 Step の直後に `GetScalar(key, env_index)` を読む」は、`EpisodeEndEvent` が持つ Step の結果から読む形に変わる。イベントの型の分け方と trace チャネルは変えない。
退けた案: 段を残し、中身は Step の中で済ませたリセットの結果を返すだけにする。097 で直した箇所を多く残せるが、段が守るものが無く、使われない境目と呼ぶ順番の規則が残る。
同じ鍵でも、`@train` で読めばその step の途中の値（速報値）、`@episode_end` で読めばエピソードの確定値になる。
ゲームが終わった step（NOOP でゲームオーバーになった場合を含む）では、ゲーム単位の鍵は、ゲームの確定値（NOOP の報酬とフレームを含む）を返す。これは改修前の完了値と同じ値である。

### 成功の測り方と受入基準

通知の中で読む遷移の結果（Step の結果）が、終端の lane でもリセット前の値になる。
PRD 094 の step trace の `$env_step` が、終端の行でも遷移の時点の値になる。
各 Run の `config/config_data.txt` で、`AtariEnv.episodic_life` と `fire_reset` の値を確かめる。
構成: Atari 1 と同じ構成、GridMaze、LunarLander（`run.@cpu`）、DropMerge（非同期・逐次）。
合格の条件: 構成ごとに、改修後 3 本の中央値の、改修前 3 本の中央値からの低下が 2% 以内。改修前の 3 本のばらつき（最大と最小の差）が 2% を超えた構成は、5 組に増やす。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
