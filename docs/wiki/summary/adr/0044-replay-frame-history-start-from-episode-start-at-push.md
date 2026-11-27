---
title: ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない
page_kind: サマリ
material: "[[docs/adr/0044-replay-frame-history-start-from-episode-start-at-push.md|0044-replay-frame-history-start-from-episode-start-at-push]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-26
---
# ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない

## 元の材料

[ADR 0044](../../../adr/0044-replay-frame-history-start-from-episode-start-at-push.md) blob:3b760196eca2de4c677e8a5f955c769efdb66b4d

## 要点

- frame stack 復元の開始境界は Push が受け取る episode_start で確定する。
- n-step の terminal や actual_n_steps は後で決まる値で、slot の開始を表す一次情報にはならない。
- extractor は保存した開始境界を使い、別エピソードや未確定 slot にまたがる復元を防ぐ。
- lane の直前 END から開始を推定する案では初期状態など追加の暗黙条件が生じる。
- 後に END も1 step の値として保存するようになったが、stack の境界判定は episode_start に従う。

## 内容

### 境界の由来

リング先頭で未書き込み slot の初期値が偶然境界に見えたり、n-step が未確定な slot を終端と誤認したりする。Push で渡した episode_start を保存すれば、Extractor と Actor が同じ開始信号に従える。

### 検証と費用

履歴開始は各 slot に約1 byte、入力の整合性検査は各 lane に状態一つを使う。遅延 Push のエラーは次の同期時に伝える。後に保存した1-step END は stack の新しい境界条件にはしない。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
