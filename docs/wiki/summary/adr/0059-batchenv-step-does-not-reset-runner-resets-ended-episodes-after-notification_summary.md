---
title: BatchEnv の終端リセットを通知後に行う案と、ADR 0060 による撤回
page_kind: サマリ
material: "[[docs/adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md|0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# BatchEnv の終端リセットを通知後に行う案と、ADR 0060 による撤回

## 元の材料

[ADR 0059](../../../adr/0059-batchenv-step-does-not-reset-runner-resets-ended-episodes-after-notification.md) blob:d009e4d51e7c00059da2416714e2989def155a20

## 要点

- 終端の通知から Env を読むと、自動リセット後の状態が返っていた。
- この ADR は通知後に Runner が終端 lane をリセットする案を採った。
- その実装では Atari の非同期学習 Runner が5〜8%遅くなった。
- ADR 0060 がこの案を取り消し、自動リセットを Step に戻した。
- 終了遷移の値は Step の結果に運び、終端でもリセット前の値を読むという目的を保つ。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
