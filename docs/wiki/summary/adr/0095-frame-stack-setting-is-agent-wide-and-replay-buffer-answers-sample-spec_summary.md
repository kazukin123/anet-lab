---
title: FrameStack の設定と Sample の仕様を決める責務
page_kind: サマリ
material: "[[docs/adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md|0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# FrameStack の設定と Sample の仕様を決める責務

## 元の材料

[重ねる枚数とキーの設定は Agent 全体に置き、Sample の束の spec は ReplayBuffer が答える](../../../adr/0095-frame-stack-setting-is-agent-wide-and-replay-buffer-answers-sample-spec.md) blob:3f449170ccae149d30e34454a61b1b6b372e9819

## 要点

- 行動選択と ReplayBuffer は、ともに重ねる枚数と対象キーを使う。
- Agent が共通の `frame_stack.*` を読み、ReplayBuffer と FrameStack の両方に配る。
- 設定の並びに置く FrameStack は、重ねる処理をどの位置で行うかを表す。
- Sample の束を作る ReplayBuffer 自身がその spec を答え、呼び出し元はその仕様から Sample 用 Forwarder を作る。
- 設定を二重に持つ案は不一致を招き、並びの外で重ねる案では処理位置を設定できないため見送った。

## 主要な概念

- [強化学習の共通基盤](../../concepts/framework/318_rl_core.md)
- [テンソル処理](../../concepts/framework/314_tensor.md)
- [Agent 共通基盤](../../concepts/framework/321_agent_core.md)
