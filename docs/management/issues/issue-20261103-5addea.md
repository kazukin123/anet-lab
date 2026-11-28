---
id: issue-20261103-5addea
title: "RFD0079: .cpp の内部リンケージと namespace インデント規約"
status: 未着手
raised: 2026-11-03
phase: RFD
scope: "RFD0079: .cpp の内部リンケージと namespace インデント規約"
kind: コード
area:
  - 311_dev_tools
  - 318_rl_core
  - 315_nn
  - 312_util
  - 317_metrics
  - 321_agent_core
  - 319_gui
  - 316_config
  - 322_env_individual
  - 324_dqn
  - 110_runner_app
  - 325_muzero
  - 313_profiling
type: リファクタリング
domain: 開発
ai_judge: 残件有り
ai_judged: 2026-11-28
docs: []
rfd: "[[0079_cpp_internal_linkage_convention_10rfd]]"
priority: 低
depends: []
parent:
value: 2
urgency: 8
risk: 1
size: 5
wsjf: 2.2
resolved:
resolution:
memo:
tags: []
---

## 背景

要件は「無名 namespace が嫌い」ではない。**インデントが実際の構造を表し、かつディスプレイ幅を使い切れること**である。
現行 .cpp の `using namespace` 方針（AGENTS.md L206）は、階層そのものを消すことでこの要件を満たしている。

## 事象

対応するPRD: [079_cpp_internal_linkage_convention_10prd.md](../../memo/0079_cpp_internal_linkage_convention_10rfd.md)
- 2026-11-23 監査（backlog）: 規約は `docs/quality/impl.md` の項番 32 に入った。`src` の `.cpp` には無名 namespace が 87 ブロックあり、本番は 41 ブロック・34 ファイルで、RFD の移行一覧（本番 16 ブロック・14 ファイル）より増えた。`tools/check_namespace_style.py`（D8）は無い。
- 2026-11-25 監査（refactor）: `src` の `.cpp` の無名 namespace は 92 ブロックになった（2026-11-23 の時点で 87）。増えた 5 のうち本番は 4 で、RFD0161・RFD0200 の `src/core/base/src/forwarder.cpp`（3）と `src/algo/dqn/src/dqn_learner.cpp`（1）である。どれも型だけを囲む。

## 影響

このPRDの作業の状態・優先度・順序をissueで管理する。

## 対応方針

対応するPRDと保存済み実装計画に従う。

## 対応内容

## 備考

監理の項目はAIの見立て（棚卸しの前）。AIの判定は移行時のコミット条件による見立てで、棚卸しで確認する。
