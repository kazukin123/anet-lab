---
id: "issue-20261103-5addea"
title: "PRD 079: .cpp の内部リンケージと namespace インデント規約"
status: "未着手"
raised: "2026-11-03"
phase: "PRD"
scope: "PRD 079: .cpp の内部リンケージと namespace インデント規約"
kind: "成果物"
area: ["318_rl_core", "315_nn", "312_util", "317_metrics", "321_agent_core", "319_gui", "316_config", "322_env_individual", "324_dqn", "110_runner_app", "325_muzero", "313_profiling"]
type: "索引"
ai_judge: 
ai_judged: 
docs: ["docs/memo/079_cpp_internal_linkage_convention_10prd.md"]
priority: "中"
depends: []
parent: 
value: 5
urgency: 5
risk: 3
size: 8
wsjf: 1.6
resolved: 
---

## 背景

要件は「無名 namespace が嫌い」ではない。**インデントが実際の構造を表し、かつディスプレイ幅を使い切れること**である。
現行 .cpp の `using namespace` 方針（AGENTS.md L206）は、階層そのものを消すことでこの要件を満たしている。

## 事象

対応するPRD: [079_cpp_internal_linkage_convention_10prd.md](../../memo/079_cpp_internal_linkage_convention_10prd.md)

## 影響

このPRDの作業の状態・優先度・順序をissueで管理する。

## 対応方針

対応するPRDと保存済み実装計画に従う。

## 対応内容

監理の項目はAIの見立て（棚卸しの前）。AIの判定は移行時のコミット条件による見立てで、棚卸しで確認する。
