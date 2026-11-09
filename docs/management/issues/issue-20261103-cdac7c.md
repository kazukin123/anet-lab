---
id: "issue-20261103-cdac7c"
title: "target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`）暫定 PRD"
status: "未着手"
raised: "2026-11-03"
phase: "PRD"
scope: "target network 更新の step 軸正規化（`soft_update_tau` / `hard_update_interval`）暫定 PRD"
kind: "成果物"
area: ["321_agent_core"]
type: "索引"
ai_judge: "残件有り"
ai_judged: "2026-11-09"
docs: ["docs/memo/940_target_update_step_axis_10prd.md"]
priority: "中"
depends: []
parent: 
value: 5
urgency: 3
risk: 3
size: 5
wsjf: 2.2
resolved: 
---

## 背景

`replay_ratio` は本記録で最大のレバーだが、これを動かすと **target network の遅れが道連れで動く**。
`SoftUpdate()` は learn_step ごとに呼ばれるため `soft_update_tau` は「勾配更新 1 回あたり」の EMA 係数であり、
同じ τ でも RR が変われば「経験 1 step あたりどれだけ target が online へ寄るか」が変わる。

## 事象

対応するPRD: [940_target_update_step_axis_10prd.md](../../memo/940_target_update_step_axis_10prd.md)

## 影響

このPRDの作業の状態・優先度・順序をissueで管理する。

## 対応方針

対応するPRDと保存済み実装計画に従う。

## 対応内容

監理の項目はAIの見立て（棚卸しの前）。AIの判定は移行時のコミット条件による見立てで、棚卸しで確認する。
