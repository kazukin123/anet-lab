---
title: Agent と機能ごとの設定ファイルの配置案
page_kind: サマリ
material: "[[docs/memo/0181_config_file_layout_10rfd.md|0181_config_file_layout_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# Agent と機能ごとの設定ファイルの配置案

## 元の材料

[RFD0181　Agent と機能ごとの設定ファイルの配置案](../../../memo/0181_config_file_layout_10rfd.md) blob:f9a0d1ff2bce0da9bd9c4012815b230f5a1c81d0

## 要点

- Agent 別の設定ファイル配置は暫定案であり、どの定義を分けるかはグリル前の既定である。
- 複数 Agent の設定が一つのファイルに集まり、機能を足すたびにどのファイルに書くか判断しにくくなっていた。
- Agent 全体、具象 Agent、機能専用 include の層を区別する案である。
- 環境別の Run 選択、共有 NN 部品、メトリクスのカタログはそれぞれの関心で管理する。
- 機能ファイルを増やす単位と、既存プロファイルを選ぶ側の責務を照合する。
- 移行は解決後の全キーと値を比較して確認し、root 名や解決規則の変更は別に扱う。

## 内容

### 起草時点の現状

7 env の解決済み設定（全キーと文字列値）の前後比較。RFD0098 の実装で `config-equivalence.json` として使った（rfd0098 の README）。

### ゴールと非ゴール

`agent.txt` は Agent 横断だけ。具象は `<Agent>.txt`、機能は `<Agent>_<feature>.txt` を具象から include する。

### 提案の骨子

名前。ファイル名は設定の prefix と一致させる（`DefaultDQNAgent.txt`）。env ファイルが Env 名（`GridMaze.txt`）なのと同じ規則として表に書く。機能ファイルは `<Agent>_<feature>.txt`。
`_main.txt` は L0〜L2 を include する。L3 は L2 から include する。`_workspace_template.txt` は変えない。

### 候補を採る場合の受入基準

`skip.txt` は残らない。`_main.txt` の include は L0〜L2 だけ。

### 先送りした段階と、作る条件

フォルダ化（`DefaultDQNAgent/` に L2・L3 を入れる、env を `envs/` へ）: 条件は、具象 Agent が 6 以上、または 1 Agent の機能ファイルが 4 本を超えたとき。include は隣解決で届くので仕組みの変更は要らない。
`nn.txt`・`nn_cnx.txt` の分割: 条件は、共有 NN 部品が Agent 固有に偏ったとき。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
