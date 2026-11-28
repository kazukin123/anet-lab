---
title: GridMaze で到達可能な迷路を再現する
page_kind: サマリ
material: "[[docs/memo/done/0100_gridmaze_solvable_maze_10rfd.md|0100_gridmaze_solvable_maze_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/gridmaze.md|GridMaze]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-28
---
# GridMaze で到達可能な迷路を再現する

## 元の材料

[GridMaze の迷路を必ず解ける形にする — 解けない seed は派生 seed で決定的に引き直す PRD](../../../memo/done/0100_gridmaze_solvable_maze_10rfd.md) blob:772f318ee7c89310774aa4227b48b0a87e199006

## 要点

- 変更前の5×5盤面では、壁密度0.2・穴密度0.1で seed 1〜60 のうち到達できたのは35個だった。
- 生成後に上下左右で通れる経路を調べ、解けた迷路はセルも乱数の順番もそのまま使う。
- 解けなければ入力 seed から派生させた seed で引き直す。初回とは別に最大1,000回まで試す。
- 画面と aux に採用 seed を出す。同じ設定にその値を書き戻せば盤面を復元できる。
- 座標や寸法が不正なとき、または引き直しの上限でも解けないときはエラーにする。

## 内容

### 問題と比較条件

固定 seed の迷路に通路がないと Run 全体でゴールできず、自動採番では学習結果に到達不能な回が混ざる。TempoRL の5×5の比較では seed 2 がこれに該当した。既存の結果と比較できるよう、初回で解けた盤面を保存する。

### 生成と公開値

開始とゴールを決めたあとに、壁と穴を避ける4近傍の経路を Reset ごとに調べる。固定の開始点とランダムなゴールが偶然重なった場合も引き直す。採用 seed は符号なし十進数で表示し、固定 seed の引き直し通知が多数の Env から重複して出ないようにする。

### 検証と見送った案

既定 seed の25セルと短い Run の metrics checksum、別個体での再現、ランダム座標、不正設定を検査する。通路を掘ると盤面分布が変わり、生成方法を入れ替えると過去の比較を失う。難易度の調整は別の問いとして残した。

## 主要な概念

- [GridMaze](../../concepts/subjects/gridmaze.md)
- [Env 個別実装](../../concepts/framework/322_env_individual.md)
- [設定管理](../../concepts/framework/316_config.md)
