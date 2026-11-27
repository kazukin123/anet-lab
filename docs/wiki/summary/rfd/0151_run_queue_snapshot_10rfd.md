---
title: 実行計画と実行体を固定する Run キュー案
page_kind: サマリ
material: "[[docs/memo/0151_run_queue_snapshot_10rfd.md|0151_run_queue_snapshot_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
updated: 2026-11-26
---
# 実行計画と実行体を固定する Run キュー案

## 元の材料

[RFD0151　実行計画と実行体を固定する Run キュー案](../../../memo/0151_run_queue_snapshot_10rfd.md) blob:c8ca0692d5f28e055fc8433d30feda09e5e6df4a

## 要点

- 実行計画と実行体を保存するキューは暫定案で、設定の保存単位、操作の入口、既存バッチとの分担は決まっていない。
- 夜間の長い Atari バッチでは条件を途中で入れ替えるたびにファイルを作り直し、実行中の bat は直接編集できなかった。
- Workspace 配下に順序を持ったエントリを置き、実行計画をスナップショットとして扱う案を比較する。
- 登録した時点の実行体を固定し、次のビルドで後続 Run のコードが変わるのを防ぐ。
- 保留・実行・完了などの状態と、順番変更や中断後の再開方法を整理する。

## 内容

### Context / Problem Statement

走行中プロセスは起動時に config を読み終えているので安全である。危険なのは 同じ bat のまだ起動していないブロックで、こちらは編集後の `Atari.txt` を読む。この日 `Atari.txt` を 1 行修正したが、それが後続 7 ブロックへ及ぼす影響は人間が頭の中で追うしかなかった。

### 効く事実

`config/config_data.txt` は解決後の実効設定で、`$include` も `.$` 選択もプロファイルも含まない（実測 843 行、選択キー 0 件）。値参照 `${}` も展開済み。そのまま `--config` へ渡せば冪等になるはず（未検証。受入条件 1 へ）。

### 提案

`--enqueue --pin-exe <name>` で投入時にコピーを取る。既定は pin しない — 全エントリで無条件にコピーするとディスクを食うため、必要な時だけ指定する（「適宜指定」の要求どおり）。

### 候補を採る場合の受入条件

冪等性: 既存 Run の `config_data.txt` を `--config` で再投入し、生成される `config_data.txt` が入力と一致する。これが破れると提案の前提が崩れるので最初に確認する。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [316 設定管理](../../concepts/framework/316_config.md)
