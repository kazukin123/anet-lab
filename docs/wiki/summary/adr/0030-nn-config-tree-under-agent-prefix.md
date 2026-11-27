---
title: NN 設定ツリーは Agent config prefix 配下に置く
page_kind: サマリ
material: "[[docs/adr/0030-nn-config-tree-under-agent-prefix.md|0030-nn-config-tree-under-agent-prefix]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-26
---
# NN 設定ツリーは Agent config prefix 配下に置く

## 元の材料

[ADR 0030](../../../adr/0030-nn-config-tree-under-agent-prefix.md) blob:766ac376b4e544a7357379bba12edc6a4d01d0da

## 要点

- QR と IQN の切り替えでは、離れた Agent 設定と NN 配線の指定をそろえる必要があった。
- NN 設定を各 Agent の prefix 配下へ移し、アルゴリズムのプロファイルにまとめる。
- 共有するブロックカタログ net.block はグローバルな置き場に残す。
- 専用の整合性検証だけを足す案では二重指定が残る。
- 旧い Run の設定記録は当時の記録として保持する。

## 内容

### 切替の単位

従来は quantile_mode と利用する NN 配線の独立した選択が食い違い、原因の設定キーを特定しにくかった。読み口を Agent 内に移し、プロファイルの一回の切り替えで両者を指定できるようにする。

### 共有物と履歴

全 Agent で再利用するブロック定義までは移動しない。Agent 間にまたがるルートのプロファイルや設定の追加検証だけでは所有関係と二重指定を解消できない。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
