---
title: Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す
page_kind: サマリ
material: "[[docs/adr/0038-actor-config-catalog-without-runmode.md|0038-actor-config-catalog-without-runmode]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# Actor 設定は名前付きカタログ参照とし、Agent インタフェースから RunMode を外す

## 元の材料

[ADR 0038](../../../adr/0038-actor-config-catalog-without-runmode.md) blob:63d6a534e3626c0d025db107d9ec1bc27e1724d8

## 要点

- 評価タグごとに行動方策や使用 network を指定できるよう、名前付き Actor 定義を設ける。
- Runner は ActorRequest で定義を選び、Agent が用途ラベルから設定を推定する必要をなくす。
- clone の要否は対応する Actor カタログ項目で指定する。
- Env が訓練・評価のサンプラーを選ぶための RunMode は残す。
- 継承先の最終値の合成・個別設定の優先順位は後の ADR 0042 が改訂した。

## 内容

### 用途の選択

従来は評価タグを分けても方策を共有し、対象 network は RunMode に結び付いていた。名前付き定義を参照すると同じ用途でも異なる ε や target/online を選べる。

### 実装境界

学習側・評価側の固定二分を Agent に再導入せず、Runner が要求を組み立てる。Actor の設定カタログに関する決定と、後に変更した設定の継承規則は別に扱う。

## 主要な概念

- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
