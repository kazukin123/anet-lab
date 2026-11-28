---
title: 設定の選択は参照先の最終値を読み、Run プロファイルの設定キーは CLI と同じ第 2 相で選択の後に勝つ
page_kind: サマリ
material: "[[docs/adr/0040-config-selection-final-value-and-run-profile-tier.md|0040-config-selection-final-value-and-run-profile-tier]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
updated: 2026-11-28
---
# 設定の選択は参照先の最終値を読み、Run プロファイルの設定キーは CLI と同じ第 2 相で選択の後に勝つ

## 元の材料

[ADR 0040](../../../adr/0040-config-selection-final-value-and-run-profile-tier.md) blob:fd436b095227c07ad415780a4832f96beeb97939

## 要点

- 選択元が後から上書きされると、宣言時点のコピーでは変更が派生先に伝わらない。
- 設定選択は参照先の最終値を伝える。
- Run プロファイルの個別キーは選択チェーンより後の相で効き、CLI の指定も尊重する。
- 選択先だけで値を上書きして終える方式では、その値を参照する別の設定に届かない。
- 継承の差分合成と個別指定の優先、定義元の相対参照は後の ADR 0042 が改めた。

## 内容

### 伝播と優先

Run プロファイルや CLI によって後から変わる値も、選択を通じて参照先へ届くようにする。カタログだけにこの規則を限定すると、同じ右辺が置き場によって違う意味になる。

### 後の改訂

旧仕様にあった選択チェーン差し替えや直書きの弱い順位は ADR 0042 が見直した。最終値を伝播する判断と、Run・CLI の同一キーの優先は引き継いだ。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
