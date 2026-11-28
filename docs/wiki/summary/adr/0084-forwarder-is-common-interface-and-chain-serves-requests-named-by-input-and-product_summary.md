---
title: 観測の加工は、共通の口 Forwarder の並びとし、並びへの依頼は「何を渡して何が欲しいか」で名付ける
page_kind: サマリ
material: "[[docs/adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md|0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# 観測の加工は、共通の口 Forwarder の並びとし、並びへの依頼は「何を渡して何が欲しいか」で名付ける

## 元の材料

[ADR 0084](../../../adr/0084-forwarder-is-common-interface-and-chain-serves-requests-named-by-input-and-product.md) blob:2b5cc8905a9dc4cee94af22764ba1c2ca791a741

## 要点

- 観測加工と Network は TensorDict を受け取り返す Forwarder の共通契約を使う。
- 重ね方や正規化を部品として並べ、各変換後の spec を求める。
- 依頼は行動選択や可視化という呼び手の名前ではなく、渡すものと必要な結果で表す。
- 後の ADR 0094 では並びを Forwarder の生成側とし、履歴も依頼から作った Forwarder が扱う。
- 依頼の種別名は RL 側が定義し、共通の基盤から利用側固有の列挙を外す。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
