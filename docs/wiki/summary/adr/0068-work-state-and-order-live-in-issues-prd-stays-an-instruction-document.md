# 作業の状態と順序は issue が持ち、PRD は指示文書としてフォルダを動かさない

## 元の材料

[ADR 0068](../../../adr/0068-work-state-and-order-live-in-issues-prd-stays-an-instruction-document.md) blob:e022f3a2cc557e464b01384eb3d93e376bff003a

## 要点

- 各設計要求に課題を対応させ、状態、優先度、順番は課題で管理する。
- 設計文書は指示文書として起票時の位置に置く。
- 文書の番号だけでは着手順を表せない。
- 利用者が直接起こした設計要求も課題の一覧で追えるようにする。
- 既存の done・frozen・dropped 配下は多数のリンクを保つため移し直さない。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
