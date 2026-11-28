---
title: テストは自分のライブラリの内部まで読み、ほかのライブラリは公開ヘッダとテスト用の口だけを通して使う
page_kind: サマリ
material: "[[docs/adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md|0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# テストは自分のライブラリの内部まで読み、ほかのライブラリは公開ヘッダとテスト用の口だけを通して使う

## 元の材料

[ADR 0080](../../../adr/0080-tests-read-own-library-internals-and-reach-other-libraries-through-public-headers-and-test-support.md) blob:9b33bfb7456ce7d8239b30010b8ef6933675075b

## 要点

- 単体テストを対象ライブラリに置き、その内部には直接アクセスできるようにする。
- 他ライブラリには公開ヘッダと明示的なテスト用の口からアクセスする。
- 相手の振る舞いが主題でなければ代役を使い、上位・兄弟ライブラリへの依存を避ける。
- テスト用のヘッダを anet/test/ に置き、別ターゲットで提供する。
- 製品コードがテスト用の口を参照するとヘッダのガードでコンパイルを止める。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
