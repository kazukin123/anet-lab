---
id: issue-20261125-ea9fef
title: メトリクスの鍵の行動の番号が不正なとき、ParseActionScalarIndex が自分の投げたエラーを捕まえて、同じ fatal を 2 回出す
status: 解決
raised: 2026-11-25
phase: 監査
scope: src/algo/dqn/src/dqn_actor.cpp の ParseActionScalarIndex
kind: コード
area:
  - 321_agent_core
type: 不具合
domain: 学習基盤
ai_judge: 残件無し
ai_judged: 2026-11-28
docs: []
rfd: "[[0207_dqn_small_fixes_and_refactors_10rfd]]"
priority: 中
depends: []
parent:
value: 2
urgency: 3
risk: 1
size: 1
wsjf: 6
resolved: 2026-11-28
resolution: 対応済み
memo:
tags: []
---

## 背景

2026-11-25 の監査（観点 defect、範囲 321_agent_core）で見つけた。分類は例外である。開発者には質問していない。同じ形の不具合（device の指定の誤りで fatal が 2 回出る。78df9c）は解決済みである。

ホットスポット補正: `dqn_actor.cpp` は、`git log --since=2026-08-27 --follow` で 90 日に 9 回変わっており、コードのファイルの上位 5%（7 回以上）に当たる。urgency を 2 から 3 へ 1 段上げた。

## 事象

- `ParseActionScalarIndex`（`src/algo/dqn/src/dqn_actor.cpp:252-261`）は、`try` の中で `std::stoll` を呼び、読み残しがあれば `ANET_SYSTEM_ERROR` を呼ぶ。`ANET_SYSTEM_ERROR` は fatal のログを出して `AnetException`（`std::runtime_error` の派生。`exception.hpp:13`）を投げるので、すぐ後ろの `catch (const std::exception&)` が捕まえて、同じ文の `ANET_SYSTEM_ERROR` をもう一度呼ぶ。
- たとえば `action_full_q_margin.[1x]` の鍵を設定に書くと起きる。`[x]` のように数字で始まらない番号は `std::stoll` の例外から入るので、1 回だけ出る。

## 影響

- 同じ fatal が 2 行出て、ログを読む人が 2 つの失敗と取り違える。止まり方は正しい。

## 対応方針

- `try` の中では読み取りだけを行い、読み残しの検査と `ANET_SYSTEM_ERROR` を `try` の外へ出す。文言は変えない。性能の区分は「関わらない」（メトリクスの鍵を解くとき）。
- 不正な番号で例外が 1 回だけ出ることのテストを足す。

## 対応内容

- 2026-11-28 行動番号の変換と読み残しの判定を分けた。`[1x]` と `[x]` で同じ診断を 1 回ずつ捕捉するテストが通過した。 [RFD0207 実装メモ](../../memo/0207_dqn_small_fixes_and_refactors_20impl.md)


- 2026-11-28 不正な番号 `[1x]`、`[x]` の各テストで例外と fatal がそれぞれ 1 回だった。重点テストの再実行も通過した。 [検証結果](../../memo/0207_dqn_small_fixes_and_refactors_20impl.md#検証結果)
$1
- 2026-11-28 状態を `対応中` から `解決` へ変更。コミットの準備で解決の候補を選んだ。AI 判定は「残件無し」。不正な行動番号で例外と fatal が各 1 回になることを確認した。
