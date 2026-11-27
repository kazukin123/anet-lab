---
title: "grill スキル拡張: goal anchor と簡素化パス PRD"
page_kind: サマリ
material: "[[docs/memo/done/0058_grill_simplification_pass_10rfd.md|0058_grill_simplification_pass_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# grill スキル拡張: goal anchor と簡素化パス PRD

## 元の材料

[grill スキル拡張: goal anchor と簡素化パス PRD](../../../memo/done/0058_grill_simplification_pass_10rfd.md) blob:36f54470c420260c2d60a6a4defa61472c92a247

## 要点

- 質問を積み上げる設計グリルで、当初の目的からのずれや、全体として過剰な仕組みを見落としやすい問題を扱う。
- 設計を進める際に、最初の目的へ立ち戻るための基準を置く。
- 個別に正当化された仕組みも、全体で省ける部分がないかを検討する。
- 簡素化案を設計決定に反映する時機をグリル手順に組み込む。
- 該当する skill の写しを同期し、判断過程を後から確認できるようにする。

## 内容

### 質問だけでは見えない条件

機構を一つずつ詳しくすると、それぞれに使い道があっても、全部を合わせたときに当初の目的を超えることがある。実際に困った場面と、どこまで直せば目的を満たすかを先に置く。

### 開始時と終了時の確認

開始時は問題と目的に合わない候補を外し、終了時は決定どうしの関係を見て、使われなくなった設定や最小の構成で足りる部分を探す。同じ確認を行う二つのスキルと二つの配置先で記述をそろえる。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
