---
title: anet-core 外の拡張点と公開ヘッダの配置
page_kind: サマリ
material: "[[docs/memo/0169_outside_core_extension_layout_10rfd.md|0169_outside_core_extension_layout_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# anet-core 外の拡張点と公開ヘッダの配置

## 元の材料

[RFD0169　anet-core 外の拡張点と公開ヘッダの配置](../../../memo/0169_outside_core_extension_layout_10rfd.md) blob:665340f89ed729562cf2a3c0006bd492978b4bd9

## 要点

- この文書に記した規約は RFD0182 に統合された。当時先送りした Agent の外部登録口と NN Head の公開も、統合先が扱う。
- 別ターゲットで実装できる拡張点と、そのために必要な interface、登録口、実例を設計文書から引けるようにする。
- 公開ヘッダを置く基準と、ライブラリ内だけで使う構築部品や実装ヘッダの扱いを定める。
- 当時の調査では公開ヘッダ40本のうち11本に core 外からの参照がなく、公開範囲と利用実態が一致していなかった。
- 外部登録口や Head の構築部品をどこまで提供するかを、必要な実例に照らして判断する。

## 内容

### 目的と問題

公開と利用が一致していない。公開ヘッダ 40 本のうち、anet-core の外（`apps/runner`、`core/envs`）から include で届くのは 29 本で、11 本はどこからも届かない。NN module と Head は interface・部品・登録口が内部にある。

### 起草時点の現状

Agent は `InitRL()` が 4 本を登録し、NN module は `InitNN()` が登録する。どちらも anet-core の中の固定の列挙である。

### ゴールと非ゴール

公開ヘッダと内部ヘッダの置き場の規約を 1 つ決め、正本を 1 か所に置く。
拡張点ごとに「anet-core の外で書けるか」「要る interface・部品・登録口」「実例」を表にして、設計文書に載せる。

### 提案の骨子

用語。「公開ヘッダ」は `core/anet-core/include/anet/`、「内部ヘッダ」は `core/anet-core/src/` の `.hpp`。「anet-core の外」は同じリポジトリの anet-core 以外のターゲット（`core/envs/*`、`apps/*`、将来の別ライブラリ）。`core/envs` も `core/` の下にあるので、「core の外」とは書かず「anet-core の外」と書く。

### 受入基準

「書ける」と宣言した拡張点には、公開ヘッダだけで実装した実例（`core/envs/` の 6 本）が表に挙がっている。

### 先送りした段階と、作る条件

Agent の外部登録口。作る条件: anet-core の外に Agent のターゲットを置く決定が出たとき（StampFly の PPO が外に出る、または 2 本目の Agent 置き場が要るとき）。形: `InitRL()` の固定の列挙はそのまま残し、外のライブラリは Env と同じく `Init<Agent>()` を公開して app が呼ぶ。内部の `ActionPolicy`・DQN 系 `Actor`／`Learner` は公開しない（外の Agent は rl.hpp の `Actor`／`Learner` を自分で実装する）。

### 決めること

公開ヘッダ: `core/anet-core/include/anet/` にあり、anet-core の外のターゲットが include できるヘッダ。
内部ヘッダ: `core/anet-core/src/` にあり、anet-core とそのテストだけが include できるヘッダ。
anet-core の外: 同じリポジトリで anet-core 以外のターゲット。`core/envs/*` と `apps/*` を含む。

### 再確認ポイント

「静的ライブラリで自己登録が落ちる」は一般的なリンカの挙動と、コメントアウトされた呼び出しから推定した。削除の前に、ビルドの `/WHOLEARCHIVE` 相当の設定が無いことを CMake で確かめる。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
