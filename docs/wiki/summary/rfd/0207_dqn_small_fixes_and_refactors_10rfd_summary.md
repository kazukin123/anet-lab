---
title: DQN 系の小さい直しとリファクタリングの束の案
page_kind: サマリ
material: "[[docs/memo/0207_dqn_small_fixes_and_refactors_10rfd.md|0207_dqn_small_fixes_and_refactors_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-28
---
# DQN 系の小さい直しとリファクタリングの束の案

## 元の材料

[RFD0207　DQN 系の小さい直しとリファクタリングをまとめて行う（学習の設定の値域、診断値の NaN、fatal の重複、Learner の State と値の振り分け、写しの集約、特性テスト）](../../../memo/0207_dqn_small_fixes_and_refactors_10rfd.md) blob:5a75305834ed1cff31e76905d6c72b4fa6a7dc2a

## 要点

- 暫定の起票で、確定した決定は無い。§5〜§7 は §4 の未決表の暫定の既定を前提にした試案である。
- 題材は `src/algo/dqn` の中の課題 12 件。方針まで課題に書かれた小さい直しを、置き場ごとに分けた 3 本（RFD0205 文書・RFD0206 基盤とアプリ・本書）の 1 本として束ねた。
- ゴールは、12 件が片付き、正しい設定の Run では学習の値・設定の記録・ログを変えないこと。わざと変えるのは、範囲外の学習の設定が構築で止まること、UQE を使わない方策の `uqe_tau` と、点の UQE の `iqn_policy_margin_mc_ratio` が NaN になって記録されなくなることだけである。
- 学習の設定の 7 つのキー（`grad_clip_tau`・`td_clip_value`・`update_interval`・`quantile_huber_kappa`・`per_alpha`・`replay_ratio`・`replay_batch_size`）の値域を、`LearnerConfig` の 1 か所で検証し、DefaultDQN と Rainbow から呼ぶ。使われない側も検証する。
- `RuntimeVars` を消し、学習の進みと PER の β を Learner が持つ。Agent の値の振り分けは、`target_policy.` のほかを Learner に任せる。
- 写しを関数 1 つずつへ寄せる: 選んだ行動の 1 本の取り出し（13 か所）、probe の抽選（4 か所）、Actor Q ヒントの入力（2 か所）、内的の割引率の選び方（6 か所）。価値の分解の診断 2 群を名前付きの番号に載せ、並びの食い違いでビルドを止める。
- 確かめは、RFD0128 の 34 構成に価値の分解 2 組と actor_approx 1 組を足した前後比較で、わざと変える 2 種類の tag を除く全 scalar の点の一致を見る。速さは Learner の更新の実時間を交互 3 巡で比べる。

## 内容

### 問題と現状

課題は 4 種類に分かれる。学習の設定の値域を検証しないので、0 以下で勾配が 0 か逆向きになり、TD の clip が黙って外れる。診断値では、UQE を使わない方策の `uqe_tau` が 0 を返し、点の UQE の比が同じ τ の複製から約 20 万の値になる。行動の番号が不正な鍵では fatal が 2 回出る。持ち主と写しでは、Learner だけが使う State を Agent が持ち、振り分けが Learner の群の一覧を写し、同じ式や手順が 2〜13 か所に写されている。テストでは、2 枚以上の stack で可視化用の forward を通すものが無い。

### 未決表に置いた既定

値域は「有限で 0 より大きい」（clip の 2 つと κ）、「1 以上」（間隔と batch）、「有限で 0 以上」（PER の α。0 の WARN は今のまま）、「有限で 0 でない」（replay ratio。負で間隔のモードを選ぶ今の形は残し、モードの設定キーへ改めるのは範囲外の課題の候補）とした。選んだ行動の取り出しは、分布の有無を値の軸の数で判定し、NaN の検査は今ある経路にだけ残す。内的の割引率は、分解かつ `separate` のときだけ `learner.intrinsic.gamma` を返す。行動選択の診断の群は、列を並べる版を足して lane・平均・session の 3 本のキーを同じ番号から作る。probe は次の観測を Perception に通さない。可視化用の forward の特性テストは、RFD0200 のあとの口で書く。実装の順は、テストと挙動を変える直しを先に、振り分けとヒントの入力を後にする。

### 条件付きの受入

受入は、ビルドとテスト、7 つのキーの端の値で止まるテスト、NaN になるテスト、fatal が 1 行であるテスト、写しが消えたことの検索（`RuntimeVars` 0 行、`expand` の式 0 行、抽選の呼び出し 1 行など）、診断の並びの入れ替えでビルドが止まる破壊の確かめ、37 構成の前後比較、Learner の更新の速さである。前後比較の基準は、本書の変更を入れる直前の作業ツリー（RFD0206 の変更を含む）から作る。これらは暫定の既定を採った場合の試案で、実装の入口のレビューで見直す。

### 実装で反映した内容

学習進行 State を DQN Learner に集約し、方策と学習設定の値域、点の UQE の診断値、不正な行動番号のエラーを修正した。選択行動の抽出、probe の抽選、内的の割引率、Actor Q ヒントの処理をそろえ、診断の並びと可視化 forward の特性を確認した。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
