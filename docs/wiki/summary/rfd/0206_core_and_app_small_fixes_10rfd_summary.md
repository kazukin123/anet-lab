---
title: 基盤とアプリの小さい直し 12 件の案
page_kind: サマリ
material: "[[docs/memo/0206_core_and_app_small_fixes_10rfd.md|0206_core_and_app_small_fixes_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-28
---
# 基盤とアプリの小さい直し 12 件の案

## 元の材料

[RFD0206　基盤とアプリの小さい直し 12 件（設定の値域と数の読み取り、Replay の n-step と先読み、行動の device の控え、テストと検査の道具、名前の整理、bat の題）](../../../memo/0206_core_and_app_small_fixes_10rfd.md) blob:e1a85980c8bc3bb61dc68de1dae70253b880c13b

## 要点

- 暫定の起票で、確定した決定は無い。§5〜§7 は §4 の未決表の暫定の既定を前提にした試案である。
- 起点は、仕様の調整がほぼ要らない小さい課題を集めて夜のうちに実装させたいという開発者の発言で、34 件を置き場ごとに 3 本（RFD0205 文書・本 RFD 基盤とアプリ・RFD0207 DQN 系）に分けた。
- 題材は 12 件。設定の値域と数の読み取り（Replay の n_step、ImageCls の 4 キー、QValuePanel の hist_bins、メトリクスの宣言の数の属性）、例外と device の扱い（先読みの待ちの片づけ、行動の device の控えの鍵）、テストと検査の道具、名前と記録の整理、bat の題である。
- 不正な設定は Run の構築の間に完全キーと指定値と期待値を示して止め、正しい設定の Run の結果は変えない。わざと変えるのは、ImageCls の構築の INFO とメトリクスの設定の記録が 1 つずつ減ることだけである。
- 既定の値の宣言の検査が移転前の core を探す件は、コードは RFD0161 のコミット 125075dc で直っており、確かめて記録するだけにした。
- 公開の口は 2 つ変える。`StateSpec::ValidateObservation` を戻り値の無い `AssertObservation` に改め、`RunManager` のコンストラクタにアプリが予約する Env name を渡す引数（既定は空）を足して、基盤から "EvalPanel" の文字を消す。
- 振る舞いが変わらないことは、RFD0103 の検証資材の 6 構成の短い Run で、時間と資源を除く全 scalar を実装の前後で比べて確かめる。

## 内容

### 問題とゴール

2026-11-20〜23 の監査と実装の検証で見つかった 12 件は、どれも課題の記録に直し方が 1 つ書いてあるか、規約から直し方が決まる。ゴールは、12 件が片付くこと、正しい設定の Run の結果を変えないこと、不正な設定を構築時に完全キーで止めることである。graph の宣言の字句の解析の作り直し、graph の値域、RunManager のコンストラクタの分割、デストラクタが worker の例外を最初に受け取る場面、テストの実行体の wx のダイアログ、`use_n_step` が偽のときの `learner.n_step` の値域は、別の関心か判断が要るので範囲の外に置いた。

### 現状

`CreateReplayBuffer` は n_step を検査せず、0 では Replay が空のまま学習が始まらない。ImageCls の設定は勾配の clip の上限・label smoothing・weight decay・学習率の下限を確かめない。QValuePanel は hist_bins が 0 でも描画の加工へ渡す。メトリクスの宣言の数の属性は先頭の数だけを読み、後ろを黙って捨てる。`BatchActionInfo::GetAction` は index の無い cuda を控えの鍵にし、2 回目に不一致で止まる。先読みの Replay は Push の待ちの結果を受け取ってから列から外すので、例外のあとに受け取り済みの待ちが残る。imagecls の Env のテストの 1 件は画像の口を初期化しない。ImageCls の Agent は渡されない上書きの接頭辞の引数を持ち、設定を 2 回ずつ記録する。2 本の bat に題の行が無い。

### 未決表に置いた既定

値域は、n_step が 1 以上、grad_clip_max_norm が有限で 0 より大きい、label_smoothing が有限で 0〜1、weight_decay が有限で 0 以上、学習率が 0 以上、hist_bins が 1 以上で、外れたら止める。数の属性は値の全体を読み切る変換にし、空・読み残し・範囲外で止める。値域の検査は今のまま足さない。控えの鍵は index を補った device にする。待ちは列から外してから結果を受け取る。テストは先頭で画像の口を初期化する。bat は AGENTS.md の形（`@echo off` の次に `title %~nx0`）にそろえる。文書は設計 150 の `n_step` の行と、設計 100 の予約名の書き方を直す。実装は段階に分けず、比較の基準の実行体をコードに手を入れる前に保存する。

### 条件付きの仕様と受入

仕様は、値域の判定の表、数の属性の読み取りの表、公開ヘッダの宣言の草案である。受入は、課題ごとに `[rfd0206]` のタグのテストで止まる値と通る値を確かめ、検索で旧い名前と基盤の "EvalPanel" が残らないことを確かめ、imagecls の Env のテストを 3 通りの順で回し、検査の道具が apps と src を引くことを確かめ、6 構成の全 scalar の前後一致と既存のテストの非退行を見る。これらは暫定の既定を採った場合の試案で、実装の入口のレビューで見直す。

### 実装で反映した内容

Replay の `n_step` を構築時に検証する。先読み処理では Push の結果を取り出す前に待ち列から除く。ImageCls、QValuePanel、メトリクスの数値属性にはそれぞれの値域と構文の検証を入れ、RunManager はアプリから受け取った予約 Env 名をほかの Env 名と一括で照合する。行動の device は補完後の値を保存する。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
