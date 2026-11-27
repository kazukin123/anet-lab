---
title: ソース配置と Agent の系統別ライブラリ
page_kind: サマリ
material: "[[docs/memo/0182_source_layout_10rfd.md|0182_source_layout_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# ソース配置と Agent の系統別ライブラリ

## 元の材料

[RFD0182　ソース配置と Agent の系統別ライブラリ](../../../memo/0182_source_layout_10rfd.md) blob:4d15cc13753955018b55e44f83f70c9a47875b99

## 要点

- 具象 Agent と共通基盤が同じフォルダにあり、DQN の大きな実装ファイルを責務ごとに分けにくくなっていた。
- ビルド入力、実行時の配置、実行データを区別し、Agent の系統別ライブラリを基盤の外へ移す。
- 各ライブラリの公開ヘッダと内部実装の境界を決め、Runner は公開初期化関数から組み立てる。
- 単体テストは自ライブラリの内部と、相手ライブラリの公開口またはテスト用の口を使う。
- ターゲットと include の依存、旧配置の参照、開発・配布物の起動を検証する。

## 内容

### 目的と問題

ソースの置き場を 1 つの規則で説明できる形にし、Agent を基盤の外のライブラリへ出して、DQN 系の実装に置き場を与える。あわせて、ライブラリとテストが読んでよい範囲を決め、依存の向きを機械が守る形にする。
DQN 系の実装が、基盤の 1 フォルダと 1 ファイルに同居したまま育っている。`dqn_based_agent.cpp` は 6,350 行で、24 日で 28% 増えた。置き場が無いので、責務を分けにくい。
基盤と Agent の境界を機械が守っていない。基盤が具象 Agent を登録し、Agent は基盤の内部ヘッダを読む。公開ヘッダに何を置くかの規約も無い。
トップの階層から、ビルドの入力・実行時の配置・実行時にできるものの別が読めない。`apps/runner` に、ソース、設定、実行体、Run フォルダが同居する。

### 起草時点の現状

動かす追跡ファイルは、`core/anet-core` 352（うち `testdata` 210）、`core/envs` 58、`apps/runner/src` 34、`apps/metrics-viewer` 110、`viewers/metrics-tools` 16、`apps/runner/tools` 11、`apps/runner/workspaces` 8 である。

### ゴール・非ゴール・成功の測り方

Agent を基盤の外のライブラリにする。基盤は具象 Agent を知らず、Agent は基盤の公開ヘッダだけを読む。テストも、この境界を素通りしない。
挙動は変えない。ただし最後のステージだけ、実行時のデータの置き場の決め方が変わる。学習の結果は、どのステージでも変えない。
テスト実行体の DLL の写しが、6〜7 組から 1 組になる。

### 確定事項

本書の範囲に、DQN 系の中のファイルの分け方を入れる。
置き場の規則は 1 つにする。ビルドの入力は `src`、実行時の配置（配る形）は `apps`、実行時にできるものは `data`、配らないビルドの出力は `out` に置く。
層とフォルダは `src/core/base`、`src/env/<名前>`、`src/algo/<系統>`、`src/apps/<app>` にする。フォルダ名は英小文字で、Env は `lunarlander`・`dropmerge`・`atari`・`gridmaze`・`cartpole`・`imagecls`（末尾の数字を外す）、系統は `dqn`・`muzero`・`imagecls`。
ライブラリのターゲットは `anet-<層>-<名前>` にする。基盤は `anet-core-base`、系統は `anet-algo-dqn`・`anet-algo-muzero`・`anet-algo-imagecls`、Env は `anet-env-<名前>`。テスト実行体は、ターゲット名に `-test` を付ける。Runner の実行体 `AnetRLRunner` は変えない。文章の中の「anet-core」は、`src/core` の層を指す語として残す。

### 実装契約

道具が workspace を引く根の計算を、新しい位置に合わせる。data はまだ `apps/runner` の下にあるので、ステージ 4 でもう一度直す。
テスト実行体の DLL の写しは、実行体ごとに写さず、1 つの手順にまとめる。同じフォルダへ同時に写さないためである。実行体の位置から引く口を確かめるテスト（`app_util_test`）は、新しい出力先で成り立つことを確かめる。
依存の向きは、共通の計算、Network、行動、学習、組み立ての順にする。あとのものが前のものを読み、逆には読まない。初期優先度の推定は学習と step 診断が使い、診断の購読の表と指標のキーは行動・学習・step 診断が使うので、共通の計算に置く。
移すための道具を、本書の検証資材（`testdata` の `rfd0182`）に置く。リリースには入れない。道具は、何をどこへ動かすかを表示するだけの実行と、移す前後で workspace と Run フォルダの数を比べる確認を持つ。移す先に同じ名前があれば、上書きせずに止まる。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
