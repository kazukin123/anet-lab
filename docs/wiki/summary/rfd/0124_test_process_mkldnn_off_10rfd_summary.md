---
title: anet-core-test の oneDNN 初期化条件
page_kind: サマリ
material: "[[docs/memo/0124_test_process_mkldnn_off_10rfd.md|0124_test_process_mkldnn_off_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-28
---
# anet-core-test の oneDNN 初期化条件

## 元の材料

[RFD0124　anet-core-test の oneDNN 初期化条件](../../../memo/0124_test_process_mkldnn_off_10rfd.md) blob:ff77749c460ad6f1ba4a66aee4bf97f626664b8b

## 要点

- CPU の畳み込みを使う Debug テストが oneDNN 内部の assert で止まることがあった。
- 一部の試験だけで MKLDNN を切っていたため、テストを単独で動かす場合とランダム順に動かす場合で結果が変わった。
- テストプロセスの開始時に MKLDNN を切り、個々のテストの実行順への依存をなくす。
- 当該テストの単独実行と出荷設定の試験群で、後続試験まで進むことを確認する。
- 製品 Run のバックエンド設定とテストの前提を分けて記録する。

## 内容

### 問題

Debug 版の anet-core-test で、MKLDNN を切るのは一部のテストの中だけで、そのテストが回ったあとでないと切れていない。テストの実行順は Catch2 の既定でランダムなので、CPU で Conv を組むテストが oneDNN の経路を通るかどうかが、回すたびに、また回すテストの選び方で変わる。Debug 版 libtorch の oneDNN は、CPU の Conv で thread 数を確かめる assert（`nthr_ == nthr`）を持ち、それに当たると実行体ごと終了コード 3 で止まる。

### 起草時点の現状

MKLDNN を切るのは、テストの中の 2 か所である。どちらもビルド構成を問わず `at::globalContext().setUserEnabledMkldnn(false)` を呼ぶ。
止まるかどうかは、先に回ったテストが MKLDNN を切ったかで決まる。RFD0028 の実装記録（16 行）は、assert が CNBlock の演算の連鎖で出るとしていたが、CNBlock を使わない題材のテストでも出る（1 の Network）。

### 確定事項

範囲。 anet-core-test の main だけを変える。ほかの 6 つの実行体の main は変えない。
本番。 `InitNN` と Run の初期化は変えない。
見張りのテスト。 テストが始まった時点で MKLDNN が切れていることを確かめるテストを 1 本足す。ほかのテストに頼らず、単独で回して通る形にする。

### 実装契約

足す・変える・消すクラスは無い。変えるのは、anet-core-test の main 関数と、テストの補助の関数 2 つの中身である。

### 検証と受入

Debug ビルドを通し、既定のテストの集合を回す。直したあとの実行は、終了コード 3 で止まらずに最後まで回り、直す前に通っていたテストが落ちない。比べる単位はテストケースの名前で、範囲は既定の集合の全部（隠しタグのテストは含まない）である。
題材の課題の対応内容に、直した内容と場所が書かれ、AI の判定が `残件無し` である。状態を `解決` にするのは開発者である。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
