---
title: 課題の作業種別と RFD へのリンク
page_kind: サマリ
material: "[[docs/memo/0120_issue_work_type_and_rfd_link_10rfd.md|0120_issue_work_type_and_rfd_link_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/311_dev_tools.md|開発プロセス・開発ツール・テスト基盤（311）]]"
updated: 2026-11-26
---
# 課題の作業種別と RFD へのリンク

## 元の材料

[RFD0120　課題の作業種別と RFD へのリンク](../../../memo/0120_issue_work_type_and_rfd_link_10rfd.md) blob:236d08c5e9ba0b40679345892d9c2cc353265608

## 要点

- 従来は課題133件中69件が「索引」で、機能追加・不具合・リファクタリングなどの仕事の種類を一覧で絞れなかった。
- type を仕事の種類にそろえ、課題を直す RFD は rfd 欄、関連するだけの文書は docs 欄で指す。
- 親・依存する課題と RFD は Obsidian のリンクで結び、単票から移動できるようにする。
- 同じ RFD が直す課題を単票から集め、判定と状態をもとに、実装済みとして wiki に取り込める時点を決める。
- 既存単票の移行では、見送った課題と実装範囲に入る課題を区別し、一覧と wiki の判定が合うか確かめる。

## 内容

### 問題

課題 133 件のうち 69 件の `type` が「索引」で、その課題が機能追加なのか、リファクタリングなのかが分からない。残りの値も、一般の課題管理の語とずれている。
RFD の実装が済んだかは、索引の課題の AI の判定「実装済み」で表す。段階に分けた RFD では、最初の段階を終えた時点でこの値が付く。wiki は、この値だけを見て RFD を実装済みとして取り込む（52ffba の事象 1）。
`parent`・`depends` は課題の ID だけを書き、RFD は `docs` にパスで書く。そのため Obsidian では、どちらもリンクとして開けない。
開発者は横断のテーマ（NGU など）を `memo` に書いているが、`memo` は自由な文の欄なので、テーマで絞れない。

### 起草時点の現状

欄の値の検査は、wiki_support.py:251 の `validate_issue` が行う。
wiki の取り込み：wiki の点検（`wiki_support.py scan`）は、`type` が索引で判定が「実装済み」の課題の `docs` にある RFD を、実装済みとして拾う（wiki_support.py:305 の `implemented_rfds`）。
統計：直下の RFD ごとに索引の課題を 1 件探し、その `status` を RFD の状態として数える。索引の課題が無いか 2 件あると止まる（stats.py:465〜500）。`type` の件数は「性質別」として数える（stats.py:448・710 行）。
開発者が使う保管庫は、リポジトリ全体を開いている。Bases のフィルタが `docs/management/issues` と書かれていることから、そう判断した。

### 確定事項

課題に `rfd` 欄を足し、その課題を直す RFD を 1 本指す。課題と RFD は N:1 で、1 件の課題が指せる RFD は 1 本までにする。1 件の課題から RFD を 2 本以上起こすときは、元の課題を親にして、RFD ごとに子の課題を作り、子に `rfd` を書く。RFD に関係するだけの課題は、`rfd` を書かずに `docs` に RFD を書く。`rfd` に書いた RFD を `docs` に重ねて書かない。
直下のどの RFD にも、`rfd` で指す課題が 1 件以上ある。RFD の範囲は、それを指す課題の全部で覆う。RFD を書くとき、題材の課題があれば、その課題に `rfd` を書く。題材で覆えない仕事があれば、その分の課題を 1 件作り、無ければ作らない。題材が無い RFD は、その仕事の課題を 1 件作る。RFD 専用の窓口の課題（索引）は作らない。
横断のテーマの欄として、Obsidian 標準の `tags` を足す。`priority`・`memo` と同じ管理外の欄にする。開発者が書き、AI は頼まれたときだけ書き、道具は検査せず、一覧にも出さない。
監査（`anet-audit`）は、RFD の課題も、ほかの課題と同じく確かめ直す。索引を除く決まりと「索引 issue の判定を監査で変えない」決まりは消す。

### 実装契約

人の `status` だけで実装済みにしないという今の決まりは、行 4 の「状態が解決」で変わる。解決は開発者が決めた事実なので、根拠に使う。
一覧（INDEX とトピック別まとめの「課題」の区画）の列は「# / ID / 状態 / WSJF / 待ち / 対象 / 工程 / 領域 / 分類 / 種類 / RFD / AIの判定 / 課題」にする。RFD の列は、RFD の番号に RFD へのリンクを付ける。
「再開候補」の欄のあとに「課題の無い RFD」の欄を作り、直下の RFD のうち `rfd` で指す課題が 1 件も無いものを、RFD へのリンクと題で並べる。無ければ「該当する RFD はありません。」と書く。
skill は `.agents/skills` を直し、`.claude/skills` へ写す（2 つの写しを同じにする）。

### 検証と受入

テストは repo ルートの `.venv` で `.\.venv\Scripts\python.exe -B core/anet-core/testdata/rfd0114/issues_test.py -v` と `.\.venv\Scripts\python.exe -B core/anet-core/testdata/rfd0095/wiki_test.py -v` を回す。どちらも、全件が成功することを合格とする。
`rfd` が過去の置き場の RFD・実在しない RFD を指すと止まる。`docs` に `rfd` と同じ RFD があると止まる。
`resolution` が、解決の課題で空か 5 つのどれでもない場合と、解決でない課題で空でない場合に止まる。
子を持つ課題の `parent`、待ちのある課題の `depends`、RFD の課題の `rfd` のリンクから、課題と RFD を開ける。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
