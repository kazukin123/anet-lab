# 文書・コメント・出荷設定の食い違い

## 元の材料

[RFD0119　文書・コメント・出荷設定の食い違い](../../../memo/0119_small_doc_comment_config_fixes_10rfd.md) blob:01a93a2a66b5792cced6390fa989bcd73f120195

## 要点

- 文書やコメント、設定の小さな食い違いを8件まとめて直す。
- 配布 ZIP 名の案内と Optuna 手引きへのリンクを実物に合わせる。
- 設定のサンプルと既定値の記法を現行の契約に照らして修正する。
- リンク切れと既定キー検査を確認し、取り除くサンプル以外の設定解決結果を保存する。
- LOD の説明やリリース物を組み立てる仕組みは、それぞれの RFD で扱う。

## 内容

### 問題

README の Doxygen の zip の名前が、リリースの手順が作る名前と違う。設計文書の optuna の手引きへのリンクが切れている。利用者が README の名前でリリースの zip を探すと見つからず、文書から手引きを開けない。
ReplayBuffer の初回優先度の補完の `@note` が、ロックを持って呼ぶ関数に `GetStats` を挙げていない。
`metrics_scalar.txt` の冒頭の記法の説明は、今は受け付けない `&step` の書き方と、一部の event・target だけを載せている。observers.cpp の中のコメントも同じく古い。
`nn.txt` の冒頭の type 一覧は、登録済みの 7 種を欠き、type ではない `Add` を含む。init.mode のコメントは 6 種のうち 3 種しか載せていない。

### 起草時点の現状

Train panel の fps が 0 でも学習は続く。
課題の状態: 状態（`解決` など）を変えるのは開発者で、AI は対応内容と AI の判定（`ai_judge`）を書く。題材の課題の判定は `残件無し` を使う（anet-create-issue の「各項目の書き方」、anet-update-issue）。

### ゴール・非ゴール・成功の測り方

題材の 8 件の課題をすべて片づける。直した箇所について、リンク切れが 0 件、既定キー検査の errors が 0 件になり、設定の解決結果は、消すサンプルのキーのほかは変わらない。新しい仕組みや概念は足さず、文書・コメント・設定を直すだけにする。
既定キー検査と、題材の 2 ファイル（030・160 の日本語版）のリンク検査が、既存の違反で落ちなくなる。以後の設定変更と文書の直しで、検査の失敗を自分の変更のものとして読める。

### 確定事項

README の zip の名前を `anet-lab-doxygen-<version>.zip` にする。
nn.txt の type 一覧を、nn_modules.cpp が登録する 29 種に合わせ、`Add` を外す。今の見出しを保ち、足りない 7 種を当てはまる見出しへ入れる（`Reshape`・`StackMerge` は【Base / Routing】、`LayerNorm2d`・`GAP2D`・`MaxPool2d` は【Normalization & Pooling】、`HybridSpatialEmbedder` は【Embedding / Bridge】、`Dropout` は新しい見出し【Regularization】）。
Atari.txt の 1650 行を `app.online.eval_panel.fps = 14.5` にする。
RFD0069 の実装記録は変えない。issue-20261103-ee32bd の対応内容に、直さない理由（リンク先はリポジトリに入ったことが無く、過去の記録は書き換えない）を書く。

### 実装契約

過去の Run の設定の写し（`docs/experiments//config/` の `net.body.MLP_RES.structure` などの行、`apps/runner/workspaces//runs/`、`core/anet-core/testdata/rfd0092/config/`）と、workspace の写し `apps/runner/workspaces/AtariRnd-01/config/frozen/nn.txt`。当時の記録である。
課題の単票の題・事象に書かれた当時の名前（`anet-lab-docs-`、`@ResNet` など）。課題の記録であり、課題の一覧と wiki のトピック別まとめの課題の区画は単票から作り直される。

### 検証と受入

どの env でも、差は `net.block.[Linear_Proj].`・`net.block.[Linear_Hid].`・`net.block.[Linear_Out].`・`net.block.[ResBlock].` で始まるキーが消えることだけである。ほかのキーの増減と値の変化は無い。
Atari では、`app.online.train_panel.fps` が `0`、`app.online.eval_panel.fps` が `14.5` で、直す前と同じである。
解決の手段（hidden test の `[rfd0103-resolve]` か、短い Run の設定の写しか）は実装計画で決める。比べた env と、消えたキーの数を実装記録に残す。
コメントだけを変える .hpp・.cpp を含むので、Debug の全体ビルドが通る。`anet-core-test.exe "[shipped_config]"` と既定のテストの集合が、直す前と同じ結果になる（直す前に落ちるテストがあれば、同じものだけが落ちる）。

## 主要な概念

- [開発プロセス・開発ツール・テスト基盤（311）](../../concepts/framework/311_dev_tools.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
