# 学習前の Evaluation Trace に表示する RND の β

## 元の材料

[RFD0129　学習前の Evaluation Trace に表示する RND の β](../../../memo/0129_eval_trace_bonus_beta_before_learning_10rfd.md) blob:5937f3da3e76890d4038d903b27c5cf051fff70a

## 要点

- 学習の入口で直近の β を記録するまで、Trace のボーナス列が空欄になる問題を扱う。
- 評価専用 Run や checkpoint を読み込んで見るだけの Run でもボーナスを表示する。
- RND 有効時は Agent の構築時に exp step 0 の β を評価し、初期値として持つ。
- その後は学習の入口で対応する exp step の値に更新し、Trace で表示するときに予定を進めない。
- 既存の checkpoint を読み戻せるように保存形式は変えず、起動直後と学習開始後を比較する。

## 内容

### 目的と問題

評価専用の Run（`learner.enabled=false`）では、学習の入口が β を評価しないので、最後まで空欄のままである。
checkpoint を読んで学習を始めずに見る使い方でも、同じく最後まで空欄のままである。

### 起草時点の現状

使っている β は定数だけである。既定 0.1（agent.txt:4）、Taiga 式 0.00005（Atari.txt:36）、Montezuma の継続 Run 0.01（実効設定の写し:816）。
β に予定を持たせた設定は、追跡しているファイル（`git grep` で `intrinsic.beta.type`・`.start`・`.end`・`.steps`・`.phase` を引いて 0 件）にも、手元の Run の実効設定にも無い（2026-11-13）。
intrinsic_test.cpp の `DQN trace scores its Actor snapshot after synchronization`（1834 行）は、学習の入口を通る前に Actor を同期した行のボーナスが NaN であることを断定する（1856 行）。

### 確定事項

RND が有効な DefaultDQN は、構築時に β の予定を exp step 0（この Run の step の始まり）で評価し、直近の β にする。学習の入口は、今までどおり評価した値で直近の β を上書きする。
checkpoint の書式を変えないので、今ある RND の checkpoint（課題の事象の 50M を含む）をそのまま読める。β を checkpoint に保存する形は、書式の識別子を上げて今ある checkpoint を読めなくし、課題の事象の checkpoint そのものを読めなくする。
Run の step は 0 から数えるので、exp step 0 は、この Run の予定の始まりの位置そのものである。黙った代替や特別な値ではない。
変えるときの条件: Run の step 数を checkpoint から復元するようになったら、構築時の評価を、復元した step で評価する形に直す（開発者の答え「step 数の復元は永続化の本実装で検討する」による）。

### 実装契約

足す口も、変える口も無い。鍵（`intrinsic.error`・`intrinsic.normalized_error`・`intrinsic.bonus`）と列の宣言は、RFD0127 のまま変えない。
wiki: トピック別まとめの 324 DQN 系列 と、内的報酬のページ（「学習の入口を通らず」の言い回し）。本 RFD の起草の書き戻しで行った。

### 成功の測り方と受入

1〜5 はテストで確かめる。6 は人が確かめる。数えるのは、テストの合否と、テストが断定する値である。

### ほかの RFD との関係

RFD0184（暫定・未実装）: 評価 session で同じ鍵を使う。本 RFD のあとは、session の開始の同期のあと、最初の step からボーナスが入る。

## 主要な概念

- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
