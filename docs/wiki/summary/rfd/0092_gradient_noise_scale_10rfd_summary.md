---
title: 勾配ノイズの大きさ（gradient noise scale）の計測 PRD
page_kind: サマリ
material: "[[docs/memo/done/0092_gradient_noise_scale_10rfd.md|0092_gradient_noise_scale_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/phenomena/gradient_noise_batch.md|勾配ノイズとbatch]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# 勾配ノイズの大きさ（gradient noise scale）の計測 PRD

## 元の材料

[勾配ノイズの大きさ（gradient noise scale）の計測 PRD](../../../memo/done/0092_gradient_noise_scale_10rfd.md) blob:19d1262838e14f27ee7c9b03ec150eb56a84ec73

## 要点

- 学習に使った同じバッチを複数に分け、更新に対する勾配ノイズの規模を learn_step 軸で測る。
- PER、IS 重み、IQN の τ、Munchausen など実際の学習条件を残し、ノイズは更新時と同じ1抽選を使う。
- 小分けにした損失ごとの勾配を取り、通常の更新経路と分けて統計を計算する。
- 未購読なら計測コストを生じさせず、BF16 AMP を使う経路でも読めるようにする。
- この値からのバッチサイズ自動調整や、層別・曲率込みの臨界バッチ推定は対象外とする。

## 内容

### 目的と問題

今のバッチ（DropMerge では 256）は、学習のどの時期からノイズに負けているか。
anet には勾配ノルムはある（`Optimize` の `grad_norm_tensor`）が、勾配の「ばらつき」と「平均の向き」を分けて測る指標は無い。

### 概念

学習の 1 回の更新は、replay から B 個の経験を取り、その平均の損失の勾配で重みを動かす。B 個は全データの一部なので、この勾配は「全データで平均したときの勾配」（真の勾配 $G$）の推定値で、取った B 個によってぶれる。
経験が同じ分布から独立に抽出されるなら、ノイズの大きさは $1/B$ で小さくなる。バッチを倍にすると、ノイズの 2 乗は半分になる。
目標が動く: TD の目標は target network から作るので、損失そのものが学習とともに変わる。ここで測るのは「その時点の更新の損失」のノイズであり、教師あり学習の値とそのまま比べない。連続する learn step をまたいで平均すると θ の動きが混ざるので、1 回の測定は 1 更新の中で閉じ、平均は測定間の EMA で取る。
何がノイズに入るか: 経験の取り方（PER）、IS 重み、IQN の τ の抽選（経験ごとに抽選）、Munchausen の項は、どれも 1 回の更新の勾配のばらつきに入る。これらを含めた「実際に当てている更新」のノイズを測るのが目的である。

### 現状

指標の値の載せ方には 2 つの経路がある。学習の更新そのものから取る値（actual チャネル）は `$update_result`、probe の値は `$agent` である（062）。既知 key で値が無いときは NaN、未知 key だけ nullopt。
metrics 定義側の `$ema` は 1 つの key の値の列にかける EMA で、分子と分母を別々に平滑化してから割る仕組みは無い。比をライブで読むには Learner が EMA を持つしかない。
指標の定義は metrics_scalar.txt にあり、群の番号は 46（replay_fit）・47（noisy）まで使っている。44・45・48・49 は空きで、48 は attention エントロピーの暫定 PRD が使う予定である。

### ゴールと非ゴール

実際の更新（PER・IS 重み・τ・Munchausen 込み。NoisyNet の ε は更新と同じ 1 抽選に固定）の noise scale を、Adam 前処理版を主読みとして、ネット全体と feature / readout ごとに `learn_step` 軸のスカラーとして記録する。
本線の DropMerge Run（BF16 AMP ON・B256/RR1・NoisyNet）で読める。
全体と feature / readout を含む 15 指標を一式で完成・検証する。 完成範囲は計測機能の正しさ、本線での動作、実時間・追加メモリの実測報告までとし、因果を評価する分岐キャンペーンは別の実験とする。

### 決定した設計

`Optimize` を `Optimize(loss, weighted_element_loss)` の 2 引数にする。3 つの Learner は今と同じく `weighted = element_loss * weights`（長さ B）と `loss = weighted.mean()` を autocast ブロックの中で作り、`weighted` を添えて渡すだけで、`loss` の計算は動かさない（学習経路の数値は変更前と同一）。測定は Optimize の先頭（`zero_grad` と `backward` の前）で行う。
公開は計 15 指標。比の raw は平均し直せないので、raw は信号と分散で出す。内訳も scale だけでなく raw を残し、未成立の調査や平滑化のやり直しに使う。各 key は自身の購読が発火した更新だけで値を公開し、未購読・未発火・未成立は NaN、未知 key だけ nullopt とする。測定自体が無い更新では全 key が NaN となる。
購読: 新しいプロファイル `metrics.scalar.@gradnoise` を置く。baseline には入れない。既定の interval は 503（素数。hard update C=200 / 120 の群で位相が固定されるのを避け、churn・noisy・replay_fit と同じ learn_step を測るので同じ更新を横に読める）。
各購読行の `IntervalGate`（初回発火と bucket-crossing）で要求を判定し、その和集合で測定する。同じ更新で複数行が発火しても、小バッチ勾配の取得は 1 組だけとする。購読されている版・群はその測定点を共有し、成立した EMA 対を各 1 回更新する。その回に記録する key は各購読行の発火条件で決める。

### 受入基準

同じ Release ビルドと本線構成を使い、計測 OFF → ON を交互に各 3 回、計 6 回測定する。ON は全 15 指標・interval 503・$k=8$・EMA 係数 0.05 とする。初期 checkpoint、warmup の長さ、warmup 後に比較する学習量（同じ learn step 数）、出力などの条件をそろえる。
従来の「NaN 率」は 値の未成立率 として扱う。計測を有効にした正常完了 Run について、実際に行われた学習更新と購読 interval から、各 key 自身の発火規則（初回発火・bucket-crossing）に従う計測予定点を求め、記録された有限値を照合する。
分母は対象 key の計測予定点数、分子はそのうち有限値を記録できなかった点数とする。通常の未計測 step や、その key が発火しない他の購読の測定点は分母に含めない。空群の群に属する key は構成から静的に NaN と決まるので、計測予定点を 0 とし、率は未定義とする。
対象区間・購読設定・予定点数・有限値数を併記する。予定点が 0 なら率は未定義とする。

### 先送りした段階と、作る条件

ε を引き直す版の総ノイズ測定: 小バッチごとに NoisyNet の ε を引き直して、ε のばらつきまで含めた noise scale を測る（forward が増え、専用の乱数が要る）。作る条件: 分岐 Run の結果が条件付きの値の読みと食い違い、かつ `47_agent_noisy/09_q_abs_diff` が大きい（ε の効きが無視できない）とき。
層ごとの内訳: feature / readout で差が見え、どの block かを特定する必要が出たとき。
PPO / SAC への対応: その Learner と観測したい損失・optimizer が具体化したときに、計算部の再利用範囲と必要な API を検討する。今回の共通部は単一損失・AdamW の用途で完結させ、複数損失の key、別 optimizer の拡張口、ロールアウト相関への対応を先に設計・実装しない。

### 複雑さ監査（最終簡素化パス）

BF16 本線・非干渉・15 指標の一式完成というゴールアンカーに照らし、残す量と先送りする機構を再確認した。15 指標は数学的な最小集合ではなく、実行中の読みやすさと終了後の再集計を両立させるために選んだ完成範囲である。
共有測定点を使う版・群ごとの EMA。購読の和集合で測り各 EMA を 1 回更新する。購読追加で時間幅が変わる性質と、再現に必要な raw の範囲を明記する。
使用パラメータ・群ごとの state 成立判定。今回の損失と無関係なパラメータや別群の未成立で、成立する観測まで失わないため。stale `.grad` を根拠にしない。
未成立理由の追加状態指標。正常完了 Run の予定点と有限値、raw・平滑化値・測定条件で読む。理由が一意に復元できない場合も含め、追加の公開契約を増やさない。

## 主要な概念

- [勾配ノイズとbatch](../../concepts/phenomena/gradient_noise_batch.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
