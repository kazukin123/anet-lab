# DQN 学習更新と Munchausen ターゲットの共通化

## 元の材料

[RFD0125　DQN 学習更新と Munchausen ターゲットの共通化](../../../memo/0125_dqn_learner_update_skeleton_10rfd.md) blob:45f3305a58850a91b00c7da662c4a06d7c690c81

## 要点

- TD・QR・IQN の三つの学習器に重複していたバッチ更新を共通の進行にまとめる。
- 分布の表し方に固有のターゲットや要素誤差は、それぞれの学習器が計算する。
- Munchausen 項と soft 価値を含むターゲットで共有できる処理も一か所に置く。
- 優先度の反映、最適化、補助損失の順序など、複製間で違っていた箇所を点検する。
- 同じ seed の決定的な比較で三方式のメトリクスと実時間を確かめ、PER・skip・Munchausen などの条件も含める。

## 内容

### 問題

本 RFD で target と書くものは、用語集の Bellmanターゲットである。Munchausen の target は、Munchausen項と soft価値ブートストラップを使う Bellmanターゲットを指す。target の式は、target return と bootstrap 価値から Bellmanターゲットを組む式を指す。
Munchausen の target は、通常 Q 用（`Learner::MakeTarget` の中の枝）と分位点用（`QuantileLearnerBase::MakeMunchausenTarget`）の 2 か所にある。2 か所は前半を同じ並びで書いている。違いは、読む出力の名前と、IQN の τ を入力へ足すかだけである。
`log_policy_mode` を足す、capture の扱いを変える、といった変更で、2〜4 か所を同じに直すことになる。
価値の分解と内的報酬は、3 つの学習器の target と要素誤差のあいだへ処理を足す。写しのままでは、同じ変更を 3 回書くことになる。RFD0173 の課題は、題材の課題を待っている。

### 起草時点の現状

対象は dqn_based_agent.cpp と dqn_based_agent.hpp である。
skip の loss は、PER が無効のとき重みを掛けない（4687 行）。TD と同じ形である。TD は、重みを loss の device へ移す呼び出しを挟むが、同じ device なので何もしない。
分位点の要素 loss は float32 である。Head が FP32 で動くので、current の分位点も target も FP32 で届く。1 を並べた float32 の Tensor を掛けても、値も型も変わらない。重み付けした要素 loss は、最適化と勾配ノイズの測定へ渡るが、1 を掛ける演算が無くなっても、勾配の値は同じである。
Munchausen の計算は実空間で行い、完成した target にだけ h を掛ける。h の逆変換を中で行う `CalcTargetQuantiles` は通常の target 専用で、逆変換を 2 回掛けないための決まりである（ADR 0035 の 45 行）。

### 確定事項

分布ごとの差し替えの口。 仮想関数を 2 つ足す。1 つは current 側の計算で、forward と、損失に使う値と、指標を返す。もう 1 つは追加の診断で、今は IQN だけが実装する。通常 Q の実装は基底に置き、分位点の側が上書きする。分位点の current 側の計算（forward、選んだ行動の分位点、指標）は、QR と IQN で 1 か所に書き、IQN は τ に関わる部分だけを足す。
`UpdateFromSamples` の扱い。 仮想関数のまま残す。`TDLearner`・`QRLearner`・`IQNLearner` からは、今と同じく外から呼べるようにする。3 つの学習器に、基底を呼ぶだけの上書きは書かない。
PER が無効のときの重み。 掛けない形にそろえる。PER が有効のときは、抽選した batch の重みをそのまま掛ける。重みを device へ移す呼び出しは置かない。
DEBUG ログ。 `element_loss=` と `weights=` の 2 行を残し、進行の中の今と同じ位置（要素誤差のあとと、重み付けの前）に置く。TD と IQN でも出るようになる。`weights=` の行は、PER が有効のときだけ出す。コメントアウトされた `obs=` の行も、進行の入力の検査のあとへ移して残す。

### 実装契約

skip の loss。今ある `ComputeSkipLoss` を呼ぶ。渡す入力と τ は、current 側の計算が返したものを使う。
要素誤差。今ある `ComputeElementError` を呼ぶ。渡す τ、current の平均、補助の統計を求めるかの指定は、current 側の計算が返したものを使う。補助の統計を求めるのは、今と同じく IQN だけである。区間 `loss` はここから 9 の終わりまで。あとに DEBUG ログ `element_loss=`。
τ を足すと渡されたときだけ、2 と 6 の τ を作る。作る件数は今と同じである。2 は、target 方式なら current と next の行数の合計、ほかの方式なら next の行数。6 は current の行数。
Munchausen項があるときは、target return に Munchausen項を足してから、bootstrap の項を足す。Munchausen項が無いときは、0 を足す形にせず、target return に bootstrap の項を足す。

### 検証と受入

「保存した実行体」は、ステージ 0 で保存したものを指す。「全 scalar の点が一致する」は、時間と資源の tag（`90_perf/`・`91_resource/`・`92_resource/` で始まるもの）を除く全 scalar の点を、tag・step・値で比べて、すべて同じであることを指す。除くのは、実行ごとに変わるためである。
27 組のすべて: `46_agent_replay_fit/` で始まる tag（replay fit の診断）に、NaN でない値の点が 1 つ以上ある。
Munchausen が有効の組: `36_agent_munchausen/` で始まる tag（Munchausen の診断）に、NaN でない値の点が 1 つ以上ある。
IQN で `learner.use_per = false` の組: `36_agent_iqn/01`〜`36_agent_iqn/03` のすべてに点があり、3 つのうち 1 つ以上が NaN でない値の点を持つ。初回優先度の経験だけを使う 04〜07 は、この組では未成立であり現行の Observer が出力しないため、実点の有無は受入にしない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
