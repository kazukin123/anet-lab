# 依頼ごとに構築する Forwarder の公開 API

## 元の材料

[RFD0200　依頼ごとに構築する Forwarder の公開 API](../../../memo/0200_forwarder_public_interface_rework_10rfd.md) blob:cd4805e57a58a78b1d8c0cb9ff305b0ee5ed87d8

## 要点

- 観測加工の定義を持つ部品と実行する部品が同じ interface を兼ね、二種類の実行口が残っていた。
- Builder が用途別の依頼と入力 spec を受けて Forwarder を作り、実行側は共通の Forward を呼ぶ。
- エピソードの開始情報は必要な入力として渡し、依頼の意味と並びの妥当性は呼び出し側が受け持つ。
- フレームを重ねる設定は Agent 全体で共有し、Sample の束の spec は束を作る ReplayBuffer が答える。
- 新しい設定へ移したときも、同じ seed と決定的なバックエンドで scalar と trace が一致することを確かめる。

## 内容

### 問題

実行するものが 2 種類ある。`Forwarder::Forward` は `TensorDict` だけを受け、`ForwardExecutor::Forward` は `episode_start` も受ける。使う側は、使い道によって口を使い分ける。
部品の口が 2 種類ある。部品は `SpecForwarder` を実装し、依頼によって動きが違う部品だけ、足して `ForwarderProvider` を実装する。
部品の種類は、forwarder.cpp の中の 3 つ（`FrameStack`・`Normalize`・`DropKeys`）に固定で、ほかの名前はエラーになる（576 行）。NN の module は、type の名前で登録する形である。
`ForwardSpec` は、依頼の指定なしで出力の spec を答える（80 行）。どの依頼の spec なのかが、口から読めない。

### ゴールと非ゴール

あわせて、本 RFD の課題（issue-20261124-6b50cb。公開の口の作り直し）が片付く。

### 確定事項

実行するものは `Forwarder` の 1 つにする。行ごとの履歴を使う変換も、依頼で作った Forwarder が受ける。episode の開始の印は、入力の `TensorDict` に鍵として含め、鍵の名前は依頼を作る側が決める（DefaultDQN は `episode_start` にする）。`ForwardExecutor` は無くす。開始の印は、CPU に置いた真偽の値のまま渡し、device へ移さない。鍵の名前が Frame のキーと重なれば、Agent が構築でエラーにする。
定義を持つ側（並びと部品）は、Forwarder を兼ねない。どの使い方も、依頼を出して Forwarder を受け取る。
並びのクラスの名前は `ForwarderBuilder`、作る口は `BuildForwarder` とする。
依頼で作ったものは `SpecForwarder` で、入力の spec と出力の spec を答える。依頼なしで spec を答える口（今の `ForwardSpec`）は無くす。

### 実装契約

`scale` を `Unscaled` にするとき: 整数の dtype のキーと、離散のキーには書けない（観測の spec の検査 `StateSpec::AssertSanity` と同じ条件）。
`labels` を `set` にするとき: 数は、`TensorSpec::HasValidLabels` の決まり（`Vector` は全要素の数、`Grid` と `Sequence` は先頭の次元の数）に合わせる。
`min_values`・`max_values` を `set` にするとき: 数は、1 つか、1 枚ぶんの全要素の数にする。
`metrics.jsonl` の `scalar` と `trace` には何も足さないので、等価性の比べ方には影響しない。

### 受入

変更の前の `values` の旧いキーのそれぞれについて、直したあとの `values` で、対応する新しいキーが同じ値を持つ。直したあとの `values` に、旧いキーは無く、対応する旧いキーの無い新しいキーも無い。
`map_order` から、旧いキーと新しいキーを除いた残りの並びが一致する。直したあとの `map_order` に、2 の新しいキーが 1 回ずつあり、旧いキーは無い。
道具の `exact_match` が真である。道具は、`metrics.jsonl` の `scalar` と `trace` の全レコード（tag・step・値）を比べ、片方にだけあるレコードが 1 つでもあれば偽にする。値が NaN どうし、値なしどうしのレコードは、同じ tag・同じ step なら一致とする。
除くのは、実時間と資源に依る `90_perf/`・`91_resource/`・`92_resource/` で始まる tag である（RFD0161 の比べ方と同じ）。道具の `--allow-lives-values` と `--allow-migrated-edges` は付けない。checkpoint の raw checksum は使わない。

### 成功の測り方

公開の型のうち、使う側が触るものが、`Forwarder`・`SpecForwarder`・`ForwarderRequest`（と `ForwardKind`）・`ForwarderBuilder` になり、使い方が「依頼を出して、受け取った Forwarder の `Forward` を呼ぶ」の 1 つになる（受入 1〜3）。
forwarder.hpp の `anet` の層の口に、RL の概念が残らない（受入 2・3）。nn.hpp の呼び出しごとの情報にある開始の印は、範囲に入れない（issue-20261124-f9f964 が持つ）。

### 関連

issue-20261124-f9f964（範囲に入れなかった、nn.hpp の側の行の開始の印）

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
