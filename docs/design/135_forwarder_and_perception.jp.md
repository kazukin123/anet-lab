# Forwarder と Perception

> 主たる観点は、観測を変換する共通の口と、Agent が利用するデータの経路です。

## 1. Forwarder の契約

`anet/forwarder.hpp` の `Forwarder::Forward` は `TensorDict` を受けて `TensorDict` を返します。呼び出す側ごとの状態は実行器が持ちます。Network の重みのように、呼び出す側どうしで共有する値は Forwarder が持てます。

`SpecForwarder` は `ForwardSpec` を提供します。入力の `TensorSpecMap` から、変換後のキー・形・dtype・属性を求めます。`Network` は `Forwarder` として呼び出せます。`ForwardCapture` に渡した鍵の要求は、Network を直接呼び出した場合と同じ契約で扱われます。 [ニューラルネットワークの forward capture](130_neural_networks.jp.md#76-forward-captureと部分forward)

## 2. 観測の spec

`TensorSpec` は batch 軸を除いた形に、値の尺度と stack 軸の有無を添えます。spec の JSON では `scale` と `stack_axis` に出力します。

| 属性 | 意味 |
|---|---|
| `scale = Scaled` | 値を作る側か正規化が尺度を決めています。既定値です。byte 画像や離散値を表す番号もこれに当たります。 |
| `scale = Unscaled` | 観測の正規化で尺度をそろえる対象です。整数 dtype または離散のキーでこの宣言をすると、spec の検査で止まります。LunarLander と CartPole の連続値の `vector` がこの属性を宣言します。 |
| `has_stack_axis = true` | batch 軸を除いた形の先頭が stack 軸です。軸の各位置が 1 枚の Frame に対応します。 |

## 3. DefaultDQN の観測加工

Agent は Frame の spec を1つ作り、ReplayBuffer と `ForwarderChain` に渡します。現在の Frame は Env の Observation と同じキーと形を持ちます。並びが返す Perception の spec を Network の入力に使い、IQN の taus も同じ入力 spec に足します。

| 経路 | 渡す値 | 並びの動作 |
|---|---|---|
| 行動選択 | Observation と lane ごとの `episode_start` | Actor の実行器が device へ転送して履歴に積み、選んだ lane を初期 Frame で埋めます。その後の部品も同じ device で動きます。 |
| 経験の保存 | `obs` と `next_obs` | `FrameWithStatsUpdate` をそれぞれに1回ずつ通します。`Normalize` は経験の受入れ時に観測統計やクリップ率を更新し、同じ Tensor を ReplayBuffer に渡します。 |
| Sample | ReplayBuffer が再構成した Frame の束 | `PerceptionFromStack` が観測、次の観測、TempoRL の窓の終点を変換します。ReplayBuffer が1枚を軸なしで返した場合、この経路で長さ1の軸を足します。 |
| 可視化 | 合成した1枚の Frame | `PerceptionFromFrame` が同じ Frame を指定枚数ぶん並べます。Agent がこの変換と Network を `TensorDictFunction` でつなぎます。 |

## 4. 部品の並べ方

`perception.structure` に `stack > norm` のように名前を並べ、`perception.block.[名前].type` で部品を指定します。`FrameStack` を書かない場合は、全キーを対象とする1枚の `FrameStack` が先頭に入ります。

| 種類 | 設定と動作 |
|---|---|
| `FrameStack` | 先頭に1つ置きます。`count` は1以上で、1枚でも stack 軸を付けます。`keys.mode` が `all` なら全キー、`listed` なら `keys.listed.names` のキーを重ねます。 |
| `Normalize` | 1つまで置けます。`mode` は `none`・`constant`・`running` のいずれかです。`Unscaled` の対象に1 Frame 分の統計を当てます。`constant` と `running` は変換したキーの尺度を `Scaled` にします。 |
| `DropKeys` | `keys` に列挙したキーを Perception とその spec から除きます。保存する Frame にはキーが残ります。 |

構築時に並びと各部品の設定を検査します。未選択モードの値も、型と値域は検査します。`constant.mean`・`constant.std` に値を並べた場合は、定数モードで正規化する各キーの1 Frame 分の要素数と照合します。空の一覧は平均0または標準偏差1を使います。

`running` は経験の受入れで観測統計と `obs_norm.clip_ratio` を更新し、`constant` はクリップ率を更新します。Perception を作る呼び出しは既存の統計を読み、診断の値を更新しません。

Run フォルダの `dot/` に `perception.structure` と `perception.detail` の図を記録します。詳しい図では部品ごとの出力の形、stack の枚数、部品名から導いた seed を確認できます。

## テストと拡張時の確認事項

`[forwarder]` で属性の JSON 出力、尺度の検査、環境が公開する属性、共通の Forwarder を通した Network の出力と capture を確認します。

`[perception]` で1枚と4枚の形、履歴と Sample の対応、統計の更新と読み取り、キーの除去、設定の検査、図を確認します。部品を足すときは変換後の spec と、各依頼で返す値を同じ形にそろえます。呼び出す側ごとの状態が必要な場合は実行器に置きます。
