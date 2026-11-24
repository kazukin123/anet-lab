# Forwarder と Perception

## 1. Forwarder の契約

`anet/forwarder.hpp` の `Forwarder::Forward` は `TensorDict` を受け取り、変換した `TensorDict` を返します。Network もこの口を実装します。Network の途中の値は、任意の `ForwardCapture` で取得できます。 [ニューラルネットワークの forward capture](130_neural_networks.jp.md#76-forward-captureと部分forward)

`ForwarderBuilder` は設定に書いた部品を組みます。呼び出し側は `BuildForwarder` に `ForwarderRequest` と、実際に渡すデータの spec を渡します。`ForwarderRequest` は変換の種類を表す `ForwardKind` と、部品が読む付帯情報を持ちます。

作られた `SpecForwarder` の `GetInputSpecs` と `GetOutputSpecs` は、その依頼の入出力の spec を返します。行ごとの履歴が必要な変換では、作られた Forwarder が履歴を持ちます。

## 2. 観測の spec

`TensorSpec` の shape は batch 軸を除いた形です。値の尺度と stack 軸は、JSON の `scale` と `stack_axis` に記録します。

| 属性 | 意味 |
|---|---|
| `scale = Scaled` | 値を作る側か正規化が尺度を決めています。byte 画像や離散値を表す番号も含みます。 |
| `scale = Unscaled` | 正規化で尺度をそろえる連続値です。整数 dtype と離散のキーでは使えません。 |
| `has_stack_axis = true` | batch 軸を除いた形の先頭が stack 軸です。軸の各位置が1枚の Frame に対応します。 |

## 3. DefaultDQN の観測加工

DefaultDQN は、Env の観測と同じ spec を Frame の spec にします。経験の受入れ用を最初に作り、出力のキー・形・dtype が Frame と一致することを確認します。ReplayBuffer にも Frame の spec を渡します。

| 依頼 | 入力 | 変換 |
|---|---|---|
| `FrameWithStatsUpdate` | `obs` と `next_obs` を1枚ずつ | 統計を更新して、元の値を ReplayBuffer に渡します。 |
| `PerceptionFromSequence` | Actor の device に移した観測と、CPU 上の真偽値 `episode_start` | lane ごとの履歴を重ねます。開始の印が真の lane は今の Frame で履歴を埋め直し、出力から開始の印を除きます。 |
| `PerceptionFromStack` | ReplayBuffer が重ねた束。`GetSampleObsSpecs` が束の spec を返します。 | Learner が観測、次の観測、TempoRL の窓の終点を変換します。 |
| `PerceptionFromFrame` | 可視化のために用意した1枚の Frame | `FrameStack` は同じ Frame を指定枚数ぶん並べます。 |

重ねる設定は Agent 直下の `frame_stack.count`、`frame_stack.keys.mode`、`frame_stack.keys.listed.names` です。コードの既定は1枚、出荷の基準プロファイルは4枚です。`all` は Frame の全キー、`listed` は一覧にある Frame のキーを選びます。Agent は同じ枚数とキーを ReplayBuffer と各依頼へ渡します。

DefaultDQN は、選んだキーで stack 軸がない出力に長さ1の軸を足し、開始の印が残っていれば除きます。Network の入力には可視化用の出力 spec を使います。行動選択用と Sample 用の出力 spec も、この入力と構築時に照合します。IQN の taus は Network の入力 spec に加えます。

## 4. 標準の部品と設定

`perception.structure` に `stack > norm` のように名前を並べ、`perception.block.[名前].type` で部品を選びます。Builder は宣言順に組みます。各部品が自分に届いた spec を検査し、出力 spec を次の部品に渡します。

| 種類 | 設定と動作 |
|---|---|
| `FrameStack` | block の設定は `type` です。枚数とキーは依頼から受け取り、自分に届いた対象キーを重ねます。Sample のキーに既に stack 軸があれば、そのまま通します。 |
| `Normalize` | `mode` は `none`・`constant`・`running` です。正規化したキーの尺度を `Scaled` に変えます。 |
| `DropKeys` | Perception のデータと spec から `keys` を除きます。経験の受入れではキーを通します。 |
| `SetSpec` | `keys` に指定したキーの `scale`、`min_values`、`max_values`、`labels` を変えます。値はそのまま通します。`forward_kinds.mode=listed` で指定した種類に絞れます。 |

`running` の統計は、経験の受入れ用の Forwarder を作るとき、部品に届く `Unscaled` の float キーを使って用意します。あとで落とすキーの統計も取ります。統計が不要なキーは、前に置いた `SetSpec` で `Scaled` にできます。別の種類で正規化を要求したキーの統計がないか、1枚ぶんの形が違う場合は構築で止まります。

`running` は観測統計と `obs_norm.clip_ratio` を経験の受入れで更新します。`constant` はクリップ率を更新します。Perception を作る処理は統計を読みます。`constant.mean` と `constant.std` に値を並べると、1枚ぶんの要素数との一致を検査します。

`SetSpec` は指定した属性だけを検査します。範囲値を設定するときは、全要素に共通の値を1個、または1枚の全要素ぶんを指定します。ラベルは Vector では全要素ぶん、Grid と Sequence では1枚の先頭次元ぶんを指定します。種類を絞った結果、各経路で正規化するキーが変わることがあります。

## 5. 図と拡張

DefaultDQN は、可視化用の Forwarder が生成する `perception.structure.dot` と `perception.detail.dot` を Run の `dot/` に記録します。図は依頼の種類を題にし、入力、設定順の部品、出力を結びます。詳細図では出力 spec と部品の seed・設定値を確認できます。

部品を追加するには、`ForwarderModuleFactory` に対応する Forward の種類を宣言し、`ForwarderModule` が依頼と入力 spec に合った `SpecForwarder` を返すようにします。`RegisterForwarderModuleFactory` で設定の `type` の名前を登録します。その依頼で変更が不要な部品は `MakePassThroughForwarder` を使えます。

Builder は、選ばれた全 block で工場の `ReadConfigKeys` を呼び、未読キーを検査します。設定を持つ部品はこの口で受け付けるキーの値を文字列として読みます。工場の `CreateModule` は設定値の型と値域を検査して部品を作ります。並び全体で使える Forward の種類との組み合わせを検査する部品は `ValidateForwardKinds` を実装します。この口は、全 block の `CreateModule` が成功したあとに呼ばれます。

`[forwarder]` では登録と公開口、`[perception]` では4種類の入出力 spec、履歴と Sample の往復、統計、設定と図を確認します。
