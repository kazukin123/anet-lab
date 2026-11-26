# Forwarder と Perception

## 元の材料

[Forwarder と Perception](../../../design/135_forwarder_and_perception.jp.md) blob:ebacf41daac668178c1fb44d4ff202d60afc1968

## 要点

- Forwarder は入力を変換する共通の口で、Builder は依頼と実際の入力 spec から必要な部品の並びを作る。
- TensorSpec は batch 軸を除いた形を持ち、値の尺度と stack 軸を明示できる。
- DefaultDQN の行動選択、経験の格納、Sample の各経路では、入力と必要な出力に合わせて部品を選ぶ。
- 重ねる枚数と対象キーは Agent が配り、ReplayBuffer が Sample で返す束の spec を答える。
- 標準部品による変換は適用されるキーと形を検査し、同じ設定でも対象に合わない経路を区別する。

## 内容

### 1. Forwarder の契約

- `ForwarderBuilder` は設定に書いた部品を組みます。呼び出し側は `BuildForwarder` に `ForwarderRequest` と、実際に渡すデータの spec を渡します。`ForwarderRequest` は変換の種類を表す `ForwardKind` と、部品が読む付帯情報を持ちます。

### 2. 観測の spec

- `TensorSpec` の shape は batch 軸を除いた形です。値の尺度と stack 軸は、JSON の `scale` と `stack_axis` に記録します。

### 3. DefaultDQN の観測加工

- `FrameWithStatsUpdate` — 入力は`obs` と `next_obs` を1枚ずつ。変換は統計を更新して、元の値を ReplayBuffer に渡します。
- 行動選択の `PerceptionFromSequence` は開始した lane の履歴を今の Frame で埋め直す。学習用の `PerceptionFromStack` には ReplayBuffer が重ねた束、可視化用の `PerceptionFromFrame` には単一の Frame を渡す。

### 4. 標準の部品と設定

- `FrameStack` — block の設定は `type` です。枚数とキーは依頼から受け取り、自分に届いた対象キーを重ねます。Sample のキーに既に stack 軸があれば、そのまま通します。
- `Normalize` は選んだキーの値を変換し、`DropKeys` は不要なキーを除く。`SetSpec` は値を通したまま指定属性を検査・変更する。

### 5. 図と拡張

- Run の構造図では依頼の種類、入力、部品の順序、出力を確認できる。新しい部品は factory に対応する変換の種類を宣言して登録し、構築時に設定値と組み合わせを検査する。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
