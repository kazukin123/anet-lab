# Replay優先度更新にはgeneration付きreplay item keyを使用する

ADR 0005では、`PrefetchingReplayBuffer`が1-step staleな経験をSampleし、その経験で学習することを性能上の選択として許容した。当時はpublic `ReplayBuffer`インターフェースを維持し、Sampleが返す物理indexを後続の`UpdatePriorities`へ返す設計だった。

その後、write-behind Pushを含む順序が`F1 Sample -> Pushによるslot上書き -> F2 Sample -> F1 UpdatePriorities`になり得ることを確認した。F1の経験tensor自体はSample時のスナップショットなので学習可能だが、物理indexだけでF1の優先度を返すと、既に同じslotへ入った別の遷移を更新する。これは許容済みのstale samplingではなく、異なるreplay itemへの誤mutationである。

## Considered Options

- **物理indexのまま更新する**: public APIは変わらないが、上書き後itemへの誤更新を防げないため棄却。
- **indexとgenerationを別tensorで返す**: identityは表現できるが、ReplayBuffer内部事情を2本の外向けmetadataとして露出するため棄却。
- **Sample/UpdateのFIFO ticketをReplayBuffer内部で対応付ける**: public APIを維持できる一方、SampleとUpdateが必ず1対1・同順序で到着する隠れた契約が必要になり、欠落、重複、将来の呼び出し変更に弱いため棄却。
- **generationと物理slotを1個の`int64` keyへpackする**: 既存の1本の往復metadataでitem identityを表現でき、各更新を独立に検証できるため採用。

## Decision

- 丸め後の`actual_capacity = capacity_per_env * num_envs`をkeyの基数とし、SumTree容量も同じ`actual_capacity`へ統一する。
- per-slot generationは未書き込み時0とし、realまたはdummyの書き込みごとに1増加させる。n-step確定や優先度更新では増加させない。
- Sampleは`replay_item_key = generation * actual_capacity + flat_slot_index`をCPU `int64`で返す。専用の`ReplayItemKey`型やaliasは設けず、名前とencode/decode helperで意味を示す。
- `UpdatePriorities`は、key generationが現在と同じ要素だけを適用する。過去generationは上書き済みstaleとしてその要素だけを棄却し、未来generation、generation 0、負値、overflowはプログラムエラーとして扱う。
- stale要素はleaf、優先度source、最大優先度、Actor/Learner比較を変更せず、stale-dropだけを計上する。同じbatchの有効要素は適用する。
- raw物理indexを受ける互換経路は設けない。keyは生成元ReplayBufferの生存期間内だけ有効であり、直列化しない。

## Consequences

- `ExperienceSamples::indices`は`replay_item_keys`へ、`UpdatePriorities`引数は`item_keys`へ改名される。内部の物理位置には`slot_index`／`flat_slot_index`を使う。
- `UpdatePriorities`は呼び出し単位の`ReplayPriorityUpdateResult`を返し、適用数、stale数、Actor/Learner比較の集約値をLearner minibatchへ対応付ける。
- duplicate keyは入力順に適用し、leafはlast-winsとする。Actor初期値との比較は最初の`actor_initial -> learner_updated`だけを記録する。
- generation配列と検証コストが増えるが、追加tensor、GPU転送、ネットワークforwardは増えない。
- ADR 0005の「1-step staleな経験を使う」という判断は維持する。本ADRは、ADR 0005当時のpublic IF不変という制約を後発のitem identity要件で置き換え、stale experienceとstale priority mutationを区別する。
