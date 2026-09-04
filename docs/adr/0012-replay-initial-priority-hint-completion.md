# Replay初期優先度ヒントを単一packed carrierとし、completionを専用coordinatorへ分離する

> **後発決定（2026-09-04）**: DQN payloadの列数`K = 2`は [ADR 0036](0036-actor-q-hint-three-columns-munchausen.md) で`K = 3`（Munchausen項を追加）へ改訂した。単一carrier、completer、推定器の責務分離は不変である。

PRD 035（`docs/memo/035_approx_actor_priority_per_10prd.md`）の初期実装では、汎用RLヘッダの`ActorQHint`がDQN固有の`Q(s,a)`、状態価値、CPU validity maskを直接表現し、`DefaultReplayBuffer`が完成待ちFIFO、推定器、フォールバック、source、counterを一括して保持した。この形では、アルゴリズム固有payloadの意味が共通transportへ漏れ、初期優先度completionのstateとFacade所有resourceの境界も不明確になる。

ADR 0010で採用したDQN近似の数式、追加forwardを行わない方針、`WithAction`で`actor_q_sa`を再gatherする契約は維持する。本ADRは、そのDQN payloadを運ぶ共通carrierと、ReplayBuffer内で初期優先度を完成させる責務だけを後発判断として置き換える。

## Considered Options

- **DQN固有の`ActorQHint`とvalidityを共通RL層に残す**: 実装差分は小さいが、非DQN AgentにもQ値と固定列数を要求し、現在すべて1であるvalidityを常設するため棄却。
- **`TensorDict`で任意のnamed tensorを運ぶ**: 拡張性は高いが、複数valueのCPU化が複数D2Hになり、1 keyへpackすると単一tensor carrierと実質同じになる。文字列keyと任意個tensorを許して共通契約を緩める利点がないため棄却。
- **単一の`float32[B,K]`を不透明に運ぶ**: 共通層がrank、dtype、batchだけを扱い、payload schemaをAgent側へ閉じ込めながら物理D2Hを1本に固定できるため採用。
- **completionを純粋decision engineにし、適用をFacadeへ残す**: sampleable確認、fallback、source、counterがFacadeへ再分散し、判断結果を逐次返すcommand APIも必要になるため棄却。
- **completionを内部coordinatorにする**: completion stateと判断を一か所に集約しつつ、共有resourceとmutexの所有はFacadeに残せるため採用。

## Decision

- 共通型を`ReplayInitialPriorityHint`とする。計算グラフから切り離した連続`float32[B,K]`（`K > 0`）を1本だけ保持し、初回CPU要求時に同期転送してcacheする。carrierが存在すれば全batch行が有効であり、別のvalidity maskは持たない。
- DQN payloadは`K = 2`のActor Qヒントとし、`actor_q_sa`と`actor_state_value`を格納する。列数、列index、pack/decodeはDQN moduleの共通helperが所有し、Actor、`DQNActionInfo::WithAction`、DQN初期優先度推定器が同じschemaを使う。
- ReplayBufferはCPU化後の各行を内部の`c10::SmallVector<float, 4>`へコピーし、推定器へ`std::span<const float>`として同期的に渡す。libtorch固有の所有型を公開推定器interfaceへ露出しない。
- 初期優先度推定器がpayload schemaとfinite判定を所有する。`ValidateHint`はschema違反をエラーにし、finiteならtrue、NaN/Infならfalseを返す。truncatedでも開始hintを先に検証する。`Estimate`の`std::nullopt`は入力または計算結果のnonfiniteだけを表し、推定器自身はfallback値を選ばない。
- `InitialPriorityCompleter`はenv別の完成待ちFIFO、mode処理、推定器、fallback、初期系sourceの決定、completion理由counterを所有する。per-slot source配列とsource別質量は`DefaultReplayBuffer`が所有する`ReplayPriorityStore`に保持し、completer、slot無効化、Learner更新の各経路が同じstoreの明示APIからsource遷移を適用する。`DefaultReplayBuffer`は`ValidIndexManager`と`metadata_mutex_`も所有し、同じ同期区間内でcompleterへ狭いinterface経由の照会・適用を許可する。
- completerはPER時だけ構築する。PERの`fixed`、`max`、`actor_approx`は同じサンプリング可能化境界で完成させる。一様ReplayBufferはcompletion経路と完成待ちFIFOを持たない。
- sampleable範囲式は`ValidIndexManager`が一元所有し、列挙、単点判定、上書き判定から共用する。

## Consequences

- 汎用RL層からQ値、固定`K = 2`、validityの概念が外れ、DQN以外のAgentも同じ単一tensor carrierを独自schemaで利用できる。
- 将来DQN payloadに部分無効行が必要になった場合は、DQN schemaの列を追加して明示的に表現できる。共通carrierへCPU validity tensorを常設しない。
- DQNの現行payloadはinline capacity内に収まり、env行ごとのheap allocationを増やさない。追加network forward、複数tensorのD2H、slot単位の長期hint保存は発生しない。
- truncated、true terminal、nonfinite、schema違反の分類が推定器とcompleterの契約として明示される。completerによる初期sourceの決定と、`ReplayPriorityStore`によるsource状態の保持を区別できる。
- 一様ReplayBufferで消費されないcompletion pendingが蓄積する経路をなくす。
- ADR 0010のDQN近似と再訪条件は引き続き有効であり、本ADRは同ADRのtransport表現に関するConsequencesだけを置き換える。
