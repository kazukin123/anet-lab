# Actor network方針の既定値解決をAgentへ戻す

> **後発決定（2026-11-19）**: [ADR 0038](0038-actor-config-catalog-without-runmode.md) により、`CreateActor` の `clone_model_override` 引数は廃止され、現在は `ActorRequest` を受け取る。clone の要否はカタログ項目の `clone_model` で決まる。Learner が更新する source/shared Network は Agent が、private clone/snapshot は Actor が所有する。

Runnerが全Agentのclone方針を必須`bool`で決めると、Agent固有の能力とRunMode別既定をRunnerが所有する責務逆転になるため、`Agent::CreateActor()`は`std::optional<bool> clone_model_override`を受け取り、Train Runnerは`std::nullopt`を渡してAgentに既定値解決を委ねる。明示overrideの解釈は各Agentの既存契約を維持し、clone非対応Agentへ共通のfail-fastを追加せず、MuZeroは引き続きshared networkを使用する。Learnerが更新するsource/shared networkはAgentが所有し、そこから派生して単一Actorだけがforwardと同期に使うprivate clone/snapshotはActorが所有する。
