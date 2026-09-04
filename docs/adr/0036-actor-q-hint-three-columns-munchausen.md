# Actor Qヒントを3列へ拡張し、近似Actor初期優先度をMunchausenターゲットと同型にする

ADR 0010 / ADR 0012のActor Qヒントは `[Q(s,a), max_a Q(s,.)]` の2列で、`DqnInitialPriorityEstimator` はその値から近似Bellmanターゲットを作る。PRD 067でLearner targetへMunchausen項とsoft価値を導入すると、従来のhintでは `per_initial_priority_mode=actor_approx` の初期優先度がLearner targetと構造的に一致しない。

**Actor Qヒントを常時 `[q_sa, state_value, munchausen_term]` の3列へ拡張する**ことを決定する。`state_value` はMunchausen OFFなら従来のmax Q、ONならsoft価値とする。`munchausen_term` はOFFなら0、ONならclip済みscaled log-policyとする。Actorが既存の行動推論で得た全行動scoreから計算し、追加network forwardは行わない。

推定器は `target = target_return + start.munchausen_term` とし、非終端時だけ `discount * h^-1(boot.state_value)` を加え、TBO有効時は完成後のtargetへ `h` を適用する。`ValidateHint` は3列すべてのfiniteを要求する。`DQNActionInfo::WithAction` は行動差し替え時に `q_sa` と `munchausen_term` を再gatherし、`state_value` を維持する。hintを持つ `WithAction` で再gather用のper-action Munchausen値が欠落していれば契約違反とする。

**Actorへ渡す設定は必要値だけを持つ狭い `ActorQHintConfig` とする**。保持するのは `enabled`、`alpha`、`entropy_tau`、`clip_value_min`、`use_tbo`、`tbo_epsilon` だけであり、Learner config全体や `log_policy_mode` は渡さない。Actor側はmodeによらずTrain Actor snapshotのonline scoreを使う近似であり、Learnerと同じnetwork sourceを再現する契約ではない。

通常Q/QRでは同一forwardの平均Qを使う。IQN+UQEでは同一forwardから得るrisk-biased action scoreを使い、分布平均を得るための追加forwardは行わない。これらの系統差は、初回sampling前の順位付けを目的とするADR 0010の近似契約の範囲として扱う。

## Considered Options

- **従来の2列を維持する**: actor_approx時にMunchausen項を表現できず、Learner targetと構造がずれるため棄却する。
- **併用を許可して診断だけ出す**: 構造的なずれを解消しないため棄却する。
- **別PRDへdeferする**: hint carrierは動的幅を運べるため、PRD 067内でproducer、codec、`WithAction`、推定器を一括して新schemaへ移す方が単純である。
- **ActorへLearner config全体を渡す**: Actorが不要な学習方針へ依存し、所有権境界を広げるため棄却する。
- **hintをtarget networkで計算する**: ADR 0010の追加forwardなしという前提に反するため棄却する。
- **ON時だけ3列にする**: schema検証とcodec契約がmode依存になるため棄却し、OFFでもゼロ列を持つK3に固定する。

## Consequences

- `kActorQHintColumnCount` は3となり、旧K2 payloadはschema違反としてfail-fastする。互換aliasや旧schema分岐は持たない。
- ADR 0010 / ADR 0012のcarrier、completer、推定器の責務分離と追加forward禁止は維持する。現行文書の同期範囲はPRD 067に従う。
- `actor_approx + Munchausen OFF` は初期優先度の数値だけを従来と同値にする。K3 transport、ゼロ列生成、一時aux tensorは許容し、命令列やRNGを含む完全不変は保証しない。
- 標準Atariのmax初期化構成はhint経路を使わず、Learner OFFの数値経路・RNG不変契約に従う。RainbowはMunchausenアルゴリズムOFFだが、共通K3 transportは利用し得る。
- `Actor` の構築引数に `ActorQHintConfig` が加わるため、DefaultDQNだけでなくRainbowのActor生成箇所も同じ変更内で更新する。Rainbowは `enabled=false` の狭いconfigを渡し、Munchausen計算を有効化せずK3 schemaへ追従する。
- Actorのper-step追加費用は、hintを生成するactor_approx構成におけるsoft価値計算とper-action Munchausen用auxに限定する。
- `CONTEXT.md` はActor Qヒントをドメイン用語としてだけ定義し、列数やnetwork sourceなどの実装契約は本ADRとPRD 067に置く。
- 再訪条件は、actor_approxとMunchausenの併用でActor/Learner順位相関が非Munchausen構成より明確に低い実測が得られた場合とする。
