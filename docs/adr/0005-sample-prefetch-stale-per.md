# ReplayBuffer サンプリング + H2D を 1ステップ stale で先読みし GPU 学習と overlap する（AsyncLoader 方式）

learner step は GPU が long pole（env とは `learn_future_` で overlap 済み）であることを NSight/Tracy 実測で確認した。その先頭で `ReplayBuffer::Sample`（PER sum-tree 抽選 + 経験 gather, ~6.5ms）と `ExperienceSamples::To`（CPU→GPU の H2D, ~6.5ms）が合わせて ~13ms/step の GPU idle バブルを作る。両者を専用 prefetch スレッド（+ copy stream + pinned + event）に offload し、現バッチの GPU 更新と「次バッチのサンプリング + H2D」を 1ステップ先読みして overlap する。`learner.use_rb_prefetch`（default false）で切替。現在の `true` は、prefetch armed 後の `Push` を同じ FIFO worker へ遅延投入する write-behind insert も含む。

先読みは **`PrefetchingReplayBuffer` decorator**（`ReplayBuffer` を wrap し `ReplayBuffer` インターフェースを提供）に閉じる。責務分担：**replay ordering＝PrefetchingReplayBuffer の 1-deep 順序境界、データ整合性＝ReplayBuffer、public RB インターフェース＝不変**。

## Considered Options

- **現行同期 Sample+To のまま**: learner step 先頭に ~13ms の GPU idle が残る。棄却。
- **prefetch を Learner に埋め込み（初版実装）**: 動くが prefetch ロジックが Learner に散らばり、`Push()` / `UpdatePriorities()` と in-flight prefetch の順序境界も Learner 側に漏れる。棄却（decorator 化対象）。
- **SampleIndices/Extract を内部 split し SampleIndices を caller 同期 / Extract+To を async（A1'、保留）**: tree/RNG に触る `SampleIndices` を caller thread へ戻せるが、`DefaultReplayBuffer` の内部 seam が増え、`SampleIndices` が caller thread に停留する。`SampleIndices` は `Extract+To` と同程度に重いため、production では採用しない。
- **PrefetchingReplayBuffer + mutation-wait（A2'、採用）**: monolithic `Sample`+`To` を bg 先読みし、`Push()` / `UpdatePriorities()` 前に in-flight prefetch を `future.wait()` する。1-deep prefetch と wrapper 経由 mutation を前提に、sample と mutation の順序境界を固定しつつ、`SampleIndices` も background に残す。
- **PrefetchingReplayBuffer + write-behind Push（追加実験、採用中）**: public `ReplayBuffer` IF と config key は増やさず、`learner.use_rb_prefetch=true` の armed 後だけ `Push()` を shallow lifetime snapshot 付きで同じ FIFO worker へ遅延投入する。cold/warmup は同期 `Push` のままにし、`UpdatePriorities()` は次 prefetch future を待つ。

## Consequences

- **replay ordering＝monolithic background `Sample+To` + armed 後 `Push` の write-behind + `UpdatePriorities` 前 `wait()`**。`SampleIndices` は worker thread で実行する。前提条件：(1) Push と UpdatePriorities は必ず PrefetchingReplayBuffer 経由（inner への生 storage/priority write 経路を残さない）、(2) in-flight prefetch は 1 本（1-deep）、(3) runner 側が `state/reward/next_state` を clone 済みの stable `BatchExperience` を渡す、(4) `action_info` は毎 step 新規生成・immutable 扱い、(5) cold/warmup の Push は同期反映、armed 後の Push は shallow copy 後に現在の future の後ろ、次 `Fetch()` の前へ FIFO 投入、(6) `wait()` は prefetch future を消費しない（prefetch 失敗は次 `Sample().get()` で表面化、queued Push 失敗は次の同期境界で回収）。
- **データ整合性＝ReplayBuffer の lock**。`storage_mutex_`（Push=exclusive / Sample=shared）+ `metadata_mutex_`。**Extract はロックフリーではない**（初版 ADR の「リング距離で安全」は誤り ―― 最古 valid スロットは write cursor 直前で次 Push に上書きされ得る）。storage shared/exclusive lock が Push⇄Extract を排他してこれを防ぐ。credit ループ 0 回で先読みが呼び出しを跨ぐケース・metrics accessor 並行も同 lock が守る。
- **現行 serial とは bit 非互換**（厳密比較は `learner.use_rb_prefetch=false`）。ただし **同 seed → 同結果（決定的）** は保つ（`true` の 2 run は sampled index 列が一致。`true` vs `false` は staleness で異なるのが正常）。
- **`ExperienceSamples::To` は ambient stream 尊重**（`getDefaultCUDAStream`→`getCurrentCUDAStream`）。prefetch は copy stream + `non_blocking=true` + `CUDAEvent`。消費側は `event.block` で H2D 完了を保証。**転送元 CPU tensor の pinned 化が必須**（`PinCpuSamples`。非 pinned だと非同期 H2D が同期 fallback し overlap しない）。
- PER サンプリングが **1ステップ stale**。大容量 replay（524288）+ 初期優先度で実害軽微。
- **コスト＝runner 側 stable 化 + caller 側 shallow `Push.snapshot` + `UpdatePriorities` の `future.wait()`**。`SampleIndices+Extract+To` と `Push.push_worker` は worker で実行する。queued Push は現在 consume する batch には見えず、次以降の prefetched batch から見える。実測優位が出ない場合、この write-behind Push は不採用または所有権再設計へ戻す。
- 適用は DQN 系（DefaultDQN/QR、Rainbow 同梱可）。**N-deep prefetch・public `ReplayBuffer` IF 変更・MuZero は対象外**。
- 実装手順：初版 `docs/memo/013_sample_prefetch_10prd.md`、AsyncLoader リファクタ `docs/memo/014_async_loader_prefetch_10prd.md`、N-deep 展望 `docs/memo/999_async_loader_ndeep.md`。
