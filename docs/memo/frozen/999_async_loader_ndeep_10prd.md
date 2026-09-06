# 将来メモ：AsyncLoader の N-deep 化（AsyncDataLoader）と split の再来

> 凍結中(再開条件: 1-deep prefetch が律速になる実測が出たら)

DI-engine の `AsyncDataLoader` 相当 ―― **prefetch 深さ N > 1（バッチを N 本先読み）+ キュー化** ―― を将来やる時の設計メモ。現行の `PrefetchingReplayBuffer`（`docs/memo/014_async_loader_prefetch_10prd.md`、ADR 0005）は **1-deep + A2'（monolithic background `Sample+To`）** であり、現在の遅延 Push は同じ FIFO worker へ `Push` を write-behind する別実験である。現行実装は internal plan/fetch split ではないため、N-deep にする時は新しい plan/fetch seam を設計し直す。

## なぜ wait だけでは N-deep にスケールしないか

A2' のような monolithic background 方式は「`UpdatePriorities()` が in-flight prefetch を `wait()` して `SampleIndices(bg) < UpdatePriorities` を固定する」発想だが、N-deep では破綻が明確になる。armed 後 `Push` を FIFO write-behind しても、深さ N の `SampleIndices` 群を決定的な program point に戻せるわけではない。

N-deep にすると in-flight prefetch が N 本（batch j+1 … j+N）あり、各 `SampleIndices` が別々の時点で tree を読む。`UpdatePriorities(j)` の決定的順序を保つには：

- **N 本全部を `wait()` する** → pipeline が直列化して N-deep の意味（深い overlap）が消える。
- **一部だけ wait** → 残りの `SampleIndices` が `UpdatePriorities` と race → 非決定。

つまり wait 方式は深さ N で「全待ち＝直列化」か「一部待ち＝非決定」の二択になり破綻する。

## なぜ split（A1）は N-deep に強いか

split＝**`SampleIndices` を caller thread で同期実行 / `Extract`+`To` を async**。現行 1-deep はこの seam を持っていないため、N-deep 化ではまず narrow な plan/fetch seam を追加する必要がある。

- tree を触るのは **caller thread の `SampleIndices` / `Push` / `UpdatePriorities` だけ**（すべて program 順）→ 深さ N に関係なく**決定的**。
- async なのは **tree を触らない `Extract`+`To`** のみ（N 本並行で OK。storage 読みは storage lock で Push と整合）。
- `UpdatePriorities` の wait が不要になる（`SampleIndices` は既に caller で同期済み）。

→ **N-deep prefetch + キュー**では split が自然な設計。「plan（SampleIndices, 同期, program 順）をキューに積み、fetch（Extract+To, async）をワーカープールで N 本並行消費」という形に乗る。

## 移行時にやること（チェックリスト）

1. **plan/fetch seam を追加して深さ N に拡張**：`ReplayBuffer` の public インターフェースは monolithic のまま、将来追加する `PlanSample()` + `ExtractPlannedSamples()` 相当の narrow channel を N 本のキューで扱えるようにする。**learner には出さない**（呼び出し順序を強制しない）。
2. **AsyncLoader を depth=N の plan キュー + fetch ワーカープールに拡張**。`Sample()` は plan（同期）→ fetch（async）を投入、消費は完成キューから。
3. **`UpdatePriorities`-wait を撤去**（split で不要）。整合性は storage lock のまま。
4. **再現性テストを N-deep でも**：同 seed 2 run の sampled index 列一致。
5. stale 深さが N に応じて深くなる（新規経験のサンプル遅延が増える）点を学習側で許容できるか確認。

## 現 AsyncLoader からの拡張方針

014/ADR 0005 の現行実装は「plan → fetch」構造ではなく、monolithic `Sample+To` を 1-deep で先読みする。N-deep 化では、fetch ワーカーの多重化、完成キュー、stale 深さの明示的管理を足す前に、`SampleIndices` を同期 plan として切り出す設計判断が必要になる。

## 参考

- DI-engine `AsyncDataLoader`（背景ワーカー + pin_memory + キューで sampling/collate/転送を学習と overlap、parallel モードは完全非同期）。torchrl `ReplayBuffer(prefetch=N)`。Acme/Reverb（replay をサービス化 + prefetching dataset）。
- 決定性 vs split の議論経緯は ADR 0005 の Considered Options。「ロック＝相互排他（整合性）≠ 順序の決定（再現性）」「再現性には順序固定＝SampleIndices を固定 program point で」。
