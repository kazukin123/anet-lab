# PRD 050: ReplayBuffer ring折り返し時のframe stack sampleability

- 起票日: 2026-08-13
- 状態: implementation ready
- 対象: `core/anet-core` ReplayBuffer（`ValidIndexManager`とそのconsumer）
- 関連: ADR 0024（本PRDで新設）、PRD 035（既知失敗allowlistの出典。履歴として変更しない）、ADR 0012（初期priority完成境界）
- 設計文書: `docs/design/150_replay_buffer.jp.md`

## Context（背景・目的）

`DefaultReplayBuffer`はリング折り返し後、すでに上書きされた過去frameを必要とするtransitionをsample可能として公開する。extractorはsampleされたphysical indexから過去方向へring sliceするため、同じphysical slotへ後から書かれた未来frameを過去frameとして読み、`[t=9, t=10, t=3]`のような時間逆転stackを学習minibatchへ静かに供給する。クラッシュも形状異常も起きないため、学習品質の劣化として観測できない。

直接原因は、sample候補列挙の`ForEachSampleableIndex()`が受け取った`stack_count`を`(void)stack_count;`で明示的に捨てていることである。候補範囲の下限は「未上書きの最古」だけで決まり、「過去stack履歴がまだ残っているか」を誰も検査しない。extractorの境界スキャンはterminalフラグしか見ず、そのフラグ自体が上書き後の新世代のものなので防波堤にならない。

本PRDは範囲概念を**ready range**（未来側確定）と**sampleable range**（ready + wrap後のhistory margin）に分離し、uniform/PER/`Size()`/可視化が同一のsampleable集合を共有することを強制する。paddingの許可は「起動直後・実episode境界」に限定し、ring上書き由来の履歴喪失はpaddingせず候補から除外する。

## 0. 決定一覧（グリル確定値）

| ID | 決定 |
|---|---|
| D1 | 範囲概念を2分離する。**ready range** = 未来側条件（N-step確定・unroll終端確定・未上書き）を満たす論理区間。**sampleable range** = ready rangeにwrap後のhistory marginを適用した最終sample候補区間 |
| D2 | 下限式: `retained_start = max(0, write_cursor - capacity_per_env)` / `history_margin = retained_start > 0 ? stack_count - 1 : 0` / `sample_start = retained_start + history_margin`。marginはenv laneごとに当該laneの`write_cursor`で判定する |
| D3 | wrap前（`retained_start == 0`）はmargin 0。起動直後・episode境界のpadding契約は不変 |
| D4 | `stack_count == 1`では全経路で挙動不変（ring最古の保持transitionを除外しない） |
| D5 | sampleable rangeの利用者: `ForEachSampleableIndex()` / `GetValidIndices1D()`（uniform・PER・可視化accessor・`DumpToLog`）/ `GetSampleableCount()`（`DefaultReplayBuffer::Size()`） |
| D6 | ready rangeの利用者: `InitialPriorityCompleter`。判定APIに`stack_count`は追加しない。初期priority完成に必要なのは未来側確定だけであり、stack marginを混ぜるとcapacityとhorizonの組合せでFIFO先頭が永久に完成しない構成が生まれるためである |
| D7 | eviction統計（`last_evicted_*`）は**ready range基準**とする。追い出されるslotは追い出しの`stack_count - 1` step前にsampleable集合から抜けるため、sampleable基準ではwrap後に統計が恒常0件となりmetricが死ぬ。「margin期間中はsample不可だった」分だけ『sample機会があったのに引かれなかった』件数をわずかに過大評価する近似を含むが、これを許容し明文化する |
| D8 | rename（クリーンブレーク）: `IsLogicalSampleable`→`IsLogicalReady`、`GetLogicalSampleableRange`→`GetLogicalReadyRange`、`LogicalSampleableRange`→`LogicalReadyRange`、`IsOverwritingSampleable`→`IsOverwritingReady`（`stack_count`引数は削除）。「sampleable」の語はstack込みの最終集合だけに使う。metric公開キー（`replaybuffer.per.last_evicted_never_sampled_ratio`等）は不変 |
| D9 | 構築時fail-fast: 既存チェックの`required_capacity_per_env = max(1, n_step) + 1`へ`+ (stack_count - 1)`を加算する。unrollは現行同様チェック対象外（最小差分） |
| D10 | extractor（`ExperienceSampleExtractor`）のAPI・実装は変更しない。安全なindexならring跨ぎstackを正しく組めることはpass済みテスト（`survive ring wrap`）が証明している |
| D11 | dummy契約は不変。dummyはlogical rangeに含め、列挙時にphysical slot単位で除外する（既存単体テストがpin）。history marginは論理幅ベースで計算し、dummyの有無と独立 |
| D12 | `ForEachSampleableIndex()`の物理index昇順列挙の不変条件を維持する（PER側`binary_search`前提） |
| D13 | dead code `GetValidCount()`を削除する（production/testとも呼び出し元ゼロを確認済み） |
| D14 | sample結果へlogical generationを運びextractorでassertする防御強化は対象外。sampler・PER・`IndexSampleResult`・item keyまで影響が広がるため、中央のsampleability不変条件での解決を先行させ、追加検証は別判断とする |
| D15 | `episode_start without done`の既知失敗2件は別契約の問題であり、本PRDに混ぜない。挙動を意図せず変えないことを受け入れ条件に含める |
| D16 | 成果物はPRD 050、ADR 0024、`CONTEXT.md`用語3件（ready range / sampleable range / history margin）、実装時の`docs/design/150_replay_buffer.jp.md`更新。PRD 035のallowlist記載は履歴として変更せず、本PRD完了後の既知失敗は`episode_start`2件のみが新しいbaselineとなる |

## 1. 現状の事実（コード確認済み）

2026-08-13時点、branch `main` / commit `6c1305e`で実測済み。

| 事実 | 根拠 |
|---|---|
| `ForEachSampleableIndex(env, stack_count, unroll_steps, n_step, fn)`は`(void)stack_count;`で引数を捨て、`GetLogicalSampleableRange()`の範囲をそのまま列挙する。物理index昇順（PER `binary_search`前提のコメントあり） | `replay_buffer_impl.hpp:152-178` |
| `GetLogicalSampleableRange()`の下限は`max(0, write_cursor - capacity_per_env_)`のみ。過去stack幅を含めない | `replay_buffer_impl.cpp:252-266` |
| extractorはphysical `time_idx`から過去へring sliceし、`is_episode_boundary`（terminalフラグ）だけを境界として見る。slotの世代は確認しないため、上書き後の新frameを過去frameとして読む | `replay_buffer_impl.cpp:1161-1188` |
| uniform/PERとも`GetValidIndices1D()`の同一集合からsampleしextractorへ渡す | `replay_buffer_impl.cpp:1482-1493` |
| `Size()`は`GetSampleableCount()`で同じ誤った集合を数える | `replay_buffer_impl.cpp:1505-1512` |
| 可視化accessor `GetTensorVector()`と`DumpToLog()`も`GetValidIndices1D()`を使う | `replay_buffer_impl.cpp:1706-1711`, `replay_buffer_impl.cpp:1779` |
| `IsOverwritingSampleable()`も`(void)stack_count;`。上書き直前のslotが範囲内かでeviction統計を判定する | `replay_buffer_impl.cpp:307-321` |
| `IsLogicalSampleable()`は`GetLogicalSampleableRange()`のContains判定のみ。唯一のproduction呼び出し元は`InitialPriorityCompleter::CompleteReady`のFIFO先頭pop条件 | `replay_buffer_impl.cpp:323-329`, `replay_buffer_impl.cpp:361-363` |
| eviction統計の経路: `RecordEvictionIfSampleable`（Push内）→`StoreLastEvictionStats`→`GetScalar`でmetric返却 | `replay_buffer_impl.cpp:1359-1384`, `replay_buffer_impl.cpp:1659-1662` |
| 構築時チェックは`required_capacity_per_env = max(1, n_step) + 1`のみで`stack_count`を見ない | `replay_buffer_impl.cpp:2063-2072` |
| `GetValidCount()`は宣言・定義があるが呼び出し元ゼロのdead code | `replay_buffer_impl.hpp:124`, `replay_buffer_impl.cpp:285-292` |
| RED既知: `Size()`が期待4に対し6 / PERがunsafe physical slot 4をsample / `Size()`が期待5に対し7 | `replay_buffer_test.cpp:2861`, `replay_buffer_test.cpp:2931`, `replay_buffer_test.cpp:2965` |
| pass済み: 安全なindexを選べばextractorはring跨ぎstackを正しく組む / `stack_count`無効時はwrap後最古もsample可能で`Size()==6` | `replay_buffer_test.cpp:2828`, `replay_buffer_test.cpp:2905` |
| `ValidIndexManager`単体テストはconsumer一致（stack=1）とdummy契約（dummyはlogical rangeに含まれ列挙側で除外）をpin | `replay_buffer_test.cpp:1743`, `replay_buffer_test.cpp:1784` |
| `ValidIndexManager`を参照するのは`replay_buffer_impl.{hpp,cpp}`と`replay_buffer_test.cpp`の3ファイルのみ | リポジトリ全体grep |

### 再現例

`num_envs=1, capacity=8, n_step=2, stack_count=3`、logical time `0..10`をPush。保持されるlogical timeは`3..10`、未来側条件を満たす開始点は`3..8`。

| 開始logical time | 必要なobs stack | 判定 |
|---:|---|---|
| 3 | 1, 2, 3 | 1, 2が上書き済み（slotには9, 10が居る）のため不可 |
| 4 | 2, 3, 4 | 2が上書き済みのため不可 |
| 5..8 | すべて保持範囲内 | 可 |

正しいsample可能集合はlogical `5..8` = physical slot `{5, 6, 7, 0}`、`Size() == 4`。現行実装はlogical `3..8`の6件を公開し、開始3をsampleすると概念上`[9, 10, 3]`の時間逆転stackが生成される。

## 2. 契約

### 2.1 範囲の定義

- **ready range**（論理区間、env laneごと）: 未上書きであり、N-stepに必要な未来観測が書き込まれ、unroll終端が確定した区間。現行`GetLogicalSampleableRange()`の実態はこれであり、`GetLogicalReadyRange()`へ改名する。
- **sampleable range**: ready rangeを基礎とし、wrap後（`retained_start > 0`）は下限へ`stack_count - 1`のhistory marginを加算した区間。dummyは現行どおり列挙時にphysical slot単位で除外する（rangeそのものはdummyを含む論理幅で計算する）。

```text
retained_start = max(0, write_cursor - capacity_per_env)
history_margin = retained_start > 0 ? stack_count - 1 : 0
sample_start   = retained_start + history_margin
sample_end     = ready range の上限（変更なし）
```

wrap前はmargin 0なので、起動直後のepisode先頭paddingは従来どおり許可される。ring上書きによる履歴喪失はepisode境界ではないため、paddingしてはならない（除外が唯一の正しい扱い）。

### 2.2 consumerの割り当て

| consumer | 使用する範囲 | 理由 |
|---|---|---|
| `ForEachSampleableIndex()` / `GetValidIndices1D()`（uniform・PER・可視化・`DumpToLog`）/ `GetSampleableCount()`（`Size()`） | sampleable | 学習へ供給し得る集合と、外部へ報告する件数・内容は同一でなければならない |
| `InitialPriorityCompleter`（`IsLogicalReady()`） | ready | 初期priority完成に必要なのは未来側確定のみ。stack marginを混ぜるとFIFO先頭が永久未完成になり得る |
| eviction統計（`IsOverwritingReady()`） | ready | D7。sampleable基準ではwrap後に統計が恒常0件化する |

### 2.3 eviction統計の近似

`IsOverwritingReady()`は「上書きされるslotが未来側条件を満たしたままだったか」を判定する。sampleable集合から見ると、当該slotはhistory margin期間（`stack_count - 1` step）の間すでにsample不可だったため、`last_evicted_never_sampled_ratio`は『sample機会があったのに引かれなかった』件数を厳密解よりわずかに過大に数える。margin幅（数step）はcapacity_per_env（数千step）に対して十分小さく、metricの用途（PER監視の傾向把握）に対して許容する。この近似は`docs/design/150_replay_buffer.jp.md` 7.5（可観測性）へ明記する。

### 2.4 構築時fail-fast

```text
required_capacity_per_env = max(1, n_step) + 1 + (stack_count - 1)
```

`capacity_per_env < required_capacity_per_env`なら既存の`ANET_SYSTEM_ERROR`経路で構築時に停止する（`replay_buffer_impl.cpp:2063-2072`の式へ加算）。現行実構成（例: DropMerge A構成 2048/env, stack 4）は余裕で通る。unrollを含めた完全な下限保証は現行チェック同様スコープ外とする。

### 2.5 命名（クリーンブレーク）

| 旧 | 新 | 備考 |
|---|---|---|
| `GetLogicalSampleableRange(env, unroll, n_step)` | `GetLogicalReadyRange(env, unroll, n_step)` | private。意味は不変 |
| `LogicalSampleableRange` | `LogicalReadyRange` | struct名のみ |
| `IsLogicalSampleable(env, logical, unroll, n_step)` | `IsLogicalReady(env, logical, unroll, n_step)` | シグネチャ不変。Completerが使用 |
| `IsOverwritingSampleable(env, time, stack, unroll, n_step)` | `IsOverwritingReady(env, time, unroll, n_step)` | ready基準になるため`stack_count`引数を削除 |
| `ForEachSampleableIndex` / `GetValidIndices1D` / `GetSampleableCount` | 名前不変 | `stack_count`を実際に使うようになり名前が真になる |

AGENTS.mdのクリーンブレーク方針に従い、テスト・設計文書の参照も同一変更内で移行する。旧名の別名・互換ラッパは残さない。

## 3. 実装範囲

### 3.1 変更

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/src/replay_buffer_impl.hpp` | rename（2.5表）。`ForEachSampleableIndex()`の`(void)stack_count;`を廃し、`GetLogicalReadyRange()`の結果へhistory margin（2.1式）を適用してから列挙する。昇順列挙・dummy除外の構造は維持。`GetValidCount()`を削除 |
| `core/anet-core/src/replay_buffer_impl.cpp` | rename追随。`IsOverwritingReady()`から`(void)stack_count;`と引数を削除。構築時チェックへ`+ (stack_count - 1)`（2.4式）。`GetValidCount()`定義削除 |
| `core/anet-core/src/replay_buffer_test.cpp` | rename追随（`1743`・`1784`の単体テスト含む）。追加テスト（4.3）を新設。既存テストの期待値は変更しない |

### 3.2 文書（実装と同一変更内で更新）

| ファイル | 変更内容 |
|---|---|
| `docs/design/150_replay_buffer.jp.md` | 2.2の`Size()`行とValidIndexManager段落（L43, L52）、2.3のextractor padding記述（L59）、3.コンポーネント表のValidIndexManager行（L92）を ready/sampleable分離契約へ書き換え。「起動直後・実episode境界はpadding可、ring上書き由来の欠損は除外」を明記。7.5へeviction統計の近似（2.3）を追記。§5-6のMermaidに旧名が残る場合は追随 |
| `CONTEXT.md` | 「Replay・PER」カテゴリへ用語3件を追加（本PRD作成時に実施済み: ready range / sampleable range / history margin） |
| `docs/adr/0024-replay-sampleable-range-excludes-overwritten-stack-history.md` | 本PRD作成時に新設済み |

## 4. 受け入れ基準

### 4.1 RED既知テストの緑化

- `ReplayBuffer excludes wrapped samples whose frame stack would read overwritten frames`（`Size()==4`、`GetTensorVector(STATE_OBS)`がlogical 5..8の4件、最古安全index=logical 5のobs/next_obs/target_return検証）
- `ReplayBuffer PER samples only safe wrapped frame-stack indices`（PER経路でphysical slot `{5,6,7,0}`のみ）
- `ReplayBuffer wrapped sampleability honors both frame stack and unroll horizons`（`Size()==5`、stack下限とunroll上限の同時成立）

### 4.2 非退行

- `ReplayBuffer frame stacking and n-step next_obs survive ring wrap`（安全indexのring跨ぎ組み立て）
- `ReplayBuffer keeps wrapped oldest samples sampleable when frame stack is disabled`（`stack_count==1`でwrap後`Size()==6`・最古sample可）
- `ValidIndexManager sampleability consumers agree before and after wrap` / `ValidIndexManager keeps dummy filtering outside the shared logical range`（rename追随のみ、アサーション不変）
- 初期padding系（`pads the beginning of an episode` / `pads the initial sample for nonzero env values` / `does not cross done boundaries per env`）、`stack_keys leaves non-stacked observations at latest frame`、`InitialPriorityCompleter`系全テスト、`ReplayBuffer PER tracks last evicted sampleable slots that were never sampled`
- `episode_start without done`の2件（`n-step returns stop at ...` / `frame stacking starts a new stack at ...`）は挙動変更なし＝引き続き既知失敗のままでよい。この2件以外のallowlist外失敗が0件であること

### 4.3 追加テスト

1. `ValidIndexManager`単体: `stack_count > 1`かつwrap後に`GetValidIndices1D` / `GetSampleableCount`が一致し、margin分だけ最古が除外されること
2. multi-env: laneごとに`write_cursor`が異なる（片方だけwrap済み）場合、marginがwrap済みlaneにのみ適用されること
3. margin近傍にdummy（truncated終端）がある場合の期待集合の固定（dummy除外とmarginの適用順で結果が変わらないこと）
4. eviction統計がwrap後も生存すること: `stack_count > 1`のwrap進行中に`last_evicted_never_sampled_ratio`が0/NaN固定にならず、未sample追い出しを数えること（ready基準の検証）
5. 構築時fail-fast境界値: `capacity_per_env == required - 1`で構築失敗、`== required`で成功

### 4.4 実行

```bash
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]" -r compact
```

ビルドはAGENTS.md記載の`VsDevCmd.bat`経由。Debug全テストで`episode_start`2件以外の失敗0件を確認する。

## 5. スコープ外

- `episode_start without done`のstack境界契約（別問題。現行設計書はproduction保証外、失敗中テストは保証を要求しており、契約自体の裁定が別途必要）
- sample結果へのgeneration運搬とextractor側assert（D14。本修正で不変条件が中央で保証された後に必要性を再評価）
- PER samplerの再設計、RBのframe stack保存方式変更（sample時再構成の設計自体は維持）
- unrollを含む完全なcapacity下限保証（現行チェック同様の限定を維持）

## 6. Further Notes

- 「sampleable」の語は現行コードで2義（未来側確定 / 最終sample候補）に使われており、これが`stack_count`無視の温床だった。本PRDでready/sampleableへ分離し、`CONTEXT.md`で用語を固定する。
- 引継ぎ検討時の案では eviction統計もsampleable range基準としていたが、追い出されるslotは定義上必ずhistory margin通過後に追い出されるため、その案ではwrap後に統計が構造的に0件となりmetricが死ぬことが実装調査で判明した。D7のready基準はこの帰結を避けるための裁定である。
- 学習影響の定量見積もり（DropMerge A構成で汚染はminibatch要素の約0.15%、Run間ブレ幅に埋没）から、修正前後の比較Runで性能有意差が出ないことは想定内である。本修正の正当化は契約の正しさ（誤ったexperienceを供給しない・`Size()`契約・統計の正確性）に置く。
