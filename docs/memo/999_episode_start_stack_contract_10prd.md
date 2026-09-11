# `episode_start` を stack / n-step 境界とするかの契約裁定 暫定 PRD

> 状態: 暫定メモ。決定事項 D1〜D7 は未確定。詳細は別途グリルで詰める。本 PRD は実装着手を意味しない。
> 起点: `done` / `truncated` を伴わない `episode_start` の扱いについて、**設計書とテストが逆のことを主張している**。
> どちらが正かの裁定が行われていないため、失敗テスト 2 件が恒久的な「既知失敗」として残り続けている。
> 関連: `done/050_replay_ring_stack_margin_10prd.md`（D15 / D16 / 非目標節で本件を明示的に除外し、
> 本件 2 件を新しい baseline と定めた）、`docs/design/150_replay_buffer.jp.md`。

## Context（背景・目的）

### 対立している 2 つの契約

同じ設計書の 3 行の中で、**同じ境界が逆に扱われている**。

| 主体 | `done` / `truncated` を伴わない `episode_start` の扱い | 位置 |
|---|---|---|
| **Actor**（`StackerActionContext`） | **境界として扱う。** 受けた lane を初期 frame で埋め直す | `docs/design/150_replay_buffer.jp.md:58` |
| **ReplayBuffer**（`DefaultReplayBuffer::Push`） | **境界として扱わない。** `episode_start` を保存も参照もしない | `docs/design/150_replay_buffer.jp.md:61` |

設計書は後者を「現行 production 経路の保証に含めない」と明記している。一方、失敗中のテスト 2 件は
保証を要求している。**行動選択時のスタックと学習サンプルのスタックで、同じエピソード境界の意味が
食い違っている**ことになる。この非対称が意図的なのか事故なのかが、本 PRD の中心的な問いである。

### 判断が要る理由

現状の実害は確認されていない。ただし「既知失敗 2 件」を放置する運用コストが継続的に発生している。

- 全テスト実行のたびに `501/503` となり、**新規の失敗が既知 2 件の陰に隠れうる**。
- 2026-08-28 の PRD062 実装時、Claude と Codex の両方が独立に「この 2 件は今回の変更で壊れたのか」を
  調査した。同じ調査が繰り返されている。

現在は `[!shouldfail]` を付けてスイートを緑にし、テストコード自体は再現可能な形で保持している
（`replay_buffer_test.cpp:2963` / `:3343`、それぞれ直前に未裁定である旨のコメントあり）。
これは裁定までの暫定措置であり、**タグが付いていること自体は契約を決めたことにならない**。

## 現行コードで確定している事実（実装の下地）

| # | 事実 | 位置 |
|---|---|---|
| 1 | `BatchState` は `done` / `truncated` / `episode_start` の 3 フラグを持つ | `core/anet-core/include/anet/rl.hpp:317`、`:410` |
| 2 | `DefaultReplayBuffer::Push` は `episode_start` を**保存も参照もしない**。境界は `done` / `truncated` のみ | `docs/design/150_replay_buffer.jp.md:61` |
| 3 | Actor 側 `StackerActionContext` は `episode_start` で lane の stack を埋め直す | `docs/design/150_replay_buffer.jp.md:58` |
| 4 | 失敗テスト 2 件は `[!shouldfail]` 済み。削除されていない | `replay_buffer_test.cpp:2963`、`:3343` |
| 5 | PRD 050 は本件を「別契約の問題」として明示的に除外し、完了後の既知失敗は本件 2 件のみが新 baseline と定めた | `done/050_replay_ring_stack_margin_10prd.md` D15 / D16 |
| 6 | PRD 050 非目標節に「現行設計書は production 保証外、失敗中テストは保証を要求しており、**契約自体の裁定が別途必要**」と記載 | 同 非目標節 |

### 失敗テストが要求している具体的挙動

**n-step**（`n_step=3`、`gamma=0.5`、push: `t0(start) t1 t2 t3(start) t4 t5`）

`t1` を sample したとき、`r1 + γ·r2` / `done=true` / `n=2` を要求する。すなわち `t3` の
`episode_start` で **n-step の積算を打ち切る**。現行は打ち切らず `t3` へまたぐ。

**frame stack**（`stack_count=3`、push: `t0(start) t1 t2 t3(start) t4`）

`t3` を sample したとき、`obs = [s3, s3, s3]` を要求する。すなわち `t3` で**新しい stack を開始**し、
`t2` / `t1` へ遡らない。現行は遡る。

どちらも「`episode_start` は `done` と同格の境界である」という 1 つの契約の 2 側面である。

## 案（グリルで選択）

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: テストを正とする** | `Push` が `episode_start` を保存し、n-step / frame stack の境界に含める | Actor と RB で境界の意味が一致する。truncation 系の env で静かな汚染が起きない | `Push` hot path とストレージに影響。serialize 互換の検討が要る |
| B: 設計書を正とする | 境界は `done` / `truncated` のみと確定し、テスト 2 件を削除または契約反転で書き直す | 実装変更ゼロ。既知失敗が消える | Actor との非対称が残る。将来 `episode_start` だけで reset する env を足したとき静かに壊れる |
| C: 現状維持 | 裁定せず `[!shouldfail]` のまま置く | 判断コストゼロ | 運用コストが継続。契約が未定義のまま新規 env が増える |

**現時点の傾き: 案 A**（テストを正にしたい）。ただし**費用対効果が未評価**のため未判断。
D2 の調査結果が判断材料になる。

## 決定事項（未確定）

| # | 論点 | メモ |
|---|---|---|
| D1 | **どちらの契約を正とするか** | 案 A / B / C。D2 の結果に依存する |
| D2 | **`done` / `truncated` を伴わない `episode_start` は production 経路で発生するか** | **これが費用対効果の中心。** 発生しないなら案 A の価値は将来の保険のみ。候補: Atari の `episodic_life` soft-reset（`done/051_atari_ale_env_10prd.md`）、time-limit truncation、run 開始時の初回 reset（これは遡る先が無いので無害）。**各 env / wrapper が reset 時に 3 フラグをどう立てるかの実測が要る** |
| D3 | 案 A の場合の保存方法 | `episode_start` を per-slot に持つか、`done` / `truncated` と統合した 1 つの境界フラグに畳むか。`Push` は hot path なので追加コストの実測が要る |
| D4 | 案 A の場合の serialize 互換 | 既存の `agent_close.anet` / RB スナップショットとの互換をどうするか。クリーンブレークで良いか |
| D5 | 案 B の場合のテスト処置 | 削除するか、契約を反転して「またぐことが正しい」と書き直すか。後者なら回帰ガードとして残る |
| D6 | **Actor と RB の非対称をどう扱うか** | 案 B を採るなら、Actor 側だけが `episode_start` を尊重している理由を設計書に明記する必要がある。現状は理由が書かれていない |
| D7 | 実施タイミング | 現状実害が確認されていないため優先度は低い。可塑性 / 保護機構の系列とは独立 |

## 受入基準（案）

案 A を採る場合。

1. **失敗中の 2 件が `[!shouldfail]` 無しで緑になる。** タグとコメントを除去する。
2. **既存の緑テストが全て緑のまま。** 特に初期 padding 系（`pads the beginning of an episode` /
   `pads the initial sample for nonzero env values` / `does not cross done boundaries per env`）と
   `survive ring wrap` の挙動を変えない。
3. **全テストで allowlist 外の失敗が 0 件。** 本件の解消により既知失敗が無くなる。
4. `Push` の throughput 劣化がノイズフロア内であること（測定は round-robin。`project_perf_ab_machine_drift` 参照）。
5. 設計書 `150_replay_buffer.jp.md` の当該記述を新契約へ同期し、Actor / RB の対称性を明記する。

案 B を採る場合。

1. テスト 2 件を D5 の決定どおり処置し、全テストが緑。
2. 設計書に「なぜ Actor だけが `episode_start` を尊重するのか」の理由を追記（D6）。

## 非目標

- **PRD 050 の再訪。** ready / sampleable 分離と history margin は完了済みで、本件とは別契約である。
- 初期 padding の挙動変更。エピソード先頭を先頭 frame で埋める既存契約は本 PRD の対象外。
- `ExperienceSampleExtractor` の API 変更（PRD 050 D10 の決定を踏襲）。
- `truncated` の意味論そのものの見直し。本 PRD は `episode_start` 単独のケースだけを扱う。
