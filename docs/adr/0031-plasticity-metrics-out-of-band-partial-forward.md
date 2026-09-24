# 可塑性メトリクスは実 forward の branch capture と学習経路外の部分 forward の 2 チャネルで測る

RR ラダー（8/4/2/1）で「損傷 ∝ 勾配 step / 修復 ∝ 新規データ」の機序が決着し、`q_gap` / ULP 余裕という代理指標がスコアの予測子でないと判明したため、表現の健康状態を直接測る srank / dormant unit 率を常設メトリクスとして導入する（062 PRD）。測定対象の penultimate 特徴（慣習名 `main_feature`）は、IQN 構成では `net.body.output` に出ない（`@iqn` プロファイルが `features` を fusion 後へ上書きする。QR / nature では出る）— 構成によって取れたり取れなかったりする口は普遍メトリクスの土台にならない。測定は **capture（GPU）と stats（CPU、GetScalar 時の lazy 計算 + cache）に分離**し、計測の起動は**購読ヒント**（metrics 定義から起動時に集約した購読一覧を agent へ渡す汎用機構）が駆動する — ON/OFF・cadence（行の `interval:N` の最小値）・target 測定の発動（`plasticity_target_*` 行の uncomment）の正が metrics 定義 1 箇所に集約され、購読が無ければ一切計算しない。チャネルは 2 系統: **@learn 系**は**学習 forward が実際に生成した中間特徴を、forward の optional capture 引数で捕捉**する（`Network::Forward` / `NetworkBody::Forward` に既定 nullptr の capture 引数を追加し、body forward 末尾で内部 state から対象 branch を detach して返す。追加 forward なし = capture コストゼロ。train mode の効果込みで「この update が見た表現」と定義が一致し、同じ UpdateResult の loss / q_gap と同一 forward。搭載は非 const 段階の `MakeBatchUpdateResult`）。**$agent probe 系**は RB へ追加する公開 API **`SampleUniqueUniform`**（`DefaultReplayBuffer` が実体 = 既存 lock 境界内で `UniqueUniformSampler` の非復元抽選 + 既存 extractor、`probe.batch_size` ≥ D、index 常に一意、優先度木・`MarkSampledOnce`・eviction 統計に非接触、RB 所有の専用 RNG。`PrefetchingReplayBuffer` は `state_->mutex` の下で write-behind Push と in-flight prefetch を規定順に settle してから inner へ委譲 — probe は「呼び出し時点までに Push 済みの全遷移」を decorator の有無に依らず読み、同 seed で決定的。通常 Sample の順序・学習結果は不変）で一様サンプルを取り、**eval mode の部分 forward** `NetworkBody::ForwardUpTo`（トポロジカルソート済み branch 列から対象の依存閉包だけを実行）で測り、agent 最新値で公開する（srank 天井と PER バイアスを解消する決定的チャネル）。疎・未測定の値は **NaN**（既知 key の「値なし」表現。nullopt は未知 key = typo WARN のまま）。stats は FP32 cast → CPU `svdvals` 固定。hot path への変更は body forward 末尾の capture 分岐 1 個のみ（**capture を渡さない呼び出し — actor 含む — は購読の有無に依らず完全不変**）、probe の追加 forward は NoGrad + eval mode（RNG 不消費）— 測定の有無が学習の数値系列を変えないことを構造で保証する。`GetTensorDictFunction`・TraceCallback は変更しない。

## Considered Options

- **(A) 2 チャネル: 実 forward の capture 引数 + 部分 forward（採用）**: @learn は定義（学習が見た表現）と実装が一致し capture コストゼロ（interval を下げる余地）。capture を渡さない呼び出しは完全不変。probe は依存閉包実行（トポロジカル順は独立 branch 同士の順序を保証しないので閉包フィルタは契約の一部）により IQN でも `taus` 注入不要・無関係 branch の計算ゼロ。branch 名は `feature_key` で利用者が明示し（意味責任は利用側）、未知名は `GetBranchNames()` の一覧付きで fail-fast（参照検証はフレームワーク側）。
- **(A') @learn も eval mode の再 forward で測る（旧案）**: hot path 完全不変だが、「この update が見た表現」という定義と矛盾（train mode の Dropout/BN 効果が再現されない）し、interval ごとに追加 forward コストが乗る。棄却（eval 測定自体は probe 系が担う）。
- **(A'') 既存 TraceCallback へ branch 出力を無条件 emit（旧案）**: actor が `MakeAction` で常に callback を渡すため、plasticity 購読ゼロの構成でも actor の毎 forward に GPU 処理（env0 slice → FP32 → clone）と nn_trace 増加が波及し、「購読ゼロ完全不活性・既存 actor trace 不変」の契約に矛盾。棄却（capture は TraceCallback と独立の専用引数）。
- **(B') 全 branch 実行して内部 state を返す**: IQN 構成でダミー `taus` の注入機構が要り、tau_embedding / fusion の無駄計算が毎測定に乗る。棄却。
- **(C) `net.body.output` に測定 branch を追加マップ**: head 入口の FP32 cast が全 key を舐めるため毎 forward に余分な cast が乗り、`Network::Forward` の戻りは head 出力なので learner からは結局別口が要る。棄却。
- **(D) GetTensorDictFunction の予約 key 経由（t-SNE PRD の 6 行案）**: IQN 構成では branch 内部の `main_feature` に届かず、agent レベル wrapper は learner 内タップには二重正規化になる。棄却（t-SNE 側は将来、同じ `ForwardUpTo` に予約 key を乗せる）。
- **(E) GPU で svdvals**: deterministic 検証課題が残り、この行列サイズでは速度利得も見込み薄。棄却。
- **配線の対抗案**: **UpdateResult 疎 optional（nullopt）** — 観測機構が nullopt を "value not found" WARN として毎件出すため不成立（WARN は typo 検出用に温存し、疎は NaN で表現。先例 = env scalar `accuracy`）。**$agent 最新値のみ** — Update と切り離され「結果を覚えて取り出す」形になり @learn の意味論が崩れる（probe 系にのみ採用）。**完全 pull（GetScalar 時に lazy forward）** — 計算ゲート不在で毎イベント forward・測定重みの曖昧化・共有 net への任意スレッド forward。棄却。**dense carry-forward** — 毎 update 点 + RR 依存 EMA の課題。棄却。**probe の復元抽出（既存 UniformSampler 再利用）** — RB が小さい序盤に重複で実効 N が減り「N ≥ D で天井解消」が崩れる。棄却（非復元 ~25 行で契約を単純に保つ）。

## Consequences

- 測定は学習経路と分離され（hot path 変更は body forward 末尾の `if (capture)` 分岐 1 個のみ、probe は RNG 不消費の eval forward）、購読の有無・cadence の変更が同 seed 再現性・学習結果に影響しない（受入で ON/OFF 等価性を実証する）。
- **metrics 定義が測定の正**: `metrics_scalar.txt` @baseline の plasticity 行（interval:100）が「全 ENV 既定 ON」を構成し、行のコメントアウト（または @min チェーン選択）だけで完全 OFF になる。learner 側の config キーは `learner.plasticity.feature_key` と `learner.plasticity.probe.batch_size` のみ。
- **購読ヒントは汎用機構**として実装し、消費側が関心キーを filter して解釈する。TraceCallback の常時 ON 問題（911 PRD §7）等への将来流用を想定。内容は metrics 定義レコード（CONTEXT.md）と同源。
- 疎メトリクスの「既知 key だが値なし = NaN」契約が確立する。同梱で観測側に非有限ガード 2 行（EMA 更新・非 LEARN 軸平均）を入れ、NaN 先例（accuracy）と `$ema` の組合せで起きる潜在汚染も塞ぐ。
- @learn 系の値は train mode の効果（Dropout mask・BN バッチ統計）を含む — 決定的な健康測定と RR 比較の主読みは probe 系が担う（現行 Atari 既定では両者一致）。capture は専用引数のため、**購読の有無に依らず actor の処理・nn_trace は不変**（単体テストで保証）。
- ReplayBuffer 抽象に `SampleUniqueUniform` が加わり、現用実装（DefaultReplayBuffer / PrefetchingReplayBuffer / テスト double）を同一変更内で実装する（クリーンブレーク方針。黙って throw する既定実装は置かない）。
- 対象は DefaultDQN 系（TD/QR/IQN、共有 learner で自動対応）+ ImageCls（@learn 系のみ。RB が無いため probe 対象外）。Rainbow は `RainbowAgent.net.*` が現行 env config に無く実行検証できないため配線見送り（将来は数行）。
- 将来の保護機構（Spectral Norm / ReDo / reset）導入時の効果測定器になる。ReDo は dormant 率がトリガー指標そのもの。
- ADR 0018（IQN bind 積 DAG）とは整合: capture 引数（body forward 末尾で内部 state を参照するのみ）も `ForwardUpTo` も branch の実行・bind 契約を変えず、DAG 構造・検証責任境界は不変。
- 仕様詳細は `docs/memo/done/062_plasticity_metrics_10prd.md`（決定事項 D1〜D11 + 配線、実装仕様、シーケンス図）。

## Implementation Follow-up (2026-08-27)

- 購読ヒントは `ScalarMetricSubscription`（source key、event、optional target、interval、runner scope、eval 名）として型付きで渡す。`RunManager` は設定全体ではなく、実際に attach した scalar 定義だけを学習開始前に 1 回 Agent へ通知する。
- DQN の online actual、target actual、probe はそれぞれ購読行の最小 interval を cadence として持つ。target 行は既定でコメントアウトし、購読ゼロ時は capture request、probe sample、統計計算を行わない。
- `ForwardUpTo` の依存解決は現行 builder と同じく input key を同名 branch より優先する。bind factor が input spec に存在すれば終端入力であり、同名 branch を ancestor に加えない。
- metrics 定義の新規 `@learn` 行はすべて `$learn_step` を明示し、疎な測定点と cadence の座標系を一致させる。
