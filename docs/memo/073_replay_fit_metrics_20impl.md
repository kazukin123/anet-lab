# PRD 073: replay 当てはまり診断 実装メモ

## 概要

2026-09-10 の合意済み計画。DefaultDQN の TD・QR・IQN に、未抽選群 / 抽選済み群の TD・損失・母数・平均年齢と PER 選択比の13指標を追加する。学習系列を維持し、購読された出力に必要な処理だけを実行する。
正本仕様は [PRD 073](073_replay_fit_metrics_10prd.md)。ユーザーの命名指定により target 生成関数は `MakeTarget` とし、`Build` は使わない。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidence: sampleable 集合、sampled_once_、storage/metadata lock、既存 extractor と generation-aware key を利用できる。Prefetching の probe は通常 batch を消費せず FIFO を settle する。named seed は名前と Agent seed から独立に導出される。
- 合意した変更: IQN の taus は生成済み Tensor を前倒しして渡さず、生成関数を渡す。MakeTarget が従来の生成位置で呼び、学習側は既存 RNG、診断側は固定 midpoint を返す。
- 計画に固定: policy 内部の autocast と補助 full query も診断条件で制御する。診断は capture を更新しない。既存コードにある正規化の Normalize は学習統計を更新しない。
- 範囲外: 長期比較実験、他 Agent への機能展開、held-out 分割、既存の無関係な未コミット変更。

## 主な変更

### ReplayBuffer と資源

- PRD §7.1 の SamplingHistoryProbeRequest / SamplingHistoryProbeResult と pure virtual ProbeSamplingHistory を追加し、Default、Prefetching、現用 test double 3種を同時に対応する。
- storage/metadata の同一 snapshot で sampleable 集合を走査し、既存 sampled_once_ で群分けする。母数と lane write cursor 基準の平均年齢を求め、要求された群だけを一様・非復元抽出する。候補保持は要求群に限定する。
- 件数不足は群ごとに nullopt。不正件数と抽出要求時の RNG 不在は fail-fast。IS weight は1。通常の抽選履歴・priority・RNGを変更しない。
- Prefetching は通常 batch を保持したまま既存 probe と同じ待機順で inner に委譲する。
- 外側 Agent は群抽出を要する購読に限り replay_fit_probe named RNG を生成する。Learner は非所有参照を受け取り、要求・一時入力・結果の State を所有する。全 OFF では専用 RNG・作業領域・結果 pack を生成しない。

### 購読と結果

- replay_fit.probe.batch_size=1024、replay_fit.iqn.num_taus=32 を追加し正整数を検証する。
- 独立 metrics.scalar.@replay_fit に PRD の13行を interval503 で定義する。既定選択へ追加せず、enabled/interval の二重設定や旧名 alias を作らない。
- 行ごとの IntervalGate で当該 update の要求を合成する。counts、U、S、実PER、TD、loss の依存を分離し、必要な共有 forward を1回だけ行う。
- 学習無効時は測定しない。学習有効かつ解決済み target が ThompsonSampling の場合、購読設定時に fail-fast。
- 結果は update ごとの BatchUpdateResult に保存し、未購読・非測定回・件数不足・分母ゼロは依存する値だけ NaN、未知 key だけ nullopt。GetScalar は保存結果を読むだけにする。

### 評価と学習経路

- 各 Learner の target 組立を MakeTarget、サンプル別誤差を ComputeElementError へ共通化し、学習・診断の双方が呼ぶ。batch 次元は入力 Tensor から取得する。
- IQN の taus 生成関数を、従来と同じ位置・順序で呼ぶ。target 行動選択、target forward、Munchausen fresh online forward、capture の学習側順序を維持する。
- 符号付き TD、IS 重み、loss の sum/mean、clip、既存診断を学習側で維持する。診断は要求された絶対 TD/loss だけを計算し、不要な全ペア損失・学習用補助統計を生成しない。
- SelectAction に任意の診断評価指定を追加する。指定時は policy 内でも FP32、IQN Greedy は全範囲固定 taus、hard UQE tail は現在 risk 区間の固定 taus を使う。point UQE、QR、soft UQE の既存スコア定義を維持する。補助 full query と不要な policy 診断を停止し、通常呼び出しの挙動は維持する。
- 既存 probe の後、UpdateFromSamples の前に NoGrad/eval/FP32 で測定する。学習 capture と正規化統計を変更せず、mode は例外時も復元する。optimizer、priority 更新、target 同期を診断から呼ばない。
- 実 PER batch は受領済みサンプルを使う。群平均、母数加重全体平均、平均の比は PRD の式を維持する。分割評価は総件数を維持し件数で加重する。
- 抽出・転送・forward・損失・集約に既存規約の ProfileRange を設ける。

## テスト

Public surface: ReplayBuffer の Push/Sample/ProbeSamplingHistory/UpdatePriorities、Agent の ConfigData/ConfigureScalarMetricSubscriptions/UpdateFromBatch、BatchUpdateResult::GetScalar、ActionPolicy::SelectAction、ObserverFactory の profile 解決。

本体変更前に固定 seed、実効設定、ビルド条件と旧実装の基準結果を保存する。test fixture/記録コードで通常抽選、parameter/buffer、RNG、priority、既存 metrics を比較可能にする。

各挙動を1つずつ RED→GREEN とし、GREEN 後に整理する。

1. tracer bullet: 母数購読から ReplayBuffer 読取、BatchUpdateResult::GetScalar まで通し、追加 forward 0 を確認する。
2. 群分け、年齢、非復元抽出: ring、世代、history margin、dummy、n-step、terminal/truncated、片群不足、prefetch。
3. TD→QR→IQN: 小規模 oracle で平均・比・母数加重、Munchausen 各 mode、TBO、Greedy/UQE、許容 Double DQN を検証する。
4. PRD §6.2 全行の購読テスト: 未選択 profile、片群、比のみ、異なる interval、PER無効、学習無効。boundary counter と測定記録で不要処理0を検証し、本体へ test-only API を追加しない。
5. 複数 update の結果保持、NaN/未知 key、設定検証、13行の ObserverFactory 配線。
6. 旧実装/新実装OFF、同一ビルドOFF/ON: BF16、DropPath、BatchNorm/Spectral Normalization、plasticity/policy churn 同時購読を含め、通常系列への非干渉を検証する。

## 検証と完了条件

- VsDevCmd.bat 経由で Debug build と DQN/ReplayBuffer/ObserverFactory 回帰テストを実行する。基準欠落や未実行項目を合格扱いしない。
- 性能は x64-RelWithDebInfo、Breakout RR4、各群1024件、IQN32、interval503、13指標。PRD参照のRR4構成を基準に他条件を揃える。
- seed73/74/75 の3組を OFF→ON、ON→OFF、OFF→ON の順で比較する。初期予算は各2M exp step。warmup後の共通区間で全指標成立20回未満なら予算を延長して再取得する。
- 1-throughput_ON/throughput_OFF の中央値 <=0.05。件数・頻度・精度・閾値を緩めない。実効設定、成立回数、実時間、throughput、主要 ProfileRange を保存する。
- PRD §7.2 / ADR0039 に MakeTarget と生成関数渡しの合意を反映し、ReplayBuffer/DQN設計を更新する。CONTEXTの既存定義は維持する。
- 実行した検証と残件を本書に追記する。過去artifactを変更せず、staging/commit/pushは行わない。

## 実装・検証記録（2026-09-10）

実装済み: 抽選履歴probe API、群別抽出、13指標の購読制御、Agent所有の遅延生成RNG、MakeTarget/ComputeElementErrorの共通化、固定分位点・eval/NoGrad/FP32評価、独立profile。機能・非干渉の検証を完了し、後述の並行稼働条件で性能の数値基準も満たした。単独稼働時の追加コストは未測定。

### 機能と非干渉

- RED→GREEN: 母数の縦断、履歴別抽出、実PER batchの再評価、profile配線。証跡は `.scratch/prd073/{tracer_red,tracer_green,groups_red,groups_green,per_red,per_green_v2,profile_red,profile_green_v2}.*`。
- 個別購読、PER無効、未購読、interval未到達、保存結果の独立性、片群要求のforward数、母数加重と比を検証した。ring世代更新・frame stack/n-step復元・prefetch FIFO、CPU/CUDAのPolicy内AMP抑制、固定risk taus、設定不正値・解決済みThompson制約も対象にした。
- TD/QR/IQN × Munchausenのoff/target/online/online_reuse × TBO × Greedy/UQEの80構成で、旧実装と共通化後の保存Tensor・RNG・forward記録が全バイト一致（`before_v4/`、`after_refactor_v2/`）。BF16・DropPath・BatchNormを含むON/OFF非干渉は別テストと実Runで検証した。
- DQN/ReplayBuffer/ObserverFactory回帰: 252ケース、250成功・既存の想定失敗2、終了コード0（`regression.test.json`）。既存Munchausen profileテストは、現行設定でコメントアウト済みのsoft_gapを要求していたため、既定OFFを確認する現行契約へ修正した。設定自体は維持した。

### 固定seedの実Run

seed73、Breakout RR4/capall/Munchausen、deterministic、warmup200K、終了400K。次の3 Runはいずれも終了コード0でagent_close.anetを保存した。

- 旧実装: `run_20260910-092437_prd073_before_v4`
- 新実装OFF: `run_20260910-093210_prd073_after_off`
- 新実装ON: `run_20260910-093724_prd073_after_on`

出力先は `.scratch/prd073/workspace/runs/`。旧実行体SHA256は `4F297F4E9D4E4988C778BCE26C2E54B379025A3F4BE150C325574ADD1AE85AD2`。
旧/新OFF、OFF/ONの両比較でonline/target archiveの各258 entryが一致し、optimizerもparam group順へ対応付けた全state/optionsが一致した。optimizerの生archive差はプロセス固有addressと格納順に由来するため、そのままchecksum成功とは扱わず対応付け後の内容を比較した（`*.canonical.json`）。
既存メトリクスはplasticity・policy churnを含む107系列・79,182点が両比較で完全一致（`before_after_off.metrics.json`、`off_on.metrics.json`）。
ONは実効設定で13行・各群1024件・IQN32・interval503を確認し、全13指標が成立した回は6回だった。性能受入の20回条件はこの短い非干渉Runでは判定しない。

初回基準の終了指定がprofile材料側に入り実効50Mのままだった試行は、自分のprocessを停止して不採用とした。次の試行は正常保存できたが終了コード取得に欠落があり、採用基準には使わず再取得した。別ディレクトリからの旧exe起動も初期化に進まず停止し、元のDLL/resource配置内に旧exeを置いた上記before_v4を採用した。失敗証跡は削除していない。

### 最終ビルドと数値oracle

- VsDevCmd経由のDebug全体ビルドと、性能測定に使うRelWithDebInfo Runnerビルドは終了コード0（`debug_all.build.log`、`runner_final.build.log`）。
- 最終実装でも旧実装の80構成と保存結果が全バイト一致（`before_v4/` 対 `after_final/`）。
- 独立した固定分布oracleはTD/QR/IQNとDouble DQNの両設定で48 assertion成功（`oracle_v2.test.log`）。初回のoracleはIQNの同点時のonline行動選択を取り違えていたため、Double DQNの有無を式に反映して再検証した。実装側の式や許容誤差は緩めていない。

### ProfileRange

Nsight Systems 2026.3.1でNVTXだけを記録した260Kの短いRunは、OFF/ONとも終了コード0。CPU samplingとcontext switchの記録は無効。以下は同期待ちを含むhost elapsed時間であり、GPU kernel単体の時間ではない。

- OFFではReplayFit/ProbeSamplingHistoryのrangeが0回（`trace_off.summary.json`）。
- ONのCaptureReplayFitは2回、平均201.241 ms。最初の回は抽選済み群が不足しており、二群と実PERの評価phaseは合計5回だった。
- probeの走査は平均1.852 ms、抽出対象の選択0.622 ms、Tensor復元42.035 ms。EvaluateReplayFitの転送は平均3.952 ms、forward 27.115 ms、誤差0.493 ms、集約30.691 ms。集約はCPUへの結果取得時の同期も含む。
- 根拠は `trace_on.summary.json` と元の `.nsys-rep` / `.sqlite`。この短いprofile Runを性能受入の3組比較や20回成立の代用にはしない。

### 性能測定の並行稼働条件

最終Runの実行中、ユーザーから「別途もともとRunが走っている中でPRD073検証Runを同時に実行している」と共有された。元のRunは停止せず、予定した6本を継続する。こちらから追加のビルド・テストは重ねていないが、元のRunの時間ごとの負荷は固定管理していない。
したがって性能値はこの並行稼働条件でのON/OFF比較として記録し、単独稼働時の追加コストの証明とは扱わない。5%という判定閾値・3組・同一step区間・20回以上の成立条件は維持する。機能、数値oracle、保存結果の一致、全13指標の成立は、それぞれの証拠から独立に判定する。

### 性能の結果

6本すべてが2,000,000 exp stepまで進み、終了コード0。実行順はseed73 OFF→ON、seed74 ON→OFF、seed75 OFF→ON。x64-RelWithDebInfo、Breakout、RR4、replay容量1,048,576、通常batch256、各群1024件、IQN32、interval503、13指標を維持した。
実行体SHA256は全6本で `2A0B203112557A2BAECEF7D70D72B53782E52F791E4F307531861B65719C1990`。seedを除くAgent/backend/環境/train設定と、追加13行を除く既存metrics行が一致した。実効backendはTF32有効、cuDNN benchmark有効、deterministic無効。

全6本に実在する共通区間 **exp step 300,032〜1,999,744** を使用した。区間の経過時間差からthroughputを求め、warmupと終了時の保存処理を除いた。

| seed | OFF区間秒 | ON区間秒 | OFF exp step/秒 | ON exp step/秒 | 速度低下率 |
|---|---:|---:|---:|---:|---:|
| 73 | 1169.005 | 1190.171 | 1453.982 | 1428.124 | 1.7784% |
| 74 | 1171.602 | 1191.321 | 1450.759 | 1426.746 | 1.6552% |
| 75 | 1176.444 | 1182.304 | 1444.788 | 1437.627 | 0.4956% |

`1 - throughput_ON / throughput_OFF` の中央値は **1.6552%（閾値5%以内）**。
同区間内の進捗ログから境界を100更新内側へ寄せた共通 **learn step 1,700〜28,000** では、全13指標が同じ更新で有限値を持つ回数が **各ON Runで52回（要求20回以上）** だった。各指標の個別件数の最小値だけで代用せず、有限値のstep集合の共通部分を数えた。
以上から、この並行稼働条件で性能の数値基準と測定成立条件を満たした。外部Runの負荷を除いた単独稼働時の性能は、この結果から確定しない。

証跡は `.scratch/prd073/performance_summary.json`、各 `perf{73,74,75}_{off,on}.execution.json` / `.throughput.json` / ONの `.metrics.json`。全点は既存 `inspect_run` readerで読み、グラフ用の間引き系列を判定に使っていない。集計スクリプトは `summarize_performance.py`、Run群は `.scratch/prd073/workspace/runs/` に保存した。

### 主な実行コマンド

リポジトリルートから実行。以下は実際に用いたテストと集計のコマンドであり、追加実行が必要という意味ではない。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test -j 4'
```

```bash
& './core/anet-core/bin/Debug/anet-core-test.exe' '[dqn],[replay_buffer],[observer_factory]' --rng-seed 73001 --reporter compact
```

```bash
& './core/anet-core/bin/Debug/anet-core-test.exe' '[replay_fit][oracle]' --rng-seed 73001 --reporter compact
```

```bash
.\.venv\Scripts\python.exe .scratch/prd073/summarize_performance.py
```

最終差分の `git diff --check` は成功。既存の無関係な変更を保持し、staging/commit/pushは実施していない。

## レビュー対応（2026-09-11）

- PRDヘッダの実装ステータスを削除し、本書への参照1文にした。実装・検証の完了状況と性能測定条件は本書に保持する。
- `ResolveReplayFitRequirements` に指標依存を集約し、Agentの群抽出RNG確保とLearnerの測定処理で共有した。Thompson判定も `ActionPolicyConfig::IsThompsonSampling` を設定検証・生成・購読制約で共有した。
- EpsilonGreedyの診断taus、UQEの診断risk taus、`MakeTargetEvaluation` の固定分位点生成を既存 `MakeTauMidpointPositions` に統一した。
- QRの `ComputeElementError` が所有する `tau_i_` を使い、診断側の再構築と学習側callerからの受け渡しを削除した。IQNは従来どおり入力に対応するtausを渡す。QRの既存生成式は変更していない。
- `MakeMunchausenTargetTerms` の引数名を宣言・定義とも `collect_diagnostics` に揃えた。
- `agent.txt` のbaselineに `replay_fit.probe.batch_size = 1024` と `replay_fit.iqn.num_taus = 32` を明示した。値は従来の既定値と同じ。
- 年齢の単位を「episode終端のdummyを含むlaneのPush回数」に揃えた。dummy自体は母数・平均の対象外だが、後続dummyのPushはlogical index差へ寄与する。CONTEXT、PRD、ReplayBuffer設計書に反映した。

### PRD §8.3で構造保証に置き換える項目

以下の2項目は、helper呼び出し回数を直接数える追加テストを作らず、**構造保証で代替**する。既存demandテストで直接確認したのは単独購読のRB要求・forward数・interval制御であり、複合購読時の分位点回帰helper呼び出しゼロを直接検証済みとは扱わない。

1. **TDのみの診断では分位点回帰helperを呼ばない。** `ResolveReplayFitRequirements` はloss指標の要求だけから `loss_u` / `loss_s` を作り、`EvaluateReplayFit` がこれを `ElementErrorRequest::loss` に渡す。QRのoverrideも同じrequestを委譲し、`QuantileLearnerBase::ComputeElementError` は `if (!request.loss) return result;` によりQR/IQN双方のloss helper呼び出しより前に戻る。通常の学習更新に必要なloss計算は、この診断のゼロ保証には含めない。
2. **異なるintervalのTD・loss複合購読でも、loss非測定回には診断の全ペア計算を残さない。** `CaptureReplayFit` は更新ごとにrequest配列をfalseで作り直し、その回に `IntervalGate::ShouldFire` が成立する指標だけをtrueにする。共有helperはその配列だけから依存を合成するため、前回のloss要求を保持しない。TDだけが測定対象になる回は上記early returnに到達し、両方が非測定なら評価自体へ進まない。

### レビュー修正の検証

- 修正前のDebug実行体で `[replay_fit],[munchausen]` をseed73001で実行し、20ケース・14,081 assertion成功、終了コード0（`.scratch/prd073/review_before.test.json` / `.test.log`）。
- VsDevCmd経由のDebug `anet-core-test` ビルドは終了コード0。`[dqn],[replay_buffer],[observer_factory]` をseed73001で実行し、253ケース中251成功・既存の想定失敗2、17,457 assertion中17,455成功・想定失敗2で、終了コード0（`review_after.build.log` / `.test.log` / `.test.json`）。
- `[replay_fit_baseline]` の明示出力を新しい `after_review/` に取得し、160 assertion成功・終了コード0。旧実装の `before_v4/` と80構成すべてのファイル名・SHA256が一致した（`review_baseline.test.json` / `.test.log` / `.comparison.json`）。証跡はいずれも `.scratch/prd073/` 配下。保存したTensor・RNG・forward記録の一致であり、性能測定の再実行ではない。
- レビュー修正の `git diff --check` は成功。既存の無関係な変更を保持し、staging/commit/pushは実施していない。
