# PRD084 Forward 事前準備リファクタリングの検証（2026-10-10）

計画は [084_noisynet_23impl.md](../../../../docs/memo/084_noisynet_23impl.md)。Config DTO 分離と実行経路変更を独立した段階で実装し、各段階で既存テストを確認した。

## 変更と責任境界

- Config DTO は ConfigData と参照キーを持つ。選択済み NetworkForwardConfig は Agent の Resource として分離した。
- PrepareForward で具象 Module／Head が設定を解釈・検証し、私有の enum・数値を持つ処理を生成する。通常の部品は本体を直接呼ぶ。
- 用途別 NetworkForwardExecution が固定した呼び先・対象 Network の寿命・State・時計種別を持つ。BindNetwork は別接続へ同じ State を共有する。
- 呼び出し側は不変 CallInfo を別文で作る。private MakeContext は情報を組み合わせるだけで、抽選・State 更新・Tensor 演算をしない。
- LinearProjectionModule の設定キャッシュ・内容比較・設定用 mutex と旧 Context 設定経路を削除した。設定取得に RTTI・dynamic_cast・型トークンは使わない。
- Body／branch／block／Head と子 Module の準備を伝達し、既存の計算・bind・trace・capture・精度保護を共用する。

## ビルド・テスト

| 検証 | 結果 |
|---|---|
| 編集前 Debug 全体ビルド・全 core | 666成功、既存1スキップ、34,873 assertions |
| Config 分離後の関連テスト | 213件、26,492 assertions |
| 準備経路・利用側移行後の関連テスト | 215件、26,407 assertions |
| 最終 NoisyNet | 51件、10,702 assertions |
| 最終 Debug 全体・Release 全体ビルド | 成功 |
| 最終全 core | 668成功、既存1スキップ、34,885 assertions |
| 公開 ConfigManager による6構成 | 全キー・文字列値・型付き設定が前後一致、44 assertions |
| 設定既定葉監査 | 25,057 assignments、3,421 audited、errors 0 |
| Noisy 指標を含める正規化契約 | RED→GREEN、Python 1件成功 |

準備時だけの解釈、設定元の寿命・変更からの独立、Network の寿命、独自設定を読む Module／Head、同名・異内容の設定、複数 Actor の独立 State と同時 Forward、target/probe 共有、時計・snapshot・保存・行対応・TD/QR/IQN/Dueling/Double-DQN/Munchausen・精度・診断非干渉を確認した。新契約は RED→GREEN とし、既存の回帰テストを現行 API へ移行した。

ログは `.scratch/prd084/forward-refactor/` の phase1-*.log、phase2-*.log、final-core-test.log、final-release-build.log、config-equivalence.log、default-leaves.log、metrics-red.log／metrics-green.log。

## 最終 Release の OFF 回帰

各500k経験、seed=1、deterministic backend、greedy_dist 無効。時刻・性能・資源・追加 Noisy 指標を除いた数値系列を (tag, step, value) に正規化し、点数と値を完全一致で照合した。

| 構成 | 点数 | SHA-256 |
|---|---:|---|
| atari_default | 78986 | `9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29` |
| atari_efficiency | 122332 | `5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e` |
| lunarlander_cpu | 88184 | `b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5` |

旧 OFF checkpoint の公開 LoadNetwork は成功（model_size=1,276,270、learner_size=1,192,239）。256経験の確認 Run も正常終了。

実効 App DTO の全既存値も一致する。config_data.txt の旧基準との生比較では、未選択 app.batchrun プロファイルの終了予算だけを500kへ明示した差があり、実際に選択される app.exp_exit_step は前後とも500k。コード変更そのものの全解決値は別途6構成で厳密一致を確認した。

## 代表 ON の前後比較

Breakout・seed=1・128 env・各500k経験。参照チェーンの AMP 継承・Body FC FP32、Actor episode/sample、Learner current/target call/sample、Munchausen target、全11診断購読、greedy_dist 無効。変更前／後を3反復 A/B交互に実行した。

Noisy 指標を含む数値系列は各 A/B と全6 Runで完全一致。109941点、SHA-256=`5c3acbb6eba510e89210025a81249b42cde1019362f42257aa70947be31b7bd6`。readout4統計・probe3指標は複数回成立。feature4統計は対象Noisy層無しによる正常な NaN（記録時は有限値の系列が無い）で、契約テストで成立する feature 統計も確認している。複数 eval／probe と正常終了を確認した。

| Run | 全体秒 | 200k後の測定区間秒 | 経験/s | CPU MiB | GPU追加 MiB |
|---|---:|---:|---:|---:|---:|
| on_before_1 | 293.142 | 212.735 | 1533.7 | 25145.37 | 7526.00 |
| on_after_1 | 254.554 | 196.586 | 1640.0 | 25146.51 | 7458.00 |
| on_before_2 | 258.591 | 209.492 | 1551.7 | 25143.89 | 7484.00 |
| on_after_2 | 237.280 | 187.196 | 1744.4 | 25150.62 | 7561.00 |
| on_before_3 | 229.465 | 182.449 | 1818.8 | 25116.52 | 7454.00 |
| on_after_3 | 231.450 | 184.622 | 1789.1 | 25124.50 | 7515.00 |

## 中央値

| 側 | 全体秒 | 200k後の測定区間秒 | 経験/s | CPU MiB | GPU追加 MiB |
|---|---:|---:|---:|---:|---:|
| before | 258.591 | 209.492 | 1551.7 | 25143.89 | 7484.00 |
| after | 237.280 | 187.196 | 1744.4 | 25146.51 | 7515.00 |

GPU はデバイス全体の起動前との差。CPU はプロセス private bytes。メモリは5秒間隔で warmup 到達後だけ採取する。区間秒は最初と最後の200k以後のメトリクスの差で、採択した step 範囲を JSON に保存する。

学習中 throughput 中央値の前後差は 12.42%。速度・score の向上は合否条件にしない。測定前後に別の Runner が無いことを確認し、同一 RTX 5090 の UUID とプロセス記録を各 execution.json に保存した。

## 形式・複雑さの監査

最終実行体 SHA-256: `1FA543538159CA74D6590EB9F42D2B54C19FCAAB04170006470E22B633939771`。変更前実行体と依存ファイル54本の保存ハッシュは不変。変更前は before/、変更後は after/ に分離し、全 source-config・設定・seed・checkpoint・実行引数・終了code・出力を保持した。
NN 指定6ファイルの Actor／Learner／Probe／lane は0件。編集した既存ファイルの BOM は HEAD と一致し、LFを維持した。PRD に完了状態行は追加していない。無関係な5ファイルの差分と index は採取時から不変。reports/INDEX.md は検証中に別作業の更新があり、現状のまま保持して external-reports-index.diff に採取した。今回の変更ファイルには含めない。

| 対象 | 裁定 |
|---|---|
| Config DTO 分離・具象準備処理・用途別実行オブジェクト | 採用 |
| 通常 Module の実行ラッパー | 本体の直接呼び出しへ縮小 |
| 設定キャッシュ・型別 Map・型トークン・公開 PrepareCall | 削除／不採用 |
| 実行中の設定切替・準備結果の重複排除 | 実需が出るまで延期 |
| State 内部 RTTI・AMP 集約・episode 用語 | 対象外 |

## 途中の失敗と未実施範囲

最初の OFF Run は設定ファイルの greedy_dist=0 がプロファイルで上書きされたため、100 episode の背景評価が有効になった。非侵襲デバッガと同一コード領域の診断 MAP で、終了待ちが背景評価 drain であることを確認した。その検証 Run だけを停止し、全artifactを failed-greedy-dist-enabled/ に保持した。CLI の明示に修正して全回帰を取り直した。測定条件は各 Run の実効設定で再確認した。
LunarLander への Atari 専用 greedy_dist CLI の誤適用は起動検証で検出した。失敗した起動記録を failed-lunar-undeclared-schedule/ に保持し、構成別に CLI を分けて成功済み Atari を保持したまま再開した。ON の初回メモリ採取は低頻度性能ログの tail に依存して warmup 到達を取り逃したため、通常の経験数ログから到達を検出する処理へ修正した。199999／200000境界・性能ログ無しの進捗・step無しの入力を確認し、不備の初回 Run を failed-warmup-sampler/ に保持して全6測定を統一手順で取り直した。
Debug テスト PDB の増大による LNK1318 は生成物を scratch へ退避して再リンクで解消した。移行中の IQN Dueling の準備処理漏れは既存 Noisy テストで検出し、共通計算へ準備済み射影を渡して修正した。失敗ログと成功ログは区別して保持する。
保留は無し。過去48 Runの全面再測定、実行中設定切替・重複排除、State 内部 RTTI の変更、AMP 集約、episode／episode_start の改名、git add／commit／push は計画の範囲外として行っていない。

## 変更ファイル

- `CONTEXT.md`
- `core/anet-core/include/anet/agent.hpp`
- `core/anet-core/include/anet/default_dqn_agent.hpp`
- `core/anet-core/include/anet/nn.hpp`
- `core/anet-core/include/anet/rainbow_agent.hpp`
- `core/anet-core/src/default_dqn_agent.cpp`
- `core/anet-core/src/dqn_based_agent.cpp`
- `core/anet-core/src/dqn_based_agent.hpp`
- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/dqn_based_heads.cpp`
- `core/anet-core/src/dqn_based_test.cpp`
- `core/anet-core/src/dqn_munchausen_test.cpp`
- `core/anet-core/src/image_cls_agent.cpp`
- `core/anet-core/src/muzero_based_agent.cpp`
- `core/anet-core/src/nn_impl.cpp`
- `core/anet-core/src/nn_impl.hpp`
- `core/anet-core/src/nn_modules.cpp`
- `core/anet-core/src/nn_test.cpp`
- `core/anet-core/src/nn_test.hpp`
- `core/anet-core/testdata/prd084/README.md`
- `core/anet-core/testdata/prd084/compare_metrics.py`
- `core/anet-core/testdata/prd084/compare_metrics_test.py`
- `core/anet-core/testdata/prd084/forward_refactor_verification.md`
- `core/anet-core/testdata/prd084/review_verification.md`
- `core/anet-core/testdata/prd084/run_forward_refactor.ps1`
- `core/anet-core/testdata/prd084/summarize_forward_refactor.py`
- `docs/adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md`
- `docs/design/130_neural_networks.en.md`
- `docs/design/130_neural_networks.jp.md`
- `docs/design/200_dqn_agents.en.md`
- `docs/design/200_dqn_agents.jp.md`
- `docs/memo/084_noisynet_10prd.md`
- `docs/memo/084_noisynet_23impl.md`
- `docs/ownership_guideline.md`

## 再レビュー修正（2026-10-10）

上記のOFF3本・ON6 Runと性能中央値は、追加レビュー修正前の実行体による記録である。今回の変更のRun証拠には流用しない。今回の出力は `.scratch/prd084/forward-refactor/review-followup-20261010/` へ分離した。

### 変更と契約

- Learner系5コンストラクタとテスト用LearnerでResourceを必須入力にし、有効時はcurrent／targetの選択との対応を準備前に検証する。空キーには既定Resource、名前付きキーには同じ末尾の非既定Resourceを要求する。無効時の休眠は維持する。
- NN内部のPreparedModuleChildrenと、メンバ関数ポインタをテンプレート引数とするPreparedHeadExecutionへ子の管理を集約した。DQNの6Preparedを削除し、通常の子／Headを直接呼ぶ。6つのForwardProjected計算本体と直接経路は編集前と一致する。
- 独自設定を読む非DQN Headで、準備時だけの解釈、通常の子へのfallback、出力と入力／parameterの勾配、全子の既定経路を確認した。
- ADR0048の後発決定、PRDの改訂／伝達経路／複雑度監査、23implの5節、設計130／200、用語、コメントと前提を同期した。

### 検証結果

| 検証 | 今回の結果 |
|---|---|
| 編集前Debug全体・準備済みForward既存テスト | 成功 |
| current／target Resource不足 | 各RED→GREEN |
| 非DQN共通Head | API不在のコンパイルRED→GREEN |
| 追加3契約テスト | 21 assertions成功 |
| Debug全体 | 成功。移行漏れを修正して再ビルド |
| CPU NoisyNet | 51成功、CUDA専用3スキップ、10667 assertions成功 |
| CPU全core | 666成功、6スキップ、33444 assertions成功 |
| 公開ConfigManagerの6構成全解決値・型付き設定 | 44 assertions、前後完全一致 |
| 設定既定葉監査 | 25100 assignments、3421 audited、errors 0 |
| Run資材 | PowerShell／Python構文と独立fixtureによる既存6 Run・今回単発の両集計モードを確認 |
| Releaseコンパイル・各testリンク | 成功（並列数1の直列再試行） |
| Runner Release通常出力リンク | 既存PID50872がEXEを使用中のLNK1104で保留 |
| Runner Release別出力リンク | 同一object／設定で成功。out／pdb／implibのみ分離 |
| Release NoisyNet（CUDAを含む） | 54件、10723 assertions成功 |
| 最終Debug全core（CUDAを含む） | 671成功、1スキップ、34906 assertions成功 |
| 最終ReleaseのOFF3本＋ON1本・旧checkpoint | 全5 Run正常終了。OFF3本とON1本の点数・checksum完全一致、旧checkpoint読込成功 |

ログは今回出力先の `current-red/green`、`target-red/green`、`head-red/green`、`final-debug-build-*`、`final-noisynet-cpu-green.log`、`final-core-test-cpu.log`、`config-equivalence.log`、`default-leaves.log`。最終CUDA検証のログは final-noisynet-release.log と final-core-test-gpu.log。CPU検証ではプロセス内だけCUDA_VISIBLE_DEVICES=-1とした。失敗・中断した試行も保存し、成功ログで置き換えていない。

### 最終実動試験（2026-10-10追記）

ユーザーが既存学習Run（PID50872）を一時停止し、実行を明示した後にOFF3本＋ON1本（各500k経験）と旧OFF checkpoint読込（256経験）を実行した。全5 Runはexit0。同じ最終Release別出力実行体を保存して使用した。停止中の既存Runは終了・再開せず、CPU時間を継続確認して再開検出を行った。既存Runはメモリに常駐しているため、今回の単発性能値を独立した性能比較の証拠にはしない。性能3反復／48 Runはユーザー選択に従い再実行しない。

| 構成 | 今回の点数 | 今回のchecksum（旧基準と一致） | 今回 |
|---|---:|---|---|
| atari_default | 78986 | `9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29` | 完全一致 |
| atari_efficiency | 122332 | `5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e` | 完全一致 |
| lunarlander_cpu | 88184 | `b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5` | 完全一致 |
| representative_on | 109941 | `5c3acbb6eba510e89210025a81249b42cde1019362f42257aa70947be31b7bd6` | 完全一致 |

修正前基準の実行体SHA-256は `1FA543538159CA74D6590EB9F42D2B54C19FCAAB04170006470E22B633939771`。別出力の最終Release実行体SHA-256は `0A75987DC0E4F0D7040F3F55F5194DE539B93F593251C124E49D80A206E90FB9`（release-link/AnetRLRunner.exe、release-artifact.json）。通常の出力は既存Runが使用する修正前実行体のまま保持した。Releaseの初回はテストとの並行実行でMSVC C3859／C1076（仮想メモリ不足）となり、進まなかった自分のNoisyNetテストだけを停止した。直列・並列数1の再試行で全コンパイルを完了し、通常出力のRunnerリンクだけが使用中EXEで止まった。別出力リンクとRelease NoisyNetのCUDA検証は成功した。初回の失敗・中断は成功証拠には含めない。


今回の集計は `after/acceptance-summary.json`、`resolved-settings-comparison.json`、`final-run-proof.json` に保存した。OFF3本の全解決値と実効App設定は、保存済み前回Runと完全一致（出力名・出力先のみ除外）。ONは代表構成の全解決値、11診断の購読、readout4統計・probe3指標の複数回成立、feature4統計の正常なNaN、複数evalを確認した。旧checkpointはmodel_size=1,276,270／learner_size=1,192,239の公開LoadNetworkと確認Runが成功した。

| Run | 全体秒 | 終了code |
|---|---:|---:|
| atari_default | 109.952 | 0 |
| atari_efficiency | 226.268 | 0 |
| lunarlander_cpu | 807.714 | 0 |
| old_checkpoint | 4.174 | 0 |
| on_after_review | 278.931 | 0 |

正常な5 Runの実行合計は 1427.039秒（約24分）。最初の実動開始から最後の正常Run終了までは約31分で、以下の補助処理修正・再試行を含む。

ON単発は200k後の区間 215.928秒、1528.1経験/s、CPU private peak 25261.19MiB、GPU起動前からの追加 6673.00MiB。反復中央値は算出しない。全5 Runの採択したGPU採取欠測は0件。

初回は専用資材のWindows PowerShell 5実行でGPUメモリ採取が失敗し、監視処理だけが中断した。次の試行は500kと保存まで進んだが終了codeがnullになった。これらは受け入れ証拠に使わず、`failed-telemetry/` と `failed-exit-observer/` に元artifactを保持した。PowerShell 7と開始直後の子process Handle取得に変更して終了code採取を確認し、正常なOFF3本を実行し直した。GPU採取の欠測は専用資材で明示記録し、学習・checksumの失敗へ偽装しない。変更は今回専用の `run_paused_forward_refactor.ps1` と `paused-run-adapter.json` に保持し、アプリ本体・設定・RNG・終了予算には手を加えていない。

### 今回の変更ファイル

BOM／LF、NN指定6ファイルのRL語0件、対象差分check、6計算本体の一致、indexと無関係な6ファイルの保持を `audit.json` へ保存する。元計画・元記録・修正前のファイル40本／実行体／DLL／設定は今回出力先のbeforeに保持する。

- `CONTEXT.md`
- `core/anet-core/include/anet/agent.hpp`
- `core/anet-core/include/anet/nn.hpp`
- `core/anet-core/include/anet/rainbow_agent.hpp`
- `core/anet-core/src/default_dqn_agent.cpp`
- `core/anet-core/src/dqn_based_agent.cpp`
- `core/anet-core/src/dqn_based_agent.hpp`
- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/dqn_based_heads.cpp`
- `core/anet-core/src/dqn_munchausen_test.cpp`
- `core/anet-core/src/nn_heads.hpp`
- `core/anet-core/src/nn_impl.cpp`
- `core/anet-core/src/nn_impl.hpp`
- `core/anet-core/src/nn_modules.cpp`
- `core/anet-core/src/nn_test.cpp`
- `core/anet-core/testdata/prd084/README.md`
- `core/anet-core/testdata/prd084/forward_refactor_verification.md`
- `core/anet-core/testdata/prd084/run_forward_refactor.ps1`
- `core/anet-core/testdata/prd084/summarize_forward_refactor.py`
- `docs/adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md`
- `docs/design/130_neural_networks.en.md`
- `docs/design/130_neural_networks.jp.md`
- `docs/design/200_dqn_agents.en.md`
- `docs/design/200_dqn_agents.jp.md`
- `docs/memo/084_noisynet_10prd.md`
- `docs/memo/084_noisynet_23impl.md`

## 2026-10-10：24impl・公開面と準備経路の整理

承認済み計画は `docs/memo/084_noisynet_24impl.md`。A1〜A4・B1〜B4・C1を実装した。B5は不採用とし、不変CallInfo・Params・getter・WithRowMappingを維持する。C2の再利用規則は変更しない。

### 今回の契約と基準

State/ContextをNN内部へ移し、StateAccessをStateのメンバへ統合した。ContextはStateポインタとCallInfoのconst参照だけを持ち、Tensorをコピーしない。時計種別は準備済み実行が保持し、公開列挙はTransientRows/ContinuingRowsの2値にした。既定実行は箱・時計を持たず、Module/Headを準備しない。名前付き準備は既定設定を拒否し、Bindは設定・箱・時計を継承する。ExecutionのNetwork取得と部分Forwardを削除し、Networkの既定部分Forwardを維持した。

probe平均側は既存NoGrad/eval/FP32保護内の既定Forwardに変更した。sample側のRNG・抽選順・購読gatingは維持する。カタログ内容のfactory宣言検証を削除し、参照存在を読込時、使用する内容を具象部品の準備時に検証する。未参照項目・使用しない設定群は休眠する。構造設定・σ decay・probe設定・再利用検証は維持した。

編集前HEADは `0dbb5cc6f5f62a2f4a1a9a8f652aefb5ff471624`。元の差分とDebug全core GREEN（666成功・6スキップ、33,444 assertions）を保存した。基準Release SHA-256は `677B8012A00B73E525F4D3FA8BFAEC0EB1E3F1D568BEE08A83E9C49389370902`。既存Runが通常EXEを使用するため、基準・最終とも既存の別出力リンク手順を使用した。objectとリンク条件を同じにし、out/pdb/implibの出力先だけを変更した。

最終Release SHA-256は `86f8ed6573c4bdbef644b59bfcc22c5e5c8dd48fb8293dbfdb4d5f244165e839`。基準の実行体・依存DLL・設定・seed・旧checkpointは `.scratch/prd084/forward-refactor/simplify-20261010/before/`、今回のRun出力は同ルートの `after/` に保存した。最終実行体は `release-link/` と `after/bin/` のSHA一致を確認した。

### テスト・監査

| 対象 | 結果 |
|---|---|
| Debug全体ビルド（MSVC初期化・並列数1） | 成功 |
| Debug NoisyNet・CPU | 53成功・3スキップ、10,628 assertions成功 |
| Debug全core・CPU | 668成功・6スキップ、33,405 assertions成功 |
| Debug NoisyNet・CUDA | 56成功、10,684 assertions成功 |
| Debug全core・CUDA | 673成功・1スキップ、34,867 assertions成功 |
| Release全体 | 全コンパイル・Core/環境テストのリンク成功。通常Runnerリンクは使用中EXEのLNK1104。別出力の最終Runnerリンク成功 |
| Release NoisyNet（CUDAを含む） | 56成功、10,684 assertions成功 |
| 設定6構成・全解決値/型付き値 | 44 assertions成功 |
| 既定葉 | 3,421項目、エラー0件 |
| BOM/LF・対象diff check | HEADのBOMを維持、CRLF0件、diff check成功 |
| 削除API・NN指定6ファイルのRL語 | 残存0件・一致0件 |
| State/Context公開面 | 定義0件・前方宣言2件 |
| ADR0048/0051 | 差分なし |

既定準備省略は2 assertionsのRED、未参照設定の休眠は2 assertionsのREDを確認してGREENにした。直接構築する既定Actorの移行漏れも既存snapshotテストでREDを確認して修正した。全子fallbackの共通Prepared契約、独自Module/Head、行対応・保持・同一時計・同時Forward・target・保存・AMP・診断非干渉の既存テストを維持した。

移行中に残った旧Head factory APIの参照はC2039で検出・修正した。テストCLIに未対応の`--device`を渡した試行は受け入れ証拠に含めず、CPUはプロセス内のCUDA_VISIBLE_DEVICES=-1、CUDAは通常環境で再実行した。失敗ログも保存し、成功ログで消していない。

ログは `logs/final-debug-build.log`、`final-noisy-cpu.log`、`final-core-cpu.log`、`final-noisy-cuda.log`、`final-core-cuda.log`、`config-equivalence.log`、`default-leaves.log`、`final-release-build.log`、`final-isolated-link.log`、`final-noisy-release.log`。監査は `audit.json`、設定は `config-final/`。通常Forwardに設定解釈・検索・設定用ロックがなく、共通部品に具体設定型を追加していないことをコード差分でも確認した。

### 最終Releaseの実動試験

| 構成 | 予算 | 点数/checksum | 終了code | 秒 |
|---|---:|---|---:|---:|
| CPU LunarLander OFF | 500,000経験 | 88,184点・旧基準と完全一致 | 0 | 1093.280 |
| 旧OFF checkpoint読込 | 256経験 | model_size=1,276,270 / learner_size=1,192,239 | 0 | 2.586 |

CPU OFF checksumは `b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5`。欠損・点数差・値差は0件。出力名/出力先を除く全解決設定も前回Runと完全一致した。記録は `acceptance-summary.json`、`logs/final-cpu-regression.log` と各 `after/*.execution.json` / `*.metrics.jsonl`。

Run資材に`-Cases`を追加し、既定のOFF3構成を維持した。CPU構成はGPUプロセス検査・GPU計測を実行しない。明示した`-Executable`で最終別出力実行体を保存し、回帰とcheckpoint確認の両方に使った。

### 保留・対象外

代表ON 500kは保留。最終確認 `2026-10-10T15:39:21.3717668+09:00` で、同じRTX 5090（GPU-0c44f747-7f1b-2d07-cd6d-a9246d000f89）に既存学習Run PID 57684 を確認した（`on-final-gpu-check.json`）。既存Runは停止・再開・終了していない。今回の最終実行体による109,941点・Noisy指標込みchecksumの受け入れは未実施であり、以前の実行体の結果で代用しない。

性能再測定、Atari OFF2本、B5、C2、新ADR、episode_start/episodeの変更は今回の計画対象外。以前のON一致・性能6 Run・48 Runは当時の実行体による記録として保持する。現在のブランチと無関係な変更を保持し、add/commit/pushは実行していない。

### 今回の変更ファイル

- `CONTEXT.md`
- `core/anet-core/include/anet/default_dqn_agent.hpp`
- `core/anet-core/include/anet/nn.hpp`
- `core/anet-core/src/default_dqn_agent.cpp`
- `core/anet-core/src/dqn_based_agent.cpp`
- `core/anet-core/src/dqn_based_agent.hpp`
- `core/anet-core/src/dqn_based_agent_test.cpp`
- `core/anet-core/src/dqn_based_heads.cpp`
- `core/anet-core/src/dqn_based_heads.hpp`
- `core/anet-core/src/dqn_based_test.cpp`
- `core/anet-core/src/nn_impl.cpp`
- `core/anet-core/src/nn_impl.hpp`
- `core/anet-core/src/nn_modules.cpp`
- `core/anet-core/src/nn_test.cpp`
- `core/anet-core/src/nn_test.hpp`（削除）
- `core/anet-core/testdata/prd084/README.md`
- `core/anet-core/testdata/prd084/run_forward_refactor.ps1`
- `docs/design/130_neural_networks.en.md`
- `docs/design/130_neural_networks.jp.md`
- `docs/design/200_dqn_agents.en.md`
- `docs/design/200_dqn_agents.jp.md`
- `docs/memo/084_noisynet_10prd.md`
- `docs/memo/084_noisynet_24impl.md`
- `docs/ownership_guideline.md`
- `core/anet-core/testdata/prd084/forward_refactor_verification.md`
