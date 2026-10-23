# PRD095 段階3 実装メモ

## 概要

PRD095 §4.5の1・2・4とCodex memoryの知見の移し替えを実施する。段階1・2の続き。2026-10-23開始。正本は095_llm_wiki_10prd.md（2026-10-22 Codex memory改訂）、先行記録は20impl・21impl。ユーザーは作業開始と、質問せず推奨案で継続することを明示した。通常の作業はこのチャットで行い、独立レビューにサブエージェントを使う場合も、直前の残量が10%以上であることを確認する。

## 主な変更

1. docs/wiki/apps/120_metrics_viewer.mdとcategories.jsonの311以外の13カテゴリを8節で作る。既存2枚も点検する。概要は設計文書、経緯はdone PRDが触れた分類対象ファイルと関連ADRから作る。機械2節はwiki_support.py maps、台帳区画はregistry.py buildで生成する。
2. reports/tools/wiki_support.py scanの開始revisionによる材料限定を除去し、常時、材料全体と最後のingest/writebackのblobを比較する。CLIはscan/mapsのままとし、互換オプションを足さない。変更ページの起点は従来どおり最後のlintを使う。両側anet-lint-wikiとwiki規約を同期する。
3. anet-ingest-wikiとanet-lint-wikiの規約を使って材料を最大12本ずつ、更新あり、設計・ADR、done PRD、実験記録・reports、その他PRDの順に取り込む。資料別要約は作らず、題材と機構のページへ集約する。知見追加の無い資料は理由を作業用台帳に残し、読み取った範囲以上の取り込み済みを記録しない。
4. Codex memoryを読み取り専用で照合する。正本が確認できる知見は正本から取り込む。正本の無い知見は、裏付けが無いまま公開資料へ起こさないことを推奨案として採り、今回移さない一覧へ残す。正本候補・対応・結果を本書へ記録する。ユーザーの好み、作業指示、失敗と次の手は移さない。
5. 実験README4本を記述単位でcampaign本文と照合し、対応する要約をwikiへ畳む。対応を確かめられない記述は独自分析としてREADMEに保持する。campaign表は期間・文書・主題・状態へ縮める。docs/experiments/README.mdへ新運用を追記する。
6. リンク・目録・孤立・引用・公開できない値・LF/BOM・skill同期・ログ追記のみ・禁止領域不変・保存状態を検査する。実装自己点検と独立レビューを行い、直せる指摘を直す。

## テスト

- Public interface / surface: reports/tools/wiki_support.pyのscan CLIと既存wiki_test.py。
- 優先behavior: 開始revision以前の未取り込み資料がpendingへ出る、ingestのblob一致で消える、更新ありとdone移動の既存契約を保持する、全16地図の機械区画生成が冪等である。
- TDD順序: まずCLIで開始revision以前の資料を検出する1テストを追加してREDを確認し、最小修正でGREENにする。その後既存のblob・移動・地図テストを実行する。文書の意味判断は実資料との照合で行い、文字列テストだけで成功扱いしない。
- 検証資材はcore/anet-core/testdata/prd095を使う。新しいC++ APIや設定は足さない。

## 検証

- .venv/Scripts/python.exe -B core/anet-core/testdata/prd095/wiki_test.py -v
- .venv/Scripts/python.exe -B core/anet-core/testdata/prd095/registry_test.py -v
- .venv/Scripts/python.exe -B reports/tools/wiki_support.py scan / maps
- .venv/Scripts/python.exe -B reports/tools/registry.py validate / build
- DebugはAGENTS.mdのVsDevCmd経由で実行し、コード変更に対する結果を記録する。
- 作業用の台帳・機械検査・差分・公開できない値の候補・memoryハッシュは.scratch/prd095/stage3/へ保存する。個人のmemoryパスや会話IDは公開する本書へ写さない。

## 前提

- 未決事項監査: ユーザー判断待ちブロッカー0。repo evidenceで解決: 材料の集合・8節・D26のdone経緯・D38のmemory非改変・D2の正本参照・ADR0056のREADME保持基準。
- 固定する前提: 未追跡の材料も含める。別作業の変更を保持する。材料の本文・PRD・ADR・CONTEXT.md・categories.jsonを変えない。正本未確認のmemory知見は公開せず未移行に記録する。質問・確認・課題の起票依頼をせず、妥当な推奨案で継続する。
- 範囲外: Claude Codeのmemoryの移し替え、同memoryのフィードバック更新、移行済み索引のポインタ置換。受入18のそれ以外を対象とする。
- git add・commit・pushをしない。log.mdは追記だけ。1単位ごとに保存する。

## 規模と作業単位の見積もり

開始時の材料は270本（約6.4MiB）。実験記録26本: 未24・済2。reports直下24本: 未22・済2。PRD144本: 未113・更新1・済30。設計15本: 未15。ADR61本: 未57・更新1・済3。合計: 未231・更新2・済37。blobは作業ツリーで比較した。全数台帳は作業用materials.json。

地図は新規14枚、合計16枚。知見ページは題材と機構に束ねて10〜25枚程度を想定し、題材が分かれる場合は追加する。1材料の照合は軽いもの1〜3分、実験記録・長いサーベイ・memory対応は5〜15分程度。全体4〜8時間は粗い見込みで、長文の精査が増えれば超過する。開始時の週次残量39%。固定の利用量換算係数は無い。

Codex memoryは主要3ファイルとrollout要約62本を開始前にハッシュ化した。MEMORYのanet-lab 56群のReusable knowledgeから272候補を抽出。うち明示的な文書・PRD・ADRへの手掛かりがある候補67、残り205はコード・測定・作業情報を含み、正本の有無をまだ判定していない。この数字を正本なし件数とは扱わない。rawと要約の重複は二重に移さない。

作業順: 地図（2〜4枚ずつ保存）→走査契約→最大12材料の取り込み→memory照合→README4本→全体検査・レビュー。途中終了時も成果が残り、未処理はscanと本書で分かるようにする。

## 経過

- 2026-10-23: 規約、PRD §4.5・4.6、ADR0056、既存wiki・ツール・先行記録を確認した。開始時の全ファイルhashとmemory hashを作業用領域へ保存した。開始前からのdirty/untrackedは保持する。

- 地図14枚（全16枚）と知見23枚を保存した。maps・registry buildを実行し、RunnerAppの.dmp参照は0件。doneのscope対応が付かない古いPRDは本文の変更対象・basename・置き場も照合して経緯を補完中。
- scan CLIのRED: 開始時から変わらないADRがpendingへ出ず失敗（scan-red.log）。開始revisionの材料フィルタだけを除去してGREEN: wiki_test.py 8テスト成功、registry_test.py 6テスト成功。規約・両側lint skill・検証READMEを更新した。DebugビルドはMSVC経由で終了0。
- 知見23枚はclaim-audit.jsonの63項目・43資料から作成した。特にQR/IQNの評価方策依存、hard C=125のn=2低分散主張の撤回、spectral_capの後続結果、成熟checkpointのbatch判断の弱化を記録した。
- memoryはMEMORY索引に未反映の最近のrolloutがあった。cwdがanet-labのrollout要約53本に298個のReusable knowledge箇条書き候補があり、索引272候補との重複を照合する。rawの62 threadと要約62本の対応も確認する。数字は移行件数ではない。

## Codex memoryの照合結果

主要3ファイルとrollout要約62本を読み取り専用で棚卸しした。rawの62 threadは全件が対応する要約を持ち、53本がanet-lab。MEMORY索引の272候補と、その53要約の298候補は同じ知見の異なる要約を含む延べ570観測であり、独立した知見570件とは数えない。提示済みのmemory_summaryは案内に使い、そこから追加の事実を公開資料へ写さなかった。

節の本文・値・条件まで一致を確認したもの17観測、作業指示・好み・工程・repo外の情報として対象外226観測、内容または時点・条件の一致を認定していない未移行327観測。関連する資料が存在するだけでは採用せず、改訂前の契約やmemoryだけの数値を現在の事実へ昇格しない。正本を持たない可能性がある知見を新しい実験記録へ起こすかの判断は、質問せず推奨案で続けるというユーザー指定に従い、裏付けが揃うまで公開記録へ起こさない案を採った。このため全570観測の移行完了とは報告しない。

レビュー前の164観測という正本参照の認定を撤回した。M170（ImageClsの旧構造をNN設計へ対応付け）、M228（Agent状態をReplayBuffer設計へ対応付け）、M241（EvalPanel同期をAtari設計へ対応付け）などは不適切だった。NoisyNetの古いAMP集約案も現行PRD084 §4.2・§8.2と一致しない。すべての旧採用候補を一度未認定へ戻し、以下だけを節の本文まで再確認した。

M/Rは今回の照合番号であり会話IDではない。本文・理由・未認定候補を含む全件の表は非公開のmemory-audit.json、rawとの対応はraw-memory-coverage.jsonに保存した。memoryの索引と本文は一切変更しない。

### 正本をwikiが参照する知見

| 照合番号 | 確認した主張と条件 | 正本の節 | wiki |
|---|---|---|---|
| M038 | R452と同じ条件、追加forward・追加抽選なし。 | [084_noisynet_10prd.md](084_noisynet_10prd.md) 7.2 targetとonline_reuse | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R346 | spectralは非退化初期化必須、zero-initはspectral_cap。 | [065_nn_spectral_norm_10prd.md](done/065_nn_spectral_norm_10prd.md) 決定事項 D12・D13 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R347 | trainingかつGradMode有効のみu/v更新。構築dummyはNoGrad。 | [065_nn_spectral_norm_10prd.md](done/065_nn_spectral_norm_10prd.md) u/vの保持とσの再計算・決定事項D5 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R372 | SHAREDのevent laneは-1、PER_LANEはlane index。 | [069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md) 4.2 行の形・F5 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [120_metrics_viewer.md](../wiki/apps/120_metrics_viewer.md) |
| R373 | 通常JSONを使いdataの順序は保証せず、取得順・keysの順序だけ保持。 | [069_metrics_trace_channel_10prd.md](done/069_metrics_trace_channel_10prd.md) 4.3 宣言（設定 DSL）・決定D7 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [120_metrics_viewer.md](../wiki/apps/120_metrics_viewer.md) |
| R380 | 履歴保護・dummyを除くsampleable集合を母集団にする。 | [073_replay_fit_metrics_10prd.md](done/073_replay_fit_metrics_10prd.md) 3.1 抽選履歴群 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [321_agent_core.md](../wiki/categories/321_agent_core.md) |
| R403 | Push時episode_startを履歴開始として保存し、stack境界をそこから復元。 | [078_replay_frame_history_integrity_10prd.md](done/078_replay_frame_history_integrity_10prd.md) 6.2 モジュールの責務 | [318_rl_core.md](../wiki/categories/318_rl_core.md) / [321_agent_core.md](../wiki/categories/321_agent_core.md) |
| R404 | 全laneを書込み前に検証。初回true、以降は直前doneまたはtruncatedと一致。 | [078_replay_frame_history_integrity_10prd.md](done/078_replay_frame_history_integrity_10prd.md) 6.2 モジュールの責務・7.3 | [318_rl_core.md](../wiki/categories/318_rl_core.md) / [321_agent_core.md](../wiki/categories/321_agent_core.md) |
| R415 | agent/env/evalのdevice文字列、auto/cpu/cuda/cuda:N。 | [070_device_spec_auto_10prd.md](070_device_spec_auto_10prd.md) 確定したD1・D2 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [312_util.md](../wiki/categories/312_util.md) |
| R416 | is_availableだけでauto解決、CUDA時はcurrent device。 | [070_device_spec_auto_10prd.md](070_device_spec_auto_10prd.md) 確定したD4・D7 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [312_util.md](../wiki/categories/312_util.md) |
| R452 | 同じ方式なら再利用、Noisy部品の方式差は初期化エラー、非Noisyは休眠。元の精度・tau・epsilonを継承。 | [084_noisynet_10prd.md](084_noisynet_10prd.md) 7.2 targetとonline_reuse | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R529 | 更新前onlineを専用probeでeval/NoGrad部分forwardし、測定時だけQ/KからFP32再計算。 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) 4. 決定した設計 1・2 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R530 | 層平均正規化entropy、1一様/0集中。個別headや因果を断定しない。 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) 4. 決定した設計 2・6. 受入基準 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R534 | head/query/sample平均をln(n)で正規化、個別head・因果の判断不可。 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) 4. 決定した設計 2・決定事項D7 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R535 | 購読時だけ診断し、SDPA不変、未測定・件数不足はNaN。 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) 3. ゴール・4. 決定した設計・6. 受入基準2 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R536 | forward入口で結果をReset、要求保持、module keyにbranch/block実行位置。 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) 4. 決定した設計 3 | [110_runner_app.md](../wiki/apps/110_runner_app.md) / [315_nn.md](../wiki/categories/315_nn.md) |
| R556 | 遷移結果はStepの結果で運ぶ。EpisodeEndEventも持ち、envは現在状態、env_stepは遷移結果。 | [0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) 冒頭の決定・Consequences | [evaluation.md](../wiki/mechanisms/evaluation.md) |

### 移さなかった知見

正本の不存在を断定しない。旧契約、仮説、工程、コードの所在だけ分かるもの、記録集合外の数値を含む。未認定候補を題材ごとに示す。

| 題材 | 照合番号 | 扱い |
|---|---|---|
| anet-lab MetricsViewer common-Step-bin comparison and scroll-prefetch overview | M001, M002, M003, M004, M005 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab Atari Kung Fu Master boss-kill RAM-metric diagnosis and intervention ranking | M007, M008, M009, M010, M011, M012 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD084 NoisyNet NN forward executor refactor and scoped staging | M013, M014, M015, M016 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab AtariView display-mode cycling and report-only commit preparation | M026, M027, M028 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD084 NoisyNet execution-settings contract and docs update | M036, M037, M039 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD083 ActionContext retirement, Run equivalence, and split commit preparation | M041, M042, M045 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab EfficientZero V2 feasibility and Batched MCTS architecture planning | M046, M047, M048 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD076 evaluation-session drain implementation and commit scope | M050, M051 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD070 device configuration unification and Run-artifact investigation | M053, M054, M055 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD082 Atari RAM progress metrics and commit-scope preparation | M056, M057 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD077 ValidIndexManager persistent sampleable-index buffer | M058, M059, M060 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ReplayBuffer frame-history integrity assay and PRD078 repair | M061, M062, M063 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab long-running build/test verification and PinnedThreadPool shutdown race | M065 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD061 Actor catalog implementation and EvalPanel defaults | M068, M069, M070 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD073 replay-fit diagnostics and subscription-gated docs workflow | M080, M081, M082, M083, M084, M085 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD072 configuration selection design and default-operator policy | M087, M088, M089, M090, M091, M092, M093 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD069 metrics trace channel, definition metadata, and commit scope | M094, M095, M096, M097, M098, M099 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD067 Munchausen RL implementation, validation, and profiling | M100, M101, M102, M103, M104 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD066 policy-churn BF16/IQN acceptance and runner wiring | M105, M106, M107 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab Atari plasticity, Spectral Normalization screening, and BTR-alignment diagnosis | M108, M109, M110, M111, M112, M113, M114 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD065 Spectral Normalization implementation, module base, and commit scope | M115, M116, M117, M119 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ReplayBuffer ring/frame-stack sampleability review, implementation, and follow-up | M121, M122, M123, M124 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD065 Spectral Normalization and deferred residual α-gate design | M125, M126, M127, M128, M129 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab DropMerge IQN experiment preparation and PRD061 Actor catalog compatibility | M134, M135, M136, M137, M138 | 同じ主張・時点・条件を認定するまで移さない |
| metrics-viewer Maven/Spring Boot release JAR workflow | M140 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab DropMerge ConvNeXt configuration and model-shape smoke validation | M148, M149, M150, M151 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab numbered Japanese design-document architecture and Mermaid documentation | M153, M154, M155, M156 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab Env instance naming, common bases, and Env-bound logging | M157, M158, M159, M160, M161 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD 035 replay-priority docs and implementation verification | M165, M166, M167, M168, M169 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ImageCls batch-first architecture review, PRD handoff, and dataset-manager ownership seams | M170, M171, M172, M173, M174, M175, M176, M178, M179, M181, M182, M183, M184, M185, M186, M187 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab DropMerge IMPALA hybrid and long-run experiment analysis | M192, M193, M194, M195, M196, M197, M198, M199, M200, M201, M202 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ImageCls MixUp/CutMix PRD planning and learner-side design | M203, M204, M205, M206, M207 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ConvNeXt CNBlock groundwork, nn cnx split, and branch-local config profiles | M208, M209, M210, M211, M212, M213, M214, M215, M216 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab WeightInitConfig.mode string migration and initializer contract | M218, M219, M220, M221, M222, M223 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab metrics.scalar @learn field resolution and update-result fallback | M224, M225, M226, M227, M228, M229 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab DQN architecture review and HTML reporting | M231, M232, M233, M234, M235 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab ImageCls augmentation, normalization, and GAP head workflow | M236, M237 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab replay-buffer transfer helper, PER readback, and PrefetchingReplayBuffer lifecycle | M238, M239, M240 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab EvalPanel model sync policy | M241, M242 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab DQN/TBO configuration, scaling, and run interpretation | M243, M244, M245, M246, M247 | 同じ主張・時点・条件を認定するまで移さない |
| anet-lab PRD075 episode steps, evaluation session observability, and commit scope | M270, M271, M272 | 同じ主張・時点・条件を認定するまで移さない |
| implement prd with docs skill and claude discovery | R284, R285, R287, R288, R289 | 同じ主張・時点・条件を認定するまで移さない |
| per observability and actor priority prd 999 | R294, R295, R296, R297, R298, R299, R300 | 同じ主張・時点・条件を認定するまで移さない |
| anet design docs reorganization and dqn replaybuffer guides | R308, R309, R311, R312, R313, R314, R315, R316, R317, R318, R319, R320 | 同じ主張・時点・条件を認定するまで移さない |
| prd065 spectral normalization common base and commit scope | R345, R348, R349, R350, R351, R352, R353, R354, R355 | 同じ主張・時点・条件を認定するまで移さない |
| atari plasticity sn btr diagnosis | R357, R358, R359, R360, R361, R362, R363, R364, R365, R366 | 同じ主張・時点・条件を認定するまで移さない |
| prd069 trace docs grill and sync | R371, R374 | 同じ主張・時点・条件を認定するまで移さない |
| prd067 munchausen rl implementation validation profiling | R376 | 同じ主張・時点・条件を認定するまで移さない |
| prd073 replay fit docs subscription gating | R377, R378, R379 | 同じ主張・時点・条件を認定するまで移さない |
| prd072 config default operator policy | R393, R394 | 同じ主張・時点・条件を認定するまで移さない |
| replaybuffer integrity assay and long test usage rules | R395, R396, R397, R398, R399, R400, R401 | 同じ主張・時点・条件を認定するまで移さない |
| prd078 replay history grill and commit preparation | R405 | 同じ主張・時点・条件を認定するまで移さない |
| prd070 device config unification and commit scope | R417, R419, R420, R421, R422, R424 | 同じ主張・時点・条件を認定するまで移さない |
| prepare simple paper details explainer | R427 | 同じ主張・時点・条件を認定するまで移さない |
| efficientzero v2 feasibility batched mcts prd | R437, R438, R439, R440, R441, R442, R443 | 同じ主張・時点・条件を認定するまで移さない |
| prd083 actioncontext retire tracy fix and commit split | R444, R445, R446 | 同じ主張・時点・条件を認定するまで移さない |
| prd084 noisynet execution settings docs update | R450, R451, R453, R454 | 同じ主張・時点・条件を認定するまで移さない |
| prd084 noisynet forward executor refactor and staging | R461, R462, R463, R464, R465, R466, R467, R468, R469 | 同じ主張・時点・条件を認定するまで移さない |
| atariview image display mode middle click | R470 | 同じ主張・時点・条件を認定するまで移さない |
| anet review run guide verification record missing | R482, R483 | 同じ主張・時点・条件を認定するまで移さない |
| kung fu master boss kill ram metric diagnosis | R484, R485, R486, R487, R488, R489, R490, R491, R492 | 同じ主張・時点・条件を認定するまで移さない |
| metricsviewer cache rebuild prd090 implementation and commit | R499, R500, R501 | 同じ主張・時点・条件を認定するまで移さない |
| temporl model grill order | R507, R508, R509 | 同じ主張・時点・条件を認定するまで移さない |
| prd091 checkpoint optimizer options implementation | R511 | 同じ主張・時点・条件を認定するまで移さない |
| gradient noise survey prd092 | R512, R513, R514, R515 | 同じ主張・時点・条件を認定するまで移さない |
| prd092 additional grill document revision | R517, R520 | 同じ主張・時点・条件を認定するまで移さない |
| prd092 gradient noise scale implementation and commit scope | R522, R523, R524 | 同じ主張・時点・条件を認定するまで移さない |
| prd093 attention entropy regrill document revision | R531, R532, R533 | 同じ主張・時点・条件を認定するまで移さない |
| dqn transformer algorithm taxonomy | R538, R539, R540, R541, R542, R543, R544 | 同じ主張・時点・条件を認定するまで移さない |
| anet survey queue priority 10 llm rl survey | R545, R546, R547 | 同じ主張・時点・条件を認定するまで移さない |
| prd087 stage2 docs and review fixes | R549, R550 | 同じ主張・時点・条件を認定するまで移さない |
| prd097 step result transition metrics implementation and com | R557, R558 | 同じ主張・時点・条件を認定するまで移さない |
| anet lab borrow rule videologger validation | R562 | 同じ主張・時点・条件を認定するまで移さない |
| prd094 evaluation trace generic diagnostics and commit scope | R565, R566, R567, R568 | 同じ主張・時点・条件を認定するまで移さない |

前後hashはmemory-before.json / memory-after.jsonに保存する。公開版には集約digestと変化の有無だけを記す。

## READMEの仕分け

初回の「段落全体が同じでなければ独自分析」という判定と193単位の保持数を撤回した。表のセルや同じ段落に別の条件・方針が混在する場合は主張ごとに分け、同じ測定・条件・結論をcampaignの節で確認して仕分け直した。原文の位置、主張、扱い、参照する正本の節、保持理由はreadme-audit.jsonへ保存した。集計単位は編集の追跡用であり、独立した知見の数ではない。

- 共通README: Atari構成の成績、capacity/lane窓、warmup、IQN本数、勾配step摩耗、weight decay、greedy断面、NoisyNet感度、target評価、evalコストはcampaignからwikiへ集約。seedをblockとして使う比較手順、Envをまたぐ一般化方針、Munchausen温度を別の報酬尺度へ適用する推論だけを分離して残した。
- Atari README: プロファイル到達値、ViT比較・再測定・capacity交絡、ラック・分母・eval密度・複製幅・過渡・Atari-5集約の要約を集約。主題セルの結果や「寄せ失敗・修正済み」を除去。成績によらずViTを常設する方針、BTR NN構成のRR2を追加探索しない方針、判定が必要な軸だけ複製を取る運用規則は、実測値から分けて残した。
- DropMerge README: campaignにあるNEET/Qの観察・結果の再掲を除去。NOOPの1物理frameと観測の表現力、裁定と方策学習の区別・candidate診断、UQEと表示Meanの区別、campaignに同値を確認できないcy07の175〜200M観測、診断を進める順序を保持。scratch B512/RR2の数値は再掲せず正本へのリンクとした。wikiから取り除いたREADMEの段落を参照していた箇所もcampaignへ付け替えた。
- LunarLander README: T50/T180、ホバリング離脱と着陸定着、seed差の部分観測仮説は08-08 campaignのまとめ4・9〜11と読み方にあるため集約。探索対象としての位置付けは残した。
- 元の判断行57件は実測・結果と独自方針が混在するため、行全体を同じ扱いにしない。実測・結果の部分とViTの成績はwikiで正本を参照し、常設・追加探索・複製の取得に関する独自方針はREADMEに残した。readme-audit.jsonでは該当する文の認定を訂正した。

Debugビルドは次の実行済みコマンドで終了0。詳細は作業用debug-build.logとdebug-build-exit.txtに保存した。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

wiki_test.py 8テスト、registry_test.py 6テストは成功した。

## 一括取り込みと自己点検の記録

- 材料270本を、更新ありを先に最大12本ずつ20単位で処理し、ingest記録233行を追記した。先に記録済みでblob一致だった37本は重複記録しなかった。その後のREADME整理の追加修正は、最終blobで再記録する。材料ごとの章構成、扱い、反映先・追加なしの理由はmaterial-audit.json、単位の記録はingest-batches.json。
- 全体scanの初回結果: pending 0、broken_links 0、unindexed 0、eligible=false。機械的な既存リストと新しい地図はmaps・registry buildで生成した。
- 全41ページ（地図16、知見25）の節、出典リンク、目録、到達性、公開できない値、campaign表、ログ追記について110項目成功。目録から到達できないページは0。初回検査では既存311の「設計文書に記述なし」の箇条書きを検査側が誤って失敗に数えたため、規約の例外を検査へ反映して再検査した。
- 旧名候補TraceSinkとDrop.pは、経緯のPRDの当時の節名・ADRの題の引用であり、現行契約としての利用ではない。推測で新名へ置き換えない。
- READMEの独自分析へwikiから直接リンクした。DropMergeのepisode途中のNOOP attractorの分析はREADMEの当該節を出典としてwikiへ取り込んだ。ViTの常設方針だけをREADMEに残し、同じ行にあった実測の要約はcampaignを出典として分離した（判断表57行の全体移行と1行の部分移行）。

### 初回に残した正本間の記載差（今回の指摘対応で是正）

- 初回には150 ReplayBuffer §2.4のK=2と、200 DQN系Agent・ADR0036のK=3が食い違っていた。今回DecodeActorQHintの3列とADR0036を再確認し、指定の1か所だけK=3へ訂正した。
- 初回には160 applications §2.2のfactor16と、210 MetricsViewer・ADR0052のfactor4が食い違っていた。今回LodBucket.LOD_FACTOR=4とADR0052を再確認し、指定の1か所だけfactor4へ訂正した。
- 初回は材料の編集禁止に従い変更しなかった。今回の指摘対応4で明示許可された日本語設計文書2か所だけを是正し、2材料を取り込み直した。英訳・ADR・PRDは変更しない。

- 変更ファイル点検で、Atari READMEの元のBOMを除いていたことを検出し、元どおり保持した。LFは維持する。
- 作業中に別作業の更新を26ファイルで検出した（英訳、housekeepingの資材・台帳、Tallec論文解説など）。今回の編集コマンドの対象外であり、変更を戻さず保存状態の差をexternal-changes.jsonへ記録した。正本PRD095・CONTEXT.md・カテゴリ定義と日本語設計文書・ADRは今回変更していない。Tallec材料は最終blobで取り込み記録を更新する。

- 地図の機械2節は再生成後に冪等（2回目の更新0件）、PRD番号の未解決は0件。registry validateは27レコード・問題0件。
- 記述なしの集計: 概要の一部に設計文書未記述が残るのは既存311。選ばなかった形は7ページ（312・313・316・317・319・320・325）、変更時の個別条件は3ページ（313・319・325）。一般のガードレールとテスト充足へのリンクは全16ページに置く。
- レビュー前の中間照合ではmemoryの65ファイルのSHA256が全件一致し、変更0件だった。ファイル名とhashをソートしてJSON化した集約SHA256も前後一致: `4e2e8847fcf401ef12345e9a21fae3838dcf1727ba3b4eca138dd73380eabf44`。ファイル内容・個人のパスは公開資料へ写していない。レビュー待機後にも最終時点で再照合する。
- 実装品質確認観点は全項目を適用範囲で自己点検した。Pythonの変更はscanの材料限定の除去のみ、テストは公開CLI経由。設定・C++ API・runtime・所有権・計測の変更は無く該当観点は対象外。docs/designの正本は変えず、160 §7.4が正本として指すwiki規約を更新した。skillの用途・出力先は変わらず一覧の更新は不要。
- 変更ファイル57本のLF/BOM・skill同期・正本PRD095/CONTEXT/categoriesの保持など79項目は成功。外部更新26ファイルには別作業が削除した作業資材も含むため、存在を強制する検査は適用せず、変化として記録し、その状態を戻さない。
- レビュー1往復目: 残量31%を確認して独立an-reviewへ依頼した。計画と本文以外の実装を対象とし、wikiの本文・目録・ログはD4に従って書いた側で確かめる。依頼・差分は作業用review-request-1.md / diff-review-1。

- レビュー1は計画・実装とも差し戻し。memoryの誤対応、READMEの混在した要約、正確なビルドコマンド欠落を修正し、同じレビュー担当へ再提出する。

- レビュー2: memoryとDebugの指摘解消を確認。Atariの独自方針2点を要約として消していたため差し戻し。実測とは分けてREADMEへ復元し、照合表と完了記録を訂正してレビュー3へ再提出する。

### 最終時点のmemory更新

01時台の最終再照合では、中間照合後にmemoryの外部更新を検出した。対象を開始時と同じ主要3ファイルとrollout要約に限ると65→64ファイル、既存2ファイルの内容更新、要約2本の消失、要約1本の追加だった。当方の操作は全て読み取り専用であり、memoryへ書き込むコマンドは実行していない。Codexアプリ側など当方以外の更新として区別し、前後のhashとファイルごとの差はmemory-before.json / memory-final.json / memory-final-change.jsonへ記録した。公開文書には個人の絶対パスや会話IDを載せない。

この更新を含めて「最終時点でも65件不変」とは報告しない。570観測の棚卸しは開始時に読んだsnapshotの結果で、新しく追加された要約は今回の移行完了に含めない。新しい記述を正本未確認のまま公開資料へ写さない。

## 最終結果と残件

- レビュー3はimpl-plan・implともOK（2件OK・差し戻し0）。応答原文は作業用review-20261023-stage3-3.mdに保存した。wiki本文・目録・ログはD4どおり著者による点検であり、意味内容の独立レビュー済みとは扱わない。
- scanの全材料照合、実装の地図14枚の新設、知見24ページの追加、README4本の仕分けと実験全体の運用追記を実施した。wikiは地図16＋知見25＝41ページ。既存材料270本に対して最後のblobを記録し、最終scanの未処理0・リンク切れ0・目録漏れ0を確認する。
- 最終のwiki8テスト・台帳6テストを再実行して成功。Debugビルドは前述の実行済みコマンドで終了0。wiki機械検査110項目・ファイル検査79項目は成功。mapsの2回目再生成の更新0・PRD番号未解決0、registry validateは27レコード・問題0。
- Codex memoryは開始時570観測のうち17を正本の節まで確認、226を対象外、327を未認定・未移行として残した。未認定を公開記録へ昇格しない。全件の移行完了とはしない。新しく外部追加された要約も今回の確認済みに含めない。memoryへの書き込み0、外部更新とhashは前節の記録を参照。
- 範囲外の残件: Claude Code memoryの知見移行、feedback改訂、移行後のmemory索引のpointer置換。初回の正本間の記載差2件は、後掲の指摘対応で是正した。
- git add・commit・pushは実行していない。ユーザーの別作業26ファイルの更新・削除を保存したまま、今回の変更57ファイルの最終内容・hash・mtimeを保存して停止する。

## 2026-10-23 段階3の指摘対応（サブエージェントなし）

改訂済みPRD095と指摘文書を入力に、既存の22implへ追記して進める。人間への質問とサブエージェントは使わず、同じ担当が自己点検する。独立レビュー済みとは報告しない。

1. 13カテゴリの分類済みファイルに触れたコミットのscopeからPRDを再集計し、done移動の有無を条件にしない。追加PRDの全文と該当ADRを読み、経緯・選ばなかった形・既存挙動の維持条件を更新する。PRD×カテゴリの照合で不足・余分を確認する。
2. 知見25ページの結論を既知の事実の言い換えへ直す。Atari関連6ページはcampaignの現在のまとめを読み、節・Run付きで補う。新しい推奨や値は作らない。
3. 未解決は出典にある問い・食い違いだけにし、出典付きで残す。該当しなければ「なし」とする。wiki規約も改訂済みの未解決・経緯の条件へ揃える。
4. 指定の日本語設計文書2か所はコード・ADR・具体設計との一致を確認してK=3 / factor4へ訂正し、取り込み直す。英訳・PRD・ADR・CONTEXT・カテゴリ定義は変更しない。

コードは変更しないためTDDとC++ビルドは今回の対象外。全材料scan、mapsの冪等性、台帳validate、出典・リンク・目録・孤立・結論の禁止表現・privacy・LF/BOM・保護対象のhashを検証し、最後に結果と残件を記録する。

### 指摘対応の照合結果

コミットscopeから抽出した98組のPRD×カテゴリを照合し、33組を11カテゴリへ追加した。313と325も再集計し、追加不足は無かった。元のdone PRDの本文・変更対象を起点にした既存のfallback行は、今回のscope抽出とは分けて作業用history-audit.jsonのextraへ記録し、既存の経緯として保持した。scopeが無いdone PRDの本文からの取り込みは§4.7の移動時fallbackによるものであり、今回新たにその方法で非done PRDを推測して足してはいない。今回の追加PRDは全て実際のコミットと対象ファイルを確認した。

| カテゴリ | scopeで確認したPRD組数 | 今回追加したPRD | 不足 |
|---|---:|---|---:|
| 312_util | 5 | 070 | 0 |
| 313_profiling | 0 | なし | 0 |
| 314_tensor | 5 | 070・083・094 | 0 |
| 315_nn | 6 | 084・092・093 | 0 |
| 316_config | 6 | 070・093 | 0 |
| 317_metrics | 4 | 094 | 0 |
| 318_rl_core | 19 | 070・084・094・097 | 0 |
| 319_gui | 3 | 082・094・097 | 0 |
| 320_env_core | 8 | 070・094・097 | 0 |
| 321_agent_core | 18 | 070・083・084・092・093・094 | 0 |
| 322_env_individual | 6 | 082・097 | 0 |
| 324_dqn | 15 | 083・084・092・093・094 | 0 |
| 325_muzero | 3 | なし | 0 |

| PRD | 今回追加したカテゴリ | 不足 |
|---|---|---:|
| 070 | 312・314・316・318・320・321 | 0 |
| 082 | 319・322 | 0 |
| 083 | 314・321・324 | 0 |
| 084 | 315・318・321・324 | 0 |
| 092 | 315・321・324 | 0 |
| 093 | 315・316・321・324 | 0 |
| 094 | 314・317・318・319・320・321・324 | 0 |
| 097 | 318・319・320・322 | 0 |

追加した8 PRDは全文を対象に、節名に依存しない候補検索と該当段落の読解で裁定と非干渉条件を拾った。ADRは追加した経緯PRDからの参照と対象の形の両条件で選び、26組を経緯へ補った。不採用・先送りの形は76組、既存挙動の維持条件は30組を追加した（いずれも同じ主張を複数カテゴリへ載せた延べ数であり、独立した知見の件数ではない）。元の選択や新しい機能の合否を2節へ写さないよう自己レビューし、ADR0058に存在しないConsidered Options節の引用を「本文」へ直し、317の不採用案の出典となるADR0057の経緯漏れも直した。

知見25ページの結論は既知項目の事実だけで1〜5行へ改め、指示・推奨の語を機械検索した。Atari関連6ページへ最新campaignのまとめから12項目を追加し、全項目に節とRun名を付けた。Atari-5は窓評価と学習中最良の集計を混同せず、50M・seed1と100M・seed2を明記した。qbertでは64 env/1Mの3 seedと比較腕を同じ突破条件で記録した。未解決はKFMの既存3問いと、qbert・lane窓・name_this_gameの出典に残る機序・原因の問いだけを出典付きで残し、他21ページは「なし」とした。定型の汎化の問いは0件。目録の要約も各ページへ揃えた。

指定設計2か所は、150の1置換と160の1置換だけであることを変更前の保存版と比較した。150のPERの責任境界は318へ、160のfactor4取り込みは120 MetricsViewerの概要へ取り込み直した。読み直した26材料を12・12・2本の単位でlogへ追記し、ADR0057は自己レビューの追加反映後にも記録した。logの既存部分は変更しない。

### 検証とレビューの扱い

- 実行: `reports/tools/wiki_support.py scan`、`reports/tools/wiki_support.py maps`を2回、`reports/tools/registry.py validate`。いずれもrepoの.venvを使用。実行結果は作業用stage3-fixのJSON・ログへ保存する。
- 自己点検: 出典、相対リンク、全ページの節と目録・孤立、結論の指示語、未解決の根拠、privacy、LF/BOM、設計2か所だけの差分、保護対象のhash、経緯98組と取り込み8 PRDのblobを確認する。実装品質確認観点全62項目を適用範囲で通し、コード・設定・所有権等の未変更部分は対象外とした。
- サブエージェントは使っていない。ユーザーの直接指示を優先し、独立レビューへの依頼は行わず、この担当による自己レビューと差し戻し相当の修正を実施した。前回のレビューOKを今回の修正版に対する独立OKとして流用しない。
- 今回のコード・runtimeの変更は無く、ビルドとコードテストは再実行しない。wikiの機械検査と出典・条件の自己点検で今回の文書修正を検証する。

### 指摘対応の最終結果

- 全材料scan: 未取り込み0、リンク切れ0、目録漏れ0（scan-final.json）。地図2回目: 更新0、PRD番号未解決0（maps-second.json）。台帳: 27レコード、問題0。
- 今回の自己点検は404項目成功。scope由来PRD×カテゴリ98組は全て経緯にあり、不足0。知見25ページの結論に指示語0、定型の未解決文0、ページのリンク切れ・孤立0。
- 43ファイルの変更前・変更後の差分、最終hashとmtimeをstage3-fixへ保存した。PRD・ADR・CONTEXT・categories定義・英訳・コードは開始時のhashを保持。別作業の変更は戻していない。
- 今回依頼の4件は対応済み。新しい未決は無し。ユーザー指定により独立レビュー未実施。この記録より前のmemory未移行・Claude側作業は今回の修正範囲外で、その残件を完了に変えていない。stage・commit・pushは行わず停止する。

## 2026-10-23 裁定語の追加修正

改訂済みPRD095 §4.6と追加修正の依頼を入力とし、新しい計画は作らず22implへ追記する。ユーザーの最新指示によりサブエージェントを使用する。

- 対象は全16地図の「選ばなかった形」97項目とwiki規約。形・理由・出典は保持し、裁定の語と出典にある再開条件だけを直す。
- NN・Agent・DQNの51項目と残り46項目を別の担当が出典節から照合する。別関心として別/後続PRDへ回したものを範囲外、そのほかで再開条件があるものを先送り、残りを不採用の順で判定する。既に3語を使っている42項目も除外しない。
- 裁定は3語だけで書き、元の語は変更一覧に残す。先送り・範囲外で出典に再開条件がある場合は、その条件を追記する。出典が条件を持たない範囲外に、こちらの判断で新しい条件を作らない。
- 規約の該当文はPRDから同じ文を写し、PRD095を材料とする最新blobのfix行をlogへ追記する。
- 全項目の3語始まり、元の形・理由・出典の保持、再開条件の転記、相対リンク・目録・公開できない値、LF/BOM・保護対象hashとscanを確認する。コードは変えないためビルドとコードテストは対象外。
- 地図の本文はPRDのD4どおり書く側で確認する。更新した計画と規約は実装品質確認観点で独立レビューし、指摘を修正して結果を追記する。今回の修正を、前回の4件に対する独立レビューの遡及承認とは扱わない。

### 裁定語の照合結果と変更一覧

全16地図・97項目を出典の節から再判定した。結果は不採用61・先送り16・範囲外20。裁定を変更した72項目は、3語以外からの正規化55項目と、既存3語の再分類17項目である。残る25項目は裁定を保持した。形・理由・出典は保持し、再開条件だけ4項目へ補った。

- ObservationPipeline（314・321・324）: PRD083の複雑さ監査および対象範囲にある、段の要件が揃う後続PRDへの条件を補った。
- TraceCallback統合（315）: PRD093 §5・§7から参照する999_nn_trace_capture_unificationのゲート（要求フィルタまたは名前付き出力の3つ目の経路、093のコミット済み前提）を補った。
- 担当別の97項目の出典節・根拠・条件の照合記録は、作業用stage3-verdictのaudit-all.jsonへ保存した。以下は裁定を変更した全項目である。

| ページ | 項目 | 元の語 | 当てはめた語 | 根拠にした出典の節 |
| --- | --- | --- | --- | --- |
| [120_metrics_viewer](../wiki/apps/120_metrics_viewer.md) | 未選択Runの背景取り込みを廃止し、選択時だけ取り込む形 | 範囲外 | 不採用 | [064_metricsviewer_idle_ingest_io_10prd.md](done/064_metricsviewer_idle_ingest_io_10prd.md) 0.3 Non-goals |
| [120_metrics_viewer](../wiki/apps/120_metrics_viewer.md) | WatchServiceなどのfilesystem通知によるevent駆動化 | 範囲外 | 不採用 | [064_metricsviewer_idle_ingest_io_10prd.md](done/064_metricsviewer_idle_ingest_io_10prd.md) 0.3 Non-goals |
| [311_dev_tools](../wiki/categories/311_dev_tools.md) | inspect_runによるcacheの生成・更新・修復・migration・削除 | 範囲外 | 不採用 | [054_inspect_run_10prd.md](done/054_inspect_run_10prd.md) §10 |
| [311_dev_tools](../wiki/categories/311_dev_tools.md) | inspect_runによるmetricの意味解釈・採否判断・Run ranking | 範囲外 | 不採用 | [054_inspect_run_10prd.md](done/054_inspect_run_10prd.md) §10・§11 |
| [311_dev_tools](../wiki/categories/311_dev_tools.md) | inspect_runのraw全点の無制限出力・plot画像生成・interactive viewer | 範囲外 | 不採用 | [054_inspect_run_10prd.md](done/054_inspect_run_10prd.md) §10 |
| [314_tensor](../wiki/categories/314_tensor.md) | 低解像度動画をNVENC用にpaddingする形 | 範囲外 | 先送り | [025_videolog_codec_auto_ffmpeg_guard_10prd.md](done/025_videolog_codec_auto_ffmpeg_guard_10prd.md) ゴール / 非ゴール |
| [314_tensor](../wiki/categories/314_tensor.md) | ObservationPipelineクラスと専用ファイル・テストの追加 | cut（後続PRDへ） | 範囲外 | [083_action_context_retire_10prd.md](083_action_context_retire_10prd.md) 複雑さ監査 |
| [315_nn](../wiki/categories/315_nn.md) | 独立した公開NoisyLinearブロック型 | 採用しない | 不採用 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §4.1 |
| [315_nn](../wiki/categories/315_nn.md) | AMPの実行設定への集約とノイズ込み重みのSN | 対象外 | 範囲外 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §8.1・§8.2・§14.2 |
| [315_nn](../wiki/categories/315_nn.md) | Noisyの有無フラグ・機能登録・実行条件の汎用比較と独立実行基盤 | 導入しない | 不採用 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §4.2・§14.2 |
| [315_nn](../wiki/categories/315_nn.md) | PPO/SAC・複数損失・別optimizer向けの拡張口 | defer | 先送り | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §8 5・§10.2 |
| [315_nn](../wiki/categories/315_nn.md) | head単位統計とattention entropy最小値 | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5・§7 |
| [315_nn](../wiki/categories/315_nn.md) | TraceCallbackのcaptureへの統合 | 先送り | 範囲外 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5・§7 |
| [315_nn](../wiki/categories/315_nn.md) | TQLのattention entropy正則化 | 先送り | 範囲外 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §7 1 |
| [315_nn](../wiki/categories/315_nn.md) | moduleのbufferにεを置きreset_noiseで引き直す形 | 棄却 | 不採用 | [0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | 毎forwardでの設定解釈と内容別キャッシュ | 削除 | 不採用 | [0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md](../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | ForwardUpToへのattention専用受け皿とNetwork専用診断メソッド | 棄却 | 不採用 | [0055-forward-capture-request-keyed-carries-module-diagnostics.md](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | NN実行Stateのスロットへ診断値を書く形 | 棄却 | 不採用 | [0055-forward-capture-request-keyed-carries-module-diagnostics.md](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | 別型の汎用診断captureを足して既存BranchCaptureも残す形 | 棄却 | 不採用 | [0055-forward-capture-request-keyed-carries-module-diagnostics.md](../adr/0055-forward-capture-request-keyed-carries-module-diagnostics.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | ReplayBufferから別に一様抽出する勾配ノイズprobe | 棄却 | 不採用 | [0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md](../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md) Considered Options |
| [315_nn](../wiki/categories/315_nn.md) | BatchNormの専用検出・禁止 | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [315_nn](../wiki/categories/315_nn.md) | attention entropy専用のfeatureキー | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5 |
| [315_nn](../wiki/categories/315_nn.md) | 構築時dummy forwardで層数を数える追加API | 作らない | 不採用 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5 |
| [316_config](../wiki/categories/316_config.md) | PRD072固定入力・goldenの再採取 | cut | 不採用 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §9 複雑さ監査 |
| [317_metrics](../wiki/categories/317_metrics.md) | step traceをtraceチャネルへ記録する配線 | PRD081へ移す | 範囲外 | [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md) §6 |
| [317_metrics](../wiki/categories/317_metrics.md) | Event/Observerの型の統一 | 別PRD | 範囲外 | [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md) §6 |
| [317_metrics](../wiki/categories/317_metrics.md) | metricsのtrace定義tagをEvaluation Traceの列として参照する形 | 却下 | 不採用 | [0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) Considered Options |
| [318_rl_core](../wiki/categories/318_rl_core.md) | Observerの周期判定をAgent内部のtriggerへ戻す形 | 範囲外 | 不採用 | [053_interval_gate_perf_ema_10prd.md](done/053_interval_gate_perf_ema_10prd.md) 5. スコープ外 |
| [318_rl_core](../wiki/categories/318_rl_core.md) | step traceをtraceチャネルへ記録する配線 | PRD081へ移す | 範囲外 | [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md) §6 |
| [318_rl_core](../wiki/categories/318_rl_core.md) | Event/Observerの型の統一 | 別PRD | 範囲外 | [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md) §6 |
| [318_rl_core](../wiki/categories/318_rl_core.md) | episode開始通知とViewの開始表示 | 先送り | 範囲外 | [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md) §5 |
| [318_rl_core](../wiki/categories/318_rl_core.md) | Actor設定キーでeval_panelだけにhintを生成させる形 | 採らない | 不採用 | [0058-approx-priority-column-computed-by-agent-step-diagnostics.md](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) 本文 |
| [318_rl_core](../wiki/categories/318_rl_core.md) | 全targetを許し値の時点差は文書だけで説明する形 | 却下 | 不採用 | [0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) Considered Options |
| [318_rl_core](../wiki/categories/318_rl_core.md) | 速報値と確定値を別の鍵名に分ける形 | 却下 | 不採用 | [0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) Considered Options |
| [319_gui](../wiki/categories/319_gui.md) | step traceをtraceチャネルへ記録する配線 | PRD081へ移す | 範囲外 | [094_evalpanel_step_history_10prd.md](094_evalpanel_step_history_10prd.md) §6 |
| [319_gui](../wiki/categories/319_gui.md) | episode開始通知とViewの開始表示 | 先送り | 範囲外 | [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md) §5 |
| [319_gui](../wiki/categories/319_gui.md) | metricsのtrace定義tagをEvaluation Traceの列として参照する形 | 却下 | 不採用 | [0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) Considered Options |
| [319_gui](../wiki/categories/319_gui.md) | 全targetを許し値の時点差は文書だけで説明する形 | 却下 | 不採用 | [0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md](../adr/0057-eval-trace-columns-declared-apart-from-metrics-and-read-at-transition-time.md) Considered Options |
| [320_env_core](../wiki/categories/320_env_core.md) | episode開始通知とViewの開始表示 | 先送り | 範囲外 | [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md) §5 |
| [320_env_core](../wiki/categories/320_env_core.md) | 速報値と確定値を別の鍵名に分ける形 | 却下 | 不採用 | [0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) Considered Options |
| [321_agent_core](../wiki/categories/321_agent_core.md) | ImageClsのTrain Actorにも遅延snapshotを適用する形 | 範囲外 | 不採用 | [036_train_actor_periodic_snapshot_10prd.md](done/036_train_actor_periodic_snapshot_10prd.md) 対象範囲 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | ObservationPipelineクラスと専用ファイル・テストの追加 | cut（後続PRDへ） | 範囲外 | [083_action_context_retire_10prd.md](083_action_context_retire_10prd.md) 複雑さ監査 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | 独立した公開NoisyLinearブロック型 | 採用しない | 不採用 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §4.1 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | AMPの実行設定への集約とノイズ込み重みのSN | 対象外 | 範囲外 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §8.1・§8.2・§14.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | Noisyの有無フラグ・機能登録・実行条件の汎用比較と独立実行基盤 | 導入しない | 不採用 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §4.2・§14.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | PPO/SAC・複数損失・別optimizer向けの拡張口 | defer | 先送り | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §8 5・§10.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | 購読間隔ごとの別EMA | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | head単位統計とattention entropy最小値 | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5・§7 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | TQLのattention entropy正則化 | 先送り | 範囲外 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §7 1 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | moduleのbufferにεを置きreset_noiseで引き直す形 | 棄却 | 不採用 | [0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md](../adr/0048-noisynet-epsilon-in-caller-execution-state-and-nn-forward-key.md) Considered Options |
| [321_agent_core](../wiki/categories/321_agent_core.md) | ReplayBufferから別に一様抽出する勾配ノイズprobe | 棄却 | 不採用 | [0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md](../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md) Considered Options |
| [321_agent_core](../wiki/categories/321_agent_core.md) | BatchNormの専用検出・禁止 | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | 未成立理由ごとの追加状態指標 | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | attention entropy専用のfeatureキー | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | 構築時dummy forwardで層数を数える追加API | 作らない | 不採用 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5 |
| [321_agent_core](../wiki/categories/321_agent_core.md) | Actor設定キーでeval_panelだけにhintを生成させる形 | 採らない | 不採用 | [0058-approx-priority-column-computed-by-agent-step-diagnostics.md](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) 本文 |
| [322_env_individual](../wiki/categories/322_env_individual.md) | episode開始通知とViewの開始表示 | 先送り | 範囲外 | [097_env_auto_reset_after_notification_10prd.md](097_env_auto_reset_after_notification_10prd.md) §5 |
| [322_env_individual](../wiki/categories/322_env_individual.md) | 速報値と確定値を別の鍵名に分ける形 | 却下 | 不採用 | [0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) Considered Options |
| [322_env_individual](../wiki/categories/322_env_individual.md) | ゲーム別RAM番地・畳み方をコードの固定表に持つ形 | 却下 | 不採用 | [0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) Considered Options |
| [322_env_individual](../wiki/categories/322_env_individual.md) | RAM番地またはラベルをmetricsキーへ直接出す形 | 却下 | 不採用 | [0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) Considered Options |
| [322_env_individual](../wiki/categories/322_env_individual.md) | frame skip窓のあとだけRAMを観測する形 | 却下 | 不採用 | [0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) Considered Options |
| [324_dqn](../wiki/categories/324_dqn.md) | ImageClsのTrain Actorにも遅延snapshotを適用する形 | 範囲外 | 不採用 | [036_train_actor_periodic_snapshot_10prd.md](done/036_train_actor_periodic_snapshot_10prd.md) 対象範囲 |
| [324_dqn](../wiki/categories/324_dqn.md) | ObservationPipelineクラスと専用ファイル・テストの追加 | cut（後続PRDへ） | 範囲外 | [083_action_context_retire_10prd.md](083_action_context_retire_10prd.md) 複雑さ監査 |
| [324_dqn](../wiki/categories/324_dqn.md) | 独立した公開NoisyLinearブロック型 | 採用しない | 不採用 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §4.1 |
| [324_dqn](../wiki/categories/324_dqn.md) | AMPの実行設定への集約とノイズ込み重みのSN | 対象外 | 範囲外 | [084_noisynet_10prd.md](084_noisynet_10prd.md) §8.1・§8.2・§14.2 |
| [324_dqn](../wiki/categories/324_dqn.md) | 購読間隔ごとの別EMA | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [324_dqn](../wiki/categories/324_dqn.md) | head単位統計とattention entropy最小値 | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5・§7 |
| [324_dqn](../wiki/categories/324_dqn.md) | TQLのattention entropy正則化 | 先送り | 範囲外 | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §7 1 |
| [324_dqn](../wiki/categories/324_dqn.md) | ReplayBufferから別に一様抽出する勾配ノイズprobe | 棄却 | 不採用 | [0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md](../adr/0054-gradient-noise-scale-in-band-microbatch-split-includes-bf16.md) Considered Options |
| [324_dqn](../wiki/categories/324_dqn.md) | 未成立理由ごとの追加状態指標 | cut | 不採用 | [092_gradient_noise_scale_10prd.md](092_gradient_noise_scale_10prd.md) §10.2 |
| [324_dqn](../wiki/categories/324_dqn.md) | attention entropy専用のfeatureキー | 作らない | 先送り | [093_attention_entropy_diagnostics_10prd.md](093_attention_entropy_diagnostics_10prd.md) §5 |
| [324_dqn](../wiki/categories/324_dqn.md) | Actor設定キーでeval_panelだけにhintを生成させる形 | 採らない | 不採用 | [0058-approx-priority-column-computed-by-agent-step-diagnostics.md](../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) 本文 |

### 追加修正の自己点検・検証

- `.venv/Scripts/python.exe .scratch/prd095/stage3-verdict/validate-verdicts.py`: 546項目成功、失敗0。全16地図97項目の裁定、元の形・理由・出典の保持、再開条件4件、各節・出典リンク・目録・公開できない値、改行/BOM、追記のみのlog、保護対象と無関係ファイルのhashを確認した。
- `git diff --check --`（今回の15ファイル）: 終了値0。READMEの裁定規則はPRD §4.6の該当ブロックと完全一致。PRD・ADR・CONTEXT・categories・正本の材料への変更0。
- 実装品質確認観点の全62項目を適用範囲に従って自己点検した。計画、PRD適合、現用規約と全地図の移行、履歴保持、検証と報告を確認。コード・設定・runtime・skillの変更は無いため、それら固有の観点、Debugビルドとコードテストは対象外。文書修正のためTDDも対象外。
- 新規の未決、判断待ち、未実施の必要検証は無し。scanの最終結果と独立レビュー結果は、完了後にこの節へ追記する。

- `.venv/Scripts/python.exe reports/tools/wiki_support.py scan`: 終了値0、未取り込み0・リンク切れ0・目録漏れ0（stage3-verdict/scan-final.json）。`eligible: false`は完了可否ではなくlint候補条件であり、未取り込み等の件数とは別である。

- 追加検査18項目も成功。関係欄のリンク、ユーザー名・マシン名・私的アドレス/認証情報を走査し、外部URLも0件。関係欄のラベル付きリンクと「なし」を認める既存書式に検査式を合わせた（成果物の変更は無し）。

### 追加修正のレビューと最終結果

- 独立した `an-review`（verdict_review）へ計画と規約を依頼し、1回のレビューで両方OK。修正要求・範囲外の検出・開発者への確認事項は全て無し。レビュー原文: `.scratch/prd095/review-20261023-0306.md`。
- 再確認ポイント: 地図本文の意味の独立再審査ではなく、D4に従って作者2担当の出典照合と機械検査を確認した結果である。564項目成功とscanの各0件は作者の実行結果であり、レビュー担当の再実行として扱っていない。規約は改訂PRDと同文で、設計文書から規約への参照にも追随漏れ無し。根拠が無く見送った指摘は無し。
- 全16地図97項目の再判定が完了し、12地図72項目を変更、再開条件4件を補完した。不採用61・先送り16・範囲外20。規約、地図12枚、log、22implの15ファイルを変更した。logは最新PRD095 blobのfix行を追記しただけ。
- 未決・起票候補・判断待ち: なし。未実施の必要検証: なし。コード変更が無いためDebugビルド・コードテストは対象外。過去工程のmemory未移行・Claude側作業の残件は、この完了に含めていない。
- 最終hash・mtimeと開始時との差分をstage3-verdictへ保存し、再読して保持を確認する。PRD・ADR・CONTEXT・categories・正本・別作業の未コミット変更は保持。stage・commit・pushは行わず停止する。
