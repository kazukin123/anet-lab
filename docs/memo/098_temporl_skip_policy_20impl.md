# TempoRL 対応 実装メモ

## 概要

正本は [PRD098](098_temporl_skip_policy_10prd.md)。DefaultDQN に、選んだ行動を何 step 続けるかを学ぶ skip Network、Actor の lane 別状態、可変長 replay 窓、skip loss、探索と観測を実装する。既定は無効。無効時の forward・乱数・sampling・checkpoint を維持する。

2026-10-23 の依頼で、初期確認以後の実装上の判断はエージェントに委任された。初回実装の実行範囲は全機能の実装、Debug ビルド、単体テスト、段0の等価性 Run、短い smoke・性能検証。段1〜4の長時間学習は設定と判定手順を用意し、未実施として残す（ユーザー回答）。学習効果の実証まで完了したとは報告しない。2026-10-24 round2の直接依頼で、GridMaze seed1・500kの分離stop/包含stopを取り直す範囲が追加された。現行契約と追加検証の正本は [22impl](098_temporl_skip_policy_22impl.md)。

## 未決事項監査

ユーザー判断が必要なブロッカー: 0。

| 決定点 | 分類 | 根拠・固定内容 |
|---|---|---|
| スコープ | repo evidenceで解決済み | PRD D1/D2/D17。6機構を実装し、NEET抑制・forward省略・SMDP・Qのn-step置換は行わない |
| Network の出力と所有 | repo evidenceで解決済み | PRD §5.3、ADR0062、既存 `NetworkModel` / `DefaultDQNAgent::CreateActor`。primary の出力鍵を維持し、skip のonline/targetとActor複製を同時同期する |
| replay と境界 | repo evidenceで解決済み | PRD §5.1/5.2、ADR0063、`ReplayExperienceStorage` / `ValidIndexManager`。readyとsampleableの遅れを分離し、PER完成とevictionは従来どおり |
| loss・探索・設定 | repo evidenceで解決済み | PRD D5〜D9/D14、§5.4〜5.7。TD/QR/IQN、分離/包含、stop/merge、q_shared、専用方策、priority_lambdaを全て実装する |
| checkpoint | repo evidenceで解決済み | PRD §5.8。無効の既存形を保持。有効時は4 networkと役割別optimizer状態。無効checkpointから有効への分岐を許しWARN、有効から無効はfail-fast |
| 検証範囲 | ユーザー判断で解決済み | 初回は全機能と短い検証、長時間は準備のみ。round2で指定されたGridMaze seed1・500kの2腕だけを追加実行 |
| 実装配置・テスト順序 | 2ximplへ固定 | 既存機能グループ内に置き、下記の縦スライスごとに1テスト→RED→最小実装→GREENで進む |
| 強制行動 | PRD範囲外 | PRD D10/§9の許容を維持。WithAction後のActor状態更新hookは追加しない |

## 主な変更

1. `core/anet-core/include/anet/rl.hpp` と `src/rl.cpp`: `ExperienceSamples` に window の長さ・最大長・割引報酬・終わり方・次観測を追加し、転送と tensor 列挙に含める。共通層はwindow/decisionという語だけを使う。skip の更新指標は既存の具体的な結果型・scalar 宣言の責任境界に合わせる。
2. `include/anet/replay_buffer.hpp`、`src/replay_buffer_impl.hpp/.cpp`: 1 step の reward・終わり方・decision を保持。窓規則を構築時に渡す口を設け、DecisionWindowRule が判断境界・終端・J上限・action一致を検証する。抽出はsamplerと同じRNGからkを一様抽選し、現行stack規則でk先の観測を組む。規則無しではwindowを生成せずRNGを増やさない。全sampleable consumerへ未来marginを適用し、ready、PER完成、evictionはn_step基準を維持する。prefetchは追加tensorも転送する。
3. `src/nn_impl.cpp` と近傍テスト: Headの出力へBody公開出力を合わせる。BodyのTensorを浅く保持し、Head用のFP32変換とは分離する。既存の透過Headが同じ鍵を返すため、既存Headの公開鍵を優先し、未公開のBody鍵を追加する。
4. `include/anet/default_dqn_agent.hpp`、必要なDQN設定DTOを持つ`include/anet/agent.hpp`、`src/default_dqn_agent.cpp`、`src/dqn_based_agent.hpp/.cpp`、`src/dqn_based_heads.hpp/.cpp`: SkipConfig/SkipHead/SkipValueReader/SkipLengthSelector/SkipTargetBuilderを既存機能グループ内へ配置する。配置・内部型の調整はPRD §5.11の許可に従い理由を本メモへ残す。公開ヘッダの変更はこれらと上記replay/rlに限定する。
5. NetworkModel はprimaryに加えてskipの対を保持する。skip入力specは参照されるprimary公開出力・obs・action・tausだけから構築する。Headのshape、IQNのrank3特徴と有効時のaction bind整合、認識した無効設定値を検証する。無効時はskip網を構築しない。skipのNN実行状態は用途別に所有し、primaryと同じnn_forward_keyを使用する。GraphVizを別グラフで出す。
6. Learner は既存target計算を再利用し、窓のR・k・終端を通す。currentのfeature seamでstop/mergeを分け、総lossのbackwardは1回、clipはNetwork別とする。GNSはprimaryのQ lossのみ。optimizerはprimary通常/σとskip通常/σの役割を持ち、保存読込・options適用も役割で対応付ける。skip有効かつFP16 AMPは構築時に失敗する。
7. Actor は全laneのprimaryを毎step計算し、独立乱数でskip長を選ぶ。状態はActorが所有しepisode_startで破棄する。clone/shared・online/targetの組を揃える。上書き後のactionでQヒントを再gatherし、decision/infoとskip auxを返す。既知scalarの未成立はNaN、未知だけnulloptとする。
8. `apps/runner/config/` のDefaultDQN共通定義とGridMaze/LunarLander/Atari/DropMerge: agent.txtの@baselineへ無効既定と共有配線を置き、envのJ・融合済み特徴・run.@skipのenabled/指標/Traceを部分上書きする。変種はCLIの葉とする。9種のscalarと任意のskip_q.[j]列を使う。共通・環境の既定だけ`?=`、選択と実験腕は`=`。tagは40文字以下。
9. `core/anet-core/testdata/prd098/`: 再現可能な段0、smoke、性能比較、段1〜4の設定と判定手順を置く。ログと一時Runは `.scratch/prd098/` に保存する。既存Run artifactは変更しない。
10. `docs/design/110_agents_and_learning.jp.md`、`130_neural_networks.jp.md`、`140_observability.jp.md`、`150_replay_buffer.jp.md`、`200_dqn_agents.jp.md`を追随させ、テスト章から検証資材へリンクする。用語とADR0062/0063はPRD起草側の既存変更を尊重する。現行wikiと出典の不一致を見つけた場合だけ、wiki規約に従い修復とfix記録を行う。
11. `src/observers.cpp` と `observers_test.cpp`: PRD指定の `50_agent_skip/03_skip_length_rate.[j]` を出力するため、scalar tagの終端を内側の添字と取り違えないよう修正する。実Runnerのdefs照合で発見し、tagが途中で切れるREDを確認した。汎用層へskip固有名の分岐は足さない。

## テスト

Public interface / surface: `CreateReplayBuffer`→`Push`→`Sample`/`SampleUniqueUniform`/`ProbeSamplingHistory`/`Size`、Network構築とForward、DefaultDQN factory、ActorのMakeAction、LearnerのUpdateFromBatch、Save/Load、GetScalar、runnerのconfig/metrics/trace。

優先behaviorとTDD順序（各行の内側でも1behaviorずつRED→GREEN）:

1. tracer bullet: 1laneの既知報酬列をdecision境界付きでPushし、公開Sampleから同じ区間内のk・割引報酬・k先の観測を得る。無効経路の既存テストを併走する。
2. replay: done・truncated/dummy・ring折返し・stack・J=1/上限・decision再開・action不一致・kの分布・容量下限・PER完成とsampleableの分離・probe/prefetch転送。
3. NN: Body公開出力がHeadと共存しdtype/勾配を保つ。skip HeadのTD/QR/IQN、分離/包含のshape、skip.head.use_dueling=trueのj=1一致とdetach、skip_q_distの分位軸。
4. factory/config: 無効既定、enabledの必須net/J、未知列挙値・不正数値・有効時の不整合bind/不正rank・FP16のfail-fast。primaryはhead.use_dueling（既定true）、skipはskip.head.use_dueling（既定false）のbool。
5. Learner: 小さい数値例でk別target、done bootstrap遮断、Munchausen各mode/TBO/分布型、IS重み、PER λ、stop/merge、Network別clip、primary限定GNS。
6. Actor: j回継続・episode_start・decision/aux・ヒントの再gather、方策追従/専用方策、clone/shared、target組と同期。
7. checkpoint: 無効の既存形、有効の4object、無効から有効のprimary optimizer引継ぎ、有効から無効の拒否、役割別options適用。
8. metrics/trace: 9種の定義と値、判断無しstepの未成立、decision/lengthと任意のskip_q.[j]のlane別読取。無効のRNG/checksumを段0の4構成で比較する。

GREEN後だけ必要な整理を行い、関連テストを再実行する。テスト専用production APIは作らない。近い既存test fixtureを使用する。

## 検証

着手前に現在のHEAD・dirty一覧・触るファイルの内容/hashを `.scratch/prd098/` に保存し、既存バイナリではなく現在ソースのbaselineを確認する。変更前Runの設定と結果を保存して、変更後に同じ設定で比較する。checksum対象と揺らぐ時間指標の扱いは既存検証資材に従い、比較対象の除外を記録する。

MSVCのビルドはAGENTS.mdの規約に従う。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
```

関連テストは `core/anet-core/bin/Debug/anet-core-test.exe` の既存タグと追加したPRD098タグで実行する。必要なbaseline・smoke・性能比較の詳細コマンドは `core/anet-core/testdata/prd098/README.md` に記録する。設定検査はPRD072の `check_default_leaves.py` と全キー・文字列値比較を使用する。Pythonは `./.venv/Scripts/python.exe` を使う。

性能はPRD §5.10の約1.2倍見込みに対し、Atari BTRの短いwarmup後の学習更新で1.3倍以下を確認する。sampleの追加stack/転送は約1.5倍のデータ量になるため、prefetchを含む実経路で測る。行動・学習の計算経路には逐次CPU/GPU往復を増やさず、Actorの状態はdevice上で処理する。指標購読時の追加同期は次段落で頻度と所要時間を見積もる。Debugのskip勾配診断だけは例外として、購読とは無関係にoptimizer適用前に1回/update同期する（追加レビュー対応節）。詳細ログを保存し、失敗・未実施は成功と区別する。

指標の読み取りでは、Actorの結果ごと・Learnerの更新結果ごとに初回参照だけGPU→CPU同期が入る。未購読なら0回、既定のskip profileならActorはtrain stepごと1回、Learnerはupdateごと1回で、key数・Trace列数を増やしても同じcacheを使う。初回実装時の `[prd098-readback]` のDebug/CUDA実測（N=64、J=10、warmup20回＋測定200回）はActorの集約・readback・全j読取が平均1.046ms、Learnerの5値pack読取が0.077ms（`88-review-tests.log`）。Atari短縮性能条件はN=128/B=256/RR=1なので、Actorは約2回/update、Learnerは1回/update。N増分を含む保守的な事前見積もりをActor 2ms/回、Learner 0.1ms/回、合計4.1ms/updateと置き、全体の実時間比較で受入を判定する。以上は初回実装時の記録である。round2のQ同梱cacheはActor N=64/J=10で2.101ms、Learnerで0.0955msを測定し、同じ条件への事前見積もりをActor4.3ms/回・Learner0.1ms/回、合計8.7ms/updateへ更新した。計測条件と結果は22implに記録する。これは計測単位の見積もりであり実時間の上限保証ではない。通常の計算経路では指標用readbackを購読時に限定する。Debugのskip勾配診断は指標と別の例外であり、未購読でも1回/update同期して非有限勾配を適用前に検出する。

自己点検では `docs/quality/impl.md` の全項目を適用範囲に沿って通し、差分・LF/BOM保持・既存dirty差分の保持を確認する。独立コンテキストの `an-review` に計画と実装を依頼し、指摘を修正して同じagentへ再提出する。応答原文は `.scratch/prd098/review-*.md` に保存する。

## 前提

- PRD §6の26項目を採用する。追加の利用者向け機能・互換層は作らない。
- `skip.enabled=false` を維持。max_lengthの実装既定は未指定0（有効時必須、明示0以下は無効でも拒否、明示1可）。feature_gradient=stop、loss_weight=1、priority_lambda=0、network_mode=action_out、skip.head.use_dueling=false。
- skip targetはActorだけが使い、Learner bootstrapはprimaryだけを使う。
- replayの1step記録は常時保存。window未指定のsample列・RNGは変えない。
- 既存の未コミット変更を戻さず、本作業の編集を局所的に追加する。git add/commit/pushは行わない。
- round2受入6のGridMaze seed1・500kの分離stop/包含stopを実行する。他の長時間学習は準備して未実施として残す。

## 2026-10-24 round2への追随

現在の設定契約と追加実装の正本は [22impl](098_temporl_skip_policy_22impl.md)。round2依頼に従い本書の主な変更・テスト・前提を更新した。SkipNetworkDefinitionはnetwork_config/head_factory、構築引数はskip_definition、Actor::ApplySkipはproposal/proposed_actions/actionsを使う。skip有効時のshape dumpはskip網とそのパラメータ数も示す。QR/IQNのGraphVizは分位平均skip_qと分布skip_q_distの軸を区別する。

round2ではDebug/Releaseビルド、core全体770成功・1skip、5環境101ケース、Python15件、最終PRD098関連29ケースが成功した。7環境の想定外設定差分0、OFF4環境78,278点完全一致、CLIの17smokeとログ/DOT再検査が成功。GridMaze seed1・500kの両型は最終+0.93／8step。ただし包含型の最初の正報酬は311,296exp stepで、分離型251,136より60,160遅く、平均報酬も悪かった。同等の学習速度とは報告しない。結果・未実施・正式レビューは22implの検証節を参照する。

以下は実施当時の経過・検証記録であり、過去の設定名と結果を保持する。round2の検証結果は22implと `.scratch/prd098/config-20261024-r2/` を参照する。

## 経過

- 2026-10-23: PRD・skill・品質観点・用語・ADR0062/0063・関連コードを照合。短い検証までの実行範囲を確認。計画を保存し実装の正本とする。
- Replayのwindow抽出を実装。判断境界・割引和・終端・truncated・ring折返し・stack・CPU転送の公開経路テストを追加。既存replay/transfer回帰は103ケース・1664 assertions成功。
- Body公開出力とskip Headの最初のshape契約を実装。未実装API／出力不在のREDを確認後、追加テスト4ケース・113 assertions成功。fixtureが恒等変換なのを2倍と取り違えた期待値と、Catch::Approxの名前空間不備はテスト側の修正として記録する。
- skip Headは射影とshapeを所有し、q_sharedのprimary基準値との合成はActor/Learnerで共用する読み取り境界に置く。primary Qへのdetachを一箇所で保証し、skip BodyへQ用の予約入力を増やさないため。
- Actorのlane別継続、episode開始時の再判断、独立RNG、clone/sharedとonline/targetの同時選択を接続。Policyのforward結果・tausはActorまで一時保持し、返却前に解放する。
- Learnerのwindow target、skip loss、IS重み、PER合成、1回のbackward、Network別clip、primary限定GNSを接続。Munchausen target方式はcurrent B + n-step終端 B + window終端 Bの3B、他方式は終端2Bを共用する。
- `.scratch/prd098/40-learner-green.log`: PRD098タグ7ケース・116 assertions成功。TDのpublic Agent/Learner経路でfinite skip lossとk=1を確認。QR/IQN、各mode組合せ、checkpoint、Run受入はまだ未検証。
- checkpoint の4 Networkと役割名付きoptimizerを接続。`47-checkpoint-green.log` はPRD098/091の10ケース・3184 assertions成功。旧checkpointからskip有効へのmoment引継ぎは、同じ入力に対する次のprimary更新で別途検証する。
- `51-metrics-green.log`: Actorの4指標と2つのTrace用scalarを追加し、9ケース・133 assertions成功。判断なしstepはNaN、jは1始まり、値はdevice側集約後に一度CPUへ転送する。
- `53-matrix.log`: TD/QR/IQN × Munchausen無効/target/online/online_reuse × 分離stop/包含merge の24条件が学習まで成功。包含側はq_shared・BF16・Noisy・TBOも有効。q_shared単独テストはfixtureのJ未設定を修正した。テスト中の型不一致と公開Agentをinner Learnerへcastしたfixtureの誤りは実装不備と分けて修正した。
- 無効なskipの初期化値と明示したnet定義も構築時に検証する。net定義を明示した無効構成だけは一時的に構築してshape/bindを検査し、そのResourceを破棄する。定義のない既定の無効構成は追加構築しない。
- 設定profileは `apps/runner/config/skip.txt` にまとめ、`_main.txt` から読む。NN blockも未選択profile内へ閉じ、無効の既定設定に葉を追加しない。7環境の全キー・文字列値一致は `config-equivalence.json` で確認済み。代入演算子検査は `62-config-leaves.log`: 27755 assignments、3491 audited、errors=0。
- 4環境の変更前Runはすべて正常終了（`56-baseline-runs.log`）。GridMaze 82.5秒、LunarLander 79.0秒、Atari 76.1秒、DropMerge 125.7秒。変更後Runの比較は進行中。
- 内部の読み取り・target・選択処理は状態を持たない小さい処理なので、専用の薄いクラスを増やさず、`ReadSkipValues`・`MakeLearningTargets`・既存ActionPolicyの値選択メソッドへ配置した。PRD §5.11に基づく配置調整。更新指標は汎用RL型でなく既存のDQN派生結果型が所有する。
- 初回独立レビューは計画OK、実装は専用IQN方策・未被覆の短い検証・設計文書内の旧説明・同期転送見積もりの4件で差し戻し。原文は `review-20261023-0625.md`。専用方策のqueryをprimary末尾へ追加し、同じunionを両Networkで1回ずつ評価する実装へ修正。平均・UQE・Thompson・spatialの公開ActorテストはREDからGREENを確認した。primaryの選択・既存診断は元のquery範囲だけを使う。
- 初回smokeは4環境ともRunが正常終了したが、後から追加したdefs照合で添字付きtagの末尾 `]` 欠落を発見した。`90-numeric-tag.log` でRED、`92-tag-numeric.log` でtag単体GREENを確認。最終smokeでは9種類すべてのdefsと値を照合する。
- 追加した数値fixtureでは、IQN入力spec、PERの実在slot、公開結果のprimary TDと合成priorityの区別を補正した。production targetの数値不一致は検出していない。各修正のビルド・失敗記録は76〜94のログに保持した。

## 検証・レビュー結果

baseline と最終実装の全 Debug ビルドが成功（`00-baseline-debug-build.log`、`95-final-debug-build.log`）。

段0は最終版の4環境で学習 scalar 全78,278点が完全一致（`97-final-equivalence.log`、各 `after2_ENV/comparison.json`）。比較するバイナリと設定の hash は各 `execution.json` に記録した。after2はproduction最終変更をビルドした93のsnapshotで、95との差分はテストだけである。初回比較の `66-after-runs.log` も同じ結果だった。

| 環境 | 一致した点数 | SHA256 |
| --- | ---: | --- |
| GridMaze | 20,095 | `54cb0e97abb891ac9799018ee5dec88d7de014e536d24cde7704840e44034ea8` |
| LunarLander | 16,494 | `9ea069dbb542d46971b111524778d10c29c6cf7df093b2401879312a207e7edf` |
| Atari | 18,442 | `cfd00ecc860aa8098ae5fd0b8c96836cc4c55a98954e7b63f14555f771eb9b66` |
| DropMerge | 23,247 | `08c721d17951a9263a7230c40f10deadb2d8fd36d69b4b725f96922a933d03b3` |

最終回帰は `[prd098],[dqn],[munchausen],[nn],[replay_buffer],[transfer],[observers]` の487ケース・158,711 assertionsがすべて成功（`96-final-regression.log`）。下表のレビュー後追加検証、24条件matrix、checkpointのprimary optimizer moment引継ぎと次更新の一致（Noisy有無）、一様抽選4,000件、容量・境界、Trace、無効設定を含む。初回の広い回帰も454ケース・158,286 assertions成功（`73-regression.log`）しており、その後の修正を96で再検証した。

7環境の既定の全キー・文字列値が一致（`config-equivalence.json`）。代入検査は27,759 assignments・3,491 audited・0 errors（`71-config-leaves.log`）。自己点検の機械確認でBOM保持・LF・diff checkが成功（`change-audit.json`）。設計文書150の既存dirty変更を保持し、それ以外の既存dirty 48ファイルは着手前のdiffと一致（`preserved-dirty.json`）。

最終版のsmokeは4環境すべて正常終了し、9種類の必要なtagがdefsと出力の双方に存在し、全skip指標が有限であることを確認（`98-final-smoke.log`、各 `smoke2_smoke_ENV/skip-metrics.json`）。GridMaze 11,947点、LunarLander 10,141点、Atari 15,013点、DropMerge 14,293点。初回smokeで見つけた添字付きtagの欠落も解消した。包含型・勾配共有型の4環境8設定は全て短いRunで正常終了（`80-profiles.log`）。75の失敗は検証側が未定義eval tagを無効化しようとしたもので、fixtureを修正して再検証した。

レビューで求められた短い検証の対応:

| behavior | 検証 |
| --- | --- |
| 専用IQNの平均・risk・spatial | `Dedicated IQN skip policy queries its own mean or risk range in one shared forward`。primaryとskipの選択、同一taus、各forwardが1回を確認 |
| Actorの上書き・hint・clone/shared同期 | `Skip Actor retains an action and resets duration on episode start`。2行動の順位を反転、継続中のhintと同期後のskip長を確認 |
| window target・done/truncated・Network別clip・hard/soft target組 | `Skip target uses window return and duration and masks only terminal bootstrap`。手計算値、両勾配norm、両targetの同期値を確認 |
| Munchausen3mode・TD/QR/IQN・TBO・IS重み・PER λ | `Skip Munchausen targets match numeric returns for TD QR IQN and TBO`。18条件の手計算target、合成priority、TDのIS付きlossを確認 |
| window付きprobe・CPU/CUDA prefetch | `Replay windows stop at the next decision and return discounted transitions`。3実行経路で窓値とhistory群の候補数・転送先を確認 |
| PER完成とwindow sampleableの分離 | `Decision window margin does not delay PER initial-priority completion`。n_stepで10件のpriority完成、Jでsampleable8件を確認 |

初回実装の独立再レビューは計画・実装ともOK（`review-20261023-0641.md`）。前回4指摘は解消し、成果物への未充足はない。判定時点で実行中だった検証の終了結果を本節へ追記した。最終全Debug・関連回帰・段0・defs付きsmokeは95/96/97/98で完了。

初回実装のAtari短縮性能比較はOFF→ON→ON→OFFの順で測定し、全4 Runが正常終了した。ON/OFF平均比 **1.298989倍** で受入条件の1.3倍以下を満たした（`100-performance.log`、`perf1_performance.json`）。N=128、B=256、BF16、prefetch有効、learn_step 64〜256の区間を使用し、他のビルド・テスト・学習を重ねなかった。結果は次のとおり。境界に近いため、十分な性能余裕や他条件での同じ比率を保証する結果とは扱わない。

| 順番 | skip | 秒/update |
| --- | --- | ---: |
| 1 | 無効 | 0.258979 |
| 2 | 有効 | 0.336693 |
| 3 | 有効 | 0.338010 |
| 4 | 無効 | 0.260427 |

依頼された「実装と短い検証」は完了。以下の長時間学習とPRD範囲外の強制行動hookを、完了した範囲へ含めない。

段1〜3の24設定を `long-config/` に生成した。段4は `run_validation.py prepare-long --checkpoint` で、選定した50M checkpointから同じ初期状態の2腕を生成できる。checkpoint未指定のため段4の実行設定は未生成と `long-config/status.json` に記録した。長時間学習は全段未実施。

## 追加レビュー対応（2026-10-23 12:19 Claude）

ユーザーの「すべて修正」に従い、中2件・低4件を対象とする。対応前の8ファイルと、後述の性能修正前のReplay2ファイルは `.scratch/prd098/claude-fix-baseline/` に保存した。ブロッカーは0。長時間学習は引き続き実行しない。

1. skip有効のFP32/BF16更新で、primary・skipの勾配NaN/Infをoptimizer適用前に検出する。publicな `TDLearner::UpdateFromSamples` へ勾配hookを仕込み、各Networkへ非有限勾配を注入するテストでRED→GREENを確認する。既存の2つの勾配normをまとめて1回検査し、同期をparameter数に比例させない。Debug・有効時のみ1回/updateの同期となる。2要素のfinite判定と真偽値のreadbackの追加費用を事前に約1ms/updateと仮置きし、GPU完了待ちを含む実際の全体性能は追加後に実測する。この値は上限保証ではない。SetupOptimizerの既存verboseログも有効時に出す。
2. 強制行動の将来接続条件をPRD098・観測パイプライン単位A・操作入力PRDへ記載する。trainで区間内の行動が変わるとSample時にfail-fastするため、強制したstep自体を `decision=true` とし、そのlaneのskip状態もリセットする。
3. 初回のOFF→ON→ON→OFF順を本文でも明記し、ON→OFF→OFF→ONの追加測定と、変更前・変更後OFFのABBA比較を同じ短縮条件で実施する。元の結果を上書きせず、各順序・秒/update・比を残す。
4. 無効な明示skip.net定義を構築・検証・破棄した際に、INFOを1回出す。定義のない既定無効時には追加処理を行わない。
5. OFF checkpoint→ON読込のprimary optimizer group数を、Torchの既存metadataから読んで本体復元前に検査する。DQN cpp内の専用Readerで現行Torch形式を読み、保存形式や汎用archive APIは変えない。`Learner::Save/LoadCheckpoint` のpublic経路でNoisyの有無が異なる2方向をRED→GREENで検証し、実際のgroup数と期待数を診断する。
6. PRD098 §5.11に `ReadSkipValues`、`ActionPolicy::SelectFromValues` / `MakeValueQueryTaus`、`Learner::MakeLearningTargets` の実装名を添える。

追加測定 `perf_reverse1` はON→OFF→OFF→ONで1.321倍となり、1.3を超過した。検査は維持し、CPU int64の離散行動について窓内一致を既存の連続storageから直接比較し、rowごとのTensorスライスを省く。連続行動・他device/配置は既存のtorch::equalを使う。抽出時のwindow length/max lengthも既存CPU int64配列から読み、値・境界・RNGを変えずにselect/itemを省く。`replay_buffer_impl.cpp` と同テストを対応範囲へ追加し、離散/連続両方の行動不一致診断と既存の窓・転送回帰を検証する。最初の8Runが終了するまではbuild/testを行わず、その後に最終版を逆順で再測定する。

関連Debugビルドと契約テスト、文書自己点検を行い、PRDはPRD観点、実装と計画は実装観点へ別々に独立レビューを依頼する。実行結果は本節へ追記する。

- NaN/Inf: `111-nan-red.log` で8条件・16 assertionsが失敗。修正後 `113-nan-green.log` ですべて成功。primary/skip × FP32/BF16 × NaN/Infを検証した。
- checkpoint: `115-checkpoint-red.log` で2方向とも失敗。`116-final-debug-build.log` の全Debugビルド成功後、`117-contract-green.log` はPRD098/091の25ケース・19,871 assertions成功。保存数と期待数を表示し、既存のoptimizer moment引継ぎも通過した。
- 性能driver: `118-validation-driver.json` で逆順・変更前後OFFの腕の割当てと平均比集計を確認。実Runの最終結果は下記に記録した。性能受入は未達である。

追加結果:

- PRD3件の改訂は独立レビューOK（`review-20261023-1256-prd.md`）。実装/計画の再レビュー原文は `review-20261023-1301-impl.md`。同期説明の矛盾を解消し、設定入力の照合不足も下記で修正した。
- `124-cache-red.log` はinclude先だけを変えると旧Runが誤再利用されるRED。`126-cache-green.log` は同じ入力だけをREUSEし、include変更では新しいlabelを要求するGREEN。実Runnerは起動せず、同じexecute経路をfakeプロセスで検証した。生成run.txtに加えてconfigルート配下の全txtの相対名・内容を `config_inputs_sha256` として記録・照合する。現在のinclude先はこのルート内txtであり、未参照txtの変更も安全側に再利用を拒否する。
- 窓最適化後の全Debugビルドは `127-optimized-debug-build.log` で成功。`128-optimized-regression.log` は `[prd098],[prd091],[replay_buffer],[transfer]` の127ケース・21,486 assertionsが成功。離散int64と連続floatの行動不一致診断、窓・prefetch・probe・転送・NaN/Inf・optimizer引継ぎを含む。
- 最適化前の逆順測定 `120-reverse-performance.log` はON→OFF→OFF→ON、秒/update=[0.4253176914062351, 0.30662500066682696, 0.2962552331155166, 0.37134375306777656]、比1.321425716で**基準未達**。この失敗記録を保持し、最適化後を `129-optimized-performance.log` の `perf_reverse2` として新規測定した（結果は下記）。元の1.298989倍を現在版の受入根拠に流用しない。
- `121-off-comparison.log` は変更前OFF→変更後OFF→変更後OFF→変更前OFF、秒/update=[0.3039635717868805, 0.2688177104573697, 0.2807968470733613, 0.2770937222521752]、変更後/前比0.945887029。この測定ではOFFの実時間増加は観測していないが、統計的な速度改善を主張しない。使用バイナリは窓最適化前のsnapshotで、通常OFFでは窓規則を使わない。最適化後の通常OFFは同じ無効経路を保ち、抽出時のCPU配列の選択にnull判定2回/sampleが加わるだけである。

最適化後Debugの逆順測定 `perf_reverse2` も1.379493002倍で基準未達。ON秒/update=0.390046852/0.357343757、OFF=0.277953112/0.263833333。Debugの診断同期を維持したまま結果を失敗として保存する。PRDの性能条件はbuild構成を指定していないため、通常実行のReleaseでも同じB/N・BF16・prefetch・計測区間・ON→OFF→OFF→ONを測る。`--build-config` で明示選択し結果にも構成を記録、旧Debug結果を成功へ読み替えない。既存Debug baselineとの比較（before/after/compare/performance-off）は引き続きDebugに限定する。

最終追加検証とレビュー:

- ReleaseのRunnerビルドは `132-release-build.log`、終了コード0（`133-release-build-exit.txt`）。全体Debugビルドと関連回帰は127/128の成功を最終結果とし、性能Runにはビルド・テストを重ねなかった。
- Releaseの逆順4 Runは全て正常終了したが、`135-release-performance.log` / `perf_release1_performance.json` のON/OFF比は **1.511019841倍で基準未達**。秒/updateは順に [0.07035937378532253, 0.047708336933283135, 0.043510415707714856, 0.06747397128492594]。ON→OFF→OFF→ON、N=128、B=256、BF16、prefetch有効、learn_step 64〜256。判定stageは超過を検出して終了コード1（`138-release-performance-exit.txt`）。Debugより実時間は短いが、両構成とも現行版の性能受入は満たしていない。
- 構成選択の模擬main検査はDebug/Releaseのbin選択・report・集計を確認。追加レビューで段0のafterにも構成混在経路が見つかり、before/after/compare/performance-offを全てDebug限定へ修正した。`136-build-config-red.log` → `137-build-config-green.log` で再現と解消を確認。実Runnerは使わない。`134-build-config-test.json` に結果を保存した。
- PRD3件は独立レビューOK。実装/計画は `review-20261023-1321-impl.md` でOK、構成選択の追加を `review-20261023-1335-impl.md` の指摘後に修正し、`review-20261023-1338-impl.md` で両方OKとなった。性能の数値結果はレビュー時点で未完了だったため、レビューOKを性能受入成功と扱わない。
- Claudeレビューの中2件・低4件と独立レビューで見つかった修正は完了。性能測定の実施要求にも対応したが、現行版で1.3倍以下という受入条件は未達として残る。閾値・計測区間・勾配診断を弱めず、旧版の成功を流用しない。

## 決まらなかったこと

- 現行版の短縮Atari性能はDebug 1.379493002倍、Release 1.511019841倍で、1.3倍以下の受入条件を満たさない。性能改善と再検証が残る。独立レビューOK・機能テスト成功と性能受入を区別する。
- 長時間学習による効果検証は今回実行しない。未実施のまま実行設定・判定手順を残す。
- 強制行動後のskip状態はPRD D10の許容を維持し、別PRDの入力機構を待つ。
