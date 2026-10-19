# PRD061：Actor 設定カタログの実装計画

## 概要

評価スロットごとに方策・network を選択できるよう、P1「設定ルート改名」と P2「Actor 設定カタログ」を実装する。P3 の Env カタログ化は対象外とする。

本書を承認済み実装計画の正本とする。PRD061・ADR0038 の確定契約を採用し、PRD072 の現行 resolver を利用する。

## 移行前の基準採取

- 現行 Debug ビルドと関連テストを実行し、既存失敗を記録する。変更前の実行体・設定を保持し、新旧比較に使う。
- PRD072 の固定17入力を基に、全環境・代表プロファイルの解決済み設定と、既定補完後の Agent Module Config を採取する。PRD061 用の記録は別に保存し、PRD072 の golden は更新しない。
- Atari・DropMerge の checkpoint と固定観測バッチについて、greedy・fixed τ による Q 値と行動を保存する。入力・checkpoint・設定・ビルドの識別情報も記録する。
- 必要な基準を採取できない場合は、その受入項目を未検証として扱い、等価性確認済みとは報告しない。

## 実装内容

### P1：設定ルート改名

- Runner 設定を `train.*` から `run.*` へ移す。学習系は `run.train.num_envs`・`run.train.runner_type`、評価系は `run.eval`・`run.eval_schedule` とする。
- コード、テスト、現用設定、Run プロファイル、Python 補助ツール、現行ドキュメントを同時に移行する。
- 改名前後の解決値がキー対応表を除いて一致することを確認する。特に `evalonly`・`eval2only`・`pl_check` の評価周期、本数、並列度を固定する。
- P1 は独立して検証可能な変更単位とする。commit・push は行わない。

### P2：Actor カタログと実行経路

- `ActorRequest { batch_env_spec, env_spec, device, seed, actor_key }` を追加し、4 Agent の `CreateActor` と全呼び出し元・テストダブルを移行する。
- Agent が typed Actor カタログを保持し、Runner は `actor` の名前参照だけを渡す。省略時は train が `train`、configured eval がタグ名。EvalPanel は参照タグの指定を使う。
- 方策は Actor ごとに生成する。スケジュールは `MakeAction` に渡す学習側 counts で更新し、EvalRunner は `Sync(source_counts)` で保持する。評価イベント自身の counts は維持する。
- Actor seed は `actor/<Runner名>` から派生する。RunMode 別共有 RNG と Agent 側の用途分岐を削除する。
- `Actor` を Module 化し、`Runner::GetActor()`・`EventField::ACTOR`・`$actor` を追加する。既知だが未成立の scalar は NaN、未知キーは `nullopt` とする。
- clone・network・device 検証を Agent 側へ集約する。未定義 Actor 参照、不正 network、shared の device 不一致、MuZero の clone 要求は fail-fast。dormant スロットでは Actor を生成・参照解決しない。
- DefaultDQN の IQN spec K は Learner の N から作る。`use_optimistic_target` のコピー元だけを `actor.[train].policy` へ変更し、既存の強制値・明示上書き順序を維持する。
- Rainbow の eval 方策分離、MuZero の温度・noise、ImageCls の clone・bf16 を各 Actor 設定へ移す。

### 設定・文書の移行

- 実測した typed 実効値を新カタログへ移植し、不使用フィールドも含めて比較する。Rainbow eval 修正など PRD 所定の挙動変更は差分として明記する。
- eval タグを `eval_target`・`eval` へ改名する。metrics の出力タグ名は維持し、参照先だけを更新する。
- 旧 policy 設定、clone override、EvalPanel の `shared` モードを削除する。旧名の互換層・専用警告は追加しない。
- 現行ガイド・標準検証手順を同期する。過去 Run・実験記録・ADR の履歴本文は保持する。

## TDD と受入検証

各挙動について、1 テストの RED → 最小実装 → GREEN を順に進める。

1. **最初の縦断テスト**：設定から Runner が指定 Actor を生成し、行動と `$actor epsilon` を観測できる。
2. **参照と検証**：既定名・明示名、未定義参照、dormant、network 選択、clone/device、EvalPanel の参照。
3. **独立性と時計**：方策状態の非共有、学習側 counts、Actor 追加による既存行動列の不変性、同一構成の再現性。
4. **Agent 固有契約**：Rainbow eval、MuZero 温度・noise、ImageCls bf16、楽観 target のコピー・警告、`[train]` なしの IQN、異なる K の shared/clone 推論、実効設定 dump。
5. **統合検証**：Debug 全体ビルド、core 全テスト、変更した Python ツールのテスト、設定既定葉検査、旧契約の現用参照検索。
6. **実 Run**：Atari の4スロット ε smoke、DropMerge IQN32・QR51 の各スロット2セッション以上、EvalPanel 起動中の再現性確認。QR の IQN 専用 metrics は NaN を正常とする。

新旧の seed が変わるため、Run 全体の bit 一致は要求しない。設定比較・固定入力推論・metrics・性能を順に確認する。

## 性能基準と完了条件

- 同条件の旧版・新版を交互に各3回測定する。ウォームアップと測定予算を揃え、中央値とばらつきを記録する。
- **train step/s の低下率、eval 所要時間の増加率をそれぞれ5％以内**とする。超過時は原因調査に戻り、閾値を緩めない。
- 長時間の学習比較は通常の完了条件に含めず、前段で疑義が残った場合に PRD の追加検証へ進む。
- 未検証項目や再現性の不一致が残る場合は明示し、PRD 完了とは扱わない。無関係な作業ツリー変更を保持する。

## 実装・検証記録

P1 / P2 のコード・設定・現行ガイドを移行した。P3 は対象外。以下の artifact はリポジトリルートからの相対パスで、巨大な実行体・checkpoint・Run はローカルの `.scratch/prd061/` に保持する。commit / push は行っていない。

### 実装した契約

| 変更前 | 変更後 |
|---|---|
| `train.num_envs` / `train.main_runner_type` | `run.train.num_envs` / `run.train.runner_type` |
| `train.eval` / `train.eval_schedule`、その他 Runner root | `run.eval` / `run.eval_schedule`、`run.*` |
| `train_policy.*` / `train_actor.*` | `actor.[train].policy.*` / `actor.[train].*` |
| `eval_policy.*` | `actor.[eval].policy.*` / `actor.[eval_target].policy.*` |
| slot の clone override / RunMode による network 選択 | Agent の typed Actor カタログの `clone_model` / `network` |
| RunMode ごとに共有する Actor RNG | `actor/<Runner名>` の seed による Actor 専用 RNG |
| `$agent epsilon` / `$agent uqe_tau` | `$actor epsilon` / `$actor uqe_tau` |
| eval1 / eval2 の参照タグ | eval_target / eval。既存 metrics 出力タグ名は維持 |

`ActorRequest` を4 AgentとRunner・テストダブルへ適用し、Agentにカタログの読取・参照解決・clone/device検証を集約した。方策はActorごとに生成し、`MakeAction` の学習側countsで更新する。EvalRunner自身の評価countsと、`Sync(source_counts)` で受け取る方策用countsは別に保持する。EvalPanelも参照タグのActor設定と学習側countsを使う。

EvalPanelの学習countsは、学習イベントのsnapshotをmutexで保護してUIへ渡す。UIから学習Runnerの可変countsを直接読まない。network同期の前にコピーを取り、network同期中はcountsのmutexを保持しない。

意図した挙動差は、Rainbow evalがtrain方策を共有しなくなること、Actorの乱数系列がRunner名で独立すること、方策スケジュールを行動生成時に更新すること、IQNのdummy spec KをLearner Nから作ること。旧名の互換層は追加していない。

### 変更前と設定比較

- 基準commit: `2c7c0fe3b8660825a51538eff705fa976ad49eb6`。変更前Debugビルド成功。保存した実行体は `before/AnetRLRunner.exe` と `before/anet-core-test.exe`、設定は `before/config/`。
- 変更前core全テスト: 583ケース中581成功、2件は想定失敗。21454 assertions中21452成功。`before/core-tests.log`。
- P1: `before/typed/` と `p1/typed/` の固定17入力で、対応表を適用した全解決キー・文字列値とtyped値の差分0。`compare_p1.py` を使用。P1関連テスト127ケース2032 assertions成功。
- P2: `p1/typed/` と `p2/typed-final/` の固定17入力について、未使用フィールドも含む旧typed値の対応先との差分0。新規カタログフィールドは別記録。`p2/typed-comparison.json`、`compare_typed.py`。PRD072 goldenは変更していない。
- `evalonly` / `eval2only` / `pl_check` の周期・本数・並列度は設定解決テストで固定した。

### 固定入力推論

`inference-v2.json` と `inference-after.json` の入力に対し、online / targetのgreedy・fixed τを比較した。行動は完全一致、Q値はrtol/atol各1e-6以内。25 assertions成功（`inference-after-v2.log`）。固定観測と結果は `before/fixed-v2/` と `after/fixed-v2/`、SHA-256は `before/identifiers.json`。

Atariは既存の学習済みcheckpointを使用した。DropMergeの既存学習済みartifactは旧NN定義で、変更前の現行コードでも構築できなかったため、変更前実行体で現行IQN32定義のcheckpointを新規作成して比較した。これは初期化checkpointの等価性確認であり、**既存の学習済みDropMerge checkpointでの確認は未検証**とする。元のRun artifactは変更していない。

### 自動テスト

- 最初の縦断テストは設定 → Runner → 指定Actor → 行動と `$actor epsilon` を対象にREDを確認し、API移行後にGREENを確認した。
- 最終Debug全体ビルド成功。core全テストは593ケース中591成功、2件は変更前と同じ想定失敗。21612 assertions中21610成功（`p2-core-full-final.log`、exit 0）。
- 契約テスト `[prd061],[episode_steps_config]`: 11ケース236 assertions成功。既定/明示Actor、dormant、未定義参照、shared device、clone、学習counts、独立RNG、IQNの異なるKとtrain項目なし、Rainbow eval、MuZero温度/noise、ImageCls bf16、楽観targetのコピー・警告、実効設定dumpを関連テストで確認した。
- Python: `inspect_run_test.py` 72件、`optuna_workspace_test.py` 16件、`optuna_metrics_gzip_test.py` 1件、`check_default_leaves_test.py` 11件成功。既定葉検査は23144代入・3270検査対象でエラー0。

### 実Runと残る受入確認

- Atariの4スロットsmokeでε=0 / 0.01 / 0.1 / 0.5を観測した。最初の診断設定に未知scalar `q_std` がありWARNを記録したため、後続Runでは正しいIQN scalarへ修正した。
- DropMerge IQN32 / QR51は両スロット各3セッションを完走した。IQN専用scalarはIQN32で成立し、QR51ではNaNとして出力されないことを確認した。QR51 exit 0。Runは `smoke/runs/prd061_DropMerge_iqn32` / `prd061_DropMerge_qr51`。
- EvalPanel起動中の同一学習設定2 Runはexit 0、時間指標を除くscalar16件が一致した。checkpoint全体のハッシュは異なるが、設定とonline / targetのTorch archive全entryは完全一致した。差はoptimizerの保存領域（parameterのアドレス由来キーを含む）にあり、optimizer全体のバイト一致は確認していない。`smoke/repro-comparison.json` と `smoke/checkpoint-comparison.json`。
- 同一checkpointの評価専用4スロット反復と診断3群の追加Runは準備済み。性能受入は下記の測定環境確認待ちで保留している。

### 性能調査

最初のAtari比較（`perf/comparison.json`）は各3回、warmup 512 train steps、測定1520 stepsで実施した。中央値はtrain 20.274→17.490 step/s（13.73%低下）、online eval 0.2325→0.2650秒（13.98%増加）、target eval 0.2475→0.2825秒（14.14%増加）で、5%基準を超過した。

この比較にはevalタグ改名によるEnv seed・評価エピソード長の差が混ざっていた。旧新版のAgent typed設定とbackend設定は一致したが、この値を同一負荷の性能判定には使えない。再測定では両版の評価タグを `eval1` / `eval2` に揃え、新版は `actor = eval_target` / `eval` と明示する。これは任意のスロット名を使う測定設定であり、productionに旧名の互換処理を加えるものではない。元の結果は保持し、5%閾値は変更しない。

DropMergeのタグ・fixed τを揃えた比較でも、最初の1組はtrain 20.678→17.245 step/sだった。原因調査で、別の `apps/runner/bin/RelWithDebInfo/AnetRLRunner_ab.exe`（PID 41880）がGPU上にあることを確認した。そのRunが稼働中か休止中かは、この確認だけでは分からない。競合の可能性を排除できないため、性能受入は未達のまま保留し、ユーザーへ稼働状態を確認した。

こちらの測定制御プロセスと、その子のPRD061 Runnerだけを停止した。既存Runは停止・変更していない。`perf-dropmerge/execution.json` の3本は完走、`prd061_perf_dm_after_1` は途中停止（`perf-dropmerge/interrupted.json`）。途中結果を完走扱いにせず、再開時は新しい出力先で交互3組を取り直す。

### 初回停止時点の残作業（後述の再開検証で対応）

1. GPU上の既存Runが稼働中かを確認し、競合しない条件で性能測定を行う。Atariの初回基準超過を解決し、DropMergeの交互3組も完了する。閾値は各5%を維持する。
2. `smoke/run-acceptance.py` で、同一checkpoint・EvalPanel起動中の4スロット2 Runと、DropMerge IQN32 / QR51の診断3群を追加採取する。準備した設定は `smoke/checkpoint-repro-{1,2}.txt` と `smoke/DropMerge-*-metrics.txt`。比較は `smoke/analyze_acceptance.py`。
3. EvalPanelのcounts snapshot修正後のDebug全体ビルドは成功済み（`p2-ui-counts-build.log`）。修正後の実Run確認は前項に含める。
4. 最終結果を本書へ追記する。**初回停止時点ではP1 / P2実装済み、性能と追加実Runの受入未完了とした。**

### レビュー指摘 B1〜B3・S1〜S3 の修正

- Atari / LunarLanderのEvalPanel専用タグに `actor = eval_target` を明示し、旧Eval1のtarget network選択を維持した。
- GridMaze MuZeroは `run.eval_device_type = cpu` を明示した。MuZeroのCPU Agentとclone不可の契約に合わせ、device不一致を黙って無視する処理は追加しない。
- MuZero環境と共通full metricsの `tau` を `$actor` へ移行した。
- LunarLanderの休眠test1 / test2は `eval_target`、ImageClsのeval_fullは `eval` へのActor参照を明示した。
- 設計ガイド140 / 160 / 200、trace targetの両エラー文言を現行契約へ同期した。
- 共通Actor既定葉7行を `?=` へ変更し、既定葉検査にも対応する用途分類と回帰テストを追加した。

検証結果:

- Debug全体ビルド成功（`review-fixes-final-build.log`）。`[prd061],[observer_factory]` は22ケース867 assertions成功。
- 新規回帰テストは現用7環境×online/batchrunを解決し、train・全評価タグ（休眠を含む）・EvalPanelのActor参照とMuZeroのdevice / tauを確認する。EvalPanelタグやActor参照をテスト側で上書きしない。
- 既定葉検査のREDを確認後、分類を修正して12テスト成功。現用監査は23166代入・3276検査対象でエラー0。
- 固定17入力を再採取し、Agent typed値の差分0。全解決キー・文字列値の差分は今回追加したActor参照、MuZero eval device、tau sourceだけ（`review-typed-comparison.json` / `review-resolved-comparison.json`）。
- 実アプリ起動はAtari online / batchrun、LunarLander online、GridMaze MuZero batchrunの4本ともexit 0（`review-smoke/execution.json`）。短い予算・batch数・診断出力だけを調整し、EvalPanelタグ・Actor・deviceは現用設定のまま。MuZeroの `$actor tau` は1.0から減衰する4点を確認した。

前回smokeでEvalPanelタグを補正して起動不良を見逃したため、上書きなしの現用参照解決と実アプリ起動を今回の検証へ追加した。ユーザーから既存Run停止中・ビルド/テスト可能との確認を受け、残る受入検証を再開する。

### 再開後の実Run受入

`smoke/acceptance-execution.json` の4 Runはすべてexit 0、`smoke/analyze_acceptance.py` の比較も成功した（`smoke/acceptance-comparison.json`）。

- Atariは同一checkpoint・seedでEvalPanelを起動し、target greedy / online ε0.01 / online ε0.05 / online greedyの4スロットを各8セッションずつ実行した。2 Runの時間指標を除くscalar66件が完全一致し、各スロットの `$actor epsilon` は0 / 0.01 / 0.05 / 0。短いsmokeとしてepisode上限64 framesを用いた。
- DropMerge IQN32 / QR51は各スロット3セッションを完走し、UQE / NOOP margin、IQN診断、quantile crossingの3群を確認した。IQN32は23診断系列すべてに有限値があり、QR51はIQN専用の6系列（両スロットの53〜55）だけがNaN相当で出力されず、それ以外は有限値だった。両スロットのτは0.85、UQEのεは0。
- この実行にはEvalPanelのcounts snapshot修正を含む最終実行体を用いた。

性能の再測定は `perf-dropmerge-review/` と `perf-atari-matched/` の新しい出力先を使用する。前回の途中Runや結果は上書きしない。

DropMergeの再測定は旧新版各3回すべてexit 0。warmup 128 train steps、測定368 steps、評価は各Runのwarmup後3セッション/スロットを用いた。各Run内のeval平均を求め、その3回の中央値で比較した（`perf-dropmerge-review/comparison.json`）。

| 指標 | 旧版中央値（3回の範囲） | 新版中央値（3回の範囲） | 悪化率 | 5%基準 |
|---|---|---|---|---|
| train step/s | 31.298（29.188〜31.381） | 31.192（29.580〜31.335） | 0.339% | 合格 |
| online eval秒 | 0.1667（0.1600〜0.1933） | 0.1700（0.1700〜0.1767） | 2.000% | 合格 |
| target eval秒 | 0.1700（0.1667〜0.2067） | 0.1700（0.1600〜0.1767） | 0.000% | 合格 |

この性能確認はbatch 4、IQN32、fixed τ、短いepisode上限16 stepsの条件に限る。長時間学習の統計等価や全batch sizeでの速度を保証するものではない。

Atariの再測定も旧新版各3回すべてexit 0。warmup 512 train steps、測定1520 steps、各Run内のwarmup後12セッション/スロットの平均を3回の中央値で比較した（`perf-atari-matched/comparison.json`）。

| 指標 | 旧版中央値（3回の範囲） | 新版中央値（3回の範囲） | 悪化率 | 5%基準 |
|---|---|---|---|---|
| train step/s | 25.596（25.335〜27.785） | 26.750（25.797〜28.280） | −4.509% | 合格 |
| online eval秒 | 0.1808（0.1650〜0.1892） | 0.1717（0.1650〜0.1875） | −5.069% | 合格 |
| target eval秒 | 0.1967（0.1767〜0.2000） | 0.1775（0.1700〜0.1992） | −9.746% | 合格 |

既存Run停止の確認後、評価タグと負荷を揃えた再測定では初回の性能低下は再現せず、両環境のtrain / evalは5%基準を満たした。初回の差をタグ改名と測定環境それぞれに何%帰属できるかは分離していない。初回結果を消したり、閾値を緩めたりしていない。

### 最終状態

B1〜B3・S1〜S3の修正、上書きなしの現用設定検証・起動smoke、同一checkpoint再現性、DropMerge診断3群、性能受入を完了した。P3、長時間の学習比較、commit / pushは実施していない。固定入力推論については、DropMergeが変更前に作成した初期化checkpointによる比較であるという範囲制限を維持する。既存の学習済みDropMerge artifactでの等価性確認済みとは報告しない。


### D1修正：eval_targetの継承（2026-09-17）

`DefaultDQNAgent.@baseline.actor.[eval_target].$` をPRD §5.3どおり `DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target` へ修正した。現行resolverで循環は生じない。Atari / DropMerge / LunarLander / GridMazeの同値の複製71行（コメント含む）を削除し、eval側の最終値を継承する構成へ戻した。

- 現用設定を読む `[actor_inheritance]` 回帰テストを追加。CLIでevalのεを0.123に変更してもtargetが0.01に残るRED（10 assertions中1失敗）を確認した後、設定修正でGREENとした。target側の明示0.234が優先され、networkはtarget / onlineを維持することも検証した。
- Debug全体ビルド成功。`[prd061],[observer_factory]` は23 cases / 877 assertionsすべて成功。
- 固定17入力のtyped Agent設定は修正前と完全一致。全キー・文字列値の比較では削除したA1/A2/A3のtarget側重複葉だけが差分となり、Agentの実効葉は一致した。PRD072 goldenは変更していない。
- 既定葉検査は23,135 assignments / 3,215 audited / 0 errors、検査器12テスト成功。`git diff --check`成功。
- 証跡は `.scratch/prd061/d1-build.log`、`d1-red.log`、`d1-green.log`、`d1-final-typed/`、`d1-comparison.json`。この追補では実Run・性能測定は再実行していない。


### 残件1〜6とD2・D3の裁定（2026-09-17）

承認済み計画に従い、D2・D3はともに案Aを採用した。

- D2：target netの無いAgentではeval_targetをdormantにするPRD・CONTEXTの契約を優先し、ImageClsはevalをactive（interval 50）、eval_targetをdormant（interval 0）へ変更した。batch size・eval window、metrics RHS、両appモードのEvalPanel参照をevalへ移した。metrics LHSの51_eval1/*、Actorカタログのeval_target、eval_full.actor=evalは維持した。Env seed domainは `eval_env/eval_target` から `eval_env/eval` へ変わるため、移行前後の同一seedによる観測列一致は要求しない。
- D3：消費者のないActor独自のGetConfigDataとRunner別dumpは追加せず、既存Agent dumpの `actor.[key].*` を実効Actor設定の確認先とした。PRD §5.2・§7.3の2箇所のみ契約を修正し、状態行は追加していない。ADR0038・CONTEXTは変更していない。
- 残件1：Atariのgreedyコメントを継承による追随の説明に修正した。DropMerge・LunarLander・GridMazeに同種の両Actor明示指定コメントは残っていない。
- 残件2：agent.txtの名指し分類を削除し、common.txtと同じ一般規則へ統一した。先行分類は維持し、@target.networkだけを選択プロファイルとして `=` へ戻した。ImageClsの6葉は `?=` のまま。検査器の3群テストで@targetの誤分類をRED（2 failures）として確認後、12テストすべてGREENとした。
- 残件3：onlineは環境のapp選択を維持し、batchrunは環境の有効なappチェーンのonline項だけを置換する。P1へ検証値を置いたテストで従来の12 failuresを確認後、チェーン保持を検証した。7環境×2モードを維持し、CartPoleのP1無しにも追随する。
- 残件4：MuZeroのeval deviceをEval設定節へ移動し、clone非対応かつAgentがCPUである理由を追記した。

検証結果：

- VsDevCmd経由のDebug全体ビルド成功（`.scratch/prd061/remaining-build.log`）。
- `anet-core-test.exe "[prd061],[config],[observer_factory]"`：128 cases / 2,727 assertions、すべて成功（`remaining-tests.log`）。
- 既定葉検査：23,138 assignments / 3,215 audited / 0 errors。検査器12テストすべて成功。
- 新規採取した `remaining-before/` と `remaining-after/` の固定17入力比較：typed Agent設定の差分0。全キー・文字列値ではImageClsの3入力だけ各14キーが変わり、承認したschedule・評価設定・metrics参照・EvalPanel参照だけだった。他14入力は差分0（`remaining-comparison.json`）。PRD072 goldenは変更していない。
- ImageCls batchrun smoke：scratch内の2クラス256枚の小規模画像fixture、train並列度2、exp予算208を使用した。現用のActor・評価タグ・EvalPanel参照・評価周期50・評価batch 128・window 256を維持し、画像等の補助出力を抑制した。exit 0、所要32.08秒。`51_eval1/03_accuracy` はstep 2 / 102 / 202で有限値0.5を出力した（`remaining-imagecls-smoke/execution.json`・`acceptance.json`）。これは合成画像による起動・評価経路の検証であり、分類精度の評価ではない。
- 差分レビューと `git diff --check` を実施。commit / pushは行っていない。


### EvalPanelの共通Greedy既定（2026-09-19）

ユーザー要求により、EvalPanelの共通参照先を `eval_panel` タグへ統一し、同名の専用Actorを4 Agentのカタログに追加した。ADR0038の名前付きカタログ参照とEvalPanelのタグ参照方式を維持し、Runner/Agentの実行時APIは変更していない。

- DefaultDQN / Rainbowはtarget側Actorのnetwork・推論設定を継承し、右側の `@greedy` 差分でGreedy / ε=0にする。DefaultDQNではGreedyと併用できないUQE専用full queryも無効化する。MuZeroは温度0・noiseなし・cloneなしの既定葉を宣言し、ImageClsはeval Actorの最大スコア選択を使う。
- 共通の表示用Env設定はeval_targetから継承し、ImageClsはevalから継承する。Atari / LunarLanderの既存Env上書きは維持した。定期評価の方策・schedule・metricsは変更しない。既定値は設定側から変更でき、コードでGreedyを強制しない。
- D2で決めたImageClsの定期評価eval=50 / eval_target=0は維持する。今回、EvalPanel参照だけは専用eval_panelへ移した。参照タグが変わる環境ではEvalPanelのEnv seed domainも `eval_panel/eval_panel` へ変わる。Actor seed domain `actor/EvalPanel` は維持する。
- 共通設定・Agentカタログ・環境設定、現行ユーザーガイドと設計160を同期した。

検証：

- `[shipped_actors]` で従来設定の51 failuresを確認後、7環境×2モードの専用Actor参照・Greedy設定をGREENとした。Rainbowも共通設定からtyped値を検証した。
- Debug全体ビルド成功。途中のリンクは並行していた設定採取の実行体ロックでLNK1168となったため、採取終了後に再実行して成功した（`.scratch/prd061/panel-greedy-final-build-retry.log`）。
- `[prd061],[config],[observer_factory]`：128 cases / 2,791 assertionsすべて成功（`panel-greedy-final-tests.log`）。
- 既定葉検査：23,235 assignments / 3,301 audited / 0 errors。検査器12テスト成功。
- 固定17入力：新規eval_panel Actorの葉だけがtyped差分で、既存typed設定は全項目一致。全キー・文字列値の差分も専用Actor・Greedyプロファイル・表示用定義・参照先だけに限定された（`panel-greedy-before/`、`panel-greedy-after/`、`panel-greedy-comparison.json`）。PRD072 goldenは更新していない。
- Atari / ImageClsのbatchrunをEvalPanel auto_start=true、exp予算32で実行し、両方exit 0。Actor・参照タグは上書きせず、Atariの実行時dumpでもtarget / Greedyを確認した。ImageClsは前回の合成画像fixtureを使用した（`panel-greedy-smoke/execution.json`）。
- `git diff --check`成功。commit / pushは行っていない。
