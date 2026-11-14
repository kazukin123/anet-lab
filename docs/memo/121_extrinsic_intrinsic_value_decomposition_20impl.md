# 外的と内的の価値の分解 実装メモ

## 概要

2026-11-14。正本は [PRD121](121_extrinsic_intrinsic_value_decomposition_10prd.md)。DefaultDQNAgent の TD・QR・IQN に、外的報酬と内的報酬を別の Head で学ぶ `intrinsic.value_mode = decomposed` を実装する。既定は `combined` とし、既定の学習系列は維持する。

PRD128 の前提5件は、現行の課題記録ですべて解決・残件無しである。現行コードは IntrinsicRewardCoordinator、共通 Learner、DqnComponentAssembler を持つ。PRD の古い行番号ではなく、現在の責務の持ち主へ変更を入れる。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidence で解決済み: PRD §5 D1〜D28、§6、§8、§9.2 の23受入項目を契約とする。PRD128 の5課題は解決済み。PRD127・129 は現行の内的報酬 Trace の経路に反映済みで、その経路を追随する。
- 2ximpl へ固定: 下記の縦スライス順、報酬成分の共通契約の実装方法、既存テストへの追加、検証資材の置き場。新しい public interface の挙動は PRD の範囲内に限定する。
- PRD 範囲外: 効果実験の Run 群・Atari の実験プロファイル・bat、TempoRL と分解の併用、観測正規化の checkpoint 保存、Rainbow の内的報酬への対応。

## 主な変更

1. **報酬成分と Replay**: BatchExperience と ExperienceSamples に成分を運ぶ経路を設け、既存の1成分の tensor の形と値を保つ。Replay の構築時に成分数を決め、Push の割引率、slot の報酬と収益、schedule 時の抽選済み報酬、初期優先度入力を成分ごとにする。旧 Push overload は残さず、全現用呼び出しとテスト用派生を移行する。2成分以上と窓規則の併用は構築時に拒否する。成分の意味は DQN 側が所有する。
2. **設定と Head**: PRD §8.1 のキー、既定、値域、休眠時の診断を実装する。DecomposedHeadFactory は既存の Head factory を組み合わせ、外的と内的の出力を別の接頭辞で返す。Network は分解時に `q` と `q_dist` を返さない。Body の shared/dedicated 入力を検証する。既存の Noisy・初期化・分布の選択規則を使う。射影設定の検査と使われない値の通知はnn_modules.cppの共通処理を使う。
3. **合わせる処理と Actor**: DQN の1つの処理で Head の出力を β で合わせ、必要な TBO の逆変換と再変換を行う。Actor・通常更新・診断の forward 結果を読む全経路へ適用する。方策の読み手は既存の鍵を使う。Actor ごとの β を扱い、hint は常時5列にする。行動差し替え時の内的 Q_sa と Munchausen 項の選び直しも行う。
4. **Learner**: Head ごとの目標値・損失・優先度、内的の割引率と終端モード、外的 bootstrap モードを実装する。Munchausen の加算項とエントロピー項は外的に置く。分布型は同じ τ を使う。merge は1回の backward、split は既存の skip の方式に従う損失別の backward・clip・Adam 状態を使う。既定の forward の回数・順序と乱数消費を保つ。
5. **報酬スケーラ**: `return_std` と内的報酬用スケーラを追加し、done の扱いのあと、β を掛ける前に変換する。内的の割引率の step 0 の値を使う。統計更新なしの変換を Trace に使う。動的スケーラの状態を外的・内的とも保存する。モード不一致は拒否し、return_std の lane 数変更は統計のみ引き継いで割引和を初期化し、1回 WARN する。
6. **診断**: Head ごとの Trace 4鍵、Learner の損失・Q・勾配・内的の割合・スケーラ倍率を追加する。Actor は平均の実空間 Q から不一致・譲った外的の価値・外的の差・内的の差を毎 step 計算する。lane 値、lane 平均、評価 session 累積は別の鍵とし、探索や強制行動で変えない。未有効時や未実施の値の NaN 契約を守る。metrics_scalar.txt に任意選択の組だけを追加する。
7. **checkpoint**: 分解と feature_gradient の違いを明示的に拒否する。既存の定数スケーラ・combined の checkpoint の形と読込みを保つ。旧い動的スケーラの checkpoint は、状態が無い種類として判定し、型不一致で拒否する。
8. **文書と検証資材**: 214_value_decomposition.jp.md を新設し、PRD §8.7 の140・150・200・212・020・030および必要な索引を更新する。英語版は anet-translate-docs の規約に従って作る。CONTEXT.md の Actor Qヒントの定義を更新する。既存 ADR0073〜0075 と旧 ADR の注記は PRD 起草済みの判断として使う。testdata/prd121/README.md と検証スクリプトに受入手順を置き、200 のテスト章から参照する。prd072のgoldenは固定commitの履歴で廃止済みプロファイルを含むため保持し、現用17構成の全キー・文字列値とscalar定義を変更前後で比較する。許す差はPRD121の追加既定値だけとする。この扱いは2026-11-15に開発者が承認し、PRD §8.7も改訂した。

## テスト

- Public interface / surface: DefaultDQNAgent の設定・Actor・UpdateFromBatch・GetScalar・step diagnostics・Save/Load、公開 Replay factory・Push・Sample・初期優先度 estimator、RewardScalerFactory の Scale/ScaleWithoutUpdate・保存読込み、出荷設定解決。
- 優先 behavior: まず小さい Agent の設定から行動・取込み・Replay・1回更新までの経路で、分解の Head が学び、別々の Q を読めることを確認する。そのあと、既知の値による各契約の検査を1件ずつ追加する。
- TDD 順序: 各項目で1テストを書いて RED を記録し、最小実装で GREEN にしてから次へ進む。必要な共通経路の変更は最初の tracer bullet の範囲で行う。RED 中に無関係な整理をしない。
  1. TD の小さい分解 Agent の1回更新と、外的・内的の値の取得。
  2. 成分ごとの手計算の Replay 収益。定数から始め、schedule、終端、打ち切り、窓拒否へ進む。既存1成分の期待値は変更しない。
  3. QR・IQN、TBO の結合と目標値、Double DQN、Munchausen、bootstrap のモード、non_episodic の終端と done/truncated 同時真。
  4. 損失重み、PER の誤差の和と初期見積もり、常時5列の hint、行動の差し替え。
  5. split の独立 clip と Adam 更新の和。既知の勾配と weight decay 0 の参照更新を使う。
  6. return_std の2 lane の手計算、統計更新なしの変換、内的の既定の恒等変換、done の適用順。
  7. Actor の fixed/learner β、Head の Trace、行動診断の lane 値と平均、1行動の NaN、lane 範囲外、評価 session のリセットと空 session。
  8. 設定判定表の全行、shared/dedicated 入力と shape、既定値と未使用値の診断。
  9. checkpoint 判定表と状態の継続、旧い定数スケーラの checkpoint、型の相違と lane 数変更。
  10. 出荷設定で任意のメトリクスの組が解決でき、鍵が一致して tag が40文字以内であること。

## 検証

### 本体変更前

- 現行 Git 状態と既存変更の範囲を確認した。既存変更を戻さず、今回の変更を記録する。
- Debug・Release 実行体と必要な DLL・設定・旧 checkpoint を `.scratch/prd121/` に保存する。既存の同名資材は上書きしない。
- PRD128/125 の既存検証資材を読み、保存・設定解決・scalar 比較の共通処理を再利用する。元の資材は変更しない。
- 同一性の5構成は Atari RND、LunarLander RNDあり・なし、GridMaze既定、DropMerge既定。seed 1・決定的設定で、学習・評価・該当する周期 target 同期が1回以上生じる長さを固定して README へ記録する。

### 実装後

- Debug 全ビルドと関連テスト、既定 core suite、出荷設定検査を実行する。既存の失敗があれば今回の回帰と区別する。
- 保存した実行体との比較は5構成の全 scalar 点を対象にし、時間と資源の tag だけを除く。tag 集合も比較する。
- PRD §9.2 の23項目を、単体・統合テスト、出荷設定検査、checkpoint、Run 比較、文書確認へ対応付けて結果を記録する。
- Atari の同一構成で combined と decomposed（Body共有・merge）を交互に3巡以上測定する。測定中は他の build/test/Run を重ねない。性能の閾値は置かず、各回の実時間と条件を記録する。
- 開発者の2026-11-14の指定: 別Runが稼働中なので、性能測定はできるだけ就寝前の25時前、または朝9時以降に寄せる。別RunをAIが停止せず、測定時に他のRunが稼働していれば性能測定を待つ。正しさの検証で採った実時間を性能の測定値として使わない。
- 通常ログと標準エラーを `.scratch/prd121/` に保存し、終了コード・失敗箇所・件数を要約する。未実施を成功としない。

ビルドとテストの基本入口（実行時は通常ログもファイルへ保存する）:

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release'
core\anet-core\bin\Debug\anet-core-test.exe '[prd121]'
core\anet-core\bin\Debug\anet-core-test.exe '[replay_buffer],[dqn],[intrinsic],[scaler],[shipped_config]'
core\anet-core\bin\Debug\anet-core-test.exe
```

検証資材の追加後、その正確な実行コマンドと結果を本節へ追記する。

## 前提

- PRD は2026-11-13の2回目の改訂までを対象とする。最初の起票行の「暫定」は履歴であり、2026-11-12の改訂で印を外した現行契約を使う。
- 既定は combined・共有割引率・episodic・combined bootstrap・loss_weight 1・priority_lambda 1・merge。内的スケーラは恒等変換。専用 gamma と fixed action_beta に既定値を設けない。
- 分解を使うときのみ Replay は2成分。Rainbow 等の利用側は1成分を維持する。
- 既存の実験記録、過去の Run artifact、過去の検証資材は書き換えない。現用の呼び出しと出荷設定基準を同じ変更で移行する。
- コミットの準備、ステージ、commit、push はこの工程に含まない。

## 経過

- 2026-11-14: 規約、PRD、現行の設定・共通 Learner・Head・Replay・報酬スケーラの入口、PRD128 の完了記録と依存5課題を照合した。未決事項監査のブロッカーは0。
- Debug・Releaseの実装前全ビルドが成功。通常ログは `.scratch/prd121/baseline-debug-build.log` と `baseline-release-build.log`。実行体・DLL・設定は `.scratch/prd121/baseline/` に保存した。Debug RunnerのSHA256は `3b806cfb3fcea6ae6d6fe5adeea77f8b7fbc1c1331adabda6ce276006a3dcc3e`、coreテストは `9919c752e1546e55b54c3f5691c39e1d4a53e77447c11a9021682699253eef3a`。
- 最初のtracer bulletをintrinsic_test.cppへ追加し、ビルド成功後に `[prd121]` の失敗を確認した。ActorからReplay・更新まで進み、`extrinsic_loss` の取得が未実装のためREQUIREが失敗した（終了値42、1 case・1 failure）。`.scratch/prd121/tracer-red.xml`・`tracer-red.log`・`tracer-red-build.log`。まだGREENではない。
- 検証資材 `core/anet-core/testdata/prd121/run_validation.py` を追加。snapshot・prepare・before・after・compareを持つ。基準5構成の凍結が成功。`before --label baseline` を実行中（起動時のsession ID 53612）。Atari RNDは終了値0・33,216点を保存した。全構成の結果は `.scratch/prd121/baseline-summary.json` を確認する。
- tracer bulletのGREENへ向けた編集中。設定DTO・読み取り、合成Headを追加し、Replayの割引率の口とDQNのforwardの口を移行中。`.scratch/prd121/make_migration_patch.py` はファイルを書き換えず、機械的な移行の差分を出す作業用道具で、出力をapply_patchで適用した。日本語の出力はPYTHONIOENCODING=utf-8とConsole.OutputEncodingをそろえる。現時点の本体は未ビルドで、Replayの中身・DqnForwardExecutor・分解の更新はまだ実装途中。自己点検・独立レビューは未実施。

## 決まらなかったこと

なし。

## 最終検証の結果（2026-11-15）

### ビルド・動作・レビュー

- Debug・Release全ビルドは成功。最後のfixture追随後も両構成で終了値0（`fixture-debug-build.log`・`fixture-release-build.log`）。
- 既定core suiteは192,103 assertion・失敗/ERROR 0・skip 1（`final-core.xml`）。その後のレビュー修正を含む関連まとめは19,882 assertion・失敗/ERROR 0・skip 0（`fixture-final-related.xml`）。通常ログも同じprefixの`.log`へ保存した。
- 既定動作の5構成は、全174,438 scalar点の正規化JSONLとtag集合が実装前と完全一致（`acceptance-final-comparison.json`）。現用17構成の全キー・文字列値の比較も成功。既存のPRD072 goldenは保持し、現用構成を別に比較する扱いをREADMEへ記録した。
- 変更前のAtari RND・DropMergeの定数スケーラcheckpointは、新Runnerで読め、元ファイルのSHA256も保持（`acceptance-final-legacy-summary.json`）。動的スケーラと分解の保存復元・型不一致は個別テストと最終関連まとめで検証した。
- 初回の独立レビューは4往復。初回7件、次のfixture追随漏れ1件を修正し、3回目で計画・実装ともOK。4回目では性能記録も検算してOK。原文は`review-20261114-2309.md`・`review-20261114-2346.md`・`review-20261114-2353.md`・`review-20261115-0012.md`。この4往復の範囲外の検出は無し。その後のcc-reviewは下記に別記する。

### Atariの性能3巡

2026-11-14 23:53に開始し、翌00:01台に完了を確認した。前日の25時前である。別Runは停止せず、起動前と各Runの間にRunner・coreテスト・cl・ninjaとの競合が無いことを確認し、測定中もビルド・テストを重ねなかった。全6 Runが終了値0。

- 実行体: Release Runner。SHA256 `a915ff6633db60bc4a8d9cb67c3502cc611230c8e7b106a07cfac247e99273a8`。全Runで同じ実行体を使用した。
- GPU: NVIDIA GeForce RTX 5090、driver 616.92、32,607 MiB。CPU側Torch threads 1。
- qbert・seed 1・CUDA・128 lane・batch 256・Replay 65,536・学習とRNDのwarm-up各4,096・replay ratio 1・RND batch 32・131,072 experience。共有Body、merge、IQN、fused optimizer、LearnerはBF16 AMP、Actor/target policyのAMPは無効。評価と終了時checkpoint保存は無効。
- 測定の主値は、`90_perf/90_elapse_hour`の差から求めた学習区間の実時間。全6回とも20,096〜130,048 experience（109,952件）で一致した。16,384以降の最初の点を使い、catalogのintervalによって20,096になった。プロセス全体の時間も起動・終了を含む別の値として保存した。
- 保存先: `.scratch/prd121/final-20261114-235358-performance/summary.json`、同じフォルダの凍結設定、各Runのresolved config。通常ログは`performance-final.log`。比較の閾値は置かない。

| 巡 | モード | 学習区間の実時間（秒） | プロセス全体（秒） | experience/秒 |
|---|---|---:|---:|---:|
| 1 | combined | 28.443 | 99.008 | 3,865.7 |
| 1 | decomposed | 29.571 | 65.339 | 3,718.2 |
| 2 | combined | 28.099 | 63.464 | 3,913.0 |
| 2 | decomposed | 29.085 | 64.527 | 3,780.4 |
| 3 | combined | 27.680 | 58.162 | 3,972.3 |
| 3 | decomposed | 29.434 | 64.465 | 3,735.5 |

学習区間の中央値はcombined 28.099秒、decomposed 29.434秒。decomposed/combinedは1.04751、約4.8%増。experience/秒の中央値は3,913.0と3,735.5である。この共有Body・merge・IQN構成での測定結果であり、ほかの構成の速度や学習効果はこの測定から判断しない。

実行した入口（PowerShell。ログは上記へ保存済み）:

```bash
$benchmarkLabel='final-'+(Get-Date -Format yyyyMMdd-HHmmss); & ./.venv/Scripts/python.exe core/anet-core/testdata/prd121/run_validation.py benchmark --config Release --label $benchmarkLabel --rounds 3
```

この入口は現在時刻から新しいlabelを作る。今回のlabelは`final-20261114-235358`だった。

### 受入の対応

| PRD §9.2 | 結果と主な根拠 |
|---|---|
| 1 | PASS。5構成・174,438点・tag集合の完全一致 |
| 2 | PASS。既存1成分Replayの断定を維持、既定core suite成功 |
| 3 | PASS。2/3成分、定数/schedule、通常/done/truncated/両方、prefetch、窓拒否 |
| 4 | PASS。combinedとのTD/QR/IQN×Munchausen×TBOの目標値の一致 |
| 5 | PASS。2つの終端モードとdone/truncatedの組み合わせ |
| 6 | PASS。外的/combined bootstrap、Double DQNの両設定、UQEのrisk選択 |
| 7 | PASS。内的gammaの別設定とschedule、成分の収益とbootstrap |
| 8 | PASS。mergeの重み1.0・0.5を含む損失の数値検査 |
| 9 | PASS。片方だけclip・独立Adam更新の和・weight decay・schedule・保存復元 |
| 10 | PASS。λ1.0・0.2・0、逆向きの誤差の和 |
| 11 | PASS。Head出力・τ対応・TBOの結合、β0の厳密な外的値 |
| 12 | PASS。実Agentのfixed 0とlearner βの行動選択 |
| 13 | PASS。設定判定・未使用値の通知・Head入力/shape、出荷設定17構成 |
| 14 | PASS。2 laneの割引和・統計・統計更新なしの変換・保存復元 |
| 15 | PASS。恒等変換、done mask→scaler→β、combined/分解の実取込み |
| 16 | PASS。Head4鍵とcombined時NaN、Q専用Traceの費用分離 |
| 17 | PASS。merge/split/combinedのメトリクス、任意profileの出荷設定 |
| 18 | PASS。checkpoint判定表・状態の継続・旧constantの2 Run読込み |
| 19 | PASS。常時5列・内的列0・Head別優先度・不正Kの拒否 |
| 20 | PASS。上記の同一Atari構成で交互3巡、全6 Run成功 |
| 21 | PASS。設計・ユーザーガイドの日英、用語、起草済みADR、検証資材の照合 |
| 22 | PASS。3行動・2 lane、TD/QR/IQN×TBO、β0、差替え不変、範囲拒否、相対1e-5 |
| 23 | PASS。lane×stepのsession累積・空/reset、eval/eval_targetのtagと長さ |

23項目の外にある人手項（EvalPanelの2 lane以上の実画面表示）は未実施。本セッションではGUIを操作しておらず、公開APIとTrace生成の自動検査を実画面の確認とは扱わない。残る設計の未決と新規起票の候補は無し。

## cc-review（2026-11-15）

開発者の依頼により、49対象を同じ依頼文でCodexとClaudeへ渡した。初回はCodexが計画・実装ともOK、Claudeが計画1件と実装4件を差し戻した。原文は`.scratch/prd121/review-20261115-0330.md`と`review-20261115-0326-claude.md`。

| Claudeの指摘 | 対応 |
|---|---|
| 計画のgolden更新とレビュー往復数が現在の記録と不一致 | 「主な変更」8を現用17構成の比較へ合わせ、初回の独立レビューは最終検証・完了報告とも4往復に統一した |
| dedicatedで不足キーが標準のfeatures名になる | 内的HeadのDueling有無を構築側から渡し、専用出力の配線検査は元のintrinsic_*とnet.body.outputのキーを示す。rank等の検証は既存Headを使う |
| Head・全Actorの列挙/値域より組み合わせの検査が先 | Headと全Actorの値を検証したあと、機能の組み合わせ、gamma必須値、全Actorのfixed必須値の順に判定する。自前のモード列挙は数値より先に検査する |
| bootstrapとlog_policy_modeの定数参照漏れ | bootstrapは既存定数へ合わせ、MunchausenValuesの3列挙定数を既定・検査・既存/新規分岐で共有した。テストの対になる定数も合わせた |
| ガイド本文に仕組みと理由がある | 020の日英§6.4/6.5の説明をNOTEへ移し、列追加の手順を本文へ残した。翻訳元blobと構造の対応を確認した |

別に、ClaudeはPRD §8.7のgolden更新要求と、PRD072の固定golden運用の不一致を挙げた。Codexは現用17構成による検証記録との整合を認めた。両者の見解を示して尋ね、開発者は「現在の17構成による比較を受け入れ、PRDを実態に合わせる」を選択した。回答後に、PRDの改訂履歴・§4・§8.7と計画を更新した。goldenは固定commitの履歴として保持し、許す追加はPRD121の既定キー/値だけとする。これにより未決は残っていない。

検証:

- 追加した診断テストは修正前に29 assertion失敗（`cc-r1-red.xml`）。最初の修正後確認で新fixtureのskip.max_length未指定により5 assertion失敗（`cc-r1-green.xml`）、その後の関連まとめでskip.net未指定により5 assertion失敗（`cc-r1-related.xml`）。両方を有効なskip構成へそろえて、追加テスト112 assertion・失敗/ERROR 0を確認した（`cc-r1-final-focus.xml`）。失敗した実行の記録も保持する。
- 最終のDebug・Releaseビルドは終了値0（`cc-r1-fixture-debug-build.log`・`cc-r1-fixture-release-build.log`）。両ストリームをログへ保存し、末尾だけを返した。
- 修正後の関連まとめは27,240 assertion・失敗/ERROR 0・skip 0、149.623秒、終了値0（`cc-r1-final-related.xml`）。filterは`[prd121],[actor_initial_priority],[action_policy],[step_diagnostics],[shipped_config],[prd127],[prd129],[dqn][munchausen]`。通常ログは`cc-r1-final-related.log`へ分けて保存し、末尾だけを返した。
- impl/prdの品質確認観点を適用範囲に従って全項目点検した。50対象の既存45ファイルでBOM保持、新規5ファイル、LF、禁止パターン、日英8組の構造・翻訳元blob、PRDの状態行・番号一意性・リンクの検査は成功（`cc-r1-checks.json`）。
- 今回の変更は構築時の検査/診断、同値の定数参照、文書に限り、学習・Replay・スケーラの数値計算と実行回数は変えていない。既存の174,438点一致・17構成比較・checkpoint読込みの証跡を保持し、性能6 Runの記録と実行体hashも保持する。性能再測定はしていない。
- 再レビューでは前の49対象を保持し、承認されたPRD改訂を足した50対象を同じ担当/Claude sessionへ渡す。再レビュー結果は本節末尾へ追記する。EvalPanelの2 lane以上の人手確認は未実施のままである。

### 検査順の残りへの対応

1回目の再レビューは両モデルとも計画・PRDがOK、実装は同じ1件が差し戻しだった。原文は`review-20261115-0401.md`と`review-20261115-0355-claude.md`。内外スケーラの組み合わせが後続の列挙・数値より先に検査され、Head init.modeもlearnerの数値より後だった。

- 共通`ConfigReader<RewardScalerConfig>`に、型・列挙とunused通知を保って組み合わせだけを後へ残す`ReadFields`を設けた。既存の`Read`は従来どおり全検証する。DQNは内外を`ReadFields`で読み、全関連キーの値の検証後、同じ`ValidateRewardScalerConfig`で行3を検査してから行5〜8へ進む。検証条件をDQNへ複製せず、未検証のconfigを構築子の外へ公開しない。
- 共通`ValidateWeightInitConfigMode`へ列挙検査を分け、既存の全検証もこの処理を呼ぶ。DQNでは数値の前にmode検査を呼び、後で実際の全初期化設定を検証する。nn.hppは既存対象で、共通実装nn_impl.cppを追加して51対象にした。PRD §8.7の移行対象と改訂行も追随し、設定値・学習・保存・診断の契約は変えていない。
- スケーラ2種と列挙・数値8種の組み合わせ、Head init.modeと他部品の数値、行3単独の拒否を追加した。修正前は44 assertion中17失敗、ERROR 0（`cc-r2-red.xml`）。
- 最初の修正後focusは137 assertion中2失敗、ERROR 0（`cc-r2-focus.xml`）。gammaの既存診断は論理キー`learner.intrinsic.gamma`を示すが、新テストが余分な`DefaultDQNAgent.`接頭辞を要求していたため、期待値を既存の診断契約へ合わせた。キーとexpected理由は引き続き検査する。
- 最終Debug・Releaseビルドは終了値0（`cc-r2-final-debug-build.log`・`cc-r2-final-release-build.log`）。両ストリームを保存し、末尾だけを返した。修正後の全関連まとめは27,340 assertion・失敗/ERROR/skip 0、150.524秒、終了値0（`cc-r2-related.xml`）。filterは`[prd121],[actor_initial_priority],[action_policy],[step_diagnostics],[shipped_config],[prd127],[prd129],[dqn][munchausen],[nn][init],[scaler]`。通常ログは`cc-r2-related.log`へ分けて保存した。両モデルへまとめて再レビューする。
- 51対象の既存46ファイルのBOM保持、新規5ファイル、LF、禁止パターン、日英8組の構造/blob、PRDの状態行・番号一意性・リンクの自己点検は成功した（`cc-r1-checks.json`、実行ログは`cc-r2-checks.log`）。
- 今回も構築時の検証だけの変更であり、性能再測定は行わない。前回の50対象から変えていない部分はレビュー結果を引き継ぐ。

### 2回目の再レビューと所在一覧の追随

Codexは計画・実装・PRDともOK、Claudeは計画・実装がOK、PRDの所在一覧1件が差し戻しだった。原文は`review-20261115-0417.md`と`review-20261115-0414-claude.md`。両モデルとも設定判定順の残りは解消したとした。

Claudeの指摘は、以前から51対象に入っているnn_modules.cpp（射影設定の検査と使われない値の通知）がPRD §8.7に無く、nn.hppを加えた改訂理由も共通初期化検査だけを挙げていた点である。PRDのコード一覧にnn_modules.cppを加え、改訂理由と計画の「主な変更」2へ射影設定値の通知の所在を追記した。変更はこのPRDと計画だけであり、コード・設定・テストは変えていない。直前の27,340 assertion成功を引き継ぎ、文書の自己点検と両モデルの確認を行う。EvalPanelの2 lane以上の人手確認は未実施のまま。

### 最終のcc-review結果

文書の所在一覧を直した3回目の再レビューで、Codex・Claudeとも計画・実装・PRDの3件すべてOK、修正要求・範囲外の検出なしとなった。原文は`.scratch/prd121/review-20261115-0421.md`と`review-20261115-0419-claude.md`。依頼は`request-20261115-0419.md`、対象の保存差分は`diff-20261115-0419/`。最終判定時にも51対象のSHA256が依頼時と同一であることを確認した。

Claudeは初回から同じsession `e2ebfe8c-4750-4dfc-9e7f-f50f6eaf4086`を再開して使い、最終SDKログの概算費用は11.6689204 USD（`claude-20261115-0419.log`）、終了値0・error=falseだった。キーは表示・保存していない。

レビュー中に別作業の課題属性のcommitでHEADが`ce44a84c`へ進み、別作業13ファイルがステージされていた。新commitはPRD121の51対象に触れておらず、snapshotも同一だったため、判定・検証への影響はない。こちらからstage/commit/pushは行わず、そのindexを保持した。最終の結果記録だけを本節と課題へ追記し、コード・設定・テストには触れていない。EvalPanelの人手確認は未実施として残す。

## 追加指摘への対応（2026-11-15）

- 書き手と読み手が共有する値をクラスのメンバ定数へ統一する。`approx_priority` は `DqnComponentAssembler`、通常Headの出力鍵は `HeadFactoryBase` が持つ。分解Headの登録名・出力接頭辞・出力鍵・optimizerのパラメータ接頭辞は `DecomposedHeadFactory` が持ち、登録名と通常Headの出力鍵から組み立てる。Qヒントの診断に出す列数も `kActorQHintColumnCount` を使う。
- `Learner::LoadCheckpoint` の内的optimizer設定の置換ログは、共通方針の失敗の扱いの表の第1行「checkpoint の値を今の設定で置換」に当たる。checkpointの学習率・weight decay・epsilonを現在の設定値で置換したとき、対象のgroupにつきRunごとに1回だけINFOを出す。ログのレベルと回数の実装は変更しない。
- `PackActorQHint` の内的tensorの不整合は、tensor名、実際のshape・device、期待するshape・deviceを診断に含める。未定義tensorはshape・deviceを参照せず、`undefined` と表示する。両tensorについて未定義・shape不一致・device不一致を回帰テストで確認する。
- 既存行の `q` / `q_dist`、`aux.find("q_values")`、`munchausen_terms` の直接書きは、今回とは別の指摘として対象外とする。演算・乱数・設定値を変えないため、性能測定は追加しない。
- 検証: 追加テスト `[prd121-review-pack]` は修正前に6条件すべて失敗（`followup-red.xml`）。修正後の `[prd121-review-pack],[hint]` は5 case・79 assertionが失敗/ERROR/skip 0（`followup-hint.xml`）。Meta deviceのtensorを使い、GPUへの転送なしでdevice不一致を確認した。
- 全体ビルドは `cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'` と、同じコマンドのpresetを `x64-Release` に変えたものが両方成功した。標準出力・標準エラーは `.scratch/prd121/followup-debug-build.log` と `followup-release-build.log` に保存し、画面には末尾だけを返した。
- 関連テストは `core/anet-core/bin/Debug/anet-core-test.exe '[prd121],[hint],[actor_initial_priority],[action_policy],[step_diagnostics],[prd127],[prd129],[dqn][munchausen],[nn][init],[scaler]' --reporter junit --out .scratch/prd121/followup-regression.xml` で実行し、24,856 assertionが失敗/ERROR/skip 0、142.353秒。通常ログと標準エラーは `.scratch/prd121/followup-regression.log` に保存し、画面には末尾とJUnitの集計だけを返した。ビルド・テストはセッションの完了通知を待ち、途中は進捗だけを確認した。
- 自己点検: 実装品質確認観点の全項目を適用判定し、対象項目を確認した。変更範囲・定数の所有者と利用箇所・診断の実値・INFOの分類・追加テストの外部挙動を照合し、BOM保持・LF・重複文字列・禁止された表記・`git diff --check` の機械検査も成功した。設定・保存形式・演算・UI契約の変更は無い。

## 完了報告

- 実施内容: PRD121の価値分解、Replay、最適化、報酬スケーラ、診断と保存を実装し、受入23項目の自動検証と性能3巡を完了した。
- 変更ファイル: 初回は`.scratch/prd121/review-r2-paths.json`の49成果物。承認されたPRDの改訂を加えた50成果物は`.scratch/prd121/review-cc-r2-paths.json`。検査順の共通処理nn_impl.cppを加えた最終51成果物は`.scratch/prd121/review-cc-r3-paths.json`。課題記録とINDEX・wikiの派生区画も更新した。既存の別作業の変更とステージを保持した。
- レビュー結果: 初回の独立レビュー4往復で計画・実装ともOK。最後は性能の実測scalar・中央値・比率をレビュー側でも検算した（`review-20261115-0012.md`）。その後のcc-reviewは初回と再レビュー3回を行い、最終は両モデルとも計画・実装・PRDすべてOK。指摘・対応・証跡は上の専用節を正とする。
- 決まらなかったこと: なし。新規起票・既存課題への追記・この場で直す・見送り推奨の起票候補は無し。
- 判断してほしいこと: EvalPanelで2 lane以上の実画面を確かめる人手項が未実施であり、自動検査の成功とは分けて残す。
- 課題記録: issue-20261110-0de9e7のAI判定を残件無し、日付を2026-11-15にした。statusは対応中。`wiki_support.py maps`は成功し、課題一覧とwiki318/324の課題区画を再生成した。

## 実装途中の確認位置

### 23:09の独立レビューへの対応

- 23:53: 3回目の独立レビューで計画・実装ともOK、修正要求・範囲外の検出は無し。原文は`review-20261114-2353.md`。性能3巡と人手UIは未実施としての判定であり、続けて競合の無い状態で性能測定を開始した（`performance-final.log`、採取labelは`performance-label.txt`）。測定中はビルド・テストを重ねない。

- 23:52確認: 2回目のレビューは、初回7件の修正を確認したうえで、関連まとめのfixture追随漏れ1件で差し戻し（原文`review-20261114-2346.md`）。`review-final-related.xml`は19,864 assertion・failure 1・ERROR 0だった。priorityの完了通知を検査するfixtureのkeysが空だったため、approx_priorityとHead別Qの要求を明示した。Q専用の検査ではpriorityを要求しない形を保った。本体の要求条件は変えていない。
- fixture修正後のDebug・Release全ビルドは両方終了値0（`fixture-debug-build.log`・`fixture-release-build.log`）。同じ関連まとめの`fixture-final-related.xml`は19,882 assertion・失敗/ERROR 0・skip 0、終了値0（131.569秒）。Traceの近似優先度の完了通知と有限値のassertionも通った。23:52時点で競合するRunner・coreテスト・cl・ninjaは無い。性能測定はまだ未実施であり、最後のレビュー確認後に開始する。

- 23:42までの再検証: `review-constants.xml`は1,486 assertion・失敗/ERROR 0（非零値の相対1e-5・零だけ絶対1e-6のActor lane/session、Q専用Trace、非有限勾配を含む）。そのあと単独の公開API入力の鍵も定数へ揃えた。`review-final-debug-build.log`は全ビルド終了値0。Release全ビルドと`review-final-related.xml`（PRD121・初期priority・方策・step診断・出荷設定・PRD127・129）を実行中。コードの最後の変更は定数参照と診断メッセージで、計算は変えていない。
- 自己点検を49ファイルへ広げた。既存44ファイルのBOMを保持、新規5ファイル、すべてLF、禁止構文0。日英8対のblob・見出し/fence/表・code block・相対リンクは全件一致。wikiの検索で残った3列の記述は、日付付きの履歴または不採用案だけである。新しい公開識別子を使う追加行は、重複の有無によらず定数を参照するように確認した。

- 初回レビューは計画OK・実装差し戻し7件。原文は`.scratch/prd121/review-20261114-2309.md`。指摘は、Q列のみでのRND費用、FP32の非有限勾配の検査、Actor診断の精度、Optimizeの計測区間、TBO時のintrinsic_q_share、追加識別子・テスト値の定数化、wikiの5列契約への追随である。範囲外の検出は無し。同じレビュー担当へまとめて再依頼する。
- Q専用Trace: Actorを渡さない要求と、Actorを渡してもsync followerを作らない要求を公開CreateStepDiagnosticsProvider→Observeで検査した。`review-trace-red.xml`は30 assertion・2 failure・1 error。fixtureのstep_tへの型合わせの前はビルドで失敗した（r1/r2）。修正後`review-trace-green.xml`は58 assertion・失敗/ERROR 0。分解時にapprox_priorityを要求した場合だけ成分報酬と系列窓を計算し、内的報酬の鍵または分解のapprox_priorityを要求した場合だけRNDの写し・採点を行う。Q列のみではどちらも行わない。212の日英の費用説明を揃えた。
- 勾配検査: forward値を変えず、選んだHeadのbackwardだけNaN/Infへするhookを使い、両optimizerの前に停止して全重みが不変であることを検査した。`review-nan-red.xml`は48 assertion・34 failure。修正後`review-nan-green.xml`は既存splitの数値検査を含む388 assertion・失敗/ERROR 0。ANET_ENABLE_TENSOR_NAN_CHECK有効時はFP32/BF16の両損失のnormをまとめて検査する。FP16は従来どおりGradScalerで両方をskipする。診断時の追加同期は検査有効時に更新につき1回であり、1回0.02〜0.10ms・100更新/秒なら2〜10ms/秒の測定前の概算。既定FP32の既存の勾配検査を分解時にも保持するものである。
- Optimizeの計測: zero_gradを終え、prepare・primary backward/norm・intrinsic zero_grad/backward/norm・勾配検査・clip・optimizer stepを区別した。計測のための不自然なブロックは作らない。
- Q比率: TD/QR/IQNの既知のHead値で、TBOの有無によらず実空間の比率になる検査を追加した。`review-share-red.xml`は10,368 assertion・576 failure、修正後`review-share-green.xml`は10,368 assertion・失敗/ERROR 0。分位点ごとに逆変換してから行動の平均を取り、abs(β Q_int)/(abs(Q_ext)+abs(β Q_int))を標本ごとに作る。Headの平均Qを表示する2指標の出力空間は変えない。
- Actorのlane値・平均・session累積の検査を、非零では相対1e-5、零だけ絶対1e-6へ変更した。Headのhintの比較も相対1e-5・絶対1e-6へ絞った。結果は`review-constants.xml`へ保存する。
- 定数化: 新しいGetScalar鍵、Headのforward関数の識別子、追加テストのTensorDict・メトリクス鍵、scalarの種類を定数へ揃えた。設定の列挙値は可能なものを所有元の定数へ揃えた。同じonlineという綴りでもnetworkの選択とMunchausenのlog方策の選択は意味が違うため別名にした。初回自己点検ではtestの追加行と単独の公開識別子の走査が足りなかった。差分生成の一時出力は省略を避けてファイルへ保存し、apply_patchでhunk単位に反映した。char配列を別定数から初期化していた4行をconst char*へ修正し、ビルドと検査を進めている。
- wiki: 318_rl_coreの現行説明を3列から5列へ追随し、wiki/logへ出典blobを付けたfix行を追記した。過去の日付付きの記録と、不採用の3列案は保持した。

### 22:50台の自己点検と最終検証

- 23:02確認: `final-core.xml`は192,103 assertion・failures 0・errors 0・skip 1、終了値0（701.825秒）。`acceptance-final-comparison.json`は5構成すべてexact_match=true、合計174,438点・tag集合とも完全一致。`acceptance-final-legacy-summary.json`は変更前Atari RNDとDropMergeのcheckpoint読込みが両方終了値0・checkpoint_unchanged=true。これらは下の実行中の記録に対する完了結果である。
- 5構成のRunログのWARNは、8 laneへ縮めた4構成で従来のspatial探索のnum_envs<32が各1件、GridMazeは0件。PRD121で足したWARNは無かった。新しい未使用値のWARNは共通方針の「ほかの設定のために使われない値」、return_stdのlane数変更は「今の構成では履歴の一部の効き目が失われる構成」に対応し、どちらも対象につき1回。前者は明示した非既定値のときだけ、後者はPRDのcheckpoint契約が指定する条件だけで出る。
- 整理後のRelease RunnerとDebugライブラリの再ビルドは終了値0。`final-release-runner-build.log`・`final-debug-core-build.log`。整理は分解Actor hintの余計な計算と、DQN内部scalarによるγの受け渡し、文字列定数の使用だけで、combinedの計算・乱数・保存の経路は変えていない。Debug全体の再ビルドと関連検査を`checked-debug-build.log`以降へ保存する。
- 日英8対の見出し・fence・表の件数、code block全文、相対リンクの実在、translated-fromのblobがすべて一致した。8対とも不一致0。独立レビューは`.scratch/prd121/review-request-20261114-2300.md`から開始した。差分は`diff-20261114-2300/`、対象47ファイルを固定した。

- Debug全ビルドは終了値0（`final-debug-build.log`）。Release全ビルドも終了値0（`final-release-build.log`）。そのあとActor hintの分解経路でcombined用のgather・soft valueを重ねて計算しないように整理し、内的gammaの診断受け渡しを`dynamic_cast`から既存GetScalarの口へ変更した。DQN内部型の`intrinsic_gamma`は行動時の評価値を読むための鍵で、汎用層やメトリクスの組へは足さない。整理後のDebugライブラリとRelease Runnerを再ビルド中。
- 現用17構成の全キー・文字列値と既存メトリクスの比較は2回とも全件成功。`acceptance-r2-config/comparison.json`と`config-20261114-224404-config/comparison.json`。後者ではテスト実行体を採取先へ複製し、元DLLの場所をPATHへ加える口も確認した。追加31〜35キーだけで、既存キー・値・scalar定義の変更なし。PRD072の固定commitのgoldenは過去の廃止済みprofileを含むので保持し、現用17構成を別に比較した。これは計画の「goldenを更新」からの変更である。
- 既定core suiteを通常ログ`final-core.log`とJUnit`final-core.xml`へ保存して実行中。変更前後の5構成の採取は`after-final.log`・`acceptance-final-summary.json`へ保存中。22:58時点でAtari RND 33,216点、LunarLander RND 44,679点、LunarLander 37,794点、GridMaze 26,950点の正規化SHA256が変更前と一致した。DropMergeは実行中。旧constant checkpointの読込みは`legacy-final.log`へ保存して実行中。未完了を成功とは扱わない。
- 直近の個別検査: `stable-inverse.xml`は12,072 assertion・失敗/ERROR 0（TD/QR/IQNの結合と数値、CUDA FP16/BF16×merge/splitの実更新を含む）。`combined-equivalence.xml`は24 assertion・失敗/ERROR 0（TD/QR/IQN×Munch×TBOのcombinedとの目標値の一致）。`split-options.xml`は272 assertion・失敗/ERROR 0（片方だけclip、Adam更新の和、weight decay、schedule、保存復元）。このほかの個別検査は本節の以前の記録とXMLを参照する。
- 自己点検の機械的な確認: 今回の47ファイルを`.scratch/prd121/owned-paths.json`へ固定。差分の空白エラー0、全ファイルw/lf、既存BOMの変化0、ヘッダのusing namespace 0、productionのdynamic_cast 0、大文字TODO 0、禁止された直訳語0。共有プロセス契約2ファイルのプロジェクト依存語の検索0。重複文字列候補は設定キーだけとなるようにモード値・報酬スケーラの種類・Headの区切りを所有元の定数へ揃えた。
- 読解による自己点検: 計画の前提・公開の検証口・縦スライス順（9〜13）、PRDの確定事項と変更範囲（12・14〜22）、設定の構築時検証・休止・未使用診断・shape（23〜29・54・65・67〜70・74・75・77）、C++配置・所有権・既存interfaceの使用・命名・コメント（30〜39・43〜45・55〜60・63〜66・69）、計測とGetScalar（40〜42・61）、公開挙動のテストと本番設定の規則（46・47・73）、文書・翻訳・資材配置（48〜52）を照合した。skill変更は無く8・53は対象外。残る受入の結果は次の最終検証と独立レビューで確定する。
- 頻繁な経路の見積もり（測定前の概算）: 新しい9列Learner診断はdevice上で作り、購読した更新につきCPU転送1回（36 byte）。Actorの4列はdevice上で計算し、購読したactionにつきCPU転送1回（16×lane数 byte）、sessionは終了時1回（16 byte）である。任意profileが未選択ならこれらの読み戻しは0回。1回0.02〜0.10 msを仮定すると、毎秒100更新を購読して2〜10 ms/秒、100 actionを購読して2〜10 ms/秒である。lane内に転送を置かない。Traceの分解hintは表示・収集したstepにつき追加1回で、60 step/秒なら1.2〜6 ms/秒の概算。動的な内的スケーラは既存RunningMeanStdと同じ2つの統計値の読み戻しを追加するため、100取込み/秒なら4〜20 ms/秒の概算。既定の恒等変換には追加転送が無い。定量評価は他Runと重ならない性能測定で行い、この概算を測定値とは呼ばない。
- 文書の020・030の説明をNOTEへ移し、日英の対応を更新した。214は新規、140・150・200・212と索引・用語集も追随した。英訳の見出し・fence・表の構造、code block、相対リンクと翻訳元blobを照合する。
- 受入#20の性能3巡はまだ未実施。別Runは停止していない。01〜09時と他Runner/build/testが存在する間は測定スクリプトが拒否する。EvalPanelで2 lane以上の実画面を確かめる人手項も未実施であり、自動のlane別値の検査を実画面の確認とは呼ばない。

- 22:30台: 数値検査を相対誤差1e-5（零付近の絶対許容1e-6）へ厳しくした。ε=0.01の1,152構成の目標値・損失は成功したが、結合の2点が失敗（numeric-tolerance.xml）。原因は既存の逆変換式のsqrt−1の桁落ちで、-0.0779864に対して-0.0779904となる。分解の新経路だけは、同じ逆関数をz=2|y|/(sqrt((1+2ε)^2+4ε|y|)+1+2ε)、x=sign(y)z(z+2)で計算する。FP32のまま引き算の桁落ちを避け、結合・Headの目標値・hint・診断・初期優先度で共有する。combinedの既存計算は維持し、基準との完全一致を検査する。契約を変えずに数式の精度を満たすための実装上の判断である。
- splitの参照更新は、clip上限0.5で外的だけが超え内的は超えないことも断定し、decay=0/0.1、定数/schedule、通常/fused、clip有無と保存復元で272項目成功（split-options.xml）。return_std実取り込みは40項目成功（return-intake-r2.xml）。同じRNDの恒等スケーラと公開スケーラの参照を使い、done→統計→βの順序と、分解Replayのβなしの値をTD平均で検査した。
- Headはshared/dedicated×Dueling有無×TD/QR/IQNをNoisy内的Head付きで検査。rankとτの不一致の4失敗（head-shapes-red.xml）を、合成Headのdummy実行で出力のrank・batch・行動・τが一致する検査を加えて直し、80項目成功（head-shapes-green.xml）。
- 現用17構成の比較は全件成功（acceptance-r2-config/comparison.json）。追加キーは31〜35件、既存値の変更と削除0、scalar定義の差0。
- CUDAのFP16/BF16×merge/splitの実Agent検査を追加し、オブジェクトをビルド中（amp-object-build.log）。まだリンク/実行していない。
- 性能器はbenchmarkモードを追加。qbert・128 lane・batch256・131,072経験、最初の16,384経験以降のelapse_hour差とprocess全体の実時間を記録し、同じ入力でcombined/decomposedを交互に3巡する。現用設定と実行体hashを固定し、他Run/build/testと1〜9時を起動前後に検知したら停止する。事前検査のみ確認済みで、別Run AnetRLRunner_ab12 PID56408を検出して測定を起動しなかった。測定は未実施。

- 最新（22:20台）: `hint-policy-red-r2.xml`でQRのUQE選択とIQN full query混入の6失敗を確認し、`MakeActionScores`で同じforwardと抽選済みτのscoreを使うよう修正。hintとTraceはHead別のrisk領域だけを使う。GREEN（hint-policy-green.xml、7,120項目）。初回REDはfixture内のint/float混在を修正した。
- Actorのβ=0・1行動・lane範囲外・WithAction不変・sessionの追加検査、損失重み0.5、GridMaze既定＋run.@rnd＋decomposed＋任意群の解決を加え、`actor-boundaries.xml`で10,801項目成功。実Agentのfixed β=0とlearner β=10も、argmaxが異なる64状態で検査し4項目成功（actor-beta.xml）。
- メトリクスprofile名はPRDの正本どおり`metrics.scalar.@value_decomposition`とした。群名は空き番号の`77_agent_value`。任意群は22行、既定外、eval/eval_targetのみ。全shipped_configはshipped-green.xmlで3,294項目成功、その後のGridMaze指定構成もactor-boundaries.xmlで成功。default leavesは34,630代入・4,142検査・エラー0（default-leaves.log）。
- Replayは2/3成分×通常/done/truncated/両方×prefetch有無、固定/可変gamma、窓と成分数0の拒否で888項目成功（replay-components-r2.xml）。初回はfixtureが保持3件に8件抽選を要求したため16例外で、抽選数を保持件数へ合わせた。本体の修正は不要だった。
- checkpoint判定表のAgent側の分解・feature_gradient・外的/内的kind不一致も成功（checkpoint-modes.xml、148項目）。旧constant checkpointの受入はまだ。
- 020/030/140/150/200/212/214/READMEの日英がある。8組の見出し・表・fence件数は一致、英語版のコード外CJKと存在しないリンクは0。140の既存翻訳済みMermaidもskillのコードブロック保持規約に合わせて原文へ揃えた。020/030/200/212の対応hashは更新済み。最終の全構造・hash検査は改めて行う。
- PRD072検証資材の現行READMEと実装は、commit107a62c8に固定した履歴入力・goldenであり、上書き禁止である。PRD121が現用基準として挙げたpathとは現在の用途が異なる。元goldenは保持し、run_validation.py configで現用17構成を別に固定し、agent.txt/metrics_scalar.txtだけをHEAD版へ戻した参照と比べる。その他のユーザー変更は前後両方へ同じものを使う。最初のr1は廃止済みrun.@nature_dqnで参照側が解決できず中断、A3のActor継承による追加キーの許容も補った。r2は現用17構成で実行中（config-compare-r2.log、session72533）。現在まで全構成で既存値・scalar定義の一致、追加キー31〜35件を確認中。今後は検証用exeを写してから走らせ、ビルドの再リンクと競合させない。
- 現在のコード検査はreturn_stdの実取り込み（return-intake）。fixtureのstep_tのnarrowingを修正し、intrinsic_test.cppのオブジェクトだけビルド済み（return-intake-object-build-r2.log）。設定比較が終わったらリンクしてテストする。まだGREENは未確認。
- 性能測定は未実施。22:20台でも別Run（AnetRLRunner_ab12.exe、DropMerge-03）が走っているので、そのRunは停止せず測定を待つ。

- 最新（22:00台）: 設定の未使用の明示的な非既定値のWARNはRED（ignored-red.xml、6失敗）→GREEN（ignored-green-r2.xml、58項目）。初回buildはProfiledValueConfigをjson化できず修正した。射影の未使用値の解釈はNN側が所有する。スケーラの共通Readerも未使用判定を持つ。
- 可塑性target captureは、分解＋Munchausen targetで2B行を返すRED（plasticity-red-r5.xml、3失敗）→nextのB行だけへ切り出してGREEN（plasticity-green.xml、既存plasticity込み6,186項目）。先行r1〜r4はfixtureのconst、実行設定参照、quantile_modeの誤りを直した記録。checkpointの未知headerの期待一覧へ第3版も加えた。
- 動的な外的スケーラ2種×内的off/constant/2種×combined/decomposedの保存復元は80項目成功（scaler-checkpoint.xml）。return_stdのlane数2/3の復元、統計保持・割引和の維持/初期化・WARN回数・種類不一致を加えて90項目成功（scaler-lanes.xml）。Agent側の種類・分解・feature_gradient不一致表と旧constant形式の受入は残る。
- 出荷設定のRED（shipped-red.xml、value_mode未定義）後、agent.txtへ既定combined、分解用既定、内的恒等スケーラ、actor3項目のlearner βを追加した。gammaとfixed action_betaは既定値を置かない。metrics_scalar.txtへ任意の@value/77_agent_valueの22行を追加した。shipped-green-build.logでビルドし、全shipped_configを実行中（session72252）。テストの期待keyを既存の正本のextrinsic_q_sa_mean等へ揃えた。PRD072の現用基準比較とdefault leaves検査はまだ。
- 140日英文書を更新し対応hashをENへ記録（3294f91ff6d2d3e2bce673134bd82d06826b16f9）。200JPは全文を読み、5列、分解取り込み、動的checkpoint、診断と検証入口を更新。200ENも同じ箇所を訳し始めたが、対応hash・構造検査と残る細部の整合は未了。212/020/030は未編集。途中の自動差分生成はread-only Python→apply_patchで適用している。
- 性能測定は未実施。残る主項目: Replay3成分/終端/打ち切り/prefetch、Actor beta0/1action/full-queryとhintのrisk領域、split decay/schedule/AMP、checkpoint不一致表、文書、既定5構成のafter比較、Debug/Release全build・全suite、自己点検・独立レビュー。

- 最新（21:30台）: `targets-numeric-r2.xml`は768構成・5,376項目成功。TD/QR/IQN、TBO、Munchausen off/target/online/online_reuse、共有/専用gamma、episodic/non_episodic、外的bootstrap2種、Double DQN、損失重み0.2/1を直積にし、4 laneに通常・done・truncated・両方を置いて、独立したdoubleの目標値・損失と比較した。最初はIQN fixtureの方策にquantile_modeを書き忘れ、64条件でtaus欠落だった。fixtureを直して成功した。
- Replay fitは実際にprobeを通すRED（`replay-fit-red-r2.xml`）→Headごとの目標値・重み付き誤差/損失へ接続しGREEN（`replay-fit-green.xml`7,186項目）。policy churnの結合漏れはRED→GREEN（`churn-green.xml`522項目）。内的HeadだけNoisyの平均/抽選forwardの結合漏れもRED→GREEN（`noisy-green.xml`405項目）。公開Q抽出関数のnulloptはRED→GREEN（`value-function-green.xml`407項目）。
- 現在は外的bootstrapのUQEの選択。QRの平均Qのargmaxとpoint/tail UQEが違う2条件でRED（`risk-red-r2.xml`。その前のrisk-red.xmlはfixtureのEnvの行動数1が原因）。DQNの既存SelectFromValuesにgreedy_onlyと抽選済みτの再利用を足し、同じforwardへ方策を当てるよう修正した。診断用SelectActionもforward_outputを保持する。Thompsonのτを外的の選択で引き直さない。`risk-green-build-r2.log`でビルド/検査中（session 47580）。最初のrisk-green-build.logはVsDevCmdパスの誤記で起動前に失敗しており、書き直した。
- 任意メトリクスの新しい群は、PRD108の将来の分類と既存の割り当て表を照合して空いている`77_agent_value`を使う予定。PRD108の既存tagの移行は本作業に入れない。まだagent.txt/metrics_scalar.txtを編集していない。

- 2026-11-14 21時台の最新位置: Traceの遅延approx_priorityも2成分へ接続し、`trace-components-red.xml`の追加成分不足→`trace-components-green.xml`で2,823項目成功（既存PRD127/129込み）。Actorが行動時に評価した内的gammaを渡す。TraceのRND採点はActorの推定器の写しを使い、スケーラ統計を更新しない。
- 休眠中のfixed action_betaの`.value()`でBad optional accessになるRED→分解時だけ評価してGREEN、`inactive-beta-green.xml`396項目。gammaのscheduleだけを明示した場合の必須値判定もRED→GREEN。対象640項目（`gamma-green.xml`）。
- DQN全体の検査は142,718項目、38失敗・1例外（`metric-green.xml`）。失敗は3 caseに集まり、dqn_munchausen_test.cppの3列hintのfixtureとshape、checkpoint-formatの動的スケーラで旧形式を期待する断定だった。5列と第3版へ移行後の該当caseは`gamma-green.xml`で成功。全DQN/既定suiteの最終再実行は未了。
- 文書を編集開始: 214日英の新設、150日英の成分・API・検証入口、README日英の214索引、CONTEXTの5列定義、testdata/prd121/README.md、glossaryの3語。README/150/214の日英の見出し・表・fenceの件数は一致、コードブロック外の未訳CJKは0。リンクの全検査はまだ。ほかの指定文書（140/200/212/020/030）は未編集。214/README/150の翻訳元情報は実際に対応を確認した版のhashを記録した。
- 次のTDDは分解時のReplay fit診断。tracerに購読を追加し、probe batchを1として実際に診断forwardを通す。`replay-fit-red-build.log`でビルド中（session 32586）。テスト本文のB=1・有限値検査をビルド開始後に追記したため、結果を見たあと必ずもう一度ビルドしてREDを確定する。まだ本体のReplay fitを直していない。

- 追加のRED→GREEN: 初期優先度の2成分の誤差和（`priority-red.xml`→`priority-green.xml`）、ActorのHead別5列hintとWithAction（`hint-red.xml`→`hint-green.xml`、既存actor_initial込み446項目）、Traceの4鍵（`trace-red.xml`→`trace-green.xml`、454項目）、splitのclip・Adamの数値（`split-red.xml`→`split-green.xml`、既存skip込み431項目）、splitのAdam状態の保存復元（`split-checkpoint-red.xml`→`split-checkpoint-green.xml`、455項目）。
- splitは通常/fused Adam × clip有無の4条件で、同じ初期重みから微分した参照更新の和と一致。checkpointで主optimizerだけを戻すと12箇所の完全一致が失敗するREDを確認し、内的optimizerも保存復元してGREENにした。勾配normはclip前に計算する。共有部分のweight decayは外的の更新のみ、内的Headは内的の更新のみで掛ける。schedule・FP16・非ゼロdecayの追加検証は残る。
- 現在の次の確認位置: Agentへ9個のLearnerメトリクスを公開する経路（4鍵しか明示転送していない）、Traceの遅延approx_priorityの2成分化、診断forward全経路へのcombiner、Learnerの数式の数値検査、設定判定表、checkpoint全行、文書・設定と受入全体。Traceの即時4鍵だけが通ったことをTrace全体の完了とは扱わない。

- Actorのlane値・平均の6条件はGREEN、171 assertions（`action-diagnostics-green-r2.xml`）。評価sessionの累積とリセットもRED→GREEN、`[prd121]` 267 assertions成功（`action-session-green.xml`）。
- skipだけでなく価値の診断も同じsessionの境界でリセットするようになったため、private関数を `ResetSkipSession` → `ResetSessionDiagnostics` に改めた。現用の旧名は残していない。
- 次の縦スライスは初期優先度。5列の既知のhintと2成分の収益・割引を使い、誤差を絶対値にして重み付きで足す。重み1・0.2・0、TBO有無でREDを確認中（`priority-red-build.log`・`priority-red.xml`）。
- 未実装・未検証の主な残り: 5列hintのpack/decode・Actor/WithAction/TraceのHead別値、初期優先度の内的成分、mergeのHead勾配norm、splitの独立clip/Adamと保存状態、QR/IQNを含むLearnerの数式・各bootstrap/終端モード・Munchausenの受入、各診断forwardへの結合の適用、設定の未使用値の通知とrequired gammaのschedule対応、checkpoint判定表全行とlane変更、agent.txt/metrics_scalar.txt/出荷設定golden、設計文書と英訳・CONTEXT、無効5構成の新旧比較、Debug/Release全ビルド・全suite・自己点検・独立レビュー。
- 性能測定は未実施。開発者の指定した時間帯と、他Runが動いていない条件を守る。今までのRun所要時間を性能測定値に読み替えない。

## 検証の経過

- tracer bullet: `[prd121]` がGREEN。TD・Actor・Replay・1回以上の更新、Head別の損失とQの取得を確認した。`.scratch/prd121/tracer-green.xml`・`tracer-green.log`、ビルドは `tracer-green-build-3.log`（終了値0）。これは受入全体の完了ではない。
- 既定5構成の基準採取はすべて終了値0。Atari RND 33,216点、LunarLander RNDあり44,679点・なし37,794点、GridMaze 26,950点、DropMerge 31,799点、合計174,438点。ハッシュと入力・実行体・Runへのポインタは `baseline-summary.json`。各構成は8,704 experience、8 lane、seed1、学習warmup256、batch16、同期評価は4,096ごとに1 episode。各実行の時間は正しさの検証の所要時間であり、性能の測定値ではない。
- 2つ目の縦スライスは `Replay recomputes only the scheduled reward component`。定数成分をPush時のgamma0.5のまま保持し、schedule成分だけSample時のgamma0.25で再計算することを、既知の2成分の報酬で検査する。REDを確認中（起動時session 26923）。
- 保存した実装前coreテストの既定suiteを実行開始。通常ログは `baseline-core-suite.log`、JUnitは `baseline-core-suite.xml`。成功・失敗は終了後に記録する。
- 保存した実装前core suiteは終了値0、179,997 assertions、失敗・エラー0、skip1、445.597秒で完了した。
- 成分ごとの再計算のREDは、内的成分の収益が4になり、期待する3と違うことを確認（`replay-red.xml`）。抽出器が成分ごとの再計算の旗も参照するようにしてGREEN。既存Replayを含む `[prd121],[replay_buffer]` は18,325 assertions、失敗0（`replay-green.xml`）。
- return_stdのREDは、既存の統計部品が分散を標本数-1で割る事実に手計算を合わせて取り直した（`scaler-red-r2.xml`）。2回目の入力を割引和でなく報酬そのもので正規化しているための失敗を確認。lane別の割引和を統計へ渡す最小実装でGREEN、既存scaler込み49 assertions成功（`scaler-green.xml`）。統計の分散とepsilonは既存のRunningMeanStdと同じとする。
- 動的な外的報酬スケーラのAgent checkpoint往復は、復元後のcountが0になっていたRED（期待2、`checkpoint-red.xml`）から、状態の保存・復元を追加してGREEN。`[prd121],[scaler]` は53 assertions成功（`checkpoint-green.xml`）。動的な状態または分解を含むAgentのcheckpointは第3版とし、旧い定数スケーラ・combinedの書式は維持する。内的スケーラの接続と保存の判定表全行の検証はまだ未完了。
- 次の縦スライスは、内的報酬スケーラの定数倍率2がcombinedの学習報酬へ反映されること。既知のTD値が4から6へ変わる条件でREDを確認中。
- 内的スケーラのREDはTDが4のままで、期待6にならないことを確認（`intrinsic-scaler-red.xml`）。共通のRewardScalerConfig読取りを追加し、取り込み・Traceの統計更新なし変換・内的スケーラの保存を接続してGREEN。既存RND/Traceと合わせて4,734 assertions成功（`intrinsic-scaler-green.xml`）。
- Actorの行動診断を実装中。TD/QR/IQN × TBO有無の6条件で、既知の2 lane・3行動の値から4指標のlane値と平均を確認する。REDは既知キーを取得できない失敗（`action-diagnostics-red.xml`）。device上で4列を作り、GetScalarの読出し時だけCPUへまとめる実装を追加した。`action-diagnostics-green-build.log` の最初のビルドは自由関数での名前空間修飾1件で失敗し、修正済み。まだGREENは確認していない。
- 同期呼出しのDQN用forwardアダプタを足した理由: NetworkForwardExecutorは継承や出力加工の口を持たず、ActionPolicyの中でforwardするため、NNの契約へβを入れずに、DQN側で出力を結合する境界が必要だった。読み手のq/q_distの意味は変えず、DQN内の引数の型をDqnForwardExecutorにした。既存のNetworkForwardExecutorから同期呼出しの間だけ借り、βを呼出しごとに渡す。旧APIのoverloadは残していない。
