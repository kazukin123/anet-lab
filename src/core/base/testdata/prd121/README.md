# PRD121 受入検証

正本は [PRD121](../../../../../docs/memo/121_extrinsic_intrinsic_value_decomposition_10prd.md)、経過と成否は [20impl](../../../../../docs/memo/121_extrinsic_intrinsic_value_decomposition_20impl.md)を参照する。ここは再実行の入口であり、未実施の検証を成功として扱わない。

## 保存した基準

本体変更前のDebug・Release Runner、DLL、Debug coreテスト、設定を`.scratch/prd121/baseline`へ保存した。snapshotは既存の構成別フォルダを上書きしない。入力は`.scratch/prd121/inputs`に凍結し、再実行時に内容が変わる場合は拒否する。

5構成はAtari RND、LunarLander RNDあり・なし、GridMaze、DropMerge。seed 1、決定的なFP32、8 lane、8,704 experience、batch 16、Replay 4,096、warm-up 256、replay ratio 1、同期評価は4,096ごとに1 episodeである。Atariはqbert、AtariとDropMergeはCUDA、ほかはCPU。RNDはbatch 8、warm-up 256。Runプロファイルの選択チェーンは凍結した出荷設定から作る。

基準は合計174,438 scalar点。内訳はAtari RND 33,216、LunarLander RND 44,679、LunarLander 37,794、GridMaze 26,950、DropMerge 31,799。実行体と入力のハッシュ、Runの所在は`.scratch/prd121/baseline-summary.json`にある。これは正しさの検証であり、性能の測定ではない。

## 自動テスト

Debug coreテストの`[prd121]`を使う。Replay、DQN、RND、scalerの既存検査も実施する。ビルドはリポジトリのMSVC初期化手順に従い、通常ログとJUnitを`.scratch/prd121`へ保存する。

| 対象 | 主な確認 |
|---|---|
| Agentの取り込みと更新 | Actor→2成分Replay→TD更新、Head別損失とQ、Agentメトリクス |
| Replay | 定数成分の保存、schedule成分だけの再計算、終端、打ち切り、prefetch、窓拒否 |
| Learner | TD・QR・IQN、TBO、Double DQN、bootstrap、gamma、終端モード、Munchausen |
| 最適化 | merge、重み、PERの絶対値の和、splitの独立clipとAdam、非有限勾配で両更新前に停止、保存復元 |
| Actor | fixed/learnerのβ、5列hint、WithAction、lane診断、平均、session累積とreset、Q専用TraceのRND費用分離 |
| スケーラ | reward_std、return_stdのlane別割引和、更新しない変換、内的の適用順、保存復元 |
| 設定 | 型・値域、必須値、休眠、未使用値の通知、Headの入力とshape、任意メトリクスの組 |
| checkpoint | モード・種類の不一致、旧い定数倍率、動的状態の継続、lane数変更 |

## 無効時の同一性

`run_validation.py`はPRD109の起動・ハッシュの処理とPRD084のscalar正規化を使う。`snapshot`で実行体を保存し、`prepare`で入力を凍結、`before`で基準を採取、`after`で変更後を採取、`compare`で比較する。基準の`before --label baseline`は採取済みである。別の採取には別のlabelを使い、既存のRunを混ぜない。

比較対象はtagの集合と、時間・資源を除く全scalar点の完全一致である。値の近似一致や選んだtagだけの比較で置き換えない。summaryの終了値と失敗欄、正規化したJSONLのSHA256、comparisonの`exact_match`を確認する。

## 性能測定

Atariの同じ入力、同じ実行体、Body共有・mergeで、combinedとdecomposedを交互に3巡以上測る。閾値は置かない。各Runの実時間と測定条件、比率を記録する。測定中はほかのRun、ビルド、テストを重ねない。

2026-11-14の開発者指定により、性能測定はできるだけ25時前か朝9時以降へ寄せる。既存の別Runは止めない。別Runが動いている場合は待つ。正しさの検証の所要時間を性能の数値へ転用しない。

`benchmark`はRelease Runnerを使い、qbert・seed 1・CUDA・128 lane・batch 256・Replay 65,536・学習とRNDのwarm-up各4,096・replay ratio 1・RND batch 32・131,072 experienceで測る。評価と終了時保存は止め、16,384 experience以降の`90_perf/90_elapse_hour`の最初と最後から同じ区間の実時間とexperience/秒を求める。プロセス全体の実時間も別に残す。AMPなどの継承値を含む設定一式、実行体のSHA256、GPU・driver、開始時刻は出力先の`config`と`summary.json`に保存する。

起動前と各Runの間に、他のRunner・coreテスト・cl・ninjaがいないことを確認する。01:00〜09:00には開始しない。失敗時はRunを成功として補わず、途中までの記録を残す。新しいlabelで最初から3巡を測り直す。

## 設定と旧checkpoint

`config`は現用17構成（Atari 5、DropMerge 3、LunarLander 3、GridMaze 2、CartPole、GridMaze MuZero、ImageCls 2）の全キー・文字列値を比較する。参照側は現在の設定を複製して`agent.txt`と`metrics_scalar.txt`だけをHEADへ戻すため、並行するEnv設定の変更を両側で同じに保つ。許す差はPRD121の追加既定値だけで、既存のscalar定義も一致させる。テスト実行体を採取先へ複製して使う。

PRD072のgoldenはcommit `107a62c8`の履歴であり、現行で廃止済みの`run.@nature_dqn`などを含む。上書きせず、現用17構成の差分を`comparison.json`へ記録する。2026-11-14の`acceptance-r2-config`と`config-20261114-2244`で始まる採取は、どちらも17構成すべて成功した。

`legacy`は変更前のAtari RND・DropMergeの`agent_close.anet`を変更後Debug Runnerの`DefaultDQNAgent.auto_load_file`から読み、学習を止めて8 experience進める。保存を止め、読んだcheckpointのSHA256が変わらないことも確かめる。新しい形式の種類・モード・状態の判定はcoreテストで検査する。

## PRDの受入項目との対応

テスト名はCatchの一覧から検索できる。数値テストはFP32で相対誤差1e-5、零付近の絶対許容1e-6を使う。分解のTBO逆変換は同じ逆関数を桁落ちのない式で計算し、既存combinedの計算は保持する。実施日時、ログと成否は20implを正とする。

| PRD §9.2 | 検証する入口 |
|---|---|
| 1 | `before`・`after`・`compare`、5構成の全scalarとtag集合 |
| 2 | `replay_buffer_test.cpp`の既存検査 |
| 3 | `Replay recomputes only the scheduled reward component`、2/3成分・4終端条件・prefetch・窓拒否 |
| 4 | `Shared episodic component targets reproduce a combined learner target`、TD/QR/IQN×Munch×TBO |
| 5〜8・10 | `Decomposed learners match component Bellman targets and losses`、既知の値の組み合わせ |
| 9 | `Decomposed split adds independently clipped Adam updates`、片方だけclip・重み減衰・schedule・保存復元 |
| 11 | `Decomposed combination preserves tau pairs and exact zero beta values`、Headのshape検査 |
| 12 | `Decomposed Agent selects fixed-zero and learner beta per Actor` |
| 13 | `Decomposition config rejects invalid explicit values even when inactive`、`Decomposition validates all Actor and Head values before combinations`、`Decomposition mode errors precede numeric errors in other components`、`Decomposition validates individual fields before scaler combinations`、`Decomposition still rejects each active scaler combination`、unused診断・専用Bodyの不足キーと入力shape検査 |
| 14 | `Return reward scaling follows per-lane discounted returns`、lane数変更の保存復元 |
| 15 | `Intrinsic return scaling masks before statistics and stores components before beta` |
| 16〜17 | `DQN decomposed values learn both heads through Actor and Replay`、AMP・Trace・診断の検査 |
| 18 | `DQN checkpoint resumes dynamic reward scaler statistics`、`legacy`、lane数変更の検査 |
| 19 | `DQN initial priorities sum independent head errors`、既存Actor hint検査 |
| 20 | `benchmark`、同一構成で交互に3巡 |
| 21 | 214・140・150・200・212・020・030の日英、索引・用語集・起草済みADRとの照合 |
| 22〜23 | `Decomposed Actor reports the intrinsic effect on action selection`、`Value decomposition ships inactive defaults and optional metrics` |

## 2026-11-15の検証結果

cc-review前のDebug・Release全ビルドは成功。既定coreは192,103 assertion・失敗/ERROR 0、初回のレビュー修正後の関連まとめは19,882 assertion・失敗/ERROR 0。5構成の全174,438 scalar点とtag集合は変更前と完全一致し、現用17構成の設定比較、旧constant checkpointのAtari RND・DropMergeの読込みも成功した。初回の独立レビューは計画・実装ともOK。

性能3巡は`.scratch/prd121/final-20261114-235358-performance/summary.json`へ保存。Release・RTX 5090・driver 616.92・共有Body・merge・IQN・Learner BF16・128 lane・batch 256で、全6回とも20,096〜130,048 experienceを比較した。学習区間の中央値はcombined 28.099秒、decomposed 29.434秒（比1.04751）。各回・起動終了込みの時間・実行体hash・条件は20implとsummaryを参照する。25時前に完了した。

2026-11-15のcc-reviewでは、設定の検査順序と専用Bodyの不足キーの診断を修正した。現在の17構成での比較を採る方針は開発者が承認し、PRD §8.7へ反映済み。修正後のビルド・関連テストと再レビューの最終結果は20implのcc-review記録を正とする。性能の計算経路は変えず、上記6 Runの実測を保持し、再測定はしていない。

## 人が確かめること

EvalPanelのEvaluation TraceへHead別の4鍵と、行動選択のlane別4鍵を任意列として追加する。2 lane以上の評価で、行ごとにそのlaneの値が出ることを確かめる。既定列には追加しない。実画面での確認を行っていない場合は、実装メモへ未実施と記録する。
