# GridMazeのRND性能測定

PRD116 §6.3のうちGridMaze分を測定した。RNDなし/既定RNDをABBAで3巡、各群6本ずつ比較し、warm-up後の同じ165,632経験に要した実時間の中央値は23.261秒/31.148秒だった。**RNDありは時間1.339倍（33.91%増）、経験/秒は25.32%低下**した。この測定に速度の合否閾値は置かない。AtariとTaiga設定は今回測定していない。

## 条件と比較方法

| 項目 | 条件 |
| --- | --- |
| 実行環境 | Ryzen 7 9700X、8 core /16 thread、メモリ61.6 GiB、Windows 11 Pro 10.0.26300 |
| 実行体 | MSVC x64 Release。基準HEAD `3d1960c7bc4ec027451b306ca31ee520550a2bb0`。当時の未コミットGUI変更も含め、同じ実行体を全Runで使用 |
| 実行体SHA256 | 機械可読データの各Runに保存。実行体・DLLと当時のソース差分は測定用のローカル複製に保持 |
| GridMaze | 256 lane、CPU、torch thread 1、CUDA不可視、run.seed=1、環境seed=6545537233705677626の固定盤面 |
| Q学習 | 当時のGridMaze設定を複製。replay batch 128、replay ratio 1、capacity 524,288、stack 1、PERあり、AMPなし |
| RND | 既定のwhiten /clipped /next_state、beta=0.1、output_dim=128、learn_ratio=0.25、learn_batch_size=32、warmup_steps=12,800 |
| 時計 | `elapse_hour @train $runner $exp_step interval:256`。16,384経験と182,016経験の時計差×3,600。起動とwarm-upを分母へ入れない |
| 終了 | 182,528経験を要求し、測定端点の出力を確保。保存された診断値を最終checkpoint内の回数とは扱わない |
| 休止したもの | 定期評価、EvalPanel自動実行、描画、画像出力、終了時checkpoint保存。学習は継続 |
| 反復 | OFF→ON→ON→OFFを3巡。各Runを順次実行し、他のRun・ビルド・テストを重ねない |

設定と実行体は測定前に複製した。設定解決の全キー比較はRND設定、群56の診断定義、Run名の差だけだった。実際のRunの設定も `inspect_run.py config --effective-only --diff` で照合し、Run名とRNDの有効化・Network設定以外の差はなかった。実効設定の写しは下表と[機械可読データ](2026-11-07_rnd_performance.json)から参照できる。

試走は各群8,192経験の測定区間で1.057秒/1.485秒だった。遅い群で約30秒を測るよう本測定の件数を選んだ。試走2本とTracy用1本を通常比較の中央値へ混ぜない。

## 通常比較の結果

| 群 | 本数 | 中央値 秒 | 最小〜最大 秒 | 経験/秒 |
| --- | ---: | ---: | ---: | ---: |
| RNDなし | 6 | 23.260999 | 21.183999〜24.516002 | 7,120.59 |
| 既定RND | 6 | 31.147997 | 30.069998〜31.587000 | 5,317.58 |

1巡ごとのON/OFF比は1.4084、1.3177、1.3057で、RNDなし側にもばらつきがある。1.3391は全6本ずつの中央値の比であり、巡回ごとの比の平均ではない。各群の学習系列は報酬の違いで変わり得るため、全体の時間差だけをRND内部の純粋な計算時間とは呼ばない。

| 順番 | 群 | 測定区間 秒 | 実効設定（Run名） |
| --- | --- | ---: | --- |
| 1 | OFF | 22.150 | [run_20261107-230211_prd116_gp_r01_off](config/run_20261107-230211_prd116_gp_r01_off.txt) |
| 2 | ON | 30.070 | [run_20261107-230238_prd116_gp_r02_on](config/run_20261107-230238_prd116_gp_r02_on.txt) |
| 3 | ON | 30.960 | [run_20261107-230313_prd116_gp_r03_on](config/run_20261107-230313_prd116_gp_r03_on.txt) |
| 4 | OFF | 21.184 | [run_20261107-230350_prd116_gp_r04_off](config/run_20261107-230350_prd116_gp_r04_off.txt) |
| 5 | OFF | 24.371 | [run_20261107-230415_prd116_gp_r05_off](config/run_20261107-230415_prd116_gp_r05_off.txt) |
| 6 | ON | 31.587 | [run_20261107-230444_prd116_gp_r06_on](config/run_20261107-230444_prd116_gp_r06_on.txt) |
| 7 | ON | 30.729 | [run_20261107-230521_prd116_gp_r07_on](config/run_20261107-230521_prd116_gp_r07_on.txt) |
| 8 | OFF | 22.920 | [run_20261107-230557_prd116_gp_r08_off](config/run_20261107-230557_prd116_gp_r08_off.txt) |
| 9 | OFF | 23.602 | [run_20261107-230627_prd116_gp_r09_off](config/run_20261107-230627_prd116_gp_r09_off.txt) |
| 10 | ON | 31.336 | [run_20261107-230655_prd116_gp_r10_on](config/run_20261107-230655_prd116_gp_r10_on.txt) |
| 11 | ON | 31.494 | [run_20261107-230732_prd116_gp_r11_on](config/run_20261107-230732_prd116_gp_r11_on.txt) |
| 12 | OFF | 24.516 | [run_20261107-230808_prd116_gp_r12_off](config/run_20261107-230808_prd116_gp_r12_off.txt) |

12本とも終了0。RNDありの全Runで補助学習の進行を確認し、記録したlearn_countの最大値は1,322だった。

## Tracyによる内訳

別のRND有効Runで、時計が16,384経験以上になったことを確認してから[公式Tracy 0.13.1](https://github.com/wolfpld/tracy/releases/tag/v0.13.1)のcaptureを8秒接続した。通常比較と同じRelease実行体を使った。計測中のCPU側の区間時間であり、GPU kernel時間ではない。

完結した `intrinsic_reward` は171 intake（43,776観測）、`Estimate` は171回、`Learn` は342回だった。intakeあたり2回・各32観測の補助学習という設定と整合する。RND全体の合計は2,339.370ms、1 intakeあたり13.681ms。以下の割合は、このRND全体を分母にしている。

| 区間 | 合計 ms | 1 intakeあたり ms | RND内の割合 |
| --- | ---: | ---: | ---: |
| target forward（採点＋学習） | 677.506 | 3.962 | 28.96% |
| predictor forward（採点＋学習） | 643.572 | 3.764 | 27.51% |
| backward | 385.904 | 2.257 | 16.50% |
| optimizer step | 435.150 | 2.545 | 18.60% |
| 統計 | 110.366 | 0.645 | 4.72% |
| 最初の観測正規化 | 28.688 | 0.168 | 1.23% |
| 転送名の区間と有限性検査等 | 28.970 | 0.169 | 1.24% |
| 上記以外 | 29.215 | 0.171 | 1.25% |

forwardが56.47%、backwardとoptimizerが35.10%を占めた。`Estimate` 全体は平均6.476ms、`Learn` 全体は1回平均3.535msだった。

「転送名の区間」はobservation/input/error/result transferの合計で、CPU上のTensorの扱いや誤差の有限性検査、診断値の設定などを含む。GPUへの転送量・GPU待ちを測った値ではない。統計の区間には二重正規化の2回目も入るが、誤差・target出力の統計と一緒なので、その費用だけを分離した値ではない。

CSVの `total_perc` は採取前も含むprocess起点の時計を分母にするため使わず、完結したRNDの親区間に含まれる子区間を集計した。除外されたRNDイベントは0。完結した親区間の最初から最後までは8.134469秒で、RND区間の合計はその28.76%だった。計測負荷を含む別Runの内訳を、通常比較のON/OFF差の代用品にしない。

生の[区間別CSV](data/2026-11-07_rnd_performance_zones.csv)、[RNDイベントCSV](data/2026-11-07_rnd_performance_events.csv)、[facadeイベントCSV](data/2026-11-07_rnd_performance_facade_events.csv)と、集計・trace SHA256は機械可読データに保存した。元のtraceは測定用ローカルフォルダの `cases/profile_on/capture.tracy` に保持している。

## 確認できた範囲と残件

- GridMazeのCPU・256 lane・既定RNDについて、§6.3の3巡以上の中央値とTracy内訳を記録した。成績・新しさ判別能力の評価ではない。
- Atari B=64のRNDなし/既定RND/Taigaの比較は未実施。画像入力やGPU実行の費用へ、この数字を移さない。
- 観測2値のGridMazeではforwardと補助学習が費用の大半だった。二重正規化の専用APIの要否は、この小さい入力だけでは決めず、Atariの内訳も材料にする。
- 出荷設定とRND本体は変更せず、測定用の設定だけを複製した。設定・出力の写しは元の改行も保持する。保存SHA256は取得時のバイト列に対する値である。

## 検証資材

[PRD116検証資材](../../../../core/anet-core/testdata/prd116/README.md)の `run_performance.py` を使った。段階はprepare、pilot、measure、profile、summarizeで、測定区間は保存済みpilot.jsonから固定する。各caseの終了値・時計・設定・実行体の識別情報は実行記録へ保存する。測定ログと元Runは `.scratch/prd116/grid-performance-20261107-r1/`、各段階の要約ログは `.scratch/prd116/grid-performance-*.log` にある。
