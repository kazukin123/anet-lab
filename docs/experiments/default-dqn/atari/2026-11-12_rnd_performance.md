# AtariのRND性能測定

PRD116 §6.3のうちAtari分を測定した。Montezuma・64 lane・CUDAで、RNDなし／既定RND／Taiga設定をABC→CBAで3巡、各群6本ずつ比較した。warm-up後の同じ89,984経験に要した実時間の中央値は19.019秒／31.584秒／280.616秒だった。**既定RNDは時間1.661倍（経験/秒は39.8%低下）、Taiga設定は14.754倍（93.2%低下）**だった。この測定に速度の合否閾値は置かない。

## 条件と比較方法

| 項目 | 条件 |
| --- | --- |
| 実行環境 | Ryzen 7 9700X（8 core /16 thread）、メモリ61.6 GiB、RTX 5090、Windows 11 Pro 10.0.26300 |
| 実行体 | MSVC x64 Release。基準HEAD `fcaa39b5b9c6cd23221a9e5c484049e3bff08304`。当時の未コミットの変更（PRD125ほか）を含む同じ実行体を全Runで使用 |
| 実行体SHA256 | [機械可読データ](2026-11-12_rnd_performance.json)の各Runに保存。実行体・DLLと当時のソース差分は測定用のローカル複製に保持 |
| 環境 | Montezuma's Revenge（ALE 0.12）、64 lane、CUDA、run.seed=1 |
| Q学習 | Atari.txtの既定のチェーン（IQN、Impalaの幅2倍のNN、replay 2,097,152、replay batch 256、replay ratio 1、BF16）。Qの学習のwarm-upだけ20,000経験に縮めた |
| 既定RND（`run.@rnd`） | whiten・clipped・next_state、beta=0.1、output_dim=512、learn_ratio=0.25、learn_batch_size=32、warmup_steps=12,800（warm-up中もpredictorを学習）、Adam eps 1e-8 |
| Taiga設定（`run.@rnd > run.@rnd_taiga`） | variance_feedback・unclipped・state、beta=0.00005、learn_ratio=1、learn_batch_size=1、warmup_steps=0、Adam eps 1e-5 |
| 時計 | `elapse_hour @train $runner $exp_step interval:64`。32,768経験と122,752経験の時計差×3,600。起動とwarm-upを分母へ入れない |
| 区間 | 試走3本（各8,192経験）で、最速の群が約20秒になる長さをlane単位で切り上げた89,984経験 |
| 休止したもの | 評価session 5つ（eval・eval_panel・eval_target・greedy・greedy_single）、EvalPanel自動実行、描画、画像出力、終了時checkpoint保存。学習は継続 |
| 反復 | なし→既定→Taiga→Taiga→既定→なしを3巡。各Runを順次実行し、他のRun・ビルドを重ねない |

設定と実行体は測定前に複製した。3群の設定解決の全キー比較で、差はRNDの設定、群56の診断定義、Run名だけだった（なしと既定で24キー、既定とTaigaで9キー）。全Runの実効設定は下表と機械可読データから参照できる。

試走の8,192経験は、なし1.822秒、既定2.980秒、Taiga 25.804秒だった。試走3本とTracy用2本を通常比較の中央値へ混ぜない。

## 通常比較の結果

| 群 | 本数 | 中央値 秒 | 最小〜最大 秒 | 経験/秒 | なしとの比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNDなし | 6 | 19.019 | 17.252〜19.819 | 4,731.1 | 1 |
| 既定RND | 6 | 31.584 | 28.275〜32.416 | 2,849.1 | 1.661 |
| Taiga設定 | 6 | 280.616 | 254.353〜307.218 | 320.7 | 14.754 |

1巡ごとの比（各巡の群の中央値の比）は、既定/なしが1.665、1.698、1.533、Taiga/なしが14.754、15.793、13.774。3巡目は3群とも1・2巡目より速かった。1.661と14.754は全6本ずつの中央値の比であり、巡ごとの比の平均ではない。各群の学習系列は報酬の違いで変わり得るため、全体の時間差だけをRND内部の純粋な計算時間とは呼ばない。

| 順番 | 群 | 測定区間 秒 | 実効設定（Run名） |
| --- | --- | ---: | --- |
| 1 | なし | 19.448 | [run_20261112-205944_prd116_ap_r01_off](config/run_20261112-205944_prd116_ap_r01_off.txt) |
| 2 | 既定 | 32.058 | [run_20261112-210039_prd116_ap_r02_on](config/run_20261112-210039_prd116_ap_r02_on.txt) |
| 3 | Taiga | 289.710 | [run_20261112-210153_prd116_ap_r03_taiga](config/run_20261112-210153_prd116_ap_r03_taiga.txt) |
| 4 | Taiga | 271.522 | [run_20261112-210855_prd116_ap_r04_taiga](config/run_20261112-210855_prd116_ap_r04_taiga.txt) |
| 5 | 既定 | 31.272 | [run_20261112-211531_prd116_ap_r05_on](config/run_20261112-211531_prd116_ap_r05_on.txt) |
| 6 | なし | 18.591 | [run_20261112-211644_prd116_ap_r06_off](config/run_20261112-211644_prd116_ap_r06_off.txt) |
| 7 | なし | 18.084 | [run_20261112-211739_prd116_ap_r07_off](config/run_20261112-211739_prd116_ap_r07_off.txt) |
| 8 | 既定 | 31.895 | [run_20261112-211831_prd116_ap_r08_on](config/run_20261112-211831_prd116_ap_r08_on.txt) |
| 9 | Taiga | 307.218 | [run_20261112-211944_prd116_ap_r09_taiga](config/run_20261112-211944_prd116_ap_r09_taiga.txt) |
| 10 | Taiga | 291.008 | [run_20261112-212719_prd116_ap_r10_taiga](config/run_20261112-212719_prd116_ap_r10_taiga.txt) |
| 11 | 既定 | 32.416 | [run_20261112-213428_prd116_ap_r11_on](config/run_20261112-213428_prd116_ap_r11_on.txt) |
| 12 | なし | 19.795 | [run_20261112-213544_prd116_ap_r12_off](config/run_20261112-213544_prd116_ap_r12_off.txt) |
| 13 | なし | 19.819 | [run_20261112-213639_prd116_ap_r13_off](config/run_20261112-213639_prd116_ap_r13_off.txt) |
| 14 | 既定 | 28.275 | [run_20261112-213736_prd116_ap_r14_on](config/run_20261112-213736_prd116_ap_r14_on.txt) |
| 15 | Taiga | 256.261 | [run_20261112-213842_prd116_ap_r15_taiga](config/run_20261112-213842_prd116_ap_r15_taiga.txt) |
| 16 | Taiga | 254.353 | [run_20261112-214457_prd116_ap_r16_taiga](config/run_20261112-214457_prd116_ap_r16_taiga.txt) |
| 17 | 既定 | 28.542 | [run_20261112-215131_prd116_ap_r17_on](config/run_20261112-215131_prd116_ap_r17_on.txt) |
| 18 | なし | 17.252 | [run_20261112-215236_prd116_ap_r18_off](config/run_20261112-215236_prd116_ap_r18_off.txt) |

18本とも終了0。RNDありの全Runで補助学習の進行を確認し、記録したlearn_countの最大値は既定958、Taiga 122,688だった。試走とTracy用の実効設定は、[pilot_off](config/run_20261112-205534_prd116_ap_pilot_off.txt)・[pilot_on](config/run_20261112-205617_prd116_ap_pilot_on.txt)・[pilot_taiga](config/run_20261112-205702_prd116_ap_pilot_taiga.txt)・[profile_on](config/run_20261112-215328_prd116_ap_profile_on.txt)・[profile_taiga](config/run_20261112-215432_prd116_ap_profile_taiga.txt)。

## Tracyによる内訳

既定RNDとTaiga設定の別Runで、時計が32,768経験以上になったことを確かめてから[公式Tracy 0.13.1](https://github.com/wolfpld/tracy/releases/tag/v0.13.1)のcaptureを8秒接続した。通常比較と同じRelease実行体を使った。計測中のCPU側の区間時間であり、GPU kernel時間ではない。割合は、完結した `intrinsic_reward` の区間の合計（RND全体）を分母にしている。

**既定RND。** 完結した `intrinsic_reward` は388 intake（24,832観測）、`Estimate` は388回、`Learn` は194回だった。RND全体の合計は5,081.5ms、1 intakeあたり13.097msで、捕捉区間8.121秒の62.57%を占めた。

| 区間 | 合計 ms | 1 intakeあたり ms | 呼び出し | RND内の割合 |
| --- | ---: | ---: | ---: | ---: |
| 統計 | 3,184.0 | 8.206 | 388 | 62.66% |
| 正規化 | 605.7 | 1.561 | 582 | 11.92% |
| target forward（採点＋学習） | 309.9 | 0.799 | 582 | 6.10% |
| predictor forward（採点＋学習） | 298.8 | 0.770 | 582 | 5.88% |
| backward | 234.1 | 0.603 | 194 | 4.61% |
| optimizer step | 208.8 | 0.538 | 194 | 4.11% |
| 誤差の転送 | 95.6 | 0.246 | 582 | 1.88% |
| 入力の転送 | 94.9 | 0.245 | 582 | 1.87% |
| 観測の転送・結果の転送 | 1.1 | 0.003 | 970 | 0.02% |

`Estimate` は合計4,201.2ms（RND全体の82.68%）、`Learn` は842.4ms（16.58%）。正規化は1回あたり平均1.041msで、582回のうち194回は `Learn` の入力の正規化だった。

**Taiga設定。** 完結した `intrinsic_reward` は45 intake（2,880観測）、`Estimate` は45回、`Learn` は2,880回だった。RND全体の合計は7,658.1ms、1 intakeあたり170.18msで、捕捉区間7.985秒の95.91%を占めた。

| 区間 | 合計 ms | 1 intakeあたり ms | 呼び出し | RND内の割合 |
| --- | ---: | ---: | ---: | ---: |
| optimizer step | 2,749.2 | 61.093 | 2,880 | 35.90% |
| backward | 2,199.9 | 48.887 | 2,880 | 28.73% |
| predictor forward（採点＋学習） | 894.2 | 19.871 | 2,925 | 11.68% |
| target forward（採点＋学習） | 647.1 | 14.380 | 2,925 | 8.45% |
| 正規化 | 552.2 | 12.271 | 2,925 | 7.21% |
| 誤差の転送 | 348.8 | 7.751 | 2,925 | 4.55% |
| 入力の転送 | 141.4 | 3.142 | 2,925 | 1.85% |
| 統計 | 26.0 | 0.578 | 45 | 0.34% |
| 観測の転送・結果の転送 | 3.4 | 0.076 | 2,970 | 0.04% |

`Estimate` は合計462.2ms（6.03%）、`Learn` は7,137.3ms（93.20%）。採取端で途切れた区間に含まれるイベントは、既定16件、Taiga 569件を集計から除いた。各区間のCSVは `data/2026-11-12_rnd_performance_{on,taiga}_{zones,events,facade_events}.csv` に保存した。
