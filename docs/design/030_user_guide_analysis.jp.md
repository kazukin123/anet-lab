# Run分析ユーザーガイド

> 主たる観点: 行程単位（成果物確認、可視化、比較、解釈）

## 1. はじめに

### 1.1 目的

本書は、Runが出力した`metrics.jsonl`をMetrics Viewerで可視化し、複数Runを同じ条件で比較し、必要に応じてOptunaのmulti-seed集計を確認する手順を説明する。

### 1.2 対象読者

- 学習曲線、評価値、loss、性能指標を比較する利用者
- Run間の差を過大評価せず、同じstep範囲で判断したい利用者
- DropMerge Optunaのseed runとsummary studyを使い分ける利用者

### 1.3 記載範囲

現行のJava/Spring Metrics Viewer、Plotly画面、Run artifact、DropMerge Optunaの閲覧方法を扱う。
新しいmetricの実装方法やOptuna探索空間の変更は設計文書と[DropMerge Optuna利用ガイド](../optuna.md)を参照する。

## 2. Metrics Viewerを起動する

本書はMetrics Viewerのjarがbuild済みであることを前提とする。Java、Maven、jarのbuild手順は[開発環境](040_development_environment.jp.md)を参照する。

通常Runを見る場合は`apps/runner`から次を起動する。

```powershell
22_metrics_viewer_java.bat
```

- URL: `http://localhost:8082`
- workspace directory: `apps/runner/workspaces`

画面上部の`Workspace` selectorで、`runs/`または`config/`を持つworkspaceを切り替える。
前回選択はbrowserへ保存され、次回起動時に同じworkspaceが存在すれば復元される。
保存済みworkspaceが存在しない場合はserverのcurrent workspaceへ戻る。

独自のworkspace群を読む場合はjarへ`--metricsviewer.workspaces-dir=<path>`を渡し、必要なら
`--metricsviewer.initial-workspace=<name>`で起動時workspaceを指定する。Viewerは選択workspaceの
`runs/`直下で、`metrics.jsonl`または`metrics.jsonl.gz`を持つdirectoryだけをRunとして認識する。
両方がある場合は`metrics.jsonl`を使う。

### 2.1 完了Runのmetricsを圧縮する

`apps/70_compress_workspace_metrics.bat`は、選択workspaceの`runs/`直下にある全Runの
`metrics.jsonl`を`metrics.jsonl.gz`へ移行する。引数なしで起動すると`runs/`を持つworkspaceを
名前順に列挙するので、番号を入力して選択する。`[0] EXIT`を選ぶとlauncherを終了する。
workspace名または絶対pathを第1引数で指定してもよい。

preflight後の`Execute compression? [YES/NO/DRY-RUN]:`には、次のいずれかを入力する。

- `YES`: gzipを作成・検証し、確定後に元の`metrics.jsonl`を削除する。
- `NO`: fileを変更せず、そのworkspaceの処理を取り消す。
- `DRY-RUN`: 対象、skip理由、必要容量だけを表示し、fileを変更しない。

非空fileの末尾が改行で完結していないRunや、Runnerが書き込み中のRunは変更せずskipする。
処理後もMetrics Viewer、TensorBoard bridge、MLflow bridge、Optuna集計はrawを優先し、
rawが無いRunではgzipを透過的に読む。
workspaceを引数指定せずに起動した場合は、`YES`、`NO`、`DRY-RUN`の結果表示後にpauseし、
キー入力後にworkspace選択へ戻る。`--no-pause`指定時はpauseを省略する。
複数workspaceを続けて処理し、最後に`[0] EXIT`で終了できる。workspaceを引数指定した場合は1回だけ処理する。

## 3. 画面の基本操作

![複数Runを選択したMetrics Viewer](assets/030_metrics_viewer_run_comparison.png)

| UI | 動作 |
|---|---|
| `Workspace` | serverが列挙したworkspaceへ切り替え、Run選択・色・viewport・凡例状態をリセットする |
| `Runs` | 行クリックで即時toggleする。同じ行を350ms以内にもう一度押すとそのRunだけを選ぶ。空選択も可能 |
| `Select All` / `Latest Only` | 全Run、または最新Runへ選択を切り替える |
| `Tags` | 表示するmetric tagを選ぶ |
| Tagsの`Filter` | 選択済みtagだけを一覧へ残す |
| Run行の背景と`%` | 選択したMetricsマスタをSQLiteへ取り込んだRun単位の進捗を示す |
| `Reload` | Run一覧、tag、現在viewportのmetric rangeを再取得して再描画する |
| `Auto Reload` | 30秒ごとにmetadataを更新し、最新を表示中のgraphだけを更新する |
| `LOD: MinMax / Mean / Band` | 右上のScroll Lock直前にある全graph共通のLOD表示mode。変更時に再取得しない |
| グラフの`Log` | 正負と0を扱えるsigned-log表示を切り替える |
| グラフの`p5–p95` | 各Runについて、現在表示している有限値のp5–p95内だけを表示する |
| `Scroll Lock` | グラフ操作を抑え、drag/swipeを縦scrollへ使う |
| Screenshotボタン | side panelを隠し、比較画像向けの表示へ切り替える |

Plotlyのmodebarではzoom、pan、画像保存、`Reset axes`を利用できる。`Autoscale`ボタンは重複を避けるため非表示である。グラフ本体のdouble-clickはPlotlyのaxis resetを維持しつつ、ViewerのReloadも実行する。

初回表示だけ最新Runを自動選択する。以後は手動の空選択と、Run消失で空になった選択を維持する。
Reloadでは既知のOFF tagを保ち、新たに発見された可視tagだけを自動的にONへ加える。
選択workspace、選択tag、LOD mode、Scroll Lock、tag別のLogとp5–p95はbrowserの`localStorage`へ保持される。Logとp5–p95はworkspaceをまたいで同名tagへ適用される。

p5–p95は、凡例で非表示にしたRunを除き、各Runについて現在のX範囲にあるLOD描画値から個別に計算する。p5未満とp95超の点は表示traceから除外されるため、その点のhoverも表示されない。残った点へPlotlyのautorangeを適用する。点数による下限はなく、tooltipには表示点数と入力点数を示す。Logとの併用時もpercentileはraw値で計算してからsigned-log座標へ変換する。手動Y zoomは維持され、`Reset View`またはPlotlyのaxis resetでフィルター後のautorangeへ戻る。グラフheaderの統計値とclient cacheのraw値は変更しない。

LODの`MinMax`は各bucketのmin、max、lastを元データの実step順に結ぶ。
`Mean`はbucket平均、`Band`はmin/max帯へ平均線を重ねる。点数が少なくL0を表示できる場合は、
どのmodeでもraw折れ線になる。グラフheaderの`Min / Max / Avg / Std`はviewportではなく、
選択Runのcommit済みraw全点に対する`TagStats`を合成した値である。

取り込み中でもRun一覧とcommit済みgraphは操作できる。Run-level errorまたはstep逆行で
隔離されたtagは`⚠`とtooltipで示し、commit済み部分は引き続き表示する。

## 4. Run比較の手順

### 4.1 比較条件を確認する

グラフを重ねる前に、各Runの次を確認する。

- `config/config_data.txt`: includeとCLI override解決後の全設定
- `config/*.txt`: コンポーネント別の注入済み設定。Envは`config/env.<Env name>.txt`
- seed、Agent/Env、Network構成、batch size、replay条件
- CPU/GPU、device index、決定論設定
- configured evalのinterval、RunMode、clone設定
- Runの停止理由と到達step

名前が似ていても設定が同じとは限らない。Run名ではなくRun artifactを正とする。

### 4.2 同じtagと同じstep軸を選ぶ

`metrics.jsonl`のscalar recordは`type`、`tag`、`step`、`value`を持つが、`step`が`train_step`、`learn_step`、`episode_step`、`exp_step`などのどれかはrecord自体に保存しない。軸はRun内のmetrics設定で決まる。

比較時は同じtag名だけでなく、`metrics.scalar.[tag]`の定義が同じstep軸を選んでいることを確認する。現行設定では、`@learn`、`@episode_end`、`@session_end`は明示がなければ`exp_step`、`@train`は`train_step`が既定である。`$exp_step`などの明示指定があればそれを優先する。

### 4.3 matched `exp_step`範囲で比較する

到達stepが異なるRunの最終点同士をそのまま比較しない。両方にデータがある共通の`exp_step`範囲を決め、同じwindowで評価する。

例:

```text
Run A: 0 - 100M exp_step
Run B: 0 - 70M exp_step

比較window: 50M - 70M exp_step
```

同じwindowで次を分けて読む。

- 水準: window内のmean/medianや評価EMA
- 安定性: 振れ幅、急落、seed間のrange/std
- 傾向: window前半と後半の差、傾き
- 速度: 同一hardware・同一並列条件での`90_perf/90_elapse_hour`差分。`exp_step_per_sec`は補助として読む

1つの長いRunだけから、別設定より高速・高性能と断定しない。停止点が違う場合は、まずmatched windowへ揃える。

### 4.4 表示条件による誤読を避ける

- EMAの`ema_alpha`が異なる曲線は滑らかさが違うため、そのまま分散比較しない。
- `interval`やeval頻度が違うと点密度が変わる。線の滑らかさを性能差と解釈しない。
- signed-logは0付近を広げる表示変換である。線間の見た目の距離をlinear scaleと同じ比率として読まない。
- configured evalとEvalPanelは別Runnerである。tagのrunner scopeを確認する。
- `exp_step_per_sec`は他process、動画出力、profiling、Optunaの並列jobに影響される。
- `90_perf/12_exp_step_per_sec`はτ=10秒の時間重みEMAである。窓の長さで重み付けするためevalなどのstallも計上されるが、瞬間値ではないため区間の切り出しには使わない。真のthroughputは`90_perf/90_elapse_hour`の差分と`exp_step`の差分から算出する。`90_perf/22_exp_step_per_sec_ema`はそれをさらに`ema_alpha`で平滑した長期線である。
- 時間重みEMA化より前のRunでは同tagがstallを過小評価している（実測で真値478 steps/sに対し1,830 steps/sを表示した例がある）。過去Runと同tagの数値を直接比較しない。
- 1 seedの差はseedぶれを含む。候補選定後は複数seedで再評価する。

### 4.5 IQN探索P0診断を読む

IQN探索では、解決済み`config/config_data.txt`に`metrics.scalar.iqn_search_p0`が合成されていることと、Policy/Learnerそれぞれのtau配置方式・本数`K/N/M`を先に確認する。診断値だけで採用を決めず、DropMergeのDouble Suika生成数・達成率、報酬、PER健全性、throughputを同じmatched `exp_step` windowで分けて読む。

- `iqn_policy_margin_mc_ratio`はUQE上位2行動のgapをrisk quantileの有限本数scaleで正規化する。`random`ではMonte Carlo平均の安定度、`fixed`および`stratified`ではforward間の乱数分散ではなく積分解像度のproxyとして扱う。
- `iqn_current_mc_scale`と`iqn_target_mc_scale`は`N`と`M`を分けて読む。`iqn_priority_mc_ratio`は現行の平均TD priority信号が両側の有限tau scaleに対してどの程度大きいかを表す。
- `iqn_first_pair_abs_td`と`iqn_first_cancellation_ratio`は初回Learner priority更新行だけを対象とする。`per_sample_initial_count=0`の区間では`iqn_first_*`が`NaN`になるため、0への改善・悪化とは解釈しない。
- TBO有効時のLearner診断は実空間ではなく、現行priorityと同じh空間の値である。TBO有効/無効Runの絶対値を直接比較しない。
- `iqn_uqe_full_q_argmax_disagreement`と`action_full_q_margin.[i]`はfull-distribution queryがあるPolicyだけで成立する。欠落時の`NaN`を一致やmargin 0と解釈しない。
- P0 group OFF/ONの負荷比較は同一binary・seed・実行条件で直列に行い、matched windowの`90_perf/90_elapse_hour`差分から算出したthroughputを比較する。他processやparallel Optuna jobがある測定は採用しない。

### 4.6 分位tail探索診断を読む

分位tail診断はPolicyやpriorityを変更する信号ではなく、既存QR / IQNのreturn distributionを観測する6 scalarである。まず解決済み`config/config_data.txt`でPolicy 5本が`eval2`、Learner 1本が`@learn`へ登録されていることを確認し、Policy側はfixed full distributionの本数`K`、Learner側はPERとTBOの有効状態を併記する。

- `policy_upper_truncated_std`と`policy_lower_truncated_std`は最終実行actionについてmedianから上下へ広がる幅であり、Q値と同じ単位で読む。差からtail asymmetryは見られるが、単一networkの幅をparametric uncertaintyや探索bonusの有効性と断定しない。
- `lower_risk_full_q_argmax_disagreement`は、full Qのargmaxが係数1の仮想的なlower-tail penaltyで変わる割合である。既存`iqn_uqe_full_q_argmax_disagreement`とは目的が異なり、risk回避Policyが実際に有効という結果ではない。
- `quantile_crossing_ratio`はtau順の隣接quantileが降下した割合である。高い区間ではupper / lower tailを分位関数の領域として解釈する信頼度が低いため、tail幅の大小より先にorderingを確認する。同値はcrossingに含めない。
- `policy_selected_crossing_depth_p90_ratio`は、最終実行action内のpositive crossing深度を分布rangeで正規化し、action event内でlane別nearest-rank p90を求めてbatch平均した無次元量である。全actionの発生頻度を測る`quantile_crossing_ratio`と組み合わせ、頻度が横ばいでp90が下がる場合は浅い局所逆転へ寄った可能性、頻度が下がってp90が上がる場合は少数の深い逆転が残る可能性として読む。Run全期間のcrossing sampleをpoolしたp90ではない。
- `upper_tail_priority_spearman`は、PERで既に偏ってsamplingされたminibatch内に限った、upper-tail幅とclip後raw priorityの順位相関である。高い正相関は両者が似た経験を強調している可能性を示すがReplayBuffer全体の冗長性を証明せず、低相関や負相関も新しい信号の有用性を証明しない。
- PER無効、batch不足、定数順位列、Policy full distribution欠損、`K < 2`では該当値が`NaN`になる。0との一致や相関0へ読み替えない。ただしcrossing深度p90は、入力が成立していてpositive crossingがない場合、またはrangeが0の場合を正常値`0`とする。TBO有効時は実空間ではなく現行Policy score / priorityと同じh空間なので、TBO有効/無効Runの絶対値を直接比較しない。

### 4.7 可塑性メトリクスを読む

`34_agent_plasticity`群は、NNの表現がどれだけ健康かを直接測る。測るのは3種類ある。**活性の分布**（どれだけのユニットが発火しているか＝`dormant_ratio` / `dead_ratio`、特徴ベクトルの大きさ＝`feature_norm`）、**方向の分布**（特徴が実効的に使っている方向数＝`srank` / `srank_ratio`）、そして**パラメータの大きさ**（`weight_norm_feature` / `weight_norm_readout`）である。用語の定義は`CONTEXT.md`の可塑性・表現統計節を参照する。

番号はdecadeがチャネル、下1桁が統計種を表す。

| decade | チャネル | 既定 |
|---|---|---|
| `0x` | actual（学習forwardが生成した特徴。測定バッチはPERが選んだ現行updateバッチ） | ON |
| `2x` | target（同じupdateのTD計算でtarget networkが生成した特徴） | OFF |
| `4x` | probe（ReplayBuffer全域から一様・非復元でsampleし、NoGrad・eval modeで部分forward） | ON |
| `6x` | weight norm / spectral sigma（パラメータ側。データに依存しないのでチャネルを持たない） | 61/62のみON |

下1桁は3チャネル共通で、`x1` dormant / `x2` dead / `x3` feature_norm / `x4` srank / `x5` srank_ratio / `x6`-`x9` はδ違いのsrank（既定OFF）。`02` / `22` / `42`のように並べれば同じ統計をチャネル間で比べられる。チャネルは購読行の有無で独立に有効化されるので、まず解決済み`config/config_data.txt`で`learner.plasticity.feature_key`と購読行を確認する。全行`$learn_step`軸である。

**probe系を基準に読む。** 一様サンプルなのでPERの偏りが入らず、`probe.batch_size`が特徴次元より大きければsrankの天井にも当たらない。Run間・`replay_ratio`間の比較はprobe系で行う。actual系との差はPERが見せている分布の偏りそのもので、actual系のdormantがprobe系より高ければ、PERが「表現が苦しんでいる状態」へ学習を集中させている兆候になる。

**`42_probe_dead_ratio`の谷が転換点の目印になる。** 学習初期に下がりきってから上昇へ転じるので、その転換が性能のピークと同じ窓に来る。`61_weight_norm_feature`も同じ窓で下げ止まって増加へ転じるため、2本を並べて一致を確認する。

**`feature_norm`は単体では読めない。** HeadはQ = w・φの形なので、`q_max`が横ばいで`43_probe_feature_norm`だけ伸びる局面は「wが縮んだ」のか「Qに寄与しない方向へφが伸びた」のか区別できない。`61` / `62`と対で見て初めて帰属が閉じる。`62`低下ならreadout縮小とのスケール移送、`62`平坦で`61`上昇ならbackbone側のscale成長、両方上昇ならNetwork全体のscale成長である。

SNを有効にしたRunでは、`61` / `62`はoptimizerが保持する生parameterのL2、`63_weight_norm_feature_effective` / `64_weight_norm_readout_effective`はforwardで実際に使うweightへSNを適用したL2である。bias、normalization affine、非SN parameterは両者で同じ寄与を保つため、差はSN weightだけに由来する。`65_spectral_sigma_feature` / `66_spectral_sigma_readout`は各群で最大のsigmaであり、該当SN layerがない群は`NaN`になる。63–66は既定OFFなので、実験時に購読行を有効化する。

`spectral`ではsigmaに応じて常に正規化されるため、61/62の成長を63/64の成長と同一視しない。`spectral_cap`ではsigmaが1以下のweightは変化せず、65/66が1を越えた群だけcapが効く。sigmaはlayer別系列ではなく群内最大値なので、異常layerの同定用途ではなく、cap発動とscale傾向の診断に使う。

比較するときの制約が3つある。

- **`srank_ratio`はチャネル間で絶対値を比較しない。** ratio = srank / min(N, D)で、Nがactual系はlearnerのbatch size、probe系は`probe.batch_size`と異なる。比べるのは時間方向の形（低下開始点・低下率・回復）だけである。
- **`dead_ratio`は`probe.batch_size`が違うRun間で水準を比較しない。** サンプル数が少ないほど「たまにしか発火しないユニット」が死んで見えるため、値そのものがバッチサイズに依存する。同一設定内の時間変化として読む。
- **weight normはparameter数に依存する。** 同一構成の時系列か、同一構成Run間だけを比べる。

その他の読み方。

- `dead_ratio`（τ=0）は`dormant_ratio`（τ=0.025）の部分集合で、通常は追随する。乖離した時だけ独立情報になる（浅い休眠ではなく不可逆な死が増えている）。振幅はdeadの方が大きく出る。
- srankは方向の分布を、dormant / deadは脱落したユニット数を測るので、ほぼ直交する。ユニットの死が主体の損傷では、srankは`min(N, D)`の何割かで平坦なまま動かないことがある。srankが動かない＝健康、ではない。
- δ違い（`x6`-`x9`）は同じ特異値ベクトルから求めるのでSVD回数を増やさない。上位方向へのエネルギー集中を見たい時だけ有効化する。
- target系はonlineの`soft_update_tau`遅れの観測で、平時はほぼ冗長なので既定OFFである。崩壊機序を精査する時に有効化し、online→targetの伝播ラグから「まだ健康なtargetが引き戻す」構図か「両方巻き込まれた自走崩壊」かを識別する。
- exp軸tagとの突き合わせは exp_step = learn_step × batch size / `replay_ratio` で換算する（batch 256でRR8=×32 / RR4=×64 / RR1=×256）。step軸選択の一般注意は4.2節と6.3節、指標定義と測定契約は[DQN系Agent](200_dqn_agents.jp.md)9.4章と[062_plasticity_metrics_10prd.md](../memo/done/062_plasticity_metrics_10prd.md)を参照する。


### 4.8 Munchausen診断を読む

`metrics.scalar.@munchausen`は`36_agent_munchausen`へ7 tagを追加する。最初に解決済み設定のenabled、mode、Double DQN OFFと、初期化ログのscore源を確認する。`target_policy=UQE`では経験分位によるrisk scoreを使い、平均Qを使う構成とは区別して読む。

| tag | 読み方 |
|---|---|
| `01_scaled_logp_mean` / `02_scaled_logp_mean_ema` | 実行actionのclip前scaled log-policyの平均とEMA。0以下 |
| `03_clip_ratio` | bonus下限clipの発生率。0〜1 |
| `04_bonus_mean` / `05_bonus_mean_ema` | targetへ1回加えるbonusの平均とEMA。`alpha * clip_value_min`〜0 |
| `06_next_entropy` | next方策entropy。0〜`ln(action数)` |
| `07_soft_gap` | soft state valueと最大平均Qの差。平均scoreなら0〜`entropy_tau * ln(action数)`、risk scoreなら負も許す |

5つのraw診断はTBO時もFP32実空間で計算し、PER OFFでも回収する。機能OFFまたは未成立の既知keyは`NaN`であり、0へ読み替えない。readbackはpriority・clip件数、IQN診断、Munchausen診断、upper-tail統計の順に一括転送する。Actorの`actor_approx`は既存action scoreによる近似なので、Learnerの経験分位近似とは別の近似として扱う。

mode間の負荷は`forward_target`、`forward_munchausen_online`、`munchausen_target`と、同じexp step区間のelapsed time差で比較する。診断や1 seedの成績だけで改善を断定しない。

## 5. Optuna結果を分析する

### 5.1 Metrics ViewerとDashboardの役割

| 対象 | 使用する画面 | 正本 |
|---|---|---|
| seed runの時系列 | Metrics Viewer | `<study>_<trial>_s<seed>/metrics.jsonl` |
| 1 trialのmulti-seed結果 | Optuna Dashboard / artifact | 代表folderの`multiseed_summary.json`、`seed_runs.json` |
| 同一params groupの再評価 | summary study | `group_summary.json`とmean/range/std objective |
| runner/config失敗調査 | Run artifact | `manifest.json`、`process.json`、`stdout.log`、`stderr.log` |

代表folder`<study_name>_<trial_name>`は`metrics.jsonl`を持たないためMetrics Viewerには出ない。`_s<seed>`付きのseed runだけを時系列比較する。summary studyはDashboard閲覧用であり、Metrics ViewerのRunではない。

Optuna Dashboardはリポジトリルートから次で起動する。

```powershell
apps\23_optuna_dashboard.bat dm_opt
```

引数にはworkspace名または絶対pathを指定する。URLは`http://127.0.0.1:8088`、storageは`<workspace>/optuna/optuna.db`、artifact storeは`<workspace>/optuna/artifacts`である。launcherはworkspace、DB、artifact storeを生成せず、いずれかが無い場合はfail-fastする。

### 5.2 scoreを読む

DropMergeのtrial valueはseed別scoreのaggregateである。`score_aggregate`を確認し、`mean`、`median`、`mean-minus-std`、`min`を混同しない。

現行のprimary scoreは指定window内の次の2 tagのmeanを平均する。

```text
21_eval/03_target_reward_ema
21_eval/04_policy_reward_ema
```

late windowの`score_60_80`、`score_80_100`、`late_slope`は伸びや頭打ちを見る補助指標であり、trial valueそのものではない。最終候補はaggregate scoreだけでなく、seed別score、range/std、matched `exp_step`時系列を併せて確認する。

`run-study --n-jobs > 1`は同一GPU上でrunnerを並列起動できるが、durationとstep/secは相互干渉を受ける。その条件のthroughputを単独Runと直接比較しない。

## 6. `inspect_run.py`でRunを検査・抽出する

Metrics Viewerは人間向けの可視化画面である。shellから構造化結果を取り出したい場合、特にAIエージェントへRun分析を依頼する場合は`inspect_run.py`を使う。Run artifactを一切変更しないread-onlyのCLIで、実行中のRunへ当てても安全である。

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py <subcommand> [RUN ...] [options]
```

| subcommand | 役割 |
|---|---|
| `runs` | Run発見。artifact、Metricsマスタ、Metricsキャッシュの状態 |
| `tags` | metric tagの一覧。定義（step座標系、source key）と到達step |
| `config` | 実効設定の抽出とRun間差分 |
| `metrics` | scalarの抽出、range集約、Run間比較 |

全subcommandに`--format json|md`と`--output PATH`がある。既定はJSONで、`--output`は一時file経由でatomicに置換する。

### 6.1 Runを見つける

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py runs --workspace dm-iqn
```

引数なしの`runs`は全workspaceのRunを列挙する。`RUN`を渡すとそのRunだけを詳しく返す。`RUN`はRun名、または既存の相対・絶対directory pathを取る。Run名の探索範囲は`apps/runner/workspaces/*/runs/`直下だけである。同名Runが複数workspaceにある場合は候補pathを示して終了値2で止まり、どのworkspaceも暗黙選択しない。`apps/runner/runs_*`のlegacy配置はdirectory pathで明示すれば読める。

`runs`はMetricsマスタを開かない。artifactのpath・size・更新時刻、`config/config_data.txt`のSHA-256、Metricsマスタの選択結果、Metricsキャッシュの状態と理由、`*.log`と`agent_close.anet`の一覧を返す。`agent_close.anet`の有無と更新時刻は、Runが完了したか途中で止まったかの手がかりになる。

### 6.2 何が取れるかを見る

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py tags run_A --format md
```

`tags`はtagごとに、解決済み定義（`step_axis`、`runner`、`scope`、`eval_name`、`eval_episodes`、`num_envs`、`event`、`target`、`source_key`、`ema_alpha`、`interval`、`clip`）と観測範囲（`count`、`min_step`、`max_step`）をJSON / Markdownで返す。`--metric`へ何を渡せるかと、各tagがどこまで到達しているかがこれで分かるので、range指定はこの結果を見てから決める。

`runner` は step counter の所有者、`scope` / `eval_name` は購読先を表す。eval1 / eval2 のセッション集約がともに `runner: "train"` でも、`eval_name` で区別できる。`eval_episodes` は1セッションの採用予定数、`num_envs` は構築済み eval Env の lane 数であり、セッション完了や並列エピソード数を保証しない。train scope の eval 情報と未指定の `clip` は `null`。

master / cache の両経路で定義の追加情報を保持する。過去 Run の定義に無い追加項目は不明（`null`）とし、tag や config から補わない。定義不在時の既存 config 導出経路（cache 未構築の `tags --no-observed` を含む）では購読先と clip を復元するが、構築後の実条件を推測しないため `eval_episodes` / `num_envs` は `null` とする。過去 artifact は書き換えない。

`source_key`は`metrics.scalar.[tag]`の右辺で採用されたmetric keyである。値の意味を確認したいときは、この文字列でコードを検索する。

`--no-observed`は観測範囲を取らず、宣言された定義だけを返す。Metricsキャッシュが使えないRunでもMetricsマスタを開かないため常に即座に返る。

### 6.3 step座標系に注意する

`tags`の`runner`列は、そのtagのstepがどのRunnerのカウンタに載っているかを示す。**同じ`exp_step`でも`runner`が違えば別の座標系である。**

```text
51_eval1/13_double_suika_created_mean   runner=train   max_step 19,993,856
51_eval1/41_noop_uqe_win_rate           runner=eval1   max_step    151,185
```

どちらも`config/config_data.txt`では`$eval.[eval1] ... $exp_step`と書かれているが、`@session_end`はtrain runnerのstep、`@train $action_info`はeval runner自身のstepに載る。train側のstep範囲をeval側のtagへ当てると、エラーにならず結果が空になる。比率は学習が進むと変わるため換算もできない。

`inspect_run.py`は相対的なrange（百分率、末尾相対、`common`）をこの座標系ごとに独立して解決するので、両方のtagを同じ呼び出しで指定してもそれぞれ正しい範囲になる。

### 6.4 metricを抽出する

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py metrics run_A run_B --metric "42_env/*" --range -4M:
```

`--metric`は完全一致のtagかglobを取る。glob meta characterは`*`と`?`だけで、`[`と`]`はリテラルとして扱うため、`51_eval1/41_noop_uqe_win_rate`のような`[tag]`記法を含むキーもそのまま書ける。

**metricは1回の呼び出しへ束ねる。** Metricsマスタの走査はRunごとに1 passで、tag数には比例しない。tagごとに呼び分けるとpassの回数だけ時間が増える。

`--range`は両端を明示する指定である。両端inclusiveなので、重なるrangeでは境界の点が両方に数えられる。

| 形 | 意味 |
|---|---|
| `10M:20M` | 絶対step。`K`/`M`/`G` suffixを取る |
| `10%:20%` | その座標系の最大観測stepに対する百分率 |
| `:20M` | 下端を省略（0から） |
| `10M:` | 上端を省略（最大観測stepまで） |
| `-4M:` | 末尾4M step |
| `-10%:` | 末尾10パーセント（`90%:100%`と同じ） |

`--range-mode common`は、同じ座標系を持つ全Runの観測範囲の交差を使う。到達stepが違うRunを同じ土俵で比べるときの標準手段で、[4.3](#43-matchedexp_step範囲で比較する)のmatched windowを自動化したものである。

```text
run_A: 0 - 20,000,000 exp_step
run_B: 0 - 16,200,000 exp_step

--range-mode common  ->  両方とも 0:16,200,000
--range -4M:         ->  A は 16.0M:20.0M、B は 12.2M:16.2M（幅は同じ4M）
```

`common`は「同じ場所を見る」、`-4M:`は「同じ幅で今を見る」。目的が違うので使い分ける。

各Run×tag×rangeについて`count`、`mean`、`population_std`、`min`/`max`、`first`/`last`、step範囲が返る。range適用前のtag全体の観測範囲も併記されるので、rangeが座標系とずれていればその場で気づける。

### 6.5 Run間で比べる

`metrics`はrangeごとに「行=tag、列=Run」の比較表を返す。cellの統計は既定で`mean`、`--stat last`などで切り替える。

- Runが2つなら`delta`と`delta_ratio`が付く。
- Runが3つ以上なら`mean`、`population_std`、`range`が付く。同一設定の反復Runをまとめて渡せば、ばらつき幅の物差しがその場で得られる。

Markdownでは比較表と詳細表の両方が出る。曲線の形を見たいときだけ`--series`を付けると、最大128点へ間引いた`step:value`列が加わる。間引きは決定的で、Metricsキャッシュ経路とMetricsマスタ経路で同じ結果になる。

### 6.6 実効設定を確認する

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py config run_A run_B --diff
```

`config`は`config/config_data.txt`を読む。ただしこのfileには実効値だけでなく、マージ元の定義namespace（`A.*`、`R.*`、`AS.*`、`M.*`、`metrics.scalar.baseline.*`など）と、選ばれなかったprofile（`metrics.scalar.full.*`など）が同居している。`.$`の選択行はAutoMergeで消えるため、**どのprofileが選ばれたかをこのfile単独から復元することはできない。**

そこで各keyには`effective`が付く。`config/<module>.txt`に同じkeyがあれば`true`、無ければ`null`（不明）である。module dumpを出していない領域（`net.*`など）があるため、確認できないものを`false`とは言わない。`--effective-only`は`true`のkeyだけへ絞る。

```text
--config-key "*replay_batch_size*"                  9件（定義namespaceを含む）
--config-key "*replay_batch_size*" --effective-only  1件（DefaultDQNAgent.learner.replay_batch_size）
```

`--diff`は値または存在有無がRun間で異なるkeyだけを返す。欠損は`present: false`で表し、値`null`と区別する。設定差がseedだけであることの確認や、反復Runが本当に同一設定であることの確認に使う。

### 6.7 設定の解決経路を確認する

```powershell
.\.venv\Scripts\python.exe viewers\metrics-tools\inspect_run.py resolution run_A run_B --format md
```

`resolution`は、selectionの適用順と`${}`値参照をRunごとに表示する。先頭selectionが`run.$`ならnamed幹として先に要約し、その後に各selectionの`key` / `term` / `resolved`と、各referenceの`source` / `target` / `value`を表示する。幹を使っていないRunでは幹要約を出さない。

読み込みは`json/config_resolution.json`の`type=json` / `tag=config_resolution` envelopeを優先し、無ければ過渡期Runの`config/config_resolution.json`素payloadを読む。両方無いPH0以前のRunは`status: missing`として正常終了する。未知のresolution `schema_version`はwarning付きでbest-effort表示し、優先sourceが壊れている場合は下位sourceへ黙ってfallbackせず`status: source_error`と終了値1を返す。

このsubcommandはmirrorだけを読み、`metrics.jsonl` / `metrics.jsonl.gz` / `metrics_cache.db`を開かない。複数Runを指定した場合も、一つのRunのresolution欠損や破損から他Runの表示を独立させる。

### 6.8 Metricsキャッシュとの関係

`metrics_cache.db`は完全にcurrentなときだけread-onlyで使い、それ以外はMetricsマスタへ自動fallbackする。判定結果と理由（`current` / `absent` / `invalid` / `partial` / `stale` / `error`）は`runs`と`metrics`の結果に載る。toolはキャッシュの作成・更新・修復・削除を一切行わない。

`metrics.jsonl`をgzip化した後にMetrics Viewerでそのworkspaceを開いていないRunでは、キャッシュが`stale: source_kind_changed`または`partial`のままになる。動作は正しいがMetricsマスタの全行走査が走るため、Viewerで一度開いておくと高速経路に乗る。

Runner実行中のRunも読める。rawは実行開始時のサイズまでを読み、未終端の末尾行を取り込まず、読み取り中にマスタが変化した場合は`provisional`と`source_changed_during_read`を立てる。

### 6.9 終了値

| 値 | 意味 |
|---|---|
| `0` | 正常。一部Runだけのtag/key欠損は`missing`として結果に載る |
| `1` | source読み取りやquery失敗、または指定したmetric/config selectorが全Runで1件も成立しない |
| `2` | 引数やrangeの構文エラー、Run未発見・曖昧性、`--output`の親directory不在 |

## 7. 最小分析チェックリスト

1. 比較対象の`config/config_data.txt`を保存した。
2. 同じtag、同じstep軸、同じeval定義を確認した。
3. 共通の`exp_step`windowを決めた。
4. scoreとthroughputを分けて読んだ。
5. EMA、interval、signed-logなど表示条件を揃えた。
6. 1 seedの結果だけで最終判断していない。
7. Optunaではaggregate方式、seed別分布、trial stateを確認した。

## 8. 関連文書

- [Run実行ユーザーガイド](020_user_guide_run.jp.md)
- [開発環境](040_development_environment.jp.md)
- [可観測性](140_observability.jp.md)
- [アプリケーションとツール](160_applications_and_tools.jp.md)
- [DropMerge Optuna利用ガイド](../optuna.md)
