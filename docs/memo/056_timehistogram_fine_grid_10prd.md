# TimeHistogram fine-bin保持・描画時集約 PRD

## 問題

TimeHistgram型の画像メトリクス(例: `52_per_thg_prio`、probe=`replaybuffer.per.distribution`)を有効にすると、学習スループットが1/3〜1/4に落ちる。

実測(Tracy、run_20260720-141314、DropMerge、`frame_interval=10`のdump): frame 22.37s中、`TimeHistogram::RebuildFromRaw`(`core/anet-core/src/heat_map.cpp:711`)が16.26s(72.71%)。7回発火・平均2.32s/回。この間のlearnは約80回=NextFrame約8回であり、**ほぼ毎フレームrebuildが発火**している。呼び出しは`Notifier::Notify`経由の`TimeHistogramObserver::OnLearn`(`core/anet-core/src/observers.cpp:130`)で**TrainThread同期**のため、この時間はまるごと学習停止になる。

原因は3層の重なりである。

| # | 層 | 現状 | コスト特性 |
|---|---|---|---|
| 1 | 取得・蓄積 | 毎learnで`AgentTensorVectorProbe::GetVector`→`PER_DIST`=RB valid**全件**のgather+正規化除算+vector化(RBはCPU常駐のため全てCPU処理。D2Hは発生しない)→`AddBatch`が生値を`cur_raw_`へ全蓄積 | O(N)/learn(N=RB valid件数)。PER_DISTはRB全体のスナップショット分布であり、frame_interval回重ねても情報増ゼロでコストだけN倍 |
| 2 | 保持 | `NextFrame`が`frames_raw_`へpushするのみで**上限なし**(`histgram.max_frames`は表示側`thm_`の幅でしかない)。さらに`thm_`(TimeHeatMap)自体も点列`buf_`(max_points=0)で1フレームあたりbins点ずつ**無限成長** | メモリ・後述rebuildコストが経過時間に線形。実質リーク |
| 3 | 再構築 | `AddBatch`でmin/maxが拡大すると`need_rebuild_`→`RebuildFromRaw`が**全生値**を`MapToBin_`(LogScaleAxisはlog1p×2/値)し直し+`thm_`全再投入 | O(累計全生値)。PERのmax priorityは新規push・優先度更新で頻繁に伸びるため、ほぼ毎フレーム発火 |

回避ノブも存在しない: `alpha`は1.0固定(configパースは`core/anet-core/include/anet/image.hpp`でコメントアウト。`metrics_image.txt`の`histgram.alpha`行はdead)、`base_min`/`base_max`はNaNハードコード(`core/anet-core/src/image.cpp`のobserver_config構築)で固定レンジを渡せない。

`interval`(=log_interval)は画像出力の間引きにすぎず、上記1〜3はログ出力と無関係に毎learn/毎frame_intervalで走る。「出力頻度を下げても重い」のはこのためである。

## 目的

1. TimeHistogramの計算・メモリコストを、入力ベクタ長Nと経過時間に対して**上限付き**にする。レンジ拡大時の処理からN依存を排除し、保持メモリを定数上限にする。
2. スナップショット系probe(PER_DIST等、毎learnで同じ母集団を返すもの)の取得をframe境界に間引く手段(`sample_mode`)を設ける。
3. probeの返却要素数に上限(`max_samples`)を設け、RB容量を将来増やしてもコスト上限を保証する。
4. 表示の意味論(軸レンジ自動追従、対数軸、フレーム単位正規化、Scale表示)と、ImageSource/Observerの公開APIを維持する。

## 前提

- 実測値・行番号はrun_20260720-141314時点のworking treeに基づく。
- ObserverがTrainThread同期で呼ばれる構造(`Notifier::Notify`)は変えない。オフスレッド化はMetricsLoggerライフサイクル(PRD 026)系のスコープであり本PRDでは扱わない。
- `ImageSource`インターフェイス(`RenderRaw`/`Reset`/`GetImageSubType`)、`TimeHistogramObserver`の公開API、EvalPanelの`GetImageData`経路は互換維持。
- `HeatMap`/`TimeHeatMap`の外部仕様は変更しない(既存の`SetGridValues`高速パスを再利用するのみ)。
- config追加はすべて既定値=現行挙動互換とする。既存run設定は無変更で従来どおり動く。
- TimeHistgramの使用実績は`frame_mode = Scale`のみ(`metrics_image.txt`)。Scroll/Overwriteは意味論を定義し直すが、視覚回帰の検証対象はScaleとする。

## 用語

### 軸空間 t

値vを縦軸binへ写像する前の中間座標。`HM_LogScaleAxis`有効時は`t = sign(v)·log1p(|v|)`、無効時は`t = v`。現行`MapToBinLogAxis_`のlog1p写像と数学的に同型だが、min/max正規化を含まない**レンジ非依存**の座標である点が異なる。

### fineグリッド

フレームごとに保持する固定境界の高分解能ヒストグラム。`fine_bins = bins × 8`(内部定数)個のfloatカウント。bin境界は軸空間t上の等間隔格子(bin幅`w`、格子位相は0起点)で定義する。

### coarse(表示)グリッド

描画時にfineグリッドから集約する`bins`個のヒストグラム。表示レンジ(`min_val_`/`max_val_`、現行のEMA拡大追従を維持)で写像する。

### 縮約

fineグリッドのレンジ拡大操作。bin幅wを2倍にし、隣接2binを1binへ加算統合して空いた半分でレンジを拡張する。格子位相が0起点で入れ子になるため、**再配分誤差ゼロ**で行える。

### snapshot / accumulate

Observerのサンプリングモード。accumulate=毎learnでprobe取得しフレームへ蓄積(現行挙動)。snapshot=frame境界の1回だけ取得(フレーム=その時点の分布)。

## 解決方針

### 1. 保持を生値からfineグリッドへ(TimeHistogram内部)

- `frames_raw_`(全フレーム生値)と`cur_raw_`を撤去し、`cur_fine_`(fine_bins個のカウント)と`frames_fine_`(保持フレーム列、後述の上限管理つき)に置き換える。
- `AddBatch`は各値をt空間へ写像しfine binへ直接加算する。O(N)は変わらないがlog1pは投入時の1回だけになり、生値の保持・再走査が消える。
- 値がfineレンジ外に出たら、包含するまで縮約(w倍化)を繰り返す。縮約はO(保持フレーム数×fine_bins)で、レンジが倍々に伸びるため生涯発生回数はO(log(最終レンジ/初期レンジ))に抑まる。
- 初期fineレンジは最初のAddBatchの[t_min, t_max]から決める(ゼロ幅は最小幅へ丸め)。
- 表示レンジ`min_val_`/`max_val_`の追従ロジック(拡大方向のみ、base_min/base_max優先)は**現行のまま維持**する。ただしその役割は「rebuildトリガ」から「描画時の写像パラメータ」に変わり、`need_rebuild_`/`RebuildFromRaw`/`AppendCurrentFrameOnly`は削除する。「レンジ変更→再構築」という概念自体を消す。

### 2. フレーム管理(max_framesを保持上限にする)

`frames_fine_`の保持数を`max_frames`以下に管理する。

- **Scale**: 保持フレーム1個=`stride_`個の論理フレーム(NextFrame単位)の算術平均。`stride_`初期値1。保持数がmax_framesに達したら全保持列を隣接2:1算術平均統合して半減し、`stride_ *= 2`。論理フレームはpendingバッファでstride_個平均してから保持列へ確定する。
- **Scroll**: 保持数がmax_frames超で先頭を破棄(直近max_frames窓)。
- **Overwrite**: リング上書き。
- 現行`TimeHeatMap`のScroll/Overwriteは点列が消えない実装(`EraseCol_`が空)で意味論が壊れているため、上記を正とする。
- メモリ上限: `max_frames × fine_bins × 4B`(既定1000×2048×4=8MB)+pending 1フレーム。

### 3. 描画=fine→coarse集約+既存グリッド描画パスの再利用

`RenderRaw`のたびに次を行う(発生はlog_interval毎とEvalPanel表示時のみ)。

1. 各保持フレームをfine→coarseへ集約する。fine bin中心のt値を表示レンジで`bins`へ写像(現行`MapToBin_`と同じ正負分割・clamp規則をt空間で適用)。
2. `HM_AutoNormValue`有効時は現行どおり**フレーム単位**でmax=1正規化する。
3. 保持フレーム列(F個)を表示列W(=max_frames)へ伸長し(F<Wの間は現行Scale表示と同様に全幅へ最近傍伸長)、W×binsグリッドを構築する。
4. `thm_`には点列を蓄積せず、`HeatMap::SetGridValues`でグリッドを流し込み`HeatMap::RenderRaw`に描画させる(Jetカラー、`HM_LogScaleValue`、`HM_FlipY`、背景色、OpenMP描画を再利用)。`SetGridValues`後に`SetValueMinMax(0,1)`を再設定して値レンジを固定する(AutoNormValue無効時はSetGridValuesの自動min/maxに任せる)。
- `TimeHistogram::RenderRaw`のゼロライン描画は現行のまま。
- コスト: O(F×fine_bins + W×bins + W×H)≈数ms。TrainThread上のホットパス(AddBatch/NextFrame)から描画・集約コストが完全に分離される。

### 4. sample_mode(Observer)

- `TimeHistogramObserverConfig`に`sample_mode`(enum: Accumulate | Snapshot、既定Accumulate)を追加する。
- Snapshot時の`OnLearn`は、frame境界step(`step % frame_interval == 0`)でのみ`GetVector`+`AddBatch`+`NextFrame`を行い、それ以外のstepでは何もしない。probe取得(gather+正規化除算+vector化)とAddBatchのコストが1/frame_intervalになる。RB側で発生するO(N)処理(gather・除算)はmax_samples(方針5)では減らせないため、その削減は本頻度削減が受け持つ。
- configキー: `histgram.sample_mode = accumulate | snapshot`(文字列、`core/anet-core/include/anet/image.hpp`のTimeHistgramConfigパースへ追加)。
- `apps/runner/config/metrics_image.txt`の`image.thg.per-prio`プロファイルへ`histgram.sample_mode = snapshot`を追加する(PER_DISTはスナップショット分布であり、蓄積は情報増ゼロのため)。

### 5. max_samples(Probe)

- `ProbeConfig`に`max_samples = 0`(0=無制限)を追加する。
- `AgentTensorVectorProbe::GetVector`(`core/anet-core/src/probe.cpp`)で、flatten後の要素数nがmax_samplesを超える場合、`stride = ceil(n / max_samples)`のスライス間引き(`flat.slice(0, 0, n, stride)`+`contiguous()`)を**memcpy・vector化の前**に適用する。RBはCPU常駐のため`.to(kCPU)`はno-opであり、本手段が削るのはコピー量とAddBatch投入量(=TimeHistogram側のビニングコスト上限)である。RB側のgather・正規化除算O(N)は減らない(方針4のsnapshotが受け持つ)。
- stride間引き(決定的)とし、乱択はしない(同seed再現性を壊さない)。RBのindex順は挿入順のため、strideサンプルは全時期を等間隔にカバーし分布推定として偏らない。
- configパース追加は本PRDではTimeHistgramセクションの`histgram.probe.value.max_samples`のみ(ProbeConfig自体は共通structのため、他observerへの展開は将来必要時にパース行を足すだけでよい)。
- `image.thg.per-prio`プロファイルへ`histgram.probe.value.max_samples = 65536`を設定する。

## ユーザーストーリー

1. DropMergeの学習を回す利用者として、PER優先度分布の時系列画像を有効にしても学習スループットを犠牲にしたくない。そうすれば、ハイパラ探索Runで常時この診断画像を出せる。
2. 長時間Run(数千万step)を回す利用者として、画像メトリクスの保持メモリと1回あたり処理時間が経過時間に対して一定であってほしい。そうすれば、Run後半の減速やメモリ枯渇を心配せず放置できる。
3. メトリクスを設定する利用者として、スナップショット系probeには`sample_mode = snapshot`を指定して取得回数を減らしたい。そうすれば、意味の重複した蓄積にコストを払わずに済む。
4. frameworkを保守する開発者として、「保持は固定fineグリッド、表示は描画時写像」という1関心1機構の構造にしたい。そうすれば、レンジ追従・フレーム正規化・表示モードを再構築ロジックと絡めずに変更できる。
5. RB容量を増やす実験をする利用者として、probeの要素数に上限を掛けたい。そうすれば、容量スケールに比例してメトリクスが重くなることを防げる。

## 実装上の決定

- `fine_bins = bins × 8`は内部定数とし、configノブにしない(表示binの1/8の量子化誤差は表示上不可視。ノブ増殖を避ける)。
- 縮約はbin幅2倍・格子位相0起点の入れ子構造で行い、旧binが常に新binへ丸ごと入るため再配分誤差ゼロ。フレームあたりO(fine_bins)の加算統合のみ。
- t空間写像により、log軸でもfine bin境界がレンジ非依存になる。現行`MapToBinLogAxis_`のmin/max正規化・正負分割は、描画時のfine→coarse写像側で同じ規則を適用して再現する。
- Scale統合を算術平均(和ではなく)にするのは、AutoNormValue無効時にも統合フレームの明度が跳ねないようにするため。AutoNormValue有効時はどのみちフレーム正規化されるため差は出ない。
- `MeanMode`/`SumMode`の点列描画上の差は、SetGridValues経由(1セル1値)では消滅する。フレーム→列伸長と統合の平均化がこれを代替する(現行の使用実績はMeanModeのみで、視覚差は表示ピクセル集約の範囲)。
- `TimeHistogramObserver`のmutex(OnLearn/GetImageData間)は現行のまま。TimeHistogramに新たな共有可変状態を追加しない。描画がOnLearnと同一ロックで直列化される構造も維持する(描画は数msに軽量化されるためロック保持は問題にならない)。
- snapshotの既定化はしない(既定accumulate)。batch TDエラー等「毎learnで異なる値が来る」probeにはaccumulateが正しく、既存設定の意味を黙って変えない。
- `ANET_PROFILE_FUNC`/`ANET_PROFILE_SCOPE`は同名で維持し、改修後のTracy比較を可能にする(`RebuildFromRaw`ゾーンは削除されるので消滅が確認信号になる)。
- 削除対象: `frames_raw_`、`cur_raw_`、`need_rebuild_`、`RebuildFromRaw`、`AppendCurrentFrameOnly`、`buffer_`(未使用ワークバッファ)。`TimeHeatMap::AddData/NextFrame`のTimeHistogramからの使用。
- dead設定行`histgram.alpha`(パースがコメントアウト)は本PRDでは触らない(復活も削除もしない。対象外参照)。

## テスト方針

heat_map系の単体テストは現存しないため、`core/anet-core/src/heat_map_test.cpp`を新設する。公開API(`AddBatch`/`NextFrame`/`RenderRaw`/`Reset`/`MinVal`/`MaxVal`)と観測可能な出力のみ検証する。

### TimeHistogram単体

- カウント保存: AutoNormValue無効でAddBatch→NextFrame後、描画グリッドの総和が入力件数と一致する(縮約を跨いでも保存される)。
- 縮約の正確性: fineレンジ拡大を強制する入力列で、拡大前後の分布形状(coarse集約結果)が量子化誤差(coarse 1bin未満)の範囲で一致する。
- 写像互換: 既知入力(線形軸・log軸それぞれ)でcoarse分布が現行`MapToBin_`の割り当てと一致する。
- Scale統合: max_frames到達で保持数が半減しstrideが倍化する。以後の保持数が上限を超えない。
- Scroll/Overwrite: 保持数上限と窓/リングの意味論。
- 境界: 空、1フレーム、全値同一(ゼロ幅レンジ)、NaN混入なし前提の負値のみ/正負混在(log軸の正負分割)。
- `RenderRaw`が各状態で正寸のwxImageを返す。

### Observer(observers_test.cppへ追加)

- mock probeで、`sample_mode = snapshot`時のGetVector呼び出しがframe境界stepのみであること。accumulate時は毎learnであること(現行挙動の回帰)。

### Probe(probe_test.cppへ追加)

- `max_samples = 0`で全件、`max_samples < n`で件数が上限以下かつstride間引きの期待要素であること。

### 性能・結合

- 既存の全テストが通ること。
- 受入の性能確認はTracy実測で行う(下記受入条件8)。単体テストでの時間計測はしない。

## 受入条件

1. `TimeHistogram`に生値保持(`frames_raw_`/`cur_raw_`)が存在せず、保持メモリが`max_frames × fine_bins × 4B`+定数を上限とする。
2. `RebuildFromRaw`が存在せず、値レンジ拡大時の処理(縮約)がO(保持フレーム数×fine_bins)で入力累計件数に依存しない。
3. `thm_`への点列蓄積(`AddData`/`NextFrame`)が無く、描画は`SetGridValues`+既存`RenderRaw`経由である。`TimeHeatMap`/`HeatMap`の外部仕様は無変更。
4. `histgram.sample_mode = snapshot`で、probe取得とAddBatchがframe境界のみになる(mock検証)。既定はaccumulateで現行挙動と一致する。
5. `histgram.probe.value.max_samples`で`AgentTensorVectorProbe`の返却がmemcpy・vector化前に間引かれる。既定0=無制限。
6. `image.thg.per-prio`プロファイル(`metrics_image.txt`)が`sample_mode = snapshot`と`max_samples = 65536`を指定している。
7. config未変更の既存メトリクス(TimeHistgram以外を含む)の挙動・出力が変わらない。
8. `52_per_thg_prio`有効のDropMerge RunのTracyで、`TimeHistogramObserver::OnLearn`配下の合計がTrainThread時間の1%未満(現状72.71%)。`RebuildFromRaw`ゾーンが存在しない。
9. Scaleモードの出力画像が従来と視覚同等(分布の山・レンジ推移・対数軸の正負分割が一致。ピクセル同一性は要求しない)。
10. 新設テストを含む全テストが通る。

## 対象外

- Observerのオフスレッド化・MetricsLoggerライフサイクル(PRD 026系)
- `histgram.alpha`パースの復活、表示レンジEMA挙動の変更、`base_min`/`base_max`のconfigノブ追加(fineグリッド化でrebuildが消えるため、これらは純粋に見た目の関心として将来判断)
- `HeatMap`/`TimeHeatMap`/`MultiPairHeatMapObserver`等、点列ベース設計の他クラスの再設計
- `TimeHeatMap`のScroll/Overwriteの現行点列挙動の再現(TimeHistogram側で意味論を定義し直す)
- VideoLogger関連(PRD 025)
- `max_samples`のTimeHistgram以外のobserverセクションへのパース展開

## 実装時のドキュメント更新

- `docs/design/140_observability.jp.md`: TimeHistgram型の保持構造(fineグリッド・max_frames上限・描画時集約)、`histgram.sample_mode`、`histgram.probe.value.max_samples`を追記する。スナップショット系probe(PER_DIST等)にはsnapshotを推奨する旨を記す。

## 補足

現行Scale表示は「全点を保持し描画時にx圧縮」、本PRD後は「事前に2^k論理フレームを平均統合」となる。max_frames(=表示列数)以上の履歴を1000列に描く時点でどのみちピクセル集約されるため、視覚差は表示解像度未満に収まる。

性能見積り(N=RB valid件数、F=保持フレーム数): 現行はAddBatch O(N)/learn+rebuild O(累計生値)がほぼ毎フレーム。改修後はsnapshot時でAddBatch O(min(N, max_samples))/frame境界のみ、縮約O(F×fine_bins)が生涯log回、描画O(F×fine_bins+W×H)がlog_interval毎。TrainThread上の定常コストはフレーム境界のgather+除算+コピー+fine加算のみになる(RBはCPU常駐であり、この経路にD2Hは存在しない)。
