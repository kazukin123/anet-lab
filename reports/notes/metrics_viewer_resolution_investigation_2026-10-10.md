# MetricsViewer 複数 Run の表示粒度: 再現調査と対応方針

調査日: 2026-10-10（Asia/Tokyo）
状態: 調査・提案のみ。Viewer の実装、設定、設計正本は未変更。

## 結論

症状を、現在起動中の Viewer の API と配信中の実際の client code を使って再現した。

1. 全タグをまとめて取得すると、request 全体の 500,000 vertex 予算を多数の系列で分ける。学習報酬の LOD が L2（256点単位）から L3（4,096点単位）へ跳び、描画点数が約2,500点から約160点へ減る。
2. cache は範囲・zoom・generation・metadata を検査するが、比較する Run 集合や表示粒度を検査しない。操作順によって同一 graph に L2 と L3 が残る。metadata 不変の Reload はこれを修正しない。
3. 同じ request と同じ pointBudget でも、level は Run ごとに独立決定する。記録点数が閾値を跨ぐと粒度が16倍異なる。

直接の引き金は memory pressure の検出ではなく、点予算と射影・cache の契約である。
4本の学習 Run × 3タグなら、現行の API でも全て L2 を維持でき、応答は約0.92 MB だった。
全タグを表示する場合の browser heap や Plotly の保持量は別途検証が必要。

## 調査条件と確認範囲

- 接続先: http://localhost:8082
- current workspace: DropMerge-02。workspace 切替・Run 再配置・cache 削除は実施していない。
- 配信中 metrics-viewer.js と repository のファイルを LF 正規化後に比較し、一致を確認。
- API の実応答を DataCache / MetricsViewerClientApp.requestVisibleData() に渡す Node の一時 harness で検証。
- DOM、Plotly 描画、UI 操作は stub 化した。browser 上でクリックを通した E2E、heap 測定、OOM の有無の検証は未実施。
- ユーザー添付画像の Run を使った。画像だけでは選択全タグ数を確定できないため、表は「全タグ選択」の明示的な再現条件。
- 青の Run は調査中も metadata の count/maxStep が増えた。ingest.state=ready は学習完了を意味しない。各 probe 内では metadata snapshot を固定。点数の数件の変動と LOD 問題は区別する。
- C++/Java の build、学習・評価 Run の追加実行は実施していない。

対象 Run:

| 略称 | Run |
|---|---|
| A（添付の青） | run_20261010-045911_dm_fb_munch_tau3e-2_ndtpen20_nzrep_map16_100m |
| B（添付の赤） | run_20261009-130921_dm_fb_munch_tau3e-2_ndtpen20_nzrep_b512rr2_100m |
| C（添付の黄） | run_20261009-035826_dm_fb_munch_tau3e-2_ndtpen20_nzsoff_100m |
| D（4本目の学習 Run） | run_20261008-200136_dm_fb_munch_tau3e-2_ndtpen20_eps0_100m |

比較タグ: 20_eps/10_train_reward、20_eps/11_train_reward_ema、21_eval/01_target_reward。
A/B/C の学習報酬は約21〜22万点、D は257,193点。

## 再現結果

### 1. 選択数で急激に粗くなる

全タグを新規取得したときの 20_eps/10_train_reward:

| 選択 Run | 選択タグ数 | request 系列数 | 各学習報酬の pointBudget | level / bucketWidth | MinMax 描画点数 |
|---|---:|---:|---:|---|---|
| A | 214 | 214 | 2,689 | L2 / 256 | 2,575 |
| A+B | 214 | 428 | 1,248 | L3 / 4,096 | 162 / 156 |
| A+B+C | 216 | 644 | 785 | L3 / 4,096 | 162 / 156 / 156 |
| A+B+C+D | 216 | 851 | 588 | 全て L3 / 4,096 | 後述 probe で再取得可能 |

最初の3行は同じ測定系列。4行目は後の測定。A の追記により後の probe では L3 の描画点数が165点になった。
Run に存在しないタグは request へ入れないため、系列数は常に Run数 × タグ数とはならない。

3 Run × 3タグだけを新規取得すると:

- 学習報酬と EMA は全て pointBudget=8,000、L2 / 256。
- 学習報酬の描画点数は 2,575 / 2,489 / 2,492。
- 評価報酬は全て raw。3,126 / 3,906 / 3,906点。
- JSON 応答は 659,951 bytes。

4本の学習 Run × 3タグでは:

- request は12系列。
- 学習報酬は全て pointBudget=8,000、L2 / 256。
- 後の測定で描画点数は 2,587 / 2,489 / 2,492 / 3,008。
- JSON 応答は 916,280 bytes、配分予算合計78,864点。
- 同じ4 Run の全タグ取得は851系列、予算合計500,000点、JSON 応答4,735,968 bytes。

タグ数を減らせば常に改善するわけではない。既取得 cache をそのまま使う場合は次の問題が残る。

### 2. 操作順で異なる粒度が残る

実際の requestVisibleData / DataCache に以下の状態遷移を与え、2回とも同じ不一致を再現した。

1. A+B+C、全タグを取得。学習報酬は全て L3 / 4,096。
2. activeTags を学習報酬1タグへ絞る。再取得0件で、粗い cache が残る。
3. B だけを選ぶ。viewport を [-50,000, 450,000] に移す。window が cache 外まで広がるので B を再取得し、B だけ L2 / 256 になる。
4. autorange へ戻し A+B+C を選ぶ。再取得0件。同一 graph が A=L3、B=L2、C=L3 となる。
5. 同じ metadata で refresh=true（Reload の再取得判定）を実行。再取得0件。不一致が残る。

測定値:

| Run | pointBudget | bucketWidth | 描画点数 |
|---|---:|---:|---:|
| A | 785 | 4,096 | 165 |
| B | 8,000 | 256 | 2,489 |
| C | 785 | 4,096 | 156 |

B の取得時 viewport.width は500,000。
復帰後の graph の viewport.width は387,209で、再取得条件の 500,000 × 0.75=375,000 を下回らない。
B の window [-550,000, 950,000] は復帰後の要求範囲を覆う。A/C の window も覆うため、どの系列も stale にならない。
「どれか stale なら graph 全体を再取得する」という処理まで進まないのがポイント。

### 3. 同時取得・同一予算でも level が揃わない

A/B/C の学習報酬を同じ request、fromStep=0、toStep=1,000,000、maxPoints=2,500 で取得した結果:

| Run | pointBudget | level | bucketWidth |
|---|---:|---:|---:|
| A | 2,500 | 3 | 4,096 |
| B | 2,500 | 2 | 256 |
| C | 2,500 | 2 | 256 |

B/C は L2 の832 bucket × 3=2,496点で予算内。
A は約861〜864 bucket × 3で2,500点を超え、L3へ跳ぶ。
実際の MinMax は min/max/last の重複を除去するが、level の選択では最大3点/bucketとして予算を判定する。

これは公開 API の明示予算を使った境界試験。frontend の既定操作が必ず2,500点を配るという主張ではない。

## 原因箇所

| 箇所 | 現行挙動 | 問題との関係 |
|---|---|---|
| metrics-viewer.js:2165 requestVisibleData | activeTags の全 graph を1つの series 配列へ入れる | 他の graph が表示粒度へ影響する |
| metrics-viewer.js:2292 _visibleSelectedTags | 選択中 Run に存在する activeTags を列挙 | 名前の visible は画面内可視性ではない。画面外も含む |
| MetricsQueryPlanner.java:93 allocatePointBudgets | request 全体の予算を全系列へ配分 | 既定8,000点を下回る |
| MetricsRangeProjector.java:269 selectLevel | 系列ごとに bucket数 × 3 <= 予算 の最小levelを選択 | 16倍刻みの急落・Run 間不一致 |
| LodBucket.java:21 | factor=16 | 256 → 4,096 → 65,536 と跳ぶ |
| metrics-viewer.js:441 needsFetch | 範囲、generation、zoom率、refresh時metadataのみ | Run集合・粒度・予算条件が cache の成立条件にない |
| metrics-viewer.js:2581 onReload | metadata が変わった graph だけ再取得 | metadata不変の粒度混在を解消しない |
| metrics-viewer.js:735 _makeSeriesTraces | MinMax projection をそのまま Plotly へ渡す | Plotly より前に点数が既に減っている |
| LodPageCache.java:46,79 | 指定levelのpageをLRU保持し、容量不足なら保持を制限 | cache-memory-mb は表示levelの決定要因ではない |

現行設計正本: docs/design/210_metrics_viewer.jp.md §2.6〜2.8、§9.2。
§2.8の「同じ request なら解像度がそろう」という説明は、現在の独立level選択とcache再利用では保証されない。
実装時にはこの説明を新しい保証へ改める必要がある。

## 対応案の比較

| 案 | 解決する点 | 残る点 | 判断 |
|---|---|---|---|
| max-points-per-request の増加だけ | 粗くなる閾値を遅らせる | 16倍の段差、独立level、cache混在、全タグのheap増加 | 単独対応として不十分 |
| graph単位の取得・予算とcache検証、共通level | Run数・他タグによる予算低下とcache混在を抑える | 序数幅が同じでもstep幅が異なる。16倍刻みも残る | 局所的な改善案 |
| graph共通のstep区間への射影 + 分割取得 + cache検証 | 比較粒度を直接保証し、段差と操作順依存を解消する | 境界再集約のコスト検証とAPI変更が必要 | 推奨 |

## 推奨方針（未実装・未合意）

### 1. 粒度を graph の契約にする

粒度は「何点を1つに畳むか」ではなく「横軸のどの step 区間を1つに畳むか」で揃える。
同じ tag、同じ viewport、同じ graph 幅なら、比較 Run 全てに共通の step 境界を使う。

例えば同じ graph の [0, 400,000] を同じ区間へ分ければ、各 Run の記録頻度が異なっても同じ範囲の min/max/mean を比較できる。
同じ ordinal bucketWidth=256 に揃えるだけでは、episode 長などにより覆う step 幅が異なる。

- 少なくとも1〜4 Runでは、同じ viewport・graph 幅の1系列あたり表示密度を維持する。
- 他の tag の追加や画面外 graph は、その密度を下げる理由にしない。
- 同じ step に複数点がある記録を失わない。min/max は実際の step を保持し、mean はcountで重み付けする。
- 全系列が生点で十分小さければ raw へ到達する。元の記録頻度を補間で偽装しない。
- 区間幅、丸め方、原点、raw/集約のgroup全体の切替条件は実装計画で確定する。固定viewportなら Run追加で境界を変えない方針。
- graph headerへ集約幅を示す案は有用だが、診断表示だけでは今回の問題を解決しない。

### 2. 表示予算と転送上限を分ける

まず graph ごとの必要密度を決め、転送上限に収まるよう request を分割する。
1 requestの上限を守るために全 graph を粗くする現在の配分を改める。

- graph内の比較系列は同じ計画で取得する。4 Run × 8,000 vertexなら1 graph最大32,000 vertexで、現在の500,000上限以内。
- 複数graphを1 requestへ束ねる場合も、graphごとの密度を維持できる数だけ束ねる。
- 画面内とその近傍を優先取得・描画する。画面外は必要時に取得する。
- browser側のwindow保持量を制限し、画面外のPlotlyオブジェクトも必要に応じて解放する。全216graphを高密度で常駐させない。
- request分割は DataFetcher.fetchMetrics() が新規呼出しごとに古い通信をabortする契約と、serverのquery supersedeを考慮する。単純な並列fetch追加は不可。
- 選択変更で古い計画をcancelし、同じ計画の分割queryを順次処理する最小構成から始める。

### 3. cache の成立条件と置換単位を合わせる

- generation と範囲に加え、共通のstep境界・密度などの表示計画を照合する。
- Run選択変更時は、既取得の系列が同じ計画で比較できるかを検証する。同じ計画なら再利用し、異なればgraphの必要系列を再取得する。
- Run数やactiveTagsが変わっただけで常に全cacheを捨てる設計にはしない。
- graph単位で応答を検証し、一貫した状態で置換する。通信待ちの表示にも異なる粒度を混ぜない。
- Reload はmetadata更新の判定と粒度整合の判定を区別する。

### 4. SQLite の序数LODと画面のstep集約を分ける

ADR 0015 と正本 §2.6 は、保存LODを序数基準とする理由を持つ。
永続LODのfactor16やL0のordinal PKを変える必要性は、この調査では認めていない。
保存済みLODを読み出し高速化に使い、画面向けの共通step区間へ再集約する案を第一候補とする。

既存の部分bucket再集約経路は出発点になるが、新しい全境界へ適用したときの性能は未検証。
境界ごとにL0を大量走査したり大量のSQL往復を発生させない処理を、小さなprobeで先に評価する。
TagStatsは表示計画に関わらずL0全点の正確な統計を維持する。
もし保存LODの序数契約を変えるなら、ADR 0015との変更点・理由を明示して別途決定する。

## 実装時の順序と受け入れ条件

1. 先に共通step射影の小さな性能probeを作り、4 Runのgraphで要求密度を満たせるか測る。
2. graph単位の表示計画・API・射影を決め、異なる記録密度でも境界が一致する契約testを置く。
3. frontendのcache整合、graph単位置換、分割queryのcancel/supersedeを実装する。
4. 画面内取得とbrowser保持量・Plotly lifetimeを制限し、4 Run × 全タグ選択で実際のブラウザを検証する。
5. docs/design/210_metrics_viewer.jp.md と英語版の現行契約・検証章を更新する。

受け入れ条件:

- 固定viewport・graph幅で1→2→3→4 Runと追加しても、各系列の集約step幅が維持される。
- 同じgraphの全Runは共通の集約境界を使う。記録件数・step頻度・到達stepが違う場合も検証する。
- 全タグ選択、少数タグ選択、タグ削減、Run単独表示→比較復帰、pan/zoom→Reset View、Reload/Auto Reloadの順序で同じ表示計画へ収束する。
- 数値境界で点予算が少し変わっても16倍粗くならない。
- 同一stepの重複点、部分bucket、未完成末尾、追記、欠損tagを正しく扱う。
- generation切替・workspace切替・409 superseded・中断時に古い/部分的な応答を混ぜない。
- 4本の学習Run × 全タグ選択で、画面内の粒度を維持し、画面外の保持量がboundedである。
- MinMax/Mean/Band、Log、percentile、hover、凡例、scroll lock、screenshot modeを回帰確認する。screenshot modeは全graphを扱うため別の保持戦略を明示する。
- 最低限queryの応答時間・応答bytes・browser heap・server heap・Plotly保持graph数を測定する。許容数値はbaseline取得後に決める。

## 制約と次の判断

今回確認した約0.92 MBは、4 Run × 3タグの転送JSONサイズであり、browser全体のmemory測定ではない。
全216タグの高密度常駐が1 GB heapなどへ収まるとは結論していない。
メモリ対策を表示粒度の低下で賄うより、可視graph数・cache常駐量・同時queryを制限する方針を推奨する。

この文書は調査成果と対応方針であり、実装着手の承認やPRDを兼ねない。


## 保存済み probe の実行確認

次のコマンドを repository の PowerShell から実行し、終了値0を確認した。

```bash
node C:\dev\anet-lab\reports\metrics_viewer_resolution_probe_2026-10-10.cjs
```

保存済み probe でも、cacheMismatchRepeated=true、equalBudgetDifferentLevels=true。
その時点のAは223,573点、全タグ時の1系列予算は1 Run=2,686、2 Run=1,247、3 Run=785、4 Run=588。
4 Run × 3タグの応答は918,268 bytes、学習報酬は全てL2 / 256だった。
上の表との差は追記中のAの点数増加による。原因と再現パターンは同じ。

このprobeは現状を観測する調査資材で、将来の修正後に現状の症状を要求する回帰testではない。
Runやworkspaceが移動した場合はその前提検査で停止し、勝手に切り替えない。

