# 指摘台帳(派生ビュー)

生成 2026-09-19。正本は findings.jsonl。この文書は毎回再生成される。

## 概況

| 状態 | 件数 |
|---|---|
| open | 1 |

open の優先度別: P0 0, P1 1, P2 0, P3 0

open の観点別: architecture 0, backlog 0, defect 0, drift 0, process 0, refactor 1, test-gap 0

## 最近の変化(14 日)

| id | 更新 | 状態 | 優先度 | 観点 | 場所 | タイトル |
|---|---|---|---|---|---|---|
| F-0001 | 2026-09-19 | open | P1 | refactor | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile | 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している |

## open(優先度順)

| id | 優先度 | 確度 | 観点 | 分類 | 場所 | タイトル | 初出 | 最終確認 |
|---|---|---|---|---|---|---|---|---|
| F-0001 | P1 | high | refactor | 1関心1機構違反 | `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile | 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している | 2026-09-19 | 2026-09-19 |

### 詳細

#### F-0001 P1 排他的な外れ値表示モードを二つのSetと分散した排他処理で保持している

- 観点/分類: refactor / 1関心1機構違反
- 場所: `apps/metrics-viewer/src/main/resources/static/metrics-viewer.js`::MetricsViewerClientApp.outlierPercentile
- 要約: 同じtagの外れ値表示はoff・p5–p95・p1–p99の択一だが、ignoreOutlierTagsとp1P99Tagsの二つの集合で表現している。排他条件をtoggle、復元、描画値の選択へ分散させている。現状の補正は存在し、表示バグを指摘するものではない。 元の優先度P2を、対象範囲内で最多のchurn90=10のファイルとして1段上げた。
- シナリオ: 新しいpercentile表示を追加すると、集合と保存キーに加えて既存toggle双方、_loadStateの競合解消、outlierPercentileの優先順を同時に変更する必要がある。現状でも二つの保存集合へ同じtagが入った状態を_loadStateが補正している。
- 対処: 外れ値表示の状態遷移を一箇所へ集約し、内部ではtagごとの単一モードとして扱う。既存保存形式と競合時p1–p99優先の挙動を保つ入出力境界を明確にする。OutlierRangePlaywrightTestの排他・永続復元・手動Y範囲を特性テストとして確認し、描画経路の性能が同等であることを確認する。新しい保存契約へ変更する場合は別途設計判断を行う。
- 根拠: apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2001 二つのtoggleが相手のSetを消す; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2023 二つのSetから優先順でpercentileを復元; apps/metrics-viewer/src/main/resources/static/metrics-viewer.js:2423 永続状態の復元にも排他違反の補正が必要


## クローズ済み・保留

無し。
