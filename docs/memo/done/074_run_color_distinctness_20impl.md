# Run 色分離 実装メモ

対象 PRD: [074_run_color_distinctness_10prd.md](074_run_color_distinctness_10prd.md)。
決定の理由と棄却案は [ADR 0041](../../adr/0041-run-color-is-comparison-role-not-run-identity.md)、用語は [CONTEXT.md](../../../CONTEXT.md) の「Run色」「分離距離」。

## 概要

Metrics Viewer の browser 側だけを変更し、選択中 Run の色を見分けられる状態にする操作を 2 つ足す。

- `Recolor` ボタン: 現在の色を無視し、選択順に palette 先頭 `#2F7DE1` を起点とした farthest-point で配る（最適配色）。
- `Auto` チェックボックス（既定 ON、`localStorage` 永続）: 選択集合が変わるたびに、既存色が `min(0.16, 今の palette で取れる最良)` を満たさない Run だけを配り直す（維持優先）。

色は 20 色 palette からのみ選び、未選択 Run の色は変更しない。選択が 20 本を超えたらラウンドを回す。選択 1 本以下では何もしない。
server、Metrics キャッシュ、HTTP API は変更しない。

## 主な変更

### 1. [index.html](../../../apps/metrics-viewer/src/main/resources/static/index.html)

`#run-section .section-controls` へ、`Select All` / `Latest Only` の右に 2 要素を足す。チェックボックスは Tags の `Filter` と同じ inline style 形式にする。

```html
<button id="btn-recolor-runs" title="Give the selected runs the most distinguishable colors">Recolor</button>
<label style="margin-left:8px; cursor:pointer; font-size:13px;"
		title="Keep the selected runs distinguishable as the selection changes">
	<input type="checkbox" id="chk-auto-recolor" style="vertical-align:middle; margin-right:2px;">Auto
</label>
```

### 2. [metrics-viewer.js](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)

| 箇所 | 変更 |
|---|---|
| 先頭の定数群 | `STORAGE_KEY_AUTO_RECOLOR = "anet.metricsviewer.autoRecolorEnabled"`、`RUN_COLOR_MIN_DISTANCE = 0.16` を追加 |
| `colorWithAlpha` の近く | `oklabDistance(hexA, hexB)` を module scope の純関数として追加（sRGB → 線形 RGB → OKLab のユークリッド距離）。補助の `hexToOklab(hex)` も同じ場所へ置く |
| 同上 | `farthestColorFrom(fixed, used)`（未使用色のうち `minDist` 最大、同値は `RUN_COLORS` の並び順で先勝ち）と `minColorDistance(color, fixed)`（`fixed` が空なら `Infinity`）を追加 |
| `MetricsViewerClientApp` constructor | `this.autoRecolorEnabled = true;` を追加 |
| `MetricsViewerClientApp` | `_applyRunColors()` を新設。(a) 未割当 Run へ runId 昇順で基本色を配る、(b) `autoRecolorEnabled` なら維持優先パス、の 2 段 |
| 同上 | `_recolorSelectedRuns({ keepExisting })` を新設。選択順の 1 ループで両方の契約を表す。`selectedRuns.length <= 1` は先頭で return |
| 同上 | `onRecolorRuns()` を新設。`_recolorSelectedRuns({ keepExisting: false })` → `refreshLists()` → `_renderCurrent()`（`onToggleLog` と同じ再描画経路） |
| 同上 | `onToggleAutoRecolor()` を新設。state 更新 → `localStorage` 保存 → `_syncAutoRecolorUi()` → `refreshLists()` → `_renderCurrent()` |
| 同上 | `_syncAutoRecolorUi()` を新設（checkbox の `checked` を state へ合わせる）。`init()` で `_syncGraphScrollLockUi()` の隣から呼ぶ |
| `refreshLists()` | 先頭で `this._applyRunColors()` を呼ぶ |
| `_loadState()` | `this.autoRecolorEnabled = localStorage.getItem(STORAGE_KEY_AUTO_RECOLOR) !== "false";`（未設定・不正値は ON） |
| `UIController.renderRunList` | 色を配るブロック（`if (!runColorMap.has(runId)) ...`）を削除し、`runColorMap` は読むだけにする。引数は現行のまま |
| `UIController.bindStaticControls` | `btn-recolor-runs` の `onclick` と `chk-auto-recolor` の `onchange` を bind |

`_recolorSelectedRuns` の中身は PRD §3.2 / §3.3 の疑似コードどおり。`keepExisting` が false のときは維持判定を飛ばして常に `farthestColorFrom` を採る。
`used.size === RUN_COLORS.length` になったら `fixed` と `used` を空へ戻す（ラウンド）。

`refreshLists()` は metadata 更新、選択変更、tag 選択変更、tag lock 切替の 4 経路から呼ばれる。
維持優先パスは冪等なので、選択が変わらない経路で走っても結果は変わらない。計算量は選択本数 × 20 の距離計算で、毎回実行して問題ない。

### 3. ドキュメント

- [030 §3 画面の基本操作](../../design/030_user_guide_analysis.jp.md): `Recolor` と `Auto` の行を追加し、`localStorage` へ保持する項目の列挙へ 1 件足す。
- [210 §3.2 browser](../../design/210_metrics_viewer.jp.md): Run 色の所有が `MetricsViewerClientApp` 側であることを反映する。
- [210 §7.4](../../design/210_metrics_viewer.jp.md): 定数表へ `RUN_COLOR_MIN_DISTANCE`、`localStorage` 表へ `anet.metricsviewer.autoRecolorEnabled` を追加し、「8 件」を「9 件」へ直す。色の決定規則を同節へ短く書く。

## テスト

### Public interface / surface

browser から観測できる面だけを検証し、内部関数を直接呼ばない。

- `#btn-recolor-runs` の click と `#chk-auto-recolor` の状態。
- Run list の chip 色（`.run-color` の `backgroundColor`。既存の `readRunChips`）。
- Plotly の trace 色（既存の `readTraceColor`）。
- `localStorage` の `anet.metricsviewer.autoRecolorEnabled`。

追加先は [RunListPlaywrightTest](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunListPlaywrightTest.java)。

### fixture

`run_01`〜`run_14` の 14 Run（tag 1 本）を route stub で返す。基本色は runId 昇順の先着順なので palette と 1 対 1 に決まる。

| Run | 基本色 | palette index |
|---|---|---:|
| `run_01` | `#2F7DE1` | 0 |
| `run_03` | `#7A5CFF` | 2 |
| `run_10` | `#E23B4F` | 9 |
| `run_13` | `#D1D83B` | 12 |
| `run_14` | `#B83280` | 13 |

`#E23B4F` と `#B83280` の分離距離は 0.1204 で、しきい値 0.16 では再割り当て、0.12 では維持になる。この組がしきい値を固定する。
20 本超えのテストだけ `run_01`〜`run_25` の別 fixture を使う。

### 優先 behavior

| # | behavior | 期待 |
|---|---|---|
| B1 | `Auto` を OFF にし、`run_01` → `run_03` → `run_10` → `run_14` の順で選んで `Recolor` を押す | 選択順に `#2F7DE1` / `#D1D83B` / `#E23B4F` / `#00B36B`。trace 色も chip 色と一致する |
| B2 | B1 の直後にもう一度 `Recolor` | 色が変わらない（冪等） |
| B3 | `Auto` 既定 ON のまま `run_10` を solo 選択し、`run_14` を追加 | `run_10` は `#E23B4F` のまま、`run_14` が `#D1D83B` へ移る |
| B4 | B3 に続けて `run_03` を追加し、その後 `run_14` を外す | `run_03` は基本色 `#7A5CFF` を維持。追加でも解除でも先行 Run の色は変わらない |
| B5 | `Auto` を OFF にして reload | OFF のまま。`localStorage` 未設定の初回ロードは ON |
| B6 | B1 の前後で未選択の `run_13` の chip | `#D1D83B` のまま変わらない（選択中の `run_03` と同色になることを許す） |
| B7 | 25 Run の fixture で `Select All` | 例外なし。先頭 20 本が相異なり、21 本目が `#2F7DE1` |
| B8 | `run_10` を solo 選択して `Recolor` | 色が変わらない（選択 1 本は no-op） |

### TDD 順序

horizontal slicing はしない。1 behavior ごとに RED を確認してから最小実装で GREEN にする。

1. **tracer bullet = B1**。HTML のボタン → `onRecolorRuns` → 最適配色 → `refreshLists` / `_renderCurrent` → chip と trace までの経路を 1 本で通す。この時点で `oklabDistance`、`farthestColorFrom`、`_recolorSelectedRuns({keepExisting:false})`、ボタンの bind、`Auto` の OFF 操作までが入る。
2. B2（冪等）。B1 の実装で通るはずなので、RED にならなければテストだけ追加して次へ進む。
3. B3。ここで維持優先パスと `RUN_COLOR_MIN_DISTANCE`、`_applyRunColors` の 2 段構成、`renderRunList` からの色割り当て移設が入る。
4. B4（追加安定・解除安定）。
5. B5（`localStorage` 永続と既定 ON）。
6. B6（未選択不変）。
7. B7（ラウンド）。
8. B8（1 本 no-op）。
9. 全体 GREEN の後に refactor（共通化・コメント整理）を行い、そのつど `mvn -B test` を再実行する。

## 検証

```bash
mvn -B -f apps/metrics-viewer/pom.xml test
```

単一テストだけ回す場合:

```bash
mvn -B -f apps/metrics-viewer/pom.xml test -Dtest=RunListPlaywrightTest
```

Playwright テストは Microsoft Edge を起動する（この環境にはインストール済み）。Edge が無い環境では `Assumptions` で skip され、失敗にはならない。

## 実装記録（2026-09-12）

TDD は計画どおり縦スライスで進めた。RED を踏んだのは B1 と B3 で、残りは既に通る契約の固定である。

| サイクル | behavior | 結果 |
|---|---|---|
| 1 | B1 ボタン＝最適配色（tracer bullet） | RED（`#chk-auto-recolor` 不在でタイムアウト）→ GREEN |
| 2 | B2 冪等 | 追加のみで GREEN |
| 3 | B3 Auto の維持優先パス | RED（`run_14` が `#B83280` のまま）→ GREEN |
| 4 | B4 追加安定・解除安定 | 追加のみで GREEN |
| 5-8 | B5 永続 / B6 未選択不変 / B7 ラウンド / B8 1 本 no-op | 追加のみで GREEN |
| 9 | B9 ボタンの結果が次の Auto パスで維持される | 追加のみで GREEN |

計画からの差分:

- **bind 位置**: `btn-recolor-runs` と `chk-auto-recolor` は `bindStaticControls` で 1 回だけ bind する。
  当初は近傍の `Select All` / `Latest Only` に合わせて `bindRunListEvents` へ置いたが、Run list が一度も描画されない
  ERROR mode でボタンが死ぬため、GREEN 後の refactor で移した。
- **B9 を追加**: ボタンと Auto の 2 契約が同居したときの相互作用（最適配色が次の維持優先パスで維持される）は
  計画の B1〜B8 では覆えていなかったため足した。
- **テスト手順**: 初期選択は最新 Run なので、`selectSingleRun` の 1 クリック目では最新 Run が同席したまま
  Auto パスが走る。`Auto` の behavior を見るテストでは、先に最新 Run を外して空選択から選び直す。
  これは仕様どおりの挙動（選択集合が変わるたびに走る）であり、実装側は変更していない。

## 前提

- `localStorage` の値が `"true"` / `"false"` 以外でも既定 ON として扱う（`!== "false"` 判定）。既定 ON なので、既存トグルの `=== "true"` とは向きが逆になる。
- `Auto` の状態は workspace をまたいで共有する（既存の `graphScrollLockEnabled` と同じ扱い）。
- 色は保存しない。reload と workspace 切替の後は基本色から始まり、`Auto` が ON なら次の選択で分離し直される。
- `renderRunList` の引数（`runs`、`selectedRunIds`、`runColorMap`）は変えない。色を配る責務だけを app 側へ移す。
- 既存の `MetricsPlotPlaywrightTest.paletteColorsRenderForRunChipsAndInitialTrace` は 11 Run すべての chip 色が相異なることを検査している。初回選択は 1 本で、維持優先パスは 1 本目を必ず維持するため、この検査は `Auto` 既定 ON でも成立する。
- Java 側へ OKLab 距離のヘルパは足さない。期待値は fixture から決まる hex で直接固定する。
- `Recolor` と `Auto` の UI 文言・tooltip は英語のまま（既存 UI に合わせる）。
