# PRD 074: 選択中 Run の色分離（Metrics Viewer）

> 起点: 2026-09-12。workspace に Run が溜まった状態で昔の Run と現在の Run を比較すると、選択中 Run へ似た色が割り当たり、グラフ上で判別できないという報告。
> 仕様合意: 2026-09-12（グリルで決定 D1〜D11 を確定。理由と棄却案は [ADR 0041](../../adr/0041-run-color-is-comparison-role-not-run-identity.md)）。
> 対象: Metrics Viewer の browser 側のみ。server、Metrics キャッシュ、HTTP API は変更しない。
> 関連: [Metrics Viewer 設計](../../design/210_metrics_viewer.jp.md)、[Run 分析ユーザーガイド](../../design/030_user_guide_analysis.jp.md)、[done/041 SQLite キャッシュ](041_metrics_sqlite_cache_10prd.md)、[done/042 Viewer refactor](042_metricsviewer_refactor_10prd.md)。

## 1. 背景

### 1.1 現行の色割り当て（実装事実、確認日 2026-09-12）

| 事実 | 根拠 |
|---|---|
| Run 色は 20 色固定の `RUN_COLORS` から取る | [metrics-viewer.js:34](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js) |
| 色は Run list 描画時に **workspace 内の全 Run** へ runId 昇順で先着順に配り、index は `runColorMap.size % 20` で決まる | 同 `UIController.renderRunList`（1084-1092 行） |
| 選択されているかどうかは色の決定に影響しない | 同上 |
| graph の line 色と band の塗りは `runColorMap` を引く。trace 名は runId で、凡例には runId が出る | 同 `PlotlyController`（388、420-449 行） |
| 消えた Run の色は削除し、workspace 切替では全消去する | 同 `refreshMetadata`（1561-1563 行）、`_resetWorkspaceState` |
| Run 色は `localStorage` へ保存しない（保存 state は 8 件） | [210 §7.4](../../design/210_metrics_viewer.jp.md) |
| `selectedRuns` は選択順の配列で、後から選んだ Run が末尾に付く。一度外して選び直すと末尾へ移動する | 同 `bindRunListEvents`（1182-1185 行）、`setSelectedRuns`（1601 行） |
| `Select All` と初回選択は server 列挙順 = runId 昇順 | [RunScanner.java:45](../../../apps/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/RunScanner.java) の `.sorted()` |
| Run list の選択中の行は `.run-row.active` で行の背景自体が変わる | [metrics-viewer.css](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) |

要点は **色が「workspace 内での位置」で決まり、「今の比較の構成」では決まらない**ことである。
比較したい 2 本が palette 上で隣り合っていれば似た色になり、20 離れていれば完全に同じ色になる。

### 1.2 何が起きるか（定量）

色の近さは OKLab 空間のユークリッド距離 ΔE で測る（sRGB → 線形 RGB → OKLab）。
現行 palette 自体が近接ペアを含む。

| ΔE | 組 |
|---:|---|
| 0.0385 | `#2FBF71` / `#00B36B`（緑） |
| 0.0643 | `#C85A17` / `#A65A2E`（茶） |
| 0.0688 | `#F2C230` / `#D1D83B`（黄） |
| 0.0828 | `#F2C230` / `#FF9F1C`（黄 / 橙） |
| 0.0945 | `#2F7DE1` / `#4656D9`（青） |

workspace の Run 数と選択本数に対し、選択集合の中に ΔE < 0.12 のペアが含まれる確率は次のとおり（現行規則、全組合せまたは 20 万標本）。

| workspace の Run 数 | 選択 2 本 | 3 本 | 4 本 | 6 本 |
|---:|---:|---:|---:|---:|
| 20 | 11.6% | 32.3% | 56.2% | 90.5% |
| 30 | 13.6%（同色 2.3%） | 37.0%（6.9%） | 62.5%（13.6%） | 93.8%（32.2%） |
| 60 | 14.6%（3.4%） | 39.1%（10.1%） | 64.9%（19.6%） | 94.7%（43.3%） |

括弧内は**完全に同じ色**のペアを含む確率で、Run 数が palette の 20 を超えた分の剰余から生じる。
30 Run の workspace で 4 本比較すると、6 割以上の組合せで見分けにくいペアが混ざり、1 割超で同色が出る。

### 1.3 ゴール

- 今選択している Run 同士が判別できる色に、ボタン 1 つで必ずできる。
- 選択を変えるたびに自動で同じ状態を保てる。
- 自動実行しても、見ている最中のグラフの色が理由なく入れ替わらない。

### 1.4 非ゴール

palette 自体の設計変更と拡張、色覚多様性（CVD）対応、Run ごとの色の手動固定と永続化、未選択 Run を含む全体の色最適化、
線種・marker による区別、server 側および Metrics キャッシュの変更は含めない。

## 2. 用語

用語集へ追加する語は [CONTEXT.md](../../../CONTEXT.md) の「Run色」「分離距離」を正とする。本 PRD 内だけで使う語は次の 3 つ。

| 用語 | 定義 |
|---|---|
| 基本色 | Run list 描画時に全 Run へ先着順で配る現行の色。分離パスの入力になる |
| 最適配色 | `Recolor` ボタンの処理。現在の色を無視し、palette 先頭を起点に選択中 Run へ最良の組を配る |
| 維持優先パス | `Auto` の処理。既存の色が条件を満たせば維持し、満たさない Run だけを配り直す |

## 3. 仕様

### 3.1 UI

| 要素 | id | 文言 | 配置 | tooltip |
|---|---|---|---|---|
| ボタン | `btn-recolor-runs` | `Recolor` | `#run-section .section-controls`（`Select All` / `Latest Only` の右） | Give the selected runs the most distinguishable colors |
| チェックボックス | `chk-auto-recolor` | `Auto` | 同上（Tags の `Filter` と同じ形式） | Keep the selected runs distinguishable as the selection changes |

- チェックボックスの状態は `localStorage` の `anet.metricsviewer.autoRecolorEnabled` へ保存する。**既定は ON**（保存値が無い場合も ON）。
- 保存は workspace をまたいで共有する（既存の `graphScrollLockEnabled` などと同じ扱い）。
- `Auto` を OFF へ戻しても、それまでに割り当てた色は元へ戻さない。

### 3.2 最適配色（`Recolor` ボタン）

現在の色を一切見ず、選択順に farthest-point で配る。同じ本数なら常に同じ色の組になる。

```text
recolorOptimal(selectedRuns, runColorMap):
  if selectedRuns.length <= 1: return            // 分離すべき相手がいない
  fixed = []; used = {}
  for runId of selectedRuns:                     // 選択順
    if used.size == RUN_COLORS.length: fixed = []; used = {}
    chosen = argmax(minDist(c, fixed) for c in RUN_COLORS \ used)
    runColorMap.set(runId, chosen); fixed.push(chosen); used.add(chosen)
```

`minDist(c, [])` は +∞ とし、argmax の同値は `RUN_COLORS` の並び順で先勝ちとする。
したがって 1 本目は必ず `#2F7DE1` になり、4 本選択なら `#2F7DE1` / `#D1D83B` / `#E23B4F` / `#00B36B` が順に付く。

### 3.3 維持優先パス（`Auto`）

選択集合が変わるたびに走る。既存の色を尊重し、条件を満たさない Run だけを配り直す。

```text
RUN_COLOR_MIN_DISTANCE = 0.16

recolorKeepingExisting(selectedRuns, runColorMap):
  fixed = []; used = {}
  for runId of selectedRuns:                     // 選択順
    if used.size == RUN_COLORS.length: fixed = []; used = {}
    candidates = RUN_COLORS \ used
    achievable = max(minDist(c, fixed) for c in candidates)
    current    = runColorMap.get(runId)
    if current != null and minDist(current, fixed) >= min(RUN_COLOR_MIN_DISTANCE, achievable):
      chosen = current                           // 既存色を維持する
    else:
      chosen = argmax(minDist(c, fixed) for c in candidates)
    runColorMap.set(runId, chosen); fixed.push(chosen); used.add(chosen)
```

`min(RUN_COLOR_MIN_DISTANCE, achievable)` は、しきい値を満たせない本数でも「今の palette で取れる最良」を維持条件にするための項である。
これが無いと、しきい値を満たせない選択サイズで実行のたびに全色が付け替わる。
すでに同じラウンドで使われた色は `minDist = 0` になるため、重複色が維持されることはない。

### 3.4 両方に共通する規則

- 色は `RUN_COLORS` の 20 色からのみ選び、新しい色を生成しない。
- 選択が 20 本を超えたら次のラウンドを始める。ラウンド内では必ず相異なる色になり、21 本目は 1 本目と同じ色から再び始まる。Toast などの通知は出さない。
- **未選択 Run の色は変更しない。** 選択中 Run と chip の色が重複することは許す（選択中かどうかは行の背景で判別する）。
- 未選択を含む全 Run への基本色の先着順割り当ては現行どおり残す。分離はその結果を入力として、選択中 Run の分だけ上書きする。
- 選択 0 件では何もしない。選択 1 件では最適配色も何もしない（維持優先パスは 1 本目を常に維持するので同じく無変化）。

### 3.5 性質と保証

| 性質 | 内容 |
|---|---|
| 冪等 | どちらのパスも、直後にもう一度実行すると色は変わらない |
| 追加安定 | 選択に Run を足したとき、既存選択の色は変わらない。既存は常に前置され、判定は自分より前の色だけに依存するため |
| 解除安定 | 選択から Run を外しても、残った Run の色は変わらない（`fixed` が縮むだけで距離は増える） |
| 決定的 | 同じ入力（選択順と現在色）からは常に同じ結果になる |
| 順序依存 | 同じ選択集合でも選んだ順が違えば、色がどの Run に付くかは変わる（最適配色では使われる色の組は同じ） |

達成できる分離度は palette の大きさで決まる。

| 選択本数 N | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 最適配色（ボタン）の最小 ΔE | 0.412 | 0.327 | 0.216 | 0.198 | 0.172 | 0.148 | 0.144 | 0.120 | 0.116 |
| 維持優先パス（Auto）の最悪値 | 0.161 | 0.161 | 0.161 | 0.161 | 0.125 | 0.115 | 0.114 | 0.108 | 0.103 |

**`Auto` は選択 5 本まで、初期色に関わらず最小 ΔE 0.16 を保証する。**
6 本以上は palette の上限により保証せず、最良努力となる（このとき、しきい値を 0.12 にしても 0.20 にしても結果は同じである）。
より強い分離が必要なら `Auto` ではなくボタンを押す。

検証方法: 初期色の組合せを N ≤ 4 で総当たり（160,000 通り）、N = 5〜10 で各 60,000 標本。冪等はそのすべてで成立。
追加安定は 30,000 件の無作為な追加で成立。25 本選択では先頭 20 本が相異なり、21 本目が `#2F7DE1` へ戻ることを確認した。

### 3.6 `Auto` の発火点

`Auto` が ON のとき、選択集合が変わりうる経路で維持優先パスを実行する。

- Run 行の click / solo click、`Select All`、`Latest Only`（いずれも `setSelectedRuns` を通る）。
- 初回表示の最新 Run 自動選択（1 本なので実質無変化）。
- metadata 更新で選択が縮んだ場合（解除安定により結果は変わらないが、同じ経路を通す）。
- `Auto` を OFF から ON へ切り替えた時点。
- workspace 切替は色ごとリセットされるため、切替後の最初の選択で走る。

`Auto` が OFF のときはボタン押下時のみ実行する。metadata 更新で新しい Run が現れても選択が変わらなければ走らない。

## 4. 実装の当たり

| 変更点 | 内容 |
|---|---|
| seam | `MetricsViewerClientApp._applyRunColors()` を新設し、`refreshLists()` の先頭で呼ぶ。(a) 未割当 Run への基本色付与（`renderRunList` から移設）、(b) `Auto` が ON なら維持優先パス、の 2 段 |
| view | `UIController.renderRunList` は `runColorMap` を読むだけにする（引数は現行のまま） |
| ボタン | `onRecolorRuns()` で最適配色を実行し、`refreshLists()` と `_renderCurrent()` を呼ぶ（`onToggleLog` と同じ再描画経路） |
| checkbox | `onToggleAutoRecolor()` で state 更新と `localStorage` 保存、ON なら即時実行して再描画。`_loadState()` と初期同期へ追加する（保存値が無い場合の既定は ON） |
| 色距離 | `oklabDistance(hexA, hexB)` を module scope の純関数として `colorWithAlpha` の近くへ追加する |
| 共通処理 | 2 つのパスは「候補から argmax を選ぶ」部分を共有する。維持判定の有無だけが違う |
| 定数 | `RUN_COLOR_MIN_DISTANCE = 0.16` と `STORAGE_KEY_AUTO_RECOLOR` を先頭の定数群へ追加する |

変更ファイルは [index.html](../../../apps/metrics-viewer/src/main/resources/static/index.html)、
[metrics-viewer.js](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.js)、
必要なら [metrics-viewer.css](../../../apps/metrics-viewer/src/main/resources/static/metrics-viewer.css) の少量追加に閉じる。

## 5. 受入条件

[RunListPlaywrightTest](../../../apps/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/view/RunListPlaywrightTest.java) へ追加する。
Run metadata は既存テストと同じく route の stub JSON で作る。

1. 基本色が衝突する 4 Run を選び `Recolor` を押すと、chip 色が選択順に `#2F7DE1` / `#D1D83B` / `#E23B4F` / `#00B36B` になる。
2. graph の trace 色が chip 色と一致する。
3. もう一度 `Recolor` を押しても色が変わらない（冪等）。
4. `Auto` を ON にすると、ボタンを押さずに 4 本すべてのペアで ΔE ≥ 0.16 になる。
5. `Auto` ON のまま 5 本目を選んでも先の 4 本の色は変わらない（追加安定）。1 本外しても残りの色は変わらない（解除安定）。
6. `Auto` の状態が `localStorage` へ入り、reload 後も保たれる。`localStorage` が空の初回は ON である。
7. 未選択 Run の chip 色は、ボタン押下と `Auto` の実行前後で変わらない。
8. 25 Run を `Select All` しても例外が出ず、先頭 20 本が相異なり、21 本目が `#2F7DE1` になる。
9. 選択 1 本で `Recolor` を押しても色が変わらない。
10. 既存の Playwright テストと checkstyle が緑のままである。

## 6. ドキュメント更新対象

作成済み（本 PRD と同じ変更に含む）:

- [ADR 0041](../../adr/0041-run-color-is-comparison-role-not-run-identity.md): 決定と棄却案。
- [CONTEXT.md](../../../CONTEXT.md): 「Run色」「分離距離」を Metrics基盤へ追加。

実装と同じ変更で更新する:

- [030 §3 画面の基本操作](../../design/030_user_guide_analysis.jp.md): `Recolor` と `Auto` の行を追加し、`localStorage` へ保持する項目の列挙へ 1 件足す。
- [210 §3.2 browser](../../design/210_metrics_viewer.jp.md): Run 色の所有が `MetricsViewerClientApp` 側であることを反映する。
- [210 §7.4 browser 側の定数と永続 state](../../design/210_metrics_viewer.jp.md): 定数表へ `RUN_COLOR_MIN_DISTANCE`、`localStorage` 表へ `anet.metricsviewer.autoRecolorEnabled` を追加し、「8 件」を「9 件」へ直す。色の決定規則も同節へ短く書く。

## 7. 決定一覧

| # | 決定 | 要点 |
|---|---|---|
| D1 | 色の帰属 | 比較ごとの役割。保存しない。同一性は runId が担う |
| D2 | 走査順 | 選択順。追加安定を優先し、同じ集合でも選んだ順で割り当て先が変わることを受け入れる |
| D3 | 役割分担 | ボタン = 最適配色（現在色を無視）／`Auto` = 維持優先。維持優先は冪等なので、分けないと `Auto` ON 中にボタンが無効化する |
| D4 | 維持しきい値 | ΔE 0.16。走査順が選択順なので上げても既存 Run は動かず、N ≥ 7 の結果は 0.12 と同じ |
| D5 | 最適配色の起点 | palette 先頭 `#2F7DE1` 固定。1 本目の現在色を維持する案より結果が良い（N=6 で 0.172 対 0.144） |
| D6 | 20 本超え | ラウンドを回して続行。Toast なし |
| D7 | 未選択 Run | 触らない。chip の重複を許す |
| D8 | `Auto` 既定 | ON |
| D9 | 用語 | CONTEXT.md へ「Run色」「分離距離」 |
| D10 | ADR | 0041 を 1 本 |
| D11 | 文言 | `Recolor` + `Auto` |

## 8. 残る制約

- **分離できる上限は palette の 20 色**である。21 本目以降は同じ色が再び現れる。上限を上げるには palette の拡張が要り、それは別 PRD とする（近接ペア ΔE 0.0385 の緑 2 色の見直しを含む）。
- **色は保存しない。** reload と workspace 切替の後は基本色から始まる。`Auto` が ON なら次の選択で分離し直され、OFF ならボタンを押すまで基本色のままになる。
- **既定 ON と しきい値 0.16 の組み合わせにより、基本色は選択した瞬間に上書きされるのが常態になる。** 基本色が残るのは未選択行だけである。
- 実験記録へ貼るスクリーンショットで色を揃えたい場合は、選ぶ順を揃えるか、記録側で runId と色を併記する。
