# 042: MetricsViewer 構造リファクタリング T2 実装計画

- 対象PRD: `docs/memo/042_metricsviewer_refactor_10prd.md`
- 対象タスク: T2 `IngestState / SeriesAvailability の enum 化`
- 状態: 実装・検証済み（2026-08-01）
- 目的: Java内部の状態語彙と判定をenumへ集約し、DB・HTTP JSONの外部表現と状態遷移を変更せずに可読性と保守性を改善する。
- 本書は、承認済み計画と実際の実装差分を同期した記録である。T1・T3以降は対象外。

## 1. 実装方針

### 1.1 IngestStateの導入

`MetricsCacheDatabase`内へpublic nested enumとして次を追加する。

```java
IngestState {
    PENDING,
    CONVERTING,
    READY,
    ERROR
}
```

- 各値は従来のDB・HTTP JSON表現に対応する小文字名を保持し、`externalName()`で返す。
- 永続・wire表現を`Enum.name()`、`toString()`、呼び出し側の小文字変換へ依存させない。
- `fromDb(String)`はnull・未知値を厳格に拒否する。
- DB妥当性判定では非例外の`isValidDbValue(String)`を使用する。
- `isStillIngesting()`は`PENDING`または`CONVERTING`を返す。
- 使用予定のない`isTerminal()`は追加しない。

### 1.2 取り込み内部interfaceのenum化

次の内部stateを`String`から`IngestState`へ変更する。

- `MetricsCacheDatabase.CachePreparation.state`
- `MetricsCacheDatabase.CacheMetadata.state`
- `MetricsIngestor.IngestOutcome.state`

取り込み処理の代入・比較はenumへ統一し、DBの`source_meta.state`へ書き込む地点だけ`externalName()`へ変換する。

状態の種類、遷移条件、gzip/rawの判定、エラー分類は変更しない。

### 1.3 SeriesAvailabilityの導入

`MetricsRepository.java`内へpackage-private top-level enumとして次を追加する。

```java
SeriesAvailability {
    OK,
    PENDING,
    NOT_FOUND,
    EMPTY
}
```

- `SeriesPlan.availability`を`String`から`SeriesAvailability`へ変更する。
- 系列検査、点数予算配分、結果構築、projection失敗時fallbackの代入・比較をenumへ統一する。
- `MetricsSeriesResult.availability`は`String`のまま維持し、レスポンス構築時だけ`externalName()`へ変換する。
- 旧`isConverting(String)`を削除し、`IngestState.isStillIngesting()`へ置換する。

### 1.4 外部変換seam

enumを外部へ直接公開しない。文字列への変換は次へ限定する。

- `source_meta.state`へのDB書き込み
- runs.json用`IngestInfo.state`の構築
- metrics.json用`MetricsSeriesResult.availability`の構築

以下の外部表現は不変とする。

- ingest state: `pending / converting / ready / error`
- series availability: `ok / pending / not_found / empty`

## 2. 不正DB stateの扱い

- null・未知のDB stateを`PENDING`へ丸めない。
- DB妥当性検査で不正値として検出する。
- 従来の無効DB処理へ進み、キャッシュを全再構築する。
- 再構築理由は`state_invalid`となり、generationが変更されることを統合試験で固定する。

## 3. テスト計画

### 3.1 enum単体契約

- 全enum値と小文字`externalName()`の対応を確認する。
- `fromDb()`が正常値を復元することを確認する。
- null・未知値について`isValidDbValue()`がfalse、`fromDb()`が例外となることを確認する。
- `isStillIngesting()`が`PENDING / CONVERTING`だけtrueとなることを確認する。

### 3.2 DB・取り込み統合試験

- Java内部stateを扱うassertionとscheduler mockをenumへ追従する。
- DBに未知stateを設定し、`state_invalid`による再構築とgeneration変更を確認する。
- DBへ直接格納されるstateの小文字文字列assertionは維持する。
- raw/gzip、正常終了、変換継続、Run errorの既存状態遷移を維持する。

### 3.3 API統合試験

- runs.jsonの`pending / converting / ready / error`が小文字のままであることを確認する。
- metrics.jsonの`ok / pending / not_found / empty`が小文字のままであることを確認する。
- `converting`を安定して確認できる専用Run fixtureを追加し、background schedulerに依存しない外部契約試験とする。

### 3.4 検証コマンド

```text
mvn -B test
git diff --check
```

実施結果:

- `mvn -B test`: 69 tests、Failures 0、Errors 0、Skipped 0
- `git diff --check`: エラーなし

## 4. 修正ファイル一覧

### 4.1 要件文書

- `docs/memo/042_metricsviewer_refactor_10prd.md`
  - T2のenum配置、外部変換seam、不正DB値、対象外、テスト契約を確定。

### 4.2 Production

- `viewers/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabase.java`
  - `IngestState`、厳格なDB変換、内部recordのenum化、不正state判定を実装。
- `viewers/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestor.java`
  - `IngestOutcome`と取り込み状態の代入・比較・DB書き込み境界をenumへ変更。
- `viewers/metrics-viewer/src/main/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsRepository.java`
  - `SeriesAvailability`、`SeriesPlan`、取り込み状態解釈、DB/JSON変換境界をenumへ変更。

### 4.3 Tests

- `viewers/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/infra/MetricsCacheDatabaseIntegrationTest.java`
  - enum契約と不正DB state再構築・generation変更試験を追加。
- `viewers/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/IngestSchedulerTest.java`
  - `IngestOutcome` mockをenumへ追従。
- `viewers/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsIngestorIntegrationTest.java`
  - Java内部state assertionをenumへ追従し、DB文字列assertionは維持。
- `viewers/metrics-viewer/src/test/java/io/github/kazukin123/anetlab/metricsviewer/service/MetricsApiIntegrationTest.java`
  - 内部fixtureをenumへ追従し、runs.jsonの`converting`を含む外部小文字契約を固定。

## 5. 対象外

- T1・T3〜T6と付録Aの実装。
- 状態の追加・削除、状態遷移規則の再設計。
- DBスキーマ、DB値、HTTP JSON形式の変更。
- `IngestInfo.state`または`MetricsSeriesResult.availability`へのenum直接公開。
- Tag statusの`error`、error code/message、SQL制約のenum化。
- ADRや恒久設計資料の更新。
- 既知バグおよび無関係な未コミット変更の修正。

## 6. 作業上の注意

- 本実装は未コミット。ユーザーの明示指示なしにcommit/pushしない。
- 作業ツリーにはPlaywrightテスト分割やRunner側変更など別作業の差分があるため、T2差分と混在させず保護する。
- 後続タスクでは本書をT2の完了記録として扱い、T2を再実装しない。
