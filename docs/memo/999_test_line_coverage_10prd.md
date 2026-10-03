# 行カバレッジ計測（OpenCppCoverage）暫定 PRD

> 起点: 2026-09-17 の定点統計（anet-stats）設計。AI-Native 化以前のコードのテスト不足を機能カテゴリ別に見たい。テストコード比率（L0）と被テスト率（L1）は git 履歴から取れるが、L1 の参照ヒューリスティックは誤検出と見逃しを両方起こす。裁定者として行カバレッジ（L2）が要る。
> 関連: [999_cicd_improvement_10prd.md](999_cicd_improvement_10prd.md)（CI 側の L2 定点ジョブ）、[AGENTS.md](../../AGENTS.md) のテスト実行規約、[core/anet-core/CMakeLists.txt](../../core/anet-core/CMakeLists.txt)（テスト実行体の定義）。

## Context / Problem Statement

2026-09-17 に C++ ソースを対象に試作した結果（L0 と L1、参照ヒューリスティック）は次のとおり。

| 事実 | 値 |
|---|---|
| C++ 全体の test 行 / prod 行 | 0.49 |
| 被テスト率（ファイル単位、間接参照込み） | 100 / 156 |
| 薄いカテゴリ | テンソル処理 0.07、プロファイリング 0.08、可視化/メトリクス 0.16、MuZero 系列 0 |
| 未テストで大きいファイル | muzero_based_agent 1,156 行、scaler_impl 746 行、muzero_rb 335 行、GridMazeEnv 289 行、CartPoleEnv 246 行 |

L1 には次の限界があり、これを数値で裁定する手段が無い。

1. **誤検出**: 公開クラス名がテストに現れるかで判定するため、汎用的な名前で当たる。GUI 共通基盤と MuZero 系列は test 行が 0 なのに被テスト扱いになった。
2. **見逃し**: 内部クラスは名前がテストに現れない。設定管理の config_impl は config.hpp 経由で実際にはテストされているのに未テスト扱いになった。
3. **「テストがある」と「テストが通る経路」の区別が付かない**: ファイル単位の有無では、大きなファイルの一部だけがテストされている状態を見分けられない。

**したがって欲しいのは**: 機能カテゴリ × 年代（AI-Native 前後）で行カバレッジを出し、テスト負債リストの並びと L1 の誤りを実測で確定する仕組み。

## 効く事実

1. **anet-core は STATIC ライブラリ**で、anet-core-test.exe に静的リンクされる。モジュールフィルタはテスト実行体 1 本でよく、libtorch などの巨大 DLL は自然に計測対象外になる。
2. テスト実行体は `core/anet-core/bin/<Config>/` に出力され、libtorch の DLL が隣にコピー済み。リポジトリルートから実行する規約がある。Env テスト 4 本（Atari、DropMerge、ImageCls、LunarLander）も `add_test` 登録済み。
3. OpenCppCoverage は PDB ベースで MSVC ビルドにそのまま使える。`--sources` と `--modules` と `--excluded_sources` で範囲を絞り、`--export_type cobertura:` で XML を出せる。`--export_type binary:` で出した複数実行体の結果は `--input_coverage` でマージできる。
4. Catch2 のタグは機能単位（`[dqn]` `[config]` `[nn]` `[replay_buffer]` など）で、GPU 依存を示すタグは無い。GPU のある開発機で全件実行する前提なら支障はない。CI 側の扱いは CI PRD の Q2。
5. 機能カテゴリの対応表 `reports/tools/categories.json`（glob から Topic 31x〜32x へ）を anet-stats と共有する。年代は git blame の行日付で判定し、境界日は AGENTS.md 作成日の 2026-05-30 を既定にする。
6. カバレッジは cloc と違い過去 revision に遡れない。計測ごとのスナップショット保存が要る。

## 提案

### 導入

- OpenCppCoverage を開発機にインストールする（winget または GitHub Releases のインストーラ）。リポジトリには持ち込まない。
- 構成は Debug を既定にする。最適化で行対応が崩れる RelWithDebInfo は速度が問題になった時の選択肢（Q1）。

### 実行

```text
OpenCppCoverage --sources core\anet-core --excluded_sources core\anet-core\src\*_test.cpp ^
  --modules anet-core-test.exe --export_type binary:out\cov\core.cov ^
  -- core\anet-core\bin\Debug\anet-core-test.exe
```

Env テストも同じ形で binary に出し、`--input_coverage` でマージして cobertura XML を 1 本にする（Q2）。

### 集計

- `reports/tools/coverage.py` が cobertura XML を読み、`categories.json` でカテゴリ化し、git blame で年代分割し、`reports/stats/coverage/<date>.json` に保存する。
- anet-stats のテスト群 L2 列に載せ、trend では最新スナップショットを表示する。
- 出力は、カテゴリ × 年代の行カバー率、未カバー行の多い prod ファイル上位、テスト負債リストの L2 列。

### 運用

- 手動、または anet-housekeeping からの任意実行。トークンは食わず時間だけ食うので、予算判断は時間側で行う。
- 頻度は月 1 回程度、または大きなテスト追加の後。

## 未決

| # | 論点 | 選択肢 |
|---|---|---|
| Q1 | 計測構成 | Debug / RelWithDebInfo。既定は Debug |
| Q2 | Env テストを含めるか | 含める（View は除外） / anet-core-test のみ |
| Q3 | 所要時間の上限 | 未計測。breakpoint 方式なので数分から数十分の見込み。1 回計測してから決める |
| Q4 | CUDA 依存テスト | 開発機で全件 / タグを新設して除外可能にする |
| Q5 | しきい値 | カテゴリ別の目標値を置く / 傾向だけ見る |
| Q6 | 保存粒度 | 計測ごとに json を残す / 最新のみ |
| Q7 | HTML レポート | OpenCppCoverage の HTML は大きい。残さない / out 配下に一時的に残す。Pages には出さない |

## 受入条件（暫定）

1. anet-core-test を 1 回走らせて、カテゴリ × 年代の行カバー率が json に出る。
2. L1 で誤検出した GUI 共通基盤と MuZero 系列、見逃した config_impl の実態が数値で確定する。
3. テスト負債リストの並びが L2 で更新される。
4. 1 回の所要時間が計測され、Q3 で決めた上限内に収まる。

## スコープ外

- 計測の CI 化。[999_cicd_improvement_10prd.md](999_cicd_improvement_10prd.md) が扱う。
- テストの追加そのもの。anet-audit の test-gap 観点から個別 PRD に切り出す。
- Java と Python のカバレッジ。Viewer の Java は test 行が prod 行を上回っており優先度が低い。
- 分岐カバレッジと関数カバレッジ。行のみ。
