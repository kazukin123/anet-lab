# PRD 079: .cpp の内部リンケージと namespace インデント規約

> 起点: 2026-09-20。Codex が .cpp のヘルパを無名 namespace へ入れ続け、都度の指摘で戻す運用が続いていた。
> 調査すると、抑止の根拠が AGENTS.md のどこにも書かれておらず（書いてあるのは「ファイル全体を囲まない」だけ）、
> 前提としている Google C++ Style Guide は無名 namespace を許容し、既存コードも 39 ブロックが採用済みで、
> **明文・前提スタイル・既存実例の 3 つが揃って逆を向いていた**。Codex は規約に従っていた。
> 本 PRD は規約そのものを確定し、既存 39 ブロックの一括是正と AGENTS.md 追記までを範囲とする。
> 関連: `AGENTS.md`「コーディング規約」（L205-206）、`third_party/catch2/include/catch.hpp`、
> `docs/memo/906_cicd_improvement_10prd.md`

## Context（背景・目的）

### 抑止のレバーがズレていた

要件は「無名 namespace が嫌い」ではない。**インデントが実際の構造を表し、かつディスプレイ幅を使い切れること**である。
現行 .cpp の `using namespace` 方針（AGENTS.md L206）は、階層そのものを消すことでこの要件を満たしている。

ところが実測すると、既存の無名 namespace は**ほぼ全部が本体非インデント**だった。

| 指標 | 値 |
|---|---|
| 無名 namespace ブロック | 39（37 ファイル） |
| 本体インデント有り | 2 |
| 本体インデント無し | **37** |

つまり無名 namespace は現状すでに幅を 1 文字も食っていない。**抑止しても幅は増えない。**
失われていたのは幅ではなく「インデント == 構造」という不変条件のほうだった。
同時に .hpp 側は名前付き namespace をインデントしているため、**hpp と cpp で不変条件が食い違っている**。

### 実害はブロック長

| 行数 | 位置 | 内包 `TEST_CASE` | 型 |
|---|---|---|---|
| 1465 | `core/anet-core/src/nn_test.cpp:31-1496` | 17 | 有 |
| 1444 | `core/anet-core/src/dqn_based_agent_test.cpp:40-1484` | 0 | 有 |
| 1006 | `core/anet-core/src/replay_buffer_test.cpp:68-1074` | 9 | 有 |
| 680 | `core/anet-core/src/image_cls_agent_test.cpp:23-703` | 0 | 有 |
| 680 | `core/envs/dropmerge1/src/DropMergeView.cpp:15-695` | 0 | 無 |

非インデントで 1465 行ある。いま読んでいる関数が内部リンケージかどうかは、スクロールしないと判定できない。
`static` なら宣言 1 行で完結する。これが無名 namespace の実害であり、幅でもインデントでもなく **locality** である。

### Google C++ Style Guide との関係

- Google は「namespace の中身をインデントしない」と明文で決めている。幅問題を構造ごと回避する解。
- 内部リンケージの手段として、無名 namespace と `static` を**並列に許容**している（どちらでもよい）。
- 一方で using-directive（`using namespace foo;`）は .cc を含めて**全面禁止**している。

したがって AGENTS.md L206 はもともと Google とは別解を採っている箇所である。
Google 解は「namespace を書いてインデントしない」、anet 解は「階層そのものを消す」。目的は同じ幅の確保。
本 PRD は anet 解を全域へ一貫させるものであり、**新しい逸脱を増やさない**。

### Catch2 の `TEST_CASE` は既に `static`

```cpp
// third_party/catch2/include/catch.hpp:6181-6188
#define INTERNAL_CATCH_TESTCASE2( TestName, ... ) \
    static void TestName(); \
    CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
    CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
    CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
    namespace{ const Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( ... ); } \
    CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
    static void TestName()
```

`INTERNAL_CATCH_UNIQUE_NAME` は `__COUNTER__` / `__LINE__` 由来（同 L704-706）。
内部リンケージも名前の一意性もマクロ側で完結しているため、
**`TEST_CASE` を無名 namespace で囲んでも何も買っていない**。登録子は自前で無名 namespace を持っている。

### 現状の分布

| 区分 | ブロック | ファイル |
|---|---|---|
| production | 16 | 14 |
| test | 23 | 23 |
| 型を含む | 19 | — |
| 型を含まない | **20** | — |
| `TEST_CASE` を内包 | 3 | — |

**39 ブロックのうち 20 は型を 1 つも含まない**。これらは新規約ではブロックごと消える。

## ゴール / 非ゴール

**ゴール**

- **G1**: 「ブロックを書いたら中身を必ずインデントする。インデントしたくないなら階層を消す」が
  .hpp / .cpp の全域で成立する（hpp と cpp の非対称が消える）。
- **G2**: 内部リンケージの手段が対象の種別ごとに一意に決まる（1 関心 1 機構）。
- **G3**: 既存 39 ブロックが新規約へ移行済みで、Codex が周辺コードを見て再発しない。
- **G4**: 規約と根拠が AGENTS.md に書かれ、「書いていないから守られない」状態が解消する。

**非ゴール**

- **NG1**: .hpp 側の変更。名前付き namespace をブロックで書きインデントする現行のままでよい。
  既に不変条件を満たしている。
- **NG2**: `using namespace` 方針（AGENTS.md L206）の見直し。本 PRD はこれを前提として一貫させる側。
- **NG3**: clang-format の導入。リポジトリに設定ファイルが無く、導入すると全ファイル一括整形になる
  （「大規模な整形変更は避ける」に抵触）。
- **NG4**: CI での機械チェック。`windows-ci.yml` は `pull_request.branches` が空で実質発火しないため、
  現時点で CI をゲートにしても意味が無い。`999_cicd_improvement` の範囲。

## 確定事項

- **D1: 判定基準は 1 行。** 「ブロックを書いたら中身を必ずインデントする。インデントしたくないなら階層を消す」。
  無名 namespace 抑止はこの不変条件の**系**であって、それ自体が目的ではない。
- **D2: 無名 namespace は型定義専用にする。** 関数・定数・`TEST_CASE` は入れない。
  型は `static` で内部リンケージを表現できない唯一のケースであり、そこだけがブロックを必要とする。
  型定義だけなら数十行に収まるので、インデントしても幅の実害が無い。
- **D3: 関数は `static`、定数は無修飾。** namespace スコープの `const` / `constexpr` は既定で内部リンケージなので
  `static` すら不要。関数を `static` にするのは locality（宣言 1 行でリンケージが読める）のため。
- **D4: `TEST_CASE` は無名 namespace の外に出す。** マクロが `static` へ展開済みで、囲む意味が無い。
- **D5: .cpp 内の名前付き namespace は維持し、インデントする。** 名前があり参照もできる実在のスコープなので、
  インデントするのが不変条件に合う。AGENTS.md L206 が禁じているのは「ファイル全体を囲む」ことであり、
  部分をまとめる名前付きブロックは対象外。
- **D6: 一括是正で行う。** 触ったファイルから順次ではなく、まとめて直す。挙動変更ゼロで機械的な差分なので、
  分散させるとレビュー単位が細切れになるだけで得が無い。
- **D7: ADR は作らない。** コーディング規約の正本は AGENTS.md。ADR は設計判断の置き場であり、
  規約を二重管理しない。
- **D8: 検出スクリプトを置く。** 39 箇所を直しても、書き手が Codex である限り再発する。
  `tools/check_namespace_style.py` に置き、`anet-prepare-commit` スキルから呼ぶ。CI は D-NG4 のとおり対象外。

## 規約（確定形）

| 書くもの | .cpp での形 | インデント | 根拠 |
|---|---|---|---|
| 定数 | ファイル直下に `constexpr` / `const` | なし | namespace スコープの const は既定で内部リンケージ |
| 関数 | ファイル直下に `static` | なし | 宣言 1 行でリンケージが読める |
| `TEST_CASE` | ファイル直下 | なし | マクロが `static` へ展開済み |
| .cpp ローカル型 | 無名 namespace、**型定義のみ** | **する** | `static` では表現できない唯一のケース |
| .cpp 内サブグループ | 名前付き namespace | **する** | 実在の参照可能スコープ |
| ファイル全体 | `using namespace` で階層を消す | — | AGENTS.md L206（現行どおり） |

## 実装契約

### パターン A: 型を含まないブロック → ブロックごと削除

対象 20 ブロック。[app_util.cpp:36](../../core/anet-core/src/app_util.cpp) は無名 namespace と `static` 関数が
同一ファイルに同居しており、規約が無いことがそのまま表れている。

```cpp
// 変更前（36-76）
namespace {

std::optional<std::filesystem::path> GetEnvironmentPath(const char* name)
{
    ...
}

bool RedirectStandardStreamFallback(FILE* stream, const std::filesystem::path& path)
{
    return std::freopen(path.string().c_str(), "a", stream) != nullptr;
}

} // namespace

constexpr size_t kMaxHistoryEntries = 10;

static std::string Trim(const std::string& value)
{
    ...
}
```

```cpp
// 変更後
static std::optional<std::filesystem::path> GetEnvironmentPath(const char* name)
{
    ...
}

static bool RedirectStandardStreamFallback(FILE* stream, const std::filesystem::path& path)
{
    return std::freopen(path.string().c_str(), "a", stream) != nullptr;
}

constexpr size_t kMaxHistoryEntries = 10;

static std::string Trim(const std::string& value)
{
    ...
}
```

ブロックが消えて 2 つの流儀が 1 つに揃う。

### パターン B: 型を含むブロック → 型だけ残してインデント

対象 19 ブロック。[nn_impl.cpp:1333](../../core/anet-core/src/nn_impl.cpp) は 114 行のうち型が 4 つ、
残りは既に `static` が付いた関数である（無名 namespace の中の `static` は冗長）。

```cpp
// 変更前（1333-1447）
namespace {

struct ConfigProfileMarkerUse {
    size_t instance_seq = 0;
    std::string block_def_name;
    std::string field_key;
};

struct ConfigProfileMarker { ... };
struct ConfigProfileBlockInstance { ... };
struct ConfigProfileExpansionPlan { ... };

static std::optional<std::string> ExtractConfigProfileMarkerGroup(const std::string& value)
{
    if (value.empty() || value[0] != '@') {
        return std::nullopt;
    }
    return value.substr(1);
}

static ConfigProfileExpansionPlan MakeConfigProfileExpansionPlan(
    const NetworkConfig& root_config, const std::string& structure_str)
{
    ...   // 60 行
}

static void ValidateConfigProfileExpansionPlan(...)
{
    ...
}

} // namespace ← 1447 行目
```

```cpp
// 変更後
// この .cpp の外から参照しない型。
namespace {

    struct ConfigProfileMarkerUse {
        size_t instance_seq = 0;
        std::string block_def_name;
        std::string field_key;
    };

    struct ConfigProfileMarker { ... };
    struct ConfigProfileBlockInstance { ... };
    struct ConfigProfileExpansionPlan { ... };

} // namespace

static std::optional<std::string> ExtractConfigProfileMarkerGroup(const std::string& value)
{
    if (value.empty() || value[0] != '@') {
        return std::nullopt;
    }
    return value.substr(1);
}

static ConfigProfileExpansionPlan MakeConfigProfileExpansionPlan(
    const NetworkConfig& root_config, const std::string& structure_str)
{
    ...
}

static void ValidateConfigProfileExpansionPlan(...)
{
    ...
}
```

ブロックが 114 行 → 約 25 行。インデント 1 段がかかるのは struct のメンバ宣言だけなので幅の実害が無い。
関数側は元から `static` と書いてあるため、**差分は「閉じ括弧を上へ移す」と「struct 群を 1 段下げる」だけ**になる。

無名 namespace の型を `static` 関数の引数・戻り値に使うのは問題ない（無名 namespace のメンバは
暗黙の using-directive で外側スコープから見える）。

### パターン C: 定数のみのブロック

[metrics_logger.cpp:23](../../core/anet-core/src/metrics_logger.cpp) の冒頭は 5 つの `constexpr` で始まる。

```cpp
// 変更前
namespace {

constexpr size_t kMaxCapturedStderr = 4096;
constexpr auto kFfmpegStartupCheckDelay = std::chrono::milliseconds(250);
// wx の Windows pipe は非ブロッキングなので、大きい frame は小分けに書き込む。
constexpr size_t kFfmpegWriteChunkBytes = 4 * 1024;
```

```cpp
// 変更後
constexpr size_t kMaxCapturedStderr = 4096;
constexpr auto kFfmpegStartupCheckDelay = std::chrono::milliseconds(250);
// wx の Windows pipe は非ブロッキングなので、大きい frame は小分けに書き込む。
constexpr size_t kFfmpegWriteChunkBytes = 4 * 1024;
```

`static` も付けない。namespace スコープの `constexpr` は既定で内部リンケージである。

### パターン D: `TEST_CASE` を内包するブロック

対象 3 ブロック（`config_test.cpp:17-221`、`nn_test.cpp:31-1496`、`replay_buffer_test.cpp:68-1074`）。
残る 20 のテストブロックは `TEST_CASE` が既に外にあるので、パターン A / B で済む。

```cpp
// 変更前（replay_buffer_test.cpp:68-1074）
namespace {

namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;

struct TestBuffer {
    std::shared_ptr<rl::ReplayBuffer> rb;
    rl::ReplayBufferConfig config;
    int64_t num_envs;
    int64_t capacity_per_env;
};

float StateValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>(env_idx * 100 + time_idx);
}

...   // 約 700 行

TEST_CASE("SumTree supports explicit floating-point value types", "[replay_buffer][per][precision]")
{
    ...
}

} // namespace ← 1074 行目。ここまで来ないと閉じが見えない
```

```cpp
// 変更後
namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;

// この .cpp 専用のテスト用型。
namespace {

    struct TestBuffer {
        std::shared_ptr<rl::ReplayBuffer> rb;
        rl::ReplayBufferConfig config;
        int64_t num_envs;
        int64_t capacity_per_env;
    };

} // namespace

static float StateValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>(env_idx * 100 + time_idx);
}

TEST_CASE("SumTree supports explicit floating-point value types", "[replay_buffer][per][precision]")
{
    ...
}
```

**`TEST_CASE` 本文のインデントは 1 文字も変わらない。** 1006 行のブロックが型定義だけの十数行になり、
閉じ括弧が画面内に収まる。

### パターン E: .cpp 内の名前付き namespace

[replay_buffer_test.cpp:3784](../../core/anet-core/src/replay_buffer_test.cpp) の
`replay_integrity_assay`（3784-4088）は非インデント。名前付き = 実在スコープなのでインデントする。

```cpp
// 変更前
namespace replay_integrity_assay {

constexpr int64_t kFrameBytes = 6;
constexpr uint64_t kFixedSeed = 20260919;

struct Transition {
    int64_t logical;
    int first;
    int last;
    bool done_episode;
};
```

```cpp
// 変更後
namespace replay_integrity_assay {

    constexpr int64_t kFrameBytes = 6;
    constexpr uint64_t kFixedSeed = 20260919;

    struct Transition {
        int64_t logical;
        int first;
        int last;
        bool done_episode;
    };
```

約 300 行に +4 桁を払うが、払う相手が実在のスコープなので不変条件は保たれる。
このグルーピングは意味を持っているので維持する（無名 + 名前 prefix へ倒さない）。

### 前方宣言の注意

相互再帰・前方宣言を持つヘルパは、**宣言側にも `static` が要る**。片方だけだとリンケージが食い違う。

```cpp
static void ApplyB(const Node& node);            // ← 宣言にも static

static void ApplyA(const Node& node)
{
    ApplyB(node);
}

static void ApplyB(const Node& node)
{
    ...
}
```

### 移行対象一覧（production 16 ブロック / 14 ファイル）

| 行数 | 位置 | 型 | パターン |
|---|---|---|---|
| 42 | `apps/runner/src/HeatMapPanel.cpp:13-55` | 無 | A |
| 40 | `core/anet-core/src/app_util.cpp:36-76` | 無 | A |
| 212 | `core/anet-core/src/config.cpp:31-243` | 無 | A |
| 48 | `core/anet-core/src/dqn_based_agent.cpp:334-382` | 有 | B |
| 239 | `core/anet-core/src/dqn_based_agent.cpp:642-881` | 有 | B |
| 68 | `core/anet-core/src/init.cpp:17-85` | 無 | A |
| 45 | `core/anet-core/src/metrics_logger.cpp:23-68` | 無 | A + C |
| 33 | `core/anet-core/src/metrics_logger.cpp:113-146` | 無 | A |
| 10 | `core/anet-core/src/muzero_proto_agent.cpp:12-22` | 無 | A |
| 114 | `core/anet-core/src/nn_impl.cpp:1333-1447` | 有 | B |
| 52 | `core/anet-core/src/profile.cpp:13-65` | 無 | A |
| 225 | `core/anet-core/src/replay_buffer_impl.cpp:617-842` | 有 | B（既にインデント済み） |
| 680 | `core/envs/dropmerge1/src/DropMergeView.cpp:15-695` | 無 | A |
| 131 | `core/envs/imagecls1/src/ImageData.cpp:23-154` | 無 | A |
| 57 | `core/envs/imagecls1/src/ImageClsView.cpp:12-69` | 無 | A（既にインデント済み） |
| 7 | `core/envs/imagecls1/src/ImageClsEnv.cpp:9-16` | 無 | A |

test は 23 ブロック / 23 ファイル。うち `TEST_CASE` を内包する 3 つがパターン D、残りは A / B。

### `tools/check_namespace_style.py`

`.cpp` を走査し、次の 2 つを違反として報告する。

1. 本体が非インデントの namespace ブロック（無名・名前付きを問わない）。
2. 型定義以外（関数・`TEST_CASE`・定数）を含む無名 namespace。

`.hpp` は対象外（NG1）。`third_party/` は除外。違反があれば非ゼロ終了。
`anet-prepare-commit` スキルから呼び、コミット前に検出する。

### AGENTS.md 追記案

「コーディング規約」の `using namespace` 2 行の直後へ入れる。

```markdown
- namespace ブロックを書いたら中身を必ずインデントする。インデントしたくない場合は階層そのものを消す
  (`.cpp` 全体の `using namespace` がこれに当たる)。Google C++ Style Guide は namespace の中身を
  インデントしないが、ANET は「インデント == 構造」を優先して逆の解を採る。
- `.cpp` ローカルの内部リンケージは、対象の種別ごとに手段を固定する。
  - 定数: ファイル直下の `constexpr` / `const`。namespace スコープでは既定が内部リンケージなので
    `static` も無名 namespace も要らない。
  - 関数: ファイル直下の `static`。無名 namespace には入れない(宣言 1 行でリンケージが読めるため)。
  - `TEST_CASE`: ファイル直下。Catch2 のマクロが `static` 関数へ展開するので囲む必要が無い。
  - 型(`struct` / `class` / `enum`): 無名 namespace に入れる。`static` では表現できないのはこれだけ。
    ブロックには型定義だけを置き、インデントする。
  - `.cpp` 内で意味のあるサブグループを作る場合は名前付き namespace を使い、インデントする。
    AGENTS.md が禁じているのは「ファイル全体を囲む」ことであり、部分をまとめるブロックは対象外。
```

## テスト

挙動変更ゼロのリファクタなので、受入の中核は「差分が機械的であること」と「検出が 0 件になること」。

- **T1**: `tools/check_namespace_style.py` が違反 0 件で終了する。
- **T2**: `anet-core-test` 全緑（既知の既存失敗を除く）。
- **T3**: Debug / RelWithDebInfo / Release の 3 構成でビルドが通る。
- **T4**: 検出スクリプト自体のテスト。非インデントブロック、関数入り無名 namespace、
  `TEST_CASE` 入り無名 namespace のそれぞれを検出し、規約適合コードを誤検出しない。
- **T5**: 差分レビュー。`static` 付与・ブロック境界移動・インデント以外の変更が含まれないこと
  （ロジック・シグネチャ・可視性の変更ゼロ）。

## Complexity audit

| # | 機構 | 裁定 | 理由 / 切ったら戻る痛み |
|---|---|---|---|
| 1 | 無名 namespace を型専用に残す | keep | 型の内部リンケージは `static` で表現できない。言語側の穴なのでブロックが最小解 |
| 2 | 無名 namespace 全廃（型も名前付き ns へ） | cut | ファイル固有名を人が毎回考える必要が出る。無名のほうが最小 |
| 3 | 無名 namespace 全面許可（現状追認） | cut | 1465 行ブロックの locality 問題が残り、hpp との非対称も残る |
| 4 | 型をグローバル直下へ裸で置く | cut | 別 TU に同名異定義があると静かに UB。ODR を naming discipline に賭けない |
| 5 | clang-format 導入で機械化 | cut（NG3） | 設定ファイルが無く、導入＝全ファイル一括整形になる |
| 6 | CI ゲート | defer（NG4） | `windows-ci.yml` が実質発火しない。`999_cicd_improvement` へ |
| 7 | 検出スクリプト + anet-prepare-commit | keep | 39 箇所を直しても、書き手が Codex である限り再発する |
| 8 | 名前付き namespace のインデント | keep | 実在スコープ。ここを非インデントにすると不変条件が壊れる |
| 9 | .hpp の変更 | cut（NG1） | 既にインデント済みで不変条件を満たしている |

## 受入基準

1. `tools/check_namespace_style.py` が違反 0 件。
2. 無名 namespace が型定義のみを含み、本体がインデントされている。
3. `anet-core-test` 全緑（既知の既存失敗を除く）。
4. Debug / RelWithDebInfo / Release の 3 構成でビルドが通る。
5. 差分に `static` 付与・ブロック境界移動・インデント以外の変更が無い。
6. AGENTS.md の「コーディング規約」に新規約と Google 逸脱の理由が入っている。

## 影響・移行

- **差分規模**: 39 ブロック / 37 ファイル。うち 20 ブロックは削除のみ（型を含まない）。
  型を含む 19 ブロックは「関数を外へ出してブロックを縮める + 型定義を 1 段インデント」。
- **コミット分割**: production（16 ブロック / 14 ファイル）と test（23 ブロック / 23 ファイル）で 2 本。
  AGENTS.md と検出スクリプトは production 側に同梱する。
- **実行時挙動は不変**。リンケージは全ケースで内部のまま変わらない。シンボル名は
  `` `anonymous namespace'::Foo `` から `Foo` へ変わるが、スタックトレース・プロファイル表示が
  読みやすくなる方向の変化。
- **副次効果（即座には効かない）**: `static` 関数は MSVC の C4505（未参照のローカル関数）対象になるため、
  `/W4` を入れれば死にヘルパを検出できる。現状は `/W4` も `/WX` も設定していないので、
  この PRD だけでは警告は増えない。
- **前方宣言**: 相互再帰ヘルパは宣言側にも `static` が要る。付け忘れるとリンケージ不一致になる。
- **AGENTS.md の更新はコード変更と同じコミットに載せる。** 規約だけ先に書いて実例が逆のままだと、
  本 PRD の起点になった状態（明文と実例が食い違う）を再生産する。
