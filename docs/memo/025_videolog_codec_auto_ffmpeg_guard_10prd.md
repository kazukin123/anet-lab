# VideoLogger 堅牢化 — ffmpeg 異常の fail-fast 検出 と コーデック `auto` 選択 PRD

> 関連コード: `core/anet-core/src/metrics_logger.cpp`（`VideoLogger`）, `core/anet-core/include/anet/metrics_logger.hpp`（`VideoLogger` / `MetricsLoggerConfig` / `ExecuteStarter`）, `apps/runner/src/HeatMapPanel.cpp`（`SweepHeatMapPanel::CreateObserver`）, `core/anet-core/src/observers.cpp`（`SweepedHeatMapObserver::OnLearn`）, `core/anet-core/src/image.cpp`（`ImageSource::Render`）。
> 本 PRD はクラッシュ診断（2026-07-01）の成果物。実装は別途（Codex 想定）、本書は self-contained に記述する。

## Context / Problem Statement

`SweepHeatMapPanel` を表示中、学習の途中でアクセス違反（`0xC0000005`, 実行アドレス `0x00000000`）でアプリごとクラッシュする。スタックは `SweepedHeatMapObserver::OnLearn` → `MetricsLogger::Log` → `LogImage_subtyped` → `VideoLogger::WriteFrame`（`stream_->IsOk()`）。

**確定した原因連鎖（実証済み）:**

1. `SweepHeatMapPanel::CreateObserver`（[HeatMapPanel.cpp:390-398](../../apps/runner/src/HeatMapPanel.cpp)）が `grid_width=128, grid_height=128, image_width=-1, image_height=-1` の `SweepedHeatMapObserver` を **パネル表示時に** `Attach`（[:450](../../apps/runner/src/HeatMapPanel.cpp)）。→「最初は動く／パネルを出すと落ちる」の理由。
2. `image_width/height=-1` → `ImageSource::Render(-1,-1)` は `RenderRaw()` を**そのまま返す**（[image.cpp:23](../../core/anet-core/src/image.cpp)）→ 動画寸法 = 生グリッド **128×128**。
3. run のコーデックは `metrics_logger.video_codec = h264_nvenc`。**NVENC は最小解像度制約（H.264 で概ね 145×49、世代/ドライバ依存）があり 128×128 を拒否**。ffmpeg が `return code -22 (Invalid argument)` で即終了し **0 バイト MKV** を残す（`ffmpeg` 単体で再現確認済み。512×512 は成功、128×128+libx264 も成功）。
4. `VideoLogger` は ffmpeg の死を検出できず、`process_` 所有のパイプ書込ストリームへの**生ポインタ `stream_`** が無効化された後も保持し続ける。次の `WriteFrame` が `stream_->IsOk()`（仮想呼び出し）を無効オブジェクトに対して行い **UAF クラッシュ**。`if (!stream_ || !stream_->IsOk()) return;`（[metrics_logger.cpp:128](../../core/anet-core/src/metrics_logger.cpp)）の null チェックも `IsOk()` も「非 null のまま指し先が死んだ」状態は検出できない。

**根本問題は 2 つ:**

- **(A) ffmpeg の異常（起動失敗 / 不意の死）を検出せず動作継続**してしまい、結果として UAF で不定にクラッシュする。異常時は明示的に落とすべき。
- **(B) NVENC が扱えない / 不適切な解像度への方針が無い**。コーデックは固定指定（`h264_nvenc` / `libx264` / passthrough）のみで、サイズ非対応時の分岐が存在しない。

## ゴール / 非ゴール

**ゴール:**

- **G1（=要件①）**: ffmpeg の起動失敗・異常終了を確実に検出し、ffmpeg の stderr を含む明瞭なログとともに `ANET_SYSTEM_ERROR` で**明示的に停止**する。サイレントな継続（現状の暗黙 UAF も、`return` での握り潰しも）を廃止する。
- **G2（=要件②）**: `metrics_logger.video_codec` に `auto` を追加。`auto` は nvenc を優先しつつ、非対応/不適切な場合に libx264 を自動選択する。**明示 `h264_nvenc` 指定** かつ 非対応解像度は構築時チェックで `ANET_SYSTEM_ERROR`（勝手なフォールバックはしない）。

**非ゴール:**

- **NG1: パディングモード（`nvenc_pad` 等）は今回入れない。** 低解像度では libx264 の方が負荷・NVENC セッション資源・互換の全面で適切（§設計判断 D4）であり、nvenc を強制維持する動機が本件に無い。出力寸法の変化・毎フレーム pad の複雑性に見合わない。将来 nvenc 強制の実需要が出たら `auto` の隣に追加可能。
- **NG2: `SweepHeatMapPanel` の 128 ハードコード自体の是正は必須としない。** `auto` 化で自動回避されるため。ただし「パネル動画を実際に出したい」場合の選択肢を §影響・移行 に記す。
- **NG3: 起動時 nvenc 能力プローブ（4:4:4 対応の事前判定等）の静的実装は行わない。** 静的判定は困難なため、`auto` は「起動直後生存チェック（G1 の副産物）」で実行時失敗を捕捉してフォールバックする（§R2）。

## 要件

### R1: ffmpeg 異常の検出と fail-fast（要件①）

対象: `VideoLogger`（[metrics_logger.cpp:64-155](../../core/anet-core/src/metrics_logger.cpp), [metrics_logger.hpp:84-102](../../core/anet-core/include/anet/metrics_logger.hpp)）。

- **R1-1 死亡検出（stream 非依存）**: `process_` に `wxEVT_END_PROCESS` ハンドラを張り、ffmpeg 終了時に **`write_mutex_` 下で** ① 死亡フラグ `ffmpeg_dead_` を立て ② `exit_code_` を記録 ③ `process_->GetErrorStream()` から ffmpeg の stderr を drain して `captured_stderr_` に保持 ④ `stream_ = nullptr` に無効化。以降 `WriteFrame` は死んだ生ポインタを触らない。
- **R1-2 起動失敗検出**: コンストラクタで launch 後、**起動直後生存チェック**（下記 R2 と共有）を行い、ffmpeg が即死していれば stderr を回収して扱う（`auto` はフォールバック、明示指定は fatal）。
- **R1-3 実行時 fail-fast**: `WriteFrame` は `write_mutex_` 取得後、**stream へ触れる前に** `ffmpeg_dead_ || stream_==nullptr` を判定し、真なら現状の `return` をやめて `ANET_SYSTEM_ERROR`（cmd / path / `exit_code_` / `captured_stderr_` を含める）。書込中に失敗（`!IsOk()`）した場合も `return` でなく `ANET_SYSTEM_ERROR`。
- **R1-4 排他の是正**: `Close()`（[metrics_logger.cpp:144](../../core/anet-core/src/metrics_logger.cpp)）に `write_mutex_` を追加（コミット `f8008e3` で `WriteFrame` にのみ施錠し `Close()` を無施錠のまま残した非対称の修正）。`stream_` 無効化・`process_` 破棄は必ず `write_mutex_` 下で行う。
- **R1-5 診断性**: エラーメッセージには最低限 `path_`・実行 cmd・exit code・ffmpeg stderr 抜粋を含める。現状 `Redirect()` で stderr はパイプ化されるが未読（[metrics_logger.cpp:100](../../core/anet-core/src/metrics_logger.cpp)）。異常時のみ読めばよい。

### R2: コーデック `auto` と解像度チェック（要件②）

対象: `MetricsLoggerConfig::video_codec`（[metrics_logger.hpp:111](../../core/anet-core/include/anet/metrics_logger.hpp)）と `VideoLogger` のコーデック分岐（[metrics_logger.cpp:75-87](../../core/anet-core/src/metrics_logger.cpp)）。

- **R2-1 値の追加**: `video_codec` は `auto` | `h264_nvenc` | `libx264` | その他（従来通り passthrough）を取る。**推奨デフォルトを `auto` に変更**（[metrics_logger.hpp:111](../../core/anet-core/include/anet/metrics_logger.hpp) の `"libx264"` → `"auto"`。§影響・移行 の判断ポイント）。
- **R2-2 nvenc 適格性判定** `IsNvencEligible(w,h) -> bool`:
  - 偶数（`(w&1)==0 && (h&1)==0`）かつ `w >= kNvencMinWidth` かつ `h >= kNvencMinHeight`。
  - 既定閾値 `kNvencMinWidth=160`, `kNvencMinHeight=64`（NVENC H.264 documented ~145×49 に安全マージン。実測: 128×128 失敗 / 512×512 成功）。**定数化**して将来のドライバ差異に対応可能に。
- **R2-3 コーデック決定（構築時）**:
  - `requested == "auto"` → 適格なら nvenc、不適格なら libx264 を選択（選択理由を `LOG::info`）。
  - `requested == "h264_nvenc"` → **不適格なら `ANET_SYSTEM_ERROR`**（「nvenc は WxH 非対応（最小 160×64・偶数）。`auto` か `libx264` を使用」）。適格なら nvenc。**勝手にフォールバックしない。**
  - それ以外 → そのまま採用（passthrough、従来挙動）。
- **R2-4 `auto` の実行時フォールバック**: `auto` で nvenc を選び起動直後生存チェック（R1-2）で即死した場合（例: GPU が 4:4:4 nvenc 非対応）、`LOG::warn` の上で **libx264 で再起動**。それも即死したら `ANET_SYSTEM_ERROR`。明示 `h264_nvenc` / `libx264` は再起動せず即死は fatal（R1）。

## 設計判断（why。self-contained のため明記）

1. **異常は継続でなく fail-fast（要件①の核）**: 動画メトリクスは学習の付随出力だが、ffmpeg 破綻を握り潰して進めると、今回のように後段で不定な UAF となり原因追跡が困難。破綻点で cmd/exit/stderr 付きで即停止する方が運用・デバッグ双方で安全。「無理やり継続させない」という明示要件に従う。
2. **検出は生 `stream_` に依存しない**: UAF の主因は「無効化された `stream_` を触ること」。よって死亡判定は `wxEVT_END_PROCESS`（プロセス由来・stream 非依存）で行い、`WriteFrame` は stream へ触れる**前に**フラグで門番する。`write_mutex_` を `WriteFrame`/`Close`/終了ハンドラで共有し、`stream_` の無効化を 1 つのロック下に集約。
3. **`auto` は静的サイズ判定＋起動時生存チェックの二段**: 解像度は静的に判定できるが、4:4:4 対応等の実行時要因は静的判定困難。前者は `IsNvencEligible` で、後者は起動直後生存チェック（R1 の副産物）でフォールバック。「対応不可 or 不適切なら libx264」という要件を静的・動的両面で満たす。
4. **低解像度は libx264 が正しい選択（Q1 への回答）**: NVENC は固定機能 HW で、真価は高解像度/高 fps のエンコード CPU オフロード。小フレーム（128×128≈49KB）を疎・15fps で出す用途では両者とも負荷些少で、むしろ nvenc は (a) 同時セッション数制限（民生 GPU で歴史的に 3〜5、多数の heatmap 動画で枯渇し得る）(b) CPU→GPU フレーム転送 (c) 4:4:4 のドライバ依存、といったコスト・不確実性を持つ。よって `auto` が小サイズを libx264 に回すのは能力回避でなく**正当な最適化**。
5. **パディングモードは非採用（Q2 への回答, NG1）**: D4 より低解像度で nvenc を強制する性能メリットが無く、pad は出力寸法変化・毎フレーム処理・config 増を招く。プロジェクトの「1 関心 1 機構 / 最小解」に反するため入れない。将来 nvenc 強制の実需要が出たら別モードとして追加する。
6. **明示指定は尊重（要件②）**: `h264_nvenc` を明示したユーザに黙って libx264 を使うのは意図に反する。明示 nvenc + 非対応は fatal にして「auto を使え」と促す。自動判断が欲しいユーザは `auto` を選ぶ、という明快な二分。
7. **既定を `auto` に**: 何も指定しない場合に最も壊れにくい挙動（=`auto`）を既定とする。既存 config は `video_codec` を明示しているため影響を受けない（§影響・移行）。

## 実装スケルトン（`core/anet-core/src/metrics_logger.cpp` 中心）

`VideoLogger` メンバ追加（[metrics_logger.hpp:84-102](../../core/anet-core/include/anet/metrics_logger.hpp)）:

```cpp
// 追加メンバ
std::atomic<bool> ffmpeg_dead_{false};
int  exit_code_ = 0;
std::string captured_stderr_;
std::string launch_cmd_;   // エラーメッセージ用に構築 cmd を保持
```

コーデック決定 + 起動 + 生存チェック（コンストラクタ [metrics_logger.cpp:64-122](../../core/anet-core/src/metrics_logger.cpp)）:

```cpp
namespace {
constexpr int kNvencMinWidth  = 160;   // NVENC H.264 ~145 + margin（実測 128 不可 / 512 可）
constexpr int kNvencMinHeight = 64;    // ~49 + margin
bool IsNvencEligible(int w, int h) {
    return (w % 2 == 0) && (h % 2 == 0) && w >= kNvencMinWidth && h >= kNvencMinHeight;
}
} // namespace

// 構築時: requested = config の video_codec
std::string chosen;
if (requested == "auto") {
    chosen = IsNvencEligible(width_, height_) ? "h264_nvenc" : "libx264";
    LOG::info() << "VideoLogger(auto): " << width_ << "x" << height_
                << " -> " << chosen;
} else if (requested == "h264_nvenc") {
    ANET_CHECK_MSG(IsNvencEligible(width_, height_),
        "h264_nvenc は " << width_ << "x" << height_
        << " 非対応(最小 " << kNvencMinWidth << "x" << kNvencMinHeight
        << "・偶数)。auto か libx264 を指定してください: " << path_);
    chosen = "h264_nvenc";
} else {
    chosen = requested;   // passthrough
}

LaunchFfmpeg(chosen);                       // 既存 launch 手順を関数化
if (DiedAtStartup()) {                       // 起動直後生存チェック(R1-2)
    DrainStderrLocked();                     // captured_stderr_ へ
    if (requested == "auto" && chosen == "h264_nvenc") {
        LOG::warn() << "nvenc 起動失敗(exit=" << exit_code_ << "): "
                    << captured_stderr_ << " -> libx264 で再起動: " << path_;
        RelaunchWith("libx264");
        ANET_CHECK_MSG(!DiedAtStartup(),
            "libx264 でも ffmpeg 起動失敗: exit=" << exit_code_
            << " stderr=" << captured_stderr_ << " path=" << path_);
    } else {
        ANET_SYSTEM_ERROR("ffmpeg 起動失敗: exit=" << exit_code_
            << " stderr=" << captured_stderr_ << " cmd=" << launch_cmd_);
    }
}
```

終了ハンドラ（launch 後に bind。ffmpeg 死を stream 非依存で捕捉）:

```cpp
process_->Bind(wxEVT_END_PROCESS, [this](wxProcessEvent& e) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    exit_code_ = e.GetExitCode();
    if (process_) DrainStderrLocked();       // GetErrorStream() を読む
    ffmpeg_dead_.store(true);
    stream_ = nullptr;                        // 以降 WriteFrame は触らない
});
```

`WriteFrame`（[metrics_logger.cpp:124-142](../../core/anet-core/src/metrics_logger.cpp)）:

```cpp
void VideoLogger::WriteFrame(const wxImage& img) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (ffmpeg_dead_.load() || stream_ == nullptr) {
        ANET_SYSTEM_ERROR("ffmpeg プロセスが異常終了: exit=" << exit_code_
            << " stderr=" << captured_stderr_ << " path=" << path_);
    }
    // ... 既存の write ループ。IsOk() が false になったら return でなく
    //     ANET_SYSTEM_ERROR(...) で停止（stderr を回収して含める）
}
```

`Close()`（[metrics_logger.cpp:144-155](../../core/anet-core/src/metrics_logger.cpp)）:

```cpp
void VideoLogger::Close() {
    std::lock_guard<std::mutex> lock(write_mutex_);   // R1-4: 追加
    if (stream_) stream_ = nullptr;
    if (process_) {
        process_->SetNextHandler(nullptr);
        process_->CloseOutput();
        delete process_;
        process_ = nullptr;
    }
}
```

**実装上の要検証ポイント（wx ライフタイム）**: 本 UAF は「ffmpeg 死亡時に `stream_` の指し先が無効化される」ことが実証されている。上記は `WriteFrame` が **stream へ触れる前に** `ffmpeg_dead_`/`stream_==nullptr` を門番する設計だが、`wxEVT_END_PROCESS` がメインスレッドの event loop で処理される前に wx 内部が stream を無効化する窓が理論上残る。実装時に元リポジトリ（128×128+nvenc）で **クラッシュが消えることを必ず確認**し、もし残る場合は `WriteFrame` 先頭に `wxProcess::Exists(pid)`（stream 非依存の生存確認）を追加する。`ExecuteStarter`（[metrics_logger.hpp:47-79](../../core/anet-core/include/anet/metrics_logger.hpp)）経由でメインスレッド起動している点に留意（bind/終了通知もメインスレッド）。

## 影響・移行

- **既定変更（判断ポイント）**: `MetricsLoggerConfig::video_codec` の既定を `"libx264"` → `"auto"` に変更（R2-1）。既存 config（本 run の `metrics_logger.video_codec = h264_nvenc` 等）は**明示指定のため影響なし**。未指定のケースのみ最も安全な `auto` になる。
- **明示 `h264_nvenc` + 小動画は挙動変化**: 従来 UAF で不定クラッシュ → 本 PRD 後は**構築時の明瞭な `ANET_SYSTEM_ERROR`**（fail-fast）。UAF よりは改善だが「落ちる」点は同じ。**パネル動画を実際に描きたい場合**は次のいずれか:
  - (推奨) `metrics_logger.video_codec = auto` にする → パネルの 128 は libx264 で出力される。
  - `SweepHeatMapPanel::CreateObserver`（[HeatMapPanel.cpp:395-396](../../apps/runner/src/HeatMapPanel.cpp)）の `image_width/height` を `-1` でなく妥当値（例 512）にする、または `grid` を偶数・十分大に。
  - ライブ表示専用で動画不要なら、パネル観測子の video 出力を無効化する経路を検討（別件・設計の匂い、NG2）。
- 他の video 経路（`43_agent_img`=512, `45_agent_img`=1024, `metrics.image` 各種）は全て適格サイズのため `auto` で従来通り nvenc が選ばれ、挙動不変。

## Testing Decisions

- **単体（CUDA/GPU 不要・常時）**:
  - `IsNvencEligible` の境界: `128x128`→false, `160x64`→true, `159x64`→false, `160x63`→false, `161x64`（奇数）→false。
  - コーデック決定ロジック: `auto`+小→libx264 / `auto`+大→nvenc / `h264_nvenc`+小→throw（`ANET_SYSTEM_ERROR`）/ `libx264`→passthrough。純関数として切り出しテスト可能に。
- **統合 / 手動（実機・GPU）**:
  - `SweepHeatMapPanel` 表示 + `video_codec=auto` → **UAF クラッシュしない**、`vector_policy...mkv`（128）が libx264 で **非 0 バイト**出力。
  - `video_codec=h264_nvenc`（明示）+ パネル表示 → 構築時に **明瞭な `ANET_SYSTEM_ERROR`**（UAF でなく）。ログに cmd/exit/stderr。
  - ffmpeg を外部から kill（実行途中）→ `WriteFrame` で `ANET_SYSTEM_ERROR` + ffmpeg stderr がログに残る（R1-3/R1-5）。
  - 回帰: 512/1024 動画は `auto` で nvenc 選択・従来通り出力。
- `git diff --check` 必須（既存 PRD 慣習）。

## 受け入れ基準

- AC1: `SweepHeatMapPanel` 表示で UAF（`0xC0000005` @ `stream_->IsOk()`）が発生しない。
- AC2: `video_codec=auto` でパネル動画（128×128）が libx264 で非 0 バイト出力される。
- AC3: `video_codec=h264_nvenc`（明示）+ 非対応解像度は、UAF でなく cmd/exit/stderr 付き `ANET_SYSTEM_ERROR` で停止する。
- AC4: ffmpeg を故意に停止させると `ANET_SYSTEM_ERROR` で停止し、ログに ffmpeg の stderr（例: `return code -22`）が残る。
- AC5: 既存の適格サイズ動画（512/1024）は `auto` で nvenc が選ばれ挙動不変。
