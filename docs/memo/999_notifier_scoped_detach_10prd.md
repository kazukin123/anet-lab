# Notifier::AttachScoped の Detach 契約修復 — wrapper ハンドルを返す PRD

> 関連: [rl.hpp:1004-1038](../../core/anet-core/include/anet/rl.hpp)(AttachScoped), [rl.cpp:899-936](../../core/anet-core/src/rl.cpp)(Detach), [rl.cpp:984-1009](../../core/anet-core/src/rl.cpp)(Notify), [image.cpp:619-632](../../core/anet-core/src/image.cpp)(MakeBundle), [RunnerFrame.cpp:391-422](../../apps/runner/src/RunnerFrame.cpp)(回避実装)。
> PRD `055`(Runner toolbar) のレビュー中に確定した既存バグの切り出し。`055` 側は [055_runner_toolbar_20impl.md](done/055_runner_toolbar_20impl.md) の「前提」で本件を別課題として明示的に回避している。実装は別途(Codex 想定)、本書は self-contained に記述する。

## Context / Problem Statement

`Notifier::AttachScoped()` は `RunnerScoped*Observer` wrapper を生成して **wrapper を Attach** し、**inner observer を返す**。一方 `Notifier::Detach(shared_ptr)` はポインタ一致(`o == obs`)で消す。したがって **AttachScoped の戻り値を Detach へ渡すと必ず一致せず、静かに失敗する**。

```cpp
// rl.hpp:1021-1026 (Train 系。Learn/EpisodeEnd 系も同型)
std::shared_ptr<TrainObserver> AttachScoped(std::shared_ptr<TrainObserver> observer, std::shared_ptr<const Runner> target_runner)
{
    auto wrapper = std::make_shared<RunnerScopedTrainObserver>(observer, target_runner);
    this->Attach(wrapper);   // Notifier が保持するのは wrapper
    return observer;         // ← 返るのは inner。Detach へ渡しても一致しない
}
```

`Notifier::Attach()` および `Attach<T>()` は「**戻り値＝Detach へ渡すもの**」という不変条件を満たしている。`AttachScoped` だけがこれを破っており、本件はその不変条件の回復である。

### 該当箇所(全 6 件)

| 箇所 | Attach | Detach | 現在の到達性 |
|---|---|---|---|
| TrainPanel | [:56](../../apps/runner/src/TrainPanel.cpp) | [:103](../../apps/runner/src/TrainPanel.cpp) | **到達しない**(下記) |
| EvalPanel | [:112](../../apps/runner/src/EvalPanel.cpp) | [DoClose :84](../../apps/runner/src/EvalPanel.cpp) / [OnClose :260](../../apps/runner/src/EvalPanel.cpp) | DoClose のみ |
| QValuePanel | [:291](../../apps/runner/src/QValuePanel.cpp) | [:936](../../apps/runner/src/QValuePanel.cpp) | Bind 済([:153](../../apps/runner/src/QValuePanel.cpp))だが wxPanel |
| Conv2dPanel | [:80](../../apps/runner/src/Conv2dPanel.cpp) | [dtor :119](../../apps/runner/src/Conv2dPanel.cpp) | 到達する |
| image.cpp `MakeBundle` | [:628](../../core/anet-core/src/image.cpp) | [:630](../../core/anet-core/src/image.cpp) | 潜在(呼出元 0) |
| trainer.cpp | [:929](../../core/anet-core/src/trainer.cpp) | なし | 戻り値破棄 |

### 実害の実測(優先度判断の根拠)

**実害はほぼ出ていない。** 本 PRD の価値は「契約の修復」であって「現在のリークの停止」ではない。

1. **`TrainPanel::OnClose` / `EvalPanel::OnClose` はどこにも Bind されていない。** TrainPanel の Bind は `wxEVT_SIZE`([:38](../../apps/runner/src/TrainPanel.cpp)) と `wxEVT_TIMER`([:65](../../apps/runner/src/TrainPanel.cpp)) だけ、EvalPanel は `wxEVT_TIMER`([:122](../../apps/runner/src/EvalPanel.cpp)) だけ。AUI ペインは既定で hide のみ([RunnerFrame.cpp:361](../../apps/runner/src/RunnerFrame.cpp) のコメント)なので発火しない。つまり TrainPanel は「静かに失敗する」以前に **Detach 自体が呼ばれていない**。
2. 実際に走る detach 経路は `EvalPanel::DoClose`([RunnerFrame.cpp:1072](../../apps/runner/src/RunnerFrame.cpp) と `~EvalPanel`)と `Conv2dPanel` の dtor で、どちらもアプリ終了時。
3. `ImageProviderManager::Remove()`([image.cpp:758](../../core/anet-core/src/image.cpp)) は呼出元 0 件。image.cpp 側は潜在バグ。なお [image.cpp:627](../../core/anet-core/src/image.cpp) に wrapper を自前生成するコードがコメントアウトで残っており、記述時点で気づかれていた痕跡がある。

### 修正が変えるもの(重要)

現在は一致するものが無いため `erase(end(), end())` となり、要素の移動も破棄も起きない＝**実質読み取りだけ**。一方 `Notifier` は mutex を持たず、[`Notify`](../../core/anet-core/src/rl.cpp) は `for (auto obs : train_observers_)` でメンバ vector を直接走査する。

したがって **本修正は「無害な no-op」を「Trainer スレッドの反復中に起こりうる実 mutation」に変える**。現状安全なのは [RunnerFrame::OnClose](../../apps/runner/src/RunnerFrame.cpp) が `StopTraining()`(join まで実施) → `DetachTrainStatusObserver()` → `eval_panel_->DoClose()` の順だからであり、これは暗黙の前提である。本 PRD ではこれを **Notifier の契約として明文化する**(mutex 導入は非ゴール、NG1)。

## ゴール / 非ゴール

**ゴール:**
- **G1**: `AttachScoped()` の戻り値をそのまま `Detach()` へ渡すと、対象 observer が確実に外れる。`Attach()` と同じ「戻り値＝Detach へ渡すもの」契約に揃える。
- **G2**: 契約変更を全呼び出し側へ同一変更内で反映し、回避実装([RunnerFrame.cpp:409-412](../../apps/runner/src/RunnerFrame.cpp))を撤去する。
- **G3**: 「Detach は Runner 停止後に呼ぶ」を `Notifier` のコメント上の契約として明記する。
- **G4**: attach → detach 後に通知が止まることを `anet-core-test` で検証する(回帰テスト)。

**非ゴール:**
- **NG1**: `Notifier` のスレッド安全化(mutex 導入)。整合性の機構であり本件(契約の一致)とは別関心。`Notify` は hot path のため性能判断が別途必要。**別課題**。
- **NG2**: `TrainPanel::OnClose` / `EvalPanel::OnClose` の Bind 欠落。本件(Notifier の契約バグ)とは別の欠陥。**別課題**、本 PRD では報告のみ。
- **NG3**: dead code の削除。`Notifier::Detach(const TrainObserver*)` / `(const LearnObserver*)` / `(const EpisodeEndObserver*)` の 3 本([rl.cpp:938-975](../../core/anet-core/src/rl.cpp))は呼出元 0、`ImageProviderManager::Remove()` も呼出元 0。既存 dead code は削除せず報告にとどめる(AGENTS.md「Surgical Changes」)。
- **NG4**: `Detach` 側で wrapper の inner を辿る案。§設計判断 2 で棄却。

## 要件

### R1: AttachScoped が wrapper を返す(G1)

**非 template 版 3 本**([rl.hpp:1021-1038](../../core/anet-core/include/anet/rl.hpp))は `return observer;` → `return wrapper;` の 1 行のみ。wrapper は各 observer 基底の派生なので **宣言(戻り型)は変更不要**。

```cpp
/// 対象 Runner のイベントだけを observer へ中継する wrapper を Attach し、**wrapper** を返す。
/// 戻り値はそのまま Detach() へ渡すこと(Attach() と同じ契約)。inner が必要な場合は呼び出し側で保持する。
std::shared_ptr<TrainObserver> AttachScoped(std::shared_ptr<TrainObserver> observer, std::shared_ptr<const Runner> target_runner)
{
    auto wrapper = std::make_shared<RunnerScopedTrainObserver>(observer, target_runner);
    this->Attach(wrapper);
    return wrapper;
}
```

**template 版**([rl.hpp:1004-1020](../../core/anet-core/include/anet/rl.hpp))は、現在 `if constexpr` を 3 本並べて最大 3 つの wrapper を作りうる。wrapper を返す設計では **どれを返すか決まらない**ため、`else if constexpr` の排他連鎖へ変更し、複数の observer 基底を継承する `T` は `static_assert` でコンパイルエラーにする(fail-fast)。戻り型は `auto` で推論する(排他分岐なので 1 型に確定する)。

```cpp
template <class T, class... Args>
auto AttachScoped(std::shared_ptr<Runner> target_runner, Args&&... args)
{
    // wrapper を 1 つだけ返す契約のため、observer 基底を 2 つ以上継承する T は受け付けない。
    static_assert(
        (std::is_base_of_v<TrainObserver, T> ? 1 : 0)
      + (std::is_base_of_v<LearnObserver, T> ? 1 : 0)
      + (std::is_base_of_v<EpisodeEndObserver, T> ? 1 : 0) == 1,
        "AttachScoped<T>: T must derive from exactly one observer base");

    auto obs = std::make_shared<T>(std::forward<Args>(args)...);
    if constexpr (std::is_base_of_v<TrainObserver, T>) {
        auto wrapper = std::make_shared<RunnerScopedTrainObserver>(obs, target_runner);
        this->Attach(wrapper);
        return wrapper;
    } else if constexpr (std::is_base_of_v<LearnObserver, T>) {
        // … RunnerScopedLearnObserver で同型
    } else {
        // … RunnerScopedEpisodeEndObserver で同型
    }
}
```

現行コードベースに observer 基底を 2 つ以上継承する型は **0 件**である(`TaggedTrainObserver` / `TaggedLearnObserver` は `TaggedObserver` ＋単一 observer、`ImageProvider` は observer 基底ではない: [observers.hpp:30,37,72,120,184,218,247,276](../../core/anet-core/include/anet/observers.hpp))。よって `static_assert` は現行の全利用を通す。

`[[nodiscard]]` は付けない。[trainer.cpp:929](../../core/anet-core/src/trainer.cpp) が戻り値を破棄しており、警告になるだけで得がない。

### R2: 呼び出し側の移行(G2)

| 対象 | 差分 |
|---|---|
| TrainPanel / EvalPanel / QValuePanel / Conv2dPanel | **なし**。`observer_` は 4 箇所とも `std::shared_ptr<anet::rl::TrainObserver>` 型で用途は Detach のみ([TrainPanel.hpp:34](../../apps/runner/src/TrainPanel.hpp) / [EvalPanel.hpp:68](../../apps/runner/src/EvalPanel.hpp) / [QValuePanel.hpp:108](../../apps/runner/src/QValuePanel.hpp) / [Conv2dPanel.hpp:58](../../apps/runner/src/Conv2dPanel.hpp))。wrapper も `TrainObserver` なので暗黙 upcast で受かる |
| trainer.cpp:929 | **なし**(戻り値破棄) |
| image.cpp `MakeBundle` | **要修正**。attach / detach が別ラムダ・別時刻のため、attach 時に受け取った wrapper を detach へ引き渡す必要がある |
| RunnerFrame | **要修正**。回避実装を撤去し `AttachScoped` 経由へ戻す |
| HeatMapPanel | 対象外。非 scoped の `Attach<T>` を使い戻り値へ `GetImageData()` を呼ぶ([:484](../../apps/runner/src/HeatMapPanel.cpp), [:536](../../apps/runner/src/HeatMapPanel.cpp))。concrete 型が要る用途は `Attach<T>`、という棲み分けは維持する |

`MakeBundle`([image.cpp:619-632](../../core/anet-core/src/image.cpp)) は attach 時に detach 手順を確定させる。`auto` が overload から wrapper 型を受けるため、新たな公開 trait は不要。

```cpp
template<typename T>
static ImageProviderBundle MakeBundle(std::shared_ptr<T> observer, std::shared_ptr<const Runner> runner)
{
    // Notifier が実際に保持するのは wrapper。attach 時に受け取った wrapper で detach 手順を確定させる。
    auto detach_fn = std::make_shared<std::function<void(std::shared_ptr<Notifier>)>>();
    return {
        observer,
        [observer, runner, detach_fn](auto n)
        {
            auto scoped = n->AttachScoped(observer, runner);   // 型は overload が決める
            std::weak_ptr scoped_wp = scoped;
            *detach_fn = [scoped_wp](std::shared_ptr<Notifier> nn) { if (auto p = scoped_wp.lock()) nn->Detach(p); };
        },
        [detach_fn](auto n) { if (*detach_fn) (*detach_fn)(n); }
    };
}
```

`RunnerFrame::AttachTrainStatusObserver`([:391-413](../../apps/runner/src/RunnerFrame.cpp)) は wrapper の自前生成と回避コメント([:409](../../apps/runner/src/RunnerFrame.cpp))を撤去し `train_status_observer_ = train_status_notifier_->AttachScoped(observer, train_runner);` にする。撤去しないと本 PRD で解消済みの問題を指すコメントが残る。`DetachTrainStatusObserver()` は変更不要。

### R3: Detach のスレッド契約を明記(G3)

`Notifier` の宣言([rl.hpp:973](../../core/anet-core/include/anet/rl.hpp))に、Attach / Detach / Notify のスレッド前提をコメントで明記する。mutex は入れない(NG1)。

- `Notify` は Trainer スレッドから observer 列を走査する。`Notifier` は同期機構を持たない。
- `Attach` / `Detach` は対象 Runner が停止している状態(初期化時、または `StopTraining()` の後)でのみ呼ぶこと。
- 本修正以前は `Detach` が実質 no-op だったため違反しても顕在化しなかったが、修正後は実際に列を書き換える。

### R4: 回帰テスト(G4)

`core/anet-core/src/observers_test.cpp` へ追加する(新規 `rl_test.cpp` を作らない)。同ファイルに `rl::Notifier` の利用と `TestRunner`([:294](../../core/anet-core/src/observers_test.cpp)) / `TestAgent` / `TestBatchEnv` / `TrainEvent` 組み立て([:542](../../core/anet-core/src/observers_test.cpp))の fixture が揃っており、新規 fixture が不要なため。CMake は `src/*test.cpp` を GLOB する([CMakeLists.txt:18](../../core/anet-core/CMakeLists.txt))ので登録作業も不要。

新タグ `[notifier][scoped]`。`TestRunner` を 2 体用意し、`FunctionTrainObserver` の呼び出し回数で検証する。

- **T1**: `AttachScoped` した inner が、対象 runner の `Notify(TrainEvent)` で 1 回呼ばれる。
- **T2**: **戻り値を `Detach` した後、`Notify` で呼ばれない**(本件の回帰テスト。修正前は失敗する)。
- **T3**: 対象外 runner の `TrainEvent` では呼ばれない(wrapper 本来の scope 機能の非退行)。
- **T4**: `FunctionLearnObserver` で T1 / T2 が成立(`EpisodeEvalObserver` 経路が `LearnObserver` 派生のため、Learn 系統も契約に含まれることを固定する)。

## 設計判断(why。self-contained のため明記)

1. **wrapper を返す案を採る**。①4 パネルの `observer_` は既に `shared_ptr<TrainObserver>` 型かつ用途が Detach のみで、**呼び出し側の宣言変更が 0 行**で済む。②`AttachScoped` の戻り値を inner として使っている箇所が **0 件**(trainer.cpp は破棄、image.cpp は元の observer を別途 bundle に保持)。③`Attach` / `Attach<T>` が既に満たしている「戻り値＝Detach へ渡すもの」という不変条件の回復であり、新しい概念を増やさない。④[RunnerFrame.cpp:409-412](../../apps/runner/src/RunnerFrame.cpp) が既に手作業で実現している形と一致する。
2. **`Detach` 側で inner を辿る案は採らない**。inner を辿るには型分岐が要り `dynamic_cast` は production 禁止(AGENTS.md「コーディング規約」)。基底に `GetInner()` を生やすのは wrapper 一族の都合で全 observer 契約を汚す。互換分岐を残す形になりクリーンブレーク方針にも反する。加えて semantics が変わり、同じ inner を 2 つの runner へ scoped attach した場合に `Detach(inner)` が両方を消してしまう。
3. **多重継承 `T` は `static_assert` で弾く**。wrapper を 1 つ返す契約と、複数 wrapper を attach する現行実装は両立しない。現行 0 件なので実害なくコンパイル時に固定できる。「起こり得ない経路のエラー処理を足さない」と fail-fast を両立させる最小手段。
4. **スレッド安全は分離する**。「整合性(lock)」と「契約の一致」は別関心であり、1 関心 1 機構で扱う。mutex は `Notify` の hot path に入るため、必要性が具体的な経路で示された時点で別 PRD にする。本 PRD では前提を明文化するにとどめる(R3)。
5. **RunnerFrame の回避実装は同一変更内で撤去する**。回避が不要になったのは本修正の結果であり、残すと解消済みの問題を指すコメントが残留する(クリーンブレーク方針「移行漏れとして扱う」)。

## 影響・移行

- 公開ヘッダ `rl.hpp` の `AttachScoped` の**契約が変わる**(戻り値の意味)。呼び出し側の実コード差分は image.cpp と RunnerFrame のみ、パネル 4 件と trainer.cpp は 0 行。
- **挙動変化**: `Detach` が実際に observer 列を書き換えるようになる。現行の全 detach 経路は Runner 停止後のため安全だが、前提が暗黙から明示に変わる(R3)。
- template 版の戻り型が `shared_ptr<T>` から wrapper 型に変わるため、inner の concrete 型を必要とする新規利用は `Attach<T>` を使うことになる。現行に該当利用はない。
- `static_assert` 追加により、observer 基底を 2 つ以上継承する型を `AttachScoped<T>` へ渡すとコンパイルエラーになる。現行 0 件。

## 触る主なファイル

- `core/anet-core/include/anet/rl.hpp` — `AttachScoped` 非 template 3 本の `return` 差し替え、template 版の排他分岐化・`static_assert`・`auto` 戻り型、`Notifier` のスレッド契約コメント(R3)。
- `core/anet-core/src/image.cpp` — `MakeBundle` の detach 手順を attach 時確定へ変更。
- `apps/runner/src/RunnerFrame.cpp` — `AttachTrainStatusObserver` の回避実装と回避コメントを撤去し `AttachScoped` 経由へ。
- `core/anet-core/src/observers_test.cpp` — `[notifier][scoped]` テスト 4 件追加。

## Testing Decisions

- **単体**: R4 の T1〜T4。特に **T2 が本 PRD の受入中核**で、修正前に失敗し修正後に通ることを確認する。
- **回帰**: `anet-core-test` 全体。既知の失敗以外が増えないこと。
- **手動**: Runner アプリを起動し、学習開始 → 通常終了でクラッシュ・ハングがないこと(Detach が実 mutation になるため、終了経路の順序 `StopTraining()` → Detach が守られていることの実地確認)。
- `git diff --check` 必須。

## 受け入れ基準

- **AC1**: `AttachScoped()` の戻り値を `Detach()` へ渡すと、以降その observer へ通知が届かない(Train 系・Learn 系の両方でテスト済み)。
- **AC2**: 4 パネルと trainer.cpp に差分が生じない(呼び出し側の宣言・Detach 行が無変更のままビルド・動作する)。
- **AC3**: `MakeBundle` の detach が Notifier 上の wrapper を実際に外す。
- **AC4**: `RunnerFrame::AttachTrainStatusObserver` が `AttachScoped` 経由になり、本件を指す回避コメントが残っていない。
- **AC5**: `Notifier` に Attach / Detach / Notify のスレッド前提が明記されている。
- **AC6**: `anet-core-test` が既存分も含めて回帰しない。
