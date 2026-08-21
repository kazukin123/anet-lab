# DropMerge NoLegal 裁定の blocked-persistence 化

- 作成日: 2026-07-22
- 前提資料: `docs/memo/005_no_legal_drop_10prd.md`（NoLegalDrop 終端の追加経緯・初版仕様）
- 由来: 2026-07-22 の終局判定レビュー（Claude レビュー + Codex コメントの 2 往復で設計合意済み）。
  本 PRD は self-contained であり、レビュー時の handoff 文書への参照は不要。

## 問題

`DropMergeEnv` の `action_mode=direct_noop` において、終了理由の分類が Env 内部の安定待ち都合で揺れる。

現行の `isNoLegalDropState()` は `isWorldSettled()` を要求する。`isWorldSettled()` は盤面上の
**全 dynamic body** の線速度・角速度を閾値判定するため、spawn 可否に無関係な底部 fruit の微動や
円形 fruit の回転だけで NoLegal 判定が抑止される。

現行設定（`use_instant_drop=true` / `use_settle_after_drop=false`）では NOOP 1 回 = 物理 1 frame
であり、`no_drop_timeout_steps=100` の下では次の経路が実装上存在する。

```text
盤面は詰み（全 DROP 不能）
→ 方策が合理的に NOOP を選ぶ
→ しかし約 100 物理 frame 以内に全盤面が isWorldSettled() にならない
→ NoLegalDrop に到達できず NoDropTimeout へ漏れる
```

このため次の問題がある。

- 終了理由が「盤面の性質」ではなく「Env の裁定品質」で決まり、方策品質の観測に使えない。
  `NoDropTimeout` に「legal DROP があるのに置かなかった（真の無操作）」と「詰みだが裁定に失敗した」が混在する。
- 詰み候補を最大 100 RL step 待つ間、NN 推論・action 選択・Replay transition・実時間を浪費する。
- run_20260721-201834 の累積 eval では NoDrop 71/83 件 vs NoLegal 50/46 件（eval1/eval2、進行中の暫定値）で、
  高得点帯でも NoDrop が発生していた。ただし現行ログには内訳がなく、詰み漏れの規模は未計測。

## 再精査による問題の再定義（2026-07-25、PRD v2）

Phase 1 実装後の 100M Run（run_20260724-221301_imp2-vit128_tgtgreedy）の診断集計より:

- train `no_drop_timeout_on_candidate`: nonzero 0 / 267,111 metric samples。
  eval2 の NoDropTimeout 19 件も全て candidate=false。
  → 「詰みが裁定失敗で NoDropTimeout へ漏れる」経路は、現行 baseline 構成ではほぼ観測されない。
- 解消した blocked run の最大は 29 物理 frame（nonzero 中央値 2 / p95 15）。
  → 29 frame 塞がった後に合法が復活した実例があり、v1 の仮値 N=10 は回復盤面を誤終端する。棄却。

その上で、本 PRD の目的は頻度・成績・計算量ではなく**終端の意味論**にあると再定義された（ユーザー決定）:

- 「正しく学習した方策は詰み盤面で意図的に投了し、それが常に NoLegalDrop として受理される」契約を
  成立させる。現行受理条件の `isWorldSettled()` は agent の obs に無い速度場に依存し、
  受理が agent から予測不能である点が本質的欠陥。
- 終端の価格構造も対象に含める。現行実効値（BLK/Overflow=game_over_penalty −10、
  NoDropTimeout=no_drop_timeout_gameover_penalty 0 で done、NoLegalDrop=罰なし、
  time_penalty≈−0.0001）では、done+0 の timeout が guaranteed-0 の避難港となり、
  危険盤面での停滞が報酬合理的になる。さらに NoLegal も 0 のため「きれいな投了」と「停滞」が
  報酬上無差別で、投了の強化に学習圧が無い。この終盤 NOOP 適正が特徴空間経由で余裕盤面へ
  染み出す、が NEET 現象（盤面に余裕があるのに NOOP を連発する現象）の有力仮説であり、
  本 PRD の終端整理は NEET の上流対策を兼ねる。

## 目的

終了理由の契約を次へ揃える。

```text
NoLegalDrop:
  裁定により DROP 不能の継続が確定した盤面で方策が NOOP した（投了の受理、罰なし）。
  受理は遅くとも「DROP 不能が N 連続物理 frame」で確定し、agent から予測可能。

NoDropTimeout:
  legal DROP が存在するのに方策が NOOP を続けた（停滞の敗着）。
  他の GameOver と同格の敗着として非 0 の罰を与える（罰値はチューニング領域。
  初期値は game_over_penalty と同値 −10 を起点。2-5 参照）。
```

到達は段階分割する（v1 の 2 Phase 構成を 3 段階に再編）。

- Phase 1（実装済み）: 診断メトリクス追加のみ。挙動完全不変。
- Phase 1b: terminal 側診断の追補（挙動不変）。N の確定と受理遅延の実測に必要。
- Phase 2: 受理判定の変更（persistence 受理、fast-forward なし）。default false。
  timeout 罰の非 0 化（既存 config キー）も併せて有効化する意図。値・適用はチューニング領域（2-5 参照）。

## 用語

- **candidate**: 「現在の fruit を置ける DROP が 1 つもない」瞬間判定。
  安定確認を含まない、pre-action / per-frame どちらでも評価可能な共有 predicate。
- **persistence**: candidate が連続成立した物理 frame 数。
- **裁定 horizon**: 「DROP 不能が N 連続物理 frame 続いた状態を、DropMerge における実効的な詰みと
  定義する」というゲームルール上の裁定。N より後に legal が復活し得るケースを誤終端ではなく
  **意図された終局扱い**とする宣言である（詰みの物理的証明ではない）。
- 物理 frame と RL step を混同しないこと。`no_drop_timeout_steps` は RL step 単位、
  本 PRD の新設定・新カウンタは物理 frame 単位（設定名も `*_frames` とする）。

## 決定経緯の要約

レビュー合意で確定した主要判断。実装時に蒸し返さないこと。

1. **裁定の中心を速度から結果へ**: 「安定（速度閾値）を待ってから詰みを判定する」のではなく、
   「DROP 不能状態の継続（blocked-persistence）」自体を裁定条件にする。確認したい事実は
   「置ける場所がない状態が持続するか」であり、速度は手段にすぎない。これにより
   閾値チューニング・角速度の間接影響（回転→摩擦→隣接 fruit を押す）・下部運動の上部への
   伝播問題がすべて「復活すれば candidate が途切れて検出される」へ還元される。
2. **missed_resignation（persistence 条件付き投了失敗メトリクス）は却下**:
   `use_instant_drop` では DROP 中 `current_rank=0` で candidate 不成立、rank 確定 frame で
   物理ループを脱出するため、次の action 選択時点の persistence は高々 1 frame しか蓄積しない。
   さらに Phase 2 導入後は persistence >= N 到達が即 NoLegal 終局となるため、
   「pre-action で persistence >= N」という状態は構造的に空集合となる。
   厳密な投了率には反実仮想の物理シミュレーションが必要であり、導入しない。
3. **代替は `blocked_drop_on_candidate`**: pre-action の瞬間 candidate 成立時に DROP を選んだ
   イベント。candidate の区間計算は `processAction()` の actual_x の取り得る区間と同一式のため、
   candidate 成立時の DROP は drop_noise がどう出ても**必ず** SpawnBlocked になる。
   よって本メトリクスの意味は「確実に死ぬ DROP を選んだ」であり、「詰みでの投了失敗」ではない。
   瞬間判定のため Phase 1 / Phase 2 で意味が変わらない。
4. **Phase 分割の再編（ユーザー決定）**: 既存比較 Run との互換を優先し、Phase 1 は
   「メトリクス追加のみ」とする。当初案にあった「timeout 時再分類を Phase 1 に含める」は取り下げ。
5. **裁定上限・timeout 再分類・no_legal_adjudication_timeout は導入しない（PRD 化時の簡約）**:
   レビュー合意時点では「裁定上限到達時も NoLegal 確定 + 診断 flag」「timeout 時の再分類」を
   採用予定だったが、これらは「安定確認を待つ」旧方式の裁定を前提にした機構だった。
   blocked-persistence 裁定では次が成り立つため、いずれも到達不能な dead 機構になる。

   ```text
   裁定ループ継続条件 = NOOP かつ candidate 成立継続 かつ persistence < N
   - candidate 成立が続く限り persistence は単調増加し、N frame で必ず NoLegal 確定
   - candidate が 1 frame でも不成立になれば即座にループ脱出し通常 state を返す（persistence リセット）
   → 裁定は高々 N frame で必ず決着する。別個の上限は不要。
   → NoDropTimeout 到達時に persistence >= N であることは構造的に不可能（事前に裁定済みのため）。
     瞬間 candidate による再分類は「振動して時々 legal が開く盤面」（=真の詰みでない）を
     NoLegal へ誤分類するリスクの再導入であり、行わない。
   ```

   代わりに Phase 1 の `no_drop_timeout_on_candidate` が観測を担う（Phase 2 ON の世界で
   本メトリクスが恒常的に非ゼロなら、振動系盤面の存在または裁定ロジックの欠陥を意味する。
   その時点で persistence 条件付き再分類等の対処を別途検討する）。

6. **fast-forward は廃止（2026-07-25 ユーザー決定、v2）**: v1 2-2 の「同一 Step 内で N frame
   進めて裁定する」方式は、問題設定の複雑化と人間プレイ感覚との乖離が許容限界を超えるため不採用。
   裁定は通常の 1 step = 1 frame 進行のまま、既存の `blocked_candidate_frames_`（RL step を跨いで
   持続する per-frame カウンタ）が N に達した step の NOOP で確定する。これに伴い v1 2-2 の
   契約検討（merge 積算・観測折り畳み・RNG 消費・max_sim_steps 干渉）は全て不要になった。
   決定経緯 5 の結論（timeout 再分類・別個上限は不要）は persistence 受理でも同様に成立する
   （「ループ継続」の記述のみ「step を跨いだカウンタ継続」に読み替え）。
7. **N=10 は棄却、仮値 N=60（= 物理 1.0 秒）**: 100M 診断で 29 frame 継続後に回復した実例が
   観測されたため（再定義セクション参照）。60 は観測最大の 2 倍超の余裕を持ち、かつ
   no_drop_timeout_steps=100 より十分小さい。「置けない状態が 1 秒続いたら終局」という
   人間感覚で説明可能な値でもある。Phase 1b の terminal 側分布で最終確定する。
8. **timeout は truncated ではなく done + 罰（b 案）を採用（2026-07-25 ユーザー決定）**:
   truncated 化は過去の議論で却下済み。加えて (i) shot clock は `no_drop_timeout_ratio` として
   obs に既在し、クロックを進めるのは agent 自身の NOOP 選択のみ＝終端は agent 支配下の
   Markov な帰結であり、外因打ち切り（truncated）ではなく done が意味論的に正しい。
   (ii) 罰値の初期値は game_over_penalty と同値（現行 −10）を起点とし、「タイムアウトも他と
   同格の敗着」と位置づける。ただし同値規律・自動参照は設けず、以後は独立にチューニング可能な
   自由値とする（グリル確定 2026-07-25）。(iii) truncated 化を
   `use_no_drop_timeout_gameover` の flip で行うと obs 次元（scalar 4/5）まで変わり
   一軸 A/B が壊れる罠もある（DropMergeEnv.cpp の StateSpec 構築が同 flag にゲートされているため）。
9. **受理は OR 構成（settled 即受理 fast-path + persistence 上限保証）**: 純 persistence 一本化だと、
   settled な詰み盤面（現行で最初の NOOP により即受理できている多数派）まで N step 待たせ、
   待機中の探索ノイズ（ε による確率的 DROP = SpawnBlocked 死）に晒す時間が延びる。
   既存の `isNoLegalDropState()`（candidate && settled）を即受理 fast-path として残し、
   `persistence >= N` を上限保証として OR する。「静止した詰みは即受理、動いていても
   1 秒詰みなら受理」。契約の本体は受理の**上限**が決定的であることで、早期受理側のばらつきは
   害にならない（価値は同一の 0、差は time_penalty 数 step 分のみ）。
10. **timeout 罰の非 0 化は受理安全網（PH2）と併せて有効化する意図**: 受理上限が保証されないまま
    罰を入れると、裁定失敗した真の詰み（正しい投了）が敗着扱いされる不条理が生じ得るため。
    ただし構成として強制はせず、OFF+罰非 0 も黙って許容する（warn も出さない。グリル確定
    2026-07-25）。罰値・適用時期・Run 設計はハイパラ探索領域であり PRD の範疇外。

## Phase 分割方針

```text
Phase 1（実装済み、2026-07-24 の 100M Run で観測済み）:
  メトリクス追加のみ。
  判定・報酬・obs・done/truncated・RNG 消費列は全構成で完全不変。

Phase 1b（挙動不変の診断追補）:
  episode 終端時点の未解消 blocked run を終端理由とセットで記録する。
  現行 PH1 は「解消した run」しか記録せず、terminal 側の分布
  （受理までの待ち frame 数、timeout 時点の継続 frame 数）が欠測しているため。
  N の最終確定と、settled 待ち受理遅延の実測に使う。

Phase 2（persistence 受理の導入）:
  use_no_legal_adjudication = true の時のみ受理判定を拡張（OR 構成、決定経緯 9）。
  fast-forward は行わない（決定経緯 6）。default false で旧挙動を厳密維持。
  適用は新 Run のみとし、resume 中の Run への途中適用は禁止
  （終端契約が変わるため。v1 の transition 圧縮という理由は FF 廃止で消えたが、
    契約変更自体が理由として残る）。
  timeout 罰の非 0 化（no_drop_timeout_gameover_penalty、既存 config キー）も併せて
  有効化する意図（決定経緯 8/10、2-5）。値・適用時期はチューニング領域で PRD 範疇外。
```

## 対象ファイル

```text
core/envs/dropmerge1/src/DropMergeEnv.hpp
core/envs/dropmerge1/src/DropMergeEnv.cpp
core/envs/dropmerge1/src/DropMergeEnv_test.cpp
apps/runner/config/DropMerge.txt        （metrics 定義。Phase 2 で設定キー追加）
```

## 現状コードの要点（2026-07-22 時点の working copy 基準）

- `DropMergeEnv::Step()`: 終了判定は NoLegalDrop → MaxStep → NoDropTimeout の順で、
  後段は `!done` ガード付き。同一 step での上書きなし。
- 物理ループ（`Step()` 内 do-while）: 既に「1 RL step = 複数物理 frame」を許す構造。
  ループ内で毎 frame `world_->Step()` → `processMerges()` → `checkGameOver()` → `calcReward()`
  （merge score を `accumulated_reward` へ積算）→ `updateDropperStatus()` → game_over なら break →
  `isWorldSettled()` 評価 → `keep_simulating` 判定、の順。
  `use_instant_drop=true` では `keep_simulating = dropper_.is_busy`（NOOP は 1 frame で脱出）。
- `time_penalty` / `noop_penalty` / `game_over_penalty` は RL step 単位でループ外 1 回。
- `isNoLegalDropState()`: DirectNoop / !game_over_ / !busy / !pending / rank 有効 /
  merge・destroy 空 / `isWorldSettled()` / `!hasAnyLegalDropForCurrentFruit()` の全 AND。
- `hasAnyLegalDropForCurrentFruit()`: 全 drop column について列中心 ± drop_noise（壁 clamp 込み）の
  範囲を `hasClearSpawnXInRange()` で判定。1 列でも clear があれば true（early-return）。
- `hasClearSpawnXInRange()`: 既存 fruit が spawn 高さで塞ぐ x 区間（blocked interval）を収集し、
  ソート後に union が範囲全体を覆うかを判定。`isSpawnAreaClear()` と同じ body filter /
  `kSpawnOverlapMargin` を使用。
- SpawnBlocked 死は `processAction()` 内で `spawnFruit()` を呼ばずに確定するため、
  **終了時の盤面 = DROP 適用前の盤面**。
- メトリクスは `GetScalar()` の `episode_just_ended_` ガード付き one-hot パターン
  （`term_reason_*` 系）と、episode 集計パターン（`ep_settle_steps_sum_` / `ep_settle_count_` /
  `ep_settle_steps_max_` → 終了時に `last_ep_*` へ確定）が既にある。新メトリクスは両パターンを踏襲する。

---

## Phase 1 仕様（メトリクス追加のみ）

### 1-1. candidate predicate の抽出

`isNoLegalDropState()` から `isWorldSettled()` を除いた部分を private helper として抽出する。

```cpp
/// NoLegal candidate 判定: 現在の fruit を置ける DROP が 1 つもない（安定確認なしの瞬間判定）。
/// 終了判定 (isNoLegalDropState) とメトリクス (blocked_drop_on_candidate 等) で共有する。
bool DropMergeEnv::isNoLegalCandidateState() const
{
    ANET_PROFILE_FUNC();

    if (action_mode_ != ActionMode::DirectNoop) return false;
    if (game_over_) return false;
    if (dropper_.is_busy) return false;
    if (dropper_.pending_body != nullptr) return false;
    if (dropper_.current_rank < 1 || dropper_.current_rank > kFruitTypeCount) return false;
    if (!merge_requests_.empty()) return false;
    if (!bodies_to_destroy_.empty()) return false;

    return !hasAnyLegalDropForCurrentFruit();
}
```

既存 `isNoLegalDropState()` は次へ書き換える（**結果不変のリファクタ**。両関数とも const・副作用なしのため
評価順の入れ替えは挙動へ影響しない）。

```cpp
bool DropMergeEnv::isNoLegalDropState() const
{
    ANET_PROFILE_FUNC();
    if (!isNoLegalCandidateState()) return false;
    return isWorldSettled();
}
```

併せて `hasAnyLegalDropForCurrentFruit()` の冒頭（`r_drop` 算出後）へ防御を 1 行追加する。

```cpp
const float limit = half_w - r_drop - 0.01f;
if (limit <= 0.0f) return false; // 果物が箱幅より大きく置ける x が存在しない。clamp(lo>hi) の UB 防止
```

production 設定（最大 fruit 半径 0.84 < half_w 1.5）では発生しないため挙動不変。
極小 fixture（後述テスト）が誤って `std::clamp` の lo > hi（UB）を踏むことを防ぐ。

### 1-2. persistence カウンタと blocked run-length 計測

member を追加する。

```cpp
// NoLegal candidate 継続計測用
int blocked_candidate_frames_ = 0;   ///< candidate の連続成立物理 frame 数（Reset でクリア）

// メトリクス用（step_count_ == 0 ブロックでリセット、既存 ep_settle_* と同パターン）
int ep_blocked_run_sum_ = 0;         ///< 解消した blocked run の frame 数合計
int ep_blocked_run_count_ = 0;       ///< 解消した blocked run の本数
int ep_blocked_run_max_ = 0;         ///< 解消した blocked run の最大 frame 数
float last_ep_mean_blocked_frames_ = 0.0f;
int last_ep_max_blocked_frames_ = 0;
bool ep_blocked_drop_on_candidate_ = false;
bool ep_no_drop_timeout_on_candidate_ = false;
```

物理ループ内の **game_over break の後**（`isWorldSettled()` 評価の隣、生存 frame のみ通過する位置）で
毎 frame 更新する。

```cpp
// NoLegal candidate の継続 frame 数を更新（Phase 2 裁定と N 決定用診断の共有カウンタ）
if (isNoLegalCandidateState()) {
    blocked_candidate_frames_++;
} else {
    if (blocked_candidate_frames_ > 0) {
        // candidate が途切れた = 解消した run として記録
        ep_blocked_run_sum_ += blocked_candidate_frames_;
        ep_blocked_run_count_++;
        ep_blocked_run_max_ = std::max(ep_blocked_run_max_, blocked_candidate_frames_);
    }
    blocked_candidate_frames_ = 0;
}
```

集計の契約:

- 記録対象は「candidate が途切れて解消した run」のみ。game_over frame（break 済みで評価に到達しない）と
  episode 終了時に継続中の run（打ち切り）は記録しない。
  Phase 2 の裁定は「candidate の連続 N frame」で確定するため、**裁定と同一定義でカウントした
  解消 run の分布**こそが「N をいくつにすれば false terminal 率がどの程度になるか」を直接与える。
  解消理由（legal 幾何の復活か、merge 発生等のガード落ちか）は区別しない。裁定側でも同じ理由で
  カウンタがリセットされるため、定義が一致する。
- `blocked_candidate_frames_` は `Reset()` でクリアする（`steps_since_last_drop_` と同じ場所）。
- コスト: DROP 中は busy ガードで short-circuit し `hasAnyLegalDropForCurrentFruit()` は走らない。
  NOOP step では 1 frame につき 1 回評価されるが、盤面に空きがあれば 1 列目で early-return する。
  既存の `ANET_PROFILE_FUNC()`（`isNoLegalCandidateState` にも付与）で実測すること。

episode 終了時（`done || truncated` ブロック、既存 `last_ep_mean_settle_steps_` 確定の隣）:

```cpp
last_ep_max_blocked_frames_ = ep_blocked_run_max_;
last_ep_mean_blocked_frames_ = (ep_blocked_run_count_ > 0)
    ? (static_cast<float>(ep_blocked_run_sum_) / ep_blocked_run_count_) : 0.0f;
```

`step_count_ == 0` ブロック（新 episode 開始時）で `ep_blocked_run_sum_/count_/max_` と
`ep_blocked_drop_on_candidate_` / `ep_no_drop_timeout_on_candidate_` をリセットする。

### 1-3. pre-action 判定と blocked_drop_on_candidate

`Step()` 冒頭、action 分類の後・`processAction()` の**前**に candidate を snapshot する。

```cpp
// action 適用前の candidate 判定（blocked_drop_on_candidate 用）
const bool pre_action_candidate = isNoLegalCandidateState();
```

`processAction()` の後で判定する。

```cpp
// 全 DROP 不能なのに DROP を選んで死んだ（確実死 DROP の選択）
if (pre_action_candidate && is_drop_action && term_reason_ == TerminationReason::SpawnBlocked) {
    ep_blocked_drop_on_candidate_ = true;
}
```

契約上 `pre_action_candidate && is_drop_action` ならば必ず SpawnBlocked になる（決定経緯 3）。
`term_reason_` の確認は、万一の不変条件破れで嘘メトリクスを出さないための防御として残す。

### 1-4. no_drop_timeout_on_candidate

既存の NoDropTimeout 判定ブロック（ショットクロック判定）内で、timeout 成立時に candidate を評価する。

```cpp
if (!done && !truncated && config_.no_drop_timeout_steps > 0 &&
    steps_since_last_drop_ >= config_.no_drop_timeout_steps) {
    term_reason_ = TerminationReason::NoDropTimeout;
    ep_no_drop_timeout_on_candidate_ = isNoLegalCandidateState();  // ← 追加
    ...（既存処理は不変）
}
```

意味: 「NoDropTimeout に至った episode のうち、timeout 時点で全 DROP 不能だったもの」
= 現行判定が詰みを NoDropTimeout へ漏らした件数の実測（Phase 2 の効果見積もりの根拠）。
定義は `term_reason_` の値ではなく「timeout 経路に入った && candidate」であり、
Phase 2 ON の世界でも振動系盤面・裁定欠陥の検出器として意味が継続する（決定経緯 5）。

### 1-5. GetScalar 追加

既存 `term_reason_*` 系の直後に、`episode_just_ended_` ガード付きで 4 キーを追加する。

```cpp
// --- NoLegal candidate 診断（PRD039） ---
if (key == "blocked_drop_on_candidate") {
    if (!episode_just_ended_) return nan;
    return ep_blocked_drop_on_candidate_ ? 1.0f : 0.0f;
}
if (key == "no_drop_timeout_on_candidate") {
    if (!episode_just_ended_) return nan;
    return ep_no_drop_timeout_on_candidate_ ? 1.0f : 0.0f;
}
if (key == "ep_mean_blocked_frames") {
    if (!episode_just_ended_) return nan;
    return last_ep_mean_blocked_frames_;
}
if (key == "ep_max_blocked_frames") {
    if (!episode_just_ended_) return nan;
    return static_cast<float>(last_ep_max_blocked_frames_);
}
```

### 1-6. metrics 設定追加（DropMerge.txt）

train 側は 42_env の 20 番台を candidate 診断ブロックとして新設、EMA は未使用の 70 番台を使う。
eval 側は eval1 / eval2 で共通に空いている 77 / 78 と 86 / 87 を使う（run-length 診断は train のみ）。

```ini
# --- NoLegal candidate 診断 (PRD039 PH1) ---
M.[42_env/21_blkdrop_cand_mean]     = $env mean.blocked_drop_on_candidate @train
M.[42_env/22_timeout_cand_mean]     = $env mean.no_drop_timeout_on_candidate @train
M.[42_env/23_ep_mean_blocked_frames] = $env mean.ep_mean_blocked_frames @train
M.[42_env/24_ep_max_blocked_frames]  = $env max.ep_max_blocked_frames @train
M.[42_env/71_blkdrop_cand_mean_ema] = $env mean.blocked_drop_on_candidate @train $ema ema_alpha:0.001
M.[42_env/72_timeout_cand_mean_ema] = $env mean.no_drop_timeout_on_candidate @train $ema ema_alpha:0.001

M.[51_eval1/77_blkdrop_cand_mean]     = $eval.[eval1] @episode_end $env mean.blocked_drop_on_candidate
M.[51_eval1/78_timeout_cand_mean]     = $eval.[eval1] @episode_end $env mean.no_drop_timeout_on_candidate
M.[51_eval1/86_blkdrop_cand_mean_ema] = $eval.[eval1] @episode_end $env mean.blocked_drop_on_candidate $ema ema_alpha:0.01
M.[51_eval1/87_timeout_cand_mean_ema] = $eval.[eval1] @episode_end $env mean.no_drop_timeout_on_candidate $ema ema_alpha:0.01

M.[52_eval2/77_blkdrop_cand_mean]     = $eval.[eval2] @episode_end $env mean.blocked_drop_on_candidate
M.[52_eval2/78_timeout_cand_mean]     = $eval.[eval2] @episode_end $env mean.no_drop_timeout_on_candidate
M.[52_eval2/86_blkdrop_cand_mean_ema] = $eval.[eval2] @episode_end $env mean.blocked_drop_on_candidate $ema ema_alpha:0.01
M.[52_eval2/87_timeout_cand_mean_ema] = $eval.[eval2] @episode_end $env mean.no_drop_timeout_on_candidate $ema ema_alpha:0.01
```

- 実装時に既存 ID との衝突を最終確認すること（2026-07-22 時点で上記はすべて空き）。
- 既知の不整合（eval1=`75_tr_nolg` / eval2=`76_tr_nolg` の ID ズレ、eval1 `84_tr_timeout_max_ema` の
  max/mean 命名齟齬）には**触らない**。

### 1-7. 挙動不変の保証

Phase 1 の変更は「観測の追加」のみである。次を保証すること。

- 全構成（direct_noop 以外の action_mode 含む）で、遷移・報酬・done/truncated・終了理由・obs・
  RNG 消費列が Phase 1 適用前と一致する。
- 追加コードは const 評価とカウンタ更新のみで、Box2D world / dropper / merge 状態へ書き込まない。

### 1-8. テスト方針

現状 `DropMergeEnv_test.cpp` には Env 名付き MaxStep ログのテストしかない。本 Phase で
詰み盤面の回帰基盤を整える。**production 本体へ test-only API を追加しないこと**（AGENTS.md）。

1. **interval union coverage の純関数化（結果不変リファクタ）**
   `hasClearSpawnXInRange()` の union 判定部（intervals 収集後のソート〜coverage 走査）を、
   `anet::rl::env::drop_merge` 名前空間の free 関数へ切り出す（無名 namespace は使わない）。

   ```cpp
   /// blocked 区間の union が [x_min, x_max] 全体を覆うか（覆っていれば clear な x が無い）
   bool DoBlockedIntervalsCoverRange(
       std::vector<std::pair<float, float>>& blocked_intervals, float x_min, float x_max);
   ```

   宣言は `DropMergeEnv.hpp` の同 namespace へ置き、テストから直接呼ぶ。
   単体テスト観点: 空 intervals / 単一区間で全覆い / 端に gap / 中間に gap / 重複・連結区間 /
   境界一致（covered_until == x_max）。

2. **極小 config fixture（public 経路のみ）**
   公開 config だけで決定的に詰み盤面を作る。fixture helper はテスト側の関数として共通化する。

   ```text
   方針: box_width / box_height を縮小し、drop_probs を単一 rank に固定、
         drop_noise = 0、spin_noise = 0、seed_mode = fixed、use_instant_drop = true で
         DROP 系列を数手実行して詰み盤面へ到達させる。
   注意: 全 rank で half_w - r - 0.01 > 0 を保つこと（1-1 の防御はあるが、fixture 自体が
         「置ける fruit が存在する」前提を壊さないため）。
   unsettled 持続が必要なケース: restitution を大きく（例 0.9〜1.0）、damping = 0 にして
         fruit を跳ね続けさせ、isWorldSettled() == false を維持する。
   ```

3. **必須回帰テスト**

   ```text
   a. 詰み + settled 盤面 + NOOP
      → 既存どおり NoLegalDrop（Phase 1 リファクタの回帰確認）
   b. 詰み盤面 + DROP
      → SpawnBlocked かつ blocked_drop_on_candidate == 1
   c. legal あり盤面（空盤面）+ NOOP × no_drop_timeout_steps
      → NoDropTimeout かつ no_drop_timeout_on_candidate == 0
   d. 詰み + unsettled（跳ね続け）盤面 + NOOP × no_drop_timeout_steps
      → 現行どおり NoDropTimeout かつ no_drop_timeout_on_candidate == 1
      （= 本 PRD が問題とする「詰み漏れ」経路そのものの固定化。Phase 2 で挙動が変わる）
   e. ep_mean/max_blocked_frames は smoke レベル（非負・妥当な範囲）で可
   ```

---

## Phase 1b 仕様（terminal 側診断の追補・挙動不変）

現行 PH1 集計は「candidate が途切れて解消した run」のみを記録し、episode 終端まで継続した run を
捨てる（1-2 の契約どおり）。このため次が欠測している。

- NoLegalDrop 受理までに blocked 状態で待った frame 数（= settled 待ちの実コスト）
- NoDropTimeout 時点で blocked が何 frame 継続していたか（= 受理失敗の深さ）

追加仕様:

- episode 終了時（`done || truncated` ブロック、既存 `last_ep_*` 確定の隣）で、継続中の
  `blocked_candidate_frames_` を `last_ep_terminal_blocked_frames_` へ確定する
  （0 = 終端時に blocked ではなかった）。
- `GetScalar()` に `episode_just_ended_` ガード付きで 2 キーを追加する:
  `ep_terminal_blocked_frames`、および既存 member `ep_blocked_run_count_` の公開キー
  `ep_blocked_run_count`。
- 終端理由は既存 `term_reason_*` one-hot と同一タイミングで読めるため、メトリクス側で
  終端理由別に切り分けられる。
- metrics 定義（DropMerge.txt）は PH1 1-6 の様式に倣い、train 42_env の 25/26 等・eval1/eval2 の
  空き ID を使う（実装時に衝突を最終確認）。
- 注意: game_over 終端（SpawnBlocked/Overflow）は物理ループを break した frame の candidate 評価を
  経ていないため、値に ±1 frame の誤差がある（診断用途では許容）。
- 挙動不変の保証は PH1 1-7 と同一基準。
- PH2 ON の世界では、NoLegalDrop episode の本値が「fast-path 受理なら小さい値、persistence 受理なら
  ≈ N」の二峰になる（PH2 受入のサニティチェックに使う）。

---

## Phase 2 仕様（persistence 受理の導入・fast-forward なし）

Phase 1b の terminal 側分布で N の妥当性を確認して default を最終確定する（実装着手は並行可）。

### 2-1. 設定キー（DropMergeEnvConfig）

```cpp
bool use_no_legal_adjudication = false;  ///< NoLegal 受理に blocked-persistence 上限保証を追加するか
int no_legal_min_blocked_frames = 60;    ///< 受理 horizon N（DROP 不能の連続物理 frame 数）= 物理 1.0 秒。仮値、PH1b で確定
```

- `use_no_legal_adjudication = false`（default）で現行挙動を**厳密に**維持する。
- バリデーション（`ANET_SYSTEM_ERROR` で fail-fast、意味破綻設定の黙殺禁止）:
  - `no_legal_min_blocked_frames >= 1`
  - `use_no_legal_adjudication && use_no_drop_timeout_gameover && no_drop_timeout_steps > 0` の場合、
    `no_legal_min_blocked_frames < no_drop_timeout_steps` を要求する
    （instant_drop では blocked NOOP 1 step = 1 frame のため、N がショットクロック以上だと
    詰み盤面が受理前に timeout に食われ、罰導入後は正しい投了が敗着扱いされる不条理が
    構造化するため）。
- 裁定の別個の上限 frame 設定は**設けない**（決定経緯 5）。
- `settle_max_steps` / `settle_velocity_threshold` / `settle_angular_threshold` は流用**しない**
  （これらは `use_settle_after_drop` の責務に残る）。
- fast-forward は行わないため、v1 で検討した物理ループ側の変更・`max_sim_steps` との干渉は存在しない。

### 2-2. 受理判定の拡張（終了判定の OR 化）

変更は終了判定行のみ。物理ループ・カウンタ更新（1-2）・報酬・obs は一切変更しない。

```cpp
// 盤面いっぱいでのNOOP判定
// 受理は OR 構成（決定経緯 9）:
//   fast-path: candidate && isWorldSettled()（現行どおり。静止した詰みは最初の NOOP で即受理）
//   上限保証: DROP 不能が N 連続物理 frame 続いたら、settle の有無に依らず受理。
//             settled は agent の obs に無い速度場依存で受理タイミングが agent から予測できないため、
//             「置けない盤面で N 回 NOOP すれば必ず終わる」という予測可能な上限を契約に加える。
const bool no_legal_adjudicated = config_.use_no_legal_adjudication &&
    blocked_candidate_frames_ >= config_.no_legal_min_blocked_frames;
const bool no_legal_drop_terminal = !game_over_ && is_noop_action &&
    (isNoLegalDropState() || no_legal_adjudicated);
```

- done/truncated / penalty / log の扱いは現行 NoLegalDrop と同一
  （done=true、truncated=false、game_over_ は立てない、game_over_penalty なし）。
- 受理 log（英語 verbose、PRD038 の `log` 使用）は fast-path / persistence のどちらで確定したかを
  区別する（例: `"Episode done: no legal drop persisted for N frames."`）。
- OFF 時は `isNoLegalDropState()` のみ = 従来挙動と厳密一致。
- ON 時も fast-path は残るため、settled な詰み（多数派）の受理タイミングは現行から変わらない。
- NoDropTimeout / MaxStep の判定・順序は ON / OFF とも変更しない。GameOver（Overflow 等）が
  NoLegal に優先する現行順序も不変。

### 2-3. カウンタの意味と境界条件

- `blocked_candidate_frames_` は PH1 の per-frame 更新（1-2）をそのまま使う。RL step を跨いで持続し、
  DROP（busy ガード）・merge 等のガード落ち・legal 復活のいずれでも 0 に戻る。カウント定義は
  PH1 の解消 run 集計と同一であり、受理と診断で定義が一致する。
- instant_drop の NOOP は 1 step = 1 frame のため、実運用では「blocked NOOP を N 連打した step」で
  受理される。DROP step 末尾の数 frame で candidate が成立した場合はその分が持ち越されるが、
  「連続 N 物理 frame」という定義に対して一貫した挙動である。
- 遅発 block（NOOP 連打の途中で物理微動により最後の合法が塞がるケース）では、ショットクロックの
  残りが N 未満だと timeout が先に成立し得る。頻度は `no_drop_timeout_on_candidate` で観測される
  （恒常的非ゼロなら N か裁定ロジックを見直す。決定経緯 5 の異常検出器の役割は不変）。
- ON 時に詰みが NoLegal で先に確定するため、NoDropTimeout は自然に
  「legal があったのに置かなかった」へ純化される。timeout 側の再分類コードは書かない（決定経緯 5）。

### 2-4. 期待効果

- 詰み + settled 盤面: 現行どおり最初の NOOP で受理（fast-path、変化なし）。
- 詰み + unsettled 盤面（Phase 1 テスト d のケース）: 高々 N step で NoLegalDrop に到達し、
  NoDropTimeout へ漏れない。受理の上限が agent から予測可能になる。
- 投了確定までの無駄 step は最大 `no_drop_timeout_steps` → 最大 N に短縮（unsettled 詰みのみ）。

### 2-5. timeout 罰の扱い（コード変更なし）

timeout 罰は既存 config キー `no_drop_timeout_gameover_penalty` の値変更のみで、本 PRD の
実装対象ではない。契約上の位置づけだけを規定する。

- 意味論: NoDropTimeout を「他の GameOver と同格の敗着」とする（ADR 0014）。現行の done+0 が
  作る guaranteed-0 の避難港（停滞が報酬合理的になる構造）が消え、停滞（NEET）が報酬上
  不利になる。`no_drop_timeout_ratio` は obs に既在のため、agent はクロック接近を観測して
  回避を学習できる。
- 初期値は game_over_penalty と同値（現行 −10）を起点とするが、独立の自由値であり、
  同値規律・自動参照・warn/fail-fast は設けない（決定経緯 8/10）。
- 有効化は受理安全網（`use_no_legal_adjudication = true`）と併せて行う意図。OFF+罰非 0 の
  構成も黙って許容する（決定経緯 10）。
- 値の決定・適用時期・Run 設計はチューニング領域で PRD 範疇外。

---

## 受入条件

### Phase 1

1. 全既存テストが緑のまま。
2. 挙動完全不変: 同 seed・同 action 系列で、Phase 1 適用前後の遷移・報酬・終了理由・
   done/truncated が一致する（1-7）。
3. 新 GetScalar 4 キーが `episode_just_ended_` ガード付きで返り、1-8 の必須テスト a〜d が緑。
4. `DoBlockedIntervalsCoverRange`（名称は実装時確定）の単体テストが緑。
5. Profile 実測（`DropMergeEnv::Step` / `isNoLegalCandidateState` / `hasAnyLegalDropForCurrentFruit` /
   `hasClearSpawnXInRange`）で、Step への追加コストが誤差レベル（目安: Step 全体の数 %未満）であること。

### Phase 1b

1. 挙動完全不変（PH1 1-7 と同一基準）。
2. `ep_terminal_blocked_frames` / `ep_blocked_run_count` が `episode_just_ended_` ガード付きで返る。
3. テスト: 詰み盤面で NoLegalDrop 終端した episode で terminal 値 > 0 /
   legal あり盤面の NoDropTimeout で 0 / blocked が解消してから終端した episode で 0。

### Phase 2

1. `use_no_legal_adjudication = false`: Phase 1b 実装と同 seed 同系列で挙動一致（新規 config キーの
   読み込み以外に差分なし）。
2. `use_no_legal_adjudication = true`:

   ```text
   a.  詰み + settled 盤面 + NOOP → 現行どおり最初の NOOP step で NoLegalDrop
       （fast-path。done=true / truncated=false / game_over_penalty なし）
   b.  詰み + unsettled（跳ね続け）盤面 + blocked NOOP × N → N 到達 step で NoLegalDrop
       （Phase 1 テスト d の期待値が NoDropTimeout → NoLegalDrop へ変わる）
   b'. 同盤面で N-1 step 時点では終了しない（境界確認）
   c.  blocked が N 未満で legal が復活する盤面 → episode 継続、カウンタはリセットされ
       PH1 の解消 run 集計に載る
   d.  legal あり盤面 + NOOP 連打 → 従来どおり NoDropTimeout（no_drop_timeout_on_candidate == 0）
   e.  バリデーション: N < 1、および gameover=true 時の N >= no_drop_timeout_steps で fail-fast
   f.  SpawnBlocked / MaxStep / Overflow の既存挙動・メトリクスを壊さない
   ```

3. Run 観測の観点（**参考**。受入条件ではない。Run 設計・A/B はチューニング領域で PRD 範疇外）:

   ```text
   - NoLegalDrop episode の ep_terminal_blocked_frames が「fast-path 受理なら小、
     persistence 受理なら ≈ N」の分布になる（受理経路のサニティ）
   - no_drop_timeout_on_candidate ≈ 0 の維持（非ゼロ恒常なら振動系 or 裁定欠陥。
     罰導入後なら「正しい投了が罰されている」サイン）
   ```

## 性能確認

candidate 判定と persistence カウンタは PH1 で既に毎 frame 走っており、PH2 の追加は終了判定行の
整数比較 1 個、PH1b の追加は episode 終了時の代入のみ。追加コストは実質ゼロだが、PH1 と同じ
Profile 項目（`DropMergeEnv::Step` / `isNoLegalCandidateState` / `hasAnyLegalDropForCurrentFruit`）の
スモーク確認は行う。

## 対象外・変更しないこと

- Overflow 判定・`game_over_grace_step`・`drop_noise` の細部変更。
- `noop_penalty` / `time_penalty` / `game_over_penalty` の値変更。
  `no_drop_timeout_gameover_penalty` の値もチューニング領域（PRD は 2-5 の契約上の位置づけのみ規定）。
- fast-forward 裁定（v1 2-2。決定経緯 6 により廃止）。
- Run 設計・A/B 計画・罰値の探索（ハイパラ探索領域。受入条件 3 は参考情報のみ）。
- `no_drop_timeout_steps` を短くする対応（レビューで否決済みの迂回策）。
- Agent / DQN / UQE / Replay 側の変更（`done=true` の既存 terminal 処理をそのまま使う）。
- `avoidable_spawn_blocked` / `noop_on_candidate` メトリクス（将来オプション。導入する場合、
  前者は「確率的に成功可能な action と確実に安全な action の区別」が必要になる点に注意）。
- timeout 時再分類・裁定上限・`no_legal_adjudication_timeout`（決定経緯 5 により不採用）。
- eval1 / eval2 の既存メトリクス ID ズレ（75/76）や命名齟齬（84）の修正。
- eval 専用処理にはしない。train / eval とも同じ判定・同じメトリクスを使う。

## 実装時の注意

- AGENTS.md 規約に従う: 日本語コメント / 英語 runtime log / 局所 diff /
  `VsDevCmd.bat` 経由の `cmd /s /c 'call ... && cmake --build --preset x64-Debug'` で検証 /
  テストは fixture・public 経路で組み、本体へ test-only API を追加しない。
- コメントには「何をしているか」に加え、candidate / persistence / 裁定 horizon の意図
  （なぜ速度でなく継続 frame なのか）を要点だけ残す。
- 新規ログは `log.verbose()`（Env 基底の prefix 付き Logger、PRD038 準拠）で英語。
  例: `"Episode done: no legal drop persisted for N frames. ..."`。
- Phase 1 実装後・Phase 2 実装後の各時点で、変更ファイル一覧と実行した検証コマンドを報告する。
- Phase 2 の default `no_legal_min_blocked_frames = 60`（= 物理 1.0 秒）は仮値（決定経緯 7）。
  Phase 1b の terminal 側分布（`ep_terminal_blocked_frames`、特に timeout 終端時の値）と
  解消 run 分布の間に N を置けているかを確認して確定し、根拠を実装メモ
  （`039_*_20impl.md`）に記録する。
