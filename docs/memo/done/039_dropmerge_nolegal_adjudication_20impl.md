# PRD039 Phase 1 NoLegal candidate 診断 実装メモ

## 概要

`039_dropmerge_nolegal_adjudication_10prd.md` の Phase 1 のみを実装する。
NoLegal 判定、報酬、Observation、`done` / `truncated`、RNG 消費列は既存の有効設定で不変とし、Phase 2 の設定と blocked-persistence 裁定は導入しない。

## 主な変更

- 安定判定を含まない `isNoLegalCandidateState()` を抽出し、既存 NoLegal 判定を `candidate && isWorldSettled()` のまま維持する。
- NoLegal candidate の連続物理 frame 数を追跡し、candidate が解消した run の平均・最大だけを episode metric として確定する。
- action 適用前の candidate を使う `blocked_drop_on_candidate` と、NoDropTimeout 成立時の `no_drop_timeout_on_candidate` を追加する。
- `GetScalar()` に `blocked_drop_on_candidate`、`no_drop_timeout_on_candidate`、`ep_mean_blocked_frames`、`ep_max_blocked_frames` を追加する。
- blocked interval の union 判定を `DoBlockedIntervalsCoverRange()` へ抽出し、同 namespace の公開関数としてテストする。
- `DropMerge.txt` に PRD 指定の train / eval metrics を追加し、既存の別目的の未コミット変更は保持する。
- `CONTEXT.md` に `NoLegal candidate` と `blocked persistence` を実装詳細なしで追加する。ADR は追加しない。

## テスト

- Public interface / surface: `DropMergeEnv::Step()`、`SingleState.done` / `truncated`、`DropMergeEnv::GetScalar()`、`DoBlockedIntervalsCoverRange()`、`DropMerge.txt` の metrics 定義。
- 優先 behavior: blocked盤面でのDROP診断、interval union、従来のsettled NoLegal、legal盤面のtimeout、unsettled blocked盤面のtimeout診断、解消runの平均・最大。
- TDD 順序: tracer bullet から始め、各 behavior を 1テスト追加 → RED確認 → 最小実装 → GREEN確認の縦スライスで進める。共通fixtureとhelperの整理はGREEN後だけ行う。
- production 本体へ test-only API を追加しない。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target DropMergeEnv-test'
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

既存 Profile 基盤で `DropMergeEnv::Step`、`isNoLegalCandidateState`、`hasAnyLegalDropForCurrentFruit`、`hasClearSpawnXInRange` の追加コストも確認する。長時間の診断 Run、Phase 2 の N 決定、OFF / ON A/B は別作業とする。

## 前提

- 挙動不変の対象は既存の有効設定とし、配置可能幅が負になる設定の `std::clamp(lo > hi)` は防御的に回避する。
- Phase 2、timeout 再分類、裁定上限、Agent / Replay、報酬値、既存 metrics ID / 名称の不整合は変更しない。
- ユーザーの無関係な未コミット変更を保持する。
