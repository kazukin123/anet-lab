# PRD 059 横断整合チェック 実装メモ

- 作業日: 2026-08-24
- 前提: LunarLander(見本)/ Atari(25impl)/ DropMerge(24impl)/ CartPole・GridMaze・ImageCls(27impl)の素材化+幹化が完了した状態からの横断確認と整合作業。
- 正本: `059_config_concept_tree_alignment_10prd.md` / ガイド `..._30mat_guide.md`

## 実施した整合作業

| 作業 | 内容 | 等価性 |
|---|---|---|
| **RainbowAgent.baseline の `@` 化** | agent.txt の定義 33 行+チェーン(`RainbowAgent.$ = @baseline` 相対化)。PH1a の `@` 化(DefaultDQNAgent のみ)からの漏れ | 全 7 overlay の dump から 33 行が純化(意図分) |
| **ALGO 素材の共通化** | LL / Atari / DropMerge に完全同一で 3 重複していた `DefaultDQNAgent.@qr / @iqn` の 4 行を **agent.txt へ 1 本化**、各 env には参照コメント 1 行を残置。配線実体(`net.@qr / @iqn`)は env 所有のまま(次元等が env 依存のため) | **dump 完全不変**(素材の置き場所は解決結果に影響しない=ADR 0030 の帰結を実証) |
| **stale コメント掃除** | CartPole の `DefaultDQNAgent.trunk / trunk2` ラダー(定義消滅済み参照)、GridMaze の `GridMazeEnv.baseline`(未定義参照)と `test1` コメント群。DropMerge 枠(24impl)の「未定義参照コメントは削除」の先例に統一 | dump 不変 |
| **`@vars` の root 化**(ユーザー裁定) | `app.@vars` → `@vars`(LL / Atari / GridMaze / ImageCls+ガイド、定義・`${}` 参照・コメントの同一文字列一括置換)。`@vars` はチェーンで選択されない `${}` 専用の定数置き場であり所有者 prefix が不要 — 059 §6.2 / D7 の元例(root `@vars`)への回帰。app 配下が必要なのはチェーンで選ぶ named 素材(`app.@budget50m` 型)だけ | 残存 0 件+定義⊇参照の対応を grep で機械確認。素材キーは dump に出ず、定義と参照が同時に変わるため実効値への影響経路なし(rename 漏れは未定義参照 fail-fast が検出)。`[config]` 91 全緑(root `@vars` の `${}` 波及は既存契約テストがカバー) |

検証: 一時テスト(`[tempdump-x]`、環境変数切替、**検証後削除済み**)で全 7 overlay(GridMaze_muzero 込み)の before/after dump をキー単位比較 — **valueDiff=0 / newKeys=0 / 消滅は RainbowAgent.baseline の 33 行のみ**(7 overlay 全てで同一結果)。

## 機械確認の結果

**チェーン RHS の無印項 全数分類**(全 config の `.$ =` 行から抽出):

| 分類 | 項 | 判定 |
|---|---|---|
| 解消済み | `DefaultDQNAgent.trunk/trunk2`、`GridMazeEnv.baseline/test1`(stale)、`RainbowAgent.baseline` | 本作業で掃除・`@` 化 |
| 意図された無印(選択の源) | `app.online / app.batchrun` | mode 選択の源 prefix。`@` 化は呼び出し側(bat/CLI)波及が大きく 059 で未裁定 → 残存(下記提案) |
| 意図された無印(素材) | `backend.deterministic / non-deterministic` | common.txt 所有 → 残存(下記提案) |
| 対象外 | `MuZeroAgent.baseline`(agent.txt) | MuZero 保留中は触らない裁定(D19/§10)と一貫させ据置 |

素材名の横断整合: ALGO=`@qr/@iqn`(agent.txt 共通)、Random=`@random`(3 env 同名・**中身は env 依存で相違**: Atari は `eps_decay_steps=0` あり/`replay_capacity` なし。共通化は見送り=命名統一のみで十分)、BODY=`net.branch.@<Name>`(Atari/ImageCls/DropMerge で同形式)、BF16=`@bf16`、予算=`app.@vars`(LL/Atari/GridMaze/ImageCls。DropMerge は online/batch 別値のため対象外の判断=24impl)。

## 指標 3(inspect_run 突合削除)の判定

env 素材(プリセット/ALGO/配線/BODY/Random/Agent baseline)の `@` 化は**完了**。dump に残る ③ 無印素材は **mode(`app.online/batchrun`)・backend・metrics 系素材(`metrics.scalar.baseline/min/full` 等・`image.*`)** のみ。これらは「機械置換で済まない箇所」(059 §8.2)として当初から個別対応扱いであり、選択の適用有無は resolution.json の selections で既に判別可能。**突合コードの削除可否はユーザー判断**(実務上は resolution 参照で足りるが、③ の完全消滅を待つなら metrics/mode/backend の `@` 化が先)。

## 提案リスト → **全 7 件裁定済み(grill 2026-08-25。監査記録は 059 §11.2)**

裁定結果: 1=keep(両スタイル許容+層供給正式許容、§2.2 明文化済み)/ 2=defer(ゲート=起因事故 or HEAD/exploration 軸導入時)/ 3=Phase 1.5 廃止・backend+metrics @ 化は即時実施(D26)・mode は無印正式仕様化(D25)/ 4・5・6=defer(ゲートは 059 §10)/ 7=defer(本記録のみ、再発時診断)。追加裁定: 上書き層の直交 2 軸化(D20 改: R→A2 / X→A3 / O→M2、全層番号付き)+`@bf16`/`@random` の agent.txt 共通化(全部入り統一)。実施記録は 29impl。以下は裁定前の原文。

1. **幹規律の明文化**: 幹のスタイルが Atari=差分幹(既定との差分+eval 系のみ、4〜7 行)と DropMerge=フル署名幹(backend/env/agent/BODY/R/X まで全部、約 17 行)に分かれた。両立は可能だが、増殖規律(D4)と「幹が R/X 層キーを供給する形」(dump に素材キーと解決済み leaf の両方が残る=24impl 103 行)の扱いを 059 §2.2 に明文化する価値がある。`run.$` チェーン合成(`run.$ = run.@a > run.@b`)による軸分割も含めて。
2. **baseline からのアルゴ的キー抽出**: 現状 `@baseline` に quantile 系キーが残る(過渡期判断)。抽出には GridMaze / CartPole のチェーンへ `@qr` を明示挿入する前準備が必要(現状は baseline 供給の qr に依存)。概念純度は上がるが必須ではない。
3. **mode / backend / metrics 素材の `@` 化(Phase 1.5 候補)**: ③ の完全消滅に必要な残り。mode は bat/CLI 波及(`app.$=app.batchrun`)、metrics は「素材と実効 tag」の個別対応(059 §8.2)を伴う。
4. **`AtariIQNTauProj512` の名前**(25impl 申し送り 5): 値参照化により `512` が実態とずれうる。カタログ名で structure 文字列からも引かれるため改名影響大 → 実害発生時に判断。
5. **`@nature` の糖衣/直書き混在**(Atari 内スタイル): 機械 rename 行と新規行の差。実害なし、気になるなら Atari 枠で統一。
6. **MuZeroAgent.baseline の `@` 化**: MuZero 再着手時に net 移設(§3.5)と同時に実施。
7. **flaky テスト 1 件の観察**: 全 core 実行で 1 回だけ 3 件目の失敗を観測(455 中 452。assertions 総数も 3 減 = 乱数依存)。直後の再実行では既知の ReplayBuffer 2 件のみに収束し再現せず。テスト名は未特定。再発したら診断する。

## 検証結果

- 全 7 overlay 等価性: valueDiff=0 / newKeys=0 / 消滅は RainbowAgent.baseline の 33 行のみ(機械証明)
- 一時テスト(`[tempdump-x]`)削除+最終ビルド(anet-core-test / AnetRLRunner)成功。exe 上に TEMP 系テストの残存なし(`[tempdump]` / `[tempdump-cgi]` / `[tempdump-x]` すべて 0 件)
- `[config]`: 91 test cases / 608 assertions、全緑
- 全 core: 455 test cases 中 453 passed / 2 failed。失敗は既知の ReplayBuffer 2 件(`episode_start without done` 系)のみ。提案 7 の flaky は本実行では再発せず
- `inspect_run_test.py`: 53 tests OK
- `resolve_workspace_test.ps1`: passed
- `git diff --check`: クリーン
