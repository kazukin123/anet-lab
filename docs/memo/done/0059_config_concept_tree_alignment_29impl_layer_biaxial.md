# PRD 059 上書き層の直交 2 軸化+③素材 @ 化完遂 実装メモ

- 作業日: 2026-08-25
- 裁定: grill 2026-08-25(059 §11.2 に監査記録。裁定待ち 7 件の決着+追加構想)
- 正本: 059 §4.1(D20 改)/ §2.2(幹規律)/ D25(mode 無印仕様化)/ D26(backend・metrics @ 化)

## 実施内容

### 1. 上書き層の直交 2 軸 rename(D20 改)

**文字=対象(A=Agent / E=Env / M=Metrics / P=app)× 番号=恒久度(大番号ほど揮発的・チェーン右=強い)**。写像:

| 旧 | 新 | 意味 |
|---|---|---|
| A | A1 | Agent の環境依存・定着値 |
| R | A2 | Agent の実験中の値 |
| X | A3 | Agent のその場の A/B 軸 |
| E | E1 | Env の上書き |
| M | M1 | Metrics の上書き |
| O | M2 | Metrics の 2 段目(用途文字 O を廃止) |
| P | P1 | app の上書き(空層) |

- 対象: 全 env config+共通(agent/nn/nn_cnx/metrics_scalar/metrics_image/common/_main)+`DropMerge_optuna.txt`+`GridMaze_muzero.txt`(層は全体規約のため muzero も対象)+`apps/12_batch_run_atari5.bat`(`X.`→`A3.`)
- 置換パターン: チェーン項(`> A` 等、単語境界+可変スペース)/ 定義行頭(コメント `#` 付き含む)/ 幹供給行(`run.@xxx : R.`→`: A2.`)/ **`${}` 内の層参照**(`${E.game}`→`${E1.game}` — 初回 after 検証の fail-fast が取りこぼしを検出し、パターンを追加した)
- 無印単文字は廃止(A=A1 の別名は残さない)。段数は現行写像で、4 段目以降は必要時に右へ追加

### 2. ③無印素材の @ 化完遂(D26 — これで指標 3 が名実達成)

- **backend**(common.txt): `backend.@deterministic` / `backend.@non-deterministic`(単語境界置換で `backend.deterministic_algorithms` 等の実効キーは不変)
- **metrics.scalar**(metrics_scalar.txt): `@baseline` / `@iqn_search_p0` / `@min` / `@full`
- **image**(metrics_image.txt): `image.thg.@per-prio` / `image.phm.@visit-maxq` / `image.phm.@per-prio` / `image.phm.@s01` / `@s02` / `image.shm.@qmax` / `@s01` / `@s02` / `image.@conv2d` / `image.@test`(所有者 prefix 配下の末端名 `@` 形)
- **net.body**(nn.txt): `net.body.@MLP` / `@MLP2` / `@MLP3` / `@MLP_RES` / `@Conv1D` / `@Conv2D` / `@ResNet` / `@ViT_*` の 10 素材 — 検証中に「③の最後の取り残し」として発見し、D26 の趣旨(③完全消滅)に含めて実施
- 参照(チェーン RHS・幹供給行・コメントラダー・GridMaze の `image.phm.@per-prio` 上書き)も同一文字列置換で追随

### 3. 共通 agent 素材の agent.txt 吸い上げ

- **`@bf16`**: 3 env の同一定義(learner 3 行+policy 系コメント選択肢)を agent.txt へ 1 本化。env 側は参照コメント 1 行(LL は force_fp32 部が残るため章維持)
- **`@random`**: **全部入りで統一**(policy_type / eps 1.0 / `eps_decay_steps=0`(旧 Atari のみ)/ `replay_capacity=256,000`(旧 GridMaze・DropMerge のみ)/ warmup / alpha=0)して agent.txt へ 1 本化。学習しない env 天井実測用のため、差分値の統一は全 env で挙動に影響しない

## 等価性検証(写像込み機械証明)

一時テスト(`[tempdump-x2]`、env var 切替、**検証後削除済み**)で全 7 overlay の before/after を取得し、**旧→新の層写像を before キーへ適用して比較**:

| 結果 | 全 7 overlay 共通 |
|---|---|
| 値差分 / 欠落 / 新規 | **0 / 0 / 0** |
| `@` 化で dump から消えた素材 | **281 キー**(backend 6+metrics.scalar 系+image 系+net.body 10 素材) |
| ③残存 | **なし**(after dump の機械チェック) |

- 初回 after 検証で `${E.game}` 参照(ユーザーが直前に導入した run_name の値参照)の rename 漏れを**未定義参照 fail-fast が検出** → `${}` 内層参照のパターン追加で解消。「静かに壊れない」設計が移行検証でも機能した実例。
- C++ コードに層名(`"A."` / `"R."` 等)のハードコードが無いことを grep で確認(rename はコード挙動に影響しない)。

## 残存(仕様内・裁定済み)

- mode(`app.online / app.batchrun`)= 選択の源として無印が正式仕様(D25)
- `MuZeroAgent.baseline` / `metrics.scalar.muzero`(GridMaze_muzero)= MuZero 保留裁定の一部(§10。再着手時に @ 化)
- コメント内の未定義参照(`image.shm.01`、CartPole 末尾 MEMO の `agent.base` 例文、muzero の `GridMazeEnv.baseline` コメント)= 実害なしの据置

## 検証結果

- 一時テスト削除+最終ビルド成功。exe 上に `[tempdump-x2]` の残存なし
- `[config]`: 91 test cases / 608 assertions、全緑
- 全 core: 455 test cases 中 453 passed / 2 failed(既知の ReplayBuffer 2 件のみ)
- `inspect_run_test.py`: 53 tests OK / `resolve_workspace_test.ps1`: passed
- `git diff --check`: クリーン(rename で変更行に乗った既存の末尾空白 8 行は除去した)
