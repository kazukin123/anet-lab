# NN ブロック別 勾配メトリクス — 背景・目的・基本方針（暫定）

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> **暫定版（2026-06-29 ダンプ）**: 背景・目的・基本方針までを固めた段階。実装の現コードでの実現方法（記録経路の拡張・CPU 同期・update比率の Δw 取得など）は **未調査＝次段階**。番号 999 はバックログ仮番号。
> 動機: NN 内部の学習挙動が grad_norm（全体一括）でしか見えず、A/B の結果が eval に出るまで「どのブロックで何が起きたか」が分からない。

## Context（背景・目的）

### 背景
- 現状、NN 分析は grad_norm など **全体集約スカラー**で傾向は掴めるが、**内部のブロック単位の挙動が見えない**。
- Conv2d 可視化等はあるが、デモ・概要確認向けで、定量的な意思決定には使えない。
- 過去に簡易コードで確認したところ、grad_norm の大半が **TransformerEncoder の FFN 部**由来だった（観察止まり。定量・定常監視はされていない）。
- 既存 `ForeachGradNorm`（`nn_util.cpp:37`）は `at::_foreach_norm` で **per-tensor の L2 norm リストを作ってから** `stack().norm(2)` で 1 本に潰している。ブロック別の素材は既にこの中間結果にある。

### 目的（北極星）
> 層／ブロック単位で「学習の健全性」と「寄与構造」を観測可能にし、チューニング時に "どこを触るか" を **最終 eval を待たずに** 決められるようにする。

測定軸は「eval が動くか」ではなく「**意思決定を下せるようになるか**」に固定する（費用対効果のぶれ止め）。**本丸は RL**。ImageCls は知見・フレームワーク拡張・モチベ確保のための寄り道。

### 分析目的と優先度
| 優先 | 分析目的 | 下せるようになる判断 | 主メトリクス |
|---|---|---|---|
| **P0** | 学習ダイナミクスの健全性診断 | どの層に手を入れるか／lr・clip 妥当性。停滞層・暴走層の特定 | grad_norm(A), update比率(D) |
| **P0** | 寄与・容量配分の把握 | 層の増減・幅変更のアーキ判断。FFN支配が絶対値の artifact か実体かの判別 | grad寄与率(B), weight_norm(C), D |
| **P1** | A/B の層レベル因果追跡 | 「なぜ良く／悪くなったか」の発生源特定 | 上記を run 間で重ねる |
| **P2** | 正則化の効き測定 | WD/Dropout/DropPath の強弱調整 | weight_norm(C) |
| **P2** | 順伝播の信号フロー診断 | 飽和・死にユニットへの対処 | activation統計(E) ※forward hook |
| **P3** | 破綻の早期検知 | 長時間 run の早期停止 | attn entropy(G), dead率(F) |

## 基本方針（確定した設計判断）

メトリクスのラベル: **(A)** grad_norm 絶対値 / **(B)** grad_norm 寄与率 / **(C)** weight_norm / **(D)** update比率 ‖Δw‖/‖w‖ / (E) activation 統計 / (F) dead 率 / (G) attention entropy。

1. **第一弾スコープ = Tier 1 ワンセット（A/B/C/D）**。いずれも backward 後の **`named_parameters` を 1 回回すループ**で完結し、forward ホットパスに触れない。P0〜P1 はこの Tier 1 で閉じる。Tier 2（E/F/G、forward hook 必要）・Tier 3（optimizer 内部統計）は後段。
2. **(D) update比率を主役級に置く**。grad_norm 絶対値(A)は「パラメータ数の多い層が大きく出る」artifact を含む。weight で正規化した update比率（‖Δw‖/‖w‖ ≒ 実効学習率）の方が層間比較に適し、**FFN 支配が見かけか実体かを判別**できる。A 単独では判断不能。
3. **粒度 = 意味カテゴリ束ね**。`named_parameters()` のフルパス（例 `...layer_0.linear1.weight`）を、**フルパスに対する順序付き正規表現パターン（先勝ち）→ カテゴリ**で振り分ける。
   - 初期カテゴリ: `attn`(self_attn) / `ffn`(linear1+linear2) / `norm`(norm*/ln/bn/gn) / `conv`(conv*/ds_conv) / `value` / `adv` …。
   - **未マッチは末端 module 名のまま 1 グループにフォールバック**（config 可変構造でも壊れない）。
   - 粒度調整はこのパターン表の編集で行う（粗く＝layer に束ね／細かく＝linear1,2 を分割／特定箇所だけ抜く）。**マッピング表が粒度の唯一の制御点**。
   - キーをフルパスにするのは、encoder の `linear` と head の `linear` のような **文脈衝突をパターンで分離**するため（末端名だけでは不可）。
4. **集約は加法性で扱う**。grad_norm / weight_norm / update比率はいずれも **二乗和が加法的**（全体 = √(Σ 末端²)）。末端さえ出せば layer / encoder / head / 網全体は合算で復元できる。第一弾は末端＋主要集約（layer / encoder・head / 網）を **生成側で合算して出す**（Viewer の集約機能に依存しない保険）。
5. **対象網は online / target を個別出力**まで。tag に網軸を持たせる（`<metric>/<net>/...`）。**差分・乖離（‖w_on−w_tgt‖ 等の相関分析）は現状やらない**（オーバースペック、将来）。

## 前提事実（実コード確認済み・クリーンに取得した分のみ）
> 調査途中でツール結果に混線が発生したため、行番号が未確認のもの（online/target net の保持場所等）は本書から除外し、次段階で取り直す。
- `ForeachGradNorm`（`nn_util.cpp:37-46`）= `at::_foreach_norm(grads, 2)` → `torch::stack(norms).norm(2)`。**per-tensor norm が中間結果**として既に存在 → ブロック別 grad_norm の素材になる。
- `CollectDefinedGrads`（`nn_util.cpp:24-35`）= defined grad のみ detach した **名前なし flat list**。→ ブロック別には `named_parameters` で名前付き収集に切り替える必要。
- `FusedAdamW::step`（`nn_util.cpp:98-187`）= **in-place** `_fused_adamw_`。`exp_avg`/`exp_avg_sq` を `state_`（key=param の TensorImpl ポインタ）保持。→ update比率(D)の Δw は in-place 更新ゆえ別途取得設計が要る。
- メトリクス記録: `MetricsLogger`（Singleton）の `LogScalar(tag, step, value)`（`metrics_logger.hpp:124`）。記録は Observer 経由で `BatchUpdateResult::GetScalar(key, index)` を引く（`dqn_based_agent.hpp`）。"grad_norm" キーは `grad_norm_tensor` を `.item()` 同期して返す（同期を optimizer step 後まで遅延する設計）。
- `QValueLearnerBase::Module()` が `torch::nn::Module&` を返す → そこから `named_parameters()` でフルパスが取得可能（粒度マッピングの鍵）。
- NN 命名（`nn_modules.cpp`、`register_module`）: Transformer = `layer_N` 内に `self_attn`/`linear1`/`linear2`/`norm1`/`norm2`、ResBlock = `conv1`/`conv2`/`ds_conv`/`bn`/`gn`、head = `value`/`adv` 等。意味カテゴリ束ねはこの命名に乗る。

## これから詰める論点（次段階・未調査）
1. **粒度マッピングの初期テーブル確定**と供給源（ハードコード／config 外部化）の最終判断。
2. **(B) 寄与率の定義**: `block²/total²`（加法的・合計 1）か `block/total`（合計 ≠ 1）か。
3. **CPU 同期**: 現状 grad_norm は optimizer step 後まで `.item()` を遅延。ブロック分割で同期点が N 倍にならないよう、**全 group の norm を 1 tensor に stack して 1 回で吸い上げる**設計が要る。
4. **(D) update比率の Δw 取得方法**（要検討の本命）: ①step 前後 snapshot（正確・コスト高）／②optimizer state（`exp_avg`/`exp_avg_sq`）から実効 update を再構成／③grad–weight 代理 `‖g‖/‖w‖`（最安・Adam 適応を無視）の三択。FusedAdamW が in-place なのが効く。
5. **記録経路の拡張**: 現 `GetScalar(key)` は単一スカラーを 1 個ずつ返す形。動的に増える多数のブロック別 tag をどう列挙・記録するか（GetScalar 拡張／別経路）。
6. **頻度制御**: 毎 update は重い → N step 間引き。tag 数（網 × scope × block × sub × metric）の見積りと `metrics_scalar.txt` フィルタ設計。
7. **`named_parameters` と grad の順序対応**: `parameters()` と `named_parameters()` の順序一致前提でグループ index を作れるか（defined フィルタとの整合）。

## 非対象（Out of Scope）
- target/online の **乖離・相関**メトリクス（差分ノルム等）。将来。
- Tier 2（activation 統計／dead 率／attention entropy、forward hook 必要）・Tier 3（optimizer 内部統計）。後段。
- 実装そのもの（本書は設計まで。実装は Codex、別の実装メモへ）。

## 後続
1. 次段階: 上記「これから詰める論点」を現コード調査で解消し、本書を更新（暫定 → 確定）。
2. 確定後: 実装メモ（`999_..._20impl.md` 等）→ Codex 実装 → 受け入れ。
3. 番号 999 はバックログ仮番号。確定時に正式採番する。
