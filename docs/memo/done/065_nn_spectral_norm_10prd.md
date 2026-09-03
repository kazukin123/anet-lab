# NN Spectral Normalization（weight_norm.mode）PRD

> 起点: 2026-08-28〜29、高 `replay_ratio` Breakout 崩壊の機序確定（①重み成長→②活性成長→③ReLU恒久死→④表現痩せ）と
> weight_decay の限界実証（[探索ブロック 04 / 05](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)）。
> 裁定: 2026-08-29〜30 グリル（D1〜D14 + 簡素化監査 6 項目）で全決定済み、Codex 実装前レビュー 8 ラウンド
> （P1×7+P2×4 / 追指摘×2 / P1×3+P2×3 = zero-init 衝突→2 モード分割 D12〜D14 /
> P1×4+P2×2 = 再正規化契約・逸脱 6・u/v 専用 RNG / P1×3+P2×3 = D14 弱化・RNG 供給経路・τ 許容集合 /
> 第 6R = RNG registry・validity sentinel・SoftCopy 不変量・実装フェーズ整合 / 第 7R = 必須 seed API 全移行・
> ImageCls sentinel 形状・65/66 語義・BTR/init 表現・未実測性能記述 / 第 8R = `SoftCopyTo` 公開 API の
> 変更前 τ 検証・受入/BTR/メトリクス表現整合）を反映済み。
> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 関連: [ADR 0032](../../adr/0032-spectral-norm-self-impl-buffer-semantics.md)（本 PRD の決定記録）、
> [062](062_plasticity_metrics_10prd.md)（可塑性メトリクス。本 PRD は「保護機構の効果測定器」の最初の利用者）、
> [063](063_plasticity_weight_norm_10prd.md)（weight norm 2 群。本 PRD のメトリクス増分は 63 の同型拡張）、
> [2026-08-17_baseline.md](../../experiments/default-dqn/atari/2026-08-17_baseline.md)（探索ブロック 15: BTR / BBF / 本記録の保護機構比較表）、
> [999_MunchausenRL_10prd.md](../999_MunchausenRL_10prd.md)（同じ「コストだけ採って補償器を採っていない」系列の別件）。

## Context（背景・目的）

### 機序と weight_decay の限界

高 `replay_ratio`（RR≥4 目安）の Breakout で起きる崩壊の機序は次の一本道に絞れている
（数値の正本は [2026-08-28_plasticity.md](../../experiments/default-dqn/atari/2026-08-28_plasticity.md)）:

```
① 重みが止まらず育つ → ② 活性が育つ → ③ ReLU の負側で恒久的に死ぬ → ④ 表現が痩せる → スコア低下
```

weight_decay は①のレバーとして正しいことが実証済み — `61_weight_norm_feature` を**用量順に完全制御**した
（1.47 → 0.94、複製 0.79 / 0.80。探索ブロック 04）。しかし 10M では **V 字**（45.8 → 30.5 → 43.4 → 41.4。探索ブロック 05）。
λ·w は w が縮むほど弱まる**均衡機構**なので、勾配側の押し上げと釣り合う点で止まり、早期に稼いだ分の大半を押し戻される。

Spectral Normalization（SN）はペナルティではなく**毎 step の射影**（forward が常に W/σ を使う）なので、この均衡が存在しない。
BTR は SN を全 conv residual layer に適用して RR=4 で走れており、本コードベースは「BTR の γ 0.997 を Munchausen なしで、
高 RR を SN なしで」回している構図が記録済み（[2026-08-17_baseline.md](../../experiments/default-dqn/atari/2026-08-17_baseline.md) 探索ブロック 15）。

### 部分適用の罠（D1 の背景）

2026-08-28 の GroupNorm 試験は「ResBlock 内部のみ適用で、ダウンサンプリング Conv 3 本と最終 Linear512 が素のまま」
だったため公正な検証にならなかった（探索ブロック 04 考察）。‖w‖ 成長を無害化する機序が、測っている特徴（`main_feature` =
`AtariLinear512 > ReLU` の出力）の直前で成立していなかった。本 PRD は適用範囲を config の判断にし、この罠を設計から消す。

### 本 PRD の中核 = メトリクスの帰結

SN の標準実装は生の W をパラメータとして保持し、forward で W/σ を使う。したがって SN 下では:

- 生パラメータのノルム（現行 `61_weight_norm_feature` の測定対象）→ **制約されず伸び続ける**
- forward が実際に使う実効重み → σ で固定される

つまり「①のブレーキが効いたか」を測るはずの 61 が何も語らなくなる。この帰結への裁定（D8）が本 PRD の中核。

## 数理と適用の定義

### WeightNormConfig（`weight_norm.mode`）

重み正規化の設定は**専用 config struct `WeightNormConfig`** として持つ（`WeightInitConfig` と同じ棚の共通部品。
2026-08-30 追加裁定 — 当初の「各ブロック既存 Config へ `weight_norm_mode` を直埋め」は、共通部品の設定が
ブロック型ごとのキー綴り（`res.` / `conv.` / `linear.` / `cn.` / `tf.`）に割れて拡張性を欠くため struct 化へ改訂）。
各 `NetworkModuleFactory` の Config に `WeightNormConfig weight_norm;` として合成し、**config キーは全ブロック型で
一律 `weight_norm.mode`**（`init.mode` と同じ「メンバ名 = キー prefix」規約）。mode は 3 値:

| 値 | 除算則 | 位置づけ |
|---|---|---|
| `none`（既定） | なし（現行動作） | — |
| `spectral` | W_eff = W / σ（常時射影） | 参照実装準拠（§数理契約）。**非退化 init 必須**（D12） |
| `spectral_cap` | W_eff = W / max(1, \|σ_raw\|) | 参照 parity を主張しない近似 cap（D13）。σ<1 は恒等なので **zero-init を正式サポート** |

未知値は fail-fast。モード空間は将来の拡張（例: Salimans & Kingma の Weight Normalization → `direction` 等）を想定した
命名で、bool flag にしない。将来モードのパラメータ追加は `WeightNormConfig` のフィールドで閉じる
（struct 定義と factory への合成は §実装仕様 3）。

SN は「活性を変換する層」ではなく「**重みの再パラメータ化**」なので、構造チェーン上の独立ブロック
（`Conv2d > SpectralNorm` のような後置）では原理的に実現できない — 後段に届くのは conv の出力テンソルであり、
W/σ での計算は W を所有するモジュール自身の forward の中でしか起きられない（libtorch C++ には Python の
parametrization フックが無い。§事実 1）。よって mode は各ブロックの config になる。

### SN の数理契約（参照実装準拠の標準一式）

**参照実装 = リポジトリ採用 libtorch と同版（2.11 系想定。実装時に build-version で確定）の
`torch.nn.utils.parametrizations.spectral_norm`**（現行推奨 API）。deprecated の旧 `torch.nn.utils.spectral_norm`
（初期化で power iteration を回さない・v 先順）ではない。参照からの意図的逸脱は §逸脱表の 6 件のみ。

対象重み W（conv は `(c_out, c_in·kh·kw)` へ view して行列扱い。bias は対象外）に対し:

```
power iteration（u/v 更新。training mode かつ勾配有効の forward ごとに 1 回、NoGrad、eps=1e-12。参照と同じ u 先順）:
    u ← normalize(W v)
    v ← normalize(Wᵀ u)
σ の計算（train / eval を問わず毎 forward、その場の W で）:
    σ = normalize(u) · (W normalize(v))        ※ u/v は detach 済み buffer、W は勾配経路あり
実効重み:
    W_eff = W / σ                              ※ spectral: 常時除算。forward は W_eff で計算する
    W_eff = W / max(1, |σ|)                    ※ spectral_cap: σ<1 は恒等（σ への勾配経路なし = subgradient 契約）
```

u/v の推定機構（ガード付き power iteration・buffer・warm-start）は両モード共通（§実装仕様 1）。
`spectral_cap` は参照 parity を主張しない近似 cap で、保証は「u/v が固定点に落ちない」ことのみ —
zero-W からの非ゼロ遷移直後の一時的 under-clamp を許容する（σ_est = |σ_raw| は下界であり、収束は通常 forward 内の
power iteration で進む。全 forward での厳密 σ_max ≤ 1 は保証しない）。厳密上限が要件化されたときの遷移時
warm-start は deferred gate（§複雑度監査）。

- **勾配は σ 経由でも流す**（∂σ/∂W = uvᵀ。detach しない）。この項が支配的特異方向への成長を打ち消す実質の正則化で、
  参照実装と同一の学習動態になる（BTR がどちらの API を使ったかは未検証のため、BTR との比較基準は
  「同系の標準 SN 動態」とする — 中核（常時除算・勾配 σ 経由・training 時 1 iteration）は新旧 API で同一）。
- `spectral` は**常時除算**なので σ<1 の層では W が「拡大」される side effect がある（成長を止めるだけでなく、
  常に σ=1 へ射影する）。この性質が zero-init と数学的に非互換（D12）であり、zero-init を保ちたい層のために
  `spectral_cap`（σ<1 恒等）が併存する（D13）。
- power iteration の回数は 1 回/training forward の**固定**（config 化しない。§複雑度監査）。
- 1 update に複数の training mode forward が走る構成では u/v がその回数だけ進む。決定的であり問題ない
  （現行 learner で training mode に入る forward は `ForwardOnlineWithTrain` の 1 系統。§事実 4）。

### u/v の保持と σ の再計算（参照実装 parity + 逸脱 6 件）

- buffer は **u / v の 2 本のみ**を named buffer として register する（σ は buffer にしない）。
  `requires_grad == false` なので既存の `ComputeParameterNormSplit` 集計（61/62）に混入せず（§事実 6）、
  CopyTo / SoftCopyTo / シリアライズは既存の named_buffers 走査が無償で運ぶ（§事実 2）。
- **power iteration（u/v の更新）は `is_training() && torch::GradMode::is_enabled()` の forward のみ**。
  target net は常時 eval（§事実 3）、actor snapshot・probe 部分 forward も eval、そして NetworkBuilder の
  構築時 dummy forward と Clone 時の同経路は **training mode のまま NoGrad で走る**（§事実 14）ため GradMode 条件で
  除外される — この 2 条件により**u/v を変異させる経路は learner の online 学習 forward だけ**が文字通り成立する。
  eval forward は buffer を一切変更しない（062 の「測定が学習系列を変えない」契約と整合）。
  （参照実装のゲートは `self.training` のみで GradMode を見ない — 意味論差として逸脱表 5 に計上。）
- **warm-start 初期化（参照 parity）**: weight 実体化と同時に u/v を randn-normalize で生成し、その場で NoGrad の
  power iteration を **15 回**（`kSpectralNormWarmStartIters = 15` = 参照実装の init と同数）回す。
  参照実装自身が初期化時に 15 回回して "reasonable values" から始める仕様であり、**これは逸脱ではなく parity**。
  （deprecated の旧 API にはこの初期化が無く、旧 API 前提だと「最初の training forward まで σ 推定がゴミ =
  learning_starts 前の actor forward が過大な W_eff で走る」問題が起きる — RL では顕在化する構造のため、
  新 API へ pin する実益の一つ。）乱数は **専用 RNG 系統**（`BuildNetwork` の必須 seed 引数 → base seed を
  そのまま受けた汎用 registry `ModuleRandomSource`（1 network 1 個）が purpose seed を**一度だけ**導出 →
  purpose stream `Get("spectral_norm")` → `ModuleContext` の汎用スロットで共有。D6。parameter 初期化の global RNG 系列を消費しないため、mode 変更で parameter 初期値が
  不変になり paired 比較が成立する）から取り、
  構築・lazy init の順序は構成で固定されるため決定的。lazy init（Linear の in_features 自動推論等）は
  weight 実体化時点で同時に行う — 初期化は 1 回だけ許される変異とし、以後の非 training forward は buffer を変更しない。
- actor snapshot は CopyTo の正確コピー。snapshot 間は W が凍結なので σ 再計算は同値を返すだけ（無駄だが無害。
  クローン時に σ を確定して再計算を skip する最適化は将来の余地として注記に留める）。

**参照実装からの逸脱（全 6 件。これ以外は参照と同一の意味論）**
— 本表は **`spectral` モードの参照対比**。`spectral_cap` は参照 parity を主張しない併存モード（D13）で本表の対象外:

| # | 逸脱 | 参照の挙動 | 本実装 | 理由 |
|---|---|---|---|---|
| 1 | 使用時 normalize | u/v を `clone` して使う（re-normalize なし） | u/v を out-of-place normalize して使う（clone 保護を兼ねる） | `SoftCopyTo` が float buffer を lerp するため（§事実 2）、soft update（Atari 実構成 tau=0.001。§事実 5）で u/v が非単位ベクトル化しうることへの頑健化 |
| 2 | FP32 強制 | autocast 文脈に従う | power iteration・σ・除算を Autocast 局所 OFF + FP32 で計算（`force_fp32` イディオム。§事実 8）。W_eff を使う conv / linear 演算は autocast 任せ | bf16 構成での power iteration の数値安定 |
| 3 | Q/K/V 別 σ | packed 行列に適用すれば 1 σ | `in_proj_weight` のスライスごとに独立 state（D11） | 「1 重み行列 = 1 σ」の契約を全ブロックで統一 |
| 4 | 退化 σ の fail-fast | 無検証（σ→0 なら inf/NaN が流れる） | σ が非有限または ≤0 なら `ANET_SYSTEM_ERROR`（層名・σ 値入り）。検証点 = **① 構築時 warm-start（mode 別基準。D12/D13）+ ③ メトリクス経路（第 3 の防衛線・best-effort。D14）**。② 起動時 config + 公開 API の二重 τ 検証（D14）は参照に対応物が無い追加検証 | Fail-Fast 原則。forward hot path では検証しない（毎 forward の D2H 同期コストを避ける — §複雑度監査）。init/buffer 経路は D14 ①② が抑止し、W lerp 由来の σ≈0 は残余リスク（D14③） |
| 5 | power iteration ゲート | `self.training` のみ | `is_training() && torch::GradMode::is_enabled()` | 構築時 dummy forward / Clone 時再構築（training+NoGrad。§事実 14）を構造的に除外するため。参照側ではこの状況が起きない（Python に構築時 dummy forward が無い）ため実害はゼロだが、「training+NoGrad の forward で参照は u/v を進め本実装は進めない」意味論差として計上 |
| 6 | power iteration の保持ガード | 候補 normalize が 0 ベクトルを返し、u/v が固定点 0 へ落ちる（以後 σ=0 = W/0） | 候補（normalize(Wv) 等）のノルムが eps 以下なら当該ベクトルを更新しない（保持） | `spectral_cap` の zero-W サポートに必須。**`spectral` でも到達可能** — 横長の conv 行列（例 32×288、nullspace 次元 ≥256）の nullspace へ v が入れば非退化 W でも Wv=0（weight 更新・stale/lerp 済み v 経由。D12 の非ゼロ init では排除できない）。両モード共通適用とし、挙動は参照より安全側 |

- **逸脱 1 の頑健性の範囲（D14② で定量契約化）**: 使用時 normalize が守る範囲は **τ ≤ 0.1（または τ = 1）**として
  数値化し、**公開 API の変更前 τ 検証（D14②(b)）+ soft update 後の再正規化契約（D14②(c)）とセットで u/v ノルムについて帰納的に閉じた**（**σ>0 の保証ではない** — W 自体の lerp が σ≈0 を通す経路は D14③ の残余リスク）— 単発の下界 1−2τ は
  再正規化なしでは帰納しない（反復 lerp で target ノルムが 0 を交差しうる）ため、再正規化が「毎 lerp 前に
  両辺が単位ノルム」を不変量化する。符号反転した正当な状態同士の大 τ lerp（例: u₁=v₁=e₁ と u₂=v₂=−e₁ を
  τ=0.5 → σ=0。符号対称性は特異ベクトルの正当な性質）は D14②(a) が設定経路で、D14②(b) が公開 API 直接呼び出しで拒否する。
  サポート範囲 = CopyTo（正確コピー）、τ ≤ 0.1 の SoftCopyTo（再正規化つき）、τ = 1（コピー相当）。

### 適用範囲（D1）

`WeightNormConfig` は**重み行列を所有する全登録ブロック型**が持つ（config キーは全型一律 `weight_norm.mode`）:
**Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder**（共通部品化。実際にどれで使うかは config の判断）。

一律規則（D11）: **粒度は 1 ブロック 1 mode** で、ON はそのブロックが所有する全**重み行列**（乗算パイプラインの行列）に
適用する。**bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外**（SN の対象は乗算
パイプラインの重み行列であり、文献も embedding / スケール系には掛けない）。各重み行列が独立の u/v を持つ。
ネットワーク内の空間的な選択はカタログの**ブロック・インスタンス単位**（`[ResA]` と `[ResB]` に別 mode）で行う。

| ブロック | ON 時の適用対象 |
|---|---|
| Linear | 自身の linear 重み |
| Conv1d / Conv2d | 自身の conv 重み |
| ResBlock | 内部の conv1 / conv2 / downsample **全部** |
| CNBlock | 内部の dwconv / pwconv1 / pwconv2 **全部**（layerscale γ・内部 norm は対象外） |
| TransformerEncoder | 全 layer の **Q / K / V**（packed `in_proj_weight` のスライスごとに独立 σ）+ out_proj + linear1 / linear2（norm affine は対象外）。**mode ≠ `none` × `use_sdpa=false` は fail-fast** |

depthwise conv（CNBlock の dwconv、`groups=channels`）も Miyato の reshape 規約 `(size(0), -1)` を一律適用する —
この行列化は dense conv でも畳み込みの真の作用素ノルムではなく代理であり、規約として一律にする。

Atari Impala backbone は standalone Conv2d（3 本）+ ResBlock 群 + Linear512 なので、
「BTR と同じ範囲（ResBlock のみ）」も「全層」も config の選択になる。

## 決定事項（2026-08-29 グリル + 簡素化監査 6 項目）

| # | 論点 | 裁定 |
|---|---|---|
| D1 | 適用範囲 | **重み行列を所有する全登録ブロック = 6 型**（Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder）。当初は Atari baseline 使用の 4 型 + CNBlock/Transformer deferred としたが、2026-08-29 追加グリルで**共通部品化**（配線は全部・実使用は config の判断、型レベルの部分適用罠を残さない）へ改訂。独立 SpectralNorm ブロック（チェーン後置）は原理的に不成立で棄却、builder の wrap 機構（前ブロックを食う DSL）は 1関心2機構 + 新DSL で棄却。Conv1d も全型共通部品の方針で配線対象（当初は ConvConfig 共有による黙殺キー回避が理由だったが、D2 の WeightNormConfig 化で ConvConfig には触れなくなり黙殺懸念自体が消滅 — 方針は不変） |
| D2 | config 形式 | bool flag でなく**専用 config struct `WeightNormConfig`**（`WeightInitConfig` と同じ棚の共通部品。各 factory Config に `weight_norm` メンバとして合成し、キーは全ブロック型一律 **`weight_norm.mode = none \| spectral \| spectral_cap`**、既定 `none`。3 値目は D13 で追加）。**2026-08-30 追加裁定**: 当初のブロック既存 Config への直埋め（`res.weight_norm_mode` 等）はキー綴りが型ごとに割れ共通部品の拡張性を欠くため struct 化へ改訂 — 将来モードのパラメータは struct フィールドで閉じる。文字列モードの拡張性（将来 `direction` 等）と house style（`norm_type` / `activation` / weight init mode が全て文字列モード）は不変。値は綴り出し `spectral`（`sn` は将来 Salimans WN 追加時に値名が衝突気味になる） |
| D3 | 数理 | **参照実装準拠の標準一式**（常時除算 / 勾配は σ 経由でも流す / power iteration 1 回・NoGrad・eps=1e-12・**u 先順** / conv reshape / bias 対象外）。参照 = リポジトリ採用 libtorch と同版の `parametrizations.spectral_norm`（deprecated 旧 API ではない。2026-08-29 Codex レビューで pin）。cap 型 `W/max(1,σ)` は **`spectral` の数理としては採らない**（参照 parity を守る）— zero-init サポート用の併存モード `spectral_cap` として D13 で採用（parity 非主張の別モード）。σ detach は方向正則化項が消え学習動態が別物になるため棄却 |
| D4 | σ の意味論 | **参照実装 parity**: σ は buffer にせず**毎 forward その場の W で再計算**。buffer は u/v の 2 本のみ。意図的逸脱は §逸脱表の 6 件のみ（使用時 normalize / FP32 強制 / QKV 別 σ / 退化 σ fail-fast / power iteration ゲートの GradMode 条件 / 保持ガード）。対抗案の「σ を buffer に焼き非 training forward は読むだけ」（追加計算ゼロ）は、参照と異なる独自意味論の記述コストが残るため棄却（[ADR 0032](../../adr/0032-spectral-norm-self-impl-buffer-semantics.md)） |
| D5 | u/v 更新点 | **`is_training() && GradMode::is_enabled()` の forward のみ**（= learner online 学習 forward だけ。target 常時 eval / actor / probe は非変異、構築時 dummy forward と Clone 時 forward は NoGrad+training のため GradMode 条件で除外 — §事実 14。2026-08-29 Codex レビュー P1-3 対応。参照ゲートは `self.training` のみのため意味論差として逸脱表 5 に計上）。soft update の u/v lerp 継承は許容し、**使用時 normalize** で頑健化（頑健性は D14② で τ ≤ 0.1（または τ=1）+ 再正規化契約として定量化 — §逸脱表 1 の注記。退化経路は D14（抑止 + 残余明示）が扱う） |
| D6 | 初期化 | **warm-start = 参照 parity + 専用 RNG**: weight 実体化時に randn-normalize + ガード付き power iteration **15 回**（`kSpectralNormWarmStartIters = 15` = 参照実装の init と同数）+ σ 検証 fail-fast（mode 別基準 = D14①）。**u/v の乱数は専用 RNG 系統**（2026-08-30 第 5R で供給経路を確定）: `NetworkBuilder::BuildNetwork()` に **必須 `seed_t` 引数**を追加する（既定値なし = クリーンブレーク。シグネチャ `BuildNetwork(config, input_specs, head_factory, seed, device)`）。agent seed から network identity の base seed を名前付き導出し、DefaultDQN / Rainbow / ImageCls は `SeedMaker(GetSeed()).MakeNamedSeed("network")`、MuZero は rep / dyn / pred ごとに `"network.rep"` / `"network.dyn"` / `"network.pred"` を使う。**現用呼び出し元は同一変更で全移行**する: DefaultDQNAgent / RainbowAgent → NetworkModel、ImageClsAgent、MuZeroNetworkModel の rep / dyn / pred、Network::Clone、全テストおよびその他の直接 `BuildNetwork` 呼び出し。**BuildNetwork は受け取った network base seed をそのまま**汎用 registry **`ModuleRandomSource`（仮名。1 network につき 1 個。2026-08-30 追加裁定で SN 専用クラスから昇格）**へ渡し、**purpose seed の導出は registry が一度だけ**行う（`SeedMaker(base).MakeNamedSeed(purpose)` で lazily 生成・キャッシュ。BuildNetwork 側では導出しない — network identity 導出と purpose 導出は別階層であり、purpose の二重導出を禁止。第 6R P1-1）。registry は purpose 名ごとの**独立 stream** を鋳造し **`ModuleContext` の汎用スロット（`std::shared_ptr<ModuleRandomSource> random_source`）**で共有する。ModuleContext は各ブロック構築時にローカル生成のため、registry の `shared_ptr` を **`BuildNetwork → NetworkBodyBuilder → NetworkStructBuilder` へ引数伝播**し、全 factory 呼び出しの `random_source` が**同一インスタンス**を指す（1 network 1 registry の実現手段）。両sub-builderの`random_source`は**既定値なしの必須引数**とし、null時にseed 0のregistryを暗黙生成するfallbackを置かない。**purpose stream 独立性は今回の必須要件**（同 purpose 再取得の同一性・異 purpose 独立性・base seed 再現をテストで保証）。SN は初利用者（`Get("spectral_norm")`）で、SN 配線 module が取得 stream を lazy init = 構築時 dummy forward 中の weight 実体化まで保持（生存は参照で担保。draw 順 = branch 実行順で決定的）。**purpose stream の独立性により、消費者の追加・ON/OFF が他 stream の draw 系列を変えない**（paired 比較性の一般化）。registry は単一 rnd_ の mixin である `RandomHolder` の形に合わないため継承しない — component 単位の RNG 所有は従来どおり RandomHolder（AgentBase / Env / Learner の棚。§事実 17）、module 層は本 registry が担う。Network / NetworkBody への RandomHolder 直継承は広い契約になるため不採用。Clone は **Network が保持する構築時 network base seed** を必須 seed 引数として `BuildNetwork(config_, input_specs_, head_factory_, construction_seed_, target_device)` へ再度渡し、再構築後の u/v は直後の CopyTo で上書きする（機能的には初期 u/v に依存しないが、seed 経路を決定的に閉じる）。**parameter 初期化の global RNG 系列を消費しない** → 同 seed なら mode（none / spectral / spectral_cap）を変えても全 parameter tensor が一致し、検証計画の paired 比較（B/C）と受入 1 の突合が成立（第 4R レビュー P1-4）。**ModuleContext の初拡張**（「今後の拡張用」の空構造体を module 乱数 registry の汎用共有口にする — 当初の「拡張しない」裁定を第 5R で改訂し、2026-08-30 に SN 専用から registry へ汎化） |
| D7 | 精度 | power iteration / σ / 除算は **FP32 固定**（Autocast 局所 OFF。`force_fp32` イディオム踏襲）。conv/linear 演算は autocast 任せ |
| D8 | メトリクス | **4 本追加・既定コメントアウト**（§メトリクス拡張）。61/62 は無改修（生ノルムを測り続ける）。「61 を実効重みで計算し直す」案は既存 Run との互換破壊（61 の意味が Run 世代で変わる）で棄却。**63〜66 の σ は eval 型**（保存済み u/v × その時点の W = target / actor が使うのと同じ規則。測定は update 適用直前で 61/62 と同じスナップショット座標系。直後の学習 forward が実際に使う σ とは 1 power iteration 分ずれる — 2026-08-29 Codex レビュー P1-4 裁定。学習 forward からの capture 案は 062 型機構の追加が重く座標系も割れるため棄却） |
| D9 | 収集の口 | **`NetworkModule` 基底の既定実装（空を返す）つき仮想関数 `GetSpectralNormEntries()`** を SN 配線 6 モジュールが override。walk は全 module への仮想呼び出し（**dynamic_cast は使わない** — AGENTS.md:170 が production での使用を禁止し仮想関数で表現せよと定める。2026-08-29 Codex レビュー P1-1 対応）。entry は **by-value の `torch::Tensor`**（weight は packed のスライス view 可、u/v は buffer ハンドル — Tensor は共有ハンドルなので by-value で安全。P1-2 対応）。buffer 命名規約のパターンマッチは文字列規約の暗黙契約化で棄却 |
| D10 | 受入 | §受入基準の 5 項目（OFF 完全不変 / ON 決定性 / smoke / 単体テスト / ラウンドロビン throughput 実測） |
| D11 | 粒度と境界（2026-08-29 追加グリル） | **1 ブロック 1 mode**（ON = 内部全重み行列。BTR 適用範囲準拠のブロック一様。per-weight override（`weight_norm2` 式の per-weight メンバ）は Gogianu 型の層選択実験 — 1 層だけ SN が全層適用を上回るケースの追試 — が pin されたら追加 = deferred gate）。対象は乗算パイプラインの**重み行列のみ**: bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外。TransformerEncoder は packed `in_proj_weight` を **Q/K/V 別 σ**（スライスごとに独立 state、実効 = cat(Wq/σq, Wk/σk, Wv/σv)。「1 重み行列 = 1 σ」の契約を全ブロックで揃える。packed のまま 1 σ は最大射影の σ で他 2 つも割る非標準結合で棄却）。**mode ≠ `none` × `use_sdpa=false` は fail-fast**（旧 MHA 経路は SDPA 等価性確認用の互換参照。functional 書き換え対応は必要が pin されたら） |
| D12 | SN × zero-init（2026-08-29 Codex レビュー） | **`spectral` は非退化 init 必須**。ResBlock 既定の zero-init（`init2.mode=constant`/0 = identity 開始。§事実 15）との併用は warm-start σ 検証で**説明付き fail-fast**（層名・σ 値・衝突キー `init2.mode=constant`・対処 `init2.mode = he` を含む）。退避策（σ floor / 遅延有効化）は設けない — ゼロ除算だけ守っても初回 update で微小 W₂ が σ=1 へフルスケール射影され、identity 開始はどのみち破壊されるため。**BTR 公式実装に zero-init は無い**（§事実 16）。実験 C は **BTR 型標準 SN = `spectral` + ローカル He init** と位置づけ、BTR の init distribution parity は主張しない |
| D13 | `spectral_cap` 併存モード | W_eff = W / max(1, \|σ_raw\|)。σ<1 は恒等（σ への勾配経路なし = subgradient 契約）で **zero-init を正式サポート**（identity 開始・初回 backward からの conv2 勾配・ゼロからの漸進成長を全て温存）。参照 parity は主張しない**弱い近似 cap**: 保証は「u/v が固定点に落ちない」（§実装仕様 1 のガード付き power iteration）ことのみで、非ゼロ遷移直後の一時的 under-clamp を許容（σ_est=\|σ_raw\| は下界、収束は通常 forward 内 PI で進む。全 forward での厳密 σ_max ≤ 1 は保証しない）。厳密上限が要件化されたときの遷移時 warm-start は deferred gate。warm-start 検証は**非有限のみ**（σ=0 は正常）。ガードは `spectral` でも横長 conv の nullspace 経由で理論上到達可能なため**逸脱 6 として計上**（挙動は参照より安全側） |
| D14 | 退化経路の抑止と残余の明示（2026-08-30 第 5R で「クロージャ」から弱化） | 実行時（forward hot path）検証は追加しない。**① 構築時** = warm-start σ 検証（`spectral`: 非有限 or ≤0 fail-fast + D12 ヒント / `spectral_cap`: 非有限のみ）— **init 由来の退化はここで閉じる**。**② 二重 τ 検証 + soft update 後の再正規化契約** = (a) soft 経路（`hard_update_interval ≤ 0`）が有効で mode ≠ none の層が 1 つでも存在する場合、`soft_update_tau` は **finite 必須・許容集合 = {0 ≤ τ ≤ 0.1} ∪ {τ = 1}**（毎 step 正確コピー相当）。負・0.1<τ<1・τ>1・NaN/Inf は agent 構築時に両キー名入り fail-fast（hard 経路では未使用のため検証対象外。`soft_update_tau` の一般値域検証は現行コードに無く、本検証は SN 層存在時のみ = 一般検証はスコープ外）。数学注記: 単位ベクトル対の lerp 下界は正確には **\|1−2τ\|** で τ≥0.9 側も下界 0.8 を持つが、許容集合は数学上の最大範囲ではなく**運用意図（小刻み追従 τ≤0.1 か正確コピー τ=1 か）による意図的制限**。(b) 公開 API **`Network::SoftCopyTo()` 自身も、source または target に SN entry がある場合は parameter / buffer を変更する前に同じ有限性・許容集合を検証**する。違反は τ 実値と許容集合入りで fail-fast し target を一切変更しない（NetworkModel の (a) は設定キー付き早期診断として併存）。(c) 検証通過後、**`Network::SoftCopyTo()` が buffer lerp 直後に dst の SN buffer（u/v）再正規化までを原子的に行う**（公開 API が自身の不変量を守る — 呼び出し側手順では SoftCopyTo 単体利用で不変量が破れる。第 6R P1-3 / 第 8R 公開 API 事前条件。§実装仕様 1。NoGrad・device 上・同期なし・SN 層ゼロなら no-op・`ANET_PROFILE_SCOPE` 対象）。単発の lerp 下界 **‖(1−τ)u+τu′‖ ≥ 1−2τ** は再正規化なしでは帰納しない（τ=0.1 でも反対向き source が続くと 1→0.8→0.62→…と反復縮小し 0 を交差する）が、(b)(c) により「毎 lerp 前に両辺が単位ノルム」を帰納不変量として適用可能な範囲へ API を閉じ、**u/v ノルムの退化（buffer のゼロ化）は帰納法で閉じる — ただしこれは u/v ノルムの保証であって σ>0 の保証ではない**。**③ 残余リスクの明示** = **soft update の W lerp が target W を σ≈0 近傍へ通す遷移は排除を証明できない**。scalar 反例: target（W=1, u=v=1, σ=1）と source（W=−9, u=−1, v=1, σ=9）は各々正常な状態だが、τ=0.1 で W′ = 0.9·1 + 0.1·(−9) = 0 — u/v は再正規化後も単位ノルムのまま σ′=0（raw W のスケールは SN 下でゲージ自由なので、online/target の大スケール反平行を証明付きで排除できない）。連続追従の τ≤0.1 運用で raw W が反平行になる経路は実質想定外だが、契約上は**残余リスク**とする。**この残余は `spectral` のみ**（`spectral_cap` は σ=0 が合法で分母 max(1,\|σ\|)=1 → W_eff=W となり非有限は発生しない — W→0 は cap のサポート内正常領域。第 6R P2-4）。検出は**メトリクス経路の online / target validity sentinel**（§実装仕様 4。第 3 の防衛線）と下流症状（`spectral` の loss 非有限）に委ねる。厳密閉鎖（SoftUpdate 後の σ 検証）は soft update = 毎 learn step のため毎 update D2H 同期となり不採用（§複雑度監査）。旧主張「supported configuration に W/0 への経路が残らない」「残余は FP32 underflow のみ」は撤回 |

## 実装仕様

### 1. 共有ヘルパ（nn 機能グループ同居、名前付き namespace）

```cpp
namespace anet::nn {   // 実名は実装裁量（無名 namespace は使わない）

struct SpectralNormState {
    torch::Tensor u;   // (rows)  named buffer として所有モジュールに register
    torch::Tensor v;   // (cols)  同上
};

enum class WeightNormMode { kNone, kSpectral, kSpectralCap };
// WeightNormConfig::mode 文字列からの解決。未知値はキー・指定値・許容値一覧入りで ANET_SYSTEM_ERROR
//（各 factory の CreateModule 検証で使用）
WeightNormMode ParseWeightNormMode(const std::string& mode);

// weight_mat = weight を (weight.size(0), -1) へ view した行列。3 関数とも
// Autocast 局所 OFF + FP32 で計算する（W が FP32 パラメータならゼロコスト cast）。

// in-place の power iteration 1 回（NoGrad・非 const）。順序は参照実装と同じ u 先で、
// 各ベクトルに保持ガードを付ける（eps=1e-12）:
//   t = W v;   u ← (‖t‖ > eps) ? normalize(t) : u    // 保持ガード: zero-W 期間は randn を保持
//   s = Wᵀ u;  v ← (‖s‖ > eps) ? normalize(s) : v
// 候補側は素朴な t/‖t‖ でなく normalize（x / clamp_min(‖x‖, eps)）で計算すること —
// torch::where は両辺を評価するため、捨てられる側で NaN テンソルを生成すると anomaly 検出系を汚す。
// このガードにより W=0 上で PI は自然に no-op となり（randn u/v 保持）、W 非ゼロ化後に通常収束へ入る
// （spectral_cap の zero-W サポートに必須。spectral でも横長 conv の nullspace 経由で理論上
// 到達可能なため逸脱 6 として計上 — 挙動は参照より安全側）。
void PowerIterationStep(const torch::Tensor& weight_mat, SpectralNormState& state);

// const・読み取り専用。σ = normalize(u)·(W normalize(v))（0-dim FP32。u/v は detach、W には勾配経路を残す）
torch::Tensor ComputeSpectralSigma(const torch::Tensor& weight_mat, const torch::Tensor& u, const torch::Tensor& v);

// warm-start: u/v を randn-normalize で生成し（乱数は rnd = ModuleRandomSource の "spectral_norm" stream。
// ModuleContext 経由で受領 — D6。
// parameter 初期化の global RNG 系列を消費しない）、ガード付き PowerIterationStep を
// kSpectralNormWarmStartIters 回実行（特例分岐なし — W=0 でもそのまま回してよい）、最後に σ を mode 別に検証:
//   spectral     : 非有限 or ≤0 → ANET_SYSTEM_ERROR（name・σ 値・衝突キー init2.mode=constant と
//                  対処 init2.mode = he のヒントを含む。D12）
//   spectral_cap : 非有限のみ → ANET_SYSTEM_ERROR（σ=0 は正常 = zero-init）
// name はエラーメッセージ用のみで state には保持しない。
SpectralNormState MakeSpectralNormState(const torch::Tensor& weight, WeightNormMode mode, const std::string& name,
                                        anet::RandomGenerator& rnd);

inline constexpr int kSpectralNormWarmStartIters = 15;   // 参照実装の init と同数（parity）

}
```

- forward の呼び出し規則: **`is_training() && torch::GradMode::is_enabled()` なら `PowerIterationStep` →
  `ComputeSpectralSigma`、それ以外は `ComputeSpectralSigma` のみ**。GradMode 条件は、NetworkBuilder の構築時
  dummy forward と Clone 時の同経路（training mode のまま NoGrad で走る。§事実 14）を構造的に除外するため —
  これで「u/v を変異させるのは learner online 学習 forward だけ」（D5）が文字通り成立する。
- σ を得た後の除算は mode 別: `spectral` = `W / σ`、`spectral_cap` = `W / max(1, |σ|)`（σ<1 分岐は恒等で
  σ への勾配経路なし。D13 の subgradient 契約）。
- **起動時 config 検証（D14②(a)）**: agent 構築時に、network の SN 層の有無（`GetSpectralNormEntries()` 非空）と
  `model.hard_update_interval` / `model.soft_update_tau` を突き合わせ、soft 経路（`hard_update_interval ≤ 0`）では
  `soft_update_tau` に **finite 必須・許容集合 {0 ≤ τ ≤ 0.1} ∪ {τ = 1}** を課し、それ以外
  （負・0.1<τ<1・τ>1・NaN/Inf）は両キー名入りで `ANET_SYSTEM_ERROR`（hard 経路は未使用のため対象外。
  実装位置は NetworkModel 構築の並び、裁量）。
- **公開 API の τ 検証（D14②(b) — 所有者は `Network::SoftCopyTo` 自身）**: source または target の
  `GetSpectralNormEntries()` が非空なら、`Network::SoftCopyTo()` は parameter / buffer を変更する前に τ が
  **finite かつ {0 ≤ τ ≤ 0.1} ∪ {τ = 1}** であることを検証する。違反は τ 実値と許容集合入りで
  `ANET_SYSTEM_ERROR` とし、target を一切変更しない。NetworkModel の D14②(a) は両設定キーを示せる
  早期診断として併存する。SN entry が source / target のどちらにも無い場合は、既存 `SoftCopyTo` の τ 契約を変更しない。
- **soft update 時の再正規化（D14②(c) — 所有者は `Network::SoftCopyTo` 自身）**: D14②(b) の検証通過後、`Network::SoftCopyTo()` が
  buffer lerp の完了直後に、dst（target）の SN buffer（u/v）再正規化までを**原子的に**行う（全 entry の u/v を
  normalize して buffer へ書き戻す。NoGrad・device 上・同期なし・SN 層ゼロなら no-op）。呼び出し側の手順にすると
  公開 API `SoftCopyTo` の単体利用者が D14 不変量を破れるため、**公開 API が自身の不変量を自分で守る**
  （2026-08-30 第 6R P1-3 / 第 8R 公開 API 事前条件。`NetworkModel::SoftUpdate()` 側の追加呼び出しは無し — 起動時 config 検証だけが
  NetworkModel に残る）。hard 経路（CopyTo = 正確コピー）は不要。learner update 毎に SN 層数比例で走るため
  **`ANET_PROFILE_SCOPE` を入れる**（安定名例 `Network::SoftCopyTo.spectral_norm_buffers`）。
  保証するのは u/v ノルムのみで **σ>0 は保証しない**（D14③）。
- 関数を更新（非 const）と読み取り（const）に分割したのは、メトリクス経路（§4）が const な収集 entry から
  σ を計算するため（2026-08-29 Codex レビュー P1-2 の const 整合）。
- **使用時 normalize は out-of-place 必須**: normalize が新 tensor を作ることが、参照実装の `u.clone()`
  保護を兼ねる — power iteration は buffer を in-place 更新するため、buffer を直接 autograd graph に載せると
  次の forward の in-place 更新が version counter と衝突する。「既に単位ベクトルなら normalize を省く」
  最適化は**禁止**（保護が消える）。
- **ANET_PROFILE_SCOPE を SN 経路に入れる**（AGENTS.md ProfileRange ルール — 頻繁に呼ばれる forward 境界の
  追加ベクトル演算は計測対象。計測名は `ClassName::Forward.spectral_norm` 等の安定名で、ヘルパ関数単位ではなく
  意味のある処理境界に置く）。

### 2. メトリクス収集の口（D9 — 基底仮想関数、dynamic_cast なし）

```cpp
struct SpectralNormEntry {
    std::string name;       // ブロック内相対の安定名（conv1 / dwconv / layer0.q 等）。エラー・診断用。
                            // walk 側が branch 名等の文脈 prefix を付与してよい。920（層別 σ）の将来の口も同じ name に乗る
    WeightNormMode mode;    // spectral / spectral_cap（実効換算の除算則がモードで変わるため）
    torch::Tensor weight;   // 対象重み。packed の場合はスライス view（Tensor は共有ハンドルなので by-value で安全）
    torch::Tensor u;        // buffer ハンドル（by-value。storage は register 済み buffer と共有）
    torch::Tensor v;        // 同上
};

// NetworkModule 基底に既定実装つきで追加（AGENTS.md:170 — production では dynamic_cast でなく仮想関数で表現）:
class NetworkModule ... {
    ...
    virtual std::vector<SpectralNormEntry> GetSpectralNormEntries() const { return {}; }   // 既定 = SN なし
};
```

SN を配線した 6 モジュール（LinearModule / Conv1dModule / Conv2dModule / ResBlockModule / CNBlockModule /
TransformerEncoderModule）が override する（Transformer は全 layer 分を集約。packed `in_proj_weight` は
Q / K / V のスライス 3 本として返す）。walk は保持済み module へ仮想呼び出しするだけで、cast も命名規約も不要。
entry の σ は const 版 `ComputeSpectralSigma(entry.weight を view した行列, entry.u, entry.v)` で計算する。

### 3. ブロック配線

共通部品の config 合成（全 factory 同型）:

```cpp
// nn.hpp（WeightInitConfig の並び）
struct WeightNormConfig {
    std::string mode = "none";   ///< "none" | "spectral" | "spectral_cap"（構築時に enum へ解決。未知値 fail-fast）
};
```

各 `NetworkModuleFactory` の `Config` に `WeightNormConfig weight_norm;` を追加し
`ANET_READ_CONFIG(config_data, weight_norm.mode);` で読む（キーは全型一律 `weight_norm.mode` — `init.mode` と
同じ「メンバ名 = キー prefix」規約）。ctor へは `const WeightNormConfig&` を渡し、module がメンバ保持 +
`GetCurrentConfigData()` へ dump + `CreateModule` で値検証 fail-fast。既存の `ResBlockConfig` / `ConvConfig` /
`LinearConfig` / `CNBlockConfig` / `TransformerConfig` は**無変更**。

- **ResBlockModule**（§事実 9）: factory Config へ上記合成 + ctor に `const WeightNormConfig&` を追加
  （init1/init2/init_ds の並び）。mode ≠ `none` 時は conv1 / conv2 / downsample の各 weight に
  独立の `SpectralNormState` を持ち（buffer 登録名は実装裁量。例 `sn_u_conv1` 等）、forward で
  `torch::nn::functional::conv2d(x, W_eff, bias, opts)` へ分岐する（W_eff は mode 別除算則 — §数理契約。opts = stride/padding 等は保持済み conv module から取得）。
  `none` 時は既存経路そのまま（分岐 1 個のみ）。
- **Conv2dModule / Conv1dModule / LinearModule**（引数バラ渡し。§事実 9）: factory Config へ上記合成 +
  ctor 引数追加（`ConvConfig` / `LinearConfig` は無変更）。lazy init（in_features / in_channels 自動推論）のモジュールは
  weight 実体化時点で `MakeSpectralNormState` + buffer 登録を行う。forward は functional 呼び出しへ分岐。
- **CNBlockModule**（§事実 12）: factory Config へ上記合成 + ctor に `const WeightNormConfig&` を追加
  （init_dw/init_pw1/init_pw2 の並び）。mode ≠ `none` 時は dwconv / pwconv1 /
  pwconv2 の各 weight に独立の state を持ち、forward を functional conv2d 分岐（dwconv は `groups=channels` を
  opts へ渡す）。lazy init（初回 forward の重み実体化）時に state 生成 + warm-start。layerscale γ・内部 norm は対象外。
- **TransformerEncoderModule**（§事実 13）: factory Config へ上記合成（`TransformerConfig` は無変更）+
  `TransformerEncoderModule` ctor に `const WeightNormConfig&` を追加。`CreateModule` の検証で値チェックと
  **mode ≠ `none` かつ `use_sdpa=false` の fail-fast**（`ValidateDropRate` の並び）を行う。mode は
  `CustomTransformerEncoderLayer` の ctor へ渡す。mode ≠ `none` 時は layer ごとに **6 state**（`in_proj_weight` の Q / K / V スライス各 1 + out_proj +
  linear1 + linear2）を named buffer で持つ。attention は `anet::nn::SdpaSelfAttention` に実効重み
  （`cat(Wq/σq, Wk/σk, Wv/σv)` と out_proj の W/σ）を渡すオーバーロードを追加して呼ぶ。FFN は
  functional::linear 分岐。buffer は layer（torch::nn::Module）に register するため `named_buffers(true)` が
  module ツリー経由で拾い、CopyTo / SoftCopyTo / serialize は無償のまま。`GetSpectralNormEntries()` は
  TransformerEncoderModule が全 layer 分を集約して返す。
- 既存の weight init（`init.mode` 系）はそのまま生 W に適用され、その後 warm-start が走る（直交）。
- SN 分岐（power iteration + σ + functional 呼び出しの一連）には `ANET_PROFILE_SCOPE` を入れる（§1。
  AGENTS.md ProfileRange ルール — 学習・実行ループから頻繁に呼ばれる forward 境界の追加演算は実測可能に保つ）。
  throughput への影響は受入 5（ラウンドロビン実測）で確認する。

### 4. メトリクス拡張（`ComputeParameterNormSplit` の拡張）

現行 `Network::ComputeParameterNormSplit(feature_key)`（§事実 6）を拡張し、同じ walk（branch 単位の module 走査 +
閉包で feature/readout 帰属、heads は常に readout）の中で各 module の `GetSpectralNormEntries()`（基底仮想関数。
非 SN module は既定実装が空を返す）を呼んで集計を足す:

| 追加 field | 定義 |
|---|---|
| `feature_effective` / `readout_effective` | 61/62 と同じ群一括 L2 だが、**SN 層の weight のみ実効換算**（`spectral`: ‖W‖_F/σ、`spectral_cap`: ‖W‖_F/max(1,\|σ\|)。σ は const 版 `ComputeSpectralSigma` でその場計算。NoGrad・u/v 非更新）。bias・非 SN パラメータは生のまま。**SN 層ゼロの群では 61/62 と同値** |
| `sigma_feature_max` / `sigma_readout_max` | 群内 SN 層の **mode 別・非 clamp σ 推定値**の max（`spectral` = 正値検証済み σ_raw / `spectral_cap` = \|σ_raw\| — stale u/v による負の σ_raw が mode の読みとずれるのを防ぐ）。`spectral_cap` で除算に用いる分母は `max(1, 報告値)` であり、報告値自身を clamp しないことで 1 の横断点を観測できる。**SN 層ゼロの群では NaN**（既知 key の「値なし」= 062 の NaN 契約） |

- 測定時点は現行どおり **update 適用直前**（cadence gate の位置も現行のまま。§事実 6）。
- **σ の意味論 = eval 型**（D8）: 測定は学習 forward の**前**に走るため、63〜66 の σ は「保存済み u/v ×
  その時点の W」— target / actor の非 training forward が使うのと同じ規則の値であり、**直後の学習 forward が
  実際に使う σ（その forward 内の power iteration 1 回分だけ進んだ u/v）とは厳密には別物**。61/62 と同じ
  update 適用直前スナップショット座標系を優先した意図的裁定であり、ずれは 1 power iteration 分の推定差のみ。
- **退化 σ の検証はこの経路でも毎測定実施 — ただし位置づけは「第 3 の防衛線（best-effort 診断）」**（D14）:
  init / buffer 経路の fail-fast は D14 ①② が担い、この経路は購読・interval・Phase 2 に依存する diagnostic —
  ただし **W lerp 由来の σ≈0（D14③ の残余リスク、`spectral` のみ）にとっては実質的な第一検出線**（層名付き）。
  σ・実効・max σ は **device 上で集計して既存 stack 拡張に相乗り**する。DQN は **validity sentinel 2 本**を
  同じ stack に積み（2 → 6 要素 + sentinel 2 = 8 要素）、target を持たない ImageCls は online sentinel だけを積む
  （2 → 6 要素 + sentinel 1 = 7 要素。2026-08-30 第 6R P1-2 / P1追加、第 7R で形状確定）:
  (i) **online sentinel** — 収集 entry ごとの **mode 別 validity**（`spectral`: finite かつ >0 / `spectral_cap`:
  finite のみ）を device 上で不正カウントへ OR 集約する。**群 max 集計は「正常層 σ=2.0 の隣の `spectral`
  σ=−0.01」を隠す**ため、報告値（65/66 = max）とは別の sentinel で検証する（不正時に報告値を NaN 化する案は
  observer の isfinite skip = silent になり fail-fast にならないため不採用）。
  (ii) **target sentinel** — **63〜66 の意味論（online 走査）は不変のまま**、同じ購読 cadence で **target 網の
  mode 別 validity のみ**を device 集約する（`Network::ComputeSpectralNormValidity()` 相当の helper）。
  D14③ の検出対象は SoftCopyTo による **target** の W lerp であり、online 走査だけでは検出線にならないため。
  検証は**既存の lazy 一括 D2H 転送後の CPU 値**で行う — **同期回数は増やさない**。sentinel 不正 ≥1 のときのみ
  該当 net（online / target）を re-walk して層名を特定し `ANET_SYSTEM_ERROR`（エラー経路 = cold path なのでコスト不問）。
- 呼び出し側（DQN learner / ImageCls）の変更は次のとおり:
  - DQN: `ConfigureScalarMetricSubscriptions` の weight_norm 分類（§事実 6）へ 4 key を追加し、既存
    `weight_norm_enabled / weight_norm_interval` に min 合成で合流（61/62 と同じ棚）。搭載 tensor
    （`plasticity_.weight_norms` の stack）を 2 → 6 要素 + validity sentinel 2 本 = 8 要素へ拡張し、
    `GetPlasticityScalar` に key を足す（sentinel は公開 key を持たない内部要素 — 転送と検証にだけ使う）。
  - ImageCls: recognized set へ 4 key を追加し、同じ cadence gate で 6 公開値 + online sentinel の 7 要素を搭載。
    lazy 一括 D2H 後、公開値を返す前に online sentinel を検証する（target sentinel は存在しない）。
- 未測定 step・購読ゼロは 062/063 の契約そのまま（NaN / 完全不活性）。

### 5. config 契約

```
# nn.txt カタログ（既定 none なのでコメントアウト行 + 説明。force_fp32 の慣習と同じ）
#net.block.[ResA].weight_norm.mode = spectral          # 重み正規化モード(none|spectral|spectral_cap)。spectral=W/σ射影(BTR系可塑性保護、zero-initとは併用不可=要init2.mode=he)。spectral_cap=W/max(1,|σ|)でzero-init可。default=none
#net.block.[ConvA].weight_norm.mode = spectral         # 同上（キーは全ブロック型で一律）
#net.block.[AtariLinear512].weight_norm.mode = spectral   # 同上
#net.block.[CN64].weight_norm.mode = spectral          # 同上
#net.block.[TransEnc].weight_norm.mode = spectral      # 同上（use_sdpa=false との併用は fail-fast）
```

- 検証規則: `none` / `spectral` / `spectral_cap` 以外は fail-fast（キー・指定値・許容値一覧を含める）。
  加えて D12（`spectral` × zero-init は warm-start で説明付き fail-fast）と D14②(a)（SN 層あり × soft 経路では
  `soft_update_tau` は finite かつ {0 ≤ τ ≤ 0.1} ∪ {τ = 1} のみ許容、それ以外は起動時 fail-fast）の検証が乗る。
- 実効値は `GetCurrentConfigData()` 経由で config dump（`config_data.txt`）に出る（Run 検証は dump が ground truth）。

### 6. metrics 行

```
# metrics_scalar.txt @baseline（34 群 6x 帯の続き。SN 実験時に uncomment）
#metrics.scalar.@baseline.[34_agent_plasticity/63_weight_norm_feature_effective] = plasticity_weight_norm_feature_effective @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/64_weight_norm_readout_effective] = plasticity_weight_norm_readout_effective @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/65_spectral_sigma_feature] = plasticity_spectral_sigma_feature @learn $learn_step $update_result interval:500
#metrics.scalar.@baseline.[34_agent_plasticity/66_spectral_sigma_readout] = plasticity_spectral_sigma_readout @learn $learn_step $update_result interval:500
```

- **既定コメントアウト**（target 系 `2x` の「uncomment だけで発動」前例）。SN OFF で uncomment しても壊れない
  （63/64 は 61/62 と同値、65/66 は NaN → isfinite skip）。
- ImageCls.txt のフラット群にも同 4 行（コメントアウト）を置く。
- **規約コメントの追記**（実装時）: metrics_scalar.txt 冒頭の 34 群規約コメント（`x1`-`x5` 既定 ON / `x6`-`x9`
  既定 OFF はチャネル decade 用）へ「**6x は独立 decade: 61/62 = 生ノルム（既定 ON）、63-66 = SN 系
  （既定コメントアウト）**」を 1 行足す — 63/64 が x3/x4 位置なのに既定 OFF であることをチャネル規約と
  読み違えさせないため。
- 読み方: **65 の 1 超え成長 + 生 61 と実効 63 の乖離拡大（61 は成長・63 は抑制）**が「SN が仕事をしている」証拠。
  65 > 1 の意味は mode で異なる — **`spectral` は除算が常時発動**（σ<1 でも σ=1 へ射影）で 65>1 は「縮小方向の
  射影が起きている」ことを表し、**`spectral_cap` は 65>1 が cap 発動条件そのもの**。**63 単体のフラットは保証されない** — 63 は概ね ‖W‖F/σ_max = √(stable rank) 相当で
  最大以外の特異値分布によって増減し、群集計には bias・非 SN パラメータも含む。63 は SN OFF の過去 Run
  （生 = 実効が恒等）と同じ土俵で接続でき、WD 実験の V 字（45.8 → 30.5 → 43.4）との対比は **61/63 の乖離軸**で読む。

### 7. 単体テスト項目

- **σ の正しさ**: 既知の小行列で `ComputeSpectralSigma` を `torch::linalg::svdvals` の最大特異値と突合
  （power iteration 収束の許容誤差つき。warm-start 15 回後 + 追加 iteration 数回で十分収束すること）。
- **勾配（σ detach 禁止の回帰。P1-7）** 3 本。テスト行列は**特異値ギャップ σ₂/σ₁ ≤ 0.5 を確保**する
  （warm-start 15 回で収束誤差 (σ₂/σ₁)^30 ≈ 1e-9 ≪ 許容。ギャップが無いと u/v が不定になり解析比較自体が不安定）:
  (1) **解析比較（主）**: L = ⟨W_eff, M⟩ = ⟨W,M⟩/σ に対し **∂L/∂W = M/σ − (⟨W,M⟩/σ²)·uvᵀ**。SVD で厳密な
  u/v/σ を与え、autograd 勾配と FP32・相対 1e-4 で一致（detach 実装は第 2 項が消えて落ちる）。
  (2) 対角行列 W = diag(a, b)（a>b>0 = ギャップ充足）で同式の手計算値と一致 — detach 実装だと値が異なる
  ケースを明示的に選ぶ。
  (3) Transformer packed: Q / K / V スライス 3 本の実効重み経由の loss から `in_proj_weight` **全体**へ
  非ゼロ勾配が戻る（スライス view の逆伝播が閉じている）。
  副検証 = FP32 中心差分（差分幅 h = 1e-2 相対・許容 1e-2 相対）との粗い整合。**FP64 参照経路は作らない**
  （本体が FP32 強制である以上、FP64 経路は別物のテストになる）。
- **cap の subgradient**: `spectral_cap` で σ<1 のとき ∂W_eff/∂W = I（σ への勾配経路なし = 恒等）、
  σ>1 のとき `spectral` と同じ勾配。
- **zero-init × mode**: `spectral` × 全ゼロ weight → **説明付き fail-fast**（メッセージに name・σ 値・
  衝突キー `init2.mode=constant`・対処 `init2.mode = he` を含む。D12）/ `spectral_cap` × 全ゼロ weight →
  構築成功・W_eff = W（恒等）・conv2 相当へ勾配が流れる・W 成長後 σ>1 で cap が発動（除算開始）。
- **保持ガード**: W=0 で PI を複数回 → u/v が randn 初期値のまま単位ノルム維持 → W 非ゼロ化 → 次の PI 1 回で
  σ > 0（固定点に落ちない。D13）。ガード経路で **NaN テンソルが生成されない**（anomaly 検出 ON で警告なし —
  候補側 normalize 計算の検証）。
- **起動時 τ 検証（D14②(a)）**: SN 層あり + soft 経路で `soft_update_tau = 0.2` → 両キー名入り fail-fast /
  `−0.1`・`1.5`・NaN → fail-fast（finite + 許容集合検証） / `0.1` 境界は OK / `1` は OK（コピー相当） /
  hard 経路（`hard_update_interval > 0`）は tau 任意で OK / SN 層なしは検証対象外。
- **公開 API の τ 検証（D14②(b)）**: source または target に SN entry がある net へ
  **`Network::SoftCopyTo()` を直接** `τ=0.5` / NaN で呼ぶと、τ 実値・許容集合入りで fail-fast する。
  失敗前後で target の全 parameter / buffer が bit 一致し、部分更新が無いことを検証する。`τ=0.1` / `1` は通り、
  source / target のどちらにも SN entry が無い net は既存 τ 契約のまま。
- **soft update 再正規化（D14②(c)）** — **`Network::SoftCopyTo()` 単体呼び**で帰納不変量を 3 段階で検証
  （呼び出し側手順に依存しないこと = 所有権の検証）: (i) 各 lerp 前の保存 u/v はノルム 1
  （前回の再正規化の帰結）(ii) lerp 結果の**再正規化前**ノルムが 0.8 以上（τ=0.1 で反対向き source を連続適用
  しても維持）(iii) 再正規化後の保存ノルムが 1。**これは u/v ノルムの検証であり σ>0 の保証ではない**（D14③）。
  SN 層ゼロの net では no-op。
- **専用 RNG（D6）**: 同 seed で mode（none / spectral / spectral_cap）を変えても全 parameter tensor が
  bit 一致する（u/v 生成が global 系列を消費しない検証） / `BuildNetwork` の seed が同じなら u/v 初期列が再現し、
  seed だけ変えると u/v のみ変わり parameter は不変。
- **ModuleRandomSource（D6・purpose 独立性 = 必須要件）**: 同 purpose の再取得が同一 generator を返す /
  異 purpose stream の独立性（片方の draw が他方の系列を変えない）/ base seed からの再現（同 seed 再構築で同一列）。
- **validity sentinel（第 6R P1-2 / P1追加）**: 正常層（σ=2.0）+ 不正層（`spectral` σ=−0.01）の混在で
  online sentinel が検出する（群 max のマスキング回避）/ SoftCopyTo で **target のみ**を退化させ（online 正常）、
  target sentinel が検出して target 側の層名を特定する。
- **Linear / Conv1d / Conv2d の config 受入**: 各々で `weight_norm.mode` の READ・`GetCurrentConfigData()` dump・
  未知値 fail-fast（`ParseWeightNormMode` の許容値一覧付きエラー）・mode ≠ none で `GetSpectralNormEntries()` 非空。
- **ZeroInit 恒等（module-level）**: Atari の pre-activation ResBlock 構成（zero-init conv2・bias=0）で公開
  forward 出力が shortcut 入力と一致 — `spectral_cap` ON でも一致（identity 開始が API 追加なしで保証される）。
- **mode=none 恒等**: forward 出力が既存実装とビット一致。SN の数値経路（state 生成・power iteration・除算）に
  不到達（`GetSpectralNormEntries()` が空を返す walk は走ってよい）。
- **ResBlock 適用範囲**: `spectral` で conv1 / conv2 / downsample 全部の entries が `GetSpectralNormEntries()` に出る。
- **CNBlock 適用範囲**: `spectral` で dwconv / pwconv1 / pwconv2 の entries が出る（depthwise の reshape `(C, k²)` 含む）。
  layerscale γ が entries に出ない。
- **Transformer 適用範囲**: `spectral` で layer ごとに Q / K / V（`in_proj_weight` のスライス 3 本）+ out_proj +
  linear1 / linear2 の entries が出る / Q / K / V の σ が独立に計算される / **mode ≠ `none` × `use_sdpa=false` が
  fail-fast**（エラーにキーと理由を含む）。
- **非変異**: eval mode forward で u/v が不変（target / probe 相当経路）。**NoGrad + training mode の forward
  （構築時 dummy forward 相当）でも u/v が不変**（GradMode ゲート）。training mode かつ勾配有効の forward でのみ更新される。
- **第 3 の防衛線（メトリクス経路の best-effort 診断）**: 人工的に σ を退化させた net（u/v を手動でゼロ化する等）に
  対し、メトリクス測定が一括 D2H 後の検証 → re-walk で**層名入り** `ANET_SYSTEM_ERROR` を出す（診断経路の検証。
  契約側の fail-fast は D14 ①② のテストが担う）。
- **buffer 継承**: CopyTo で u/v が正確コピー / SoftCopyTo で lerp される / serialize round-trip で復元される。
- **使用時 normalize**: u/v を人工的に非単位化しても σ が単位化済みの値と一致する。
- **FP32 経路**: bf16 autocast 下で power iteration / σ が FP32 で走り、W_eff を使う conv/linear は autocast どおり。
- **メトリクス**: SN 層ゼロ群で `*_effective` が 61/62 と同値・`sigma_*` が NaN / SN 層あり群で実効 < 生（σ>1 の場合）/
  interface 経由の収集が walk の帰属（feature/readout）と一致。
- **warm-start 決定性**: 同 seed で u/v 初期値・σ が一致（専用 generator。2 回構築で再現）。
- **ON/OFF 等価性**: mode=none の同 seed Run でparameter初期値と学習系列がSNコード追加前と一致（受入1の単体版）。checkpointはserialize / load後のparameter・buffer復元をテストし、raw archive checksumは合否に使わない。

## 複雑度監査（グリル簡素化パスの記録）

| 対象 | verdict | pin / ゲート |
|---|---|---|
| power iteration ヘルパ + 6 ブロック mode | keep | 崩壊対策そのもの（探索 04/05 + BTR 実証） |
| warm-start 初期化 | keep | learning_starts 前の junk 正規化（RL 構造上の実害） |
| 使用時 normalize | keep | SoftCopyTo の buffer lerp（実装事実）への頑健化 |
| FP32 強制 | keep | learner bf16=true の実構成 |
| メトリクス 4 本 + interface 口 | keep | 61 が①のブレーキを語らなくなる実測由来の問題。readout 側 64/66 は 61/62 との対称性 + 同 walk で追加コスト微小 |
| `spectral_cap` 併存モード + ガード付き PI | keep（2026-08-29 追加） | SN × zero-init の数学的非互換（D12。ResBlock 既定の identity 開始）を、既定 init を壊さず解消する唯一の経路。ガードは cap の zero-W サポートに必須（u/v 固定点死の回避） |
| 起動時 τ 検証（D14②(a)） | keep（2026-08-29 追加） | 「小角度 drift」の定性文言を τ≤0.1（または τ=1）の定量契約へ。起動時 1 回で hot path コストゼロ |
| `SoftCopyTo` 公開 API の変更前 τ 検証（D14②(b)） | keep（2026-08-30 第 8R） | 汎用機構が自身の局所的事前条件を所有する。source / target の SN entry 有無と host scalar τ だけで判定し、違反時の部分更新を防ぐ。NetworkModel の設定検証はキー付き早期診断として併存 |
| SoftUpdate 後の SN buffer 再正規化（D14②(c)） | keep（2026-08-29 第 4R） | u/v ノルムの帰納不変量化 = **既知の buffer 退化経路の抑止**（σ>0 の保証ではない — D14③）。コスト = soft update 毎の normalize ×2×SN 層数（device 上・同期なし・SN 層ゼロなら no-op・PROFILE 対象） |
| module 乱数 registry `ModuleRandomSource`（BuildNetwork seed → SeedMaker purpose 派生 → ModuleContext 汎用スロット） | keep（第 4R 導入、第 5R 供給経路確定、2026-08-30 に SN 専用から汎化） | mode 変更で parameter 初期値不変 = B/C の paired 比較と受入 1 の突合が成立。purpose stream 独立により将来の消費者追加も他 stream を乱さない（house の seed 導出部品を再利用） |
| stochastic module（DropPath / Dropout 等）の purpose stream 化 | **deferred gate** | 標準 Dropout は ATen の functional に generator 引数が無くマスク自前実装（cuDNN 融合経路の喪失）が要る + 移行は既存 Run との同 seed 連続性を切る — 「構成変更に対する再現性の直交化」の実益が pin されたら、purpose 名の追加 + 独自の等価性物語を持つ別 PRD で（DropPath は自前 `bernoulli_` のため技術的には即移行可能） |
| CNBlock / TransformerEncoder への mode | keep（2026-08-29 改訂） | 当初 deferred gate としたが、共通部品化（配線は全部・実使用は config の判断、型レベルの部分適用罠を残さない）で全 6 型配線へ改訂（D1/D11） |
| per-weight override（ResBlock の init1/init2 と同様に `weight_norm1` / `weight_norm2` の per-weight `WeightNormConfig` メンバを足す形） | **deferred gate** | Gogianu 型の層選択実験（1 層だけ SN）が pin されたら追加。空間的選択は当面ブロック・インスタンス単位で足りる |
| mode ≠ `none` × `use_sdpa=false` の旧 MHA 経路対応 | **cut**（fail-fast） | 旧経路は SDPA 等価性確認用の互換参照。functional multi_head_attention_forward への書き換えは必要が pin されたら |
| 層別 σ の詳細 | **deferred gate** | [920_nn_block_metrics](../920_nn_block_metrics_10prd.md) の領分 |
| actor クローン時に σ を確定して再計算 skip | **deferred gate** | actor 側 throughput が実測で問題になったとき |
| power iteration 回数の config 化 / σ₀ 目標係数 | **cut** | 文献標準は 1 回・σ₀ 無し。必要になった実験が存在しない |
| 独立 SpectralNorm ブロック（wrap DSL） | **cut** | 原理的に不成立（重み所有 forward 内でしか実現できない）+ 1関心2機構 |
| σ buffer 継承（非 training forward は読むだけ） | **cut** | PyTorch と異なる独自意味論の記述コスト。[ADR 0032](../../adr/0032-spectral-norm-self-impl-buffer-semantics.md) |
| 61 の実効重み差し替え | **cut** | 既存 Run との互換破壊（61 の意味が Run 世代で変わる） |
| forward 毎の退化 σ 検証 | **cut** | 毎 forward × SN 層数の D2H 同期コスト。init / buffer 経路は D14 ①② が抑止（W lerp 由来は D14③ の残余）、メトリクス経路は validity sentinel を含む第 3 の防衛線 |
| learner の低頻度周期 σ 検証 | **cut** | 残余リスク（W lerp 由来の σ≈0 = D14③）の検出は第 3 防衛線と下流症状に委ねる裁定。毎/低頻度の D2H 同期を避ける判断を維持 |
| SoftUpdate 後の σ 検証（厳密閉鎖） | **cut** | soft update = 毎 learn step のため毎 update D2H 同期。第 5R で D14 を弱化し、残余リスクを明示する側を採った |
| σ floor / 遅延有効化（`spectral` × zero-init の退避策） | **cut** | ゼロ除算だけ守っても初回 update で微小 W₂ が σ=1 へフルスケール射影され identity 開始は破壊される（D12）。zero-init 温存の正道は `spectral_cap` |
| cap の遷移時 warm-start（厳密 σ_max ≤ 1 保証） | **deferred gate** | `spectral_cap` の under-clamp 許容（D13 の弱い契約）で足りなくなり、厳密上限が要件化されたとき |

## 検証計画（実行は実装後・結果は実験記録側へ）

検証は **screening / confirmation の 2 段構え**とする。損傷モードが RR で異なることが実測で出ており
（下記）、アッセイ合格のみで運用点有効と判定しない。

**screening = RR8 Breakout 5M・4 腕**（各 ×2 複製。ユニット死モードの高速アッセイ）:

| 腕 | mode | init | 役割 |
|---|---|---|---|
| A | none | zero-init（現行既定） | 無保護対照（実績 = `run_20260829-143027` / `run_20260829-153617`（plasticity_rr8_breakout）— 下表の −57% / −53% の出典） |
| B | none | `init2.mode = he` | **init 変更の confound 分離**（D6 の専用 RNG により同 seed で B/C の parameter 初期値が一致する paired 比較 — C との差分だけが SN の帰属になる） |
| C | `spectral`（全層） | `init2.mode = he` | 参照準拠 SN（D12 により he 必須） |
| D | `spectral_cap`（全層） | zero-init（現行既定） | identity 開始を温存する近似 cap |

判定は単一 Run の last 値でなく**複数 Run の終盤平均ブレ幅基準**（eval ピーク高の既知変動 ±26% を明記した上で読む）:

| 観測 | 判定 |
|---|---|
| 生 61 と実効 63 の乖離拡大（61 は成長・63 は抑制） | ①のブレーキが均衡でなく射影として効いた（**63 単体は stable rank で増減しうるためフラットは保証されない**。WD の V 字 45.8→30.5→43.4 との対比は 61/63 乖離軸で読む） |
| `02_dead_ratio` の谷後増加の抑制 | ③への波及が切れた（現行は谷から 6.7 倍増） |
| eval ピーク後落差の縮小 | 崩壊の緩和（現行 −57% / −53%） |
| `65_sigma` が **1 を超えて**単調成長 | クランプ量が実在した証拠（`spectral` は常時除算で 65>1 = 縮小方向の射影、`spectral_cap` は 65>1 = cap 発動条件。生 61 の成長と対で読む） |

**confirmation = RR1 @ 50M（運用点）**: screening の勝ち腕構成で実施する。**主読みは、終盤窓の eval1 が無保護の
新アンカー（2026-08-29 実測 ≈427〜454 帯、`run_20260829-163959_a5_breakout_apex`）を超えるか**。
srank 浸食の抑制はその機序説明として併読する — RR8 アッセイが再現しない**第 2 の損傷モード = ランク浸食**
（同 Run 実測: probe srank 440→354 の単調 −20%。RR8 アッセイでは −4〜6% で不再現）が
`45_probe_srank_ratio` の低下抑制として出るか。主従をこの順に固定するのは、機序指標だけ改善してスコアが
動かない形（WD 実測: `wn_feat` は用量順に完全制御・スコアは悪化）を再び踏まないため。
SN は σ（支配特異方向）を直接クランプする機構であり、ユニット死よりランク浸食側にこそ効く可能性がある —
screening だけで判定すると SN の一番の得意分野を測り損ねうる。

出典 Run はいずれも workspace `atari-2nd`。実験記録（2026-08-29 campaign）は別セッションで起票予定で、
それまでの検証の正は Run フォルダ側にある（Run フォルダ = 真実）。

適用範囲（BTR 適用範囲準拠 = ResBlock のみ vs 全層）は実験側の判断だが、**初手は全層**（standalone Conv2d 3 本 +
ResBlock 全部 + Linear512）を推奨 — GroupNorm 試験の部分適用の罠を踏まないため。

## 測定上の注意

- **σ<1 の層は拡大される**（常時射影）。「成長を止める」だけの機構ではない。
- **ON 腕は初期スケールから別物**: 非退化 init（Conv = He / AtariLinear512 = Xavier）の σ_max は本構成の形状
  （conv 64×576 / Linear 512×3136）でいずれも概算 ≈1.9〜2.0。`spectral` ON の瞬間に全 SN 層の実効重みが
  約半分になり、立ち上がりの学習曲線・q_max スケール
  （BF16 ULP 余裕の文脈含む）は OFF 腕と初手から別物になる。**立ち上がりの差を SN の保護効果と誤読しない**。
  さらに C 腕（`spectral`）は D12 により `init2.mode = he` 化するため **conv2 の identity 開始も失う** —
  zero-init 由来の効果と SN の帰属は B 腕（none + he）が分離する。`spectral_cap`（D 腕）が温存するのは
  **conv2 の zero-init（identity 開始）と σ<1 重みの非拡大のみ**で、σ>1 の非ゼロ層（He 初期化の Conv /
  Xavier 初期化の AtariLinear512、σ≈1.9〜2.0）は D 腕でも初回から約 1/σ に縮小される — ネットワーク全体の
  初期実効スケールが A 腕と近い保証はない。
- **weight_decay との併用**: SN 下の生 W への WD は σ を縮めるだけで実効重みに直接効かず、勾配スケール（実効学習率）を
  変える別経路になる。ベースライン採用構成は WD=0 であり、SN 実験も WD=0 で開始する。
- 61/62 は生ノルム（SN 下では制約されない）、63/64 は実効ノルム。**混同しない**（62 と同じくノルム絶対値は
  同構成 Run の時系列・同構成 Run 間の比較専用）。
- u/v の進みは training mode **かつ勾配有効**の forward の回数に従う。update 構造を変える改修
  （forward 回数が変わる）をしたら σ 推定の追従性が変わりうる（決定性は保たれる）。
- **63〜66 の購読と退化 σ 検出の関係**: init / buffer 経路の退化は D14 ①② が抑止するが、**W lerp 由来の σ≈0 は
  残余リスク（D14③、`spectral` のみ — `spectral_cap` は σ=0 が合法で W_eff=W となり非有限は発生しない）であり、
  メトリクス経路の online / target validity sentinel が実質的な第一検出線**（層名付き）。購読の無い `spectral`
  Run では残余リスクの検出は loss 非有限などの下流症状に落ちる。SN 実験では測定目的でも uncomment が前提であり、
  購読を切らないこと。

## スコープ外

- reset 系（Shrink-and-Perturb / ReDo）— 抑制系と別系統の保護機構。探索ブロック 05 の「次の検証」に残置。
- Munchausen（[999_MunchausenRL_10prd.md](../999_MunchausenRL_10prd.md)）、eval N 本平均（[060](../060_eval_batch_episodes_10prd.md)）。
- ResBlock への leaky ReLU / LayerNorm 追加など③への直接介入。
- `direction`（Salimans WN）等の追加モード実装（モード空間の予約のみ）。
- Rainbow / MuZero 専用の SN 実験・smoke・メトリクス配線（062 D8 と同じ理由: 実行 smoke 不能）はスコープ外。
  ただし `BuildNetwork` の必須 seed API への clean-break 移行は現用コードをビルド可能に保つため対象内（D6）。

## 受入基準

1. **OFF 完全不変**（手順で証明する）: (a) **本改修直前の base commit** で smoke 構成を固定実行
   （config・`train.seed`・step 数は実装計画で 1 組に固定して記載）し、解決済みconfigとmetricsマスタの
   主要 tag（loss / q_max / 61/62）の checksum を記録 → (b) 改修後ビルドで**同一コマンド**を実行 →
   (c) metrics checksum 一致。`agent_close.anet`のraw SHA-256は同一base実行体の再実行間でも一致しない既存serialize非決定性があるため合否ゲートにせず、サイズとhashを観測値として記録する
   （実測と恒久対処の観点は [930_serialize_10prd](../930_serialize_10prd.md) §決定性と等価性検証へ記録済み）。加えて `weight_norm.mode = none`（既定）では
   **SN の数値経路（state 生成・power iteration・除算）に不到達**
   （`GetSpectralNormEntries()` が空を返す仮想関数呼び出し自体は走ってよい）。
2. **ON 決定性**: `spectral`（+ `init2.mode = he`）と `spectral_cap`（zero-init のまま）の**各々**で
   同 seed 2 Run の学習系列一致（determinism 既定 ON 前提）。
3. **smoke**: Atari 構成 mode=`spectral`（全層）+ **`init2.mode = he`（全 ResBlock へ明示上書き。D12）** +
   63〜66 uncomment → `inspect_run.py tags` で `34_agent_plasticity/61`〜`65`が status=ok・count>0。
   `66_spectral_sigma_readout`はreadout側にSN対象層がある構成でのみcount>0とし、対象層がなければ契約どおりNaN（status=ok・count=0）とする。
   61/62 は従来どおり生ノルム。**別項: `spectral_cap`（全層）は init 変更なし（zero-init のまま）で
   同じ smoke が通ること**を確認する。
4. **単体テスト**: §実装仕様 7 の全項目が緑。
5. **throughput**（実測・許容差つき）: workload = 受入 1 と同じ smoke 構成。**ラウンドロビン配置**
   （実験機は 1 時間で最大 8% の throughput ドリフトがあるため、ブロック配置は符号を誤らせる — 交互配置必須）で
   (a) OFF（`none`）vs base commit を各 2 本: steps/s 平均差 **< 2%**（ゲート）。
   (b) ON（全層 `spectral`）を同配置で各 2 本: 実測を記録（ゲートなし — SN の実コスト把握が目的。
   ANET_PROFILE_SCOPE の計測値も添える）。

## 実装フェーズ（Codex 向け）

- **Phase 1 = SN 本体**: 共有ヘルパ + 6 ブロック配線 + warm-start + 受入 1/2/4（メトリクス以外）+
  **docs/design 同期**: `130_neural_networks.jp.md`（共通部品 `WeightNormConfig`（キー一律 `weight_norm.mode`）と
  SN 契約、`ModuleRandomSource` の registry 契約（purpose stream 独立性）の追記）。
  Phase 1 の独立価値は受入 1/2/4 の完了（OFF 完全不変と ON 決定性の確立）に限る。**効果実験の開始は
  Phase 2 完了後**（Phase 2 のメトリクス + validity sentinel が D14③ の検出線であり、検証計画も 63〜66 の
  購読を前提とするため。第 6R P2-7）。
- **Phase 2 = メトリクス**: 収集の口（基底仮想関数）+ `ComputeParameterNormSplit` 拡張 + 4 key 配線 + metrics 行 +
  受入 3 + **docs/design 同期**: `030_user_guide_analysis.jp.md` §4.7（63〜66 の読み方 — 063 が 61/62 で追記した
  項の並びに 1 項）と `200_dqn_agents.jp.md`（learner の購読 key 追加分）。

docs/design の同期は AGENTS.md「AI エージェントの作業ルール」（NN / config / metrics の設計・利用方法に影響する
変更では `docs/design/` 配下の関連ドキュメントも対象）に基づく。
Phase 1 で止めても悪化しない（既定 none なので存在自体が無害）。逆順は無意味（測る対象が無い）。

## 現行コードで確定している事実（実装の下地）

1. **libtorch C++ に `spectral_norm` は無い**。`libtorch/include/torch/csrc/api/include/torch/nn/utils/` にあるのは
   `clip_grad.h` / `convert_parameters.h` / `rnn.h` のみ。Python の parametrization 機構は C++ API 非公開。
2. **buffer は clone / soft update / シリアライズで既に全対応**。`Network::CopyTo`（`nn_impl.cpp:1789-1809`）は
   named_parameters / named_buffers を key 一致で copy_、`SoftCopyTo`（`:1811-1840`）はパラメータを
   `_foreach_lerp_`、**float buffer も lerp**（int buffer は copy_）。u/v を register_buffer するだけで
   actor snapshot の clone も target 同期も無償で乗る。
3. **target net は常時 eval mode**（生成 `dqn_based_agent.cpp:369-370`、別 ctor `:397-398`、Load 後 `:519-520`）。
   `ForwardTarget`（`:412-415`）はモード切替をしない。learner の学習 forward だけが
   `ForwardOnlineWithTrain`（`:406-410`）の `TrainingModeGuard(*online_net_, true)` で train に入る。
4. `ForwardOnlineWithTrain` の呼び出しは TD / QR / IQN の 3 箇所（062 §事実 3）。probe 部分 forward
   （`ForwardOnlineUpTo`）は NoGrad + eval 固定。
5. **Atari 実構成は soft update**: `A1.model.soft_update_tau = 0.001`（`Atari.txt:441`）、hard は `@nature` のみ
   （`soft_update_tau = 0` + `hard_update_interval = 10,000`、`Atari.txt:388-389`）。
6. **`Network::ComputeParameterNormSplit`**（`nn_impl.cpp:1553-1591`）: branch 単位の module 走査で
   `requires_grad() == true` のみ累積（`:1573`）→ buffer は混入しない。閉包は
   `ComputeDependencyClosure`（`:1173-1207`）、heads は常に readout（`:1585`）。呼び出しは cadence gate
   （`dqn_based_agent.cpp:2518-2524`、update 適用直前）、購読分類は `ConfigureScalarMetricSubscriptions`
   （`:1878-1898`、`weight_norm_interval` min 合成 `:1881-1884`）、搭載 `:2371-2373`、取り出し
   `dqn_based_agent.hpp:355-366`（未測定 NaN）。ImageCls 同型（`image_cls_agent.cpp:362-366`, `:490`）。
7. **ModuleContext は空構造体**（`nn_impl.hpp:246-248`）で RNG の受け渡し口が無い。重み初期化は
   global RNG（`torch::nn::init::*` 直呼び）で、seed は `MasterSeedManager::ApplyTorchSeed`
   （`random.cpp:147-157`、`torch::manual_seed` / `cuda::manual_seed_all`）が反映済み。house の seed 導出部品 =
   `MasterSeedManager::GetGroupSeed` / `SeedMaker::MakeNamedSeed・MakeIndexedSeed` / `RandomHolder`
   （`random.hpp:80-118`）。本 PRD はこの ModuleContext を初拡張し（「今後の拡張用」）、SN 用 RNG の共有口とする（D6）。
8. **FP32 強制イディオム**: `LayerNorm2dModule`（`nn_modules.cpp:635-720`）— パラメータ生成時 dtype 固定
   （`:656-660`）+ forward で `anet::Autocast disable_amp(device, false, kFloat32)` + 入力 cast（`:662-666`）+
   `GetCurrentConfigData` へ dump（`:674`）。`BatchNorm2dModule`（`:575-607`）も同型。
9. **ブロック config パターン**: ResBlock は config 構造体渡し（`ResBlockConfig` `nn_modules.cpp:790-804`、
   factory パース `:1041-1085`、検証 + ctor `:1088-1100`、dump `:966-981`）で flag 追加は 4 箇所・ctor 不変。
   Conv2d / Conv1d / Linear は引数バラ渡し（ctor `:187-190` / `:69-72`、`LinearConfig` `:2195-2198`、
   `ConvConfig` `:2200-2206`、factory `:2220-2245` / `:2247-2277` / `:2279-2309`）で 5 箇所・ctor 変更。
   **ConvConfig は Conv1d / Conv2d 共有**。パース prefix は `res.` / `conv.` / `linear.`（factory Config の
   メンバ名がキー prefix になる — `init.mode` 等。`WeightNormConfig weight_norm` メンバはこの規約で全型一律
   `weight_norm.mode` を得る）。
10. **config の enum 検証**: bool/値パースは fail-fast（`config.cpp:315-321`）。既定 OFF の flag は
    nn.txt でコメントアウト行 + `default=...` 注記が慣習（`nn.txt:54, 152, 196`）。
11. ResBlock の `activation` は relu / silu の 2 択、`norm_type` は none / batch / group のみ（③直接対策を
    ResBlock 内部へ入れるなら別 PRD）。`LayerNorm`（1D）と `LeakyReLU` は登録済みブロックで config のみで試せる。
12. **CNBlockModule**（`nn_modules.cpp:1108-1223`）: 重み所有は dwconv（depthwise, `groups=channels`）/ pwconv1 /
    pwconv2 の conv 3 本。layerscale γ（`gamma_`、1D）と内部 norm（LayerNorm2d）は重み行列でない。config 構造体
    渡し（`cn.` prefix、factory `:1225-`）+ 初回 forward の lazy init（`:1135-1181`）で、flag 追加は ResBlock と
    同じ 4 箇所パターン・ctor 不変。
13. **TransformerEncoderModule**（`nn_modules.cpp:1845-1956`）: `CustomTransformerEncoderLayer`（`:1702-1829`）×
    num_layers。layer の重み所有は MHA 保持器（packed `in_proj_weight` (3E,E) + `out_proj`）と linear1 / linear2、
    norm1 / norm2 / 最終 norm は affine。SDPA 経路は `anet::nn::SdpaSelfAttention(mha_, x)` の free 関数
    （`:1751, 1785`。PRD 012 で MHA を保持器に残し forward だけ関数化済み）なので実効重みを渡すオーバーロードが
    素直。旧経路（`use_sdpa=false`）は `mha_->forward()` が内部パラメータを直参照（`:1755, 1789`）し、注入には
    functional 書き換えが要る（→ fail-fast で回避、D11）。config は `tf.` prefix の構造体渡し
    （`TransformerConfig` `:1831-1842`、factory `:1929-1956`、`ValidateDropRate` の並びに検証追加可）。
14. **NetworkBuilder の構築時 dummy forward は training mode のまま NoGrad で走る**（`nn_impl.cpp:1913-1920`:
    `torch::NoGradGuard` 下で `body->Forward(dummy_input)`。`eval()` への切替は NetworkModel 側の生成後
    `dqn_based_agent.cpp:357` 等で行われる）。Clone 時の再構築も同経路。→ D5 の GradMode ゲートが両者を
    構造的に除外する根拠（2026-08-29 Codex レビュー P1-3）。
15. **ResBlock の conv2 は既定で zero-init**: `init2.mode = "constant"` が factory 既定（`nn_modules.cpp:1044`）で
    `constant_val` の既定は 0.0（`nn.hpp:122`）。Atari 側にも「全 ResBlock は既定の init2.mode=constant で conv2 を
    zero-init し、identity から開始する」と明記（`Atari.txt:492`）。→ D12 の根拠（`spectral` との数学的非互換）。
16. **BTR 公式実装（networks.py）に zero-init は存在しない**: PyTorch 既定 init のままで、SN の適用は
    residual block の conv 2 本のみ（2026-08-29 ユーザーによるコード確認）。本 PRD の実験 C は
    `spectral` + `init2.mode = he` で **BTR 同様の非 zero-init 側を測る BTR 型標準 SN** だが、ローカル He と
    PyTorch 既定 init は分布が異なるため init distribution parity は主張しない。適用範囲だけを揃える場合は ResBlock のみ。
17. **forward 時の確率的 module は global RNG 頼み**: DropPath は `bernoulli_(keep_prob)` を generator 未指定で
    呼び（`nn_modules.cpp:57`）、`torch::nn::Dropout` は ATen 内部で per-device の global generator を参照する
    （`functional::dropout` に generator を渡す口が ATen API に無い）。component 単位の RNG 所有は
    `RandomHolder` 継承が house 前例（`AgentBase` `agent.hpp:74` / `DiscreteBatchEnvBase` `env.hpp:102` /
    `ImageClsLearner` `image_cls_agent.hpp:234`）— module 層にはこの棚が無かった（→ D6 の registry が新設）。
