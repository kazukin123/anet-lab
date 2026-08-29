# Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する

高 `replay_ratio` の可塑性喪失崩壊（重み成長→活性成長→ReLU 恒久死→表現痩せ）に対し、weight_decay は
`weight_norm_feature` を用量順に制御できるが λ·w の均衡機構ゆえ V 字で押し戻される（2026-08-28 実験）。BTR が
RR=4 を回せている Spectral Normalization（毎 step の射影 W_eff = W/σ）を、**重み行列を所有する全登録ブロック**
（Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder）の**文字列モード
`weight_norm_mode = none | spectral`**（既定 `none`）として導入する（065 PRD。共通部品化 — 配線は全型・実使用は
config の判断とし、型レベルの部分適用罠を残さない）。libtorch C++ には Python の parametrization 機構（`torch.nn.utils.spectral_norm`）が公開されて
いないため power iteration を自前実装する。SN は重みの再パラメータ化であり、W/σ での計算は W を所有するモジュール
自身の forward 内でしか実現できない — 構造チェーン上の独立ブロック（`Conv2d > SpectralNorm` の後置）は原理的に
不成立で、mode は各ブロックの config になる。**粒度は 1 ブロック 1 mode**（ON = 所有する全重み行列。BTR 準拠の
ブロック一様で、空間的な選択はカタログのブロック・インスタンス単位が担う）。対象は乗算パイプラインの重み行列のみ —
bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外。TransformerEncoder は packed
`in_proj_weight` を **Q / K / V 別 σ**（スライスごとに独立 state。「1 重み行列 = 1 σ」の契約を全ブロックで揃える）で
扱い、**`spectral` × `use_sdpa=false` は fail-fast**（旧 MHA 経路は SDPA 等価性確認用の互換参照）。**保持するのは u / v の 2 buffer のみ**（named buffer 登録。
`requires_grad == false` なので既存の weight norm メトリクス 61/62 に混入せず、`CopyTo` / `SoftCopyTo` /
シリアライズの既存 named_buffers 走査が無償で運ぶ）。**power iteration（u/v の更新）は training mode の forward
のみ**（本コードベースでは target net が常時 eval・actor / probe も eval のため、u/v を変異させる経路は learner の
online 学習 forward だけになる）。**σ は buffer にせず、train / eval を問わず毎 forward その場の W から
`σ = normalize(u)·(W normalize(v))` で再計算する**（PyTorch parity）。勾配は σ 経由でも流す（∂σ/∂W = uvᵀ が
支配的特異方向の成長を打ち消す実質の正則化）。PyTorch からの逸脱は 2 点のみ: **(1) 使用時 normalize** —
`SoftCopyTo` が float buffer を lerp するため soft update（Atari 実構成 tau=0.001）で u/v が非単位ベクトル化しうる
ことへの頑健化。**(2) warm-start 初期化** — RL は learning_starts 前に actor forward が走るため、randn 初期化のみ
だと最初の training forward まで σ が大幅過小評価（W_eff 過大）になる。weight 実体化時に global RNG（manual_seed
済み・構築順固定で決定的）で u/v を生成し、NoGrad の power iteration を k 回（実装定数）回す。power iteration /
σ / 除算は FP32（Autocast 局所 OFF = `force_fp32` イディオム）で、W_eff を使う conv / linear 演算は autocast に任せる。

## Considered Options

- **(A) u/v buffer + σ 毎 forward 再計算（PyTorch parity。採用）**: 一般実装（PyTorch / BTR / Gogianu）と同じ
  意味論で文献との比較可能性が最大。soft (Polyak) update × SN は文献に前例が薄いが、一般実装も Polyak で buffer を
  放置または lerp しており、「u/v は近似でよい、σ は使う場で W と突き合わせる」という同じ受け皿に乗る。非 training
  forward の追加コストは matvec + dot × SN 層数（forward 本体の 0.1% 未満、kernel launch 数本）で、現在の律速は
  GPU Learner 側のため許容。actor snapshot は W 凍結中 σ 再計算が同値を返すだけ（クローン時に σ を確定して skip
  する最適化は将来の余地）。
- **(B) σ も buffer に焼き、非 training forward は読むだけ**: 追加計算ゼロで契約も 1 行（「直近 training forward の
  σ を使う」）だが、PyTorch と異なる独自意味論になり、その差異の記述・説明コストが恒久に残る。soft update では
  σ が lerp EMA 継承（W の lerp と整合的で二次のズレ）と、挙動として劣ってはいないが、独自性に見合う利得が無い。棄却。
- **(C) 独立 SpectralNorm ブロック（チェーン後置）**: 後段に届くのは前段 conv の出力テンソルで、W に手が届かない。
  原理的に不成立。棄却。
- **(D) builder の wrap 機構（`Conv2d > SpectralNorm` を検出して conv を SN モジュール内部へ畳み込む新 DSL）**:
  「前のブロックを食うブロック」という驚きの意味論 + 失敗系（前段が ReLU 等）+ 畳み込んだ後どのみち SN 側が
  functional conv を再実装する。複合ブロック（ResBlock）内部には効かず flag 方式と併存 = 1 関心 2 機構。棄却。
- **(E) bool flag（`spectral_norm = true`）**: 実装コストは同じだが、将来の重み正規化手法（Salimans & Kingma の
  Weight Normalization 等）でモード空間を拡張できない。house style（`norm_type` / `activation` / weight init mode）
  も文字列モード側。棄却し `weight_norm_mode = none | spectral` とする。
- **(F) cap 型 `W/max(1, σ)` / σ detach**: cap 型は文献非標準で BTR 再現の参照点を失う。σ detach は方向正則化項が
  消え学習動態が別物になる。いずれも棄却（σ<1 層が拡大される標準 SN の side effect は注記で明示）。
- **(G) Atari baseline 使用 4 型（Linear / Conv1d / Conv2d / ResBlock）に限定し CNBlock / Transformer は
  deferred gate（当初案）**: 「使う構成が来たら足す」だが、型レベルで部分適用の罠（GroupNorm 試験が ResBlock
  内部のみ適用で不公正になった構図のブロック型版）が残り、共通ヘルパゆえ配線コストも低い。2026-08-29 追加グリルで
  全 6 型へ改訂（実使用は config の判断）。ブロック内の per-weight override キー（`res.conv1_weight_norm_mode` 式）は
  逆に生やさない — 層選択実験（Gogianu 型）が pin されたら追加する deferred gate。

## Consequences

- OFF（`none`、既定）は挙動完全不変（SN コード不到達。同 seed で学習系列 + `agent_close.anet` 一致を受入で実証）。
  ON も決定的（u/v 初期化は manual_seed 済み global RNG・構築順固定、power iteration に乱数なし）。
- u/v が buffer としてシリアライズ形式・Run 互換に乗る（`spectral` の Run は resume でも u/v が正確に復元される）。
- 既存メトリクス 61/62（生パラメータノルム）は無改修で意味不変。SN 下では生ノルムが制約されないため、
  実効ノルム（63/64、SN 層のみ ‖W‖_F/σ 換算）と max σ（65/66）を `ComputeParameterNormSplit` の拡張として追加し、
  「生 61 ↑ + σ 65 ↑ + 実効 63 フラット」の三点で SN の作動を読む（SN 層ゼロ群では 63/64 = 61/62、65/66 = NaN）。
  収集は SN 保持モジュールが実装する interface（(weight, u, v) 列を返す）経由で、buffer 命名規約に依存しない。
- eval forward は u/v を変更しない（062 の「測定が学習の数値系列を変えない」契約と整合。probe 部分 forward からも安全）。
- 1 update に複数の training mode forward が走る構成では u/v がその回数進む（決定的・許容）。
- per-weight override キー、層別 σ、actor クローン時の σ 確定 skip、`spectral` × `use_sdpa=false` の旧経路対応、
  power iteration 回数 config、σ₀ 目標係数は deferred / cut（065 PRD の複雑度監査）。
- 仕様詳細は `docs/memo/065_nn_spectral_norm_10prd.md`（決定事項 D1〜D10、実装仕様、受入基準）。
