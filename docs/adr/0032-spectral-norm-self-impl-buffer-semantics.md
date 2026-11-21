# Spectral Normalization は自前 power iteration + u/v named buffer で実装し、σ は毎 forward その場の W で再計算する

高 `replay_ratio` の可塑性喪失崩壊（重み成長→活性成長→ReLU 恒久死→表現痩せ）に対し、weight_decay は
`weight_norm_feature` を用量順に制御できるが λ·w の均衡機構ゆえ V 字で押し戻される（2026-08-28 実験）。BTR が
RR=4 を回せている Spectral Normalization（毎 step の射影 W_eff = W/σ）を、**重み行列を所有する全登録ブロック**
（Linear / Conv1d / Conv2d / ResBlock / CNBlock / TransformerEncoder）の**専用 config struct `WeightNormConfig`**
（`WeightInitConfig` と同じ棚の共通部品。各 factory Config に `weight_norm` メンバとして合成し、キーは全ブロック型
一律 **`weight_norm.mode = none | spectral | spectral_cap`**、既定 `none` — 当初のブロック既存 Config への直埋めは
キー綴りが型ごとに割れるため 2026-08-30 に struct 化へ改訂）として導入する（065 PRD。共通部品化 — 配線は
全型・実使用は config の判断とし、型レベルの部分適用罠を残さない）。正規化モードが 2 本あるのは
**SN × zero-init の数学的非互換**のため — ResBlock は既定で conv2 を zero-init（`init2.mode=constant`/0、
identity 開始）しており、常時射影の `spectral` は W=0 で σ=0（未定義）。ゼロ除算だけ守っても初回 update で微小 W₂
が σ=1 へフルスケール射影され identity 開始はどのみち破壊されるため、退避策（σ floor / 遅延有効化）は設けず、
**`spectral` は非退化 init 必須**とする（zero-init との併用は warm-start σ 検証で、層名・σ 値・衝突キー
`init2.mode=constant`・対処 `res.init2.mode = he` を含む説明付き fail-fast。BTR 公式実装 networks.py に
zero-init が無いことも確認済み。本 PRD の `spectral` + ローカル He init は **BTR 同様の非 zero-init 側を測る
BTR 型標準 SN** であり、PyTorch 既定 init との init distribution parity は主張しない）。zero-init を温存したい層には
**`spectral_cap` = W / max(1, |σ_raw|)**（σ<1 は恒等・σ への勾配経路なし）を併存させ zero-init を正式サポートする
— 参照 parity を主張しない弱い近似 cap で、保証は「u/v が固定点に落ちない」ことのみ（非ゼロ遷移直後の一時的
under-clamp を許容。厳密 σ_max ≤ 1 が要件化されたときの遷移時 warm-start は deferred）。libtorch C++ には Python の parametrization 機構（`torch.nn.utils.spectral_norm`）が公開されて
いないため power iteration を自前実装する。SN は重みの再パラメータ化であり、W/σ での計算は W を所有するモジュール
自身の forward 内でしか実現できない — 構造チェーン上の独立ブロック（`Conv2d > SpectralNorm` の後置）は原理的に
不成立で、mode は各ブロックの config になる。**粒度は 1 ブロック 1 mode**（ON = 所有する全重み行列。BTR 適用範囲準拠の
ブロック一様で、空間的な選択はカタログのブロック・インスタンス単位が担う）。対象は乗算パイプラインの重み行列のみ —
bias・normalization affine・layerscale γ・embedding テーブル・cls token は対象外。TransformerEncoder は packed
`in_proj_weight` を **Q / K / V 別 σ**（スライスごとに独立 state。「1 重み行列 = 1 σ」の契約を全ブロックで揃える）で
扱い、**mode ≠ `none` × `use_sdpa=false` は fail-fast**（旧 MHA 経路は SDPA 等価性確認用の互換参照）。**保持するのは u / v の 2 buffer のみ**（named buffer 登録。
`requires_grad == false` なので既存の weight norm メトリクス 61/62 に混入せず、`CopyTo` / `SoftCopyTo` /
シリアライズの既存 named_buffers 走査が無償で運ぶ）。**power iteration（u/v の更新）は
`is_training() && GradMode::is_enabled()` の forward のみ** — target net は常時 eval、actor / probe も eval、
NetworkBuilder の構築時 dummy forward と Clone 時の同経路は training mode のまま NoGrad で走る
（`nn_impl.cpp:1913-1920`）ため GradMode 条件で除外され、u/v を変異させる経路は learner の online 学習 forward
だけになる。**σ は buffer にせず、train / eval を問わず毎 forward その場の W から
`σ = normalize(u)·(W normalize(v))` で再計算する**。勾配は σ 経由でも流す（∂σ/∂W = uvᵀ が
支配的特異方向の成長を打ち消す実質の正則化）。**参照実装 = リポジトリ採用 libtorch と同版（2.11 系想定、実装時に
build-version で確定）の `torch.nn.utils.parametrizations.spectral_norm`**（現行推奨 API。deprecated の旧
`spectral_norm` ではない）— power iteration の u 先順・初期化時の 15 回ウォームアップ（weight 実体化時に
u/v を生成し 15 回回す = 参照の init と同数の parity。乱数は `NetworkBuilder::BuildNetwork()` の必須 seed 引数から
供給する。agent seed から network identity の base seed を名前付き導出し、DefaultDQN / Rainbow / ImageCls は
`"network"`、MuZero は `"network.rep"` / `"network.dyn"` / `"network.pred"` を使う。Clone は Network が保持する
構築時 base seed を `BuildNetwork()` へ再度渡す。必須 API への移行はこれら現用経路と全直接呼び出しを同一変更で行う。
BuildNetwork は受け取った base seed をそのまま汎用 registry `ModuleRandomSource`（仮名・1 network 1 個）へ渡し、
purpose stream `Get("spectral_norm")` の purpose seed は registry が `SeedMaker::MakeNamedSeed` で**一度だけ**導出する。
purpose stream 独立性は必須要件でテスト保証。registry は builder 階層へ同一インスタンスを伝播し、`ModuleContext` の
汎用スロットで共有する — 「今後の拡張用」空構造体の初拡張。purpose stream の独立性により消費者の追加・ON/OFF が
他 stream の draw 系列を変えない。registry は単一 rnd_ の `RandomHolder` の形に合わず継承しない — component 単位の
所有は AgentBase 等の RandomHolder、module 層は本 registry。DropPath / Dropout 等の stream 化は同 seed 連続性を切るため
別 PRD の deferred）で、parameter 初期化の
global 系列を消費しない・構築順固定で決定的）は参照に一致させる。power iteration には**ベクトル単位の保持ガード**を付ける（候補 = normalize(Wv) 等のノルムが
eps 以下なら当該ベクトルを更新しない — `spectral_cap` の zero-W 期間に u/v が固定点 0 へ落ちて cap が恒久不活性に
なるのを防ぐ。候補側は normalize（x/clamp_min(‖x‖,eps)）で計算し、where の両辺評価による NaN テンソル生成が
anomaly 検出系を汚さないようにする。warm-start も同じガード付き step で特例分岐なし。逸脱 (6) として
計上する — `spectral` でも横長 conv の nullspace 経由で理論上到達可能なため）。**意図的逸脱は 6 件のみ**: (1) **使用時 normalize**（参照は clone のみ）— `SoftCopyTo` が
float buffer を lerp するため soft update（Atari 実構成 tau=0.001）で u/v が非単位ベクトル化しうることへの頑健化。
頑健性は τ ≤ 0.1（または τ=1）として定量契約化（(4) の起動時 config 検証 + 公開 API の変更前 τ 検証 + soft update 後の再正規化契約とセットで **u/v ノルムについて**帰納的に閉じる — σ>0 は保証しない（W lerp 由来の残余は (4)③）。符号反転状態同士の大 τ lerp = σ 退化構成は二重 τ 検証で拒否される）。(2) **FP32 強制** —
power iteration / σ / 除算を Autocast 局所 OFF + FP32（`force_fp32` イディオム）で計算し、W_eff を使う conv /
linear 演算は autocast に任せる。(3) **Q/K/V 別 σ**。(4) **退化 σ の fail-fast（抑止 + 残余明示。2026-08-30 第 5R で「クロージャ」から弱化）** —
①構築時 warm-start σ 検証（`spectral`: 非有限 or ≤0 / `spectral_cap`: 非有限のみ）で init 由来を閉じ、
②起動時 config 検証（SN 層が存在し soft 経路 = `hard_update_interval ≤ 0` なら `soft_update_tau` は finite 必須・
許容集合 {0 ≤ τ ≤ 0.1} ∪ {τ = 1}、それ以外（負・0.1<τ<1・τ>1・NaN/Inf）は両キー名入り fail-fast。下界は正確には
|1−2τ| で τ≥0.9 側も 0.8 を持つが、許容集合は運用意図（小刻み追従か正確コピーか）による意図的制限）+
**公開 API `Network::SoftCopyTo()` 自身の変更前 τ 検証**（source または target に SN entry がある場合、parameter / buffer を
変更する前に同じ有限性・許容集合を検証。違反は τ 実値・許容集合入りで fail-fast し target を一切変更しない。
NetworkModel の検証は設定キー付き早期診断として併存）+
**soft update 時の SN buffer 再正規化契約**（検証通過後、`Network::SoftCopyTo()` が lerp 直後に dst の u/v 再正規化まで
原子的に行う — 公開 API が自身の不変量を守る（呼び出し側手順では SoftCopyTo 単体利用で不変量が破れる）・
PROFILE 対象 — 単発の lerp 下界 ‖(1−τ)u+τu′‖ ≥ 1−2τ は再正規化なしでは帰納しない（反復 lerp で target ノルムが
0 を交差しうる）ため「毎 lerp 前に両辺が単位ノルム」を帰納不変量化する。**保証は u/v ノルムのみで σ>0 ではない**）
で buffer 退化経路を抑止する。③**残余リスクの明示**: soft update の W lerp が target W を σ≈0 近傍へ通す遷移は
排除を証明できない（scalar 反例: 正常な target（W=1, u=v=1）と source（W=−9, u=−1, v=1）の τ=0.1 lerp で
W′=0 — u/v は再正規化後も単位ノルムのまま σ′=0。raw W のスケールは SN 下でゲージ自由）。**この残余は `spectral` のみ**（`spectral_cap` は σ=0 合法・分母 max(1,|σ|)=1 で W_eff=W となり非有限が
発生しない）。検出はメトリクス経路の **online / target validity sentinel**（mode 別 validity を device で OR 集約し
既存の一括 D2H 転送に相乗り（同期回数不変）。群 max は不正 σ を隠すため報告値 65/66 とは別 sentinel。63〜66 の
意味論は online のまま、target は validity のみ同 cadence 集約 — D14③ の検出対象が target の W lerp であるため。
違反時のみ該当 net を re-walk で層名特定）と下流症状（`spectral` の loss 非有限）に委ねる。forward hot path・
SoftUpdate 後の σ 検証は毎 update 同期コストのため不採用。(5) **power iteration ゲートの GradMode 条件** — 参照のゲートは
`self.training` のみで、training+NoGrad の forward では参照は u/v を進め本実装は進めない。参照側では
この状況が起きない（Python に構築時 dummy forward が無い）ため実害はゼロだが、意味論差として計上。
(6) **power iteration の保持ガード** — 候補 normalize のノルムが eps 以下なら当該ベクトルを保持（参照は 0 ベクトル化
して u/v が固定点死 = 以後 σ=0）。`spectral_cap` の zero-W サポートに必須で、`spectral` でも横長の conv 行列
（例 32×288、nullspace 次元 ≥256）の nullspace へ v が入れば非退化 W でも Wv=0 と理論上到達可能なため、
両モード共通適用の逸脱として計上（挙動は参照より安全側）。

## Considered Options

- **(A) u/v buffer + σ 毎 forward 再計算（参照実装 parity。採用）**: 一般実装（PyTorch / BTR / Gogianu）と同系の
  意味論で文献との比較可能性が最大（BTR が新旧どちらの API を使ったかは未検証だが、中核 — 常時除算・勾配 σ 経由・
  training 時 1 iteration — は両 API で同一のため比較基準は揺れない）。soft (Polyak) update × SN は文献に前例が薄いが、一般実装も Polyak で buffer を
  放置または lerp しており、「u/v は近似でよい、σ は使う場で W と突き合わせる」という同じ受け皿に乗る。非 training
  forward の追加コストは matvec + dot × SN 層数で、batch / 空間サイズに依存し小 batch の actor では無視できない
  可能性がある。実コストは 065 PRD 受入 5 のラウンドロビン throughput 計測で確定する。actor snapshot は W 凍結中
  σ 再計算が同値を返すだけ（クローン時に σ を確定して skip
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
  も文字列モード側。棄却し `WeightNormConfig`（キー一律 `weight_norm.mode`、現行 3 値: `none | spectral | spectral_cap`）とする。
- **(F) cap 型 `W/max(1, σ)` / σ detach**: cap 型は文献非標準で BTR 再現の参照点を失うため **`spectral` の数理と
  しては棄却のまま**だが、zero-init サポートの併存モード **`spectral_cap` として採用**した（parity 非主張の別モード。
  2026-08-29 zero-init 衝突の裁定）。σ detach は方向正則化項が消え学習動態が別物になるため棄却
  （σ<1 層が拡大される標準 SN の side effect は注記で明示 — この性質こそが zero-init 非互換の核心で、
  cap モード併存の理由でもある）。
- **(H) 参照を deprecated の旧 `torch.nn.utils.spectral_norm` に pin（当初記述）**: v 先順・初期化ウォームアップ無し。
  2026-08-29 Codex レビューで現行 2.11 系の推奨 API（parametrizations）と不一致と指摘され、現行 API へ pin し直した —
  warm-start が「自前の逸脱」から「参照 parity（init 15 回）」になり、参照先の将来消滅リスクも消える。棄却。
- **(G) Atari baseline 使用 4 型（Linear / Conv1d / Conv2d / ResBlock）に限定し CNBlock / Transformer は
  deferred gate（当初案）**: 「使う構成が来たら足す」だが、型レベルで部分適用の罠（GroupNorm 試験が ResBlock
  内部のみ適用で不公正になった構図のブロック型版）が残り、共通ヘルパゆえ配線コストも低い。2026-08-29 追加グリルで
  全 6 型へ改訂（実使用は config の判断）。ブロック内の per-weight override（init1/init2 と同様の
  `weight_norm1` / `weight_norm2` 式 per-weight `WeightNormConfig` メンバ）は逆に生やさない —
  層選択実験（Gogianu 型）が pin されたら追加する deferred gate。

## Consequences

- OFF（`none`、既定）は挙動完全不変（SN コード不到達。同 seed で学習系列 + `agent_close.anet` 一致を受入で実証）。
  ON も決定的（u/v 初期化は `BuildNetwork` の seed 引数 → `ModuleRandomSource` registry の
  `"spectral_norm"` stream（`ModuleContext` 経由）・構築順固定、power iteration に乱数なし。専用 RNG は parameter 初期化の global 系列を
  消費しないため、**mode 変更で parameter 初期値が不変** = SN 有無の paired 比較と受入突合が成立する）。
- u/v が buffer としてシリアライズ形式・Run 互換に乗る（`spectral` の Run は resume でも u/v が正確に復元される）。
- 既存メトリクス 61/62（生パラメータノルム）は無改修で意味不変。SN 下では生ノルムが制約されないため、
  実効ノルム（63/64。SN 層のみ `spectral`: ‖W‖_F/σ、`spectral_cap`: ‖W‖_F/max(1,|σ|) 換算）と max σ（65/66。
  mode 別の非 clamp σ 推定値: `spectral` は正値検証済み σ_raw、`spectral_cap` は |σ_raw|。cap の実際の分母は
  max(1, 報告値)）を `ComputeParameterNormSplit` の拡張として追加し、**65 の 1 超え成長（`spectral` は常時除算 = 縮小方向射影の表示、`spectral_cap` は cap 発動条件）と
  生 61 / 実効 63 の乖離拡大**で SN の作動を読む（63 単体は ‖W‖F/σ_max = √(stable rank) 相当で特異値分布により
  増減しうるためフラットは保証されない。SN 層ゼロ群では 63/64 = 61/62、65/66 = NaN）。
  収集は `NetworkModule` 基底の既定実装（空を返す）つき仮想関数 `GetSpectralNormEntries()` 経由
  （(weight, u, v) を by-value の `torch::Tensor` で返す — packed のスライス view も安全）。dynamic_cast は
  使わない（AGENTS.md の production 禁止規約に準拠し仮想関数で表現）。buffer 命名規約にも依存しない。
  63〜66 の σ は eval 型（保存済み u/v × その時点の W。update 適用直前測定のため、直後の学習 forward が使う
  σ とは 1 power iteration 分ずれる — 61/62 と同じスナップショット座標系を優先した意図的裁定）。
- eval forward は u/v を変更しない（062 の「測定が学習の数値系列を変えない」契約と整合。probe 部分 forward からも安全）。
- 1 update に複数の training mode forward が走る構成では u/v がその回数進む（決定的・許容）。
- 実験の対照設計への含意: `spectral` は非退化 init 必須のため、既存 zero-init 対照との差には init 変更が交絡する。
  SN の帰属判定は none + he init の対照腕が分離する（065 PRD 検証計画の 4 腕 screening）。
- per-weight override キー、層別 σ、actor クローン時の σ 確定 skip、mode ≠ none × `use_sdpa=false` の旧経路対応、
  power iteration 回数 config、σ₀ 目標係数、σ floor / 遅延有効化、learner の低頻度周期 σ 検証、
  cap の遷移時 warm-start、stochastic module（DropPath / Dropout）の purpose stream 化は deferred / cut
  （065 PRD の複雑度監査）。
- 仕様詳細は `docs/memo/done/065_nn_spectral_norm_10prd.md`（決定事項 D1〜D14、実装仕様、受入基準）。
