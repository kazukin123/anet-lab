# 標準 Spectral Normalization と residual identity 開始を両立する α ゲート暫定 PRD

> 状態: **暫定メモ / deferred gate**。本 PRD は実装着手を意味しない。
> 起点: [PRD 065](done/065_nn_spectral_norm_10prd.md) の zero-init × Spectral Normalization 裁定後に、
> 代替案として Near-Zero Init / α ゲート / normalization γ=0 の 3 案を比較した。
> 現時点で将来候補として残すのは **α ゲートだけ**とする。
> 関連: [ADR 0032](../adr/0032-spectral-norm-self-impl-buffer-semantics.md)。

## Problem Statement

PRD 065 では、ResBlock の最終 conv weight を完全に zero-init する現行方式と、常時
`W / σ` を使う標準 `spectral` は両立しないと裁定した。

- 初期状態では `σ=0` となり、標準 SN の除算が未定義になる。
- weight がゼロを離れた直後も、標準 SN は微小 weight を実効 spectral norm 約 1 へ射影するため、
  「identity から徐々に residual branch を育てる」という生 weight ZeroInit の意味論を維持できない。

PRD 065 はこの問題に対し、`W / max(1, |σ_raw|)` を使う独立モード `spectral_cap` を採用した。
これは完全な zero-init、初期 residual 出力ゼロ、初回 backward からの最終 conv weight 勾配を維持する。

一方、4 腕 screening の結果によっては、次の組み合わせが同時に観測される可能性がある。

1. zero-initによるidentity開始には効果がある。
2. 参照準拠の標準 `spectral` にも効果がある。
3. しかし `spectral_cap + zero-init` は両者の実用上の利点を回収できない。

この場合に備え、**生 weight ZeroInitではなく、residual branchの出力を0で初期化した学習可能αでゲートする方式**を、
標準 `spectral` とidentity開始を組み合わせる別候補として記録する。

## Solution

ResBlock の residual branch の最終出力に学習可能なゲートαを掛け、shortcutとの加算直前に適用する。

\[
z = shortcut(x) + \alpha f_{SN}(x), \qquad \alpha_0 = 0
\]

pre-activationでは`y=z`、post-activationでは既存どおり`y=activation(z)`とする。αゲートは既存の
post-add処理を変更せず、加算へ入るresidual寄与だけを制御する。

`f_SN` 内のweightはHe等の非退化initと標準 `spectral` を使う。したがってSN stateは正常に初期化でき、
α=0によって初期residual寄与だけを厳密にゼロにする。

この方式は **ZeroInit互換とは呼ばない**。保証するのは初期residual出力ゼロであり、生weight ZeroInitの
学習動態ではない。特にα=0の初回backwardではresidual branchのweight勾配が0になり、まずαが育つ。

PRD 065の`weight_norm.mode`は変更しない。αゲートはweight normalizationとは別関心なので、
将来実装する場合も`weight_norm.mode`へ`spectral_alpha`等の複合モードを追加せず、ResBlock側の独立した
設定Interfaceとして扱う。

## 検討経緯: zero-init × SN の3案

| 案 | 内容 | 現行コードでの実現性 | 裁定 |
|---|---|---|---|
| 1. Near-Zero Init | 最終convを完全な0でなく、極小分散の乱数で初期化する | `init2.mode=trunc_normal`と`trunc_std`により設定だけで構成可能 | **cut** |
| 2. αゲート | 非退化weightへ標準SNを適用し、residual出力へ0初期化した学習可能αを掛ける | ResBlockにgate parameterと乗算が無いため実装が必要 | **本PRDの唯一のdeferred候補** |
| 3. normalization γ=0 | 最終conv後のnormalization affine scaleを0初期化してresidual出力を止める | 現行Atari ResBlockはpre-activationかつnormなし。γ初期値Interfaceも無い | **cut** |

### 1. Near-Zero Initを棄却した理由

`W = εW₀` としても、標準SNでは次のようにεが相殺される。

\[
\frac{W}{\sigma(W)}
=
\frac{\varepsilon W_0}{\varepsilon\sigma(W_0)}
=
\frac{W_0}{\sigma(W_0)}
\]

したがってNear-Zero Initは`σ=0`という数値上の構築失敗を避けるだけで、標準`spectral`の初期residualを
identity近傍には保たない。`spectral_cap`ならσ<1の間はNear-Zeroを維持できるが、同モードは完全なzero-initを
既に正式サポートするため、中間的な近似initを追加する価値がない。

### 2. αゲートを残す理由

αゲートは、標準`spectral`の数理を変えずに初期residual寄与を厳密にゼロへできる。
`spectral_cap`とは異なる仕組みなので、4腕screeningで「標準SNとidentity開始は個別には有効だが、
cap方式では組み合わせられない」と示された場合の代替仮説になる。

ただし、生weight ZeroInitとの差を契約として保持する。

| 初期化方式 | 初期residual寄与 | 初回の最終conv weight勾配 | 最初に育つもの |
|---|---:|---:|---|
| 生weight ZeroInit | 0 | 通常は非ゼロ | 最終conv weight |
| αゲート (`α=0`) | 0 | 0 | α |

この差があるため、αゲートの結果を「ZeroInitの再現」と解釈しない。

### 3. normalization γ=0を棄却した理由

normalization γ=0も、最終的にはresidual branchへ0初期化した乗算ゲートを置く方式であり、αゲートと
主要な機序が重なる。一方でnormalizationを追加すると、activation分布、scale不変性、統計処理、
ResBlock内の順序まで同時に変わる。

現行Atari ResBlockはpre-activationかつ`norm_type=none`であり、内部normを有効化してもnorm2は最終convの
直前に置かれる。引用案の「最終conv直後のnormalization γ=0」とは構造が異なる。αゲートだけを調べたい局面で
normalizationまで導入すると因果分離が悪いため、本件では採らない。

将来、normalization自体を採用する独立の理由が生じた場合、そのPRDでγ初期値を検討する。

## User Stories

1. 実験オーナーとして、標準SNを変更せずidentity開始を組み合わせる候補を残したい。そうすることで、
   `spectral_cap`が期待を外した場合にも設計ラリーを最初からやり直さずに済む。
2. 実験オーナーとして、αゲートと生weight ZeroInitの初回勾配差を明記したい。そうすることで、両方式を
   同じ学習動態として誤解しないで済む。
3. NN設定者として、αゲートとweight normalizationを独立に選びたい。そうすることで、`none` / `spectral` /
   `spectral_cap`の既存意味論を複合モードで汚さずに済む。
4. 実装者として、gateの適用位置をresidual branch全体の終端へ固定したい。そうすることで、pre/post activationや
   将来の内部処理差にかかわらず「shortcutへ加える寄与をgateする」という契約を保てる。
5. レビュアーとして、実装開始条件を4腕screeningの観測へ結び付けたい。そうすることで、仮説だけで実験腕と
   Interfaceを増やさずに済む。
6. 実験結果の読者として、αゲート腕を標準SN＋zero-output gateと呼びたい。そうすることで、BTR parityや
   ZeroInit互換を過剰主張せずに済む。
7. 保守者として、αゲートを使わない既存ResBlockのparameter集合とforwardを不変にしたい。そうすることで、
   PRD 065および既存Runの解釈へ影響を与えずに済む。

## Implementation Decisions（暫定）

| # | 決定 | 内容 |
|---|---|---|
| D1 | 候補の位置づけ | αゲートだけをdeferred候補として残す。Near-Zero Initとnormalization γ=0は本PRDから実装しない |
| D2 | SNとの関係 | 実験候補は非退化init＋標準`spectral`＋α初期値0。`spectral`の数理・state・メトリクス契約は変更しない |
| D3 | 関心の分離 | αゲートはResBlockのresidual制御であり、weight normalization modeではない。独立設定にする |
| D4 | 適用位置 | residual branch内の全変換後、DropPathとshortcut加算の直前で、branch寄与全体へ適用する |
| D5 | 初期出力契約 | α=0では加算へ入るresidual寄与が0になる。pre-activation・同一shapeなら出力は入力と一致する。post-activationでは既存のpost-add activation、downsample時は既存shortcut射影が残るため、block出力全般をidentityとは呼ばない |
| D6 | 初回勾配契約 | α=0ではαに勾配が流れ、residual branchのweight勾配は0になり得ることを正常動作としてpinする |
| D7 | 既存挙動 | gate無効時はparameterを追加せず、forward・serialize・copy・soft copy・cloneの既存挙動を変えない |
| D8 | 実装時期 | §Promotion Gateを満たすまで実装しない。PRD 065の実装・4腕screeningを妨げない |

### 実装へ昇格するときに裁定する事項

- αをblock単位scalarとするか、normalization非依存のper-channel scaleとするか。最小Interfaceはscalarだが、
  per-channel方式はチャンネル間のgate勾配相殺を避けられる可能性がある。
- 設定キー名と、無効 / 有効を表す設定形。0を無効化sentinelとして再解釈せず、gateの有無と初期値を
  現行契約として明示できる形にする。
- ResBlockだけを対象にするか、既存LayerScaleを持つ他のresidual moduleとの共通化まで行うか。
  初回実装では単一利用箇所のための汎用Scale moduleを先に作らない。
- αの観測用メトリクスが必要か。必要性はscreeningで「gateが開いた時点を説明できない」ことが判明した場合にpinする。

## Promotion Gate

PRD 065の4腕screeningを先に実施する。

| 腕 | 構成 | 読み方 |
|---|---|---|
| A | `none + zero-init` | 現行基準 |
| B | `none + He init` | Aとの比較でzero-initの効果を測る |
| C | `spectral + He init` | Bとの比較で標準SNの効果を測る |
| D | `spectral_cap + zero-init` | Aとの比較でzero-init環境のcap効果を測る |

αゲート実装へ昇格するのは、複製Runの終盤平均ブレ幅を超えて次が同時に成立した場合だけとする。

1. **A > B**: zero-init / identity開始に実効がある。
2. **C > B**: 標準`spectral`に実効がある。
3. **Dが実用候補にならない**: `spectral_cap + zero-init`が上の利点を十分に回収しない、またはcap固有の
   under-clamp / 学習動態が問題になる。

昇格後の追加腕を暫定的にEとする。

| 腕 | 構成 | 主比較 |
|---|---|---|
| E | `spectral + He init + α=0` | C vs Eで標準SN下のzero-output gate効果、D vs Eで実用候補比較 |

次の場合はclosedのままとする。

- AとBの差がブレ幅内で、zero-initの実効が確認できない。
- BとCの差がブレ幅内、またはCが悪化し、標準SNの実効が確認できない。
- Dが勝ち腕または十分な実用候補となり、別gate方式を追加する理由がない。

## Testing Decisions（昇格時）

テストはResBlockの公開forward、parameter転送、Networkのcopy/serialize経路から観測し、test-only Interfaceを追加しない。

1. α=0の初回forwardでresidual寄与が厳密に0となり、pre/post activationとdownsampleの各構成で既存のshortcut・post-add処理だけを適用した期待出力と一致する。
2. 非退化init＋標準`spectral`で構築でき、SNのwarm-start検証を通る。
3. 初回backwardでαの勾配がfiniteかつ非ゼロとなる入力を構成する。
4. α=0では最終conv weight勾配が0であることを、方式固有の期待挙動として確認する。
5. αを非ゼロにした次のbackwardでは最終conv weightへfiniteな勾配が流れる。
6. gate無効時はα parameterが存在せず、既存ResBlockと同一seedでparameterと出力が一致する。
7. copy / soft copy / clone / checkpoint round-tripでαが他の登録parameterと同じ契約に乗る。
8. 同一seedの2 Runでαを含む学習系列が一致する。
9. 追加乗算を含むON構成のthroughputをラウンドロビンで測定する。許容幅は昇格時にpinする。

## Out of Scope

- PRD 065の`spectral` / `spectral_cap`の数理、power iteration、buffer、seed、copy、メトリクス契約の変更。
- PRD 065の4腕screeningへのE腕の先行追加。
- Near-Zero Initの追加実験。既存設定で構成可能でも、標準SNとのidentity開始両立策にはならない。
- normalization γ=0、BatchNorm / GroupNorm / LayerNormの追加、normalization affine初期値Interfaceの拡張。
- αゲートをZeroInit、BTR準拠、ReZero完全再現と呼ぶこと。
- 汎用Mul / Scale moduleや新しいNN DSLの先行追加。
- CNBlockの既存LayerScale契約の変更。

## Further Notes

- 本PRDは「標準SNとidentity開始を同時に欲しくなった場合の退避候補」を失わないための記録であり、
  現時点の推奨実装ではない。
- 現時点の第一候補はPRD 065の`spectral_cap + zero-init`である。これはαゲートと異なり、初回backwardから
  最終conv weightへ勾配が流れる生weight ZeroInitの動態を維持する。
- 将来αゲートを実装しても、結果は「標準SN＋zero-output gate」として報告し、ZeroInit腕とは分けて読む。
