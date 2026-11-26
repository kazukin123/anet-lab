# Spectral Normalization と残差分岐の α ゲート

## 元の材料

[RFD0150　Spectral Normalization と残差分岐の α ゲート](../../../memo/0150_residual_alpha_gate_10rfd.md) blob:1d3c0e65718d69db496474820ec80535649c410c

## 要点

- この暫定案の着手は保留である。残差分岐のゲートをどの条件で試すかを記録し、採用の判断を残した。
- 標準の Spectral Normalization を使いながら、学習開始時の残差ブロックを恒等写像に近づけたい。
- 残差側の重みをゼロ初期化する方法とスペクトル正規化を組み合わせる問題を出発点にする。
- 小さい重みで始める方式、正規化層の γ をゼロにする方式、残差出力へ α を掛ける方式を比較した。
- 将来候補は α ゲートに絞り、初期の写像と学習が始まった後の分岐の成長を評価する。

## 内容

### Problem Statement

「identity から徐々に residual branch を育てる」という生 weight ZeroInit の意味論を維持できない。

### Solution

ResBlock の residual branch の最終出力に学習可能なゲートαを掛け、shortcutとの加算直前に適用する。
学習動態ではない。特にα=0の初回backwardではresidual branchのweight勾配が0になり、まずαが育つ。

### User Stories

`spectral_cap`が期待を外した場合にも設計ラリーを最初からやり直さずに済む。
将来の内部処理差にかかわらず「shortcutへ加える寄与をgateする」という契約を保てる。

### Implementation Decisions

ResBlockだけを対象にするか、既存LayerScaleを持つ他のresidual moduleとの共通化まで行うか。
αの観測用メトリクスが必要か。必要性はscreeningで「gateが開いた時点を説明できない」ことが判明した場合にpinする。

### Promotion Gate

RFD0065の4群screeningを先に実施する。
αゲート実装へ昇格するのは、複製Runの終盤平均ブレ幅を超えて次が同時に成立した場合だけとする。

### Testing Decisions

テストはResBlockの公開forward、parameter転送、Networkのcopy/serialize経路から観測し、test-only Interfaceを追加しない。

### Out of Scope

Near-Zero Initの追加実験。既存設定で構成可能でも、標準SNとのidentity開始両立策にはならない。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
