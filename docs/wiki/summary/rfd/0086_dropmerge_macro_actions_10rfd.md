# DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化）暫定 RFD

## 元の材料

[DropMerge のマクロ行動（落ち着くまで待つ・繰り返し回数の離散化）暫定 RFD](../../../memo/0086_dropmerge_macro_actions_10rfd.md) blob:105b7706309717172b417b1b238641c779ca2a04

## 要点

- DropMerge の Env 側で複数 step をまとめる行動を試すための暫定 RFD である。
- 盤面が静止するまで、または設定した物理フレーム上限まで NOOP を続ける行動を候補とする。
- もう一つの候補は、指定した行動と離散的な繰り返し回数の組を ActionSpec に加える方式である。
- 1フレームの NOOP と列ごとの DROP も維持し、従来の細かいタイミングで操作できるようにする。
- Agent 側では Env が宣言する行動数を使い、マクロ行動の効果を従来の行動構成と比較する。

## 内容

### 提案の骨子

繰り返し回数の離散化: 対象の行動と k の組を設定で指定し、「その行動を k 回続ける」行動を足す。最初の対象は NOOP を想定する。

## 主要な概念

- [DropMerge](../../concepts/subjects/dropmerge.md)
- [行動の時間尺度](../../concepts/methods/temporal_abstraction.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
