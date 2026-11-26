# DefaultDQNAgent / DropMerge 探索記録

## 元の材料

[DefaultDQNAgent / DropMerge 探索記録](../../../../../experiments/default-dqn/dropmerge/README.md) blob:ac924f7259e8bd5fb64e86d4d19f3f9a10498c2f

## 要点

- 長期学習、QR/IQN、Atari から移した設定、勾配ノイズ、TempoRL の各 campaign を引ける。
- instant drop を有効にしても、NOOP 中は物理が 1 frame 進む。
- UQE を使う評価では Mean 最大の行動と選択された行動が食い違い得る。
- cy07 の長期窓の終端内訳は、この索引にある独自集計が正本である。
- 終端の分類、物性移動中の待機、長すぎる待機を区別し、学習側の条件は一つずつ比較する。

## 内容

### campaign と独自の分析

長期継続と scratch、学習設定の比較と評価専用 Run を区別する。stack 4 の grid、直前行動、timeout 接近の観測を合わせると、物理が動く間だけ待つ行動は表現できる。実際にその技能を獲得したかは別途映像と trace で見る。

### 終端分類と診断

cy07 の 175〜200M は NoLegal EMA が Eval1 約46.7%、Eval2 約45.2%、SpawnBlocked 約37〜39%。Phase 2 は終端裁定を直す。実効設定と終端指標、映像、学習条件の順に調べる。

## 主要な概念

- [DropMerge](../../../../concepts/subjects/dropmerge.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [勾配ノイズとbatch](../../../../concepts/phenomena/gradient_noise_batch.md)
