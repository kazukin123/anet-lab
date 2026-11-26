# Mixup / CutMix 導入（ImageCls）

## 元の材料

[Mixup / CutMix 導入（ImageCls）](../../../memo/done/0030_imagecls_mixup_cutmix_10rfd.md) blob:6d02548b3176b6ca097548f171d5f3aeb5864233

## 要点

- Mixup と CutMix で、Food-101 の画像と教師ラベルをバッチ内の組み合わせに従って混合する。
- 複数サンプルと損失が必要なので、画像を1件ずつ返す Env から離し、ImageClsLearner の更新時に適用する。
- 混合画像を作る重い Tensor 処理は学習デバイスに置く。
- 比率などのスカラー処理は CPU に残し、教師ラベルを混合比に応じて損失へ反映する。
- ImageClsView は Env が返した混合前の画像とラベルを表示する。

## 内容

### バッチの作成と損失

MixUp は2枚の画像を線形補間し、CutMix はバッチ内で組み替えた相手の画像から矩形を貼り付ける。元画像と相手画像のクラスについてそれぞれ既存の交差エントロピーを計算し、混合比で加重平均する。CutMix の比率には実際に貼った面積を使い、既存の label smoothing は両項に適用する。画像やラベルを混合するのは Learner の更新直前であり、Env と ImageClsView から見えるデータは混合前のままとする。

### 再現条件と測定

Learner の抽選には Agent の seed から名前付きで派生した専用 seed を使う。並べ替えで自分自身と組になった例もそのまま受け入れる。バッチが1件の場合、設定が無効の場合、またはその更新で混合を引かなかった場合は元の画像と教師値で学習する。

混合した教師に対する予測の確からしさは、二つのクラスの予測確率を混合比で足した `target_prob_mix` で追う。境界を含む矩形での面積補正、元画像が残る範囲、バッチサイズ、混合しないときの従来の損失との一致を検証する。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
