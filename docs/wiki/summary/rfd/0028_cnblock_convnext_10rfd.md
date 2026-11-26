# CNBlock（ConvNeXt v1）導入

## 元の材料

[CNBlock（ConvNeXt v1）導入](../../../memo/done/0028_cnblock_convnext_10rfd.md) blob:9c3ff5088db13d91c8e79a08b67f1eb153aa9fc7

## 要点

- ImageCls のネットワーク候補として ConvNeXt v1 の CNBlock を追加する。
- 7×7 の depthwise 畳み込みとチャネルを拡張する inverted bottleneck で特徴を変換する。
- LayerNorm、LayerScale、Stochastic Depth を残差経路と組み合わせる。
- 既存の NetworkModuleFactory とネットワーク設定から構成する。
- GRN を使う ConvNeXt v2 は扱わず、まず v1 と既存の ResNet 系を比較できるようにする。

## 内容

### 確定した設計判断

ブロック名は "CNBlock"（`ConvNeXtBlock` は長いため短縮）。V1/V2の接尾辞は付けない。
既存 ResBlock と同じ `NetworkModuleFactory` パターンで `NetworkModuleRepository` に登録する自己完結ブロックとして実装する。

### 前提事実

CNBlockFactoryでも `cn.droppath_rate` 検証にそのまま再利用する。

### 設計方針

`kernel_size`/`stride`/`dilation` はResBlockのような個別config化をせず、`padding = kernel_size / 2` の固定計算のみ（stride=1・dilation=1固定は判断15の通り）。
`head_init_scale`（論文のfine-tuning用パラメータ。スクラッチ学習では1.0固定=no-opのため省略）。

### 受け入れ基準

既存LayerNorm回帰なし: `eps`未指定時に従来通り1e-5になり、LunarLander/Hybrid/ViT branchの既存挙動が変わらないこと。
CNBlock channel不一致: 入力channel数が`cn.channels`と異なる場合、初回forwardで`ANET_SYSTEM_ERROR`（shape不一致でのtensor add破綻より先にfail-fastする）。
CNBlock norm_type=none: `cn.norm_type=none`で内部normステップがスキップされ、`layernorm2d`時と出力が変わること（norm有/無のcode-path確認）。未知の`norm_type`で`ANET_SYSTEM_ERROR`。

### テスト項目リスト

LayerNorm2d config必須検証（num_channels未指定/0でエラー）。
CNBlock droppath train時のshortcut近似一致（droppath_rate≈0.99）。
CNBlock LayerScale gamma初期値確認（layerscale_init反映、<=0で無効化）。

### 後続

DropMerge向けCNBlock branch設計（stem/downsample調整を含む）を後継PRDとして着手する（判断7）。
必要に応じてConvNeXt v2（GRN対応）を別PRDで検討する。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [ImageCls](../../concepts/subjects/imagecls.md)
- [316 設定管理](../../concepts/framework/316_config.md)
