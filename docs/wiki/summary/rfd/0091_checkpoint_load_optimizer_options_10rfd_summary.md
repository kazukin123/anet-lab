---
title: checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる PRD
page_kind: サマリ
material: "[[docs/memo/done/0091_checkpoint_load_optimizer_options_10rfd.md|0091_checkpoint_load_optimizer_options_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
updated: 2026-11-28
---
# checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる PRD

## 元の材料

[checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる PRD](../../../memo/done/0091_checkpoint_load_optimizer_options_10rfd.md) blob:877c1769e7f4afbbb31c7bdbc79753241fdbd518

## 要点

- checkpoint から optimizer を復元すると、引継ぎ Run が指定した学習率などが保存時の値に戻っていた。
- optimizer の学習状態を引き継ぎつつ、新しい Run の設定値を有効にする。
- 対象には学習率、weight decay、Adam の eps と NoisyNet 用 parameter group の設定が含まれる。
- 同じ設定で継続するときの数値は従来の再開と合わせる。
- 設定を変えた場合は、ダンプの表示だけでなく実際に更新で使う値を確認する。

## 内容

### checkpoint で戻っていた設定

DQN は AdamW またはその派生の FusedAdamW を使う。NoisyNet がなければパラメータ群は一つ、有効ならノイズの sigma 用とその他の二つになる。LibTorch の読み込みは更新回数やモーメントだけでなく各群のオプションも復元するため、新しい Run で指定した学習率などが保存時の値に戻っていた。

### 状態を引き継ぎながら設定を反映する

構築と読み込み後に共通の関数でオプションを組み立て、学習率は `learner.alpha`、eps は `learner.adam_eps`、重み減衰は通常群が `learner.weight_decay`、sigma 群が `learner.noisy_weight_decay` から適用する。checkpoint との差は群・オプション・設定キー・値とともに記録する。betas と amsgrad は設定キーを持たず、既定値のまま使う。

同じ設定を使えば両方とも同じ float から double への変換を経るので、厳密な比較でも不要な警告が出ない。群の数自体に互換性がない checkpoint は LibTorch が読み込み時に検出する。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [316 設定管理](../../concepts/framework/316_config.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
