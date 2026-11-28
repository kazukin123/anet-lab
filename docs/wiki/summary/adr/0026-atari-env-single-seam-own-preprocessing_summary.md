---
title: AtariEnv は SingleDiscreteEnv として登録し、前処理は env 内自前・frame stack は既存 stucker/RB に委ねる
page_kind: サマリ
material: "[[docs/adr/0026-atari-env-single-seam-own-preprocessing.md|0026-atari-env-single-seam-own-preprocessing]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
updated: 2026-11-28
---
# AtariEnv は SingleDiscreteEnv として登録し、前処理は env 内自前・frame stack は既存 stucker/RB に委ねる

## 元の材料

[ADR 0026](../../../adr/0026-atari-env-single-seam-own-preprocessing.md) blob:e790eb5a3934b2d1f9ac37005cd0d4eca5f23f84

## 要点

- ALE の1インスタンスを SingleDiscreteEnv として登録し、lane 並列化は共通基盤に任せる。
- Atari 固有のフレームスキップ、最大値合成、縮小は環境で行う。
- 環境の出力と ReplayBuffer への保存は uint8 の単フレームにする。
- batch-native な AtariVectorEnv の内蔵 stack をそのまま保存すると容量が増え、autoreset の契約も重なる。
- 後の ADR 0086 では DefaultDQN の Actor と replay 抽選が Perception の FrameStack 定義を共有する。

## 内容

### 前処理と境界

スプライトのちらつきを抑えるため、単なる ALE 内蔵スキップでは省かれる2フレームの最大値合成を行う。縮小などは合成入力で検査できるように ALE と分け、thread 数や並列実行は既存設定を使う。

### 打ち切りの時点

追記では episode_frames と truncation の基準を ALE の episode frame 番号にそろえた。独自カウンターとの食い違いを避け、リセット時に行う NOOP や FIRE を含む時点を ALE 側で表す。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
