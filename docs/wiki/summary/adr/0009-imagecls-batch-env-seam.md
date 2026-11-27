---
title: ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ
page_kind: サマリ
material: "[[docs/adr/0009-imagecls-batch-env-seam.md|0009-imagecls-batch-env-seam]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-26
---
# ImageCls batch-native env の seam：EnvRepository を single/batch variant 化＋Factory/Builder 再レイヤ

## 元の材料

[ADR 0009](../../../adr/0009-imagecls-batch-env-seam.md) blob:0904d60522ccaa97e780f1c1399ccaf83c04f069

## 要点

- EnvRepository は1個の registry に、class ID ごとに single または batch の factory を排他的に登録する。
- BatchEnvBuilder が single のラップと native batch の生成を引き受け、ImageCls は batch 単位で処理する。
- Env の名前と RunMode は生成時に決める。重複名は識別できるエラーにする。
- 不変な Dataset の manifest/cache を共有し、サンプラー・乱数器・エピソード状態は個体に置く。
- 後の追記では Train/Eval 間の DatasetKey・spec 一致を強制する案を撤回した。Actor を作れるかは対象 Env の spec を受け取る Agent が決める。

## 内容

### 登録と生成

学習・評価とも消費者は batch を要求する。単体と batch を別々の registry に持つと登録・検索が二重になり、同じ factory が両方の能力を持つ形では使わない単体生成の口が残る。単体の登録を取得した Builder は既存 wrapper を使い、native batch の登録はその factory に生成を委ねる。

### 名前と資源

RunManager は train、設定済み評価タグ、EvalPanel の名前を事前に照合する。一度生成に成功した名前は同じ RunManager の寿命中に再利用せず、生成失敗による予約は残さない。Dataset は同じキーかつ同じ設定で共有し、同じキーに異なる設定を与えたときは差分を示して停止する。

### 追記で変わった検証

休止した評価宣言では名前とスキーマを検査し、manifest の作成やデコード用プールの起動はしない。Train と Eval の manifest は別々に検査する。RunManager に ImageCls 特有の DatasetKey の解釈を持たせず、適用済み設定の記録には Module 共通の口を使う。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [ImageCls](../../concepts/subjects/imagecls.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
