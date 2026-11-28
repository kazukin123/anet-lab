---
title: EvalPanel の Joystick と操作マッピング案
page_kind: サマリ
material: "[[docs/memo/0159_input_mapping_joystick_10rfd.md|0159_input_mapping_joystick_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/apps/110_runner_app.md|RunnerApp（110）]]"
  - "[[docs/wiki/concepts/framework/319_gui.md|319 GUI 共通基盤]]"
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# EvalPanel の Joystick と操作マッピング案

## 元の材料

[RFD0159　EvalPanel の Joystick と操作マッピング案](../../../memo/0159_input_mapping_joystick_10rfd.md) blob:dbeef8356deacc60c8ba7bbed8fda5b9bbb563f2

## 要点

- 固定された矢印とテンキーだけでは Atari の複合行動を入力できない問題を扱う。
- Joystick の方向・スティック・ボタンとキーボードを、行動へのマッピングとして設定できるようにする案である。
- ゲームごとの行動集合の違いは Env の別名表で吸収し、Atari 共通の入力表を使えるようにする。
- 入力の解釈と、EvalPanel で強制行動を実行する経路の責務を分ける。
- 環境で選べない行動やマッピング不整合をどう知らせるかを含め、実機で確かめる。

## 内容

### 手動操作の問題

EvalPanel の矢印とテンキーだけでは Atari の方向と発射を組み合わせた行動を十分に選べない。キーボードとゲームパッドの入力が、環境ごとに定義の異なる離散行動へどう対応するかが論点だった。

### 起草時に確認した Joystick API

使用していた wxWidgets の wxJoystick は、接続台数・利用可否・スティック位置・ボタン状態・十字キーの角度をそれぞれ問い合わせられた。SetCapture で移動やボタンのイベントを受ける方法もある。原本は押された組み合わせを読み取る用途では状態の問い合わせで足りると見込んだ。

### 暫定の操作案と検証

方向やボタンの組み合わせとキーボードの入力を設定で行動に対応させ、環境が持つ行動の別名から実際の行動を解決する。入力の取り込みと評価での強制行動を分ける。最終的な入力表、未対応の行動の扱い、実機での軸やボタンの対応はグリルと検証に残した。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [Atari横断](../../concepts/subjects/atari.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
