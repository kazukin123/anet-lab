# Agent の更新結果を表示する View の案

## 元の材料

[RFD0139　Agent の更新結果を表示する View の案](../../../memo/frozen/0139_agent_update_result_view_10rfd.md) blob:02e71ad7492c1f4287170c8a1ee980137e50371a

## 要点

- この View の構想は凍結中である。MixUp・CutMix 後の学習入力を表示したいという実運用上の要求が生じれば再開する。
- ImageClsView が示すのは環境が出した画像と正解ラベル、Actor の推論結果である。
- MixUp・CutMix で Learner が作る混合画像や教師ラベルは、環境の観測には含まれない。
- 係数や混合範囲など、損失の計算に実際に使った情報を見られる仕組みを検討する。
- 候補は学習結果や LearnEvent からデータを渡す表示で、どの情報をどの寿命で持つかは未確定である。

## 内容

### Context

現行の `ImageClsView` は Env 側 View として実装されており、`TrainEvent` の `experience.state.obs` と `action_info` を表示する。

### Problem

Agent/Learner 側で生成・計算された学習時情報を、人間が確認する標準的な可視化手段が弱い。
paired target、lambda、CutMix bbox、mix mode を確認できない。

### Goal

Env View とは別に、Agent/Learner の update result 由来データを表示する仕組みを用意する。

### Non-Goals

`ImageClsView` を MixUp/CutMix 専用 UI に変更しない。
Env の `state.obs` に Agent/Learner 内部の mixed data を逆流させない。

### Proposed Direction

`ImageClsView` のような Env View は、従来通り Env observation と Actor 推論結果を表示する。

### Acceptance Criteria

`ImageClsView` は pre-mix の Env observation 表示として維持される。
表示のために Env state や replay/experience の契約を歪めない。

### Follow-Up Candidates

現行 `BatchUpdateResult::GetTensor` / `GetTensorVector` の利用実績を調査する。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [319 GUI 共通基盤](../../concepts/framework/319_gui.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
