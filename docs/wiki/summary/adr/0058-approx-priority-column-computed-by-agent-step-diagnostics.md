# step trace の近似優先度は Agent が出す step 部品が計算し、Actor の設定キーで eval_panel にだけヒントを作らせることはしない

## 元の材料

[ADR 0058](../../../adr/0058-approx-priority-column-computed-by-agent-step-diagnostics.md) blob:765d6d883be85a99ee355b3bbfbb6c0dff13528b

## 要点

- Evaluation Trace の近似優先度は Agent が作る step 用部品で求める。
- Actor と同じヘルパーを使い、aux の Q からヒントを構成し直す。
- replay と同じ n-step の組み立てと推定器を使い、追加 forward は行わない。
- PER の有効・無効にかかわらず DQN 系で計算できる。
- パネル専用の設定キーを Actor に追加せず、n-step の窓を GUI に置かない。

## 主要な概念

- [RunnerApp（110）](../../concepts/apps/110_runner_app.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
