# kung_fu_master（KFM）

## 今の結論

ボスへ向かう方策はseedによって成立したが、試した設定変更だけでは定着しなかった。
最終スコアの断絶は75,000点の1UPへの到達に対応し、ボスや面の進行を示すものではなかった。
近似優先度が貴重な経験を目立たせるか、新しさや方策の違いが練習を保つかは未検証である。

## 分かっていること

- 観測: 実時間優先100Mのseed 1・2は途中の差が大きく、50.7Mまでの最良評価は68,404と44,417、91.1Mでは59,076と58,583へ近づいた。Runは `run_20260923-020945_atari_kung_fu_master` と `run_20260923-095708_atari_kung_fu_master`。後半の学習を一律に無駄とは言えない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ・探索ブロック01）。
- 観測: q_gapの縮小とaction_churnの減少は成績が落ちなかったseed 2にも出た。同じ2本のRunで、この変化だけではseed 1の頭打ちを説明できない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ4・内部指標）。
- 観測: 2 seedの63,412ゲームの最終スコアの断絶は75,000点の1UPと対応し、ボスや面の進行を示さない。Runは上記seed 1・2。生スコアとクリップ後returnも異なる量である（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、探索ブロック02・03）。
- 観測: Atari-5の100M比較でもaction_churnは5ゲームすべてで減ったが、q_gapの縮小はbattle_zoneだけだった。KFMの指標の動きはゲームをまたいで一様ではない（出典: [Atari-5実験記録](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック04「内部指標」、同節の対象Run）。
- 観測: ボスは距離で現れ、命中時に点は入らず、撃破で2,000点、さらに出口で体力とタイマーが得点へ換算される。命中数には10〜11発の変動が記録されている（出典: [RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、§2「ゲームの仕組み」；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.1）。
- 観測: 実時間優先3 seedのうちseed 3はボス行きを学び、評価の面クリアは15M以降ほぼ100%だった。Runは `run_20260925-130017_atari_kung_fu_master_rr1_100m_seed3`。同じ記録でseed 1・2はその方策を使わなかった（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.2）。
- 観測: 10月の報酬クリップ解除、ライフ区切り、低いε、PER α 0.6、UQEなどの30〜50M比較ではボス行きが定着しなかった。正本がRun名を示していない比較は、2026-10-14〜17の記録の範囲として扱う（出典: [Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、§2；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.2）。
- 分析: クリップ後の報酬密度はボス行きのseed 3が0.042、1面の稼ぎが0.048だった。γ=0.997の視野と、当てても0点・撃破もクリップ後+1という条件では、途中の上達が報酬へ現れにくいと正本は分析している（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.1〜2.3、seed 3の上記Run）。
- 観測と未検証の推測: 正本の実時間優先構成では寿命中の期待抽選回数は約1回、未抽選で消える割合は約1/4。α=0.2では優先度20倍でも抽選重みは約1.8倍となる。一方、ボスの各場面でTD誤差が目立たないという説明自体はまだ推測である（出典: [抽選回数のPRD](../../memo/999_replay_eviction_sample_count_10prd.md)、§2、`run_20261016-042902_kung_fu_master_rr1_50m_noclip200_seed2`；[Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、§2；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.3）。
- 観測: seed 3のボス命中は4〜5Mに跳ね、撃破÷命中も練習とともに上がった。10月のUQEのRunではlane4区分の命中が7.1% / 7.6% / 10.8% / 10.2%で、lane間の違いが練習を保つ形にはならなかった。前者は上記seed 3、後者は正本がRun名を示さない2026-10-17の4〜12Mの集計である（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.6）。
- 文献上の留保: 報酬クリップを外した結果はゲーム・実装で割れ、サーベイのR2D2系1 seedの図の目視ではKFMはクリップ側が高かった。この値をanetのクリップ解除試験の因果説明には使わない（出典: [報酬スケールと割引率のサーベイ](../../../reports/reward_scale_discount_survey_2026-09-23.md)、§3.4・§7）。
- 文献上の留保: 調査したLLM探索・報酬の研究にはAtariで条件付きの実証があるが、KFMのボス攻略を直接改善した実証は見つかっていない。新しさが練習を保つという見込みも、このプロジェクトでは仮説である（出典: [LLM事前知識のサーベイ](../../../reports/llm_prior_knowledge_rl_survey_2026-10-18.md)、§9；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.5）。

## 分かっていないこと

- ボス命中・撃破・出口換算の各場面で、学習側と近似Actorの優先度はどれほど高くなるか（出典: [Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、§2・§3）？
- 新しさや目的の違う方策は、ボスに挑む頻度と撃破までの練習を継続させられるか（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.5・§2.6）？
- RAMのボス命中・撃破の対応は、2面以降でも成立するか（出典: [RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、§2・§7「確認すること」）？

## 撤回・更新

- 2026-10-18: seed 1の50M以降の頭打ちから後半の学習を一律に無駄とする読み → seed 2は伸び続けており一般化しない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ3・探索ブロック01。Runは上記seed 1・2）。
- 2026-10-18: 最終スコアの断絶をボスや面進行の代理とする読み → 75,000点の1UPへの到達であり、進行は別に見る（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、探索ブロック02。Runは上記seed 1・2）。

## 関連

- [Atari環境](../../design/220_atari_env.jp.md) §4、[観測と可視化](../../design/140_observability.jp.md) traceチャネル。
- [CONTEXT.md](../../../CONTEXT.md) の「近似Actor初期優先度」「raw priority」「鏡写しインスタンス」。
- [RunnerAppの実装の地図](../apps/110_runner_app.md)。
- [Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、[LLM ActorのPRD](../../memo/999_llm_actor_10prd.md)（提案は各PRDで扱う）。
