# kung_fu_master（KFM）

## 今の結論

実時間優先3 seedのうちseed3はボス行きを学び、15M以降の評価の面クリアはほぼ100%だったが、2026-10に設定を変えた22本はどれもボス行きに入らなかった。既定の設定を10Mで16 seed回すと7本がボスに挑んだが撃破÷命中は上がらず、定着はseed3を含めて約20本に1本だった。最終スコア75,000点の断絶は1UPへの到達に対応し、ボスや面の進行は示さなかった。

## 分かっていること

- 観測: 実時間優先100Mのseed 1・2は途中の差が大きく、50.7Mまでの最良評価は68,404と44,417、91.1Mでは59,076と58,583へ近づいた。Runは `run_20260923-020945_atari_kung_fu_master` と `run_20260923-095708_atari_kung_fu_master`。後半の学習を一律に無駄とは言えない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ・探索ブロック01）。
- 観測: q_gapの縮小とaction_churnの減少は成績が落ちなかったseed 2にも出た。同じ2本のRunで、この変化だけではseed 1の頭打ちを説明できない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ4・内部指標）。
- 観測: 2 seedの63,412ゲームの最終スコアの断絶は75,000点の1UPと対応し、ボスや面の進行を示さない。Runは上記seed 1・2。生スコアとクリップ後returnも異なる量である（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、探索ブロック02・03）。
- 観測: Atari-5の100M比較でもaction_churnは5ゲームすべてで減ったが、q_gapの縮小はbattle_zoneだけだった。KFMの指標の動きはゲームをまたいで一様ではない（出典: [Atari-5実験記録](../../experiments/default-dqn/atari/2026-09-21_realtime-profile-atari5.md)、探索ブロック04「内部指標」、同節の対象Run）。
- 観測: ボスは距離で現れ、命中時に点は入らず、撃破で2,000点、さらに出口で体力とタイマーが得点へ換算される。命中数には10〜11発の変動が記録されている（出典: [RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、§2「ゲームの仕組み」；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.1）。
- 観測: 実時間優先3 seedのうちseed 3はボス行きを学び、評価の面クリアは15M以降ほぼ100%だった。Runは `run_20260925-130017_atari_kfm_rr1_100m_seed3`。同じ記録でseed 1・2はその方策を使わなかった（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.2）。
- 観測: seed 3 の学習中のゲームの命中率は 3-4M の 10.2% から 4-5M の 51.0% へ跳ね、撃破 ÷ 命中は 1M 窓で 0.24 → 0.40 → 0.71 と上がった。eval（ε=0.01）の面クリア率は 6-8M 54.7%、8-10M 73.8%、18-20M 100%、ε=0 断面は 10.3M で 78.4%、20.4M 以降 100% だった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、探索ブロック 04「比較相手: seed 3 の立ち上がり」。Run: `run_20260925-130017_atari_kfm_rr1_100m_seed3`）。
- 観測: 2026-10 に設定を変えた 22 本（クリップ外し 2・ライフで区切る 1・sticky 0 1・ボス用セット系 5・広い ε の UQE 2・batch 1024 3・IMPALA+ViT の Body 8）は、どれもボス行きに入らず、学習中のゲームで面を抜けたのは各 Run で通算 0〜12 本（0.1% 以下）だった。seed 3 は学習中のゲームの 62% で面を抜けている（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ1・対象 Run）。
- 観測: 報酬クリップを外して 1/200 倍にすると、命中率は 10-30M に 2.2〜3.5% へ増えたが、seed 1 は撃破 0 のまま 30M 以降挑まず、seed 2 の撃破 ÷ 命中は 0.03〜0.18 だった。eval2 40-50M は 56,792 / 30,820 で、基準 3 seed の 37,892〜76,604 の中だった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ2・探索ブロック 01。Run: `run_20261015-210918_kfm_rr1_50m_noclip200_seed1`, `run_20261016-042902_kfm_rr1_50m_noclip200_seed2`）。
- 観測: ライフで区切る（train の `episodic_life`）と 50M までボスに 1 度も当てず、eval のゲーム長は 10M 以降 7,823〜8,183 step、eval2 は 10-15M に 52,315（基準 seed 1 / 3 は 30,081 / 24,163）だった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ3・探索ブロック 01。Run: `run_20261016-114250_kfm_rr1_50m_elife_seed1`）。
- 観測: ボス用セット（sticky 0・低い ε・クリップ外し・PER α 0.6）の seed 1 と、その上の n-step 10、n-step 10 + γ 0.999 は 30M まで命中 0 で、5-10M に 4 万点前後の稼ぎに固まった。UQE を足した腕は 5-10M に命中 14.6% まで挑んだが、撃破 ÷ 命中は 0.02 で、15M 以降は挑まなかった。広い ε（0.4〜0.01）の UQE の 2 本は命中 0.5% 以下だった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ4・5、探索ブロック 02・03。Run: `run_20261017-011011_kfm_rr1_30m_boss_seed1`, `run_20261017-095357_kfm_rr1_30m_boss_n10_seed1`, `run_20261017-184606_kfm_rr1_30m_boss_n10_g999_seed1`, `run_20261017-142416_kfm_rr1_30m_boss_uqe_seed1`, `run_20261018-005352_kfm_uq_s1`, `run_20261018-051750_kfm_uq99_s1`）。
- 分析: クリップ後の報酬密度はボス行きのseed 3が0.042、1面の稼ぎが0.048だった。γ=0.997の視野と、当てても0点・撃破もクリップ後+1という条件では、途中の上達が報酬へ現れにくいと正本は分析している（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.1〜2.3、seed 3の上記Run）。
- 観測と未検証の推測: 正本の実時間優先構成では寿命中の期待抽選回数は約1回、未抽選で消える割合は約1/4。α=0.2では優先度20倍でも抽選重みは約1.8倍となる。一方、ボスの各場面でTD誤差が目立たないという説明自体はまだ推測である（出典: [抽選回数のPRD](../../memo/999_replay_eviction_sample_count_10prd.md)、§2、`run_20261016-042902_kfm_rr1_50m_noclip200_seed2`；[Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、§2；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.3）。
- 観測: seed 3のボス命中は4〜5Mに跳ね、撃破÷命中も練習とともに上がった。10月のUQEのRunではlane4区分の命中が7.1% / 7.6% / 10.8% / 10.2%で、lane間の違いが練習を保つ形にはならなかった。前者は上記seed 3、後者は 4〜12M の集計で、Run は `run_20261017-142416_kfm_rr1_30m_boss_uqe_seed1`（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.6；[KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、探索ブロック 02）。
- 観測: ボス行きに入った seed 3 と入らなかった seed 2 の設定の差は `run.seed` だけだった（出典: [seed 集団の PRD](../../memo/999_seed_population_early_stop_10prd.md)、§2。Run: `run_20260925-130017_atari_kfm_rr1_100m_seed3`）。
- 観測: seed 3 と同じ既定の設定を 10M で 16 seed（s04〜s19）回すと、0.5M 窓のボス命中のピークが 10% 以上の Run は 7 本だった。挑んでいる窓（命中 10% 以上）の撃破 ÷ 命中の最高は s08 の 0.22（5.5M）で、ほかは 0.09 以下。s08 も下がって 7.5M 以降は挑まず、ボス行きは 1 本も定着しなかった。batch 1024（30M、3 seed）と IMPALA+ViT の Body（10M、8 seed）でも定着しなかった（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.2；[KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、探索ブロック 04〜06。16 本の Run 名は同記録の対象 Run。例 `run_20261018-153241_kfm_10m_s08`）。
- 観測: 挑み始めるきっかけは撃破ではなかった。既定 10M の s13 は 4.0〜5.5M に命中 21〜32% で挑んだが 10M まで撃破 0、s12・s18 も撃破 0〜1 本で挑んでは退いた。s08 の学習中のスコアは、挑んでいる 5.0〜6.0M の 11,027〜11,191 から、挑まなくなった 7.5M 以降の 17,521〜18,067 へ上がった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ7・探索ブロック 04。Run: `run_20261018-224452_kfm_10m_s13`, `run_20261018-153241_kfm_10m_s08`）。
- 観測: batch 1024 の 3 seed は、s04・s05 が 30M まで命中率のピーク 1.7 / 2.7%、s06 は 16-20M に約 10〜15% 挑んだが撃破 ÷ 命中は 0.07 以下だった。30M の学習 step は 29,100 で、B256 の 10M（38,200）より少ない（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ8・探索ブロック 05。Run: `run_20261019-084922_kfm_b1k_s04`, `run_20261019-131520_kfm_b1k_s05`, `run_20261019-174028_kfm_b1k_s06`）。
- 観測: IMPALA+ViT の Body の 8 seed は、挑んだ割合（命中率のピーク 30% 以上が 3 本）は既定と同程度だったが、撃破 ÷ 命中の最高は 0.05 で、eval2 7.5-10M の平均は 16,406（既定 16 本は 27,148）、実時間は既定の約 1.28 倍だった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、現時点のまとめ9・探索ブロック 06。Run: `run_20261019-220636_kfm_vit_s04` ほか同記録の 8 本）。
- 観測: RAM メトリクスでは、seed 3 の学習中のゲームのうち `floor_clear` = 2（2 面を抜けた）の 5 本が、どれも `boss_kill` = 1・`boss_hit` = 11 だった。面の番号（0x9F）は 1 ゲームの中で 2 回増える（面 2 → 3）が、2 面のボスの撃破と命中は数えられていない（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、探索ブロック 04「seed 3 の RAM メトリクス」；[RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、Context の RAM 表の 0x9F。Run: `run_20260925-130017_atari_kfm_rr1_100m_seed3`）。
- 分析: seed 3 は安定してボスを倒していたので、ボス戦に勝つための情報は今の観測に入っていると正本は分析している。既定の 16 seed のうち 7 本が挑んだが撃破 ÷ 命中が上がった Run は無く、seed 3 は挑み始めの 0.24 から 0.40 → 0.71 と上がった。既定の設定でボス行きが定着するのは、seed 3 を含めて約 20 本に 1 本である（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.3）。
- 分析: 方策の多様性は ① 行動のゆらぎ（同じネットワークで ε だけを Actor ごとに変える）、② 目的を変えた方策の家族（NGU・Agent57）、③ 別々のパラメータ（独立の seed など）の 3 段に分けられ、seed 3 だけがボス行きに入ったのは ③ の違いだったと正本は整理している（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.6）。
- 文献上の留保: 公表値では、KFM で 20 万点を超えるのは R2D2・NGU・Agent57・MuZero など Actor が数百・フレームが数十億〜数百億の分散学習ばかりで、200M フレーム級の DQN 系はどれも 8 万点未満だった（Atari-5 の公開データ）（出典: [seed 集団の PRD](../../memo/999_seed_population_early_stop_10prd.md)、§2）。
- 文献上の留保: 報酬クリップを外した結果はゲーム・実装で割れ、サーベイのR2D2系1 seedの図の目視ではKFMはクリップ側が高かった。この値をanetのクリップ解除試験の因果説明には使わない（出典: [報酬スケールと割引率のサーベイ](../../../reports/reward_scale_discount_survey_2026-09-23.md)、§3.4・§7）。
- 文献上の留保: 調査したLLM探索・報酬の研究にはAtariで条件付きの実証があるが、KFMのボス攻略を直接改善した実証は見つかっていない。新しさが練習を保つという見込みも、このプロジェクトでは仮説である（出典: [LLM事前知識のサーベイ](../../../reports/llm_prior_knowledge_rl_survey_2026-10-18.md)、§9；[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.5）。
- 観測: 実時間優先 50M で ε ラダーを NoisyNet に置き換える（ε 0）と、eval2 5-10M は 31,447（基準 3 seed は 12,141〜21,347）と速いが、10M 以降は 34,089〜37,596 で横ばいになり、50M までボスへの命中・撃破・面クリアは 0 本だった（出典: [2026-10-11_noisynet-realtime.md](../../experiments/default-dqn/atari/2026-10-11_noisynet-realtime.md)、現時点のまとめ6・探索ブロック 01。Run: `run_20261013-181350_kfm_rr1_50m_nz_eps0`）。

## 分かっていないこと

- ボス命中・撃破・出口換算の各場面で、学習側と近似Actorの優先度はどれほど高くなるか（出典: [Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、§2・§3）？
- 新しさや目的の違う方策は、ボスに挑む頻度と撃破までの練習を継続させられるか（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、§2.5・§2.6）？
- 2 面を抜けたゲームで、ボスの命中・撃破が 1 面の分しか数えられないのはなぜか（2 面のボスの体力が 0xCC に無いのか、ボス戦なしで抜けたのか）。面の番号の 3 面より先と、5 面を抜けたあとの値も確かめていない（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、pending；[RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、既知の限界と後続「0x9F の確認範囲」）？
- 挑んでいる間に撃破 ÷ 命中が上がるかどうかを何が決めるか。seed 3 と s08 は挑み始めの撃破 ÷ 命中がほぼ同じ（0.24 と 0.22）だったが、seed 3 だけが上がった（出典: [KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、pending）？

## 撤回・更新

- 2026-10-18: seed 1の50M以降の頭打ちから後半の学習を一律に無駄とする読み → seed 2は伸び続けており一般化しない（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、現時点のまとめ3・探索ブロック01。Runは上記seed 1・2）。
- 2026-10-18: 最終スコアの断絶をボスや面進行の代理とする読み → 75,000点の1UPへの到達であり、進行は別に見る（出典: [100M実験記録](../../experiments/default-dqn/atari/2026-09-23_realtime-profile-100m.md)、探索ブロック02。Runは上記seed 1・2）。
- 2026-10-19: ボスに挑み始めること（点火）が壁だという見立て → 既定の 16 seed のうち 7 本が挑んでおり挑むことは珍しくなく、壁は挑んでいる間に撃破 ÷ 命中が上がらないことにある（出典: [内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、冒頭の 2026-10-21 改訂の注記・§2.3）。
- 2026-10-23: RAM の面の番号（0x9F）は面 1 → 2 の切り替わりしか確かめていない → 面 2 → 3 でも増え、死亡とゲームオーバーでは増えないことを実 Run で確かめた。残る問いは 2 面のボスの命中・撃破の数え方（出典: [RAMメトリクスPRD](../../memo/082_atari_ram_progress_metrics_10prd.md)、冒頭の 2026-10-23 改訂・Context の RAM 表；[KFM 2026-10 の実験記録](../../experiments/default-dqn/atari/2026-10-15_kfm-boss-route.md)、探索ブロック 04。Run: `run_20260925-130017_atari_kfm_rr1_100m_seed3`）。

## 関連

- [AtariEnv のゲーム固有 RAM メトリクスは番号付き汎用キーで参照し、意味はゲーム別の RAM 定義が持つ](../../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)。
- [Atari環境](../../design/220_atari_env.jp.md) §4、[観測と可視化](../../design/140_observability.jp.md) traceチャネル。
- [CONTEXT.md](../../../CONTEXT.md) の「近似Actor初期優先度」「raw priority」「鏡写しインスタンス」。
- [RunnerAppの実装の地図](../apps/110_runner_app.md)。
- [Evaluation Trace PRD](../../memo/094_evalpanel_step_history_10prd.md)、[内的報酬のPRD](../../memo/999_intrinsic_reward_novelty_10prd.md)、[LLM ActorのPRD](../../memo/999_llm_actor_10prd.md)（提案は各PRDで扱う）。
