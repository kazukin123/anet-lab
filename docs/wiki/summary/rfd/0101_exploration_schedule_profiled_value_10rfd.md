# 探索・学習パラメータの時間変化

## 元の材料

[RFD0101　探索・学習パラメータの時間変化](../../../memo/done/0101_exploration_schedule_profiled_value_10rfd.md) blob:52057c781fa3190e6ce81c4e6ca114f11bc9ced8

## 要点

- 探索率を下げる時刻と並列環境ごとの探索率の差を独立に指定する。
- 一定値、直線、cosine、その組み合わせに指数減衰を加え、一定期間の終了後に探索率をゼロにできる。
- lane ごとのラダーでは両端の値をそれぞれ時間の関数にし、その時点の値から各 lane を補間する。
- PER の重み補正、学習率、割引率など学習側の値にも共通の ProfiledValue を使う。
- 値を固定した場合と旧設定を移した場合に同じ系列を保つことを確認する。

## 内容

### 目的と問題

行動を選ぶときの探索のパラメータ（ε、UQE の ε と τ、lane ごとのラダー）と、学習側のパラメータ（PER β、学習率、weight_decay、target の soft update の τ、Munchausen の α と entropy_tau、割引率 γ）を、学習の進みに合わせて変化させる形を、既存の `ProfiledValue`（一定・直線・cosine・cosine の繰り返し・段の組み合わせ）と、新しく足す指数で自由に書けるようにする。
探索で書ける形が直線 1 本だけ。 ε は「start から end へ decay_steps で直線に下げ、あとは end のまま」しか書けない。途中で切る、しばらく保ってから下げる、指数で下げる、が書けない。
lane ごとのラダーは時間で変わらない。 spatial exploration を有効にすると、start と end は「lane の両端の値」の意味に変わり、学習の最後まで同じラダーのままになる。ラダーを途中で狭める・消す、が書けない。
学習側に時間変化の器が無い。 時間で変わるのは PER β（直線）だけで、学習率・weight_decay・soft update の τ・Munchausen の α と entropy_tau・γ は定数しか書けない。学習率を途中で下げる、β を予算の半分で収束させる、を Run の中で書けない。

### 起草時点の現状

BTR の公開実装（`Agent.py` の `EpsilonGreedy.update_eps` と `choose_action`）: ε は遷移ごとに `eps -= (eps - eps_final) / eps_steps`（＝時定数 eps_steps の指数減衰、既定 2M、下限 0.01）。学習の開始前と、予算の半分（`total_frames / 2`）までの間だけ使い、その後は NoisyNet だけで探索する。
学習側の定数のキー（`learner.alpha`・`weight_decay`・`soft_update_tau`・`munchausen.*`・`gamma`）は、現用の設定 5 ファイルと bat 3 本に計 92 か所ある（`learner.alpha` 22、ほか 70。2026-10-24 の grep）。

### ゴール・非ゴール・成功の測り方

退けた案を名指しし、それが本当の制約か（クリーンブレークで移せないか）を添える。
探索の方式そのもの（ε-greedy・UQE・Thompson sampling）の追加や変更。方式は方策の種類の問題で、パラメータの時間変化という本 PRD の関心とは別。制約ではなく対象を絞る判断で、後から同じ器の上に足せる。
MuZero の温度 `temp_start/end/decay_steps`。試作 Agent の探索パラメータで、本線の DQN 系と設定の器が違い、今は本線に戻す予定が無い。ProfiledValue の器自体は使えるので、本線に戻すときに同じ形で足す。制約ではなく対象を絞る判断である。
段階 1 の効果の確かめ方（後続の実験の設計）: BTR の ε の形（指数で減らして 25M で 0）を設定だけで書き、anet の BTR 寄せの qbert で、ε 0.01 を残した群（`run_{t}_a5_qb_btrlike_s1` と同じ設定）と 25M で 0 にする群の 2 群を各 3 seed、50M で回す。見るのは 45〜50M の学習中の第 8 面突破率（29,000 点以上かつ round_clear ≥ 8）で、同設定 seed のブレ幅を超えて差があるかで判定する。改善が無くても結果を実験記録に残して比較を終える。

### 検証と受入

単体テスト（切り替えだけで済む）: 時間の値（`epsilon` など）とラダーの両端（`spatial.epsilon.first/last` など）を両方書いた設定で、`spatial.enabled` だけを変えると、それぞれの側の値で動く。使わない側の値は評価しない。値そのものの検査（0〜1 の範囲など）は両方の側に行う。
実 Run: BTR の ε の形で Atari の短い Run を回し、`epsilon` の scalar が 25M の境目で 0 になることを確かめる（予算を短くした確認用の設定で境目を手前に置いてよい）。qbert の成績比較は受入に含めない。
単体テスト（書き戻し）: 学習率と weight_decay の phased で、段の境目の update から optimizer の全 group の lr と weight_decay が新しい値になり、σ 専用 group の weight_decay は `noisy_weight_decay` のまま。
単体テスト（γ）: 定数の γ では、抽選時に作り直した収益と Push 時の保存値が全遷移で一致し、学習は保存値の経路を使う。schedule の γ では、境目の前に Push した遷移を境目の後に抽選すると、収益と bootstrap が後の γ で計算される。初期優先度のヒントは Push 時の γ。報酬スケーラの post_scale は構築時の γ。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
