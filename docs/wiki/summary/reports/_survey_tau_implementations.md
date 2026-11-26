# 実装サーベイ: IQN / FQF / QR-DQN における τ(quantile fraction)生成

## 元の材料

[実装サーベイ: IQN / FQF / QR-DQN における τ(quantile fraction)生成](../../../../reports/_survey_tau_implementations.md) blob:21173189868b4c76281db65db8cc8dab6ffc57fb

## 要点

- 調査日の主要 IQN 実装では、学習時に一様乱数から τ を作る例が中心だった。
- 決定的な行動を指定したときだけ等間隔に切り替える実装がある。
- d3rlpy にも学習と推論で τ の生成を切り替える分岐がある。
- 調べた実装から層化・準乱数で τ を生成する例は確認できなかった。
- 調査対象は2026年8月12日に取得したブランチで、commit の固定には限界がある。

## 内容

### 2. DeepMind DQN Zoo / rlax / Acme【公式】

運営: Google DeepMind。DQN Zoo は「DeepMind で開発された DQN 系エージェントのリファレンス実装集」を自称する公式リポジトリ。

### 3. pfrl(旧 ChainerRL、Preferred Networks)【公式フレームワーク】

運営: Preferred Networks(ChainerRL の PyTorch 後継)。τ は `torch.rand` の一様乱数を 3 箇所(K: 行動選択用 / N: online / N': target)で独立に引く。

### 4. Tianshou(清華大学 thu-ml)【公式フレームワーク】

運営: THU-ML(清華大学)。IQN の τ 生成はネットワーク側 `ImplicitQuantileNetwork.forward` で一様乱数。

### 5. d3rlpy(Takuma Seno 氏、オフライン RL ライブラリ)【公式フレームワーク(個人発・現在は広く利用)】

作者: Takuma Seno(妹尾卓磨)氏。学習時は一様乱数、推論時は等間隔 linspace に切り替える明示的な train/eval 分岐を持つ。

### 6. Stable-Baselines3 Contrib【公式フレームワーク】

運営: Stable-Baselines3 チーム(Antonin Raffin 氏ら)。IQN/FQF は無く QR-DQN のみ。τ は `quantile_huber_loss` 内で固定中点として生成。

### 7. CleanRL【公式フレームワーク】

運営: CleanRL プロジェクト(Costa Huang 氏ら)。アルゴリズム一覧に IQN / QR-DQN / FQF は無い。分布型は C51 のみ。

### 8. DI-engine(OpenDILab / SenseTime 系)【公式フレームワーク】

運営: OpenDILab。IQN 用 `QuantileHead` は一様乱数だが、行動選択側の τ に beta function(リスク歪曲)を通せる点が特徴。FQF 用 head は cumsum + 中点 detach(toshikwa 系と同型)。

### 9. 有名第三者リファレンス実装

作者: Toshiki Watanabe 氏(Qiita @ku2482)。Tianshou の FQF が明示的に依拠する再現実装。IQN の τ は一様乱数(online N 個 / target N' 個を独立に)。

### 11. GitHub issue / discussion での τ サンプリング議論

τ サンプリングの偏り・stratified 化を主題にした issue / discussion は、Dopamine / Tianshou / pfrl / dqn_zoo / toshikwa / BY571 の各リポジトリを対象に検索した範囲では見つからなかった。

## 主要な概念

- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
