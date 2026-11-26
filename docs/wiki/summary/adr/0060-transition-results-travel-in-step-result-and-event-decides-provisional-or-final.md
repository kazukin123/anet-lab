# 遷移の結果は Step の結果で運び、速報値か確定値かは読むイベントで分ける

## 元の材料

[ADR 0060](../../../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md) blob:5ad93c9e1d179e8a4f32048ff5cd1b92c3975729

## 要点

- Step の結果自身が、その遷移で得た値のキーに答える。
- Env 本体は問い合わせ時点の状態を答え、自動リセットは Step の中で行う。
- 速報値と確定値は train と episode_end など、読むイベントの違いで区別する。
- EpisodeEndEvent は終了 Step の結果を持ち、評価セッションもそこから集約する。
- 通知後までリセットを遅らせる旧方式は、Atari の非同期学習の速度低下を理由に取り消した。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
