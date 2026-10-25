# anet-lab Change Log

## v0.3.0 以降の非互換変更

### PRD101 ProfiledValue のキー対応

探索・学習の時間の値をProfiledValueへ統一した。定数はbare、段の定数だけは `phase.[name].value` を使う。非constantの下へ継承したbareは保持するが評価・dumpしない。DropMergeとLunarLanderのlogラダー末端0を維持し、中間laneも旧系列から変わる。

| 旧キー | 新キー |
|---|---|
| `eps_start` | `epsilon.start` |
| `eps_end` | `epsilon.end` |
| `eps_decay_steps` | `epsilon.steps` |
| `uqe_eps_start` | `uqe.epsilon.start` |
| `uqe_eps_end` | `uqe.epsilon.end` |
| `uqe_eps_decay_steps` | `uqe.epsilon.steps` |
| `uqe_tau_start` | `uqe.tau.start` |
| `uqe_tau_end` | `uqe.tau.end` |
| `uqe_tau_decay_steps` | `uqe.tau.steps` |
| `use_spatial_exploration` | `spatial.enabled` |
| `spatial_scale_type` | `spatial.scale` |
| `uqe_use_tail_mean` | `uqe.use_tail_mean` |
| `sync_interval.value` | `sync_interval` |
| `learning_rate.value` | `learning_rate` |
| `learner.alpha` | `learner.learning_rate` |
| `per_beta_start` | `per_beta.start` |
| `per_beta_end` | `per_beta.end` |
| `per_beta_step` | `per_beta.steps` |

時間変化は `type = linear` を明示し、旧decay長が0ならconstantとbare開始値へ移す。spatialの旧start/endは `spatial.epsilon.first/last`・`spatial.uqe.epsilon.first/last`・`spatial.uqe.tau.first/last` へ別に写す。共有baseのspatial epsilonは0.4/0.01。既存のbare `weight_decay`・`gamma`・Munchausen値・soft update τの行はそのまま定数になる。PER βは更新末尾に次の抽選要求用の値を評価し、適用時機を維持する。（PRD101）
