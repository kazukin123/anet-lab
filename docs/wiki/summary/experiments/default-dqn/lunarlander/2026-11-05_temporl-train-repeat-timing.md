# TempoRLのTrainの繰り返しの時期と長さの比較

## 元の材料

[TempoRLのTrainの繰り返しの時期と長さの比較](../../../../../experiments/default-dqn/lunarlander/2026-11-05_temporl-train-repeat-timing.md) blob:e36d72429dce7f6d34135daad699c8c6d0729944

## 要点

- warmupの間だけ繰り返す群は、基準の立ち上がりを再現しなかった。
- 50万exp stepで繰り返しを止めても、その後は基準と同じ帯だった。
- スキップ長を無作為に選んでも、基準と同じ帯だった。
- これまでの比較と合わせると、LunarLanderの得は序盤のTrainの繰り返しで説明できる。
- 同じRunのfullとsingleは、立ち上がりでも同じ帯だった。
- warmupだけの群と、繰り返しの無い過去の群との差は確定しない。

## 内容

### 条件と判定方法

評価のActorは全群で同じで、full（`[eval]`・`[greedy]`）とsingle（`[eval_single]`・`[greedy_single]`）を同じ重みで走らせた。補助のlossは全群で動いている。warmupの間に専用の無作為方策は無く、学習前のネットワークのままTrainの方策で動く。8,192 exp stepはlane 1本あたり32 stepで、replay容量の1.6%にあたる。

### 繰り返しと教材

Trainのスキップの指標はtrain_step×256、kmaxはlearn step×32＋8,192でexp stepへ直した。範囲は3 seedの最小〜最大である。

### 比較に使った過去のRun

繰り返しの無い過去の群。報酬は`21_eval/02_policy_reward`で、どの群も評価を毎step判断で行う。ビルドも評価のsessionも今回と違うので、同じseed番号でも同じ軌跡ではない。

### 解釈の限界と残る問い

どの群も3 seedである。warmupだけの群ではs1だけが早く抜け（0.25〜0.5Mの局の長さ219.4）、基準とのRun全体の範囲の重なりもs1による。

50万までの証拠は0.5Mより後だけである。fullの点はs1とs3が501,248 exp stepの直前まで基準と全点一致し、s2は14,848 exp stepから違う。singleの点はs1が130,048 exp stepから違うが、理由は確かめていない。同じ設定の区間なので、0.5Mまでの一致は同等の証拠にならない。

### Runと実効config

実効configは各Runの`config/config_data.txt`をそのまま複製し、公開できない値が混ざっていないか走査した。完了済みRunの`metrics_cache.db`の全点から集計した。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
