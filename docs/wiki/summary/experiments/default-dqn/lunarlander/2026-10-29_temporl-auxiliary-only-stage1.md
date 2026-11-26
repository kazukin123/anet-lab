# TempoRLを行動に反映しないSingleの比較（PRD104 ステージ1）

## 元の材料

[TempoRLを行動に反映しないSingleの比較（PRD104 ステージ1）](../../../../../experiments/default-dqn/lunarlander/2026-10-29_temporl-auxiliary-only-stage1.md) blob:31940db9bb777bc182dec9bfe661a76a27575a92

## 要点

- Singleの行動変化窓は、同seedの無効を3本ともAUCで改善した。
- 通常TempoRLは3seedともSingle各群よりAUCが高い。
- k=1対照は改善を再現しなかった。 平均16.55、範囲−32.78〜50.35で無効と同じ帯。1stepの別Headを足すだけで得が出る、とは読めない。
- 終了窓＋splitは終盤平均が179.99と高めだが、AUCの改善は安定しなかった。
- 同じ重みのfullとsingleは同じ帯で、どちらも無効より高かった（2026-11-04の評価専用Run）。
- 決定的モードの追加比較では、乱数と target 計算を分けると STOP×Single と無効が一致した。最初に見えた STOP の差は無効の別標本として扱う。

## 内容

### 条件と判定方法

2M到達の根拠は固定2M予算・終了コード0・終了直前の定期ログである。全Runの最後の記録値は1,998,848であり、正確な終端experienceがログへ出たという意味ではない。旧中断Runは今回の21本に再利用していない。ステージ2は実行していない。

全群で`skip.max_length=4`、`learner.skip.loss_weight=1`、`sample_weight=uniform`、`priority_lambda=0`、`skip.head.use_dueling=false`、`skip.network_mode=action_out`。無効群のSKIP関連設定は非稼働。Singleは宣言されたActorすべてに適用する。

### AUC・3seedの範囲・EMA200

Single全5群はPRDの定義では無効「以上」であり、無効を「上回る」群はない。通常TempoRLだけが無効を上回る。行動変化窓mergeは終了窓mergeより同seedのAUCが高いのは2/3本であり、seed3は18.08対18.58と僅かに下回る。

### 立ち上がりと終盤

行動変化窓mergeとSTOPは前半の低成績期間を短くする傾向がある。一方、終盤平均は無効と近いか下回る。立ち上がりの改善はサンプル効率として重要だが、312評価点を独立した312回の学習試行とは扱わない。

### 対照・教材と解釈の限界

SKIP有効の各Runでlossと勾配normの有限点を確認した。Singleの判断率は全記録点で1。kmaxは各点1〜4、終了窓2群は各点3.5以上、k=1対照は各点1である。kmaxと抽選kはLearnerの全期間平均であり、policy報酬のexp step区間とは混ぜていない。

### 同じ重みのfullとsingleの評価（2026-11-04）

singleと無効はどちらも毎step判断するので、215対167の差は評価時の行動の選び方ではなく、学習済みの重みの差である。

### ステージ0との関係・残る問い

Qbertのステージ0記録のg=3.8165／0.7662／0.9635と、追加参考のsplit g=0.6149は保持する。元3本の点推定は「評価時の固定が損の主因」の行に入ったが、同じcheckpointの評価時の比較であり、今回のLLの学習実験とは別の証拠である。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
