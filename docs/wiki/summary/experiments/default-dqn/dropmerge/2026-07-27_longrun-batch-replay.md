# 長期 Run における batch / replay 探索履歴

## 元の材料

[長期 Run における batch / replay 探索履歴](../../../../../experiments/default-dqn/dropmerge/2026-07-27_longrun-batch-replay.md) blob:ac092a6acbb43c26d26bc44e8b9de8a01cf6fbe6

## 要点

- 成熟 checkpoint の B512/RR2 の大幅優位という初期判断は、cy07・cy08 の長期評価を調べて弱めた。
- cy08_b256 の40–100M と cy07 終盤の差は1〜3%にとどまる。
- 以前の約13.2%差は cy08 初回の20–30M の単一窓との比較だった。
- B512/RR2 の優位は「batch を大きくした効果」と「1 experience あたりの再生 sample 数を増やした効果」をまだ分離できていない。
- 後続の QR51 / B256/RR1 scratch 100M は、B128/RR1.25 と同等水準の終盤 reward をより短い実時間で維持した。B256/RR1 は短期探索基準、B512/RR2 は成熟 checkpoint の最終成績優先という段階別の使い分けで campaign を閉じる。

## 内容

### この文書の読み方・更新規則

後続の知見で解釈が変わった場合は、元のブロックを削除せず `判断更新`、`superseded`、`invalidated` を追記する。

実効設定の正本は、各 Run artifact の `config/config_data.txt` とする。

### 比較用の量

この campaign では、`num_envs=256` のときの learner 更新量を次の二つに分けて扱う。

同じ `updates / exp-step` でも、batch size と replay sample 量が違えば同じ学習ではない。 同様に、同じ `samples / new experience` でも optimizer update 回数、target の soft update 回数、勾配ノイズが変わる。

### 探索ブロック 01: 初期長期ライン B128/RR1.25

「replay ratio が大きいほど入念に学習できる」という直感だけでは、長期 Run の実時間効率を説明できなかった。 新しい経験が入る速度に対して同じ ReplayBuffer 分布を細かく更新しすぎ、連続 update の限界効用が低下している可能性を候補とした。

### 探索ブロック 02: PER alpha を 0.1 へ弱める

観察上は NEET が起きやすくなった懸念があったが、単一の非決定論 Run であり因果関係は確定できない。

「DropMerge では幅広い経験を反映した方がよい」という仮説を支持する決定的な結果にはならなかった。

### 探索ブロック 03: RR0.5 で update 密度を下げる

replay sample 数は 1 experience あたり 0.5 まで減ったが、学習進捗は B128/RR1.25 に対して大きく悪化しなかった。

### 探索ブロック 04: B256/RR1 を長期ラインへ採用

cy01 からの分岐では、B128/RR1.25 と RR0.5 より reward 水準が良く、実時間効率も良好だった。

一方で長期後半は明確な収束というより、ノイズを伴う緩い改善と停滞が混在した。

### 探索ブロック 05: cy03 から B512/RR2 を 50M 分岐

Double Suika 率は B512/RR2 が約5.9%、B256/RR1 が約4.6%だった。

### 探索ブロック 06: B512/RR2 の長期継続と alpha 調整

B512/RR2 は cycle をまたいでも破綻せず、Double Suika を含む性能改善を継続した。

### 探索ブロック 07: cy07 同一 checkpoint から B512/RR2 と B256/RR1 を再比較

B256/RR1 は 実時間 あたりの step 消化に優れ、短時間で B512/RR2 の近傍まで戻る。 しかし最終成績優先では、後半 plateau と ReplayBuffer の未学習経験の多さが弱点となった。

### 探索ブロック 08: B512/RR1 で batch と sample budget を分離する

したがって、現時点の事前期待は「最終成績の新主力」より「B512/RR2 の改善理由を分離する診断」である。

### 探索ブロック 09: cy07 から B512/RR2 を再開した追加 100M

Train reward の見た目は plateau に近いが、cy08 終盤の Eval reward と Double Suika 率は cy07 終盤を上回った。 したがって、cy07 時点で完全に収束していたとは言えず、B512/RR2 には追加の改善余地が残っていた。

### 探索ブロック 10: scratch 開始 B512/RR2 の Q バブルと条件付き NEET

NOOPでも物理演算と既存fruitのmergeが進み、そのstepのmerge scoreを報酬として受け取れるため、物性安定待ちは局所的に合理的な技能である。 本Runはその技能自体を獲得したが、薄いNOOP優位が広い状態へ汎化し、NoDropTimeout増加とEval reward低下を伴った。したがって「技能の発見には成功したが、適用範囲と価値推定に失敗した」Runと扱う。

### 探索ブロック 11: B128/RR1.25対照の終了とcampaignクローズ

30〜35.59Mのmatched exp-step windowを旧cy01と比較した。

### 次campaignへ保留する残件

Run全体serialize後のoptimizer再探索: optimizer / scheduler / scalerを含むsave / load契約を確立した後、実効学習率を確認して再探索する。今回の`alpha`分岐は学習率A/Bとして再利用しない。

## 主要な概念

- [DropMerge](../../../../concepts/subjects/dropmerge.md)
- [Replay ratioとlane窓](../../../../concepts/methods/replay_ratio_lane_window.md)
- [勾配ノイズとbatch](../../../../concepts/phenomena/gradient_noise_batch.md)
