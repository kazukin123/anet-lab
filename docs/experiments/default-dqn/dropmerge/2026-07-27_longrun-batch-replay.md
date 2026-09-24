# 長期 Run における batch / replay 探索履歴

## メタデータ

| 項目 | 内容 |
|---|---|
| 対象 Agent | DefaultDQNAgent 系 |
| 対象 Env | DropMerge |
| 開始日 | 2026-07-27 |
| 最終更新 | 2026-08-12 |
| 状態 | 一旦クローズ（機構分離・再現性の残件は次 campaign へ保留） |
| 主目的 | 長期 Run の最終成績を優先しつつ、実時間効率を悪化させる冗長な更新を減らす |
| 比較上の注意 | 非決定論・単一 lineage が中心。小差を因果効果として断定しない |

## 現時点のまとめ

1. 成熟した checkpoint から継続する最終成績優先ラインは `batch_size=512`, `replay_ratio=2.0` とする。scratch 初期からの適用は Q 過熱リスクがあり、主力としない。
2. `batch_size=256`, `replay_ratio=1.0` は throughput に優れるが、cy07 の同一 checkpoint 分岐では 107M までに性能差を回収できず、後半も停滞した。
3. B512/RR2 の優位は「batch を大きくした効果」と「1 experience あたりの再生 sample 数を増やした効果」をまだ分離できていない。
4. `batch_size=512`, `replay_ratio=1.0` は、その分離に使える未実行の診断候補である。ただし最終成績を直接改善する事前期待は低い。
5. 設定上の `alpha` を変更した Run は、checkpoint から AdamW の param group options が復元されるため、学習率 A/B としては無効化する。batch / replay の履歴まで無効になるわけではない。
6. ここでの判断は single-seed、非決定論、継続学習 lineage 上のものとし、普遍的な最適値とは扱わない。
7. cy07 から B512/RR2 を再開した追加 100M では、終盤の Eval と Double Suika 率が cy07 終盤を上回った。改善余地は残るが、異常終了により checkpoint は保存されていない。
8. B512/RR2 を scratch から開始した 80M Run では、Q バブル、条件付き NEET、Eval reward 低下が同期した。成熟 lineage での成功を scratch 初期へ一般化しない。
9. IQN 対応後 binary の QR51 / B128/RR1.25 対照は 35.59M で正常 close した。旧 cy01 と同一 window の Eval 差は約2〜3%、Q / loss は同水準であり、QR 互換性と scratch 安定性を再確認した。
10. 後続の QR51 / B256/RR1 scratch 100M は、B128/RR1.25 と同等水準の終盤 reward をより短い実時間で維持した。B256/RR1 は短期探索基準、B512/RR2 は成熟 checkpoint の最終成績優先という段階別の使い分けで campaign を閉じる。

## この文書の読み方・更新規則

- 以下は完成した結論だけを並べた報告書ではなく、判断時点ごとの実験台帳である。
- 各ブロックの `探索観点 → 探索結果 → 考察 → 次の探索` は、その時点で得られていた情報を残す。
- 後続の知見で解釈が変わった場合は、元のブロックを削除せず `判断更新`、`superseded`、`invalidated` を追記する。
- 実効設定の正本は、各 Run artifact の `config/config_data.txt` とする。
- step 数は読みやすさのため概数で表記する。精密な再分析時は artifact の最終行と更新時刻を再取得する。

## 比較用の量

この campaign では、`num_envs=256` のときの learner 更新量を次の二つに分けて扱う。

```text
optimizer updates / exp-step = num_envs * replay_ratio / replay_batch_size
replayed samples / new experience = replay_ratio
```

| 略称 | batch size | replay ratio | updates / exp-step | samples / new experience |
|---|---:|---:|---:|---:|
| B128/RR1.25 | 128 | 1.25 | 2.5 | 1.25 |
| B128/RR0.5 | 128 | 0.5 | 1.0 | 0.5 |
| B256/RR1 | 256 | 1.0 | 1.0 | 1.0 |
| B512/RR2 | 512 | 2.0 | 1.0 | 2.0 |
| B512/RR1 | 512 | 1.0 | 0.5 | 1.0 |

同じ `updates / exp-step` でも、batch size と replay sample 量が違えば同じ学習ではない。
同様に、同じ `samples / new experience` でも optimizer update 回数、target の soft update 回数、勾配ノイズが変わる。

---

## 探索ブロック 01: 初期長期ライン B128/RR1.25

**記録時点:** 2026-07-27〜2026-07-29  
**状態:** superseded  
**確度:** 単一 lineage、非決定論、長期観測あり

### 探索観点

PRD039 / PRD040、IMPALA2 Stem、prev_action などを含む長期 Run 構成で、設定を大きく変えず継続学習した場合にどこまで伸びるかを見る。
開始時点では、学習を入念に行うため `batch_size=128`, `replay_ratio=1.25` を使用した。

### 探索条件・対象 Run

| Run | 親 | 主な差分 | 到達状況 |
|---|---|---|---|
| `run_20260727-021415_apx-ll_cy01` | 初期 Run | B128/RR1.25 | 約200M、正常 close |
| `run_20260728-030241_apx-ll_cy02` | cy01 | 同設定継続 | 約200M、正常 close |

両 Run の実効設定は、それぞれの `config/config_data.txt` で B128/RR1.25、`alpha=1e-4`、`per_alpha=0.2`、PER initial priority fixed 0.3 を確認した。

### 探索結果

- cy01 後半から cy02 にかけて、Double Suika や max rank の改善兆候は残る一方、Train / Eval reward は高止まりと停滞が目立った。
- Q 値、loss、NoLegal 終端などに直ちに Run を破棄すべき暴走は見られなかった。
- 1 exp-step あたり 2.5 回の optimizer update があり、実時間消費に対して追加 update が十分な改善へ変換されていない疑いが生じた。

### 考察

「replay ratio が大きいほど入念に学習できる」という直感だけでは、長期 Run の実時間効率を説明できなかった。
新しい経験が入る速度に対して同じ ReplayBuffer 分布を細かく更新しすぎ、連続 update の限界効用が低下している可能性を候補とした。

この時点では、停滞原因を batch size、replay ratio、PER 分布、探索経験の質のどれかに特定できていない。

### 次の探索

cy01 の同じ checkpoint から一軸ずつ分岐し、以下を比較する。

1. `per_alpha=0.1`
2. `replay_ratio=0.5`
3. `batch_size=256`, `replay_ratio=1.0`

### 判断更新

B256/RR1 と B512/RR2 の後続結果により、B128/RR1.25 は長期主力から外した。
ただし「小 batch が悪い」と単独で結論したわけではなく、B128/RR1.25 の update 密度を含む組み合わせに停滞があった、という証拠として残す。

---

## 探索ブロック 02: PER alpha を 0.1 へ弱める

**記録時点:** 2026-07-29  
**状態:** completed / 採用見送り  
**確度:** low、単一分岐、定量的な優位なし

### 探索観点

長期後半の停滞が、高 TD error の経験へ PER が偏りすぎ、幅広い経験を学習へ反映できていないためかを確認する。

### 探索条件・対象 Run

| Run | 親 | 主な差分 |
|---|---|---|
| `run_20260729-152349_apx-ll_cy02-pera010` | cy01 | `per_alpha: 0.2 → 0.1`。B128/RR1.25 は維持 |

### 探索結果

- 明確な成績改善は確認できなかった。
- 観察上は NEET が起きやすくなった懸念があったが、単一の非決定論 Run であり因果関係は確定できない。
- 「DropMerge では幅広い経験を反映した方がよい」という仮説を支持する決定的な結果にはならなかった。

### 考察

PER の優先度分布だけを平坦化しても、optimizer update の冗長性や新鮮な経験の取り込み速度は直接解決しない。
NEET の印象差も、この Run だけを根拠に `per_alpha` の効果とみなすには弱い。

### 次の探索

PER をさらに動かす前に、同じ cy01 から replay ratio と batch size を変更し、update 密度の影響を優先して調べる。

---

## 探索ブロック 03: RR0.5 で update 密度を下げる

**記録時点:** 2026-07-30  
**状態:** completed  
**確度:** medium-low、同一 checkpoint 分岐、単一 Run

### 探索観点

B128/RR1.25 の停滞に対し、replay ratio を下げて新しい経験に対する連続 update 数を減らす。
学習 sample 数を減らすため、最終成績よりも「余分な update が本当に必要か」の診断を目的とした。

### 探索条件・対象 Run

| Run | 親 | 主な差分 | 到達状況 |
|---|---|---|---|
| `run_20260730-003330_apx-ll_cy02-rr050` | cy01 | B128、`replay_ratio: 1.25 → 0.5` | 約200M、正常 close |

### 探索結果

- 実時間は大幅に短縮した。
- replay sample 数は 1 experience あたり 0.5 まで減ったが、学習進捗は B128/RR1.25 に対して大きく悪化しなかった。
- 少なくとも B128/RR1.25 の追加 update の多くは、実時間に比例した改善を生んでいない可能性が高まった。

### 考察

RR0.5 自体を最終成績設定として採用する根拠ではなく、「update 回数を 2.5 から 1.0 / exp-step へ落としても成立する」という情報が得られた。
一方、replay sample 数まで半減するため、長期の最終成績を優先する構成としては学習機会を捨てすぎる懸念が残った。

### 次の探索

optimizer update は 1.0 / exp-step のまま、sample budget を 1.0 へ戻せる B256/RR1 を試す。

---

## 探索ブロック 04: B256/RR1 を長期ラインへ採用

**記録時点:** 2026-07-30〜2026-08-01  
**状態:** completed / 後に性能主力から superseded  
**確度:** medium、同一 checkpoint 分岐と複数 cycle の継続観測

### 探索観点

RR0.5 で得た実時間効率を活かしつつ、1 experience あたりの replay sample 数を 1.0 へ戻す。
B128/RR1.25 より update 回数を減らし、batch をまとめることで勾配分散も下げる。

### 探索条件・対象 Run

| Run | 親 | 主な差分 | 到達状況 |
|---|---|---|---|
| `run_20260730-152535_apx-ll_cy02-b256` | cy01 | B256/RR1 | 約212M、正常 close |
| `run_20260731-082845_apx-ll_cy03` | cy02-b256 | 同設定継続 | 約200M、正常 close |
| `run_20260801-000736_apx-ll_cy04` | cy03 | 同設定継続 | 約200M、正常 close |

### 探索結果

- cy01 からの分岐では、B128/RR1.25 と RR0.5 より reward 水準が良く、実時間効率も良好だった。
- cy03、cy04 へ継続しても学習は成立し、Double Suika 達成や高 rank の増加が観測された。
- 一方で長期後半は明確な収束というより、ノイズを伴う緩い改善と停滞が混在した。

### 考察

B256/RR1 の改善理由は GPU 効率だけではない。
同じ新規経験量に対して optimizer update を細かく刻みすぎず、より大きな batch で一回の勾配を安定させたことが、B128/RR1.25 より有効だった可能性がある。

ただしこの比較では batch size、update 回数、sample budget が同時に変わるため、どの要因が支配的かは分離できない。

### 次の探索

同じ cy03 checkpoint から、`updates / exp-step=1.0` を維持した B512/RR2 を短期分岐する。
これにより optimizer update 回数を揃えたまま、batch size と sample budget を増やした効果を見る。

### 判断更新

当時は B256/RR1 を実時間と成績のバランスが良い主力とした。
後の cy07 同一 checkpoint 分岐では、B256/RR1 が 107M で停滞し B512/RR2 の性能水準へ届かなかったため、現在は実時間効率の対照として扱う。

---

## 探索ブロック 05: cy03 から B512/RR2 を 50M 分岐

**記録時点:** 2026-08-01  
**状態:** completed / 主力候補へ採用  
**確度:** medium、同一 checkpoint・同一 step 範囲、単一 Run

### 探索観点

B256/RR1 と `updates / exp-step=1.0` を揃えたまま、batch size と replay sample 数を倍増する。
最終成績の改善が、増えた実時間に見合うかを短期で確認する。

### 探索条件・対象 Run

| Run | 親 | 設定 | 比較範囲 |
|---|---|---|---|
| `run_20260801-000736_apx-ll_cy04` | cy03 | B256/RR1 | 0〜50M |
| `run_20260801-184718_apx-ll_cy04_b512-rr200` | cy03 | B512/RR2 | 0〜50M |

### 探索結果

- 0〜50M の target / policy reward の平均水準は、B512/RR2 が約2214、B256/RR1 が約2154で、B512/RR2 が約2.8%高かった。
- Double Suika 率は B512/RR2 が約5.9%、B256/RR1 が約4.6%だった。
- probe の実所要時間は B512/RR2 が約6.7時間、B256/RR1 が約3.9時間で、性能差より実時間差の方が大きかった。
- B512/RR2 は PER の unsampled eviction を抑え、より多くの経験を一度は学習へ反映できる傾向を示した。
- 一方で SpawnBlocked、候補 DROP failure などには悪化方向の黄色信号があり、50M だけで全面採用を断定できる結果ではなかった。
- Q 値、loss、TD error、NEET 系メトリクスには即時棄却を要する破綻はなかった。

### 考察

B512/RR2 は実時間効率では不利だが、最終成績優先なら検証を続ける価値がある程度の差を示した。
この比較で同じなのは optimizer update 回数であり、B512 と RR2 の効果は分離されていない。

当時の判断は「勝ち確定」ではなく、「少し良さそうで、平日は長時間触れないため次 cycle の長期設定として採用する」であった。

### 次の探索

B256/RR1 で完了した cy04 checkpoint から B512/RR2 を cy05 として継続し、200M 単位の長期挙動、後半の傾き、Double Suika、Q / loss / PER / NEET を確認する。

---

## 探索ブロック 06: B512/RR2 の長期継続と alpha 調整

**記録時点:** 2026-08-02〜2026-08-06  
**状態:** batch / replay 観測は completed、alpha 比較は invalidated  
**確度:** batch / replay は medium、alpha 効果は判定不能

### 探索観点

B512/RR2 を長期主力として複数 cycle 継続したとき、短期の優位が維持されるかを見る。
Eval の波立ちや後半停滞に対しては、学習率を `1e-4 → 5e-5 → 7.5e-5` と調整する案も試した。

### 探索条件・対象 Run

| Run | 親 | config 上の主な設定 | 到達状況 |
|---|---|---|---|
| `run_20260802-045009_apx-ll_cy05_b512-rr200` | B256/RR1 cy04 | B512/RR2、`alpha=1e-4` | 約200M、正常 close |
| `run_20260803-080518_apx-ll_cy05_b512-rr200-a5e5` | B256/RR1 cy04 | B512/RR2、config `alpha=5e-5` | 約200M、正常 close |
| `run_20260804-083500_apx-ll_cy06_b512-rr200-a7e5` | a5e5 | B512/RR2、config `alpha=7.5e-5` | 約200M、正常 close |
| `run_20260805-115101_apx-ll_cy07` | cy06 | B512/RR2、config `alpha=7.5e-5` | 約200M、正常 close |

### 探索結果

- B512/RR2 は cycle をまたいでも破綻せず、Double Suika を含む性能改善を継続した。
- 一方、200M 内で伸び続ける区間と高止まりする区間があり、明確な収束判定には至らなかった。
- Eval の波立ちや Q max の一時スパイクはあったが、loss、TD error、NEET、終盤性能を伴う持続的暴走としては観測されなかった。
- 当初は `alpha=5e-5` / `7.5e-5` により劣化が抑えられた可能性を議論したが、後に checkpoint save / load の契約上の問題が判明した。

### 考察

DefaultDQN の checkpoint は AdamW optimizer state と param group options を保存・復元する。
そのため、load 後の optimizer では設定ファイルの `alpha` より checkpoint 内の学習率が優先され、上記 Run は実質的に `alpha=1e-4` を引き継いだ可能性が高い。

したがって、これらを学習率 A/B として解釈してはいけない。
一方、B512/RR2 で長期学習が成立したこと、各 cycle の性能・安定性・PER 観測は、alpha 以外の時系列証拠として残る。

checkpoint の save / load 契約、optimizer の扱い、Run 全体の serialization 方針は `docs/memo/930_serialize_10prd.md` に暫定整理している。

### 次の探索

1. 学習率を再比較する場合は、optimizer options の load 契約を明示してから行う。
2. 先に同じ cy07 checkpoint から B512/RR2 と B256/RR1 を分岐し、batch / replay の判断を再検証する。

### 判断更新: invalidated

`a5e5`、`a7e5` という Run 名と `config_data.txt` の設定値は、実効 optimizer learning rate の証拠にならない。
これらの Run を「低学習率で改善した／しなかった」という根拠から除外する。

---

## 探索ブロック 07: cy07 同一 checkpoint から B512/RR2 と B256/RR1 を再比較

**記録時点:** 2026-08-06〜2026-08-08  
**状態:** completed  
**確度:** medium、同一 checkpoint・単一 Run。B512 側は 37.2M で事故中断

### 探索観点

長期後半の cy07 checkpoint から B256/RR1 へ戻した場合、B512/RR2 の性能を保ちながら throughput を改善できるかを見る。
初期 lineage での B256 採用判断が、成熟した policy にも成立するかを再確認する。

### 探索条件・対象 Run

| Run | 親 | 設定 | 到達・停止 |
|---|---|---|---|
| `run_20260806-143445_apx-ll_cy08` | cy07 | B512/RR2 | 約37.2M、プロセス落ちにより中断、close checkpoint なし |
| `run_20260807-162046_apx-ll_cy08_b256` | cy07 | B256/RR1 | 約107.1M、正常 close |

### 探索結果

#### 同一 step 20〜30M

| 指標 | B512/RR2 | B256/RR1 | 傾向 |
|---|---:|---:|---|
| Train reward EMA | 約2213 | 約2081 | B512優位 |
| Eval target reward | 約2781 | 約2409 | B512優位 |
| Eval policy reward | 約2568 | 約2314 | B512優位 |
| Eval target/policy 平均 | 約2674 | 約2362 | B512が約13.2%高い |
| Eval1 Double Suika 率 | 約18.9% | 約10.5% | B512優位 |
| Eval2 Double Suika 率 | 約13.8% | 約10.0% | B512優位 |
| throughput | 約2500 | 約3429 | B256が約37%高い |
| PER unsampled eviction | 約5.9% | 約29.4% | B512の方が広く再生 |

B256/RR1 は速いが、同一 step では明確に性能水準が低かった。

#### B256/RR1 の継続観測

| B256 step window | Eval target/policy 平均 | Train reward EMA | Eval1 DS | Eval2 DS |
|---|---:|---:|---:|---:|
| 40〜50M | 約2538 | 約2118 | 約15.1% | 約12.2% |
| 50〜75M | 約2529 | 約2117 | 約14.0% | 約12.9% |
| 75〜100M | 約2519 | 約2107 | 約14.4% | 約12.2% |
| 100〜107M | 約2446 | 約2090 | 約14.1% | 約10.8% |

- B256 は 40〜50M までにかなり回復し、B512 の 30〜37M 水準との差を約5%程度まで縮めた。
- しかし 50M 以降は Eval、Train、Double Suika とも横ばいから弱含みで、107M までに B512 水準を回収しなかった。
- Q 値、loss、TD error の発散はなかった。
- reset NOOP margin は負を維持し、timeout は微小で、NEET が停滞の主因である証拠はなかった。
- PER unsampled eviction は約29%に張り付き、B512/RR2 の約6%より多くの経験が一度も sample されずに押し出された。

### 考察

B256/RR1 は wall-clock あたりの step 消化に優れ、短時間で B512/RR2 の近傍まで戻る。
しかし最終成績優先では、後半 plateau と ReplayBuffer の未学習経験の多さが弱点となった。

B512/RR2 の利点候補は二つ残る。

1. 大 batch による勾配推定の安定化
2. RR2 により 1 experience あたり二倍の sample を消費し、ReplayBuffer を広く学習すること

現在の比較だけではこの二つを分離できない。

### 次の探索

最終成績を取りに行く継続 Run は B512/RR2 へ戻す。
機構を分離する診断として、B512/RR1 を同じ saved checkpoint から短く試す余地がある。

### 現在の判断

- **性能主力:** B512/RR2
- **実時間効率の対照:** B256/RR1
- **B256/RR1 の継続:** 107M で十分。追加 200M は優先しない

---

## 探索ブロック 08: B512/RR1 で batch と sample budget を分離する

**記録時点:** 2026-08-08  
**状態:** deferred / 未実行
**確度:** 未実行

### 探索観点

B256/RR1 と同じ `samples / new experience=1.0` のまま batch size を 512 にする。
これにより B512/RR2 の優位が、大 batch 自体によるものか、RR2 の追加 replay sample によるものかを診断する。

### 探索条件・対象 Run

推奨する初期条件は次のとおり。

- 親 checkpoint: `run_20260807-162046_apx-ll_cy08_b256` の保存 checkpoint
- 変更軸: `replay_batch_size: 256 → 512` のみ
- 維持: `replay_ratio=1.0`
- 初回 pause: 30M
- 可能なら同じ親 checkpoint から B512/RR2 も分岐し、lineage と開始 policy を揃える

### 期待される識別

| 結果 | 解釈候補 |
|---|---|
| B512/RR1 が B256/RR1 より高く、B512/RR2 に近い | 大 batch による勾配安定化の寄与が大きい |
| B512/RR1 が B256/RR1 と同程度で、B512/RR2 より低い | RR2 の追加 replay sample と ReplayBuffer coverage の寄与が大きい |
| B512/RR1 が両者より悪い | update 回数 0.5 / exp-step が少なすぎる。soft target update 頻度低下も候補 |

### 探索結果

未実行。

### 考察

B512/RR1 は B256/RR1 と sample budget が同じでも、optimizer update と soft target update の回数が半分になる。
また、PER unsampled eviction は RR に強く依存するため、B256/RR1 の約29%から大きく改善しない可能性が高い。

したがって、現時点の事前期待は「最終成績の新主力」より「B512/RR2 の改善理由を分離する診断」である。

### 次の探索

30M 時点で次を確認して継続可否を決める。

1. 同一 step の Eval target / policy reward と Double Suika 率
2. wall-clock throughput
3. PER unsampled eviction、sample age、priority 分布
4. Q、loss、TD error、NEET 系メトリクス
5. B512/RR2 と比較した 1 時間あたりの性能上昇

明確な性能上昇がなく、PER coverage も B256/RR1 相当なら診断完了として止める。

---

## 探索ブロック 09: cy07 から B512/RR2 を再開した追加 100M

**記録時点:** 2026-08-08
**状態:** completed / 事故中断
**確度:** medium-low、同一 checkpoint・単一 Run・非決定論。close checkpoint なし

### 探索観点

cy07 の保存 checkpoint から最終成績優先ラインの B512/RR2 を再開し、cy07 終盤以降にも改善余地が残るかを見る。
見た目上の plateau が実際の収束なのか、Double Suika 率や Eval 終盤値では改善が継続しているのかを確認する。

### 探索条件・対象 Run

| Run | 親 | 実効設定 | 到達・停止 |
|---|---|---|---|
| `run_20260808-023912_apx-ll_cy08` | `run_20260805-115101_apx-ll_cy07` の close checkpoint | B512/RR2 | `100,170,496 exp_step`、約13時間47分。`bad allocation` により異常終了、close checkpoint なし |

実効設定は Run artifact の `config/config_data.txt` で、親 checkpoint、`replay_batch_size=512`、`replay_ratio=2.0` を確認した。
Double Suika 率は `ep_maxrank_max >= 12` の episode 割合として再集計した。

### 探索結果

#### cy07 終盤と cy08 終盤の比較

| 指標 | cy07 175〜200M | cy08 75〜100M | 傾向 |
|---|---:|---:|---|
| Eval target reward EMA | 約2636 | 約2708 | cy08 が約2.7%高い |
| Eval policy reward EMA | 約2493 | 約2603 | cy08 が約4.4%高い |
| Eval target/policy 平均 | 約2564 | 約2655 | cy08 が約3.5%高い |
| Eval1 Double Suika 率 | 15.66% | 18.01% | +2.35ポイント |
| Eval2 Double Suika 率 | 12.38% | 15.46% | +3.08ポイント |
| 両 Eval の Double Suika 率平均 | 14.02% | 16.73% | +2.71ポイント |
| Train reward EMA の終盤水準 | 約2176 | 約2184 | ほぼ横ばい |
| PER unsampled eviction EMA | 約5.90% | 約5.87% | 同水準を維持 |

- 50〜75M では Eval が一時的に低下し、`q_max_max=49.86` の単発スパイクも発生したが、75〜100M では Eval と Double Suika 率が回復した。
- 終盤の TD error EMA、loss EMA、通常の Q 水準は cy07 終盤と同程度で、Q スパイクに続く持続的な発散はなかった。
- reset NOOP margin は終盤も Eval1 / Eval2 の UQE、Q とも平均が負であり、持続的な reset NOOP 優位や NEET 再発の証拠はなかった。
- throughput EMA は前半の約2400から後半約2146まで低下した。ただし IQN ビルドとの並行実行とメモリ逼迫を伴っており、B512/RR2 固有の低下とは解釈しない。

### 考察

Train reward の見た目は plateau に近いが、cy08 終盤の Eval reward と Double Suika 率は cy07 終盤を上回った。
したがって、cy07 時点で完全に収束していたとは言えず、B512/RR2 には追加の改善余地が残っていた。

一方、cy08 内の改善は単調ではなく、50〜75M の低下から終盤に回復した形である。
single-seed、非決定論、同一 lineage の結果であるため、B512/RR2 の普遍的優位や今後の継続改善までは断定しない。

`bad allocation` は IQN ビルドと並行したメモリ逼迫時に発生し、close checkpoint が保存されなかった。
そのため保存済み lineage の終点は cy07 とし、cy08 は「改善余地を示したが成果を保存できなかった診断 Run」と扱う。

### 次の探索

性能主力は引き続き B512/RR2、実時間効率の対照は B256/RR1 とする。
batch size と replay sample budget の寄与を分離する場合は、探索ブロック 08 の B512/RR1 診断を同じ保存 checkpoint から実施する。

### 現在の判断

- **B512/RR2:** cy07 後にも改善余地が残る証拠を得たが、収束は未確認
- **保存済み lineage の終点:** cy07
- **cy08 の扱い:** 成績傾向の診断証拠。checkpoint 継続元にはできない
- **B512/RR1:** 未実行の機構分離診断として維持

---

## 探索ブロック 10: scratch 開始 B512/RR2 の Q バブルと条件付き NEET

**記録時点:** 2026-08-10

**状態:** completed / 正常 close

**確度:** medium-low、single-seed・非決定論。v0.3.0 release binary と過去 Run の binary 差を含む

### 探索観点

成熟した QR lineage で性能優先設定として採用した B512/RR2 を、scratch 初期から適用しても安定に学習できるかを確認する。
特に、B128/RR1.25 から段階的に B256/RR1、B512/RR2 へ移した過去 lineage と異なり、初期の不安定な価値推定へ replay sample budget 2.0 を直接与えた場合の Q、NEET、終端、報酬を観測する。

### 探索条件・対象 Run

| Run | 親 | 実効設定 | 到達・停止 |
|---|---|---|---|
| `run_20260810-001224_dm_B512R200★` | なし（scratch） | v0.3.0、QR51、IMPALA2-ViT128、B512/RR2 | `80,000,256 exp-step`、約9時間53分。正常 close、checkpoint 899,155,310 bytes、stderr 空 |

Run artifact の `config/config_data.txt` で、`use_qr=true`、`num_quantiles=51`、`replay_batch_size=512`、`replay_ratio=2.0`、`gamma=0.997`、n-step 1、`use_optimistic_target=false` を確認した。
Env は `direct_noop`、`use_instant_drop=true`、stack 4、NoLegal Phase 2（N=60）、prev-action trio / DROP marker ON、NoDrop timeout 100 / penalty -10、`time_penalty=0`、`noop_penalty=0` である。

外部映像証拠として、Train の `2026-08-10 09-55-56.mp4` と Eval の `2026-08-10 10-13-52.mp4` を取得した。保存先は Run artifact と別管理のため、文書からローカル絶対パスは参照しない。

### 探索結果

#### Run 内の推移

| 指標 | 10〜20M | 40〜50M | 70〜80M | 判断 |
|---|---:|---:|---:|---|
| Eval target reward EMA | 約830 | 約734 | 約707 | 初期ピークから低下 |
| Eval policy reward EMA | 約813 | 約642 | 約659 | 初期ピークを回収せず |
| `q_max_real_mean` | 約33.9 | 約437.6 | 約73.9 | 大きな山の後に収縮中 |
| `q_max_real_max` 平均 | 約90.7 | 約6,936.5 | 約654.2 | 40〜50M の区間最大は約25,226 |
| loss EMA | 約0.059 | 約31.9 | 約0.86 | 収縮したが未正常化 |

70〜80Mの最終記録値はEval target約768、policy約696、`q_max_real_mean`約59.5、`q_max_real_max`約402、loss約0.42まで回復した。
しかし区間水準は10〜20Mのrewardを回収せず、正常なB128/RR1.25のQ / loss水準にも戻っていない。

#### B128/RR1.25 scratch 基準との同一 step 比較

| 指標（70〜80M） | B512/RR2 scratch | cy01 B128/RR1.25 scratch | 傾向 |
|---|---:|---:|---|
| Eval target reward EMA | 約707 | 約1621 | B512/RR2が約56%低い |
| Eval policy reward EMA | 約659 | 約1551 | B512/RR2が約57%低い |
| `q_max_real_mean` | 約73.9 | 約8.17 | B512/RR2が約9倍 |
| `q_max_real_max` 平均 | 約654 | 約12.97 | B512/RR2が約50倍 |
| loss EMA | 約0.86 | 約0.026 | B512/RR2が約33倍 |
| Eval1 NoDropTimeout EMA | 約7.6% | 約1.5% | B512/RR2で増加 |
| Eval2 NoDropTimeout EMA | 約10.1% | 約1.3% | B512/RR2で増加 |

比較対象は主要なAgent / Env / NN設定を揃えているが、single-seed、非決定論、binary時点差があるため、B512/RR2の因果効果量とは断定しない。

#### NOOP、終端、映像所見

- 70〜80MのEval NOOP UQE優位率EMAはEval1約84.0%、Eval2約85.1%だった。一方、margin EMAは両Evalとも約+0.0038で、広い状態で薄いNOOP優位が残った。これらのNOOP EMAはartifact上でtrain-step軸のため、`num_envs=256`を掛けてexp-stepへ換算した。
- NoDropTimeout EMAの70〜80M平均はEval1約7.6%、Eval2約10.1%。最終値は約2.4% / 3.4%まで回復したが、終盤window全体では高い。
- 80M最終時点のNoLegal EMAは両Evalとも0、SpawnBlocked EMAは約96.4%。Double Suika（Rank 12）はログで未観測だった。
- reset NOOP marginは負であり、Reset直後からNOOPへ固定される方策ではなかった。映像でも、数回DROPして盤面を形成した後、果実の落下・移動中にNOOPを選び、混雑盤面ではDROPへ戻る状態依存性を確認した。
- Eval Q tableでは、物性待ち状態でNOOPのmeanがDROP群を上回り、分布stdも小さかった。混雑盤面では差が消えたため、全actionの順位が無秩序になった全面的なNN破綻ではない。

### 考察

NOOPでも物理演算と既存fruitのmergeが進み、そのstepのmerge scoreを報酬として受け取れるため、物性安定待ちは局所的に合理的な技能である。
本Runはその技能自体を獲得したが、薄いNOOP優位が広い状態へ汎化し、NoDropTimeout増加とEval reward低下を伴った。したがって「技能の発見には成功したが、適用範囲と価値推定に失敗した」Runと扱う。

Qの絶対値は20〜50Mに大きく膨張し、その後収縮した一方、映像上のaction順位には状態依存の構造が残った。
絶対Qの校正不良と相対順位の部分的な有効性は両立しており、Qバブルが局所的な待機価値を過剰汎化した有力仮説である。

ただし、B512/RR2が原因とは確定しない。single-seed、非決定論、binary差があり、初期経験分布や偶発的な局所 attractor も候補に残る。
成熟checkpointからのB512/RR2は探索ブロック05〜09で性能改善と安定継続を示しているため、その結果を無効化しない。

### 次の探索

`run_20260809-210009_iqn_qr`は探索ブロック11の時点で終了した。
この対照だけでB512/RR2との因果を確定せず、B512/RR1によるbatch sizeとsample budgetの分離を将来課題として残す。

### 現在の判断

- **B512/RR2の成熟checkpoint継続:** 最終成績優先ラインとして維持
- **B512/RR2のscratch開始:** Q過熱と局所技能の過剰汎化リスクがあるため主力から外す
- **本Run:** 正常checkpointを持つ条件付きNEET / Qバブルの診断artifact
- **因果判断:** B512/RR2を第一候補とするが未確定
- **次の対照:** B128/RR1.25の`run_20260809-210009_iqn_qr`は正常終了。結果は探索ブロック11へ更新

---

## 探索ブロック 11: B128/RR1.25対照の終了とcampaignクローズ

**記録時点:** 2026-08-12
**状態:** completed / campaign close
**確度:** medium-low、single-seed・非決定論。35.59Mの短期対照と別campaignの100M結果を含む

### 探索観点

IQN対応後binaryでもQR51 / B128/RR1.25のscratch安定性と旧cy01の傾向を再現できるかを確認し、scratch B512/RR2で観測したQ / NEETバブルがQR経路や新binary全般の破綻ではないことを確認する。
あわせて、後続のB256/RR1 scratch 100Mを参照し、長期Run campaignの運用上の基準を確定する。

### 探索条件・対象 Run

| Run | 親 | 実効設定 | 到達・停止 |
|---|---|---|---|
| `run_20260809-210009_iqn_qr` | なし（scratch） | QR51、IMPALA2-ViT128、B128/RR1.25 | `35,594,496 exp-step`。正常 close、checkpoint 899,156,352 bytes、stderr空 |

Run artifactの`config/config_data.txt`で、`quantile_mode=qr`、`qr.num_quantiles=51`、`replay_batch_size=128`、`replay_ratio=1.25`、`per_alpha=0.2`、`use_optimistic_target=false`、`use_spatial_exploration=true`を確認した。
AutoPause後の再開を含むため、Runフォルダの開始・終了時刻から得られるwall timeはthroughput比較に使用しない。

### 探索結果

30〜35.59Mのmatched exp-step windowを旧cy01と比較した。

| 指標 | 終了済みQR対照 | 旧cy01 B128/RR1.25 | 差・判断 |
|---|---:|---:|---|
| Eval target reward EMA | 約1386.0 | 約1417.7 | 約2.2%低い |
| Eval policy reward EMA | 約1304.3 | 約1343.9 | 約2.9%低い |
| `q_max_real_mean` | 約6.52 | 約6.84 | 同水準 |
| `q_max_real_max`平均 | 約11.10 | 約11.63 | 同水準 |
| loss EMA | 約0.0273 | 約0.0277 | 同水準 |
| Double Suika | 両Eval 0 / 546 | 両Eval 0 / 546 | この早期windowでは未到達 |

Evalの小差はsingle-seed・非決定論の範囲を分離できない。一方、scratch B512/RR2で見られた数十〜数千規模のQやloss膨張はなく、Q / lossの健康域と旧cy01に近い立ち上がりを再現した。
35.59Mで終了したため、最終成績やNoLegal獲得を評価する長期Runとしては不足しているが、互換性・初期安定性の対照としては目的を満たした。

別campaignで完了したQR51 / B256/RR1 scratch 100Mは、90〜100MのEval target reward EMA約1629、policy reward EMA約1584で、過去B128/RR1.25基準と同等水準を維持した。詳細は[IQN導入・QR比較](2026-08-09_iqn.md)を参照する。

### 判断更新

- **scratchの安定参照:** B128/RR1.25。更新密度が高く実時間は重いが、初期Q / lossの基準として残す。
- **100M級ハイパラ探索の高速基準:** B256/RR1。scratch QRでは品質を維持したが、成熟checkpoint分岐では停滞したため、長期最終性能までは保証しない。
- **成熟checkpointの最終成績優先:** B512/RR2。scratch初期からは使わず、十分学習した後の継続候補とする。
- **普遍的な最適値:** 未確定。段階別の使い分けは今回lineageの運用判断であり、batch sizeを自動的に上げれば常に改善するとは結論しない。
- **保存済みlineageの終点:** cy07。cy08は改善余地を示したがcheckpointを保存できなかった診断Runであり、再実行はcampaign closeの条件にしない。

### クローズ判断

このcampaignは、長期lineageをさらに延長して収束を証明するよりも、得られた段階別設定を次のIQN / QR探索へ引き継ぐ段階に達したと判断する。
B512/RR1、batch schedule、ReplayBuffer後半調整、multi-seedは未解決だが、いずれも現lineageのクローズを妨げる必須作業ではない。

---

## 次campaignへ保留する残件

1. **B512/RR1分離診断:** B512/RR2の優位がbatch sizeとreplay sample budgetのどちらに由来するかを切り分ける。機構理解の最優先残件だが、採用設定を決めるための必須Runではない。
2. **段階的batch schedule:** B128→B256→B512のProfiledValue化を検討する。切替時はbatch sizeとRRを一組として定義し、replay credit、prefetch済みbatch、実効sample数、切替stepを診断できる実装が前提となる。
3. **長期plateauのreplay診断:** ReplayBuffer capacity、sample age、PER coverageを観測してから一軸比較する。`per_alpha=0.1`の再試行は優先しない。
4. **Run全体serialize後のoptimizer再探索:** optimizer / scheduler / scalerを含むsave / load契約を確立した後、実効学習率を確認して再探索する。今回の`alpha`分岐は学習率A/Bとして再利用しない。
5. **再現性:** 採用候補のmulti-seedまたは独立scratch Runで、single-lineage依存を切り分ける。
6. **NEET介入:** 学習後Evalで持続的NEETが再発した場合だけ、episode forensicを先に行い、小さい`noop_penalty` / `time_penalty`を一軸比較する。
7. **真の収束:** B512/RR2をさらにcycle継続したときの限界は未確認。現lineageは追加実時間に対する情報利得が小さいため、今回のcampaignでは追わない。

## 次回追記用テンプレート

```markdown
## 探索ブロック NN: タイトル

**記録時点:** YYYY-MM-DD  
**状態:** pending | active | completed | superseded | invalidated  
**確度:** 比較条件、seed、非決定論性

### 探索観点

何を識別したいか。どの仮説同士を比較するか。

### 探索条件・対象 Run

親 checkpoint、変更した一軸、維持した設定、比較 step、停止条件。

### 探索結果

主目的 score、機構の健全性、Env 挙動、throughput / 実所要時間を分けて記録する。

### 考察

観測事実と推測を分ける。比較不能要因があれば明記する。

### 次の探索

次に変える一軸、pause step、採用・棄却条件。
```
