# DefaultDQNAgent / DropMerge 探索記録

DropMerge の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-07-27〜 | [長期 Run: batch / replay 探索](2026-07-27_longrun-batch-replay.md) | 長期継続学習における batch size、replay ratio、PER、実時間効率 | active |
| 2026-08-09〜 | [IQN 導入・QR 比較](2026-08-09_iqn.md) | IQN 32/32 の成立性、QR51 比較、Q 値バブルと一時的 NEET、fixed-grid control | active |

## 現時点の判断

最終更新: 2026-08-10

| 設定・探索 | 現在の扱い | 根拠の概要 |
|---|---|---|
| `batch_size=512`, `replay_ratio=2.0` | 成熟checkpoint継続時の最終成績優先ライン | 同一checkpointの短期比較と後続長期Runで、B256/RR1より高い性能水準を示した。ただしscratch開始の80M RunではQバブルと条件付きNEETが発生した |
| `batch_size=128`, `replay_ratio=1.25` | scratch基準・次の対照 | cy01の立ち上がりとQ / lossは安定。再開済みQR対照Runで再確認中 |
| `batch_size=256`, `replay_ratio=1.0` | 実時間効率の対照。性能主力としては見送り | throughput は高いが、cy07 分岐の 107M では後半が停滞し、B512/RR2 の水準を回収できなかった |
| `batch_size=512`, `replay_ratio=1.0` | pending | B256/RR1 と同じ sample budget で optimizer update 回数だけを減らす診断候補。未実行 |
| `alpha=5e-5` / `7.5e-5` | invalidated | checkpoint load により AdamW の param group options が復元され、設定ファイル上の alpha が実効学習率になっていない可能性が高い |
| `per_alpha=0.1` | 採用見送り | 単一分岐では明確な改善がなく、NEET 増加の懸念もあった。確度は低い |
| IQN 32/32 random | 100M基準Run完了。長期主力への採用は保留 | Q/NEETバブルは自力鎮火し正常close。90–100MでQR51よりEval target rewardが約16%低く、Double Suika未観測 |
| IQN fixed-grid control | 次の優先診断 | current / target / train-policy tauを`fixed`へ寄せ、IQN固有のsampling varianceとQRとの差を切り分ける |

詳細な条件、当時の解釈、判断の更新履歴は campaign 文書を参照してください。

## DropMerge 共通の学習・挙動知見

以下は複数のcampaignを横断して使う運用上の用語と判断基準である。
Run固有の数値は各campaign文書、終端契約は[PRD039](../../../memo/039_dropmerge_nolegal_adjudication_10prd.md)、prev-action観測は[PRD040](../../../memo/040_dropmerge_prev_action_obs_10prd.md)を正本とする。

### NOOP 系方策の用語

| 用語 | この文書での意味 | 扱い |
|---|---|---|
| NEET | legal DROPがあるのにNOOPを長く選び、進行を止める方策 | 学習後のEvalで持続し、報酬や終端品質を落とす場合は失敗 |
| 高度NEET | timeout直前にDROPを混ぜるなど、単純な100連続NOOPを避けながら待機状態へ戻る方策 | NoDropTimeout件数やReset指標だけでは検出できない |
| 物性安定待ち | 落下・転がり・連鎖mergeが続く間、追加DROPを控える短いNOOP | 局所的には合理的で、人間的な技能になり得る |
| NoLegalDrop | 現在fruitを置けるDROPがない盤面でNOOPを選び、Envが投了として受理した終端 | 失敗行動ではなく、詰み認識と投了の獲得結果 |

Train序盤のNEETは「待ち続けると不利」という経験を得る学習材料になり得る。
問題とするのは、学習が進んだ後のEvalで長いNOOPが再発し、Eval reward、Double Suika、終端品質を同時に悪化させる場合である。

### 物性安定待ちが成立する理由

`direct_noop`ではNOOPも1 RL stepとして物理演算を進める。既に盤面上にあるfruitがそのstepでmergeすると、merge scoreはNOOPを選んだtransitionの報酬になる。
`use_instant_drop=true`はNOOP中の物理進行を消す設定ではなく、現在の構成ではNOOP 1回につき物理1 frameが進む。

stack 4のgrid、`no_drop_timeout_ratio`、prev-action trio / DROP markerがあるため、方策は盤面の移動、直前action、timeout接近を組み合わせて「動いている間は待ち、落ち着いたらDROPする」技能を表現できる。
この短い待機自体は悪くない。次を伴った時点で、合理的技能から過剰汎化したNEETへ変わったと判断する。

- NOOP優位が物性移動のない広い状態へ拡大する
- NoDropTimeoutや長いNOOP chainが増える
- Eval reward、Double Suika、max rankが低下する
- Q / loss / TD errorの異常と同期する

### 高度NEETとQ値の読み方

高度NEETはReset直後ではなく、数回DROPした後の特定盤面から始まることがある。
その場合、`reset_noop_uqe_margin` / `reset_noop_q_margin`が負でも、episode途中のNOOP attractorは否定できない。

また、Q絶対値の膨張とaction相対順位の維持は両立する。
全actionに共通するoffsetやscaleが過大でも、NOOPとDROPの順位が状態依存で残れば、見た目には一貫した方策として動く。
Qバブルの判定では画面の単一状態だけでなく、`q_max_real_mean/max`、loss、TD error、Eval rewardを併用する。

Eval policyがUQEの場合、Q tableの`Mean`最大actionが必ず選択されるわけではない。
risk-tail scoreも含めた選択なので、action highlightとMean順位の差をバグとみなさない。

IQNではQ / loss / grad応答がQRより大きく見えたRunがあるが、QR51かつ`use_optimistic_target=false`のscratch RunでもQ / NEETバブルを観測した。
したがってIQNやoptimistic targetは増幅候補にはなり得ても、NEETの必要条件ではない。

### NoLegalDropの獲得には時間がかかる

NoLegal Phase 2は、詰み盤面が物理微動していてもblocked persistenceがN frame続けばNoLegalDropとして受理し、終端分類を予測可能にする仕組みである。
settled盤面は最初のNOOPで受理し、unsettled盤面は現在の実効設定ではN=60までに受理する。

これは方策へ投了を直接教えるreward shapingではない。
Agentは、満杯近くの比較的稀な状態で「どのDROPも失敗する」と認識し、SpawnBlockedになるDROPではなくNOOPを選ぶ必要がある。
さらにunsettled時は、裁定までNOOPを維持しなければならないため、n-step 1では不利・有利の伝播に時間がかかる。

実例では、scratch B512/RR2 Runの80M最終時点で両EvalのNoLegal EMAは0、SpawnBlockedは約96.4%だった。
一方、成熟したcy07の175〜200M windowではNoLegal EMAがEval1約46.7%、Eval2約45.2%、SpawnBlockedが約37〜39%まで変化した。
この差から、NoLegal獲得はEnv実装の成立確認と方策学習を分け、長期指標として観測する。

NoLegal、NoDropTimeout、SpawnBlockedは次のように区別する。

| 終端 | 方策・盤面の意味 | 主な診断 |
|---|---|---|
| NoLegalDrop | 全DROP不能を認識してNOOPした | NoLegal EMA、terminal blocked frames、blocked run count |
| NoDropTimeout | legal DROPがあるのにNOOPを続けた | timeout EMA、`no_drop_timeout_on_candidate` |
| SpawnBlocked | 選んだDROPがspawn位置で失敗した | SpawnBlocked EMA、`blocked_drop_on_candidate` |

Phase 2有効時に`no_drop_timeout_on_candidate≈0`なら、NoDropTimeoutへ詰み盤面が大量に誤分類されている証拠は弱い。
Phase 2は終端契約の修正であり、高度NEETを直接治す機構とは扱わない。

### 診断は複数指標を組み合わせる

| 観測 | 有力な解釈 |
|---|---|
| Reset marginが正 | 盤面形成前からのReset NEET |
| Reset marginが負、episode中NOOP優位率が高い | 盤面形成後の局所attractor / 高度NEET |
| NOOP優位率が高いがmarginが薄い | 小さな順位差が広い状態へ汎化している可能性 |
| timeout増加、`timeout_cand≈0` | legal DROPを残した回避可能な待機 |
| NoLegal増加、SpawnBlocked低下 | 詰み認識と投了方策の獲得候補 |
| Q / loss上昇とEval reward低下が同期 | 価値校正不良を伴う方策悪化 |
| 物性移動中だけ短くNOOPしrewardが維持・上昇 | 合理的な物性安定待ち |

NOOP EMA系metricの一部はhistorical artifactでtrain-step軸である。
同じtag名でも`metrics.scalar`定義を確認し、必要なら`train_step * num_envs`でexp-stepへ換算する。Reset指標だけでepisode途中や終端を推定しない。

### 介入の順序

1. Run artifactの実効設定とstep軸を確認する。
2. NoLegal / NoDropTimeout / SpawnBlockedの終端契約とcandidate指標を確認する。
3. 映像またはepisode forensicで、物性移動中の短い待機と高度NEETを分ける。
4. batch / replay、Q / loss / TD、探索方策など、待機価値を増幅した学習側要因を一軸比較する。
5. 持続的Eval NEETが残る場合だけ、小さい`noop_penalty`または`time_penalty`を一軸A/Bする。

`noop_penalty` / `time_penalty`は合理的な物性安定待ちも抑制するため、最初から基準設定へ入れない。
`use_spatial_exploration`導入時期とNEET再発時期には観察上の近さがあるが、長期汎化への利点もあり、因果は未確定として保留する。
