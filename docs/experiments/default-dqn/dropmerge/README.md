# DefaultDQNAgent / DropMerge 探索記録

DropMerge の DefaultDQNAgent 系 Run に関する探索索引です。

## campaign

| 期間 | 文書 | 主題 | 状態 |
|---|---|---|---|
| 2026-07-27〜2026-08-12 | [長期 Run: batch / replay 探索](2026-07-27_longrun-batch-replay.md) | 長期継続学習における batch size、replay ratio、PER、実時間効率 | 一旦クローズ。機構分離・再現性は次campaignへ保留 |
| 2026-08-09〜 | [IQN 導入・QR 比較](2026-08-09_iqn.md) | IQN成立性、QR51比較、tau sampling 5方式、K/N/M本数探索 | active |
| 2026-10-01〜2026-10-15 | [Atari からのフィードバック](2026-10-01_atari-feedback.md) | Munchausen、IQN と UQE の相互作用、target 更新、NoDropTimeout 罰、NoisyNet と NEET、RR・NN 構造・γ、100M からの分岐 | completed |
| 2026-10-15〜 | [勾配ノイズと batch / replay ratio の機序](2026-10-15_gradient-noise-batch.md) | PRD092 の勾配ノイズ、B・RR・学習率の分岐、B の上限、評価専用 Run による判定、NoisyNet の揺らぎへの感度、Run ごとのぶれと学習中の評価 ×8、学習の中身の多角比較 | active |

## 現時点の判断

最終更新: 2026-10-19

| 設定・探索 | 現在の扱い | 根拠の概要 |
|---|---|---|
| `batch_size=512`, `replay_ratio=2.0` | 評価での優位は確認できない（2026-10-19 判断更新。以前は成熟checkpoint継続時の最終成績優先ライン） | 170→230Mの分岐（42D）で評価専用Runの平均点は対照と±0.5%。学習側の+0.5〜1.2%はNoisyNetの揺らぎへの感度の差だった。7月のcy07/cy08も学習中の評価は300Mの間ほぼ横ばいで、B256の枝との差は1〜3%（[勾配ノイズの機序](2026-10-15_gradient-noise-batch.md) 探索ブロック05・06・09）。scratch開始の80M RunではQバブルと条件付きNEETが発生した |
| `batch_size=128`, `replay_ratio=1.25` | historical scratch reference | cy01の立ち上がりとQ / lossは安定。IQN対応後binaryのQR対照も35.59Mまで旧cy01と同水準で正常closeした |
| `batch_size=256`, `replay_ratio=1.0` | QR / IQNハイパラ探索の高速基準として採用。長期最終成績ラインは未確定 | scratch QR51が100Mで過去B128/RR1.25 QRと同等の終盤rewardを維持し、同条件IQNより約9%高速だった。RR1によるReplay coverage低下は残る |
| `batch_size=512`, `replay_ratio=1.0` | 採用しない | 50→150Mの分岐（41b）で分岐直後に学習側+2.4%上がり、30〜40Mで消えた。学習中の評価は対照と同点（[勾配ノイズの機序](2026-10-15_gradient-noise-batch.md) 探索ブロック02）。100→170Mの分岐（31d・34b）の学習側+0.6〜2.0%は評価で確かめていない（[Atari からのフィードバック](2026-10-01_atari-feedback.md) 探索ブロック12・14） |
| `batch_size=1024`, `replay_ratio=2.0` | 採用しない | 170→230Mの分岐（42H）で学習中の評価は分岐後20Mまでに対照より約3%高いが、40–60Mで対照が追いつき、最終の評価専用Runも+0.3%。実時間は対照の2.2倍（同 探索ブロック04・12） |
| `batch_size=1024`, `replay_ratio=1.0` | 保留（継続候補）。32GB GPUで使えるBの上限 | 計算量・実時間は対照とほぼ同じ。170→230Mの分岐（42J）で、学習中の評価は対照が40–60Mで着く水準に0–20Mで着き、最終の評価専用Run（Eval1・Eval3）も+0.9%（約2.4σ）。その後は横ばいで、勾配ノイズの比も対照並みに戻った。B2048はGPUメモリが溢れて1/4の速度、B4096は最初の更新で止まる（同 探索ブロック03・08・12・13） |
| `batch_size=256`, `replay_ratio=2.0` | 採用しない | 取りこぼしは29%→5.6%に減ったが、ダブルスイカ率は増えず、評価専用Runは−0.8%。重みの増え方は対照の約3倍（同 探索ブロック10・13） |
| 学習率 7.1e-5（B256/RR1、checkpoint継続） | 採用しない | 50→150Mの分岐（41c）で分岐直後に学習側+2.3%上がり、30〜40Mで消えた。学習中の評価は対照と同点（同 探索ブロック02） |
| Munchausen（τ0.03、α0.9、clip −1、Double DQN OFF） | 採用 | IQNの100M checkpointの評価専用Runで、Eval1 1,524→1,796（+18%）、Greedy 1,572→1,831（+16%）。2 seedの差は1〜2%。QR51ではGreedyのNEETは消えたが、UQEでの利得はseedで+4% / −7%に割れた（[Atari からのフィードバック](2026-10-01_atari-feedback.md) 探索ブロック01・04） |
| IQN と QR51 | IQN32 stratifiedを継続 | 「IQNはQR51に負ける」は評価方策UQEとの相互作用で、Greedyで評価するとIQN 1,572対QR51 1,552。Munchausen入りの2 seed平均はIQN 1,784対QR51 1,671（同 探索ブロック02・04） |
| ReLU（Aストリーム、τ埋め込み、main_feature末尾） | 持ち込まない | Aストリームは50Mで差なし、全部ReLUはmain_feature末尾の90%がdormant、τ ReLUはQバブルで−30%（50M、Munchausen前の基準。同 探索ブロック03） |
| target の hard 更新（C=200） | 採らない。soft 0.005を維持 | IQN+Mでsoftの2 seedより3方策とも低く、QR+MではNoDropTimeoutが39〜58%出た（同 探索ブロック05） |
| NoDropTimeout 罰 −20 | 採用 | 評価専用Runの時間切れの終局が21〜62%から0〜5%へ。待ちは締切直前に落とす長考型に変わっただけで、成績は−10と同水準（同 探索ブロック06） |
| ε ラダーを NoisyNet に置き換える（`uqe_eps_start=0`、τ ラダーは残す） | 採用（基準） | NEETの入口はε ラダーの上端のlane。εを0にするだけだと評価専用Runで−7〜−11%、NoisyNetで置き換えると同水準に戻りNEETは0。spatial explorationを切ると−5〜−10%（同 探索ブロック08）。学習側の平均点にはNoisyNetの揺らぎへの感度が混ざる（[勾配ノイズの機序](2026-10-15_gradient-noise-batch.md) 探索ブロック06） |
| n-step 3 | 採らない | ε ラダー基準の上で長考型NEETに入り、評価専用Runで−10〜−12%（[Atari からのフィードバック](2026-10-01_atari-feedback.md) 探索ブロック07） |
| replay ratio 4 | 実時間を優先してRR1を維持 | NoisyNet基準で100Mの学習中の評価が+4.1%（1 Run、Runごとのぶれの内側）。同じ実時間ではRR1の300M継続が約10%上。ε ラダーの上のRR4+SNは−11〜−16%（同 探索ブロック07・09・11） |
| 共有トランク `Linear_2048` | 維持 | RR1で512と4,096はどちらも−8%、列の位置を残す縮約（map16）は−21%、Transformer 8層は−9%（同 探索ブロック10） |
| γ0.999 | 0.997を維持 | 100Mでnzrepと同点（−0.6%）（同 探索ブロック13） |
| PER 初期優先度（`per_initial_priority_mode=max`、`per_initial_priority=0.03`） | 採用しない | 一度もsampleされずに捨てられる割合は設計どおり23% / 35%に動いたが、100→120Mの学習側の平均点は±0.8%以内（同 探索ブロック12） |
| 評価方策のUQE τ | 0.85のまま | 0.5と0.85の差は4 checkpoint中1つで約+1%（約2σ）、他は差なし。1組目の+1.8%は別の局の組で再現しなかった（同 探索ブロック07） |
| 腕の判定方法 | target ネットの評価（Eval1・Eval3）で、最終の評価専用Run（各ランナー数千局、DS率は複数の評価seed）と学習中の評価の経過の両方を見る | NoisyNetを使う構成では、学習側の平均点に揺らぎへの感度の差が混ざる。online ネットのEval2は同じ設定の複製でも2.2%ずれた（target の2本は0.0%）。最終checkpointだけでは到達の速さの差を見落とす。学習中の評価は1セッション1局なら平均点の2%以上、Eval1・Eval3を8局にすると20M窓で1%程度の差が読める（実時間+9%）（同 探索ブロック05〜07・11・12） |
| `alpha=5e-5` / `7.5e-5` | invalidated | checkpoint load により AdamW の param group options が復元され、設定ファイル上の alpha が実効学習率になっていない可能性が高い |
| `per_alpha=0.1` | 採用見送り | 単一分岐では明確な改善がなく、NEET 増加の懸念もあった。確度は低い |
| IQN 32/32 random | superseded（基準はIQN32 stratified＋Munchausenへ移った。下の「IQN と QR51」の行） | 同一B256/RR1のQR51に対し、90–100MのEval target rewardが約8%、Policy rewardが約13%低い。Q/TD/lossは安定し正常close |
| IQN tau sampling | `stratified`採用、`antithetic`は代替候補 | 両方式とも2-seedでrandom改善を再現し、seed 2の50Mでも同率。全stratum被覆と2-seed Targetからstratifiedを標準化。systematic / fixedは棄却 |
| IQN tau本数 | target `M=32`、Policy `K=32`を暫定維持。次はcurrent N | M探索では変更利得なし。K8は2-seedで20M首位を再現し50Mまで安定したが、K32比の速度利得がなくPolicy marginも低い。K8/K32長期matched A/Bは保留 |
| batch sizeの段階的引き上げ | 採用しない（2026-10-19 判断更新） | 停滞区間（170→230M）でBを上げた3腕は、評価で同じ水準に早く着いたが、60Mでは対照が追いつき、最終の評価専用Runの差は+0.2〜0.9%だった。学習側の水準の段差の大部分はNoisyNetの揺らぎへの感度の差（[勾配ノイズの機序](2026-10-15_gradient-noise-batch.md)） |

運用上は、B128/RR1.25をscratch安定参照、B256/RR1を基準として使う。B512/RR2を成熟checkpointの最終成績優先ラインとする扱いは、評価専用Runで優位が確認できなかったため2026-10-19に取り下げた。
詳細な条件、当時の解釈、判断の更新履歴、保留残件はcampaign文書を参照してください。

## DropMerge 共通の学習・挙動知見

以下は複数のcampaignを横断して使う運用上の用語と判断基準である。
Run固有の数値は各campaign文書、終端契約は[PRD039](../../../memo/done/039_dropmerge_nolegal_adjudication_10prd.md)、prev-action観測は[PRD040](../../../memo/done/040_dropmerge_prev_action_obs_10prd.md)を正本とする。

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
