# 実ロボットでサンプル効率・オフライン学習を行う候補サーベイ

調査日: 2026-08-14  
範囲: 個人が電子工作・3Dプリント・簡単な機械加工を行える前提で、定番から変化球までの実機候補を比較する。価格は特記しない限り 2026-08-14 に確認した税・送料・関税別の表示価格、評価と予算帯は本調査独自の工学的見積もりである。

## 目次

- [1. 結論](#1-結論)
  - [1.1 第一候補](#11-第一候補)
  - [1.2 二足歩行への回答](#12-二足歩行への回答)
- [2. ロボットより先に決める選定軸](#2-ロボットより先に決める選定軸)
  - [2.1 Offline RL と模倣学習の違い](#21-offline-rl-と模倣学習の違い)
  - [2.2 実機で本当に高価なのは reset](#22-実機で本当に高価なのは-reset)
- [3. 候補マップ](#3-候補マップ)
  - [3.1 定番系](#31-定番系)
  - [3.2 脚ロボ系](#32-脚ロボ系)
  - [3.3 面白系](#33-面白系)
- [4. 定番候補の詳細](#4-定番候補の詳細)
  - [4.1 自動 reset 制御リグ](#41-自動-reset-制御リグ)
  - [4.2 卓上アームと双腕](#42-卓上アームと双腕)
  - [4.3 固定ハンドと挿抜セル](#43-固定ハンドと挿抜セル)
  - [4.4 車輪ロボット](#44-車輪ロボット)
- [5. 二足・四足の現実性](#5-二足四足の現実性)
  - [5.1 四足](#51-四足)
  - [5.2 平面拘束二足](#52-平面拘束二足)
  - [5.3 自由歩行 humanoid](#53-自由歩行-humanoid)
- [6. 面白系候補の詳細](#6-面白系候補の詳細)
  - [6.1 Soft robotics と触覚](#61-soft-robotics-と触覚)
  - [6.2 卓上ゲームと自己回収](#62-卓上ゲームと自己回収)
  - [6.3 特殊移動体](#63-特殊移動体)
- [7. 推奨する三つの具体案](#7-推奨する三つの具体案)
  - [7.1 CycleArm](#71-cyclearm)
  - [7.2 BoomBiped](#72-boombiped)
  - [7.3 RotaryCell](#73-rotarycell)
- [8. データ契約と評価](#8-データ契約と評価)
  - [8.1 保存するもの](#81-保存するもの)
  - [8.2 比較指標](#82-比較指標)
- [9. 代表的な研究資産](#9-代表的な研究資産)
- [10. 懸念と調査限界](#10-懸念と調査限界)
- [11. 参考資料](#11-参考資料)

## 1. 結論

### 1.1 第一候補

アルゴリズムを主役にするなら、総合第一候補は **低価格 leader-follower アーム + 可逆治具 + 成功センサ** である。SO-101 は follower/leader とも 6 servo の構成、BOM、3D プリント、校正手順が公開され、LeRobot の収録・学習・実機評価経路へ接続できる。[Hugging Face SO-101 (2026/08), Source the parts / assembly]

```text
The follower arm uses 6x STS3215 motors with 1/345 gearing.

follower arm は 1/345 ギア比の STS3215 motor を6個使用する。
```
[Hugging Face SO-101 (2026/08), assembly]

2026-08-14 時点で Seeed Studio の motor-only kit は USD 260、3D プリント部品は別売 USD 35 である。これは完成した leader-follower 2台、カメラ、治具、予備 servo の総額ではない。[Seeed Studio (2026/08), product page]

次点は、研究目的によって二つに分かれる。

- 純粋な sample-efficiency 比較: Furuta pendulum、ball-and-beam、2軸 ball maze のような自動 reset リグ。
- 接触を含む Offline RL: D'Claw 型の dial/valve、plug-in/plug-out、drawer open/close のような正逆タスクセル。

### 1.2 二足歩行への回答

二足は十分に候補だが、**最初から自由歩行 humanoid を選ぶのではなく、boom で sagittal plane に拘束した planar biped を選ぶ**のがよい。STRIDE は 5-link planar biped を boom で平面拘束し、USD 2,000 未満、加工不要の構成を報告している。[STRIDE (2024/07), Sections I–II]

```text
The platform is low-cost (below $2000) and easy to customize without the need of machining.

この platform は 2,000 USD 未満で、機械加工なしに変更しやすい。
```
[STRIDE (2024/07), Section I]

この形なら、二足特有の接地衝撃、underactuation、摩擦、歩容適応は残しながら、横転、走行範囲逸脱、大部分の人手起こしを機構側で除ける。後半は本調査の設計評価であり、STRIDE 論文の主張そのものではない。

## 2. ロボットより先に決める選定軸

### 2.1 Offline RL と模倣学習の違い

大規模な実ロボデータの多くは、成功 demonstration を中心とした imitation/VLA 用データであり、報酬付きの Offline RL dataset と同義ではない。DROID は 76,000 trajectory、350時間、564 scene、86 task を公開しているが、主成分は teleoperation demonstration である。[DROID (2024/03), project summary]

```text
76k demonstration trajectories or 350h of interaction data, collected across 564 scenes and 86 tasks

564 scene・86 task から収集した 76,000 demonstration trajectory、または350時間の interaction data
```
[DROID (2024/03), project summary]

Open X-Embodiment も 22 embodiment にわたる 100万件超の実機 trajectory を統合しており、事前学習・表現学習の資産として強い。[Open X-Embodiment (2023/10), Dataset Overview]

```text
It contains 1M+ real robot trajectories spanning 22 robot embodiments.

22種類の robot embodiment にまたがる100万件超の実機 trajectory を含む。
```
[Open X-Embodiment (2023/10), Dataset Overview]

一方、Real Robot Challenge 2022 の TriFinger task は、実機6台由来の dataset、reward、timeout、D4RL 互換 interface、追加データ禁止の評価 protocol を明示している。[Real Robot Challenge (2022), Tasks / Datasets]

```text
learn a policy ... using only the provided dataset

提供された dataset だけを使って方策を学習する。
```
[Real Robot Challenge (2022), Learn a policy from the provided dataset]

したがって自作 dataset で Offline RL を名乗るなら、成功デモだけでなく、失敗、途中停止、回復、弱い controller、safe random residual を含め、`reward` と `next_observation` を保存できる機構が必要になる。

### 2.2 実機で本当に高価なのは reset

sample efficiency は transition 数だけでは決まらない。人手 reset、成功判定、故障確認、再校正の wall-clock 時間が実験速度を支配する。実機自律収集のスケール化では、環境設計と計装が大きな負担になることも報告されている。[Mirchandani et al. (2025), Abstract]

```text
significant effort required for environment design and instrumentation

環境設計と計装に大きな労力が必要である。
```
[Mirchandani et al. (2025), Abstract]

候補選定では次の順で評価する。

1. 自動または相互 reset できるか。
2. 成功を camera 推定ではなく encoder、limit switch、load cell、AprilTag などで再計算できるか。
3. 失敗を含む mixed-quality data を安全に集められるか。
4. 1時間あたりの有効 transition と episode 数が多いか。
5. 消耗部品をその日のうちに交換できるか。

## 3. 候補マップ

評点は本調査独自で、`◎ > ○ > △ > ×`。予算帯も部品価格そのものではなく、カメラ、治具、電源、予備部品、安全設備を含む初期計画用の概算である。

### 3.1 定番系

| 候補 | Offline | Online少数試行 | Reset | 工作負荷 | 概算帯 | 向く研究 |
|---|---:|---:|---:|---:|---:|---|
| Furuta / cart-pole / reaction wheel | ○ | ◎ | ◎ | ○ | S | world model、system ID、offline-to-online |
| 2軸 ball maze / ball-and-beam | ○ | ◎ | ◎ | ○ | S | visual control、遅延、摩擦、POMDP |
| SO-101 + 可逆治具 | ◎ | ○ | ○ | ○ | M | BC、IQL/CQL系、human correction |
| OpenMANIPULATOR-X | ◎ | ○ | ○ | 低 | L | ROS2、精密 pick/place、接触作業 |
| TriFinger / D'Claw 型固定 hand | ◎ | ◎ | ◎ | 高 | M–L | dexterity、dataset composition、tactile |
| TurtleBot / 小型 RC car | ○ | ◎ | ◎ | 低–○ | M | navigation、goal relabel、safe adaptation |
| Dual SO-101 / ALOHA 型 | ◎ | ○ | △ | 高 | L–XL | bimanual、long horizon、teleoperation |
| Mobile manipulator | ◎ | ○ | △ | 高 | L–XL | navigation + manipulation、階層方策 |

### 3.2 脚ロボ系

| 候補 | Offline | Online少数試行 | Reset | 工作負荷 | 概算帯 | 評価 |
|---|---:|---:|---:|---:|---:|---|
| RealAnt 型小型四足 | ○ | ○ | ○ | ○ | M | 安価な direct real-world RL の有力候補 |
| Unitree Go2 EDU | ○ | ○ | ○ | 低 | XL | 機体より API 契約、安全設備が焦点 |
| STRIDE 型 boom biped | ○ | ◎ | ○ | 高 | L | 二足研究の第一候補 |
| LeRobot Humanoid | △ | ○ | △ | 非常に高い | XL | experimental な full-stack DIY biped |
| ToddlerBot | ○ | ○ | △ | 非常に高い | XL | whole-body teleop と sim-to-real |
| Berkeley Humanoid Lite | △ | ○ | × | 非常に高い | XL | hardware/locomotion 研究そのもの |
| Unitree R1/G1 EDU | ○ | ○ | × | 低 | XL+ | 完成機だが個人の初号機には過大 |

### 3.3 面白系

| 候補 | Reset 案 | 観測 → 行動 | 研究上の問い |
|---|---|---|---|
| 触覚 dial-lock | CW↔CCW | tactile image/current → finger delta | visionなしの状態推定、不確実性 |
| 腱駆動 soft tentacle | ring hopper | marker/current → tendon increment | hysteresis、latent dynamics |
| spool winding cell | 巻く↔ほどく | tension/camera → reel/traverse | rare jam、risk-sensitive RL |
| pinball / pachinko | ball trough | high-speed camera → flipper timing | sparse reward、sequence model |
| 卓上 goalie | return chute | ball track → carriage velocity | OOD速度、reaction margin |
| spherical rover | 壁内A↔B | IMU/pose → internal wheel torque | contact-rich POMDP |
| servo snake | 環状 course | joint/current → CPG residual | raw policy と低次元 action の比較 |
| wall crawler | dockへ帰還 | IMU/current/camera → wheel speed | slip prediction、risk constraint |
| soft fish | 円形水槽 | pose/IMU → tail frequency/bias | fluid hysteresis、morphology shift |
| Crazyflie | charging pad | pose/IMU → attitude residual | safe exploration、自動着陸 |
| vibrobot swarm | 閉鎖 arena | multi-pose → PWM | multi-agent credit、個体差 |

## 4. 定番候補の詳細

### 4.1 自動 reset 制御リグ

Furuta pendulum、cart-pole、reaction wheel、ball maze は、ロボットらしい sensing、遅延、摩擦、backlash、電流制限を残しながら、episode を高速に回せる。完成品を買うより、DYNAMIXEL/BLDC、絶対 encoder、home sensor、mechanical stop を組み合わせて自作すると、action clamp と故障モードまで所有できる。

特に面白いのは次の段階構成である。

- Phase 1: state-only、低次元 action、完全自動 reset。
- Phase 2: camera 観測を追加し、state policy を teacher にする。
- Phase 3: payload、摩擦材、電圧、遅延を変え、offline adaptation を測る。
- Phase 4: 収集 policy の質と多様性を変え、同じ transition budget で比較する。

### 4.2 卓上アームと双腕

SO-101 は最速で dataset collection に入れる低価格候補である。上位に進むなら、OpenMANIPULATOR-X は 4DOF + 1DOF gripper、500g payload、ROS/DYNAMIXEL SDK を公式仕様に持つ。[ROBOTIS OpenMANIPULATOR-X (2026/08), Hardware Specification]

人間サイズの双腕へ進む OpenArm は、7DOF arm、QDD backdrivable motor、nominal 4.1kg payload、complete bimanual system USD 6,500 を掲げるが、Linux/CAN、安全固定、E-stop を前提とする。[LeRobot OpenArm (2026/08), What's Unique / Safety Guide]

アーム課題は、物体を自由空間へ落とす pick-and-place より、次の可逆課題から始めるとよい。

- drawer open ↔ close
- plug insert ↔ remove
- dial left ↔ right
- object bin A ↔ bin B
- latch lock ↔ unlock

### 4.3 固定ハンドと挿抜セル

固定 hand は mobile arm より workspace が狭く、object pose を fixture/encoder で測りやすい。ROBEL の D'Claw は三指 manipulation、D'Kitty は四足 locomotion を対象とし、論文時点で累計14,000 hardware training hours と hardware-safety score を報告した。[ROBEL (2020/10), Abstract]

```text
robust enough to sustain on-hardware reinforcement learning from scratch

実機上でゼロから強化学習を行えるだけの頑健性を持つ。
```
[ROBEL (2020/10), Abstract]

最初の自作題材としては、自由物体の in-hand rotation より、軸に固定した valve、dial、key、connector の方が reset と ground-truth reward を設計しやすい。

### 4.4 車輪ロボット

車輪は転倒がなく、A↔B goal、dock、loop course によって連続収集しやすい。TurtleBot3 Burger は ROS ベースの小型 platform で、公式仕様は 138 × 178 × 192 mm、公式米国ストアの Raspberry Pi 4 4GB 構成は USD 783.50 である。[ROBOTIS TurtleBot3 (2026/08), Features / store]

アルゴリズム比較を主役にするなら、TurtleBot 完成品でなくても、1/18–1/24 RC chassis、overhead camera、AprilTag、soft bumper、radio kill switch でよい。parking、puck pushing、goal navigation は自動成功判定を作りやすい。

## 5. 二足・四足の現実性

### 5.1 四足

四足は支持多角形と self-righting の余地があり、自由歩行二足より先に locomotion 研究へ入れる。RealAnt は材料費約 EUR 350、1時間未満の組立、10分未満の実機経験からの歩行を報告しているが、これは2020年当時の論文条件であり、現在の調達価格や再現時間の保証ではない。[RealAnt (2020/11), Abstract]

```text
learn to walk from scratch from less than 10 minutes of experience

10分未満の経験からゼロから歩行を学習する。
```
[RealAnt (2020/11), Abstract]

Unitree Go2 は公式ページ上 USD 1,600 からだが、二次開発は構成差がある。購入前に low-level API、foot force、onboard compute、warranty、転倒時の責任範囲を型番単位で確認する必要がある。[Unitree Go2 (2026/08), model comparison]

### 5.2 平面拘束二足

sample efficiency を比較しつつ二足をやるなら、STRIDE 型が最も筋がよい。推奨する学習順は次の通り。

1. 既存 gait の joint-target residual。
2. 速度追従と step timing。
3. 摩擦材・斜面・payload への offline adaptation。
4. push recovery と failure prediction。
5. 最後に full action policy。

最初から torque policy で自由歩行を学ばせるより、baseline controller 周辺で安全な suboptimal data を集める方が、機械故障とアルゴリズム失敗を分離できる。

### 5.3 自由歩行 humanoid

2026年には、個人が検討できる open humanoid が増えている。

- LeRobot Humanoid: 2026年5月公開。現行 biped の部品費は約 USD 2,500。hardware、simulation、identification、training environment を含むが、公式自身が experimental と明記する。[Hugging Face LeRobot Humanoid (2026/05), TL;DR]
- ToddlerBot: 0.56m、3.4kg、30DOF、USD 6,000 未満。CoRL 2025 論文と open-source hardware/software、teleoperation を持つ。[ToddlerBot (2025), Abstract]
- Berkeley Humanoid Lite: 0.8m、16kg、22DOF級、3D-printed cycloidal gearbox、USD 5,000 未満。RL locomotion の zero-shot sim-to-real を報告する。[Berkeley Humanoid Lite (2025/04), Abstract]

LeRobot Humanoid は特に maker 向けだが、完成度の位置づけも明確である。

```text
The platform is experimental, but already connects the full loop.

platform は実験段階だが、すでに一連の loop を接続している。
```
[Hugging Face LeRobot Humanoid (2026/05), TL;DR]

市販 humanoid の安価な表示価格にも注意がいる。Unitree R1 は USD 4,900 からだが、公式表では secondary development は EDU のみ。G1 の USD 13,500 consumer 版も二次開発非対応で、EDU は要問い合わせである。[Unitree R1 (2026/08), Parameter table] [Unitree G1 Store (2026/08), purchase notice]

```text
This product does not support secondary development.

この製品は二次開発に対応しない。
```
[Unitree G1 Store (2026/08), purchase notice]

自由歩行 humanoid を選ぶのは、sample efficiency だけでなく、actuator、gearbox、fall recovery、配線、battery、sim-to-real そのものを研究したい場合である。初号機では、吊り具、soft mat、E-stop、current/temperature limit、fall classifier、予備リンクを同時に設計する。

## 6. 面白系候補の詳細

### 6.1 Soft robotics と触覚

- 腱駆動 tentacle: ring/peg 到達。compliance で破損しにくく、hysteresis と hidden state が自然な研究対象になる。
- 触覚 dial-lock: RGB を隠し、tactile image と motor current だけで目標角へ回す。
- pneumatic gripper carousel: soft object を連続供給し、成功率だけでなく圧力・損傷 proxy を最適化する。
- spool winding: tension と画像から reel speed/traverse を制御し、rare jam を扱う。

### 6.2 卓上ゲームと自己回収

- pinball/pachinko: trough が ball を自動回収し、switch event から報酬を得る。
- 2軸 ball maze: A↔B course または ball elevator で相互 reset する。
- tabletop goalie: programmable launcher と return chute で速度分布を制御する。
- air-hockey puck shepherd: opponent controller が次 episode の reset を兼ねる。
- cup-and-ball crane: ball が紐で拘束され、失敗しても失われない dynamic skill。

### 6.3 特殊移動体

- servo snake: action を各関節にするか CPG parameter residual にするかを比較できる。
- spherical rover: shell が内部機構を保護し、壁接触を含む POMDP になる。
- wall crawler: steel board と permanent magnet、catch tether で slip-aware control を試せる。
- soft fish: 円形水槽で reset-free に近づけるが、防水が主作業になりやすい。
- Crazyflie: cage と charging pad を用意すれば自動着陸まで閉じられるが、墜落回収は残る。

## 7. 推奨する三つの具体案

### 7.1 CycleArm

**狙い:** Offline RL と human correction を最短で実機化する。

- Hardware: SO-101 leader + follower、固定 camera 2台、joint log、E-stop。
- Fixture: drawer、rotary connector、A/B bin。limit switch、encoder、load cell のいずれかで成功を判定。
- Data: teleop success、意図的失敗、scripted controller、policy rollout、human intervention。
- Baseline: behavior cloning を必須とし、その後に IQL/CQL/TD3+BC 系、最後に短時間 online fine-tune。
- 成功条件: 同じ固定 dataset で seed 間差より手法差が大きく、5/15/30分追加収集の budget curve が取れる。

### 7.2 BoomBiped

**狙い:** 「二足をやった感」を残しつつ、実験を回せる形にする。

- Hardware: STRIDE 型 4 actuated joint + circular boom、pelvis IMU、joint encoder、foot contact、current/temperature。
- Data: hand-designed gait、parameter sweep、safe residual noise、路面・payload 変更。
- Task: step-in-place、速度追従、push recovery、rough-terrain adaptation。
- Baseline: system identification + MPC/既存 gait。学習器は residual から始める。
- 成功条件: 転倒介入なしで1時間収集でき、摩擦・payload holdout で改善が見える。

### 7.3 RotaryCell

**狙い:** 最も高いデータ回転率で contact-rich Offline RL を行う。

- Hardware: D'Claw 型 2–3 finger、軸固定 dial/valve、絶対 encoder、replaceable fingertip、tactile option。
- Task: CW↔CCW、lock↔unlock、insert↔remove。
- Data: controller quality、摩擦材、指先材、電圧、遅延を系統的に変える。
- Baseline: state-only → vision/tactile → sensor dropout の順で難しくする。
- 成功条件: 無人で数時間回り、失敗・回復・safety clamp を含む dataset が自然に蓄積する。

## 8. データ契約と評価

### 8.1 保存するもの

- raw observation と取得 timestamp
- 同期・前処理後 observation
- policy command と、clamp 後に actuator へ実際に出た action
- reward を後から再計算できる ground-truth sensor
- `terminal` と時間切れ `truncation` の区別
- current、temperature、voltage、通信欠損
- safety clamp、human intervention、E-stop
- reset transition と reset 所要時間
- robot/calibration/fixture/operator/controller の provenance

### 8.2 比較指標

- success rate と return
- 実機 transition 数
- 追加 online 分数
- operator reset 分数と intervention 回数
- wall-clock time to criterion
- 故障・再校正・停止時間
- energy、peak current、impact proxy
- holdout robot、payload、摩擦、照明での性能

SERL は PCB insertion、cable routing、object relocation を対象に、方策あたり平均25–50分の学習を報告している。これは robot、controller、reward、reset を含む system engineering の成果であり、algorithm 名だけを移植すれば再現するという意味ではない。[SERL (2024/01), project summary]

```text
it trains policies in just 25 to 50 minutes

方策をわずか25〜50分で学習する。
```
[SERL (2024/01), project summary]

## 9. 代表的な研究資産

引用数は OpenAlex の特定 record を 2026-08-14 に取得した参考値である。preprint と会議版の統合差が大きく、研究影響度の厳密な比較には使わない。[OpenAlex (2026/08), Works API records]

| 資産 | 著者・主所属 | 発表 | OpenAlex引用数参考 | この調査での役割 |
|---|---|---|---:|---|
| [SERL](https://serl-robot.github.io/) | Jianlan Luo et al.; UC Berkeley、UW、Stanford、Intrinsic | ICRA 2024 | [38](https://openalex.org/W4401416214) | sample-efficient real robot RL の system baseline |
| [TriFinger](https://arxiv.org/abs/2008.03596) | Manuel Wüthrich et al.; MPI-IS | arXiv 2020 | [24](https://openalex.org/W3047943996) | 固定 hand と real offline benchmark の hardware |
| [ToddlerBot](https://proceedings.mlr.press/v305/shi25a.html) | Haochen Shi et al.; Stanford | CoRL 2025 | [0](https://openalex.org/W4407124430) | 小型 open humanoid、teleop、digital twin |
| [Berkeley Humanoid Lite](https://lite.berkeley-humanoid.org/) | Yufeng Chi et al.; UC Berkeley | RSS 2025 | [1](https://openalex.org/W4415307010) | open 3D-printed humanoid と sim-to-real |
| [STRIDE](https://arxiv.org/abs/2407.02648) | Yuhao Huang et al.; UW–Madison | arXiv 2024 | [0](https://openalex.org/W4400375613) | planar boom biped |
| [RealAnt](https://arxiv.org/abs/2011.03085) | Rinu Boney et al.; Aalto University | arXiv 2020 | [0](https://openalex.org/W4320458131) | 低価格 direct real-world quadruped RL |

## 10. 懸念と調査限界

- 日本国内の在庫、PSE/技適、輸送、関税、保険、研究室外での安全要件は調査していない。
- 表示価格は完成した実験設備の価格ではない。PC/GPU、camera、3D print、治具、予備 actuator、E-stop、cage/gantry、工数を含めると大きく増える。
- Open-source は即再現可能を意味しない。LeRobot Humanoid は公式にも experimental、Berkeley Humanoid Lite は高電力 electronics の注意を明記する。
- Unitree の consumer 表示価格と developer 向け EDU 契約は別物である。型番ごとの SDK/low-level control 契約を発注前に書面確認する必要がある。
- OpenAlex 引用数は record 分裂の影響を受け、特に arXiv DOI record の 0 は「引用されていない」とは限らない。
- 本調査はロボット候補の選定であり、ANET の action space、continuous-control algorithm、real-time adapter の実装可否は別途コード監査が必要である。

## 11. 参考資料

[Hugging Face SO-101, 2026/08] Hugging Face. “SO-101.” LeRobot Documentation. https://huggingface.co/docs/lerobot/so101

[Seeed Studio, 2026/08] Seeed Studio. “SO-ARM101 Pro Kit (Motors Only).” https://www.seeedstudio.com/SO-ARM101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html

[STRIDE, 2024/07] Yuhao Huang, Yicheng Zeng, Xiaobin Xiong. “STRIDE: An Open-Source, Low-Cost, and Versatile Bipedal Robot Platform for Research and Education.” arXiv. https://arxiv.org/abs/2407.02648

[DROID, 2024/03] Alexander Khazatsky et al. “DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.” RSS 2024. https://droid-dataset.github.io/

[Open X-Embodiment, 2023/10] Open X-Embodiment Collaboration. “Open X-Embodiment: Robotic Learning Datasets and RT-X Models.” https://robotics-transformer-x.github.io/

[Real Robot Challenge, 2022] Max Planck Institute for Intelligent Systems. “Real Robot Challenge 2022 Tasks.” https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/tasks.html

[Mirchandani et al., 2025] Suvir Mirchandani et al. “So You Think You Can Scale Up Autonomous Robot Data Collection?” CoRL, PMLR 270. https://proceedings.mlr.press/v270/mirchandani25a.html

[ROBOTIS OpenMANIPULATOR-X, 2026/08] ROBOTIS. “OpenMANIPULATOR-X Specification.” https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/

[LeRobot OpenArm, 2026/08] Hugging Face / OpenArm. “OpenArm.” LeRobot Documentation. https://github.com/huggingface/lerobot/blob/main/docs/source/openarm.mdx

[ROBEL, 2020/10] Michael Ahn et al. “ROBEL: Robotics Benchmarks for Learning with Low-Cost Robots.” CoRL, PMLR 100. https://proceedings.mlr.press/v100/ahn20a.html

[ROBOTIS TurtleBot3, 2026/08] ROBOTIS. “TurtleBot3 Features / Store.” https://emanual.robotis.com/docs/en/platform/turtlebot3/features/ ; https://www.robotis.us/turtlebot-3/

[RealAnt, 2020/11] Rinu Boney et al. “RealAnt: An Open-Source Low-Cost Quadruped for Education and Research in Real-World Reinforcement Learning.” arXiv. https://arxiv.org/abs/2011.03085

[Unitree Go2, 2026/08] Unitree Robotics. “Unitree Go2.” https://www.unitree.com/go2/

[Hugging Face LeRobot Humanoid, 2026/05] Virgile Batto et al. “LeRobot Humanoid: An Open, Low-Cost, 3D-Printed Humanoid for Robot Learning.” Hugging Face. https://huggingface.co/blog/VirgileBatto/lerobot-humanoid

[ToddlerBot, 2025] Haochen Shi, Weizhuo Wang, Shuran Song, C. Karen Liu. “ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation.” CoRL, PMLR 305. https://proceedings.mlr.press/v305/shi25a.html

[Berkeley Humanoid Lite, 2025/04] Yufeng Chi et al. “Demonstrating Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot.” RSS 2025. https://lite.berkeley-humanoid.org/

[Unitree R1, 2026/08] Unitree Robotics. “Unitree R1.” https://www.unitree.com/mobile/R1/

[Unitree G1 Store, 2026/08] Unitree Robotics. “Unitree G1.” https://shop.unitree.com/collections/humanoid-robot/products/unitree-g1

[SERL, 2024/01] Jianlan Luo et al. “SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning.” ICRA 2024. https://serl-robot.github.io/

[OpenAlex, 2026/08] OpenAlex. “Works API records.” https://openalex.org/
