---
title: 実ロボットでサンプル効率・オフライン学習を行う候補サーベイ
page_kind: サマリ
material: "[[reports/survey/real_robot_offline_rl_candidates_2026-08-14.md|real_robot_offline_rl_candidates_2026-08-14]]"
material_kind: サーベイ
concepts:
  - "[[docs/wiki/concepts/subjects/robot_tasks.md|実ロボット課題の選定]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# 実ロボットでサンプル効率・オフライン学習を行う候補サーベイ

## 元の材料

[実ロボットでサンプル効率・オフライン学習を行う候補サーベイ](../../../../../reports/survey/real_robot_offline_rl_candidates_2026-08-14.md) blob:4b53970627027e42f6cb77ecea32a58c46bd8bd1

## 要点

- 繰り返しデータを集める装置は、自動で初期状態に戻り、成功・失敗が人手なしで分かるかを含めて選ぶ。
- アルゴリズムを主役にするなら、総合第一候補は 低価格 leader-follower アーム + 可逆治具 + 成功センサ である。SO-101 は follower/leader とも 6 servo の構成、BOM、3D プリント、校正手順が公開され、LeRobot の収録・学習・実機評価経路へ接続できる。[Hugging Face SO-101 (2026/08), Source the parts / assembly]
- 純粋な sample-efficiency 比較: Furuta pendulum、ball-and-beam、2軸 ball maze のような自動 reset リグ。
- 接触を含む Offline RL: D'Claw 型の dial/valve、plug-in/plug-out、drawer open/close のような正逆タスクセル。
- キット本体、可逆治具、電源、予備部品を含む運用費は異なり、部品価格だけでは長時間の収集条件を比較できない。

## 内容

### 1. 結論

2026-08-14 時点で Seeed Studio の motor-only kit は USD 260、3D プリント部品は別売 USD 35 である。これは完成した leader-follower 2台、カメラ、治具、予備 servo の総額ではない。[Seeed Studio (2026/08), product page]

### 2. ロボットより先に決める選定軸

大規模な実ロボデータの多くは、成功 demonstration を中心とした imitation/VLA 用データであり、報酬付きの Offline RL dataset と同義ではない。DROID は 76,000 trajectory、350時間、564 scene、86 task を公開しているが、主成分は teleoperation demonstration である。[DROID (2024/03), project summary]

### 3. 候補マップ

評点は本調査独自で、`◎ > ○ > △ > ×`。予算帯も部品価格そのものではなく、カメラ、治具、電源、予備部品、安全設備を含む初期計画用の概算である。

### 4. 定番候補の詳細

Phase 1: state-only、低次元 action、完全自動 reset。

SO-101 は最速で dataset collection に入れる低価格候補である。上位に進むなら、OpenMANIPULATOR-X は 4DOF + 1DOF gripper、500g payload、ROS/DYNAMIXEL SDK を公式仕様に持つ。[ROBOTIS OpenMANIPULATOR-X (2026/08), Hardware Specification]

### 5. 二足・四足の現実性

四足は支持多角形と self-righting の余地があり、自由歩行二足より先に locomotion 研究へ入れる。RealAnt は材料費約 EUR 350、1時間未満の組立、10分未満の実機経験からの歩行を報告しているが、これは2020年当時の論文条件であり、現在の調達価格や再現時間の保証ではない。[RealAnt (2020/11), Abstract]

### 6. 面白系候補の詳細

腱駆動 tentacle: ring/peg 到達。compliance で破損しにくく、hysteresis と hidden state が自然な研究対象になる。

### 7. 推奨する三つの具体案

狙い: Offline RL と human correction を最短で実機化する。

### 8. データ契約と評価

SERL は PCB insertion、cable routing、object relocation を対象に、方策あたり平均25–50分の学習を報告している。これは robot、controller、reward、reset を含む system engineering の成果であり、algorithm 名だけを移植すれば再現するという意味ではない。[SERL (2024/01), project summary]

### 9. 代表的な研究資産

引用数は OpenAlex の特定 record を 2026-08-14 に取得した参考値である。preprint と会議版の統合差が大きく、研究影響度の厳密な比較には使わない。[OpenAlex (2026/08), Works API records]

## 主要な概念

- [実ロボット課題の選定](../../../concepts/subjects/robot_tasks.md)
- [評価と比較の条件](../../../concepts/measurement/evaluation.md)
- [322 Env個別実装](../../../concepts/framework/322_env_individual.md)
