============================================================

# Agent 実装規約：変数・オブジェクトの所有者決定ガイドライン

（Ownership Guide for Adding New Variables / Objects）

本ドキュメントは、 Agent 系実装において、  
**新しい変数・オブジェクトを追加するとき「どこに配置すべきか」を迷わず判断できるための統一基準**を定義する。

---

# ■ 1. “状態 (State)” と “資源 (Resource)” の区別

### ● State（状態）

- モジュール内部のロジックを実現するために必要な「変化する値」
- そのモジュール自身が更新する
- 他のモジュールが同じ値を直接更新してはならない

例：

- epsilon（ActionPolicy 内部状態）
- decay 統計
- EMA/Q統計（StabilityController）
- warmup / interval counter（ReplayHandler）
- 学習 step counter（LearnerCore）

---

### ● Resource（資源）

- 複数モジュール間で利用される共有可能な「外部リソース」
- “箱” としての性質を持ち、長寿命で参照され続ける
- 共有Resourceの所有者は Agent（Hub）
- 共有Resourceから派生し、単一モジュールだけが利用・更新するprivate Resourceは、そのモジュールが所有する

例：

- Agent所有のsource policy_net / target_net
- optimizer
- ReplayBuffer
- RNG
- Config（読み取り専用）
- Actorだけがforwardと同期に使うprivate network snapshot

---

# ■ 2. 所有者決定フローチャート

## **Step 1：その値は State か Resource か？**

- 内部ロジックに従属し、頻繁に変化 → **State**
- 外部リソースとして利用され、複数から参照される → **Resource**

---

## **Step 2：更新(Change) する主体はどこか？（State 決定根拠）**

State の場合、  **“更新する責務を持つモジュールが唯一の所有者”** となる。

例：

- epsilon → ActionPolicy
- Q-EMA → StabilityController
- warmup/interval → ReplayHandler
- update_step / target_sync → LearnerCore

Resource の場合はここでは判断しない。

---

## **Step 3：読み(Read)たい主体は複数か？（Resource判定）**

- Yes → Resource（共有リソース）→ Agent が所有
- No → Stateかprivate ResourceかをStep 4で判定し、単一モジュール内に閉じ込める

---

## **Step 4：“箱(Container)” か “中身(Value)” か？**

- 複数モジュールが使う箱 → 共有Resource（NN, Buffer, Optimizer）→ Agent に置く
- 単一モジュールだけが使う派生した箱 → private Resource → 利用・更新するモジュールに置く
- 内部変量 → State（epsilon, EMA など）→ モジュールに置く

例：Learnerが更新しActorがsourceとして参照するonline networkはAgent所有の共有Resourceとする。一方、そのonline networkから複製され、特定Actorだけがforwardと同期に使うnetwork snapshotはActor所有のprivate Resourceとする。

---

## **Step 5：依存方向が正しいか？（最重要）**

- 循環依存（A→B→A）が発生しない配置を選ぶ
- 行動選択（Policy）が Learner に依存する構図は絶対禁止
- Resource は Agent が所有することで依存方向を一方向に統一できる

---

## **Step 6：寿命管理は“判断基準にしない”**

※ DQNAgentV2 の現行構成では全モジュールが Agent と同寿命のため、  
寿命による判断は不要。  
（将来の分散構成時にのみ利用する）

---

# ■ 3. 最終決定原則（暗記用まとめ）

### ◎ State（内部状態）

- 更新者が唯一の所有者
- モジュールに閉じ込める
- ActionPolicy / Stability / LearnerCore / ReplayHandler に分散配置

### ◎ Resource（共有資源）

- 複数利用されるもの
- 所有者は Agent
- NN / Optimizer / ReplayBuffer / RNG / Config

### ◎ Private Resource（モジュール専用資源）

- 共有Resourceから派生し、単一モジュールだけが利用・更新する箱
- 利用・更新するモジュールに閉じ込める
- 例：Actor-private network snapshot

### ◎ 判断基準の優先順位

1. **責務（Responsibility）**
2. **依存方向（Dependency Direction）**
3. **State か Resource か（Container or Value）**
4. （現段階では寿命を考慮しない）

---

# ■ 4. よくある誤り

- “複数で読むから State を Agent に置く” → ✗  
  → 更新者が複数でないなら State → モジュールが所有

- “Learner が NN を更新するから NN を Learner が持つ” → ✗  
  → NN は Resource、複数利用、依存方向を壊す → Agent が所有

- “ReplayBuffer は ReplayHandler が使うからそこに置く” → ✗  
  → Buffer は Resource、Runner と Learner 両方が使用 → Agent が所有

- “NN はすべて無条件に Agent が所有する” → ✗
  → source/shared network は Agent が所有する。特定Actorだけが利用・同期するprivate snapshotはActorが所有する

---

# ■ 5. 運用ルール

- 開発中に変数を追加する際は、このガイドラインの Step1〜5 に必ず従う
- 迷った場合は必ず “責務” と “依存方向” の観点から再判定する
- 例外を作らず、一貫性を最優先する

--
