# PRD101：探索・学習パラメータの ProfiledValue 化 実装メモ

## 概要
2026-10-25 レビュー後の PRD101 と承認済み計画を正本とし、段階1（探索側）、段階2（学習側）の順に実装・受入検証する。PER β は既存の適用時機を維持する。qbert の成績比較は後続実験であり完了条件に含めない。

## 主な変更
### 段階1
- ProfiledValue に exponential と time_constant を追加する。root は漸近し、phase は指定長で終わる。
- root の定数は bare で読む。root .value の読み取りを消し、旧記法専用の検出・WARN・fail-fast は足さない。phase 内 .value は維持する。
- 設定読み取りと記録を分け、dump は実効 type の使用欄だけを記録する。非 constant 下の bare は保持するが評価・記録しない。
- DefaultDQN・Rainbow・skip の ε・UQE ε・τ・spatial を新キーへ移行し、Greedy と target_policy のコピー・上書きも対応する。
- SpatialLadderSchedule は定数の両端では tensor を再利用し、変動時だけ更新する。lane 割当てと1 lane の既存挙動を保つ。
- log は max(端, 1e-4) の列を指定両端へ線形伸縮する。clamp 後の両端が等しければ線形補間、両端が下限以上なら従来の生成経路を使う。
- D11 の指標を追加し、Thompson の uqe_tau は常に NaN。ε・τ・β の linear 時間評価は従来 float 演算順とし、double・整数の既存挙動は維持する。

### 段階2
- 7値を ProfiledValue 化。β 以外の6値は UpdateFromBatch の先頭で評価する。β は step 0 から開始し各更新末尾で次回抽選用に評価し、prefetch の要求順序も保つ。
- 全 optimizer group の学習率・weight_decay を更新し、σ 専用 decay は維持する。soft update には現在値を渡す。
- Actor の評価済み γ・Munchausen 値を行動結果に保持し To・WithAction でも保持する。step診断は既存 GetScalar 経路で受け取る。
- Replay Push・Sample・診断用抽選へ γ を明示的に渡し、builder・extractor に永続保持しない。MuZero は定数を渡す。
- 定数 γ は保存済み収益、schedule γ は step 報酬を既存 builder と同じ逆順の和で再計算し bootstrap と一致させる。
- prefetch は抽選済み報酬列と長さを保持し、消費側 Sample の γ で確定する。上書き後の Replay は読み直さない。TempoRL窓と診断用抽選も対応する。
- checkpoint 比較・上書き、scaler、ホライズン表示は step 0 の値。値域・有限性・spectral norm は D21。

### 移行
- 変更前の実行体・DLL・設定・4環境出力と、段階1完了時の段階2用基準を保存する。
- 現用設定・bat・テスト・PRD072 golden/比較器を移行し、sync_interval.value と設定・ImageClsテストの root .value も bare にする。
- DropMerge 本体 last=0 は中間laneも変わる。等価性検査だけ last=1e-4。
- 日本語設計100・150・200、ガイド020、CHANGELOG、必要なwikiを追随させる。既存の用語を再利用する。en・過去記録はPRDの除外規則を適用する。
- 残存検査から CHANGELOG 対応表と phase 内 .value を除く。
- 稼働用batと実行体の入替は D25 の Run/連鎖キュー終了後に同時に行う。無関係の変更を保持し、コミット・pushしない。

## テスト
Public interface / surface: Config読み取りとdump、方策 UpdateSchedule/GetScalar、Actor/WithAction/診断、Learner UpdateFromBatch、Replay Push/Sample/診断抽選、checkpoint、runner-facing設定とmetrics。
優先 behavior / TDD順序:
1. 設定読込→方策生成→UpdateSchedule→GetScalarで指数から0への切替を1テストで RED→GREEN。
2. bare・継承・dump・同期周期、指数/phase境界、不正値、ラダー0/微小端/下限連続性/1lane/旧値一致と方策・指標を1挙動ずつ RED→GREEN。
3. optimizer全group、β初回/次回時機、soft update、checkpoint、Actor hintとstep診断。
4. γ境界前Push/後Sample、done/truncated/wrap、TempoRL、CPU/CUDA prefetchと診断抽選。定数経路と値を維持。
GREEN後だけ整理する。計画外の公開挙動が必要なら計画へ戻る。

## 検証
MSVC VsDevCmd.bat を call する cmd /s /c 経由で Debug 全体をビルドする。関連テストとcore全体テストを行う。
段階ごとに7環境の全キー・文字列値、check_default_leaves.py errors=0、4環境のchecksum一致（PRD §6.1-1だけ除外）を確認する。
短い実Runでε停止と学習率段階切替を確認する。資材は core/anet-core/testdata/prd101/、結果は .scratch/prd101/。実測から期待値を自動承認しない。
自己点検は実装品質確認観点の全項目。独立 an-review に計画・実装を渡し、差し戻し対応を終える。未実施・未決を本書へ記録する。

## 前提
- PRD blob: dd3db5b2aa11a926e69917f5320b64c802f8048a。
- 設定型は現用配置の include/anet/agent.hpp、実行部品は src/dqn_based_agent.* に置く。
- 頻繁な経路に新しい同期readbackを加えない。ラダー統計はCPU端点から計算して保持する。schedule γ は必要な報酬列だけバッチへ追加し、定数構成では追加のコピー・再計算をしない。
- 通常sandboxのMXC起動失敗のため、必要なコマンドは承認付き実行を使う。
- 開始時は AnetRLRunner のローカル稼働なし。稼働用入替の確認は別途 D25 に従う。

## 経過と結果
- 2026-10-25: 計画保存。コード・設定の既存変更はなし。wiki等の既存変更は保持。