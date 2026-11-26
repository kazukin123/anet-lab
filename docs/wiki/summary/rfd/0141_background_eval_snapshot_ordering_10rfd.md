# バックグラウンド評価が複製する Network の更新時点

## 元の材料

[RFD0141　バックグラウンド評価が複製する Network の更新時点](../../../memo/0141_background_eval_snapshot_ordering_10rfd.md) blob:97e3719337a827ab5188bc71ee70e496675bd232

## 要点

- 起草時に特定できたのはスレッドの実行順で複製対象が揺れる可能性であり、対処方式と実装は後続の検討に残した。
- 学習イベントを受けて評価ジョブを投入した後、評価側が重みを複製する前に次の Learner 更新が始まる可能性がある。
- 同じ発火 step でも評価に使う重みの世代が変わると、決定的な実行条件でも再現性を損なう。
- 候補は学習イベント通知中に評価用 Actor の複製を終えてから次の更新を投入する方式である。
- 複製後の評価計算は並行して実行できるため、待ち合わせの範囲をコピーに絞れる。
- 実行順、複製の負荷、既存 Runner への適用範囲を確かめてから方式を確定する。

## 内容

### Context

`std::shared_mutex` の取得順、または「どの learner update 後の network を eval が参照したか」を固定しない。
明文化し、`LearnEvent[n]` に対応する eval が明示的に確定した単一 network version だけを評価する契約を定めることである。

### 起草時点の現行コードで確定している事実

LearnThread_0: `PipelineTrainRunner` が所有する 1 worker の `PinnedThreadPool`。
一方、foreground eval（`use_background=false`）は enqueue ではなく同期実行のため、この競合自体が発生しない。

### 問題

clone なし actor は learner と同じ network を共有し、各 eval batch の forward ごとに `shared_lock` を取得・解放する。

### 候補を採る場合の目標契約

`LearnEvent[n]` によって起動された snapshot eval は、明示的に確定した 単一 network versionだけを全 batch で使用する。
eval に使用する network version の確定は、次 learner update との順序が定義された同期境界で行う。

### 解決候補

`LearnEvent[n]` の通知処理中、次 learn を投入する前に eval actor の snapshot copy を完了する。
learner の `unique_lock` は eval 完了まで待つため、追加 network copy 無しで pass 内一貫性を得られる。

### 既存ドキュメントとの責務分担

本 RFD は lock の有無ではなく、eval trigger と snapshot の時間的順序へ限定する。
ADR 0006: ATen/cuDNN の演算決定論を扱う。本 RFD は backend flag が対象としない application thread scheduling を扱う。

### 候補を採る場合の検証方針

少ステップで決定的に episode 終端する test env を用意する（`EvalRunner::DoStep()` は env を回して episode 終端まで走るため）。
同一の seed / event 列で複数回実行し、eval trigger と version の対応列が一致することを検証する。

### 受け入れ基準

Serial / Pipeline 両経路のthread時系列、2種類の非決定境界、既存mutexが保証する範囲がテストとドキュメントで一致している。
live-weight evalを残す場合、そのmodeはsnapshot evalと設定・ログ・metric上で区別される。

### 後続

採用方式確定後に実装メモを作成し、barrier制御の回帰テストから実装する。

## 主要な概念

- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
