# eval スケジュールの初回発火の抑止 暫定 PRD

> 2026-11-03 番号を 999 から 918 へ振り直した（PRD114）。

> 起点: 2026-09-21 の Atari-5 地ならし。`run.@eval2ch` の ε=0 断面チャネル（`greedy_dist`、
> 400 エピソード）が learn_step=0 で発火し、5 ゲーム中 2 つで背景セッションが 1 時間級になり、
> 次回発火時に train を 692.6s / 885.5s 停止させた。この点は未学習方策の測定で、
> 実験記録が使っていない捨て点である。
> 関連: [076_eval_session_drain_on_exit](done/076_eval_session_drain_on_exit_10prd.md)（Run 終了側の対）、
> [052 eval 定義/スケジュール分離](done/052_eval_schedule_separation_10prd.md)、
> [053 interval bucket-crossing](done/053_interval_gate_perf_ema_10prd.md)

## Context

`run.eval_schedule.[tag].interval` で組んだ eval は、**step 値によらず初回に必ず 1 回発火する。**
実体は `IntervalGate::ShouldFire`（`core/anet-core/include/anet/util.hpp:271`）の `!fired_` 枝である。

```cpp
bool ShouldFire(uint64_t step) {
    const uint64_t bucket = step / interval_;

    // 初回は step 値によらず発火し、以後の基準バケットを決める。
    if (!fired_) {
        fired_ = true;
        last_bucket_ = bucket;
        return true;
    }

    // 複数バケットを跨いでも発火は 1 回（catch-up しない）。
    if (bucket > last_bucket_) {
        last_bucket_ = bucket;
        return true;
    }

    return false;
}
```

`EpisodeEvalObserver` はこのゲートを持ち（`observers.hpp:236`、生成は `observers.cpp:508`、
判定は `observers.cpp:576`）、学習開始直後の 1 回目を必ず実行する。

**これはバケット境界の副作用ではなく明示的な規則である。**したがって発火グリッドの位相を
ずらすだけの変更では抑止できない。

### 実測

2026-09-21、Atari-5 の 5 ゲームを同一設定（10M / RR4 / `run.@eval2ch`）で回したときの
`greedy_dist` learn_step=0 セッション。400 エピソード / `eval_batch_size` 16。

| game | セッション所要 | 次回発火時の `waited` | `mean.episode_steps` | `mean.episode_return` |
|---|---|---|---|---|
| qbert | 53.6s | なし | 275 | 0 |
| phoenix | 63.4s | なし | 386 | 0.90 |
| battle_zone | 292.8s | なし | 1,578 | 3.50 |
| name_this_game | 3,806.4s | **692.6s** | **26,963** | 0 |
| double_dunk | 4,110.2s | **885.5s** | **26,992** | −1.46 |

未学習の貪欲方策が行動を退化させ、**全エピソードが打ち切り上限 27,000 step
（`max_episode_frames` 108,000 / `frame_skip` 4）に張り付く**。張り付いた 2 ゲームだけが
セッション所要時間で 1 時間級になる。

**セッション所要時間は train の停止時間ではない。** `use_background = true` では
`eval_pool_` の別スレッドで走り、`OnLearn`（`observers.cpp:565`）が毎 learn step で呼ぶのは
`RethrowCompletedBackgroundEval()`（future の状態確認）だけである。train が実際に止まるのは
**次回発火時に前回が終わっていない場合の `WaitBackgroundEval`** に限られ、それが上表の `waited` 列である。

- **train 停止は 2/5 ゲームで 692.6s と 885.5s、合計約 26 分。**
- **自己限定的で暴走しない。** セッション所要は 400 エピソード × 上限 27,000 step ÷ 16 lane で
  上に有界（実測 4,110s）であり、発火間隔の実時間（5.05M exp ÷ 約 1,600 exp/s ≒ 3,150s）を
  超えた分だけが停止になる。滞留は最大 1 本。
- **背景実行中の資源競合は測れなかった。** t=0 セッションは warmup 直後（0.2M〜6.4M）に走り、
  この区間はスループット自体が立ち上がりの谷にある（Breakout 素の曲線で 0.5-1M 3,251 →
  2-5M 1,930 → 5-10M 2,040 exp/s）。セッション中と直後の差は name_this_game −12.5% /
  double_dunk −1.3% と揃わず、**セッションの影響か曲線の形か分離できない。ゼロの可能性がある。**
- **どのゲームで起きるかは事前に分からない。** 同一の未学習方策・同一設定で 5 ゲーム中 2 つだけが該当し、
  スコアの高低・エピソード長の長短のいずれとも対応していない（phoenix は学習後のエピソードが
  最長の部類だが t=0 は 386 step で終わる）。
- **Breakout でも発生済み。** `apps/runner/config/Atari.txt` の `run.@eval2ch` 注記に
  「2026-09-18 の RR1 で 49 分」として記録がある。
- **この点は捨て点である。** 実験記録
  [2026-09-15 探索ブロック 07](../experiments/default-dqn/atari/2026-09-15_replay-lane-window.md)
  の ε=0 断面 9 点ペア表は 5.2M 始まりで、learn_step=0 の点を使っていない。

### 規模と優先度

1 Run につき 1 回だけの固定コストで、予算を 10M から 50M へ伸ばしても増えない。
**5 ゲーム × 50M（約 47.5h）のスイープに対して train 停止は約 26 分 = 0.9%。**

**したがって本 PRD の根拠は実行時間の節約ではない。** 主たる価値は

1. **ε=0 断面の系列に、未学習方策を測った無意味な点が先頭に入らないこと**（解析時の取り違えを防ぐ）
2. 停止が有界とはいえ、**発生の有無をゲーム別に事前予測できない**ままにしないこと

の 2 点である。実行時間だけを見るなら着手は急がない。

## ゴール

- eval スケジュールが**初回発火を抑止できる**ようにする。
- 既定では現行挙動を変えない。既存の Run 定義・実験記録のレシピが同じ発火列を再現する。
- `IntervalGate` の他の利用者（メトリクス observer 群、`dqn_based_agent`、`image_cls_agent`）の
  挙動を変えない。

## 非ゴール

- **Run 終了時に背景 eval セッションが捨てられる問題**。これは
  [PRD 076](done/076_eval_session_drain_on_exit_10prd.md) の範囲で、発火列の先頭と末尾という別の端である。
- **未学習の貪欲方策が行動退化すること自体**。方策として正当な挙動であり、本 PRD は
  「それを測らない」だけを扱う。
- **eval エピソード上限（`max_episode_frames`）の変更**。上限を下げれば t=0 は安くなるが、
  学習後のエピソードを打ち切って測定値にバイアスを入れる。
- **メトリクス側の interval への同種キーの追加**。`IntervalGate` へ足す機構は共有だが、
  設定キーを公開するのは `run.eval_schedule` のみとする。

## 確定事項

### D1: キーの形 — 裁定待ち

`run.eval_schedule.[tag]` に足すキーの形に 2 案ある。**本 PRD はここだけ未決とする。**

| | 案 A: `skip_first`（回数） | 案 B: `offset`（step 値） |
|---|---|---|
| 値 | 抑止する初回発火の回数（非負整数） | 発火を開始する step（非負整数） |
| 今回の用途 | `skip_first = 1` | `offset = <interval と同値>` |
| 単位換算 | **不要** | **必要**（下記） |
| 一般性 | 発火回数の粒度でしか刻めない | 任意の step で開始できる |

**案 A を推す理由は単位換算である。** `interval` は learn_step 基準で、`Atari.txt` の
`run.@btreval_r*` 注記にあるとおり `replay_ratio` ごとに手換算が要る。

```
250,000 exp / 1 更新あたり exp → RR1 = 977   RR2 = 1953   RR4 = 3906
```

`offset` を同じ単位で入れると、**腕ごとにこの換算をもう一度書く**ことになる。
`run.@eval2ch`（RR4 で `greedy_dist` interval 78,907）と `run.@eval2ch_r1`（RR1 で 39,453）は
すでに同じ値を 2 系統持っており、offset を足すと 4 箇所になる。回数なら換算が要らず
`replay_ratio` に依存しない。

**判定に効かない論点として、発火グリッドの位相ずれは理由にしない。** offset を interval の
倍数に取れば絶対グリッドは動かず、また Run あたりの断面は 10 点程度しかないため、
多少の位相ずれは Run 間の k 番目対比較を壊さない。

### D2: 既定値は現行挙動

新キーの既定は「抑止なし」（案 A なら 0、案 B なら 0）。キーを書かない既存の Run 定義は
発火列が 1 点も変わらない。

### D3: `IntervalGate` への追加は既定で不活性

`IntervalGate` は 4 ファイル 10 箇所で使われる共有部品である
（`util.hpp` / `observers.hpp` / `dqn_based_agent.hpp` / `image_cls_agent.hpp`）。
抑止の判定は **`ShouldFire` の先頭に早期 return を 1 本足す形**とし、既定値では
その分岐に入らないことをコード上で自明にする。既存の `!fired_` 規則そのものは変えない。

`step` は `uint64_t` なので、既定 0 の比較は恒偽であり、既存利用に対して
**証明可能に無害**である。

### D4: `interval = 0`（definition-only）との関係

`interval = 0` はスケジュール無効を意味し、`trainer.cpp:919` で dormant 扱いになる。
このとき新キーは読まれず、値が何であってもエラーにしない。

### D5: 検証は interval と同じ強さで行う

`trainer.cpp:777-797` の `interval` と同様に、負値を `ANET_SYSTEM_ERROR` で弾く。
キー自体は任意（未指定なら既定値）。

### D6: 適用は全 eval タグ共通

`run.eval_schedule.[tag]` 単位で指定する。`greedy_dist` だけの特別扱いはしない。
`[eval]`（ε=0.01 / 10 エピソード）の t=0 は安価なので実用上は指定しないが、
機構としては同じ扱いにする。

## 実装契約

### `core/anet-core/include/anet/util.hpp`

`IntervalGate` に抑止用のメンバを 1 つ足し、コンストラクタへ既定値付き引数を追加する。
`ShouldFire` の先頭で抑止条件を判定して早期 return する。既存の 1 引数コンストラクタは
既定値経由でそのまま使えること。

### `core/anet-core/include/anet/observers.hpp` / `src/observers.cpp`

`EpisodeEvalObserver` のコンストラクタ（`observers.hpp:220`）へ新パラメータを足し、
`observers.cpp:508` の `eval_gate_.emplace()` へ渡す。`observers.cpp:576` の判定箇所は変更しない。

### `core/anet-core/src/trainer.cpp`

- `EvalScheduleConfig`（`trainer.cpp:752`）へフィールドを 1 つ追加。
- `trainer.cpp:777-797` のパースで新キーを読み、D5 の検証を行う。
- `trainer.cpp:926` の `scheduled` ログへ値を出す（既定値のときも出す。Run ログから
  発火列が再構成できることを保つ）。
- `trainer.cpp:976-982` の `AttachScoped<EpisodeEvalObserver>` へ渡す。

### `apps/runner/config/Atari.txt`

`run.@eval2ch` と `run.@eval2ch_r1` の `greedy_dist` に対して抑止を有効化する。
`run.@eval2ch_r1_50m` も同系列なら揃える。

既存の腕を書き換えると実験記録に残る過去レシピの発火列が変わるため、
**新しい腕を足すか、既存の腕を書き換えるか**は起票時点では決めない。
判定は D1 の裁定と同時に行う。

## テスト

`core/anet-core/src/util_test.cpp`（`IntervalGate` の既存テストと同じ場所）へ足す。

1. **既定値で挙動が変わらない。** 新パラメータを渡さない構築で、既存テストと同じ発火列になる。
2. **抑止が効く。** 抑止を指定した構築で、最初の N 回が発火せず、その後の発火が
   指定なしの N+1 回目以降と**同じ step で**起きる（絶対グリッドが動かないこと）。
3. **抑止量が発火列より大きい場合。** すべて抑止され、`ShouldFire` が常に false を返しても
   異常終了しない。
4. **飛び飛びの step。** 抑止区間を 1 回の呼び出しで跨いだとき、catch-up せず 1 回だけ発火する
   （既存の bucket-crossing 契約を壊さない）。

`trainer` 側は設定パースの検証（負値の `ANET_SYSTEM_ERROR`、`interval = 0` との共存）を
既存の eval schedule テストへ足す。

## 受入基準

1. 新キーを書かない既存の Run 定義で、eval の発火 step 列が変更前と一致する。
2. 抑止を指定した Run で、`greedy_dist` の learn_step=0 セッションが実行されない。
3. 抑止後の 1 回目以降の発火 step が、抑止なしの 2 回目以降と一致する。
4. Run ログの `eval.[tag]: scheduled (...)` 行から、抑止設定を含む発火条件が読める。
5. `IntervalGate` を使う既存 4 ファイルのテストが変更なしで通る。
6. Atari-5 の `name_this_game` または `double_dunk` を 10M で 1 本回し、
   `eval.[greedy_dist]: waited for previous session` が 1 度も出ないことを確認する
   （抑止前の実測はそれぞれ 692.6s / 885.5s）。**wall-clock の短縮幅は受入条件にしない** —
   背景セッションの資源競合が分離できていないため、短縮幅は 26 分ぶんの停止解消を
   下限とするのみで、上限は未知である。

## 影響・移行

- **設定の後方互換は保たれる。** 新キーは任意で、既定が現行挙動である。
- **実験記録との関係。** 抑止を有効にした Run は ε=0 断面の点数が 1 点減る（10 点 → 9 点）。
  記録の 9 点ペア表は learn_step=0 を使っていないため、**対比較の作り方は変わらない**。
  抑止前後の Run を対にするときは、新 Run の k 番目を旧 Run の k+1 番目と対応させる。
- **`run.@eval2ch` の注記を更新する。** 「先頭は learn_step 0 でも発火する（この 1 回は
  捨て点になる）」は抑止後に事実でなくなる。
- **PRD 076 とは独立に適用できる。** 発火列の先頭（本 PRD）と末尾（076）で、
  同じファイルに触るが競合はしない。
