# RFD0205: 用語集と設計 140・160 を、今のコードと ADR に合わせる小さい直しをまとめて行う

> 2026-11-27 起票（暫定）。質疑を省いた。確定した決定は持たず、AI が置いた既定を §5 の未決表に置いた。§6・§7 は §5 の暫定の既定を前提に書き、前提にした行の番号を名指す。
> 2026-11-27 改訂。入口レビューを受け、checkpoint の保存内容、agent step の定義、設計 140 の記録フローと受入条件をそろえた。
> 起点: 2026-11-27 の会話。開発者は、仕様の調整がほぼ要らない小さい課題を集めて、Codex に夜のうちに実装させたいと述べた。課題は置き場ごとに 3 本の RFD に分けた（本書が文書、[RFD0206](0206_core_and_app_small_fixes_10rfd.md) が基盤とアプリ、[RFD0207](0207_dqn_small_fixes_and_refactors_10rfd.md) が DQN 系）。分け方は開発者が選んだ（§10）。
> 課題: 題材は 10 件（§9 の対応表）。

## 1. 目的と問題

### 1.1 問題

2026-11-23 と 11-25 の定常整備の監査（観点 drift）で、用語集（`CONTEXT.md`）と設計文書 140（可観測性）・160（アプリケーションとツール）・200（DQN 系）・212（RND）・020（Run の実行の利用者の手引き）に、今のコードや ADR と食い違う記述と、欠けた記述が見つかった（P11）。どれも文書だけの直しで、直し方は課題に書いてある。

食い違いは 3 種類に分かれる。

- **用語集の定義が今の動きと合わない**: network snapshot と評価セッションが `clone_model` の条件を書かない（75512f）、trace の避ける語が RFD の番号を記号なしの「144」と書く（89526b）、target return が「ReplayBuffer に保持される値」とだけ書く（ff1a3f）、Agent checkpoint と checkpoint 引き継ぎが今の保存の中身を書かない（d367c3）、step の軸と lane を定義なしで使う（fb50b6）。
- **設計文書から ADR へたどれない**: 160 と 140 が本文で仕組みを書くのに、その判断の ADR へのリンクが無い（0fcb2a、bdbcac）。
- **設計文書の図・並び・属性が今の形と合わない**: 160 の図の node 名の重なり、コードマップの欠け、FPS の選択肢（f3cad3）、140 の trace の行の時刻（195ad5）、140 の節番号・置き場・図・コードマップ（ae4c45）。

### 1.2 ゴール

- 題材の課題 10 件がすべて片付く（AI の判定が「残件無し」になる）。
- 用語集と設計文書 140・160・200・212・020 の該当箇所が、今のコードと ADR のとおりに読める。英訳（`.en.md`）も和文に追随する（U11）。

### 1.3 非ゴール（退けた案と、本当の制約か）

- **コードを変えない。** 食い違いはどれも文書の側にある（§3 で確かめた）。本当の制約である。コードを直す課題は [RFD0206](0206_core_and_app_small_fixes_10rfd.md)・[RFD0207](0207_dqn_small_fixes_and_refactors_10rfd.md) が持つ。
- **ADR 0047 に「行の時刻は未実装」の注記を足さない**（U8）。課題 195ad5 は足すかを決めるとしていた。ADR は当時の判断の記録で、実装の状態は課題（[issue-20261103-955e50](../management/issues/issue-20261103-955e50.md)、RFD0081）が持つ（P8）。注記を足すと RFD0081 の実装のあとに古くなる。本当の制約ではなく、選択である。
- **`sim_step` の扱い**は扱わない。課題 [issue-20261123-ad7829](../management/issues/issue-20261123-ad7829.md) が、消すか進めるかの判断を持つ。用語集に `sim_step` の項を足すと、その判断を先取りする。
- **ほかの避ける語の残り**（[issue-20261123-c7c3ae](../management/issues/issue-20261123-c7c3ae.md)）、**コードの名前とエラー文の避ける語**（[issue-20261123-6bb71d](../management/issues/issue-20261123-6bb71d.md)）、**ADR へのたどり方の全体の方針**（[issue-20261114-fbcbca](../management/issues/issue-20261114-fbcbca.md)）、**160 §7.4 を開発環境の文書へ移すこと**（[issue-20261115-3e5b77](../management/issues/issue-20261115-3e5b77.md)）、**画像の Observer の置き場の見直し**（[issue-20261123-eac719](../management/issues/issue-20261123-eac719.md)）は扱わない。どれも別の判断を持つ別の課題で、本書の直しはそれらの判断に依らない。本当の制約ではなく、範囲の分け方である。
- **観測の正規化の統計を checkpoint に保存する仕事**（[issue-20261113-b96b2c](../management/issues/issue-20261113-b96b2c.md)）は扱わない。入ったときに、Agent checkpoint の列挙をその課題が直す。
- **ほかの RFD の本文の節番号の参照は直さない**（U10）。RFD は書いた時点の決定の記録である。本当の制約ではなく、選択である。

### 1.4 機構とゴールの対応

新しい機構・概念は足さない。用語集に足す 6 項（U5）は、設計文書 010 §3.3 と 120 が既に定めている語を、用語集から引けるようにするもので、新しい概念ではない。

### 1.5 成功の測り方

- 題材の課題 10 件の AI の判定が「残件無し」になる。
- 次の定常整備の監査（観点 drift）で、§3 の食い違いが再び挙がらない。

## 2. 前提

| 番号 | 種類 | 内容 | 出典 | 確かめ |
|---|---|---|---|---|
| P1 | 守る線 | `CONTEXT.md` は用語集で、実装の詳細を含めず、定義は 1〜2 文にする | [CONTEXT-FORMAT.md](../../.agents/skills/anet-create-rfd/CONTEXT-FORMAT.md) の Rules | 対象外 |
| P2 | 事実 | DefaultDQN は、`clone_model` が真の Actor にだけ `sync_interval` を渡す | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の 946 行 | 対象外 |
| P3 | 過去の判断 | γ が schedule の成分は、抽選時に保存済みの step 報酬から収益を作り直す。報酬の成分ごとに当てる | [ADR 0065](../adr/0065-scheduled-gamma-recomputes-return-at-sample.md)、[ADR 0073](../adr/0073-replay-keeps-reward-components-and-builds-returns-per-component-discount.md) | 対象外 |
| P4 | 事実 | `DefaultDQNAgent::Save` は Network と Learner のあとに、内的報酬（RND 本体、補助学習の credit、抽選の乱数、成功回数、内的報酬スケーラ）を書き、拡張形式では外的の報酬スケーラも書く | [default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) の 724 行から、[issue-20261123-d367c3](../management/issues/issue-20261123-d367c3.md) の事象 | 対象外 |
| P5 | 事実 | step の軸（train・exp・update・learn など）の定義は設計 010 §3.3 の表にあり、lane は設計 120 の BatchEnv の節が書く | [010 §3.3](../design/010_framework_overview.jp.md)、[120](../design/120_environments.jp.md) の 35 行 | 対象外 |
| P6 | 過去の判断 | 設計文書から ADR へのたどり方は、設計 200 §11.2 の ADR の節（題と対応する節を番号順に並べる）を手本にする | [200 §11.2](../design/200_dqn_agents.jp.md)、[issue-20261118-7f7757](../management/issues/issue-20261118-7f7757.md) | 対象外 |
| P7 | 既存部品の働き | 英訳は先頭の行に元の和文の blob を持ち、翻訳の skill は blob が今の和文と違う英訳を「更新あり」として訳し直す | [anet-translate-docs](../../.agents/skills/anet-translate-docs/SKILL.md) の 17・21 行 | 対象外 |
| P8 | 守る線 | 過去の記録（ADR・実験記録・履歴資料）を書き換えない。コミット済みの ADR の本文は書き換えず、判断を変えるときに先頭へ注記を足す | [AGENTS.md](../../AGENTS.md) の「クリーンブレーク方針」、[anet-create-rfd](../../.agents/skills/anet-create-rfd/SKILL.md) の「Offer ADRs sparingly」 | 対象外 |
| P9 | 事実 | 160 §7.4 を移す課題 3e5b77 と、画像の Observer の置き場の課題 eac719 は、どちらも未着手である（2026-11-27） | [issue-20261115-3e5b77](../management/issues/issue-20261115-3e5b77.md)、[issue-20261123-eac719](../management/issues/issue-20261123-eac719.md) の status | 対象外 |
| P10 | 事実 | trace の行に時刻を付ける RFD0081 は未実装で、課題 955e50 は調査中である。`MetricsLogger::LogTrace` は `type`・`tag`・`step`・`lane`・`data` だけを書く | [issue-20261103-955e50](../management/issues/issue-20261103-955e50.md)、[metrics_logger.hpp](../../src/core/base/include/anet/metrics_logger.hpp) の 201〜206 行（課題 195ad5 の事象） | 対象外 |
| P11 | 事実 | 題材の事象は 2026-11-23・11-25 の監査で確かめたもので、行番号はその日の値である。2026-11-27 に主な箇所を引き直し、§3 に今の行番号を書いた | 各課題の事象、§3 | 対象外 |
| P12 | 事実 | exp step は、1 回の Step で得た transition の数（lane の数）ずつ進む | [trainer.cpp](../../src/core/base/src/trainer.cpp) の 312・626・760 行（`step_counts_.exp_step += result->n_transitions`） | 対象外 |

## 3. 現状（2026-11-27 に確かめた）

| 課題 | 確かめたこと |
|---|---|
| 75512f | [CONTEXT.md](../../CONTEXT.md) 14〜16 行の **Train Actor network snapshot** は、`sync_interval` を宣言したときだけ定期 snapshot を行うと書き、`clone_model` の条件を書かない。コードは `clone_model` が真のときだけ `sync_interval` を渡す（P2）。443〜445 行の **評価セッション** は「一つのnetwork snapshotを使い」と書く。`apps/runner/config/common.txt` の 9 行は、`clone_model=false` は学習中の network を共有すると書く |
| 89526b | CONTEXT.md 586 行の trace の `_Avoid_` が「144のEpisode Journal」と書く。指す先は [RFD0144](0144_episode_forensics_10rfd.md) |
| ff1a3f | CONTEXT.md 42〜43 行の **target return** が「ReplayBuffer に保持される値」とだけ書く（P3 と食い違う） |
| d367c3 | CONTEXT.md 518〜524 行の **Agent checkpoint** と **checkpoint 引き継ぎ** が、保存の中身を header・Config 文字列・Network・optimizer と書き、RND・報酬スケーラ・補助学習の抽選の乱数を書かない（P4）。設計 212 の 406 行の見出しが「9.7 保存と再開」で、228・269・387・410 行も「再開」を使う。設計 200 の 644 行は、この見出しのアンカー `#97-保存と再開` を引く。200 の 663 行は「NetworkとOptimizerを引き継ぐ操作」のままで、173・825 行も「再開」を使う。設計 020 の 534 行は、引き継ぎの例を `R.auto_load_file` と書く |
| fb50b6 | CONTEXT.md は exp step・train step・agent step・lane を、用語の項なしで使う（lane は 23 行）。step の軸の定義は 010 §3.3 にだけある（P5）。010 §3.3 の表に agent step と lane は無い |
| 0fcb2a | 設計 160 に `adr/` へのリンクが無い。§9（446 行）の関連文書は ADR の節を持たない。本文の「ADR 0022」（§5.1）は文字だけである |
| f3cad3 | 160 §4 の mermaid の図が、198 行（`DB[(Run-local SQLite cache)]`）と 219 行（`DB[(optuna.db)]`）で同じ node の ID `DB` を使う。§3.1（72 行から）のコードマップに EvalTracePanel・HeatMapPanel・Conv2dPanel・ErrorDialog と `gui.hpp`・`gui.cpp` が無い。§7.1 は FPS の選択肢に `Config` を書かない |
| 195ad5 | 140 の trace チャネルの節（311 行から）が、trace の行が `timestamp` を持つと書く。実装は持たない（P10） |
| bdbcac | 140 §10（457 行）の関連文書は ADR の節を持たず、140 がリンクする ADR は 0028・0029・0037 の 3 本である |
| ae4c45 | 140 の 264 行と 311 行の見出しが `### 6.x` のまま、335 行が `### 6.1`。`env_step` の節（469 行）が §10 の下、TempoRL の節（428 行）が §9 の途中にある。§4 の図に SessionEnd の Observer が無く、`ImageObserver` という実在しないクラスがある。§3 のコードマップに metrics・image・heat_map・gui が無い |

## 4. 用語

本書が用語集に足す語は U5 に挙げる。足す定義は 010 §3.3 と 120 の定義を写すもので、新しい意味を持たせない。

## 5. 未決表

| 番号 | 論点 | 暫定の既定 | 理由と依存 |
|---|---|---|---|
| U1 | network snapshot と評価セッションの定義（75512f） | **Train Actor network snapshot** に「`clone_model` が真の Actor だけが持ち、`sync_interval` を宣言したときに定期的に取り直す」と書く。**評価セッション** の「一つのnetwork snapshotを使い」を、「`clone_model` が真なら一つの network snapshot を使い、偽なら学習中の network を共有する」に改める | コードの条件（P2）と common.txt の説明に合わせる。1〜2 文に収める（P1） |
| U2 | trace の避ける語の番号（89526b） | 「144のEpisode Journal」を「RFD0144 の Episode Journal」に改める | 番号の書き方を RFD0195 の 4 桁に合わせる |
| U3 | target return の定義（ff1a3f） | 「ReplayBuffer に保持される値」を「γ が定数の成分は Push で保存した値、schedule の成分は抽選時に保存済みの 1 step の報酬から作る値」に改める | P3 |
| U4 | checkpoint の 2 語と、「再開」の言い換え（d367c3） | (a) **Agent checkpoint** の DefaultDQN の中身に、内的報酬（RND とそのスケーラ、補助学習の credit・抽選の乱数・累計成功回数）と動的な報酬スケーラを足し、含まないものを「Replay・step・Actor と Replay の乱数」と書き直す。**checkpoint 引き継ぎ** も同じ中身に合わせる。(b) 設計 212 の、checkpoint の読み込みを指す「再開」を「引き継ぎ」に改める（見出し「9.7 保存と再開」は「9.7 保存と引き継ぎ」）。200 の 644 行は checkpoint の読み込みを指す「再開時」も言い換え、リンクを新しい見出しに合わせる。200 の 173・663・825 行のうち checkpoint の読み込みを指す「再開」も同じに改め、663 行の「NetworkとOptimizerを引き継ぐ操作」を (a) の中身に合わせる。checkpoint の読み込みを指さない「再開」（一時停止からの再開など）は残す。(c) 020 の 534 行の `R.auto_load_file` を `A3.auto_load_file` に改める | (a) は P4。(b) は用語集が checkpoint 引き継ぎの避ける語に「再開」を挙げるため。(c) は、R が旧名の上書き層で、出荷の設定と bat が A3 を使うため（課題 d367c3 の事象）。完全キーで書く案もあるが、利用者の手引きの例は bat と同じ書き方にそろえる |
| U5 | step の軸と lane の項（fb50b6） | 用語集に **train step**・**exp step**・**update step**・**learn step**・**agent step**・**lane** の 6 項を足す。train・exp・update・learn は 010 §3.3 の表の定義を写す。agent step は「行動を受けて 1 lane の Env が 1 回進む単位で、Atari では設定に応じて複数のエミュレータフレームを含む。N lane すべてが遷移する BatchEnv では、各 lane が 1 agent step 進むと exp step は N 進む」とし、`_Avoid_` に「`$agent_step`（Evaluation Trace の列の取り出し先で、単位ではない）」を書く。lane は 120 の定義（BatchEnv の中の 1 本の Env）を写す。置き場は **エピソード長** の項の前にまとめる | 定義の正は 010 §3.3 と 120（P5）。`sim_step` は扱わない（§1.3）。agent step の文は、用語集 97 行（TempoRL のスキップ）の使い方と、exp step が transition の数を数える定義（010 §3.3、P12）から組んだ。実装する側は、定義を書く前に 010・120・220 とコード（`StepCounts`、`rl.hpp` 106〜112 行）で確かめ、食い違えば直してよい |
| U6 | 160 の ADR の節（0fcb2a） | 160 §9 に「ADR」の小節を設け、0016・0021・0022・0045・0056・0057・0066・0068・0069・0081・0082 の 11 本を番号順に、200 §11.2 と同じ形（題のリンクと、対応する 160 の節）で並べる。§5.1 の「ADR 0022」の文字をリンクにする。0056・0068・0069・0081 は §7.4 に対応させる | P6。§7.4 を移す課題 3e5b77 は未着手なので（P9）、リンクは 160 に置く。3e5b77 が移すときに、リンクも一緒に移る |
| U7 | 160 の図・コードマップ・FPS（f3cad3） | §4 の Optuna 側の node の ID を `ODB` にする（ラベルは `optuna.db` のまま。その node を指す辺も `ODB` にする）。§3.1 のコードマップに EvalTracePanel・HeatMapPanel・Conv2dPanel・ErrorDialog の行と、`src/core/base/include/anet/gui.hpp`・`src/core/base/src/gui.cpp`（`gui::TracePlayer` ほか）の行を足す。§7.1 の FPS の選択肢の先頭に `Config`（設定の値）を足す | 課題の対応方針のとおり。node の ID は図の全体で一意でないと 1 つの node にまとまる |
| U8 | trace の行の属性と ADR 0047（195ad5） | 140 の trace チャネルの節の文を、今の契約（固定属性は `tag`・`step`・`lane`。時刻は持たない）に戻す。ADR 0047 には注記を足さない。行の時刻は RFD0081 の実装で書き足す | 実装と記録された行が持たない（P10）。ADR の扱いは §1.3 |
| U9 | 140 の ADR の節（bdbcac） | 140 §10 に「ADR」の小節を設け、0028・0029・0034・0037・0043・0047・0055・0060・0070・0071・0072 の 11 本を番号順に、200 §11.2 と同じ形で並べる。0047 の行には「行の時刻は未実装」と添える | P6。0047 の時刻は U8 のとおり未実装なので、節を読んだ人が時刻があると読まないようにする |
| U10 | 140 の節番号・置き場・図・コードマップ（ae4c45） | (a) §6 の小節を並びの順に 6.1 step座標系・6.2 trace チャネル・6.3 Evaluation Traceの列と読み取り・6.4 遷移値のtarget `env_step`（§10 の下から移す）・6.5 TempoRL の scalar と Trace（§9 の途中から移す）と振る。§9 の確認事項の一覧を前の段落とつなげる。(b) §4 の静的構造の図に `SessionEndObserver`・`MetricsLogSessionEndObserver`・`MetricsLogTraceObserver` を足す。`ImageObserver` を、実在する Train の 3 種（`HeatMapVectorObserver`・`MultiPairHeatMapObserver`・`Conv2dVisualizationObserver`）と Learn の 2 種（`TimeHistogramObserver`・`SweepedHeatMapObserver`）に直し、作り手を `ImageProviderFactory`・`ImageProviderManager` にする。§2 の「Image/Graph Observer」の行にも作り手を書く。(c) §5.1 の記録の流れの図に、SessionEnd の scalar と EpisodeEnd の trace の記録経路を足す。(d) §3 のコードマップに `metrics.hpp`・`metrics.cpp`、`image.hpp`・`image.cpp`、`heat_map.hpp`・`heat_map.cpp`、`gui.hpp`（`TracePlayer`）の行を足す。(e) 140 の §6 の節番号を書く参照は、追跡しているファイルの全体で 5 ファイル 6 行あり（§6.4）、どれも RFD の本文なので直さない | (a) は、節の番号で引けるようにするため。env_step と TempoRL は記法と指標の説明なので §6 に置く。(b)(c) はコードのクラスに合わせる（課題 ae4c45 の事象。`observers.hpp` の 73〜287 行、`image.hpp` の 320〜354 行）。画像の Observer の置き場を見直す課題 eac719 は未着手で（P9）、見直したときに図もその課題が直す。(e) は §1.3 のとおり。RFD0081 の本文の「140 §6.x」（trace チャネル）は、直したあとの §6.2 に当たる（§6.5） |
| U11 | 英訳の追随 | 和文を直した設計文書（140・160・200・212・020）の英訳は、和文を直し終えてから、翻訳の skill（`anet-translate-docs`）でファイルを指定して追随させる。英訳を手で部分的に直さない | 英訳は元の和文の blob を持ち、skill が差を見て訳し直す（P7）。手で直すと blob の行と中身がずれる。用語集の訳語の一覧も skill が保つ |
| U12 | 確かめ方 | build とテストは要らない。§7 の検索と、[memo README](README.md) の「リンクの注意」のリンク切れ検査で確かめる | 文書だけの直しで、コードを変えない（§1.3） |
| U13 | 3 本の RFD の実装の順 | 本書 → RFD0206 → RFD0207 の順に実装する | 本書は build が要らず、先に済む。RFD0207 は設計 200 の別の行（§9.1 の NaN の条件など）を直しうるので、本書のあとに入る（§6.5） |

## 6. 実装契約

### 6.1 変えるもの

| ファイル | 直す箇所 | 未決表 |
|---|---|---|
| [CONTEXT.md](../../CONTEXT.md) | Train Actor network snapshot、評価セッション、trace の `_Avoid_`、target return、Agent checkpoint、checkpoint 引き継ぎ、step の軸と lane の 6 項の追加 | U1〜U5 |
| [設計 212](../design/212_rnd_lifelong_novelty.jp.md) | checkpoint の読み込みを指す「再開」、§9.7 の見出し | U4 |
| [設計 200](../design/200_dqn_agents.jp.md) | 644 行の本文とアンカー、173・663・825 行（checkpoint の読み込みを指す「再開」と引き継ぐものの列挙） | U4 |
| [設計 020](../design/020_user_guide_run.jp.md) | 534 行の引き継ぎの例 | U4 |
| [設計 160](../design/160_applications_and_tools.jp.md) | §9 の ADR の節、§5.1 の ADR 0022 のリンク、§4 の図の node の ID、§3.1 のコードマップ、§7.1 の FPS の選択肢 | U6・U7 |
| [設計 140](../design/140_observability.jp.md) | trace の行の属性、§10 の ADR の節、§6 の節番号と置き場、§9 の一覧、§2 の表、§4・§5.1 の図、§3 のコードマップ | U8〜U10 |
| 上の設計文書の英訳（`.en.md`） | 翻訳の skill で追随 | U11 |

### 6.2 変えないもの（本当の制約か）

- コード、設定、テスト。食い違いは文書の側にある（§3）。本当の制約である。
- ADR の本文と注記（U8。§1.3）。
- ほかの RFD の本文（U10 の (e)。§1.3）。
- 過去の記録（`docs/memo/done/` などの保管場所、`docs/experiments/`、`reports/`、`docs/wiki/log.md`）。P8 による本当の制約である。

### 6.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。ただし、一覧に無い機能グループを新しく立てるときは、作る前に開発者に確かめる。

足す・変える・消すクラスは無い。

### 6.4 移行対象

文書だけを直すので、移す現用のコード・設定・テストは無い。名前を変えるもの（212 の見出し「保存と再開」と、140 の §6 の節番号）を指す箇所は、追跡しているファイルの全体を 2026-11-27 に検索した。

| 検索 | 当たり | 扱い |
|---|---|---|
| `git grep -n "保存と再開"` | 設計 212 の見出し 1 行と、200 の 644 行のアンカー（`#97-保存と再開`） | どちらも直す（U4） |
| 下の枠の検索 3（140 の §6 の節番号を書く参照） | 5 ファイル 6 行: `docs/memo/0081_evalpanel_episode_trace_10rfd.md`・`0127_eval_trace_intrinsic_reward_columns_10rfd.md`・`0184_eval_session_step_trace_10rfd.md` が各 1 行、`docs/memo/done/0094_evalpanel_step_history_20impl.md` が 1 行、`docs/memo/done/0103_temporl_reference_alignment_10rfd.md` が 2 行 | 直す 0 件、残す 6 行（RFD の本文と保管場所の記録。U10 の (e)） |
| `git grep -n "R.auto_load_file" -- docs/design` | 020 の 534 行（和文）と、その英訳 | 和文を直し、英訳は U11 |

検索 3（縦棒を含むので表の外に書く。範囲は追跡しているファイルの全体）:

```text
git grep -n -E "140_observability\.(jp|en)\.md#6|140 ?(の )?§6\.[0-9x]|140§6"
```

### 6.5 ほかの RFD との実装の順序

- 3 本の順は U13 のとおり。RFD0206・RFD0207 が 140・200 を直すときは、本書のあとの節番号と行で直す。
- [RFD0081](0081_evalpanel_episode_trace_10rfd.md)（trace の行の時刻、未実装）は、移行対象の表で「140 §6.x」（trace チャネル）を直すと書く。本書のあとは §6.2 に当たる。RFD0081 の実装は、U8 で戻した文に時刻を書き足す。どちらが先でもよいが、RFD0081 が先に入ったときは、U8 は不要になり、U9 の 0047 の行の添え書きも外す。
- 課題 3e5b77（160 §7.4 の移動）と eac719（画像の Observer の置き場）は、本書のあとに入る前提で、移すときに本書が足したリンクと図を一緒に直す（P9）。

## 7. 受入

| 番号 | 確かめること | 確かめ方 |
|---|---|---|
| A1 | network snapshot と評価セッション（U1） | `CONTEXT.md` の **Train Actor network snapshot** と **評価セッション** の 2 項の本文に、それぞれ `clone_model` の文字が 1 回以上ある |
| A2 | trace の避ける語（U2） | `git grep -n "144のEpisode" -- CONTEXT.md` が 0 行。`git grep -n "RFD0144 の Episode Journal" -- CONTEXT.md` が 1 行 |
| A3 | target return（U3） | `CONTEXT.md` の **target return** の本文に「抽選時」の文字があり、「ReplayBuffer に保持される値」の文字が無い |
| A4 | checkpoint の 2 語と言い換え（U4） | (a) `CONTEXT.md` の **Agent checkpoint** の本文に RND と報酬スケーラの文字がある。(b) `git grep -n "保存と再開" -- docs/design` が 0 行（和文と英訳とも。英訳は U11 のあと）。(c) `git grep -n "R.auto_load_file" -- docs/design` が 0 行（英訳は U11 のあと）。(d) 設計 200・212 の和文で「再開」を検索し、当たった行がどれも checkpoint の読み込みを指さないことを、実装計画に行ごとに書く。(e) 設計 200 の和英リンクをたどり、対応する設計 212 の §9.7 の見出しに一致することを確認する |
| A5 | step の軸と lane（U5） | `CONTEXT.md` に、`**train step**`・`**exp step**`・`**update step**`・`**learn step**`・`**agent step**`・`**lane**` で始まる見出しの行が 1 行ずつある。agent step の `_Avoid_` に `$agent_step` がある。足した 6 項の定義が、それぞれ 1〜2 文である |
| A6 | 160 の ADR の節（U6） | 160 の和文の新設 ADR 節に、下の枠の検索 1 が示す 11 本のリンクと、それぞれに対応する § 番号がある。「ADR 0022」の文字がリンクの中にある |
| A7 | 160 の図とコードマップ（U7） | 160 の和文で `DB[(Run-local SQLite cache)]` を含む行が 1 行、`DB[(optuna.db)]` を含む行が 1 行で、その行は `ODB[(optuna.db)]` である。Optuna 側の両方の辺が `ODB` に接続される。§3.1 に `EvalTracePanel`・`HeatMapPanel`・`Conv2dPanel`・`ErrorDialog`・`gui.hpp` の文字がある。§7.1 の FPS の選択肢に `Config` がある |
| A8 | trace の行の属性（U8） | 140 の和文の trace チャネルの節に、trace の行が `timestamp` を持つと書く文が無い。`git diff -- docs/adr` が空 |
| A9 | 140 の ADR の節（U9） | 140 の和文の新設 ADR 節に、下の枠の検索 2 が示す 11 本のリンクと、それぞれに対応する § 番号がある |
| A10 | 140 の節と図（U10） | `git grep -n "^### 6.x" -- docs/design/140_observability.jp.md` が 0 行。`### 6.1`〜`### 6.5` の見出しが 1 行ずつある。§10 の下に `env_step` の見出しが無い。140 の和文に `ImageObserver` の文字が無い。§4 の図で、SessionEnd と trace の Observer が対応する通知に接続され、列挙した 5 種の画像 Observer は `ImageProviderFactory` と対応する通知に接続される。§5.1 の図で SessionEnd から `LogScalar`、EpisodeEnd から `LogTrace` への経路を確かめる。§3 に `metrics.hpp`・`image.hpp`・`heat_map.hpp`・`gui.hpp` の文字がある |
| A11 | 英訳（U11） | 和文を直した 5 つの設計文書（010 は直さない）の英訳の 1 行目の `translated-from` の blob が、直したあとの和文の `git hash-object` と一致し、`progress:done` である |
| A12 | リンク | [memo README](README.md) の「リンクの注意」のリンク切れ検査を `docs` に回し、`BROKEN` の行のうち、本書で直したファイルのものが 0 行 |
| A13 | 避ける語 | 直したファイルの差分の足した行に、[AGENTS.md](../../AGENTS.md) の避ける語の表で「検索」が「する」の語が 0 件（外部の固有名・コードの中は数えない） |
| A14 | 課題 | 題材の課題 10 件の事象を 1 件ずつ今の文書と照らし、残りが無いことを実装計画に書く |

A6・A9 の検索（縦棒を含むので表の外に書く）:

```text
検索 1: git grep -n -o -E "\.\./adr/00(16|21|22|45|56|57|66|68|69|81|82)-" -- docs/design/160_applications_and_tools.jp.md
検索 2: git grep -n -o -E "\.\./adr/00(28|29|34|37|43|47|55|60|70|71|72)-" -- docs/design/140_observability.jp.md
```

## 8. 成功の測り方

§1.5 のとおり。

## 9. issue の対応表

| issue | 題 | area | 根拠の位置 | 直す | 受入 |
|---|---|---|---|---|---|
| [issue-20261123-75512f](../management/issues/issue-20261123-75512f.md) | CONTEXT.md の Train Actor network snapshot と評価セッションが clone_model の条件を書かない | 324_dqn、321_agent_core | [CONTEXT.md](../../CONTEXT.md) 14〜16・443〜445 行、[default_dqn_agent.cpp](../../src/algo/dqn/src/default_dqn_agent.cpp) 946 行 | U1 | A1 |
| [issue-20261123-89526b](../management/issues/issue-20261123-89526b.md) | CONTEXT.md の trace の _Avoid_ が RFD0144 を「144」とだけ書く | 317_metrics | CONTEXT.md 586 行 | U2 | A2 |
| [issue-20261123-ff1a3f](../management/issues/issue-20261123-ff1a3f.md) | CONTEXT.md の target return が ADR 0065・0073 を反映しない | 318_rl_core | CONTEXT.md 42〜43 行 | U3 | A3 |
| [issue-20261123-d367c3](../management/issues/issue-20261123-d367c3.md) | Agent checkpoint・checkpoint 引き継ぎの説明が保存の中身と合わず、212 が避ける語「再開」を使う | 324_dqn、321_agent_core | CONTEXT.md 518〜524 行、[212](../design/212_rnd_lifelong_novelty.jp.md) 406 行、[200](../design/200_dqn_agents.jp.md) 644・663 行、[020](../design/020_user_guide_run.jp.md) 534 行 | U4 | A4 |
| [issue-20261123-fb50b6](../management/issues/issue-20261123-fb50b6.md) | CONTEXT.md が step の軸・agent step・lane を用語として持たない | 318_rl_core、317_metrics | CONTEXT.md、[010 §3.3](../design/010_framework_overview.jp.md) | U5 | A5 |
| [issue-20261123-0fcb2a](../management/issues/issue-20261123-0fcb2a.md) | 設計 160 が本文で扱う ADR へのリンクを持たない | 110_runner_app、311_dev_tools | [160](../design/160_applications_and_tools.jp.md) §9（446 行） | U6 | A6 |
| [issue-20261123-f3cad3](../management/issues/issue-20261123-f3cad3.md) | 設計 160 の図の node 名の重なり、コードマップの欠け、FPS の選択肢 | 110_runner_app、120_metrics_viewer | 160 の 198・219 行、§3.1、§7.1 | U7 | A7 |
| [issue-20261125-195ad5](../management/issues/issue-20261125-195ad5.md) | 設計 140 が trace の行は timestamp を持つと書く | 317_metrics | [140](../design/140_observability.jp.md) の trace チャネルの節（311 行から） | U8 | A8 |
| [issue-20261125-bdbcac](../management/issues/issue-20261125-bdbcac.md) | 設計 140 が本文で書く仕組みの ADR へのリンクを持たない | 317_metrics、318_rl_core | 140 §10（457 行） | U9 | A9 |
| [issue-20261125-ae4c45](../management/issues/issue-20261125-ae4c45.md) | 設計 140 の章立てと図が今の形に合わない | 317_metrics、318_rl_core | 140 の 264・311・335・428・469 行、§4・§5.1 の図、§3 | U10 | A10 |

## 10. グリルの材料

開発者の発言（2026-11-27。決定として扱わない）:

1. 「Codexにリセット権2つの期限が近い。とりあえずCodexに作業振って今日はねたい。仕様調整がほぼ要らない小さいのをいっぱい集めたPRDを作りたい。」
2. 「34件全部で一つのPRD？A~Dで4つのPRD？寝てる間に回したくて。バラPRDでも順番に実装するようにCodexさんに言えば大丈夫かな」
3. 置き場ごとに 3 本（文書・基盤とアプリ・DQN 系）に分ける案に「3本で」。

AI の案（同日）: 題材は、規模 2 以下で、待ちが無く、RFD の結び付きが無く、課題の対応方針が 1 つに決まっている課題から選んだ。方針が「どちらかを決める」もの（8eff2e・fa1be2・fe014b・3cb74a・29c3b2・ad7829・6bb71d・585e60・3ebab4・55c396・eb6135・b2a0b1・abb245）は外した。

## 11. 関連

- 設計 [010](../design/010_framework_overview.jp.md)・[020](../design/020_user_guide_run.jp.md)・[120](../design/120_environments.jp.md)・[140](../design/140_observability.jp.md)・[160](../design/160_applications_and_tools.jp.md)・[200](../design/200_dqn_agents.jp.md)・[212](../design/212_rnd_lifelong_novelty.jp.md)
- [RFD0206](0206_core_and_app_small_fixes_10rfd.md)・[RFD0207](0207_dqn_small_fixes_and_refactors_10rfd.md)・[RFD0081](0081_evalpanel_episode_trace_10rfd.md)・[RFD0144](0144_episode_forensics_10rfd.md)
