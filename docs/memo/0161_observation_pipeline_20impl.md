# RFD0161 強制行動と Perception の実装メモ

## 概要

[RFD0161](0161_observation_pipeline_10rfd.md) に沿い、強制行動、共通基盤、Perception の順で実装する。各段階の境界で全 Debug テストを実行し、同じ seed の学習 Run を比較する。

## 主な変更

1. 強制行動。基盤に lane 単位の `ForcedActions` と `MakeActionOptions::forced_actions` を置く。Eval Runner は Reset 前に検証し、呼び出しの step 数と options を受ける。Actor は方策の選択と乱数の消費を済ませてから強制を反映する。DQN は確定した行動から Q ヒントと診断を得る。TempoRL の強制 lane はその step を新しい判断とし、次の step で判断し直す。MuZero と ImageCls も行動を確定する。Runner とパネル、テストを同じ経路へ移し、旧い差し替えと専用 aux を削除する。
2. 共通基盤。`TensorSpec` の `scale`（既定 `Scaled`）と `has_stack_axis`（既定 false）を追加し、JSON に出す。LunarLander と CartPole の vector を `Unscaled` と宣言する。DefaultDQN の現行 spec 生成は既に足す軸に属性を立てる。`ForwardCapture` を `Forwarder` と同じ公開ヘッダへ移し、Network で既存の forward と capture の挙動を保つ。`SpecForwarder` は変換後の spec を返す拡張の口とする。
3. Perception。基盤の `ForwarderChain` に `FrameStack`・`Normalize`・`DropKeys` を登録し、名前から seed を導く。1 枚でも stack 軸を出し、保存用の stack の設定を Learner へ渡す。Actor は lane 履歴を持つ実行器で、Sample は束から、可視化は 1 枚から Perception を得る。push は統計更新用 Forwarder で得た Frame を ReplayBuffer に渡す。並びと ReplayBuffer の両方に同じ Frame の spec を渡す。正規化は `Unscaled` のキーにだけ行い、既存の正規化を利用して各 Frame に同じ統計を当てる。Q Network の入力 spec、並びの図と `obs_norm.*` 診断を並びから得る。
4. NN と利用側の移行。先頭 module の stack 軸の宣言と WARN、`StackMerge` の次元検査、離散 Grid の属性判定を実装する。出荷設定、現行テスト、17 構成の設定基準を §7.5 と §7.8 に沿って移す。既存の設定基準は実際に解決した設定から再作成して差分を点検する。日本語設計文書 110・130・140・150・200・220、新設の日本語設計文書 135、設計文書の一覧、対象 ADR の注記、機能カテゴリと wiki を反映する。既存の英語版 110・130・140・200 は契約の変わる箇所を更新する。
5. 追加の検証条件。開発者の指示に合わせ、全環境の同一性の比較で両側に決定論プロファイルを明示する。段階ごとに保管した Runner を使い、GridMaze を取り直す。

### 廃止・移行した名前

| 旧 | 新 |
| --- | --- |
| `BatchActionInfo::WithAction`、各具象の `WithAction` | `MakeActionOptions::forced_actions` を入力にして Actor が行動を確定 |
| `EvalRunner::DoStep(int64_t)` | `EvalRunner::DoStep` に `StepCounts` と `MakeActionOptions` を渡す |
| aux の `munchausen_terms` | Actor が確定した行動の Munchausen 項を Q ヒントへ直接入れる |
| `NetworkForwardCapture` | `anet/forwarder.hpp` の `ForwardCapture` |
| `StuckerConfig`、`DefaultDQNAgent.stucker.use_stacker` | `ForwarderChain` の `FrameStack`。省略時は1枚を重ねる |
| `DefaultDQNAgent.stucker.stack_count`、`DefaultDQNAgent.stucker.stack_keys` | `DefaultDQNAgent.perception.block.[stack].count`、`DefaultDQNAgent.perception.block.[stack].keys.mode` と `keys.listed.names` |
| `DefaultDQNAgent.obs_norm.pass_through`、`DefaultDQNAgent.obs_norm.use_dynamic_scaling` | `DefaultDQNAgent.perception.block.[norm].mode` の `none`、`constant`、`running` |
| `DefaultDQNAgent.obs_norm.constant_mean`、`DefaultDQNAgent.obs_norm.constant_std` | `DefaultDQNAgent.perception.block.[norm].constant.mean`、`DefaultDQNAgent.perception.block.[norm].constant.std` |
| `DefaultDQNAgent.obs_norm.use_clipping`、`DefaultDQNAgent.obs_norm.clip_range` | `DefaultDQNAgent.perception.block.[norm].constant` と `DefaultDQNAgent.perception.block.[norm].running` の各同名フィールド |
| `DefaultDQNAgent.obs_norm.use_centering`、`DefaultDQNAgent.obs_norm.epsilon`、`DefaultDQNAgent.obs_norm.use_robust_update`、`DefaultDQNAgent.obs_norm.robust_warmup_count`、`DefaultDQNAgent.obs_norm.robust_std_threshold`、`DefaultDQNAgent.obs_norm.post_process_type`、`DefaultDQNAgent.obs_norm.post_process_threshold` | `DefaultDQNAgent.perception.block.[norm].running` の各同名フィールド |
| aux の `raw_obs`、`norm_obs` | 削除。Network 入力は `nn_trace` または Perception の出力で確かめる |

旧スタック設定の名前を使う検証は 2 箇所ある。`src/core/base/testdata/rfd0072/perception_migration.py` は過去の固定入力を移行し、`src/core/base/src/forwarder_test.cpp` は旧キーが未読になることを受入 29 で確かめる。受入 18 の検索では両行を表示して照合する。

利用側のプロファイルや `A1`・`A2` が設定を上書きする箇所も、同じ相対キーへ移す。メトリクスの `obs_norm.*` と RND 自身の観測正規化のキーは引き続き使う。

## テスト

| 順 | 観測する入口と結果 |
| --- | --- |
| 1 | `EvalRunner::DoStep` へ 2 lane のうち 1 lane の強制を渡し、同じ step で Env が受け取る行動と Runner の lane 別の強制フラグを観測する。最初の失敗を確認して成立させる |
| 2 | `ForcedActions` の型・範囲・連続 Env と Reset 前の失敗、マスクで外した lane の挙動を、公開の Runner・Actor から 1 条件ずつ通す |
| 3 | 実際の DQN Actor に強制を渡し、未指定 lane、Q ヒントと Munchausen、QR・IQN の診断、TempoRL の継続中の割り込みを順に確かめる。MuZero と ImageCls の差し替えを各 Actor の呼び出しから確かめる |
| 4 | `TensorSpec` の JSON 出力と spec 検証、Env が公開する尺度、Network を `Forwarder` として使う呼び出しと capture の一致を順に通す |
| 5 | 1 枚の Frame から実際の並びで Perception と spec を得るテストから始める。1 枚・4 枚の 4 経路、Replay での復元と境界、3 モードの正規化と診断、DropKeys、設定検査、module の WARN と形、図の順に観測を足す |

各行を観測できる挙動に分け、テスト 1 件の失敗を確認し、その挙動を成立させてから次へ進む。通過したテストを保ちながら整理する。タグは `[forced_action]`・`[forwarder]`・`[perception]` を使う。出荷構成の診断は `[shipped_config]` で確認する。

## 検証

- 各段階で Debug をビルドし、`out/test/Debug` 以下の全 `*-test.exe` を引数なしで回す。各終了コードと失敗を記録する。関連タグでも個別に実行する。RFD §9 の固定文字列を指定範囲で検索し、`git diff --check` を通す。
- 各段階の前後の Release Runner と出荷設定を取得する。段階ごとに別の出力先を使い、Run のプロファイル、ゲーム、seed、終了経験数、比較結果を記録する。強制行動は LunarLander と GridMaze、共通基盤と Perception はこれらに DropMerge と Atari を加える。学習 1,000 ステップを基準に、解決された lane 数・warm-up・replay credit と Runner の完了通知から終了経験数を求め、Run の設定とログで確認する。比較に必要なレコードが欠ける場合は RFD の上限で両側を再取得する。
- 全環境で比較する両側の起動時に `backend.$=backend.@deterministic` を指定する。Atari は `E1.game=qbert` も指定し、実効 `AtariEnv.game` を確認する。比較ごとに `backend.deterministic_algorithms` と `backend.cudnn_deterministic` の実効値を確認する。
- `src/core/base/testdata/rfd0097/compare_metrics.py` を追加除外オプションなしで実行する。比較の前側で TD 誤差と学習リターンの件数を確認し、`exact_match` が真であることを判定基準にする。Perception の Run で `dot/perception.structure` と `dot/perception.detail` の出力も確認する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 4'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 4'
git diff --check
```

## 前提

- RFD の再グリルとレビューは完了済み。先行の RFD0191 は `2573b18e` で取り込み済み。現行の ReplayBuffer の重ね直し、RND の加工と統計、正規化の保存と並行処理は RFD の境界に従う。
- 設定の既定は、並びを省略した場合に 1 枚の `FrameStack`、出荷設定の基準では `stack > norm` の 4 枚と素通し。設定で使う対象一覧と尺度属性は RFD §7.5 に従う。
- GridMaze の比較に差が出た場合と、軸の扱いを宣言できない先頭 module が出荷設定で見つかった場合は、原因を確認して RFD §12 の扱いに従う。RFD0142 が追加する Frame のキーの往復は同 RFD の検証で行う。

## 経過

- 2026-11-23。変更前の Release Runner をビルドした。Q*bert の出荷設定、seed 1、128 lane、456,320 経験で、起動から正常終了まで 163.94 秒。TD 誤差 1,000 件、学習リターン 1,068 件。起動の設定は出荷ファイルを読み、出力先・終了経験数・エラーダイアログ・終了時保存を実行用に指定した。設定とログは `.scratch/rfd0161-atari-20261123-131347/qbert`、集計は同じ親フォルダの `baseline-counts.json`。実効 `run.$` は `run.@v5_iqn_impala_x2 > run.@a5 > run.@a5_apex > run.@va_base > run.@hard125 > run.@munch > run.@rfit > run.@a5_metrics > run.@cap2m > run.@eval2ch_r1 > run.@to_100m`。開発者は Atari の学習 Run による比較を選択した。
- 2026-11-23。強制行動の変更前の基準 Run は、どちらも出荷設定、seed 1、256 lane、pipeline。GridMaze は `run.$` の指定なし、131,072 経験、TD 誤差 500 件、学習リターン 499 件（`.scratch/rfd0161-before-forced-20261123-132037/grid`）。LunarLander は `run.@munch > run.@skip`、40,704 経験、TD 誤差 125 件、学習リターン 90 件（`.scratch/rfd0161-before-forced-lunar-20261123-132505/lunar`）。LunarLander の最初の試行では終了経験数の見積もりが多かったため、解決された replay batch 256、replay ratio 8 に合わせて取り直した。
- 2026-11-23。Eval Runner の 2 lane のうち 1 lane を強制するテストを先に追加した。変更前の Debug ビルドは `forced_actions`・`ForcedActions`・step 数と options を受ける `DoStep` が存在せずに失敗した。
- 2026-11-23。§7.9 の構築の確かめとして、現行 GridMaze の vector を 4 枚重ね、branch を `MLP_FC1`（Linear）だけにして起動した。`Run initialized` まで進んだ。最初の行動選択では追加の軸が Head に残って shape エラーになった。構築時の WARN を確かめるテストでは、Linear に続く処理で Head に合う形を作る。ログは `.scratch/rfd0161-linear-preflight-20261123-132841/linear/linear.log`。
- 2026-11-23。構築とリンクの失敗を確認しながら lane 番号とマスクの指定を追加した。マスクで外した lane の行動は方策どおりになり、範囲外・型・長さ・連続行動は Env の Reset 前に拒否した。未定義の Tensor の診断が libtorch 内で失敗したため、未定義と明記する診断へ直した。`anet-core-base-test.exe "[forced_action]"` は 2 件・20 断定すべて通過した。ビルドログは `.scratch/rfd0161-forced-rejection-green.log`。
- 2026-11-23。Core を引数なしで走らせ、666 件通過、1 件スキップ、33,796 断定通過を確認した（`.scratch/rfd0161-core-early-suite.log`）。DQN は強制後のヒントと未指定 lane の行動、次の呼び出しの乱数の一致を確認した。TempoRL は継続中の強制、判断と長さの記録、次の判断を確認した。MuZero は探索結果、ImageCls は予測確率を保ったまま、各 Actor が強制行動を返した。その後、Runner とパネルの呼び出しを移し、旧い差し替えを削除した。受入 1・2・6 の検索は 0 行。
- 2026-11-23。基準 Run のあとで `metrics_scalar.txt` に別作業のタグが 2 件追加されたため、比較条件をそろえるために変更前のコミット `7760382e` から出荷設定を取り出した。旧 Runner の実行ファイルとともに `.scratch/rfd0161-stage-a-baseline-7760382e` に保管した。
- 2026-11-23。移行後の Debug 全体と Release Runner のビルドが通過した。引数なしの DQN 全実行で 2 件のテストが失敗した。置き換えたテストが余分な行動選択を実行したため、セッション集計を取る Actor と強制を確認する Actor を分離した。CUDA の確認には、用いたテスト用 Network の前提に合わせて BF16 autocast を指定した。Core の全実行ではログ収集のテストが 1 件失敗し、再実行で確認する。最初のログは `.scratch/rfd0161-forced-verify-20261123-141024`。
- 2026-11-23。強制行動の変更前後で GridMaze の全 29,337 点が完全一致した。LunarLander は、先に取った 40,704 経験の基準 Run に並行作業のメトリクスが入っていたため、保管した旧 Runner と固定の出荷設定で取り直した。前後の実効設定の差は出力先だけになったが、13,825 対 13,821 点で不一致だった。出荷設定では `backend.deterministic_algorithms=false`、`backend.cudnn_deterministic=false`。同じ実行体の繰り返しを診断する。`grid-comparison.json` は `.scratch/rfd0161-after-forced-20261123-143402`、条件をそろえ直した LunarLander の `comparison.json` は `.scratch/rfd0161-lunar-before-fixed-config-20261123-143706` にある。
- 2026-11-23。LunarLander の出荷設定を seed 1、40,704 経験で繰り返して、旧 Runner 同士で 13,825 対 13,803 点、新 Runner 同士で 13,821 対 13,819 点の不一致を確認した。まったく同じ実行体でも学習リターンと記録件数が変わった（`.scratch/rfd0161-lunar-repeat-20261123-144010/{before,after}-comparison.json`）。両側で指定 `backend.$=backend.@deterministic` を足すと、実効設定の `deterministic_algorithms` と `cudnn_deterministic` がともに `true` になり、TD 誤差と学習リターンを含む全 13,823 点が一致した（`.scratch/rfd0161-lunar-deterministic-probe-20261123-144252/comparison.json`）。
- 2026-11-23。Debug のテスト実行体 9 本を引数なしで走らせた。DQN は修正後の再実行で 379 件・161,400 断定通過。Core は 2 回目にもログ収集と workspace 履歴のテストで失敗したが、失敗した 3 件は個別実行で通過し、3 回目の全件実行は 666 件通過・1 件スキップ・33,796 断定通過。MuZero、ImageCls、Atari・DropMerge・GridMaze・ImageCls・LunarLander の Env の全実行も失敗 0 件。`git diff --check` は空白のエラーなし。ログは `.scratch/rfd0161-forced-verify-20261123-141024`。
- 2026-11-23。比較した GridMaze の出荷 Run と LunarLander の決定的な診断 Run は、どちらも `trace` の記録が 0 件だった。設定で記録された scalar は全件を比較した。旧 Runner の TD 誤差と学習リターンは、GridMaze がそれぞれ 500 件と 499 件、LunarLander の診断 Run が 125 件と 92 件。
- 2026-11-23。出荷時のバックエンドのままでは、段階 A の等価性は受入未達。出荷ファイルと Run プロファイルを使い、非決定的なバックエンドを選ぶ環境に限って比較の前後で決定的な設定を実行時に指定してよいか、開発者の判断を待つ。決まったらこの計画の段階 A の受入確認から再開する。段階 C・B は未着手。
- 2026-11-23。開発者は LunarLander・DropMerge・Atari の比較時だけ、両側を決定的なバックエンドで動かす案を選んだ。Atari は Q*bert の学習 Run で比べる。RFD の G2・D35・§9 と実装計画の検証手順に反映した。
- 2026-11-23。RFD のレビューを受け、§9 に Q*bert の指定と実効値の確認を足し、D35・U4 に学習 Run の裁定を反映した。P141 の実測の対象を §9 で比較するメトリクスに限った。再レビューは OK。受入 10 は、GridMaze の両側で出荷実効値 `deterministic_algorithms=true`・`cudnn_deterministic=false`、LunarLander の両側で比較用の実効値 `true`・`true` を確認し、既存の比較結果で成立した。段階 C の前の Runner と設定を保存し、DropMerge と Q*bert の基準 Run を開始した。
- 2026-11-23。段階 C の `TensorSpec` の JSON テストを追加した。属性が無い旧実装で `scale` の存在を問う断定が失敗し、新しい属性と JSON の出力を追加した。
- 2026-11-23。`[forwarder]` の JSON テストで 6 断定が通過した。続けて尺度の検査を追加する前に、整数の `Unscaled` が通ってしまい、離散キーでは尺度を示さずに失敗することを確認した。
- 2026-11-23。段階 C の変更前の Q*bert は、seed 1、128 lane、456,320 経験、174.68 秒、TD 誤差 1,000 件、学習リターン 1,065 件（`.scratch/rfd0161-stage-c-before-20261123-150622/qbert`）。解決された Run プロファイルは変更前の計測と同じ。DropMerge は有効な `run.$` なし、seed 1、256 lane、282,112 経験、185.44 秒、TD 誤差 1,000 件、学習リターン 878 件（`.scratch/rfd0161-stage-c-before-20261123-151335/drop`）。最初は 281,600 経験で 998 件だったので、512 経験を足した。両環境とも `deterministic_algorithms=true`、`cudnn_deterministic=true`、終了コード 0。
- 2026-11-23。RFD のレビュー原文は `.scratch/rfd0161/review-20261123-1509-rfd.md` と `.scratch/rfd0161/review-20261123-1512-rfd.md` に保存した。
- 2026-11-23。`[forwarder]` で `StateSpec` の整数または離散の `Unscaled` の拒否を確かめた。次に、LunarLander と CartPole が未宣言のままでは尺度のテストが失敗し、`Unscaled` を宣言すると通過した。GridMaze・DropMerge・Atari は `Scaled` のテストが通過した。CartPole は既存の Env テストと同様に Debug のテスト実行体を作った。
- 2026-11-23。Network を `Forwarder` として呼ぶテストは、口と `ForwardCapture` が未定義のためコンパイルに失敗した。`anet/forwarder.hpp` に共通の口を置き、既存 capture の処理を移し、18 ファイルの呼び出しを移行した。固定文字列 `NetworkForwardCapture` の `src`・`apps` での検索は 0 行。Debug 全体と Release Runner を別のログに出してビルド中。
- 2026-11-23。段階 C の Debug 全体のビルドは通過した。`[forwarder],[capture]` の 7 件・74 断定で、尺度の検査、Network を `Forwarder` 経由で呼び出した出力と capture、既存の要求鍵の検査を確認した。全件テストの出力先は `.scratch/rfd0161-stage-c-debug-20261123-153146`、Release ビルドの出力先は `.scratch/rfd0161-stage-c-release-20261123-153322.log`。日本語の設計文書に Forwarder と spec の契約を記載し、機能カテゴリが `314_tensor` になることを確認した。
- 2026-11-23。段階 C の Release Runner をビルドした。4 環境とも前後の実効設定は出力先だけが異なり、比較対象のメトリクスが完全一致した。GridMaze 29,337 点（50.09 秒）、LunarLander 13,823 点（64.61 秒）、Q*bert 70,654 点（164.61 秒、うち trace は 1,302 点）、DropMerge 99,390 点（157.83 秒）。前後の backend の実効値は GridMaze が `deterministic_algorithms=true`・`cudnn_deterministic=false`、ほかの 3 環境が `true`・`true`。比較 JSON と Run は `.scratch/rfd0161-stage-c-after-20261123-155302`。この Runner を `apps/runner/bin/Release/AnetRLRunner-rfd0161-stage-c.exe` に保管した（SHA-256 `62251A77BF4DAC464735DDC5B4A3D6365EEE23C12B9BFD64D407EB5DE9994AE1`）。
- 2026-11-23。段階 C の Debug 全件実行で DQN 379 件を含む Core 以外の 9 本がすべて通過した。Core で段階 A でも観測した `Logger level methods preserve wx levels and prefix` が失敗した。個別実行は 7 断定が通過し、Core 全件を再確認している。
- 2026-11-23。Core の次の全件実行は別のログ収集テスト `Run notifications occur once per source and target across threads` が失敗した。学習比較の終了後に行った Core の全件実行は 669 件通過・1 件スキップ・33,809 断定通過（`anet-core-base-test-final.log`）。段階 C の受入 11〜15・17 を確認し、段階 B の作業を始めた。
- 2026-11-23。段階 B の最初の `[perception]` は、`ForwarderChain` が無い状態でコンパイルに失敗した。1 枚の Frame を変換する並びを実装し、並びの分割は NN と同じ関数を使うように移した。並びを省略・枚数を省略・4 枚を指定した各場合で、形・dtype・値・spec の 18 断定が通過した（`.scratch/rfd0161-b-perception-stack-{red,green}.log`）。Actor の履歴、Sample、Frame の保存と一部キーだけの stack を比べるテストでは、実行器などが未実装のためコンパイルに失敗した（`.scratch/rfd0161-b-frame-paths-red.log`）。

- 2026-11-23。段階 B では、Actor の履歴・Replay の 1 枚／4 枚・キーの取り除き・統計更新と正規化の形を順にテストした。各テストは未実装で失敗し、実装後に通過した。ログは `.scratch/rfd0161-b-frame-paths-{red,green}.log`、`rfd0161-b-drop-keys-{red,green}.log`、`rfd0161-b-normalize-{red,green}.log`。最後のログは 18 件・188 断定。統計更新では元の Tensor を保持した。
- 2026-11-23。DefaultDQN の行動選択・経験受入れ・Sample・可視化を Perception につなぎ、出荷設定 4 ファイルを移行した。NN のテストは実装前に 3 件・4 断定失敗し、4 次元の StackMerge、離散 Grid の属性と形の不整合、先頭 Linear の警告の不足を捉えた（`.scratch/rfd0161-b-nn-red.log`）。実装後は ReplayBuffer の往復テストも通過し、1 枚と 4 枚の両方で episode 先頭・打ち切り・短縮 N-step・skip 窓を Actor の観測と照合した。最初の `[shipped_config]` は Forwarder の `block` を既存の NN カタログ検査が拾い、1 件・7 断定失敗した。テスト側で対象のカタログを区別するよう直した。
- 2026-11-23。固定 17 構成はコミット `107a62c8` を保ち、Resolver が解決した結果から Perception に移したキーと値を採取した。17 件すべてで、ほかの値、元の基準のほかのキーの表示順、解決記録と上書きの期待値を保存した。入力の宣言から作る独立の期待値とも一致した。最初の解決結果は `.scratch/rfd0072-differential/profiled-b8f918fe6ff3/actual`。
- 2026-11-23。段階 B の初回学習比較は 4 環境とも正常終了し、比較対象の全 213,204 点が一致した。GridMaze 29,337 点（51.91 秒）、LunarLander 13,823 点（73.10 秒）、Q*bert 70,654 点（190.83 秒、うち trace は 1,302 点）、DropMerge 99,390 点（175.73 秒）。ログ・比較 JSON は `.scratch/rfd0161-stage-b-after-20261123-170611`。自己点検で、FrameStack の `all` から ReplayBuffer に空の一覧を返していた点を具体的なキーの列挙へ直し、詳細図に各段の dtype・尺度・stack 軸と、読んだ設定値を記録するようにした。修正後の実行体で学習比較を取り直す。
- 2026-11-23。DQN の初回全件は 380 件中 39 件失敗した。長さ 1 の軸が残った観測を Head へ渡していた共通 fixture と個別設定に Flatten を加え、可視化へ渡す IQN の補助入力は FrameStack の対象外にした。Rainbow のテストは廃止した aux から行動出力の検査へ移した。対象の再実行は 84 件・22,517 断定すべて通過（`.scratch/rfd0161-debug-dqn-final-20261123-172609/focused.log`）。Runner の出荷構成確認では、固定在庫のうち、現行 Atari から消えた Nature と Classic IQN の Run 名を、テスト入力で現行の Env・Agent・NN プロファイルから再構成した。
- 2026-11-23。段階 B の修正版（Release Runner の SHA-256 `57712597F738F9BD87909336566D68F5FCA64E59778EE03E0CFC2E9AD048A456`）で 4 環境とも完全一致した。LunarLander は 13,823 点（67.40 秒）、Q*bert は trace 1,302 点を含む 70,654 点（166.33 秒）、DropMerge は 99,390 点（193.96 秒）。比較用の実効値はこの 3 環境の両側とも `deterministic_algorithms=true`・`cudnn_deterministic=true`。実効設定の差は Perception の移行と出力先だけ。Q*bert の実効ゲーム名は `qbert`、Run プロファイルは段階 C と同じ。各 Run に `dot/perception.structure.dot` と `dot/perception.detail.dot` があり、後者で形・dtype・尺度・stack 軸・設定値を確認した。比較 JSON と Run は `.scratch/rfd0161-stage-b-after-20261123-172151`。
- 2026-11-23。開発者の指示により、GridMaze も比較前後で `backend.$=backend.@deterministic` を明示することにした。RFD の D35 と §9 を更新し、独立の再レビューは OK（`.scratch/rfd0161/review-20261123-1734-rfd.md`）。4 版すべてで実効値 `deterministic_algorithms=true`・`cudnn_deterministic=true` を確認し、各 131,072 経験で正常終了した。段階 A・C・B の各前後で TD 誤差 500 件と学習リターン 499 件を含む比較対象全 29,337 点が一致した。結果は `.scratch/rfd0161-grid-explicit-deterministic-20261123-173154`。GridMaze の出荷時のバックエンドによる比較は先行確認の記録とする。
- 2026-11-23。修正版の Core の関連 36 件・1,302 断定と、固定 17 構成の解決結果 157 断定が通過した（`.scratch/rfd0161-debug-core-gui-final-20261123-173003`）。先行の引数なしの Core は背景の評価ログがログ捕捉へ混ざり 2 件失敗し、ほかの環境・Agent の 8 本はすべて通過した（`.scratch/rfd0161-debug-core-and-env-20261123-172609`）。出荷設定の代入規則の検査はソース移転前の探索先を直して実行した。移行した 4 ファイルで指摘 0 件、補助の `metrics_viewer.py` にある代入 4 件も設定として拾った。検査器の回帰テスト 16 件は通過した。
- 2026-11-23。修正版で引数なしの Core を再実行し、680 件通過・1 件スキップ・34,043 断定通過（`.scratch/rfd0161-debug-core-gui-final-20261123-173003/all-core.log`）。`[shipped_config]` では DropMerge の QR51 プロファイルが IQN 専用の `iqn_fusion` を出力先にして止まった。段階 C の実行体と変更前の設定でも同じ失敗を再現した（`.scratch/rfd0161-drop-qr-before-20261123-173835/qr/qr.log`）。`run.@qr51_control` で `skip_context` の参照先を QR が組む `main_feature` に設定した。
- 2026-11-23。Runner の `[shipped_config]` は固定在庫 17 件のうち DefaultDQN の 13 構成で Agent を作り、stack 軸を扱う宣言に関する WARN が 0 件であることを確認した。76 断定すべて通過（`.scratch/rfd0161-shipped-config-final.log`）。出荷 Network で軸の処理を先頭に足した定義は 0 件。既存のテスト用 Network では 9 定義に Flatten を足した。旧契約の検索 S1〜S12、現行の日英設計文書と用語集の検索、既存ファイル 102 件の BOM、レビュー対象の LF、`git diff --check` が通過した。
- 2026-11-23。引数なしの DQN 381 件・161,401 断定がすべて通過した（`.scratch/rfd0161-debug-dqn-final-20261123-172609/all.log`）。段階 B の Debug テスト実行体 10 本は失敗 0 件、Core にスキップ 1 件。QR51 を直した最終設定の DropMerge 既定 Run は、決定論プロファイル、seed 1、282,112 経験で 163.83 秒、終了コード 0。TD 誤差 1,000 件と学習リターン 878 件を含む比較対象全 99,390 点が段階 C と一致した（`.scratch/rfd0161-drop-default-final-20261123-173947/drop-comparison.json`）。
- 2026-11-23。独立レビューで、Frame を保存する経路の `obs_norm.clip_ratio` が更新されないことを指摘された。段階 C までの DefaultDQN は経験の両側を `NormalizeAndUpdateStats` に通し、値の返り値を使わず、診断を更新していた。定数・統計つきの2モードについて、既存部品と同じクリップ率を保持し、保存する Tensor と読み出し時の診断を変えないことを回帰テストへ追加した。
- 2026-11-23。初回の実装レビューは差し戻しだった（`.scratch/rfd0161/review-20261123-1755-impl.md`）。クリップ率のテストは実装前に4断定が失敗した（`.scratch/rfd0161-perception-clip-red.log`）。経験の受入れで従来の `NormalizeAndUpdateStats` を使って診断を更新し、保存するには元の Tensor を返すよう直すと、両モードの10断定が通った（`.scratch/rfd0161-perception-clip-green.log`）。さらに保存前後の値そのものを照合する断定を加えた。旧名・新名の表、英語版4件と日本語版の翻訳元ハッシュ、`type` の読取、テストの TensorDict キー、`ForwardCapture` の宣言も直した。
- 2026-11-23。同じ独立レビュー担当に修正版を出し、計画・実装とも OK（`.scratch/rfd0161/review-20261123-1807-impl.md`）。再レビュー中に試験のクリップしきい値を定数で共有し、`NamedBlockConfig` のコンストラクタの宣言をフィールドの前に置いた。メンバーの初期化順と処理は同じである。
- 2026-11-23。クリップ率と設定読取を直した Release Runner（SHA-256 `F90D117FD609298DB586E428445BC6B0FB6182ACF6ABFA8A9EF37813654562BE`）で段階 B の比較を取り直した。出荷ファイルと seed 1 を使い、4 環境の前後に `backend.$=backend.@deterministic` を指定した。設定の写しで `deterministic_algorithms=true` と `cudnn_deterministic=true` を前後とも確認した。観測加工の設定移行以外の実効設定の差は出力先だけで、GridMaze では出力名も異なる。比較ツールに追加除外は渡さず、4 組すべて `exact_match=true`。全 Run が正常終了し、各 Run に `dot/perception.structure.dot` と `dot/perception.detail.dot` がある。比較 JSON と修正後の Run は `.scratch/rfd0161-stage-b-reviewed-20261123-180614`。

| 環境 | Run プロファイル | 終了経験数 | TD 誤差 | 学習リターン | trace | 一致した点数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GridMaze | 出荷既定 | 131,072 | 500 | 499 | 0 | 29,337 |
| LunarLander | `run.@munch > run.@skip` | 40,704 | 125 | 92 | 0 | 13,823 |
| Atari（Q*bert） | 冒頭の基準 Run と同じ並び。実効 `AtariEnv.game=qbert` | 456,320 | 1,000 | 1,065 | 1,302 | 70,654 |
| DropMerge | 出荷既定 | 282,112 | 1,000 | 878 | 0 | 99,390 |
| 計 | | | | | | 213,204 |

- 2026-11-23。再レビュー後、Debug のテスト実行体 10 本を引数なしで確認し、失敗 0 件。Core は 681 件通過・1 件スキップ・34,055 断定通過、DQN は 381 件・161,401 断定通過、ほか 8 本はすべて通過した。観測加工の関連 37 件・1,314 断定、RFD0072 の固定 17 構成・157 断定、Runner の DefaultDQN 出荷 13 構成・76 断定も通過した。ログは `.scratch/rfd0161-reviewed-core-gui-20261123-181039`、`.scratch/rfd0161-review-fixed-dqn-all.log`、`.scratch/rfd0161-review-fixed-debug-env-20261123-180613` にある。
- 2026-11-23。最終ソースで Debug 全体と Release Runner のビルドが成功した（`.scratch/rfd0161-debug-relink-20261123-181636.log`、`.scratch/rfd0161-release-relink-20261123-181516.log`）。宣言順の整理を含む Release Runner の SHA-256 は `7B31CB6A586513FD2DD3DF72F2EC574BB8AC1A4AA51883C43611C51106535F1E`。当時の固定文字列検索 S1〜S12 は 0 件だったが、S5 は文字列を分けて書いたため、固定入力の移行と受入 29 が使う旧名を検出できていなかった。今回の再検査では、両方を S5 の結果に含める（§「廃止・移行した名前」）。

- 2026-11-23。追加レビューで、S5 が見逃していた検証用途 2 行を訂正した。検索では未読キーのテストと RFD0072 の移行資材に各 1 行、製品コード・出荷設定に 0 行。固定した 17 構成の比較は 157 断定が通過した。C++ の定数と宣言順、観測加工と強制行動のコメントの修正後、Debug 全体と Release Runner のビルドが通過した。

- 2026-11-23。JSON の shape・dtype・scale・stack_axis の欄名は、出力と図、テストで同じ契約を使うため、既存の `TensorSpec` に置いた。テストのキーと設定の列挙値は宣言した定数を参照する。S1〜S12 は受入 18 の検証 2 行を含めて期待どおりだった（`.scratch/rfd0161/retired-identifiers-cc-final.txt`）。
- 2026-11-23。最終ソースで Debug 全体、GUI テスト、Release Runner をビルドした。Debug テスト実行体 10 本を引数なしで実行し、計 1,228 件・204,156 断定通過、失敗 0 件。CUDA が使えない場合の Core のテスト 1 件がスキップされた。ImageCls 環境テストは出力をファイルに直接送るとダイアログ待ちになったため、PowerShell で出力を受けて取り直し、20 件・197 断定が通過した。関連タグは Core 17 件・279 断定、DQN 4 件・74 断定、ImageCls Agent 1 件・4 断定、MuZero 1 件・16 断定が通過。固定 17 構成（157 断定）と出荷 DefaultDQN 13 構成（76 断定）も通過した（`.scratch/rfd0161/verification-cc-final/`）。
- 2026-11-23。最終 Release Runner（SHA-256 `6C9EE014CF0AF28C5C6BAACFCEC4627C7FC7EB9B365C3C999CEC9DEE035383B0`）と段階 B の受入済み Run（`.scratch/rfd0161-stage-b-reviewed-20261123-180614/`）を比較した。両側に `backend.$=backend.@deterministic` を指定し、各環境で seed 1、`deterministic_algorithms=true`、`cudnn_deterministic=true` を確認。GridMaze 29,337 点、LunarLander 13,823 点、Atari Q*bert 70,654 点、DropMerge 99,390 点の全 213,204 点が完全一致した。実効設定は出力先だけが異なり、4 環境で観測加工の概要図と詳細図もバイト一致した（`.scratch/rfd0161/identity-cc-final/`）。

## 完了報告

- 強制行動を `MakeActionOptions` に移し、指定 lane の行動と判断の記録を Actor が確定するようにした。DefaultDQN の観測加工と入力 spec を同じ Perception の定義へ統合し、出荷設定、テスト、設計文書を移行した。[設計文書](../design/135_forwarder_and_perception.jp.md)
- 段階 A・C・B の同一性は、全環境で前後とも決定論プロファイルを明示して確認した。段階 B のレビュー修正後も、4 環境で合計 213,204 点が完全一致した。出荷構成で stack 軸の宣言に関する WARN は 0 件だった。
- 独立レビューは差し戻し 1 回、修正後の計画・実装とも OK。再レビューでは経験の受入れでクリップ率を更新しつつ、保存する Frame の値と読み出し時の診断を維持することを再確認した。`issue-20261103-aa5758` と `issue-20261103-b754ff` は残件無しと判定した。
- RFD0161 の範囲で開発者の判断を待つ項目と、新規の起票候補はなし。Frame に直前行動を加えた場合の往復は RFD0142 で確かめる。

## 決まらなかったこと

- RFD の別レビューで、表題の暫定表記と、§9 の既存の説明に trace の lane が無いことが見つかった。比較ツールは lane も照合する。
- 出荷設定の監査ツールは、今回の移行対象 4 ファイルの指摘が 0 件だった。全体の監査では、既存の `metrics_viewer.py` の Python 代入 4 件を設定ファイルの代入として拾った。
