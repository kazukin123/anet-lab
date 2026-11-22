# PRD191 RND の設定、内的報酬と補助学習 実装メモ

## 概要

[PRD191](191_rnd_raw_error_and_recent_learning_10prd.md) の §6〜§9 に沿い、コードと設定を先に実装し、設計文書 212・200・150 の日本語版を続けて更新する。出荷既定は whiten、標準化、誤差の待ち 12,800 件、predictor の待ち 0 件、intake 学習（比率 0.25・32 件）。

## 主な変更

1. 推定器の構造図を、DQN が primary の図と同じ時機に記録する。RND がある場合に target と predictor の structure と detail が出る。
2. RND と補助学習の設定を PRD §8.1 に揃える。whiten と variance_feedback は既存の計算順を保ち、誤差の統計と predictor の待ちを別々に判定する。出荷設定、バッチ、テスト、変更履歴と PRD194 の期待を移行する。
3. none の倍率、診断用の累積統計、スケーラ後の 0 と Trace を実装する。checkpoint の既存書式を使う。
4. lane と時間で読む tensor の置き場を core/base に追加する。DefaultDQNAgent は rollout の完了時に学ぶか、次の rollout に配分する。intake の抽選と同じ名前付き乱数を使う。
5. コード・設定の検証を記録し、212 を PRD §8.7 の順で書き直す。200 と 150 の対応箇所を更新する。

## テスト

- 設定、Agent の生成と intake、推定器の公開操作、Run の dot と metrics、checkpoint の読込を観測する。
- 最初は、RND を有効にした Agent が初期化で 4 つの構造図を記録する経路を通す。失敗を確認し、推定器から図を取得して記録する最小の変更で通す。
- 続いて、現在の設定から移した待ちと clip、未選択モードの値の妥当性、none の採点、スケーラを通した分解報酬と Trace、生の観測の保存と読み出し、エポックのまとめ学習、次の rollout への配分、読込後の状態を順に検証する。挙動ごとにテストの失敗を確認してから実装し、通過後に必要な整理をする。

## 検証

- Debug ビルドと core/DQN の関連 suite、既定 suite、出荷設定のテストを実行する。VS の初期化は AGENTS.md の VsDevCmd.bat 経由で行う。
- Release Runner の前後に移行前後の同義な設定を与える。LunarLander、seed 1、決定的 CUDA、torch 1 thread、100,000 経験、評価・描画・画像出力を止める。無効、累積、EMA、variance_feedback、collect_errors 相当で時間と資源の 3 prefix を除く全 scalar を比べ、既定の旧 checkpoint を読む。
- 受入 2 の旧キー検索をコードと設定の終了時と設計文書の終了時に行う。PRD194 の監査を更新した期待値と照合する。
- Atari の 128 lane で intake、rollout 一括、rollout 分散を 3 巡以上、順を入れ替えて測る。一括の 16 step の所要時間は別の Run で記録する。数値に合否の閾値は設けない。
- 212 の設定表を PRD と agent.txt と突き合わせる。受入 18 の読みやすさは開発者が確認する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-base-test anet-algo-dqn-test --parallel 4'
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --target AnetRLRunner --parallel 2'
out\test\Debug\anet-core-base-test.exe '[prd191],[intrinsic],[shipped_config]'
out\test\Debug\anet-algo-dqn-test.exe '[prd191],[intrinsic],[shipped_config]'
.\.venv\Scripts\python.exe src/core/base/testdata/prd194/audit_config_messages.py --expected src/core/base/testdata/prd194/expected.jsonl
```

比較と計測の実行手順は [PRD191 検証資材](../../src/core/base/testdata/prd191/README.md) を参照する。

## 前提

- PRD に暫定の印はない。判断を求める未決は 0 件。用語は CONTEXT.md の intake と rollout（補助学習）、設定の分け方は ADR 0088 に従う。
- 現行の whiten と variance_feedback の forward、更新順、intake の credit と抽選、checkpoint の書式を引き継ぐ。none で読む誤差統計の書式は累積。rollout の中身と学習の進み具合は checkpoint に入れず、読み込み後に空から始める。
- Agent の内的報酬を合成する既存の経路で、採点前の待ちの判定を取得し、スケーラで統計を更新してから報酬の値を 0 にする。Trace は推定器の写しの判定を使う。
- 選んでいないモードの値も読んで検証する。lane と学習計画の組み合わせの検査は、学習する構成で選んだモードに当てる。
- 開発者が稼働中のバッチを止める。現用設定と 11・12・13 番バッチの RND 設定を同じ変更で移し、現用 RND の起動を新 Runner にそろえる。旧実行体と解決済み設定を明示している過去の比較経路は記録として扱う。
- PRD の範囲外は Atari での長期の学習効果、Replay とフレーム積層の置き場の移行、RND 外の設定整理、設計文書の英訳。

## 旧設定キーの移行

左列の前置きは `DefaultDQNAgent.intrinsic.`。移行理由は PRD191 D7〜D9 と ADR 0088。出荷設定とバッチの書き換えは [CHANGELOG](../../CHANGELOG.md) にも記録する。

| 旧キー | 新しい指定 |
| --- | --- |
| `rnd.statistics_mode` | `rnd.statistics.mode` |
| `rnd.obs_norm.*` | `rnd.statistics.whiten.obs_norm.*`。feedback の観測 clip は `rnd.statistics.variance_feedback.obs_norm.clip_range` |
| `rnd.multiplier_mode` | 選んだ方式の `use_clipping` に変換 |
| `rnd.multiplier_max` | 選んだ方式の `multiplier_max` |
| `rnd.warmup_steps` | 旧モードに応じて誤差の統計の待ちか predictor の待ち |
| `rnd.warmup_mode` | learn_predictor は標準化の待ち。collect_errors は標準化の待ち 0 と predictor の待ち。variance_feedback は predictor の待ち |
| `rnd.error_norm.use_centering` | `rnd.error_norm.standardize.use_centering` |
| `rnd.error_norm.average_mode` | `rnd.error_norm.standardize.average_mode` |
| `rnd.error_norm.ema_steps` | `rnd.error_norm.standardize.ema_steps` |
| `learn_ratio` | `learn.intake.ratio` |
| `learn_batch_size` | `learn.intake.batch_size` |

## 経過

- 2026-11-22 未決事項を監査した。対象課題は 027698、4976bf、ce0101、c1744a。
- 変更前として現行 Release Runner と設定 17 ファイルを `.scratch/prd191/before-ac5135ff` に退避し、実行体のコピーが SHA-256 `129F09E77DC4276AE4D09662BE52FB838E31E55C90D9484DDE3B9DBC709C2B53` で一致することを確認した。退避を行った時点の HEAD は `cfe71a07`。Run は受入検証時に実施する。

### 実装と検証の経過

- 構造図は変更前に 4 ファイルの欠落で失敗し、実装後の DQN テストは有効・無効の 10 断定が一致した。
- 待ちの分離は変更前に統計の未成立と学習回数の 2 断定で失敗し、変更後は 3 断定が一致した。割らない式のスケーラのテストは変更前の報酬 0 で失敗し、実装後は 9 断定が一致した。
- rollout の 4 lane・8 件バッチのテストは既存の intake からの学習で失敗し、一括と分散の実装後に比率 1 と 2 の両条件が通った。
- 先に変更前の LunarLander 5 条件を完走した。比較入力と scalar は .scratch/prd191/functional-01/。起動・計測の手順は [PRD191 検証資材](../../src/core/base/testdata/prd191/README.md) に記録した。
- 旧キー 11 語を追跡済み apps と src で検索し、指定した検証資材を除いて 0 件。PRD194 の診断監査では、RND の旧 ERROR 名と旧 ERROR 統計への呼び出しを削除し、モード別の倍率上限・待ち・観測正規化への呼び出しと統計モード不一致の診断を採取して期待の一覧を更新した。監査は 211 行一致。
- Debug の core 全体の初回実行は 665 件中 663 成功、1 失敗、1 skip。失敗は Main View common default is Eval。退避済みの変更前 common.txt で app.main_view が train に設定されており、今回そのファイルは編集していない。
- 11、12、13 番バッチは変更前の写しを同じ退避先に保存した。現用 RND の分岐は新しい Runner に揃えた。既存 Run のプロセス操作は行っていない。
### 前後比較と設定の検証

- Release Runner で 5 条件を各 100,000 経験実行した。追跡時間と資源の prefix 3 個以外の全 212,798 scalar 点が一致した。旧累積 checkpoint を読み込んだ追加 Run も完走した。後のレビュー対応では rollout 専用の読み出しと計測だけを変更し、この 5 条件が使う intake 経路には手を加えていない。
- `.scratch/prd191/functional-01/config-comparison.json` で解決後の全設定を比較した。Run 名と PRD191 の移行 24 キーを除くと、条件ごとに 1,002〜1,024 キーの文字列が一致した。移行 24 キーも、旧値の移し替えと新既定の期待と一致した。`.scratch/prd191/verify_resolved.py` を使用。
- `.scratch/prd191/verify_config_smoke.py` の短い Runner 6 本が通った。旧キーは読まれない設定一覧に出て、現在の待ちは 12,800 のまま。未選択モードを含めた 5 条件では対象キーの WARN・INFO と未読一覧への掲載が無かった。
- `check_default_leaves.py` は移転前の core を固定参照していたため、追跡中の検査器を変えず、検索ルートだけ src に読み替えた。変更対象の設定とバッチに違反は無かった。全体では既存の Metrics Viewer の Dash 属性 4 件を設定キーとして検出した。記録は `.scratch/prd191/default-leaf-audit/default-leaf-audit.json`。
- 構造図は Release の有効 Run で 4 ファイル、無効 Run で該当ファイル 0。比較元と実行体・設定は `.scratch/prd191/before-ac5135ff` に保存した。
- DQN 全体の初回実行は 376 件中 375 成功。初期優先度の比較で 1 断定が僅差で失敗し、同じバイナリでそのテストだけ実行すると 10 断定すべて通った。
- 初回独立レビューの原文は `.scratch/prd191/review-20261122-0412.md`。実装計画の順序・実行コマンド・旧新対応表、README の LF、設定と Trace の検証を補った。rollout ではコピー・抽選・学習の計測区間を分け、選択番号は同じ CUDA device に 1 回だけ非同期で送る。
- rollout 抽選は lane と時間の組ごとに別の値を入れ、独立に組み立てたバッチでの predictor loss と比較する。置き場を空にする操作は部品の公開操作で確認し、Agent では直後から次の rollout の周期で学ぶことと、次の観測群での loss から確認する。
- `final-debug-build.log` の Debug ビルドが成功。core の関連 43 件・4,267 断定、DQN の関連 41 件・5,828 断定がすべて通過した。core では別変更で不一致が出ている PRD110 の出荷設定チェックを関連テストの再実行から外した。待ちの境界・Trace・分解と合成の Replay 報酬・checkpoint の抽選状態をこの再実行で確認した。
- 2 回目の独立レビュー（`.scratch/prd191/review-20261122-0423.md`）は計画が OK、実装は計測区間の書き方 1 件の差し戻し。計測専用ブロックを既存の `ANET_PROFILE_SCOPE_END` と `ANET_PROFILE_SCOPE_NEXT` へ置き換え、コピー・抽選・学習の境界を保った。
- 3 回目の独立レビュー（`.scratch/prd191/review-20261122-0425.md`）で計画と実装がともに OK。計測区間の書き方と設計文書 212 の参照先・待ちの説明を再確認した。根拠が無く見送った指摘は無かった。受入 18 は開発者の読解確認待ち。


### Atari の時間計測の試行

Montezuma、CUDA、128 lane。最初は replay ratio 4 と Q の hard update 間隔 125 で試した。Replay 容量を計測区間を収める 131,072 件にそろえた。32,768〜49,152 経験の経過時間を、各群を先頭にする順で 3 巡測った。別の旧 Runner プロセスも稼働していた。9 試行の SHA-256 がすべて `545bb30d4564319badf849dba06139b71bafc3cf4065e3d2301a77bbcecd6bf2` であることを確認した。計測開始後に計測区間のマクロ表記を変更したため、9 試行にはその直前にビルドした Release Runner を使用した。受入 17 の構成は replay ratio 8 と Q の hard update 間隔 250。

| 補助学習 | 1 巡目 (秒) | 2 巡目 (秒) | 3 巡目 (秒) | 中央値 (秒) |
| --- | ---: | ---: | ---: | ---: |
| intake | 14.027 | 13.142 | 14.011 | 14.011 |
| rollout 一括 | 12.356 | 12.490 | 13.199 | 12.490 |
| rollout 分散 | 12.405 | 12.426 | 12.618 | 12.426 |

元の記録は `.scratch/prd191/perf-01/medians.json`。

別の Run の Tracy で `intrinsic_rollout_batch` を 2 回採取した。最初の一括区間が 702.801 ms、次が 108.619 ms。各区間に同じスレッドの `intrinsic_learn` が 16 回ずつ含まれることを `tracy-csvexport -u` で確認した。値はコピーの後から 16 回分の抽選と学習を終えるまでのホスト経過時間。元の記録は `.scratch/prd191/profile-01/rollout.tracy` と `rollout-zones.csv`。


### Atari の時間計測（受入 17）

Montezuma、CUDA、128 lane、replay ratio 8、Q の hard update 間隔 250。Replay 容量は計測区間を収める 131,072 件。計測マクロ修正後の Release Runner で 3 方式を各 3 回、開始順を入れ替えて実行した。今回のコードの指摘対応はコメントの追加で、計測に使う処理は同じ。別の旧 Runner プロセスも稼働していた。

| 補助学習 | 1 巡目 (秒) | 2 巡目 (秒) | 3 巡目 (秒) | 中央値 (秒) |
| --- | ---: | ---: | ---: | ---: |
| intake | 25.852 | 28.654 | 27.975 | 27.975 |
| rollout 一括 | 24.991 | 28.176 | 27.915 | 27.915 |
| rollout 分散 | 24.777 | 28.654 | 28.556 | 28.556 |

計測区間は 32,768〜49,152 経験。元の記録は `.scratch/prd191/perf-rr8/medians.json`。9 試行の Runner の SHA-256 は `780d255a498a45d6f20769226866a603d6c040882461d10e3bd5fe6776bf6414`。

同じ設定の別 Run で Tracy の一括区間を2回採取した。最初が 2,641.977 ms、次が 93.340 ms。どちらも同じスレッドの `intrinsic_learn` が16回あり、コピーのあとから抽選と学習を終えるまでを測った。元の記録は `.scratch/prd191/profile-rr8/rollout.tracy` と `rollout-zones.csv`。


### 2 モデルのレビュー対応

- 初回の原文は Codex が `.scratch/prd191/review-20261122-0829.md`、Claude が `.scratch/prd191/review-20261122-0819-claude.md`。両方が計画と実装を差し戻した。
- 両方の指摘に合わせ、150 の部品表から 212 を参照し、受入 12 で非選択モードの不正値・診断の指定値・intake の上限を適用する条件を確かめるテストを補った。計画の用語も合わせた。
- 212 に Taiga の Run 連鎖・β・AdamW の設定、DropMerge の待ち、Env 別の Network を戻した。200 には無効時の構築と検証の挙動を戻し、Env 別の設定・構造を 212 で参照できるようにした。処理段階のコメントとテストの定数も補った。
- Atari の Trace 列 440 と 450 は、着手前に退避した `.scratch/prd191/before-ac5135ff/config/Atari.txt` に存在した。退避版と現用を比較すると `Atari.txt` の PRD191 による差分は RND の設定キーだけだった。再レビューでは列 440 と 450 を並行変更として明記する。
- 診断の監査は 211 行一致した。日本語設計文書の節リンク、既存ファイルの BOM、変更ファイルの LF、差分の空白、検証用 Python 2 本の構文を確認した。
- 同じ Debug・Release のコマンドでビルドし直した。Core は `[prd191],[intrinsic],[shipped_config] ~[prd110]` の43件・4,271断定、DQN は `[prd191],[intrinsic],[shipped_config]` の41件・5,831断定が通った。ビルドとテストの出力は `.scratch/prd191/cc-fix-*.log`。
- 2 回目の原文は Codex が `.scratch/prd191/review-20261122-0855.md`、Claude が `.scratch/prd191/review-20261122-0849-claude.md`。Claude は計画と実装を OK とし、Codex は工程末の要約と設定エラーの断定を差し戻した。
- 設定エラーのテストでは、期待範囲の数値で通らないよう、完全キーと指定値の欄を一体で照合する。工程末の要約も更新した。
- Debug の Core・DQN テスト実行体を上記と同じビルド手順で再作成した。今回の対象テストは Core `RND rejects invalid configuration before building networks` が1件・24断定、DQN `DQN intrinsic validates schedule settings only when enabled` と `DQN validates RND rollout settings and only checks lane combinations when selected` が2件・37断定で通った。例文から指定値を省いた場合は、期待範囲に同じ数値が残っていても照合に失敗することを3例で確認した。出力は `.scratch/prd191/cc-check-fields-*.log`。


- 3 回目は Codex・Claude ともに実装計画と実装を OK と判定した。原文は `.scratch/prd191/review-20261122-0903.md` と `.scratch/prd191/review-20261122-0900-claude.md`。両方が工程末の要約と、完全キーに続く指定値の照合を再確認した。

### 工程末の確認

- **実施内容**　生の予測誤差、段階ごとの待ち、一括・分散 rollout、RND の構造図を実装し、設定・バッチと設計文書を更新した。受入 1〜17・19〜20 の結果と最新のビルド・テストの記録は、上の検証とレビュー対応の節に記した。
- **変更ファイル**　RND 推定器、DQN Agent、lane × 時間の tensor 保管部品、対応するテスト、出荷設定・11〜13 番バッチ、検証資材・変更履歴、設計文書 [212](../design/212_rnd_lifelong_novelty.jp.md)・[200](../design/200_dqn_agents.jp.md)・[150](../design/150_replay_buffer.jp.md)、関連課題 4 件と派生一覧。
- **レビュー結果**　2 モデルのレビューの判定と対応は、直前の「2 モデルのレビュー対応」に記した。
- **決まらなかったこと**　PRD072 の既定キーの検査が移転前の `core` を探索して停止する問題を [issue-20261122-4adabb](../management/issues/issue-20261122-4adabb.md) に起票した。検証では一時コピーで探索先を `src` に読み替えた。Metrics Viewer の Dash 属性 4 件が検出され、PRD191 の変更対象の検出は 0 件だった。
- **判断してほしいこと**　受入 18 のうち、開発者が設計文書 212 の §2〜§7 を読み、設定から段階・モード・待ちをたどれるかの確認は未実施。関連課題 c1744a は残件有り。他の 3 件は残件無し。課題一覧および派生区画を再生成した。
- **次にできること**　受入 18 の読解確認が届いたら実装記録と課題の判定に反映する。

