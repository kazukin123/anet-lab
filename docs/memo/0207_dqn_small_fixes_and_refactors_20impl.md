# RFD0207 DQN 系の小修正とリファクタリング 実装メモ

## 概要

[RFD0207](0207_dqn_small_fixes_and_refactors_10rfd.md) に従い、DQN 系の 12 課題を進める。RFD0206 の完了時点の Runner を前側に使い、出荷設定を含む 37 構成で、設定記録・ログ・scalar の差を確認する。

## 主な変更

- 可視化の特性を確認し、DefaultDQN と Rainbow の学習値域、不正な行動番号の fatal、方策の `uqe_tau` と点 UQE の比を修正する。
- 学習の進みと PER の β を Learner に移し、Assembler が Learner の値を共通の読み方で扱う。2 群の診断を名前で順序付け、割引率、Actor Q ヒント、選択行動の抽出、probe の抽選をそれぞれの共通部品に集める。
- 設計 030・200 の和英と、現行の保存構想に使う RFD0143 を改訂する。RFD0143 の改訂箇所は RFD の観点でも独立レビューを受ける。

## テスト

| 実装順と観測する挙動 | 配置先 |
| --- | --- |
| e43866 可視化の形と複製値を、実装前から通る特性テストで確認 | `src/algo/dqn/src/dqn_concrete_test.cpp` |
| eb0a0c 学習値域、直接指定の非有限値、PER の警告を RED から確認 | `dqn_concrete_test.cpp` |
| ea9fef 不正な番号の例外と fatal の件数、c702d2 未使用の `uqe_tau`、770358 点と分布の比を RED から確認 | `dqn_actor_test.cpp` |
| 84c03f 初期 per_beta、更新・復元・Policy churn の既存の期待 | `dqn_concrete_test.cpp`、`dqn_{actor,learner,math,munchausen}_test.cpp`、`dqn_fixture_test.hpp` |
| 6a8e46 値の委譲、5ebf2f 2 群の順序、d74c3f 共有・個別 gamma | 既存テストと 37 構成。2 群は一時的に位置を入れ替え、各ビルドが失敗したあと元に戻す |
| de8cfd 合成 Q、外的・内的それぞれの値、Replay 初期優先度 | `dqn_intrinsic_test.cpp` の分解診断、既存の `dqn_{concrete,actor,munchausen}_test.cpp` |
| 9e8834 2D〜4D の行動選択、9f1801 probe の値と乱数 | 既存テスト、短縮比較、呼び出しの検索 |

最初の挙動変更の確認は、`use_grad_clip = false` かつ `grad_clip_tau = 0` の `ConfigData` を DefaultDQN と Rainbow の設定構築へ渡し、完全キー・値・期待範囲が診断に出るところまでを通す。旧実装で拒否しない失敗を確認し、共通検証を入れて同じテストを通す。挙動を直すほかのテストも、旧実装で予期した失敗を確認してから直す。不正な設定は有効な境界も通し、点の診断は有効な分布の指標と照合する。リファクタリングでは既存の期待値と短縮 Run の点を照合する。

## 検証

- 標準の MSVC 初期化から Debug 全体と DQN・基盤の既定スイート。対象テスト、実行コマンド、終了値を `.scratch/rfd0207/` に残す。
- A13（等価性）は `.scratch/rfd0207/validation.py` の `snapshot`、`prepare`、`before`、`after`、`compare`。seed 1、8 lane、8,704 experience、FP32、決定論的 backend。並行採取時は CUDA の短縮 Run を 1 本ずつにする。
- RFD §7 の検索条件、2 群の順の破壊検査と復元ビルド、和英のハッシュとリンクを確認する。A14（性能）の基準 Release は着手前に保存し、測定は夜間の短縮範囲から外す。
- 比較資材で改めた RND のキーは `intrinsic.learn_batch_size` → `intrinsic.learn.intake.batch_size`（8）、`intrinsic.rnd.warmup_steps` → `intrinsic.rnd.error_norm.standardize.warmup_steps`（256）。旧既定の predictor 待ちは 0。12 課題の検証結果は課題の対応内容にも記録する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target all AnetRLRunner-gui-test'
```

```powershell
.\out\test\Debug\anet-core-base-test.exe --order rand --rng-seed 2899524567 --anet-test-failure-dialog=off
```

```powershell
.\out\test\Debug\anet-algo-dqn-test.exe --anet-test-failure-dialog=off --durations yes --rng-seed 27071128
```

Assembler の委譲時に増える共有ロック取得を見積もる。RunnerApp の学習 Observer が正規化の平均と標準偏差を `GetTensor` で 100 更新ごとに読む経路で、取得が 2 回増える。競合がない取得と解放を 1 回 1 μs と置くと、この経路の追加は 100 更新あたり 2 μs、1 更新換算 0.02 μs。これは推定値で、性能測定は A14（性能）が対象。

## 前提

RFD0205・RFD0206 の完了時点を土台とする。未決事項監査の開発者判断待ちは 0 件。以下は採用した前提。過去の ADR や検証資材は出典として扱う。

| 固定した前提 | 採用内容 |
| --- | --- |
| U1（値域の置き場） | LearnerConfig の同じ検証を両 Agent の設定読取後と Learner 直接構築で使う |
| U2（残りの値域） | κ は正、α は 0 以上、replay_ratio は非ゼロ、batch は 1 以上。α の既存 WARN は PER の actor_approx に限る |
| U3（UQE を使わない方策） | UQE 使用かつ非 Thompson のときだけ uqe_tau を返す |
| U4（点 UQE の比） | 2 点以上の tail または非 spatial Thompson の決定用全範囲で算出し、点では NaN |
| U5（不正な行動番号） | 変換例外と読み残しから同じ fatal を 1 回出す |
| U6（学習の状態） | learn_step と per_beta を Learner に移し、既存の評価時機を保つ |
| U7（値の振り分け） | Assembler は target_policy を先に読み、それ以外を共有ロック内で Learner に委ねる |
| U8（行動の選択） | 2D〜4D の選択を 1 関数に移し、NaN 検査は既存の内部呼び出しだけで行う |
| U9（Actor Q ヒント） | 共通関数が補助値、bootstrap、TBO、Munchausen の入力を組み立てる。分解時の step 診断は、合成 Q と各 Head に使う値を得るため、`MakeActorQHint` の 2 呼び出しを維持する |
| U10（内的の割引率） | 分解かつ separate のとき内的 gamma、それ以外は外的 gamma を返す |
| U11（診断の順序） | 名前付きの番号を使い、学習の群と行動選択の列のずれをコンパイル時に止める |
| U12（probe の抽選） | 4 群で共通の抽選を使い、obs のみ Perception に通す |
| U13（可視化の特性） | 4 枚の FrameStack で all と listed、2 キーの形と値を観測 |
| U14（後続候補） | 負の replay_ratio をモード指定に移す件は候補に記録 |
| U15（実施順） | 可視化、値域、fatal、2 診断、状態、振り分け、診断順序、割引率、ヒント、行動選択、probe の順 |
| 性能の測定 | 着手前 Release を保存し、A14 の実行は朝の残件にする |
| 短縮比較の資材 | Debug に現行の設定を凍結し、RFD0182 の部品から 37 入力。旧 RND は batch 8 と標準化待ち 256 を現行キーへ移す |

## 経過

- 2026-11-28 入口の独立レビューで一度差し戻し。設定・診断の条件、比較、性能の残件、所有権、テストを修正し、再レビュー OK。
- 2026-11-28 夜間に推奨で採った。U1（値域の置き場）では、LearnerConfig の同じ検証を両 Agent の設定読取後と Learner 直接構築で使う。完全キーの診断を保ち、設定型の直接指定にも同じ境界を適用するため。
- 2026-11-28 夜間に推奨で採った。U2（残りの値域）では、κ は正、α は 0 以上、replay_ratio は非ゼロ、batch は 1 以上。α の既存 WARN は PER の actor_approx に限る。課題の7キーと既存のモードをすべて扱うため。
- 2026-11-28 夜間に推奨で採った。U3（UQE を使わない方策）では、UQE 使用かつ非 Thompson のときだけ uqe_tau を返す。未使用の既知指標は NaN とする契約に合うため。
- 2026-11-28 夜間に推奨で採った。U4（点 UQE の比）では、2 点以上の tail または非 spatial Thompson の決定用全範囲で算出し、点では NaN。同じ τ の複製は分散の推定にならず、可視化用の全範囲は別だから。
- 2026-11-28 夜間に推奨で採った。U5（不正な行動番号）では、変換例外と読み残しから同じ fatal を 1 回出す。ログの文言を保ち、重複する経路だけを除くため。
- 2026-11-28 夜間に推奨で採った。U6（学習の状態）では、learn_step と per_beta を Learner に移し、既存の評価時機を保つ。書き込み元と所有元をそろえるため。
- 2026-11-28 夜間に推奨で採った。U7（値の振り分け）では、Assembler は target_policy を先に読み、それ以外を共有ロック内で Learner に委ねる。指標一覧の写しを減らし、Agent の fallback を保つため。
- 2026-11-28 夜間に推奨で採った。U8（行動の選択）では、2D〜4D の選択を 1 関数に移し、NaN 検査は既存の内部呼び出しだけで行う。13 箇所の形と失敗条件を保つため。
- 2026-11-28 夜間に推奨で採った。U9（Actor Q ヒント）では、共通関数が補助値、bootstrap、TBO、Munchausen の入力を組み立てる。分解時の 2 呼び出しは、合成 Q の計算の重複を避けられるかで決める。合成 Q の列と Head 別の Replay ヒントを両方保つため。
- 2026-11-28 夜間に推奨で採った。U10（内的の割引率）では、分解かつ separate のとき内的 gamma、それ以外は外的 gamma を返す。6 箇所で同じ条件にし、無効な設定が効く道を除くため。
- 2026-11-28 夜間に推奨で採った。U11（診断の順序）では、名前付きの番号を使い、学習の群と行動選択の列のずれをコンパイル時に止める。指標の名前、値と lane ごとの選択を一緒に保つため。
- 2026-11-28 夜間に推奨で採った。U12（probe の抽選）では、4 群で共通の抽選を使い、obs のみ Perception に通す。標準の部品の乱数・統計を保ち、次の観測の未使用の計算を除くため。
- 2026-11-28 夜間に推奨で採った。U13（可視化の特性）では、4 枚の FrameStack で all と listed、2 キーの形と値を観測。可視化へ 1 枚を渡す公開経路を検査するため。
- 2026-11-28 夜間に推奨で採った。U14（後続候補）では、負の replay_ratio をモード指定に移す件は候補に記録。今回の有効な設定を引き継ぐため。
- 2026-11-28 夜間に推奨で採った。U15（実施順）では、可視化、値域、fatal、2 診断、状態、振り分け、診断順序、割引率、ヒント、行動選択、probe の順。変更が依存する順に並べ、同じ行を何度も直さずに済むため。
- 2026-11-28 夜間に推奨で採った。性能の測定では、着手前 Release を保存し、A14 の実行は朝の残件にする。長い Run の夜間実行を控える依頼に従うため。
- 2026-11-28 夜間に推奨で採った。短縮比較の資材では、Debug に現行の設定を凍結し、RFD0182 の部品から 37 入力。旧 RND は batch 8 と標準化待ち 256 を現行キーへ移す。過去の採取物を保持し、前後の有効設定を共通にするため。
- 2026-11-28 変更前の Debug Runner を保存した。SHA-256 は `9b29565051609032bd3628dae091f400f222623369cd47cda40a5f43783c645d`、依存 DLL と凍結した設定は `.scratch/rfd0207/baseline/manifest.json`。37 入力の生成は終了値 0（`.scratch/rfd0207/prepare.log`）。
- 2026-11-28 入口レビューの原文を `.scratch/rfd0207/review-20261128-0034-rfd-initial.md` と `review-20261128-0034-rfd-followup.md` に保存。変更前 Release のビルドは終了値 0。Runner の SHA-256 は `6d72a3efff2c707adbf8054b2e9de82b1cceec4a31381a9f9835dcda7cb0bfe9` で、DLL とともに `.scratch/rfd0207/baseline/Release/` に保存した。
- 2026-11-28 A15（可視化の特性）。最初に既存コードへ公開経路の特性テストを追加。未修正の Debug 実行体で `all`・`listed` の 16 アサーションが通過した（`characterization-visualization.log`）。
- 2026-11-28 A2（設定値の境界）。変更前は 0・負の設定が複数通ることを再現。直接構築ケースの最初の fixture には Actor の forward キーがなく、設定を読み切るよう修正した。共通検証を入れた後は `[learner_values],[per]` の 21 ケース・314 アサーションが通過。既定の PER の WARN も通過した（`red-values.log`、`green-values.log`）。
- 2026-11-28 A3（未使用の uqe_tau）・A4（点の比）・A5（fatal の重複）。修正前の試行で不正な数字の後続文字のログが 2 件、Greedy と非 spatial EpsilonGreedy の数値、点の比の数値を確認。修正後は 6 ケース・135 アサーションが通過（`red-diagnostics.log`、`green-diagnostics.log`）。非 spatial Thompson の有効な比、tail の既存値、可視化用 full query を併用した点の NaN も確認した。
- 2026-11-28 学習回数の移行で、既存テスト本体の 8 行に加え、共通 fixture の `dqn_fixture_test.hpp` でも 1 行が回数を読んでいたため、同じアクセサへ移行した。
- 2026-11-28 A6（学習状態）。Debug の DQN ビルド、更新前の per_beta と Policy churn の対象 3 ケース・16 アサーションを確認した。RFD0143 はウォームアップの状態の意味と所有範囲を明確にし、独立した再レビューで OK。
- 2026-11-28 夜間に推奨で採った。U9（Actor Q ヒント）では、分解時の step 診断の MakeActorQHint を 2 呼び出しのままとした。合成 Q の列と分解した Head の列で使う変換・soft value が異なるため、1 呼び出しへ寄せるためにその計算を外へ複製するのを避けた。
- 2026-11-28 A13（短縮 Run）の前側で診断 DropMerge が `loss_ema` の取得元未指定で停止した。旧資材の明示取得元の規則を現在の `@full : [tag]` 記法へ適用し、12 宣言に `$agent` を付与。分解 2 構成にも内的 gamma と両 Head の loss の検証用指標を追加し、3 構成を同一の前側 Debug Runner で再採取して通過した。残る 34 構成も前側で通過。採取ログは `validation-before.log` と `validation-before-r2.log`。
- 2026-11-28 A1（Debug 全体）。最初のビルドでは Actor テストがヒント設定の旧 bool を参照して止まった。既存テストを同じ外的 bootstrap モードの指定へ移し、全体と Runner GUI テストのビルドが成功した（`full-debug-build-r2.log`）。
- 2026-11-28 A1（既定 DQN）。最初の実行で旧 fixture の replay_ratio=0 と旧 IQN 限定の κ 診断を期待するテストを検出。負の値で同じ interval モードを指定し、新仕様の値域診断にテストを揃えた。手を入れたのはテスト資材と診断の期待だけ。
- 2026-11-28 実装計画・実装とも独立した再レビューで OK。初回と再レビューの原文は `.scratch/rfd0207/review-20261128-0127-impl-initial.md`、`.scratch/rfd0207/review-20261128-0135-impl-followup.md`。
- 2026-11-28 A10（並びの破壊）。学習・行動の 2 か所でそれぞれ隣り合う名前と値を入れ替え、どちらも指定の static_assert で終了値 2。元へ戻すとそれぞれ終了値 0（`diagnostic-order.log`）。

## 検証結果

- A1（最終ビルド）。Debug 全体と Runner GUI テストを終了値 0 で再ビルドした（`full-debug-final.log`）。A13（短縮 Run）の採取後に Runner を再リンクした差分は、C++ の無名名前空間内のインデントのみ。
- A1（基盤）。Debug の `anet-core-base-test` は seed 2899524567 の乱順で 697 件通過、1 件スキップ。34,310 アサーションが通った（`core-full.log`）。
- A1（DQN）。既定スイートの再実行は seed 27071128、終了値 0。391 件・161,632 アサーションが通過（`dqn-full-r2.log`）。
- A6（学習状態）。`src`、`docs/design`、RFD0143 の旧 `RuntimeVars` 識別子は 0 件。
- A7（振り分け）。Assembler の scalar・Tensor の 3 読み口で Learner 側の群や Replay の接頭辞の判定が 0 件。target policy の先行経路と DefaultDQN の fallback を維持。
- A8（選んだ行動の取り出し）。共通関数の呼び出しは 13 箇所。Learner 8、NetworkModel 2、Actor 3。quantile の NaN 検査は元の呼び出しだけに残した。
  - `dqn_network_model.cpp` の 335・339 行（2 箇所）。
  - `dqn_actor.cpp` の 144・165・178 行（3 箇所）。
  - `dqn_learner.cpp` の 597・604・2217・2488・2599・2642・2772・2873 行（8 箇所）。
- A9（probe）。`SampleUniqueUniform(` は Learner 内で 1 件。Noisy と Policy churn が既存の `MakeTauMidpointPositions` を使い、学習本体の次の観測の変換を残す。
- A10（診断の順）。学習と行動選択の両方で、隣り合う項の入れ替えをコンパイルが拒否し、復元後に対象オブジェクトが再ビルドできた。
- A12（内的 gamma）。選択条件は `GetIntrinsicGammaConfig` に集めた。検索に残る行は定数の定義と初期値、DefaultDQN の設定検証 3 行、共通関数。
- 設計 030・200 の和英で見出し階層、表、フェンスが一致。未訳の日本語本文 0、対象文書のリンク切れ 0。翻訳元 hash は 030 `2ea50974f718df22675fd14159227b17d557aeb3`、200 `88e240f4d6814499e868575c4e74de6cee7a76b0`。
- 機械検査の基準コミットは `a60bae83cbf8c43d872882f910ac9318249e0df7`。先行 RFD0205 の設計 200 の内容から今回の差分を分けた。日本語の比較元の hash は `119bb14b92b25756e00b9e57f26007600583df7c` に一致し、実装の独立レビュー用の 39 差分は `.scratch/rfd0207/diff-20261128-0131/`。

- 最初の DQN 既定スイートは 390 ケース中 379 通過、11 失敗（81 アサーション）。旧 fixture の replay_ratio=0 と κ の旧診断への依存があり、許容境界と新しい診断へ合わせた。再実行は `dqn-full-r2.log`。
- 実装レビューで不足が分かった正負の更新モードは公開 Learner の更新回数で確認する。分解 step 診断では、同じ行動と固定の 2 成分報酬を実 ReplayBuffer に入れ、`PER_VALUES` を読む。可視化の捕捉データはテストの局所所有とする。

- A2（設定値の境界）の追加テストの初回は、Replay からまだ学習できない時点の戻り値を数えて失敗した（`dqn-review-targets.log`）。最初の 4 ステップをウォームアップで経験の登録に使い、次の 4 ステップで正 4 回・負 2 回を確認するようにした。Replay の初期優先度は、同じ lane の最初の格納位置を読むテストへ変更した。追加テストのステップ型の縮小変換を MSVC が拒否したため、公開 API と同じ `rl::step_t` に揃えた（`dqn-review-build-r2.log`）。

- 追加の重点テストは 26 ケース・371 アサーションが通過（`dqn-review-targets-r2.log`）。正負の実更新と、lane ごとの同じ初期優先度を確認した。
- A14（性能）は夜間依頼に従い測定を保留した。比較元の Release と DLL は `.scratch/rfd0207/baseline/Release/`、実行体の SHA-256 は `.scratch/rfd0207/baseline/release-manifest.json`。凍結した Runner 設定は `.scratch/rfd0207/baseline/config/`、比較入力は `.scratch/rfd0207/inputs/performance.txt`（SHA-256 `0c094d602ce97906bba4e848dbef2b20d45989270d186c66ea71f2fb992b0b45`）。設定解決のテストは 1 ケース・4 アサーションが通過（`performance-config.log`）。解決結果の qbert、IQN、CUDA BF16、128 lane、batch 256、終了経験 82,176 を確認した（`performance-resolved.json`）。
- A14（性能）を再開する際は、変更後の Release Runner を別途ビルドし、同じ設定で基準側の捨て Run 後に新旧を交互に 3 巡以上計る。RFD0125 の手順（`src/core/base/testdata/rfd0125/run_validation.py`）に沿い、`90_perf/98_rfd0125_time` の学習ステップ 64 と 256 の実時間差を 192 更新で割る。変更後の中央値が変更前の最大値以下かを判定する。

### A13（短縮 Run の前後比較）

前後の 37 構成を同じ設定・seed で採取した。比較対象 1,077,110 点のうち 1,074,925 点が値と step を含め一致。消失は 15 構成の 2,185 点で、未使用の target `uqe_tau` の 1,054 点と、点 UQE の `iqn_policy_margin_mc_ratio` の 1,131 点に限られた。追加 0、ほかの消失 0。設定テキスト、Agent 設定 JSON、保存した設定一式、WARN/ERROR の文言・順・回数は全 37 構成で一致。Run の保存先と名称、WARN/ERROR のソース位置を正規化した。`90_perf/`、`91_resource/`、`92_resource/` の除外点を含む採取結果は `validation-after.log` に記録した。

| 構成 | 一致した scalar 点 | target `uqe_tau` の消失 | 点 UQE の比の消失 |
| --- | ---: | ---: | ---: |
| `default_Atari` | 23,229 | 0 | 0 |
| `default_DropMerge` | 30,747 | 527 | 0 |
| `default_GridMaze` | 27,382 | 0 | 0 |
| `default_LunarLander` | 24,376 | 0 | 87 |
| `diagnostics_DropMerge_explicit_sources` | 48,499 | 0 | 0 |
| `DropMerge_iqn32_actor_approx` | 38,112 | 527 | 0 |
| `iqn_base` | 24,342 | 0 | 87 |
| `iqn_no_per` | 21,632 | 0 | 87 |
| `iqn_online` | 27,528 | 0 | 87 |
| `iqn_online_reuse` | 27,480 | 0 | 87 |
| `iqn_online_skip` | 37,765 | 0 | 87 |
| `iqn_skip` | 34,605 | 0 | 87 |
| `iqn_target` | 27,556 | 0 | 87 |
| `iqn_target_tbo_skip` | 39,887 | 0 | 87 |
| `iqn_tbo` | 26,546 | 0 | 87 |
| `none_base` | 20,118 | 0 | 0 |
| `none_no_per` | 16,682 | 0 | 0 |
| `none_online` | 23,240 | 0 | 0 |
| `none_online_reuse` | 23,240 | 0 | 0 |
| `none_online_skip` | 33,433 | 0 | 0 |
| `none_skip` | 30,307 | 0 | 0 |
| `none_target` | 23,284 | 0 | 0 |
| `none_target_tbo_skip` | 35,597 | 0 | 0 |
| `none_tbo` | 22,250 | 0 | 0 |
| `qr_base` | 23,843 | 0 | 0 |
| `qr_no_per` | 20,439 | 0 | 0 |
| `qr_online` | 26,941 | 0 | 0 |
| `qr_online_reuse` | 26,941 | 0 | 0 |
| `qr_online_skip` | 37,204 | 0 | 0 |
| `qr_skip` | 34,020 | 0 | 0 |
| `qr_target` | 27,021 | 0 | 0 |
| `qr_target_tbo_skip` | 39,260 | 0 | 0 |
| `qr_tbo` | 25,867 | 0 | 0 |
| `rnd_Atari_b8` | 30,063 | 0 | 0 |
| `rnd_LunarLander_b8` | 31,187 | 0 | 87 |
| `rnd_LunarLander_separate` | 32,151 | 0 | 87 |
| `rnd_LunarLander_shared` | 32,151 | 0 | 87 |
| 合計 | 1,074,925 | 1,054 | 1,131 |

採取は `validation-before.log`、`validation-before-r2.log`、`validation-after.log`。比較は `validation-compare.log`（元の前側 34、前側を取り直した 3 とも終了値 0）、構成別の差は `.scratch/rfd0207/after/` の `comparison-207.json`。分解した LunarLander の両構成で内的 gamma と両 Head の loss を確認し、separate で内的 gamma が予定どおり変化した。DropMerge の actor_approx では 38,112 点が一致し、消えた点は未使用の target `uqe_tau` 527 点だった。

### A14（Learner 更新の性能測定）

既存 Run を Pause したとの連絡を受け、変更後の Release をビルドした（`.scratch/rfd0207/performance-release-build.log`、終了値 0）。起動前の 3 秒間に既存 Runner の CPU 時間の増分は 0 秒で、ほかの Runner・テスト・ビルドは動いていなかった。保存済みの `performance.txt` を使い、seed 1、Atari qbert、IQN、CUDA BF16、128 lane、batch 256、82,176 experience で基準側の捨て Run と基準→変更後の 3 巡を順に実行した。7 本とも終了値 0。比較では Run 名を正規化し、測定 6 本の解決済み設定が捨て Run と一致した。基準と変更後の Release に同梱した DLL 50 本は SHA-256 が一致した。

| 巡 | 基準（ms/更新） | 変更後（ms/更新） |
| --- | ---: | ---: |
| 1 | 40.307 | 41.922 |
| 2 | 39.958 | 40.578 |
| 3 | 40.536 | 40.766 |
| 中央値 | 40.307 | 40.766 |
| 最大値 | 40.536 | 41.922 |

`90_perf/98_rfd0125_time` の学習ステップ 64・256 の値の差を秒に換算し、192 更新で割った。捨て Run の 40.578 ms/更新は判定から除外した。変更後の中央値 40.766 ms/更新は基準の最大値 40.536 ms/更新を 0.229 ms（0.57%）上回り、性能受入は未達。差は丸め前の値から計算した。各 Run の測定点から集計を再計算して照合した。

ビルドは Visual Studio の開発者環境で `cmake --build --preset x64-Release --target AnetRLRunner`。測定は `.venv/Scripts/python.exe -B -u .scratch/rfd0207/performance.py --rounds 3` で直列に実行し、受入未達を区別する終了値 3 を返した。出力は `.scratch/rfd0207/performance-run.log` と `.scratch/rfd0207/performance/20261128-044350/summary.json`。変更後 Runner の SHA-256 は `a673f6fafffea1fe18e3e2494d0d529f4cc1508cc40f4bf0305eda609d81e76d`。

#### 同じ条件で再測定

保存した同一の Release 実行体と `performance.txt` で、基準側の捨て Run と新旧交互の 3 巡を再実施した。両回の Runner と設定入力のハッシュは一致した。保存した解決済み設定も Run 名を除いて一致し、Release の DLL 50 本の一致を再確認した。捨て Run と測定 6 本はいずれも完走した。

| 巡 | 基準（ms/更新） | 変更後（ms/更新） |
| --- | ---: | ---: |
| 1 | 40.307 | 40.021 |
| 2 | 41.068 | 40.167 |
| 3 | 40.099 | 41.391 |
| 中央値 | 40.307 | 40.167 |
| 最大値 | 41.068 | 41.391 |

変更後の中央値 40.167 ms/更新が基準の最大値 41.068 ms/更新を 0.901 ms 下回り、性能受入は合格。 初回の 0.229 ms（0.57%）の上限超過は今回再現しなかった。各回の捨て Run を除いた全 6 巡の参考値でも、変更後の中央値 40.672 ms/更新は基準の最大値 41.068 ms/更新以内だった。

測定の終了値は 0。再測定のログは `.scratch/rfd0207/performance-rerun-20261128-071646.log`、生データは `.scratch/rfd0207/performance/20261128-071647/summary.json`。

### 再レビューに向けた追加確認

- 検証コマンドのフェンスをひな形に合わせて PowerShell にした。分解時の step 診断の Q ヒントに関する決定を前提表に反映し、12 課題の状態変更の履歴を「備考」へ移した。
- 設計 200 §9.3（エラー境界）の和英に、DefaultDQN と Rainbow がモードを問わず有限で正の `learner.quantile_huber_kappa` を求めることを記載した。翻訳元 `9c10f7138623e7554b9942ca099db9f8fd8b62a8`、見出し・表・フェンス・リンクを照合した。
- `per_beta`、`uqe_tau`、`q_sa`、`state_value`、`q_values`、`gamma` は、持ち主の内部ヘッダまたは既存の Learner スケジュールキーから取得する。Debug 全体と Runner GUI テストの再ビルドは成功（`.scratch/rfd0206/cc-followup-debug-build.log`）。重点テスト 8 件・237 アサーションが成功（`cc-dqn-focus.log`）。

- `.\out\test\Debug\anet-algo-dqn-test.exe --anet-test-failure-dialog=off --durations yes --rng-seed 27071128` で DQN の既定スイートを再実行した。終了値 0。391 件・161,632 アサーション成功（`cc-dqn-full.log`）。

### 課題の判定

- 「残件無し」 e43866、eb0a0c、ea9fef、c702d2、770358、84c03f、6a8e46、de8cfd、9f1801。
- A14（性能測定）の再測定で受入条件を満たした。5ebf2f（診断の並び）、d74c3f（内的 gamma）、9e8834（行動値の取得）も「残件無し」。
