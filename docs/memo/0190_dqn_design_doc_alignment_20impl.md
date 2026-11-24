# DQN 系 Agent の設計文書と実装の整合 実装メモ

## 概要

[PRD190](0190_dqn_design_doc_alignment_10rfd.md) の課題 8 件に対応する。DQN の用語、設定名、部品の置き場、方策と Actor Q ヒントの説明を実装にそろえ、設計判断を設計文書から引けるようにする。τ の生成のエラー文とテスト一覧で表示する名前を実名の GenerateTaus にそろえる。

## 主な変更

- 設計200で D1〜D12 を反映する。コードマップには dqn_config.hpp、内的報酬、frame stack の置き場を追加する。§4 と §10 は dqn_*_test.cpp と dqn_test.cpp の案内に、PRD101、内的報酬、frame stack のテストを加える。
- 設計140、ユーザーガイド030、設計200のプロファイル名を修正する。設計200の設定キーは出荷設定とコードから確認する。
- 設計200の §11.2 に ADR 22 本、設計214の §8 に 2 本を、該当節付きで掲載する。ADR 0010、0012、0013 の冒頭に後発決定の注記を追加する。
- dqn_math.cpp のエラー文 4 か所と検証関数の名前 3 か所、dqn_test.cpp のテスト名 14 本を置き換える。
- 課題 8 件に対応と確かめを記録し、判定を更新する。wiki の課題一覧を再生成する。

## テスト

- 確認する面は GenerateTaus の呼び出しが返すエラー文と、テスト実行体が表示するテスト名。
- 最初に、既存の不正な配置方式のテストで、固定範囲と lane 別下限の両方について、例外メッセージに `GenerateTaus received sample_mode=invalid` が含まれることを一時的に確かめる。変更前の失敗と、診断文を変えたあとの成功を記録する。受入 10 の差分を守るため、この検証用の assertion は取り除く。
- 次に、テスト名で絞った一覧が変更前には 0 本であることを確認し、14 本の先頭を改名する。既存のタグと本体は PRD の受入 10 で差分を確認し、最終版でタグ [tau] を実行する。
- 検証関数の改名は識別子の整合をとるための変更であり、検索と差分比較で確認する。

## 検証

- 着手前に PRD §7 受入 1 と受入 8 を実行し、終了後に受入 2〜11 を確認する。
- ビルドは AGENTS.md の Windows/MSVC の手順に従い、VsDevCmd.bat を呼んでから x64-Debug を使う。RED と GREEN は anet-algo-dqn-test を対象とし、受入 11 は Debug 全体のビルドと out/test/Debug/anet-algo-dqn-test.exe の [tau] を実行する。
- 四つの和文と三つの ADR の相対リンクのパスを確認する。設計200のクラス図とシーケンス図は方策の所有とヒントの条件を本文と照合する。
- 変更行と対象外ファイルのハッシュを比較し、実装レビュー観点で自己点検する。計画と実装を独立レビューに出す。

## 前提

- e72bab34 の改名後のファイルを基準にする。受入 1 の検索は改訂した PRD のパスで照合済み。
- 設計200の §2.4 に接頭辞の無い frame stack キー、§3 に「複数Policy」、§6.1 に方策の選択と旧いヒント条件が残っていたため D6、D9、D11 を適用する。§4 と §10 は現行のテスト名を掲載済みであり、D3 の三つのテストを追加する。
- PRD の用語集と DropMerge 用 RND の wiki の記載は確認済み。英訳は D17 に従い、翻訳 skill の対象として完了報告に残す。
- DQN 系以外の ADR の案内は §8 U2 に記録済みの範囲外。

## 経過と結果

- 2026-11-19 着手前の再照合で PRD182 のファイル名短縮を検出。開発者の選択を受けて、PRD190 の参照先と受入を現行名へ更新した。
- 受入 1 の機械検査は、TDLearner 1 件、TauGenerator の型 0 件、GenerateTaus の定義 2 件、target_policy_ 2 件、設定読み込み 7 件、設定型 6 件、検証関数 2 件、DropMerge RND の β 1 件、TauGenerator 21 件。TDLearner の本体、ヒントの条件、target 方策の既定・上書き、出荷プロファイルと文書を読み合わせた。D19 の 2 ファイルには着手前の差分が無い。
- 受入 8 の着手前ハッシュは、追跡ファイルの差分 `d5e76adcfe4d24b4ef8ca754419592232fbf4154`、未追跡ファイル一覧 `e92ef8801b903b730c29e9d11ef662a61ae4352b`。
- PRD の改訂を独立レビューに出した。初回指摘の検証関数と行番号の対応を直し、既存 wiki が参照する §1.2 を維持した。再レビューは OK。原文は `.scratch/prd190/review-20261119-2347-prd-round1.md` と `.scratch/prd190/review-20261119-2347-prd.md`。

## 受入の記録

| 項目 | 確認結果 |
|---|---|
| 1 | 着手前の検索と読み合わせは「経過と結果」のとおり。 |
| 2・3 | 旧表記と接頭辞の無いキーは 3 検索とも 0 件。コードマップ、両節のテスト一覧、DropMerge、プロファイル名、設定キーを確認した。 |
| 4 | 設計200の §2.4・§3・§5・§6.1・§7.2 で、Actor の方策生成と所有、Agent の target 方策と Learner の参照、PER 有効かつ actor_approx のときのヒントを照合した。§2.1.11 と ReadTargetPolicy を照合し、Greedy と epsilon 0 の既定、楽観設定のコピー・WARN・値の初期化、上書きと spatial.enabled を確認した。§2.4 の τ の既定は dqn_config.hpp、ReadTargetPolicy、agent.txt を照合した。通常 Q の計算を持つ dqn::Learner、具象型 TDLearner、二つの GenerateTaus をコードと照合し、通常 Q の型を示す元の 2 行が一致した。 |
| 5・6 | 設計200 §11.2 に ADR 22 本、設計214 §8 に 2 本。番号、掲載順、対応節を D15・D16 と照合した。ADR 0010・0012・0013 の既存行に削除は無く、追記は冒頭の各注記 1 行と空行。 |
| 7 | CONTEXT.md の TauGenerator は 0 件。 |
| 8 | 実装後のハッシュは、追跡ファイルの差分 `199792ee19b31490a7b66223d96d46a90cb7b855`、未追跡ファイル一覧 `678f384d8f22d7f871d8ffbee7099c116db99fb4`。下記の比較で、着手前のハッシュとの差を説明できた。 |
| 9 | 相対リンク切れは 0 件。課題 8 件の対応内容に、その件で行った修正と確認、実装記録へのリンクを追記した。レビュー後、8 件すべての AI 判定 `残件無し` と派生一覧への反映を確認した。 |
| 10 | src・apps の旧名は 0 件。差分で診断文 4 か所の文字列が `GenerateTaus` で始まることを確認した。検証関数の定義・呼び出し 3 行とテスト名 14 行を合わせて計 21 行。両 C++ ファイルを実装前の基準と全文比較し、名前の置換だけで一致した。 |

追跡ファイルの差分は、着手前と同じ基準 62da1be5 に固定して再計算すると `d5e76adcfe4d24b4ef8ca754419592232fbf4154` で一致した。比較時の HEAD 69927612 との差は別作業の PRD191 とその課題・wiki の差分である。除外外で残る既存差分は data/workspaces/metrics_viewer_tag_sets.json と docs/anet.asta。未追跡一覧は、別作業で追加された ADR 0082 と PRD192 の各 1 本を除くと `e92ef8801b903b730c29e9d11ef662a61ae4352b` で一致した。

自動照合は `.\.venv\Scripts\python.exe .scratch/prd190/verify.py` で通過した。出力は .scratch/prd190/acceptance.log に保存した。git diff --check で空白エラーは無かった。

030 の IQN 探索の案内は、同文書 §6.7 とそろえ、プロファイルの確認先も resolution に直した。config_data.txt は選択の結果として解決された設定値を確認するために使う。

### テスト駆動で確かめた結果

- RED と GREEN は `cmake --build --preset x64-Debug --target anet-algo-dqn-test`（VsDevCmd.bat 経由）でビルドしたあと、`out/test/Debug/anet-algo-dqn-test.exe '[tau][validation]'` を実行した。ビルドは両回成功。RED は両方の例外メッセージに旧名があり、部分文字列を比較する 2 assertion が失敗した。GREEN は両方の例外メッセージに `GenerateTaus received sample_mode=invalid` が含まれ、2 assertion が成功した。診断文を確かめるための一時変更は取り除いた。
- `out/test/Debug/anet-algo-dqn-test.exe 'GenerateTaus*' --list-tests` は改名前に 0 本だった。各実行のログは .scratch/prd190/ の red-build.log、red-test.log、green-build.log、green-test.log、test-names-before.log。

### 最終版の検証

- 受入 11。VsDevCmd.bat 経由の `cmake --build --preset x64-Debug` は終了値 0。`out/test/Debug/anet-algo-dqn-test.exe 'GenerateTaus*' --list-tests` は 14 本。
- `out/test/Debug/anet-algo-dqn-test.exe '[tau]'` は終了値 0、14 本・96 assertion が通過。`out/test/Debug/anet-algo-dqn-test.exe '[tau][cuda]'` は終了値 0、CUDA 上で 1 本・30 assertion が通過した。
- ビルド・テストの詳細は `.scratch/prd190/final-build.log`、`final-tau.log`、`final-cuda.log`、`test-names-after.log` に保存した。ビルドとテストは標準出力と標準エラーをファイルへまとめ、終了値・末尾を確認した。

### 自己点検

- C++ では `TauGenerator` を `GenerateTaus` に、`ValidateTauGeneratorArguments` を `ValidateGenerateTausArguments` に変更した。入力検証の条件・失敗の時機・エラーの詳細、テストの実行条件、付与タグと本体は元のものと一致した。診断文は英語で、対象名を現行の呼び出し関数とそろえた。
- 対象 10 ファイルが LF で、既存 9 ファイルの BOM の有無も基準コミットと照合した。
- PRD §4 D17 により英訳は別途翻訳する。英訳に残る旧名と、wiki の課題一覧が単票の題として引用する旧名を確認した。コード、設定、対象の和文、用語集に旧名は無い。
- レビュー観点の全項目を突き合わせた。今回の実装で確かめられなかった検証は無い。

## 完了報告

- 2026-11-20、PRD190 の実装と検証を完了した。和文 4 本と ADR 3 本を更新し、DQN の C++ 2 ファイルで τ 生成の名前をそろえた。
- 実装計画と実装は独立レビューの初回で OK。診断文の確認は RED から GREEN まで行い、最終差分は予定の 21 行と一致した。レビュー原文は `.scratch/prd190/review-20261120-0002.md`。
- 対象 8 件の AI 判定を `残件無し` とした。`wiki_support.py maps` を実行し、単票、課題 INDEX、wiki の派生一覧に反映されたことを確認した。
- 新たに判断が要る項目は無い。D17 の英訳 4 本は、後続の翻訳工程で和文へ追随させる。
