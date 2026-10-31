# TempoRL 著者実装との整合 実装メモ

## 概要

[PRD103](103_temporl_reference_alignment_10prd.md) を正本に、loss ごとの Adam 更新 `split`、skip のサンプル重み `kmax`、greedy 評価 session と判断率を実装する。`stop` / `merge` と `uniform` の学習計算・RNG・保存形式は維持する。GridMaze の skip 方策だけは確定済み D3 に従って epsilon-greedy へ移す。

対象はユーザーが 2026-10-28 に確認済み。基準コミットは `e0fc65f401844a3e6aeca3d1b92c3c2bd0a3f386`。着手時の Atari.txt の grad_clip_tau の説明変更、11/13_batch_run.bat、PRD921 および untracked は既存変更として保持する。

## 未決事項監査

- ユーザー判断が必要なブロッカー: 0。
- repo evidence で解決済み: `DefaultDQNAgentConfig` は共通の `net.block.*` を skip_net へ供給する。`NetworkModel` は観測・primary 公開出力・taus の参照鍵から skip の入力を作る。T3/T4 に新しい設定記法は要らない。構築・forward の確認は下の最初の検証に含める。
- repo evidence で解決済み: skip 有効時の FP16 AMP は現行契約で拒否済み。split も FP32/BF16 を対象とし、この制限を保つ。
- repo evidence で解決済み: `Learner::Optimize` は TD/QR/IQN 共通、GNS の入力は primary の weighted_element_loss。`Actor::GetScalar` は現在 policy に委譲しており session 用の鍵をここで扱える。
- repo evidence で解決済み: ADR0062 の注記と PRD098・第三弾の注記はすでに存在する。完了済みの改訂を重ねない。
- 2ximpl に固定: optimizer の追加状態は Learner が所有する private Resource。primary の通常・sigma をまとめ、decay 0 の追加 optimizer にする。既存 optimizer は primary Q と skip Network を引き続き所有する。
- PRD 範囲外: 第三弾の全 k・割り込み、専用 replay、専用 lr、Munchausen/可塑性の追加対策。既定の採否を決める長時間実験は段別の設定と判定手順を準備し、短縮受入と混同しない。

## 主な変更

1. `core/anet-core/testdata/prd103/` に検証ドライバと README を置く。コード・設定変更前に binary と設定一式を snapshot し、hash と基準コミットを記録する。旧 binary＋旧設定、新 binary＋旧設定、新 binary＋現行設定を分ける。評価停止対象は各入力を resolver に通した結果から列挙する。既存 prd098 の凍結入力は変えない。
2. 公開ヘッダ `include/anet/agent.hpp` の LearnerConfig に `split` / `sample_weight` を定義し、`include/anet/default_dqn_agent.hpp` で読む・検証する。skip 無効でも列挙値を検証し、未知値はキー・値・期待値付きで失敗させる。設定の既定は agent.txt に `?=` で置く。
3. `src/dqn_based_agent.hpp/.cpp` の Learner で、同一 forward の Q loss と重み付き skip loss を別々に微分する。両勾配を取り終わるまでパラメータを更新しない。primary Q、primary skip、skip Network の順にそれぞれの norm を計算・clip する。既存 optimizer の step の後、保存した skip の primary 勾配を割り当てて追加 optimizer を step する。primary の decay は Q 側で一度だけ。追加 optimizer の lr/eps/betas は primary と同じ。スケジュールと Load 後の options 上書きも両方へ適用する。
4. 既存モードの Optimize は従来の loss 合算式・演算順を保持する。GNS は引き続き Q loss のみ、policy churn は2回の optimizer 更新全体の前後で測る。追加 norm/clip 指標は dqn::BatchUpdateResult に保持し、非 split は既知の NaN とする。
5. split の checkpoint に primary 全パラメータの skip 側状態を表す役割 `primary_skip` を追加し、追加 optimizer を保存する。4役割の archive から split は追加状態を初期化し一度 WARN。5役割を stop/merge へ読むと失敗。保存順の役割照合、Noisy sigma、現在 options の適用を検証する。既存4役割のバイト形式は変えない。
6. `ComputeSkipLoss` は kmax のときだけ行ごとの loss に `max_lengths / mean(max_lengths)` を掛ける。IS weight と併用し、TD と priority の定義を変えない。
7. Actor は判断した長さの histogram と lane step 数を Sync 間で保持する。histogram は action と同じ device に置き、毎 step の CPU 読み戻しを増やさない。GetScalar の要求時に判断率・平均・長さ別率を返し、Sync で累積を消す。未成立・無効は NaN、未知鍵は既存 policy へ委譲する。
8. PRD §5.7 の設定・ツール・テスト・現行ドキュメントを移行する。Atari の session は greedy、学習 tag の添字は2桁。各 env の greedy session、評価指標と報酬、GridMaze の探索葉を追加する。Atari の既存コメント変更は保持する。過去 Run と測定記録は変更しない。
9. T3 は primary の畳み込みを trunk として公開し、skip 側で tau を融合する独立 profile、T4 は skip 側に観測からの畳み込みと tau 融合を持つ独立 profile とする。既定の構造は維持する。T2/T10 は既存の葉による腕として検証ドライバへ置く。
10. 設計200・140（対応する英語版があれば同様）、CHANGELOG、wiki の現行記述を今回の結果に追随する。PRD の歴史的な調査結果や実験結果は書き換えない。

## テスト

移行する旧名→新名の一覧（現用範囲）:

- Atariの`greedy_dist`→`greedy`: `run.eval.[tag]`、`run.eval_schedule.[tag]`、`$eval.[tag]`、actor catalogの対応tag、atari5.pyの`eval_name`。評価条件と`53_evalg`群は維持する。
- tagの`03_skip_length_rate.[1]`～`[9]`→`03_skip_length_rate.[01]`～`[09]`（各EnvのJ内）。`[10]`とsource keyの`skip_length_rate.[1]`～`[J]`は据え置く。metrics scalar定義、observerテスト、現用検証器と日英設計文書を同時に移行し、過去Runは保持する。

Public interface / surface: `DefaultDQNAgentConfig`、`DefaultDQNAgent` の構築・CreateActor・学習更新・Save/Load、`Learner::UpdateFromSamples`、Actor の MakeAction/Sync/GetScalar、Runner の解決済み設定・metrics.jsonl・metrics.scalar.defs。

TDD は一つの behavior ごとにテスト追加→実際の RED→最小実装→GREEN の順で行う。全テストを先に書かない。

1. tracer bullet: 小さい共有 Network の公開学習経路で split の1更新を行い、独立計算した Q/skip の Adam 更新の和と一致する。未対応設定で RED を確認して設定と split 更新を接続する。
2. clip の3分割、weight decay の一回適用、Noisy sigma、BF16/fused、QだけのGNS、split以外のNaNを順に確認する。stop/mergeの既存テストを維持する。
3. kmax の異なる3行の数値例、重みの和=B、IS重みとの積、uniform維持、不正値・無効時検証。
4. split checkpoint 往復と継続更新、merge→splitの初期化、split→merge拒否、options上書き。
5. Actor の複数lane・episode開始・判断なし・Sync前後の割合と平均。未知鍵と既知未成立の区別。
6. T3/T4 の設定を resolver と Network 構築・forward に通し、共有/非共有と IQN 出力形状を確認する。
7. 4 env の設定・metrics、tag の移行と評価の学習側step軸を公開 Runner で確認する。

## 検証

- baseline は変更前 binary・config・hash を保存し、4 env skip無効と GridMaze/LunarLander skip有効の6構成を取得する。8 lane、batch16、warmup256、8704 experience、FP32・決定論的backend、評価停止で、性能/資源を除く全scalarを完全一致で比較する。
- Debug 全体ビルド、`[prd103]`、既存 `[prd098]`、関係する DQN/NN/observer/replay 契約を実行する。不要な繰り返しは避け、失敗の終了コードとログを保存する。
- 設定の全キー・文字列値比較と `check_default_leaves.py`、旧識別子の現用範囲検索、差分と文字コードを確認する。
- GridMaze/LunarLander の greedy 指標・runner=train、Atari の greedy session の smoke を行う。変更後のskip有効の値をREADMEに新基準として残す。
- Release の Atari BTR（N128/B256/BF16/prefetch）で merge/split の逆順4Runを測り、64〜256更新の実時間比をREADMEに残す。競合する学習・ビルドがある間は性能計測を開始せず、その条件を記録する。閾値は追加しない。
- 失敗・未実施は成功扱いせず、理由と残作業を本書に追記する。自己点検後、独立コンテキストの an-review へ計画と実装を出し、差し戻しへ対応する。コミットしない。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core\anet-core\bin\Debug\anet-core-test.exe "[prd103]"
core\anet-core\bin\Debug\anet-core-test.exe "[prd098]"
```

## 前提

- 新規公開キー・tag は PRD の表に限定する。追加のライブラリや汎用NN記法は作らない。
- split の共有部 backward は更新ごとに1回分増え、勾配一時保存と Adam の追加状態を要する。既定の学習順は変えない。Actor の集計は device 上の J 要素の histogram に加算し、stepごとの新しい同期転送は0回とする。
- T3/T4 の profile は既定構成の checkpoint と互換にしない。T3 の primary の名前が変わることをREADMEへ記す。
- PRD §9 の長時間実験後の採否、性能閾値、機序の問いは実装を止めない。短い受入から学習効果を断定しない。

## 経過

- 2026-10-28: PRD103 を対象として確定。監査と現行実装の照合を行った。まだコード・設定の変更、ビルド、Run は行っていない。
- 着手前の binary/config を `testdata/prd103/baseline-*` へ保存。84 files の hash と HEAD を manifest に記録し、設定 snapshot テストは15 assertions成功。6構成の baseline は `.scratch/prd103/validation/` へ順次保存中。
- split 更新の RED: 共有部2 tensorsだけが独立 Adam の和と不一致。最小実装後は6 assertions成功。checkpoint 継続の RED: role count 4（期待5）。追加状態の保存・読込後は13 assertions成功し、次の更新も完全一致。
- kmax の RED: k_max=[1,2,3] で loss=1.5（期待1.8333）。最小実装後、uniform/kmax × PER有無の4条件、12 assertions成功。PERのテスト入力は公開Pushで世代を作り、実在する更新先を使う。
- Actor session の RED: 新しい既知キーにnullopt。device上の累積とSyncリセット後、27 assertions成功。2 laneの途中episode開始、判断なしのstep、長さ別率、無効/未成立NaN、未知キー、添字違反を確認。
- split指標のRED: normキーにnullopt。追加後、clip有無×標準/fused AdamWの4条件、68 assertions成功。3集合の独立clip、decay一回、checkpoint後の継続完全一致を数式から確認。
- baseline6構成がすべてexit 0で完了。正規化点数はGrid OFF26946、Lunar OFF24399、Atari OFF23185、Drop OFF31152、Grid ON37977、Lunar ON34628。各hashと実行引数は `.scratch/prd103/validation/before_*/` に保存。

## 検証・レビュー結果

上記の数値検証は `VsDevCmd.bat -arch=x64 -host_arch=x64` 経由の `cmake --build --preset x64-Debug --target anet-core-test` と `[split_numeric]` / `[skip_kmax]` で実施。

- Debug全体buildは最終実行でexit 0。直前のLNK1104は自分のsmoke/resolverがexeを使用中だったためで、全検証終了後にリンクを再実行した。Release Runner buildもexit 0（その後の診断文修正はDebugで確認）。
- `run_validation.py compare --label after`: 6構成すべて完全一致。固定入力・hash・点数は検証READMEに記載。
- `run_current.py smoke --label smoke1 --audit`: GridMaze/LunarLander/Atariすべて成功。新指標の実点とdefsのrunner=train・exp_step、actor/session・Grid方策・Atari改名を確認。
- `verify_config_migration.py`: 8構成すべて意図しない全キー・文字列差分0。`check_default_leaves.py`: 既存Atari fpsの2指摘（変更前snapshotにも存在）。新しい違反0。
- `prepare_experiments.py`: 段1～4の24入力すべてresolver成功、学習Run開始0。T3/T4は独立profileで既定構造を保持。
- Python5ファイルのAST検査成功、git diff --check成功。現用の旧session名検索は移行を検証するnegative assertionだけ。設計200/140の日英、CHANGELOG、wikiを追随。
- 最終の`[prd103],[prd098],[gradnoise],[observers],[observer_factory]`は87 test cases / 36239 assertions成功、`[prd103-arms]`は1 case / 15 assertions成功（.scratch/prd103/final-tests.jsonとfinal-*.log）。自己点検はdocs/quality/impl.mdの全項目を走査済み。該当しない新クラス・新C++ファイル・借用API・UI・skill項目は対象外。追加ResourceはLearner所有、Actorの累積はdevice上で更新し、GetScalarでだけCPUへまとめて読む。
- `run_current.py performance --label perf-ready --prepare`: N128/B256/BF16/prefetchとmerge/splitの4入力をresolverで検証成功。学習は開始していない。

## 決まらなかったこと

- 性能受入4は未解決。既存AnetRLRunner_ab（Atari5-01、qbert 50M、loss_weight=0.3）が稼働中のため、競合下の測定を避けた。`run_current.py performance --label perf-acceptance`はprocess guardで測定前に停止。4 Runの実時間比・README実測表・PRD §9.4への比の転記が残る。質問していない（既存Run停止は要求されておらず、性能比較は終了後に実施可能）。
- split/kmaxの既定採用、長時間学習の機序・性能閾値はPRD §9のまま。質問していない（本実装のブロッカーではない）。
- 独立レビュー初回: 計画OK、実装は2件差し戻し（原文.scratch/prd103/review-20261028-1.md）。旧→新一覧を追記し、実験レシピのmetrics選択をenvのrun.@skipから継承するよう修正。追加前後の全scalar定義を比較する検査で24入力すべて再解決成功（errors=0）。旧レシピはexperiments-review1へ保持。
- 独立レビュー2回目: 計画・実装ともOK、修正要求・範囲外検出・開発者への確認事項なし（原文.scratch/prd103/review-20261028-2.md）。性能未実施は成功扱いしない。

## 完了報告

実装・機能検証・独立レビューを終了。split更新、kmax重み、greedy session指標、設定・現用ツールの移行、T3/T4と長時間実験入力、日英設計文書・CHANGELOG・wikiを更新した。既存の未コミット変更は保持し、コミットしていない。

PRD103全体の受入は未完了。残る受入4は既存Atari Run終了後のRelease性能4Runと比の記録。採否・機序の長時間実験は実行せず、PRDの未決を維持した。判断してほしいことは現時点でなし。今回の手戻りから改善案を整理する場合は`$anet-improve`を使える。
