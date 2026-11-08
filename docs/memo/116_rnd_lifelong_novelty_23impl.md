# RND の公開実装に合わせた選択肢とNetwork修正 実装メモ

> 初回実装は [20impl](116_rnd_lifelong_novelty_20impl.md)、採点・学習の独立化は [21impl](116_rnd_lifelong_novelty_21impl.md)、経験件数基準の刻みは [22impl](116_rnd_lifelong_novelty_22impl.md)。

## 概要

[PRD116](116_rnd_lifelong_novelty_10prd.md) と `.scratch/prd116/request-20261107-reference.md` に従う追加実装。基準は着手時HEAD `4005f4de`。既定のwhiten・clipped・next_stateを維持し、別々のtarget/predictor、Head初期値、Taigaの統計・観測・倍率・Adam epsを設定で選ぶ。PRDは入力資料として保持する。

## 主な変更

- `apps/runner/config/nn.txt` のConv1d/Conv2d S/Mの葉をconv.*へ修正。歴史的Run設定の写しは保持する。
- `core/anet-core/src/intrinsic_rnd.hpp/.cpp`: target.net/predictor.netの必須設定、head.init、multiplier_mode、statistics_mode、adam_eps、推定器内のVarianceFeedbackStatistics、targetの累積出力統計、書式3とモード一致検査。共有のNormalizer/RunningMeanStdは変更しない。
- `core/anet-core/include/anet/default_dqn_agent.hpp` と `src/default_dqn_agent.cpp`: intrinsic.observationを設定し、stateではmaskなし。bonus_min/bonus_pos_ratioと絶対値基準のbonus_share。credit・抽選・呼び順とlock範囲は維持。
- `apps/runner/config/agent.txt`、Atari/GridMaze/LunarLander: 新契約の既定、2ネット、診断4つ。AtariはAtariRndConv1/2/3・FC・Leakyのカタログとrnd_taigaを追加。既存の比較用プロファイルとbatは保持する。
- `core/anet-core/src/intrinsic_test.cpp` と出荷設定テスト: 公開API・設定・checkpoint・診断でPRDの追加8行と改訂3行を確認する。
- `docs/design/212_rnd_lifelong_novelty.jp.md`、200日英、212英語版の未訳表示、`core/anet-core/testdata/prd116/README.md`、索引issueの対応内容を更新する。212英訳本文の新規翻訳は別途。
- 追加依頼request-20261107-runname.md: PRD098検証ドライバのRun名接頭辞をRUN_PREFIXへ集め、PRD116 wrapperからprd116を設定する。対象はtestdata/prd098/run_validation.py・testdata/prd116/run_validation.py・同README・本記録。既存Run・基準・r1〜r3・11/06の記録・profileと計測tagは保持し、新しいRunを起こさない検査で名前を確認する。
- 追加依頼「GridMazeだけ実施」: `core/anet-core/testdata/prd116/run_performance.py` と同測定器の `run_performance_test.py` を追加し、同README・212日本語版§5.1/§5.2・本記録・索引issueの対応内容を更新する。`docs/experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md` と同名JSON、config配下の15本の実効設定、data配下のTracy CSV3点を保存し、wikiのintrinsic_novelty・index・logへ取り込む。§6.3のGridMaze分を測定し、Atariは起動しない。

## テスト

- 面: Factory/Create、Estimate/Learn/Diagnostics/Save/Load、DefaultDQNAgentConfig、CreateLearner/UpdateFromBatch/GetScalar、出荷設定の解決とNetwork構築。
- TDD1: 2ネットの必須設定と構築・採点・学習を1テストでREDにし、分離設定とHead初期値の最小実装でGREENにする。既存テストの設定を同じ契約へ移す。
- TDD2: clipped/unclipped、whiten/variance_feedbackの手計算3stepとN=2の順序、warm-up、targetの分散を確認し、統計と倍率の実装を通す。
- TDD3: state観測とdone mask、負ボーナスの診断、WARN、Adam eps、書式3とモード不一致・保存継続を各behaviorでRED→GREENにする。
- TDD4: 層カタログの1D/2D寸法、Atariの2ネットのパラメータ数、Taigaの8葉と既定一致を出荷設定で確認する。現用の旧契約の参照を除く。
- 各サイクルは該当テストを先に失敗させ、最小実装の後に通す。設定変更の前後は全キー比較を行い、別作業の変更を保持する。

## 検証

ユーザーが2026-11-07にRunの切れ目確保、設定変更・ビルド・テスト可と回答した。詳細stdout/stderrは `.scratch/prd116/reference-*.log` に保存し、終了値と失敗要約だけ取得する。待機は30秒を基本とする。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
core/anet-core/bin/Debug/anet-core-test.exe '[prd116],[scaler],[shipped_config]'
```

CNNのCPUテストは当該プロセスだけ `OMP_NUM_THREADS=1` とする。Debug libtorchのoneDNNのthread assertionは既存CNNテストにも記録されており、指定なしで再現、指定ありで成功した。本番backendは変更しない。

設定解決は `.scratch/prd116/reference_resolve.py` と[prd103-resolve]で、保存済み `apps/runner/workspaces/Atari-04/runs/run_20261106-234258_mz_rt_base_s1/config/config_data.txt` と全キー比較する。BOM・LF・旧名・リンク・定数・規約も確認する。無効時checksumは4環境、smokeはPRD §6.4の3群（GridMaze、Atari KFM、Montezuma）である。性能測定はユーザー確認が必要なので実施しない。

## 前提

- 実装上の未決0。VarianceFeedbackStatisticsはRNDの非公開の状態とし、観測の各fieldとscalar誤差をCPU doubleで逐次更新する。lane間の統計依存があるため、各laneの正規化は直前までの統計を使う。
- state.obsはTrainerのstate_.Clone()で、Actorのframe stackはMakeAction内の処理。生の最新観測を保持している。
- 2ネットの設定の共通参照はresolverで実際に確認する。参照が解決しないときだけ同じBodyを2か所に宣言し、その理由を記録する。
- 入力shape・mode・既存Normalizer/初期値の共通検証は所有する機構で検証する。ignored葉の比較は指定経路に依らず解決値とRND既定を比較し、構築時に葉ごと1回WARN。
- 既存の別作業（MetricsViewer、Atari比較設定、bat、asta、未追跡資料）とindexを保持する。PRD079のnamespace規約に従う。
- cc-reviewの両モデルへ同じ依頼を渡し、ユーザーの指示によりNGはまとめて修正して再レビューする。実装方針の判断が必要な場合だけ確認する。

## 契約の移行

| 旧 | 現行 |
| --- | --- |
| intrinsic.rnd.net.* | intrinsic.rnd.target.net.* と intrinsic.rnd.predictor.net.*（両方必須） |
| Conv1d_S/M・Conv2d_S/Mのout_channels/kernel_size/stride/padding | 同じblockのconv.out_channels/conv.kernel_size/conv.stride/conv.padding |
| RndNoveltyEstimator-2 | RndNoveltyEstimator-3とstatistics_mode |

旧契約の別名・互換読み・特別WARNは追加しない。旧識別子は拒否テストだけに残る。212英語版本文はユーザー指定により旧翻訳を維持し、先頭でprogress:chapter 0/5と更新待ちを明示する。Run設定の写し・履歴資料は保持する。200英語版は今回の4節を日本語と対応させ、既存の検証済みblobから変わった範囲だけ翻訳した。

## 追加費用の見積もり（未実測）

- target出力の累積分散のため、EstimateごとにN×output_dim×4 byteをCPUへ戻す。Atari N=128・d=512なら256KiB/intake。転送帯域を仮に10GB/sとするとデータ量だけで約0.026ms/intakeで、呼出頻度F回/秒なら0.026×F ms/秒。GPU同期・起動の待ちを別に仮定した10〜100μsを加えると約0.036〜0.126ms/intake。これは下限に近い概算で、混雑・Debug版・演算依存は未評価。
- facadeのbonus_min・bonus_pos_ratioはintakeごとに各1回のscalar読み出しを増やす。CUDA入力なら2回の同期があり、1回10〜100μsの仮定で0.02〜0.2ms/intake（F回/秒なら0.02F〜0.2F ms/秒）。CPU入力ならdevice転送は発生しない。laneループ内のdevice転送は追加していない。
- variance_feedbackの観測統計はCPU doubleでN件を順に更新する。観測の再帰依存があるためbatch平均へ置換しない。誤差とtarget出力はループ前にまとめてCPUへ戻す。採点forwardはbatchのまま、Learnは更新後の統計を読むだけである。
- Taigaのratio=1・k=1はN回のforward対・backward/optimizer step/intakeで、既定N=128・ratio=.25・k=32の1回と異なる。既存の各Learnの転送境界を維持する。TracyはEstimate/Learnの正規化・転送・forward・統計・backward・optimizer、facadeの抽選・学習を分けている。速度の合否は未設定・未測定。

## 本家との残る差

| 項目 | Taigaら | anet |
| --- | --- | --- |
| Qの手法 | C51（51 atom、±10） | IQN。C51はない |
| 畳み込みのpadding | SAME（Flatten 7,744） | VALID（Flatten 3,136）。Burdaらと同じ |
| Qの入力の埋め込み | IQNでもDense 512はτの後 | AtariNatureはτの前にもLinear 512 |
| Qの初期値 | variance scaling（1/√3、fan_in、uniform） | he・xavier |
| PERの細部 | √priority、Dopamineの重み正規化 | anetのPER |
| Learnの入力 | 採点と同じ正規化済み観測 | 統計更新後に正規化し直す |
| 統計のn | agent step（エピソード開始でも増加） | 採点した観測件数 |
| 実行環境 | TF1、当時のALE | anet、ALE 0.12 |

## 経過と結果

- 着手時に依頼と改訂PRD・22implを照合。Run境界の回答を受領。実装前に計画を保存した。
- TDD1 RED: Debugのtest targetビルド成功後、新しい2ネット設定だけのFactoryテストが旧rnd.net要求の例外で失敗（1 test /1 assertion、終了42）。別々のNetworkConfigへ変更したGREENは1 test /2 assertions成功、終了0。ログはreference-net-red/green-build/test.log。
- TDD2開始: 既存テストの共通設定を2ネットへ移し、unclippedの負ボーナスを公開Estimateで確かめるテストを追加。GridMaze/LunarLanderは共通のRndMlp branchを両方から参照する設定へ移行し、解決の確認は後続で行う。
- TDD2 unclipped: 旧clampによるα=1（期待約-0.5669）をREDで確認し、GREENは2 tests /5 assertions成功。variance_feedbackは手計算3step・N=2・保存継続・mode拒否を追加し、旧実装で20 assertionsの不一致を確認した。共通RunningMeanStdのGetVarは不偏分散なので、target診断だけn−1/nで母分散へ変換し、GREENは[prd116] 28 tests /1360 assertions成功。共有Scalerは変更していない。
- TDD3 state観測: doneでbonus=0になる旧挙動をREDで確認し、選択したstateをEstimate/Learnへ渡す変更後は[prd116] 29 tests /1372 assertions成功。optionalの未成立値をテストで参照する前にREQUIREを追加した。WARN回数、無効時の観測設定非解釈、eps、未知mode・head初期値・欠落roleの拒否も確認した。
- TDD4 出荷設定: 旧Atariのtarget.net不足でRED。新しい専用CNNと2つのBody、カタログのconv.*、Taigaを追加し、実物のcheckpointから数えたパラメータとoptimizer stateを含む約33.2MBの増分が期待どおりでGREEN（21 assertions）。CPUのoneDNN assertionはOMP_NUM_THREADS=1で回避した。
- Debug全体ビルド成功（reference-full-build.log）。最初の関連スイートは48 tests /4058 assertions成功。Head初期値の重み差とstate/next_state両方の非重複学習を補い、test target再ビルド・関連49 tests /4221 assertions成功（reference-coverage-build.log、reference-coverage-suite.log）。CPU/CUDA両入力を含む。以後の修正と最終確認は「2モデルレビューと修正」に記録する。
- GridMaze/LunarLanderの共通branch参照は両方resolver成功。実際のMontezuma保存設定との全キー比較はbase75差分・rainbow225差分・taiga247差分を全保存（reference-resolve/comparison.json）。現在のbaseの差はRND契約・共通Conv葉・新カタログ・既存のRAM診断追加と実行先など。rainbow→taigaの31差分はRND18葉＋診断13葉のみで、Q側のnoop=0・Nature構造・Munchausen指標なし、Taiga8値を確認した。従来のrainbow_like / no_munch / batは変更していない。
- check_default_leaves.pyは2件で終了1。Atariのapp.online.eval_panel.fps・train_panel.fpsは22implで既知の既存違反で、今回増分は0。全キー比較と[shipped_config]は成功。
- 無効時checksumと3群smokeを実行中。ログはreference-off-run.log、reference-off-compare.log、reference-smoke-{env}.log、Runと集計はbaseline-20261106/reference-*に分離する。既存の実行体・Runは上書きしない。検証用の8 lane/Q batch16/capacity4096/warmup256/RR1/評価休止は記録し、RNDのTaiga8葉は保持する。
- 無効時4環境は完了。GridMaze 26,946点、LunarLander 24,399点、Atari 23,185点、DropMerge 31,152点、計105,682点のchecksumとmetrics.scalar.defsが一致した（reference-off-compare.log、終了0）。smokeは継続中。

## 自己点検

docs/quality/impl.mdの全75項目を走査した。機械検査はreference-static.jsonとGit差分、設定・受入は上記テストと解決結果を使う。

| 項番 | 判断と根拠 |
| --- | --- |
| 1〜7、62、71、72 | ヘッダusing/dynamic_cast/TODO、計画名・入力元、差分、UTF-8/LF/BOM、旧契約、文字列候補を確認。設定キー・ログ・単発期待値・1か所だけの列挙値を除外し、複数回使う列挙値とfeature・role・ReLUの識別子は定数を共用。初回は列挙値の除外範囲を誤ったため、レビュー指摘で是正した。212英語本文の保留は明示的な依頼による |
| 8、53 | skillの変更なし |
| 9〜17、74 | 23implを先に保存し、TDDのRED/GREENと優先behaviorを記録。PRDは未変更。索引issueへ実装・確認結果を追記する。性能は未実施、長時間検証は進行中として明記 |
| 18〜22 | 2ネットとconv.*へ移行し、互換層なし。歴史資料を維持。212英訳保留だけは依頼された例外 |
| 23〜28 | shapeは[B]+specと[B,F]、mode・値域は構築時。Normalizer/初期値は共通検証を呼ぶ。warm-up/無効は意図した休止。新しいworker・retry・fallbackなし |
| 29、54、67〜70 | 共通の既定は?=、Run・実験選択は=。新tagとrnd_taiga・出荷設定の数値固定は依頼項番8・10・11の明示指定を優先（一般規約の番号・変種profile・env値固定の例外）。既存profileのmetrics連鎖を変えず、RND13項目の追加を全キーで確認 |
| 30〜35、55〜60、63〜66、75 | 新C++ファイル・public API追加なし。小さな状態はRND内の非公開helper。namespaceは型の既存区画に合わせる。設定はANET_READ_CONFIG、指示付き初期化、include順、状態と資源の所有、static純粋helper、初期化順を確認 |
| 36〜41、61 | 日本語の段階コメントと既存TODOを維持。頻繁な経路の計測境界と転送の回数・概算を記録。lane内のdevice同期なし |
| 42〜45 | 新診断を構築時に登録し、未成立NaN・無効nulloptを維持。UIはatomic読むだけ。統計・optimizerはRND、刻み・観測選択はfacade。ログ英語 |
| 46〜52、73 | 公開Factory/agent/checkpointと設定解決で確認。手計算入力・回数・観測対象・拒否条件をPRD §6.2と照合。文書212JP・200日英、212EN保留、検証資材READMEを追随。性能未実施と進行中の検証を成功扱いしない |

## 2モデルレビューと修正

- 初回（20261107-1840）: 計画はCodex・ClaudeともOK、実装は両方差し戻し。原文は `.scratch/prd116/review-20261107-1840-impl.md` と `-claude.md`。Claude費用概算2.1756639 USD。
- Codexのみ: variance_feedbackがカテゴリIDも正規化し、合法なID 2→0が負IDになってone-hot化で落ちる。公開Factoryのカテゴリ/連続混合でRED（66 assertions中30不一致、Class values must be non-negativeを再現）。共通Normalizerと同じくカテゴリ・非浮動小数点をそのままNN境界へ渡し、連続値だけを再帰へ入れる。byteの非カテゴリ画像は先にfloatへ変換する既存契約を維持。カテゴリ単独/混合とint64/uint8/floatの各IDでEstimate・Learn・入力不変・target出力・保存継続を確認する。
- 両方: RndMlpのプロファイル境界4行、繰り返す列挙値・ReLU識別子の定数化、feedbackエラー3箇所の実値不足を修正。shape/dtype/deviceと最初の異常値を出し、大きなtensor全体はログへ出さない。
- Claudeのみ: 最終の定数整理後の検証記録不足。整理前からの結果を継承するだけにせず、修正後にDebug全体ビルド・関連スイートを実行し、その結果を追記する。
- 修正後のDebug全体ビルドと[prd116],[scaler],[shipped_config]は成功（50 tests /4371 assertions、reference-review-fixes-build.log、reference-review-fixes-suite.log、両方終了0、OMP_NUM_THREADS=1）。カテゴリ単独/混合×int64/uint8/floatの6通りと保存継続を含む。定数整理だけの直前スイートも49 tests /4221 assertions成功（reference-final-test-build.log、reference-final-suite.log）。
- 同一行が複数あった列挙値の残りも全て定数へ置換し、最終のtest targetビルドと関連50 tests /4371 assertions成功（reference-constants-complete-build.log、reference-constants-complete-suite.log、両方終了0）。enumの直書きはconstant/ReLUの定義1か所ずつだけで、統計・倍率・観測は既存の定数を参照している。
- RndMlpの区切り修正前後は、resolverでGridMazeの822葉、LunarLanderの942葉が文字列値まで全一致（reference-branch-compare.log）。byte画像の変換と連続値の再帰は変えず、手計算3step/N=2・CPU/CUDAを含む既存テストも通った。
- 無効時checkpointを新しい実行体で比較し、header/config・Networkのバイトとoptimizerの内容が一致（reference-checkpoint.log、終了0）。既存libtorchのアドレス情報によりraw SHA256は一致しない。書き換えずに元ファイルと比較記録を保持した。
- smoke資材の初回GridMazeは、Run連鎖が1要素のケースの挿入を誤り、期待した保存先にRunが無く失敗した。run.$の代入と連鎖を分離する修正後、reference-smoke-r2で取り直している。reference-smoke-r1の失敗記録は保存した。テスト資材の失敗を本体smoke成功に数えない。
- Claudeの範囲外検出: Atariに既存のnet.branchプロファイル境界違反がある。今回のRND部以外の既存カタログで、変更対象へ広げていない。
- smokeはGridMazeがreference-smoke-r2、KFMがreference-smoke-r1、Montezumaが修正後実行体のreference-smoke-r3で進行中。GridMaze/KFMはwhitenであり、今回修正したvariance_feedbackのカテゴリ経路を通らないため、保存済み実行体のRun結果をそのまま検証する。Montezumaは新しい実行体で独立して開始した。どれもseed1・100k経験である。
- 再レビュー（20261107-1907）: Codexは計画・実装ともOK、Claudeは計画OK・実装1件差し戻し。追加指摘はtestの診断キーclip_ratioの定数化漏れだけで、kClipRatioへ集約した。ほかの初回指摘は両モデルとも解消確認。原文はreview-20261107-1907-impl.mdと-claude.md。Claudeログ表示の費用概算は3.9963219 USD。
- 最後の診断キー定数化後はDebug全体ビルド成功、該当するunclippedテスト1 test /3 assertions成功（reference-clip-key-build.log、reference-clip-key-test.log、両方終了0）。productionの処理は直前の50 tests /4371 assertions成功時と同一。Diagnostics().atのキー直書きは検索0件。最終の静的検査・差分空白検査も終了0、200英語版の翻訳元blobは日本語現内容と一致した。

## smokeの結果と未達の受入条件

元Runを変更せず、集計は `.scratch/prd116/reference-smoke-acceptance.json` に保存する。判定する点はlearn_count>0で、bonus_shareのNaN例外はPRDどおりである。

| 群 | 実行と観測 | 受入判定 |
| --- | --- | --- |
| GridMaze | seed1、100k経験、終了0。warm-up後5,448点、learn_count最大2,724。群56の不正な非有限値0。正のbonus_meanは17点（0.312041%）。前半平均8.6066401e-5、後半平均0 | 減少条件は満たすが、「正の点9割以上」に未達 |
| Atari KFM | seed1、100k経験、終了0。warm-up後5,448点、記録したlearn_count最大2,724。群56の不正な非有限値0、clip_ratio最大0 | 条件を満たす |
| Montezuma | seed1、100k経験、終了0。記録6,233点、learn_countの最大記録値99,984。bonus_minが負の点6,233/6,233（100%）、群56の不正な非有限値0 | 条件を満たす |

GridMazeの正の点9割という条件は未解決として残す。whiten・clipped・next_stateの計算式と統計を維持することも依頼の明示条件なので、この数値を満たすために式・既定値・受入閾値を変更しない。clippedのα=1は仕様上許された値であり、0のボーナス自体を計算異常とは扱わない。成績・探索効果の結論には広げず、受入条件と実測の不一致を記録する。PRD本文は別枠の入力資料として保持し、現時点で受入条件の全充足とは報告しない。

初回の順次起動スクリプトが修正後バイナリでMontezumaを旧ラベルreference-smoke-r1に起動しようとした際は、実行体のfingerprint検査で拒否された（reference-smoke-Montezuma.log）。旧snapshotを上書きせず、既に新ラベルreference-smoke-r3で実行している。GridMaze/KFMとMontezumaの並行中の実時間を群の性能比較には使わない。

3群ともRunnerの終了は0で完走した。smoke判定スクリプトの終了はKFM・Montezumaが0、GridMazeが条件未達を検出して1。最後の記録点は99,984経験であり、上表のlearn_countは記録点での最大値で、checkpoint内の最終累計回数と同じとは報告しない。性能のABBA・中央値・Tracy内訳は測定していない。

## 残る事項

- GridMazeの「正のbonus_meanの点が9割以上」は未達として索引issueの対応内容にも記録した。既定式の変更、設定を探して条件を満たす追加実験、PRDの受入条件の改訂は今回行わない。実装のレビューOKと、全受入条件の成功を区別する。
- 性能測定は実施確認が必要なため未実施。212英語本文は依頼どおり翻訳待ち（progress:chapter 0/5）。
- 既存のfps既定葉2件とAtariカタログのprofile境界違反は今回の修正範囲外。新規起票はしていない。

## 最終レビューと完了報告

2026-11-07 20:07の3回目のレビューで、Codex・Claudeとも計画・実装がOK。不足点・修正要求は0件。原文は `.scratch/prd116/review-20261107-2007-impl.md` と `review-20261107-2007-impl-claude.md`。Claudeログの表示額は5.9685523 USD。

Claudeの確認事項として、GridMazeの後半bonus平均0が想定どおりか、不具合の兆しかは未判断である。20implの旧smokeの最後の四分の一の平均5.93e-5との差には22implと23implが挟まり、保存記録だけでは原因を切り分けられない。今回のレビューOKは受入閾値の達成や、この差を正常と確定する判断ではない。原因調査と受入条件の見直しのどちらへ進むかは、結果を示して開発者の判断に委ねる。

追加仕様の実装、指摘修正、Debugビルド、関連50 tests /4371 assertions、最終キー修正の1 test /3 assertions、無効時4環境105,682点と指標定義、checkpoint内容、設定の全キー比較、3群各100k経験の実行・判定と静的検査を記録した。KFM・Montezumaはsmoke条件成功、GridMazeは1条件未達として保持。PRD本文と無関係な変更は保持し、ステージ・commit・pushは実施していない。

新たな起票はせず、今回の結果は既存の索引issueの対応内容へ反映した。索引のAI判定は既に「実装済み」、ai_judgedも同日であり、status・監理の項目・派生一覧の入力は変更していない。レビュー結果を記録するこの節と索引の完了行以外に、最終レビュー後の実装・設定・設計文書の変更はない。

## 追加調整: 検証Run名

request-20261107-runname.mdに従い、PRD098ドライバのRUN_PREFIX（既定prd098）をPRD116 wrapperでprd116へ差し替え、次に起こすRunから名前と検索globの接頭辞をPRD116にした。r1〜r3・11/06のRun・フォルダ・基準・記録・実行体の複製、run.@prd098_validationと90_perf/98_prd098_timeは保持する。今回は新規Runを起こさず、次に別目的でGridMazeの新ラベルを実行するときに実フォルダ名とexecution.jsonの引数を確認する。

検証は `.venv/Scripts/python.exe -X utf8 .scratch/prd116/runname_check.py`。wrapperの実際のmodule読み込みとRUN_PREFIX代入を通し、stage本体は実行せず、execute内の名前生成代入とglobの式だけを評価した。変更前はPRD116もprd098名になってRED（runname-red.log、終了1）、変更後はPRD098の既定名・PRD116名・glob・保存された旧REUSEパスが全て期待どおりでGREEN（runname-green.log、終了0）。Popenは禁止する差し替えを置き、Runner起動とRun artifactへの書き込みは行っていない。Python2ファイルの構文も成功。GridMazeの追加smokeと実際の新しいRun名の確認は未実施で、今回の名前変更の確認だけを理由に取り直さない。

同じMSVC初期化付きDebug全体ビルドも終了0（runname-build.log、ninja: no work to do）。編集したPRD098 PythonのCRLFをLFへそろえ、UTF-8・BOM保持・構文を再確認して終了0（runname-static.log）。名前の検査も再実行して終了0。全75項目の自己点検では、C++・設定・公開APIの変更は対象外で、今回は名前の共通化、差分範囲、改行・BOM、検証と記録、既存artifact保持を確認した。索引には今回の対応内容だけを追記し、監理の項目・statusは変えていない。

追加差分のレビュー（20261107-2023）はCodex・Claudeとも計画・実装OK。不足点・修正要求はなし。原文は `.scratch/prd116/review-20261107-2023-impl.md` と `review-20261107-2023-impl-claude.md`、Claudeログの表示額は7.9922205 USD。新Runの実フォルダ確認、GridMazeの以前からの未達と原因未切分けは残る。ステージ・commit・pushは実施していない。

## 追加調整: GridMaze smokeの判定

以下は共通検査を維持した最初の調整の記録であり、その後に開発者が選んだ有限性検査の追加修正と最新の再判定は末尾の「この場で直す」を参照する。

改訂済みPRD116 §6.4（blob 68f6e58d52d1b5b12984f720ce86918b2dfc04cc）とrequest-20261107-gridmaze-smoke.mdに従い、GridMazeの符号・減少条件を外し、完走・群56の有限性（既存のNaN例外を除く）・補助学習の進行で判定するよう修正した。対象は検証ドライバ・README・212日本語版§5.2・本記録・索引issueの5ファイル。観測が正規化した(x, y)の2値で盤面固定なので、限られた入力に慣れると既定のbonusが0になる理由を記した。`.venv/Scripts/python.exe -X utf8 .scratch/prd116/gridmaze_smoke_check.py` は実際の集計・判定部分を抽出して検証し、変更前はゼロbonusで受入失敗（gridmaze-smoke-red.log、終了1）、変更後はゼロ・負bonusと既存NaN例外の許容、欠測・非有限・学習なしの拒否の計8ケースが成功（gridmaze-smoke-green.log、終了0）。保存済みGridMaze r2は完走・学習後5,448点・記録したlearn_count最大2,724で新条件に合格し、Atari r1（5,448点）・Montezuma r3（6,233点）の判定と3群すべての13指標の集計は旧結果と一致した。新判定は `.scratch/prd116/gridmaze-smoke-reassessment.json` へ別名保存し、旧集計・実行記録・PRD・indexのSHA256不変を確認した。新しいRunは起こさず、RND本体・設定・PRD・英語版・§4.3と実行体の複製を変更していない。MSVC初期化付きDebug全体ビルドも成功（gridmaze-smoke-build.log、ninja: no work to do）、Python構文・UTF-8・LF・BOM保持・差分空白を確認した。設定を変えていないため既定葉検査・[shipped_config]は再実行しない。全品質項目を適用範囲に沿って自己点検し、実装上の未決はない。上の未達と原因未切分けの記述はPRD改訂前の履歴であり、今回の改訂後のGridMaze smokeには残らない。 独立レビュー（20261107-2215）は計画・実装ともOKで、原文は `.scratch/prd116/review-20261107-2215-gridmaze.md`。範囲外の検出1件として、共通の有限性検査はbonus_mean=0でbonus_shareのNaNだけでなく±Inf・欠測も許容する既存問題があった。今回の変更前から存在し、共通検査を維持する依頼の範囲外なので未修正とし、工程末の「この場で直す」候補として開発者の選択を待つ。


## この場で直す: bonus_shareのNaN例外

開発者が前回レビューの範囲外の検出を「この場で修正して再レビュー」と選んだため、共通検査の例外を、03_bonus_shareが記録済み・値がNaN・同じ点の01_bonus_mean=0の場合だけに限定した。±Infと欠測を拒否し、READMEにも明記した。上の選択待ちは解消済みである。検証は `.venv/Scripts/python.exe -X utf8 .scratch/prd116/gridmaze_smoke_check.py`。追加の3ケースは変更前に拒否できずRED（gridmaze-smoke-finite-red.log、拒否5/8、終了1）、変更後は従来の8ケースを含む11ケースが成功（gridmaze-smoke-finite-green.log、終了0）。保存Runを再検査するとGridMaze r2とMontezuma r3は合格し、それぞれ13指標の集計も旧結果と一致した。一方、KFM r1は学習後5,448点中4,008点でbonus_shareが欠測し、最初の検出はexp step 13,168だった。すべてbonus_mean=0の点で、旧検査が見逃していた欠測を厳密化後に拒否するため、KFMの保存記録の判定は合格から不合格へ変わる。これは検証コードの回帰失敗ではなく、修正が検出すべき既存記録の不足であり、212 §5.2も現在の不合格を示すよう更新した。新判定は `.scratch/prd116/gridmaze-smoke-reassessment-finite.json`、全欠測点はsmoke-finite-inventory.jsonへ別名保存し、旧集計と最初の再判定JSONは保持した。保存済み集計・実行記録・PRD・indexのSHA256不変、Python構文・UTF-8/LF/BOM・相対リンク・差分空白、MSVC初期化付きDebug全体ビルド（no work to do、終了0）を確認した（gridmaze-smoke-finite-static.log、gridmaze-smoke-finite-build.log）。新しいRun、過去の欠測の補完、出力側の変更は行っていない。設定・本体・PRD・英語版・212 §4.3も保持し、既定葉検査と[shipped_config]は再実行していない。追加修正の検証は成功し、保存KFMのsmoke不合格を全受入条件成功とは扱わない。 再レビュー（20261107-2227）は計画・実装ともOKで、不足点・修正要求・新たな範囲外の検出は0件。原文は `.scratch/prd116/review-20261107-2227-gridmaze.md`。今回の追加修正は完了し、保存KFMの欠測による受入不合格は既知の結果として残す。新規の起票候補・判断待ちはなく、ステージ・commit・pushは実施していない。


## 残件消化: GridMazeの性能測定

2026-11-07、開発者がRunの切れ目を確保し、GridMazeだけの§6.3性能測定を実施するよう指示した。基準HEADは3d1960c7bc4ec027451b306ca31ee520550a2bb0。Release Runnerをビルドし、実行体・DLL・現在の設定を `.scratch/prd116/grid-performance-20261107-r1/` へ複製する。現在のGridMazeの256 lane・CPU・torch thread 1、同じQ設定とseed1、評価・描画休止でRNDなし/既定RNDを比較する。既定RNDはwhiten・clipped・next_state・ratio .25・batch32を保持し、warm-up12,800件を越えた16,384経験から測る。testdata/prd116/run_performance.pyを追加し、設定解決でRNDと診断以外の差が無いことを確認し、8,192経験の試走2本から遅い群で約30秒となる同一経験区間を選ぶ。ABBA3巡の12本は1本ずつ順次起動し、同区間の時計差・経験/秒・群ごとの中央値とON/OFF比を保存する。Tracyは通常比較へ混ぜず、別のRND有効Runのwarm-up後に公式0.13.1のcaptureを8秒接続し、CSVのtarget/predictor forward・backward・optimizer・正規化・統計・転送の区間時間と件数を記録する。組込みクライアントのprotocol76とローカルTracy0.13.2のprotocol77は異なるため、一致版を `.scratch/prd116/tracy-0.13.1/` に取得した（third_partyは変更しない）。詳細stdout/stderrは各caseのlauncher.log/capture.log、全体の進捗はgrid-performance-*.logへ保存し、モデルへは進捗・終了値・要約だけを返す。進捗はプロセスの終了・Run時計の更新・CPU時間で判断し、経過時間だけで停止しない。測定中に他のビルド・テスト・Runは重ねない。新しい設定やアルゴリズムは追加せず、Atariの測定は未実施のまま保持する。

### 実行結果と保存

試走2本は8,192経験でOFF 1.057秒 /ON 1.485秒となり、本測定区間を165,632経験へ固定した。ABBA3巡の12本はすべて終了0で、16,384〜182,016経験の時計が揃った。中央値はOFF 23.260999秒 /ON 31.147997秒、ON/OFFは1.339065（実時間33.91%増、経験/秒25.32%減）。範囲はOFF 21.183999〜24.516002秒 /ON 30.069998〜31.587000秒、各巡の比は1.4084 /1.3177 /1.3057だった。ONの全6本で補助学習の進行を確認し、記録したlearn_countの最大値は1,322。これは最終checkpointの回数ではない。成績による学習系列の変化も含む全体比較であり、時間差をRND内部だけの純粋な費用とは扱わない。

別RunのTracyも終了0。完結したintrinsic_reward 171件（43,776観測）に含まれるEstimate 171回 /Learn 342回を集計した。CPU側のRND区間合計は2,339.370ms、1 intakeあたり13.681ms。forward合計56.47%、backwardとoptimizer合計35.10%、統計4.72%、最初の正規化1.23%だった。転送名の区間はCPUのTensor処理・検査等も含み、GPU転送の数字ではない。統計には2回目の正規化以外も含むため、二重正規化だけの費用をこの結果から断定しない。CSVのprocess起点total_percは使わず、同じthreadの完結した親区間に含まれる子区間だけを合算し、除外イベント0を確認した。

正本は [実験記録](../experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md) と同名JSON。15本（本測定12・試走2・Tracy1）の実効設定とTracy CSV3点は元のバイト列を保存し、manifestのSHA256を付けた。取得時のCRLFも保持するRun出力の写しであり、新規コード・説明文書のLF規約とは分けて扱う。実行体・DLL・source-diff.patch・元Run・traceは `.scratch/prd116/grid-performance-20261107-r1/` に保持した。公開用の設定・JSONでユーザー固有パスや秘密値の候補を検査した。設計212の§5.1・§5.2、検証README、索引issueの対応内容とwikiへGridMazeの結果とAtari未実施の範囲を反映した。PRD本文・英語版・本体・出荷設定は変更していない。

### 検証

Release全体ビルドはテスト実行体の再ビルドも始めたため中断し、MSVC初期化付きReleaseのAnetRLRunnerターゲットをビルドして終了0（gridmaze-performance-runner-build.log）。全体ビルド成功とは報告しない。測定終了後のMSVC初期化付きDebugのAnetRLRunnerターゲットは終了0（gridmaze-performance-debug-build.log、no work to do）。測定中にこれらを重ねていない。prepareの[prd103-resolve]2件は成功し、全キー差21件はRND・群56・Run名だけ。実際のOFF/ON Runもinspect_run.pyで照合して差8件はRND・Run名だけ、警告0（grid-performance-effective-diff.log）。本体と出荷設定は変更していないので、既定葉検査と[shipped_config]は再実行しない。

測定器の公開経路はCLIのprepare /pilot /measure /profile /summarizeで、観測対象はRunの終了値・同じ経験区間の時計・補助学習の進行・CSVである。新規PythonコードもTDDの適用対象であり、本体を変更しないことだけでは対象外にならない。初回のCLI実装では先行するREDを実施せず、実装後にprepare成功→pilot2本→measure12本→profile1本→summarizeの正常経路を確認した。既存記録の拒否も、保護分岐を実装した後でmeasureの再実行を試し、非0終了・明示的な拒否メッセージとmeasurements.json /summary.jsonのSHA256不変を確認した（grid-performance-reuse-check.log）。この2つのbehaviorは実装先行であり、初回の先行REDは未実施の手順漏れとして残す。過去の実行をTDD済みとは扱わない。§6.3のGridMaze分は中央値と内訳を取得済みであり、Atari B=64の3群比較は残る。速度の合否閾値は追加しない。

全品質項目を適用範囲に従って自己点検した。`.venv/Scripts/python.exe -X utf8 .scratch/prd116/check_grid_performance_archive.py` は終了0で、Python構文、UTF-8/LF、既存BOM保持、18点の保存SHA256と元データの完全一致、ABBA12本の終了値・件数・実行体一致、保存JSONと実測・中央値・比の一致、表の丸め、リンク、wikiの出典blob、差分空白を確認した。個人情報候補の走査で見つかった絶対パスは当該公開リポジトリの測定保存先だけで、ユーザー名・認証情報・メール・URL・IPの混入はなかった。新規コードのコメントを日本語、ログを英語で揃え、計画の主な変更と実差分を照合した。C++・本体API・契約の移行・skillの変更は無い。未決の実装判断は無く、独立レビューへ進む。

### レビュー指摘への対応

初回の独立レビューは計画・実装とも差し戻し。原文は `.scratch/prd116/review-20261107-2335-grid-performance.md`。測定値の不一致や範囲外の検出は無く、測定CLI自身のTDD適用と実際の順序の記載、失敗診断の実値・期待条件が不足していた。上記へ実際の順序と初回先行REDの未実施を明記し、診断には対象case・パス・終了値・時計端点・実際/期待SHA・検出件数等を加えた。本体と測定の正常系の計算・順序は変更していない。

診断修正の前に合成artifactの検査を用意し、正常集計は成功したが、既存記録の拒否診断に保存先が含まれないため終了1を確認した（grid-performance-diagnostics-red.log）。この最初のREDの後、同種の診断をまとめて修正し、正常集計・記録不変・件数/予算/実行体の不整合・実行体変更・Runner終了失敗・端点不足・時計不正・学習未進行・capture未開始・完結区間無しの12 behaviorが終了0になった（grid-performance-diagnostics-green.log）。残りの診断ケースの変更前REDは採っておらず、12件すべてを先行REDで確認したとは報告しない。検査は `core/anet-core/testdata/prd116/run_performance_test.py` として保存し、READMEから案内する。CLIは子プロセスで呼び、executeの失敗は合成artifactとテスト内のプロセス差し替えを使う。Runner・CIM問い合わせ・新規性能Runは実行しない。未実施だった初回の順序と、後から実施した回帰検証を区別して再レビューする。

再レビューは計画・実装ともOKで、不足点・修正要求・範囲外の検出・開発者への確認事項は0件。原文は `.scratch/prd116/review-20261107-2341-grid-performance.md`。正規配置のrun_performance_test.pyも12 behavior成功・終了0（grid-performance-diagnostics-canonical-green.log）、最終の保存整合検査も終了0（grid-performance-final-static.log）。wikiの読み取りscanは終了0、リンク切れ0・目録漏れ0で今回の出典の未取り込みは無かった（grid-performance-wiki-scan.json）。無関係なpending資料は更新しない。GridMazeの§6.3測定と記録は完了し、Atariの3群比較は既存の残件として保持する。新しい起票候補・判断待ちは無い。PRD・本体・出荷設定・元Runを保持し、ステージ・commit・pushは行っていない。

