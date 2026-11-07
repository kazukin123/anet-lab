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

