# Atari の観測の縮小方式を設定で切り替える PRD

> 2026-10-23 起票（暫定。質疑を省いて起草）。
> 2026-10-23 再グリル・改訂: 方式名、面積平均と偶数丸め、対応解像度1〜160、評価個別上書き、画像ごとの一致率、速度低下2%以内を確定。OpenCVの丸めの説明と84以外の画素数の説明を訂正し、事前調査の報告値と今回確認した事実を分けた。
> 関連: [ADR 0026](../adr/0026-atari-env-single-seam-own-preprocessing.md)、[ADR 0025](../adr/0025-ale-external-build-reference-and-license.md)、[PRD088](088_nn_adaptive_max_pool_10prd.md)、[BTRコード読解](../../reports/btr_code_reading_2026-08-31.md)、[Qbert比較記録](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)。

## 1. 目的と問題

Atariの観測を210×160から正方形へ縮小する方式を選べるようにし、Qbertで縮小方式だけを変えた比較を可能にする。現行のadaptive average poolingと、BTRが使うOpenCV INTER_AREAは、非整数比の縮小で同じ計算ではない（§2.1）。前処理の差を学習成績の差の候補として調べるため、現行方式を既定に残して面積で重み付けする方式を追加する。

本PRDの成功は、同じ入力の数値一致を検証でき、ほかの条件を揃えた比較ができることにある。Qbertの改善は未検証の仮説として扱い、学習効果と機能の受入を分ける。

## 2. 根拠と確認状況

### 2.1 今回コードで確認した事実

- 現行の[ResizeGrayscale](../../core/envs/atari1/src/AtariPreprocess.cpp)は、入力をfloat32にして`interpolate(mode=torch::kArea)`を実行し、`round`・0〜255への制限・uint8への変換を行う。[AtariEnv](../../core/envs/atari1/src/AtariEnv.cpp)には、Resetから使うCaptureObservationとStepの2経路に呼び出しがある。
- PyTorchのareaモードはadaptive average poolingへ進む。1軸の出力画素iは、元の画素の添字`floor(i·in/out)`以上、`ceil((i+1)·in/out)`未満を等しい重みで平均する。たとえば210→84の最初の窓は3画素を等しく平均する。出力の区間が覆う元画像の長さ2.5に応じて端の画素を部分的に重み付けする計算とは異なる（[PyTorch functional.py、area分岐](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/functional.py#L4331-L4339)、[AdaptivePooling.h](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/AdaptivePooling.h)）。これらはAPIの意味を確認する出典であり、anetの利用版をv2.7.0へ変更する要求ではない。
- BTRの公開実装は`cv2.resize(..., interpolation=cv2.INTER_AREA)`を呼ぶ（[BTR、commit 093ca77、AtariPreprocessingCustom.py](https://github.com/VIPTankz/BTR/blob/093ca77f08540d646f12fd942ee5fda763e0fd8a/AtariPreprocessingCustom.py#L224-L228)）。OpenCVの非整数比の縮小は、元画素と出力画素の重なる長さを各軸で重みにする（[OpenCV 4.8.1、computeResizeAreaTabとarea分岐](https://github.com/opencv/opencv/blob/4.8.1/modules/imgproc/src/resize.cpp#L3015-L3060)）。
- OpenCVの丸めを一律に偶数丸めとは呼べない。2×2縮小には、非負の合計に2を足して4で整数除算する経路があり、平均0.5は1になる。非整数比のarea経路は浮動小数の加算とsaturate_castを使う。本PRDはOpenCVの全分岐・実行環境の厳密再現を要求せず、anetの規則と84×84の許容差を別に定める（[OpenCV 4.8.1、2×2経路](https://github.com/opencv/opencv/blob/4.8.1/modules/imgproc/src/resize.cpp#L2730-L2735)、[非整数比の加算・変換](https://github.com/opencv/opencv/blob/4.8.1/modules/imgproc/src/resize.cpp#L2948-L2978)）。
- [Config::ReadConfig](../../core/anet-core/include/anet/config.hpp)は共通のAtariEnv設定を読み、評価用prefixの明示値があれば上書きする。[RunManager](../../core/anet-core/src/trainer.cpp)は定期評価とEvalPanel用のEnv生成へ`run.eval.[tag].env`を渡す。Runの葉とCLIの優先は[設定設計§2.1](../design/100_runtime_and_configuration.jp.md#21-設定の解決)および[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)・[ADR 0042](../adr/0042-config-inheritance-as-differential-base.md)に従う。
- [Atari設計§4.3](../design/220_atari_env.jp.md#43-観測と前処理チェーン)の「cv2 INTER_AREAと同系」は数値一致を意味しない。実装時に方式の違いが分かる記述へ更新する。

### 2.2 引き継いだ事前調査の報告値

2026-10-23の暫定起票時に、QbertをALE 0.9でランダムに2,992 step動かし、最後の2フレームの最大を取った同一画像を両方式で比べると、24.9%の画素が異なり、16.1%は差8以上、0.60%は差32以上、最大差59だったと報告された。同時に、BTRの公開模型をALE 0.9・sticky 0.25で評価した20ゲームすべてが第8面を越えたと報告された。

これらは暫定起票からの引継ぎ値で、今回再測定していない。元の入力一式・集計出力・完全な実行条件への出典は未確認であり、今回の受入証拠にも、縮小方式が成績差の原因である根拠にも使わない。受入2では新たに固定する入力・期待値・生成条件を使用する。

## 3. ゴール・非ゴール・効果の確かめ方

### ゴール

- **G1**: 縮小方式を設定で選択でき、学習・定期評価・EvalPanelに通常の設定継承で適用できる。
- **G2**: 面積重み付き方式が§4.2の計算規則と、84×84のOpenCV正解データに対する§6の許容差を満たす。
- **G3**: 現行方式を既定に保ち、既定の学習系列を変更しない。
- **G4**: 84×84・64 envのEnv Step区間の速度低下を2%以内に収める。

方式選択とRunプロファイルはG1の比較条件を作り、固定画像・数値照合はG2の方式の取り違えを検出する。共通既定と等価性確認はG3を、Step単位の見積もりと交互測定はG4を支える。

### 非ゴール

OpenCV依存の追加、ALE内蔵ベクトル環境への置換、RGB観測や縮小以外の前処理の変更、既定方式の切り替えは行わない。理由と再開条件は§7に示す。拡大を含む出力はpixel_areaの対応範囲外とする。

### 学習効果の確かめ方

仮説は「BTRとの縮小の差を減らすとQbertの第8面以降への進行が改善する」である。64 env・replay 1M・行動ネットの毎step同期・減衰するε・BTRに寄せたNoisyNet初期値を共通にし、方式だけ異なるseed 1の50M経験のRunをまず1本ずつ比較する。解決後の設定で、方式以外の差がないことを確認し、実行機や外部依存版の差があれば併記する。

[既存のQbert比較記録](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)に合わせ、第8面突破はゲームごとの生スコア29,000以上かつ`round_clear ≥ 8`で数える。主に45〜50Mの完了ゲーム中の突破割合を比べ、初突破の経験数と、20〜25Mの完了ゲーム中で第7面以降に到達した割合を併記する。後者はround_clearが6以上のゲームで数える（0始まりの面番号からクリア回数を数える定義は[PRD082](082_atari_ram_progress_metrics_10prd.md)のQbert RAM表・設定例）。得点だけでは進行を代用しない。

両Runが50Mへ達した時点で、改善・差を確認できず・悪化のいずれかと比較条件を実験記録に残し、初回比較を終える。初突破の早さだけや1 seedの差だけで因果を確定しない。機能受入は成績の改善を条件にせず、既定方式の変更は§7の条件が揃った後の別判断とする。

## 4. 確定事項と実装契約

### 4.1 設定と失敗

| 項目 | 契約 | 理由・前提 |
| --- | --- | --- |
| 方式 | 文字列`AtariEnv.resize_mode`。`adaptive_average`と`pixel_area`の2値。既定は前者 | G1/G3。計算の性質を名前にし、ライブラリの全分岐との一致を名前に含めない |
| 現行方式 | 既存の計算と有効なscreen_sizeの扱いを保持する | G3。新方式の対応範囲を理由に現行方式を狭めない |
| 新方式の解像度 | 整数の`screen_size`が1〜160（両端を含む） | 入力は210×160、出力は正方形なので、この範囲はどちらの軸も拡大しない。84専用の制限を増やさない |
| 不正指定 | 未知値・空の方式名、不正なscreen_sizeの型・値・組み合わせは構築完了前にANET_SYSTEM_ERROR | [Fail-Fast原則](../../AGENTS.md#fail-fast-原則)。既定への置換、別方式への代替、設定値の丸めはしない |
| 診断 | 対象の設定キー、指定値、期待する方式または範囲を含む。評価個別指定ならそのprefixも識別できる | 利用者が誤った指定箇所を修正できることが必要 |

### 4.2 面積重み付き縮小の規則

元画素は幅・高さが1の領域として扱う。各出力画素が元画像で覆う長方形と各元画素の重なり面積を、その長方形の面積で割って重みにする。その重みで画素値を平均し、全寄与を合計した最後に最も近い整数へ丸め、ちょうど半整数なら偶数を選ぶ。結果を0〜255に収め、単フレームuint8の`[1,S,S]`観測を返す。途中で画素値を整数へ丸めない。

各出力画素に対する重みは非負で合計1、整数比では通常の箱平均になる。出力84×84では元の各軸の寄与は最大4画素以内だが、この上限を小さいscreen_sizeにも適用しない。より広い元領域を覆う場合も同じ面積の規則で計算する。

丸めはanet側の明示的な規則であり、「OpenCVは常に偶数丸めだから」という理由にはしない（§2.1）。84×84の参照出力との許容差は§6の受入2に従う。

### 4.3 所有・適用・設定例

縮小は従来どおりAtariEnvのCPU側で行い、生成した観測を指定deviceへ渡す。方式と解像度に依存する重みは各Envの構築時に一度準備し、以後再利用する。ResetとStepは同じ設定を使う。理由は、[ADR 0026](../adr/0026-atari-env-single-seam-own-preprocessing.md)の前処理所有権を保ち、stepごとの重み生成を避けるためである。

[Atari.txt](../../apps/runner/config/Atari.txt)に以下を追加する。これは将来の実装契約であり、本PRDの改訂で設定ファイルを編集するものではない。

```text
AtariEnv.resize_mode ?= adaptive_average
run.@pixel_area : AtariEnv.resize_mode = pixel_area
```

方式選択には既存のRun選択チェーンへ`run.@pixel_area`を加える。Runの共通値はworkspaceの同じ共通キーより優先する一方、評価個別の明示値まで強制しない。CLIの優先も既存の解決規則を保つ（§2.1）。たとえば共通がpixel_areaでも、次の指定を置いた評価だけは現行方式になる。

```text
run.eval.[eval_target].env.resize_mode = adaptive_average
```

評価個別指定がなければ定期評価とEvalPanelは共通値を継承する。個別指定がある場合はそのEnvに適用する。方式をRun全体へ強制する専用分岐は追加しない。

読み取った実効値を既存のconfig dumpへ含め、各Env構築時にEnv名と`resize_mode`を英語ログで1回記録する。毎stepは記録しない。理由は、同じRun内の評価個別上書きを追跡できるようにするためである。

### 4.4 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
| --- | --- | --- | --- |
| AtariEnvConfig | 変更 | 縮小方式を読み取り、実効設定として記録する | AtariEnv |
| AtariEnv | 変更 | 方式・解像度を検証し、ResetとStepへ同じ縮小を適用する | AtariEnv |
| GrayscaleResizer | 新規 | 縮小設定と重みを保持し、各方式の観測生成を担う | AtariPreprocess |

削除するクラスはない。内部の型、格納形式、引数の型、関数分割、計算の分け方は、上記の数値・所有・性能契約を満たす実装計画で決める。

### 4.5 名前・ADRと現用範囲の更新

[CONTEXT.md](../../CONTEXT.md)のAtari・観測用語と衝突しない。adaptive average poolingと面積重み付き縮小は一般的な画像処理の用語で、プロジェクト固有概念の追加ではないため用語集へ増やさない。Resizerは画像縮小の責務を表す接尾辞である。

ADR 0026のSingle接続、自前前処理、stack・並列化の責務は変更しない。方式の選択追加はその境界の内側に収まり、既定も維持するため、新しいADRも既存ADRへの注記も不要とする。

実装時は次を同じ変更内で更新する。

- AtariPreprocessの縮小処理と現用呼び出し、AtariEnvの設定・検証・Reset/Step。
- Atari.txtの既定とRunプロファイル、AtariEnv_testの現用呼び出し・golden・設定検証。
- PRD099の固定データ、生成スクリプト、比較・性能検証資材とREADME。既存の`core/anet-core/testdata/prdNNN/README.md`の規約に従い、PRD099資材は`core/anet-core/testdata/prd099/`へまとめ、[Atari設計書](../design/220_atari_env.jp.md)のテストの章から参照する。
- [Atari設計書（日本語）](../design/220_atari_env.jp.md)の設定・前処理・テスト、および[英訳](../design/220_atari_env.en.md)。利用例を追加する場合も確定した方式名に揃える。

現行adaptive_averageは新契約の一方式として保持する。廃止した関数呼び出しを移す場合は現用利用側をまとめて移し、旧引数のoverloadやaliasを残さない。過去のRun、既存golden、実験記録・ADRは保持し、読み替える互換コードは足さない。既存の未コミット変更のうち本PRDと無関係のものは編集しない。

## 5. 性能の見積もりと内部方式を決める条件

通常の1 laneのStepは1回縮小する。終端してResetも起こる場合は、その観測生成分が加わる。210×160→84×84で重なる元画素の組を直接数えると60,480寄与／lane、64 laneのBatch Stepなら3,870,720寄与である。これは画素数と区間の重なりからの計算量の見積もりで、採用する計算の分け方を指定しない。

1寄与の処理時間をc、呼び出し固定費をaとすると、新方式の通常Batch Stepに必要なCPU総仕事量の概算は`64a + 3,870,720c`、終端lane分は別途加わる。lane並列のため、これをそのまま経過時間とはしない。経過時間への寄与は、実際のworker配置で測定する。時間単価と現行方式との差は未測定である。単価、固定費、終端頻度を実装計画で実測し、現行Batch Step時間Tに対する追加時間を比較する。追加時間が`0.01T`（受入の低下許容2%の半分）を超える見込みなら、内部方式の本採用前に試作測定を行う。単価未確認のまま「1%未満」と扱わない。

構築時の重み準備をstepへ移す、thread poolの呼び出し回数・並びを変えるなどの案でも、同じ頻度×単価の確認を行う。これにより、実装内部の選択を未測定の楽観値で確定しない。

## 6. 検証と受入

以下は実装後の受入である。今回のPRD改訂ではコードのテスト・ビルド・Runを実施しない。未実施を合格として扱わず、実装計画に結果・コマンド・ログの場所を残す。

1. **既定の等価性**: 同seed・決定的backend・同じ依存と既存設定値で、変更前、変更後のキー省略、変更後のadaptive_average明示を比較する。短いAtari Runの学習に関するscalar全点・traceの系列が一致し、checksumも一致すること。時刻・実時間・資源使用量を除く対象は比較前に固定し、学習値の差を除外しない。ResetとStepの既存観測テストも保持する。
2. **OpenCVの正解との一致**: Qbert・Breakout・Battle Zoneから各8枚、計24枚の210×160灰色画像と84×84の期待画像を固定する。pixel_areaは**各画像について**全画素の絶対差1以下、かつ99.9%以上の画素が完全一致すること。1枚でも外れれば失敗とし、許容を自動的に広げない。比較は最終uint8画像で行う。生成はOpenCV 4.8.1のINTER_AREAで行い、入力・期待値・生成スクリプト・seed・採取条件・ALE/OpenCV版・生成環境と画像checksumを保存する。画像データは計975,744 byte（約0.98 MB）で、1 MB未満を維持する。現行方式に参照画像と異なる出力があることも確認し、両方式の取り違えを検出する。
3. **規則と境界**: 重みが非負で各出力画素への合計が1になること、定数画像の保持、画素境界にかかる入力、整数比の箱平均を独立に検証する。半整数の例は0.5→0・1.5→2・2.5→2を含み、途中丸めを検出できる入力も使う。Atari入力に対しS=1・84・160を確認する。整数比の箱平均は合成入力でも確認し、OpenCVの2×2専用丸めをanetの期待値に持ち込まない。
4. **設定と継承**: 2方式の読み取り・既定・config dump、Runプロファイルの共通指定、評価個別上書き、定期評価とEvalPanelの生成を検証する。未知値・空の方式名、screen_sizeの不正型・0・負数、pixel_areaで161以上を指定した場合は構築時に失敗し、キー・値・期待条件がメッセージにあること。adaptive_averageに新方式専用の上限を適用しないこと。
5. **速度**: Qbert・84×84・64 envで、同じseed・行動入力条件・worker設定・device・依存版を使う。測るのはBatch EnvのStep区間に処理されたlane遷移数／秒で、学習・描画の時間を混ぜない。Step内の終端・自動Resetの費用は含める。まず現行方式どうしの交互対照を5組測り、各組の速度比が0.99〜1.01に収まることを確認する。その後、各組をadaptive_average→pixel_areaの順で5組実行する。各試行は30秒準備後に60秒計測する。組ごとのpixel_area／adaptive_average速度比の中央値が0.98以上なら合格、下回れば不合格。対照が1%以内に収まらない、他の重い処理が並行する、試行が欠ける場合は判定保留とし、環境が整ってから対照を含め測り直す。各試行の値と中央値を保存する。
6. **実Runと表示**: run.@pixel_areaを含むRunで、学習・定期評価・EvalPanelの実効方式をそれぞれのdumpと構築ログから確認する。評価個別上書きの有無も照合する。Evaluation Traceの観測プレビューが、その表示に使う同じ灰色入力の参照出力と整合することを確認する。見た目だけで受入2の数値確認を代用しない。

受入5・6は機械が測定可能になってから行ってよいが、実装後も未完了として明示する。通常の実行体を上書きせず、必要な差分ビルドの実行体を`.scratch/prd099/bin`へ隔離する手順と設定の後始末は実装計画で具体化する。全再ビルドは行わない。

## 7. 作らないもの・先送り

| 項目 | 裁定 | 理由と再開条件 |
| --- | --- | --- |
| OpenCVをリンクする実装 | 不採用 | ADR 0026の自前前処理を保つ。クリーンブレークで置換は可能だが、今回の面積計算に依存追加を必要としない |
| ALE内蔵ベクトル環境への置換 | 範囲外 | ADR 0026の並列化・stackの責務まで変える別の関心。Envが律速となり、既存RBとの接続も含めて再設計するときに別PRDで扱う |
| 拡大を含むpixel_area | 範囲外 | 縮小比較とは別の関心。拡大が必要になったときに数値契約を別PRDで決める。クリーンブレークで将来拡張することは妨げない |
| RGB観測・灰色化・max-pool等の変更 | 範囲外 | 縮小だけの比較を成立させる実験上の制約。変更自体は可能だが、別軸の比較として別PRDで扱う |
| 84専用の解像度制約 | 不採用 | G1の解像度設定との整合から、両軸とも拡大しない1〜160を認める。84以外の性能同等までは本PRDの受入に加えない |
| OpenCVの全分岐・全環境との完全一致 | 不採用 | anetの面積平均・偶数丸めを契約とし、比較で使う84の固定画像に許容差を定める。参照環境の特殊な丸めを実装契約へ増やさない |
| 既定をpixel_areaへ切り替える | 先送り | 実験結果が必要。Qbert比較とAtari-5の残り、とくにBattle Zoneへの影響が分かってから別判断とし、採用時には方式を比較記録で区別する |
| ALE 0.12と0.9のエミュレータ一致の調査 | 先送り | 別の比較軸であり、まず縮小のA/Bを行う。条件を寄せても差が説明できないときに別の調査として開く。現時点で一致を確認済みとは扱わない |

## 8. 未決と後続作業

仕様上の未決: なし。名前・丸め・対応範囲・失敗・性能閾値・細部の推奨8件は2026-10-23の質疑で確定した。内部の型や関数分割を決めないことは、仕様の未決ではない。

本工程で行うのはPRD起草・自己点検・独立レビューと関連wikiの更新である。コード・設定の実装、golden生成、受入1〜6、Qbert比較の実測は後続工程で行う。新たな999 PRDの起票候補はなし。§7は理由と再開条件を本PRDに保持し、この工程で別のPRDを自動起票しない。

## 9. 参照

- [BTRコード、commit 093ca77] VIPTankz. AtariPreprocessingCustom.py. [固定commit](https://github.com/VIPTankz/BTR/blob/093ca77f08540d646f12fd942ee5fda763e0fd8a/AtariPreprocessingCustom.py)。
- [OpenCV 4.8.1] OpenCV. resize.cpp. [固定tag](https://github.com/opencv/opencv/blob/4.8.1/modules/imgproc/src/resize.cpp)。
- [PyTorch v2.7.0] PyTorch. functional.py / AdaptivePooling.h. [areaモード](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/functional.py#L4331-L4339)、[窓の定義](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/AdaptivePooling.h)。
- [Qbert比較記録、2026/09] [BTRとの差を詰める比較](../experiments/default-dqn/atari/2026-09-27_btr-gap-qbert-bz.md)、判定指標・探索ブロック03/06。
