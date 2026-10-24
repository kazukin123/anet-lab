# PRD099 Atari縮小方式の検証

[PRD](../../../../docs/memo/099_atari_resize_mode_10prd.md)・[実装メモ](../../../../docs/memo/099_atari_resize_mode_20impl.md)・[Atari設計](../../../../docs/design/220_atari_env.jp.md)。リポジトリルートから実行する。

## 固定画像

- frames_210x160.u8: Qbert、Breakout、Battle Zone各8枚、計24枚の灰色画像。各Stepで最後の2フレームの最大を取った入力。
- expected_84x84.u8: 同じ入力へOpenCV 4.8.1のINTER_AREAを適用した期待値。
- 形式: uint8、row-major。画像データは計975,744 byte。meta.jsonに配列形、採取seed、Step間隔、ALE/OpenCV/Python/NumPyの版、生成環境、OpenCV build識別子、入力・期待値のSHA256を記録する。
- 許容: 各画像について全画素の絶対差1以下、99.9%以上が完全一致すること。画像全体での平均に緩和しない。旧方式と異なる画素があることも検査する。

make_golden_cv2.pyはOpenCV 4.8.1以外を拒否し、既存の固定ファイルを上書きしない。WSLの既存BTR venvで、毎回新しい作業用出力先に再生成する:

```bash
$goldenOutput = '/mnt/c/dev/anet-lab/.scratch/prd099/golden-' + (Get-Date -Format yyyyMMdd-HHmmss); wsl.exe -d Ubuntu-24.04 -e bash -lc ('~/BTR/.venv/bin/python -I /mnt/c/dev/anet-lab/core/anet-core/testdata/prd099/make_golden_cv2.py ' + $goldenOutput)
```

固定画像を変更する場合は、生成物をそのまま差し替えず、入力のSHA256・生成条件・受入結果の差を確認してから意図した更新として扱う。

## C++テスト

既存CMake presetを使い、AGENTS.mdのVsDevCmd経由でAtariEnv-testを差分ビルドする。高速化後の実行体は.scratch/prd099/bin-perf/Debugへ隔離してある。2026-10-23の.scratch/prd099/bin/Releaseは実験用として保持する。通常出力の実行体を上書きする必要はない。

```bash
.\.scratch\prd099\bin-perf\Debug\AtariEnv-test.exe "[prd099]"
```

```bash
.\.scratch\prd099\bin-perf\Debug\AtariEnv-test.exe "[atari]"
```

数値テストはROM不要。工場・Reset・Stepを通すPRD099のテストはATARI_ROM_DIRからPong/QbertのROMが得られることをREQUIREし、未解決を成功やskipにしない。既存の通常ROM依存テストのskip方針は変えない。

テストは、既定と実効設定、非整数比の面積平均、固定24画像、半整数の偶数丸め（210×160入力の1×1・84×84縮小を含む）、途中丸めなし、定数と入力範囲の保持、整数比、S=1/84/160、未知・空の方式、不正型・範囲、評価個別上書き、Runプロファイル、Reset/Stepへの適用を確認する。privateな重みの取り出し口は設けず、定数の保持と独立な合成入力の期待値から数値規則を検査する。

## 既定の学習系列の等価性

run_validation.pyは、隔離Release Runnerを使い、CPU・1 torch thread・4 env・16,384経験の短い決定的Runを逐次実行する。学習更新を含め、出力先は.scratch/prd099/acceptanceに固定する。実行体・設定一式・引数・終了コードをケースごとに保存し、既存ディレクトリの上書きを拒否する。

コード変更前に2本の基準を採取し、再現性を確認する:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd099/run_validation.py before
```

変更後の隔離Release Runnerで、キー省略とadaptive_average明示の両方を比較する:

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd099/run_validation.py after
```

比較にはPRD094のcompare_metricsを再利用し、scalar全点とtraceの行・定義・WARNを照合する。時刻は数値化の対象にせず、90_perf/・91_resource/・92_resource/だけを除く。学習値を除外して一致させない。解決後の全キー・文字列値も照合し、出力先・Run名以外の既存値を保持し、追加はAtariEnv.resize_mode=adaptive_averageだけを許す。実効Envの設定はRun内のconfig/env.*.txtでも確認する。

## 性能と実Runの受入

短い縮小単体の試作測定は内部方式の判断材料であり、PRDの性能受入を代替しない。正式な受入は、負荷のない環境でQbert・84×84・64 envのBatch Env Step区間を測定する。

現行方式同士の対照を5組実行し、各組の速度比が0.99〜1.01に収まることを確認する。その後adaptive_average→pixel_areaを5組、各試行30秒準備＋60秒計測で比較し、組ごとの速度比の中央値が0.98以上なら合格。終端・自動Resetの費用を含み、学習や描画の時間を混ぜない。競合する重い処理、対照のばらつき超過、試行の欠損があれば判定保留とする。

学習・定期評価・EvalPanelの実効設定と構築ログ、Evaluation Traceの観測プレビューも、測定可能な環境で確認する。未実施事項は実装メモへ残し、合格として扱わない。

## pixel_area高速化の出力不変性

既存のRelease契約テストも、高速化した隔離実行体で確認する。

```bash
.\.scratch\prd099\bin-perf\Release\AtariEnv-test.exe "[prd099]"
```

2026-10-24の最適化前の計算をAtariEnv_test.cppのPixelAreaReferenceResizerへ固定している。隠しテストは固定24枚・乱数1000枚・境界5種類とS=1..160の直積、計164,640組・1,418,126,640画素の完全一致を要求する。3000×3000の64bit積算を要する入力も含む。既存OpenCVの許容と固定画像は変更しない。

```bash
.\.scratch\prd099\bin-perf\Release\AtariEnv-test.exe "[prd099-perf-exact]"
```

縮小単体の参考測定は、同じプロセス内で最適化前参照・最適化後・既定を5巡ローテーションし、各方式50回準備後、1試行2000回の値と5値の中央値を出す。正式な64 envの性能受入は代替しない。

```bash
.\.scratch\prd099\bin-perf\Release\AtariEnv-test.exe "[prd099-resize-probe]"
```

同じpixel_area縮小器のResizeは並行実行しない。各Envが独自の縮小器を持つことで、そのEnvの直列のReset/Stepで作業領域を再利用する。出力は作業領域を借用せず独立したTensorである。
