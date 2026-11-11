# PRD116 RND 検証

初回実装の記録は [20impl](../../../../docs/memo/116_rnd_lifelong_novelty_20impl.md)、Estimate/Learnの独立化とdevice・保存の契約を見直した記録は [21impl](../../../../docs/memo/116_rnd_lifelong_novelty_21impl.md)、補助学習のcredit・件数・bool結果を見直した記録は [22impl](../../../../docs/memo/116_rnd_lifelong_novelty_22impl.md)。公開実装のNetwork形状、別々のtarget/predictor、統計・観測・倍率の選択肢は [23impl](../../../../docs/memo/116_rnd_lifelong_novelty_23impl.md)。

## 無効時の同一性

`run_validation.py` は PRD098 の短い4環境の比較を再利用し、保存先だけを `.scratch/prd116/baseline-20261106/` に分離する。過去の PRD098 の基準は変更しない。変更前に Debug 実行体・DLL・設定を複製し、`[prd093-config-snapshot]` で解決済み設定を採取済み。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd116/run_validation.py before
```

変更後は `after --label` で新しい名前を指定し、同じ名前の `compare --label` で全 scalar 点と `metrics.scalar.defs` を照合する。時刻・資源の群以外は除外しない。実行済みのラベルを別バイナリで上書きしない。

## 正しさ

MSVCを初期化してDebugビルド後、`anet-core-test.exe "[prd116],[scaler],[shipped_config]"` をリポジトリルートから実行する。公開Factory・倍率・補助学習・保存復元・設定・facadeを検証する。本番コードのbackend設定は変更しない。

`checkpoint --label` は GridMaze の16 experienceで変更前後の無効 checkpoint を保存する。libtorch optimizer が C++ パラメータアドレスを含むため、同じ旧実行体同士でもファイル全体のバイト列は一致しない。raw SHA256 を残し、header/config と Network はバイト比較、optimizer は宣言されたアドレスと生成 serialization_id だけを除いて展開内容を比較する。元の checkpoint は書き換えない。`compare_checkpoint.py` が比較を担当する。

## 性能測定

GridMaze分は2026-11-07に実施し、[実験記録](../../../../docs/experiments/default-dqn/gridmaze/2026-11-07_rnd_performance.md)へ3巡の中央値・Tracy内訳・全Runの実効設定を保存した。CPU・256 lane・Releaseで、RNDありの時間はなしの1.339倍だった。Atari B=64のRNDなし・既定RND・Taigaの3群は未実施のままである。合否の閾値は置かない。

`run_performance.py` は今回のGridMaze専用の採取器で、prepare（Release実行体・DLL・設定の複製と解決比較）、pilot（時間から同じ経験区間を選択）、measure（ABBA3巡）、profile（別Runでwarm-up後にTracy接続）、summarize（中央値・内訳の集計）を順に実行する。全段階で同じ `--out` を指定する。既存の出力へRunを追加し直すことは拒否する。profileには同じ通信版の `tracy-capture.exe`・`tracy-csvexport.exe` のフォルダを `--tracy-tools` で渡す。Tracyは通常比較へ接続しない。

今回の実行は `--out C:\dev\anet-lab\.scratch\prd116\grid-performance-20261107-r1`、profileのツールは `--tracy-tools C:\dev\anet-lab\.scratch\prd116\tracy-0.13.1\tools`。16,384経験から測り、試走後の区間は165,632経験だった。設定・本体を変えていないため、既定葉検査と[shipped_config]は再実行していない。別のRunやビルド・テストを測定へ重ねない。

測定器自体の検査は次のコマンド。合成artifactでCLIの集計・既存記録の拒否と各失敗診断を確かめる。Runner起動とCIM問い合わせはテスト内で差し替え、性能Runを起こさない。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd116/run_performance_test.py
```

## smoke

次に起こすsmokeのRun名は `run_<t>_prd116_<label>_tmp_smoke_<env>` となる。既存のprd098名のRun・基準・記録は変更しない。

GridMazeは完走・群56の有限性・補助学習の進行を確認し、bonusの符号と減少は判定しない。観測が正規化した(x, y)の2値で盤面固定のため、限られた入力に慣れて新しさが尽きると、既定のclipped・whitenではbonusが0になる。Atari KFMは100k experienceで失敗・NaN・clip_ratioの張り付きを確認する。手順と実行結果は実装記録へ追記する。

`run_validation.py smoke --env GridMaze Atari Montezuma --label reference-smoke-r1` は各100k experienceを順に実行する。GridMazeとAtari KFMは既存の連鎖へ `run.@rnd`、MontezumaはAtariの連鎖へ `run.@rainbow_like > run.@rnd > run.@rnd_taiga` を加える。PRD098の短縮条件（8 lane、Q replay batch 16、capacity 4096、Q warm-up 256、RR1、評価休止など）は最後の検証用profileで選ぶ。RNDの8設定値は変更しない。

実行体を保存し、13指標の件数・範囲・前後四分の一の平均と受入判定を記録する。`config/config_data.txt` の実効 `warmup_steps` より exp step が大きい点を対象にする。learn_predictor の warm-up 中は学習が進むが誤差統計が NaN なので、learn_count で区切らない。KFM は `warmup_steps=12,800` を明示し、100k 件の途中で warm-up を終える。`03_bonus_share` が記録され、その値がNaNで同じ点の `01_bonus_mean` が0の場合だけ例外として許す。欠測や±Infは許さない。GridMazeは対象点が1つ以上あり、群56の全指標が有限（前述のNaN例外を除く）であること、KFMはclip_ratioの全点が1未満、Montezumaはbonus_minが負の点が9割以上であることを確認する。機構のsmokeであり、成績・性能の比較ではない。再実行時に別の実行体なら新しいlabelを使い、記録済みRunを上書きしない。
