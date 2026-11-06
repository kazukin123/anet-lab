# PRD116 RND 検証

実装記録は [20impl](../../../../docs/memo/116_rnd_lifelong_novelty_20impl.md)。

## 無効時の同一性

`run_validation.py` は PRD098 の短い4環境の比較を再利用し、保存先だけを `.scratch/prd116/baseline-20261106/` に分離する。過去の PRD098 の基準は変更しない。変更前に Debug 実行体・DLL・設定を複製し、`[prd093-config-snapshot]` で解決済み設定を採取済み。

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd116/run_validation.py before
```

変更後は `after --label` で新しい名前を指定し、同じ名前の `compare --label` で全 scalar 点と `metrics.scalar.defs` を照合する。時刻・資源の群以外は除外しない。実行済みのラベルを別バイナリで上書きしない。

## 正しさ

MSVC を初期化して Debug ビルド後、`anet-core-test.exe "[prd116],[scaler]"` をリポジトリルートから実行する。公開 Factory・倍率・補助学習・保存復元・設定・facade を検証する。

`checkpoint --label` は GridMaze の16 experienceで変更前後の無効 checkpoint を保存する。libtorch optimizer が C++ パラメータアドレスを含むため、同じ旧実行体同士でもファイル全体のバイト列は一致しない。raw SHA256 を残し、header/config と Network はバイト比較、optimizer は宣言されたアドレスと生成 serialization_id だけを除いて展開内容を比較する。元の checkpoint は書き換えない。`compare_checkpoint.py` が比較を担当する。

## 性能測定

2026-11-06 のユーザー指示により未実施。別の Run が動いている間は実行しない。実行可能になったら、PRD §6.3 の Atari B=64 と GridMaze で有効・無効の ABBA を3巡以上行い、実時間中央値と target/predictor forward、backward、統計、転送の内訳を記録する。合否の閾値は置かない。

## smoke

GridMaze は warm-up 後の bonus と減少傾向、bonus_share の有限性を確認する。Atari KFM は 100k experience で失敗・NaN・clip_ratio の張り付きを確認する。手順と実行結果は実装記録へ追記する。

`run_validation.py smoke --env GridMaze Atari --label smoke-r1` は各100k experienceを順に実行する。PRD098 の短縮設定を再利用し、既存 Run の連鎖へ `run.@rnd` を追加する。実行体を保存し、出力と8指標の件数・範囲・前後四分の一の平均を記録する。再実行時に別の実行体なら新しい label を使う。
