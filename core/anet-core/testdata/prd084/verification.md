# PRD084 検証記録

本書は実装・検証の記録。実装の正本は `docs/memo/084_noisynet_20impl.md`。以下は修正前の条件による実施結果である。レビューで参照profileとPRDの不一致、CUDA AMP内の選択的FP32結合検証、Learner sample性能測定の不足を確認したため、全受け入れ完了の判定は取り下げる。過去のRunと数値は保持し、現行の補完結果は [review_verification.md](review_verification.md) に分ける。

## 責任境界レビュー

- `NetworkForwardConfig` は const ConfigData と既定要求の識別だけを持ち、具象設定型・Noisyフィールド・Noisy読み取りを持たない。
- Noisy structural / forward / holding 型は LinearModule と6具体Headの private 型。factory は宣言検証をその具体型へ委譲する。
- Body/branch/block/Head は同じ context を渡す。Actor/Learner は設定内容を解釈せず、箱・時計・行対応を所有して渡す。
- 共通射影ヘルパーは σ の数学的初期化と factorized Linear演算だけ。σエントリの利用先は optimizer 分類と診断のみ。
- 拡張テストは独自設定を読む Module/Head を既存伝達経路へ登録し、共通型・caller・既存Moduleを変更しない。probe側も独自Head設定をμ/sampleの両側へ保つ。
- `Network::Forward(input)` は唯一の宣言の既定引数で GetDefault を使う。旧MuOnly API・別overload・global NetworkForwardNoisyConfigを残さない。ForwardUpTo/内部forwardのcontextは必須。

## 基本・診断

- 最新全体Debug build: `.scratch/prd084/noisy-optimizer-save-coverage-3-build.log` exit0。
- 最新全core: `.scratch/prd084/final-all-core-test-3.log` 657passed /1skipped /34597assertions /exit0。skipはCUDAが利用可能なためCUDA非対応環境のケースを実行しなかったもの。
- 関連NN/DQN: `.scratch/prd084/final-related-test.log` 290cases /26890assertions /exit0。
- RED/GREEN・fixture修正・coverageの履歴は `.scratch/prd084/progress.md`。既存挙動の追加coverageはREDを捏造しない。
- σの符号、σ0=0、gradient一致、rank/bias/share、保持境界、SN/精度、6Headの同期・保存、通常/Fused AdamWのgroup/decay/moment、clip、診断gating/NaN、24target構成の非干渉を確認。
- 既定葉監査: `.scratch/prd084/default-leaf-audit-final.log` errors0。

## Atari 結合

Release SHA256: `45A7DAE3DD85179F320B23CE182F03A8A8B3F4119E632FAB4F0CED1AEBE8604F`。

| 構成 | Run | 経験 | 終了 | eval | probe | readout統計 |
|---|---|---:|---:|---:|---:|---|
| AMP継承 /call /batch | run_20261004-033233_prd084_final_nz_learning | 500k | 0 | 2 | 10 | 3統計10回、bias ratio9回（初期分母0はNaN） |
| FP32 /episode /sample | run_20261004-033932_prd084_final_nz_reference | 500k | 0 | 2 | 10 | 4統計各10回 |

出力は `.scratch/prd084/new/final-verified/`。全11行の購読を確認し、feature4統計は対象なしNaN（既存Observerが出力省略）。feature成立値とNaNは契約テストで確認する。両Runはgreedy_dist=0をCLIで固定。初回 `new/final` の誤設定Runはexit-1で未受け入れとし、上書きせず保存。

## OFF 回帰・旧checkpoint・性能

最終OFFの3構成はすべて旧基準と完全一致。欠損・点数差・数値差なし。既存全キーの文字列値も一致し、追加キーは各 `*.settings-comparison.json` に記録する。

| 構成 | 点数 | 正規化SHA256 |
|---|---:|---|
| Atari既定 | 78,986 | 9adcb0b9e1e5d6cb95b1666ca6716284f30a684b1ba294066b71c6008e93fb29 |
| 指定SNチェーン | 122,332 | 5696fb4de42f8e308ec0740887697971d98f356dfd4f3c9147b42879132be57e |
| CPU LunarLander | 88,184 | b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5 |

旧OFF checkpoint: `run_20261004-043643_prd084_final_old_checkpoint_lunarlander` exit0。ログで `LoadNetwork: model_size=1276270` と `learner_size=1192239` の読込完了を確認。checkpoint SHA256は `43674EF8A169FA7C16A388FC4F2532C9D4F4537F3C5B1AF397606994B22D35FE`。

## 性能測定

`.scratch/prd084/performance-final/summary.json` / `summary.md` に11組・44Runを保存。全Run exit0。同一Release SHA、seed、終了予算、200k warmup、全設定キーの比較軸以外の一致、A/Bの交互順と非重複、診断ONの3 probe指標とOFFの購読・出力不在を検証した。

共有・保持方式の軸はActorの `nz_episode` だけを変更し、Learner current/targetのcall/batchは維持する。precision軸はBodyのforce_fp32とLearner AMPを変更する。学習比較の3組は既存AMPと初期化を維持し、Noisy Body LinearのSN modeごとにNoisy enabledだけを比較する。各側の数値は2反復の平均。CPU private bytesは500ms間隔のサンプル最大、working-setはOSのpeakを記録する。GPUはデバイス全体の起動前との差で、allocator固有の追加量ではない。

Breakout / seed=1 / 128 env / 224,000経験。各比較は A/B を2反復。速度・スコア改善は合否条件にしない。

GPU はデバイス全体の5秒間隔の使用量から起動前の値を引く。CPU はプロセスの peak private bytes。経験/s は起動から終了までの全体値、学習中経験/s は warmup 200k 後のメトリクス平均。

OFF 行は A=Noisy ON / B=Noisy OFF。その他の比較は README の行対応を参照。

| 比較 | A 秒 | B 秒 | A 経験/s | B 経験/s | A 学習中経験/s | B 学習中経験/s | CPU追加 MiB B-A | GPU追加 MiB B-A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostics | 168.17 | 172.79 | 1333.1 | 1296.7 | 279.8 | 261.2 | -9.93 | -0.50 |
| hold_call | 150.15 | 153.68 | 1492.6 | 1460.1 | 370.0 | 344.2 | -15.11 | -5.00 |
| hold_count32 | 155.34 | 162.07 | 1442.8 | 1394.9 | 379.2 | 317.6 | 17.87 | 42.00 |
| learning_sn_none | 77.54 | 81.67 | 2888.9 | 2743.0 | 2089.2 | 1928.9 | 72.43 | -187.00 |
| learning_sn_spectral | 76.18 | 78.90 | 2941.5 | 2839.1 | 2105.3 | 1932.3 | 53.88 | 40.00 |
| learning_sn_spectral_cap | 75.68 | 78.67 | 2959.6 | 2847.4 | 2138.4 | 1957.9 | 64.07 | 41.00 |
| off | 186.26 | 155.00 | 1209.2 | 1445.9 | 279.4 | 396.0 | -65.19 | 22.50 |
| precision | 167.71 | 84.36 | 1336.9 | 2672.3 | 277.4 | 1567.7 | -5073.21 | -1194.50 |
| share | 161.99 | 145.71 | 1382.9 | 1537.5 | 352.2 | 467.1 | -3.99 | 53.00 |
| sn_cap | 159.08 | 163.35 | 1408.3 | 1372.0 | 317.6 | 316.9 | -10.17 | 236.00 |
| sn_spectral | 163.82 | 165.82 | 1368.1 | 1351.7 | 345.3 | 309.3 | -22.27 | -223.00 |

## 完了確認

公開API・設定・責任境界・現行NN/DQN設計の日本語/英語・設定例・専用検証手順を同期。旧 overload / MuOnly factory / global NetworkForwardNoisyConfigは残さない。無関係な作業中ファイルは保持し、stage・commit・pushは行っていない。
