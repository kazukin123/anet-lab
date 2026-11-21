# PRD084 レビュー指摘の修正計画

## 概要
7 クラスの重複を具象の射影 Module に集約し、設定の責任境界を維持する。行対応による GPU→CPU 同期と毎 forward の設定再解釈を取り除き、参照 profile を PRD §12 に揃えて検証を補完する。承認済みの本計画を実装の正本とする。無関係な変更と過去の Run artifact は保持する。
NN の RL 用語は今回変更しない。episode_start、時計の Actor／Learner／Probe 区分、保持値 episode は後続の検討事項として記録する。

## 主な変更
- 内部の具象 LinearProjectionModule を Body Linear と6つの Head が利用する。構造設定・実行設定の具体型、宣言検証、利用時の制約、再利用検証、保持・抽選はこのクラス内部に集約する。各 Head は特徴量の取得と TD・QR・IQN・Dueling の出力構成を担当する。
- 射影部品は既存の torch::nn::Linear を利用する。登録階層、μ・σ の parameter 名、初期化順、SN の buffer・更新条件、FP32 保護を維持する。Body の SN 経路では、既存処理で計算した μ の出力を射影部品へ渡す。
- NetworkForwardConfig は現在の不透明・不変な受け渡しを維持する。共通の Noisy 設定型・読み取り処理は追加しない。数値ヘルパーは演算だけを共有する。
- 解釈済み設定を射影 Module 内部にキャッシュする。同一性は既定設定かどうかと設定内容で判定し、キー名・アドレスだけでは判定しない。初期化時に準備し、直接 forward する経路では初回に作る。共有 Network の同時利用を同期で保護し、Tensor 演算中はロックしない。以後は shape・device・時計など実行時の条件だけを検証する。
- NetworkForwardContext::row_mapping を、不変な NetworkForwardRowMapping に置き換える。CPU の index 列と元行数から構築し、型・範囲を一度だけ検証して使用 device へ転送する。各層は行数・device を確認して再利用し、min/max の取得や CPU 転送を行わない。
- Munchausen の2つの結合経路は、元の B 行と [current, next] の対応を CPU 側で構築する。current／next に同じ ε を使う契約を維持する。旧 Tensor 形式の alias・overload は残さない。

### 設定・文書
- nz_reference／nz_reference_off は既存チェーンの AMP と標準 call/batch を継承する。4層の default 初期化、σ₀=0.5、Body 2 FC の force_fp32=true は維持し、ON／OFF は Noisy の有効化だけを変える。
- episode/sample は参照 profile に含めず、2本目の結合 Run の Actor 設定として指定する。Learner は call/batch を維持する。
- PRD、現行 NN・DQN 設計、設定例、core/anet-core/testdata/prd084/README.md を同期する。PRD の完了状態行を削除する。
- 過去の全体 FP32 の結果は当時の条件による記録として残し、新契約の証拠と区別する。以前の全受け入れ完了の記述には、今回確認した不足を明記する。
- PRD084 の編集で BOM が外れた11ファイルだけ、編集前の BOM を復元する。改行コードと無関係な変更は保持する。

## テスト
Public interface / surface は Network の Forward・ForwardUpTo・ValidateForwardConfig、構築・clone・同期・保存読込、Actor の行動選択、Learner の UpdateFromBatch、設定解決と Run 出力とする。

既存テストを GREEN と確認してから共通化する。新しい行対応の契約とキャッシュの挙動は、1挙動ずつ RED→GREEN とする。
- Body と全6 Head の出力・勾配、保持境界、同一時計の再 forward、μ-only、SN、rank 2／3、parameter 名、clone・同期・保存読込。
- 同名・異内容の設定、設定オブジェクトの寿命、複数 Actor の同時 forward、診断非干渉。独自設定を読む既存 Module／Head の契約テストを維持する。
- 行対応の不正 index・元行数・入力行数・device を検証し、CPU／CUDA で参照計算と一致させる。各射影から mapping の CPU 転送が消えたことをコード・計測で確認する。
- CUDA autocast 下の Body FC force_fp32 ON／OFF、Head と SN の FP32 保護。
- 最初の tracer bullet は、設定から構築した Body・Head の Forward に行対応を渡し、参照出力と一致すること。以後は行対応の不正入力、キャッシュ内容・寿命・同時利用の順に、1テストの RED→GREEN を完了してから次へ進む。

## 検証

MSVC 初期化経由の Debug 全体ビルド、関連テスト、全 core テスト、設定の既定葉監査を行う。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
core/anet-core/bin/Debug/anet-core-test.exe
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

- コード編集前に現実装の実行体・依存 DLL・設定・seed・checkpoint を .scratch/prd084/review-fix/before/ に保存する。変更後は別の出力先を使う。
- OFF 回帰: Atari 既定、SN 付き効率優先チェーン、CPU LunarLander を各500k経験。従来の除外条件で数値 metrics checksum・点数・既存設定値を照合し、旧 OFF checkpoint の読込。
- Atari 結合2本: 学習比較構成の AMP 継承・call/batch と、修正参照構成の AMP 継承・Body FP32・Actor episode/sample。Breakout・seed=1・128 env・500k経験、全診断購読、正常終了、複数 eval／probe、readout 成立値と feature 正常 NaN。不足時は条件を保って予算延長。
- 性能12比較・計48 Run: 2反復 A/B/A/B。Noisy ON/OFF、Actor batch/sample、Learner current・target batch/sample、Actor call/count32 と call/episode、Noisy Body SNなし/spectral/spectral_cap、同じ AMP 下の force_fp32 ON/OFF、診断 ON/OFF、学習比較構成の3 SN 条件で Noisy ON/OFF。保持比較は両側 sample。
- 性能予算224k経験、warmup既存200k。throughput・時間・追加メモリと全解決済み設定を保存し、比較軸以外の一致を確認。速度・スコア改善は合否条件にしない。

## 前提

- 参照構成は既存チェーンの AMP と call/batch を継承し、Body 2 FC の force_fp32=true、4射影の default 初期化・σ₀=0.5 を維持する。
- 結合 Run は Breakout・seed=1・128 env、Learner call/batch とし、2本目の Actor だけ episode/sample にする。
- NN の RL 用語の改名は本段階の対象外とし、後続の22implで扱う。旧 Run artifact と無関係な変更は保持する。
- 性能の改善・スコアの改善を受け入れ条件にしない。

## 実施結果（2026-10-04）

本計画の修正と受け入れは完了。詳細は core/anet-core/testdata/prd084/review_verification.md に記録する。PRDには完了状態を書かない。

- 具象LinearProjectionModuleへの集約、内容別キャッシュ、不変NetworkForwardRowMapping、MunchausenのCPU行対応、AMP継承profileの修正を実装。
- Debug全体・Release Runnerのビルドはexit0。Noisy47件、関連304件、全core665件中664成功・1skip。CUDA autocastとhot forwardのCPU転送・scalar同期0件を確認。
- before/の実行体・DLL・設定・seed・checkpointを保存。after-verified/のOFF3本は各500kで数値点・checksum・既存設定値が一致。旧OFF checkpoint読込もexit0。
- Atari結合2本は各500kでexit0、eval2回・probe10回。参照構成はAMP内のBody2FC FP32、Actorのみepisode/sample、Learner call/batchを確認。
- performance/の12比較48 Runは各224k・2反復A/B/A/Bで全exit0。全設定の非軸一致、warmup200k、同一実行体、非重複を確認し、時間・throughput・メモリを保存。
- 既定葉監査errors0、指定11ファイルのBOM/LFを復元。過去の全体FP32結果と失敗した検証起動は条件・理由を明記して保持。
- NNのRL用語は合意どおり変更せず後続事項として残す。無関係な変更は保持し、commit/stage/pushは行わない。
