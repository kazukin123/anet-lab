# PRD 084 NoisyNet 実装計画・最新版

`NetworkForwardContext::GetDefault()` は Module 固有の設定値を持たず、各計算部品の既定動作を明示的に要求する。Noisy の計算部品はこれを μ-only と解釈する。名前付き設定の項目不足とは区別し、暗黙の fallback は行わない。

`Network::Forward()` は context 引数に `NetworkForwardContext::GetDefault()` を既定値として持ち、省略時にも同じ伝達経路で各 Module の既定動作を使う。別 overload は追加しない。`ForwardUpTo()` と内部 Module / Head の context 引数は必須のままとする。

## 概要

DefaultDQN の TD・QR・IQN、Dueling を対象に、基本 NoisyNet、診断、Atari 結合・性能検証の3段階を実装する。本書 `docs/memo/084_noisynet_20impl.md` を実装の正本とする。

## 主な変更

### API・設定・基本演算

NN 側に以下を追加し、Actor・Learner は中身を解釈せず渡す。

| 型・API | 契約 |
|---|---|
| `NetworkForwardConfig` | 解決済み設定データの不変な受け渡し。具象 Module に依存するフィールド・設定型・解釈を持たない |
| `NetworkForwardState` | 呼び出し側が所有する不透明な箱。ε・抽選時刻・専用 RNG を保持 |
| `NetworkForwardContext` | 設定・状態と、時計の種類／値、`episode_start`、結合 batch の行対応を渡す |
| `Network::MakeForwardState(seed)` | private Resource を生成 |
| `Network::Forward` / `ForwardUpTo` | Forward の context 省略時は `GetDefault()` を使う。ForwardUpTo の context は必須 |
| module・Head の forward／検証 | 固有設定の解釈、必要項目、保持制約、再利用可否を局所的に検証 |

Body・branch・block・Head まで context を伝える。旧 overload は残さない。dummy forward、既存診断、対象外 Agent、`TensorDictFunction` の返す処理は明示的な `GetDefault()` を渡す。`TensorDictFunction` 自体の関数型は変更しない。

#### 実行設定の責任境界（2026-10-03 修正）

- `NetworkForwardNoisyConfig` と共通層の Noisy 専用読み取り処理を削除する。具体的な設定型・解釈・検証は利用する `NetworkModule` の具象クラス内部に閉じ、Head の計算部品にも同じ原則を適用する。
- `NetworkForwardConfig` は指定された項目配下の設定データをコピーして不変に保持し、読み取り専用で渡す。利用する Module が増えても、共通型、伝達経路、Actor・Learner、他の Module の変更を要求しない。
- 共通の射影ヘルパーは演算の共用に限定し、具象設定を共通層へ移す手段にしない。既存の検証時期・休眠の契約は維持する。
- `GetDefault()` は各計算部品の既定動作を明示的に要求し、Module 固有の設定値を持たない。Noisy での μ-only という解釈は具象クラスが所有する。名前付き設定の項目不足を既定動作へ読み替えない。
- Public surface は `NetworkForwardConfig` と `Network::Forward` / `ForwardUpTo`。独自設定を読むテスト用 Module と Head を追加し、共通型・呼び出し側・既存 Module を変更せず通常・部分 forward に反映できることを RED→GREEN で確認する。元の設定データを変更しても解決済み設定が変わらないことも確認する。

設定を次で固定する。

| 対象 | キー・既定値 |
|---|---|
| カタログ | `nn_forward.[key].noisy.mode / hold / share / count` |
| Actor の参照 | `actor.[key].nn_forward_key` |
| Learner の参照 | `learner.current.nn_forward_key`、`learner.target.nn_forward_key` |
| Body Linear | `noisy.enabled=false`、`noisy.sigma0=0.5`、`force_fp32=false` |
| Head 最終射影 | `head_projection.noisy.enabled=false`、`head_projection.noisy.sigma0=0.5` |
| σ decay | `learner.noisy_weight_decay=0` |

- 共通設定に `act_sample`（sample/call/batch）と `mu_only` を定義する。学習 Actor・Learner 両用途は前者、評価 Actor は後者を選ぶ。既定葉は `?=`、実験・Run・選択宣言は `=` とする。
- 参照欠落・未知キー、記述された認識済み値の型・列挙・値域、sample の episode/batch は設定解決時に検証する。必要項目の不足、時計との不整合、`online_reuse` は各計算部品が初期化時に検証する。休眠は PRD §11.2 に従う。
- Body と Head は既存 NN 機能グループ内の射影ヘルパーを共用する。既存の μ の parameter 名・初期化を保ち、Noisy 適用時だけ `sigma_weight` / `sigma_bias` を追加する。
- sample は μ と σ の Linear を分ける等価式で計算する。ε は dim 0 以外へ broadcast し、`[B,q,p]` の重みを実体化しない。μ-only は σ を計算グラフへ入れない。
- SN は既存機構で μ だけを正規化する。`force_fp32` は Noisy と独立に適用し、Head・SN の FP32 保護と既存 AMP の配置・実効精度を維持する。

### 状態・学習・診断

- Actor は1箱、Learner は現在値・target 用の2箱を所有する。ε は箱と module 実体ごとに保持し、既存 acting・学習・τ の RNG とは独立した named seed を使う。
- Actor の時計は行動選択ごと、Learner は実際の更新ごとに進める。`count` は episode 境界でリセットしない。保持中の shape・device が要求と不整合なら黙って再初期化せず fail-fast にする。
- target 構築へ同じ設定・箱・`learn_step` を渡す。Munchausen の `[2B]` 入力は元遷移との行対応を明示し、current/next に同じ `[B]` の ε を使う。Double-DQN と `online_reuse` は PRD §7.1–7.2 に従う。
- Network は μ・σ の対応、weight/bias、所属 branch を持つエントリを列挙する。利用先は optimizer 分類と診断だけとする。通常 AdamW／FusedAdamW に σ 専用 group を設け、OFF 時は既存の単一 group を保つ。
- μ-only の現在値では σ を更新・decay しない。sample の学習では、勾配が成立した σ も既存の全体 grad norm・clip に含める。
- clone・hard/soft copy・snapshot・保存読込は μ・σ を含める。ε・時計・RNG は含めず、snapshot 更新も再抽選境界にしない。

診断は `47_agent_noisy` にまとめる。

- 8統計：`noisy_sigma_mean_abs_{feature|readout}_{weight|bias}`、`noisy_sigma_l2_ratio_{feature|readout}_{weight|bias}`。
- 3 probe 指標：`noisy_q_abs_diff`、`noisy_action_disagreement_ratio`、`noisy_action_entropy`。

統計は既存 `feature_key` の依存閉包で分類する。ratio の分母は **σ を持つ層の対応する μ だけ**を使い、SN 層では実効 μ に換算する。既存 weight norm 61〜64 から σ を除く。分母0は ratio を `NaN` とし、成立する mean-abs まで無効にしない。

probe 設定は B=128、R=32、K=32、購読の既定間隔503更新とする。キーは `learner.noisy_probe.probe.batch_size`、`learner.noisy_probe.num_draws`、`learner.noisy_probe.iqn.num_taus` とする。

専用 RNG で一様・非復元抽選し、更新前に FP32・eval・NoGrad・固定 midpoint τ で測る。Noisy mode 以外の Module / Head 固有の opaque 実行設定を揃えて、μ-only を1回、sample を **B状態・batch共有でR回**評価する。同じ反復内では状態間で ε を共有し、反復間では独立にする。三指標は推論を共用して逐次集計する。統計だけの購読では probe を実行しない。

既存 probe は μ-only、actual 指標は通常 forward の実際の出力を使う。μ-only の追加 Q 統計は公開しない。未測定・不足・対象なしは既知キーなら `NaN`、未知キーだけ `nullopt` とする。

## テスト

Public interface / surface は Network の構築・Forward・ForwardUpTo・MakeForwardState、設定読み込み、Actor の行動選択、Learner の更新、GetScalar、clone・同期・保存読込、runner の解決済み設定・Run 出力とする。

各挙動を1テストずつ RED→GREEN とし、GREEN 後だけ整理する。

1. **経路の tracer bullet**：設定から構築した Noisy なし Network に context を通す。全呼び出しを移行し、無効時回帰を確認してから μ・σ を追加する。
2. **演算**：参照計算との出力・入力／μ／σ 勾配一致、bias 有無、σ₀=0、符号付き σ、batch/sample、rank 2/3、初期化、SN、精度を確認する。
3. **状態・target**：保持境界、lane 更新、同一時計の再 forward、Actor 間分離、snapshot、TD/QR/IQN、Dueling、Double-DQN、Munchausen 全方式、再利用条件を確認する。
4. **optimizer・保存**：独立 decay、μ-only の σ 凍結、σ込み clip、parameter 登録、clone・同期・保存読込を確認する。
5. **診断**：既知の Q・行動頻度から三指標を検証する。feature/readout 統計、非Noisy層の分母除外、NaN、購読 gating、診断 ON/OFF の更新結果・RNG・ε・永続 buffer の一致を確認する。

## 検証

既存 NN・DQN・Munchausen テストを拡張して `[noisynet]` を付ける。MSVC 初期化経由で Debug 全体ビルド、関連テスト、全 core テストを実行する。


```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
core/anet-core/bin/Debug/anet-core-test.exe
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

**コード編集前の基準採取**は、実行体・依存 DLL・解決済み設定・seed・checkpoint・出力を `.scratch/prd084/old/` に保存する。変更後は `new/` と比較する。

- 回帰対象は Atari 既定チェーン、SN付きサンプル効率優先チェーン、CPU LunarLander。deterministic backend と greedy_dist 無効化を前後で揃える。
- サンプル効率優先チェーンは9月25日の実験記録にある `v5_iqn_impala_x2 > a5 > a5_apex > va_base > hard500 > rr4 > munch > rfit > a5_metrics > btrnet > btrsn12 > cap2m > eval2ch` を使い、128 env・Breakout・seed=1で固定する。
- 初期回帰予算は500k経験とし、終了予算だけを変更する。学習スケジュールは維持する。数値 metrics を `(tag, step, value)` に正規化して checksum 比較し、時刻・性能・資源計測・追加 Noisy 指標を除外する。欠損・点数差は失敗とし、旧 OFF checkpoint の読込も確認する。

**Atari 結合は2本**とする。上記チェーンの4層へ Noisy を追加した AMP継承・call/batch と、`run.@nz_reference` の FP32・episode/sample を、Breakout・seed=1・500k経験で実行する。

参照 profile は4層を default 初期化・σ₀=0.5とし、Body の2 FC に `force_fp32=true` を指定する。対照の `run.@nz_reference_off` は初期化・精度を同じにして Noisy だけを外す。両結合 Run は全診断を購読し、構成別の成立値／NaN、複数回の eval・probe、正常終了を確認する。不足時は条件を維持して予算を延長する。

**性能測定**は参照 profile から1軸ずつ変更し、交互実行で行う。OFFとの差、batch/sample、call/count/episode、SNなし/spectral/spectral_cap、FP32/AMP継承、診断 ON/OFFを測る。count の比較値は32とする。SN軸は Noisy Body Linear へ実際に SN を付け、学習比較構成の ON/OFF も測る。throughput・時間・追加メモリを記録し、速度向上やスコア改善を合否条件にしない。

現行 NN・DQN 設計、設定例、PRD の明確化箇所を同期し、専用検証手順は `core/anet-core/testdata/prd084/README.md` にまとめる。設定の既定葉監査と既存実効値比較、責任境界レビューを行う。3段階すべての受け入れ完了を PRD 084 の完了とする。

## 前提

当初計画の承認時における対象 PRD の SHA-256 は `3C6A21DCB935607576F8A13E9894A85F9EB975C7FC619C992529AF02CF46C443`。未決事項のブロッカーは0。前回の指摘は次の扱いで固定し、実装時に PRD の該当記述も揃える。

- **診断の受け入れ**：指定の Atari 構成では readout の4統計と probe の3指標を成立値、feature の4統計を正常な `NaN` とする。feature 統計の成立は別の契約テストで確認する。
- **episode 保持**：同じ時計の `episode_start` は一度だけ反映し、再 forward で再抽選しない。
- **初期 σ/μ 比**：約0.87は SN なし・default 初期化での目安とする。厳密テストは実際の μ・σ から期待値を計算する。

AMP 集約、Head 分離、ノイズ込み重みの SN、完全な Run 再開は対象外。無関係な未コミット変更は保持する。

## 実装・受け入れ結果（2026-10-04）

以下は当時の実施結果である。レビューで参照 profile と PRD の不一致、CUDA AMP 内の選択的 FP32 結合検証、Learner sample の性能測定の不足を確認したため、全受け入れ完了の判定は取り下げる。修正の正本は [21impl](084_noisynet_21impl.md) とする。Module / Head固有の設定は具象クラス内に閉じ、`Network::Forward(input)` は唯一の宣言の既定引数で `GetDefault()` を使う。

- 基本・状態・target・optimizer・保存: MSVC初期化経由の全体Debug build成功。全core 657passed /1skipped /34597assertions /exit0。skipはCUDAが利用可能な場合の非対応環境ケース。
- 診断: 8統計・3 probe指標、NaN、購読gating、独自Head設定の保持、24target構成の非干渉を契約テストで確認。
- 回帰: 500k経験のOFF3構成は旧基準と全数値checksum・点数・既存全キーの文字列値が一致。旧OFF checkpointの読込もexit0。
- Atari結合: AMP継承call/batch・FP32 episode/sampleの2本は500k経験・exit0・eval各2回・probe各10回。readout4統計とfeatureの正常なNaNを確認。
- 性能: 11比較を2反復のA/B/A/B、計44Runで実測し、全Run exit0。比較軸以外の全設定、実行体、交互順、診断の実行有無を照合。速度・スコアの改善を合否条件にしていない。

実行体SHA・Run名・数値比較・性能表は[専用検証記録](../../core/anet-core/testdata/prd084/verification.md)。基準・Run・checkpoint・設定・ログは `.scratch/prd084/old/` / `new/final-verified/` / `performance-final/` に保持する。
