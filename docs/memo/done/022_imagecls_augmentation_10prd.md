# ImageCls 過学習対策①: train 専用データ拡張（水平フリップ + RandomResizedCrop）

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 過学習対策の対になる構造側施策は `023_imagecls_gap_head_10prd.md`（最終段 Flatten→GAP）。
> 両者は独立に実装・A/B 可能。効果切り分けのため **1 個ずつ** 入れて評価する。

## Context（背景・目的）

ImageCls（Food-101 / 101 クラス / 224×224）は **train accuracy 高・eval accuracy 低の過学習**
状態（ユーザー確認, 2026-06-25。`38_agent/03_accuracy` と `21_eval/01_test_reward` の乖離）。

主因の一つが **データ拡張ゼロ**:

- `ImageDataSource::get` は `Scale(224,224)` のリサイズのみで、flip / crop / jitter を一切しない
  （`ImageData.hpp:101-103`）。
- RL ループは train を **毎 step ランダムに無限サンプリング**する（`ImageClsEnv::FetchRandomImageState`,
  `ImageClsEnv.cpp:84-110`）。エポックの概念が無く、同一画像を素のまま何度も見るため実質「暗記」になりやすい。

目的: **train 経路にだけ軽量なデータ拡張**を入れて正則化を効かせ、eval accuracy を上げ
train-eval gap を縮める。eval 経路は無加工のまま（評価の一貫性を保つ）。

## 確定した設計判断

1. **拡張の適用点 = `ImageClsEnv::FetchRandomImageState`（train mode のみ）**。eval mode は無加工で素通り。
   - 理由: (a) train/eval 分岐が `mode` 引数で自然に書ける (b) 再現性のため env が既に持つ
     `rnd_`（`RandomHolder`）を乱数源に使える (c) `ImageDataSource::get` を純粋な I/O に保てる
     (d) torch dataset である `ImageDataSource` に乱数状態を持ち込まない。
   - 代替案（却下）: `ImageDataSource` に `augment` フラグ＋内部 RNG を持たせる。→ 乱数源が env と二重化し
     seed 再現性の管理が複雑化する（同一 seed で run が再現する性質を重視。cf. 015 deterministic）。
2. **乱数源 = env の `rnd_`**。seed 固定で完全再現。`num_envs` 並列でも各 env インスタンスの `rnd_` で独立に進む。
3. **phase1 の拡張 = ① 水平フリップ p=0.5 ② RandomResizedCrop 相当**（scale ∈ [0.7, 1.0]、
   aspect ratio ∈ [3/4, 4/3]、出力 224×224）。color jitter / RandAugment は **phase2（任意・別途）**。
   - Food-101 は皿・料理の自然画像で左右対称性があり hflip は安全。scale 下限は 0.7 と控えめ（料理の主要被写体を切り落としすぎない）。
4. **config で ON/OFF・強度を制御**（`ImageClsEnv.*.augment.*`）。既定は後方互換のため **OFF**、
   `ImageCls.txt` の baseline 側で **train ON** にする。これにより拡張なし build と config だけで A/B 可能。
5. **tensor 演算で実施し、出力は [3,224,224] uint8 を厳守**（obs spec 不変）。
   - flip = `tensor.flip({2})`（W 軸）。
   - crop は実装方式を Codex 裁量とする（候補: ⓐ uint8→float→ランダム矩形 crop→`interpolate` で 224 復元→uint8、
     ⓑ reflect-pad 後に 224 ランダムクロップ＝interpolate 不要で軽量）。**hot path なので軽量側を優先**。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `100e483`（行番号は現 working tree 基準。env 系は本 PRD 対象の未コミット変更を含む）。

- `ImageClsEnv::FetchRandomImageState(mode)`（`ImageClsEnv.cpp:84-110`）: `IsEval(mode)` で source 切替 →
  `rnd_->RandUint64() % data_size` で index 抽選 → `source->get(idx)` → `obs.Set(kGrid, example.data)` /
  `obs.Set(kVector, example.target.clone())`。**ここが train/eval 共通の唯一の取得口**。
- `ImageClsEnv` は `anet::RandomHolder` 継承で `rnd_` 利用可（`ImageClsEnv.hpp:40`, `.cpp:84-91` で使用済み）。
- `ImageDataSource::get`（`ImageData.hpp:88-120`）: `LoadFile` → `Scale(224)` → `[H,W,3]` uint8 →
  `permute({2,0,1})` → `clone()` で `[3,H,W]` uint8 を返す。
- obs spec `kGrid`: `uint8 [3, image_height, image_width]`（`ImageClsEnv.cpp:54-58`）。train/eval で同一型。
- `ImageClsEnvConfig`（`ImageClsEnv.hpp:15-38`）: `ANET_READ_CONFIG` で平坦に読む既存パターン。ここに augment 設定を追加する。
- `Reset` / `Step` は共に `mode` を受け `FetchRandomImageState(mode)` を呼ぶ（`.cpp:120-168`）。

## 設計方針

### A. config 追加（`ImageClsEnv.hpp` / `.cpp`）
`ImageClsEnvConfig` に augment フィールドを追加し `ANET_READ_CONFIG` で読む（既存平坦パターン踏襲）:

- `augment.enabled`（bool, 既定 false）
- `augment.hflip_p`（double, 既定 0.5）
- `augment.rrc_scale_min` / `rrc_scale_max`（既定 0.7 / 1.0）
- `augment.rrc_ratio_min` / `rrc_ratio_max`（既定 0.75 / 1.333）

### B. 適用点（`ImageClsEnv.cpp`）
`FetchRandomImageState` で `!IsEval(mode) && config_.augment.enabled` のとき、`obs.Set(kGrid, ...)` の
**直前**に `example.data` へ拡張を適用。各変換の確率/パラメータは `rnd_` からサンプル。

### C. helper（`ImageClsEnv.cpp`）
名前付き helper（無名 namespace は使わない方針）または private メンバ `ApplyTrainAugment(tensor, rnd)` を追加。
入力・出力とも `[3,224,224] uint8`。

### D. ProfileScope
拡張は per-step hot path のため、helper 先頭に `ANET_PROFILE_SCOPE(augment)` を入れる（性能測定ルール）。

## 非対象（Out of Scope）

- eval 側拡張 / TTA、color jitter・RandAugment（phase2）、Mixup / CutMix（別検討）。
- `ImageDataSource` の責務変更、obs spec / dtype 変更、リサイズ方式（短辺リサイズ等）の変更。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/envs/imagecls1/src/ImageClsEnv.hpp` | `ImageClsEnvConfig` に `augment.*` フィールド追加 |
| `core/envs/imagecls1/src/ImageClsEnv.cpp` | config 読み出し、`FetchRandomImageState` の train 分岐、`ApplyTrainAugment` helper、profile |
| `apps/runner/config/ImageCls.txt` | `ImageClsEnv.baseline.augment.*`（train ON）追加 |

## 受け入れ基準

1. **ビルド緑**（x64-Debug）、既存テスト緑。
2. **eval 無加工**: eval mode では拡張を通らない（`augment.enabled` でも eval は素通り）ことを確認。
3. **再現性**: 同一 seed・同一系列で 2 run の拡張結果が一致（`rnd_` 由来の決定性。簡易 UT か手動確認）。
4. **効果（ユーザー実測）**: 同設定・seed 違いの複数 run で、eval accuracy 終盤平均が **ブレ幅を超えて改善**、
   または train-eval gap が縮小する（cf. 構成比較はブレ幅基準・run config dump 確認）。
5. **perf**: `90_perf/12_exp_step_per_sec` の劣化が許容内（拡張は CPU env 側。Tracy / metrics で確認）。

## 正直なリスク

- env は CPU 側で既に律速気味（cf. 020 device transfer findings: perf 律速は env 側）。拡張追加で step/sec が落ちうる。
  → reflect-pad+crop 等の軽量実装を優先し、profile で実測。重ければ flip のみ phase1 に縮める判断もあり。
- 拡張が強すぎると train を阻害。scale 下限 0.7・hflip のみ等の控えめ既定から始める。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

- 機能確認は runner を online モードで起動し、ImageClsView で eval 画像が無加工・train が拡張されることを目視。
- 効果（eval accuracy / gap）はユーザーが seed 違い複数 run の終盤平均で評価。

## 後続

1. 実装メモ `022_..._20impl.md`（必要なら）→ Codex 実装 → 受け入れ緑 → ユーザー A/B 評価。
2. 効果が出たら color jitter / RandAugment（phase2）を検討。023（GAP）と組合せて最終 gap を評価。
