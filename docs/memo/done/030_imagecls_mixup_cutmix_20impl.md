# ImageCls Mixup/CutMix 実装メモ

## 概要
`docs/memo/030_imagecls_mixup_cutmix_10prd.md` に従い、ImageCls の Learner 側へ batch-level Mixup/CutMix を追加する。Env と `ImageClsView` は変更せず、`uint8 grid -> NetworkBoundaryPreprocessor -> float / 255` の既存契約を維持する。

既定は `ImageClsAgent.mixup.enabled=false` で後方互換にし、runner 設定では現在の active branch/run_name を保ったまま ConvNeXt run 用の Mixup/CutMix 設定と `target_prob_mix` scalar metric を追加する。

## 主な変更
- `ImageClsAgentConfig` に `mixup.*` と `learn_log_interval` を追加し、`prob`/`switch_prob` は `[0,1]`、`*_alpha` と `learn_log_interval` は `>=0` を `ANET_SYSTEM_ERROR` で検証する。
- `ImageClsLearner` は `RandomHolder` を継承し、Agent の実効 seed（`GetSeed()`）から `SeedMaker(...).MakeNamedSeed("learner")` で派生した seed を受け取る。prob/switch/lambda/bbox/perm は `GetRandomGenerator()` と device 別 `torch::Generator` で生成し、torch global RNG には依存しない。
- `UpdateFromBatch` の Forward 前に `grid` を Mixup または CutMix し、loss は `lam * CE(target_a) + (1-lam) * CE(target_b)`、accuracy は混合前 target、`target_prob_mix` は mixed target 確率の batch mean とする。
- `ImageClsUpdateResult::GetScalar()` は `loss`、`accuracy`、`target_prob_mix` だけを返す。`mix.lambda` や `mix.mode` は常設 scalar にしない。
- `LOG::verbose()` は `learn_log_interval > 0` かつ指定間隔のときだけ出し、Mix 無効時も learner 全体ログとして出す。
- `apps/runner/config/ImageCls.txt` に Mixup/CutMix 有効化キー、コメントアウトした `learn_log_interval`、`target_prob_mix` / EMA metric を追加する。

## テスト
- Public interface / surface: `ImageClsAgentConfig`、`ImageClsLearner::UpdateFromBatch`、`ImageClsUpdateResult::GetScalar`、`ImageClsAgent::CreateLearner`、runner config の scalar key。
- 優先 behavior:
  1. config default/round-trip と範囲外 fail-fast。
  2. `mixup.enabled=false` で既存 loss/accuracy と `target_prob_mix` が hard target 確率になる。
  3. `B < 2` / `prob=0` で bypass する。
  4. seed 固定の Mixup が batch size を維持し、元 label accuracy と mixed target 確率を返す。
  5. seed 固定の CutMix が patch を貼り替え、面積補正 lambda を loss/metric に反映する。
  6. 同 seed で結果が再現し、torch global RNG の変更に影響されない。
  7. Agent 経由で同じ Agent seed から作った Learner が同じ mix 結果を返し、異なる Agent seed では mix stream が変わる。
  8. `learn_log_interval` が disabled/interval どおり動く。
- TDD 順序: `core/anet-core/src/image_cls_agent_test.cpp` へ 1 behavior ずつ RED -> GREEN で追加し、production 側へ test-only API は増やさない。必要な観測は test network module/head と update result scalar から確認する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[image_cls][mixup]"
core\anet-core\bin\Debug\anet-core-test.exe "[image_cls],[config]"
git diff --check
```

## 前提
- 現在の `ImageCls.txt` の active branch/run_name は変更しない。
- `ImageClsView` の mixed image 表示、長時間 A/B 精度評価、RandAugment/Random Erasing は範囲外。
- `CONTEXT.md` と ADR は更新しない。
