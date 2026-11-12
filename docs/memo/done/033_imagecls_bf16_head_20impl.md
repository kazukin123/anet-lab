# ImageCls BF16 Head 化 実装メモ

## 概要
`docs/memo/033_imagecls_bf16_head_10prd.md` を正本として、ImageCls の分類最終層を body 末尾の `LinearOut` から汎用 `anet::nn::LinearHead` へ移す。これにより `Network::Forward` の既存 head 境界で Head=FP32 を構造的に保証し、その上で ImageCls learner / actor の forward のみを BF16 autocast 対象にする。

既定値は `ImageClsAgent.bf16.enabled=false`, `ImageClsAgent.bf16.learner=true`, `ImageClsAgent.bf16.actor=false` とし、既存 run は明示的に有効化されるまで FP32 のまま動く。

## 主な変更
- `anet::nn` の `nn_heads` に汎用 `LinearHead` / `LinearHeadFactory` を追加する。入力キーは `features`、出力キーは constructor 引数で指定し、ImageCls では `"logits"` を使う。
- `ImageClsAgent` は `env_spec.action_spec.GetNumActions()` を head の `out_features` に使い、`WeightInitConfig.mode="he"` の `LinearHeadFactory` を `BuildNetwork` に渡す。
- `ImageClsAgentConfig` に `bf16.enabled`, `bf16.learner`, `bf16.actor` を追加し、config round-trip と既定値をテストする。
- `ImageClsLearner::UpdateFromBatch` と `ImageClsActor::MakeAction` は、`network_->Forward(...)` だけを `Autocast(device_.type(), enabled, torch::kBFloat16)` で囲む。loss/backward/clip/step と softmax/argmax は autocast scope 外に置く。
- `Network::Forward` と head 経由の `GetTensorDictFunction` は、head 実行前に `features.device().type()` の autocast を無効化し、`features.To(torch::kFloat32)` を head へ渡す。CPU eval で `bf16.actor=true` を使っても Head=FP32 を保つ。
- 実測診断で `bf16.enabled=true` の学習停滞が確認されたため、`BatchNorm2d` は Conv autocast 後の BF16 activation を FP32 に上げ、統計更新と正規化を autocast 無効化 scope で実行する。これにより BatchNorm 多用構成でも Head と同様に FP32 安定性を保つ。
- `apps/runner/config/ImageCls.txt` は全 branch 末尾の `LinearOut` を削除し、`net.body.output.[features] = main_feature` に変更する。`net.block.[LinearOut]` は削除し、BF16 3 設定を追加する。

## テスト
- Public interface / surface: `anet::nn::LinearHeadFactory`, `ImageClsAgentConfig`, `ImageClsAgent` の network 構築・actor/learner forward・checkpoint round-trip、`ImageCls.txt` の runner-facing config。
- 優先 behavior: 汎用 head が任意出力キーの logits を返すこと、ImageCls が env class count 由来の head を持つこと、CPU/CUDA の head 実行が FP32 を保つこと、BF16 設定が forward scope だけに効くこと、Conv2d autocast 後の BatchNorm2d が CPU/CUDA とも FP32 出力を保つこと。
- TDD 順序:
  1. 汎用 `LinearHead` の RED を追加し、`features[B,in] -> custom_key[B,out]`、GraphViz info、TensorDictFunction を確認してから実装する。
  2. ImageCls の小型 network helper を `Flatten` + head 化する RED を追加し、head factory 配線と `bf16` config を実装する。
  3. CPU autocast 有効時も head output が FP32 になる RED を追加し、`Network::Forward` / head function の autocast disable device を修正する。
  4. learner / actor の BF16 設定が forward scope だけを通す RED を CPU で観測可能な dtype / output behavior として追加し、ImageCls forward guard を実装する。
  5. 実測 run での BF16 学習停滞を受け、`Conv2d > BatchNorm2d` の BF16 autocast 後も BatchNorm 出力が FP32 になる RED を CPU/CUDA で追加し、`BatchNorm2dModule` 側で autocast を無効化する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe "[image_cls]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提
- 旧 body-only ImageCls checkpoint の後方互換は非対象。
- DQN 側 `anet::rl::dqn::LinearHeadFactory` の汎用版への寄せ替えは非対象。
- 複数 seed の実測 perf / accuracy 評価はユーザー側の後続評価とし、本実装では build/test と runner-facing config の整合までを確認する。
- `CONTEXT.md` は用語衝突なしのため更新しない。ADR も作成しない。
