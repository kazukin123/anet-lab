# DefaultDQNAgent IQN 実装メモ

## 概要

`001_iqn_10prd.md` を正本として、DefaultDQNAgent に IQN（Implicit Quantile Networks）を追加する。
DefaultDQNAgent の分布表現設定は `quantile_mode = none | qr | iqn` へクリーンブレークし、RainbowAgent の QR 契約は維持する。

## 主な変更

- DefaultDQNAgent の `use_qr` / `num_quantiles` を `quantile_mode` / `qr.num_quantiles` へ置換し、policy と learner の tau rule、検証、既定値を追加する。target policy の fixed 既定は optimistic policy コピー後に復元する。
- `NetworkBranchConfig` を `bind_terms` / `bind_concat_dim` へ変更し、`*` による feature-last 積、concat 次元検証、依存解決、GraphViz、ToJson を同時に移行する。未使用入力 WARN は bind factor と direct output mapping の双方に現れない input だけを対象にする。
- `CosineEmbedding`、`IQNHead`、`IQNDuelingHead`、`TauGenerator` を追加し、ActionPolicy / IQNLearner が shallow-copy した NN 入力へ `taus` を注入する。
- IQN policy は EpsilonGreedy/Greedy で `[0,1]`、UQE で実効下限、非 spatial Thompson で `[0,1]`、spatial Thompson で per-env 下限を使う。IQN+UQEの任意full queryはrisk/full tausを同じforwardへ連結し、risk scoreをQ hintとmetric、full分布をQValuePanelへ分離して公開する。
- IQNLearner は target-policy/current/target の3系統の taus を分離し、current N を sum、target M を mean、Huber を κ で除算する専用 loss を使う。QR loss は変更しない。N=1 の `q_std` は0とする。
- `NetworkModel` は `bool distributional` を受け取り、Rainbow は縮退解決後の値を渡す。
- LunarLanderとDropMergeはevalだけfull queryを有効化する。DropMergeは2048次元backboneとV/A各1024次元streamを維持したIQN構成を現用化する。
- PRD、CONTEXT、ADR 0018、新規 ADR 0019、NN/DQN 設計書、現用 runner config を新契約へ同期する。過去 memo、Rainbow 設定、`_main.txt` は変更しない。

## テスト

- Public interface / surface: `ConfigData`、`DefaultDQNAgent`、`Actor::MakeAction`、`NetworkConfig` / `NetworkBuilder` / `Network::Forward`、各 Head の TensorDict function、learner の public update 経路、runner config artifact。
- 優先 behavior: fixed×3 の公開 IQN tracer、generic bind、TauGenerator、CosineEmbedding、IQN Head、config/policy、IQN loss、IQNLearner、QR/none/Rainbow/ImageCls 回帰の順に確認する。
- TDD 順序: 1つの observable behavior ごとに RED → 最小 GREEN → 必要な refactor を行い、全テストを先に書かない。
- 重点ケース: observation 非汚染、direct output 非WARN、bind 不正文法と batch/concat エラー、fixed の midpoint grid と RNG 非消費、N≠M・κ≠1 の loss 既知値、N=1 有限性、optimistic target の tau rule、Rainbow の縮退後 distributional 判定。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test AnetRLRunner --parallel 1'
core\anet-core\bin\Debug\anet-core-test.exe --anet-test-failure-dialog=off "[nn][bind]"
core\anet-core\bin\Debug\anet-core-test.exe --anet-test-failure-dialog=off "[iqn]"
ctest --preset x64-Debug --output-on-failure
git diff --check
```

LunarLander は `_main.txt` を編集せず CLI override で QR / IQN を各1回実行し、`config/config_data.txt`、metrics、stdout/stderr から設定反映、learner update、eval、finite 値を確認する。

## 前提

- `quantile_mode` の既定は `qr`、QR は51点、policy tau は train=random×32、eval/target=fixed×32、learner current/target は random×64。
- IQN の `uqe_tau_*` は UQE では常時、ThompsonSampling では spatial 時だけ検証する。
- Rainbow IQN、risk distortion、probe 呼び出し側の taus 自動補完、CNN rank 組合せ保証、既存 ownership の広域整理は対象外。
- staging、commit、push は行わない。
