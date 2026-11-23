# PRD063 可塑性 weight norm 実装計画

## 概要

- DQN/ImageClsへ、`feature_key`の依存閉包でfeature/readoutを分けた更新前parameter L2 normを追加する。
- 計測は購読駆動とし、未購読時は計算、forward、RNG消費、D2Hを一切発生させない。
- 新config、target/probe/EMA、層別汎用指標、保護機構、Rainbow配線は追加しない。

## NN・公開契約

- `ForwardUpTo`内の閉包探索を、input key優先規則と未知branchの一覧付きfail-fastを維持した`NetworkBody::ComputeDependencyClosure()`へ抽出する。
- 公開型`NetworkParameterNormSplit { feature, readout }`と、constな`Network::ComputeParameterNormSplit(feature_key)`を追加する。
- feature群は閉包内branch、readout群は閉包外branchとheadの再帰parameterとする。対象は`requires_grad()==true`のみ。
- 各群をFP32で二乗和して平方根を取り、0次元Tensorを元device上で返す。空群は0、完全にTensorを持たないNetworkだけCPUを使用する。
- `NoGrad`で計算し、parameter dtype・値を変更せず、forwardとRNGを使用しない。v1ではparameter集合をcacheしない。

## Agent・metrics配線

- DQNの`PlasticityState`へweight norm専用の有効フラグ、最小interval、pending値を追加する。activation actual/target/probeの状態とcadenceは独立させる。
- increment前の`vars_.learn_step`でgateし、`UpdateFromSamples()`直前にonline networkを測定する。更新後に生成される`BatchUpdateResult`へ、更新前に確定した2値を長さ2のFP32 device Tensorとして焼き込む。
- ImageClsもactivation captureとは別の購読状態を持ち、`StepCounts.learn_step`でgateする。既存排他ロック内、optimizer更新前かつ学習forwardと同じ重みの時点で測定する。
- 両UpdateResultは2キーを常に認識する。非測定stepは`NaN`、未知keyだけ`nullopt`とする。初回`GetScalar()`で2値を一括D2Hし、以後cacheを再利用する。
- 測定処理へ安定した`plasticity_weight_norm` profile rangeを追加する。
- baselineとImageClsへ`34_agent_plasticity/61_weight_norm_feature`と`34_agent_plasticity/62_weight_norm_readout`を追加し、いずれも`@learn $learn_step $update_result interval:500`とする。
- `feature_key`はweight normまたは既存plasticity指標の購読時だけ必須・存在検証し、購読ゼロならNoCareとする。

## TDD・検証

- Public surfaceは`Network::ComputeParameterNormSplit()`、DQN/ImageClsの`ConfigureScalarMetricSubscriptions()`と`BatchUpdateResult::GetScalar()`、Runnerのmetrics定義解決とする。
- Network＋DQN、ImageCls、統合検証の縦切りで、各behaviorについて1テストずつRED→GREEN→整理を行う。
- NNではIQN風DAGの帰属、ForwardUpToとの閉包一致、input key優先、未知branch、手計算L2、weight/bias/norm affine、frozen parameter除外、空群、BF16→FP32、dtype非破壊、forward/RNG非接触を検証する。
- DQNでは2キーだけの購読でcapture/probeが不活性なこと、interval最小値、非測定stepの`NaN`、更新前値、lazy一括D2H、購読ゼロを検証する。
- ImageClsでも独立cadence、更新前値、`NaN`、lazy D2H、activation capture非発動を同じ契約で検証する。
- metrics定義から`$update_result`、`$learn_step`、interval、source keyが購読情報へ保持されることを確認する。
- VsDevCmd経由のMSVC Debugでcore testsとRunnerをビルドし、関連tagと全`anet-core-test`を実行する。
- DQNとImageClsのsmokeで`inspect_run.py tags`が06/07を`status=ok`、`count>0`、target=`update_result`として報告することを確認する。
- deterministicな短いDQN Runを同seed・weight norm ON/OFFで実行し、共通learning scalar系列と`agent_close.anet`を完全一致で比較する。
- interval 100のON/OFF Runでwarmup後throughputを比較し、明らかな持続低下がないことを報告する。差が見えても既定値は自動変更しない。

## 文書同期・複雑性監査

- 063 PRDへ、名前付き返却型、`requires_grad`限定、v1 no-cacheの合意を反映する。
- `CONTEXT.md`へ「weight norm分割（feature/readout）」を追加し、activationの`feature_norm`と区別する。
- NN設計、DQN設計、分析ガイドを、依存閉包split、購読契約、06/07とq_max・activation normの三点読みへ同期する。
- keep: 閉包共通化、norm split、購読・疎結果配線、metrics/docs/test。
- defer: parameter集合cache、共有parameter/model parallel専用処理。
- cut: 新config、target/probe/EMA、層別汎用基盤、保護機構、Rainbow配線、新ADR。
- 既存の未コミット変更を保持し、Git staging・commit・pushは行わない。
