# PRD084：NN Forward公開面と準備経路のまとめ整理

## 概要

HEAD `d3fffc25acb06276322764e093a813803d616931` を基準として、引き継ぎ `handoff_nn_bundle_2026-10-11.md` の A1〜A2、B、C1〜C4、D1〜D2 を実装する。実行器生成を統合し、公開ヘッダから内部署名を外し、target用途の箱をNetwork別に分離する。Learnerの用途ペアとDQN射影Headの定型処理を集約する。

未決事項監査のユーザーブロッカーは0。既定APIが別経路になっていること、BindNetworkがmodule実体別のεと共通RNGを保持することは現行コードで確認済み。引き継ぎの採否を固定し、今回の計画を正本とする。

## 主な変更

- A1: CreateForwardExecutorへ既定設定を統合する。既定時はStateと部品準備なし、clock_kind/seedは使用しない。CreateDefaultForwardExecutorを削除し、実行時の既定分岐も削除する。利用側のIsDefault分岐をなくす。テストの簡易生成はファイル内helperに限定する。
- A2: RunForwardをnn_impl.cppのstatic自由関数へ移し、Body・Head本体と準備済み呼び先を共用する。MakeContextはImpl内部へ移し、Executorのprivate constructorはunique_ptr<Impl>だけを受ける。nn.hppからNetworkModuleExecutor／NetworkHeadExecutor／NetworkForwardState／NetworkForwardContextの前方宣言も削除する。
- B: BindNetworkを削除する。target用途はtarget側を既存seed `network_forward.target`、online側を新seed `network_forward.target_online` で独立生成する。診断も各Networkの既定実行器を独立生成する。NetworkModelの明示的な2 Network構築では同一Networkを拒否する。Double DQNとMunchausen onlineのsampleは乱数の割り当てが変わるが分布を維持する。
- C1: Learnerはcurrent_forward_、TargetForwardsのtarget_forwards_／diagnostic_forwards_を所有する。TargetEvaluationは選択した用途ペアとtarget captureを参照する。NetworkModelのcapture getterを追加し、4か所のtarget Forward分岐を同じ呼び出しへ統一する。ForwardTargetは削除し、ForwardOnlineWithTrainを維持する。
- C2: ActionPolicyのSelectActionとForwardForAction／ForwardForActionWithTausは実行器を参照で受ける。所有側だけがshared_ptrを保持する。
- C3: 未使用NetworkModel::ForwardOnlineを削除し、残る訓練モード復元契約でテストを維持する。
- C4: agent.hppのForwardSelection付近に `nn_forward.[key]` を作る共通helperを置く。AgentのprefixはMakeDefaultConfigKeyで追加する。DefaultDQNAgentConfigはMakeForwardConfigKeyで実行設定の完全なキーだけを公開し、基底の汎用MakeDefaultConfigKeyはprotectedのままにする。設定の解釈やResource保持は増やさない。カタログ・probe・Learner対応検査を揃える。
- D1: DQN内のProjectionHead<Derived,N>へ射影array、CreateExecutor、ValidateForwardReuse、GetNoisyParameterEntries、Forwardを集約する。6 Headは固有の計算と登録を保持し、value→adv順、parameter名、初期化・登録順を変えない。
- D2: LinearProjectionModuleの準備時設定解釈ではmode／hold／share／countを各1回読み、既存のエラー文言を維持する。
- PRD084、設計130／200日英、用語集、所有権ガイド、READMEを同期する。英訳マーカーは今回の日本語working blobへ更新する。ADR0048／0051は冒頭の後発注記だけを追記し、本文と過去計画・検証節を保持する。

## テスト

- Public surface: Network::CreateForwardExecutor、Executor::Forward、ActionPolicy、NetworkModel、Learner、Headの実行・準備契約。
- 既存Debug全体ビルドと関連テストGREENを確認する。最初は既定設定の統合APIで部品準備0回・通常Forwardとの出力／勾配一致をRED→GREENにする。
- 次はDouble DQN target用途のStateが2箱になる契約をRED→GREENにする。TD／QR／IQN、currentとの独立、同一時計の再利用、独立抽選、同一Networkの拒否を確認する。
- C・Dは1挙動ずつ進め、GREEN後に整理する。既定／名前付き、設定元寿命、Network寿命、別Actor独立、保持境界、保存読込、行対応、Dueling、Munchausen全方式、出力・勾配・AMP・診断非干渉、独自Module／Headと共通Prepared部品を維持する。

## 検証

基準・ログ・変更後出力は `.scratch/prd084/forward-refactor/bundle-20261011/` へ分離する。着手前HEAD・index・差分・実行体SHA、DLL、設定、seed、旧OFF checkpointをbeforeへ保存する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 1'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe
```

CPUではプロセス限定 `CUDA_VISIBLE_DEVICES=-1`、CUDAではこの環境変数を除いて上記を実行する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --parallel 1'
```

- appsを含む全体をビルドする。実行体が使用中なら停止せず同じ最終object・リンク条件で別出力先へリンクする。
- 設定6構成の全キー・文字列値を比較する。既定葉監査、HEADとのBOM一致、LF、対象diff checkを実施する。無関係なbat編集による件数差は原因を記載する。
- 機械検査ログ: core/appsのCreateDefaultForwardExecutor／BindNetwork／ForwardTarget(／ForwardOnline(が0件、nn.hppの内部4型・RunForward・MakeContextが0件、target分岐0件、DQN Headの共通3メソッドが各1定義、公開型の概要欠落・宣言後///が0件。NN指定6ファイルのActor／Learner／Probe／lane一致0件を維持する。
- 最終Releaseで `run_forward_refactor.ps1 -Stage off` の既定3構成を各500k経験実行し、旧goldのchecksum・点数・解決値を完全一致で検証する。旧OFF checkpointも読み込む。
- 代表ONは `-Stage on-after` で単発500k経験、Noisy指標込み109,941点・checksum `5c3acbb6eba510e89210025a81249b42cde1019362f42257aa70947be31b7bd6` と完全一致、全診断・複数eval／probe・正常終了を確認する。CPU OFFは88,184点・checksum `b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5` を維持する。
- 同じGPUに別の学習Runがある場合GPU実行を保留し、既存プロセスは操作しない。CPU OFFと旧checkpoint確認は実行する。性能再測定は行わない。Double DQN＋Noisyの短いCPU smokeは任意とする。
- 実行体SHA、結果、変更ファイル、保留理由、online target sampleの乱数割り当て変更をforward_refactor_verification.md「まとめ整理 2026-10-11」へ追記する。

## 前提

- 公開API統合・内部署名の隔離・独立箱・用途ペア・参照渡し・共通Head基底は採用する。通常部品の直接呼び出しと実行interfaceは維持する。
- Body／Branch／Structの専用実行器は、これらの単位の設定準備が必要になるまで延期する。現状ではPreparedを3段追加し、行数を減らさず公開面を隠す以上の価値がない。
- CallInfoのstruct化、同じキーだけによる再利用判定、Networkの公開constructor閉鎖は不採用。CallInfoの不変性とWithRowMapping、再利用規則を維持する。
- 数値、parameter登録・初期化順、RNG消費順（Bのtarget online sampleを除く）、保持、AMP、保存、設定値、メトリクス、episode_start／episodeを維持する。新用語・ADR・互換APIは追加しない。
- 無関係な変更、現在のindex・ブランチ、過去artifactを保持し、add・commit・pushは実施しない。
