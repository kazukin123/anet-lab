# PRD084：NN Forwardの公開面と準備経路の整理

## 概要

引き継ぎ案A1〜A4・B1〜B4・C1を採用する。B5は採用せず、NetworkForwardCallInfoの不変クラスとWithRowMappingを維持する。C2の再利用規則変更は見送る。数値・RNG・設定値は維持し、設定内容の検証時期だけをC1に従って変更する。

## 主な変更

- StateとContextの定義をnn_impl.hppへ移し、公開側は前方宣言だけにする。StateAccessをStateへ統合し、内部型管理は維持する。テストのState取得仕掛けとnn_test.hppを削除する。
- ContextはStateポインタと不変CallInfoの参照を持つ。clock_kindを除き、clock・episode_start・行対応はCallInfoへ委譲する。参照のlifetimeは同期Forward内に限定し、Tensorをコピーしない。
- 時計の列挙はkTransientRowsとkContinuingRowsの2値にする。保持方式との整合は準備時、shape/device/時計値は演算時に検証する。
- ExecutionのGetNetwork/ForwardUpToを削除する。Networkの既定ForwardUpToは維持する。
- PrepareDefaultForwardを追加し、State/時計を持たずModule/Headを準備しない既定実行を生成する。PrepareForwardは名前付き専用とし、既定設定を拒否する。
- BindNetwork(network)は設定・State・時計種別を引き継ぐ。既定実行のBindは接続先の既定実行を返す。Learnerの既定Resource/無効時/既存診断を移行し、Resource対応検査とNetwork寿命を維持する。
- probeのμ-only側を同じNoGrad/eval/FP32保護内のNetwork::Forwardへ移す。mean設定・実行・引数・時計加算を削除し、sample設定・専用RNG・抽選順・購読gatingを維持する。
- カタログ内容のfactory宣言検証を削除する。参照存在は読込時、使用する内容は具象部品の準備時に検証する。未参照項目と部品が使わない設定群は検証しない。構造設定・σ decay・probe設定・ValidateForwardReuseは維持する。
- PRD084、設計130/200日英、用語集、所有権ガイド、READMEを同期する。日付付き改訂と複雑度裁定をPRDへ追加する。ADR0048/0051と過去の記録は保持する。HEAD0dbb5cc6のBOM/改訂順修正は再変更しない。

## テスト

既存GREENの後、新契約を1挙動ずつRED→GREENで検証する。

- 既定ExecutionでModule/Head準備0回、通常Forwardとの出力/勾配一致、名前付き準備への既定設定の拒否。
- Bindの設定元寿命からの独立、設定/State/時計の共有、Actor独立State、Network寿命。
- 不変CallInfoのコピー/WithRowMappingと内部Contextの読取り専用参照/既定値。
- 未参照の不正設定を許容し、利用部品の準備で拒否。未知参照・不足・保持制約・再利用検証を保持。
- 準備済み部分Forwardテストを既定部分/準備済み全Forwardへ目的に応じて移行。
- 出力/勾配/保持/同時計/同時利用/target/保存/AMP/診断非干渉と独自Module/Head・NN共通Preparedの既存テスト。

## 検証

編集前のHEAD・差分・Debug GREEN・Release実行体を記録し、実行体/DLL/設定/seed/checkpointを.scratch/prd084/forward-refactor/simplify-20261010/beforeへ保存する。変更後は出力を分離する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe "[noisynet]"
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe
```

CPU検証ではプロセス内だけCUDA_VISIBLE_DEVICES=-1を指定し、CUDA検証では復元する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --parallel 1'
```

- 設定6構成の全解決値/既定葉、BOM/LF/diff check、削除API/宣言検証/mean probeの検索を実施。kNoneはNetworkForwardClockのものだけ対象。NN6ファイルのRL語0件を維持。
- Run資材に-Casesを追加し既定3構成を維持。CPU OFFはGPU検査/計測に依存させない。代表ONでは同じGPUの別学習Runがあれば保留し、既存プロセスを操作しない。
- 最終Release CPU LunarLander OFF500k: 88184点、b76541e3b8319f9b72e2987f4a6f4dc2afeb26104fc57dabbaee18c67cb5ebd5一致。旧OFF checkpoint読込も確認。
- 最終Release代表ON500k単発: Noisy込み109941点、5c3acbb6eba510e89210025a81249b42cde1019362f42257aa70947be31b7bd6一致。構成/診断/複数eval・probe/終了を確認。性能再測定なし。
- SHA/結果/変更ファイル/保留理由をcore/anet-core/testdata/prd084/forward_refactor_verification.mdの新節へ追記する。

## 前提

C1採用・B5不採用の今回の回答を引き継ぎより優先する。parameter登録・初期化順・RNG消費・保持・AMP・保存・設定ファイル/解決値・episode_start/episodeを維持。Execution interface・NN共通Prepared・行対応型・再利用規則は維持。新語/ADR/互換APIを追加しない。ブランチ・無関係な変更・過去artifactを保持し、add/commit/pushしない。
