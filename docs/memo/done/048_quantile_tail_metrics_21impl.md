# PRD 048 crossing深度p90追補 実装メモ

## 概要

更新されたPRD 048に従い、既存5metricへ6本目の観測専用scalar `policy_selected_crossing_depth_p90_ratio` を追加する。既存 `20impl` とその実装を維持し、本メモを追加実装の正本とする。Policy score、loss、priority、action、sampling、RNG、forward回数、公開設定、checkpoint／Replay schemaは変更しない。

## 主な変更

- `DQNActionInfo` の既存quantile tail診断payloadへ、QRの `q_quantiles` またはIQNのfixed `full_q_quantiles` に対するdetached aliasを保持する。quantile値は複製せず、値順の再sortもしない。
- Policy側5metricの初回 `GetScalar()` 参照時に最終実行actionのquantile列をgatherし、正の隣接crossing深度をaction内rangeで正規化する。laneごとにnearest-rank p90（`ceil(0.9n)`）を求め、そのbatch平均を既存4scalarと同じpacked CPU cacheへ同梱する。
- crossingなし、またはrangeが0のlaneは正常値 `0` とする。`K < 2`、IQN full query無効／非fixed、必要Tensor欠損は他のPolicy quantile診断と同じく `NaN` とする。
- `WithAction()` は既存payloadを共有し、CPU cacheだけを破棄する現在の契約を維持する。追加metricも差し替え後actionへ追従する。
- DropMergeの `eval2` に `52_eval2/60_policy_selected_crossing_depth_p90_ratio_ema` をEMA `0.01`、`interval:100`で1本だけ追加する。Train、`eval1`、LunarLander、raw版には登録しない。
- DQN設計、可観測性、Run分析ガイドへ、event内lane別p90のbatch平均、h-space、`NaN`／`0`条件、crossing頻度との解釈差を追記する。`CONTEXT.md`と既存ADRは変更しない。

## テスト

- Public interface / surface: `DQNActionInfo::GetScalar("policy_selected_crossing_depth_p90_ratio")`、`WithAction()`、DropMergeの解決済みscalar設定。
- 優先 behavior: 選択列 `[0, 2, 1, 4, 2]` が正規化深度 `[0.25, 0.5]` とnearest-rank p90 `0.5`を返す。次にcrossingなし／range 0が `0`、`K < 2`とIQN入力欠損が `NaN`、複数laneのbatch平均、`WithAction()`追従、取得順序とcache共有を確認する。
- TDD 順序: QR Policyの公開 `GetScalar()` 1テストをRED→GREENにし、以後1 behaviorずつ追加する。GREEN後にだけ既存cache処理を最小限リファクタし、IQN fixed full queryと欠損ケースを通す。
- 非干渉: 既存4 Policy metric、action、forward回数、loss、priority、Replay適用順、RNG列が不変であることを既存回帰テストとコード構造で確認する。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test -- -j 1'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][qr][action_policy][metrics]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][iqn][action_policy]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提

- 既存 `20impl` と未コミット変更は保持し、追加metricに必要な箇所だけを局所編集する。
- full suiteに残るReplayBufferの既知NG 5件は受入済みbaselineとして扱い、件数または失敗内容が増えないことを確認する。
- paired throughput smokeと低下2%以内の評価は、既存合意どおり今回の受入から除外する。追加forward・追加wait boundaryがないことはテストとコード構造で確認する。
