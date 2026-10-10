# PRD084：NN Forward の事前準備へのリファクタリング

## 概要

設定の解釈・キャッシュ検索を通常の Forward 経路から外す。具象 Module が作った準備済み処理を、用途ごとの Forward 実行オブジェクトが保持する。実行オブジェクトは専用 State と Context の生成も管理する。

コード編集前に本計画を保存する。無関係な変更、過去の Run artifact、現在のブランチは保持する。add・commit・push は実施しない。

## 主な変更


- Actor／Learner の Config DTO は設定値と `nn_forward_key` だけを保持する。DefaultDQNAgentConfig のカタログは `ConfigData` を保持し、解決済みの NN オブジェクトは Agent の Resource として分離する。
- `NetworkForwardConfig` はキーと選択済み設定データを保持する不変な搬送型にする。名前付き項目のコンストラクタはキーと項目内の `ConfigData` を受け取る形へ移行する。具象設定型・解釈・検証は各 Module 内に閉じる。
- `Network::PrepareForward(config, clock_kind, seed)` は、特定 Network に結び付いた `NetworkForwardExecution` を生成する。設定の解釈と利用時制約の検証はここで完了する。
- `NetworkForwardExecution::BindNetwork(network, config)` は、同じ用途の State と時計種別を共有する別の実行オブジェクトを生成する。初期化時だけ使用し、実行中の設定切替機能にはしない。
- `NetworkForwardCallInfo` は時計の値、`episode_start`、行対応を持つ不変な値オブジェクトとする。設定・State・時計種別は持たない。
- `Forward` と `ForwardUpTo` は `info` を受け取る。内部の private な `MakeContext(info)` が State・時計種別と組み合わせて Context を生成し、その後に準備済み処理を実行する。Context 生成では抽選・State 更新・Tensor 計算を行わない。

呼び出し側は、情報の生成と実行を別の文にする。

```cpp
const NetworkForwardCallInfo info({
    .clock = action_selection_id,
    .episode_start = state.episode_start,
});

auto output = actor_forward->Forward(input, info);
```

時計を進める判断は Actor／Learner に残す。同じ行動選択・更新内の複数 Forward には同じ時計を渡す。

### 実装の進め方

1. **Config DTO の分離**
   - カタログの解決済みオブジェクトと参照を Config DTO から移す。Actor／Learner へは初期化時の Resource として渡す。
   - 宣言、参照欠落・未知キー、休眠、必要項目の検証時期を維持する。
   - 設定ファイル、解決済み全キー・文字列値、保存用設定の内容を変更しない。この段階で既存テストを通す。

2. **準備済み Forward 経路への移行**
   - Module 用と Head 用の共通実行 interface を設ける。通常の Module／Head 本体もその interface を満たし、初期化時に呼び先を固定する。
   - 実行設定を使う具象 Module だけが、私有の解釈済み設定を持つ別オブジェクトを生成する。通常の層にはラッパーや追加の仮想関数呼び出しを挟まない。
   - Body・branch・block・Head と、内部に子 Module を持つ部品まで準備を伝達する。既存の bind、依存閉包、trace、capture、FP32 保護を共用し、計算本体を複製しない。
   - LinearProjectionModule の設定キャッシュ、内容比較、設定用 mutex、Forward 中の設定取得を削除する。Noisy の解釈済み値は私有の enum・数値として保持する。
   - Actor は用途専用の実行オブジェクトを持つ。Learner の target 用途では、online／target に結び付いた実行オブジェクトが内部で同じ State を共有する。probe の比較側も現行の共有関係を維持する。
   - 実行オブジェクトは対象 Network の寿命を保証する。重みは既存 Module のものを参照し、clone 先には新しく準備する。同期・保存には実行オブジェクトや State を含めない。
   - 既定動作の `Network::Forward(input)` と既存の TensorDictFunction を維持する。名前付き設定を Context で渡す旧経路・不要になった API は削除し、現用コード・テストを移行する。

設定取得に RTTI・dynamic_cast・型トークンを使わない。既存 State 内部の型管理は今回変更しない。μ・σ、初期化順、RNG、保持境界、AMP、optimizer、診断、保存契約は維持する。


### レビュー修正と NN 共通化（2026-10-10）

- Learner 系5コンストラクタの forward_resources を必須にし、target_policy の直後へ移す。有効時は名前付き参照に対する非既定Resourceとキー末尾 [nn_forward_key] の一致を準備前に検証する。空キーは既定Resourceのみ。無効時の休眠は維持する。
- PreparedModuleChildren<N>をanet::nnのnn_impl.hppへ移し、ResBlock/CNBlockとHeadから共用する。nn_heads.hppのPreparedHeadExecution<Head, ForwardMethod, N>は本体参照と準備結果を所有し、コンパイル時メンバ関数とindex展開で計算本体を呼ぶ。全子が既定なら生成関数はnullptrを返す。
- DQNの6入れ子Preparedを削除する。1射影またはvalue→advの2射影という順序を維持する。共通部品はDQN/Noisy/具体設定型を解釈せず、準備を具象Moduleへ委譲する。追加の仮想呼び出し、std::function、RTTI、型トークンは導入しない。
- ADR0048冒頭に0051の後発決定を注記する。PRDに日付付き改訂、§4.2の準備済み/既定経路、§13の裁定参照を加える。Rainbowの古いコメント、ForwardSelectionの字下げを直す。準備時のbranch走査はnullスキップ、PrepareForward宣言はshared_ptr所有を明記する。
- 用語をNN実行オブジェクト/NN呼び出し情報に統一し、設計130/200・README・検証記録を同期する。C++識別子は維持する。

## テスト


- 編集前に Debug 全体ビルドと既存テストの GREEN を確認する。Release 基準実行体・DLL・設定・seed・checkpoint・差分を `.scratch/prd084/forward-refactor/before/` に保存し、変更後の出力は分離する。
- 新しい契約は1挙動ずつ RED→GREEN で確認する。設定の解釈が準備時だけであること、設定元の寿命・変更から独立すること、独自設定を読む Module／Head の追加が共通型や既存部品の変更を要求しないことを検証する。
- 複数 Actor の独立 State と同時 Forward、target 用途の State 共有、同じ時計の再 Forward、snapshot、clone、保存読込、行対応、TD／QR／IQN・Dueling・Double-DQN・Munchausen 全方式を確認する。
- 既存の出力・勾配・精度・診断非干渉テストを移行する。MSVC 初期化経由で Debug 全体ビルド、関連テスト、全 core テスト、設定既定葉監査と全解決値比較を実施する。
- **最終 Release の OFF 回帰**：Atari 既定、SN 付き効率優先チェーン、CPU LunarLander を各500k経験で実行する。従来の除外条件による checksum・点数・既存設定値の完全一致を必須とする。
- **代表 ON 比較**：Breakout・seed=1・128 env、参照チェーンの AMP 継承・Body FP32 を使用する。Actor は episode/sample、Learner current／target は call/sample、Munchausen は target 方式、全診断購読、greedy_dist 無効とする。変更前／後を3反復の A/B 交互、各500k経験で実行する。
- ON 比較では Noisy 指標も含め、時刻・性能・資源計測を除く数値系列の一致を確認する。warmup200k以降の throughput、時間、メモリの各 Run 値と中央値を保存する。同じ GPU に別の学習 Run がある場合、測定は保留する。
- 合否は挙動一致と通常 Forward からの設定解釈・検索・設定用ロックの除去で判定する。速度向上は必須にせず、再現する悪化は調査・修正する。性能48 Run の全面再測定は行わない。


### レビュー修正のテスト

- 既存GREENを確認し、Learnerの名前付き参照への既定Resource、current/target取り違え、個別不一致を1挙動ずつRED→GREENにする。正しい対応、空キー+明示既定、無効時の休眠も確認する。
- 非DQNのHeadが共通部品を使い、独自設定の子と通常の子で準備時だけの解釈、片側fallback、全子既定、出力と勾配を確認する。
- GREEN後に通常3Head、Dueling3Head、ResBlock/CNBlockを移行する。既存の出力/勾配/保持/μ-only/保存/IQN rank3/同一時計/診断非干渉とTD/QR/IQN/Rainbow/Munchausenの契約を維持する。

## 検証

MSVC初期化経由でDebug/Release全体をビルドし、NoisyNet、関連テスト、全coreを実行する。

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --parallel 2'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe '[noisynet]'
```

```bash
core/anet-core/bin/Debug/anet-core-test.exe
```

```bash
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Release --parallel 2'
```

```bash
.\.venv\Scripts\python.exe core/anet-core/testdata/prd072/check_default_leaves.py
```

- 設定6構成の全解決値比較、BOM/LF、NN指定6ファイルのRL語0件、git diff --checkを確認する。共通部品のDQN/Noisy非依存、6Prepared削除、計算本体と直接経路の維持、Forwardからの設定解釈/検索/設定用ロック除去を監査する。
- 元のOFF3構成と代表ON3反復の結果は[検証記録](../../core/anet-core/testdata/prd084/forward_refactor_verification.md)に保持する。
- 今回は最終ReleaseでOFF3本+ON1本を各500k経験で実行する。OFFは旧checksumと点数、ONは代表構成のNoisy込み109941点と完全一致を要求する。旧OFF checkpoint読込も確認する。
- Run資材に明示出力先とON単発を追加し、.scratch/prd084/forward-refactor/review-followup-20261010/after/へ分離する。ON条件は既存代表構成と同一。全診断の成立値/NaN、複数eval/probe、正常終了を確認する。同じGPUの別学習Runがあれば測定は保留し、既存プロセスは停止しない。
- 今回の実行体SHA、結果、変更ファイルを検証記録に追記する。以前の性能6Runは当時の実行体による測定として区別する。

## 前提


PRD084、現行設計130／200、用語集、所有権ガイド、検証 README を新契約へ同期する。変更理由は新 ADR に記録し、ADR0048 と過去の検証記録は当時の記録として保持する。PRD に完了状態の行を書かず、BOM・改行を維持する。

| 対象 | 裁定 |
|---|---|
| Config DTO の分離・準備済み処理・用途別実行オブジェクト | 採用 |
| 通常 Module の実行ラッパー | 本体を直接呼ぶ形へ縮小 |
| 設定キャッシュ・型別 Map・型トークン・公開 PrepareCall | 削除／不採用 |
| 実行中の設定切替・準備結果の重複排除 | 実際に必要になるまで延期 |
| State 内部の RTTI、AMP 集約、episode 用語の変更 | 対象外 |
| 実装段階 | Config 分離と実行経路変更の独立した2段階 |

- 今回のユーザー選択はOFF3本+ON1本。性能3反復と過去48Runは再測定しない。共通部品は既存NN内部ヘッダに置き、他Agentに新しい動作を追加しない。
- 設定値、parameter登録、初期化/射影順序、RNG、保持、AMP、保存形式、episode_start/episodeを維持する。
- 元計画・検証記録・差分と実行体は .scratch/prd084/forward-refactor/review-followup-20261010/before/ に保持する。
