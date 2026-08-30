# Run Serialization（Run 全体 Save / Load）暫定 PRD

> 状態: 暫定メモ。Run 全体の Save / Load を再設計するため、現行 checkpoint の事実、Adam/AdamW で判明した問題、将来 contract の要求を記録する。
> 起点: `auto_load_file` で長期 Run をサイクル継続した際、Run 側で変更した `learner.alpha` が optimizer load によって実効値へ反映されていなかった件。
> 本 PRD は設計候補の記録であり、設定名、archive 形式、実装順の確定や実装着手を意味しない。

## Problem Statement

現行の checkpoint は Agent、より正確には `DefaultDQNAgent` 固有の archive であり、Run 全体を継続する仕組みではない。

現行 `DefaultDQNAgent::Save()` は、archive header、Config 文字列、online/target Network、inner Learner を保存する。inner `dqn::Learner::Save()` の payload は optimizer だけであり、`auto_load_file` は新しい Agent の構築中に Network と optimizer を復元する。

一方、少なくとも次の State は同じ checkpoint から復元されない。

- ReplayBuffer の内容、priority、sample RNG、prefetch 状態
- `RuntimeVars`、update credit、warmup latch、PER beta などの学習進行 State
- RewardScaler、ObservationNormalizer の統計
- Run の `StepCounts` と schedule の進行位置
- Env の episode、物理、lane、RNG State
- Actor-private snapshot、同期進行、Actor RNG
- Observer の EMA など、表示・集計に継続性が必要な State

したがって、現行 `auto_load_file` は「旧 Run の完全再開」ではなく、「新しい Run へ Network と optimizer の一部を移植する」操作である。しかし、何を引き継ぎ、何を新規設定または reset するかを利用者が選択できず、保存済み Config、現在の Config、load 後の実効 State の優先順位も明示されていない。

この曖昧さが、今回 Adam/AdamW の learning rate で実害になった。

## 今回確認した Adam/AdamW の事実

現行フローは次のように動作する。

1. 新しい Run の Config から、現在指定された `learner.alpha` を持つ optimizer を構築する。
2. `auto_load_file` から Network を load する。
3. `Learner::Load()` が LibTorch の optimizer serialization を使って AdamW を load する。
4. optimizer の moment、step だけでなく `param_groups` の options も復元され、構築時の options が checkpoint 側の値で置き換わる。

`param_groups` の options には learning rate のほか、`weight_decay`、`eps`、betas などが含まれる。このため、例えば checkpoint が `alpha=1e-4` で作られ、新しい Run の Config が `alpha=5e-5` でも、load 後の実効 learning rate は `1e-4` へ戻る。

今回確認した長期 Run 系列では、途中サイクルで行った `alpha` の変更は実効 optimizer へ反映されていなかったと判断する。該当 Run の差を learning rate 変更の効果として解釈しない。また、この系列で同じ `alpha` A/B を再検証することは本 PRD の目的外とする。

optimizer options を含めて復元する挙動自体は、同一 Run を完全に再開する **exact continuation** では妥当である。一方、moment と step は継続しつつ現在 Config で learning rate を変更する **retune continuation** や、Network weights だけを移植する **weights-only transfer** では不適切である。問題は LibTorch の挙動そのものではなく、ANET が load の意図を区別せず、実効値も公開していないことである。

## Solution Direction

Save / Load の所有者を Agent 固有 API から Run 全体の serialization contract へ引き上げる。

設定では、単一の `auto_load_file` だけでなく、少なくとも次を表現できるようにする。

- **何を Save するか**: 対象 State group
- **何処へ Save するか**: Run artifact、世代 checkpoint、明示 path などの destination
- **何を Load するか**: 対象 State group
- **何処から Load するか**: 同一 manifest、過去 Run、別 checkpoint などの source
- **どの方針で Load するか**: checkpoint 値を継続する、現在 Config で上書きする、新規初期化する、完全一致を要求する

Save 対象、Load source、Load policy は State group ごとに定義可能とする。ただし、関連 State を無制限に別 source から合成すると整合しない Run を作れるため、exact continuation profile では同一 checkpoint generation を要求し、意図的な transfer/retune だけを明示 opt-in で許可する。

多数の個別 boolean を追加するのではなく、Run-level manifest と少数の profile、State group ごとの override で表現する方向を優先する。

## State Group の暫定分類

| State group | 主な内容 | 主な整合対象 |
|---|---|---|
| Run metadata | Run ID、lineage、cycle、保存時刻、schema version、実効 Config fingerprint | 全 group の provenance |
| Step / schedule | `train_step`、`exp_step`、episode count、schedule origin/進行位置 | epsilon、tau、PER beta、将来の LR/Batch/RR schedule |
| Agent model | online/policy Network、target Network、model version | optimizer、Actor snapshot |
| Optimizer | optimizer 種別、moment、step、param group、options | model parameter identity、現在 Config |
| Agent runtime | `RuntimeVars`、update credit、warmup latch、scaler、normalizer | Step、Replay、Network |
| Actor / Eval | private snapshot、sync age、Actor/Eval RNG、必要な進行 State | source model version、Step |
| Replay | Experience、sampleable state、generation、PER priority、sample RNG | Agent/Env schema、N-step contract |
| Env | lane ごとの episode、物理状態、現在 Observation、Env RNG | Runner state、Step、Agent observation schema |
| RNG | root seed、Torch CPU/CUDA、Agent、Actor、Replay、Env の各 stream | deterministic continuation contract |
| Observer / artifact | EMA、集計 window、writer の継続情報 | metric segment と Step 軸 |

非同期 queue や prefetch buffer のような一時的 State は、そのまま dump することを既定にしない。Save 前に処理を quiesce し、所有モジュールの authoritative state へ反映したうえで保存するか、再構築可能であることを contract に明記する。

## Load Policy の暫定分類

名称は未確定だが、State group ごとに次の意味を区別する。

| Policy 候補 | 意味 |
|---|---|
| `continue` | checkpoint の State と options をそのまま復元し、完全継続を優先する |
| `retune` | 学習 State は復元するが、明示された現在 Config の options を load 後に適用する |
| `reset` | checkpoint の対象 State を読まず、現在 Config から新規構築する |
| `require_match` | checkpoint と現在 Config が一致する場合だけ復元し、不一致は load 前に fail-fast する |

`continue`、`retune`、`reset` は単なる実装都合ではなく実験の意味を変えるため、Run artifact に必ず記録する。省略時に曖昧な既定動作へ fallback しない。高水準 profile が既定方針を与える場合も、解決後の group 別 policy を manifest へ展開して残す。

## Adam/AdamW に必要な contract

Optimizer group では、少なくとも次を明示する。

### Exact continuation

- moment、optimizer step、param group、learning rate、`weight_decay`、`eps`、betas を checkpoint から復元する。
- 現在 Config と checkpoint options が異なる場合、checkpoint が黙って勝つのではなく、不一致と最終実効値を明示する。
- exact continuation profile では、原則として不一致を load 前に fail-fast する。明示的に checkpoint options を採用する policy だけが継続可能とする。

### Retune continuation

- moment と optimizer step は checkpoint から復元する。
- learning rate など、retune 対象として明示した options は現在 Config を load 後に適用する。
- checkpoint 値、現在 Config 値、load 後の実効値を log と Run artifact に残す。
- options 間に互換性がない場合は、黙って一部だけ適用せず fail-fast する。

### Weights-only transfer / optimizer reset

- Agent model だけを load し、optimizer は現在 Config から新規構築する。
- moment と optimizer step が残っていないことを明示する。
- target Network を model と一緒に load するか、online から再同期するかも別の State/policy として明示する。

learning rate だけを特別扱いしない。Optimizer options 全体について、どの値が checkpoint、現在 Config、profile override のどれから来たかを追跡する。

## Manifest と実効値の要求

checkpoint または Run serialization bundle は、データ本体とは別に manifest を持つ。manifest には少なくとも次を含める。

- serialization schema version と作成した ANET build/version
- Run ID、source Run、lineage、checkpoint generation
- 保存時の全 Step 軸
- 含まれる State group、保存先、サイズ、整合性情報
- 各 group の format/version と互換性 identifier
- 保存時の Config と、Config だけでは分からない runtime effective values
- load 時に選択した source、policy、Config 差分、最終実効値
- reset、欠落、意図的な非復元を含む continuation fidelity

現行 `config/config_data.txt` は新しい Run が要求した解決済み Config の記録であり、load 後に checkpoint が上書きした runtime value の正本にはならない。将来は requested config と effective loaded state を区別し、少なくとも optimizer の実効 learning rate、options、step を起動直後から確認可能にする。

## Save / Load の安全性

- Load は、schema、必須 group、Network shape、parameter identity、Config compatibility、source generation の整合を **全対象へ事前検証してから** State を変更する。
- 途中まで load した object を公開しない。失敗時は Run 開始前に fail-fast する。
- Save は一時領域または未確定 generation へ書き、全 group と manifest の確定後に完成 checkpoint として公開する。
- background worker の例外を握り潰さず、Save/Load 呼び出し元へ伝播させる。
- 明示した group を保存できない場合、黙って省略しない。`auto` profile だけが WARN 付き fallback を定義できる。
- Run close、手動 snapshot、AutoPause、異常終了 recovery で、どこまで同じ fidelity を保証するかを別途定義する。

## 決定性と等価性検証

Run serialization は、checkpoint を **等価性アッセイの対象として使える**ことを要求に含める。改修前後で挙動が変わっていないことを証明する手段として、checkpoint が使えるかどうかという観点である。

### 現行の観測

2026-08-30、[065](065_nn_spectral_norm_10prd.md) の「OFF 完全不変」検証で次が判明した。

- 同一の保存済み実行体を、同一 seed・同一 config・determinism 既定 ON で 2 回実行しても、`agent_close.anet` の raw SHA-256 は一致しない。
- 全 checkpoint のサイズは同じで、base/new ペアと base/base-repeat ペアの最初の差分位置も同じ model archive 内 offset `0x025A7EE4` だった。
- 同じ Run の metrics マスタ（`loss` / `q_max_mean` / `q_max_max` / `61` / `62` の全 `(tag, step, value)`）は完全一致していた。したがって学習側の非決定ではなく **serialize 側の非決定**である。原因は未特定。

サイズ一致かつ差分開始位置一致は、可変長の内容差ではなく **固定位置フィールドの内容差**を示唆する（保存時刻、build identifier、未初期化 padding のいずれか）。作り直し時に当該 offset 周辺を直接 dump すれば切り分けられる。

この帰結として、062 / 063 / 065 が受入基準へ据えていた「`agent_close.anet` の checksum 一致で OFF 完全不変を証明する」手順は現行コードでは成立しない。065 は受入 1 を metrics checksum のみのゲートへ改訂し、checkpoint はサイズと hash を観測値として記録するに留めた。

### タイムスタンプとの両立

「再実行でバイト一致」を archive 全体へ課すと、保存時刻・Run ID・build version を書けなくなる。これらは manifest の仕事（§Manifest と実効値の要求の Run metadata）なので、決定性の要求は archive 全体ではなく **State group ごとの payload** へ課す。

| 領域 | 決定性の要求 |
|---|---|
| manifest / Run metadata | 要求しない。保存時刻、Run ID、lineage、build version は非決定で当然であり、むしろ持つことが仕事である |
| State group の payload | 決定論 Run（学習系列が bit 一致する条件）では、同一 seed・同一 config・同一 build の再実行で **group ごとの payload digest が一致する** |

manifest は既に「含まれる State group、保存先、サイズ、整合性情報」を持つ要求があるので、その整合性情報を **payload digest** として定義すれば、等価性アッセイは「manifest の group digest を比較する」で閉じる。raw ファイル比較は使わない。同時に、この digest は load 側の破損検出・generation 突合とも同じ値を共有できる。

payload 側で決定性を成立させるには、少なくとも次を contract に含める。

- group payload に保存時刻、絶対 path、ホスト名、非決定な走査順（`unordered_map` 等）を混ぜない。可変情報は manifest 側へ寄せる。
- 未初期化領域を書き出さない。padding は明示的にゼロ埋めする。
- tensor の serialize 順を安定名で固定する。

決定性を要求する範囲は「学習系列が bit 一致する条件下の Run」に限る。非決定論 Run の bit-exact 再現は Out of Scope のままである。

## User Stories

1. 長期 Run 実施者として、Run を停止しても Step と schedule を含めて同じ進行位置から再開したい。
2. 実験者として、Network と Adam moment は引き継ぎつつ learning rate だけを変更したい。
3. 実験者として、Network weights だけを別 Run から移植し、optimizer、Replay、Env は新規初期化したい。
4. 実験者として、Agent、Env、Step、Replay、RNG のうち何を何処へ保存したかを manifest で確認したい。
5. 実験者として、各 State group を何処から load し、何を reset したかを Run artifact から再確認したい。
6. 実験比較者として、Config に書いた値と load 後の実効 optimizer 値が異なる場合に、その差を見落としたくない。
7. 保守者として、互換性のない State group を混ぜた Run を部分 load のまま開始させたくない。
8. 保守者として、新しい Agent や Env を Run serialization へ追加するとき、State group と互換性 contract を局所的に実装したい。
9. 実装者として、改修前後で OFF 構成が完全に不変であることを、checkpoint の group digest 比較で証明したい。

## Testing Decisions

### Adam/AdamW regression

checkpoint の optimizer learning rate を `1e-4`、現在 Config を `5e-5` とした fixture で、次を確認する。

| Load policy | 実効 learning rate | moment / step | 期待結果 |
|---|---:|---|---|
| `continue` | `1e-4` | 復元 | checkpoint 値と採用理由を記録 |
| `retune` | `5e-5` | 復元 | 現在 Config の override と差分を記録 |
| `reset` | `5e-5` | 新規 | optimizer State を復元しない |
| `require_match` | - | - | 不一致を事前検出して fail-fast |

同じテストを `weight_decay`、`eps`、betas、複数 param group に広げ、未知 option、param group 数不一致、parameter identity 不一致を黙って受理しないことを確認する。

### Run-level consistency

- `StepCounts` と各 schedule が保存時の累積位置から継続すること。
- Step だけを復元し、Replay や learner runtime を reset する組合せが profile contract に反する場合は拒否されること。
- Agent model、optimizer、Actor snapshot の model version が一致すること。
- Env exact continuation では、lane、episode、物理、Observation、RNG が次 step と整合すること。
- Env reset policy では、再開ではなく新規 episode になった事実を manifest に記録すること。
- Replay の N-step、PER、generation、sample RNG を保存対象にした場合、load 後の sampleable contract が一致すること。
- 異なる checkpoint generation や source Run の group 混在を、明示 opt-in なしで拒否すること。
- 途中で中断した Save、欠損 group、破損 manifest、未知 schema を完成済み checkpoint として load しないこと。
- load の途中失敗後に、一部だけ復元された Run を開始しないこと。

### 決定性 regression

- 決定論設定で同一 Run を 2 回実行し、State group ごとの payload digest が一致すること。manifest の Run metadata は一致しなくてよい。
- group digest が manifest へ記録され、load 側が source と突合できること。
- 保存時刻や build identifier を group payload へ混入させた変更が、この digest 比較で検出されること。
- この regression が緑になるまで、他 PRD の等価性受入で raw checkpoint checksum をゲートに使わないこと。現行は非決定であり、判定不能である。

## Out of Scope

- 本メモの段階で最終的な設定 key、ファイル配置、archive library、圧縮形式を確定すること。
- 現行 `auto_load_file` へ個別 boolean を足すだけの暫定拡張。
- 今回の長期 Run 系列で learning rate A/B をやり直すこと。
- serialization と同時に Agent、Env、Replay の所有権を全面的に再設計すること。
- 非決定論 Run の bit-exact 再現を、保存対象と検証なしに保証すること。
- metrics の全履歴を一つの旧 Run directoryへ追記し続けること。segment/lineage の見せ方は別途設計する。

## 未決事項

- exact continuation、retune、weights-only などの高水準 profile を何種類にするか。
- State group の粒度と依存関係を manifest でどう表現するか。
- 一つの bundle に全 group を格納するか、manifest から複数 artifact を参照するか。
- Env exact continuation を最初から対象にするか、初期 Phase は Env reset を明示した再開に限定するか。
- ReplayBuffer の大容量 Save / Load を実時間とディスク容量の両面でどう成立させるか。
- Run 中 snapshot の consistency barrier をどの層が所有するか。
- CUDA RNG、非決定論 kernel、Box2D 等の物理状態について、どの fidelity を保証可能と呼ぶか。
- 現行 archive の非決定の実体（model archive 内 offset `0x025A7EE4` 付近）が、保存時刻、build identifier、未初期化 padding のどれか。
- payload digest を group 単位だけで持つか、bundle 全体の合成 digest も持つか。
- 現行 `.anet` checkpoint と `auto_load_file` の互換移行期間をどう設けるか。
- `config_data.txt` と serialization manifest の requested/effective Config をどう重複なく提示するか。

## Further Notes

- `StepCounts` は単なる表示値ではない。epsilon、tau、PER beta、Actor snapshot、将来の learning rate、Batch size、ReplayRatio schedule の意味を決めるため、Run continuation の中核 State として扱う。
- Adam moment を継続し learning rate を変えることは有効な retune であり、optimizer reset と同一ではない。両者を一つの `load optimizer=true/false` に潰さない。
- 「保存できる object が各自 `Save()` を持つ」ことと、「Run として整合した時点を保存する」ことは別問題である。Run-level coordinator は group の所有権を奪わず、quiesce、順序、manifest、transaction を統括する。
- 将来 schedule を導入する場合、cycle ごとに 0 へ戻す schedule と lineage 累積 step で継続する schedule を設定上区別する。暗黙の step offset で補正しない。
- load された値が Config と異なる場合は WARN だけで済ませるか fail-fast するかを policy で決める。少なくとも silent override は禁止する。
