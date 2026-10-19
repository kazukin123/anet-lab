# checkpoint 引き継ぎで optimizer の設定（学習率ほか）を効かせる PRD

> 起点: 2026-10-14 の PRD 整理。DropMerge の次の候補（後半で学習率を下げる、B256 で学習率を半分にする）を、
> 既存 Run の checkpoint から分岐して試すと、設定ファイルで変えた学習率が効かない。
> 事実は [930](930_serialize_10prd.md) の起点そのもので、本 PRD はそのうち Optimizer の retune だけを先に切り出す。
> 改訂: 2026-10-15 再グリル（起票とは別の枠。`grill-with-docs` の anet 版手順: ゴールアンカー → スコープスクリーン → 詳細 5 問 → 最終簡素化パス）。
> options は group ごとに丸ごと置換（旧「決めること 2」を「全 options」で解消）、param group 数の不一致は LibTorch が既に fail-fast するので実装しない
> （旧「決めること 4」を取り下げ）、受入を Learner 単位・Agent 単位・smoke に分解、語を「引き継ぎ」へ統一（CONTEXT.md に 2 項目）、
> 決定の記録は ADR 0053、§10 複雑さ監査と §11 文書同期を追加。見送り 3 件は 930 の「091 からの見送り事項」に記録。
> 関連: [930_serialize_10prd.md](930_serialize_10prd.md)（Run 全体の Save / Load。本 PRD の上位）、
> [ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)（本 PRD で新設）、
> [CONTEXT.md](../../CONTEXT.md)（Agent checkpoint、checkpoint 引き継ぎ）、
> [done/011_fused_adamw_10prd.md](done/011_fused_adamw_10prd.md)（FusedAdamW）、
> [084_noisynet_10prd.md](084_noisynet_10prd.md)（Noisy の sigma 用 param group）。
> 行番号は 2026-10-15 時点。実装時は近傍のシンボル名で再検索する。

## 1. 問題

`DefaultDQNAgent.auto_load_file` で checkpoint を引き継ぐと、optimizer の設定値が checkpoint 側の値に戻る。
設定ファイルや CLI で `learner.alpha` を変えても、実効の学習率は checkpoint を作った Run の値のままになる。
警告も出ないので、気づけない。

同じことが学習率以外の optimizer の設定にも起きる。

| 設定キー（`DefaultDQNAgent.` 配下） | 既定（`@baseline`） | 例 |
|---|---|---|
| `learner.alpha` | `5e-4` | 引き継ぎ時に学習率を下げても効かない |
| `learner.weight_decay` | `1e-2` | 引き継ぎ時に重み減衰を変えても効かない |
| `learner.adam_eps` | `1e-5` | `@bf16` プロファイルは `1e-4` にする（`agent.txt`）。bf16 を切り替えて引き継いでも eps は元のまま |
| `learner.noisy_weight_decay` | `0` | NoisyNet の sigma 用 param group の重み減衰。同上 |

影響しないもの: batch size、replay ratio、soft τ、PER の設定、評価間隔など、optimizer の外にある設定は、
引き継ぎ時も設定ファイルの値が効いている。

## 2. ゴール / 非ゴール

**解く問題**: 上記。DropMerge の長期 Run で実際に起き、学習率を変えたつもりの Run が変わっていなかった。

**ゴール**: checkpoint から分岐した Run で、設定ファイルの optimizer 設定（lr・weight_decay・eps・sigma 用 weight_decay）が
そのまま実効値になる。moment と step は checkpoint から引き継ぐ。checkpoint と設定が違っていたことが、ログで分かる。

**非ゴール**: Run 全体の継続（Replay、`StepCounts` と schedule、正規化統計、RNG）、`continue` / `require_match` の選択肢、
manifest や Run artifact への実効値記録、学習率 schedule、ImageCls・Rainbow・MuZero、device をまたぐ Load。これらは 930 か別 PRD。

## 3. 現状（コードで確認したこと）

**param group**: LibTorch の optimizer は parameter を「param group」という束に分けて持ち、束ごとに自分の options
（lr、weight_decay、eps、betas、amsgrad）を持つ。`step()` は group ごとに、その group の options でその group の parameter を更新する。
anet では NoisyNet OFF なら group は 1 つ（online network の全 parameter）、ON なら 2 つ（group 0 = sigma 以外、group 1 = sigma。
設計文書 200 の「σ 専用 group」）。checkpoint は「group ごとの options」と「parameter ごとの state（moment と step）」を
別々のサブ archive に持ち、読み込みは group を位置順（0 番目同士、1 番目同士）に対応させる。

- optimizer は Learner の構築時に設定から作る。`Learner::SetupOptimizer`
  （`core/anet-core/src/dqn_based_agent.cpp:2512`）が `AdamWOptions(alpha).weight_decay(weight_decay).eps(adam_eps)` を作り、
  NoisyNet が無ければ group 1 つ、あれば 2 つ（sigma の重み減衰は `noisy_weight_decay`）で
  `AdamW` か `FusedAdamW`（`AdamW` の派生、`core/anet-core/include/anet/nn_util.hpp`）を構築する。
  TDLearner / QRLearner / IQNLearner の各コンストラクタから呼ばれ、group ごとの options を作る場所はここだけ。
- 引き継ぎは構築の後に起きる。`DefaultDQNAgent` の構築の最後で `auto_load_file` があれば `LoadNetwork` を呼ぶ
  （`core/anet-core/src/default_dqn_agent.cpp:227-229`）。`LoadNetwork`（同 `:306`）は header、保存時の Config 文字列
  （verbose ログに出すだけ）、Network、Learner の順に読む。
- `Learner::Load`（`dqn_based_agent.cpp:3195`）は `archive.ReadTorchObject(*optimizer_)` だけ。
  LibTorch 2.11 の `serialize(InputArchive&, Optimizer&)`（`torch/csrc/api/include/torch/optim/serialize.h`）は、
  state を parameter へ戻したあと `current_options = saved_options` で各 group の options（lr、weight_decay、eps、betas、amsgrad）を
  checkpoint の値で丸ごと置き換える。options だけ読まない口は公開 API に無い。読み込みの後に設定を当て直す処理も無い。
- 同じ関数が、保存された group の数と現在の group の数、group ごとの parameter 数を `TORCH_CHECK` で照合し、違えば例外で止まる
  （`loaded state dict has a different number of parameter groups`）。NoisyNet の有無が違う引き継ぎは、今でも読み込み時点で fail-fast する。
- `FusedAdamW::load`（`core/anet-core/src/nn_util.cpp:196`）は `AdamW::load` を呼んでから device 側の step tensor を捨てるだけ。
  `save` は override していないので payload は `AdamW` と同じで、`use_fused_optimizer` の違いは引き継ぎに影響しない。
- betas と amsgrad は設定キーが無く、構築時も checkpoint 側も LibTorch の既定なので、値は常に一致する。
- `LearnerConfig` の `alpha`・`weight_decay`・`adam_eps`・`noisy_weight_decay` は `float`。設定を変えずに引き継げば、
  checkpoint 側も同じ float→double 経路で作られた値なので、完全一致の比較で差は出ない。
- 構築時のログ `Learner: lr=... weight_decay=... eps=...`（`dqn_based_agent.cpp:2515`、verbose）は読み込みの**前**に出るので、
  実効値ではなく設定値を示している。ログを見ても気づけない理由はここにある。
- `learner.enabled=false` でも optimizer は構築・読み込みされる（設計文書 200 §6.2）。評価専用プロファイル `run.@evalonly` は
  optimizer のキーを触らない。
- ImageCls は学習率を更新ごとに設定し直している（`core/anet-core/src/image_cls_agent.cpp:409` の
  `SetAdamWLearningRate`）ので、学習率については同じ問題が起きない。`weight_decay` は同じ経路で checkpoint の値に戻る
  （eps は設定キーが無い）。Rainbow と MuZero には引き継ぎの経路が無い。

## 4. 確定した設計

1. **引き継ぎ時は、optimizer の options を現在の設定で置き換える。** moment と optimizer の step は checkpoint から引き継ぐ。
   930 の分類でいう Optimizer group の `retune` を、唯一の挙動にする（[ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md)）。
2. **options を設定から作る場所を 1 つにする。** `SetupOptimizer` の「group ごとの `AdamWOptions` を作る」処理を関数へ切り出す
   （例: `BuildOptimizerOptions()` が group 順の `std::vector<torch::optim::AdamWOptions>` を返す。Noisy OFF は 1 要素、ON は 2 要素）。
   構築時と `Learner::Load` の直後で同じ関数を使う。読み込みの後に別の処理でキー別に値を書き戻す形にはしない。
3. **読み込み直後の手順**（`Learner::Load` の `ReadTorchObject` の直後）: group ごとに、checkpoint 由来の options と 2 の関数が作った options を
   項目ごと（lr、weight_decay、eps、betas、amsgrad）に比べ、差のあった項目を warn に出し、そのあと options を丸ごと代入し、
   group ごとに実効値を info に出す。比較は完全一致。
4. **ログの契約**（英語。`LOG::warn()` / `LOG::info()`）。
   - warn は差のあった項目ごとに 1 行。group 番号、option 名、対応する設定キー名、checkpoint の値、設定値を含む。
     設定キーの対応は lr → `learner.alpha`、weight_decay → `learner.weight_decay`（sigma group は `learner.noisy_weight_decay`）、
     eps → `learner.adam_eps`。betas と amsgrad は設定キーが無いので key 無しで報告する（既定では差が出ない）。
     例: `Learner::Load: optimizer option differs from checkpoint; config applied. group=1 option=weight_decay key=learner.noisy_weight_decay checkpoint=0.01 config=0`
   - info は group ごとに 1 行。例: `Learner::Load: optimizer options applied from config. group=0 lr=5e-05 weight_decay=0.01 eps=1e-05`
   - 構築時の verbose ログは変えない。
5. **設定キーは足さない。** checkpoint の値をそのまま使いたい場合は、設定を checkpoint と同じ値にすればよい
   （checkpoint の Config 文字列は verbose ログで確認できる）。
6. **特別扱いしないもの**: `learner.enabled=false`（読み込みと当て直しは同じに動く。評価専用プロファイルでは差が出ないので warn も出ない）、
   `use_fused_optimizer` の違い（payload は同じ）、param group 数の不一致（LibTorch が止める。追加の検査は書かない）。

設定を変えずに引き継ぐ場合は、checkpoint の値と設定値が同じなので、挙動は今と変わらない。

## 5. User Stories

1. 実験者として、checkpoint から引き継いだ Run で学習率を下げ、その効果を確かめたい。
2. 実験者として、引き継いだ Run の実効の学習率を、ログで確かめたい。
3. 実験者として、設定と checkpoint の値が違っていたときに、それを見落としたくない。

## 6. 確定事項

| # | 事項 | 確定 | 却下・見送りした案と理由 |
|---|---|---|---|
| 1 | 値が違うときにどちらを使うか | 現在の設定（retune）。差は `LOG::warn` に出す | 設定キーで `continue` / `retune` を選ぶ → 起きた痛みが無く、930 の policy として入れる。違ったら fail-fast → 値を変えることが引き継ぎの目的そのもので、意図しない状態と局所判定できない |
| 2 | 対象の options | 全 options。group ごとに設定から作った `AdamWOptions` を丸ごと代入する | lr・weight_decay・eps だけをキー別に書き戻す → 対象キーの一覧をもう 1 つ持つことになり、将来キーを足したとき書き戻し側だけ漏れる。betas を特別扱いする分岐も不要になる |
| 3 | 対象の Agent | DefaultDQN（`dqn::Learner`） | ImageCls の `weight_decay` も揃える → 事故が起きていない。ImageCls で optimizer 設定を変えて引き継ぐ実験が出たら同じ形で直す（930 に記録） |
| 4 | param group の数が checkpoint と違う場合 | 何も足さない。LibTorch が読み込み時点で `TORCH_CHECK` により fail-fast する（§3） | 読み込み前に自前の検査を足す → 既にある検査の重複 |
| 5 | 実効値の見せ方 | ログだけ（§4-4） | 学習率を GetScalar で出す → 学習率 schedule が無い間は一定値で、ログと `config_data.txt` で足りる。schedule を入れるときに起票する（930 に記録） |

## 7. 受入基準

1. **Learner 単位の往復テスト**（`core/anet-core/src/dqn_based_agent_test.cpp`、tag `[dqn][learner][serialize]`）。
   `TDLearner` A を `alpha=1e-4` ほかで構築し、test-only subclass（既存の `UseSgd` と同じ組み立て）から偽の勾配で 1 回 `step` して
   moment と step を作り、`Save` する。`alpha=5e-5`、`weight_decay`、`adam_eps` を変えた Learner B へ `Load` する。
   - 全 group の `lr`・`weight_decay`・`eps` が B の設定値になる。
   - `exp_avg`・`exp_avg_sq`・`step` が A と一致する。
   - `anet::test::LogCaptureGuard` で、warn が差のあった項目の数だけ出て、各行にキー名・checkpoint の値・設定値が入る。
   - NoisyNet ON では group が 2 つで、sigma group の `weight_decay` が設定の `noisy_weight_decay` になる。
   - 同じ設定で `Load` すると warn は 0 件で、options と state は今の実装と一致する（等価性）。
2. **Agent 単位のログ検査**: 既存の Save→`auto_load_file` fixture（`DefaultDQNAgent creates the initial snapshot from an auto-loaded network` と同じ組み立て）で
   `learner.alpha` を変えて読み、warn 行と info 行が出る（`LoadNetwork` 経路の被覆）。
3. **smoke**: 既存 Run の `agent_close.anet` を、**その Run と同じ device** の設定で CLI の `auto_load_file` で読み、
   `learner.alpha` を変えて短く回す。ログに warn 行と info 行が出て、exit code 0 で終わる。device をまたぐ読み込みは
   別の理由で落ちる（930 の「保存時と異なる device への Load」）ので、この smoke では使わない。
4. 本体に test-only API を足さない（テストは test-only subclass と public 経路で組む）。

## 8. 対象外

- Run 全体の継続（ReplayBuffer、正規化の統計、`StepCounts` と schedule、Env、RNG）。930 が扱う。
- checkpoint の値を残す `continue` や、違ったら止める `require_match` の選択肢。930 の policy として入れる。
- manifest、実効値の Run artifact への記録。930 が扱う。
- 学習率の schedule。
- device をまたぐ Load（CUDA で保存した checkpoint を CPU で読む等）。930 の「保存時と異なる device への Load」が扱う。
- ImageCls の `weight_decay` の当て直し。gate: ImageCls で optimizer 設定を変えて引き継ぐ実験が出たら。
- 実効学習率の GetScalar メトリクス。gate: 学習率 schedule を入れるとき。

## 9. 930 との関係

930 は「現行 `auto_load_file` へ個別 boolean を足すだけの暫定拡張」を対象外にしている。本 PRD は設定キーを足さず、
Optimizer group の `retune` を唯一の挙動にするだけなので、これに当たらない。930 が入るときは、本 PRD の挙動が
Optimizer group の `retune` policy になり、`continue` と `require_match` が選択肢として加わる。そのとき ADR 0053 の先頭へ注記を足す。
本 PRD で見送った 3 件（policy の設定キー、GetScalar メトリクス、ImageCls）は 930 の「091 からの見送り事項」に置いた。

過去の Run の読み方は 930 の記述のとおり。引き継ぎ時に optimizer の設定を変えた Run は、checkpoint の値で学習していた。

## 10. 複雑さ監査（2026-10-15 再グリル）

### 10.1 スコープスクリーンと総量の裁定

| 機構・概念 | 裁定 | 根拠（実際の痛み） |
|---|---|---|
| 読み込み直後に optimizer の options を現在の設定へ当て直す | keep | DropMerge の長期 Run で起きた事故 |
| options を作る処理を 1 か所にし、構築時と Load 直後で共用 | keep | 当て直しの形であって別機構ではない。2 か所に分かれると将来キーを足したとき片方だけ更新する温床 |
| 差のあった項目ごとの warn | keep | 同じ事故で「ログを見ても気づけなかった」 |
| group ごとの実効値 info 行 | keep（shrink: 1 行/group） | 読み込み後に optimizer が実際に持つ値を示す唯一の行。最も弱い候補 |
| param group 数不一致の fail-fast | cut | LibTorch が既に止める（§3） |
| `continue` / `require_match` の設定キー | defer（930） | 起きた痛みなし |
| 実効学習率の GetScalar メトリクス | defer（gate: 学習率 schedule） | 今は一定値。ログと config dump で足りる |
| ImageCls の weight_decay 当て直し | defer（gate: ImageCls で optimizer 設定を変えて引き継ぐ実験） | 事故なし。lr は毎 update 設定し直すので問題なし |
| ADR 0053、CONTEXT.md の 2 項目 | keep | 読み込み直後に options を上書きするコードは文脈無しでは「checkpoint を尊重すべき」と直されかねない。継続・再開・引き継ぎの 3 語混在 |
| Agent 単位のログ検査テスト | keep | 無いと `LoadNetwork` 経路の被覆が smoke だけになる。既存 fixture の流用で安い |

### 10.2 要件の実在性

想像上の要件は含まない。`weight_decay`・`eps`・`noisy_weight_decay` の当て直しは、起きたのが `alpha` だけでも、
`alpha` と同じ機構に無償で帯同する（別に作るものは無い）。smoke の同一 device 制約は、2026-10-15 に 930 へ記録された実際の失敗に基づく。

### 10.3 決定の残滓

起票時の暫定判断のうち、後の事実で前提が変わったもの: 旧「決めること 4」（LibTorch が既に fail-fast するので実装しない）、
旧「決めること 2」（丸ごと置換で「全 options」へ）、旧 §3-3（差のあった項目ごとの warn へ精密化）、旧受入 1（Learner 単位と Agent 単位へ分解）、
語「継続」（「引き継ぎ」へ）。矛盾する既存 ADR は無い。

### 10.4 最小解との差分

確定した要求（引き継ぎで設定の optimizer 値が効く、moment と step は引き継ぐ、差がログで分かる）だけを満たす最小設計は、
`Learner::Load` の読み込み直後に設定から作った options を比較・warn・代入・info する 1 か所の処理と、それを支えるテストである。
本 PRD の設計はこれと一致し、差分は無い。

### 10.5 段階の独立性

段階 1（本改訂、文書だけ: PRD、CONTEXT.md、ADR 0053、930 への記録）と段階 2（実装枠: コード、テスト、設計文書 200 / 020）。
用語集の 2 項目は現行の挙動でも真（options の規則は書かず「旧 Run の再開ではない」だけ）、ADR は決定の記録で、
設計文書は未実装の挙動を書かないよう段階 2 へ回す。どちらで止めても前より悪くならない。

### 10.6 成功の測り方

1. 受入 1・2 のテストが緑（lr・weight_decay・eps が設定値、moment・step が一致、warn の件数と内容）。
2. 受入 3 の smoke のログに warn 行と info 行があり、info 行の lr がその Run の `config/config_data.txt` の `learner.alpha` と一致する。
3. 名指しの事故の再発なし: 次の DropMerge の引き継ぎ Run（後半で学習率を下げる、B256 で半分）のログに warn 行が出て、
   `inspect_run.py config --diff` で親 Run との `learner.alpha` の差が見える。その学習曲線の差を学習率の効果として解釈できる。

## 11. 文書同期

### 11.1 今回の文書改訂（本 PRD と同じ変更）

- [CONTEXT.md](../../CONTEXT.md): 「Agent checkpoint」「checkpoint 引き継ぎ」を「Run管理」に追加。
- [ADR 0053](../adr/0053-checkpoint-load-applies-current-config-to-optimizer-options.md): 新設。
- [930](930_serialize_10prd.md): 「091 からの見送り事項」を追加。

### 11.2 コード実装時に同期する対象（実装枠が同じ変更で更新）

- [200_dqn_agents.jp.md](../design/200_dqn_agents.jp.md) §8.2 checkpoint: 引き継ぎの契約に「optimizer の options は現在の設定が決め、
  checkpoint と違った項目は WARN に出す」を足し、語を「引き継ぎ」へ揃える。NoisyNet の σ 専用 group の行も同じ契約に従うことを添える。
  `.en.md` も同じ変更で。
- [020_user_guide_run.jp.md](../design/020_user_guide_run.jp.md) §6.4: 見出しの「再開」を「引き継ぎ」へ。
  手順に 1 文「optimizer の学習率などは新しい Run の設定が効く。checkpoint と違う値はログに WARN が出る」を足す。
  0xx ガイドは PRD も ADR も参照しない。`.en.md` も同じ変更で。
