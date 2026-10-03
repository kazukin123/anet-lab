# NoisyNet の ε は module に持たせず、呼び出し側所有の NN 実行状態と NN 実行設定キーで駆動する

NoisyNet（PRD 084）のノイズサンプル ε と保持状態を、参照実装（BTR、Kaixhin 系）のように `Linear` module の buffer として持たせず、呼び出し側（Actor、Learner の現在値、target 構築）がそれぞれ 1 つ所有する **NN 実行状態** に置き、forward の引数として設定・状態・呼び出しごとの入力を明示的に渡すことにした。理由は 3 つある。(1) anet-lab では shared network を Train Actor・eval Actor・Learner が `shared_lock` で同時に forward するため、module 内の「現在の ε」は読み手どうしで競合する。(2) Train Actor network snapshot の `CopyTo` と hard update は named buffer を複製するため、module に ε を置くと同期のたびにノイズが混ざる（BTR では hard update の `load_state_dict` が実際に ε を複製し、置換 step で target の ε が online の ε になる）。(3) DQN 側のコードが Noisy の有無を調べずに済む。駆動方法（sample / μ-only、保持、共有）は名前付きカタログ **`nn_runtime.[name]`** の項目として NN 層が所有し、各用途がキーで参照する。ε の共有は「実行状態の箱 × network の層 × 時計の値」が一致するときだけ、という単一規則で定める。

## Considered Options

- **module が ε を buffer として持ち、`reset_noise()` で引き直す（参照実装の形）**: 実装は最短だが、同時 forward の競合と snapshot・hard update による混入を避けられない。棄却。
- **用途ごとのキー配下に設定を直接置き、共用は `@` プロファイルと `.$` で書く**: C++ 側の境界（NN 所有の設定型を不透明に渡す）はカタログ案と同じで、名前解決と「未知のキーの fail-fast」が不要になる。[ADR 0038](0038-actor-config-catalog-without-runmode.md) が policy について「カタログ + key」を似た概念の二重化として却下し、選択チェーンで解いた形でもある。今回は「実行時に名前の一覧を残し、将来 GUI 等から名前で切り替える」拡張方針を優先してカタログを選んだ。実害ではなく方針による選択なので、実行時に名前を引く機構はその切替が要件になるまで作らない（PRD 084 §13 のゲート）。この方針を取り下げるときは、設定の読み取り口と設定ファイルの数行を変えるだけで C++ の境界は変わらない。
- **AMP（autocast）も同じ実行設定へ束ねる**: 一度は採用したが外した。ノイズは target 構築全体で 1 つの契約なのに精度は target 行動選択と価値評価で別であり、束ねると target の設定が 2 つになって揃える規則が要る。`Network::Forward` が精度を適用する形にすると ImageCls・MuZero・Rainbow の呼び出しと plasticity の probe チャネル（Learner と同じ autocast）へ波及する。AMP の集約は全 Agent を対象にした別 PRD（`docs/memo/999_nn_runtime_amp_consolidation_10prd.md`）で裁定する。

## Consequences

- `Network::Forward` / `ForwardUpTo` / `TensorDictFunction`、`NetworkModule::Forward`、`NetworkHead::Forward` が実行情報を受け取る。渡さない経路は作らず、構築時の dummy forward、状態スイープ、追加診断、NN 実行設定を持たない Agent はコード固定の μ-only を渡す。対象外 Agent の構造設定に Noisy な `Linear` があっても μ-only で動き、エラーにしない。
- 所有は [ownership_guideline.md](../ownership_guideline.md) の private Resource に当たる。箱は呼び出し側が所有し、中身（層ごとの ε と抽選時の時計、乱数）は NN の計算部品が forward 中に更新する。時計は呼び出し側の State（Learner は `learn_step`、Actor は行動選択回数）で、呼び出しごとの入力として今の値だけを渡す。
- Actor・Learner・共通実行処理は Noisy の有無で forward の挙動と設定検証を切り替えない。例外は Network が返す σ エントリの列挙で、使えるのは optimizer の parameter 分類と診断だけ。
- カタログ項目の識別は CONTEXT.md のカタログの語に合わせて「キー」と呼ぶ（tag は metrics と評価タグの語）。
- 仕様の正本は `docs/memo/084_noisynet_10prd.md`（§4.2、§7.1、§9、§11、§13）。
