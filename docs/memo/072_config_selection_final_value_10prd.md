# PRD 072: カタログ設定の組み立てと完成結果の参照

- 起票日: 2026-09-07。改訂日: 2026-09-12。
- 関連: [PRD 061](061_eval_slot_policy_override_10prd.md)、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)、[設定基盤の現行設計](../design/100_runtime_and_configuration.jp.md)、[用語集](../../CONTEXT.md)。
- 本改訂は仕様とマニュアル草稿の確定まで。リゾルバの実装・検証は本 PRD の後続作業、Actor のコード・設定移行は PRD 061 とする。以下の新契約は実装済みを意味しない。

## 1. 背景とゴール

PRD 061 では、online 評価用の `DefaultDQNAgent.actor.[eval]` を基に、network だけ target に変えた `DefaultDQNAgent.actor.[eval_target]` を定義する。現行の選択は、その時点の source 配下をコピーする。先に作った `[eval_target]` には、後段の A2 / A3 が `[eval]` や共通プロファイルへ加えた変更が届かない。

本 PRD のゴールは、カタログの利用者が「完成した共通設定を継承し、必要な差分を加える」と書けること、および原則と設定例から最終値・存在するキーを予測できることである。選択を依存順に並べ替えると、後段の上書きが負けるため、読み取り対象の完成結果と書き込みの優先順位を分けて扱う。

変更対象はカタログの組み立てとカタログからの参照。その他の通常選択、`run.$`、include、CLI の優先順位、`${}` の段数は維持する。新しい設定モード、互換スイッチ、policy 専用カタログ、Actor 移行は追加しない。

## 2. 現行動作と変更境界

[config_impl.cpp](../../core/anet-core/src/config_impl.cpp) の `ResolutionEngine` / `ResolveSelection` / `ApplyTerm` では、CLI を解決入力へ注入し、Run プロファイルを展開した後、直書き leaf、root 選択の宣言順、各 term の左から右、term が生成した nested 選択の順で反映する。最後に CLI の実効 leaf を再適用し、値参照を 1 段展開する。

現行の `ApplyTerm` は working map のその時点の配下キーを写し、含まれていた `.$` をコピー先で解決する。カタログ項目も区別しない。本 PRD では次の境界で完成結果を使う。

| 選択のコピー先 | 選択元 | 本 PRD の読み方 |
|---|---|---|
| 有効なカタログ項目または配下 | プロファイル | 後段変更と CLI を含む完成値・最終キー集合 |
| 任意の場所 | カタログ項目または配下 | 同上。項目全体・一部分・カタログ外へのコピーを含む |
| 上記以外 | 通常の prefix・上書き層・プロファイル | 既存の、その適用位置での通常選択 |

`@baseline` 内に `actor.[eval]` を書くことは定義の在庫を置くこと。未選択のプロファイル内に `[key]` があるだけで有効化しない。選択によって実効側へ現れたカタログを組み立てる。既存の NN block などにも `[key]` はあるため、「既存設定にはカタログ参照がない」とは仮定しない。既存設定の不変性は §7 の差分ゼロで確認する。

## 3. 確定契約

以下の原則番号をマニュアル例・受入条件から参照する。

### P1: 完成値と最終キー集合を継承する

§2 の対象では、source に後段の選択・上書き層・CLI を適用した完成結果を読む。既存キーの値だけでなく、後から増えたキー、チェーン差し替えで消えたキーも反映する。カタログの一部分を参照する場合は、その部分の完成結果だけを読む。

これは選択の値とキー集合の契約であり、Actor Config 等による既定値補完を含まない。`${}` は後述の最終段で解決し、ここで多段値参照へ拡張しない。

### P2: 同じ選択キーは最後のチェーンへ差し替える

カタログ項目・配下の同じ `.$` に複数の指定が適用される場合、最後の指定だけを、その指定の適用位置で採用する。右辺がプロファイル、カタログ、その混合のどれでも同じ。以前のチェーンだけが供給していた値は残さない。

項目自身への独立した直書き・別の配下選択まで削除しない。それぞれ元の適用位置を保持し、採用チェーンと重なるキーは P4 の優先順位で決める。以前のチェーン内でだけ生成された nested 選択は、独立した指定には含めない。

### P3: 継承元の選択命令をコピー先で再実行しない

完成したカタログからは結果の値を受け取り、継承元の `.$` を継承先の選択命令として持ち込まない。継承先自身に宣言した `.$` は別の指定であり、通常の適用順に従う。これにより、親の設定方法が子の独立した指定を後から書き戻すことを防ぐ。

### P4: 完成結果の読み取りと書き込み優先順位を分ける

- 実効側のデフォルト直書きは選択より弱い。各 term が持つ直書きも、その term が生成する選択より先に反映する。
- チェーンは左から右へ適用し、右が後勝ち。root 選択は宣言順、nested 選択はそれを生成した term の位置を保つ。後段の A2 / A3 の leaf は、前段の選択結果を上書きできる。
- 全体選択と部分選択が重なるときも適用順で決める。「深いパスだから強い」という例外は設けない。
- A1 / A2 / A3 は用途上の名前であり、数字自体に優先順位はない。チェーン内の位置が順位を決める。
- 依存先を先に計算しても、書き込み順位を計算順へ置き換えない。include の順序が宣言の優先順位を変える場合まで、順序非依存とはしない。

### P5: CLI・Run プロファイル・値参照

CLI の解決入力への注入と実効 leaf の最終上書きは維持する。source のプロファイル／カタログを CLI で変更すると P1 の継承先へ届き、継承先の実効キーへの CLI はその継承結果にも勝つ。CLI の `.$` 指定も、採用するチェーンを決める入力となる。

`run.$` は通常選択より先に Run プロファイルを展開する現行契約を維持する。`${full.key}` は選択と CLI の後で値を 1 段参照する。未定義・連鎖・未解決の値参照は既存どおり fail-fast とする。

### P6: 必要な部分の依存関係を検証する

循環は実際に参照する部分の依存関係で判定する。カタログ名が相互に現れるだけでは循環ではない。`policy` と `sync_interval` の独立した部分を相互参照できる一方、同じ値の解決が自身へ戻る参照は経路付きで fail-fast にする。

選択の深さ上限は 10 を維持する。起点の選択を深さ 1 とし、nested 選択または完成結果を求めて辿る選択ごとに 1 増える。兄弟 term の数、パスのセグメント数、反復周回数ではない。カタログの完成結果をキャッシュしても長い依存経路の検証を省略しない。直接宣言した 10 個の選択を辿る成功例と 11 個の失敗例を M14 に固定する。通常選択の既存の深さ・循環の保護も維持する。

採用された参照先を解決した結果が空の場合、プロファイル・カタログ項目（その部分参照を含む）は未定義参照として fail-fast にする。まだ定義を展開していないだけの途中状態で未定義と決めない。空の通常上書き層は許容する。未選択プロファイルの在庫は有効化・参照しない。エラーには選択キー、指定 term、解決先、循環／深さの場合は経路と上限を含める。

### P7: 採用結果を記録する

`config_resolution.json` のカタログに関する `selections` は、同じ選択キーについて最終採用チェーンだけを、その適用順に載せる。継承元の選択を継承先で再実行したような記録を作らない。通常選択の適用記録は従来どおり維持する。

`schema_version = 1`、`selections`（`key` と `chain` の `term` / `resolved`）、`references` の形式を維持する。不採用履歴や採用状態を示すフィールドは追加しない。記録は設定の再読込用ではなく、解決結果を確認する診断情報である。

## 4. ユーザーマニュアル草稿

### 4.1 読む順序と記法

まず P1〜P7 で「どの完成結果を、どの位置で反映するか」を決め、次に記法を読む。以下の Actor の例は **PRD 061 導入後の仕様**であり、リゾルバへ渡す設定の例である。現行 Runner でそのまま学習できる完全な設定ファイルを示すものではない。例内の値は resolver 出力を指し、型付き Config の既定補完前である。

| 記法・要素 | 役割 |
|---|---|
| `前提となる対象・プロファイル : key = value` | 左で対象やプロファイルを示し、右で具体的な Key-Value を書く。左辺の `:` は `.` と同義の見た目上の区切りであり、新しい scope や優先順位ではない |
| `DefaultDQNAgent.actor.@eval_base` | 設定を共通化するプロファイル。Actor キーではなく、選択して使う記述上の部品 |
| `DefaultDQNAgent.actor.[eval]` | 実効側に残り、Actor キー `eval` で参照するカタログ項目 |
| `.$ = source > difference` | コピー先を組み立てる選択チェーン。右が後勝ち |
| `@baseline` | 選択元を owner 相対で記述する形。`DefaultDQNAgent : $ = @baseline` なら `DefaultDQNAgent.@baseline` |
| `A2` / `A3` | チェーン中で適用する通常上書き層。空でもよい |
| `run.$` / include / CLI / `${full.key}` | Run の選択束の展開／宣言入力の取り込み／明示上書き／単一値の参照。P5 の既存順序で扱う |

`:` は左辺に 1 個だけ置ける既存構文であり、先頭・末尾には置かない。任意の位置でキーを切り分ける用途は推奨しない。例では `DefaultDQNAgent.@baseline : actor.[eval].policy.eps_start = 0.01` のように、前提と具体的な Key-Value を区切る。

**左辺の `@` は原則 1 個を推奨する。** 外側の選択で内側の定義も一緒に切り替える必要がある場合は入れ子を許容する。禁止・新しい WARN・構文制約にはしない。右辺の参照数は対象外なので、`$ = @baseline > @iqn > A2` は通常の書き方である。2026-09-11 の現用設定確認では、左辺に複数の `@` を持つ例は Atari の `run.@a5 : @vars.max_exp_step = 50,000,000` など、Run 選択と定義を連動させる用途に存在した。これを一律に書き換える必要はない。

### 4.2 設定例と期待結果

各 M 番号は独立した入力であり、前の例の変更を持ち越さない。「追加する」「置き換える」と記す分岐も、その例の入力だけを起点にする。CLI の表は設定ファイルへの追記ではなく、別途 resolver に渡す override のキーと値を表す。

#### M01: 共通プロファイルと network 差分

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval]` は EpsilonGreedy、eps_start / eps_end がともに 0.05、network が online。`DefaultDQNAgent.actor.[eval_target]` は同じ policy で network が target。`@eval_base` / `@target` の名前は実効設定に残らず、空の A2 / A3 は正常。

根拠: P1・P4・P6。完成した eval を受け取った後、右の @target が network を上書きする。

#### M02: 共通プロファイルへの後段変更

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
A2 : actor.@eval_base.policy.eps_start = 0.01
A2 : actor.@eval_base.policy.eps_end = 0.01
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.@eval_base` の完成値、`DefaultDQNAgent.actor.[eval]`、`DefaultDQNAgent.actor.[eval_target]` の eps_start / eps_end はすべて 0.01。network はそれぞれ online / target のまま。

根拠: P1。カタログは @baseline 適用途中の古いプロファイルではなく、A2 を含む完成値を読む。

#### M03: カタログへの後段変更と新しいキー

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
A2 : actor.[eval].policy.eps_start = 0.01
A2 : actor.[eval].policy.eps_end = 0.01
A2 : actor.[eval].policy.use_amp = true
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval]` と `DefaultDQNAgent.actor.[eval_target]` の eps_start / eps_end はともに 0.01、両方に policy.use_amp = true が存在する。`DefaultDQNAgent.actor.@eval_base` 自体の eps は 0.05 のままで、use_amp は追加されない。

根拠: P1・P4。A2 は eval を変更し、継承先はその完成値と増えたキーを読む。変更は継承元プロファイルへ逆流しない。

#### M04: 継承先だけの後段変更と CLI の範囲

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
A3 : actor.[eval_target].policy.eps_start = 0.02
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: CLI なしでは `DefaultDQNAgent.actor.[eval].policy.eps_start = 0.05`、`DefaultDQNAgent.actor.[eval_target].policy.eps_start = 0.02`。以下の CLI 分岐はそれぞれ独立してこの入力へ適用する。

根拠: P1・P4・P5。継承結果より後の A3 は子だけを変更し、CLI の実効キー指定は最後に勝つ。

| CLI override（各行は独立） | eval の eps_start | eval_target の eps_start |
|---|---|---|
| `DefaultDQNAgent.actor.@eval_base.policy.eps_start=0.03` | 0.03 | 0.02（A3 が子を上書き） |
| `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` | 0.04 | 0.02（完成した親は届くが A3 が勝つ） |
| `A3.actor.[eval_target].policy.eps_start=0.06` | 0.05 | 0.06 |
| `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` | 0.05 | 0.09 |

親からの CLI 伝播そのものは、上の入力から A3 の 1 行だけを除いた分岐でも検証する。その分岐で `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` を渡すと両方 0.04、さらに `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` を同時に渡すと親 0.04・子 0.09 となる。

#### M05: 直書きより選択結果が勝つ

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.@baseline : actor.[eval].policy.eps_start = 0.01
DefaultDQNAgent : actor.[eval_target].policy.eps_start = 0.02
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval].policy.eps_start` と `DefaultDQNAgent.actor.[eval_target].policy.eps_start` はともに 0.05。@baseline 内の leaf を同プロファイル内の選択行より前へ移しても同じ。

根拠: P4。実効側の直書きと同じ term 内の直書きは選択より弱い。子だけ変えるには M04 のように後段の層へ書く。

#### M06: 最後のプロファイルチェーンへ差し替える

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.actor.@greedy : policy.policy_type = Greedy
DefaultDQNAgent.actor.@sync : value = 400
A2 : actor.[eval_target].clone_model = true
A2 : actor.[eval_target].policy.eps_start = 0.02
A2 : actor.[eval_target].sync_interval.$ = DefaultDQNAgent.actor.@sync
A3 : actor.[eval_target].$ = DefaultDQNAgent.actor.@greedy > DefaultDQNAgent.actor.@target
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval_target]` は policy.policy_type = Greedy、network = target、clone_model = true、policy.eps_start = 0.02、sync_interval.value = 400。旧チェーンだけが供給した policy.eps_end は存在しない（0 ではない）。親 eval の eps_start / eps_end は 0.05 のまま。子全体の記録には A3 の採用チェーンだけが載り、旧 eval 継承チェーンは載らない。子自身が A2 で指定した sync_interval の選択は独立して残る。

根拠: P2・P4・P7。旧チェーンを削っても A2 の独立した指定は残る。新チェーンと同じキーで競合すれば A3 が勝つ。

#### M07: 最後のカタログチェーンへ差し替える

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.@baseline : actor.[greedy].policy.policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[greedy].network = online
A2 : actor.[eval_target].clone_model = true
A3 : actor.[eval_target].$ = DefaultDQNAgent.actor.[greedy] > DefaultDQNAgent.actor.@target
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval_target]` は Greedy、target、clone_model = true。policy.eps_start / eps_end は両方存在しない。`[greedy]` はここでは実際に Actor キーとして参照可能な定義であり、共通設定のためだけの仮名ではない。eval は EpsilonGreedy のまま。

根拠: P1・P2・P4。参照先がカタログでも旧継承値は残らず、右の @target が network を上書きする。

#### M08: 多段継承と親の選択命令を再実行しないこと

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.@baseline : actor.[eval_target_check].$ = DefaultDQNAgent.actor.[eval_target]
A2 : actor.[eval].policy.eps_start = 0.01
A2 : actor.[eval_target].policy.eps_start = 0.02
A3 : actor.[eval_target_check].clone_model = true
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval].policy.eps_start = 0.01`、`DefaultDQNAgent.actor.[eval_target].policy.eps_start = 0.02`、`DefaultDQNAgent.actor.[eval_target_check].policy.eps_start = 0.02`。確認用 Actor eval_target_check は target、clone_model = true。親の eval 継承命令を確認用 Actor で再実行して eps_start を 0.01 に戻さない。記録上も eval_target_check のチェーンは eval_target 参照の 1 本だけ。

根拠: P1・P3・P4・P7。各段は親の完成結果を受け取り、自身の宣言だけを適用する。

#### M09: 部分継承とカタログ外へのコピー

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = Greedy
DefaultDQNAgent.actor.@eval_base : policy.use_amp = false
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.[eval].policy
DefaultDQNAgent.@baseline : actor.[eval_target].network = target
DefaultDQNAgent.@baseline : target_policy.$ = DefaultDQNAgent.actor.[eval].policy
A2 : actor.[eval].policy.use_amp = true
DefaultDQNAgent : $ = @baseline > A2
```

期待結果: `DefaultDQNAgent.actor.[eval].policy`、`DefaultDQNAgent.actor.[eval_target].policy`、`DefaultDQNAgent.target_policy` はすべて policy_type = Greedy、use_amp = true。eval_target の network は target。`DefaultDQNAgent.target_policy.network` は存在しない。

根拠: P1・P3・P4。選択した policy 部分だけの完成結果をコピーする。コピー先がカタログ外でも同じ。

#### M10: 独立した部分の相互参照

入力:

```text
DefaultDQNAgent.@baseline : actor.[eval].sync_interval.value = 400
DefaultDQNAgent.@baseline : actor.[eval_target].policy.policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.actor.[eval_target].policy
DefaultDQNAgent.@baseline : actor.[eval_target].sync_interval.$ = DefaultDQNAgent.actor.[eval].sync_interval
DefaultDQNAgent : $ = @baseline
```

期待結果: 正常終了し、`DefaultDQNAgent.actor.[eval]` と `DefaultDQNAgent.actor.[eval_target]` の両方が policy.policy_type = Greedy、sync_interval.value = 400 を持つ。

根拠: P1・P6。policy の依存先と sync_interval の依存先はそれぞれ直書きで終わり、同じ部分へ戻らない。

#### M11: 実際の循環

入力:

```text
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.actor.[eval_target].policy
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.[eval].policy
DefaultDQNAgent : $ = @baseline
```

期待結果: `DefaultDQNAgent.actor.[eval].policy` → `DefaultDQNAgent.actor.[eval_target].policy` → `DefaultDQNAgent.actor.[eval].policy` の経路を含むエラーで停止する。空コピーとして成功させない。

根拠: P6。同じ policy 部分の完成を相互に待つ。

#### M12: 全体と部分の適用順

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@greedy : policy_type = Greedy
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.@greedy
DefaultDQNAgent : $ = @baseline
```

期待結果: `DefaultDQNAgent.actor.[eval_target]` は policy.policy_type = Greedy、network = target。この入力の eval_target の選択 2 行だけを逆順にした独立分岐では、policy.policy_type = EpsilonGreedy、network = target。どちらも全体と部分の選択キーは異なるので両方が記録に残る。

根拠: P2・P3・P4・P7。部分選択を特別に強くしない。逆順では、子自身の部分指定の後に親の完成 policy が書かれる。

#### M13: 未定義参照と空の上書き層

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent : $ = @baseline > A3
```

期待結果: 空の A3 を許容し、`DefaultDQNAgent.actor.[eval].policy.policy_type = Greedy` で正常終了する。次の各分岐はこの入力の eval の右辺だけを置き換える。

根拠: P6。通常上書き層の空と、宣言済み部品を要求する参照の空を区別する。

| eval の右辺（各行は独立） | 期待結果 |
|---|---|
| `DefaultDQNAgent.actor.@missing` | 未定義プロファイルとして fail-fast |
| `DefaultDQNAgent.actor.[missing]` | 未定義カタログとして fail-fast |
| `DefaultDQNAgent.actor.@eval_base.sync_interval` | 指定部分が空のため fail-fast |

いずれも `DefaultDQNAgent.actor.[eval].$`、指定 term、解決先をエラーに含める。カタログ部分の未定義も同じ規則で検証する。

#### M14: 10 段の境界（NN block の検証用カタログ）

入力:

```text
net.block.[DepthBase] : type = ReLU
net.block.[Depth01] : $ = net.block.[DepthBase]
net.block.[Depth02] : $ = net.block.[Depth01]
net.block.[Depth03] : $ = net.block.[Depth02]
net.block.[Depth04] : $ = net.block.[Depth03]
net.block.[Depth05] : $ = net.block.[Depth04]
net.block.[Depth06] : $ = net.block.[Depth05]
net.block.[Depth07] : $ = net.block.[Depth06]
net.block.[Depth08] : $ = net.block.[Depth07]
net.block.[Depth09] : $ = net.block.[Depth08]
net.block.[Depth10] : $ = net.block.[Depth09]
```

期待結果: `net.block.[Depth01]` から `net.block.[Depth10]` まで、すべて type = ReLU で正常終了する。最長経路は Depth10 の選択（深さ 1）から Depth01 の選択（深さ 10）まで。DepthBase は直書きだけなので、選択の深さを増やさない。

根拠: P6。完成結果の参照でも選択の依存経路を数える。

失敗分岐の入力は上の 11 行に次の 1 行を追加したもの。この分岐も他の M 例と組み合わせない。

```text
net.block.[Depth11] : $ = net.block.[Depth10]
```

期待結果: Depth11 から Depth01 の選択まで 11 段になるため、上限 10 と経路を含むエラーで停止する。キャッシュや宣言の並べ順によって成功へ変わらない。外側の通常選択から生成した nested 選択を使うテストでは、外側の深さも数える。

#### M15: 外側のプロファイルと内側の定義を一緒に切り替える

入力:

```text
DefaultDQNAgent.@baseline : actor.@eval_base.policy.policy_type = EpsilonGreedy
DefaultDQNAgent.@baseline : actor.@eval_base.policy.eps_start = 0.05
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval]
DefaultDQNAgent.@cautious : actor.@eval_base.policy.eps_start = 0.01
DefaultDQNAgent.@unused : actor.[unused].$ = DefaultDQNAgent.actor.@missing
DefaultDQNAgent : $ = @baseline > @cautious
```

期待結果: `DefaultDQNAgent.actor.[eval]` と `DefaultDQNAgent.actor.[eval_target]` は EpsilonGreedy、eps_start = 0.01。@cautious をチェーンから除く独立分岐では両方 0.05。未選択の @unused は実効側へ出ず、その中の @missing 参照で停止しない。@eval_base がコピーにより生まれる前に未定義扱いしない。

根拠: P1・P6 と §4.1。定義も選択と連動させる目的がある場合の、左辺に @ が 2 個ある許容例。

#### M16: 通常選択の読み方は維持する

入力:

```text
DefaultDQNAgent.@policy_base : policy_type = Greedy
DefaultDQNAgent.@policy_base : use_amp = false
DefaultDQNAgent.@baseline : target_policy.$ = DefaultDQNAgent.@policy_base
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.@policy_base
A2 : @policy_base.use_amp = true
DefaultDQNAgent : $ = @baseline > A2
```

期待結果: `DefaultDQNAgent.target_policy.use_amp = false`、`DefaultDQNAgent.actor.[eval].policy.use_amp = true`。policy_type は両方 Greedy。通常選択 target_policy は、その位置で読んだプロファイルを後段の変更で読み直さない。

根拠: §2・P1・P4。通常選択まで完成値参照へ広げると、この false が true に変わって契約違反となる。

#### M17: 配下チェーンの差し替えと最終キー集合の伝播

入力:

```text
DefaultDQNAgent.actor.@eps_policy : policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eps_policy : eps_start = 0.05
DefaultDQNAgent.actor.@eps_policy : eps_end = 0.05
DefaultDQNAgent.actor.@greedy_policy : policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.actor.@eps_policy
DefaultDQNAgent.@baseline : actor.[eval].network = online
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval]
A2 : actor.[eval_target].network = target
A3 : actor.[eval].policy.$ = DefaultDQNAgent.actor.@greedy_policy
DefaultDQNAgent : $ = @baseline > A2 > A3
```

期待結果: `DefaultDQNAgent.actor.[eval]` と `DefaultDQNAgent.actor.[eval_target]` は Greedy、policy.eps_start / eps_end は両方に存在しない。network は online / target。eval.policy の記録は @greedy_policy だけで、eval_target に親の policy.$ の記録を複製しない。

根拠: P1・P2・P3・P4・P7。配下でも最後のチェーンへ差し替え、消えたキーを継承先に残さない。

## 5. 実装ノート

実装の主対象は [config_impl.cpp](../../core/anet-core/src/config_impl.cpp) と [config_test.cpp](../../core/anet-core/src/config_test.cpp)。まず「通常の適用位置を持つ指定」と「カタログの完成結果への依存」を区別する。プロファイルの展開で生成される宣言、owner 相対の term、CLI が変更した宣言を取り込んだ上で、同じカタログ選択キーの採用チェーンを確定する。

完成結果は値とキー集合の両方を解決する。部分参照では要求した部分への依存を辿り、継承元の選択命令を子へ再配置しない。独立した指定の順位は残す。キャッシュ等で解決順を変えても、適用順位・循環経路・深さ・記録が変化してはならない。

この説明は責務と制約であり、データ構造やアルゴリズムの指定ではない。設定全体の Jacobi 反復、last-writer graph、周回数の安全弁を必須方式にしない。実 config で生じない writer graph を注入するためだけの API やテストも要求しない。選んだ方式は M01〜M17 と既存テストで説明・検証できる最小構成にする。

通常選択の snapshot 読み、プロファイルの dormant 状態、`run.$` の先行展開、CLI 2 相、`${}` 1 段展開を維持する。カタログかどうかの判定は `[key]` セグメントと有効な定義の文脈に基づき、Actor 固有の prefix や eval という名前へ固定しない。

## 6. 受入条件とマニュアルの対応

各 M 例の「入力・期待結果・根拠」を実装時の resolver テストへ対応付ける。表の条件を一部の happy path だけで代用しない。

| 受入 ID | 対応例・入力 | 確認する結果と原則 |
|---|---|---|
| A01 | M01〜M03 | 共通プロファイル・カタログの後段変更と追加キーが届く。変更は親へ逆流しない（P1） |
| A02 | M04 の CLI なし／各 CLI 分岐 | 子だけの変更、source CLI の伝播と子 CLI の最優先を区別する（P4・P5） |
| A03 | M05、M12 と逆順分岐 | 直書きより選択が勝ち、全体・部分は適用順で勝敗が変わる（P4） |
| A04 | M06・M07・M17 | プロファイル／カタログ、項目全体／配下のチェーン差し替えで旧継承値が消え、独立した指定は順位を保つ（P2） |
| A05 | M08・M09 | 多段・部分継承とカタログ外への参照は完成結果を使い、親の命令を再実行しない（P1・P3） |
| A06 | M10・M11 | 独立した部分の相互参照は成功、実際の循環は経路付きエラー（P6） |
| A07 | M13 の各分岐 | 未定義プロファイル／カタログ／部分はエラー、空の通常層は成功（P6） |
| A08 | M14 の成功／失敗分岐 | 10 段成功・11 段失敗。通常選択を含む nested 深さの既存テストも維持（P6） |
| A09 | M15、既存の相対 term・入れ子展開テスト | 後から生じるプロファイルを読める。未選択の在庫は実効化しない（P1・P6） |
| A10 | M16、既存の root 宣言順・nested 選択テスト | カタログ以外の通常選択と後段 overlay の優先順位を維持（§2・P4） |
| A11 | M06・M08・M12・M17、既存の解決記録テスト | カタログは最終採用チェーンだけ。通常選択の適用記録と JSON 形式を維持（P7） |
| A12 | 既存の include・trunk・CLI・値参照テスト | 入力順、Run プロファイル、CLI、1 段値参照とその異常系を維持（P5） |
| A13 | §7 の全対象 | 実効設定と解決記録の差分ゼロ。差分があれば未完了 |

チェーン差し替えの追加境界として、M06 の入力に `DefaultDQNAgent.actor.@greedy : clone_model = false` を追加する独立分岐では、A2 の true より後の新チェーンが勝ち、`DefaultDQNAgent.actor.[eval_target].clone_model = false` となることを確認する。M13 では M09 の `target_policy` の右辺を存在しない `DefaultDQNAgent.actor.[eval].sync_interval` へ置き換える独立分岐も追加し、カタログの部分参照が空ならエラーになることを確認する。

CLI のチェーン指定、同じ `.$` の include による後勝ち、owner 相対参照の生成先は既存の適用順を入力として検証する。テストのために優先順位を変更しない。

## 7. 既存設定の差分ゼロを完了条件にする

PRD 072 のリゾルバ実装前後で、同じ設定入力・include・CLI・注入値を固定し、ConfigManager が公開する `config_data.txt` 相当の実効設定と `json/config_resolution.json` の payload を比較する。解決記録は内容だけでなく順序も対象。外側のログ時刻等を比較対象と混同しない。以下は今後の実装時の検証であり、今回実施した結果ではない。

| Env 設定 | 必須の比較入力 |
|---|---|
| Atari.txt | 下記の Run プロファイル 5 チェーン |
| DropMerge.txt | 実装着手時の現行 IQN32 チェーンを CLI 込みで固定し、さらに `run.@iqn32_stratified` と `run.@qr51_control` |
| LunarLander.txt | ファイル既定と `run.@repro` |
| ImageCls.txt | ファイル既定、`run.@resnet18ish_hr`、`run.@convnext_atto_hr` |
| GridMaze.txt | ファイル既定（現行の Agent / NN 選択を含む） |
| GridMaze_muzero.txt | ファイル既定（MuZero の選択を含む） |
| CartPole.txt | ファイル既定（NN block カタログの選択を含む） |

Run プロファイルを宣言していない Env に比較用の新プロファイルは作らず、既定入力を使う。実装時に代表入力が変わっていたら、採取対象の名前・入力を先に記録し、比較範囲を無断で減らさない。

Atari の `run.$` に渡す 5 チェーン:

1. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard125>run.@munch`
2. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@capall`
3. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@evalonly>run.@greedy_eval>run.@to_50`
4. `run.@nature_dqn`
5. `run.@classic_iqn_impala_x2`

既存の resolver / trunk / CLI テストをすべて通し、上記入力の実効設定・解決記録が**差分ゼロ**になって初めて PRD 072 の実装完了とする。差分は単に記録して許容せず、原因と契約を再確認し、解消するまで未完了とする。既存 Run artifact を書き換えて一致させない。事前 baseline を採れなかった場合はその不足を記録し、比較合格とは扱わない。反復周回数や長時間学習の成績は受入条件にしない。

## 8. 検討経緯と最後の簡素化監査

初期案は全選択の最終値読み取りを目指し、依存順への並べ替え、同順再実行、Jacobi 反復と writer graph の検討へ広がった。今回、必要なのはカタログの組み立てと完成結果の参照であり、その他の通常選択は維持する、と範囲を確定した。

| 旧案 | 今回の判断と理由 |
|---|---|
| 選択を依存順へ並べ替える | 却下。後段 overlay が負けるなど書き込み優先順位を変える |
| 書きかけ map を同順で繰り返し読む | 必須方式から除外。前段が親を戻すと、その古い値を再びコピーしうる |
| 全設定の Jacobi 反復、last-writer graph、周回数の安全弁 | 必須方式から除外。通常選択まで変える根拠がなく、今回の契約に方式依存の仕組みを増やす |
| 回数による収束・長い通常 prefix の伝播 | 受入から削除。カタログ契約の正しさを測らず、通常選択の変更を要求していた |
| policy カタログ + policy_key | 追加しない。共通プロファイルと Actor カタログで表せる |

合意したゴールに照らした最後の簡素化監査:

| 観点 | 判断 | 残す／削る内容と理由 |
|---|---|---|
| 仕組み全体の過剰さ | keep / cut | 完成値・最終キー集合、チェーン差し替え、優先順位、必要な検証と採用記録を残す。全体反復と不採用履歴の仕組みは削る |
| 要求の実在性 | keep | 後段変更が継承先に届かない問題と、設定例から結果を予測できない問題に対応する。仮想的な将来拡張は加えない |
| 前提変更後の残滓 | cut | 全選択を最終値化するための方式指定・周回数・writer graph 内部注入テストを削る |
| 最小解との差分 | shrink | 原則、必要な異常系、解決記録、マニュアルと受入対応に絞る。新モード・互換層・広範な整形は加えない |
| フェーズの独立性 | keep | PRD 072 は既存 NN カタログも対象に resolver 単独で完了可能。Actor コード・設定移行は PRD 061 で行う |
| 成功の測定可能性 | keep | 例とテストの一致、既存実効設定・記録の差分ゼロ、実装を知らず原則から結果を説明できること。長時間学習を要求しない |

## 9. この文書改訂の検証範囲

グリル中のメモリ内モデルでは、順序を持つ直書きと参照を使い、(1)プロファイル後段変更、(2)カタログ後段変更、(3)子の後段変更、(4)直書きより選択が勝つ、(5)差し替えによる旧キー消失、(6)カタログ外への部分参照、(7)独立した部分の相互参照、(8)全体の後に部分、(9)部分の後に全体、(10)実際の循環、の **10 例を確認済み**。これは概念モデルの確認記録であり、M01〜M17 全体や production resolver が通ったという意味ではない。

| 検証 | 今回の位置付け |
|---|---|
| 4 文書の用語・対象範囲・リンク、例の入力／期待結果／原則の対応、変更差分 | 確認済み。17 例と 13 受入項目の対応、リンク先、UTF-8 / LF、コードフェンス、差分の空白を点検 |
| メモリ内モデルの 10 例 | グリル中の限定した確認。実装証明ではない |
| 実パーサによる `:` 正規化、include、入れ子プロファイル展開・owner 相対参照との統合 | 未実施。実装時の A09・A10・A12 で確認 |
| C++ のビルド・resolver テスト、10 段境界、JSON 解決記録 | 未実施。実装時に確認 |
| 各 Env / Run プロファイルの実効設定・解決記録の前後比較 | 未実施。§7 の完了 gate として残す |

今回は文書だけを変更し、コード・設定・Run artifact は変更しない。
