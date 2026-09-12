# PRD 072: 設定継承の差分適用と最終値参照

- 起票日: 2026-09-07。改訂日: 2026-09-13。
- 関連: [PRD 061](061_eval_slot_policy_override_10prd.md)、[ADR 0042](../adr/0042-config-inheritance-as-differential-base.md)、[設定基盤の設計](../design/100_runtime_and_configuration.jp.md)、[用語集](../../CONTEXT.md)。
- 本改訂はレビュー用の設計文書更新。以下は合意した実装対象の契約であり、コード・現用設定・テスト・goldenの移行は未実施である。
- 2026-09-12版の実装とgolden比較の記録は[20impl](072_config_selection_final_value_20impl.md)と[ADR 0040](../adr/0040-config-selection-final-value-and-run-profile-tier.md)に保持する。過去の検証結果を本改訂の合格証拠にしない。
- 本PRDは`999_config_run_profile_override_precedence_10prd.md`を吸収する。ActorのAPI・設定カタログへの移行はPRD 061が扱う。

## 1. 背景とゴール

PRD 072の当初の課題は、選択元への後段変更が継承先へ届かないことと、Runプロファイルの指定が通常選択に負けることだった。2026-09-12版では最終値参照とRunの優先を実装したが、選択元を「宣言を運ぶ入れ物」と「組み立て済みノード」に分け、ドット無しの名前を上書き層として判定している。

この区別は、別々の選択元から来た選択宣言をコピー先で差し替え、以前の選択だけが生成した葉を消すためのものだった。しかし、利用者が求める継承は差分適用である。`A2 > A3`でA3にないキーはA2から残す。A2の葉が直書きか継承由来かで結果を変える必要はない。

**ゴール: `$`でベースを組み立て、その場所の個別指定・部分指定で上書きできる。複数の選択元は各最終値を右側の差分で重ねる。** 名前のドット数、A1等の予約名、`[key]`の有無で継承の規則を変えない。

上書き層の名称・段数は設定運用上の約束として残せるが、resolverが識別する種類にはしない。新しい構文・モード・互換スイッチ・キー削除演算子・葉の削除用の由来追跡は追加しない。

## 2. 実装済みの先行版から変わること

| 論点 | 2026-09-12版のコード・テスト | 本改訂の契約 |
|---|---|---|
| 同じ設定の個別葉と`$` | 継承元が個別葉に勝つ | 個別葉が継承したベースに勝つ |
| 全体と部分の`$` | 宣言の行順で後勝ち | 同じ設定では部分指定が勝ち、行順に依存しない |
| 別々の選択元 | 運ばれた選択宣言を差し替え、旧選択の葉を除外 | 各最終値の差分合成。右側にない葉は残す |
| ドット無しの選択元 | 上書き層として特別扱い | Common等も通常prefixとして扱う |
| `@b`の参照先 | コピー先を基準に決まる | 宣言の定義元を基準に決まる |
| 選択の記録 | プロファイル内の宣言もコピー先のキーで記録 | 宣言の定義位置で記録 |
| 弱い既定値の直書き | 選択で上書きされる | 強い個別指定になるため、必要箇所をベースへ移す |

最終値の伝播、Run・CLIの同一キーに対する優先、未選択プロファイルの休止、未定義・自己供給・循環・深さの検証、`${}`の1段参照は維持する。公開APIとresolution JSONのフィールド構成は変更しない。

## 3. 確定契約

### P1: 選択は参照先の最終値を差分合成する

`X.$ = Y > Z`は、YとZがそれぞれのベース、部分指定、個別指定、Run・CLIを反映した最終値とキー集合を読み、YへZの差分を重ねる。Zが持つキーだけを上書きし、ZにないキーはYから残す。値の由来による削除はしない。

選択元が`@`プロファイル、`[key]`カタログ、通常prefix、その一部分のいずれでも同じである。参照先への後段変更・追加キーは、その参照先の優先順位を解決した最終値として継承先へ届く。継承先の変更は参照先へ逆流しない。

参照先の`.$`は参照先の組み立てに用い、コピー先の選択命令として再実行しない。未選択プロファイルの定義は休止したまま保持し、選択で供給される内側プロファイルも、必要になった定義の最終値を読む(M13)。先にすべての在庫を実効化することは要求しない。

この契約は文字列設定の値とキー集合についてのもの。型付きConfigによる既定補完、使用していないキーの自動削除、`${}`の多段展開は含まない。

### P2: 同じ設定では個別・部分指定を優先する

1. その設定の`.$`で全体のベースを作る。
2. その設定の配下に書いた部分の`.$`を重ねる。全体より具体的な部分の指定が強い。
3. その設定に直接書いた葉を重ねる。同じ葉を供給するベース・部分選択より個別葉が強い。

異なるキーの宣言を並べ替えても、この優先関係は変わらない。プロファイル内外で同じ規則を使う。`Env.@a.$ = @b`と`Env.@a.value = own`では、@bにvalueがあってもownが勝つ。通常の`Env.value = own`も`Env.$`の結果に勝つ。

この具体性の比較は、各選択元を組み立てる範囲の規則である。**別々の選択元を`A2 > A3`で重ねるときは、A3の最終値がA2の同名キーに勝つ。** A2内に深い部分指定があっても、後段A3の値より強くしない(M19)。A1等の数字自体には優先順位がない。

直接書いた既定値も個別指定になる。弱い既定値が必要な箇所は、チェーン先頭のプロファイルに置く(§7)。外側の選択でA2から供給した値は、参照先に直接書いてある葉や部分指定を無条件には上書きしない(M02、M07)。

### P3: 同じ入力キーの書き直しと差分合成を区別する

同じ設定キーを複数回定義した場合は、include等を取り込む従来の後勝ちで最後の値を採用する。`Env.$ = A2`の後に`Env.$ = A3`と書けば、Envのベース指定はA3だけである。A2を一度実行してから消すことは要求しない。

`Env.$ = A2 > A3`は別の指定である。A2とA3の両方の最終値を差分合成する。各選択元の中に同じ相対位置の`.$`があっても、それらをEnvの単一の選択宣言へまとめて差し替えない。A3にないA2由来の葉は、継承で得た葉も含めて残す(M06)。

したがって、異なる入力キーの順序非依存(P2)と、同じ入力キーの再指定の後勝ちは両立する(M18)。空値は既存の文字列値であり、キー削除の指示として解釈しない。

### P4: Run・CLIの同一キーに対する優先を維持する

通常の設定値はP1〜P3で決め、そのキーへのRunプロファイルの明示指定が勝ち、CLIの明示指定がさらに勝つ。解決入力キー(`.$`・`@`を含む定義)でもCLIをRunプロファイルより強くする。

Runプロファイルは通常選択より前に展開し、葉についてはキーごとの最終指定を畳んで一度だけ適用する。Runの項は左から右の後勝ち。自身のリテラル葉を対象とし、別の`run.$`を供給するネストは禁止する。CLIの`run.$`指定はRun選択そのものを変更する。

Run・CLIによる参照先の最終値変更は継承先へ伝播する。ただし、親キーへのCLI指定が「親から継承した値」に付いたまま、子の個別指定まで無条件に上書きするわけではない。子の結果は子自身のP2でも決まり、子のキーを直接指定するCLIが最優先になる(M04)。

第1相・第2相という先行版の実装表現を保つかは実装時に決める。契約は優先順位、伝播、適用対象、診断結果であり、選択の再実行を必須にしない。

### P5: 相対参照は定義元で解決する

単独の`@name`は、選択宣言の定義元の名前空間を基準に読む。通常の`Env.$ = @a`は`Env.@a`、プロファイル自身の継承`Env.@a.$ = @b`は同じ名前空間の`Env.@b`を指す。`Env.@a.@b`を暗黙に探す規則にはしない。

`Other.$ = Env.@a`で使っても、@aの中の`@b`は`Env.@b`のままである。Other.@bが存在しても参照先を差し替えない。完全修飾したtermは記載どおりのprefixを指す(M16)。

`${full.key}`は解決済みの最終値を1段参照し、未定義・連鎖・未解決の値参照をfail-fastにする。これは選択の`@name`の名前解決とは別の契約である。

### P6: 必要な依存関係を検証する

名前のセグメント数や予約名で選択を禁止しない。`Common.$`を定義したCommonの継承、`A2.$`を持つA2の継承も、通常の依存関係として扱う(M17)。

自己供給・実際の循環は従来どおり経路付きでfail-fastにする。`X.$ = X`や`X.part.$ = X`を検出し、sourceとownerが包含関係にあるだけでは禁止しない。`app.$ = app.online > P1`など、参照元への書き戻しがない形は許容する。独立した部分の相互参照を、名前だけで循環と判定しない。

選択の深さ上限は10。起点を1とし、最終値を求めて辿る選択の依存グラフの最長経路で数える。兄弟termの数・パスのセグメント数・反復回数ではない。キャッシュや宣言順で判定を変えない。Runの先行展開は深さに数えない。

未定義のプロファイル・カタログ・その部分への参照はエラー。空の通常prefixは既存どおり許容する。この空許容はA1等の名前やドット無しに限定しない。未選択プロファイル内の不正な参照は有効化しない。エラーには選択キー、指定term、解決先、必要な経路・上限を含める。

### P7: 解決記録は宣言の定義位置を示す

JSONは`schema_version = 1`と、既存の`selections` / `references` / `overrides`の形を維持する。`selections[].key`は選択宣言がある定義位置、`chain[].term`は記載したterm、`resolved`は定義元で解決した参照先を示す。`Env.$ → Env.@a`と`Env.@a.$ → Env.@b`を記録し、後者を`Env.$ → Env.@b`へ置き換えない(M16)。

`run.$`のentryは先頭に維持する。その他は入力の宣言順とtermの左から右を基準に依存先を辿り、実際に使う定義を初めて参照した順で記録する。同じ定義の再参照でコピー先名のentryを増やさない。この記録順はP2の上書き優先順位を表すものではない。

`references`は従来の1段値参照を記録する。`overrides`はRunの最終指定が同じキーのRun適用前の最終値を変えた場合だけ、`key` / `by` / `from` / `to`で記録する。途中の4→1→4は載せない。`to`はRunの値であり、同じキーをCLIが上書きした後の実効値とは異なりうる。最終値は`config_data.txt`で確認する。

不採用履歴、削除した葉の由来、新しい採用状態フィールドは追加しない。旧記録との完全一致を要求せず、差分の理由と新仕様の期待値を検証する(§7)。

## 4. ユーザーマニュアル草稿

### 4.1 読む順序と記法

1. `$`はベース。個別指定・部分指定で上書きできる。
2. `Y > Z`はそれぞれの最終値の差分合成。右側にないキーは残る。
3. 短い`@name`は定義元を基準に読み、使う場所で参照先を変えない。

| 記法・要素 | 役割 |
|---|---|
| `DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05` | 名前付きプロファイルの定義。`:`はプロファイルprefixと内部キーの区切りとして使う |
| `DefaultDQNAgent.actor.[eval].$ = DefaultDQNAgent.actor.@eval_base` | カタログ項目のベース指定 |
| `DefaultDQNAgent.actor.[eval].policy.eps_start = 0.01` | 継承したベースより強い個別指定 |
| `DefaultDQNAgent.$ = @baseline > A2 > A3` | 定義元の@baselineと通常prefix A2・A3の最終値を順に合成 |
| `run.$` / include / CLI / `${full.key}` | Run選択／入力の取り込み／明示上書き／1段値参照 |

`:`のパーサ仕様は変更しないが、説明・新規記述では`@`プロファイルの区切り以外に使うことを推奨しない。通常キーは`Env.$ = @a`のように`.`で書く。左辺の`@`は原則1個を推奨し、外側の選択で内側の定義も切り替える必要がある場合だけ入れ子を使う。禁止・新しいWARNは追加しない。

以下のActorカタログ例はPRD 061導入後の設定モデルを用いたresolver入力であり、現行Runnerでそのまま学習できる完全な設定ファイルではない。値は型付きConfigの既定補完前のものを示す。

### 4.2 設定例と期待結果

各 M 番号は独立した入力であり、前の例の変更を持ち越さない。CLI の表は設定ファイルへの追記ではなく、別途 resolver に渡す override のキーと値を表す。

以下 M01〜M07 で共通に使う定義を `BASE` と呼ぶ。

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.05
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.05
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.$ = @baseline > A2 > A3
```

#### M01: 共通プロファイルと network 差分

入力: `BASE` のみ。

期待結果: `[eval]` は EpsilonGreedy、eps_start / eps_end がともに 0.05、network が online。`[eval_target]` は同じ policy で network が target。`@eval_base` / `@target` の名前は実効設定に残らず、空の A2 / A3 は正常。

根拠: P1・P2。`[eval]` を受け取った後、右の `@target` が network を上書きする。

#### M02: 参照先の個別指定と外側からの差分

入力: `BASE`に追加。

```text
A2.actor.@eval_base.policy.eps_start = 0.01
A2.actor.@eval_base.policy.eps_end = 0.01
```

期待結果: `@eval_base`、`[eval]`、`[eval_target]`のeps_start / eps_endは0.05。BASEで`DefaultDQNAgent.actor.@eval_base`自身に直接書いた0.05が、外側の選択でA2から供給される0.01より強い。**旧M02の「A2が直接定義を上書きして0.01になる」は改訂対象。**

0.01を指定する独立分岐では、A2の2行を使わず、次の同一キーの再指定をBASEの後に書く。

```text
DefaultDQNAgent.actor.@eval_base : policy.eps_start = 0.01
DefaultDQNAgent.actor.@eval_base : policy.eps_end = 0.01
```

この分岐では3者とも0.01になり、networkはonline / targetのまま。参照元へ有効に加えた変更は最終値として伝播する。根拠: P1・P2・P3。

#### M03: 後段で増えたキーも届く

入力: `BASE` に追加。

```text
A2.actor.[eval].policy.eps_start = 0.01
A2.actor.[eval].policy.use_amp = true
```

期待結果: `[eval]` と `[eval_target]` の eps_start はともに 0.01、両方に `policy.use_amp = true` が存在する。`@eval_base` 自体の eps_start は 0.05 のままで、use_amp は追加されない。

根拠: P1。変更は継承元プロファイルへ逆流しない。

#### M04: 継承先だけの後段変更と CLI の範囲

入力: `BASE` に追加。

```text
A3.actor.[eval_target].policy.eps_start = 0.02
```

期待結果: CLI なしでは `[eval].policy.eps_start = 0.05`、`[eval_target].policy.eps_start = 0.02`。以下の CLI 分岐はそれぞれ独立してこの入力へ適用する。

根拠: P1・P2・P4。継承は土台を配り、後段の直接指定がその上に乗る。

| CLI override（各行は独立） | eval の eps_start | eval_target の eps_start |
|---|---|---|
| `DefaultDQNAgent.actor.@eval_base.policy.eps_start=0.03` | 0.03 | 0.02（A3 が子を上書き） |
| `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` | 0.04 | 0.02（最終値は届くが A3 が勝つ） |
| `A3.actor.[eval_target].policy.eps_start=0.06` | 0.05 | 0.06 |
| `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` | 0.05 | 0.09 |

親からの CLI 伝播そのものは、A3 の 1 行を除いた分岐でも検証する。その分岐で `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` を渡すと両方 0.04、さらに `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` を同時に渡すと親 0.04・子 0.09 となる。

親からの最終値の伝播と子自身の優先を分けて検証する。2026-09-12版で実装したRun・CLIの伝播は本改訂でも維持する。

#### M05: 同じ設定では個別指定がベースに勝つ

入力: `BASE`に追加。

```text
DefaultDQNAgent.@baseline : actor.[eval].policy.eps_start = 0.01
DefaultDQNAgent.actor.[eval_target].policy.eps_start = 0.02
```

期待結果: `[eval].policy.eps_start = 0.01`、`[eval_target].policy.eps_start = 0.02`。両方の個別葉をそれぞれの選択行より前に置いても同じ。個別葉を消した箇所は継承したベースの値になる。

プロファイル自身の継承も独立入力で確認する。

```text
Env.@a : value = own
Env.@a : $ = @b
Env.@b : value = base
Env.$ = @a
```

期待結果は`Env.value = own`。通常の`Env.value = own`と`Env.$ = @b`の組でもownが勝つ。根拠: P2。

#### M06: 別々の選択元は差分を重ね、前の葉を残す

入力: `BASE`に追加。

```text
DefaultDQNAgent.actor.@greedy : policy.policy_type = Greedy
DefaultDQNAgent.actor.@sync : value = 400
A2.actor.[eval_target].clone_model = true
A2.actor.[eval_target].policy.eps_start = 0.02
A2.actor.[eval_target].sync_interval.$ = DefaultDQNAgent.actor.@sync
A3.actor.[eval_target].$ = DefaultDQNAgent.actor.@greedy > DefaultDQNAgent.actor.@target
```

期待結果: `[eval_target]`はpolicy_type = Greedy、network = target、clone_model = true、eps_start = 0.02、sync_interval.value = 400。**前の選択元から来たeps_end = 0.05も残る。** A3の結果にeps_endがないためである。A3の参照先を同じ値のカタログへ変えても同じ。

独立入力として、由来を分けずに残すことを確認する。

```text
DefaultDQNAgent.actor.@eps : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eps : policy.eps_end = 0.05
DefaultDQNAgent.actor.@greedy : policy.policy_type = Greedy
A2.actor.[eval].$ = DefaultDQNAgent.actor.@eps
A3.actor.[eval].$ = DefaultDQNAgent.actor.@greedy
DefaultDQNAgent.$ = A2 > A3
```

期待結果: `[eval].policy.policy_type = Greedy`、`[eval].policy.eps_end = 0.05`。A2も通常prefixとして解決され、`A2.actor.[eval].policy.eps_end = 0.05`が存在する。A2にeps_endを直接書く分岐でも、継承から得る分岐でも結果は同じ。

新しい`@greedy`に`clone_model = false`を追加した最初の例の分岐では、右側A3がそのキーを持つためfalseが勝つ。根拠: P1・P2・P3。

#### M07: 多段継承

入力: `BASE` に追加。

```text
DefaultDQNAgent.@baseline : actor.[eval_target_check].$ = DefaultDQNAgent.actor.[eval_target]
A2.actor.[eval].policy.eps_start = 0.01
A2.actor.[eval_target].policy.eps_start = 0.02
A3.actor.[eval_target_check].clone_model = true
```

期待結果: `[eval].policy.eps_start = 0.01`、`[eval_target].policy.eps_start = 0.02`、`[eval_target_check].policy.eps_start = 0.02`。`[eval_target_check]` は network = target、clone_model = true。親の eval 継承命令を再実行して 0.01 に戻さない。記録上も `[eval_target_check]` のチェーンは 1 本だけ。

根拠: P1・P2・P3・P7。各段は親の最終値を受け取り、自身の宣言だけを適用する。

直接宣言の分岐(独立入力):

```text
DefaultDQNAgent.actor.@eval_policy : eps_start = 0.05
DefaultDQNAgent.actor.[eval].policy.$ = DefaultDQNAgent.actor.@eval_policy
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval]
A2.actor.[eval].policy.eps_start = 0.01
DefaultDQNAgent.$ = @baseline > A2
```

期待結果: 両者とも0.05。`[eval].policy.$`という部分への直接指定が、外側のAgent選択でA2から届く0.01より強い。**旧M07のこの分岐は改訂対象。** A2の行を`DefaultDQNAgent.actor.[eval].policy.eps_start = 0.01`へ置き換えた独立分岐では、個別葉が勝ち、両者とも0.01となる。いずれも参照元の`policy.$`を継承先で再実行せず、記録に架空の`[eval_target].policy.$`を増やさない。

#### M08: 部分継承とカタログ外へのコピー

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = Greedy
DefaultDQNAgent.actor.@eval_base : policy.use_amp = false
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.[eval].policy
DefaultDQNAgent.@baseline : actor.[eval_target].network = target
DefaultDQNAgent.@baseline : target_policy.$ = DefaultDQNAgent.actor.[eval].policy
A2.actor.[eval].policy.use_amp = true
DefaultDQNAgent.$ = @baseline > A2
```

期待結果: `[eval].policy`、`[eval_target].policy`、`DefaultDQNAgent.target_policy` はすべて policy_type = Greedy、use_amp = **true**。`[eval_target].network` は target。`target_policy.network` は存在しない。

根拠: P1。**コピー先がカタログでもカタログ外でも同じ**。選択した policy 部分だけの最終値をコピーする。

#### M09: 同じ設定では部分の指定が全体のベースに勝つ

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eval_base : network = online
DefaultDQNAgent.actor.@greedy : policy_type = Greedy
DefaultDQNAgent.actor.@target : network = target
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.@greedy
DefaultDQNAgent.$ = @baseline
```

期待結果: `[eval_target]`はpolicy.policy_type = Greedy、network = target。eval_targetの選択2行を逆順にしても同じ。両方の定義位置を解決記録へ載せる。

根拠: P2・P7。同じ設定の部分への指定が全体のベースに勝つ。別々の選択元の順位はM19で確認する。

#### M10: 実際の循環

入力:

```text
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.actor.[eval_target].policy
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.[eval].policy
DefaultDQNAgent.$ = @baseline
```

期待結果: `[eval].policy` → `[eval_target].policy` → `[eval].policy` の経路を含むエラーで停止する。空コピーとして成功させない。

根拠: P6。

#### M11: 未定義参照と空の通常prefix

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.$ = @baseline > A3
```

期待結果: 空の A3 を許容し、`[eval].policy.policy_type = Greedy` で正常終了する。次の各分岐は eval の右辺だけを置き換える。

根拠: P6。通常prefixの空と、宣言済み部品を要求する参照の空を区別する。A3をCommonやLibrary.Commonへ名前変更した分岐でも同じ。

| eval の右辺（各行は独立） | 期待結果 |
|---|---|
| `DefaultDQNAgent.actor.@missing` | 未定義プロファイルとして fail-fast |
| `DefaultDQNAgent.actor.[missing]` | 未定義カタログとして fail-fast |
| `DefaultDQNAgent.actor.@eval_base.sync_interval` | 指定部分が空のため fail-fast |

いずれも選択キー、指定 term、解決先をエラーに含める。

#### M12: 10 段の境界

入力:

```text
net.block.[DepthBase].type = ReLU
net.block.[Depth01].$ = net.block.[DepthBase]
net.block.[Depth02].$ = net.block.[Depth01]
net.block.[Depth03].$ = net.block.[Depth02]
net.block.[Depth04].$ = net.block.[Depth03]
net.block.[Depth05].$ = net.block.[Depth04]
net.block.[Depth06].$ = net.block.[Depth05]
net.block.[Depth07].$ = net.block.[Depth06]
net.block.[Depth08].$ = net.block.[Depth07]
net.block.[Depth09].$ = net.block.[Depth08]
net.block.[Depth10].$ = net.block.[Depth09]
```

期待結果: `[Depth01]` から `[Depth10]` まですべて type = ReLU で正常終了する。最長経路は Depth10 の選択（深さ 1）から Depth01 の選択（深さ 10）まで。DepthBase は直書きだけなので深さを増やさない。

失敗分岐は上へ `net.block.[Depth11].$ = net.block.[Depth10]` を追加したもの。11 段になるため上限 10 と経路を含むエラーで停止する。キャッシュや宣言の並べ順によって成功へ変わらない。選択元プロファイル内に別の選択がある場合も、その定義を解決する依存経路として外側からの深さを数える。

根拠: P6。

#### M13: 外側のプロファイルと内側の定義を一緒に切り替える

入力:

```text
DefaultDQNAgent.@baseline : actor.@eval_base.policy.policy_type = EpsilonGreedy
DefaultDQNAgent.@baseline : actor.@eval_base.policy.eps_start = 0.05
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval]
DefaultDQNAgent.@cautious : actor.@eval_base.policy.eps_start = 0.01
DefaultDQNAgent.@unused : actor.[unused].$ = DefaultDQNAgent.actor.@missing
DefaultDQNAgent.$ = @baseline > @cautious
```

期待結果: `[eval]` と `[eval_target]` は EpsilonGreedy、eps_start = 0.01。@cautious をチェーンから除く独立分岐では両方 0.05。未選択の @unused は実効側へ出ず、その中の @missing 参照で停止しない。選択される `@baseline` の中に未選択の内側プロファイルがある分岐(`DefaultDQNAgent.@baseline : actor.@unused.policy.$ = DefaultDQNAgent.actor.@missing` を追加)でも停止しない。未選択の内側定義は休止した在庫として保持し、実際に選択したときだけその定義元を基準に解決する。定義の供給と参照の依存を扱い、@eval_baseの供給がまだ終わっていない途中状態を未定義扱いしない。旧実装の「コピー先で命令を再実行する」を維持条件にはしない。

根拠: P1・P6 と §4.1。左辺に @ が 2 個ある許容例。

#### M14: Run プロファイルが選択チェーンに勝つ

入力:

```text
net.@p : branch.[t].structure = SiLU
Agent.net.$ = net.@p
Other.$ = Agent.net.branch.[t]
run.@arm : Agent.net.branch.[t].structure = ReLU
run.$ = @arm
```

期待結果: `Agent.net.branch.[t].structure = ReLU`、`Other.structure = ReLU`(Runの葉が最終値に含まれ、参照する選択へ届く)。`config_resolution.json`の`overrides`に`by = run.@arm`、`from = SiLU`、`to = ReLU`が載る。

根拠: P1・P4・P7。当初のSiLUへの握り潰しは先行版で修正済みであり、本改訂では非回帰として確認する。

#### M15: Run プロファイルと上書き層・CLI の順位

入力:

```text
Agent.@baseline : learner.gamma = 0.9
A2.learner.gamma = 0.997
Agent.$ = @baseline > A2
run.@arm : Agent.learner.gamma = 0.99
run.$ = @arm
```

期待結果: `Agent.learner.gamma = 0.99`。この入力へ CLI `Agent.learner.gamma=0.95` を与えた独立分岐では 0.95。

`run.$ = @a > @b`で両者が同じ葉キーを持つ分岐では@bが勝つ。`run.@arm : @vars.n = 5`と`key = ${@vars.n}`を持つ分岐では、@varsは実効葉の上書きとは別の解決入力として扱う。これにCLI `@vars.n=7`を渡すと`key = 7`になる。

Runプロファイルが`X.$`を同一キーの入力として書き直す既存の用法も非回帰とする。チェーンだけを供給する独立入力(`Agent.@baseline : learner.gamma = 0.9`、`A2.learner.gamma = 0.997`、`run.@arm : Agent.$ = @baseline > A2`、`run.$ = @arm`)にCLI `Agent.$=@baseline`を渡すと、A2は適用されずgammaは0.9になる。冒頭の入力のように`run.@arm : Agent.learner.gamma = 0.99`もあれば、そのRunの葉指定が勝って0.99になる。CLI `run.$=@other`はRun選択そのものを差し替える。

根拠: P4。

#### M16: 定義元の相対参照と解決記録

```text
Env.@a : $ = @b
Env.@b : value = 10
Other.@b : value = 20
Env.$ = @a
Other.$ = Env.@a
```

期待結果: `Env.value = 10`、`Other.value = 10`。Other.@bは使わない。記録は`Env.$ → Env.@a`、`Env.@a.$ → Env.@b`、`Other.$ → Env.@a`であり、@a自身の継承をOther.$へ置き換えて再記録しない。根拠: P1・P5・P7。

#### M17: 通常prefixのroot選択を継承できる

```text
Common.@base : value = 10
Common.$ = @base
Other.$ = Common
```

期待結果: Common.valueとOther.valueは10。CommonをA2、Library.Common等へ名前変更し、参照も対応させた分岐で同じ規則になる。名前だけによる上書き層エラーを出さない。実際の自己供給・循環の別分岐はエラー。根拠: P1・P6。

#### M18: 同じキーの再指定とチェーンを区別する

```text
Env.@a : left_only = 1
Env.@b : right_only = 2
Env.$ = @a
Env.$ = @b
```

期待結果: Env.right_only = 2で、Env.left_onlyは存在しない。入力として採用するEnv.$は最後の@bだけ。最後の2行を`Env.$ = @a > @b`の1行へ変えた独立分岐では、両方のキーが存在する。根拠: P3。

#### M19: 選択元をまたぐとチェーンの右側が勝つ

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@greedy : policy_type = Greedy
A2.actor.[eval].policy.$ = DefaultDQNAgent.actor.@greedy
A3.actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.$ = A2 > A3
```

期待結果: `[eval].policy.policy_type = EpsilonGreedy`。A2内では部分指定を解決してGreedyとなるが、A3の結果を後で重ねるためEpsilonGreedyが勝つ。`A3 > A2`へ変える分岐ではGreedy。根拠: P1・P2。

## 5. 実装時の責務と今回の作業境界

実装の主対象は[config_impl.cpp](../../core/anet-core/src/config_impl.cpp)と公開ConfigManager経由のテストである。選択元の最終値を、各設定の個別・部分指定より弱いベースとして合成する。依存の計算順と上書き順位を混同しない。

宣言を運ぶ入れ物かどうかの判定、上書き層専用のroot選択除外・禁止、コピー先を基準にした再実行を除去する。旧チェーンの葉を消すための親子由来追跡は追加しない。定義元の相対参照、内側プロファイルの休止と供給、Run・CLIの最終値伝播を同じ契約に収める。

これは責務の指定であり、評価用データ構造や反復方式を必須化するものではない。公開APIやテスト専用APIを増やさず、M01〜M19で説明できる最小の実装を選ぶ。解決記録の順序は決定的にし、キャッシュの有無で内容を変えない。

**今回行うのは設計文書の更新のみ。コード・現用設定・テスト・比較器・goldenはレビュー後の実装で変更する。** 既存の未コミット変更、旧golden、Run artifactを保持する。staging・commit・pushは行わない。

## 6. 受入条件と既存テストの改訂

テストの正本は本改訂の原則・具体例・期待結果とする。先行版のテストが通ることだけを目的に、逆の優先順位を残さない。期待値の一括再採取だけで変更を承認しない。

| 受入ID | 対応例 | 確認する結果 |
|---|---|---|
| A01 | M01〜M03 | 最終値と追加キーが伝播し、親へ逆流しない。外側の差分より参照先の個別指定が強い |
| A02 | M04 | 親CLIの伝播と子自身の個別指定・CLIの優先を区別する |
| A03 | M05、M09 | 個別葉はベースに勝ち、同じ設定の部分指定は全体に勝つ。行順の逆転でも不変 |
| A04 | M06、M18 | 別々の選択元は差分合成で葉を残す。同じ入力キーの再指定は最後だけを採用 |
| A05 | M07、M08 | 多段・部分・カタログ外の継承で最終値を使い、命令を再実行しない |
| A06 | M10、自己供給分岐 | 実際の循環・自己供給を経路付きで拒否し、独立部分の相互参照は許容 |
| A07 | M11、M17 | 未定義部品は拒否、空の通常prefixとCommon/A2のroot継承は許容 |
| A08 | M12 | 深さ10は成功、11は失敗。キャッシュ・宣言順で判定を変えない |
| A09 | M13、M16 | 内側定義の供給と未選択の休止、定義元基準の相対参照 |
| A10 | M08、M19 | カタログ外も同じ規則。選択元をまたぐと具体性より`>`の右側が勝つ |
| A11 | M14 | Runの葉が通常設定に勝ち、参照先の最終値へ伝播 |
| A12 | M15 | Runの項の後勝ち、CLIの最優先、解決入力キーとrun.$の指定 |
| A13 | M16、M14 | 定義位置の記録、run.$先頭、Runの最終差分だけのoverrides、schema_version=1 |
| A14 | 既存入力・値参照テスト | include・同一キー再指定・パーサ、`${}`1段と異常系を維持 |
| A15 | §7の17入力 | 移行後の値一致を必須にし、解決記録・行順の差分を理由付きで確認 |

既存テストの変更点を実装時に個別に記録する。

| 既存例・テスト | 必要な改訂 |
|---|---|
| M02 | 外側A2より参照先自身の個別葉が強い。直接再指定・CLIによる伝播の分岐も保持 |
| M05 / A13 same owner inheritance | 個別指定のownが継承元のbaseに勝つ |
| M06 | eps_endが残る。通常prefixのA2自身の解決結果も存在する |
| M07の直接宣言分岐 | 部分選択が外側のベースに勝つ。個別葉で上書きする分岐は0.01が伝播 |
| M09逆順分岐 | 行順によらず部分指定のGreedyが勝つ |
| A07 overlay root / root owner | 名前による拒否を削除し、通常の成功例と実際の循環例へ分ける |
| 同一ownerの解決記録 | 2件目のkeyをEnv.@a.$へ変更。葉の競合があれば個別優先の期待値も改訂 |
| 残りの記録比較 | コピー先の架空の宣言ではなく、使用した定義位置を期待値とする |

## 7. 現用設定の移行と17件の比較

### 7.1 実効値を維持する最小の移行

新規則は既定値の意味も変える。例示だけで移行範囲を決めず、§7.2の17入力について、固定configの個別葉と旧goldenを照合する。以下はRun・CLIによる同一キーの明示指定を考慮した静的棚卸しであり、新resolverを実行した検証結果ではない。対象24キーの個別葉は、2026-09-13の現用設定とmanifestの固定configで同一だった。

#### backend: 性能・再現性に関わる共有既定値

[common.txt](../../apps/runner/config/common.txt)の共有既定値と、Atari / DropMerge / ImageCls / LunarLanderの通常選択が衝突する。

```text
backend.cudnn_benchmark = false
backend.deterministic_algorithms = true
backend.$ = backend.@non-deterministic
```

旧実効値はcudnn_benchmark = true、deterministic_algorithms = falseだが、移行せず新規則を適用すると直書きが勝ち、それぞれfalse / trueになる。高速化を選んだRunで決定化が有効になり、性能だけでなくSDPA等の数値挙動・再現性にも影響する。[Atari.txt](../../apps/runner/config/Atari.txt)には同seed再現の設定について「+11%コスト」と記録されている。この値は当時の測定条件に依存し、今回再測定した値ではない。

[12_batch_run.bat](../../apps/12_batch_run.bat)の`BK`は`backend.$=backend.@non-deterministic`という**選択キー**のCLI指定である。P4で最優先になるのはbackend.$自体であり、その継承結果が別キーの個別葉より強くなるわけではない。したがって、**現行batのBK指定ではこの競合を回避できない**。葉そのものへのCLI指定は別だが、batへ対症的な葉指定を増やすことを移行方法にはしない。

さらに`lunarlander-repro`では、`run.@repro`が選ぶbackend.@deterministicのcudnn_deterministic = trueが、common.txtの個別葉falseに負ける。17入力を保持するにはbackendの移行対象は次の3キーになる。

```text
# common.txt: 3個の個別葉をこの通常プロファイルへ移し、未指定時のベースにする
backend.@defaults : cudnn_benchmark = false
backend.@defaults : deterministic_algorithms = true
backend.@defaults : cudnn_deterministic = false
backend.$ = backend.@defaults
```

高速化を選ぶ設定のチェーンは、次の形で既定値より強くする。

```text
backend.$ = backend.@defaults > backend.@non-deterministic
```

決定論設定を選ぶチェーンも同じ関係にする。commonのベースを残すことで、backend.$を指定しないGridMaze / GridMaze_muzero / CartPoleも従来値を保つ。backend.@non-deterministicとbackend.@deterministicはこの3キーをすべて持つため、既存CLIがどちらかを単独選択する場合も値を保てることを確認する。Runプロファイルが供給するbackend.$も移行漏れの検索対象とする。

#### Atari: ゲームとIQN出力先

現在のAtari設定の抜粋は次のとおり。

```text
AtariEnv.$ = AtariEnv.@v5_noop30 > E1
E1.game = breakout
AtariEnv.game = pong
```

先行版ではbreakout、新規則のままでは個別指定が勝ってpongになる。pongを弱い既定値として維持する移行例は次の形。

```text
AtariEnv.@defaults : game = pong
AtariEnv.$ = @defaults > @v5_noop30 > E1
E1.game = breakout
```

移行後もbreakoutを維持する。AtariのIQN出力先でも、直接書かれた`DefaultDQNAgent.net.body.output.[features] = main_feature`と、選択するiqn_fusionが衝突する。これらを含め、新仕様で実効値が変わる既定値だけをベース側へ移す。`@defaults`は既存のプロファイル記法による通常名であり、予約名にはしない。

#### 17入力の移行対象一覧

**A15の文字列値一致で数えると24種。** レビューで挙がった22種に、数値として同じでも表記が異なる`DropMergeEnv.no_drop_timeout_gameover_penalty`と、派生入力lunarlander-reproの`backend.cudnn_deterministic`を加える。DropMergeは挙動に関わる14種と表記差1種の計15キーであり、`no_drop_timeout_*`は下表の2キーへ分けて数える。

「旧golden」は移行後も保持する値、「移行なし」は個別葉が勝った場合の値である。キーの種類数は入力をまたいで重複排除し、env別の件数は各envに属するmanifest入力の和集合とする。

| キー | 旧golden | 移行なし | 影響する入力 |
|---|---|---|---|
| `backend.cudnn_benchmark` | `true` | `false` | Atari全6、DropMerge全3、ImageCls全3、lunarlander-default |
| `backend.deterministic_algorithms` | `false` | `true` | 同上 |
| `backend.cudnn_deterministic` | `true` | `false` | lunarlander-repro |
| `AtariEnv.game` | `breakout` | `pong` | Atari全6 |
| `DefaultDQNAgent.net.body.output.[features]` | `iqn_fusion` | `main_feature` | atari-1〜4、atari-6。atari-5はもともとmain_feature |
| `DropMergeEnv.grid_cols` | `58` | `40` | DropMerge全3 |
| `DropMergeEnv.grid_rows` | `46` | `64` | DropMerge全3 |
| `DropMergeEnv.action_mode` | `direct_noop` | `move_fast` | DropMerge全3 |
| `DropMergeEnv.damping` | `1.0` | `0.5` | DropMerge全3 |
| `DropMergeEnv.friction` | `0.3` | `0.1` | DropMerge全3 |
| `DropMergeEnv.fruit_scores` | 末尾`1000.0` | 末尾`820` | DropMerge全3。先頭11要素は同じ |
| `DropMergeEnv.game_over_penalty` | `-10.0` | `0` | DropMerge全3 |
| `DropMergeEnv.no_drop_timeout_steps` | `100` | `200` | DropMerge全3 |
| `DropMergeEnv.no_drop_timeout_gameover_penalty` | `-10` | `-10.0` | DropMerge全3。数値は同じだが文字列表記が異なる |
| `DropMergeEnv.restitution` | `0.1` | `0.05` | DropMerge全3 |
| `DropMergeEnv.settle_velocity_threshold` | `0.5` | `0.1` | DropMerge全3 |
| `DropMergeEnv.time_penalty` | `0.0` | `-0.0001` | DropMerge全3 |
| `DropMergeEnv.use_fast_move` | `true` | `false` | DropMerge全3 |
| `DropMergeEnv.use_instant_drop` | `true` | `false` | DropMerge全3 |
| `DropMergeEnv.use_no_legal_adjudication` | `true` | `false` | DropMerge全3 |
| `LunarLanderEnv.ground_y` | `0.0` | `0.5` | LunarLander全2 |
| `LunarLanderEnv.landing_detection_mode` | `not_awake` | `contact` | LunarLander全2 |
| `LunarLanderEnv.turbulence_power` | `0.5` | `1.5` | LunarLander全2 |
| `LunarLanderEnv.wind_power` | `3.0` | `15.0` | LunarLander全2 |

| Env設定 | 影響するキーの種類数 | 内訳 |
|---|---|---|
| Atari | 4 | backend 2、game 1、IQN出力先1。atari-5単体は3 |
| DropMerge | 17 | backend 2、Envの値変化14、Envの表記差1 |
| LunarLander | 7 | backend 3、Env 4。default単体は6、repro単体は5 |
| ImageCls | 2 | backend 2 |
| GridMaze / GridMaze_muzero / CartPole | 各0 | この入力群では移行前後で値が変わる個別葉なし |

レビュー時の現用7envの棚卸しでは、「全体↔部分」の値が競合する箇所は0件だった。この報告は現用入力の移行量の見積もりであり、P2の部分指定優先を検証した結果ではない。M09・M19の専用例と17入力の実装後比較は引き続き必要とする。

#### DropMerge・LunarLanderの移行量と配置

[DropMerge.txt](../../apps/runner/config/DropMerge.txt)は、選択する@baseline / @G5846 / @heavy / E1の値と、素のDropMergeEnv.*が重複している。必要な変更は、上表の**15個の個別葉の定義位置をベースプロファイルへ移し、既存チェーンの先頭にそのプロファイルを加えること**。内訳は値変化14個と表記差1個である。元の既定値の文字列とコメントを保持し、既存プロファイルやE1が供給する実験値は書き換えない。

```text
DropMergeEnv.$ = @defaults > @baseline > @G5846 > @heavy > E1
```

LunarLanderも上表のEnvの4個の個別葉をベースへ移し、`@defaults > @trunk > E1`とする。キー削除で差分を隠す、`-10`を`-10.0`へgolden側で正規化する、同値で衝突しない別の個別葉まで一括移動する、といった変更は行わない。この15＋4個の移動は実効値を保つための移行そのものであり、「無関係な整形をしない」方針と両立する。

全既定値の一括移動、無関係な並べ替え、後方互換モードによる旧順位の復活は行わない。既存のユーザー変更を保持して移行し、現用設定と比較用の固定入力には対応する同じ移行を適用する。

### 7.2 比較条件

比較入力の正本は[manifest.json](../../core/anet-core/testdata/prd072/manifest.json)。旧goldenは[testdata/prd072/baseline](../../core/anet-core/testdata/prd072/baseline/)に保持し、commit `107a62c8ae01cb758f3cd49d98e8424386160e5d`の固定config、CLI、注入値との対応を残す。

| Env設定 | 比較入力数と内容 |
|---|---|
| Atari | 6。manifestのRunチェーンをそのまま使う |
| DropMerge | 3。既定、iqn32_stratified、qr51_control |
| LunarLander | 2。既定、repro |
| ImageCls | 3。既定、resnet18ish_hr、convnext_atto_hr |
| GridMaze / GridMaze_muzero / CartPole | 各1。既定入力 |

- **値**: 移行前goldenと、対応する移行後入力の全キー・値の一致を17件すべてで必須とする。診断用の非@prefixも含め、比較から都合の悪いキーを除外しない。
- **移行の重点確認**: §7.1の24キーを入力別に照合する。backendの通常選択・既存batと同じCLI選択・Runのrepro選択、DropMergeの14個の値変化と1個の表記差を含める。24キーだけへの比較縮小は行わず、全キー一致と、変更0件だった3envの値保持も確認する。
- **解決記録**: 定義位置と移行で増減する選択を新仕様の期待値と比較する。旧記録との違いは入力ごとに理由を示す。`references`・`overrides`にも差分があれば同様に確認し、既存の空overrides確認を黙って外さない。
- **順序**: `Map().Order()`を前後比較して差分位置をレポートする。完全一致は必須にしないが、Runのdumpを人間が比較する用途で読みづらくなっていないか確認する。
- **採取証拠**: 旧goldenのハッシュ・採取条件・元ファイルを保持する。差分を消すために一括再採取しない。既存goldenを人間が削除してから明示captureする運用と、比較器の上書き拒否・manifest整合性は維持する。

レビュー後の実装では、固定入力への移行内容と新しい記録の期待値を再実行可能な形で残し、比較器・AGENTS.mdの手順を同じ変更で更新する。現用設定の追加変更を旧goldenへ混ぜず、今回の移行差分を区別する。

VsDevCmd経由の通常Debugビルド、設定テスト全体、17入力の比較、capture拒否、旧判定の識別子検索、`git diff --check`、UTF-8 / LFを検証する。現在は文書レビュー段階であり、これらの実装検証は未実施である。

## 8. 検討経緯と簡素化の判断

2026-09-12版は、チェーン差し替えで旧選択の葉を消すために宣言の扱いを分けた。2026-09-13の再確認では「継承は差分適用」「$はベース、個別指定で上書き可能」を採用し、その区別の前提を改めた。既存テストの結果を優先して、意図と逆の仕様を残さない。

| 検討対象 | 判断 | 理由 |
|---|---|---|
| 上書き層の予約名・ドット数判定・明示登録 | cut | 通常prefixの差分合成で扱える。名前だけで意味を変えない |
| 旧選択の生成物を削除するための由来追跡 | cut | 別々の選択元に由来する葉は、右側にない限り残す |
| 全設定の最終値参照 | keep | 後段変更を継承先へ届ける実在の要求 |
| 定義元の相対参照と記録 | keep | 同じプロファイルが使用先によって別の設定を指す非対称をなくす |
| 既定値の移行 | shrink | AtariのゲームとIQN配線等の衝突に限定し、無関係な整形をしない |
| 互換モード・新しい構文・削除演算子 | cut | 合意した差分合成に不要 |
| 固定入力・旧golden・差分レポート | keep | 17件の実験条件を変えないことと、診断変更を機械的に確認する |
| 実装前の文書レビュー | keep | 本更新だけで仕様・期待値をレビュー可能にする。実装は別の開始指示後 |

成功は「上書き層を識別するコードが不要」「個別・部分指定の行順で結果が変わらない」「移行後17件の値一致」で測る。過去の検証は20implへ参照を残し、本改訂の未実施検証と混同しない。

## 9. 今回扱わないこと

- Actor APIや設定カタログへの移行(PRD 061)、NNブロックの未知キー検証など消費側の契約。
- 上書き層の運用上の命名・段数の再設計、未使用キーの自動削除。
- 過去のRun artifactの書き換え、互換モード、staging・commit・push。
