# PRD 072: 設定選択の最終値参照と Run プロファイルの優先順位

- 起票日: 2026-09-07。改訂日: 2026-09-12(同日 Claude レビューと反対 2 件の裁定を反映: §1-(b)、§2、P1〜P7、§4.1、M04、M14、§5〜§9)。
- 関連: [PRD 061](061_eval_slot_policy_override_10prd.md)、[ADR 0038](../adr/0038-actor-config-catalog-without-runmode.md)、[設定基盤の現行設計](../design/100_runtime_and_configuration.jp.md)、[用語集](../../CONTEXT.md)。
- 本 PRD は `999_config_run_profile_override_precedence_10prd.md` を吸収する。同文書は本 PRD 成立をもって廃止する。
- 本改訂は仕様とマニュアル草稿の確定まで。リゾルバの実装・検証は後続作業、Actor のコード・設定移行は PRD 061 とする。以下の新契約は実装済みを意味しない。

## 1. 背景とゴール

設定解決には、上書きが黙って効かなくなる経路が 2 つある。どちらも fail-fast せず、警告も出ず、解決後の dump を人間が読むまで気づけない。

**(a) 選択が古いスナップショットを写す。** `ResolveSelection` は宣言時点の配下キーをコピーする。`DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target` と書くと、後段の A2 / A3 が `[eval]` へ加えた変更は `[eval_target]` へ届かない。PRD 061 の `[eval]` / `[eval_target]` はこの形なので、差分を二重に書くことになる。

**(b) Run プロファイルの葉キーが選択チェーンに握り潰される。** `ExpandNamedTrunk` が展開した葉は直書きと同じ tier で `effective_map_` へ写り、その後の `ApplyTerm` が同じキーを書くと負ける(§2)。選択チェーンが同じキーを持つと Run プロファイルの指定が消える。2026-08-31 に `run.@head_relu` の 3 本中 1 本が無効化され、2026-09-12 に `run.@btrnet` で同じことが再発した。どちらも `net.@iqn` 所有の `[tau_embedding].structure` を踏んでいる。

**ゴールは、`X.$ = Y > Z` を「X は Y に Z の差分を重ねたもの」と読めるようにし、上書きの成否を書いた位置から予測できるようにすること。** 覚える規則を 1 文に減らし、`[key]` の有無で挙動が変わる特別扱いを作らない。

追加しない: 新しい設定モード、互換スイッチ、policy 専用カタログ、Actor 移行、`+=` 相当の差分演算子。

## 2. 現行動作と非対称

[config_impl.cpp](../../core/anet-core/src/config_impl.cpp) の実効適用順序。

| # | 工程(実行順) | 書き込み先 | 実装 |
|---|---|---|---|
| 1 | CLI 第 1 相(全 CLI キーを解決入力へ注入) | `working_map_` | `:25-27` |
| 2 | Run プロファイル展開(`run.$`) | `working_map_` | `:33`, `:186` |
| 3 | デフォルト直書き / include のスナップショット(**Run プロファイルの葉を含む**) | `working_map_` → `effective_map_` | `:38-45` |
| 4 | 選択チェーン展開(プロファイル + 上書き層) | `working_map_` と `effective_map_` | `:315`, `:322` |
| 5 | CLI 第 2 相(`IsResolverInputKey` でない CLI キーの再適用) | `effective_map_` | `:60-64` |
| 6 | 値参照 `${}` 展開 | `effective_map_` | `:67` |

非対称は 2 点。

- **工程 4 は工程 3 を上書きする。** Run プロファイルの葉は工程 2 で `working_map_` に入り、工程 3 で直書きと同じ tier として `effective_map_` へ写る。その後の選択(工程 4)が同じキーを書けば負ける。`ExpandNamedTrunk` が `effective_map_` に書かないこと自体が原因ではなく、Run プロファイルの葉が直書き tier に畳まれていることが原因である。[CONTEXT.md](../../CONTEXT.md) の「Run プロファイルは上のラダーや上書き層より強い」と逆になっている
- **工程 4 のコピーは、その時点の値を写す。** 後段の項が参照先を変えても反映されない

CLI だけが 2 相を持ち、葉については最優先になっている。解決入力キー(`.$`・`@`)は工程 1 の注入を工程 2 の trunk 展開が上書きするため、`@vars.*` や `backend.$` では Run プロファイルが CLI に勝つ(P4 で直す)。また CLI の葉が継承先へ届くのは、参照先がプロファイルや葉だけの prefix で、工程 1 で注入した値を誰も上書きしない場合に限る。組み立て済みの prefix(`[eval]` 等)では工程 4 のコピーが工程 1 の値を `working_map_` 上で上書きし、工程 5 は `effective_map_` だけを戻すので、継承先には届かない(M04 の 2 行目が要求する挙動は現行では成立しない)。

## 3. 確定契約

以下の原則番号をマニュアル例・受入条件から参照する。

### P1: 選択は参照先の最終値を読む

`X.$ = Y > Z` は「X は Y に Z の差分を重ねたもの」を意味する。**最終値とは、選択段・Run プロファイル第 2 相・CLI 第 2 相を全て適用した後の値とキー集合**である。Y / Z へ後段の選択・上書き層・Run プロファイル・CLI が加えた変更は X へ届く。第 2 相の葉はそのキーへの最優先の書き込みとして最終値に含まれ、参照する選択へ届く(M04 の 2 行目、M14)。書き込み位置は選択の宣言位置のまま(P2)なので、継承先の後段指定は元の順位を保つ。既存キーの値だけでなく、後から増えたキー、チェーン差し替えで消えたキーも反映する。

**参照先が `@` プロファイルでも `[key]` カタログでも通常の prefix でも同じ。** コピー先の種類でも区別しない。`[key]` は「名前で参照される項目」という identity の宣言であって、選択の読み方を変えない。

値は `working_map_` の最終状態から読む(`@` 配下の葉を含む。M02 の `A2 : actor.@eval_base.policy.eps_start` はこれで届く)。参照先に含まれる `.$` を写すかどうかは、**参照先が宣言の袋か組み立て済みノードか**で決まる。宣言の袋は 2 種類で、プロファイル(`@` セグメントを持つ prefix)と上書き層(チェーンの term に置かれる root 直下の単一セグメント名。CONTEXT の A1〜A3 / E1 / M1 / M2 / P1)である。袋の配下の `.$` は宣言としてコピー先へ相対的に写し、コピー先で実体化する(M06 の `A3 : actor.[eval_target].$`、`DefaultDQNAgent.@iqn : net.$ = net.@iqn`)。上書き層の配下の `.$` は root 選択として在処では解決しない。在処で解決すると、組み立てた葉(`A2.actor.[eval].policy.eps_end`)が層の葉として別経路で写り、チェーンを差し替えても残る(M06 の上書き層差し替え分岐)。同じ単一セグメント名を term と root 選択の owner の両方に使う形(`A2.$ = …`、`DefaultDQNAgent` を term に置く)は fail-fast にする(P6)。袋のうち、参照先そのものの `.$`(suffix が `.$` だけ)を継承として実体化するのはプロファイルだけとし(`Env.@a.$ = @b`)、上書き層の root `.$`(`A2.$`)は fail-fast にする(P6。現用設定に無い)。それ以外の非 `@` の参照先(`DefaultDQNAgent.actor.[eval]`、`app.online`、`net.block.[Linear_120]` など複数セグメントの prefix)は組み立て済みノードで、その配下の `.$` は root で宣言したものでも写さない。値だけを最終値で写す(M07 と、その直接宣言の分岐)。`.$` の由来(宣言由来かコピー生成か)だけでは、未展開の定義と反映済みの命令を区別できない。写した `.$` は、コピー先が実効側なら nested 選択として解決して `working_map_` には残さず(現行 `:315` の変更)、コピー先が `@` を含む(プロファイルの中へ写す)なら宣言として保存し解決しない(現行 `:318` の dormant 扱い。M13 の内側プロファイルの分岐)。単一セグメントの owner そのもの(`DefaultDQNAgent` 等)を参照先に置くと上書き層として扱われる。現用設定にその形は無い。

これは選択の値とキー集合の契約であり、型付き Config による既定値補完を含まない。`${}` は最終段で解決し、ここで多段値参照へ拡張しない。

### P2: 書き込みは宣言位置に置き、後から書いたものが勝つ

選択が生成した結果は、その選択を宣言した位置へ書く。読みが最終値でも、書き込み位置は動かさない。したがって後段の直書き・上書き層・CLI は、前段の選択結果へ上書きできる。

- 実効側のデフォルト直書きは選択より弱い。各 term が持つ直書きも、その term が生成する選択より先に反映する
- チェーンは左から右へ適用し、右が後勝ち。root 選択は宣言順、nested 選択はそれを生成した term の位置を保つ
- 全体選択と部分選択が重なるときも適用順で決める。「深いパスだから強い」という例外は設けない
- A1 / A2 / A3 は用途上の名前であり、数字自体に優先順位はない。チェーン内の位置が順位を決める
- 依存先を先に計算しても、書き込み順位を計算順へ置き換えない
- term のプロファイル自身の `.$`(継承。P3)も「その term が生成する選択」であり、term の葉の直後に書く。基底側が派生プロファイル自身の葉より後に書くことになる(§4.1)

### P3: 同じコピー先への複数の `.$` は最後のチェーンへ差し替える

term のコピーで現れた `.$` は 2 種類に分かれる。判定は現行の機構そのもの(`ApplyTerm` の `target_key`、[config_impl.cpp:313-319](../../core/anet-core/src/config_impl.cpp))である。

- **プロファイル自身の継承**: 写した `.$` の `target_key` が解決中の選択キーと同じ(suffix が `.$` だけ。`Env.@a.$ = @b` を `Env.$ = @a` が写す形、[config_test.cpp:939](../../core/anet-core/src/config_test.cpp))。その term の一部として、term の葉の直後に適用する。差し替えの対象にしない。相対 term はコピー先 owner を基準に解決する(現行どおり)
- **sub-prefix への宣言**: `target_key` が `owner.sub.$` の形(`@baseline` の `actor.[eval_target].$`、`@iqn` の `net.$`)。root 宣言と、外側の選択の各 term から届くものを合わせて、**最後に適用されるものだけを、その適用位置で採用する**。右辺がプロファイル、カタログ、その混合のどれでも同じ。以前のチェーンだけが供給していた値とキーは残さない

項目自身への独立した直書き・別の配下選択まで削除しない。それぞれ元の適用位置を保持し、採用チェーンと重なるキーは P2 の優先順位で決める。以前のチェーン内でだけ生成された nested 選択は、独立した指定には含めない。

**組み立て済みノードの `.$` は継承先の選択命令として持ち込まない**(P1。M07 の `[eval_target]` と、`[eval]` に root で宣言した `policy.$` の分岐)。宣言の袋(プロファイル・上書き層)の `.$` は写して実体化する(M06 の A3、M09 の @baseline)。継承先自身に宣言した `.$` は別の指定であり、通常の適用順に従う。上書き層の宣言 `.$`(`A3.actor.[eval_target].$`)は在処では解決せず、コピー先で実体化した選択だけが存在する。層の中に組み立てた葉は生まれないので、差し替え後に旧チェーンの値が別経路で残らない。層に直接書いた葉(`A2 : actor.[eval].policy.eps_end = 0.05`)は独立した直書きとして写り、差し替えでは消えない(現用設定に上書き層内の `.$` は無い)。

### P4: 優先順位は 4 段

```
デフォルト直書き・include  <  選択チェーン・上書き層  <  Run プロファイル  <  CLI
```

Run プロファイルに CLI と同じ第 2 相を与える。工程 4(選択チェーン展開)の後、CLI 第 2 相の前に、Run プロファイルの葉キーを `effective_map_` へ再適用する。

**解決入力キー(`.$`・`@` 素材)でも CLI が Run プロファイルに勝つ。** 現行は CLI 第 1 相(工程 1)の後に trunk 展開(工程 2)が同じキーを無条件に上書きするため、CLI の `@vars.max_exp_step=…` や `backend.$=…` が `run.@a5` / `run.@repro` の指定に負ける。trunk 展開の後に CLI の解決入力キー(`run.$` を除く)を再注入し、4 段の順位を全キーで成立させる(M15 の分岐)。`run.$` は trunk そのものなので現行どおり展開前に効く。

- 第 1 相は残す。Run プロファイルが `X.$` を差し替えて選択を駆動する用法は第 1 相が支える
- 第 2 相の対象は `IsResolverInputKey` が false のキーのみ（`.$` で終わらず `@` セグメントを含まない）。CLI 第 2 相と同一判定を共有する。`@vars.*` は除外され、値参照の入力として `working_map_` に残る
- `run.$` の項を左→右で畳み、キーごとの最終値を確定してから 1 回だけ `effective_map_` へ適用する。途中の書き戻しは適用も記録もしない(Atari の `run.@a5 : A2.learner.replay_ratio = 1` → `run.@rr4 : … = 4` は 4 だけを見る)。`MapType` は `anet::OrderedMap` なので宣言順が保たれ決定的
- 入れ子は辿らない。Run プロファイルが持つ自身のリテラル葉キーだけを再適用する
- 再適用した葉は最終値の一部であり、P1 により参照する選択へ届く(M14)。伝播のためにコピー辺を辿る後処理や選択の再実行は要らず、評価器の入力に第 2 相の葉を最上位の書き込みとして含めるだけでよい。「入れ子は辿らない」は再適用する Run 側キーの範囲(自身のリテラル葉)の話で、伝播とは別

### P5: CLI・`run.$`・値参照

CLI の解決入力への注入と実効 leaf の最終上書きは維持し、解決入力キーは trunk 展開の後に再注入して Run プロファイルの同じキーに勝たせる(P4)。参照先を CLI で変更すると、第 1 相のキー(`@…` プロファイルや `A3.…` 上書き層)でも組み立て済み prefix への第 2 相のキー(M04 の 2 行目)でも P1 により継承先へ届き、継承先の実効キーへの CLI はその継承結果にも勝つ。CLI の `.$` 指定も、採用するチェーンを決める入力となる。

`run.$` は通常選択より先に Run プロファイルを展開する現行契約を維持する。`${full.key}` は選択と CLI の後で値を 1 段参照する。未定義・連鎖・未解決の値参照は既存どおり fail-fast とする。

### P6: 必要な部分の依存関係を検証する

循環は実際に参照する部分の依存関係で判定する。名前が相互に現れるだけでは循環ではない。同じ値の解決が自身へ戻る参照は経路付きで fail-fast にする。

**自己供給の検出**: 選択の書き込み範囲(`owner + suffix`)が、同じ選択のどの term の source prefix とも交わってはならない。判定は P1 の最終キー集合で行い、交わったら経路付きで fail-fast にする。`X.part.$ = X`(全キーが自分の source 配下へ落ちる)と `X.$ = X`(現行の term 循環検出でも止まる)はこれで止まる。`X.$ = X.part` は `X.part.part.*` が無い限り交わらず正常で、実設定の 9 件(`app.$ = app.online` が 7 env。CartPole 以外は `> P1` 付き。GridMaze_muzero の `MuZeroAgent.$ = MuZeroAgent.baseline > A1` と `metrics.scalar.$ = metrics.scalar.muzero > M1`)はすべて書き戻し 0 件である。owner と source の包含関係そのものは禁止しない。

選択の深さ上限は 10 を維持する。深さは**依存グラフ上の最長経路長**で数える。起点の選択を 1 とし、nested 選択または最終値を求めて辿る選択ごとに 1 増える。循環が fail-fast なので依存グラフは DAG であり、ノードごとに「1 + 子の最大値」を memo する DP で順序非依存に求まる。visited を刈る DFS は最初に到達した深さを返すので使わない。兄弟 term の数、パスのセグメント数、反復周回数ではない。最終値をキャッシュしても長い依存経路の検証を省略しない。`run.$` の展開は選択より前の別工程であり、深さに数えない。

採用された参照先を解決した結果が空の場合、プロファイル・カタログ項目（その部分参照を含む）は未定義参照として fail-fast にする。まだ定義を展開していないだけの途中状態で未定義と決めない。空の通常上書き層は許容する。未選択プロファイルの在庫は有効化・参照しない。上書き層(term に置かれる単一セグメント名)が自分自身のチェーン(`A2.$`)を持つ場合と、root 選択の owner(`DefaultDQNAgent`)が term に置かれた場合は fail-fast にする(上書き層は在処で解決しないため、その名前の root 選択は適用されない)。エラーには選択キー、指定 term、解決先、循環／深さの場合は経路と上限を含める。

### P7: 採用結果を記録する

`config_resolution.json` の `selections` は、**P3 が適用したチェーンを、その適用順に載せる**。P3 で不採用になった宣言は載せない。プロファイル自身の継承(P3 の第 1 種)は適用されるので、root の記録と同じ `key` で続けて載る([config_test.cpp:939](../../core/anet-core/src/config_test.cpp) の 2 件。`key` はどちらもコピー先 `Env.$`)。実効側 prefix の `.$` は写さないので、継承元の選択を継承先で再実行したような記録は生じない。

Run プロファイル第 2 相で**第 2 相の最終値が第 2 相前の実効値と異なる**キーだけを `overrides` として記録する。項ごとの途中の書き戻し(4 → 1 → 4)は記録しない。

```json
"overrides": [
  { "key": "...structure", "by": "run.@head_relu", "from": "... > SiLU", "to": "... > ReLU" }
]
```

事後 dump から上書きの成否を機械的に確認できるようにする。`overrides` は任意フィールドとして足し、`schema_version` は 1 のまま据え置く(消費側 [inspect_run.py:42](../../viewers/metrics-tools/inspect_run.py) / `:1965`、[config_test.cpp:679](../../core/anet-core/src/config_test.cpp) / `:711`、`docs/design/100` の変更が不要になる。上げる場合はこの 3 箇所を同時に更新する)。不採用履歴や採用状態を示すフィールドはこれ以上追加しない。

## 4. ユーザーマニュアル草稿

### 4.1 読む順序と記法

覚える規則は 2 つだけ。

> **読み**: `X.$ = Y > Z` は「X は Y に Z の差分を重ねたもの」。Y / Z が後で変われば(Run プロファイルや CLI の第 2 相で変えた場合も含めて)X も変わる。
> **書き**: 後から書いたものが勝つ。強さは 直書き < 選択・上書き層 < Run プロファイル < CLI。

プロファイル自身の `.$`(`X.@a.$ = @b`)は「a は b を土台にする」であり、term の葉の直後に適用されるので b の値が a 自身の葉に勝つ(P2・P3)。「a = b + 差分」を書きたいときは差分を別 term に置く(`X.$ = @b > @a_diff`)。実設定にこの形は無い。

以下の Actor の例は **PRD 061 導入後の仕様**であり、リゾルバへ渡す設定の例である。現行 Runner でそのまま学習できる完全な設定ファイルを示すものではない。例内の値は resolver 出力を指し、型付き Config の既定補完前である。

| 記法・要素 | 役割 |
|---|---|
| `前提となる対象・プロファイル : key = value` | 左で対象やプロファイルを示し、右で具体的な Key-Value を書く。左辺の `:` は `.` と同義の見た目上の区切りであり、新しい scope や優先順位ではない |
| `DefaultDQNAgent.actor.@eval_base` | 設定を共通化するプロファイル。実効側へ組み込む前の定義 |
| `DefaultDQNAgent.actor.[eval]` | 実効側に残り、コードから名前で参照するカタログ項目。**選択の読み方は `@` と同じ** |
| `.$ = source > difference` | 選択チェーン。右が後勝ち |
| `@baseline` | 選択元を owner 相対で記述する形。`DefaultDQNAgent : $ = @baseline` なら `DefaultDQNAgent.@baseline` |
| `A2` / `A3` | チェーン中で適用する通常上書き層。空でもよい |
| `run.$` / include / CLI / `${full.key}` | Run の選択束の展開／宣言入力の取り込み／明示上書き／単一値の参照 |

`:` は左辺に 1 個だけ置ける既存構文であり、先頭・末尾には置かない。**左辺の `@` は原則 1 個を推奨する。** 外側の選択で内側の定義も一緒に切り替える必要がある場合は入れ子を許容する。禁止・新しい WARN・構文制約にはしない。

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

#### M02: 参照先への後段変更が届く

入力: `BASE` に追加。

```text
A2 : actor.@eval_base.policy.eps_start = 0.01
A2 : actor.@eval_base.policy.eps_end = 0.01
```

期待結果: `@eval_base` の最終値、`[eval]`、`[eval_target]` の eps_start / eps_end はすべて 0.01。network はそれぞれ online / target のまま。

根拠: P1。選択は @baseline 適用途中の古い値ではなく、A2 を含む最終値を読む。

#### M03: 後段で増えたキーも届く

入力: `BASE` に追加。

```text
A2 : actor.[eval].policy.eps_start = 0.01
A2 : actor.[eval].policy.use_amp = true
```

期待結果: `[eval]` と `[eval_target]` の eps_start はともに 0.01、両方に `policy.use_amp = true` が存在する。`@eval_base` 自体の eps_start は 0.05 のままで、use_amp は追加されない。

根拠: P1。変更は継承元プロファイルへ逆流しない。

#### M04: 継承先だけの後段変更と CLI の範囲

入力: `BASE` に追加。

```text
A3 : actor.[eval_target].policy.eps_start = 0.02
```

期待結果: CLI なしでは `[eval].policy.eps_start = 0.05`、`[eval_target].policy.eps_start = 0.02`。以下の CLI 分岐はそれぞれ独立してこの入力へ適用する。

根拠: P1・P2・P5。継承は土台を配り、後段の直接指定がその上に乗る。

| CLI override（各行は独立） | eval の eps_start | eval_target の eps_start |
|---|---|---|
| `DefaultDQNAgent.actor.@eval_base.policy.eps_start=0.03` | 0.03 | 0.02（A3 が子を上書き） |
| `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` | 0.04 | 0.02（最終値は届くが A3 が勝つ） |
| `A3.actor.[eval_target].policy.eps_start=0.06` | 0.05 | 0.06 |
| `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` | 0.05 | 0.09 |

親からの CLI 伝播そのものは、A3 の 1 行を除いた分岐でも検証する。その分岐で `DefaultDQNAgent.actor.[eval].policy.eps_start=0.04` を渡すと両方 0.04、さらに `DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.09` を同時に渡すと親 0.04・子 0.09 となる。

2 行目と 4 行目は第 2 相のキーであり、現行実装では継承先へ届かない(§2)。P1 で届くようにする。

#### M05: 同じ term 内では直書きより選択が勝つ

入力: `BASE` に追加。

```text
DefaultDQNAgent.@baseline : actor.[eval].policy.eps_start = 0.01
DefaultDQNAgent : actor.[eval_target].policy.eps_start = 0.02
```

期待結果: `[eval].policy.eps_start` と `[eval_target].policy.eps_start` はともに 0.05。@baseline 内の leaf を同プロファイル内の選択行より前へ移しても同じ。

根拠: P2。実効側の直書きと同じ term 内の直書きは選択より弱い。子だけ変えるには M04 のように後段の層へ書く。

#### M06: 最後のチェーンへ差し替える

入力: `BASE` に追加。

```text
DefaultDQNAgent.actor.@greedy : policy.policy_type = Greedy
DefaultDQNAgent.actor.@sync : value = 400
A2 : actor.[eval_target].clone_model = true
A2 : actor.[eval_target].policy.eps_start = 0.02
A2 : actor.[eval_target].sync_interval.$ = DefaultDQNAgent.actor.@sync
A3 : actor.[eval_target].$ = DefaultDQNAgent.actor.@greedy > DefaultDQNAgent.actor.@target
```

期待結果: `[eval_target]` は policy.policy_type = Greedy、network = target、clone_model = true、policy.eps_start = 0.02、sync_interval.value = 400。旧チェーンだけが供給した policy.eps_end は**存在しない**（0 ではない）。親 eval の eps_start / eps_end は 0.05 のまま。記録には A3 の採用チェーンだけが載り、旧 eval 継承チェーンは載らない。子自身が A2 で指定した sync_interval の選択は独立して残る。

右辺をカタログへ置き換えた独立分岐（`A3 : actor.[eval_target].$ = DefaultDQNAgent.actor.[greedy] > DefaultDQNAgent.actor.@target` と `DefaultDQNAgent.@baseline : actor.[greedy].policy.policy_type = Greedy`）でも結果は同じ。

上書き層差し替え分岐(独立入力):

```text
DefaultDQNAgent.actor.@eps : policy.policy_type = EpsilonGreedy
DefaultDQNAgent.actor.@eps : policy.eps_end = 0.05
DefaultDQNAgent.actor.@greedy : policy.policy_type = Greedy
A2 : actor.[eval].$ = DefaultDQNAgent.actor.@eps
A3 : actor.[eval].$ = DefaultDQNAgent.actor.@greedy
DefaultDQNAgent : $ = A2 > A3
```

期待結果: `[eval].policy.policy_type = Greedy`、`policy.eps_end` は存在しない。A2 の `.$` は在処では解決されないので、`A2.actor.[eval].policy.eps_end` という組み立て葉は生まれず、層経由で写ることもない。`A2 : actor.[eval].policy.eps_end = 0.05` と直接書いた分岐では、独立した直書きとして eps_end = 0.05 が残る。

根拠: P1・P2・P3・P7。参照先の種類によらず旧継承値は残らない。

#### M07: 多段継承

入力: `BASE` に追加。

```text
DefaultDQNAgent.@baseline : actor.[eval_target_check].$ = DefaultDQNAgent.actor.[eval_target]
A2 : actor.[eval].policy.eps_start = 0.01
A2 : actor.[eval_target].policy.eps_start = 0.02
A3 : actor.[eval_target_check].clone_model = true
```

期待結果: `[eval].policy.eps_start = 0.01`、`[eval_target].policy.eps_start = 0.02`、`[eval_target_check].policy.eps_start = 0.02`。`[eval_target_check]` は network = target、clone_model = true。親の eval 継承命令を再実行して 0.01 に戻さない。記録上も `[eval_target_check]` のチェーンは 1 本だけ。

根拠: P1・P2・P3・P7。各段は親の最終値を受け取り、自身の宣言だけを適用する。

直接宣言の分岐(独立入力):

```text
DefaultDQNAgent.actor.@eval_policy : eps_start = 0.05
DefaultDQNAgent.actor.[eval] : policy.$ = DefaultDQNAgent.actor.@eval_policy
DefaultDQNAgent.@baseline : actor.[eval_target].$ = DefaultDQNAgent.actor.[eval]
A2 : actor.[eval].policy.eps_start = 0.01
DefaultDQNAgent : $ = @baseline > A2
```

期待結果: `[eval].policy.eps_start` と `[eval_target].policy.eps_start` はともに 0.01。`[eval]` に root で宣言した `policy.$` は `[eval]` の組み立てに使われるだけで、`[eval_target]` へは写らない(写せば 0.05 に戻る)。記録に `[eval_target].policy.$` は無い。

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
A2 : actor.[eval].policy.use_amp = true
DefaultDQNAgent.$ = @baseline > A2
```

期待結果: `[eval].policy`、`[eval_target].policy`、`DefaultDQNAgent.target_policy` はすべて policy_type = Greedy、use_amp = **true**。`[eval_target].network` は target。`target_policy.network` は存在しない。

根拠: P1。**コピー先がカタログでもカタログ外でも同じ**。選択した policy 部分だけの最終値をコピーする。

#### M09: 全体と部分の適用順

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

期待結果: `[eval_target]` は policy.policy_type = Greedy、network = target。eval_target の選択 2 行だけを逆順にした独立分岐では policy.policy_type = EpsilonGreedy、network = target。どちらも選択キーが異なるので両方が記録に残る。

根拠: P2・P3・P7。部分選択を特別に強くしない。

#### M10: 実際の循環

入力:

```text
DefaultDQNAgent.@baseline : actor.[eval].policy.$ = DefaultDQNAgent.actor.[eval_target].policy
DefaultDQNAgent.@baseline : actor.[eval_target].policy.$ = DefaultDQNAgent.actor.[eval].policy
DefaultDQNAgent.$ = @baseline
```

期待結果: `[eval].policy` → `[eval_target].policy` → `[eval].policy` の経路を含むエラーで停止する。空コピーとして成功させない。

根拠: P6。

#### M11: 未定義参照と空の上書き層

入力:

```text
DefaultDQNAgent.actor.@eval_base : policy.policy_type = Greedy
DefaultDQNAgent.@baseline : actor.[eval].$ = DefaultDQNAgent.actor.@eval_base
DefaultDQNAgent.$ = @baseline > A3
```

期待結果: 空の A3 を許容し、`[eval].policy.policy_type = Greedy` で正常終了する。次の各分岐は eval の右辺だけを置き換える。

根拠: P6。通常上書き層の空と、宣言済み部品を要求する参照の空を区別する。

| eval の右辺（各行は独立） | 期待結果 |
|---|---|
| `DefaultDQNAgent.actor.@missing` | 未定義プロファイルとして fail-fast |
| `DefaultDQNAgent.actor.[missing]` | 未定義カタログとして fail-fast |
| `DefaultDQNAgent.actor.@eval_base.sync_interval` | 指定部分が空のため fail-fast |

いずれも選択キー、指定 term、解決先をエラーに含める。

#### M12: 10 段の境界

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

期待結果: `[Depth01]` から `[Depth10]` まですべて type = ReLU で正常終了する。最長経路は Depth10 の選択（深さ 1）から Depth01 の選択（深さ 10）まで。DepthBase は直書きだけなので深さを増やさない。

失敗分岐は上へ `net.block.[Depth11] : $ = net.block.[Depth10]` を追加したもの。11 段になるため上限 10 と経路を含むエラーで停止する。キャッシュや宣言の並べ順によって成功へ変わらない。外側の通常選択から生成した nested 選択を使うテストでは、外側の深さも数える。

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

期待結果: `[eval]` と `[eval_target]` は EpsilonGreedy、eps_start = 0.01。@cautious をチェーンから除く独立分岐では両方 0.05。未選択の @unused は実効側へ出ず、その中の @missing 参照で停止しない。選択される `@baseline` の中に未選択の内側プロファイルがある分岐(`DefaultDQNAgent.@baseline : actor.@unused.policy.$ = DefaultDQNAgent.actor.@missing` を追加)でも停止しない。`@baseline` のコピーは `DefaultDQNAgent.actor.@unused.policy.$` を宣言として保存するだけで解決せず、後で `actor.@unused` を選択したときに実体化する(現行の dormant と同じ)。@eval_base がコピーにより生まれる前に未定義扱いしない。

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

期待結果: `Agent.net.branch.[t].structure = ReLU`、`Other.structure = ReLU`(第 2 相の葉が最終値に含まれ、参照する選択へ届く)。`config_resolution.json` の `overrides` に `by = run.@arm`、`from = SiLU`、`to = ReLU` が載る。

根拠: P1・P4・P7。**現行実装では両方 SiLU になる**（2026-08-31 / 2026-09-12 の事故）。

#### M15: Run プロファイルと上書き層・CLI の順位

入力:

```text
Agent.@baseline : learner.gamma = 0.9
A2 : learner.gamma = 0.997
Agent.$ = @baseline > A2
run.@arm : Agent.learner.gamma = 0.99
run.$ = @arm
```

期待結果: `Agent.learner.gamma = 0.99`。この入力へ CLI `Agent.learner.gamma=0.95` を与えた独立分岐では 0.95。

`run.$ = @a > @b` で両者が同じ葉キーを持つ分岐では `@b` が勝つ。`run.@arm : @vars.n = 5` と `key = ${@vars.n}` を持つ分岐では、`@vars` は第 2 相の対象外として現行どおり解決する。Run プロファイルが `X.$` を差し替える既存の用法（file-tail overwrite）も非回帰とする。解決入力キーの競合分岐: `run.@arm : @vars.n = 5` に CLI `@vars.n=7` を渡すと `key = 7`(CLI が trunk 展開に勝つ)。Run プロファイルが葉ではなくチェーンだけを持つ入力(`Agent.@baseline : learner.gamma = 0.9`、`A2 : learner.gamma = 0.997`、`run.@arm : Agent.$ = @baseline > A2`、`run.$ = @arm`。`run.@arm` の葉指定は含めない)に CLI `Agent.$=@baseline` を渡すと A2 は適用されず gamma は 0.9(CLI のチェーンを採用)。上の入力のまま渡すと `run.@arm : Agent.learner.gamma = 0.99` の第 2 相が勝って 0.99 になる。CLI `run.$=@other` は現行どおり trunk そのものを差し替える。

根拠: P4・P5。

## 5. 実装ノート

実装の主対象は [config_impl.cpp](../../core/anet-core/src/config_impl.cpp) と [config_test.cpp](../../core/anet-core/src/config_test.cpp)。

**P1（最終値読み）**: 「通常の適用位置を持つ指定」と「参照先の最終値への依存」を区別する。プロファイルの展開で生成される宣言、owner 相対の term、CLI が変更した宣言を取り込んだ上で、同じコピー先へ届く `.$` の採用チェーンを確定する(プロファイル自身の継承は差し替えの対象ではなく、その term の一部として適用する。P3)。最終値は値とキー集合の両方を解決する。部分参照では要求した部分への依存を辿り、継承元の選択命令を子へ再配置しない。独立した指定の順位は残す。キャッシュ等で解決順を変えても、適用順位・循環経路・深さ・記録が変化してはならない。第 2 相の葉(Run プロファイル・CLI)は評価器の入力に最上位の書き込みとして含める。現行 `:60-64` の `effective_map_` パッチはこの評価器へ統合し、伝播のための後処理を別に持たない。深さは P6 の最長経路 DP で求める。`.$` を写すのは参照先が宣言の袋(プロファイル、または root 直下の単一セグメント名である上書き層)のときだけで、実効側へ写した `.$` は nested 選択として解決して `working_map_` には残さない(現行 `:315` の変更)。`@` を含むコピー先へ写した `.$` は保存して解決しない(現行 `:318` のまま)。組み立て済みノード(複数セグメントの非 `@` prefix)の配下の `.$` は写さない。上書き層配下の `.$` は root 選択の走査(`:38-46`)から除外し、term と root 選択 owner の兼用は fail-fast にする。

**P4（Run プロファイル第 2 相）**: `ExpandNamedTrunk` が採用した項と葉キーを保持し、項を畳んでキーごとの最終値を作ってから `Resolve()` の CLI 第 2 相の直前で 1 回だけ適用する。trunk 展開の後に CLI の解決入力キー(`run.$` 以外)を再注入する。`IsResolverInputKey` の判定を CLI と共有する。P4 単体の実装量は小さい(第 2 相の再適用と `overrides` の記録)。伝播は P1 の評価器が担う。

この説明は責務と制約であり、データ構造やアルゴリズムの指定ではない。設定全体の Jacobi 反復、last-writer graph、周回数の安全弁を必須方式にしない。実 config で生じない writer graph を注入するためだけの API やテストも要求しない。選んだ方式は M01〜M15 と既存テストで説明・検証できる最小構成にする。

通常選択の dormant 状態、`run.$` の先行展開、CLI 2 相、`${}` 1 段展開を維持する。**`[key]` かどうかで選択の読み方を分岐させない。**

## 6. 受入条件とマニュアルの対応

各 M 例の「入力・期待結果・根拠」を実装時の resolver テストへ対応付ける。表の条件を一部の happy path だけで代用しない。**A11・A12 は現行実装で RED であることを先に確認する。** M04 の 2 行目の A3 無し分岐と M14 の `Other` も現行では RED である(§2)。

| 受入 ID | 対応例・入力 | 確認する結果と原則 |
|---|---|---|
| A01 | M01〜M03 | 参照先への後段変更と追加キーが届く。変更は親へ逆流しない（P1） |
| A02 | M04 の CLI なし／各 CLI 分岐 | 子だけの変更、親 CLI の伝播と子 CLI の最優先を区別する（P2・P5） |
| A03 | M05、M09 と逆順分岐 | 直書きより選択が勝ち、全体・部分は適用順で勝敗が変わる（P2） |
| A04 | M06 とカタログ分岐・上書き層差し替え分岐 | チェーン差し替えで旧継承値が消え、独立した指定は順位を保つ。上書き層が運ぶ `.$` 宣言がコピー先で差し替えとして働き、層の在処では解決されないので旧チェーンの葉が層経由で残らない（P1・P3） |
| A05 | M07 と直接宣言の分岐・M08 | 多段・部分継承とカタログ外への参照は最終値を使い、親の命令を再実行しない。参照先に root で宣言した部分選択も継承先へ写さない（P1・P3） |
| A06 | M10、自己供給の分岐 | 実際の循環は経路付きエラー。`X.part.$ = X` と `X.$ = X` は自己供給として経路付きエラー、`X.$ = X.part` で書き戻しの無い形(`app.$ = app.online > P1`)は正常（P6） |
| A07 | M11 の各分岐 | 未定義プロファイル／カタログ／部分はエラー、空の通常層は成功。上書き層の自己チェーン(`A2.$`)と term / owner の兼用はエラー（P6） |
| A08 | M12 の成功／失敗分岐 | 10 段成功・11 段失敗。nested 深さの既存テストも維持（P6） |
| A09 | M13 と内側プロファイルの分岐、既存の相対 term・入れ子展開テスト | 後から生じるプロファイルを読める。未選択の在庫は実効化せず、素材へ写した `.$` は保存して解決しない（P1・P6） |
| A10 | M08 の `target_policy`、既存の root 宣言順・nested 選択テスト | **カタログ外の通常選択も最終値を読む**（P1） |
| A11 | M14 | Run プロファイルの葉キーが選択チェーンに勝つ（P4） |
| A12 | M15 と各分岐 | Run プロファイルが上書き層に勝ち、CLI がさらに勝つ。解決入力キー(`@vars` / `X.$`)でも CLI が Run プロファイルに勝ち、`run.$` の CLI 差し替えは維持。項の順序、`@vars` 非回帰、`X.$` 差し替えの非回帰（P4・P5） |
| A13 | M06・M07・M09・M14、既存の解決記録テスト | P3 が適用したチェーンだけを適用順に載せ(同一 owner の継承は [config_test.cpp:939](../../core/anet-core/src/config_test.cpp) の 2 件が両方 `key = Env.$` で載る。`key` も pin する)、`overrides` には第 2 相の最終値が実効値を変えたキーだけを記録し、項ごとの書き戻し(4 → 1 → 4)は載せない（P7） |
| A14 | 既存の include・trunk・CLI・値参照テスト | 入力順、Run プロファイル、CLI、1 段値参照とその異常系を維持（P5） |
| A15 | §7 の全対象 | 実効設定の差分ゼロ、`selections` / `references` の内容と順序の一致、`overrides` が空。差分があれば未完了 |

チェーン差し替えの追加境界として、M06 の入力に `DefaultDQNAgent.actor.@greedy : clone_model = false` を追加する独立分岐では、A2 の true より後の新チェーンが勝ち `clone_model = false` となることを確認する。M11 では M08 の `target_policy` の右辺を存在しない `DefaultDQNAgent.actor.[eval].sync_interval` へ置き換える独立分岐も追加する。

CLI のチェーン指定、同じ `.$` の include による後勝ち、owner 相対参照の生成先は既存の適用順を入力として検証する。テストのために優先順位を変更しない。

## 7. 既存設定の差分ゼロを完了条件にする

実装前後で、同じ設定入力・include・CLI・注入値を固定し、ConfigManager が公開する `config_data.txt` 相当の実効設定と `json/config_resolution.json` の payload を比較する。解決記録は内容だけでなく順序も対象。新フィールド `overrides` は空であることを条件とし、`selections` / `references` は内容と順序の一致を求める(空の `overrides` の有無だけを差分と数えない)。外側のログ時刻等を比較対象と混同しない。**baseline は config ツリーを commit hash で固定して採取する**（探索キャンペーン中は設定が日次で変わるため）。

| Env 設定 | 必須の比較入力 |
|---|---|
| Atari.txt | 下記の Run プロファイル 6 チェーン |
| DropMerge.txt | 実装着手時の現行 IQN32 チェーンを CLI 込みで固定し、さらに `run.@iqn32_stratified` と `run.@qr51_control` |
| LunarLander.txt | ファイル既定と `run.@repro` |
| ImageCls.txt | ファイル既定、`run.@resnet18ish_hr`、`run.@convnext_atto_hr` |
| GridMaze.txt | ファイル既定（現行の Agent / NN 選択を含む） |
| GridMaze_muzero.txt | ファイル既定（MuZero の選択を含む） |
| CartPole.txt | ファイル既定（NN block カタログの選択を含む） |

Run プロファイルを宣言していない Env に比較用の新プロファイルは作らず、既定入力を使う。実装時に代表入力が変わっていたら、採取対象の名前・入力を先に記録し、比較範囲を無断で減らさない。

Atari の `run.$` に渡す 6 チェーン:

1. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard125>run.@munch`
2. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@capall`
3. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@btrnet>run.@btrsn>run.@envs64`
4. `run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>run.@evalonly>run.@greedy_eval>run.@to_50`
5. `run.@nature_dqn`
6. `run.@classic_iqn_impala_x2`

3 は Run プロファイルが `DefaultDQNAgent.$` と `net.branch.[main_feature].$` の両方を差し替える形を含む。

**2026-09-12 の静的調査では、P1 の適用により実効設定が変わる箇所は 0 件だった**（§9）。したがって P1 の差分ゼロは構造上成立するはずで、差分が出た場合は調査か契約の理解が誤っている。差分は単に記録して許容せず、原因と契約を再確認し、解消するまで未完了とする。既存 Run artifact を書き換えて一致させない。事前 baseline を採れなかった場合はその不足を記録し、比較合格とは扱わない。反復周回数や長時間学習の成績は受入条件にしない。

**P4 による差分は 0 件を想定する**(§9: 既知の衝突は設定側で回避済み、`qr.num_quantiles` は同値)。差分が出たら、それが本 PRD が見つけた既存の握り潰しである。

## 8. 検討経緯

| 旧案 | 判断と理由 |
|---|---|
| 最終値読みをカタログ（`[key]`）に限定する | **却下**。限定の理由は既存 config への影響回避であって原理ではなく、`[key]` の有無で挙動が変わる特別扱いを生む。§9 の調査で影響 0 件が確認できたため、全選択で揃える |
| identity を持つ項目だけ最終値読みにする | 却下。identity は消費側（コードが名前で引く）の性質で、設定解決時点では誰も知らない。生成側の意味論を変える根拠にならない |
| 組み立てをチェーン後段へ置く順序の約束 | 却下。読みは最終値・書きは宣言位置という要求が両立しない。組み立てを A2 / A3 の後ろへ置くと、上書き層が最後の言葉であるという前提が壊れる |
| 継承専用の記号を新設する（`&` 等） | 却下。全選択で揃えれば見分ける必要がない |
| 選択を依存順へ並べ替える | 却下。後段 overlay が負けるなど書き込み優先順位を変える |
| 書きかけ map を同順で繰り返し読む | 必須方式から除外。前段が親を戻すと、その古い値を再びコピーしうる |
| 全設定の Jacobi 反復、last-writer graph、周回数の安全弁 | 必須方式から除外。今回の契約に方式依存の仕組みを増やす |
| 回数による収束・長い通常 prefix の伝播 | 受入から削除。契約の正しさを測らない |
| policy カタログ + policy_key | 追加しない。共通プロファイルと Actor カタログで表せる |
| `+=` 相当の差分演算子 | 追加しない。`=` は置き換えであり、差分は右辺の継承表現で足りる |
| 握り潰しを WARN で可視化するだけ（順序は変えない） | 却下。検出に必要な準備は第 2 相の実装と同じなので、要求どおり直すほうがよい（旧 999 案 B） |
| 第 2 相の葉(Run プロファイル・CLI)は終端上書きで、選択を通って伝播しない | 却下（2026-09-12 レビュー）。M04 の A3 無し分岐「両方 0.04」と 2026-09-08 の裁定「CLI 最終値まで継承」に反し、`run.@x : …actor.[eval].policy.eps_start = 0` で `[eval]` だけが変わる stale copy を Run 層に作り直す。P1 の評価器に第 2 相の葉を含めれば後処理も再実行も要らない |
| owner と source が包含関係にある選択を fail-fast にする | 却下。`app.$ = app.online > P1` など実設定 9 件が落ちる。自己供給(書き込み範囲が source と交わる)だけを止める（P6） |
| 2026-09-08 の滑り例(`X.$ = X.part` に `X.part.part.part.value = 1`)を正常 config として周回で収束させる | 撤回。書き込み範囲が source と交わる自己供給として fail-fast にする（P6） |
| 非 `@` の参照先からは一律に `.$` を写さない | 却下（2026-09-12 Codex）。上書き層 A3 が運ぶ `.$` 宣言も写されず M06 の差し替えが起きない |
| `.$` の由来(宣言由来だけ写す)で決める | 却下（2026-09-12 Codex）。`[eval]` に root で宣言した `policy.$` が `[eval_target]` へ写されて再実行され、後段 A2 の 0.01 が 0.05 に戻る。宣言の袋(プロファイル・上書き層)か組み立て済みノードかで決める（P1・P3） |
| 上書き層の `.$` を在処でも root 選択として解決し、組み立て値を層の葉として写す | 却下（2026-09-12 Codex）。`A2 : actor.[eval].$ = @eps` を A3 で `@greedy` に差し替えても、A2 の在処で組み立てた `eps_end` が葉として写って残る。上書き層の `.$` は在処で解決しない（P1・P3） |
| 第 2 相で Run プロファイルの項を逐次再適用し、変更ごとに記録する | 却下（2026-09-12 Codex）。`run.@a5 : A2.learner.replay_ratio = 1` → `run.@rr4 : … = 4` で 4 → 1 → 4 と揺れ、最終値不変でも `overrides` が増えて A15 に反する。キーごとの最終値を確定して 1 回だけ適用する（P4・P7） |
| 解決入力キーは Run プロファイルが CLI に勝つ(現行のまま) | 却下（2026-09-12 レビュー、ユーザー裁定）。「CLI は最優先」が葉だけの規則になり 2 文に増える。trunk 展開後に CLI の解決入力キーを再注入して全キーで成立させる（P4） |

## 9. この文書改訂の検証範囲

### 静的調査（2026-09-12、実施済み）

`apps/runner/config/*.txt` を include 単位で結合し、**「選択 S が項 Ti の中で宣言され、S の参照先へ Tj（j>i）が書いている」箇所**を走査した。これが P1 で実効設定が変わる母集団である。`run.$` は選択より前に展開されるため対象外とした。P1 で結果が変わり得る形は他に 2 つあるが、いずれも 0 件である。(a) 後段の root 選択が参照先へ書く形: 素材でも上書き層でもない参照先は 9 種(`net.block.[Linear_120]` / `[Linear_84]` / `FC1` / `FC2` / `FC3`、`app.online`、`app.batchrun`(`run.@a5` / `run.@plasticity` の `app.$` が参照する。root の `app.$ = app.batchrun > P1` はコメントアウト)、`MuZeroAgent.baseline`、`metrics.scalar.muzero`)で、いずれもその prefix を owner とする選択を持たない。親 prefix を owner とする選択(`app.$` / `MuZeroAgent.$` / `metrics.scalar.$`)はあるが、その term(`P1` / `A1` / `M1` と参照先自身)に `online.*` / `batchrun.*` / `baseline.*` / `muzero.*` を書く葉は無い(P6 の自己供給検査と同じ条件)。(b) 第 2 相の葉が参照先へ書く形: 工程 2 が工程 4 より先に走ることは理由にならない。工程 4 の選択が同じキーを `working_map_` 上で潰せば、参照先の読みは潰れた値になる(M14 がその形で、現行では `Other` が SiLU を読む)。0 件の理由は、Run プロファイルの葉で選択の参照先 prefix の配下にあるものが上書き層 A2 / A3 へ書くものだけで(E1 / M1 / M2 / P1 / A1 へ書く Run プロファイルの葉は無い)、owner が A2 / A3 またはその祖先である選択が無いことにある。工程 2 で入れた値は工程 4 で潰されず、第 2 相の値は第 1 相の値と同じになる。これは「上書き層からの書き込み 534 行は一度も素材の中へ書いていない」の裏返しである。bat が渡す CLI も同じで、`run.$` / `backend.$` は第 1 相専用、`E1.game` / `app.run_name` / evalonly で手渡す `A3.auto_load_file` は第 1 相と第 2 相の両方に入るが、E1 / A3 へ書く選択は無く、`app.run_name` は参照先(`app.online` / `app.batchrun` / `P1`)の配下に無い。

| env | チェーン | 該当 |
|---|---|---|
| Atari | 22 | **0** |
| DropMerge | 14 | **0** |
| LunarLander | 6 | **0** |
| GridMaze / GridMaze_muzero | 3 / 3 | **0** |
| CartPole | 2 | **0** |
| ImageCls | 1 | **0** |

選択は全 98 本、参照先 61 種。**上書き層からの書き込み 534 行は、一度も素材の中へ書いていない。** 走査器は合成入力で陽性 1 例・陰性 2 例を固定してから実行した。

**これは静的解析であり、resolver を実行した比較ではない。** §7 の差分ゼロ gate の代わりにはならない。

P4 で差分になり得た既知の衝突は次の 1 キー(2 回)で、いずれも設定側で回避済みのため現在の差分は 0 件である(`run.@head_relu` は Atari.txt に無く、`run.@btrnet` は `DefaultDQNAgent.$` の差し替えに書き換え済み)。

| キー | Run プロファイル | 勝っている定義 | 現状 |
|---|---|---|---|
| `DefaultDQNAgent.net.branch.[tau_embedding].structure` | `run.@head_relu`（Atari） | `net.@iqn` | 2026-08-31 に回避済み。2026-09-12 に `run.@btrnet` で再発し、`DefaultDQNAgent.$` のチェーン差し替えへ書き換えて回避 |

`DefaultDQNAgent.qr.num_quantiles`（DropMerge `run.@qr51_control` 対 `agent.txt @baseline`）は両方 51 で同値のため差分にならない。

### メモリ内モデル（グリル中の限定確認）

順序を持つ直書きと参照を使い、(1)プロファイル後段変更、(2)カタログ後段変更、(3)子の後段変更、(4)直書きより選択が勝つ、(5)差し替えによる旧キー消失、(6)カタログ外への部分参照、(7)独立した部分の相互参照、(8)全体の後に部分、(9)部分の後に全体、(10)実際の循環、の 10 例を確認した。**概念モデルの確認記録であり、M01〜M15 全体や production resolver が通ったという意味ではない。**

### 未実施

| 検証 | 位置付け |
|---|---|
| 実パーサによる `:` 正規化、include、入れ子プロファイル展開・owner 相対参照との統合 | 実装時の A09・A14 で確認 |
| C++ のビルド・resolver テスト、10 段境界、JSON 解決記録 | 実装時に確認 |
| 各 Env / Run プロファイルの実効設定・解決記録の前後比較 | §7 の完了 gate |

今回は文書だけを変更し、コード・設定・Run artifact は変更しない。

## 10. スコープ外

- **NN ブロックの未知キー fail-fast**。2026-08-30 の `res.init2.mode` 黙殺はブロック設定側の別機構で、PRD 065 のレビューでも「065 に畳まず別 PRD」と裁定済み。本 PRD は config resolver のみを扱う
- **上書き層の名前・段数の再設計**。PRD 059 で確定済み
- **`config_data.txt` の出力内容**。プロファイルを出力しない現行仕様は維持する
- **Actor のコード・設定移行**。PRD 061 で行う
