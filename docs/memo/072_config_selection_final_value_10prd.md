# PRD 072: 設定リゾルバの `$` 選択が source の最終値を読むようにする(書き込み優先順位は不変)

- 起票日: 2026-09-07、改訂: 2026-09-08(Codex レビュー「nested 選択の後回しで後段 overlay が負ける」を受け、§3〜§6 を「順序の並べ替え」から「読み取りだけ最終値にする」へ書き直し)
- 状態: **起票済み・実装待ち**。PRD 061 P2 の前提として先に対応する
- 対象: `core/anet-core/src/config_impl.cpp`(`ConfigResolver::Resolve` / `ResolveSelection` / `ApplyTerm`)、`core/anet-core/src/config_test.cpp`(`[config][resolver]` 群)
- 関連: PRD 061(§5.3 のカタログ項目間継承がこの PRD を必要とする)、`done/059_config_concept_tree_alignment_10prd.md`(選択チェーン / プロファイル / 上書き層の用語と現行リゾルバ)、ADR 0038
- 発見経緯: PRD 061 のグリル(2026-09-07)で、`DefaultDQNAgent.actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target` のようなカタログ項目間の継承が、後段の上書き層(A2 / A3)による `[eval]` の変更を取りこぼすことが判明した。初版は「選択を依存順に並べ替える」案だったが、Codex レビューで並べ替えが「右が後勝ち」を壊すことが示された

## 1. 背景

選択チェーン(`<スロット>.$ = A > B > C`)は右が後勝ちで、term の配下キーを owner 配下へコピーして実効値を作る。カタログ項目(`X.[a]`)をチェーンの term に置けるので、「`X.[b]` は `X.[a]` を継承して差分を足す」を既存記法で書ける。しかしリゾルバはコピー時点の値を写すため、`[a]` が**後から**書き換わっても `[b]` へ届かない。

一方で、コピーがどの位置で書くか(優先順位)は現行のとおり正しい。問題は「いつ読むか」だけであり、「いつ書くか」を変えてはならない。

## 2. 現行コードで確定している事実(2026-09-08 時点)

- `ConfigResolver::Resolve`(`config_impl.cpp:30-77`): (1) `run.$` の trunk 展開 → (2) working map を走査し、`.$` 以外・素材以外の**直書き leaf を先に実効 map へ置く**(`:38-45`)→ (3) `.$` で終わる root 選択を宣言順(OrderedMap の挿入順 = include 順・ファイル内順)に解く(`:48-57`)→ (4) CLI leaf override を重ねる(`:59-64`)→ (5) `${}` を 1 段展開する(`:66-67`)
- `ApplyTerm`(`:288-334`): term の resolved prefix 配下のキーを**その時点の値で** owner 配下へコピーし、コピーで現れた nested `.$` は**その term の直後に即時**解決する(`:326-333`)。後続 term のコピーは同じ target key を上書きする
- したがって現行の**書き込み優先順位**は「直書き leaf < 選択(root は宣言順、term は左→右、nested はそれが現れた term の位置)< CLI leaf」であり、これは 059 で意図した契約である
- **stale 読みの例**(PRD 061): `DefaultDQNAgent.$ = @baseline > … > A2 > A3` の term `@baseline` をコピーした時点で `DefaultDQNAgent.actor.[eval_target].$` が nested 選択として現れ、直後に `[eval]` の**その時点の値**をコピーする。その後の A2 term が `DefaultDQNAgent.actor.[eval].policy.eps_start` を書いても `[eval_target]` には反映されない
- **優先順位が効いている例**(Codex レビュー、DropMerge の ALGO 素材と同構造):

```text
DefaultDQNAgent.@iqn.net.$ = net.@iqn
net.@iqn.branch.[value_stream].bind = iqn_fusion
A2.net.branch.[value_stream].bind = alternative
DefaultDQNAgent.$ = @iqn > A2
```

  term `@iqn` のコピーで現れた `DefaultDQNAgent.net.$` が即時展開されて `iqn_fusion` を書き、続く term `A2` が `alternative` を書く → **`alternative` が勝つ**。nested 選択を後回しにすると `iqn_fusion` が最後に書かれて逆転する。root 選択同士でも同じで、重なる領域を書く 2 つの root 選択の順序を入れ替えれば宣言順の勝敗が反転する
- 既存の保護: 循環検出(`:265-269`、経路付き)、深さ上限 `kMaxSelectionDepth = 10`(`:260-263`)、未定義素材の fail-fast(`:304-309`)、`resolution.json` に選択の記録(`selections` / `references`)
- 既存 config に「別の選択の産物を source にする選択」は無い(env の `train.eval.[tag].env.$ = AtariEnv.@… > E1` は素材とファイルローカル層だけを source にする)。したがって読み取り時点を変えても既存 config の実効値は変わらないはずである(受入で実証する)

## 3. ゴールと非ゴール

- ゴール: 選択のコピーが **source の最終値**と**最終キー集合**を読む。継承元が後段の term や overlay で書き換わっても、継承先へ届く
- ゴール: **書き込みの優先順位は現行のまま**。宣言順・term 順・nested の位置・CLI の位相を一切変えない
- 非ゴール: 選択の並べ替え(初版の案。却下)、選択チェーンの記法変更、trunk(`run.$`)の意味変更、`${}` 値参照の段数変更、CLI 上書きの位相変更

## 4. 確定契約

1. **優先順位(書き込み順)は不変**: 直書き leaf < 選択(root は宣言順、term は左→右、term のコピーで現れた nested 選択はその term の直後)< CLI leaf。同じキーへの後の書き込みが勝つ。初版の「依存順に並べ替える」規則は採らない
2. **読み取りは最終値**: 選択のコピーが target へ書く値は、**全展開が終わった後の source キーの値**である。source 自身が別の選択のコピー先なら、その最終値を再帰的に読む
3. **キー集合も最終**: コピー対象は「全展開が終わった後に source prefix 配下に存在するキー」の集合である。後段の term や overlay が source に新しいキーを足した場合も target に現れる(値だけを遅延させる方式では漏れるため、この項を契約に含める)
4. **意味論の一文**: 「各選択を宣言位置で展開したものとして扱い、コピーする値と対象キー集合だけを展開完了後の source の状態で確定する」
5. **循環と非収束**: 参照を辿って自分へ戻る場合は経路付きで `ANET_SYSTEM_ERROR`(既存の term 経路の循環検出と深さ上限 10 は維持)。§5 の反復が上限周回で収束しない場合も、変化し続けているキーを含めて fail-fast
6. **trunk / CLI / `${}`**: `run.$` の trunk 展開は従来どおり先頭。CLI leaf override は従来どおり選択の後に重ね、`${}` 展開はその後の 1 段のまま
7. **記録**: `resolution.json` の `selections` は 1 周目の展開順で記録し、内容・順序とも現行と同じ

## 5. 実装ノート(Codex 向け)

- **方式: 現行の展開を同じ順序で不動点まで反復する。** `Resolve()` の (3) 選択フェーズをループで包み、1 周ごとに working map が変化したかを検出する。`ApplyTerm` の順序ロジック(term 左→右、nested 即時)には触れない
  - 1 周目は現行と同一。2 周目は 1 周目で蓄積した working map を source として同じ順序で再展開するので、stale だった値は最終値に置き換わり、後段で追加されたキーも写る。一方で後段 term / leaf のコピーも同じ順序で再実行されるため、優先順位は現行実装そのもので保たれる(Codex の例は 2 周目も `iqn_fusion` → `alternative` の順に書かれる)
  - 収束は参照鎖の深さ + 1 周で保証される。周回上限は `kMaxSelectionDepth` を流用し、超えたら「変化し続けているキー(前周との差分)」を含めて fail-fast
  - `selections_` の記録と「素材未定義」検査は 1 周目だけ行う(2 周目以降はキー集合が広がるだけで縮まない)。循環検出は毎周そのまま走らせてよい
  - 直書き leaf の実効 map 投入は 1 周目の前に 1 回だけ。CLI leaf と `${}` はループの後に従来どおり
  - 変化検出は「`Set` で値が変わったキーがあったか」の dirty flag で足りる(map の全比較は不要)
- **検討して採らなかった方式: 参照(lazy reference)モデル。** コピー時に値でなく「source キーへの参照」を置き、展開完了後に参照を辿って評価する。意味論は明示的だが、参照の位置に優先順位を再現する仕組みと、prefix 参照のキー集合展開(不動点)が別途必要になり、変更が大きい。不動点反復は優先順位を現行実装で保証するので、まずこちらで実装する
- 既存テスト(`config_test.cpp:661-1203` の `[config][resolver]` 群)は全て緑を維持する

## 6. 受入条件

1. **前段 nested + 後段 leaf の優先順位維持**(§2 の Codex の例をそのまま): 結果は `alternative`。root 選択 2 本が重なる領域を書く場合も宣言順の勝敗が変わらない
2. **後段 overlay の伝播**: `X.[b].$ = X.[a]`、その後の term(`X.$ = … > L` の `L`)が `X.[a].k = 2` を書く → `X.[b].k == 2`。同じ config を include 順を入れ替えて与えても結果が同じ
3. **後から増えたキーの伝播**: 後段 term が `X.[a]` に新しいキーを足す → `X.[b]` にも現れる
4. **nested の位置の維持**: 素材のコピーで現れた nested 選択が、それを生んだ選択の後段 term の値を読みつつ、その後段 term に上書きされる(1 と 2 の組合せ。PRD 061 の `[eval_target]` の形)
5. **循環 / 非収束**: `X.[a].$ = X.[b]`、`X.[b].$ = X.[a]` は経路付きで fail-fast。周回上限超過の fail-fast をテストで固定。既存の循環・深さ・未定義素材テストは緑
6. **trunk / CLI**: 既存の `[trunk]` / `[cli]` テストが緑。CLI leaf が選択の後に勝つことは不変
7. **等価性**: リポジトリ管理下の全 env config(`Atari.txt` / `DropMerge.txt` / `LunarLander.txt` / `GridMaze.txt` / `GridMaze_muzero.txt` / `ImageCls.txt` / `CartPole.txt`)× 代表 Run プロファイル(各ファイルの主要 `run.@…`)で、修正前後の実効 dump(`config_data.txt`)と `resolution.json` が一致する。差が出た場合は「現行が古い値を読んでいた箇所」なので、盲目的に受け入れず 1 件ずつ確認して記録する

## 7. スコープ外

- 選択記法や上書き層の設計変更(059 の範囲)
- 選択の依存順並べ替え(初版の案。優先順位を壊すため却下)
- `${}` 値参照の多段化
- PRD 061 の Actor 設定カタログそのもの
