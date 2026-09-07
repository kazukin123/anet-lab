# PRD 072: 設定リゾルバの `$` 選択を依存順で解決する

- 起票日: 2026-09-07
- 状態: **起票済み・実装待ち**。PRD 061 P2 の前提として先に対応する
- 対象: `core/anet-core/src/config_impl.cpp`(`ConfigResolver::Resolve` / `ResolveSelection` / `ApplyTerm`)、`core/anet-core/src/config_test.cpp`(`[config][resolver]` 群)
- 関連: PRD 061(§5.3 のカタログ項目間継承がこの PRD を必要とする)、`done/059_config_concept_tree_alignment_10prd.md`(選択チェーン / プロファイル / 上書き層の用語と現行リゾルバ)、ADR 0038
- 発見経緯: PRD 061 のグリル(2026-09-07)で、`DefaultDQNAgent.actor.[eval_target].$ = DefaultDQNAgent.actor.[eval] > DefaultDQNAgent.actor.@target` のようなカタログ項目間の継承が、後段の上書き層(A2 / A3)による `[eval]` の変更を取りこぼすことが判明した

## 1. 背景

選択チェーン(`<スロット>.$ = A > B > C`)は右が後勝ちで、term の配下キーを owner 配下へコピーして実効値を作る。カタログ項目(`X.[a]`)をチェーンの term に置けるので、「`X.[b]` は `X.[a]` を継承して差分を足す」を既存記法で書ける。しかしリゾルバは次の順で解くため、`[a]` の**後から**書き換わる値が `[b]` へ届かない。

## 2. 現行コードで確定している事実(2026-09-07 時点)

- `ConfigResolver::Resolve`(`config_impl.cpp:30-77`): 先に `run.$` の trunk を展開し、次に working map を走査して `.$` で終わる root 選択キーを**宣言順**(OrderedMap の挿入順 = include 順・ファイル内順)に集め、その順で `ResolveSelection` する。素の leaf はこの走査で実効 map へ置かれる
- `ApplyTerm`(`:288-334`): term の resolved prefix 配下のキーを owner 配下へコピーし、**コピーで生じた nested `.$` はその term の直後に再帰的に解決する**(`:326-333`)
- 依存の例: `DefaultDQNAgent.$ = @baseline > @iqn > A1 > @bf16 > A2 > A3`(`Atari.txt:19`)が `@baseline` term をコピーした時点で `DefaultDQNAgent.actor.[eval_target].$` が nested 選択として現れ、直後に `[eval]` の**その時点の値**をコピーする。その後の A2 term が `DefaultDQNAgent.actor.[eval].policy.eps_start` を書いても `[eval_target]` には反映されない
- root 宣言の順序に依存する回避(`[eval_target].$` を env ファイル末尾に置く等)は include 順に依存し、`common.txt` / `agent.txt` のような共通ファイルでは成立しない
- 既存の保護: 循環検出(`:265-269`、経路付き)、深さ上限 `kMaxSelectionDepth = 10`(`:260-263`)、未定義素材の fail-fast(`:304-309`)、`resolution.json` に選択の記録(`selections` / `references`)
- 既存 config には「別の選択の owner 配下を source にする選択」は無い(env の `train.eval.[tag].env.$ = AtariEnv.@… > E1` は素材とファイルローカル層だけを source にしており、`AtariEnv.$` の産物を source にしない)。したがって順序変更で既存 config の実効値は変わらないはずである(受入で実証する)

## 3. ゴールと非ゴール

- ゴール: `$` 選択の解決結果が**宣言順と include 順に依存しない**。カタログ項目間の継承が、後段の上書き層による source 側の変更を含めて解決される
- 非ゴール: 選択チェーンの記法変更、trunk(`run.$`)の意味変更、`${}` 値参照の段数変更、CLI 上書きの位相の変更

## 4. 確定契約

1. **依存規則**: 選択 S(owner O_S、terms の resolved prefix の集合 P_S)は、別の選択 S'(S' ≠ S)の owner O_S' が P_S のいずれかの prefix(等しい場合を含む)であるとき、S' の後に解決する。「S' が S の source を書く可能性がある」を静的に判定する規則であり、実際に書くかどうかは見ない
2. **nested 選択の扱い**: term のコピーで現れた `.$` キーは即時解決せず、同じ待ち行列へ入れて依存規則で並べ直す。待ち行列が空になるまで「依存の無い選択を解く → 新たに現れた選択を加える」を繰り返す
3. **循環**: 依存規則で循環(S が S' に依存し S' が S に依存する)が生じた場合、経路を含めて `ANET_SYSTEM_ERROR`。既存の term 経路の循環検出と深さ上限は維持する
4. **trunk と CLI**: `run.$` の trunk 展開は従来どおり先頭で行う。CLI leaf override は従来どおり選択の後に重ね、`${}` 展開はその後の 1 段のまま
5. **記録**: `resolution.json` の `selections` は解決した順に記録する(順序は依存順に変わりうるが、集合と各 chain の内容は不変)
6. 自己依存(自分の owner 配下の素材を term にする通常形、例 `DefaultDQNAgent.$ = @baseline`)は依存と見なさない

## 5. 実装ノート(Codex 向け)

- `Resolve` の root 選択列挙と `ApplyTerm` の nested 即時解決を、「未解決選択の集合」+「依存グラフ(owner prefix と source prefix の prefix 関係)」による繰り返し解決に置き換える。判定は文字列の prefix 比較で足りる(セグメント境界を `.` で見る)
- 依存の無い選択が複数あるときは宣言順で解く(現行との差を最小にする)
- nested 選択が現れるたびに依存グラフへ追加する。同じ選択キーが再び現れた場合(同一 owner への二重宣言)は既存の後勝ち規則に従う
- 既存テスト(`config_test.cpp:661-1203` の `[config][resolver]` 群)は全て緑を維持する

## 6. 受入条件

1. **等価性**: リポジトリ管理下の全 env config(`Atari.txt` / `DropMerge.txt` / `LunarLander.txt` / `GridMaze.txt` / `GridMaze_muzero.txt` / `ImageCls.txt` / `CartPole.txt`)× 代表 Run プロファイル(各ファイルの主要 `run.@…`)で、修正前後の実効 dump(`config_data.txt`)が一致し、`resolution.json` の `selections` が順序を除いて一致する
2. **伝播**: 最小 config の単体テスト — `X.[a].k = 1`、`X.[b].$ = X.[a]`、その後の上書き層で `X.[a].k = 2` を `X.$ = … > L` 経由で与えたとき、`X.[b].k == 2` になる。同じ config を include 順を入れ替えて与えても結果が同じ
3. **nested**: 素材のコピーで現れた nested 選択が、それを生んだ選択の後段 term の値を見る(PRD 061 の `[eval_target]` の形)
4. **循環**: `X.[a].$ = X.[b]`、`X.[b].$ = X.[a]` は経路付きで fail-fast。既存の循環・深さ・未定義素材テストは緑
5. **trunk / CLI**: 既存の `[trunk]` / `[cli]` テストが緑。CLI leaf が選択の後に勝つことは不変

## 7. スコープ外

- 選択記法や上書き層の設計変更(059 の範囲)
- `${}` 値参照の多段化
- PRD 061 の Actor 設定カタログそのもの
