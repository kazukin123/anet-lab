# PRD 059 Phase 1 素材化ガイド(env 担当枠共通)

## 目的と正本

- 各 env 設定ファイルの named 素材を `@` 宣言へ移行し、ALGO 切替をチェーン 1 項に畳み、dump(`config_data.txt`)を「①実効値+②無印上書き層」だけへ純化する(059 §6.1 の③=`@` 化前の無印素材の消滅)。
- 設計の正本は `docs/memo/059_config_concept_tree_alignment_10prd.md`(特に §2〜§4)。本ガイドは作業手順と過渡期判断の焼き込み。矛盾時は PRD が勝つ。
- **見本は `apps/runner/config/LunarLander.txt`**。素材宣言・`:` 糖衣・ALGO 素材・`${}` 予算一点化・named 幹見本のすべてが入っている。迷ったら LL の形に合わせる。

## 前提(ワークツリー状態)

- PH0(ConfigResolver)はコミット済み。PH1a(`<agent>.net` rename+validator)、PH2(幹前段+inspect_run `resolution`)、LunarLander 見本素材化が**未コミットでワークツリーに載っている**。この上へ積む。
- チェーン既定素材(`DefaultDQNAgent.@baseline / @fast / @balance / @heavy / @bf16`)と各 env のチェーン参照の機械追随は**実施済み**。残作業は env ファイル固有の素材群。

## 作業項目(env ファイル単位)

1. **env プリセットの `@` 化**: `<Env>.<preset>.` → `<Env>.@<preset>.` へ rename し、チェーン参照(`<Env>.$ = ...`)とテスト用参照(`train.eval.[tag].env.$` 等)を追随させる。コメントアウトされたラダー行も同時に rename する(将来の有効化で壊れないように)。
2. **ALGO 素材の導入**(059 D12): env ファイル内に以下を定義し、`DefaultDQNAgent.$` チェーンへ `@qr` / `@iqn` の 1 項として挿入する。旧 2 行同期(`R.quantile_mode` / `A.quantile_mode` と root `DefaultDQNAgent.net.$` 直書き)は**削除**する(素材が供給するため)。

   ```text
   DefaultDQNAgent.@qr  : quantile_mode = qr
   DefaultDQNAgent.@qr  : net.$ = net.@qr

   DefaultDQNAgent.@iqn : quantile_mode = iqn
   DefaultDQNAgent.@iqn : net.$ = net.@iqn
   ```

3. **配線素材の `@` 化**: `net.qr.* / net.iqn.*` → `net.@qr : ... / net.@iqn : ...`(`:` 糖衣)。参照元は上記 ALGO 素材の `net.$` のみになる。
4. **予算の一点化**(059 D11): `app.online.exp_pause_step` と `app.batchrun.exp_exit_step` が同じ値を二重管理している場合は `@vars : max_exp_step` へ一点化し、両キーを `${@vars.max_exp_step}` 参照にする。schedule 系(decay/beta 等)は予算非連動が既定(D11: 予算比例ルールを共通機構にしない)。env ローカルに「予算の半分」等の連動方針がある場合は、値参照に演算機能が無い(D7)ため定数 2 つで書き、`@vars` への隣接定義か named 幹での束ね(複数 namespace の値スロットを束ねられるのは root 展開される幹のみ)で同期忘れを局所化する。形は各枠で調整。
5. **named 幹**(059 §2.3、任意): 「使う構成にだけ」(D4)。複数スロットを束ねて 1 行で切り替える実需(再現構成・検収構成)がある場合のみ定義する。LL の `run.@repro` が見本。素材直積の事前定義は禁止。

## スタイル規約(過渡期判断の焼き込み)

- **`@baseline` からアルゴ的キーを抜かない**: baseline には quantile 系キーが残っているが、現段階では抽出しない。未素材化 env がコード既定へ落ちて dump が退行するのを防ぐため。抽出は全 env 素材化完了後の横断整合チェックで判断する。
- **ALGO 素材(`DefaultDQNAgent.@qr/@iqn`)と配線素材(`net.@qr/@iqn`)は env ファイル所有**: 配線実体(TauProj の次元等)が env 依存のため。agent.txt への共通化は横断整合チェック時に判断する。定義の重複は現段階では正
- **`:` 糖衣は素材定義行に使う**(`X.@name : key = value`)。素材名の桁を揃えると読みやすい。読み口(実効キーへの直書き)には使わない。
- **チェーンのコメントラダー文化は維持**: 切替候補は `#` ラダーで並べる現行流儀のまま。
- **上書き層は無印のまま触らない**(D20 改)。命名は直交 2 軸=文字(対象: A=Agent / E=Env / M=Metrics / P=app)×番号(恒久度: 大番号ほど揮発的・チェーン右)。全層番号付き(A1/A2/A3、E1、M1/M2、P1)。旧 R/X/O は A2/A3/M2 へ吸収済み(grill 2026-08-25)。実験行(E1 overlay 等)も素材化の対象外。
- **参照スタイル**: チェーン項は、自分(LHS)配下の素材は相対 `@name`、他所有者の素材は絶対(`net.@iqn` / `LunarLanderEnv.@trunk`)。既存行の形式に合わせる。

## 等価性検証(必須)

素材化は**実効値を 1 つも変えない**リネームである。以下で機械証明する:

1. 作業前に一時テスト(`config_test.cpp` へ `[tempdump]` タグで追加)を書き、対象 overlay の `ConfigManager::GetConfigData().ToPropertiesString()` をファイルへ書き出す(before)。
2. 素材化後に同じダンプを取り(after)、行単位で比較する。受け入れ基準:
   - **値の差分 0 件**(同一キーで値が変わった行なし)
   - **新規キー 0 件**
   - **消滅行は意図分のみ**(`@` 化で dump から消えた旧無印素材、削除した旧 2 行同期の行)— 消滅行を 1 行ずつ列挙して確認する
3. named 幹を定義した場合は、幹を CLI 第 1 相(`run.$=run.@<name>`)で有効化した dump が、同じ選択を個別 CLI で渡した dump と一致することを確認する(LL の一時テスト形式を参照)。
4. **検証後、一時テストは削除**して原状復帰する。

## 完了条件と制約

- 対象 overlay の dump に `@` なし素材(旧 named プリセット・`net.qr/iqn`・`AS.` 等)が残っていない。
- `[config]` テスト全緑+全 core テスト(既知の ReplayBuffer 2 件のみ失敗)+`resolve_workspace_test.ps1` 成功+`git diff --check` クリーン。
- smoke: 対象 env で QR / IQN 各 1 回、`app.exp_exit_step=1` 相当の起動確認(NN ヘッダログまで到達)。
- 他枠担当のファイルは変更しない(下表)。共通ファイル(agent.txt / common.txt / nn.txt / _main.txt)の変更が必要になった場合は変更せず報告する。
- `git add` / commit / push は実行しない(人間が実施)。

## 担当分担

| 担当 | ファイル |
|---|---|
| DropMerge 枠(Codex) | `DropMerge.txt` |
| Atari 枠(Claude) | `Atari.txt`(+NatureDQN 再現の named 幹化、named RUN_BUDGET 見本) |
| 本体枠 | `CartPole.txt` / `GridMaze.txt` / `ImageCls.txt`、共通ファイル、横断整合チェック |
