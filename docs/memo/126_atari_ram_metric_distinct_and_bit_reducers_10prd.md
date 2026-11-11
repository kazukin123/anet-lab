# Atari の RAM メトリクスに値の種類数とビットの変化を数える畳み方を足し、RAM 定義と縮小方式の読み取りを整理する PRD

> 2026-11-12 起票（グリルあり）。題材は課題 [issue-20261107-949e8a](../management/issues/issue-20261107-949e8a.md)（値の種類数とビットを数える畳み方が無い）。グリルの途中で、同じ解析のコードを整理する課題 [issue-20261111-9ff3b7](../management/issues/issue-20261111-9ff3b7.md) を、開発者の選択で範囲へ全部取り込んだ。
> 行番号は 2026-11-12 の作業ツリーの位置で、実装中の [PRD117](117_warn_policy_and_unread_config_10prd.md) の未コミットの変更（`AtariEnvConfig` の検証を factory と env の構築で共用する変更と、`[prd117]` のテスト）を含む。実装のときは近傍の名前で探し直す。
> 関連: 設計文書 [220 §4.9](../design/220_atari_env.jp.md)（RAM メトリクス）・[220 §4.2・§4.3](../design/220_atari_env.jp.md)（設定キーと縮小方式）、[ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)（番号付きの汎用キーとゲーム別の RAM 定義）、[ADR 0060](../adr/0060-transition-results-travel-in-step-result-and-event-decides-provisional-or-final.md)（`ram_metric.[n]` は Step の結果が答える）、[PRD082](done/082_atari_ram_progress_metrics_10prd.md)（RAM メトリクスの元の PRD）、[PRD099](done/099_atari_resize_mode_10prd.md)（縮小方式）、[PRD108](108_metrics_tag_numbering_10prd.md)（tag の番号の整理。RAM の tag の枠を 1〜9 に広げる予定。暫定・未実装）、[PRD117](117_warn_policy_and_unread_config_10prd.md)（読まれない設定の扱い）。

## 1. 目的と問題

### 問題

RAM メトリクスの畳み方は `max_seen`・`min_seen`・`inc_count`・`dec_count`・`reach_count:N` の 5 語だけで、どれも 1 バイトの値の大小か、値の遷移の回数を数える（§2.1）。1 ゲームで観測した値の種類と、ビットごとの変化を数える語が無い。そのため Montezuma's Revenge で、訪れた部屋の数（部屋番号 0x83 の種類数）と、持ち物（0xC1 のビット）を種類ごとに取った回数・使った回数を記録できない（§2.2）。

あわせて、RAM 定義を読むコードが `AtariEnvConfig` のコンストラクタに並び、畳み方の名前と種別が 5 か所に散らばり、env の設定クラスが Run 側の設定の階層（`run.eval.[..].env.`）を名指しで知っている。縮小方式（`resize_mode`）は文字列のまま 3 つの層で検証している（§2.3）。畳み方を足すたびに、この散らばった箇所を直すことになる。

### ゴール

- **G1** 全ゲームの RAM 定義で使える畳み方として、1 ゲームで観測した値の種類数を数える語を足す（D1）。
- **G2** 同じく、指定したビットが 0 から 1 になった数と、1 から 0 になった数を数える語を足す（D2〜D4）。
- **G3** Montezuma の部屋と持ち物を新しい語で数えるときの定義の例と、番号を付けるときに要る事実を、Atari の設定を決める側の材料として本書に残す（§8。Atari.txt は変えない。D8）。
- **G4** 題材の課題が挙げた 3 つの指標（訪れた部屋の数、持ち物の種類ごとの取得、Run 全体で見つけた部屋の数）の行き先を決め、本書に入れた分をすべて片付ける。Run 全体の和集合は作らない（D7）。
- **G5** 課題 9ff3b7 の整理を片付ける。RAM 定義の読み取りを 1 か所にまとめて畳み方の名前と種別を 1 つの表にし、評価の上書きを拒否する専用の検査を env の設定クラスから消して Run 側の階層の名指しを除き（D10）、縮小方式は設定を読むときに 1 回だけ検証する。既存の 5 語の結果、ERROR の文言、観測の画素は変えない（D11。例外は D6 の語の一覧と、D10 の評価の上書きの誤りの文言と時機）。

### 非ゴール

- **N1** tag の番号の枠を広げること（[PRD108](108_metrics_tag_numbering_10prd.md) の範囲）。消える案は「本書で RAM の 5 番以降の scalar の tag を作る」案である。本当の制約ではなく、tag の番号を全体で振り直す PRD108 の関心に任せる判断である。
- **N2** 既存の 5 語の意味、NaN の規則、集計の単位（実ゲーム 1 回）を変えること。消える案は「`inc_count` を『新しい値に入った回数』に直す」案である。本当の制約として扱う。設定と文書はクリーンブレークで移せるが、過去の Run の trace は書き換えないので（[AGENTS.md「クリーンブレーク方針」](../../AGENTS.md)）、同じ名前のまま意味が変わると過去の Run と比べられない。新しい語を足せば済む。
- **N3** 学習の信号（報酬・観測）を変えること。消える案は「RAM の値で報酬を整形する」案である。本当の制約として扱う（[PRD082](done/082_atari_ram_progress_metrics_10prd.md) の NG1。ゲームの仕様と学習の信号を変えない）。
- **N4** Run 全体で見つけた値の和集合（Burda らの「見つけた部屋の数」）を出すこと。開発者の判断で作らない。1 ゲームごとの値の mean と max を出しておけば、後でざっくり集計できる（D7）。消える案は §4.1 の 3 案（env の個体の一生、metrics の累積、ログと後処理）で、本当の制約ではなく、要らないという判断である。
- **N5** Atari.txt（`apps/runner/config/Atari.txt`）の変更。Montezuma を含むゲーム別の RAM 定義と番号付け、metrics の行は変えない。Atari の設定を決める側が、実験の段取りと合わせて決める（D8）。消える案は「本書で Montezuma の番号 1〜4 を結び直す」案で、本当の制約ではなく、担う場を分ける判断である。

## 2. 現状（2026-11-12 にコードで確認）

### 2.1 畳み方と値の出口

- 畳み方の種別は 5 つ（[AtariEnv.hpp:49](../../core/envs/atari1/src/AtariEnv.hpp#L49)）。定義の行 `<0x80..0xFF> <畳み方>` を読んで種別を決め（[AtariEnv.cpp:304-326](../../core/envs/atari1/src/AtariEnv.cpp#L304)）、知らない語は設定を読むときに ERROR にする（文言は `expected max_seen, min_seen, inc_count, dec_count, or reach_count:N.`）。
- ゲームの開始（`reset_game()` 直後）に初期値を取り、到達値の語は初期値から、回数の語は 0 から始める（[AtariEnv.cpp:242-248](../../core/envs/atari1/src/AtariEnv.cpp#L242)）。毎 `act()` 後に前のフレームの値と比べて畳む（[AtariEnv.cpp:250-261](../../core/envs/atari1/src/AtariEnv.cpp#L250)）。集計は実ゲーム 1 回で、episodic-life の soft reset をまたいで続く（[220 §4.9](../design/220_atari_env.jp.md)）。
- 値の出口は 3 つ。Step の結果の `ram_metric.[n]`（trace と scalar が読む。[AtariEnv.cpp:845-848](../../core/envs/atari1/src/AtariEnv.cpp#L845)）、AuxData の `ram_metric.<label>`（AtariView の表示。[AtariEnv.cpp:734-736](../../core/envs/atari1/src/AtariEnv.cpp#L734)）、ゲーム完了ログの `ram_metric:` の行（[AtariEnv.cpp:744-750](../../core/envs/atari1/src/AtariEnv.cpp#L744)）。
- 用語集は畳み方を 2 つの族に分ける。`_seen` は初期値を含む到達値、`_count` は遷移の回数である（[CONTEXT.md「畳み方」](../../CONTEXT.md)）。
- ADR 0046 は 5 語を閉じた列挙とし、「語を足すのは既存契約を壊さない追加である」と書く。毎フレームの比較は 1 バイトの参照で、コストは無視できるとした（[ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md)、Consequences・Considered Options）。
- metrics 側の集約は、同じ時点の lane をまとめる `mean.`・`max.`・`min.`・`std.` と、時間方向の EMA だけで、ゲームをまたいで値の集合をまとめる仕組みは無い（[140 §6](../design/140_observability.jp.md)）。

### 2.2 Montezuma で数えられないもの

- 今の Montezuma の番号付き定義は `1:room_max`（0x83 `max_seen`）、`2:sword_only`（0xC1 `reach_count:32`）、`3:inv_max`（0xC1 `max_seen`）、`4:x_min`（0xAA `min_seen`）で、番号なしに `room_up`（0x83 `inc_count`）がある（[Atari.txt:1405-1410](../../apps/runner/config/Atari.txt#L1405)、[220 §4.9](../design/220_atari_env.jp.md) の表）。
- `room_up` は部屋番号が増えた回数で、行き来でも増える。`mz_nm_rnd_b0p1_s1` で、得点が 400 のまま 1 ゲームで 41 回になったゲームがあった（[課題 949e8a](../management/issues/issue-20261107-949e8a.md)、事象）。`room_max` は最大の番号だけで、開始の部屋（1）の左の部屋 0 へ行っても増えない（[220 §4.9](../design/220_atari_env.jp.md)）。
- 0xC1 は持ち物のビットで、鍵は 2、部屋 6 の剣は 32 と確かめてある（[220 §4.9](../design/220_atari_env.jp.md)）。OC_Atari の注釈では鍵 2・4・8・16、剣 32・64、たいまつ 128、アミュレット 1 である（[課題 949e8a](../management/issues/issue-20261107-949e8a.md)、事象）。2026-11-06 の ALE（ale_py 0.12.0、sticky 0、1 フレームずつ）での手元の探索では、部屋 7 で剣を持ったまま鍵を取ると 0xC1 が 32 から 34 になり、鍵と剣のビットが同時に 1 になっていた（探索の記録は git 管理外）。
- 今の 5 語はバイト全体を比べるので、`sword_only` は鍵を持ったまま剣を拾った分を数えない。剣で敵を倒した回数は定義を置かず、得点で読んでいる（[220 §4.9](../design/220_atari_env.jp.md)）。
- Burda らは、訓練中に全 env・全ゲームで見つけた部屋の数を探索の指標にしている（arXiv 1810.12894、§3.1・Figure 3・4。[課題 949e8a](../management/issues/issue-20261107-949e8a.md)、事象）。anet の RAM メトリクスはゲームごとの集計で、これは出せない。

### 2.3 読み取りの散らばり（課題 9ff3b7 の事象を今のコードで確かめ直した）

- `AtariEnvConfig` のコンストラクタ（[AtariEnv.cpp:378-449](../../core/envs/atari1/src/AtariEnv.cpp#L378)）の後半 41 行が RAM 定義の読み取りで、正規表現 2 本の組み立て、全設定キーの走査、ゲームごとの定義の解析が並ぶ。
- キーの形を 2 回確かめる（419 行の既定のキーの正規表現と、427 行の `[..]` の確認）。
- 評価の上書きの拒否は、`run.eval.` で始まるか・`.env.ram_metric.` を含むか・正規表現に合うかの AND で判定する（[AtariEnv.cpp:413-418](../../core/envs/atari1/src/AtariEnv.cpp#L413)）。env の設定クラスが Run 側の階層を名指しで知っている。今は最初に作られる train の env の設定を読むとき、ROM を読む前に `Atari RAM metric override is unsupported: <キー>='<値>'.` で止まる。
- ゲーム名の形の正規表現が 2 か所にある（412 行と、`Validate` の 527 行）。実効設定への記録を `my_config_json_` へ直接書く（438〜439 行）。
- 番地は正規表現で全体一致させたあと、`stoul` の例外処理と読み残しの確認をしている（[AtariEnv.cpp:286-303](../../core/envs/atari1/src/AtariEnv.cpp#L286)）。全体一致を通った文字列は、どちらにも引っかからない。
- 畳み方の名前と種別は、種別の列挙・if の連鎖（306 行）・`Begin` の条件（246 行）・`Observe` の switch（253 行）・ERROR の文言に現れる。
- 縮小方式は文字列のまま持ち、検証が 3 か所にある。設定の読み取り（[AtariEnv.cpp:386-392](../../core/envs/atari1/src/AtariEnv.cpp#L386)。既定のキーと個別指定の両方を検証する）、`Validate`（525〜526 行）、`GrayscaleResizer` のコンストラクタ（[AtariPreprocess.cpp:66-87](../../core/envs/atari1/src/AtariPreprocess.cpp#L66)。文字列を受けて中で真偽値に直す）。
- `Validate` には、読み取りのときに検証しない項目（`game` の形、`frame_skip`、`repeat_action_probability`、`noop_max`、`mode`、`difficulty`、`max_episode_frames`）の検証も入っている（[AtariEnv.cpp:521-551](../../core/envs/atari1/src/AtariEnv.cpp#L521)）。factory の `ValidateConfig` と `AtariEnv` のコンストラクタが呼ぶ（[AtariEnv.hpp:156-159](../../core/envs/atari1/src/AtariEnv.hpp#L156)、[AtariEnv.cpp:471](../../core/envs/atari1/src/AtariEnv.cpp#L471)）。
- `AtariEnvConfig` は `src/` にあるモジュールの内側の型で、引数なしで作れてフィールドは書き換えられる（[AtariEnv.hpp:68-93](../../core/envs/atari1/src/AtariEnv.hpp#L68)）。追跡しているコードで、作ったあとにフィールドを書き換える使い方は無い。
- Run の起動では、設定にある `run.eval.[<tag>]` のすべて（定義だけのタグも含む）について、学習の開始前に接頭辞 `run.eval.[<tag>].env` で env の設定を検証する（[trainer.cpp:942](../../core/anet-core/src/trainer.cpp#L942)、[trainer.cpp:1088-1092](../../core/anet-core/src/trainer.cpp#L1088)、[trainer.cpp:1122](../../core/anet-core/src/trainer.cpp#L1122)）。`eval_panel` も設定に `run.eval.[eval_panel]` としてあるので、ここを通る（[Atari.txt:1649](../../apps/runner/config/Atari.txt#L1649)）。train の env と Agent は、この検証より前に作られる。評価の接頭辞で env を作る入口は、この検証（1092 行）と評価の env の構築（[trainer.cpp:1275](../../core/anet-core/src/trainer.cpp#L1275)）だけである。
- その検証の直後に、そのタグの下で読まれなかった設定キーがあれば `Unread config keys in run.eval.[<tag>]:` で止める（[trainer.cpp:1123](../../core/anet-core/src/trainer.cpp#L1123)、[config.cpp:353](../../core/anet-core/src/config.cpp#L353)）。タグの設定は元の設定から切り出したもので、読まれた印を元の設定と共有する（[config.cpp:314-324](../../core/anet-core/src/config.cpp#L314)、[config.cpp:601-614](../../core/anet-core/src/config.cpp#L601)）。そのため、env の設定が上書きの接頭辞の下の `ram_metric.*` を読まなければ、その行はここで止まる（コードを読んだ見込みで、受入 4 で確かめる）。書いた設定が読まれないことの検出は、[PRD117](117_warn_policy_and_unread_config_10prd.md) の D3 が全体の決まりにする。
- 評価の上書きの専用の検査は [PRD082](done/082_atari_ram_progress_metrics_10prd.md) の §7 の 6 で作った。RAM の番地と意味は ROM の事実なので学習と評価で変えず（「ROM の事実は Runner 別に変えない」）、上書きは読まないと決めた。読まない上書きは黙って無視されるので、専用の検査で止めた。

## 3. 用語

- **畳み方**（reducer）: [CONTEXT.md](../../CONTEXT.md) の語。本書で 3 語を足す（D1〜D3）。
- **マスク**: ビットの語の引数で、数える対象のビットを 1 にした 1 バイトの値。新しい用語ではなく、普通の意味で使う。
- **ビットが立つ・落ちる**: あるビットが前のフレームの 0 から今のフレームの 1 になることを「立つ」、1 から 0 になることを「落ちる」と書く。

## 4. 確定事項

| D | 決定 | 理由（依存する決定・事実） |
|---|---|---|
| D1 | 畳み方 `distinct_seen` を足す。1 ゲームで観測した値（`reset_game()` 直後の値を含む）の種類数を返す。値は 1〜256 で、開始の値から動かなかったゲームは 1 | 用語集の族では、初期値を含む到達値が `_seen` である（§2.1）。初期値を含めるのは `max_seen`・`min_seen` と同じ（§2.1）。Montezuma の部屋番号に使うと、開始の部屋を含む訪れた部屋の数になる（§8） |
| D2 | 畳み方 `bit_rise_count:<マスク>` を足す。毎フレーム、マスクの中で立ったビットの数を足す（同じフレームで 2 つ立てば 2） | 開発者の選択（§4.1 の D・E と比べた。番地の側にマスクを付ける形は §4.1 の I で退けた）。鍵と剣のビットは同時に 1 になりうるので（§2.2）、マスク 1 つで「持ち物を種類ごとに取った回数」を、すでに同じ種類を持っているときの取得も含めて数えられる |
| D3 | 畳み方 `bit_fall_count:<マスク>` を足す。毎フレーム、マスクの中で落ちたビットの数を足す | 開発者の選択（§4.1 の G と比べた）。`inc_count`／`dec_count` と同じ対の形にする。剣が消えた回数（剣で倒した回数）と扉で鍵を使った回数を、得点から推し量らずに数えられる（§2.2） |
| D4 | マスクは、`0x` に続く 16 進 2 桁（`0x01`〜`0xFF`。英字の大小は問わない）か、数字だけの 10 進（値が 1〜255。先頭の 0 は許す）で書く。判定は §6.1 の表による。それ以外（値 0、`0x00`、256 以上、`0x` に 1 桁や 3 桁、`0X`、符号つき、空、数字以外）は設定を読むときに ERROR | 10 進は `reach_count:N` と同じ形と読み方（先頭の 0 を許す）。16 進は開発者の指示で（§4.1 の K と比べて両方を残した）、`0x` と 2 桁の形は番地の書き方（[AtariEnv.cpp:286](../../core/envs/atari1/src/AtariEnv.cpp#L286) の `0x[0-9A-Fa-f]{2}`）にそろえた。マスク 0 は何も数えず、動かない設定になる（[AGENTS.md「Fail-Fast 原則」](../../AGENTS.md)）。`reach_count:N` は 10 進のまま変えない（指示がマスクに限られたため） |
| D5 | 新しい 3 語の観測の時機、集計の単位、NaN の規則、値の出口は今の 5 語と同じにする。ビットの 2 語は 0 から始め、ゲームの開始の時点ですでに 1 のビットは数えない | §2.1 の今の契約（220 §4.9、ADR 0046）。回数の族は 0 から始める（[AtariEnv.cpp:244-247](../../core/envs/atari1/src/AtariEnv.cpp#L244)） |
| D6 | 新しい ERROR の文言は §6.1 のとおり。知らない語の ERROR は、並べる語を 8 語にする | 今の文言の形を崩さず、語を足すだけにする。D1〜D3 で語が増えるので、一覧だけは変わる（D11 の例外） |
| D7 | Run 全体で見つけた値の和集合は作らない | 開発者の判断（N4）。1 ゲームごとの値の mean と max で後からざっくり集計できる |
| D8 | Atari.txt は変えない。Montezuma の定義の例と、番号を付けるときに要る事実は §8 に材料として書く | 開発者の判断（N5）。§8 は決定ではない |
| D9 | 課題 9ff3b7 を全部（RAM 定義の読み取りと縮小方式の両方）範囲に入れる | 開発者の判断（作業効率のため 1 つの PRD で片付ける） |
| D10 | 評価の上書き（`run.eval.[<tag>].env.ram_metric.*`）を拒否する専用の検査を、env の設定から消す。RAM 定義は今と同じく既定のキーからだけ読み、上書きの接頭辞の下の `ram_metric.*` は読まない。書かれた行は、Run の起動時のタグごとの読まれない設定キーの検査で、学習の開始前に止まる（文言は §6.1）。env の設定クラスは Run 側の階層（`run.eval.`）を名指ししない | 開発者の選択（§4.1 の F・J と比べた）。専用の検査を作った理由は、読まない上書きが黙って無視されることだった（§2.3）。今はタグごとの読まれない設定キーの検査が同じ誤りを同じ起動時に止めるので（§2.3）、同じ誤りを 2 つの仕組みで止めていた。書いた設定が読まれないことの検出は汎用の仕組みが持つ（PRD117 の D3）。評価の接頭辞で env を作る入口は Run の起動の中にしかない（§2.3）。env の設定は上書きの仕組みを知らずに済む（[AGENTS.md「汎用機構と利用側の責任境界」](../../AGENTS.md)） |
| D11 | 整理（D9）の前後で、既存の 5 語の結果、設定ファイルから来る ERROR の文言と時機、観測の画素、実効設定に記録する `resize_mode` の値（`adaptive_average`・`pixel_area` の文字列）と verbose ログの `resize_mode=` の表記を変えない。例外は D6 の語の一覧と、D10 の評価の上書きの誤り（文言と時機）の 2 つだけ。「文言を変えない」は設定ファイルから届く ERROR についての決まりで、届かない 2 つの検査（RAM 定義のキーの形の 2 回目の確認、番地の数値の読みの例外処理。§2.3）は消す | 課題 9ff3b7 の対応方針（ERROR の文言は変えない）と N2・N3 から導いた。縮小方式は PRD099 で「既定の学習系列を保つ」とした（[322 の地図](../wiki/categories/322_env_individual.md)、変えるときの条件）。実効設定の値はテストが文字列で見ている（[AtariEnv_test.cpp:1596](../../core/envs/atari1/src/AtariEnv_test.cpp#L1596)・[:1714](../../core/envs/atari1/src/AtariEnv_test.cpp#L1714)・[:1829](../../core/envs/atari1/src/AtariEnv_test.cpp#L1829)）。2 回目のキーの形の確認は、既定のキーの正規表現（419 行）を通ったキーだけが届くので当たらず、番地の例外処理は 0x と 2 桁の全体一致を通った文字列では起きない（§2.3） |
| D12 | ADR は書き換えず、新しい ADR も起こさない。用語集の「畳み方」の項に 3 語を足す | ADR 0046 は語を足すことを既存契約を壊さない追加とした（§2.1）。D10 は戻しやすく、ADR の 3 条件を満たさない。用語集は語を列挙している（§2.1） |
| D13 | 性能の受入は置かない | 1 フレームあたりの追加は、1 バイトの読みと、集合の更新かビットの数え 1 回だけである（ADR 0046 の「コストは無視できる」。§2.1） |
| D14 | レベルをまたいだときに部屋を区別する語（番地 2 つの組を数える語）は先送りにする。再開条件: 2 レベル目に入る Run が出たとき | 実験の結果が要る。0xB9 がレベルを表すかは確かめておらず（[課題 949e8a](../management/issues/issue-20261107-949e8a.md)、対応方針）、今の Run は 2 レベル目に届いていない（[Montezuma の wiki](../wiki/subjects/montezuma_revenge.md)）。それまでは 2 レベル目の部屋は 1 レベル目と同じ番号として数える（220 §4.9 に書く。D15） |
| D15 | 220 §4.9 は、畳み方の段落に 3 語の定義と D4 のマスクの書き方を足し、「`distinct_seen` は 1〜256」「ビットの意味は利用側の定義が持つ」「レベルをまたぐと部屋番号が重なる（D14）」を 1 文ずつ書く。上書きの文（「受け付けず、設定読み込み時に fail-fast する」「同じ形を入力に書くと fail-fast する」）は、「上書きは読まない。書いた行は Run の起動時に読まれない設定キーとして止まる」に改める（D10）。ゲーム別の表と Montezuma の注記は変えない | Atari.txt を変えないので、表は今の設定と合ったままにする（D8） |
| D16 | `AtariEnvConfig` は設定から読んで作るものとし、設定から読まずに組み立ててフィールドを書き換える使い方は認めない。縮小方式の再検査（`Validate` の中）は消し、縮小方式の文字列の検証は設定の読み取りの 1 か所だけにする（G5）。読み取りのときに検証しない項目（`game` の形、`frame_skip` など。§2.3）の検査を `Validate` に残すか読み取りへ移すか、`screen_size` の 2 つ目の検査をどちらへ寄せるかは、D11 の範囲で実装計画が決める | `AtariEnvConfig` はモジュールの内側の型で、そういう使い方は追跡しているコードに無い（§2.3）。factory の検証が `frame_skip` などを見ることに依存するテストがある（`[prd117][factory_config]`）ので、検査そのものは残す |

### 4.1 選ばなかった形

| 案 | 論点 | 退けた理由 |
|---|---|---|
| A. env の個体の一生の種類数（ゲームをまたいで仕切り直さない語） | 和集合（D7） | 作らない判断（N4）。加えて、lane ごとの値の `max.` は Run 全体の和集合より小さく出て（lane 3 が部屋 0、lane 9 が部屋 7 を初めて見つけても 1 つしか増えない）、集計の単位（実ゲーム 1 回）に例外ができる |
| B. metrics 側に、全 lane・全ゲームで見た値の種類数を累積する集計 | 和集合（D7） | 作らない判断（N4）。加えて、汎用の metrics の記法に時間方向の新しい演算が入り、Env の畳み方とは別の関心になる |
| C. ゲーム完了ログに見た値の集合を書き、後処理で和集合を数える | 和集合（D7） | 作らない判断（N4）。加えて、人が読むログの書式を道具が読むことになる |
| D. `mask_rise_count:<マスク>`（マスクの中が全部 0 から、どれかが 1 になった回数） | ビットの数え方（D2） | 鍵を 1 本持っているときに 2 本目を取った分（0xC1 が 2 から 6）を数えない |
| E. `mask_reach_count:<マスク>:<値>`（マスクを掛けた値が指定の値に入った回数） | ビットの数え方（D2） | D と同じく、すでに持っているときの取得を数えない。引数が 2 つになり、検証が増える。決まった並びを数えたいときは今の `reach_count:N` で足りる |
| F. 専用の拒否と Run 側の階層の名指しを残し、判定を正規表現 1 本に縮めるだけ | 評価の上書き（D10） | 時機と文言は変わらないが、env の設定クラスが Run 側の階層を知ったままで、課題 9ff3b7 の事象が残る。同じ誤りを汎用の検査と 2 つの仕組みで止める |
| G. 立つ側の語だけ足す | 落ちる側（D3） | 倒した回数と扉の回数を得点から推し量るままになる。作る手間は小さい |
| H. 9ff3b7 を取り込まない、または RAM の部分だけ取り込む | 範囲（D9） | 開発者が作業効率のため全部を 1 つの PRD で片付けると判断した。RAM の部分だけ取り込むには、課題を子 2 件に分ける手間がかかる |
| I. ビットの語を足さず、番地の側にマスクを付けて既存の語を使う（例 `0xC1&0x20 inc_count`） | ビットの数え方（D2・D3） | 1 ビットのマスクなら `inc_count`・`dec_count` で同じ値になり、マスクを掛けた値に `distinct_seen` や `reach_count` も使える。しかし複数ビットのマスクでは値の大小しか見ないので、鍵 4 本分を 1 つの定義で数えられない（0→6 は 2 本拾っても `inc_count` が +1、6→10 は 1 本使って 1 本拾っても `inc_count` が +1 で `dec_count` は 0）。番号の枠が 1〜4 しか無い（N1）ので、1 つの定義で種類を数えられる D2・D3 を採った |
| J. 専用の拒否を残し、各 env の設定が自分の上書きの接頭辞の下の `ram_metric.*` だけを探す（今の文言のまま） | 評価の上書き（D10） | Run 側の階層の名指しは消えるが、同じ誤りを汎用の検査と 2 つの仕組みで止める。残る利点は「この上書きは使えない」と直接言う文言と、Run の起動を通らない入口でも止まることで、そういう入口は無い（§2.3） |
| K. マスクを 16 進（`0x` に 2 桁）だけで書く | マスクの書き方（D4） | `0x` の付け忘れ（`20` を `0x20` のつもりで書く）を ERROR で見つけられ、判定が 1 通りになる。しかし文書と設定の例はビットの値を 10 進で書いており（鍵 2、剣 32）、その値をそのまま写せなくなる。開発者が 10 進と 16 進の両方を選んだ |

## 5. 仕様

### 5.1 新しい 3 語の定義（D1〜D5）

前のフレームの値を p、今のフレームの値を c、マスクを m、ゲームの開始の値を v0 とする（どれも 0〜255 の 1 バイト）。

| 語 | ゲームの開始での値 | 毎フレームの更新 | 値域 |
|---|---|---|---|
| `distinct_seen` | 1（v0 だけを見た） | c をこれまでに見た値の集まりに入れ、その種類数を値にする | 1〜256 |
| `bit_rise_count:m` | 0 | m の中で、p で 0・c で 1 のビットの数を足す | 0 以上の整数 |
| `bit_fall_count:m` | 0 | m の中で、p で 1・c で 0 のビットの数を足す | 0 以上の整数 |

例（Montezuma の 0xC1 で、鍵 = 2、剣 = 32 とする。鍵の 2 本目が 4 のビットになるのは未確認で、例のための仮定）:

| 0xC1 の前→後 | `bit_rise_count:30` | `bit_fall_count:30` | `bit_rise_count:32` | `bit_fall_count:32` |
|---|---|---|---|---|
| 0→2（鍵を拾う） | +1 | 0 | 0 | 0 |
| 2→6（鍵 1 本を持って 2 本目を拾う） | +1 | 0 | 0 | 0 |
| 2→34（鍵を持ったまま剣を拾う） | 0 | 0 | +1 | 0 |
| 34→32（扉で鍵を使う） | 0 | +1 | 0 | 0 |
| 32→0（剣で敵を倒す） | 0 | 0 | 0 | +1 |
| 0→6（同じフレームで 2 つ立つ） | +2 | 0 | 0 | 0 |

ゲームの開始で 0xC1 が 2 なら、そのビットは立った回数に数えない（開始の時点ですでに 1 なので、立つ遷移が無い。D5）。

`distinct_seen` の例（Montezuma の 0x83）: 部屋 1 で始まり、1→0→1→2→6→2 と動いたゲームは 4（1・0・2・6）。部屋 1 と 2 を 20 回行き来したゲームは 2。

### 5.2 書き方（D4）

```
AtariEnv.ram_metric.[montezuma_revenge].[rooms_visited] ?= 0x83 distinct_seen
AtariEnv.ram_metric.[montezuma_revenge].[key_get]       ?= 0xC1 bit_rise_count:30
AtariEnv.ram_metric.[montezuma_revenge].[sword_used]    ?= 0xC1 bit_fall_count:0x20
```

（例。Atari.txt にはこの行を入れない。D8）

## 6. 実装契約

### 6.1 設定と失敗の扱い（D4・D6・D10・D11）

設定キーは今と同じ `AtariEnv.ram_metric.[<game>].[<label>]` で、値の 2 つ目（畳み方）に 3 語が加わる。判定は上から順に当て、最初に当たった行で扱う。

| # | 畳み方の文字列 | 扱い |
|---|---|---|
| 1 | `max_seen`・`min_seen`・`inc_count`・`dec_count`・`distinct_seen` のどれかと完全一致 | 受け付ける |
| 2 | `reach_count:` で始まる | 今と同じ（0〜255 の 10 進。外れたら今の文言で ERROR） |
| 3 | `bit_rise_count:` か `bit_fall_count:` で始まる | `:` のあとのマスクを下の表で判定する。受け付けたらその語、拒否なら ERROR（文言は下） |
| 4 | それ以外 | ERROR（文言は下） |

マスクの文字列の判定（D4）。上から順に当て、最初に当たった行で扱う。

| # | マスクの文字列 | 扱い |
|---|---|---|
| 1 | `0x` で始まる | 残りが 16 進の数字（`0-9`・`A-F`・`a-f`）ちょうど 2 文字で、値が 1〜255 なら受け付ける。それ以外（`0x00`、`0x1`、`0x100`、`0xGG`、`0x` だけ）は拒否 |
| 2 | 1 文字以上で、すべて `0-9` | 値が 1〜255 なら受け付ける（`030` は 30）。値が 0 か 256 以上なら拒否 |
| 3 | それ以外（空、`0X1E`、`-1`、`+30`、`1E` など） | 拒否 |

ERROR の文言（`<キー>` は定義のキー、`<語>` は畳み方の文字列、`<値>` は定義の値の全体）:

```
<キー> has invalid reducer '<語>' in '<値>'; expected bit_rise_count:MASK with MASK in 1..255 or 0x01..0xFF.
<キー> has invalid reducer '<語>' in '<値>'; expected bit_fall_count:MASK with MASK in 1..255 or 0x01..0xFF.
<キー> has invalid reducer '<語>' in '<値>'; expected max_seen, min_seen, distinct_seen, inc_count, dec_count, reach_count:N, bit_rise_count:MASK, or bit_fall_count:MASK.
```

評価の上書き（D10）: env の設定は、上書きの接頭辞の下の `ram_metric.*` を読まず、検査もしない（既定のキー `AtariEnv.ram_metric.` の形の検査は今のまま）。書かれた行は、Run の起動でタグ `<tag>` の env の設定を検証した直後に、今の汎用の文言で止まる（`<相対キー>` はタグの下の相対のキーで、例 `env.ram_metric.[pong].[top]`）。

```
Unread config keys in run.eval.[<tag>]:
  <相対キー> = <値>
Expected keys read by this component: [...]
```

`Atari RAM metric override is unsupported` の ERROR は無くなる。

D11 により、ほかの ERROR（番地、番号付けの行、キーの形、ゲーム名、縮小方式、`screen_size`、`Validate` が見る項目）の文言と、設定ファイルから来るときの時機は変えない。届かない 2 つの検査は消す（D11）。次の 2 つは契約に効かない実装の詳細で、この制約の中で実装計画が決める。

- 読み取りのときに検証しない項目の検査と、`screen_size` の 2 つ目の検査の置き場（D16）。
- `GrayscaleResizer` が自分の寸法の前提（正の寸法、`pixel_area` の縮小先が元の短辺以下）を検査し続けるか。縮小方式の文字列の検査は持たない（§6.3）。

### 6.2 値の出口（D5）

新しい 3 語の値は、今の 5 語と同じく Step の結果の `ram_metric.[n]`、AuxData の `ram_metric.<label>`（整数）、ゲーム完了ログの `ram_metric:` の行に出る。出口を足さない。

### 6.3 クラス一覧

> 従えない事情があれば、実装する側が理由を実装計画に書いて変えてよい（質問しない）。

| 名前 | 区分 | 責務 | 置き場（機能グループ） |
|---|---|---|---|
| `RamMetricReducer` | 変更 | 畳み方の種別。`DistinctSeen`・`BitRiseCount`・`BitFallCount` を足す | AtariEnv |
| `RamMetricDefinition` | 変更 | 定義 1 行の番地・畳み方・引数。ビットの語のマスクを持つ | AtariEnv |
| `RamMetricState` | 変更 | 1 ゲームの畳み込みの状態。見た値の集まりとビットの数えを足す | AtariEnv |
| `AtariEnvConfig` | 変更 | Atari の設定の読み取り。RAM 定義の読み取りを 1 つの関数に任せ、畳み方の名前と種別を 1 つの表から引き、評価の上書きの専用の検査を持たない（Run 側の階層を知らない）。縮小方式は設定を読むときに 1 回だけ検証する | AtariEnv |
| `atari::ResizeMode` | 新規 | 縮小方式（`adaptive_average`・`pixel_area`）を表す。既存の `ImageDrawMode` と同じ形の名 | AtariPreprocess |
| `GrayscaleResizer` | 変更 | 縮小方式の文字列の検証を持たない（検証は設定の読み取りだけ） | AtariPreprocess |

### 6.4 移行対象

検索した語と範囲: `reach_count`・`dec_count`・`inc_count`・`min_seen`・`5 種`・`5 語`・`override is unsupported`・`env.ram_metric`・`resize_mode`・`Expected adaptive_average or pixel_area`・`GrayscaleResizer`・`AtariEnvConfig`（追跡しているファイルの全体を `git grep` で。`docs/memo/done/` と `docs/experiments/` の過去の記録は直さないので除いた）。

| 対象 | 直すこと |
|---|---|
| [AtariEnv.hpp](../../core/envs/atari1/src/AtariEnv.hpp)・[AtariEnv.cpp](../../core/envs/atari1/src/AtariEnv.cpp) | §6.3 のクラスの変更。D1〜D6・D10 |
| [AtariPreprocess.hpp](../../core/envs/atari1/src/AtariPreprocess.hpp)・[AtariPreprocess.cpp](../../core/envs/atari1/src/AtariPreprocess.cpp) | `GrayscaleResizer` が縮小方式の文字列を検証しないようにする（G5） |
| [AtariEnv_test.cpp](../../core/envs/atari1/src/AtariEnv_test.cpp) の `RAM metric reducers distinguish the initial value from transitions`・`AtariEnvConfig reads all RAM reducers and shared metric numbers`・`AtariEnvConfig rejects malformed RAM definitions before ROM loading` | 新しい 3 語とマスクの受け付け・拒否を足す。`rejects malformed` の `run.eval.[eval_target].env.ram_metric.[pong].[top]` の行（724 行）は、env の設定では ERROR にならなくなる（D10）ので、この表から外し、受入 4 のテストへ移す |
| 同 `GrayscaleResizer preserves adaptive average and uint8 rounding`・`GrayscaleResizer uses pixel overlap for fractional downsampling`・`Pixel area matches each fixed OpenCV reference image`・`Pixel area preserves numeric rules and all supported sizes`（[1721〜1804 行](../../core/envs/atari1/src/AtariEnv_test.cpp#L1721)）と、hidden の `Pixel area optimization preserves the fixed integer reference`（`[.][prd099-perf-exact]`）・`PRD099 resize prototype timings`（`[.][prd099-resize-probe]`） | 縮小器を作る呼び出しの縮小方式の引数を、`GrayscaleResizer` の新しい受け口に合わせる。期待値は変えない（受入 7）。hidden のテストも同じ実行体に入るので、直さないとビルドが通らない |
| 同 `Atari Run profile selects pixel area while evaluation keeps explicit overrides`（`[shipped_config]`、[1709〜1716 行](../../core/envs/atari1/src/AtariEnv_test.cpp#L1709)） | 設定の構造体の `resize_mode` を文字列と比べている 3 か所（1711・1713・1716 行）の比べる相手を、`AtariEnvConfig` の新しい持ち方に合わせる。実効設定の文字列を見る行（1714 行）は変えない（D11） |
| 同ファイルのテストの参照実装 `PixelAreaReferenceResizer` | 変えない（2026-10-24 の高速化前の実装を固定した参照のため） |
| [220_atari_env.jp.md](../design/220_atari_env.jp.md) §4.9 | D15。286 行の「畳み方は … の 5 種」を 8 種にする |
| [core/anet-core/testdata/prd082/README.md](../../core/anet-core/testdata/prd082/README.md) 3 行目 | 「`[ram_metric]` は、設定検証、5 種の畳み方、…を検証する」の「5 種」を 8 種にする（220 §4.9 が、畳み方を変えるときの検証手順として名指しする文書） |
| [220_atari_env.jp.md](../design/220_atari_env.jp.md) §4.3（185 行の `GrayscaleResizer(src_h, src_w, S, mode)`） | `mode` の説明を、`GrayscaleResizer` の新しい受け口に合わせる |
| [220_atari_env.en.md](../design/220_atari_env.en.md) §4.9 | 本書では直さない。英訳の工程（`anet-translate-docs`）が jp の変更を追う |
| [CONTEXT.md](../../CONTEXT.md)「畳み方」 | 3 語を足す（本書の起草で済ませた） |
| [ADR 0046](../adr/0046-atari-ram-metrics-bound-per-game-to-numbered-keys.md) | 変えない（D12） |
| `apps/runner/config/Atari.txt` | 変えない（D8） |
| [Montezuma の wiki](../wiki/subjects/montezuma_revenge.md) | 番号付き定義の記述が 2026-11-09 の 2・3 の追加を反映していないので、220 §4.9 に合わせる（本書の起草で済ませた） |

互換層は持たない（新しい語の追加と、内側の整理だけで、旧い契約のための別名・分岐は要らない）。過去の Run の記録と ADR は書き換えない。

## 7. 成功の測り方と受入

### 7.1 成功の測り方

- 新しい 3 語で書いた定義が、全ゲームの RAM 定義で受け付けられ、§5.1 の値を返す（受入 1〜3）。
- 畳み方の名前を書く箇所が、名前と種別の表の 1 か所になる（受入 6）。これで、畳み方を 1 つ足すときに直す箇所は、種別、名前と種別の表、畳み込みの処理の 3 か所（引数を持つ語は、引数の読みを加えた 4 か所）になる。課題 9ff3b7 の影響「reducer を 1 つ足すと 5 か所を直す」の解消を、名前の箇所が 1 つになったことで確かめる。
- 効果（Montezuma の群を、訪れた部屋の数と持ち物の取得・使用で比べられること）は、Atari の設定を決める側が新しい語を Montezuma の定義に置いた Run で確かめる。本書の受入には数えない（D8）。

### 7.2 受入

テストは既定のスイートで確かめる。ROM が要るテスト（`[rom]` のタグ）は、ROM のある環境で通す。

1. **3 語の値**（`[atari][ram_metric]`、ROM なし）: `RamMetricState` で次を確かめる。
   - ビットの 2 語: §5.1 の 0xC1 の表の各行を、別々のゲームとして与える（「前」の値でゲームを始め、「後」の値を 1 回観測する）。4 つの語の値が、その行の増分（+1 は 1、+2 は 2、0 は 0）になる。続けて、開始の値 2 で始めて 2 を 1 回観測したゲームで `bit_rise_count:30` が 0（開始の時点ですでに 1 のビットを数えない）。
   - `distinct_seen`: 開始の値 1 で始め、0・1・2・6・2 を順に観測したゲームで 4。開始の値 1 で始めて何も観測しないゲームで 1。開始の値 0 で始め、1〜255 を順に観測したゲームで 256。
2. **書き方の受け付けと拒否**（`[atari][ram_metric][config]`、ROM なし）: マスク `30`・`030`・`0x1E`・`0x1e`・`1`・`255`・`0xFF` を受け付け、`0`・`000`・`0x00`・`256`・`0x100`・`0x1`・`0X1E`・`-1`・`+30`・`1E`・空を拒否する。拒否の ERROR に §6.1 の文言（定義のキーと `expected bit_rise_count:MASK with MASK in 1..255 or 0x01..0xFF.` など）が含まれる。知らない語の ERROR に 8 語の一覧が含まれる。
3. **3 語の出口**（`[atari][ram_metric][config]`、ROM なし）: 3 語を使う定義を番号付けしたゲームで、番号が既知の番号に入り、実効設定に定義の行がそのまま記録される（今の `reports accepted RAM declarations` と同じ確かめ方）。
4. **評価の上書き**（`[atari][ram_metric][config]`、ROM なし）: `AtariEnv.game = pong` に `run.eval.[evalg].env.ram_metric.[pong].[top] = 0x8D max_seen` を足した設定で、Run の起動と同じ順に次を行う。(1) `MakeSubConfigData("run.eval")` でタグ `evalg` の設定を切り出す。(2) `AtariEnvFactory::ValidateConfig` を接頭辞 `run.eval.[evalg].env` で呼ぶ。ERROR にならない。(3) (1) の設定に `RequireAllKeysRead("run.eval.[evalg]")` を呼ぶ。`Unread config keys in run.eval.[evalg]:` と `env.ram_metric.[pong].[top] = 0x8D max_seen` を含む ERROR になる。あわせて、接頭辞なしの `AtariEnvConfig` も同じ設定で ERROR にならない。
5. **Run 側の階層の名指しの不在**: `git grep -n -F -e 'run.eval' -e 'run\.eval' -- core/envs/atari1/src/AtariEnv.cpp core/envs/atari1/src/AtariEnv.hpp` が 0 件（固定文字列の 2 つで、地の文と正規表現の中の書き方の両方を引く。コメントも数える。テストのファイルは範囲に入れない）。
6. **語の名前の置き場**: `git grep -n -E "max_seen|min_seen|distinct_seen|inc_count|dec_count|reach_count|bit_rise_count|bit_fall_count" -- core/envs/atari1/src/AtariEnv.cpp` の当たりのうち、コメントだけの行を除いたものが、すべて名前と種別の表の行である（ERROR の文言の途中に現れる語も当たる。ERROR の文言は語の名前を表から取る）。知らない語の ERROR の一覧の並び（`max_seen, min_seen, distinct_seen, inc_count, dec_count, reach_count:N, bit_rise_count:MASK, or bit_fall_count:MASK`）は、受入 2 のテストに書いた文字列と照らす。
7. **縮小方式の不変**（`[atari][preprocess]`・`[atari][prd099]`）: 既定のスイートの `[atari][preprocess]` と `[atari][prd099]` のテストが、期待値と ERROR の照合を変えずに通る。変えてよいのは、縮小器を作る呼び出しの縮小方式の引数と、設定の構造体の `resize_mode` と比べる相手だけ（§6.4）。`Atari resize configuration rejects unknown modes at reading` と `Pixel area rejects invalid sizes with the responsible configuration key` は書き換えずに通る。
8. **検証の 1 か所**: `git grep -n -F "Expected adaptive_average or pixel_area" -- core/envs/atari1/src ':!*_test.cpp'` が 1 件（テストのファイルは除く。テストは文言を照合するため）。`GrayscaleResizer` が縮小方式の文字列を検証しない: `git grep -n -F "GrayscaleResizer mode=" -- core/envs/atari1/src ':!*_test.cpp'` が 0 件（今は [AtariPreprocess.cpp:80](../../core/envs/atari1/src/AtariPreprocess.cpp#L80) に 1 件）。
9. **整理の不変**（D11）: 既定のスイートの、`[atari]` か `[prd117]` のどちらかのタグを持つテストのうち、§6.4 で直すと書いたもの以外が、書き換えずに通る。§6.4 で直すテストも、期待値と ERROR の照合は変えない（変えるのは §6.4 に書いた箇所だけ）。`my_config_json_` が `core/envs/atari1/src/` に 0 件（`git grep -n "my_config_json_" -- core/envs/atari1/src`）。ゲーム名の形の正規表現の文字列が AtariEnv.cpp で 1 行（`git grep -n -F "[a-z0-9]+(?:_[a-z0-9]+)*" -- core/envs/atari1/src/AtariEnv.cpp` が 1 件）。
10. **文書**: 220 §4.9 に 3 語の定義、マスクの書き方、D15 の 3 文があり、上書きの文が D15 のとおり改まっている。用語集の「畳み方」に 3 語がある。

## 8. Montezuma で使うときの材料（決定ではない。Atari の設定を決める側へ）

本書は Atari.txt を変えない（D8）。グリルで確かめた次の事実を、番号を決めるときの材料として残す。

- **定義の例**: 訪れた部屋の数 `0x83 distinct_seen`、剣を取った回数 `0xC1 bit_rise_count:32`、剣で倒した回数 `0xC1 bit_fall_count:32`、鍵を取った回数 `0xC1 bit_rise_count:30`、扉で鍵を使った回数 `0xC1 bit_fall_count:30`。マスクの 30（鍵 2・4・8・16）と、剣の 2 本目の 64 は OC_Atari の注釈で、確かめたのは鍵 2 と剣 32 だけである（§2.2）。
- **訪れた部屋の数だけでは左右の扉を見分けられない**: 部屋 1 で鍵を取り左の扉から部屋 0 へ入ったゲームも、右の扉で部屋 2 へ入ったゲームも 2 になる。実験記録は左右の扉を `room_max` が 1 か 2 以上かで見分けている（[Taiga ら寄せの記録](../experiments/default-dqn/atari/2026-11-08_montezuma-rnd-taiga-aligned.md)、29 行）。
- **`sword_get`（`bit_rise_count:32`）は `sword_only` の改良にあたる**: 鍵を持ったまま剣を拾った分も数える。`inv_max` の役目（剣を持ったか）は、剣を取った回数が 1 以上かで読める。
- **5 番以降の tag の枠が無い**: scalar の tag は、群の中の 50〜59 番が番号 1〜4 で埋まっている（[Atari.txt:1494-1508](../../apps/runner/config/Atari.txt#L1494)）。5 番以降を番号付けすると、trace の列は足せるが、scalar の mean・max は [PRD108](108_metrics_tag_numbering_10prd.md) の枠が入るまで出せない。
- **番号の結び直しと過去の Run**: Montezuma の番号 1〜3 は、すでに全ゲーム共通の意味（面クリア・ボス）から外れている（[220 §4.9](../design/220_atari_env.jp.md)）。[PRD082](done/082_atari_ram_progress_metrics_10prd.md) の「既存の番号の意味は変えない」は共通の意味についての決まりである。結び直すと、同じ列の意味が Run の前後で変わるので、過去の Run とはラベル（Run の `config/config_data.txt`）で読み分ける。
- **`distinct_seen` を部屋番号に使うときの未確認のこと**: `reset_game()` 直後の 0x83 が開始の部屋（1）か。部屋の切り替えや死亡の演出の途中で 0x83 が一時的に別の値を取らないか。220 §4.9 の「どこにも進まなかったゲームも 1」は `max_seen` での観測で、`max_seen` には出にくい一時的な値も、種類数にはそのまま入る。ROM での探索で確かめる。
- **未確認のこと**: 死んだときに持ち物が消えるか（消えるなら、落ちた回数に死亡の分が混ざる）。鍵を 2 本以上持ったとき、使った鍵のビットがどう動くか（ビットが詰め直されると、立つ側と落ちる側の両方が増える）。0xB9 がレベルを表すか（D14）。

## 9. issue の対応表

| ID | 題 | area | 現在のコードで確かめた根拠の位置 | 直す確定事項 | 確かめる受入 |
|---|---|---|---|---|---|
| [issue-20261107-949e8a](../management/issues/issue-20261107-949e8a.md) | Atari の RAM メトリクスに値の種類数とビットを数える畳み方が無く、Montezuma の訪れた部屋の数や持ち物を記録できない | 322_env_individual | [AtariEnv.hpp:49](../../core/envs/atari1/src/AtariEnv.hpp#L49)、[AtariEnv.cpp:250-261](../../core/envs/atari1/src/AtariEnv.cpp#L250)、[AtariEnv.cpp:304-326](../../core/envs/atari1/src/AtariEnv.cpp#L304) | D1〜D8・D12〜D15 | 1・2・3・10 |
| [issue-20261111-9ff3b7](../management/issues/issue-20261111-9ff3b7.md) | AtariEnvConfig のコンストラクタに RAM メトリクスの記法の解析と検証が並び、縮小方式は文字列のまま 3 つの層で 3 回検証している | 322_env_individual | [AtariEnv.cpp:378-449](../../core/envs/atari1/src/AtariEnv.cpp#L378)、[AtariEnv.cpp:286-303](../../core/envs/atari1/src/AtariEnv.cpp#L286)、[AtariEnv.cpp:521-551](../../core/envs/atari1/src/AtariEnv.cpp#L521)、[AtariPreprocess.cpp:66-87](../../core/envs/atari1/src/AtariPreprocess.cpp#L66) | D9〜D11・D16 | 4〜9 |

課題 949e8a の影響のうち「公開の結果（見つけた部屋の数）と同じ物差しで比べられない」は、D7（作らない判断）で残す。課題を解決にするときの記録に、そう書く。

## 10. 決めること（未決表）

なし（細部はグリルで決めた。D14 は先送りで、再開条件を書いた）。
