# WARN は利用者が直すべき状態に限り、読まれない設定キーは NoCare にせず検出する

Logs pane が未確認の警告を数えて印を出すようになったが、意図して止めた評価タグの通知で普通の Run にも毎回 WARN が出て印が意味を失い、一方で、書いた設定が読まれないことは NoCare として黙って通り、block の設定キーが読まれないまま意図と別の Network で Run が回った（[PRD117](../memo/117_warn_policy_and_unread_config_10prd.md)）。**WARN は「利用者が確かめて、設定・コード・実行環境を直すべき状態」に限って、意図した休止と、設定どおりに起きた結果（checkpoint の値を今の設定で置き換えた、など）の通知は INFO にし、読まれない設定キーは、実体化したカタログ項目では構築時に止め、それ以外は Run の構築の終わりに WARN で一覧する**ことを決定する。

## Considered Options

- **休止の通知を WARN のまま対象ごとに 1 回にする**: 普通の Run で印が付き続け、0 件にするには休止のたびに参照している設定を消して回ることになる。却下。
- **checkpoint を読むときの通知を、無い部分の初期化も含めてすべて INFO にする**: 意図した移行か checkpoint の取り違えかを部品が見分けられず、取り違えると学習していない Network で評価した結果が黙って出る。初期化は WARN のままにする。却下。
- **構築が終わったあとの異常も一律に止める**: 出力の一部だけが欠ける異常で、長い Run が失われる。WARN を出して続ける。却下。
- **読まれない設定キーの一覧を最初から ERROR にする**: 読まれなくても誤りではない設定キー（在庫）の見分けが足りているかを Run で確かめる前に止めると、正しい設定の Run が起動できなくなる。実績を見てから決める。先送り。
- **既定キー（`?=`）を一覧の対象から外す**: 共通の設定の綴り違いと旧キーが黙って残る。却下。
- **使わなかったカタログ項目も Run のたびに検査する**: 書きかけの定義が印を付け続けるか、関係のない Run を止める。出荷設定の検査のテストで見る。却下。

## Consequences

- [ADR 0027](0027-eval-definition-schedule-separation.md) と [ADR 0047](0047-mirror-eval-instance-subscribes-dormant-trace-and-trace-rows-carry-timestamp.md) の「dormant タグを参照する metrics は WARN」と、[ADR 0053](0053-checkpoint-load-applies-current-config-to-optimizer-options.md) の「checkpoint と違った項目は WARN で示す」は、レベルの部分だけを INFO に改める。skip する振る舞いと、options を今の設定で置き換える判断は残る。
- ADR 0027 の「旧位置キーの検出は NoCare」と、クリーンブレーク方針の「移行後の旧キーは NoCare」は、「ほかの読まれない設定キーと同じに見つかる」に変わる。旧キー専用の検出を作らない決まりは変わらない。
