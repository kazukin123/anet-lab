# ProfiledValue の定数は裸のスカラー 1 行で書き、root の `.value` 記法は置かない

ProfiledValue を探索と学習の定数（学習率・weight_decay・γ など 90 か所超）へ広げると、定数のたびに `X.value = v` と 1 段増える。設定の器は平らな map で `X` と `X.type` が共存できるので、root の定数は `X = v` の 1 行で書き、これを `value` 欄とすることに決める。root の `.value` 記法は無くし（書かれていたら fail-fast。現用の設定に無い）、段の中の定数だけ `.phase.[name].value` で書く。`type` の既定は constant なので定数は 1 行で済み、既存の定数の行は書き換えずに ProfiledValue 化できる。`.type` が constant でない層の下に書いた bare は `value` 欄として眠る（ProfiledValue の既存の dormant の契約と同じ）。この眠る規則は文書に書くだけにし、検査は足さない。

## Considered Options

- **常に `.value` を書く**: 記法は 1 つだが、定数の行が全部 1 段増え、既存の定数の行を書き換える。却下
- **bare と `.value` の両記法を許し、両方書いたら fail-fast にする**: 読み取りが default と override の 2 つの prefix を見るので、ベース＋部分上書きの普通の層構成まで fail-fast になる。`.value` を無くせば規則そのものが要らない。却下
- **bare を書いたら type を constant に強制する**: baseline が bare で書いた定数を Run プロファイルが `.type = linear` で差し替える用法が fail-fast になる。却下
- **眠る bare が確定葉（`=`）なら fail-fast、既定葉（`?=`）なら眠ってよい**: 既定葉の情報を読み取りまで運ぶ配線が要り、env の確定葉の定数を Run プロファイルで schedule にする用法も止まる。先送り（眠った値に気づかずに Run を回した事故が起きたら、静的検査か実行時検査を起こす）

詳細契約は `docs/memo/101_exploration_schedule_profiled_value_10prd.md` の D3・D4、先送りは同 §7。
