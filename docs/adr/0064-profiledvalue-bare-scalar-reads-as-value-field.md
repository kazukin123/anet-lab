# ProfiledValue の定数は裸のスカラー 1 行で書き、root の `.value` 記法は置かない

ProfiledValue を探索と学習の定数（学習率・weight_decay・γ など 90 か所超）へ広げると、定数のたびに `X.value = v` と 1 段増える。設定の器は平らなキーの集まりで `X` と `X.type` が共存できるので、root の定数は `X = v` の 1 行で書いて `value` 欄とし、root の `.value` 記法は置かない（段の中の定数だけ `.phase.[name].value`）ことに決める。既存の定数の行を書き換えずに ProfiledValue 化でき、2 つの記法を無くせば記法の衝突規則も要らない。眠る bare の規則と先送りした検査は PRD101 の D3・§7 による。

## Considered Options

- **常に `.value` を書く**: 記法は 1 つだが、定数の行が全部 1 段増え、既存の定数の行を書き換える。却下
- **bare と `.value` の両記法を許し、両方書いたら fail-fast にする**: 読み取りが default と override の 2 つの prefix を見るので、ベース＋部分上書きの普通の層構成まで fail-fast になる。却下
- **bare を書いたら type を constant に強制する**: baseline が bare で書いた定数を Run プロファイルが `.type = linear` で差し替える用法が fail-fast になる。却下
- **眠る bare が確定葉（`=`）なら fail-fast、既定葉（`?=`）なら眠ってよい**: 既定葉の情報を読み取りまで運ぶ配線が要り、env の確定葉の定数を Run プロファイルで schedule にする用法も止まる。先送り（眠った値に気づかずに Run を回した事故が起きたら起こす）
