# ReplayBufferのframe履歴開始はPush時のepisode_startで確定し、n-step確定metadataから導出しない

`DefaultReplayBuffer`はframe stackを保存せず、sample時に開始slotから過去へ`stack_count - 1`件遡って再構成する。
再構成の境界判定には、slotに保存された`terminal && actual_n_steps <= 1`を使っていた。この2値はn-step結果の確定時に書かれる派生値であり、
境界の一次情報ではない。整合性アッセイと公開APIの最小再現で、この代理指標が2通りに破れることを確認した。

- laneの先頭を表す印がどこにも無く、初回満杯までは末尾の未書込みslotの初期値`terminal=true / actual_n_steps=0`が偶然の境界として働いていた。
  末尾が書込み済みになる初回満杯ちょうどでこの境界が消え、起動時paddingへ物理ring末尾の新しいframeが混入する。
- 観測は書込み済みでもn-step結果が未確定のslotは、初期値または旧世代の値のまま境界に見える。bootstrap先の`next_obs`は
  sample対象より新しいslotを含むため、正当な過去frameを捨てて最新frameで埋め直す。wrap後は旧世代の実終端が同じ誤認を起こす。

**frame履歴の境界は、Push時に入力`BatchState::episode_start`からslot単位の「履歴開始」として確定し、extractorはそれだけを読む**ことを決定する。
END（`done` / `truncated`）はn-step queueの一時値のままslot単位には保持しない。`terminals_` / `actual_n_steps_`はn-step出力としてだけ残す。
`DefaultReplayBuffer::Push`は「laneの初回の実pushは`episode_start == true`、以降は次の実pushの`episode_start`が直前の実pushの`done || truncated`と一致、
終端観測dummyは対象外」を全laneのpreflightで検証し、違反は`ANET_SYSTEM_ERROR`で止める。

理由は次の3つである。第一に、frame stackの再構成は「この観測に同一episodeの前任がいるか」という過去側の問いであり、過去側の信号である
`episode_start`が直接の答えになる。ENDから境界を作ると「1つずらす + lane先頭の初期状態」という導出が挟まり、導出には仮定が隠れる。
lane先頭の起動padding汚染は、その初期状態が未書込みslotの値へ暗黙に依存していた事故そのものだった。
第二に、slot単位で境界を消費するのはextractorだけである。n-stepはENDをPush時に消費し終わり、後からslot単位で参照しない。
したがってslotに持つ1件は履歴開始で足り、ENDを重複して持つ必要がない。Env継ぎ目の契約（ADR 0034）が`episode_start(t+1) == done(t) || truncated(t)`を
保証するので、入力に含まれる冗長性は「一致を仮定する」のではなく「Pushで検証する」ことに使う。
第三に、Actor側の`StackerActionContext`は既に`episode_start`で stack を初期化している。学習時のstackが同じ信号の同じ関数になり、
行動時と学習時の境界の定義が構成上一致する。

一般実装も同じ原則に立つ。DeepMind DQNの`TransitionTable`は`add`でterminalを即時に書き`concatFrames`でterminalを越えたframeをzero-outする
（1-stepなのでENDがPush時に確定している）。Dopamine v2の`circular_replay_buffer`は`add()`で直前がterminalなら`stack_size - 1`個のzero transitionを先に挿入し、
n-step returnはsample時に即時terminalから計算する。Gymnasiumの`FrameStackObservation`は`reset()`だけで再初期化し、SB3の`StackedObservations`は
vec envの`dones`から導出する。境界は書込みまたはresetの時点で確定した情報から作られ、後から確定する派生値を使う実装はない。
anetはn-step導入で`terminal`が派生値になった時点でこの前提が崩れていた。

## Considered Options

- **lane状態からSTARTを導出する（Dopamine型）**: 「直前がENDだったか」をlaneに持ち、Push時に履歴開始へ写す。slot当たりの費用は同じ1 byteだが、
  導出とlane先頭の初期状態がextractor側の暗黙前提として残り、Actor stackerとは別信号になる。入力`episode_start`との不整合を検出できない。却下。
- **Push時に暫定値（`terminal=false`や`terminal=done, actual_n_steps=1`）を書く**: 未確定slotの誤認は消えるが、境界がn-step出力の派生述語のままで、
  Builderやunrollの変更で再び壊れ得る。lane先頭の問題は別途必要になる。却下。
- **`episode_start`とdone/dummyの和集合を境界にする**: 源が2つになり、done無しの`episode_start`でstackだけ切れてn-stepは切れない非対称が残る。
  冗長性を検証に使えない。却下。
- **done無しの`episode_start`を正規入力として受理し、n-stepも切る**: Env継ぎ目の`ValidateEpisodeStructure`がその入力を毎stepで拒否するため、
  productionでは到達不能な入力のための新機構になる。直前遷移をterminal扱いするか破棄するかの追加裁定も要る。却下。
- **後続遷移のn-step結果がすべて確定するまでsampleを遅らせる**: 既存のready/sampleable契約を余分に狭め、episode開始直後の経験の供給が遅れる。却下。
- **sample結果へgenerationを運びextractorでassertする**: ADR 0024で保留した防御。今回の欠陥は同世代の未確定metadataでも起きるため、世代検証では防げない。
  引き続き別判断とする。

## Consequences

- slot当たり1 byte相当の履歴開始と、lane当たり1 boolの検証状態が増える。Atari 1M slotで約1 MiB。追加のTensorコピー、GPU転送、network forwardは増えない。
- extractorは`terminals_` / `actual_n_steps_`を境界に使わない。未書込みslotの初期値は境界の意味を持たなくなる。
- ADR 0024の「安全な開始indexならextractorは変更しない」という判断を更新する。history marginとready/sampleable rangeの分離は維持する。
  ring上書きの欠損はhistory marginが除外し、episode由来の不足は履歴開始からpaddingする、という分担は変わらない。
- `DefaultReplayBuffer::Push`は`state.episode_start`を必須入力として読む。PRD 050 D15以来未裁定だった「doneを伴わないepisode_start」の期待失敗2件は
  契約外入力として決着し、fail-fastテストへ置き換える。初回pushの`episode_start`を省略していたテストを移行する。
- Prefetchのwrite-behind Pushで起きた契約違反は、既存契約どおり次の同期境界で再送出される。
- 設計書150の「未書込領域または保存済みterminalによる境界」の記述を履歴開始へ書き換え、`CONTEXT.md`に用語「履歴開始」を追加する。
- 詳細設計は`docs/memo/done/078_replay_frame_history_integrity_10prd.md`。
