# PRD067 Munchausen RL 実装・検証記録

対象: [PRD](067_MunchausenRL_10prd.md)、[承認済み実装計画](067_MunchausenRL_20impl.md)。
比較基準は `8f8104f`。実装開始時の HEAD は `1786103`。
本書はコミット対象の検証記録として、実行条件・結果・計測方法・制約をまとめる。生ログ・Run artifact・作業用スクリプトはローカル保管物であり、本書の参照先には含めない。

## 実装

- TD / QR / IQN の既定 OFF と3つの log-policy mode、FP32実空間の soft target、N-step の先頭 bonus、TBO の逆変換と最終変換。
- UQE が所有する減衰中の tau を使う D15 の soft 楽観ターゲット。QR の hard スコア計算を抽出し、IQN の hard 経路は維持。
- K3 Actor Qヒント、行動差し替え時の再 gather、初期優先度、PER ON/OFF の診断 readback。
- 診断7 tag、Atari profile、初期化ログ、3種類の ProfileRange、現行設計文書の同期。

## テスト契約との対応

| PRD項目 | 検証経路 |
|---|---|
| 1–4 | `UpdateFromSamples` の9組、forward記録、IQN N=4/M=3、tau生成順、終端とN-stepの独立double oracle |
| 5–7 | 同oracleのTBO ON/OFF、clip下限0、alpha=0・低温極限、CUDA BF16出力とFP32診断 |
| 8–10 | ConfigDataの休眠値検証、Double DQN、明示・copy経由Thompson、copy後overlay、UQE許可 |
| 11–12 | PER有効のpriority/clip→IQN→Munchausen→upper-tail pack、PER無効の診断、5 raw値とRunの7 tag |
| 13–14 | K3 codec、WithAction、非finite全列、旧K2拒否、TBOと終端を含む初期優先度oracle |
| 15 | 各modeの同一seed反復でlossとTD errorが完全一致 |
| 16 | 既存DQN・Replay/PERテストとRainbowの既定OFF |
| 17 | `UpdateFromBatch`、実ReplayBuffer、3 Learner×3 mode×PER ON/OFFのcapture行対応 |
| 18–19 | 抽出前のQR hard UQE固定値、point/tailと減衰tau、risk低温極限、QR pointの実hard選択 |
| 20 | Greedy / EpsilonGreedyの平均スコアと同一seed結果 |

テストのためのproduction APIは追加していない。fixture、test側subclass、既存の公開更新・設定・Actor経路を用いた。

## 実行環境・コマンド

Windows / MSVC、libtorch 2.11.0+cu130。Debugは単体・回帰検証、RelWithDebInfoは変更前後の学習に使用する。
すべてのCMakeビルドは `VsDevCmd.bat -arch=x64 -host_arch=x64` の後に実行する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug -j 4'
core\anet-core\bin\Debug\anet-core-test.exe "[dqn],[per],[replay_buffer],[tau],[replay_hint]" --rng-seed 67001
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-RelWithDebInfo --target AnetRLRunner -j 2'
```

## 学習検証

初回のOFF比較とON 5本には、[Atari設定](../../apps/runner/config/Atari.txt)を読み込むworkspaceと、以下の共通条件を用いた。再実行時のworkspace・Run名は実行先に合わせて指定する。

| 設定 | 値 |
|---|---|
| `app.$` | `app.batchrun>P1` |
| `app.batchrun.exp_exit_step` / `@vars.max_exp_step` | ともに `400000` |
| `A3.learner.update_warmup_steps` | `200000` |
| `train.seed` / `E1.game` | `1` / `breakout` |
| OFFの `run.$` / `backend.$` | `run.@v5_iqn_impala_x2` / `backend.@deterministic` |
| ONの `run.$` / `backend.$` | `run.@munchausen` / `backend.@non-deterministic` |
| ONの `A3.learner.use_double_dqn` | `false` |

ONの3 modeは `A3.learner.munchausen.log_policy_mode` で指定した。追加の2本はともに `target` modeとし、actor_approxでは `A3.learner.per_initial_priority_mode=actor_approx`、D15では `A3.use_optimistic_target=true` と `A3.train_policy.policy_type=UQE` を指定した。D15以外のON Runは `A3.use_optimistic_target=false` とした。

OFF比較は `37_agent_qtd/*` と `38_agent_loss/*` をtag別に集め、step昇順へ安定ソートして同step内の発行順を保持した。各点をキー順 `tag`, `step`, `value` のJSONにし、Pythonの `json.dumps(..., ensure_ascii=False, separators=(",", ":"), allow_nan=False)` と末尾LFでUTF-8化した系列のSHA-256を比較した。値の丸めやRun artifactの書き換えは行っていない。

既存の別実験を停止せず、ON RunはNsight SystemsのNVTX記録下で実行するため、throughputはこの実行環境での観測値として扱う。性能・成績の合否閾値は設けない。

## 検証結果

すべての必須受入条件を満たした。

- Debug 全体ビルド、RelWithDebInfo Runnerビルド: exit 0。
- Munchausen境界検証: 7,284 assertions / 12 cases、exit 0（CUDA BF16も実行）。
- DQN・Replay/PER・tau・hint回帰: 10,171 assertions / 230 cases、exit 0。228 cases成功、既存の`shouldfail` 2 casesは期待どおりの失敗。episode_start境界に関する既存テストであり、今回追加の失敗ではない。
- OFF: `8f8104f`と変更後のBreakout、seed 1、400k、warmup 200k、deterministic Runで、全18系列・各780点のSHA-256が完全一致。
- ON: 通常backendの5本がすべて400kを完走、exit 0。7診断tagとlossは全点finite、診断は契約範囲内。実効設定とresolution、初期化ログも全条件一致。

### ON Run一覧

| 構成 | Run | 受入結果 |
|---|---|---|
| `target` | `run_20260905-195946_tmp_smoke_067_target_breakout` | PASS |
| `online` | `run_20260905-200212_tmp_smoke_067_online_breakout` | PASS |
| `online_reuse` | `run_20260905-200536_tmp_smoke_067_online_reuse_breakout` | PASS |
| `target_actor_approx` | `run_20260905-200742_tmp_smoke_067_target_actor_approx_breakout` | PASS |
| `target_risk` | `run_20260905-200939_tmp_smoke_067_target_risk_breakout` | PASS |

actor_approxのsample比率は0.546875〜1.000000、Actor/Learner pair数は140〜256。設定のONだけでなく実利用を確認した。
D15 Runでは`use_optimistic_target=true`、`target_policy.policy_type=UQE`、初期化ログ`score_source=risk_biased`を確認。riskのsoft gapはfiniteを条件とし、平均スコアの非負条件を適用していない。

### 250k〜400k区間の計測

Nsight Systems 2026.3.1で`--trace=nvtx --sample=none --cpuctxsw=none`を使用した。TrainEvent通知はpipeline末尾2 batchが未発行となるため、elapsed scalarは399744で終わる。400kを外挿せず、`PipelineTrainRunner::DoStep`の終了時刻で計算した。128 transitions × 3,125回 = 400,000を確認し、250kだけ249984/250112の隣接時刻で補間した。区間の150,000 transitionsを実測時間差で割る。

| 構成 | 区間秒 | 実 throughput (exp/s) | exp_step_per_sec の区間サンプル平均 |
|---|---:|---:|---:|
| `target` | 24.542 | 6112.0 | 5590.8 |
| `online` | 26.939 | 5568.1 | 5283.1 |
| `online_reuse` | 23.876 | 6282.6 | 5557.8 |
| `target_actor_approx` | 26.386 | 5684.8 | 5506.1 |
| `target_risk` | 25.414 | 5902.2 | 5641.4 |

以下は同区間内のProfileRange平均時間（ms）。各存在phaseは586回。`—`は呼び出し0回。非同期CUDA完了時間単独ではなく、CPU上のrange経過時間である。

| 構成 | forward_target | forward_munchausen_online | munchausen_target |
|---|---:|---:|---:|
| `target` | 3.011 | — | 0.953 |
| `online` | 3.124 | 1.989 | 0.941 |
| `online_reuse` | 2.899 | — | 1.037 |
| `target_actor_approx` | 3.153 | — | 1.027 |
| `target_risk` | 2.578 | — | 1.025 |

別実験を停止せず、build/testと重なるRunもあるため、これらの値からmodeの優劣や改善率を断定しない。性能・成績の合否ゲートは追加していない。

## 独立な受入再確認（2026-09-05 21:52〜22:17）

上記とは別のバイナリ・別の Run 群で、受入基準 1・3 を launcher（`apps/12_batch_run.bat`）から再取得した。

- **OFF 等価性**: `AnetRLRunner_base.exe`（`munchausen` 文字列 0 件、17:20 ビルド）と実装後バイナリで
  `run.@v5_iqn_impala_x2` / 400k / warmup 200k / `backend.@deterministic` / seed 1 を各 1 本。
  `37_agent_qtd/*` + `38_agent_loss/*` の **18 系列・各 780 点で SHA-256 が完全一致**。
- **ON smoke 5 本**: 診断 7 tag が全 mode で `status=ok`、count 780、finite、契約範囲内。
  `03_clip_ratio` は全 mode 0（`scaled_logp ≈ −0.04` に対し `clip_value_min = −1.0` で下限に届かない）。
  `06_next_entropy` は 400k 時点で 1.36 前後（`ln A = 1.386` の 98%）で、方策はまだほぼ一様。
  3 mode で `01_scaled_logp_mean` が target −0.0417、online −0.0444、online_reuse −0.0469 と分離し、
  mode 切り替えが実際に別計算になっていることを確認した。

### env 固有の注意: D15 Run は `train_policy.policy_type` の明示が要る

初回は `A3.use_optimistic_target=true` だけを与えて `score_source=mean` になり、受入を落とした。
Atari は [Atari.txt](../../apps/runner/config/Atari.txt) の `A1.train_policy.policy_type = EpsilonGreedy` が
`@baseline` の `UQE` を潰しているため、`use_optimistic_target=true` は **EpsilonGreedy を target_policy へコピーする**。
`GetRiskScoreSpec()` は EpsilonGreedy に対して `nullopt` を返すので risk 経路に入らない。

PRD 受入基準 3 が指定するとおり `train_policy.policy_type=UQE` を併記すると
`score_source=risk_biased risk_tau=0.9 use_tail_mean=0` になり PASS した
（`run_20260905-221616_tmp_smoke_067_target_risk_breakout`）。
`use_optimistic_target` は fail-fast しないため、**env 側 A 層の `train_policy` を確認せずにこの腕を作ると黙って平均スコアで走る**。

### ProfileRange の取得条件

`ProfileRange` は NVTX / Tracy 専用で、Run artifact にもログにも残らない
（[profile.hpp](../../core/anet-core/include/anet/profile.hpp)。Tracy は `ANET_ENABLE_TRACY` のコンパイル時スイッチ）。
上記§「250k〜400k区間の計測」の値は Nsight Systems をアタッチして取得したものであり、
**通常の launcher 実行では再現できない**。区間別内訳が必要になった時点で外部プロファイラで別途取得する。

### 50M 実運用での throughput

400k 区間は経過 25 秒前後で warmup 直後のばらつきが乗るため、50M の実 Run で測り直した。
`ARM → 対照 → ARM` の順で連続実行されており、機械ドリフトを挟み撃ちできる。

| | ARM r1 | 対照（Munchausen OFF・Double OFF） | ARM r2 |
|---|---:|---:|---:|
| `exp_step_per_sec`（25-50M） | 4913 | 5062 | 4880 |
| 総時間 | 2.807h | 2.669h | 2.761h |
| eval1 `game_len` | 3785 | 3266 | 3734 |

ARM 平均 4896 対 対照 5062 で **−3.3%**。ARM 2 本の差は 0.7% なのでこの窓のドリフトは小さい。
ただし ARM は成績が上がった結果 **eval エピソードが 14% 長い**ので、この差には eval コスト増が含まれる。
Learner の正味コストはこれより小さい。`target` mode の `obs ∥ next_obs` 2B forward と、
Double DQN OFF で消える next_obs online forward が概ね相殺している。


## 追補: Actor Qヒント設定の共通化

実装後の合意に従い、`MunchausenConfig` を `LearnerConfig` の外へ移し、`ActorQHintConfig::munchausen` として保持する形へ修正した。設定キー・既定値・数値式は変更していない。ヒント全体は `emit_actor_q_hint`、Munchausen計算は `munchausen.enabled` が制御し、Actorは `log_policy_mode` を参照しない。PRD・ADR 0036・現行設計文書も同期した。

- `VsDevCmd.bat` 経由の `cmake --build --preset x64-Debug -j 4`: exit 0。
- `anet-core-test.exe "[munchausen],[actor],[actor_initial],[rainbow]" --rng-seed 67001 --reporter compact`: 7,517 assertions / 43 cases、すべて成功。
- TD / QR / IQN、Munchausen ON/OFF、TBO ON/OFFのActorについて、3 mode間のヒント完全一致とforward 1回を確認した。

初回実装時のOFF比較とON学習5本は、この設定構造変更前の実行記録である。この追補では設定構造とActorのmode非依存性を関連テストで検証し、学習Runは再実行していない。
