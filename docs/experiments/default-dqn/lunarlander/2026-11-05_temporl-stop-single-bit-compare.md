# TempoRLのSTOP×Singleと無効のbit比較

## 条件と判定方法

| 項目 | 内容 |
|---|---|
| 目的 | STOP×Single（行動にskipを使わず、skipのlossの勾配も共有部へ返さない）は、[Singleの7群比較](2026-10-29_temporl-auxiliary-only-stage1.md)のSTOP対照（`t104r1_ctl`）で同seedの無効を3本ともAUCで改善した。skipの学習がQ側へ届く経路があるのか、差が乱数の順番などにとどまるのかを、同じseedのbit比較で分ける |
| 対象 | LunarLander、DefaultDQNAgent、IQN＋Munchausen（`log_policy_mode=target`、`target_taus`・`current_taus`は`sample_mode=random`の32τ）、J=4、256 env、batch256、replay ratio8、`update_warmup_steps=8,192`、`learner.skip.priority_lambda=0`。全本seed 1 |
| 実行 | 2026-11-05 14:50〜17:15 JST。11本を順次実行した。終了コードは全本0、stderrは全本空、logにWARN・ERRORなし |
| 決定的モード | 全本で`backend.cudnn_benchmark=false`、`backend.cudnn_deterministic=true`、`backend.deterministic_algorithms=true`、`backend.deterministic_warn_only=false` |
| 実行体 | パッチなしは2026-11-03 22:02のRelease、SHA256 `0408933b25dd502867e1ec2fda00a35c440bbfa2d0f41923c0a9f94bece92d96`。パッチ版は、下のパッチを当てた2ファイルだけを同じオプションでコンパイルし、同じビルドの残りのobjとリンクしたもの、SHA256 `f12293694cd8a0fbfa26f8826449fc6c60a479ed05c0e0dab34421c5cf48752e`。パッチは診断用で、ソースには入れていない |
| bit比較 | 各Runの`metrics.jsonl`の、同じtagの(step, 値)の列を先頭から完全一致で比べた。対象は両方のRunにあるtagで、`90_perf/`の4 tagは実時間に依るので外した。`50_agent_skip/`のtagをskip側、それ以外をQ側と呼ぶ。STOPだけにある`50_agent_skip/`の18 tagと`[greedy]`の評価の4 tagは、無効との比較に入らない |
| 報酬の指標 | `21_eval/02_policy_reward`（UQE τ0.25、固定32τ、1 episode、200 learn stepごと）の点の平均。Run全体の値は[Singleの7群比較](2026-10-29_temporl-auxiliary-only-stage1.md)のAUCと同じ計算 |
| 比較上の注意 | DropMergeの学習RunとRTX 5090を共有して実行したので、実時間は群の効果に使わない |

| 群・Run識別子 | 実行体 | skip | `learner.skip.loss_weight` | 範囲（exp step） |
|---|---|---|---:|---:|
| 無効 `sv_off_a`・`sv_off_b` | パッチなし | 無効 | — | 10万 |
| 無効＋`[greedy]` `sv_off_g` | パッチなし | 無効 | — | 10万 |
| STOP `sv_stop_w1`・`sv_stop_w1L` | パッチなし | 有効 | 1 | 10万・100万 |
| STOP重み0 `sv_stop_w0`・`sv_stop_w0L` | パッチなし | 有効 | 0 | 10万・100万 |
| パッチ版の無効 `sv_pb_off`・`sv_pbL_off` | パッチ版 | 無効 | — | 10万・100万 |
| パッチ版のSTOP `sv_pb_stop`・`sv_pbL_stop` | パッチ版 | 有効 | 1 | 10万・100万 |

STOPの群は`learner.skip.feature_gradient=stop`、`learner.skip.target_window=action_change`、すべてのActorで`skip.max_length_mode=single`、`[greedy]` sessionつきである。`sv_stop_w1L`と`run_20261029-162638_t104r1_ctl_s1`の実効configで違うのは、決定的モードの3キー、あとから足された`skip.nn_forward_mode`（5 Actor）、metricsとappのキーで、ほかに実行体のビルドが違う。

## 現時点のまとめ

1. **skipの学習の中身はQ側へ届いていない。** パッチなしのSTOP×Singleで`loss_weight`を1と0にすると、100万exp stepまでQ側の239,083点が全点一致した。評価の報酬とTrainの報酬の点を含む。差があるのは`50_agent_skip/11_loss`・`12_td_abs_mean`・`15_grad_norm`の3 tagだけで、重み0のgrad_normは全点0、最後のlossは0.030対0.493だった。
2. **乱数とtargetのforwardを分けると、STOP×Singleは無効と全点一致した。** 窓長の抽選と窓のtargetのτを別の乱数列にし、窓のtargetを別のforwardで計算するパッチで、100万exp stepまで共通の103 tag・239,213点が一致した。パッチ版のSTOPでもskipは学習しており、lossは5.20から最後の500点の平均0.038へ下がり、grad_normは30,984点すべて0でなかった。skipのNetwork、4 groupのoptimizer、skipのlossのbackward、Actorのskipのforwardがあっても、一致は崩れなかった。
3. **パッチは無効の経路を変えていない。** パッチなしの無効とパッチ版の無効は10万exp stepまで21,987点が全点一致した。`[greedy]` sessionを足しても一致は崩れなかった（21,987点）。
4. **パッチで分けた差は更新の分布を変えない。STOP×Singleは無効と同じ分布の別の標本で、STOP対照の平均57.25対無効20.87の差は偶然である。** 根拠は下の「差の経路」のコードの読みと、2の全点一致である。多数のseedで分布を測ったのではない。
5. **同じseedで順番だけが違うと、差の向きが区間で入れ替わる。** パッチなしのSTOPからパッチ版の無効を引くと、0.25〜0.5Mが−43.10、0.5〜1Mが+19.21、0.25〜1Mが−1.56だった。10/29のRunでSTOPと無効の差が最も大きかった0.25〜0.5Mが、このseed 1の対では逆向きである。
6. **無効の帯はSTOP対照を含めて広がる。** Run全体のAUCで、無効5本とSTOP対照3本の8本は−6.83〜102.58で、無効5本だけの−6.83〜68.42より広い。Single行動変化窓の18.08〜73.04はこの帯の中にある。同seedの無効を3本とも改善したことはSTOP対照でも偶然で起きたので、行動変化窓の根拠にならない。通常TempoRLの131.40〜143.39は、3本とも帯の上端より上である。
7. **STOP対照の差は、偶然で起きる範囲に入る。** STOP対照3本と無効5本の並べ替えで差が観測以上になる割合は、Run全体が10/56、0.25〜0.5Mが1/56だった。0.25〜0.5Mは結果を見てから選んだ区間で、選び方を補正していない。
8. **差は最初の学習から出て、warmupの教材は同じだった。** 無効とSTOPはActor側の`33_agent_action/21_iqn_policy_margin_mc_ratio_ema`が8,448 exp stepまでの33点で一致し、8,704で割れた。Singleのスキップ長と判断率は全点1である。warmupの間の教材の違いでは、STOP×Singleの差を説明できない。
9. **決定的モードはGPUを共有しても再現した。** 同じ設定の無効2本は10万exp stepまで21,987点、STOPの10万と100万の前半は重み1・0とも39,038点が全点一致した。どれもDropMergeの学習と同時に実行している。

## bit比較の結果

| 比べたもの | 実行体 | 範囲（exp step） | 共通のtag | 比べた点 | 結果 |
|---|---|---:|---:|---:|---|
| 無効 `sv_off_a` 対 `sv_off_b` | パッチなし | 10万 | 101 | 21,987 | 全点一致 |
| 無効 `sv_off_a` 対 無効＋`[greedy]` `sv_off_g` | パッチなし | 10万 | 101 | 21,987 | 全点一致 |
| STOP `sv_stop_w1` 対 `sv_stop_w1L`の前半 | パッチなし | 10万 | 123 | 39,038 | 全点一致 |
| STOP重み0 `sv_stop_w0` 対 `sv_stop_w0L`の前半 | パッチなし | 10万 | 123 | 39,038 | 全点一致 |
| STOP `sv_stop_w1` 対 重み0 `sv_stop_w0` | パッチなし | 10万 | 123 | 39,038 | Q側は全点一致。skip側の3 tagだけ差 |
| STOP `sv_stop_w1L` 対 重み0 `sv_stop_w0L` | パッチなし | 100万 | 125 | Q側239,083、skip側182,441 | Q側は全点一致。skip側の3 tagだけ差 |
| 無効 `sv_off_a` 対 STOP `sv_stop_w1` | パッチなし | 10万 | 101 | 21,987 | 67 tagに差、34 tagは一致 |
| 無効 `sv_off_a` 対 パッチ版の無効 `sv_pb_off` | パッチなし／パッチ版 | 10万 | 101 | 21,987 | 全点一致 |
| パッチ版の無効 `sv_pb_off` 対 パッチ版のSTOP `sv_pb_stop` | パッチ版 | 10万 | 101 | 21,987 | 全点一致 |
| パッチ版の無効 `sv_pbL_off` 対 パッチ版のSTOP `sv_pbL_stop` | パッチ版 | 100万 | 103 | 239,213 | 全点一致 |

`sv_pbL_off`と`sv_pbL_stop`で一致したtagの例（点数、最後の値）:

| tag | 点 | 最後の点 |
|---|---:|---|
| `21_eval/02_policy_reward` | 155 | 994,048で26.53 |
| `21_eval/01_target_reward` | 155 | 994,048で224.80 |
| `20_eps/10_train_reward` | 2,332 | step 3,515で−22.72 |
| `38_agent_loss/01_loss` | 3,873 | 999,680で0.05455 |
| `36_agent_munchausen/01_scaled_logp_mean` | 3,873 | 999,680で−0.05360 |

skipが学習していたことの確かめ（100万exp step。最後の500点の平均）:

| tag | パッチ版のSTOP | パッチなしのSTOP |
|---|---:|---:|
| `50_agent_skip/11_loss` | 0.03795 | 0.04004 |
| `50_agent_skip/12_td_abs_mean` | 0.11745 | 0.11756 |
| `50_agent_skip/13_sample_k_mean` | 1.5112 | 1.4584 |
| `50_agent_skip/14_kmax_mean` | 2.0187 | 1.9162 |
| `50_agent_skip/15_grad_norm` | 0.1015 | 0.0917 |

## 差の経路

無効とパッチなしのSTOPの差は、次の3つである。

| 差 | パッチなし | パッチ版 | 更新の分布への影響（コードの読み） |
|---|---|---|---|
| 窓長の抽選 | replayのsamplerと同じ乱数列（`DrawWindowLengths`） | 別の乱数列 | サンプルを選ぶ確率は変わらない。乱数列の位置が進むだけである |
| targetのτ | 通常と窓を並べた行数で、同じ乱数列から引く | 窓の分は別の乱数列 | `MakeNormalizedTauPositions`は`sample_mode=random`で行と列を独立に引くので、1行の分布は行数によらない |
| targetのforward | current B行と、通常と窓の終点2B行を合わせた3B行を1回で計算する | 通常と窓を2B行ずつ別々に計算する | 各行の計算は行ごとに閉じている。このNetworkの正規化はLayerNormだけで、batch内の行をまたぐ層は無い。`MakeScaledLogPolicy`と`MakeMunchausenTargetTerms`も行ごとの計算で、batch平均を取るのは診断の値だけである |

skipのTD誤差は`CombinePriorityErrors`で`priority_lambda > 0`のときだけPERの優先度へ入り、このRunでは0である。skipのNetworkと方策の乱数の種は名前つきで作るので、skipの有無でreplayとLearnerの種はずれず、Q側の乱数列も進まない。Singleはスキップ長を常に1にする。ActorのskipのforwardはSingleでも動くが、その結果は長さの選択に使わない。

## 同じseedで順番だけが違うときの報酬

seed 1、決定的モード、`21_eval/02_policy_reward`。パッチ版のSTOPはパッチ版の無効と全点同じ値である。

| 区間（exp step） | 点 | パッチ版の無効 | パッチなしのSTOP | 差 |
|---|---:|---:|---:|---:|
| 0〜1M | 155 | 24.87 | 20.74 | −4.13 |
| 0.25〜0.5M | 39 | 42.64 | −0.46 | −43.10 |
| 0.5〜1M | 78 | 70.23 | 89.44 | +19.21 |
| 0.25〜1M | 117 | 61.03 | 59.48 | −1.56 |

## 過去のRunとの関係

`21_eval/02_policy_reward`の平均。どのRunも決定的モードではなく、2M。

| 群・Run識別子 | Run全体 s1／s2／s3 | 0.25〜0.5M s1／s2／s3 | 0.5〜1M s1／s2／s3 | 1〜2M s1／s2／s3 |
|---|---|---|---|---|
| 無効 `t98_ll_off` | 68.42／2.32／−6.83 | −19.98／−72.14／−43.08 | 64.23／37.61／6.74 | 135.00／129.88／100.89 |
| 無効 `t104r1_off` | 64.93／4.50／−6.83 | −22.96／−23.40／−43.08 | 25.55／37.82／6.74 | 145.86／132.33／100.89 |
| STOP対照 `t104r1_ctl` | 102.58／50.95／18.20 | 32.19／85.03／−18.46 | 96.15／114.65／16.93 | 157.68／138.46／111.61 |
| Single行動変化窓 `t104r1_ac` | 73.04／62.65／18.08 | 52.11／15.29／−116.11 | 75.01／72.39／35.18 | 101.34／170.51／153.32 |
| 通常TempoRL `t104r1_mrg` | 131.40／143.39／140.51 | 34.65／113.88／123.68 | 135.76／141.39／151.29 | 184.97／212.49／209.26 |

無効は5本と数える。`t98_ll_off_s3`と`t104r1_off_s3`は、ビルドが違うが評価の312点がすべて一致した。s1とs2は14,848 exp stepの評価点から違う。

| 区間 | 無効5本とSTOP対照3本の帯 | STOP対照の平均−無効の平均 | 並べ替えの割合 |
|---|---|---:|---:|
| Run全体 | −6.83〜102.58 | +30.57 | 10/56＝0.18 |
| 0.25〜0.5M | −72.14〜85.03 | +69.23 | 1/56＝0.018 |
| 0.5〜1M | 6.74〜114.65 | +41.52 | 5/56＝0.089 |
| 1〜2M | 100.89〜157.68 | +7.13 | 18/56＝0.32 |

並べ替えの割合は、8本から3本を選ぶ56通りのうち、選んだ3本の平均と残り5本の平均の差が観測以上になる割合である。統計的有意差の検定ではない。

## パッチ

元のソースは、`replay_buffer_impl.cpp`がコミット`f554b3f0`、`dqn_based_agent.cpp`がコミット`5ee57331`の版で、どちらも実行体のビルドより前である。

```diff
--- a/core/anet-core/src/replay_buffer_impl.cpp
+++ b/core/anet-core/src/replay_buffer_impl.cpp
@@ -127,8 +127,11 @@ private:
     int limit_;
 };
 
-torch::Tensor DrawWindowLengths(const torch::Tensor& max_lengths, const torch::Generator& generator)
+torch::Tensor DrawWindowLengths(const torch::Tensor& max_lengths, const torch::Generator&)
 {
+    // stop-vs-off 診断: 窓長の抽選を sampler の乱数列から切り離す。
+    static anet::RandomGenerator window_random(anet::seed_t(0x53564231ULL));
+    const auto generator = window_random.GetTorchGenerator(torch::kCPU);
     // 同じ上限をまとめて抽選し、batch件数分の小さいTensor確保を避ける。
     std::map<int64_t, std::vector<int64_t>> groups;
     const auto maximum = max_lengths.accessor<int64_t, 1>();
--- a/core/anet-core/src/dqn_based_agent.cpp
+++ b/core/anet-core/src/dqn_based_agent.cpp
@@ -4637,35 +4637,30 @@ TargetEvaluation Learner::MakeTargetEvaluation(bool diagnostic)
 LearningTargets Learner::MakeLearningTargets(const ExperienceSamples& samples, const anet::TensorDict& obs,
     const anet::TensorDict& next_obs, const torch::Tensor& current_all, const TargetEvaluation& evaluation)
 {
+    // stop-vs-off 診断: primary は skip の有無によらず同じ呼び出しで作る。
+    auto [target, diagnostics] = MakeTarget(samples, obs, next_obs, current_all, evaluation);
     if (!skip_current_forward_) {
-        auto [target, diagnostics] = MakeTarget(samples, obs, next_obs, current_all, evaluation);
         return {.primary = target, .diagnostics = diagnostics};
     }
     ANET_PROFILE_FUNC();
-    const auto batch = samples.actions.size(0);
     if (!samples.window.lengths.defined()) ANET_SYSTEM_ERROR("Skip learner requires replay window lengths.");
-    // 通常の n-step と、同じサンプル起点から抽選した k-step 窓の R・k・終端を並べ、同じ target 規則を適用する。
+    // stop-vs-off 診断: 窓の target は別の forward と別の乱数列で作り、primary 側の状態に触れない。
     auto window_obs = obs_norm_ ? obs_norm_->Normalize(samples.window.next_obs) : samples.window.next_obs;
-    auto combined = samples;
-    combined.actions = torch::cat({samples.actions, samples.actions});
-    combined.target_returns = torch::cat({samples.target_returns, samples.window.returns});
-    combined.n_steps = torch::cat({samples.n_steps, samples.window.lengths});
-    combined.next_state.terminals = torch::cat({samples.next_state.terminals, samples.window.terminals});
-    anet::TensorDict endpoints;
-    for (const auto& [key, value] : next_obs) endpoints.Set(key, torch::cat({value, window_obs.At(key)}));
-    const TargetEvaluation paired{
-        .diagnostic = evaluation.diagnostic,
-        .info = evaluation.info.WithRowMapping(anet::nn::NetworkForwardRowMapping(
-            torch::arange(batch, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).repeat({2}), batch, endpoints.device())),
-        .forwards = evaluation.forwards, .target_capture = evaluation.target_capture,
-        .current_taus = evaluation.current_taus, .target_taus = evaluation.target_taus};
-    // currentはBのまま、終端を2Bで渡す。Munchausen target方式はcurrentを1回だけ含む3Bになる。
-    auto [target, diagnostics] = MakeTarget(combined, obs, endpoints, current_all, paired);
-    if (evaluation.target_capture) {
-        plasticity_.target_capture_batch_size = batch;
-        plasticity_.target_capture_offset = config_.munchausen.enabled && config_.munchausen.log_policy_mode == "target" ? batch : 0;
-    }
-    return {.primary = target.narrow(0, 0, batch), .window = target.narrow(0, batch, batch), .diagnostics = diagnostics};
+    auto window_samples = samples;
+    window_samples.target_returns = samples.window.returns;
+    window_samples.n_steps = samples.window.lengths;
+    window_samples.next_state.terminals = samples.window.terminals;
+    static anet::RandomGenerator window_random(anet::seed_t(0x53564232ULL));
+    const auto make_window_taus = [this](int64_t rows, bool current) {
+        const auto& rule = current ? config_.iqn.current_taus : config_.iqn.target_taus;
+        return GenerateTaus(rows, rule.num_taus, rule.sample_mode, 0.0f, 1.0f, device_, window_random);
+    };
+    TargetEvaluation window_evaluation = evaluation;
+    window_evaluation.target_capture = nullptr;
+    window_evaluation.current_taus = [make_window_taus](int64_t rows) { return make_window_taus(rows, true); };
+    window_evaluation.target_taus = [make_window_taus](int64_t rows) { return make_window_taus(rows, false); };
+    const auto window_result = MakeTarget(window_samples, obs, window_obs, current_all, window_evaluation);
+    return {.primary = target, .window = window_result.first, .diagnostics = diagnostics};
 }
```

## 解釈の限界と残る問い

- seed 1だけで、範囲は100万exp stepまでである。1〜2Mは測っていない。
- 乱数とtargetのforwardは同時に分けたので、それぞれが差にどれだけ効くかは分かれていない。
- 比べたのは指標の全点で、重みそのものではない。
- 「更新の分布を変えない」はコードの読みである。10/29のSTOP対照は決定的モードではなく、同じRunをbit比較したものではない。
- この結果はSTOP×Singleだけに当たる。skipのlossの勾配を共有部へ返す合算と分割、Actorが繰り返すfullには当たらない。
- Single行動変化窓（合算）の範囲は8本の帯の中にあるが、無効と同じ分布かどうかは測っていない。Single行動変化窓が無効と別の分布かは分からない。

## Runと実効config

実効configは各Runの`config/config_data.txt`をそのまま複製し、公開できない値が混ざっていないか走査した。複製内の`app.runs_dir`は当時の逐語値である。実時間は順次実行のドライバーの記録で、DropMergeと同時に実行したものである。

| Run／config | 実行体 | 実時間（秒） |
|---|---|---:|
| [run_20261105-145027_sv_off_a](config/run_20261105-145027_sv_off_a.txt) | パッチなし | 124 |
| [run_20261105-145231_sv_off_b](config/run_20261105-145231_sv_off_b.txt) | パッチなし | 122 |
| [run_20261105-145434_sv_off_g](config/run_20261105-145434_sv_off_g.txt) | パッチなし | 168 |
| [run_20261105-145721_sv_stop_w1](config/run_20261105-145721_sv_stop_w1.txt) | パッチなし | 180 |
| [run_20261105-150021_sv_stop_w0](config/run_20261105-150021_sv_stop_w0.txt) | パッチなし | 175 |
| [run_20261105-150316_sv_stop_w1L](config/run_20261105-150316_sv_stop_w1L.txt) | パッチなし | 1,899 |
| [run_20261105-153456_sv_stop_w0L](config/run_20261105-153456_sv_stop_w0L.txt) | パッチなし | 1,870 |
| [run_20261105-162249_sv_pb_off](config/run_20261105-162249_sv_pb_off.txt) | パッチ版 | 109 |
| [run_20261105-162436_sv_pb_stop](config/run_20261105-162436_sv_pb_stop.txt) | パッチ版 | 143 |
| [run_20261105-162700_sv_pbL_off](config/run_20261105-162700_sv_pbL_off.txt) | パッチ版 | 1,046 |
| [run_20261105-164426_sv_pbL_stop](config/run_20261105-164426_sv_pbL_stop.txt) | パッチ版 | 1,836 |

比較に使った過去のRunの実効configは、[Singleの7群比較](2026-10-29_temporl-auxiliary-only-stage1.md)と[Trainの繰り返しの時期と長さの比較](2026-11-05_temporl-train-repeat-timing.md)にある。

| 群 | seed1 | seed2 | seed3 |
|---|---|---|---|
| 無効 `t98_ll_off` | [run_20261025-230909_t98_ll_off_s1](config/run_20261025-230909_t98_ll_off_s1.txt) | [run_20261026-001350_t98_ll_off_s2](config/run_20261026-001350_t98_ll_off_s2.txt) | [run_20261026-010415_t98_ll_off_s3](config/run_20261026-010415_t98_ll_off_s3.txt) |
| 無効 `t104r1_off` | [run_20261029-151147_t104r1_off_s1](config/run_20261029-151147_t104r1_off_s1.txt) | [run_20261029-173704_t104r1_off_s2](config/run_20261029-173704_t104r1_off_s2.txt) | [run_20261029-201541_t104r1_off_s3](config/run_20261029-201541_t104r1_off_s3.txt) |
| STOP対照 `t104r1_ctl` | [run_20261029-162638_t104r1_ctl_s1](config/run_20261029-162638_t104r1_ctl_s1.txt) | [run_20261029-191858_t104r1_ctl_s2](config/run_20261029-191858_t104r1_ctl_s2.txt) | [run_20261029-213936_t104r1_ctl_s3](config/run_20261029-213936_t104r1_ctl_s3.txt) |
| Single行動変化窓 `t104r1_ac` | [run_20261029-154259_t104r1_ac_s1](config/run_20261029-154259_t104r1_ac_s1.txt) | [run_20261029-181912_t104r1_ac_s2](config/run_20261029-181912_t104r1_ac_s2.txt) | [run_20261029-205257_t104r1_ac_s3](config/run_20261029-205257_t104r1_ac_s3.txt) |
| 通常TempoRL `t104r1_mrg` | [run_20261029-152756_t104r1_mrg_s1](config/run_20261029-152756_t104r1_mrg_s1.txt) | [run_20261029-175720_t104r1_mrg_s2](config/run_20261029-175720_t104r1_mrg_s2.txt) | [run_20261029-203446_t104r1_mrg_s3](config/run_20261029-203446_t104r1_mrg_s3.txt) |
