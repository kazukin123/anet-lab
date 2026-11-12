# PRD067 Munchausen RL 実装計画

実装・受入検証完了（2026-09-05）。[検証記録](067_MunchausenRL_30validation.md)にテスト、OFF完全一致、ON5本、区間性能と実行証拠をまとめた。以下は承認された計画として保持する。

追補: 実装後の合意により、`MunchausenConfig` を共通型へ移し、`ActorQHintConfig::munchausen` としてまとめて保持する形へ変更した。Actorは `log_policy_mode` を参照しない。以下の初版計画にある「狭い設定」からの変更理由と現行契約はADR 0036およびPRD D7に記載する。

## 概要

TD / QR / IQN に既定 OFF の Munchausen RL を追加する。3つの log-policy mode、K3 Actor Qヒント、診断、D15 の soft 楽観ターゲットまでを必須範囲とする。
比較基準はコミット `8f8104f`。既存 launcher 差分と未追跡ファイルは保持する。

## 主な変更

- `LearnerConfig::MunchausenConfig` の5項目を追加する。既定は enabled=false、log_policy_mode=target、alpha=0.9、entropy_tau=0.03、clip_value_min=-1。OFF 時も値域とmodeを検証する。ON + Double DQN、ON + Thompson targetは設定解決後に構築エラーにする。
- QRの既存処理を `MakeRiskBiasedScore` へ抽出し数値・RNGを維持する。`ActionPolicy::GetRiskScoreSpec()` からUQE所有の現在tauとtail-mean設定を得る。IQN hardは維持し、softは既存分位点の経験分位近似とする。
- Munchausen helperにはcurrent/next方策スコアと、価値用next平均Qを別引数で渡す。診断5値をhelper内で完成させ、risk方策でもsoft_gapは平均Q基準とする。
- target modeは正規化済みcurrent/nextを連結しtargetを2Bで1回forward。onlineはcurrent・target-valueの後でNoGrad・eval fresh online forward。online_reuseは既存train-mode出力をdetach。ONではhard選択を呼ばない。
- IQNの生成順はcurrent→target-value→必要ならfresh online。bonus側はtargetでM本、他2modeでN本、next側はM本。OFFの計算経路・RNG消費は維持する。
- NoGrad・FP32実空間で方策・bonus・soft bootstrapを計算する。TBOは分位点ごとに逆変換し、完成targetだけを再変換。QR/IQNのON専用target組立を追加し既存CalcTargetQuantilesはOFF専用。bonusはN-step先頭へ1回、終端でも残す。
- target modeのplasticity captureは[2B,F]を検証して後半B行を渡す。forward_target、forward_munchausen_online、munchausen_targetを計測し、既存初期化ログへmode・スコア源を追記する。
- Actor、codec、WithAction、推定器、Rainbow構築箇所をK3へ移行する。狭いActorQHintConfigと既存scoreを使い追加forwardなし。WithActionはq_saとbonusを再gather。OFFの第3列はゼロ、旧K2はschema違反。
- readback順はpriority・clip件数→IQN診断→Munchausen診断→upper-tail統計。PER OFFでも回収し、既知の未成立keyはNaN。baseline、@munchausen、診断7tag、Atari Run profile、現行設計文書を同期する。

## TDD・検証

1. 8f8104fのRelWithDebInfo buildで変更前OFF Runを取得し、コマンド・effective config・resolution・終了コード・系列hashを保存する。
2. QR hard UQEを既存挙動テストで固定して抽出。TD target既知値をtracer bulletとして、1 behaviorごとにRED→GREENを完了する。QR/IQN、残り2mode、K3、D15へ拡張しGREEN後だけrefactorする。
3. ConfigData/Factory、UpdateFromSamples、Actor、WithAction、推定器、GetScalarからPRDの20項目を検証する。9組、N≠M、TBO、N-step/終端、clip、CUDA BF16、forward回数/mode/tau、PER ON/OFF、capture、K3、Rainbow OFF、UQE減衰を含む。独立oracleと短い同seed反復でtarget/loss/TD errorを確認しproductionへtest-only APIを追加しない。
4. VsDevCmd.bat経由のDebug全体buildとDQN/Replay/PER/tauテストを実行する。学習検証は前後ともRelWithDebInfo。
5. OFF比較はBreakout、seed 1、400k exp step、warmup 200k、backend.@deterministic、run.@v5_iqn_impala_x2。37_agent_qtd/*と38_agent_loss/*を{tag,step,value}へ正規化し系列SHA-256完全一致を必須とする。
6. ONは通常backendでtarget、online、online_reuse、target+actor_approx、target+UQE optimisticの5本。PRD指定のtmp_smoke名を使う。診断7tag、finite loss、actor_approxの実利用、D15解決設定を確認し、250k〜400kのthroughputとProfileRangeを記録する。

## 前提

- PRD §3を合意したhelper引数へ訂正する。stratifiedは乱数を消費し、乱数非依存なのはfixedと訂正する。tau生成実装は変更しない。
- IQN経験分位とActor scoreの近似を区別し、hard/softの厳密一致は保証しない。
- OFF hash条件を緩和しない。基準Run未取得・不一致を合格扱いしない。
- D15を省いて完了としない。成績/性能合否閾値、action mask、Rainbow設定公開、SAC共通化は追加しない。
- Git staging/commit/pushは行わない。
