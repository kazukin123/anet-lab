# DropMergeEnv 直前行動観測(prev-action trio + DROP 列マーカー・暫定)

> 設計分担: Claude=設計/PRD、実装=Codex、Run/commit=ユーザー。
> 本書は self-contained。実装時は行番号ではなく、近傍のシンボル名で再検索する。
> 032(LunarLander 版)の DropMerge 固有再設計。PRD900(Agent 側共通部品)は実装しない。

## Context(背景・目的)

主目的は**性能・安定性の向上**。LunarLander では直前 action の観測化
([032_lunarlander_obs_include_action_10prd.md](032_lunarlander_obs_include_action_10prd.md))で
成績・安定性の両面効果を確認済みであり、DropMerge へ横展開する。NEET(長い NOOP 連鎖)は並行課題で
あり本 PRD の主目的ではない(改善したら副次効果)。

DropMerge(`direct_noop`、`use_instant_drop=true`、stack4)での情報分析:

- NOOP/DROP の binary 履歴は既存 obs からほぼ復元可能(`no_drop_timeout_ratio` が DROP でリセット、
  非 DROP で +1/no_drop_timeout_steps 刻み増加。rank シフトも補助信号)。
- **復元不能な真の追加情報は次の 2 つ**:
  1. **DROP した命令列(58-way)** — 投下痕跡は drop_noise・転がり・合体カスケードで盤面から消え得る。
  2. **自己行動の同定** — 探索行動は確率的、replay は過去方策、a_{t-1} は stack 窓外の入力から選ばれた。
- 期待効果の主経路は「帰属の well-posed 化(盤面変化を自分の投入と自律的物理進行に分離)による安定性」と
  「列情報による短期空間戦略の精密化」。
- 最大リスクは方策指紋ショートカット(行動固執/perseveration)。監視メトリクスをセットで導入する。

効果判定の一次指標は**長期 Run(700M+)での最終 eval 成績到達点**(reward EMA 終盤水準+瞬発 MaxRank)。
短期 A/B には無害性確認(同 step 帯でアンカーとブレ幅内)だけを求め、効果証明は求めない。

## 確定した設計判断

1. **2 つの独立 flag**(いずれも default **false**)。個別に ON/OFF でき、将来の単独効果測定を可能にする。
   - `DropMergeEnv.obs_include_prev_action` — **prev-action trio**: 既存 `vector` の末尾へ 3 scalar を連結。
   - `DropMergeEnv.obs_prev_drop_marker` — **DROP 列マーカー**: 既存 `grid` の dropper class(`kFruitTypeCount+1`=12)
     を再利用し、直前 DROP の命令列を top row に描画。
2. **意味論は 032 と同一**: 「obs_t には、その obs へ至らせた行動 a_{t-1} を入れる」。
   `Reset()` 直後は未行動(trio 全ゼロ・マーカー無し)。
3. **記録するのは選択した命令**であり、執行の成否は問わない(`is_busy` 中に棄却された DROP 命令も DROP として
   記録する)。`steps_since_last_drop_` のリセットが命令ベース(`is_drop_action`)である既存挙動と一貫させる。
   現行の instant_drop 構成では棄却は実質発生しない。
4. **対応 action mode は direct 系のみ**(`Direct` / `DirectNoop`)。move 系でいずれかの flag が ON なら
   構築時に `ANET_SYSTEM_ERROR`(Fail-Fast 原則: 明示要求の不履行)。
   - Direct では noop 成分が常時 0 になるだけで意味論は成立。
   - move 系は drop_x の意味論が不成立で、マーカーは class 12(現在の dropper 位置表示)が現役のため衝突する。
5. **新しい obs キーは作らない**。trio は `ObsKeys::kVector` 連結(stacker・NN config 無変更で通る)、
   マーカーは既存 `grid` の既存 num_classes 範囲内(spec shape 完全不変)。
   - NN 側変更ゼロの根拠: `vector_feature`(SpatialEmbedder)は入力次元非依存・parameter-free の broadcast、
     初段 Conv2d は初回 forward で in_channels 自動取得(lazy)、初段以降の shape/parameter は完全不変、
     branch bind 行も不変。情報は Stem/Transformer/head の全段へ届く。
   - 例外は 2 つ(いずれも本 PRD の範囲では非該当): ①`obs_include_prev_action=true` は初段 conv の
     weight shape が変わるため既存 `.anet` snapshot の `auto_load_file` と**形状非互換**。
     `obs_prev_drop_marker` **単独なら形状は完全互換**(grid spec・channel 数不変。ただし観測意味論が
     変わるため、既存 snapshot からの学習継続が妥当かは別判断)。②旧世代 `HybridSpatialEmbedder` 系ブロック
     (`[Embed4064]` 等の `embed.scalar_dim = 4` 明示固定)を使う旧 body へ切り替える場合のみ
     scalar_dim の修正が必要(現行チェーンではインスタンス化されない)。
6. **frame stack 4 により自動で 4 履歴になる**(trio は vector の一部として、マーカーはフレーム毎描画として)。
   最新 1 手のみの構成は採らない。
7. **RNG を消費しない**。flag ON でも乱数消費列は不変。flag OFF は従来とビット一致。
8. **固執監視メトリクス** `ep_same_drop_col_ratio` を追加する。これは方策挙動の測定であり obs 拡張とは独立
   なので、**direct 系では flag に関係なく常時有効**にする。DROP 命令列を持たない move 系では未定義とし、
   `GetScalar` は **NaN を返して metric 側にサイレントに無視させる**(nullopt は設定ミス疑いの WARN を
   誘発するため使わない。mode 切替のたびに metrics 設定の切替を強要しないための意図的選択)。
9. obs_norm は現行 `obs_norm.pass_through = true`(agent.txt の DefaultDQNAgent.baseline)で恒等のため
   特別扱い不要。pass_through=false 構成でも 0/1 flag と [-1,1] スカラーは SymLog でほぼ恒等であり許容
   (032 判断 #10 と同じ)。
10. flag ON の Run は**新 observation 契約=新 baseline 系譜**として run 名 suffix(`_pa`)で明示する。
    既存 Run 群との厳密比較は「flag OFF はビット一致」で担保する。

## 仕様

### Config

```txt
DropMergeEnv.obs_include_prev_action = false  # default。直前actionのtrio(valid/noop/drop_x)をvectorへ追加
DropMergeEnv.obs_prev_drop_marker = false     # default。直前DROP命令列のマーカーをgrid top rowへ描画
```

- `DropMergeEnvConfig` に `bool obs_include_prev_action = false;` と `bool obs_prev_drop_marker = false;` を
  追加し、`ANET_READ_CONFIG` で読む(config dump へ自動で載る)。
- コンストラクタで `action_mode_` 解決後に gating:

```cpp
if ((config_.obs_include_prev_action || config_.obs_prev_drop_marker) &&
    action_mode_ != ActionMode::Direct && action_mode_ != ActionMode::DirectNoop) {
    ANET_SYSTEM_ERROR("DropMergeEnv: obs_include_prev_action / obs_prev_drop_marker require "
        "action_mode=direct or direct_noop. actual=" << config_.action_mode);
}
```

- `apps/runner/config/DropMerge.txt` の実験セクションにコメントアウト行を追加:

```txt
#E.obs_include_prev_action = true   # 直前actionのtrioをobsへ追加(040)
#E.obs_prev_drop_marker = true      # 直前DROP列マーカーをgridへ描画(040)
```

### Obs レイアウト: prev-action trio(`obs_include_prev_action=true` 時)

既存 vector(`use_no_drop_timeout_gameover=true` なら 5、false なら 4 scalar)の**末尾に 3 scalar を追加**。
既存 index の順序・値は不変。

| 追加 index | 内容 | labels | min | max |
|---|---|---|---|---|
| base+0 | `prev_action_valid`: Reset 直後=0、行動後=1 | `prev_valid` | 0 | 1 |
| base+1 | `prev_action_noop`: 直前が NOOP なら 1 | `prev_noop` | 0 | 1 |
| base+2 | `prev_action_drop_x`: 直前 DROP の命令列中心を [-1,1] 正規化。非 DROP は 0 | `prev_drop_x` | -1 | 1 |

- `drop_x` の正規化: 命令 index `i`(0..N-1、N=`num_drop_actions_`)に対し `x_norm = (2i+1-N)/N`。
  これは `processAction` の座標マッピング `dropper_.x = min_x + (i+0.5)*cell_w` を半幅で正規化した値と一致する。
- `grid_cols=58`(偶数)では最小絶対値が ±1/58 で **x_norm=0.0 になる列は存在しない**ため、
  「中央 DROP」と「非 DROP(=0)」の数値衝突は起きない(noop flag でも区別可能)。
- 値の例(direct_noop、N=58): Reset=[0,0,0] / NOOP 後=[1,1,0] / DROP_0 後=[1,0,-57/58] /
  DROP_57 後=[1,0,+57/58]。
- `GetSpec()` は flag に応じて shape / labels / min_values / max_values を拡張する。
  スカラー次元の導出(`kBaseScalarObsDim` / `kNoDropTimeoutScalarObsDim` を使う箇所)と `makeState()` の
  書き込み、`vec_buffer_` の確保サイズの **3 点を必ず一致**させる。

### Obs レイアウト: DROP 列マーカー(`obs_prev_drop_marker=true` 時)

- 直前 action が DROP(命令 index `i`)の場合のみ、`makeState()` の果物充填の**後**に
  top row(`grid_rows-1`)の対応列セルへ `kFruitTypeCount+1`(=12)を書き込む(既存 dropper 描画と同じ
  上書き慣行)。直前が NOOP または Reset 直後は描画しない(class 12 プレーンは全ゼロ)。
- 対応列は命令中心座標 `x = min_x + (i+0.5)*cell_w_cmd` を既存 dropper 描画と同じ
  `target_c = (int)((x - min_x)/cell_w_grid)` + clamp で grid 列へ変換する
  (`drop_divisions == grid_cols` の既定では `target_c == i`)。
- direct 系では既存の `draw_dropper` 分岐(`use_dropper_x_grid`)が無効のため class 12 は死にプレーンであり、
  マーカーとの衝突は起きない。spec(shape / num_classes / min / max)は**完全不変**。
- トレードオフ(意図的): マーカーセル 1 個分の果物情報を上書きする(top row に果物が来るのは overflow 間際
  のみ)。class 12 の意味は「move 系=現在の dropper 位置 / direct 系=直前 DROP 命令列」とモード依存になる。
  コード内コメントと CONTEXT.md 用語で明示する。

### 状態遷移

- メンバ `int64_t last_action_ = -1;` を追加(-1=未行動)。
- `Reset()`: `last_action_ = -1;` に戻す(trio 全ゼロ・マーカー無しに対応)。
- `Step(action)`: 先頭(既存の per-episode メトリクスリセットブロックの後)で `last_action_ = action;` を
  設定してから既存処理を行う。`makeState()` は `last_action_` と `action_mode_` から
  NOOP / DROP(命令 index)を復号して trio とマーカーを生成する(direct_noop: 0=NOOP、k>0 → i=k-1。
  direct: k → i=k)。

### 固執監視メトリクス(direct 系・flag 非依存)

- per-episode scalar **`ep_same_drop_col_ratio`** = 連続する DROP 命令ペアのうち命令 index が完全一致した
  割合。エピソード内 DROP 数が 2 未満なら 0.0。棄却された DROP 命令(busy 中)もペアに数える
  (設計判断 3 の命令ベース意味論と一貫)。
- **direct 系限定**: DROP 命令列を持たない move 系では未定義。move 系では `GetScalar("ep_same_drop_col_ratio")`
  が **NaN を返し、metric 側でサイレントに無視される**(nullopt は設定ミス疑い WARN を誘発するため不可)。
- 実装: エピソード内トラッカー(`ep_drop_count_`、`ep_same_col_count_`、直前 DROP 命令 index)を
  `Step()` の per-episode リセットブロック(`step_count_ == 0`)で初期化し、DROP 命令のたびに更新。
  エピソード終了時に `last_ep_same_drop_col_ratio_` へ確定し、`GetScalar("ep_same_drop_col_ratio")` は
  既存の per-episode キー(`ep_end_fruit_count` 等)と同じ `episode_just_ended_` ゲートの返却規約に従う。
- `apps/runner/config/DropMerge.txt` へ metric 行を追加(既存 42_env / 51_eval1 / 52_eval2 の番号帯に合わせる):

```txt
M.[42_env/26_same_col_ratio_mean]     = $env mean.ep_same_drop_col_ratio @train
M.[42_env/73_same_col_ratio_mean_ema] = $env mean.ep_same_drop_col_ratio @train $ema ema_alpha:0.001
M.[51_eval1/79_same_col_ratio_mean]     = $eval.[eval1] @episode_end $env mean.ep_same_drop_col_ratio
M.[51_eval1/88_same_col_ratio_mean_ema] = $eval.[eval1] @episode_end $env mean.ep_same_drop_col_ratio $ema ema_alpha:0.01
M.[52_eval2/79_same_col_ratio_mean]     = $eval.[eval2] @episode_end $env mean.ep_same_drop_col_ratio
M.[52_eval2/88_same_col_ratio_mean_ema] = $eval.[eval2] @episode_end $env mean.ep_same_drop_col_ratio $ema ema_alpha:0.01
```

- **位置づけ(重要)**: 本メトリクスは A/B の判定材料ではなく「**崩壊検知+失敗解釈の補助**」。
  完了済みアンカー Run には存在せず、高値が「元からの列集中戦略」か「prev-action 起因の固執」かを
  単独では区別できないため、**因果判定には使わない**(採否ゲートはあくまで reward のアンカー比ブレ幅)。
  - 固執による絶対棄却条件は複合のみ: eval の ratio EMA が持続的に 1 近傍(目安 >0.9)へ張り付き、
    **かつ** eval reward がアンカー比で劣化。ratio 高値単独では棄却しない。
  - NOOP 側の固執は本メトリクスでは検出できないため、既存監視セットを併用する:
    `reset_noop_uqe_margin` / `reset_noop_q_margin`(51_eval1/43・44 等)、`tr_timeout_mean`、
    `timeout_cand_mean`、`noop_uqe_win_rate`。
  - 両 flag OFF の control Run は事前には取らない(訓練進行を揃えた control は 50〜100M 必要でコスト非対称。
    有害な固執は reward ゲートで検出され、無害な固執は一次指標上の棄却理由にならない)。
    長期 Run 後に単独 flag ablation を行う場合、その OFF Run が本メトリクスの control を兼ねる。

## 非対象(Non-goals)

- PRD900(Agent 側共通部品)の実装。恒久化はトリガー駆動:
  ①別 Env でも必要になったら PRD900 へ、②列 grounding 精度不足が疑われたら token/spatial map 拡張へ。
- one-hot 59 表現、別 obs キー、Transformer token 注入、spatial action map の新 Grid キー。
- move 系 action mode への意味論定義。
- 単独 flag ablation Run の事前実施(長期 Run 後の選択肢)。
- `obs_norm` の除外設定。reward の観測化。

## 受け入れ基準

1. 両 flag OFF(default)で従来と**ビット一致**: obs(vector / grid)・spec・乱数消費列すべて不変。
2. `obs_include_prev_action=true` で `GetSpec()` の vector spec が base+3 次元
   (labels 末尾 `prev_valid, prev_noop, prev_drop_x`、min/max 拡張)になり、`ValidateObservation` を通過。
3. trio の値が意味論どおり: Reset 全ゼロ / NOOP 後 [1,1,0] / DROP_i 後 [1,0,(2i+1-N)/N]。
4. `obs_prev_drop_marker=true` で DROP 後の obs のみ top row 対応列が 12、NOOP/Reset 後はマーカー無し。
   grid spec は不変で one-hot 前処理(`num_classes=13`)を通過。
5. flag ON でも既存 scalar(先頭 base 個)と grid(マーカーセル以外)は OFF 時と同値
   (同 seed・同 action 列で一致)。
6. direct / direct_noop の両 mode で動作。move 系+flag ON は構築時 `ANET_SYSTEM_ERROR`。
7. train / eval1 / eval2 の全経路で spec と実 obs の次元が一致し、NN config 無変更で学習が回る。
8. run の config dump(`config/config_data.txt`、`json/DropMergeEnv.json`)に両 flag が出力される。
9. `ep_same_drop_col_ratio` が direct 系で定義どおりに算出され(flag 非依存)、metric 行で収集できる。
   move 系では `GetScalar` が NaN を返し、metric 側でサイレントに無視される(WARN を出さない)。
10. 既存テスト含め全テスト green。既存の未コミット差分(NoLegal PH1 等)を上書き・巻き戻ししない。

## テスト項目(DropMergeEnv_test.cpp へ追加)

> 注意: bare default の action_mode は move_fast。trio / marker / ratio のテストは env config で
> `action_mode = direct_noop`(または direct)を明示して構築する。

1. default(両 OFF): spec 次元・labels 従来どおり、direct 系で grid に class 12 が現れない(regression)。
2. trio ON: spec 検証(次元・labels 末尾 3 個・min/max)。
3. trio ON: Reset 直後 [0,0,0]、NOOP 後 [1,1,0]、DROP_0 / 中央付近 / DROP_{N-1} 後の drop_x が
   `(2i+1-N)/N` に一致(近似比較)。
4. marker ON: DROP_i 後に top row の列 i のみ 12、他セルは果物値を維持。続けて NOOP するとマーカーが消える。
5. ON/OFF 同 seed 並走: 同一 action 列で、先頭 base scalar 全一致・grid はマーカーセル以外全一致
   (乱数列不変の間接検証)。
6. direct mode: 構築成功、noop 成分が常時 0、DROP 意味論は direct_noop と同一。
7. move / move_fast + 各 flag ON: 構築時 `ANET_SYSTEM_ERROR`。
8. 終端まわり: done / truncated 到達 step でも obs 次元が維持される。
9. `ep_same_drop_col_ratio`: 既知の DROP 列シーケンス(例: [3,3,5,5,5] → 3/4=0.75)で検証。
   DROP 2 未満のエピソードは 0.0。NOOP を挟んでも連続 DROP ペアの定義が崩れないこと。
10. 棄却 DROP 命令の記録(設計判断 3 の回帰): `use_instant_drop=false` で DROP_i → busy 中に DROP_j(j≠i)。
    果物が増えない(執行棄却)こと、直後 obs の trio が [1,0,x_j]・マーカーが列 j を示すこと、
    ratio tracker がペア (i,j) を棄却命令込みで数えることを検証。
11. move 系(両 flag OFF): `GetScalar("ep_same_drop_col_ratio")` が NaN を返す(nullopt でないこと)。
12. MetricsLogger 未 Init で構築してもクラッシュしない(既存ガードの回帰)。

## 実装対象

- `core/envs/dropmerge1/src/DropMergeEnv.hpp`
  - `DropMergeEnvConfig::obs_include_prev_action` / `obs_prev_drop_marker`、`last_action_` メンバ、
    ratio 用トラッカーメンバ。
- `core/envs/dropmerge1/src/DropMergeEnv.cpp`
  - `ANET_READ_CONFIG` ×2、コンストラクタの mode gating、`GetSpec()` 拡張、`makeState()` の trio 書き込みと
    マーカー描画、`Reset()` / `Step()` の `last_action_` 更新、ratio トラッカー更新と `GetScalar` キー追加。
- `core/envs/dropmerge1/src/DropMergeEnv_test.cpp` — 上記テスト項目。
- `apps/runner/config/DropMerge.txt` — コメント例 2 行+metric 行 6 本。
- NN config・anet-core 本体は**変更しない**。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target DropMergeEnv-test'
core\envs\dropmerge1\bin\Debug\DropMergeEnv-test.exe
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug'
git diff --check
```

runner smoke(ユーザー実施): 両 flag ON で起動し、nn_viz の vector 入力ラベルに `prev_*` が出ること、
eval panel の grid 表示でマーカーが視認できること、config dump に両 flag が載ることを確認。

## 実験計画(prefix 方式)

1. `run_20260725-133253_imp2-vit128_tgtg-nose`(spatial OFF 100M)の結果で長期 Run の spatial 設定を決定。
2. その設定+**両 flag ON**、run 名 suffix `_pa`、`app.online.exp_pause_step = 100,000,000` で A/B Run を起動。
   **trio+marker は単一の候補パッケージとして採否判断する**(単独効果は未判定のまま昇格し、
   切り分けは長期 Run 後の単独 flag ablation で行う。700M Run からはどちらが効いたかは判定できない)。
3. 100M 自動 pause 中に、対応アンカー(spatial ON=`run_20260724-221301_imp2-vit128_tgtgreedy` /
   OFF=nose)と同 step 帯を比較。判定はブレ幅基準(同構成ペア Run の seed 間変動を参照)で、
   **同等以上なら合格**(効果証明は求めない)。
   固執による棄却は複合条件のみ: `ep_same_drop_col_ratio` の eval EMA が 1 近傍(目安 >0.9)へ持続的に
   張り付き、**かつ** eval reward がアンカー比で劣化している場合(高値単独では棄却しない)。
   あわせて NOOP 系監視セット(reset_noop margin 43/44・tr_timeout・timeout_cand・noop_uqe_win_rate)を確認。
4. 合格なら resume してそのまま 700M 長期 Run 本体へ昇格(連続実行を維持)。不合格なら Run 破棄、
   prev_action 無しで仕切り直し。
5. 前提条件: C ドライブ空き確保(700M で 60〜70GB 想定)は **A/B 開始前**に必要(ユーザー作業)。
   resume 後の自動 pause は再発火しないため、700M 終端は手動停止または別途設定。

## 関連

- [032_lunarlander_obs_include_action_10prd.md](032_lunarlander_obs_include_action_10prd.md) —
  LunarLander 版先行検証(同型の意味論・テスト構成)。
- [900_agent_prev_action_obs_10prd.md](../900_agent_prev_action_obs_10prd.md) —
  恒久・env 非依存の共通部品版(トリガー①の行き先)。本 PRD とは併用しない。
- [039_dropmerge_nolegal_adjudication_10prd.md](039_dropmerge_nolegal_adjudication_10prd.md) —
  同一ファイル群に未コミット差分あり(保持すること)。
