# WeightInitConfig.mode の int→string 化

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 動機の上流は `028_cnblock_convnext_10prd.md`（ConvNeXt v1 の CNBlock 導入）。ConvNeXt が要求する `trunc_normal_(std=0.02)` 初期化を追加しようとしたところ、`WeightInitConfig.mode` が数値 0~4 の5値に達しており、これ以上数値を増やすと可読性の限界であるとユーザーから指摘があった。**本書は CNBlock 本体とは独立に先行実装する**（028 は本書適用後の string mode を前提に設計されている）。

## Context（背景・目的）

`WeightInitConfig::mode`（`core/anet-core/include/anet/nn.hpp:48-53`）は現在 `int`（0:Default, 1:Xavier, 2:He, 3:Orthogonal, 4:Constant）で、`net.block.[XXX].init.mode = <数値>` として `apps/runner/config/*.txt` 全体で52箇所使われている。ConvNeXt導入で `trunc_normal`（PyTorch/timm 標準の切断正規分布初期化）を追加する必要が生じたが、数値を単純に "5" として追加すると、設定ファイルを読む人間が数値と初期化手法の対応をコメント無しでは判別できず、可読性がすでに限界に達している（既に `nn.txt:5-10` に `#init.mode = 1 # Xavier Uniform` のような早見表コメントが必要になっている状態）。

目的: `WeightInitConfig.mode` を人間可読な文字列へ変更する。**後方互換は持たない**（数値との併用・fallbackは入れない）。既存の全 `.txt` 設定・コード内デフォルト値を新規格へ一括書き換える。

## 確定した設計判断

1. `WeightInitConfig.mode`（`nn.hpp:48-53`）を `int mode = 1;` から `std::string mode = "xavier";` に変更する。デフォルト値は現行動作（Xavier）を維持する。
2. マッピング: `0:Default(no-op) → "default"`, `1:Xavier → "xavier"`, `2:He → "he"`, `3:Orthogonal → "orthogonal"`, `4:Constant → "constant"`。既存の文字列config（`ResBlockConfig::activation`="silu", `norm_type`="batch", `activation_mode`="post" 等、`nn_modules.cpp:657-659`）と同じ**小文字規約**に揃える。既存の文字列値はいずれも単語1つ（アンダースコアなし）だが、後続PRD（028）で追加する `trunc_normal` は2単語になる。既存に前例はないが可読性のため意図的にsnake_caseを採用し、他の5値は単語のまま変えない。
3. `WeightInitializer::Initialize`（`core/anet-core/src/nn_impl.hpp:18-72`）の分岐を数値比較から文字列比較へ書き換える。**未知の文字列が来た場合は `ANET_SYSTEM_ERROR` で明示的に失敗させる**。現行実装は `mode` が 0~4 いずれにも一致しない場合、weight初期化は静かに何もせず（no-opではなく、単に該当分岐が無いだけ）、かつ bias だけは `else` 分岐（`nn_impl.hpp:63-70`）でゼロ初期化されるという非対称な潜在バグがある。文字列化を機に「未知値は即エラー」という一貫した挙動に修正する（挙動改善として明記）。
4. **後方互換は持たない**（数値との併用・fallbackなし）。既存の全ての `init.mode = <数値>` 記述を文字列に一括書き換える。移行漏れがあれば設定パーサはビルド時に検出できず、実行時に文字列比較へ失敗して `ANET_SYSTEM_ERROR` で落ちる（`024_nn_dropout_droppath_10prd.md` の `Drop.p`→`dropout_rate` ハード改名と同じ「fallback/WARNを入れない」設計判断）。
5. ADR（`docs/adr/0008-weight-init-mode-string.md`）を作成する。既存 `docs/adr/0007-nn-dropout-config-semantics.md` と同型の短編とし、「後戻りしにくい決定」として、mode種類の増加（今後 trunc_normal を含め6種以上になる）で数値の意味が読み手に伝わらなくなったため文字列へハード改名し、fallbackを持たせない、という判断を記録する。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `0598b1b`。`nn.hpp` / `nn_impl.hpp` / `nn_modules.cpp` / `nn_test.cpp` は未コミット変更なし（行番号は HEAD 基準で安定）。`nn_modules.cpp` は全2128行、`nn_impl.hpp` は全234行、`nn.hpp` は全147行。

- **`WeightInitConfig` の現状**（`nn.hpp:48-53`）:
  ```cpp
  struct WeightInitConfig {
      int mode = 1;                       ///< 0:Default, 1:Xavier, 2:He, 3:Orthogonal 4:Constant
      double manual_gain = 0.0f;          ///< Orthogonal時の手動ゲイン (0.0なら自動計算)
      std::string nonlinearity = "relu";  ///< "relu", "linear", "tanh" etc.
      double constant_val = 0.0;          ///< for Constant
  };
  ```
- **`WeightInitializer::Initialize`**（`nn_impl.hpp:18-72`、テンプレート関数、`layer->weight`/`layer->bias` を持つ torch::nn Module ハンドル向け）:
  - `mode==0`: 即 return（no-op、weight/biasとも触らない）。
  - `mode==1`: `torch::nn::init::xavier_uniform_(weight)`。
  - `mode==2`: `GetNonlinearityType(config.nonlinearity)` を介し `torch::nn::init::kaiming_normal_(weight, 0.0, torch::kFanOut, nonlinearity_mode)`。
  - `mode==3`: `manual_gain>0` ならそれを、なければ `torch::nn::init::calculate_gain(nonlinearity)` を使い `torch::nn::init::orthogonal_(weight, gain)`。
  - `mode==4`: `torch::nn::init::constant_(weight, config.constant_val)`。
  - bias（`:61-71`）: `bias.defined() && mode!=0` のとき、`mode==4` なら `constant_(bias, constant_val)`、それ以外（1,2,3、および現状該当分岐のない5以上）は `constant_(bias, 0.0)`。
  - **未知の数値（5以上）が来た場合**: weight側はどの `if/else if` にも一致せず何も実行されない（silent no-op）が、bias側は「`mode!=0` かつ `mode!=4`」に該当してゼロ初期化される。weight未初期化・bias強制ゼロという非対称な現状挙動（判断3の修正対象）。
- **`GetNonlinearityType`**（`nn_modules.cpp:39-46`）: 文字列→`torch::nn::init::NonlinearityType` の変換。`"relu"/"linear"/"tanh"/"leaky_relu"` を判定し、未知文字列は黙って `torch::kReLU` にフォールバックする（＝この関数自体は動作するが、`Initialize`側の`mode`文字列化とは無関係）。
- **`net.block.[XXX].init.mode` を数値で設定しているコード内デフォルト値**（確認済み、全て `WeightInitConfig config;` のメンバ初期化直後・`ANET_READ_CONFIG` 呼び出し前に明示上書きするパターン）:
  - `ResBlockModuleFactory::Config`（`nn_modules.cpp:901-905`）: `init1.mode=2`(He) / `init2.mode=4`(Constant/ZeroInit) / `init_ds.mode=2`(He)。
  - `Conv1dModuleFactory::Config`（`:1886`）: `init.mode=2`(He)。
  - `Conv2dModuleFactory::Config`（`:1914`）: `init.mode=2`(He)。
  - `LinearModuleFactory::Config`（`:1861-1868` 付近）: 明示デフォルト上書きなし。`WeightInitConfig` 自体の構造体デフォルト（`mode=1`→文字列化後`"xavier"`）がそのまま使われる。
  - 上記以外にも `WeightInitConfig` を使う箇所が `rainbow_agent.cpp` / `muzero_based_agent.cpp` / `dqn_based_heads.cpp` / `dqn_based_heads.hpp` / `default_dqn_agent.cpp` / `dqn_based_test.cpp` / `nn_test.cpp` に存在することを `grep -rl "WeightInitConfig"` で確認済み（個々の行番号は実装時に再grepして洗い出す）。
- **ANET_READ_CONFIG マクロ**: `ReadConfig(config_data, "フィールド名", field)` に展開される（`config.hpp:194-196` 付近）。フィールド追加はメンバ変数追加＋この1行追加で機械的に完結する（`ResBlockFactory::Config` 等で実証済みのパターン）。
- **設定ファイル側の使用状況**（`grep -rn "init\.mode\s*=\|init1\.mode\s*=\|init2\.mode\s*=\|init_ds\.mode\s*="` で確認、`runs/` 配下の過去実行ログを除く）:

  | ファイル | 箇所数 |
  |---|---|
  | `apps/runner/config/nn.txt` | 13 |
  | `apps/runner/config/ImageCls.txt` | 9 |
  | `apps/runner/config/LunarLander.txt` | 6 |
  | `apps/runner/config/CartPole.txt` | 4 |
  | `apps/runner/config/GridMaze.txt` | 2 |
  | `apps/runner/config/agent.txt` | 4 |
  | `apps/runner/config/DropMerge.txt` | 14 |

  合計52箇所、全て `net.block.[XXX].init.mode = N` または `A.head_init.mode = N`（`agent.txt` 内、DQN系ヘッド初期化）の形。値はいずれも 0~3 の範囲内（4=Constantの使用例は設定ファイル側では未確認、コード内デフォルトのみ）。
- **`apps/runner/tools/dropmerge_optuna.py:382,420,425`**: Optuna study が生成する config 文字列リテラルとして `"net.block.[OptConvInit].init.mode = 2"` 等をハードコードしている（Optunaのハイパーパラメータとして数値を探索しているのではなく、固定値の文字列リテラル）。
- **`apps/runner/11_batch_run.bat:76,102-108`**: 該当行は全て `REM` （バッチファイルのコメントアウト構文）で実行に影響しない。
- **`runs/*/config/*.txt`**: 過去実行時の config dump（ground truth ログ、`feedback_check_run_config_dump` 参照）であり、書き換え対象外。

## 設計方針

### A. `WeightInitConfig` の型変更（`nn.hpp`）

```cpp
struct WeightInitConfig {
    std::string mode = "xavier";        ///< "default"/"xavier"/"he"/"orthogonal"/"constant"（"trunc_normal"は028で追加）
    double manual_gain = 0.0f;          ///< Orthogonal時の手動ゲイン (0.0なら自動計算)
    std::string nonlinearity = "relu";  ///< "relu", "linear", "tanh" etc.
    double constant_val = 0.0;          ///< for Constant
};
```

### B. `WeightInitializer::Initialize` の書き換え（`nn_impl.hpp:18-72`）

```cpp
template <typename T>
static void Initialize(T& layer, const WeightInitConfig& config)
{
    if (config.mode == "default") return;

    auto& weight = layer->weight;
    torch::NoGradGuard no_grad;

    if (config.mode == "xavier") {
        torch::nn::init::xavier_uniform_(weight);
    } else if (config.mode == "he") {
        auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);
        torch::nn::init::kaiming_normal_(weight, 0.0, torch::kFanOut, nonlinearity_mode);
    } else if (config.mode == "orthogonal") {
        double gain = 1.0;
        if (config.manual_gain > 0.0f) {
            gain = config.manual_gain;
        } else {
            try {
                auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);
                gain = torch::nn::init::calculate_gain(nonlinearity_mode);
            } catch (...) {
                ANET_SYSTEM_ERROR("Unknown nonlinearity: " << config.nonlinearity);
            }
        }
        torch::nn::init::orthogonal_(weight, gain);
    } else if (config.mode == "constant") {
        torch::nn::init::constant_(weight, config.constant_val);
    } else {
        // 未知のmode文字列は即エラー（現状の「weight無処理・biasだけゼロ初期化」という非対称な潜在バグを修正）
        ANET_SYSTEM_ERROR("Unknown WeightInitConfig.mode: \"" << config.mode
            << "\" expected one of: default, xavier, he, orthogonal, constant");
    }

    auto& bias = layer->bias;
    if (bias.defined() && config.mode != "default") {
        if (config.mode == "constant") {
            torch::nn::init::constant_(bias, config.constant_val);
        } else {
            torch::nn::init::constant_(bias, 0.0);
        }
    }
}
```

`else`分岐（未知値エラー）を weight 側の if-else チェーンの最後に追加することで、`mode=="default"` の早期returnと合わせて全ケースを網羅する。bias側のロジックは元のまま（`mode!="default"` かつ `mode!="constant"` ならゼロ初期化）で変更不要。

### C. コード内デフォルト値の書き換え（`nn_modules.cpp`）

数値リテラルを対応する文字列へ機械的に置換する。実装時は `grep -rn "\.mode\s*=\s*[0-9]" core/anet-core/src` で全箇所を洗い出し、下記マッピングに従って書き換える。

| 旧(数値) | 新(文字列) |
|---|---|
| `0` | `"default"` |
| `1` | `"xavier"` |
| `2` | `"he"` |
| `3` | `"orthogonal"` |
| `4` | `"constant"` |

確認済みの該当箇所（前提事実節を参照）: `ResBlockModuleFactory::Config`（`:901-905`、3箇所）、`Conv1dModuleFactory::Config`（`:1886`）、`Conv2dModuleFactory::Config`（`:1914`）。加えて `rainbow_agent.cpp` / `muzero_based_agent.cpp` / `dqn_based_heads.cpp` / `dqn_based_heads.hpp` / `default_dqn_agent.cpp` / `dqn_based_test.cpp` / `nn_test.cpp` 内の該当箇所も同様に置換する。

### D. 設定ファイルの書き換え（`apps/runner/config/*.txt`）

前提事実節の表にある7ファイル・52箇所を、上記マッピング表に従って機械的に置換する（`init.mode = 2` → `init.mode = he` 等）。`apps/runner/tools/dropmerge_optuna.py:382,420,425` も同様（Pythonの文字列リテラル書き換え）。`apps/runner/11_batch_run.bat` はコメントアウト済みで実害がないため対応不要（任意）。`runs/*/config/*.txt` は書き換えない。

## 非対象（Out of Scope）

- `trunc_normal` モードの追加そのもの（`028_cnblock_convnext_10prd.md` で扱う。本書は既存5値の文字列化のみ）。
- `nonlinearity` フィールド（既にstring）や `manual_gain`/`constant_val` の設計変更。
- 数値との後方互換・fallback・WARN（意図的に入れない）。
- `GetNonlinearityType` の未知文字列フォールバック挙動の修正（既存のまま、本書のスコープ外）。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/nn.hpp:48-53` | `WeightInitConfig.mode` を `int`→`std::string`、コメント更新 |
| `core/anet-core/src/nn_impl.hpp:18-72` | `WeightInitializer::Initialize` の比較ロジックを文字列化、未知値エラー分岐を追加 |
| `core/anet-core/src/nn_modules.cpp` | コード内デフォルト値の書き換え（`:901-905`, `:1886`, `:1914` 他） |
| `rainbow_agent.cpp` / `muzero_based_agent.cpp` / `dqn_based_heads.cpp` / `dqn_based_heads.hpp` / `default_dqn_agent.cpp` / `dqn_based_test.cpp` / `nn_test.cpp` | `WeightInitConfig.mode` 数値リテラルの書き換え（実装時に再grepして洗い出す） |
| `apps/runner/config/{nn,ImageCls,LunarLander,CartPole,GridMaze,agent,DropMerge}.txt` | `init.mode = N` / `head_init.mode = N` を文字列表記へ一括置換（計52箇所） |
| `apps/runner/tools/dropmerge_optuna.py:382,420,425` | 生成する設定文字列リテラルを文字列表記へ |
| `docs/adr/0008-weight-init-mode-string.md` | 新規ADR（判断5） |

## 受け入れ基準

1. **ビルド緑**（x64-Debug、`anet-core-test`）。
2. 書き換え後の全設定ファイルで、既存のResBlock/Conv2d/Linear等のネットワークが**従来と数値的に同一の初期化結果**になること（同一seedで乱数消費パターンが変わらないことを確認。`mode`の意味自体は変えていないため、文字列化前後で数値は完全一致するはず）。
3. 未知文字列（例: `init.mode = foo`）指定時に `ANET_SYSTEM_ERROR` で例外発生することをテストで確認。
4. `mode` 未指定時（デフォルト`"xavier"`）が従来の数値デフォルト（`1`=Xavier）と同じ挙動になること。
5. `nn_test.cpp` 内の既存 `WeightInitConfig` 関連テストを文字列表記に更新し、緑になること。

## 正直なリスク / 注意

- **52箇所の機械的置換は grep漏れのリスクがある**。実装時は `grep -rn "\.mode\s*=\s*[0-9]"` を `core/anet-core/src` と `apps/runner/config` の両方で最終確認し、置換漏れがゼロであることを目視で保証すること。置換漏れがあると、文字列比較（`config.mode == "xavier"`）に対して数値由来の文字列（例: 設定ファイルの `init.mode = 2` がstring型フィールドにそのまま読み込まれた場合の `"2"`)が渡り、`ANET_SYSTEM_ERROR` で即座に落ちる（fail-fastなので実害は起動時エラーに留まる）。
- **`WeightInitConfig` を使う周辺ファイル**（`rainbow_agent.cpp` 等）は本書執筆時点で内容を精査していない。実装時に個別確認が必要。
- 未知mode文字列を即エラーにする挙動改善（判断3）は、現状の「weight無処理・biasゼロ初期化」という非対称バグを踏んでいた既存configが仮にあった場合、それを顕在化させる（fail-fastになる）。現状の52箇所は全て0~4の範囲内と確認済みのため、この顕在化は起きない見込み。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[nn]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

- 機能確認: ImageCls / DropMerge / LunarLander / CartPole / GridMaze の各 runner を起動し、config パースエラーが出ないこと、既存branchが従来通り学習を開始できることを確認。
- 数値的回帰確認: 書き換え前後で同一seed・同一configの学習を数ステップ実行し、loss/出力が完全一致することを確認（初期化ロジック自体は変えていないため一致するはず）。

## 後続

1. 本書適用後、`028_cnblock_convnext_10prd.md`（`trunc_normal` モード追加＋CNBlock導入）に着手する。
2. ADR `0008-weight-init-mode-string.md` を本書と合わせて作成する（判断5）。
