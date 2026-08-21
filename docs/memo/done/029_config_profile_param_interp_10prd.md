# net.config_profile によるブロック群パラメータ補間機構

> 設計分担: Claude=設計/PRD、実装=Codex。本書は self-contained。
> 動機の上流は `028_cnblock_convnext_10prd.md`（ConvNeXt v1 の CNBlock 導入）。ConvNeXt の Linear Stochastic Depth Schedule（全18ブロックで droppath_rate を 0→0.1 に線形補間）を設定ファイルに18個手書きするのは、層が増えると破綻する。本書は **branch 内のブロック群へパラメータを補間展開する汎用機構** を新設する。**CNBlock に依存しない独立機能**だが、PRD028 の droppath 補間が最初の利用者。

## Context（背景・目的）

anet-lab の NN は宣言的 DSL（`net.block` で部品定義、`net.branch` で組み立て順）で記述する。ここで ConvNeXt の droppath 線形補間のように「ブロック群に沿ってパラメータを段階的に変える」ケースを表現しようとすると、現状は各ブロックインスタンスを個別定義して値を手書きするしかない（18ブロックなら18個）。

公式 PyTorch/timm 実装では、これはモデル構築コード側で `torch.linspace(0, drop_path_rate, sum(depths))` を一度に生成し、ステージを跨ぐ通し番号で各ブロックへ配っている。設定はスカラー1個（`drop_path_rate=0.1`）だけ持つ。設定ファイルに全深さ分の値を書き下すのは業界的にも異例。

anet-lab の宣言的 DSL に、この「スカラー1個の補間ポリシー + 構築時の自動展開」を持ち込む。実装の要は `NetworkStructBuilder`（DSL を解釈してブロック列を生成する層、PyTorch でいうモデル構築コードに相当）である。

**用語**: 本書で「補間」と呼ぶのは、あるグループに属するブロック群を出現順に並べ、`i/(N-1)`（i=通し番号, N=総数）に応じて値を線形（等）に配ること。時系列（訓練ステップ）に沿った変化ではないため、`schedule` ではなく `config_profile`（`config_data` の分布の輪郭）という語を採る。

## 確定した設計判断

1. **セクション名は `net.config_profile.[name]`**（`net.block`/`net.branch`/`net.body` と同列の新トップレベル）。「`config_data` の profile（分布の輪郭）」の意。`net.profile` は既存の `ANET_PROFILE_*`（性能計測）と混同するため避け、`param` は torch の `parameters()` と紛れるため `config_` を採る。
2. **マーカーは `@<group>`**。block 定義の double フィールド値が `@` で始まればマーカーとして扱い、そのフィールドが補間対象・`<group>` が所属グループを同時に示す。例: `net.block.[CN96].cn.droppath_rate = @cn_dp01`。
   - **`$` との棲み分け**: 将来構想の `$`（config レイヤーでの静的変数置換）とは記号で分ける。config_profile は「NN構造依存の動的計算（出現順 `i/N`）」であり、`$` の静的置換とは本質的に別機構。同じ記号に乗せず、別記号 `@` にするのが設計的に正しい。
3. **補間の本質は「branch 内でのブロック出現順の通し位置 `i/(N-1)`」**。公式 ConvNeXt/timm の `torch.linspace(0, dpr, sum(depths))` をステージ跨ぎ通し番号で配るのと同じ。`(*N)` 展開も出現順にカウントするので、`CN96(*3) > ... > CN768(*3)` がそのままステージ跨ぎラダーになる（各ステージで 0 リセットしない）。
4. **スコープは branch 内**。同一グループ名が複数 branch に跨って出現したら `ANET_SYSTEM_ERROR`。初手は branch 内に閉じる（ConvNeXt は単一 branch で完結し、これで公式と一致する）。将来 `NetworkBodyBuilder` を topo 順先行に組み替えれば branch 跨ぎへ一般化できる道だけ残す。
5. **展開は `CreateModule` の前**（config_data 書き換え方式）。対象ブロック（CNBlock 等）は展開後の実数を受け取るだけで **無改造**。
6. **汎用機構**（任意の double フィールド × 補間ルール）として設計するが、初手の実証は CNBlock droppath のみ。`type=linear` のみ実装する。
7. **profile の検証と値生成は `ConfigProfile` interface に寄せる**。`ConfigProfileConfig` は読み取った設定値の保持に留め、`type=linear` 固有の必須項目チェックと `i/N` からの実効値生成は `LinearConfigProfile` が担当する。`NetworkStructBuilder` は profile object から値を受け取るだけにし、補間式や type 分岐を持たない。

## 前提事実（実コード確認済み）

> 基準コミット: HEAD `0598b1b`。`nn.hpp` / `nn_impl.cpp` / `config.hpp` / `nn_test.cpp` は未コミット変更なし。

- **セクションパース機構**: `NetworkConfig::NetworkConfig`（`nn_impl.cpp:364-367`）が `ReadBlockConfig`/`ReadBranchConfig` を呼ぶ。`ReadBlockConfig`（`nn_impl.cpp:239-298`）は正規表現 `net\.block\.\[([^\]]+)\]\.(.+)`（`:251`）で `net.block.[tag].subkey = value` を全 config から集め、tag 毎に `NetworkBlockConfig{type, config_data}` へ構造化する（`:276-294`）。**この関数を雛形に regex を `net\.config_profile\.\[([^\]]+)\]\.(.+)` へ差し替えた `ReadConfigProfileConfig` を1つ足せば新セクションを読める**。
- **config_data の全キー走査**: `ConfigData::Map()`（`config.hpp:49-52`）が内部の `OrderedMap<string,string>` を返す。**挿入順（=記述順）を保持**。`ToJson`/`ToString`（`:80-101`）が `for (const auto& kv : map_)` で舐めている通り、キー/値ペアを走査でき、値が `@` で始まるキーを検出可能。
- **config_data はコピー可・値は全て文字列**: コピーコンストラクタ（`config.hpp:24`）、`Set(key, value)`（`:31-42`、任意型を文字列化して格納）、内部は `MapType map_`（`:105`、`OrderedMap<string,string>`）。「config_data をコピーして該当キーを実数文字列で `Set` 上書き → `CreateModule`」が成立する。
  - **注意**: マーカー文字列 `@cn_dp01` のまま CNBlockFactory の `ANET_READ_CONFIG(config_data, cn.droppath_rate)`（double `Read`）に渡すと、double パースに失敗して default 0.0 に落ちる（`ConfigData::Read(double)` は `config.hpp:72`）。よって展開は必ず `CreateModule` の前に走らせる。
- **`(*N)` 展開とブロック生成順**: `NetworkStructBuilder::Build`（`nn_impl.cpp:652-693`）が `structure_str` を `SplitPipelineString` でトークン分割（`:658`）し、各トークンを順に処理。`\(\*(\d+)\)` 正規表現（`:660`）で repeat_count を抽出（`:668-671`）、`for (int r=0; r<repeat_count; ++r)`（`:673`）内で毎回 `factory->CreateModule(block_cfg.config_data, ctx)`（`:684`）。**このトークン処理順（`(*N)` 展開含む）がそのまま出現順**。インスタンス名は `block_def_name + "_" + idx`（`:686-687`）で連番。
- **branch 構築と topo ソート**: `NetworkBodyBuilder::Build`（`nn_impl.cpp:700-759`）は現状「各 branch を個別 build（`:708-709` で `NetworkStructBuilder::Build` 呼び出し）→ その後 topo ソート（`:734-752`, Kahn's algorithm）」の順。**branch 跨ぎ通し番号を将来やるならこの順序組み替え**が必要（初手は branch 内に閉じるので不要）。
- **ModuleContext**（`nn_impl.hpp:193-196`）は空の拡張用構造体。マーカー方式では config_data 書き換えで足りるため使わない（型/位置で暗黙特定する代替案でのみ必要だった）。

## 設計方針

### A. 新セクションのパース（`nn.hpp` + `nn_impl.cpp`）

`nn.hpp` に構造体を宣言し、`NetworkConfig`（`nn.hpp:38-46`）にメンバを追加:

```cpp
struct ConfigProfileConfig {
    std::string type = "linear";  ///< 補間ルール（初手はlinearのみ）
    double start = 0.0;           ///< i=0 の値（default 0.0）
    double end = 0.0;             ///< i=N-1 の値
    bool has_end = false;         ///< end 明示指定の有無（未指定はエラー）
};

struct NetworkConfig {
    std::map<std::string, NetworkBlockConfig> block_configs;
    std::map<std::string, NetworkBranchConfig> branches;
    std::map<std::string, std::string> output_keys;
    std::map<std::string, ConfigProfileConfig> config_profiles;  // 追加
    ...
};
```

`nn_impl.cpp` に `ReadConfigProfileConfig`（`ReadBlockConfig` を雛形に、regex を `net\.config_profile\.\[([^\]]+)\]\.(.+)` へ）を追加し、`NetworkConfig` コンストラクタ（`:364-367`）から `config_profiles = ReadConfigProfileConfig(config_data, config_prefix);` を呼ぶ。subkey は `type`/`start`/`end` を読み、`end` が来たら `has_end=true`。旧 `min`/`max` は後方互換 fallback せず未知キーとして fail-fast する。

読み取り後は `ConfigProfile` interface へ object 化し、`ValidateConfig()` で型固有の必須項目を検証する。初期実装では `LinearConfigProfile` のみを作る。

### B. 2パス展開（`NetworkStructBuilder::Build` 内、branch 単位）

現状の単一 for ループ（`nn_impl.cpp:663-691`）を2パス化する。`Build` は `root_config`（`config_profiles` を含む）を受け取れるので、branch 内で完結する。

**パス1（集計）**: トークンを出現順に走査（`(*N)` 展開込みでインスタンス単位に数える）。各インスタンスの `block_cfg.config_data` を `Map()` で舐め、値が `@G` 形式のものを検出。グループ G ごとに「(グローバル・インスタンス連番, 該当フィールドキー) の順序付きリスト」と総数 N を構築する。

**パス2（生成）**: 各インスタンス生成時、それがグループ G の i 番目なら、G のポリシーから `value(i)` を計算し、**config_data をコピーして該当フィールドキーを実数文字列で `Set` 上書き**してから `CreateModule` に渡す。マーカーでないフィールドはそのまま。マーカーを1つも含まないブロックは、コピーせず現状通り `block_cfg.config_data` を渡してよい（回帰リスク最小化）。

補間値の計算（linear）:
```
value(i) = start + (end - start) * i / (N - 1)  // i ∈ [0, N-1]
value(0) = start,  value(N-1) = end             // 両端含む（torch.linspace 互換）
N == 1     → value = start                      // ゼロ除算回避（torch.linspace(a,b,1)=[a] に倣う）
```

`start > end` は有効で、下降方向の linear profile として扱う。

擬似コード（`Build` 内）:
```cpp
// パス1: マーカー集計（トークン→インスタンス展開順）
struct MarkerRef { size_t instance_seq; std::string field_key; };
std::map<std::string, std::vector<MarkerRef>> groups;  // group名 → 出現順リスト
// ... トークンを (*N) 展開しながら instance_seq をインクリメントし、
//     各 config_data.Map() を走査して値が "@G" のものを groups[G] に push

// パス2: 生成
size_t instance_seq = 0;
for (each token, each repeat r) {
    ConfigData cfg = block_cfg.config_data;  // コピー
    // このインスタンスのマーカー付きフィールドを実効値へ上書き
    for (auto& [key, val] : cfg.Map()) {
        if (val が "@G" 形式) {
            const auto& refs = groups.at(G);
            int i = (refs 内でこの instance_seq の位置);
            int N = refs.size();
            double v = InterpValue(config_profiles.at(G), i, N);
            cfg.Set(key, std::to_string(v));
        }
    }
    auto module = factory->CreateModule(cfg, ctx);
    ...
    instance_seq++;
}
```

（実装上は「instance_seq → グループ内での i」を引ける索引をパス1で作っておくと2重ループを避けられる。ここでは方針を示すに留める。）

### C. バリデーション（fail-fast、AGENTS.md 流）

- `@G` に対応する `net.config_profile.[G]` が `config_profiles` に無い → `ANET_SYSTEM_ERROR`（グループ名・使用箇所を含める）。
- `config_profiles.[G].has_end == false`（`end` 未指定）→ `ANET_SYSTEM_ERROR`。`start` 未指定は 0.0 で可。
- `type` が未知（`linear` 以外）→ `ANET_SYSTEM_ERROR`（将来 cosine 等を足すまで）。
- 旧 `min`/`max` は未知キーとして `ANET_SYSTEM_ERROR`。後方互換 fallback は入れない。
- `net.config_profile.[G]` は定義されているが使用ブロックが 0 個 → `LOG::warn()`（設定ミス示唆、無害。`ANET_LOG_WARN` ではなく `LOG::warn()`）。
- 同一グループ G が複数 branch に跨って出現 → `ANET_SYSTEM_ERROR`（初手はスコープ外。`NetworkStructBuilder::Build` が branch 単位で呼ばれる以上、グループ名の branch 間重複は `NetworkBodyBuilder` 側で検出する必要がある。実装は「build 済み branch のグループ名集合を保持し重複を弾く」で足りる）。

### D. config dump / ground truth

展開後の実効値（実数）が config_data に入るので、各ブロックの `GetCurrentConfigData` は実効値をダンプ → `runs/<name>/config/config_data.txt` に per-block の実数 droppath が残る（[[feedback_check_run_config_dump]] と整合）。「元がマーカーだった」情報は消えるが、ポリシー定義（`net.config_profile.[G]`）自体が config として dump されるので辿れる。

## 使用例（PRD028 の ConvNeXt-Tiny）

```
# ブロック定義（channel別4定義、droppathはマーカー）
net.block.[CN96].type  = CNBlock
net.block.[CN96].cn.channels = 96
net.block.[CN96].cn.droppath_rate = @cn_dp01
# CN192 / CN384 / CN768 も同様に cn.droppath_rate = @cn_dp01

# 補間ポリシー
net.config_profile.[cn_dp01].type = linear
net.config_profile.[cn_dp01].start = 0.0
net.config_profile.[cn_dp01].end   = 0.1

# structure（(*N) 展開、出現順=ステージ跨ぎ通し番号）
net.branch.ConvNeXtT.structure = ... > CN96(*3) > ... > CN192(*3) > ... > CN384(*9) > ... > CN768(*3) > ...
```

→ 計18個の CNBlock に、出現順で `linspace(0, 0.1, 18)` の droppath が自動配分される。ステージ跨ぎで単調増加（CN96群 0.0000/0.0059/0.0118 → … → CN768群 0.0882/0.0941/0.1000）。

## 非対象（Out of Scope）

- `type=cosine`/`step`/明示 `list` 等の他補間ルール（将来）。初手は linear のみ。
- branch 跨ぎ集計（将来、`NetworkBodyBuilder` の topo 順先行組み替え）。
- 順序明示構文（`@G#3` のように出現順でなく明示 index を振る）。
- `$` config レイヤー静的置換機構（別構想、本書とは別記号・別機構）。
- double 以外のフィールド（int/string）への補間。

## 影響ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/nn.hpp` | `ConfigProfileConfig` 構造体宣言、`NetworkConfig` に `config_profiles` メンバ追加 |
| `core/anet-core/src/nn_impl.cpp` | `ReadConfigProfileConfig` 追加（`ReadBlockConfig` 雛形）／`ConfigProfile` interface と `LinearConfigProfile` 追加／`NetworkConfig` コンストラクタから呼ぶ／`NetworkStructBuilder::Build` を2パス化しマーカー検出+補間展開／`NetworkBodyBuilder::Build` に branch 跨ぎ重複検出 |
| `core/anet-core/src/nn_test.cpp` | `[nn][config_profile]` 単体テスト追加 |
| `apps/runner/config/nn.txt` | コメントアウト済みの `config_profile` 最小サンプル追加 |

## 受け入れ基準

1. **ビルド緑**（x64-Debug、`anet-core-test`）。
2. `CN96(*3) > ... > CN768(*3)`（計18）+ `@cn_dp01` + `net.config_profile.[cn_dp01](linear, start=0, end=0.1)` で、各 CNBlock の実効 droppath が `linspace(0, 0.1, 18)` に一致（config dump または直接検査で per-block 実数を確認）。
3. **出現順がステージ跨ぎで通る**こと（CN96群→CN768群で単調増加、各ステージ 0 リセットしない）。
4. `@` 未定義グループ → `ANET_SYSTEM_ERROR`。`end` 未指定 → `ANET_SYSTEM_ERROR`。旧 `min/max` → `ANET_SYSTEM_ERROR`。未知 type → `ANET_SYSTEM_ERROR`。使用 0 個 → `LOG::warn()`。同一グループ branch 跨ぎ → `ANET_SYSTEM_ERROR`。
5. **N=1 グループで `start` を返す**（ゼロ除算しない）。
6. **マーカーを含まない既存 branch**（ResNet18ish/ViT 等）が無影響で回帰なし（パス1でマーカー検出0なら従来と完全同一のブロック生成になること）。

## テスト項目リスト

1. linear 補間値の正確性（N=18 で `linspace(0,0.1,18)` 一致）
2. 出現順ステージ跨ぎ（channel 別ブロックを跨いで単調増加）
3. `(*N)` 展開との併用（`CN96(*3)` が3インスタンスとしてカウントされる）
4. 未定義グループ → エラー
5. `end` 未指定 → エラー
6. 旧 `min/max` → エラー
7. 未知 type → エラー
8. 使用 0 個 → warn
9. 同一グループ branch 跨ぎ → エラー
10. N=1 のとき `start` 返し（ゼロ除算なし）
11. `NetworkConfig::ToJson()` が `start/end` を出す
12. マーカー無しブロックの無影響（既存 branch 回帰なし）

## 正直なリスク / 注意

- **`NetworkStructBuilder::Build` の2パス化は既存の単純 for ループ（`nn_impl.cpp:663-691`）を触る**ため、マーカー無しの既存 branch（ResNet18ish/Hybrid/ViT/LunarLander MLP 等）が回帰しないことが最重要。パス1でマーカー検出が 0 なら、パス2はコピーも上書きもせず現状の `CreateModule(block_cfg.config_data, ctx)` と完全同一経路を通ること。テストで既存 branch の生成結果不変を担保する。
- **`@` プレフィックスが既存の net.block 値と衝突しないこと**: 既存 config の double フィールド値に `@` 始まりは無いと確認済み（`init.mode`/`num_features` 等いずれも数値または通常文字列）。metrics DSL の `$runner` 等とも別レイヤー（あちらは metrics パーサ、こちらは net.block パーサ）。
- **branch 跨ぎ重複検出の実装位置**: `NetworkStructBuilder::Build` は branch 単位で呼ばれるため、単体ではグループ名の branch 間重複を検出できない。`NetworkBodyBuilder::Build`（全 branch を build する層、`:700-716`）で「各 branch が使ったグループ名集合」を集約し重複を弾く必要がある。実装時はこの2層の責務分担に注意。

## 検証

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[config_profile]"
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check -- core/anet-core/include/anet/nn.hpp core/anet-core/src/nn_impl.cpp core/anet-core/src/nn_test.cpp apps/runner/config/nn.txt docs/memo/029_config_profile_param_interp_10prd.md docs/memo/029_config_profile_param_interp_20impl.md
```

- 機能確認: ImageCls の ConvNeXtT branch（PRD028）を起動し、`runs/<name>/config/config_data.txt` に18個の補間済み droppath 実数が per-block で載ることを確認。
- 回帰確認: マーカーを使わない既存 branch（ResNet18ish 等）を起動し、従来通り学習が回ること。

## 後続

1. 本書は PRD027（mode 文字列化）と独立に実装可能。PRD028（CNBlock）は本書 + 027 に依存し、3点が揃って初めて ConvNeXt-Tiny が実働評価できる（実装・評価は束ねる）。
2. 将来必要なら `type=cosine`/`step`、branch 跨ぎ集計、順序明示構文を別 PRD で拡張する。
