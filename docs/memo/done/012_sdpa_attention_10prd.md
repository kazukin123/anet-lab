# SDPA / FlashAttention 化（`CustomTransformerEncoderLayer` の self-attention 経路差し替え）仕様書

## Context（背景・目的）

`CustomTransformerEncoderLayer`（`core/anet-core/src/nn_modules.cpp:1251-1346`）の self-attention が学習 forward の GPU ホットスポット。3 トレースで確定済み：

- **Tracy / NSight(launch_blocking=OFF)** では CPU 側 NVTX が backpressure 待ちを吸って見かけが歪む（`attn_norm` が 1.273ms に膨張する等）。
- **NSight(launch_blocking=ON)＝真の GPU 帰属**では `CustomTransformerEncoderLayer::forward.self_attn` が **1.571ms（layer 2.428ms の 65%）**、`TransformerEncoderModule::Forward` が **forward 14.7ms の 63%**。→ forward の GPU 時間は attention に集中。

原因は LibTorch の `torch::nn::MultiheadAttention` を使っていること（`nn_modules.cpp:1295,1321`）：

1. **非 Fused**: C++ MHA は `linear → chunk → view/transpose → bmm → softmax → dropout → bmm → linear` に分解され（`activation.h:686,792,825,882,904,908,912`）、`[bsz*num_heads, tgt_len, src_len]` の attention 行列を毎回 materialize する（DropMerge の `TransEnc_128` は seq≈160, nhead=8, batch=128 で `[1024,160,160]≈26M` 要素 ×4 layer）。softmax は memory-bound。
2. **`torch::equal` 同期**: C++ 版 `multi_head_attention_forward` は self-attention 判定を `torch::equal(query,key) && torch::equal(key,value)`（`activation.h:683`）で行う。Python の `query is key` ポインタ恒等短絡が **C++ には無い**ため、同一テンソルを 3 つ渡しても eq カーネル + `all()` リダクション + `.item()` D2H が走る。1 層 2 回 ×4 層＝8 回/forward の冗長カーネル兼同期点。
3. **need_weights=true 既定**（`activation.h:830` の forward デフォルト）で、`std::get<0>` で捨てている注意重みの平均化（`activation.h:919`）まで計算している。

これらを、ATen の `at::scaled_dot_product_attention`（FlashAttention / mem-efficient へ自動ディスパッチ）を直接呼ぶ手書き self-attention に置き換えて解消する。`torch::nn::MultiheadAttention` は **パラメータ保持器として残し、forward 経路のみ差し替える**ことで、学習済み重み・チェックポイントを完全互換に保つ。

期待効果（NSight(ON) の self_attn を基準）：softmax + attention 行列 materialize の消滅（flash は行列を materialize しない）、`torch::equal` 8 回の除去、注意重み平均化の除去、`[S,B,E]` 転置往復の除去。AMP(fp16/bf16) パスで flash backend が効くため最大、fp32 では mem-efficient/math フォールバックで効果は中程度。

## 1. 前提事実（調査済み・再調査不要）

### 1.1 SDPA op シグネチャ（`ATen/ops/scaled_dot_product_attention.h` で確認済み、LibTorch 2.11.0+cu130）

```cpp
// torch/torch.h 経由で宣言済み（追加 include 不要）。torch:: 別名も可。
at::Tensor at::scaled_dot_product_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask = {},
    double dropout_p = 0.0, bool is_causal = false,
    std::optional<double> scale = std::nullopt,
    bool enable_gqa = false);
```

- 入力は `[..., L, E_head]`。本件は `[B, num_heads, S, head_dim]` を渡す。出力は query と同形 `[B, num_heads, S, head_dim]`。
- **既定 scale = `1/sqrt(query.size(-1))` = `1/sqrt(head_dim)`** で、nn::MHA の `scaling = 1/std::sqrt(head_dim)`（`activation.h:679`）と一致 → **`scale` は渡さない（nullopt）**。q を事前スケールしない（SDPA が内部で適用する）。
- **CompositeImplicitAutograd** op なので autograd 対応済み。backward は fused backward が自動で使われる（手書き backward 不要）。
- backend（flash / mem-efficient / math）は入力 dtype・形状・SDP context から自動選択。flash は half 精度（fp16/bf16）が条件。

### 1.2 `MultiheadAttentionImpl` のメンバ（`torch/nn/modules/activation.h:822-865` で確認済み、すべて public）

```cpp
MultiheadAttentionOptions options;   // options.num_heads(), options.dropout() 等
bool _qkv_same_embed_dim{};          // true: in_proj_weight 結合 / false: q/k/v_proj_weight 分離
Tensor in_proj_weight;               // [3*E, E]
Tensor in_proj_bias;                 // [3*E]
Tensor bias_k, bias_v;               // add_bias_kv 用（既定 undefined）
Linear out_proj = nullptr;           // out_proj->weight [E,E], out_proj->bias [E]
Tensor q_proj_weight, k_proj_weight, v_proj_weight;  // _qkv_same_embed_dim=false 時のみ
int64_t head_dim{};                  // = E / num_heads
```

`mha_`（`torch::nn::MultiheadAttention` ModuleHolder）から `mha_->in_proj_weight` 等で直接アクセス可。`register_module("self_attn", ...)` を維持する限り、パラメータ名 `self_attn.in_proj_weight` / `self_attn.in_proj_bias` / `self_attn.out_proj.weight` / `self_attn.out_proj.bias` は不変。

### 1.3 nn::MHA の正確な計算（差し替えで再現すべきセマンティクス、`activation.h` で確認済み）

現状の構築は `MultiheadAttentionOptions(d_model, nhead)` のみ（`nn_modules.cpp:1257`）＝既定オプション：`dropout=0`, `bias=true`, `add_bias_kv=false`, `add_zero_attn=false`, `_qkv_same_embed_dim=true`。呼び出しは self（Q=K=V）, key_padding_mask/attn_mask なし。よって再現すべき計算は：

1. `qkv = linear(x, in_proj_weight, in_proj_bias)` → 最終次元で `chunk(3)` → q, k, v（`activation.h:686`）
2. ヘッド分割: 最終次元 E を `(num_heads, head_dim)`（head_dim 最内・連続）に分割（`activation.h:825`）
3. `attn = softmax(q·kᵀ / sqrt(head_dim)) · v`（`activation.h:792,882,904,908`）
4. ヘッド結合 → `out = linear(attn, out_proj.weight, out_proj.bias)`（`activation.h:912` 以降）

→ 1.〜3. を `at::scaled_dot_product_attention` が等価に行う（modulo fp 演算順序・backend 差）。

### 1.4 入力形状とチャネル順

`CustomTransformerEncoderLayer::forward` の入力 `src` は **`[Batch, SeqLen, d_model]`**（`nn_modules.cpp:1278`）。現状は nn::MHA が `[SeqLen, Batch, E]` 固定のため `transpose(0,1)` で往復している（`nn_modules.cpp:1292,1299,1320,1323`）。手書き SDPA は **`[B,S,E]` のまま処理でき、この転置往復を全廃**できる（self-attention は per-token 射影なので `[B,S,E]`／`[S,B,E]` で結果同一）。

## 2. 設計方針

**Approach A を採用：`torch::nn::MultiheadAttention` submodule をパラメータ保持器として残し、`mha_->forward(...)` の呼び出しだけを手書き SDPA ヘルパに差し替える。**

- 長所: パラメータ名・初期化・チェックポイントが完全不変（移行不要）。`torch::equal`・非 fused・転置往復・need_weights を一掃。最小差分。
- 採用しなかった案: **Approach B**（`self_attn` を自前 `qkv_proj`/`out_proj` Linear に置換）。パラメータ名が変わり旧チェックポイント非互換・重み移行が必要になるため棄却。フレームワーク既定の重み初期化（`_reset_parameters`）も再実装が要る。

A/B 計測と安全な切り戻しのため config フラグ `tf.use_sdpa`（既定 true）で旧 nn::MHA 経路と切替可能にする（FusedAdamW の `use_fused_optimizer` 前例に倣う）。submodule はどちらのモードでも生成・登録されるため、フラグはチェックポイントに影響しない。

## 3. 手書き SDPA ヘルパ（`core/anet-core/src/nn_modules.cpp`、`CustomTransformerEncoderLayer` の直前に追加）

`namespace anet` 直下の自由関数（**無名 namespace 禁止**：プロジェクト規約）。

```cpp
// self-attention(Q=K=V) を at::scaled_dot_product_attention で計算する。
// mha の学習済みパラメータ（in_proj / out_proj）を流用し、計算のみ fused 経路に置換する。
// 入出力ともに [B, S, E]。
torch::Tensor SdpaSelfAttention(const torch::nn::MultiheadAttention& mha, const torch::Tensor& x)
{
    namespace F = torch::nn::functional;

    // 未対応オプションは暗黙フォールバックせず明示的に落とす（規約）
    ANET_CHECK_MSG(mha->_qkv_same_embed_dim,
        "SdpaSelfAttention: separate q/k/v projection is not supported.");
    ANET_CHECK_MSG(!mha->bias_k.defined() && !mha->bias_v.defined(),
        "SdpaSelfAttention: add_bias_kv is not supported.");

    const int64_t B = x.size(0);
    const int64_t S = x.size(1);
    const int64_t E = x.size(2);
    const int64_t num_heads = mha->options.num_heads();
    const int64_t head_dim = mha->head_dim;          // = E / num_heads

    // in-proj（QKV 一括）→ [B, S, 3E] → 最終次元で 3 分割
    torch::Tensor qkv = F::linear(x, mha->in_proj_weight, mha->in_proj_bias);
    std::vector<torch::Tensor> chunks = qkv.chunk(3, /*dim=*/-1);   // 各 [B, S, E]

    // [B,S,E] → [B,S,H,head_dim] → [B,H,S,head_dim]（head_dim 最内＝nn::MHA と同じ分割順）
    auto to_heads = [&](const torch::Tensor& t) {
        return t.reshape({B, S, num_heads, head_dim}).transpose(1, 2);
    };
    torch::Tensor q = to_heads(chunks[0]);
    torch::Tensor k = to_heads(chunks[1]);
    torch::Tensor v = to_heads(chunks[2]);

    // dropout は学習時のみ（現状 config は dropout 未設定＝0。将来用に options から拾う）
    const double dropout_p = mha->is_training() ? mha->options.dropout() : 0.0;

    // fused SDPA。scale 既定 = 1/sqrt(head_dim) が nn::MHA と一致するため scale は渡さない。
    torch::Tensor attn = at::scaled_dot_product_attention(
        q, k, v, /*attn_mask=*/{}, dropout_p, /*is_causal=*/false);   // [B,H,S,head_dim]

    // ヘッド結合 [B,H,S,head_dim] → [B,S,H,head_dim] → [B,S,E]
    attn = attn.transpose(1, 2).reshape({B, S, E});

    // out-proj
    return F::linear(attn, mha->out_proj->weight, mha->out_proj->bias);
}
```

実装注記：
- `transpose(1,2)` 後は非連続。flash backend が選ばれない場合の保険として `q/k/v` に `.contiguous()` を付ける選択肢があるが、まず無しで backend 選択を NSight で確認し、flash が出なければ追加（チューニング項目、§9-3）。
- `attn.transpose(1,2).reshape(...)` の reshape は非連続入力でも内部でコピーするため `.contiguous().view()` 同等。
- `ANET_CHECK_MSG`（既存利用、`nn_modules.cpp:1363` に前例）。

## 4. `CustomTransformerEncoderLayer::forward` 改修（`nn_modules.cpp:1274-1336`）

`use_sdpa_` で分岐。SDPA 経路では転置往復を除去し、profile 区間名 `self_attn` は維持（前後比較のため）。

### 4.1 Pre-LN パス（`nn_modules.cpp:1286-1299`）

```cpp
// --- Attention Block ---
ANET_PROFILE_SCOPE(attn_norm);
torch::Tensor x_norm = norm1_->forward(x);

ANET_PROFILE_SCOPE_NEXT(self_attn);
torch::Tensor attn_out;
if (use_sdpa_) {
    attn_out = anet::SdpaSelfAttention(mha_, x_norm);             // [B,S,E]、転置不要
} else {
    torch::Tensor x_norm_t = x_norm.transpose(0, 1);             // 旧経路（A/B 用に温存）
    attn_out = std::get<0>(mha_->forward(x_norm_t, x_norm_t, x_norm_t)).transpose(0, 1);
}

ANET_PROFILE_SCOPE_NEXT(attn_residual);
x = x + attn_out;
// （以降の FFN ブロックは変更なし）
```

### 4.2 Post-LN パス（`nn_modules.cpp:1319-1323`）

```cpp
ANET_PROFILE_SCOPE(self_attn);
torch::Tensor attn_out;
if (use_sdpa_) {
    attn_out = anet::SdpaSelfAttention(mha_, x);
} else {
    torch::Tensor x_t = x.transpose(0, 1);
    attn_out = std::get<0>(mha_->forward(x_t, x_t, x_t)).transpose(0, 1);
}
ANET_PROFILE_SCOPE_NEXT(attn_residual_norm);
x = norm1_->forward(x + attn_out);
// （以降の FFN ブロックは変更なし）
```

## 5. フラグ配線（`tf.use_sdpa`）

1. **`TransformerConfig`**（`nn_modules.cpp:1348-1355`）に `bool use_sdpa = true;`（Doxygen コメント付き）を追加。
2. **`CustomTransformerEncoderLayer` コンストラクタ**（`nn_modules.cpp:1253`）に末尾引数 `bool use_sdpa` を追加し、メンバ `bool use_sdpa_;` に保持。
3. **`TransformerEncoderModule` のレイヤ生成ループ**（`nn_modules.cpp:1366-1369`）で `config_.use_sdpa` を渡す。
4. **`TransformerEncoderModuleFactory::Config`**（`nn_modules.cpp:1441-1447`）に `ANET_READ_CONFIG(config_data, tf.use_sdpa);` を追加。
5. **`GetCurrentConfigData`**（`nn_modules.cpp:1418-1428`）に `cd.Set("use_sdpa", ToConfigBool(config_.use_sdpa));` を追加（既存メトリクス出力に合わせる）。

## 6. 外部仕様（config 追加）

| キー | 型 | 既定値 | 意味 |
|---|---|---|---|
| `net.block.[*].tf.use_sdpa` | bool | `true` | self-attention を `at::scaled_dot_product_attention`（Flash/mem-efficient）で計算。false で従来 `torch::nn::MultiheadAttention.forward` 経路（非 fused・`torch::equal` 同期あり、挙動差の切り分け用） |

- `apps/runner/config/nn.txt` の `[TransEnc]` ブロック（`:146-152`）に `net.block.[TransEnc].tf.use_sdpa = true` + 用途コメントを追記。
- `apps/runner/config/DropMerge.txt` の `[TransEnc_64]` / `[TransEnc_128]` / `[TransEnc_256]` ブロック（`:698-715`）にも同キーを追記。
- 未指定時は `ANET_READ_CONFIG` が構造体既定値 `true` を採用（既存読み込みフローと同じ）。

## 7. 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/src/nn_modules.cpp` | `SdpaSelfAttention` ヘルパ追加、`forward` 2 パスの分岐化、`TransformerConfig` / レイヤ構築 / Factory / `GetCurrentConfigData` の `use_sdpa` 配線 |
| `apps/runner/config/nn.txt` | `[TransEnc].tf.use_sdpa` 追記 |
| `apps/runner/config/DropMerge.txt` | `[TransEnc_64/128/256].tf.use_sdpa` 追記 |
| `core/anet-core/src/nn_test.cpp` | テスト追加（§9） |

`muzero_based_agent.cpp` 等の他 Transformer 利用箇所は本 PRD のスコープ外（同じ `TransformerEncoderModule` を経由するなら自動的に効く）。

## 8. 既存利用可能な部品（再利用先）

- `at::scaled_dot_product_attention`（`torch/torch.h` 経由で宣言済み）。
- `torch::nn::functional::linear`（既存 `linear1_->forward` 等と同じ F::linear）。
- `ANET_PROFILE_SCOPE` / `ANET_PROFILE_SCOPE_NEXT`（`anet/profile.hpp`、区間名 `self_attn` 等は維持）。
- `ANET_CHECK_MSG`（`nn_modules.cpp:1363` に前例）/ `ToConfigBool`（`nn_modules.cpp:1425`）。
- `ANET_READ_CONFIG`（Factory 既存パターン）。
- `torch::nn::MultiheadAttention` の public メンバ（§1.2）。

## 9. 検証方針

テストは `core/anet-core/src/nn_test.cpp` に追加（`anet-core-test` ターゲット、`*_test.cpp` 同居規約）。

1. **数値一致（核心）**: 同一構成（d_model=128, nhead=8, dim_ff=512 など複数）の `CustomTransformerEncoderLayer` を 1 個生成し、**同一パラメータ・同一入力**で `use_sdpa=true`（`SdpaSelfAttention`）と `use_sdpa=false`（nn::MHA）の forward 出力を `allclose`（fp32, rtol≈1e-4, atol≈1e-5。backend 差のため bitwise 非一致）。norm_first ∈ {true, false} の両パス。
2. **勾配一致**: 上記同構成で出力の `.sum()` を backward し、各パラメータ grad を両経路で `allclose`（fused backward の妥当性確認）。
3. **ヘッド分割の同型性**: `SdpaSelfAttention` の `to_heads` 後 `[B,H,S,head_dim]` から手動 bmm+softmax で計算した参照値と SDPA 出力が一致（scale/分割順のリグレッション固定）。
4. **チェックポイント互換**: nn::MHA を含む層を save → `use_sdpa=true` で load → forward が load 前と一致（パラメータ名不変の確認。Approach A の要）。
5. **未対応オプション検出**: `add_bias_kv=true` 等で構築した MHA に `SdpaSelfAttention` を適用 → `ANET_CHECK_MSG` で明示的に落ちる。
6. **CUDA 経路**: `torch::cuda::is_available()` 時、CUDA 上でも 1.〜2. を実施（half 精度での flash 経路を含めるなら autocast 下でも実行）。
7. **ビルド**: VsDevCmd 経由で x64-Debug をビルドし `core\anet-core\bin\Debug\anet-core-test.exe` を実行（AGENTS.md 必須事項）。
8. **実測（ユーザー実施）**: DropMerge run（`TransEnc_128`）で NSight(launch_blocking=ON) の `self_attn` 区間と steps/sec を前後比較。`tf.use_sdpa=false` で従来挙動に戻ることも確認。AMP パスで flash backend が選択されているか NSight の kernel 名で確認（`§9-3` の contiguous 要否判断）。

## 10. Out of Scope

- causal mask / key_padding_mask / cross-attention 対応（self-attention・全可視のみ。`is_causal=false`, mask なし）。
- attention dropout の新規有効化（現状 config 未設定＝0。ヘルパは options から拾うが値は 0 のまま）。
- separate q/k/v projection（`_qkv_same_embed_dim=false`）対応（`ANET_CHECK_MSG` で拒否）。
- Approach B（自前 Linear への置換）・重み移行。
- FFN / LayerNorm の融合や CUDA Graphs 化（別 PRD 候補。本件は self_attn 経路のみ）。
- `sdp_kernel` context による backend 強制指定（まず自動選択で計測し、必要なら別途）。
