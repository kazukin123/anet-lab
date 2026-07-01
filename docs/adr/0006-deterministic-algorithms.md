# 学習の同 seed 再現性のため `setDeterministicAlgorithms` をグローバル既定で有効化する

学習 Run の同 seed → 同結果（再現性）が失われていた。真因は **SDPA（`at::scaled_dot_product_attention`、`nn_modules.cpp:1289`）が CUDA で Memory-Efficient backend に落ち、その backward が gradient を atomic 加算するため非決定**であること（forward は決定的なので「eval は再現・train だけ run ごとに割れる」症状）。既定 ON の `backend.cudnn_deterministic` は **cuDNN 畳み込み専用**で、ATen の flash/mem-efficient カーネルである SDPA には効かない穴だった。

`torch::globalContext().setDeterministicAlgorithms(true, /*warn_only=*/false)` の 1 行で再現性が復帰することを実機確認した（mem-efficient は決定的 backward 変種を持ち、本フラグで atomic 回避経路に切替わる）。これを `backend.deterministic_algorithms`（既定 true）+ `backend.deterministic_warn_only`（既定 false）として config 化する。実装手順は `docs/memo/015_deterministic_algorithms_10prd.md`。

## Considered Options

- **現状維持（何もしない）**: train が同 seed 非再現のまま。棄却。
- **SDPA backend を math に固定（`setSDPUseMath(true)` + flash/efficient/cuDNN を無効）**: 決定化はできるが math は遅い。mem-efficient の決定 backward で同じ再現性が得られ、backend を局所的に縛る複雑さ・脆さ（版差）を払う価値がない。棄却（将来 flash が選ばれて決定 backward 非対応で throw した場合の退避策としてのみ、PRD にコメントとして残す）。
- **SDPA 化を撤回し旧 `torch::nn::MultiheadAttention` 経路へ戻す**: SDPA は実時間メリットが出なかった（attention は実時間ボトルネックでない）が、非決定は他の atomic 系 op でも起こり得るため本質解でなく、差分・互換コストも大きい。棄却（本 ADR の範囲外）。
- **`setDeterministicAlgorithms` を既定 false（opt-in）で導入**: 速度優先・将来 op 追加での throw 事故を避けられるが、本プロジェクトは `cudnn_deterministic=true` で既に「決定論的を既定」方針。SDPA という穴を既定で塞ぐべきで、ここだけ opt-in にするのは一貫しない。棄却。
- **`setDeterministicAlgorithms(true, warn_only=false)` をグローバル既定で有効化（採用）**: 1 行で SDPA に限らず全 ATen op の非決定を塞ぐ。`cudnn_deterministic=true` と同方針・別レイヤーで併用。決定版が無い op に当たった時の退避用に `deterministic_warn_only` を公開。`setSDPUse*` / `CUBLAS_WORKSPACE_CONFIG` はコードに入れず参照コメントとして残す（最小コード）。

## Consequences

- **再現性＝同 seed → 同結果（train 含む）**。`deterministic_algorithms=false` の従来とは結果が変わり得る（厳密比較は設定を合わせる）。`cudnn_deterministic`（cuDNN 畳み込み限定）は別レイヤーとして据え置き・併用。
- **backend グローバル設定なので全 Run（DQN/QR/Rainbow/MuZero 問わず）に効く**。`InitRL` 一箇所で適用。メトリクス（`Log(backend_config)`）に 2 フラグが自動で載る。
- **throw リスク**: 決定版が無い op を踏むと `warn_only=false` では op 名付きで throw（既定 true なので Run が落ちる）。これは silent 非決定より望ましい失敗（loud failure）。落ちた場合は `deterministic_warn_only=true` で再ビルド無しに非決定運転へ退避できる（**ただし再現性は失う**、診断・暫定運転用）。
- **`CUBLAS_WORKSPACE_CONFIG`**: 決定モードで cuBLAS GEMM がこの env（`:4096:8` 等）を要求する場合がある。**失敗モードは silent でなく throw**。当該環境では不要だった（throw せず再現）。将来 CUDA/cuBLAS/形状変更で要求 throw が出たら、CUDA 初期化前に env を設定する（`ApplyCudaLaunchBlockingConfig` と同じ枠）。
- **コスト**: 実測で `deterministic_algorithms=false` 比 **約 11〜13% 遅い**（Codex 実装後の計測）。SDPA backward 限定でなく scatter/index_add/reduction 等の決定版を全 op に強制する広域スイッチのためコストは aggregate。当初「attention 非ボトルネックゆえ無視可」と見積もったのは誤りだった。通常の構成比較は seed 違い複数 Run のブレ幅基準（bit 再現は不要）なので、速度が要る局面は `false` を選んでよい（再現が要る時＝デバッグ/正確な再走/回帰だけ `true`）。既定 `true` は決定論方針で据え置き。
- **将来**: attention が実時間を食う構成（長系列・大 head 等）になったら、cuDNN attention の決定経路や flash の決定 backward を「速い決定経路」として再検討する余地がある（`setSDPUse*` / `setSDPPriorityOrder`、版差は `ATen/Context.h` 確認）。
