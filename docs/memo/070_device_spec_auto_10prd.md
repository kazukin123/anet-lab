# PRD 070: device 指定の auto 対応とキー体系統一

> 確定。D1〜D10、complexity audit、受入基準を実装契約とする。
> 2026-09-23 実装前改訂: 現行の評価キーは `run.*`。旧キー専用検出は行わず、`config_data` は指定値を保つ。実効 device は各所有クラスの個別 JSON に記録し、`auto`→CPU は WARN しない。
> 起点: 2026-09-06。リリース ZIP (v0.3.0) を CUDA 非搭載機で起動したところ、ウィンドウ表示後に無言終了した。
> 直接の原因は `agent.device_type = 1` のまま CUDA デバイスを確保しに行ったこと。設定で CPU へ切り替えようと
> したが、device 指定が 2 系統 3 組のキーに分かれ、env 別 config が common を後勝ちで上書きするため取りこぼした。
> 関連: PRD 071（捕まらない fatal の可視化。本 PRD の fail-fast を利用者へ見せる側）、
> `apps/runner/config/common.txt`、`core/anet-core/src/tensor_util.cpp`、`core/anet-core/src/trainer.cpp`

## Context（背景・目的）

### 実測（2026-09-06）

| 環境 | 設定 | 結果 |
|---|---|---|
| CUDA 非搭載機（ZIP v0.3.0 / 最新ビルドの両方） | `agent.device_type = 1` | ウィンドウ表示後に無言終了。`stderr.log` に `cudaGetDeviceCount() returned cudaErrorNotSupported. Likely using older driver or on CPU machine` |
| CUDA 非搭載機 | `agent.device_type = 0` | 正常動作 |
| CUDA 搭載機 + `CUDA_VISIBLE_DEVICES=-1` | `agent.device_type = 1` | NN 構築時に main thread で `c10::Error`、エラーダイアログ表示（正常系） |

CPU 機での無言終了そのものは PRD 071 で扱う。本 PRD は **「そもそも CUDA を要求しない設定を書ける」** ことを扱う。

### 現状の device 指定

| キー | 型 | 既定 | 読み取り位置 |
|---|---|---|---|
| `agent.device_type` / `agent.device_index` | int / int | 1 / -1 | `agent.hpp:311`、`agent.cpp:82` の `MakeDevice()` |
| `env.device_type` / `env.device_index` | int / int | 0 / -1 | `env.hpp:341`、`env.cpp:938` の `MakeDevice()` |
| `run.eval_device_type` / `run.eval_device_index` | 文字列 / int | `"cpu"` / 0 | `trainer.cpp:732`、`GetEvalDevice()` |

問題は 4 つある。

1. **2 系統ある。** int の `device_type` と文字列の `eval_device_type` が同じ概念を別表現で書いている。
2. **auto が無い。** 実行環境に GPU があるかどうかを設定側から表現できない。
3. **後勝ちで上書きされる。** `common.txt:36` の `agent.device_type = 1` を 0 にしても、
   `LunarLander.txt:25` / `CartPole.txt:86` が 1 を再指定するので効かない。今回の「CPU に切り替えたのに動かない」の実体。
4. **可用性チェックが無い。** コードベースで `torch::cuda::is_available()` を見ているのは
   `random.cpp:153`（seed 設定）の 1 箇所だけで、device 解決側は設定値をそのまま `torch::Device(kCUDA, index)` にする。

加えて `common.txt` は `run.eval_device_type ?= cuda  # default: cpu` で、コメントと実値が食い違っている。

### 変換点は 2 箇所しかない

- `MakeDevice(int type, int index)`（`tensor_util.cpp:10`）— agent と env の唯一の変換点
- `GetEvalDevice()`（`trainer.cpp:746`）— eval の変換点

この 2 箇所を 1 つのパーサへ寄せれば、キー体系の統一と auto の導入が同時に済む。
旧キー専用の検出は、現行のクリーンブレーク方針に従って設けない。

## 比較履歴

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A: 文字列キー `device` へ統一** | `agent.device` / `env.device` / `run.eval_device` = `auto`\|`cpu`\|`cuda`\|`cuda:N`。パーサ 1 つ | 2 系統が 1 系統になる。index が値へ畳まれキー数が半減。auto を自然に置ける | config 全ファイルの書き換えが要る |
| B: `device_type` に `-1 = auto` を追加 | int のまま auto を足す | 差分最小。config はほぼそのまま | `device_index = -1` が既に「current device」の意味で使われており紛らわしい。eval 側の文字列系統との不揃いが残る |
| C: `device_type` が int と文字列を両受け | どちらの書き方も許す | 移行コストゼロ | 型が緩む。設定ダンプの見た目が揃わない。2 系統が永続化する |

案 A を採用する。今回の詰まりの遠因が「同じ概念が 2 系統 3 組に散っていること」なので、キーを減らす方向が本筋。

## 確定した D1〜D10

| # | 決定 | 契約 |
|---|---|---|
| D1 | キー名 | `agent.device` / `env.device` / `run.eval_device`。いずれも文字列 1 キー。`*_index` キーは持たない |
| D2 | 値の文法 | `auto` \| `cpu` \| `cuda` \| `cuda:<非負整数>`。前後空白は trim、大小文字は無視（`ToLower` で正規化）。それ以外の値は fail-fast |
| D3 | 既定値 | `agent.device = auto`、`run.eval_device = auto`、`env.device = cpu`。env だけ auto にしないのは、auto にすると CUDA 機で全 env が GPU へ移り、既存 Run と比較不能になるため |
| D4 | auto の解決 | `torch::cuda::is_available()` が true なら `cuda`（index 指定なし＝current device）、false なら `cpu`。判定条件はこの 1 つだけ |
| D5 | 明示 cuda × 利用不可 | fail-fast。エラーには **cuda を指定している全キーの一覧**と `auto` への変更方法を含める。1 キーずつ潰させない |
| D6 | 旧キー | リポジトリ管理下の現用設定・コード・テストを新キーへ移す。旧キー専用の検出・互換読みは設けず、現行設定契約外のキーとして扱う |
| D7 | index の表現 | `cuda:N` のみ。`cuda`（index 無し）は現行の `device_index = -1`（current device）と同義 |
| D8 | 実装位置 | `anet::ParseDevice(const std::string& spec) -> torch::Device` を `tensor_util` に置き、内部で auto を解決する。`MakeDevice(int, int)` は削除する |
| D9 | 実効値の記録 | 採用したクラスが Run ログ（`LOG::info`）と個別 JSON へ残す。`json/run.json.data.effective_eval_device`、`json/env.json.data.effective_device`、`json/agent.json.data.effective_device` を使う。`config_data` と `config/*.txt` は指定値を保持する。`auto`→CPU は正常な選択なので WARN しない |
| D10 | 検証タイミング | 値の文法と明示 cuda の可用性検査（D5）は `InitRL()` で行う。NN 構築より前に止める |

## Complexity audit

- Keep: パーサ 1 関数、`InitRL()` での検証 1 箇所、所有クラスの実効値ログと個別 JSON。
- Shrink: `DeviceSpec` のような中間型は作らず `std::string -> torch::Device` の 1 関数へ畳む。auto の判定条件は `is_available()` だけに限る。
- Defer: 複数 GPU の割り当てポリシー、device ごとの個別フォールバック。
- Cut: 旧キーの互換読み・専用検出、int 値の受理、CPU-only ビルド構成（別件）。

## 受入基準

1. `agent.device` 未指定で、CUDA 機では `cuda`、CPU 機では `cpu` へ解決され、どちらも Run ログに実効値が残る。
2. CPU 機で `agent.device = cuda` を明示すると NN 構築前に停止し、cuda を指定している全キーが列挙される。
3. 現用コード・同梱設定・現行ドキュメントから旧キーが消え、旧キー専用の検出・互換読みがない。
4. `agent.device = cuda:1` で index 1 が反映される。`cuda` は current device のまま。
5. `config_data` と `config/*.txt` は `auto` を含む指定値を保ち、`json/run.json` / `json/env.json` / `json/agent.json` の個別フィールドと Run ログに実効 device が残る。`auto`→CPU の WARN は出ない。
6. `env.device` 未指定の Run は従来どおり CPU env で、既存 Run と挙動が一致する。
7. 同梱 config（`common.txt` / 各 env config / `_workspace_template.txt`）から旧キーが消えている。

## 非目標

- 複数 GPU の負荷分散・自動選択。
- CPU-only ビルド構成（CUDA 版 LibTorch のリンクを外す話）。現在の ZIP は CUDA ランタイム DLL を同梱しており、
  同梱 DLL に `nvcuda.dll`（ドライバ）への静的依存が無いことは確認済みなので、ドライバ非搭載機でも DLL ロード自体は通る。
- 無言終了そのものの修正。PRD 071 で扱う。
- `torch::cuda::is_available()` 以外の条件（compute capability、メモリ量など）による判定。
