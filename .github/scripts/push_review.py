import os
import time
import sys
from google import genai
from github import Github, Auth

# --- 設定エリア ---
TARGET_PATHS = ["core/anet-core", "core/envs", "apps"]
EXCLUDE_KEYWORDS = []
# TPM制限(250k)を考慮した1リクエストあたりの文字数制限（安全圏）
CHUNK_CHAR_LIMIT = 150000
# ----------------

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
auth = Auth.Token(os.environ["GITHUB_TOKEN"])
g = Github(auth=auth)
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])

event_name = os.environ.get("GITHUB_EVENT_NAME")
sha = os.environ.get("GITHUB_SHA")

def is_target_file(filepath):
    if not filepath.endswith(('.cpp', '.hpp', '.h')): return False
    return any(filepath.startswith(p) for p in TARGET_PATHS)

# 1. ファイルの収集
all_files = []
if event_name == "push":
    commit = repo.get_commit(sha)
    for f in commit.files:
        if is_target_file(f.filename):
            all_files.append({"path": f.filename, "content": f.patch})
else:
    for path in TARGET_PATHS:
        try:
            items = repo.get_contents(path)
            while items:
                item = items.pop(0)
                if item.type == "dir":
                    items.extend(repo.get_contents(item.path))
                elif is_target_file(item.path):
                    all_files.append({"path": item.path, "content": item.decoded_content.decode()})
        except: pass

if not all_files:
    print("レビュー対象のファイルが見つかりませんでした。")
    exit(0)

# 2. 【追加機能】ファイルリストを先に Summary へ出力
file_list_md = "\n".join([f"- `{f['path']}` ({len(f['content'])} chars)" for f in all_files])
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
    f.write(f"### 🔍 レビュー対象ファイル一覧\n{file_list_md}\n\n---\n")

print(f"Total files found: {len(all_files)}")

# 3. CHUNK作成（TPM制限対策）
chunks = []
current_chunk = []
current_chars = 0

for f in all_files:
    f_size = len(f['content'])
    if current_chars + f_size > CHUNK_CHAR_LIMIT and current_chunk:
        chunks.append(current_chunk)
        current_chunk = []
        current_chars = 0
    current_chunk.append(f)
    current_chars += f_size
if current_chunk:
    chunks.append(current_chunk)

# 4. レビュー実行（RPD制限対策）
full_results = ""
for i, chunk in enumerate(chunks):
    print(f"Reviewing chunk {i+1}/{len(chunks)}...")
    
    combined_code = ""
    for f in chunk:
        combined_code += f"\n--- File: {f['path']} ---\n{f['content']}\n"

    prompt = f"""
あなたはシニアC++エンジニア、および強化学習（RL）の実装エキスパートです。

## プロジェクトの前提
- 言語・ライブラリ: C++20, libtorch (PyTorch C++ API), wxWidgets
- フレームワーク名: anet (独自RLフレームワーク)
- 優先事項: 学習の進行を妨げる論理的誤りの検出を最優先とする。

## 指示事項
回答は以下の4つのセクションに分け、箇条書きで記述してください。

### 1. 【最優先】学習進行・Tensor関連の不具合
以下を含め、強化学習の収束やTensorの挙動に関する致命的な問題をチェックしてください。
- **アルゴリズム** 強化学習のアルゴリズム実装として矛盾や異常、非一般的実行がないか
- **Clone漏れ**: Tensorが参照渡しや浅いコピー（shallow copy）になっており、Replay Buffer内や過去の計算結果が意図せず上書きされていないか。
- **勾配管理（detach）**: Tensor処理に関連して `detach()`や`clone()`等の必要な処理が適切に行われ、不要な計算グラフが保持されたり意図しない勾配更新が起きていないか。
- **Shape/Type誤り**: `view`や`reshape`時のサイズ不整合。特にBatchサイズとFeatureサイズの取り違えがないか。
- **デバイス整合性**: CPUとGPU(CUDA)のTensorが混在して演算エラーにならないか。
- **リソース管理**: `torch::NoGradGuard` の適用範囲が適切か。

### 2. その他の不具合
- **C++20/標準仕様**: メモリリーク、スレッド安全性（特に関数/変数への並行アクセス）、境界外アクセス。
- **GUI関連（wxWidgets）**: 学習スレッドからUIスレッドへの直接操作など、スレッドセーフでない描画処理がないか。

### 3. 改善事項（リファクタリング・一貫性）
- **モダンC++**: 基本的にはC++17を前提としつつ、指示付き初期化子は利用推奨。より安全で可読性の高い記述への提案。
- **一貫性**: 命名規則や設計パターンが既存のコード（anet）の流儀に沿っているか。

### 4. 性能改善（影響が大きい場合のみ）
- 頻繁に呼ばれるループ内での不要なTensorコピー、不適切なメモリアロケーション。
- CUDAカーネルを効率的に動かすためのデータ配置の懸念。
---
以下のコードをレビューしてください。
---
{combined_code}
"""

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        full_results += f"### 📦 チャンク {i+1} の指摘\n{response.text}\n\n"
        
        # TPMリセットのため、チャンク間では60秒待機
        if i < len(chunks) - 1:
            print("TPMリセット待ち(60s)...")
            time.sleep(60)

    except Exception as e:
        # エラー時は即座に停止して、無駄なリクエスト（課金リスク回避）を防ぐ
        error_msg = f"\n### ❌ エラー中断\n{e}"
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(error_msg)
        sys.exit(1)

# 5. 最終結果を出力
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
    f.write(f"### 🤖 Gemini レビュー結果\n\n{full_results}")
