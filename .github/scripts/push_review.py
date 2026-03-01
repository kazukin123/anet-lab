import os
import time
import sys # 追加
from google import genai
from github import Github, Auth

# --- 設定エリア ---
TARGET_PATHS = ["core/anet-core", "core/envs", "apps"] 
EXCLUDE_KEYWORDS = [] 
# ----------------

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
auth = Auth.Token(os.environ["GITHUB_TOKEN"])
g = Github(auth=auth)
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])

event_name = os.environ.get("GITHUB_EVENT_NAME")
sha = os.environ.get("GITHUB_SHA")

review_targets = []

def is_target_file(filepath):
    if not filepath.endswith(('.cpp', '.hpp', '.h')): return False
    return any(filepath.startswith(p) for p in TARGET_PATHS)

# ファイル収集
if event_name == "push":
    commit = repo.get_commit(sha)
    for f in commit.files:
        if is_target_file(f.filename):
            review_targets.append({"path": f.filename, "content": f.patch})
    mode_text = "今回の差分レビュー"
else:
    mode_text = "プロジェクト全体の一括レビュー"
    for path in TARGET_PATHS:
        try:
            contents = repo.get_contents(path)
            while contents:
                item = contents.pop(0)
                if item.type == "dir":
                    contents.extend(repo.get_contents(item.path))
                elif is_target_file(item.path):
                    review_targets.append({"path": item.path, "content": item.decoded_content.decode()})
        except: pass

if not review_targets:
    print("対象コードなし。")
    exit(0)

full_results = ""
error_occurred = False

for target in review_targets:
    print(f"Reviewing: {target['path']}...")
    
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
ファイル: {target['path']}
内容:
{target['content']}
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        full_results += f"#### 📄 {target['path']}\n{response.text}\n\n"
        
        # 次のリクエストまで5秒待機
        time.sleep(5) 

    except Exception as e:
        # エラーが起きたら情報を残して即座に終了する
        error_msg = f"### ❌ APIエラーにより処理を中断しました\n原因: {e}\n"
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(error_msg)
        print(error_msg)
        
        # 重要：sys.exit(1) でスクリプトとGitHub Actions全体を異常終了させる
        sys.exit(1) 

# 正常に全件終わった場合のみここに来る
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
    f.write(f"### 🤖 Gemini {mode_text} (完了)\n\n{full_results}")

if event_name == "push" and full_results:
    repo.get_commit(sha).create_comment(f"### 🤖 Gemini Push Review\n\n{full_results}")