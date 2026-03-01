import os
import google.generativeai as genai
from github import Github, Auth

# --- 設定エリア ---
# レビュー対象にするディレクトリ（リポジトリルートからの相対パス）
TARGET_PATHS = ["core/anet-core", "core/envs", "apps"] 

# 除外したいディレクトリ名やファイル名の部分一致（Box2D、外部ライブラリなど）
EXCLUDE_KEYWORDS = [ ]
# ----------------

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# クォータに余裕がある場合は 'gemini-1.5-pro'、速度重視なら 'gemini-1.5-flash'
model = genai.GenerativeModel('gemini-1.5-flash') 

auth = Auth.Token(os.environ["GITHUB_TOKEN"])
g = Github(auth=auth)
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])

event_name = os.environ.get("GITHUB_EVENT_NAME")
sha = os.environ.get("GITHUB_SHA")

code_content = ""

def is_target_file(filepath):
    # 1. 拡張子チェック
    if not filepath.endswith(('.cpp', '.hpp', '.h')):
        return False
    # 2. 指定したターゲットパスに含まれているか
    in_target = any(filepath.startswith(p) for p in TARGET_PATHS)
    # 3. 除外キーワードが含まれていないか
    is_excluded = any(k in filepath for k in EXCLUDE_KEYWORDS)
    return in_target and not is_excluded

if event_name == "push":
    commit = repo.get_commit(sha)
    for file in commit.files:
        if is_target_file(file.filename):
            code_content += f"\n--- File: {file.filename} ---\n{file.patch}\n"
    mode_text = "今回の差分レビュー"
else:
    # 手動実行：ターゲットパス配下を再帰的に取得
    mode_text = "プロジェクト全体の特定パス一括レビュー"
    for path in TARGET_PATHS:
        try:
            items = repo.get_contents(path)
            while items:
                item = items.pop(0)
                if item.type == "dir":
                    if not any(k in item.path for k in EXCLUDE_KEYWORDS):
                        items.extend(repo.get_contents(item.path))
                elif is_target_file(item.path):
                    if item.size < 100000: # 100KB以上の巨大ファイルは避ける
                        decoded = item.decoded_content.decode()
                        code_content += f"\n--- File: {item.path} ---\n{decoded}\n"
        except Exception as e:
            print(f"Path not found or error: {path} ({e})")

if not code_content:
    print("レビュー対象のコードが見つかりませんでした。")
    exit(0)

# あなたのプロンプト（指示事項）
prompt_text = f"""
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
以下のコードについて【{mode_text}】を行ってください。
{code_content}
"""

try:
    response = model.generate_content(prompt_text)
    
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
        f.write(f"### 🤖 Gemini {mode_text}\n\n{response.text}")

    if event_name == "push":
        repo.get_commit(sha).create_comment(f"### 🤖 Gemini Push Review\n\n{response.text}")
except Exception as e:
    error_msg = f"API制限（Quota Exceeded）またはエラーが発生しました。\n{e}"
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
        f.write(f"### ❌ レビュー失敗\n{error_msg}")
    print(error_msg)
