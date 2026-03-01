import os
import google.generativeai as genai
from github import Github

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.0-flash')
g = Github(os.environ["GITHUB_TOKEN"])
repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])

# 最新のコミットを取得
sha = os.environ.get("GITHUB_SHA")
commit = repo.get_commit(sha)

# 差分（Diff）の取得
diff_text = ""
for file in commit.files:
    if file.filename.endswith(('.cpp', '.hpp', '.h')):
        diff_text += f"\n--- File: {file.filename} ---\n{file.patch}\n"

if not diff_text:
    print("No C++ changes detected.")
    exit(0)

# プロンプト
prompt = f"""
## 指示:
あなたはシニアC++エンジニア、および強化学習（RL）の実装エキスパートです。
以下のC++コードの差分（またはファイル全体）をレビューし、日本語でフィードバックしてください。

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
以下にレビュー対象のコードを提示します：

{diff_text}
"""

response = model.generate_content(prompt)

# 結果をコミットコメントとして投稿
commit.create_comment(f"### 🤖 Gemini Push Review\n\n{response.text}")

# 同時にGitHubのAction Summaryにも表示（後で読み返しやすい）
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
    f.write(f"### 🤖 Gemini Review for {sha[:7]}\n\n{response.text}")
