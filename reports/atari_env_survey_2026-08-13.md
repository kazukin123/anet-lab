# Survey: Atari RL 環境 — ALE を中心としたゲームバリエーション・ライブラリ・ベンチマーク動向

Date: 2026-08-13
Scope: ALE (Arcade Learning Environment) を中心とした Atari 強化学習環境の調査。ゲームバリエーション(flavor/モード/難易度)、C++ からの利用方法、Gymnasium 連携と Python エコシステム、ベクトル化・高速実行系、ベンチマークプロトコル(Atari 100k / 200M / Atari-5 / CALE)、代替・派生環境(MinAtar / JAX 系 / OCAtari / stable-retro 等)を対象とする。調査は Web 一次ソース(公式ドキュメント・GitHub・PyPI JSON API・arXiv・Semantic Scholar API)の取得に基づく。

## 目次

1. [ALE 本体 — プロジェクト概要とゲームバリエーション](#1-ale-本体--プロジェクト概要とゲームバリエーション)
   - プロジェクト概要と保守状況 / ゲーム数の 3 層構造 / flavor(モード×難易度) / v0・v4・v5 の違い / マルチエージェント / 原典論文
2. [C++ インターフェース](#2-c-インターフェース)
   - ALEInterface API / ビルド(CMake + vcpkg) / C++ から見た ROM の扱い / ベクトル化実装の C++ 実体
3. [Gymnasium 連携と Python エコシステム](#3-gymnasium-連携と-python-エコシステム)
   - 登録イディオム / 標準前処理 / ale-py パッケージと Windows 対応 / RL ライブラリのベースライン / PettingZoo / Shimmy
4. [ベクトル化・高速実行系](#4-ベクトル化高速実行系)
   - ALE 内蔵 AtariVectorEnv / XLA・GPU 対応 / EnvPool
5. [ベンチマークプロトコルと研究動向](#5-ベンチマークプロトコルと研究動向)
   - 古典 57 ゲーム 200M プロトコル / Machado et al. 2018 勧告 / Atari 100k / Atari-5 / rliable と IQM / CALE(連続行動)
6. [代替・派生環境](#6-代替派生環境)
   - MinAtar / JAX ネイティブ系 / OCAtari・HackAtari・AtariARI / Gym Retro・stable-retro / Atari 以遠の代替
7. [批判・懸念](#7-批判懸念)
8. [総合評価](#8-総合評価)
9. [調査の限界](#9-調査の限界)
10. [ソースリスト](#10-ソースリスト)

---

## 1. ALE 本体 — プロジェクト概要とゲームバリエーション

### 1.1 プロジェクト概要と保守状況

ALE は Atari 2600 エミュレータ Stella の上に構築された RL 研究用フレームワークで、エミュレーション詳細をエージェント設計から分離することを設計思想とする。

```
It is built on top of the Atari 2600 emulator Stella and separates the details of emulation from agent design.

ALE は Atari 2600 エミュレータ Stella の上に構築されており、エミュレーションの詳細をエージェント設計から分離している。
```

[ale-github (2026/08 取得), README]

管理主体は Farama Foundation(Gymnasium / PettingZoo と同一団体)。GitHub API の実測(2026-08-13)で Star 2,441、ライセンス GPL-2.0、最終 push 2026-08-10 と、現在も活発に保守されている [ale-github (2026/08 取得)]。最新リリースは v0.12.0(GitHub Releases API では公開日 2026-05-29。§9 の日付注記参照)で、リリース間隔は年 2〜4 回程度 [ale-releases (2026/08 取得)]。

ALE がベンダリングしている Stella は 2007 年頃のスナップショットのフォークであり(`src/ale/emucore/` に取り込み、`namespace ale::stella` で独自保守)、正確な Stella バージョン番号はソース・ドキュメントいずれにも明記がない [ale-emucore (2026/08 取得), OSystem.hxx 著作権表記 "1995-2007"]。

### 1.2 ゲーム数の 3 層構造

「ゲーム数」は数え方が 3 層あり、レポートや比較の際は区別が必要である。

| 層 | 数 | 根拠 |
|---|---|---|
| エミュレータが対応するゲーム実装 | **104** | `src/ale/games/supported/` の .cpp ファイル実査 [ale-github (2026/08 取得)] |
| Gymnasium 登録環境(variant 込み) | **210**(v0.11 以降) | 公式ドキュメント [ale-docs-env (2026/08 取得)] |
| flavor(モード×難易度の組合せ) | ゲームごとに展開 | 同上 |

v0.11 で旧名前空間(`*NoFrameskip-v4` 等)が整理され、登録環境数は大幅に削減された。

```
In v0.11, the number of registered Atari environments was significantly reduced from 960 to 210.

v0.11 で、登録されている Atari 環境の数は 960 から 210 へと大幅に削減された。
```

[ale-docs-env (2026/08 取得)]

### 1.3 flavor(ゲームモード × 難易度)

ALE はゲームごとに複数のモードと難易度を公開しており、その組合せを flavor と呼ぶ。これは Machado et al. 2018 で導入された概念である。

```
We follow the convention ... and refer to the combination of difficulty level and game mode as a flavor of a game.

我々は慣例に従い、難易度とゲームモードの組合せをゲームの flavor と呼ぶ。
```

[ale-docs-env (2026/08 取得)]

公式 flavor 表からの例 [ale-docs-env (2026/08 取得)]:

| ゲーム | modes | difficulties | flavor 数 |
|---|---|---|---|
| Adventure | 0, 1, 2 | 0–3 | 12 |
| Asteroids | 0–31, 128 | 0, 3 | 66 |
| Breakout | 12 種 | 0, 1 | 24 |
| Space Invaders | 0–15 | 0, 1 | 32 |

### 1.4 環境バージョン v0 / v4 / v5 の違い

Gymnasium 登録環境のバージョンサフィックスは、frameskip と sticky actions(行動の確率的リピート)のデフォルトが異なる [ale-docs-env (2026/08 取得); gymnasium-v029-atari (2026/08 取得)]。

| バージョン | repeat_action_probability (sticky actions) | frameskip |
|---|---|---|
| v0 | 0.25 | (2,5) 確率的 |
| v4 | 0.0 | (2,5)(`Deterministic-v4` 派生は決定的) |
| v5 | **0.25** | **4 固定** |

v5 は Machado et al. 2018 の推奨プロトコル(sticky actions 有効化+確率的 frameskip 廃止)に準拠したものである [ale-docs-env (2026/08 取得)]。公式ドキュメントは sticky actions の性能影響について注意を明記している。

```
Importantly, `repeat_action_probability=0.25` can negatively impact the performance of agents.

重要な点として、`repeat_action_probability=0.25` はエージェントの性能に悪影響を与えうる。
```

[ale-docs-env (2026/08 取得)]

### 1.5 マルチエージェント環境

ALE は 23 タイトルの対戦/協力ゲーム(Boxing, Pong, Warlords, Combat, Entombed 競争/協力版など)を PettingZoo API 経由で提供する。導入論文は Terry & Black 2020 "Multiplayer Support for the Arcade Learning Environment" [ale-docs-ma (2026/08 取得)]。

```
Most games have two players, with the exception of Warlords and a couple of Pong variations which have four players.

ほとんどのゲームは 2 人用で、例外として Warlords と一部の Pong バリエーションは 4 人用である。
```

[pettingzoo-atari (2026/08 取得)]

### 1.6 原典論文

| 論文 | venue | 被引用数 (Semantic Scholar, 2026-08-13) |
|---|---|---|
| Bellemare, Naddaf, Veness, Bowling. "The Arcade Learning Environment: An Evaluation Platform for General Agents" | JAIR Vol.47, 2013 | 3,369 |
| Machado, Bellemare, Talvitie, Veness, Hausknecht, Bowling. "Revisiting the Arcade Learning Environment" | JAIR Vol.61, 2018 | 627(レコード分裂により実勢はより多い可能性) |
| Farebrother, Castro. "CALE: Continuous Arcade Learning Environment" | NeurIPS 2024 Datasets and Benchmarks | 5 |

README の引用指示では、sticky actions / flavor を使う場合は Machado et al. 2018 を、連続行動を使う場合は CALE 論文を併引用するよう明記されている [ale-github (2026/08 取得)]。

---

## 2. C++ インターフェース

### 2.1 ALEInterface API

中心クラスは `ale::ALEInterface`(ヘッダ: `src/ale/ale_interface.hpp`、インストール後は `include/ale/` 配下)。2026-08-13 時点の master ブランチのヘッダから確認した主要 API [ale-source (2026/08 取得)]:

- `void loadROM(fs::path rom_file)` — ROM 読み込み
- `reward_t act(Action action, float paddle_strength = 1.0)` — 1 ステップ実行(`paddle_strength` は v0.10 の連続行動対応で追加)
- `bool game_over(bool with_truncation = true) const` / `void reset_game()` / `int lives()`
- `ActionVect getLegalActionSet()` / `getMinimalActionSet()` — 全 18 行動 / 有効行動のみ
- `const ALEScreen& getScreen()` / `getScreenGrayscale(...)` / `getScreenRGB(...)` — 画面観測
- `const ALERAM& getRAM()` / `setRAM(...)` — RAM 観測・操作
- `ModeVect getAvailableModes()` / `setMode(...)` / `getAvailableDifficulties()` / `setDifficulty(...)` — flavor 制御
- `ALEState cloneState(bool include_rng = false)` / `restoreState(...)` / `cloneSystemState()` / `restoreSystemState()` — 状態クローン/復元(planning・MCTS 系に有用)
- `setBool` / `setInt` / `setFloat` — 設定(例: `ale.setInt("random_seed", 123)`)

公式ドキュメントのランダムエージェント最小例(原文ママ):

```cpp
#include <iostream>
#include <ale_interface.hpp>

int main(int argc, char** argv) {
    ale::ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.loadROM(argv[1]);
    ale::ActionVect legal_actions = ale.getLegalActionSet();
    float totalReward = 0.0;
    while (!ale.game_over()) {
        Action a = legal_actions[std::rand() % legal_actions.size()];
        float reward = ale.act(a);
        totalReward += reward;
    }
    return 0;
}
```

[ale-docs-cpp (2026/08 取得)]

### 2.2 ビルド(CMake + vcpkg manifest)

要件は C++17 コンパイラ + CMake 3.14 以上(トップ CMakeLists の `cmake_minimum_required` を直接確認)。依存管理は vcpkg の manifest mode をネイティブサポートし、README は CMake を "a first class citizen" と位置づけている [ale-github (2026/08 取得)]。

`vcpkg.json`(master 実査)の内容 [ale-source (2026/08 取得)]:

- 必須依存は **zlib のみ**
- feature `sdl`(sdl2 — 表示/音声用)、feature `vector`(opencv4 — ベクトル化環境の前処理用)

CMake オプション(トップ CMakeLists 実査): `BUILD_CPP_LIB`(既定 ON)/ `BUILD_PYTHON_LIB`(既定 ON)/ `SDL_SUPPORT`(既定 OFF)/ `BUILD_VECTOR_LIB`(既定 OFF)/ `BUILD_VECTOR_XLA_LIB` / `BUILD_WASM_LIB`。`VCPKG_ROOT` 環境変数から toolchain を自動発見するロジックがある [ale-source (2026/08 取得)]。

インストールされる成果物: ヘッダ一式(`include/ale/`)、CMake config(`lib/cmake/ale/`)、pkg-config ファイル。利用側は以下で完結する [ale-docs-cpp (2026/08 取得)]:

```cmake
find_package(ale REQUIRED)
target_link_libraries(YourTarget ale::ale-lib)
```

注意点として、**microsoft/vcpkg の ports にも ConanCenter にも ALE は登録されていない**(GitHub API 404 で確認)。つまり C++ 利用の公式ルートは「ソースからの CMake ビルド(依存解決に vcpkg manifest を利用)」であり、`vcpkg.json` は ALE 自身の依存を取るためのもので、ALE 本体が vcpkg port として配布されているわけではない [ale-source (2026/08 取得)]。

### 2.3 C++ から見た ROM の扱い

pip 版 ale-py は v0.9.0 以降 ROM を同梱するが(§3.3)、これは Python パッケージ側の仕組みであり、**C++ から使う場合は ROM の .bin ファイルを自己調達して `loadROM()` にパス渡しする必要がある**(公式 C++ ドキュメントの例も `ale.loadROM("asterix.bin")` とファイルパス渡し)[ale-docs-cpp (2026/08 取得)]。

### 2.4 ベクトル化実装の C++ 実体

C++ ベクトル化実装は `src/ale/vector/`(`env_vectorizer.{hpp,cpp}`, `preprocessed_env.{hpp,cpp}`, `action_queue.hpp`, `result_staging.hpp`)にあり、`BUILD_VECTOR_LIB=ON` + opencv4 でビルドできる。Python バインディング経由でなく C++ から直接使える実体が存在する [ale-source (2026/08 取得)]。

---

## 3. Gymnasium 連携と Python エコシステム

### 3.1 登録イディオムと環境生成

現行の環境登録・生成イディオムは以下の通り [ale-docs-gym (2026/08 取得)]:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5")
```

連続行動は `gym.make("ALE/Breakout-v5", continuous=True)` で有効化でき、行動空間は極座標(radius, theta)+ fire の 3 次元 Box となる [ale-docs-gym (2026/08 取得)]。

### 3.2 標準前処理(AtariPreprocessing + FrameStackObservation)

事実上の標準 DQN 前処理は Gymnasium の `AtariPreprocessing` ラッパーに実装されており、ドキュメントは Machado et al. 2018 を明示引用している [gymnasium-wrappers (2026/08 取得)]。実装内容:

1. NoopReset(reset 時に最大 30 no-op)
2. frame skipping(既定 4)
3. 直近 2 フレームの max-pooling(フリッカー対策)
4. terminal_on_life_loss(既定無効)
5. 210×160 → 84×84 リサイズ
6. グレースケール化(既定有効)

フレームスタックは `FrameStackObservation` を併用する(`stack_size` 指定、rolling 方式)[gymnasium-wrappers-obs (2026/08 取得)]。

### 3.3 ale-py パッケージと Windows 対応

PyPI JSON API の実測(2026-08-13)[pypi-ale-py (2026/08 取得)]:

- 最新版 0.12.0、`requires_python >= 3.10`、cp310〜cp314 wheel
- **Windows wheel あり**(`win_amd64`)。他に manylinux x86_64/aarch64、macOS x86_64/arm64

ROM は v0.9.0 以降 pip パッケージに同梱され、AutoROM は不要になった。

```
Atari ROMs are packaged within the PyPI installation such that users no longer require pip install "gym[accept-rom-license]" (AutoROM)

Atari の ROM は PyPI インストールに同梱されており、ユーザーはもはや pip install "gym[accept-rom-license]"(AutoROM)を必要としない。
```

[ale-release-v090 (2024/05)]

AutoROM 自体も 2025 年 8 月に正式に非推奨化された [autorom (2025/08), README]。ただし conda-forge 版 ale-py は ROM の著作権状態を確認できないとして ROM を同梱せず、合法的に入手した ROM を `ALE_ROMS_DIR` で指すよう求めており、法的位置づけには揺らぎが残る(§7.4)[conda-forge-ale (2026/08 取得)]。

### 3.4 主要バージョン履歴

日付は GitHub Releases API の `published_at` 実測(§9 の日付注記参照)[ale-releases (2026/08 取得)]:

| バージョン | 公開日 | 主な内容 |
|---|---|---|
| v0.9.0 | 2024-05-20 | Gymnasium 単一バックエンド化、ROM の PyPI 同梱 |
| v0.10.0 | 2024-09-24 | 連続行動(CALE)統合、ale.farama.org 開設 |
| v0.11.0 | 2025-04-26 | C++ 製 AtariVectorEnv 導入、登録環境 960→210 整理 |
| v0.11.1 | 2025-05-29 | vector env 修正、XLA 実験サポート |
| v0.11.2 | 2025-07-12 | Windows/Linux の音声・描画修正 |
| v0.12.0 | 2026-05-29 | scikit-build-core + nanobind 移行、WASM/NPM 配布、XLA GPU 対応、観測コピー削減(3 回→1 回) |

### 3.5 Atari ベースラインを提供する RL ライブラリ

いずれも公式ドキュメント確認済み(tier: 公式):

| ライブラリ | Atari 対応 | URL |
|---|---|---|
| CleanRL | `dqn_atari.py` / `dqn_atari_jax.py`(JAX 版は PyTorch 版比 25–50% 高速と記載)。10M step ベンチ表を公開 | https://docs.cleanrl.dev/rl-algorithms/dqn/ |
| Stable-Baselines3 | `make_atari_env` ヘルパー(前処理+並列化を一括)+ `VecFrameStack` | https://stable-baselines3.readthedocs.io/en/master/guide/examples.html |
| Dopamine (Google) | DQN / C51 / Rainbow / IQN 等の JAX 実装(TF 実装は legacy) | https://github.com/google/dopamine |
| Sample Factory | 高速 PPO 特化。Atari 統合と EnvPool 統合を両方ドキュメント化 | https://www.samplefactory.dev/ |

### 3.6 PettingZoo(マルチエージェント)と Windows 制約

PettingZoo の Atari 環境(23 タイトル)は `multi-agent-ale-py` を基盤とするが、**multi-agent-ale-py 0.1.12 の wheel は macOS arm64 / Linux x86_64 のみで win_amd64 wheel が存在しない**(PyPI JSON 実測)。Windows では sdist の自前ビルドが必要になる [pypi-ma-ale (2026/08 取得)]。

### 3.7 Shimmy(互換レイヤ)

Shimmy は旧 Gym(v21–26)・DM 系 API を Gymnasium API へ橋渡しする変換ツールで、ALE も対応対象に含む [shimmy (2026/08 取得)]。ale-py が Gymnasium ネイティブ統合を持つ現在、Atari 用途で Shimmy が必須になる場面は限定的である(この評価は本レポートの整理であり、ソース記述は対応リストの範囲)。

---

## 4. ベクトル化・高速実行系

### 4.1 ALE 内蔵 AtariVectorEnv(v0.11 以降)

v0.11.0 で、EnvPool にインスパイアされた C++ 非同期ベクトル化環境が ALE 本体に導入された。

```
Inspired by the EnvPool implementation, we've implemented an asynchronous vectorisation environment in C++, in particular, the standard Atari preprocessing including frame skipping, frame stacking, observation resizing, etc.

EnvPool の実装にインスパイアされ、我々は非同期ベクトル化環境を C++ で実装した。具体的には frame skipping、frame stacking、観測リサイズなどの標準 Atari 前処理を含む。
```

[ale-release-v0110 (2025/04)]

利用法は `gymnasium.make_vec("ALE/{game}-v5", num_envs)` または `ale_py.AtariVectorEnv("{rom}", num_envs)`。前処理(frame skip / グレースケール / 84×84 リサイズ / frame stack / NoOp reset / fire reset / episodic life)を内蔵し、出力は `(num_envs, stack_size, height, width)`。same-step / next-step の両 autoreset、スレッドアフィニティ設定、`send`/`recv` による非同期分割ステップに対応する [ale-docs-vector (2026/08 取得)]。

XLA サポートは v0.11.1 で実験導入され、v0.12.0 で GPU 対応が加わった [ale-releases (2026/08 取得)]。

### 4.2 EnvPool

EnvPool(sail-sg / Sea AI Lab)は C++ スレッドプール+ロックフリーキューによるバッチ環境実行エンジンで、Atari を筆頭に約 15 環境ファミリーに対応する。性能クレームは README より:

```
~1M raw FPS with Atari games; ~3M raw FPS with MuJoCo simulator on DGX-A100; ~20x throughput of Python subprocess-based vector env

Atari で raw 100 万 FPS、DGX-A100 上の MuJoCo で raw 300 万 FPS、Python subprocess ベースのベクトル環境比で約 20 倍のスループット。
```

[envpool-github (2026/08 取得), README]

論文は NeurIPS 2022 Datasets and Benchmarks Track、被引用数 84(Semantic Scholar, 2026-08-13)[envpool-paper (2022/06)]。

**保守状況の重要な変化**: 0.8.4(2023-10-30)を最後に約 2.5 年停滞していたが、2026 年 3 月からリリースが再開された(0.9.0 = 2026-03-23 → 1.2.5 = 2026-05-20、いずれも PyPI/GitHub 実測)。1.x 系では gym 依存の除去、macOS 対応、そして **win_amd64 wheel(cp311–314)の新規提供**が行われ、「EnvPool は Linux 専用」という通説は 1.x 系では過去のものになった [envpool-releases (2026/08 取得); pypi-envpool (2026/08 取得)]。この情報は PyPI / GitHub の一次ソースのみに基づく(§9 参照)。

---

## 5. ベンチマークプロトコルと研究動向

### 5.1 古典 57 ゲーム 200M プロトコルとマイルストーン

古典プロトコルは 57 ゲーム × 200M frames 訓練、人間正規化スコア HNS = (agent − random)/(human − random) の median を集計する形式が長年の基準だった(近年は IQM へ移行、§5.5)。主要マイルストーン(被引用数はすべて Semantic Scholar, 2026-08-13):

| 論文 | venue | 被引用数 | 要点 |
|---|---|---|---|
| Rainbow (Hessel et al.) | AAAI 2018 | 2,686 | DQN 系 6 拡張の統合 |
| R2D2 (Kapturowski et al.) | ICLR 2019 | 577 | 分散+再帰リプレイ |
| MuZero (Schrittwieser et al.) | Nature 2020 | 2,656 | 57 ゲームで当時 SOTA |
| Agent57 (Badia et al.) | ICML 2020 | 597 | 全 57 ゲームで人間超えを初達成 |

```
We propose Agent57, the first deep RL agent that outperforms the standard human benchmark on all 57 Atari games.

我々は Agent57 を提案する。57 の Atari ゲームすべてで標準的な人間ベンチマークを上回った初の深層 RL エージェントである。
```

[agent57 (2020/03), abstract]

### 5.2 Machado et al. 2018 勧告(v5 デフォルトの源流)

Machado et al. 2018(JAIR、被引用数 627)は評価方法論の乱立への懸念から、sticky actions によるstochasticity 導入、flavor(mode/difficulty)対応を含む新版 ALE と評価のベストプラクティスを提示した。

```
a new version of the ALE that supports multiple game modes and provides a form of stochasticity we call sticky actions

複数のゲームモードをサポートし、我々が sticky actions と呼ぶ形のstochasticityを提供する新しいバージョンの ALE
```

[machado2018 (2017/09), abstract]

この勧告が現在の v5 デフォルト(`repeat_action_probability=0.25`, `frameskip=4`)に反映されている [ale-docs-env (2026/08 取得)]。

### 5.3 Atari 100k(サンプル効率ベンチマーク)

起源は SimPLe 論文(Kaiser et al., ICLR 2020、被引用数 1,004)。100k interactions ≒ リアルタイム 2 時間のプレイに相当する低データ域を定義した。

```
Our experiments evaluate SimPLe on a range of Atari games in low data regime of 100k interactions between the agent and the environment, which corresponds to two hours of real-time play.

我々の実験では、エージェントと環境の相互作用 10 万回という低データ域で SimPLe を評価する。これはリアルタイムのプレイ 2 時間に相当する。
```

[simple (2019/03), abstract]

26 ゲームの選定基準は「既存の最先端モデルフリー深層 RL で解けること」(同論文本文)。100k agent steps = 400k frames(frameskip 4)。標準プロトコルでは sticky actions **なし**が慣行である(BBF が sticky actions 付きへの移行を提案していることの裏返し)[bbf (2023/05)]。

主要記録:

| 手法 | venue | 被引用数 | 結果 |
|---|---|---|---|
| EfficientZero (Ye et al.) | NeurIPS 2021 | 339 | mean 194.3% / median 109.0% human |
| BBF (Schwarzer et al.) | ICML 2023 | 170 | IQM human-normalized > 1(superhuman)、単一 GPU 約 6 時間 |
| EfficientZero V2 (Wang et al.) | ICML 2024 Spotlight | 42 | normalized mean 2.428 / median 1.286 |
| DreamerV3 (Hafner et al.) | Nature 2025 | 1,311(arXiv 版) | 単一構成で 150+ タスク横断(Atari は対象の一部) |

BBF はベンチマーク自体のゴールポスト更新を提案しており、26 ゲームから ALE 全 55 ゲームへの拡大・sticky actions 導入・「Rainbow の最終性能を 2 時間のプレイで再現できるか」という新目標を挙げている [bbf (2023/05), 本文]。

### 5.4 Atari-5(廉価サブセット)

Aitchison, Sweetser, Hutter(ICML 2023、被引用数 52)は、5 ゲーム(Battle Zone, Double Dunk, Name This Game, Phoenix, Q*bert)の回帰で 57 ゲーム median を推定するサブセットを提案した。

```
a subset of five ALE games, called Atari-5, which produces 57-game median score estimates within 10% of their true values.

Atari-5 と呼ばれる 5 つの ALE ゲームのサブセットで、57 ゲームの median スコアを真値の 10% 以内で推定できる。
```

[atari5 (2022/10), abstract]

採用状況は「ablation 用の廉価サブセット」が中心で(Beyond The Rainbow が ablation に使用等)、標準の完全置換には至っていない [atari5 (2022/10); btr (2024/11)]。

### 5.5 rliable と IQM(統計的評価の現行標準)

Agarwal et al. "Deep RL at the Edge of the Statistical Precipice"(NeurIPS 2021 Outstanding Paper、被引用数 977)は、少数 run の点推定(mean/median)による比較の統計的問題を指摘し、IQM(interquartile mean)+ stratified bootstrap 信頼区間 + performance profiles を提案した [rliable (2021/08)]。公式実装は google-research/rliable。BBF・EfficientZero V2・BTR など 2023 年以降の Atari 論文は軒並み IQM ± 信頼区間で報告しており、現行標準といえる [bbf (2023/05); btr (2024/11)]。

### 5.6 CALE(連続行動版 ALE)

Farebrother & Castro(NeurIPS 2024 D&B、被引用数 5)は同じ Stella エミュレータ上で連続行動をサポートする CALE を提案した。

```
The CALE uses the same underlying emulator of the Atari 2600 gaming system (Stella), but adds support for continuous actions.

CALE は Atari 2600 ゲームシステムの同じ基盤エミュレータ(Stella)を使用するが、連続行動のサポートを追加する。
```

[cale (2024/10), abstract]

行動空間は 3 次元(極座標 (r, θ) + fire)。初期結果では SAC が DQN / Data-Efficient Rainbow に対し大幅に劣後し(一部ゲームで逆転)、連続制御の未解決領域を示す [cale (2024/10)]。実装は v0.10.0(2024-09-24)で ALE 本体に統合済みで、C++ 側は `act(Action, float paddle_strength)` の後方互換な拡張として実現されている [ale-release-v0100 (2024/09)]。

---

## 6. 代替・派生環境

### 6.1 MinAtar(ミニチュア Atari)

Young & Tian 2019(arXiv、被引用数 164 + 分裂レコード 29)。表現学習の計算コストを削り行動学習の研究に集中させるため、5 ゲーム(Seaquest, Breakout, Asterix, Freeway, Space Invaders)を 10×10 グリッド × ゲーム固有チャネルのバイナリ状態で再実装した。

```
Each game plays out on a 10x10 grid with n channels corresponding to game-specific objects, such as ball, paddle and brick in the game Breakout.

各ゲームは 10×10 グリッド上で展開され、n 個のチャネルが Breakout におけるボール・パドル・ブロックのようなゲーム固有オブジェクトに対応する。
```

[minatar (2019/03), abstract]

Revisiting Rainbow(Obando-Ceron & Castro, ICML 2021)が主要テストベッドとして採用するなど、「小規模環境での網羅的実験」の定番となっている [minatar-github (2026/08 取得); revisiting-rainbow (2021)]。

### 6.2 JAX ネイティブ系

- **gymnax**(Lange 2022, Star 913): MinAtar の JAX 移植を含む環境ライブラリ。移植は 4/5 ゲームで Seaquest-MinAtar は環境表に存在しない。A100・2000 並列で Breakout 0.19s/1M steps 等の実測を README に掲載 [gymnax (2026/08 取得)]。
- **JAXAtari**(TU Darmstadt k4ntz ラボ系, 2025–2026): Atari ゲームロジックを JAX でネイティブ再実装した object-centric フレームワーク。16 環境を featured として列挙。docs は JIT + GPU 並列化で「最大 16,000 倍の学習速度」を主張(自己申告値)。**ROM を実行するエミュレータではなくゲームロジックの再実装**である点が本質的な特徴/制約(Octax 論文による第三者言及でも同旨)。論文は GitHub citation(2026)のみで arXiv 掲載は未確認 [jaxatari (2026/08 取得); octax (2025/10)]。
- **Octax**(Radji et al., arXiv 2025-10): CHIP-8(Atari の前身)エミュレーションベースの JAX アーケード環境群。「Atari ゲームへの end-to-end GPU 代替」を標榜し、CPU エミュレータ比で桁違いの高速化を主張 [octax (2025/10), abstract]。
- **A Differentiable Atari VCS**(Maier et al., arXiv 2026-06): Atari 2600 の微分可能エミュレータ(Julia 版 jutari / JAX 版 jaxtari)。xitari との bit-for-bit 検証済み(64/64 RAM・画面一致)。JAXAtari と異なりエミュレータの移植であり、XAI の ground truth 用途 [diff-vcs (2026/06)]。

### 6.3 OCAtari / HackAtari / AtariARI(状態注釈・改変系)

- **OCAtari**(Delfosse et al., RLJ/RLC 2024、被引用数 36): ALE 上の object-centric 状態抽出。画面からのルールベース抽出(VEM)と RAM からの抽出(REM)の 2 モードで、論文 v2 時点 VEM 46 / REM 44 ゲーム(README 現在は ~57)。AtariARI の「改良・拡張・object-centric 版」を自認する [ocatari (2023/06); ocatari-github (2026/08 取得)]。
- **HackAtari**(Delfosse et al., arXiv 2024-06、被引用数 13): OCAtari 上でゲーム要素の改変(色スワップ、カリキュラム用簡易化、報酬関数注入など)により、汎化・頑健性テスト用の novel variant を生成する。C51/PPO の頑健性欠陥を実証 [hackatari (2024/06)]。
- **AtariARI**(Anand et al., NeurIPS 2019、被引用数 284): 22 ゲームの RAM(128 bytes/step)に状態変数ラベル(エージェント位置、score/lives 等)を注釈し、表現学習の評価に使う。ALE 公式 FAQ がスプライト位置抽出の回答として公式に参照している [atariari (2019/06); ale-faq (2026/08 取得)]。

### 6.4 Gym Retro / stable-retro(Atari 2600 以遠のコンソール)

オリジナルの Gym Retro(OpenAI、約 1000 ゲーム統合、Atari 2600 / NES / SNES / Genesis / GBA 等)は 2026-05-29 に GitHub リポジトリがアーカイブ(read-only)化された。ROM は非同梱で自己調達が必要 [gym-retro (2026/08 取得)]。

後継の stable-retro(Farama Foundation 配下)が PR の受け皿となっている。

```
Since gym-retro is in maintenance now, you can instead submit PRs with new games or features here in stable-retro.

gym-retro は現在メンテナンスモードのため、新しいゲームや機能の PR は代わりにここ stable-retro に提出できる。
```

[stable-retro (2026/08 取得), README]

Gymnasium API 化済み。対応プラットフォームは Sega Master System/Genesis/CD/32X/Saturn/Dreamcast、NES、SNES、Nintendo 64/DS、Atari 2600、アーケード等(一部はビルド設定依存)。ROM は No-Intro checksum 検証つきの自己調達 [stable-retro-docs (2026/08 取得)]。

### 6.5 Atari 以遠の代替ベンチマーク(簡潔)

- **Procgen**(Cobbe et al., ICML 2020、被引用数 728): 16 の手続き生成環境。ALE の汎化評価の弱さ(train/test を level 単位で分離できない)を動機とし、64×64 RGB・15 離散アクション。リポジトリは Maintenance 状態 [procgen (2019/12)]。
- **Crafter**(Hafner, ICLR 2022、被引用数 222): 2D Minecraft 風手続き生成サバイバル。22 の達成項目で能力スペクトラムを評価。JAX 高速版の Craftax(ICML 2024、被引用数 87)が派生 [crafter (2021/09); craftax (2024/02)]。
- **NetHack Learning Environment / MiniHack**(NeurIPS 2020 / 2021、被引用数 236 / 123): ターミナルベース roguelike。手続き生成・高難度で、現行 SOTA エージェントにも「extremely challenging」。MiniHack はレベル・報酬をカスタム設計できるサンドボックス [nle (2020); minihack (2021)]。

---

## 7. 批判・懸念

### 7.1 飽和論と擁護論

Atari 100k は BBF / EfficientZero V2 で IQM > 1(superhuman)に達しており、26 ゲーム版は飽和気味というのがコミュニティの共通認識である(BBF 自身が新ゴール提案 — §5.3)。一方、第一線研究者からの擁護論もある。Castro(Google DeepMind、CALE 共著者)は「ALE は solved」という査読者評へ反論するブログを公開した。

```
aggregate results can hide important per-game differences between algorithms!

集計結果は、アルゴリズム間のゲームごとの重要な差異を隠しうる!
```

[castro-blog (2024/12)]

未解決の具体例として、100k と 10M でハイパーパラメータ最適値が大きく異なること、ゲーム単位のランキング逆転、CALE での連続制御エージェントの大幅劣後を挙げている [castro-blog (2024/12)]。

### 7.2 プロトコル分裂の注意点

- v5 デフォルトの sticky actions(0.25)は性能に悪影響を与えうると公式が明記しており、比較時はパラメータの一致確認が必要 [ale-docs-env (2026/08 取得)]。
- Atari 100k は sticky actions なし、古典 57 ゲームは v5 化で sticky actions ありが標準と、**サブベンチマーク間でプロトコルが分裂している** [bbf (2023/05); ale-docs-env (2026/08 取得)]。
- Korkmaz(AAAI 2026)は、低データ域の性能順位が大データ域で保存されない(単調関係がない)ことを示しており、100k の順位を 200M へ外挿できないことを示唆する [korkmaz (2026/07)]。

### 7.3 CPU 律速批判(GPU ネイティブ環境派)

ALE は CPU エミュレーションであるため、GPU ネイティブ環境(JAX 系)からの代替提案が相次いでいる。Octax は「JAX コミュニティが待望していた Atari ゲームへの end-to-end GPU 代替」を自称し、CPU エミュレータの桁違いの高速化を動機に挙げる [octax (2025/10)]。ALE 側も AtariVectorEnv(C++ マルチスレッド)や XLA GPU 対応で応じている(§4.1)[ale-releases (2026/08 取得)]。なお Craftax の批判対象は Crafter/NetHack/Minecraft であり Atari を直接名指ししてはいない [craftax (2024/02)]。

### 7.4 ROM ライセンスの揺らぎ

pip 版は ROM 同梱に移行したが(§3.3)、conda-forge は著作権状態を確認できないとして同梱を拒否しており、配布チャネルによって判断が割れている [conda-forge-ale (2026/08 取得)]。C++ 利用では ROM 自己調達が前提となる点も含め、ROM の法的位置づけは完全には解決していない。

---

## 8. 総合評価

以下は本レポートの綜合判断である(各事実の出典は前掲各節)。

**ALE が第一候補であることは調査結果から強く支持される。** 根拠: (1) Farama Foundation により活発に保守されており(最終 push 2026-08-10、v0.12.0 が 2026-05)、Gymnasium エコシステムの正式な一部である。(2) DQN 以来の膨大な比較事例(Rainbow / MuZero / Agent57 / Atari 100k 系)がすべてこの環境上にあり、「広く事例比較が可能」という要求に最も合致する。(3) 決定的に重要な点として、**ALE は正規の C++ ライブラリとして設計されている** — `ALEInterface` ヘッダ + CMake config + pkg-config がインストールされ、状態クローン・flavor 制御・RAM 観測まで C++ API で完結する。Python/GymEnv 連携を経由せず C++ フレームワークへ直結する選択肢が公式にサポートされている。

**C++ 統合の実務上の要点**: vcpkg/Conan のレジストリには存在しないため、ソースからの CMake ビルド(vcpkg manifest で zlib 等を解決)が公式ルート。必須依存は zlib のみと軽量。ROM は .bin の自己調達が必要(pip 同梱 ROM は Python パッケージ側の仕組み)。ベクトル化が必要なら `BUILD_VECTOR_LIB=ON`(+opencv4)で C++ 実体をそのまま使える。

**プロトコル選択**: 新規に始めるなら v5 相当(sticky actions 0.25 / frameskip 4 = Machado et al. 2018 勧告)が現行標準。ただし Atari 100k 系の事例と比較する場合は sticky actions なしが慣行である点に注意。評価集計は median でなく IQM + 信頼区間(rliable 流)が現行標準。フル 57 ゲームが重すぎる場合、Atari-5 が ablation 用の廉価サブセットとして査読論文でも使われている。

**代替環境の位置づけ**: MinAtar は「アルゴリズム検証を安価に回す」用途で ALE と補完関係にあり、導入コストも低い。JAX 系(JAXAtari / Octax)は高速だが、ROM 再実装ゆえに ALE との数値比較可能性が失われる点と論文実績の薄さから、事例比較目的では現時点で主役にならない。stable-retro は Atari 2600 以遠への拡張パスとして控えに置ける。

---

## 9. 調査の限界

- **v0.12.0 のリリース日**: PyPI JSON API 由来の日付(2025-05-29)と GitHub Releases API 由来の日付(2026-05-29)がサブエージェント間で食い違った。v0.11.2(2025-07-12)より後であることから GitHub API の 2026-05-29 を採用したが、PyPI 側の生データ再確認はしていない。
- **EnvPool の 2026 年復活**(1.x 系、Windows/macOS wheel、gym 依存除去)は調査時点の PyPI / GitHub 一次ソースのみに基づき、第三者による検証記事は未確認。
- 被引用数はすべて Semantic Scholar(補助: OpenAlex)。Google Scholar 値は取得手段がなく未確認。EnvPool(S2: 84 / OpenAlex: 14)、MinAtar(164 + 分裂レコード 29)、Machado 2018 などでレコード分裂による過小計上の可能性がある。
- R2D2 の abstract は OpenReview の bot 認証で取得できず、具体的数値(Atari-57 SOTA 更新幅等)は未検証。
- HNS 式の一次出典(Mnih et al. 2015, Nature)は未フェッチ。
- Machado et al. 2018 の「loss-of-life でエピソードを打ち切らない」勧告の verbatim は abstract から取得できておらず、本文未確認(v5 デフォルトが同論文準拠であること自体は ALE 公式 docs で確認済み)。
- JAXAtari の論文は GitHub citation(2026)のみで arXiv 掲載を発見できなかった。HackAtari の venue(RLC 2024 workshop)は著者側リスト経由の間接情報。
- ALE がベンダリングする Stella の正確なバージョン番号はソース・ドキュメントに明記がなく、特定できなかった(2007 年頃のスナップショットであることのみ確認)。
- multi-agent-ale-py の Windows 非対応は wheel の有無からの判断で、sdist ビルドの実際の成否は未検証。

---

## 10. ソースリスト

### 公式ドキュメント・リポジトリ(ALE / Farama)

- [ale-github, 2026/08 取得] Farama Foundation. "Arcade-Learning-Environment." GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment
- [ale-docs-cpp, 2026/08 取得] Farama Foundation. "C++ Interface." ALE Documentation. https://ale.farama.org/cpp-interface/
- [ale-docs-env, 2026/08 取得] Farama Foundation. "Environments." ALE Documentation. https://ale.farama.org/environments/
- [ale-docs-gym, 2026/08 取得] Farama Foundation. "Gymnasium Interface." ALE Documentation. https://ale.farama.org/gymnasium-interface/
- [ale-docs-vector, 2026/08 取得] Farama Foundation. "ALE Vector Environment Guide." ALE Documentation. https://ale.farama.org/vector-environment/
- [ale-docs-ma, 2026/08 取得] Farama Foundation. "Multi-Agent Environments." ALE Documentation. https://ale.farama.org/multi-agent-environments/
- [ale-faq, 2026/08 取得] Farama Foundation. "docs/faq.md." GitHub. https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/master/docs/faq.md
- [ale-releases, 2026/08 取得] Farama Foundation. "Releases." GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases
- [ale-release-v090, 2024/05] Farama Foundation. "Release v0.9.0." GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.9.0
- [ale-release-v0100, 2024/09] Farama Foundation. "Release v0.10.0." GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.10.0
- [ale-release-v0110, 2025/04] Farama Foundation. "Release v0.11.0." GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases/tag/v0.11.0
- [ale-source, 2026/08 取得] Farama Foundation. master ブランチのソース実査(`src/ale/ale_interface.hpp`, `CMakeLists.txt`, `src/ale/CMakeLists.txt`, `vcpkg.json`, `src/ale/games/supported/`, `src/ale/vector/`). GitHub. https://github.com/Farama-Foundation/Arcade-Learning-Environment
- [ale-emucore, 2026/08 取得] Farama Foundation. "src/ale/emucore/OSystem.hxx." GitHub. https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/master/src/ale/emucore/OSystem.hxx
- [autorom, 2025/08] Farama Foundation. "AutoROM (deprecation notice)." GitHub. https://github.com/Farama-Foundation/AutoROM
- [pettingzoo-atari, 2026/08 取得] Farama Foundation. "Atari Environments." PettingZoo Documentation. https://pettingzoo.farama.org/environments/atari/
- [shimmy, 2026/08 取得] Farama Foundation. "Shimmy Documentation." https://shimmy.farama.org/
- [gymnasium-wrappers, 2026/08 取得] Farama Foundation. "Misc Wrappers — AtariPreprocessing." Gymnasium Documentation. https://gymnasium.farama.org/api/wrappers/misc_wrappers/
- [gymnasium-wrappers-obs, 2026/08 取得] Farama Foundation. "Observation Wrappers — FrameStackObservation." Gymnasium Documentation. https://gymnasium.farama.org/api/wrappers/observation_wrappers/
- [gymnasium-v029-atari, 2026/08 取得] Farama Foundation. "Atari." Gymnasium Documentation (v0.29.0). https://gymnasium.farama.org/v0.29.0/environments/atari/

### パッケージレジストリ

- [pypi-ale-py, 2026/08 取得] "ale-py (JSON API)." PyPI. https://pypi.org/pypi/ale-py/json
- [pypi-envpool, 2026/08 取得] "envpool (JSON API)." PyPI. https://pypi.org/pypi/envpool/json
- [pypi-ma-ale, 2026/08 取得] "multi-agent-ale-py (JSON API)." PyPI. https://pypi.org/pypi/multi-agent-ale-py/json
- [conda-forge-ale, 2026/08 取得] conda-forge. "ale-py." Anaconda.org. https://anaconda.org/conda-forge/ale-py

### 査読論文・プレプリント

- [bellemare2013, 2013] Bellemare, M. G., Naddaf, Y., Veness, J., Bowling, M. "The Arcade Learning Environment: An Evaluation Platform for General Agents." JAIR 47. https://arxiv.org/abs/1207.4708
- [machado2018, 2017/09] Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M. "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." JAIR 61. https://arxiv.org/abs/1709.06009
- [cale, 2024/10] Farebrother, J., Castro, P. S. "CALE: Continuous Arcade Learning Environment." NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2410.23810
- [simple, 2019/03] Kaiser, Ł. et al. "Model-Based Reinforcement Learning for Atari." ICLR 2020. https://arxiv.org/abs/1903.00374
- [efficientzero, 2021/10] Ye, W., Liu, S., Kurutach, T., Abbeel, P., Gao, Y. "Mastering Atari Games with Limited Data." NeurIPS 2021. https://arxiv.org/abs/2111.00210
- [bbf, 2023/05] Schwarzer, M., Obando-Ceron, J., Courville, A., Bellemare, M., Agarwal, R., Castro, P. S. "Bigger, Better, Faster: Human-level Atari with human-level efficiency." ICML 2023. https://arxiv.org/abs/2305.19452
- [ez-v2, 2024/03] Wang, S., Liu, S., Ye, W., You, J., Gao, Y. "EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data." ICML 2024. https://arxiv.org/abs/2403.00564
- [dreamerv3, 2023/01] Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. "Mastering diverse control tasks through world models." Nature 2025 (arXiv:2301.04104). https://arxiv.org/abs/2301.04104
- [rainbow, 2018] Hessel, M. et al. "Rainbow: Combining Improvements in Deep Reinforcement Learning." AAAI 2018. https://arxiv.org/abs/1710.02298
- [r2d2, 2019] Kapturowski, S. et al. "Recurrent Experience Replay in Distributed Reinforcement Learning." ICLR 2019. https://openreview.net/forum?id=r1lyTjAqYX
- [muzero, 2019/11] Schrittwieser, J. et al. "Mastering Atari, Go, chess and shogi by planning with a learned model." Nature 2020. https://arxiv.org/abs/1911.08265
- [agent57, 2020/03] Badia, A. P. et al. "Agent57: Outperforming the Atari Human Benchmark." ICML 2020. https://arxiv.org/abs/2003.13350
- [atari5, 2022/10] Aitchison, M., Sweetser, P., Hutter, M. "Atari-5: Distilling the Arcade Learning Environment down to Five Games." ICML 2023. https://arxiv.org/abs/2210.02019
- [rliable, 2021/08] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021. https://arxiv.org/abs/2108.13264
- [btr, 2024/11] Clark, T., Towers, M., Evers, C., Hare, J. "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a Desktop PC." ICML 2025. https://arxiv.org/abs/2411.03820
- [korkmaz, 2026/07] Korkmaz, E. "Principled Analysis of Deep Reinforcement Learning Evaluation and Design Paradigms." AAAI 2026. https://arxiv.org/abs/2607.07769
- [envpool-paper, 2022/06] Weng, J. et al. "EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine." NeurIPS 2022 Datasets and Benchmarks. https://arxiv.org/abs/2206.10558
- [minatar, 2019/03] Young, K., Tian, T. "MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments." arXiv. https://arxiv.org/abs/1903.03176
- [revisiting-rainbow, 2021] Obando-Ceron, J., Castro, P. S. "Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research." ICML 2021. https://arxiv.org/abs/2011.14826
- [ocatari, 2023/06] Delfosse, Q., Blüml, J., Gregori, B., Sztwiertnia, S., Kersting, K. "OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments." RLJ/RLC 2024. https://arxiv.org/abs/2306.08649
- [hackatari, 2024/06] Delfosse, Q., Blüml, J., Gregori, B., Kersting, K. "HackAtari: Atari Learning Environments for Robust and Continual Reinforcement Learning." arXiv. https://arxiv.org/abs/2406.03997
- [atariari, 2019/06] Anand, A., Racah, E., Ozair, S., Bengio, Y., Côté, M.-A., Hjelm, R. D. "Unsupervised State Representation Learning in Atari." NeurIPS 2019. https://arxiv.org/abs/1906.08226
- [octax, 2025/10] Radji, W., Michel, T., Piteau, H. "Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX." arXiv. https://arxiv.org/abs/2510.01764
- [diff-vcs, 2026/06] Maier, A., Bayer, S., Krauss, P. "A Differentiable Atari VCS: A Complex, Fully Known Ground Truth for Explainable AI." arXiv. https://arxiv.org/abs/2606.22447
- [procgen, 2019/12] Cobbe, K. et al. "Leveraging Procedural Generation to Benchmark Reinforcement Learning." ICML 2020. https://arxiv.org/abs/1912.01588
- [crafter, 2021/09] Hafner, D. "Benchmarking the Spectrum of Agent Capabilities." ICLR 2022. https://arxiv.org/abs/2109.06780
- [craftax, 2024/02] Matthews, M. et al. "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning." ICML 2024. https://arxiv.org/abs/2402.16801
- [nle, 2020] Küttler, H. et al. "The NetHack Learning Environment." NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/569ff987c643b4bedf504efda8f786c2-Abstract.html
- [minihack, 2021] Samvelyan, M. et al. "MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research." NeurIPS 2021 Datasets and Benchmarks. https://openreview.net/forum?id=skFwlyefkWJ

### GitHub リポジトリ・その他(第三者含む)

- [envpool-github, 2026/08 取得] sail-sg. "envpool README." GitHub. https://github.com/sail-sg/envpool
- [envpool-releases, 2026/08 取得] sail-sg. "envpool Releases." GitHub. https://github.com/sail-sg/envpool/releases
- [cleanrl-docs, 2026/08 取得] CleanRL. "DQN — Documentation." https://docs.cleanrl.dev/rl-algorithms/dqn/
- [sb3-docs, 2026/08 取得] Stable-Baselines3. "Examples." Read the Docs. https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
- [dopamine-github, 2026/08 取得] Google. "dopamine README." GitHub. https://github.com/google/dopamine
- [sample-factory, 2026/08 取得] "Sample Factory Documentation." https://www.samplefactory.dev/
- [minatar-github, 2026/08 取得] Young, K. "kenjyoung/MinAtar." GitHub. https://github.com/kenjyoung/MinAtar
- [gymnax, 2026/08 取得] Lange, R. T. "RobertTLange/gymnax." GitHub. https://github.com/RobertTLange/gymnax
- [jaxatari, 2026/08 取得] k4ntz Lab (TU Darmstadt). "JAXAtari." GitHub / Read the Docs. https://github.com/k4ntz/JAXAtari / https://jaxatari.readthedocs.io/
- [ocatari-github, 2026/08 取得] k4ntz Lab (TU Darmstadt). "OC_Atari." GitHub. https://github.com/k4ntz/OC_Atari
- [gym-retro, 2026/08 取得] OpenAI. "openai/retro (archived 2026-05-29)." GitHub. https://github.com/openai/retro
- [stable-retro, 2026/08 取得] Farama Foundation. "stable-retro." GitHub. https://github.com/Farama-Foundation/stable-retro
- [stable-retro-docs, 2026/08 取得] Farama Foundation. "Stable-Retro Documentation." https://stable-retro.farama.org/
- [castro-blog, 2024/12] Castro, P. S. (Google DeepMind). "In Defense of Atari - the ALE is not 'solved'!" 個人ブログ. https://psc-g.github.io/posts/research/rl/atari_defense/
- [stella-site, 2026/08 取得] Stella team. "Stella: A multi-platform Atari 2600 VCS emulator." https://stella-emu.github.io/
- [s2-api, 2026/08 取得] Semantic Scholar Graph API(全被引用数の出典). https://api.semanticscholar.org/graph/v1/
