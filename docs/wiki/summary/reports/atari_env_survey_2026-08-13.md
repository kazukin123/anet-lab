---
title: "Survey: Atari RL 環境 — ALE を中心としたゲームバリエーション・ライブラリ・ベンチマーク動向"
page_kind: サマリ
material: "[[reports/atari_env_survey_2026-08-13.md|atari_env_survey_2026-08-13]]"
material_kind: 報告
concepts:
  - "[[docs/wiki/concepts/subjects/atari.md|Atari横断]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-26
---
# Survey: Atari RL 環境 — ALE を中心としたゲームバリエーション・ライブラリ・ベンチマーク動向

## 元の材料

[Survey: Atari RL 環境 — ALE を中心としたゲームバリエーション・ライブラリ・ベンチマーク動向](../../../../reports/atari_env_survey_2026-08-13.md) blob:d13a16d5365aecd204885d43203346e11b4db32b

## 要点

- ALE は Atari 2600 エミュレータ Stella の上に構築された RL 研究用フレームワークで、エミュレーション詳細をエージェント設計から分離することを設計思想とする。
- Gymnasium 登録環境のバージョンサフィックスは、frameskip と sticky actions(行動の確率的リピート)のデフォルトが異なる [ale-docs-env (2026/08 取得); gymnasium-v029-atari (2026/08 取得)]。
- Machado et al. 2018(JAIR、被引用数 627)は評価方法論の乱立への懸念から、sticky actions によるstochasticity 導入、flavor(mode/difficulty)対応を含む新版 ALE と評価のベストプラクティスを提示した。
- 26 ゲームの選定基準は「既存の最先端モデルフリー深層 RL で解けること」(同論文本文)。100k agent steps = 400k frames(frameskip 4)。標準プロトコルでは sticky actions なしが慣行である(BBF が sticky actions 付きへの移行を提案していることの裏返し)[bbf (2023/05)]。
- Atari 100k は sticky actions なし、古典 57 ゲームは v5 化で sticky actions ありが標準と、サブベンチマーク間でプロトコルが分裂している [bbf (2023/05); ale-docs-env (2026/08 取得)]。

## 内容

### 1. ALE 本体 — プロジェクト概要とゲームバリエーション

ALE がベンダリングしている Stella は 2007 年頃のスナップショットのフォークであり(`src/ale/emucore/` に取り込み、`namespace ale::stella` で独自保守)、正確な Stella バージョン番号はソース・ドキュメントいずれにも明記がない [ale-emucore (2026/08 取得), OSystem.hxx 著作権表記 "1995-2007"]。

v0.11 で旧名前空間(`*NoFrameskip-v4` 等)が整理され、登録環境数は大幅に削減された。

ALE はゲームごとに複数のモードと難易度を公開しており、その組合せを flavor と呼ぶ。これは Machado et al. 2018 で導入された概念である。

### 2. C++ インターフェース

要件は C++17 コンパイラ + CMake 3.14 以上(トップ CMakeLists の `cmake_minimum_required` を直接確認)。依存管理は vcpkg の manifest mode をネイティブサポートし、README は CMake を "a first class citizen" と位置づけている [ale-github (2026/08 取得)]。

注意点として、microsoft/vcpkg の ports にも ConanCenter にも ALE は登録されていない(GitHub API 404 で確認)。つまり C++ 利用の公式ルートは「ソースからの CMake ビルド(依存解決に vcpkg manifest を利用)」であり、`vcpkg.json` は ALE 自身の依存を取るためのもので、ALE 本体が vcpkg port として配布されているわけではない [ale-source (2026/08 取得)]。

### 3. Gymnasium 連携と Python エコシステム

連続行動は `gym.make("ALE/Breakout-v5", continuous=True)` で有効化でき、行動空間は極座標(radius, theta)+ fire の 3 次元 Box となる [ale-docs-gym (2026/08 取得)]。

AutoROM 自体も 2025 年 8 月に正式に非推奨化された [autorom (2025/08), README]。ただし conda-forge 版 ale-py は ROM の著作権状態を確認できないとして ROM を同梱せず、合法的に入手した ROM を `ALE_ROMS_DIR` で指すよう求めており、法的位置づけには揺らぎが残る(§7.4)[conda-forge-ale (2026/08 取得)]。

### 4. ベクトル化・高速実行系

v0.11.0 で、EnvPool にインスパイアされた C++ 非同期ベクトル化環境が ALE 本体に導入された。

XLA サポートは v0.11.1 で実験導入され、v0.12.0 で GPU 対応が加わった [ale-releases (2026/08 取得)]。

### 5. ベンチマークプロトコルと研究動向

起源は SimPLe 論文(Kaiser et al., ICLR 2020、被引用数 1,004)。100k interactions ≒ リアルタイム 2 時間のプレイに相当する低データ域を定義した。

Aitchison, Sweetser, Hutter(ICML 2023、被引用数 52)は、5 ゲーム(Battle Zone, Double Dunk, Name This Game, Phoenix, Q*bert)の回帰で 57 ゲーム median を推定するサブセットを提案した。

### 6. 代替・派生環境

Young & Tian 2019(arXiv、被引用数 164 + 分裂レコード 29)。表現学習の計算コストを削り行動学習の研究に集中させるため、5 ゲーム(Seaquest, Breakout, Asterix, Freeway, Space Invaders)を 10×10 グリッド × ゲーム固有チャネルのバイナリ状態で再実装した。

A Differentiable Atari VCS(Maier et al., arXiv 2026-06): Atari 2600 の微分可能エミュレータ(Julia 版 jutari / JAX 版 jaxtari)。xitari との bit-for-bit 検証済み(64/64 RAM・画面一致)。JAXAtari と異なりエミュレータの移植であり、XAI の ground truth 用途 [diff-vcs (2026/06)]。

### 7. 批判・懸念

Atari 100k は BBF / EfficientZero V2 で IQM > 1(superhuman)に達しており、26 ゲーム版は飽和気味というのがコミュニティの共通認識である(BBF 自身が新ゴール提案 — §5.3)。一方、第一線研究者からの擁護論もある。Castro(Google DeepMind、CALE 共著者)は「ALE は solved」という査読者評へ反論するブログを公開した。

v5 デフォルトの sticky actions(0.25)は性能に悪影響を与えうると公式が明記しており、比較時はパラメータの一致確認が必要 [ale-docs-env (2026/08 取得)]。

### 8. 総合評価

ALE が第一候補であることは調査結果から強く支持される。 根拠: (1) Farama Foundation により活発に保守されており(最終 push 2026-08-10、v0.12.0 が 2026-05)、Gymnasium エコシステムの正式な一部である。(2) DQN 以来の膨大な比較事例(Rainbow / MuZero / Agent57 / Atari 100k 系)がすべてこの環境上にあり、「広く事例比較が可能」という要求に最も合致する。(3) 決定的に重要な点として、ALE は正規の C++ ライブラリとして設計されている — `ALEInterface` ヘッダ + CMake config + pkg-config がインストールされ、状態クローン・flavor 制御・RAM 観測まで C++ API で完結する。Python/GymEnv 連携を経由せず C++ フレームワークへ直結する選択肢が公式にサポートされている。

プロトコル選択: 新規に始めるなら v5 相当(sticky actions 0.25 / frameskip 4 = Machado et al. 2018 勧告)が現行標準。ただし Atari 100k 系の事例と比較する場合は sticky actions なしが慣行である点に注意。評価集計は median でなく IQM + 信頼区間(rliable 流)が現行標準。フル 57 ゲームが重すぎる場合、Atari-5 が ablation 用の廉価サブセットとして査読論文でも使われている。

## 主要な概念

- [Atari横断](../../concepts/subjects/atari.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
