---
title: 環境
page_kind: サマリ
material: "[[docs/design/120_environments.jp.md|120_environments.jp]]"
material_kind: 設計文書
concepts:
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/subjects/gridmaze.md|GridMaze]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# 環境

## 元の材料

[環境](../../../design/120_environments.jp.md) blob:3fede7bf243005699373bcec92e8c6684b71d863

## 要点

- EnvSpec は観測と行動の契約を表し、構築後に shape や行動数を変えない。
- Reset と Step は次の行動に使う状態と、完了した遷移について通知する値を区別する。
- 単体環境を batch にまとめた場合も、完了した group の数と次の episode_start が対応する。
- 具象 Env は共通契約を保ちつつ、GridMaze の迷路、LunarLander の画面、DropMerge の報酬などを個別に実装する。
- 終了時の処理と返した結果の所有を明示し、利用側が無効なバッファを参照するのを防ぐ。

## 内容

### 2. 基本概念と外部contract

- `BatchEnvSpec`は`num_envs`、`num_threads`、`episode_scope`を持つ。`episode_scope=PER_LANE`では各laneが一つのepisode group、`SHARED`ではbatch全体が一つのepisode groupである。JSON表記は`per_lane` / `shared`とする。

### 3. コンポーネント定義

- `BatchEnv` — batch化されたReset/Stepとspec、device、shutdownを公開する。
- Envの公開初期化関数は、同じ関数が複数回呼ばれてもEnv factoryとView creatorを1回だけ登録する。途中で失敗した場合は、以後の呼び出しも失敗する。`ViewRepository`へ同じclass IDのcreatorを直接登録すると例外になり、先に登録したcreatorが残る。

### 5. 静的構造

- single Envのbatch実行方法は共通基盤が担当する。batch生成自体がドメイン処理であるImageClsはnative `BatchEnv`として実装する。

### 6. 処理フロー

- `env.class_id`から`EnvRepository`がsingle/batchいずれかの具象factoryを解決する。
- single経路はworker方式に応じてvectorizedまたはthread-pool wrapperへ格納する。native ImageClsでは同じworker設定をSource内sample処理へ適用し、decode/cache lookupからaugmentationまでを同一workで実行する。

### 7. 設定・lifetime・エラー・性能特性

- Env固有設定は各factoryが同じConfigDataから読み取る。未知のclass ID、不正なworker設定、矛盾したspecは暗黙に補正せず失敗させる。
- Envのmutable stateとRNGはEnv instanceごとに分離する。ImageClsのimmutable Dataset/manifest/cacheだけはDatasetKey単位でprocess共有する。
- `num_envs`が小さい、またはEnv Stepが軽量な場合、thread-poolの同期costが利益を上回ることがある。

### 8. テストと拡張時の確認事項

- single具象Env testは`VectorizedDiscreteBatchEnv`を通る経路も含む。ImageCls testはnative batch、Dataset catalog/cache、eval window、worker方式を直接検証する。Envまたはwrapperを変更する場合は、必要なworker方式を明示して回帰testを追加する。

## 主要な概念

- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [GridMaze](../../concepts/subjects/gridmaze.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
