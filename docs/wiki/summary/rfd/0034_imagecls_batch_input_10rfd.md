---
title: ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter
page_kind: サマリ
material: "[[docs/memo/done/0034_imagecls_batch_input_10rfd.md|0034_imagecls_batch_input_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/subjects/imagecls.md|ImageCls]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
  - "[[docs/wiki/concepts/framework/320_env_core.md|320 Env 共通基盤]]"
updated: 2026-11-26
---
# ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter

## 元の材料

[ImageCls 専用 batch 入力コンポーネント / BatchEnv adapter](../../../memo/done/0034_imagecls_batch_input_10rfd.md) blob:b132e1234ea64195bed84c45e6ea8f4ac9614969

## 要点

- ImageCls の個体ごとの画像読み出しを、バッチ単位の入力と BatchEnv アダプターに移す。
- 従来は各 Env が学習用・評価用のデータソースを持ち、1枚ずつデコードした後にフレームワーク側でまとめていた。
- Dataset の責務とサンプル選択を整理し、学習と評価の契約を保つ。
- 画像キャッシュの上限を明示し、auto の容量超過や確保失敗は警告してキャッシュなしへ切り替える。
- full_ram を明示したときに確保できなければエラーにし、設定の意図を保つ。

## 内容

### 設定外部仕様

env-local DataSource設定: 使用する`DatasetKey`、augment、eval samplingを定義する。
trainer / eval設定: train/evalのB、起動頻度、background、actor cloneなどを定義する。
Datasetのpath、shape、cache policyをeval env prefixから直接上書きしない。eval envが変えられるのは参照する`dataset_key`とenv-localなSource設定だけである。同じデータを異なるcache policyで扱う場合は別`DatasetKey`を明示する。
`ConfigData::Read`は現行、型変換失敗時にWARN後defaultへ戻り得るため、本機能の明示設定では「キーが存在するのに型変換へ失敗した」場合を`ANET_SYSTEM_ERROR`にするvalidated readが必要である。
evalの`accuracy`は直近eval windowの正解率。fullなら全件、subsetならそのwindowのvalid sampleを対象とする。

### 外部動作仕様

同一keyへ異なるresolved configを要求した場合は、先行instanceを再利用せずfail-fastする。
生成済みDataset/cacheはeviction/resetせずprocess終了まで保持する。既存keyに対応するconfigまたはDataset実ファイルをin-place変更した場合はprocessを再起動する。同一process中に更新版を併用する場合は、新directoryと新DatasetKeyで別Datasetとして登録する。
ManagerとDatasetはthread-safe、Sourceはenv-local single-caller contractとする。
`NextBatch`呼出しごとにgrid / targetsのfresh storageを生成し、返却後はimmutableとする。
batch内の重複indexはdecode/cache lookupの重複作業を減らすためdedupeする。共有cacheの排他責任はDatasetにある。

### クラス別責務・所有権

pathはabsolute化、separator統一、lexical normalizationを行い、Windowsではcase-insensitiveに比較する。I/O無しの`RegisterCatalog`を維持するためsymlink/junctionの実体解決は行わず、別aliasは異configとしてfail-fastしてよい。
`Instance()`はImageCls factory / Source経路だけから初回利用され、他Envでは生成・登録されない。ImageClsを使わないprocessの実質的なresource消費はない。
preflight成功後だけ全新規entryをatomic commitする。manifest I/Oやcache確保は行わず、entryを`Registered`状態にするだけとする。
`Get(index)`は複数Sourceから同時に呼べるthread-safe APIとする。
manifestと画像fileの内容がprocess中に変更されないことを値の一貫性と再現性の前提とする。

### コミュニケーション図

基準案ではtrainとevalは異なるDatasetKeyなので別Dataset/cacheを持つ。

### 詳細設計・現行コード制約

env登録の実経路は`Init*()`だが、GridMaze / LunarLander / CartPole / DropMergeは`ANET_REGISTER_ENV_FACTORY`によるstatic登録も持ち、同一class_idへ二重登録される。現行registryは上書きするため顕在化していない。
`MetricsLogEpisodeEndObserver`はEpisodeEndEventごとにenv全体のscalarを記録するため、1 windowでB eventを出すと同じaccuracyとEMAがB回前進する。evalで代表laneだけdoneにする理由である。
`ImageDatasetManager`はprocess-wide config/instance registryを担当する独立singletonとする。
`DecodeResizedImage`はRNGを持たないfree functionとする。
GetSpec / GetBatchSpec / GetDeviceは従来のobs/action semanticsをbatchへ拡張する。

### 実装フェーズ

旧top-level`BatchEnvFactory` interfaceを削除し、`CreateBatchEnv(name, seed, num_envs)`を維持したまま`DefaultBatchEnvFactory`を`BatchEnvBuilder`へ改名する。
PRD 037のreserved name preflightとrun-local name registryを`RunManager`に維持し、改名先Builderへ移さない。
`ANET_REGISTER_ENV_FACTORY`使用4箇所とmacro定義を削除し、`Init*()`登録へ一本化する。
`DiscreteBatchEnvBase::GetScalar`の無prefix fallbackを`ANET_SYSTEM_ERROR`へ変更する。
検証: 全既存Envとtestが従来経路でgreen。改名前後でmain Train、configured Eval、EvalPanelのnameとlane name、および重複時の`ANET_SYSTEM_ERROR`が不変。

### 決定経緯

episodeとepochを分離し、train episodeは可視化/ログcadence、epochはDataset cycleとする。
ImageClsでは役割が挙動ではなく構造（Sampler族の選択、train episode / eval window終端契約）を決めるため、Sampler 1:1をEnv生成時から保証する。GetSpecはReset前に呼ばれるが、生成時に役割・Dataset・Samplerが確定しているため曖昧さがない。
EvalPanel独自の`app.eval_panel.env.*` overlayを設けて`.$`合成で共有する代案は、「configured evalと同じ内容」という意図が機械検証不能になりdriftし得るため不採用。GUIでmetrics evalと異なるデータを見たい場合は、定義専用の寝タグを1個宣言して参照する。
episode終端は選択tagのeval windowとする。旧ランダム復元抽出＋`max_steps=100`終端は互換対象外で、`model_sync.mode=episode`はwindow単位の同期になる（既定のtimeモードは無影響）。
要点は「manifestは早く（fail-fastのため）、payloadは遅く（容量のため）」の非対称である。

## 主要な概念

- [ImageCls](../../concepts/subjects/imagecls.md)
- [322 Env個別実装](../../concepts/framework/322_env_individual.md)
- [320 Env 共通基盤](../../concepts/framework/320_env_core.md)
