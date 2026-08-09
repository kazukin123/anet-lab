# Envログ name prefix 自動付与 PRD

## 問題

[`037_env_instance_name_10prd.md`](037_env_instance_name_10prd.md)で全Envインスタンスへ`name`を付与し、Env本体のtext logへ`[<Env name>] `prefixを付けた。しかし現在の実装には2つの問題がある。

1. prefix付与が各ログ行の手書き連結(`LOG::xxx() << "[" << GetName() << "] " << ...`)であり、prefix書式の知識が全出力行へ分散している。新しいログ行を書くたびに付け忘れ・書式ゆらぎが構造的に起こり得る。
2. `[<Env name>] `の囲み`[ ]`は、lane表記(`train[37]`)が既に`[ ]`を含むため二重になり(`[train[37]]`)、視覚ノイズが大きい。本文にも`Rank [ 5 ]`のような`[ ]`過多の箇所がある。

現状の手書き・未対応箇所は次のとおり。

| 箇所 | 件数 | 現状 |
|---|---:|---|
| `DropMergeEnv.cpp`の`LOG::verbose/info/error` | 10 | `"[" << GetName() << "] "`手書き |
| `ImageClsEnv.cpp`の`LOG::verbose` | 3 | 同上 |
| `LunarLanderEnv.cpp`の`ANET_LOG_DEBUG` | 6 | 同上(マクロ引数内) |
| `env.cpp` batch側の`ANET_LOG_DEBUG`(`DiscreteBatchEnvBase` ctor、`VectorizedDiscreteBatchEnv` ctor、`ThreadPoolDiscreteEnv` ctor/`Step`) | 4 | prefix無し。どのBatchEnvのseed/actionログか識別できない |
| `env.cpp`の`DiscreteBatchEnvBase::GetScalar` WARN | 1 | prefix無し。クラス名を本文へ手書き |

## 目的

1. Env本体ログのname prefix付与を、各行の手書きから基底クラスの仕組みへ移す。
2. prefix書式の知識を1箇所へ集約し、書式を`<Env name>: `へ変更して`[ ]`二重を解消する。
3. `LOG::info()`と同じstream記法・flush契約を維持したまま、`log.info() << "..."`で透過的にprefix付きログを書けるようにする。
4. `ANET_LOG_DEBUG`系にも、guard・ビルド消去特性を保ったまま同じprefixを適用できる手段を提供する。
5. 部品をEnv専用にせず、prefix付きログを出したい任意のクラスが再利用できる汎用部品として`anet::log`へ置く。

## 前提

- PRD 037実装済み: `SingleDiscreteEnvBase`/`BatchEnvBase`がimmutableな`name`を保持し`GetName()`を`final override`する。`BatchEnvBase`はlane name(`<name>[lane_index]`)を構築時に確定する。
- nameの契約(人間向け・不透明・挙動非依存・Run内一意)はPRD 037から変更しない。
- 既存log基盤(`anet::log::WxLogStream`、`LOG::info()`等のfree関数、`FileLogger`、`LogFormatter`)の構造・flush契約・レベル体系は変更しない。
- 適用対象はEnv本体ログ(single/batch)。Runner、Agent、Viewへの適用は行わないが、部品は使える状態にする。
- PRD 037の`[<Env name>] `書式規定は本PRDが上書きする。PRD 037本文は履歴として改訂しない。

## 用語

### Logger

prefix文字列を保持し、prefix書き込み済みの`WxLogStream`を生成する軽量オブジェクト。`anet::log`名前空間の汎用部品。

### prefix

Loggerが保持する`"<name>: "`形式の文字列。1ログ行の本文先頭に一度だけ出力される。

## 解決方針

### 1. 汎用部品`log::Logger`を追加する

`log.hpp`の`anet::log`へ`Logger`クラスを追加する。

```cpp
// prefix付きでWxLogStreamを生成する軽量logger。
// Envなどインスタンス識別が必要なクラスがメンバとして保持する。
class Logger {
public:
    Logger() = default;
    explicit Logger(std::string prefix) : prefix_(std::move(prefix)) {}

    WxLogStream info() const    { return make(wxLOG_Message); }
    WxLogStream verbose() const { return make(wxLOG_Info); }
    WxLogStream warn() const    { return make(wxLOG_Warning); }
    WxLogStream error() const   { return make(wxLOG_Error); }

    // ANET_LOG_DEBUG_PREFIXED などの合成用素材
    const std::string& prefix() const { return prefix_; }
private:
    WxLogStream make(wxLogLevel level) const
    {
        WxLogStream s(level);
        s.stream_ << prefix_;
        return s;
    }
    std::string prefix_;
};
```

- レベルメソッドは`LOG::info()`等と同じレベル対応(`info`=`wxLOG_Message`、`verbose`=`wxLOG_Info`、`warn`=`wxLOG_Warning`、`error`=`wxLOG_Error`)とする。
- 生成した`WxLogStream`には既存のrvalue版`operator<<`群がそのまま効くため、stream記法、一時オブジェクト寿命、デストラクタでのflush、warning以上の即時flushという既存契約は`LOG::`直用と同一になる。
- prefixは構築時に一度だけ確定するimmutable値とし、setterを設けない。

### 2. Env基底がprotectedメンバ変数`log`として保持する

- `SingleDiscreteEnvBase`: prefix = `<lane name>: `(例 `train[37]: `)
- `BatchEnvBase`: prefix = `<BatchEnv name>: `(例 `train: `)

```cpp
class SingleDiscreteEnvBase : public SingleDiscreteEnv {
public:
    explicit SingleDiscreteEnvBase(std::string name);

    const std::string& GetName() const override final { return name_; }
private:
    std::string name_;   // 宣言順初期化のため log より前に置く(logの構築がname_を使う)
protected:
    log::Logger log;     // log.info() << "..." で name prefix付きログを出す
};
```

- メンバ名は意図的に小文字`log`とし、`xxx_`サフィックス規約から外す。ログ出力行`log.info() << ...`を「ログを出す文」として読める疑似namespace的な扱いを優先する。この逸脱は本PRDが規定するLogger保持メンバに限る例外であり、一般のメンバ命名へ適用しない。
- 宣言順初期化に依存するため、`name_`(BatchEnvBaseでは`name_`と`lane_names_`)より後に`log`を宣言し、constructor初期化リストで`log(name_ + ": ")`のようにnameからprefixを構築する。
- 具象Envはconstructor本体からも`log`を使用できる(基底が先に初期化済み)。
- `SingleDiscreteEnvBase`と`BatchEnvBase`を同時に継承するクラスは存在しないため、`log`の曖昧化は起きない。

### 3. 記法規約

- Envメンバ関数内のログは`log.info()`等を使用する。`LOG::info()`等の直用はEnv外(factory、free関数、View)に限る。
- Envメンバ関数内に`LOG::`直用が現れたらprefix忘れを疑う、というレビュー信号として扱う。
- 名前解決は次のとおり成立する。unqualifiedな`log.`はクラスメンバが優先される。`log::Logger`のようなqualified使用はnamespaceへ解決される(qualified lookupはnamespace・型のみ考慮する)。各cppの`namespace LOG = anet::log;`エイリアスとも共存する。

### 4. `ANET_LOG_DEBUG_PREFIXED`マクロを追加する

`log.hpp`のマクロセクション(`ANET_LOG_DEBUG`の直後)へ追加する。

```cpp
// prefix付きdebugログ。log::Logger 型のメンバ/変数 log が見えるスコープ内で使用する。
// (SingleDiscreteEnvBase / BatchEnvBase が protected log を提供する)
#define ANET_LOG_DEBUG_PREFIXED(expr) ANET_LOG_DEBUG(log.prefix() << expr)
```

- guard(デバッガ接続+レベル有効時のみ式評価)とビルド消去(`ANET_ENABLE_DEBUG_LOG=0`でno-op)は、内側の`ANET_LOG_DEBUG`から自動継承する。無効時は`log.prefix()`も`expr`も評価されない。
- `Logger::debug()`メンバ関数は追加しない。debugログの価値(guardによる整形コストゼロ、`__FILE__`/`__LINE__`/`__func__`自動付与、ビルド消去)はマクロでしか維持できない。
- `log`が見えないスコープでの誤用はコンパイルエラーとして検出される。

### 5. prefix書式を`<Env name>: `へ変更する

- `[<Env name>] ` → `<Env name>: `。
- lane name自体の書式(`<name>[<lane index>]`)はPRD 037の契約のまま変更しない(View表示にも使用されるため)。
- 出力例: `12:34:56.789 [I] train[37]: Merged fruits into Rank 5 episode_score_=...`
- あわせて`DropMergeEnv`本文中の`Rank [ 5 ]`表記は`Rank 5`へ掃除する(囲み`[ ]`削減の一環)。
- `LogFormatter`のレベル表記(`[I] `等)とタイムスタンプ書式は変更しない。

### 6. 既存出力行を置換する

| 対象 | 変更 |
|---|---|
| `DropMergeEnv.cpp` 10箇所 | `LOG::xxx() << "[" << GetName() << "] " << ...` → `log.xxx() << ...` |
| `ImageClsEnv.cpp` 3箇所 | 同上 |
| `env.cpp` `DiscreteBatchEnvBase::GetScalar` WARN 1箇所 | `LOG::warn()` → `log.warn()`(本文のクラス名・関数名手書きは維持してよい) |
| `LunarLanderEnv.cpp` `ANET_LOG_DEBUG` 6箇所 | `ANET_LOG_DEBUG("[" << GetName() << "] " << ...)` → `ANET_LOG_DEBUG_PREFIXED(...)` |
| `env.cpp` batch側`ANET_LOG_DEBUG` 4箇所 | `ANET_LOG_DEBUG_PREFIXED`化(prefix新規付与) |

- コメントアウト済みログは対象外とし、触らない。
- 置換によりログレベル・既存本文(`Rank [ n ]`掃除を除く)・flush契約を変更しない。

## ユーザーストーリー

1. Envを実装する開発者として、ログ行に`GetName()`の連結を書かずにprefix付きログを出したい。そうすれば、付け忘れや書式ゆらぎなしに出力元を識別できるログを書ける。
2. ログを読む利用者として、`[train[37]]`のような二重括弧のない簡潔なprefixで出力元を読みたい。そうすれば、視覚ノイズなくTrain/Eval/laneを判別できる。
3. debugログを使う開発者として、`ANET_LOG_DEBUG`のguard・ビルド消去特性を保ったままprefixを付けたい。そうすれば、通常実行の性能を犠牲にせずデバッグ時の出力元識別ができる。
4. batch側の実装を調査する開発者として、`seed=`などのdebugログがどのBatchEnv(train/eval1/EvalPanel)のものか識別したい。そうすれば、複数BatchEnvが並ぶRunでも初期化順や乱数系列を追跡できる。
5. frameworkを保守する開発者として、prefix付きログ部品をEnv専用にせず汎用部品として持ちたい。そうすれば、将来RunnerやAgentへ同じ仕組みを追加コストなしに展開できる。
6. レビューする開発者として、Env内の`LOG::`直用をprefix忘れの兆候として機械的に見つけたい。そうすれば、ログ規約の逸脱を目視の記憶に頼らず検出できる。

## 実装上の決定

- `Logger`は既存`log.hpp`内へのヘッダオンリー追加とし、新規ファイルを作らない。
- `Logger::make`は`WxLogStream::stream_`(public)へprefixを書き込む。これは既存のfree `operator<<`群と同じ依存であり、`WxLogStream`側の変更は行わない。
- prefix文字列はEnv構築時に一度だけ構築する。出力行ごとの追加コストはprefix 1個のstream書き込みのみで、従来の手書き3項連結(`"[" << GetName() << "] "`)より減る。ログレベル無効時にも整形が走る点は既存`LOG::`直用と同一であり、悪化しない。debug系はguardにより無効時コストゼロを維持する。
- `Logger`のレベルメソッドと`prefix()`はconstとし、constメンバ関数からも使用できるようにする。
- `Logger`はimmutable、`WxLogStream`は行ごとの一時オブジェクトであり、共有mutable stateを追加しない。
- unqualifiedな数学関数`log()`の呼び出しは`core`配下に存在しないことを確認済み。以後も数学関数は`std::log`とqualifiedで書く。
- テストのために本体へtest-only APIを追加しない。検証は公開API(`Logger`単体、Env基底を継承したtest double)で行う。

## テスト方針

公開APIと観測可能なログを検証する。private実装や文字列生成helperの実装詳細は検証しない。

### Logger単体

- `Logger("train[0]: ").info()`等が対応レベルの`WxLogStream`を返し、flush前の`stream_.str()`がprefixで始まること。
- `prefix()`が構築時文字列をそのまま返すこと。
- 空prefix(`Logger()`)では本文のみが出力されること。

### Env基底経由

- `SingleDiscreteEnvBase`派生のtest doubleで、`log.info() << "body"`の出力が`<lane name>: body`となること。
- `BatchEnvBase`派生で`<BatchEnv name>: body`となること。
- constメンバ関数から`log`を使用できること(コンパイルで担保)。

### 置換の回帰

- `DropMergeEnv`の代表ログ(Verbose/Info/Errorから各1)が新書式`<Env name>: `を1回だけ含み、既存本文とレベルを維持すること。PRD 037で追加したログ書式テストがあれば新書式へ追従する。
- `ANET_LOG_DEBUG_PREFIXED`は`ANET_LOG_DEBUG`への委譲であり、`ANET_ENABLE_DEBUG_LOG=0`での消去は既存マクロと同一機構のため個別テストを追加しない。使用箇所がコンパイルできることをビルドで担保する。
- 既存の具象Env test・core testが全て通ること。

## 受入条件

1. `log::Logger`が`anet::log`の汎用部品として存在し、`info`/`verbose`/`warn`/`error`/`prefix`を提供する。
2. `SingleDiscreteEnvBase`と`BatchEnvBase`がprotectedメンバ`log`を保持し、prefixはそれぞれ`<lane name>: `/`<BatchEnv name>: `である。
3. Env本体のactiveなログ行に`GetName()`の手書きprefix連結が残っていない。
4. `ANET_LOG_DEBUG_PREFIXED`が定義され、`LunarLanderEnv.cpp`の6箇所と`env.cpp` batch側の4箇所が使用する。
5. prefix書式は`<Env name>: `であり、`[ ]`囲みが撤去されている。`DropMergeEnv`本文の`Rank [ n ]`は`Rank n`になっている。
6. ログレベル、既存本文(`Rank [ n ]`掃除を除く)、warning以上の即時flush契約が維持されている。
7. nameの意味・挙動非依存契約(PRD 037)に変更がない。nameまたはprefixでEnv挙動を分岐するproduction codeが存在しない。
8. `Logger`にsetterがなく、prefixは構築時のみ決定される。
9. Envメンバ関数内の新規ログは`log.`記法を使用し、`LOG::`直用がEnvメンバ関数内に残っていない。
10. 既存テストが全て通る。

## 対象外

- thread-local logging context、structured logging、JSON logging
- `LOG::`free関数群、`WxLogStream`、`FileLogger`、`LogFormatter`の再設計
- Runner、Agent、ViewへのLogger適用(部品の再利用は妨げない)
- lane name書式(`<name>[<lane index>]`)の変更
- `LogFormatter`のレベル表記(`[I] `等)・タイムスタンプ書式の変更
- コメントアウト済みログの整理・復活
- `Logger::debug()`メンバ関数の追加
- PRD 037本文の改訂(履歴として維持。書式規定は本PRDが上書きする)

## 実装時のドキュメント更新

- `docs/design/140_observability.jp.md`: Env text logのprefix書式を`<Env name>: `へ更新し、`Logger`部品、記法規約(Env内=`log.`/Env外=`LOG::`)、`ANET_LOG_DEBUG_PREFIXED`を追記する。
- `docs/design/120_environments.jp.md` 2.4節(Env name): 基底が保持する`log`メンバと、具象Envがprefix手書きをしない契約を追記する。

## 補足

メンバ名`log`は命名規約(`xxx_`サフィックス)からの意図的な逸脱である。ログ出力行の可読性(`log.info() << ...`が「ログを出す文」として読めること)を、メンバ変数であることの明示より優先した。この例外は本PRDのLogger保持メンバに限り、他のメンバ命名へ波及させない。

将来RunnerやAgentで同様のインスタンス識別ログが必要になった場合は、`log::Logger`をメンバ`log`として保持する同じ規約を適用すれば、`ANET_LOG_DEBUG_PREFIXED`を含めそのまま再利用できる。
