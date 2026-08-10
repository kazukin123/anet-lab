// anet/util.hpp

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <type_traits>
#include "anet/diag.hpp"

namespace anet {

    template <typename Container, typename T>
    inline bool Contains(const Container& container, const T& value) {
        return std::find(container.begin(), container.end(), value) != container.end();
    }

    // ============================================================
    // EmaFilter
    // ============================================================

    /**
     * @brief Exponential Moving Average filter.
     * @tparam T 数値型（float, double, int 等）
     */
    template <typename T>
    class EmaFilter {
        static_assert(std::is_arithmetic<T>::value,
            "EmaFilter<T> requires arithmetic type T.");

    public:
        /**
         * @brief コンストラクタ（浮動小数点型は decay = 0.01、整数型は 1）
         *
         * decay は「新しい値にどれだけ寄せるか」を表す係数 α。
         * α が小さい → 過去が強く残る（平滑）
         * α が大きい → 追従が速い
         */
        EmaFilter() : decay_(DefaultDecay()) { ValidateDecay(decay_); }

        /**
         * @brief decay (α) を明示指定。
         */
        explicit EmaFilter(T decay) : decay_(decay) { ValidateDecay(decay_); }

        explicit EmaFilter(T decay, T value) : decay_(decay) {
            ValidateDecay(decay_);
            Set(value);
        }

        /**
         * @brief decay (α) の変更
         */
        void SetDecay(T decay) {
            ValidateDecay(decay);
            decay_ = decay;
        }

        /**
         * @brief 初期値をセットし、履歴を破棄してこの値から再開始する。
         * @param v 初期化する値
         */
        void Set(T v) {
            // 明示値は既知の初期値として扱い、バイアス補正期間をスキップする。
            value_ = v;
            weight_ = T(1);
            init_ = true;
        }

        /**
         * @brief `ema = x;` は Set(x) と同義（初期値セット）。
         */
        EmaFilter& operator=(T v) {
            Set(v);
            return *this;
        }

        /**
         * @brief 履歴を破棄し、次の Update を初回扱いに戻す。
         * 値自体は保持されるが、統計的には未確定となる。
         */
        void Restart() {
            // 次の更新まで現在の公開値を保持し、新しい重み系列へ戻す。
            if (init_) value_ = Value();
            weight_ = T(0);
            init_ = false;
        }

        /**
         * @brief EMA 更新。初回出力は x と同値。
         * @param x 新しい観測値
         */
        void Update(T x) {
            //ANET_ASSERT(!std::isnan(x));
            //ANET_ASSERT(!std::isinf(x));
            if (std::isnan(x)) return;
            if (std::isinf(x)) return; 

            if (!init_) {
                // ゼロ初期化状態から、最初の観測値と重み和を蓄積する。
                value_ = decay_ * x;
                weight_ = decay_;
                init_ = true;
            } else {
                // 内部値と観測済み重み和を同じ減衰率で更新する。
                value_ += decay_ * (x - value_);
                weight_ += decay_ * (T(1) - weight_);
            }
        }

        /**
         * @brief 現在値を取得。
         */
        T Value() const { return init_ ? value_ / weight_ : value_; }

        /**
         * @brief 暗黙読み取りを許可（代入方向は operator= のみ）。
         */
        operator T() const { return Value(); }

        /**
         * @brief 値が統計的に有効か（初回更新済みか）を返す。
         */
        bool IsInitialized() const { return init_; }

    private:
        static constexpr T DefaultDecay() {
            if constexpr (std::is_integral_v<T>) {
                return T(1);
            } else {
                return T(0.01);
            }
        }

        static void ValidateDecay(T decay) {
            if (!std::isfinite(decay) || decay <= T(0) || decay > T(1)) {
                ANET_SYSTEM_ERROR(
                    "EmaFilter decay is invalid: decay=" << decay
                    << " expected=finite value in (0, 1]");
            }
        }

        T decay_;        ///< α：新しい値に寄せる割合
        bool init_ = false; ///< 初回更新済みか
        T value_{};      ///< ゼロ初期化から蓄積したEMA内部値
        T weight_{};     ///< 観測済みサンプルの重み和
    };


    // ============================================================
    // OrderedMap Iterator
    // ============================================================

    template<typename K, typename V>
    class OrderedMap;

    template<typename K, typename V>
    class OrderedMapIterator
    {
    public:
        using value_type = std::pair<const K&, const V&>;

        OrderedMapIterator(const OrderedMap<K, V>* owner, size_t pos)
            : owner_(owner), pos_(pos)
        {
        }

        bool operator==(const OrderedMapIterator& rhs) const {
            return owner_ == rhs.owner_ && pos_ == rhs.pos_;
        }

        bool operator!=(const OrderedMapIterator& rhs) const {
            return !(*this == rhs);
        }

        // ++it
        OrderedMapIterator& operator++() {
            ++pos_;
            return *this;
        }

        // it++
        OrderedMapIterator operator++(int) {
            OrderedMapIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        value_type operator*() const {
            const K& key = owner_->Order()[pos_];
            const V& val = owner_->Raw().at(key);
            return { key, val };
        }

    private:
        const OrderedMap<K, V>* owner_;
        size_t pos_;
    };


    // ============================================================
    // OrderedMap
    // ============================================================
    template<typename K, typename V>
    class OrderedMap
    {
    public:
        using iterator = OrderedMapIterator<K, V>;
        using const_iterator = OrderedMapIterator<K, V>;

        // ------------------------------------------
        // 値セット
        // ------------------------------------------
        void Set(const K& key, const V& value)
        {
            auto it = kv_.find(key);
            if (it == kv_.end()) {
                order_.push_back(key);
            }
            kv_[key] = value;
        }

        // ------------------------------------------
        // 存在チェック
        // ------------------------------------------
        bool Has(const K& key) const {
            return kv_.find(key) != kv_.end();
        }

        // ------------------------------------------
        // 値取得
        // ------------------------------------------
        const V& Get(const K& key) const {
            return kv_.at(key);
        }

        const V& GetOr(const K& key, const V& defaultValue) const
        {
            auto it = kv_.find(key);
            return (it == kv_.end()) ? defaultValue : it->second;
        }

        // ------------------------------------------
        // STL互換 find()
        // ------------------------------------------
        iterator find(const K& key)
        {
            if (!Has(key)) return end();

            // order_ 内の index を探す（定義順としての位置）
            for (size_t i = 0; i < order_.size(); ++i) {
                if (order_[i] == key) {
                    return iterator(this, i);
                }
            }
            return end(); // 理論的にはありえない
        }

        const_iterator find(const K& key) const
        {
            if (!Has(key)) return end();

            for (size_t i = 0; i < order_.size(); ++i) {
                if (order_[i] == key) {
                    return const_iterator(this, i);
                }
            }
            return end();
        }

        // ------------------------------------------
        // 削除
        // ------------------------------------------
        void Erase(const K& key)
        {
            if (!Has(key)) return;

            kv_.erase(key);

            for (auto it = order_.begin(); it != order_.end(); ++it) {
                if (*it == key) {
                    order_.erase(it);
                    break;
                }
            }
        }

        // ------------------------------------------
        // 基本操作
        // ------------------------------------------
        void Clear()
        {
            kv_.clear();
            order_.clear();
        }

        size_t Size() const { return kv_.size(); }
        bool Empty() const { return kv_.empty(); }

        // ------------------------------------------
        // iterator API
        // ------------------------------------------
        iterator begin() { return iterator(this, 0); }
        iterator end() { return iterator(this, order_.size()); }

        const_iterator begin() const { return const_iterator(this, 0); }
        const_iterator end()   const { return const_iterator(this, order_.size()); }

        // ------------------------------------------
        // 補助
        // ------------------------------------------
        const std::vector<K>& Order() const { return order_; }
        const std::unordered_map<K, V>& Raw() const { return kv_; }

    private:
        std::unordered_map<K, V> kv_;  // 高速検索
        std::vector<K> order_;        // 定義順保持
    };


    static std::unordered_map<std::string, std::string>
        MakeReverseMap(const std::unordered_map<std::string, std::string>& m)
    {
        std::unordered_map<std::string, std::string> rev;
        rev.reserve(m.size());
        for (auto& kv : m) {
            rev.emplace(kv.second, kv.first);
        }
        return rev;
    }

} // namespace anet
