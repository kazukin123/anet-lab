#pragma once

#include <type_traits>

/**
 * @file ema_filter.h
 * @brief Exponential Moving Average (EMA) filter.
 *
 * 差分型 EMA を用いた滑らかな逐次平均フィルタ。
 * value ← value + α * (x - value)
 *
 * - `Set(x)` / `operator=(x)` は「この値から再開始」
 * - `Update(x)` は EMA 更新
 * - `Restart()` は履歴のみ破棄し、次の Update を初回扱いに戻す
 *
 * @code
 * #include "anet/ema_filter.h"
 *
 * int main() {
 *   anet::EmaFilter<float> ema;   // decay = 0.01
 *
 *   ema = 10.0f;       // 初期値セット
 *   ema.Update(12.0f);
 *   ema.Update(11.0f);
 *
 *   float a = ema;      // 暗黙読み取り
 *
 *   ema.Restart();      // 履歴破棄（値は残る）
 *   ema.Update(100.0f); // 再初回扱い → 100 から再スタート
 *
 *   float b = ema.Value();
 * }
 * @endcode
 */

namespace anet {

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
         * @brief コンストラクタ（decay = 0.01）
         *
         * decay は「新しい値にどれだけ寄せるか」を表す係数 α。
         * α が小さい → 過去が強く残る（平滑）
         * α が大きい → 追従が速い
         */
        EmaFilter() : decay_(T(0.01)) {}

        /**
         * @brief decay (α) を明示指定。
         */
        explicit EmaFilter(T decay) : decay_(decay) {}

        /**
         * @brief decay (α) の変更
         */
        void SetDecay(T decay) { decay_ = decay; }

        /**
         * @brief 初期値をセットし、履歴を破棄してこの値から再開始する。
         * @param v 初期化する値
         */
        void Set(T v) {
            value_ = v;
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
            init_ = false;
        }

        /**
         * @brief EMA 更新。初回のみ value = x と同義。
         * @param x 新しい観測値
         */
        void Update(T x) {
            if (!init_) {
                value_ = x;
                init_ = true;
            }
            else {
                value_ += decay_ * (x - value_);
            }
        }

        /**
         * @brief 現在値を取得。
         */
        T Value() const { return value_; }

        /**
         * @brief 暗黙読み取りを許可（代入方向は operator= のみ）。
         */
        operator T() const { return value_; }

        /**
         * @brief 値が統計的に有効か（初回更新済みか）を返す。
         */
        bool IsInitialized() const { return init_; }

    private:
        T decay_;        ///< α：新しい値に寄せる割合
        bool init_ = false; ///< 初回更新済みか
        T value_{};      ///< 現在値。init_ == false の間は意味を持たない
    };

    template <class K, class V>
    class LinkedHashMap {
    public:
        explicit LinkedHashMap(bool access_order = false) : access_order_(access_order) {}

        std::size_t size() const { return map_.size(); }
        bool empty() const { return map_.empty(); }
        void clear() {
            map_.clear();
            order_.clear();
        }

        // ==== iterator ====
        struct iterator {
            using list_iterator = typename std::list<K>::iterator;
            iterator(list_iterator it, LinkedHashMap* owner) : it_(it), owner_(owner) {}

            iterator& operator++() { ++it_; return *this; }
            bool operator==(const iterator& rhs) const { return it_ == rhs.it_; }
            bool operator!=(const iterator& rhs) const { return it_ != rhs.it_; }

            std::pair<const K&, V&> operator*() const {
                auto mit = owner_->map_.find(*it_);
                return { mit->first, mit->second.first };
            }

            list_iterator it_;
            LinkedHashMap* owner_;
        };

        struct const_iterator {
            using list_iterator = typename std::list<K>::const_iterator;
            const_iterator(list_iterator it, const LinkedHashMap* owner) : it_(it), owner_(owner) {}

            const_iterator& operator++() { ++it_; return *this; }
            bool operator==(const const_iterator& rhs) const { return it_ == rhs.it_; }
            bool operator!=(const const_iterator& rhs) const { return it_ != rhs.it_; }

            std::pair<const K&, const V&> operator*() const {
                auto mit = owner_->map_.find(*it_);
                return { mit->first, mit->second.first };
            }

            list_iterator it_;
            const LinkedHashMap* owner_;
        };

        iterator begin() { return iterator(order_.begin(), this); }
        iterator end() { return iterator(order_.end(), this); }

        const_iterator begin() const { return const_iterator(order_.cbegin(), this); }
        const_iterator end()   const { return const_iterator(order_.cend(), this); }

        const_iterator cbegin() const { return const_iterator(order_.cbegin(), this); }
        const_iterator cend()   const { return const_iterator(order_.cend(), this); }

        iterator find(const K& key) {
            auto it = map_.find(key);
            if (it == map_.end()) return end();
            return iterator(it->second.second, this);
        }

        const_iterator find(const K& key) const {
            auto it = map_.find(key);
            if (it == map_.end()) return end();
            return const_iterator(it->second.second, this);
        }

        bool contains(const K& key) const {
            return map_.find(key) != map_.end();
        }

        V& operator[](const K& key) {
            auto it = map_.find(key);
            if (it == map_.end()) {
                order_.push_back(key);
                auto list_it = std::prev(order_.end());
                auto inserted = map_.emplace(key, MapValue(V{}, list_it));
                return inserted.first->second.first;
            }
            if (access_order_) Touch(it);
            return it->second.first;
        }

        void put(const K& key, const V& value) {
            auto it = map_.find(key);
            if (it != map_.end()) {
                it->second.first = value;
                if (access_order_) Touch(it);
                return;
            }
            order_.push_back(key);
            map_.emplace(key, MapValue(value, std::prev(order_.end())));
        }

        V* get(const K& key) {
            auto it = map_.find(key);
            if (it == map_.end()) return nullptr;
            if (access_order_) Touch(it);
            return &it->second.first;
        }

        bool erase(const K& key) {
            auto it = map_.find(key);
            if (it == map_.end()) return false;
            order_.erase(it->second.second);
            map_.erase(it);
            return true;
        }

    private:
        using ListIter = typename std::list<K>::iterator;
        using MapValue = std::pair<V, ListIter>;
        using Map = std::unordered_map<K, MapValue>;

        void Touch(typename Map::iterator it) {
            order_.splice(order_.end(), order_, it->second.second);
        }

        bool access_order_;
        std::list<K> order_;
        Map map_;
    };
} // namespace anet
