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

} // namespace anet
