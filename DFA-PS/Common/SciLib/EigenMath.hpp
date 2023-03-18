#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <numeric>

namespace SciLib {
template <typename T> auto expm(const Eigen::MatrixBase<T> &_m) {
    const T &m(_m.derived());
    typename T::PlainObject res;
    res = m.exp();
    return res;
}

#define UNWRAP(...) __VA_ARGS__
#define SCILIB_EIGEN_MATH_CUMOP(name, op)                                                          \
    template <int axis, typename T> auto cum##name(const Eigen::EigenBase<T> &_m) {                \
        const T &m(_m.derived());                                                                  \
        using ValT = T::Scalar;                                                                    \
                                                                                                   \
        auto M = _m.rows(), N = _m.cols();                                                         \
        typename T::PlainObject res(M, N);                                                         \
        if constexpr (axis == 0) {                                                                 \
            for (decltype(M) c = 0; c < N; c++) {                                                  \
                std::partial_sum(m.col(c).begin(), m.col(c).end(), res.col(c).begin(), op);        \
            }                                                                                      \
        } else {                                                                                   \
            for (decltype(M) r = 0; r < M; r++) {                                                  \
                std::partial_sum(m.row(r).begin(), m.row(r).end(), res.row(r).begin(), op);        \
            }                                                                                      \
        }                                                                                          \
        return res;                                                                                \
    }

SCILIB_EIGEN_MATH_CUMOP(prod, std::multiplies<ValT>{});
SCILIB_EIGEN_MATH_CUMOP(sum, std::plus<ValT>{});
SCILIB_EIGEN_MATH_CUMOP(max, UNWRAP([](const ValT &l, const ValT &r) { return std::max(l, r); }));
SCILIB_EIGEN_MATH_CUMOP(min, UNWRAP([](const ValT &l, const ValT &r) { return std::min(l, r); }));
#undef SCILIB_EIGEN_MATH_CUMOP
#undef UNWRAP

} // namespace SciLib