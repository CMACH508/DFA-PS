#pragma once
#include <Eigen/Dense>

namespace SciLib {
template <typename T> inline auto cov(const Eigen::EigenBase<T> &_mat) {

    typename T::PlainObject cov, centered;
    const T &mat(_mat.derived());

    centered = mat.rowwise() - mat.colwise().mean();
    cov = (centered.adjoint() * centered) / static_cast<double>(mat.rows() - 1);

    return cov;
}

template <typename T> auto corrcov(const Eigen::EigenBase<T> &_cov) {
    typename T::PlainObject C, sd;
    const T &cov(_cov.derived());

    C = cov;
    sd = C.diagonal().array().sqrt().matrix();

    const int N = C.cols();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            C.coeffRef(i, j) = C.coeff(i, j) / (sd.coeff(i, 0) * sd.coeff(j, 0));
            if (j != i)
                C.coeffRef(j, i) = C.coeff(i, j);
        }
    }

    return C;
}
template <typename T> inline auto corr(const Eigen::EigenBase<T> &_mat) {
    auto C = cov(_mat);
    C = corrcov(C);
    return C;
}

// Row as sample.
template <typename T1, typename T2>
inline double NLL_gaussian(const Eigen::EigenBase<T1> &_X, const Eigen::EigenBase<T2> &_S) {
    const T1 &X(_X.derived());
    const T2 &S(_S.derived());

    double N = X.rows(), d = X.cols();
    double res = 0;
    const T2 iS = S.inverse();
    for (int i = 0; i < N; ++i) {
        res += X.row(i) * iS * X.row(i).transpose();
    }
    return 0.5 * ((std::log(S.determinant()) + d * std::log(2 * M_PI)) * N + res);
}
// Assuming S is diagonal and is a col vector.
template <typename T1, typename T2>
inline double NLL_gaussian_diag(const Eigen::EigenBase<T1> &_X, const Eigen::EigenBase<T2> &_S) {
    const T1 &X(_X.derived());
    const T2 &S(_S.derived());

    double N = X.rows(), d = X.cols();
    const T2 iS = 1.0 / S.array();

    return 0.5 * ((S.array().log().sum() + d * std::log(2 * M_PI)) * N +
                  (X.array().pow(2).matrix() * iS).sum());
}

// mu is a row vector.
template <typename T1, typename T2, typename T3>
auto mvnpdf(const Eigen::MatrixBase<T1> &_X, const Eigen::MatrixBase<T2> &_mu,
            const Eigen::MatrixBase<T3> &_S) {
    const T1 &X(_X.derived());
    const T2 &mu(_mu.derived());
    const T3 &S(_S.derived());

    T1 Xmu = X.rowwise() - mu.row(0);
    T2 iS = S.inverse();

    int N = X.rows();
    double d = X.cols();
    T1 res(N, 1);
    for (int i = 0; i < N; ++i) {
        res.coeffRef(i, 0) = Xmu.row(i) * iS * Xmu.row(i).transpose();
    }

    res = (-0.5 * res.array()).exp() * std::pow(2 * M_PI, -d / 2) * std::pow(S.determinant(), -0.5);
    return res;
}
} // namespace SciLib