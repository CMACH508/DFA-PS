#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <random>

namespace SciLib {
enum MoveInStiefelManifoldPolicy { basic_geod, crank_nicholson };
template <MoveInStiefelManifoldPolicy policy = MoveInStiefelManifoldPolicy::crank_nicholson,
          typename T = Eigen::MatrixXd>
T move_in_stiefel_manifold(const Eigen::EigenBase<T> &_X, const Eigen::EigenBase<T> &_grad,
                           double t = 0.2) {
    const T &X = static_cast<T>(_X), &grad = static_cast<T>(_grad);
    int n = X.rows(), p = X.cols();

    if constexpr (policy == MoveInStiefelManifoldPolicy::basic_geod) {
        T D = grad - X * grad.transpose() * X; //(n,p)
        Eigen::HouseholderQR<Eigen::MatrixXd> qr((T::Identity(n, n) - X * X.transpose()) * D);
        T Q = T(qr.householderQ()).leftCols(p);                        //(n,p)
        T R = qr.matrixQR().topRows(p).triangularView<Eigen::Upper>(); //(p,p)

        T XQ(n, 2 * p);
        XQ << X, Q;

        T XDR(2 * p, 2 * p);
        XDR << X.transpose() * D, -R.transpose(), R, T::Zero(p, p);

        T res = XQ * (t * XDR).exp() * T::Identity(2 * p, p);
        return res;
    } else if constexpr (policy == MoveInStiefelManifoldPolicy::crank_nicholson) {
        T D = -(grad - X * grad.transpose() * X);           //(n,p)
        T Px = T::Identity(n, n) - 0.5 * X * X.transpose(); //(n,n)
        T U(n, 2 * p), V(n, 2 * p);
        T Px_D = Px * D;
        U << Px_D, X;
        V << X, -Px_D;

        T res = X - t * U *
                        (T::Identity(2 * p, 2 * p) + t * 0.5 * V.transpose() * U)
                            .lu()
                            .solve(V.transpose() * X);
        return res;
    }
}

template <typename T> inline T orthogonalize(const Eigen::EigenBase<T> &_X) {
    const T &X = static_cast<T>(_X);
    T res = X * (X.transpose() * X).inverse().llt().matrixL();
    return res;
}

template <typename T> T sample_in_stiefel_manifold(int n, int p) {
    static auto eng = std::default_random_engine();
    static auto dist = std::normal_distribution<double>();

    T X = T(n, p).unaryExpr([](double x) { return dist(eng); });
    X = orthogonalize(X);

    return X;
}

} // namespace SciLib