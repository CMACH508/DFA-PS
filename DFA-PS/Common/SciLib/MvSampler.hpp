#pragma once

#include <iostream>
#include <random>

#include <Eigen/Core>

namespace SciLib {
template <typename EngineType, typename DistType, typename ValT = double>
auto homo_rand(EngineType &engine, DistType &dist, int n, int d = 1) {

    Eigen::Matrix<ValT, Eigen::Dynamic, Eigen::Dynamic> res(n, d);
    for (auto row : res.rowwise()) {
        for (auto &ele : row) {
            ele = dist(engine);
        };
    };
    return res;
};
// Random number
template <typename EngineType, typename ValT = double>
auto rand(EngineType &engine, int n, int d = 1, ValT lb = 0, ValT ub = 1) {
    static std::uniform_real_distribution<ValT> uniformdist(lb, ub);
    return homo_rand<EngineType, std::uniform_real_distribution<ValT>, ValT>(engine, uniformdist, n,
                                                                             d);
};

// Normal distribution
template <typename EngineType, typename ValT = double>
auto randn(EngineType &engine, int n, int d = 1, ValT mu = 0, ValT sigma = 1) {
    static std::normal_distribution<ValT> normdist(mu, sigma);

    return homo_rand<EngineType, std::normal_distribution<ValT>, ValT>(engine, normdist, n, d);
};

// Sigma: lower triangle
template <typename DerivedA, typename DerivedB, typename EngineType>
auto mvnrnd(EngineType &engine, const Eigen::EigenBase<DerivedA> &mu,
            const Eigen::EigenBase<DerivedB> &SigmaL, int n = 1) {

    using T = typename DerivedA::Scalar;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res = randn(engine, mu.size(), n);
    // cout << "res " << res << endl;
    res = static_cast<DerivedB>(SigmaL) * res;
    res = res.colwise() + static_cast<DerivedA>(mu);
    res.transposeInPlace();
    return res;
};

// Multivariate t distribution
template <typename DerivedA, typename DerivedB, typename EngineType>
auto mvtrnd(EngineType &engine, const Eigen::EigenBase<DerivedA> &mu,
            const Eigen::EigenBase<DerivedB> &SigmaL, int n = 1, int df = 1) {

    using ValT = typename DerivedA::Scalar;
    int d = mu.size();
    DerivedA zero_mu;
    zero_mu.resizeLike(mu);
    zero_mu.setZero();

    Eigen::Matrix<ValT, Eigen::Dynamic, Eigen::Dynamic> res = mvnrnd(engine, zero_mu, SigmaL, n);

    //
    std::chi_squared_distribution<ValT> chi_dist{df};
    for (auto row : res.rowwise()) {
        row = row * ((ValT)df / chi_dist(engine));
    };
    res = res.rowwise() + static_cast<DerivedA>(mu).transpose();
    return res;
};

}; // namespace SciStaLib