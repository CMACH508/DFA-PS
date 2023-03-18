#pragma once
#include <Eigen/Dense>

namespace SciLib {
// Minimum MSE of Low Randk Approximation of matrix A by rank k.
template <typename T> inline double LRA_min_mse(const T &A, int k) {
    int d = std::min(A.rows(), A.cols());
    if (k >= d) {
        return 0;
    } else {
        Eigen::VectorXd v = A.bdcSvd().singularValues();
        std::sort(v.begin(), v.end(), std::greater<double>());
        return v.bottomRows(d - k).array().pow(2).sum() / (A.size());
    }
}
} // namespace SciLib