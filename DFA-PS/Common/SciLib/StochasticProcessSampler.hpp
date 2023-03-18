#pragma once

namespace SciLib {
// Return (N, n_path) matrix.
// The first row is all x0.
template <typename MatType, typename EngineType>
MatType sample_geometric_brawnian_motion(EngineType &engine, double x0, double dt, double mu,
                                         double vol, int N, int n_path = 10000) {
    MatType res(N, n_path);

    return res;
};
} // namespace SciLib