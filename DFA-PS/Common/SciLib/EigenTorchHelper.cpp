#include "stdafx.h"
#include "EigenTorchHelper.hpp"

namespace SciLib {

void write_1d2d_tensor(const torch::Tensor &t, const std::string &filename) {
    auto szs = t.sizes();
    if (szs.size() == 1) {
        std::ofstream file(filename);
        for (int i = 0; i < szs[0]; ++i) {
            file << t[i].item<float>() << std::endl;
        }
    } else if (szs.size() == 2) {
        std::ofstream file(filename);
        for (int i = 0; i < szs[0]; ++i) {
            for (int j = 0; j < szs[1] - 1; ++j) {
                file << t[i][j].item<float>() << ",";
            }
            file << t[i][szs[1] - 1].item<float>() << std::endl;
        }
    } else {
        fmt::print(fmt::fg(fmt::color::red),
                   "(write_1d2d_tensor): The dimension is larger than 3 which is not supported.\n");
        std::terminate();
    }
}
} // namespace SciLib