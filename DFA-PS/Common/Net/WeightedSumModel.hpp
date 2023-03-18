#pragma once
#include "BasisFunctionModule.hpp"
#include "NetBase.hpp"
#include "WeightFunctionModule.hpp"

// Normalization of weights happens at this module, not WeightFunctionModule.
struct WeightedSumModel : NetBase {
    WeightedSumModel(const boost::json::object &config, int window, int in_dim, int out_dim);

    // void finalize() override;
    torch::Tensor forward(const torch::Tensor &x) override;
    void to(torch::Device device) override;

    int k;

    std::shared_ptr<BasisFunctionModuleBase> basis;
    std::shared_ptr<WeightFunctionModuleBase> weight;

    void copy_parms(std::shared_ptr<NetBase> &new_net,
                    const std::vector<int> &selected_index) override;
};