#pragma once

#include "../Net/ResNet.hpp"
struct RatioModuleOption {
    int C = 4, W = 3;
    std::vector<int> channel{8, 16, 16}, stride{2, 2, 1};
    std::vector<int> linear_seq_size{6, 4, 1};
    int extra_feature_size = 3; // default 3: (min, max, sum);
};
// Input (N, C, W), C: market_feature_size.
struct RatioModuleImpl : public torch::nn ::Module {
    RatioModuleImpl(const RatioModuleOption &option);

    RatioModuleOption option;

    torch::nn::Sequential seq = nullptr;
    torch::nn::AdaptiveAvgPool1d pool = nullptr;
    torch::nn::Linear linear1 = nullptr, linear2 = nullptr;

    torch::nn::BatchNorm1d bn2 = nullptr;
    torch::nn::Sequential linear_seq = nullptr;

    // Currently extra feature is ((N, topk), (N, topk))
    torch::Tensor forward(const torch::Tensor &input, const std::vector<torch::Tensor> &extra);
};
TORCH_MODULE(RatioModule);
