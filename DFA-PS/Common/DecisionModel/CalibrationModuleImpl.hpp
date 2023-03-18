#pragma once

//#include <torch/torch.h>

#include "../Net/ChannelSpatialAttention.hpp"

struct CalibrationModuleOptions {
    int C, A, W;
    std::vector<int> channel, stride;
};

// Input (N, C, A, W), C: asset size, A: asset features.
// + Input (N, A)
struct CalibrationModuleImpl : public torch::nn::Module {
    CalibrationModuleImpl(const CalibrationModuleOptions &option);

    CalibrationModuleOptions option;

    torch::nn::Conv2d conv0 = nullptr;
    torch::nn::BatchNorm2d bn_X = nullptr,bn0 = nullptr;

    torch::nn::Sequential seq = nullptr;

    torch::nn::AdaptiveAvgPool2d pool = nullptr;
    torch::nn::Dropout dropout = nullptr;
    torch::nn::Linear linear = nullptr;

    // Asset feature input, basis weight.
    torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &_w);
};
TORCH_MODULE(CalibrationModule);