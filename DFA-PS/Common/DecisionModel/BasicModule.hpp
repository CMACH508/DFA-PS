#pragma once

// #include <torch/torch.h>
#include "../Net/ChannelAttention.hpp"

struct BasicModuleOptions {
    int C, W, A;
    int lstm_num_layer;
    int lstm_out_channel;
};

// Input : (N, C, W), C for hidden factor size.
// Output : (N, A)
struct BasicModuleImpl : public torch::nn::Module {
    BasicModuleImpl(const BasicModuleOptions &option);

    BasicModuleOptions option;

    // SEBlock se = nullptr;
    // torch::nn::BatchNorm1d bn1 = nullptr;
    torch::nn::LSTM lstm = nullptr;
    torch::nn::BatchNorm1d bn2 = nullptr;
    torch::nn::Dropout dropout = nullptr;
    torch::nn::Linear linear = nullptr;

    torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(BasicModule);