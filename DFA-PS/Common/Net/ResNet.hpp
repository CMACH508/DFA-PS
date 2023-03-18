#pragma once
#include <torch/torch.h>

struct ResBasic1dBlockImpl : public torch::nn::Module {
    ResBasic1dBlockImpl(int channel, int out_channel, int stride);

    //    torch::nn::Sequential seq;
    torch::nn::Conv1d conv1 = nullptr, conv2 = nullptr;
    torch::nn::BatchNorm1d bn1 = nullptr, bn2 = nullptr;

    torch::nn::Sequential downsample = nullptr;

    torch::Tensor forward(const torch::Tensor &input);
};

TORCH_MODULE(ResBasic1dBlock);
