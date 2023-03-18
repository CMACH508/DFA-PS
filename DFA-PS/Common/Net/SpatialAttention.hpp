#pragma once

#include <torch/torch.h>

struct CBAMSpatialBlockOption {
    int kernel_size_height = 7;
    int kernel_size_weight = 3;
};

struct CBAMSpatialBlockImpl : public torch::nn::Module {
    CBAMSpatialBlockImpl(const CBAMSpatialBlockOption &option = CBAMSpatialBlockOption());

    CBAMSpatialBlockOption option;

    // torch::nn::MaxPool1d p1;
    // torch::nn::Linear linear1, linear2;
    torch::nn::Conv2d conv;

    torch::Tensor forward(const torch::Tensor &x);
};

TORCH_MODULE(CBAMSpatialBlock);