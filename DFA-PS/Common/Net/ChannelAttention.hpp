#pragma once

#include <torch/torch.h>

struct SEBlockOptions {
    int channel;
    int reduced_channel;
};

// Can be 4D: Input (N, C, H, W), Output (N, C, 1, 1)
// Or 3d: Input (N, C, L), Output (N, C, 1)
struct SEBlockImpl : public torch::nn::Module {
    SEBlockImpl(const SEBlockOptions &option);

    SEBlockOptions option;

    // torch::nn::MaxPool1d p1;
    torch::nn::Linear linear1, linear2;

    torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(SEBlock);

struct CAM2BlockOptions {
    int channel;
    int reduced_channel;
};

struct CAM2BlockImpl : public torch::nn::Module {
    CAM2BlockImpl(const CAM2BlockOptions &option);

    CAM2BlockOptions option;

    torch::nn::Conv1d conv;
    torch::nn::Linear linear1, linear2;

    torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(CAM2Block);