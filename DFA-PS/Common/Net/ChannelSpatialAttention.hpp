#pragma once

#include "ChannelAttention.hpp"
#include "SpatialAttention.hpp"

struct CAM2_CABMBlockImpl : public torch::nn::Module {
    CAM2_CABMBlockImpl(const CAM2BlockOptions &coption, const CBAMSpatialBlockOption &soption);

    CAM2Block channel_attn;
    CBAMSpatialBlock spatial_attn;

    torch::nn::BatchNorm2d bn;
    // torch::nn::ReLU relu;
    torch::Tensor forward(const torch::Tensor &x);
};
TORCH_MODULE(CAM2_CABMBlock);