#include "stdafx.h"

#include "ChannelSpatialAttention.hpp"

CAM2_CABMBlockImpl::CAM2_CABMBlockImpl(const CAM2BlockOptions &coption,
                                       const CBAMSpatialBlockOption &soption)
    : channel_attn(register_module("channel", CAM2Block(coption))),
      spatial_attn(register_module("spatial", CBAMSpatialBlock(soption))),
      bn(register_module("bn",
                         torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(coption.channel)))) {}

torch::Tensor CAM2_CABMBlockImpl::forward(const torch::Tensor &x) {

    auto c = channel_attn(x); //(N, C, 1, 1)
    auto s = spatial_attn(x); //(N, 1, H, W)
                              // std::cout << c.sizes() << " " << s.sizes() << std::endl;
    // std::cout << x.sizes() << std::endl;
    return torch::nn::functional::relu(bn(x * (c * s)));
}