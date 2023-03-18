#include "stdafx.h"

#include "SpatialAttention.hpp"

CBAMSpatialBlockImpl::CBAMSpatialBlockImpl(const CBAMSpatialBlockOption &option)
    : option(option),
      conv(register_module(
          "conv",
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(2, 1, {option.kernel_size_height, option.kernel_size_weight})
                  .padding({(option.kernel_size_height - 1) / 2,
                            (option.kernel_size_weight - 1) / 2})))) {}

torch::Tensor CBAMSpatialBlockImpl::forward(const torch::Tensor &x) {
    auto y = torch::cat({x.mean(1, true), std::get<0>(x.max(1, true))}, 1);
    y = at::sigmoid(conv(y));
    return y;
}
