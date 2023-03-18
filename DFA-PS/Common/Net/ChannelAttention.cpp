#include "stdafx.h"

#include "ChannelAttention.hpp"

SEBlockImpl::SEBlockImpl(const SEBlockOptions &option)
    : option(option), linear1(register_module(
                          "linear1", torch::nn::Linear(option.channel, option.reduced_channel))),
      linear2(
          register_module("linear2", torch::nn::Linear(option.reduced_channel, option.channel))) {}

torch::Tensor SEBlockImpl::forward(const torch::Tensor &x) {
    auto [y, _] = x.flatten(2).max(2);                 //(N,C)
    y = torch::nn::functional::leaky_relu(linear1(y)); //(N,C_o);
    y = at::sigmoid(linear2(y));                       //(N, C)
    if (x.dim() == 3) {
        y = y.view({x.size(0), x.size(1), 1});
    } else {
        y = y.view({x.size(0), x.size(1), 1, 1});
    }
    return x * y;
}

CAM2BlockImpl::CAM2BlockImpl(const CAM2BlockOptions &option)
    : option(option),
      conv(register_module(
          "conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(option.channel, option.channel, 2)))),
      linear1(
          register_module("linear1", torch::nn::Linear(option.channel, option.reduced_channel))),
      linear2(
          register_module("linear2", torch::nn::Linear(option.reduced_channel, option.channel))) {}

torch::Tensor CAM2BlockImpl::forward(const torch::Tensor &x) {
    auto [m, _] = x.flatten(2).max(2, true);  //(N,C,1)
    auto s = x.flatten(2).std(2, true, true); //(N,C,1)
    auto y = torch::cat({m, s}, 2);
    y = conv(y).flatten(1);
    y = torch::nn::functional::leaky_relu(linear1(y)); //(N,C_o);
    y = at::sigmoid(linear2(y));                       //(N, C)
    if (x.dim() == 3) {
        y = y.view({x.size(0), x.size(1), 1});
    } else {
        y = y.view({x.size(0), x.size(1), 1, 1});
    }
    return y;
}