#include "stdafx.h"

#include "ResNet.hpp"

ResBasic1dBlockImpl::ResBasic1dBlockImpl(int channel, int out_channel, int stride) {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv1d(
            torch::nn::Conv1dOptions(channel, out_channel, 3).padding({1}).stride(stride)));
    bn1 =
        register_module("bn1", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channel)));

    conv2 = register_module(
        "conv2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channel, out_channel, 3).padding({1})));
    bn2 =
        register_module("bn2", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channel)));

    if ((stride != 1) || (channel != out_channel)) {
        downsample = register_module(
            "downsample",
            torch::nn::Sequential{
                torch::nn::Conv1d(torch::nn::Conv1dOptions(channel, out_channel, 1).stride(stride)),
                torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out_channel))});
    }
}

torch::Tensor ResBasic1dBlockImpl::forward(const torch::Tensor &input) {
    // residual = input;
    auto y = torch::nn::functional::relu(bn1(conv1(input)));
    y = torch::nn::functional::relu(bn2(conv2(y)));
    auto residual = input;
    if (!downsample.is_empty()) {
        residual = downsample->forward(residual);
    }
    return torch::nn::functional::relu(y + residual);
}