#include "stdafx.h"

#include "CalibrationModuleImpl.hpp"
#include "../SciLib/EigenTorchHelper.hpp"

CalibrationModuleImpl::CalibrationModuleImpl(const CalibrationModuleOptions &option)
    : option(option) {

    int input_channel = 16;
    bn_X = register_module("bn_X", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(option.C)));

    conv0 = register_module(
        "conv0", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(option.C, input_channel, {5, 1}).stride({1, 1})));
    bn0 = register_module("bn0",
                          torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(input_channel)));
    //   torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(option.C)));

    int output_channel;
    int output_sz = option.A - 5 + 1; // 5: kernel_size in conv0
    torch::nn::Sequential seq;
    for (size_t i = 0; i < option.channel.size(); ++i) {
        auto stride = option.stride[i];
        output_channel = option.channel[i];

        seq->push_back(CAM2_CABMBlock(CAM2BlockOptions{input_channel, input_channel / 2},
                                      CBAMSpatialBlockOption()));

        seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(input_channel, output_channel, {3, 3}).padding({1, 1})));
        seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(output_channel)));
        seq->push_back(torch::nn::LeakyReLU());

        seq->push_back(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(output_channel, output_channel, {3, 3})
                                  .stride(stride)
                                  .padding({1, 1})));
        seq->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(output_channel)));
        seq->push_back(torch::nn::LeakyReLU());

        input_channel = output_channel;
        output_sz = (output_sz - 1) / stride + 1;
    }
    this->seq = register_module("seq", seq);

    /*pool = register_module(
        "pool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({4, 1})));*/
    dropout = register_module("dropout", torch::nn::Dropout(0.25));
    fmt::print("Output size before final linear size: {}.\n", output_channel * output_sz);
    linear = register_module("linear", torch::nn::Linear(output_channel * output_sz, option.A));
}

torch::Tensor CalibrationModuleImpl::forward(const torch::Tensor &x, const torch::Tensor &w) {
    torch::Tensor X;
    if (w.numel() == 0) {
        X = x;
    } else {
        X = torch::cat({x, w.unsqueeze(1).unsqueeze(3).expand({-1, -1, -1, option.W})},
                       1); //(B,F,A,W), A: assets, F: features
    }
    X = bn_X(X);
    auto y = torch::nn::functional::leaky_relu(bn0(conv0(X))); //(B, First input_channel 16, )
    y = seq->forward(y);                                       //(B, C, ,1), C: last channel
    y = y.flatten(1);
    y = dropout(y);
    y = linear(y);
    y = at::sigmoid(y);
    return y;
}