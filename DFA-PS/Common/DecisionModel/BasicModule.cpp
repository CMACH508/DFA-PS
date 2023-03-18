#include "stdafx.h"

#include "BasicModule.hpp"

BasicModuleImpl::BasicModuleImpl(const BasicModuleOptions &option) : option(option) {
    // se = register_module("SE", SEBlock(SEBlockOptions{option.C, option.C / 2}));
    lstm = register_module("lstm",
                           torch::nn::LSTM(torch::nn::LSTMOptions(option.C, option.lstm_out_channel)
                                               .num_layers(option.lstm_num_layer)
                                               .batch_first(true)
                                               .dropout(0.5)));

    // bn1 = register_module("bn1",
    // torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(option.C)));
    bn2 = register_module(
        "bn2", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(option.lstm_out_channel)));
    dropout = register_module("dropout", torch::nn::Dropout(0.5));
    linear = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(
                                           option.W * option.lstm_out_channel, option.A)));
}

torch::Tensor BasicModuleImpl::forward(const torch::Tensor &input) {
    auto [y, _] = lstm(input); //(N,W,C_o)
    y = torch::nn::functional::relu(bn2(y.transpose(1, 2))).flatten(1);
    // y = dropout(bn2(y.transpose(1, 2)).flatten(1));
    // y = dropout(y);
    y = linear(y);
    return y;
}