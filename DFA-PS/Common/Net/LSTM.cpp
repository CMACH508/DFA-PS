#include "stdafx.h"
#include "LSTM.hpp"

LSTM::LSTM(const boost::json::object &config, int Channel, int Length, int Out)
    : lstm(torch::nn::LSTM(
          torch::nn::LSTMOptions(Channel,
                                 static_cast<int>(config.at("intermediate_output").as_int64()))
              .num_layers(static_cast<int>(config.at("layers").as_int64()))
              .batch_first(true)
              .bidirectional(false))),
      linear(torch::nn::Linear(
          Length * static_cast<int>(config.at("intermediate_output").as_int64()), Out)) {
    register_module("weight-lstm", lstm);
    // register_parameter("Weight-A", A);
    // register_parameter("Weight-b", b);
    register_module("weight-linear", linear);

    register_parms(this);
}
torch::Tensor LSTM::forward(const torch::Tensor &x) {
    int batch_sz = static_cast<int>(x.size(0));

    // std::cout << x << std::endl;
    auto x1 = std::get<0>(lstm->forward(x)); //(N, intermediate_output, c)

    // auto x2 = torch::einsum("noc,ko->nkc", {x1, A});
    // std::cout << "x1: " << x1 << std::endl;
    auto y = linear->forward(x1.flatten(1));

    return y;
}

void LSTM::to(torch::Device device) { torch::nn::Module::to(device); }