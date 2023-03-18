#include "stdafx.h"
#include "WeightFunctionModule.hpp"
#include "ModelSelectionHelper.hpp"

LSTMWeightFunctionModule::LSTMWeightFunctionModule(const boost::json::object &config, int window,
                                                   int in_dim, int k)
    : WeightFunctionModuleBase(window, in_dim, k),
      lstm(
          torch::nn::LSTM(torch::nn::LSTMOptions(
                              in_dim, static_cast<int>(config.at("intermediate_output").as_int64()))
                              .num_layers(static_cast<int>(config.at("layers").as_int64()))
                              .batch_first(true)
                              .bidirectional(false))),
      linear(torch::nn::Linear(
          window * static_cast<int>(config.at("intermediate_output").as_int64()), k)) {
    register_module("weight-lstm", lstm);
    // register_parameter("Weight-A", A);
    // register_parameter("Weight-b", b);
    register_module("weight-linear", linear);
};
torch::Tensor LSTMWeightFunctionModule::forward(const torch::Tensor &x) {
    int batch_sz = static_cast<int>(x.size(0));

    // std::cout << x << std::endl;
    auto x1 = std::get<0>(lstm->forward(x)); //(N, intermediate_output, c)

    // auto x2 = torch::einsum("noc,ko->nkc", {x1, A});
    // std::cout << "x1: " << x1 << std::endl;
    auto y = linear->forward(x1.flatten(1));

    return y;
};

void LSTMWeightFunctionModule::copy_parms(std::shared_ptr<WeightFunctionModuleBase> &new_net,
                                          const std::vector<int> &selected_index) {
    NET_CASTING_CHECKER(WeightFunctionModuleBase, LSTMWeightFunctionModule);

    copy_lstm_helper(lstm, new_net_casted->lstm, selected_index, {});
}