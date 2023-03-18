#pragma once
#include "NetBase.hpp"
// Input: (Batch, Length, Channel)
// Output: (Batch, Out)
struct LSTM : NetBase, torch::nn::Module {

    LSTM(const boost::json::object &config, int Channel, int Length, int Out);

    torch::Tensor forward(const torch::Tensor &x) override;
    void to(torch::Device device) override;

    torch::nn::LSTM lstm;
    torch::nn::Linear linear;

    // int layers, intermediate_output;

    // Final layer to generate weights.
    // torch::Tensor A, b;
};