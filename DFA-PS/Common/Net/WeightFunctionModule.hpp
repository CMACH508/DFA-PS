#pragma once
#include <boost/json.hpp>
#include <torch/torch.h>

// Input: (N, L, C)
// Output:(N, k)
struct WeightFunctionModuleBase : public torch::nn::Module {
    // WeightFunctionModuleBase(){};
    int window, in_dim, k;
    WeightFunctionModuleBase(int window, int in_dim, int k)
        : window(window), in_dim(in_dim), k(k){};

    virtual torch::Tensor forward(const torch::Tensor &x) = 0;
    virtual void copy_parms(std::shared_ptr<struct WeightFunctionModuleBase> &new_net,
                            const std::vector<int> &selected_index){};

    // int k;
};

struct LSTMWeightFunctionModule : WeightFunctionModuleBase {

    LSTMWeightFunctionModule(const boost::json::object &config, int window, int in_dim, int k);

    torch::Tensor forward(const torch::Tensor &x);

    torch::nn::LSTM lstm;
    torch::nn::Linear linear;

    // int layers, intermediate_output;

    // Final layer to generate weights.
    // torch::Tensor A, b;
    virtual void copy_parms(std::shared_ptr<struct WeightFunctionModuleBase> &new_net,
                            const std::vector<int> &selected_index) override;
};

struct WeightFunctionModuleFactory {
    std::shared_ptr<WeightFunctionModuleBase> operator()(const boost::json::object &config,
                                                         int window, int in_dim, int k) {
        std::shared_ptr<WeightFunctionModuleBase> weight;
        auto weight_type = config.at("type").as_string();
        if (weight_type == "lstm") {
            fmt::print("Creating ");
            fmt::print(fmt::emphasis::underline, "LSTMWeightFunctionModule");
            fmt::print("..\n");
            weight = std::make_unique<LSTMWeightFunctionModule>(config.at("lstm").as_object(),
                                                                window, in_dim, k);
        } else {
            // throw std::runtime_error();
            fmt::print(fmt::fg(fmt::color::red), "Invalid weight type: {}.", weight_type);
            std::terminate();
        }
        return weight;
    };
};