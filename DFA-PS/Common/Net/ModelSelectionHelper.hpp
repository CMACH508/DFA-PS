#pragma once

#include <torch/torch.h>

#define TOSTR(name) std::string(#name)

#define NET_CASTING_CHECKER(Base, Derived)                                                         \
    std::shared_ptr<Derived> new_net_casted;                                                       \
    try {                                                                                          \
        new_net_casted = std::dynamic_pointer_cast<Derived>(new_net);                              \
    } catch (...) {                                                                                \
        fmt::print(fmt::fg(fmt::color::red),                                                       \
                   "[{0}]: Can't cast shared_ptr<{1}> to "                                         \
                   "shared_ptr<{0}>. Please check passed pointer.",                                \
                   TOSTR(Derived), TOSTR(Base));                                                   \
        std::terminate();                                                                          \
    }                                                                                              \
    if (new_net->parameters().size() != new_net_casted->parameters().size()) {                     \
        fmt::print(fmt::fg(fmt::color::red),                                                       \
                   "[{}]: number of parameters is not equal when copy_parms. Please check.\n",     \
                   TOSTR(Derived));                                                                \
        std::terminate();                                                                          \
    }

// Fix output size. Only input_dim change.
void copy_lstm_helper(torch::nn::LSTM &this_m, const torch::nn::LSTM &that_m,
                      const std::vector<int> &input_index, const std::vector<int> &output_index);

void copy_linear_helper(torch::nn::Linear &this_m, const torch::nn::Linear that_m,
                        const std::vector<int> &input_index, const std::vector<int> &output_index);

template <typename T> inline bool abs_compare(T a, T b) { return (std::abs(a) < std::abs(b)); };