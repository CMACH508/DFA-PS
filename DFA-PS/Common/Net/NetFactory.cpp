#include "stdafx.h"
#include "ENRBF.hpp"
#include "LSTM.hpp"
#include "NetFactory.hpp"
#include "WeightedSumModel.hpp"

std::shared_ptr<NetBase> create_net(const boost::json::object &config, int window, int in,
                                    int out) {
    if (config.at("net_type").as_string() == "native") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Native TFA");
        fmt::print(" network...\n");
        return std::make_shared<NativeTFALinear>(in * window);
    } else if (config.at("net_type").as_string() == "enrbf") {
        if (window != 1) {
            fmt::print("Window of enrbf net must be 1. However the given value is {}.\n", window);
            std::terminate();
        }
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "ENRBF");
        fmt::print(" network...\n");
        return std::make_shared<ENRBF>(config.at("enrbf").as_object(), in, out);
    } else if (config.at("net_type").as_string() == "lstm") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "LSTM");
        fmt::print(" network...\n");
        return std::make_shared<LSTM>(config.at("lstm").as_object(), in, window, out);
    } else if (config.at("net_type").as_string() == "weighted_sum") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Weighted-Sum");
        fmt::print(" network...\n");
        return std::make_shared<WeightedSumModel>(config.at("weighted_sum").as_object(), window, in,
                                                  out);
    } else {
        fmt::print(fmt::fg(fmt::color::red), "Invalid net_type {}.\n",
                   config.at("net_type").as_string());
        return nullptr;
    }
};
