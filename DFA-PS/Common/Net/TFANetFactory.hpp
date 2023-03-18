#pragma once
#include "WeightedSumModel.hpp"

struct TFANetFactory {
    std::shared_ptr<TFANetBase> operator()(boost::json::object &config, int dim) {
        std::shared_ptr<TFANetBase> net;
        auto net_type = config["net_type"].as_string();
        if (net_type == "weighted_sum") {
            net = std::make_shared<WeightedSumModel>(config, dim);
        } else {
            throw std::runtime_error(fmt::format("Invalid net_type: {}.", net_type));
        }
        return net;
    };
};