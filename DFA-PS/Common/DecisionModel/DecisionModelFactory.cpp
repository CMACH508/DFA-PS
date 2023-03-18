#include "stdafx.h"

#include "DecisionModelFactory.hpp"

std::shared_ptr<DecisionModelBase> create_decision_model(boost::json::object &config, int window,
                                                         int n_in, int n_out,
                                                         int asset_feature_size,
                                                         int market_feature_size) {
    if (auto p = config.if_contains("model_type"); p) {
        if (p->as_string() == "TwoLevel") {
            return std::make_shared<TwoLevelPolicyNetwork>(config, window, n_in, n_out,
                                                           asset_feature_size, market_feature_size);
        } else {
            fmt::print(fmt::fg(fmt::color::yellow),
                       "[create_decision_model]: invalid model_type {}...\n",
                       config["model_type"].as_string());
            return nullptr;
        }
    } else {
        fmt::print(fmt::fg(fmt::color::red),
                   "[create_decision_model]: no model_type, use 'TwoLevel' as default...\n");
        return std::make_shared<TwoLevelPolicyNetwork>(config, window, n_in, n_out,
                                                       asset_feature_size, market_feature_size);
    }
};