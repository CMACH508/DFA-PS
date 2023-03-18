#pragma once
#include "TwoLevelPolicyNetwork.hpp"

std::shared_ptr<DecisionModelBase> create_decision_model(boost::json::object &config, int window,
                                                         int n_in, int n_out,
                                                         int asset_feature_size,
                                                         int market_feature_size);