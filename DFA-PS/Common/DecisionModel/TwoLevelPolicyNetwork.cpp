#include "stdafx.h"
#include "TwoLevelPolicyNetwork.hpp"

#include "SciLib/EigenTorchHelper.hpp"

TwoLevelPolicyNetwork::TwoLevelPolicyNetwork(boost::json::object &config, int window, int n_in,
                                             int n_out, int asset_feature_size,
                                             int market_feature_size)
    : PolicyNetworkBase(config, window, n_in, n_out, asset_feature_size, market_feature_size) {

    // Create 3 modules.
    ratio_module_q = boost::json::value_to<bool>(config.at("RatioModule").at("use"));
    if (ratio_module_q) {
        ratio_module = RatioModule(RatioModuleOption{
            market_feature_size, window,
            SciLib::json_array_to_vector<int>(config.at("RatioModule").at("channel").as_array()),
            SciLib::json_array_to_vector<int>(config.at("RatioModule").at("stride").as_array()),
            SciLib::json_array_to_vector<int>(
                config.at("RatioModule").at("linear_seq_size").as_array())});

        alpha_wrapper =
            std::make_shared<TrivialAlphaModelWrapper>(config["RatioModule"].as_object());
    }
    basic_module_q = boost::json::value_to<bool>(config.at("BasisModule").at("use"));
    if (basic_module_q) {
        basic_module = BasicModule(BasicModuleOptions{
            n_in, window, n_out,
            boost::json::value_to<int>(config.at("BasisModule").at("lstm_num_layer")),
            boost::json::value_to<int>(config.at("BasisModule").at("lstm_output_channel"))});
    }
    calibration_module_q = boost::json::value_to<bool>(config.at("CalibrationModule").at("use"));
    if (calibration_module_q) {
        fmt::print(fmt::fg(fmt::color::yellow), "Use CalibrationModule...\n");
        calibration_module = CalibrationModule(CalibrationModuleOptions{
            basic_module_q
                ? asset_feautre_size + 1
                : asset_feautre_size, // If use basis module, then v_b will be concatenated.
            n_out, window,
            SciLib::json_array_to_vector<int>(
                config.at("CalibrationModule").at("channel").as_array()),
            SciLib::json_array_to_vector<int>(
                config.at("CalibrationModule").at("stride").as_array())});
    }
    if (!basic_module_q && !calibration_module_q) {
        fmt::print(
            fmt::fg(fmt::color::red),
            "Neither basis module nor calibration module is used. You must as least use one.");
        std::terminate();
    }

    if (allow_short) {
        fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModel]: Allowing short...\n");
        beta_wrapper = std::make_shared<AllowingShortBetaModelWrapper>(config, n_out);
    } else {
        beta_wrapper = std::make_shared<TrivialBetaModelWrapper>(config, n_out);

        if (ratio_module_q) {
            fmt::print(fmt::fg(fmt::color::red),
                       "[DecisionModel]: Short is not allowed. But Ratio Module is used, do you "
                       "confirm this is intennded? Press [y] to continue.\n");
            std::string q;
            std::cin >> q;
            if (q[0] != 'y') {
                std::terminate();
            }
        }
    }

    if (use_gpu) {
        if (ratio_module_q) {
            ratio_module->to(device);
        }
        if (basic_module_q) {
            basic_module->to(device);
        }
        if (calibration_module_q) {
            calibration_module->to(device);
        }
    }

    // Register parameters.
    std::vector<torch::optim::OptimizerParamGroup> parms;
    if (ratio_module_q) {
        parms.push_back(ratio_module->parameters());
    }
    if (basic_module_q) {
        parms.push_back(basic_module->parameters());
    }
    if (calibration_module_q) {
        parms.push_back(calibration_module->parameters());
    }

    opt = std::make_shared<torch::optim::Adam>(parms, lr);
}

std::pair<torch::Tensor, SciLib::FullBetaPortfolioWeight>
TwoLevelPolicyNetwork::calculate_portfolio_weight(const DecisionModelDataSet &data) {
    torch::Tensor alpha;

    auto basis_weight = cal_score(data.factor_input, data.asset_feature);
    auto beta = beta_wrapper->forward(basis_weight, data.price_nan_mask);

    if (ratio_module_q) {
        alpha = alpha_wrapper->forward(ratio_module->forward(
            data.market_feature, {beta.long_sel_weight, beta.short_sel_weight}));
    } else {
        alpha =
            torch::tensor(
                {static_cast<float>(config.at("RatioModule").at("min_value").as_double())}, device)
                .expand({data.market_feature.size(0), -1});
    }
    return {alpha, beta};
}

torch::Tensor TwoLevelPolicyNetwork::cal_score(torch::Tensor factor_input,
                                               torch::Tensor asset_feature) {
    torch::Tensor w;
    if (basic_module_q) {
        w = basic_module->forward(factor_input);
        if (calibration_module_q) {
            w = calibration_module->forward(asset_feature, w) * w;
        }
    } else if (calibration_module_q) {
        w = calibration_module->forward(asset_feature, torch::Tensor());
    }
    return w;
}

void TwoLevelPolicyNetwork::set_eval() {
    if (basic_module_q) {
        basic_module->eval();
    }
    if (calibration_module_q) {
        calibration_module->eval();
    }
    if (ratio_module) {
        ratio_module->eval();
    }
}
void TwoLevelPolicyNetwork::set_train() {
    if (basic_module_q) {
        basic_module->train();
    }
    if (calibration_module_q) {
        calibration_module->train();
    }
    if (ratio_module) {
        ratio_module->train();
    }
}

void TwoLevelPolicyNetwork::before_pretrain() {
    PolicyNetworkBase::before_pretrain();

    if (pretrain_option.use) {
        // Optimizer
        std::vector<torch::optim::OptimizerParamGroup> parms;
        if (basic_module_q) {
            parms.push_back(basic_module->parameters());
        }
        if (ratio_module_q) {
            parms.push_back(ratio_module->parameters());
        }
        if (calibration_module_q) {
            parms.push_back(calibration_module->parameters());
        }
        pretrain_opt = std::make_unique<torch::optim::AdamW>(parms, pretrain_option.lr);
    }
}

void TwoLevelPolicyNetwork::report_model() {
    fmt::print(fmt::fg(fmt::color::yellow), "Decision Model Number of parameters: \n");
    if (ratio_module_q) {
        fmt::print("Ratio Net: {}\n", SciLib::vec_tensor_numel(ratio_module->parameters()));
    }
    if (basic_module_q) {
        fmt::print("Basis Net: {}\n", SciLib::vec_tensor_numel(basic_module->parameters()));
    }
    if (calibration_module_q) {
        fmt::print("Calibration Net: {}\n",
                   SciLib::vec_tensor_numel(calibration_module->parameters()));
    }
}

void TwoLevelPolicyNetwork::save(const std::string &name_, const std::string &sub_name, bool info) {
    if (info)
        fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModel]: Saving backup {}...\n", name_);

    std::string name = name_;
    if (!sub_name.empty()) {
        name = fmt::format("{}/{}", name_, sub_name);
        std::filesystem::create_directories(fmt::format("backup/{}", name));
    }
    // save module
    if (basic_module_q) {
        torch::save(basic_module, fmt::format("backup/{}/DecisionModel_BasicModule.pt", name));
    }
    if (calibration_module_q) {
        torch::save(calibration_module,
                    fmt::format("backup/{}/DecisionModel_CalibrationModule.pt", name));
    }
    if (ratio_module_q) {
        torch::save(ratio_module, fmt::format("backup/{}/DecisionModel_RatioModule.pt", name));
    }
};
void TwoLevelPolicyNetwork::load(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModel]: Loading backup from {}...\n", name);

    // Load parameters.
    if (basic_module_q) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[DecisionModel]: Loading parameters for Basic Module...\n");
        torch::load(basic_module, SciLib::path_join(name, "DecisionModel_BasicModule.pt"));
    }

    if (calibration_module_q) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[DecisionModel]: Loading parameters for Calibration Module...\n");
        torch::load(calibration_module,
                    SciLib::path_join(name, "DecisionModel_CalibrationModule.pt"));
    }
    if (ratio_module_q) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[DecisionModel]: Loading parameters for Ratio Module...\n");
        torch::load(ratio_module, SciLib::path_join(name, "DecisionModel_RatioModule.pt"));
    }
};