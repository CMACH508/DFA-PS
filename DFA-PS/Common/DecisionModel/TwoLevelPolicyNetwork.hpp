#pragma once
#include "PolicyNetworkBase.hpp"

class TwoLevelPolicyNetwork : public PolicyNetworkBase {
  public:
    TwoLevelPolicyNetwork(boost::json::object &config, int window, int n_in, int n_out,
                          int asset_feature_size, int market_feature_size);

    torch::Tensor cal_score(torch::Tensor factor_input, torch::Tensor asset_feature) override;
    std::pair<torch::Tensor, SciLib::FullBetaPortfolioWeight>
    calculate_portfolio_weight(const DecisionModelDataSet &data) override;

    void before_pretrain() override;

    void save(const std::string &name, const std::string &sub_name = "", bool info = true) override;
    // name: full path: backup/UUID.
    void load(const std::string &name) override;

    void report_model() override;

  private:
    RatioModule ratio_module = nullptr;
    BasicModule basic_module = nullptr;
    CalibrationModule calibration_module = nullptr;

    bool ratio_module_q = true, basic_module_q = true, calibration_module_q = true;

    void set_eval() override;
    void set_train() override;
};
