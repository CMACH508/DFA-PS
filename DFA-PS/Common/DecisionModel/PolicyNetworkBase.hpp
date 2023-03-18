#pragma once
#include "DecisionModelBase.hpp"

struct AlphaWeightWrapperBase {
    // BetaWeightWrapperBase();
    // TrivialAlphaModelWrapper(const boost::json::object &config);
    virtual torch::Tensor forward(const torch::Tensor &ori_weight) = 0;
};

struct BetaWeightWrapperBase {
    int topk = 0, assets = 0;

    BetaWeightWrapperBase();
    BetaWeightWrapperBase(int topk, int assets);

    // Fill weight with nan_mask to be -inf.
    virtual torch::Tensor process_nan_mask(const torch::Tensor &ori_weight,
                                           const torch::Tensor &nan_mask, const torch::Tensor &val);
    // Here weight is the weight processed by process_nan_mask.
    // [Warning]: there is a bug when topk < valid assets. But currently I have no good method to
    // deal with it.
    virtual torch::Tensor select_topk(const torch::Tensor &weight);
    // For train.
    virtual SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &ori_weight) {
        return {torch::Tensor(), torch::Tensor()};
    };
    // For test.
    virtual SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &ori_weight,
                                                    const torch::Tensor &nan_mask) = 0;

    virtual ~BetaWeightWrapperBase() = default;
};

struct PreTrainOptions {
    double lr = 1e-5;
    int epochs = 1000;
    int resample_interval = 100;
    int period = 6;
    bool use = true;

    PreTrainOptions(boost::json::object &config);
};

class PolicyNetworkBase : public DecisionModelBase {
  public:
    PolicyNetworkBase(boost::json::object &config, int window, int n_in, int n_out,
                      int asset_feature_size, int market_feature_size);

    virtual torch::Tensor cal_score(torch::Tensor factor_input, torch::Tensor asset_feature) = 0;

    void pretrain(const DataPool &data_pool, const std::string &id) override;
    virtual void before_pretrain();
    virtual void after_pretrain() {}

    std::pair<torch::Tensor, PortfolioPerformance>
    train_epoch(const DecisionModelDataSet &data) override;
    void save(const std::string &name, const std::string &sub_name = "", bool info = true) override;
    // name: full path: backup/UUID.
    void load(const std::string &name) override;

    virtual void set_eval() {}
    virtual void set_train() {}

    // virtual void report_model(){};

    void before_test() override { set_eval(); }

  protected:
    PreTrainOptions pretrain_option;
    SciLib::TableWriter pretrain_tb;

    std::shared_ptr<torch::optim::AdamW> pretrain_opt;
    std::shared_ptr<torch::optim::Adam> opt;

    std::shared_ptr<AlphaWeightWrapperBase> alpha_wrapper;
    std::shared_ptr<BetaWeightWrapperBase> beta_wrapper;

    std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)> metric_fn;
};

struct TrivialAlphaModelWrapper : AlphaWeightWrapperBase {
    double min_v;
    TrivialAlphaModelWrapper(const boost::json::object &config);
    torch::Tensor forward(const torch::Tensor &ori_weight) override;
};
struct TrivialBetaModelWrapper : BetaWeightWrapperBase {
    TrivialBetaModelWrapper(const boost::json::object &config, int assets);

    SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &ori_weight) override;
    SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &ori_weight,
                                            const torch::Tensor &nan_mask) override;
};
struct AllowingShortBetaModelWrapper : BetaWeightWrapperBase, public torch::nn::Module {
    double nv;
    // torch::nn::BatchNorm1d bn = nullptr;

    AllowingShortBetaModelWrapper(const boost::json::object &config, int n_out);

    SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &sel_weight) override {
        return {};
    };

    SciLib::FullBetaPortfolioWeight forward(const torch::Tensor &ori_weight,
                                            const torch::Tensor &nan_mask) override;
};