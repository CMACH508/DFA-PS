#pragma once

#include "../DecisionModel/torchENRBF.hpp"
#include "../EigenTorchHelper.hpp"
//#include "../FactorModel/PolynomialTFAModel.hpp"
#include "../FactorModel/create_tfa_model.hpp"
#include "../JsonHelper.hpp"
#include "../Util.hpp"
#include "StrategyBase.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// 1. Traing happen at two stage. The first is use all training data to update model. The second is
// on test stage, update the model at daily frequency.
struct SSMTFAStrategy {

    // assets: not including risk-free.
    SSMTFAStrategy(boost::json::object &config, int assets, std::shared_ptr<TFAModelBase> ssm,
                   std::shared_ptr<DecisionModelBase> alpha_model,
                   std::shared_ptr<DecisionModelBase> beta_model);

    int assets;
    double r_f, r_c;
    bool costQ;

    boost::json::object config;

    bool test_on_train_dataset;
    // assets: not including risk-free.

    virtual void save_config(const std::string &filename = "config.json");

    virtual void train_test(const Dataset &data);

    virtual void begin_test();
    std::pair<Performance, TMatd> test(const TMatd &test, const TMat<bool> &nan_mask,
                                       bool write_output, const std::string &portfolio_output = "");
    virtual void after_test();

    int ssm_hidden_dim, step_epochs;
    int ssm_max_batch_size = 0;
    // int topk;

    int ssm_epochs, decision_model_epochs;

    bool use_gpu;
    torch::Device device = torch::kCPU;
    std::shared_ptr<TFAModelBase> ssm;
    std::shared_ptr<DecisionModelBase> alpha_model, beta_model;
    std::shared_ptr<WeightWrapperBase> alpha_wrapper, beta_wrapper;
    std::unique_ptr<SSMTFAStrategyDataPool> data_pool;

    void backup();
    void restore();
    void train_test(const Dataset &data);
    void begin_test();
    // std::pair<double, TVecd> step(const TVecd &y, const TVec<bool> &nan_mask);

    bool update_decision_model = true, update_ssm_model = true;

    torch::Tensor last_ssm_perdicted_hidden_factor;
    torch::Tensor train_hidden_factors, train_returns, train_returns_nan_mask;
    // torch::Tensor train_hidden_factors, train_returns;
    torch::Tensor test_hidden_factors, test_returns, test_returns_nan_mask;

    // Given a predicted hidden factor, generate decision.
    virtual std::pair<double, TVecd> calculate_portfolio(const TVecd &x);
    // Given a sequence of hidden factor, calculate target function, e.g. sharpe ratio.
    virtual torch::Tensor calculate_performance(const torch::Tensor &xs,
                                                const torch::Tensor &returns,
                                                const torch::Tensor &nan_mask);

    torch::Tensor train_epoch(const torch::Tensor &factors, const torch::Tensor &returns,
                              const torch::Tensor &nan_mask);

    // alpha, beta, sharp ratio
    virtual void post_update_decision();
};

std::tuple<std::shared_ptr<DecisionModelBase>, std::shared_ptr<DecisionModelBase>>
create_alpha_beta(boost::json::object &config, int assets);

// template <typename StrategyType>
void run_SSMTFAStrategy(const Dataset &data, boost::json::object &config) {
    // assert(train.cols() == test.cols());

    int assets = static_cast<int>(data.assets);

    fmt::print("Initializing SSM-TFA strategy...\n");

    //----------------------------------------------------------------------------
    auto filter_model = create_tfa_model(config, assets);
    auto [alpha_model, beta_model] = create_alpha_beta(config, assets);

    SSMTFAStrategy strategy(config, assets, filter_model, alpha_model, beta_model);

    fmt::print("Now training strategy...\n");
    bool costQ{boost::json::value_to<double>(config["cost"]) != 0};
    strategy.train_test(data);
};
