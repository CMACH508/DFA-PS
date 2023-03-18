#pragma once
// #define FMT_HEADER_ONLY
#include "SciLib/JsonHelper.hpp"
// #include "../Net/NetBase.hpp"
// #include "../Net/NetFactory.hpp"

#include "CalibrationModuleImpl.hpp"
#include "BasicModule.hpp"
#include "RatioModuleImpl.hpp"

#include "../Util.hpp"
#include "Metrics.hpp"
#include "SciLib/TableWriter.hpp"

// #include "../BackTest/Util.hpp"

// #include "spdlog/spdlog.h"
// #include "spdlog/stopwatch.h"

#include <memory>
// #include <vector>
#include <functional>

namespace F = torch::nn::functional;

struct DecisionModelBase {
    DecisionModelBase(boost::json::object &config, int window, int n_in, int n_out,
                      int asset_feature_size, int market_feature_size);

    virtual std::pair<torch::Tensor, PortfolioPerformance>
    train_epoch(const DecisionModelDataSet &data) = 0;
    //( investment_ratio (N), beta (N, n_out)
    virtual std::pair<torch::Tensor, SciLib::FullBetaPortfolioWeight>
    calculate_portfolio_weight(const DecisionModelDataSet &data) = 0;

    virtual void report_model(){};

    // name: single name, e.g. UUID
    virtual void save(const std::string &name, const std::string &sub_name = "", bool info = true);
    // name: full path: backup/UUID.
    virtual void load(const std::string &name);

    // virtual void pretrain(const DataPool &data_pool, const std::string &id);
    virtual void pretrain(const DataPool &data_pool, const std::string &id) {}
    virtual void train(const DataPool &data_pool, const std::string &id);

    virtual void before_train();
    virtual void after_train();
    virtual void before_epoch();
    virtual void after_epoch();
    virtual void before_test() {}
    virtual void after_test() {}

    boost::json::object config;
    std::string id_buf;

    torch::Device device = torch::kCPU;

    // Settings related for returns computation.
    double r_f, r_c;
    bool allow_short;
    float initial_cash;
    int holding_period;
    bool only_use_periodic_price; // data period on test set.
    int data_period;

    int write_test_portfolio_interval;
    // Net.
    bool use_gpu = false;

    // void set_use_gpu();

    int64_t seed;
    double lr = 1e-3;
    int epoch, epochs, resample_interval;
    int window;
    int n_in, n_out; // k: components, n_out: assets
    int market_feature_size;
    int asset_feautre_size;

    spdlog::stopwatch sw;
    SciLib::TableWriter tb;

    PortfolioPerformance test_performance;
    SciLib::FullPortfolioData train_portfolio, test_portfolio;

    SciLib::ReturnCalculator cal;

    virtual std::string get_simple_performace_desc();

    virtual std::pair<PortfolioPerformance, SciLib::FullPortfolioData>
    test(const DecisionModelDataSet &data, const std::string &filename = "");

    //  virtual std::pair<torch::Tensor, torch::Tensor> calculate_portfolio_weight(const
    //  torch::Tensor &factor_input);

    template <bool train_mode = true>
    SciLib::FullPortfolioData
    calculate_returns_with_short(const torch::Tensor &alpha,
                                 const SciLib::FullBetaPortfolioWeight &weight,
                                 const torch::Tensor &prices, const torch::Tensor &dates);

    void _init_table_writer();

    virtual ~DecisionModelBase() = default;
};
