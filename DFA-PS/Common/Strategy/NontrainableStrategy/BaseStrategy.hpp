#pragma once

#include <boost/json.hpp>
#include "torch/torch.h"
#include "../../SciLib/Finance.hpp"
#include "../../Util.hpp"
#include "../../DecisionModel/Metrics.hpp"

struct StrategyInput {
    // 2D: (W, A)
    TMatd history_returns, history_prices;
    TMatb valid_mask;
};

struct BaseStrategy {
    explicit BaseStrategy(const boost::json::object &config);

    boost::json::object config;
    std::string name{""};
    int top_k, window, holding_period;
    SciLib::ReturnCalculator cal;
    double short_ratio = 0.25;

    virtual void init(const DataSet &data){};
    virtual void print_info();
    virtual void run(const DataSet &data);
    virtual SciLib::PortfolioWeight<TVecd> cal_weight(const StrategyInput &X) = 0;
};

// Read data from market index and calculate returns.
struct MarketIndexStrategy : BaseStrategy {
    explicit MarketIndexStrategy(const boost::json::object &config);

    TVecd index;
    void init(const DataSet &data) override; // Read index data.
    void run(const DataSet &data) override;
    SciLib::PortfolioWeight<TVecd> cal_weight(const StrategyInput &X) { return {TVecd(), 0}; };
};

struct EqualWeightStrategy : BaseStrategy {
    explicit EqualWeightStrategy(const boost::json::object &config);
    // void print_info() override;
    SciLib::PortfolioWeight<TVecd> cal_weight(const StrategyInput &X) override;
};

struct BuyAndHoldStrategy : BaseStrategy {

    double cash;
    TVecd volume;

    explicit BuyAndHoldStrategy(const boost::json::object &config);
    void init(const DataSet &data) override;
    void run(const DataSet &data) override;
    SciLib::PortfolioWeight<TVecd> cal_weight(const StrategyInput &X) override {
        return {TVecd(), 0.0};
    };

    SciLib::PortfolioWeight<TVecd> cal_topk_equal_weight(const StrategyInput &X);
};

struct MomentumStrategy : BuyAndHoldStrategy {

    explicit MomentumStrategy(const boost::json::object &config);

    SciLib::PortfolioWeight<TVecd> cal_weight(const StrategyInput &X) override {
        return cal_topk_equal_weight(X);
    }
    void init(const DataSet &data) override{};
    void run(const DataSet &data) override { BaseStrategy::run(data); };
};

std::shared_ptr<BaseStrategy> create_nontrainable_strategy(const boost::json::object &config);