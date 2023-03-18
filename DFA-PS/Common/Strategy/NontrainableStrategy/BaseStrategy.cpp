#include "stdafx.h"
#include "BaseStrategy.hpp"
#include "../../SciLib/EigenTorchHelper.hpp"
#include "../../SciLib/STDHelper.hpp"
//#include "fort.hpp"

BaseStrategy::BaseStrategy(const boost::json::object &config)
    : config(config), top_k(static_cast<int>(config.at("topk").as_int64())),
      window(static_cast<int>(config.at("window").as_int64())),
      holding_period(static_cast<int>(config.at("holding_period").as_int64())),
      cal{config.at("allow_short").as_bool(),    config.at("only_use_periodic_price").as_bool(),
          config.at("initial_cash").as_double(), holding_period,
          config.at("cost").as_double(),         torch::kCPU} {}

void BaseStrategy::print_info() {
    fmt::print(fmt::fg(fmt::color::yellow), "Running {} strategy...\n", name);

    fort::char_table table;
    table << "Allow short" << cal.allow_short << fort::endr << "Cost" << cal.r_c << fort::endr
          << "Holding period" << cal.data_period << fort::endr;
    std::cout << table.to_string();
}

void BaseStrategy::run(const DataSet &data) {
    spdlog::stopwatch sw;

    std::filesystem::create_directories("result");
    std::ofstream("result/config.json") << config << std::endl;

    print_info();

    init(data);

    int N1 = data.train_returns.rows(), N2 = data.test_returns.rows();
    // std::cout << data.assets << std::endl;
    TMatd returns_full(N1 + N2, data.assets), close_full(N1 + N2, data.assets);
    close_full << data.train_close, data.test_close;
    returns_full << data.train_returns, data.test_returns;
    TMatb valid_mask_full(N1 + N2, data.assets);
    valid_mask_full << data.train_valid_mask, data.test_valid_mask;

    TVeci inds(N2);
    std::iota(inds.begin(), inds.end(), 0);
    TVeci selected_inds = inds(Eigen::indexing::seq(0, Eigen::indexing::last, holding_period));
    int sel_N2 = selected_inds.size();

    TMatd subset_test_close(sel_N2, data.assets);
    TVeci subset_test_dates(sel_N2);

    StrategyInput X;
    std::vector<SciLib::PortfolioWeight<TVecd>> ws;
    for (int i = 0; i < sel_N2; ++i) {
        int k = selected_inds.coeff(i);

        subset_test_close.row(i) = data.test_close.row(k);
        subset_test_dates.coeffRef(i) = data.test_dates[k];

        // Construct input.
        X.history_prices = close_full.middleRows(N1 + k - window + 1, window);
        X.history_returns = returns_full.middleRows(N1 + k - window + 1, window);
        X.valid_mask = valid_mask_full.middleRows(N1 + k - window + 1, window);

        ws.emplace_back(cal_weight(X));
    }
    auto [alpha, beta] = SciLib::convert_portfolio_weight(ws);
    auto price_tensor = SciLib::mat_to_tensor(subset_test_close);
    auto dates_tensor = SciLib::mat_to_tensor(subset_test_dates);
    auto portfolio = cal.calculate_returns(alpha, beta, price_tensor, dates_tensor);
    portfolio.write("result/portfolio.csv");

    auto performance = SciLib::calculate_portfolio_performance(portfolio, cal.data_period);
    performance.report();
    performance.write("result/performance.csv", true);

    fmt::print(fmt::fg(fmt::color::yellow), "Finished using {:6.3f} minutes.",
               static_cast<double>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(sw.elapsed()).count()) /
                   (60 * 1000));
}

MarketIndexStrategy::MarketIndexStrategy(const boost::json::object &config) : BaseStrategy(config) {
    name = "Market Index";
}
void MarketIndexStrategy::init(const DataSet &data) {
    fmt::print("Reading index data...\n");
    auto test_start = data.test_dates[0];

    std::string index_filename = SciLib::path_join(
        SciLib::json_string_to_string(config.at("data_path").as_string()), "index.csv");
    SciLib::check_path_exists(index_filename);

    rapidcsv::Document doc(index_filename, rapidcsv::LabelParams(0, 0), rapidcsv::SeparatorParams(),
                           rapidcsv::ConverterParams(true));
    //
    std::vector<std::string> _all_dates = doc.GetRowNames();
    int i = 0;
    for (; i < _all_dates.size(); ++i) {
        if (boost::lexical_cast<int>(_all_dates[i]) >= test_start) {
            break;
        }
    }
    //  Index value
    std::vector<double> index_value = doc.GetColumn<double>(0);
    index.resize(_all_dates.size() - i);
    for (int j = 0; j < index.size(); ++j) {
        index.coeffRef(j) = index_value[i + j];
    }
}
void MarketIndexStrategy::run(const DataSet &data) {
    init(data);

    int N2 = index.size();
    TVeci inds(N2);
    std::iota(inds.begin(), inds.end(), 0);
    TVeci selected_inds = inds(Eigen::indexing::seq(0, Eigen::indexing::last, holding_period));
    int sel_N2 = selected_inds.size();

    TVecd wealth = index(selected_inds);

    SciLib::FullPortfolioData portfolio;
    portfolio.wealth = SciLib::mat_to_tensor(TVecd(wealth.bottomRows(sel_N2 - 1)));
    portfolio.returns = SciLib::mat_to_tensor(
        TVecd(wealth.bottomRows(sel_N2 - 1).array() / wealth.topRows(sel_N2 - 1).array() - 1));

    auto performance = SciLib::calculate_portfolio_performance(portfolio, cal.data_period);
    performance.report();
    performance.write("result/performance.csv", true);
}
EqualWeightStrategy::EqualWeightStrategy(const boost::json::object &config) : BaseStrategy(config) {
    name = "Equal Weight";
    fmt::print("Short and topk is invalid in this strategy.\n");
    cal.allow_short = false;
    this->config["allow_short"] = false;
}

SciLib::PortfolioWeight<TVecd> EqualWeightStrategy::cal_weight(const StrategyInput &X) {
    // std::cout << "X rows: " << X.valid_mask.rows() << std::endl;
    TVecb valid_mask = X.valid_mask.bottomRows(1).transpose();
    TVecd w = static_cast<double>(valid_mask.sum()) / valid_mask.size() * valid_mask.cast<double>();
    // fmt::print("w: {}\n", w.size());
    return {w, 0.99};
}

BuyAndHoldStrategy::BuyAndHoldStrategy(const boost::json::object &config) : BaseStrategy(config) {
    name = "Buy and Hold";
    if (cal.allow_short) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Short trading is not allowed in buy and hold strategy.\n");
        cal.allow_short = false;
    }
}
void BuyAndHoldStrategy::init(const DataSet &data) {
    BaseStrategy::init(data);

    auto pw = cal_topk_equal_weight({data.train_returns, data.train_close, data.train_valid_mask});
    // std::cout << pw.w.transpose() << std::endl;

    TVecd zero_vec = TVecd::Zero(data.assets);
    TVecd long_weight = (pw.w.array() > 0).select(pw.w, zero_vec);
    // TVecd short_weight = zero_vec;
    // if (cal.allow_short) {
    //     short_weight =
    //         TVecd((pw.w.array() < 0)
    //                   .select(-pw.w, zero_vec)); // Here short_weight is converted to positive.
    // }

    // TVecd full_w = (1 + short_ratio) * long_weight - short_ratio * short_weight;
    //  std::cout << full_w.transpose() << std::endl;
    volume = 0.99 * long_weight.array() / data.train_close.bottomRows(1).transpose().array();
    cash = 1 - 0.99 - 0.99 * cal.r_c;
}
void BuyAndHoldStrategy::run(const DataSet &data) {
    spdlog::stopwatch sw;

    std::filesystem::create_directories("result");
    std::ofstream("result/config.json") << config << std::endl;

    print_info();

    init(data);

    int N1 = data.train_returns.rows(), N2 = data.test_returns.rows();

    TVeci inds(N2);
    std::iota(inds.begin(), inds.end(), 0);
    TVeci selected_inds = inds(Eigen::indexing::seq(0, Eigen::indexing::last, holding_period));
    int sel_N2 = selected_inds.size();

    TVecd wealth(sel_N2 + 1);
    wealth.coeffRef(0) = cal.initial_cash;
    for (int i = 0; i < sel_N2; ++i) {
        int k = selected_inds.coeff(i);
        TVecd p = data.test_close.row(k).transpose();
        wealth.coeffRef(i + 1) = cash + volume.dot(p);
    }
    // Only wealth and returns are needed to calculate performance.
    SciLib::FullPortfolioData portfolio;
    portfolio.wealth = SciLib::mat_to_tensor(TVecd(wealth.bottomRows(sel_N2)));
    portfolio.returns = SciLib::mat_to_tensor(
        TVecd(wealth.bottomRows(sel_N2).array() / wealth.topRows(sel_N2).array() - 1));
    // std::cout << portfolio.wealth << std::endl;

    auto performance = SciLib::calculate_portfolio_performance(portfolio, cal.data_period);
    performance.report();
    performance.write("result/performance.csv", true);

    fmt::print(fmt::fg(fmt::color::yellow), "Finished using {:6.3f} minutes.",
               static_cast<double>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(sw.elapsed()).count()) /
                   (60 * 1000));
}

SciLib::PortfolioWeight<TVecd> BuyAndHoldStrategy::cal_topk_equal_weight(const StrategyInput &X) {
    // mean returns;
    TVecd mean_returns = X.history_returns.colwise().sum().transpose().array() /
                         X.valid_mask.cast<double>().colwise().sum().transpose().array();
    // Find assets with largets returns.
    auto [mx_returns, mx_ind] = SciLib::keep_topk(mean_returns, top_k, std::greater<double>{});

    // Find assets with smallest returns.
    auto [mn_returns, mn_ind] = SciLib::keep_topk(mean_returns, top_k);

    int A = X.history_returns.cols();
    TVecd w(A);
    w.setConstant(0);
    w(mx_ind).setConstant(1.0 / top_k);
    w(mn_ind).setConstant(-1.0 / top_k);
    return {w, short_ratio};
}

MomentumStrategy::MomentumStrategy(const boost::json::object &config) : BuyAndHoldStrategy(config) {
    name = "Momentum";
}

std::shared_ptr<BaseStrategy> create_nontrainable_strategy(const boost::json::object &config) {
    std::string type = SciLib::json_string_to_string(config.at("strategy_type").as_string());
    std::shared_ptr<BaseStrategy> strategy;
    if (type == "equal_weight") {
        strategy = std::make_shared<EqualWeightStrategy>(config);
    } else if (type == "buy_and_hold") {
        strategy = std::make_shared<BuyAndHoldStrategy>(config);
    } else if (type == "momentum") {
        strategy = std::make_shared<MomentumStrategy>(config);
    } else if (type == "market_index") {
        strategy = std::make_shared<MarketIndexStrategy>(config);
    } else {
        fmt::print(fmt::fg(fmt::color::red), "Invalid strategy type: {}\n", type);
    }
    return strategy;
}