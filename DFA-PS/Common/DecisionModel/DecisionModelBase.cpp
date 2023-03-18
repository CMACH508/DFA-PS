#include "stdafx.h"
#include <fmt/chrono.h>
#include "DecisionModelBase.hpp"
#include "SciLib/STDHelper.hpp"
#include "SciLib/MISC.hpp"
#include "SciLib/EigenTorchHelper.hpp"
// #include "../ModelUtil.hpp"

DecisionModelBase::DecisionModelBase(boost::json::object &config, int window, int n_in, int n_out,
                                     int asset_feature_size, int market_feature_size)
    : config(config), seed(config["seed"].as_int64()), n_in(n_in), n_out(n_out),
      asset_feautre_size(asset_feature_size), market_feature_size(market_feature_size),
      r_f(boost::json::value_to<double>(config["risk_free"])),
      r_c(boost::json::value_to<double>(config["cost"])), lr(config["lr"].as_double()),
      resample_interval(config["resample_interval"].as_int64()),
      allow_short(config["allow_short"].as_bool()),
      initial_cash(static_cast<float>(boost::json::value_to<double>(config["initial_cash"]))),
      holding_period(config["holding_period"].as_int64()), window(window),
      only_use_periodic_price(config["only_use_periodic_price"].as_bool()),
      data_period(only_use_periodic_price ? holding_period : 1),
      epochs(static_cast<int>(config["epochs"].as_int64())),
      write_test_portfolio_interval(config["write_test_portfolio_interval"].as_int64()) {
    std::cout << "Decision Model begin" << std::endl;
    if (seed != -1) {
        torch::manual_seed(static_cast<uint64_t>(seed));
        torch::cuda::manual_seed_all(static_cast<uint64_t>(seed));
    }

    // at::globalContext().setDeterministicAlgorithms(true, false);
    // at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);

    if (config["use_gpu"].as_bool()) {
        if (torch::cuda::is_available()) {
            use_gpu = true;
#ifdef _WIN32
            LoadLibraryA("torch_cuda.dll");
#endif

        } else {
            fmt::print(fmt::fg(fmt::color::red), "[DecisionModel]: GPU is set in config file, but "
                                                 "is not availiable. So CPU will be used.\n");
        }
    }
    device = use_gpu ? torch::kCUDA : torch::kCPU;

    fmt::print("Cost: {}\n", r_c);

    // GPU is set after _init_net();
    cal = SciLib::ReturnCalculator{
        allow_short, only_use_periodic_price, initial_cash, holding_period, r_c, device};
    _init_table_writer();
};

void DecisionModelBase::save(const std::string &name_, const std::string &sub_name, bool info) {
    if (info)
        fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModel]: Saving backup {}...\n", name_);

    std::string name = name_;
    if (!sub_name.empty()) {
        name = fmt::format("{}/{}", name_, sub_name);
        std::filesystem::create_directories(fmt::format("backup/{}", name));
    }
};
void DecisionModelBase::load(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModelBase]: Loading backup from {}...\n",
               name);
};

void DecisionModelBase::before_train() {
    fmt::print("[DecisionModelBase]: Using lR: {}\n", lr);
    std::filesystem::create_directories(fmt::format("portfolios/{}", id_buf));

    tb.set_file(fmt::format("history/{}/decision_model_train_history.csv", id_buf));
    tb.print_header();
    tb.write_header();

    report_model();
};
void DecisionModelBase::after_train() {
    fmt::print(fmt::fg(fmt::color::yellow), "Best result: \n");
    tb.print_best_row("T.ARR", false, "");
};
void DecisionModelBase::before_epoch() {
    tb.new_row();
    sw.reset();
}

void DecisionModelBase::train(const DataPool &data_pool, const std::string &id) {
    pretrain(data_pool, id);

    id_buf = id;
    // SciLib::create_parent_directory()

    before_train();

    DecisionModelDataSet train_data;
    for (epoch = 1; epoch <= epochs; ++epoch) {
        before_epoch();
        // std::cout << alpha_net->parameters()[0].narrow(0, 0, 1) << std::endl;
        if (epoch % resample_interval == 1) {

            train_data = data_pool.sample_train_dataset();

            if (epoch == 1) [[unlikely]] {
                train_data.validate();
                test(train_data, fmt::format("portfolios/{}/train_portfolio_initial.csv", id));
                test(data_pool.test_data,
                     fmt::format("portfolios/{}/test_portfolio_initial.csv", id));
            }
        }

        auto [target, train_performance] = train_epoch(train_data);

        std::tie(test_performance, test_portfolio) = test(data_pool.test_data);

        tb["Epoch"] = epoch;
        tb["L"] = target.item<double>();
        tb["Wea"] = train_performance.Wealth;
        tb["ARR"] = train_performance.ARR * 100;
        tb["Vol"] = train_performance.AVol;
        tb["ASR"] = train_performance.ASR;
        tb["SoR"] = train_performance.SoR;
        tb["MDD"] = train_performance.MDD * 100;
        tb["CR"] = train_performance.CR;
        tb["T.Wea"] = test_performance.Wealth;
        // std::cout << test_performance.Wealth << std::endl;
        tb["T.ARR"] = test_performance.ARR * 100;
        tb["T.Vol"] = test_performance.AVol;
        tb["T.ASR"] = test_performance.ASR;
        tb["T.SoR"] = test_performance.SoR;
        tb["T.MDD"] = test_performance.MDD * 100;
        tb["T.CR"] = test_performance.CR;
        tb["Time"] = SciLib::stopwatch_elapsed_seconds(sw);

        after_epoch();
        if ((write_test_portfolio_interval > 0) && (epoch % write_test_portfolio_interval == 0)) {
            test_portfolio.write(fmt::format("portfolios/{}/test_{}.csv", id, epoch));
            test(train_data, fmt::format("portfolios/{}/train_{}.csv", id, epoch));
        }
    }

    // Write train portfolio.
    test(train_data, fmt::format("portfolios/{}/train_portfolio_final.csv", id));
    test(data_pool.test_data, fmt::format("portfolios/{}/test_portfolio_final.csv", id));

    after_train();
}

void DecisionModelBase::after_epoch() {
    tb.print_row();
    tb.write_row();

    if (tb.update_best_row()) {
        // Write data when new best result.
        test_portfolio.write(fmt::format("portfolios/{}/test_portfolio_best.csv", id_buf));
        train_portfolio.write(fmt::format("portfolios/{}/train_portfolio_best.csv", id_buf));
        save(id_buf, "best", false);
    }

    // Periodicly write train history.
    if ((write_test_portfolio_interval > 0) && (epoch % write_test_portfolio_interval == 1)) {

        auto best_row2 = tb.get_best_row("T.ARR", false);
        tb.write_row_with_header(
            tb.get_best_row(),
            fmt::format("backup/{}/{}-[{} {:6.3f} {:5.2f}]-[{} {:6.3f} {:5.2f}].csv", id_buf, epoch,
                        tb.loc<int>(tb.get_best_row(), "Epoch"),
                        tb.loc<double>(tb.get_best_row(), "T.ARR"),
                        tb.loc<double>(tb.get_best_row(), "T.MDD"), tb.loc<int>(best_row2, "Epoch"),
                        tb.loc<double>(best_row2, "T.ARR"), tb.loc<double>(best_row2, "T.MDD")));
    }
}

std::string DecisionModelBase::get_simple_performace_desc() {
    return fmt::format("{} {:6.3f} {:5.2f}", tb.loc<int>(tb.get_best_row(), "Epoch"),
                       tb.loc<double>(tb.get_best_row(), "T.ARR"),
                       tb.loc<double>(tb.get_best_row(), "T.MDD"));
}

std::pair<PortfolioPerformance, SciLib::FullPortfolioData>
DecisionModelBase::test(const DecisionModelDataSet &data, const std::string &filename) {
    torch::NoGradGuard nograd;
    before_test();

    auto [alpha, beta] = calculate_portfolio_weight(data);

    SciLib::FullPortfolioData portfolio{
        cal.calculate_returns<false>(alpha, beta, data.prices, data.dates)};

    if (!filename.empty())
        portfolio.write(filename);
    // Warning: here interval is 1.0 as price data is daily.
    return std::make_pair(
        calculate_performance(portfolio.wealth, portfolio.returns, data_period, r_f, r_f),
        portfolio);
};

void DecisionModelBase::_init_table_writer() {
    tb = SciLib::TableWriter("Epoch", "L", "Wea", "ARR", "Vol", "ASR", "SoR", "MDD", "CR", "T.Wea",
                             "T.ARR", "T.Vol", "T.ASR", "T.SoR", "T.MDD", "T.CR", "Time");
    // tb.set_file("decision_model_train_history.csv");
    tb.set_col_type(1, "Epoch");
    tb.set_type_formatter("{: >6.3f}", "{: ^6}", 0);
    tb.set_col_formatter("{: >5}", "{: ^5}", "Epoch");
    tb.set_col_formatter("{: >6.2f}", "{: ^6}", "L", "Wea", "T.Wea", "ARR", "T.ARR", "MDD", "T.MDD",
                         "Time");
    tb.add_sep_after("Epoch", "CR", "T.CR");

    tb.set_monitor_col("T.CR", false);
};
