#include "stdafx.h"
#include "TFATradingSystem.hpp"

#include "DecisionModel/DecisionModelFactory.hpp"
#include "SciLib/STDHelper.hpp"
#include "SciLib/EigenTorchHelper.hpp"

TFATradingSystem::TFATradingSystem(){};
TFATradingSystem::TFATradingSystem(const std::string &file) {
    read_config(file);
    init_by_config();
    create_ssm();
};

TFATradingSystem::TFATradingSystem(const boost::json::object &config) : config(config) {
    init_by_config();
};

void TFATradingSystem::init_by_config() {
    if (config["decision_model"].as_object()["use_gpu"].as_bool() && torch::cuda::is_available()) {
        use_gpu = true;
        fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Decision model using GPU.\n");
    }

    use_precomputed_hidden_factor =
        config["decision_model"].as_object()["use_precomputed_hidden_factor"].as_bool();
    precomputed_hidden_factor_path = SciLib::json_string_to_string(
        config["decision_model"].as_object()["precomputed_hidden_factor_path"].as_string());
    if (use_precomputed_hidden_factor) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "[TFATradingSystem]: Using precomputed hidden_factor: {}.\n",
                   precomputed_hidden_factor_path);
    }

    decision_window = config["decision_model"].as_object()["window"].as_int64();
    decision_holding_period = config["decision_model"].as_object()["holding_period"].as_int64();
    window_type = SciLib::json_string_to_string(
        config["decision_model"].as_object()["window_type"].as_string());
    only_use_periodic_price =
        config["decision_model"].as_object()["only_use_periodic_price"].as_bool();

    read_data();
};

void TFATradingSystem::contruct_model() {}

void TFATradingSystem::read_config(const std::string &file) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: reading config {}\n", file);

    config = SciLib::read_config(file);
};

void TFATradingSystem::read_data() {
    std::string data_path{SciLib::json_string_to_string_view(config["data_path"].as_string())};
    double scale_returns = boost::json::value_to<double>(config["scale_returns"]);
    auto remove_abnormal_assets = config["remove_abnormal_assets"].as_bool();
    data = read_dataset(data_path, scale_returns, remove_abnormal_assets);

    // std::ofstream("close.tab") << data.train_close << std::endl;
    // std::ofstream("close_nan_mask.txt") << data.train_valid_mask << std::endl;
};
void TFATradingSystem::write_ssm_reconstructed_data() {
    fmt::print(fmt::fg(fmt::color::yellow),
               "[TFATradingSystem]: Now generating ssm recontructed data.\n");

    auto y0 = zero_CBT(fa->window, {1, 1, fa->hidden_dim}, fa->device);
    auto [Y, X] = fa->reconstruct(y0, data.train_returns);
    Eigen::IOFormat mat_format(6, Eigen::DontAlignCols, ", ");
    std::ofstream("Y.csv") << Y.format(mat_format) << std::endl;
    std::ofstream("X.csv") << data.train_returns.format(mat_format) << std::endl;
    std::ofstream("X-reconstructed.csv") << X.format(mat_format) << std::endl;
};

void TFATradingSystem::generate_decision_model_data_pool() {
    data_pool = fa->generate_data_pool(data, decision_window, decision_holding_period,
                                       only_use_periodic_price, window_type);
};

void TFATradingSystem::create_ssm() {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Constructing Factor Model...\n");
    fa = create_tfa(config["factor_model"].as_object(), data.assets);
    fa->init(data.train_returns);
    // ssm->init_parm(data.train_returns);
}

void TFATradingSystem::load_ssm(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Loading ssm...\n");
    fa = create_tfa(config["factor_model"].as_object(), data.assets);
    fa->load(name);
}

void TFATradingSystem::before_train_fa() {
    generate_decision_model_data_pool();
    SciLib::write_1d2d_tensor(data_pool.train_recovered_hidden_factors, "Y-train-initial.csv");
    SciLib::write_1d2d_tensor(data_pool.test_recovered_hidden_factors, "Y-test-initial.csv");

    SciLib::writematrix(data.train_returns, SciLib::path_join(SciLib::json_string_to_string(
                                                                  config["data_path"].as_string()),
                                                              "X-train.csv"));
    SciLib::writematrix(data.test_returns, SciLib::path_join(SciLib::json_string_to_string(
                                                                 config["data_path"].as_string()),
                                                             "X-test.csv"));
};
void TFATradingSystem::train_fa() {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Training ssm...\n");
    fa->train(data);
}
void TFATradingSystem::after_train_fa() {
    write_ssm_reconstructed_data();
    generate_decision_model_data_pool();
}

void TFATradingSystem::create_decision_model() {
    // decison_model
    decision_model = ::create_decision_model(
        config["decision_model"].as_object(), decision_window, data_pool.factor_dim, data.assets,
        data_pool.num_asset_feature, data_pool.num_market_feature);
};

void TFATradingSystem::load_decision_model(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Loading decision_model...\n");
    decision_model = ::create_decision_model(
        config["decision_model"].as_object(), decision_window, data_pool.factor_dim, data.assets,
        data_pool.num_asset_feature, data_pool.num_market_feature);
    decision_model->load(name);
}

void TFATradingSystem::train_decision_model(const std::string &id) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Training decision_model...\n");

    //-------------------------------------------------------------------------------------------------
    // Train decision model.
    int fa_hidden_dim = fa->hidden_dim;
    auto decision_config = config["decision_model"].as_object();

    data_pool.validate();
    // std::cout << data_pool.test_input.device() << std::endl;
    data_pool.to(decision_model->device);
    // std::cout << data_pool.test_input.device() << std::endl;

    decision_model->train(data_pool, id);
};

void TFATradingSystem::read_precomputed_hidden_factor() {
    data_pool = DataPool(
        data, use_gpu ? torch::kCUDA : torch::kCPU, decision_holding_period,
        only_use_periodic_price, window_type, decision_window,
        static_cast<int>(config["decision_model"].as_object()["max_batch_size"].as_int64()));
    data_pool.read_precomputed_hidden_factor(precomputed_hidden_factor_path);
}
void TFATradingSystem::fresh_run(const std::string &id) {
    const auto start = std::chrono::steady_clock::now();

    if (!use_precomputed_hidden_factor) {
        create_ssm();
        before_train_fa();
        train_fa();
        after_train_fa();
    } else {
        read_precomputed_hidden_factor();
    }

    create_decision_model();
    train_decision_model(id);

    const auto end = std::chrono::steady_clock::now();
    std::cout << "Total time: "
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() / 3600.0
              << " hours." << std::endl;
};

void TFATradingSystem::save(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Saving backup {}...\n", name);
    // SciLib::create_parent_directory(name);
    std::filesystem::create_directories(fmt::format("backup/{}", name));

    // ssm
    if (fa != nullptr) {
        fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Saving ssm...\n");
        fa->save(name);
    }

    SciLib::write_1d2d_tensor(data_pool.train_recovered_hidden_factors,
                              fmt::format("backup/{}/Y-train.csv", name));
    SciLib::write_1d2d_tensor(data_pool.test_recovered_hidden_factors,
                              fmt::format("backup/{}/Y-test.csv", name));

    // decision_model
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Saving decision_model...\n");
    decision_model->save(name);

    // Save history
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]:Saving history...\n");
    std::time_t t = std::time(nullptr);
    std::string tm = fmt::format("{:%Y-%m-%d %H_%M_%S}", fmt::localtime(t));
    auto pdesc = decision_model->get_simple_performace_desc();
    std::filesystem::create_directories(fmt::format("history/[{}]-{}", pdesc, name));

    std::ofstream(fmt::format("history/[{}]-{}/full-config.json", pdesc, name)) << config;
    decision_model->tb.write_row_with_header(
        decision_model->tb.get_best_row(),
        fmt::format("history/[{}]-{}/best-result.csv", pdesc, name));

    if (fa != nullptr) {
        fa->tb.write(fmt::format("history/[{}]-{}/ssm-train-history.csv", pdesc, name));
    }
}
void TFATradingSystem::load(const std::string &name) {
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Loading backup {}...\n", name);
    read_config(SciLib::path_join(name, "full-config.json"));
    init_by_config();

    // Create model.
    create_ssm();
    generate_decision_model_data_pool();
    create_decision_model();

    // Load model.
    fmt::print("what are you doing.\n");
    fa->load(name);
    fmt::print("what are you doing xxc.\n");
    decision_model->load(name);
};

void TFATradingSystem::test(const std::string &backup) {
    SciLib::check_path_exists(backup);
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Running test on {}...\n", backup);

    load(backup);

    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Writing reconstructed data...\n");
    write_ssm_reconstructed_data();
    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Writing data_pool...\n");
    generate_decision_model_data_pool();

    fmt::print(fmt::fg(fmt::color::yellow), "[TFATradingSystem]: Run test on DecisonModel...\n");
    data_pool.to(decision_model->device);
    auto [perf, portfolio] =
        decision_model->test(data_pool.test_data, "test_performance_final_on_backup.csv");
    perf.report();
};
