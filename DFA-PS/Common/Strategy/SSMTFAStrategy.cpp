#pragma once
#include "SSMTFAStrategy.hpp"
#include <Windows.h>

std::tuple<std::shared_ptr<DecisionModelBase>, std::shared_ptr<DecisionModelBase>>
create_alpha_beta(boost::json::object &config, int assets) {
    std::string _alpha_model{
        json_string_to_string_view(config["strategy"].as_object()["alpha_model"].as_string())};
    std::string _beta_model{
        json_string_to_string_view(config["strategy"].as_object()["beta_model"].as_string())};
    bool allowing_short{config["allowing_short"].as_bool()};

    std::shared_ptr<DecisionModelBase> alpha_model, beta_model;
    boost::json::object decision_model_config = config["decision_model"].as_object();
    int decision_model_n_in =
        static_cast<int>(config["factor_model"].as_object()["hidden_dim"].as_int64());

    if (_alpha_model == "enrbf") {
        alpha_model = std::make_shared<ENRBF>(decision_model_config["alpha_model"].as_object(),
                                              decision_model_n_in, 1);
    } else if (_alpha_model == "mlp") {
        // alpha_model = std::make_shared<TrivalAlphaModelWrapper<MLP>>(
        //    decision_model_config["alpha_model"].as_object(), decision_model_n_in);
    };

    auto beta_model_config = decision_model_config["beta_model"].as_object();
    if (_beta_model == "enrbf") {
        if (!allowing_short) {
            beta_model = std::make_shared<ENRBF>(beta_model_config, decision_model_n_in, assets);
        } else {
            beta_model = std::make_shared<ENRBF>(beta_model_config, decision_model_n_in, assets);
        }
    } else if (_beta_model == "mlp") {
        /*if (!allowing_short) {
            beta_model = std::make_shared<TrivalBetaModelWrapper<MLP>>(beta_model_config,
                                                                       decision_model_n_in, assets);
        } else {
            beta_model = std::make_shared<AllowingShortBetaModelWrapper<MLP>>(
                beta_model_config, decision_model_n_in, assets);
        }*/
    } else if (_beta_model == "mlpn") {
        /*fmt::print("Using MLPN model.\n");
        if (!allowing_short) {
            beta_model = std::make_shared<TrivalBetaModelWrapper<MLPN>>(
                beta_model_config, decision_model_n_in, assets);
        } else {
            beta_model = std::make_shared<AllowingShortBetaModelWrapper<MLPN>>(
                beta_model_config, decision_model_n_in, assets);
        }*/
    };

    return make_tuple(alpha_model, beta_model);
};

SSMTFAStrategy::SSMTFAStrategy(boost::json::object &config, int assets,
                               std::shared_ptr<TFAModelBase> ssm,
                               std::shared_ptr<DecisionModelBase> alpha_model,
                               std::shared_ptr<DecisionModelBase> beta_model)
    : // topk(static_cast<int>(config["decision_model"].as_object()["topk"].as_int64())),
      ssm_hidden_dim(static_cast<int>(config["factor_model"].as_object()["hidden_dim"].as_int64())),
      step_epochs(static_cast<int>(config["decision_model"].as_object()["step_epochs"].as_int64())),
      decision_model_epochs(
          static_cast<int>(config["decision_model"].as_object()["epochs"].as_int64())),
      ssm_epochs(static_cast<int>(config["factor_model"].as_object()["epochs"].as_int64())),
      ssm_max_batch_size(
          static_cast<int>(config["factor_model"].as_object()["batch_size"].as_int64())),
      ssm(ssm), alpha_model(alpha_model), beta_model(beta_model),
      use_gpu(config["use_gpu"].as_bool()) {
    if (use_gpu)
        device = torch::kCUDA;
};

void SSMTFAStrategy::backup() {
    ssm->backup();
    alpha_model->backup();
    beta_model->backup();
};
void SSMTFAStrategy::restore() {
    ssm->restore();
    alpha_model->restore();
    beta_model->restore();
};

void SSMTFAStrategy::train_test(const Dataset &data) {

    auto &returns = data.train_returns;
    auto &test_returns = data.test_returns;

    std::cout << "Now training Factor model..." << std::endl;
    if (use_gpu) {
        // LoadLibraryA("torch_cuda.dll");
        // auto t = torch::tensor({5, 5}, torch::kCUDA);
        ssm->set_use_gpu();
    }
    auto best_ssm_epoch = ssm->train(returns, test_returns, ssm_epochs, ssm_max_batch_size);
    // config["ssm"].as_object()["epochs"] = best_ssm_epoch;

    std::cout << "Now training ENRBF model..." << std::endl;
    //  std:: std::vector<double> history_returns;
    std::cout << "Calculating hidden factors..." << std::endl;
    // Important: use predicted hidden factors.
    TMatf predicted_hidden_factors = ssm->compute_predicted_hidden_factors(returns).cast<float>();
    std::cout << predicted_hidden_factors.topRows(20) << std::endl;
    std::cout << "correlations between hidden factors:" << std::endl;
    auto [corr, cov] = corr_cov(predicted_hidden_factors);
    std::cout << corr << std::endl;

    fmt::print("train(): Initializing parameters for ENRBF model...\n");
    alpha_model->init_parm(predicted_hidden_factors);
    beta_model->init_parm(predicted_hidden_factors);
    if (use_gpu) {
        alpha_model->set_use_gpu();
        beta_model->set_use_gpu();
    }
    // Use predicted returns as hidden factor
    // auto hidden_factors = ssm.compute_predicted_returns(returns);
    // auto predicted_returns = ssm.compute_predicted_returns(returns);
    // cout << hidden_factors.topRows(50) << endl;
    // cout << predicted_returns << endl;
    // Convert a matrix to a  std::vector of Var
    fmt::print("Create train target function...\n");
    TMatf f_returns = returns.cast<float>();
    this->train_hidden_factors = mat_to_tensor(predicted_hidden_factors).to(device);
    this->train_returns = mat_to_tensor(f_returns).to(device);
    this->train_returns_nan_mask = mat_to_tensor(data.train_valid_mask).to(device);

    // std::cout << this->train_hidden_factors[0] << std::endl;

    //----------------------------------------------------------------------------------------------
    // Train decision model.
    fmt::print("Training models...\n");
    std::string decision_model_history_header =
        fmt::format("{:6} {:7} | {:7} {:7} {:7} {:7} | {:7} {:7} {:7}\n", "Epochs", "Target", "mu",
                    "risk", "sharpe", "Wealth", ".mu", ".risk", ".sharpe");

    int best_epoch = 1;
    Performance best_test_performance;

    std::ofstream decision_model_history("decision_model_history.csv");
    decision_model_history << decision_model_history_header;
    fmt::print("{}", decision_model_history_header);

    std::string decision_model_history_row;
    for (int i = 1; i <= decision_model_epochs; ++i) {
        // cout << "Epoch: " << i << endl;
        // cout << "Before training: " << alpha_model->_parms[0]->val() << endl;
        auto [X, P, mask] = data_pool->sample();

        auto target = train_epoch(X, P, mask);
        // cout << "After training: " << alpha_model->_parms[0]->val() << endl;
        /*if (i == decision_model_epochs) {
            target_gm.save("train_graph.json");
        }*/
        if (i % 2 == 1) {

            // update_decision_model = false;
            // update_ssm_model = false;
            //// Don't forget to reset initial hidden factors.
            //// cout << "Now test on train set." << endl;
            // ssm->before_epoch();
            // auto [train_performance, train_portfolios] =
            //     test(returns, data.train_valid_mask, i == decision_model_epochs, "train.csv");
            // write_portfolios(train_portfolios, "train.csv");

            // cout << "Now test on test set." << endl;
            update_decision_model = false;
            update_ssm_model = false;
            auto [test_performance, test_portfolios] =
                test(test_returns, data.test_valid_mask, i == decision_model_epochs, "test.csv");

            if (test_performance.sharpe > best_test_performance.sharpe) {
                best_epoch = i;
                best_test_performance = test_performance;

                config["decision_model"].as_object()["epochs"] = best_epoch;
                write_portfolios(test_portfolios, "best_test.csv");
            }
            // cout << "Post train and test." << endl;

            /* decision_model_history_row = fmt::format(
                "{:6} {:7.4f} | {:7.4f} {:7.4f} {:7.4f} {:7.4f}| {:7.4f} {:7.4f} {:7.4f}\n", i,
                -target.item<double>(), train_performance.mean, train_performance.risk,
                train_performance.sharpe,
                train_portfolios.coeff(train_portfolios.rows() - 1, train_portfolios.cols() - 1),
                test_performance.mean, test_performance.risk, test_performance.sharpe);*/
            decision_model_history_row =
                fmt::format("{:6} {:7.4f} | {:7.4f} {:7.4f} {:7.4f}\n", i, -target.item<double>(),
                            test_performance.mean, test_performance.risk, test_performance.sharpe);

        } else {
            decision_model_history_row =
                fmt::format("{:6} {:7.4f} | {: ^7} {: ^7} {: ^7} {: ^7}| {: ^7} {: ^7} {: ^7}\n", i,
                            -target.item<double>(), "", "", "", "", "", "", "");
        }
        fmt::print("{}", decision_model_history_row);
        decision_model_history << decision_model_history_row;
        // print("{}\n", train_performance.sharpe);
    };
    fmt::print("All finished. Best epoch: {}, best sharpe:{}, MDD: {}\n", best_epoch,
               best_test_performance.sharpe, best_test_performance.mdd);
    // info("Final wealth in train epoch: {}\n", )
    // save_best_config();
};

void SSMTFAStrategy::begin_test(){
    // test_hidden_factors.clear();
    // test_returns.clear();
};
torch::Tensor SSMTFAStrategy::train_epoch(const torch::Tensor &factors,
                                          const torch::Tensor &returns,
                                          const torch::Tensor &valid_mask) {
    auto target = calculate_performance(factors, returns, valid_mask);
    alpha_model->clean();
    beta_model->clean();
    // std::cout << target << " grad: " << target.requires_grad() << std::endl;
    target.backward();
    alpha_model->update();
    beta_model->update();

    // target = calculate_performance(factors, returns);
    return target;
};

std::pair<double, TVecd> SSMTFAStrategy::calculate_portfolio(const TVecd &x) {
    torch::Tensor xi = mat_to_tensor(x.cast<float>().transpose().eval()).to(device); // 1*n_in
    auto alpha = alpha_wrapper->forward(alpha_model->forward(xi));
    auto beta = beta_wrapper->forward(beta_model->forward(xi));
    // std::cout << "tx: " << tx << std::endl;
    return {alpha[0][0].item<double>(), tensor_to_mat<double>(beta).transpose()};
};

std::pair<double, TVecd> SSMTFAStrategy::step(const TVecd &y, const TVec<bool> &nan_mask) {
    auto predicted_hidden_factor = ssm->predict();
    auto portfolio = calculate_portfolio(predicted_hidden_factor.first);

    // std::cout << "fac in test: " << pred.first.transpose() << std::endl;
    // std::cout << "portfolio in test: " << portfolio.first << " " << portfolio.second <<
    // std::endl;
    if (update_decision_model) {
        // test_hidden_factors.emplace_back(last_ssm_perdicted_hidden_factor);
        // last_ssm_perdicted_hidden_factor = mat_to_tensor(pred.first.cast<float>().eval());

        // test_returns.emplace_back(mat_to_tensor(y.cast<float>().eval()));

        if ((test_hidden_factors.size(0) > 2) && (step_epochs > 0)) {
            // step_update();
        };
    };
    ssm->update_hidden_factor(y);
    if (update_ssm_model) {
        ssm->update_parms();
    };

    return portfolio;
};

void SSMTFAStrategy::post_update_decision(){};