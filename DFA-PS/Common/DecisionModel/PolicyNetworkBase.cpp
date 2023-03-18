#include "stdafx.h"
#include "PolicyNetworkBase.hpp"

PolicyNetworkBase::PolicyNetworkBase(boost::json::object &config, int window, int n_in, int n_out,
                                     int asset_feature_size, int market_feature_size)
    : DecisionModelBase(config, window, n_in, n_out, asset_feature_size, market_feature_size),
      pretrain_option(config["pretrain"].as_object()),
      metric_fn(create_metric_fn(SciLib::json_string_to_string(config["metric"].as_string()))) {}

std::pair<torch::Tensor, PortfolioPerformance>
PolicyNetworkBase::train_epoch(const DecisionModelDataSet &data) {
    // before_epoch();
    set_train();
    opt->zero_grad();
    auto [alpha, beta] = calculate_portfolio_weight(data);

    int N = alpha.size(0);
    // auto portfolio = cal.calculate_returns(alpha, beta, data.prices, data.dates);
    train_portfolio = cal.calculate_returns(alpha, beta, data.prices, data.dates);

    auto target =
        -metric_fn(train_portfolio.returns, train_portfolio.wealth); // Remember to negate.

    target.backward();
    opt->step();

    return {target,
            calculate_performance(train_portfolio.wealth.detach_(),
                                  train_portfolio.returns.detach_(), data_period, r_f, r_f)};
}

void PolicyNetworkBase::pretrain(const DataPool &data_pool, const std::string &id) {
    before_pretrain();

    if (pretrain_option.use) {
        fmt::print("[DecisionModel]: Pretrain using lR: {}\n", pretrain_option.lr);
        std::filesystem::create_directories(fmt::format("portfolios/{}", id));
        pretrain_tb.set_file(fmt::format("history/{}/pretrain_history.csv", id));
        pretrain_tb.print_header();
        pretrain_tb.write_header();

        DecisionModelPretrainDataSet data;
        for (int i = 1; i <= pretrain_option.epochs; ++i) {
            pretrain_tb.new_row();

            if (i % pretrain_option.resample_interval == 1) {
                data = data_pool.sample_pretrain_dataset(pretrain_option.period);
                data.nan_mask.logical_not_();
            }

            pretrain_opt->zero_grad();

            // Calculate weight.
            torch::Tensor basis_weight = cal_score(data.factor_input, data.asset_feature);

            // std::terminate();
            auto target = torch::nn::functional::mse_loss(
                // basis_weight * data.nan_mask,
                basis_weight,
                data.real_returns * 25); // We want to approximate weight rank, so
                                         // multiply 25 to amplify differences.
            target.backward();

            pretrain_opt->step();

            pretrain_tb["Epoch"] = i;
            pretrain_tb["MSE"] = target.item<double>();

            pretrain_tb.print_row();
            pretrain_tb.write_row();
        }
        pretrain_opt.reset();
    }

    after_pretrain();
}

void PolicyNetworkBase::before_pretrain() {
    if (pretrain_option.use) {
        // Table
        pretrain_tb = SciLib::TableWriter("Epoch", "MSE");
        // pretrain_tb.set_file("pretrain_history.csv");
        pretrain_tb.set_col_type(1, "Epoch");
        pretrain_tb.set_col_formatter("{: >5}", "{: ^5}", "Epoch");
        pretrain_tb.set_col_formatter("{: >9.5f}", "{: ^9}", "MSE");
    }
}

void PolicyNetworkBase::save(const std::string &name, const std::string &sub_name, bool info) {
    DecisionModelBase::save(name, sub_name, info);

    // save optimizer
    torch::serialize::OutputArchive archive;
    opt->save(archive);
    archive.save_to(fmt::format("backup/{}/DecisionModel_NetOptimizer.pt", name));
}
void PolicyNetworkBase::load(const std::string &name) {
    DecisionModelBase::load(name);
    // optimizer.
    fmt::print(fmt::fg(fmt::color::yellow), "[PolicyNetworkBase]: Loading optimizer...\n");
    torch::serialize::InputArchive iarch;
    iarch.load_from(SciLib::path_join(name, "DecisionModel_NetOptimizer.pt"));
    opt->load(iarch);
}
BetaWeightWrapperBase::BetaWeightWrapperBase(){};
BetaWeightWrapperBase::BetaWeightWrapperBase(int topk, int assets) : topk(topk), assets(assets){};

// Fill weight with nan_mask to be -inf.
torch::Tensor BetaWeightWrapperBase::process_nan_mask(const torch::Tensor &ori_weight,
                                                      const torch::Tensor &nan_mask,
                                                      const torch::Tensor &val) {
    return ori_weight.index_put({nan_mask}, val);
}
// Here weight is the weight processed by process_nan_mask.
// [Warning]: there is a bug when topk < valid assets. But currently I have no
// good method to deal with it.
torch::Tensor BetaWeightWrapperBase::select_topk(const torch::Tensor &weight) {
    auto N = weight.size(0);
    if (topk < assets) {
        auto [sel_weight, sel_ind] = weight.topk(topk, -1, true, false);
        auto w = torch::zeros({N, assets})
                     .to(weight.device())
                     .scatter_(1, sel_ind,
                               torch::nn::functional::softmax(
                                   sel_weight, torch::nn::functional::SoftmaxFuncOptions(1)));
        return w;
    } else {
        return weight;
    }
}

TrivialAlphaModelWrapper::TrivialAlphaModelWrapper(const boost::json::object &config)
    : min_v(config.at("min_value").as_double()){};
torch::Tensor TrivialAlphaModelWrapper::forward(const torch::Tensor &ori_weight) {
    // auto xi = process_missing_price(ori_weight, returns);
    return min_v + (1 - min_v) * torch::sigmoid(ori_weight);
};

TrivialBetaModelWrapper::TrivialBetaModelWrapper(const boost::json::object &config, int assets)
    : BetaWeightWrapperBase(static_cast<int>(config.at("topk").as_int64()), assets){};

SciLib::FullBetaPortfolioWeight TrivialBetaModelWrapper::forward(const torch::Tensor &ori_weight) {
    return {select_topk(ori_weight)};
}

SciLib::FullBetaPortfolioWeight TrivialBetaModelWrapper::forward(const torch::Tensor &ori_weight,
                                                                 const torch::Tensor &nan_mask) {
    // auto xi = BetaModel::forward(X);
    auto xi = process_nan_mask(ori_weight, nan_mask,
                               torch::tensor({-std::numeric_limits<float>::infinity()}));

    return forward(xi);
};

AllowingShortBetaModelWrapper::AllowingShortBetaModelWrapper(const boost::json::object &config,
                                                             int n_out)
    : BetaWeightWrapperBase(static_cast<int>(config.at("topk").as_int64()), n_out), nv(1.0 / topk) {
    fmt::print("[AllowingShortBetaModelWrapper]: topk {}, selected assets {}\n", topk, 2 * topk);
};

SciLib::FullBetaPortfolioWeight
AllowingShortBetaModelWrapper::forward(const torch::Tensor &ori_weight,
                                       const torch::Tensor &nan_mask) {
    auto N = ori_weight.size(0);
    // auto wi = process_nan_mask(ori_weight, nan_mask);
    auto long_ori_weight = process_nan_mask(
        ori_weight, nan_mask, torch::tensor({-std::numeric_limits<float>::infinity()}));
    auto [long_sel_weight, long_sel_ind] = long_ori_weight.topk(topk, -1, true, false);
    auto long_weight =
        torch::zeros({N, assets})
            .to(ori_weight.device())
            .scatter_(1, long_sel_ind,
                      torch::nn::functional::softmax(long_sel_weight,
                                                     torch::nn::functional::SoftmaxFuncOptions(1)));

    auto short_ori_weight = -process_nan_mask(
        ori_weight, nan_mask, torch::tensor({std::numeric_limits<float>::infinity()}));
    auto [short_sel_weight, short_sel_ind] = short_ori_weight.topk(topk, -1, true, false);
    auto short_weight =
        torch::zeros({N, assets})
            .to(ori_weight.device())
            .scatter_(1, short_sel_ind,
                      torch::nn::functional::softmax(short_sel_weight,
                                                     torch::nn::functional::SoftmaxFuncOptions(1)));

    return {long_weight, short_weight, long_sel_weight, short_sel_weight};
};

PreTrainOptions::PreTrainOptions(boost::json::object &config)
    : lr(config["lr"].as_double()), epochs(static_cast<int>(config["epochs"].as_int64())),
      resample_interval(static_cast<int>(config["resample_interval"].as_int64())),
      period(static_cast<int>(config["period"].as_int64())), use(config["use"].as_bool()) {}
