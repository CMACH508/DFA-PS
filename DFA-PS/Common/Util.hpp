#pragma once

#include "SciLib/EigenHelper.hpp"
#include "SciLib/JsonHelper.hpp"
#include "SciLib/STDHelper.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <rapidcsv.h>
#include <torch/torch.h>

void write_portfolios(const TMatd &portfolios, const std::string &filename);

struct Performance {
    double mean = 0;
    double risk = 0;
    double sharpe = 0;
    double mdd = 0;
};

template <typename T> double sharpe_ratio(const T &returns) {
    double mean = returns.mean();
    return mean / sqrt(returns.array().pow(2).mean() - mean * mean);
};

template <typename T> Performance cal_performance(const T &returns) {
    // int N = returns.size();
    double mean = returns.mean();
    double risk = sqrt(returns.array().pow(2).mean() - mean * mean);
    // print("Mean return: {:6.4f}, risk: {:6.4f}, sharpe_ratio: {:6.4f}\n", mean, risk, mean /
    // risk);
    double mdd = std::accumulate(
        returns.begin(), returns.end(), 0.0,
        [max_v = double(returns.coeff(0))](const double &old_mdd, const double &b) mutable {
            double mdd = (max_v - b) / max_v;
            max_v = std::max(max_v, b);
            return std::max(old_mdd, mdd);
        });
    return Performance{mean, risk, mean / risk, mdd};
};

struct DataSet {
    int assets;
    TMatd train_close, test_close;
    TMatd train_returns, test_returns; // Note the first row of returns is always nan, don't use it.
    TMat<bool> train_valid_mask, test_valid_mask;
    std::vector<int> train_dates, test_dates;

    // 1d features: (Days, Assets, Channel)
    torch::Tensor train_asset_features, test_asset_features;

    std::vector<int> remove_abnormal_assets(double max_v = 10, double min_v = -0.9,
                                            double max_price_change = 100);
    void validate();
};

// Note nan price will be replaced by EPS(1e-10).
DataSet read_dataset(const std::string &path, double scale_returns,
                     bool remove_abnormal_assets = true);

struct DecisionModelDataSet {
    //(N, W, A)
    torch::Tensor factor_input;
    //(N, A, C)
    torch::Tensor asset_feature, market_feature;
    //(N, A)
    torch::Tensor prices, price_nan_mask;
    torch::Tensor dates;

    void to(const torch::Device &device);
    void validate(bool info = true);
    void normalize();
    void report_size();
};
struct DecisionModelPretrainDataSet {
    //(N, W, A)
    torch::Tensor factor_input;
    //(N, A, C)
    torch::Tensor asset_feature, market_feature;
    //(N,A)
    torch::Tensor real_returns;
    torch::Tensor nan_mask;

    void normalize();
    void to(const torch::Device &device);
};
template <typename T> void static inline normalize(T &data) {
    // The last dim is window.
    data.asset_feature = data.asset_feature / (data.asset_feature.slice(3, 0, 1) + 0.5);
    // data.asset_feature = data.asset_feature / (data.asset_feature.slice(3, w-1, w) + 1);
}

struct DataPool {
    // 2D: (n, hidden_dim);
    torch::Tensor train_recovered_hidden_factors, test_recovered_hidden_factors;
    // 2D: (n, dim)
    torch::Tensor train_returns, test_returns;
    torch::Tensor train_prices, test_prices;
    torch::Tensor train_nan_mask, test_nan_mask; // Here is nan mask, 1 for nan.
    // 3D: (n, window, hidden_dim)
    torch::Tensor train_predicted_hidden_factors_pool, test_predicted_hidden_factors_pool;
    // 1D
    torch::Tensor train_dates_all, test_dates_all, test_dates;
    // 3D: (N, A, C)
    torch::Tensor train_asset_features, test_asset_features;
    // 2D
    torch::Tensor train_market_features, test_market_features;

    DecisionModelDataSet test_data;

    int holding_period;
    int window_size;
    int factor_dim;
    int num_market_feature, num_asset_feature;
    bool only_use_periodic_price;
    bool use_forward_window; // If false, than use history hidden factors.
    //(n, window, hidden_dim)
    // torch::Tensor test_input, test_input_mask;
    int max_batch_size = 32;
    torch::Device device = torch::kCPU;

    DataPool();
    DataPool(const DataSet &data, torch::Device, int holding_period, bool only_use_periodic_price,
             const std::string &window_type, int window_size, int max_batch_size = 32);
    //
    void construct_market_features();
    // train_input, train_prices
    // train_input: (n, window, hidden_dim)
    DecisionModelDataSet sample_train_dataset() const;
    DecisionModelPretrainDataSet sample_pretrain_dataset(int period) const;

    torch::Tensor sample_factor_input(int start, int end, int period) const;
    torch::Tensor sample_asset_feature(int start, int end, int period) const;
    torch::Tensor sample_market_feature(int start, int end, int period) const;
    // price, date, nan_mask
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_price_date(int start, int end,
                                                                              int period) const;

    // If only_use_periodic_price, then after this call, test_prices if not full and price between
    // two consecutive points is discarded.
    void construct_test_dataset();
    void read_precomputed_hidden_factor(const std::string &path);
    void to(const torch::Device &device);

    void report_size();
    void validate() const;
};

template <typename T> TMat<bool> generate_mask(const T &mat) {
    TMat<bool> res = TMat<bool>::Constant(mat.rows(), mat.cols(), true);
    // res.setConstant(1);
    int m = static_cast<int>(mat.rows()), n = static_cast<int>(mat.cols());
    for (int j = 0; j < n; ++j) {
        // Find the first non missing value.
        int start = 0, end = m - 1;
        for (; start < m; ++start) {
            if (!std::isnan(mat.coeffRef(start, j))) {
                break;
            }
        }
        // Find the last non-non index.
        for (; end > 0; --end) {
            if (!std::isnan(mat.coeffRef(end, j))) {
                break;
            }
        }
        // Fill mask.
        if (start <= end) {
            for (int i = start; i <= end; ++i) {
                res.coeffRef(i, j) = false;
            }
        }
    }
    return res;
};

void report_dataset(const DataSet &dataset);

inline boost::json::object rewrite_config_paths(const boost::json::object &config_) {
    boost::json::object config = config_;
    config["data_path"] =
        SciLib::absolute_path(SciLib::json_string_to_string(config["data_path"].as_string()));
    if (config["decision_model"].as_object()["use_precomputed_hidden_factor"].as_bool()) {
        auto &obj_p = config["decision_model"].as_object()["precomputed_hidden_factor_path"];
        obj_p = SciLib::absolute_path(SciLib::json_string_to_string(obj_p.as_string()));
    }
    return config;
};
