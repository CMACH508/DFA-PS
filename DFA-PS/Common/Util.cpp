#include "stdafx.h"
#include "Util.hpp"
#include "SciLib/EigenTorchHelper.hpp"
#include "SciLib/STDHelper.hpp"
#include "SciLib/Finance.hpp"

void write_portfolios(const TMatd &portfolios, const std::string &filename) {
    size_t N = portfolios.rows(), C = portfolios.cols(), d = (C - 2) / 2;

    std::ofstream file(filename);
    for (size_t i = 0; i < d; ++i) {
        file << "R" << std::to_string(i) << ",";
    };
    for (size_t i = 0; i < d; ++i) {
        file << "W" << std::to_string(i) << ",";
    };
    file << "Investment Ratio,Portfolio Return,Wealth" << std::endl;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < C - 1; ++j) {
            file << portfolios.coeff(i, j) << ",";
        };
        file << portfolios.coeff(i, C - 1) << std::endl;
    }
}

void DataSet::validate() {
    if (train_asset_features.size(0) != train_returns.rows())
        fmt::print(fmt::fg(fmt::color::red),
                   "Length of train_asset_features is not equal to the train_returns\n");
    if (test_asset_features.size(0) != test_returns.rows())
        fmt::print(fmt::fg(fmt::color::red),
                   "Length of train_asset_features is not equal to the train_returns\n");

    std::vector<long long> asset_sz{train_asset_features.size(1), test_asset_features.size(1),
                                    train_returns.cols(), test_returns.cols()};
    if (!std::equal(asset_sz.cbegin() + 1, asset_sz.cend(), asset_sz.cbegin()))
        fmt::print(fmt::fg(fmt::color::red), "Assets size not equal: {}\n", asset_sz);
}

std::vector<int> DataSet::remove_abnormal_assets(double max_v, double min_v,
                                                 double max_price_change) {
    // Delete blank columns in train dataset.
    std::set<int> selected_index_set;
    fmt::print(fmt::fg(fmt::color::yellow), "[DataSet]: Removing abnormal assets in data.\n");
    // Check returns.
    for (int j = 0; j < train_returns.cols(); ++j) {
        bool train_valid =
            SciLib::vec_valid(TVecd(train_returns.col(j)), 100, max_v, min_v, std::to_string(j));
        bool test_valid =
            SciLib::vec_valid(TVecd(test_returns.col(j)), 100, max_v, min_v, std::to_string(j));
        if (train_valid && test_valid) {
            selected_index_set.insert(j);
        }
    }
    // Remove assets that has very large growth in test dataset.
    for (int j = 0; j < train_close.cols(); ++j) {
        double start = 1, end = 1;
        if (selected_index_set.find(j) != selected_index_set.end()) {
            // Find first valid price.
            for (int i = 0; i < test_close.rows(); ++i) {
                if (test_valid_mask.coeff(i, j)) {
                    start = test_close.coeff(i, j);
                    break;
                }
            }
            // Find last valid price in test data.
            for (auto i = test_close.rows() - 1; i >= 0; --i) {
                if (test_valid_mask.coeff(i, j)) {
                    end = test_close.coeff(i, j);
                    break;
                }
            }
            // std::cout << end / start << std::endl;
            if ((end / start > max_price_change) || (start / end) > max_price_change) {
                fmt::print(fmt::fg(fmt::color::red),
                           "Vector {}: initial {}, final {}. Price ratio is larger than "
                           "{}, remove it.\n",
                           j, start, end, max_price_change);
                selected_index_set.erase(j);
            }
        }
    }
    std::vector<int> selected_index(selected_index_set.begin(), selected_index_set.end());
    std::sort(selected_index.begin(), selected_index.end());
    if (selected_index.size() < assets) {
        assets = selected_index.size();
        train_close = TMatd(train_close(Eigen::indexing::all, selected_index));
        test_close = TMatd(test_close(Eigen::indexing::all, selected_index));
        train_returns = TMatd(train_returns(Eigen::indexing::all, selected_index));
        test_returns = TMatd(test_returns(Eigen::indexing::all, selected_index));
        train_valid_mask = TMat<bool>(train_valid_mask(Eigen::indexing::all, selected_index));
        test_valid_mask = TMat<bool>(test_valid_mask(Eigen::indexing::all, selected_index));

        train_asset_features =
            train_asset_features.index_select(1, torch::tensor(selected_index)).clone().detach();
        test_asset_features =
            test_asset_features.index_select(1, torch::tensor(selected_index)).clone().detach();
    }
    fmt::print("[DataSet]: Final number of assets: {}.\n", assets);
    return selected_index;
};

DataSet read_dataset(const std::string &path, double scale_returns, bool remove_abnormal_assets) {
    fmt::print("Now reading data {}\n", path);

    auto config_path = std::filesystem::path(path) / std::filesystem::path("config.json");
    fmt::print("Data config path: {}\n", std::filesystem::absolute(config_path).string());
    SciLib::check_path_exists(config_path.string());

    auto config = SciLib::read_config(config_path.string());

    auto close_data_file =
        std::filesystem::path(path) /
        std::filesystem::path(SciLib::json_string_to_string(config["adjusted_price"].as_string()));
    fmt::print("Price file path: {}\n", std::filesystem::absolute(close_data_file).string());

    SciLib::check_path_exists(close_data_file.string());

    auto read_file = [&](const std::string &filename) -> std::pair<rapidcsv::Document, TMatd> {
        rapidcsv::Document doc(filename, rapidcsv::LabelParams(0, 0), rapidcsv::SeparatorParams(),
                               rapidcsv::ConverterParams(true));

        size_t assets = doc.GetColumnCount(), N = doc.GetRowNames().size();
        TMatd full = TMatd::Constant(assets, N, std::numeric_limits<double>::quiet_NaN());

        for (int i = 0; i < assets; ++i) {
            std::vector<double> col = doc.GetColumn<double>(i);
            /* for (int j = 0; j < N; ++j) {
                 full.coeffRef(j, i) = col[j];
             }*/
            std::copy(col.begin(), col.end(), full.data() + i * N);
        }
        full.transposeInPlace();

        return {doc, full};
    };

    // Read price.
    auto [price_doc, price] = read_file(close_data_file.string());
    int assets = price.cols();

    // Dates
    std::vector<std::string> _all_dates = price_doc.GetRowNames();
    std::vector<int> all_dates;
    for (auto &ele : _all_dates) {
        all_dates.emplace_back(boost::lexical_cast<int>(ele));
    }
    int N = all_dates.size();

    // Calculate returns data
    price = (price.array() == 0)
                .select(std::numeric_limits<double>::quiet_NaN(), price); // Replace 0 price to nan.
    SciLib::forward_fillna(price);
    TMat<bool> price_mask = 1 - price.array().isNaN();
    TMatd returns =
        TMatd::Constant(price.rows(), price.cols(), std::numeric_limits<double>::quiet_NaN());
    returns.bottomRows(N - 1) =
        (price.bottomRows(N - 1).array() / price.topRows(N - 1).array() - 1).matrix();
    returns *= scale_returns;
    SciLib::fillna(price, 1e-10); // Fill nan price to 0.
    SciLib::fillna(returns, 0.0); // Fill nan returns to 0.
    fmt::print("File read...\n");

    auto split_it =
        std::upper_bound(all_dates.begin(), all_dates.end(), config["split"].as_int64());
    auto train_dates = std::vector(all_dates.begin(), split_it);
    auto test_dates = std::vector(split_it, all_dates.end());
    int N1 = train_dates.size(), N2 = test_dates.size();

    DataSet res;
    res.assets = assets;
    res.train_dates = train_dates;
    res.test_dates = test_dates;
    res.train_close = price.topRows(N1);
    res.test_close = price.bottomRows(N2);
    res.train_returns = returns.topRows(N1);
    res.test_returns = returns.bottomRows(N2);
    res.train_valid_mask = price_mask.topRows(N1);
    res.test_valid_mask = price_mask.bottomRows(N2);
    //
    auto feature_file = SciLib::absolute_path(SciLib::path_join(
        path, SciLib::json_string_to_string(config.at("asset_features").as_string())));

    fmt::print(fmt::fg(fmt::color::yellow), "Loading asset features {}\n", feature_file);
    torch::Tensor asset_features;
    torch::load(asset_features, feature_file);
    asset_features.detach_();
    asset_features.nan_to_num_(0.0); // Replace nan.
    std::cout << "Full asset features: " << asset_features.sizes() << std::endl;
    res.train_asset_features = asset_features.narrow(0, 0, N1).clone().detach();
    res.test_asset_features = asset_features.narrow(0, N1, N2).clone().detach();

    res.validate();
    std::vector<int> selected;
    if (remove_abnormal_assets) {
        selected = res.remove_abnormal_assets(scale_returns * 5, scale_returns * -0.9, 100);
    } else {
        selected.resize(res.assets);
        std::iota(selected.begin(), selected.end(), 0);
    }
    SciLib::write_vector(selected, SciLib::path_join(path, "selected_index.csv"));
    res.validate();

    fmt::print("DataSet has been read.\n");
    fmt::print(fmt::fg(fmt::color::yellow), "Total assets: {}\n", selected.size());

    return res;
};

void report_dataset(const DataSet &dataset) {
    fmt::print("{:-^100}\n", "");
    fmt::print("Total assets: {}\n", dataset.assets);
    fmt::print("Total rows: {}\n", dataset.train_dates.size() + dataset.test_dates.size());
    fmt::print("Train data range: {} to {}, total rows: {}\n", dataset.train_dates.front(),
               dataset.train_dates.back(), dataset.train_dates.size());
    fmt::print("Test data range: {} to {}, total rows: {}\n", dataset.test_dates.front(),
               dataset.test_dates.back(), dataset.test_dates.size());
    fmt::print("NaN rows percentage in train dataset: {:5.2f}%\n",
               static_cast<double>(dataset.train_valid_mask.rowwise().any().sum()) /
                   dataset.train_returns.rows() * 100);
    fmt::print("NaN rows percentage in test dataset: {:5.2f}%\n",
               static_cast<double>(dataset.test_valid_mask.rowwise().any().sum()) /
                   dataset.test_returns.rows() * 100);
    fmt::print("{:-^100}\n", "");
};

DataPool::DataPool(){};
DataPool::DataPool(const DataSet &data, torch::Device device, int holding_period,
                   bool only_use_periodic_price, const std::string &window_type, int window_size,
                   int max_batch_size)
    : holding_period(holding_period), only_use_periodic_price(only_use_periodic_price),
      use_forward_window(window_type == "forward" ? true : false), window_size(window_size),
      device(device), max_batch_size(max_batch_size) {
    fmt::print(fmt::fg(fmt::color::yellow), "[DataPool]: max batch size: {}\n",
               this->max_batch_size);
    /*TMatf mat = data.train_close.cast<float>().eval();
    std::cout << mat.row(0) << std::endl;
    auto t = torch::from_blob(mat.data(), {mat.rows(), mat.cols()});
    std::cout << t[0] << std::endl;*/

    train_prices =
        SciLib::mat_to_tensor(data.train_close.cast<float>().eval()).to(device).detach_();
    test_prices = SciLib::mat_to_tensor(data.test_close.cast<float>().eval()).to(device).detach_();
    train_returns =
        SciLib::mat_to_tensor(data.train_returns.cast<float>().eval()).to(device).detach_();
    test_returns =
        SciLib::mat_to_tensor(data.test_returns.cast<float>().eval()).to(device).detach_();

    train_nan_mask =
        SciLib::mat_to_tensor(data.train_valid_mask).logical_not_().to(device).detach_();
    test_nan_mask = SciLib::mat_to_tensor(data.test_valid_mask).logical_not_().to(device).detach_();
    int N1 = data.train_dates.size(), N2 = data.test_dates.size();
    train_dates_all =
        torch::from_blob(const_cast<int *>(data.train_dates.data()), {N1}, torch::kInt32)
            .to(device)
            .detach_();
    test_dates_all =
        torch::from_blob(const_cast<int *>(data.test_dates.data()), {N2}, torch::kInt32)
            .to(device)
            .detach_();
    train_asset_features = data.train_asset_features;
    test_asset_features = data.test_asset_features;
    num_asset_feature = train_asset_features.size(2);
    // Compute market features.
    construct_market_features();
};

void DataPool::construct_market_features() {
    fmt::print(fmt::fg(fmt::color::yellow), "[DataPool]: constructing market features...\n");

    auto full_prices = torch::cat({train_prices, test_prices});
    auto full_valid_mask = torch::cat({train_nan_mask, test_nan_mask}).logical_not_();
    auto full_returns = torch::cat({train_returns, test_returns});

    num_market_feature = 4;
    auto full_market_features = torch::cat(
        {(full_prices.sum(1) / full_valid_mask.sum(1)).unsqueeze(1),
         SciLib::cal_ADR(full_returns).unsqueeze(1), SciLib::cal_OBOS(full_returns).unsqueeze(1),
         SciLib::cal_ADL(full_returns).unsqueeze(1)},
        1);
    std::cout << "full market: " << full_market_features.sizes() << std::endl;
    auto N1 = train_prices.size(0), N2 = test_prices.size(0);
    train_market_features = full_market_features.narrow_copy(0, 0, N1);
    test_market_features = full_market_features.narrow_copy(0, N1, N2);
}

// train_input, train_prices are all holding_period-inverval data (not daily).
// length(train_input) = length(train_prices).
DecisionModelDataSet DataPool::sample_train_dataset() const {
    // The first row is in the first half of the whole dataset.
    /*auto start =
        torch::randint(holding_period, static_cast<int>(train_prices.size(0) / 2), {1}).item<int>();
        int end = train_prices.size(0);*/

    int min_start = static_cast<int>(train_prices.size(0) - holding_period * max_batch_size);
    int start, end;
    if (min_start < holding_period) {
        // If data length is less then max_batch_size, than batch size is half of total data.
        /*fmt::print(
            fmt::fg(fmt::color::red),
            "Total data is less than max_batch_size, batch size if set to half of all data.");*/
        start = torch::randint(holding_period, train_prices.size(0) / 2, {1}).item<int>();
        end = start + train_prices.size(0) / 2;
    } else {
        start = torch::randint(holding_period, min_start, {1}).item<int>();
        end = start + holding_period * max_batch_size;
    }

    // fmt::print("Start: {}, holding_period: {}\n", start, holding_period);
    auto [train_prices, train_dates, train_input_mask] =
        sample_price_date(start, end, holding_period);

    // const int real_end = start + (train_prices.size(0) - 1) * holding_period + 1;
    int real_end = end;

    const torch::Tensor train_input = sample_factor_input(start, real_end, holding_period);
    const torch::Tensor train_asset_feature = sample_asset_feature(start, real_end, holding_period);
    const torch::Tensor train_market_feature =
        sample_market_feature(start, real_end, holding_period);

    DecisionModelDataSet train_data{train_input,  train_asset_feature, train_market_feature,
                                    train_prices, train_input_mask,    train_dates};
    train_data.validate(false);
    // train_data.normalize();
    return train_data;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DataPool::sample_price_date(int start, int end, int period) const {
    torch::Tensor train_prices, train_dates;
    if (only_use_periodic_price) {
        train_prices = this->train_prices.slice(0, start, end, period).clone().detach_();
        train_dates = this->train_dates_all.slice(0, start, end, period).clone().detach_();
    } else {
        train_prices = this->train_prices.slice(0, start, end).clone().detach_();
    }

    torch::Tensor train_nan_mask =
        this->train_nan_mask.slice(0, start, end, period).clone().detach_();
    return {train_prices, train_dates, train_nan_mask};
}
torch::Tensor DataPool::sample_factor_input(int start, int end, int period) const {
    torch::Tensor train_input;
    if (use_forward_window) {
        train_input =
            train_predicted_hidden_factors_pool.slice(0, start, end, period).clone().detach_();
    } else {
        // Remember to correctly calculate end. (train_prices.size(0) - 1) * holding_period is the
        // last availiable index.
        train_input = train_recovered_hidden_factors.slice(0, start - window_size + 1, end)
                          .unfold(0, window_size, period)
                          .transpose(1, 2)
                          .clone()
                          .detach_();
    }
    return train_input;
}
torch::Tensor DataPool::sample_asset_feature(int start, int end, int period) const {
    return train_asset_features.slice(0, start - window_size + 1, end)
        .unfold(0, window_size, period)
        .transpose(1, 2)
        .clone()
        .detach_();
}
torch::Tensor DataPool::sample_market_feature(int start, int end, int period) const {
    return train_market_features.slice(0, start - window_size + 1, end)
        .unfold(0, window_size, period)
        .clone()
        .detach_();
}

DecisionModelPretrainDataSet DataPool::sample_pretrain_dataset(int period) const {
    int end = train_prices.size(0);
    auto start =
        torch::randint(period, static_cast<int>(train_prices.size(0) / 2), {1}).item<int>();

    auto [train_prices1, _1, train_nan_mask1] =
        sample_price_date(start, end - holding_period, period);
    auto [train_prices2, _2, train_nan_mask2] =
        sample_price_date(start + holding_period, end, period);
    // std::cout << "price size 1： " << train_prices1.size(0) << std::endl;
    // std::cout << "price size 2： " << train_prices1.size(0) << std::endl;

    auto real_returns =
        ((train_prices2 / train_prices1 - 1) * train_nan_mask1.logical_not_()).clone().detach_();

    // Here use -2 to sample (N-1) length data.
    const int real_end = start + (train_prices1.size(0) - 1) * period + 1;

    const torch::Tensor train_input = sample_factor_input(start, real_end, period);
    const torch::Tensor train_asset_feature = sample_asset_feature(start, real_end, period);
    const torch::Tensor train_market_feature = sample_market_feature(start, real_end, period);

    DecisionModelPretrainDataSet data{train_input, train_asset_feature, train_market_feature,
                                      real_returns, train_nan_mask1};
    // data.normalize(); //Normalize will lead to -inf for some features. So not using it.
    return data;
}
void DataPool::construct_test_dataset() {
    fmt::print(fmt::fg(fmt::color::yellow), "[DataPool]: Constructing test dataset.\n");

    torch::Tensor factor_input_, price_nan_mask_, prices_, dates_, asset_feature_, market_feature_;

    factor_dim = train_recovered_hidden_factors.size(1);

    int end = test_prices.size(0);

    fmt::print("[DataPool]: Input dim: {}. Window size: {}.\n", factor_dim, window_size);
    // std::cout << "end: " << end << std::endl;
    if (use_forward_window) {
        factor_input_ =
            test_predicted_hidden_factors_pool.slice(0, 0, end, holding_period).clone().detach_();
    } else {
        // If use history hidden factors as factor_input, then the last day in the first sample is
        // today.
        torch::Tensor _test_recoverd_hidden_factors = test_recovered_hidden_factors;
        torch::Tensor _test_asset_features = test_asset_features;
        torch::Tensor _test_market_features = test_market_features;

        if (window_size > 1) {
            _test_recoverd_hidden_factors =
                torch::cat({train_recovered_hidden_factors.slice(0, -(window_size - 1)),
                            test_recovered_hidden_factors});
            _test_asset_features = torch::cat(
                {train_asset_features.slice(0, -(window_size - 1)), test_asset_features});
            _test_market_features = torch::cat(
                {train_market_features.slice(0, -(window_size - 1)), test_market_features});
        }

        factor_input_ = _test_recoverd_hidden_factors.unfold(0, window_size, holding_period)
                            .transpose(1, 2)
                            .clone()
                            .detach_();
        asset_feature_ = _test_asset_features.unfold(0, window_size, holding_period)
                             .transpose(1, 2)
                             .clone()
                             .detach_();
        market_feature_ =
            _test_market_features.unfold(0, window_size, holding_period).clone().detach_();
    }
    price_nan_mask_ = test_nan_mask.slice(0, 0, end, holding_period).clone().detach_();

    if (only_use_periodic_price) {
        prices_ = test_prices.slice(0, 0, end, holding_period).clone().detach_();
    }
    dates_ = test_dates_all.slice(0, 0, end, holding_period).clone().detach_();

    test_data = {factor_input_, asset_feature_, market_feature_, prices_, price_nan_mask_, dates_};

    test_data.validate();
    // test_data.normalize();
}

void DataPool::read_precomputed_hidden_factor(const std::string &path) {
    std::string train_file = SciLib::path_join(path, "Y-train.csv"),
                test_file = SciLib::path_join(path, "Y-test.csv");
    fmt::print(fmt::fg(fmt::color::yellow),
               "[DataPool]: Reading precomputed hidden factors: {} and {}.\n",
               SciLib::absolute_path(train_file), SciLib::absolute_path(test_file));

    torch::NoGradGuard nograd;
    train_recovered_hidden_factors =
        SciLib::mat_to_tensor(SciLib::load_csv_file<float>(train_file)).to(device).detach_();
    test_recovered_hidden_factors =
        SciLib::mat_to_tensor(SciLib::load_csv_file<float>(test_file)).to(device).detach_();

    construct_test_dataset();
}

void DecisionModelDataSet::to(const torch::Device &device) {
    for (auto &ele :
         {&factor_input, &price_nan_mask, &prices, &dates, &asset_feature, &market_feature}) {
        if (ele->numel() > 0)
            *ele = ele->to(device);
    }
}
void DecisionModelPretrainDataSet::normalize() { ::normalize(*this); }
void DecisionModelPretrainDataSet::to(const torch::Device &device) {
    for (auto &ele : {&factor_input, &asset_feature, &market_feature, &real_returns}) {
        if (ele->numel() > 0)
            *ele = ele->to(device);
    }
}

void DecisionModelDataSet::validate(bool info) {
    if (info) {
        fmt::print(fmt::fg(fmt::color::yellow), "[DecisionModelDataSet]: Validating dataset...\n");
        std::cout << "Factor input: " << factor_input.sizes() << std::endl;
        std::cout << "Asset features: " << asset_feature.sizes() << std::endl;
        std::cout << "Market features: " << market_feature.sizes() << std::endl;
    }
    if (info)
        fmt::print("[DecisionModelDataSet]: All fields non-empty: ...");
    std::vector numel{factor_input.numel(), price_nan_mask.numel(), prices.numel(),
                      dates.numel(),        asset_feature.numel(),  market_feature.numel()};
    if (std::all_of(numel.cbegin(), numel.cend(), [](const auto &e) { return e > 0; })) {
        if (info)
            std::cout << "pass." << std::endl;
    } else {
        if (info)
            fmt::print(fmt::fg(fmt::color::red), "failed: {}\n", numel);
        std::terminate();
    };

    if (info)
        fmt::print("[DecisionModelDataSet]: Batch size equal: ...");
    std::vector bsz{factor_input.size(0), price_nan_mask.size(0), prices.size(0),
                    dates.size(0),        asset_feature.size(0),  market_feature.size(0)};
    if (std::equal(bsz.begin() + 1, bsz.end(), bsz.begin())) {
        if (info)
            std::cout << "pass." << std::endl;
    } else {
        if (info)
            fmt::print(fmt::fg(fmt::color::red), "failed: {}\n", bsz);
    };

    if (info)
        fmt::print("[DecisionModelDataSet]: Mask size equal: ...");
    if (prices.sizes() == price_nan_mask.sizes()) {
        if (info)
            std::cout << "pass." << std::endl;
    } else {
        if (info)
            fmt::print(fmt::fg(fmt::color::red), "failed: {}\n", bsz);
    }
}

void DecisionModelDataSet::normalize() { ::normalize(*this); }

void DataPool::to(const torch::Device &device) {
    this->device = device;
    for (auto &ele : {&train_recovered_hidden_factors, &test_recovered_hidden_factors,
                      &train_prices, &test_prices, &train_nan_mask, &test_nan_mask,
                      &train_predicted_hidden_factors_pool, &test_predicted_hidden_factors_pool,
                      &train_dates_all, &test_dates_all, &test_dates, &train_market_features,
                      &test_market_features, &train_asset_features, &test_asset_features}) {
        if (ele->numel() > 0)
            *ele = ele->to(device);
    }

    test_data.to(device);
}

void DataPool::report_size() {
    fmt::print(fmt::fg(fmt::color::yellow), "[DataPool]: Reporting dataset size: \n");
    std::cout << "Test: " << std::endl;
    test_data.report_size();
    std::cout << "Train: " << std::endl;
    auto train_data = sample_train_dataset();
    train_data.report_size();
}

void DataPool::validate() const {
    fmt::print(fmt::fg(fmt::color::yellow), "[DataPool]: Validating data pool... \n");
    fmt::print("[DataPool]: Checking nan...\n");
    std::vector<std::string> names{"train_recovered_hidden_factors",
                                   "test_recovered_hidden_factors",
                                   "train_prices",
                                   "test_prices",
                                   "train_nan_mask",
                                   "test_nan_mask",
                                   "train_asset_features",
                                   "test_asset_features",
                                   "train_market_features",
                                   "test_market_features"};
    int i = 0;
    for (auto &ele :
         {&train_recovered_hidden_factors, &test_recovered_hidden_factors, &train_prices,
          &test_prices, &train_nan_mask, &test_nan_mask, &train_asset_features,
          &test_asset_features, &train_market_features, &test_market_features}) {
        if (ele->isnan().sum().item<int>() > 0) {
            fmt::print(fmt::fg(fmt::color::red), "[DataPool]: variable {} contains nan values.",
                       names[i]);
            std::abort();
        }
        if (ele->isinf().sum().item<int>() > 0) {
            fmt::print(fmt::fg(fmt::color::red), "[DataPool]: variable {} contains inf values.",
                       names[i]);
            std::abort();
        }
        ++i;
    }
    fmt::print("[DataPool]: Validating finished... \n");
}

void DecisionModelDataSet::report_size() {
    std::cout << "Factor input: " << factor_input.sizes() << std::endl;
    std::cout << "Market feature : " << market_feature.sizes() << std::endl;
    std::cout << "Asset feature: " << asset_feature.sizes() << std::endl;

    std::cout << "Prices: " << prices.sizes() << std::endl;
    std::cout << "Prices nan mask: " << price_nan_mask.sizes() << std::endl;
}