#include "stdafx.h"
#include "TFABase.hpp"
#include "../Net/NetFactory.hpp"
#include "Net/ModelSelectionHelper.hpp"
#include "SciLib/EigenTorchHelper.hpp"
#include "SciLib/MatrixManifolds.hpp"
#include "SciLib/EigenStat.hpp"
#include "SciLib/EigenML.hpp"
#include "SciLib/MatrixManifolds.hpp"
#include "SciLib/MISC.hpp"

TFAParms::TFAParms(int dim, int hidden_dim)
    : A(SciLib::sample_in_stiefel_manifold<TMatd>(dim, hidden_dim)), b(TVecd::Zero(dim)),
      Sx(TVecd::Constant(dim, 1.0)) {
    // Sy = (1.0 / ((A.transpose() * A).diagonal()).array()).matrix();
    Sy = TVecd::Constant(hidden_dim, 1.0);
};

void TFAParms::write(const std::string &name) const {
    Eigen::IOFormat mat_format(6, Eigen::DontAlignCols, ", ");
    std::ofstream(name + "_A.csv") << A.format(mat_format);
    if (B.rows() > 0) {
        std::ofstream(name + "_state_B.csv") << B.format(mat_format);
    }
    std::ofstream(name + "_b.csv") << b.format(mat_format);
    std::ofstream(name + "_Sx.csv") << Sx.format(mat_format);
    std::ofstream(name + "_Sy.csv") << Sy.format(mat_format);
};
void TFAParms::read(const std::string &name) {
    A = SciLib::readmatrix(name + "_A.csv");
    B = SciLib::readmatrix(name + "_B.csv");
    Sx = SciLib::readmatrix(name + "_Sx.csv");
    Sy = SciLib::readmatrix(name + "_Sy.csv");
}

CBT zero_CBT(int size, std::initializer_list<int64_t> dims, torch::Device device) {
    CBT y0(size);
    for (int i = 0; i < size; ++i) {
        y0.push_back(torch::zeros(dims, device));
    }
    return y0;
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// 1. Initialization of A and Sx is very crucial to convergence.
TFABase::TFABase(boost::json::object &config, int dim)
    : config(config), dim(dim), fix_Sx(config["fix_Sx"].as_bool()),
      hidden_dim(static_cast<int>(config["hidden_dim_ratio"].as_double() * dim)),
      lr(config["lr"].as_double()),
      window(static_cast<int>(config["neural_net_TFA"].as_object()["window"].as_int64())),
      seed(static_cast<int>(config["seed"].as_int64())),
      max_batch_size(static_cast<int>(config["max_batch_size"].as_int64())),
      epochs(static_cast<int>(config["neural_net_TFA"].as_object()["epochs"].as_int64())),
      test_interval(static_cast<int>(config["test_interval"].as_int64())),
      re_orthogonalize(static_cast<int>(config["re_orthogonalize"].as_int64())),
      tb("Epoch", "S MSE", "S LL", "S d", "O MSE", "O LL", "LL", "T.S MSE", "T.S LL", "T.O MSE",
         "T.O LL", "T.LL", "T.FMSE", "Time"),
      ori_hidden_dim(hidden_dim), model_selection(config.at("model_selection").as_bool()),
      model_select_threshold(config.at("threshold").as_double()),
      min_hidden_dim(config.at("min_hidden_dim").as_double()) {
    srand(seed);
    // torch::cuda::manual_seed(seed + 1);
    torch::manual_seed(seed + 1);
    at::globalContext().setDeterministicCuDNN(true);

    fmt::print(fmt::fg(fmt::color::yellow), "[TFABase]: Dim: {}, Hidden_dim: {}\n", dim,
               hidden_dim);
    fmt::print("[TFABase]: Base LR: {}\n", lr);

    tb.set_file("ssm_history.csv");
    tb.set_col_type(1, "Epoch", "S Dim");
    tb.set_col_formatter("{: >4}", "{: ^4}", "Epoch", "S d");
    tb.set_col_formatter("{: >5.2f}", "{: ^5}", "Time");
    // tb.add_sep_after("Epoch", "LL", "T.LL");

    init_model_selection();
}

void TFABase::init(const TMatd &data) {
    parms = TFAParms(dim, hidden_dim);
    // parms.read("I:\\Research\\NewLab\\TFA\\Reference\\TFA_Shi20100623\\Init");
    TMatd S = SciLib::cov(data);
    // std::cout << S.topRows(5).leftCols(5) << std::endl;

    Eigen::SelfAdjointEigenSolver<TMatd> solver(S);
    TVecd V = solver.eigenvalues();
    TMatd D = solver.eigenvectors();

    parms.A = D.rightCols(hidden_dim);
    // std::cout << parms.A.topLeftCorner(5, 5) << std::endl;
    parms.Sx.setConstant(
        std::abs((S - parms.A * V.bottomRows(hidden_dim).asDiagonal() * parms.A.transpose())
                     .diagonal()
                     .mean()));
    parms.Sy = V.bottomRows(hidden_dim).array();
    // parms.Sy.setConstant(1);

    fmt::print("Minimum MSE by LRA: {}.\n", SciLib::LRA_min_mse(data, hidden_dim));
}

void TFABase::cal_residuals(const TVecd &x) {
    varepsilon = y_hat - y_til;
    e = x - x_hat;
}

void TFABase::before_train(const DataSet &data) {
    fmt::print("[TFA]: Total epochs: {}\n", epochs);
    fmt::print("[TFA]: Sample size on train: ({}, {})\n", data.train_returns.rows(),
               data.train_returns.cols());
    fmt::print("[TFA]: Sample size on test: ({}, {}\n", data.test_returns.rows(),
               data.test_returns.cols());

    parms.write("tmp/Parms_Init");

    tb.print_header();
    tb.write_header();

    ori_Sy_file = std::ofstream("ori_Sy.csv");
}

int TFABase::train(const DataSet &data, bool save_ssm_train_history, const std::string &filename) {
    if (epoch > epochs)
        return 0;

    auto &X = data.train_returns;

    // fmt::print("LR: {}\n", lr);

    before_train(data);

    int best_epoch = 0;
    double best_test_ev = 1e99;

    std::ofstream Sx_file("S_x.csv"), Sy_file("S_y.csv");

    for (; epoch <= epochs; ++epoch) {

        before_epoch();

        double Ev = 0, Varepsilonv = 0, OLL = 0, SLL = 0;
        const int N = X.rows();
        // fmt::print("epoch {}: {}\n", epoch, N);
        bool finished = false;
        int j = 0, batch_size = max_batch_size;

        // Initial states.
        CBT y0 = zero_CBT(window, {1, 1, hidden_dim}, device);
        if (Y_hat_1.rows() > 0) {
            TVecf _y0 = Y_hat_1.row(0).cast<float>();
            for (int i = 0; i < window; ++i) {
                y0.push_back(torch::from_blob(static_cast<float *>(_y0.data()), {1, 1, hidden_dim})
                                 .to(device)
                                 .clone());
            }
        }
        // std::cout << y0.back() << std::endl;
        while (!finished) {
            // fmt::print("New batch.\n");
            if (j + max_batch_size >= N) {
                batch_size = N - j;
                // fmt::print("batch size: {}\n", batch_size);
                finished = true;
            }

            batch = X.middleRows(j, batch_size);
            // fmt::print("{}x{}\n", batch.rows(), batch.cols());
            before_train_batch(y0, batch);
            train_batch(y0, batch);
            after_train_batch(y0, batch);

            Varepsilonv += Varepsilon.squaredNorm();
            Ev += E.squaredNorm();
            SLL -= SciLib::NLL_gaussian_diag(Varepsilon, parms.Sy);
            OLL -= SciLib::NLL_gaussian_diag(E, parms.Sx);

            j += batch_size;
        }

        Ev /= (N * dim);
        Varepsilonv /= (N * hidden_dim);
        SLL /= N;
        OLL /= N;

        tb["Epoch"] = epoch;
        tb["S MSE"] = Varepsilonv;
        tb["S LL"] = SLL;
        tb["S d"] = hidden_dim;
        tb["O MSE"] = Ev;
        tb["O LL"] = OLL;
        tb["LL"] = SLL + OLL;

        if (epoch % test_interval == 1) {
            auto y0 = torch::zeros({1, window, hidden_dim}, device);
            auto [VarepsilonV, SLL, Ev, OLL, FMSE] = test(y0, data.test_returns);

            if (FMSE < best_test_ev) {
                best_epoch = epoch;
                best_test_ev = FMSE;
            }
            tb["T.S MSE"] = Varepsilonv;
            tb["T.S LL"] = SLL;
            tb["T.O MSE"] = Ev;
            tb["T.O LL"] = OLL;
            tb["T.LL"] = SLL + OLL;
            tb["T.FMSE"] = FMSE;
        }

        Eigen::IOFormat mat_format(6, Eigen::DontAlignCols, ", ");
        Sx_file << parms.Sx.transpose().format(mat_format) << std::endl;
        Sy_file << parms.Sy.transpose().format(mat_format) << std::endl;

        after_epoch();
    }

    fmt::print("[TFA]: Best epoch: {}. Restruction MSE: {}.\n", best_epoch, best_test_ev);
    after_train();
    // cout << A << endl << B << endl << Sy << endl << Sx << endl;
    // cout << (Sx + A.transpose() * Sy.inverse() * A).inverse() << endl;
    return best_epoch;
};
void TFABase::before_train_batch(CBT &y0, const TMatd &X) {
    this->n = X.rows();
    Varepsilon.resize(n, hidden_dim);
    E.resize(n, dim);
    Y_hat.resize(n, hidden_dim);
    Y_hat_1.resize(n, hidden_dim);

    // Here y_hat is read from y0 tensor.
    y_hat = TMapf(static_cast<float *>(y0.back().to(torch::kCPU).data_ptr()), hidden_dim, 1)
                .cast<double>();
}
void TFABase::train_batch(CBT &y0, const TMatd &X) {
    for (int i = 0; i < n; ++i) {
        TVecd x = X.row(i).transpose();
        before_train_batch_single_loop(y0, x);
        Y_hat_1.row(i) = y_hat.transpose();

        cal_y_til(y0);
        cal_x_til();
        cal_y_hat(y0, x);
        cal_x_hat();
        cal_residuals(x);

        update_y0(y0);
        Y_hat.row(i) = y_hat.transpose();
        Varepsilon.row(i) = varepsilon.transpose();
        E.row(i) = e.transpose();

        after_train_batch_single_loop(y0, x);
    }
    before_update_parms(X);
    update_parms();
}

void TFABase::after_train_batch(CBT &y0, const TMatd &X) {
    // Model selection.
    if (model_selection) {
        model_select(y0);
    }
}
double TFABase::cal_nll() {
    return SciLib::NLL_gaussian_diag(Varepsilon, parms.Sy) + SciLib::NLL_gaussian_diag(E, parms.Sx);
}
void TFABase::update_parms() {

    // TMatd Xmu = batch.rowwise() - parms.b.transpose();
    //
    /*if (SciLib::contain_nan(Y_hat)) {
        std::cout << "There are nans in Y_hat." << std::endl;
        SciLib::write_mat(Y_hat, "Y_hat_nan.csv");
    }*/
    TMatd grad_A = lr * E.transpose() * Y_hat / E.rows();
    // TMatd grad_A =
    //     ((Y_hat.transpose() * Y_hat).llt().solve(Y_hat.transpose() * Xmu)).transpose() - parms.A;
    // double loss1 = (E.array().pow(2)).mean();
    // TMatd parms_A_old = parms.A;
    // std::cout << "yes" << std::endl;
    // std::cout << (parms.A.transpose() * parms.A).diagonal().transpose() << std::endl;
    parms.A = SciLib::move_in_stiefel_manifold<SciLib::MoveInStiefelManifoldPolicy::basic_geod>(
        parms.A, grad_A, 0.2);
    if ((re_orthogonalize > 0) && (epoch % re_orthogonalize == 0)) {
        parms.A = SciLib::orthogonalize(parms.A);
    }
    parms.b.noalias() += lr * E.colwise().mean().transpose();

    // std::cout << "Delta Sx: " << E.array().pow(2).colwise().mean() << std::endl;

    if (!fix_Sx) {
        parms.Sx =
            (1 - lr) * parms.Sx + lr * (E.array().pow(2)).matrix().colwise().mean().transpose();
        // parms.Sx = (E.array().pow(2)).matrix().colwise().mean().transpose();
    }
    parms.Sy = (1 - lr) * parms.Sy +
               lr * (Varepsilon.array().pow(2)).matrix().colwise().mean().transpose();
    // parms.Sy = (Varepsilon.array().pow(2)).matrix().colwise().mean().transpose();
    update_base_parms_buffer();
}

void TFABase::after_train() {
    parms.write("tmp/Parms_Final");

    ori_Sy_file.close();
    SciLib::writematrix(Varepsilon, "tmp/Varepsilon_final.csv");
    SciLib::writematrix(E, "tmp/E_final.csv");
    SciLib::writematrix(Y_hat, "tmp/Y_hat_final.csv");
    SciLib::writematrix(Y_hat_1, "tmp/Y_hat_1_final.csv");
}

void TFABase::before_epoch() {
    sw.reset();
    tb.new_row();
}
void TFABase::after_epoch() {
    tb["Time"] = SciLib::stopwatch_elapsed_seconds(sw);
    tb.print_row();
    tb.write_row();

    // std::cout << "Sx: " << parms.Sx.transpose() << std::endl;
    if (SciLib::contain_nan(parms.Sx)) {
        fmt::print(fmt::fg(fmt::color::red), "nan in Sx, terminating...\n");
        std::terminate();
    }
    if (SciLib::contain_nan(parms.Sy)) {
        fmt::print(fmt::fg(fmt::color::red), "nan in Sy, terminating...\n");
        std::terminate();
    }
}

// S MSE, SLL, O MSE, OLL, F MSE
std::tuple<double, double, double, double, double> TFABase::test(torch::Tensor &y0,
                                                                 const TMatd &X) {
    backup();

    // Note that x fitting error is the same as x forecast error. But here forecast_error_x is
    // the error AFTER parameter updating. So the value of fitting_error_x not equal to
    // forecast_error_x.
    double fitting_error_x = 0, forecast_error_x = 0, error_y = 0;
    CBT _y0(window);
    for (int i = 0; i < window; ++i) {
        _y0.push_back(y0[i].unsqueeze(0));
    }
    const int N = static_cast<int>(X.rows());
    TMatd V(N, hidden_dim), E(N, dim), FE(N, dim);
    for (int i = 0; i < N; i++) {
        TVecd x = X.row(i).transpose();

        cal_y_til(_y0);
        cal_x_til();
        cal_y_hat(_y0, x);
        cal_x_hat();
        cal_residuals(x);
        update_y0(_y0);

        V.row(i) = varepsilon.transpose();
        E.row(i) = e.transpose();
        FE.row(i) = (x - x_til).transpose();
        // forecast_error_x += (x - x_til).squaredNorm();
    }

    restore();

    error_y = V.squaredNorm() / (N * hidden_dim);
    fitting_error_x = E.squaredNorm() / (N * dim);
    forecast_error_x = FE.squaredNorm() / (N * dim);

    double SLL = -SciLib::NLL_gaussian_diag(V, parms.Sy) / N,
           OLL = -SciLib::NLL_gaussian_diag(E, parms.Sx) / N;

    return std::make_tuple(error_y, SLL, fitting_error_x, OLL, forecast_error_x);
}

DataPool TFABase::generate_data_pool(const DataSet &data, int window_size, int holding_period,
                                     bool only_use_periodic_price, const std::string &window_type) {
    // y0
    fmt::print("[TFA]: Generating data pool...\n");
    auto y0 = zero_CBT(window, {1, 1, hidden_dim}, device);

    DataPool data_pool(data, device, holding_period, only_use_periodic_price, window_type,
                       window_size);

    spdlog::stopwatch sw;
    fmt::print("[TFA]: Generating data for train...");
    std::tie(data_pool.train_recovered_hidden_factors,
             data_pool.train_predicted_hidden_factors_pool) =
        generate_factors(y0, data.train_returns, window_size);
    fmt::print("Using {} seconds.\n", sw);

    sw.reset();
    fmt::print("[TFA]: Generating data for testing...");
    std::tie(data_pool.test_recovered_hidden_factors,
             data_pool.test_predicted_hidden_factors_pool) =
        generate_factors(y0, data.test_returns, window_size);
    fmt::print("Using {} seconds.\n", sw);

    SciLib::write_1d2d_tensor(data_pool.train_recovered_hidden_factors, "Y-train.csv");
    SciLib::write_1d2d_tensor(data_pool.test_recovered_hidden_factors, "Y-test.csv");

    data_pool.construct_test_dataset();
    data_pool.report_size();

    return data_pool;
};

torch::Tensor TFABase::predict_hidden_factors(CBT &y0, int n) {
    torch::Tensor res = torch::zeros({n, hidden_dim});
    for (int i = 0; i < n; ++i) {
        cal_y_til(y0);
        torch::Tensor factor = SciLib::coldvec_to_3d_ftensor(y_hat, device);
        res[i] = factor.flatten(0);
        y0.push_back(factor);
    }

    return res;
}

std::pair<TMatd, TMatd> TFABase::predict(CBT &y0, int n) {
    auto hidden_factors = predict_hidden_factors(y0, n);
    TMatd Y = TMapf(static_cast<float *>(hidden_factors.to(torch::kCPU).data_ptr()), n, hidden_dim)
                  .cast<double>();
    TMatd X = (Y * parms.A.transpose()).rowwise() + parms.b.transpose();
    return {Y, X};
};

TMatd TFABase::recover_hidden_factor(CBT &y0, const TMatd &X) {
    const int N = X.rows();
    torch::Tensor recovered_hidden_factors = torch::zeros({N, hidden_dim}, device);

    for (int i = 0; i < N; ++i) {
        const TVecd x = X.row(i).transpose();
        cal_y_til(y0);
        cal_x_til();
        cal_y_hat(y0, x);
        torch::Tensor factor = SciLib::coldvec_to_3d_ftensor(y_hat, device);
        recovered_hidden_factors[i] = factor.flatten(0);
        y0.push_back(factor);
    }
    TMatd res = TMapf(static_cast<float *>(recovered_hidden_factors.to(torch::kCPU).data_ptr()), N,
                      hidden_dim)
                    .cast<double>();
    return res;
};

std::pair<TMatd, TMatd> TFABase::reconstruct(CBT &y0, const TMatd &X) {
    TMatd Y = recover_hidden_factor(y0, X);
    TMatd _X = (Y * parms.A.transpose()).rowwise() + parms.b.transpose();
    return std::make_pair(Y, _X);
};

std::pair<torch::Tensor, torch::Tensor> TFABase::generate_factors(CBT &y0, const TMatd &X,
                                                                  int predict_window) {
    torch::NoGradGuard nograd;

    const int N = X.rows();
    torch::Tensor recovered_hidden_factors = torch::zeros({N, hidden_dim}, device),
                  predicted_hidden_factors = torch::zeros({N, predict_window, hidden_dim}, device);
    // y_hat_t = y0.back();
    y_hat = TMapf(static_cast<float *>(y0.back().data_ptr()), hidden_dim, 1).cast<double>();
    for (int i = 0; i < N; ++i) {
        cal_y_til(y0);
        cal_x_til();
        cal_y_hat(y0, X.row(i));
        torch::Tensor factor = SciLib::coldvec_to_3d_ftensor(y_hat, device);
        recovered_hidden_factors[i] = factor.flatten(0);

        y0.push_back(factor);
        CBT _y0(window);
        for (int i = 0; i < window; ++i) {
            _y0.push_back(y0[i].clone());
        }
        predicted_hidden_factors[i] = predict_hidden_factors(_y0, predict_window);
    }
    return {recovered_hidden_factors, predicted_hidden_factors};
};

void TFABase::print_parms() {
    // TFAModelBase::print_parms();
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "---A---" << std::endl
              << parms.A.row(0).rightCols(5) << std::endl
              << "---Sx---" << std::endl
              << parms.Sx.transpose().rightCols(5) << std::endl
              << "---Sy---" << std::endl
              << parms.Sy.transpose().rightCols(5) << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
};
void TFABase::write_parms(const std::string &tag) {
    parms.write(tag);
    SciLib::writematrix(batch, fmt::format("tmp/X_{}.csv", tag));
    if (n > 0) {
        SciLib::writematrix(Varepsilon, fmt::format("tmp/Varepsilon_{}.csv", tag));
        SciLib::writematrix(E, fmt::format("tmp/E_{}.csv", tag));
    }
};

void TFABase::model_select(CBT &y0) {
    auto q = model_select_components();
    if (q) {
        adjust_base_parms();
        adjust_y0(y0);

        // print_table_header();
        tb.print_header();
    }
};

bool TFABase::model_select_components() {
    auto [it1, it2] = std::minmax_element(parms.Sy.begin(), parms.Sy.end(), abs_compare<double>);
    // const double threshold = (*it2) * model_select_threshold;
    const double threshold = model_select_threshold * (*it2);
    if ((*it1) < threshold) {
        // if ((*it1) < model_select_threshold) {
        //  Execute model selection.

        fmt::print(fmt::fg(fmt::color::yellow),
                   "[LinearTFA]: Model selection start. Max variance: {: 7.4f}. Min variance: "
                   "{: 7.4f}.\n",
                   *it2, *it1);

        std::iota(selected_index.begin(), selected_index.end(), 0);

        int i1 = 0;
        std::vector<int> removed_index;
        for (int i2 = 0; i2 < hidden_dim; ++i2) {
            if ((parms.Sy.coeffRef(i2) > threshold)) {
                selected_index[i1] = i2;

                ++i1;
            } else {
                /* fmt::print(fmt::fg(fmt::color::yellow),
                            "[LinearTFA]: Variance of component {} is {: 7.4f}. It's less then "
                            "{} of max "
                            "variance {: 7.4f}. Try remove it.\n",
                            i2, parms.Sy.coeffRef(i2), model_select_threshold, *it2);*/
                fmt::print(fmt::fg(fmt::color::yellow),
                           "[LinearTFA]: Variance of component {} is {: 7.4f}. It's less then "
                           "threshold {: 7.4f}. Try remove it.\n",
                           i2, parms.Sy.coeffRef(i2), model_select_threshold, *it2);
                removed_index.emplace_back(i2);
            }
        }
        // If too much components are removed, than don't do it.
        if (i1 < min_hidden_dim * dim) {
            fmt::print(fmt::fg(fmt::color::red),
                       "After model selection, only {} will be kept which is too small. Model "
                       "selection is cancelled and turned off.\n",
                       i1);
            model_selection = false;
            return false;
        }
        selected_index.resize(i1);
        selected_index_map = TVec<int>(selected_index_map(selected_index));

        hidden_dim = selected_index.size();
        fmt::print(fmt::fg(fmt::color::yellow), "[LinearTFA]: New hidden dimension {}.\n",
                   hidden_dim);
        return true;
    }

    return false;
}

void TFABase::adjust_base_parms() {
    fmt::print(fmt::fg(fmt::color::yellow), "[LinearTFA]: Now adjusting base parameters.\n");
    parms.Sy = decltype(parms.Sy)(parms.Sy(selected_index));
    parms.A = decltype(parms.A)(parms.A(Eigen::indexing::all, selected_index));
    update_base_parms_buffer();
};

void TFABase::adjust_y0(CBT &y0) {
    torch::NoGradGuard nograd;
    auto selected_index_t = torch::tensor(selected_index).to(device);
    for (auto &ele : y0) {
        ele = ele.index_select(2, selected_index_t);
    }
};

void TFABase::init_model_selection() {
    if (model_selection) {
        ori_hidden_dim = hidden_dim;
        fmt::print("[LinearTFA]: Using model selection. Start from hidden_dim: {}.\n", hidden_dim);

        selected_index.resize(hidden_dim);
        selected_index_map.resize(hidden_dim);
        std::iota(selected_index.begin(), selected_index.end(), 0);
        std::iota(selected_index_map.begin(), selected_index_map.end(), 0);
    }
}