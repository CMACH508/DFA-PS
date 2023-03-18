#pragma once
#include "StrategyBase.hpp"
#include "fmt/format.h"

void StrategyBase::save_config(const std::string &filename) {
    std::ofstream file(filename);
    file << config;
    fmt::print("Saving best config {} finished.\n", filename);
};

StrategyBase::StrategyBase(boost::json::object &config, int assets)
    : config(config), assets(assets), r_f(boost::json::value_to<double>(config["risk_free"])),
      r_c(boost::json::value_to<double>(config["cost"])),
      test_on_train_dataset(
          config["decision_model"].as_object()["test_on_train_dataset"].as_bool()){};
std::pair<Performance, TMatd> StrategyBase::test(const TMatd &test, const TMat<bool> &nan_mask,
                                                 bool write_output,
                                                 const std::string &portfolio_output) {
    auto N = test.rows(), d = test.cols();
    assert((N > 0) && (d > 0));

    backup();
    //(d, d, 1,1, 1): (risk asset returns, risk assets weights, risk free weight, portfolio
    // return,
    // wealth)
    // The wealth is the wealth before the portfolio return.
    // So after executing the portfolio, wealth will change into the value of next row.
    TMatd portfolios(N + 1, 2 * d + 3);
    portfolios.setZero();
    portfolios(Eigen::seqN(0, test.rows()), Eigen::seqN(0, test.cols())) = test;
    portfolios.coeffRef(0, 2 * d + 2) = 1;

    this->begin_test();
    TVecd old_portfolio(d);
    old_portfolio << 0.0, 0.0, 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        // cout << &(strategy->ssm->x_hat) << " " << strategy->ssm->x_hat.transpose() << endl;
        auto portfolio = this->step(test.row(i).transpose(), nan_mask.row(i).transpose());

        // 3.
        // write portfolio to table.
        portfolios(i, Eigen::seqN(d, d)) = portfolio.second;
        portfolios.coeffRef(i, 2 * d) = portfolio.first; // risk free weights.
                                                         /* if (i == 0)
                                                              cout << "in test alpha: " << portfolio.first << endl;
                                                         */ // portfolio return
        TVecd new_portfolio = portfolio.second;
        if (costQ) {
            portfolios.coeffRef(i, 2 * d + 1) =
                (1 - portfolio.first) * r_f +
                portfolio.first * (portfolio.second.dot(test.row(i)) -
                                   r_c * (new_portfolio - old_portfolio)
                                             .array()
                                             .abs()
                                             .matrix()
                                             .dot((1 + test.row(i).array()).matrix()));
        } else {
            // print("no cost\n");
            portfolios.coeffRef(i, 2 * d + 1) =
                (1 - portfolio.first) * r_f + portfolio.first * portfolio.second.dot(test.row(i));
        };
        old_portfolio = new_portfolio;
        // wealth
        portfolios.coeffRef(i + 1, 2 * d + 2) =
            portfolios.coeff(i, 2 * d + 2) * (1 + portfolios.coeff(i, 2 * d + 1));
        // info("Day {}/{}, Wealth {}", i + 1, N, portfolios.coeff(i + 1, 2 * d + 2));
        // cout << portfolio.first << ", " << portfolio.second.transpose() << endl;
    };

    if (write_output) {
        write_portfolios(portfolios, portfolio_output);
    };

    restore();
    return std::make_pair(cal_performance(portfolios.col(2 * d + 1).topRows(portfolios.rows() - 1)),
                          portfolios);
};