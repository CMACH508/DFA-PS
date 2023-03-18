#include "stdafx.h"

#include "Finance.hpp"

namespace SciLib {

torch::Tensor cal_ADR(const torch::Tensor &returns) {
    auto up = (returns > 0).sum(1);
    auto sz = (returns != 0).sum(1);
    // Add 1 to avoid zero denominator.
    return up / (sz - up + 1);
};
torch::Tensor cal_OBOS(const torch::Tensor &returns) {
    auto up = (returns > 0).sum(1);
    auto sz = (returns != 0).sum(1);
    return 2 * up - sz;
};
torch::Tensor cal_ADL(const torch::Tensor &returns) {
    auto obos = cal_OBOS(returns);
    return obos.cumsum(0);
};

// interval: count of days between each return. 1.0 for daily returns.
torch::Tensor ARR(const torch::Tensor &returns, double interval) {
    return torch::mean(returns) * (251 / interval);
};

torch::Tensor AVol(const torch::Tensor &returns, double interval) {
    return torch::std(returns) * sqrt(251 / interval);
};
torch::Tensor MDD(const torch::Tensor &wealth) {

    auto [cum_max_wealth, indices] = wealth.cummax(0);
    auto mdd = torch::max(1 - wealth / cum_max_wealth);

    // assert(var.numel() == 1);
    return mdd;
};
torch::Tensor ASR(const torch::Tensor &returns, double interval, double r_f) {
    return ARR(returns, interval) / AVol(returns, interval);
};
torch::Tensor SoR(const torch::Tensor &returns, double interval, double r_f, double MAR) {
    // auto lvol =；
    namespace F = torch::nn::functional;
    const auto negative_returns = F::relu(-returns, F::ReLUFuncOptions());
    return ARR(returns, interval) /
           (negative_returns.pow(2).sum() / torch::sum(negative_returns > 0)).sqrt();
};
torch::Tensor CR(const torch::Tensor &wealth, const torch::Tensor &returns, double interval) {
    // std::cout << ARR(returns, interval) << std::endl;
    // std::cout << MDD(wealth) << std::endl;
    // std::cout << "CR: " << ARR(returns) / MDD(wealth) << std::endl;
    return ARR(returns, interval) / MDD(wealth);
};

// ----------------------------Metrics---------------------------------------
torch::Tensor sharpe_ratio(const torch::Tensor &returns, const torch::Tensor &wealth) {
    auto [var, mean] = torch::var_mean(returns);

    // assert(var.numel() == 1);
    return mean / var.sqrt();
}

torch::Tensor pure_return(const torch::Tensor &returns, const torch::Tensor &wealth) {
    return torch::mean(returns);
}

torch::Tensor simple_up_down_risk(const torch::Tensor &returns, const torch::Tensor &wealth) {
    auto mean = torch::mean(returns);
    auto up_risk = torch::std(torch::nn::functional::relu(returns));
    // Note: negative returns is -relu(-x), but std(-relu(-x))=std(relu(-x))
    auto down_risk = torch::std(torch::nn::functional::relu(-returns));

    return (mean + up_risk) / (down_risk);
}
torch::Tensor return_to_mdd(const torch::Tensor &returns, const torch::Tensor &wealth) {
    return torch::mean(returns) / MDD(wealth);
}

torch::Tensor basic_gop(const torch::Tensor &returns, const torch::Tensor &wealth) {
    int N = wealth.size(0);
    // return (wealth.slice(0, 1, N).log() - wealth.slice(0, 0, N - 1).log()).mean();
    return (wealth[N - 1].log() - wealth[0].log()) / static_cast<double>(N - 1);
}

void FullPortfolioData::write(const std::string &filename) {
    if (filename != "") {
        std::ofstream file(filename);

        int assets = price.size(1);
        // Write header
        file << fmt::format("Date,Wealth,Return,");

        for (int i = 1; i <= assets; ++i) {
            file << fmt::format("P{},", i);
        }

        file << fmt::format("Cash,");
        for (int i = 1; i <= assets; ++i) {
            file << fmt::format("Volume{},", i);
        }

        file << fmt::format("InvestmentRatio, ShortRatio,");
        for (int i = 1; i <= assets; ++i) {
            if (i != assets)
                file << fmt::format("Beta{},", i);
            else
                file << fmt::format("Beta{}", i) << std::endl;
        }
        // Write data.
        // First row.
        file << fmt::format(",{},,", initial_cash);
        for (int i = 1; i <= assets; ++i) {
            file << ",";
        }

        file << fmt::format("{},", initial_cash);
        for (int i = 1; i <= assets; ++i) {
            file << ",";
        }
        for (int i = 1; i <= assets; ++i) {
            file << ",";
        }
        file << std::endl;

        int m = price.size(0);
        // auto _price = price.slice(0, 0, m - 1);
        const auto full_data = torch::cat(
            {wealth.narrow(0, 1, m).unsqueeze(1), returns.unsqueeze(1), price, cash.unsqueeze(1),
             volume, investment_ratio.unsqueeze(1), short_ratio.unsqueeze(1), beta},
            1);
        // std::cout << "write investment_ratio: " << investment_ratio.narrow(0, 0, 5) << std::endl;
        int n = full_data.size(1);
        for (int i = 0; i < m; ++i) {
            file << fmt::format("{}, ", dates[i].item<int>());
            for (int j = 0; j < n - 1; ++j) {
                file << fmt::format("{}, ", full_data[i][j].item<float>());
            }
            file << fmt::format("{}", full_data[i][n - 1].item<float>()) << std::endl;
        }
        //// write final row.
        //// wealth, return, cash.
        // file << fmt::format("{},{},{},", wealth[m - 1], returns[m - 2], cash[m - 1]);
        //// volume.
        // for (int j = 0; j < n - 1; ++j) {
        //     file << fmt::format("{},", volume[m - 2][j].item<float>());
        // }
        // file << fmt::format("{}", full_data[m - 2][n - 1].item<float>()) << std::endl;
    }
}

void PortfolioPerformance::report() {
    fort::char_table table;
    table << "ARR"
          << "AVol"
          << "ASR"
          << "SoR"
          << "MDD"
          << "CR" << fort::endr << ARR << AVol << ASR << SoR << MDD << CR << fort::endr;
    std::cout << table.to_string();
}
void PortfolioPerformance::write(const std::string &filename, bool reformat) {
    std::ofstream file(filename);
    file << "ARR, AVol, ASR, SoR, MDD, CR" << std::endl;
    if (reformat) {
        file << fmt::format("{:5.2f}, {:4.3f}, {:4.3f}, {:4.3f}, {:5.2f}, {:4.3f}", ARR * 100, AVol,
                            ASR, SoR, MDD * 100, CR)
             << std::endl;
    } else {
        file << fmt::format("{}, {}, {}, {}, {}, {}", ARR, AVol, ASR, SoR, MDD, CR) << std::endl;
    }
};

PortfolioPerformance calculate_portfolio_performance(const FullPortfolioData &portfolio,
                                                     int interval, double r_f, double MAR) {
    const auto &wealth = portfolio.wealth;
    const auto &returns = portfolio.returns;

    PortfolioPerformance res;
    res.Wealth = wealth[wealth.size(0) - 1].item<double>();
    res.ARR = SciLib::ARR(returns, interval).item<double>();
    res.AVol = SciLib::AVol(returns, interval).item<double>();
    res.ASR = SciLib::ASR(returns, interval, r_f).item<double>();
    res.SoR = SciLib::SoR(returns, interval, r_f, MAR).item<double>();
    res.MDD = SciLib::MDD(wealth).item<double>();
    res.CR = SciLib::CR(wealth, returns, interval).item<double>();
    return res;
}

}; // namespace SciLib