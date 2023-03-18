#include "stdafx.h"
#include "Metrics.hpp"

std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)>
create_metric_fn(const std::string &name) {
    if (name == "sharpe_ratio") {
        fmt::print("Using sharpe ratio metric.\n");
        return SciLib::sharpe_ratio;
    } else if (name == "simple_up_down_risk") {
        fmt::print("Using simple version up-down side risk metric: (M + up_risk) / down_risk.\n");
        return SciLib::simple_up_down_risk;
    } else if (name == "pure_return") {
        fmt::print("Using pure return metric: M.\n");
        return SciLib::pure_return;
    } else if (name == "return_to_mdd") {
        fmt::print("Using return to MDD metric: E/MDD.\n");
        return SciLib::return_to_mdd;
    } else if (name == "basic_gop") {
        fmt::print("Using basic GOP metric: E(log(R)).\n");
        return SciLib::basic_gop;
    } else {
        std::cout << "Invalid metric name: " + name + ".\n" << std::endl;
        std::terminate();
    }
};

void PortfolioPerformance::report() {
    fort::char_table table;
    table << "Wea"
          << "ARR"
          << "Vol"
          << "ASR"
          << "SoR"
          << "MDD"
          << "CR" << fort::endr;
    table << fort::separator;
    table << Wealth << ARR << AVol << ASR << SoR << MDD << CR << fort::endr;
    std::cout << table.to_string() << std::endl;
}

PortfolioPerformance calculate_performance(const torch::Tensor &wealth,
                                           const torch::Tensor &returns, double interval,
                                           double r_f, double MAR) {
    PortfolioPerformance res;
    res.Wealth = wealth[wealth.size(0) - 1].item<double>();
    res.ARR = SciLib::ARR(returns, interval).item<double>();
    res.AVol = SciLib::AVol(returns, interval).item<double>();
    res.ASR = SciLib::ASR(returns, interval, r_f).item<double>();
    res.SoR = SciLib::SoR(returns, interval, r_f, MAR).item<double>();
    res.MDD = SciLib::MDD(wealth).item<double>();
    res.CR = SciLib::CR(wealth, returns, interval).item<double>();
    return res;
};