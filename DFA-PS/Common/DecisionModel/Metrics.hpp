#pragma once
#include <torch/torch.h>
#include <functional>
#include "../SciLib/Finance.hpp"

std::function<torch::Tensor(const torch::Tensor &, const torch::Tensor &)>
create_metric_fn(const std::string &name);

struct PortfolioPerformance {
    double Wealth;
    double ARR, AVol, ASR, SoR, MDD, CR;

    void report();
};
PortfolioPerformance calculate_performance(const torch::Tensor &wealth,
                                           const torch::Tensor &returns, double interval = 1.0,
                                           double r_f = 0.0, double MAR = 0.0);