#pragma once
#include <torch/torch.h>

namespace SciLib {
// Data Validation
//   Check if a vec contains value larger than max_v, nan, inf
// max_v:previous 600
template <typename T>
bool vec_valid(const Eigen::EigenBase<T> &_x, int n_unique = 100, double max_v = 10.0,
               double min_v = -0.9, const std::string &id = "") {
    const T &x = static_cast<T>(_x);

    std::set<double> s(x.begin(), x.end());
    if (s.size() < n_unique) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Vector {} has distinct values less than {}. "
                   "Remove it.\n",
                   id, n_unique);
        return false;
    }
    if (*s.crbegin() > max_v) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Vector {}: max value {:10.4f}, larger than {}. Remove it.\n", id, *s.crbegin(),
                   max_v);
        return false;
    }
    if (*s.cbegin() < min_v) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Vector {}: min value {:10.4f}, smaller than {}. Remove it.\n", id, *s.cbegin(),
                   max_v);
        return false;
    }
    return true;
};
// Indicators -------------------------------------------------------------------
//  returns is a 2d matrix with rows for periods and columns for assets.
torch::Tensor cal_ADR(const torch::Tensor &returns);
torch::Tensor cal_OBOS(const torch::Tensor &returns);
torch::Tensor cal_ADL(const torch::Tensor &returns);

// interval: count of days between each return. 1.0 for daily returns.
torch::Tensor ARR(const torch::Tensor &returns, double interval = 1.0);
torch::Tensor AVol(const torch::Tensor &returns, double interval = 1.0);
torch::Tensor MDD(const torch::Tensor &wealth);
torch::Tensor ASR(const torch::Tensor &returns, double interval = 1.0, double r_f = 0.0);
torch::Tensor SoR(const torch::Tensor &returns, double interval = 1.0, double r_f = 0.0,
                  double MAR = 0.0);
// ARR/MDD
torch::Tensor CR(const torch::Tensor &wealth, const torch::Tensor &returns, double interval = 1.0);

// Metrics -------------------------------------------------------------------
torch::Tensor sharpe_ratio(const torch::Tensor &returns,
                           const torch::Tensor &wealth = torch::Tensor());
torch::Tensor pure_return(const torch::Tensor &returns,
                          const torch::Tensor &wealth = torch::Tensor());
// (E + std(R+))/std(R-). R+:positive returns.
torch::Tensor simple_up_down_risk(const torch::Tensor &returns,
                                  const torch::Tensor &wealth = torch::Tensor());
// E/MDD
torch::Tensor return_to_mdd(const torch::Tensor &returns,
                            const torch::Tensor &wealth = torch::Tensor());

// mean(ln(R))
torch::Tensor basic_gop(const torch::Tensor &returns,
                        const torch::Tensor &wealth = torch::Tensor());

// Backtest -------------------------------------------------------------------
template <typename VecType> struct PortfolioWeight {
    VecType w;  // Normalized.
    double rho; // Ratio.
};
struct FullBetaPortfolioWeight {
    torch::Tensor long_weight, short_weight; // Full length normalized vector. Both are positive.
    torch::Tensor long_sel_weight,
        short_sel_weight; // Full length only selected weight (not normalized).
};

template <typename T>
std::pair<torch::Tensor, FullBetaPortfolioWeight>
convert_portfolio_weight(const std::vector<PortfolioWeight<T>> &vec) {
    int N = vec.size();
    int A = vec[0].w.size();
    // std::cout << A << std::endl;
    torch::Tensor rho = torch::zeros({N, 1}, torch::kDouble),
                  w = torch::zeros({N, A}, torch::kDouble);
    for (int i = 0; i < N; ++i) {
        rho[i][0] = vec[i].rho;
        for (int j = 0; j < A; ++j) {
            w[i][j] = vec[i].w.coeff(j);
        }
    }

    auto long_weight = torch::nn::functional::relu(w);
    auto short_weight = torch::nn::functional::relu(-w);

    for (int i = 0; i < N; ++i) {
        auto ls = long_weight[i].sum();
        auto ss = short_weight[i].sum();
        if ((ls != torch::zeros({1})).item<bool>()) {
            long_weight[i] = long_weight[i] / ls;
        }
        if ((ss != torch::zeros({1})).item<bool>()) {
            // Remember short_weight is also positive.
            short_weight[i] = short_weight[i] / ss;
        }
    }
    return {rho, {long_weight, short_weight}};
};

struct FullPortfolioData {
    torch::Tensor wealth, returns, price, investment_ratio, short_ratio, beta, cash, volume;
    double initial_cash;
    torch::Tensor dates;

    void write(const std::string &filename);
};

struct PortfolioPerformance {
    double Wealth;
    double ARR, AVol, ASR, SoR, MDD, CR;

    void report();
    void write(const std::string &filename, bool reformat = false);
};

PortfolioPerformance calculate_portfolio_performance(const FullPortfolioData &portfolio,
                                                     int interval = 1, double r_f = 0.0,
                                                     double MAR = 0.0);

struct ReturnCalculator {
    // ReturnCalculator(){};

    bool allow_short, only_use_periodic_price;
    double initial_cash;
    int data_period;
    double r_c;

    torch::Device device = torch::kCPU;

    // prices and investment_ratio, beta have the same width. But we don't need to adjust volume at
    // the
    // last period. Then length of final alphas, betas will be size of W minus 1.
    // Best for batch training.
    template <bool train_mode = false>
    FullPortfolioData calculate_returns(const torch::Tensor &_alpha,
                                        const FullBetaPortfolioWeight &weight,
                                        const torch::Tensor &prices, const torch::Tensor &dates) {
        auto dtype = _alpha.dtype();
        int N = prices.size(0), A = prices.size(1);

        torch::Tensor alpha, beta;
        if (allow_short) {
            alpha = torch::tensor({0.99}, dtype)
                        .to(device)
                        .expand({N, 1}); // As all money will be invested, leave a little fraction
                                         // to pay fee. Note that total weight is 1.
            // beta = -_alpha * weight.long_weight + (1 + _alpha) * weight.short_weight;
            beta = (1 + _alpha) * weight.long_weight - _alpha * weight.short_weight;
        } else {
            alpha = _alpha;
            beta = weight.long_weight;
        }

        FullPortfolioData res;

        auto volume = torch::zeros({A}, dtype).to(device);
        auto cash = torch::tensor({initial_cash}, dtype).to(device);

        std::vector<torch::Tensor> Ws(N + 1);
        Ws[0] = torch::tensor({initial_cash}, dtype).to(device);
        int period = only_use_periodic_price ? 1 : data_period;

        std::vector<torch::Tensor> _alphas, _alphas_short, _betas, _cashs, _volumes;
        if constexpr (!train_mode) {
            _alphas.resize(N), _alphas_short.resize(N), _betas.resize(N), _cashs.resize(N),
                _volumes.resize(N);
        }

        for (int i = 0; i < N; ++i) {
            int j = i / period;

            if (i % period == 0) {
                auto pre_wealth = cash + torch::dot(prices[i], volume);
                auto new_volume = pre_wealth * alpha[j] * beta[j] / prices[i];
                auto diff_volume = new_volume - volume;
                cash = cash - torch::dot(diff_volume, prices[i]) -
                       torch::dot(diff_volume.abs(), prices[i]) * r_c;

                volume = new_volume;

                Ws[i + 1] = cash + alpha[j] * pre_wealth;
            } else {
                Ws[i + 1] = cash + torch::dot(volume, prices[i]);
            }

            if constexpr (!train_mode) {
                _cashs[i] = cash;
                _volumes[i] = volume.unsqueeze(0);
                _alphas[i] = alpha[j];
                if (allow_short) {
                    _alphas_short[i] = _alpha[j];
                } else {
                    _alphas_short[i] = torch::tensor({0}, device);
                }
                _betas[i] = beta[j].unsqueeze(0);
            };
        }
        // Ws[N - 1] = cash + torch::dot(volume, prices[N - 1]);

        torch::Tensor W = torch::cat(Ws);

        if constexpr (train_mode) {
            return {W, W.narrow(0, 1, N) / W.narrow(0, 0, N) - 1};
        } else {
            torch::Tensor cashs = torch::cat(_cashs);
            torch::Tensor volumes = torch::cat(_volumes);
            torch::Tensor alphas = torch::cat(_alphas);
            torch::Tensor alphas_short = torch::cat(_alphas_short);
            // std::cout << "investment_ratio in fun: " << alphas.narrow(0, 0, 5) << std::endl;

            torch::Tensor betas = torch::cat(_betas);

            return {W,
                    W.narrow(0, 1, N) / W.narrow(0, 0, N) - 1,
                    prices,
                    alphas,
                    alphas_short,
                    betas,
                    cashs,
                    volumes,
                    initial_cash,
                    dates};
        }
    }
};

// Interactive account.
class Account {};
}; // namespace SciLib