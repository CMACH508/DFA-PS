#pragma once
#include <boost/json.hpp>
#include <fmt/format.h>
#include <torch/torch.h>

// (N, L, I): (batch, window, in_dim)
// (N, k, I): (batch, k, out_dim
struct BasisFunctionModuleBase : public torch::nn::Module {
    /*  BasisFunctionModuleBase(boost::json::object &config, int dim, int window, int k)
          : config(config), C(C), window(window),
            output(static_cast<int>(config["output"].as_int64())), k(k){};*/
    int window, in_dim, out_dim, k;
    BasisFunctionModuleBase(int window, int in_dim, int out_dim, int k);

    virtual torch::Tensor forward(const torch::Tensor &x) = 0;
    virtual void copy_parms(std::shared_ptr<struct BasisFunctionModuleBase> &new_net,
                            const std::vector<int> &selected_index){};

    void report();

    // boost::json::object &config;
    // int C, window, output, k;
};

//(N, L, I) -> (N, k, I)
// sum(k_i x_i), k_i: scalar.
struct SimpleLinearBasisFunctionModule : BasisFunctionModuleBase {
    SimpleLinearBasisFunctionModule(const boost::json::object &config, int window, int in_dim,
                                    int k);

    torch::Tensor forward(const torch::Tensor &x) override;

    torch::Tensor A, b;

    void copy_parms(std::shared_ptr<struct BasisFunctionModuleBase> &new_net,
                    const std::vector<int> &selected_index) override;
};

// sum(k_i * x_i), k_i: vector.
struct SimpleLinearScaleBasisFunctionModule : BasisFunctionModuleBase {
    SimpleLinearScaleBasisFunctionModule(const boost::json::object &config, int window, int in_dim,
                                         int k);

    torch::Tensor forward(const torch::Tensor &x) override;

    torch::Tensor A, b;

    void copy_parms(std::shared_ptr<struct BasisFunctionModuleBase> &new_net,
                    const std::vector<int> &selected_index) override;
};

//(N, L, I) -> (N, k, O)
struct LSTMBasisFunctionModule : BasisFunctionModuleBase {
    LSTMBasisFunctionModule(const boost::json::object &config, int window, int in_dim, int out_dim,
                            int k);

    torch::Tensor forward(const torch::Tensor &x) override;

    std::vector<torch::nn::LSTM> lstms;
    std::vector<torch::nn::Linear> linears;

    void copy_parms(std::shared_ptr<struct BasisFunctionModuleBase> &new_net,
                    const std::vector<int> &selected_index) override;
};

struct BasisFunctionModuleFactory {
    std::shared_ptr<BasisFunctionModuleBase> operator()(const boost::json::object &config,
                                                        int window, int in_dim, int out_dim, int k);
};