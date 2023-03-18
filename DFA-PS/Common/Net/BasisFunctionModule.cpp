#include "stdafx.h"
#include "BasisFunctionModule.hpp"
#include "ModelSelectionHelper.hpp"

BasisFunctionModuleBase::BasisFunctionModuleBase(int window, int in_dim, int out_dim, int k)
    : window(window), in_dim(in_dim), out_dim(out_dim), k(k) {}

void BasisFunctionModuleBase::report() {
    int n = 0;
    for (auto &ele : parameters()) {
        n += ele.numel();
    }

    fmt::print("Total parameters in BasisFunctionModule: {}\n", n);
}
//----------------------------------------------------------
SimpleLinearBasisFunctionModule::SimpleLinearBasisFunctionModule(const boost::json::object &config,
                                                                 int window, int in_dim, int k)
    : BasisFunctionModuleBase(window, in_dim, in_dim, k), A(torch::randn({k, window})),
      b(torch::randn({1, k, in_dim})) {
    register_parameter("weight-A", A);
    register_parameter("weight-b", b);

    for (auto &ele : {A, b}) {
        torch::nn::init::xavier_uniform_(ele);
    }
}

torch::Tensor SimpleLinearBasisFunctionModule::forward(const torch::Tensor &x) {
    int batch_sz = x.size(0);
    // std::cout << x << std::endl;
    auto x1 = torch::einsum("nli,kl->nki", {x, A});
    auto y = x1 + b.expand({batch_sz, -1, -1});
    return y;
}

void SimpleLinearBasisFunctionModule::copy_parms(std::shared_ptr<BasisFunctionModuleBase> &new_net,
                                                 const std::vector<int> &selected_index) {
    NET_CASTING_CHECKER(BasisFunctionModuleBase, SimpleLinearBasisFunctionModule);

    torch::NoGradGuard nograd;
    A.copy_(new_net_casted->A);
    b.copy_(new_net_casted->b.index_select(
        2, torch::tensor(selected_index).to(new_net_casted->b.options().device())));
}

//------------------------------------------------------------------------------------------
SimpleLinearScaleBasisFunctionModule::SimpleLinearScaleBasisFunctionModule(
    const boost::json::object &config, int window, int in_dim, int k)
    : BasisFunctionModuleBase(window, in_dim, in_dim, k), A(torch::randn({k, window, in_dim})),
      b(torch::randn({1, k, in_dim})) {
    register_parameter("weight-A", A);
    register_parameter("weight-b", b);

    for (auto &ele : {A, b}) {
        torch::nn::init::xavier_uniform_(ele);
    }
}

torch::Tensor SimpleLinearScaleBasisFunctionModule::forward(const torch::Tensor &x) {
    int batch_sz = x.size(0);
    // std::cout << x << std::endl;
    auto x1 = torch::einsum("nli,kli->nki", {x, A});
    auto y = x1 + b.expand({batch_sz, -1, -1});
    return y;
}

void SimpleLinearScaleBasisFunctionModule::copy_parms(
    std::shared_ptr<BasisFunctionModuleBase> &new_net, const std::vector<int> &selected_index) {
    NET_CASTING_CHECKER(BasisFunctionModuleBase, SimpleLinearScaleBasisFunctionModule);

    torch::NoGradGuard nograd;
    A.copy_(new_net_casted->A);
    b.copy_(new_net_casted->b.index_select(
        2, torch::tensor(selected_index).to(new_net_casted->b.options().device())));
}

//-----------------------------------------------------------------------------------------------------
LSTMBasisFunctionModule::LSTMBasisFunctionModule(const boost::json::object &config, int window,
                                                 int in_dim, int out_dim, int k)
    : BasisFunctionModuleBase(window, in_dim, out_dim, k) {
    for (int i = 0; i < k; ++i) {
        lstms.emplace_back(torch::nn::LSTM(
            torch::nn::LSTMOptions(in_dim,
                                   static_cast<int>(config.at("intermediate_output").as_int64()))
                .num_layers(static_cast<int>(config.at("layers").as_int64()))
                .batch_first(true)
                .bidirectional(false)));
        linears.emplace_back(torch::nn::Linear(torch::nn::Linear(
            window * static_cast<int>(config.at("intermediate_output").as_int64()), out_dim)));

        register_module(fmt::format("LSTM-{}", i), lstms[i]);
        register_module(fmt::format("Linears-{}", i), linears[i]);
    }
}
torch::Tensor LSTMBasisFunctionModule::forward(const torch::Tensor &x) {
    std::vector<torch::Tensor> Y;
    for (int i = 0; i < k; ++i) {
        auto [y, _] = lstms[i]->forward(x);
        torch::Tensor y1 = linears[i]->forward(y.flatten(1));
        Y.push_back(y1.unsqueeze(1));
    }
    return torch::cat(Y, 1);
}
void LSTMBasisFunctionModule::copy_parms(std::shared_ptr<BasisFunctionModuleBase> &new_net,
                                         const std::vector<int> &selected_index) {
    NET_CASTING_CHECKER(BasisFunctionModuleBase, LSTMBasisFunctionModule);

    auto selected_index_t = torch::tensor(selected_index);
    // std::cout << "Old paramters: " << new_net_casted->parameters()[0] << std::endl;
    for (int i = 0; i < k; ++i) {
        // LSTM part: only change factor_input.
        copy_lstm_helper(lstms[i], new_net_casted->lstms[i], selected_index, {});
        // Linear part
        copy_linear_helper(linears[i], new_net_casted->linears[i], {}, selected_index);
    }
    // std::cout << "New paramters: " << this->parameters()[0] << std::endl;
}

std::shared_ptr<BasisFunctionModuleBase>
BasisFunctionModuleFactory::operator()(const boost::json::object &config, int window, int in_dim,
                                       int out_dim, int k) {
    std::shared_ptr<BasisFunctionModuleBase> basis;
    auto basis_type = config.at("type").as_string();
    if (basis_type == "simple-linear") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "SimpleLinearBasisFunctionModule");
        fmt::print("...\n");
        basis = std::make_unique<SimpleLinearBasisFunctionModule>(config, window, in_dim, k);
    } else if (basis_type == "simple-linear-scale") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "SimpleLinearBasisFunctionModule");
        fmt::print("...\n");
        basis = std::make_unique<SimpleLinearScaleBasisFunctionModule>(config, window, in_dim, k);
    } else if (basis_type == "lstm") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "LSTMBasisFunctionModule");
        fmt::print("...\n");
        basis = std::make_unique<LSTMBasisFunctionModule>(config.at(basis_type).as_object(), window,
                                                          in_dim, out_dim, k);
    } else {
        fmt::print(fmt::fg(fmt::color::red), "Invalid basis type: {}.\n", basis_type);
        // throw std::runtime_error(fmt::format("Invalid basis type: {}.", basis_type));
        std::terminate();
    }
    basis->report();
    return basis;
}