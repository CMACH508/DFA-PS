#pragma once
#include "NetBase.hpp"
struct ENRBF : NetBase, torch::nn::Module {
    int k; // k: components, n: dimension
    // std::vector<torch::Tensor> W, c, mu, iS;
    torch::Tensor A, b, mu, iS;

    ENRBF(const boost::json::object &config, int n_in, int n_out);

    // void init_parm(const TMatf& Y) override;
    torch::Tensor forward(const torch::Tensor &) override;
    void init_parms(const torch::Tensor &iS) override;
    void to(torch::Device device) override;

    void copy_parms(std::shared_ptr<struct NetBase> &new_net,
                    const std::vector<int> &selected_index) override;
};