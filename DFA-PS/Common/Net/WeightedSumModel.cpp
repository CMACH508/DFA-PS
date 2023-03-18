#include "stdafx.h"
#include "WeightedSumModel.hpp"

WeightedSumModel::WeightedSumModel(const boost::json::object &config, int window, int in_dim,
                                   int out_dim)
    : k(static_cast<int>(config.at("k").as_int64())),
      basis(
          BasisFunctionModuleFactory()(config.at("basis").as_object(), window, in_dim, out_dim, k)),
      weight(WeightFunctionModuleFactory()(config.at("weight").as_object(), window, in_dim, k)) {

    register_parms(basis, weight);
};

torch::Tensor WeightedSumModel::forward(const torch::Tensor &x) {
    auto _basis = basis->forward(x);
    auto _weight = weight->forward(x);
    auto res = torch::einsum("nkc,nk->nc", {_basis, _weight});
    return res;
};

void WeightedSumModel::to(torch::Device device) {
    // TFANetBase::set_use_gpu();

    basis->to(device);
    // LoadLibraryA("torch_cuda.dll");
    weight->to(device);
};

// std::vector<torch::Tensor> WeightedSumModel::parameters() { return parms; }

void WeightedSumModel::copy_parms(std::shared_ptr<NetBase> &new_net,
                                  const std::vector<int> &selected_index) {

    std::shared_ptr<WeightedSumModel> new_net_casted;
    try {
        new_net_casted = std::dynamic_pointer_cast<WeightedSumModel>(new_net);
    } catch (...) {
        fmt::print(fmt::fg(fmt::color::red),
                   "[WeightSumModel]: Can't cast shared_ptr<NetBase> to "
                   "shared_ptr<WeightedSumModel>. Please check passes pointer.");
        std::terminate();
    }
    // NetBase::copy_parms(selected_index);
    basis->copy_parms(new_net_casted->basis, selected_index);
    weight->copy_parms(new_net_casted->weight, selected_index);
}