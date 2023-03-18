#pragma once
#include <torch/torch.h>
#include "SciLib/EigenHelper.hpp"
#include "SciLib/EigenTorchHelper.hpp"

// 1. Input: (batch, window, in_dim), Output: (batch, out_dim)
// 2. Remember to call register_parms() in derived struct.
struct NetBase {
    virtual ~NetBase() = default;

    // Push parameters in finalize().
    virtual torch::Tensor forward(const torch::Tensor &x) = 0;
    virtual void to(torch::Device device) = 0;

    template <typename T, typename... Ts> void register_parms_(T m, Ts... modules) {
        const auto m_parms = m->parameters();
        parms.insert(parms.end(), m_parms.begin(), m_parms.end());
    };
    template <typename T, typename... Ts> void register_parms(T m, Ts... modules) {
        register_parms_(m);

        if constexpr (sizeof...(modules) != 0) {
            // std::cout << "register parameters." << std::endl;

            register_parms(modules...);
        } else {
            // std::cout << "First back up of parameters." << std::endl;
            _parms.resize(parms.size());
            for (int i = 0; i < parms.size(); ++i) {
                _parms[i] = parms[i].detach().clone();
            }
        }
    };

    std::vector<torch::Tensor> parms, _parms;

    virtual void backup();
    virtual void restore();

    virtual void report_net();
    virtual void init_parms(const torch::Tensor &iS){};
    // size of new_net is usually larger than this.
    virtual void copy_parms(std::shared_ptr<struct NetBase> &new_net,
                            const std::vector<int> &selected_index){};
    virtual void write_parms(const std::string &filename){};
};

struct NativeTFALinear : NetBase, torch::nn::Module {
    torch::Tensor w;

    NativeTFALinear(int n_in) : w(register_parameter("w", torch::zeros({n_in}).uniform_(-1, 1))) {
        register_parms(this);
    };

    // void init_parm(const TMatf& Y) override;
    torch::Tensor forward(const torch::Tensor &pre_X) override {
        auto X = pre_X.squeeze(1);
        auto y = X * w;
        return y;
    };
    void to(torch::Device device) override { torch::nn::Module::to(device); };

    void copy_parms(std::shared_ptr<struct NetBase> &new_net,
                    const std::vector<int> &selected_index) override {
        torch::NoGradGuard nograd;
        auto selected_index_t = torch::tensor(selected_index);

        auto _new_net = std::dynamic_pointer_cast<struct NativeTFALinear>(new_net);
        w.copy_(_new_net->w.index_select(0, selected_index_t));
    };

    void write_parms(const std::string &filename) override {
        SciLib::writematrix(SciLib::tensor_to_mat<float>(w), filename + "_w.csv");
    };
};