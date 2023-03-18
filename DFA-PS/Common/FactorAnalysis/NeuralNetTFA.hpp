#pragma once

#include "LinearTFA.hpp"

// Independent LSTM
// 1. Internal mod initialization is at constructor.
// 2. Data send to Neural net work is always (N, L, C).
// 3. If window > 1, we may not have enough data at first some steps, then we use zero.
// 4. The class itself doesn't have states (e.g. y0) as a member.
// 5. 3d (1, window, hidden_dim) y0 is explicitly passed as an argument so that it's not a member
// variable.
// 6. Net: (batch, window, hidden_dim) -> (batch, hidden_dim)
// 7. pb and parms is initialized in *init()*;

struct NeuralNetTFA : LinearTFA {
    NeuralNetTFA(boost::json::object &config, int dim);
    void cal_y_til(CBT &y0) override;

    void init(const TMatd &data) override;
    virtual void construct_opt();
    void save(const std::string &name) override;
    void load(const std::string &name) override;
    void backup() override;
    void restore() override;

    void before_train_batch(CBT &y0, const TMatd &X) override;
    void before_train_batch_single_loop(CBT &y0, const TVecd &x) override;
    void after_train_batch_single_loop(CBT &y0, const TVecd &x) override;

    // NeuralNetTFABuffer buffer;
    std::vector<torch::Tensor> batch_input_buffer, batch_target_buffer;

    // TVecd y_hat_1;
    //  torch::Tensor y_hat_t;

    // Neural Net.
    double train_lr = 5e-3;
    int train_steps = 1;
    std::shared_ptr<torch::optim::SGD> opt, subtrain_opt;
    std::shared_ptr<NetBase> net;

    // subtrain for model selection.
    double subtrain_lr;
    int subtrain_epochs;
    torch::Tensor subtrain_input, subtrain_target;

    //(n, dim)
    // void train_batch(CBT &y0, const TMatd &X) override;

    virtual void update_parms() override;
    virtual torch::Tensor update_net_parms(std::shared_ptr<torch::optim::SGD> &opt,
                                           const torch::Tensor &input, const torch::Tensor &output,
                                           int epochs = 1);
    virtual torch::Tensor calculate_target(const torch::Tensor &input, const torch::Tensor &output);
    // virtual std::pair<torch::Tensor, torch::Tensor> predict(int n = 1);

    virtual void model_select(CBT &y0) override;
    virtual bool model_select_components() override; // Result: if hidden dimension is adjusted.
    virtual void prepare_subtrain_data();
    // virtual void adjust_hidden_dim(bool if_adjusted);
    virtual void adjust_base_parms() override;
    virtual void adjust_net_parms();
    virtual void adjust_y0(CBT &y0) override;
    virtual void model_select_subtrain();

    void before_train(const DataSet &data) override;

    void print_parms() override;
    void write_parms(const std::string &tag);
};

struct NeuralNetTFAIOSmoothing : NeuralNetTFA {
    NeuralNetTFAIOSmoothing(boost::json::object &config, int dim) : NeuralNetTFA(config, dim){};

    void before_update_parms(const TMatd &X) override;
};

namespace boost {
namespace serialization {
template <class Archive> void serialize(Archive &ar, TFAParms &p, const unsigned int version);

template <class Archive> void serialize(Archive &ar, NeuralNetTFA &m, const unsigned int version);
}; // namespace serialization
}; // namespace boost
