#pragma once
// #include "PredictiveTFA.hpp"
// #include "TFANet/TFANetBase.hpp"

#include "TFABase.hpp"

// Independent LSTM
// 1. Internal mod initialization is at constructor.
// 2. Data send to Neural net work is always (N, L, C).
// 3. If window > 1, we may not have enough data at first some steps, then we use zero.
// 4. The class itself doesn't have states (e.g. y0) as a member.
// 5. 3d (1, window, hidden_dim) y0 is explicitly passed as an argument so that it's not a member
// variable.
// 6. Net: (batch, window, hidden_dim) -> (batch, hidden_dim)
// 7. pb and parms is initialized in *init()*;

struct LinearTFAParmsBuffer {
    TVecd ori_Sy; // Use a vector to store diagonal elements.

    TVecd iS_episilon, iS_e;
    TMatd F, AT_x_iS_e, AT_S_X;
    Eigen::LLT<TMatd> F_llt;
};
struct LinearTFA : TFABase {
    LinearTFA(boost::json::object &config, int dim);

    void cal_y_til(CBT &y0) override;
    void cal_y_hat(CBT &y0, const TVecd &x) override;

    void init(const TMatd &data) override;

    void save(const std::string &name) override;
    void load(const std::string &name) override;

    LinearTFAParmsBuffer pb;

    //(n, dim)
    // void train_batch(CBT &y0, const TMatd &X) override;

    void update_base_parms_buffer() override;

    void adjust_base_parms() override;

    // void before_train(const TMatd &X) override;
    void before_epoch() override;
    void after_epoch() override;
    void after_train_batch(CBT &y0, const TMatd &X) override;
    void update_parms() override;
};

namespace boost {
namespace serialization {
template <class Archive> void serialize(Archive &ar, TFAParms &p, const unsigned int version);

template <class Archive> void serialize(Archive &ar, LinearTFA &m, const unsigned int version);
}; // namespace serialization
}; // namespace boost

struct LinearTFAASSmoothingBuffer {
    TMatd F;
    Eigen::LLT<TMatd> F_llt;
};
struct LinearTFARSSmoothing : LinearTFA {
    LinearTFARSSmoothing(boost::json::object &config, int dim);

    LinearTFAASSmoothingBuffer spb;
    void update_base_parms_buffer() override;
    void before_update_parms(const TMatd &X) override;
};

struct LinearTFAIOSmoothingI : LinearTFA {
    LinearTFAIOSmoothingI(boost::json::object &config, int dim);

    LinearTFAASSmoothingBuffer spb;
    void update_base_parms_buffer() override;
    void before_update_parms(const TMatd &X) override;
};

struct FA : LinearTFA {
    using LinearTFA::LinearTFA;

    void cal_y_hat(CBT &y0, const TVecd &x) override;
    using TFABase::adjust_base_parms;

    void cal_residuals(const TVecd &x) override;
    // using TFABase::update_parms;
    void update_parms() override {
        TFABase::update_parms();
        fmt::print("min Sx: {}\n", parms.Sx.diagonal().array().minCoeff());
        fmt::print("min Sy: {}\n", parms.Sy.diagonal().array().minCoeff());
    }
};