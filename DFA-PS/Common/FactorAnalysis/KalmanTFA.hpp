#pragma once
#include "TFABase.hpp"

struct KalmanTFA : TFABase {
    KalmanTFA(boost::json::object &config, int dim);

    TMatd P_t_t, K_t, P_tp1_t;
    // TVecd y_tp1_t;

    void init(const TMatd &data) override;
    void before_epoch();
    // void before_train_batch(CBT &y0, const TMatd &X) override;
    // void cal_x_hat() override;
    void cal_y_til(CBT &y0) override;
    void cal_y_hat(CBT &y0, const TVecd &x) override;
    // void update_base_parms_buffer() override;
    void update_parms() override;
    void adjust_base_parms() override;
};

struct KalmanTFARTSSmoothing : KalmanTFA {
    // KalmanTFARTSSmoothing(boost::json::object &config, int dim);
    using KalmanTFA::KalmanTFA;

    // TMatd P_t_prime, G, Y_hat_tp1_t;
    std::vector<TMatd> P_t_t_, P_tp1_t_;
    std::vector<TVecd> Y_til;

    // void update_base_parms_buffer() override;
    void before_epoch() override;
    void after_train_batch_single_loop(CBT &y0, const TVecd &x) override; // Collect data

    void before_update_parms(const TMatd &X) override;
};