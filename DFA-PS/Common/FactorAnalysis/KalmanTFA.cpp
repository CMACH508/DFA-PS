#include "stdafx.h"
#include "KalmanTFA.hpp"

KalmanTFA::KalmanTFA(boost::json::object &config, int dim) : TFABase(config, dim) {}

void KalmanTFA::init(const TMatd &data) {
    TFABase::init(data);

    parms.B = TMatd::Zero(hidden_dim, hidden_dim);
    parms.B.diagonal() = TVecd::Random(hidden_dim);
    parms.Sy = (parms.Sy.array() * (1 - parms.B.diagonal().array())).matrix();

    P_tp1_t = TMatd::Identity(hidden_dim, hidden_dim);
}

void KalmanTFA::before_epoch() {
    TFABase::before_epoch();
    P_tp1_t = TMatd::Identity(hidden_dim, hidden_dim);
}

void KalmanTFA::cal_y_til(CBT &y0) {
    y_til = (parms.B.diagonal().array() * y_hat.array()).matrix();
}

void KalmanTFA::cal_y_hat(CBT &y0, const TVecd &x) {
    TMatd APAT = parms.A * P_tp1_t * parms.A.transpose();
    APAT.diagonal().noalias() += parms.Sx;
    K_t = (APAT.transpose().llt().solve((P_tp1_t * parms.A.transpose()).transpose())).transpose();
    y_hat = y_til + K_t * (x - x_til);

    P_t_t = (TMatd::Identity(hidden_dim, hidden_dim) - K_t * parms.A) * P_tp1_t;
    // y_tp1_t = (parms.B.diagonal().array() * y_hat.array()).matrix();
    P_tp1_t = parms.B * P_t_t * parms.B.transpose();
    P_tp1_t.diagonal() += parms.Sy;
}

void KalmanTFA::update_parms() {
    TFABase::update_parms();
    parms.B.diagonal().noalias() +=
        lr * (Varepsilon.transpose() * Y_hat_1).diagonal() / Y_hat_1.rows();
}
void KalmanTFA::adjust_base_parms() {
    P_tp1_t = TMatd(P_tp1_t(selected_index, selected_index));
    parms.B = decltype(parms.B)(parms.B(selected_index, selected_index));
    TFABase::adjust_base_parms();
}
void KalmanTFARTSSmoothing::before_epoch() {
    KalmanTFA::before_epoch();
    P_t_t_.clear();
    P_tp1_t_.clear();
    Y_til.clear();
}
void KalmanTFARTSSmoothing::after_train_batch_single_loop(CBT &y0, const TVecd &x) {
    P_t_t_.push_back(P_t_t);
    P_tp1_t_.push_back(P_tp1_t);
    Y_til.push_back(y_til);
}
void KalmanTFARTSSmoothing::before_update_parms(const TMatd &X) {
    TMatd P_t_prime = P_t_t_.back();
    TVecd y_hat_p = Y_hat.bottomRows(1).transpose();
    // Note: Y_hat , P_t_t_t, P_tp1_t_, Y_til are aligned.
    for (int i = n - 2; i >= 0; --i) { // Here use Y_hat for time reference.
        TMatd &P_t_t = P_t_t_[i];
        TMatd &P_tp1_t = P_tp1_t_[i];
        TVecd &y_tp1_t = Y_til[i];

        // TMatd G_t = P_t_t * parms.B.diagonal().asDiagonal() * P_tp1_t.inverse();
        TMatd G_t =
            (P_tp1_t.transpose().llt().solve(parms.B.diagonal().asDiagonal() * P_t_t)).transpose();
        P_t_prime = P_t_t - G_t * (P_tp1_t - P_t_prime) * G_t.transpose();
        y_hat_p = Y_hat.row(i).transpose() + G_t * (y_hat_p - y_tp1_t);

        Y_hat_1.row(i + 1) = y_hat_p.transpose();
        Y_hat.row(i) = y_hat_p.transpose();
    }

    TMatd X_b = X.rowwise() - parms.b.transpose();
    E = X_b - Y_hat * parms.A.transpose();
    Varepsilon = Y_hat - Y_hat_1 * parms.B.transpose();
}
