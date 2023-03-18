#include "stdafx.h"
#include "LinearTFA.hpp"
#include "../Net/NetFactory.hpp"
#include "SciLib/EigenTorchHelper.hpp"
#include "../ModelUtil.hpp"
#include "../Net/ModelSelectionHelper.hpp"
#include "SciLib/EigenML.hpp"
#include "SciLib/MatrixManifolds.hpp"
// Create net.

LinearTFA::LinearTFA(boost::json::object &config, int dim) : TFABase(config, dim){};
void LinearTFA::init(const TMatd &data) {
    TFABase::init(data);
    pb.ori_Sy = parms.Sy;

    parms.B = TMatd::Zero(hidden_dim, hidden_dim);
    parms.B.diagonal() = TVecd::Random(hidden_dim);
    // parms.B.diagonal().setConstant(1);

    update_base_parms_buffer();
}

void LinearTFA::save(const std::string &name) {
    // save base data
    std::ofstream ofs(fmt::format("backup/{}/SSM_Class.bin", name));
    boost::archive::text_oarchive oa(ofs);
    oa << *this;
    ofs.close();
};

void LinearTFA::load(const std::string &name) {
    // load base data
    fmt::print(fmt::fg(fmt::color::yellow), "[LinearTFA]: Loading base data for NeuraNetTFA...\n");
    std::ifstream ifs(fmt::format("{}/SSM_Class.bin", name));
    boost::archive::text_iarchive ia(ifs);
    ia >> *this;

    this->init(TMatd());
}

void LinearTFA::cal_y_til(CBT &y0) {
    y_til = (parms.B.diagonal().array() * y_hat.array()).matrix();
}

void LinearTFA::cal_y_hat(CBT &y0, const TVecd &x) {
    TVecd b = pb.AT_x_iS_e * (x - parms.b) + (pb.iS_episilon.array() * y_til.array()).matrix();
    y_hat = pb.F_llt.solve(b);
    // y_hat = pb.iF * b;
}
void LinearTFA::update_parms() {
    TFABase::update_parms();
    // std::cout << "Before Max B: " << parms.B.diagonal().array().abs().maxCoeff() << std::endl;
    parms.B.diagonal().noalias() +=
        lr * (Varepsilon.transpose() * Y_hat_1).diagonal() / Y_hat_1.rows();
    /*double mb = parms.B.diagonal().array().abs().maxCoeff();
    std::cout << "After Max B: " << mb << std::endl;
    if (mb > 1) {
        SciLib::writematrix(Y_hat_1, "Y_hat_1.csv");
        SciLib::writematrix(Varepsilon, "Varepsilon.csv");
        std::terminate();
    }
    fmt::print("min Sx: {}\n", parms.Sx.diagonal().array().minCoeff());
    fmt::print("min Sy: {}\n", parms.Sy.diagonal().array().minCoeff());*/
    /* parms.B.diagonal().noalias() =
         ((Y_hat_1.transpose() * Y_hat_1).inverse() * Y_hat_1.transpose() * Y_hat).diagonal();*/
}
void LinearTFA::update_base_parms_buffer() {
    pb.iS_e = (1.0 / parms.Sx.array()).matrix();
    pb.iS_episilon = (1.0 / parms.Sy.array()).matrix();

    pb.F = parms.A.transpose() * pb.iS_e.asDiagonal() * parms.A;
    pb.F.diagonal().noalias() += pb.iS_episilon;
    // pb.iF = pb.F.inverse();
    pb.F_llt.compute(pb.F);
    pb.AT_x_iS_e = parms.A.transpose() * pb.iS_e.asDiagonal();

    TMatd m = pb.F.inverse() * parms.A.transpose() * pb.iS_e.asDiagonal();
}
void LinearTFA::adjust_base_parms() {
    parms.B = decltype(parms.B)(parms.B(selected_index, selected_index)); // Note the order.
    TFABase::adjust_base_parms();
}
// void LinearTFA::before_train(const TMatd &X) {
//
// };

void LinearTFA::before_epoch() {
    TFABase::before_epoch();
    Eigen::IOFormat mat_format(6, Eigen::DontAlignCols, ", ");
    ori_Sy_file << pb.ori_Sy.transpose().format(mat_format) << std::endl;
}

void LinearTFA::after_epoch() {
    static int i = 0;

    TFABase::after_epoch();
};
void LinearTFA::after_train_batch(CBT &y0, const TMatd &X) {
    TFABase::after_train_batch(y0, X);

    // update ori_Sy
    pb.ori_Sy.setZero();
    for (int i = 0; i < selected_index_map.size(); ++i) {
        pb.ori_Sy.coeffRef(selected_index_map.coeff(i)) = parms.Sy.coeffRef(i);
    }
}

namespace boost {
namespace serialization {
template <class Archive> void serialize(Archive &ar, TFAParms &p, const unsigned int version) {
    ar &p.A;
    ar &p.Sx;
    ar &p.Sy;
};

template <class Archive> void serialize(Archive &ar, LinearTFA &m, const unsigned int version) {
    ar &m.parms;
    ar &m._parms;

    ar &m.epoch;
    ar &m.hidden_dim;
};

}; // namespace serialization
}; // namespace boost

LinearTFARSSmoothing::LinearTFARSSmoothing(boost::json::object &config, int dim)
    : LinearTFA(config, dim) {}

void LinearTFARSSmoothing::update_base_parms_buffer() {
    LinearTFA::update_base_parms_buffer();

    // std::cout << parms.B.topLeftCorner(5, 5) << std::endl;
    TVecd BT_iS_B = (parms.B.diagonal().array().pow(2) * pb.iS_episilon.array()).matrix();
    spb.F = parms.A.transpose() * pb.iS_e.asDiagonal() * parms.A;
    spb.F.diagonal() += BT_iS_B;
    spb.F_llt.compute(spb.F);
};

void LinearTFARSSmoothing::before_update_parms(const TMatd &X) {
    TMatd X_b = X.rowwise() - parms.b.transpose();
    TMatd AT_S_X = X_b * pb.AT_x_iS_e.transpose();
    for (int i = n - 1; i >= 0; --i) {
        TVecd Y_hat_p_1 = Y_hat.row(i).transpose(); // y_{t+1}
        TVecd b = AT_S_X.row(i - 1).transpose() +   // x_t, note the index is i-1.
                  parms.B.transpose() * (pb.iS_episilon.array() * Y_hat_p_1.array()).matrix();
        TVecd y_hat_p = spb.F_llt.solve(b);
        Y_hat_1.row(i) = y_hat_p.transpose();
        if (i > 0)
            Y_hat.row(i - 1) = y_hat_p.transpose();
    }
    E = X_b - Y_hat * parms.A.transpose();
    Varepsilon = Y_hat - Y_hat_1 * parms.B.transpose();
}

LinearTFAIOSmoothingI::LinearTFAIOSmoothingI(boost::json::object &config, int dim)
    : LinearTFA(config, dim) {}

void LinearTFAIOSmoothingI::update_base_parms_buffer() {
    LinearTFA::update_base_parms_buffer();

    // std::cout << parms.B.topLeftCorner(5, 5) << std::endl;
    spb.F = parms.A.transpose() * pb.iS_e.asDiagonal() * parms.A;
    TVecd BT_iS_B = (parms.B.diagonal().array().pow(2) * pb.iS_episilon.array()).matrix();
    spb.F.diagonal() += BT_iS_B + pb.iS_episilon;

    spb.F_llt.compute(spb.F);
};

void LinearTFAIOSmoothingI::before_update_parms(const TMatd &X) {
    // if (epoch < 600)
    //     return;

    std::cout << "Before smoothing: " << -cal_nll() << std::endl;
    TMatd X_b = X.rowwise() - parms.b.transpose();
    TMatd AT_S_X = X_b * pb.AT_x_iS_e.transpose();
    for (int i = n - 1; i >= 1;
         --i) { // Note end condition is >=1 as we can't inference y_0 in this algorithm.
        TVecd Y_hat_p_1 = Y_hat.row(i).transpose();       // y_{t+1}
        TVecd Y_hat_m_1 = Y_hat_1.row(i - 1).transpose(); // y_{t-1}
        TVecd b =
            AT_S_X.row(i - 1).transpose() + // x_t
            (parms.B.diagonal().array() * pb.iS_episilon.array() * Y_hat_p_1.array()).matrix() +
            (pb.iS_episilon.array() * parms.B.diagonal().array() * Y_hat_m_1.array()).matrix();
        TVecd y_hat_p = spb.F_llt.solve(b);

        Y_hat_1.row(i) = y_hat_p.transpose();
        Y_hat.row(i - 1) = y_hat_p.transpose();
    }
    E = X_b - Y_hat * parms.A.transpose();
    Varepsilon = Y_hat - Y_hat_1 * parms.B.transpose();
    std::cout << "After smoothing: " << -cal_nll() << std::endl;
}

void FA::cal_y_hat(CBT &y0, const TVecd &x) {
    TVecd b = pb.AT_x_iS_e * (x - parms.b);
    y_hat = pb.F_llt.solve(b);
}

void FA::cal_residuals(const TVecd &x) {
    varepsilon = y_hat;
    e = x - x_hat;
}
