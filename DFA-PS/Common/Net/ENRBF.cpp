#include "stdafx.h"
#include "ENRBF.hpp"

ENRBF::ENRBF(const boost::json::object &config, int n_in, int n_out)
    : k(static_cast<int>(config.at("k").as_int64())) {
    A = register_parameter("A", torch::nn::init::xavier_uniform_(torch::zeros({n_out, k, n_in})));
    // A = register_parameter("A", torch::zeros({n_out, k, n_in}));
    b = register_parameter("b", torch::zeros({1, n_out, k}));
    mu = register_parameter("mu", torch::zeros({1, k, n_in}));

    auto iSh = torch::nn::init::xavier_uniform_(torch::zeros({k, n_in, n_in}));
    auto _iS = torch::einsum("nij,nlj->nil", {iSh, iSh});
    for (int i = 0; i < k; ++i) {
        _iS[i] = _iS[i].inverse();
    }
    iS = register_parameter("iS", _iS);

    // std::cout << "A: " << A.sizes() << std::endl;
    register_parms(this);
};

// sample*n_in
// void ENRBF::init_parm(const TMatf& Y) {
//	if (Y.cols() != n_in) {
//		throw std::runtime_error("Cols of Y is not equal to n_in.");
//	}
//
//	TVecf means = Y.colwise().mean().transpose();
//	torch::Tensor tmeans = torch::from_blob(means.data(), { n_in });
//	// std::cout << tmeans << std::endl;
//	// std::cout << "mu[i]" << mu[0] << std::endl;
//	for (size_t i = 0; i < k; ++i) {
//		// std::cout << i << std::endl;
//		// std::cout << "rand " << tmeans + torch::rand({n_in, 1}) * means.mean() <<
// std::endl; 		mu[0][i].variable_data().copy_(tmeans + torch::rand({ n_in }) *
// means.mean());
//	}
//
//	TMatf S_base = (Y.transpose() * Y / Y.rows()).inverse().llt().matrixU();
//	torch::Tensor S = torch::from_blob(S_base.data(), { n_in, n_in });
//	for (size_t i = 0; i < k; ++i) {
//		iS[i].variable_data().copy_(S);
//	}
//
//	// register_parameter("A", A);
//	// register_parameter("bb", b);
//	// register_parameter("mu", mu);
//	// register_parameter("Inverse Sigma", iS);
//
//	finallize();
//};

// x: (s, i), s for sample;
// current s=1.
// A: (o, k, i), x: (n_i, s), , w:
torch::Tensor ENRBF::forward(const torch::Tensor &pre_X) {
    // std::cout << "pre_X: " << pre_X.sizes() << std::endl;
    auto X = pre_X.squeeze(1);
    // std::cout << x << std::endl;
    // std::cout << x.unsqueeze(1) << std::endl;
    //
    // std::cout << X.narrow(0, 0, 1) << std::endl;
    int s = X.size(0);
    torch::Tensor Xmu = X.unsqueeze(1).expand({-1, k, -1}) - mu.expand({s, -1, -1}); // (s, k, i )
    // std::cout << "Xmu: " << Xmu.sizes() << " " << Xmu[0] << std::endl;
    //  torch::Tensor Ly = (x.unsqueeze(0).expand({k, -1, -1}) - mu).transpose(1, 2); // k*1*n_in
    //  torch::Tensor Ry = x.unsqueeze(0).expand({k, -1, -1}) - mu;                   // k*n_in*1
    //  std::cout << "Xmu: " << Xmu.narrow(0, 0, 5) << std::endl;

    // torch::Tensor e = torch::bmm(torch::bmm(Ly, iS), Ry).squeeze();   // k
    torch::Tensor w = torch::einsum("ski,kij,skj->sk", {Xmu, iS, Xmu}); // w: (s, k)
    /*if (s > 100) {
        std::cout << "w before in forward: " << w[69] << std::endl;
    }*/
    w = torch::nn::functional::softmax(-0.5 * w,
                                       torch::nn::functional::SoftmaxFuncOptions(1)); // (s, k)
    // std::cout << "w: " << w[0] << std::endl;

    // std::cout << "w: " << w.narrow(0, 0, 5) << std::endl;
    // std::cout << "nw: " << nw.narrow(0, 0, 5) << std::endl;

    // abort();

    torch::Tensor Axb =
        torch::einsum("si,oki->sok", {X, A}) + b.expand({s, -1, -1}); // Axb: (s, o, k)
    // std::cout << "Axb: " << Axb[0].narrow(0, 0, 5);
    torch::Tensor y = torch::einsum(
        "sok,sk->so", {Axb, w}); //(s, o)
                                 // std::cout << "y before softmax: " << y << std::endl;
                                 // return F::softmax(y, F::SoftmaxFuncOptions(1)); // n_outm
                                 // std::cout << "Xmu: " << Xmu.narrow(0, 0, 5) << std::endl;
                                 // std::cout << "w: " << w.narrow(0, 0, 5) << std::endl;
                                 // std::cout << "nw: " << nw.narrow(0, 0, 5) << std::endl;
                                 // std::cout << "A: " << A << std::endl;
                                 // std::cout << "b: " << b << std::endl;
                                 // std::cout << "x: " << X.narrow(0, 0, 1) << std::endl;
                                 // std::cout << "Axb: " << Axb.narrow(0, 0, 5) << std::endl;
    // std::cout << "y: " << y << std::endl;
    //  abort();
    // std::terminate();
    /*static int ct = 0;
    ++ct;
    std::cout << y << std::endl;
    if (ct > 3)
        std::terminate();*/
    /*   if (s > 100) {
           std::cout << "w in forward: " << w[69] << std::endl;
           std::cout << "y in forward: " << y[69] << std::endl;
       }*/
    return y;
};
void ENRBF::init_parms(const torch::Tensor &iS) {
    torch::NoGradGuard nograd;
    for (int i = 0; i < k; ++i) {
        auto disturb = torch::randn({iS.size(0), iS.size(1)}) / 1;
        // std::cout << iS + disturb * disturb.transpose(0, 1) << std::endl;
        // this->iS[i] = iS + disturb * disturb.transpose(0, 1);
        this->iS[i] = (iS + disturb * disturb.transpose(0, 1)).inverse();
    }
};

void ENRBF::to(torch::Device device) { torch::nn::Module::to(device); };

void ENRBF::copy_parms(std::shared_ptr<struct NetBase> &new_net,
                       const std::vector<int> &selected_index) {
    torch::NoGradGuard nograd;
    auto selected_index_t = torch::tensor(selected_index);

    auto _new_net = std::dynamic_pointer_cast<ENRBF>(new_net);
    A.copy_(_new_net->A.index_select(2, selected_index_t).index_select(0, selected_index_t));
    b.copy_(_new_net->b.index_select(1, selected_index_t));
    mu.copy_(_new_net->mu.index_select(2, selected_index_t));
    iS.copy_(_new_net->iS.index_select(2, selected_index_t).index_select(1, selected_index_t));
};