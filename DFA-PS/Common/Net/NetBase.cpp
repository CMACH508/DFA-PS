#include "stdafx.h"
#include "NetBase.hpp"

// NetBase::NetBase() {};

void NetBase::report_net() {
    fmt::print("{:-^30}\n", "Model Report");
    int64_t n = 0;
    for (auto &ele : parms) {
        n += ele.numel();
    };
    fmt::print("Number of parameters: {}\n", n);
    fmt::print("{:-^30}\n", "");
};

void NetBase::backup() {
    torch::NoGradGuard nograd;
    for (int i = 0; i < parms.size(); ++i) {
        _parms[i].copy_(parms[i]);
    }
};

void NetBase::restore() {
    torch::NoGradGuard nograd;
    for (int i = 0; i < parms.size(); ++i) {
        parms[i].copy_(_parms[i]);
    }
};