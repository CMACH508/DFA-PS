#include "stdafx.h"

#include "RatioModuleImpl.hpp"

RatioModuleImpl::RatioModuleImpl(const RatioModuleOption &option) : option(option) {
    torch::nn::Sequential s;
    int in = option.C, out;
    for (int i = 0; i < option.channel.size(); ++i) {
        out = option.channel[i];
        s->push_back(ResBasic1dBlock(in, out, option.stride[i]));

        in = out;
    }
    seq = register_module("seq", s);

    pool = register_module("pool",
                           torch::nn::AdaptiveAvgPool1d(torch::nn::AdaptiveAvgPool1dOptions(1)));
    linear1 = register_module("linear1", torch::nn::Linear(out, 3));

    out = 3 + option.extra_feature_size * 2;

    bn2 = register_module("bn2", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(out)));

    torch::nn::Sequential _linear_seq;
    for (size_t i = 0; i < option.linear_seq_size.size(); ++i) {
        _linear_seq->push_back(torch::nn::Linear(out, option.linear_seq_size[i]));
        if (i < option.linear_seq_size.size() - 1)
            _linear_seq->push_back(torch::nn::ReLU());
        out = option.linear_seq_size[i];
    }
    linear_seq = register_module("linear_seq", _linear_seq);
}

torch::Tensor RatioModuleImpl::forward(const torch::Tensor &input,
                                       const std::vector<torch::Tensor> &extra) {
    auto y = seq->forward(input);
    y = linear1(pool(y).flatten(1));
    y = torch::cat({y, std::get<0>(extra[0].min(1, true)), std::get<0>(extra[0].max(1, true)),
                    extra[0].sum(1, true), std::get<0>(extra[1].min(1, true)),
                    std::get<0>(extra[1].max(1, true)), extra[1].sum(1, true)},
                   1);
    return linear_seq->forward(bn2(y));
};