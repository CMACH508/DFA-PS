#pragma once

#include <torch/torch.h>

template <typename ModelType>
void load_parameters_from_backup(ModelType &model, const std::string &file) {
    std::vector<torch::Tensor> parameters;
    torch::load(parameters, file);
    if (parameters.size() != model->parms.size()) {
        fmt::print(
            fmt::fg(fmt::color::red),
            "Number of parameters in net and backup is different. The backup maybe corrupted.\n");
    } else {
        const int N = model->parms.size();
        torch::NoGradGuard nograd;
        for (int i = 0; i < N; ++i) {
            model->parms[i].copy_(parameters[i]);
        }
    }
};