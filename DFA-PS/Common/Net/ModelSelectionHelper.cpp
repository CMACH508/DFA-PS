#include "stdafx.h"
#include "ModelSelectionHelper.hpp"

void copy_lstm_helper(torch::nn::LSTM &this_m, const torch::nn::LSTM &that_m,
                      const std::vector<int> &input_index, const std::vector<int> &output_index) {

    torch::NoGradGuard nograd;

    const int that_out_dim = this_m->options.hidden_size();

    auto this_p = this_m->parameters();
    auto that_p = that_m->parameters();
    torch::Device device = that_p[0].options().device();

    auto input_index_t = torch::tensor(input_index).to(device),
         output_index_t = torch::tensor(output_index).to(device);

    for (int i = 0; i < that_p.size(); ++i) {
        if (this_p[i].dim() == 2) {
            if (this_p[i].sizes() == that_p[i].sizes()) {
                // If elements are equal.
                this_p[i].copy_(that_p[i]);
                continue;
            }
            auto target = that_p[i].clone();
            if ((input_index.size() > 0)) {
                target = target.index_select(1, input_index_t);
            }
            if (output_index.size() > 0) {
                torch::Tensor row_indices = torch::cat({input_index_t, input_index_t + that_out_dim,
                                                        input_index_t + 2 * that_out_dim,
                                                        input_index_t + 3 * that_out_dim});
                target = target.index_select(0, row_indices);
            }
            // std::cout << "This: " << this_p[i].sizes() << std::endl << target.sizes() <<
            // std::endl;
            this_p[i].copy_(target);
        } else {
            if (output_index.size() > 0)
                this_p[i].copy_(that_p[i].index_select(0, output_index_t));
            else
                this_p[i].copy_(that_p[i]);
        }
    }
}

void copy_linear_helper(torch::nn::Linear &this_m, torch::nn::Linear that_m,
                        const std::vector<int> &input_index, const std::vector<int> &output_index) {

    torch::NoGradGuard nograd;

    const int out_dim = this_m->options.out_features();

    auto this_p = this_m->parameters();
    auto that_p = that_m->parameters();
    auto input_index_t = torch::tensor(input_index), output_index_t = torch::tensor(output_index);

    for (int i = 0; i < that_p.size(); ++i) {
        if (this_p[i].dim() == 2) {
            auto target = that_p[i].clone();
            if ((input_index.size() > 0)) {
                target = target.index_select(1, input_index_t);
            }
            if (output_index.size() > 0) {
                target = target.index_select(0, output_index_t);
            }
            this_p[i].copy_(target);
        } else {
            if (output_index.size() > 0)
                this_p[i].copy_(that_p[i].index_select(0, output_index_t));
            else
                this_p[i].copy_(that_p[i]);
        }
    }
}