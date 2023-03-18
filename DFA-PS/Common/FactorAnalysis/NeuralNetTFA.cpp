#include "stdafx.h"
#include "NeuralNetTFA.hpp"
#include "../Net/NetFactory.hpp"
#include "SciLib/EigenTorchHelper.hpp"
#include "../ModelUtil.hpp"
#include "../Net/ModelSelectionHelper.hpp"
#include "SciLib/EigenML.hpp"
#include "SciLib/MatrixManifolds.hpp"
// Create net.

NeuralNetTFA::NeuralNetTFA(boost::json::object &config, int dim)
    : LinearTFA(config, dim),
      train_lr(config.at("neural_net_TFA").as_object().at("train_lr").as_double()),
      train_steps(static_cast<int>(config.at("neural_net_TFA").at("train_steps").as_int64())),
      subtrain_lr(config.at("neural_net_TFA").as_object().at("subtrain_lr").as_double()),
      subtrain_epochs(config.at("neural_net_TFA").as_object().at("subtrain_epochs").as_int64()) {
    // Deal with GPU.
    if (config["neural_net_TFA"].as_object()["use_gpu"].as_bool()) {
        if (torch::cuda::is_available()) {
            use_gpu = true;
#ifdef _WIN32
            LoadLibraryA("torch_cuda.dll");
#endif

        } else {
            fmt::print(fmt::fg(fmt::color::red), "[NeuralNetTFA]: GPU is set in config file, but "
                                                 "is not availiable. So CPU will be used.\n");
        }
    }
    device = use_gpu ? torch::kCUDA : torch::kCPU;
};
void NeuralNetTFA::init(const TMatd &data) {
    LinearTFA::init(data);

    net = create_net(config["neural_net_TFA"].as_object(), window, hidden_dim, hidden_dim);
    construct_opt();
}

void NeuralNetTFA::construct_opt() {
    if (use_gpu) {
        fmt::print("[NeuralNetTFA]: Using GPU...\n");
        net->to(torch::kCUDA);
    }
    opt = std::make_unique<torch::optim::SGD>(net->parms, torch::optim::SGDOptions(train_lr));
    subtrain_opt =
        std::make_unique<torch::optim::SGD>(net->parms, torch::optim::SGDOptions(subtrain_lr));

    fmt::print("[NeuralNetTFA]: Train LR: {}\n", train_lr);
    fmt::print("[NeuralNetTFA]: Subtrain LR: {}\n", subtrain_lr);
};

void NeuralNetTFA::save(const std::string &name) {
    // save base data
    std::ofstream ofs(fmt::format("backup/{}/SSM_Class.bin", name));
    boost::archive::text_oarchive oa(ofs);
    oa << *this;
    ofs.close();

    // save parameters
    torch::save(net->parms, fmt::format("backup/{}/SSM_NetParameters.pt", name));
    TFABase::write_parms(fmt::format("backup/{}/Parms", name));

    // save optimizer
    torch::serialize::OutputArchive oarch;
    opt->save(oarch);
    oarch.save_to(fmt::format("backup/{}/SSM_NetOptimizer.pt", name));
};

void NeuralNetTFA::load(const std::string &name) {
    std::cout << "Begin" << std::endl;
    // load base data
    fmt::print(fmt::fg(fmt::color::yellow),
               "[NeuralNetTFA]: Loading base data for NeuraNetTFA...\n");
    std::cout << SciLib::path_join(name, "SSM_Class.bin") << std::endl;
    std::ifstream ifs(SciLib::path_join(name, "SSM_Class.bin"));
    boost::archive::text_iarchive ia(ifs);
    ia >> *this;

    // this->init(TMatd());

    // load parameters
    fmt::print(fmt::fg(fmt::color::yellow),
               "[NeuralNetTFA]: Loading parameters for NeuraNetTFA...\n");
    load_parameters_from_backup(net, SciLib::path_join(name, "SSM_NetParameters.pt"));

    // load optimizer
    fmt::print(fmt::fg(fmt::color::yellow),
               "[NeuralNetTFA]: Loading optimizer for NeuraNetTFA...\n");
    torch::serialize::InputArchive iarch;
    iarch.load_from(SciLib::path_join(name, "SSM_NetOptimizer.pt"));
    opt->load(iarch);
};

void NeuralNetTFA::before_train_batch(CBT &y0, const TMatd &X) {
    TFABase::before_train_batch(y0, X);

    batch_input_buffer.clear();
    batch_target_buffer.clear();
}

void NeuralNetTFA::before_train_batch_single_loop(CBT &y0, const TVecd &x) {
    batch_input_buffer.emplace_back(torch::cat(std::vector(y0.begin(), y0.end()), 1));
}

void NeuralNetTFA::after_train_batch_single_loop(CBT &y0, const TVecd &x) {
    auto y_hat_t = SciLib::coldvec_to_3d_ftensor(y_hat, device);
    batch_target_buffer.emplace_back(y_hat_t.squeeze(0));
}
void NeuralNetTFA::cal_y_til(CBT &y0) {
    torch::NoGradGuard nograd;

    auto y = torch::cat(std::vector(y0.begin(), y0.end()), 1);
    auto fy = net->forward(y);
    y_til =
        TMapf(static_cast<float *>(fy.to(torch::kCPU).data_ptr()), hidden_dim, 1).cast<double>();
}

void NeuralNetTFA::update_parms() {
    TFABase::update_parms();
    auto batch_input = torch::cat(batch_input_buffer).detach_();
    auto batch_output_target = torch::cat(batch_target_buffer).detach_();

    update_net_parms(opt, batch_input, batch_output_target, train_steps);
}

torch::Tensor NeuralNetTFA::update_net_parms(std::shared_ptr<torch::optim::SGD> &opt,
                                             const torch::Tensor &input,
                                             const torch::Tensor &target, int epochs) {
    torch::Tensor _loss;
    for (int i = 0; i < epochs; ++i) {
        opt->zero_grad();
        auto batch_pred = net->forward(input);
        auto loss = calculate_target(batch_pred, target);
        loss.backward();
        opt->step();

        _loss = loss;
    }

    return _loss;
}
torch::Tensor NeuralNetTFA::calculate_target(const torch::Tensor &input,
                                             const torch::Tensor &output) {
    auto target = torch::nn::functional::mse_loss(input, output);
    return target;
};

void NeuralNetTFA::backup() {
    TFABase::backup();
    net->backup();
};

void NeuralNetTFA::restore() {
    TFABase::restore();
    net->restore();
};

void NeuralNetTFA::print_parms() {
    TFABase::print_parms();
    net->report_net();
};
void NeuralNetTFA::before_train(const DataSet &data) {
    LinearTFA::before_train(data);

    auto &X = data.train_returns;
    //   write_parms("E1S0");
    //  print_table_header();

    // Init enrbf
    if (fresh_train) {
        if (config["neural_net_TFA"].as_object()["net_type"].as_string() == "enrbf") {
            TMatd iS_epsilon = (1.0 / parms.Sy.array()).matrix().asDiagonal();
            TMatd iS_e = (1.0 / parms.Sx.array()).matrix().asDiagonal();
            TMatd F = (iS_epsilon + parms.A.transpose() * iS_e * parms.A)
                          .llt()
                          .solve(parms.A.transpose() * iS_e);
            TMatd C = X.transpose() * X / X.rows();
            TMatf mS = (F * C * F.transpose()).cast<float>();
            TMatf imS = mS.llt().solve(TMatf::Identity(hidden_dim, hidden_dim));
            torch::Tensor iS = SciLib::mat_to_tensor(mS);
            net->init_parms(iS);
        }
    }
};

void NeuralNetTFA::write_parms(const std::string &tag) {
    TFABase::write_parms(tag);
    net->write_parms(fmt::format("tmp/Net_{}", tag));

    SciLib::writematrix(batch, fmt::format("tmp/X_{}.csv", tag));
    if (!batch_input_buffer.empty()) {
        SciLib::writematrix(SciLib::tensor_to_mat<float>(torch::cat(batch_input_buffer).flatten(1)),
                            fmt::format("tmp/batch_input_{}.csv", tag));
    }
    if (!batch_target_buffer.empty()) {
        SciLib::writematrix(
            SciLib::tensor_to_mat<float>(torch::cat(batch_target_buffer).flatten(1)),
            fmt::format("tmp/batch_target_{}.csv", tag));
    }
    if (n > 0) {
        SciLib::writematrix(Varepsilon, fmt::format("tmp/Varepsilon_{}.csv", tag));
        SciLib::writematrix(E, fmt::format("tmp/E_{}.csv", tag));
    }
};

void NeuralNetTFA::model_select(CBT &y0) {

    auto q = model_select_components();
    if (q) {
        adjust_base_parms();
        if (subtrain_epochs > 0)
            prepare_subtrain_data();
        adjust_net_parms();
        if (subtrain_epochs > 0)
            model_select_subtrain();
        adjust_y0(y0);

        // print_table_header();
        tb.print_header();
    }
};

bool NeuralNetTFA::model_select_components() {
    auto [it1, it2] = std::minmax_element(parms.Sy.begin(), parms.Sy.end(), abs_compare<double>);
    // const double threshold = (*it2) * model_select_threshold;
    const double threshold = model_select_threshold;
    if ((*it1) < threshold) {
        // if ((*it1) < model_select_threshold) {
        //  Execute model selection.

        fmt::print(fmt::fg(fmt::color::yellow),
                   "[NeuralNetTFA]: Model selection start. Max variance: {: 7.4f}. Min variance: "
                   "{: 7.4f}.\n",
                   *it2, *it1);

        std::iota(selected_index.begin(), selected_index.end(), 0);

        int i1 = 0;
        std::vector<int> removed_index;
        for (int i2 = 0; i2 < hidden_dim; ++i2) {
            if ((parms.Sy.coeffRef(i2) > threshold)) {
                selected_index[i1] = i2;

                ++i1;
            } else {
                /* fmt::print(fmt::fg(fmt::color::yellow),
                            "[NeuralNetTFA]: Variance of component {} is {: 7.4f}. It's less then "
                            "{} of max "
                            "variance {: 7.4f}. Try remove it.\n",
                            i2, parms.Sy.coeffRef(i2), model_select_threshold, *it2);*/
                fmt::print(fmt::fg(fmt::color::yellow),
                           "[NeuralNetTFA]: Variance of component {} is {: 7.4f}. It's less then "
                           "threshold {: 7.4f}. Try remove it.\n",
                           i2, parms.Sy.coeffRef(i2), model_select_threshold, *it2);
                removed_index.emplace_back(i2);
            }
        }
        // If too much components are removed, than don't do it.
        if (i1 < min_hidden_dim * dim) {
            fmt::print(fmt::fg(fmt::color::red),
                       "After model selection, only {} will be kept which is too small. Model "
                       "selection is cancelled and turned off.",
                       i1);
            model_selection = false;
            return false;
        }
        selected_index.resize(i1);
        selected_index_map = TVec<int>(selected_index_map(selected_index));

        hidden_dim = selected_index.size();
        fmt::print(fmt::fg(fmt::color::yellow), "[NeuralNetTFA]: New hidden dimension {}.\n",
                   hidden_dim);
        return true;
    }

    return false;
}

void NeuralNetTFA::model_select_subtrain() {
    fmt::print(fmt::fg(fmt::color::yellow),
               "[NeuralNetTFA]: Subtrain for model selection begins...\n");

    for (int i = 1; i <= subtrain_epochs; ++i) {

        auto loss = update_net_parms(subtrain_opt, subtrain_input, subtrain_target);
        fmt::print(fmt::fg(fmt::color::yellow), "[NeuralNetTFA]: Epoch {}, loss: {}...\n", i,
                   loss.item<float>());
    }
}

void NeuralNetTFA::prepare_subtrain_data() {
    fmt::print(fmt::fg(fmt::color::yellow), "[NeuralNetTFA]: Prepare subtrain data.\n");
    auto selected_index_t = torch::tensor(selected_index).to(device);
    auto batch_input = torch::cat(batch_input_buffer).detach_();
    subtrain_input = batch_input.index_select(2, selected_index_t).clone().detach_();
    //[Note]: we want the new net to approximate result of original net.
    auto batch_output_target = net->forward(batch_input);
    subtrain_target = batch_output_target.index_select(1, selected_index_t).clone().detach_();
};
void NeuralNetTFA::adjust_base_parms() {
    fmt::print(fmt::fg(fmt::color::yellow), "[NeuralNetTFA]: Now adjusting base parameters.\n");
    parms.Sy = decltype(parms.Sy)(parms.Sy(selected_index));
    parms.A = decltype(parms.A)(parms.A(Eigen::indexing::all, selected_index));
    update_base_parms_buffer();
};
void NeuralNetTFA::adjust_net_parms() {
    fmt::print(fmt::fg(fmt::color::yellow), "[NeuralNetTFA]: Now copying net parameters.\n");
    // Create new net and copy parameters.
    auto new_net = create_net(config["neural_net_TFA"].as_object(), window, hidden_dim, hidden_dim);
    new_net->copy_parms(net, selected_index);
    net = new_net;

    construct_opt();
};

void NeuralNetTFA::adjust_y0(CBT &y0) {
    torch::NoGradGuard nograd;
    auto selected_index_t = torch::tensor(selected_index).to(device);
    for (auto &ele : y0) {
        ele = ele.index_select(2, selected_index_t);
    }
};
namespace boost {
namespace serialization {
template <class Archive> void serialize(Archive &ar, TFAParms &p, const unsigned int version) {
    ar &p.A;
    ar &p.Sx;
    ar &p.Sy;
};

template <class Archive> void serialize(Archive &ar, NeuralNetTFA &m, const unsigned int version) {
    ar &m.parms;
    ar &m._parms;

    ar &m.epoch;
    ar &m.hidden_dim;
};

}; // namespace serialization
}; // namespace boost
