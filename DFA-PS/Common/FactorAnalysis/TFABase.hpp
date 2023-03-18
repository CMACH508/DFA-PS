#pragma once
#include <torch/torch.h>
#include "SciLib/EigenHelper.hpp"
#include "../Util.hpp"
#include "../Net/WeightedSumModel.hpp"
#include <boost/circular_buffer.hpp>
#include "SciLib/TableWriter.hpp"

struct TFAParms {
    TMatd A, B;
    TVecd b, Sy, Sx; // Use a vector to store diagonal elements.

    TFAParms(){};
    // Normalized Xavier.
    // B is not initialized here.
    TFAParms(int dim, int hidden_dim);

    void write(const std::string &name) const;
    void read(const std::string &path);
};

using CBT = boost::circular_buffer<torch::Tensor>;

CBT zero_CBT(int size, std::initializer_list<int64_t> dims, torch::Device device);

// 1. elements is CBT is always 3D : (1,1,hidden_dim).

struct TFABase {
    TFABase(boost::json::object &config, int dim);
    virtual ~TFABase() = default;

    virtual void cal_y_hat(CBT &y0, const TVecd &x) = 0;
    virtual void cal_y_til(CBT &y0) = 0;

    virtual void cal_x_til() { x_til = parms.A * y_til + parms.b; };
    virtual void cal_x_hat() { x_hat = parms.A * y_hat + parms.b; };

    virtual void update_y0(CBT &y0) {
        torch::Tensor factor = SciLib::coldvec_to_3d_ftensor(y_hat, device);
        y0.push_back(factor);
    };
    virtual void cal_residuals(const TVecd &x);
    virtual void init(const TMatd &data);
    virtual void save(const std::string &name){};
    virtual void load(const std::string &name){};

    virtual int train(const DataSet &data, bool save_ssm_train_history = true,
                      const std::string &filename = "ssm_history.csv");
    virtual void before_epoch();
    virtual void before_train_batch(CBT &y0, const TMatd &X);
    virtual void train_batch(CBT &y0, const TMatd &X);
    virtual void before_update_parms(const TMatd &X){}; // For smoothing.
    virtual void update_parms();
    virtual void before_train_batch_single_loop(CBT &y0, const TVecd &x){};
    virtual void after_train_batch_single_loop(CBT &y0, const TVecd &x){};
    virtual void after_train_batch(CBT &y0, const TMatd &X);
    double cal_nll();
    virtual std::tuple<double, double, double, double, double> test(torch::Tensor &y0,
                                                                    const TMatd &X);

    //(recovered, predicted)
    virtual std::pair<torch::Tensor, torch::Tensor> generate_factors(CBT &y0, const TMatd &X,
                                                                     int predict_window);

    virtual void update_base_parms_buffer(){};
    virtual void backup() { _parms = parms; };
    virtual void restore() { std::swap(parms, _parms); };
    // 2D tensor.
    virtual torch::Tensor predict_hidden_factors(CBT &y0, int n = 1);
    // Both hidden factor and observed variable.
    virtual std::pair<TMatd, TMatd> predict(CBT &y0, int n = 1);
    virtual TMatd recover_hidden_factor(CBT &y0, const TMatd &X);
    // (Recoverd hidden factor y, reconstructed x).
    virtual std::pair<TMatd, TMatd> reconstruct(CBT &y0, const TMatd &X);

    virtual DataPool generate_data_pool(const DataSet &data, int window_size, int holding_period,
                                        bool only_use_periodic_price,
                                        const std::string &window_type);

    // virtual void const DataSet &print_table_header();
    virtual void print_parms();
    virtual void write_parms(const std::string &path_prefix);
    virtual void before_train(const DataSet &data);
    // Initialize y_0 before each epoch.

    virtual void after_train();
    virtual void after_epoch();
    // virtual void after_batch(CBT &y0){};

    int hidden_dim, dim = 0, window, seed, max_batch_size, epoch = 1, epochs, test_interval,
                    re_orthogonalize = -1;
    double lr;
    bool use_gpu = false, fresh_train = true;
    torch::Device device = torch::kCPU;
    spdlog::stopwatch sw;

    bool fix_Sx;
    // rows： samples.
    // For batch processing. col count = number of data points.
    int n = 0;
    TMatd batch, E, Varepsilon, Y_hat, Y_hat_1; // Y_hat stores hidden factors, doesn't containing
                                                // y_0, while the firset element of Y_hat_1 is y_0.
    TFAParms parms, _parms;
    // y_til (unconditional prediction), y_hat,x_til
    // (unconditional prediction given y_til), x_hat (prediction of x given y_hat),
    TVecd y_til, y_hat, x_til, x_hat, varepsilon, e;

    boost::json::object config;

    SciLib::TableWriter tb; // Print training history.

    //--------------------------------------------------------------------------------------------
    // Model selection.
    int ori_hidden_dim;
    bool model_selection;
    double model_select_threshold;
    double min_hidden_dim;
    std::vector<int> selected_index; // store index in current components. e.g. {0,5,6} means 0, 5,
                                     // 6 of current components will be selected;
    TVec<int>
        selected_index_map; // stores the original indices of components.  {0,5,6} means current has
                            // 3 components, the original index(from model is created) are 0, 5, 6.
    std::ofstream ori_Sy_file;

    virtual void init_model_selection();
    virtual void model_select(CBT &y0);
    virtual bool model_select_components(); // Result: if hidden dimension is adjusted.
    // virtual void adjust_hidden_dim(bool if_adjusted);
    virtual void adjust_base_parms();
    virtual void adjust_y0(CBT &y0);
};