#ifndef __TFA__MLP__
#define __TFA__MLP__

#include "DecisionModelBase.hpp"
//#include "third_party/Eigen/Cholesky"

using namespace std;
using namespace DynAutoDiff;

struct MLP : DecisionModelBase {

    vector<shared_ptr<Var<>>> W, b;
    vector<int> sizes;
    int layers;

    MLP(boost::json::object &config, int n_in, int n_out) : DecisionModelBase(config, n_in, n_out) {

        sizes = json_int_array_to_vector(
            config["mlp"].as_object()["intermediate_layer_size"].as_array());
        sizes.insert(sizes.begin(), n_in);
        sizes.push_back(n_out);
        layers = static_cast<int>(sizes.size() - 1);

        W.resize(layers);
        b.resize(layers);

        for (int i = 0; i < layers; ++i) {
            auto node = pmat<>(sizes[i + 1], sizes[i]);
            node->setRandom();
            *node = node->val() / sqrt(node->cols() / 2);
            W[i] = node;

            auto _b = pvec<>(sizes[i + 1]);
            b[i] = _b;
        };

        register_parameter(W, b);
        finallize();
        report_parameter();
    };

    void init_parm(const TMatd &Y) override{

    };

    shared_ptr<Var<>> create_model(const shared_ptr<Var<>> &y) override {
        assert(y != nullptr);
        assert(y->size() > 0);
        auto x = linear(W[0], y, b[0]);
        for (int i = 1; i < W.size(); ++i) {
            x = linear(W[i], tanh(x), b[i]);
        }
        /*auto x = sigmoid(linear(W[0], y, b[0]));
        for (int i = 1; i < W.size(); ++i) {
            x = sigmoid(linear(W[i], x, b[i]));
        }*/
        return x;
    };
};

struct MLPN : DecisionModelBase {
    vector<vector<shared_ptr<Var<>>>> W, b;
    // vector<int> sizes;
    vector<vector<int>> sizes;
    size_t experts, layers;

    MLPN(boost::json::object &config, int n_in, int n_out)
        : DecisionModelBase(config, n_in, n_out) {
        sizes = json_2d_array_to_2d_vector<int>(
            config["mlpn"].as_object()["intermediate_layer_size"].as_array());
        cout << sizes[0][0] << endl;
        experts = sizes.size();
        W.resize(experts);
        b.resize(experts);

        for (int i = 0; i < experts; ++i) {
            sizes[i].insert(sizes[i].begin(), n_in);
            sizes[i].push_back(n_out);

            W[i].resize(sizes[i].size() - 1);
            b[i].resize(sizes[i].size() - 1);
        };

        for (int j = 0; j < experts; ++j) {
            for (int i = 0; i < W[j].size(); ++i) {
                auto node = pmat<>(sizes[j][i + 1], sizes[j][i]);
                node->setRandom();
                *node = node->val() / sqrt(node->cols() / 2);
                W[j][i] = node;

                auto _b = pvec<>(sizes[j][i + 1]);
                b[j][i] = _b;
            }
        };

        register_parameter(W, b);
        finallize();
        report_parameter();
    };

    void init_parm(const TMatd &Y) override{

    };

    shared_ptr<Var<>> create_model(const shared_ptr<Var<>> &y) override {
        assert(y != nullptr);
        assert(y->size() > 0);

        vector<shared_ptr<Var<>>> output(experts);
        // auto s = y->val().size();
        for (size_t i = 0; i < experts; ++i) {
            output[i] = linear(W[i][0], y, b[i][0]);
            for (size_t j = 1; j < W[i].size(); ++j) {
                output[i] = linear(W[i][j], tanh(output[i]), b[i][j]);
            }
        }

        /*auto x = sigmoid(linear(W[0], y, b[0]));
        for (int i = 1; i < W.size(); ++i) {
            x = sigmoid(linear(W[i], x, b[i]));
        }*/
        return sum(output);
    };
};

#endif // !__TFA__MLP__
