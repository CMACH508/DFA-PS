#pragma once
#include <string>
#include "DecisionModel/DecisionModelBase.hpp"
#include "FactorAnalysis/NeuralNetTFA.hpp"
#include "Util.hpp"
#include "FactorAnalysis/TFAFactory.hpp"

struct TFATradingSystem {
    TFATradingSystem();
    TFATradingSystem(const std::string &file);
    TFATradingSystem(const boost::json::object &config);
    void init_by_config();
    void contruct_model();

    // 4 files will be needed for save/load: {name}_config.json,{name}_SSM_Class.bin,
    // {name}_SSM_parms.pt, {name}_DecisionModel.bin, {name}_DecisionModel.pt
    void load(const std::string &name);
    void save(const std::string &name);

    boost::json::object config;
    bool use_precomputed_hidden_factor = false, use_gpu = false, only_use_periodic_price = true;
    std::string precomputed_hidden_factor_path, window_type;
    int decision_window = 1, decision_holding_period;
    DataSet data;
    DataPool data_pool;

    std::unique_ptr<TFABase> fa = nullptr;
    // int dim, hidden_dim;
    std::shared_ptr<DecisionModelBase> decision_model = nullptr;

    void read_config(const std::string &file);
    void read_data();
    void write_ssm_reconstructed_data();
    void generate_decision_model_data_pool();

    void create_ssm();
    void load_ssm(const std::string &name);
    void before_train_fa();
    void train_fa();
    void after_train_fa();
    void create_decision_model();
    void load_decision_model(const std::string &name);
    void train_decision_model(const std::string &id);
    void read_precomputed_hidden_factor();

    // "backup/uuid"
    void test(const std::string &backup);

    // void run(int mode, const std::string &name);
    void fresh_run(const std::string &id);
};