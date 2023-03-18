#include "stdafx.h"
#include "Test.hpp"
#include "../SciLib/STDHelper.hpp"

// void run_system_with_config(boost::json::object &config, const std::string folder_name,
//                             const std::string &continue_backup) {
//     std::filesystem::create_directory(folder_name);
//
//     if (continue_backup != "") {
//         // continue
//         TFATradingSystem system;
//         system.load(continue_backup);
//         system.fresh_run();
//         system.save("backup/backup");
//
//     } else {
//         // Fresh run.
//         TFATradingSystem system(config);
//         system.fresh_run();
//         system.save("backup/backup");
//
//         // std::cout << "parms[0] of ssm: " << system.ssm->net->parms[0] << std::endl;
//     }
//
//     for (auto file : {"S_y.csv", "S_x.csv", "X.csv", "X-reconstructed.csv", "Y.csv",
//                       "train_recovered_hidden_factors.txt", "test_recovered_hidden_factors.txt",
//                       "ssm_history.csv", "decision_model_train_history.csv"}) {
//         std::filesystem::copy(std::filesystem::path(file), std::filesystem::path(folder_name),
//                               std::filesystem::copy_options::overwrite_existing);
//     }
// };

// void run_test(const std::string &dir) {
//     auto config_path = SciLib::path_join(dir, "full-config.json");
//
//
//     auto config = SciLib::read_config(config_path);
//     TFATradingSystem system;
//     // system.load("backup/backup");
//     system.test("backup/backup");
// }