#include "stdafx.h"

#define BOOST_UUID_FORCE_AUTO_LINK

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <random>
#include <thread>
#include "TFATradingSystem.hpp"

#include "SciLib/STDHelper.hpp"

namespace po = boost::program_options;

std::string generate_sid() {
    static boost::uuids::random_generator_mt19937 gen;
    return to_string(gen());
};

// base, override, full
auto read_configs(const po::variables_map &vm) {
    std::string base_config_file, override_config_file;
    try {
        base_config_file = SciLib::absolute_path(vm["config"].as<std::string>());
        override_config_file =
            vm.count("override") ? SciLib::absolute_path(vm["override"].as<std::string>()) : "";
    } catch (...) {
        fmt::print(fmt::fg(fmt::color::red), "Failed to parse command line arguments.");
    }

    fmt::print(fmt::fg(fmt::color::yellow), "Fresh train. Base config: {}, override config: {}.\n",
               base_config_file, override_config_file);

    boost::json::object base_config = SciLib::read_config(base_config_file), override_config,
                        full_config = base_config;
    if (!override_config_file.empty()) {
        override_config = SciLib::read_config(override_config_file);
        full_config = SciLib::merge_json(full_config, override_config);
    }

    return std::make_tuple(base_config, override_config, full_config);
}

// Here sid is a name used for backup.
void run_once(const std::string &sid, const boost::json::object &base_config,
              const boost::json::object &override_config, const boost::json::object &full_config_) {

    fmt::print(fmt::fg(fmt::color::yellow), "ID: {}\n", sid);

    // Backup configs.
    std::filesystem::create_directories(fmt::format("backup/{}", sid));
    SciLib::write_json(base_config, fmt::format("backup/{}/base-config.json", sid));
    if (!override_config.empty()) {
        SciLib::write_json(override_config, fmt::format("backup/{}/override-config.json", sid));
    }
    auto full_config = rewrite_config_paths(full_config_);
    SciLib::write_json(full_config, fmt::format("backup/{}/full-config.json", sid));

    TFATradingSystem system(full_config);
    system.fresh_run(sid);
    system.save(sid);

    auto pdesc = system.decision_model->get_simple_performace_desc();
    // std::ofstream(fmt::format("history/[{}]-{}/config.json", pdesc, sid)) << override_config;
    SciLib::write_json(override_config, fmt::format("history/[{}]-{}/config.json", pdesc, sid));
};

// Run with mode.
void run_main(int mode, const std::string &sid, boost::json::object &base_config,
              boost::json::object &override_config, boost::json::object &full_config) {
    switch (mode) {
    case 0:
        fmt::print(fmt::fg(fmt::color::yellow), "[Mode 0]: Running from config file.\n");
        run_once(sid, base_config, override_config, full_config);
        break;

    case 1:
        fmt::print(fmt::fg(fmt::color::yellow), "[Mode 1]: Only running SSM.\n");
        full_config["decision_model"].as_object()["use_precomputed_hidden_factor"] = false;
        full_config["decision_model"].as_object()["epochs"] = 10;
        run_once(sid, base_config, override_config, full_config);
        break;
    default:
        break;
    }
};

int main(int argc, char **argv) {

#ifdef _WIN32
    LoadLibraryA("torch_cuda.dll");
#endif

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()(
        "help,H",
        fmt::format(
            "Produce help message.\n\nExamples:\n\n{0}    : training with default "
            "config file config.json\n\n{0} -C myconfig.json    config file is "
            "myconfig.json.\n\n{0} -C ../base_config.json -OC config.json    : use config.json to "
            "override base_config.json.\n\n{0} -M1     Use default backup name backup/backup to "
            "run mode 1.",
            SciLib::get_filename(argv[0]))
            .c_str())("name", po::value<std::string>()->default_value(""),
                      "Name for saving backup. If it's not set, then a UUID will be used.")(
        "config,C", po::value<std::string>(), "config file for training.")(
        "override,O", po::value<std::string>(), "use this config to override base config.")(
        "mode,M", po::value<int>()->default_value(0),
        "Mode 0: Run from config file.\n"
        "Mode 1: Only run SSM. use_precomputed_hidden_factor will be set to false. Epochs for "
        "decision model will be set to 10.\n"
        "Mode 2: Only run decision model. use_precomputed_hidden_factor will be set to true.")(
        "load-backup,L", po::value<std::string>()->default_value("backup"),
        "Valid when mode are set.")("backup,B", po::value<std::string>()->default_value("backup/"),
                                    "Set a name for backup.")(
        "repeat", po::value<int>()->default_value(1), "Repeat run with different random seeds.")(
        "sim", po::value<int>()->default_value(2), "Simultaneously run multiple tasks in one GPU.")(
        "test,T", po::value<std::string>(), "Given a directory of backup, run test.");

    po::positional_options_description p;
    p.add("config", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    // Print help
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    /*if (vm["test"].as<bool>()) {
        fmt::print(fmt::fg(fmt::color::orange), "Testing.\n");
        run_test("config.json");
        return 0;
    }*/
    // Fresh train.
    //
    auto name = vm["name"].as<std::string>();
    // Train
    if (vm.count("config")) {
        auto configs = read_configs(vm);

        // Training.
        int repeat = vm["repeat"].as<int>();
        int mode = vm["mode"].as<int>();

        if (repeat == 1) {
            auto sid = name;
            if (name.empty())
                sid = generate_sid();
            auto [base_config, override_config, full_config] = configs;
            run_main(mode, sid, base_config, override_config, full_config);
        } else {
            fmt::print(fmt::fg(fmt::color::yellow), "Repeat {} times...", repeat);

            int sim = vm["sim"].as<int>();

            std::vector<std::string> sids;
            std::vector<int64_t> seeds;
            std::random_device rd;
            std::uniform_int_distribution<> dist(0, 1e9);
            for (int i = 0; i < repeat; ++i) {
                auto sid = name + "-" + std::to_string(i);
                if (name.empty())
                    sid = generate_sid();
                sids.emplace_back(sid);
                seeds.emplace_back(dist(rd));
            }

            boost::asio::thread_pool pool(sim);
            for (int i = 0; i < repeat; ++i) {
                boost::asio::post(pool, [&, i]() {
                    boost::json::object override_config = std::get<1>(configs);
                    boost::json::object full_config = std::get<0>(configs);
                    boost::json::object seed_config = {{"decision_model", {{"seed", seeds[i]}}}};
                    SciLib::merge_json_in_place(override_config, seed_config);
                    SciLib::merge_json_in_place(full_config, override_config);

                    run_main(mode, sids[i], std::get<0>(configs), override_config, full_config);
                });
            }
            pool.join();
        }
    } else if (vm.count("test")) {
        // Test
        TFATradingSystem system;
        system.test(vm["test"].as<std::string>());
    } else {
        fmt::print(fmt::fg(fmt::color::red),
                   "You need at least set --config for training or --test for test.\n");
        std::cout << desc << std::endl;
    }

    return 0;
}
