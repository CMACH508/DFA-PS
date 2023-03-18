#include "stdafx.h"
#include "TFAFactory.hpp"
#include "NeuralNetTFA.hpp"
#include "KalmanTFA.hpp"

std::unique_ptr<TFABase> create_tfa(boost::json::object &config, int dim) {
    auto method = SciLib::json_string_to_string(config["method"].as_string());
    if (method == "TFA") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "NeuralNetTFA");
        fmt::print(" model...\n");
        return std::make_unique<NeuralNetTFA>(config, dim);
    } else if (method == "FA") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "FA");
        fmt::print(" model...\n");
        return std::make_unique<FA>(config, dim);
    } else if (method == "kalman") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Kalman");
        fmt::print(" model...\n");
        return std::make_unique<KalmanTFA>(config, dim);
    } else if (method == "linearTFA") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Linear TFA");
        fmt::print(" model...\n");
        return std::make_unique<LinearTFA>(config, dim);
    } else if (method == "linearTFA-RI-smoothing") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Linear TFA with Reverse Inference Smoothing");
        fmt::print(" model...\n");
        return std::make_unique<LinearTFARSSmoothing>(config, dim);
    } else if (method == "linearTFA-IO-smoothing-1") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline,
                   "Linear TFA with Intertemporal Optimization Smoothing I");
        fmt::print(" model...\n");
        return std::make_unique<LinearTFAIOSmoothingI>(config, dim);
    } else if (method == "kalman-RTS") {
        fmt::print("Creating ");
        fmt::print(fmt::emphasis::underline, "Kalman TFA with RTS");
        fmt::print(" model...\n");
        return std::make_unique<KalmanTFARTSSmoothing>(config, dim);
    } else {
        fmt::print(fmt::fg(fmt::color::red), "Invalid TFA Model: {}\n", method);
        std::abort();
        return nullptr;
    }
}