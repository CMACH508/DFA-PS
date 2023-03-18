#pragma once
#include <boost/asio.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

#define EIGEN_USE_MKL_ALL

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/color.h>
#include <fmt/compile.h>
#include <fmt/chrono.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <torch/torch.h>

#include <boost/circular_buffer.hpp>
#include <boost/json.hpp>
#include <boost/tokenizer.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/positional_options.hpp>

#include "third_party/save_load_eigen.h"

#include <rapidcsv.h>

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <fort.hpp>

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>
#include <numeric>
#include <chrono>