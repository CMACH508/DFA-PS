#pragma once
#include"TFABase.hpp"
std::unique_ptr<TFABase> create_tfa(boost::json::object &config, int dim);