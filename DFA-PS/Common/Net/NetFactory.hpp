#pragma once
#include "NetBase.hpp"

std::shared_ptr<NetBase> create_net(const boost::json::object &config, int window, int in, int out);
