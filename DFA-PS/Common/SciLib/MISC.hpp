#pragma once

#include <spdlog/stopwatch.h>

namespace SciLib {
double stopwatch_elapsed_seconds(const spdlog::stopwatch &sw);
}