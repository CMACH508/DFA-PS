#include "stdafx.h"
#include "MISC.hpp"

namespace SciLib {
double stopwatch_elapsed_seconds(const spdlog::stopwatch &sw) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::milliseconds>(sw.elapsed()).count()) /
           1000;
}
} // namespace SciLib