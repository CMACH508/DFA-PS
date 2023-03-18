#pragma once

#include "../JsonHelper.hpp"
#include "../Util.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

// 1. Traing happen at two stage. The first is use all training data to update model. The second is
// on test stage, update the model at daily frequency.
struct StrategyBase {};
