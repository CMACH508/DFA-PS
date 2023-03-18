## Windows

### Prerequisites

#### Visual Studio 2022

Download and install `preview` version from [Visual Studio](https://visualstudio.microsoft.com/vs/). 

#### VCPKG

It's a C++ package manager. Install vcpkg:

```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install

```
It's better to put it in a short path, e.g. `C:`.

Then install packages:

```
./vcpkg install boost:x64-windows eigen3:x64-windows spdlog:x64-windows libfort:x64-windows
```

#### Intel OneAPI MKL

Download from [Intel OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). When installing, at least select `MKL`.

### Build

1. Open `.sln` with VS 2022 and set solution configurations to be `Release`, then push `Ctrl+Shift+B` to build the project. On directory `x64/Release` you will see `.exe` file. 
1. Usually the `.exe` can't be run directly because it requires some `.dll` from libtorch so you have to add the directory `libtorch_win/lib` to `PATH`. If you encounter the problem of can't find mkl, then you have to manually add mkl path, the default path is `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64`. To add a path in script, try something like:

```
set PATH=C:/windows;C:/windows/system32;C:/windows/syswow64;%~dp0../libtorch_win/lib
```
on Windows. `%~dp0` means the current directory.


### Run Experiments

On directory `Experiments`, click `R.bat` to run experiments.

## Linux

I mainly use Windows so there maybe some unfixed problems when running on Linux.

### Prerequisites

#### Install Packages

##### Debian and Ubuntu

1. On debian, there is an official source for `intel-mkl`. Run

```
sudo vi /etc/apt/sources.list
```
Then add following lines:
```
deb http://deb.debian.org/debian sid main contrib non-free
deb-src http://deb.debian.org/debian sid main contrib non-free
```


2. Run following command to install system packages:
```
sudo apt update
sudo apt upgrade

sudo apt install -y build-essential cmake git wget intel-mkl python3
```

3. Run `install.sh` to install necessary C++ packages. If you encounter network problems, you may have to set proxy:

```bash
export http_proxy=http://127.0.0.1:10809
export https_proxy=http://127.0.0.1:10809
```

On WSL2, you can use:

```bash
export hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
export https_proxy="http://${hostip}:10809"
export http_proxy="http://${hostip}:10809"
```

##### Arch

Run following commands is enough:

```
pacman -Syu
pacman -Sy archlinux-keyring
pacman -Sy base-devel cmake wget git intel-mkl eigen boost fmt spdlog
```

The `eigen` from `pacman` may not be latest, if you have problem related to `eigen` then install it from git, see `install.sh`.

##### Other

On other Linux system, you can also install `intel-mkl` from Intel official website [Get Intel® oneAPI Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).

### Build

Requirements:

1. Your compiler should support `C++17` standard such as `gcc-11` and newer. According to your compiler version, you have to modify `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` in `CMakeLists.txt` (currently it's `gcc-11, g++-11`). 

2. If you install MKL from Intel website, then you have to set some paths:

```
echo 'export MKLROOT=~/intel/oneapi/mkl/latest' >> ~/.bashrc

source ~/.bashrc
```

If you use MKL from system source, then you also need to set `MKLROOT`. For example, on Arch, it's:

```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

After all requirements are satisfied, to build projects, run:

```bash
cmake .
make -j8
```

The output will be in directory `x64/Release`.

## Command Line Options for DFA-PS

Run `DFA-PS` without any parameters to print help. Remember to set the path for `MKL` and `libtorch`. On Windows:

```
set "PATH=C:/windows;C:/windows/system32;C:/windows/syswow64;%~dp0../../../../../libtorch/lib;C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64"
```

On Linux, it seems there is no need to set path.

### Train

Fresh run:
```
DFA-PS -C config.json --override new_config.json --mode=0 --name=test
```
The first `config.json` is a base file whose content will be updated by the second `config.json`. `mode=0` means run factor model and decision model. `--name=test` will set the backup directory to `backup/test`. If you don't set `--name` then a UUID (a random number) will be used as name.

Due to training of NN-TFA and policy network are separated, a two pass run is preferred. First run factor analysis model to calculate hidden factors:

```bash
DFA-PS -C config.json --override new_config.json --mode=1 --name=factor
```
Then in `new_config.json` set 

```json
"use_precomputed_hidden_factor": true,
"precomputed_hidden_factor_path": "backup/factor",
```

and run decision model using hidden factors calculated before:

```bash
DFA-PS -C config.json --override new_config.json --mode=0 --name=factor
```

In fact, `mode=1` will forcibly set `use_precomputed_hidden_factor` to `false`, so you can set config before the first pass. 


### Test

Load saved model and run test:

```
DFA-PS --test=backup/test
```



## Code Structure

Codes are mainly located in directory `Common`. Directory `NeuralnetTFA` only contains some code to invoke model and deal with command line options. Major parts of `Common` and their responsibilities are listed as follows: 
1. `main.cpp`: deal with command line options.
	1. `run_once()`: given a configure object, run `TFATradingSystem`. It's the entry point.
1. `Util.hpp/.cpp`: data structure for dataset, reading data. Important components:
	1. `DataSet`: represents dataset on disk, use	`Eigen::Matrix` to store price data, `torch::Tensor` to store feature data.
	2. `read_dataset`: read data from disk into `DataSet`.
	3. `DecisionModelDataSet`: represents a input sample to decision model.
	4. `DataPool`: represents train and test data including asset data, market indicators and factor data calculated from factor analysis model. It provides `sample_train_dataset()` to generate training samples (`DecisionModelDataSet`).
1. `Net`: basic networks:
	1. `ChannelAttention.hpp/.cpp`: channel attention.
	2. `SpatialAttention.hpp/.cpp`: spatial attention.
	3. `ChannelSpatialAttention.hpp/.cpp`: combine two attentions.
1. `DecisionModel`: generate portfolio weights. Inheritance structure: `DecisionModelBase <- PolicyNetworkBase <- TwoLevelPolicyNetwork`. `TwoLevelPolicyNetwork` is the model in our paper.
	1. `DecisionModelBase.hpp/.cpp -> DecisionModelBase::train()/test()`: train and test the policy network.
	2. `PolicyNetworkBase.hpp/.cpp -> PolicyNetworkBase::train_epoch()`: train the neural network a single epoch.
	2. `TwoLevelPolicyNetwork.hpp/.cpp -> TwoLevelPolicyNetwork::cal_score()`: generate score vector.
	3. `PolicyNetworkBase.hpp/.cpp -> AllowingShortBetaModelWrapper`: given a full score vector of all assets, generate long and short weights by select a subset.
2. `FactorAnalysis`: factor analysis model, e.g. NN-TFA. Inheritance paradigm is used: `TFABase <- LinearTFA <- NeuralNetTFA`. 
	1. `TFABase::train()`: virtual and concrete method for training model.
	2. `TFABase::update_parms()`: update parameters for observe equation and covariance matrix for state equation.
	3. `TFABase::generate_data_pool`: generate `DataPool` from `DataSet`.
3. `TFATradingSystem`: high-level modules that use `FactorAnalysis` model and `DecisionModel`:
	1. `TFATradingSystem::train_fa()/train_decision_model()`: train factor analysis model and decision model.
	2. `TFATradingSystem::test()`: test on saved model.
3. `SciLib`: some convenient functions, for example, `STDHelper` for dealing with file paths.
	1. `Finance.hpp/.cpp -> FullPortfolioData`: represents a portfolio including volume and prices, e.t.c.
	2. `Finance.hpp/.cpp -> PortfolioPerformance`: a struct to store portfolio performance such as `ARR, MDD`.
	3. `Finance.hpp/.cpp -> calculate_portfolio_performance()`: given a `FullPortfolioData`, calculate `PortfolioPerformance`.
	2. `Finance.hpp/.cpp -> ReturnCalculator`: given portfolio weights from decision model, calculate `FullPortfolioData`. 

## Development Guide

### Design a New Network

You can derive from `PolicyNetworkBase` and at least override `cal_score()`. If you want to save and load model, then override `save(), load()`.  Override `set_train(), set_eval()` is useful if your network uses `DropOut` layer. Then modify `DecisonModel -> DecisionModelFactory::create_decision_model()` by adding a line to create your model.


### New Method Other Than Neural Network

If you want to design a new method that doesn't use neural network, then you have to derive from `DecisonModelBase` and override `train_epoch(), cal_culate_portfolio_weight()`. Then modify `create_decision_model()`.

