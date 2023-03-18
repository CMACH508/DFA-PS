#

## Why C++

The project is mainly written with C++. Why choose it? First the method is heavily rely on matrix calculation, and performance is critical. Second I love C++ more than python.

## Project Structure

1. `Reproduce`: baseline experiments.
1. `Data`: data folder. See `ReadMe.md` in it for more information.
2. `DFA-PS`: code folder. See `ReadMe.md` in it for more information.
3. `libtorch`: `libtorch` C++ version package. Download from [Libtorch](https://pytorch.org/get-started/locally/). For windows, select `Stable - Windows - LibTorch - C++/Java - Release Version`. For Linux, select `Stable - Linux - LibTorch - C++/Java - cxx11 ABI`. Extract contents to directory `libtorch` so the directory structure will be like:

```txt
libtorch
--bin
--cmake
--include
...
```

One Linux, run

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
```

## Use with your own data

1. Prepare data. See `Data/ReadMe.md`.
1. Copy a config file, slightly modify it and run. See `DFA-PS/ReadMe.md`.