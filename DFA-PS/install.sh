mkdir tmp

cd tmp

# boost
wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz
tar --bzip2 -xf boost_1_81_0.tar.bz2
cd boost_1_81_0
./bootstrap.sh --with-python=python3 --prefix=/usr/local
mkdir build
sudo ./b2 --build-dir=build install -j 8
cd ..

# spdlog
git clone https://github.com/gabime/spdlog.git
cd spdlog
cmake .
make install
cd ..

# Eigen
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake ..
sudo make install
cd ../..

#fmt
git clone https://github.com/fmtlib/fmt.git
cd fmt
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ..

rm -rf tmp
