#boost
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2
tar --bzip2 -xf boost_1_79_0.tar.bz2
cd boost_1_79_0
./bootstrap.sh
mkdir build
# default installation dir: \usr\local
sudo ./b2 --build-dir=build install -j 4
# build test only.
cd ..
rm -rf boost_1_79_0
rm boost_1_79_0.tar.bz2

#Eigen
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake ..
sudo make install
cd ~
rm -r eigen

#fmt
git clone https://github.com/fmtlib/fmt.git
cd fmt
mkdir build
cd build
cmake ..
make -j8
sudo make install
cd ~
rm -r fmt

#spdlog
git clone https://github.com/gabime/spdlog.git
cd spdlog
cmake .
make install
cd ~
rm -r spdlog



