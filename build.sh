export CC=/usr/local/bin/gcc
export CXX=/usr/local/bin/g++

#unset MKLROOT
rm -rf build
mkdir build
cd build
cmake ..
#export MKLROOT=/home/wuhui/mkl-dnn-bk/external/mklml_lnx_2018.0.1.20171227
make -j `nproc`
cd ..
