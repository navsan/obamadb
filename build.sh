#!/usr/bin/env bash
# This script should be used for fresh builds.
# otherwise, cd build && make

git submodule init
git submodule update
pushd .
cd third_party/glog
git apply ../glog_cmake.patch
popd
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=1  ..
make -j4
