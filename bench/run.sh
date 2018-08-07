#!/bin/bash

# Specify the proxy host
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=9090

# Specify the standalone Android C++ compiler
if [ $1 = "armv7a" ]; then
  export TVM_NDK_CC=/Users/kerenzhou/Codes/android-toolchain-armv7/bin/arm-linux-androideabi-clang++
else
  export TVM_NDK_CC=/Users/kerenzhou/Codes/android-toolchain-arm64/bin/aarch64-linux-android-clang++
fi

export TVM_NUM_THREADS=1

# python [arch] [remote] [key] [opt:backend] [opt:data_type] [opt:layout] [opt:workloads] [opt:schedule]
python bench.py $1 $2 $3 $4 $5 $6 $7 $8
