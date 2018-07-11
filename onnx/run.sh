#!/bin/bash

# Specify the proxy host
export TVM_ANDROID_RPC_PROXY_HOST=0.0.0.0

# Specify the standalone Android C++ compiler
if [ $2 = "armv7a" ]; then
  export TVM_NDK_CC=/Users/kerenzhou/Codes/android-toolchain-armv7/bin/arm-linux-androideabi-g++
else
  export TVM_NDK_CC=/Users/kerenzhou/Codes/android-toolchain-arm64/bin/aarch64-linux-android-g++
fi

# python [script] [arch] [model] [backend] [opt_level]
python $1 $2 $3 $4 $5
