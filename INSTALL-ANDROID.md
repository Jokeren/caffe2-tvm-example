## Build NNVM/TVM on macOS

llvm

```
$ brew install llvm
```

tvm

https://docs.tvm.ai/install/index.html

```
git clone --recursive https://github.com/dmlc/tvm
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:${PYTHONPATH}
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..  # change configurations before make
make -j4
```

### Build Android RPC

maven

```
$ brew install maven
```

gradle

https://docs.gradle.org/current/userguide/build_environment.html#sec:gradle_configuration_properties

```
$ brew install gradle # touch gradle.properties and configure build environment
```

ndk-bundle

```
sdkmanger ndk-bundle
```

.bashrc

```
export PATH=${ANDROID_HOME}/platform-tools:${PATH}
export PATH=${ANDROID_HOME}/ndk-bundle:${PATH}
```

Android Standalone Toolchain

```
$${ANDROID_HOME}/ndk-bundle/build/tools/make-standalone-toolchain.sh --platform=android-28 --use-llvm --arch=arm64 --install-dir=/path/to/tvm/cc
$ export TVM_NDK_CC=/path/to/tvm/cc/bin/xxx-clang++
```

tvm4j

```
cd tvm
make jvmpkg
make jvminstall
```

android_rpc/app/src/main/jni/config.mk

https://github.com/dmlc/tvm/tree/master/apps/android_rpc

```
# reference: https://developer.android.com/ndk/guides/
# arm64-v8a
# armeabi-v7a	
APP_ABI = arm64-v8a

APP_PLATFORM = android-28

# whether enable OpenCL during compile
USE_OPENCL = 1

# the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
ADD_C_INCLUDES = /path/to/opencl/header

# the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
ADD_LDLIBS = /path/to/libOpenCL.so
```

build android_rpc app

```
cd android_rpc
gradle clean build
./dev_tools/gen_keystore.sh
./dev_tools/sign_apk.sh
```
