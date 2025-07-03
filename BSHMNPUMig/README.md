# 结论
这里保留 npu 上 tensorflow 框架安装部署迁移过程，用于避坑。对模型本身来说， npu 不支持 tf 框架下权重文件为动态 shape 的模型，模型迁移作罢；

# 开源仓参考
https://modelscope.cn/models/iic/cv_unet_universal-matting/summary

# 安装开源框架 tensorflow 1.15
注，tensorflow 1.15 配套的 Python 版本是：Python3.7.x（3.7.5~3.7.11）。

## gcc 版本降级（gcc-9 g++-9）
python3.7.5 无法使用 ubuntu22.04 默认安装的 gcc 11 的版本进行编译安装，因此需要进行 gcc 版本降级；

gcc 版本过高，原计划安装 gcc 7.3.0 的版本，apt-get install 方式以及源码编译安装方式，均安装失败。改用 gcc 9 的版本；
```dockerfile
RUN apt-get clean && apt-get update && \
    apt-get install --no-install-recommends -y gcc-9 g++-9 && \
    apt-get install --no-install-recommends -y vim-tiny vim sudo git wget zip unzip tar curl gzip && \
    apt-get install --no-install-recommends -y make cmake zlib1g zlib1g-dev openssl libsqlite3-dev lsb-release openssh-server && \
    apt-get install --no-install-recommends -y libssl-dev libffi-dev libbz2-dev liblzma-dev libxslt1-dev pciutils libblas-dev && \
    apt-get install --no-install-recommends -y gfortran libblas3 tree sox iputils-ping ffmpeg xz-utils locales && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 110 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 110 && \
    update-alternatives --config gcc && \
    update-alternatives --config g++ && \
```
- 这里指定安装 gcc 和 g++ 为 9 的版本；
- 使用 update-alternatives 配置默认的 gcc 和 g++ 编译器为版本 9。额外步骤是为了防止安装其他开发工具/库时，会更改对应的gcc/g++版本。见 FAQ “FileNotFoundError: [Errno 2] No such file or directory: 'c++': 'c++'”

## python版本替换
python3.7.5 的环境变量配置，通过 ENV 合入工程化构建文件；

设置 python3.7.5 环境变量。未设置对应的环境变量 dockerfile 工程化构建失败，必须加对应的环境变量
```dockerfile
export LD_LIBRARY_PATH=${HOME}/python3.7.5/lib:$LD_LIBRARY_PATH
export PATH=${HOME}/python3.7.5/bin:$PATH
```
环境变量的设置对于非 root 用户安装的python尤其重要（非 root 用户无法使用软链接的方式指定 python 默认版本）
```dockerfile
ln -sf /usr/local/python3.7.5/bin/python3 /usr/bin/python3 && \
ln -sf /usr/local/python3.7.5/bin/python3 /usr/bin/python && \
ln -sf /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3 && \
ln -sf /usr/local/python3.7.5/bin/pip3 /usr/bin/pip
```

## tf 1.15 安装前准备
对于 x86 架构，可直接跳过安装前准备；

对于 aarch64 架构，由于 TensorFlow 依赖 h5py ，而 h5py 依赖 HDF5 ，需要先编译安装 HDF5 ，否则使用 pip 安装 h5py 会报错，以下步骤以 root 用户操作；
```shell
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz --no-check-certificate
tar -zxvf hdf5-1.10.5.tar.gz
cd hdf5-1.10.5/
./configure --prefix=/usr/local/hdf5
make -j16 && make install
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:$LD_LIBRARY_PATH
pip3 install "Cython<3"
pip3 install wheel
pip3 install numpy
pip3 install h5py==2.8.0
```

## tf 1.15 安装
安装前准备完成之后，安装 tf 1.15 .whl 包，注意 aarch64 架构，
```shell
wget -c https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/turing/resource/toolbox/tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
pip install tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
python
>>> import tensorflow as tf
>>>
>>> exit()
```

# tf adpater 安装
根据对应的 CANN 版本找配套关系的 tfadapter 插件，

https://gitee.com/ascend/tensorflow/tags
```shell
pip install npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl -t /usr/local/Ascend/tfplugin
# 加 tfplugin 的环境变量
export PYTHONPATH=/usr/local/Ascend/tfplugin:$PYTHONPATH
pip list | grep npu
npu-bridge             1.15.0
```

# FAQ
部分问题记录
## ubuntu 22.04 默认 gcc-11 g++-11 装不了 python 3.7.5
## ubuntu 22.04 装不了 gcc-7 g++7
## gcc-9 g++-9 可以编译安装 python 3.7.5
## FileNotFoundError: [Errno 2] No such file or directory: 'c++': 'c++'
执行 pip install tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl 时，安装 grpcio 报错
```shell
Collecting grpcio>=1.8.6 (from tensorflow==1.15.0)
Downloading https://files.pythonhosted.org/packages/31/2e/1e2cd0edeaaeaae0ab9df2615492725d0f2f689d3f9aa3b088356af7a584/grpcio-1.62.3.tar.gz (26.3MB)
```
参考链接：https://blog.csdn.net/weixin_51611994/article/details/140249896

通过 `apt-get install -y build-essential` 解决；

【注意】build-essential 安装会触发安装最新版的 gcc-11 g++-11 ，安装完之后务必执行，
```dockerfile
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 110 && \
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 110 && \
update-alternatives --config gcc && \
update-alternatives --config g++
```

## Failed to build these modules: _ctypes
```shell
Python-3.7.5# ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared
Python-3.7.5# make -j32
```
报错信息，
```shell
Failed to build these modules:
_ctypes
```
参考链接：https://bugs.python.org/issue30090

通过安装 `apt-get install libffi-dev` 解决；

## 安装 tensorflow-1.15.0 出现 h5py 安装报错
```shell
pip install tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl

  error: pkg-config probably not installed: FileNotFoundError
  -----------------------------------------
  ERROR: Failed building wheel for h5py
  Running setup.py clean for h5py
Successfully built numpy
Failed to built h5py
```
需要 `apt-get install pkg-config` 模块；

## pip install h5py==2.8.0 出现 lhdf5 相关文件找不到
安装 h5py 时，
```shell
pip install h5py==2.8.0

/usr/bin/ld: cannot find -lhdf5: No such file or directory
/usr/bin/ld: cannot find -lhdf5_hl: No such file or directory
```
export HDF5 的环境变量貌似没有生效？
```shell
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:$LD_LIBRARY_PATH
# 挂软链接
/usr/lib# ln -s /usr/local/hdf5/lib/libhdf5.so ./libhdf5.so
/usr/lib# ln -s /usr/local/hdf5/lib/libhdf5_hl.so ./libhdf5_hl.so
```
