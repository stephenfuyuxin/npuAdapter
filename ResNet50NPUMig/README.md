# 参考链接
github开源仓
```shell
url=https://github.com/pytorch/examples.git
commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
```
gitee npu迁移参考
```shell
url=https://gitee.com/ascend/ModelZoo-PyTorch.git
code_path=PyTorch/built-in/cv/classification
```

# miniconda
miniconda 可用可不用，推荐容器方案，容器中不再使用 conda 隔离环境，
```shell
# 安装
bash Miniconda3.sh
source ~/.bashrc
# 创建
conda create --name ResNet50_for_PyTorch python=3.7
# 激活
conda activate ResNet50_for_PyTorch
# 停用
conda deactivate
```

# torch& torch_npu
```shell
# torch
~$ wget https://download.pytorch.org/whl/torch-1.11.0-cp37-cp37m-manylinux2014_aarch64.whl
~$ pip3 install torch-1.11.0-cp37-cp37m-manylinux2014_aarch64.whl
# torch_npu
~$ wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch1.11.0/torch_npu-1.11.0.post14-cp37-cp37m-linux_aarch64.whl
~$ pip3 install torch_npu-1.11.0.post14-cp37-cp37m-linux_aarch64.whl
```

# ModelZoo-PyTorch_ResNet50_for_PyTorch
```shell
~$ git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```
- 非root，如HwHiAiUser：/home/HwHiAiUser/ModelZoo-PyTorch
- root：/root/ModelZoo-PyTorch
```shell
# ResNet50_for_PyTorch 源码包根目录
cd ~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch
```

# 汇总 apt-get install& pip install 依赖
执行 apt-get install 依赖需要在 pip install 依赖之前，

## apt-get install
```shell
apt-get clean && apt-get update && apt-get install -y patch
```

## pip install
```shell
pip install numpy
pip install pyyaml
pip install torchvision==0.12.0  # 对应pytorch 1.11.0
pip install apex-0.1+ascend-cp37-cp37m-linux_aarch64.whl  # 通过 git 编译 apex 得到 .whl 包
pip install decorator
pip install sympy
pip install scipy
pip install attr  # 经过验证不需要
pip install attrs
pip install psutil
pip install tdqm
pip install onnx
pip install aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl  # ais_bench工具，.whl 需提前下载
pip install ais_bench-0.0.2-py3-none-any.whl  # ais_bench工具，.whl需提前下载
```

### apex
```shell
git clone -b master https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.7
cd apex/dist/
pip3 uninstall apex
pip3 install apex-0.1+ascend-cp37-cp37m-linux_aarch64.whl
```

### 安装 ais_bench 推理工具
参考链接：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench

根据README进行安装，
```shell
# 所需依赖
aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl
ais_bench-0.0.2-py3-none-any.whl
# 安装aclruntime
pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
# 安装ais_bench推理程序
pip3 install ./ais_bench-{version}-py3-none-any.whl
```

# 准备数据集
用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压，其中，ImageNet2012的目录结构为，
```shell
├── ImageNet2012
      ├──train
           ├──类别1
                 │──图片1
                 │──图片2
                 │   ...       
           ├──类别2
                 │──图片1
                 │──图片2
                 │   ...   
           ├──...                     
      ├──val  
           ├──类别1
                 │──图片1
                 │──图片2
                 │   ...       
           ├──类别2
              │──图片1
                 │──图片2
                 │   ...                
```

# 单卡训练
```shell
cd ~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch
bash ./test/train_full_1p.sh --data_path=/the/path/of/ImageNet2012
```
在源代码根目录下，
```shell
~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch# ll
checkpoint_npu0model_best.pth.tar
checkpoint_npu0.pth.tar
```
得到 .pth.tar 权重文件，后续推理将会用到该权重文件，
```shell
~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch# bash ./test/train_full_1p.sh --data_path=/the/path/of/ImageNet2012
Final result
================================
Final Train Log: ~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch/test/output/0/train_0.log
Final Performance images/sec : 1511.897
Final Train Accuracy : 64.271
E2E Training Duration sec : 68704
total 200596

~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch ll | grep checkpoint_npu0
checkpoint_npu0model_best.pth.tar
checkpoint_npu0.pth.tar
```
执行单卡训练之前，需要注意以下存在的问题，
- 非 root 用户，需要根据实际情况修改 toolkit 路径；
- 根据实际情况，修改所用推理卡的 npuid；
- train_full_1p.sh 中存在语法错误；

## 非 root 用户，需要根据实际情况修改 toolkit 路径
这里，以 HwHiAiUser 举例，注释掉原有的 CANN_INSTALL_PATH ，用实际环境 CANN_INSTALL_PATH 进行替代， 
```shell
~$ vim ./test/env_npu.sh
if [ -f $CANN_INSTALL_PATH_CONF ]; then
    CANN_INSTALL_PATH=$(cat $CANN_INSTALL_PATH_CONF | grep Install_Path | cut -d "=" -f 2)
else
    #CANN_INSTALL_PATH="/usr/local/Ascend"
    CANN_INSTALL_PATH="/home/HwHiAiUser/Ascend"
fi
```

## 根据实际情况，修改所用推理卡的 npuid
```shell
~$ vim ./test/train_full_1p.sh
# 指定训练所使用的npu device卡id，根据实际情况进行修改，通过 npu-smi info 查看对应的 chip_id 为准
# 默认为0，如果是单卡应该不需要修改
#device_id=0
# 如果8卡全部本地持久化到容器，如，用最后一张卡，则指定为npuid为7
device_id=7
```

## train_full_1p.sh 中存在语法错误
```shell
~$ vim ./test/train_full_1p.sh
#!/bin/bash 下面 set -x
# 性能看护结果汇总
# 获取性能数据，不需要修改
# 吞吐量
ActualFPS=${FPS}
# 单迭代训练时长
#TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`
TrainingTime=`awk 'BEGIN{printf "%.2f\n", "${batch_size}"*1000/"${FPS}"}'`
```
- 第一行代码：由于变量引用方式不正确，`awk` 会尝试将 `'${batch_size}'` 和 `'${FPS}'` 作为字符串进行计算。这会导致计算结果不正确，甚至可能报错，因为字符串不能直接参与数学运算；
- 第二行代码：这行代码正确地将变量的值传递给 `awk`，因此计算结果是正确的。`awk` 会将 `${batch_size}` 和 `${FPS}` 的值代入公式 `batch_size * 1000 / FPS`，并输出结果；

# 模型推理相关
执行路径，
```shell
~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch
```

## 数据预处理
数据预处理将原始数据集转换为模型输入的数据。执行 imagenet_torch_preprocess.py 脚本，完成数据预处理，
```shell
python3 imagenet_torch_preprocess.py resnet ./ImageNet/val ./prep_dataset
```
- ./ImageNet/val 原始数据集val路径；
- ./prep_dataset 预处理完成保存权重；

在 ImageNet2012所在目录下，根据实际情况，执行上面命令
```shell
/the/path/of/ImageNet2012# tar -xvf ILSVRC2012_img_val.tar -C ./val
pip install tdqm
~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch# python3 imagenet_torch_preprocess.py resnet /the/path/of/ImageNet2012/val ./prep_dataset
~/ModelZoo-PyTorch/PyTorch/built-in/cv/classification/ResNet50_for_PyTorch/prep_dataset# ls
ILSVRC2012_val_xxxxxxxx.bin
```

## 模型转换
之前训练完成，得到 .pth.tar 的文件，

### 导出onnx文件
使用 pth2onnx.py 导出 onnx 文件，
```shell
pip install onnx
python3 pth2onnx.py ./checkpoint_npu1.pth.tar
```
实际执行，
```shell
python3 pth2onnx.py ./checkpoint_npu0.pth.tar
```
得到目标文件，
```shell
~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch# ll | grep resnet50
resnet50_official.onnx
```

### 使用 ATC 工具将 ONNX 模型转 OM 模型
执行命令查看芯片名称，
```shell
npu-smi info
```
该设备芯片名为Ascend910B3 （请根据实际芯片填入）
```shell
# npu-smi info
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 86.0        40                0    / 0             |
| 0                         | 0000:00:00.0  | 0           0    / 0          3346 / 65536         |
+===========================+===============+====================================================+
```
ATC命令参考，
```shell
atc --model=resnet50_official.onnx --framework=5 --output=resnet50_bs64 --input_format=NCHW --input_shape="actual_input_1:64,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend910B3 --insert_op_conf=aipp_resnet50.aippconfig
```
参数说明：
- model：为 ONNX 模型文件；
- framework：5代表 ONNX 模型；
- output：输出的 OM 模型；
- input_format：输入数据的格式；
- input_shape：输入数据的 shape ；
- log：日志级别；
- soc_version：处理器型号；
- insert_op_conf: AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用；

运行成功后生成 resnet50_bs64.om 模型文件，
```shell
~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch# atc --model=resnet50_offical.onnx --framework=5 --output=resnet50_bs64 --input_format=NCHW --input_shape="actual_input:1,64,3,224,224" --enable_small_channel=1 --log_level=-1 --soc_version=Ascend910B3 --insert_op_config=resnet50_bs64.json
ATC start working now, please wait for a moment.
...
ATC run success, welcome to the next use.

~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch ll | grep resnet50
resnet50_bs64.om
```

## 推理验证
创建推理结果保存目录，并设置权限
```shell
mkdir -p result
# 如果是 root 用户
chmod -R 600 ./result
# 如果是 非 root 用户，如 HwHiAiUser 用户
chmod 750 ResNet50_for_PyTorch
chmod 750 result
```

### 安装ais_bench工具
参考链接：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench

根据README进行安装，
```shell
# 所需依赖
aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl
ais_bench-0.0.2-py3-none-any.whl
# 安装aclruntime
pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
# 安装ais_bench推理程序
pip3 install ./ais_bench-{version}-py3-none-any.whl
```

## 开始推理
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m ais_bench --model ./resnet50_bs64.om --input ./prep_dataset/ --output ./ --output_dirname result --outfmt TXT
```
参数说明：
- model：模型地址；
- input：预处理完的数据集文件夹；
- output：推理结果保存地址；
- output_dirname: 推理结果保存文件夹；
- outfmt：推理结果保存格式；

```shell
~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch# python3 -m ais bench --model ./resnet50_bs64.onnx --input ./prep_dataset/ --output ./ --output_dtype result --outfmt TXT
[INFO] open device 0 success
[INFO] create new context
[INFO] load model ./resnet50_bs64.onnx success
[INFO] create model description success
[INFO] try get model batchsize:64
[INFO] output path:./result
[INFO] save filespath: filesd size:196608 tensorsize:12582912 filesperbatch:64 runcount:782
[INFO] warm up 1 done
Inference array Processing: 100% |******| 782/782
[INFO] ----------------Performance Summary-----------------
[INFO] NPU_compute_time (ms): min = 4.734375, max = 5.126953125, mean = 4.815102501598465, median = 4.796875, percentile(99%) = 5.002968749999999
[INFO] throughput 1000*batchsize.mean(64)/NPU_compute_time.mean(4.815102501598465): 13291.513519962238
[INFO] -------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```

运行成功后会在result/xxxx_xx_xx-xx-xx-xx（时间戳）下生成推理输出的txt文件，
```shell
~/ModelZoo-Pytorch/Pytorch/built-in/cv/classification/ResNet50_for_Pytorch/results# ls
ILSVRC2012_val_xxxxxxxx_0.txt
```

