# 参考链接
github开源仓
```shell
https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
branch=master
commit_id=90c83db3f06d364c4abd115825868641b95f6181
```
gitee npu迁移参考
```shell
https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch
```

# miniconda
miniconda 可用可不用，推荐容器方案，容器中不再使用 conda 隔离环境，
```shell
# 安装
bash Miniconda3.sh
source ~/.bashrc
# 创建
conda create --name CRNN_Builtln_for_PyTorch python=3.7
# 激活
conda activate CRNN_Builtln_for_PyTorch
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

# ModelZoo-PyTorch_CRNN_BuiltIn_for_Pytorch
```shell
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
```
- 非root，如HwHiAiUser：/home/HwHiAiUser/ModelZoo-PyTorch
- root: /root/ModelZoo-PyTorch
```shell
# CRNN_BuiltIn_for_Pytorch 源码包根目录
cd ~/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch
```

# 根据 requirements.txt 安装依赖
```shell
pip install onnx==1.12.0
pip install numpy==1.21.6
pip install lmdb==1.3.0
pip install Pillow==9.2.0
pip install matplotlib
```

# 数据集准备
数据集链接：https://aistudio.baidu.com/datasetdetail/138872

`testing_lmdb.zip(148.95M) 下载`

下载解压之后将其中的 `IIIT5K_3000` 目录移动到 `~/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch` 重命名 `IIIT5K_lmdb`

# 数据预处理
绝对路径：~/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/CRNN_BuiltIn_for_Pytorch
```shell
CRNN_BuiltIn_for_Pytorch# python3 parse_testdata.py ./IIIT5K_lmdb input_bin
```
如果报错 `torchvision/io/image.py:13: UserWarning: Failed to load image Python extension:` ，安装 torchvision==0.12.0 ，
```shell
CRNN_BuiltIn_for_Pytorch# pip install torchvision==0.12.0
```
执行后，二进制文件生成在 ./input_bin/ 路径下，标签数据 label.txt 生成在当前目录下，
```shell
# test_*.bin
RNN_BuiltIn_for_Pytorch# cd input_bin/
RNN_BuiltIn_for_Pytorch/input_bin# ls
test_*.bin

# label.txt
RNN_BuiltIn_for_Pytorch# cat label.txt
```

# 获取权重文件
模型官方实现没有提供对应的权重，所以此处使用NPU自行训练的权重结果作为原始输入，对应下载权重，

https://gitee.com/link?target=https%3A%2F%2Fascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com%2Fc-version%2FCRNN_for_PyTorch%2Fzh%2F1.3%2Fm%2FCRNN_for_PyTorch_1.3_model.zip

下载后 unzip 解压获取里面的 checkpoint.pth 权重文件，将 checkpoint.pth 文件移动到源代码根目录，
```shell
CRNN_BuiltIn_for_Pytorch# ls | grep *.pth
checkpoint.pth
```

# 导出 ONNX 文件
使用 pth2onnx.py 导出 onnx 文件。运行 pth2onnx.py 脚本，
```shell
CRNN_BuiltIn_for_Pytorch# python pth2onnx.py ./checkpoint.pth ./crnn_npu_dy.onnx
```
获得 crnn_npu_dy.onnx 文件。

# 使用 ATC 工具将 ONNX 模型转 OM 模型
先设置下环境变量，
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
```
npu-smi info 获取芯片名称（${chip_name}）
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
Ascend${chip_name} 请根据实际查询结果填写，
```shell
# 命令参考
atc --model=crnn_npu_dy.onnx --framework=5 --output=crnn_final_bs16 --input_format=NCHW --input_shape="actual_input_1:16,1,32,100" --log=error --soc_version=Ascend${chip_name}
# 实际执行
CRNN_BuiltIn_for_Pytorch# atc --model=crnn_npu_dy.onnx --framework=5 --output=crnn_final_bs16 --input_format=NCHW --input_shape="actual_input_1:16,1,32,100" --log=error --soc_version=Ascend910B3
```
执行过程中可能存在失败的情况，这些依赖是必需的，
```shell
pip install decorator
pip install sympy
pip install scipy
pip install attrs
pip install psutil
```
执行 atc 转换成功后生成 crnn_final_bs16.om 模型文件。

# 安装 ais_bench 推理工具
参考链接：https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench，根据README进行安装，
```shell
# 所需依赖
aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl
ais_bench-0.0.2-py3-none-any.whl
# 安装aclruntime
pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
# 安装ais_bench推理程序
pip3 install ./ais_bench-{version}-py3-none-any.whl
```

# 执行推理
先设置下环境变量，
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
```
命令参考，
```shell
python3 -m ais_bench --model ./crnn_final_bs16.om --input ./input_bin --output ./ --output_dirname result --device 0 --batchsize 16 --output_batchsize_axis 1
```
实际执行，
```shell
CRNN_BuiltIn_for_Pytorch# cd ..
cv# chmod 750 CRNN_BuiltIn_for_Pytorch
CRNN_BuiltIn_for_Pytorch# python3 -m ais_bench --model ./crnn_final_bs16.om --input ./input_bin --output ./ --output_dirname result --device 0 --batchsize 16 --output_batchsize_axis 1
```
运行成功后会在 CRNN_BuiltIn_for_Pytorch/result 下生成推理输出的 bin 文件。

# 精度验证
运行脚本 postpossess_CRNN_pytorch.py 进行精度测试，精度会打屏显示，

命令参考，
```shell
python3 postpossess_CRNN_pytorch.py ./result ./label.txt
```
实际执行，
```shell
CRNN_BuiltIn_for_Pytorch# python3 postpossess_CRNN_pytorch.py ./result ./label.txt
```
结果展示，
```shell
... ...
... ...
****************************************
rightNum: 2247
totalNum: 3000
accuracy_rate 74.90
****************************************
```

# 性能验证
可使用 ais_bench 推理工具的纯推理模式验证不同 batch_size 的 om 模型的性能，命令参考，
```shell
python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
```
实际执行，
```shell
CRNN_BuiltIn_for_Pytorch# python -m ais_bench --model=./crnn_final_bs16.om --loop=20 --batchsize=16
```
结果展示，
```shell
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model ./crnn_final_bs16.om success
[INFO] create model description success
[INFO] warm up 1 done
loop inference exec: (20/20)  0%|                                 | 0/1 [00:00<?, ?it/s]
Inference array Processing: 100%|                         | 1/1 [00:00<00:00, 21.14it/s]
[INFO] ------Performance Summary------
[INFO] NPU compute_time (ms): min = 1.675080114449917797, max = 1.9220008850097656, mean = 1.801199722229039, median = 1.7849998474121094, percentile(99%) = 1.9176311111450195
[INFO] throughput 1000*batchsize.mean(16)/NPU_compute_time.mean(1.801199722229039): 8882.968280528967
[INFO] ------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
(CRNN Built for PyTorch) root@ak-g5688v2-1003:~/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/CRNN_Built_for_Pytorch#
```
