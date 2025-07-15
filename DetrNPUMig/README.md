# 参考链接
https://github.com/facebookresearch/detr

https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Detr

# docker images
https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

mindie:2.0.RC2-800I-A2-py311-openeuler24.03-lts

# docker run
```shell
# vim run.sh
#!/bin/bash
docker run -it -d --net=host --shm-size=500g --privileged \
--name byd-detr-0-7  \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/sbin:/usr/local/sbin \
-v /home/fuyuxin:/home/fuyuxin \
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.RC2-800I-A2-py311-openeuler24.03-lts \
/bin/bash
```

# 迁移代码仓
## github 代码原仓
```shell
url=https://github.com/facebookresearch/detr
branch=master
commit_id=b9048ebe86561594f1472139ec42327f00aba699
model_name=DETR
```
## gitee 适配昇腾 NPU 实现
```shell
url=https://gitee.com/ascend/ModelZoo-PyTorch
tag=v.0.4.0
code_path=ACL_PyTorch/contrib/cv/detection
```
通过 git 获取对应 commit_id 的代码方法如下，
```shell
git clone {repository_url}        # 克隆仓库的代码
cd {repository_name}              # 切换到模型的代码仓目录
git checkout {branch/tag}         # 切换到对应分支
git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```
具体执行过程记录，
```shell
/home/fuyuxin/
# git clone https://gitee.com/ascend/ModelZoo-PyTorch

Detr]# pwd
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr

/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# tree
.
├── detr_excute_omval.py
├── detr_FPS.py
├── detr_onnx2om.py
├── detr.patch
├── detr_postprocess.py
├── detr_preprocess.py
├── detr_pth2onnx.py
├── LICENSE
├── modelzoo_level.txt
├── public_address_statement.md
├── README.md
├── requirements.txt
└── transformer.py
```
在已下载的源码包根目录下，参考命令，
```shell
git clone https://github.com/facebookresearch/detr.git
cd detr
git checkout b9048ebe86561594f1472139ec42327f00aba699
修改patch文件第298行，将***替换为https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
patch -p1 < ../detr.patch
cd ..
```
具体执行过程记录，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# git clone https://github.com/facebookresearch/detr.git
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# cd detr
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr/detr# git checkout b9048ebe86561594f1472139ec42327f00aba699
# 不用修改 patch 第 298 行，.patch 已经被更新过，直接执行 patch 命令启用补丁即可，
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr/detr# patch -p1 < ../detr.patch
patching file d2/converter.py
patching file datasets/coco.py
patching file datasets/transforms.py
patching file engine.py
patching file hubconf.py
patching file main.py
patching file models/backbone.py
patching file models/detr.py
patching file models/matcher.py
patching file models/position_encoding.py
patching file models/transformer.py
patching file test_all.py
patching file util/misc.py
```

# 安装依赖
gitee 代码参考如下，
```shell
pip3 install -r requirements.txt
pip3 install pycocotools==2.0.3
```
具体执行过程记录，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# cat requirements.txt
onnx==1.10.1
torch==1.5.0
torchvision==0.6.0
numpy==1.21.4
Pillow==8.4.0
decorator==5.1.1
packaging==21.3
Cython==0.29.26
scipy==1.7.3
tqdm==4.64.1
opencv-python==4.4.0.46
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# yum install -y cmake
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# pip install Pillow Cython opencv-python
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# pip3 install pycocotools
```

# 获取原始数据集
本模型支持 coco val 5000张图片的验证集。请用户根据代码仓readme获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到 coco val2017.zip 验证集及 instances_val2017.json 数据标签。

该链接已失效，https://gitee.com/link?target=http%3A%2F%2Fimages.cocodataset.org%2Fannotations%2Fannotations_trainval2017.zip

作为替代，https://blog.csdn.net/qq_41847324/article/details/86224628

训练集的标签：http://images.cocodataset.org/annotations/annotations_trainval2017.zip

验证集：http://images.cocodataset.org/zips/val2017.zip

具体执行过程记录，及目录结构
```shell
unzip val2017.zip
unzip annotations_trainval2017.zip
mkdir coco_data
mv annotations coco_data
mv val2017 coco_data
# coco_data目录结构需满足:
coco_data
    ├── annotations
    └── val2017
```

# 数据预处理
数据预处理将原始数据集转换为模型输入的数据，参考命令，
```shell
python3.7 detr_preprocess.py --datasets=coco_data/val2017 --img_file=img_file --mask_file=mask_file
```
具体执行过程记录，
```shell
# openeuler24.03 安装 mesa-libGL
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# yum install mesa-libGL -y
# ubuntu22.04 安装 mesa-libGL 相关
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglu1-mesa-dev mesa-common-dev libglx-mesa0 libgbm-dev
# numpy 降级
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# pip install numpy==1.24
# 数据预处理
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_preprocess.py --datasets=/home/fuyuxin/coco_data/val2017 --img_file=img_file --mask_file=mask_file
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ll ./img_file/
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ll ./mask_file/
```

# 模型转换
使用 PyTorch 将模型权重文件 .pth 转换为 .onnx 文件，再使用 ATC 工具将 .onnx 文件转为离线推理模型文件 .om 文件。

## 获取 .pth 权重文件
从源码包中获取权重文件：detr.pth，将权重文件放入model文件夹

detr.pth权重文件获取

链接: https://pan.baidu.com/s/1iz18BwU6E141hEmwigpe_w

密码: du65

```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# mkdir model
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# mv /the/path/of/detr.pth ./model
```

## 导出 onnx 文件（.pth 转 .onnx）
使用 detr.pth 导出 onnx 文件，运行 detr_pth2onnx.py 脚本，参考如下，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_pth2onnx.py --batch_size=1
```
获得 detr_bs1.onnx 文件（本模型只支持bs1与bs4），
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ll ./model/
detr_bs1.onnx
```

## 使用ATC工具将 ONNX 模型转 OM 模型（.onnx 转 .om）
配置环境变量，
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
查看芯片型号，执行命令查看芯片名称（${chip_name}）--- "910B4" --- "Ascend910B4"
```shell
# npu-smi info
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 86.0        40                0    / 0             |
| 0                         | 0000:00:00.0  | 0           0    / 0          2828 / 32768         |
+===========================+===============+====================================================+
```
执行ATC命令，参考如下，
```shell
mkdir auto_om
python3.7 detr_onnx2om.py --batch_size=1 --auto_tune=False --soc_version=${chip_name}
```
参数说明：
- batch_size：批大小，即1次迭代所使用的样本量；
- auto_tune：模型优化参数；
运行成功后生成 auto_om 模型文件夹，

具体执行过程记录，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# mkdir auto_om
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_onnx2om.py --batch_size=1 --auto_tune=False --soc_version=Ascend910B4
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ll ./auto_om/
```

# 开始推理验证
## 安装ais_bench推理工具
请访问ais_bench推理工具代码仓，根据readme文档进行工具安装，

https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench

所用镜像 py 相关版本确认，
```shell
# python --version
python 3.11.6
# pip --version
pip 23.3.1(python 3.11)
```
OS架构确认，
```shell
# uname -m && cat /etc/*release && uname -r
aarch64
openeuler 24.03 (LTS)
5.15.0
```
下载安装 ais_bench 所需的 .whl 包，安装命令参考，
```shell
# 安装aclruntime
pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
# 安装ais_bench推理程序
pip3 install ./ais_bench-{version}-py3-none-any.whl
```
具体执行过程记录，
```shell
# pip install /the/path/of/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl
# pip install /the/path/of/ais_bench-0.0.2-py3-none-any.whl
```
## 执行推理
命令参考，
```shell
mkdir result
python3.7 detr_excute_omval.py --ais_path=ais_infer.py --img_path=img_file --mask_path=mask_file --out_put=out_put --result=result --batch_size=1 > bs1_time.log
```
参数说明：
- ais_path：ais_bench推理工具推理文件路径；
- img_path：前处理的图片文件路径；
- mask_path：前处理的mask文件路径；
- out_put：ais_infer推理数据输出路径；
- result：推理数据最终汇总路径；
- batch_size：batch大小，可选1或4；

### 需要修改 detr_excute_omval.py 脚本
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# cp detr_excute_omval.py detr_excute_omval.py.org
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# vim detr_excute_omval.py
```
修改如下，
```python
import os
import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#parser.add_argument('--ais_path', default='ais_infer.py')
parser.add_argument('--ais_path', default='ais_bench')
parser.add_argument('--img_path', default='img_file')
parser.add_argument('--mask_path', default='mask_file')
parser.add_argument('--out_put', default='out_put')
parser.add_argument('--result', default='result')
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()

if not os.path.exists(args.out_put):
    os.mkdir(args.out_put)
if not os.path.exists(args.result):
    os.mkdir(args.result)

shape_9 = [[768, 1280, 24, 40], [768, 768, 24, 24], [768, 1024, 24, 32], [1024, 768, 32, 24], [1280, 768, 40, 24],
           [768, 1344, 24, 42], [1344, 768, 42, 24], [1344, 512, 32, 42], [512, 1344, 16, 42]]
print(args)
if args.batch_size == 1:
    for i in shape_9:
        #command = 'python3.7 {} --model "auto_om/detr_bs4_{}_{}.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}"  --outfmt BIN'.format(
        command = 'python3 -m  {} --model "auto_om/detr_bs1_{}_{}.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}"  --outfmt BIN'.format(
            args.ais_path, i[0], i[1], args.img_path, i[0], i[1], args.mask_path, i[0], i[1], args.out_put)
        print(command)
        os.system(command)
    mv_command = 'mv {}/*/* {}'.format(args.out_put, args.result)
    os.system(mv_command)
elif args.batch_size == 4:
    for i in shape_9:
        #command = 'python3.7 {} --model "auto_om/detr_bs4_{}_{}.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}"  --outfmt BIN'.format(
        command = 'python3 -m  {} --model "auto_om/detr_bs4_{}_{}.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}"  --outfmt BIN'.format(
            args.ais_path, i[0], i[1], args.img_path, i[0], i[1], args.mask_path, i[0], i[1], args.out_put)
        print(command)
        os.system(command)
    mv_command = 'mv {}/*/* {}'.format(args.out_put, args.result)
    os.system(mv_command)
```
原因：脚本 detr_excute_omval.py 调用推理工具 ais_bench 推理om模型，而 ais_bench 存在更新，需对参数进行更新，
- 改 "--ais_path=ais_infer.py" 为 "--ais_path=ais_bench" ；
- 改 python 执行为 "python3 -m ais_bench *.om" 通过 python 调用 ais_bench 形式；

### 修改命令参数（--ais_path=ais_bench）执行推理
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# mkdir result
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# mkdir out_put
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# chmod 750 out_put
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_excute_omval.py --ais_path=ais_bench --img_path=img_file --mask_path=mask_file --out_put=out_put --result=result --batch_size=1 > bs1_time.log
```

### 推理结果
执行 detr_excute_omval.py 脚本后，推理结果路径最终在 ./result 目录下，并生成推理 info 日志文件；
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ll . | grep log
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# ls ./result/
```

## 精度验证
调用 detr_postprocess.py 脚本，命令参考，
```shell
export PYTHONPATH=usr/local/detr
python3.7 detr_postprocess.py --coco_path=coco_data --result=result
```
- usr/local/detr：源码库路径；
- coco_path：数据集路径；
- result：om推理出的数据存放路径；

具体执行过程记录，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# export PYTHONPATH=/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr/detr:$PYTHONPATH
```

### 精度结果及展示结果对比
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_postprocess.py --coco_path=/the/path/of/coco_data --result=./result
# 精度结果如下，
Accumulating evaluation results...
DONE (t=9.27s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.806
```

### kimi 对精度结果的解析
要计算这个模型的精度需考虑不同的指标。这里给出的指标包括平均精度（AP）和平均召回率（AR），它们基于不同交并比（IoU）阈值和不同面积范围。通常，关注平均精度（AP）@[ IoU=0.50:0.95 | area= all | maxDets=100 ]，这个指标考虑了从0.50到0.95的 IoU 范围，所有面积的物体，以及最多100个检测结果。

根据给出的指标，平均精度（AP）@[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.417。这个值表示模型在所有 IoU 阈值从0.50到0.95的平均精度，是一个综合的性能指标。

因此，这个模型的精度可以认为是0.417。这个值越高，表示模型的检测性能越好。在实际应用中，我们通常希望这个值尽可能高，但也要考虑其他指标和实际需求。

所以，这个模型的精度是 \boxed{0.417}。

## 性能验证
解析推理日志文件，计算性能数据，命令参考，
```shell
python3.7 detr_FPS.py --log_path=bs1_time.log --batch_size=1
```
- log_path：推理info日志文件路径；

bz=1 性能结果，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_FPS.py --log_path=./bs1_time.log --batch_size=1
183.43551027671745
```

bz=4 性能结果，
```shell
/home/fuyuxin/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Detr# python detr_FPS.py --log_path=./bs4_time.log --batch_size=4
233.93885829424002
```

## 执行 detr_excute_omval.py 推理 npu-smi info 中 AICore /显存占用情况
```shell
# npu-smi info
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 139.1       45                0    / 0             |
| 0                         | 0000:C1:00.0  | 16          0    / 0          3397 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 84.7        37                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          2830 / 32768         |
+===========================+===============+====================================================+
| 2     910B4               | OK            | 84.1        38                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          2829 / 32768         |
+===========================+===============+====================================================+
| 3     910B4               | OK            | 87.0        39                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          2829 / 32768         |
+===========================+===============+====================================================+
| 4     910B4               | OK            | 85.7        45                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          2829 / 32768         |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 86.9        45                0    / 0             |
| 0                         | 0000:02:00.0  | 0           0    / 0          2829 / 32768         |
+===========================+===============+====================================================+
| 6     910B4               | OK            | 93.7        43                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2829 / 32768         |
+===========================+===============+====================================================+
| 7     910B4               | OK            | 83.9        42                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          2830 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 3594524       | python3                  | 617                     |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 2                                                            |
+===========================+===============+====================================================+
```









