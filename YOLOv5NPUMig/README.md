# 参考链接
https://github.com/ultralytics/yolov5

https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch

# docker images
https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

mindie:2.0.RC2-800I-A2-py311-openeuler24.03-lts

# docker run
```shell
# vim run.sh
#!/bin/bash
docker run -it -d --net=host --shm-size=500g --privileged \
--name byd-yolov5-0-7  \
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
YOLOv5 每个版本主要有4个开源模型，分别为 YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，四个模型的网络结构基本一致，只是其中的模块数量与卷积核个数不一致。YOLOv5s 模型最小，其它的模型都在此基础上对网络进行加深与加宽。

从描述来看，yolov5本身还有版本迭代更新，分为以下几个版本，
```shell
url=https://github.com/ultralytics/yolov5
tag=v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0
model_name=yolov5
```
yolov5 自身各版本之间模型结构的差异，比如 Conv 模块各版本差异示例如下，
| yolov5 版本 | Conv 模块激活函数 |
| ---------- | --------------- |
| 2.0	       | LeakyRelu      |
| 3.0	       | LeakyRelu      |
| 3.1	       | hswish         |
| 4.0	       | SiLU           |
| 5.0	       | SiLU           |
| 6.0	       | SiLU           |
| 6.1	       | SiLU           |
| 6.2	       | SiLU           |
| 7.0	       | SiLU           |

https://gitee.com/ascend/modelzoo-GPL/tree/master/built-in/ACL_Pytorch/Yolov5_for_Pytorch

gitee 上命令执行例，以 yolov5 v6.1 为准；

# 获取 github 源码
参考如下，
```shell
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0  # 切换到所用版本
```
以 yolov5 v6.1 版本作为执行例，
```shell
# git clone https://github.com/ultralytics/yolov5.git
# cd yolov5/
# git checkout v6.1
```

# 获取 OM 推理代码
由于 v7.0 版本的开源 yolov5 模型，head 层发生了变动，所以后处理也需要做相应修改；
```shell
# 返回 yolov5_for_pytorch 目录
cd ..
# tag v7.0 需要执行以下操作
git apply 7.0.patch
```
如果用 yolov5 v7.0 需要额外执行 git apply ，执行例以 v6.1 为准，则无须执行；

将推理部署代码放到 yolov5 源码相应目录下，
```shell
# git clone https://gitee.com/ascend/modelzoo-GPL.git
```
按照 tree 放置文件（这个文件系统结构在后面实测过程中有更新），
```shell
Yolov5_for_Pytorch
 └── common                        放到yolov5下
   ├── util                        模型/数据接口
   ├── quantify                    量化接口
   ├── atc_cfg                     atc转模型配置文件
   └── patch                       v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0 兼容性修改
 ├── model.yaml                    放到yolov5下 
 ├── pth2onnx.sh                   放到yolov5下
 ├── onnx2om.sh                    放到yolov5下
 ├── aipp.cfg                      放到yolov5下
 ├── om_val.py                     放到yolov5下
 ├── yolov5_preprocess_aipp.py     放到yolov5下
 ├── yolov5_preprocess.py          放到yolov5下
 ├── yolov5_postprocess.py         放到yolov5下
 └── requirements.txt              放到yolov5下
```
实际执行如下（这里在 .pth 转 .onnx 时有文件系统结构更新），
```shell
# cd /home/fuyuxin/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch
/home/fuyuxin/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch# cp model.yaml pth2onnx.sh onnx2om.sh aipp.cfg om_val.py yolov5_postprocess.py yolov5_preprocess_aipp.py yolov5_preprocess.py requirements.txt /home/fuyuxin/yolov5/
/home/fuyuxin# cd yolov5
/home/fuyuxin/yolov5/# mkdir Yolov5_for_Pytorch
/home/fuyuxin/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch# cp -r common/ /home/fuyuxin/yolov5/Yolov5_for_Pytorch
/home/fuyuxin/yolov5# tree
.
├── aipp.cfg
├── model.yaml
├── om_val.py
├── onnx2om.sh
├── pth2onnx.sh
├── requirements.txt
├── Yolov5_for_Pytorch
│   └── common
│       ├── atc_cfg
│       ├── __init__.py
│       ├── patch
│       ├── quantify
│       └── util
├── yolov5_postprocess.py
├── yolov5_preprocess_aipp.py
└── yolov5_preprocess.py
```

# 安装依赖

## 安装 msit surgeon 组件
请访问 msit 推理工具代码仓，根据 readme 文档进行工具安装 msit surgeon 组件，
```shell
Ascend/msit：
https://gitee.com/ascend/msit/tree/master/msit/#/ascend/msit/blob/master/msit/./docs/install/README.md
```
msit 工具安装：https://gitee.com/ascend/msit/blob/master/msit/docs/install/README.md

安装方式包括：源代码安装和pip源安装，用户可以按需选取，
- 源代码安装：使用源码安装，保证是最新的 msit 功能；
- pip源安装：pip 安装 msit 包，一般一个季度发包一次；

这里，用源码安装方式，参考如下，
```shell
git clone https://gitee.com/ascend/msit.git
# 1. git pull origin 更新最新代码 
cd msit/msit

# 2. 安装 msit 包
pip install .

# 3. 通过以下命令，查看组件名，根据业务需求安装相应的组件
# 参考各组件功能介绍:(https://gitee.com/ascend/msit/tree/master/msit#%E5%90%84%E7%BB%84%E4%BB%B6%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D)
msit install -h

# 4. 如果需要安装llm：
msit install llm

# 5. 安装之后可以使用 msit check 命令检查安装是否成功：
msit check llm 
```
实际执行如下，
```shell
/home/fuyuxin# git clone https://gitee.com/ascend/msit.git
/home/fuyuxin# cd msit/msit
/home/fuyuxin/msit/msit# pip install .
/home/fuyuxin/msit/msit# msit install benchmark
/home/fuyuxin/msit/msit# msit install surgeon
/home/fuyuxin/msit/msit# msit check surgeon
XXXX-XX-XX XX:XX:XX,XXX - XXX - msit_logger - INFO - msit-surgeon
XXXX-XX-XX XX:XX:XX,XXX - XXX - msit debug logger - INFO -     OK
```

## 安装 requirements.txt 依赖
```shell
pip3 install -r requirements.txt
```
实际执行如下，
```shell
/home/fuyuxin/yolov5# pip install pycocotools opencv-python Pillow seaborn pyyaml opencv-python-headless
```

# 准备数据集
模型使用 coco2017 val数据集 进行推理精度/性能评估，在 **yolov5 源码根目录下新建 coco 文件夹，数据集放到 coco 里**，文件结构如下，
```shell
coco
├── val2017
   ├── 00000000139.jpg
   ├── 00000000285.jpg
   ……
   └── 00000581781.jpg
├── instances_val2017.json
└── val2017.txt
```
其中，文件 val2017.txt 中保存 .jpg 相对路径，请自行生成该 .txt 文件，文件内容示例如下，
```shell
./val2017/00000000139.jpg
./val2017/00000000285.jpg
……
./val2017/00000581781.jpg
```

# 模型推理
模型推理提供两种方式，区别如下：
## nms 后处理脚本（nms_script）
直接用官网 export.py 导出 onnx 模型，模型结构和官网一致，推理流程也和官方一致，NMS后处理采用脚本实现，
- 注意：如果使用的是 nms_script 方式，需要修改 model.yaml 文件，将其中的配置 conf_thres:0.4 和 iou_thres:0.5 修改为 conf_thres:0.001 和 iou_thres:0.6 ，后续该方式下精度测试也是采用修改后的配置；
## nms 后处理算子（nms_op）
- 注意：为提升模型端到端推理性能，我们对上一步导出的 onnx 模型做了修改，增加后处理算子，将 NMS 后处理的计算集成到模型中。后处理算子存在阈值约束，要求 conf>0.1 ，由于其硬性要求，所以 model.yaml 文件默认设置 conf_thres:0.4 。使用 nms_op 方式，不需要修改 model.yaml 文件；

# 模型转换
模型权重文件 .pth 转 .onnx ，再通过 ATC 工具 .onnx 转为离线推理模型 .om 文件；

## 获取权重文件
在链接中找到所需版本下载，也可以使用下述命令下载，
```shell
wget https://github.com/ultralytics/yolov5/releases/download/v${tag}/${model}.pt
```
命令参数说明：
- ${tag}：模型版本，可选 [2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]；
- ${model}：模型大小，可选 yolov5[n/s/m/l]，当前未适配 x；

跟执行例保持一致，实际使用时，tag 使用 6.1，model 使用 yolov5s；
```shell
/home/fuyuxin/yolov5# wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
/home/fuyuxin/yolov5# ll -h | grep yolov5s.pt
15M yolov5s.pt
```

## 导出 ONNX 模型
运行 bash pth2onnx.sh 导出动态 shape 的 ONNX 模型，模型参数在 model.yaml 中设置，
```shell
bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_script  # nms_script
bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_op  # nms_op
```
命令参数说明：
- tag：模型版本，可选 [2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]；
- model：模型大小，可选 yolov5[n/s/m/l]；
- nms_mode：模型推理方式，可选 [nms_op/nms_script]。nms_op 方式下，pth 导出 onnx 模型过程中会增加 NMS 后处理算子，后处理算子的参数 class_num、conf_thres 和 iou_thres 在 model.yaml 中设置；

这里根据 nms_script 和 nms_op 给出了两种转换方式。如果用 nms_script 需要修改 model.yaml 文件，如果用 nms_op 保持 model.yaml 文件默认，无需修改；

### model.yaml 文件示例及默认参数
```shell
# parameters
img_size: [640, 640]  # height, width
class_num: 80  # number of classes
conf_thres: 0.4  # object confidence threshold, conf>0.1 for nms_op
iou_thres: 0.5  # IOU threshold for NMS

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
stride: [8, 16, 32]
```

### 脚本 pth2onnx.sh 执行报错 os/pip 依赖汇总
这里，用 nms_op 方式，model.yaml 维持默认参数设置，
```shell
/home/fuyuxin/yolov5# bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_op
```
执行报错，os/pip 依赖汇总如下，
```shell
# openeuler24.03 安装 mesa-libGL
/home/fuyuxin/yolov5# yum install mesa-libGL -y
# ubuntu22.04 安装 mesa-libGL 相关
/home/fuyuxin/yolov5# apt-get install -y libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglu1-mesa-dev mesa-common-dev libglx-mesa0 libgbm-dev
# numpy 降级
/home/fuyuxin/yolov5# pip install numpy==1.24
```

### 下载 Arial.ttf 失败，手动下载，放到目标路径
下载链接：https://ultralytics.com/assets/Arial.ttf
```shell
wget -c https://ultralytics.com/assets/Arial.ttf
```
若无法下载则尝试访问 yolov5 的 github 仓库或其他官方资源获取正确的文件；

推荐预先手动下载该文件并放置到指定路径，实际执行如下，
```shell
mkdir -p ~/.config/Ultralytics/
cp /the/path/of/Arial.ttf ~/.config/Ultralytics/
```

### 调整 common 文件夹的位置，放到 yolov5 源代码根目录路径下
.pth 转 .onnx 执行报错，需要调整 common 路径的位置，将 Yolov5_for_Pytorch 下 common 及所有子目录/文件放到 yolov5 源代码根目录下，
```shell
/home/fuyuxin/yolov5/Yolov5_for_Pytorch# mv -r common/ ../
/home/fuyuxin/yolov5# tree
.
├── aipp.cfg
├── model.yaml
├── om_val.py
├── onnx2om.sh
├── pth2onnx.sh
├── requirements.txt
├── common
│   ├── atc_cfg
│   ├── __init__.py
│   ├── patch
│   ├── quantify
│   └── util
├── yolov5_postprocess.py
├── yolov5_preprocess_aipp.py
└── yolov5_preprocess.py
```

### yolov5s.pt 转 onnx 成功
```shell
/home/fuyuxin/yolov5# bash pth2onnx.sh --tag 6.1 --model yolov5s --nms_mode nms_op
=== pth2onnx args ===
 tag: 6.1
 model: yolov5s
 nms_mode: nms_op
Updated 106 paths from the index
HEAD is now at 3752807c YOLOv5 v6.1 release (#6739)
方式二 nms后处理算子
export: data=data/coco128.yaml, weights=['yolov5s.pt'], imgsz=[640, 640], batch_size=1, device=cpu, half=False, inplace=False, train=False, optimize=False, int8=False, dynamic=True, simplify=False, opset=11, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['onnx']
YOLOv5 🚀 v6.1-0-g3752807c torch 2.1.0 CPU

Fusing layers...
Model Summary: 213 layers, 7225885 parameters, 0 gradients

PyTorch: starting from yolov5s.pt with output shape (1, 3, 80, 80, 85) (14.8 MB)

ONNX: starting export with onnx 1.18.0...
ONNX: export success, saved as yolov5s.onnx (28.9 MB)

Export complete (28.87s)
Results saved to /home/fuyuxin/yolov5-npumig-wd/yolov5
Detect:          python detect.py --weights yolov5s.onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')
Validate:        python val.py --weights yolov5s.onnx
Visualize:       https://netron.app
Fusing layers...
Model Summary: 213 layers, 7225885 parameters, 0 gradients
pth导出onnx模型 Success

/home/fuyuxin/yolov5# ll -h | grep onnx
28M yolov5s.onnx
28M yolov5s_nms.onnx
```

## 使用 ATC 工具将 ONNX 模型转 OM 模型
### 配置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 查看芯片类型
执行命令查看芯片名称（${chip_name}）---> "910B4" ---> "Ascend910B4"，
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

### 导出非量化 OM 模型
运行 onnx2om.sh 脚本，参考执行，
```shell
bash onnx2om.sh --tag 6.1 --model yolov5s --nms_mode nms_script --bs 4 --soc Ascend310P3  # nms_script
bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend310P3  # nms_op
```
实际执行如下，
- bs=4
```shell
# nms_op
/home/fuyuxin/yolov5# bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend910B4
=== onnx2om args ===
 tag: 6.1
 model: yolov5s_nms
 nms_mode: nms_op
 quantify: False
 bs: 4
 soc: Ascend910B4
 with_aipp: False
nms后处理算子
ATC start working now, please wait for a moment.
....
ATC run success, welcome to the next use.

onnx导出om模型 Success

/home/fuyuxin/yolov5# ll -h | grep om
15M yolov5s_nms_bs4.om
```
- bs=8
```shell
/home/fuyuxin/yolov5# bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 8 --soc Ascend910B4
=== onnx2om args ===
 tag: 6.1
 model: yolov5s_nms
 nms_mode: nms_op
 quantify: False
 bs: 8
 soc: Ascend910B4
 with_aipp: False
nms后处理算子
ATC start working now, please wait for a moment.
....
ATC run success, welcome to the next use.

onnx导出om模型 Success

/home/fuyuxin/yolov5# ll -h | grep om
15M yolov5s_nms_bs8.om
```

### 导出量化OM模型（可选） --- 可选，这个没做
- 量化存在精度损失，要使用实际数据集进行校准以减少精度损失。提供 generate_data.py 生成校准数据，calib_img_list.txt 中提供默认的校准数据，根据实际数据路径修改。运行脚本会新建 calib_data 文件夹，将生成的数据 bin 文件放到该文件夹下；
```shell
python3 common/quantify/gen_calib_data.py
```
- 导出 OM 模型时设置 --quantify 参数，使能模型量化，量化对性能的提升视模型而定，实际效果不同；
```shell
bash onnx2om.sh --tag 6.1 --model yolov5s --nms_mode nms_script --bs 4 --soc Ascend310P3 --quantify True  # nms_script
bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend310P3 --quantify True  # nms_op
```
- 部分网络层量化后损失较大，可在 simple_config.cfg 中配置不需要量化的层名称，默认为空列表。skip_layers.cfg 中提供了参考写法，通常网络的首尾卷积层量化损失大些，其他版本可以用 Netron 打开模型，查找不需要量化的层名称；

# 开始推理验证

## 安装 ais_bench 工具
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

## 执行推理 & 精度验证
如果有多卡推理的需求，请跳过该步骤，om_val.py 该脚本不支持多卡推理,

运行 om_val.py 推理 OM 模型，模型参数在 model.yaml 中设置，结果默认保存在 predictions.json ，参考命令如下，
```shell
python3 om_val.py --tag 6.1 --model=yolov5s_bs4.om --nms_mode nms_script --batch_size=4  # nms_script
python3 om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4  # nms_op
```
实际执行如下，
```shell
# nms_op bs=4
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4
# nms_op bs=4
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs8.om --nms_mode nms_op --batch_size=8
```

### 执行 om_val.py 推理报错，需确保 coco 数据集处理完毕并放置到 yolov5 源代码根目录下
报错如下，执行 om_val.py 推理报错，结果保存文件 predictions.json 也没有在当前工作目录生成，
```shell
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs4.om success
[INFO] create model description success
0it [00:00, ?it/s]
Traceback (most recent call last):
  File "/home/fuyuxin/yolov5/om_val.py", line 81, in <module>
    main(opt, cfg)
  File "/home/fuyuxin/yolov5/om_val.py", line 53, in main
    summary.report(gpt.batch_size, output_prefix=None, display_all_summary=False)
  File "/usr/local/lib/python3.11/site-packages/ais_bench/infer/summary.py", line 201, in report
    ...
  File "/usr/local/lib/python3.11/site-packages/ais_bench/infer/summary.py", line 66, in get_list_info
    raise RuntimeError(f'summary.get_list_info failed: inner error')
RuntimeError: summary.get_list_info failed: inner error
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```
这里，在数据集准备时，需要将 coco 数据预处理完之后，将 coco 目录放置到 yolov5 源代码根目录下，再执行 om_val.py 推理，

### 再次执行推理 & 精度验证
- bs=4
```shell
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs4.om success
[INFO] create model description success
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [01:25<00:00, 14.62it/s]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 96.94300079345703, max = 85520.703125, mean = 43273.10063741455, median = 43368.703125, percentile(99%) = 84719.418359375
[INFO] throughput 1000*batchsize.mean(4)/NPU_compute_time.mean(43273.10063741455): 0.09243617723435195
[INFO] ------------------------------------------------------
saving results to yolov5s_nms_bs4_6.1_predictions.json
loading annotations into memory...
Done (t=1.04s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=14.64s).
Accumulating evaluation results...
DONE (t=2.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```
- bs=8
```shell
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs8.om --nms_mode nms_op --batch_size=8
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs8.om success
[INFO] create model description success
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [01:27<00:00,  7.13it/s]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 170.79200744628906, max = 87683.1328125, mean = 44170.01817277832, median = 44202.45703125, percentile(99%) = 86853.075859375
[INFO] throughput 1000*batchsize.mean(8)/NPU_compute_time.mean(44170.01817277832): 0.18111833164085825
[INFO] ------------------------------------------------------
saving results to yolov5s_nms_bs8_6.1_predictions.json
loading annotations into memory...
Done (t=1.03s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=14.98s).
Accumulating evaluation results...
DONE (t=2.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```

## 推理性能验证                            
可使用 ais_bench 推理工具的纯推理模式验证不同 batch_size 的 OM 模型的性能，参考命令如下，
```shell
python3 -m ais_bench --model=yolov5s_bs4.om --loop=1000 --batchsize=4  # nms_script
python3 -m ais_bench --model=yolov5s_nms_bs4.om --loop=1000 --batchsize=4  # nms_op
```
实际执行如下，
- bs=4
```shell
# nms_op bs=4
/home/fuyuxin/yolov5# python -m ais_bench --model=yolov5s_nms_bs4.om --loop=1000 --batchsize=4
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs4.om success
[INFO] create model description success
[INFO] warm up 1 done
loop inference exec: (1000/1000)|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]
Inference array Processing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.81s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.27099609375, max = 2.52606201171875, mean = 2.288239074707031, median = 2.2869873046875, percentile(99%) = 2.30322998046875
[INFO] throughput 1000*batchsize.mean(4)/NPU_compute_time.mean(2.288239074707031): 1748.0690912998807
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```
- bs=8
```shell
# nms_op bs=8
/home/fuyuxin/yolov5# python -m ais_bench --model=yolov5s_nms_bs8.om --loop=1000 --batchsize=8
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs8.om success
[INFO] create model description success
[INFO] warm up 1 done
loop inference exec: (1000/1000)|                                                                                                                                                       | 0/1 [00:00<?, ?it/s]
Inference array Processing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.51s/it]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 3.49267578125, max = 3.70703125, mean = 3.507873291015625, median = 3.5078125, percentile(99%) = 3.52001953125
[INFO] throughput 1000*batchsize.mean(8)/NPU_compute_time.mean(3.507873291015625): 2280.584085089283
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] end to reset device 0
[INFO] end to finalize acl
```

# duo卡推理
这里，如果硬件环境是 300IDUO 卡，则使用这里的 duo 卡版本推理方案，

910b4的卡，这里可以正常运行，

## 数据预处理
将原始数据转换为模型输入的数据。执行 yolov5_preprocess.py 脚本，完成数据预处理，
```shell
python3 yolov5_preprocess.py --data_path="./coco" --nms_mode nms_script --tag 6.1
python3 yolov5_preprocess.py --data_path="./coco" --nms_mode nms_op --tag 6.1
```
命令参数说明：
- data_path：coco数据集的路径；
- nms_mode：模型推理方式，可选 [nms_op/nms_script] , 默认 nms_script 推理方式；
- tag：模型版本，可选 [2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0] 执行后当前目录下生成 ./prep_data 目录用于储存预处理完的二进制数据，并且生成 path_list.npy 用于储存图片路径，生成 img_info 目录用于储存图片原始 shape 信息；

实际执行如下，
```shell
# nms_op 
/home/fuyuxin/yolov5# python yolov5_preprocess.py --data_path="./coco" --nms_mode nms_op --tag 6.1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [01:33<00:00, 53.66it/s]
The dataset has been processed.The image name is stored in img_name.npy,the shape of the image is stored in the img_info folder
/home/fuyuxin/yolov5# ll -h | grep img_name
313K img_name.npy
/home/fuyuxin/yolov5# ll -h | grep img_info
128K img_info
```

## 数据集推理
目前 ais_bench 已经支持多卡推理，若执行下述命令报错，请重新安装最新 ais_bench 工具。参考命令如下，
```shell
# nms_script
python3 -m ais_bench --m yolov5s_bs4.om --input ./prep_data --output ./results --output_dirname om_output --device 0,1
# nms_op
python3 -m ais_bench --m yolov5s_bs4.om --input ./prep_data,./img_info --output ./results --output_dirname om_output --device 0,1
```
命令参数说明：
- m：.om 模型路径；
- input：预处理生成的 ./prep_data 路径，如果使用 nms_op 则需要增加 ./img_info 路径；
- output：推理结果保存的地址，会在 ./results 下生成子目录；
- output_dirname：推理结果子目录名；
- device：现支持多卡推理，格式参考运行示例；

实际执行如下，
```shell
# nms_op
/home/fuyuxin/yolov5# python -m ais_bench --m yolov5s_nms_bs4.om --input ./prep_data,./img_info --output ./results --output_dirname om_output --device 0,1
[INFO] multidevice:[0, 1] run begin
[INFO] subprocess_0 main run
[INFO] subprocess_1 main run
[INFO] acl init success
[INFO] acl init success
[INFO] open device 0 success
[INFO] create new context
[INFO] open device 1 success
[INFO] create new context
[INFO] load model yolov5s_nms_bs4.om success
[INFO] create model description success
[INFO] try get model batchsize:4
[INFO] output path:./results/om_output/device0_0
[INFO] load model yolov5s_nms_bs4.om success
[INFO] create model description success
[INFO] try get model batchsize:4
[INFO] output path:./results/om_output/device1_1
[INFO] get filesperbatch files0 size:2457600 tensor0size:9830400 filesperbatch:4 runcount:1250
[INFO] get filesperbatch files0 size:2457600 tensor0size:9830400 filesperbatch:4 runcount:1250
[INFO] warm up 1 done
[INFO] subprocess_0 qsize:0 now waiting
[INFO] warm up 1 done
[INFO] subprocess_1 qsize:1 now waiting
[INFO] subprocess_1 qsize:2 ready to infer run
Inference array Processing:   7%|█████████▋                                                                                                                                 | 87/1250 [00:00<00:12, 95.71it/s][INFO] subprocess_0 qsize:2 ready to infer run
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:13<00:00, 94.13it/s]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.462890625, max = 3.5009765625, mean = 2.664621630859375, median = 2.635986328125, percentile(99%) = 3.107733154296875
[INFO] throughput 1000*batchsize.mean(4)/NPU_compute_time.mean(2.664621630859375): 1501.151215495443
[INFO] ------------------------------------------------------
Inference array Processing:  89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉               | 1113/1250 [00:12<00:01, 91.08it/s][INFO] unload model success, model Id is 1
Inference array Processing:  98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████   | 1223/1250 [00:13<00:00, 92.96it/s][INFO] end to reset device 1
Inference array Processing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1250/1250 [00:13<00:00, 90.60it/s]
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.4697265625, max = 3.5400390625, mean = 2.67673623046875, median = 2.6490478515625, percentile(99%) = 3.121630859375
[INFO] throughput 1000*batchsize.mean(4)/NPU_compute_time.mean(2.67673623046875): 1494.3571781443404
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] end to finalize acl
[INFO] end to reset device 0
[INFO] end to finalize acl
[INFO] multidevice run end qsize:4 result:0
[INFO] i:1 device_1 throughput:1501.151215495443                 start_time:1752744657.0383043 end_time:1752744670.3209918
[INFO] i:0 device_0 throughput:1494.3571781443404                 start_time:1752744658.0158062 end_time:1752744671.815301
[INFO] summary throughput:2995.508393639783

/home/fuyuxin/yolov5# ll ./results/om_output/
total 2244
drwxr-x--- 2 root root 270336 Jul 17 17:31 device0_0
-rw-r----- 1 root root 872398 Jul 17 17:31 device0_0_summary.json
drwxr-x--- 2 root root 270336 Jul 17 17:31 device1_1
-rw-r----- 1 root root 872673 Jul 17 17:31 device1_1_summary.json
```

## 后处理和精度验证
将推理结果转换为字典并储存进json文件，用于计算精度；
```shell
python3 yolov5_postprocess.py --nms_mode nms_script --ground_truth_json "./coco/instances_val2017.json" --output "./results/om_output/device0_0" --onnx yolov5s.onnx
```
命令参数说明：
- ground_truth_json：coco数据集标杆文件；
- output：推理结果保存的路径，需要指定到 bin 文件所在目录。单卡推理时路径为 ./results/om_output ；
- onnx：为 onnx 模型路径；
- nms_mode：模型推理方式，可选 [nms_op/nms_script] , 默认 nms_script 推理方式；

实际执行如下，
```shell
# nms_op
/home/fuyuxin/yolov5# python3 yolov5_postprocess.py --nms_mode nms_op --ground_truth_json "./coco/instances_val2017.json" --output "./results/om_output/device0_0" --onnx yolov5s_nms.onnx
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 5317.95it/s]
saving results to yolov5s_nms_predictions.json
loading annotations into memory...
Done (t=0.94s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.32s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=15.16s).
Accumulating evaluation results...
DONE (t=2.43s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
```
这里存在问题， --output 没法指定多卡场景（--output "./results/om_output/" 会报错，也无法指定多个 --output ，yolov5_postprocess.py 代码实现并不支持），涉及 yolov5_postprocess.py 的代码修改，让其支持 --output 指定多卡用于指定到多卡 bin 文件的目录；

# aipp
这里，aipp 是昇腾内置的图像预处理加速模块，主要为了提升预处理速度，对性能会有所提升。不过，这里模型推理方式，仅支持 nms_script 推理方式，这里暂不深究；

