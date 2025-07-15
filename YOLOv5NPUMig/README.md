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
YOLOv5每个版本主要有4个开源模型，分别为YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，四个模型的网络结构基本一致，只是其中的模块数量与卷积核个数不一致。YOLOv5s模型最小，其它的模型都在此基础上对网络进行加深与加宽。

从描述来看，yolov5本身还有版本迭代更新，分为以下几个版本，
```shell
url=https://github.com/ultralytics/yolov5
tag=v2.0/v3.1/v4.0/v5.0/v6.0/v6.1/v6.2/v7.0
model_name=yolov5
```
yolov5自身各版本之间模型结构的差异，比如Conv模块各版本差异示例如下，
| yolov5版本 | Conv模块激活函数 |
| --------- | --------------- |
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

gitee 上命令执行例，以 yolov6 v6.1 为准

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
# 返回yolov5_for_pytorch目录
cd ..
# tag v7.0 需要执行以下操作
git apply 7.0.patch
```
如果用 yolov5 v7.0 需要额外执行 git apply ，执行例以 v6.1 为准，则无须执行；

将推理部署代码放到 yolov5 源码相应目录下，
```shell
# git clone https://gitee.com/ascend/modelzoo-GPL.git
```
按照 tree 放置文件，
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
实际执行如下，
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
请访问msit推理工具代码仓，根据readme文档进行工具安装surgeon组件，
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
模型使用 coco2017 val数据集 进行精度评估，在 yolov5 源码根目录下新建 coco 文件夹，数据集放到 coco 里，文件结构如下，
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
val2017.txt 中保存 .jpg 相对路径，请自行生成该 txt 文件，文件内容实例如下，
```shell
./val2017/00000000139.jpg
./val2017/00000000285.jpg
……
./val2017/00000581781.jpg
```

# 模型推理
模型推理提供两种方式，区别如下：
## nms后处理脚本（nms_script）
直接用官网export.py导出onnx模型，模型结构和官网一致，推理流程也和官方一致，NMS后处理采用脚本实现。
- 注意：如果使用的是nms_script方式，需要修改model.yaml文件，将其中的配置conf_thres:0.4和iou_thres:0.5修改为conf_thres:0.001和iou_thres:0.6，后续该方式下精度测试也是采用修改后的配置。
## nms后处理算子（nms_op）
- 注意：为提升模型端到端推理性能，我们对上一步导出的onnx模型做了修改，增加后处理算子，将NMS后处理的计算集成到模型中。后处理算子存在阈值约束，要求 conf>0.1，由于其硬性要求，所以model.yaml文件默认设置conf_thres:0.4。使用nms_op方式，不需要修改model.yaml文件。

# 模型转换
模型权重文件 .pth 转 .onnx ，再 ATC 工具 .onnx 转为离线推理模型 .om 文件

## 获取权重文件
在链接中找到所需版本下载，也可以使用下述命令下载，
```shell
wget https://github.com/ultralytics/yolov5/releases/download/v${tag}/${model}.pt
```
命令参数说明：
- ${tag}：模型版本，可选 [2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]；
- ${model}：模型大小，可选 yolov5[n/s/m/l]，当前未适配X；
跟执行例保持一致，

这里，tag 用 6.1，model 用 yolov5s；
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
- tag：模型版本，可选[2.0/3.1/4.0/5.0/6.0/6.1/6.2/7.0]；
- model：模型大小，可选yolov5[n/s/m/l]；
- nms_mode：模型推理方式，可选[nms_op/nms_script]。nms_op 方式下，pth 导出 onnx 模型过程中会增加 NMS 后处理算子，后处理算子的参数 class_num、conf_thres 和 iou_thres 在 model.yaml 中设置；

这里根据nms_script和nms_op给出了两种转换方式。如果用 nms_script 需要修改 model.yaml 文件，如果用 nms_op 保持 model.yaml 文件默认，无需修改，

### model.yaml 默认参数
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
这里，用 nms_op 方式，model.yaml 保持默认参数设置，
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
原因：网络问题或链接无效；

解决方法：检查网络连接是否正常；

确保链接 https://ultralytics.com/assets/Arial.ttf 是有效的。如果链接无效，可以尝试访问 yolov5 的 github 仓库或其他官方资源获取正确的文件。

如果网络问题导致下载失败，可以尝试手动下载该文件并放置到指定路径（如 /root/.config/Ultralytics/ ）。
```shell
mkdir -p /root/.config/Ultralytics/
cp /the/path/of/Arial.ttf /root/.config/Ultralytics/
```

### 调整 common 文件夹的位置，放到yolov5路径下
执行报错，需要调整 common 路径的位置，
```shell
/home/fuyuxin/yolov5/Yolov5_for_Pytorch# cp -r common/ ../
/home/fuyuxin/yolov5# tree
.
├── aipp.cfg
├── model.yaml
├── om_val.py
├── onnx2om.sh
├── pth2onnx.sh
├── requirements.txt
├── Yolov5_for_Pytorch
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
/home/fuyuxin/yolov5# ll -h | grep ".onnx"
28M yolov5s.onnx
28M yolov5s_nms.onnx
```

## 使用 ATC 工具将 ONNX 模型转 OM 模型
### 配置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 查看芯片类型
执行命令查看芯片名称（${chip_name}）--- "910B4" --- "Ascend910B4"，
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
```shell
# nms_op
/home/fuyuxin/yolov5# bash onnx2om.sh --tag 6.1 --model yolov5s_nms --nms_mode nms_op --bs 4 --soc Ascend910B4
/home/fuyuxin/yolov5# ll -h | grep ".om"
15M yolov5s_nms_bs4.om
```

### 导出量化OM模型（可选） --- 可选，这个没做

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
# nms_op
/home/fuyuxin/yolov5# python om_val.py --tag 6.1 --model=yolov5s_nms_bs4.om --nms_mode nms_op --batch_size=4
```
报错如下，
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
结果保存文件 predictions.json 也没有生成，

## 推理性能验证                            
可使用 ais_bench 推理工具的纯推理模式验证不同 batch_size 的 OM 模型的性能，参考命令如下，
```shell
python3 -m ais_bench --model=yolov5s_bs4.om --loop=1000 --batchsize=4  # nms_script
python3 -m ais_bench --model=yolov5s_nms_bs4.om --loop=1000 --batchsize=4  # nms_op
```
实际执行如下，
```shell
# nms_op
/home/fuyuxin/yolov5# python -m ais_bench --model=yolov5s_nms_bs4.om --loop=1000 --batchsize=4
```
