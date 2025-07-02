# KwaiVGI/LivePortrait 参考链接
https://github.com/KwaiVGI/LivePortrait

# conda环境
可用可不用，容器方案即可
```shell
# create env using conda
conda create -n LivePortrait python=3.10
conda activate LivePortrait
```

# torch相关配套关系
```shell
# torch=2.1.0
pip3 install torch-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# tourch_npu=2.1.0
pip3 install torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

torchvision=0.16.0
pip3 install torchvision==0.16.0

torchaudio
```

# 拖开源代码
```shell
git clone https://github.com/KwaiVGI/LivePortrait
cd LivePortrait
```

# OS依赖
```shell
apt-get install -y ffmpeg
```

# pip依赖
开源仓 requirements_base.txt 和 requirements.txt
requirements_base.txt
```shell
numpy==1.26.4
pyyaml==6.0.1
opencv-python==4.10.0.84
scipy==1.13.1
imageio==2.34.2
lmdb==1.4.1
tqdm==4.66.4
rich==13.7.1
ffmpeg-python==0.2.0
onnx==1.16.1
scikit-image==0.24.0
albumentations==1.4.10
matplotlib==3.9.0
imageio-ffmpeg==0.5.1
tyro==0.8.5
gradio==5.1.0
pykalman==0.9.7
pillow>=10.2.0
```
requirements.txt
```shell
-r requirements_base.txt

onnxruntime-gpu==1.18.0
transformers==4.38.0
```
存在的问题，

1、onnxruntime-gpu==1.18.0 这个装不上，提示没有这个版本；
```shell
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu==1.18.0 (from versions: none)
ERROR: No matching distribution found for onnxruntime-gpu==1.18.0
```
2、onnxruntime 需要额外安装；
```shell
pip3 install onnxruntime
```
3、其他 pip 依赖缺失；
```shell
pip3 install decorator
pip3 install attrs
pip3 install psutil
```

# msit
这里为了使用 ais_bench 安装 msit 工具，

gitee.com链接：https://gitee.com/ascend/msit/blob/master/msit/docs/install/README.md

这部分可以参考镜像构建工程中的离线安装，构建镜像时打入镜像中固化，参考如下
```dockerfile
RUN wget -q http://172.17.0.1:3000/msit.tar -P /home/liveportrait && \
    tar -xzvf msit.tar && \
    cd ~/msit/msit/ && \
    pip install wheel==0.45.1 && \
    pip3 install . && \
    msit download all --dest ./pkg-cache && \
    cd ~ && \
    rm -rf ~/*.tar ~/msit && \
    rm -rf ~/.cache/pip
```

# 验证ais_bench是否安装成功
```shell
~/LivePortrait/msit/msit$ python3
```
```python
>>> from ais_bench.infer.interface import InferSession
>>> 
>>> exit()
```
未报错即可

# 预训练权重
直接用开源仓的预训练权重获取方法

从 HuggingFace 下载预训练权重的最简单方法是：
```shell
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```
若无法访问 HuggingFace 平台，可以访问其镜像网站 hf-mirror 进行下载操作：
```shell
# !pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```
或者，可以从Google Drive或百度云（进行中）下载所有预训练权重。解压并将它们放置在 ./pretrained_weights 目录下，

实际放在了 ~/LivePortrait/pretrained_weights 目录下，

Google Drive：https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib

baiduyun：https://pan.baidu.com/s/1MGctWmNla_vZxDbEp2Dtzw?pwd=z5cn

确保目录结构如所示包含本仓库该路径其中展示的内容。

# onnx 转 om
将 landmark.onnx 转 landmark.om
```shell
~/LivePortrait$ cd ~/LivePortrait/pretrained_weights/liveportrait/
~/LivePortrait/pretrained_weights/liveportrait$ atc --model=landmark.onnx --framework=5 --output=landmark --input_shape="input:1,3,224,224" --soc_version=Ascend910B3
~/LivePortrait/pretrained_weights/liveportrait$ ll
-rw------- 1 HwHiAiUser HwHiAiUser  67961748 Jan 16 16:21 landmark.om
-rw-r--r-- 1 HwHiAiUser HwHiAiUser 114666491 Dec 31 15:07 landmark.onnx
```

# 修改 live_portrait_pipeline.py
可参考 diff 回显
```shell
diff live_portrait_pipeline.py live_portrait_pipeline.py.org
```

# 修改 human_landmark_runner.py
可参考 diff 回显
```shell
diff human_landmark_runner.py human_landmark_runner.py.org
```

# 修改 inference.py
可参考 diff 回显
```shell
diff inference.py inference.py.org
```

# 运行 inference.py
```shell
~/LivePortrait$ python3 inference.py
```

# 部分中间过程报错记录

## 报错，启用算子执行同步模式
```shell
python3 inference.py
```
```shell
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is GridSampler3D.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, pleace set the environment variable ASCEND_LAUNCH_BLOCKING=1.
[ERROR] 2024-12-31-15:30:14 (PID:484057, Device:0, RankID:-1) ERR00100 PTA call acl api failed
```
PyTorch训练或在线推理场景，可通过此环境变量控制算子执行时是否启动同步模式；

由于NPU模型训练时默认算子异步执行，导致算子执行过程中出现报错时，打印的报错堆栈信息并不是实际的调用栈信息。当设置为“1”时，强制算子采用同步模式运行，这样能够打印正确的调用栈信息，从而更容易地调试和定位代码中的问题。设置为“0”时则会采用异步方式执行。默认配置为0；
```shell
export ASCEND_LAUNCH_BLOCKING=1
```
针对 inference.py 进行自动迁移，加入torch/torch_npu两行调度代码之后，`export ASCEND_LAUNCH_BLOCKING=1`可`unset`维持默认值
```python
# coding: utf-8
"""
for human
"""

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline

torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
```
修改 inference.py 之后，可以 `unset ASCEND_LAUNCH_BLOCKING` 取消算子执行同步模式；

## 报错，GE error/GridSampler3D/500002
```shell
python3 inference.py
```
GridSampler3D算子后续通过cann版本解决在npu上进行适配
```shell
RuntimeError: InnerRun:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:200 OPS function error: GridSampler3D, error code is 500002
[ERROR] 2024-12-31-15:35:08 (PID:491533, Device:0, RankID:-1) ERR01100 OPS call acl api failed
[Error]: A GE error occurs in the system.
        Rectify the fault based on the error information in the ascend log.
E89999: Inner Error!
E89999: [PID: 491533] 2024-12-31-15:35:08.619.694 op[GridSampler3D75], runParams is null.[FUNC:Tiling4GridSampler3D][FILE:grid_sampler_3d.cc][LINE:157]
        TraceBack (most recent call last):
       [Exec][Op]Execute op failed. op type = GridSampler3D, ge result = 4294967295[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
```
