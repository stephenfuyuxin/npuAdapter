# 参考链接
https://huggingface.co/briaai/RMBG-2.0

https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0

https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/summary

模型全量下载，.safetensors和onnx文件都下载

# 使用 liveportrait 镜像工程化构建
参考 npuAdapter 下 LivePortraitNPUMig

参考 dockerbuild/Ascend/LivePortrait 下镜像工程

# torch版本配套关系
Python 3.10.12

torch                  2.1.0

torch-npu              2.1.0.post8

torchvision            0.16.0

# 模型指定 requirements.txt
torch

torchvision

pillow

kornia

transformers

```shell
pip install kornia
pip list | grep kornia
kornia                  0.8.0
kornia_rs               0.1.8
```

# 若存在问题，缺少'transformers_modules.RMBG-2'提示
```shell
ModuleNotFoundError: No module named 'transformers_modules.RMBG-2'
```
通过更新transformers版本及安装timm依赖解决
```shell
pip install transformers==4.42.4
pip install timm
```

# Usage 改写脚本
命名推理入口脚本为 example.py，则
```python
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch_npu
import time
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from torch_npu.contrib import transfer_to_npu

#model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
model = AutoModelForImageSegmentation.from_pretrained('/data/fuyuxin/zhaohang/RMBG2.0/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
#model.to('cuda')
model.to('npu')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#image = Image.open(input_image_path)
image = Image.open("/data/fuyuxin/zhaohang/RMBG2.0/RMBG-2.0/p1.jpg")
#input_images = transform_image(image).unsqueeze(0).to('cuda')
input_images = transform_image(image).unsqueeze(0).to('npu')

# Prediction
with torch.no_grad():
    start_time = time.time()
    #preds = model(input_images)[-1].sigmoid().cpu()
    preds = model(input_images)[-1].sigmoid().npu()
    end_time = time.time()
    timelength = end_time - start_time
    print(f"======================================{timelength}")
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")
```

# 安装torchvision_npu
https://gitee.com/ascend/vision

开源参考如下，
```shell
# 下载Torchvision Adapter代码，进入插件根目录
git clone https://gitee.com/ascend/vision.git vision_npu
cd vision_npu
git checkout v0.16.0-6.0.0
# 安装依赖库
pip3 install -r requirement.txt
# 初始化CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh # Default path, change it if needed.
# 编包
python setup.py bdist_wheel
# 安装
cd dist
pip install torchvision_npu-0.16.*.whl
```

# error: invalid command 'bdist_wheel'
https://blog.csdn.net/Dontla/article/details/132237839

通过安装 wheel 依赖解决
```shell
pip install wheel
```

最终所用的 torch/torch_npu/torchvision/torchvison_npu 关联关系
torch                  2.1.0

torch-npu              2.1.0.post8

torchvision            0.16.0

torchvision-npu        0.16.0+git450a7cb

