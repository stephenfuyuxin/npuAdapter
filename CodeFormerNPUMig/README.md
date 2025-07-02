# github 参考
https://github.com/sczhou/CodeFormer

# npu driver& firmware
driver
```shell
/usr/local/Ascend/driver# cat version.info
Version=24.1.rc3
ascendhal_version=7.35.23
aicpu_version=1.0
tdt_version=1.0
log_version=1.0
prof_version=2.0
dvppkernels_version=1.1
tsfw_version=1.0
Innerversion=V100R001C19SPC001B124
compatible_version=[V100R001C13],[V100R001C15],[V100R001C17],[V100R001C18],[V100R001C19]
compatible_version_fw=[7.0.0,7.5.99]
package_version=24.1.rc3
```
firmware
```shell
/usr/local/Ascend/firmware# cat version.info
Version=7.5.0.1.129
firmware_version=1.0
package_version=24.1.rc3
compatible_version_drv=[23.0.0,23.0.0.],[24.0,24.0.],[24.1,24.1.]
```

# docker images& docker run
## docker images
昇腾镜像仓库：https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

所用镜像名称：mindie:2.0.RC2-800I-A2-py311-openeuler24.03-lts

## docker run
```shell
# vim run.sh
```
```shell
#!/bin/bash
docker run -it -d --net=host --shm-size=500g --privileged \
--name sig-mha-codeformer-0-7  \
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

# python& pip version
python
```shell
# python --version
Python 3.11.6
```
pip
```shell
# pip --version
pip 23.3.1 from /usr/lib/python3.11/site-packages/pip (python 3.11)
```

# codeformer install
使用容器方案，未使用 conda 隔离环境，开源参考如下，
```shell
# git clone this repository
git clone https://github.com/sczhou/CodeFormer
cd CodeFormer

# create new anaconda env
conda create -n codeformer python=3.8 -y
conda activate codeformer

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib (only for face detection or cropping with dlib)
```

## pip install 
```shell
CodeFormer]# pip3 install -r requirements.txt

Successfully installed PySocks-1.7.1 addict-2.4.0 beautifulsoup4-4.13.4 future-1.0.0 gdown-5.2.0 imageio-2.37.0 lazy-loader-0.4 lmdb-1.6.2 lpips-0.1.4 opencv-python-4.11.0.86 platformdirs-4.3.8 scikit-image-0.25.2 soupsieve-2.7 tb-nightly-2.20.0a20250629 tensorboard-data-server-0.7.2 tifffile-2025.6.11 werkzeug-3.1.3 yapf-0.43.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```
pip install 全量参考，pip freeze --no-deps.txt

## python basicsr/setup.py develop
执行 setup.py 之前，需要先安装 OS 依赖 mesa-libGL，然后执行 setup.py，
```shell
CodeFormer]# yum install mesa-libGL -y
CodeFormer]# python basicsr/setup.py develop
```

## conda install -c conda-forge dlib (only for face detection or cropping with dlib)
容器方案，未使用 conda 隔离环境，以下步骤替代，
```shell
# yum install -y cmake boost-devel openblas-devel lapack-devel
# yum install -y libX11-devel mesa-libGL-devel
```
安装 dlib 必需的开发工具和编译器，
```shell
# yum install -y cmake gcc-c++
# yum install -y gtk3-devel boost-devel openblas-devel lapack-devel libX11-devel mesa-libGL-devel
```
上述过程执行完毕之后，安装 dlib,
```shell
# pip install dlib
```

# Download Pre-trained Models
开源参考如下，
```shell
Download the facelib and dlib pretrained models from [Releases | Google Drive | OneDrive] to the weights/facelib folder. You can manually download the pretrained models OR download by running the following command:
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib (only for dlib face detector)
```
实际执行，
```shell
CodeFormer]# python scripts/download_pretrained_models.py facelib
CodeFormer]# python scripts/download_pretrained_models.py dlib
```
开源参考如下，
```shell
Download the CodeFormer pretrained models from [Releases | Google Drive | OneDrive] to the weights/CodeFormer folder. You can manually download the pretrained models OR download by running the following command:
python scripts/download_pretrained_models.py CodeFormer
```
实际执行，
```shell
CodeFormer]# python scripts/download_pretrained_models.py CodeFormer
```

# Prepare Testing Data
开源参考如下，
```shell
You can put the testing images in the inputs/TestWhole folder. If you would like to test on cropped and aligned faces, you can put them in the inputs/cropped_faces folder. You can get the cropped and aligned faces by running the following command:
# you may need to install dlib via: conda install -c conda-forge dlib
python scripts/crop_align_face.py -i [input folder] -o [output folder]
```
实际执行，
```shell
CodeFormer]# python scripts/crop_align_face.py -i inputs/TestWhole -o inputs/cropped_faces
```
inputs/TestWhole，inputs下并没有这个文件或文件夹；

inputs/cropped_faces，这个文件夹执行前已存在，执行完之后文件并无变化；

额外多下载了这个文件，CodeFormer/weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat

# Testing
开源参考如下，
```shell
[Note] If you want to compare CodeFormer in your paper, please run the following command indicating --has_aligned (for cropped and aligned face), as the command for the whole image will involve a process of face-background fusion that may damage hair texture on the boundary, which leads to unfair comparison.
Fidelity weight w lays in [0, 1]. Generally, smaller w tends to produce a higher-quality result, while larger w yields a higher-fidelity result. The results will be saved in the results folder.
```

## Face Restoration (cropped and aligned face) --- 人脸修复
开源参考如下，
```shell
# For cropped and aligned faces (512x512)
python inference_codeformer.py -w 0.5 --has_aligned --input_path [image folder]|[image path]
```
-w 0.5: 权重参数，取值范围为[0,1]，取值越小，则图像修复的质量更高，取值越大，则会产生保真度更高的图片；

--has_aligned: 这个参数表示输入的人脸图像已经经过对齐处理。CodeFormer 需要对齐的人脸图像才能进行修复，如果你的图像已经对齐，则需要添加这个参数；

--input_path [image folder]|[image path]: 这个参数指定了输入图像的路径，可以是单个图像文件路径或包含多个图像的文件夹路径；

### 修改 inference_codeformer.py 脚本
使用自动迁移，对 inference_codeformer.py 脚本进行修改，

1、自动迁移，引入 `import torch_npu` 和 `from torch_npu.contrib import transfer_to_npu` ；

2、将运行设备从 cuda 改为 npu ，将 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 改为 `torch.device('npu' if torch_npu.npu.is_available() else 'cpu')`；

PyTorch 2.4.0 及之前版本，自动迁移参考样例如下，
```python
import torch 
import torch_npu 
... 
from torch_npu.contrib import transfer_to_npu
```
实际修改，
```shell
CodeFormer]# cp inference_codeformer.py inference_codeformer.py.org
CodeFormer]# vim inference_codeformer.py
```
加入自动迁移相关代码，
```python
  1 import os
  2 import cv2
  3 import argparse
  4 import glob
  5 import torch
  6 import torch_npu
  7 from torchvision.transforms.functional import normalize
  8 from basicsr.utils import imwrite, img2tensor, tensor2img
  9 from basicsr.utils.download_util import load_file_from_url
 10 from basicsr.utils.misc import gpu_is_available, get_device
 11 from facelib.utils.face_restoration_helper import FaceRestoreHelper
 12 from facelib.utils.misc import is_gray
 13
 14 from basicsr.utils.registry import ARCH_REGISTRY
 15
 16 from torch_npu.contrib import transfer_to_npu
```
推理设备设置相关代码变更加入脚本，
```python
 58 if __name__ == '__main__':
 59     print(torch_npu.npu.is_available())
 60     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 61     device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
 62     print(device)
 63     # device = get_device()
 64     # print(device)
 65     parser = argparse.ArgumentParser()
```

### 修改完成之后推理运行
```shell
CodeFormer]# python inference_codeformer.py -w 0.5 --has_aligned --input_path inputs/cropped_faces
```

## Whole Image Enhancement --- 整体增强
该场景适用于图片中包含多张人像，用于图片效果整体增强，开源参考如下，
```shell
# For whole image
# Add '--bg_upsampler realesrgan' to enhance the background regions with Real-ESRGAN
# Add '--face_upsample' to further upsample restorated face with Real-ESRGAN
python inference_codeformer.py -w 0.7 --input_path [image folder]|[image path]
```

### 修改 retinaface.py 脚本
在已修改inference_codeformer.py 脚本的基础上，需要额外修改 retinaface.py 脚本，

绝对路径：CodeFormer/facelib/detection/retinaface
```shell
retinaface]# cp retinaface.py retinaface.py.org
retinaface]# vim retinaface.py
```
引入 `import torch_npu`，修改推理设备 `device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')`
```python
  1 import cv2
  2 import numpy as np
  3 import torch
  4 import torch_npu
  5 import torch.nn as nn
  6 import torch.nn.functional as F
  7 from PIL import Image
  8 from torchvision.models._utils import IntermediateLayerGetter as IntermediateLayerGetter
  9
 10 from facelib.detection.align_trans import get_reference_facial_points, warp_and_crop_face
 11 from facelib.detection.retinaface.retinaface_net import FPN, SSH, MobileNetV1, make_bbox_head, make_class_head, make_landmark_head
 12 from facelib.detection.retinaface.retinaface_utils import (PriorBox, batched_decode, batched_decode_landm, decode, decode_landm,
 13                                                  py_cpu_nms)
 14
 15 from basicsr.utils.misc import get_device
 16 # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 17 device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
 18 # device = get_device()
```

### 修改完之后推理运行
```shell
CodeFormer]# python inference_codeformer.py -w 0.7 --input_path inputs/whole_imgs
```

## Video Enhancement --- 视频修复
开源参考如下，
```shell
# For Windows/Mac users, please install ffmpeg first
conda install -c conda-forge ffmpeg
# For video clips
# Video path should end with '.mp4'|'.mov'|'.avi'
python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 --input_path [video path]
```
如果是容器方案，未用 conda 隔离环境，则需在容器中安装 ffmpeg 使用，
```shell
yum install -y ffmpeg
```

### 开源未提供模糊视频
github 开源仓并未提供模糊视频，若有需要可自行找一些模糊视频进行修复处理；

## Face Colorization (cropped and aligned face) --- 上色修复
开源参考如下，
```shell
# For cropped and aligned faces (512x512)
# Colorize black and white or faded photo
python inference_colorization.py --input_path [image folder]|[image path]
```

### 修改 inference_colorization.py 脚本
```shell
CodeFormer]# cp inference_colorization.py inference_colorization.py.org
CodeFormer]# vim inference_colorization.py
```
加入自动迁移、设置推理设备相关代码，
```python
  1 import os
  2 import cv2
  3 import argparse
  4 import glob
  5 import torch
  6 import torch_npu
  7 from torchvision.transforms.functional import normalize
  8 from basicsr.utils import imwrite, img2tensor, tensor2img
  9 from basicsr.utils.download_util import load_file_from_url
 10 from basicsr.utils.misc import get_device
 11 from basicsr.utils.registry import ARCH_REGISTRY
 12
 13 from torch_npu.contrib import transfer_to_npu
 14
 15 pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth'
 16
 17 if __name__ == '__main__':
 18     print(torch_npu.npu.is_available())
 19     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 20     device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
 21     print(device)
 22     # device = get_device()
 23     parser = argparse.ArgumentParser()
```

### 修改完之后运行
```shell
CodeFormer]# python inference_colorization.py --input_path inputs/gray_faces
```
注，png 文件中存在 iCCP 块，标准的 sRGB 图像不应该在 iCCP 块中包含额外的色彩配置文件，但这个并不影响输出；

## Face Inpainting (cropped and aligned face) --- 破损修复
开源参考如下，
```shell
# For cropped and aligned faces (512x512)
# Inputs could be masked by white brush using an image editing app (e.g., Photoshop) 
# (check out the examples in inputs/masked_faces)
python inference_inpainting.py --input_path [image folder]|[image path]
```

### 修改 inference_inpainting.py 脚本
```shell
CodeFormer]# cp inference_inpainting.py inference_inpainting.py.org
CodeFormer]# vim inference_inpainting.py
```
加入自动迁移、设置推理设备相关代码，
```python 
  1 import os
  2 import cv2
  3 import argparse
  4 import glob
  5 import torch
  6 import torch_npu
  7 from torchvision.transforms.functional import normalize
  8 from basicsr.utils import imwrite, img2tensor, tensor2img
  9 from basicsr.utils.download_util import load_file_from_url
 10 from basicsr.utils.misc import get_device
 11 from basicsr.utils.registry import ARCH_REGISTRY
 12
 13 from torch_npu.contrib import transfer_to_npu
 14
 15 pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth'
 16
 17 if __name__ == '__main__':
 18     print(torch_npu.npu.is_available())
 19     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 20     device = torch.device('npu' if torch_npu.npu.is_available() else 'cpu')
 21     print(device)
 22     # device = get_device()
 23     parser = argparse.ArgumentParser()
```

### 修改完之后运行
```shell
CodeFormer]# python inference_inpainting.py --input_path inputs/masked_faces
```
