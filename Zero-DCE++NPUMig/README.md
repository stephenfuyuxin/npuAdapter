# github参考
https://github.com/Li-Chongyi/Zero-DCE_extension

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

# npu-smi info
910b4 32G 8卡
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
--name sig-mha-zerodce-0-7  \
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

# zero-dce_extension install
使用容器方案
```shell
git clone https://github.com/Li-Chongyi/Zero-DCE_extension
cd Zero-DCE_extension/Zero-DCE++
Zero-DCE++]# pwd
Zero-DCE_extension/Zero-DCE++
```
参考开源仓库文件结构，保持一致，
```shell
├── data
│   ├── test_data 
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── model.py # Zero-DEC++ network
├── dataloader.py
├── snapshots_Zero_DCE++
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```

# Test: lowlight_test.py
```shell
cd Zero-DCE++
python lowlight_test.py
```
这里直接通过 lowlight_test.py 脚本推理验证
```shell
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.
```
该脚本将处理“test_data”文件夹下子文件夹中的图片，并在“data”文件夹中创建一个新的文件夹“result”。您可以在“result”文件夹中找到增强后的图片。
- 测试数据目录：Zero-DCE++/data/test_data
- 推理结果目录：Zero-DCE++/data/result_Zero_DCE++

## 自动迁移
引入 `import torch_npu` 和 `from torch_npu.contrib import transfer_to_npu`；
```shell
Zero-DCE++]# cp lowlight_test.py lowlight_test.py.org
Zero-DCE++]# vim lowlight_test.py
```
PyTorch 2.4.0 及之前版本，自动迁移参考样例如下，
```python
import torch 
import torch_npu 
... 
from torch_npu.contrib import transfer_to_npu
```
在 lowlight_test.py 中加入自动迁移相关代码，
```python
  1 import torch
  2 import torch_npu
  3 import torch.nn as nn
  4 import torchvision
  5 import torch.backends.cudnn as cudnn
  6 import torch.optim
  7 import os
  8 import sys
  9 import argparse
 10 import time
 11 import dataloader
 12 import model
 13 import numpy as np
 14 from torchvision import transforms
 15 from PIL import Image
 16 import glob
 17 import time
 18
 19 from torch_npu.contrib import transfer_to_npu
 20
```

## 运行python lowlight_test.py
```shell
Zero-DCE++]# python lowlight_test.py
```
回显，
```shell
/usr/local/lib64/python3.11/site-packages/torch_npu/contrib/transfer_to_npu.py:295: ImportWarning:
*************************************************************************************************************
The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.npu and torch.nn.Module.npu now..
The torch.cuda.DoubleTensor is replaced with torch.npu.FloatTensor cause the double type is not supported now..
The backend in torch.distributed.init_process_group set to hccl now..
The torch.cuda.* and torch.cuda.amp.* are replaced with torch.npu.* and torch.npu.amp.* now..
The device parameters have been replaced with npu in the function below:
torch.logspace, torch.randint, torch.hann_window, torch.rand, torch.full_like, torch.ones_like, torch.rand_like, torch.randperm, torch.arange, torch.frombuffer, torch.normal, torch._empty_per_channel_affine_quantized, torch.empty_strided, torch.empty_like, torch.scalar_tensor, torch.tril_indices, torch.bartlett_window, torch.ones, torch.sparse_coo_tensor, torch.randn, torch.kaiser_window, torch.tensor, torch.triu_indices, torch.as_tensor, torch.zeros, torch.randint_like, torch.full, torch.eye, torch._sparse_csr_tensor_unsafe, torch.empty, torch._sparse_coo_tensor_unsafe, torch.blackman_window, torch.zeros_like, torch.range, torch.sparse_csr_tensor, torch.randn_like, torch.from_file, torch._cudnn_init_dropout_state, torch._empty_affine_quantized, torch.linspace, torch.hamming_window, torch.empty_quantized, torch._pin_memory, torch.autocast, torch.load, torch.Generator, torch.set_default_device, torch.Tensor.new_empty, torch.Tensor.new_empty_strided, torch.Tensor.new_full, torch.Tensor.new_ones, torch.Tensor.new_tensor, torch.Tensor.new_zeros, torch.Tensor.to, torch.Tensor.pin_memory, torch.nn.Module.to, torch.nn.Module.to_empty
*************************************************************************************************************
warnings.warn(msg, ImportWarning)
/usr/local/lib64/python3.11/site-packages/torch_npu/contrib/transfer_to_npu.py:250: RuntimeWarning: torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu, which currently does not support them, if you need to enable them, please do not use transfer_to_npu.
warnings.warn(msg, RuntimeWarning)
data/test_data/real/132_2_.png
0.06954550743103027
data/test_data/real/474_1_.png
0.002228260040283203
data/test_data/real/761_2_.png
0.0030651092529296875
data/test_data/real/101_1_.png
0.002067089080810547
data/test_data/real/893_1_.png
0.002007722854614258
data/test_data/real/512_1_.png
0.003720521926879883
data/test_data/real/661_0_.png
0.002080678939819336
data/test_data/real/744_1_.png
0.003317594528198242
data/test_data/real/132_3_.png
0.002027273178100586
data/test_data/real/1021_3_.png
0.002007007598876953
data/test_data/real/11_0_.png
0.0018761157989501953
data/test_data/real/142_3_.png
0.0019893646240234375
data/test_data/real/744_3_.png
0.0021851062774658203
data/test_data/real/101_3_.png
0.0019958019256591797
data/test_data/real/101_2_.png
0.0019180774688720703
data/test_data/real/126_3_.png
0.0020608901977539062
data/test_data/real/761_3_.png
0.0020494461059570312
data/test_data/real/474_0_.png
0.002100229263305664
data/test_data/real/11_3_.png
0.0020275115966796875
data/test_data/real/450_2_.png
0.002325296401977539
data/test_data/real/893_0_.png
0.0022292137145996094
data/test_data/real/147_2_.png
0.0019974708557128906
data/test_data/real/893_2_.png
0.002646207809448242
data/test_data/real/159_0_.png
0.002222299575805664
data/test_data/real/126_1_.png
0.0020325183868408203
data/test_data/real/1021_0_.png
0.001990079879760742
data/test_data/real/126_2_.png
0.0019598007202148438
data/test_data/real/435_0_.png
0.0029969215393066406
data/test_data/real/893_3_.png
0.0019664764404296875
data/test_data/real/142_1_.png
0.002029895782470703
data/test_data/real/11_2_.png
0.002032756805419922
data/test_data/real/761_1_.png
0.002203702926635742
data/test_data/real/512_0_.png
0.0020134449005126953
data/test_data/real/744_2_.png
0.001977205276489258
data/test_data/real/142_2_.png
0.002328634262084961
data/test_data/real/132_0_.png
0.0020246505737304688
data/test_data/real/474_3_.png
0.002218961715698242
data/test_data/real/101_0_.png
0.0020837783813476562
data/test_data/real/147_3_.png
0.002030611038208008
data/test_data/real/435_2_.png
0.0021486282348632812
data/test_data/real/761_0_.png
0.0020723342895507812
data/test_data/real/450_1_.png
0.002031087875366211
data/test_data/real/132_1_.png
0.0021355152130126953
data/test_data/real/744_0_.png
0.0020935535430908203
data/test_data/real/435_3_.png
0.002173185348510742
data/test_data/real/661_3_.png
0.002508878707885742
data/test_data/real/159_1_.png
0.002057313919067383
data/test_data/real/159_3_.png
0.0020999908447265625
data/test_data/real/147_0_.png
0.0023560523986816406
data/test_data/real/142_0_.png
0.002034902572631836
data/test_data/real/147_1_.png
0.002616405487060547
data/test_data/real/661_1_.png
0.002078533172607422
data/test_data/real/1021_1_.png
0.0019240379333496094
data/test_data/real/512_3_.png
0.0021996498107910156
data/test_data/real/512_2_.png
0.0021038055419921875
data/test_data/real/450_3_.png
0.0019533634185791016
data/test_data/real/450_0_.png
0.0021915435791015625
data/test_data/real/474_2_.png
0.0020830631256103516
data/test_data/real/11_1_.png
0.002420186996459961
data/test_data/real/435_1_.png
0.0025200843811035156
data/test_data/real/159_2_.png
0.002333402633666992
data/test_data/real/661_2_.png
0.0019152164459228516
data/test_data/real/1021_2_.png
0.0019528865814208984
data/test_data/real/126_0_.png
0.0021648406982421875
0.2077476978302002
```
