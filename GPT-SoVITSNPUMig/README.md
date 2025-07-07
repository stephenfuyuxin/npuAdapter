# 开源仓参考
https://github.com/RVC-Boss/GPT-SoVITS

# 搭建开发环境
已安装cann (toolkit、kernel环境)，python，编译器等

# apt安装软件包
apt install -y ffmpeg libsox-dev

# pip安装软件包
lightning_fabric torch==2.1.0 pyyaml==6.0.2 numpy==1.26.0 
psutil ffmpeg-python gradio pypinyin opencc tornado scipy ml-dtypes decorator cloudpickle absl-py onnxruntime transformers pytorch-lightning cn2an jieba_fast wordsegment librosa LangSegment==0.2.0（numpy==1.26.0）einops matplotlib

## torch_npu==2.1.0
torch_npu下载链接（根据python版本按需下载）
```shell
https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

# 模型适配代码修改
进入GPT_SoVITS文件夹

## GPT_SoVITS/inference_webui.py
在 `import torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## GPT_SoVITS/module/mel_processing.py
删除代码
```python
from librosa.util import normalize, pad_center, tiny
```

## GPT_SoVITS/prepare_datasets/1-get-text.py
在 `import torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
在 `import librosa,torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## GPT_SoVITS/prepare_datasets/3-get-semantic.py
在 `import logging, librosa, utils, torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## GPT_SoVITS/s1_train.py
在 `import torch, platform` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
修改，
```python
process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
process_group_backend="hccl"
```

## GPT_SoVITS/s2_train.py
在 `import torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
修改，
```python
backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "hccl"
backend = "hccl"
```

## config.py
在 `import torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## tools/uvr5/webui.py
在 `import torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## webui.py
在 `import json,yaml,warnings,torch` 下一行增加，
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## cuda.py（前提：安装lightning_fabric）
找到pip安装包的存储路径修改 `cuda.py` ，修改 `_check_cuda_matmul_precision` 函数，
```shell
vim *****/python3.10.12/lib/python3.10/site-packages/lightning_fabric/accelerators/cuda.py
```
修改，
```python
if not torch.cuda.is_available() or not _is_ampere_or_later(device):
if not torch.cuda.is_available():
```

# 注意事项
## 需要重定向 tmpdir
在当前目录新建 tmpdir 文件
```shell
mkdir tmpdir
export TMPDIR=/{绝对路径}/
```
必须写绝对路径，否则报错如下，
```shell
RuntimeError: InnerRun:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:200 OPS function error: Identity, error code is 500002
[ERROR] 2025-01-08-23:21:51 (PID:225981, Device:0, RankID:-1) ERR01100 OPS call acl api failed
[Error]: A GE error occurs in the system.
Rectify the fault based on the error information in the ascend log.
E40021: [PID:225981] 2025-01-08-23:21:51.183.024 Failed to compile Op [trans_TransData_51,]. (oppath: [Compile /home/HwHiAiUser/ascend-toolkit/8.0.RC3/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/trans_data.py failed with errormsg/stack: File "/home/HwHiAiUser/ascend-toolkit/8.0.RC3/python/site-packages/tbe/tvm/error_mgr/tbe_python_error_mgr.py", line 111, in raise_the_python_err raise TBEPythonError(args_dict)
tvm.error.mgr.the_python_error_mgr.TBEPythonError: [EB9999] [errClass:NA][errCode:EB0500]message:dir path is invalid[errSolution:N/A]During handling of the above exception, another exception occurred:tvm.error.mgr.TBEPythonError: [EB9999] [errClass:NA][errCode:EB0500]message:compile cce error： , errpath: /home/HwHiAiUser/GPT-SoVITS/TEMP/tmp16b6zyt0/ , path target:/home/HwHiAiUser/GPT-SoVITS/kernel_meta/kernel_108500985825518633342227/kernel_meta/the_transdata_3a4934b66338d2015687817db0b839e1a3b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53e2b9d53
```
到此适配完成，可以开始测试。
