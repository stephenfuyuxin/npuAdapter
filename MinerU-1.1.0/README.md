# MinerU - magic-pdf 1.1.0

- [参考链接](#参考链接)
- [版本配套关系](#版本配套关系)
- [模型文件](#模型文件)
  - [下载](#下载)
  - [压缩/解压缩](#压缩/解压缩)
- [修改记录](#修改记录)
  - [修改 doc_analyze_by_custom_model.py 文件](#修改-doc_analyze_by_custom_model.py-文件)
  - [onnx 转 om](#onnx-转-om)
  - [修改 det、rec、utility 文件](#修改-det、rec、utility-文件)
    - [det](#det)
    - [rec](#rec)
    - [utility](#utility)
  - [修改 config.yaml 文件](#修改-config.yaml-文件)
  - [修改 infer_engine.py 文件](#修改-infer_engine.py-文件)
  - [修改 rapid_table.py 文件](#修改-rapid_table.py-文件)
  - [修改 magic-pdf.json 新增 unitable 推理分支](#修改-magic-pdf.json-新增-unitable-推理分支)
    - [unitable 模型权重](#unitable-模型权重)
    - [修改 ppocr_273_mod.py 文件](#修改-ppocr_273_mod.py-文件)
  - [修改 magic-pdf.json 文件](#修改-magic-pdf.json-文件)
- [通过 magic-pdf 进行推理](#通过-magic-pdf-进行推理)
- [FAQ](#FAQ)
  - [问题1 下载模型文件时若出现代理相关报错](#问题1-下载模型文件时若出现代理相关报错)
  - [问题2 下载模型文件时若出现认证相关报错](#问题2-下载模型文件时若出现认证相关报错)
  - [问题3 执行模型下载操作时候，出现传参错误](#问题3-执行模型下载操作时候，出现传参错误)
  - [问题4 算子 torchvision-nms 运行在 CPU 上](#问题4-算子-torchvision-nms-运行在-CPU-上)
  - [问题5 有关 acl 尝试销毁的 Stream 不在当前上下文而报错销毁失败](#问题5-有关-acl-尝试销毁的-Stream-不在当前上下文而报错销毁失败)

# 参考链接
MinerU github 链接: [MinerU github](https://github.com/opendatalab/MinerU)

MinerU magic-pdf 工具: [MinerU magic-pdf Release](https://github.com/opendatalab/MinerU/releases)

MinerU Command Line 参考: [MinerU Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)

工程化使用可参考 `dockerbuild/Ascend/Template/README.md`，并根据实际情况进行修改以适配构建工程

# 版本配套关系
**表1** 版本配套表，
  | 软件配套                       |   版本      | 备注说明                                        |
  | ----------------------------- | ----------- | ---------------------------------------------- |
  | driver, firmware              | 24.1.0.3    |                                                |
  | cann-toolkit, cann-kernels    | 8.0.0       |                                                |
  | python                        | 3.10.14     |                                                |
  | torch                         | 2.1.0       |                                                |
  | torch_npu                     | 2.1.0.post8 |                                                |
  | torchvision                   | 0.16.0      |                                                |
  | trochvision_npu               | 0.16.0+xxx  |                                                |
  | magic-pdf                     | 1.1.0       |                                                |

# 模型文件
这里包含下载模型文件和压缩/解压缩模型文件。

## 下载
模型文件下载，自从 Hugging Face 整体迁移到 CDN 集群后，静态代理可用性变差，因网络原因，使用国内的 Model Scope 下载模型文件。
```python
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```
脚本 `download_models.py` 会自动下载模型文件并配置好配置文件中的模型目录。

配置文件可以在用户目录中找到，文件名为 `magic-pdf.json`，模型文件下载完成之后，这里显示如下，
```sh
model_dir is: /root/.cache/modelscope/hub/opendatalab/PDF-Extract-Kit-1___0/models
layoutreader_model_dir is: /root/.cache/modelscope/hub/ppaanngggg/layoutreader
The configuration file has been configured successfully, the path is: /root/magic-pdf.json
```
注：文件 `magic-pdf.json` 后面会用到，需保留。

## 压缩/解压缩
下载完的模型文件进行压缩，`tar` 包在工程化构建时可通过 `wget` 传输到镜像构建工程时解压缩使用。
```sh
cd ~/.cache/
tar -czvf mineru-modelscope.tar modelscope/
```
使用时，可解压缩，
```sh
tar -xzvf mineru-modelscope.tar -C ~/.cache/
```
使用 `magic-pdf` 进行推理，首次推理会触发下载 `PaddleOCR` 预训练推理模型，相关回显类似如下，
```sh
download https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar to /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.tar
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.89M/4.89M [00:03<00:00, 1.23MiB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar to /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11.0M/11.0M [00:10<00:00, 1.06MiB/s]
download https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar to /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.19M/2.19M [00:00<00:00, 4.84MiB/s]
2025-04-25 07:52:03,667 - DownloadModel - DEBUG: /usr/local/python3.10.14/lib/python3.10/site-packages/rapid_table/models/slanet-plus.onnx already exists
[2025-04-25 07:52:03,667] [DEBUG] download_model.py:34 - /usr/local/python3.10.14/lib/python3.10/site-packages/rapid_table/models/slanet-plus.onnx already exists
```
这些文件通常用于部署 `PaddleOCR` 模型时，加载预训练的权重和配置，以便快速实现文本检测、识别和方向分类等功能，

- `ch_PP-OCRv4_det_infer.tar` 是 `PaddleOCRv4` 中文文本检测（Detection）模型文件。识别图像中文字位置和边界；

- `ch_PP-OCRv4_rec_infer.tar` 是 `PaddleOCRv4` 中文文本识别（Recognition）模型文件。识别已检测文本区域具体文字内容；

- `ch_ppocr_mobile_v2.0_cls_infer.tar` 是 `PaddleOCR Mobile v2.0` 中文方向分类（Classification）模型文件。判断文本方向（例如水平、竖直等），以便更好地处理不同方向的文本；

这部分也可以进行压缩，`tar` 包在工程化构建时可通过 `wget` 传输到镜像构建工程时解压缩使用。
```sh
cd ~
tar -czvf paddleocr.tar .paddleocr/
```
使用时，可解压缩，
```sh
tar -xzvf paddleocr.tar -C ~
```

# 修改记录

## 修改 doc_analyze_by_custom_model.py 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/magic_pdf/model`
```sh
cp doc_analyze_by_custom_model.py doc_analyze_by_custom_model.py.bak
vim doc_analyze_by_custom_model.py
```
修改如下，新增
```python
import torch_npu
import torchvision_npu
```

## onnx 转 om
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime/models/`
```sh
ch_ppocr_mobile_v2.0_cls_infer.onnx
ch_PP-OCRv4_det_infer.onnx
ch_PP-OCRv4_rec_infer.onnx
```
将上述 `onnx` 文件转 `om` 文件，操作如下，
```sh
atc --model=ch_ppocr_mobile_v2.0_cls_infer.onnx --framework=5 --output=ch_ppocr_mobile_v2.0_cls_infer --input_shape="x:-1,3,-1,-1" --soc_version=Ascend910B3 
atc --model=ch_PP-OCRv4_det_infer.onnx --framework=5 --output=ch_PP-OCRv4_det_infer --input_shape="x:-1,3,-1,-1" --soc_version=Ascend910B3 --precision_mode=must_keep_origin_dtype --allow_hf32=false
atc --model=ch_PP-OCRv4_rec_infer.onnx --framework=5 --output=ch_PP-OCRv4_rec_infer --input_shape="x:-1,3,-1,-1" --soc_version=Ascend910B3 --precision_mode=must_keep_origin_dtype --allow_hf32=false
```
分别得到对应名称的 `om` 文件，
```sh
ch_ppocr_mobile_v2.0_cls_infer_linux_aarch64.om
ch_PP-OCRv4_det_infer_linux_aarch64.om
ch_PP-OCRv4_rec_infer_linux_aarch64.om
```

## 修改 det、rec、utility 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/paddleocr/tools/infer`
```sh
cp predict_det.py predict_det.py.bak
vim predict_det.py
cp predict_rec.py predict_rec.py.bak
vim predict_rec.py
cp utility.py utility.py.bak
vim utility.py
```

### det
```python
239         if self.use_onnx:
240             input_dict = {}
241             input_dict[self.input_tensor.name] = img
242             #outputs = self.predictor.run(self.output_tensors, input_dict)
243             outputs = self.predictor.infer([img], mode='dymshape', custom_sizes=1000000000)
```

### rec
```python
29 import acl
608                 if self.use_onnx:
609                     context, ret = acl.rt.get_context()
610                     input_dict = {}
611                     input_dict[self.input_tensor.name] = norm_img_batch
612                     #outputs = self.predictor.run(self.output_tensors, input_dict)
613                     outputs = self.predictor.infer([norm_img_batch], mode='dymshape', custom_sizes=1000000000)
614                     preds = outputs[0]
615                     acl.rt.set_context(context)
```

### utility
```python
28 from ais_bench.infer.interface import InferSession
181     if args.use_onnx:
182         import onnxruntime as ort
183         model_file_path = model_dir
184         target_models = ['ch_PP-OCRv4_det_infer.onnx', 'ch_PP-OCRv4_rec_infer.onnx']
185         if any(model_file_path.endswith(model) for model in target_models):
186             model_file_path = model_file_path.replace(".onnx", "_linux_aarch64.om")
187         if not os.path.exists(model_file_path):
188             raise ValueError("not find model file path {}".format(
189                 model_file_path))
190         #sess = ort.InferenceSession(model_file_path)
191         sess = InferSession(0, model_file_path)
192         return sess, sess.get_inputs()[0], None, None
```

## 修改 config.yaml 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime`
```sh
cp config.yaml config.yaml.bak
vim config.yaml
```
修改如下，分别在 `Det` 、 `Cls` 、 `Rec` 中添加 `om_model_path` 的内容，
```python
16 Det:
23     model_path: models/ch_PP-OCRv4_det_infer.onnx
24     om_model_path: /usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer_linux_aarch64.om
38 Cls:
45     model_path: models/ch_ppocr_mobile_v2.0_cls_infer.onnx
46     om_model_path: /usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime/models/ch_ppocr_mobile_v2.0_cls_infer_linux_aarch64.om
53 Rec:
60     model_path: models/ch_PP-OCRv4_rec_infer.onnx
61     om_model_path: /usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer_linux_aarch64.om
```

## 修改 infer_engine.py 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/rapidocr_onnxruntime/utils`
```sh
cp infer_engine.py infer_engine.py.bak
vim infer_engine.py
```
修改如下，新增，
```python
import acl
from ais_bench.infer.interface import InferSession
```
在 `OrtInferSession` 类构造函数中新增 `om_model_path` 及 `self.om_session` 引入 `acl.session` 会话机制处理 `om` 推理，
```python
class OrtInferSession:
    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("OrtInferSession")

        model_path = config.get("model_path", None)
        om_model_path = config.get("om_model_path", None)
        self._verify_model(model_path)

        self.cfg_use_cuda = config.get("use_cuda", None)
        self.cfg_use_dml = config.get("use_dml", None)

        self.had_providers: List[str] = get_available_providers()
        EP_list = self._get_ep_list()

        sess_opt = self._init_sess_opts(config)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=EP_list,
        )
        self._verify_providers()
        self.om_session = InferSession(0, om_model_path)
``` 
在 `__call__()` 加入 `pyACL` 用于上下文获取、模型推理及上下文释放，并返回推理结果，
```python
    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        #input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            context, ret = acl.rt.get_context()
            out = self.om_session.infer([input_content], mode='dymshape', custom_sizes=1000000000)
            acl.rt.set_context(context)
            return out
            #return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e
```

## 修改 rapid_table.py 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/magic_pdf/model/sub_modules/table/rapidtable`
```sh
cp rapid_table.py rapid_table.py.bak
vim rapid_table.py
```
修改如下，新增
```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
修改 `ocrmodel_name` 初始化 `ocr` 引擎分支，不论当前推理硬件是否为 `CUDA` ，都从 `rapidocr_onnxruntime` 模块导入 `RapidOCR` 启用推理加速用于检测、分类和识别，
```python
        self.table_model = RapidTable(input_args)

        # if ocr_engine is None:
        #     self.ocr_model_name = "RapidOCR"
        #     if torch.cuda.is_available():
        #         from rapidocr_paddle import RapidOCR
        #         self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        #     else:
        #         from rapidocr_onnxruntime import RapidOCR
        #         self.ocr_engine = RapidOCR()
        # else:
        #     self.ocr_model_name = "PaddleOCR"
        #     self.ocr_engine = ocr_engine

        self.ocr_model_name = "RapidOCR"
        #if torch.cuda.is_available():
        #    from rapidocr_paddle import RapidOCR
        #    self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        #else:
        #    from rapidocr_onnxruntime import RapidOCR
        #    self.ocr_engine = RapidOCR()
        from rapidocr_onnxruntime import RapidOCR
        self.ocr_engine = RapidOCR()
```

## 修改 magic-pdf.json 新增 unitable 推理分支
magic-pdf 默认 `rapid_table` 表格处理，通过 `slanet_plus.onnx` 进行特定优化或增强处理，以下为修改 `magic-pdf.json` 配置项举例，
```json
    "table-config": {
        "model": "rapid_table",
        "sub_model": "slanet_plus",
        "enable": true,
        "max_time": 400
```
可按需增加 `unitable` 推理分支，即仍接入 `rapid_table` 模型，将 `sub_model` 改为 `ubitable` 分支，
```json
    "table-config": {
        "model": "rapid_table",
        "sub_model": "unitable",
        "enable": true,
        "max_time": 400
```
注，`rapid_table` 对应一整页的、行数比较长的表格，效果不是很好，存在直接少整列、行之间错乱的情况。对比下来 `rapid_table` 的 `unitable` 模型识别效果最好，但性能较差。

### unitable 模型权重
- 下载链接：https://www.modelscope.cn/models/RapidAI/RapidTable/files
- unitable权重绝对路径，python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/rapid_table/models`
- 重命名，将 `decoder.pth` 、 `encoder.pth` 、 `vocab.json` 都添加前缀 `unitable_` ，变为 `unitable_decoder.pth` 、 `unitable_encoder.pth` 、 `unitable_vocab.json`
```sh
/usr/local/python3.10.14/lib/python3.10/site-packages/rapid_table/models# ll
-rw-r--r-- 1 root root   7758305 Apr 25 03:51 slanet-plus.onnx
-rw-r--r-- 1 root root 160306370 Apr 28 08:30 unitable_decoder.pth
-rw-r--r-- 1 root root 345783752 Apr 28 08:30 unitable_encoder.pth
-rw-r--r-- 1 root root    251594 Apr 28 08:30 unitable_vocab.json
```
其中，`rapid_table` 默认是 `slanet_plus` 这个模型，升级 `rapid_table` 并支持 `unitable` 这个模型需要额外的 `.pth` 权重及 `json` 文件。

### 修改 ppocr_273_mod.py 文件
python安装路径，如 `/usr/local/python3.10.14/lib/python3.10/site-packages/magic_pdf/model/sub_modules/ocr/paddleocr`
```sh
cp ppocr_273_mod.py ppocr_273_mod.py.bak
vim ppocr_273_mod.py
```
修改如下，不论设备是否满足 `cuda` 加速，统一都走 `onnx` 推理（在 `npu`推理场景下走 `onnx` 推理会强制切到 `om` 推理）
```python
class ModifiedPaddleOCR(PaddleOCR):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.lang = kwargs.get('lang', 'ch')
        # 在cpu架构为arm且不支持cuda时调用onnx、
        #if not torch.cuda.is_available() and platform.machine() in ['arm64', 'aarch64']:
        #    self.use_onnx = True
        #    onnx_model_manager = ONNXModelSingleton()
        #    self.additional_ocr = onnx_model_manager.get_onnx_model(**kwargs)
        #else:
        #    self.use_onnx = False
        self.use_onnx = True
        onnx_model_manager = ONNXModelSingleton()
        self.additional_ocr = onnx_model_manager.get_onnx_model(**kwargs)
```

## 修改 magic-pdf.json 文件
在 [下载](#下载) 章节中，执行 `download_models.py` 脚本自动生成用户目录下的 `magic-pdf.json` 文件，并自动配置默认模型路径。

对于 Ascend NPU 场景，还需要额外修改 `device-mode` 为 `npu` ，执行以下操作，
```sh
sed -i 's|cpu|npu|g' ~/magic-pdf.json
```

# 通过 magic-pdf 进行推理
参考 [MinerU Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)
```sh
# show version
magic-pdf -v

# command line example
magic-pdf -p {some_pdf} -o {some_output_dir} -m [ocr|txt|auto]
```

# FAQ

## 问题1 下载模型文件时若出现代理相关报错
类似如下，
```sh
Proxy tunneling failed: FilteredUnable to establish SSL connection.
```
则，去使能当前 `http_proxy` 和 `https_proxy` 设置
```sh
unset http_proxy
unset https_proxy
```

## 问题2 下载模型文件时若出现认证相关报错
类似如下，
```sh
ERROR: cannot verify gcore.jsdelivr.net's certificate, issued by ‘CN=Sectigo RSA Domain Validation Secure Server CA,O=Sectigo Limited,L=Salford,ST=Greater Manchester,C=GB’:
  Self-signed certificate encountered.
To connect to gcore.jsdelivr.net insecurely, use `--no-check-certificate'.
```
则，`wget` 加上 `--no-check-certificate` 参数，如下
```python
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py --no-check-certificate
```

## 问题3 执行模型下载操作时候，出现传参错误
类似如下，
```sh
Traceback (most recent call last):
  File "/root/download_models.py", line 44, in <module>
    model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns)
TypeError: snapshot_download() got an unexpected keyword argument 'allow_patterns'
```
则，修改 `download_models.py` 脚本后再执行下载操作，
```python
44     # model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns)
45     model_dir = snapshot_download('opendatalab/PDF-Extract-Kit-1.0')
```

## 问题4 算子 torchvision:nms 运行在 CPU 上
类似如下，
```sh
[W compiler_depend.ts:51] Warning: CAUTION: The operator 'torchvision::nms' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
```
引入 `torchvision_npu` 将 `torchvision` 相关算子运行在 npu 上，目前 `torchvision_npu` 仅有 `0.16.0` 的配套，因此，`torch` 与 `torchvision` 对应使用 `2.1.0` 和 `0.16.0` 的配套。

## 问题5 有关 acl 尝试销毁的 Stream 不在当前上下文而报错销毁失败
类似如下，
```python
[Error]: The stream is not in the current context.
        Check whether the context where the stream is located is the same as the current context.
EE9999: Inner Error!
EE9999: [PID: 65276] 2025-04-27-08:22:06.816.143 Stream destroy failed, stream is not in current ctx, stream_id=2.[FUNC:StreamDestroy][FILE:api_impl.cc][LINE:1096]
        TraceBack (most recent call last):
       rtStreamDestroyForce execute failed, reason=[stream not in current context][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
       destroy stream force failed, runtime result = 107003[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
 (function operator())
```
报错并未产生实质性影响，推理结果仍然以 `output` 的形式输出完毕，大概推理结束后 `acl.session` 并未正常结束退出的报错。
