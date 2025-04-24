# MinerU - magic-pdf 1.3.6

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
  - [修改 magic-pdf.json 文件](#修改-magic-pdf.json-文件)
- [通过 magic-pdf 进行推理](#通过-magic-pdf-进行推理)
- [FAQ](#FAQ)
  - [问题1 下载模型文件时若出现代理相关报错](#问题1-下载模型文件时若出现代理相关报错)
  - [问题2 下载模型文件时若出现认证相关报错](#问题2-下载模型文件时若出现认证相关报错)
  - [问题3 执行模型下载操作时候，出现传参错误](#问题3-执行模型下载操作时候，出现传参错误)
  - [问题4 data did not match any variant of untagged enum ModelWrapper](#问题4-data-did-not-match-any-variant-of-untagged-enum-ModelWrapper)
  - [问题5 cannot import tv_tensors from torchvision_npu](#问题5-cannot-import-name-tv_tensors-from-torchvision_npu)

# 参考链接
MinerU github 链接: [MinerU github](https://github.com/opendatalab/MinerU)

MinerU magic-pdf 工具: [MinerU magic-pdf Release](https://github.com/opendatalab/MinerU/releases)

MinerU Command Line 参考: [MinerU Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)

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
  | magic-pdf                     | 1.3.6       |                                                |

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
这里代码变了 --- 存疑（？？？） ，暂时不做任何修改，
```python
        self.table_model = RapidTable(input_args)

        # self.ocr_model_name = "RapidOCR"
        # if torch.cuda.is_available():
        #     from rapidocr_paddle import RapidOCR
        #     self.ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
        # else:
        #     from rapidocr_onnxruntime import RapidOCR
        #     self.ocr_engine = RapidOCR()

        # self.ocr_model_name = "PaddleOCR"
        self.ocr_engine = ocr_engine
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
magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
```
当前NPU上执行存在问题，参考 [FAQ](#FAQ) 问题4、问题5。

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

## 问题4 data did not match any variant of untagged enum ModelWrapper
类似如下，
```sh
xxxx-xx-xx xx:xx:xx.xxx | ERROR | magic_pdf.tools.cli:parse_doc:134 - data did not match any variant of untagged enum ModelWrapper at line 249230 column 3
```
1、通常与 `transformers` 库的版本不兼容有关。检查当前使用的 `transformers` 库版本，并尝试升级到最新版本。实际调测，将 `transformers` 从 `4.42` 升级到 `4.47` 。

2、因模型文件或配置文件存在损坏，确保使用的模型文件和配置文件是完整且未损坏。实际调测，重新下载模型文件确实存在retry的权重文件，多次下载确保无遗漏。

按照上述操作，再无上述报错出现。

## 问题5 cannot import tv_tensors from torchvision_npu
类似如下，
```sh
xxxx-xx-xx xx:xx:xx.xxx | ERROR | magic_pdf.tools.cli:parse_doc:134 - cannot import name 'tv_tensors' from 'torchvision_npu' (/usr/local/python3.10.14/lib/python3.10/site-packages/torchvision/__init__.py)
... ... ...
... ... ...
ImportError: cannot import name 'tv_tensors' from 'torchvision_npu' (/usr/local/python3.10.14/lib/python3.10/site-packages/torchvision/__init__.py)
```
目前初步判断 `tv_tensors` 在 `torchvision_npu` 中并未适配，除非不用 `torchvision_npu` 而直接使用原生 `torchvision` ，但这样的话，在NPU上的性能会较差。
