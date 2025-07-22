# Qwen3-1.7B-Base 基于 MindSpeed 预训练
在 MindSpeed 框架下实现 Qwen-3 模型的无缝运行，这里记录预训练的过程；

# 开源参考
https://github.com/NVIDIA/Megatron-LM

https://gitee.com/ascend/MindSpeed-LLM

https://gitee.com/ascend/MindSpeed

https://www.modelscope.cn/Qwen/Qwen3-1.7B-Base

https://modelers.cn/datasets/AI_Connect/alpaca/tree/main/data

# 硬件环境
在 MindSpeed 框架下运行 Qwen3-1.7B-Base 参考硬件配置如下，以 800I A2 单机8卡预训练为例，

| 类型   | 硬件      | 配套                 |
| ------ | -------- | -------------------- |
| 预训练 | 800I A2   | Ascend 910B4 32G * 8 |

# 参考配套
- driver: 24.1.rc3
- firmware: 7.5.0.1.129
- cann-toolkit: 910b_8.1.RC1
- cann-kernels: 8.1.RC1_linux
- python: 3.11.6
- PyTorch: 2.1.0
- torch_npu: 2.1.0.post12
- apex: 0.1

# torch& torch_npu
镜像基于 python 3.11.6 的配套，
```shell
pip install /the/path/of/torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install /the/path/of/torch_npu-2.1.0.post12-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

# apex
```shell
/home/fuyuxin/qwen3pretrain# git clone -b master https://gitee.com/ascend/apex.git
/home/fuyuxin/qwen3pretrain# cd apex/
/home/fuyuxin/qwen3pretrain/apex# bash scripts/build.sh --python=3.11
```
这里 python3.11 可编译生成，过程中会自动拉取 apex 官方源码，请保证网络畅通，生成的二进制包在 apex/dist 目录下，
```shell
/home/fuyuxin/qwen3pretrain/apex/apex/dist# ll
apex-0.1+ascend-cp311-cp311-linux_aarch64.whl

/home/fuyuxin/qwen3pretrain/apex/apex/dist# pip3 install --upgrade apex-0.1+ascend-cp311-cp311-linux_aarch64.whl
Processing ./apex-0.1+ascend-cp311-cp311-linux_aarch64.whl
Installing collected packages: apex
Successfully installed apex-0.1+ascend
```

# 目录结构
```shell
/home/fuyuxin/qwen3pretrain# # tree
.
├── apex
├── dataset
│   └── train-00000-of-00001-a09b74b3ef9c3b56.parquet
├── Megatron-LM
├── MindSpeed
├── MindSpeed-LLM
├── qwen3-1point7b-base
├── qwen3-1point7b-mcore
└── run.sh
```
相关目录说明如下，
- apex，Ascend apex adapter 开源仓，自动混合精度训练功能用于模型训练；
- dataset，数据集目录用于放置 Alpaca 数据集以及数据预处理之后的数据集文件；
- Megatron-LM，megatron 开源仓；
- MindSpeed，mindspeed 开源仓；
- MindSpeed-LLM，mindspeed-llm开源仓；
- qwen3-1point7b-base，Qwen3-1.7B-Base 权重目录，用于预训练；
- qwen3-1point7b-mcore，Qwen3-1.7B-Base 进行权重转换之后的 mcore 权重；
- run.sh，镜像启动容器脚本；

# 获取开源仓
```shell
/home/fuyuxin/qwen3pretrain# git clone https://gitee.com/ascend/MindSpeed-LLM.git
/home/fuyuxin/qwen3pretrain# git clone https://github.com/NVIDIA/Megatron-LM.git
/home/fuyuxin/qwen3pretrain/Megatron-LM# cd Megatron-LM
/home/fuyuxin/qwen3pretrain/Megatron-LM# git checkout core_r0.8.0
/home/fuyuxin/qwen3pretrain/Megatron-LM# cp -r megatron ../MindSpeed-LLM/
/home/fuyuxin/qwen3pretrain/Megatron-LM# cd ..
/home/fuyuxin/qwen3pretrain# cd MindSpeed-LLM
/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# mkdir logs
/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# mkdir dataset
/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# mkdir ckpt
```

# 安装加速库
```shell
/home/fuyuxin/qwen3pretrain# git clone https://gitee.com/ascend/MindSpeed.git
/home/fuyuxin/qwen3pretrain# cd MindSpeed
/home/fuyuxin/qwen3pretrain/MindSpeed# checkout commit from MindSpeed core_r0.8.0
/home/fuyuxin/qwen3pretrain/MindSpeed# git checkout 2c085cc9
/home/fuyuxin/qwen3pretrain/MindSpeed# pip install -r requirements.txt
/home/fuyuxin/qwen3pretrain/MindSpeed# pip3 install -e .
/home/fuyuxin/qwen3pretrain/MindSpeed# cd ../MindSpeed-LLM
/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# pip install --upgrade pip setuptools wheel
/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# pip install -r requirements.txt
```
由于首发最新版本支持，要求 transformers 版本为 4.51.3 ，用户需执行以下命令，
```shell
pip install transformers==4.51.3
```

# 权重转换

## 权重下载
下载 `Qwen3-1.7B-Base` 模型权重，这个权重版本仅用于预训练场景，
```shell
git clone https://www.modelscope.cn/Qwen/Qwen3-1.7B-Base.git
```

## 权重转换
MindSpeed-LLM 提供脚本将已经 huggingface 开源权重转换为 mcore 权重，用于训练、推理、评估等任务，

使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本，
```shell
/home/fuyuxin/qwen3pretrain# cd MindSpeed-LLM

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# cp tests/0day/qwen3/qwen3-1.7b/ckpt_convert_qwen3_1point7b_hf2mcore.sh tests/0day/qwen3/qwen3-1.7b/ckpt_convert_qwen3_1point7b_hf2mcore.sh.org

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# vim tests/0day/qwen3/qwen3-1.7b/ckpt_convert_qwen3_1point7b_hf2mcore.sh
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 1 \
       --load-dir /home/fuyuxin/qwen3pretrain/qwen3-1point7b-base \
       --save-dir /home/fuyuxin/qwen3pretrain/qwen3-1point7b-mcore \
       --tokenizer-model /home/fuyuxin/qwen3pretrain/qwen3-1point7b-base/tokenizer.json \
       --model-type-hf qwen3 \
       --params-dtype bf16 \
       --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# bash tests/0day/qwen3/qwen3-1.7b/ckpt_convert_qwen3_1point7b_hf2mcore.sh
......
saving checkpoint at iteration       1 to /home/fuyuxin/qwen3pretrain/qwen3-1point7b-mcore in torch format
  successfully saved checkpoint from iteration       1 to /home/fuyuxin/qwen3pretrain/qwen3-1point7b-mcore
INFO:root:Done!
```

# 数据预处理
数据集处理使用方法如下，请根据实际需要修改以下参数，

Alpaca数据集（train-00000-of-00001-a09b74b3ef9c3b56.parquet）获取，

https://modelers.cn/datasets/AI_Connect/alpaca/tree/main/data

对 data_convert_qwen3_1point7b_pretrain.sh 中 preprocess_data.py 参数说明如下，

| 参数名   | 含义                |
|---------|-----------------|
| --input | 数据集路径  |
| --tokenizer-name-or-path | 模型 tokenizer 目录    |
| --output-prefix | 数据集处理完的输出路径及前缀名  |

```shell
/home/fuyuxin/qwen3pretrain# cd MindSpeed-LLM

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# cp tests/0day/qwen3/qwen3-1.7b/data_convert_qwen3_1point7b_pretrain.sh tests/0day/qwen3/qwen3-1.7b/data_convert_qwen3_1point7b_pretrain.sh.org

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# vim tests/0day/qwen3/qwen3-1.7b/data_convert_qwen3_1point7b_pretrain.sh
# 修改 ascend-toolkit 路径
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# mkdir ./dataset

python ./preprocess_data.py \
    --input /home/fuyuxin/qwen3pretrain/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path /home/fuyuxin/qwen3pretrain/qwen3-1point7b-base \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /home/fuyuxin/qwen3pretrain/dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# bash tests/0day/qwen3/qwen3-1.7b/data_convert_qwen3_1point7b_pretrain.sh

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# ll /home/fuyuxin/qwen3pretrain/dataset/
enwiki_text_document.bin
enwiki_text_document.idx
train-00000-of-00001-a09b74b3ef9c3b56.parquet
```

# 训练
预训练使用方法如下，

需根据实际情况修改脚本中以下变量，

  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

```shell
/home/fuyuxin/qwen3pretrain# cd MindSpeed-LLM

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# cp tests/0day/qwen3/qwen3-1.7b/pretrain_qwen3_1point7b_4K_ptd.sh tests/0day/qwen3/qwen3-1.7b/pretrain_qwen3_1point7b_4K_ptd.sh.org

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# vim tests/0day/qwen3/qwen3-1.7b/pretrain_qwen3_1point7b_4K_ptd.sh
#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
#CKPT_LOAD_DIR=/home/fuyuxin/qwen3pretrain/qwen3-1point7b-mcore
CKPT_LOAD_DIR=
CKPT_SAVE_DIR=/home/fuyuxin/qwen3pretrain/MindSpeed-LLM/ckpt
DATA_PATH=/home/fuyuxin/qwen3pretrain/dataset/enwiki_text_document
TOKENIZER_PATH=/home/fuyuxin/qwen3pretrain/qwen3-1point7b-base

TP=8
PP=1
SEQ_LENGTH=4096
TRAIN_ITERS=2000
ROUTER_BALANCING_TYPE='softmax_topk'

......
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
#    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_mcore_qwen3_1point7b.log

/home/fuyuxin/qwen3pretrain/MindSpeed-LLM# bash tests/0day/qwen3/qwen3-1.7b/pretrain_qwen3_1point7b_4K_ptd.sh
```
在 800I A2(G5680V2 32G\*8) 可运行 Qwen3-1.7B-Base 预训练，上述 pretrain_qwen3_1point7b_4K_ptd.sh 配置为无初始权重开启随机初始化训练；
