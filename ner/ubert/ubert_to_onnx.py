
# python: 将bert模型原始格式 转换成 onnx格式(微软)，以便用java加载模型 & 部署
# https://cloud.tencent.com/developer/article/2160621
# 环境准备(cpu)：
# pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install onnx

# 环境准备（GPU）：
# pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
import argparse
import logging
from argparse import Namespace

import torch
from transformers import BertTokenizer, BertModel

from ner.ubert.modeling_ubert import UbertModel, UbertPipelines

logger = logging.getLogger(__name__)

configs = Namespace()
# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.pretrained_model_path = "pretrained_model"

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UbertPipelines.pipelines_args(total_parser)
args = total_parser.parse_args()

# 设置一些训练要使用到的参数
args.pretrained_model_path = 'pretrained_model' #预训练模型的路径，我们提供的预训练模型存放在HuggingFace上
args.default_root_dir = './'  #默认主路径，用来放日志、tensorboard等
args.max_epochs = 5
# args.gpus = 0
args.batch_size = 1

model = UbertPipelines(args)
# model.to(configs.device)
state_dict = torch.load('checkpoint/last.ckpt', map_location=torch.device('cpu'))
model.model.load_state_dict(state_dict['state_dict'])
# model.load_state_dict(torch.load('checkpoint/last.ckpt', map_location=torch.device('cpu'))['state_dict'])


tokenizer = BertTokenizer.from_pretrained('pretrained_model')


def export_onnx_model(args, model, tokenizer, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids': torch.ones(1, 10, 128, dtype=torch.int64),
                  'attention_mask': torch.ones(1, 128, dtype=torch.int64),
                  'token_type_ids': torch.ones(1, 128, dtype=torch.int64),
                  'span_labels': torch.ones(1,10, 128, 128, dtype=torch.float32),
                  'span_labels_mask': torch.ones(10, 128, 128, dtype=torch.float32)}
        # outputs = model(**inputs)

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model.model.model,  # model being run
                          (inputs['input_ids'],  # model input (or a tuple for multiple inputs)
                           inputs['attention_mask'],
                           inputs['token_type_ids'],
                           inputs['span_labels'],
                           inputs['span_labels_mask']),  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input_ids',  # the model's input names
                                       'input_mask',
                                       'segment_ids'],
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                        'input_mask': symbolic_names,
                                        'segment_ids': symbolic_names})
        logger.info("ONNX Model exported to {0}".format(onnx_model_path))


export_onnx_model(configs, model, tokenizer, "onnx/bert.onnx")
