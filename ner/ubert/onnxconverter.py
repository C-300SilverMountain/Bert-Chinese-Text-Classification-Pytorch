import argparse
import sys
from typing import Union, List

from ner.ubert.modeling_cust import UbertPipelines

sys.path.append('/gemini/cai/projs/search-ner-refactor/')
import numpy as np
import onnxruntime as ort
import torch
from transformers import BertTokenizer
from confs import args, conf

import logging
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import MODEL_TYPES
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers import optimizer
from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, IOBinding, OrtValue, SessionOptions

def load_model(ckpt_path):
    args.pretrained_model_path = 'pretrained_model'
    args.load_checkpoints_path = ckpt_path
    pipeline = UbertPipelines(args)
    # tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, additional_special_tokens=[
    #     '[unused' + str(i + 1) + ']' for i in range(99)])
    model = pipeline.model.model
    tokenizer = pipeline.tokenizer
    model = model.eval()
    return model, tokenizer

def compose_query(query_str):
    entity_list = [{"entity_type": col} for col in conf['cols']]
    res_json = {"task_type": "抽取任务", "subtask_type": "实体识别", "text": query_str, "choices": entity_list, "id": 0}
    return res_json

def get_input(tokenizer, max_length, device, batch_size=1):
    # test_data = [{"task_type": "抽取任务", "subtask_type": "实体识别", "text": "李彦宏厦门博乐德平台拍卖有限公司",
    #     "choices": [{"entity_type": "人名"}, {"entity_type": "地名"}, {"entity_type": "公司"}, {"entity_type": "行业"},
    #         {"entity_type": "公司类别"}, {"entity_type": "品牌"}], "id": 0}]
    test_data = [compose_query('彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户')]

    batch_data = test_data * batch_size

    input_ids = []
    attention_mask = []
    token_type_ids = []
    span_labels_masks = []
    item = test_data[0]
    for item in batch_data:
        input_ids0 = []
        attention_mask0 = []
        token_type_ids0 = []
        span_labels_masks0 = []
        for choice in item['choices']:
            texta = item['task_type'] + '[SEP]' + item['subtask_type'] + '[SEP]' + choice['entity_type']
            textb = item['text']
            encode_dict = tokenizer.encode_plus(texta, textb, max_length=max_length, padding='max_length', truncation='longest_first')

            encode_sent = encode_dict['input_ids']
            encode_token_type_ids = encode_dict['token_type_ids']
            encode_attention_mask = encode_dict['attention_mask']
            span_label_mask = np.zeros((max_length, max_length)) - 10000

            if item['task_type'] == '分类任务':
                span_label_mask[0, 0] = 0
            else:
                question_len = len(tokenizer.encode(texta))
                span_label_mask[question_len:, question_len:] = np.zeros((
                    max_length - question_len, max_length - question_len))
            input_ids0.append(encode_sent)
            attention_mask0.append(encode_attention_mask)
            token_type_ids0.append(encode_token_type_ids)
            span_labels_masks0.append(span_label_mask)

        input_ids.append(input_ids0)
        attention_mask.append(attention_mask0)
        token_type_ids.append(token_type_ids0)
        span_labels_masks.append(span_labels_masks0)
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    token_type_ids = torch.tensor(token_type_ids).to(device)
    span_labels_mask = torch.tensor(np.array(span_labels_masks)).to(device)

    return input_ids, attention_mask, token_type_ids, span_labels_mask

def convert(model, tokenizer, max_length=args.max_length, export_file_name=None):
    input_ids, attention_mask, token_type_ids, span_labels_mask = get_input(tokenizer, max_length, 'cpu')
    inp_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
        'span_labels_mask': span_labels_mask}

    torch.onnx.export(model,  # model being run
                      (input_ids, attention_mask, token_type_ids,
                      span_labels_mask),  # model input (or a tuple for multiple inputs)
                      export_file_name,  # where to save the model (can be a file or file-like object)
                      verbose=True,      # 是否以字符串的形式显示计算图
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input_ids', 'attention_mask', 'token_type_ids',
                          'span_labels_mask'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'},
                          'token_type_ids': {0: 'batch_size'}, 'span_labels_mask': {0: 'batch_size'},
                          'output': {0: 'batch_size'}})


def optimize_onnx(
    onnx_path: str,
    onnx_optim_model_path: str,
    fp16: bool,
    use_cuda: bool,
    num_attention_heads: int = 0,
    hidden_size: int = 0,
    architecture: str = "bert",
) -> None:
    """
    ONNX Runtime transformer graph optimization.
    Performs some operator fusion (merge several nodes of the graph in a single one)
    and may convert some nodes to reduced precision.
    :param onnx_path: ONNX input path
    :param onnx_optim_model_path: where to save optimized model
    :param fp16: use mixed precision (faster inference)
    :param use_cuda: perform optimization on GPU (should )
    :param num_attention_heads: number of attention heads of a model (0 -> try to detect)
    :param hidden_size: hidden layer size of a model (0 -> try to detect)
    :param architecture: model architecture to optimize. One of [bert, bart, gpt2]
    """
    optimization_options = FusionOptions(model_type=architecture)
    optimization_options.enable_gelu_approximation = False  # additional optimization
    if architecture == "distilbert":
        optimization_options.enable_embed_layer_norm = False
    if architecture not in MODEL_TYPES:
        logging.info(f"Unknown architecture {architecture} for Onnx Runtime optimizer, overriding with 'bert' value")
        architecture = "bert"
    opt_level = 1 if architecture == "bert" else 0
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=onnx_path,
        model_type=architecture,
        use_gpu=use_cuda,
        opt_level=opt_level,
        num_heads=num_attention_heads,  # automatic detection with 0 may not work with opset 13 or distilbert models
        hidden_size=hidden_size,  # automatic detection with 0
        optimization_options=optimization_options,
    )
    if fp16:
        # use_symbolic_shape_infer set to false because doesn't work after ONNX package v1.10.2
        optimized_model.convert_float_to_float16(use_symbolic_shape_infer=False)  # FP32 -> FP16
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(onnx_optim_model_path)


def create_model_for_provider(
    path: str,
    provider_to_use: Union[str, List],
    nb_threads: int = 0,
    optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    enable_profiling: bool = False,
    log_severity: int = 2,
) -> InferenceSession:
    """
    Create an ONNX Runtime instance.
    :param path: path to ONNX file or serialized to string model
    :param provider_to_use: provider to use for inference
    :param nb_threads: intra_op_num_threads to use. You may want to try different parameters,
        more core does not always provide best performances. 0 let ORT choose the best value.
    :param optimization_level: expected level of ONNX Runtime optimization. For GPU and NLP, extended is the one
        providing kernel fusion of element wise operations. Enable all level is for CPU inference.
        see https://onnxruntime.ai/docs/performance/graph-optimizations.html#layout-optimizations
    :param enable_profiling: let Onnx Runtime log each kernel time.
    :param log_severity: Log severity level. 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal.
    :return: ONNX Runtime inference session
    """
    options = SessionOptions()
    options.graph_optimization_level = optimization_level
    options.enable_profiling = enable_profiling
    options.log_severity_level = log_severity
    if isinstance(provider_to_use, str):
        provider_to_use = [provider_to_use]
    if provider_to_use == ["CPUExecutionProvider"]:
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = nb_threads
    return InferenceSession(path, options, providers=provider_to_use)


if __name__ == '__main__':
    # ckpt_path = ('/gemini/cai/projs/Fengshenbang-LM/fengshen/examples/ubert/ckpt/checkpoint-11-9-1/last'
    #              '.ckpt')
    # ckpt_path = args.load_checkpoints_path
    ckpt_path = 'checkpoint/last.ckpt'
    # onnx_file_name = "/gemini/cai/projs/search-ner-refactor/onnx/model_ckpt/ner.onnx"
    onnx_file_name = 'onnx/ubert.onnx'
    # onnx_optim_file_name = "/gemini/cai/projs/search-ner-refactor/onnx/model_ckpt/ner_opti.onnx"
    # provider_to_use = "CUDAExecutionProvider"
    provider_to_use = "CPUExecutionProvider"

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(ort.get_device())
    print(f'available providers: {ort.get_available_providers()}')
    onnx_optim_file_name = 'onnxopt/opt.onnx'
    model, tokenizer = load_model(ckpt_path)
    convert(model, tokenizer, max_length=args.max_length, export_file_name=onnx_file_name)
    #
    # optimize_onnx(onnx_file_name, onnx_optim_file_name, fp16=False, use_cuda=False)

    input_ids, attention_mask, token_type_ids, span_labels_mask = get_input(tokenizer, args.max_length, 'cpu')
    inp_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
        'span_labels_mask': span_labels_mask}
    inp_dict_numpy = {k: inp_dict[k].cpu().numpy() for k in inp_dict}
    t_session = create_model_for_provider(onnx_file_name, provider_to_use, nb_threads=1)
    # opti_session = create_model_for_provider(onnx_optim_file_name, provider_to_use, nb_threads=1)

    out = t_session.run(['output'], input_feed=inp_dict_numpy)
    print(np.argwhere(out[0] > 0))
    print(out[0].mean())

    # out = opti_session.run(['output'], input_feed=inp_dict_numpy)
    # print(np.argwhere(out[0] > 0))
    # print(out[0].mean())
    #
    # res = model(**inp_dict)
    # print(torch.argwhere(res[0] > 0))
    # print(res[0].mean())