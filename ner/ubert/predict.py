
import onnxruntime as ort
import json
import sys
import traceback

import numpy as np
from flask import Flask, request, Response
from transformers import BertTokenizerFast

from ner.ubert import model_encoder, model_decoder

tokenizer = BertTokenizerFast.from_pretrained('pretrained_model', additional_special_tokens=[
    '[unused' + str(i + 1) + ']' for i in range(99)])

# providers = ['CUDAExecutionProvider']
providers = ['CPUExecutionProvider']

# peek_folder(args.parent_path)

ort_sess = ort.InferenceSession('onnx/ubert.onnx', providers=providers,)
print('onnx loaded..')


def infer(inp_texts, if_score=True):
    res = {}
    it0 = time.time()
    inp_batch_data = model_encoder.compose_queries(inp_texts)
    inp_dict = model_encoder.encode(tokenizer, inp_batch_data)
    # inp_dict = get_input(tokenizer, args.max_length, 'cuda', batch_size=1)
    inp_dict_numpy = {k: inp_dict[k].numpy() for k in inp_dict}
    out = ort_sess.run(['output'], input_feed=inp_dict_numpy)
    span_logits = sigmoid(out[0])
    pos = np.argwhere(span_logits > 0.5)
    infer_res = model_decoder.decode(span_logits, pos, inp_batch_data, tokenizer)
    dt = time.time() - it0
    extraction_res = extract_entities(infer_res, if_score)
    res['data'] = extraction_res
    res['cost_time'] = f'{(time.time() - it0) * 1000:.0f}ms'
    res['msg'] = 'query successful'
    res['code'] = 200
    res['version'] = '1.2.0'

    return res


def sigmoid(x):
    x = np.clip(x, -500, 500)
    s = 1 / (1 + np.exp(-x))
    return s


import time


def parse_query():
    if request.method == 'POST':
        input_data = json.loads(request.get_data(parse_form_data=True))
        input_texts = input_data['q']
        prams_score = int(input_data['if_scores'])
    else:
        input_texts = [request.args['q']]
        prams_score = int(request.args['if_scores'])

    if_score = True
    if prams_score is not None and prams_score == 0:
        if_score = False
    return input_texts, if_score


def extract_entities(inp_list: [], if_score=True):
    res_list = []
    for inp in inp_list:
        res = {'query': inp['text']}
        for ent in inp['choices']:
            k = ent['entity_type']
            v_full_list = ent['entity_list']
            v_list = []
            for v in v_full_list:
                if if_score:
                    v_list.append(v)
                else:
                    v_list.append(v['entity_name'])
            res[k] = v_list
        res_list.append(res)
    return res_list


inp_texts = ["马云阿里巴巴这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。"]
res = infer(inp_texts)
print(res)
