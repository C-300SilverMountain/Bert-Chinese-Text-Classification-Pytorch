#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
onnxruntime-gpu Version: 1.13.1
transformers Version: 4.21.3
torch Version: 1.13.0a0+08820cb
'''
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel
from transformers import BertTokenizer, BertModel

# https://blog.csdn.net/zhonglongshen/article/details/132225485
# RoBERTa-wwm-ext:
# https://blog.csdn.net/abc50319/article/details/107903416
# https://blog.51cto.com/u_15127521/4519264
# https://www.biaodianfu.com/chineser-nlp-llm.html

# https://aistudio.baidu.com/projectdetail/5456683
# https://ost.51cto.com/posts/11594
# https://blog.csdn.net/superman_xxx/article/details/130995504
# https://segmentfault.com/a/1190000044425102
# https://blog.csdn.net/qq_43692950/article/details/133768324
# https://www.biaodianfu.com/chineser-nlp-llm.html
# http://www.yxxxx.ac.cn/ch/reader/download_pdf_file.aspx?journal_id=yxxxx&file_name=3618E016C89268EC45CA2F8FA0BBA66081564A44AFFCC5BD5091CA1B1D467EE158454322A8C621694447D0503BBC019C46F2490F365C3CE91CC5A5A613265953&open_type=self&file_no=20221210

model_path = "chinese_roberta_pretrain"
MODEL_ONNX_PATH = "chinese_roberta_pretrain/saved_dict/raw_bert_dynamic.onnx"
OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
model = AutoModel.from_pretrained(model_path)
model.eval()

def batch_embeddings_by_roberta(docs, max_length=300):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    # 对文本进行分词、编码和填充
    input_ids = []
    attention_masks = []
    for doc in docs:
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])


    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    # 提取最后一层的CLS向量作为文本表示
    last_hidden_state = outputs.last_hidden_state
    cls_embeddings = last_hidden_state[:, 0, :]

    return cls_embeddings

def embeddings_by_roberta(doc, max_length=300):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    encoded_dict = tokenizer.encode_plus(
        doc,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # 前向传播
    with torch.no_grad():
        outputs = model(input_id, attention_mask=attention_mask)

    # 提取最后一层的CLS向量作为文本表示
    last_hidden_state = outputs.last_hidden_state
    cls_embeddings = last_hidden_state[:, 0, :]
    return cls_embeddings[0]


def build_args(seq_len):
    org_input_ids = torch.LongTensor([[i for i in range(seq_len)]])
    org_token_type_ids = torch.LongTensor([[1 for i in range(seq_len)]])
    org_input_mask = torch.LongTensor(
        [[0 for i in range(int(seq_len / 2))] + [1 for i in range(seq_len - int(seq_len / 2))]])
    return org_input_ids, org_token_type_ids, org_input_mask


def pytorch_roberta_2_onnx():
    seq_len = 300
    args = build_args(seq_len)
    torch.onnx.export(model,
                      args,
                      MODEL_ONNX_PATH,
                      verbose=True,
                      operator_export_type=OPERATOR_EXPORT_TYPE,
                      opset_version=12,
                      input_names=['input_ids', 'attention_mask', 'token_type_ids'],  # 需要注意顺序！不可随意改变, 否则
                      output_names=['last_hidden_state', 'pooler_output'],  # 需要注意顺序, 否则在推理阶段可能用错output
                      do_constant_folding=True,
                      dynamic_axes={"input_ids": {0: "batch_size", 1: "length"},
                                    "token_type_ids": {0: "batch_size", 1: "length"},
                                    "attention_mask": {0: "batch_size", 1: "length"},
                                    "pooler_output": {0: "batch_size"},
                                    "last_hidden_state": {0: "batch_size"}}
                      )
    print("Export of {} complete!".format(MODEL_ONNX_PATH))


def run_roberta_on_onnx(onnx_path):
    tokenizer = BertTokenizer.from_pretrained("chinese_roberta_pretrain")
    sent = '你好，你叫什么名字'
    tokenized_tokens = tokenizer(sent)
    input_ids = np.array([tokenized_tokens['input_ids']], dtype=np.int64)
    attention_mask = np.array([tokenized_tokens['attention_mask']], dtype=np.int64)
    token_type_ids = np.array([tokenized_tokens['token_type_ids']], dtype=np.int64)

    model = onnx.load(onnx_path)
    sess = ort.InferenceSession(bytes(model.SerializeToString()))
    result = sess.run(
        output_names=None,
        input_feed={"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids}
    )[0]

    last = result[:, 0, :]
    print(last)
    print(len(last[0]))


if __name__ == '__main__':
    print('------')
    # 检测原生roberta是否正常 --- 批量操作
    # res = batch_embeddings_by_roberta(["你好，你叫什么名字"])
    # print(res)
    # print(len(res))
    # print(len(res[0]))

    # 检测原生roberta是否正常 --- 单个操作
    # res = embeddings_by_roberta("你好，你叫什么名字")
    # print(res)
    # print(len(res))


    # 将roberta转换成 onnx 格式
    # pytorch_roberta_2_onnx()

    # 在onnx环境下运行 roberta
    run_roberta_on_onnx('chinese_roberta_pretrain/saved_dict/raw_bert_dynamic.onnx')
