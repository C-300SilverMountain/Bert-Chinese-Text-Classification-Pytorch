import numpy as np
import torch
from transformers import BertTokenizerFast

from .confs import conf, args


def compose_queries(query_str_list: [str]):
    res_json_list = []
    for query_str in query_str_list:
        res_json = {"task_type": "抽取任务", "subtask_type": "实体识别", "text": query_str,
            "choices": [{"entity_type": col} for col in conf['cols']], "id": 0}
        res_json_list.append(res_json)
    return res_json_list


def encode(tokenizer: BertTokenizerFast, inp_batch_data: [str]) -> dict:
    # test_data = [{"task_type": "抽取任务", "subtask_type": "实体识别", "text": "李彦宏厦门博乐德平台拍卖有限公司",
    #     "choices": [{"entity_type": "人名"}, {"entity_type": "地名"}, {"entity_type": "公司"}, {"entity_type": "行业"},
    #         {"entity_type": "公司类别"}, {"entity_type": "品牌"}], "id": 0}]

    # batch_data = test_data * batch_size
    #  = compose_queries(query_str_list)
    max_length = args.max_length
    batch_data = inp_batch_data

    input_ids = []
    attention_mask = []
    token_type_ids = []
    span_labels_masks = []
    for item in batch_data:
        input_ids0 = []
        attention_mask0 = []
        token_type_ids0 = []
        span_labels_masks0 = []
        for choice in item['choices']:
            texta = item['task_type'] + '[SEP]' + item['subtask_type'] + '[SEP]' + choice['entity_type']
            textb = item['text']
            encode_dict = tokenizer.encode_plus(texta, textb, max_length=max_length, padding='max_length',
                                                truncation='longest_first')

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

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    span_labels_masks = torch.tensor(np.array(span_labels_masks))
    inp = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
        'span_labels_mask': span_labels_masks}
    return inp

# def encode_to_tensor(tokenizer: BertTokenizerFast, device, query_str_list: [str]):
#     inp = encode(tokenizer, query_str_list)
#     for k in inp:
#         inp[k] = inp[k].to(device)
#     return inp


# def encode_to_numpy(tokenizer: BertTokenizerFast, query_str_list: [str]):
#
#
