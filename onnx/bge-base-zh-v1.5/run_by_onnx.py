import os.path
# from common import logger
from transformers import AutoTokenizer
import onnxruntime as ort
from typing import List, Union, Any
from tqdm import tqdm
import torch
import numpy as np


class EmbModelEncoder:
    def __init__(self, key: str = None, model_path: str = None, tag: str = None, b_normalize: bool = True):
        self.key = key
        self.model_path = model_path
        self.tag = tag
        # if len(model_path) == 0 or not os.path.exists(self.model_path):
            # logger.error(f'模型文件不存在：${model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        self.ort_session = ort.InferenceSession(f'{model_path}/raw_bert_dynamic.onnx')
        self.b_normalize = b_normalize

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 256, max_length: int = 512) \
            -> Union[object, List[Any]]:
        try:
            r = self.encode_inner(sentences, batch_size, max_length).tolist()
            # logger.info(f'编码长度为 {len(sentences)} : {len(r)}')
            return r
        except Exception as e:
            # logger.error(f'模型编码错误: {sentences}', e)
            return np.array([])

    def encode_inner(self, sentences: Union[str, List[str]], batch_size: int = 256, max_length: int = 512) \
            -> np.ndarray:
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Embeddings", disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            )
            in_dict_numpy = {k: self.to_numpy(inputs[k]) for k in self.input_names}
            ort_out_all = self.ort_session.run(
                None,
                in_dict_numpy,
            )
            ort_out_emb = ort_out_all[0][:, 0]
            if self.b_normalize:
                embeddings = torch.nn.functional.normalize(torch.tensor(ort_out_emb), dim=-1)
            else:
                embeddings = ort_out_emb
            all_embeddings.append(embeddings.numpy())
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


if __name__ == '__main__':
    key = '111'
    model_path = r'model/saved_dict'
    tag = 'tag'
    emb_encoder = EmbModelEncoder(key, model_path, tag)
    one = emb_encoder.encode('样例数据-1')
    print(one)
    # text_list = []
    # for one in range(1, 2):
    #     text_list.append(f'语料附件独守空房睡觉了{one}')
    # # text_list = ['福建省代理费', '放大了']
    # ll = emb_encoder.encode(text_list)
    # print(ll)
