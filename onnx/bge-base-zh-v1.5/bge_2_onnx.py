
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel
from transformers import BertTokenizer, BertModel

# model_path = "model/BAAI/bge-base-zh-v1.5"
model_path ="model/Finetune"
# MODEL_ONNX_PATH = "model/saved_dict/raw_bert_dynamic.onnx"
MODEL_ONNX_PATH = "model/Finetune/raw_bert_dynamic.onnx"
OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
model = AutoModel.from_pretrained(model_path)
model.eval()

def build_args(seq_len):
    org_input_ids = torch.LongTensor([[i for i in range(seq_len)]])
    org_token_type_ids = torch.LongTensor([[1 for i in range(seq_len)]])
    org_input_mask = torch.LongTensor(
        [[0 for i in range(int(seq_len / 2))] + [1 for i in range(seq_len - int(seq_len / 2))]])
    return org_input_ids, org_token_type_ids, org_input_mask

def run_roberta_on_onnx(onnx_path):
    tokenizer = BertTokenizer.from_pretrained("model/BAAI/bge-base-zh-v1.5")
    sent = '样例数据-1'
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
    )
    # https://blog.csdn.net/ECNU_LZJ/article/details/103653133
    last = result[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(torch.tensor(last), dim=-1)

    print(sentence_embeddings)
    print(last)
    # print(len(last[0]))

def pytorch_roberta_2_onnx():
    seq_len = 512
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

if __name__ == '__main__':
    print('------')
    # 将roberta转换成 onnx 格式
    pytorch_roberta_2_onnx()

    # 在onnx环境下运行 roberta
    # run_roberta_on_onnx('model/saved_dict/raw_bert_dynamic.onnx')