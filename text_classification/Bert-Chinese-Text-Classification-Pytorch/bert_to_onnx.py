import torch
from importlib import import_module

# python: 将bert模型原始格式 转换成 onnx格式(微软)，以便用java加载模型 & 部署
# https://cloud.tencent.com/developer/article/2160621
# 环境准备(cpu)：
# pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install onnx

# 环境准备（GPU）：
# pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config('THUCNews')
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path))


def build_args():
    pad_size = config.pad_size
    ids = torch.LongTensor([[0]*pad_size])
    seq_len = torch.LongTensor([0])
    mask = torch.LongTensor([[0]*pad_size])
    return [ids, seq_len, mask]

if __name__ == '__main__':
    args = build_args()
    torch.onnx.export(model,
                      args,
                      'THUCNews/saved_dict/model.onnx',
                      export_params=True,
                      opset_version=11,
                      input_names=['ids', 'seq_len', 'mask'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'ids': {0: 'batch_size'},  # variable lenght axes
                                    'seq_len': {0: 'batch_size'},
                                    'mask': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    print("转换onnx成功")
