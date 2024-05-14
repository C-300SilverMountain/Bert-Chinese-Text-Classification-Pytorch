# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

# 注释掉，方便编译器运行
# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 注释掉，方便编译器运行
    # model_name = args.model  # bert
    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("took on load data:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
    time_dif = get_time_dif(start_time)
    print("took on train:", time_dif)
    print("训练完成，模型bert.ckpt已保存到：THUCNews/saved_dict目录下")
    print("可以执行bert_to_onnx.py，转换成onnx格式，以便onnxruntime-java部署")