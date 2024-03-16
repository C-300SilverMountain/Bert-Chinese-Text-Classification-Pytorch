import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ner_datasets import NERDataset
from tqdm import tqdm
import json
import time, sys
import numpy as np
from sklearn.metrics import f1_score


def train(epoch, model, device, loader, optimizer, gradient_accumulation_steps):
    model.train()
    time1 = time.time()
    for index, data in enumerate(tqdm(loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        # 反向传播，计算当前梯度
        loss.backward()
        # 梯度累积步数
        if (index % gradient_accumulation_steps == 0 and index != 0) or index == len(loader) - 1:
            # 更新网络参数
            optimizer.step()
            # 清空过往梯度
            optimizer.zero_grad()

        # 100轮打印一次 loss
        if index % 100 == 0 or index == len(loader) - 1:
            time2 = time.time()
            tqdm.write(
                f"{index}, epoch: {epoch} -loss: {str(loss)} ; each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")


def validate(model, device, loader):
    model.eval()
    acc = 0
    f1 = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted_labels = torch.max(outputs.logits, dim=2)
            predicted_labels = predicted_labels.detach().cpu().numpy().tolist()
            true_labels = labels.detach().cpu().numpy().tolist()

            predicted_labels_flat = [label for sublist in predicted_labels for label in sublist]
            true_labels_flat = [label for sublist in true_labels for label in sublist]

            accuracy = (np.array(predicted_labels_flat) == np.array(true_labels_flat)).mean()
            acc = acc + accuracy
            f1score = f1_score(true_labels_flat, predicted_labels_flat, average='macro')
            f1 = f1 + f1score

    return acc / len(loader), f1 / len(loader)


def main():
    labels_path = "./data/labels.json"
    model_name = './pretrained_model'
    train_json_path = "./data/train.json"
    val_json_path = "./data/dev.json"
    max_length = 300
    epochs = 5
    batch_size = 1
    lr = 1e-4
    gradient_accumulation_steps = 16
    model_output_dir = "output"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载label
    with open(labels_path, "r", encoding="utf-8") as r:
        labels_map = json.loads(r.read())

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels_map))
    model.to(device)

    # 加载数据
    print("Start Load Train Data...")
    train_dataset = NERDataset(tokenizer, train_json_path, labels_map, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Start Load Validation Data...")
    val_dataset = NERDataset(tokenizer, val_json_path, labels_map, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print("Start Training...")
    best_acc = 0.0
    for epoch in range(epochs):
        train(epoch, model, device, train_loader, optimizer, gradient_accumulation_steps)
        print("Start Validation...")
        acc, f1 = validate(model, device, val_loader)
        print(f"Validation : acc: {acc} , f1: {f1}")

        if best_acc < acc: # 保存准确率最高的模型
            print("Save Model To ", model_output_dir)
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
            best_acc = acc

if __name__ == '__main__':
    main()

