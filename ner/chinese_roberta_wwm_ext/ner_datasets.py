from torch.utils.data import Dataset, DataLoader
import torch
import json

# 构建数据集
class NERDataset(Dataset):
    def __init__(self, tokenizer, file_path, labels_map, max_length=300):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_map = labels_map

        self.text_data = []
        self.label_data = []
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                self.text_data.append(text)
                self.label_data.append(label)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        labels = self.label_data[idx]

        # 使用分词器对句子进行处理
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # 将标签转换为数字编码
        label_ids = [self.labels_map[l] for l in labels]

        if len(label_ids) > self.max_length:
            label_ids = label_ids[0:self.max_length]

        if len(label_ids) < self.max_length:
            # 标签填充到最大长度
            label_ids.extend([0] * (self.max_length - len(label_ids)))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.LongTensor(label_ids)
        }

