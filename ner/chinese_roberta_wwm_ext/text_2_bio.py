import json


# 将数据转为  BIO 标注形式
def dimension_label(path, save_path, labels_path=None):
    label_dict = ['O']
    with open(save_path, "a", encoding="utf-8") as w:
        with open(path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                text_label = ['O'] * len(text)
                for label_key in label:  # 遍历实体标签
                    B_label = "B-" + label_key
                    I_label = "I-" + label_key
                    if B_label not in label_dict:
                        label_dict.append(B_label)
                    if I_label not in label_dict:
                        label_dict.append(I_label)
                    label_item = label[label_key]
                    for entity in label_item:  # 遍历实体
                        position = label_item[entity]
                        start = position[0][0]
                        end = position[0][1]
                        text_label[start] = B_label
                        for i in range(start + 1, end + 1):
                            text_label[i] = I_label
                line = {
                    "text": text,
                    "label": text_label
                }
                line = json.dumps(line, ensure_ascii=False)
                w.write(line + "\n")
                w.flush()

    if labels_path:  # 保存 label ，后续训练和预测时使用
        label_map = {}
        for i,label in enumerate(label_dict):
            label_map[label] = i
        with open(labels_path, "w", encoding="utf-8") as w:
            labels = json.dumps(label_map, ensure_ascii=False)
            w.write(labels + "\n")
            w.flush()


if __name__ == '__main__':
    path = "./cluener_public/dev.json"
    save_path = "./data/dev.json"
    dimension_label(path, save_path)

    path = "./cluener_public/train.json"
    save_path = "./data/train.json"
    labels_path = "./data/labels.json"
    dimension_label(path, save_path, labels_path)

