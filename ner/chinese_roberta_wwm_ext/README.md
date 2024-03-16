# 参考文献
1、https://www.biaodianfu.com/chineser-nlp-llm.html
2、https://blog.csdn.net/qq_43692950/article/details/133768324

# 使用指南
1. [下载数据集](https://www.cluebenchmarks.com/introduce.html)
参考cluener_public目录下的README.md指南下载

2. [下载chinese-roberta-wwm-ext预训练大模型](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
参考pretrained_model目录下的README.md指南下载

3. 将人工标注的数据转换成模型能识别的格式
执行text_2_bio.py，得到data目录，该目录生成三个文件dev.json、labels.json、train.json

4. 执行训练
执行train.py

5. 执行预测
执行entity_predict.py
输入案例：
请输入：根据北京市住房和城乡建设委员会总体工作部署，市建委调配给东城区118套房源，99户家庭全部来到现场
{'government': ['北京市住房和城乡建设委员会', '市建委']}

请输入：为星际争霸2冠军颁奖的嘉宾是来自上海新闻出版局副局长陈丽女士。最后，为魔兽争霸3项目冠军
{'game': ['星际争霸2'], 'position': ['上海新闻出版局'], 'name': ['副局长', '陈丽']}

请输入：作出对成钢违纪辞退处理决定，并开具了退工单。今年8月，公安机关以不应当追究刑事责任为由
{'company': ['成钢'], 'government': ['公安机关']}
