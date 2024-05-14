# 微调：https://cloud.baidu.com/qianfandev/model/49
# 微调：https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md

#生成训练语料
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine --model_name_or_path model/BAAI/bge-base-zh-v1.5 --input_file toy_finetune_data.jsonl --output_file toy_finetune_data_minedHN.jsonl --range_for_sampling 2-200 --negative_number 15 --use_gpu_for_searching

#训练
torchrun --nproc_per_node 1 -m FlagEmbedding.baai_general_embedding.finetune.run --output_dir model --model_name_or_path model/BAAI/bge-base-zh-v1.5 --train_data ./toy_finetune_data_minedHN.jsonl --learning_rate 1e-5 --num_train_epochs 5 --per_device_train_batch_size 1 --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 64 --passage_max_len 256 --train_group_size 1 --negatives_cross_device --logging_steps 10 --save_steps 1000 --query_instruction_for_retrieval ""
