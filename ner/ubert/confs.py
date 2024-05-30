import argparse
import os

run_env = os.getenv('RUN_ENV', 'local')

conf = {}
cols = ['人名', '地名', '公司', '行业', '公司类别', '品牌']
conf['cols'] = cols
version = "1.2.0"
conf['version'] = version

args = argparse.Namespace(pretrained_model_path='/gemini/cai/cache_ner/ubert_test/', output_save_path='./predict.json',
                          load_checkpoints_path='', max_extract_entity_number=1, train=False, threshold=0.5,
                          num_workers=8, batchsize=8, max_length=128, monitor='train_loss', mode='min',
                          checkpoint_path='./checkpoint/', filename='model-{epoch:02d}-{train_loss:.4f}',
                          save_top_k=3, every_n_epochs=1, every_n_train_steps=100, save_weights_only=False,
                          learning_rate=1e-05, weight_decay=0.1, warmup=0.01, num_labels=10, logger=True,
                          checkpoint_callback=None, enable_checkpointing=True, default_root_dir=None,
                          gradient_clip_val=None, gradient_clip_algorithm=None, process_position=0, num_nodes=1,
                          num_processes=1, devices=None, gpus=None, auto_select_gpus=False, tpu_cores=None,
                          ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True,
                          overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=False,
                          accumulate_grad_batches=None, max_epochs=None, min_epochs=None, max_steps=-1,
                          min_steps=None, max_time=None, limit_train_batches=1.0, limit_val_batches=1.0,
                          limit_test_batches=1.0, limit_predict_batches=1.0, val_check_interval=1.0,
                          flush_logs_every_n_steps=None, log_every_n_steps=50, accelerator=None, strategy=None,
                          sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top',
                          weights_save_path=None, num_sanity_val_steps=2, resume_from_checkpoint=None, profiler=None,
                          benchmark=False, deterministic=False, reload_dataloaders_every_n_epochs=0,
                          reload_dataloaders_every_epoch=False, auto_lr_find=False, replace_sampler_ddp=True,
                          detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None,
                          plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False,
                          multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False,
                          terminate_on_nan=None)
# args.threshold_index = 0.5  # 设置位置识别的概率阈值
# args.threshold_entity = 0.5  # 设置实体类型的概率阈值
# args.threshold_event = 0.5  # 设置事件类型的概率阈值
# args.threshold_relation = 0.5  # 设置关系类型的概率阈值
args.threshold = 0.5

# args.load_checkpoints_path = ('/gemini/cai/projs/Fengshenbang-LM/fengshen/examples/ubert/ckpt/checkpoint-11-9-1/last'
#                               '.ckpt')
if run_env == 'local':
    # args.load_checkpoints_path = ('/gemini/cai/projs/Fengshenbang-LM/fengshen/examples/ubert/ckpt/checkpoint-11-17-1-epochs-4'
    #                     '/last.ckpt')
    args.load_checkpoints_path = '/gemini/cai/projs/search-ner-refactor/train/ckpt/checkpoint-12-14-v43-epochs-5/last.ckpt'
    args.pretrained_model_path = '/gemini/cai/cache_dir/ubert'
    # args.onnx_path = "/gemini/cai/projs/search-ner-refactor/opti_ner_batch_16.onnx"
    # args.onnx_path = "/gemini/cai/projs/search-ner-refactor/opti_ner_fp16_11_7.onnx"
    args.onnx_path = "/gemini/cai/projs/search-ner-refactor/onnx/model_ckpt/ner_opti_12_14_v4.onnx"
    args.parent_path = '/gemini/cai/projs/search-ner-refactor'
else:
    args.load_checkpoints_path = '/data/modelfiles/eric/ner_ckpt/11-17-epc4-last.ckpt'
    args.pretrained_model_path = '/data/modelfiles/eric/ubert_pretrain/'
    # args.onnx_path = '/data/modelfiles/eric/opti_ner_batch_16.onnx'
    # args.onnx_path = '/data/modelfiles/eric/opti_ner_fp16_11_7.onnx'
    args.onnx_path = '/data/modelfiles/eric/ner_opti_12_14_v4.onnx'
    args.parent_path = '/data/modelfiles/eric'

args.save_weights_only = False
args.gpus = 0
args.benchmark = False
args.max_length = 50
args.num_labels = len(cols)

