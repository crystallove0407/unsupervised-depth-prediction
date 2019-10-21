import tensorflow as tf
import numpy as np
import configparser
import argparse
import sys
import os

from model.undpflow_model import Undpflow
from test.test_kitti_depth import test_kitti_depth


# manually select one or several free gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# use CPU only
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# autonatically select one free gpu
#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
#os.system('rm tmp')

"""
# Train 3-frame flow network
python3 main.py -c ../config/flow3.ini -t train_flow
nohup python3 main.py -c ../config/flow3.ini -t train_flow &> ../results/training/flow3.txt &
python3 main.py -c ../config/flow5.ini -t train_flow

# Train depth & pose networks
python3 main.py -c ../config/dp3.ini -t train_dp --restore_flow_model=../results/KITTI_RAW_128_416_UnDepthflow_flow_pwc_b8_3frames/checkpoints/kitti_3frames/model-397292
python3 main.py -c ../config/dp5.ini -t train_dp --restore_flow_model=../results/KITTI_RAW_128_416_UnDepthflow_flow_pwc_b8_5frames_full/checkpoints/kitti_5frames/model-130313

# Continuing training
python3 main.py -c ../config/dp5.ini -t train_dp \
    --cont_model=../results/KITTI_RAW_128_416_UnDepthflow_dp_b4_5frames_full/checkpoints/kitti_5frames/model-100241 \
    --restore_flow_model=../results/KITTI_RAW_128_416_UnDepthflow_flow_pwc_b8_5frames_full/checkpoints/kitti_5frames/model-130313

# Test on KITTI dataset
python3 main.py -c ../config/test_dp_kitti.ini -t kitti_eval \
    --restore_dp_model=../results/KITTI_RAW_128_416_UnDepthflow_dp_b4_3frames/checkpoints/kitti_3frames/model-140827

# Evaluate predicted depth maps on KITTI dataset 
python kitti_eval/eval_depth.py --split=eigen --kitti_dir=/home/waterman/dataset/KITTI/ --pred_file=../results/images/kitti/test_kitti.npy
"""

argparser = argparse.ArgumentParser()
argparser.add_argument("-c", "--config", type=str, help="Specify config file", default="./config/config.ini")
argparser.add_argument("-t", "--type", type=str, help="'all', 'dp' or 'flow'", default='all')
argparser.add_argument("--cont_model", type=str, help="Continue training, specify the model to continue training.", default=None)
argparser.add_argument("--restore_flow_model", type=str, help="restore trained flow model into dp model", default=None)
argparser.add_argument("--restore_dp_model", type=str, help="restore trained dp model", default=None)
argparser.add_argument("--retrain", action='store_true', default=False)
argparser.add_argument("--height", type=int, help="Specify the image height", default=256)
argparser.add_argument("--width", type=int, help="Specify the image width", default=832)


def main(_):
    parser = argparser.parse_args()
    config = configparser.ConfigParser()
    config.read(parser.config)

    if parser.type not in ('train_flow', 'train_dp', 'train_all', 'kitti_eval'):
        raise ValueError("Training mode should be one of (train_flow, train_dp, train_all, kitti_eval)")

    if parser.type == 'train_dp':
        assert parser.restore_flow_model != None

    if parser.type == 'kitti_eval':
        assert parser.restore_dp_model != None

        test_kitti_depth(data_list_file=config['test']['data_list_file'],
                         img_dir=config['test']['img_dir'],
                         height=int(config['test']['img_height']),
                         width=int(config['test']['img_width']),
                         restore_dp_model=parser.restore_dp_model,
                         save_dir=config['test']['save_dir'])
    else:
        run_config = config['run']
        dataset_config = config['dataset']

        model = Undpflow(batch_size=int(run_config['batch_size']),
                         iter_steps=int(run_config['iter_steps']),
                         initial_learning_rate=float(run_config['initial_learning_rate']),
                         decay_steps=int(run_config['decay_steps']),
                         decay_rate=float(run_config['decay_rate']),
                         is_scale=bool(run_config['is_scale']),
                         num_input_threads=int(run_config['num_input_threads']),
                         buffer_size=int(run_config['buffer_size']),
                         beta1=float(run_config['beta1']),
                         num_gpus=int(run_config['num_gpus']),
                         num_scales=int(run_config['num_scales']),
                         save_checkpoint_interval=int(run_config['save_checkpoint_interval']),
                         write_summary_interval=int(run_config['write_summary_interval']),
                         display_log_interval=int(run_config['display_log_interval']),
                         allow_soft_placement=bool(run_config['allow_soft_placement']),
                         log_device_placement=bool(run_config['log_device_placement']),
                         regularizer_scale=float(run_config['regularizer_scale']),
                         cpu_device=run_config['cpu_device'],
                         save_dir=run_config['save_dir'],
                         checkpoint_dir=run_config['checkpoint_dir'],
                         model_name=run_config['model_name'],
                         summary_dir=run_config['summary_dir'],
                         dataset_config=dataset_config
                         )

        model.train(train_mode=parser.type,
                    retrain=parser.retrain,
                    cont_model=parser.cont_model,
                    restore_flow_model=parser.restore_flow_model)



if __name__ == '__main__':
    tf.app.run()
