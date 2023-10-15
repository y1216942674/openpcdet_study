import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # 配置文件的路径
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    #指定批次大小
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    #用于指定数据加载器的工作进程数
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    #为实验设置一个额外的标签，用于标识不同的实验设置
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    #用于指定要从哪个检查点开始训练，即恢复训练时使用的检查点文件
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    #用于指定预训练模型的路径
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    #用于指定预训练模型的路径
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    #用于指定分布式训练时使用的 TCP 端口
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    # 用于指定分布式训练中的本地排名
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    #可以用于设置额外的配置键值对，可以通过命令行传递多个键值对，例如 --set TEST.FOO 1 TEST.BAR 2
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    # 用于指定等待检查点的最大分钟数
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    #用于指定训练从哪个 epoch 开始
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    #用于指定评估实验的标签，用于标识不同的实验设置
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    # 如果设置为 True，则将对所有检查点进行评估
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    #用于指定要评估的检查点所在的目录
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    # 如果设置为 True，将评估结果保存到文件中
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    #如果设置为 True，将计算推理的时间
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()
    ## 从配置文件中加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    #假设 args.cfg_file 的值是 'cfgs/experiment/model.yaml'，则 cfg.TAG 的值将被设置为 'model'
    cfg.TAG = Path(args.cfg_file).stem
     #若args.cfg_file 的值是 'cfgs/experiment/model.yaml'
#args.cfg_file.split('/')[1:-1] 从列表中选取索引为 1 到倒数第 1 个元素的部分，得到 ['experiment']，这是实验组的子目录名称
#cfg.EXP_GROUP_PATH 将被赋值为 'experiment'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024) #设置随机数生成器的种子为 1024，以确保每次运行代码时都使用相同的随机数序列。
##是否使用了命令行替代参数
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

##单个检查点（模型参数）进行评估
'''
model: 要评估的模型。
test_loader: 用于测试的数据加载器。
args: 包含命令行参数的命名空间，其中包括检查点路径、预训练模型路径等。
eval_output_dir: 评估结果的输出目录。
logger: 日志记录器，用于输出日志信息。
epoch_id: 当前评估的检查点的训练轮数。
dist_test: 是否进行分布式测试，默认为 False。
'''
def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    #从指定的检查点文件加载模型参数，并将其移动到 GPU 上进行评估
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    # start evaluation
    #这个函数用于在一个 epoch 内对模型进行测试，将测试结果保存在指定的输出目录 eval_output_dir 中
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )



'''
获取尚未评估的检查点（模型参数）
ckpt_dir: 检查点文件的存储目录。
ckpt_record_file: 记录已经评估过的检查点的文件路径。
args: 包含命令行参数的命名空间，其中包括起始轮数 start_epoch。
'''

def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None



###用于循环地评估检查点并记录评估结果
def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)




def main():
    args, cfg = parse_config()
##计算每个 CUDA 函数的执行时间，从而计算推理（inference）的时间。
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

##是否使用分布式训练
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True


###确定批次的大小
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus


###输出文件的路径
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

###评估文件的路径
    eval_output_dir = output_dir / 'eval'

##根据不同的情况，评估结果的输出目录将会有所不同。如果没有使用 --eval_all 参数，那么输出目录将包含特定的评估结果目录，
# 每个目录对应一个特定的训练轮次。如果使用了 --eval_all 参数，那么输出目录将包含一个名为 'eval_all_default' 的目录，其中将包含所有训练轮次的评估结果。
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'


##如果有额外的标签，包含额外的标签。
    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    ###创建日志文件
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

#是否在分布式训练模式下进行评估
    if dist_test:
        #将总的批量大小打印到日志中，总批量大小等于每个GPU上的批量大小乘以GPU的数量。
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        #循环遍历了 args 对象中的所有属性和值，对于每个属性和值，这行代码将其格式化为字符串，然后使用 logger.info() 方法记录到日志中。
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    #   每个配置项以 key: value 的形式记录到日志中
    log_config_to_file(cfg, logger=logger)
##根据不同的情况设置变量 ckpt_dir 的值，用于指定检查点文件的存储目录
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

#构建测试集的数据加载器
    '''
    dataset_cfg: 数据集的配置，通常包含数据的路径、类别等信息。
    class_names: 类别名称，用于指定数据集中的物体类别。
    batch_size: 批次大小，决定了每次迭代加载多少个样本。
    dist: 是否进行分布式训练，即是否在多个设备上并行加载数据。
    workers: 数据加载器使用的工作线程数，用于并行加载数据。
    logger: 日志记录器，用于记录日志信息。
    training: 标志，表示是否在训练模式。在这里设置为 False，表示构建测试数据加载器。


    '''
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

#构建网络模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
##不进行梯度计算的情况下对模型进行评估   
    with torch.no_grad():
        if args.eval_all:  #意味着您想要评估所有检查点。在这种情况下，将调用 repeat_eval_ckpt 函数，该函数会循环遍历未评估的检查点，并进行评估
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:   #意味着您想要评估指定的检查点。在这种情况下，将调用 eval_single_ckpt 函数，该函数会对单个检查点进行评估
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
