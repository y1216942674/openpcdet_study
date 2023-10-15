import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

#该函数用于解析命令行参数和配置文件，以构建配置对象。
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    #从命令行传入的配置文件路径
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    # 批次大小（可选）
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    # 训练轮数（可选）
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    #数据加载器的工作线程数，默认为 4
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    #实验的额外标签
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    #指定的检查点文件路径  检查点就是存储的模型状态，如权重、优化器状态、损失值等   在pytoch中的  .pth文件
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    #预训练模型路径，用于加载预训练模型
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    #分布式训练的启动器选项  可以选择不进行分布式训练（'none'）、使用PyTorch的分布式训练方式（'pytorch'）或使用Slurm集群管理系统的分布式训练方式（'slurm'）。
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    #分布式训练时使用的 TCP 端口号
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    #是否使用同步批归一化 Batch Normalization   主要使用在分布式训练中
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    #是否固定随机种子，使用方法：python train.py --fix_random_seed
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # 检查点保存间隔的轮数 ，通常会在每个 epoch 或一定的训练迭代之后，保存一个检查点。
    # 这样，如果训练中断，可以选择最近的检查点来继续训练。检查点的文件通常包含模型参数、优化器参数、损失值等信息
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    #用于分布式训练的本地 rank
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # 最大保存的检查点数量
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    #是否将所有迭代合并为一个轮次
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')

    ##如果要去重新设置cfg的话，可以在这里使用命令行设置某个参数的值，   如：python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --set OPTIMIZATION.BATCH_SIZE_PER_GPU 2
    #可以使用命令行来重新设置配置文件中的某些参数
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, 
                        help='set extra config keys if needed')
    #最大等待分钟数
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    #开始训练的轮数
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    #评估的检查点轮数
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    #是否将日志保存到文件中
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    #是否使用 tqdm 来记录中间的损失值。
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    #日志迭代间隔
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    #检查点保存时间间隔（秒）
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    # 是否禁用 GPU 统计信息
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    # 是否使用混合精度训练
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

###加载yaml文件下的参数---保存为字典格式在cfg中
    cfg_from_yaml_file(args.cfg_file, cfg)
    #假设 args.cfg_file 的值是 'cfgs/experiment/model.yaml'，则 cfg.TAG 的值将被设置为 'model'   从路径中提取文件名（不包含扩展名）
    cfg.TAG = Path(args.cfg_file).stem
    #若args.cfg_file 的值是 '/path/experiment/config.yaml'
#args.cfg_file.split('/')[1:-1] 从列表中选取索引为 1 到倒数第 1 个元素的部分
#cfg.EXP_GROUP_PATH 将被赋值为 'path/experiment'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    print(f'cfg.EXP_GROUP_PATH: {cfg.EXP_GROUP_PATH}')
#配置文件中获取是否启用混合精度训练的设置。如果配置文件中没有明确设置，则使用默认值 False
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)
####使用set参数重新设置cfg的参数
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    #解析命令行参数
    args, cfg = parse_config()
    #####是否进行多GPU训练
    #若为'none'，则不进行分布式训练
    if args.launcher == 'none':
        dist_train = False #不进行分布式训练
        total_gpus = 1 #GPU数量为1
    else: ##若为'pytorch'或'slurm'，则进行分布式训练
        #函数会返回两个值：total_gpus 表示总的GPU数量，cfg.LOCAL_RANK 表示当前进程的本地GPU编号。然后，设置 dist_train 为 True，表示进行分布式训练。
        #%s 是 args.launcher 的值，即 'pytorch' 或 'slurm'。
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

########确定训练时使用的批量大小（batch_size）和训练轮数（epochs）
    if args.batch_size is None: #若未指定批量大小，则使用配置文件中的批量大小
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else: #若指定了批量大小，则需要确保批量大小能够被 GPU 数量整除
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

#即没有在命令行中指定训练轮数，那么将使用配置文件中的总轮数
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
#############如果使用了随机种子
    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)##将一个特定的随机种子值赋给随机数生成器。这可以确保在每次运行时使用相同的种子，从而使得随机性的结果可复现。
###构建了输出目录的路径，用于存储实验结果、日志和检查点等 
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    print('Output dir : {}'.format(output_dir))
######这行代码定义了检查点（checkpoint）的保存路径  
    ckpt_dir = output_dir / 'ckpt'
#创建输出目录，确保路径存在 ，不存在则创建
    output_dir.mkdir(parents=True, exist_ok=True)
##创建检查点保存的目录
    ckpt_dir.mkdir(parents=True, exist_ok=True)
#datetime.datetime.now().strftime('%Y%m%d-%H%M%S')：这部分代码获取当前日期和时间，并将其格式化为类似于 "年月日-时分秒" 的字符串，例如 "20230801-154236"。
#('train_%s.log' % ...)：这部分代码使用上述格式化后的时间字符串，构建日志文件名，例如 "train_20230801-154236.log"。
#创建一个带有时间戳的日志文件，以记录训练过程中的详细信息。
    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) #生成一个日志文件的路径
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK) ##创建日志记录器  log_file：这是之前定义的日志文件的路径     rank=cfg.LOCAL_RANK：这个参数是可选的，用于指定日志的等级。

    # log to file
    ##日志开始的标志
    logger.info('**********************开始 写入**********************') ##只是写入 不会打印在终端
    #获取当前程序中可见的 GPU 设备列表，如果环境变量 'CUDA_VISIBLE_DEVICES' 没有被设置，就默认将所有的 GPU 设备都设置为可见。
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    #记录了当前程序中可见的 GPU 设备列表
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train: #dist_train--True或者Flase          是否处于分布式训练模式，看上面给的参数
        #它记录分布式模式的训练信息，包括总的批大小（total_gpus * args.batch_size）
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        #如果 dist_train 为 False，说明程序正在以单进程模式进行训练。在这种情况下，它记录单进程模式的训练信息
        logger.info('训练使用的是单GPU')
####将训练过程中的一些关键参数记录到日志文件中，并使用 log_config_to_file 函数将配置信息也记录到日志中        
    for key, val in vars(args).items():   #vars(args) 返回一个包含所有命令行参数及其值的字典
        logger.info('{:16} {}'.format(key, val)) ## 使用格式化字符串将参数名和参数值以规整的格式写入日志。
####将配置信息和配置文件复制到日志文件夹中
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
##创建了一个 TensorBoard 的日志记录器对象 tb_log，用于将训练过程中的信息写入 TensorBoard 日志文件中
#log_dir: 指定 TensorBoard 日志文件的保存路径
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None


###表示正在创建数据加载器、网络和优化器
    logger.info("----------- Create dataloader & network & optimizer -----------")
    ###创建数据加载器
    '''
    dataset_cfg: 数据集的配置，包括数据路径、数据增强设置等。
    class_names: 数据集中的目标类别名称。
    batch_size: 批处理大小。
    dist: 是否进行分布式训练。
    workers: 数据加载器使用的工作线程数。
    logger: 日志记录器，用于记录信息。
    training: 是否用于训练。
    merge_all_iters_to_one_epoch: 是否将所有迭代合并为一个周期。
    total_epochs: 训练的总周期数。
    seed: 随机数种子，用于重现性。


    这个函数会返回三个值：
    train_set: 训练数据集。
    train_loader: 训练数据加载器。
    train_sampler: 训练数据采样器，用于处理数据采样。
    '''
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

###构建模型并将其放置在GPU上进行训练
###build_network: 使用配置文件中的模型配置（cfg.MODEL）以及类别数量和训练数据集来构建模型。这个函数会返回一个模型实例
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    #决定是否使用同步批归一化
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda() #将模型移动到GPU上进行训练

###########构建优化器----优化器负责根据损失函数的梯度信息来更新模型的参数，以使模型逐渐收敛到更好的状态。
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)



    # load checkpoint if it is possible
    #加载预训练模型的检查点
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        ##model.load_params_from_file() 函数的作用是从指定的检查点文件中加载模型的参数
        ##to_cpu: 是否将模型参数加载到 CPU 上。
        ##logger: 日志记录器，用于记录信息。
        ##filename: 检查点文件的路径。
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
####加载检查点（checkpoint）文件以及相应的训练状态，目的是在训练过程中恢复之前的训练状态，
# 以便可以从上次训练中断的地方继续训练。如果有提供检查点文件，将会从该文件恢复状态；如果没有提供，会尝试从检查点目录中找到最新的检查点文件，并从中恢复状态
    if args.ckpt is not None:
        ###若存在，就会加载该检查点文件，并从中恢复模型的参数、优化器状态以及训练状态
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]




    model.train() #将模型设置为训练模式 # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train: #判断检查是否正在进行分布式训练
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    #输出关于模型的信息日志，包括模型名称以及模型中的参数数量
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)


#构建学习率调度器（lr_scheduler）和学习率预热调度器（lr_warmup_scheduler）
    '''
    optimizer: 优化器，即用于更新模型参数的算法，如随机梯度下降（SGD）、Adam 等。
    total_iters_each_epoch: 每个训练轮次中的总迭代次数（batch 数量）。
    total_epochs: 训练的总轮次。
    last_epoch: 上一个训练轮次的索引，用于恢复训练时从上次的轮次继续训练。
    optim_cfg: 包含优化器相关配置的字典，如学习率、学习率衰减等。
    
    '''
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

#####执行模型的训练过程
    '''
    model: 待训练的模型。
    optimizer: 优化器，用于更新模型参数。
    train_loader: 训练数据加载器，用于获取训练数据批次。
    model_func: 模型函数的修饰器，可能是对模型的额外操作。
    lr_scheduler: 学习率调度器，用于自动调整学习率。
    optim_cfg: 优化器的配置。
    start_epoch: 起始训练轮次。
    total_epochs: 总训练轮次。
    start_iter: 起始迭代次数（batch 迭代）。
    rank: 用于分布式训练的本地 rank。
    tb_log: TensorBoard 的日志记录器。
    ckpt_save_dir: 模型检查点保存的目录。
    train_sampler: 训练数据采样器。
    lr_warmup_scheduler: 学习率预热调度器。
    ckpt_save_interval: 模型检查点保存的间隔轮次。
    max_ckpt_save_num: 最大保存的模型检查点数量。
    merge_all_iters_to_one_epoch: 是否将所有迭代合并为一个轮次。
    logger: 日志记录器，用于记录训练过程信息。
    logger_iter_interval: 训练迭代时日志记录的间隔。
    ckpt_save_time_interval: 模型检查点保存的时间间隔。
    use_logger_to_record: 是否使用日志记录器来记录信息。
    show_gpu_stat: 是否显示 GPU 使用情况。
    use_amp: 是否使用混合精度训练。
    cfg: 包含配置信息的字典。
    
    '''
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

###检查训练数据集是否使用了共享内存,并在使用的情况下清理共享内存
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()
###训练已结束，输出日志信息--------实验组路径、标签和额外标记
    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
####评估开始
    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
##构建用于测试/评估的数据加载器    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    ##定义了评估输出结果的目录路径
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    ##确保评估输出目录存在，如果目录不存在，则创建
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    ##只评估最后 args.num_epochs_to_eval 轮的结果
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

###目的是在指定的检查点上进行多次评估，以获得更稳定的结果
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
