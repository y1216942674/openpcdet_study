import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    '''
    用于初始化数据集对象。参数包括：
    dataset_cfg: 数据集的配置信息。
    class_names: 类别名称列表。
    training: 表示是否用于训练，默认为 True。
    root_path: 数据集的根路径。
    logger: 日志记录器，用于记录日志信息。
    ext: 数据文件的扩展名，默认为 .bin
    '''
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

##加载检测数据的格式
    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    ##配置网络模型文件路径
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    ##点云数据路径
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    ##用于指定预训练模型的路径
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    ##指定点云数据文件的扩展名
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    ##加载网络模型的配置文件
    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    ##创建数据集对象
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
###创建网络模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    ###从指定文件加载模型参数
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    ###将模型参数加载到GPU上
    model.cuda()
    ###设置模型为评估模式
    model.eval()
    #进行推理和可视化
    ##不会计算梯度
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):   #遍历demo_dataset中的每个数据样本，其中idx表示索引，data_dict包含了当前样本的数据。
            logger.info(f'Visualized sample index: \t{idx + 1}') #通过日志记录器记录当前正在可视化的样本索引。
            data_dict = demo_dataset.collate_batch([data_dict]) #将当前样本的数据字典转换为批次格式，以便输入模型。
            load_data_to_gpu(data_dict) # 将数据移动到 GPU 上，以便进行模型推理
            pred_dicts, _ = model.forward(data_dict) #使用预训练模型对数据进行推理，获得预测的结果。
            ##使用可视化工具库（可能是 V 对象的方法）绘制场景。该方法会绘制点云和预测的边界框，分数以及标签。
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'], ##这是点云数据，通过索引 data_dict['points'] 获取。在这里，通过 [:, 1:] 这种索引方式，只选择了点云中除第一列外的所有列，可能是为了去除点云中的某些特定信息
                ##ref_scores=pred_dicts[0]['pred_scores']: 这是预测的分数
                #ref_labels=pred_dicts[0]['pred_labels']: 这是预测的标签
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            #如果 OPEN3D_FLAG 为 True，则图形会持续显示，如果为 False，则绘制完图形后立即停止执行。这可能是为了在某些情况下，避免图形界面一直保持打开状态。
            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
