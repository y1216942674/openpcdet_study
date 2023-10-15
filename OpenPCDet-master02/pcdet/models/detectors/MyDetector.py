from .detector3d_template import Detector3DTemplate

## 导入自己定义的模块
class PointRCNN_cls(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # 在框架中增加额外模块cls_head
        self.module_topology = [
            'vfe', 'backbone_3d', 'cls_head', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()