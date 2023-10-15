from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()##建立网络
##建立好的model存储在module_list中
    def forward(self, batch_dict): 
        for cur_module in self.module_list:  ##依次实例化model
            batch_dict = cur_module(batch_dict) #更新每个model的输出，便于传递给下一层
#如果是训练，就计算loss
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
#如果是测试，就进行后处理
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict) ##后处理的主要作用是去掉繁杂的预测框，返回（标准格式的）检测框，置信度，类别存储在pred_dicts中，在传给dataset中
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
###调用每个head的get_loss函数，得到loss和tb_dict    head表示预测   可能有第二阶段的预测
        loss_rpn, tb_dict = self.dense_head.get_loss()
        #若还有第二阶段的预测
        #loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        #若有两阶段的预测，就有两个损失
        #loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
