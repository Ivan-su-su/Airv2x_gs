from numpy import record
import torch.nn as nn

from opencood.models.how2comm_modules.how2comm_deformable import How2comm
import torch
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.how2comm_modules.feature_flow import ResNetBEVBackbone
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.task_heads.segmentation_head import BevSegHead 
def transform_feature(feature_list, delay):
    return feature_list[delay]


class Airv2xHow2com(Airv2xBase):
    def __init__(self, args):
        super().__init__(args)
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        self.init_encoders(args)
        modality_args = args["modality_fusion"]
        if 'resnet' in modality_args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(modality_args['base_bev_backbone'], 64)
            self.resnet = True
        else:
            self.backbone = BaseBEVBackbone(modality_args['base_bev_backbone'], 64)
            self.resnet = False
        # used to downsample the feature map for efficient computation
        
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.dcn = False

        self.fusion_net = How2comm(args['fusion_args'], args)
        self.frame = args['fusion_args']['frame']
        self.delay = 1
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]
        self.downsample_rate = args['fusion_args']['downsample_rate']
        
        self.multi_scale = args['fusion_args']['multi_scale']
        # self.outC = args["outC"]
        self.outC = 128 * 2
        self.outC = 64
        self.cls_head = nn.Conv2d(
                self.outC, args["anchor_number"] * args["num_class"], kernel_size=1
            )
        self.reg_head = nn.Conv2d(
                self.outC, 7 * args["anchor_number"], kernel_size=1
            )
        self.obj_head = nn.Conv2d(
                    self.outC, args["anchor_number"], kernel_size=1
                )
        if args['backbone_fix']:
            self.backbone_fix()
        self.time = 0
        self.regroup_feature_list_large = []
    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if "vehicle" in self.collaborators:
            for p in self.veh_model.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_model.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_model.parameters():
                p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        if self.args["task"] == "det":
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False
            if self.args["obj_head"]:
                for p in self.obj_head.parameters():
                    p.requires_grad = False

        elif self.args["task"] == "seg":
            for p in self.seg_head.parameters():
                p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        batch_dict_list = []  
        feature_list = []  
        feature_2d_list = []  
        matrix_list = []
        regroup_feature_list = []  
        output_dict = {}
        batch_output_dict, record_len = self.extract_features(data_dict)
        if self.resnet:
            spatial_features = batch_output_dict["spatial_features"]
            spatial_features_2d = self.backbone(spatial_features)
        else:
            spatial_features = batch_output_dict["spatial_features"]
            batch_output_dict = self.backbone(batch_output_dict)
            spatial_features_2d = batch_output_dict['spatial_features_2d']
        comm_rates = spatial_features.count_nonzero().item()
        pairwise_t_matrix = data_dict["img_pairwise_t_matrix_collab"]
        
        # N, C, H', W'
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor

        matrix_list.append(pairwise_t_matrix)  
        regroup_feature_list.append(self.regroup(
            spatial_features_2d, record_len))  
        self.regroup_feature_list_large.append(
            self.regroup(spatial_features, record_len))

        pairwise_t_matrix = matrix_list[0].clone().detach()  
        
        if self.time == 0:
            history_feature = transform_feature(self.regroup_feature_list_large, 0)
            self.time = 1
        else:
            history_feature = transform_feature(self.regroup_feature_list_large, self.delay)
            
        psm_single = self.cls_head(spatial_features_2d)

        if self.delay == 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(spatial_features, psm_single, record_len,pairwise_t_matrix,self.backbone,[self.shrink_conv, self.cls_head, self.reg_head])
        elif self.delay > 0:
            fused_feature, communication_rates, result_dict, offset_loss, commu_loss, _, _ = self.fusion_net(spatial_features, psm_single,record_len,pairwise_t_matrix,self.backbone,[self.shrink_conv, self.cls_head, self.reg_head], history=history_feature)
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict.update({"psm": psm, "rm": rm})

        if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})

        output_dict.update(
                {"mask": 0, "com": communication_rates, "comm_rate": comm_rates}
            )
        # output_dict = {'psm': psm,
        #                 'rm': rm
        #             }

        # output_dict.update(result_dict)
        
        # output_dict.update({'comm_rate': communication_rates,
        #                     "offset_loss": offset_loss,
        #                     'commu_loss': commu_loss,
        #                     "mask": 0
        #                     })
        return output_dict
