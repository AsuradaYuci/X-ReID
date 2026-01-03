import os.path

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from collections import OrderedDict
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .SP_TPM import TPM_space_shift
from .TP_TPM import TPM_temp_shift



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
#     model_path = clip._download(url)  #
    model_path1 = 'dataset_cc/Pretrain-models/ViT-B-16.pt'  # 不用下载,用下载好的
    model_path2 = 'YCY/Pretrained_models/ViT-B-16.pt'  # 不用下载,用下载好的
    if os.path.exists(model_path1):
        model_path = model_path1
    elif os.path.exists(model_path2):
        model_path = model_path2
    else:
        model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.sie_coe = cfg.MODEL.SIE_COE    # 1

        #### 1.CLIP_backbone fc
        self.classifier_clip_temp = nn.Linear(self.in_planes_proj + self.in_planes, num_classes, bias=False)
        self.classifier_clip_temp.apply(weights_init_classifier)
        self.classifier_clip_frame = nn.Linear(self.in_planes_proj + self.in_planes, self.num_classes, bias=False)
        self.classifier_clip_frame.apply(weights_init_classifier)
        #### 2. CLIP_backbone bn
        self.bn_feat_temp = nn.BatchNorm1d(self.in_planes)
        self.bn_feat_temp.bias.requires_grad_(False)
        self.bn_feat_temp.apply(weights_init_kaiming)

        self.bn_feat_frame = nn.BatchNorm1d(self.in_planes)
        self.bn_feat_frame.bias.requires_grad_(False)
        self.bn_feat_frame.apply(weights_init_kaiming)

        self.bn_proj_temp = nn.BatchNorm1d(self.in_planes_proj)
        self.bn_proj_temp.bias.requires_grad_(False)
        self.bn_proj_temp.apply(weights_init_kaiming)

        self.bn_proj_frame = nn.BatchNorm1d(self.in_planes_proj)
        self.bn_proj_frame.bias.requires_grad_(False)
        self.bn_proj_frame.apply(weights_init_kaiming)

        self.bottleneck_local = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_local.bias.requires_grad_(False)
        self.bottleneck_local.apply(weights_init_kaiming)

        ####### 2.  TSM FC +BN
        self.classifier_TSM_tp0 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_TSM_tp0.apply(weights_init_classifier)
        self.classifier_TSM_frame0 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_TSM_frame0.apply(weights_init_classifier)
        self.bn_TSM_tp0 = nn.BatchNorm1d(self.in_planes)
        self.bn_TSM_tp0.bias.requires_grad_(False)
        self.bn_TSM_tp0.apply(weights_init_kaiming)
        self.bn_TSM_frame0 = nn.BatchNorm1d(self.in_planes)
        self.bn_TSM_frame0.bias.requires_grad_(False)
        self.bn_TSM_frame0.apply(weights_init_kaiming)

        self.classifier_TSM_tp1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_TSM_tp1.apply(weights_init_classifier)
        self.classifier_TSM_frame1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_TSM_frame1.apply(weights_init_classifier)
        self.bn_TSM_tp1 = nn.BatchNorm1d(self.in_planes)
        self.bn_TSM_tp1.bias.requires_grad_(False)
        self.bn_TSM_tp1.apply(weights_init_kaiming)
        self.bn_TSM_frame1 = nn.BatchNorm1d(self.in_planes)
        self.bn_TSM_frame1.bias.requires_grad_(False)
        self.bn_TSM_frame1.apply(weights_init_kaiming)

        ######## 3. CLIP backbone
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        # Trick: freeze patch projection for improved stability
        # https://arxiv.org/pdf/2104.02057.pdf
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        trunc_normal_(self.cv_embed, std=.02)
        print('camera number is : {}'.format(camera_num))

        dataset_name = cfg.DATASETS.NAMES
        self.TSM_space = TPM_space_shift(dim=768, num_heads=12)
        self.TSM_temp = TPM_temp_shift(dim=768, num_heads=12)
        # self.CRM = CRM(dim=768, num_heads=12)
        # self.TMD_mamba = MambaLayer(dim=768)
        # self.norm3_mamba = nn.LayerNorm(768)
        # self.norm4_mamba = nn.LayerNorm(768)
        # self.SSP = ImageSpecificPrompt()
        # self.TAT = TemporalAttentionTransformer(T=cfg.INPUT.SEQ_LEN, embed_dim=512, layers=1)
        # width=768  layers=12  heads=12, droppath=None, use_checkpoint, T=8
        # self.SAT = Transformer_SP(width=512, layers=1, heads=8, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # "meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"
        # self.temppool = visual_prompt(sim_head='LSTM', T=cfg.INPUT.SEQ_LEN)
        # self.TMD = Temporal_Memory_Difusion(width=768, layers=1, heads=12, droppath=None, T=cfg.INPUT.SEQ_LEN)
        # self.ln_post = LayerNorm(512)

    def forward(self, x = None, get_image= False, cam_label= None, eval_all=False):

        B, T, C, H, W = x.shape  # B=64, T=4.C=3 H=256,W=128

        if get_image == True:
            x = x.view(-1, C, H, W)  # 256,3,256,128

            image_features, image_features_proj = self.image_encoder(x)

            if self.model_name == 'ViT-B-16':
                image_features = image_features[:, 0]
                image_features = image_features.view(B, T, -1)  # torch.Size([12, 8, 768])
                image_features = image_features.mean(1)

                img_feature_proj = image_features_proj[:,0]
                img_feature_proj = img_feature_proj.view(B, T, -1)  # torch.Size([64, 4, 512])
                img_feature_proj = img_feature_proj.mean(1)  # torch.Size([64, 512])

                feat = self.bn_feat_temp(image_features)  # torch.Size([16, 768])
                feat_proj = self.bn_proj_temp(img_feature_proj)  # torch.Size([16, 512])
                out_feat = torch.cat([feat, feat_proj], dim=1)  # torch.Size([64, 1280])

                return out_feat

        if self.model_name == 'ViT-B-16':
            x = x.view(-1, C, H, W)  # torch.Size([64, 3, 256, 128])

            cv_embed = self.sie_coe * self.cv_embed[cam_label]

            # cv_embed = cv_embed.repeat((1, B)).view(B, -1)  # torch.Size([64, 768])
            # cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
            if eval_all:
                cv_embed = cv_embed.repeat((1, B)).view(B, -1)  # torch.Size([64, 768])
                cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([64, 768])
            else:
                cv_embed = cv_embed.repeat((1, T)).view(B * T, -1)  # torch.Size([40, 768])
            #
            image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)  # torch.Size([40, 163, 768])
            #  # torch.Size([40, 163, 512])
            #######
            TSM_input = image_features.clone().detach()
            ###################################################
            img_feature = image_features[:, 0]  # torch.Size([64, 768]
            img_feature2 = img_feature.view(B, T, -1)  # torch.Size([4, 10, 768])
            img_feature2 = img_feature2.mean(1)  #  # torch.Size([4, 768])
            #####1.temporal information####
            # f_tp = self.temppool(img_feature2)  # b, 768
            ######
            img_feature_proj = image_features_proj_raw[:, 0]  # torch.Size([64, 512])
            img_feature_proj2 = img_feature_proj.view(B, T, -1)  # torch.Size([4, 10, 512])
            img_feature_proj2 = img_feature_proj2.mean(1)  # torch.Size([16, 512])
            ###################################################

        feat = self.bn_feat_temp(img_feature2)  # torch.Size([16, 768])
        feat_proj = self.bn_proj_temp(img_feature_proj2)  # torch.Size([16, 512])
        feat_local = self.bn_feat_frame(img_feature)
        feat_proj_local = self.bn_proj_frame(img_feature_proj)

        ###### TSM ###### intra_modal
        TSM_out_frame0 = self.TSM_space(TSM_input, B, T)  # BT, 768
        TSM_out_temporal0 = TSM_out_frame0.view(B, T, -1).mean(1)  # B, 768
        TSM_frame0 = self.bn_TSM_frame0(TSM_out_frame0)
        TSM_temporal0 = self.bn_TSM_tp0(TSM_out_temporal0)

#         TSM_out_frame1 = self.TSM_temp(TSM_input, B, T)  # BT, 768
#         TSM_out_temporal1 = TSM_out_frame1.view(B, T, -1).mean(1)  # B, 768
#         TSM_frame1 = self.bn_TSM_frame1(TSM_out_frame1)
#         TSM_temporal1 = self.bn_TSM_tp1(TSM_out_temporal1)

        if self.training:
            ####### 1, CLIP_backbone
            out_feat = torch.cat([feat, feat_proj], dim=1)  # torch.Size([64, 1280])
            clip_score_temp = self.classifier_clip_temp(out_feat)
            x_local = torch.cat((feat_local, feat_proj_local), dim=1)
            clip_score_frame = self.classifier_clip_frame(x_local)
            ###### 2.
            cls_score_tp0 = self.classifier_TSM_tp0(TSM_temporal0)
            cls_score_frame0 = self.classifier_TSM_tp0(TSM_frame0)
#             cls_score_frame0 = self.classifier_TSM_frame0(TSM_frame0)

#             cls_score_tp1 = self.classifier_TSM_tp1(TSM_temporal1)
#             cls_score_frame1 = self.classifier_TSM_frame1(TSM_frame1)
#             cls_score_frame1 = self.classifier_TSM_tp1(TSM_frame1)
            score_list = [clip_score_temp, cls_score_tp0, clip_score_frame, cls_score_frame0]
            feat_list = [img_feature2, img_feature_proj2, TSM_out_temporal0]
            return score_list, feat_list, out_feat

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
#                 out_feat0 = torch.cat([feat, feat_proj, TSM_temporal0, TSM_temporal1], dim=1)  # torch.Size([64, 1280])
#                 # out_feat1 = torch.cat([feat, TSM_temporal], dim=1)  # torch.Size([64, 1280])
#                 return out_feat0, feat, TSM_temporal0, TSM_temporal1
                # return f_tp
                out1 = torch.cat([feat, feat_proj, TSM_temporal0], dim=1)  # torch.Size([64, 1280])
#                 out2 = torch.cat([feat, TSM_temporal0,], dim=1)  # torch.Size([64, 1280])
                out3 = torch.cat([feat, TSM_temporal0], dim=1)  # torch.Size([64, 1280])
#                 out4 = torch.cat([TSM_temporal0, TSM_temporal1], dim=1)
                return out1, feat, out3, TSM_temporal0


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class, camera_num):
    model = build_transformer(num_class, camera_num, cfg)
    return model

