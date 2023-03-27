# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ntsnet network wrapper."""
import math
import os
import time
import threading
import numpy as np
from mindspore import ops, load_checkpoint, load_param_into_net, Tensor, nn
from mindspore.ops import functional as F, stop_gradient
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore.common.dtype as mstype

# from src.resnet import resnet50
from src.my_resnet import resnet50
from src.config import config

m_for_scrutinizer = config.m_for_scrutinizer
K = config.topK
input_size = config.input_size
num_classes = config.num_classes
lossLogName = config.lossLogName
batch_size = config.batch_size


def _fc(in_channel, out_channel):
    '''Weight init for dense cell'''
    stdv = 1 / math.sqrt(in_channel)
    weight = Tensor(np.random.uniform(-stdv, stdv, (out_channel, in_channel)).astype(np.float32))
    bias = Tensor(np.random.uniform(-stdv, stdv, (out_channel)).astype(np.float32))
    return nn.Dense(in_channel, out_channel, has_bias=True,
                    weight_init=weight, bias_init=bias).to_float(mstype.float32)


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    stdv = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
    weights = Tensor(np.random.uniform(-stdv, stdv, shape).astype(np.float32))
    shape_bias = (out_channels,)
    biass = Tensor(np.random.uniform(-stdv, stdv, shape_bias).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=input_size):
    """
    generate default anchor

    :param anchors_setting: all information of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
    return center_anchors, edge_anchors, anchor_areas


class Navigator(nn.Cell):
    """Navigator"""

    def __init__(self):
        """Navigator init"""
        super(Navigator, self).__init__()
        self.down1 = _conv(2048, 128, 3, 1, padding=1, pad_mode='pad')
        self.down2 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.down3 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.ReLU = nn.ReLU()
        self.tidy1 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy2 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy3 = _conv(128, 9, 1, 1, padding=0, pad_mode='same')
        self.opConcat = ops.Concat(axis=1)
        self.opReshape = ops.Reshape()

    def construct(self, x):
        """Navigator construct"""
        batch_size = x.shape[0]
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1)
        t2 = self.tidy2(d2)
        t3 = self.tidy3(d3)
        t1 = self.opReshape(t1, (batch_size, -1, 1))
        t2 = self.opReshape(t2, (batch_size, -1, 1))
        t3 = self.opReshape(t3, (batch_size, -1, 1))
        return self.opConcat((t1, t2, t3))


class FeatureEnhanceBlock(nn.Cell):
    def __init__(self):
        super(FeatureEnhanceBlock, self).__init__()
        # 1*1卷积核降维
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=-1)  # axis=-1
        self.opConcat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        self.softmax = ops.Softmax(axis=1)  # dim=1
        self.relu = nn.ReLU()
        self.conv1 = _conv(2048, 128, 3, stride=1, pad_mode='same')
        self.conv2 = _conv(128, 128, 3, stride=2, pad_mode='same')
        self.conv3 = _conv(128, 128, 3, stride=2, pad_mode='same')
        self.downsample = _conv(128, 128, 3, stride=2, pad_mode='same')
        self.tidy1 = _conv(128, 6, 1, stride=1, pad_mode='same')
        self.tidy2 = _conv(128, 6, 1, stride=1, pad_mode='same')
        self.tidy3 = _conv(128, 9, 1, stride=1, pad_mode='same')

    def construct(self, x):  # x : feature bs,2048,14,14
        batch_size = x.shape[0]
        d1 = self.relu(self.conv1(x))
        d2 = self.relu(self.conv2(d1))
        d3 = self.relu(self.conv3(d2))
        d2_1 = self.softmax(self.aap1(d2))  # bs,128,4,1
        e2_1 = d2_1 * d1  # setp1  bs,128,14,14
        d1_final = d1 - e2_1  # setp2 bs,128,14,14
        d2_2 = d2 + self.downsample(e2_1)  # setp3 bs,128,7,7
        d3_1 = self.softmax(self.aap1(d3))  # bs,128,1,1
        e3_1 = d3_1 * d2_2  # step4 bs,128,7,7
        d2_final = d2_2 - e3_1  # step5 bs,128,7,7
        d3_final = d3 + self.downsample(e3_1)  # step6 bs,128,4,4

        t1 = self.tidy1(d1_final)
        t2 = self.tidy2(d2_final)
        t3 = self.tidy3(d3_final)
        t1 = self.opReshape(t1, (batch_size, -1, 1))
        t2 = self.opReshape(t2, (batch_size, -1, 1))
        t3 = self.opReshape(t3, (batch_size, -1, 1))
        output = self.opConcat((t1, t2, t3))
        return output


def rerange(input, index):
    '''

    @param input:需要重新排序的tensor  N,X,W
    @param dim:排序的维度 2
    @param index:排序索引 N,W 需要广播
    @return:排序完的tensor，和输入同维度。
    '''
    opReshape = ops.Reshape()
    index = opReshape(index, (index.shape[0], 1, index.shape[1]))
    index = F.broadcast_to(index, (index.shape[0], input.shape[1], index.shape[2]))
    output = ops.GatherD()(input, 2, index)
    return output


class SearchTransfer(nn.Cell):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        # 1*1卷积核降维
        self.conv_trans = _conv(4096, 2048, 1, stride=1, pad_mode='valid')
        self.flod = _conv(18432, 2048, 1, stride=1, pad_mode='valid')
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.unfold = nn.Unfold(ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='same')
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.batmatmul = ops.BatchMatMul()
        self.argmax = ops.ArgMaxWithValue(axis=1)

    # 反向传播不是forward 而是 construct
    def construct(self, part_ref, part_target):
        '''

        @param part_ref: part_feature_I0
        @param part_target: part_feature_I1
        @return:
        '''
        # 进行unfold
        part_ref_unfold1 = self.unfold(part_ref)  # K=V 维度 N, C*k*k, Hr*Wr
        part_target_unfold = self.unfold(part_target)  # Q 维度 N, C*k*k, H*W
        # 进行reshape,合并最后两个维度,因为mindspore和pytorch的区别
        # K=V   (bs,18432,49)
        part_ref_unfold1 = self.opReshape(part_ref_unfold1,
                                          (part_ref_unfold1.shape[0], part_ref_unfold1.shape[1], -1))
        # Q   (bs,18432,49)
        part_target_unfold = self.opReshape(part_target_unfold,
                                            (part_target_unfold.shape[0], part_target_unfold.shape[1], -1))
        # L2归一化
        part_ref_unfold = self.l2_normalize(part_ref_unfold1)  # N, C*k*k, Hr*Wr
        part_target_unfold = self.l2_normalize(part_target_unfold)  # N, C*k*k, H*W
        # 转置，方便后面相乘 (bs,49,18432)
        part_ref_unfold = self.transpose(part_ref_unfold, (0, 2, 1))  # N,Hr*Wr, C*k*k # 改变顺序
        # 进行相乘,Cross relevance  (bs,49,49)
        R_part = self.batmatmul(part_ref_unfold, part_target_unfold)  # [N, Hr*Wr, H*W]
        # 取最大值 (bs,49) (bs,49)
        max_index, max_value = self.argmax(R_part)  # [N, H*W]  最大值的索引, 最大值
        # (bs,18432,49)
        part_ref_rerang_unflod = rerange(part_ref_unfold1, max_index)
        # 使用卷积代替flod
        part_ref_rerang = self.opReshape(part_ref_rerang_unflod,
                                         (part_ref_rerang_unflod.shape[0], part_ref_rerang_unflod.shape[1], 7, 7))
        part_ref_rerang = self.flod(part_ref_rerang)

        # # 使用mindspore.ops.col2im 实现flod
        # # 为了使用ops.col2im 需要讲维度变为  (bs,2048,18432/2048,7*7)
        # part_ref_rerang_unflod = self.opReshape(part_ref_rerang_unflod,
        #                                         (bs, c, int(part_ref_rerang_unflod.shape[1] / c), -1))
        # output_size = Tensor(input_data=[h, w], dtype=mstype.int32)
        # # part_ref_rerang = ops.col2im(part_ref_rerang, output_size, kernel_size=[3, 3], dilation=[1, 1],
        # #                              padding_value=[1, 1], stride=[1, 1])
        # part_ref_rerang = part_ref_rerang_unflod.col2im(output_size, kernel_size=[3, 3], dilation=[1, 1],
        #                                                 padding_value=[1, 1], stride=[1, 1])
        # part_ref_rerang = self.opReshape(part_ref_rerang,(bs,c,h,w)) # 很奇怪，不reshape的话concat会报错，说维度不一致。
        # V^和part_features_I1融合
        con_res = self.concat_op((part_ref_rerang, part_target))
        # 维度转换 4096->2048 1*1卷积
        part_res = self.conv_trans(con_res)
        # maxvalue生成Mash bs,1,7,7
        mask = self.opReshape(max_value, (max_value.shape[0], 1, part_ref_rerang.shape[2], part_ref_rerang.shape[3]))
        # part_res 和 mask相乘再和part_feature_I1相加
        part_res = part_res * mask
        part_res = part_res + part_target
        return part_res


class ContextBlock(nn.Cell):
    def __init__(self):
        super(ContextBlock, self).__init__()
        # 1*1卷积核降维
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=-1)  # axis=-1

        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        self.aap3 = ops.AdaptiveAvgPool2D(3)
        self.aap5 = ops.AdaptiveAvgPool2D(5)
        self.softmax = ops.Softmax(axis=2)
        self.conv1 = _conv(2048, 8192, 1, stride=1, pad_mode='valid')
        self.layernorm = nn.LayerNorm([8192, 7, 7], begin_norm_axis=1,
                                      begin_params_axis=1)  # begin_norm_axis=1, begin_params_axis=1 这两个参数还没搞清楚
        self.relu = nn.ReLU()
        self.conv2 = _conv(8192, 2048, 1, stride=1, pad_mode='valid')

    # 反向传播不是forward 而是 construct
    def construct(self, x):  # x : part_features_tran
        # psp
        batch, channel, height, width = x.shape
        # (bs*4,2048,1)
        aap1 = self.aap1(x)
        aap1 = self.opReshape(aap1, (aap1.shape[0], aap1.shape[1], -1))
        # (bs*4,2048,9)
        aap3 = self.aap3(x)
        aap3 = self.opReshape(aap3, (aap3.shape[0], aap3.shape[1], -1))
        # (bs*4,2048,25)
        aap5 = self.aap5(x)
        aap5 = self.opReshape(aap5, (aap5.shape[0], aap5.shape[1], -1))
        # (bs*4,2048,35)
        psp_feature = self.concat_op((aap1, aap3, aap5))  # axis = -1
        psp_feature = self.softmax(psp_feature)  # axis =2
        # 将要x  reshape
        input_x = self.opReshape(x, (batch, height * width, channel))
        # x 乘 psp_feature，再乘psp_feature的转置。    x * pf * pf^T
        psp_feature_T = self.opReshape(psp_feature, (psp_feature.shape[0], psp_feature.shape[2], psp_feature.shape[1]))
        # bs*4,49,2048,2048
        context_mask = ops.matmul(psp_feature, psp_feature_T)
        # bs*4,49,2048
        context_mask = ops.matmul(input_x, context_mask)
        # psp_feature转置
        contex = self.opReshape(context_mask, (batch, channel, height, width))
        # channel_add_conv
        # conv2d
        conv_contex = self.conv1(contex)
        conv_contex = self.layernorm(conv_contex)  # layernorm 的两个参数没特别清楚。
        conv_contex = self.relu(conv_contex)
        conv_contex = self.conv2(conv_contex)
        output = x + conv_contex
        return output


class NTS_NET(nn.Cell):
    """Ntsnet"""

    def __init__(self, topK=6, resnet50Path=""):
        """Ntsnet init"""
        super(NTS_NET, self).__init__()
        # feature_extractor = resnet50(1001)
        feature_extractor = resnet50(num_classes=num_classes)
        if resnet50Path != "":
            param_dict = load_checkpoint(resnet50Path)
            param_not_load = load_param_into_net(feature_extractor, param_dict)
            print(param_not_load)
        self.feature_extractor = feature_extractor  # Backbone
        # self.feature_extractor.end_point = _fc(512 * 4, num_classes)
        # self.navigator = Navigator()  # Navigator
        self.topK = topK
        self.num_classes = num_classes
        # self.scrutinizer = _fc(2048 * (m_for_scrutinizer + 1), num_classes)  # Scrutinizer
        self.teacher = _fc(512 * 4, num_classes)  # Teacher
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.Pad_ops = ops.Pad(((0, 0), (0, 0), (self.pad_side, self.pad_side), (self.pad_side, self.pad_side)))
        self.np_edge_anchors = edge_anchors + 224
        self.edge_anchors = Tensor(self.np_edge_anchors, mstype.float32)
        self.opzeros = ops.Zeros()
        self.opones = ops.Ones()
        self.concat_op = ops.Concat(axis=1)
        self.nms = P.NMSWithMask(0.25)
        self.topK_op = ops.TopK(sorted=True)
        self.opReshape = ops.Reshape()
        self.opResizeLinear = ops.ResizeBilinear((224, 224))
        self.transpose = ops.Transpose()
        self.opsCropResize = ops.CropAndResize(method="bilinear_v2")
        self.min_float_num = -65536.0
        self.selected_mask_shape = (1614,)
        self.unchosen_score = Tensor(self.min_float_num * np.ones(self.selected_mask_shape, np.float32),
                                     mstype.float32)
        self.gatherND = ops.GatherNd()
        self.gatherD = ops.GatherD()
        self.gather = ops.Gather()
        self.squeezeop = P.Squeeze()
        self.select = P.Select()
        self.perm = (1, 2, 0)
        self.box_index = self.opzeros(((K,)), mstype.int32)
        self.bigbox_index = self.opzeros(((1,)), mstype.int32)
        # self.crop_size = Tensor([224, 224],mstype.int32) 之前的写法，现在写死
        self.crop_size = (224, 224)
        self.perm2 = (0, 3, 1, 2)
        self.m_for_scrutinizer = m_for_scrutinizer
        self.sortop = ops.Sort(descending=True)
        self.stackop = ops.Stack()
        self.sliceop = ops.Slice()

        # 增加的
        self.myConv1 = _conv(1024, 10 * num_classes, 1, 1, pad_mode='pad', padding=1)
        self.amp = nn.MaxPool2d(30, 1, pad_mode='valid')  # 最大池化 k=30
        self.aap1 = ops.AdaptiveAvgPool2D(1)  # 全局平均池化
        self.classifier2_max = _fc(10 * num_classes, num_classes)
        self.classifier3_concat1 = _fc(2048 + 10 * num_classes, num_classes)
        self.layernorm1 = nn.LayerNorm([2048], begin_norm_axis=1, begin_params_axis=1)
        self.layernorm2 = nn.LayerNorm([10 * num_classes], begin_norm_axis=1, begin_params_axis=1)
        self.opConcat_1 = ops.Concat(axis=1)
        self.opDropout = nn.Dropout(keep_prob=0.8)
        self.classifier4_box = _fc(2048, num_classes)
        self.FeatureEnhanceBlock = FeatureEnhanceBlock()
        self.concat_op_0 = ops.Concat(axis=0)
        self.classifier7_transfer = _fc(2048, num_classes)
        self.layernorm3 = nn.LayerNorm([2048], begin_norm_axis=1, begin_params_axis=1)
        self.classifier8_concat2 = _fc(2048 + 4 * 2048, num_classes)
        self.layernorm4 = nn.LayerNorm([4 * 2048], begin_norm_axis=1, begin_params_axis=1)
        self.opStack = ops.Stack(axis=-1)
        self.opSum = ops.ReduceSum(keep_dims=False)
        self.argmax = ops.ArgMaxWithValue(axis=0, keep_dims=True)
        # context
        self.GlobalContext = ContextBlock()
        # transfer
        self.SearchTransfer1 = SearchTransfer()
        self.SearchTransfer2 = SearchTransfer()
        self.SearchTransfer3 = SearchTransfer()

    def construct(self, x):
        """Ntsnet construct"""
        batch_size = x.shape[0]
        resnet_out, feature_low, rpn_feature, feature = self.feature_extractor(x)
        # logits2_max
        feature_low_1 = stop_gradient(feature_low)  # (bs,1024,28,28)
        x2 = self.myConv1(feature_low_1)  # (bs,10*class_n,30,30)
        x2 = self.amp(x2)  # (bs,10*class_n,1,1)
        x2 = self.opReshape(x2, (batch_size, -1))  # (bs,10*class_n)
        logits2_max = self.classifier2_max(x2)  # (bs,class_n)
        # logits3_concat1
        feature1 = self.layernorm1(stop_gradient(feature))  # (bs,2048)
        feature2 = self.layernorm2(stop_gradient(x2))  # (bs,10*class_n)
        x3 = self.opConcat_1((feature1, feature2))  # (bs,2048+ 10*class_n)
        logits3_concat1 = self.classifier3_concat1(x3)

        # parts
        x_pad = self.Pad_ops(x)
        batch_size = x.shape[0]
        rpn_feature = F.stop_gradient(rpn_feature)
        # rpn_score = self.navigator(rpn_feature)
        rpn_score = self.FeatureEnhanceBlock(rpn_feature)
        edge_anchors = self.edge_anchors
        top_k_info = []
        current_img_for_teachers = []
        # current_img_boxs = []
        for i in range(batch_size):
            # using navigator output as scores to nms anchors
            rpn_score_current_img = self.opReshape(rpn_score[i:i + 1:1, ::], (-1, 1))
            bbox_score = self.squeezeop(rpn_score_current_img)
            bbox_score_sorted, bbox_score_sorted_indices = self.sortop(bbox_score)
            bbox_score_sorted_concat = self.opReshape(bbox_score_sorted, (-1, 1))
            edge_anchors_sorted_concat = self.gatherND(edge_anchors,
                                                       self.opReshape(bbox_score_sorted_indices, (1614, 1)))
            bbox = self.concat_op((edge_anchors_sorted_concat, bbox_score_sorted_concat))
            _, _, selected_mask = self.nms(bbox)
            selected_mask = F.stop_gradient(selected_mask)
            bbox_score = self.squeezeop(bbox_score_sorted_concat)
            scores_using = self.select(selected_mask, bbox_score, self.unchosen_score)
            # select the topk anchors and scores after nms
            _, topK_indices = self.topK_op(scores_using, self.topK)
            topK_indices = self.opReshape(topK_indices, (K, 1))
            bbox_topk = self.gatherND(bbox, topK_indices)
            top_k_info.append(self.opReshape(bbox_topk[::, 4:5:1], (-1,)))
            # crop from x_pad and resize to a fixed size using bilinear
            temp_pad = self.opReshape(x_pad[i:i + 1:1, ::, ::, ::], (3, 896, 896))
            # temp_pad = self.transpose(temp_pad, self.perm)
            tensor_image = self.opReshape(temp_pad, (1,) + temp_pad.shape)
            tensor_box = self.gatherND(edge_anchors_sorted_concat, topK_indices)  # # 按照topk_indices从bbox取出值
            tensor_box = tensor_box / 895
            # 代码错误，使用opsCropResize的话，tensor_image的维度反了，,增加opReshape操作
            tensor_image = self.opReshape(tensor_image, (1, 896, 896, 3))
            current_img_for_teacher = self.opsCropResize(tensor_image, tensor_box, self.box_index, self.crop_size)
            current_img_for_teacher = self.opReshape(current_img_for_teacher, (-1, 3, 224, 224))
            current_img_for_teachers.append(current_img_for_teacher)

            _, tensor_bigbox = self.argmax(tensor_box)
            # current_img_box = self.opsCropResize(tensor_image, Tensor(tensor_bigbox), self.bigbox_index,
            #                                      (448, 448))  # tensor_box(1,4)
            # current_img_box = self.opReshape(current_img_box, (-1, 3, 448, 448))
            # current_img_boxs.append(current_img_box)

        # box logits
        # box_img = self.concat_op_0(current_img_boxs)
        box_img = x
        _, _, rpn_feature, _ = self.feature_extractor(box_img)
        x_box = self.aap1(rpn_feature)
        x_box = self.opDropout(x_box)
        box_feature = x_box
        x_box = self.opReshape(x_box, (batch_size, -1))
        logits4_box = self.classifier4_box(x_box)  # (bs,class_n)

        # part logits
        feature = self.opReshape(feature, (batch_size, 1, -1))
        top_k_info = self.stackop(top_k_info)
        top_k_info = self.opReshape(top_k_info, (batch_size, self.topK))
        current_img_for_teachers = self.stackop(current_img_for_teachers)
        current_img_for_teachers = self.opReshape(current_img_for_teachers, (batch_size * self.topK, 3, 224, 224))
        current_img_for_teachers = F.stop_gradient(current_img_for_teachers)
        # extracor features of topk cropped images
        _, _, feature1_3, pre_teacher_features = self.feature_extractor(current_img_for_teachers)
        pre_teacher_features = self.opReshape(pre_teacher_features, (batch_size, self.topK, 2048))
        # pre_scrutinizer_features = pre_teacher_features[::, 0:self.m_for_scrutinizer:1, ::]
        # pre_scrutinizer_features = self.opReshape(pre_scrutinizer_features, (batch_size, self.m_for_scrutinizer, 2048))
        # pre_scrutinizer_features = self.opReshape(self.concat_op((pre_scrutinizer_features, feature)), (batch_size, -1))
        # using topk cropped images, feed in scrutinzer and teacher, calculate loss
        # scrutinizer_out = self.scrutinizer(pre_scrutinizer_features)
        teacher_out = self.teacher(pre_teacher_features)

        # Logits7_transfer Transfer & GlobalContext
        [_, c, w, h] = feature1_3.shape
        parts_features = self.opReshape(feature1_3, (batch_size, self.topK, c, w, h))  # (bs,topk,2048,7,7)
        part_features_all = parts_features
        part_features_I0 = part_features_all[:, 0, ...]
        part_features_I1 = part_features_all[:, 1, ...]
        part_features_I2 = part_features_all[:, 2, ...]
        part_features_I3 = part_features_all[:, 3, ...]
        S1 = self.SearchTransfer1(part_features_I0, part_features_I1)
        S2 = self.SearchTransfer2(part_features_I0, part_features_I2)
        S3 = self.SearchTransfer3(part_features_I0, part_features_I3)
        # 对part imgs进行特征提取
        parts_features_transfer = self.concat_op_0((part_features_I0, S1, S2, S3))
        #  GlobalContext
        transfer_feature = self.GlobalContext(parts_features_transfer)
        transfer_feature = self.aap1(transfer_feature)  # (bs*topk,2048,1,1)
        transfer_feature1 = self.opReshape(transfer_feature, (batch_size * 4, -1))
        transfer_feature2 = self.opReshape(transfer_feature, (batch_size, -1))
        transfer_feature1 = self.layernorm3(transfer_feature1)
        logits7_transfer = self.classifier7_transfer(transfer_feature1)  # (bs*topk,class_n)

        # Logits8_concat2   Concat Box and Parts loss
        box_feature = self.opReshape(box_feature, (batch_size, -1))
        box_feature = self.layernorm3(box_feature)
        transfer_feature2 = self.layernorm4(transfer_feature2)
        concat_feature = self.opConcat_1((box_feature, transfer_feature2))
        logits8_concat2 = self.classifier8_concat2(concat_feature)

        # logits9_gate
        logits9_gate = self.opStack(
            [stop_gradient(logits3_concat1), stop_gradient(logits4_box),
             stop_gradient(logits8_concat2)])  # (bs,class_n,3)
        logits9_gate = self.opSum(logits9_gate, -1)  # (bs,class_n)
        return resnet_out, teacher_out, top_k_info, logits2_max, logits3_concat1, logits4_box, logits7_transfer,\
               logits8_concat2, logits9_gate
        # (batch_size, 200),(batch_size, 200),(batch_size,6, 200),(batch_size,6)


class WithLossCell(nn.Cell):
    """WithLossCell wrapper for ntsnet"""

    def __init__(self, backbone, loss_fn):
        """WithLossCell init"""
        super(WithLossCell, self).__init__(auto_prefix=True)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.oneTensor = Tensor(1.0, mstype.float32)
        self.zeroTensor = Tensor(0.0, mstype.float32)
        self.opReshape = ops.Reshape()
        self.opOnehot = ops.OneHot()
        self.oplogsoftmax = ops.LogSoftmax()
        self.opZeros = ops.Zeros()
        self.opOnes = ops.Ones()
        self.opRelu = ops.ReLU()
        self.opGatherD = ops.GatherD()
        self.squeezeop = P.Squeeze()
        self.reducesumop = ops.ReduceSum()
        self.oprepeat = ops.repeat_elements
        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.sliceop = ops.Slice()

    def construct(self, image_data, label):
        """WithLossCell construct"""
        batch_size = image_data.shape[0]
        origin_label = label
        labelx = self.opReshape(label, (-1, 1))
        origin_label_repeatk_2D = self.oprepeat(labelx, rep=K, axis=1)
        origin_label_repeatk = self.opReshape(origin_label_repeatk_2D, (-1,))
        origin_label_repeatk_unsqueeze = self.opReshape(origin_label_repeatk_2D, (-1, 1))
        resnet_out, teacher_out, top_k_info, logits2_max, logits3_concat1, logits4_box, logits7_transfer, logits8_concat2, logits9_gate = self._backbone(
            image_data)
        teacher_out = self.opReshape(teacher_out, (batch_size * K, -1))
        log_softmax_teacher_out = -1 * self.oplogsoftmax(teacher_out)
        log_softmax_teacher_out_result = self.opGatherD(log_softmax_teacher_out, 1, origin_label_repeatk_unsqueeze)
        log_softmax_teacher_out_result = self.opReshape(log_softmax_teacher_out_result, (batch_size, K))
        oneHotLabel = self.opOnehot(origin_label, num_classes, self.oneTensor, self.zeroTensor)
        # using resnet_out to calculate resnet_real_out_loss
        resnet_real_out_loss = self._loss_fn(resnet_out, oneHotLabel)
        # using scrutinizer_out to calculate scrutinizer_out_loss
        # scrutinizer_out_loss = self._loss_fn(scrutinizer_out, oneHotLabel)

        # 添加的损失
        loss2_max = self._loss_fn(logits2_max, oneHotLabel)
        loss3_concat1 = self._loss_fn(logits3_concat1, oneHotLabel)
        loss4_box = self._loss_fn(logits4_box, oneHotLabel)

        # using teacher_out and top_k_info to calculate ranking loss
        loss = self.opZeros((), mstype.float32)
        num = top_k_info.shape[0]
        for i in range(K):
            log_softmax_teacher_out_inlabel_unsqueeze = self.opReshape(
                self.sliceop(log_softmax_teacher_out_result, (0, i), (log_softmax_teacher_out_result.shape[0], 1)),
                (-1, 1))
            # log_softmax_teacher_out_inlabel_unsqueeze = self.opReshape(log_softmax_teacher_out_result[::, i:i + 1:1],(-1, 1))
            compareX = log_softmax_teacher_out_result > log_softmax_teacher_out_inlabel_unsqueeze
            pivot = self.opReshape(self.sliceop(top_k_info, (0, i), (top_k_info.shape[0], 1)), (-1, 1))
            # pivot = self.opReshape(top_k_info[::, i:i + 1:1], (-1, 1))
            information = 1 - pivot + top_k_info
            loss_p = information * compareX
            loss_p_temp = self.opRelu(loss_p)
            loss_p = self.reducesumop(loss_p_temp)
            loss += loss_p
        rank_loss = loss / num
        oneHotLabel2 = self.opOnehot(origin_label_repeatk, num_classes, self.oneTensor, self.zeroTensor)
        # using teacher_out to calculate teacher_loss
        teacher_loss = self._loss_fn(teacher_out, oneHotLabel2)

        # loss7_transfer
        origin_label_repeatk_2D_4 = self.oprepeat(labelx, rep=4, axis=1)
        origin_label_repeatk_4 = self.opReshape(origin_label_repeatk_2D_4, (-1,))
        oneHotLabel3 = self.opOnehot(origin_label_repeatk_4, num_classes, self.oneTensor, self.zeroTensor)
        loss7_transfer = self._loss_fn(logits7_transfer, oneHotLabel3)

        loss8_concat2 = self._loss_fn(logits8_concat2, oneHotLabel)
        loss9_gate = self._loss_fn(logits9_gate, oneHotLabel)

        # print(resnet_real_out_loss, rank_loss, teacher_loss, loss2_max, loss3_concat1, loss4_box,loss7_transfer,loss8_concat2,loss9_gate)
        total_loss = resnet_real_out_loss + rank_loss + teacher_loss + loss2_max + loss3_concat1 + loss4_box + loss7_transfer + loss8_concat2 + loss9_gate
        return total_loss

    @property
    def backbone_network(self):
        """WithLossCell backbone"""
        return self._backbone
