import torch
import torch.nn as nn
import math
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, distance2bbox, reduce_mean
from mmrotate.core import build_bbox_coder,multiclass_nms_rotated

from mmdet.models import HEADS, build_loss

from mmcv.cnn import Scale
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

from torch.nn import functional as F

# from ..builder import ROTATED_HEADS
# from .rotated_anchor_head import RotatedAnchorHead

INF = 100000000


@HEADS.register_module()
class RotatedCenterNetHead_only(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 num_classes,
                 norm,
                 num_features,
                 num_cls_convs,
                 num_box_convs,
                 num_share_convs,
                 use_deformable,
                 bbox_coder,
                 only_proposal,
                 loss_center_heatmap=dict(
                     type='CustomGaussianFocalLoss',
                     alpha=0.25,
                     ignore_high_fp=0.85,
                     loss_weight=0.5),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_angle=dict(type='SmoothFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.2),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(RotatedCenterNetHead_only, self).__init__(init_cfg)
        self.out_kernel = 3
        self.norm = norm
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.only_proposal = only_proposal
        self.strides =  [8,16,32,64,128]#[4,8,16,32,64]## [8, 16, 32, 64, 128]
        # self.strides = [8, 16, 32]
        self.hm_min_overlap = 0.95
        self.delta = (1-self.hm_min_overlap)/(1+self.hm_min_overlap)
        self.sizes_of_interest = [[0, 80], [64, 160],
                                  [128, 320], [256, 640], [512, 10000000]]
        # self.sizes_of_interest = [[0, 80], [64, 160],
        #                           [128, 320]]
        self.min_radius = 4
        self.with_agn_hm = True
        self.not_norm_reg = True

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_angle = build_loss(loss_angle)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        self.center_nms = False
        self.pre_nms_topk_train = 4000
        self.pre_nms_topk_test = 1000
        self.nms_thresh_train = 0.9
        self.nms_thresh_test = 0.9
        self.post_nms_topk_train = 2000
        self.post_nms_topk_test = 256
        self.more_pos_topk = 9
        self.score_thresh = 0.0001
        self.not_nms = False

        self.head_configs = {"cls": (num_cls_convs, use_deformable),
                             "bbox": (num_box_convs, use_deformable),
                             "share": (num_share_convs, use_deformable)}
        self.channels = {
            'cls': in_channel,
            'bbox': in_channel,
            'share': in_channel,
        }
        h_bbox_coder = dict(type='DistancePointBBoxCoder')
        self.h_bbox_coder = build_bbox_coder(h_bbox_coder)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self._init_layers()

    def _init_layers(self):
        self._build_tower(self.head_configs, self.channels)
        self.scales = nn.ModuleList(
            [Scale(scale=1.0) for _ in range(self.num_features)])
        self.agn_hm = self._build_head(self.in_channel, 1)
        self.bbox_pred = self._build_head(self.in_channel, 4)
        self.angle_pred = self._build_head(self.in_channel, 1)

        # add angle prediction in bbox_pred
        #self.angle_pred = self._build_head(self.in_channel, 1)


    def _build_head(self, in_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Conv2d(
            in_channel, out_channel, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )
        return layer

    def _build_tower(self, head_configs, channels):
        # init the     1.<cls_tower>    2.<bbox_tower>     3.<share_tower>
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                    channel,
                    channel,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if self.norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

    def init_weights(self):
        """Initialize weights of the head."""
        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
            self.angle_pred,
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.agn_hm.bias, bias_value)
        torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        torch.nn.init.constant_(self.angle_pred.bias, bias_value)
        torch.nn.init.normal_(self.angle_pred.weight, std=0.01)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls (List[Tensor]): cls predict for
                all levels, the channels number is num_classes.
            bbox_reg (List[Tensor]): bbox_reg predicts for all levels,
                the channels number is 4.
            agn_hms (List[Tensor]): agn_hms predicts for all levels,
                the channels number is 1.
        """
        return multi_apply(self.forward_single, feats,
                           [i for i in range(len(feats))])

    def forward_single(self, feat, i):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            cls(Tensor): cls predicts, the channels number is class number: 80
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1

        """
        feat = self.share_tower(feat)       # not used
        cls_tower = self.cls_tower(feat)    # not used
        bbox_tower = self.bbox_tower(feat)

        if not self.only_proposal:
            clss = self.cls_logits(cls_tower)
        else:
            clss = None
        agn_hms = self.agn_hm(bbox_tower)
        angle_pred = self.angle_pred(bbox_tower)

        reg = self.bbox_pred(bbox_tower)
        reg = self.scales[i](reg)
        bbox_reg = F.relu(reg)

        return clss, bbox_reg, agn_hms, angle_pred

    def forward_single_test(self, feat, i):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            cls(Tensor): cls predicts, the channels number is class number: 80
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1

        """

        feat = self.share_tower(feat)       # not used
        cls_tower = self.cls_tower(feat)    # not used
        bbox_tower = self.bbox_tower(feat)

        if not self.only_proposal:
            clss = self.cls_logits(cls_tower)
        else:
            clss = None
        agn_hms = self.agn_hm(bbox_tower)
        angle_pred = self.angle_pred(bbox_tower)
        reg = self.bbox_pred(bbox_tower)
        reg = self.scales[i](reg)
        bbox_reg = F.relu(reg)
        return clss, bbox_reg, agn_hms, angle_pred

    def loss(self,
              cls_scor_per_level,
              reg_pred_per_level,
              agn_hm_pred_per_level,
              angle_pred_per_levle,
              gt_bboxes,
              img_metas,
              gt_bboxes_ignore=None):
        """Compute losses of the dense head.
            gt_labels
        Args:
            agn_hm_pred_per_level (list[Tensor]): center predict heatmaps for
               all levels with shape (B, 1, H, W).
            reg_pred_per_level (list[Tensor]): reg predicts for all levels with
               shape (B, 5, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [tl_x, tl_y, br_x, br_y,tl_angle] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_centernet_loc (Tensor): loss of center heatmap.
                - loss_centernet_agn_pos (Tensor): loss of
                - loss_centernet_agn_neg (Tensor): loss of.
        """
        # --------------------------------------------------#
        # grids->list, len(grids)==5, grids[0]==>[13600,2]
        # grids[1] == > [32400, 2],grids[2]==>[850,2],grids[3]==>[221,2],
        # grids[4]==>[63,2]
        # --------------------------------------------------#
        grids = self.compute_grids(agn_hm_pred_per_level)
        shapes_per_level = grids[0].new_tensor(
            [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])

        # --------------------------------------------------------------#
        # pos_inds->[21], 中心点在batch中每个特征中的索引位置
        # reg_targets->[43750,4], 中心点偏移量
        # flattened_hms->[43750,1]， 置信度
        # --------------------------------------------------------------#
        pos_inds, reg_targets, flattened_hms, reg_angle = \
            self.get_targets2(
                grids, shapes_per_level, gt_bboxes)

        # --------------------------------------------------------------#
        # reg_pred, agn_hm_pred， 将每个特征图和bach上的预测值展平
        # --------------------------------------------------------------#
        reg_pred, agn_hm_pred, angle_pred = self._flatten_outputs(
            reg_pred_per_level, agn_hm_pred_per_level, angle_pred_per_levle)

        flatten_points = torch.cat(
            [points.repeat(len(img_metas), 1) for points in grids])

        losses = self.compute_losses(
            pos_inds, reg_targets, flattened_hms,
            reg_pred, agn_hm_pred, reg_angle, angle_pred, flatten_points)

        return losses

    def compute_losses(self, pos_inds, reg_targets, flattened_hms,
                       reg_pred, agn_hm_pred, reg_angle, pred_angle, flatten_points):
        '''
        Inputs:
            pos_inds: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            reg_angle:   M x 1
            pred_angle:  M x 1
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes

        '''
        assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = torch.tensor(
            len(pos_inds), dtype=torch.float, device=reg_pred[0].device)
        num_pos_avg = max(reduce_mean(num_pos_local), 1.0)

        losses = {}
        #  reg_target:[M, 5], reg_inds:[M]
        #  reg_targets.max(dim=1)[0]: the max value in every line, >0 means in bbox
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_pred = reg_pred[reg_inds]
        reg_targets_pos = reg_targets[reg_inds]

        # flatten_points_pos: [M,2]
        flatten_points_pos = flatten_points[reg_inds]  # added by mmz

        # flatten_hms: [M,class]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        reg_norm = max(reduce_mean(reg_weight_map.sum()).item(), 1.0)

        # pos_angle_preds = pred_anle[reg_inds]
        # pos_angle_targets = reg_angle[reg_inds]

        # added by mmz
        pos_decoded_bbox_preds = self.h_bbox_coder.decode(flatten_points_pos, reg_pred)
        pos_decoded_target_preds = self.h_bbox_coder.decode(
            flatten_points_pos, reg_targets_pos)
        reg_loss = self.loss_bbox(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds,
            weight=reg_weight_map,
            avg_factor=reg_norm)
        cat_agn_heatmap = flattened_hms.max(dim=1)[0]  # M
        agn_heatmap_loss = self.loss_center_heatmap(
            agn_hm_pred,
            cat_agn_heatmap,
            pos_inds,
            avg_factor=num_pos_avg
        )

        pred_angle = pred_angle[reg_inds]
        reg_angle = reg_angle[reg_inds]
        loss_angle = self.loss_angle(
            pred_angle,
            reg_angle.squeeze(),
            avg_factor=num_pos_avg)

        losses['loss_centernet_loc'] = reg_loss
        losses['loss_centernet_heatmap'] = agn_heatmap_loss
        losses['loss_centernet_angel'] = loss_angle

        return losses


    def compute_grids(self, agn_hm_pred_per_level):
        grids = []
        for level, agn_hm_pred in enumerate(agn_hm_pred_per_level):
            h, w = agn_hm_pred.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32, device=agn_hm_pred.device)
            shifts_y = torch.arange(
                0, h * self.strides[level],
                step=self.strides[level],
                dtype=torch.float32, device=agn_hm_pred.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)+self.strides[level] // 2
            shift_y = shift_y.reshape(-1)+self.strides[level] // 2
            #ga = shifts_x.new_zeros(shift_x.size())
            grids_per_level = torch.stack((shift_x, shift_y), dim=1)

            grids.append(grids_per_level)
        return grids

    def get_targets2(self, grids, shapes_per_level, gt_bboxes):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

        Retuen:
            pos_inds: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        '''

        # get positive pixel index
        pos_inds = self._get_label_inds(gt_bboxes, shapes_per_level)
        # pos_inds = None
        heatmap_channels = self.num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l_]) * self.strides[l_]
            for l_ in range(L)]).float()  # M
        reg_size_ranges = \
            torch.cat(
                [shapes_per_level.new_tensor(self.sizes_of_interest[l_]).float().view(1, 2).expand(num_loc_list[l_], 2)
                 for l_ in range(L)])  # M x 2
        grids = torch.cat(grids, dim=0)  # M x 3  x,y,theta
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        reg_angles = []
        # gt_bboxes:list[tensor] list: num_images, tensor: n target
        for i in range(len(gt_bboxes)):  # images
            #
            boxes = gt_bboxes[i]  # N x 5, c_x,c_y,w,h,angle
            area = boxes[:, 2] * boxes[:, 3]
            N = boxes.shape[0]
            gt_ctr, gt_wh, gt_angle = torch.split(boxes[None].expand(M, N, 5), [2, 2, 1], dim=2)
            cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
            rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                                   dim=-1).reshape(M, N, 2, 2)
            # grid [M,2], gt_ctr[M,N,2], offset:[M,N,2]
            offset = grids.view(M, 1, 2) - gt_ctr.view(M, N, 2)
            offset = torch.matmul(rot_matrix, offset[..., None])
            offset = offset.squeeze(-1)
            w, h = gt_wh[..., 0], gt_wh[..., 1]
            offset_x, offset_y = offset[..., 0], offset[..., 1]
            left = w / 2 + offset_x
            right = w / 2 - offset_x
            top = h / 2 + offset_y
            bottom = h / 2 - offset_y
            reg_target = torch.stack((left, top, right, bottom), -1)

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - self.INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if self.only_proposal else heatmap_channels)))
                continue
            centers = boxes[:, (0, 1)]  # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # MxNx2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() *
                               strides_expanded).float() + strides_expanded / 2

            # grid: [M,3]
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) -
                         centers_discret) ** 2).sum(dim=2) == 0)  # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0  # M x N
            is_center3x3 = self.get_center3x3(
                grids, centers, strides) & is_in_boxes  # M x N
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges)  # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level  # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) -
                      centers_expanded) ** 2).sum(dim=2)  # M x N
            dist2[is_peak] = 0
            radius2 = self.delta ** 2 * 2 * area  # N
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N)  # M x N
            reg_target, reg_angle = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, gt_angle)  # M x 5

            flattened_hm = self._create_agn_heatmaps_from_dist(
                weighted_dist2.clone())  # M x 1

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)
            reg_angles.append(reg_angle)

        # transpose im first training_targets to level first ones
        # reg_targets = self._transpose(reg_targets, num_loc_list)

        #     This function is used to transpose image first training targets to
        #         level first ones
        for im_i in range(len(reg_targets)):
            reg_targets[im_i] = torch.split(
                reg_targets[im_i], num_loc_list, dim=0)

        targets_level_first = []
        for targets_per_level in zip(*reg_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0))
        reg_targets = targets_level_first

        # flattened_hms = self._transpose(flattened_hms, num_loc_list)
        for im_i in range(len(flattened_hms)):
            flattened_hms[im_i] = torch.split(
                flattened_hms[im_i], num_loc_list, dim=0)

        hms_level_first = []
        for hms_per_level in zip(*flattened_hms):
            hms_level_first.append(
                torch.cat(hms_per_level, dim=0))
        flattened_hms = hms_level_first

        for im_i in range(len(reg_angles)):
            reg_angles[im_i] = torch.split(
                reg_angles[im_i], num_loc_list, dim=0)
        angle_level_first = []
        for angle_per_level in zip(*reg_angles):
            angle_level_first.append(
                torch.cat(angle_per_level, dim=0))
        reg_angles = angle_level_first

        for i in range(len(reg_targets)):
            reg_targets[i] = reg_targets[i] / float(self.strides[i])

        reg_targets = torch.cat([x for x in reg_targets], dim=0)      # MB x 5
        flattened_hms = torch.cat([x for x in flattened_hms], dim=0)  # MB x C
        reg_angles = torch.cat([x for x in reg_angles], dim=0)        # MB x 1

        return pos_inds, reg_targets, flattened_hms, reg_angles

    def _get_label_inds(self, gt_bboxes, shapes_per_level):
        '''
        Inputs:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N',,bbox 的中心点坐标在2个batch,每个bache 5 个特征图中的index
        '''
        pos_inds = []

        # ---------------------------------#
        # strides: [8,16,32,64,128]
        # B ：n 特征图的数量
        # L :5,
        # --------------------------------#
        L = len(self.strides)
        B = len(gt_bboxes)

        # ---------------------------------#
        # shapes_per_level: L x 2 [(h_l, w_l)]_L
        # --------------------------------#
        shapes_per_level = shapes_per_level.long()

        # ---------------------------------#
        # 长*宽,shapes_per_level:[L,2]
        # loc_per_level:[L,1]
        # --------------------------------#
        loc_per_level = (shapes_per_level[:, 0]
                         * shapes_per_level[:, 1]).long()
        level_bases = []
        s = 0

        # ----------------------------------------------------#
        # s1: B*长*宽, s2: s1+B*长*宽, s3: s2+B*长*宽
        # level_bases [s1,s2,s3,s4,s5]
        #  strides_default: [8,16,32,64,128]
        # ---------------------------------------------------#
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long()
        strides_default = shapes_per_level.new_tensor(self.strides).float()

        # ---------------------------------#
        # B: boxs的数量
        # --------------------------------#
        for im_i in range(B):
            bboxes = gt_bboxes[im_i]  # n x 5
            n = bboxes.shape[0]
            centers = bboxes[:, (0, 1)]  # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            # -------------------------------------------------------------#
            # centers->[4,5,2]  四个点在5个特征图上
            # strides_default -> [5]
            # strides->[4,5,2], 每个特征图上的尺度
            # -------------------------------------------------------------#
            strides = strides_default.view(1, L, 1).expand(n, L, 2)

            # -------------------------------------------------------------#
            # 中心点除以不长等于中心点在每个特征图上的坐标
            # -------------------------------------------------------------#
            centers_inds = (centers / strides).long()  # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                      im_i * loc_per_level.view(1, L).expand(n, L) + \
                      centers_inds[:, :, 1] * Ws + \
                      centers_inds[:, :, 0]  # n x L
            is_cared_in_the_level = self.assign_fpn_level_2(bboxes)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind)  # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        return pos_inds  # N
    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2)  # Lx2
        crit = ((boxes[:, 2:] - boxes[:, :2]) ** 2).sum(dim=1) ** 0.5 / 2  # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level

    def assign_fpn_level_2(self, boxes):
        '''
        Inputs:
            boxes: n x 4,[x,y,w,h,a]
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2)  # Lx2
        crit = ((boxes[:, 2]) ** 2 + (boxes[:, 3]) ** 2) ** 0.5 / 2  # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
                                (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level

    def get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() *
                           strides_expanded).float() + strides_expanded / 2
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
            (dist_y <= strides_expanded[:, :, 0])

    def _get_reg_targets(self, reg_targets, dist, mask, angle):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        dist[mask == 0] = INF * 1.0
        #
        # ga[mask==0]  =  INF * 1.0

        min_dist, min_inds = dist.min(dim=1)  # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds]  # M x N x 5 --> M x 5
        reg_targets_per_im[min_dist == INF] = - INF
        reg_angle = angle[
            range(len(angle)), min_inds]
        reg_angle[min_dist == INF] = - INF

        return reg_targets_per_im, reg_angle

    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 5
            size_ranges: M x 2
        '''
        crit = ((reg_targets_per_im[:, :, :2] +
                 reg_targets_per_im[:, :, 2:]) ** 2).sum(dim=2) ** 0.5 / 2
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
                                (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level

    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps

    def _create_agn_heatmaps_from_dist_ga(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''

        pass

    def _flatten_outputs(self, reg_pred, agn_hm_pred, angle_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        reg_pred = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], 0)
        agn_hm_pred = torch.cat([x.permute(0, 2, 3, 1).reshape(-1)
                                 for x in agn_hm_pred], 0) if self.with_agn_hm else None
        angle_pred = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(-1) for x in angle_pred], 0)
        return reg_pred, agn_hm_pred, angle_pred



    def get_bboxes(self, clss_per_level, reg_pred,
                   agn_hm_pred, angle_pred, img_metas, cfg=None):

        '''
            由预测值生成 bbox，用于后续求loss
        '''

        #  grids hold the center of featmap grid
        grids = self.compute_grids(agn_hm_pred)

        #
        agn_hm_pred = [x.sigmoid() for x in agn_hm_pred]

        # generate bboxs from heatmap and bbox prediction
        boxlists = multi_apply(self.predict_single_level, grids, agn_hm_pred,
                               reg_pred, angle_pred, self.strides, [None for _ in agn_hm_pred])

        boxlists = [torch.cat(boxlist) for boxlist in boxlists]
        final_boxlists = []
        for b in range(len(boxlists)):
            final_boxlists.append(self.nms_and_topK(boxlists[b], nms=True))
        return final_boxlists

    def predict_single_level(self, grids, heatmap, reg_pred, angle_pred,stride,
                             agn_hm, is_proposal=False):
        N, C, H, W = heatmap.shape
        # put in the same format as grids

        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1)  # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C)  # N x HW x C

        reg_pred_ = reg_pred * stride
        box_regression = reg_pred_.view(N, 4, H, W).permute(0, 2, 3, 1)
        angle_regression = angle_pred.view(N, 1, H, W).permute(0, 2, 3, 1)

        # fusion bbox and angel into one tensor
        box_regression = torch.cat([box_regression, angle_regression],dim=-1)
        box_regression = box_regression.reshape(N, -1, 5)

        candidate_inds = heatmap > self.score_thresh  # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # N
        pre_nms_topk = self.pre_nms_topk_train \
            if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk)  # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i]  # HW x C
            per_candidate_inds = candidate_inds[i]  # n
            per_box_cls = per_box_cls[per_candidate_inds]  # n

            per_candidate_nonzeros = per_candidate_inds.nonzero(as_tuple=False)
            per_box_loc = per_candidate_nonzeros[:, 0]  # n
            per_class = per_candidate_nonzeros[:, 1]  # n

            per_box_regression = box_regression[i]  # HW x 5
            per_box_regression = per_box_regression[per_box_loc]  # n x 5
            per_grids = grids[per_box_loc]  # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i]  # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]

            detections = self.bbox_coder.decode(
                per_grids, per_box_regression, max_shape=None)
            # detections[:, 2] = torch.max(
            #     detections[:, 2].clone(), detections[:, 0].clone() + 0.01)
            # detections[:, 3] = torch.max(
            #     detections[:, 3].clone(), detections[:, 1].clone() + 0.01)
            scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls  # n
            scores = torch.unsqueeze(scores, 1)  # n x 1
            #boxlist = torch.cat([detections, scores], dim=1)
            boxlist = detections
            results.append(boxlist)
        return results

    def nms_and_topK(self, boxlist, max_proposals=-1, nms=True):

        cfg = self.test_cfg
        scores = boxlist[:, -1]
        padding = scores.new_zeros(scores.shape[0], 1)
        scores = torch.cat([scores.view(scores.shape[0],1), padding],dim=1)
        _,_, keep = multiclass_nms_rotated(
            boxlist[:, :5], scores, 0.1,cfg.nms,cfg.max_per_img,return_inds=True)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        result = boxlist[keep]

        num_dets = len(result)
        post_nms_topk = self.post_nms_topk_train if self.training else \
            self.post_nms_topk_test
        if num_dets > post_nms_topk:
            cls_scores = result[:, 4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                num_dets - post_nms_topk + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]

        return result

    #
    # def refine_bboxes(self, clss_per_level, reg_pred,
    #                    agn_hm_pred, angle_pred):
    #
    #     '''
    #         由预测值生成 bbox，用于后续求loss
    #     '''
    #
    #     #  grids hold the center of featmap grid
    #     grids = self.compute_grids(agn_hm_pred)
    #
    #     #
    #     agn_hm_pred = [x.sigmoid() for x in agn_hm_pred]
    #
    #     # generate bboxs from heatmap and bbox prediction
    #     boxlists = multi_apply(self.predict_single_level, grids, agn_hm_pred,
    #                            reg_pred, angle_pred, self.strides, [None for _ in agn_hm_pred])
    #     final_boxlists = boxlists
    #
    #     return final_boxlists

    def refine_bboxes(self, cls_scores, bbox_preds, agn_hm_pred_per_level, angle_preds):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(agn_hm_pred_per_level)
        assert num_levels == len(agn_hm_pred_per_level)
        num_imgs = agn_hm_pred_per_level[0].size(0)
        for i in range(num_levels):
            assert num_imgs == agn_hm_pred_per_level[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        #featmap_sizes = [bbox_preds[i].shape[-2:] for i in range(num_levels)]
        # featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
        #                                                bbox_preds[0].dtype,
        #                                                bbox_preds[0].device)

        mlvl_points = self.compute_grids(agn_hm_pred_per_level)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

