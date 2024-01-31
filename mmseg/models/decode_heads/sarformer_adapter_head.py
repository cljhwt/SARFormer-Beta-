# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Dict
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from mmdet.models.utils import multi_apply, preprocess_panoptic_gt
from .merge_modules import *
from ultralytics import YOLO

from .automatic_mask_generator import SamAutomaticMaskGenerator,sam_model_registry
from segment_anything import SamPredictor

@MODELS.register_module()
class SARFormer_Adapter(MMDET_Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        self.channels=kwargs["feat_channels"]
        self.num_queries=kwargs["num_queries"]
        self.query_feat = None

        self.feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(self.feat_channels, self.num_classes + 1)
        #检查一下维度
        self.enc_output = nn.Linear(self.channels, self.feat_channels)
        self.enc_output_norm = nn.LayerNorm(self.feat_channels)

        #SAM
        self.sam_checkpoint = kwargs["sam_checkpoint"]
        self.model_type = kwargs["model_type"]
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        if torch.cuda.is_available():
            sam.to(device="cuda")
            sam = sam.cuda()
        # mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_batch= 128,
        #     points_per_side=32,
        #     pred_iou_thresh=0.86,
        #     stability_score_thresh=0.92,
        #     crop_n_layers=1,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=640*640/100,  # Requires open-cv to run post-processing
        # )
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
        )
        # self.mask_predictor = MyPredictor(sam)
        self.mask_generator=mask_generator
        self.scale = 1
        save_dir = './'


    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()
            #构造masks和另外两个数组

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas


    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,images,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples,images)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],images,
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """

        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples,images)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        #将mask进行上采样适应尺寸
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        #这行代码使用了F.softmax函数对mask_cls_results进行softmax操作。dim=-1表示对最后一个维度进行softmax操作。[..., :-1]用于移除最后一个维度上的最后一个元素，因为最后一个元素对应的是背景类别
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        #使用torch.einsum函数执行张量的乘法操作。cls_score是分类的概率分布，mask_pred是遮罩预测结果。该乘法操作将分类的概率分布与遮罩预测结果相乘，并根据乘法结果生成分割的logits（对数概率）。
        # seg_logits是一个张量，形状为[batch_size, num_classes, height, width]
        #对数概率，为了将预测结果用于像素级别的分类概率计算和损失函数计算
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList,images) -> Tuple[List[Tensor]]:

        feat, adapter_c = x[0], x[1]

        device = x[0][0].device
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]

        self.batch_size = len(batch_img_metas)
        batch_size=self.batch_size
        mask_generator=self.mask_generator
        # mask_predictor=self.mask_predictor

         # 定义像素平均值和标准差
        mean = torch.tensor([123.675, 116.28, 103.53]).to(device)
        std = torch.tensor([58.395, 57.12, 57.375]).to(device)

        # 可能的分解方向
        mask_features, multi_scale_memorys = self.pixel_decoder(feat)
        adapter_feat, adapter_memo = self.pixel_decoder(adapter_c)

        mask_predictor = mask_generator.predictor
        images_re = (images * std[:, None, None] + mean[:, None, None]) / 255.0
        model = YOLO("yolov8n.pt")
        results = model.predict(images_re,verbose=False)
        YoloBoxes = [np.array(result.boxes.data.cpu()) for result in results]

        mask_queries = []
        for i in range(batch_size):
            YoloBox=YoloBoxes[i]
            len_yolo=len(YoloBox)

            image_temp = images[i:i+1, :, :, :]
            mask = mask_generator.generate(image_temp)

            if len_yolo:
                shape = image_temp.shape
                masks_filted = mask_filt(shape, YoloBox, mask)
                input_boxes = torch.tensor(YoloBox[:, :-2], device=device)
                transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, shape[-2:])
                yolo_masks, a, b = mask_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                yolo_masks = torch.squeeze(yolo_masks, 1)

                segmentation_tensors = [mask_dict['segmentation'] for mask_dict in masks_filted]
                # segmentation_tensors = [torch.tensor(arr) for arr in segmentation_tensors]  # 将NumPy数组转换为PyTorch tensor
                try:
                    segmentation_tensors_stacked = torch.stack(segmentation_tensors, dim=0)
                    sam_masks = torch.cat((yolo_masks, segmentation_tensors_stacked), dim=0)
                except Exception as e:
                    sam_masks = yolo_masks
                sam_masks = torch.squeeze(sam_masks, 1)
            else:
                num_sam = len(mask)
                if num_sam >= self.num_queries:
                    mask = sorted(mask, key=(lambda x: x['area']), reverse=True)
                mask = [item["segmentation"] for item in mask]
                sam_masks = torch.stack(mask, axis=-1).clone().detach().permute(2, 0, 1).to(device) if mask else torch.randn(1,*images.shape[2:],device=device,dtype=torch.float32) / 6. + 0.5

            # mid_time = time.time()
            # execution_time = mid_time - start_time
            # print(f"代码块的运行时间为：{execution_time}秒")
            num_sam = len(sam_masks)
            if num_sam < self.num_queries:
                mask_placeholder = torch.randn(self.num_queries - num_sam, *sam_masks.shape[1:], device=device,
                                               dtype=torch.float32) / 6. + 0.5
                mask_placeholder = float_to_mask(mask_placeholder, 0.5)
                x_start = torch.cat((sam_masks, mask_placeholder), dim=0)
            elif num_sam >= self.num_queries:
                select_mask = [True] * self.num_queries + [False] * (num_sam - self.num_queries)
                random.shuffle(select_mask)
                x_start = sam_masks[select_mask]
            else:
                x_start = sam_masks

            index = torch.randperm(x_start.shape[0]).to(device)
            x_start = (x_start[index] * 2. - 1.) * self.scale
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
            x_start = ((x_start / self.scale) + 1) / 2.
            x_start = x_start.to(torch.float32)
            mask_queries.append(x_start)

            # end_time = time.time()
            # execution_time2 = end_time - start_time
            # print(f"代码块的运行时间为：{execution_time2}秒")

        mask_queries = torch.stack(mask_queries, dim=0)
        # 第三步，mask对齐到attn_mask
        sam_mask = F.interpolate(mask_queries, multi_scale_memorys[0].shape[-2:], mode="bilinear", align_corners=False)
        sam_mask = (sam_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                  1) < 0.5).bool().detach()
        sam_mask = ~sam_mask

        #Adapter接到query_feat
        exmask_features = adapter_feat.flatten(2).permute(0, 2, 1).to(torch.float64)
        exnoise_masks = F.interpolate(mask_queries, (adapter_feat.shape[2], adapter_feat.shape[3]), mode="bilinear",
                                      align_corners=False)
        exnoise_masks = exnoise_masks.flatten(2).to(torch.float64)
        da = torch.sum(exnoise_masks, dim=2).unsqueeze(2)
        da[da == 0] = 1
        da = da.repeat(1, 1, self.channels)
        query_feat = torch.einsum("bqs,bsc->bqc", exnoise_masks, exmask_features)
        query_feat = torch.div(query_feat, da).permute(1, 0, 2).to(torch.float32)
        query_feat=query_feat.permute(1, 0, 2)

        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])  # (4,256,2,2)
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c) (4,4,256)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        # query_feat = self.query_feat.weight.unsqueeze(0).repeat(
        #     (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []

        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            sam_mask[torch.where(
                sam_mask.sum(-1) == sam_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                # cross_attn_mask=attn_mask,
                cross_attn_mask=sam_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            cls_pred, mask_pred, sam_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                                               (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list


    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        if self.training:  # this is to insure self.label_enc participate in the model
            loss_dict['loss_cls']+= 0.0*self.enc_output_norm.weight.sum()
            loss_dict['loss_cls'] += 0.0 * self.enc_output.weight.sum()
            loss_dict['loss_cls'] += 0.0 * self.enc_output_norm.bias.sum()
            loss_dict['loss_cls'] += 0.0 * self.enc_output.bias.sum()

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:

        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask
