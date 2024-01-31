_base_ = ['../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth'
crop_size = (640, 640)
# data_preprocessor = dict(size=crop_size)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #像素平均值和标准差
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
num_classes = 150
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
              dict(type='WandbVisBackend')]
depths = [2, 2, 18, 2]
model = dict(
    type='EncoderDecoderMerge',
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type='BEiT',
        img_size=(640, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1,
        frozen_stages=1,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=None,
    decode_head=dict(
        type='SamMergeHead',
        in_channels=[768, 768, 768, 768],
        strides=[16, 16, 16, 16],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        sam_checkpoint = "../segment-anything-main/checkpoints/sam_vit_b_01ec64.pth",
        model_type = "vit_b",
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # auxiliary_head=dict(
    #     type='AdapterFCNHead',
    #     in_channels=768,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=150,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 640) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2560),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
#

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
# backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
# backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# custom_keys = {
#     'backbone': dict(lr_mult=0.1, decay_mult=1.0),
#     'backbone.patch_embed.norm': backbone_norm_multi,
#     'backbone.norm': backbone_norm_multi,
#     'absolute_pos_embed': backbone_embed_multi,
#     'relative_position_bias_table': backbone_embed_multi,
#     'query_embed': embed_multi,
#     'query_feat': embed_multi,
#     'level_embed': embed_multi
# }
# custom_keys.update({
#     f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
#     for stage_id, num_blocks in enumerate(depths)
#     for block_id in range(num_blocks)
# })
# custom_keys.update({
#     f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
#     for stage_id in range(len(depths) - 1)
# })
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.0001, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    # paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=10000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


train_dataloader = dict(batch_size=2,num_workers=1)
val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader
find_unused_parameters = False