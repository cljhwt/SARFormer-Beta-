# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple
from functools import partial

from segment_anything.modeling import Sam,ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = MyPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: torch.Tensor) -> List[Dict[str, Any]]:

        # Generate masks
        mask_data = self._generate_masks(image)
        device=mask_data["boxes"].device
        # Filter small disconnected regions and holes in masks
        #还没完成tensor改造
        # if self.min_mask_region_area > 0:
        #     mask_data = self.postprocess_small_regions(
        #         mask_data,
        #         self.min_mask_region_area,
        #         max(self.box_nms_thresh, self.crop_nms_thresh),
        #     )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle,device) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: torch.Tensor) -> MaskData:
        orig_size = image.shape[-2:]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        # data.to_numpy()
        return data

    def _process_crop(
        self,
        image: torch.Tensor,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        # x0, y0, x1, y1 = crop_box
        # cropped_im = image[y0:y1, x0:x1, :]
        # cropped_im_size = cropped_im.shape[:2]
        cropped_im_size=tuple(image.shape[-2:])
        if self.predictor.is_image_set==False:
            self.predictor.set_image(image)
        batch_size=image.size()[0]
        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, tuple(orig_size),batch_size)
            data.cat(batch_data)
            del batch_data
        # self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        batch_size
    ) -> MaskData:
        orig_h, orig_w = orig_size
        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        # # 复制 in_points
        # in_points_expanded = in_points.unsqueeze(1).repeat(1,batch_size , 1).permute(1, 0, 2)
        # # 复制 in_labels
        # in_labels_expanded = in_labels.unsqueeze(1).repeat(1,batch_size).permute(1,0)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    @torch.no_grad()
    def generate_batch(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        orig_size = image.shape[-2:]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        im_size = tuple(image.shape[-2:])
        batch_size = image.size()[0]
        # Get points for this crop
        points_scale = np.array(im_size)[None, ::-1]
        points_for_image = self.point_grids[layer_idxs[0]] * points_scale

        input_image = self.predictor.transform.apply_image_torch(image)
        transformed_points = self.predictor.transform.apply_coords(points_for_image, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        # 复制 in_points
        in_points_expanded = in_points.unsqueeze(1).repeat(1,batch_size , 1).permute(1, 0, 2)
        # 复制 in_labels
        in_labels_expanded = in_labels.unsqueeze(1).repeat(1,batch_size).permute(1,0)

        batched_input = []

        for i in range(batch_size):
            input_dict = {
                'image': image[i],
                'original_size': im_size,
                'point_coords': in_points_expanded,
                'point_labels': in_labels_expanded,
            }
            batched_input.append(input_dict)

        return batched_input


class MyPredictor(SamPredictor):
    def set_image(
        self,
        image: torch.Tensor,
        image_format: str = "RGB",
    ) -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = torch.flip(image, [-1])
        input_image = self.transform.apply_image_torch(image).to(self.device)
        input_image_torch = input_image.permute(0,1, 3, 2)

        self.set_torch_image(input_image_torch, [image.shape[-1],image.shape[-2]])

    @torch.no_grad()
    def set_torch_image(
            self,
            transformed_image: torch.Tensor,
            original_image_size: Tuple[int, ...],
    ) -> None:

        assert (
                len(transformed_image.shape) == 4
                and transformed_image.shape[1] == 3
                and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        #免去归一化
        input_image = self.model.preprocess(transformed_image)
        # input_image = transformed_image
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    @torch.no_grad()
    def predict_torch(
            self,
            point_coords: Optional[torch.Tensor],
            point_labels: Optional[torch.Tensor],
            boxes: Optional[torch.Tensor] = None,
            mask_input: Optional[torch.Tensor] = None,
            multimask_output: bool = True,
            return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

def rle_to_mask(rle: Dict[str, Any], device) -> torch.Tensor:
    """从未压缩的 RLE 数据计算二进制掩码."""
    h, w = rle["size"]
    mask = torch.BoolTensor(h * w).to(device)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose(0, 1).to(device)  # 转置以符合 C 顺序并移动到 GPU

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def remove_small_regions(
        mask: torch.Tensor, area_thresh: float, mode: str
) -> Tuple[torch.Tensor, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"

    working_mask = (correct_holes ^ mask).to(torch.uint8)
    labeled_mask, num_labels = F.connected_components(working_mask, connectivity=8)

    sizes = torch.bincount(labeled_mask.flatten())[1:]  # Label 0 is background label
    small_regions = torch.nonzero(sizes < area_thresh).squeeze(1) + 1

    if len(small_regions) == 0:
        return mask, False

    fill_labels = torch.cat([torch.tensor([0]), small_regions])

    if not correct_holes:
        fill_labels = torch.tensor([i for i in range(num_labels) if i not in fill_labels])

        # If every region is below threshold, keep the largest
        if len(fill_labels) == 0:
            fill_labels = torch.tensor([torch.argmax(sizes) + 1])

    mask = torch.isin(labeled_mask, fill_labels)

    return mask, True

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f,map_location='cpu')
        sam.load_state_dict(state_dict)
    return sam
