# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Face dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Face_utils.py
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Thyroid dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/Thyroid_utils.py
"""
from pathlib import Path

import numpy as np

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as Thyroid_mask

import datasets.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os
import pydicom
from scipy import signal


class ThyroidDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, brighness_levels=5):
        super(ThyroidDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.brighness_levels = brighness_levels
        self.prepare = ConvertThyroidPolysToMask(return_masks)
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        org_im = body_cut(get_im_from_dcm(os.path.join(self.root, path)))
        im_batch = create_imbatch(org_im, self.brighness_levels)
        # im = gray_to_pil(increase_count(body_cut(im), 20)).convert("RGB")
        return gray_to_pil(org_im), im_batch    

    def __getitem__(self, idx):
        # Copy from load image function of CocoDetector
        id = self.ids[idx]
        org_img, img = self._load_image(id)
        target = self._load_target(id)
        # img, target = super(ThyroidDetection, self).__getitem__(idx)
        
        # Additional code
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        org_img, target = self.prepare(org_img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def get_im_from_dcm(path: str):
    im = pydicom.dcmread(path).pixel_array
    return im

def body_cut(im: np.ndarray):
    return  im[:256, :256] if im.shape[:2] != (512, 512) else im

def gray_to_pil(im: np.ndarray):
    im = Image.fromarray(np.uint16(im))
    return im

def increase_count(im: np.ndarray, factor: int=1):
    # scharr = np.array([[1, 1, 1],

    #                 [1, 2, 1],

    #                 [1, 1, 1]])*factor
    
    # scharr = np.array([[1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1],
    #             [1, 1, 1, 1, 1]])*factor
    
    scharr = np.array([[1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 4, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]])*factor
    

    im = signal.convolve2d(im, scharr, boundary='symm', mode='same')
    # im = ndimage.gaussian_filter(im.astype(np.float32)*1000*factor, sigma=1, mode='reflect').astype(np.uint16)
    im[im < 0] = 0
    im[im > 255] = 255
    return im

def create_imbatch(im: np.ndarray, brighness_levels: int):
    # im shape HxW
    im_batch = np.vstack([increase_count(im, 2**i) for i in range(brighness_levels)]).reshape(brighness_levels, *im.shape)
    im_batch = im_batch.transpose(1, 2, 0) # transpose to HxWxnum_level
    return im_batch.astype(np.uint8)

def convert_Thyroid_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = Thyroid_mask.frPyObjects(polygons, height, width)
        mask = Thyroid_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertThyroidPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_Thyroid_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to Thyroid api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_thyroid_transforms(image_set, brighness_levels=5):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(
            means * (brighness_levels // 3) + means[:brighness_levels % 3], 
            stds * (brighness_levels // 3) + stds[:brighness_levels % 3]
        )
    ])

    scales = [256, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Thyroid path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train", root / "annotations" / f'train.json'),
        "val": (root / "images" / "val", root / "annotations" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = ThyroidDetection(img_folder, ann_file, transforms=make_thyroid_transforms(image_set, args.brighness_levels), return_masks=args.masks, brighness_levels=args.brighness_levels)
    return dataset

