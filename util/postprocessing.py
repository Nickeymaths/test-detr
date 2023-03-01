from datasets.thyroid import ThyroidDetection
import os
import glob
import pydicom
import pandas as pd
import numpy as np
import ast
from datasets.thyroid import body_cut, get_im_from_dcm, gray_to_pil, increase_count, make_thyroid_transforms, create_imbatch
import cv2

def get_gt_info_from_dcm(fp):
    """

    Args:
            roidb (pydicom dataset): ROI dataset read from dicom image

    Returns:
            list: list of bounding boxes sorted by top left y axis, thus 1st element is thyroid's \
            ROI, 2nd element is shoulder's ROID.
    """
    
    ds = pydicom.dcmread(fp)
    roidb = ds[0x0057, 0x1001]
    gt_boxes = []
    top_left_ys = []
    for roi_dataset in roidb:
        bbox_info = list(roi_dataset[0x0057, 0x105B])    # color = roi_dataset[0x0057, 0alue
        x_min, x_max, y_min, y_max = (
            np.min(bbox_info[0::3]),
            np.max(bbox_info[0::3]),
            np.min(bbox_info[1::3]),
            np.max(bbox_info[1::3]),
        )

        gt_boxes.append([x_min, y_min, x_max, y_max])
        top_left_ys.append(y_min)

    sorted_gt_bboxes = list(
        map(gt_boxes.__getitem__, np.argsort(top_left_ys)))
    labels = [2, 1]
    return sorted_gt_bboxes, labels

def add_gt_info(summary_tb, root_dir):
    gt_bboxes = []
    gt_labels = []
    img_ids = []
    
    for img_id in summary_tb['img_id']:
        bbox, label = get_gt_info_from_dcm(os.path.join(root_dir, str(img_id)+'.dcm'))
        gt_bboxes.append(bbox)
        gt_labels.append(label)
        img_ids.append(img_id)
    
    additional_info = pd.DataFrame({'img_id': img_ids, 'gt_bboxes': gt_bboxes, 'gt_labels': gt_labels})
    return pd.merge(left=summary_tb, right=additional_info, how='left', on=['img_id'])

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Command help to extract bbox of shoulder, \
                                     thyroid ROI from dicom image then add them to existing csv file",
        add_help=add_help,
    )
    parser.add_argument(
        "--summary_fp",
        help="Path to summary csv file which contains img_id, predicted results by model",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--root_dir",
        help="Path to dicom image directory which contains all dicom images in summary table",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--out",
        help="Path to output folder which places generated annotation file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--brighness_levels",
        help="For adding brighness to blur image",
        type=float,
        default=5
    )
    return parser

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, 2, 1, 3}
        The (0, 1) position is at the top left corner,
        the (2, 3) position is at the bottom right corner
    bb2 : dict
        Keys: {0, 2, 1, 3}
        The (0, 1) position is at the top left corner,
        the (2, 3) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_info(gt_bboxes, pred_bboxes, gt_labels, pred_labels):
    accumulation = []
    for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
        gt_bbox_idx = gt_labels.index(pred_label)
        if len(gt_labels) == len(gt_bboxes):
            gt_bbox = gt_bboxes[gt_bbox_idx]
            accumulation.append(get_iou(pred_bbox, gt_bbox))
        else:
            print('LACK ANNOTATION: [gt_box: {}, gt_label: {}]'.format(len(gt_bboxes), len(gt_labels)))
    return accumulation

def create_gt_pred_tb(summary_tb, args):
    summary_tb['pred_bboxes'] = summary_tb['pred_bboxes'].apply(lambda x: ast.literal_eval(x))
    summary_tb['pred_labels'] = summary_tb['pred_labels'].apply(lambda x: ast.literal_eval(x))

    df = add_gt_info(summary_tb, args.root_dir)
    df['iou'] = df.apply(lambda x: get_iou_info(x.gt_bboxes, x.pred_bboxes, x.gt_labels, x.pred_labels), axis=1)
    df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    return df

def draw_bbox(img, bboxes, labels, color=(0, 255, 0)):
    if len(bboxes) == 0:
        return
    
    for idx, box in enumerate(bboxes):
        bbox = [int(c) for c in box]
        # bbox = box.cpu().data.numpy()
        # bbox = bbox.astype(np.int32)
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ])
        bbox = bbox.reshape((4, 2))
        cv2.polylines(img, [bbox], True, color, 2)
        cv2.putText(img, str(labels[idx]), bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

def gen_gtapred_img(df, args):
    for idx, row in df.iterrows():
        img_id = row.img_id
        orig_image = body_cut(get_im_from_dcm(os.path.join(args.root_dir, img_id+'.dcm')))
        orig_image = gray_to_pil(increase_count(orig_image, args.brighness_levels)).convert("RGB")

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        draw_bbox(img, row.pred_bboxes, row.pred_labels, color=(0, 255, 0))
        draw_bbox(img, row.gt_bboxes, row.gt_labels, color=(0, 0, 255))

        img_save_path = os.path.join(args.out, img_id+".png")
        cv2.imwrite(img_save_path, img)

def main(args):
    os.makedirs(args.out, exist_ok=True)
    summary_tb = pd.read_csv(args.summary_fp, index_col=False)
    df = create_gt_pred_tb(summary_tb, args)
    gen_gtapred_img(df, args)

if __name__=='__main__':
    args = get_args_parser().parse_args()
    main(args)
    
    