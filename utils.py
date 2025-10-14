import os
import base64
import cv2
import json


def get_labelme_gt(json_path):
    with open(json_path, 'r') as f:
        info = json.load(f)

    gt_bboxes = []
    for shape in info['shapes']:
        p1, p2 = shape['points']
        xmin = int(min(p1[0], p2[0]))
        xmax = int(max(p1[0], p2[0]))
        ymin = int(min(p1[1], p2[1]))
        ymax = int(max(p1[1], p2[1]))
        gt_bboxes.append([xmin, ymin, xmax, ymax])
    return gt_bboxes


class Evaluator:
    def __init__(self, prefix):
        self.prefix = prefix
        self.n_gt = 0
        self.n_pred = 0
        self.n_right = 0
        self.precision = 0
        self.recall = 0
    
    def update(self, gt=0, pred=0, right=0):
        self.n_gt += gt
        self.n_pred += pred
        self.n_right += right
    
    def summary(self):
        self.precision = self.n_right / self.n_pred
        self.recall = self.n_right / self.n_gt
        print(f'{self.prefix} precision: {self.precision:.2%}={self.n_right}/{self.n_pred}')
        print(f'{self.prefix} recall: {self.recall:.2%}={self.n_right}/{self.n_gt}')


def evaluate_image(gt_bboxes, pred_bboxes):
    ret = {'gt': 0, 'pred': 0, 'right': 0}
    if gt_bboxes:
        ret['gt'] = 1
    if pred_bboxes:
        ret['pred'] = 1
    if gt_bboxes and pred_bboxes:
        ret['right'] = 1
    return ret


def get_iou(b1, b2):
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    xmin = max(x11, x21)
    xmax = min(x12, x22)
    ymin = max(y11, y21)
    ymax = min(y12, y22)

    if xmin >= xmax or ymin >= ymax:
        return 0
    inter_area = (ymax - ymin) * (xmax - xmin)
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def evaluate_bbox(gt_bboxes, pred_bboxes, iou_th=0.5):
    ret = {'gt': len(gt_bboxes), 'pred': len(pred_bboxes), 'right': 0}

    pred_masks = [0] * len(pred_bboxes)
    for gtb in gt_bboxes:
        matchi = -1
        max_iou = 0
        for i, pdb in enumerate(pred_bboxes):
            iou = get_iou(gtb, pdb)
            if iou > max_iou:
                matchi = i
                max_iou = iou
        if matchi > -1 and max_iou > iou_th:
            if pred_masks[matchi] == 0:
                pred_masks[matchi] = 1
                ret['right'] += 1
    return ret


def evaluate(result):
    eimg = Evaluator('image')
    ebox = Evaluator('bbox')
    for image_path, gt_bboxes, pred_bboxes in result:
        eimg.update(**evaluate_image(gt_bboxes, pred_bboxes))
        ebox.update(**evaluate_bbox(gt_bboxes, pred_bboxes, iou_th=0.5))
    eimg.summary()
    ebox.summary()