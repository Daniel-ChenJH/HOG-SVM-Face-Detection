# -*- encoding: utf-8 -*-
'''
@File    :   wbf.py
@Time    :   2022/12/13 16:35:26
@Version :   1.0
@Desc    :   HOG-SVM人脸检测任务wbf候选框筛选函数
'''

import warnings
import numpy as np
from numba import jit


@jit(nopython=True)
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):  # 遍历第一个模型生成的boxes
            score = scores[t][j]        # 对应的score
            if score < thr:             # 过滤score小于 thr的框
                continue
            label = int(labels[t][j])   # 确定种类标签
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])
            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]   # t --> 第t个模型
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)  # 建立字典new_boxes -->{label: [[label, score, weight, model index, x1, y1, x2, y2], ...]}

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])    # current_boxes --> [label, score, weight, model index, x1, y1, x2, y2], ...]
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]  # 根据score排序

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):   # boxes --> [label, score, weight, model index, x1, y1, x2, y2]
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]  # score
        conf_list.append(b[1])
        w += b[2]     # weight
    box[0] = boxes[0][0]   # label
    if conf_type == 'avg':
        box[1] = conf / len(boxes)   # score为平均值
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
    box[2] = w
    box[3] = -1 # model index field is retained for consistensy but is not used.
    box[4:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):   # boxes_list第一次时为空，第二次时为[[first]]， new_box --> [[label, score, weight, model index, x1, y1, x2, y2], ...]中遍历
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:  # 判断label值是否相同
            continue
        iou = bb_intersection_over_union(box[4:], new_box[4:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):  # 0.6, 0.5
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()
    # filtered_boxes -->{label: array([[label, score, weight, model index, x1, y1, x2, y2], ...])}其中各个key中已根据score大小进行排序
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    # print("filtered_boxes", filtered_boxes)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]   # boxes --> array([[label, score, weight, model index, x1, y1, x2, y2], ...])
        new_boxes = []
        weighted_boxes = []
        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)   # 第一次循环返回值为-1和iou_thr
            if index != -1:
                new_boxes[index].append(boxes[j])
                # print("new_boxes %d" % j, new_boxes)
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])     # new_boxes和weight_boxes填充值为boxes[0]
                weighted_boxes.append(boxes[j].copy())
        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / weighted_boxes[i][2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i][1] = weighted_boxes[i][1] *  clustered_boxes[idx, 2].sum() / weights.sum()

            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / (weighted_boxes[i][2] + weights[mask].sum())
            elif not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1] * min(weights.sum(), len(clustered_boxes)) / weights.sum()    # 对score进行操作
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

