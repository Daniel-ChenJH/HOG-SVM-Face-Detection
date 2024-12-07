# -*- encoding: utf-8 -*-
'''
@File    :   detection_softnms.py
@Time    :   2022/12/13 16:35:26
@Version :   1.0
@Desc    :   HOG-SVM人脸检测任务检测函数
'''

import os
import pickle
import cv2
from tqdm import tqdm
from get_hog import get_hog_skimage
from build_neg_sample import mkdir
import numpy as np
import torch

def get_block(img, record):
    block = img[record[0]:record[0] + record[2], record[1]:record[1] + record[2]]
    return block


def sliding_window(img_name, img, classifier):
    # 设置初始窗长与原图高的比例为1:8
    scale = 4
    # 设置初始滑窗长度（正方形）和步长
    win_size = img.shape[0] // scale
    # 保证步长大于1
    step_size = max(win_size // 10, 1)
    #step_size = 2
    candidate_box = []
    score = []
    while scale >= 2:
        # 设置窗的左上角为A点，初始化为（0，0）
        Ax = 0
        Ay = 0
        # 层优先滑窗
        while Ay + win_size < img.shape[0]:
            while Ax + win_size < img.shape[1]:
                win_hog = get_hog_skimage(cv2.resize(img[Ay:Ay + win_size, Ax:Ax + win_size], (100, 100))).reshape(
                    (1, -1))
                y_pred = classifier.predict(win_hog)
                decision = classifier.decision_function(win_hog)
                # print(decision)
                if y_pred == 1:
                    candidate_box.append((Ax, Ay, win_size))
                    score.append(decision)
                Ax += step_size
            Ax = 0
            Ay += step_size
        Ay = 0
        # result的结构是：每行表示一个候选框，每行的元素依次是左上角x坐标、左上角y坐标、窗长
        if len(candidate_box) > 0:
            break
        else:
            scale -= 1
            win_size = img.shape[0] // scale

    if len(candidate_box) == 0:
        print("\n" + img_name + "未检测到人脸\n")
    else:
        # print(len(candidate_box))
        # print(len(score))
        best_index = score.index(max(score))
        tmp = best_index
        # rel_img = cv2.rectangle(img, (candidate_box[best_index][0], candidate_box[best_index][1]), (
        #     candidate_box[best_index][0] + candidate_box[best_index][2],
        #     candidate_box[best_index][1] + candidate_box[best_index][2]),
        #                         (240, 124, 130), 2)
        # cv2.imwrite(max_output_path + "\\" + filename[:-4] + "_max.jpg", rel_img)
        # cv2.imshow("candidate_box", rel_img)
        # cv2.waitKey(0)
        # 进行非极大值抑制(默认IoU阈值为0.2)
        #result = nms(candidate_box, score,threshold=0.2)
################## soft-NMS #######################
#使用了pytorch，先进行数据类型的转换
        candidate_box = np.array(candidate_box)
        score = np.array(score)
        result,scores = soft_nms_pytorch(candidate_box, score)
###################################################
        # 绘制候选框
        # result_img = rel_img
        # for best_index in range(len(result)):
        #     if best_index != tmp:
        #         result_img = cv2.rectangle(rel_img, (result[best_index][0], result[best_index][1]),
        #                                (result[best_index][0] + result[best_index][2],
        #                                 result[best_index][1] + result[best_index][2]), (255, 0, 0), 2)

        # cv2.imwrite(nms_output_path + "\\" + filename[:-4] + "_nms.jpg", result_img)
        # # cv2.imshow("result", result_img)
        # cv2.waitKey(0)
        tmp_list=[]  
        result = transsize(result)
        if len(result)==0:return None
        for best_index in range(len(result)):
            tmp_list.append([int(result[best_index][0]*img.shape[0]), int(result[best_index][1]*img.shape[1]),
                                        int(result[best_index][2]*img.shape[0]),
                                            int(result[best_index][3]*img.shape[1])])
        gt=get_gt_bbox(r'face-detection\Annoatation\boundingbox', img_name)
        return calculate_stats(gt,tmp_list)   

def transsize(or_box):
    win_size = or_box[:,2]
    tmp = np.zeros((or_box.shape[0],2))
    tmp[:,0] = or_box[:,0]+win_size
    tmp[:,1] = or_box[:,1]+win_size
    box = np.concatenate((or_box[:,:2],tmp),axis=1)
    return box

def get_gt_bbox(box_path, img_name):
    """
    获取图片人脸的边界框信息。
    :param box_path: boundingbox目录path
    :param img_name: 对应图片的名称
    :return: 返回一个包含四个int整数的列表，前两个为边框左上角坐标、后两个为右下角坐标
    """
    with open(box_path + "\\" + img_name[:-4] + '_boundingbox.txt', 'r', encoding='utf-8') as f:
        axis = f.readline()
    axis_list = axis.split(' ')
    # 删除最后一个空元素，防止float转换时报错。
    axis_list.pop()
    # print(axis_list)

    # 将字符串元素转换为整形元素
    axis_new_list = list(map(lambda x: int(float(x)), axis_list))
    # print(axis_new_list)
    return axis_new_list

    
def calculate_stats(gt,pred_list):
    iou_list=np.array([calculateIou(pred,gt) for pred in pred_list])
    if len(pred_list)==1: return iou_list[0]
    else:
        if len(iou_list[iou_list>0.5])>1: return sum(iou_list[iou_list>0.5])/len(iou_list[iou_list>0.5])
        else:return  max(iou_list)

def calculateIou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
 
    w = max(0, min(x2, xx2) - max(x1, xx1))
    h = max(0, min(y2, yy2) - max(y1, yy1))
    interarea = w * h
    union = (x2-x1)*(y2-y1) + (xx2-xx1)*(yy2-yy1) - interarea
 
    # 交并比
    return interarea / union

def soft_nms_pytorch(or_box, box_scores, sigma=0.5, thresh=0.1, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        or_box:        boxes coordinate tensor (format:[x1,y1,win_size])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the selected boxes
    """
    if torch.cuda.is_available():
        cuda = 1
        device = 'cuda'
    else:
        device = 'cpu'
    
    win_size = or_box[:,2]
    tmp = np.zeros((or_box.shape[0],2))
    tmp[:,0] = or_box[:,0]+win_size
    tmp[:,1] = or_box[:,1]+win_size
    box = np.concatenate((or_box[:,:2],tmp),axis=1)
    
    
    if type(box) != torch.Tensor:
        box = torch.tensor(box,dtype=torch.float,device=device)
    if type(box_scores) != torch.Tensor:
        box_scores = torch.tensor(box_scores,dtype=torch.float,device=device)

    # Indexes concatenate boxes with the last column
    N = box.shape[0]

    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    box = torch.cat((box, indexes), dim=1)

    # The order of boxes coordinate is [x1,y1,x2,y2]
    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = box[:, 2]
    y2 = box[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            maxpos = maxpos.item()
            if tscore < maxscore:
                box[i], box[maxpos + i + 1] = box[maxpos + i + 1].clone(), box[i].clone()
                scores[i], scores[maxpos + i + 1] = scores[maxpos + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()
        else:
            maxscore = scores[-1]
            maxpos = -1 

        # IoU calculate
        if pos < N:
            xx1 = np.maximum(box[i, 0].to("cpu").numpy(), box[pos:, 0].to("cpu").numpy())
            yy1 = np.maximum(box[i, 1].to("cpu").numpy(), box[pos:, 1].to("cpu").numpy())
            xx2 = np.minimum(box[i, 2].to("cpu").numpy(), box[pos:, 2].to("cpu").numpy())
            yy2 = np.minimum(box[i, 3].to("cpu").numpy(), box[pos:, 3].to("cpu").numpy())
        #print(box[i, 0],xx1)
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
            ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))
        # Gaussian decay
            weight = torch.exp(-(ovr * ovr) / sigma)
            scores[pos:] = (weight * scores[pos:].T).T
            
    scores = scores.to("cpu").numpy() 
    flag = 0   #判断阈值是否过大
    tmp =thresh
    # select the boxes and keep the corresponding indexes
    while flag != 1:
        count = 0
        for i in range(len(scores)):
            if scores[i] > tmp:
                count += 1
        keep = np.zeros([count,1],dtype=int)
        count = 0
        for i in range(len(scores)):
            if scores[i] > tmp:
                keep[count] = i
                count += 1
                flag = 1 
        keep = keep.astype(int)
        tmp /= 2
    
    result = np.zeros([len(keep),3],dtype=int)
    for i in range(0,len(keep)-1):
        result[i] = or_box[keep[i],:3]
    return result, scores

def nms(box, evaluation, threshold=0.2):
    """
    对候选集进行非极大值抑制操作
    :param box: 候选集
    :param evaluation: 评价指标
    :param threshold: IoU阈值，默认为0.2
    :return: 非极大值抑制后的候选集
    """
    results = []
    while len(box) != 0:
        # step 1：根据置信度得分进行排序
        max_score = max(evaluation)
        max_index = evaluation.index(max_score)
        # step 2：选择置信度最高的候选框加入到最终结果列表，并在候选框中删除
        results.append(box[max_index])
        del box[max_index]
        del evaluation[max_index]
        # step 3：计算所有边界框的面积（但是由于同一scale下的面积相等，所以可以省略此步骤）
        # step 4：计算置信度最高的边框与其余边框的IoU
        box_temp = []  # 此处解决for循环list下表超出问题，具体可以参考：https://blog.csdn.net/weixin_43269020/article/details/88191630
        temp_score = []
        for index, value in enumerate(box):
            IoU = cal_IoU(results[-1], value)
            if IoU < threshold:
                box_temp.append(value)
                temp_score.append(evaluation[index])
        box = box_temp
        evaluation = temp_score
    return results


def cal_IoU(box1, box2):
    """
    两个边界框的交集部分除以它们的并集
    :param box1: 阈值最大的边框参数列表
    :param box2: 候选框参数列表
    :return: 二者的IoU
    """
    box1_area = box1[2] ** 2
    box2_area = box2[2] ** 2
    left_column_max = max(box1[0], box2[0])
    right_column_min = min(box1[0] + box1[2], box2[0] + box2[2])
    up_row_max = max(box1[1], box2[1])
    down_row_min = min(box1[1] + box1[2], box2[1] + box2[2])

    if left_column_max >= right_column_min or up_row_max >= down_row_min:
        return 0
    else:
        cross_area = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return cross_area / (box1_area + box2_area - cross_area)


if __name__ == '__main__':
    boundingbox_path = r'face-detection\Annoatation\boundingbox'
    original_path = r'face-detection\Image\original'
    aligned_path = r'face-detection\Image\original'
    negative_path = r'face-detection\Image\negative'
    max_output_path = r'output\max'
    nms_output_path = r'output\nms'
    model_pkl_folder = r'pkl\model'

    mkdir(max_output_path)
    mkdir(nms_output_path)

    # 读取svm模型
    with open(model_pkl_folder + "\\svm.model", 'rb') as f:
        clf = pickle.load(f)

    iou_list = []
    # 对图像进行识别
    for filename in tqdm(os.listdir(original_path)[:3]):
        image = cv2.imread(original_path + "\\" + filename, 0)
        tmp=sliding_window(filename, image, clf)
####################测试单张照片###################################
    #    if filename == "train_0627.jpg":
    #        image = cv2.imread(original_path + "\\" + filename, 0)
    #        sliding_window(filename, image, clf)
##################################################################
    if tmp is not None:
        iou_list.append(tmp)
    print("检测图片保存完成！\n")
    print(iou_list)
    print(sum(iou_list)/len(iou_list))
