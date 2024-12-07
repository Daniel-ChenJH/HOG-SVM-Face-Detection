# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/12/13 16:35:26
@Version :   1.0
@Desc    :   HOG-SVM人脸检测任务主函数
'''

# Import libs here:
import os

if __name__ == '__main__':
    # print("------------------Part 1：构建负例------------------")
    # build_neg_sh = 'python build_neg_sample.py'
    # a = os.system(build_neg_sh)
    # if a:
    #     print("运行build_neg_sample.py出错！\n")

    # print("------------------Part 2：提取特征------------------")
    # get_hog_sh = 'python get_hog.py'
    # b = os.system(get_hog_sh)
    # if b:
    #     print("运行get_hog.py出错！\n")

    # print("------------------Part 3：训练模型------------------")
    # train_sh = 'python train.py'
    # b = os.system(train_sh)
    # if b:
    #     print("运行train.py出错！\n")

    print("------------------Part 3：人脸检测------------------")
    detection_sh = 'python detection.py'
    b = os.system(detection_sh)
    if b:
        print("运行detection.py出错！\n")


