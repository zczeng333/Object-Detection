# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:16:43 2018

@author: zzc93
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
import cv2
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import yolo_utils

'''部分一：分类阈值过滤'''


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    针对整张图片进行操作, 每个像素点对应有5个锚框, 每个锚框对应80个类别。
    经过此函数, 可以区分有无包含检测物体的锚框, 并且返回每个锚框对应的最大可能类别, 对应类别评分以及锚框位置尺寸
    通过阈值来过滤对象和分类的置信度。

    参数：
        box_confidence  - tensor类型, 维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
        boxes - tensor类型, 维度为(19,19,5,4), 包含了所有的锚框的（px,py,ph,pw）。
        box_class_probs - tensor类型, 维度为(19,19,5,80), 包含了所有单元格中所有锚框的所有对象( c1,c2,c3, ···, c80 )检测的概率。
        threshold - 实数, 阈值, 如果分类预测的概率高于它, 那么这个分类预测的概率就会被保留。
        box_scores - 锚框得分, 维度为(19,19,5,80)

    返回：
        scores - tensor 类型, 维度为(None,), 包含了保留了的锚框的分类概率。
        boxes - tensor 类型, 维度为(None,4), 包含了保留了的锚框的(b_x, b_y, b_h, b_w)
        classess - tensor 类型, 维度为(None,), 包含了保留了的锚框的索引

    注意："None"是因为你不知道所选框的确切数量, 因为它取决于阈值。
          比如：如果有10个锚框, scores的实际输出大小将是（10,）
    """

    # 第一步：计算锚框的得分
    # box_scores:锚框得分(最终函数返回值), box_confidence:指示锚框内是否含有物体, box_class_probs:指示锚框中每个类别对应的概率
    box_scores = box_confidence * box_class_probs

    # 第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)  # 寻找box_scores最后一个轴中最大元素的索引(也即一个锚框中80类中得分最高类的索引)
    box_class_scores = K.max(box_scores, axis=-1)  # 寻找box_scores最后一个轴中最大元素的得分(也即一个锚框中80类中的最高类别得分)

    # 第三步：根据阈值创建掩码
    filtering_mask = (box_class_scores >= threshold)  # 选择在阈值范围规定下的得分, 并将其编码为True或False以指示框中是否有目标

    # 对scores, boxes 以及 classes使用掩码
    # tf.boolean(tensor,mask):将掩码mask中为True的部分对应的张量tensor部分保存, 为False的部分对应的张量tensor部分舍弃
    scores = tf.boolean_mask(box_class_scores, filtering_mask)  # 获得满足阈值要求的锚框(认为锚框内存在检测物体), 其中最高类别得分
    boxes = tf.boolean_mask(boxes, filtering_mask)  # 获得满足阈值要求的锚框(认为锚框内存在检测物体)对应的尺寸(b_x, b_y, b_h, b_w)
    classes = tf.boolean_mask(box_classes, filtering_mask)  # 获得满足阈值要求的锚框(认为锚框内存在检测物体), 其中最高类别(对应的数字索引)



    return scores, boxes, classes


"""
'''测试部分一'''
with tf.Session() as test_a:	#启动tensorflow图计算
	#tf.random_normal(shape,mean,stddev,seed):创建大小为shape的随机数张量, 随机数服从均值为mean, 方差为stddev,随机种子seed(设置之后每次生成的随机数相同)
    box_confidence = tf.random_normal([19,19,5,1], mean=1, stddev=4, seed=1)
    boxes = tf.random_normal([19,19,5,4],  mean=1, stddev=4, seed=1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

    test_a.close()
"""

'''部分二：非最大值抑制'''


def iou(box1, box2):
    """
    针对某两个锚框进行处理, 计算出其两者之间的交并比, 并返回交并比数值
    实现两个锚框的交并比的计算

    参数：
        box1 - 第一个锚框, 元组类型, (x1, y1, x2, y2)
        box2 - 第二个锚框, 元组类型, (x1, y1, x2, y2)

    返回：
        iou - 实数, 交并比。
    """
    # 计算相交的区域的面积
    xi1 = np.maximum(box1[0], box2[0])  # 靠右的左边框
    yi1 = np.maximum(box1[1], box2[1])  # 靠上的下边框
    xi2 = np.minimum(box1[2], box2[2])  # 靠左的右边框
    yi2 = np.minimum(box1[3], box2[3])  # 靠下的上边框
    inter_area = (xi1 - xi2) * (yi1 - yi2)

    # 计算并集, 公式为：Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # 计算交并比
    iou = inter_area / union_area

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    为锚框实现非最大值抑制（ Non-max suppression (NMS)）

    参数：
        scores - tensor类型, 维度为(None,), yolo_filter_boxes()的输出
        boxes - tensor类型, 维度为(None,4), yolo_filter_boxes()的输出, 已缩放到图像大小（见下文）
        classes - tensor类型, 维度为(None,), yolo_filter_boxes()的输出
        max_boxes - 整数, 预测的锚框数量的最大值
        iou_threshold - 实数, 交并比阈值。

    返回：
        scores - tensor类型, 维度为(,None), 每个锚框的预测的可能值
        boxes - tensor类型, 维度为(4,None), 预测的锚框的坐标
        classes - tensor类型, 维度为(,None), 每个锚框的预测的分类

    注意："None"是明显小于max_boxes的, 这个函数也会改变scores、boxes、classes的维度, 这会为下一步操作提供方便。

    """
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")  # 用于tf.image.non_max_suppression(), 缺省时boxes个数默认上限为10
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # 初始化变量max_boxes_tensor

    # 使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    # tf.image.non_max_suppression(boxes,scores,max_output_size,iou_threshold,score_threshold,name),该函数用于实现非最大值抑制
    # boxes:锚框位置尺寸信息, scores:锚框对应类别(已完成最大可能类别选择)的得分, max_output_size:非最大值抑制最多选择框数, iou_threshold:交并比阈值
    # 函数返回值为各个框的索引值, 是一个一维数组(索引值从0开始)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # 使用K.gather()来选择保留的锚框
    # K.gather(reference, indices), 该函数用于根据索引在张量中找出对应子张量
    # reference:被搜寻张量, indices:索引值
    scores = K.gather(scores, nms_indices)  # 寻找非最大值抑制后的锚框分数
    boxes = K.gather(boxes, nms_indices)  # 寻找非最大值抑制后的锚框位置尺寸
    classes = K.gather(classes, nms_indices)  # 寻找非最大值抑制后的锚框类别



    return scores, boxes, classes


"""
'''测试部分二'''
with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

    test_b.close()
"""

'''部分三：对所有框进行过滤'''


def yolo_eval(yolo_outputs, image_shape=(720., 1280.),
              max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数, 框坐标和类。

    参数：
        yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片）, 包含4个tensors类型的变量：
                        box_confidence ： tensor类型, 维度为(None, 19, 19, 5, 1)
                        box_xy         ： tensor类型, 维度为(None, 19, 19, 5, 2)
                        box_wh         ： tensor类型, 维度为(None, 19, 19, 5, 2)
                        box_class_probs： tensor类型, 维度为(None, 19, 19, 5, 80)
        image_shape - tensor类型, 维度为（2,）, 包含了输入的图像的维度, 这里是(608.,608.)
        max_boxes - 整数, 预测的锚框数量的最大值
        score_threshold - 实数, 可能性阈值
        iou_threshold - 实数, 交并比阈值

    返回：
        scores - tensor类型, 维度为(,None), 每个锚框的预测的可能值
        boxes - tensor类型, 维度为(4,None), 预测的锚框的坐标
        classes - tensor类型, 维度为(,None), 每个锚框的预测的分类
    """

    # 获取YOLO模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 可信度分值过滤
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # 缩放锚框, 以适应原始图像
    # 本代码中实现的是在(608,608,3)RGB图像上的YOLO, 而输入输出图像并不一定是(608,608,3), 所以需要缩放锚框以匹配输入输出图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # 使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    # print(type(boxes))
    # temp_box=np.array(boxes)
    # for i in range(temp_box.shape[1]):
    #     temp=temp_box[:,i]

    return scores, boxes, classes


"""
'''测试部分三'''
with tf.Session() as test_c:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

    test_c.close()
"""

sess = K.get_session()  # 创建会话启动计算图


def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    运行存储在sess的计算图以预测image_file的边界框, 打印出预测的图与信息。

    参数：
        sess - 包含了YOLO计算图的TensorFlow/Keras的会话。
        image_file - 存储在images文件夹下的图片名称
    返回：
        out_scores - tensor类型, 维度为(None,), 锚框的预测的可能值。
        out_boxes - tensor类型, 维度为(None,4), 包含了锚框位置信息。
        out_classes - tensor类型, 维度为(None,), 锚框的预测的分类索引。 
    """
    # 图像预处理
    image, image_data = yolo_utils.preprocess_image(image_file, model_image_size=(608, 608))

    # 运行会话并在feed_dict中选择正确的占位符.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # 打印预测信息
    # if is_show_info:
    # print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    # 指定要绘制的边界框的颜色
    colors = yolo_utils.generate_colors(class_names)
    # 在图中绘制边界框
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存已经绘制了边界框的图
    # image.save(os.path.join("out", image_file), quality=100)


    cv2.imshow('Image', np.array(image))

    # 打印出已经绘制了边界框的图
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes


'''定义分类, 锚框与图像维度'''
class_names = yolo_utils.read_classes("model_data/coco_classes.txt")  # 读入类别标签文本
anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")  # 读入锚框尺寸(w,h), 共五种锚框
image_shape = (720., 1280.)
yolo_model = load_model("model_data/yolov2.h5")  # 加载训练模型(该模块位于keras模块下)
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))  # 将模型的输出转化为边界框
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)  # 过滤锚框
# 将视频文件转化为图像进行处理
vc = cv2.VideoCapture("video/test_video1.mp4")
totalFrameNumber = vc.get(7)  # 获取总视频帧数
c = 1
step = 3  # 定义帧数间隔
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
    print('File open error!')
if rval:
    while c < totalFrameNumber:
        rval, frame = vc.read()
        frame = cv2.resize(frame, (1280, 720))  # 将图像转化为处理的标准尺寸
        if (c % step == 0):  # every 5 fps write frame to img
            # 计算需要在前面填充几个0
            num_fill = int(len("0000") - len(str(1))) + 1
            # 对索引进行填充f
            # 开始绘制, 不打印信息, 不绘制图
            out_scores, out_boxes, out_classes = predict(sess, frame, is_show_info=False, is_plot=False)
        c = c + 1
        cv2.waitKey(1)
vc.release()
cv2.destroyAllWindows()
