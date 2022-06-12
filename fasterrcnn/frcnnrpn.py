import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.frcnn import FasterRCNN
from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input)
from utils.utils_bbox import DecodeBox


'''
    该类专门用于获得Faster-Rcnn第一阶段rpn层得到的建议框，
    模型路径与主干网络要求与预测一样
'''
class FRCNN(object):
    _defaults = {
        
        "model_path"    : 'logs/ep106-loss0.647-val_loss1.006.pth',
        "classes_path"  : 'model_data/voc_classes.txt',
        
        "backbone"      : "resnet50",
       
        "confidence"    : 0.5,
        
        "nms_iou"       : 0.3,
        
        'anchors_size'  : [8, 16, 32],
        
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    
    def generate(self):
        
        self.net    = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
    
    def detect_image(self, image, crop = False, count = False):
        
        image_shape = np.array(np.shape(image)[0:2])
        
        #   计算resize后的图片的大小，resize后的图片短边为600
        
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        
        image       = cvtColor(image)
        
        #   给原图像进行resize，resize到短边为600的大小上
        
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        
        #   添加上batch_size维度
        
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            
            
            _, _, rois, _,= self.net(images)
            #   对建议框进行归一化
            rois[...,[0,2]] = (rois[...,[0,2]])/input_shape[1]
            rois[...,[1,3]] = (rois[...,[1,3]])/input_shape[0]
            rois = rois.cpu().numpy()
            box_xy, box_wh = (rois[0][:, 0:2] + rois[0][:, 2:4])/2, rois[0][:, 2:4] - rois[0][:, 0:2]
            #   对建议框进行还原到原图像的scale
            rois[0][:, :4] = self.bbox_util.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
            
        
        
        #   边框厚度
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
        
        #   图像绘制
        
        for i, roi in enumerate(rois[0]):
            top, left, bottom, right = roi

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            
            draw = ImageDraw.Draw(image)

            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline='red')
            del draw

        return image

    def get_FPS(self, image, test_interval):
        
        #   计算输入图片的高和宽
        
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        
        image       = cvtColor(image)
        
        #   给原图像进行resize，resize到短边为600的大小上
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        
        #   添加上batch_size维度 
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.net(images)
                
                #   利用classifier的预测结果对建议框进行解码，获得预测框
                
                results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    
    #   检测图片
    
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        
        #   计算输入图片的高和宽
        
        image_shape = np.array(np.shape(image)[0:2])
        input_shape = get_new_img_size(image_shape[0], image_shape[1])
        
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        
        image       = cvtColor(image)
        
        
        #   给原图像进行resize，resize到短边为600的大小上
        
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        
        #   添加上batch_size维度
        
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
