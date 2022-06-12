import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import scipy.signal
import random

from frcnnpre import FRCNN
from utils.utils import get_classes
from utils.utils_map import get_map
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shutil

if __name__ == "__main__":
    
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅计算map，已计算过ground truth，不再重复计算。
    map_mode        = 0
    
    classes_path    = 'model_data/voc_classes.txt'
    
    #   MINOVERLAP用于指定想要获得的mAP0.x
    MINOVERLAP      = 0.5
    #   是否开启VOC_map计算的可视化
    map_vis         = False
    #数据集目录
    VOCdevkit_path  = 'VOCdevkit'

    #指定是否删除之前预测的结果

    del_predetection = True
    
    #   结果输出的文件夹，默认为map_out
    map_out_path    = 'map_out'

    model_dir = 'logs'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    image_ids = random.sample(image_ids,1000)

    if del_predetection:
        shutil.rmtree(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    model_list = [model for model in os.listdir(model_dir) if '.pth' in model]
    model_list.sort()
    
    maplist = []

    if map_mode == 0:

        shutil.rmtree(os.path.join(map_out_path, 'ground-truth'))
        
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth'))

        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    for model in model_list:
        print("Load model of "+model[:5])
        frcnn = FRCNN(model_path = os.path.join(model_dir, model),confidence = 0.01, nms_iou = 0.5)
        print("Load model done.")

        print("Get predict result.")
        if not os.path.exists(map_out_path+'/detection-results/'+model[:5]) or (len(os.listdir(map_out_path+'/detection-results/'+model[:5])) != len(image_ids)):
            for image_id in tqdm(image_ids):
                image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                if map_vis:
                    image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
                model_results = map_out_path+'/detection-results/'+model[:5]
                if not os.path.exists(model_results):
                    os.makedirs(model_results)
                frcnn.get_map_txt(image_id, image, class_names, model_results)
        print("Get predict result done.")

        print("Get map.")
        map = get_map(True,MINOVERLAP, True, model[:5], path = map_out_path)
        print("Get map done for "+model[:5])
        maplist.append(map)
    
        iters = range(len(maplist))

        plt.figure()
        plt.plot(iters, maplist, 'red', linewidth = 2, label='Test mAP')
        
        try:
            if len(maplist) < 25:
                num = 5
            else:
                num = 15    
            plt.plot(iters, scipy.signal.savgol_filter(maplist, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth test mAP')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(map_out_path, "epoch_testmAP.png"))
        plt.cla()
        plt.close("all")


    