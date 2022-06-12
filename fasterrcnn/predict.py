
import time

import cv2
import numpy as np
from PIL import Image

from frcnnpre import FRCNN
import os



if __name__ == "__main__":
    os.environ['DISPLAY'] = ':0'
    frcnn = FRCNN()
    
    mode = "predict"
    # 是否裁剪、计数
    crop            = False
    count           = False
    
    fps_image_path  = "img/1.jpg"
    
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = frcnn.detect_image(image, crop = crop, count = count)
                #r_image.show()
                r_image.save('./img/res1i.jpg')