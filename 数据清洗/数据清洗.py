import os
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def filter_bad_blured_channel(file_pathname):
    """
    一个循环读取，先检测是否损坏，然后检测通道数，然后模糊检测，最后再检测相似度
    """
    for filename in os.listdir(file_pathname):
        file_path = os.path.join(file_pathname,filename)
        # if filename.endswith('.jpg'):
        #     os.remove(file_path)

        img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        try:
            img.shape
            #检测是否为损坏文件，因为如果没有读取进入，将返回None
        except:
            os.remove(file_path)

        image_var = cv.Laplacian(img, cv.CV_64F).var()  # 检测模糊度是否低于某个值

        if img.ndim == 2:
            os.remove(file_path)#检测是否为单通道图片
        elif image_var < 100:
            #如果过于模糊，删除图片
            os.remove(file_path)

def filter_similarity(file_pathname):
    filename = os.listdir(file_pathname)
    filter_list = []
    for i in range(len(filename)):
        try:
            img1_file_path = os.path.join(file_pathname,filename[i])
            img2_file_path = os.path.join(file_pathname, filename[i+1])

            img1 = cv.imdecode(np.fromfile(img1_file_path, dtype=np.uint8), -1)
            img2 = cv.imdecode(np.fromfile(img2_file_path, dtype=np.uint8), -1)

            if similarity(img1,img2):
                filter_list.append(img2_file_path)
        except:
            print("读取完毕")

    for remove_name in filter_list:
        os.remove(remove_name)

def similarity(img1,img2):
    img1 = cv.resize(img1,[150, 150])
    img2 = cv.resize(img2,[150,150])

    img1 = cv.normalize(img1,img1,0, 1, cv.NORM_MINMAX, -1)
    img2 = cv.normalize(img2,img2,0, 1, cv.NORM_MINMAX, -1)

    ssim = compare_ssim(img1, img2, multichannel=True)

    if ssim >= 0.98:
        return True
    else:
        return False