import os
import cv2 as cv
import numpy as np

# 图像二值化（阈值）
def threshold(file_path):
    for filename in os.listdir(file_path):
        file = os.path.join(file_path,filename)
        img = cv.imdecode(np.fromfile(file,dtype=np.uint8),-1)#这时候就已经是Ndarray了
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, img_after = cv.threshold(gray, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#ret：True或False，代表有没有读到图片  dst：目标图像

        cv.imwrite('test.png',img_after)

def geometric_transformation(file_path):
    for filename in os.listdir(file_path):
        file = os.path.join(file_path, filename)
        img = cv.imdecode(np.fromfile(file, dtype=np.uint8), -1)

        height, width, _ = img.shape
        img_resize = cv.resize(img, (int(0.5 * width), int(0.5 * height))) # 缩放图片大小

        translate = np.float32([[1, 0, 100], [0, 1, 50]])
        translation = cv.warpAffine(img, translate, (width, height))#图片平移 先宽后高

        rotate = cv.getRotationMatrix2D((width/ 2.0, height/ 2.0), 180, 1.0) #第一个参数为原点，除二就是中心，第二个为旋转角度，第三个为缩放大小
        Rotation = cv.warpAffine(img, rotate, (width, height))

        cv.imshow('Rotation', Rotation)
        cv.waitKey(0)

def Wave_filtering(file_path):
    for filename in os.listdir(file_path):
        file = os.path.join(file_path, filename)
        img = cv.imdecode(np.fromfile(file, dtype=np.uint8), -1)

        blur_img = cv.blur(img, (5, 5)) # 均值模糊
        GaussianBlur_img = cv.GaussianBlur(img, (5, 5), 0) #高斯模糊
        median_img = cv.medianBlur(img, 5)# 中值模糊
        bilateral_img = cv.bilateralFilter(img, 9, 75, 75)# 双边模糊