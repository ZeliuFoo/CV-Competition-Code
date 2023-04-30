import os
import pandas as pd
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

def load_pic(file_pathname): # 数据加载
    pic_array = []
    label_array = []

    for filename in os.listdir(file_pathname):
        filenames = os.path.join(file_pathname,filename)
        pic = cv.imdecode(np.fromfile(filenames,dtype=np.uint8),-1)
        label = filename.split('.')[0]

        pic_array.append(pic)
        label_array.append(label)

    label_array = pd.DataFrame(label_array)
    label_array = label_array.replace({'dog':0,'cat':1})

    return np.array(pic_array), np.array(label_array)

data, label = load_pic('D:\文献\比赛\数据可视化/train')
np.save('data',data)
np.save('label',label)

x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2,random_state=1337)
