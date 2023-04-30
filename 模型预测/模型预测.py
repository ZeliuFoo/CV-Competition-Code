import keras.saving.saved_model.load
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, precision_recall_fscore_support, \
    auc, roc_curve,classification_report
import matplotlib.pyplot as plt

val_dir = 'D:\文献\比赛\模型构建与训练\output/val'
test_dir = 'D:\文献\比赛\模型构建与训练\output/test'

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,target_size=(224,224),batch_size=64,shuffle=False,seed=2077) # set as validation data

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224,224),batch_size=1,shuffle=False,seed=2077) # set as validation data

model = keras.models.load_model('D:\文献\比赛\模型构建与训练\model.h5')

labels = test_generator.labels
classes = test_generator.class_indices
prediction = model.predict(test_generator,verbose=1)
predicted_class = np.argmax(prediction, axis=1)

# average=None,取出每一类的R,P,F1值
recall, precision, f1score, _ = precision_recall_fscore_support(y_true=labels, y_pred=predicted_class, labels=[0,1,2,3,4,5,6,7,8,9], average=None)
print('各类单独F1:', f1score)
print('各类F1取平均：', f1score.mean())
# >>>各类单独F1: [ 0.75        0.66666667  0.5         0.5       ]
# >>>各类F1取平均： 0.604166666667

# y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
# y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
# fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)
#
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, '--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
#
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()


print(classification_report(predicted_class,labels))