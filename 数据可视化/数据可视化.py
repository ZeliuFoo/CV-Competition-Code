import matplotlib.pyplot as plt
import numpy as np

x = np.array(list('01'))
y = np.random.randint(1, 100, 2)

# 作图
# x,y参数：x，y值
# width：宽度比例
# facecolor柱状图里填充的颜色

plt.title('data')

plt.bar(x, y, width=0.5, facecolor='lightblue')
plt.show()