import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 图像预处理：加载 Lena 图像
img = cv2.imread("Exercises_14/lena.jpg", cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("无法找到 lena.jpg, 请确保已安装OpenCV样例图像或自行下载。")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化（灰度图）
equalized_gray = cv2.equalizeHist(img_gray)

# 显示原图与灰度均衡化结果
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img_gray, cmap='gray'), plt.title('Original Gray')
plt.subplot(122), plt.imshow(equalized_gray, cmap='gray'), plt.title('Equalized Gray')
plt.show()

# 2. 提取BGR通道并分别均衡化
channels = cv2.split(img)

equalized_channels = [cv2.equalizeHist(ch) for ch in channels]

# 合并均衡化后的通道
equalized_img = cv2.merge(equalized_channels)
equalized_img_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)

# 显示原图与通道均衡化图
fig, axs = plt.subplots(2, 3, figsize=(15,10))

colors = ('Blue Channel', 'Green Channel', 'Red Channel')
for i, color in enumerate(colors):
    axs[0, i].imshow(channels[i], cmap='gray')
    axs[0, i].set_title(f'Original {color}')
    axs[0, i].axis('off')

    axs[1, i].imshow(equalized_channels[i], cmap='gray')
    axs[1, i].set_title(f'Equalized {color}')
    axs[1, i].axis('off')

plt.show()

# 显示原图和彩色均衡化图像对比
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original RGB')
plt.subplot(122), plt.imshow(equalized_img_rgb), plt.title('Equalized RGB')
plt.show()

# 3. ROI 或对象检测（此处为示例，手动定义ROI）
roi = img_gray[220:400, 220:400]  # Lena脸部ROI示例区域

plt.figure(figsize=(5,5))
plt.imshow(roi, cmap='gray'), plt.title('ROI Example')
plt.show()

# 4. 特征定义与提取（示例：颜色直方图特征）
def color_histogram(img, bins=32):
    hist = cv2.calcHist([img], [0,1,2], None, [bins]*3, [0,256]*3)
    cv2.normalize(hist, hist)
    return hist.flatten()

features_original = color_histogram(img)
features_equalized = color_histogram(equalized_img)

print("原图颜色直方图特征长度:", len(features_original))
print("均衡化图像颜色直方图特征长度:", len(features_equalized))

# 5. 训练与分类示例（此处仅示意，没有实际分类数据集）
from sklearn.neighbors import KNeighborsClassifier

# 假设两个特征向量代表两个不同的类别
X_train = [features_original, features_equalized]
y_train = [0, 1]  # 假设类别标签

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)

# 测试分类（使用原图特征模拟分类）
predicted_class = classifier.predict([features_original])
print(f"分类结果为：{predicted_class[0]}（0为原图类，1为均衡化图类）")
