import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 Lena 彩色图像
image = cv2.imread("Lena_HistogramEqualization\lena.jpg")
if image is None:
    raise ValueError("无法读取图像，请检查路径")

# 拆分 B, G, R 通道
b, g, r = cv2.split(image)

# 对每个通道做直方图均衡化
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# 合并为新图像
equalized_img = cv2.merge((b_eq, g_eq, r_eq))

# 显示前后对比（原图 vs 均衡化后）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)), plt.title("Equalized (BGR channels)")
plt.show()

# 显示各通道均衡化前后
channel_names = ['Blue', 'Green', 'Red']
channels_before = [b, g, r]
channels_after = [b_eq, g_eq, r_eq]

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 2, 2*i+1), plt.imshow(channels_before[i], cmap='gray'), plt.title(f'{channel_names[i]} Before')
    plt.subplot(3, 2, 2*i+2), plt.imshow(channels_after[i], cmap='gray'), plt.title(f'{channel_names[i]} After')
plt.tight_layout()
plt.show()
