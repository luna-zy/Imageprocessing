import cv2
import matplotlib.pyplot as plt

# 读取 Lena 图像
img = cv2.imread("Lena_HistogramEqualization\lena.jpg")

# 检查图像是否正确读取
if img is None:
    raise ValueError("图像加载失败，请确认路径是否正确")

# 图像通道：B, G, R
colors = ('b', 'g', 'r')
channel_names = ['Blue', 'Green', 'Red']

# 创建图像直方图窗口
plt.figure(figsize=(10, 5))
for i, color in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=f'{channel_names[i]} channel')

plt.title("Lena Color Histogram (BGR)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
