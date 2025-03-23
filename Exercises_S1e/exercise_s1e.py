import cv2
import numpy as np
import matplotlib.pyplot as plt

def fast_watershed_test(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理（二值化 + 去噪）
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去噪（开运算）
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换 + 前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 未知区域
    #subtract() is used to subtract the pixel values of two images
    
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 生成 markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 所有 marker 加 1，背景是 1
    markers[unknown == 255] = 0  # 未知区域设为 0

    # 应用分水岭算法
    img_watershed = img.copy()
    cv2.watershed(img_watershed, markers)

    # 用红线标出边界（边界像素值是 -1）
    img_watershed[markers == -1] = [0, 0, 255]

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(markers, cmap='jet')
    plt.title("Markers")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB))
    plt.title("Watershed Result")

    plt.tight_layout()
    plt.show()

# 测试运行
fast_watershed_test("TestImages/TestImages/coffee_grains.jpg")
