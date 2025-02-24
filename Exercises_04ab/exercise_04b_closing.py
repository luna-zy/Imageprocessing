import cv2
import numpy as np
import sys

# NumPy 版本 - 先膨胀再腐蚀
def custom_closing(image, kernel_size):
    """ 手写形态学闭运算（NumPy 版本） """
    # 先执行膨胀
    dilated = custom_dilate(image, kernel_size)
    # 再执行腐蚀
    closed = custom_erode(dilated, kernel_size)
    return closed

# Python List 版本 - 先膨胀再腐蚀
def manual_closing(image, kernel_size):
    """ 纯 Python 版形态学闭运算（适用于小图像，速度较慢） """
    dilated = manual_dilate(image, kernel_size)
    closed = manual_erode(dilated, kernel_size)
    return closed

# OpenCV 版本 - 一步完成闭运算
def cv_closing(image, kernel_size):
    """ OpenCV 版本形态学闭运算 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def exercise_04b_closing(i, input_file, output_file, method="numpy"):
    """ 形态学闭运算处理 """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  # 计算 (2*i+1) x (2*i+1) 结构元素大小
    closed_image = image.copy()

    # 执行 i 次闭运算
    for _ in range(i):
        if method == "numpy":
            closed_image = custom_closing(closed_image, kernel_size)
        elif method == "list":
            closed_image = manual_closing(closed_image, kernel_size)
        elif method == "opencv":
            closed_image = cv_closing(closed_image, kernel_size)

    # 保存结果
    cv2.imwrite(output_file, closed_image)
    print(f"Closing of size {i} applied and saved to {output_file} (Method: {method})")

# 使用 NumPy 实现形态学腐蚀
def custom_erode(image, kernel_size):
    """ 手写形态学腐蚀（NumPy 版本）"""
    h, w = image.shape
    pad = kernel_size // 2  
    eroded_image = np.copy(image)
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=255)

    for y in range(h):
        for x in range(w):
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            eroded_image[y, x] = np.min(local_region)

    return eroded_image

# 使用 NumPy 实现形态学膨胀
def custom_dilate(image, kernel_size):
    """ 手写形态学膨胀（NumPy 版本）"""
    h, w = image.shape
    pad = kernel_size // 2  
    dilated_image = np.copy(image)
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    for y in range(h):
        for x in range(w):
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            dilated_image[y, x] = np.max(local_region)

    return dilated_image

# 纯 Python 列表实现腐蚀
def manual_erode(image, kernel_size):
    h, w = image.shape
    pad = kernel_size // 2  
    eroded_image = image.copy()

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            local_region = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    local_region.append(image[y + ky, x + kx])  
            eroded_image[y, x] = min(local_region)

    return eroded_image

# 纯 Python 列表实现膨胀
def manual_dilate(image, kernel_size):
    h, w = image.shape
    pad = kernel_size // 2  
    dilated_image = image.copy()

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            local_region = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    local_region.append(image[y + ky, x + kx])  
            dilated_image[y, x] = max(local_region)

    return dilated_image

# 测试闭运算操作
i1 = 1  # 3x3 结构元素
i2 = 2  # 5x5 结构元素
input_file = "Exercises_04ab/immed_gray_inv.pgm"
output_file1 = "Exercises_04ab/immed_gray_inv_clo1.pgm"
output_file2 = "Exercises_04ab/immed_gray_inv_clo2.pgm"

# 运行不同的闭运算方法
exercise_04b_closing(i1, input_file, output_file1, method="list")   # NumPy 版
exercise_04b_closing(i2, input_file, output_file2, method="list")  # OpenCV 版

# 显示图像
img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
img_closed1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
img_closed2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

if img_original is not None and img_closed1 is not None and img_closed2 is not None:
    cv2.imshow("Original Image", img_original)
    cv2.imshow(f"Closed Image (list, i={i1})", img_closed1)
    cv2.imshow(f"Closed Image (list, i={i2})", img_closed2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
