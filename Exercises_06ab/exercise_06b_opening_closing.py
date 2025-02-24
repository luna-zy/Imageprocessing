import cv2
import numpy as np
import sys

def manual_opening(image, kernel_size):
    """ Opening operation using pure Python lists """

    eroded = manual_erode(image, kernel_size)
    opened = manual_dilate(eroded, kernel_size)
    return opened

def manual_closing(image, kernel_size):
    """ 纯 Python 版形态学闭运算（适用于小图像，速度较慢） """
    dilated = manual_dilate(image, kernel_size)
    closed = manual_erode(dilated, kernel_size)
    return closed


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


# 交替滤波：先闭运算，再开运算
def custom_opening_closing(image, kernel_size):
    """ NumPy 实现 Opening-Closing 交替滤波 """
    closed = custom_closing(image, kernel_size)  # 先执行 Closing
    opened_closed = custom_opening(closed, kernel_size)  # 再执行 Opening
    return opened_closed

# 纯 Python 实现 Opening-Closing 交替滤波
def manual_opening_closing(image, kernel_size):
    """ 纯 Python 版 Opening-Closing 交替滤波（适用于小图像，速度较慢） """
    closed = manual_closing(image, kernel_size)  # 先执行 Closing
    opened_closed = manual_opening(closed, kernel_size)  # 再执行 Opening
    return opened_closed

# OpenCV 实现 Opening-Closing 交替滤波
def cv_opening_closing(image, kernel_size):
    """ OpenCV 版本 Opening-Closing 交替滤波 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # 先闭运算
    opened_closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 再开运算
    return opened_closed

def exercise_06b_opening_closing(i, input_file, output_file, method="numpy"):
    """ 交替滤波处理（Opening-Closing） """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  
    filtered_image = image.copy()

    for _ in range(i):
        if method == "numpy":
            filtered_image = custom_opening_closing(filtered_image, kernel_size)
        elif method == "opencv":
            filtered_image = cv_opening_closing(filtered_image, kernel_size)
        elif method == "list":
            filtered_image = manual_opening_closing(filtered_image, kernel_size)

    cv2.imwrite(output_file, filtered_image)
    print(f"Opening-Closing of size {i} applied and saved to {output_file} (Method: {method})")

# NumPy 方式 - 形态学开运算
def custom_opening(image, kernel_size):
    """ 先腐蚀再膨胀（手写 NumPy 实现）"""
    eroded = custom_erode(image, kernel_size)
    opened = custom_dilate(eroded, kernel_size)
    return opened

# NumPy 方式 - 形态学闭运算
def custom_closing(image, kernel_size):
    """ 先膨胀再腐蚀（手写 NumPy 实现）"""
    dilated = custom_dilate(image, kernel_size)
    closed = custom_erode(dilated, kernel_size)
    return closed

# NumPy 方式 - 形态学腐蚀
def custom_erode(image, kernel_size):
    """ NumPy 实现形态学腐蚀 """
    h, w = image.shape
    pad = kernel_size // 2
    eroded_image = np.copy(image)
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=255)

    for y in range(h):
        for x in range(w):
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            eroded_image[y, x] = np.min(local_region)

    return eroded_image

# NumPy 方式 - 形态学膨胀
def custom_dilate(image, kernel_size):
    """ NumPy 实现形态学膨胀 """
    h, w = image.shape
    pad = kernel_size // 2
    dilated_image = np.copy(image)
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    for y in range(h):
        for x in range(w):
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            dilated_image[y, x] = np.max(local_region)

    return dilated_image

# 运行测试
i1 = 2  # 5x5 结构元素
i2 = 4  # 9x9 结构元素
input_file = "Exercises_04ab/immed_gray_inv.pgm"
output_file1 = "Exercises_06ab/immed_gray_inv_ope2clo2.pgm"
output_file2 = "Exercises_06ab/immed_gray_inv_ope4clo4.pgm"

# 运行不同的 Opening-Closing 交替滤波方法
exercise_06b_opening_closing(i1, input_file, output_file1, method="list")  
exercise_06b_opening_closing(i2, input_file, output_file2, method="list")  

# 显示图像
img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

if img_original is not None and img_filtered1 is not None and img_filtered2 is not None:
    cv2.imshow("Original Image", img_original)
    cv2.imshow(f"Opening-Closing (NumPy, i={i1})", img_filtered1)
    cv2.imshow(f"Opening-Closing (OpenCV, i={i2})", img_filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
