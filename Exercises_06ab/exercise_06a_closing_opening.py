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


# 交替滤波：先开运算，再闭运算
def custom_closing_opening(image, kernel_size):
    """ NumPy 实现 Closing-Opening 交替滤波 """
    opened = custom_opening(image, kernel_size)  # 先执行 Opening
    closed_opened = custom_closing(opened, kernel_size)  # 再执行 Closing
    return closed_opened

# 纯 Python 实现 Closing-Opening 交替滤波
def manual_closing_opening(image, kernel_size):
    """ 纯 Python 版 Closing-Opening 交替滤波（适用于小图像，速度较慢） """
    opened = manual_opening(image, kernel_size)  # 先执行 Opening
    closed_opened = manual_closing(opened, kernel_size)  # 再执行 Closing
    return closed_opened

# OpenCV 实现 Closing-Opening 交替滤波
def cv_closing_opening(image, kernel_size):
    """ OpenCV 版本 Closing-Opening 交替滤波 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 先开运算
    closed_opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # 再闭运算
    return closed_opened

def exercise_06a_closing_opening(i, input_file, output_file, method="numpy"):
    """ 交替滤波处理（Closing-Opening） """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  
    filtered_image = image.copy()

    for _ in range(i):
        if method == "numpy":
            filtered_image = custom_closing_opening(filtered_image, kernel_size)
        elif method == "opencv":
            filtered_image = cv_closing_opening(filtered_image, kernel_size)
        elif method == "list": 
            filtered_image = manual_closing_opening(filtered_image, kernel_size)

    cv2.imwrite(output_file, filtered_image)
    print(f"Closing-Opening of size {i} applied and saved to {output_file} (Method: {method})")

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
output_file1 = "Exercises_06ab/immed_gray_inv_clo2ope2.pgm"
output_file2 = "Exercises_06ab/immed_gray_inv_clo4ope4.pgm"

# 运行不同的 Closing-Opening 交替滤波方法w
exercise_06a_closing_opening(i1, input_file, output_file1, method="list")   # NumPy 版
exercise_06a_closing_opening(i2, input_file, output_file2, method="list")   # NumPy 版

# 显示图像
img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

if img_original is not None and img_filtered1 is not None and img_filtered2 is not None:
    cv2.imshow("Original Image", img_original)
    cv2.imshow(f"Closing-Opening ( i={i1})", img_filtered1)
    cv2.imshow(f"Closing-Opening ( i={i2})", img_filtered2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
