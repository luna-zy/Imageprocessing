import cv2
import numpy as np
import sys

# 使用 NumPy 实现形态学膨胀
def custom_dilate(image, kernel_size):
    """ 手写形态学膨胀（NumPy 版本）"""
    h, w = image.shape
    pad = kernel_size // 2  # 计算填充大小
    dilated_image = np.copy(image)

    # 在边界填充，以防止索引越界
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    # 遍历图像（忽略填充部分）
    for y in range(h):
        for x in range(w):
            # 取 (2*i+1) x (2*i+1) 局部窗口
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # 取最大值（膨胀操作）
            dilated_image[y, x] = np.max(local_region)

    return dilated_image

# 纯 Python 列表实现形态学膨胀
def manual_dilate(image, kernel_size):
    """ 纯 Python 版形态学膨胀（适用于小图像，速度较慢） """
    h, w = image.shape
    pad = kernel_size // 2  # 计算填充大小
    dilated_image = image.copy()  # 复制原始图像

    # 遍历每个像素点
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            # 获取局部区域
            local_region = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    local_region.append(image[y + ky, x + kx])  # 手动添加每个像素

            # 计算局部区域最大值（膨胀操作）
            dilated_image[y, x] = max(local_region)

    return dilated_image

# 使用 OpenCV 进行形态学膨胀
def cv_dilate(image, kernel_size):
    """ OpenCV 版本形态学膨胀 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=1)

def exercise_03b_dilation(i, input_file, output_file, method="numpy"):
    """ 形态学膨胀处理 """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  # 计算 (2*i+1) x (2*i+1) 结构元素大小
    dilated_image = image.copy()

    # 执行 i 次膨胀
    for _ in range(i):
        if method == "numpy":
            dilated_image = custom_dilate(dilated_image, kernel_size)
        elif method == "list":
            dilated_image = manual_dilate(dilated_image, kernel_size)
        elif method == "opencv":
            dilated_image = cv_dilate(dilated_image, kernel_size)

    # 保存结果
    cv2.imwrite(output_file, dilated_image)
    print(f"Dilation of size {i} applied and saved to {output_file} (Method: {method})")

# 测试膨胀操作
i1 = 1  # 3x3 结构元素
i2 = 2  # 5x5 结构元素
input_file = "Exercises_03ab/immed_gray_inv.pgm"
output_file1 = "Exercises_03ab/immed_gray_inv_dil1.pgm"
output_file2 = "Exercises_03ab/immed_gray_inv_dil2.pgm"

# 运行不同的膨胀方法
exercise_03b_dilation(i1, input_file, output_file1, method="list")  # 纯 Python 列表方式
exercise_03b_dilation(i2, input_file, output_file2, method="list")  # 纯 Python 列表方式

# 显示图像
img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
img_dilated1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
img_dilated2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

if img_original is not None and img_dilated1 is not None and img_dilated2 is not None:
    cv2.imshow("Original Image", img_original)
    cv2.imshow(f"Dilated Image (i={i1})", img_dilated1)
    cv2.imshow(f"Dilated Image (i={i2})", img_dilated2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
