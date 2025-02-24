import cv2
import numpy as np
import sys

# 使用 NumPy 实现形态学腐蚀
def custom_erode(image, kernel_size):
    """ 手写形态学腐蚀（NumPy 版本）"""
    h, w = image.shape
    pad = kernel_size // 2  # 计算填充大小
    eroded_image = np.copy(image)

    # 在边界填充，以防止索引越界
    padded_image = np.pad(image, pad_width=pad, mode='constant', constant_values=255)

    # 遍历图像（忽略填充部分）
    for y in range(h):
        for x in range(w):
            # 取 (2*i+1) x (2*i+1) 局部窗口
            local_region = padded_image[y:y + kernel_size, x:x + kernel_size]
            # 取最小值（腐蚀操作）
            eroded_image[y, x] = np.min(local_region)

    return eroded_image

# 纯 Python 列表实现形态学腐蚀
def manual_erode(image, kernel_size):
    """ 纯 Python 版形态学腐蚀（适用于小图像，速度较慢） """
    h, w = image.shape
    pad = kernel_size // 2  # 计算填充大小
    eroded_image = image.copy()  # 复制原始图像

    # 遍历每个像素点
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            # 获取局部区域
            local_region = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    local_region.append(image[y + ky, x + kx])  # 手动添加每个像素

            # 计算局部区域最小值（腐蚀操作）
            eroded_image[y, x] = min(local_region)

    return eroded_image

# 使用 OpenCV 进行形态学腐蚀
def cv_erode(image, kernel_size):
    """ OpenCV 版本形态学腐蚀 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, kernel, iterations=1)

def exercise_03a_erosion(i, input_file, output_file, method="numpy"):
    """ 形态学腐蚀处理 """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  # 计算 (2*i+1) x (2*i+1) 结构元素大小
    eroded_image = image.copy()

    # 执行 i 次腐蚀
    for _ in range(i):
        if method == "numpy":
            eroded_image = custom_erode(eroded_image, kernel_size)
        elif method == "list":
            eroded_image = manual_erode(eroded_image, kernel_size)
        elif method == "opencv":
            eroded_image = cv_erode(eroded_image, kernel_size)

    # 保存结果
    cv2.imwrite(output_file, eroded_image)
    print(f"Erosion of size {i} applied and saved to {output_file} (Method: {method})")

# 测试腐蚀操作
i1 = 1  # 3x3 结构元素
i2 = 2  # 5x5 结构元素
input_file = "Exercises_03ab/immed_gray_inv.pgm"
output_file1 = "Exercises_03ab/immed_gray_inv_ero1.pgm"
output_file2 = "Exercises_03ab/immed_gray_inv_ero2.pgm"

# 运行不同的腐蚀方法

exercise_03a_erosion(i1, input_file, output_file1, method="list")  

exercise_03a_erosion(i2, input_file, output_file2, method="list")   


# 显示图像
img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
img_eroded1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
img_eroded2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)

if img_original is not None and img_eroded1 is not None and img_eroded2 is not None:
    cv2.imshow("Original Image", img_original)
    cv2.imshow(f"Eroded Image (i={i1})", img_eroded1)
    cv2.imshow(f"Eroded Image (i={i2})", img_eroded2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
