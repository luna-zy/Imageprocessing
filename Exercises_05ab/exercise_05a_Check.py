import cv2
import numpy as np
import sys

# 形态学开运算（已在 Exercise 04a 实现）
def exercise_04a_opening(i, input_file, output_file, method="numpy"):
    """ 形态学开运算处理 """
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  
    opened_image = image.copy()

    for _ in range(i):
        if method == "numpy":
            opened_image = custom_opening(opened_image, kernel_size)
        elif method == "opencv":
            opened_image = cv_opening(opened_image, kernel_size)

    cv2.imwrite(output_file, opened_image)
    print(f"Opening of size {i} applied and saved to {output_file} (Method: {method})")

# NumPy 方式 - 形态学开运算
def custom_opening(image, kernel_size):
    """ 先腐蚀再膨胀（手写 NumPy 实现）"""
    eroded = custom_erode(image, kernel_size)
    opened = custom_dilate(eroded, kernel_size)
    return opened

# OpenCV 方式 - 形态学开运算
def cv_opening(image, kernel_size):
    """ 使用 OpenCV 直接执行开运算 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

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

# 比较两张图片是否相同
def compare_images(image1_path, image2_path, output_file):
    """ 比较两张图像是否相同 """
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Unable to read one of the images.")
        sys.exit(1)

    # 计算图像差异
    difference = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    diff_sum = np.sum(difference)

    if diff_sum == 0:
        result = "Images are identical."
    else:
        result = f"Images are different. Total pixel difference: {diff_sum}"

    # 保存比较结果
    with open(output_file, "w") as f:
        f.write(result + "\n")

    print(result)
    print(f"Comparison result saved to {output_file}")

# 运行测试
i = 1  # 使用 3x3 结构元素
input_file = "Exercises_04ab/immed_gray_inv.pgm"
output_file1 = "Exercises_05ab/exercise_04a_output_01a.pgm"
output_file2 = "Exercises_05ab/exercise_04a_output_02a.pgm"
compare_output = "Exercises_05ab/exercise_02b_output_01a.txt"

# 先执行一次 Opening
exercise_04a_opening(i, input_file, output_file1, method="list")

# 再次对输出图像执行 Opening
exercise_04a_opening(i, output_file1, output_file2, method="list")

# 比较两张图像是否相同
compare_images(output_file1, output_file2, compare_output)
