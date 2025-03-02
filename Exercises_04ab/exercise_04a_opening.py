import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
from Exercises_03ab.exercise_03a_erosion import custom_erode, manual_erode, cv_erode
from Exercises_03ab.exercise_03b_dilation import custom_dilate, manual_dilate, cv_dilate  
# NumPy 
def custom_opening(image, kernel_size):
    """ Opening operation using NumPy """

    # 先执行腐蚀
    eroded = custom_erode(image, kernel_size)
    # 再执行膨胀
    opened = custom_dilate(eroded, kernel_size)
    return opened

# Python List 
def manual_opening(image, kernel_size):
    """ Opening operation using pure Python lists """

    eroded = manual_erode(image, kernel_size)
    opened = manual_dilate(eroded, kernel_size)
    return opened

# OpenCV 版本 - 一步完成开运算
def cv_opening(image, kernel_size):
    """ OpenCV version of morphological opening """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def exercise_04a_opening(i, input_file, output_file, method="numpy"):
    """ Opening operation """

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read {input_file}")
        sys.exit(1)
    
    kernel_size = 2 * i + 1  # 计算 (2*i+1) x (2*i+1) 结构元素大小
    opened_image = image.copy()

    # 执行 i 次开运算
    for _ in range(i):
        if method == "numpy":
            opened_image = custom_opening(opened_image, kernel_size)
        elif method == "list":
            opened_image = manual_opening(opened_image, kernel_size)
        elif method == "opencv":
            opened_image = cv_opening(opened_image, kernel_size)

    # 保存结果
    cv2.imwrite(output_file, opened_image)
    print(f"Opening of size {i} applied and saved to {output_file} (Method: {method})")

if __name__ == "__main__":
    # 测试开运算操作
    i1 = 1  # 3x3 结构元素
    i2 = 2  # 5x5 结构元素
    input_file = "Exercises_04ab/immed_gray_inv.pgm"
    output_file1 = "Exercises_04ab/immed_gray_inv_ope1.pgm"
    output_file2 = "Exercises_04ab/immed_gray_inv_ope2.pgm"
    output_txt1 = "Exercises_04ab/exercise_04a_output_01.txt"
    output_txt2 = "Exercises_04ab/exercise_04a_output_02.txt"
    # 运行不同的开运算方法
    exercise_04a_opening(i1, input_file, output_file1, method="numpy")   # NumPy 版
    exercise_04a_opening(i2, input_file, output_file2, method="numpy")  # OpenCV 版

    # 显示图像
    img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img_opened1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
    img_opened2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)
    print(exercise_02b_compare(output_file1, "Exercises_04ab\immed_gray_inv_20051123_ope1.pgm",output_txt1))
    print(exercise_02b_compare(output_file2, "Exercises_04ab\immed_gray_inv_20051123_ope2.pgm",output_txt2))

    if img_original is not None and img_opened1 is not None and img_opened2 is not None:
        cv2.imshow("Original Image", img_original)
        cv2.imshow(f"Opened Image (i={i1})", img_opened1)
        cv2.imshow(f"Opened Image (i={i2})", img_opened2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to load images for display.")
