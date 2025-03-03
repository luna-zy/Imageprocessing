import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
from Exercises_03ab.exercise_03a_erosion import custom_erode, manual_erode, cv_erode
from Exercises_03ab.exercise_03b_dilation import custom_dilate, manual_dilate, cv_dilate  
from Exercises_04ab.exercise_04b_closing import custom_closing, manual_closing, cv_closing, exercise_04b_closing
from Exercises_04ab.exercise_04a_opening import custom_opening, manual_opening, cv_opening, exercise_04a_opening


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



if __name__ == "__main__":


    # 运行测试
    i1 = 2  # 5x5 结构元素
    i2 = 4  # 9x9 结构元素
    input_file = "Exercises_04ab/immed_gray_inv.pgm"
    output_file1 = "Exercises_06ab/immed_gray_inv_clo2ope2.pgm"
    output_file2 = "Exercises_06ab/immed_gray_inv_clo4ope4.pgm"
    output_txt1 = "Exercises_06ab/exercise_06a_output_01.txt"
    output_txt2 = "Exercises_06ab/exercise_06a_output_02.txt"
    compare_file1= "Exercises_06ab/immed_gray_inv_20051123_clo2ope2.pgm"
    compare_file2= "Exercises_06ab/immed_gray_inv_20051123_clo4ope4.pgm"

    # 运行不同的 Closing-Opening 交替滤波方法w
    exercise_06a_closing_opening(i1, input_file, output_file1, method="numpy")   # NumPy 版
    exercise_06a_closing_opening(i2, input_file, output_file2, method="numpy")   # NumPy 版

    # 显示图像
    img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
    img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)
    print(exercise_02b_compare(output_file1, compare_file1,output_txt1))
    print(exercise_02b_compare(output_file2, compare_file2,output_txt2))

    if img_original is not None and img_filtered1 is not None and img_filtered2 is not None:
        cv2.imshow("Original Image", img_original)
        cv2.imshow(f"Closing-Opening ( i={i1})", img_filtered1)
        cv2.imshow(f"Closing-Opening ( i={i2})", img_filtered2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to load images for display.")
