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



# 确保调用这个脚本时不会执行下面的代码
# make sure the code below is not executed when calling this script
if __name__ == "__main__":

    # 运行测试
    i1 = 2  # 5x5 结构元素
    i2 = 4  # 9x9 结构元素
    input_file = "Exercises_04ab/immed_gray_inv.pgm"
    output_file1 = "Exercises_06ab/immed_gray_inv_ope2clo2.pgm"
    output_file2 = "Exercises_06ab/immed_gray_inv_ope4clo4.pgm"
    output_txt1 = "Exercises_06ab/exercise_06b_output_01.txt"
    output_txt2 = "Exercises_06ab/exercise_06b_output_02.txt"
    compare_file1= "Exercises_06ab/immed_gray_inv_20051123_ope2clo2.pgm"
    compare_file2= "Exercises_06ab/immed_gray_inv_20051123_ope4clo4.pgm"
# 运行不同的 Opening-Closing 交替滤波方法
    exercise_06b_opening_closing(i1, input_file, output_file1, method="numpy")   # NumPy 版
    exercise_06b_opening_closing(i2, input_file, output_file2, method="numpy")  

# 显示图像
    img_original = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    img_filtered1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
    img_filtered2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)
    print(exercise_02b_compare(output_file1, compare_file1,output_txt1))
    print(exercise_02b_compare(output_file2, compare_file2,output_txt2))

    if img_original is not None and img_filtered1 is not None and img_filtered2 is not None:
        cv2.imshow("Original Image", img_original)
        cv2.imshow(f"Opening-Closing (NumPy, i={i1})", img_filtered1)
        cv2.imshow(f"Opening-Closing (OpenCV, i={i2})", img_filtered2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to load images for display.")
