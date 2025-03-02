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

# 运行测试
i = 1  # 使用 3x3 结构元素
input_file = "Exercises_04ab/immed_gray_inv.pgm"
output_file1 = "Exercises_05ab/exercise_04b_output_01b.pgm"
output_file2 = "Exercises_05ab/exercise_04b_output_02b.pgm"
compare_output = "Exercises_05ab/exercise_02b_output_01b.txt"

# 先执行一次 Closing
exercise_04b_closing(i, input_file, output_file1, method="numpy")

# 再次对输出图像执行 Closing
exercise_04b_closing(i, output_file1, output_file2, method="numpy")

# 比较两张图像是否相同
exercise_02b_compare(output_file1, output_file2, compare_output)
