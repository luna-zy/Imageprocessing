import cv2
import numpy as np
import sys
import os
# prviate libraries
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Exercises_06ab.exercise_06a_closing_opening import exercise_06a_closing_opening
from Exercises_02ab.exercise_02b_compare import exercise_02b_compare
# 形态学 Closing-Opening 交替滤波

# 读取图像
input_file = "Exercises_04ab/immed_gray_inv.pgm"
output_file1 = "Exercises_09a/exercise_09a_output_01.pgm"
output_file2 = "Exercises_09a/exercise_09a_output_02.pgm"
i1 = 1  # 3x3 
image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

# 计算一次 Closing-Opening
exercise_06a_closing_opening(i1, input_file, output_file1, method="numpy")

# 计算两次 Closing-Opening
exercise_06a_closing_opening(i1, input_file, output_file2, method="numpy")

# 计算像素差异
flag=exercise_02b_compare(output_file1, output_file2, "Exercises_09a/exercise_09a_output_01.txt")

# 结果检查
if flag == 1:
    print("Closing-Opening is idempotent: result_1 == result_2")
else:
    print(f"Closing-Opening is NOT idempotent")

result_1 = cv2.imread(output_file1, cv2.IMREAD_GRAYSCALE)
result_2 = cv2.imread(output_file2, cv2.IMREAD_GRAYSCALE)
# 显示图像
cv2.imshow("Original", image)
cv2.imshow("Closing-Opening Once", result_1)
cv2.imshow("Closing-Opening Twice", result_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
