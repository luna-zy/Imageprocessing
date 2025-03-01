import sys
import os
import numpy as np
import cv2

#function to calculate the infimum of two images
def exercise_02c_inf(input_file1, input_file2, output_file):
    """ 计算两幅 PGM 图像的下确界（inf） """
    image1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Unable to read one of the images.")
        sys.exit(1)

    # 计算逐像素最小值
    inf_image = np.minimum(image1, image2)

    # 保存结果
    cv2.imwrite(output_file, inf_image)
    print(f"Infimum image saved to {output_file}")


# 运行测试
input_file1 = "Exercises_02c_1/image1.pgm"
input_file2 = "Exercises_02c_1/image2.pgm"
output_sup = "Exercises_02c_1/image1_sup_image2.pgm"
output_inf = "Exercises_02c_1/image1_inf_image2.pgm"


# 计算 `inf`（下确界）
exercise_02c_inf(input_file1, input_file2, output_inf)

# 显示图像
img1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
#img_sup = cv2.imread(output_sup, cv2.IMREAD_GRAYSCALE)
img_inf = cv2.imread(output_inf, cv2.IMREAD_GRAYSCALE)

exercise_02c_inf(input_file1, input_file2, output_inf)

# 显示图像
img1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
#img_sup = cv2.imread(output_sup, cv2.IMREAD_GRAYSCALE)
img_inf = cv2.imread(output_inf, cv2.IMREAD_GRAYSCALE)

if img1 is not None and img2 is not None and img_inf is not None:
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    #cv2.imshow("Supremum (Max)", img_sup)
    cv2.imshow("Infimum (Min)", img_inf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")