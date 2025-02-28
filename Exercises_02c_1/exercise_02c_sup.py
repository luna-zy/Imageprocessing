import cv2
import numpy as np
import sys

# 计算上确界（supremum）
#function of calculating the supremum of two images
def exercise_02c_sup(input_file1, input_file2, output_file):
    """ 计算两幅 PGM 图像的上确界（sup） """
    image1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        print("Error: Unable to read one of the images.")
        sys.exit(1)

    # 计算逐像素最大值
    sup_image = np.maximum(image1, image2)

    # 保存结果
    cv2.imwrite(output_file, sup_image)
    print(f"Supremum image saved to {output_file}")

# test
input_file1 = "Exercises_02c_1/image1.pgm"
input_file2 = "Exercises_02c_1/image2.pgm"
output_sup = "Exercises_02c_1/image1_sup_image2.pgm"
output_inf = "Exercises_02c_1/image1_inf_image2.pgm"

#  `sup`
exercise_02c_sup(input_file1, input_file2, output_sup)

# display images
img1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
img_sup = cv2.imread(output_sup, cv2.IMREAD_GRAYSCALE)


if img1 is not None and img2 is not None and img_sup is not None:
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.imshow("Supremum (Max)", img_sup)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load images for display.")
