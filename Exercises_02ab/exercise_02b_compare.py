import cv2 

def exercise_02b_compare(input_file1, input_file2):
    """ 比较两幅 PGM 图像是否相同 """
    image1 = cv2.imread(input_file1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(input_file2, cv2.IMREAD_GRAYSCALE)
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    flag = 0  # 默认图像不同

    if (h1, w1) == (h2, w2):  # 如果尺寸相同
        if (image1 == image2).all():  # 判断所有像素是否相同
            flag = 1  # 设为相同

    return flag


if __name__ == "__main__":

    # 读取两张 PGM 图像

    image1 = cv2.imread("Exercises_02ab/cam_74_threshold100.pgm", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("Exercises_02ab/cam_74.pgm", cv2.IMREAD_GRAYSCALE)

    # 获取图像尺寸
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    flag = 0  # 默认图像不同

    if (h1, w1) == (h2, w2):  # 如果尺寸相同
        if (image1 == image2).all():  # 判断所有像素是否相同
            flag = 1  # 设为相同

    # 输出结果
    print(flag)

    # 将结果写入文件
    with open("Exercises_02ab/exercise_02b_output_01.txt", "w") as f:
        f.write(str(flag))
    # imshow()
    cv2.imshow("image1", image1)
    cv2.imshow("image2", image2)

    cv2.waitKey(0)  # wait for a key press to close the window
    cv2.destroyAllWindows()  # close all windows