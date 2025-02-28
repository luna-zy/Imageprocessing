import cv2  # 导入OpenCV库


def exercises_02a_thresh(inputfile,outputfile,value,model='manual'):
    # 读取PGM灰度图像 read the PGM image
    image = cv2.imread(inputfile, cv2.IMREAD_GRAYSCALE)
    #value = 100  # 设置阈值


    #####
    # 获取图像宽(w)和高(h) get the image dimensions
    h, w = image.shape


    print(f"Image size: Width={w}, Height={h}")

    # 复制原图，避免修改原始数据 copy the original image to avoid modifying the original data
    image_out1 = image.copy()

    # use loop for manual thresholding
    if model == 'manual':
    # 使用循环进行手动阈值处理 loop for manual thresholding
        for j in range(h):  # 遍历列（高度）    # loop through the columns (height)
            for i in range(w):  # 遍历行（宽度） # loop through the rows (width)
                if image[j, i] >= value:  # 如果像素值大于等于阈值  # if the pixel value is greater than or equal to the threshold
                    image_out1[j, i] = 255  # 设为白色 # set to white
                else:
                    image_out1[j, i] = 0  # 设为黑色 # set to black


    #####
    if model == 'opencv':
        # 使用 OpenCV 自带的 threshold() 进行阈值处理   # use OpenCV's threshold() for thresholding
        # threshold() 函数的返回值有两个，第一个是阈值，第二个是处理后的图像 # threshold() returns two values: the threshold and the processed image
        _, image_out2 = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)

        # 保存处理后的图像 save the processed image
        cv2.imwrite(outputfile, image_out1)  # 保存循环处理的结果 # save the result of the loop
        #cv2.imwrite("Exercises_02ab/cam_74_threshold100.pgm", image_out2)  # 保存 OpenCV 处理的结果 # save the result of OpenCV

    # 显示图像 show the images
    cv2.imshow("origin", image)  # 显示原始图像 # show the original
    


if __name__ == "__main__":
    # 读取PGM灰度图像 read the PGM image

    inputfile = "Exercises_02ab/cam_74.pgm"
    outputfile = "Exercises_02ab/cam_74_threshold100_1.pgm"


    image = cv2.imread("Exercises_02ab/cam_74.pgm", cv2.IMREAD_GRAYSCALE)
    value = 100  # 设置阈值


    #####
    # 获取图像宽(w)和高(h) get the image dimensions
    h, w = image.shape


    print(f"Image size: Width={w}, Height={h}")

    # 复制原图，避免修改原始数据 copy the original image to avoid modifying the original data
    image_out1 = image.copy()

    # 使用循环进行手动阈值处理 loop for manual thresholding
    for j in range(h):  # 遍历列（高度）    # loop through the columns (height)
        for i in range(w):  # 遍历行（宽度） # loop through the rows (width)
            if image[j, i] >= value:  # 如果像素值大于等于阈值  # if the pixel value is greater than or equal to the threshold
                image_out1[j, i] = 255  # 设为白色 # set to white
            else:
                image_out1[j, i] = 0  # 设为黑色 # set to black
            
            
    ######

    # 使用 OpenCV 自带的 threshold() 进行阈值处理   # use OpenCV's threshold() for thresholding
    # threshold() 函数的返回值有两个，第一个是阈值，第二个是处理后的图像 # threshold() returns two values: the threshold and the processed image
    _, image_out2 = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)

    # 保存处理后的图像 save the processed image
    cv2.imwrite("Exercises_02ab/cam_74_threshold100_1.pgm", image_out1)  # 保存循环处理的结果 # save the result of the loop
    cv2.imwrite("Exercises_02ab/cam_74_threshold100.pgm", image_out2)  # 保存 OpenCV 处理的结果 # save the result of OpenCV

    # 显示图像 show the images
    cv2.imshow("origin", image)  # 显示原始图像 # show the original image
    cv2.imshow("output1 (Manual Thresholding)", image_out1)  # 显示手动阈值处理结果 # show the result of manual thresholding
    cv2.imshow("output2 (OpenCV Thresholding)", image_out2)  # 显示OpenCV的阈值处理结果 # show the result of OpenCV thresholding
 
    cv2.waitKey(0)  # 等待按键关闭窗口 # wait for a key press to close the window
    cv2.destroyAllWindows()  # 关闭所有窗口 # close all windows