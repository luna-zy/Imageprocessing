import cv2  # 导入OpenCV库

# 读取PGM灰度图像
image = cv2.imread("Exercises_02ab/cam_74.pgm", cv2.IMREAD_GRAYSCALE)
value = 100  # 设置阈值


#####
# 获取图像宽(w)和高(h)
h, w = image.shape


print(f"Image size: Width={w}, Height={h}")

# 复制原图，避免修改原始数据
image_out1 = image.copy()

# 使用循环进行手动阈值处理
for j in range(h):  # 遍历列（高度）
    for i in range(w):  # 遍历行（宽度）
        if image[j, i] >= value:  # 如果像素值大于等于阈值
            image_out1[j, i] = 255  # 设为白色
        else:
            image_out1[j, i] = 0  #
            
            
######

# 使用 OpenCV 自带的 threshold() 进行阈值处理
_, image_out2 = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)

# 保存处理后的图像
cv2.imwrite("Exercises_02ab/cam_74_threshold100_1.pgm", image_out1)  # 保存循环处理的结果
cv2.imwrite("Exercises_02ab/cam_74_threshold100.pgm", image_out2)  # 保存 OpenCV 处理的结果

# 显示图像
cv2.imshow("原图", image)  # 显示原始图像
cv2.imshow("output1 (Manual Thresholding)", image_out1)  # 显示手动阈值处理结果
cv2.imshow("output2 (OpenCV Thresholding)", image_out2)  # 显示OpenCV的阈值处理结果

cv2.waitKey(0)  # 等待按键关闭窗口
cv2.destroyAllWindows()  # 关闭所有窗口