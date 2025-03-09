import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载模板图像与目标图像
img_template = cv2.imread('Exercises_15\shuchuizi.png', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('Exercises_15/input1.png', cv2.IMREAD_GRAYSCALE)

# 检查是否成功加载
if img_template is None or img_scene is None:
    raise FileNotFoundError("请确保图片路径正确！")

# 步骤1：创建BRISK特征检测器
brisk = cv2.BRISK_create()

# 步骤2：检测关键点与计算描述符
kp_template, des_template = brisk.detectAndCompute(img_template, None)
kp_scene, des_scene = brisk.detectAndCompute(img_scene, None)

# 步骤3：创建暴力匹配器（汉明距离，适合二进制特征）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# 使用knnMatch进行特征匹配
matches = bf.knnMatch(des_template, des_scene, k=2)

# 步骤4：使用Lowe比值测试过滤匹配
good_matches = []
ratio_thresh = 0.75
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# 显示匹配结果
img_matches = cv2.drawMatches(img_template, kp_template, img_scene, kp_scene, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('BRISK 特征匹配结果')
plt.show()

# 步骤5：基于匹配点定位对象
if len(good_matches) > 10:
    # 提取匹配点的坐标
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # 使用RANSAC算法计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 获得模板图像尺寸
    h, w = img_template.shape
    
    # 模板图像的四个角点坐标
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    
    # 映射模板的角点到目标图像中
    dst = cv2.perspectiveTransform(pts, H)

    # 在场景图像上绘制检测结果
    img_detected = cv2.polylines(cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR), 
                                 [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)

    plt.figure(figsize=(8,6))
    plt.imshow(img_detected)
    plt.title('对象检测结果')
    plt.show()
else:
    print("匹配特征点不足，无法准确定位对象。")
