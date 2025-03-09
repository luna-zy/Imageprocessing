import cv2
import numpy as np
import matplotlib.pyplot as plt


# load the template image and scene image
img_template = cv2.imread('Exercises_15/lena.jpg', cv2.IMREAD_COLOR_RGB)
img_scene_mirror = cv2.imread('Exercises_15/lena_mirror.jpg', cv2.IMREAD_COLOR_RGB)
img_scene_rotated = cv2.imread('Exercises_15/lena_rotated.jpg', cv2.IMREAD_COLOR_RGB)
img_scene_resized = cv2.imread('Exercises_15/lena_resized.jpg', cv2.IMREAD_COLOR_RGB)
img_scene_blended = cv2.imread('Exercises_15/blended_scene.jpg', cv2.IMREAD_COLOR_RGB)


# BRISK 特征检测与匹配
brisk = cv2.BRISK_create()

def match_images(img1, img2, title):
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_threshold = 300  # 设定一个匹配阈值
    match_success = len(matches) > match_threshold
    
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title(f"{title} - {'Matched' if match_success else 'Not Matched'}")
    plt.show()

    print(f"{title}: {'success' if match_success else 'fail'}, point: {len(matches)}")

# 进行特征匹配
match_images(img_template, img_scene_mirror, "BRISK Matching - Mirror")
match_images(img_template, img_scene_rotated, "BRISK Matching - Rotated")
match_images(img_template, img_scene_resized, "BRISK Matching - Resized")
match_images(img_template, img_scene_blended, "BRISK Matching - Blended")

# 判断是否匹配成功
