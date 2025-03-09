#将Exercises_15/lena.jpg旋转、缩放、镜像后，使用BRISK特征检测器进行特征匹配
#使用opencv对lena图像进行预处理：
#旋转：cv2.getRotationMatrix2D()、cv2.warpAffine()
#缩放：cv2.resize()
#镜像：cv2.flip()
import numpy as np
import cv2
import matplotlib.pyplot as plt
# 旋转图像
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# 缩放图像
def resize_image(image, scale):
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w*scale), int(h*scale)))

# 镜像图像
def mirror_image(image):
    return cv2.flip(image, 1)

# 把lena.jpg旋转缩放后，与一个室内场景图合成，到一个不同的图片scene.jpg中方便测试


import cv2
import numpy as np

def blend_images(scene_img_path, person_img_path, angle=0, scale_factor=0.3, position=(50, 50)):
    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_COLOR)
    person_img = cv2.imread(person_img_path, cv2.IMREAD_UNCHANGED)

    if scene_img is None or person_img is None:
        raise ValueError("检查图片路径是否正确！")

    h, w = person_img.shape[:2]

    # 旋转并缩放人物图片
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale_factor)
    rotated_scaled_person = cv2.warpAffine(person_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 裁剪掉黑色背景
    gray = cv2.cvtColor(rotated_scaled_person, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w_crop, h_crop = cv2.boundingRect(thresh)
    cropped_person = rotated_scaled_person[y:y+h_crop, x:x+w_crop]

    # 确定插入位置和大小
    scene_h, scene_w = scene_img.shape[:2]
    scale_factor = min(scale_factor, scene_w / w * 0.3, scene_h / h * 0.3)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized_person = cv2.resize(cropped_person, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x, y = position
    new_w = min(new_w, scene_w - x)
    new_h = min(new_h, scene_h - y)

    if resized_person.shape[2] == 4:
        person_rgb = resized_person[:,:,:3]
        alpha_mask = resized_person[:,:,3] / 255.0
    else:
        person_rgb = resized_person
        alpha_mask = np.ones((new_h, new_w))

    scene_crop = scene_img[y:y+new_h, x:x+new_w]

    for c in range(3):
        scene_crop[:,:,c] = (scene_crop[:,:,c]*(1-alpha_mask) + person_rgb[:,:,c]*alpha_mask).astype(np.uint8)

    scene_img[y:y+new_h, x:x+new_w] = scene_crop

    return scene_img


# 加载模板图像
img_template = cv2.imread('Exercises_15/lena.jpg', cv2.IMREAD_COLOR)
if img_template is None:
    raise FileNotFoundError("make sure the image path is correct!")


# 旋转、缩放、镜像图像
img_rotated = rotate_image(img_template, 45)
img_resized = resize_image(img_template, 0.5)
img_mirror = mirror_image(img_template)

# 显示处理后的图像
plt.figure(figsize=(12, 6))
plt.subplot(221)
plt.imshow(img_rotated, cmap='gray')
plt.title('rotated')
plt.subplot(222)
plt.imshow(img_resized, cmap='gray')
plt.title('resized')
plt.subplot(223)
plt.imshow(img_mirror, cmap='gray')
plt.title('flipped')
plt.show()

# 保存处理后的图像
cv2.imwrite('Exercises_15/lena_rotated.jpg', img_rotated)
cv2.imwrite('Exercises_15/lena_resized.jpg', img_resized)
cv2.imwrite('Exercises_15/lena_mirror.jpg', img_mirror)


# 合成图像示例
scene_path = "Exercises_15/OIP.jpeg"  # 室内场景图路径
person_path = "Exercises_15/lena.jpg"  # 人物透明图路径（带 Alpha 透明通道）
result = blend_images(scene_path, person_path, angle=90, scale_factor=0.3, position=(50, 50))

cv2.imwrite("Exercises_15/blended_scene.jpg", result)

# BRISK特征检测与匹配
img_original = img_template
img_blended = result

brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img_template, None)
kp2, des2 = brisk.detectAndCompute(img_blended, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img_template, kp1, img_blended, kp2, matches[:50], None, flags=2)
cv2.imwrite("Exercises_15/matched_img.jpg", matched_img)

cv2.imshow("BRISK Matching", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()