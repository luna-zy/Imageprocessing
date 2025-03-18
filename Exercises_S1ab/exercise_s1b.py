import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# 区域生长算法 - 扫描整个图像并标记不同区域
def region_growing(image_path, threshold=10, connectivity=4):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法读取输入图像，请检查路径")

    h, w = image.shape
    segmented = np.zeros((h, w), dtype=np.int32)
    region_count = 0  # 记录区域数量
    colors = []  # 存储每个区域的颜色

    # 连接方式
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity 只能是 4 或 8")

    # 逐像素扫描图像，寻找未标记的区域
    for y in range(h):
        for x in range(w):
            if segmented[y, x] == 0:  # 未标记的像素
                region_count += 1
                seed_value = image[y, x]
                queue = [(x, y)]
                segmented[y, x] = region_count
                region_color = np.random.randint(0, 255, (3,)).tolist()  # 生成随机颜色
                colors.append(region_color)

                while queue:
                    sx, sy = queue.pop(0)
                    for dx, dy in neighbors:
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < w and 0 <= ny < h and segmented[ny, nx] == 0:
                            if abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                                segmented[ny, nx] = region_count
                                queue.append((nx, ny))

    # 生成彩色分割图
    segmented_color = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if segmented[y, x] > 0:
                segmented_color[y, x] = colors[segmented[y, x] - 1]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(segmented_color), plt.title(f'Region Growing Result ({region_count} Regions)')
    plt.show()

    return segmented_color

# 示例调用
region_growing("TestImages\TestImages\particles.png", threshold=15, connectivity=8)
