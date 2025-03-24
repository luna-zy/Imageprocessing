import cv2
import numpy as np
import os

# 创建输出目录
output_dir = "output_masks"
os.makedirs(output_dir, exist_ok=True)

# 子区域处理函数（bounding box内）
def process_bounding_box_subregion(image, x, y, w, h):
    
    sub_img = image[y:y+h, x:x+w]

    # 转为灰度图
    gray = cv2.cvtColor(sub_img, cv2.COLOR_RGB2GRAY) if len(sub_img.shape) == 3 else sub_img.copy()

    # 高斯模糊 + OTSU 阈值化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed

# 对单张图片的全部bbox生成整体mask
def generate_mask_with_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    print(img.shape)  # 打印图像尺寸 (高, 宽, 通道)

    if img is None:
        raise FileNotFoundError(f"图片加载失败，请检查路径：{image_path}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        # 检查边界框范围是否在图像内
        x, y, bw, bh = box['x'], box['y'], box['w'], box['h']
        if x+bw > w or y+bh > h:
            raise ValueError(f"Bounding Box 超出图像范围：{box}")

        sub_mask = process_bounding_box_subregion(img, x, y, bw, bh)
        mask[y:y+bh, x:x+bw] = sub_mask

    return img, mask


# 叠加mask与原图
def overlay_mask_on_image(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[mask == 255] = [0, 255, 0]  # 绿色高亮
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlay

# 处理图像
image_infos = {
    "Mammo001": {
        "path": "Exercises_seg/Mammo001.png",
        "boxes": [{'x': 1116, 'y': 1724, 'w': 218, 'h': 105}]
    },
    "Mammo002": {
        "path": "Exercises_seg/Mammo002.png",
        "boxes": [
            {'x': 267, 'y': 488, 'w': 64, 'h': 64},
            {'x': 169, 'y': 384, 'w': 93, 'h': 85}
        ]
    }
}

# 批量处理图片
for name, info in image_infos.items():
    img, mask = generate_mask_with_boxes(info["path"], info["boxes"])
    overlay = overlay_mask_on_image(img, mask)

    # 保存结果
    mask_path = os.path.join(output_dir, f"{name}_mask.png")
    overlay_path = os.path.join(output_dir, f"{name}_overlay.png")
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)

    print(f"处理完毕: {name}")
    print(f"Mask保存至: {mask_path}")
    print(f"叠加结果保存至: {overlay_path}")

print("所有图像处理完成！")
