import numpy as np 
import cv2 

import numpy as np
import cv2
import matplotlib.pyplot as plt

def color_clustering(image_path, k=4,attempts =6,stop_conds = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.90) , color_space='BGR'):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败")

    # 根据指定颜色空间转换图像
    if color_space == 'HSV':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == 'Lab':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif color_space == 'RGB':
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_converted = img.copy()  # 默认 BGR

    # 将图像重塑为 Nx3 的像素数据
    pixel_data = np.float32(img_converted.reshape((-1, 3)))

    # 设置 KMeans 参数
    stop_conds=  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.90)
    attempts = 10

    # 执行 KMeans 聚类
    _, labels, centers = cv2.kmeans(pixel_data, k, None, stop_conds, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # 将聚类结果转换回图像
    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(img.shape)
    
    # 显示原图和分割结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(clustered, cv2.COLOR_BGR2RGB)), plt.title(f"Clustered Image (k={k})")
    plt.tight_layout()
    plt.show()
    return labels, centers


# Let's open the image
img_path = 'TestImages/TestImages/coffee_grains.jpg'
img= cv2.imread(img_path) 
# We need to re-format the data, we currently have three matrices (3 color values BGR) 
pixel_data = np.float32(img.reshape((-1,3)))
# then perform k-means clustering with random centers
# we can set accuracy to (i.e.) 90 (epsilon)
# and set a maximum number of iterations to 50

number_of_clusters = 2
stop_conds= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.90) 
number_of_attempts = 6
color_clustering(img_path, number_of_clusters, number_of_attempts, stop_conds, 'BGR')
#_, regions, centers  = cv2.kmeans(pixel_data, number_of_clusters, None, stop_conds, number_of_attempts , cv2.KMEANS_RANDOM_CENTERS) 
#print(regions)
# convert data to image format again again, with its original dimensions
#regions = np.uint8(centers)[regions.flatten()]
#segmented_image = regions.reshape((img.shape))
# We display original image and result 'segmented' image 
# Probably we need to adjust the number of regions
# And we have to think that we only are considering color information (no neighborhood)
#cv2.imshow("original_image", img)
#cv2.imshow("segmented_image", segmented_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()