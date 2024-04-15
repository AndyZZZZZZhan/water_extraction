from osgeo import gdal
import numpy as np
import cv2
import glob
import os
import jenkspy
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_img(input_file):

    input_data = gdal.Open(input_file)
    rows = input_data.RasterYSize #获取图像的高度
    cols = input_data.RasterXSize #获取图像的宽度
    bands = input_data.RasterCount #获取图像波段数
    # GDT_Byte =1, GDT_UInt16=2,G DT_UInt32=4, GDT_Int32=5, GDT_Float32 =6
    datatype = input_data.GetRasterBand(1).DataType
    print('数据类型：', datatype)
    array_data = input_data.ReadAsArray()  # 转换为numpy数组
    array_data[np.isnan(array_data)] = np.nanmax(array_data)
    print(array_data.dtype)

    del input_data
    print("行数：", rows)
    print("列数：", cols)
    print("波段数：", bands)

    return array_data

def write_img(read_path, img_array, output_directory):
    read_pre_data = gdal.Open(read_path)
    img_tranf = read_pre_data.GetGeoTransform()  # 仿射矩阵
    img_proj = read_pre_data.GetProjection()  # 地图投影信息

    datatype = gdal.GDT_Byte
    img_height, img_width = img_array.shape

    # 获取原文件名
    original_filename = os.path.basename(read_path)
    # 构建新文件名
    new_filename = os.path.splitext(original_filename)[0] + '_test.tif'
    # 完整的保存路径
    new_filepath = os.path.join(output_directory, new_filename)

    driver = gdal.GetDriverByName('Gtiff')
    dataset = driver.Create(new_filepath, img_width, img_height, 1, datatype)
    dataset.SetGeoTransform(img_tranf)
    dataset.SetProjection(img_proj)
    dataset.GetRasterBand(1).WriteArray(img_array)

    del dataset

def compress(array_data):
    cutmin = np.min(array_data)
    cutmax = np.max(array_data)

    array_data[array_data < cutmin] = cutmin
    array_data[array_data > cutmax] = cutmax
    compress_data = np.around((array_data - cutmin)*255/(cutmax - cutmin))
    compress_data = np.array(compress_data, dtype='uint8')
    return compress_data

def natural_breaks(array, block_size, num_bins):
    rows, cols = array.shape
    breaks_image = np.zeros_like(array)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            block = array[r:r + block_size, c:c + block_size]

            if np.all(block == 255):
                breaks_image[r:r + block_size, c:c + block_size] = block
            else:
                # 使用jenkspy计算自然断点
                data_sorted = np.sort(block.flatten())
                filtered_data = data_sorted[data_sorted < 255]
                breaks = jenkspy.jenks_breaks(filtered_data, num_bins)
                threshold = breaks[2]
                print("初始水图阈值为：",threshold)
                block[block > threshold] = 255
                breaks_image[r:r + block_size, c:c + block_size] = block

    return breaks_image

def clustering_and_otsu(array):
    rows, cols = array.shape
    processed_image = np.zeros_like(array)

    # 检查整个图像是否全是255
    if not np.all(array == 255):
        # 创建Mean-Shift聚类器
        data = np.argwhere(array != 255)
        bandwidth = estimate_bandwidth(data, quantile=0.05, n_samples=10000)
        print("聚类bandwidth：", bandwidth)
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        # 进行聚类
        clustering.fit(data)
        # 获取聚类结果
        labels = clustering.labels_
        cluster_centers = clustering.cluster_centers_

        result = np.zeros_like(array)
        for i, center in enumerate(cluster_centers):
            cluster_data = data[labels == i]
            original_cluster_values = array[cluster_data[:, 0], cluster_data[:, 1]]
            _, binary_cluster = cv2.threshold(original_cluster_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_cluster = np.where(binary_cluster == 0, 255, 0)
            for idx, (x, y) in enumerate(cluster_data):
                result[x, y] = binary_cluster[idx]

    return result

def Hole_fill(array):
    array[array >= 128] = 255
    array[array < 128] = 0

    print("开始填补孔洞")
    # 计算连通域
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(array, connectivity=8)
    output_array = np.zeros_like(array)

    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 100:
            output_array[labels == label] = 0
        else:
            output_array[labels == label] = 255

    fill_img = 255 - output_array

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fill_img, connectivity=8)
    fill_array = np.zeros_like(array)

    print("开始消除碎斑")
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 3:
            fill_array[labels == label] = 0
        else:
            fill_array[labels == label] = 255

    fill_array = fill_array.astype(np.uint8)
    return fill_array

def batch(folder_path):
    tif_files = glob.glob(os.path.join(folder_path, '*.tif'))
    for file_path in tif_files:
        # 指定输出目录
        output_directory = os.path.join(os.path.dirname(file_path), '_flood')
        print("正在处理：", file_path)
        array = read_img(file_path)
        print("数据读取成功")
        compress_array = compress(array)
        print("归一化完成")
        jenks_array = natural_breaks(compress_array, 1000, 5)
        print("自然断点法完成")
        water_image = clustering_and_otsu(jenks_array)
        print("水体提取完成")
        post_image = Hole_fill(water_image)
        print("后处理完成")
        write_img(post_image, output_directory)
        print("保存完成")

batch(r"……")
