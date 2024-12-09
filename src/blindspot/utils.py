import numpy as np
from PIL import Image

# 可以处理含 nan 数据的norm
def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    return (data - _min) / (_max - _min)

def fill_nan_with_max(arr):
    arr[arr == np.nan] = np.nanmax(arr)
    return arr

def fill_nan_with_min(arr):
    arr[arr == np.nan] = np.nanmin(arr)
    return arr

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data

# 读取图片并保存成 numpy 矩阵
def read_png_to_array(file_path):
    img = Image.open(file_path)
    return np.array(img)