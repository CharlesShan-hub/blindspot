import numpy as np
from PIL import Image

# 可以处理含 nan 数据的norm
def norm(data):
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    return (data - _min) / (_max - _min)

# 读取txt文件并保存为numpy矩阵
def read_txt_to_matrix(file_path):
    data = np.loadtxt(file_path, delimiter='\t')
    return data

# 读取图片并保存成 numpy 矩阵
def read_png_to_array(file_path):
    img = Image.open(file_path)
    return img