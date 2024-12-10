import numpy as np
from scipy.optimize import curve_fit
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from clib.utils import glance
from .utils import *
from .detection import *

BASE_PATH = ""

def get_all_proj_info():
    '''
        读取全部项目基础信息(读 CSV 文件)
    '''
    info = {}
    with open(Path(BASE_PATH) / 'pathinfo.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            info[int(row[0])] = {
                'index': int(row[0]),
                'width': int(row[1]),
                'height': int(row[2]),
                'temp_l': int(row[3]),
                'temp_h': int(row[4]),
                'num_l': int(row[5]),
                'num_h': int(row[6]),
                'scale': int(row[7]),
                'path': Path(row[8]),
                'active': row[9] == 'True'
            }
    return info

def get_proj_info_by_index(index):
    '''
        根据 proj_id 从 CSV 中筛选项目基本信息
    '''
    with open(Path(BASE_PATH) / 'pathinfo.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            if int(index) == int(row[0]):
                return  {
                    'index': int(row[0]),
                    'width': int(row[1]),
                    'height': int(row[2]),
                    'temp_l': int(row[3]),
                    'temp_h': int(row[4]),
                    'num_l': int(row[5]),
                    'num_h': int(row[6]),
                    'scale': int(row[7]),
                    'path': Path(row[8]),
                    'active': row[9] == 'True'
                }
    return False

def change_info(info):
    """
        修改某一项的元数据
    """
    all_info = get_all_proj_info()
    with open(Path(BASE_PATH) / 'pathinfo.csv', 'w', newline='') as write_csv_file:
        csv_writer = csv.writer(write_csv_file)
        csv_writer.writerow(['index', 'width', 'height', 'temp_l', 'temp_h', 'num_l', 'num_h', 'scale', 'path', 'active'])
        for _, value in all_info.items():
            if value['index'] != info['index']:
                csv_writer.writerow([
                    value['index'], value['width'], value['height'], value['temp_l'], value['temp_h'], 
                    value['num_l'], value['num_h'], value['scale'], value['path'], value['active']
                ])
            else:
                csv_writer.writerow([
                    info['index'], info['width'], info['height'], info['temp_l'], info['temp_h'], 
                    info['num_l'], info['num_h'], info['scale'], info['path'], info['active']
                ])

def delete_info(info):
    white_list = ['index','width','height','temp_l','temp_h','num_l','num_h','scale','path','active']
    black_list = [key for key in info if key not in white_list]
    for key in black_list:
        del info[key]

def load_low_imgs(info):
    '''
        低温图片
    '''
    path = Path(BASE_PATH) / str(info['index']) / f'{info["temp_l"]}'
    imgs = np.zeros((info['num_l'], info['height'], info['width']), dtype=np.float32)
    for i,img_path in enumerate(path.rglob('*')):
        imgs[i] = read_png_to_array(path / img_path)
    info['img_l'] = imgs

def load_high_imgs(info):
    '''
        高温图片
    '''
    path = Path(BASE_PATH) / str(info['index']) / f'{info["temp_h"]}'
    imgs = np.zeros((info['num_h'], info['height'], info['width']), dtype=np.float32)
    for i,img_path in enumerate(path.rglob('*')):
        imgs[i] = read_png_to_array(path / img_path)
    info['img_h'] = imgs

def load_low_voltages(info):
    '''
        低温电压
    '''
    if hasattr(info,'img_l') == False:
        load_low_imgs(info)
    info['vol_l'] = info['img_l'].astype(np.float32)/65536 * info['scale'] * 1.01

def load_high_voltages(info):
    '''
        高温电压
    '''
    if hasattr(info,'img_h') == False:
        load_high_imgs(info)
    info['vol_h'] = info['img_h'].astype(np.float32)/65536 * info['scale'] * 1.01

def load_low_noice(info):
    '''
        噪声 (默认噪声通过低温计算)
    '''
    if hasattr(info,'vol_l') == False:
        load_low_voltages(info)
    info['noice_l'] = np.std(info['vol_l'],axis=0)

def load_high_noice(info):
    '''
        邓老师让加的,通过高温电压算的噪声
    '''
    if hasattr(info,'vol_h') == False:
        load_high_voltages(info)
    info['noice_h'] = np.std(info['vol_h'],axis=0)

def pixel_voltage_response(info):
    '''
        像元响应电压
    '''
    if hasattr(info,'vol_h') == False:
        load_high_voltages(info)
    if hasattr(info,'vol_l') == False:
        load_low_voltages(info)
    info['vol_response'] = np.average(info['vol_h'], axis=0) - np.average(info['vol_l'], axis=0)

def pixel_voltage_responsivity(info):
    '''
        像元电压响应率
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    sigma = np.array(5.673e-12,dtype=np.float64)
    temp = (info["temp_h"]+273.15)**4 - (info["temp_l"]+273.15)**4
    area = np.array(15e-3,dtype=np.float64) ** 2
    L = 0.006
    D = 0.006
    n = 1 if L/D > 1 else 0
    p = (sigma * temp * area) / (4 * (L/D)**2 + n)
    info['vol_responsivity'] = info['vol_response'] / p

def dead_pixel_threshold(info):
    '''
        国标法(死像元)
        红外焦平面阵列参数测试方法, 国标 GB/T 17444-2013
    '''
    if hasattr(info,'vol_responsivity') == False:
        pixel_voltage_responsivity(info) 
    threshold = 0.1 * np.average(info['vol_responsivity'])
    return info['vol_responsivity'] < threshold

def overheated_pixel_threshold(info):
    '''
        国标法(过热像元)
        红外焦平面阵列参数测试方法, 国标 GB/T 17444-2013
    '''
    if hasattr(info,'noice_l') == False:
        load_low_noice(info) 
    threshold = 10 * np.average(info['noice_l'])
    return info['noice_l'] > threshold

def pixel_three_sigma(info):
    '''
        盲元 3sigma
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    mean = np.mean(info['vol_response'])
    sigma = np.std(info['vol_response'])
    return np.abs(info['vol_response']-mean) > 3 * sigma

def curved_surface_fitting(info,times=3):
    '''
        盲元 曲面拟合
        张北伟,曹江涛,丛秋梅. 基于曲面拟合的红外图像盲元检测方法 [J]. 红外技术, 2017, 39  (11): 1007-1011.
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    def poly_surface(xy, a, b, c, d, e, f):
        (x,y) = xy
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
    image = np.average(info['vol_l'], axis=0)
    (rows, cols) = image.shape
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)
    X, Y = np.meshgrid(x, y)
    [X_flat, Y_flat, Z_flat] = [i.flatten() for i in [X,Y,image]]
    popt, _ = curve_fit(poly_surface, (X_flat, Y_flat), Z_flat)
    S = poly_surface((X, Y), *popt).reshape(rows, cols)
    sigma = np.sqrt(np.sum((image.ravel() - S.ravel())**2) / (image.size - 1))
    return np.abs(image-S) > times * sigma

def morphologic():
    '''
        李丽萍,袁祁刚,朱华,等. 一种新的红外焦平面阵列盲元检测算法 [J]. 红外技术, 2014, 36  (02): 106-109.
    '''
    pass

def pixel_double_source(info,threshold=0.1):
    '''
        双参考源检测法(红外焦平面器件盲元检测及补偿算法 周慧鑫 2004)
        需要手动设置电压的阈值
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    info['pixel'] = np.abs(info['vol_response']) > threshold


def plot_3d(gray_img, zlim=None):
    # 提取 x 和 y 轴坐标
    # 提取 x 和 y 轴坐标
    x = np.arange(gray_img.shape[1])
    y = np.arange(gray_img.shape[0])
    x, y = np.meshgrid(x, y)

    # 创建一个新的图形对象和一个三维轴对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维曲面图
    surf = ax.plot_surface(x, y, gray_img, cmap='viridis')

    # 设置颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if zlim is not None:
        ax.set_zlim(zlim)

    # 显示图形
    plt.show()