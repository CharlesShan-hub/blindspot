import numpy as np
from scipy.optimize import curve_fit
from .base import *

__all__ = [
    'dead_pixel_threshold',
    'overheated_pixel_threshold',
    'pixel_three_sigma',
    'dect_curved_surface_fitting',
    'pixel_double_source'
]

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

def dect_curved_surface_fitting(info,times=3):
    '''
        盲元 曲面拟合
        张北伟,曹江涛,丛秋梅. 基于曲面拟合的红外图像盲元检测方法 [J]. 红外技术, 2017, 39  (11): 1007-1011.
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    def poly_surface(xy, a, b, c, d, e, f):
        (x,y) = xy
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
    image = info['vol_response']
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
