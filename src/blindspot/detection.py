'''
    盲元检测算法复现

    1. 命名规则: dect_名称_盲元类型(如果有)
    2. 单参考原 source 范围是 'l' 或 'h'
'''

import numpy as np
from .base import *
from .utils import curved_surface_fitting

__all__ = [
    'dect_gb_dead',
    'dect_gb_overheated',
    'dect_gb',
    'dect_three_sigma',
    'dect_curved_surface_fitting',
    'dect_double_source'
]

def dect_gb_dead(info):
    '''
        国标法(死像元)
        红外焦平面阵列参数测试方法, 国标 GB/T 17444-2013
        https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=DED06F10FA8CD7529082C3746F735B1A
    '''
    if hasattr(info,'vol_responsivity') == False:
        pixel_voltage_responsivity(info) 
    return info['vol_responsivity'] < 0.5 * np.average(info['vol_responsivity'])

def dect_gb_overheated(info, dead_mask = None):
    '''
        国标法(过热像元)
        红外焦平面阵列参数测试方法, 国标 GB/T 17444-2013
        https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=DED06F10FA8CD7529082C3746F735B1A
    '''
    if hasattr(info,'noice_l') == False:
        load_low_noice(info)
    image = np.average(info['noice_l'], axis=0)
    if dead_mask is None:
        dead_mask = dect_gb_dead(info)
    return image > 2 * np.mean(image[:, ~dead_mask])

def dect_gb(info):
    '''
        国标法(过热像元)
        红外焦平面阵列参数测试方法, 国标 GB/T 17444-2013
        https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=DED06F10FA8CD7529082C3746F735B1A
    '''
    if hasattr(info,'noice_l') == False:
        load_low_noice(info)
    dead_mask = dect_gb_dead(info)
    overheat_mask = info['noice_l'] > 2 * np.mean(info['noice_l'][~dead_mask])
    return np.logical_or(dead_mask, overheat_mask)

def dect_three_sigma(info, source='l'):
    '''
        盲元 3sigma
        刘会通, 马红伟. 红外焦平面非均匀性校正若干方案的设计和分析[J].  激光与红外, 2003(4): 277-279.
    '''
    if hasattr(info,f'vol_{source}') == False:
        load_low_voltages(info) if source=='l' else load_high_voltages(info)
    image = np.average(info[f'vol_{source}'], axis=0)
    return np.abs(image - np.mean(image)) > 3 * np.std(image)

def dect_curved_surface_fitting(info, source='l', times=3):
    '''
        盲元 曲面拟合
        张北伟,曹江涛,丛秋梅. 基于曲面拟合的红外图像盲元检测方法 [J]. 红外技术, 2017, 39  (11): 1007-1011.
    '''
    if hasattr(info,f'vol_{source}') == False:
        load_low_voltages(info) if source=='l' else load_high_voltages(info)
    I = np.average(info[f'vol_{source}'], axis=0)
    S = curved_surface_fitting(I)
    sigma = np.sqrt(np.sum((I.ravel() - S.ravel())**2) / (I.size - 1))
    return np.abs(I-S) > times * sigma

def dect_morphologic(info):
    '''
        李丽萍,袁祁刚,朱华,等. 一种新的红外焦平面阵列盲元检测算法 [J]. 红外技术, 2014, 36  (02): 106-109.
    '''
    pass

def dect_double_source(info,threshold=0.1):
    '''
        双参考源检测法(红外焦平面器件盲元检测及补偿算法 周慧鑫 2004)
        需要手动设置电压的阈值
    '''
    if hasattr(info,'vol_response') == False:
        pixel_voltage_response(info)
    info['pixel'] = np.abs(info['vol_response']) > threshold
