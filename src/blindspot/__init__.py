import numpy as np
import csv
from pathlib import Path
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
                'path': Path(row[8])
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
                    'path': Path(row[8])
                }
    return False

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
    info['vol_l'] = info['img_l'].astype(np.float32)/65536 * info['scale']

def load_high_voltages(info):
    '''
        高温电压
    '''
    if hasattr(info,'img_h') == False:
        load_high_imgs(info)
    info['vol_h'] = info['img_h'].astype(np.float32)/65536 * info['scale']

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
    sigma = np.array(5.673,dtype=np.float64) * 10**(-12)
    temp = info["temp_h"]**4 - info["temp_l"]**4
    area = np.array(15*10**(-3),dtype=np.float64) ** 2
    L = 0.006
    D = 0.006
    n = 1 if L/D > 1 else 0
    p = (sigma * temp * area) / (4 * (L/D)**2 + n)
    info['vol_responsivity'] = info['vol_response'] / p

def dead_pixel_threshold(info):
    '''
        死像元 国标法
    '''
    if hasattr(info,'vol_responsivity') == False:
        pixel_voltage_responsivity(info) 
    threshold = 0.1 * np.average(info['vol_responsivity'])
    info['dead'] = info['vol_response'] > threshold

def overheated_pixel_threshold(info):
    '''
        过热像元 国标法
    '''
    if hasattr(info,'noice_l') == False:
        load_low_noice(info) 
    threshold = 10 * np.average(info['noice_l'])
    info['overheated'] = info['noice_l'] > threshold

    

