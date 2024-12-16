from pathlib import Path
import csv
import numpy as np
from .utils import read_png_to_array

__all__ = [
    # Based on Source
    'get_all_proj_info',
    'get_all_active_proj_info',
    'get_proj_info_by_index',
    'change_info',
    'delete_info',
    'load_low_imgs',
    'load_high_imgs',
    'load_low_voltages',
    'load_high_voltages',
    'load_low_noice',
    'load_high_noice',
    'load_bad_mask',
    'pixel_voltage_response',
    'pixel_voltage_responsivity',

    # Based on Scene
    # 'get_all_video_info',

]

def get_all_proj_info():
    '''
        读取全部项目基础信息(读 CSV 文件)
    '''
    from . import BASE_PATH
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

def get_all_active_proj_info():
    """ 筛选出开启的样本
    """
    info = {}
    for k, v in get_all_proj_info().items():
        if v['active']:
            info[k] = v
    return info

def get_proj_info_by_index(index):
    '''
        根据 proj_id 从 CSV 中筛选项目基本信息
    '''
    from . import BASE_PATH
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
    from . import BASE_PATH
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
    from . import BASE_PATH
    path = Path(BASE_PATH) / str(info['index']) / f'{info["temp_l"]}'
    imgs = np.zeros((info['num_l'], info['height'], info['width']), dtype=np.float32)
    for i,img_path in enumerate(path.rglob('*')):
        imgs[i] = read_png_to_array(path / img_path)
    info['img_l'] = imgs

def load_high_imgs(info):
    '''
        高温图片
    '''
    from . import BASE_PATH
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
        通过高温电压算的噪声
    '''
    if hasattr(info,'vol_h') == False:
        load_high_voltages(info)
    info['noice_h'] = np.std(info['vol_h'],axis=0)

def load_bad_mask(info,method:str):
    '''
        盲元表
    '''
    from . import BASE_PATH
    path = Path(BASE_PATH) / 'bad' / method
    assert path.exists()
    info['bad'] = read_png_to_array(path / f'{info['index']}.png')
    
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
