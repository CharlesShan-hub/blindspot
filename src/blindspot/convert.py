from pathlib import Path
from PIL import Image
import numpy as np
from .utils import read_txt_to_matrix

pixel_size = 2
BASE_PATH = ""

# 输入待转换目录根目录, 返回内部所有的项目文件夹, 可能有失效文件夹
def get_src_list():
    base_path = Path(BASE_PATH)
    assert base_path.is_dir(), f"The provided base_path '{base_path}' is not a directory."
    src_list = [base_path]
    for path in Path(base_path).rglob('*'):
        if path.is_dir() == False:
            continue
        if (len(list(path.glob('*.dat'))) != 2):
            continue
        src_list.append(path)
    return src_list

# 获取项目文件夹内的信息，如果文件夹失效，返回 false
def get_src_info(proj_path):
    info = {}

    if (proj_path / '像元噪声均值_V.txt').exists() == False:
        return False
    info['noice'] = read_txt_to_matrix(proj_path / '像元噪声均值_V.txt')

    temp = [int(i.name.split('C')[0]) for i in list(proj_path.glob('*.dat'))]
    info["temp_l"] = min(temp)
    info["temp_h"] = max(temp)

    if (proj_path / 'BadPixel.png').exists():
        with Image.open(proj_path / 'BadPixel.png') as img:
            info['width'], info['height'] = img.size
    elif '320_256_30' in str(proj_path):
        info['width'], info['height'] = (320, 256)
    elif '640_512_MW' in str(proj_path):
        info['width'], info['height'] = (640, 512)
    elif '2024-03-14-15-54-21-积分时间36ms' in str(proj_path):
        info['width'], info['height'] = (320, 256)
    elif '2024-02-28-10-54-12' in str(proj_path):
        info['width'], info['height'] = (320, 256)
    else:
        return False
    
    info['num_l'] = int((proj_path / f'{info["temp_l"]}C.dat').stat().st_size / (info['width'] * info['height'] * pixel_size))
    info['num_h'] = int((proj_path / f'{info["temp_h"]}C.dat').stat().st_size / (info['width'] * info['height'] * pixel_size))

    info['img_l'] = np.zeros((info['num_l'], info['height'], info['width']), dtype=np.uint16)
    with open(proj_path / f'{info["temp_l"]}C.dat', 'rb') as f:
        for i in range(info['num_l']):
            image_bytes = f.read(info['width'] * info['height'] * pixel_size)
            image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.uint16)
            image_data = image_data.reshape((info['height'], info['width']))
            info['img_l'][i] = image_data
    info['img_h'] = np.zeros((info['num_h'], info['height'], info['width']), dtype=np.uint16)
    with open(proj_path / f'{info["temp_h"]}C.dat', 'rb') as f:
        for i in range(info['num_h']):
            image_bytes = f.read(info['width'] * info['height'] * pixel_size)
            image_data = np.frombuffer(image_bytes, dtype='<u2').astype(np.uint16)
            image_data = image_data.reshape((info['height'], info['width']))
            info['img_h'][i] = image_data
    
    info['scale'] = int(round(np.average(info['noice']) / np.average(np.std(info['img_l'].astype(np.float32)/65535, axis=0)),3) / 1.001)
    if info['scale'] < 2:
        return False

    return info