import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

__all__ = [
    'norm',
    'fill_nan_with_max',
    'fill_nan_with_min',
    'read_txt_to_matrix',
    'read_png_to_array',
    'plot_3d',
    'plot_wave'
]

# 可以处理含 nan 数据的norm
def norm(data: np.ndarray) -> np.ndarray:
    _min = np.nanmin(data)
    _max = np.nanmax(data)
    return (data - _min) / (_max - _min)

def fill_nan_with_max(arr: np.ndarray) -> np.ndarray:
    arr[arr == np.nan] = np.nanmax(arr)
    return arr

def fill_nan_with_min(arr: np.ndarray) -> np.ndarray:
    arr[arr == np.nan] = np.nanmin(arr)
    return arr

def read_txt_to_matrix(file_path) -> np.ndarray:
    data = np.loadtxt(file_path, delimiter='\t')
    return data

def read_png_to_array(file_path) -> np.ndarray:
    img = Image.open(file_path)
    return np.array(img)

def float_to_rgb16(value):
    # 将0到1的值映射到0到255的范围
    gray_value = int(value * 255)
    # 将灰度值转换为16进制字符串，并确保它是两位数
    hex_value = format(gray_value, '02x')
    # 返回RGB16进制字符串，由于是灰度，所以R、G、B值相同
    return f'#{hex_value}{hex_value}{hex_value}'

def plot_3d(gray_img, zlim=None):
    """ 绘制灰度图灰度值的 3D 图
    """
    x = np.arange(gray_img.shape[1])
    y = np.arange(gray_img.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, gray_img, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if zlim is not None:
        ax.set_zlim(zlim)
    plt.show()

def plot_wave(p: tuple, r: int, bad: np.ndarray, \
              proj_id: int, save_dir: str, double_temp: bool, \
              vol_l: np.ndarray, vol_h: np.ndarray, \
              vol_l_avg: np.ndarray = None,vol_h_avg: np.ndarray = None, \
              noice_l_avg: np.ndarray = None,noice_h_avg: np.ndarray = None):
    """ 根据电压和盲元表绘制电压曲线

    Args:
        p (tuple): 中心点坐标
        r (int): 中心点周围扩展像素个数
        bad (np.ndarray): 盲元表
        proj_id (int): 实验 ID
        save_dir (str): 保存目录
        double_temp (bool): 绘制两个温度曲线
        vol_l (np.ndarray): 低温电压
        vol_h (np.ndarray): 高温电压
        vol_l_avg (np.ndarray): 低温平均电压
        vol_h_avg (np.ndarray): 高温平均电压
    """
    
    s = bad.shape
    d = 1+2*r

    if vol_l_avg is None:
        vol_l_avg = np.average(vol_l,axis=0)
    if vol_h_avg is None :
        vol_h_avg = np.average(vol_h,axis=0)
    if noice_l_avg is None:
        noice_l_avg = np.average(np.std(vol_l_avg,axis=0))
    if noice_h_avg is None :
        noice_h_avg = np.average(np.std(vol_h_avg,axis=0))
    fig, axs = plt.subplots(d, d, figsize=(7*d, 7*d))

    color_image = np.ones((d, d), dtype=np.float64) * 0.0
    x1 = max(p[0]-r,0)
    x2 = min(p[0]+r+1,s[0])
    y1 = max(p[1]-r,0)
    y2 = min(p[1]+r+1,s[1])
    temp = norm(vol_l_avg[x1:x2, y1:y2])
    i1 = r - (p[0] - x1)
    i2 = i1 + temp.shape[0]
    j1 = r - (p[1] - y1)
    j2 = j1 + temp.shape[1]
    color_image[i1:i2, j1:j2] = temp

    for i in range(d):
        for j in range(d):
            q = (p[0]-r+i, p[1]-r+j)
            if q[0]<0 or q[0]>(s[0]-1) or q[1]<0 or q[1]>(s[1]-1):
                continue
            plt.subplot(d,d,1+d*i+j)
            if bad[q[0], q[1]] == 255:  # 坏点的像素值是255
                color = ['red','darkred']
            else:
                color = ['green','darkgreen']
            axs[i][j].set_facecolor(float_to_rgb16(color_image[i][j]))
            if double_temp == True:
                axs[i][j].plot(vol_l[:, q[0], q[1]], color=color[0])
                axs[i][j].set_ylim(vol_l_avg[q[0],q[1]]-noice_l_avg*9,vol_l_avg[q[0],q[1]]+noice_l_avg*3)
                ax_twin = axs[i][j].twinx()
                ax_twin.plot(vol_h[:, q[0], q[1]], color=color[1])
                ax_twin.set_ylim(vol_h_avg[q[0],q[1]]-noice_h_avg*3,vol_h_avg[q[0],q[1]]+noice_h_avg*9)
            else:
                axs[i][j].plot(vol_l[:, q[0], q[1]], color=color[0])
                axs[i][j].set_ylim(vol_l_avg[q[0],q[1]]-noice_l_avg*3,vol_l_avg[q[0],q[1]]+noice_l_avg*3)
            axs[i][j].set_title(f'{q}')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    if (Path(save_dir) / f'{proj_id}').exists() == False:
        (Path(save_dir) / f'{proj_id}').mkdir()
    plt.savefig(Path(save_dir) / f'{proj_id}' / f"{p[0]}_{p[1]}")