'''
    盲元补偿算法复现
'''

import numpy as np
from scipy.signal import convolve2d

__all__ = [
    'compensate_median',
    'compensate_pyramid',
    'compensate_adaptive'
]

def compensate_median(image, mask, kernel_size=3):
    '''
    中值滤波盲元补偿
    :param image: 输入图像 (2D numpy array)
    :param mask: 盲元掩码 (2D boolean array, True表示盲元)
    :param kernel_size: 滤波窗口大小
    :return: 补偿后的图像
    '''
    result = image.copy()
    rows, cols = image.shape
    pad = kernel_size // 2
    
    # 获取盲元坐标
    blind_coords = np.argwhere(mask)
    
    for r, c in blind_coords:
        r_min = max(0, r - pad)
        r_max = min(rows, r + pad + 1)
        c_min = max(0, c - pad)
        c_max = min(cols, c + pad + 1)
        
        window = image[r_min:r_max, c_min:c_max]
        window_mask = mask[r_min:r_max, c_min:c_max]
        
        # 只取非盲元的值
        valid_values = window[~window_mask]
        
        if valid_values.size > 0:
            result[r, c] = np.median(valid_values)
        # 如果窗口内全是盲元，则保持原值或取窗口中值（这里保持原值）
            
    return result

def compensate_pyramid(image, mask, kernel_size=3):
    '''
    金字塔滤波盲元补偿 (加权均值滤波)
    :param image: 输入图像 (2D numpy array)
    :param mask: 盲元掩码 (2D boolean array, True表示盲元)
    :param kernel_size: 滤波窗口大小 (通常为3或5)
    :return: 补偿后的图像
    '''
    result = image.copy()
    
    # 生成金字塔权重核 (Gaussian-like)
    if kernel_size == 3:
        kernel = np.array([[1, 2, 1],
                           [2, 0, 2],
                           [1, 2, 1]], dtype=float)
    elif kernel_size == 5:
        # 简单的5x5近似
        kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 0, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=float)
    else:
        # 默认均值滤波
        kernel = np.ones((kernel_size, kernel_size))
        kernel[kernel_size//2, kernel_size//2] = 0
        
    # 归一化卷积处理
    # 只对有效像素进行加权平均
    valid_mask = (~mask).astype(float)
    masked_image = image * valid_mask
    
    # 分子：加权和
    numerator = convolve2d(masked_image, kernel, mode='same', boundary='symm')
    
    # 分母：权重的和 (仅计算有效像素的权重)
    denominator = convolve2d(valid_mask, kernel, mode='same', boundary='symm')
    
    # 避免除以零
    denominator[denominator == 0] = 1
    
    filled = numerator / denominator
    
    # 仅替换盲元位置
    result[mask] = filled[mask]
    
    return result

def compensate_adaptive(image, mask, kernel_size=3):
    '''
    自适应盲元补偿 (基于Sobel边缘检测)
    参考胡鹏博的方法：边缘区域使用均值滤波(或保留边缘特征)，平坦区域使用中值滤波
    :param image: 输入图像 (2D numpy array)
    :param mask: 盲元掩码 (2D boolean array, True表示盲元)
    :param kernel_size: 滤波窗口大小 (默认为5)
    :return: 补偿后的图像
    '''
    # 1. 初步使用中值滤波填充盲元，以便计算梯度
    filled = compensate_median(image, mask, kernel_size)
    
    # 2. 计算Sobel梯度
    # Sobel算子
    sobel_x_kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=float)
    sobel_y_kernel = np.array([[1, 2, 1], 
                               [0, 0, 0], 
                               [-1, -2, -1]], dtype=float)
                               
    grad_x = convolve2d(filled, sobel_x_kernel, mode='same', boundary='symm')
    grad_y = convolve2d(filled, sobel_y_kernel, mode='same', boundary='symm')
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 3. 确定边缘阈值
    # 简单的统计阈值：均值 + 0.5 * 标准差
    threshold = np.mean(magnitude) + 0.5 * np.std(magnitude)
    edge_mask = magnitude > threshold
    
    # 4. 对边缘区域应用方向性均值滤波
    # 计算梯度方向 (弧度)
    angle = np.arctan2(grad_y, grad_x)
    
    # 将角度转换为 0-180 度
    angle_deg = np.rad2deg(angle) % 180
    
    # 针对边缘区域的盲元进行方向性滤波
    # 边缘方向与梯度方向垂直
    # 0度(水平梯度) -> 垂直边缘 -> 上下方向均值
    # 90度(垂直梯度) -> 水平边缘 -> 左右方向均值
    # 45度 -> 对角边缘 -> 反对角方向均值
    # 135度 -> 反对角边缘 -> 对角方向均值
    
    result = filled.copy()
    
    blind_on_edge_indices = np.argwhere(np.logical_and(mask, edge_mask))
    pad = kernel_size // 2
    rows, cols = image.shape
    
    for r, c in blind_on_edge_indices:
        grad_angle = angle_deg[r, c]
        
        # 确定滤波方向 (沿着边缘方向滤波，即垂直于梯度方向)
        # 梯度方向 0度 (水平) -> 边缘是垂直的 -> 应该沿Y轴滤波
        # 梯度方向 90度 (垂直) -> 边缘是水平的 -> 应该沿X轴滤波
        
        # 梯度方向 0 +/- 22.5 -> 水平梯度 -> 垂直边缘 -> 沿Y轴滤波
        if (grad_angle >= 0 and grad_angle < 22.5) or (grad_angle >= 157.5 and grad_angle < 180):
            # 垂直边缘，沿列方向取均值
            r_min = max(0, r - pad)
            r_max = min(rows, r + pad + 1)
            window_vals = filled[r_min:r_max, c]
            result[r, c] = np.mean(window_vals)
            
        # 梯度方向 90 +/- 22.5 -> 垂直梯度 -> 水平边缘 -> 沿X轴滤波
        elif grad_angle >= 67.5 and grad_angle < 112.5:
             # 水平边缘，沿行方向取均值
            c_min = max(0, c - pad)
            c_max = min(cols, c + pad + 1)
            window_vals = filled[r, c_min:c_max]
            result[r, c] = np.mean(window_vals)
            
        # 梯度方向 45 +/- 22.5 -> 梯度方向为 / -> 边缘方向为 \ -> 沿反对角线滤波 (\)
        elif grad_angle >= 22.5 and grad_angle < 67.5:
            # 沿反对角线方向 (\)
            vals = []
            for k in range(-pad, pad + 1):
                rr, cc = r + k, c + k
                if 0 <= rr < rows and 0 <= cc < cols:
                    vals.append(filled[rr, cc])
            if vals:
                result[r, c] = np.mean(vals)
                
        # 梯度方向 135 +/- 22.5 -> 梯度方向为 \ -> 边缘方向为 / -> 沿对角线滤波 (/)
        elif grad_angle >= 112.5 and grad_angle < 157.5:
            # 沿对角线方向 (/)
            vals = []
            for k in range(-pad, pad + 1):
                rr, cc = r + k, c - k
                if 0 <= rr < rows and 0 <= cc < cols:
                    vals.append(filled[rr, cc])
            if vals:
                result[r, c] = np.mean(vals)
        else:
            # 默认情况 (应该不会执行到这里)
            pass

    return result, edge_mask