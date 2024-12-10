import blindspot as bs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from clib.utils import glance,save_array_to_img

bs.BASE_PATH = '/Users/kimshan/Public/data/mang_yuan2'

# for _,info in bs.get_all_proj_info().items():
info = bs.get_proj_info_by_index(4)

# 单参考源 - 国标法
bad = bs.dead_pixel_threshold(info)
bad = bs.overheated_pixel_threshold(info)

# 单参考源 - 曲面拟合法
bad = bs.curved_surface_fitting(info)

# 单参考源 - 形态学方法


# 单参考源 - 统计方法 3sigma检测法
# bad = bs.pixel_three_sigma(info)

# 单参考源 - 统计方法 特征直方图


# 单参考源 - 统计方法 多元统计特性


# 双参考源


# 多参考源 - 多温统计 3sigma 检测法


# 多参考源 - 定标检测法

# img =np.average(info['vol_l'], axis=0)

save_array_to_img(bs.curved_surface_fitting(info,times=7),Path("./4-7s.png"))
# np.isnan(img)
# glance([img,bs.curved_surface_fitting(info,times=6)],title=["Origin","6 Sigma"])
# glance([img,bs.curved_surface_fitting(info,times=7)],title=["Origin","7 Sigma"])
# glance([img,bs.curved_surface_fitting(info,times=8)],title=["Origin","8 Sigma"])


# bs.dead_pixel_threshold(info)
# bs.overheated_pixel_threshold(info)
# print(np.argwhere(info['dead'] == True))
# print(np.argwhere(info['overheated'] == True))
# print(len(np.argwhere(info['dead'] == True)),info['width']*info['height'])
# print(len(np.argwhere(info['overheated'] == True)),info['width']*info['height'])

# bs.pixel_three_sigma(info)
# print(np.argwhere(info['overheated'] == True))
# print(len(np.argwhere(info['overheated'] == True)),info['width']*info['height'])

# bs.pixel_double_source(info,threshold=0.6)
# # print(np.argwhere(info['overheated'] == True))
# print(len(np.argwhere(info['pixel'] == True)),info['width']*info['height'])

# # 绘制死像元mask
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(info['vol_response'], cmap='gray')  # 使用灰度颜色映射
# plt.title('Dead Pixel Mask')
# plt.colorbar()  # 显示颜色条

# # # 绘制过热像元mask
# plt.subplot(1, 2, 2)
# plt.imshow(info['pixel'], cmap='gray')  # 使用热图颜色映射
# plt.title('Overheated Pixel Mask')
# plt.colorbar()  # 显示颜色条

# # 显示图像
# plt.show()

