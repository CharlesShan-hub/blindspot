import blindspot as bs
import numpy as np
import matplotlib.pyplot as plt

bs.BASE_PATH = '/Users/kimshan/Public/data/mang_yuan2'

for _,info in bs.get_all_proj_info().items():
    bs.dead_pixel_threshold(info)
    bs.overheated_pixel_threshold(info)

    print(len(np.argwhere(info['dead'] == True)))
    
    # 绘制死像元mask
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(info['dead'], cmap='gray')  # 使用灰度颜色映射
    # plt.title('Dead Pixel Mask')
    # plt.colorbar()  # 显示颜色条

    # # 绘制过热像元mask
    # plt.subplot(1, 2, 2)
    # plt.imshow(info['overheated'], cmap='gray')  # 使用热图颜色映射
    # plt.title('Overheated Pixel Mask')
    # plt.colorbar()  # 显示颜色条

    # # 显示图像
    # plt.show()

# def test(info):
#     info[3]='c'

# def main():
#     info = {1:'a',2:'b'}
#     test(info)
#     print(info)
# main()


