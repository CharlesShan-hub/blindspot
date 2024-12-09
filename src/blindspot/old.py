# def old_urved_surface_fitting(info):
#     '''
#         盲元 曲面拟合
#         张北伟,曹江涛,丛秋梅. 基于曲面拟合的红外图像盲元检测方法 [J]. 红外技术, 2017, 39  (11): 1007-1011.
#     '''
#     if hasattr(info,'vol_response') == False:
#         pixel_voltage_response(info)
    
#     def fit_surface(image): # 进行曲面拟合
#         def poly_surface(xy, a, b, c, d, e, f):
#             (x,y) = xy
#             return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
#         (rows, cols) = image.shape
#         x = np.linspace(0, cols - 1, cols)
#         y = np.linspace(0, rows - 1, rows)
#         X, Y = np.meshgrid(x, y)
#         [X_flat, Y_flat, Z_flat] = [i.flatten() for i in [X,Y,image]]
#         popt, _ = curve_fit(poly_surface, (X_flat, Y_flat), Z_flat)
#         return poly_surface((X, Y), *popt).reshape(rows, cols)
    
#     def calculate_sigma(image, S):# 计算像元灰度标准差。
#         V = image.ravel()
#         S = S.ravel()
#         sigma = np.sqrt(np.sum((V - S)**2) / (image.size - 1))
#         return sigma

#     def detect_blind_pixels(image, S, sigma):
#         """
#         检测盲元。
#         """
#         # glance(np.abs(image-S) > 3 * sigma)
#         blind_pixels = []
#         for i in range(1,image.shape[0]-1):
#             for j in range(1,image.shape[1]-1):
#                 if np.abs(image[i, j] - S[i, j]) >3 * sigma:
#                     blind_pixels.append((i, j))
        
#         return blind_pixels
    
#     def gaussian_blind_compensation(image, blind_pixels):
#         """
#         使用八邻域高斯补偿方法进行盲元补偿。
#         """
#         print(len(blind_pixels))
#         for i, j in blind_pixels:
#             # 获取邻域像素
#             neighbors = image[i-1:i+2, j-1:j+2]

#             # 计算距离
#             distances = np.sqrt(np.array([
#                 [2, 1, 2],
#                 [1, 0, 1],
#                 [2, 1, 2]
#             ]))

#             # 计算高斯权重
#             weights = np.exp(-0.5 * distances**2)

#             # 加权平均
#             compensation_value = np.sum(weights * neighbors) / np.sum(weights)

#             # 更新图像
#             image[i, j] = compensation_value

#         return image
    
#     def pad_image_with_mean(image, padding_size=1):
#         # 计算图像的平均灰度值
#         mean_value = np.mean(image)
        
#         # 获取原图像的尺寸
#         rows, cols = image.shape
        
#         # 创建一个新的图像数组，其大小为原图像大小加上两倍的填充大小
#         padded_image = np.full((rows + 2 * padding_size, cols + 2 * padding_size), mean_value, dtype=image.dtype)
        
#         # 将原图像复制到新图像的中心
#         padded_image[padding_size:padding_size + rows, padding_size:padding_size + cols] = image
        
#         return padded_image
    
#     image = np.average(info['img_l'], axis=0)
#     image = pad_image_with_mean(image, 1)
#     while True:
#         S = fit_surface(image)
#         # glance([image,S])
#         # plot_3d(image, zlim=(38000, 46000))
#         # plot_3d(S, zlim=(38000, 46000))
#         sigma = calculate_sigma(image, S)
#         blind_pixels = detect_blind_pixels(image, S, sigma)
#         if len(blind_pixels) == 0:
#             break
#         image = gaussian_blind_compensation(image, blind_pixels)
#     return blind_pixels
