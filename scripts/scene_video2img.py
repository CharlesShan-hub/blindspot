import cv2
import os

# 设置视频文件路径
video_path = '/Users/kimshan/Public/data/blindpoint/scene/videos/2024-12-11-14-32-03-crop.mp4'

# 设置保存图片的目录
output_dir = '/Users/kimshan/Public/data/blindpoint/scene/videos/img1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 每隔n帧抽取一帧
n = 30

# 使用OpenCV打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
frame_id = 0

# 读取视频中的帧
while True:
    ret, frame = cap.read()

    # 如果读取帧失败，则退出循环
    if not ret:
        break

    # 每隔n帧抽取一帧
    if frame_count % n == 0:
        # 设置图片文件名
        filename = os.path.join(output_dir, f'frame_{frame_id:04d}.jpg')
        # 保存图片
        cv2.imwrite(filename, frame)
        frame_id += 1

    frame_count += 1

# 释放视频捕获对象
cap.release()
print(f"Frames extracted: {frame_id}")
