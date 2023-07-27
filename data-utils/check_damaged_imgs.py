# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 13:54
# @Author  : xjw
# @FileName: 检查是否存在破损图片.py
# @Software: PyCharm
# @Blog    ：https://github.com/Joran1101?tab=repositories
import cv2
import os
from pathlib import Path

image_folder = '/home/xjw/yolov5-cbam/seal-detect/'

for img_path in Path(image_folder).glob('*'):

    try:
        # 尝试使用OpenCV读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'img {img_path} is None')
            os.remove(im_path)

        # 读取成功,图片正常
        print(f'Loaded {img_path} successfully')

    except Exception as e:

        # OpenCV读取失败,可能图片损坏
        print(f'Error loading {img_path}: {e}')

        # 删除损坏图片
        print(f'Deleting corrupted image {img_path}')
        os.remove(img_path)