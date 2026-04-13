#!/usr/bin/env python3
"""
CCPD数据集字符切割脚本
直接从文件名解析标注，切割7个字符图像
"""
import cv2
import numpy as np
import os
from tqdm import tqdm

# ============================================================
# CCPD字符映射表
# ============================================================
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
             "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
ALPHABETS = ['A','B','C','D','E','F','G','H','J','K',
             'L','M','N','P','Q','R','S','T','U','V',
             'W','X','Y','Z']
ADS       = ['A','B','C','D','E','F','G','H','J','K',
             'L','M','N','P','Q','R','S','T','U','V',
             'W','X','Y','Z',
             '0','1','2','3','4','5','6','7','8','9']

CHAR_H = 40
CHAR_W = 20

def parse_filename(fname):
    try:
        name = os.path.splitext(fname)[0]
        parts = name.split('-')
        if len(parts) < 5:
            return None

        # 车牌号在第4段（index=4）
        label_part = parts[4]
        indices = list(map(int, label_part.split('_')))
        if len(indices) != 7:
            return None

        chars = []
        chars.append(PROVINCES[indices[0]])   # 省份
        chars.append(ALPHABETS[indices[1]])   # 字母
        for i in range(2, 7):
            chars.append(ADS[indices[i]])     # 后5位

        # 顶点在第3段（index=3）
        vertex_part = parts[3]
        vertices = vertex_part.split('_')
        pts = []
        for v in vertices:
            x, y = v.split('&')
            pts.append([int(x), int(y)])

        return chars, pts
    except:
        return None

def order_points(pts):
    """
    将4个点排列为：左上、右上、右下、左下
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上：x+y最小
    rect[2] = pts[np.argmax(s)]   # 右下：x+y最大

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：y-x最小
    rect[3] = pts[np.argmax(diff)]  # 左下：y-x最大

    return rect

def warp_plate(img, pts):
    """
    透视变换，将车牌矫正为标准矩形 440x140
    """
    rect = order_points(pts)
    dst  = np.array([[0,0],[440,0],[440,140],[0,140]],
                    dtype=np.float32)
    M     = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (440, 140))
    return warped

def split_chars(plate_gray):
    """
    将440x140的灰度车牌图像切割为7个字符
    基于固定位置切割（标准蓝牌布局）
    返回7个 40x20 的字符图像列表
    """
    # 二值化
    _, binary = cv2.threshold(plate_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 固定切割位置（440px宽，7个字符）
    # 第1字符：省份（稍宽）
    # 第2字符：字母
    # 第3~7字符：数字字母，中间有圆点分隔
    char_x = [10, 70, 130, 180, 230, 295, 345]
    char_w = 50
    chars  = []

    for x in char_x:
        char = binary[5:135, x:x+char_w]
        if char.size == 0:
            char = np.zeros((130, char_w), dtype=np.uint8)
        char_resized = cv2.resize(char, (CHAR_W, CHAR_H),
                                  interpolation=cv2.INTER_AREA)
        chars.append(char_resized)

    return chars

def build_char_dataset(ccpd_dir, output_dir, max_imgs=None):
    """
    主函数：遍历CCPD数据集，切割字符保存到分类目录
    目录结构：output_dir/类别索引/xxx.jpg
    """
    # 所有类别
    ALL_CHARS = PROVINCES + ALPHABETS + ADS
    NUM_CLASSES = len(ALL_CHARS)
    print(f"总类别数: {NUM_CLASSES}")

    # 创建输出目录
    for i in range(NUM_CLASSES):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    files = [f for f in os.listdir(ccpd_dir)
             if f.lower().endswith('.jpg')]
    if max_imgs:
        files = files[:max_imgs]

    print(f"共 {len(files)} 张图片")

    saved   = [0] * NUM_CLASSES
    success = 0
    fail    = 0

    for fname in tqdm(files, desc="切割字符"):
        result = parse_filename(fname)
        if result is None:
            fail += 1
            continue

        chars, pts = result
        fpath = os.path.join(ccpd_dir, fname)
        img   = cv2.imread(fpath)
        if img is None:
            fail += 1
            continue

        # 透视变换矫正车牌
        try:
            plate_color = warp_plate(img, pts)
        except:
            fail += 1
            continue

        plate_gray = cv2.cvtColor(plate_color, cv2.COLOR_BGR2GRAY)

        # 切割7个字符
        char_imgs = split_chars(plate_gray)
        if len(char_imgs) != 7:
            fail += 1
            continue

        # 按类别保存
        for i, (char, cimg) in enumerate(zip(chars, char_imgs)):
            if char not in ALL_CHARS:
                continue
            idx = ALL_CHARS.index(char)
            save_name = os.path.join(
                output_dir, str(idx),
                f"{success}_{i}.jpg")
            cv2.imwrite(save_name, cimg)
            saved[idx] += 1

        success += 1

    print(f"\n处理完成！成功: {success}，失败: {fail}")
    print(f"字符图片保存到: {output_dir}")

    # 统计各类别数量
    print("\n各类别样本数（前10）:")
    for i in range(min(10, NUM_CLASSES)):
        print(f"  {ALL_CHARS[i]}: {saved[i]}")

    total_chars = sum(saved)
    print(f"\n总字符图片数: {total_chars}")

if __name__ == '__main__':
    build_char_dataset(
        ccpd_dir   = r"D:\Desktop\ccpd_base",
        output_dir = r"D:\Desktop\char_dataset",
        max_imgs   = None   # 改成None处理全部13286张
    )