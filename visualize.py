import os
import glob
import cv2
import yaml
import argparse
import numpy as np
import sys
import numpy as np

# visualize.py
# 从 dataset/Images/train 加载图片，从 dataset/label/train 加载标签，使用 dataset/charmap.yaml 显示字符
# 左右键切换图片，q 或 ESC 退出


def load_charmap(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    # 支持两种情况：1) 顶层是 {'char_map': {A:0,...}} 2) 直接是 {A:0,...}
    if isinstance(data, dict) and 'char_map' in data and isinstance(data['char_map'], dict):
        cmap = data['char_map']
    elif isinstance(data, dict):
        cmap = data
    else:
        return {}

    idx2char = {}
    for k, v in cmap.items():

        try:
            iv = int(v)
            idx2char[iv] = str(k)
        except Exception:
            # 无法解析则忽略该项
            continue
    return idx2char

def list_images(img_dir):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    files = sorted(files)
    return files

def load_labels_for_image(label_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, base + '.txt')
    if not os.path.exists(label_path):
        return []
    lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split()
            if len(parts) < 5:
                continue
            try:
                idx = int(float(parts[0]))
                vals = list(map(float, parts[1:5]))
                lines.append((idx, vals))
            except:
                continue
    return lines

def draw_annotations(img, annots, idx2char, color=(0,255,0)):
    h, w = img.shape[:2]

    for (idx, vals) in annots:
        cx, cy, bw, bh = vals
        # 检测是否为归一化坐标（<=1）或像素坐标（>1）
        if max(cx, cy, bw, bh) <= 1.0:
            x_center = cx * w
            y_center = cy * h
            box_w = bw * w
            box_h = bh * h
        else:
            x_center = cx
            y_center = cy
            box_w = bw
            box_h = bh

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        # clip
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        text = idx2char.get(idx, str(idx))
        # put label above box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tx = x1
        ty = y1 - 6
        if ty - th < 0:
            ty = y1 + th + 6
        cv2.rectangle(img, (tx, ty - th - 2), (tx + tw, ty + 2), color, -1)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

def main(root):
    img_dir = os.path.join(root, 'dataset', 'Images', 'train')
    label_dir = os.path.join(root, 'dataset', 'labels', 'train')
    charmap_path = os.path.join(root, 'dataset', 'char_map.yaml')

    imgs = list_images(img_dir)
    if not imgs:
        print('未找到图片: ', img_dir)
        return

    if not os.path.exists(charmap_path):
        print('未找到 chmap 文件: ', charmap_path)
        return

    idx2char = load_charmap(charmap_path)

    i = 0
    n = len(imgs)
    window = 'AnnotViewer'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    left_codes = {81, 2424832, 65361}
    right_codes = {83, 2555904, 65363}
    quit_codes = {27, ord('q'), ord('Q')}

    while True:
        img_path = imgs[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # some Windows paths with unicode may fail with cv2.imread; try fallback
        if img is None:
            try:
                with open(img_path, 'rb') as f:
                    data = f.read()
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                print('无法读取图片:', img_path)
                i = (i + 1) % n
                continue

        annots = load_labels_for_image(label_dir, img_path)
        vis = img.copy()
        draw_annotations(vis, annots, idx2char)
        # 状态栏文字：文件名 与 索引信息
        fname = os.path.basename(img_path)
        info = f'{i+1}/{n}  {fname}'
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(window, vis)

        k = cv2.waitKey(0)
        if k is None:
            break
        # k may be negative on some systems
        if k in quit_codes:
            break
        if k in left_codes:
            i = (i - 1) % n
            continue
        if k in right_codes:
            i = (i + 1) % n
            continue
        # fallback: allow 'a'/'d' keys
        if k & 0xFF == ord('a'):
            i = (i - 1) % n
            continue
        if k & 0xFF == ord('d'):
            i = (i + 1) % n
            continue
        # any other key: next
        i = (i + 1) % n

    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='可视化 dataset 标注 (YOLO 格式 idx x_center ycenter w h)')
    parser.add_argument('--root', type=str, default='.', help='工程根目录，包含 dataset 文件夹')
    args = parser.parse_args()

    main(args.root)