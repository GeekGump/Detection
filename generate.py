import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import yaml
import numpy as np

# -------------------------- 配置参数 --------------------------
# 字符集配置（排除I和O的大写字母+数字）
TOTAL_SAMPLES = 500          # 总样本数
EXCLUDE_CHARS = {'I', 'O'}
LETTERS = [chr(i) for i in range(65, 91) if chr(i) not in EXCLUDE_CHARS]
DIGITS = [str(i) for i in range(10)]
CHARS = LETTERS + DIGITS  # 总字符集（34个）

# 图片与标注配置
NUM_CHARS_PER_IMG = (5, 7)       # 每张图字符数量范围
FONT_SIZE_RANGE = (30, 50)       # 字体大小范围
BG_COLOR_RANGE = (0, 255)        # 背景颜色（RGB）
TEXT_COLOR_OFFSET = 128          # 字符与背景的最小亮度差（避免看不清）
BLUR_RADIUS = 1.5                # 背景模糊半径
IMG_SIZE = (256, 256)            # 图片尺寸（宽, 高）
FONT_DIR = "fonts"               # 字体文件夹路径（需自行放入.ttf字体）
DATASET_ROOT = "dataset"         # 数据集根目录
os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "images"), exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "labels"), exist_ok=True)
os.makedirs(FONT_DIR, exist_ok=True)
SPLITS = ["train", "test", "val"]# 数据集划分
SPLIT_RATIOS = [0.5, 0.2, 0.2]   # 划分比例（5:2:2）

# -------------------------- 初始化配置 --------------------------
# 1. 生成字符映射表（字符→索引）
char_to_idx = {c: i for i, c in enumerate(CHARS)}
with open(os.path.join(DATASET_ROOT, "char_map.yaml"), "w") as f:
    yaml.dump({"char_map": char_to_idx}, f, default_flow_style=False)
print(f"字符映射已保存至：{DATASET_ROOT}/char_map.yaml")

# 额外生成 YOLO 格式的数据集配置
# build names mapping from char_to_idx (invert char->idx to idx->char)
idx_to_char = {v: k for k, v in char_to_idx.items()}
names = {i: idx_to_char[i] for i in sorted(idx_to_char.keys())}

yolo_cfg = {
    "path": DATASET_ROOT,            # dataset root dir
    "train": "images/train",         # train images (relative to 'path')
    "val": "images/val",             # val images (relative to 'path')
    "test": "images/test",           # test images (optional)
    "names": names
}
yolo_yaml_path = os.path.join(DATASET_ROOT, "OCR.yaml")
with open(yolo_yaml_path, "w") as f:
    yaml.dump(yolo_cfg, f, default_flow_style=False, sort_keys=False)
print(f"YOLO 数据集配置已保存至：{yolo_yaml_path}")



# 2. 检查字体文件（需确保fonts文件夹下有.ttf字体）
if not os.path.exists(FONT_DIR):
    raise FileNotFoundError(f"字体文件夹{FONT_DIR}不存在，请放入.ttf字体文件！")
FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(".TTF") or f.endswith(".ttf")]
if not FONTS:
    raise ValueError(f"{FONT_DIR}中未找到.ttf字体文件！")

# 3. 创建目录结构
for split in SPLITS:
    os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)
print("目录结构已创建：images/train/test/val + labels/train/test/val")

# -------------------------- 辅助函数 --------------------------
def get_luminance(rgb):
    """计算RGB颜色的亮度（ITU-R BT.601标准）"""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

def is_overlapping(bbox1, bbox2):
    """检查两个边界框是否重叠"""
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2
    return not (x2 <= a1 or x1 >= a2 or y2 <= b1 or y1 >= b2)

# -------------------------- 核心生成逻辑 --------------------------
def generate_dataset(total_samples=100):
    """生成数据集（默认1万张图）"""
    sample_idx = 0
    for split, ratio in zip(SPLITS, SPLIT_RATIOS):
        num_samples = int(total_samples * ratio)
        print(f"正在生成{split}数据：{num_samples}张")
        
        for i in range(num_samples):
            # 1. 随机选择字符（允许重复）
            num_chars = random.randint(*NUM_CHARS_PER_IMG)
            selected_chars = random.choices(CHARS, k=num_chars)
            
            # # 随机背景色+模糊
            bg_color = (random.randint(*BG_COLOR_RANGE), 
                        random.randint(*BG_COLOR_RANGE), 
                        random.randint(*BG_COLOR_RANGE))
            # # 2. 直接用随机背景色创建图像（无需后续paste）
            # img = Image.new("RGB", IMG_SIZE, bg_color)
            # draw = ImageDraw.Draw(img)
            # img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            image_size = IMG_SIZE
            noise = np.random.rand(*image_size, 3) * 255
            img = Image.fromarray(noise.astype('uint8'))
            # Apply a slight Gaussian blur to make the noise slightly blurred
            blur_radius = 1.0  # reduce for less blur, increase for more (e.g., 0.5..2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            draw = ImageDraw.Draw(img)  # 模糊后重新初始化Draw对象
            
            # 3. 绘制字符并记录标注
            char_infos = []  # 存储（字符, 边界框）
            for char in selected_chars:
                # 随机字体
                font_path = random.choice(FONTS)
                font_size = random.randint(*FONT_SIZE_RANGE)
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except IOError:
                    continue  # 跳过加载失败的字体
                
                # 随机字符颜色（确保与背景有对比度）
                # while True:
                text_color = (random.randint(*BG_COLOR_RANGE), 
                                random.randint(*BG_COLOR_RANGE), 
                                random.randint(*BG_COLOR_RANGE))
                #     if abs(get_luminance(bg_color) - get_luminance(text_color)) > TEXT_COLOR_OFFSET:
                #         break
                
                # 计算字符边界框
                try:
                    bbox = font.getbbox(char)  # PIL 9.2.0+支持
                except AttributeError:
                    # 旧版本PIL兼容：用getsize
                    width, height = font.getsize(char)
                    bbox = (0, 0, width, height)
                char_w = bbox[2] - bbox[0]
                char_h = bbox[3] - bbox[1]
                
                # 尝试放置字符（避免重叠，最多100次尝试）
                placed = False
                for _ in range(100):
                    x = random.randint(0, IMG_SIZE[0] - char_w-40)
                    y = random.randint(0, IMG_SIZE[1] - char_h-40)
                    # char_bbox = (x, y, x + char_w, y + char_h)
                    char_bbox = draw.textbbox((x, y), char, font=font)
                    
                    # 检查重叠
                    if not any(is_overlapping(char_bbox, eb) for _, eb in char_infos):
                        # 绘制字符
                        draw.text((x, y), char, fill=text_color, font=font)
                        text_box = draw.textbbox((x, y), char, font=font)
                        char_infos.append((char, text_box))
                        placed = True
                        break
                
                if not placed:
                    print(f"跳过字符{char}（无法放置）")
                    continue
            
            # 4. 保存图片与标注
            img_name = f"{sample_idx:04d}.png"
            img_path = os.path.join(DATASET_ROOT, "Images", split, img_name)
            img.save(img_path)
            
            # 保存标注（格式：x1 y1 x2 y2 字符）
            label_path = os.path.join(DATASET_ROOT, "labels", split, img_name.replace(".png", ".txt"))
            with open(label_path, "w") as f:
                for char, bbox in char_infos:
                    idx = char_to_idx[char]
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    w = bbox[2] - bbox[0]
                    h = abs(bbox[3] - bbox[1])  # Ensure height is always positive
                    # 规范化坐标到[0,1]
                    label = (x_center / IMG_SIZE[0], y_center / IMG_SIZE[1],
                             w / IMG_SIZE[0], h / IMG_SIZE[1])
                    f.write(f"{idx} {label[0]} {label[1]} {label[2]} {label[3]}\n")
                    # bbox = (bbox[0] / IMG_SIZE[0], bbox[1] / IMG_SIZE[1],
                    #         bbox[2] / IMG_SIZE[0], bbox[3] / IMG_SIZE[1])
                    # f.write(f"{idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            sample_idx += 1
            print(f"已生成：{sample_idx}/{total_samples}", end="\r")
    
    print(f"数据集生成完成！共{total_samples}张图，保存至{DATASET_ROOT}")

# -------------------------- 运行程序 --------------------------
if __name__ == "__main__":
    generate_dataset(total_samples=TOTAL_SAMPLES)  # 可调整总样本数