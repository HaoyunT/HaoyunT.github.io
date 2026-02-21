import os
from PIL import Image

src_dir = 'images'
dst_dir = os.path.join(src_dir, 'cropped')
os.makedirs(dst_dir, exist_ok=True)
target_size = (320, 140)
project_imgs = [
    '自动驾驶照片0.png',
    'airsim图片.png',
    '星际争霸图片.png',
    '强化学习网站图片.png'
]

for fname in project_imgs:
    img_path = os.path.join(src_dir, fname)
    if not os.path.exists(img_path):
        print(f'未找到: {fname}')
        continue
    img = Image.open(img_path)
    img = img.convert('RGB')
    w, h = img.size
    # 保持内容居中，object-fit: contain风格
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    bg = Image.new('RGB', target_size, (255,255,255))
    left = (target_size[0] - new_w) // 2
    top = (target_size[1] - new_h) // 2
    bg.paste(img_resized, (left, top))
    bg.save(os.path.join(dst_dir, fname))
    print(f'Cropped: {fname}')

print('项目图片裁剪完成，已保存到 images/cropped/')
