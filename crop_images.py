import os
from PIL import Image

src_dir = 'images'
dst_dir = os.path.join(src_dir, 'cropped')
os.makedirs(dst_dir, exist_ok=True)
target_size = (320, 140)

for fname in os.listdir(src_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(src_dir, fname)
        img = Image.open(img_path)
        img = img.convert('RGB')
        w, h = img.size
        # 居中裁剪或加白边
        if w / h > target_size[0] / target_size[1]:
            new_h = target_size[1]
            new_w = int(w * new_h / h)
        else:
            new_w = target_size[0]
            new_h = int(h * new_w / w)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_size[0]) // 2
        top = (new_h - target_size[1]) // 2
        img_cropped = img_resized.crop((left, top, left + target_size[0], top + target_size[1]))
        img_cropped.save(os.path.join(dst_dir, fname))
        print(f'Cropped: {fname}')

print('全部裁剪完成，已保存到 images/cropped/')
