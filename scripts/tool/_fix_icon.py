"""生成多分辨率 YOLO.ico (16×16 ~ 256×256)"""
from PIL import Image

src = Image.open('../../YOLO.png').convert('RGBA')
sizes = [16, 32, 48, 64, 128, 256]
frames = [src.resize((s, s), Image.LANCZOS) for s in sizes]

# ICO 格式要求最大图排第一（Windows 取最佳匹配）
frames[0].save('YOLO.ico', format='ICO', sizes=[(s, s) for s in sizes],
               append_images=frames[1:])
print('OK: YOLO.ico with sizes', sizes)

# Verify
v = Image.open('../../YOLO.ico')
print(f'Frames: {v.n_frames}')
for i in range(v.n_frames):
    v.seek(i)
    print(f'  [{i}] {v.size}')
