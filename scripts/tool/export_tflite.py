"""Export trained .pt to TFLite INT8 (standalone, for 11_fall project)."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from ultralytics import YOLO

PT = Path(r"D:\Projects\11_fall\runs\detect\runs\11_fall_0520_1756\weights\best.pt")
DATA = r"D:\Projects\11_fall\datasets\data.yaml"

assert PT.exists(), f"Model not found: {PT}"

model = YOLO(str(PT))

# FP32 (already done, but fast if cached)
print("[1/3] FP32 TFLite...")
fp32 = model.export(format='tflite', imgsz=640, int8=False, simplify=True)
print(f"  OK: {fp32}")

# FP16
print("[2/3] FP16 TFLite...")
fp16 = model.export(format='tflite', imgsz=640, half=True, simplify=True)
print(f"  OK: {fp16}")

# INT8 full integer quantization
print("[3/3] INT8 TFLite...")
int8 = model.export(format='tflite', imgsz=640, int8=True, data=DATA, simplify=True)
print(f"  OK: {int8}")

# Copy to 10_Detection assets
DET = Path(r"D:\Projects\10_Detection\app\src\main\assets\model")
DET.mkdir(parents=True, exist_ok=True)

import shutil
for src in [fp32, fp16, int8]:
    name = Path(src).name
    dst = DET / name
    shutil.copy2(src, dst)
    print(f"  Copied to: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

# Labels
labels_dst = DET / "labels.txt"
with open(labels_dst, 'w') as f:
    f.write('\n'.join(['standing', 'sitting', 'squatting', 'fallen']))
print(f"  Labels: {labels_dst}")

print("\nDone! TFLite models ready for Android deployment.")
