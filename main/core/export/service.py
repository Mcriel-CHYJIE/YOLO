"""导出业务逻辑"""
from pathlib import Path
from datetime import datetime
from main.core.base import ROOT

def run_export(weights_path: str, fmt: str, imgsz: int, half: bool, int8: bool,
               nms: bool, gpu_ok: bool, log_callback=None):
    """执行模型导出，返回 (ok: bool, message: str)"""
    w = weights_path
    if not w or not Path(w).exists():
        return False, 'Weights file not found'
    try:
        from ultralytics import YOLO
        model = YOLO(w)
        kw = dict(
            format=fmt, imgsz=imgsz, half=half, int8=int8, nms=nms,
            device='0' if gpu_ok else 'cpu',
        )
        if fmt == 'onnx':
            kw['opset'] = 12
            kw['simplify'] = True
            kw['dynamic'] = False
        out = model.export(**kw)
        sz = Path(out).stat().st_size / 1e6
        lines = [f'{fmt.upper()} exported!', f'  Path: {out}', f'  Size: {sz:.2f} MB']
        if fmt == 'onnx':
            lines.append(f'  Opset: {kw.get("opset","—")} | Simplify: {kw.get("simplify","—")}')
        return True, '\n'.join(lines)
    except Exception as e:
        import traceback; traceback.print_exc()
        return False, str(e)
