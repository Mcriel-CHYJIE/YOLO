"""训练工作线程"""
from scripts.tabs.base import ROOT, DATA_YAML, IS_FALL, StreamEmitter
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
import json


class Trainer(QThread):
    log = pyqtSignal(str); status = pyqtSignal(str,float,float,float); chart = pyqtSignal(); done = pyqtSignal(bool,str)

    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self._stop = False
        self.history = {'epoch':[],'train_loss':[],'mAP50':[],'mAP50_95':[],'precision':[],'recall':[]}
        self.best_map = 0.0
        self.stdout_emitter = StreamEmitter(self.log)
        self.stderr_emitter = StreamEmitter(self.log)

    def stop(self): self._stop = True

    def _check_gpu_memory(self):
        import torch
        if not torch.cuda.is_available(): return None
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            return {'total': total_mem, 'allocated': allocated_mem, 'free': free_mem}
        except: return None

    def _save_log(self, ok, error=''):
        try:
            d = ROOT / 'runs' / 'training_logs'; d.mkdir(parents=True, exist_ok=True)
            best_m95 = 0.0
            if self.history['mAP50_95'] and any(v > 0 for v in self.history['mAP50_95']):
                best_m95 = max(self.history['mAP50_95'])
            entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'project': ROOT.name, 'status': 'success' if ok else 'failed', 'error': error,
                'config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v for k, v in self.cfg.items()},
                'results': {'best_mAP50': round(self.best_map, 4), 'best_mAP50_95': round(best_m95, 4),
                    'final_mAP50': round(self.history['mAP50'][-1], 4) if self.history['mAP50'] else 0,
                    'final_loss': round(self.history['train_loss'][-1], 4) if self.history['train_loss'] else 0,
                    'epochs_trained': len(self.history['epoch'])},
                'history': {k: [round(v, 4) for v in vals] for k, vals in self.history.items()},
            }
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(d / f'{ts}.json', 'w', encoding='utf-8') as f: json.dump(entry, f, ensure_ascii=False, indent=2)
            hp = d / 'training_history.json'
            h = json.loads(hp.read_text('utf-8')) if hp.exists() else []
            h.append(entry); hp.write_text(json.dumps(h, ensure_ascii=False, indent=2), 'utf-8')
            self.log.emit(f'📁 Log saved: {ts}.json')
        except: pass

    def run(self):
        import sys, torch
        original_stdout = sys.stdout; original_stderr = sys.stderr
        try:
            sys.stdout = self.stdout_emitter; sys.stderr = self.stderr_emitter
            from ultralytics import YOLO
            cfg = self.cfg; m = YOLO(cfg['model'])
            exp = f'{ROOT.name}_{datetime.now().strftime("%m%d_%H%M")}'
            self.log.emit(f'🚀 {cfg["model"]} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
            if cfg.get('lr0', 0) <= 0: cfg['lr0'] = 0.001; self.log.emit(f'⚠️  LR0 was 0, reset to 0.001')
            if torch.cuda.is_available() and cfg['device'] != 'cpu':
                gi = self._check_gpu_memory()
                if gi: self.log.emit(f'💻 GPU Memory: {gi["total"]:.1f}GB total, {gi["free"]:.1f}GB free')
                torch.cuda.empty_cache()
            def cb(t):
                if self._stop: t.stop = True; return
                try:
                    ep = t.epoch; loss = 0.0
                    if hasattr(t,'loss') and t.loss is not None: loss = float(t.loss)
                    if loss == 0.0 and hasattr(t,'tloss') and t.tloss is not None: loss = float(t.tloss)
                    mt = getattr(t,'metrics',None) or {}
                    m50 = float(mt.get('metrics/mAP50(B)',0)); m95 = 0.0
                    for k in ['metrics/mAP50_95(B)','metrics/mAP50_95','metrics/mAP_0.5:0.95']:
                        v = mt.get(k)
                        if v is not None:
                            try: m95 = float(v); break
                            except: pass
                    p = float(mt.get('metrics/precision(B)',0)); r = float(mt.get('metrics/recall(B)',0))
                    self.history['epoch'].append(ep); self.history['train_loss'].append(loss)
                    self.history['mAP50'].append(m50); self.history['mAP50_95'].append(m95)
                    self.history['precision'].append(p); self.history['recall'].append(r)
                    if m50 > self.best_map: self.best_map = m50
                    self.status.emit(f'Epoch {ep}/{cfg["epochs"]}', min(ep/max(cfg['epochs'],1),1.0), self.best_map, m50)
                    self.chart.emit()
                except: pass
            m.add_callback('on_train_epoch_end', cb)
            train_args = dict(data=DATA_YAML, epochs=cfg['epochs'], batch=cfg['batch'], imgsz=cfg['imgsz'],
                lr0=cfg['lr0'], lrf=cfg['lrf'], optimizer=cfg['optimizer'], patience=cfg['patience'],
                device=cfg['device'], warmup_epochs=3, warmup_momentum=0.8, cos_lr=cfg['cos_lr'],
                flipud=0.0 if IS_FALL else 0.3, fliplr=0.5, mosaic=1.0, mixup=0.2, workers=cfg.get('workers', 4),
                iou=cfg['iou'], close_mosaic=cfg['close_mosaic'],
                copy_paste=cfg['copy_paste'] if cfg['copy_paste'] > 0 else 0,
                degrees=cfg['degrees'] if cfg['degrees'] > 0 else 0, multi_scale=cfg['multi_scale'],
                hsv_h=cfg.get('hsv_h', 0.015), hsv_s=cfg.get('hsv_s', 0.7), hsv_v=cfg.get('hsv_v', 0.4),
                translate=cfg.get('translate', 0.15), scale=cfg.get('scale', 0.6),
                project='runs', name=exp, exist_ok=True, amp=True, verbose=False)
            train_args = {k: v for k, v in train_args.items() if v is not None}
            try: m.train(**train_args)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    self.log.emit('💡 Reduce batch size or image size'); raise
                raise
            stopped = self._stop
            self._save_log(ok=not stopped, error='Stopped by user' if stopped else '')
            self.done.emit(not stopped, f'⏹ Stopped' if stopped else f'✅ Done | Best mAP@0.5 = {self.best_map:.4f}')
        except BaseException as e:
            import traceback; traceback.print_exc()
            self._save_log(ok=False, error=str(e))
            self.log.emit(f'❌ {e}')
            self.done.emit(False, f'Failed: {e}')
        finally:
            sys.stdout = original_stdout; sys.stderr = original_stderr
