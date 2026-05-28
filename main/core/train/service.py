"""训练业务逻辑 — 配置构建 + 训练工作线程"""
from pathlib import Path
from main.core.base import ROOT, cfg, DATA_YAML, StreamEmitter
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime
import json, torch, threading
from ultralytics import YOLO
from main.config import ATTENTION_FILE, load_paths


# ══════════════════════════════════════════════════════════════
# 配置构建
# ══════════════════════════════════════════════════════════════

def build_train_config(
    model_combo_items, model_current_text, studio_gpu_ok, studio_cpu_count,
    ep_val, bs_val, sz_text, opt_text, dev_idx, sch_idx,
    pt_val, lr0_val, lrf_val, wu_val, wk_val,
    iou_val, cm_val, cp_val, dg_val, ms_checked,
    momentum_val, wd_val, hsv_h_val, hsv_s_val, hsv_v_val,
    translate_val, scale_val, cls_pw_val,
    mosaic_val=1.0, mixup_val=0.2,
    flip_lr_val=0.5, flipud_val=0.0,
    shear_val=0.0, perspective_val=0.0,
    dropout_val=0.0, warmup_momentum_val=0.8,
    amp_val=True, cache_val=False,
    lora_rank=0,
) -> dict:
    """从 UI 控件值构建训练参数字典"""
    print(f'[build_train_config] entered', flush=True)
    if lr0_val <= 0:
        lr0_val = 0.001
    print(f'[build_train_config] cfg.get', flush=True)
    g = cfg.get('training', {})
    print(f'[build_train_config] model_name={model_current_text}', flush=True)
    model_name = model_current_text
    if model_name.endswith('.yaml'):
        model_path = model_name
    else:
        print(f'[build_train_config] resolving model path...', flush=True)
        _p = load_paths()
        print(f'[build_train_config] paths={_p}', flush=True)
        _md = _p.get('models_dir', str(ROOT / 'models'))
        print(f'[build_train_config] models_dir={_md}', flush=True)
        try:
            local = Path(_md) / model_name
            print(f'[build_train_config] local={local}', flush=True)
            _exists = local.exists()
            print(f'[build_train_config] exists={_exists}', flush=True)
            model_path = str(local) if _exists else model_name
        except Exception as _e:
            print(f'[build_train_config] Path error: {_e}', flush=True)
            model_path = model_name
    print(f'[build_train_config] building dict...', flush=True)
    result = dict(
        model=model_path, epochs=ep_val, batch=bs_val, imgsz=int(sz_text),
        lr0=lr0_val, lrf=lrf_val, optimizer=opt_text, patience=pt_val,
        device='0' if studio_gpu_ok and dev_idx == 0 else 'cpu',
        cos_lr=sch_idx == 0, warmup_epochs=wu_val, workers=wk_val,
        momentum=momentum_val, weight_decay=wd_val,
        mixup=mixup_val,
        cls_pw=cls_pw_val,
        mosaic=mosaic_val,
        iou=iou_val,
        close_mosaic=cm_val, copy_paste=cp_val, degrees=dg_val,
        multi_scale=ms_checked,
        hsv_h=hsv_h_val, hsv_s=hsv_s_val, hsv_v=hsv_v_val,
        translate=translate_val, scale=scale_val,
        flip_lr=flip_lr_val, flipud=flipud_val,
        shear=shear_val, perspective=perspective_val,
        dropout=dropout_val, warmup_momentum=warmup_momentum_val,
        amp=amp_val, cache=cache_val,
        lora_rank=lora_rank,
    )
    _mdl = result.get('model', '?')
    print(f'[build_train_config] done, model={_mdl}', flush=True)
    return result


# ══════════════════════════════════════════════════════════════
# Trainer — 训练工作线程
# ══════════════════════════════════════════════════════════════

class Trainer(QObject):
    """线程内训练管理器 — 非 QThread，用 threading.Thread 获得完整原生栈"""
    log = pyqtSignal(str); status = pyqtSignal(str,float,float,float,float); chart = pyqtSignal(); done = pyqtSignal(bool,str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop_event = threading.Event()
        self._thread = None
        self.history = {'epoch':[],'train_loss':[],'mAP50':[],'mAP50_95':[],'precision':[],'recall':[]}
        self.best_map = 0.0
        self.stdout_emitter = StreamEmitter(self.log)
        self.stderr_emitter = StreamEmitter(self.log)

    def stop(self):
        self._stop_event.set()

    def isRunning(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        print(f'[Trainer] start() called', flush=True)
        threading.stack_size(4 * 1024 * 1024)
        self._thread = threading.Thread(target=self._run, daemon=True)
        print(f'[Trainer] thread created, starting...', flush=True)
        self._thread.start()
        print(f'[Trainer] thread started, id={self._thread.ident}', flush=True)

    def _check_gpu_memory(self):
        if not torch.cuda.is_available(): return None
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - allocated_mem
            return {'total': total_mem, 'allocated': allocated_mem, 'free': free_mem}
        except: return None

    def _save_log(self, ok, error=''):
        try:
            d = Path(self._train_dir) / 'training_logs'; d.mkdir(parents=True, exist_ok=True)
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
            self.log.emit(f' Log saved: {ts}.json')
        except: pass

    def _run(self):
        import sys
        import matplotlib
        matplotlib.use('Agg')  # 禁止弹出图表窗口
        print(f'[Trainer] _run entered', flush=True)
        torch.set_num_threads(1)
        original_stdout = sys.stdout; original_stderr = sys.stderr
        try:
            sys.stdout = self.stdout_emitter; sys.stderr = self.stderr_emitter
            cfg = self.cfg
            _paths = load_paths()
            self._train_dir = _paths.get('train_output', '')
            
            # 检查是否从头开始训练
            if not cfg['model'] or cfg['model'].strip() == '':
                self.log.emit(' [Scratch] Training from scratch (no pretrained weights)')
                m = YOLO('yolo11n.yaml')  # 使用默认配置创建新模型
            else:
                m = YOLO(cfg['model'])
            
            # ── 注入注意力模块 ──
            try:
                import json
                if ATTENTION_FILE.exists():
                    attn = json.loads(ATTENTION_FILE.read_text('utf-8')).get('type', 'none')
                    if attn != 'none':
                        from main.core.train.attention import inject_attention
                        replaced = inject_attention(m.model, attn)
                        self.log.emit(f' [{attn.upper()}] Injected attention into {replaced} C2f blocks')
            except Exception as e:
                self.log.emit(f' [WARN] Attention injection failed: {e}')

            # ── LoRA 注入 ──
            lora_r = cfg.get('lora_rank', 0)
            if lora_r > 0:
                try:
                    from main.core.train.attention import inject_lora
                    replaced = inject_lora(m.model, lora_r)
                    self.log.emit(f' [LoRA] Injected LoRA (rank={lora_r}) into {replaced} Conv2d layers')
                    # 统计可训练参数量
                    total = sum(p.numel() for p in m.model.parameters())
                    trainable = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
                    self.log.emit(f'  Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total')
                except Exception as e:
                    self.log.emit(f' [WARN] LoRA injection failed: {e}')
            
            exp = f'{ROOT.name}_{datetime.now().strftime("%m%d_%H%M")}'
            model_name = cfg['model'] if cfg['model'] else 'Scratch'
            self.log.emit(f' {model_name} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
            if cfg.get('lr0', 0) <= 0: cfg['lr0'] = 0.001; self.log.emit(f'  LR0 was 0, reset to 0.001')
            if torch.cuda.is_available() and cfg['device'] != 'cpu':
                gi = self._check_gpu_memory()
                if gi: self.log.emit(f' GPU Memory: {gi["total"]:.1f}GB total, {gi["free"]:.1f}GB free')
            def cb(t):
                if self._stop_event.is_set(): t.stop = True; return
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
                    self.status.emit(f'Epoch {ep}/{cfg["epochs"]}', min(ep/max(cfg['epochs'],1),1.0), self.best_map, m50, loss)
                    self.chart.emit()
                except: pass
            m.add_callback('on_train_start', lambda t: self.log.emit(' Training started (first epoch may take a while to initialize)'))
            # ── 每个 batch 处理完后发射进度（每 10% 报一次） ──
            _last_batch_pct = [0]
            def on_batch(t):
                try:
                    ni = t.ni if hasattr(t, 'ni') else 0
                    nf = t.epoch_len if hasattr(t, 'epoch_len') else 1
                    pct = int(ni / max(nf, 1) * 100)
                    if pct >= _last_batch_pct[0] + 10:
                        _last_batch_pct[0] = pct
                        self.log.emit(f'  Epoch progress: {pct}% ({ni}/{nf} batches)')
                except:
                    pass
            m.add_callback('on_train_batch_end', on_batch)
            m.add_callback('on_train_epoch_end', cb)
            train_args = dict(data=str(DATA_YAML), epochs=cfg['epochs'], batch=cfg['batch'], imgsz=cfg['imgsz'],
                lr0=cfg['lr0'], lrf=cfg['lrf'], optimizer=cfg['optimizer'], patience=cfg['patience'],
                device=cfg['device'], warmup_epochs=cfg.get('warmup_epochs', 3), warmup_momentum=cfg.get('warmup_momentum', 0.8), cos_lr=cfg['cos_lr'],
                momentum=cfg.get('momentum', 0.937), weight_decay=cfg.get('weight_decay', 0.0005),
                cls_pw=cfg.get('cls_pw', 0.75),
                flipud=cfg.get('flipud', 0.0), fliplr=cfg.get('flip_lr', 0.5), mosaic=cfg.get('mosaic', 1.0), mixup=cfg.get('mixup', 0.2), workers=cfg.get('workers', 4),
                iou=cfg['iou'], close_mosaic=cfg['close_mosaic'],
                copy_paste=cfg['copy_paste'] if cfg['copy_paste'] > 0 else 0,
                degrees=cfg['degrees'] if cfg['degrees'] > 0 else 0, multi_scale=cfg['multi_scale'],
                hsv_h=cfg.get('hsv_h', 0.015), hsv_s=cfg.get('hsv_s', 0.7), hsv_v=cfg.get('hsv_v', 0.4),
                translate=cfg.get('translate', 0.15), scale=cfg.get('scale', 0.6),
                shear=cfg.get('shear', 0.0), perspective=cfg.get('perspective', 0.0),
                dropout=cfg.get('dropout', 0.0),
                amp=cfg.get('amp', True), cache=cfg.get('cache', False),
                project=self._train_dir, name=exp, exist_ok=True, verbose=False)
            train_args = {k: v for k, v in train_args.items() if v is not None}
            # ── 校验数据集路径 ──
            _dp = train_args.get('data', '')
            if not _dp or not Path(_dp).exists():
                self.log.emit(f' [ERROR] Dataset not configured or not found')
                self.log.emit(f'  Checked: {_dp}')
                self.log.emit(f'  Go to Settings → Dataset dir to configure the correct path.')
                raise FileNotFoundError(f'Dataset not found: {_dp}. Configure it in Settings → Dataset dir.')
            try: m.train(**train_args)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    self.log.emit(' Reduce batch size or image size'); raise
                raise
            stopped = self._stop_event.is_set()
            self._save_log(ok=not stopped, error='Stopped by user' if stopped else '')
            self.done.emit(not stopped, f' Stopped' if stopped else f' Done | Best mAP@0.5 = {self.best_map:.4f}')
        except BaseException as e:
            import traceback; traceback.print_exc()
            self._save_log(ok=False, error=str(e))
            self.log.emit(f' {e}')
            self.done.emit(False, f'Failed: {e}')
        finally:
            sys.stdout = original_stdout; sys.stderr = original_stderr
