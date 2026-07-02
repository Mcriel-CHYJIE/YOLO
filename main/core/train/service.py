"""训练业务逻辑 — 配置构建 + 训练工作线程"""
# ⚠️ 必须在 import torch 之前设置，防止 CUDA 碎片化 OOM
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
from main.core.base import ROOT, cfg, DATA_YAML, StreamEmitter
from PyQt5.QtCore import QObject, pyqtSignal
from datetime import datetime
import json, torch, threading
from ultralytics import YOLO
from main.config import load_paths


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
    cls_loss_val='bce', focal_gamma_val=2.0, focal_alpha_val=0.75,
    asl_gamma_pos_val=0.0, asl_gamma_neg_val=4.0, iou_loss_val='ciou',
    fusion_type_val='none',
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
        # Resolve .yaml path against models_dir (same as .pt resolution)
        _md = load_paths().get('models_dir', str(ROOT / 'models'))
        local = Path(_md) / model_name
        model_path = str(local) if local.exists() else model_name
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
        # custom loss config
        cls_loss=cls_loss_val,
        focal_gamma=focal_gamma_val, focal_alpha=focal_alpha_val,
        asl_gamma_pos=asl_gamma_pos_val, asl_gamma_neg=asl_gamma_neg_val,
        iou_loss=iou_loss_val,
        fusion=fusion_type_val,
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
        self.history = {'epoch':[],'train_loss':[],'mAP50':[],'mAP50_95':[],'precision':[],'recall':[],'lr':[]}
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
            # 使用 mem_get_info 获取驱动级真实空闲（含其他进程占用）
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_mem = free_bytes / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            return {'total': total_mem, 'allocated': allocated_mem, 'free': free_mem,
                    'used_by_all': (total_bytes - free_bytes) / (1024**3)}
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

    def _generate_partial_charts(self, cfg):
        """OOM 后在 CPU 上生成缺失的验证图表(confusion matrix, PR curves, val_batch)"""
        try:
            if not hasattr(self, '_exp_dir') or not self._exp_dir:
                return
            best = self._exp_dir / 'weights' / 'best.pt'
            if not best.exists():
                best = self._exp_dir / 'weights' / 'last.pt'
            if not best.exists():
                self.log.emit(' [Chart] No weights found, skipping charts')
                return
            torch.cuda.empty_cache()
            self.log.emit(f' [Chart] Loading {best.name} on CPU...')
            m = YOLO(str(best))
            self.log.emit(f' [Chart] Generating validation plots (CPU, batch=1)...')
            m.val(
                data=str(DATA_YAML),
                imgsz=cfg['imgsz'],
                batch=1,
                device='cpu',
                plots=True,
                project=str(self._exp_dir),
                name='.',
                exist_ok=True,
            )
            torch.cuda.empty_cache()
            # 列出生成的文件
            for f in self._exp_dir.glob('*.png'):
                self.log.emit(f' [Chart]   {f.name}')
            self.log.emit(f' [Chart] Done')
        except Exception as e:
            self.log.emit(f' [Chart] Failed (non-critical): {e}')
            torch.cuda.empty_cache()

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
            
            # ── 注入移动到 on_train_start 回调（在 trainer 创建好模型后执行）──
            cfg_for_cb = {
                'attention': cfg.get('attention', 'none'),
                'lora_rank': cfg.get('lora_rank', 0),
                'fusion': cfg.get('fusion', 'none'),
                'cls_loss': cfg.get('cls_loss', 'bce'),
                'focal_gamma': cfg.get('focal_gamma', 2.0),
                'focal_alpha': cfg.get('focal_alpha', 0.75),
                'asl_gamma_pos': cfg.get('asl_gamma_pos', 0.0),
                'asl_gamma_neg': cfg.get('asl_gamma_neg', 4.0),
                'iou_loss': cfg.get('iou_loss', 'ciou'),
            }
            log_emit = self.log.emit

            def _on_train_start(trainer):
                """Inject modules after trainer creates the model."""
                model = trainer.model
                # Save trainer ref for loss restore later
                self._ultra_trainer = trainer
                # ── Attention ──
                attn = cfg_for_cb.get('attention', 'none')
                if attn and attn.lower() != 'none':
                    try:
                        from main.core.train.attention import inject_attention
                        replaced = inject_attention(model, attn)
                        log_emit(f' [{attn.upper()}] Injected attention into {replaced} C2f blocks')
                    except Exception as e:
                        log_emit(f' [WARN] Attention injection failed: {e}')
                # ── LoRA ──
                lora_r = cfg_for_cb.get('lora_rank', 0)
                if lora_r > 0:
                    try:
                        from main.core.train.attention import inject_lora
                        replaced = inject_lora(model, lora_r)
                        log_emit(f' [LoRA] Injected LoRA (rank={lora_r}) into {replaced} Conv2d layers')
                        total = sum(p.numel() for p in model.parameters())
                        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        log_emit(f'  Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total')
                    except Exception as e:
                        log_emit(f' [WARN] LoRA injection failed: {e}')
                # ── Multi-scale fusion ──
                fusion_type = cfg_for_cb.get('fusion', 'none')
                if fusion_type and fusion_type != 'none':
                    try:
                        from main.core.train.neck import inject_multiscale_fusion
                        result = inject_multiscale_fusion(model, fusion_type)
                        if result:
                            log_emit(f' [Neck] Multi-scale fusion: {result}')
                    except Exception as e:
                        log_emit(f' [WARN] Fusion injection failed: {e}')
                # ── Rebuild EMA after any structural change ──
                # Attention (AttentionWrapper replaces C3k2) and fusion
                # (inserted into Sequential) both change state_dict keys.
                # The EMA was created before these injections and its keys
                # no longer match the model's state_dict.
                if hasattr(trainer, 'ema'):
                    from ultralytics.utils.torch_utils import ModelEMA
                    trainer.ema = ModelEMA(trainer.model)
                    log_emit(f' [EMA] Rebuilt after model structure change')
                # ── Loss function patching ──
                try:
                    from main.core.train.loss import patch_yolo_loss, restore_original_loss
                    loss_cfg_dict = {k: cfg_for_cb[k] for k in
                        ['cls_loss','focal_gamma','focal_alpha','asl_gamma_pos','asl_gamma_neg','iou_loss']}
                    patched = patch_yolo_loss(model, loss_cfg_dict)
                    if patched:
                        log_emit(f' [Loss] Custom loss: ' + ', '.join(patched))
                    trainer._restore_loss = restore_original_loss
                except Exception as e:
                    log_emit(f' [WARN] Loss patch failed: ' + str(e))

            m.add_callback('on_train_start', _on_train_start)

            # 输出目录命名：{日期}_{预设} 或 {日期}_{模型}
            preset = cfg.get('preset_name', '')
            if not preset:
                m_path = cfg.get('model', '')
                preset = Path(m_path).stem if m_path else ROOT.name
            exp = f'{datetime.now().strftime("%m%d_%H%M")}_{preset}'
            exp_dir = Path(self._train_dir) / exp
            weights_dir = exp_dir / 'weights'
            self._exp_dir = exp_dir  # 保存以便 OOM 时告知路径
            self.log.emit(f' ────────────────────────────────────────────')
            self.log.emit(f' Output: {exp_dir}')
            self.log.emit(f' Weights: {weights_dir / "best.pt"}')
            self.log.emit(f' ────────────────────────────────────────────')
            self.log.emit(f' {cfg["model"] if cfg["model"] else "Scratch"} | {cfg["epochs"]}ep | batch={cfg["batch"]}')
            if cfg.get('lr0', 0) <= 0: cfg['lr0'] = 0.001; self.log.emit(f'  LR0 was 0, reset to 0.001')
            if torch.cuda.is_available() and cfg['device'] != 'cpu':
                gi = self._check_gpu_memory()
                if gi:
                    self.log.emit(f' GPU Memory: {gi["total"]:.1f}GB total, {gi["free"]:.1f}GB free (其他进程占用 {gi.get("used_by_all", 0):.1f}GB)')
                    if gi['free'] < 10.0:
                        self.log.emit(f' ⚠️  空闲显存不足 ({gi["free"]:.1f}GB)，训练可能 OOM')
                        self.log.emit(f'  请关闭占用 GPU 的其他程序（浏览器/游戏/串流等）后重试')
                    elif gi['free'] < 12.0:
                        self.log.emit(f' ⚠️  空闲显存偏低 ({gi["free"]:.1f}GB)，建议关闭其他 GPU 程序')
            # ── 自动检测 batch 是否可能爆显存 ──
            if torch.cuda.is_available() and cfg['device'] != 'cpu':
                try:
                    model_name = str(cfg.get('model', '')).lower()
                    bs = cfg['batch']
                    # 额外模块等级: 0=none, 1=attention/LoRA, 2=+fusion, 3=+P2+fusion
                    extra_level = 0
                    if cfg.get('lora_rank', 0) > 0 or cfg.get('attention', 'None') != 'None':
                        extra_level = 1
                    if cfg.get('fusion', 'none') != 'none':
                        extra_level = 2
                    if '-p2' in model_name or model_name.startswith('yolo') and '-p2' in model_name:
                        extra_level = 3
                    imgsz = cfg.get('imgsz', 640)
                    is_hd = imgsz >= 720
                    # Per-model baseline batch (RTX 5070 Ti 16GB, no extras)
                    # Scale: base → +attention ×0.95 → +fusion ×0.85 → +P2 ×0.55
                    hd_scale = 0.42  # 960 versus 640: ×(640/960)² ≈ 0.44, rounded for safety
                    if 'yolo11n' in model_name:
                        base = 36
                    elif 'yolov8n' in model_name:
                        base = 40
                    elif 'yolo11s' in model_name:
                        base = 28
                    elif 'yolov8s' in model_name:
                        base = 24
                    elif 'yolo11m' in model_name or 'yolov8m' in model_name:
                        base = 16
                    elif 'yolo11x' in model_name or 'yolov8x' in model_name:
                        base = 8
                    else:
                        base = 24
                    multipliers = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.55}
                    mult = multipliers.get(extra_level, 1.0)
                    safe_max = int(base * mult) if not is_hd else max(int(base * mult * hd_scale), 2)
                    extra_label = {0:'标准',1:'+注意力/LoRA',2:'+注意力+融合',3:'+P2+注意力+融合'}
                    if bs > safe_max:
                        self.log.emit(f' ⚠️  batch={bs} 在当前配置下可能爆显存')
                        self.log.emit(f'  识别: {extra_label.get(extra_level, "自定义")}')
                        self.log.emit(f'  建议: batch ≤ {safe_max}')
                        self.log.emit(f'  仍将使用 batch={bs} 启动，OOM 时请降低 batch')
                except Exception as _e:
                    pass
            def cb(t):
                if self._stop_event.is_set(): t.stop = True; return
                try:
                    ep = t.epoch; loss = 0.0
                    if hasattr(t,'loss') and t.loss is not None: loss = float(t.loss)
                    if loss == 0.0 and hasattr(t,'tloss') and t.tloss is not None: loss = float(t.tloss)
                    mt = getattr(t,'metrics',None) or {}
                    m50 = float(mt.get('metrics/mAP50(B)',0)); m95 = 0.0
                    for k in ['metrics/mAP50-95(B)','metrics/mAP50_95(B)','metrics/mAP50_95','metrics/mAP_0.5:0.95']:
                        v = mt.get(k)
                        if v is not None:
                            try: m95 = float(v); break
                            except: pass
                    p = float(mt.get('metrics/precision(B)',0)); r = float(mt.get('metrics/recall(B)',0))
                    lr_val = float(t.lr[0]) if hasattr(t, 'lr') and isinstance(getattr(t, 'lr', None), (list, tuple)) and len(t.lr) > 0 else (
                        float(t.optimizer.param_groups[0]['lr']) if hasattr(t, 'optimizer') and t.optimizer else 0.0
                    )
                    self.history['epoch'].append(ep); self.history['train_loss'].append(loss)
                    self.history['mAP50'].append(m50); self.history['mAP50_95'].append(m95)
                    self.history['precision'].append(p); self.history['recall'].append(r)
                    self.history['lr'].append(lr_val)
                    if m50 > self.best_map: self.best_map = m50
                    self.status.emit(f'Epoch {ep}/{cfg["epochs"]}', min(ep/max(cfg['epochs'],1),1.0), self.best_map, m50, loss)
                    if torch.cuda.is_available():
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        self.log.emit(f'  [Epoch {ep}] GPU peak alloc={peak:.1f}G reserved={reserved:.1f}G')
                        torch.cuda.reset_peak_memory_stats()
                    self.chart.emit()
                except: pass
            m.add_callback('on_train_start', lambda t: self.log.emit(' Training started (first epoch may take a while to initialize)'))
            # ── 每个 batch 处理完后发射进度（每 10% 报一次，含真实显存） ──
            _last_batch_pct = [0]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            def on_batch(t):
                try:
                    ni = t.ni if hasattr(t, 'ni') else 0
                    nf = t.epoch_len if hasattr(t, 'epoch_len') else 1
                    pct = int(ni / max(nf, 1) * 100)
                    if pct >= _last_batch_pct[0] + 10:
                        _last_batch_pct[0] = pct
                        mem_line = f'  Epoch progress: {pct}% ({ni}/{nf} batches)'
                        if torch.cuda.is_available():
                            alloc = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                            peak = torch.cuda.max_memory_allocated() / 1e9
                            mem_line += f' | GPU alloc={alloc:.1f}G reserved={reserved:.1f}G peak={peak:.1f}G'
                        self.log.emit(mem_line)
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
                    self.log.emit(f'')
                    self.log.emit(f' ╔══════════════════════════════════════════╗')
                    self.log.emit(f' ║  CUDA Out of Memory                     ║')
                    self.log.emit(f' ╚══════════════════════════════════════════╝')
                    self.log.emit(f'  已完成 {len(self.history["epoch"])} 个 epoch')
                    if torch.cuda.is_available():
                        peak = torch.cuda.max_memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        self.log.emit(f'  OOM 时峰值: alloc={peak:.1f}G reserved={reserved:.1f}G')
                    if hasattr(self, '_exp_dir') and self._exp_dir:
                        wd = self._exp_dir / 'weights'
                        self.log.emit(f'  部分权重已保存到:')
                        self.log.emit(f'    {wd / "last.pt"}')
                        self.log.emit(f'    {wd / "best.pt"}')
                        self.log.emit(f'  可将 last.pt 作为预训练恢复训练')
                    self.log.emit(f'  降低 batch 或 imgsz 后重试')
                    self._generate_partial_charts(cfg)
                    raise
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
        # Restore original loss function
        restore_src = getattr(self, '_ultra_trainer', None)
        if restore_src and hasattr(restore_src, '_restore_loss'):
            try:
                restore_src._restore_loss()
            except Exception:
                pass
