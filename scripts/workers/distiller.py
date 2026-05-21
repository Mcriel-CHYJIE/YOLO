"""蒸馏工作线程"""
from scripts.tabs.base import ROOT, DATA_YAML
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime
import json, torch


class DistillationLossWrapper:
    """Wrapper class for distillation loss (picklable for multiprocessing)"""
    def __init__(self, orig_loss, teacher_model, alpha, device):
        self.orig_loss = orig_loss; self.teacher_model = teacher_model
        self.alpha = alpha; self.device = device
    def __call__(self, preds, batch):
        import torch.nn.functional as F
        det_loss, loss_items = self.orig_loss(batch, preds=preds)
        with torch.no_grad():
            to = self.teacher_model(batch['img'].to(self.device))
            t_feats = to[1]['feats'] if isinstance(to, tuple) else to.get('feats', [])
        s_feats = preds.get('feats', [])
        distill, n = 0.0, min(len(s_feats), len(t_feats))
        for i in range(n):
            sp = s_feats[i].view(s_feats[i].size(0), -1)
            tp = t_feats[i].view(t_feats[i].size(0), -1)
            md = min(sp.size(-1), tp.size(-1))
            distill += F.mse_loss(sp[:, :md], tp[:, :md].detach())
        return (1.0 - self.alpha) * det_loss.sum() + self.alpha * distill / max(n, 1), loss_items


class Distiller(QThread):
    log = pyqtSignal(str); progress = pyqtSignal(int, dict); done = pyqtSignal(bool, str)
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self._stop = False
    def stop(self): self._stop = True
    def run(self):
        try: self._train()
        except BaseException as e:
            import traceback; traceback.print_exc()
            self.log.emit(f' {e}'); self.done.emit(False, str(e))
    def _train(self):
        import torch.nn.functional as F, torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR
        import numpy as np, shutil, time
        from ultralytics import YOLO
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import colorstr
        from types import SimpleNamespace; from copy import deepcopy
        cfg = self.cfg; device = cfg['device']
        if torch.cuda.is_available() and 'cuda' in str(device):
            try:
                gi = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                al = torch.cuda.memory_allocated(0) / (1024**3)
                self.log.emit(f' GPU Memory: {gi:.1f}GB total, {gi-al:.1f}GB free')
                torch.cuda.empty_cache()
            except: pass
        self.log.emit(f'[Teacher] {cfg["teacher"]}')
        teacher = YOLO(cfg['teacher']); teacher.model.to(device); teacher.model.eval()
        for p in teacher.model.parameters(): p.requires_grad = False
        self.log.emit(f'[Student] {cfg["student"]}')
        student = YOLO(cfg['student'])
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils import IterableSimpleNamespace
        student.model.args = IterableSimpleNamespace(**(DEFAULT_CFG_DICT | student.model.args))
        for p in student.model.parameters(): p.requires_grad = True
        student.model.to(device); student.model.train()
        alpha = cfg['alpha']; orig_loss = student.model.loss
        student.model.loss = DistillationLossWrapper(orig_loss, teacher.model, alpha, device)
        self.log.emit(f'[Data] {cfg["data"]}')
        data_dict = check_det_dataset(cfg['data'])
        cfg_ns = SimpleNamespace(imgsz=cfg['imgsz'], batch=cfg['batch'], workers=0,
            lr0=cfg['lr0'], weight_decay=cfg.get('weight_decay', 0.0005), momentum=cfg.get('momentum', 0.937),
            rect=False, cache=None, single_cls=False, stride=32, pad=0.0, prefix=colorstr('train: '),
            task='detect', classes=None, fraction=1.0, augment=True, hyp=None, data=data_dict,
            mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode='flip', cutmix=0.0, degrees=0.0,
            translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
            bgr=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4, mask_ratio=4, overlap_mask=False)
        ds = build_yolo_dataset(cfg_ns, data_dict['train'], cfg['batch'], data_dict, mode='train', rect=False, stride=32)
        dl = build_dataloader(ds, batch=cfg['batch'], workers=0, shuffle=True)
        self.log.emit(f'[Data] {len(ds)} samples, {len(dl)} batches/epoch')
        wd = cfg.get('weight_decay', 0.0005); mb = cfg.get('momentum', 0.937)
        opt = optim.AdamW(student.model.parameters(), lr=cfg['lr0'], weight_decay=wd, betas=(mb, 0.999))
        we = cfg.get('warmup_epochs', 5)
        sch = CosineAnnealingLR(opt, T_max=cfg['epochs'] - we, eta_min=cfg['lr0'] * 0.01)
        sd = ROOT / 'runs' / 'distill' / cfg['name']; sd.mkdir(parents=True, exist_ok=True)
        bm50, be, pc, hist = 0.0, 0, 0, {'epoch':[],'loss':[],'map50':[],'map50_95':[]}
        total_batches = len(dl)
        self.log.emit('\n Starting distillation training...')
        for ep in range(1, cfg['epochs'] + 1):
            if self._stop: break
            if ep <= we:
                for pg in opt.param_groups: pg['lr'] = cfg['lr0'] * (ep / we)
            student.model.train(); el, nb = 0.0, 0
            ep_progress = f'{ep}/{cfg["epochs"]}'
            self.log.emit(f'\n[Epoch {ep_progress}] LR: {opt.param_groups[0]["lr"]:.6f} | Batches: {total_batches}')
            bt = time.time()
            for bi, batch in enumerate(dl):
                if self._stop: break
                try:
                    img = batch['img'].to(device, non_blocking=True)
                    img = img.float() / 255.0 if img.dtype == torch.uint8 else img; batch['img'] = img
                    for k in ['cls','bboxes','batch_idx']:
                        if k in batch and isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device, non_blocking=True)
                    loss, _ = student.model.loss(student.model(img), batch)
                    opt.zero_grad(); loss.backward(); opt.step()
                    el += loss.item(); nb += 1
                    bi2 = bi + 1
                    if bi2 % max(total_batches // 10, 1) == 0 or bi2 == total_batches:
                        et = (time.time() - bt) / bi2 * (total_batches - bi2)
                        self.log.emit(f'  [{bi2}/{total_batches}] | Loss: {loss.item():.4f} | ETA: {et:.0f}s')
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e): torch.cuda.empty_cache(); continue
                    raise
            sch.step(); avg = el / max(nb, 1); vm50, vm95 = 0.0, 0.0
            try:
                vm = YOLO(cfg['student'])
                vm.model.load_state_dict(student.model.state_dict()); vm.model.to(device); vm.model.eval()
                r = vm.val(data=cfg['data'], device=device, batch=min(cfg['batch'], 16), imgsz=cfg['imgsz'],
                          plots=False, verbose=False, save=False, rect=True)
                vm50, vm95 = float(r.box.map50), float(r.box.map)
                self.log.emit(f'  [Val] mAP50: {vm50:.4f}, mAP50-95: {vm95:.4f}')
            except Exception as ve:
                self.log.emit(f'[Val Error] {ve}')
            if vm50 > bm50:
                bm50, be, pc = vm50, ep, 0
                torch.save({'model': student.model.state_dict(), 'names': student.names,
                    'args': student.model.args, 'version': '8.0.0'}, str(sd / 'best5.20.pt'))
                self.log.emit(f'   New best mAP50: {vm50:.4f} (epoch {ep})')
            else: pc += 1
            if ep % 10 == 0:
                torch.save({'model': student.model.state_dict(), 'names': student.names,
                    'args': student.model.args, 'version': '8.0.0'}, str(sd / f'epoch_{ep}.pt'))
            hist['epoch'].append(ep); hist['loss'].append(avg); hist['map50'].append(vm50); hist['map50_95'].append(vm95)
            self.log.emit(f'Epoch {ep}/{cfg["epochs"]} | Loss: {avg:.4f} | mAP50: {vm50:.4f}'
                         + (f' | Best: {bm50:.4f}@{be}' if bm50 > 0 else ''))
            self.progress.emit(ep, {'epoch': ep, 'total': cfg['epochs'], 'loss': avg, 'map50': vm50,
                'best_map50': bm50, 'best_epoch': be, 'lr': opt.param_groups[0]['lr'], 'history': dict(hist)})
            if pc >= cfg['patience'] and ep > we + 20: break
        torch.save({'model': student.model.state_dict(), 'names': student.names,
            'args': student.model.args, 'version': '8.0.0'}, str(sd / 'last.pt'))
        with open(str(sd / 'history.json'), 'w') as f: json.dump(hist, f, indent=2)
        self.log.emit(f'Done! Best mAP50: {bm50:.4f}')
        self.done.emit(True, str(sd))
